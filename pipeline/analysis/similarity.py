"""
Waveform-similarity QC (ports `7/8.Waveform_similarity_analysis_*.ipynb`).

At a fixed reference station, cross-correlate every event's P (Z) or S (N) phase window
against a template event (default the largest magnitude). High, stable NCC across the
sequence => a repeating source/path (a tight cluster, as expected for these dense swarms);
NCC drops or amplitude/time trends flag waveform changes. Returns a tidy DataFrame for the
QC notebook to plot NCC / amplitude vs magnitude and vs time.

This is a diagnostic (not a relocation input and not part of regression): the original
clusters Gwangyang/Kimcheon/Jangsung ran it; the Gyeongju pilot did not.
"""
from __future__ import annotations

import os
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max

from pipeline import config
from pipeline.core import sumio, waveforms

UNDEF = -12345.0


def _ref_station(cfg, comp="Z"):
    """The station appearing (with `comp`) in the most events — a robust similarity ref."""
    c = Counter()
    for ev in waveforms.load_catalog(cfg):
        for f in glob(os.path.join(config.event_wf_dir(cfg, ev["event_id"]), f"*{comp}.sac")):
            c[os.path.basename(f).split(".")[2]] += 1
    return c.most_common(1)[0][0] if c else None


# --------------------------------------------- per-cluster waveform-similarity (gather + CC matrix)
def _reloc_path(cfg):
    """dt.cc relocation if present (the headline product), else the dt.ct one."""
    cc = os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc")
    return cc if os.path.exists(cc) else os.path.join(config.dtct_dir(cfg), "hypoDD.reloc")


def cluster_events_by_cid(cfg, min_events=4, reloc_path=None):
    """Group the relocated events into hypoDD sub-clusters: `{cid: [event_id … ordered by time]}`.

    Each `hypoDD.reloc` row (which carries a `cid`) is matched to a catalog `event_id` by nearest
    origin time (|Δt| < 2 s). Sub-clusters with `< min_events` are dropped; result is ordered
    largest-first."""
    df = sumio.read_reloc(reloc_path or _reloc_path(cfg))
    if df.empty:
        return {}
    cat_times = [(e["event_id"], e["origin"]) for e in waveforms.load_catalog(cfg)]
    out = {}
    for cid, g in df.groupby("cid"):
        evs, seen = [], set()
        for _, r in g.sort_values("time").iterrows():
            best, best_dt = None, 2.0
            for eid, o in cat_times:
                dt = abs(float(o - r["time"]))
                if dt < best_dt:
                    best, best_dt = eid, dt
            if best and best not in seen:
                seen.add(best); evs.append(best)
        if len(evs) >= min_events:
            out[int(cid)] = evs
    return dict(sorted(out.items(), key=lambda kv: -len(kv[1])))


def nearest_common_station(cfg, event_ids, comp="Z", min_frac=1.0):
    """The station that recorded **all** (or ≥`min_frac`) of `event_ids`, **nearest** to the group
    centroid. Station coords from `used_stations_csv`, distance via `gps2dist_azimuth`. If none reaches
    `min_frac`, relax to the max achievable coverage and warn (a station gap never aborts)."""
    import warnings
    from obspy.geodetics.base import gps2dist_azimuth
    used = pd.read_csv(config.used_stations_csv(cfg))
    coord = {r.Code: (r.Latitude, r.Longitude) for r in used.itertuples()}
    cat = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
    evs = [e for e in event_ids if e in cat]
    if not evs:
        return None
    cov = Counter()
    for eid in evs:
        seen = set()
        for f in glob(os.path.join(config.event_wf_dir(cfg, eid), f"*{comp}.sac")):
            seen.add(os.path.basename(f).split(".")[2])
        for st in seen:
            cov[st] += 1
    n = len(evs)
    cand = [s for s, c in cov.items() if c >= min_frac * n and s in coord]
    if not cand and cov:
        mx = max(cov.values())
        cand = [s for s, c in cov.items() if c == mx and s in coord]
        warnings.warn(f"nearest_common_station: no station covers all {n} events; "
                      f"using max-coverage {mx}/{n}")
    if not cand:
        return None
    clat = float(np.mean([cat[e]["lat"] for e in evs]))
    clon = float(np.mean([cat[e]["lon"] for e in evs]))
    cand.sort(key=lambda s: gps2dist_azimuth(clat, clon, coord[s][0], coord[s][1])[0])
    return cand[0]


def _auto_post(cfg, event_ids, station, vp=6.0, vs=3.46, coda_min=5.0, coda_factor=2.0,
               post_min=6.0, post_max=60.0):
    """Auto window length **after P** (seconds), from the **hypocentral** distance of the cluster
    centroid to `station`: estimate `S-P = d_hypo*(1/vs - 1/vp)` (d_hypo = hypot(epicentral, median
    depth)), then `post = (S-P) + max(coda_min, coda_factor*(S-P))` so the window spans P, S and the
    coda — short for nearby clusters, long for distant/deep ones. Clamped to `[post_min, post_max]`;
    falls back to `post_max` if the geometry is unavailable."""
    from obspy.geodetics.base import gps2dist_azimuth
    try:
        coord = {r.Code: (r.Latitude, r.Longitude)
                 for r in pd.read_csv(config.used_stations_csv(cfg)).itertuples()}
        cat = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
        evs = [cat[e] for e in event_ids if e in cat]
        clat = float(np.mean([e["lat"] for e in evs]))
        clon = float(np.mean([e["lon"] for e in evs]))
        depth = float(np.median([e["depth"] for e in evs if e.get("depth") is not None]))
        epi = gps2dist_azimuth(clat, clon, *coord[station])[0] / 1000.0
        d_hypo = float(np.hypot(epi, max(depth, 0.0)))
        sp = d_hypo * (1.0 / vs - 1.0 / vp)
        return float(min(post_max, max(post_min, sp + max(coda_min, coda_factor * sp))))
    except Exception:                                       # noqa: BLE001
        return post_max


def _phase_window(cfg, eid, station, sensor, comp="Z", bandpass=(5, 20), pre=1.0, post=20.0):
    """The **full** waveform window (P + S + coda) for one event at one station, aligned on P (t=0).

    Reads the SAC, takes the P pick from header `a` (S from `t0`), bandpasses, slices
    `[P-pre, P+post]` — `post` long enough to span S and the coda — demeans + normalizes. Returns
    `(Trace, sp)` where `sp` is the S-P time in seconds (for the S annotation) or None; `(None, None)`
    if unusable."""
    fs = glob(os.path.join(config.event_wf_dir(cfg, eid), f"{eid}.*.{station}.{sensor}{comp}.sac"))
    if not fs:
        return None, None
    tr = read(fs[0])[0]
    a = tr.stats.sac.get("a", UNDEF)                     # P
    t0 = tr.stats.sac.get("t0", UNDEF)                   # S
    pk = a if comp == "Z" else t0
    if pk == UNDEF:
        return None, None
    ptime = tr.stats.starttime - tr.stats.sac.b + pk
    sl = tr.copy().detrend("demean").taper(0.05)
    if bandpass:
        sl = sl.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1], corners=4, zerophase=True)
    win = sl.slice(ptime - pre, ptime + post).detrend("demean")
    if len(win.data) < 4:
        return None, None
    sp = (t0 - a) if (a != UNDEF and t0 != UNDEF and comp == "Z") else None
    return win.normalize(), sp


def _cluster_windows(cfg, event_ids, station, comp="Z", bandpass=(5, 20), pre=1.0, post=20.0):
    """`(kept_event_ids, [Trace…], [sp…])` — the full P+S+coda windows for `event_ids` at `station`,
    in input order (drop events with no usable window). `sp` is the per-event S-P time (s) or None."""
    used = pd.read_csv(config.used_stations_csv(cfg))
    sensor = dict(zip(used["Code"], used["Sensor"])).get(station)
    kept, wins, sps = [], [], []
    for eid in event_ids:
        w, sp = _phase_window(cfg, eid, station, sensor, comp, bandpass, pre, post)
        if w is not None:
            kept.append(eid); wins.append(w); sps.append(sp)
    return kept, wins, sps


def cluster_cc_matrix(cfg, event_ids, station, comp="Z", bandpass=(5, 20),
                      pre=1.0, post=None, max_shift_s=0.5):
    """`(kept_event_ids, ncc[N,N])` — the waveform NCC matrix of `event_ids` at `station`, computed on
    the **full P+S+coda window** (`pre`/`post`, 5-20 Hz). `post=None` auto-sizes it from the
    hypocentral distance (`_auto_post`). Symmetric, diag=1, max-lag NCC over ±`max_shift_s`."""
    if post is None:
        post = _auto_post(cfg, event_ids, station)
    kept, wins, _ = _cluster_windows(cfg, event_ids, station, comp, bandpass, pre, post)
    n = len(kept)
    M = np.eye(n, dtype=float)
    if n >= 2:
        shift = int(max_shift_s * wins[0].stats.sampling_rate)
        for i in range(n):
            for j in range(i + 1, n):
                _, c = xcorr_max(correlate(wins[i], wins[j], shift), abs_max=False)
                M[i, j] = M[j, i] = float(c)
    return kept, M


def similarity(cfg, station=None, comp="Z", template_event=None,
               pre=0.5, post=3.5, highpass=1.0, rms_win=0.3, max_shift_s=0.5) -> pd.DataFrame:
    """NCC of each event's phase window vs a template at one station.

    Returns columns: event_id, time, mag, ncc, shift_s, p_rms (sorted by time). The chosen
    station + template event are stored on df.attrs."""
    used = pd.read_csv(config.used_stations_csv(cfg))
    sensor_of = dict(zip(used["Code"], used["Sensor"]))
    station = station or _ref_station(cfg, comp)
    cat = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
    hdr = "a" if comp == "Z" else "t0"

    def _win(eid):
        sensor = sensor_of.get(station)
        fs = glob(os.path.join(config.event_wf_dir(cfg, eid),
                               f"{eid}.*.{station}.{sensor}{comp}.sac"))
        if not fs:
            return None, None
        tr = read(fs[0])[0]
        pk = tr.stats.sac.get(hdr, UNDEF)
        if pk == UNDEF:
            return None, None
        ptime = tr.stats.starttime - tr.stats.sac.b + pk
        sl = tr.copy().detrend("demean").taper(0.05)
        if highpass:
            sl = sl.filter("highpass", freq=highpass, corners=6, zerophase=True)
        win = sl.slice(ptime - pre, ptime + post).detrend("demean").normalize()
        rms = float(np.sqrt(np.mean(
            tr.copy().detrend("demean").slice(ptime, ptime + rms_win).data ** 2)))
        return win, rms

    slices = {e: _win(e) for e in sorted(cat)}
    slices = {e: v for e, v in slices.items() if v[0] is not None}
    if not slices:
        return pd.DataFrame(columns=["event_id", "time", "mag", "ncc", "shift_s", "p_rms"])
    template_event = template_event or max(slices, key=lambda e: cat[e]["mag"])
    templ = slices[template_event][0]
    shift_samp = int(max_shift_s * templ.stats.sampling_rate)

    rows = []
    for e, (win, rms) in slices.items():
        lag, coeff = xcorr_max(correlate(templ, win, shift_samp), abs_max=False)
        rows.append(dict(event_id=e, time=cat[e]["origin"].datetime, mag=cat[e]["mag"],
                         ncc=round(float(coeff), 4),
                         shift_s=round(lag / templ.stats.sampling_rate, 4),
                         p_rms=rms))
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df.attrs.update(station=station, template=template_event, comp=comp)
    return df
