"""
AI picking stage (ports the *windowed* loop of `02.ML_picking_for_all_events`).

SeisBench PhaseNet ('stead') runs on each event-station 3-component record (raw +
bandpass 1-40 Hz, the picker's own normalisation). Picks are restricted to a
phase-velocity window around the origin, then the best P+S pair is chosen
(max probability sum, S after P, S-P < sp_max_gap) with a single-phase fallback.

Outputs, per event:
  - picks/<event_id>_picks.csv  columns: Event_ID, Network, Station, Phase, Time, Probability
  - sac.a (P) / sac.t0 (S) written into the run's waveforms_100km/*.sac (all 3 comps)

PhaseNet inference is run on CPU to match the baseline (the notebook fixes
device=cpu); GPU can introduce sub-sample pick drift.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd
from obspy import read, Stream, UTCDateTime

from pipeline import config


def _get_time(p):  return p.peak_time if hasattr(p, "peak_time") else p.start_time
def _get_prob(p):  return p.peak_value if hasattr(p, "peak_value") else p.probability


def load_model(weights="stead", device="cpu"):
    import torch
    import seisbench.models as sbm
    model = sbm.PhaseNet.from_pretrained(weights)
    model.to(torch.device(device))
    return model


# ------------------------------------------------------------- per-event pick
def pick_event(cfg, model, used_table, ev) -> list[dict]:
    pw = cfg.pick_window
    evdp, vp, vs = pw["evdp"], pw["vp"], pw["vs"]
    win_p_off, win_s_off = pw.get("p_off", -1.0), pw.get("s_off", 4.0)
    bp = cfg.pick_bandpass
    eid = ev["event_id"]
    wf = config.event_wf_dir(cfg, eid)
    results = []

    for r in used_table.itertuples():
        net, sta, chan = r.Network, r.Code, r.Sensor
        fs = [f"{wf}/{eid}.{net}.{sta}.{chan}{c}.sac" for c in ("Z", "N", "E")]
        if not all(os.path.exists(f) for f in fs):
            continue
        try:
            st = Stream()
            for f in fs:
                st += read(f)
            tr_z = st.select(component="Z")[0]
            sac = tr_z.stats.sac
            if not hasattr(sac, "dist") or sac.dist == -12345.0:
                continue

            hypo_dist = np.hypot(sac.dist, evdp)
            origin = tr_z.stats.starttime - sac.b
            win_start = origin + hypo_dist / vp + win_p_off
            win_end = origin + hypo_dist / vs + win_s_off

            t0, t1 = max(t.stats.starttime for t in st), min(t.stats.endtime for t in st)
            st.trim(t0, t1)
            stf = (st.copy().detrend("demean")
                     .filter("bandpass", freqmin=bp["freqmin"], freqmax=bp["freqmax"],
                             corners=bp["corners"], zerophase=bp["zerophase"])
                     .detrend("demean"))
            out = model.classify(stf, P_threshold=cfg.p_threshold, S_threshold=cfg.s_threshold)

            p_picks = sorted([p for p in out.picks if p.phase == "P"
                              and win_start <= _get_time(p) <= win_end], key=_get_time)
            s_picks = sorted([s for s in out.picks if s.phase == "S"
                              and win_start <= _get_time(s) <= win_end], key=_get_time)
            pairs = [(p, s) for p in p_picks for s in s_picks
                     if 0 < _get_time(s) - _get_time(p) < cfg.sp_max_gap_s]
            best_p, best_s = (max(pairs, key=lambda x: _get_prob(x[0]) + _get_prob(x[1]))
                              if pairs else (None, None))
            if not best_p and p_picks:
                best_p = max(p_picks, key=_get_prob)
            if not best_s and s_picks:
                best_s = (max([s for s in s_picks if _get_time(s) > _get_time(best_p)],
                              key=_get_prob) if best_p else max(s_picks, key=_get_prob))

            for pk in (best_p, best_s):
                if pk is not None:
                    results.append(dict(Event_ID=eid, Network=net, Station=sta,
                                        Phase=pk.phase, Time=_get_time(pk),
                                        Probability=round(_get_prob(pk), 3)))
        except Exception as e:  # noqa: BLE001  (mirror the notebook's permissive skip)
            print(f"[pick] {eid} {sta}: {e}")
    return results


# ------------------------------------------------- write picks into SAC headers
def write_sac_picks(cfg, used_table, picks_df):
    """Set sac.a (P) / sac.t0 (S) on all 3 components for each pick."""
    sensor_of = dict(zip(used_table["Code"], used_table["Sensor"]))
    for eid, grp in picks_df.groupby("Event_ID"):
        wf = config.event_wf_dir(cfg, eid)
        for _, row in grp.iterrows():
            net, sta = row["Network"], row["Station"]
            chan = sensor_of.get(sta)
            pick_time = UTCDateTime(row["Time"])
            for c in ("Z", "N", "E"):
                f = f"{wf}/{eid}.{net}.{sta}.{chan}{c}.sac"
                if not os.path.exists(f):
                    continue
                tr = read(f)[0]
                dt = pick_time - (tr.stats.starttime - tr.stats.sac.b)
                if row["Phase"] == "P":
                    tr.stats.sac.a = dt
                else:
                    tr.stats.sac.t0 = dt
                tr.write(f, format="SAC")


# --------------------------------------------------------------- orchestration
def run_picking(cfg, events=None, device="cpu", model=None, used_table=None,
                write=True, write_sac=True) -> dict:
    from pipeline.core import waveforms
    if used_table is None:
        used_table = pd.read_csv(config.used_stations_csv(cfg))
    if model is None:
        model = load_model(cfg.picker_weights, device)
    catalog = waveforms.load_catalog(cfg)
    if events is not None:
        catalog = [e for e in catalog if e["event_id"] in set(events)]

    out_dir = config.assert_writable(config.picks_dir(cfg))
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for ev in catalog:
        picks = pick_event(cfg, model, used_table, ev)
        counts[ev["event_id"]] = len(picks)
        if picks:
            df = pd.DataFrame(picks)
            if write:
                df.to_csv(config.picks_csv(cfg, ev["event_id"]), index=False)
            if write and write_sac:
                write_sac_picks(cfg, used_table, df)
    return counts
