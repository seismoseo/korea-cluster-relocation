"""
Stage — cross-correlation differential times `dt.cc` for HypoDD (ports
`2.HypoDD/02.dt.cc/01.Detailed_delay_time_measurement_from_Xcorr_revised.ipynb`
+ `02.Get_Pairs_Greater_Than_CC_threshold.ipynb`).

For every event pair and shared station it measures a sub-sample differential time by
cross-correlating the re-referenced waveforms (run `core.rereference` first):

  * P on the Z component, S as the higher-CC of N / E;
  * each trace is interpolated to `interp_hz` (lanczos a=20), demeaned/tapered, bandpass
    filtered, demeaned/tapered again, then sliced `±(pre, post)` s around the SAC pick
    (`a` for P, `t0` for S) and normalised;
  * the second event's window is slid over `±margin` s (step `slide_step`); the lag with
    the highest `obspy.signal.cross_correlation.correlate`/`xcorr_max` coefficient wins;
  * differential time `diff = (t1 + shift - ot1) - (t2 - ot2)` where `t*` are absolute
    picks and `ot*` are the (re-referenced) origins `starttime - b`.

Outputs under `runs/<cluster>/2.HypoDD/02.dt.cc/`: per-pair `dt.cc_{P,S}/dt.cc_{P,S}_<e1>_<e2>`,
the cc>=threshold `dt.cc_P_0.7`/`dt.cc_S_0.7`, their concatenation `dt.cc_0.7_combined`, and
`dt.cc_0.7_combined_no_main` (every pair-block touching the mainshock cuspid dropped).

PERFORMANCE / POLITENESS (shared 64-core box): work is parallelised over pairs with a
`ProcessPoolExecutor` capped at `min(cfg.num_cores, |sched_getaffinity|)` — launch under
`taskset -c <cpulist>` to scope it. Two faithful speedups vs the notebook (identical
numbers): interpolated+filtered full traces are cached per worker, and the slide loop
slices a data *view* instead of deep-copying the full trace 1000x. `slide_step` (default
0.001 s, the baseline grid) is the speed/precision knob.

Cuspid <-> event-dir mapping is by `.sum` `ID-NUM % cuspid_offset` (the scheme
`hypoinverse.write_phs` stamps), so a dropped/unlocated event cannot shift the others.
"""
from __future__ import annotations

import os
from glob import glob
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max

from pipeline import config
from pipeline.core import sumio

UNDEF = -12345.0
_PICK = {"P": "a", "S": "t0"}

# ----------------------------------------------------------- worker-global state
_COMMON = _STATIONS = _EID = _XC = _OVR = _OUTP = _OUTS = None
_CACHE: dict = {}


def _init_worker(common, stations, eid, xc, ovr, outp, outs):
    global _COMMON, _STATIONS, _EID, _XC, _OVR, _OUTP, _OUTS, _CACHE
    _COMMON, _STATIONS, _EID, _XC, _OVR, _OUTP, _OUTS = \
        common, stations, eid, xc, ovr, outp, outs
    _CACHE = {}


# ------------------------------------------------------------- window selection
def _full_trace(eid, station, comp, fmin, fmax):
    """Interpolated + filtered full trace for (event, station, comp, band), cached."""
    key = (eid, station, comp, fmin, fmax)
    tr = _CACHE.get(key)
    if tr is None:
        fs = glob(os.path.join(_COMMON, eid, f"{eid}.*.{station}.*{comp}.sac"))
        if not fs:
            raise FileNotFoundError(f"{eid}/{station}/{comp}")
        tr = (read(fs[0])[0]
              .interpolate(sampling_rate=_XC["interp_hz"], method="lanczos", a=20)
              .detrend("demean").taper(0.05)
              .filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
              .detrend("demean").taper(0.05))
        if not tr.stats.network:                # canonical name is {eid}.{net}.{code}.{chan}.sac
            tr.stats.network = os.path.basename(fs[0]).split(".")[1]
        _CACHE[key] = tr
    return tr


def _pick_time(tr, hdr):
    v = tr.stats.sac.get(hdr, UNDEF)
    if v == UNDEF:
        raise ValueError(f"no {hdr} pick")
    return tr.stats.starttime + v - tr.stats.sac.b


def _measure(tr1, tr2, hdr, pre, post, shift_samp, margin, step):
    """Slide tr2's window over +/-margin; return (shift_s, coeff) at best CC.

    Mirrors cc_measurement_revised_*: tr1 sliced+normalised once; tr2 windowed per slide
    (a data view, not a copy); shift = shift_samp/interp_hz - slide of the best CC."""
    p1 = _pick_time(tr1, hdr)
    tr1_slice = tr1.slice(p1 - pre, p1 + post).copy().normalize()
    arr2 = _pick_time(tr2, hdr)
    best_cc, best_shift, best_slide = -2.0, 0.0, 0.0
    for slide in np.arange(-margin, margin, step):
        tr2_slice = tr2.slice(arr2 - pre + slide, arr2 + post + slide)   # view, read-only
        shift, coeff = xcorr_max(correlate(tr1_slice, tr2_slice, shift_samp), abs_max=False)
        if coeff > best_cc:
            best_cc, best_shift, best_slide = coeff, shift, slide
    return np.round(best_shift / _XC["interp_hz"] - best_slide, 3), best_cc


def _window(pair):
    """(pre, post, fmin, fmax) for a pair, applying any xcorr_pair_override."""
    pre, post = _XC["pre"], _XC["post"]
    fmin, fmax = _XC["bandpass"]
    s = set(pair)
    for key, ov in _OVR.items():
        if s & set(key):
            pre, post = ov.get("pre", pre), ov.get("post", post)
            fmin, fmax = ov.get("bandpass", (fmin, fmax))
            break
    return pre, post, fmin, fmax


def _fmt(net, station, diff, coeff, phase):
    return (f"{net}{station}".ljust(6) + " " * 5 + str(np.round(diff, 5)).ljust(9)
            + " " * 3 + str(np.round(coeff, 5)).ljust(11) + " " * 4 + phase + "\n")


def _header(e1, e2):
    return f"#    {_EID[e1]}      {_EID[e2]}       0.0\n"


# ------------------------------------------------------------------ pair workers
def _pair_P(pair):
    e1, e2 = pair
    pre, post, fmin, fmax = _window(pair)
    shift_samp, margin, step = _XC["shift_samp"], _XC["margin"], _XC["slide_step"]
    lines = [_header(e1, e2)]
    for sta in _STATIONS:
        try:
            tr1 = _full_trace(e1, sta, "Z", fmin, fmax)
            tr2 = _full_trace(e2, sta, "Z", fmin, fmax)
            shift, coeff = _measure(tr1, tr2, "a", pre, post, shift_samp, margin, step)
            t1, ot1 = _pick_time(tr1, "a"), tr1.stats.starttime - tr1.stats.sac.b
            t2, ot2 = _pick_time(tr2, "a"), tr2.stats.starttime - tr2.stats.sac.b
            diff = (t1 + shift - ot1) - (t2 - ot2)
            net = tr1.stats.network or sta[:0]
            lines.append(_fmt(net, sta, diff, coeff, "P"))
        except Exception:                                   # noqa: BLE001 (notebook skip)
            pass
    with open(os.path.join(_OUTP, f"dt.cc_P_{e1}_{e2}"), "w") as f:
        f.writelines(lines)


def _pair_S(pair):
    e1, e2 = pair
    pre, post, fmin, fmax = _window(pair)
    shift_samp, margin, step = _XC["shift_samp"], _XC["margin"], _XC["slide_step"]
    lines = [_header(e1, e2)]
    for sta in _STATIONS:
        try:
            best = None                                     # (coeff, shift, tr1, tr2)
            for comp in _XC["s_comps"]:
                tr1 = _full_trace(e1, sta, comp, fmin, fmax)
                tr2 = _full_trace(e2, sta, comp, fmin, fmax)
                shift, coeff = _measure(tr1, tr2, "t0", pre, post, shift_samp, margin, step)
                if best is None or coeff >= best[0]:
                    best = (coeff, shift, tr1, tr2)
            coeff, shift, tr1, tr2 = best
            t1, ot1 = _pick_time(tr1, "t0"), tr1.stats.starttime - tr1.stats.sac.b
            t2, ot2 = _pick_time(tr2, "t0"), tr2.stats.starttime - tr2.stats.sac.b
            diff = (t1 + shift - ot1) - (t2 - ot2)
            net = tr1.stats.network or sta[:0]
            lines.append(_fmt(net, sta, diff, coeff, "S"))
        except Exception:                                   # noqa: BLE001
            pass
    with open(os.path.join(_OUTS, f"dt.cc_S_{e1}_{e2}"), "w") as f:
        f.writelines(lines)


# ----------------------------------------------------------- threshold + combine
def _filter_combine(pair_dir, pairs, phase, threshold, out_file):
    """Concatenate per-pair files (pair order); keep headers + cc>=threshold data lines."""
    with open(out_file, "w") as o:
        for e1, e2 in pairs:
            p = os.path.join(pair_dir, f"dt.cc_{phase}_{e1}_{e2}")
            if not os.path.exists(p):
                continue
            for line in open(p):
                if line.startswith("#"):
                    o.write(line)
                else:
                    try:
                        cc = float(line[23:34].replace(" ", ""))
                        if threshold <= cc <= 1.0:
                            o.write(line)
                    except ValueError:
                        pass


def _drop_mainshock(in_file, out_file, cuspid):
    """Drop every pair-block whose header references the mainshock cuspid."""
    cs = str(cuspid)
    with open(in_file) as fi, open(out_file, "w") as fo:
        skip = False
        for line in fi:
            if line.startswith("#"):
                skip = cs in line.split()[1:3]
                if not skip:
                    fo.write(line)
            elif not skip:
                fo.write(line)


# --------------------------------------------------------------- orchestration
def run_xcorr(cfg, velmodel="kim1983", cores=None) -> dict:
    """Measure dt.cc for all event pairs and build the threshold/combined/no_main files.

    Returns {"pairs": n, "stations": n, "combined": path, "no_main": path|None}."""
    out = config.assert_writable(config.dtcc_dir(cfg))
    out_p, out_s = os.path.join(out, "dt.cc_P"), os.path.join(out, "dt.cc_S")
    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_s, exist_ok=True)
    common = config.waveforms_dir(cfg)

    # events (by cuspid) + stations + pairs
    sumdf = sumio.read_sum(config.sum_file(cfg, velmodel))
    dirs = sorted(glob(os.path.join(common, "20*")))
    events, eid = [], {}
    for r in sumdf.itertuples():
        idx = int(r.id) % cfg.cuspid_offset
        if idx < len(dirs):
            e = os.path.basename(dirs[idx])
            events.append(e)
            eid[e] = int(r.id)
    stations = sorted({os.path.basename(f).split(".")[2]
                       for e in events for f in glob(os.path.join(common, e, "*.sac"))})
    pairs = list(combinations(events, 2))

    xc = dict(interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5, margin=0.5,
              cc_threshold=0.7, p_comp="Z", s_comps=("N", "E"), shift_samp=500,
              slide_step=0.001)
    xc.update(cfg.xcorr)                       # cluster overrides (keeps defaults above)
    xc["bandpass"] = tuple(xc["bandpass"])
    xc["s_comps"] = tuple(xc["s_comps"])

    ncores = max(1, min(cores or cfg.num_cores, len(os.sched_getaffinity(0))))
    print(f"[xcorr] {len(pairs)} pairs x {len(stations)} stations, {ncores} workers "
          f"(slide_step={xc['slide_step']}s, band={xc['bandpass']}Hz)")
    with ProcessPoolExecutor(max_workers=ncores, initializer=_init_worker,
                             initargs=(common, stations, eid, xc,
                                       dict(cfg.xcorr_pair_overrides), out_p, out_s)) as ex:
        list(ex.map(_pair_P, pairs))
        list(ex.map(_pair_S, pairs))

    thr = xc["cc_threshold"]
    p07, s07 = os.path.join(out, "dt.cc_P_0.7"), os.path.join(out, "dt.cc_S_0.7")
    _filter_combine(out_p, pairs, "P", thr, p07)
    _filter_combine(out_s, pairs, "S", thr, s07)
    combined = os.path.join(out, "dt.cc_0.7_combined")
    with open(combined, "w") as o:
        for f in (p07, s07):
            o.write(open(f).read())

    no_main = None
    ms_cuspid = eid.get(cfg.mainshock_event_id) if cfg.mainshock_event_id else None
    if ms_cuspid is not None:
        no_main = os.path.join(out, "dt.cc_0.7_combined_no_main")
        _drop_mainshock(combined, no_main, ms_cuspid)
    print(f"[xcorr] wrote {combined}"
          + (f" + no_main (drop cuspid {ms_cuspid})" if no_main else ""))
    return dict(pairs=len(pairs), stations=len(stations),
                combined=combined, no_main=no_main)
