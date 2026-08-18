"""
End-to-end orchestration: chain the stage functions for one cluster.

Stages: stations -> waveforms -> picking -> hypoinverse -> ph2dt -> dtct
        -> rereference -> xcorr -> dtcc.
The default `through="dtct"` runs the catalog (dt.ct) chain unchanged. The dt.cc branch
(rereference -> xcorr -> dtcc) is appended *after* dtct: rereference rewrites the run's
waveforms_100km origins to the HYPOINVERSE solution (a prerequisite for xcorr), and does
NOT affect ph2dt/dtct (which read the .arc, not the SACs) — so the validated dt.ct path
is untouched unless you request --through dtcc.

Both the CLI (cli/run_pipeline.py) and the JupyterLab notebooks call run_cluster(),
so there is a single execution path.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from pipeline import config
from pipeline.core import (stations, waveforms, picking, hypoinverse, hypodd,
                           rereference, xcorr, focal_mechanism)

# `focal_mechanism` is an OPT-IN tail stage (needs a phasenet_plus picking run for polarity);
# it is appended last so the default through="dtct"/"dtcc" relocation chains never trigger it.
# `report` is the final OPT-IN tail stage: it compiles the standard beamer PDF run summary
# (pipeline.reporting) from whatever products already exist on disk — never fails the run.
STAGES = ["stations", "waveforms", "picking", "hypoinverse", "ph2dt", "dtct",
          "rereference", "xcorr", "dtcc", "focal_mechanism", "report"]


def _count_located_events(cfg, velmodel: str) -> int:
    """Count rows in the HYPOINVERSE .sum for `velmodel` (or 0 if absent). Used as the
    >=2-events gate for the relative-relocation chain (ph2dt + dtct + xcorr + dtcc)."""
    import os
    sum_path = config.sum_file(cfg, velmodel)
    if not os.path.exists(sum_path):
        return 0
    with open(sum_path) as f:
        # .sum has one header line and one row per located event.
        return sum(1 for ln in f if ln.strip() and not ln.lstrip().startswith("DATE"))


def run_cluster(cfg, stage_from="stations", through="dtct",
                velmodels=("kim1983", "kim2011"), arc_velmodel="kim1983",
                device="cpu", events=None, dtcc_variant="default", cores=None,
                fm_velmodel=None, xcorr_resume=False, verbose=True) -> dict:
    i0, i1 = STAGES.index(stage_from), STAGES.index(through)
    todo = set(STAGES[i0:i1 + 1])
    res = {}
    timings: dict[str, float] = {}                # stage name -> wall-clock seconds

    def log(msg):
        if verbose:
            print(f"[{cfg.name}] {msg}", flush=True)

    @contextmanager
    def _time(stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings[stage] = time.perf_counter() - t0

    # Up-front xcorr backend notice — printed BEFORE the long picking stage so the user sees
    # immediately whether the GPU path is active or will fall back to CPU (and why), instead of
    # discovering it deep into the run at the xcorr stage.
    if "xcorr" in todo:
        _xb = getattr(cfg, "xcorr_backend", "obspy")
        try:
            eff, msg = xcorr.probe_xcorr_backend(_xb)
            if eff.startswith("cctorch_gpu"):
                log(f"GPU xcorr: ACTIVE — {msg}")
            elif _xb.startswith("cctorch_gpu"):
                log(f"GPU xcorr: UNAVAILABLE in this env → CPU fallback ({msg})")
            else:
                log(f"xcorr backend: {msg}")
        except Exception as e:  # noqa: BLE001 — a probe failure must never block the run
            log(f"xcorr backend: could not probe ({type(e).__name__})")

    if "stations" in todo:
        with _time("stations"):
            used = stations.run_stations(cfg)
        res["stations"] = len(used)
        log(f"stations: {len(used)} used  ({timings['stations']:.1f}s)")
    if "waveforms" in todo:
        with _time("waveforms"):
            res["waveforms"] = waveforms.run_waveforms(cfg, events=events)
        n_wf = sum(res["waveforms"].values())
        log(f"waveforms: gathered {n_wf} files / "
            f"{len(res['waveforms'])} events  ({timings['waveforms']:.1f}s)")
        # Guard: 0 gathered means no source SAC for this backend — almost always the WRONG
        # --source (e.g. a pre-2020 / STP cluster re-run as the default NECIS, which reads
        # kma_waveforms/ instead of stp_download/SAC/). Refuse to proceed: otherwise the
        # downstream stages silently ride a PRIOR run's cached outputs and report a stale
        # "success". Fail clearly with the likely fix.
        if n_wf == 0:
            sub = "stp_download/SAC" if getattr(cfg, "wf_source", "") == "stp_sac" else "kma_waveforms"
            raise RuntimeError(
                f"0 waveforms gathered for '{cfg.name}' (wf_source={getattr(cfg, 'wf_source', '?')}): "
                f"no SAC found under {cfg.src_root}/{sub}.\n"
                f"  - Pre-2020 events are STP-served — re-run with `--source stp` (or `mixed`).\n"
                f"  - Or you used --skip-download with nothing downloaded — drop it to fetch.\n"
                f"Refusing to continue on possibly-stale cached outputs in runs/{cfg.name}/.")
    if "picking" in todo:
        with _time("picking"):
            res["picking"] = picking.run_picking(cfg, events=events, device=device)
        log(f"picking: {sum(res['picking'].values())} picks / "
            f"{len(res['picking'])} events  ({timings['picking']:.1f}s)")
    if "hypoinverse" in todo:
        with _time("hypoinverse"):
            res["hypoinverse"] = list(hypoinverse.run_hypoinverse(cfg, velmodels=velmodels))
        log(f"hypoinverse: located with {res['hypoinverse']}  ({timings['hypoinverse']:.1f}s)")
    # Single-event guard: ph2dt + dtct + xcorr + dtcc are RELATIVE methods and need >=2
    # located events. ph2dt itself doesn't check and crashes with SIGFPE (divide-by-zero
    # in the per-pair statistics) on a single-event .arc -- so a partial NECIS download
    # that yields only 1 located event would kill the run instead of producing a clean
    # "absolute-locations-only" result. Skip the relative chain in that case and continue
    # straight to focal_mechanism (which CAN still run on 1 event if there are polarities).
    _relative_stages = {"ph2dt", "dtct", "rereference", "xcorr", "dtcc"}
    if todo & _relative_stages:
        n_located = _count_located_events(cfg, arc_velmodel)
        if n_located < 2:
            log(f"only {n_located} event(s) located -- skipping ph2dt/dtct/xcorr/dtcc "
                f"(relative-relocation stages need >=2 events)")
            todo -= _relative_stages
    if "ph2dt" in todo:
        if getattr(cfg, "reloc_backend", "hypodd") == "relocdd_py":
            # relocDD-py rebuilds dt.ct/event.dat from the located .sum + SAC picks in its
            # own dtct/dtcc stages (run_relocdd_py_full), so the Fortran ph2dt stage is
            # skipped. It would otherwise fail for the HypoSVI path, which produces no
            # HYPOINVERSE .arc for prep_ph2dt/ncsn2pha to consume.
            res["ph2dt"] = "skipped (relocdd_py rebuilds dt.ct from the .sum)"
            log("ph2dt: skipped — relocdd_py backend rebuilds dt.ct from the .sum")
        else:
            with _time("ph2dt"):
                hypodd.prep_ph2dt(cfg, velmodel=arc_velmodel)
                hypodd.run_ph2dt(cfg)
            res["ph2dt"] = "ok"
            log(f"ph2dt: dt.ct / event.dat written  ({timings['ph2dt']:.1f}s)")
    if "dtct" in todo:
        with _time("dtct"):
            res["dtct"] = hypodd.run_dtct(cfg, velmodel=arc_velmodel)
        log(f"dtct: {res['dtct'].split('/pipeline/')[-1]}  ({timings['dtct']:.1f}s)")
    if "rereference" in todo:
        with _time("rereference"):
            res["rereference"] = rereference.rereference_origins(cfg, velmodel=arc_velmodel)
        log(f"rereference: {res['rereference']} events -> {arc_velmodel} origins  "
            f"({timings['rereference']:.1f}s)")
    if "xcorr" in todo:
        with _time("xcorr"):
            res["xcorr"] = xcorr.run_xcorr(cfg, velmodel=arc_velmodel, cores=cores,
                                           xcorr_backend=getattr(cfg, "xcorr_backend", "obspy"),
                                           resume=xcorr_resume)
        log(f"xcorr: {res['xcorr']['pairs']} pairs x {res['xcorr']['stations']} stations  "
            f"({timings['xcorr']:.1f}s)")
    if "dtcc" in todo:
        with _time("dtcc"):
            res["dtcc"] = hypodd.run_dtcc(cfg, variant=dtcc_variant)
        log(f"dtcc[{dtcc_variant}]: {res['dtcc'].split('/pipeline/')[-1]}  ({timings['dtcc']:.1f}s)")
    if "focal_mechanism" in todo:
        with _time("focal_mechanism"):
            res["focal_mechanism"] = focal_mechanism.run_focal_mechanism(
                cfg, velmodel=fm_velmodel or cfg.fm_velmodel)
        nhi = sum(q in cfg.fm_quality_keep for q in res["focal_mechanism"].values())
        log(f"focal_mechanism: {len(res['focal_mechanism'])} mechanisms, "
            f"{nhi} high-confidence [{'/'.join(cfg.fm_quality_keep)}]  "
            f"({timings['focal_mechanism']:.1f}s)")
    if "report" in todo:
        # Compile the standard beamer PDF run summary from on-disk products. This is a
        # convenience product, NOT the scientific output, so a failure here (missing
        # tectonic, a bad figure) must never fail the run — log a warning and move on.
        pdf = None
        with _time("report"):
            try:
                from pipeline import reporting
                pdf = reporting.make_run_summary(
                    cfg.name, velmodel=arc_velmodel,
                    fm_velmodel=fm_velmodel or getattr(cfg, "fm_velmodel", arc_velmodel))
                res["report"] = pdf
            except Exception as e:  # noqa: BLE001
                res["report"] = f"skipped ({type(e).__name__})"
                log(f"report: SKIPPED — {type(e).__name__}: {e}")
        if pdf:
            log(f"report: {pdf.split('/pipeline/')[-1]}  ({timings['report']:.1f}s)")

    # End-of-run timing summary (only if there's any timing — i.e. at least one stage ran)
    if verbose and timings:
        total = sum(timings.values())
        col_w = max(len(s) for s in timings) + 2
        print(f"\n[{cfg.name}] === stage timings ===", flush=True)
        for stage in STAGES:                                     # canonical pipeline order
            if stage in timings:
                t = timings[stage]
                pct = 100 * t / total if total > 0 else 0.0
                print(f"  {stage:<{col_w}} {t:>7.1f}s   {pct:>5.1f}%", flush=True)
        print(f"  {'TOTAL':<{col_w}} {total:>7.1f}s", flush=True)
    res["_timings"] = timings
    return res
