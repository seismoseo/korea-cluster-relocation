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

from pipeline import config
from pipeline.core import (stations, waveforms, picking, hypoinverse, hypodd,
                           rereference, xcorr)

STAGES = ["stations", "waveforms", "picking", "hypoinverse", "ph2dt", "dtct",
          "rereference", "xcorr", "dtcc"]


def run_cluster(cfg, stage_from="stations", through="dtct",
                velmodels=("kim1983", "kim2011"), arc_velmodel="kim1983",
                device="cpu", events=None, dtcc_variant="default", cores=None,
                verbose=True) -> dict:
    i0, i1 = STAGES.index(stage_from), STAGES.index(through)
    todo = set(STAGES[i0:i1 + 1])
    res = {}

    def log(msg):
        if verbose:
            print(f"[{cfg.name}] {msg}", flush=True)

    if "stations" in todo:
        used = stations.run_stations(cfg)
        res["stations"] = len(used)
        log(f"stations: {len(used)} used")
    if "waveforms" in todo:
        res["waveforms"] = waveforms.run_waveforms(cfg, events=events)
        log(f"waveforms: gathered {sum(res['waveforms'].values())} files / {len(res['waveforms'])} events")
    if "picking" in todo:
        res["picking"] = picking.run_picking(cfg, events=events, device=device)
        log(f"picking: {sum(res['picking'].values())} picks / {len(res['picking'])} events")
    if "hypoinverse" in todo:
        res["hypoinverse"] = list(hypoinverse.run_hypoinverse(cfg, velmodels=velmodels))
        log(f"hypoinverse: located with {res['hypoinverse']}")
    if "ph2dt" in todo:
        hypodd.prep_ph2dt(cfg, velmodel=arc_velmodel)
        hypodd.run_ph2dt(cfg)
        res["ph2dt"] = "ok"
        log("ph2dt: dt.ct / event.dat written")
    if "dtct" in todo:
        res["dtct"] = hypodd.run_dtct(cfg)
        log(f"dtct: {res['dtct'].split('/pipeline/')[-1]}")
    if "rereference" in todo:
        res["rereference"] = rereference.rereference_origins(cfg, velmodel=arc_velmodel)
        log(f"rereference: {res['rereference']} events -> {arc_velmodel} origins")
    if "xcorr" in todo:
        res["xcorr"] = xcorr.run_xcorr(cfg, velmodel=arc_velmodel, cores=cores)
        log(f"xcorr: {res['xcorr']['pairs']} pairs x {res['xcorr']['stations']} stations")
    if "dtcc" in todo:
        res["dtcc"] = hypodd.run_dtcc(cfg, variant=dtcc_variant)
        log(f"dtcc[{dtcc_variant}]: {res['dtcc'].split('/pipeline/')[-1]}")
    return res
