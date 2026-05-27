"""Run the full (or partial) relocation chain for a cluster.

    python -m pipeline.cli.run_pipeline --cluster gwangyang                       # -> dt.ct
    python -m pipeline.cli.run_pipeline --cluster kimcheon --stage-from picking --through dtct
    # dt.cc branch (heavy cross-correlation) — PIN CORES on the shared box:
    taskset -c 0-9 python -m pipeline.cli.run_pipeline --cluster gwangyang --through dtcc --cores 10
    taskset -c 0-9 python -m pipeline.cli.run_pipeline --cluster gwangyang \\
        --stage-from rereference --through dtcc --dtcc-variant no_main --cores 10

Stages: stations waveforms picking hypoinverse ph2dt dtct rereference xcorr dtcc.
"""
import argparse

from pipeline import config
from pipeline.core import pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--stage-from", default="stations", choices=pipeline.STAGES)
    ap.add_argument("--through", default="dtct", choices=pipeline.STAGES)
    ap.add_argument("--velmodels", default="kim1983,kim2011")
    ap.add_argument("--arc-velmodel", default="kim1983",
                    help="velocity model whose HYPOINVERSE .arc feeds ph2dt AND whose "
                         ".sum the rereference/xcorr stages use")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--events", default=None, help="comma-separated event_ids (default all)")
    ap.add_argument("--dtcc-variant", default="default",
                    help="which hypodd_dtcc_variants entry to relocate (default/no_main/...)")
    ap.add_argument("--cores", type=int, default=None,
                    help="xcorr worker cap (default cfg.num_cores; always pin with taskset)")
    args = ap.parse_args()

    cfg = config.load_cluster(args.cluster)
    events = args.events.split(",") if args.events else None
    velmodels = tuple(v for v in args.velmodels.split(",") if v)
    pipeline.run_cluster(cfg, stage_from=args.stage_from, through=args.through,
                         velmodels=velmodels, arc_velmodel=args.arc_velmodel,
                         device=args.device, events=events,
                         dtcc_variant=args.dtcc_variant, cores=args.cores)


if __name__ == "__main__":
    main()
