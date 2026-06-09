"""Compile a one-deck beamer PDF summary of a finished cluster run.

    python -m pipeline.cli.make_summary --cluster tongyeong
    python -m pipeline.cli.make_summary --cluster tongyeong --no-animate
    python -m pipeline.cli.make_summary --cluster gwangyang --velmodel kim2011

Output: runs/<cluster>/summary/<cluster>_summary.pdf (figures under summary/figs/).
Requires tectonic on PATH or $TECTONIC_BIN (see pipeline.reporting).
"""
import argparse

from pipeline import reporting


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--velmodel", default="kim1983",
                    help="velocity model whose .sum/reloc drive the figures (default kim1983)")
    ap.add_argument("--fm-velmodel", default=None,
                    help="velocity model for the focal-mechanism figures (default cfg.fm_velmodel)")
    ap.add_argument("--no-animate", action="store_true",
                    help="skip the time-lapse GIF / animated slide (faster)")
    ap.add_argument("--anim-fps", type=int, default=6)
    args = ap.parse_args()

    reporting.make_run_summary(
        args.cluster, velmodel=args.velmodel, fm_velmodel=args.fm_velmodel,
        animate=not args.no_animate, anim_fps=args.anim_fps,
    )


if __name__ == "__main__":
    main()
