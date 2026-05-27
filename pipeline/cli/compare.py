"""Regression CLI: compare a cluster's fresh outputs vs the frozen baseline.

    python -m pipeline.cli.compare --cluster gwangyang
    python -m pipeline.cli.compare --cluster gwangyang --velmodels kim1983,kim2011
"""
import argparse

from pipeline import config
from pipeline.regression import compare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--velmodels", default="kim1983,kim2011")
    args = ap.parse_args()

    cfg = config.load_cluster(args.cluster)
    vms = tuple(v for v in args.velmodels.split(",") if v)
    df = compare.compare_all(cfg, velmodels=vms)

    with __import__("pandas").option_context("display.width", 200,
                                             "display.max_columns", 50):
        print(df.to_string(index=False))
    n_pass = int(df["passed"].sum())
    print(f"\n{n_pass}/{len(df)} stages PASS"
          + ("  -> ALL PASS" if n_pass == len(df) else "  -> see FAILs above"))


if __name__ == "__main__":
    main()
