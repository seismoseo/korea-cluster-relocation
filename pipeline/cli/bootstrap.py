"""Bootstrap HypoDD relative-location uncertainty (95% error bars) by resampling the
differential-time data and re-inverting `n` times (global-observation resample; inversion held
fixed at the calibrated hypoDD.inp). Writes `bootstrap_errors.csv` into the relocation dir; the
viz functions then draw 95% (±1.96σ) error bars. Deterministic for a given --seed.

    python -m pipeline.cli.bootstrap --cluster kimcheon --suffix _pnplus --branch dtcc --n 1000
    python -m pipeline.cli.bootstrap --cluster gwangyang --suffix _pnplus --branch both --cores 10
"""
import argparse
import os

import numpy as np

from pipeline import config
from pipeline.core import hypodd, sumio


def _run(cfg, branch, n, seed, cores):
    df = hypodd.bootstrap_relocation(cfg, branch=branch, n=n, seed=seed, cores=cores)
    bdir = config.dtcc_dir(cfg) if branch == "dtcc" else config.dtct_dir(cfg)
    rl = sumio.read_reloc(os.path.join(bdir, "hypoDD.reloc"))
    print(f"\n[{branch}] {cfg.name}: {int(df.ex95.notna().sum())}/{len(df)} events with a 95% CI "
          f"(n={n}, seed={seed})")
    print(f"  median 95% half-width (m):  ex={np.nanmedian(df.ex95):6.1f}  ey={np.nanmedian(df.ey95):6.1f}  "
          f"ez={np.nanmedian(df.ez95):6.1f}")
    print(f"  median HypoDD internal (m): ex={rl.ex.median():6.2f}  ey={rl.ey.median():6.2f}  "
          f"ez={rl.ez.median():6.2f}   <- known to underestimate")
    print(f"  -> {os.path.join(bdir, 'bootstrap_errors.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--suffix", default="",
                    help="run-dir suffix, e.g. _pnplus (output_root = runs/<cluster><suffix>)")
    ap.add_argument("--branch", default="dtcc", choices=("dtcc", "dtct", "both"))
    ap.add_argument("--n", type=int, default=1000, help="bootstrap replicas (default 1000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible)")
    ap.add_argument("--cores", type=int, default=None, help="parallel HypoDD runs (default cfg.num_cores)")
    args = ap.parse_args()

    cfg = config.load_cluster(args.cluster)
    if args.suffix:
        cfg = config.tune(cfg, output_root=os.path.join(config.RUNS_ROOT, f"{args.cluster}{args.suffix}"))
    for br in (("dtct", "dtcc") if args.branch == "both" else (args.branch,)):
        _run(cfg, br, args.n, args.seed, args.cores)


if __name__ == "__main__":
    main()
