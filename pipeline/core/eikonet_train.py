"""Train an EikoNet travel-time network for a layered velocity model, for HypoSVI.

This is the one-time, reproducible recipe behind the `loc_backend="hyposvi"` path.
It trains one P and one S EikoNet over a geographic box, writing each checkpoint
dir plus an `eikonet_meta.json` that the HypoSVI adapter
(`pipeline/core/hyposvi_backend.py`) reads to reconstruct the model.

Run for any velocity model in `cfg.velocity_models` (kim1983, kim2011, ...):

    python -m pipeline.core.eikonet_train --velmodel kim1983 \
        --out pipeline/velocity_models/eikonet_kim1983

Two hard-won settings are baked in as defaults (see docs/python_backend/README.md):

  * projection MUST carry `+units=km`. EikoNet projects lon/lat -> UTM but does
    not divide by 1000, while depth and velocity are km / km·s⁻¹. Without +units=km
    the eikonal loss is unit-inconsistent by 1000x and travel times come back
    ~1000x too large (HypoSVI then returns NaN).
  * sample count is capped (default 150k). EikoNet's train loop leaks the autograd
    graph within an epoch (peak RSS scales with samples/epoch): 500k -> ~230 GB
    (thrashes a shared box), 150k -> ~40-70 GB. A 1D layered model converges fine
    at 150k with more epochs.

The geographic box must CONTAIN every cluster you intend to locate with this model
(the kim1983 reference box is all of Korea: lon 125-131, lat 33-40, depth 0-50 km).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# Cap thread fan-out BEFORE importing numpy/torch (EikoNet spawns one thread per
# core otherwise and, combined with the per-epoch graph leak, balloons RSS).
_NTH = os.environ.get("EIKONET_THREADS", "16")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NTH)

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(int(_NTH))


# Reference all-Korea box for the bundled kim1983 / kim2011 models. Override per run.
DEFAULT_BBOX = {"lon": (125.0, 131.0), "lat": (33.0, 40.0), "depth": (0.0, 50.0)}
DEFAULT_PROJECTION = "+proj=utm +zone=52 +ellps=WGS84 +units=km"   # +units=km is mandatory


def _eikonet_modules(eikonet_dir=None):
    d = eikonet_dir or os.environ.get("EIKONET_DIR")
    if d and d not in sys.path:
        sys.path.insert(0, d)
    from EikoNet.database import Graded1DVelocity
    from EikoNet.model import Model
    return Graded1DVelocity, Model


def _best_checkpoint(model_dir):
    cks = glob.glob(os.path.join(model_dir, "Model_Epoch_*.pt"))
    if not cks:
        raise RuntimeError(f"no checkpoints written in {model_dir}")
    return min(cks, key=lambda f: float(f.split("ValLoss_")[1][:-3]))


def train_phase(velmodel_name, phase, rows, out_dir, bbox, projection,
                samples=150000, epochs=60, eikonet_dir=None, device="cpu"):
    """Train one phase (P or S) and write its checkpoint dir + eikonet_meta.json.
    `rows` is a list of (depth_km, velocity_km_s) layer points. Returns the meta dict."""
    Graded1DVelocity, Model = _eikonet_modules(eikonet_dir)
    ph = phase.lower()
    csv_path = os.path.join(out_dir, f"{velmodel_name}_{ph}.csv")
    pd.DataFrame(rows, columns=["Depth", "V"]).to_csv(csv_path, header=False, index=False)

    xmin = [bbox["lon"][0], bbox["lat"][0], bbox["depth"][0]]
    xmax = [bbox["lon"][1], bbox["lat"][1], bbox["depth"][1]]
    vmodel = Graded1DVelocity(csv_path, xmin=xmin, xmax=xmax, projection=projection)
    model_dir = os.path.join(out_dir, f"{velmodel_name}_{ph}")
    os.makedirs(model_dir, exist_ok=True)
    # Clear stale checkpoints from a previous run so best-of selection can't pick up
    # an old (e.g. CPU) checkpoint with a lower val-loss than this run produces.
    for old in glob.glob(os.path.join(model_dir, "Model_Epoch_*.pt")):
        os.remove(old)
    model = Model(model_dir, vmodel, device=device)
    model.Params["Training"]["Number of sample points"] = int(samples)
    model.Params["Training"]["Number of Epochs"] = int(epochs)
    model.Params["Training"]["Save Every * Epoch"] = 10

    t0 = time.time()
    model.train()
    print(f"[eikonet_train] {velmodel_name} {phase} done in {(time.time()-t0)/60:.1f} min")

    best = _best_checkpoint(model_dir)
    meta = {
        "velmodel": velmodel_name, "phase": phase.upper(),
        "csv": os.path.basename(csv_path), "xmin": xmin, "xmax": xmax,
        "projection": projection, "best_checkpoint": os.path.basename(best),
        "best_val_loss": float(best.split("ValLoss_")[1][:-3]),
        "normalisation": "OffsetMinMax", "residual_blocks": 10,
    }
    json.dump(meta, open(os.path.join(model_dir, "eikonet_meta.json"), "w"), indent=2)
    # Remove the regenerable training-sample cache to keep the artifact small.
    for npy in ("Xp.npy", "Yp.npy"):
        p = os.path.join(model_dir, npy)
        if os.path.exists(p):
            os.remove(p)
    return meta


def _step_rows(vrows, depth_max, eps=0.01):
    """Convert eq-cycle layer-top (velocity, depth) pairs into (depth, velocity) rows
    that make EikoNet's interpolator approximate a LAYERED step model — matching how
    HYPOINVERSE reads the same `.crh` (constant velocity within each layer), so the
    two backends share the same velocity model. Also extends the bottom layer to
    `depth_max` (the training box can be deeper than the deepest defined layer).

    Layers [(v0,z0),(v1,z1),(v2,z2)] -> (z0,v0),(z1-eps,v0),(z1,v1),(z2-eps,v1),(z2,v2),(depth_max,v2)."""
    layers = sorted(((float(v), float(z)) for (v, z) in vrows), key=lambda r: r[1])
    rows = []
    for i, (vel, dep) in enumerate(layers):
        if i > 0:
            rows.append((dep - eps, layers[i - 1][0]))    # bottom of previous layer
        rows.append((dep, vel))                           # top of this layer
    if depth_max > layers[-1][1]:
        rows.append((depth_max, layers[-1][0]))           # extend bottom layer to box floor
    return rows


def train_velmodel(vmodel, out_dir, bbox=None, projection=DEFAULT_PROJECTION,
                   samples=150000, epochs=60, eikonet_dir=None, device="cpu"):
    """Train P+S EikoNets for a pipeline VelModel (has .name, .p_rows, .s_rows).

    .p_rows / .s_rows are (velocity, depth) layer-tops in the eq-cycle convention.
    Returns {'P': meta, 'S': meta}."""
    os.makedirs(out_dir, exist_ok=True)
    bbox = bbox or DEFAULT_BBOX
    out = {}
    for phase, vrows in (("P", vmodel.p_rows), ("S", vmodel.s_rows)):
        rows = _step_rows(vrows, bbox["depth"][1])
        out[phase] = train_phase(vmodel.name, phase, rows, out_dir, bbox, projection,
                                 samples=samples, epochs=epochs, eikonet_dir=eikonet_dir,
                                 device=device)
        print(f"[eikonet_train] {vmodel.name} {phase}: best={out[phase]['best_checkpoint']} "
              f"(val={out[phase]['best_val_loss']:.4f})")
    return out


def _vmodel_from_csv(name, csv_path):
    """Build a velocity-model object (.name, .p_rows, .s_rows) from a user CSV.

    CSV = one row per layer TOP, columns `depth_km, vp_kms, vs_kms` (header optional):
        0,5.98,3.40
        15,6.38,3.79
        32,7.95,4.58
    Returned in the eq-cycle convention p_rows = ((velocity, depth), ...)."""
    from types import SimpleNamespace
    df = pd.read_csv(csv_path, header=None, comment="#")
    # drop a header row if the first row isn't numeric
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.astype(float)
    if df.shape[1] < 3:
        raise SystemExit(f"{csv_path}: need 3 columns (depth_km, vp_kms, vs_kms), got {df.shape[1]}")
    depth, vp, vs = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    p_rows = tuple((float(v), float(z)) for v, z in zip(vp, depth))
    s_rows = tuple((float(v), float(z)) for v, z in zip(vs, depth))
    return SimpleNamespace(name=name, p_rows=p_rows, s_rows=s_rows)


def _resolve_device(device):
    if device == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train P+S EikoNets for a velocity model (HypoSVI).")
    ap.add_argument("--velmodel", required=True, help="velocity-model name (names the output, e.g. kim1983 or my1d)")
    ap.add_argument("--vel-csv", default=None,
                    help="train YOUR OWN 1-D model from a CSV (depth_km,vp_kms,vs_kms per layer top). "
                         "If omitted, --velmodel is looked up in the bundled cluster configs.")
    ap.add_argument("--cluster", default=None, help="cluster whose config holds the velocity model (config lookup only)")
    ap.add_argument("--out", default=None, help="output dir (default: pipeline/velocity_models/eikonet_<velmodel>)")
    ap.add_argument("--samples", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lon", default=None, help="lon0,lon1 box (default 125,131 — must cover your cluster)")
    ap.add_argument("--lat", default=None, help="lat0,lat1 box (default 33,40)")
    ap.add_argument("--depth", default=None, help="dep0,dep1 km box (default 0,50)")
    ap.add_argument("--device", default="auto", help="'auto' (GPU if available) | 'cpu' | 'cuda:N'")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.getcwd())

    if args.vel_csv:
        # User-supplied velocity model — no config registration needed.
        vmodel = _vmodel_from_csv(args.velmodel, args.vel_csv)
    else:
        from pipeline import config as _cfg
        import importlib
        names = [args.cluster] if args.cluster else list(_cfg.CLUSTER_NAMES)
        vmodel = None
        for nm in names:
            try:
                mod = importlib.import_module(f"pipeline.clusters.{nm}")
            except Exception:
                continue
            vmodel = next((v for v in mod.CONFIG.velocity_models if v.name == args.velmodel), None)
            if vmodel is not None:
                break
        if vmodel is None:
            raise SystemExit(
                f"velocity model {args.velmodel!r} not found in any cluster config. "
                "Provide your own with --vel-csv depth_km,vp_kms,vs_kms.")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "velocity_models", f"eikonet_{args.velmodel}")
    out = os.path.abspath(out)
    device = _resolve_device(args.device)
    print(f"[eikonet_train] velmodel={args.velmodel} out={out} device={device} "
          f"samples={args.samples} epochs={args.epochs}")

    bbox = dict(DEFAULT_BBOX)
    if args.lon:   bbox["lon"]   = tuple(float(x) for x in args.lon.split(","))
    if args.lat:   bbox["lat"]   = tuple(float(x) for x in args.lat.split(","))
    if args.depth: bbox["depth"] = tuple(float(x) for x in args.depth.split(","))

    train_velmodel(vmodel, out, bbox=bbox, samples=args.samples,
                   epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
