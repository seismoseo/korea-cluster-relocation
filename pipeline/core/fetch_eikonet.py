"""Download pretrained EikoNet weights for HypoSVI from the PocketQuake GitHub release.

The `.pt` weights are too large to track in git (~90 MB each), so they ship as release
assets. The training recipe, velocity CSV, and each `eikonet_meta.json` ARE tracked —
this script reads the meta to learn the exact local filename and pulls the matching
asset into place, so the backend's auto-discovery then finds it.

    python -m pipeline.core.fetch_eikonet                 # all bundled velocity models
    python -m pipeline.core.fetch_eikonet --velmodel kim1983

Asset naming on the release: `<velmodel>_<phase>.pt` (e.g. kim1983_p.pt). It is saved
locally as `pipeline/velocity_models/eikonet_<velmodel>/<velmodel>_<phase>/<best_checkpoint>`
where <best_checkpoint> comes from that model's eikonet_meta.json.

Primary path uses the `gh` CLI (`gh release download`); falls back to `curl` against the
public asset URL if `gh` is unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

DEFAULT_REPO = "seismoseo/PocketQuake"
DEFAULT_TAG = "eikonet-weights-v1"


def _velmodels_root():
    from pipeline import config
    return os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "velocity_models")


def _bundled_velmodels():
    """Velocity models that have a tracked eikonet_<vm>/ dir (with meta.json)."""
    root = _velmodels_root()
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if name.startswith("eikonet_") and os.path.isdir(os.path.join(root, name)):
            out.append(name[len("eikonet_"):])
    return out


def _target_path(velmodel, phase):
    """Local destination for a weight: model_dir/<best_checkpoint> (from meta.json)."""
    model_dir = os.path.join(_velmodels_root(), f"eikonet_{velmodel}", f"{velmodel}_{phase}")
    meta_path = os.path.join(model_dir, "eikonet_meta.json")
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"no eikonet_meta.json at {model_dir} — is '{velmodel}' a bundled model?")
    best = json.load(open(meta_path)).get("best_checkpoint")
    if not best:
        raise RuntimeError(f"eikonet_meta.json at {model_dir} has no best_checkpoint")
    return model_dir, os.path.join(model_dir, best)


def _have_gh():
    return shutil.which("gh") is not None


def _download_gh(repo, tag, asset, dest_dir):
    subprocess.run(["gh", "release", "download", tag, "--repo", repo,
                    "--pattern", asset, "--dir", dest_dir, "--clobber"], check=True)
    return os.path.join(dest_dir, asset)


def _download_curl(repo, tag, asset, dest_path):
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    subprocess.run(["curl", "-fSL", url, "-o", dest_path], check=True)
    return dest_path


def fetch(velmodel, repo=DEFAULT_REPO, tag=DEFAULT_TAG, force=False):
    """Fetch P+S weights for one velocity model. Returns list of local paths written."""
    written = []
    for phase in ("p", "s"):
        model_dir, target = _target_path(velmodel, phase)
        if os.path.isfile(target) and not force:
            print(f"[fetch_eikonet] {velmodel} {phase.upper()}: already present ({os.path.basename(target)})")
            written.append(target)
            continue
        asset = f"{velmodel}_{phase}.pt"
        os.makedirs(model_dir, exist_ok=True)
        print(f"[fetch_eikonet] {velmodel} {phase.upper()}: downloading {asset} from {repo}@{tag} ...")
        if _have_gh():
            tmp = _download_gh(repo, tag, asset, model_dir)
            if os.path.abspath(tmp) != os.path.abspath(target):
                shutil.move(tmp, target)
        else:
            _download_curl(repo, tag, asset, target)
        print(f"[fetch_eikonet]   -> {target}")
        written.append(target)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch pretrained EikoNet weights for HypoSVI.")
    ap.add_argument("--velmodel", default="all", help="velocity model name, or 'all' (default)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag holding the .pt assets")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args(argv)

    vms = _bundled_velmodels() if args.velmodel == "all" else [args.velmodel]
    if not vms:
        raise SystemExit("no bundled EikoNet velocity models found (need eikonet_<vm>/ dirs with meta.json)")
    for vm in vms:
        fetch(vm, repo=args.repo, tag=args.tag, force=args.force)
    print(f"[fetch_eikonet] done: {', '.join(vms)}")


if __name__ == "__main__":
    main()
