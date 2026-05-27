"""
Freeze (and later verify) checksums of the committed per-cluster baseline outputs
that the regression harness compares against. The baselines are the existing
HYPOINVERSE `.sum`, HypoDD `.reloc`, `used_stations_100km.csv`, and `picks/*.csv`
files inside each read-only cluster directory.

Usage:
    python -m pipeline.regression.freeze_baseline freeze   # write the manifest
    python -m pipeline.regression.freeze_baseline verify   # check nothing drifted

The manifest is written under pipeline/regression/ (never into a cluster dir).
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

from pipeline import config

MANIFEST = os.path.join(config.PIPELINE_ROOT, "regression", "baseline_checksums.txt")

# Baseline artifacts to freeze, as globs relative to each cluster src dir.
BASELINE_GLOBS = (
    "1.HypoInv/**/*.sum",
    "2.HypoDD/**/hypoDD.reloc",
    "station_table/used_stations_100km.csv",
    "picks/*.csv",
)


def _iter_baseline_files():
    for d in config.CLUSTER_SRC_DIRS:
        root = os.path.join(config.PROJECT_ROOT, d)
        for pat in BASELINE_GLOBS:
            for f in glob.glob(os.path.join(root, pat), recursive=True):
                if os.path.isfile(f):
                    yield f


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    rows = []
    for f in sorted(set(_iter_baseline_files())):
        rel = os.path.relpath(f, config.PROJECT_ROOT)
        rows.append(f"{_sha256(f)}  {rel}")
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"froze {len(rows)} baseline files -> {MANIFEST}")


def verify():
    if not os.path.exists(MANIFEST):
        sys.exit(f"no manifest at {MANIFEST}; run 'freeze' first")
    expected = {}
    with open(MANIFEST) as fh:
        for line in fh:
            line = line.strip()
            if line:
                digest, rel = line.split("  ", 1)
                expected[rel] = digest
    drift = []
    for rel, digest in expected.items():
        path = os.path.join(config.PROJECT_ROOT, rel)
        if not os.path.exists(path):
            drift.append(f"MISSING  {rel}")
        elif _sha256(path) != digest:
            drift.append(f"CHANGED  {rel}")
    if drift:
        print("BASELINE DRIFT DETECTED:")
        print("\n".join(drift))
        sys.exit(1)
    print(f"OK: all {len(expected)} baseline files unchanged")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "freeze"
    {"freeze": freeze, "verify": verify}[cmd]()
