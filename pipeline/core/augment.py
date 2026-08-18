"""Helpers for incremental cluster augmentation (adding events to a processed run).

The augmentation workflow (driven by the caller, e.g. PocketQuake's `--augment`) re-runs
HYPOINVERSE over the whole cluster after new events were gathered + picked. hyp1.40 locates
each event independently, so existing events' solutions are expected to reproduce — but that
is VERIFIED here, not assumed: `sum_snapshot` captures the pre-augment `.sum` solutions and
`verify_sums` compares them after the re-run. Events whose origin moved beyond tolerance
must have their cached dt.cc pairs invalidated (xcorr.invalidate_pairs) because the dt.cc
values reference the origin times.

`clear_bootstrap_caches` removes cached bootstrap error products whose event set changed
(belt-and-braces on top of the nev/evhash provenance tags in hypodd.bootstrap_relocation).
"""
from __future__ import annotations

import os
from glob import glob

from pipeline import config
from pipeline.core import sumio, evmap


def sum_snapshot(cfg, velmodels=("kim1983", "kim2011")) -> dict:
    """{velmodel: {cuspid: (time UTCDateTime, lat, lon, depth_km)}} from the current `.sum`
    files; a velmodel with no/empty .sum maps to {}."""
    snap = {}
    for vm in velmodels:
        df = sumio.read_sum(config.sum_file(cfg, vm))
        snap[vm] = {int(r.id): (r.time, float(r.lat), float(r.lon), float(r.depth))
                    for r in df.itertuples()}
    return snap


def verify_sums(cfg, snapshot, tol_t=0.005, tol_deg=1e-4, tol_km=0.05) -> dict:
    """Compare the current `.sum` files against a pre-augment `sum_snapshot`.

    Returns {velmodel: [event_id]} of previously-located events whose ORIGIN TIME moved
    beyond tol_t seconds — or which vanished from the new `.sum`. Only these invalidate
    cached dt.cc pairs: a dt.cc value is `(t1+shift-ot1) - (t2-ot2)`, so it depends on the
    two origin TIMES but not at all on epicenter/depth (those live in event.dat, rebuilt
    every run). Position-only moves beyond tol_deg/tol_km are logged for the record but do
    NOT trigger pair invalidation. Note the `.sum` prints seconds at 0.01 s and lat/lon at
    1e-4 deg, so one-quantum flips (a re-location started from re-referenced headers can
    converge one print-quantum away) are the typical "move" here — a 10 ms time flip still
    matters for dt.cc (10x the 1 ms xcorr slide), hence it invalidates."""
    dir_of = evmap.dir_of_cuspid(cfg)
    moved = {}
    for vm, old in snapshot.items():
        cur = sum_snapshot(cfg, velmodels=(vm,))[vm]
        bad, pos_only = [], []
        for cusp, (t0, lat0, lon0, z0) in old.items():
            if cusp not in cur:
                print(f"[augment] WARN {vm}: cuspid {cusp} ({dir_of.get(cusp, '?')}) "
                      f"vanished from the re-located .sum")
                bad.append(cusp)
                continue
            t1, lat1, lon1, z1 = cur[cusp]
            delta = (f"dt={t1 - t0:+.3f}s dlat={lat1 - lat0:+.5f} dlon={lon1 - lon0:+.5f} "
                     f"dz={z1 - z0:+.2f}km")
            if abs(t1 - t0) > tol_t:
                print(f"[augment] WARN {vm}: cuspid {cusp} ({dir_of.get(cusp, '?')}) origin "
                      f"time moved — cached dt.cc pairs stale: {delta} (tol {tol_t}s)")
                bad.append(cusp)
            elif (abs(lat1 - lat0) > tol_deg or abs(lon1 - lon0) > tol_deg
                    or abs(z1 - z0) > tol_km):
                print(f"[augment] {vm}: cuspid {cusp} ({dir_of.get(cusp, '?')}) position "
                      f"moved (dt.cc unaffected, no invalidation): {delta}")
                pos_only.append(cusp)
        moved[vm] = sorted({dir_of[c] for c in bad if c in dir_of})
        n_ok = len(old) - len(bad) - len(pos_only)
        print(f"[augment] {vm}: existing .sum rows reproduced {n_ok}/{len(old)}"
              + (f" — {len(bad)} time-moved/vanished, {len(pos_only)} position-only"
                 if (bad or pos_only) else ""))
    return moved


def clear_bootstrap_caches(cfg) -> list:
    """Delete bootstrap_errors.csv + bootstrap_samples.npz in the dt.ct branch, the dt.cc
    branch, and every dt.cc variant subdir. Returns the deleted paths."""
    dirs = [config.dtct_dir(cfg), config.dtcc_dir(cfg)]
    dirs += [d for d in glob(os.path.join(config.dtcc_dir(cfg), "*"))
             if os.path.isdir(d) and os.path.exists(os.path.join(d, "hypoDD.inp"))]
    gone = []
    for d in dirs:
        for fn in ("bootstrap_errors.csv", "bootstrap_samples.npz"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                os.remove(p)
                gone.append(p)
    return gone
