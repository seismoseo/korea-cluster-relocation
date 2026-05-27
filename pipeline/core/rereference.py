"""
Stage — re-reference SAC origins to the HYPOINVERSE solution
(ports `5.Modify_origin_time_to_HypoInv_time.ipynb`).

After gather + picking, each run SAC in `waveforms_100km/` carries the *catalog* origin
(`nz*` = catalog_KST − kst_offset) and picks `a` (P) / `t0` (S) expressed relative to it.
HypoDD's catalog `event.dat` uses the HYPOINVERSE origins, so the cross-correlation `dt.cc`
must be measured against those same origins. This stage rewrites each event's SAC reference
time (`nz*`) to its HYPOINVERSE `<velmodel>.sum` origin and re-expresses `a`/`t0` relative to
the new origin, **in place** under `runs/<cluster>/waveforms_100km/`.

The physical (absolute) pick and sample times are unchanged: obspy recomputes the SAC begin
offset `b` from `starttime` on write, so `starttime` is preserved and `(starttime − b)` becomes
the new origin. The operation is **idempotent** for a fixed target velmodel — `ptime_abs =
(starttime − b) + a` is recovered from the *current* header each run, so re-running it (or
re-running it against a different velmodel) never corrupts the pick.

Default velmodel is **kim1983** — the model whose `.arc` feeds ph2dt/dt.ct, so dt.ct and dt.cc
share one origin reference (verified: the baseline `waveforms_100km` reference time equals the
kim1983 `.sum` origin).

Event ↔ dir mapping is by cuspid (`.sum` `ID-NUM % cuspid_offset` → sorted-dir index, the same
scheme `hypoinverse.write_phs` stamps), so a dropped/unlocated event cannot shift the others.
"""
from __future__ import annotations

import os
from glob import glob

from obspy import read

from pipeline import config
from pipeline.core import sumio

UNDEF = -12345.0   # SAC "header not set" sentinel


def rereference_origins(cfg, velmodel="kim1983") -> int:
    """Rewrite every run-tree event's SAC origin (nz*) + picks (a/t0) to the
    `<velmodel>.sum` solution. Returns the number of events re-referenced."""
    sumdf = sumio.read_sum(config.sum_file(cfg, velmodel))
    origin_of = {int(r.id) % cfg.cuspid_offset: r.time for r in sumdf.itertuples()}

    wf_root = config.assert_writable(config.waveforms_dir(cfg))
    dirs = sorted(glob(os.path.join(wf_root, "20*")))

    n_ev = n_files = 0
    for idx, origin in sorted(origin_of.items()):
        if idx >= len(dirs):
            print(f"[rereference] WARN cuspid index {idx} >= {len(dirs)} event dirs — skipped")
            continue
        for f in glob(os.path.join(dirs[idx], "*.sac")):
            tr = read(f)[0]
            s = tr.stats.sac
            ref = tr.stats.starttime - s.b          # current reference (origin)
            a, t0 = s.get("a", UNDEF), s.get("t0", UNDEF)
            if a != UNDEF:
                s.a = (ref + a) - origin            # = ptime_abs - new_origin
            if t0 != UNDEF:
                s.t0 = (ref + t0) - origin
            s.nzyear, s.nzjday = origin.year, origin.julday
            s.nzhour, s.nzmin, s.nzsec = origin.hour, origin.minute, origin.second
            s.nzmsec = int(round(origin.microsecond / 1000))
            tr.write(f, format="SAC")
            n_files += 1
        n_ev += 1
    print(f"[rereference] {velmodel}: re-referenced {n_files} SAC across {n_ev} events "
          f"under {wf_root}")
    return n_ev
