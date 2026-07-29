"""Canonical cuspid <-> event-dir mapping — the ONE place the id scheme lives.

Legacy scheme: ``cuspid = cfg.cuspid_offset + index over sorted(waveforms_100km/20*)``. That derives
both the event list and the numbering from a filesystem glob, so any stale directory silently shifts
every subsequent id: uf_2011 had 446 dirs for a 445-event catalog and every cuspid >= 17 was off by
one — wrong origins would have been attached to 428 events had the mismatch not crashed first.

Manifest scheme (opt-in): if ``<output_root>/event_manifest.csv`` exists (columns
``event_id,event_idx``; written by the caller's staging, e.g. ufpipe's stage.py), then
``cuspid = cfg.cuspid_offset + event_idx`` and ONLY manifest events exist. The id is then the
caller's own catalog key: stable across full/QC subsets, immune to dir-count drift, meaningful in
every downstream file (.sum, .arc, event.dat, dt.ct, dt.cc, hypoDD.reloc) — and same-second doublets
keep distinct identities. Stale dirs are simply not in the manifest and are ignored.

Every stage that maps ids to dirs (write_phs, rereference, xcorr, hypodd, viz, focal_mechanism)
must go through these two functions rather than re-deriving the enumerate scheme.
"""
import os
from glob import glob

from pipeline import config


def manifest_path(cfg):
    return os.path.join(cfg.output_root, "event_manifest.csv")


def dir_of_cuspid(cfg):
    """{cuspid -> event_id (waveform dir basename)}. Manifest scheme when the manifest exists,
    legacy sorted-dir enumeration otherwise."""
    dirs = sorted(glob(os.path.join(config.waveforms_dir(cfg), "20*")))
    mp = manifest_path(cfg)
    if os.path.exists(mp):
        import pandas as pd
        m = pd.read_csv(mp, dtype={"event_id": str})
        have = {os.path.basename(d) for d in dirs}
        out = {}
        for r in m.itertuples():
            eid = str(r.event_id)
            if eid in have:
                out[cfg.cuspid_offset + int(r.event_idx)] = eid
        return out
    return {cfg.cuspid_offset + i: os.path.basename(d) for i, d in enumerate(dirs)}


def cuspid_of_dir(cfg):
    """{event_id (dir basename) -> cuspid}; inverse of dir_of_cuspid."""
    return {e: c for c, e in dir_of_cuspid(cfg).items()}
