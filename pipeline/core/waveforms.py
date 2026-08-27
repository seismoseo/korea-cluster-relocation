"""
Stage 2 + 3 — gather per-event waveforms and inject SAC headers (ports notebooks
`2.Gather_and_process_selected_waveforms` and `3.Modify_SAC_filename`).

For each catalog event it copies the *selected-sensor* 3-component SAC from the raw
archive, sets event/station headers (evla/evlo/stla/stlo) and the origin reference
time (origin_utc = catalog_kst - kst_offset), and writes the renamed file
`<event_id>.<net>.<sta>.<chan>.sac` into the run's `waveforms_100km/`.

`load_catalog()` lives here and is reused by picking/hypoinverse. Each event's id is
the origin UTC time `YYYYMMDDHHMMSS`, which equals the raw archive's event-dir name.
"""
from __future__ import annotations

import os
from glob import glob

import pandas as pd
from obspy import read, UTCDateTime

from pipeline import config
from pipeline.core import stations


# ------------------------------------------------------------------- catalog
def load_catalog(cfg) -> list[dict]:
    """KMA catalog rows -> [{event_id, origin(UTC), lat, lon, depth, mag, index}].

    Column names vary by cluster (Year vs year, BOM, ...) so they're normalised.
    origin_utc = catalog time (KST) - kst_offset_hours.
    """
    df = pd.read_csv(cfg.event_catalog_csv, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    out = []
    for i, r in df.iterrows():
        kst = UTCDateTime(int(r["year"]), int(r["month"]), int(r["day"]),
                          int(r["hour"]), int(r["minute"]), int(r["second"]))
        origin = kst - cfg.kst_offset_hours * 3600
        out.append(dict(
            event_id=origin.strftime("%Y%m%d%H%M%S"),
            origin=origin,
            lat=float(r["latitude"]), lon=float(r["longitude"]),
            depth=float(r["depth"]), mag=float(r["magnitude"]),
            index=int(i),
        ))
    return out


# ----------------------------------------------------------- per-event gather
def gather_event(cfg, used_table, ev) -> list[str]:
    """Copy + header-inject one event's selected-sensor SAC; return written paths."""
    raw_dir = stations.event_raw_dir(cfg, ev["event_id"])
    if raw_dir is None:
        return []
    layout = stations.event_layout(cfg, raw_dir)  # per-event for mixed; identity otherwise
    out_dir = config.assert_writable(config.event_wf_dir(cfg, ev["event_id"]))
    os.makedirs(out_dir, exist_ok=True)

    sensor_of = dict(zip(used_table["Code"], used_table["Sensor"]))
    coord_of = {r.Code: (r.Latitude, r.Longitude) for r in used_table.itertuples()}
    o = ev["origin"]
    written = []
    for sensor in cfg.sensor_priority:
        pat = stations.wf_glob(cfg, sensor, "*", layout=layout)
        # Group raw files by destination channel FIRST. NECIS event archives can split
        # one station-channel into MULTIPLE segment files (e.g. -120..+301 s covering
        # the event, plus a +300..+350 s tail); the old per-file loop wrote them all to
        # the SAME destination, so the LAST segment silently overwrote the one holding
        # the event -- healthy stations then "disappeared" downstream (observed on
        # 2026_Haenam event 20260815183944: 15 stations reduced to +300 s fragments).
        # Now: of the candidate segments, keep the one that covers the origin; if none
        # does, keep the longest.
        by_chan = {}
        for src in glob(os.path.join(raw_dir, pat)):
            net, code, chan = stations.parse_sac_name(cfg, os.path.basename(src), layout=layout)
            if sensor_of.get(code) != sensor:
                continue
            by_chan.setdefault((net, code, chan), []).append(src)
        for (net, code, chan), srcs in by_chan.items():
            if len(srcs) > 1:
                def _score(f):
                    st = read(f, headonly=True)[0].stats
                    covers = st.starttime <= o and st.endtime > o
                    return (covers, float(st.endtime - st.starttime))
                srcs = sorted(srcs, key=_score)
                print(f"  [gather] {code}.{chan}: {len(srcs)} segments -> using "
                      f"{os.path.basename(srcs[-1])}")
            tr = read(srcs[-1])[0]
            s = tr.stats.sac
            s.evla, s.evlo = ev["lat"], ev["lon"]
            s.stla, s.stlo = coord_of[code]
            s.nzyear, s.nzjday = o.year, o.julday
            s.nzhour, s.nzmin, s.nzsec = o.hour, o.minute, o.second
            dst = os.path.join(out_dir, f"{ev['event_id']}.{net}.{code}.{chan}.sac")
            tr.write(dst, format="SAC")
            written.append(dst)
    return written


def run_waveforms(cfg, used_table=None, events=None) -> dict:
    """Gather every catalog event (or a subset). Returns {event_id: n_files}."""
    if used_table is None:
        used_table = pd.read_csv(config.used_stations_csv(cfg))
    catalog = load_catalog(cfg)
    if events is not None:
        catalog = [e for e in catalog if e["event_id"] in set(events)]
    return {e["event_id"]: len(gather_event(cfg, used_table, e)) for e in catalog}
