"""
Station misorientation lookup + per-event tables (ports `6.Make_Misorientation_Table_
for_Used_Stations.ipynb`).

KMA sensor orientations drift over time; PCA/MinT analyses estimate each station's
misorientation angle per time interval, tabulated by sensor class in
`station_table/{bb,acc,sp}_PCA_MinT_*.csv` (columns: station, network, start, end,
median_PCA, ...). For an event's origin DATE we look up each used station's angle (the
interval covering that date) and attach it as a `Misorientation` column — the input that
`core.rotation` uses to correct N/E before rotating to radial/transverse.

All four clusters carry the same three tables (bb=HH, acc=HG, sp=EL) under their own
station_table/. Outputs go under runs/<cluster>/station_table/.
"""
from __future__ import annotations

import os
from glob import glob

import pandas as pd

from pipeline import config
from pipeline.core import waveforms

_TABLE_GLOB = {"HH": "bb_PCA_MinT_*.csv", "HG": "acc_PCA_MinT_*.csv", "EL": "sp_PCA_MinT_*.csv"}


def _load_tables(cfg) -> dict:
    """{sensor: DataFrame} of the per-sensor PCA misorientation tables (None if absent)."""
    d = os.path.join(cfg.src_root, "station_table")
    out = {}
    for sensor, pat in _TABLE_GLOB.items():
        fs = sorted(glob(os.path.join(d, pat)))
        out[sensor] = pd.read_csv(fs[0]) if fs else None
    return out


def find_angle(tables, sensor, code, origin_date):
    """median_PCA for (sensor, station code) on `origin_date` ('YYYY-MM-DD'); the row
    whose [start, end] interval covers the date. None if no table/row matches."""
    t = tables.get(sensor)
    if t is None:
        return None
    sub = t[t["station"] == code].reset_index(drop=True)
    for i in range(len(sub)):
        if str(sub["start"][i]) <= origin_date <= str(sub["end"][i]):
            return float(sub["median_PCA"][i])
    return None


def misorientation_for_event(cfg, used_table, origin_date, tables=None) -> dict:
    """{code: angle|None} for the used stations on `origin_date`."""
    tables = tables or _load_tables(cfg)
    return {r.Code: find_angle(tables, r.Sensor, r.Code, origin_date)
            for r in used_table.itertuples()}


def run_misorientation(cfg, write=True) -> dict:
    """Write a per-event `used_stations_100km_event<i>.csv` (Misorientation column added),
    mirroring the baseline notebook. Returns {event_id: n_stations_with_angle}."""
    used = pd.read_csv(config.used_stations_csv(cfg))
    tables = _load_tables(cfg)
    out_dir = config.assert_writable(config.station_table_dir(cfg))
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for i, ev in enumerate(waveforms.load_catalog(cfg), start=1):
        date = ev["origin"].strftime("%Y-%m-%d")
        ang = misorientation_for_event(cfg, used, date, tables)
        t = used.copy()
        t["Misorientation"] = t["Code"].map(ang)
        if write:
            t.to_csv(os.path.join(out_dir, f"used_stations_100km_event{i}.csv"), index=False)
        counts[ev["event_id"]] = int(t["Misorientation"].notna().sum())
    return counts
