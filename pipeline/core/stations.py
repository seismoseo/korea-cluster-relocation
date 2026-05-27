"""
Stage 0 + 1 — station selection (ports notebooks `0.Select_nearby_stations` and
`1.Select_used_stations`).

  select_nearby(cfg)  -> stations in the master table within `radius_km` of the
                         epicenter                       -> kma_stations_100km.csv
  discover_used(cfg)  -> of those, the ones that actually have Z-component data in
                         the raw archive, one sensor each -> used_stations_100km.csv

Sensor precedence is HH > EL > HG (a station with HH uses HH; otherwise EL if it
has EL; otherwise HG), replicating the baseline notebook exactly.
"""
from __future__ import annotations

import os
from glob import glob

import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

from pipeline import config


# ---------------------------------------------------------------- raw archive
def raw_archive_root(cfg):
    """Directory holding per-event raw waveforms for the cluster's backend."""
    if cfg.wf_source == "kma_archive":
        return os.path.join(cfg.src_root, "kma_waveforms")
    if cfg.wf_source == "stp_sac":
        return cfg.stp_sac_root
    raise ValueError(f"unknown wf_source {cfg.wf_source!r}")


def _event_dirs(cfg):
    return sorted(glob(os.path.join(raw_archive_root(cfg), "20*")))


def parse_sac_name(cfg, basename):
    """(network, code, channel) from a raw SAC filename, per backend.

    kma_archive: KS.AMD..HHZ.D.2021.201.161224.SAC      -> ([0]=KS, [1]=AMD, [3]=HHZ)
    stp_sac    : 20170423131909.KS.CIGB.HGE.sac         -> ([1]=KS, [2]=CIGB, [3]=HGE)
    """
    p = basename.split(".")
    if cfg.wf_source == "stp_sac":
        return p[1], p[2], p[3]
    return p[0], p[1], p[3]


def wf_glob(cfg, sensor, comp):
    """Per-backend glob (relative to a raw event dir) for one sensor + component.

    kma_archive flattens components in the event dir; stp_sac nests them in HH/HG/EL
    subdirs. The pattern dict is `kma_archive_glob` or `stp_sac_glob` accordingly."""
    g = cfg.stp_sac_glob if cfg.wf_source == "stp_sac" else cfg.kma_archive_glob
    return g[sensor].format(comp=comp)


# ------------------------------------------------------------------- stage 0
def select_nearby(cfg) -> pd.DataFrame:
    """Stations from the master table(s) within `radius_km` of the epicenter."""
    master = pd.concat([pd.read_csv(p) for p in cfg.station_master_csvs],
                       ignore_index=True)
    evla, evlo = cfg.epicenter
    dist_km = master.apply(
        lambda r: gps2dist_azimuth(evla, evlo, r["Latitude"], r["Longitude"])[0] / 1000.0,
        axis=1,
    )
    return master[dist_km <= cfg.radius_km].reset_index(drop=True)


# ------------------------------------------------------------------- stage 1
def discover_used(cfg, nearby: pd.DataFrame) -> pd.DataFrame:
    """Restrict `nearby` to stations with Z-component data in the raw archive and
    assign one sensor per station (precedence HH > EL > HG)."""
    codes = set(nearby["Code"])
    avail = {s: set() for s in cfg.sensor_priority}
    for ed in _event_dirs(cfg):
        for sensor in cfg.sensor_priority:
            for f in glob(os.path.join(ed, wf_glob(cfg, sensor, "Z"))):
                c = parse_sac_name(cfg, os.path.basename(f))[1]
                if c in codes:
                    avail[sensor].add(c)

    HH = avail.get("HH", set())
    HG = avail.get("HG", set())
    EL = avail.get("EL", set())
    el_only = {c for c in EL if c not in HH}                 # EL not in HH
    hg_only = {c for c in HG if c not in HH and c not in EL}  # HG not in HH/EL
    used = HH | el_only | hg_only

    def _label(code):
        if code in HH:
            return "HH"
        if code in hg_only:
            return "HG"
        return "EL"

    table = nearby[nearby["Code"].isin(used)].reset_index(drop=True)
    table["Sensor"] = table["Code"].map(_label)
    return table


# --------------------------------------------------------------- orchestration
def run_stations(cfg, write=True) -> pd.DataFrame:
    """Run stage 0 + 1 and (optionally) write both CSVs under the run root."""
    nearby = select_nearby(cfg)
    used = discover_used(cfg, nearby)
    if write:
        out_dir = config.assert_writable(config.station_table_dir(cfg))
        os.makedirs(out_dir, exist_ok=True)
        nearby.to_csv(config.nearby_stations_csv(cfg), index=False)
        used.to_csv(config.used_stations_csv(cfg), index=False)
    return used
