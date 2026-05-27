"""
ZRT rotation (ports `4.ZRT_Rotation.ipynb`).

For each event/station the horizontal N/E are rotated to radial/transverse using the
event->station backazimuth corrected by the station misorientation:

    ba = (baz - mis) % 360
    r  = -e*sin(ba) - n*cos(ba)
    t  = -e*cos(ba) + n*sin(ba)        (the baseline's Aki&Richards-style convention)

The event/station coordinates and origin come from the gathered SAC headers (evla/evlo,
stla/stlo, starttime-b), so this works per-event without re-reading the catalog. Traces
are demeaned/tapered and sliced to [origin-pre, origin+post] (default 50/200 s) before
rotation. Output Z/R/T SAC go under runs/<cluster>/rotated/<event_id>/.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth

from pipeline import config
from pipeline.core import waveforms, misorientation


def rotate_ne_rt(n, e, ba):
    ba = np.radians(ba)
    return -e * np.sin(ba) - n * np.cos(ba), -e * np.cos(ba) + n * np.sin(ba)


def rotate_event(cfg, ev, sensor_of, tables, pre=50.0, post=200.0) -> int:
    """Rotate one event's stations; write Z/R/T. Returns the number of stations rotated."""
    eid = ev["event_id"]
    wf = config.event_wf_dir(cfg, eid)
    out = config.assert_writable(os.path.join(config.run_root(cfg), "rotated", eid))
    date = ev["origin"].strftime("%Y-%m-%d")
    n_ok = 0
    for z in sorted(glob(os.path.join(wf, f"{eid}.*Z.sac"))):
        try:
            p = os.path.basename(z).split(".")
            net, code, chan = p[1], p[2], p[3]
            sensor = chan[:2]
            trZ = read(z)[0]
            trN = read(z.replace(f"{sensor}Z.sac", f"{sensor}N.sac"))[0]
            trE = read(z.replace(f"{sensor}Z.sac", f"{sensor}E.sac"))[0]
            s = trZ.stats.sac
            baz = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[2]
            mis = misorientation.find_angle(tables, sensor_of.get(code, sensor), code, date) or 0.0
            origin = trZ.stats.starttime - s.b

            def _cut(tr):
                return (tr.copy().detrend("demean").taper(0.05)
                        .slice(origin - pre, origin + post).detrend("demean").taper(0.05))
            cz, cn, ce = _cut(trZ), _cut(trN), _cut(trE)
            r, t = rotate_ne_rt(cn.data, ce.data, (baz - mis) % 360)
            cn.data, ce.data = r, t
            cz.stats.channel, cn.stats.channel, ce.stats.channel = f"{sensor}Z", f"{sensor}R", f"{sensor}T"
            os.makedirs(out, exist_ok=True)
            cz.write(os.path.join(out, f"{eid}.{net}.{code}.{sensor}Z.sac"), format="SAC")
            cn.write(os.path.join(out, f"{eid}.{net}.{code}.{sensor}R.sac"), format="SAC")
            ce.write(os.path.join(out, f"{eid}.{net}.{code}.{sensor}T.sac"), format="SAC")
            n_ok += 1
        except Exception as exc:                            # noqa: BLE001 (notebook skip)
            print(f"[rotate] {eid} {os.path.basename(z)}: {exc}")
    return n_ok


def run_rotation(cfg, pre=50.0, post=200.0) -> dict:
    """Rotate every event's used stations to Z/R/T. Returns {event_id: n_rotated}."""
    used = pd.read_csv(config.used_stations_csv(cfg))
    sensor_of = dict(zip(used["Code"], used["Sensor"]))
    tables = misorientation._load_tables(cfg)
    return {ev["event_id"]: rotate_event(cfg, ev, sensor_of, tables, pre, post)
            for ev in waveforms.load_catalog(cfg)}
