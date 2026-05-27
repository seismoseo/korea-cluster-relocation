"""
Waveform-similarity QC (ports `7/8.Waveform_similarity_analysis_*.ipynb`).

At a fixed reference station, cross-correlate every event's P (Z) or S (N) phase window
against a template event (default the largest magnitude). High, stable NCC across the
sequence => a repeating source/path (a tight cluster, as expected for these dense swarms);
NCC drops or amplitude/time trends flag waveform changes. Returns a tidy DataFrame for the
QC notebook to plot NCC / amplitude vs magnitude and vs time.

This is a diagnostic (not a relocation input and not part of regression): the original
clusters Gwangyang/Kimcheon/Jangsung ran it; the Gyeongju pilot did not.
"""
from __future__ import annotations

import os
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max

from pipeline import config
from pipeline.core import waveforms

UNDEF = -12345.0


def _ref_station(cfg, comp="Z"):
    """The station appearing (with `comp`) in the most events — a robust similarity ref."""
    c = Counter()
    for ev in waveforms.load_catalog(cfg):
        for f in glob(os.path.join(config.event_wf_dir(cfg, ev["event_id"]), f"*{comp}.sac")):
            c[os.path.basename(f).split(".")[2]] += 1
    return c.most_common(1)[0][0] if c else None


def similarity(cfg, station=None, comp="Z", template_event=None,
               pre=0.5, post=3.5, highpass=1.0, rms_win=0.3, max_shift_s=0.5) -> pd.DataFrame:
    """NCC of each event's phase window vs a template at one station.

    Returns columns: event_id, time, mag, ncc, shift_s, p_rms (sorted by time). The chosen
    station + template event are stored on df.attrs."""
    used = pd.read_csv(config.used_stations_csv(cfg))
    sensor_of = dict(zip(used["Code"], used["Sensor"]))
    station = station or _ref_station(cfg, comp)
    cat = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
    hdr = "a" if comp == "Z" else "t0"

    def _win(eid):
        sensor = sensor_of.get(station)
        fs = glob(os.path.join(config.event_wf_dir(cfg, eid),
                               f"{eid}.*.{station}.{sensor}{comp}.sac"))
        if not fs:
            return None, None
        tr = read(fs[0])[0]
        pk = tr.stats.sac.get(hdr, UNDEF)
        if pk == UNDEF:
            return None, None
        ptime = tr.stats.starttime - tr.stats.sac.b + pk
        sl = tr.copy().detrend("demean").taper(0.05)
        if highpass:
            sl = sl.filter("highpass", freq=highpass, corners=6, zerophase=True)
        win = sl.slice(ptime - pre, ptime + post).detrend("demean").normalize()
        rms = float(np.sqrt(np.mean(
            tr.copy().detrend("demean").slice(ptime, ptime + rms_win).data ** 2)))
        return win, rms

    slices = {e: _win(e) for e in sorted(cat)}
    slices = {e: v for e, v in slices.items() if v[0] is not None}
    if not slices:
        return pd.DataFrame(columns=["event_id", "time", "mag", "ncc", "shift_s", "p_rms"])
    template_event = template_event or max(slices, key=lambda e: cat[e]["mag"])
    templ = slices[template_event][0]
    shift_samp = int(max_shift_s * templ.stats.sampling_rate)

    rows = []
    for e, (win, rms) in slices.items():
        lag, coeff = xcorr_max(correlate(templ, win, shift_samp), abs_max=False)
        rows.append(dict(event_id=e, time=cat[e]["origin"].datetime, mag=cat[e]["mag"],
                         ncc=round(float(coeff), 4),
                         shift_s=round(lag / templ.stats.sampling_rate, 4),
                         p_rms=rms))
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df.attrs.update(station=station, template=template_event, comp=comp)
    return df
