"""
Lightweight matplotlib plots for monitoring the pipeline from JupyterLab.

Every function takes a ClusterConfig and returns a matplotlib Figure, so the same
calls work in a notebook (inline) and in scripts. Kept dependency-light (matplotlib
+ obspy only; no PyGMT) so the monitoring notebooks run anywhere the pipeline does.
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime

from pipeline import config
from pipeline.core import sumio, waveforms


def map_catalog(cfg, velmodel="kim1983", source="sum", ax=None):
    """Epicenter map coloured by depth, with the used stations. source = 'sum'|'reloc'."""
    if source == "sum":
        df = sumio.read_sum(config.sum_file(cfg, velmodel))
    else:
        df = sumio.read_reloc(config.dtct_dir(cfg) + "/hypoDD.reloc")
    sta = pd.read_csv(config.used_stations_csv(cfg))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.scatter(sta.Longitude, sta.Latitude, marker="^", s=40, c="0.6",
               edgecolor="k", label=f"stations ({len(sta)})", zorder=2)
    sc = ax.scatter(df.lon, df.lat, c=df.depth, s=60, cmap="viridis_r",
                    edgecolor="k", zorder=3)
    plt.colorbar(sc, ax=ax, label="depth (km)", shrink=0.8)
    ax.scatter(*cfg.epicenter[::-1], marker="*", s=300, c="red",
               edgecolor="k", zorder=4, label="cluster center")
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"{cfg.region} — {len(df)} events ({source}:{velmodel})")
    ax.legend(loc="best", fontsize=8); ax.set_aspect("equal", "datalim")
    return ax.figure


def depth_sections(cfg, velmodel="kim1983", source="sum"):
    """Lon-depth and lat-depth cross-sections."""
    df = (sumio.read_sum(config.sum_file(cfg, velmodel)) if source == "sum"
          else sumio.read_reloc(config.dtct_dir(cfg) + "/hypoDD.reloc"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    a1.scatter(df.lon, df.depth, s=40, c="steelblue", edgecolor="k")
    a1.set(xlabel="longitude", ylabel="depth (km)", title=f"{cfg.region} lon-depth")
    a1.invert_yaxis()
    a2.scatter(df.lat, df.depth, s=40, c="steelblue", edgecolor="k")
    a2.set(xlabel="latitude", ylabel="depth (km)", title="lat-depth")
    a2.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_picks(cfg, event_id, station=None, comp="Z", pre=5, post=25):
    """Vertical-component waveform with PhaseNet P (red) / S (blue) picks for one
    station (default: the station with the most picks for the event)."""
    picks = pd.read_csv(config.picks_csv(cfg, event_id))
    if station is None:
        station = picks["Station"].value_counts().index[0]
    sp = picks[picks.Station == station]
    wf = config.event_wf_dir(cfg, event_id)
    sensor = pd.read_csv(config.used_stations_csv(cfg)).set_index("Code").loc[station, "Sensor"]
    f = f"{wf}/{event_id}.*.{station}.{sensor}{comp}.sac"
    matches = glob.glob(f)
    fig, ax = plt.subplots(figsize=(11, 3), dpi=110)
    if not matches:
        ax.set_title(f"{station}: no {comp} trace"); return fig
    tr = read(matches[0])[0].detrend("demean").filter("highpass", freq=1.0)
    t0 = UTCDateTime(sp.iloc[0]["Time"]) if len(sp) else tr.stats.starttime
    w = tr.slice(t0 - pre, t0 + post)
    ax.plot(w.times(), w.data, "k", lw=0.5)
    for _, r in sp.iterrows():
        pt = UTCDateTime(r["Time"]) - w.stats.starttime
        c = "r" if r["Phase"] == "P" else "b"
        ax.axvline(pt, color=c, lw=1.5)
        ax.text(pt, ax.get_ylim()[1] * 0.8, f"{r['Phase']} {r['Probability']}", color=c, fontsize=8)
    ax.set(xlabel="time (s)", title=f"{cfg.region} {event_id} — {station}.{sensor}{comp} (red=P, blue=S)")
    fig.tight_layout()
    return fig


def cumulative_events(cfg, velmodel="kim1983"):
    """Cumulative event count over time (sanity / sequence overview)."""
    df = sumio.read_sum(config.sum_file(cfg, velmodel)).sort_values("time")
    t = [u.datetime for u in df.time]
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=110)
    ax.step(t, range(1, len(t) + 1), where="post")
    ax.set(xlabel="time", ylabel="cumulative events",
           title=f"{cfg.region} — {len(df)} located events")
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def plot_3c(cfg, event_id, station=None, pre=5, post=25):
    """Z/N/E waveforms (one event/station) with P (red, sac.a) / S (blue, sac.t0) marks.

    A quick check that all three components landed and the picks line up. Default station:
    the one with the most picks for the event."""
    sta = pd.read_csv(config.used_stations_csv(cfg)).set_index("Code")
    wf = config.event_wf_dir(cfg, event_id)
    if station is None:                         # default station: most-picked, else any with a Z SAC
        pc = config.picks_csv(cfg, event_id)
        if os.path.exists(pc):
            station = pd.read_csv(pc)["Station"].value_counts().index[0]
        else:
            zs = glob.glob(f"{wf}/{event_id}.*Z.sac")
            station = os.path.basename(zs[0]).split(".")[2] if zs else None
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), dpi=110, sharex=True)
    if station is None or station not in sta.index:
        axes[0].set_title(f"{cfg.region} {event_id}: no station data"); return fig
    sensor = sta.loc[station, "Sensor"]
    for ax, comp in zip(axes, ("Z", "N", "E")):
        m = glob.glob(f"{wf}/{event_id}.*.{station}.{sensor}{comp}.sac")
        if not m:
            ax.set_ylabel(f"{comp}: (none)"); continue
        tr = read(m[0])[0]
        s = tr.stats.sac
        ref = tr.stats.starttime - s.b
        ptime = ref + s.a if s.get("a", -12345.0) != -12345.0 else ref
        w = tr.copy().detrend("demean").slice(ptime - pre, ptime + post)
        ax.plot(w.times() - pre, w.data, "k", lw=0.5)
        for hdr, col in (("a", "r"), ("t0", "b")):
            v = s.get(hdr, -12345.0)
            if v != -12345.0:
                ax.axvline((ref + v) - ptime, color=col, lw=1.3)
        ax.set_ylabel(f"{sensor}{comp}")
    axes[0].set_title(f"{cfg.region} {event_id} — {station} (red=P, blue=S; t=0 at P)")
    axes[-1].set_xlabel("time from P (s)")
    fig.tight_layout()
    return fig


def cc_histogram(cfg, threshold=0.7):
    """Histogram of dt.cc cross-correlation coefficients (P vs S) in dt.cc_0.7_combined."""
    from pipeline.regression.compare import _parse_dtcc
    path = os.path.join(config.dtcc_dir(cfg), "dt.cc_0.7_combined")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    if not os.path.exists(path):
        ax.set_title("no dt.cc_0.7_combined (run xcorr first)"); return fig
    d = _parse_dtcc(path)                       # {(id1,id2,sta,phase): (delay, cc)}
    ccs = {"P": [], "S": []}
    for (_, _, _, phase), (_, cc) in d.items():
        ccs.setdefault(phase, []).append(cc)
    bins = np.linspace(threshold, 1.0, 25)
    for phase, col in (("P", "tab:blue"), ("S", "tab:orange")):
        if ccs.get(phase):
            ax.hist(ccs[phase], bins=bins, alpha=0.6, color=col,
                    label=f"{phase} (n={len(ccs[phase])})")
    ax.axvline(threshold, color="k", ls="--", lw=0.8)
    ax.set(xlabel="cc coefficient", ylabel="count",
           title=f"{cfg.region} — dt.cc coefficients (>= {threshold})")
    ax.legend(); fig.tight_layout()
    return fig


def compare_epicenters(cfg, velmodel="kim1983", variant="default"):
    """Side-by-side epicenter maps: dt.ct (left) vs dt.cc (right) HypoDD relocations."""
    ct = sumio.read_reloc(os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"))
    cc_path = os.path.join(config.dtcc_dir(cfg),
                           "hypoDD.reloc" if variant == "default" else f"{variant}/hypoDD.reloc")
    cc = sumio.read_reloc(cc_path)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    for ax, df, lab in ((a1, ct, "dt.ct"), (a2, cc, f"dt.cc:{variant}")):
        if len(df):
            sc = ax.scatter(df.lon, df.lat, c=df.depth, s=60, cmap="viridis_r", edgecolor="k")
            plt.colorbar(sc, ax=ax, label="depth (km)", shrink=0.8)
        ax.set(xlabel="longitude", ylabel="latitude", title=f"{cfg.region} — {lab} ({len(df)} ev)")
        ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    return fig
