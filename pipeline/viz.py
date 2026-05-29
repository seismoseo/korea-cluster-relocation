"""
Lightweight matplotlib plots for monitoring the pipeline from JupyterLab.

Every function takes a ClusterConfig and returns a matplotlib Figure, so the same
calls work in a notebook (inline) and in scripts. Kept dependency-light (matplotlib
+ obspy only; no PyGMT) so the monitoring notebooks run anywhere the pipeline does.
`plot_3d_plane` is the one exception: it returns an interactive plotly Figure and
imports plotly lazily, so plotly stays an optional dependency.
"""
from __future__ import annotations

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime

from pipeline import config
from pipeline.core import sumio, waveforms


def _use_helvetica():
    """Use Helvetica for every plot's text if the fonts are available (`config.HELVETICA_DIR`,
    env-overridable via $HELVETICA_DIR); otherwise leave the matplotlib default in place so a
    public clone without the fonts still renders. Runs once at import.

    Helvetica lacks some glyphs we annotate with (the up/down arrows ↑/↓, etc.), so we set a font
    *fallback chain* `["Helvetica", "DejaVu Sans"]`: text renders in Helvetica, and any glyph it is
    missing falls back to DejaVu Sans (matplotlib does per-glyph fallback across the family list).
    This fixes the 'Glyph missing from font' warnings at the source — no glyph is left unrenderable."""
    import matplotlib.font_manager as fm
    font_dir = getattr(config, "HELVETICA_DIR", None)
    try:
        if font_dir and os.path.isdir(font_dir):
            for fpath in fm.findSystemFonts(font_dir):
                fm.fontManager.addfont(fpath)
        names = {f.name for f in fm.fontManager.ttflist}
        if "Helvetica" in names:
            mpl.rcParams["font.family"] = ["Helvetica", "DejaVu Sans"]   # primary + glyph fallback
    except Exception:
        pass                                   # never let font setup break plotting
    mpl.rcParams["axes.unicode_minus"] = False


_use_helvetica()


def _reloc_path(cfg):
    """The headline HypoDD relocation: dt.cc if it exists (the high-end product — far smaller
    errors), else the dt.ct catalog relocation. Returns (path, branch_label)."""
    cc = os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc")
    if os.path.exists(cc):
        return cc, "dt.cc"
    return os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"), "dt.ct"


def _format_lonlat(ax):
    """Plain decimal-degree tick labels on a lon/lat axis — no offset / exponential notation
    (matplotlib's default for tight ~0.05° ranges shows an ugly '+1.28e2' offset)."""
    from matplotlib.ticker import ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)
    ax.ticklabel_format(useOffset=False, style="plain", axis="both")
    for lab in ax.get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")


def _load_bootstrap(cfg, branch):
    """Load the bootstrap per-replica samples for a branch ('dt.cc'->dtcc, 'dt.ct'->dtct):
    {event_id: aligned X/Y/Z samples [n, 3] in metres (Z +down)}, or None if not computed. The 95%
    error bar in any frame is the 2.5–97.5 percentile half-width of the samples projected onto that
    axis (`_pct_hw`). See `hypodd.bootstrap_relocation` (writes `bootstrap_samples.npz`)."""
    bdir = config.dtcc_dir(cfg) if "cc" in branch else config.dtct_dir(cfg)
    p = os.path.join(bdir, "bootstrap_samples.npz")
    if not os.path.exists(p):
        return None
    data = np.load(p)["data"]
    if data.size == 0:
        return None
    out = {}
    for row in data:
        out.setdefault(int(row[0]), []).append(row[1:4])
    return {e: np.asarray(v, float) for e, v in out.items()} or None


def _pct_hw(samples, v=None):
    """95% half-width = (P97.5 − P2.5) / 2 of the bootstrap samples. With `v` (unit vector), of the
    samples projected onto `v` (a scalar); otherwise per-axis (a 3-vector). All in the sample units (m)."""
    a = samples if v is None else samples @ v
    return (np.percentile(a, 97.5, axis=0) - np.percentile(a, 2.5, axis=0)) / 2.0


def map_catalog(cfg, velmodel="kim1983", source="sum", ax=None):
    """Epicenter map coloured by depth, with the used stations. source = 'sum'|'reloc'.
    For source='reloc' the headline dt.cc relocation is shown (dt.ct fallback)."""
    branch = velmodel
    if source == "sum":
        df = sumio.read_sum(config.sum_file(cfg, velmodel))
    else:
        path, branch = _reloc_path(cfg)
        df = sumio.read_reloc(path)
    sta = pd.read_csv(config.used_stations_csv(cfg))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.scatter(sta.Longitude, sta.Latitude, marker="^", s=40, c="0.6",
               edgecolor="k", label=f"Stations ({len(sta)})", zorder=2)
    sc = ax.scatter(df.lon, df.lat, c=df.depth, s=60, cmap="viridis_r",
                    edgecolor="k", zorder=3)
    plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)
    boot = _load_bootstrap(cfg, branch) if source == "reloc" else None
    if boot:                                     # 95% bootstrap X/Y error bars (percentile, m -> deg)
        for r in df.itertuples():
            if int(r.id) not in boot:
                continue
            hw = _pct_hw(boot[int(r.id)])        # (E, N, Z) metres
            ax.errorbar(r.lon, r.lat, xerr=hw[0] / (111320.0 * np.cos(np.deg2rad(r.lat))),
                        yerr=hw[1] / 111320.0, fmt="none", ecolor="0.4",
                        elinewidth=0.7, capsize=2, zorder=2)
    ax.scatter(*cfg.epicenter[::-1], marker="*", s=300, c="red",
               edgecolor="k", zorder=4, label="Cluster center")
    bl = " + 95% bootstrap" if boot else ""
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{cfg.region} — {len(df)} events ({source}:{branch}{bl})")
    ax.legend(loc="best", fontsize=8); ax.set_aspect("equal", "datalim")
    _format_lonlat(ax)
    return ax.figure


def depth_sections(cfg, velmodel="kim1983", source="sum"):
    """Lon-depth and lat-depth cross-sections (source='reloc' uses the headline dt.cc relocation,
    with 95% bootstrap depth/horizontal error bars when the bootstrap cache exists)."""
    branch = "dt.cc"
    if source == "sum":
        df = sumio.read_sum(config.sum_file(cfg, velmodel))
    else:
        path, branch = _reloc_path(cfg)
        df = sumio.read_reloc(path)
    boot = _load_bootstrap(cfg, branch) if source == "reloc" else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    a1.scatter(df.lon, df.depth, s=40, c="steelblue", edgecolor="k", zorder=3)
    a1.set(xlabel="Longitude", ylabel="Depth (km)", title=f"{cfg.region} longitude-depth")
    a2.scatter(df.lat, df.depth, s=40, c="steelblue", edgecolor="k", zorder=3)
    a2.set(xlabel="Latitude", ylabel="Depth (km)", title="Latitude-depth")
    if boot:                                     # 95% bootstrap bars: depth (Z) + horizontal (X/Y)
        for r in df.itertuples():
            if int(r.id) not in boot:
                continue
            hw = _pct_hw(boot[int(r.id)])        # (E, N, Z) metres
            a1.errorbar(r.lon, r.depth, xerr=hw[0] / (111320.0 * np.cos(np.deg2rad(r.lat))),
                        yerr=hw[2] / 1000.0, fmt="none", ecolor="0.4", elinewidth=0.7, capsize=2, zorder=2)
            a2.errorbar(r.lat, r.depth, xerr=hw[1] / 111320.0,
                        yerr=hw[2] / 1000.0, fmt="none", ecolor="0.4", elinewidth=0.7, capsize=2, zorder=2)
    a1.invert_yaxis(); a2.invert_yaxis()
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
        is_p = r["Phase"] == "P"
        c = "r" if is_p else "b"
        ax.axvline(pt, color=c, lw=1.5)
        lab = f"{r['Phase']} {r['Probability']}"
        pol = r.get("Polarity") if hasattr(r, "get") else (r["Polarity"] if "Polarity" in r else None)
        if is_p and pol is not None and pd.notna(pol) and pol != "":
            lab += f"  {'↑' if float(pol) >= 0 else '↓'}{float(pol):+.2f}"   # first-motion polarity
        ax.text(pt, ax.get_ylim()[1] * 0.8, lab, color=c, fontsize=8)
    ax.set(xlabel="Time (s)", title=f"{cfg.region} {event_id} — {station}.{sensor}{comp} (red=P, blue=S)")
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
    pol = None                                  # first-motion polarity (P) from the picks CSV
    pc = config.picks_csv(cfg, event_id)
    if os.path.exists(pc):
        pp = pd.read_csv(pc)
        pp = pp[(pp.Station == station) & (pp.Phase == "P")]
        if len(pp) and "Polarity" in pp.columns and pd.notna(pp.iloc[0]["Polarity"]):
            pol = float(pp.iloc[0]["Polarity"])
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
        if comp == "Z" and pol is not None:     # mark P first-motion polarity on the vertical
            ax.annotate(f"{'↑' if pol >= 0 else '↓'}{pol:+.2f}", xy=(0, 0.86),
                        xycoords=("data", "axes fraction"),
                        color=("r" if pol >= 0 else "b"), fontsize=10, fontweight="bold", ha="center")
        ax.set_ylabel(f"{sensor}{comp}")
    pol_txt = "" if pol is None else f"; P polarity {'↑ up' if pol >= 0 else '↓ down'} ({pol:+.2f})"
    axes[0].set_title(f"{cfg.region} {event_id} — {station} (red=P, blue=S; t=0 at P{pol_txt})")
    axes[-1].set_xlabel("Time from P (s)")
    fig.tight_layout()
    return fig


def plot_polarities(cfg, event_id, win=0.4, sort="azimuth", min_weight=0.0, max_stations=45):
    """First-motion record section: P-aligned vertical-component snippets (±`win` s around the P
    pick), one per station, sorted by source→station azimuth, coloured by the PhaseNet+ first-motion
    polarity (up = red, down = blue; |polarity| sets opacity). A ↑/↓ marker + station + azimuth +
    polarity are annotated at each trace. This is the up/down-vs-azimuth pattern that constrains the
    focal mechanism. Needs a phasenet_plus picks CSV (with the `Polarity` column)."""
    from obspy.geodetics.base import gps2dist_azimuth
    pc = config.picks_csv(cfg, event_id)
    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=120)
    if not os.path.exists(pc):
        ax.set_title(f"{cfg.region} {event_id}: no picks"); return fig
    picks = pd.read_csv(pc)
    if "Polarity" not in picks.columns:
        ax.set_title(f"{cfg.region} {event_id}: no polarity (needs a phasenet_plus picking run)")
        return fig
    P = picks[(picks.Phase == "P") & picks.Polarity.notna()].copy()
    P = P[P.Polarity.abs() >= min_weight]
    sta = pd.read_csv(config.used_stations_csv(cfg)).set_index("Code")
    wf = config.event_wf_dir(cfg, event_id)
    rows = []
    for _, r in P.iterrows():
        code = r.Station
        if code not in sta.index:
            continue
        m = glob.glob(f"{wf}/{event_id}.*.{code}.{sta.loc[code, 'Sensor']}Z.sac")
        if not m:
            continue
        tr = read(m[0])[0]
        s = tr.stats.sac
        try:
            az = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[1]
        except Exception:                       # noqa: BLE001
            az = np.nan
        pt = UTCDateTime(r.Time)
        w = tr.copy().detrend("demean").slice(pt - win, pt + win)
        if len(w.data) < 3:
            continue
        d = w.data.astype(float)
        d = d / (np.max(np.abs(d)) or 1.0)
        rows.append(dict(code=code, az=float(az), pol=float(r.Polarity),
                         t=w.times() - (pt - w.stats.starttime), d=d))
    if not rows:
        ax.set_title(f"{cfg.region} {event_id}: no usable first motions"); return fig
    rows = sorted(rows, key=lambda x: (x["az"] if sort == "azimuth" else x["code"]))[:max_stations]
    for i, rr in enumerate(rows):
        up = rr["pol"] >= 0
        col = "tab:red" if up else "tab:blue"
        ax.axhline(i, color="0.9", lw=0.3, zorder=0)
        ax.plot(rr["t"], rr["d"] * 0.45 + i, color=col, lw=0.7,
                alpha=0.35 + 0.6 * min(abs(rr["pol"]), 1.0))
        ax.annotate("↑" if up else "↓", xy=(0, i), color=col, fontsize=11,
                    fontweight="bold", ha="center", va="center")
        ax.text(-win * 1.03, i, rr["code"], ha="right", va="center", fontsize=7)
        ax.text(win * 1.03, i, f"{rr['az']:.0f}° {rr['pol']:+.2f}", ha="left", va="center", fontsize=7)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set(xlabel="Time from P (s)", yticks=[], xlim=(-win * 1.35, win * 1.45), ylim=(-1, len(rows)),
           title=f"{cfg.region} {event_id} — P first-motion polarity "
                 f"(red = up, blue = down; sorted by azimuth, n={len(rows)})")
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


def _load_mechanisms(path):
    """Read mechanisms.csv and keep ONE row per event — SKHASH's **preferred** solution. A
    multi-solution event has several rows of the same quality; SKHASH plots (and we should report)
    the one with the highest `prob_mech`. Sorting by quality (A→D) then prob_mech (high→low) before
    `drop_duplicates` keeps that row, so the table, the obspy beachballs, and the reference plane all
    match the SKHASH beachball PNG. (Plain `drop_duplicates` kept an arbitrary solution.)"""
    m = pd.read_csv(path)
    by = ["quality"] + (["prob_mech"] if "prob_mech" in m.columns else [])
    asc = [True] + ([False] if "prob_mech" in m.columns else [])
    return m.sort_values(by, ascending=asc, kind="mergesort").drop_duplicates("event_id", keep="first")


def mechanism_table(cfg, velmodel=None):
    """Tidy focal-mechanism table (one row per event, preferred solution) for notebook display."""
    velmodel = velmodel or cfg.fm_velmodel
    path = config.fm_mech_csv(cfg, velmodel)
    if not os.path.exists(path):
        return pd.DataFrame()
    m = _load_mechanisms(path)
    cols = ["event_id", "quality", "strike", "dip", "rake", "fault_plane_uncertainty",
            "num_p_pol", "num_sp_ratios", "azimuthal_gap", "sta_distribution_ratio",
            "origin_depth_km"]
    return m[[c for c in cols if c in m.columns]].reset_index(drop=True)


def map_mechanisms(cfg, velmodel=None, quality_keep=("A", "B"), ax=None):
    """Locations + focal mechanisms together: located epicenters as depth-coloured dots, with the
    high-confidence (quality in `quality_keep`) beachballs offset on a ring around the cluster
    centroid (leader line to the true epicenter) so a tight cluster stays legible. obspy `beach()`.

    Reads `config.fm_mech_csv(cfg, velmodel)`; needs a phasenet_plus focal_mechanism run."""
    import matplotlib as mpl
    velmodel = velmodel or cfg.fm_velmodel
    path = config.fm_mech_csv(cfg, velmodel)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.5), dpi=120)
    if not os.path.exists(path):
        ax.set_title(f"{cfg.region} — no mechanisms.csv\n(run focal_mechanism with "
                     f"picker_weights='phasenet_plus')", fontsize=10)
        return ax.figure
    from obspy.imaging.beachball import beach
    m = _load_mechanisms(path)

    norm = mpl.colors.Normalize(vmin=float(m.origin_depth_km.min()),
                                vmax=float(m.origin_depth_km.max()))
    cmap = plt.get_cmap("viridis_r")
    sc = ax.scatter(m.origin_lon, m.origin_lat, c=m.origin_depth_km, cmap=cmap, norm=norm,
                    s=55, edgecolor="k", lw=0.5, zorder=4, label=f"Located events ({len(m)})")
    plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)

    keep = m[m.quality.isin(list(quality_keep))]
    disp = m.reset_index(drop=True)                     # all mechanisms; A/B bold, C/D faint
    clon, clat = float(m.origin_lon.mean()), float(m.origin_lat.mean())
    ext = max(m.origin_lon.max() - m.origin_lon.min(),
              m.origin_lat.max() - m.origin_lat.min(), 0.012)
    R = ext * 1.7                                       # ring radius from centroid
    n = max(len(disp), 1)
    bwidth = min(ext * 0.8, 2 * np.pi * R / n * 0.6)    # diameter; shrink if ring is crowded
    for i, r in disp.iterrows():
        is_hi = r.quality in quality_keep
        ang = 2 * np.pi * i / n + np.pi / 2
        bx, by = clon + R * np.cos(ang), clat + R * np.sin(ang)
        ax.plot([r.origin_lon, bx], [r.origin_lat, by], "-", color="0.6", lw=0.5, zorder=3)
        ax.add_collection(beach((r.strike, r.dip, r.rake), xy=(bx, by), width=bwidth,
                                facecolor=cmap(norm(r.origin_depth_km)), edgecolor="k",
                                linewidth=0.7 if is_hi else 0.4,
                                alpha=0.95 if is_hi else 0.45, zorder=5 if is_hi else 4))
        ax.text(bx, by + bwidth * 0.6, r.quality, ha="center", va="bottom", fontsize=8,
                fontweight="bold" if is_hi else "normal", zorder=6)
    pad = R + bwidth * 0.7 + ext * 0.15            # zoom to the cluster + beachball ring
    ax.set_xlim(clon - pad, clon + pad)
    ax.set_ylim(clat - pad, clat + pad)
    ax.set_aspect("equal", "box")
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{cfg.region} — locations + focal mechanisms "
                 f"({len(keep)} high-confidence [{'/'.join(quality_keep)}] / {len(m)} events, {velmodel})")
    ax.legend(loc="best", fontsize=8)
    _format_lonlat(ax)
    return ax.figure


def _fault_ref(cfg, velmodel=None):
    """Reference (strike, dip, rake, cuspid, magnitude) for the fault sections = the
    largest-magnitude high-confidence (A/B) mechanism; fallback = best-quality available.
    None if there are no mechanisms."""
    velmodel = velmodel or cfg.fm_velmodel
    path = config.fm_mech_csv(cfg, velmodel)
    if not os.path.exists(path):
        return None
    m = _load_mechanisms(path)
    hi = m[m.quality.isin(list(cfg.fm_quality_keep))]
    pool = hi if len(hi) else m
    if not len(pool):
        return None
    if "magnitude" in pool.columns and pool["magnitude"].notna().any() and pool["magnitude"].max() > 0:
        r = pool.sort_values("magnitude", ascending=False).iloc[0]
    else:
        r = pool.iloc[0]                                    # best quality (already sorted)
    return dict(strike=float(r.strike), dip=float(r.dip), rake=float(r.rake),
                cuspid=int(r.cuspid), quality=str(r.quality),
                mag=float(r.magnitude) if "magnitude" in r and pd.notna(r.magnitude) else float("nan"),
                event_id=str(r.event_id))


def _svd_normal(x, y, z):
    """Unit normal of the least-squares plane through points (x, y, z) — the smallest-singular-value
    (SVD) direction of the centred coordinates. Returned in the SAME coordinate frame as the inputs."""
    c = np.column_stack((np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)))
    c = c - c.mean(axis=0)
    n = np.linalg.svd(c)[2][-1]
    return n / (np.linalg.norm(n) or 1.0)


def _best_fit_plane(x, y, z):
    """Strike/dip (deg) of the best-fit plane through (x, y, z) via SVD — the data-driven fault
    orientation (ported from the originals' `calculate_strike_dip_svd`). x, y = HypoDD .reloc X, Y
    (East, North); **z = .reloc Z, which is positive DOWN** (Z tracks the `depth` column, +1000 m/km).
    Dip = the plane's angle from horizontal = arccos|n_vertical| (independent of the up/down sense);
    strike is the horizontal trend — a line, so its 180° sense is irrelevant to the along/across
    sections. Needs ≥ 3 points."""
    n = _svd_normal(x, y, z)
    n_e, n_n, n_d = n if n[2] >= 0 else -n
    dip = float(np.degrees(np.arccos(np.clip(abs(n_d), -1.0, 1.0))))
    strike = float((np.degrees(np.arctan2(n_n, -n_e)) + 360) % 360)
    return strike, dip


def fault_sections(cfg, velmodel=None, strike=None, dip=None, color_by="time"):
    """Relocated seismicity in fault coordinates — a 2×2 figure styled after the original dt.cc
    notebooks: (1) fault-plane map view, (2) along-strike depth section, (3) across-strike depth
    section (with the dashed fault-dip line), (4) fault-plane (along-dip) view.

    Orientation: explicit `strike`/`dip` win; otherwise the **best-fit plane of the relocated cloud**
    (SVD) — the data-driven fault. The focal mechanism is only overlaid for comparison (beachball +
    annotation), never forced as the frame, so under-constrained mechanisms can't distort the view.
    Centred on the largest-magnitude event. Markers are coloured by origin time (`coolwarm`), sized by
    magnitude, hollow with coloured edges. Reads the headline dt.cc relocation (dt.ct fallback)."""
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from obspy.imaging.beachball import beach

    velmodel = velmodel or cfg.fm_velmodel
    reloc, branch = _reloc_path(cfg)
    fig, axarr = plt.subplots(2, 2, figsize=(11, 10), dpi=130, constrained_layout=True)
    axes = axarr.ravel()
    if not os.path.exists(reloc):
        axes[0].set_title(f"{cfg.region}: no HypoDD reloc (run ph2dt→dtcc first)"); return fig
    d = sumio.read_reloc(reloc).reset_index(drop=True)
    if not len(d):
        axes[0].set_title(f"{cfg.region}: empty reloc"); return fig

    ref = _fault_ref(cfg, velmodel)          # focal mechanism — for overlay/comparison only

    # --- orientation: explicit args > SVD best-fit plane of the cloud > focal mechanism ---
    svd_strike = svd_dip = None
    if len(d) >= 3:
        svd_strike, svd_dip = _best_fit_plane(d.x, d.y, d.z)
    used_strike = strike if strike is not None else (
        svd_strike if svd_strike is not None else (ref["strike"] if ref else 0.0))
    used_dip = dip if dip is not None else (
        svd_dip if svd_dip is not None else (ref["dip"] if ref else 90.0))

    # --- centre on the largest-magnitude event (reference); fallback = cloud centroid ---
    refrow = d[d.id == ref["cuspid"]] if ref else d.iloc[0:0]
    if not len(refrow) and "mag" in d and d.mag.notna().any():
        refrow = d.loc[[d.mag.idxmax()]]
    x0, y0, z0 = (float(refrow.iloc[0].x), float(refrow.iloc[0].y), float(refrow.iloc[0].z)) \
        if len(refrow) else (float(d.x.mean()), float(d.y.mean()), float(d.z.mean()))

    rx, ry = (d.x - x0).to_numpy(), (d.y - y0).to_numpy()         # metres, relative to centre
    th = np.deg2rad(90.0 - used_strike)
    along = (rx * np.cos(th) + ry * np.sin(th)) / 1000.0          # km, +ve in strike azimuth
    across = (-rx * np.sin(th) + ry * np.cos(th)) / 1000.0        # km
    dep = (d.z.to_numpy() - z0) / 1000.0                          # km, +down (Z is positive down)
    along_dip = across * np.cos(np.deg2rad(used_dip)) + dep * np.sin(np.deg2rad(used_dip))   # km

    # --- 95% bootstrap error bars, rotated into the fault frame (percentile of projected samples) ---
    boot = _load_bootstrap(cfg, branch)
    sig_al = sig_ac = sig_dp = sig_ad = None
    if boot:
        ct, st, cdip, sdip = np.cos(th), np.sin(th), np.cos(np.deg2rad(used_dip)), np.sin(np.deg2rad(used_dip))
        v_al = np.array([ct, st, 0.0]); v_ac = np.array([-st, ct, 0.0])
        v_z = np.array([0.0, 0.0, 1.0]); v_ad = np.array([-st * cdip, ct * cdip, sdip])
        sig_al, sig_ac, sig_dp, sig_ad = (np.full(len(d), np.nan) for _ in range(4))
        for i, e in enumerate(d.id.astype(int)):
            if e in boot:
                s = boot[e]
                sig_al[i], sig_ac[i] = _pct_hw(s, v_al) / 1000.0, _pct_hw(s, v_ac) / 1000.0
                sig_dp[i], sig_ad[i] = _pct_hw(s, v_z) / 1000.0, _pct_hw(s, v_ad) / 1000.0

    # --- colour (origin time by default, as in the originals) + magnitude-scaled hollow markers ---
    mag = np.nan_to_num(d.mag.to_numpy(), nan=0.0)
    sz = np.clip(5.0 * np.exp(2.0 * mag), 25, 1500)
    if color_by == "time" and "time" in d and d.time.notna().any():
        cv = np.array(mdates.date2num([t.datetime for t in d.time]))
        norm = mcolors.Normalize(vmin=cv.min(), vmax=cv.max() if cv.max() > cv.min() else cv.min() + 1)
        cmap = plt.get_cmap("coolwarm"); cbar_label = "Origin time"
    elif color_by == "mag":
        cv = mag; norm = mcolors.Normalize(vmin=cv.min(), vmax=max(cv.max(), cv.min() + 0.1))
        cmap = plt.get_cmap("viridis"); cbar_label = "Magnitude"
    else:
        cv = d.depth.to_numpy(); norm = mcolors.Normalize(vmin=np.nanmin(cv), vmax=np.nanmax(cv))
        cmap = plt.get_cmap("viridis_r"); cbar_label = "Depth (km)"
    rgba = cmap(norm(cv))

    def _style(ax):
        ax.set_aspect("equal", "box"); ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_facecolor("#FAFAFA"); ax.tick_params(labelsize=11)

    L = (max(np.ptp(rx), np.ptp(ry)) / 1000.0) or 0.2            # km, cloud half-extent
    pad = 1.25 * L
    su, du = np.sin(np.deg2rad(used_strike)), np.cos(np.deg2rad(used_strike))
    # one symmetric square range for all section panels so equal-aspect panels pack uniformly
    R = 1.15 * max(np.nanmax(np.abs(along)), np.nanmax(np.abs(across)),
                   np.nanmax(np.abs(dep)), np.nanmax(np.abs(along_dip)), 1e-3)

    # panel 1 — fault-plane map view (relative E–N km), with fault lines + section labels
    ax = axes[0]
    ax.scatter(rx / 1000.0, ry / 1000.0, s=sz, facecolors="none", edgecolors=rgba,
               linewidth=1.8, zorder=4)
    ax.plot([-pad * su, pad * su], [-pad * du, pad * du], color="0.35", lw=1.1, ls="-", zorder=2)
    ax.plot([pad * du, -pad * du], [-pad * su, pad * su], color="0.35", lw=1.1, ls="--", zorder=2)
    if ref and not np.isnan(ref["rake"]):
        ax.add_collection(beach((ref["strike"], ref["dip"], ref["rake"]),
                                xy=(-0.78 * pad, 0.78 * pad), width=0.34 * pad,
                                facecolor="0.45", edgecolor="k", linewidth=0.8, zorder=5))
    for sgn, lab in ((1, "A'"), (-1, "A")):                      # along-strike ends
        ax.text(sgn * 0.97 * pad * su, sgn * 0.97 * pad * du, lab, fontsize=18, fontweight="bold",
                ha="center", va="center", zorder=6)
    for sgn, lab in ((1, "B"), (-1, "B'")):                      # across-strike ends
        ax.text(sgn * 0.97 * pad * du, -sgn * 0.97 * pad * su, lab, fontsize=18, fontweight="bold",
                ha="center", va="center", zorder=6)
    ax.set(xlim=(-pad, pad), ylim=(-pad, pad), xlabel="E (km)", ylabel="N (km)",
           title="Fault-plane map view"); _style(ax)

    # panel 2 — along-strike depth section (A–A')
    ax = axes[1]
    if boot is not None:
        ax.errorbar(along, dep, xerr=sig_al, yerr=sig_dp, fmt="none", ecolor="0.55",
                    elinewidth=0.6, capsize=1.5, zorder=3)
    ax.scatter(along, dep, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    ax.text(-0.92 * R, -0.88 * R, "A", fontsize=16, fontweight="bold")
    ax.text(0.86 * R, -0.88 * R, "A'", fontsize=16, fontweight="bold")
    ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="Along-strike distance (km)",
           ylabel="Depth rel. to reference (km)", title="Along-strike (A–A')")
    _style(ax); ax.invert_yaxis()

    # panel 3 — across-strike depth section (B–B'), with the dashed fault-dip line. Draw the actual
    # trace of the best-fit plane in this section (from the SVD normal, rotated into fault coords),
    # through the cloud centre — so the line dips the correct way and overlays the data.
    ax = axes[2]
    if boot is not None:
        ax.errorbar(across, dep, xerr=sig_ac, yerr=sig_dp, fmt="none", ecolor="0.55",
                    elinewidth=0.6, capsize=1.5, zorder=3)
    ax.scatter(across, dep, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    nrm = _svd_normal(d.x, d.y, d.z)                             # (E, N, down)
    n_ac = -nrm[0] * np.sin(th) + nrm[1] * np.cos(th)            # across-strike component
    ac0, dp0 = float(np.mean(across)), float(np.mean(dep))
    xx = np.linspace(-R, R, 50)
    if abs(nrm[2]) > 1e-6:                                       # depth = dp0 − (n_across/n_down)(across−ac0)
        ax.plot(xx, dp0 - (n_ac / nrm[2]) * (xx - ac0), color="k", lw=1.0, ls="--", zorder=1,
                label=f"Dip {used_dip:.0f}°")
    else:                                                        # vertical plane
        ax.axvline(ac0, color="k", lw=1.0, ls="--", zorder=1, label=f"Dip {used_dip:.0f}°")
    ax.text(-0.92 * R, -0.88 * R, "B", fontsize=16, fontweight="bold")
    ax.text(0.86 * R, -0.88 * R, "B'", fontsize=16, fontweight="bold")
    ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="Across-strike distance (km)",
           ylabel="Depth rel. to reference (km)", title="Across-strike (B–B')")
    _style(ax); ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)

    # panel 4 — fault-plane (along-dip) view
    ax = axes[3]
    if boot is not None:
        ax.errorbar(along, along_dip, xerr=sig_al, yerr=sig_ad, fmt="none", ecolor="0.55",
                    elinewidth=0.6, capsize=1.5, zorder=3)
    ax.scatter(along, along_dip, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    ax.text(-0.92 * R, -0.88 * R, "A", fontsize=16, fontweight="bold")
    ax.text(0.86 * R, -0.88 * R, "A'", fontsize=16, fontweight="bold")
    ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="Along-strike distance (km)",
           ylabel="Along-dip distance (km)", title="Fault-plane view (along-dip)")
    _style(ax); ax.invert_yaxis()

    # shared colour bar (origin time formatted as dates)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85)
    cbar.set_label(cbar_label)
    if color_by == "time":
        ticks = np.linspace(norm.vmin, norm.vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([mdates.num2date(t).strftime("%Y-%m-%d") for t in ticks])

    src = "manual" if strike is not None else ("best-fit plane" if svd_strike is not None else "mechanism")
    fmtxt = (f"; mechanism {ref['strike']:.0f}°/{ref['dip']:.0f}° ({ref['quality']})"
             if ref else "")
    btxt = "  (bars = 95% bootstrap)" if boot is not None else ""
    fig.suptitle(f"{cfg.region} — relocated seismicity in fault coordinates ({branch}){btxt}\n"
                 f"strike {used_strike:.0f}°, dip {used_dip:.0f}° [{src}]{fmtxt}", fontsize=13)
    return fig


def compare_epicenters(cfg, velmodel="kim1983", variant="default"):
    """Side-by-side epicenter maps: dt.ct (left) vs dt.cc (right) HypoDD relocations."""
    ct = sumio.read_reloc(os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"))
    cc_path = os.path.join(config.dtcc_dir(cfg),
                           "hypoDD.reloc" if variant == "default" else f"{variant}/hypoDD.reloc")
    cc = sumio.read_reloc(cc_path)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    for ax, df, lab, br in ((a1, ct, "dt.ct", "dt.ct"), (a2, cc, f"dt.cc:{variant}", "dt.cc")):
        if len(df):
            sc = ax.scatter(df.lon, df.lat, c=df.depth, s=60, cmap="viridis_r", edgecolor="k", zorder=3)
            plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)
            boot = _load_bootstrap(cfg, br) if variant == "default" else None
            for r in (df.itertuples() if boot else ()):
                if int(r.id) not in boot:
                    continue
                hw = _pct_hw(boot[int(r.id)])    # (E, N, Z) metres
                ax.errorbar(r.lon, r.lat,
                            xerr=hw[0] / (111320.0 * np.cos(np.deg2rad(r.lat))),
                            yerr=hw[1] / 111320.0,
                            fmt="none", ecolor="0.4", elinewidth=0.7, capsize=2, zorder=2)
        ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{cfg.region} — {lab} ({len(df)} ev)")
        ax.set_aspect("equal", "datalim")
        _format_lonlat(ax)
    fig.tight_layout()
    return fig


def relocation_counts(cfg, velmodel="kim1983"):
    """Located-event counts at each stage — `.sum` (HYPOINVERSE absolute), dt.ct (catalog
    relocation), dt.cc (cross-correlation relocation, the high-end product). Counts shrink
    `.sum ≥ dt.ct ≥ dt.cc` because HypoDD keeps only events with enough inter-event links: dt.ct
    drops events isolated in catalog differential-time space, and dt.cc further drops events whose
    waveforms don't cross-correlate well. (Each HypoDD run re-clusters independently, so the
    ordering is not strictly monotonic.) Returns a tidy DataFrame."""
    def _n(path, kind):
        if not os.path.exists(path):
            return None
        try:
            return int(len(sumio.read_sum(path) if kind == "sum" else sumio.read_reloc(path)))
        except Exception:
            return None
    rows = [
        (".sum (absolute)", _n(config.sum_file(cfg, velmodel), "sum")),
        ("dt.ct (catalog)", _n(os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"), "reloc")),
        ("dt.cc (cross-corr, high-end)", _n(os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc"), "reloc")),
    ]
    return pd.DataFrame([{"stage": s, "events": n} for s, n in rows])


def plot_3d_plane(cfg, velmodel=None, color_by="time"):
    """Interactive 3-D view (**plotly**) of the dt.cc-relocated hypocentres with the SVD best-fit fault
    plane overlaid as a translucent patch — rotate/zoom in a notebook. Returns a plotly Figure.

    Hypocentres: relative E–N–depth (km) about the cloud centroid, coloured by origin time (default;
    `color_by="depth"`), sized by magnitude. Plane: the relocation cloud's best-fit plane
    (`_best_fit_plane`) through the centroid, drawn over the cloud extent. Depth axis points down.
    Reads the headline dt.cc reloc (dt.ct fallback). plotly is imported lazily (optional dependency)."""
    import plotly.graph_objects as go
    import matplotlib.dates as mdates
    velmodel = velmodel or cfg.fm_velmodel
    reloc, branch = _reloc_path(cfg)
    d = sumio.read_reloc(reloc) if os.path.exists(reloc) else pd.DataFrame()
    if not len(d):
        return go.Figure().update_layout(
            title=f"{cfg.region}: no HypoDD reloc (run ph2dt→dtcc first)")

    x0, y0, z0 = float(d.x.mean()), float(d.y.mean()), float(d.z.mean())
    E = (d.x - x0).to_numpy() / 1000.0
    N = (d.y - y0).to_numpy() / 1000.0
    dep = (d.z - z0).to_numpy() / 1000.0                        # km, +down (Z is positive down)
    mag = np.nan_to_num(d.mag.to_numpy(), nan=0.0)
    size = np.clip(4 + 3 * mag, 3, 18)

    if color_by == "time" and "time" in d and d.time.notna().any():
        cvals = np.array(mdates.date2num([t.datetime for t in d.time]))
        tv = np.linspace(cvals.min(), cvals.max(), 5)
        cbar = dict(title="Origin time", tickvals=list(tv),
                    ticktext=[mdates.num2date(v).strftime("%Y-%m-%d") for v in tv])
        cscale = "RdBu_r"
    else:
        cvals, cbar, cscale = d.depth.to_numpy(), dict(title="Depth (km)"), "Viridis_r"

    boot = _load_bootstrap(cfg, branch)            # 95% bootstrap error bars in E/N/depth (percentile)
    err = {}
    if boot:
        _hw = {int(i): (_pct_hw(boot[int(i)]) / 1000.0 if int(i) in boot else None) for i in d.id}
        def _e(j):
            return [_hw[int(i)][j] if _hw.get(int(i)) is not None else 0.0 for i in d.id]
        err = dict(error_x=dict(type="data", array=_e(0), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"),
                   error_y=dict(type="data", array=_e(1), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"),
                   error_z=dict(type="data", array=_e(2), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"))
    data = [go.Scatter3d(x=E, y=N, z=dep, mode="markers", name="Hypocentres",
                         marker=dict(size=size, color=cvals, colorscale=cscale, colorbar=cbar,
                                     line=dict(width=0.5, color="black"), opacity=0.95),
                         text=[f"{int(i)}  M{m:.1f}" for i, m in zip(d.id, mag)], **err)]
    title = f"{cfg.region} — 3-D relocated seismicity ({branch})"
    if boot:
        title += "  (bars = 95% bootstrap)"
    if len(d) >= 3:
        strike, dip = _best_fit_plane(d.x, d.y, d.z)                        # for the label
        # build the plane patch in the SAME (E, N, depth-down) frame as the plotted points, directly
        # from the SVD normal, so it lies in the cloud (no strike/dip round-trip, no sign ambiguity).
        n = _svd_normal(E, N, dep)
        u1 = np.cross(n, [0.0, 0.0, 1.0])
        u1 = (np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u1) < 1e-9 else u1 / np.linalg.norm(u1))
        u2 = np.cross(n, u1); u2 = u2 / np.linalg.norm(u2)
        L = (max(np.ptp(E), np.ptp(N), np.ptp(dep)) / 2.0) or 0.2
        c0 = np.array([E.mean(), N.mean(), dep.mean()])
        corners = [c0 + a * u1 + b * u2 for a in (-L, L) for b in (-L, L)]   # idx 0,1,2,3
        cx = [c[0] for c in corners]; cy = [c[1] for c in corners]; cz = [c[2] for c in corners]
        data.append(go.Mesh3d(x=cx, y=cy, z=cz, i=[0, 0], j=[1, 3], k=[3, 2],
                              opacity=0.3, color="gray", hoverinfo="name",
                              name=f"Best-fit plane {strike:.0f}/{dip:.0f}", showscale=False))
        title += f"  +  best-fit plane (strike {strike:.0f}°, dip {dip:.0f}°)"
    fig = go.Figure(data=data)
    fig.update_layout(title=title, height=700, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="E (km)", yaxis_title="N (km)",
                                 zaxis=dict(title="Depth (km)", autorange="reversed"),
                                 aspectmode="data"))
    return fig


def _first_motion_sign(tr, ptime, noise=(0.5, 0.05), win=0.3, k=3.0):
    """Analyst-style first-motion sign at a P pick: sign of the first post-P sample whose |amplitude|
    exceeds `k` × the pre-P noise std (fallback: the first sample after P). +1 up / −1 down / 0 unread."""
    s = tr.copy().detrend("demean")
    n0 = s.slice(ptime - noise[0], ptime - noise[1])
    nstd = float(np.std(n0.data)) if len(n0.data) > 3 else 0.0
    w = s.slice(ptime, ptime + win)
    if len(w.data) < 2:
        return 0
    dd = w.data.astype(float)
    if nstd > 0 and np.any(np.abs(dd) > k * nstd):
        idx = int(np.argmax(np.abs(dd) > k * nstd))
    else:
        idx = 1
    return int(np.sign(dd[min(idx, len(dd) - 1)]))


def polarity_quality(cfg, velmodel=None):
    """PhaseNet+ P first-motion polarity quality: (1) confidence |polarity| distribution, (2) pick
    probability distribution, (3) SKHASH `polarity_misfit` per event vs # P polarities, coloured by
    mechanism quality (lower misfit = polarities more self-consistent with a double-couple). Reads the
    run's `picks/*_picks.csv` + `mechanisms.csv`. A first-order, ground-truth-free quality check."""
    import glob as _glob
    velmodel = velmodel or cfg.fm_velmodel
    pol, prob = [], []
    for pf in sorted(_glob.glob(os.path.join(config.picks_dir(cfg), "*_picks.csv"))):
        df = pd.read_csv(pf)
        if "Polarity" not in df.columns:
            continue
        P = df[(df.Phase == "P") & df.Polarity.notna()]
        pol += list(P.Polarity.astype(float))
        if "Probability" in P.columns:
            prob += list(pd.to_numeric(P.Probability, errors="coerce").dropna())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=120)
    if pol:
        ap = np.abs(np.array(pol, float))
        axes[0].hist(ap, bins=20, range=(0, 1), color="steelblue", edgecolor="k")
        frac_lo = float((ap < 0.5).mean())
        axes[0].axvline(0.5, color="r", ls="--")
        axes[0].text(0.5, axes[0].get_ylim()[1] * 0.92, f"  {frac_lo * 100:.0f}% < 0.5", color="r")
        axes[0].set(xlabel="PhaseNet+ confidence |polarity|", ylabel="P picks",
                    title=f"Polarity confidence (n={len(ap)})")
    if prob:
        axes[1].hist(np.array(prob, float), bins=20, range=(0, 1), color="seagreen", edgecolor="k")
    axes[1].set(xlabel="Pick probability", ylabel="P picks", title="Pick probability")

    mp = config.fm_mech_csv(cfg, velmodel)
    if os.path.exists(mp):
        m = _load_mechanisms(mp)
        colq = {"A": "tab:green", "B": "tab:olive", "C": "tab:orange", "D": "tab:red"}
        for q, g in m.groupby("quality"):
            axes[2].scatter(g.num_p_pol, g.polarity_misfit, c=colq.get(q, "gray"),
                            label=q, s=55, edgecolor="k")
        axes[2].axhline(20, color="0.5", ls=":")
        axes[2].legend(title="Quality", fontsize=8)
    axes[2].set(xlabel="P polarities used", ylabel="SKHASH polarity misfit (%)",
                title="Mechanism polarity misfit")
    fig.suptitle(f"{cfg.region} — PhaseNet+ polarity quality", fontsize=13)
    fig.tight_layout()
    return fig


def polarity_vs_manual(cfg, win=0.3, k=3.0, conf_bins=(0.0, 0.3, 0.6, 1.01)):
    """Quasi-ground-truth check where manual picks exist (e.g. Gwangyang `manual_picks/`): compare the
    PhaseNet+ polarity SIGN to the first-motion sign read at the **manual** P pick (SAC header `a`) on
    the vertical. Plots overall agreement + agreement vs PhaseNet+ confidence. The manual first motion
    is a proxy (an analyst reads the raw first swing). Returns a Figure (graceful note if unavailable)."""
    import glob as _glob
    mroot = os.path.join(getattr(cfg, "src_root", "") or "", "manual_picks")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=120)
    if not os.path.isdir(mroot):
        for a in axes:
            a.axis("off")
        axes[0].set_title(f"{cfg.region}: no manual_picks/ — manual proxy not available")
        return fig
    recs = []                                                # (|polarity|, agree 0/1)
    for evdir in sorted(_glob.glob(os.path.join(mroot, "20*"))):
        eid = os.path.basename(evdir)
        pc = config.picks_csv(cfg, eid)
        if not os.path.exists(pc):
            continue
        pk = pd.read_csv(pc)
        if "Polarity" not in pk.columns:
            continue
        pk = pk[(pk.Phase == "P") & pk.Polarity.notna()]
        pmap = {str(r.Station): float(r.Polarity) for r in pk.itertuples()}
        for z in _glob.glob(os.path.join(evdir, "*Z.sac")):
            sta = os.path.basename(z).split(".")[2]
            if sta not in pmap:
                continue
            try:
                tr = read(z)[0]
                a = tr.stats.sac.get("a", -12345.0)
                if a == -12345.0:
                    continue
                ptime = tr.stats.starttime - tr.stats.sac.b + a
                ms = _first_motion_sign(tr, ptime, win=win, k=k)
            except Exception:                                # noqa: BLE001
                continue
            if ms == 0:
                continue
            ppol = pmap[sta]
            recs.append((abs(ppol), int(np.sign(ppol) == ms)))
    if not recs:
        for a in axes:
            a.axis("off")
        axes[0].set_title(f"{cfg.region}: no matched manual/PhaseNet+ first motions")
        return fig
    conf = np.array([r[0] for r in recs]); agree = np.array([r[1] for r in recs])
    axes[0].bar([0, 1], [int((agree == 1).sum()), int((agree == 0).sum())],
                color=["tab:green", "tab:red"], edgecolor="k")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Agree", "Disagree"])
    axes[0].set(ylabel="P picks",
                title=f"{cfg.region} — PhaseNet+ vs manual first motion\n"
                      f"overall agreement {agree.mean() * 100:.0f}% (n={len(recs)})")
    centers, rates, ns = [], [], []
    for lo, hi in zip(conf_bins[:-1], conf_bins[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum():
            centers.append((lo + hi) / 2); rates.append(float(agree[sel].mean())); ns.append(int(sel.sum()))
    axes[1].bar(centers, rates, width=0.22, color="steelblue", edgecolor="k")
    for c, rt, n in zip(centers, rates, ns):
        axes[1].text(c, min(rt + 0.03, 1.02), f"n={n}", ha="center", fontsize=8)
    axes[1].set(xlabel="PhaseNet+ confidence |polarity|", ylabel="Agreement with manual first motion",
                ylim=(0, 1.08), title="Agreement vs confidence")
    fig.tight_layout()
    return fig
