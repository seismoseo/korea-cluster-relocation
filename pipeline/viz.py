"""
Lightweight matplotlib plots for monitoring the pipeline from JupyterLab.

Every function takes a ClusterConfig and returns a matplotlib Figure, so the same
calls work in a notebook (inline) and in scripts. Kept dependency-light (matplotlib
+ obspy only; no PyGMT) so the monitoring notebooks run anywhere the pipeline does.
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
    public clone without the fonts still renders. Runs once at import."""
    import matplotlib.font_manager as fm
    font_dir = getattr(config, "HELVETICA_DIR", None)
    try:
        if font_dir and os.path.isdir(font_dir):
            for fpath in fm.findSystemFonts(font_dir):
                fm.fontManager.addfont(fpath)
        names = {f.name for f in fm.fontManager.ttflist}
        if "Helvetica" in names:
            mpl.rcParams["font.family"] = "Helvetica"
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
    ax.scatter(*cfg.epicenter[::-1], marker="*", s=300, c="red",
               edgecolor="k", zorder=4, label="Cluster center")
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{cfg.region} — {len(df)} events ({source}:{branch})")
    ax.legend(loc="best", fontsize=8); ax.set_aspect("equal", "datalim")
    return ax.figure


def depth_sections(cfg, velmodel="kim1983", source="sum"):
    """Lon-depth and lat-depth cross-sections (source='reloc' uses the headline dt.cc relocation)."""
    df = (sumio.read_sum(config.sum_file(cfg, velmodel)) if source == "sum"
          else sumio.read_reloc(_reloc_path(cfg)[0]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    a1.scatter(df.lon, df.depth, s=40, c="steelblue", edgecolor="k")
    a1.set(xlabel="Longitude", ylabel="Depth (km)", title=f"{cfg.region} longitude-depth")
    a1.invert_yaxis()
    a2.scatter(df.lat, df.depth, s=40, c="steelblue", edgecolor="k")
    a2.set(xlabel="Latitude", ylabel="Depth (km)", title="Latitude-depth")
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


def _best_fit_plane(x, y, z):
    """Strike/dip (deg) of the best-fitting plane through points (x, y, z) via SVD — the data-driven
    fault orientation. Ported from the original notebooks' `calculate_strike_dip_svd`: the plane
    normal is the smallest-singular-value direction; flip it up, then dip = acos(n_up),
    strike = atan2(n_north, -n_east) (Aki & Richards right-hand rule). x,y = HypoDD .reloc X,Y
    (east, north), z = .reloc Z (up-positive). Needs ≥ 3 points."""
    coords = np.column_stack((np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)))
    coords = coords - coords.mean(axis=0)
    normal = np.linalg.svd(coords)[2][-1]
    n_e, n_n, n_u = normal if normal[2] > 0 else -normal
    dip = float(np.degrees(np.arccos(np.clip(n_u, -1.0, 1.0))))
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
    dep = -(d.z.to_numpy() - z0) / 1000.0                         # km, +down, rel. to centre
    along_dip = across * np.cos(np.deg2rad(used_dip)) + dep * np.sin(np.deg2rad(used_dip))   # km

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
    ax.scatter(along, dep, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    ax.text(-0.92 * R, -0.88 * R, "A", fontsize=16, fontweight="bold")
    ax.text(0.86 * R, -0.88 * R, "A'", fontsize=16, fontweight="bold")
    ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="Along-strike distance (km)",
           ylabel="Depth rel. to reference (km)", title="Along-strike (A–A')")
    _style(ax); ax.invert_yaxis()

    # panel 3 — across-strike depth section (B–B'), with the dashed fault-dip line
    ax = axes[2]
    ax.scatter(across, dep, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    xx = np.linspace(-R, R, 50)
    ax.plot(xx, xx * np.tan(np.deg2rad(used_dip)), color="k", lw=1.0, ls="--", zorder=1,
            label=f"Dip {used_dip:.0f}°")
    ax.text(-0.92 * R, -0.88 * R, "B", fontsize=16, fontweight="bold")
    ax.text(0.86 * R, -0.88 * R, "B'", fontsize=16, fontweight="bold")
    ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="Across-strike distance (km)",
           ylabel="Depth rel. to reference (km)", title="Across-strike (B–B')")
    _style(ax); ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)

    # panel 4 — fault-plane (along-dip) view
    ax = axes[3]
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
    fig.suptitle(f"{cfg.region} — relocated seismicity in fault coordinates ({branch})\n"
                 f"strike {used_strike:.0f}°, dip {used_dip:.0f}° [{src}]{fmtxt}", fontsize=13)
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
            plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)
        ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{cfg.region} — {lab} ({len(df)} ev)")
        ax.set_aspect("equal", "datalim")
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
