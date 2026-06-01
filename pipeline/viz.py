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


def _event_magnitudes(cfg):
    """{cuspid: KMA local magnitude} from the event catalog (KST origins), matched to the UTC event dirs
    via the cuspid scheme (cuspid = cfg.cuspid_offset + index over sorted waveform dirs; reloc/mechanism
    `id` IS the cuspid). The reloc `mag` column is 0, so this catalog lookup is the magnitude source.
    Returns {} on any failure (callers fall back to a default marker size)."""
    try:
        cat = pd.read_csv(cfg.event_catalog_csv)
        cat.columns = [c.lower() for c in cat.columns]   # catalogs vary in case (Year vs year); normalise
        eid2mag = {}
        for r in cat.itertuples():
            t = UTCDateTime(int(r.year), int(r.month), int(r.day), int(r.hour), int(r.minute),
                            int(r.second)) - cfg.kst_offset_hours * 3600
            eid2mag[t.strftime("%Y%m%d%H%M%S")] = float(r.magnitude)
        dirs = sorted(glob.glob(os.path.join(config.waveforms_dir(cfg), "20*")))
        return {cfg.cuspid_offset + i: eid2mag[os.path.basename(d)]
                for i, d in enumerate(dirs) if os.path.basename(d) in eid2mag}
    except Exception:                                # noqa: BLE001
        return {}


def _mag_size(mags, smin=20.0, smax=1200.0):
    """Marker areas ∝ exp(magnitude) (the originals' `s = 5·exp(2·M)`), clipped to [smin, smax]. NaN
    entries (events without a catalog match) use the median magnitude so they still plot."""
    m = np.asarray(mags, dtype=float)
    if not np.isfinite(m).any():
        m = np.ones_like(m)
    med = np.nanmedian(m)
    m = np.where(np.isfinite(m), m, med)
    return np.clip(5.0 * np.exp(2.0 * m), smin, smax)


def _mag_for(cfg, ids):
    """Magnitude array aligned to `ids` (cuspids) from `_event_magnitudes` (NaN where unmatched)."""
    mg = _event_magnitudes(cfg)
    return np.array([mg.get(int(i), np.nan) for i in ids], dtype=float)


def _cuspid_event_ids(cfg):
    """{cuspid: event_id (UTC YYYYMMDDHHMMSS)} from the sorted waveform dirs — the cuspid scheme
    (cuspid = cfg.cuspid_offset + index). The inverse of the `id` carried through reloc/mechanisms."""
    dirs = sorted(glob.glob(os.path.join(config.waveforms_dir(cfg), "20*")))
    return {cfg.cuspid_offset + i: os.path.basename(d) for i, d in enumerate(dirs)}


# bootstrap-flagged "under-constrained" events are dropped from the dt.cc views: their relative location
# is poorly determined (large 95% spread / few stable replicas) — e.g. a shallow, large-azimuthal-gap event
# whose good CC data still can't resolve it. Tunable.
BOOT_DROP_HORIZ_KM = 0.1            # drop if the 95% horizontal half-width √(ex95²+ey95²) exceeds this (km)
BOOT_DROP_VERT_KM  = 0.1            # drop if the 95% vertical half-width ez95 exceeds this (km); set None to disable
BOOT_DROP_MIN_NBOOT_FRAC = 0.6      # ...or if relocated in fewer than this fraction of replicas


def _boot_underconstrained(cfg, branch):
    """Cuspids the bootstrap flags as under-constrained (drop them from the dt.cc views): horizontal 95%
    half-width > `BOOT_DROP_HORIZ_KM`, vertical 95% half-width > `BOOT_DROP_VERT_KM` (when not None),
    `n_boot` below `BOOT_DROP_MIN_NBOOT_FRAC` of the replicas, or no CI at all. Empty set when there's no
    bootstrap cache (so plots are unchanged without one). The three constants are module-level so a
    notebook can `viz.BOOT_DROP_VERT_KM = 0.5` to relax per-run."""
    bdir = config.dtcc_dir(cfg) if "cc" in branch else config.dtct_dir(cfg)
    p = os.path.join(bdir, "bootstrap_errors.csv")
    if not os.path.exists(p):
        return set()
    import re
    n = next((int(m.group(1)) for m in [re.search(r"\bn=(\d+)", open(p).readline())] if m), None)
    df = pd.read_csv(p, comment="#")
    bad = set()
    for r in df.itertuples():
        horiz = np.hypot(r.ex95, r.ey95) / 1000.0 if np.isfinite(r.ex95) else np.inf
        vert  = r.ez95 / 1000.0 if np.isfinite(r.ez95) else np.inf
        if (not np.isfinite(r.ex95) or horiz > BOOT_DROP_HORIZ_KM
                or (n and r.n_boot < BOOT_DROP_MIN_NBOOT_FRAC * n)
                or (BOOT_DROP_VERT_KM is not None and vert > BOOT_DROP_VERT_KM)):
            bad.add(int(r.id))
    return bad


def map_catalog(cfg, velmodel="kim1983", source="sum", ax=None,
                include_all=False, show_errors=True):
    """Epicenter map coloured by depth, with the used stations. source = 'sum'|'reloc'.
    For source='reloc' the headline dt.cc relocation is shown (dt.ct fallback).

    `include_all=True` adds, on top of the relocated events, the events that the dt.cc
    relocation dropped (i.e. they exist in the HypoInverse .sum but not in the dt.cc reloc:
    HypoDD clustering threshold OBSCC=4) plus the events the bootstrap flagged as
    under-constrained. Dropped events are drawn as **hollow squares** at their absolute
    (.sum) location so you can see the whole catalog in one view. Default `False` preserves
    the v1.1.x behavior (filtered to bootstrap-OK reloc events only).

    `show_errors=False` skips the 95% bootstrap error bars — useful for a clean summary view
    when paired with `include_all=True`.
    """
    branch = velmodel
    ndrop = 0
    extra = pd.DataFrame()                       # events to overlay as "dropped" markers
    if source == "sum":
        df = sumio.read_sum(config.sum_file(cfg, velmodel))
    else:
        path, branch = _reloc_path(cfg)
        df = sumio.read_reloc(path)
        drop_boot = _boot_underconstrained(cfg, branch)   # bootstrap-flagged under-constrained
        ndrop = int(df.id.isin(drop_boot).sum())
        df_kept = df[~df.id.isin(drop_boot)]
        if include_all:
            sum_df = sumio.read_sum(config.sum_file(cfg, velmodel))
            in_reloc = set(df.id.astype(int))
            # union of: events not in the dt.cc reloc + events bootstrap-dropped
            extra_ids = (set(int(i) for i in sum_df.id) - in_reloc) | set(int(i) for i in drop_boot)
            extra = sum_df[sum_df.id.astype(int).isin(extra_ids)]
        df = df_kept
    sta = pd.read_csv(config.used_stations_csv(cfg))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.scatter(sta.Longitude, sta.Latitude, marker="^", s=40, c="0.6",
               edgecolor="k", label=f"Stations ({len(sta)})", zorder=2)
    # depth normalization spans BOTH the kept and the dropped events so colors stay consistent
    depths = pd.concat([df.depth, extra.depth]) if len(extra) else df.depth
    norm = mpl.colors.Normalize(vmin=float(depths.min()), vmax=float(depths.max()))
    sc = ax.scatter(df.lon, df.lat, c=df.depth, s=_mag_size(_mag_for(cfg, df.id)),
                    cmap="viridis_r", norm=norm, edgecolor="k", zorder=3)
    plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)
    if len(extra):
        # legend entry only (single dummy, neutral grey); the actual depth-coloured edges are
        # drawn per-event below — matplotlib doesn't natively support `c=` + `facecolor="none"`
        # together (it warns, with `c` winning).
        ax.scatter([], [], marker="s", facecolor="none", edgecolor="0.3", linewidth=1.5,
                   label=f"Dropped from dt.cc ({len(extra)}, .sum)")
        cmap = plt.get_cmap("viridis_r")
        for r in extra.itertuples():
            ax.scatter(r.lon, r.lat, s=_mag_size(np.array([_mag_for(cfg, [r.id])[0]]))[0],
                       facecolor="none", edgecolor=cmap(norm(r.depth)),
                       linewidth=1.6, marker="s", zorder=3.5)
    boot = _load_bootstrap(cfg, branch) if source == "reloc" else None
    if boot and show_errors:
        for r in df.itertuples():
            if int(r.id) not in boot:
                continue
            hw = _pct_hw(boot[int(r.id)])        # (E, N, Z) metres
            ax.errorbar(r.lon, r.lat, xerr=hw[0] / (111320.0 * np.cos(np.deg2rad(r.lat))),
                        yerr=hw[1] / 111320.0, fmt="none", ecolor="0.4",
                        elinewidth=0.7, capsize=2, zorder=2)
    ax.scatter(*cfg.epicenter[::-1], marker="*", s=300, c="red",
               edgecolor="k", zorder=4, label="Cluster center")
    bl = " + 95% bootstrap" if boot and show_errors else ""
    dl = (f", {len(extra)} dt.cc-dropped overlaid" if len(extra)
          else (f", {ndrop} under-constrained dropped" if ndrop else ""))
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{cfg.region} — {len(df) + len(extra)} events ({source}:{branch}{bl}{dl})")
    ax.legend(loc="best", fontsize=8); ax.set_aspect("equal", "datalim")
    _format_lonlat(ax)
    return ax.figure


def depth_sections(cfg, velmodel="kim1983", source="sum",
                   include_all=False, show_errors=True):
    """Lon-depth and lat-depth cross-sections (source='reloc' uses the headline dt.cc relocation,
    with 95% bootstrap depth/horizontal error bars when the bootstrap cache exists).

    `include_all=True` overlays events the dt.cc reloc dropped (HypoDD clustering / bootstrap
    under-constrained) at their .sum locations as hollow squares. `show_errors=False` skips
    the bootstrap bars."""
    branch = "dt.cc"
    extra = pd.DataFrame()
    if source == "sum":
        df = sumio.read_sum(config.sum_file(cfg, velmodel))
    else:
        path, branch = _reloc_path(cfg)
        df = sumio.read_reloc(path)
        drop_boot = _boot_underconstrained(cfg, branch)
        df_kept = df[~df.id.isin(drop_boot)]
        if include_all:
            sum_df = sumio.read_sum(config.sum_file(cfg, velmodel))
            in_reloc = set(df.id.astype(int))
            extra_ids = (set(int(i) for i in sum_df.id) - in_reloc) | set(int(i) for i in drop_boot)
            extra = sum_df[sum_df.id.astype(int).isin(extra_ids)]
        df = df_kept
    boot = _load_bootstrap(cfg, branch) if source == "reloc" else None
    sz = _mag_size(_mag_for(cfg, df.id))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    a1.scatter(df.lon, df.depth, s=sz, c="steelblue", edgecolor="k", zorder=3)
    a1.set(xlabel="Longitude", ylabel="Depth (km)", title=f"{cfg.region} longitude-depth")
    a2.scatter(df.lat, df.depth, s=sz, c="steelblue", edgecolor="k", zorder=3)
    a2.set(xlabel="Latitude", ylabel="Depth (km)", title="Latitude-depth")
    if len(extra):
        sz_e = _mag_size(_mag_for(cfg, extra.id))
        a1.scatter(extra.lon, extra.depth, s=sz_e, marker="s", facecolor="none",
                   edgecolor="firebrick", linewidth=1.5, zorder=3,
                   label=f"dt.cc-dropped (.sum, n={len(extra)})")
        a2.scatter(extra.lat, extra.depth, s=sz_e, marker="s", facecolor="none",
                   edgecolor="firebrick", linewidth=1.5, zorder=3)
        a1.legend(loc="best", fontsize=8)
    if boot and show_errors:
        for r in df.itertuples():
            if int(r.id) not in boot:
                continue
            hw = _pct_hw(boot[int(r.id)])
            a1.errorbar(r.lon, r.depth, xerr=hw[0] / (111320.0 * np.cos(np.deg2rad(r.lat))),
                        yerr=hw[2] / 1000.0, fmt="none", ecolor="0.4", elinewidth=0.7, capsize=2, zorder=2)
            a2.errorbar(r.lat, r.depth, xerr=hw[1] / 111320.0,
                        yerr=hw[2] / 1000.0, fmt="none", ecolor="0.4", elinewidth=0.7, capsize=2, zorder=2)
    a1.invert_yaxis(); a2.invert_yaxis()
    fig.tight_layout()
    return fig


def _read_dt_pairs(path):
    """Parse a HypoDD `dt.ct` / `dt.cc` file → `{(id1, id2): n_obs}` (P + S combined).

    Each event pair is a block starting with `# id1 id2 [otc]` followed by per-station
    observation rows; `n_obs` is the row count for that pair. Returns `{}` if the file
    doesn't exist (so callers can degrade gracefully when a branch wasn't run)."""
    if not os.path.exists(path):
        return {}
    pairs = {}
    cur = None
    with open(path) as f:
        for ln in f:
            if ln.startswith("#"):
                _, a, b, *_ = ln.split()
                cur = (int(a), int(b))
                pairs[cur] = 0
            elif cur is not None and ln.strip():
                pairs[cur] += 1
    return pairs


def _branch_dt_file(cfg, branch):
    """Path to the dt.* file actually fed to HypoDD for `branch` ('dtct' | 'dtcc').

    For dtcc the filename is the cluster's configured `cc_file` (e.g. `dt.cc_0.7_combined`)
    — the same one the bootstrap resamples — so this stays in sync with whatever the
    cluster module declared."""
    if branch == "dtct":
        return os.path.join(config.dtct_dir(cfg), "dt.ct")
    cc_file = cfg.hypodd_dtcc_variants["default"].cc_file
    return os.path.join(config.dtcc_dir(cfg), cc_file)


def link_map(cfg, velmodel=None, branch="dtcc", min_obs=1, ax=None,
             cmap_name="viridis", lw_range=(0.4, 2.5)):
    """Inter-event link map: a line between each (id1, id2) HypoDD pair, coloured by the
    number of differential-time observations (P + S, combined) for that pair.

    Reads `dt.ct` (`branch="dtct"`) or the cluster's `cc_file` (`branch="dtcc"`) for the
    pair counts, and `hypoDD.reloc` from the same branch directory for epicenters.
    `min_obs` drops sparser pairs; line width also scales linearly with obs count so
    strong doublets stand out. Pairs are drawn weakest-first so the strongest sit on top."""
    branch_dir = config.dtct_dir(cfg) if branch == "dtct" else config.dtcc_dir(cfg)
    reloc_path = os.path.join(branch_dir, "hypoDD.reloc")
    if not os.path.exists(reloc_path):
        raise FileNotFoundError(
            f"need {branch} hypoDD.reloc at {reloc_path} — run the {branch} stage first")
    raw_pairs = {k: v for k, v in _read_dt_pairs(_branch_dt_file(cfg, branch)).items()
                 if v >= min_obs}
    reloc = sumio.read_reloc(reloc_path)
    coords = {int(r.id): (float(r.lon), float(r.lat)) for r in reloc.itertuples()}
    # Restrict to pairs where BOTH events made it through this branch's reloc — the dt.* file
    # may list pairs that HypoDD's clustering or bootstrap subsequently dropped, and we don't
    # want the title count to claim links that aren't actually drawn.
    pairs = {k: v for k, v in raw_pairs.items() if k[0] in coords and k[1] in coords}

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)

    nobs_max = max(pairs.values()) if pairs else 1
    norm = mpl.colors.Normalize(vmin=min_obs, vmax=max(nobs_max, min_obs + 1))
    cmap = plt.get_cmap(cmap_name)
    lw_lo, lw_hi = lw_range
    span = max(1, nobs_max - min_obs)
    for (a, b), n in sorted(pairs.items(), key=lambda x: x[1]):
        x = (coords[a][0], coords[b][0])
        y = (coords[a][1], coords[b][1])
        lw = lw_lo + (lw_hi - lw_lo) * (n - min_obs) / span
        ax.plot(x, y, color=cmap(norm(n)), lw=lw, alpha=0.85, zorder=2)
    # epicenters: hollow white dots on top so the links are still readable through them
    ax.scatter(reloc.lon, reloc.lat, s=70, c="white",
               edgecolor="k", linewidth=1.0, zorder=3)
    if pairs:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        plt.colorbar(sm, ax=ax, label=f"{branch} observations per pair", shrink=0.8)

    nlabel = "dt.ct" if branch == "dtct" else "dt.cc"
    nev   = len(coords)
    ndrop = len(raw_pairs) - len(pairs)
    drop_str = f", {ndrop} unreloc-dropped" if ndrop else ""
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{cfg.region} — {nlabel} inter-event links "
                 f"({nev} events, {len(pairs)} pairs, {sum(pairs.values())} obs{drop_str})")
    ax.set_aspect("equal", "datalim")
    _format_lonlat(ax)
    return ax.figure


def link_maps(cfg, velmodel=None, min_obs=1):
    """Side-by-side `link_map` for both branches (dt.ct left, dt.cc right) so the user
    can compare how cross-correlation tightens the connectivity. Missing branches are
    skipped silently (returns whichever figures could be built)."""
    have_ct = os.path.exists(os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"))
    have_cc = os.path.exists(os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc"))
    if not (have_ct or have_cc):
        return None
    ncol = int(have_ct) + int(have_cc)
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 6.5), dpi=110, squeeze=False)
    i = 0
    if have_ct:
        link_map(cfg, velmodel=velmodel, branch="dtct", min_obs=min_obs, ax=axes[0, i]); i += 1
    if have_cc:
        link_map(cfg, velmodel=velmodel, branch="dtcc", min_obs=min_obs, ax=axes[0, i])
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
    sta = pd.read_csv(config.used_stations_csv(cfg)).drop_duplicates("Code").set_index("Code")
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
    # `Sensor` lookup can return a Series if the station table has dup rows (e.g. STP
    # returning surface + borehole entries); force scalar so the f-string glob still works.
    sensor_val = sta.loc[station, "Sensor"]
    sensor = sensor_val.iloc[0] if hasattr(sensor_val, "iloc") else sensor_val
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


def plot_record_section(cfg, event_id, prob_min=None,
                        max_stations=60, max_dist_km=100.0,
                        pre_s=2.0, post_s=60.0, normalize_per_trace=True):
    """Distance record section for one event — the primary visual QC for picking quality.

    Z-component traces ordered by **hypocentral** distance (y-axis, km) vs time relative to the event
    origin (x-axis, s). AI picks meeting the probability threshold are overlaid as vertical ticks
    (P=red, S=blue); the predicted P and S moveout lines `t = hypo_dist / V` are drawn red/blue dashed
    so you can SEE whether the picks fall on the real arrivals or on later phases / noise.

    Pattern follows `/home/msseo/works/12.Ridgecrest/scripts/02_phase_picking.py` (Ridgecrest reference):
    both the y-axis and the moveout use hypocentral distance `sqrt(epi² + evdp²)`, where `evdp` is the
    **per-event catalog depth** (fallback to `cfg.pick_window["evdp"]`). Picks from the CSV are
    converted to origin-relative time via `t_pick = UTCDateTime(pick.Time) − origin_abs`, with
    `origin_abs = starttime − sac.b + sac.o` — robust to whether the SAC reference (nz*) is the catalog
    origin or a post-rereference origin.

    `prob_min`: `None` (default) uses the picker's own thresholds (`cfg.p_threshold` for P,
    `cfg.s_threshold` for S — shows every picker-emitted pick, most diagnostic); a scalar applies
    one threshold to both phases; a `(p_thr, s_thr)` tuple sets them independently."""
    from obspy.geodetics.base import gps2dist_azimuth
    from pipeline.core import waveforms
    pc = config.picks_csv(cfg, event_id)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=120)
    if not os.path.exists(pc):
        ax.set_title(f"{cfg.region} {event_id}: no picks csv"); return fig
    picks = pd.read_csv(pc)
    if prob_min is None:
        p_thr, s_thr = cfg.p_threshold, cfg.s_threshold
    elif isinstance(prob_min, (int, float)):
        p_thr = s_thr = float(prob_min)
    else:
        p_thr, s_thr = float(prob_min[0]), float(prob_min[1])

    # per-event focal depth (matches the picking-window evdp computation in pick_event):
    # catalog depth if present and positive, else the cluster default
    evdp_default = float(cfg.pick_window.get("evdp", 10.0))
    evdp = evdp_default
    try:
        for e in waveforms.load_catalog(cfg):
            if str(e.get("event_id", "")) == str(event_id):
                d = e.get("depth")
                if d is not None and float(d) > 0:
                    evdp = float(d)
                break
    except Exception:                                # noqa: BLE001
        pass

    wf = config.event_wf_dir(cfg, event_id)
    rows = []
    for f in sorted(glob.glob(f"{wf}/{event_id}.*Z.sac")):
        try:
            tr = read(f)[0]
            s = tr.stats.sac
            code = os.path.basename(f).split(".")[2]
        except Exception:                            # noqa: BLE001
            continue
        # epicentral distance (km), recompute from station/event coords if SAC header absent
        epi = getattr(s, "dist", None)
        if epi is None or not np.isfinite(epi) or epi == 0:
            try:
                epi = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[0] / 1000.0
            except Exception:                        # noqa: BLE001
                continue
        # HYPOCENTRAL distance for both the y-axis and the moveout (matches Ridgecrest line 894)
        hypo = float(np.hypot(float(epi), evdp))
        if hypo > max_dist_km:
            continue
        # absolute event origin: SAC reference (nz*) + sac.o, robust to post-rereference SACs.
        # mirrors Ridgecrest's t_rel = tr.times() + (starttime - origin_abs) at line 1296.
        sac_o = float(getattr(s, "o", 0.0))
        if sac_o == -12345.0:
            sac_o = 0.0
        origin_abs = tr.stats.starttime - s.b + sac_o
        w = tr.copy().detrend("demean").slice(origin_abs - pre_s, origin_abs + post_s)
        if len(w.data) < 3:
            continue
        d = w.data.astype(float)
        if normalize_per_trace:
            d = d / (np.max(np.abs(d)) or 1.0)
        t_rel = w.times() + float(w.stats.starttime - origin_abs)   # sample time − origin
        rows.append(dict(code=code, hypo=hypo, origin=origin_abs, t=t_rel, trace=d))
    if not rows:
        ax.set_title(f"{cfg.region} {event_id}: no usable Z traces"); return fig
    rows = sorted(rows, key=lambda x: x["hypo"])[:max_stations]
    hypos = np.array([r["hypo"] for r in rows])
    min_h, max_h = float(hypos.min()), float(hypos.max())
    # Per-trace y-amplitude: size to the typical inter-station spacing, not the cluster's
    # max distance. With ~1-3 km spacing, the old `0.04 * max_h` -> ~4 km gain made adjacent
    # traces overlap on deep / sparse-near-field clusters (e.g. changnyeong, hypo 17-100 km).
    if len(hypos) >= 3:
        median_gap = float(np.median(np.diff(np.sort(hypos))))
        dscale = max(2.0, 0.45 * median_gap)
    else:
        dscale = max(2.0, 0.04 * max_h)

    for r in rows:
        ax.plot(r["t"], r["trace"] * dscale + r["hypo"], color="0.25", lw=0.5, alpha=0.85, zorder=2)
        ax.text(post_s * 1.005, r["hypo"], r["code"], ha="left", va="center", fontsize=6, color="0.4")

    # overlay picks: pick time relative to event origin = absolute_pick − origin_abs (per station)
    code_origin = {r["code"]: r["origin"] for r in rows}
    code_hypo = {r["code"]: r["hypo"] for r in rows}
    for _, p in picks.iterrows():
        if p.Station not in code_origin:
            continue
        thr = p_thr if p.Phase == "P" else s_thr
        if not np.isfinite(p.Probability) or p.Probability < thr:
            continue
        t_pick = float(UTCDateTime(p.Time) - code_origin[p.Station])
        col = "tab:red" if p.Phase == "P" else "tab:blue"
        ax.vlines(t_pick, code_hypo[p.Station] - dscale, code_hypo[p.Station] + dscale,
                  color=col, lw=1.3, alpha=0.8, zorder=4)

    # predicted moveout against HYPOCENTRAL distance (Ridgecrest line 913-917). Constant Vp/Vs is
    # the **depth-weighted vertical average** of the HypoInverse velocity model (cfg.fm_velmodel)
    # integrated from the surface to the event's focal depth -- so the moveout line matches the
    # model that produced the absolute locations, while staying a single straight line. The
    # picking window still uses cfg.pick_window['vp']/['vs'] (that scheme is unchanged). Falls
    # back to cfg.pick_window if no matching velocity model is configured.
    def _avg_v_to_depth(rows, depth_km):
        """rows = ((v_kms, top_depth_km), ...) sorted ascending by top_depth. Returns
        depth_km / sum(layer_thickness / v_layer) -- the average vertical-ray velocity."""
        if not rows or depth_km <= 0:
            return rows[0][0] if rows else 0.0
        total_t, used = 0.0, 0.0
        for i, (v, top) in enumerate(rows):
            next_top = rows[i + 1][1] if i + 1 < len(rows) else float("inf")
            top = max(top, 0.0)
            if top >= depth_km:
                break
            slab = min(next_top, depth_km) - top
            if slab <= 0:
                continue
            total_t += slab / float(v)
            used += slab
        return (used / total_t) if total_t > 0 else float(rows[0][0])

    vm = next((m for m in cfg.velocity_models if m.name == cfg.fm_velmodel), None)
    if vm and vm.p_rows and vm.s_rows:
        vp = _avg_v_to_depth(vm.p_rows, evdp)
        vs = _avg_v_to_depth(vm.s_rows, evdp)
        vlabel = f" ({vm.name} avg to {evdp:.1f} km)"
    else:
        vp = cfg.pick_window.get("vp", 5.9)
        vs = cfg.pick_window.get("vs", 3.0)
        vlabel = " (pick_window fallback)"
    dd = np.linspace(evdp, max_h * 1.05, 80)
    ax.plot(dd / vp, dd, "r--", lw=0.9, alpha=0.55,
            label=f"P moveout (Vp={vp:.2f} km/s){vlabel}", zorder=3)
    ax.plot(dd / vs, dd, "b--", lw=0.9, alpha=0.55,
            label=f"S moveout (Vs={vs:.2f} km/s){vlabel}", zorder=3)
    ax.legend(loc="lower right", fontsize=8)

    # Auto-fit y-axis to the data range with a small padding -- the old fixed (0, max_h*1.05)
    # left huge empty space at top and bottom for deep / sparse-near-field clusters where
    # data only occupies (say) 17-95 km of a 0-100 km axis.
    span = max(max_h - min_h, dscale)
    ax.set(xlabel="Time from origin (s)", ylabel="Hypocentral distance (km)",
           xlim=(-pre_s, post_s * 1.06),
           ylim=(max(0.0, min_h - 0.1 * span - dscale), max_h + 0.1 * span + dscale),
           title=f"{cfg.region} {event_id} (depth {evdp:.1f} km) — distance record section "
                 f"(P picks ≥ {p_thr:.2f}, S picks ≥ {s_thr:.2f}; {len(rows)} stations)")
    ax.grid(alpha=0.3)
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
    """Tidy focal-mechanism table (one row per event, preferred solution) for notebook display.

    Both nodal planes are listed: `strike/dip/rake` is the SKHASH-reported plane (NP1) and
    `strike2/dip2/rake2` is the auxiliary plane (NP2), derived from NP1 with obspy `aux_plane`. A
    double-couple mechanism is fully described by either plane; the fault is one of the two."""
    velmodel = velmodel or cfg.fm_velmodel
    path = config.fm_mech_csv(cfg, velmodel)
    if not os.path.exists(path):
        return pd.DataFrame()
    m = _load_mechanisms(path).reset_index(drop=True)
    if {"strike", "dip", "rake"}.issubset(m.columns):     # auxiliary (conjugate) nodal plane
        from obspy.imaging.beachball import aux_plane
        aux = [aux_plane(float(r.strike), float(r.dip), float(r.rake)) for r in m.itertuples()]
        m["strike2"], m["dip2"], m["rake2"] = (np.round([a[i] for a in aux], 1) for i in range(3))
    cols = ["event_id", "quality", "strike", "dip", "rake", "strike2", "dip2", "rake2",
            "fault_plane_uncertainty", "num_p_pol", "num_sp_ratios", "azimuthal_gap",
            "sta_distribution_ratio", "origin_depth_km"]
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
    msz = (_mag_size(m.magnitude.to_numpy()) if "magnitude" in m.columns else 55)   # area ∝ M_L
    sc = ax.scatter(m.origin_lon, m.origin_lat, c=m.origin_depth_km, cmap=cmap, norm=norm,
                    s=msz, edgecolor="k", lw=0.5, zorder=4, label=f"Located events ({len(m)})")
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


def _polinfo_for_event(cfg, velmodel, event_id):
    """Per-station SKHASH polarity / S-P-ratio detail for one event from `OUT/out_polinfo.csv`.

    `out_polinfo.csv`'s `event_id` column is actually the **cuspid** (200000, 200001, ...)
    while the caller's `event_id` may be the UTC string (e.g. "20250206173534") — translate
    via `mechanisms.csv` which carries both columns. Accepts either form.

    Columns kept: sta_code (NET.STA.LOC.CHAN), p_polarity (signed weight; |val| is the
    polarity confidence used by SKHASH, sign is the first-motion direction), takeoff
    (deg from downward vertical, 0-180), azimuth (deg from N), sp_ratio (S/P amplitude
    ratio or NaN if not in the inversion).
    """
    p = os.path.join(config.fm_out_dir(cfg, velmodel), "out_polinfo.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    key = str(event_id)
    # try direct match first (in case the polinfo file already uses UTC ids)
    sub = df[df.event_id.astype(str) == key]
    if sub.empty:
        # translate UTC event_id -> cuspid via mechanisms.csv
        mech_path = config.fm_mech_csv(cfg, velmodel)
        if os.path.exists(mech_path):
            m = pd.read_csv(mech_path)
            hit = m[m.event_id.astype(str) == key]
            if len(hit):
                cuspid = int(hit.iloc[0].cuspid)
                sub = df[df.event_id.astype(str) == str(cuspid)]
    if sub.empty:
        return sub
    out = sub.copy()
    # split "KS.AGSA.--.HGZ" -> "AGSA"
    out["station"] = out["sta_code"].astype(str).str.split(".").str[1]
    return out


def _lower_hemisphere_xy(azimuth_deg, takeoff_deg, r_unit=1.0):
    """Equal-area lower-hemisphere projection of a take-off vector.

    SKHASH convention: takeoff is measured from the **downward** vertical (0 = straight
    down, 180 = straight up). Lower-hemisphere projection plots only the downgoing
    rays directly; upgoing rays are mapped antipodally to the opposite side of the
    sphere with the same polarity sense. Returns (x, y) on a unit-radius circle:
    north = +y, east = +x.
    """
    az = np.deg2rad(np.asarray(azimuth_deg))
    th = np.deg2rad(np.asarray(takeoff_deg))
    # antipodal flip for upgoing rays (takeoff > 90 deg)
    up = th > np.pi / 2
    th_eff = np.where(up, np.pi - th, th)
    az_eff = np.where(up, az + np.pi, az)
    r = r_unit * np.sqrt(2.0) * np.sin(th_eff / 2.0)
    return r * np.sin(az_eff), r * np.cos(az_eff)


def plot_custom_beachball(cfg, event_id, velmodel=None, ax=None,
                          show_polarities=True, show_sp_ratios=True,
                          show_station_labels=False, polarity_min_weight=0.0,
                          label_min_weight=0.7, sp_log_clip=(-2.0, 2.0)):
    """Beachball + per-station polarity markers + S/P amplitude-ratio markers for one event.

    Layered on top of obspy `beach((strike, dip, rake))` (the same renderer used by
    `map_mechanisms`), the function overlays the SKHASH inversion's per-station data so
    you can see WHY a given quality grade was assigned — polarity markers fall (mostly)
    on the right side of each nodal plane for a good fit; S/P ratios cluster around the
    SKHASH-predicted amplitude pattern.

    Polarity markers (filled triangles): up-pointing red for compressional first motion
    (`p_polarity > 0`), down-pointing blue for dilatational. Triangle size is
    proportional to the SKHASH polarity weight `|p_polarity|` (which embeds both the
    PhaseNet+ probability and any SKHASH down-weighting from azimuth-distance gates).

    S/P ratio markers (small offset circles): viridis colormap of `log10(sp_ratio)`,
    clipped to `sp_log_clip` to keep the colorbar legible. Only stations with a
    non-NaN `sp_ratio` get a circle.

    Reads `mechanisms.csv` (event row) and `OUT/out_polinfo.csv` (per-station rows).
    """
    velmodel = velmodel or cfg.fm_velmodel
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 6.6), dpi=120)

    # --- mechanism row -------------------------------------------------------
    mech_path = config.fm_mech_csv(cfg, velmodel)
    if not os.path.exists(mech_path):
        ax.set_title(f"{cfg.region} {event_id}\n(no mechanisms.csv)", fontsize=10)
        ax.set_axis_off()
        return ax.figure
    m_all = _load_mechanisms(mech_path)
    row = m_all[m_all.event_id.astype(str) == str(event_id)]
    if row.empty:
        ax.set_title(f"{cfg.region} {event_id}\n(no mechanism for this event)", fontsize=10)
        ax.set_axis_off()
        return ax.figure
    r = row.iloc[0]
    strike, dip, rake = float(r.strike), float(r.dip), float(r.rake)

    # --- nodal-plane beachball (obspy beach) ---------------------------------
    from obspy.imaging.beachball import beach
    ax.add_collection(beach((strike, dip, rake), xy=(0, 0), width=2.0,
                            facecolor="0.85", edgecolor="k", linewidth=1.2,
                            alpha=0.95, zorder=2))
    # unit circle for the focal sphere outline
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "-", color="k", lw=0.6, zorder=3)

    # --- per-station polarity + S/P overlays ---------------------------------
    pol = _polinfo_for_event(cfg, velmodel, event_id)
    n_pol_drawn = n_sp_drawn = 0
    if not pol.empty:
        x, y = _lower_hemisphere_xy(pol.azimuth.to_numpy(), pol.takeoff.to_numpy())
        pol = pol.assign(_x=x, _y=y)

        if show_polarities:
            up = pol[pol.p_polarity > polarity_min_weight]
            dn = pol[pol.p_polarity < -polarity_min_weight]
            # marker size in points^2 from polarity weight (clip to a visible range)
            def _sz(w):
                return 40 + 220 * np.clip(np.abs(np.asarray(w)), 0.0, 1.0)
            if len(up):
                ax.scatter(up._x, up._y, marker="^", s=_sz(up.p_polarity),
                           facecolor="#d62728", edgecolor="k", linewidth=0.7,
                           alpha=0.95, zorder=5, label=f"Up ({len(up)})")
            if len(dn):
                ax.scatter(dn._x, dn._y, marker="v", s=_sz(dn.p_polarity),
                           facecolor="#1f77b4", edgecolor="k", linewidth=0.7,
                           alpha=0.95, zorder=5, label=f"Down ({len(dn)})")
            n_pol_drawn = len(up) + len(dn)

        if show_sp_ratios and "sp_ratio" in pol.columns:
            sp = pol[pol.sp_ratio.notna() & (pol.sp_ratio > 0)].copy()
            if len(sp):
                sp_log = np.log10(sp.sp_ratio.to_numpy())
                # Diverging colormap centred at 0 (log10(S/P) = 0 means S amp == P amp).
                # Auto-range to the actual data's symmetric extent so the contrast IS visible
                # instead of getting compressed by a fixed clip. (v1.1.4 had a fixed ±2 clip
                # but real chungju/sangju S/P log values span only -0.3 to +0.8, which made
                # every marker look identical in the middle of a viridis ramp.)
                vmax = max(0.05, float(np.percentile(np.abs(sp_log), 95)))
                # honour the user's clip kwarg as an OUTER bound (so callers can still cap
                # outliers if they want)
                vmax = min(vmax, max(abs(sp_log_clip[0]), abs(sp_log_clip[1])))
                norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
                ox, oy = 0.07, -0.07         # offset so the S/P marker doesn't sit on the triangle
                sc = ax.scatter(sp._x + ox, sp._y + oy, marker="o",
                                c=np.clip(sp_log, -vmax, vmax), cmap="RdBu_r", norm=norm,
                                s=55, edgecolor="k", linewidth=0.5, alpha=0.95, zorder=6)
                cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, location="right")
                cb.set_label("log$_{10}$ (S/P amplitude)", fontsize=8)
                cb.ax.tick_params(labelsize=8)
                n_sp_drawn = len(sp)

        if show_station_labels:
            # Only label the high-confidence picks (|p_polarity| >= label_min_weight) so the
            # rim stays legible on dense networks. Place labels just outside the unit circle
            # along the station's effective azimuth (antipodal-flipped for upgoing rays).
            labelled = pol[np.abs(pol.p_polarity) >= label_min_weight]
            for _, p in labelled.iterrows():
                az_eff = np.deg2rad(float(p.azimuth)) + (np.pi if float(p.takeoff) > 90 else 0.0)
                lx, ly = 1.09 * np.sin(az_eff), 1.09 * np.cos(az_eff)
                rot = -np.degrees(az_eff)                          # tangent to the circle
                if rot < -90: rot += 180
                if rot > 90: rot -= 180
                ax.text(lx, ly, p.station, ha="center", va="center",
                        fontsize=5.5, color="0.3", rotation=rot, zorder=7)

    # --- frame + title -------------------------------------------------------
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal", "box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    # cardinal labels
    for txt, xy in (("N", (0, 1.08)), ("E", (1.08, 0)), ("S", (0, -1.08)), ("W", (-1.08, 0))):
        ax.text(*xy, txt, ha="center", va="center", fontsize=9, color="0.3")

    mag = float(r.magnitude) if "magnitude" in r and pd.notna(r.magnitude) else float("nan")
    pol_misfit = float(r.polarity_misfit) if "polarity_misfit" in r and pd.notna(r.polarity_misfit) else float("nan")
    sp_misfit = float(r.sp_misfit) if "sp_misfit" in r and pd.notna(r.sp_misfit) else float("nan")
    title = (f"{cfg.region} {event_id}   M{mag:.1f}   grade {r.quality}\n"
             f"strike/dip/rake = {strike:.0f}°/{dip:.0f}°/{rake:.0f}°   "
             f"pol misfit {pol_misfit:.1f}%   sp misfit {sp_misfit:.1f}\n"
             f"polarities {n_pol_drawn}   s/p ratios {n_sp_drawn}")
    ax.set_title(title, fontsize=9)
    if show_polarities and n_pol_drawn:
        ax.legend(loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(-0.02, 1.0))
    return ax.figure


def _fault_ref(cfg, velmodel=None, mech_select="highest_quality"):
    """Reference (strike, dip, rake, cuspid, magnitude) for the fault sections + 3D plane.

    `mech_select` controls how the reference event is chosen when there's more than one
    candidate mechanism:

      - ``"highest_quality"`` (default): pick the **best-graded** mechanism first
        (A → B → C → D); within the best-available grade, the largest-magnitude wins.
        So a grade-A M2.2 beats a grade-B M3.5 — quality always wins over magnitude.
        This matches the intuition that lower-grade solutions are less reliable
        regardless of source size.
      - ``"largest_magnitude"``: legacy v1.3.1 behaviour. Within
        `cfg.fm_quality_keep` (typically A+B as a unified pool), pick the largest
        magnitude; falls back to all-quality if the pool is empty. Use this to
        reproduce a pre-v1.3.2 view, or when you specifically want the mainshock
        regardless of its grade.

    Returns ``None`` if there are no mechanisms."""
    velmodel = velmodel or cfg.fm_velmodel
    path = config.fm_mech_csv(cfg, velmodel)
    if not os.path.exists(path):
        return None
    m = _load_mechanisms(path)
    if not len(m):
        return None

    if mech_select == "highest_quality":
        # Try A, then B, then C, then D (or whatever order SKHASH grades exist in).
        # Within the chosen grade, largest magnitude wins.
        for grade in ("A", "B", "C", "D"):
            pool = m[m.quality == grade]
            if len(pool):
                break
        else:
            pool = m                                    # no graded events at all; take any
    elif mech_select == "largest_magnitude":
        hi = m[m.quality.isin(list(cfg.fm_quality_keep))]
        pool = hi if len(hi) else m
    else:
        raise ValueError(f"unknown mech_select={mech_select!r} "
                         f"(expected 'highest_quality' or 'largest_magnitude')")

    if not len(pool):
        return None
    if "magnitude" in pool.columns and pool["magnitude"].notna().any() and pool["magnitude"].max() > 0:
        r = pool.sort_values("magnitude", ascending=False).iloc[0]
    else:
        r = pool.iloc[0]                                    # best ordering (mechanisms.csv is sorted)
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


def _mechanism_plane(cfg, velmodel, svd_strike=None, mech_select="highest_quality"):
    """Strike/dip of the fault plane from the reference mechanism. Reference selection
    rule is controlled by ``mech_select`` (see `_fault_ref`); default `"highest_quality"`
    picks the best-graded mechanism first (A → B → C → D), largest magnitude as the
    in-grade tiebreaker.

    A focal mechanism has two nodal planes (NP1 from SKHASH + the conjugate NP2 via obspy
    `aux_plane`). Either could be the actual rupture surface — disambiguation usually comes
    from the **aftershock spatial distribution**: the plane whose strike matches the cloud's
    elongation is the one we plot as the fault. That's the SVD strike of the relocated cloud
    (`_best_fit_plane`). For clusters elongated enough for SVD to be meaningful (~2:1 aspect
    or more, e.g. sangju's 360 × 140 m 2019 swarm), this picks the correct plane; for nearly
    round clouds the choice is mathematically arbitrary anyway and SVD's pick is harmless.

    If you've reasoned out the rupture plane from independent evidence (regional stress field,
    aftershock migration timing, geological mapping), pass explicit `strike=`/`dip=` via
    `fault_sections(..., strike=, dip=)` — that wins over this heuristic.

    Returns (strike, dip, source_str) or None if no mechanism is available.
    """
    ref = _fault_ref(cfg, velmodel, mech_select=mech_select)
    if ref is None:
        return None
    from obspy.imaging.beachball import aux_plane
    s1, d1, r1 = ref["strike"], ref["dip"], ref["rake"]
    s2, d2, _ = aux_plane(s1, d1, r1)
    if svd_strike is None:
        # No cloud SVD to compare against -- default to NP1, the SKHASH-reported plane.
        return s1, d1, f"NP1 of M{ref['mag']:.1f} {ref['quality']} mechanism"
    # Circular distance mod 180° (a strike line is bidirectional, so 180° rotations are equivalent).
    def _dstr(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)
    use_np2 = _dstr(s2, svd_strike) < _dstr(s1, svd_strike)
    label = "NP2 (aux)" if use_np2 else "NP1"
    s, d = (s2, d2) if use_np2 else (s1, d1)
    return s, d, f"{label} of M{ref['mag']:.1f} {ref['quality']} mechanism"


def fault_sections(cfg, velmodel=None, strike=None, dip=None, color_by="time",
                   frame_from="auto", mech_select="highest_quality"):
    """Relocated seismicity in fault coordinates — a 2×2 figure styled after the original dt.cc
    notebooks: (1) fault-plane map view, (2) along-strike depth section, (3) across-strike depth
    section (with the dashed fault-dip line), (4) fault-plane (along-dip) view.

    Orientation precedence:
      1. Explicit `strike` + `dip` args (always win).
      2. `frame_from` controls the automatic choice:
         - "auto" (default): use the mainshock's nodal plane (NP1 or its auxiliary, whichever
           is closer to the SVD strike — i.e. the plane that matches the seismicity elongation)
           when a high-confidence mechanism is available; otherwise fall back to SVD.
         - "svd": always use the SVD best-fit plane (the v1.1.1 default behavior).
         - "mechanism": always use the mainshock's nodal plane (raises if no mechanism).

    `mech_select` controls which mechanism is treated as the "reference" — see `_fault_ref`.
    Default `"highest_quality"` prefers a small grade-A over a larger grade-B; pass
    `"largest_magnitude"` to fall back to v1.3.1 behaviour.

    Markers coloured by origin time, sized by magnitude. Reads the headline dt.cc relocation
    (dt.ct fallback).
    """
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
    ndrop = int(d.id.isin(_boot_underconstrained(cfg, branch)).sum())   # under-constrained -> drop
    d = d[~d.id.isin(_boot_underconstrained(cfg, branch))].reset_index(drop=True)
    if not len(d):
        axes[0].set_title(f"{cfg.region}: empty reloc"); return fig

    ref = _fault_ref(cfg, velmodel, mech_select=mech_select)  # focal mechanism — for overlay/comparison

    # --- orientation: explicit args > frame_from selection > sensible fallback ---
    svd_strike = svd_dip = None
    if len(d) >= 3:
        svd_strike, svd_dip = _best_fit_plane(d.x, d.y, d.z)
    mech_plane = _mechanism_plane(cfg, velmodel, svd_strike=svd_strike,
                                  mech_select=mech_select) if ref else None
    if strike is not None and dip is not None:
        used_strike, used_dip = strike, dip
        used_source = "explicit args"
    elif frame_from == "svd":
        used_strike = svd_strike if svd_strike is not None else (ref["strike"] if ref else 0.0)
        used_dip = svd_dip if svd_dip is not None else (ref["dip"] if ref else 90.0)
        used_source = "SVD best-fit plane"
    elif frame_from == "mechanism":
        if mech_plane is None:
            raise RuntimeError("frame_from='mechanism' but no focal mechanism available")
        used_strike, used_dip, used_source = mech_plane
    else:                                     # "auto"
        if mech_plane is not None:
            used_strike, used_dip, used_source = mech_plane
        elif svd_strike is not None:
            used_strike, used_dip, used_source = svd_strike, svd_dip, "SVD best-fit plane"
        else:
            used_strike, used_dip, used_source = 0.0, 90.0, "default (N-S vertical)"

    # --- centre on the reference event ---
    # frame_from="mechanism" REQUIRES the mainshock hypocenter as the centring point: the
    # nodal plane is defined to pass through the mainshock, not through the cloud centroid.
    # For "svd"/"auto", we prefer the mainshock row when available (so the SVD plane gets
    # anchored where the inversion's strongest signal lives), fall back to the largest-magnitude
    # row, then to the cloud centroid.
    refrow = d[d.id == ref["cuspid"]] if ref else d.iloc[0:0]
    if used_source.startswith(("NP1", "NP2")) and not len(refrow):
        raise RuntimeError(
            f"frame_from='mechanism' requires the mainshock (cuspid {ref['cuspid']}) to be in "
            f"the HypoDD .reloc, but it isn't -- HypoDD's clustering may have dropped it. "
            f"Re-run with frame_from='svd' or pass explicit strike/dip.")
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
        v_e = np.array([1.0, 0.0, 0.0]); v_n = np.array([0.0, 1.0, 0.0])
        sig_al, sig_ac, sig_dp, sig_ad, sig_e, sig_n = (np.full(len(d), np.nan) for _ in range(6))
        for i, e in enumerate(d.id.astype(int)):
            if e in boot:
                s = boot[e]
                sig_al[i], sig_ac[i] = _pct_hw(s, v_al) / 1000.0, _pct_hw(s, v_ac) / 1000.0
                sig_dp[i], sig_ad[i] = _pct_hw(s, v_z) / 1000.0, _pct_hw(s, v_ad) / 1000.0
                sig_e[i], sig_n[i] = _pct_hw(s, v_e) / 1000.0, _pct_hw(s, v_n) / 1000.0

    # --- colour (origin time by default) + KMA-magnitude-scaled hollow markers (reloc mag is 0) ---
    mag = _mag_for(cfg, d.id)                      # KMA local magnitude per event (NaN if unmatched)
    sz = _mag_size(mag, smin=25, smax=1500)
    if color_by == "time" and "time" in d and d.time.notna().any():
        cv = np.array(mdates.date2num([t.datetime for t in d.time]))
        norm = mcolors.Normalize(vmin=cv.min(), vmax=cv.max() if cv.max() > cv.min() else cv.min() + 1)
        cmap = plt.get_cmap("coolwarm"); cbar_label = "Origin time"
    elif color_by == "mag":
        cv = np.nan_to_num(mag, nan=float(np.nanmedian(mag)) if np.isfinite(mag).any() else 1.0)
        norm = mcolors.Normalize(vmin=cv.min(), vmax=max(cv.max(), cv.min() + 0.1))
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
    if boot is not None:
        ax.errorbar(rx / 1000.0, ry / 1000.0, xerr=sig_e, yerr=sig_n, fmt="none", ecolor="0.55",
                    elinewidth=0.6, capsize=1.5, zorder=3)
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

    # panel 3 — across-strike depth section (B–B'), with the dashed fault-dip line. Draw the trace
    # of the CHOSEN fault plane (from used_strike, used_dip — mechanism or explicit kwarg) in this
    # section, anchored at the centring point (across = 0, depth = 0 = reference event). Previously
    # this used the SVD normal, which made the drawn slope inconsistent with the `Dip N°` label
    # whenever the mechanism plane disagreed with the SVD plane.
    ax = axes[2]
    if boot is not None:
        ax.errorbar(across, dep, xerr=sig_ac, yerr=sig_dp, fmt="none", ecolor="0.55",
                    elinewidth=0.6, capsize=1.5, zorder=3)
    ax.scatter(across, dep, s=sz, facecolors="none", edgecolors=rgba, linewidth=1.8, zorder=4)
    # In the (across, depth) section the chosen plane's normal projects to
    #   (n_across, n_down) = (-sin(dip), -cos(dip))
    # (derivation: a plane with strike S and dip δ has normal (cos S sin δ, -sin S sin δ, -cos δ)
    #  in (E, N, +down). The fault-frame rotation by (90°-S) takes that to (0, -sin δ, -cos δ) —
    #  zero along strike by construction, hence the 2D slope = -tan(dip) in (across, depth).)
    # The plane passes through the centring point (across=0, depth=0) since the relative
    # coordinates `across`, `dep` are measured from the reference event.
    dip_rad = np.deg2rad(used_dip)
    xx = np.linspace(-R, R, 50)
    if abs(np.cos(dip_rad)) > 1e-6:               # non-vertical plane: depth = -tan(dip) * across
        ax.plot(xx, -np.tan(dip_rad) * xx, color="k", lw=1.0, ls="--", zorder=1,
                label=f"Dip {used_dip:.0f}°")
    else:                                          # vertical plane
        ax.axvline(0.0, color="k", lw=1.0, ls="--", zorder=1, label=f"Dip {used_dip:.0f}°")
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

    fmtxt = (f"; mechanism {ref['strike']:.0f}°/{ref['dip']:.0f}° ({ref['quality']})"
             if ref else "")
    btxt = ("  (bars = 95% bootstrap" + (f"; {ndrop} under-constrained dropped)" if ndrop else ")")
            if boot is not None else "")
    fig.suptitle(f"{cfg.region} — relocated seismicity in fault coordinates ({branch}){btxt}\n"
                 f"strike {used_strike:.0f}°, dip {used_dip:.0f}° [{used_source}]{fmtxt}",
                 fontsize=13)
    return fig


def compare_epicenters(cfg, velmodel="kim1983", variant="default"):
    """Side-by-side epicenter maps: dt.ct (left) vs dt.cc (right) HypoDD relocations."""
    ct = sumio.read_reloc(os.path.join(config.dtct_dir(cfg), "hypoDD.reloc"))
    cc_path = os.path.join(config.dtcc_dir(cfg),
                           "hypoDD.reloc" if variant == "default" else f"{variant}/hypoDD.reloc")
    cc = sumio.read_reloc(cc_path)
    if variant == "default":                     # drop bootstrap-flagged under-constrained events
        ct = ct[~ct.id.isin(_boot_underconstrained(cfg, "dt.ct"))]
        cc = cc[~cc.id.isin(_boot_underconstrained(cfg, "dt.cc"))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    for ax, df, lab, br in ((a1, ct, "dt.ct", "dt.ct"), (a2, cc, f"dt.cc:{variant}", "dt.cc")):
        if len(df):
            sc = ax.scatter(df.lon, df.lat, c=df.depth, s=_mag_size(_mag_for(cfg, df.id)),
                            cmap="viridis_r", edgecolor="k", zorder=3)   # circle area ∝ M_L
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


def location_table(cfg, branch=None, save=True):
    """Final relocation table — locations + bootstrap 95% errors, one neat row per event. This is the
    headline deliverable: the relocated hypocentres with their data-driven uncertainty.

    Columns: `event_id` (UTC), `origin_time`, `latitude`, `longitude`, `depth_km`, `magnitude` (KMA
    local), `ex95_m`/`ey95_m`/`ez95_m` (bootstrap 95% half-widths, E/N/Z metres), `n_boot` (replicas the
    event relocated in), `cc_links`/`ct_links` (HypoDD inter-event links), and `under_constrained` (the
    bootstrap-flagged events the plots drop — `_boot_underconstrained`: horizontal 95% half-width >
    `BOOT_DROP_HORIZ_KM` km, or `n_boot` < `BOOT_DROP_MIN_NBOOT_FRAC`·n). Reads the headline dt.cc reloc
    (dt.ct fallback) unless `branch` ('dtcc'|'dtct') is given; merges the cached `bootstrap_errors.csv`
    when present (NaN errors otherwise). When `save`, writes `<branch dir>/final_locations.csv`. Returns
    the DataFrame sorted by origin time. Empty frame if there is no reloc."""
    if branch is None:
        path, lab = _reloc_path(cfg)
        branch = "dtcc" if "cc" in lab else "dtct"
    bdir = config.dtcc_dir(cfg) if "cc" in branch else config.dtct_dir(cfg)
    path = os.path.join(bdir, "hypoDD.reloc")
    if not os.path.exists(path):
        return pd.DataFrame()
    d = sumio.read_reloc(path)
    if not len(d):
        return pd.DataFrame()
    eids = _cuspid_event_ids(cfg)
    mags = _event_magnitudes(cfg)
    drop = _boot_underconstrained(cfg, "dt.cc" if "cc" in branch else "dt.ct")
    bpath = os.path.join(bdir, "bootstrap_errors.csv")
    bmap = {}
    if os.path.exists(bpath):
        bmap = {int(r.id): r for r in pd.read_csv(bpath, comment="#").itertuples()}
    rows = []
    for r in d.itertuples():
        cid = int(r.id)
        b = bmap.get(cid)
        be = (lambda v: round(float(v), 1) if b is not None and np.isfinite(v) else np.nan)
        rows.append(dict(
            event_id=eids.get(cid, str(cid)),
            origin_time=(r.time.strftime("%Y-%m-%d %H:%M:%S")
                         if hasattr(r.time, "strftime") else str(r.time)),
            latitude=round(float(r.lat), 5), longitude=round(float(r.lon), 5),
            depth_km=round(float(r.depth), 3),
            magnitude=round(float(mags[cid]), 1) if cid in mags else np.nan,
            ex95_m=be(b.ex95) if b is not None else np.nan,
            ey95_m=be(b.ey95) if b is not None else np.nan,
            ez95_m=be(b.ez95) if b is not None else np.nan,
            n_boot=int(b.n_boot) if b is not None else 0,
            cc_links=int(r.nccp + r.nccs), ct_links=int(r.nctp + r.ncts),
            under_constrained=bool(cid in drop)))
    out = pd.DataFrame(rows).sort_values("origin_time").reset_index(drop=True)
    if save:
        try:
            out.to_csv(os.path.join(bdir, "final_locations.csv"), index=False)
        except Exception:                        # noqa: BLE001 — never let saving break display
            pass
    return out


def _ellipsoid_points(center, samples, ngrid=14):
    """Surface points of the 95% bootstrap error ellipsoid for one event (fed to a plotly Mesh3d with a
    convex hull). The ellipsoid's shape is the sample **covariance** and its size the empirical **95%
    Mahalanobis radius** (95% of the bootstrap samples fall inside — consistent with the percentile error
    bars). `center` and `samples` are in the plotted (E, N, depth) km frame; the sample mean offset is
    removed so the ellipsoid sits on the plotted hypocentre. Returns an (N, 3) point cloud, or None if the
    covariance is degenerate (too few / collinear samples)."""
    s = np.asarray(samples, float)
    s = s - np.median(s, axis=0)
    if len(s) < 4:
        return None
    cov = np.cov(s.T)
    if not np.all(np.isfinite(cov)):
        return None
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-18, None)
        d2 = np.einsum("ij,jk,ik->i", s, np.linalg.inv(cov), s)
        r = float(np.sqrt(np.percentile(d2, 95.0)))
    except np.linalg.LinAlgError:
        return None
    u = np.linspace(0, 2 * np.pi, ngrid); v = np.linspace(0, np.pi, ngrid)
    uu, vv = np.meshgrid(u, v)
    sph = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], -1).reshape(-1, 3)
    return (sph * (r * np.sqrt(vals))) @ vecs.T + np.asarray(center, float)


def plot_3d_plane(cfg, velmodel=None, color_by="time", error="bars", frame_from="auto",
                  mech_select="highest_quality"):
    """Interactive 3-D view (**plotly**) of the dt.cc-relocated hypocentres with a fault plane
    overlaid as a translucent patch — rotate/zoom in a notebook. Returns a plotly Figure.

    Hypocentres: relative E–N–depth (km), sized by magnitude. `frame_from` controls the plane:
      - "auto" (default): mainshock's nodal plane through the mainshock hypocentre when a
        high-confidence mechanism exists; otherwise the SVD plane through the cloud centroid.
      - "svd": SVD best-fit plane through the cloud centroid (the v1.1.1 default).
      - "mechanism": mainshock's nodal plane through the mainshock hypocentre (raises if no
        mechanism, or the mainshock isn't in the reloc).

    `mech_select` chooses which mechanism counts as "the mainshock" — see `_fault_ref`.
    Default `"highest_quality"` prefers the best-graded mechanism over the largest one.

    NP1/NP2 disambiguation: the plane whose strike matches the SVD strike better is selected
    as "the fault" (the other is the auxiliary). Depth axis points down.
    `error`: `"bars"` / `"ellipsoid"` / `"none"`. Reads the headline dt.cc reloc."""
    import plotly.graph_objects as go
    import matplotlib.dates as mdates
    velmodel = velmodel or cfg.fm_velmodel
    reloc, branch = _reloc_path(cfg)
    d = sumio.read_reloc(reloc) if os.path.exists(reloc) else pd.DataFrame()
    if len(d):
        d = d[~d.id.isin(_boot_underconstrained(cfg, branch))].reset_index(drop=True)   # drop under-constrained
    if not len(d):
        return go.Figure().update_layout(
            title=f"{cfg.region}: no HypoDD reloc (run ph2dt→dtcc first)")

    # --- centring: prefer the mainshock hypocentre (required for frame_from='mechanism'),
    # fall back to the cloud centroid for SVD / when no mainshock row is in the reloc.
    ref = _fault_ref(cfg, velmodel, mech_select=mech_select)
    refrow = d[d.id == ref["cuspid"]] if ref else d.iloc[0:0]
    if frame_from == "mechanism" and not len(refrow):
        raise RuntimeError(
            f"frame_from='mechanism' requires the mainshock (cuspid {ref['cuspid']}) to be in "
            f"the HypoDD .reloc, but it isn't -- run with frame_from='svd' instead.")
    if (frame_from in ("mechanism", "auto")) and len(refrow):
        x0 = float(refrow.iloc[0].x); y0 = float(refrow.iloc[0].y); z0 = float(refrow.iloc[0].z)
    else:
        x0, y0, z0 = float(d.x.mean()), float(d.y.mean()), float(d.z.mean())
    E = (d.x - x0).to_numpy() / 1000.0
    N = (d.y - y0).to_numpy() / 1000.0
    dep = (d.z - z0).to_numpy() / 1000.0                        # km, +down (Z is positive down)
    mag = _mag_for(cfg, d.id)                                   # KMA local magnitude (reloc mag is 0)
    mfill = np.where(np.isfinite(mag), mag, np.nanmedian(mag) if np.isfinite(mag).any() else 1.0)
    size = np.clip(4 + 4 * mfill, 3, 22)                        # plotly marker px, grows with M_L

    if color_by == "time" and "time" in d and d.time.notna().any():
        cvals = np.array(mdates.date2num([t.datetime for t in d.time]))
        tv = np.linspace(cvals.min(), cvals.max(), 5)
        cbar = dict(title="Origin time", tickvals=list(tv),
                    ticktext=[mdates.num2date(v).strftime("%Y-%m-%d") for v in tv])
        cscale = "RdBu_r"
    else:
        cvals, cbar, cscale = d.depth.to_numpy(), dict(title="Depth (km)"), "Viridis_r"

    boot = _load_bootstrap(cfg, branch)            # 95% bootstrap uncertainty in E/N/depth (percentile)
    err = {}
    if boot and error == "bars":                   # error_x/y/z whiskers on the markers
        _hw = {int(i): (_pct_hw(boot[int(i)]) / 1000.0 if int(i) in boot else None) for i in d.id}
        def _e(j):
            return [_hw[int(i)][j] if _hw.get(int(i)) is not None else 0.0 for i in d.id]
        err = dict(error_x=dict(type="data", array=_e(0), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"),
                   error_y=dict(type="data", array=_e(1), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"),
                   error_z=dict(type="data", array=_e(2), thickness=0.8, width=2, color="rgba(80,80,80,0.5)"))
    data = [go.Scatter3d(x=E, y=N, z=dep, mode="markers", name="Hypocentres",
                         marker=dict(size=size, color=cvals, colorscale=cscale, colorbar=cbar,
                                     line=dict(width=0.5, color="black"), opacity=0.95),
                         text=[f"{int(i)}  M{m:.1f}" for i, m in zip(d.id, mfill)], **err)]
    title = f"{cfg.region} — 3-D relocated seismicity ({branch})"
    if boot and error == "ellipsoid":              # one translucent 95% ellipsoid per event, marker-coloured
        import matplotlib.colors as _mc
        mcmap = plt.get_cmap("coolwarm" if color_by == "time" else "viridis_r")
        lo, hi = float(np.min(cvals)), float(np.max(cvals))
        mnorm = _mc.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        first = True
        for i, cid in enumerate(d.id.astype(int)):
            if cid not in boot:
                continue
            pts = _ellipsoid_points((E[i], N[i], dep[i]), boot[cid] / 1000.0)
            if pts is None:
                continue
            rr, gg, bb, _ = mcmap(mnorm(cvals[i]))
            data.append(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], alphahull=0,
                                  color=f"rgb({int(255*rr)},{int(255*gg)},{int(255*bb)})", opacity=0.18,
                                  hoverinfo="skip", showscale=False, flatshading=True,
                                  name="95% bootstrap ellipsoid", showlegend=first))
            first = False
    if boot:
        title += "  (ellipsoids = 95% bootstrap)" if error == "ellipsoid" else (
            "  (bars = 95% bootstrap)" if error == "bars" else "")
    if len(d) >= 3:
        svd_strike, svd_dip = _best_fit_plane(d.x, d.y, d.z)
        mech_plane = _mechanism_plane(cfg, velmodel, svd_strike=svd_strike,
                                      mech_select=mech_select) if ref else None
        # plane orientation per frame_from
        if frame_from == "svd" or (frame_from == "auto" and mech_plane is None):
            strike, dip = svd_strike, svd_dip
            plane_source = "SVD best-fit plane"
            # Use the SVD normal directly from the same (E, N, dep) frame — no round-trip.
            n = _svd_normal(E, N, dep)
        else:                                               # "auto" with mech / "mechanism"
            if mech_plane is None:
                raise RuntimeError("frame_from='mechanism' but no focal mechanism available")
            strike, dip, plane_source = mech_plane
            # Build the plane normal in the (E, N, +down) frame from strike/dip. Convention:
            # strike measured from N clockwise; dip is rotation of the plane about strike
            # (positive dip tilts the dip-direction edge downward).
            s = np.deg2rad(strike); dipr = np.deg2rad(dip)
            n = np.array([np.cos(s) * np.sin(dipr), -np.sin(s) * np.sin(dipr), -np.cos(dipr)])
        # build two orthonormal in-plane axes from the normal
        u1 = np.cross(n, [0.0, 0.0, 1.0])
        u1 = (np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u1) < 1e-9 else u1 / np.linalg.norm(u1))
        u2 = np.cross(n, u1); u2 = u2 / np.linalg.norm(u2)
        # The plane is anchored at the CENTRING point (E=0, N=0, dep=0 in the plotted frame —
        # which is the mainshock hypocentre when frame_from='auto'/'mechanism' with a mechanism,
        # else the cloud centroid). Span the patch over the data's projection onto (u1, u2).
        c0 = np.array([0.0, 0.0, 0.0])                       # plane passes through plotted origin
        P = np.column_stack((E, N, dep)) - c0
        p1, p2 = P @ u1, P @ u2
        m1 = 0.08 * (np.ptp(p1) or 0.2); m2 = 0.08 * (np.ptp(p2) or 0.2)
        a1lo, a1hi = p1.min() - m1, p1.max() + m1
        a2lo, a2hi = p2.min() - m2, p2.max() + m2
        corners = [c0 + a * u1 + b * u2 for a in (a1lo, a1hi) for b in (a2lo, a2hi)]   # idx 0,1,2,3
        cx = [c[0] for c in corners]; cy = [c[1] for c in corners]; cz = [c[2] for c in corners]
        data.append(go.Mesh3d(x=cx, y=cy, z=cz, i=[0, 0], j=[1, 3], k=[3, 2],
                              opacity=0.3, color="gray", hoverinfo="name",
                              name=f"{plane_source} {strike:.0f}/{dip:.0f}", showscale=False))
        title += f"  +  {plane_source} (strike {strike:.0f}°, dip {dip:.0f}°)"
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
