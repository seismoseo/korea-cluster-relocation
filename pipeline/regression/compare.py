"""
Regression harness: compare freshly-produced framework outputs against the frozen
per-cluster baselines and report per-stage metrics + PASS/FAIL.

Determinism map (see README):
  * picks (P3)      : near-deterministic (PhaseNet on CPU) -> tight match expected
  * .sum  (P4)      : deterministic given identical phases -> small location diffs
  * dt.ct .reloc(P6): HypoDD is relative; absolute anchor may shift rigidly, so we
                      separate the rigid translation from the relative-structure error
  * dt.cc .reloc(P8): judgment-dependent (hand-tuned xcorr) -> report-only soft gates

Every function returns a plain dict so the same numbers render in the CLI and in the
JupyterLab dashboard notebook.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth

from pipeline import config
from pipeline.core import sumio


# --------------------------------------------------------------- small helpers
def _epi_km(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0] / 1000.0


def _enu_m(lat, lon, dep_km, lat0, lon0):
    """Local east/north/down (metres) about (lat0, lon0)."""
    e = np.array([gps2dist_azimuth(lat0, lon0, lat0, lo)[0] * np.sign(lo - lon0) for lo in lon])
    n = np.array([gps2dist_azimuth(lat0, lon0, la, lon0)[0] * np.sign(la - lat0) for la in lat])
    return e, n, np.asarray(dep_km) * 1000.0


# --------------------------------------------------------------------- stations
def compare_stations(cfg) -> dict:
    new = pd.read_csv(config.used_stations_csv(cfg))
    base = pd.read_csv(config.baseline_used_stations(cfg))
    np_ = set(map(tuple, new[["Code", "Sensor"]].values))
    bp = set(map(tuple, base[["Code", "Sensor"]].values))
    return dict(stage="stations", n_new=len(new), n_base=len(base),
                set_equal=bool(np_ == bp),
                only_new=sorted(np_ - bp), only_base=sorted(bp - np_),
                passed=bool(np_ == bp))


# ----------------------------------------------------------------------- picks
def compare_picks(cfg, tol_s=0.1, min_frac=0.95) -> dict:
    base_dir = config.baseline_picks_dir(cfg)
    tot_shared = tot_match = tot_base = 0
    worst = 0.0
    per_event = {}
    for bf in sorted(glob.glob(f"{base_dir}/*_picks.csv")):
        eid = os.path.basename(bf).split("_")[0]
        nf = config.picks_csv(cfg, eid)
        if not os.path.exists(nf):
            per_event[eid] = "no framework picks"
            continue
        new, base = pd.read_csv(nf), pd.read_csv(bf)
        kn = {(r.Station, r.Phase): UTCDateTime(r.Time) for r in new.itertuples()}
        kb = {(r.Station, r.Phase): UTCDateTime(r.Time) for r in base.itertuples()}
        shared = set(kn) & set(kb)
        dts = [abs(kn[k] - kb[k]) for k in shared]
        m = sum(1 for d in dts if d <= tol_s)
        tot_shared += len(shared); tot_match += m; tot_base += len(kb)
        worst = max([worst] + dts)
        per_event[eid] = dict(n_new=len(new), n_base=len(base), shared=len(shared),
                              matched=m, max_dt=round(max(dts + [0]), 4))
    frac = tot_match / tot_shared if tot_shared else 0.0
    repro = tot_match / tot_base if tot_base else 0.0
    return dict(stage="picks", shared=tot_shared, matched=tot_match, n_base=tot_base,
                match_frac=round(frac, 4), reproduced_frac=round(repro, 4),
                max_dt_s=round(worst, 4), per_event=per_event,
                passed=bool(frac >= min_frac and worst <= tol_s + 1e-9 and repro >= min_frac))


# ------------------------------------------------------------------------- .sum
def compare_sum(cfg, velmodel, dh_med=0.5, dh_max=1.0, dz_max=1.0, rms_max=0.05) -> dict:
    new = sumio.read_sum(config.sum_file(cfg, velmodel))
    base = sumio.read_sum(config.baseline_sum(cfg, velmodel))
    m = new.merge(base, on="id", suffixes=("_n", "_b"))
    dh = m.apply(lambda r: _epi_km(r.lat_n, r.lon_n, r.lat_b, r.lon_b), axis=1)
    dz = (m.depth_n - m.depth_b).abs()
    drms = (m.rms_n - m.rms_b).abs()
    passed = (len(new) == len(base) == len(m) and dh.median() < dh_med
              and dh.max() < dh_max and dz.max() < dz_max and drms.max() < rms_max)
    return dict(stage=f"sum:{velmodel}", n_new=len(new), n_base=len(base), matched=len(m),
                dh_med_km=round(dh.median(), 4), dh_max_km=round(dh.max(), 4),
                dz_med_km=round(dz.median(), 4), dz_max_km=round(dz.max(), 4),
                rms_max_s=round(drms.max(), 4), passed=bool(passed))


# ----------------------------------------------------------------------- .reloc
def compare_reloc(cfg, kind="dtct", variant="default",
                  rel_horiz_max=100.0, rel_depth_max=150.0) -> dict:
    """Compare a HypoDD relocation. Separates the rigid translation (absolute anchor)
    from the centroid-aligned relative-structure error, which is the meaningful
    fidelity metric for a relative-relocation method."""
    if kind == "dtct":
        new_p, base_p = config.dtct_dir(cfg) + "/hypoDD.reloc", config.baseline_reloc_dtct(cfg)
    else:
        new_p = os.path.join(config.dtcc_dir(cfg),
                             "hypoDD.reloc" if variant == "default" else f"{variant}/hypoDD.reloc")
        base_p = config.baseline_reloc_dtcc(cfg, variant)
    if not (os.path.exists(new_p) and os.path.exists(base_p)):
        return dict(stage=f"reloc:{kind}:{variant}", error="missing reloc file",
                    new=new_p, base=base_p, passed=False)
    new = sumio.read_reloc(new_p).sort_values("id")
    base = sumio.read_reloc(base_p).sort_values("id")
    if len(new) == 0:
        return dict(stage=f"reloc:{kind}:{variant}", n_new=0, n_base=len(base),
                    note="empty/failed reloc (see hypoDD.sum; e.g. MAXDATA0)", passed=False)
    m = new.merge(base, on="id", suffixes=("_n", "_b"))
    if len(m) == 0:
        return dict(stage=f"reloc:{kind}:{variant}", n_new=len(new), n_base=len(base),
                    matched=0, note="no overlapping event ids", passed=False)
    lat0, lon0 = base.lat.mean(), base.lon.mean()
    en = _enu_m(m.lat_n.values, m.lon_n.values, m.depth_n.values, lat0, lon0)
    eb = _enu_m(m.lat_b.values, m.lon_b.values, m.depth_b.values, lat0, lon0)
    off = [float(np.mean(en[k] - eb[k])) for k in range(3)]            # rigid translation
    res = [(en[k] - eb[k]) - off[k] for k in range(3)]                  # relative residual
    rh = np.hypot(res[0], res[1]); rz = np.abs(res[2])
    # cluster-shape correlation (pairwise inter-event distances)
    def pdist(e, n, z):
        out = []
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                out.append(np.sqrt((e[i]-e[j])**2 + (n[i]-n[j])**2 + (z[i]-z[j])**2))
        return np.array(out)
    corr = float(np.corrcoef(pdist(*en), pdist(*eb))[0, 1]) if len(m) > 2 else 1.0
    passed = (len(new) == len(base) == len(m)
              and rh.max() < rel_horiz_max and rz.max() < rel_depth_max)
    return dict(stage=f"reloc:{kind}:{variant}", n_new=len(new), n_base=len(base), matched=len(m),
                rigid_translation_m=dict(E=round(off[0]), N=round(off[1]), Z=round(off[2])),
                rel_horiz_rms_m=round(float(np.sqrt(np.mean(rh**2))), 1),
                rel_horiz_max_m=round(float(rh.max()), 1),
                rel_depth_rms_m=round(float(np.sqrt(np.mean(rz**2))), 1),
                rel_depth_max_m=round(float(rz.max()), 1),
                shape_corr=round(corr, 4), passed=bool(passed))


# ------------------------------------------------------------- dt.cc content
def _parse_dtcc(path) -> dict:
    """dt.cc_0.7_combined -> {(cuspid1, cuspid2, station, phase): (delay, cc)}.

    Per-line floats are parsed at the fixed columns the writer uses (delay [11:20],
    cc [23:34]); malformed baseline rows (e.g. a stray '-117.9' longitude) are skipped."""
    out, pair = {}, None
    for line in open(path):
        if line.startswith("#"):
            t = line.split()
            pair = (t[1], t[2]) if len(t) >= 3 else None
        elif line.strip() and pair:
            sta, phase = line[0:6].strip(), line.split()[-1]
            try:
                out[(pair[0], pair[1], sta, phase)] = (float(line[11:20]), float(line[23:34]))
            except ValueError:
                pass
    return out


def compare_dtcc_content(cfg, cc_tol=0.05, delay_tol=0.02) -> dict:
    """Report-only: how well the regenerated dt.cc matches the baseline dt.cc, per
    (pair, station, phase). dt.cc is hand-tuned / judgment-dependent, so this is a
    diagnostic (median |Δcc|, |Δdelay|, matched fraction) — never a hard gate."""
    new_p = os.path.join(config.dtcc_dir(cfg), "dt.cc_0.7_combined")
    base_p = os.path.join(cfg.src_root, "2.HypoDD", "02.dt.cc", "dt.cc_0.7_combined")
    if not (os.path.exists(new_p) and os.path.exists(base_p)):
        return dict(stage="dtcc_content", error="missing dt.cc_0.7_combined", passed=None)
    new, base = _parse_dtcc(new_p), _parse_dtcc(base_p)
    common = set(new) & set(base)
    good = [k for k in common if 0.0 <= base[k][1] <= 1.0 and abs(base[k][0]) < 5.0]
    dcc = [abs(new[k][1] - base[k][1]) for k in good]
    ddl = [abs(new[k][0] - base[k][0]) for k in good]
    matched = sum(1 for k in good
                  if abs(new[k][1] - base[k][1]) <= cc_tol and abs(new[k][0] - base[k][0]) <= delay_tol)
    return dict(stage="dtcc_content", n_new=len(new), n_base=len(base),
                shared=len(common), good_shared=len(good), matched=matched,
                match_frac=round(matched / len(good), 4) if good else 0.0,
                med_dcc=round(float(np.median(dcc)), 5) if dcc else None,
                med_ddelay_s=round(float(np.median(ddl)), 5) if ddl else None,
                passed=None)   # report-only


# --------------------------------------------------------------- orchestration
def compare_all(cfg, velmodels=("kim1983", "kim2011"), write=True) -> pd.DataFrame:
    rows = [compare_stations(cfg), compare_picks(cfg)]
    for vm in velmodels:
        try:
            rows.append(compare_sum(cfg, vm))
        except FileNotFoundError:
            pass
    try:
        rows.append(compare_reloc(cfg, "dtct"))
    except FileNotFoundError:
        pass
    # dt.cc (report-only): default reloc + content; no_main only if its baseline exists
    try:
        if os.path.exists(os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc")):
            rows.append(compare_reloc(cfg, "dtcc", "default"))
        if os.path.exists(config.baseline_reloc_dtcc(cfg, "no_main")) and \
                os.path.exists(os.path.join(config.dtcc_dir(cfg), "no_main", "hypoDD.reloc")):
            rows.append(compare_reloc(cfg, "dtcc", "no_main"))
        rows.append(compare_dtcc_content(cfg))
    except (FileNotFoundError, KeyError):
        pass
    # flatten per_event out of the picks row for the table
    flat = []
    for r in rows:
        r = {k: v for k, v in r.items() if k != "per_event"}
        flat.append(r)
    df = pd.DataFrame(flat)
    if write:
        os.makedirs(config.assert_writable(config.regression_dir(cfg)), exist_ok=True)
        df.to_csv(os.path.join(config.regression_dir(cfg), "compare_report.csv"), index=False)
    return df
