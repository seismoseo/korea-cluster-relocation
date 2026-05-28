"""
HypoDD relative relocation backbone (ports `2.HypoDD/00.ph2dt` + `01.dt.ct`).

  prep_ph2dt(cfg)  : copy the HYPOINVERSE .arc (kim1983, matching the dt.ct 1-D model),
                     patch S-phase flag (ES@col46 -> 'N'@col11), ncsn2pha -> .pha (+ _mod
                     copy), build station.dat from <Region>_hyp.sta, write ph2dt.inp.
  run_ph2dt(cfg)   : run ph2dt -> dt.ct, event.dat, event.sel.
  run_dtct(cfg)    : copy ph2dt outputs into 01.dt.ct/, write hypoDD.inp (catalog-only,
                     IDAT=2), run hypoDD -> hypoDD.reloc (+ .loc/.res).

Binaries ncsn2pha, ph2dt, hypoDD must be on PATH.
The dt.cc cross-correlation branch is built in core/xcorr.py + run_dtcc here.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
from glob import glob

from pipeline import config

# dt.cc inter-event distance cutoffs (WDCC/WDCT) are scaled to the cluster size when LSQR is used:
# a cluster of diameter >= DTCC_DIST_REF_KM keeps the configured cutoffs; tighter clusters shrink
# them (so long-distance pairs to spatially peripheral, poorly-linked events are cut and those
# events drop out instead of destabilising the LSQR solution). Never tighter than DTCC_DIST_FMIN x.
DTCC_DIST_REF_KM = 1.5
DTCC_DIST_FMIN = 0.1


# ---------------------------------------------------------------- ph2dt prep
def _patch_arc_es(src_arc, dst_arc):
    """HypoDD wants the S-phase weight flag patched: 'ES' at col 46 -> 'N' at col 11."""
    with open(src_arc) as f:
        lines = f.readlines()
    with open(dst_arc, "w") as f:
        for line in lines:
            if line[46:48] == "ES":
                f.write(line[:11] + "N" + line[12:])
            else:
                f.write(line)


def _fix_pha_longitude(pha_path):
    """ncsn2pha is a USGS NorCal tool that assumes WEST longitude, so it writes
    Korean EAST longitudes as negative (e.g. -127.7273). The original workflow
    flipped the sign by hand; we automate it here by turning the longitude's minus
    into a space (preserving column width) in each '#' event header. Without this,
    every station sits >MAXDIST away and ph2dt produces an empty dt.ct."""
    lines = open(pha_path).readlines()
    with open(pha_path, "w") as f:
        for line in lines:
            if line.startswith("#"):
                lon = line.split()[8]                 # YR MO DY HR MI SC LAT LON ...
                if lon.startswith("-"):
                    line = line.replace("-" + lon[1:], " " + lon[1:], 1)
            f.write(line)


def _sta_to_stationdat(hyp_sta, station_dat):
    """Convert <Region>_hyp.sta (HYPOINVERSE) -> HypoDD station.dat (NETSTA lat lon)."""
    with open(hyp_sta) as f:
        lines = f.readlines()
    with open(station_dat, "w") as f:
        for line in lines:
            net, sta = line[6:8], line[0:5].replace(" ", "")
            lat = round(int(line[15:17]) + float(line[18:25]) / 60, 5)
            lon = round(int(line[26:29]) + float(line[30:37]) / 60, 5)
            f.write(f"{net}{sta} {lat} {lon}\n")


PH2DT_INP = """* ph2dt.inp - input control file for program ph2dt
* Input station file:
station.dat
* Input phase file:
{pha}
*MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS
   {MINWGHT}       {MAXDIST}     {MAXSEP}    {MAXNGH}     {MINLNK}      {MINOBS}     {MAXOBS}
"""


def prep_ph2dt(cfg, velmodel="kim1983"):
    """Build the ph2dt input set from the HYPOINVERSE .arc of `velmodel`."""
    d = config.assert_writable(config.ph2dt_dir(cfg))
    os.makedirs(d, exist_ok=True)
    region = cfg.region

    arc = os.path.join(d, f"{region}.arc")
    shutil.copyfile(config.arc_file(cfg, velmodel), arc)
    _patch_arc_es(arc, os.path.join(d, f"{region}_dd.arc"))

    subprocess.run(["ncsn2pha", f"{region}_dd.arc", f"{region}.pha"],
                   cwd=d, check=True, stdout=subprocess.DEVNULL)
    _fix_pha_longitude(os.path.join(d, f"{region}.pha"))   # East-longitude sign fix
    shutil.copyfile(os.path.join(d, f"{region}.pha"), os.path.join(d, f"{region}_mod.pha"))

    _sta_to_stationdat(config.sta_hyp_file(cfg), os.path.join(d, "station.dat"))

    p = cfg.ph2dt
    with open(os.path.join(d, "ph2dt.inp"), "w") as f:
        f.write(PH2DT_INP.format(pha=f"{region}_mod.pha", MINWGHT=p.MINWGHT,
                                 MAXDIST=p.MAXDIST, MAXSEP=p.MAXSEP, MAXNGH=p.MAXNGH,
                                 MINLNK=p.MINLNK, MINOBS=p.MINOBS, MAXOBS=p.MAXOBS))
    return d


def run_ph2dt(cfg):
    d = config.ph2dt_dir(cfg)
    subprocess.run(["ph2dt", "ph2dt.inp"], cwd=d, check=True, stdout=subprocess.DEVNULL)
    return d  # leaves dt.ct, event.dat, event.sel


# ------------------------------------------------------------- hypoDD.inp
def build_hypodd_inp(inp, iter_sets=None) -> str:
    """Render a hypoDD.inp string from a HypoDDInp. The first (cross-correlation)
    data line is left blank when inp.cc_file is None -> catalog-only relocation.
    `iter_sets` overrides inp.iter_sets (used by the adaptive-damping loop)."""
    # cc and src are read by HypoDD's getinp as filename lines; blank => none.
    cc = inp.cc_file or ""
    src = ""
    rows = inp.iter_sets if iter_sets is None else iter_sets
    iters = "\n".join("    " + "  ".join(str(x) for x in row) for row in rows)
    top = "  ".join(str(x) for x in inp.top)
    vel = "  ".join(str(x) for x in inp.vel)
    return f"""* RELOC.INP
*--- input file selection
* cross correlation diff times:
{cc}
* catalog P diff times:
dt.ct
* event file:
{inp.event_file}
* station file:
station.dat
*--- output file selection
* original locations:
hypoDD.loc
* relocations:
hypoDD.reloc
* station information:
hypoDD.sta
* residual information:
hypoDD.res
* source parameter information:
*hypoDD.src
{src}
*--- data type selection: IDAT IPHA DIST
    {inp.idat}     {inp.ipha}     {inp.dist}
*--- event clustering: OBSCC OBSCT
    {inp.obscc}      {inp.obsct}
*--- solution control: ISTART ISOLV NSET
    {inp.istart}       {inp.isolv}       {len(rows)}
*--- data weighting: NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP
{iters}
*--- 1D model: NLAY RATIO
   {inp.nlay}     {inp.ratio}
* TOP
{top}
* VEL
{vel}
*--- event selection: CID / ID
    0
"""


def _exec_hypodd_once(d):
    """Run hypoDD once in directory `d` (which must already hold hypoDD.inp + inputs),
    capture stdout to hypoDD.sum, archive per-iteration *.reloc.0* into reloc/, and
    guard against an empty reloc (the MAXDATA0 overflow). Returns the hypoDD.reloc path."""
    os.makedirs(os.path.join(d, "reloc"), exist_ok=True)
    proc = subprocess.run(["hypoDD", "hypoDD.inp"], cwd=d,
                          capture_output=True, text=True)
    with open(os.path.join(d, "hypoDD.sum"), "w") as out:
        out.write(proc.stdout)
    for f in glob(os.path.join(d, "*.reloc.0*")):           # like the baseline hypoDD.sh
        shutil.move(f, os.path.join(d, "reloc", os.path.basename(f)))
    reloc = os.path.join(d, "hypoDD.reloc")
    if not os.path.exists(reloc) or os.path.getsize(reloc) == 0:
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(
            "hypoDD produced no relocations. Last output:\n  " + "\n  ".join(tail)
            + "\n(A 'STOP >>> Increase MAXDATA0' means the dt data exceeds the binary's "
            "compiled SVD array limit — switch that cluster to LSQR (isolv=2), recompile "
            "hypoDD with a larger hypoDD.inc, or relocate a tighter sub-cluster.)")
    return reloc


_CND_RE = re.compile(r"acond \(CND\)=\s*([0-9.]+)")
_KV_RE = re.compile(r"([a-z_]+)=\s*(-?[0-9.]+)")


def _parse_iteration_cnds(log_path):
    """Pair each LSQR iteration's weighting signature with its condition number, from hypoDD.log.

    Each iteration prints a `Weighting parameters for this iteration:` block (wt_ccp/wt_ccs/maxr_cc/
    maxd_cc, wt_ctp/wt_cts/maxr_ct/maxd_ct, damp) followed by `... acond (CND)= <value>`. Returns a
    list of (signature, cnd) where signature = (wt_ccp,wt_ccs,maxr_cc,maxd_cc,wt_ctp,wt_cts,maxr_ct,
    maxd_ct) — enough to match each iteration to its iter_sets row regardless of NITER bookkeeping."""
    out, cur = [], None
    if not os.path.exists(log_path):
        return out
    for line in open(log_path, errors="ignore"):
        if "Weighting parameters for this iteration" in line:
            cur = {}
        elif cur is not None and ("wt_ccp=" in line or "wt_ctp=" in line):
            cur.update({k: float(v) for k, v in _KV_RE.findall(line)})
        elif cur is not None and "acond (CND)=" in line:
            m = _CND_RE.search(line)
            if m:
                sig = tuple(cur.get(k) for k in ("wt_ccp", "wt_ccs", "maxr_cc", "maxd_cc",
                                                 "wt_ctp", "wt_cts", "maxr_ct", "maxd_ct"))
                out.append((sig, float(m.group(1))))
            cur = None
    return out


def _max_cnd_per_set(log_path, iter_sets, tol=0.02):
    """Worst (max) condition number per weighting set, matching each logged iteration to its
    iter_sets row by weighting signature (row cols 1..8 = WTCCP..WDCT). Returns {row_idx: max_cnd}."""
    res = {}
    for sig, cnd in _parse_iteration_cnds(log_path):
        if any(v is None for v in sig):
            continue
        for i, row in enumerate(iter_sets):
            rsig = [float(x) for x in row[1:9]]
            if all(abs(a - b) <= tol + tol * abs(b) for a, b in zip(sig, rsig)):
                res[i] = max(res.get(i, 0.0), cnd)
                break
    return res


def _cluster_diameter_km(cfg):
    """Characteristic cluster diameter (km) from the dt.ct relocation = 2 x the 95th-percentile
    epicentral distance from the centroid. Used to size the dt.cc inter-event distance cutoffs.
    Returns None if there is no dt.ct relocation yet."""
    import numpy as np
    from pipeline.core import sumio
    p = os.path.join(config.dtct_dir(cfg), "hypoDD.reloc")
    if not os.path.exists(p):
        return None
    d = sumio.read_reloc(p)
    if not len(d):
        return None
    r = np.sqrt((d.x - d.x.mean()) ** 2 + (d.y - d.y.mean()) ** 2) / 1000.0
    return 2.0 * float(np.percentile(r, 95))


def _scale_distance_cutoffs(iter_sets, cluster_km, ref_km=DTCC_DIST_REF_KM, fmin=DTCC_DIST_FMIN):
    """Scale each weighting set's inter-event distance cutoffs (WDCC = col 4, WDCT = col 8) to the
    cluster size: factor f = clamp(cluster_km / ref_km, fmin, 1). A '-9' (no cutoff) stays '-9'.
    Returns (scaled_iter_sets, f)."""
    f = min(1.0, max(fmin, cluster_km / ref_km))
    out = []
    for row in iter_sets:
        row = list(row)
        for col in (4, 8):
            if row[col] != -9:
                row[col] = round(row[col] * f, 2)
        out.append(tuple(row))
    return tuple(out), f


def _exec_hypodd(d, inp, adapt_damping=False, cnd_range=(40.0, 80.0), max_attempts=12):
    """Run hypoDD in `d`, writing hypoDD.inp from `inp`. For LSQR runs with `adapt_damping`, modulate
    each weighting set's DAMP so its condition number (CND) lands in `cnd_range` — the HypoDD-
    recommended ~40–80 band. SVD (isolv=1) has no damping to tune, so it just runs once.

    Why: with LSQR (forced when the dt data exceeds the SVD MAXDATA0 limit) a too-small DAMP leaves
    the system ill-conditioned (CND ≫ 80); poorly-linked events (e.g. those with no cross-correlation
    data) then take wild, unstable steps. Higher DAMP lowers CND and stabilises them. HypoDD is cheap
    for these clusters, so we iterate: run → read CND per set → nudge DAMP toward the band → re-run."""
    if not (adapt_damping and getattr(inp, "isolv", 1) == 2):
        with open(os.path.join(d, "hypoDD.inp"), "w") as f:
            f.write(build_hypodd_inp(inp))
        return _exec_hypodd_once(d)

    lo, hi = cnd_range
    mid = (lo + hi) / 2.0
    sets = [list(r) for r in inp.iter_sets]                 # mutable working copy (DAMP = col -1)
    int_damp = [isinstance(r[-1], int) for r in inp.iter_sets]
    best, history, reloc = None, [], None
    for attempt in range(max_attempts):
        with open(os.path.join(d, "hypoDD.inp"), "w") as f:
            f.write(build_hypodd_inp(inp, iter_sets=sets))
        reloc = _exec_hypodd_once(d)
        cnds = _max_cnd_per_set(os.path.join(d, "hypoDD.log"), sets)
        if not cnds:                                        # nothing to tune on (e.g. no CND logged)
            break
        score = max(max(0.0, c - hi) + max(0.0, lo - c) for c in cnds.values())
        history.append(([s[-1] for s in sets], {k: round(v, 1) for k, v in sorted(cnds.items())},
                        round(score, 1)))
        if best is None or score < best[0]:
            best = (score, [list(r) for r in sets])
        if score <= 0.0:                                    # every set inside the band
            break
        for i, c in cnds.items():                           # higher DAMP -> lower CND
            newd = sets[i][-1] * (c / mid) ** 0.5
            newd = min(2000.0, max(1.0, newd))
            sets[i][-1] = int(round(newd)) if int_damp[i] else round(newd, 2)
    # make the returned reloc correspond to the best damping found
    if best is not None and [list(r) for r in sets] != best[1]:
        with open(os.path.join(d, "hypoDD.inp"), "w") as f:
            f.write(build_hypodd_inp(inp, iter_sets=best[1]))
        reloc = _exec_hypodd_once(d)
    if history:
        with open(os.path.join(d, "damping_calibration.txt"), "w") as f:
            f.write(f"adaptive LSQR damping — target CND {lo:.0f}-{hi:.0f}\n"
                    "attempt: DAMP per set -> max CND per set (worst-band violation)\n")
            for a, (damps, cnds, score) in enumerate(history):
                f.write(f"  {a}: {damps} -> {cnds}  (score {score})\n")
            if best is not None:
                f.write(f"chosen DAMP per set: {[r[-1] for r in best[1]]}\n")
    return reloc


def run_dtct(cfg):
    """Catalog-only (dt.ct) HypoDD relocation; returns the hypoDD.reloc path."""
    src = config.ph2dt_dir(cfg)
    d = config.assert_writable(config.dtct_dir(cfg))
    os.makedirs(d, exist_ok=True)
    for fn in ("dt.ct", "event.dat", "event.sel", "station.dat"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(d, fn))
    # dt.ct is a hard regression baseline — keep its damping fixed (no adaptive tuning).
    return _exec_hypodd(d, cfg.hypodd_dtct, adapt_damping=False)


def _event_cuspid(cfg, event_id, velmodel="kim1983"):
    """Map a UTC event_id (sorted-dir name) to its HypoDD cuspid via the .sum ID-NUM."""
    from glob import glob as _glob
    from pipeline.core import sumio
    sumdf = sumio.read_sum(config.sum_file(cfg, velmodel))
    dirs = sorted(_glob(os.path.join(config.waveforms_dir(cfg), "20*")))
    for r in sumdf.itertuples():
        idx = int(r.id) % cfg.cuspid_offset
        if idx < len(dirs) and os.path.basename(dirs[idx]) == event_id:
            return int(r.id)
    return None


def run_dtcc(cfg, variant="default"):
    """Cross-correlation (dt.cc) HypoDD relocation for one variant; returns hypoDD.reloc.

    Requires `core.xcorr.run_xcorr` to have written the cc file (e.g. dt.cc_0.7_combined)
    into 02.dt.cc/. The "default" variant runs in 02.dt.cc/ itself; named variants
    (no_main, kim2011, ...) run in a 02.dt.cc/<variant>/ subdir. A variant whose
    event_file is "event.sel" relocates WITHOUT the mainshock: we rebuild event.sel from
    event.dat dropping the mainshock cuspid (ph2dt's own event.sel means something else)."""
    inp = cfg.hypodd_dtcc_variants[variant]
    base = config.dtcc_dir(cfg)
    d = config.assert_writable(base if variant == "default" else os.path.join(base, variant))
    os.makedirs(d, exist_ok=True)
    src = config.ph2dt_dir(cfg)
    for fn in ("dt.ct", "event.dat", "event.sel", "station.dat"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(d, fn))
    if inp.event_file == "event.sel" and cfg.mainshock_event_id:
        cus = _event_cuspid(cfg, cfg.mainshock_event_id)
        if cus is not None:
            rows = open(os.path.join(d, "event.dat")).readlines()
            with open(os.path.join(d, "event.sel"), "w") as f:
                f.writelines(r for r in rows if r.split() and r.split()[-1] != str(cus))
    if inp.cc_file:                                          # xcorr wrote it under `base`
        cc_src = os.path.join(base, inp.cc_file)
        if not os.path.exists(cc_src):
            raise FileNotFoundError(
                f"cross-correlation file {cc_src} not found — run the xcorr stage first.")
        if os.path.realpath(d) != os.path.realpath(base):
            shutil.copyfile(cc_src, os.path.join(d, inp.cc_file))
    # dt.cc is report-only (judgment-dependent). When LSQR is forced (large dt set vs the SVD
    # MAXDATA0 limit), flexibly modulate two ill-conditioning levers: (1) the inter-event distance
    # cutoffs, scaled to the cluster size so spatially peripheral / poorly-linked events drop out;
    # (2) DAMP, tuned so the condition number stays in the ~40–80 band.
    adapt = (inp.isolv == 2)
    if adapt:
        L = _cluster_diameter_km(cfg)
        if L:
            scaled, f = _scale_distance_cutoffs(inp.iter_sets, L)
            inp = dataclasses.replace(inp, iter_sets=scaled)
            print(f"  [dt.cc:{variant}] LSQR: cluster diameter {L:.2f} km -> distance-cutoff "
                  f"scale {f:.2f}; adaptive damping on (CND target 40-80)")
    return _exec_hypodd(d, inp, adapt_damping=adapt)


def run_backbone(cfg, velmodel="kim1983"):
    """P5+P6: ph2dt prep -> ph2dt -> dt.ct relocation."""
    prep_ph2dt(cfg, velmodel=velmodel)
    run_ph2dt(cfg)
    return run_dtct(cfg)
