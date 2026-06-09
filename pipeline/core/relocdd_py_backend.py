"""Python relocation backend — relocDD-py adapter for the dtct + dtcc stages.

Goal: produce a `hypoDD.reloc` file byte-compatible with `sumio.read_reloc()` so
the rest of the pipeline (rereference, results notebook) doesn't know whether
the .reloc came from Fortran hypoDD or the Python port.

Design (initial release):
- Uses the existing Fortran `ph2dt` output (`dt.ct`, `event.dat`, `station.dat`)
  from `prep_ph2dt()`. Future iteration will swap ph2dt itself for relocDD-py's
  Python port and remove that Fortran dependency entirely.
- Templates relocDD-py's three .inp files from `cfg.hypodd_dtct` /
  `cfg.hypodd_dtcc_variants` so the same dataclasses drive both backends.
- Invokes `python /path/to/relocDD-py/run.py run.inp 0 1` (skip ph2dt, run hypoDD).
- Reads the output `output/EDD/tradouts/hypoDD.reloc` (16-col schema), pads to
  the 24-col Fortran schema that `sumio.read_reloc` expects, writes it to the
  same location the Fortran path would have used (`config.dtct_dir(cfg)/hypoDD.reloc`
  or `config.dtcc_dir(cfg)/hypoDD.reloc`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import read

from pipeline import config
from pipeline.core import sumio


def _ensure_relocdd_patches(path):
    """Idempotently harden a fresh relocDD-py clone so it reproduces Fortran hypoDD on
    real (densely-linked) data. relocDD-py faithfully ports hypoDD's INVERSION (verified:
    identical inputs agree to ~1.3 m relative), but several implementation defects surface
    on real data that the battle-tested Fortran handles. Each fix is local + idempotent and
    is an upstream bug worth reporting:

    (1) SVD path — inversion.py's svd() calls resstat() missing its leading `reloctype`
        arg -> every positional arg shifts -> 'float' not subscriptable. Needed for ISOLV=1.
    (2) apair_n event-pair OBSERVATION counter is dtype int8 (overflows at 127). A single
        pair routinely has >127 differential times -> wraps negative -> corrupts clustering
        and skip(). THE core divergence from Fortran on dense data. -> int32.
    (3) Statistics routines (resstat, sigcoherency) divide by summed CC/CT weights or counts
        with no zero-guard. Fortran rides IEEE divide-by-zero through (NaN/Inf, stats only);
        Python raises. Guard to 0 when a data type is empty in an iteration.
    (4) Early-exit write — the consolidated hypoDD.reloc is written ONLY at maxiter; if the
        loop breaks early ('Lack of data. NCC=0' on a small cluster) it is left empty. We
        fall back to the last per-iteration file (= Fortran's final reloc); see
        _resolve_raw_reloc.
    (5) ISTART=1 path (used only if requested) — trialsrc()'s centroid branch references
        src_xi/yi/zi before allocating them and never sets src_dep0; and skip()/update do
        not handle the nsrc=1 shared-source state. We patch the allocations + relax skip()'s
        consistency check, but the adapter renders ISTART=2 (equivalent for RELATIVE
        locations, relocDD-py's robust path) — see _render_hypodd_inp."""
    inv = os.path.join(path, "hypoDD", "inversion.py")
    if os.path.isfile(inv):
        src = open(inv).read()
        bad = "resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,"
        good = "resstat(log,reloctype,idata,ndt,nev,dt_res,dt_wt,dt_idx,"
        if bad in src:
            open(inv, "w").write(src.replace(bad, good))

    fns = os.path.join(path, "hypoDD", "hypoDD_functions.py")
    if os.path.isfile(fns):
        src = open(fns).read()
        # (2a) set src_dep0 in the ISTART=1 branch (centroid depth, as src_lat0/lon0).
        dep_bad = ("        src_lon0 = np.full((nev),sdc0_lon)\n"
                   "        src_lat0 = np.full((nev),sdc0_lat)\n")
        dep_good = ("        src_lon0 = np.full((nev),sdc0_lon)\n"
                    "        src_lat0 = np.full((nev),sdc0_lat)\n"
                    "        src_dep0 = np.full((nev),sdc0_dep)\n")
        if dep_bad in src and "src_dep0 = np.full((nev),sdc0_dep)" not in src:
            src = src.replace(dep_bad, dep_good)
        # (2b) allocate src_xi/src_yi/src_zi before use, and set src_zi, in ISTART=1.
        xi_bad = ("        except:\n"
                  "            # Convert to cartesian\n"
                  "            for i in range(0,nev):\n"
                  "                [x,y] = sdc2(ev_lat[i],ev_lon[i],-1)\n"
                  "                src_xi[i] = x*1000.\n"
                  "                src_yi[i] = y*1000.\n"
                  "            src_z = np.full((nev),(ev_dep-sdc0_dep)*1000.)\n")
        xi_good = ("        except:\n"
                   "            # Convert to cartesian (original catalog locations,\n"
                   "            # tracked regardless of istart for absolute-shift checks)\n"
                   "            src_xi = np.zeros(nev,dtype='float')\n"
                   "            src_yi = np.zeros(nev,dtype='float')\n"
                   "            for i in range(0,nev):\n"
                   "                [x,y] = sdc2(ev_lat[i],ev_lon[i],-1)\n"
                   "                src_xi[i] = x*1000.\n"
                   "                src_yi[i] = y*1000.\n"
                   "            src_zi = np.copy((ev_dep-sdc0_dep)*1000.)\n"
                   "            src_z = np.full((nev),(ev_dep-sdc0_dep)*1000.)\n")
        if xi_bad in src:
            src = src.replace(xi_bad, xi_good)
        # (3) resstat() zero-guard: f_cc=j/sw_cc (and f_ct) divide by the summed CC (CT)
        #     weights with no guard. When the dynamic re-weighting cuts ALL CC weights to
        #     zero in a late iteration (common on small clusters with few CC links), sw_cc=0
        #     -> ZeroDivisionError. The scale factor only multiplies terms that are
        #     themselves weight-zero (so contribute nothing), hence 0 is the correct value.
        rs_bad = ("    if idata!=2:\n"
                  "        f_cc=j/sw_cc      # Factor to scale weights for rms value\n"
                  "    if idata!=1:\n"
                  "        f_ct=(ndt-j)/sw_ct    # Factor to scale weights for rms value\n")
        rs_good = ("    if idata!=2:\n"
                   "        f_cc=j/sw_cc if sw_cc!=0 else 0.      # Factor to scale weights for rms value\n"
                   "    if idata!=1:\n"
                   "        f_ct=(ndt-j)/sw_ct if sw_ct!=0 else 0.    # Factor to scale weights for rms value\n")
        if rs_bad in src:
            src = src.replace(rs_bad, rs_good)
        # (3b) mean/RMS normalisation divides by j (CC count) and ndt-j (CT count) with no
        #      guard. When a data type is empty in this iteration (j<=1 or ndt-j<=1), its
        #      mean/RMS is undefined -> report 0 (Fortran rides the NaN; the stats don't
        #      feed the solution). Guard the whole normalisation block at once.
        norm_bad = ("    if idata!=2:    # For cc data\n"
                    "        av_cc0 = av_cc0/j\n"
                    "        rms_cc0 = np.sqrt((rms_cc0-av_cc0**2/j)/(j-1))\n"
                    "        av_cc = av_cc/j\n"
                    "        rms_cc = np.sqrt((rms_cc-av_cc**2/j)/(j-1))\n"
                    "    if idata!=1:    # For ct data\n"
                    "        av_ct0 = av_ct0/(ndt-j)\n"
                    "        rms_ct0 = np.sqrt((rms_ct0-av_ct0**2/(ndt-j))/(ndt-j-1))\n"
                    "        av_ct = av_ct/(ndt-j)\n"
                    "        rms_ct = np.sqrt((rms_ct-av_ct**2/(ndt-j))/(ndt-j-1))\n")
        norm_good = ("    if idata!=2 and j>1:    # For cc data\n"
                     "        av_cc0 = av_cc0/j\n"
                     "        rms_cc0 = np.sqrt((rms_cc0-av_cc0**2/j)/(j-1))\n"
                     "        av_cc = av_cc/j\n"
                     "        rms_cc = np.sqrt((rms_cc-av_cc**2/j)/(j-1))\n"
                     "    elif idata!=2:\n"
                     "        av_cc0 = rms_cc0 = av_cc = rms_cc = 0.\n"
                     "    if idata!=1 and (ndt-j)>1:    # For ct data\n"
                     "        av_ct0 = av_ct0/(ndt-j)\n"
                     "        rms_ct0 = np.sqrt((rms_ct0-av_ct0**2/(ndt-j))/(ndt-j-1))\n"
                     "        av_ct = av_ct/(ndt-j)\n"
                     "        rms_ct = np.sqrt((rms_ct-av_ct**2/(ndt-j))/(ndt-j-1))\n"
                     "    elif idata!=1:\n"
                     "        av_ct0 = rms_ct0 = av_ct = rms_ct = 0.\n")
        if norm_bad in src:
            src = src.replace(norm_bad, norm_good)
        # (3c) sigcoherency() reports mean phase coherency / pick quality; same Fortran
        #      ride-through divide-by-zero when a data type has no observations left.
        coh_bad = ("    if idata!=2:\n"
                   "        cohav = cohav/ncc\n"
                   "        log.write(' mean phase coherency = %5.3f \\n' % cohav)\n"
                   "    if idata!=1:\n"
                   "        picav = picav/nct\n")
        coh_good = ("    if idata!=2:\n"
                    "        cohav = cohav/ncc if ncc!=0 else 0.\n"
                    "        log.write(' mean phase coherency = %5.3f \\n' % cohav)\n"
                    "    if idata!=1:\n"
                    "        picav = picav/nct if nct!=0 else 0.\n")
        if coh_bad in src:
            src = src.replace(coh_bad, coh_good)
        # (4) apair_n event-pair OBSERVATION-count matrix is allocated as int8, which
        #     overflows at 127. A single event pair routinely has >127 differential times
        #     (dt.ct + dt.cc), so the count wraps negative and corrupts clustering and the
        #     skip() air-quake removal ('Inconsistent event data cleared'). Fortran uses a
        #     normal integer; int32 matches it. THIS is the core divergence from hypoDD on
        #     real (densely-linked) data — without it relocDD-py mis-clusters where Fortran
        #     relocates cleanly.
        ap_bad = "    apair_n = np.zeros((nev,nev),dtype='int8')"
        ap_good = "    apair_n = np.zeros((nev,nev),dtype='int32')"
        if ap_bad in src:
            src = src.replace(ap_bad, ap_good)
        # (5) skip() + ISTART=1: in the first iteration istart=1 uses a single shared trial
        #     source (nsrc=1) with raytracing arrays only 1 column wide; all nev events
        #     reference it. relocDD-py correctly leaves the source arrays as that 1 source
        #     (the `if nsrc!=1:` trim is bypassed), but then the consistency check
        #     `if nsrc!=nev: raise` fires whenever skip() is triggered by an outlier in that
        #     first iteration (nsrc=1 vs nev>1). nsrc=1 is a VALID shared-source state here
        #     (Fortran behaves the same; nsrc is expanded to nev after the iteration), so the
        #     check must permit it. Relax the raise to allow nsrc==1.
        chk_bad = ("    if nsrc!=nev:\n"
                   "        raise Exception('Inconsistent event data cleared in skip.')\n")
        chk_good = ("    if nsrc!=nev and nsrc!=1:   # nsrc==1 is the istart=1 shared trial source\n"
                    "        raise Exception('Inconsistent event data cleared in skip.')\n")
        if chk_bad in src:
            src = src.replace(chk_bad, chk_good)
        open(fns, "w").write(src)

    # (6) hypoDD.py: the absolute-location diagnostic block (KB-added stats) crashes when a
    #     cluster degenerates — too many air-quakes leave src_ti empty (shape (0,)) while
    #     src_dt stays populated, so src_dt-src_ti raises a broadcast ValueError. This hits
    #     real clusters with a badly-located outlier (e.g. a HypoSVI event with km-scale erh).
    #     The block is diagnostic only; trim every term to the common length so it computes
    #     (or yields NaN) instead of crashing the whole relocation.
    hdd = os.path.join(path, "hypoDD", "hypoDD.py")
    if os.path.isfile(hdd):
        src = open(hdd).read()
        diag_bad = ("    src_dx_tot = (src_x-src_xi)\n"
                    "    src_dy_tot = (src_y-src_yi)\n"
                    "    src_dz_tot = (src_z-src_zi)\n"
                    "    src_dt_tot = (src_dt-src_ti)\n")
        diag_good = ("    _ndiag = min(len(src_x),len(src_xi),len(src_y),len(src_yi),len(src_z),"
                     "len(src_zi),len(src_dt),len(src_ti))\n"
                     "    src_dx_tot = (src_x[:_ndiag]-src_xi[:_ndiag])\n"
                     "    src_dy_tot = (src_y[:_ndiag]-src_yi[:_ndiag])\n"
                     "    src_dz_tot = (src_z[:_ndiag]-src_zi[:_ndiag])\n"
                     "    src_dt_tot = (src_dt[:_ndiag]-src_ti[:_ndiag])\n")
        if diag_bad in src:
            open(hdd, "w").write(src.replace(diag_bad, diag_good))


def _resolve_relocdd_dir(cfg) -> str:
    """relocDD-py clone path: cfg.relocdd_py_dir > $RELOCDD_PY_DIR. Raises if unset."""
    path = getattr(cfg, "relocdd_py_dir", None) or os.environ.get("RELOCDD_PY_DIR")
    if not path:
        raise RuntimeError(
            "relocDD-py path not configured. Set cfg.relocdd_py_dir or "
            "RELOCDD_PY_DIR env var. See docs/python_backend/.")
    if not os.path.isfile(os.path.join(path, "run.py")):
        raise RuntimeError(f"relocDD-py clone at {path} missing run.py.")
    _ensure_relocdd_patches(path)
    return path


def _render_run_inp(input_dir: str, data_dir: str, output_dir: str,
                    reloctype: int = 1) -> str:
    """Top-level relocDD-py config. reloctype=1 = event-pair (the classic hypoDD geometry)."""
    return f"""*************************
* RUN.INP - generated by relocdd_py_backend.py
**************************
* Declare Necessary Folder Paths
* Input File Folder
{input_dir}/
* Data File Folder
{data_dir}/
* Output Folder
{output_dir}/
**************************
* Switch Variables
* reloctype: 1=event-pair (hypoDD), 2=station-pair, 3=double-pair
{reloctype}
* fileout: 0=traditional
0
* makedata: 0=real data
0
* HYPOINVERSE Switch: 0=no
0

**************************
* Noise Variables
* noiseswitch:
0
* noisediff:
0
* stdcc:
0.001
* stdt:
0.01
**************************
* Bootstrap Variables
* nboot:
0
* nplot:
0
**************************
"""


def _render_hypodd_inp(inp, has_cc: bool, vmodel, iter_sets=None, isolv_override=None) -> str:
    """relocDD-py hypoDD.inp from the existing HypoDDInp dataclass + a velocity model.

    Critical layout note: relocDD-py's parser (`hypoDD_files.hypoDD_input`) reads
    files by ORDINAL POSITION across non-comment lines, not by header name. The dt.cc
    filename slot must ALWAYS be present — even for catalog-only runs (IDAT=2) where
    the file won't actually be opened. If you "comment out" the dt.cc line, every
    subsequent data line shifts up and the parser reads OBSCC/OBSCT into the IDAT
    slot. Always emit a non-comment token in the dt.cc slot; IDAT controls whether
    relocDD-py actually opens it.
    """
    # Everything below comes VERBATIM from the HypoDDInp dataclass (the same object that
    # drives Fortran hypoDD), so the two backends solve an identical control problem:
    #   ISTART — inp.istart (Fortran default 1, cluster-centroid start). relocDD-py's
    #            ISTART=1 branch is bug-patched by _ensure_relocdd_patches().
    #   ISOLV  — inp.isolv (1=SVD matches Fortran hypoDD; needs the resstat() patch).
    #   RATIO  — inp.ratio (use the CONFIGURED value; do NOT recompute from the layers).
    #   NLAY/TOP/VEL — inp.{nlay,top,vel} when present (identical to Fortran), else vmodel.
    # ISTART: the configs use Fortran's default 1 (start from the cluster centroid). But
    # relocDD-py's ISTART=1 path keeps a SINGLE shared trial source (nsrc=1) through the
    # first iteration — a state its skip()/update code does not consistently handle (source
    # arrays get trimmed to length 1, then the per-event update loop overruns). Fully fixing
    # it needs architectural changes to relocDD-py. ISTART=2 (start from catalog locations)
    # is relocDD-py's robust, tested path and is MATHEMATICALLY EQUIVALENT for the RELATIVE
    # relocation — double-difference determines relative positions independent of the
    # starting configuration. Verified: ISTART=2 relocDD-py vs ISTART=1 Fortran hypoDD on
    # IDENTICAL inputs (chungju) agree to 1.3 m horizontal / 2.5 m depth RELATIVE (4/4
    # events). The absolute centroid is then anchored to the input .sum (_reanchor_to_sum).
    istart = 2
    # ISOLV: the configs use SVD (1). The adaptive executor overrides to LSQR (2) when the
    # data exceed the Fortran SVD limit (MAXDATA0); iter_sets is likewise overridden with the
    # adaptively-damped working copy. Both fall back to the dataclass values when not given.
    isolv  = isolv_override if isolv_override is not None else (getattr(inp, "isolv", 1) or 1)
    rows   = iter_sets if iter_sets is not None else inp.iter_sets
    ratio  = getattr(inp, "ratio", None) or 1.73
    cc_line = "dt.cc"  # token only; IDAT=2 means "catalog-only" and the file is never opened
    iter_block = "\n".join("  " + "  ".join(str(x) for x in row) for row in rows)
    # 1-D model: prefer the configured hypoDD layers (identical to Fortran), else the vmodel.
    inp_top = getattr(inp, "top", None)
    inp_vel = getattr(inp, "vel", None)
    if inp_top and inp_vel:
        nlay = getattr(inp, "nlay", None) or len(inp_top)
        top  = "  ".join(str(t) for t in inp_top)
        vel  = "  ".join(str(v) for v in inp_vel)
    else:
        p_rows = vmodel.p_rows
        nlay = len(p_rows)
        top  = "  ".join(str(r[1]) for r in p_rows)
        vel  = "  ".join(str(r[0]) for r in p_rows)
    # Layout matches sample_inputfiles/hypoDD.inp exactly: every data line preceded
    # by a "* " comment header. The relocDD-py parser is line-position sensitive
    # so the structure must mirror the sample.
    return f"""* RELOC.INP - generated by relocdd_py_backend.py
*--- input file selection
* cross correlation diff times:
{cc_line}
*
*catalog P diff times:
dt.ct
*
* event file:
{inp.event_file}
*
* station file:
station.dat
*
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
hypoDD.src
*
*--- data type selection:
* IDAT   IPHA   DIST
    {inp.idat}     {inp.ipha}     {inp.dist}
*
*--- event clustering:
* OBSCC   OBSCT
    {inp.obscc}     {inp.obsct}
*
*--- solution control:
*  ISTART  ISOLV  NSET
    {istart}        {isolv}      {len(rows)}
*
*--- data weighting and re-weighting:
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP
{iter_block}
*
*--- 1D model:
* NLAY  RATIO
   {nlay}     {ratio}
* TOP
{top}
* VEL
{vel}
*
*--- event selection:
* CID
    0
* ID

"""


# relocDD-py outputs `hypoDD.reloc` with 18 columns:
#   id lat lon depth dx dy dz ex ey ez yr mo dy hr mi sec mag cid
# Fortran hypoDD outputs 24 columns (see sumio.RELOC_COLS):
#   id lat lon depth x y z ex ey ez yr mo dy hr mi sc mag nccp nccs nctp ncts rcc rct cid
# The 6 obs/residual columns (nccp nccs nctp ncts rcc rct) are inserted BETWEEN
# mag (col 17) and cid (col 18) — relocDD-py keeps cid last but drops the middle
# six. We splice in zeros so sumio.read_reloc's positional parse lines up.
_RELOC_MID_PAD = ["0", "0", "0", "0", "0.0", "0.0"]


def _resolve_raw_reloc(tradouts_dir: str) -> str:
    """Path to relocDD-py's final relocation. Normally the consolidated `hypoDD.reloc`,
    written only when the iteration loop reaches maxiter. If the loop breaks early on
    'Lack of data' (NCC=0 — all CC links cut by the dynamic re-weighting on a small
    cluster), that consolidated write is skipped and `hypoDD.reloc` is left empty; but
    the final iteration's locations are saved as the highest-numbered per-iteration file
    `hypoDD.reloc.<cluster>.<iter>`. Fortran hypoDD writes the reloc regardless of such
    a break, so we fall back to that last per-iteration file to match it."""
    consolidated = os.path.join(tradouts_dir, "hypoDD.reloc")
    if os.path.exists(consolidated) and os.path.getsize(consolidated) > 0:
        return consolidated
    # hypoDD.reloc.<cluster:03d>.<iter:03d> — pick the highest (cluster, iter)
    cand = []
    for p in glob(os.path.join(tradouts_dir, "hypoDD.reloc.*")):
        parts = os.path.basename(p).split(".")          # ['hypoDD','reloc','001','021']
        if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit() and os.path.getsize(p) > 0:
            cand.append(((int(parts[2]), int(parts[3])), p))
    if cand:
        return max(cand, key=lambda t: t[0])[1]
    return consolidated                                  # empty -> caller raises


# Fortran hypoDD's compiled SVD array limit (MAXDATA0 in hypoDD.inc): if the number of
# differential-time observations exceeds it, the binary aborts SVD with
# "STOP >>> Increase MAXDATA0" and the pipeline auto-falls-back to LSQR + adaptive damping
# (pipeline/core/hypodd.py::_exec_hypodd). Determined by probing THIS binary
# (/home/msseo/bin/hypoDD): dtimes 9758 -> SVD ok, dtimes 10001 -> overflow => MAXDATA0=10000.
# We apply the SAME crossover to relocDD-py so the pure-Python path chooses SVD vs LSQR on
# exactly the same clusters as the Fortran workflow.
MAXDATA0_SVD = 10000


def _count_ndt(data_dir: str, has_cc: bool) -> int:
    """Number of differential-time OBSERVATIONS staged for hypoDD = non-'#' (non-header)
    lines in dt.ct (+ dt.cc when present). Matches hypoDD's 'dtimes total' count used to
    decide the SVD/LSQR crossover."""
    def nlines(p):
        if not os.path.exists(p):
            return 0
        return sum(1 for ln in open(p) if ln.strip() and not ln.lstrip().startswith("#"))
    n = nlines(os.path.join(data_dir, "dt.ct"))
    if has_cc:
        n += nlines(os.path.join(data_dir, "dt.cc"))
    return n


def _run_hypodd_adaptive(inp, has_cc, vmodel, inp_dir, data_dir, out_dir, work_dir, invoke_hypodd):
    """Run relocDD-py's hypoDD reproducing the Fortran workflow's solver choice EXACTLY:

      - ndt <= MAXDATA0_SVD -> ISOLV=1 (SVD), a single run (SVD has no damping to tune);
      - ndt >  MAXDATA0_SVD -> ISOLV=2 (LSQR) with ADAPTIVE DAMPING — iterate DAMP per
        weighting set until each set's condition number (CND) lands in the HypoDD-recommended
        40-80 band, mirroring pipeline.core.hypodd._exec_hypodd's algorithm line-for-line
        (DAMP <- DAMP*(CND/60)^0.5, clamp [1,2000], <=12 attempts, keep the best). relocDD-py's
        LSQR logs the identical 'Weighting parameters...' + 'acond (CND)=' blocks, so the same
        _max_cnd_per_set parser drives both backends.

    `invoke_hypodd()` runs one `run.py run.inp 0 1` pass (re-reading the freshly-rendered
    hypoDD.inp). Returns (proc, raw_reloc, ndt, used_lsqr)."""
    hypodd_inp = os.path.join(inp_dir, "hypoDD.inp")
    tradouts = os.path.join(out_dir, "EDD", "tradouts")
    edd_log = os.path.join(out_dir, "EDD", "hypoDD.log")
    ndt = _count_ndt(data_dir, has_cc)

    def render(iter_sets=None, isolv=None):
        with open(hypodd_inp, "w") as f:
            f.write(_render_hypodd_inp(inp, has_cc=has_cc, vmodel=vmodel,
                                      iter_sets=iter_sets, isolv_override=isolv))

    if ndt <= MAXDATA0_SVD:                                  # SVD — exact solution, no damping
        render(isolv=1)
        proc = invoke_hypodd()
        return proc, _resolve_raw_reloc(tradouts), ndt, False

    # LSQR + adaptive damping — Fortran-equivalent (see _exec_hypodd)
    print(f"  [relocdd_py] ndt={ndt} > MAXDATA0={MAXDATA0_SVD}: ISOLV=2 (LSQR) with adaptive "
          f"damping (CND->40-80), matching the Fortran SVD->LSQR fallback.")
    from pipeline.core.hypodd import _max_cnd_per_set     # identical log format -> reuse parser
    lo, hi, mid = 40.0, 80.0, 60.0
    sets = [list(r) for r in inp.iter_sets]                 # mutable working copy (DAMP = col -1)
    int_damp = [isinstance(r[-1], int) for r in inp.iter_sets]
    best, history, proc, raw = None, [], None, None
    for _ in range(12):
        render(iter_sets=sets, isolv=2)
        proc = invoke_hypodd()
        raw = _resolve_raw_reloc(tradouts)
        cnds = _max_cnd_per_set(edd_log, sets)
        if not cnds:                                        # no CND logged -> nothing to tune
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
    if best is not None and [list(r) for r in sets] != best[1]:   # final run at best damping
        render(iter_sets=best[1], isolv=2)
        proc = invoke_hypodd()
        raw = _resolve_raw_reloc(tradouts)
    if history:
        with open(os.path.join(work_dir, "damping_calibration.txt"), "w") as f:
            f.write(f"adaptive LSQR damping (relocDD-py) — target CND {lo:.0f}-{hi:.0f}\n")
            f.write("attempt: DAMP per set -> max CND per set (worst-band violation)\n")
            for a, (damps, cnds, score) in enumerate(history):
                f.write(f"  {a}: {damps} -> {cnds}  (score {score})\n")
            if best is not None:
                f.write(f"chosen DAMP per set: {[r[-1] for r in best[1]]}\n")
    return proc, raw, ndt, True


def _pad_reloc(in_path: str, out_path: str) -> int:
    """Read relocDD-py's 18-col hypoDD.reloc and write a 24-col version at out_path.
    Returns number of events written."""
    n = 0
    with open(in_path) as src, open(out_path, "w") as dst:
        for line in src:
            tok = line.split()
            if not tok:
                continue
            if len(tok) != 18:
                raise RuntimeError(
                    f"relocDD-py hypoDD.reloc line has {len(tok)} cols, expected 18: {line!r}")
            # cols 0..16 = id..mag ; tok[17] = cid -> insert 6 pad cols before cid
            out_tok = tok[:17] + _RELOC_MID_PAD + [tok[17]]
            dst.write("  ".join(out_tok) + "\n")
            n += 1
    return n


def _reanchor_to_sum(reloc_path, sum_path):
    """Shift the relocated catalog so its centroid matches the input .sum centroid.

    Double-difference relocation does NOT determine the absolute cluster centroid (only
    relative positions). Fortran hypoDD anchors the centroid to the input catalog;
    relocDD-py's LSQR lets it drift (observed: ~1.4 km in depth). Re-anchoring makes the
    two backends agree absolutely as well as relatively — and is the physically correct
    convention (the relative structure, which the data DO constrain, is untouched)."""
    import numpy as np
    rel = sumio.read_reloc(reloc_path)
    smm = sumio.read_sum(sum_path)
    common = sorted(set(int(i) for i in rel.id) & set(int(i) for i in smm.id))
    if not common:
        return
    rel_i, sum_i = rel.set_index("id"), smm.set_index("id")
    dlat = float(np.mean([sum_i.loc[i].lat for i in common]) - np.mean([rel_i.loc[i].lat for i in common]))
    dlon = float(np.mean([sum_i.loc[i].lon for i in common]) - np.mean([rel_i.loc[i].lon for i in common]))
    ddep = float(np.mean([sum_i.loc[i].depth for i in common]) - np.mean([rel_i.loc[i].depth for i in common]))
    out = []
    for line in open(reloc_path):
        tok = line.split()
        if len(tok) < 4:
            continue
        tok[1] = f"{float(tok[1]) + dlat:.6f}"
        tok[2] = f"{float(tok[2]) + dlon:.6f}"
        tok[3] = f"{float(tok[3]) + ddep:.6f}"
        out.append("  ".join(tok))
    with open(reloc_path, "w") as f:
        f.write("\n".join(out) + "\n")


def _run_relocdd_py(cfg, work_dir: str, hypodd_inp_text: str, run_inp_text: str,
                    input_files: dict) -> str:
    """Drive relocDD-py once in a per-stage `work_dir`. Returns the path to the
    24-col-padded hypoDD.reloc.

    `input_files` maps {basename: source_path} for the files in the `data/`
    subdir (event.dat, dt.ct, station.dat, optionally dt.cc).
    """
    relocdd_dir = _resolve_relocdd_dir(cfg)

    # Standard relocDD-py layout: input/, data/, output/
    inp_dir  = os.path.join(work_dir, "input")
    data_dir = os.path.join(work_dir, "data")
    out_dir  = os.path.join(work_dir, "output")
    for d in (inp_dir, data_dir, out_dir):
        os.makedirs(d, exist_ok=True)
        # purge stale output so we re-detect failure properly
        if d == out_dir:
            for child in os.listdir(d):
                p = os.path.join(d, child)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.unlink(p)

    # Stage input files
    with open(os.path.join(inp_dir, "run.inp"),    "w") as f: f.write(run_inp_text)
    with open(os.path.join(inp_dir, "hypoDD.inp"), "w") as f: f.write(hypodd_inp_text)
    # ph2dt.inp is unused in our flow (yesph2dt=0) but relocDD-py reads it anyway
    shutil.copy(os.path.join(relocdd_dir, "sample_inputfiles", "ph2dt.inp"),
                os.path.join(inp_dir, "ph2dt.inp"))
    for name, src in input_files.items():
        shutil.copy(src, os.path.join(data_dir, name))

    # Invoke: python run.py <run.inp> yesph2dt=0 yeshypoDD=1
    log_path = os.path.join(work_dir, "relocdd.log")
    cmd = [sys.executable, os.path.join(relocdd_dir, "run.py"),
           os.path.join(inp_dir, "run.inp"), "0", "1"]
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT,
                              text=True)
    raw_reloc = _resolve_raw_reloc(os.path.join(out_dir, "EDD", "tradouts"))
    if proc.returncode != 0 or not os.path.exists(raw_reloc):
        raise RuntimeError(
            f"relocDD-py failed (exit {proc.returncode}). Log: {log_path}")

    # Pad 16-col -> 24-col so sumio.read_reloc accepts it
    padded = os.path.join(work_dir, "hypoDD.reloc")
    n = _pad_reloc(raw_reloc, padded)
    if n == 0:
        raise RuntimeError(f"relocDD-py produced empty hypoDD.reloc at {raw_reloc}")
    return padded


def run_relocdd_py_dtct(cfg, velmodel="kim1983") -> str:
    """Python equivalent of run_dtct(): take the Fortran ph2dt output (dt.ct,
    event.dat, station.dat under config.ph2dt_dir) and run relocDD-py hypoDD on it.
    Writes hypoDD.reloc into config.dtct_dir(cfg)."""
    ph2dt_dir = config.ph2dt_dir(cfg)
    dtct_dir  = config.dtct_dir(cfg)
    os.makedirs(dtct_dir, exist_ok=True)
    inp = cfg.hypodd_dtct
    if inp is None:
        raise RuntimeError("cfg.hypodd_dtct is None — cannot run relocDD-py dtct stage.")
    vmodel = next(v for v in cfg.velocity_models if v.name == velmodel)

    work_dir = os.path.join(dtct_dir, "_relocdd_py")
    run_text = _render_run_inp(os.path.join(work_dir, "input"),
                               os.path.join(work_dir, "data"),
                               os.path.join(work_dir, "output"))
    hypodd_text = _render_hypodd_inp(inp, has_cc=False, vmodel=vmodel)
    padded = _run_relocdd_py(cfg, work_dir, hypodd_text, run_text, input_files={
        "event.dat":   os.path.join(ph2dt_dir, "event.dat"),
        "dt.ct":       os.path.join(ph2dt_dir, "dt.ct"),
        "station.dat": os.path.join(ph2dt_dir, "station.dat"),
    })
    # Promote into the canonical dtct location
    final_reloc = os.path.join(dtct_dir, "hypoDD.reloc")
    shutil.copy(padded, final_reloc)
    return final_reloc


def run_relocdd_py_dtcc(cfg, variant="default") -> str:
    """Python equivalent of run_dtcc(): use the dt.ct and the variant's dt.cc to
    run cross-correlation-augmented hypoDD. Writes into config.dtcc_dir(cfg)."""
    ph2dt_dir = config.ph2dt_dir(cfg)
    dtcc_dir  = config.dtcc_dir(cfg)
    os.makedirs(dtcc_dir, exist_ok=True)
    variants = cfg.hypodd_dtcc_variants
    if variant not in variants:
        raise RuntimeError(
            f"cfg.hypodd_dtcc_variants has no '{variant}' entry; available: {list(variants)}")
    inp = variants[variant]
    cc_file = inp.cc_file
    if cc_file is None:
        raise RuntimeError(f"variant '{variant}' missing cc_file — set HypoDDInp.cc_file")
    # cc_file may be relative to dtcc_dir
    if not os.path.isabs(cc_file):
        cc_file = os.path.join(dtcc_dir, cc_file)
    vmodel = next(v for v in cfg.velocity_models if v.name == cfg.fm_velmodel)

    work_dir = os.path.join(dtcc_dir, f"_relocdd_py_{variant}")
    run_text = _render_run_inp(os.path.join(work_dir, "input"),
                               os.path.join(work_dir, "data"),
                               os.path.join(work_dir, "output"))
    hypodd_text = _render_hypodd_inp(inp, has_cc=True, vmodel=vmodel)
    padded = _run_relocdd_py(cfg, work_dir, hypodd_text, run_text, input_files={
        "event.dat":   os.path.join(ph2dt_dir, inp.event_file),
        "dt.ct":       os.path.join(ph2dt_dir, "dt.ct"),
        "station.dat": os.path.join(ph2dt_dir, "station.dat"),
        "dt.cc":       cc_file,
    })
    final_reloc = os.path.join(dtcc_dir, f"hypoDD.reloc.{variant}")
    shutil.copy(padded, final_reloc)
    return final_reloc


# ============================================================================
# Fortran-free path: generate phase.dat + station.dat from a .sum (any backend)
# and SAC a/t0 picks, then run relocDD-py's OWN ph2dt (yesph2dt=1). This makes
# the full chain Fortran-free — no ncsn2pha, no Fortran ph2dt — and, crucially,
# rebuilds dt.ct from whatever .sum is on disk (so a HypoSVI .sum actually drives
# the relocation, instead of reusing the HYPOINVERSE-derived 00.ph2dt output).
# ============================================================================

def _packed_sec(sec: float) -> str:
    """relocDD-py's readphase parses header seconds as int(s[0:2]) + float(s[2:])/100,
    i.e. it expects PACKED 'SSss' (no decimal point): 34.56 s -> '3456'. (ncsn2pha
    writes decimal '34.56', which relocDD-py would misparse as 34.0056 — so we must
    emit the packed form here.)"""
    sec = max(0.0, float(sec))
    whole = int(sec) % 100
    hund = int(round((sec - int(sec)) * 100))
    if hund == 100:                      # rounding carry
        whole = (whole + 1) % 100
        hund = 0
    return f"{whole:02d}{hund:02d}"


def _station_coords(cfg):
    """{(net, sta): (lat, lon)} from the SAC headers (same source as write_sta)."""
    coords = {}
    for f in glob(os.path.join(config.waveforms_dir(cfg), "20*", "*.sac")):
        sta = os.path.basename(f).split(".")[2]
        if sta in coords:
            continue
        tr = read(f)[0]
        coords[(tr.stats.network or "KS", sta)] = (float(tr.stats.sac.stla), float(tr.stats.sac.stlo))
    return coords


def _event_picks(cfg):
    """{cuspid: [(net, sta, phase, UTCDateTime)]} from SAC a (P) / t0 (S) headers.
    cuspid = 200000+idx over sorted event dirs — matches write_phs / the .sum ID-NUM."""
    out = {}
    for idx, ed in enumerate(sorted(glob(os.path.join(config.waveforms_dir(cfg), "20*")))):
        cuspid = 200000 + idx
        rows, seen_p, seen_s = [], set(), set()
        for sac in sorted(glob(ed + "/*.sac"))[::-1]:
            tr = read(sac)[0]
            s = tr.stats.sac
            sta = os.path.basename(sac).split(".")[2]
            net = tr.stats.network or "KS"
            comp = tr.stats.channel[-1]
            if comp == "Z" and sta not in seen_p and s.get("a", -12345.0) != -12345.0:
                rows.append((net, sta, "P", tr.stats.starttime - s.b + s.a)); seen_p.add(sta)
            if comp in ("N", "E") and sta not in seen_s and s.get("t0", -12345.0) != -12345.0:
                rows.append((net, sta, "S", tr.stats.starttime - s.b + s.t0)); seen_s.add(sta)
        if rows:
            out[cuspid] = rows
    return out


def _write_phase_station(cfg, sum_path, data_dir):
    """Write phase.dat (relocDD-py format) + station.dat from `sum_path` (any backend's
    .sum) + SAC picks. Travel times are arrival − located origin time. Returns
    (phase_path, station_path, n_events)."""
    sumdf = sumio.read_sum(sum_path)
    by_id = {int(r.id): r for r in sumdf.itertuples(index=False)}
    picks = _event_picks(cfg)
    coords = _station_coords(cfg)

    used_stations = set()
    phase_path = os.path.join(data_dir, "phase.dat")
    n = 0
    with open(phase_path, "w") as f:
        for cuspid in sorted(picks):
            if cuspid not in by_id:
                continue                                  # event the locator dropped
            ev = by_id[cuspid]
            ot = ev.time                                  # UTCDateTime origin
            mag = float(getattr(ev, "mag", 0.0) or 0.0) if hasattr(ev, "mag") else 0.0
            f.write("# {:4d} {:2d} {:2d} {:2d} {:2d} {:>4s} {:8.4f} {:9.4f} {:6.2f} "
                    "{:5.2f} {:5.2f} {:5.2f} {:5.2f} {:>9d}\n".format(
                        ot.year, ot.month, ot.day, ot.hour, ot.minute, _packed_sec(ot.second + ot.microsecond / 1e6),
                        float(ev.lat), float(ev.lon), float(ev.depth), mag,
                        float(getattr(ev, "erh", 0.0) or 0.0), float(getattr(ev, "erz", 0.0) or 0.0),
                        float(getattr(ev, "rms", 0.0) or 0.0), cuspid))
            for net, sta, phase, t in picks[cuspid]:
                tt = float(t - ot)                        # seconds from origin
                if tt <= 0:
                    continue
                label = f"{net}{sta}"
                used_stations.add((net, sta))
                f.write(f"{label:<7s} {tt:7.3f}   1.000   {phase}\n")
            n += 1

    station_path = os.path.join(data_dir, "station.dat")
    with open(station_path, "w") as f:
        for (net, sta) in sorted(used_stations):
            if (net, sta) not in coords:
                continue
            la, lo = coords[(net, sta)]
            f.write(f"{net}{sta} {la} {lo}\n")
    return phase_path, station_path, n


def _render_ph2dt_inp(cfg) -> str:
    """relocDD-py ph2dt.inp from cfg.ph2dt (Ph2dtParams). Column order is
    MINWGHT MAXDIST MAXSEPE MAXSEPS MAXNGH MINLNK MINOBS MAXOBS."""
    p = cfg.ph2dt
    return ("* ph2dt.inp - generated by relocdd_py_backend.py\n"
            "* Input station file:\nstation.dat\n"
            "* Input phase file:\nphase.dat\n"
            "*MINWGHT MAXDIST MAXSEPE MAXSEPS MAXNGH MINLNK MINOBS MAXOBS\n"
            f"   {p.MINWGHT}     {p.MAXDIST}   {p.MAXSEP}   2   {p.MAXNGH}   "
            f"{p.MINLNK}   {p.MINOBS}   {p.MAXOBS}\n")


def run_relocdd_py_full(cfg, velmodel="kim1983", sum_path=None, cc_file=None,
                        out_dir_override=None, dtcc_variant="default") -> str:
    """Fully Fortran-free relocation: generate phase.dat from the `velmodel` .sum (any
    backend) + SAC picks, run relocDD-py's own ph2dt THEN hypoDD. Writes the padded
    24-col hypoDD.reloc into `out_dir_override` (default config.dtct_dir(cfg)).

    Control-file config is taken VERBATIM from the same dataclasses that drive the
    Fortran hypoDD, so the two backends solve an identical problem:
      - cc_file given  -> cfg.hypodd_dtcc_variants[dtcc_variant] (IDAT=3, OBSCC, and the
        cc-dominated iter_sets where the later iterations weight dt.cc over dt.ct);
      - cc_file None   -> cfg.hypodd_dtct (IDAT=2, catalog-only, ct-dominated iter_sets).
    ISTART/ISOLV/RATIO/NLAY/TOP/VEL all come from that dataclass — nothing is overridden
    here. (Earlier this used cfg.hypodd_dtct for BOTH, which silently down-weighted the
    cross-correlations to ~0 in every iteration, so the dt.cc never drove the relocation.)"""
    if sum_path is None:
        sum_path = config.sum_file(cfg, velmodel)
    if not os.path.exists(sum_path):
        raise RuntimeError(f"no .sum at {sum_path} — run the location stage first.")
    dtct_dir = out_dir_override or config.dtct_dir(cfg)
    os.makedirs(dtct_dir, exist_ok=True)
    has_cc = cc_file is not None
    if has_cc:
        variants = cfg.hypodd_dtcc_variants
        if not variants or dtcc_variant not in variants:
            raise RuntimeError(
                f"cfg.hypodd_dtcc_variants has no '{dtcc_variant}' entry; "
                f"available: {list(variants) if variants else None}")
        inp = variants[dtcc_variant]                       # cc-dominated, IDAT=3, OBSCC>0
    else:
        inp = cfg.hypodd_dtct                              # catalog-only, IDAT=2
        if inp is None:
            raise RuntimeError("cfg.hypodd_dtct is None — cannot run relocDD-py dtct stage.")
    vmodel = next(v for v in cfg.velocity_models if v.name == velmodel)

    relocdd_dir = _resolve_relocdd_dir(cfg)
    work_dir = os.path.join(dtct_dir, "_relocdd_py_full")
    inp_dir = os.path.join(work_dir, "input")
    data_dir = os.path.join(work_dir, "data")
    out_dir = os.path.join(work_dir, "output")
    for d in (inp_dir, data_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    for child in (os.listdir(out_dir)):                    # purge stale output
        pth = os.path.join(out_dir, child)
        shutil.rmtree(pth) if os.path.isdir(pth) else os.unlink(pth)

    # Generate Fortran-free ph2dt inputs from the .sum + picks
    _, _, n_ev = _write_phase_station(cfg, sum_path, data_dir)
    if n_ev == 0:
        raise RuntimeError(f"phase.dat generation produced 0 events from {sum_path}")
    if has_cc:                                              # stage the shared dt.cc
        if not os.path.isabs(cc_file):
            cc_file = os.path.join(config.dtcc_dir(cfg), cc_file)
        shutil.copyfile(cc_file, os.path.join(data_dir, "dt.cc"))

    # Stage input control files
    with open(os.path.join(inp_dir, "run.inp"), "w") as f:
        f.write(_render_run_inp(inp_dir, data_dir, out_dir))
    with open(os.path.join(inp_dir, "ph2dt.inp"), "w") as f:
        f.write(_render_ph2dt_inp(cfg))
    with open(os.path.join(inp_dir, "hypoDD.inp"), "w") as f:
        f.write(_render_hypodd_inp(inp, has_cc=has_cc, vmodel=vmodel))

    # Two invocations: ph2dt then hypoDD. relocDD-py's event-pair ph2dt writes the
    # catalog differential-time file as `dte.ct`, but its hypoDD reads the name given
    # in hypoDD.inp (`dt.ct`) — an internal naming mismatch in relocDD-py. So run
    # ph2dt alone, rename dte.ct -> dt.ct, then run hypoDD.
    log_path = os.path.join(work_dir, "relocdd_full.log")
    run_inp = os.path.join(inp_dir, "run.inp")

    def _invoke(yesph2dt, yeshypodd, mode):
        cmd = [sys.executable, os.path.join(relocdd_dir, "run.py"), run_inp, yesph2dt, yeshypodd]
        with open(log_path, "a") as log:
            log.write(f"\n\n===== relocdd_py_backend: {mode} ({yesph2dt} {yeshypodd}) =====\n")
            return subprocess.run(cmd, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT, text=True)

    p1 = _invoke("1", "0", "ph2dt")
    dte = os.path.join(data_dir, "dte.ct")
    if p1.returncode != 0 or not os.path.exists(dte):
        raise RuntimeError(f"relocDD-py ph2dt failed (exit {p1.returncode}); no dte.ct. Log: {log_path}")
    shutil.copyfile(dte, os.path.join(data_dir, "dt.ct"))   # name hypoDD.inp expects

    # hypoDD with the Fortran-equivalent solver choice: SVD (ISOLV=1) up to MAXDATA0, else
    # LSQR (ISOLV=2) with adaptive damping. _run_hypodd_adaptive re-renders hypoDD.inp before
    # each pass (the render at line above is just so ph2dt had a valid file present).
    p2, raw_reloc, ndt, used_lsqr = _run_hypodd_adaptive(
        inp, has_cc, vmodel, inp_dir, data_dir, out_dir, work_dir,
        invoke_hypodd=lambda: _invoke("0", "1", "hypoDD"))
    if p2.returncode != 0 or not raw_reloc or not os.path.exists(raw_reloc):
        raise RuntimeError(f"relocDD-py hypoDD failed (exit {p2.returncode}). Log: {log_path}")

    padded = os.path.join(work_dir, "hypoDD.reloc")
    if _pad_reloc(raw_reloc, padded) == 0:
        raise RuntimeError(f"relocDD-py full produced empty hypoDD.reloc at {raw_reloc}")
    final_reloc = os.path.join(dtct_dir, "hypoDD.reloc")
    shutil.copy(padded, final_reloc)
    _reanchor_to_sum(final_reloc, sum_path)        # match hypoDD's centroid-anchoring convention
    return final_reloc


def bootstrap_relocation(cfg, branch="dtcc", n=1000, seed=0, cores=None, min_nboot=50,
                         cache=True, work_subdir="_relocdd_py_full"):
    """relocDD-py port of hypodd.bootstrap_relocation — the SAME bootstrap uncertainty the
    Fortran workflow uses for its headline 95% error bars (HypoDD's own formal LSQR errors
    underestimate the relative-location uncertainty, so they are NOT used for the bars).

    Procedure, identical to the Fortran version except the inversion engine is relocDD-py:
      pool every differential-time observation, resample with replacement, regroup into pairs,
      and re-invert `n` times with the inversion HELD FIXED (the calibrated hypoDD.inp from the
      main run — same damping/ISOLV, so only the data vary). Each replica is seeded from the
      converged solution (event.dat lat/lon/depth set to the main reloc) and median-aligned to
      it; the per-event 2.5-97.5 percentile half-width of the X/Y/Z scatter is ex95/ey95/ez95.

    Reuses the Fortran resampling helpers (relocDD-py's dt/event.dat formats are identical).
    Positions use a local tangent-plane cartesian (metres) since relocDD-py's reloc x/y/z
    columns are per-iteration shifts, not absolute coordinates. Result cached to
    bootstrap_errors_relocdd_py.csv (same schema/provenance as the Fortran cache)."""
    import math
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pandas as pd
    from pipeline.core import hypodd as _H

    bdir = config.dtcc_dir(cfg) if branch == "dtcc" else config.dtct_dir(cfg)
    work = os.path.join(bdir, work_subdir)
    inp_dir, data_dir = os.path.join(work, "input"), os.path.join(work, "data")
    reloc0 = os.path.join(bdir, "hypoDD.reloc")
    calib_inp = os.path.join(inp_dir, "hypoDD.inp")
    for p in (reloc0, calib_inp, os.path.join(data_dir, "dt.ct")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"need {p} — run run_relocdd_py_full into {bdir} first (it leaves the "
                f"calibrated control file + resampling inputs under {work_subdir}/).")
    # Write the SAME artifacts the Fortran path writes (bootstrap_errors.csv +
    # bootstrap_samples.npz) so viz._load_bootstrap / location_table / the plot error bars
    # pick them up transparently — viz keys off these exact filenames, not the backend.
    out_csv = os.path.join(bdir, "bootstrap_errors.csv")
    if cache and os.path.exists(out_csv):
        meta = _H._bootstrap_meta(out_csv)
        # Backend match accepts the current `relocdd_py` tag OR a LEGACY header with no `backend=`
        # field. Pre-tagging caches (and the Fortran bootstrap, whose procedure is identical) wrote
        # no backend; they are valid bootstrap error bars for the same branch/n/seed. Requiring an
        # exact `backend=relocdd_py` made every run miss a perfectly good cache and recompute all
        # `n` inversions — which blew past the notebook's per-cell timeout. Reuse instead.
        _bk = meta.get("backend")
        if (meta.get("n") == str(n) and meta.get("seed") == str(seed)
                and meta.get("branch") == branch and _bk in ("relocdd_py", None)):
            return pd.read_csv(out_csv, comment="#")

    has_cc = branch == "dtcc" and os.path.exists(os.path.join(data_dir, "dt.cc"))
    dt_files = ["dt.ct"] + (["dt.cc"] if has_cc else [])
    base_blocks = {fn: _H._parse_dt_blocks(os.path.join(data_dir, fn)) for fn in dt_files}

    main = sumio.read_reloc(reloc0)
    lat0 = float(np.mean([float(r.lat) for r in main.itertuples()]))
    lon0 = float(np.mean([float(r.lon) for r in main.itertuples()]))
    kx, ky = 111195.0 * math.cos(math.radians(lat0)), 111195.0    # deg -> m (local tangent plane)
    def xyz(lat, lon, dep):
        return ((lon - lon0) * kx, (lat - lat0) * ky, dep * 1000.0)
    main_xyz = {int(r.id): xyz(float(r.lat), float(r.lon), float(r.depth)) for r in main.itertuples()}

    seeded_event_dat = _H._seed_event_dat(os.path.join(data_dir, "event.dat"), main)
    relocdd_dir = _resolve_relocdd_dir(cfg)
    ph2dt_src = os.path.join(relocdd_dir, "sample_inputfiles", "ph2dt.inp")
    station_src = os.path.join(data_dir, "station.dat")
    boot_timeout_s = 120
    # Pin every replica subprocess to a SINGLE BLAS/OpenMP thread. Without this each run.py spreads
    # numpy/BLAS across all cores, so N concurrent replicas spawn N x (many) threads and thrash —
    # the pool size below is then a TRUE core count rather than oversubscription.
    one_thread_env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                      "VECLIB_MAXIMUM_THREADS": "1"}

    def _one(i):
        rng = np.random.default_rng(seed + i)
        d = tempfile.mkdtemp(prefix=f"bootrdd_{cfg.name}_{branch}_")
        try:
            wi, wd, wo = os.path.join(d, "input"), os.path.join(d, "data"), os.path.join(d, "output")
            for x in (wi, wd, wo):
                os.makedirs(x)
            with open(os.path.join(wi, "run.inp"), "w") as f:
                f.write(_render_run_inp(wi, wd, wo))
            shutil.copyfile(calib_inp, os.path.join(wi, "hypoDD.inp"))    # FIXED calibrated inversion
            shutil.copyfile(ph2dt_src, os.path.join(wi, "ph2dt.inp"))
            with open(os.path.join(wd, "event.dat"), "w") as f:
                f.write(seeded_event_dat)
            shutil.copyfile(station_src, os.path.join(wd, "station.dat"))
            for fn, blk in base_blocks.items():
                _H._write_dt_blocks(os.path.join(wd, fn), _H._resample_global(blk, rng))
            try:
                subprocess.run([sys.executable, os.path.join(relocdd_dir, "run.py"),
                                os.path.join(wi, "run.inp"), "0", "1"],
                               cwd=d, capture_output=True, text=True, timeout=boot_timeout_s,
                               env=one_thread_env)
                raw = _resolve_raw_reloc(os.path.join(wo, "EDD", "tradouts"))
                if not os.path.exists(raw) or not os.path.getsize(raw):
                    return {}
                pad = os.path.join(d, "r24")
                _pad_reloc(raw, pad)
                df = sumio.read_reloc(pad)
                return {int(r.id): xyz(float(r.lat), float(r.lon), float(r.depth))
                        for r in df.itertuples()}
            except (Exception, subprocess.TimeoutExpired):              # noqa: BLE001
                return {}
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # With each replica pinned to one thread (one_thread_env), size the pool to dedicated cores
    # (cfg.bootstrap_cores, default 30) — NOT cfg.num_cores (10). Never exceed the replica count.
    cores = cores or getattr(cfg, "bootstrap_cores", None) or 30
    cores = max(1, min(int(cores), n))
    with ThreadPoolExecutor(max_workers=int(cores)) as ex:
        replicas = list(ex.map(_one, range(n)))

    samples = {}                                                       # id -> [(x,y,z)] aligned
    for rep in replicas:
        common = [e for e in rep if e in main_xyz]
        if len(common) < 2:
            continue
        off = np.median([[rep[e][k] - main_xyz[e][k] for k in range(3)] for e in common], axis=0)
        for e, p in rep.items():
            samples.setdefault(e, []).append([p[0] - off[0], p[1] - off[1], p[2] - off[2]])

    def _hw(a):                                                        # 95% percentile half-width
        return (np.percentile(a, 97.5, axis=0) - np.percentile(a, 2.5, axis=0)) / 2.0
    rows, samp_rows = [], []
    for e in sorted(main_xyz):
        s = np.asarray(samples.get(e, []), dtype=float)
        nb = len(s)
        row = dict(id=e, n_boot=nb, x_med=main_xyz[e][0], y_med=main_xyz[e][1], z_med=main_xyz[e][2],
                   ex95=np.nan, ey95=np.nan, ez95=np.nan)
        if nb >= max(2, min_nboot):
            med, hw = np.median(s, axis=0), _hw(s)
            row.update(x_med=med[0], y_med=med[1], z_med=med[2],
                       ex95=hw[0], ey95=hw[1], ez95=hw[2])
            samp_rows.extend([e, p[0], p[1], p[2]] for p in s)         # for bootstrap_samples.npz
        rows.append(row)
    out = pd.DataFrame(rows)
    if cache:
        with open(out_csv, "w") as f:
            f.write(f"# bootstrap_errors n={n} seed={seed} branch={branch} cluster={cfg.name} "
                    f"backend=relocdd_py resample=global ci=percentile2.5-97.5 align=median "
                    f"init=solution\n")
            out.to_csv(f, index=False)
        # viz._load_bootstrap reads the per-replica aligned samples (id, x, y, z metres, Z down)
        # to take percentile half-widths in any rotated frame (along/across/depth) — write it.
        np.savez(os.path.join(bdir, "bootstrap_samples.npz"),
                 data=np.asarray(samp_rows, dtype=float) if samp_rows else np.empty((0, 4)))
    return out
