"""
HYPOINVERSE absolute location (ports `1.HypoInv/{01.Make_PHS_File,02.Make_STA_file}`
+ the per-cluster `<Region>.sh` run wrapper + the YijianZhou `mk_sta` conversion).

  write_sta(cfg)            -> STA/<Region>.sta (CSV) + STA/<Region>_hyp.sta (HYPOINVERSE)
  write_phs(cfg)            -> PHS/<Region>.phs (COP3, picks from SAC a/t0, cuspid 200000+i)
  run_hypoinverse(cfg, ...) -> per velocity model: <vm>/<Region>.{sum,arc,prt} via hyp1.40

Velocity-model `.crh` files are symlinked from the baseline (`VelModel.source_dir`)
for byte-identical input, or regenerated from the model's layer rows if no source.
hyp1.40 is run with cwd = the run's 1.HypoInv dir so its relative paths resolve.
"""
from __future__ import annotations

import os
import subprocess
from glob import glob

from obspy import read
from obspy.geodetics.base import gps2dist_azimuth

from pipeline import config
from pipeline.core import waveforms


# ---------------------------------------------------------- small formatters
def _deg_min_hundredths(angle):
    """NB01 degreetominute: integer degrees + minutes*100 (truncated)."""
    return int(angle), int(100 * 60 * (angle - int(angle)))


def _weight(bins, epi_km, phase):
    idx = 1 if phase == "P" else 2
    for row in bins:
        if epi_km < row[0]:
            return row[idx]
    return bins[-1][idx]


# --------------------------------------------------------------------- STA
def write_sta(cfg):
    """Write STA/<Region>.sta (CSV) and STA/<Region>_hyp.sta (HYPOINVERSE fmt 2)."""
    coords = {}  # sta -> (net, stla, stlo)
    for f in glob(os.path.join(config.waveforms_dir(cfg), "20*", "*.sac")):
        sta = os.path.basename(f).split(".")[2]
        if sta in coords:
            continue
        tr = read(f)[0]
        coords[sta] = (tr.stats.network or "KS", tr.stats.sac.stla, tr.stats.sac.stlo)

    sta_dir = config.assert_writable(config.sta_dir(cfg))
    os.makedirs(sta_dir, exist_ok=True)

    with open(config.sta_file(cfg), "w") as fh:
        for sta in sorted(coords):
            net, la, lo = coords[sta]
            fh.write(f"{net}.{sta},{la},{lo},0.0,100.0\n")

    with open(config.sta_hyp_file(cfg), "w") as fh:
        for sta in sorted(coords):
            net, la, lo = coords[sta]
            la_a, lo_a = abs(la), abs(lo)
            lat = "{:2} {:7.4f}{}".format(int(la_a), 60 * (la_a - int(la_a)), "N")
            lon = "{:3} {:7.4f}{}".format(int(lo_a), 60 * (lo_a - int(lo_a)), "E")
            fh.write("{:<5} {}  HHZ  {}{}{:4}\n".format(sta, net, lat, lon, 0))
    return config.sta_hyp_file(cfg)


# --------------------------------------------------------------------- PHS
def write_phs(cfg):
    """Write PHS/<Region>.phs (COP3) from the picks stored in the SAC a/t0 headers."""
    catalog = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
    bins = cfg.phs_dist_weight_bins
    os.makedirs(config.assert_writable(config.phs_dir(cfg)), exist_ok=True)
    event_dirs = sorted(glob(os.path.join(config.waveforms_dir(cfg), "20*")))

    with open(config.phs_file(cfg), "w") as f:
        for idx, ed in enumerate(event_dirs):
            eid = os.path.basename(ed)
            ev = catalog.get(eid)
            if ev is None:
                continue
            la_d, la_m = _deg_min_hundredths(ev["lat"])
            lo_d, lo_m = _deg_min_hundredths(ev["lon"])
            f.write(f"{eid}00{la_d}N{str(la_m).zfill(4)}{lo_d}E{str(lo_m).zfill(4)}\n")

            seen_p, seen_s = set(), set()
            for sac in sorted(glob(ed + "/*.sac"))[::-1]:
                tr = read(sac)[0]
                s = tr.stats.sac
                sta = os.path.basename(sac).split(".")[2]
                net = (tr.stats.network or "KS")
                chan3 = tr.stats.channel[:3]
                comp = tr.stats.channel[-1]
                if comp == "Z" and sta not in seen_p and s.get("a", -12345.0) != -12345.0:
                    epi = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[0] / 1000.0
                    pw = _weight(bins, epi, "P")
                    ot = tr.stats.starttime - s.b + s.a
                    f.write(f"{sta.ljust(5)}{net.ljust(4)}{chan3.ljust(4)}{'IP'.ljust(3)}{pw}"
                            f"{ot.year}{str(ot.month).zfill(2)}{str(ot.day).zfill(2)}"
                            f"{str(ot.hour).zfill(2)}{str(ot.minute).zfill(2).ljust(3)}"
                            f"{str(ot.second).zfill(2)}{str(ot.microsecond).zfill(6)[:2]}\n")
                    seen_p.add(sta)
                if comp in ("N", "E") and sta not in seen_s and s.get("t0", -12345.0) != -12345.0:
                    epi = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[0] / 1000.0
                    sw = _weight(bins, epi, "S")
                    ot = tr.stats.starttime - s.b + s.t0
                    f.write(f"{sta.ljust(5)}{net.ljust(4)}{chan3.ljust(4)}    "
                            f"{ot.year}{str(ot.month).zfill(2)}{str(ot.day).zfill(2)}"
                            f"{str(ot.hour).zfill(2)}{str(ot.minute).zfill(2).ljust(15)}"
                            f"{str(ot.second).zfill(2)}{str(ot.microsecond).zfill(6)[:2]}"
                            f"{'ES'.ljust(3)}{sw}\n")
                    seen_s.add(sta)
            f.write(" " * 66 + "200" + str(idx).zfill(3) + "\n")
    return config.phs_file(cfg)


# --------------------------------------------------------- velocity models
def _write_crh(path, header, rows):
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for vel, dep in rows:
            fh.write(f" {vel:.2f} {dep:.2f}\n")


def _provision_crh(cfg, vmodel):
    d = config.assert_writable(config.velmodel_dir(cfg, vmodel.name))
    os.makedirs(d, exist_ok=True)
    for suf, rows, lbl in (("_p.crh", vmodel.p_rows, "P"), ("_s.crh", vmodel.s_rows, "S")):
        dst = os.path.join(d, vmodel.name + suf)
        if vmodel.source_dir:
            src = os.path.join(vmodel.source_dir, vmodel.name + suf)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        else:
            _write_crh(dst, f"{vmodel.name} {lbl} wave velocity", rows)


# --------------------------------------------------------- hyp1.40 control
HYP_TEMPLATE = """
REP T T
CON {CON}
MIN {MIN}
ZTR {ZTR0} {ZTR1}
DIS {DIS}
RMS {RMS}
ERF T
TOP F
LST {LST}
KPR {KPR}
H71 {H71}
STA 'STA/{region}_hyp.sta'
CRH 1 '{model}/{model}_p.crh'
CRH 2 '{model}/{model}_s.crh'
SAL 1 2
PHS 'PHS/{region}.phs'
FIL
PRT '{model}/{region}.prt'
SUM '{model}/{region}.sum'
ARC '{model}/{region}.arc'
LOC
STO
"""


def run_hypoinverse_model(cfg, vmodel):
    """Locate all events with one velocity model; return the .sum path."""
    _provision_crh(cfg, vmodel)
    hc = cfg.hyp_control
    cmds = HYP_TEMPLATE.format(
        CON=hc.CON, MIN=hc.MIN, ZTR0=hc.ZTR[0], ZTR1=hc.ZTR[1],
        DIS=" ".join(map(str, hc.DIS)), RMS=" ".join(map(str, hc.RMS)),
        LST=" ".join(map(str, hc.LST)), KPR=hc.KPR, H71=" ".join(map(str, hc.H71)),
        region=cfg.region, model=vmodel.name,
    )
    hyp = config.hyp_dir(cfg)
    subprocess.run(["hyp1.40"], input=cmds.encode(), cwd=hyp, check=True,
                   stdout=subprocess.DEVNULL)
    # hyp1.40 leaves fort.* scratch files behind
    for f in glob(os.path.join(hyp, "fort.*")):
        os.remove(f)
    return config.sum_file(cfg, vmodel.name)


def run_hypoinverse(cfg, velmodels=None, write_inputs=True) -> dict:
    """Write STA + PHS (once) and locate over the requested velocity models."""
    if write_inputs:
        write_sta(cfg)
        write_phs(cfg)
    models = cfg.velocity_models
    if velmodels is not None:
        wanted = set(velmodels)
        models = [m for m in models if m.name in wanted]
    return {m.name: run_hypoinverse_model(cfg, m) for m in models}
