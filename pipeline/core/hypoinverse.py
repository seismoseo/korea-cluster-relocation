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


def _weight_prob(prob, bins):
    """Map a PhaseNet+ pick probability in [0, 1] to a HypoInverse weight code (0=full,
    ..., 4=drop). `bins` is descending-by-threshold: ((0.90, 0), (0.70, 1), ..., (0.00, 4)).
    Falls back to the last bin (code 4) for None/NaN."""
    if prob is None:
        return bins[-1][1]
    for threshold, code in bins:
        if prob >= threshold:
            return code
    return bins[-1][1]


def _load_picks_csv(cfg, event_id):
    """Return {(station, phase): probability} for one event, or {} if the CSV is absent
    or unreadable. Picks CSVs are written by `pipeline/core/picking.py:pick_event*`
    alongside the SAC headers (columns: Event_ID, Network, Station, Phase, Time,
    Probability, Polarity, Amplitude)."""
    csv_path = config.picks_csv(cfg, event_id)
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        out = {}
        for r in df.itertuples(index=False):
            try:
                p = float(r.Probability)
            except (TypeError, ValueError):
                continue
            if p == p:                                            # filter NaN
                out[(str(r.Station), str(r.Phase))] = p
        return out
    except Exception:                                              # noqa: BLE001
        return {}


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
    """Write PHS/<Region>.phs (COP3) from the picks stored in the SAC a/t0 headers.

    Weight-code column 18 of each phase line is set per `cfg.phs_weight_scheme`:
      - "distance" (default for source clusters): epicentral-distance bins
        (`cfg.phs_dist_weight_bins`) -- byte-identical to v0.5.x behavior.
      - "probability" (default for PocketQuake-scaffolded clusters): PhaseNet+ pick
        probability bins (`cfg.phs_prob_weight_bins`). Falls back to distance for any
        pick whose probability is missing in the picks CSV (e.g. when the picker
        didn't write one). The scheme decision is logged once per call.
    """
    catalog = {e["event_id"]: e for e in waveforms.load_catalog(cfg)}
    dist_bins = cfg.phs_dist_weight_bins
    scheme = getattr(cfg, "phs_weight_scheme", "distance")
    prob_bins = getattr(cfg, "phs_prob_weight_bins", ())
    os.makedirs(config.assert_writable(config.phs_dir(cfg)), exist_ok=True)
    event_dirs = sorted(glob(os.path.join(config.waveforms_dir(cfg), "20*")))

    n_prob_p = n_prob_s = n_dist_fallback_p = n_dist_fallback_s = 0
    with open(config.phs_file(cfg), "w") as f:
        for idx, ed in enumerate(event_dirs):
            eid = os.path.basename(ed)
            ev = catalog.get(eid)
            if ev is None:
                continue
            picks_lookup = _load_picks_csv(cfg, eid) if scheme == "probability" else {}
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
                    if scheme == "probability":
                        prob = picks_lookup.get((sta, "P"))
                        if prob is not None:
                            pw = _weight_prob(prob, prob_bins); n_prob_p += 1
                        else:
                            pw = _weight(dist_bins, epi, "P"); n_dist_fallback_p += 1
                    else:
                        pw = _weight(dist_bins, epi, "P")
                    ot = tr.stats.starttime - s.b + s.a
                    f.write(f"{sta.ljust(5)}{net.ljust(4)}{chan3.ljust(4)}{'IP'.ljust(3)}{pw}"
                            f"{ot.year}{str(ot.month).zfill(2)}{str(ot.day).zfill(2)}"
                            f"{str(ot.hour).zfill(2)}{str(ot.minute).zfill(2).ljust(3)}"
                            f"{str(ot.second).zfill(2)}{str(ot.microsecond).zfill(6)[:2]}\n")
                    seen_p.add(sta)
                if comp in ("N", "E") and sta not in seen_s and s.get("t0", -12345.0) != -12345.0:
                    epi = gps2dist_azimuth(s.evla, s.evlo, s.stla, s.stlo)[0] / 1000.0
                    if scheme == "probability":
                        prob = picks_lookup.get((sta, "S"))
                        if prob is not None:
                            sw = _weight_prob(prob, prob_bins); n_prob_s += 1
                        else:
                            sw = _weight(dist_bins, epi, "S"); n_dist_fallback_s += 1
                    else:
                        sw = _weight(dist_bins, epi, "S")
                    ot = tr.stats.starttime - s.b + s.t0
                    f.write(f"{sta.ljust(5)}{net.ljust(4)}{chan3.ljust(4)}    "
                            f"{ot.year}{str(ot.month).zfill(2)}{str(ot.day).zfill(2)}"
                            f"{str(ot.hour).zfill(2)}{str(ot.minute).zfill(2).ljust(15)}"
                            f"{str(ot.second).zfill(2)}{str(ot.microsecond).zfill(6)[:2]}"
                            f"{'ES'.ljust(3)}{sw}\n")
                    seen_s.add(sta)
            # Cuspid = cuspid_offset + sorted-dir index. Byte-identical to the old
            # '200'+zfill(3) form for idx <= 999, but doesn't cap there — dense
            # sequences (e.g. Buan 2024) exceed 999 events per cluster.
            f.write(" " * 66 + str(cfg.cuspid_offset + idx) + "\n")
    if scheme == "probability":
        print(f"[write_phs] {cfg.name}: probability-weighted "
              f"P {n_prob_p} prob + {n_dist_fallback_p} dist-fallback, "
              f"S {n_prob_s} prob + {n_dist_fallback_s} dist-fallback")
    return config.phs_file(cfg)


# --------------------------------------------------------- velocity models
def _write_crh(path, header, rows):
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for vel, dep in rows:
            fh.write(f" {vel:.2f} {dep:.2f}\n")


def _provision_crh(cfg, vmodel):
    """Materialise <model>_{p,s}.crh under the run's velmodel dir.

    If `vmodel.source_dir` is set AND the source file exists, symlink it (byte-identical
    inputs vs the baseline source-cluster runs). Otherwise fall back to writing the CRH
    from the in-config `p_rows` / `s_rows` -- this is what auto-scaffolded PocketQuake
    clusters need, since the source-root they generate doesn't carry a hand-curated
    `1.HypoInv/<model>/` tree. Previously the `if vmodel.source_dir:` check was a string
    truthiness test, so an absent source produced a *broken symlink*; hyp1.40 then printed
    `*** ERROR - CRUST FILE DOES NOT EXIST` and silently fell through with no velocity
    model, which mis-located shallow tight clusters like chungju.
    """
    d = config.assert_writable(config.velmodel_dir(cfg, vmodel.name))
    os.makedirs(d, exist_ok=True)
    for suf, rows, lbl in (("_p.crh", vmodel.p_rows, "P"), ("_s.crh", vmodel.s_rows, "S")):
        dst = os.path.join(d, vmodel.name + suf)
        src = (os.path.join(vmodel.source_dir, vmodel.name + suf)
               if vmodel.source_dir else None)
        if os.path.lexists(dst):
            os.remove(dst)
        if src and os.path.isfile(src):
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
    """Write STA + PHS (once) and locate over the requested velocity models.

    Dispatches on cfg.loc_backend: "hypoinverse" (Fortran hyp1.40, default) or
    "hyposvi" (Python). The hyposvi adapter is not wired yet — fail loudly rather
    than silently producing Fortran results, so --loc-backend hyposvi never lies."""
    backend = getattr(cfg, "loc_backend", "hypoinverse")
    if backend == "hyposvi":
        from pipeline.core import hyposvi_backend
        return hyposvi_backend.run_hyposvi(cfg, velmodels=velmodels)
    if backend != "hypoinverse":
        raise ValueError(f"unknown loc_backend {backend!r} (expected 'hypoinverse' | 'hyposvi')")
    if write_inputs:
        write_sta(cfg)
        write_phs(cfg)
    models = cfg.velocity_models
    if velmodels is not None:
        wanted = set(velmodels)
        models = [m for m in models if m.name in wanted]
    return {m.name: run_hypoinverse_model(cfg, m) for m in models}
