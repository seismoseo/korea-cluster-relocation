"""Focal-mechanism stage: SKHASH double-couple inversion from PhaseNet+ data.

Only meaningful after a **phasenet_plus** picking run — those picks carry the first-motion
`Polarity` and per-pick `Amplitude` that SKHASH needs (the SeisBench `stead` picker emits
neither). This stage:

  1. reads the HYPOINVERSE `.sum` locations (cuspid = `hypoinverse.write_phs` scheme) and the
     per-event phasenet_plus picks,
  2. writes SKHASH inputs (catalog, P-polarity, S/P-ratio, station list, velocity model) into
     `runs/<cluster>/3.FocalMech/<velmodel>/IN/`,
  3. runs SKHASH (which ray-traces takeoff angles from the velocity model and computes azimuths
     from the station geometry — `compute_mech.py` stfile path), and
  4. keeps the well-constrained (quality A/B) mechanisms.

SKHASH is an EXTERNAL tool (`config.SKHASH_DIR`, override via $SKHASH_DIR) — not vendored.
Polarity weighting: SKHASH treats `p_polarity` sign as up/down and |p_polarity| as the weight,
dropping |value| < `min_polarity_weight` — so we pass the raw PhaseNet+ `phase_polarity`.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.core import sumio, waveforms


def _cuspid_maps(cfg):
    """{cuspid -> event_id}, replicating hypoinverse.write_phs (enumerate sorted event dirs)."""
    event_dirs = sorted(glob.glob(os.path.join(config.waveforms_dir(cfg), "20*")))
    return {cfg.cuspid_offset + i: os.path.basename(ed) for i, ed in enumerate(event_dirs)}


def _vmodel_depth_vp(cfg, velmodel):
    """(depth_km, Vp) rows ascending in depth, from the cluster VelModel.p_rows (vp, depth_top)."""
    vm = next((v for v in cfg.velocity_models if v.name == velmodel), None)
    if vm is None or not vm.p_rows:
        raise ValueError(f"velocity model {velmodel!r} has no p_rows for SKHASH takeoff tracing")
    return sorted(((float(d), float(v)) for v, d in vm.p_rows))


def _pre_p_noise(wf_dir, eid, net, sta, sensor, p_time, pre=(2.0, 0.3)):
    """Peak |amplitude| over the 3 components in a pre-P window [P-pre[0], P-pre[1]] s — the noise
    level for the S/P SNR gate. Mirrors how PhaseNet+ measures signal amplitude (peak over comps).
    Returns 0.0 if the waveforms can't be read (→ the ratio is then dropped)."""
    from obspy import read, UTCDateTime
    pt = UTCDateTime(str(p_time))
    peak = 0.0
    for c in ("Z", "N", "E"):
        fs = glob.glob(f"{wf_dir}/{eid}.*.{sta}.{sensor}{c}.sac")
        if not fs:
            continue
        try:
            w = read(fs[0])[0].detrend("demean").slice(pt - pre[0], pt - pre[1])
            if len(w.data):
                peak = max(peak, float(np.max(np.abs(w.data))))
        except Exception:  # noqa: BLE001
            pass
    return peak


def run_focal_mechanism(cfg, velmodel=None, num_cpus=1) -> dict:
    velmodel = velmodel or cfg.fm_velmodel
    in_dir, out_dir = config.fm_in_dir(cfg, velmodel), config.fm_out_dir(cfg, velmodel)
    config.assert_writable(config.fm_dir(cfg, velmodel))
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    sumf = config.sum_file(cfg, velmodel)
    if not os.path.exists(sumf):
        raise FileNotFoundError(f"missing {sumf} — run the hypoinverse stage first")
    sm = sumio.read_sum(sumf)
    cusp2eid = _cuspid_maps(cfg)
    cat_mag = {e["event_id"]: e.get("mag", 0.0) for e in waveforms.load_catalog(cfg)}
    used = pd.read_csv(config.used_stations_csv(cfg))
    coord = {(str(r.Network), str(r.Code)): (float(r.Latitude), float(r.Longitude),
                                             float(r.Elevation), str(r.Sensor))
             for r in used.itertuples()}

    # ---- catalog (SKHASH catfile) ----
    cat = pd.DataFrame(dict(
        time=[t.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for t in sm.time],
        latitude=sm.lat, longitude=sm.lon, depth=sm.depth,
        horz_uncert_km=sm.erh.fillna(0.5).clip(lower=0.1),
        vert_uncert_km=sm.erz.fillna(1.0).clip(lower=0.1),
        mag=[cat_mag.get(cusp2eid.get(int(i)), 0.0) for i in sm.id],
        event_id=sm.id.astype(int)))
    catfile = os.path.join(in_dir, "eq_catalog.csv")
    cat.to_csv(catfile, index=False)

    # ---- per-event polarity + S/P ratio from the phasenet_plus picks ----
    # Data-quality gates: (Q1) drop polarities from P picks with probability < fm_min_pick_prob;
    # (Q2) drop S/P ratios whose P or S SNR (vs a pre-P noise window) < fm_sp_min_snr.
    pol_rows, amp_rows, st_used = [], [], {}
    n_pol_lowprob = n_sp_lowsnr = 0
    for cusp in sm.id.astype(int):
        eid = cusp2eid.get(int(cusp))
        pf = config.picks_csv(cfg, eid) if eid else None
        if not pf or not os.path.exists(pf):
            continue
        pk = pd.read_csv(pf)
        if "Polarity" not in pk.columns:
            raise ValueError(f"{pf} has no Polarity column — focal mechanisms need a "
                             f"phasenet_plus picking run (picker_weights='phasenet_plus').")
        wfdir = config.event_wf_dir(cfg, eid)
        for (net, sta), g in pk.groupby(["Network", "Station"]):
            key = (str(net), str(sta))
            if key not in coord:
                continue
            lat, lon, elev, sensor = coord[key]
            chan = f"{sensor}Z"
            st_used[key] = (lat, lon, elev, chan)
            P, S = g[g.Phase == "P"], g[g.Phase == "S"]
            if not len(P):
                continue
            prow = P.iloc[0]
            # (Q1) pick-probability gate on the polarity
            if float(prow.Probability) < cfg.fm_min_pick_prob:
                n_pol_lowprob += 1
            else:
                pol = 0.0 if pd.isna(prow.Polarity) else float(prow.Polarity)
                pol_rows.append(dict(event_id=int(cusp), station=sta, network=net,
                                     location="--", channel=chan, p_polarity=round(pol, 3)))
            # (Q2) S/P amplitude ratio with an SNR gate (pre-P noise from the SAC)
            if cfg.fm_use_sp_ratio and len(S):
                pamp, samp = float(prow.Amplitude), float(S.iloc[0].Amplitude)
                if pamp > 0 and samp > 0:
                    noise = _pre_p_noise(wfdir, eid, net, sta, sensor, prow.Time)
                    snr_ok = noise > 0 and min(pamp, samp) / noise >= cfg.fm_sp_min_snr
                    if snr_ok:
                        amp_rows.append(dict(event_id=int(cusp), station=sta, network=net,
                                             location="--", channel=chan,
                                             sp_ratio=round(samp / pamp, 5)))
                    else:
                        n_sp_lowsnr += 1

    polfile = os.path.join(in_dir, "pol.csv")
    pd.DataFrame(pol_rows).to_csv(polfile, index=False)
    ampfile = os.path.join(in_dir, "amp.csv")
    pd.DataFrame(amp_rows).to_csv(ampfile, index=False)

    # ---- station list (SKHASH stfile -> azimuth/takeoff geometry) ----
    stfile = os.path.join(in_dir, "stations.csv")
    pd.DataFrame([dict(network=net, station=sta, location="--", channel=chan,
                       latitude=lat, longitude=lon, elevation=elev)
                  for (net, sta), (lat, lon, elev, chan) in sorted(st_used.items())]
                 ).to_csv(stfile, index=False)

    # ---- velocity model (depth, Vp) ----
    vmfile = os.path.join(in_dir, f"{velmodel}.txt")
    with open(vmfile, "w") as f:
        f.write("# Depth (km), Vp (km/s)\n")
        for d, v in _vmodel_depth_vp(cfg, velmodel):
            f.write(f"{d}, {v}\n")

    # ---- control file (absolute paths so cwd=SKHASH_DIR is irrelevant for I/O) ----
    outfile1 = os.path.join(out_dir, "out.csv")
    ctl = os.path.join(config.fm_dir(cfg, velmodel), "control_file.txt")
    params = {
        "input_format": "skhash", "catfile": catfile, "fpfile": polfile,
        "stfile": stfile, "vmodel_paths": vmfile,
        "outfile1": outfile1, "outfile_pol_info": os.path.join(out_dir, "out_polinfo.csv"),
        "outfolder_plots": out_dir,
        "npolmin": cfg.fm_npolmin, "min_polarity_weight": cfg.fm_min_polarity_weight,
        "max_agap": cfg.fm_max_agap, "max_pgap": cfg.fm_max_pgap,
        "delmax": cfg.fm_delmax_km, "num_cpus": num_cpus,
    }
    if cfg.fm_use_sp_ratio and amp_rows:
        params["ampfile"] = ampfile
    with open(ctl, "w") as f:
        for k, v in params.items():
            f.write(f"${k}\n{v}\n\n")

    # ---- run SKHASH ----
    proc = subprocess.run([sys.executable, os.path.join(config.SKHASH_DIR, "SKHASH.py"), ctl],
                          cwd=config.SKHASH_DIR, text=True, capture_output=True)
    if not os.path.exists(outfile1):
        print("---- SKHASH stdout (tail) ----\n", proc.stdout[-2500:])
        print("---- SKHASH stderr (tail) ----\n", proc.stderr[-1500:])
        raise RuntimeError("SKHASH produced no output file")

    # ---- parse + filter ----
    mech = pd.read_csv(outfile1)
    mech.insert(1, "cuspid", mech["event_id"])
    mech["event_id"] = mech["event_id"].map(lambda c: cusp2eid.get(int(c), str(c)))
    mech.to_csv(config.fm_mech_csv(cfg, velmodel), index=False)
    keep = mech[mech["quality"].isin(list(cfg.fm_quality_keep))]
    breakdown = mech["quality"].value_counts().reindex(list("ABCD")).fillna(0).astype(int).to_dict()
    print(f"[focal_mechanism] gates: dropped {n_pol_lowprob} polarities (pick prob < "
          f"{cfg.fm_min_pick_prob}), {n_sp_lowsnr} S/P ratios (SNR < {cfg.fm_sp_min_snr})")
    print(f"[focal_mechanism] {cfg.name}/{velmodel}: {len(mech)} mechanisms "
          f"(polarities n={len(pol_rows)}, S/P n={len(amp_rows)}); quality {breakdown}; "
          f"{len(keep)} high-confidence [{'/'.join(cfg.fm_quality_keep)}] "
          f"-> {os.path.relpath(config.fm_mech_csv(cfg, velmodel), config.PROJECT_ROOT)}")
    return {r.event_id: r.quality for r in mech.itertuples()}
