"""
Gwangyang sequence (2021 + 2026 re-run) — the development anchor cluster.

11 events, KS network, KMA SAC archive (`kma_waveforms/`). Has manual picks and a
Seo2022 multi-velocity-model study, so it exercises the framework's N-velocity-model
support. All values below are read from the existing baseline:
  - HYPOINVERSE control block: Gwangyang_sequence/1.HypoInv/Gwangyang.sh
  - velocity models: 1.HypoInv/{kim1983,kim2011}/*.crh (symlinked, not regenerated)
  - ph2dt / hypoDD params: 2.HypoDD/00.ph2dt/ph2dt.inp, 01.dt.ct/hypoDD.inp
"""
import os

from pipeline import config
from pipeline.config import (
    ClusterConfig, VelModel, HypControl, Ph2dtParams, HypoDDInp,
)

SRC = os.path.join(config.PROJECT_ROOT, "Gwangyang_sequence")
_HYP = os.path.join(SRC, "1.HypoInv")

# The Seo2022 velocity-model sensitivity set — reuse the existing .crh by symlink.
# (kim1983/kim2011 are omitted here: they are the standalone gated models below and
#  the Seo2022 copies are byte-identical to 1.HypoInv/{kim1983,kim2011}.)
_SEO2022 = (
    "ak135", "chang2006", "iasp91", "kim1983_mod", "kim1983_twomod",
    "kim1999", "kim2011_GB", "kim2011_GM", "kim2011_OB",
)

# HypoDD iteration weight rows (from 01.dt.ct/hypoDD.inp):
# (NITER, WTCCP, WTCCS, WRCC, WDCC, WTCTP, WTCTS, WRCT, WDCT, DAMP)
_DTCT_ITERS = (
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5, -9, -9, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  8,  6, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  6,  4, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  5,  3, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  3,  1, 8),
)
# dt.cc schedule (from 02.dt.cc/hypoDD.inp): catalog-dominant for iters 1-3, then
# cc-dominant for iters 4-7 (DAMP 8->6). Paired with OBSCC=4. Distinct from _DTCT_ITERS.
_DTCC_ITERS = (
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,  -9, -9, 8),
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,   8,  6, 8),
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,   6,  4, 8),
    (4, 1.0,  0.8,    6,   4,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    4,   2,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    3,   1,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    2,   0.5, 0.01, 0.005, -9, -9, 6),
)

CONFIG = ClusterConfig(
    name="gwangyang",
    region="Gwangyang",
    src_root=SRC,
    event_catalog_csv=os.path.join(SRC, "event_catalog", "event_catalog.csv"),
    station_master_csvs=(os.path.join(SRC, "station_table", "KS_station.csv"),),
    epicenter=(35.00, 127.72),
    radius_km=100.0,
    region_bounds=(34.4, 35.6, 126.9, 128.6),
    kst_offset_hours=9,

    # waveform source: KMA SAC archive. File-globs are relative to kma_waveforms/<event>/
    # with a {comp} placeholder (Z/N/E); availability is checked on Z only.
    wf_source="kma_archive",
    kma_archive_glob={
        "HH": "*.v/SAC/HH/*HH{comp}*SAC",
        "HG": "*.a/SAC/HG/*HG{comp}*SAC",
        "EL": "*.v/SAC/EL/*EL{comp}*SAC",
    },
    stp_sac_root=None,
    sensor_priority=("HH", "HG", "EL"),
    target_sampling_hz=100.0,

    # AI picking — PhaseNet 'stead', bandpass 1-40 Hz (matches 02.ML_picking notebook).
    picker_weights="stead",
    p_threshold=0.2,
    s_threshold=0.2,
    sp_max_gap_s=15.0,
    pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),

    # HYPOINVERSE control (Gwangyang.sh): DIS 4 50 1 3, RMS 4 .12 2 4, ZTR 10 F.
    hyp_control=HypControl(
        CON=50, MIN=4, ZTR=(10, "F"), DIS=(4, 50, 1, 3),
        RMS=(4, 0.12, 2, 4), H71=(4, 1, 3), KPR=3, LST=(2, 0, 1),
    ),
    velocity_models=(
        VelModel("kim1983",
                 p_rows=((5.98, 0.00), (6.38, 15.0), (7.95, 32.0)),
                 s_rows=((3.40, 0.00), (3.79, 15.0), (4.58, 32.0)),
                 source_dir=os.path.join(_HYP, "kim1983")),
        VelModel("kim2011",
                 p_rows=((5.63, 0.00), (6.17, 7.29), (6.58, 20.7), (7.77, 31.3)),
                 s_rows=((3.40, 0.00), (3.60, 7.29), (3.70, 20.7), (4.45, 31.3)),
                 source_dir=os.path.join(_HYP, "kim2011")),
        *[VelModel(m, source_dir=os.path.join(_HYP, "Seo2022", m)) for m in _SEO2022],
    ),

    # HypoDD
    ph2dt=Ph2dtParams(MINWGHT=0, MAXDIST=200, MAXSEP=10, MAXNGH=200,
                      MINLNK=1, MINOBS=1, MAXOBS=500),
    hypodd_dtct=HypoDDInp(
        idat=2, ipha=3, dist=500, obscc=0, obsct=0, istart=1, isolv=1,
        iter_sets=_DTCT_ITERS, nlay=3, ratio=1.73,
        top=(0.0, 15.0, 32.0), vel=(5.98, 6.38, 7.95),
        cc_file=None, event_file="event.dat",
    ),
    hypodd_dtcc_variants={
        "default": HypoDDInp(idat=3, ipha=3, dist=500, obscc=4, istart=1, isolv=1,
                             iter_sets=_DTCC_ITERS, nlay=3, ratio=1.73,
                             top=(0.0, 15.0, 32.0), vel=(5.98, 6.38, 7.95),
                             cc_file="dt.cc_0.7_combined", event_file="event.dat"),
        "no_main": HypoDDInp(idat=3, ipha=3, dist=500, obscc=4, istart=1, isolv=1,
                             iter_sets=_DTCC_ITERS, nlay=3, ratio=1.73,
                             top=(0.0, 15.0, 32.0), vel=(5.98, 6.38, 7.95),
                             cc_file="dt.cc_0.7_combined_no_main", event_file="event.sel"),
        "kim2011": HypoDDInp(idat=3, ipha=3, dist=500, obscc=4, istart=1, isolv=1,
                             iter_sets=_DTCC_ITERS, nlay=4, ratio=1.73,
                             top=(0.0, 7.29, 20.7, 31.3), vel=(5.63, 6.17, 6.58, 7.77),
                             cc_file="dt.cc_0.7_combined", event_file="event.dat"),
    },

    # cross-correlation (dt.cc) — the busy event-pair window override is captured here.
    xcorr=dict(interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5,
               margin=0.5, cc_threshold=0.7, p_comp="Z", s_comps=("N", "E")),
    xcorr_pair_overrides={
        frozenset({"20210827220322"}): dict(pre=0.05, post=0.05, bandpass=(1, 40)),
    },
    # event_id is UTC; cuspid 200002 in the baseline event.dat = "20210827 22032316"
    # = 2021-08-28 07:03 KST, the M2.2 mainshock (the no_main variant drops exactly its
    # pairs: 110->90 dt.cc headers). The old value "20210827092315" was the KST string of
    # cuspid 200001 (the M1.6) and never matched any UTC event dir.
    mainshock_event_id="20210827220322",

    cuspid_offset=200000,
    num_cores=10,
)
