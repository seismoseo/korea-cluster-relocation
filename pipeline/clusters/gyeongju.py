"""
Gyeongju 2017-04 swarm — the STP-SAC backend cluster.

16 events, KS + KG networks. Unlike the other three clusters (flat KMA SAC archive), the
waveforms were downloaded via STP into
  stp_download/SAC/<event_id>/{HH,HG,EL}/<ts>.<net>.<code>.<chan>.sac
so this config sets `wf_source="stp_sac"` + `stp_sac_glob` (the per-event gather/discover
route through `stations.wf_glob` + `parse_sac_name`). Everything else is standard: the
HYPOINVERSE control block is byte-for-byte the others' (verified via 1.HypoInv/Gyeongju.sh:
CON 50 / MIN 4 / ZTR 10 F / DIS 4 50 1 3 / RMS 4 .12 2 4), plus the kim1983/kim2011 models
(symlinked from the baseline) and the shared ph2dt/hypoDD params.

Gyeongju's frozen baseline has only the *default* dt.cc relocation (no `no_main`); the
no_main variant is still defined for symmetry but is not part of regression.
"""
import os

from pipeline import config
from pipeline.config import ClusterConfig
from pipeline.clusters._base import STD_HYP, STD_PH2DT, _kim_models, _dtct_inp, _dtcc_variants

SRC = os.path.join(config.PROJECT_ROOT, "201704_Gyeongju_swarm")
_HYP = os.path.join(SRC, "1.HypoInv")

CONFIG = ClusterConfig(
    name="gyeongju",
    region="Gyeongju",
    src_root=SRC,
    event_catalog_csv=os.path.join(SRC, "event_catalog", "event_catalog.csv"),
    station_master_csvs=(os.path.join(SRC, "station_table", "KS_station.csv"),
                         os.path.join(SRC, "station_table", "KG_station.csv")),
    epicenter=(35.82, 129.40),
    radius_km=100.0,
    region_bounds=(35.4, 36.2, 129.0, 129.8),
    kst_offset_hours=9,

    # waveform source: STP-downloaded SAC, components nested in HH/HG/EL subdirs.
    wf_source="stp_sac",
    kma_archive_glob={},
    stp_sac_root=os.path.join(SRC, "stp_download", "SAC"),
    stp_sac_glob={"HH": "HH/*HH{comp}*.sac",
                  "HG": "HG/*HG{comp}*.sac",
                  "EL": "EL/*EL{comp}*.sac"},
    sensor_priority=("HH", "HG", "EL"),
    target_sampling_hz=100.0,

    # AI picking — same PhaseNet 'stead' setup as the other clusters.
    picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
    pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),

    hyp_control=STD_HYP,
    velocity_models=_kim_models(_HYP),

    ph2dt=STD_PH2DT,
    hypodd_dtct=_dtct_inp(isolv=1),
    hypodd_dtcc_variants=_dtcc_variants(isolv=1),

    mainshock_event_id="20170423125302",     # M1.0 2017-04-23 21:53 KST (no_main only)
    cuspid_offset=200000,
    num_cores=10,
)
