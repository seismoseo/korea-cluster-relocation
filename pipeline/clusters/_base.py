"""
Factory for the standard KMA-archive cluster config (Kimcheon, Jangsung, ...).

These clusters share everything except identity/epicenter: the same KMA SAC archive
layout, sensor priority, PhaseNet picking params, HYPOINVERSE control block
(verified identical across their <Region>.sh), ph2dt/hypoDD params, and the
kim1983/kim2011 crustal models. Gwangyang is written out explicitly because it also
carries the Seo2022 multi-velocity-model set and an xcorr pair override.
"""
import os

from pipeline.config import (
    ClusterConfig, VelModel, HypControl, Ph2dtParams, HypoDDInp,
)

STD_HYP = HypControl(CON=50, MIN=4, ZTR=(10, "F"), DIS=(4, 50, 1, 3),
                     RMS=(4, 0.12, 2, 4), H71=(4, 1, 3), KPR=3, LST=(2, 0, 1))
STD_PH2DT = Ph2dtParams(MINWGHT=0, MAXDIST=200, MAXSEP=10, MAXNGH=200,
                        MINLNK=1, MINOBS=1, MAXOBS=500)
DTCT_ITERS = (
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5, -9, -9, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  8,  6, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  6,  4, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  5,  3, 8),
    (4, 0.01, 0.005, -9, -9, 1.0, 0.5,  3,  1, 8),
)
# dt.cc weighting differs from dt.ct: the baseline 02.dt.cc/hypoDD.inp uses a 7-row
# schedule (NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP) where iterations
# 1-3 are catalog-dominant (WTCTP=1.0, WTCTS=0.8) and 4-7 flip to cc-dominant
# (WTCCP=1.0, WTCCS=0.8, DAMP=6). It also sets OBSCC=4. Reusing DTCT_ITERS here was a
# bug (it kept the cc weights pinned at 0.01/0.005, so the cc data never drove the fit).
DTCC_ITERS = (
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,  -9, -9, 8),
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,   8,  6, 8),
    (4, 0.01, 0.005, -9,  -9,  1.0,  0.8,   6,  4, 8),
    (4, 1.0,  0.8,    6,   4,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    4,   2,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    3,   1,   0.01, 0.005, -9, -9, 6),
    (4, 1.0,  0.8,    2,   0.5, 0.01, 0.005, -9, -9, 6),
)
KMA_GLOB = {
    "HH": "*.v/SAC/HH/*HH{comp}*SAC",
    "HG": "*.a/SAC/HG/*HG{comp}*SAC",
    "EL": "*.v/SAC/EL/*EL{comp}*SAC",
}
KIM1983 = dict(p=((5.98, 0.00), (6.38, 15.0), (7.95, 32.0)),
               s=((3.40, 0.00), (3.79, 15.0), (4.58, 32.0)))
KIM2011 = dict(p=((5.63, 0.00), (6.17, 7.29), (6.58, 20.7), (7.77, 31.3)),
               s=((3.40, 0.00), (3.60, 7.29), (3.70, 20.7), (4.45, 31.3)))


def _kim_models(hyp_dir):
    return (
        VelModel("kim1983", p_rows=KIM1983["p"], s_rows=KIM1983["s"],
                 source_dir=os.path.join(hyp_dir, "kim1983")),
        VelModel("kim2011", p_rows=KIM2011["p"], s_rows=KIM2011["s"],
                 source_dir=os.path.join(hyp_dir, "kim2011")),
    )


def _dtct_inp(isolv=1):
    # isolv: 1 = SVD (small clusters, full error estimates; limited array size),
    #        2 = LSQR (scales to large datasets; use when SVD overflows MAXDATA0).
    return HypoDDInp(idat=2, ipha=3, dist=500, istart=1, isolv=isolv, iter_sets=DTCT_ITERS,
                     nlay=3, ratio=1.73, top=(0.0, 15.0, 32.0), vel=(5.98, 6.38, 7.95),
                     cc_file=None, event_file="event.dat")


def _dtcc_variants(isolv=1):
    base = dict(idat=3, ipha=3, dist=500, obscc=4, istart=1, isolv=isolv, iter_sets=DTCC_ITERS,
                nlay=3, ratio=1.73, top=(0.0, 15.0, 32.0), vel=(5.98, 6.38, 7.95))
    return {
        "default": HypoDDInp(cc_file="dt.cc_0.7_combined", event_file="event.dat", **base),
        "no_main": HypoDDInp(cc_file="dt.cc_0.7_combined_no_main", event_file="event.sel", **base),
    }


def kma_cluster(name, region, src_root, epicenter, region_bounds,
                station_masters=("KS_station.csv",), mainshock_event_id=None,
                extra_velmodels=(), dtct_isolv=1):
    hyp = os.path.join(src_root, "1.HypoInv")
    return ClusterConfig(
        name=name, region=region, src_root=src_root,
        event_catalog_csv=os.path.join(src_root, "event_catalog", "event_catalog.csv"),
        station_master_csvs=tuple(os.path.join(src_root, "station_table", m)
                                  for m in station_masters),
        epicenter=epicenter, radius_km=100.0, region_bounds=region_bounds, kst_offset_hours=9,
        wf_source="kma_archive", kma_archive_glob=dict(KMA_GLOB),
        sensor_priority=("HH", "HG", "EL"), target_sampling_hz=100.0,
        picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
        pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),
        hyp_control=STD_HYP,
        velocity_models=(*_kim_models(hyp), *extra_velmodels),
        ph2dt=STD_PH2DT, hypodd_dtct=_dtct_inp(dtct_isolv),
        hypodd_dtcc_variants=_dtcc_variants(dtct_isolv),
        mainshock_event_id=mainshock_event_id, cuspid_offset=200000, num_cores=10,
    )
