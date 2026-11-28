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
    # layers (nlay/top/vel) left unset: HypoDD uses the P layers of the --arc-velmodel model
    # (core/hypodd.resolve_layers); RATIO stays the configured 1.73
    return HypoDDInp(idat=2, ipha=3, dist=500, istart=1, isolv=isolv, iter_sets=DTCT_ITERS,
                     ratio=1.73, cc_file=None, event_file="event.dat")


def _dtcc_variants(isolv=1):
    base = dict(idat=3, ipha=3, dist=500, obscc=4, istart=1, isolv=isolv, iter_sets=DTCC_ITERS,
                ratio=1.73)
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


# STP-fetched SAC per-event tree (matches the Gyeongju 2017 hand-assembled cluster). Used by
# PocketQuake's auto-scaffold when --source stp is selected (older events that NECIS no longer
# serves as event-segment downloads but that STP at mara.snu.ac.kr can still pull).
STP_GLOB = {"HH": "HH/*HH{comp}*.sac",
            "HG": "HG/*HG{comp}*.sac",
            "EL": "EL/*EL{comp}*.sac"}


def stp_cluster(name, region, src_root, epicenter, region_bounds,
                station_masters=("KS_station.csv",), mainshock_event_id=None,
                extra_velmodels=(), dtct_isolv=1):
    """Same defaults as kma_cluster but routed through wf_source='stp_sac'. The per-event SAC
    tree is expected at <src_root>/stp_download/SAC/<event_id>/{HH,HG,EL}/<ts>.<NET>.<CODE>.<CHAN>.sac
    -- matching what PocketQuake's stp_bridge produces (and what the 2017 Gyeongju notebook
    produced before PocketQuake existed). All other params identical so per-cluster behavior
    only differs in the fetch backend, not in picking / location / mechanism."""
    hyp = os.path.join(src_root, "1.HypoInv")
    return ClusterConfig(
        name=name, region=region, src_root=src_root,
        event_catalog_csv=os.path.join(src_root, "event_catalog", "event_catalog.csv"),
        station_master_csvs=tuple(os.path.join(src_root, "station_table", m)
                                  for m in station_masters),
        epicenter=epicenter, radius_km=100.0, region_bounds=region_bounds, kst_offset_hours=9,
        wf_source="stp_sac", kma_archive_glob={},
        stp_sac_root=os.path.join(src_root, "stp_download", "SAC"),
        stp_sac_glob=dict(STP_GLOB),
        sensor_priority=("HH", "HG", "EL"), target_sampling_hz=100.0,
        picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
        pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),
        hyp_control=STD_HYP,
        velocity_models=(*_kim_models(hyp), *extra_velmodels),
        ph2dt=STD_PH2DT, hypodd_dtct=_dtct_inp(dtct_isolv),
        hypodd_dtcc_variants=_dtcc_variants(dtct_isolv),
        mainshock_event_id=mainshock_event_id, cuspid_offset=200000, num_cores=10,
    )


def mixed_cluster(name, region, src_root, epicenter, region_bounds,
                  station_masters=("KS_station.csv",), mainshock_event_id=None,
                  extra_velmodels=(), dtct_isolv=1):
    """Cluster whose catalog spans the STP/NECIS transition (events from both backends).

    Layout on disk: BOTH `<src_root>/stp_download/SAC/<event_id>/{HH,HG,EL}/*` AND
    `<src_root>/kma_waveforms/<event_id>/{a,v}/SAC/{HH,HG,EL}/*` can coexist. The PocketQuake
    orchestrator routes each catalog event to one source at download time (try-STP-first with
    NECIS fallback, or strict date cutoff); the eq-cycle gather stage then enumerates
    BOTH trees via `stations._event_dirs(cfg)` and dispatches `parse_sac_name`/`wf_glob`
    per-event based on which layout that event landed in (`stations.event_layout`).

    Other parameters identical to kma_cluster/stp_cluster so the relocation behaviour does
    not depend on the per-event source."""
    hyp = os.path.join(src_root, "1.HypoInv")
    return ClusterConfig(
        name=name, region=region, src_root=src_root,
        event_catalog_csv=os.path.join(src_root, "event_catalog", "event_catalog.csv"),
        station_master_csvs=tuple(os.path.join(src_root, "station_table", m)
                                  for m in station_masters),
        epicenter=epicenter, radius_km=100.0, region_bounds=region_bounds, kst_offset_hours=9,
        wf_source="mixed",
        kma_archive_glob=dict(KMA_GLOB),
        stp_sac_root=os.path.join(src_root, "stp_download", "SAC"),
        stp_sac_glob=dict(STP_GLOB),
        sensor_priority=("HH", "HG", "EL"), target_sampling_hz=100.0,
        picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
        pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),
        hyp_control=STD_HYP,
        velocity_models=(*_kim_models(hyp), *extra_velmodels),
        ph2dt=STD_PH2DT, hypodd_dtct=_dtct_inp(dtct_isolv),
        hypodd_dtcc_variants=_dtcc_variants(dtct_isolv),
        mainshock_event_id=mainshock_event_id, cuspid_offset=200000, num_cores=10,
    )


# ------------------------------------------------------------- clusters anywhere (FDSN)
def layered_velmodel(name, tops_km, vp_km_s, vpvs, hyp_lines=(), hyp_files=(), takeoff_models=()):
    """VelModel from layer tops (km, starting at 0, increasing; the last layer is the half-space),
    their P velocities (km/s) and one Vp/Vs ratio for the S model. Optional: HYPOINVERSE model
    commands + their files (VelModel.hyp_lines/hyp_files) and further takeoff models for SKHASH,
    each ((tops_km), (vp_km_s))."""
    tops = tuple(float(t) for t in tops_km)
    vp = tuple(float(v) for v in vp_km_s)
    if len(tops) != len(vp) or tops[0] != 0.0 or list(tops) != sorted(set(tops)):
        raise ValueError(f"{name}: need equal-length tops/vp with tops starting at 0 and increasing")
    return VelModel(name, p_rows=tuple(zip(vp, tops)),
                    s_rows=tuple((round(v / vpvs, 4), t) for v, t in zip(vp, tops)),
                    hyp_lines=tuple(hyp_lines), hyp_files=tuple(hyp_files),
                    takeoff_rows=tuple(tuple(zip(map(float, v), map(float, t))) for t, v in takeoff_models))


def global_cluster(name, region, src_root, epicenter, region_bounds, velocity_models,
                   sensors=("HH", "EH", "HN"), station_masters=("FDSN_station.csv",),
                   radius_km=100.0, hypodd_vpvs=1.73, mainshock_event_id=None, dtct_isolv=1,
                   magnitude_label="catalog magnitude", xcorr_pair_overrides=None):
    """Cluster built from FDSN data anywhere (PocketQuake-global).

    Same processing as kma_cluster except: the catalog is UTC (ISO `time` column, no KST offset);
    per-event SAC sit in the same `<src_root>/kma_waveforms/<event_id>/<dir>.{v,a}/SAC/<band>/`
    layout for any sensor bands, `.a` for accelerometers (instrument code N or G); one sensor per
    station in the order of `sensors`; the given velocity models (the first locates, relocates and
    feeds the focal mechanisms; HypoDD Vp/Vs = `hypodd_vpvs`); a station code used by two networks
    stops the run instead of warning. `xcorr_pair_overrides` ({frozenset(event ids): window}) gives event pairs
    that include one of the listed events their own cross-correlation window, e.g. the onset window
    dict(pre=0.05, post=0.05, bandpass=(1, 40)) for the larger events."""
    import dataclasses
    glob = {s: f"*.{'a' if s[1:2] in ('N', 'G') else 'v'}/SAC/{s}/*{s}{{comp}}*SAC" for s in sensors}
    models = tuple(velocity_models)
    return ClusterConfig(
        name=name, region=region, src_root=src_root,
        event_catalog_csv=os.path.join(src_root, "event_catalog", "event_catalog.csv"),
        station_master_csvs=tuple(os.path.join(src_root, "station_table", m)
                                  for m in station_masters),
        epicenter=epicenter, radius_km=radius_km, region_bounds=region_bounds, kst_offset_hours=0,
        wf_source="kma_archive", kma_archive_glob=glob,
        sensor_priority=tuple(sensors), sensor_precedence=tuple(sensors), target_sampling_hz=100.0,
        picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
        pick_window=dict(evdp=15.0, vp=5.9, vs=3.0),
        hyp_control=STD_HYP,
        velocity_models=models, default_velmodels=(models[0].name,), fm_velmodel=models[0].name,
        ph2dt=STD_PH2DT,
        hypodd_dtct=dataclasses.replace(_dtct_inp(dtct_isolv), ratio=hypodd_vpvs),
        hypodd_dtcc_variants={k: dataclasses.replace(v, ratio=hypodd_vpvs)
                              for k, v in _dtcc_variants(dtct_isolv).items()},
        station_code_collisions="error", magnitude_label=magnitude_label,
        mainshock_event_id=mainshock_event_id, cuspid_offset=200000, num_cores=10,
        xcorr_pair_overrides=dict(xcorr_pair_overrides or {}),
    )
