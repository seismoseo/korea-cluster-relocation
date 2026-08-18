"""
Shared configuration for the earthquake-cycle cluster relocation framework.

A cluster is *data, not logic*: each cluster is one `ClusterConfig` instance
(see `clusters/<name>.py`). Stage code in `core/` and the CLIs in `cli/` take a
`ClusterConfig` and read every path / parameter from it.

Two hard rules live here:
  * all framework outputs go under `cfg.output_root` (= pipeline/runs/<cluster>/),
    never into the read-only cluster directory `cfg.src_root`;
  * `assert_writable(path)` refuses any path that resolves inside a cluster's
    committed baseline tree, so the existing .sum/.reloc stay frozen for regression.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------- roots
# Derived from this file's location so a clone works at ANY path: config.py lives at
# <PROJECT_ROOT>/pipeline/config.py, so the project root is two levels up. The only
# invariant is that `pipeline/` stays exactly one level under the project root, with the
# per-cluster data dirs (config.CLUSTER_SRC_DIRS) as its siblings.
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))   # .../<root>/pipeline
PROJECT_ROOT = os.path.dirname(PIPELINE_ROOT)                # .../<root>
RUNS_ROOT = os.path.join(PIPELINE_ROOT, "runs")

# Source (read-only) cluster directories. Used by the non-destructive guard.
# Kept independent of the cluster REGISTRY to avoid an import cycle.
CLUSTER_SRC_DIRS = (
    "Gwangyang_sequence",
    "Kimcheon_cluster",
    "Jangsung_cluster",
    "201704_Gyeongju_swarm",
    "Changnyeong_cluster",
    "Chungju_cluster",
    "Sangju_cluster",
    "Yesan_cluster",
    "West_gyeongju_cluster",
    "2019_changnyeong_cluster",
    "Haman_cluster",
    "Uiseong_cluster",
    "Taean_cluster",
    "Hampyeong_cluster",
    "Buyeo_cluster",
    "Yeoncheon_cluster",
    "Youngcheon_cluster",
    "Gwangju_cluster",
    "Donghae_cluster",
    "Yeoju_cluster",
    "Jecheon_cluster",
    "Yeongyang_cluster",
    "Kimcheon_updated_cluster",
    "2019_suncheon_cluster",
    "Gokseong_cluster",
    "Gurye_cluster",
    "Wanju_cluster",
    "Jangsu_cluster",
    "Namwon_cluster",
    "Suncheon_cluster",
    "Jeongseon_cluster",
    "Jeongseon_ff_cluster",
    "Geochang_cluster",
    "Gwangyang_stp_cluster",
    "Tongyeong_cluster",
    "Goseong_cluster",
    "Hongcheon_cluster",
    "Yeongyang23_cluster",
    "West_jeju_cluster",
    "Gunwi_cluster",
    "Muan_cluster",
    "Jangsu_deep_cluster",
    "East_gyeongju_cluster",
    "f738_reuse_cluster",
    "f738_fresh_cluster",
    "f1218_reuse_cluster",
    "f555_reuse_cluster",
    "f1115_reuse_cluster",
    "f1215_reuse_cluster",
    "f1058_reuse_cluster",
    "f969_reuse_cluster",
    "f1324_reuse_cluster",
    "f1086_reuse_cluster",
    "f1057_reuse_cluster",
    "f1194_reuse_cluster",
    "f426_reuse_cluster",
    "f939_reuse_cluster",
    "f1022_reuse_cluster",
    "f1180_reuse_cluster",
    "f856_reuse_cluster",
    "f1024_reuse_cluster",
    "f1185_reuse_cluster",
    "f1203_reuse_cluster",
    "f936_reuse_cluster",
    "f776_reuse_cluster",
    "f1171_reuse_cluster",
    "f929_reuse_cluster",
    "f953_reuse_cluster",
    "f1437_reuse_cluster",
    "f1255_reuse_cluster",
    "f1038_reuse_cluster",
    "f613_reuse_cluster",
    "f583_reuse_cluster",
    "f886_reuse_cluster",
    "f1145_reuse_cluster",
    "f1040_reuse_cluster",
    "f690_reuse_cluster",
    "f1098_reuse_cluster",
    "f1327_reuse_cluster",
    "f1208_reuse_cluster",
    "f1112_reuse_cluster",
    "f1376_reuse_cluster",
    "f412_reuse_cluster",
    "f1079_reuse_cluster",
    "f809_reuse_cluster",
    "f125_reuse_cluster",
    "f711_reuse_cluster",
    "f803_reuse_cluster",
    "f732_reuse_cluster",
    "f785_reuse_cluster",
    "f321_reuse_cluster",
    "f251_reuse_cluster",
    "f594_reuse_cluster",
    "f566_reuse_cluster",
    "f515_reuse_cluster",
    "f71_reuse_cluster",
    "f257_reuse_cluster",
    "f306_reuse_cluster",
    "f1178_reuse_cluster",
    "f1174_reuse_cluster",
    "f1334_reuse_cluster",
    "f1121_reuse_cluster",
    "f1107_reuse_cluster",
    "f1142_reuse_cluster",
    "f1217_reuse_cluster",
    "f1201_reuse_cluster",
    "f804_reuse_cluster",
    "f595_reuse_cluster",
    "f1132_reuse_cluster",
    "f509_reuse_cluster",
    "f539_reuse_cluster",
    "f535_reuse_cluster",
    "f520_reuse_cluster",
    "f263_reuse_cluster",
    "f256_reuse_cluster",
    "f25_reuse_cluster",
    "f59_reuse_cluster",
    "f91_reuse_cluster",
    "f247_reuse_cluster",
    "f184_reuse_cluster",
    "f208_reuse_cluster",
    "f316_reuse_cluster",
    "f368_reuse_cluster",
    "f214_reuse_cluster",
    "f307_reuse_cluster",
    "f389_reuse_cluster",
    "f563_reuse_cluster",
    "f647_reuse_cluster",
    "f414_reuse_cluster",
    "f852_reuse_cluster",
    "f827_reuse_cluster",
    "f813_reuse_cluster",
    "f790_reuse_cluster",
    "f731_reuse_cluster",
    "f694_reuse_cluster",
    "f585_reuse_cluster",
    "f482_reuse_cluster",
    "f909_reuse_cluster",
    "f880_reuse_cluster",
    "f928_reuse_cluster",
    "f1105_reuse_cluster",
    "f1085_reuse_cluster",
    "f1020_reuse_cluster",
    "f1045_reuse_cluster",
    "f1052_reuse_cluster",
    "f1140_reuse_cluster",
    "f1128_reuse_cluster",
    "f1113_reuse_cluster",
    "f1148_reuse_cluster",
    "f1210_reuse_cluster",
    "f1186_reuse_cluster",
    "f1152_reuse_cluster",
    "f1230_reuse_cluster",
    "f1232_reuse_cluster",
    "f1220_reuse_cluster",
    "f1221_reuse_cluster",
    "f1335_reuse_cluster",
    "f1377_reuse_cluster",
    "f1397_reuse_cluster",
    "f1423_reuse_cluster",
    "f1424_reuse_cluster",
    "f280_b515_reuse_cluster",
    "f928_b515_reuse_cluster",
    "f358_b515_reuse_cluster",
    "f1044_b515_reuse_cluster",
    "f828_b515_reuse_cluster",
    "f889_b515_reuse_cluster",
    "f909_b515_reuse_cluster",
    "f1026_b515_reuse_cluster",
    "f817_b515_reuse_cluster",
    "f865_b515_reuse_cluster",
    "f710_b515_reuse_cluster",
    "f734_b515_reuse_cluster",
    "f885_b515_reuse_cluster",
    "f1078_b515_reuse_cluster",
    "f793_b515_reuse_cluster",
    "f924_b515_reuse_cluster",
    "f920_b515_reuse_cluster",
    "f555_b515_reuse_cluster",
    "f647_b515_reuse_cluster",
    "f792_b515_reuse_cluster",
    "f1121_b515_reuse_cluster",
    "f852_b515_reuse_cluster",
    "f927_b515_reuse_cluster",
    "f886_b515_reuse_cluster",
    "f895_b515_reuse_cluster",
    "f1062_b515_reuse_cluster",
    "f1058_b515_reuse_cluster",
    "f487_b515_reuse_cluster",
    "f544_b515_reuse_cluster",
    "f610_b515_reuse_cluster",
    "f717_b515_reuse_cluster",
    "f921_b515_reuse_cluster",
    "f1025_b515_reuse_cluster",
    "f678_b515_reuse_cluster",
    "f712_b515_reuse_cluster",
    "f480_b515_reuse_cluster",
    "f859_b515_reuse_cluster",
    "f249_b515_reuse_cluster",
    "f592_b515_reuse_cluster",
    "f221_b515_reuse_cluster",
    "f38_b515_reuse_cluster",
    "f116_b515_reuse_cluster",
    "f470_b515_reuse_cluster",
    "f1144_b515_reuse_cluster",
    "f1143_b515_reuse_cluster",
    "f934_b515_reuse_cluster",
    "f899_b515_reuse_cluster",
    "f679_b515_reuse_cluster",
    "f887_b515_reuse_cluster",
    "f881_b515_reuse_cluster",
    "f604_b515_reuse_cluster",
    "f753_b515_reuse_cluster",
    "f596_b515_reuse_cluster",
    "f670_b515_reuse_cluster",
    "f857_b515_reuse_cluster",
    "f915_b515_reuse_cluster",
    "f926_b515_reuse_cluster",
    "f474_b515_reuse_cluster",
    "f922_b515_reuse_cluster",
    "f202_b515_reuse_cluster",
    "f327_b515_reuse_cluster",
    "f226_b515_reuse_cluster",
    "f277_b515_reuse_cluster",
    "f222_b515_reuse_cluster",
    "f401_b515_reuse_cluster",
    "f439_b515_reuse_cluster",
    "f176_b515_reuse_cluster",
    "f346_b515_reuse_cluster",
    "f414_b515_reuse_cluster",
    "f110_b515_reuse_cluster",
    "f153_b515_reuse_cluster",
    "f162_b515_reuse_cluster",
    "f40_b515_reuse_cluster",
    "f203_b515_reuse_cluster",
    "f215_b515_reuse_cluster",
    "f570_b515_reuse_cluster",
    "f490_b515_reuse_cluster",
    "f495_b515_reuse_cluster",
    "f539_b515_reuse_cluster",
    "f410_b515_reuse_cluster",
    "f424_b515_reuse_cluster",
    "f418_b515_reuse_cluster",
    "f409_b515_reuse_cluster",
    "f356_b515_reuse_cluster",
    "f345_b515_reuse_cluster",
    "f269_b515_reuse_cluster",
    "f228_b515_reuse_cluster",
    "f378_b515_reuse_cluster",
    "f197_b515_reuse_cluster",
    "f674_b515_reuse_cluster",
    "f677_b515_reuse_cluster",
    "f775_b515_reuse_cluster",
    "f741_b515_reuse_cluster",
    "f840_b515_reuse_cluster",
    "f622_b515_reuse_cluster",
    "f660_b515_reuse_cluster",
    "f629_b515_reuse_cluster",
    "f888_b515_reuse_cluster",
    "f908_b515_reuse_cluster",
    "f911_b515_reuse_cluster",
    "f904_b515_reuse_cluster",
    "f907_b515_reuse_cluster",
    "f954_b515_reuse_cluster",
    "f933_b515_reuse_cluster",
    "f1085_b515_reuse_cluster",
    "f1124_b515_reuse_cluster",
    "f1260_b515_reuse_cluster",
    "uf_subregion_reuse_cluster",
    "2024_kimcheon_cluster",
    "uf_2016_cluster",
    "uf_2016_qc_cluster",
    "Andong_cluster",
    "uf_2016_eqt_cluster",
    "uf_2016_eqt_qc_cluster",
    "uf_2016_stead_cluster",
    "uf_2016_stead_qc_cluster",
    "uf_2016_original_cluster",
    "uf_2016_original_qc_cluster",
    "uf_2010_cluster",
    "uf_2010_qc_cluster",
    "Geoje_cluster",
    "uf_2012_cluster",
    "uf_2012_qc_cluster",
    "uf_2011_cluster",
    "uf_2011_qc_cluster",
    "uf_2013_cluster",
    "uf_2014_cluster",
    "uf_2014_qc_cluster",
    "uf_2015_cluster",
    "uf_2015_qc_cluster",
    "uf_2013_qc_cluster",
    "2026_haenam_cluster",)
CLUSTER_NAMES = ("gwangyang", "kimcheon", "jangsung", "gyeongju", "changnyeong", "chungju", "sangju", "yesan", "west_gyeongju", "2019_changnyeong", "haman", "uiseong", "taean", "hampyeong", "buyeo", "yeoncheon", "youngcheon", "gwangju", "donghae", "yeoju", "jecheon", "yeongyang", "kimcheon_updated", "2019_suncheon", "gokseong", "gurye", "wanju", "jangsu", "namwon", "suncheon", "jeongseon", "jeongseon_ff", "geochang", "gwangyang_stp", "tongyeong", "goseong", "hongcheon", "yeongyang23", "west_jeju", "gunwi", "muan", "jangsu_deep", "east_gyeongju", "f738_reuse", "f738_fresh", "f1218_reuse", "f555_reuse", "f1115_reuse", "f1215_reuse", "f1058_reuse", "f969_reuse", "f1324_reuse", "f1086_reuse", "f1057_reuse", "f1194_reuse", "f426_reuse", "f939_reuse", "f1022_reuse", "f1180_reuse", "f856_reuse", "f1024_reuse", "f1185_reuse", "f1203_reuse", "f936_reuse", "f776_reuse", "f1171_reuse", "f929_reuse", "f953_reuse", "f1437_reuse", "f1255_reuse", "f1038_reuse", "f613_reuse", "f583_reuse", "f886_reuse", "f1145_reuse", "f1040_reuse", "f690_reuse", "f1098_reuse", "f1327_reuse", "f1208_reuse", "f1112_reuse", "f1376_reuse", "f412_reuse", "f1079_reuse", "f809_reuse", "f125_reuse", "f711_reuse", "f803_reuse", "f732_reuse", "f785_reuse", "f321_reuse", "f251_reuse", "f594_reuse", "f566_reuse", "f515_reuse", "f71_reuse", "f257_reuse", "f306_reuse", "f1178_reuse", "f1174_reuse", "f1334_reuse", "f1121_reuse", "f1107_reuse", "f1142_reuse", "f1217_reuse", "f1201_reuse", "f804_reuse", "f595_reuse", "f1132_reuse", "f509_reuse", "f539_reuse", "f535_reuse", "f520_reuse", "f263_reuse", "f256_reuse", "f25_reuse", "f59_reuse", "f91_reuse", "f247_reuse", "f184_reuse", "f208_reuse", "f316_reuse", "f368_reuse", "f214_reuse", "f307_reuse", "f389_reuse", "f563_reuse", "f647_reuse", "f414_reuse", "f852_reuse", "f827_reuse", "f813_reuse", "f790_reuse", "f731_reuse", "f694_reuse", "f585_reuse", "f482_reuse", "f909_reuse", "f880_reuse", "f928_reuse", "f1105_reuse", "f1085_reuse", "f1020_reuse", "f1045_reuse", "f1052_reuse", "f1140_reuse", "f1128_reuse", "f1113_reuse", "f1148_reuse", "f1210_reuse", "f1186_reuse", "f1152_reuse", "f1230_reuse", "f1232_reuse", "f1220_reuse", "f1221_reuse", "f1335_reuse", "f1377_reuse", "f1397_reuse", "f1423_reuse", "f1424_reuse", "f280_b515_reuse", "f928_b515_reuse", "f358_b515_reuse", "f1044_b515_reuse", "f828_b515_reuse", "f889_b515_reuse", "f909_b515_reuse", "f1026_b515_reuse", "f817_b515_reuse", "f865_b515_reuse", "f710_b515_reuse", "f734_b515_reuse", "f885_b515_reuse", "f1078_b515_reuse", "f793_b515_reuse", "f924_b515_reuse", "f920_b515_reuse", "f555_b515_reuse", "f647_b515_reuse", "f792_b515_reuse", "f1121_b515_reuse", "f852_b515_reuse", "f927_b515_reuse", "f886_b515_reuse", "f895_b515_reuse", "f1062_b515_reuse", "f1058_b515_reuse", "f487_b515_reuse", "f544_b515_reuse", "f610_b515_reuse", "f717_b515_reuse", "f921_b515_reuse", "f1025_b515_reuse", "f678_b515_reuse", "f712_b515_reuse", "f480_b515_reuse", "f859_b515_reuse", "f249_b515_reuse", "f592_b515_reuse", "f221_b515_reuse", "f38_b515_reuse", "f116_b515_reuse", "f470_b515_reuse", "f1144_b515_reuse", "f1143_b515_reuse", "f934_b515_reuse", "f899_b515_reuse", "f679_b515_reuse", "f887_b515_reuse", "f881_b515_reuse", "f604_b515_reuse", "f753_b515_reuse", "f596_b515_reuse", "f670_b515_reuse", "f857_b515_reuse", "f915_b515_reuse", "f926_b515_reuse", "f474_b515_reuse", "f922_b515_reuse", "f202_b515_reuse", "f327_b515_reuse", "f226_b515_reuse", "f277_b515_reuse", "f222_b515_reuse", "f401_b515_reuse", "f439_b515_reuse", "f176_b515_reuse", "f346_b515_reuse", "f414_b515_reuse", "f110_b515_reuse", "f153_b515_reuse", "f162_b515_reuse", "f40_b515_reuse", "f203_b515_reuse", "f215_b515_reuse", "f570_b515_reuse", "f490_b515_reuse", "f495_b515_reuse", "f539_b515_reuse", "f410_b515_reuse", "f424_b515_reuse", "f418_b515_reuse", "f409_b515_reuse", "f356_b515_reuse", "f345_b515_reuse", "f269_b515_reuse", "f228_b515_reuse", "f378_b515_reuse", "f197_b515_reuse", "f674_b515_reuse", "f677_b515_reuse", "f775_b515_reuse", "f741_b515_reuse", "f840_b515_reuse", "f622_b515_reuse", "f660_b515_reuse", "f629_b515_reuse", "f888_b515_reuse", "f908_b515_reuse", "f911_b515_reuse", "f904_b515_reuse", "f907_b515_reuse", "f954_b515_reuse", "f933_b515_reuse", "f1085_b515_reuse", "f1124_b515_reuse", "f1260_b515_reuse", "uf_subregion_reuse", "2024_kimcheon", "uf_2016", "uf_2016_qc", "andong", "uf_2016_eqt", "uf_2016_eqt_qc", "uf_2016_stead", "uf_2016_stead_qc", "uf_2016_original", "uf_2016_original_qc", "uf_2010", "uf_2010_qc", "geoje", "uf_2012", "uf_2012_qc", "uf_2011", "uf_2011_qc", "uf_2013", "uf_2014", "uf_2014_qc", "uf_2015", "uf_2015_qc", "uf_2013_qc", "2026_Haenam")


# ------------------------------------------------- picker backends (by model)
# SeisBench PhaseNet weights run via the default backend (picking.load_model ->
# seisbench.models.PhaseNet.from_pretrained). EQNet PhaseNet+ runs via the in-process
# EQNet backend (core/eqnet_backend.py) and is the only picker that emits first-motion
# POLARITY + per-pick AMPLITUDE, which the focal-mechanism (SKHASH) stage needs.
SEISBENCH_MODELS = frozenset({"stead", "original", "instance", "ethz", "scedc", "geofon", "neic"})
EQNET_MODELS = frozenset({"phasenet_plus"})

# ---- external tools (NOT vendored — like the hyp1.40/hypoDD/ph2dt binaries). Set via
#      environment variables. Unset = feature disabled; call sites raise a clear error
#      when the feature is actually used, so unrelated workflows are unaffected. ----
# EQNet (AI4EPS) clone providing PhaseNet+ (needed only when picker_weights="phasenet_plus").
EQNET_DIR = os.environ.get("EQNET_DIR")
EQNET_WEIGHTS = os.environ.get("EQNET_WEIGHTS") or (
    os.path.join(EQNET_DIR, "docs", "model_phasenet_plus", "model_99.pth") if EQNET_DIR else None)
PNPLUS_MIN_PROB = float(os.environ.get("PNPLUS_MIN_PROB", "0.3"))   # EQNet default pick threshold
PNPLUS_HIGHPASS = float(os.environ.get("PNPLUS_HIGHPASS", "0.0"))   # Hz; 0 = raw (PhaseNet+ wants raw)
PNPLUS_NT = int(os.environ.get("PNPLUS_NT", str(1024 * 36)))        # samples per inference patch
# SKHASH focal-mechanism tool (needed only for the focal_mechanism stage).
SKHASH_DIR = os.environ.get("SKHASH_DIR")
# Optional Helvetica fonts for plot text (viz.py registers them if present; falls back to the
# matplotlib default sans-serif otherwise, so a public clone never breaks).
HELVETICA_DIR = os.environ.get("HELVETICA_DIR")


# ----------------------------------------------------------- parameter blocks
@dataclass(frozen=True)
class VelModel:
    """One HYPOINVERSE / HypoDD crustal model.

    `p_rows`/`s_rows` are ((velocity_km_s, depth_to_top_km), ...) used for the
    HypoDD 1-D model and documentation. If `source_dir` is set, the framework
    SYMLINKS the existing `<name>_p.crh` / `<name>_s.crh` from there into the run
    tree (byte-identical to the baseline) rather than regenerating them.
    """
    name: str
    p_rows: tuple = ()
    s_rows: tuple = ()
    source_dir: Optional[str] = None
    ztr_override: Optional[float] = None


@dataclass(frozen=True)
class HypControl:
    """HYPOINVERSE control parameters (templated into the <Region>.sh heredoc)."""
    CON: int = 50
    MIN: int = 4
    ZTR: tuple = (10, "F")          # trial depth + fix flag
    DIS: tuple = (4, 50, 1, 3)      # distance weighting
    RMS: tuple = (4, 0.12, 2, 4)    # residual weighting
    H71: tuple = (4, 1, 3)          # hypo71 summary format
    KPR: int = 3
    LST: tuple = (2, 0, 1)


@dataclass(frozen=True)
class Ph2dtParams:
    MINWGHT: int = 0
    MAXDIST: int = 200
    MAXSEP: int = 10
    MAXNGH: int = 200
    MINLNK: int = 1
    MINOBS: int = 1
    MAXOBS: int = 500


@dataclass(frozen=True)
class HypoDDInp:
    """A hypoDD.inp configuration (dt.ct baseline or a dt.cc variant)."""
    idat: int = 2                   # 0 synth, 1 cc, 2 catalog, 3 cc+cat
    ipha: int = 3                   # 1 P, 2 S, 3 P&S
    dist: int = 500
    obscc: int = 0
    obsct: int = 0
    istart: int = 1                 # 1 single source, 2 network
    isolv: int = 1                  # 1 SVD, 2 lsqr
    iter_sets: tuple = ()           # rows: (NITER,WTCCP,WTCCS,WRCC,WDCC,WTCTP,WTCTS,WRCT,WDCT,DAMP)
    nlay: int = 3
    ratio: float = 1.73
    top: tuple = (0.0, 15.0, 32.0)
    vel: tuple = (5.98, 6.38, 7.95)
    cc_file: Optional[str] = None   # cross-correlation dt.cc file (None => catalog only)
    event_file: str = "event.dat"   # event.sel to drop excluded events (e.g. mainshock)


# ----------------------------------------------------------------- the config
@dataclass(frozen=True)
class ClusterConfig:
    # identity / paths
    name: str                       # "gwangyang"
    region: str                     # "Gwangyang" -> <Region>.{sh,phs,sum,arc,pha}
    src_root: str                   # existing cluster dir (READ-ONLY inputs + baseline)
    event_catalog_csv: str
    station_master_csvs: tuple      # (KS_station.csv, [KG_station.csv])
    epicenter: tuple                # (lat, lon) center for the radius filter
    output_root: str = ""           # filled in __post_init__ if empty -> runs/<name>
    radius_km: float = 100.0
    region_bounds: Optional[tuple] = None   # (latmin, latmax, lonmin, lonmax) for viz
    kst_offset_hours: int = 9               # origin_utc = catalog_kst - 9h

    # waveform source backend
    wf_source: str = "kma_archive"          # "kma_archive" | "stp_sac"
    kma_archive_glob: dict = field(default_factory=dict)
    # stp_sac: per-event SAC under <stp_sac_root>/<event_id>/<sensor>/, files named
    # <ts>.<net>.<code>.<chan>.sac; globs are relative to the event dir with a {comp}
    # placeholder, e.g. {"HH": "HH/*HH{comp}*.sac", ...}.
    stp_sac_root: Optional[str] = None
    stp_sac_glob: dict = field(default_factory=dict)
    sensor_priority: tuple = ("HH", "HG", "EL")
    target_sampling_hz: float = 100.0

    # AI picking (SeisBench PhaseNet)
    picker_weights: str = "stead"
    p_threshold: float = 0.2
    s_threshold: float = 0.2
    pick_bandpass: dict = field(
        default_factory=lambda: dict(freqmin=1.0, freqmax=40.0, corners=4, zerophase=True)
    )
    sp_max_gap_s: float = 15.0
    pick_window: dict = field(
        default_factory=lambda: dict(evdp=15.0, vp=5.9, vs=3.0)
    )

    # focal mechanisms (SKHASH; only meaningful for a phasenet_plus picker run)
    fm_velmodel: str = "kim1983"             # which .sum/.arc + crustal model feeds SKHASH
    fm_min_polarity_weight: float = 0.3      # drop |phase_polarity| below this (ambiguous first motion)
    fm_min_pick_prob: float = 0.5            # drop polarities from P picks with probability below this
    fm_quality_keep: tuple = ("A", "B")      # SKHASH quality grades kept as "high confidence"
    fm_use_sp_ratio: bool = True             # also feed S/P amplitude ratios (combined inversion)
    fm_sp_min_snr: float = 3.0               # drop S/P ratios where P or S SNR < this (pre-P noise)
    fm_npolmin: int = 8                      # SKHASH minimum number of polarities
    fm_delmax_km: float = 120.0              # SKHASH max source-receiver distance
    # SKHASH SKIP thresholds (relaxed so shallow / one-sided clusters still get a graded solution;
    # NOTE SKHASH still hard-floors the quality GRADE to D when azimuthal_gap>90° or takeoff_gap>60°,
    # so under-covered mechanisms are honestly flagged D). max_pgap is capped at 90 by SKHASH.
    fm_max_agap: float = 180.0               # max azimuthal gap to attempt a solution
    fm_max_pgap: float = 90.0                # max takeoff-angle gap to attempt a solution (<=90)

    # HYPOINVERSE
    hyp_control: HypControl = field(default_factory=HypControl)
    velocity_models: tuple = ()             # tuple[VelModel]
    # COP3 P/S weight code by epicentral distance: ((max_km, Pcode, Scode), ...)
    # P: <20->0, <50->1, <70->2, <100->3, else 4 ; S: <20->1, <50->2, else 3.
    # Used when phs_weight_scheme == "distance".
    phs_dist_weight_bins: tuple = (
        (20, 0, 1), (50, 1, 2), (70, 2, 3), (100, 3, 3), (1e9, 4, 3),
    )
    # Probability-based weighting (v1.0.0). When phs_weight_scheme == "probability", write_phs
    # reads the per-event picks CSV (Station, Phase, Probability) and maps probability -> weight
    # code via phs_prob_weight_bins, instead of using the epicentral-distance bins above. Same
    # mapping for P and S; bins are descending-by-threshold so the first match wins. hyp1.40
    # itself still does its own distance taper internally (DIS command), so the Python-side
    # weight code is now driven by AI pick confidence rather than distance. Source clusters
    # (gwangyang/jangsung/kimcheon/gyeongju) keep "distance" for v0.5.0 baseline byte-identity;
    # PocketQuake-scaffolded clusters default to "probability" (see pocketquake/scaffold.py).
    phs_weight_scheme: str = "distance"     # "distance" | "probability"
    phs_prob_weight_bins: tuple = (         # (prob_threshold_inclusive, weight_code)
        (0.90, 0), (0.70, 1), (0.50, 2), (0.30, 3), (0.00, 4),
    )

    # ph2dt + hypoDD
    ph2dt: Ph2dtParams = field(default_factory=Ph2dtParams)
    hypodd_dtct: Optional[HypoDDInp] = None
    hypodd_dtcc_variants: dict = field(default_factory=dict)

    # Relocation backend selection. Default is the legacy Fortran chain (hyp1.40 +
    # hypoDD + ph2dt). Set to the Python equivalents to opt out of the Fortran
    # dependency entirely. Both backends produce the same on-disk artifacts
    # (`<Region>.sum`, `hypoDD.reloc`) so downstream stages don't know the difference.
    loc_backend: str = "hypoinverse"          # "hypoinverse" | "hyposvi"
    reloc_backend: str = "hypodd"             # "hypodd" | "relocdd_py"
    # dt.cc cross-correlation kernel. Default "cctorch_gpu_batched" = single-process,
    # VRAM-bounded, cross-pair FFT executor (memory-safe, bit-exact vs obspy, ~3-6× faster on
    # GPU). It AUTO-FALLS-BACK to "obspy" (the CPU baseline) when no usable CUDA/torch is found
    # (a GPU smoke-test in run_xcorr), so CPU-only machines and others' runs are unaffected.
    # "cctorch_cpu"/"cctorch_gpu" = the older per-pair batched PyTorch paths.
    xcorr_backend: str = "cctorch_gpu_batched"   # "obspy" | "cctorch_cpu" | "cctorch_gpu" | "cctorch_gpu_batched"
    # Cache interpolated+filtered traces on disk (runs/<cluster>/wf_interp_cache/) so RE-RUNS skip
    # the dominant ~0.3 s/trace 100→1000 Hz interpolation. Deterministic + keyed by source mtime +
    # interp/band params → bit-exact and self-invalidating. Used by the cctorch_gpu_batched path.
    xcorr_interp_cache: bool = True
    # HypoSVI: trained EikoNet checkpoint paths per phase. Required when
    # loc_backend == "hyposvi". A model is per-velocity-model (kim1983 / kim2011);
    # see pipeline/velocity_models/eikonet_kim1983/.
    # If None, the backend auto-discovers bundled weights under
    # pipeline/velocity_models/eikonet_<velmodel>/ (fetch via `python -m
    # pipeline.core.fetch_eikonet`); env HYPOSVI_EIKONET_P/_S override.
    hyposvi_eikonet_p: Optional[str] = None
    hyposvi_eikonet_s: Optional[str] = None
    hyposvi_velmodel: Optional[str] = None    # which velmodel's EikoNet to use (default: 1st velocity model)
    hyposvi_device: str = "auto"              # "auto" (GPU if available, else CPU) | "cpu" | "cuda:N"
    hyposvi_epochs: int = 175                 # SVGD iterations per event batch
    # SVGD particles initialise inside region_bounds (+ margin), NOT the full EikoNet
    # domain — otherwise particles scatter across all-Korea and never converge
    # (symptom: +/-20-30 km "uncertainty", biased depth). Generic across clusters: the
    # box is derived from each cluster's own region_bounds.
    hyposvi_box_margin_deg: float = 0.15      # pad region_bounds before seeding particles
    hyposvi_depth_max_km: float = 25.0        # particle-init depth range is 0..this
    # RBF kernel width (km) for the SVGD repulsion. DEFAULT None = HypoSVI's DYNAMIC bandwidth
    # h=med^2/log n (median heuristic): it SELF-SCALES to the particle cloud, shrinking as the
    # particles collapse, so it converges robustly for ANY depth scale (incl. very shallow
    # mining-induced clusters). Cost: an over-dispersed, ~10-20x inflated reported uncertainty
    # (Smith et al. 2022 sec 4.3) -- but it never fails. A STATIC value (km) restores honest,
    # HYPOINVERSE-level uncertainty (calibration ~15 km for mid-crustal events, notebook 19) BUT a
    # fixed sigma >> source depth over-repels and BREAKS the SVGD for shallow events ("kernel density
    # failure"; verified on the Samcheok ~1 km mining cluster, notebook 20). So keep None by default
    # and set a static sigma ONLY a posteriori for specific DEEP, well-resolved clusters.
    hyposvi_rbf_sigma: Optional[float] = None
    # relocDD-py clone path. Required when reloc_backend == "relocdd_py". Resolved
    # by the backend module from this field or $RELOCDD_PY_DIR.
    relocdd_py_dir: Optional[str] = None

    # cross-correlation (dt.cc)
    xcorr: dict = field(
        default_factory=lambda: dict(
            interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5,
            margin=0.5, cc_threshold=0.7, p_comp="Z", s_comps=("N", "E"),
        )
    )
    xcorr_pair_overrides: dict = field(default_factory=dict)
    mainshock_event_id: Optional[str] = None

    cuspid_offset: int = 200000             # HypoDD cuspid = offset + catalog index
    num_cores: int = 10
    bootstrap_cores: int = 30               # worker pool for the relocation bootstrap (each replica
                                            # subprocess is pinned to 1 BLAS thread, so this is a true
                                            # core count, NOT multiplied by per-replica threads)

    def __post_init__(self):
        if not self.output_root:
            object.__setattr__(self, "output_root", os.path.join(RUNS_ROOT, self.name))


# ---------------------------------------------------- parameter tuning helper
# Plain-dict fields: an override dict is MERGED into the existing dict (override wins),
# so a notebook can bump one key without restating the rest.
_TUNE_DICT_FIELDS = ("pick_window", "pick_bandpass", "xcorr", "xcorr_pair_overrides")
# Frozen nested dataclass fields: an override may be a replacement instance OR a dict of
# field overrides applied via dataclasses.replace on the nested block.
_TUNE_NESTED_FIELDS = ("hyp_control", "ph2dt")


def tune(cfg, **overrides):
    """Return a copy of `cfg` with `overrides` applied; `cfg` itself stays frozen/unchanged.

    For interactive parameter studies in the controlled notebook:
        cfg = config.tune(cfg, p_threshold=0.15)                  # scalar
        cfg = config.tune(cfg, xcorr=dict(slide_step=0.01))       # dict MERGE
        cfg = config.tune(cfg, hyp_control=dict(MIN=6))           # nested replace

    - dict fields (pick_window, pick_bandpass, xcorr, xcorr_pair_overrides) merge the
      override into the existing dict;
    - nested frozen blocks (hyp_control, ph2dt) take a replacement instance or a dict of
      field overrides;
    - all other fields are a straight scalar override.
    `output_root` is already filled on `cfg` and carried forward unchanged by replace
    (so tuned runs still resolve to runs/<name>); pass output_root=... to re-point it.
    `velocity_models` (a tuple) is replaced wholesale, not merged.
    """
    repl = {}
    for k, v in overrides.items():
        if k in _TUNE_DICT_FIELDS:
            merged = dict(getattr(cfg, k))
            merged.update(v)
            repl[k] = merged
        elif k in _TUNE_NESTED_FIELDS and not dataclasses.is_dataclass(v):
            repl[k] = dataclasses.replace(getattr(cfg, k), **v)
        else:
            repl[k] = v
    return dataclasses.replace(cfg, **repl)


# ---------------------------------------------------- output path resolvers
def run_root(cfg):              return cfg.output_root
def station_table_dir(cfg):     return os.path.join(cfg.output_root, "station_table")
def nearby_stations_csv(cfg):   return os.path.join(station_table_dir(cfg), "kma_stations_100km.csv")
def used_stations_csv(cfg):     return os.path.join(station_table_dir(cfg), "used_stations_100km.csv")
def waveforms_dir(cfg):         return os.path.join(cfg.output_root, "waveforms_100km")
def event_wf_dir(cfg, eid):     return os.path.join(waveforms_dir(cfg), eid)
def picks_dir(cfg):             return os.path.join(cfg.output_root, "picks")
def picks_csv(cfg, eid):        return os.path.join(picks_dir(cfg), f"{eid}_picks.csv")

def hyp_dir(cfg):               return os.path.join(cfg.output_root, "1.HypoInv")
def phs_dir(cfg):               return os.path.join(hyp_dir(cfg), "PHS")
def phs_file(cfg):              return os.path.join(phs_dir(cfg), f"{cfg.region}.phs")
def sta_dir(cfg):               return os.path.join(hyp_dir(cfg), "STA")
def sta_file(cfg):              return os.path.join(sta_dir(cfg), f"{cfg.region}.sta")
def sta_hyp_file(cfg):          return os.path.join(sta_dir(cfg), f"{cfg.region}_hyp.sta")
def velmodel_dir(cfg, vm):      return os.path.join(hyp_dir(cfg), vm)
def sum_file(cfg, vm):          return os.path.join(velmodel_dir(cfg, vm), f"{cfg.region}.sum")
def arc_file(cfg, vm):          return os.path.join(velmodel_dir(cfg, vm), f"{cfg.region}.arc")
def prt_file(cfg, vm):          return os.path.join(velmodel_dir(cfg, vm), f"{cfg.region}.prt")

def fm_dir(cfg, vm):            return os.path.join(cfg.output_root, "3.FocalMech", vm)
def fm_in_dir(cfg, vm):         return os.path.join(fm_dir(cfg, vm), "IN")
def fm_out_dir(cfg, vm):        return os.path.join(fm_dir(cfg, vm), "OUT")
def fm_mech_csv(cfg, vm):       return os.path.join(fm_dir(cfg, vm), "mechanisms.csv")

def hypodd_dir(cfg):            return os.path.join(cfg.output_root, "2.HypoDD")
def ph2dt_dir(cfg):             return os.path.join(hypodd_dir(cfg), "00.ph2dt")
def dtct_dir(cfg):              return os.path.join(hypodd_dir(cfg), "01.dt.ct")
def dtcc_dir(cfg):              return os.path.join(hypodd_dir(cfg), "02.dt.cc")

def regression_dir(cfg):        return os.path.join(cfg.output_root, "regression")
def compare_report(cfg, stage): return os.path.join(regression_dir(cfg), f"compare_{stage}.csv")


# ---------------------------------------------------- baseline (read-only) resolvers
def baseline_used_stations(cfg): return os.path.join(cfg.src_root, "station_table", "used_stations_100km.csv")
def baseline_waveforms_dir(cfg): return os.path.join(cfg.src_root, "waveforms_100km")
def baseline_picks_dir(cfg):     return os.path.join(cfg.src_root, "picks")
def baseline_sum(cfg, vm):       return os.path.join(cfg.src_root, "1.HypoInv", vm, f"{cfg.region}.sum")
def baseline_reloc_dtct(cfg):    return os.path.join(cfg.src_root, "2.HypoDD", "01.dt.ct", "hypoDD.reloc")
def baseline_reloc_dtcc(cfg, variant="default"):
    base = os.path.join(cfg.src_root, "2.HypoDD", "02.dt.cc")
    return os.path.join(base, "hypoDD.reloc") if variant == "default" \
        else os.path.join(base, variant, "hypoDD.reloc")


# --------------------------------------------------------------- safety guard
class NonDestructiveError(RuntimeError):
    pass


def assert_writable(path):
    """Refuse to write anywhere inside a read-only cluster baseline tree.

    The framework must only ever write under pipeline/ (its runs/ tree). This
    guard catches a mis-pointed output_root or a stray hardcoded path before it
    can clobber a committed .sum / .reloc baseline.
    """
    rp = os.path.realpath(path)
    for d in CLUSTER_SRC_DIRS:
        root = os.path.realpath(os.path.join(PROJECT_ROOT, d))
        if rp == root or rp.startswith(root + os.sep):
            raise NonDestructiveError(
                f"Refusing to write under read-only cluster baseline:\n  {path}\n"
                f"  (resolves inside {root}). Framework outputs must go under {RUNS_ROOT}."
            )
    return path


# ------------------------------------------------------------- cluster registry
def load_cluster(name) -> ClusterConfig:
    """Import clusters/<name>.py and return its CONFIG."""
    mod = importlib.import_module(f"pipeline.clusters.{name}")
    return mod.CONFIG


def get_registry() -> dict:
    """All cluster configs that currently exist (skips not-yet-written ones)."""
    reg = {}
    for name in CLUSTER_NAMES:
        try:
            reg[name] = load_cluster(name)
        except ModuleNotFoundError:
            continue
    return reg
