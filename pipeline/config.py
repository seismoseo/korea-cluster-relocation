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
)
CLUSTER_NAMES = ("gwangyang", "kimcheon", "jangsung", "gyeongju")


# ------------------------------------------------- picker backends (by model)
# SeisBench PhaseNet weights run via the default backend (picking.load_model ->
# seisbench.models.PhaseNet.from_pretrained). EQNet PhaseNet+ runs via the in-process
# EQNet backend (core/eqnet_backend.py) and is the only picker that emits first-motion
# POLARITY + per-pick AMPLITUDE, which the focal-mechanism (SKHASH) stage needs.
SEISBENCH_MODELS = frozenset({"stead", "original", "instance", "ethz", "scedc", "geofon", "neic"})
EQNET_MODELS = frozenset({"phasenet_plus"})

# ---- external tools (NOT vendored — like the hyp1.40/hypoDD/ph2dt binaries). Override
#      with environment variables so a clone on another machine stays portable. ----
# EQNet (AI4EPS) clone providing PhaseNet+ (needed only when picker_weights="phasenet_plus").
EQNET_DIR = os.environ.get("EQNET_DIR", "/home/msseo/works/14.EQNet/EQNet")
EQNET_WEIGHTS = os.environ.get(
    "EQNET_WEIGHTS", os.path.join(EQNET_DIR, "docs", "model_phasenet_plus", "model_99.pth"))
PNPLUS_MIN_PROB = float(os.environ.get("PNPLUS_MIN_PROB", "0.3"))   # EQNet default pick threshold
PNPLUS_HIGHPASS = float(os.environ.get("PNPLUS_HIGHPASS", "0.0"))   # Hz; 0 = raw (PhaseNet+ wants raw)
PNPLUS_NT = int(os.environ.get("PNPLUS_NT", str(1024 * 36)))        # samples per inference patch
# SKHASH focal-mechanism tool (needed only for the focal_mechanism stage).
SKHASH_DIR = os.environ.get("SKHASH_DIR", "/home/msseo/works/44.SKHASH/SKHASH/SKHASH")


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
    phs_dist_weight_bins: tuple = (
        (20, 0, 1), (50, 1, 2), (70, 2, 3), (100, 3, 3), (1e9, 4, 3),
    )

    # ph2dt + hypoDD
    ph2dt: Ph2dtParams = field(default_factory=Ph2dtParams)
    hypodd_dtct: Optional[HypoDDInp] = None
    hypodd_dtcc_variants: dict = field(default_factory=dict)

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
