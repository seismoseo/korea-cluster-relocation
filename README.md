# korea-cluster-relocation

A unified, parameterized **earthquake-cluster relocation framework**: one codebase
(`pipeline/`) reproduces a per-cluster relocation workflow for *any* cluster that follows
the same directory layout — **a cluster is a config, not a fork of the code**.

```
KMA catalog ─▶ event-window SAC ─▶ AI picking (SeisBench PhaseNet)
            ─▶ HYPOINVERSE absolute location (N velocity models)
            ─▶ re-reference origins ─▶ HypoDD relative relocation (dt.ct catalog + dt.cc cross-correlation)
```

It was built from four Korean clusters (Gwangyang, Kimcheon, Jangsung, Gyeongju — KS/KG
networks, KMA-archive **and** STP-SAC waveform backends) and validated against their frozen
results. **Non-destructive by design:** the framework only ever writes under
`pipeline/runs/<cluster>/`; your cluster directories are read-only inputs (a `config.assert_writable`
guard refuses any write inside them).

> This repository tracks the **framework code + docs + notebooks only**. Waveforms and
> per-cluster baselines stay on your machine — point the configs at your own cluster dirs.

## Install

```bash
git clone https://github.com/seismoseo/korea-cluster-relocation.git
cd korea-cluster-relocation
bash tools/setup-git-filters.sh          # once: strips notebook outputs on commit
export PYTHONPATH=$(pwd)                  # so `import pipeline` works
```
- **Python** (miniforge): `obspy`, `seisbench` (+ `torch`), `pandas`, `numpy`, `matplotlib`.
- **External binaries on `PATH`** (not pip-installable): `hyp1.40` (HYPOINVERSE), `ncsn2pha`,
  `ph2dt`, `hypoDD`.
- `config.PROJECT_ROOT` auto-derives from the repo location (`pipeline/config.py`'s parent), so a
  clone works at any path — the only invariant is that `pipeline/` sits one level under the repo
  root with the cluster dirs as its siblings.

## Directory convention (per cluster)

Each cluster is a sibling directory of `pipeline/` with this layout (only the small reference
files need to exist up front; the rest are produced by the pipeline or supplied by you):

```
<ClusterDir>/                         # e.g. Gwangyang_sequence/
  event_catalog/event_catalog.csv     # KMA catalog: Year,Month,Day,Hour,Minute,Second,Latitude,Longitude,Depth,Magnitude (KST)
  station_table/KS_station.csv        # master station table(s): Network,Code,Latitude,Longitude,Elevation [,Borehole]
  kma_waveforms/<event_id>/...        # KMA-archive backend: per-event raw SAC  (or)
  stp_download/SAC/<event_id>/{HH,HG,EL}/...   # STP-SAC backend: per-event raw SAC in sensor subdirs
  1.HypoInv/<velmodel>/<vm>_p.crh, <vm>_s.crh  # crustal models (HYPOINVERSE .crh)
  1.HypoInv/<Region>.sh                # HYPOINVERSE control block (CON/MIN/ZTR/DIS/RMS/...)
  2.HypoDD/00.ph2dt, 01.dt.ct, 02.dt.cc        # (frozen baselines, if you have them, for regression)
```
Event ids are the **UTC** origin time `YYYYMMDDHHMMSS` (= `catalog_KST − kst_offset_hours`), which is
also the raw per-event directory name. Framework outputs mirror this layout under
`pipeline/runs/<cluster>/`.

## Run

```bash
cd korea-cluster-relocation
PY=python    # or /path/to/miniforge3/bin/python

# catalog (dt.ct) chain for a cluster (PhaseNet on CPU to match the reference run):
$PY -m pipeline.cli.run_pipeline --cluster gwangyang
$PY -m pipeline.cli.run_pipeline --cluster kimcheon --stage-from picking --through hypoinverse

# dt.cc branch (rereference -> xcorr -> dtcc) is HEAVY — PIN CORES on a shared box:
taskset -c 0-9 $PY -m pipeline.cli.run_pipeline --cluster gwangyang --through dtcc --cores 10

# focal mechanisms: pick with PhaseNet+ (emits polarity + S/P amplitude), then SKHASH:
$PY -m pipeline.cli.run_pipeline --cluster gwangyang --picker phasenet_plus --through hypoinverse
$PY -m pipeline.cli.run_pipeline --cluster gwangyang --picker phasenet_plus \
    --stage-from focal_mechanism --through focal_mechanism

# regression vs your frozen baselines:
$PY -m pipeline.cli.compare --cluster gwangyang
```
Stages: `stations waveforms picking hypoinverse ph2dt dtct rereference xcorr dtcc` (+ opt-in
`focal_mechanism`). The default `--through dtct` runs the catalog chain; the dt.cc branch is appended
only when requested.

**Picker / focal mechanisms.** `--picker` (or `cfg.picker_weights`) selects `stead` (default, SeisBench
PhaseNet) or `phasenet_plus` (EQNet PhaseNet+), the latter additionally emitting first-motion polarity +
amplitude. The opt-in `focal_mechanism` stage feeds those into **SKHASH** (double-couple inversion; keeps
quality A/B). EQNet and SKHASH are external tools — set `$EQNET_DIR` / `$SKHASH_DIR` (see `pipeline/config.py`).

## Controlled, stage-by-stage workflow (recommended)

For a fully-controlled run where you **inspect each intermediate product and tune parameters**, open
`pipeline/notebooks/02_controlled_run.ipynb` in JupyterLab, set `CLUSTER`, and run stage by stage.
Each stage is **PARAMS → RUN → INSPECT → PLOT**; tuning uses `config.tune()`, which returns a
modified *copy* of the frozen config (dict fields like `xcorr`/`pick_window` merge; nested blocks
like `hyp_control`/`ph2dt` take field overrides):

```python
cfg = config.tune(cfg0, p_threshold=0.15)               # scalar
cfg = config.tune(cfg,  xcorr=dict(slide_step=0.001))   # merge into the xcorr dict
cfg = config.tune(cfg,  hyp_control=dict(MIN=6))        # nested replace
```

**Re-run discipline:** stage 6 (`rereference`) rewrites SAC origins in place (prerequisite for
dt.cc). If you re-tune **picking** afterwards, re-run the unit *waveforms → picking → hypoinverse →
ph2dt+dtct → rereference* (re-running `waveforms` first resets origins to catalog time). `xcorr`
cross-correlates **all** event pairs — keep iterations cheap with a small `SUBSET` and a coarse
`xcorr.slide_step` (0.01); use `0.001` for the final product, under `taskset`.

Auxiliary QC (station misorientation, ZRT rotation, waveform similarity) is in
`pipeline/notebooks/01_qc.ipynb`; a quick run-all + regression dashboard is `00_run_and_inspect.ipynb`.

## Adding a new cluster

1. Create the cluster directory (above) as a sibling of `pipeline/`.
2. Write `pipeline/clusters/<name>.py`:
   - **KMA-archive** clusters are one call to the factory:
     ```python
     import os
     from pipeline import config
     from pipeline.clusters._base import kma_cluster
     CONFIG = kma_cluster(name="<name>", region="<Region>",
                          src_root=os.path.join(config.PROJECT_ROOT, "<ClusterDir>"),
                          epicenter=(lat, lon), region_bounds=(la0, la1, lo0, lo1),
                          dtct_isolv=1)   # 2 = LSQR for large clusters (SVD MAXDATA0 overflow)
     ```
   - **STP-SAC** clusters (sensor-subdir layout, multi-network) are a bespoke `ClusterConfig`
     with `wf_source="stp_sac"` + `stp_sac_glob` + multiple `station_master_csvs` — see
     `pipeline/clusters/gyeongju.py` as the template.
3. Register the name in `config.CLUSTER_NAMES` and the directory in `config.CLUSTER_SRC_DIRS`.
4. (Optional) freeze regression baselines: `python -m pipeline.regression.freeze_baseline freeze`.

## Regression

`python -m pipeline.cli.compare --cluster <name>` reports PASS/FAIL per stage against frozen
baselines: stations (exact), picks (≥95% within 0.1 s), `.sum` (event count + epicenter/depth/RMS
tolerances), dt.ct `.reloc` (rigid translation **separated** from centroid-aligned relative error —
`shape_corr` is the cluster-geometry fidelity). dt.cc is **report-only** (hand-tuned, judgment-
dependent). `freeze_baseline {freeze,verify}` manages the SHA-256 manifest. Details + the validation
table: [`pipeline/README.md`](pipeline/README.md).

## Layout

```
pipeline/        config.py (+ tune, path resolvers, write guard) · clusters/ · core/ ·
                 analysis/ · regression/ · viz.py · cli/ · notebooks/ · runs/ (gitignored)
tools/           nbstrip.py + setup-git-filters.sh (notebook output stripping)
README.md        this file        CLAUDE.md   guidance for Claude Code
```
See [`pipeline/README.md`](pipeline/README.md) for stage internals, the determinism/validation
table, and findings.
