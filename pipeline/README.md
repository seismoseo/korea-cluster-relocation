# Earthquake-cycle cluster relocation — unified pipeline

One parameterized framework (cluster = config) that reproduces the per-cluster
notebook workflows under `10.Earthquake_cycle_project/`:

```
KMA catalog -> event-window waveforms -> AI picking (PhaseNet) -> HYPOINVERSE
absolute location (N velocity models) -> HypoDD relative relocation (dt.ct catalog; dt.cc xcorr)
```

It mirrors the Ulsan-Fault `pipeline/` design: a frozen `ClusterConfig` + path
resolvers + a non-destructive write guard (`config.py`), stage functions (`core/`),
thin argparse CLIs (`cli/`), a regression harness (`regression/`), and thin
JupyterLab monitoring notebooks (`notebooks/`) that import the *same* functions.

**Non-destructive:** the framework only ever writes under `pipeline/runs/<cluster>/`.
The existing cluster directories are read-only inputs + the frozen regression baseline
(`config.assert_writable()` refuses any path inside them).

## Layout
```
pipeline/
  config.py            ClusterConfig + VelModel/HypControl/Ph2dtParams/HypoDDInp + resolvers + guard
                       + tune(cfg, **overrides) (frozen-config copy for parameter studies)
  clusters/            gwangyang.py kimcheon.py jangsung.py gyeongju.py (+ _base.kma_cluster factory)
                       (cluster SRC paths derive from config.PROJECT_ROOT -> portable clone)
  core/                stations waveforms picking hypoinverse rereference xcorr hypodd sumio pipeline
                       misorientation rotation        (AUX/QC: ZRT inputs)
  analysis/            similarity.py                   (waveform-similarity QC, diagnostic)
  regression/          compare.py (metrics) + freeze_baseline.py (checksum the baselines)
  viz.py               matplotlib plots: map_catalog depth_sections plot_picks cumulative_events
                       plot_3c cc_histogram compare_epicenters
  cli/                 run_pipeline.py  compare.py
  notebooks/           02_controlled_run.ipynb (stage-by-stage + tune)  00_run_and_inspect.ipynb
                       (run-all + regression dashboard)  01_qc.ipynb (misorientation / ZRT / similarity)
  runs/<cluster>/      ALL framework outputs (gitignore-able, regenerable)
../tools/              nbstrip.py + setup-git-filters.sh (notebook output stripping)
```
`config.PROJECT_ROOT`/`PIPELINE_ROOT` derive from `config.py`'s location (`__file__`), so a clone runs
at any path as long as `pipeline/` sits one level under the repo root.

## How to run
```bash
export PYTHONPATH=/home/msseo/works/10.Earthquake_cycle_project
PY=/home/msseo/miniforge3/bin/python

# freeze the baseline checksums once (regression targets):
$PY -m pipeline.regression.freeze_baseline freeze

# catalog (dt.ct) chain for a cluster (PhaseNet on CPU to match the baseline):
$PY -m pipeline.cli.run_pipeline --cluster gwangyang
$PY -m pipeline.cli.run_pipeline --cluster kimcheon --stage-from picking --through hypoinverse

# dt.cc branch (rereference -> xcorr -> dtcc) is HEAVY (cross-correlation slide loop):
# PIN CORES on the shared box. xcorr caps workers at min(--cores, cfg.num_cores).
taskset -c 0-9 $PY -m pipeline.cli.run_pipeline --cluster gwangyang --through dtcc --cores 10
taskset -c 0-9 $PY -m pipeline.cli.run_pipeline --cluster gwangyang \
    --stage-from rereference --through dtcc --dtcc-variant no_main --cores 10

# regression vs the frozen baseline:
$PY -m pipeline.cli.compare --cluster gwangyang
```
Stages: `stations waveforms picking hypoinverse ph2dt dtct rereference xcorr dtcc`. The
default `--through dtct` runs the catalog chain unchanged; `rereference` (rewrite SAC
origins to the kim1983 `.sum`) -> `xcorr` (build `dt.cc`) -> `dtcc` are appended only when
requested. From JupyterLab, open `notebooks/00_run_and_inspect.ipynb` (or `01_qc.ipynb`
for misorientation/ZRT/similarity), set `CLUSTER`, Run-All.

## Controlled, stage-by-stage workflow + `config.tune()`

`notebooks/02_controlled_run.ipynb` runs each stage individually so you can inspect the intermediate
product and tune parameters before advancing (**PARAMS → RUN → INSPECT → PLOT** per stage). Parameter
tuning uses `config.tune(cfg, **overrides)`, which returns a modified **copy** of the frozen config:

```python
cfg = config.tune(cfg0, p_threshold=0.15)              # scalar replace
cfg = config.tune(cfg,  xcorr=dict(slide_step=0.001))  # dict fields MERGE (xcorr/pick_window/pick_bandpass)
cfg = config.tune(cfg,  hyp_control=dict(MIN=6))       # nested frozen blocks: field override (hyp_control/ph2dt)
```
`tune()` never mutates `cfg0`; `output_root` is preserved so tuned runs still land in `runs/<name>`
(pass `output_root=...` to redirect, e.g. an A/B scratch dir).

**Re-run discipline (the one footgun):** the `rereference` stage rewrites the run's `waveforms_100km`
SAC origins (`nz*`/`b`) + picks (`a`/`t0`) **in place** to the `arc_velmodel` `.sum` — a prerequisite
for `xcorr`. It is idempotent for a fixed velmodel, but if you **re-tune picking after** rereferencing,
re-run the unit **waveforms → picking → hypoinverse → ph2dt+dtct → rereference** (re-running
`waveforms` first re-injects the catalog origin, so picking starts from a known reference). `xcorr`
cross-correlates **all** event pairs (no subset arg) — keep iterations cheap with a coarse
`xcorr.slide_step` (0.01) and/or a small upstream event subset; use `slide_step=0.001` (baseline grid)
for the final product, under `taskset`.

## Adding a cluster
Most KMA-archive clusters are one line via the factory:
```python
# clusters/<name>.py
from pipeline.clusters._base import kma_cluster
CONFIG = kma_cluster(name="...", region="...", src_root="...",
                     epicenter=(lat, lon), region_bounds=(la0, la1, lo0, lo1))
```
Then register the name in `config.CLUSTER_NAMES`.

## Validation & determinism
`compare_all()` reports PASS/FAIL per stage against the frozen baseline.

| Stage | Determinism | Metric / tolerance |
|---|---|---|
| stations | deterministic | `(Code,Sensor)` set-equality |
| picks (PhaseNet, CPU) | near-deterministic | shared picks within 0.1 s; ≥95% |
| `.sum` (HYPOINVERSE) | deterministic given phases | event count; epicenter Δ; depth Δ; RMS Δ |
| dt.ct `.reloc` (HypoDD) | relative method | rigid translation **separated** from centroid-aligned relative error |
| dt.cc `.reloc` (HypoDD) | relative method | same separation; report-only (hand-tuned xcorr, judgment-dependent) |
| dt.cc content | judgment-dependent | report-only: matched fraction + median \|Δcc\|, \|Δdelay\| vs baseline |

**Anchor result — Gwangyang (11 events): all stages PASS, now including dt.cc.** Stations
exact; picks 808/808 within 0 s (99.6% of baseline reproduced); `.sum` epicenters within
~0.1 km / depths < 0.7 km; **dt.ct** relocation reproduces the relative cluster structure to
~6 m horizontal / 47 m depth (after a 61 m / 431 m rigid shift); **dt.cc** relocation
reproduces it to **5.1 m horizontal RMS / 9.1 m depth RMS, shape correlation 0.987**
(11/11 events) — tighter than dt.ct, as expected from cross-correlation. The regenerated
`dt.cc` matches the baseline cc almost exactly (median Δcc = 0.0).

## Status (built & validated)
- [x] P0 scaffold, config, `.sum`/`.reloc` readers, 90 baseline files frozen
- [x] P1 stations, P2 waveforms, P3 picking, P4 HYPOINVERSE (N velocity models)
- [x] P5/P6 HypoDD backbone (ph2dt + dt.ct), incl. the **East-longitude ncsn2pha
      sign fix** (a hidden manual step now automated)
- [x] **P7 origin re-reference** (`core.rereference`) — rewrite `waveforms_100km` SAC
      origins/picks to the kim1983 `.sum` (prerequisite for matching the baseline dt.cc)
- [x] **P8 dt.cc cross-correlation** (`core.xcorr` + `hypodd.run_dtcc`): parallel slide-loop
      port (cached traces, polite `taskset`-scoped pool); dt.cc HypoDD variants (default +
      no_main, with the 7-row OBSCC=4 weighting fixed in `_base`/`gwangyang`)
- [x] **Gyeongju** `stp_sac` backend (`core.stations.wf_glob` + `clusters/gyeongju.py`)
- [x] **AUX/QC** ports: `core.misorientation`, `core.rotation` (ZRT), `analysis.similarity`
      + `notebooks/01_qc.ipynb`
- [x] regression harness (now incl. dt.cc reloc + content) + CLIs + monitoring/QC notebooks + viz
- [x] All four clusters run end-to-end: Gwangyang (anchor), Kimcheon (26 ev, LSQR),
      Jangsung (4 ev), Gyeongju (16 ev, STP-SAC)

## Findings
1. **Baselines are not uniformly complete.** Kimcheon's *HYPOINVERSE* `.sum` has 26
   events but its *HypoDD* `.reloc` baseline has only **4** (a partial run). The
   framework links all co-located events (within ph2dt `MAXSEP`=10 km) → a more
   complete, consistent result across clusters.
2. **HypoDD `MAXDATA0` — RESOLVED via LSQR.** Kimcheon's full 26-event dt.ct
   (12 640 diff-times) overflows `hypoDD`'s **SVD** array limit. Switching that cluster
   to **LSQR** (`ISOLV=2`, set per-cluster: `dtct_isolv=2` in `clusters/kimcheon.py`)
   relocates the full set (15 well-linked events) with no overflow. Small clusters keep
   SVD (`ISOLV=1`) to match their baselines. A belt-and-suspenders recompile of `hypoDD`
   with larger `MAXDATA0/MAXEVE` needs the HypoDD Fortran source distribution (not on
   this machine; gfortran+make are available if it's provided).
3. **Closely-spaced events — framework is *more rigorous* than the baseline.** 3
   Kimcheon events on 2020-06-07 (within 7 min) overlap in time. The baseline's picks
   for them are **contaminated by neighbouring-event arrivals** (pick offsets of −90 s
   and +128…+302 s from origin — i.e. the adjacent events; the notebook's *no-window*
   residue). The framework's travel-time windowing correctly rejects these, so it makes
   fewer but clean picks and declines to locate on contaminated data. The other 23/26
   events reproduce to median ~69 m. (A future improvement for genuine overlapping-event
   detection would be continuous-stream processing, not window loosening.)
4. **Gyeongju** (`stp_sac` backend, DONE) reproduces the deterministic stages well —
   stations exact (30/30), picks 386/388, `.sum` within 0.04 km (kim2011) / 0.27 km
   (kim1983), dt.ct shape-corr 0.96. Its **dt.cc** relocation is *report-only*: the cc
   measurements match the baseline exactly (median Δcc = 0.0), but the relocated cluster
   shape is poorly constrained (shape-corr ~0.3). Gyeongju is a tiny M0.6–1.0 swarm where
   the ~0.07 s dt.cc delay offset (from the framework's own slightly-different `.sum`
   origins) is comparable to the true inter-event differential times — i.e. the dt.cc
   relocation is genuinely ill-conditioned for this swarm, not a code error.
5. **dt.cc reproduces the baseline relative relocation where the cluster is
   well-constrained.** Gwangyang 5.1 m / shape 0.987 (11/11), Jangsung shape 0.999 (4/4).
   The cross-correlation port is exact (per-pair cc median Δ = 0.0 vs the baseline files);
   the only systematic difference is a small dt.cc *delay* offset that tracks the framework
   using its own `.sum` origins (vs the baseline's), which the relative relocation absorbs
   for well-spaced clusters but not for the tiniest swarm (Gyeongju, #4). dt.cc is therefore
   **report-only** in the regression (judgment-dependent), gated only on the dt.ct chain.

## Remaining
- [ ] HypoDD recompile with larger `MAXDATA0/MAXEVE` is the only open item, and it is
      **blocked** — the HypoDD Fortran source is not on this machine. It is also unnecessary
      in practice: per-cluster **LSQR** (`dtct_isolv=2`, used by Kimcheon) already avoids the
      SVD `MAXDATA0` overflow (finding #2). If the source is provided, `gfortran`+`make` are
      available and the larger array limits would let the big clusters use SVD too.
