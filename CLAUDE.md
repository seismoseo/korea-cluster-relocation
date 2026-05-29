# korea-cluster-relocation — guidance for Claude Code

Unified earthquake-cluster relocation framework. **A cluster is data, not logic:** each cluster is one
`ClusterConfig` (`pipeline/clusters/<name>.py`); stage code in `pipeline/core/` + the CLIs read every
path/parameter from it. Full docs: top-level `README.md` + `pipeline/README.md`.

## Hard rules
- **Non-destructive.** The framework writes ONLY under `pipeline/runs/<cluster>/`. The per-cluster data
  dirs are read-only inputs + frozen regression baselines; `config.assert_writable(path)` refuses any
  write inside them. Never edit a baseline.
- **Don't hardcode paths.** `config.PROJECT_ROOT`/`PIPELINE_ROOT` derive from `config.py`'s `__file__`;
  cluster `src_root` derives from `config.PROJECT_ROOT`. Keep it that way (portable clone). The only
  invariant: `pipeline/` is one level under the repo root, cluster dirs as siblings.
- **This repo tracks framework code + docs + notebooks only.** Waveforms (~18 GB) and `pipeline/runs/`
  are gitignored. Notebook outputs are stripped by a git clean filter — run `bash tools/setup-git-filters.sh`
  once per clone *before* the first `git add` of a notebook.

## Pipeline
`stations → waveforms → picking → hypoinverse → ph2dt → dtct → rereference → xcorr → dtcc`,
plus an opt-in tail stage `focal_mechanism`. Default `--through dtct` is the catalog chain; the dt.cc
branch is appended only when requested. PhaseNet runs on CPU to match the reference run. External
binaries on PATH: `hyp1.40`, `ncsn2pha`, `ph2dt`, `hypoDD`.

**Picker option (`cfg.picker_weights`).** Default `stead` (SeisBench PhaseNet). Set `phasenet_plus`
(EQNet PhaseNet+, `core/eqnet_backend.py`) for an alternative picker that additionally emits per-pick
first-motion **Polarity** and **Amplitude** (extra columns in `picks/<eid>_picks.csv`). Run-time toggle:
`config.tune(cfg, picker_weights="phasenet_plus")` or `--picker phasenet_plus`. Use a separate
`output_root` to keep it beside a `stead` run for comparison.

**Focal mechanisms (`focal_mechanism` stage, `core/focal_mechanism.py`).** SKHASH double-couple
inversion from the phasenet_plus polarity + S/P amplitude ratio. Needs a phasenet_plus picking +
hypoinverse run; SKHASH ray-traces takeoff angles from the cluster velocity model and computes azimuths
from the station geometry (no `.arc` parsing). Writes `runs/<cluster>/3.FocalMech/<vm>/{IN,OUT,
mechanisms.csv}` + beachballs; keeps quality **A/B** (`cfg.fm_quality_keep`). Polarity is the robust
signal (vertical first motion); the vertical-component S/P ratio is a secondary enhancement
(`cfg.fm_use_sp_ratio`). **Data-quality gates:** polarities only from P picks with prob ≥
`fm_min_pick_prob` (0.5); S/P ratios only where P & S SNR ≥ `fm_sp_min_snr` (3, vs a pre-P noise window).
**Coverage:** mechanism quality is set by focal-sphere coverage, NOT magnitude — SKHASH skips/floors-to-D
when azimuthal gap > 90° or takeoff gap > 60° (the grade thresholds are hardcoded in SKHASH). `fm_max_agap`
/`fm_max_pgap` (relaxed to 180/90) are the *skip* thresholds so coverage-limited clusters still get a
graded (often D) solution. Validated: **Gwangyang 3/11 events A/B** (deep ~14 km, all-around stations),
consistent ~N-striking strike-slip; **Jangsung/Kimcheon** (shallow ~0.3–6 km → takeoff gap) and
**Gyeongju** (one-sided stations → azimuthal gap) relocate fine but yield only **D** mechanisms.
**Results viewer:** `notebooks/03_results.ipynb` (cluster-parameterized) shows the key figures together —
locations (`viz.map_catalog` .sum + dt.ct/dt.cc reloc, `depth_sections`, `compare_epicenters`), **picks +
first-motion polarity** (`viz.plot_3c` marks the P polarity; `viz.plot_polarities` is a P-aligned record
section sorted by azimuth, red=up/blue=down), and focal mechanisms (`viz.map_mechanisms` beachballs on a
leader-line ring + `viz.mechanism_table` + the SKHASH gallery), and **`viz.fault_sections`** (relocated
catalog rotated into the fault frame — a 2×2 figure: fault-plane map view + along/across-strike depth
sections + a fault-plane along-dip view, coloured by origin time). Its orientation is the **relocation
cloud's own best-fit plane (SVD, `_best_fit_plane`), with the focal mechanism overlaid only for comparison**
(`strike=`/`dip=` override) — so an under-constrained mechanism (e.g. grade-D Jangsung) can't distort the
view; the header prints both, making any disagreement explicit. The phasenet_plus run goes through the full
relocation chain; its HypoDD reloc matches the stead baseline to ≈100 m.
**Located counts** shrink `.sum ≥ dt.ct ≥ dt.cc` (HypoDD keeps only events with enough inter-event links;
dt.ct drops catalog-isolated events, dt.cc further drops poorly-correlating ones). **dt.cc is the high-end
product** (errors of metres) and is the headline relocation; `viz.relocation_counts` tabulates the stages.
Plot text uses **Helvetica** when available (`viz._use_helvetica`, `$HELVETICA_DIR`/`config.HELVETICA_DIR`),
falling back to the matplotlib default otherwise.

**External (NOT vendored, like the binaries; env-overridable in `config.py`):** EQNet clone `$EQNET_DIR`
(+ `$EQNET_WEIGHTS`) for phasenet_plus; SKHASH `$SKHASH_DIR` for focal mechanisms.

## Working notes
- **Parameter studies:** use `config.tune(cfg, **overrides)` (frozen-config copy; dict fields merge,
  nested blocks take field overrides). See `notebooks/02_controlled_run.ipynb`.
- **`rereference` mutates SAC origins in place** (idempotent per velmodel). If picking is re-tuned after
  it, re-run waveforms→picking→hypoinverse→…→rereference as a unit. (Documented in both READMEs.)
- **`xcorr` is heavy** (slide-loop over all event pairs) — run under `taskset -c <cpulist>`, cap with
  `--cores`/`cfg.num_cores`, and coarsen `xcorr.slide_step` for quick looks (0.001 = final grid).
- **dt.cc is report-only** in regression (hand-tuned/judgment-dependent); the hard gates are stations,
  picks, `.sum`, and dt.ct. `shape_corr` = cluster-geometry fidelity (translation/rotation-invariant).
- **Adaptive LSQR dt.cc tuning** (`core/hypodd.py`, only when `isolv==2` — forced when the dt set exceeds
  the SVD MAXDATA0 limit; dt.ct baselines untouched). HypoDD LSQR needs the right damping and inter-event
  distance cutoffs or poorly-linked events destabilise the solution. `run_dtcc` modulates both: (1) the
  WDCC/WDCT **distance cutoffs are scaled to the cluster size** (`_cluster_diameter_km` from the dt.ct
  reloc; `_scale_distance_cutoffs`, ref `DTCC_DIST_REF_KM`=1.5) so a compact cluster cuts long-distance
  pairs and spatially peripheral / zero-cross-correlation events drop out; (2) **DAMP is auto-tuned per
  weighting set so the condition number lands in ~40–80** (`_exec_hypodd` parses per-iteration CND from
  `hypoDD.log`, re-runs until in band; writes `damping_calibration.txt`). E.g. Kimcheon: CND 196→~60, the
  3 zero-CC fliers (one at 8 km depth) removed, dt.cc SVD fault plane 322°→40° (matching its mechanisms).
  **Reproducible & automatic:** the tuning is *code*, run by the `dtcc` stage every time — no manual
  intervention. The search is deterministic (HypoDD + xcorr are deterministic), so a from-scratch re-run
  yields byte-identical `hypoDD.reloc` and the same chosen DAMP. Reproduce end-to-end with
  `run_pipeline --cluster kimcheon --picker phasenet_plus --through dtcc`; `damping_calibration.txt` is the
  audit trail of the per-set DAMP / CND.
- **Bootstrap location uncertainty** (`core/hypodd.bootstrap_relocation`, opt-in). HypoDD's a-posteriori
  ex/ey/ez badly underestimate the true relative-location error (Kimcheon dt.cc ≈0.3/0.3/1 m vs bootstrap
  median ≈4/4/52 m — ~50–300×, depth worst, especially under LSQR). The bootstrap resamples the
  differential-time data **globally** (pool all obs, draw with replacement, regroup into pairs;
  `_resample_global`), re-runs HypoDD `n` (≈1000) times with the **inversion held fixed** (copies the
  calibrated `hypoDD.inp`), and per event takes the **2.5–97.5 percentile half-width** of the X/Y/Z scatter
  (95%; percentile, not σ — robust to the heavy tail of the global resample, where a weakly-linked event
  flies ~km in a few % of replicas). Each replica is **median-aligned** to the main solution (a *mean* offset
  would let one flier hijack the whole replica's alignment). `branch="dtcc"` resamples dt.ct+dt.cc;
  `branch="dtct"` resamples dt.ct. Deterministic (`np.random.default_rng(seed+i)`), parallel (ThreadPool),
  **cached** to `bootstrap_errors.csv` (+ per-replica samples `bootstrap_samples.npz`; the cache header tags
  `align`/`ci` so a method change auto-invalidates). CLI: `python -m pipeline.cli.bootstrap --cluster <name>
  --suffix _pnplus --branch both`. `viz` then draws the 95% bars — recomputed from the samples in each plot's
  frame (rotated into along/across/depth for `fault_sections`; `error_x/y/z` in `plot_3d_plane`); no cache ⇒
  no bars (graceful). Poorly-linked events (low `n_boot`) honestly get large bars.
- **Run things politely** on the shared box (taskset + bounded cores). Verify no baseline drift with
  `python -m pipeline.regression.freeze_baseline verify`.
- Adding a cluster: see README "Adding a cluster" (`_base.kma_cluster` factory or a bespoke `stp_sac`
  config like `clusters/gyeongju.py`), then register in `config.CLUSTER_NAMES`/`CLUSTER_SRC_DIRS`.
