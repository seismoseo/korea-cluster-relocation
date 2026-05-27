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
`stations → waveforms → picking (PhaseNet) → hypoinverse → ph2dt → dtct → rereference → xcorr → dtcc`.
Default `--through dtct` is the catalog chain; the dt.cc branch is appended only when requested.
PhaseNet runs on CPU to match the reference run. External binaries on PATH: `hyp1.40`, `ncsn2pha`,
`ph2dt`, `hypoDD`.

## Working notes
- **Parameter studies:** use `config.tune(cfg, **overrides)` (frozen-config copy; dict fields merge,
  nested blocks take field overrides). See `notebooks/02_controlled_run.ipynb`.
- **`rereference` mutates SAC origins in place** (idempotent per velmodel). If picking is re-tuned after
  it, re-run waveforms→picking→hypoinverse→…→rereference as a unit. (Documented in both READMEs.)
- **`xcorr` is heavy** (slide-loop over all event pairs) — run under `taskset -c <cpulist>`, cap with
  `--cores`/`cfg.num_cores`, and coarsen `xcorr.slide_step` for quick looks (0.001 = final grid).
- **dt.cc is report-only** in regression (hand-tuned/judgment-dependent); the hard gates are stations,
  picks, `.sum`, and dt.ct. `shape_corr` = cluster-geometry fidelity (translation/rotation-invariant).
- **Run things politely** on the shared box (taskset + bounded cores). Verify no baseline drift with
  `python -m pipeline.regression.freeze_baseline verify`.
- Adding a cluster: see README "Adding a cluster" (`_base.kma_cluster` factory or a bespoke `stp_sac`
  config like `clusters/gyeongju.py`), then register in `config.CLUSTER_NAMES`/`CLUSTER_SRC_DIRS`.
