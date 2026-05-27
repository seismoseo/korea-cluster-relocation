"""
Unified earthquake-cycle cluster relocation framework.

One parameterized pipeline (cluster = config) that reproduces the per-cluster
notebook workflows under 10.Earthquake_cycle_project: KMA catalog -> event-window
waveforms -> AI picking (SeisBench PhaseNet) -> HYPOINVERSE absolute location
(N velocity models) -> HypoDD relative relocation (dt.ct catalog + dt.cc xcorr).

Design mirrors the Ulsan pipeline (02.Ulsan_Fault_detection/KS_KG/models/pipeline):
a frozen ClusterConfig + path resolvers + a non-destructive write guard, stage
functions in `core/`, thin argparse CLIs in `cli/`, and a regression harness that
compares fresh outputs against the committed baselines (which are never modified).
"""
__version__ = "0.1.0"
