"""Validation harness: compare obspy / cctorch_cpu / cctorch_gpu xcorr backends.

Runs each backend through the FULL `_pair_P` pipeline (which internally batches all
stations into a single CCTorch call for the CCTorch backends), parallelised exactly
like the production `run_xcorr` (`ProcessPoolExecutor` with `--cores` workers; GPU
auto-caps at 1 and uses `spawn` context). This means the wall-clock numbers reflect
realistic deployment overhead, NOT just the inner CC kernel timing.

Acceptance criteria for "safe to use":
  - median |Δshift| ≤ 1 sample (1 ms at the 1 kHz interp rate)
  - 99 th-percentile |ΔCC| ≤ 1e-3

The CPU↔cctorch_cpu comparison isolates pure numerical drift (no CUDA noise);
GPU adds CUBLAS reduction non-determinism on top.

Usage:
    python tools/validate_cctorch_xcorr.py --cluster buyeo --max-pairs 20 --cores 20
    python tools/validate_cctorch_xcorr.py --cluster yeoncheon --max-pairs 100 --cores 20
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from glob import glob
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline import config
from pipeline.core import sumio, xcorr


def _parse_dtcc(path):
    """Yield (e1, e2, sta, diff, cc) for each line in a dt.cc_P_<e1>_<e2> file."""
    with open(path) as f:
        head = f.readline().strip()
        if not head.startswith("#"):
            return
        parts = head.split()
        # Header: "#    <id1>      <id2>       0.0" — but we only need to know it's a header.
        # We'll re-derive the pair from the filename.
        e1, e2 = os.path.basename(path).split("_")[-2:]
        for line in f:
            if line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 4:
                continue
            sta_full = tok[0]
            try:
                diff = float(tok[1]); cc = float(tok[2])
                yield e1, e2, sta_full, diff, cc
            except ValueError:
                continue


def _run_backend(backend, cluster, n_pairs, ncores):
    """Parallel _pair_P over the first n_pairs of `cluster` with the given backend.
    Writes dt.cc to a temp dir, parses, returns (rows, wall) and the temp dir for cleanup."""
    cfg = config.load_cluster(cluster)
    velmodel = cfg.fm_velmodel
    common = config.waveforms_dir(cfg)
    sumdf = sumio.read_sum(config.sum_file(cfg, velmodel))
    dirs = sorted(glob(os.path.join(common, "20*")))
    events, eid = [], {}
    for r in sumdf.itertuples():
        idx = int(r.id) % cfg.cuspid_offset
        if idx < len(dirs):
            e = os.path.basename(dirs[idx])
            events.append(e); eid[e] = int(r.id)
    stations = sorted({os.path.basename(f).split(".")[2]
                       for e in events for f in glob(os.path.join(common, e, "*.sac"))})
    pairs = list(combinations(events, 2))[:n_pairs]
    print(f"  {len(pairs)} pairs × {len(stations)} stations  workers={ncores}", flush=True)

    xc = dict(interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5, margin=0.5,
              cc_threshold=0.7, p_comp="Z", s_comps=("N", "E"), shift_samp=500,
              slide_step=0.001)
    xc.update(cfg.xcorr); xc["bandpass"] = tuple(xc["bandpass"])
    xc["s_comps"] = tuple(xc["s_comps"])

    tmpdir = tempfile.mkdtemp(prefix=f"xcorr_validate_{backend}_")
    out_p = os.path.join(tmpdir, "P"); out_s = os.path.join(tmpdir, "S")
    os.makedirs(out_p, exist_ok=True); os.makedirs(out_s, exist_ok=True)

    # GPU mode needs spawn context to avoid CUDA-in-fork failures.
    mp_ctx = mp.get_context("spawn") if backend == "cctorch_gpu" else None

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=ncores, initializer=xcorr._init_worker,
                             initargs=(common, stations, eid, xc,
                                       dict(cfg.xcorr_pair_overrides),
                                       out_p, out_s, backend),
                             mp_context=mp_ctx) as ex:
        list(ex.map(xcorr._pair_P, pairs))
    wall = time.time() - t0

    rows = []
    for p in glob(os.path.join(out_p, "dt.cc_P_*")):
        for row in _parse_dtcc(p):
            rows.append(row)
    return rows, wall, tmpdir


def _diff_report(rows_baseline, rows_test, label):
    keyed_b = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows_baseline}
    keyed_t = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows_test}
    common = sorted(set(keyed_b) & set(keyed_t))
    if not common:
        print(f"\n=== {label} : NO COMMON OBSERVATIONS ===")
        return False
    dshift = np.array([keyed_t[k][0] - keyed_b[k][0] for k in common])
    dcc    = np.array([keyed_t[k][1] - keyed_b[k][1] for k in common])
    print(f"\n=== {label} (n={len(common)} obs) ===")
    print(f"  |Δshift| median={np.median(np.abs(dshift))*1000:.3f} ms   "
          f"95%={np.percentile(np.abs(dshift), 95)*1000:.3f} ms   "
          f"max={np.max(np.abs(dshift))*1000:.3f} ms")
    print(f"  |ΔCC|    median={np.median(np.abs(dcc)):.2e}   "
          f"99%={np.percentile(np.abs(dcc), 99):.2e}   "
          f"max={np.max(np.abs(dcc)):.2e}")
    median_dshift_ms = float(np.median(np.abs(dshift))) * 1000
    p99_dcc          = float(np.percentile(np.abs(dcc), 99))
    verdict = (median_dshift_ms <= 1.0 and p99_dcc <= 1e-3)
    print(f"  → verdict: {'PASS' if verdict else 'FAIL'} "
          f"(criteria: median |Δshift|≤1ms AND 99% |ΔCC|≤1e-3)")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cluster", default="buyeo")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--cores", type=int, default=20,
                    help="workers for ObsPy + CCTorch-CPU (GPU auto-caps at 1)")
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()

    ncores_avail = len(os.sched_getaffinity(0))
    ncores = max(1, min(a.cores, ncores_avail))

    print(f"validate_cctorch_xcorr: cluster={a.cluster}, max-pairs={a.max_pairs}, "
          f"cores={ncores} (avail={ncores_avail})")
    tmpdirs = []

    print(f"\n--- obspy backend ({ncores} workers) ---", flush=True)
    r_obspy, t_obspy, td = _run_backend("obspy", a.cluster, a.max_pairs, ncores)
    tmpdirs.append(td)
    print(f"  {len(r_obspy)} obs  wall={t_obspy:.1f}s", flush=True)

    print(f"\n--- cctorch_cpu backend ({ncores} workers) ---", flush=True)
    r_cpu, t_cpu, td = _run_backend("cctorch_cpu", a.cluster, a.max_pairs, ncores)
    tmpdirs.append(td)
    print(f"  {len(r_cpu)} obs  wall={t_cpu:.1f}s  "
          f"speedup={t_obspy/max(t_cpu,0.001):.1f}×", flush=True)
    pass_cpu = _diff_report(r_obspy, r_cpu, "obspy ↔ cctorch_cpu")

    pass_gpu = True
    if not a.skip_gpu:
        try:
            import torch
            gpu_avail = torch.cuda.is_available()
        except ImportError:
            gpu_avail = False
        if gpu_avail:
            print(f"\n--- cctorch_gpu backend (1 worker, single-GPU, spawn ctx) ---",
                  flush=True)
            r_gpu, t_gpu, td = _run_backend("cctorch_gpu", a.cluster, a.max_pairs, 1)
            tmpdirs.append(td)
            print(f"  {len(r_gpu)} obs  wall={t_gpu:.1f}s  "
                  f"speedup={t_obspy/max(t_gpu,0.001):.1f}×", flush=True)
            pass_gpu = _diff_report(r_obspy, r_gpu, "obspy ↔ cctorch_gpu")
        else:
            print("\n(no CUDA; skipping cctorch_gpu)")
            pass_gpu = True

    print(f"\n=== SUMMARY ===")
    print(f"  obspy      : baseline (parallel, {ncores} workers)  wall={t_obspy:.1f}s")
    print(f"  cctorch_cpu: {'PASS' if pass_cpu else 'FAIL'}   "
          f"wall={t_cpu:.1f}s   speedup={t_obspy/max(t_cpu,0.001):.1f}×")
    if not a.skip_gpu and gpu_avail:
        print(f"  cctorch_gpu: {'PASS' if pass_gpu else 'FAIL'}   "
              f"wall={t_gpu:.1f}s   speedup={t_obspy/max(t_gpu,0.001):.1f}×")

    for td in tmpdirs:
        shutil.rmtree(td, ignore_errors=True)

    sys.exit(0 if (pass_cpu and pass_gpu) else 1)


if __name__ == "__main__":
    main()
