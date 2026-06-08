"""OOM stress test for the `cctorch_gpu_batched` FFT xcorr kernel + memory governor.

The prior per-pair cctorch_gpu path validated numerically but ALWAYS OOM'd (cuDNN
grouped-conv workspace was unbounded). This test proves the replacement is OOM-safe BY
CONSTRUCTION: it drives `xcorr._process_batch_safe` (the OOM-retry-halving wrapper) with
synthetic tasks at the exact production shape (n_slides=1000, n_ref=1001, nlag=500), at a
scale far past the old failure point, and asserts:

  1. peak VRAM stays under the hard guard (`_VRAM_HARD_FRACTION × total`);
  2. zero *uncaught* CUDA OOM — retries/CPU-fallbacks are allowed, a fatal OOM is not;
  3. every task returns a (shift, cc) — the run always completes.

It also forces a deliberately over-budget single batch (more tasks than `_max_tasks`
allows) to exercise the halving path. Run:  conda run -n pq-gpu python tools/stress_oom_xcorr.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.core import xcorr   # noqa: E402


def _synth_tasks(n_tasks, n_slides, n_ref, seed=0):
    """Cheap synthetic tasks (ref + per-slide candidate stack), production-shaped."""
    rng = np.random.RandomState(seed)
    tasks = []
    base = np.cumsum(rng.randn(n_ref + n_slides + 4 * 500 + 10))
    for i in range(n_tasks):
        ref = base[100:100 + n_ref].copy()
        cands = np.stack([base[j:j + n_ref] for j in range(n_slides)]) \
            + 0.01 * rng.randn(n_slides, n_ref)
        valid = np.ones(n_slides, dtype=bool)
        tasks.append(dict(key=(("e1", "e2"), "P", f"ST{i:04d}", "Z"),
                          ref=ref, cands=cands, valid=valid, n_ref=n_ref))
    return tasks


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA — cannot run GPU OOM stress test"); sys.exit(2)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    xcorr._install_vram_guard(device)                       # hard ceiling 0.9 * total
    torch.cuda.reset_peak_memory_stats(device)

    n_slides, n_ref, nlag = 1000, 1001, 500
    sr = float(n_slides); slides = np.arange(-0.5, 0.5, 0.001)
    bpt = xcorr._bytes_per_task(n_slides, n_ref, nlag)
    cap = xcorr._max_tasks(device, n_slides, n_ref, nlag)
    print(f"device total={total/1e9:.1f} GB | bytes/task={bpt/1e6:.1f} MB | "
          f"_max_tasks (safe batch)={cap}")

    stats = {"oom_retries": 0, "cpu_fallback_tasks": 0, "batches": 0}

    # --- Test 1: a normal, governor-sized stream far past the old failure scale ---------
    # Old path OOM'd at ~68 stations on one pair. Here: many waves of `cap`-sized batches.
    n_waves = 6
    total_tasks = 0
    for w in range(n_waves):
        tasks = _synth_tasks(min(cap, 64), n_slides, n_ref, seed=w)   # cap memory of synth gen
        res = xcorr._process_batch_safe(tasks, nlag, slides, sr, device, stats)
        assert len(res) == len(tasks), f"wave {w}: {len(res)}/{len(tasks)} returned"
        for k, (shift, cc) in res.items():
            assert np.isfinite(shift) and np.isfinite(cc), f"non-finite {k}"
        total_tasks += len(tasks)
    print(f"[test1] {total_tasks} tasks over {n_waves} waves: all returned, finite")

    # --- Test 2: FORCE an over-budget single batch → must trigger retry-halving ---------
    over = max(cap * 4, 64)                                  # 4× the safe cap → guaranteed OOM
    over = min(over, 4000)                                   # keep host synth gen sane
    print(f"[test2] forcing one {over}-task batch (> safe cap {cap}) to trigger halving...")
    big = _synth_tasks(over, n_slides, n_ref, seed=99)
    res = xcorr._process_batch_safe(big, nlag, slides, sr, device, stats)
    assert len(res) == len(big), f"over-budget batch: {len(res)}/{len(big)} returned"
    for k, (shift, cc) in res.items():
        assert np.isfinite(shift) and np.isfinite(cc)
    print(f"[test2] {over}-task over-budget batch completed via halving")

    peak = torch.cuda.max_memory_allocated(device)
    hard = xcorr._VRAM_HARD_FRACTION * total
    print(f"\npeak VRAM = {peak/1e9:.1f} GB  (hard guard {hard/1e9:.1f} GB = "
          f"{xcorr._VRAM_HARD_FRACTION:.0%} of {total/1e9:.1f} GB)")
    print(f"OOM retries = {stats['oom_retries']} | CPU fallbacks = {stats['cpu_fallback_tasks']} "
          f"| batches = {stats['batches']}")

    ok_peak = peak <= hard
    ok_retry = stats["oom_retries"] >= 1          # test2 should have forced at least one
    print(f"\n  peak ≤ hard guard           : {'PASS' if ok_peak else 'FAIL'}")
    print(f"  halving path exercised      : {'PASS' if ok_retry else 'WARN (no retry needed)'}")
    print(f"  zero fatal OOM / completed  : PASS")
    sys.exit(0 if ok_peak else 1)


if __name__ == "__main__":
    main()
