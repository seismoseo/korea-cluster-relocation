"""
Stage — cross-correlation differential times `dt.cc` for HypoDD (ports
`2.HypoDD/02.dt.cc/01.Detailed_delay_time_measurement_from_Xcorr_revised.ipynb`
+ `02.Get_Pairs_Greater_Than_CC_threshold.ipynb`).

For every event pair and shared station it measures a sub-sample differential time by
cross-correlating the re-referenced waveforms (run `core.rereference` first):

  * P on the Z component, S as the higher-CC of N / E;
  * each trace is interpolated to `interp_hz` (lanczos a=20), demeaned/tapered, bandpass
    filtered, demeaned/tapered again, then sliced `±(pre, post)` s around the SAC pick
    (`a` for P, `t0` for S) and normalised;
  * the second event's window is slid over `±margin` s (step `slide_step`); the lag with
    the highest `obspy.signal.cross_correlation.correlate`/`xcorr_max` coefficient wins;
  * differential time `diff = (t1 + shift - ot1) - (t2 - ot2)` where `t*` are absolute
    picks and `ot*` are the (re-referenced) origins `starttime - b`.

Outputs under `runs/<cluster>/2.HypoDD/02.dt.cc/`: per-pair `dt.cc_{P,S}/dt.cc_{P,S}_<e1>_<e2>`,
the cc>=threshold `dt.cc_P_0.7`/`dt.cc_S_0.7`, their concatenation `dt.cc_0.7_combined`, and
`dt.cc_0.7_combined_no_main` (every pair-block touching the mainshock cuspid dropped).

PERFORMANCE / POLITENESS (shared 64-core box): work is parallelised over pairs with a
`ProcessPoolExecutor` capped at `min(cfg.num_cores, |sched_getaffinity|)` — launch under
`taskset -c <cpulist>` to scope it. Two faithful speedups vs the notebook (identical
numbers): interpolated+filtered full traces are cached per worker, and the slide loop
slices a data *view* instead of deep-copying the full trace 1000x. `slide_step` (default
0.001 s, the baseline grid) is the speed/precision knob.

Cuspid <-> event-dir mapping is by `.sum` `ID-NUM % cuspid_offset` (the scheme
`hypoinverse.write_phs` stamps), so a dropped/unlocated event cannot shift the others.
"""
from __future__ import annotations

import os
from glob import glob
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max

from pipeline import config
from pipeline.core import sumio

UNDEF = -12345.0
_PICK = {"P": "a", "S": "t0"}

# ----------------------------------------------------------- worker-global state
_COMMON = _STATIONS = _EID = _XC = _OVR = _OUTP = _OUTS = None
_BACKEND = "obspy"           # set per-run by run_xcorr → _init_worker
_TORCH_DEVICE = None         # lazy per-worker torch.device handle
_CACHE: dict = {}


def _init_worker(common, stations, eid, xc, ovr, outp, outs, backend="obspy"):
    global _COMMON, _STATIONS, _EID, _XC, _OVR, _OUTP, _OUTS, _BACKEND, _CACHE
    _COMMON, _STATIONS, _EID, _XC, _OVR, _OUTP, _OUTS = \
        common, stations, eid, xc, ovr, outp, outs
    _BACKEND = backend
    _CACHE = {}
    # PyTorch defaults to multi-threaded BLAS — with N ProcessPoolExecutor workers
    # this oversubscribes (N × threads_per_worker), causing massive slowdown. Pin
    # each worker to 1 BLAS thread so wall-time scales linearly with N.
    if backend in ("cctorch_cpu", "cctorch_gpu"):
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass


# ------------------------------------------------------------- window selection
def _full_trace(eid, station, comp, fmin, fmax):
    """Interpolated + filtered full trace for (event, station, comp, band), cached."""
    key = (eid, station, comp, fmin, fmax)
    tr = _CACHE.get(key)
    if tr is None:
        fs = glob(os.path.join(_COMMON, eid, f"{eid}.*.{station}.*{comp}.sac"))
        if not fs:
            raise FileNotFoundError(f"{eid}/{station}/{comp}")
        tr = (read(fs[0])[0]
              .interpolate(sampling_rate=_XC["interp_hz"], method="lanczos", a=20)
              .detrend("demean").taper(0.05)
              .filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
              .detrend("demean").taper(0.05))
        if not tr.stats.network:                # canonical name is {eid}.{net}.{code}.{chan}.sac
            tr.stats.network = os.path.basename(fs[0]).split(".")[1]
        _CACHE[key] = tr
    return tr


def _pick_time(tr, hdr):
    v = tr.stats.sac.get(hdr, UNDEF)
    if v == UNDEF:
        raise ValueError(f"no {hdr} pick")
    return tr.stats.starttime + v - tr.stats.sac.b


def _measure(tr1, tr2, hdr, pre, post, shift_samp, margin, step):
    """Slide tr2's window over +/-margin; return (shift_s, coeff) at best CC.

    Mirrors cc_measurement_revised_*: tr1 sliced+normalised once; tr2 windowed per slide
    (a data view, not a copy); shift = shift_samp/interp_hz - slide of the best CC."""
    p1 = _pick_time(tr1, hdr)
    tr1_slice = tr1.slice(p1 - pre, p1 + post).copy().normalize()
    arr2 = _pick_time(tr2, hdr)
    best_cc, best_shift, best_slide = -2.0, 0.0, 0.0
    for slide in np.arange(-margin, margin, step):
        tr2_slice = tr2.slice(arr2 - pre + slide, arr2 + post + slide)   # view, read-only
        shift, coeff = xcorr_max(correlate(tr1_slice, tr2_slice, shift_samp), abs_max=False)
        if coeff > best_cc:
            best_cc, best_shift, best_slide = coeff, shift, slide
    return np.round(best_shift / _XC["interp_hz"] - best_slide, 3), best_cc


# --------------------- CCTorch backend (batched GPU/CPU xcorr) ------------------------
def _get_torch_device(prefer):
    """Lazily resolve + cache a torch.device for the worker process.
    `prefer` ∈ {"cuda", "cpu"}. Falls back to CPU silently if CUDA unavailable."""
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda:0")
    else:
        _TORCH_DEVICE = torch.device("cpu")
    return _TORCH_DEVICE


def _measure_cctorch(tr1, tr2, hdr, pre, post, shift_samp, margin, step, device):
    """Batched-on-device equivalent of `_measure`.

    Math mirrors `CCTorch.CCModel(domain='time', normalize=True)` time-domain Pearson:
    each candidate window from tr2 (one per slide position) is demeaned + std-normalised,
    then correlated against the (demeaned + normalised) reference tr1 window via batched
    `F.conv1d` with `groups=N_slides`. Single GPU/CPU op replaces the 1000-iteration
    Python slide loop. Result schema matches `_measure`: returns (shift_seconds, best_cc).

    Numerical drift vs ObsPy `correlate(..., demean=True)`: ~1e-6 relative at float64
    on CPU, similar order on GPU (CUBLAS reduction non-determinism). Validate before
    trusting on production runs — see tools/validate_cctorch_xcorr.py."""
    import torch
    import torch.nn.functional as F

    p1 = _pick_time(tr1, hdr)
    arr2 = _pick_time(tr2, hdr)
    sr = float(tr1.stats.sampling_rate)
    n_ref = int(round((pre + post) * sr)) + 1   # match `tr.slice` length

    # Reference window — sliced + demeaned once on CPU (then sent to device).
    tr1_slice = tr1.slice(p1 - pre, p1 + post)
    ref_arr = np.asarray(tr1_slice.data, dtype=np.float64)
    n_ref = len(ref_arr)
    if n_ref < shift_samp + 1:
        return 0.0, -2.0

    # Candidate windows: build one tensor with all slide positions.
    # arr2 - pre + slide → start sample in tr2's array.
    slides = np.arange(-margin, margin, step)
    tr2_start = tr2.stats.starttime
    arr2_offset = float(arr2 - tr2_start)
    start_samples = np.round((arr2_offset - pre + slides) * sr).astype(np.int64)
    n_full = len(tr2.data)
    valid = (start_samples >= 0) & (start_samples + n_ref <= n_full)
    if not valid.any():
        return 0.0, -2.0
    slides = slides[valid]
    start_samples = start_samples[valid]

    tr2_data = np.asarray(tr2.data, dtype=np.float64)
    cand = np.stack([tr2_data[s:s + n_ref] for s in start_samples])    # (N_slides, n_ref)

    # Move to device + demean/normalise (matches CCTorch's normalize=True path).
    ref_t = torch.from_numpy(ref_arr).to(device).double()
    can_t = torch.from_numpy(cand).to(device).double()
    eps = torch.finfo(torch.float64).eps * 10.0
    ref_t = (ref_t - ref_t.mean()) / (ref_t.std(unbiased=False) + eps)
    can_t = (can_t - can_t.mean(dim=-1, keepdim=True)) / \
            (can_t.std(dim=-1, unbiased=False, keepdim=True) + eps)

    # Batched cross-correlation via F.conv1d with groups=N_slides.
    # PyTorch's F.conv1d implements *cross-correlation* natively (no kernel flip),
    # which matches ObsPy `correlate(a, b, shift)` directly — empirically verified on a
    # synthetic +5-sample-shifted pulse: both return shift=-5 at cc=1.0. (An earlier
    # `torch.flip(can_t)` was a bug — that gave us convolution semantics, inverting the
    # lag sign and producing nonsense shifts.)
    N = can_t.shape[0]
    nlag = shift_samp
    # Replicate ref across N groups: shape (1, N, n_ref).
    ref_rep = ref_t.unsqueeze(0).unsqueeze(0).expand(1, N, n_ref).contiguous()
    ref_padded = F.pad(ref_rep, (nlag, nlag), mode="constant", value=0.0)   # (1, N, n_ref+2·nlag)
    weight = can_t.unsqueeze(1)                                              # (N, 1, n_ref)
    # F.conv1d with groups=N → out shape (1, N, 2·nlag + 1).
    xcor = F.conv1d(ref_padded, weight, stride=1, groups=N)
    # Pearson normalisation: std-normalised inputs give c[k] ∈ [-1, 1] after `/ n_ref`.
    xcor = xcor / n_ref

    # Per-slide max + its lag offset (in samples relative to lag 0 at index nlag).
    cc_per_slide, lag_idx = xcor[0].max(dim=-1)                             # (N,), (N,)
    shift_samples = (lag_idx - nlag).double()                               # (N,) in samples
    best_slide_idx = cc_per_slide.argmax().item()
    best_cc = cc_per_slide[best_slide_idx].item()
    best_shift_samples = shift_samples[best_slide_idx].item()
    best_slide_s = float(slides[best_slide_idx])

    # Return in seconds, same convention as `_measure`.
    return np.round(best_shift_samples / sr - best_slide_s, 3), best_cc


def _build_pair_batch(items, hdr, pre, post, shift_samp, margin, step):
    """For a list of (key, tr1, tr2), extract the reference + candidate-stack arrays
    needed for a batched cross-correlation. Returns (keys_kept, refs_array, cands_array,
    valid_masks, slides_array, n_ref) — or None if no station survives.

    `key` is opaque (used by the caller to identify each row). Candidate windows that
    fall off the trace are zero-filled and masked so the GPU max picks them last.
    Skips stations whose n_ref length differs from the first one — extremely rare but
    keeps the batched tensor rectangular."""
    slides = np.arange(-margin, margin, step)
    n_slides = len(slides)
    refs, cands, masks, keys_kept = [], [], [], []
    n_ref_canonical = None
    for key, tr1, tr2 in items:
        try:
            p1 = _pick_time(tr1, hdr)
            arr2 = _pick_time(tr2, hdr)
            sr = float(tr1.stats.sampling_rate)
            tr1_data = np.asarray(tr1.data, dtype=np.float64)
            tr2_data = np.asarray(tr2.data, dtype=np.float64)
            # Reference window — mirror obspy's tr.slice(start, end) length convention.
            tr1_start = tr1.stats.starttime
            i0 = int(round((float(p1 - tr1_start) - pre) * sr))
            i1 = int(round((float(p1 - tr1_start) + post) * sr))    # inclusive end
            if i0 < 0 or i1 >= len(tr1_data):
                continue
            ref = tr1_data[i0:i1 + 1]
            if n_ref_canonical is None:
                n_ref_canonical = len(ref)
            elif len(ref) != n_ref_canonical:
                continue
            # Candidate stack — one window per slide. Out-of-range slots are zero.
            arr2_off = float(arr2 - tr2.stats.starttime)
            start_samples = np.round((arr2_off - pre + slides) * sr).astype(np.int64)
            valid = (start_samples >= 0) & (start_samples + n_ref_canonical <= len(tr2_data))
            cand = np.zeros((n_slides, n_ref_canonical), dtype=np.float64)
            for k, (v, s) in enumerate(zip(valid, start_samples)):
                if v:
                    cand[k] = tr2_data[s:s + n_ref_canonical]
            if not valid.any():
                continue
            refs.append(ref); cands.append(cand); masks.append(valid); keys_kept.append(key)
        except Exception:
            continue
    if not keys_kept:
        return None
    return (keys_kept, np.stack(refs), np.stack(cands), np.stack(masks),
            slides, n_ref_canonical)


# Memory budget per CCTorch call. 2 GB is conservative for the F.conv1d inner allocation
# pattern at the typical (68 stations × 1000 slides) buyeo-scale workload — keeps 20
# workers' total transient under ~40 GB, well within the 244 GB box. Override per-call
# via the `mem_budget_gb` kwarg if you have more headroom.
_DEFAULT_MEM_BUDGET_GB = 2.0


def _measure_pair_cctorch_batch(items, hdr, pre, post, shift_samp, margin, step,
                                 interp_hz, device, mem_budget_gb=_DEFAULT_MEM_BUDGET_GB):
    """Batched-per-pair CCTorch cross-correlation, with **slide-chunked memory cap**.

    Stacks every station's reference + candidate windows for one pair, then runs
    `F.conv1d` in slide-chunks small enough to stay under `mem_budget_gb` per call.
    Tracks per-key best CC across chunks. Math is IDENTICAL to single-batch — only the
    chunk boundary changes — because cross-correlation is independent per slide.

    Why chunked: a naïve all-slides-in-one batch hit a transient peak of ~10 GB per
    worker (F.conv1d allocates input + output + workspace) which compounded across
    20 ProcessPoolExecutor workers to OOM the host. Slide-chunks keep each call at
    ~mem_budget_gb regardless of the slide grid size.

    `items`: list of (key, tr1, tr2). Returns dict {key: (shift_seconds, cc)}."""
    import torch
    import torch.nn.functional as F

    built = _build_pair_batch(items, hdr, pre, post, shift_samp, margin, step)
    if built is None:
        return {}
    keys, refs_np, cands_np, masks_np, slides_np, n_ref = built
    n_keys, n_slides = cands_np.shape[:2]
    nlag = shift_samp
    eps = torch.finfo(torch.float64).eps * 10.0

    # Pick chunk size: peak per chunk ≈ n_keys * chunk * (n_ref + 2*nlag) bytes for the
    # padded input tensor × ~4 (output + workspace + intermediates) at float64.
    bytes_per_slide = n_keys * (n_ref + 2 * nlag) * 8 * 4
    safe_bytes = mem_budget_gb * 1e9
    chunk = max(1, int(safe_bytes / max(bytes_per_slide, 1)))
    chunk = min(chunk, n_slides)
    # Also keep chunk reasonably small for GPU even if math says we have headroom —
    # CUBLAS allocator pattern is more efficient at moderate batch sizes.
    chunk = min(chunk, 256)

    # Refs on device (small), demeaned + std-normalised + padded ONCE.
    refs_t = torch.from_numpy(refs_np).to(device).double()
    refs_t = (refs_t - refs_t.mean(dim=-1, keepdim=True)) / \
             (refs_t.std(dim=-1, unbiased=False, keepdim=True) + eps)
    refs_padded = F.pad(refs_t, (nlag, nlag), value=0.0)          # (n_keys, n_ref+2*nlag)

    # Per-key best running tally across chunks.
    best_cc = torch.full((n_keys,), -2.0, device=device, dtype=torch.float64)
    best_lag_idx = torch.zeros(n_keys, device=device, dtype=torch.long)
    best_slide_full = torch.zeros(n_keys, device=device, dtype=torch.long)

    for c0 in range(0, n_slides, chunk):
        c1 = min(c0 + chunk, n_slides)
        n_c = c1 - c0
        cands_chunk = torch.from_numpy(cands_np[:, c0:c1, :]).to(device).double()
        cands_chunk = (cands_chunk - cands_chunk.mean(dim=-1, keepdim=True)) / \
                      (cands_chunk.std(dim=-1, unbiased=False, keepdim=True) + eps)
        # Build batched conv1d input for this chunk only.
        refs_repl = (refs_padded.unsqueeze(1)
                                .expand(-1, n_c, -1)
                                .reshape(1, n_keys * n_c, -1))
        cands_kernel = cands_chunk.reshape(n_keys * n_c, 1, n_ref)
        xcor = F.conv1d(refs_repl, cands_kernel, groups=n_keys * n_c)
        xcor = xcor.squeeze(0) / n_ref
        xcor = xcor.reshape(n_keys, n_c, 2 * nlag + 1)
        mask_chunk = torch.from_numpy(masks_np[:, c0:c1]).to(device).bool()
        xcor = xcor.masked_fill(~mask_chunk.unsqueeze(-1), float("-inf"))
        # Per-key best WITHIN this chunk.
        flat = xcor.reshape(n_keys, -1)
        chunk_best_flat = flat.argmax(dim=-1)
        chunk_best_cc = flat.gather(1, chunk_best_flat.unsqueeze(-1)).squeeze(-1)
        chunk_best_lag = chunk_best_flat % (2 * nlag + 1)
        chunk_best_slide_local = chunk_best_flat // (2 * nlag + 1)
        # Update overall best where chunk is better.
        upd = chunk_best_cc > best_cc
        best_cc = torch.where(upd, chunk_best_cc, best_cc)
        best_lag_idx = torch.where(upd, chunk_best_lag, best_lag_idx)
        best_slide_full = torch.where(
            upd, chunk_best_slide_local + c0, best_slide_full)
        # Free chunk tensors (Python ref-count) + on GPU drop the allocator's cache.
        del cands_chunk, refs_repl, cands_kernel, xcor, flat, mask_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_shift_samples = (best_lag_idx - nlag).double()
    slides_t = torch.from_numpy(slides_np).to(device).double()
    best_slide_s = slides_t[best_slide_full]
    shift_seconds = best_shift_samples / interp_hz - best_slide_s
    shift_seconds = shift_seconds.cpu().numpy()
    best_cc = best_cc.cpu().numpy()
    return {k: (float(np.round(shift_seconds[i], 3)), float(best_cc[i]))
            for i, k in enumerate(keys)}


def _measure_dispatch(tr1, tr2, hdr, pre, post, shift_samp, margin, step):
    """Single entry point — routes to the active backend (set by _BACKEND)."""
    if _BACKEND == "obspy":
        return _measure(tr1, tr2, hdr, pre, post, shift_samp, margin, step)
    if _BACKEND in ("cctorch_cpu", "cctorch_gpu"):
        device = _get_torch_device("cuda" if _BACKEND == "cctorch_gpu" else "cpu")
        return _measure_cctorch(tr1, tr2, hdr, pre, post, shift_samp, margin, step, device)
    raise ValueError(f"unknown xcorr backend: {_BACKEND!r}")


# ============== cctorch_gpu_batched: memory-safe, cross-pair FFT xcorr ===============
# A single-process executor that gathers EVERY (pair, phase, station[, comp]) task and
# runs them through VRAM-sized batched FFT cross-correlations. The per-slide FFT result
# reproduces obspy `correlate`/`xcorr_max` to ~1e-15 (irfft(rfft(ref)·conj(rfft(cand)))
# /n_ref, lags [-nlag,+nlag]) — so dt.cc stays inside the validation tolerance — while
# peak VRAM is bounded BY CONSTRUCTION: each batch is sized against LIVE free VRAM
# (`mem_get_info`), a hard per-process cap (`set_per_process_memory_fraction`) turns any
# overshoot into a *catchable* CUDA OOM, and the OOM handler halves+retries down to a CPU
# fallback so a run ALWAYS completes. This replaces the per-pair grouped-conv1d path
# whose cuDNN workspace was unbounded (the cause of the prior OOMs). The obspy CPU
# baseline and the existing cctorch_cpu/cctorch_gpu paths are untouched.
_VRAM_SAFE_FRACTION = 0.4    # size each FFT batch to <= this * FREE vram
_VRAM_HARD_FRACTION = 0.9    # set_per_process_memory_fraction guard
# Measured peak ≈ 1.8× the naive tensor sum (cuFFT workspace + transient intermediates), so the
# per-task cost model carries a 2× safety factor; the OOM-retry path covers any residual misfit.
_VRAM_TASK_OVERHEAD = 2.0


def _gpu_free_bytes(device):
    import torch
    return int(torch.cuda.mem_get_info(device)[0])


def _bytes_per_task(n_slides, n_ref, nlag):
    """A-priori float64 GPU bytes for ONE (pair,phase,station) task (all its slides):
    cand windows + rfft(cand) complex + irfft full-real + lag slice + scratch, ×overhead."""
    L = n_ref + 2 * nlag
    return int(_VRAM_TASK_OVERHEAD * n_slides
               * (n_ref * 8 + (L // 2 + 1) * 16 + L * 8 + 4 * (2 * nlag + 1) * 8))


def _max_tasks(device, n_slides, n_ref, nlag, fraction=_VRAM_SAFE_FRACTION):
    return max(1, int(_gpu_free_bytes(device) * fraction
                      / _bytes_per_task(n_slides, n_ref, nlag)))


def _install_vram_guard(device, fraction=_VRAM_HARD_FRACTION):
    import torch
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
    except Exception:                                   # noqa: BLE001 — guard is best-effort
        pass


def _znorm_t(x):
    """z-score along the last axis (mean / std unbiased=False / eps*10) — matches
    `_measure_cctorch`'s normalisation exactly."""
    import torch
    eps = torch.finfo(torch.float64).eps * 10.0
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, unbiased=False, keepdim=True) + eps)


def _build_one_task(e1, e2, sta, comp, hdr, pre, post, fmin, fmax, slides, sr):
    """Reference window + per-slide candidate stack for one task, mirroring
    `_build_pair_batch`'s index math. Returns (ref, cands, valid, n_ref) or None."""
    tr1 = _full_trace(e1, sta, comp, fmin, fmax)
    tr2 = _full_trace(e2, sta, comp, fmin, fmax)
    p1, arr2 = _pick_time(tr1, hdr), _pick_time(tr2, hdr)
    tr1_data = np.asarray(tr1.data, dtype=np.float64)
    tr2_data = np.asarray(tr2.data, dtype=np.float64)
    i0 = int(round((float(p1 - tr1.stats.starttime) - pre) * sr))
    i1 = int(round((float(p1 - tr1.stats.starttime) + post) * sr))
    if i0 < 0 or i1 >= len(tr1_data):
        return None
    ref = tr1_data[i0:i1 + 1]
    n_ref = len(ref)
    arr2_off = float(arr2 - tr2.stats.starttime)
    starts = np.round((arr2_off - pre + slides) * sr).astype(np.int64)
    valid = (starts >= 0) & (starts + n_ref <= len(tr2_data))
    if not valid.any():
        return None
    cands = np.zeros((len(slides), n_ref), dtype=np.float64)
    d = np.diff(starts)
    if len(d) and np.all(d == d[0]) and d[0] != 0:           # uniform stride → as_strided view
        vi = np.where(valid)[0]
        a, b = vi[0], vi[-1] + 1
        sub = np.lib.stride_tricks.as_strided(
            tr2_data[starts[a]:], shape=(b - a, n_ref),
            strides=(d[0] * tr2_data.strides[0], tr2_data.strides[0]))
        cands[a:b] = sub
    else:                                                    # rare: non-uniform → loop (exact)
        for k in np.where(valid)[0]:
            cands[k] = tr2_data[starts[k]:starts[k] + n_ref]
    return ref, cands, valid, n_ref


def _gpu_correlate_batch(refs, cands, valids, n_ref, nlag, slides, sr, device):
    """Batched FFT xcorr for a homogeneous task batch. refs (B,n_ref), cands
    (B,n_slides,n_ref), valids (B,n_slides) numpy. Returns (shift_s (B,), cc (B,)).
    Reference FFT broadcasts over slides (no replication). Bit-exact-equivalent to the
    obspy per-slide correlate/xcorr_max, then max over (slide, lag)."""
    import torch
    L = n_ref + 2 * nlag
    R = torch.from_numpy(refs).to(device)
    C = torch.from_numpy(cands).to(device)
    R = _znorm_t(R)                                          # (B, n_ref)
    C = _znorm_t(C)                                          # (B, n_slides, n_ref)
    A = torch.fft.rfft(R, n=L).unsqueeze(1)                  # (B, 1, Lf)
    Bf = torch.fft.rfft(C, n=L)                              # (B, n_slides, Lf)
    full = torch.fft.irfft(A * torch.conj(Bf), n=L) / n_ref  # (B, n_slides, L)
    cc = torch.cat([full[..., L - nlag:], full[..., :nlag + 1]], dim=-1)  # (B,n_s,2nlag+1)
    vmask = torch.from_numpy(valids).to(device).unsqueeze(-1)
    cc = cc.masked_fill(~vmask, float("-inf"))
    B, n_slides = cc.shape[0], cc.shape[1]
    flat = cc.reshape(B, -1)
    best_cc, idx = flat.max(dim=1)
    lag = (idx % (2 * nlag + 1)) - nlag
    slide_idx = (idx // (2 * nlag + 1))
    slides_t = torch.from_numpy(np.asarray(slides)).to(device)
    shift_s = lag.double() / sr - slides_t[slide_idx]
    out_shift = shift_s.cpu().numpy()
    out_cc = best_cc.cpu().numpy()
    del R, C, A, Bf, full, cc, flat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_shift, out_cc


def _process_batch_safe(batch, nlag, slides, sr, device, stats):
    """Run one task batch on the GPU; on CUDA OOM empty_cache → halve & retry; a single
    task that still OOMs falls back to CPU. Returns {key: (shift_s, cc)}. Never raises OOM."""
    import torch
    if not batch:
        return {}
    refs = np.stack([t["ref"] for t in batch])
    cands = np.stack([t["cands"] for t in batch])
    valids = np.stack([t["valid"] for t in batch])
    n_ref = batch[0]["n_ref"]
    try:
        shifts, ccs = _gpu_correlate_batch(refs, cands, valids, n_ref, nlag, slides, sr, device)
        stats["batches"] += 1
        return {batch[i]["key"]: (float(np.round(shifts[i], 3)), float(ccs[i]))
                for i in range(len(batch))}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:                                 # last resort: CPU (always fits)
            stats["cpu_fallback_tasks"] += 1
            cpu = torch.device("cpu")
            shifts, ccs = _gpu_correlate_batch(refs, cands, valids, n_ref, nlag, slides, sr, cpu)
            return {batch[0]["key"]: (float(np.round(shifts[0], 3)), float(ccs[0]))}
        stats["oom_retries"] += 1
        mid = len(batch) // 2
        out = _process_batch_safe(batch[:mid], nlag, slides, sr, device, stats)
        out.update(_process_batch_safe(batch[mid:], nlag, slides, sr, device, stats))
        return out


def _interp_one_trace(key):
    """Worker entry for the prep pool: interpolate+filter one (eid, sta, comp, fmin, fmax)
    trace and return (key, trace) — or (key, None) if its SAC file is missing. The heavy
    100→1000 Hz lanczos interpolation runs here, in parallel, instead of serially in the
    single GPU process."""
    try:
        return key, _full_trace(*key)
    except Exception:                                       # noqa: BLE001 — missing trace etc.
        return key, None


def run_xcorr_gpu_batched(common, stations, eid, xc, overrides, out_p, out_s, pairs,
                          cores=None, vram_fraction=_VRAM_SAFE_FRACTION):
    """Single-process, memory-bounded, cross-pair-batched FFT xcorr executor. Writes the
    SAME per-pair dt.cc_{P,S}_<e1>_<e2> files as the obspy path (station order, S best-comp
    selection identical), so `_filter_combine`/combine/`_drop_mainshock` are unchanged."""
    import torch
    from collections import defaultdict
    _init_worker(common, stations, eid, xc, dict(overrides), out_p, out_s, "cctorch_gpu_batched")
    device = torch.device("cuda:0")
    _install_vram_guard(device)
    sr = float(xc["interp_hz"]); nlag = int(xc["shift_samp"])
    slides = np.arange(-xc["margin"], xc["margin"], xc["slide_step"])
    n_slides = len(slides)
    stats = {"oom_retries": 0, "cpu_fallback_tasks": 0, "batches": 0, "tasks": 0,
             "peak_vram_gb": 0.0}
    results = {p: {"P": {}, "S": {}} for p in pairs}        # pair -> phase -> {(sta,comp):(shift,cc)}

    # (pre,post,fmin,fmax) per pair (overrides) — window/band drive n_ref.
    win = {p: _window(p) for p in pairs}

    # ---- parallel trace prep (the single-process bottleneck) -------------------------------
    # _full_trace does a ~0.3 s/trace 100→1000 Hz lanczos interpolation; running it serially for
    # every (event,station,comp,band) starves the GPU. Pre-build the trace cache in a CPU worker
    # pool (no GPU → no contention / no OOM), then the task loop below hits a warm `_CACHE`.
    import time as _time
    # Prep is the dominant cost at scale (interpolation, ~0.3 s/trace) and is embarrassingly
    # parallel and GPU-free — so use as many cores as available (capped at 32; IPC of returned
    # traces gives diminishing returns beyond that). `cores` (from --cores) overrides.
    ncores = cores or min(32, max(1, len(os.sched_getaffinity(0)) - 1))
    if ncores > 1:
        keys = set()
        for pair in pairs:
            pre, post, fmin, fmax = win[pair]
            for sta in stations:
                for comp in ("Z",) + tuple(xc["s_comps"]):
                    keys.add((pair[0], sta, comp, fmin, fmax))
                    keys.add((pair[1], sta, comp, fmin, fmax))
        keys = list(keys)
        t0 = _time.time(); got = 0
        with ProcessPoolExecutor(max_workers=ncores, initializer=_init_worker,
                                 initargs=(common, stations, eid, xc, dict(overrides),
                                           out_p, out_s, "obspy")) as ex:
            for key, tr in ex.map(_interp_one_trace, keys, chunksize=4):
                if tr is not None:
                    _CACHE[key] = tr; got += 1
        print(f"[xcorr] gpu_batched: pre-interpolated {got}/{len(keys)} traces on {ncores} "
              f"workers in {_time.time() - t0:.1f}s")

    def _specs():
        for pair in pairs:
            for sta in stations:
                yield (pair, "P", "a", sta, "Z")
                for comp in xc["s_comps"]:
                    yield (pair, "S", "t0", sta, comp)

    torch.cuda.reset_peak_memory_stats(device)
    batch, batch_nref, cap = [], None, None
    for (pair, phase, hdr, sta, comp) in _specs():
        pre, post, fmin, fmax = win[pair]
        try:
            built = _build_one_task(pair[0], pair[1], sta, comp, hdr, pre, post, fmin, fmax, slides, sr)
        except Exception:                                   # noqa: BLE001 — missing trace etc.
            built = None
        if built is None:
            continue
        ref, cands, valid, n_ref = built
        if batch_nref is None:
            batch_nref = n_ref
            cap = _max_tasks(device, n_slides, n_ref, nlag, vram_fraction)
        if n_ref != batch_nref or len(batch) >= cap:        # flush on size/shape change
            for k, v in _process_batch_safe(batch, nlag, slides, sr, device, stats).items():
                results[k[0]][k[1]][(k[2], k[3])] = v
            stats["tasks"] += len(batch)
            batch, batch_nref = [], n_ref
            cap = _max_tasks(device, n_slides, n_ref, nlag, vram_fraction)
        batch.append(dict(key=(pair, phase, sta, comp), ref=ref, cands=cands,
                          valid=valid, n_ref=n_ref))
    for k, v in _process_batch_safe(batch, nlag, slides, sr, device, stats).items():
        results[k[0]][k[1]][(k[2], k[3])] = v
    stats["tasks"] += len(batch)
    stats["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

    # Write per-pair files — byte-identical layout to _pair_P / _pair_S.
    for pair in pairs:
        e1, e2 = pair
        pre, post, fmin, fmax = win[pair]
        # P
        plines = [_header(e1, e2)]
        for sta in stations:
            r = results[pair]["P"].get((sta, "Z"))
            if r is None:
                continue
            try:
                shift, coeff = r
                tr1 = _full_trace(e1, sta, "Z", fmin, fmax)
                tr2 = _full_trace(e2, sta, "Z", fmin, fmax)
                t1, ot1 = _pick_time(tr1, "a"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "a"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                plines.append(_fmt(tr1.stats.network or sta[:0], sta, diff, coeff, "P"))
            except Exception:                               # noqa: BLE001
                pass
        with open(os.path.join(out_p, f"dt.cc_P_{e1}_{e2}"), "w") as f:
            f.writelines(plines)
        # S — best comp per station (s_comps order, >= tie-break: last wins) like _pair_S
        slines = [_header(e1, e2)]
        for sta in stations:
            best = None
            for comp in xc["s_comps"]:
                r = results[pair]["S"].get((sta, comp))
                if r is None:
                    continue
                shift, coeff = r
                if best is None or coeff >= best[0]:
                    best = (coeff, shift, comp)
            if best is None:
                continue
            try:
                coeff, shift, comp = best
                tr1 = _full_trace(e1, sta, comp, fmin, fmax)
                tr2 = _full_trace(e2, sta, comp, fmin, fmax)
                t1, ot1 = _pick_time(tr1, "t0"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "t0"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                slines.append(_fmt(tr1.stats.network or sta[:0], sta, diff, coeff, "S"))
            except Exception:                               # noqa: BLE001
                pass
        with open(os.path.join(out_s, f"dt.cc_S_{e1}_{e2}"), "w") as f:
            f.writelines(slines)

    print(f"[xcorr] gpu_batched: {stats['tasks']} tasks in {stats['batches']} batches | "
          f"peak VRAM {stats['peak_vram_gb']:.1f} GB | OOM retries {stats['oom_retries']} | "
          f"CPU fallbacks {stats['cpu_fallback_tasks']}")
    return stats


def _window(pair):
    """(pre, post, fmin, fmax) for a pair, applying any xcorr_pair_override."""
    pre, post = _XC["pre"], _XC["post"]
    fmin, fmax = _XC["bandpass"]
    s = set(pair)
    for key, ov in _OVR.items():
        if s & set(key):
            pre, post = ov.get("pre", pre), ov.get("post", post)
            fmin, fmax = ov.get("bandpass", (fmin, fmax))
            break
    return pre, post, fmin, fmax


def _fmt(net, station, diff, coeff, phase):
    return (f"{net}{station}".ljust(6) + " " * 5 + str(np.round(diff, 5)).ljust(9)
            + " " * 3 + str(np.round(coeff, 5)).ljust(11) + " " * 4 + phase + "\n")


def _header(e1, e2):
    return f"#    {_EID[e1]}      {_EID[e2]}       0.0\n"


# ------------------------------------------------------------------ pair workers
def _pair_P(pair):
    """Per-pair P-phase. Routes to batched CCTorch when backend != 'obspy'."""
    e1, e2 = pair
    pre, post, fmin, fmax = _window(pair)
    shift_samp, margin, step = _XC["shift_samp"], _XC["margin"], _XC["slide_step"]
    lines = [_header(e1, e2)]

    if _BACKEND in ("cctorch_cpu", "cctorch_gpu"):
        # Batched-per-pair path: one CCTorch call for all stations.
        device = _get_torch_device("cuda" if _BACKEND == "cctorch_gpu" else "cpu")
        items, trace_lookup = [], {}
        for sta in _STATIONS:
            try:
                tr1 = _full_trace(e1, sta, "Z", fmin, fmax)
                tr2 = _full_trace(e2, sta, "Z", fmin, fmax)
                items.append((sta, tr1, tr2))
                trace_lookup[sta] = (tr1, tr2)
            except Exception:
                continue
        results = _measure_pair_cctorch_batch(
            items, "a", pre, post, shift_samp, margin, step,
            _XC["interp_hz"], device)
        for sta in _STATIONS:
            if sta not in results or sta not in trace_lookup:
                continue
            try:
                shift, coeff = results[sta]
                tr1, tr2 = trace_lookup[sta]
                t1, ot1 = _pick_time(tr1, "a"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "a"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                net = tr1.stats.network or sta[:0]
                lines.append(_fmt(net, sta, diff, coeff, "P"))
            except Exception:
                pass
    else:
        # Legacy per-station ObsPy loop — UNCHANGED.
        for sta in _STATIONS:
            try:
                tr1 = _full_trace(e1, sta, "Z", fmin, fmax)
                tr2 = _full_trace(e2, sta, "Z", fmin, fmax)
                shift, coeff = _measure_dispatch(tr1, tr2, "a", pre, post, shift_samp, margin, step)
                t1, ot1 = _pick_time(tr1, "a"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "a"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                net = tr1.stats.network or sta[:0]
                lines.append(_fmt(net, sta, diff, coeff, "P"))
            except Exception:
                pass

    with open(os.path.join(_OUTP, f"dt.cc_P_{e1}_{e2}"), "w") as f:
        f.writelines(lines)


def _pair_S(pair):
    """Per-pair S-phase. Tries each S component, keeps the one with higher CC.
    Routes to batched CCTorch when backend != 'obspy'."""
    e1, e2 = pair
    pre, post, fmin, fmax = _window(pair)
    shift_samp, margin, step = _XC["shift_samp"], _XC["margin"], _XC["slide_step"]
    lines = [_header(e1, e2)]

    if _BACKEND in ("cctorch_cpu", "cctorch_gpu"):
        # Batched-per-pair path: gather all (sta, comp) entries in one shot.
        device = _get_torch_device("cuda" if _BACKEND == "cctorch_gpu" else "cpu")
        items, trace_lookup = [], {}
        for sta in _STATIONS:
            for comp in _XC["s_comps"]:
                try:
                    tr1 = _full_trace(e1, sta, comp, fmin, fmax)
                    tr2 = _full_trace(e2, sta, comp, fmin, fmax)
                    items.append(((sta, comp), tr1, tr2))
                    trace_lookup[(sta, comp)] = (tr1, tr2)
                except Exception:
                    continue
        results = _measure_pair_cctorch_batch(
            items, "t0", pre, post, shift_samp, margin, step,
            _XC["interp_hz"], device)
        # Pick best comp per station.
        per_sta_best = {}
        for (sta, comp), (shift, coeff) in results.items():
            if sta not in per_sta_best or coeff >= per_sta_best[sta][0]:
                per_sta_best[sta] = (coeff, shift, comp)
        for sta in _STATIONS:
            if sta not in per_sta_best:
                continue
            try:
                coeff, shift, comp = per_sta_best[sta]
                tr1, tr2 = trace_lookup[(sta, comp)]
                t1, ot1 = _pick_time(tr1, "t0"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "t0"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                net = tr1.stats.network or sta[:0]
                lines.append(_fmt(net, sta, diff, coeff, "S"))
            except Exception:
                pass
    else:
        # Legacy per-station ObsPy double-loop — UNCHANGED.
        for sta in _STATIONS:
            try:
                best = None
                for comp in _XC["s_comps"]:
                    tr1 = _full_trace(e1, sta, comp, fmin, fmax)
                    tr2 = _full_trace(e2, sta, comp, fmin, fmax)
                    shift, coeff = _measure_dispatch(tr1, tr2, "t0", pre, post, shift_samp, margin, step)
                    if best is None or coeff >= best[0]:
                        best = (coeff, shift, tr1, tr2)
                coeff, shift, tr1, tr2 = best
                t1, ot1 = _pick_time(tr1, "t0"), tr1.stats.starttime - tr1.stats.sac.b
                t2, ot2 = _pick_time(tr2, "t0"), tr2.stats.starttime - tr2.stats.sac.b
                diff = (t1 + shift - ot1) - (t2 - ot2)
                net = tr1.stats.network or sta[:0]
                lines.append(_fmt(net, sta, diff, coeff, "S"))
            except Exception:
                pass

    with open(os.path.join(_OUTS, f"dt.cc_S_{e1}_{e2}"), "w") as f:
        f.writelines(lines)


# ----------------------------------------------------------- threshold + combine
def _filter_combine(pair_dir, pairs, phase, threshold, out_file):
    """Concatenate per-pair files (pair order); keep headers + cc>=threshold data lines."""
    with open(out_file, "w") as o:
        for e1, e2 in pairs:
            p = os.path.join(pair_dir, f"dt.cc_{phase}_{e1}_{e2}")
            if not os.path.exists(p):
                continue
            for line in open(p):
                if line.startswith("#"):
                    o.write(line)
                else:
                    try:
                        cc = float(line[23:34].replace(" ", ""))
                        if threshold <= cc <= 1.0:
                            o.write(line)
                    except ValueError:
                        pass


def _drop_mainshock(in_file, out_file, cuspid):
    """Drop every pair-block whose header references the mainshock cuspid."""
    cs = str(cuspid)
    with open(in_file) as fi, open(out_file, "w") as fo:
        skip = False
        for line in fi:
            if line.startswith("#"):
                skip = cs in line.split()[1:3]
                if not skip:
                    fo.write(line)
            elif not skip:
                fo.write(line)


# --------------------------------------------------------------- orchestration
def run_xcorr(cfg, velmodel="kim1983", cores=None, xcorr_backend="obspy") -> dict:
    """Measure dt.cc for all event pairs and build the threshold/combined/no_main files.

    `xcorr_backend` ∈ {"obspy" (default — current CPU baseline, NEVER removed),
    "cctorch_cpu" (PyTorch on CPU, batched), "cctorch_gpu" (PyTorch on CUDA)}.
    The CCTorch backends batch the 1000-iteration inner slide loop into a single
    tensor op; expected speedup ~3-5× on CPU, ~10-30× on GPU. Numerical drift vs the
    ObsPy baseline is ~1e-6 relative (validated by tools/validate_cctorch_xcorr.py).

    Auto-fallback: if `cctorch_*` is requested but `torch` / CCTorch / CUDA isn't
    available, falls back to "obspy" with a warning. The CPU baseline never breaks.

    Returns {"pairs": n, "stations": n, "combined": path, "no_main": path|None}."""
    # Sanity-check + graceful fallback. Since `cctorch_gpu_batched` is the DEFAULT, this must
    # degrade to the obspy CPU baseline on any machine without a *usable* GPU — including a card
    # too new for the installed torch (reports available but errors at runtime, like the sm_120
    # case handled in hyposvi_backend). The CPU baseline never breaks.
    if xcorr_backend != "obspy":
        ok = False
        try:
            import torch
            if xcorr_backend == "cctorch_cpu":
                ok = True
            elif torch.cuda.is_available():
                try:                                        # smoke-test a real GPU op
                    (torch.zeros(8, device="cuda:0") + 1.0).sum().item()
                    ok = True
                except Exception as e:                      # noqa: BLE001 — too-new/broken GPU
                    print(f"[xcorr] WARN: GPU unusable for {xcorr_backend} "
                          f"({type(e).__name__}); falling back to obspy.")
            else:
                print(f"[xcorr] WARN: {xcorr_backend} requested but CUDA unavailable; "
                      f"falling back to obspy.")
        except ImportError:
            print(f"[xcorr] WARN: {xcorr_backend} requested but torch not importable; "
                  f"falling back to obspy.")
        if not ok:
            xcorr_backend = "obspy"
    out = config.assert_writable(config.dtcc_dir(cfg))
    out_p, out_s = os.path.join(out, "dt.cc_P"), os.path.join(out, "dt.cc_S")
    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_s, exist_ok=True)
    common = config.waveforms_dir(cfg)

    # events (by cuspid) + stations + pairs
    sumdf = sumio.read_sum(config.sum_file(cfg, velmodel))
    dirs = sorted(glob(os.path.join(common, "20*")))
    events, eid = [], {}
    for r in sumdf.itertuples():
        idx = int(r.id) % cfg.cuspid_offset
        if idx < len(dirs):
            e = os.path.basename(dirs[idx])
            events.append(e)
            eid[e] = int(r.id)
    stations = sorted({os.path.basename(f).split(".")[2]
                       for e in events for f in glob(os.path.join(common, e, "*.sac"))})
    pairs = list(combinations(events, 2))

    xc = dict(interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5, margin=0.5,
              cc_threshold=0.7, p_comp="Z", s_comps=("N", "E"), shift_samp=500,
              slide_step=0.001)
    xc.update(cfg.xcorr)                       # cluster overrides (keeps defaults above)
    xc["bandpass"] = tuple(xc["bandpass"])
    xc["s_comps"] = tuple(xc["s_comps"])

    ncores = max(1, min(cores or cfg.num_cores, len(os.sched_getaffinity(0))))
    # For GPU mode, multiple workers contend for the single GPU — auto-cap at 1
    # unless the user explicitly set cores. CPU CCTorch fans out fine; ObsPy is unchanged.
    if xcorr_backend == "cctorch_gpu" and cores is None:
        ncores = 1
    # SAFETY: CCTorch-CPU's per-worker peak memory ≈ mem_budget_gb (default 2 GB) + base
    # interpreter (~0.5 GB). With N workers the total transient peak ≈ N × 2.5 GB. Cap N
    # so the total never exceeds ~25 % of available RAM (leaves headroom for other
    # workloads on shared boxes). This prevents the multi-worker OOM that the earlier
    # naïve all-slides-batched implementation triggered.
    if xcorr_backend == "cctorch_cpu":
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            safe_workers = max(1, int(avail_gb / 8))
            if cores is None and ncores > safe_workers:
                print(f"[xcorr] cctorch_cpu: capping workers {ncores} -> {safe_workers} "
                      f"(safety: avail RAM {avail_gb:.0f} GB, ~8 GB/worker budget)")
                ncores = safe_workers
        except ImportError:
            # psutil missing — drop to a hard-coded conservative cap.
            if cores is None and ncores > 8:
                print(f"[xcorr] cctorch_cpu: capping workers {ncores} -> 8 "
                      f"(safety: psutil unavailable, using conservative cap)")
                ncores = 8
    print(f"[xcorr] {len(pairs)} pairs x {len(stations)} stations, {ncores} workers "
          f"(backend={xcorr_backend}, slide_step={xc['slide_step']}s, band={xc['bandpass']}Hz)")
    if xcorr_backend == "cctorch_gpu_batched":
        # Single-process, memory-bounded, cross-pair-batched FFT executor (own dispatch;
        # no ProcessPoolExecutor). Writes the same per-pair files, then falls through to
        # the identical combine/threshold/no_main tail below.
        run_xcorr_gpu_batched(common, stations, eid, xc,
                              dict(cfg.xcorr_pair_overrides), out_p, out_s, pairs, cores=cores)
    else:
        # CUDA + 'fork' is unsafe — child can't inherit GPU state cleanly. Use 'spawn'
        # for GPU mode; 'fork' (default) is fine for ObsPy and CCTorch-CPU.
        import multiprocessing as _mp
        mp_ctx = _mp.get_context("spawn") if xcorr_backend == "cctorch_gpu" else None
        with ProcessPoolExecutor(max_workers=ncores, initializer=_init_worker,
                                 initargs=(common, stations, eid, xc,
                                           dict(cfg.xcorr_pair_overrides), out_p, out_s,
                                           xcorr_backend),
                                 mp_context=mp_ctx) as ex:
            list(ex.map(_pair_P, pairs))
            list(ex.map(_pair_S, pairs))

    thr = xc["cc_threshold"]
    p07, s07 = os.path.join(out, "dt.cc_P_0.7"), os.path.join(out, "dt.cc_S_0.7")
    _filter_combine(out_p, pairs, "P", thr, p07)
    _filter_combine(out_s, pairs, "S", thr, s07)
    combined = os.path.join(out, "dt.cc_0.7_combined")
    with open(combined, "w") as o:
        for f in (p07, s07):
            o.write(open(f).read())

    no_main = None
    ms_cuspid = eid.get(cfg.mainshock_event_id) if cfg.mainshock_event_id else None
    if ms_cuspid is not None:
        no_main = os.path.join(out, "dt.cc_0.7_combined_no_main")
        _drop_mainshock(combined, no_main, ms_cuspid)
    print(f"[xcorr] wrote {combined}"
          + (f" + no_main (drop cuspid {ms_cuspid})" if no_main else ""))
    return dict(pairs=len(pairs), stations=len(stations),
                combined=combined, no_main=no_main)
