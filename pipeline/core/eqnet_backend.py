"""EQNet PhaseNet+ picking backend (in-process; the polarity/amplitude-aware picker).

PhaseNet+ is the only picker here that emits first-motion **polarity** and per-pick
**amplitude** — the inputs the focal-mechanism (SKHASH) stage needs. The inference path
(model build, forward, peak extraction) is ported from the validated Ulsan continuous
pipeline (`02.Ulsan_Fault_detection/KS_KG/models/pipeline/core.py`); here it is driven on
the per-event windowed SAC the cluster framework already gathers.

The EQNet clone is an EXTERNAL dependency (`config.EQNET_DIR`, override via $EQNET_DIR),
not vendored — like the hyp1.40/hypoDD binaries. `read_mseed` inside EQNet's dataset reads
MiniSEED only, so we transcode the event's 3-component SAC to temporary MiniSEED (preserving
NET.STA.LOC.CHAN ids), feed every station of one event on a single data_list line (one forward
pass), and get one pick set per station (with polarity/amplitude), mirroring the SeisBench path.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

from pipeline import config


def _ensure_eqnet_on_path():
    if config.EQNET_DIR not in sys.path:
        sys.path.insert(0, config.EQNET_DIR)


# --------------------------------------------------------------- model + inference
def load_pnplus(device="cpu"):
    """Build PhaseNet+ and load the bundled EQNet weights onto `device` (default CPU —
    cluster jobs are tiny and we stay off the GPU used by other runs)."""
    import torch
    _ensure_eqnet_on_path()
    import eqnet  # noqa: F401
    net = eqnet.models.__dict__["phasenet_plus"].build_model(
        backbone="unet", in_channels=1, out_channels=3)
    ckpt = torch.load(config.EQNET_WEIGHTS, map_location="cpu", weights_only=False)  # trusted local file
    net.load_state_dict(ckpt["model"], strict=True)
    net.to(torch.device(device)).eval()
    return net


def _pnplus_postprocess(meta, output, polarity_scale=1, event_scale=16):
    """Trim model outputs back to the un-padded patch size (mirrors EQNet predict.py)."""
    nt, nx = int(meta["nt"]), int(meta["nx"])
    meta["data"] = meta["data"][:, :, :nt, :nx]
    if "phase" in output:
        output["phase"] = output["phase"][:, :, :nt, :nx]
    if output.get("polarity") is not None:
        output["polarity"] = output["polarity"][:, :, : (nt - 1) // polarity_scale + 1, :nx]
    if output.get("event_center") is not None:
        output["event_center"] = output["event_center"][:, :, : (nt - 1) // event_scale + 1, :nx]
    if output.get("event_time") is not None:
        output["event_time"] = output["event_time"][:, :, : (nt - 1) // event_scale + 1, :nx]
    return meta, output


def _pnplus_infer(net, meta, min_prob):
    """One PhaseNet+ forward pass for a patch -> picks (with polarity/amplitude) + events.
    Channel order: phase 0=noise,1=P,2=S ; polarity 1=up,2=down (see EQNet postprocess)."""
    import torch
    from eqnet.utils import detect_peaks, extract_picks, extract_events
    output = net(meta)
    meta, output = _pnplus_postprocess(meta, output)
    dt = meta["dt_s"]
    phase = torch.softmax(output["phase"], dim=1)
    polarity = (torch.softmax(output["polarity"], dim=1)
                if output.get("polarity") is not None else None)
    ts, ti = detect_peaks(phase, vmin=min_prob, kernel=128, dt=dt.min().item())
    picks = extract_picks(
        ti, ts, file_name=meta["file_name"], station_id=meta["station_id"],
        begin_time=meta.get("begin_time"), begin_time_index=meta.get("begin_time_index"),
        dt=dt, vmin=min_prob, phases=["P", "S"], polarity_score=polarity, waveform=meta["data"])
    events = []
    if output.get("event_center") is not None:
        event_prob = torch.sigmoid(output["event_center"])
        es, ei = detect_peaks(event_prob, vmin=min_prob, kernel=16, dt=dt.min().item() * 16.0)
        events = extract_events(
            ei, es, file_name=meta["file_name"], station_id=meta["station_id"],
            begin_time=meta.get("begin_time"), begin_time_index=meta.get("begin_time_index"),
            dt=dt, vmin=min_prob, event_time=output["event_time"], waveform=meta["data"])
    return dict(picks=picks, events=events)


# --------------------------------------------------------------- event driver
def _sac_to_mseed(sac_path, out_dir):
    """Read one gathered SAC and write a MiniSEED copy with NET.STA.LOC.CHAN taken from the
    filename `<eid>.<net>.<sta>.<chan>.sac` (robust regardless of SAC header completeness)."""
    from obspy import read
    base = os.path.basename(sac_path)
    parts = base.split(".")
    # <eid>.<net>.<sta>.<chanCOMP>.sac  -> net, sta, chan(=HHZ etc.)
    net, sta, chan = parts[1], parts[2], parts[3]
    tr = read(sac_path)[0]
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, "", chan
    out = os.path.join(out_dir, f"{net}.{sta}.{chan}.mseed")
    tr.write(out, format="MSEED")
    return out


def pick_event_pnplus(net, comp_files, min_prob=None, highpass=None, sampling_rate=None):
    """Run PhaseNet+ over one event's component SAC files (any number of stations) in a
    single forward pass. Returns a picks DataFrame with columns:
        station_id, phase, time (ISO str), probability, polarity, amplitude
    `station_id` is NET.STA.LOC.<band> (3 components grouped), e.g. "KS.ADO..HH".
    """
    import torch
    import torch.utils.data
    _ensure_eqnet_on_path()
    from eqnet.data import SeismicTraceIterableDataset

    min_prob = config.PNPLUS_MIN_PROB if min_prob is None else min_prob
    highpass = config.PNPLUS_HIGHPASS if highpass is None else highpass
    sampling_rate = 100.0 if sampling_rate is None else sampling_rate
    comp_files = [f for f in comp_files if os.path.exists(f)]
    if not comp_files:
        return pd.DataFrame(columns=["station_id", "phase", "time", "probability",
                                     "polarity", "amplitude"])

    rows = []
    with tempfile.TemporaryDirectory() as td:
        mseeds = [_sac_to_mseed(f, td) for f in comp_files]
        list_path = os.path.join(td, "data_list.txt")
        with open(list_path, "w") as fh:
            fh.write(",".join(mseeds))            # all stations of this event on one line
        # cut_patch must be enabled AFTER construction (its _count() only supports h5 cut_patch)
        dataset = SeismicTraceIterableDataset(
            data_path="", data_list=list_path, format="mseed", dataset="seismic_trace",
            training=False, sampling_rate=sampling_rate, highpass_filter=highpass,
            cut_patch=False, nt=config.PNPLUS_NT)
        dataset.cut_patch = True
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
        with torch.inference_mode():
            for meta in loader:
                r = _pnplus_infer(net, meta, min_prob)
                for sub in r["picks"]:
                    for p in sub:
                        rows.append(dict(
                            station_id=p["station_id"], phase=p["phase_type"],
                            time=p["phase_time"], probability=float(p["phase_score"]),
                            polarity=float(p.get("phase_polarity") or 0.0),
                            amplitude=float(p.get("phase_amplitude") or 0.0)))
    return pd.DataFrame(rows)
