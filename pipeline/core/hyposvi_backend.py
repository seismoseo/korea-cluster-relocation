"""Python absolute-location backend — HypoSVI adapter for the HypoInverse stage.

Goal: produce a HYPOINVERSE-format `.sum` (`1.HypoInv/<vm>/<Region>.sum`) that
`sumio.read_sum()` parses, so the rest of the pipeline (ph2dt prep, focal
mechanism, viz) doesn't know whether the absolute locations came from Fortran
hyp1.40 or HypoSVI.

HypoSVI locates with a pre-trained EikoNet travel-time network (one P + one S
checkpoint), NOT a layered .crh model. The checkpoint pair is selected per run
via cfg.hyposvi_eikonet_p / _s (or env HYPOSVI_EIKONET_P / _S); each checkpoint
dir carries an `eikonet_meta.json` written at training time recording the exact
domain (xmin/xmax/projection/csv) needed to reconstruct the VelocityClass.

CRITICAL — EikoNet unit convention: EikoNet projects lon/lat -> UTM but does NOT
divide by 1000, while depth and the velocity CSV are in km / km/s. The EikoNet
must therefore be TRAINED with a `+units=km` projection so all three axes are km;
otherwise the eikonal loss |grad T| = 1/v is unit-inconsistent and travel times
come back ~1000x too large (the locator then returns NaN). The meta.json's
projection string is the single source of truth and is reused verbatim here.

Picks are read straight from the SAC a (P) / t0 (S) headers, mirroring
hypoinverse.write_phs(), so the same arrivals drive both backends. We do NOT
refactor write_phs — the Fortran path must stay byte-identical.
"""
from __future__ import annotations

import json
import math
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.geodetics import gps2dist_azimuth

from pipeline import config


def _auto_device():
    """Return 'cuda:0' only if torch can ACTUALLY run on the GPU, else 'cpu'.

    A trivial GPU op is the reliable test: `torch.cuda.is_available()` returns True even for a
    GPU too NEW for the installed CUDA wheel (e.g. RTX PRO 6000 Blackwell = sm_120 on a PyTorch
    that tops out at sm_90), where every real kernel then raises 'no kernel image available'.
    We fall back to CPU with a one-line note instead of crashing mid-locate."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        try:
            _ = (torch.zeros(8, device="cuda:0") + 1.0).sum().item()   # smoke test
            return "cuda:0"
        except Exception as e:                                          # noqa: BLE001
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:                                          # noqa: BLE001
                name = "?"
            print(f"[hyposvi] GPU present ({name}) but unusable by this PyTorch build "
                  f"({type(e).__name__}) — using CPU. Install a CUDA wheel matching your GPU's "
                  f"compute capability (https://pytorch.org/get-started/locally/) to enable it.")
            return "cpu"
    except Exception:                                                  # noqa: BLE001
        return "cpu"


# --------------------------------------------------------------- EikoNet loading
def _ensure_on_path(env_var, default_hint):
    """Put a repo clone on sys.path from an env var (EikoNet / HypoSVI live outside PyPI)."""
    d = os.environ.get(env_var)
    if d and d not in sys.path:
        sys.path.insert(0, d)
    return d


def _hyposvi_velmodel(cfg):
    """Which velocity model HypoSVI locates with: cfg.hyposvi_velmodel > first
    velocity model name > 'kim1983'."""
    vm = getattr(cfg, "hyposvi_velmodel", None)
    if vm:
        return vm
    vms = getattr(cfg, "velocity_models", None)
    return vms[0].name if vms else "kim1983"


def _bundled_eikonet_dir():
    """pipeline/velocity_models — the bundled-weights root (next to the config pkg)."""
    return os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "velocity_models")


def _bundled_ckpt(velmodel, phase):
    """Resolve the bundled best-val checkpoint for <velmodel> <phase> from its
    eikonet_meta.json, or None if not present. Fetch with `python -m
    pipeline.core.fetch_eikonet`."""
    model_dir = os.path.join(_bundled_eikonet_dir(), f"eikonet_{velmodel}", f"{velmodel}_{phase}")
    meta_path = os.path.join(model_dir, "eikonet_meta.json")
    if not os.path.isfile(meta_path):
        return None
    best = json.load(open(meta_path)).get("best_checkpoint")
    if not best:
        return None
    ckpt = os.path.join(model_dir, best)
    return ckpt if os.path.isfile(ckpt) else None


def _resolve_eikonet_paths(cfg):
    """(p_ckpt, s_ckpt): cfg.hyposvi_eikonet_p/_s > env HYPOSVI_EIKONET_P/_S > bundled
    weights under pipeline/velocity_models/eikonet_<velmodel>/ (auto-discovered from
    eikonet_meta.json). Raises with guidance if none resolve."""
    vm = _hyposvi_velmodel(cfg)
    p = (getattr(cfg, "hyposvi_eikonet_p", None) or os.environ.get("HYPOSVI_EIKONET_P")
         or _bundled_ckpt(vm, "p"))
    s = (getattr(cfg, "hyposvi_eikonet_s", None) or os.environ.get("HYPOSVI_EIKONET_S")
         or _bundled_ckpt(vm, "s"))
    if not p or not s:
        raise RuntimeError(
            f"HypoSVI EikoNet checkpoints for '{vm}' not found. Either fetch the bundled "
            f"weights (`python -m pipeline.core.fetch_eikonet --velmodel {vm}`), or set "
            "cfg.hyposvi_eikonet_p/_s / HYPOSVI_EIKONET_P / HYPOSVI_EIKONET_S. "
            "See docs/python_backend/README.md.")
    for label, path in (("P", p), ("S", s)):
        if not os.path.isfile(path):
            raise RuntimeError(f"HypoSVI {label}-EikoNet checkpoint not found: {path}")
    return p, s


class _MetaVelocity:
    """Minimal stand-in VelocityClass for reloading a trained EikoNet at inference: carries only
    `projection`/`xmin`/`xmax` (all `Model._projection` + normalisation read). `.eval()` is never
    called at inference, so the full velocity model is not needed to *use* a trained network — this
    lets non-1-D models (e.g. the 3-D `neasia`) reload without their training grid."""
    def __init__(self, xmin, xmax, projection):
        self.xmin = xmin
        self.xmax = xmax
        self.projection = projection

    def eval(self, Xp):
        raise RuntimeError("_MetaVelocity is inference-only; the training grid is not loaded.")


def _load_eikonet(ckpt_path, device):
    """Reconstruct an EikoNet Model from a checkpoint + its sibling eikonet_meta.json."""
    # EikoNet / HypoSVI are external clones; make them importable.
    _ensure_on_path("HYPOSVI_DIR", "/path/to/HypoSVI")
    eikonet_dir = os.environ.get("EIKONET_DIR")
    if eikonet_dir and eikonet_dir not in sys.path:
        sys.path.insert(0, eikonet_dir)
    from EikoNet.database import Graded1DVelocity
    from EikoNet.model import Model

    model_dir = os.path.dirname(ckpt_path)
    meta_path = os.path.join(model_dir, "eikonet_meta.json")
    if not os.path.isfile(meta_path):
        raise RuntimeError(
            f"missing {meta_path} — the EikoNet training domain (xmin/xmax/projection/csv) "
            "is required to reconstruct the model. Re-run training with the metadata writer.")
    meta = json.load(open(meta_path))
    if "csv" in meta:
        # 1-D model: rebuild the graded velocity from its CSV.
        csv_path = meta["csv"]
        if not os.path.isabs(csv_path):
            # csv may sit next to the checkpoint (model_dir) or one level up (the
            # <velmodel> dir holding both phase subdirs). Try both.
            base = os.path.basename(csv_path)
            for cand in (os.path.join(model_dir, base), os.path.join(os.path.dirname(model_dir), base)):
                if os.path.isfile(cand):
                    csv_path = cand
                    break
            else:
                raise RuntimeError(f"EikoNet velocity csv {base!r} not found near {model_dir}")
        vmodel = Graded1DVelocity(csv_path, xmin=meta["xmin"], xmax=meta["xmax"],
                                  projection=meta["projection"])
    else:
        # 3-D (or any non-1-D) model: at INFERENCE the velocity comes from the trained network,
        # not the VelocityClass — Model only reads .projection/.xmin/.xmax (for input
        # normalisation). So a lightweight stub carrying those from the meta is sufficient to
        # reload the model (the full 3-D grid is only needed for training).
        vmodel = _MetaVelocity(meta["xmin"], meta["xmax"], meta["projection"])
    model = Model(model_dir, vmodel, device=device)
    model.load(ckpt_path)
    return model, meta


# --------------------------------------------------------------- input adapters
def _consistent_origin_time(hyp, picks, mp, ms, device):
    """Origin time = median_s( arrival_s − TravelTime(hyp, station_s) ) over the picks.

    This is the SAME self-consistency hyp1.40 enforces for the Fortran path: the origin
    must satisfy org = pick − TT(hypocentre) so that, in the double-difference relocation,
    the catalog term (pick−org)−TT cancels and the residual reduces to the pure cross-
    correlation lag (origin-time independent). HypoSVI's native OT is an SVGD-marginalised
    estimate that does NOT satisfy this, which otherwise leaks the OT error into dt.ct/dt.cc.
    The HypoSVI HYPOCENTRE is untouched — only the origin time is made self-consistent.
    Returns UTCDateTime or None."""
    import torch
    if not {"X", "Y", "Z", "PhasePick", "DT"}.issubset(picks.columns):
        return None
    implied = []
    for model, phase in ((mp, "P"), (ms, "S")):
        ph = picks[picks.PhasePick == phase]
        if len(ph) == 0:
            continue
        Xp = np.zeros((len(ph), 6), dtype=np.float32)
        Xp[:, 0], Xp[:, 1], Xp[:, 2] = float(hyp[0]), float(hyp[1]), float(hyp[2])
        Xp[:, 3], Xp[:, 4], Xp[:, 5] = ph.X.values, ph.Y.values, ph.Z.values
        with torch.no_grad():
            tt = model.TravelTimes(torch.tensor(Xp).to(device)).cpu().numpy()
        for k, (_, r) in enumerate(ph.iterrows()):
            implied.append(UTCDateTime(pd.Timestamp(r.DT).to_pydatetime()) - float(tt[k]))
    if not implied:
        return None
    return UTCDateTime(float(np.median([t.timestamp for t in implied])))


def _stations_df(cfg):
    """HypoSVI Stations frame: Network, Station, X=lon, Y=lat, Z=depth_km (=-elev).

    Built from the same SAC headers write_sta() reads, so the roster matches the
    Fortran path exactly."""
    coords = {}
    for f in glob(os.path.join(config.waveforms_dir(cfg), "20*", "*.sac")):
        sta = os.path.basename(f).split(".")[2]
        if sta in coords:
            continue
        tr = read(f)[0]
        s = tr.stats.sac
        elev = float(getattr(s, "stel", 0.0) or 0.0)
        coords[sta] = (tr.stats.network or "KS", float(s.stla), float(s.stlo), elev)
    rows = [(net, sta, lo, la, -elev / 1000.0)        # Z is depth (km), down-positive
            for sta, (net, la, lo, elev) in sorted(coords.items())]
    return pd.DataFrame(rows, columns=["Network", "Station", "X", "Y", "Z"])


def _events_dict(cfg):
    """Build HypoSVI EVTS = {cuspid: {'Picks': DataFrame}} from SAC a/t0 headers.

    cuspid = 200000+idx over sorted event dirs — identical to write_phs()'s id
    scheme, so .sum ID-NUM lines up with the Fortran path and with ph2dt later.
    Picks columns: Network, Station, PhasePick('P'|'S'), DT(UTCDateTime), PickError(s)."""
    evts, meta = {}, {}
    event_dirs = sorted(glob(os.path.join(config.waveforms_dir(cfg), "20*")))
    for idx, ed in enumerate(event_dirs):
        cuspid = 200000 + idx
        rows = []
        seen_p, seen_s = set(), set()
        for sac in sorted(glob(ed + "/*.sac"))[::-1]:
            tr = read(sac)[0]
            s = tr.stats.sac
            sta = os.path.basename(sac).split(".")[2]
            net = tr.stats.network or "KS"
            comp = tr.stats.channel[-1]
            if comp == "Z" and sta not in seen_p and s.get("a", -12345.0) != -12345.0:
                t = tr.stats.starttime - s.b + s.a
                rows.append((net, sta, "P", pd.Timestamp(t.datetime), 0.1))
                seen_p.add(sta)
            if comp in ("N", "E") and sta not in seen_s and s.get("t0", -12345.0) != -12345.0:
                t = tr.stats.starttime - s.b + s.t0
                rows.append((net, sta, "S", pd.Timestamp(t.datetime), 0.2))
                seen_s.add(sta)
        if len(rows) < 2:
            # HypoSVI's log-likelihood forms differential times between observation
            # PAIRS (location.py: T_obs[:,pairs]), so it needs >=2 observations; a
            # single pick leaves T_obs 1-D and raises IndexError mid-LocateEvents,
            # killing the whole run. Skip the under-determined event with a note —
            # same outcome as a no-pick event (absent from the .sum), no crash.
            if rows:
                print(f"[hyposvi] skipping event {cuspid}: only {len(rows)} observation(s) "
                      f"— under-determined (HypoSVI needs >=2)")
            continue
        picks = pd.DataFrame(rows, columns=["Network", "Station", "PhasePick", "DT", "PickError"])
        evts[str(cuspid)] = {"Picks": picks}
        meta[str(cuspid)] = cuspid
    return evts, meta


# --------------------------------------------------------------- output adapter
_SUM_HEADER = ("   DATE     TIME, SEC ,   LAT  ,   LON   , DEPTH,PREF,MAG,NM,NUM,GAP,"
               " DMIN, RMS ,  ERH,  ERZ,QASR,    ID-NUM,LOC,DUR,MAG1,AMP,MAG1,EXT,MAG ,")


def _azimuthal_gap(ev_lat, ev_lon, picks, stations):
    """Largest gap (deg) between station azimuths — HYPOINVERSE GAP column."""
    st = stations.set_index("Station")
    az = []
    for sta in picks["Station"].unique():
        if sta not in st.index:
            continue
        r = st.loc[sta]
        az.append(gps2dist_azimuth(ev_lat, ev_lon, float(r.Y), float(r.X))[1])
    if len(az) < 2:
        return 359.0
    az = sorted(az)
    gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)] + [az[0] + 360.0 - az[-1]]
    return max(gaps)


def _sum_row(cuspid, loc, picks, stations):
    """One HYPOINVERSE H71 `.sum` line from a HypoSVI event solution.

    loc['Hypocentre'] is [lon, lat, depth_km] (projection already inverted to
    lon/lat by HypoSVI); loc['HypocentreError'] is [ex, ey, ez] in km."""
    lon, lat, dep = float(loc["Hypocentre"][0]), float(loc["Hypocentre"][1]), float(loc["Hypocentre"][2])
    ex, ey, ez = (float(v) for v in loc["HypocentreError"])
    erh = math.hypot(ex, ey)
    ot = UTCDateTime(pd.Timestamp(loc["OriginTime"]).to_pydatetime())
    rms = float(loc.get("OriginTime_std", 0.0) or 0.0)
    num = int(len(picks))
    gap = _azimuthal_gap(lat, lon, picks, stations)
    qual = "B" if (rms < 0.3 and gap < 180) else "D"
    date = f"{ot.year:04d}/{ot.month:02d}/{ot.day:02d} {ot.hour:02d}:{ot.minute:02d}"
    sec = ot.second + ot.microsecond / 1e6
    # SEC at full (ms) precision, NOT the HYPOINVERSE 0.01 s: the relocDD-py phase.dat
    # derives travel times from this origin, and a 0.01 s rounding here costs ~60 m
    # (0.01 s × 6 km/s) in the relative relocation vs the full-precision Fortran path.
    # Columns mirror read_sum's expectations; unused MAG/DUR/etc left blank.
    return (f"{date},{sec:08.5f}, {lat:7.4f}, {lon:8.4f},{dep:6.2f},     ,  ,  ,"
            f"{num:3d},{gap:3.0f}, 0.0,{rms:5.2f},{erh:5.1f},{ez:5.1f},{qual}   ,"
            f"{cuspid:10d},   ,     ,  ,     ,  ,     ,  ")


def posterior_cov(loc):
    """Full posterior covariance (km^2) of the HypoSVI solution, for oriented error ellipses.

    Uses HypoSVI's own stored `loc['Covariance']` — the 3x3 KDE covariance of the SVGD
    particle cloud, in the EikoNet's PROJECTED km (axis order [E, N, depth]), and the SAME
    matrix the reported ERH/ERZ derive from (err = sqrt(diag(cov)) * z_0.95). Its off-diagonals
    give the E-N orientation and the depth<->horizontal trade-off, so a 2-D ellipse can be
    drawn in every plane (k=2.448 for 95% joint), exactly like the HYPOINVERSE .prt covariance.

    NOTE: do NOT use loc['SVGD_points'] for this — after LocateEvents HypoSVI re-projects those
    back to lon/lat DEGREES (mixed units with depth-km), so their np.cov is meaningless.

    Returns (cov_ee, cov_nn, cov_en, cov_zz, cov_ez, cov_nz); NaN if unavailable."""
    C = loc.get("Covariance") if loc else None
    if C is None:
        return (np.nan,) * 6
    C = np.asarray(C, dtype=float)
    if C.shape != (3, 3):
        return (np.nan,) * 6
    return (float(C[0, 0]), float(C[1, 1]), float(C[0, 1]),
            float(C[2, 2]), float(C[0, 2]), float(C[1, 2]))


def _write_sum(cfg, vmname, results, stations):
    """Write 1.HypoInv/<vmname>/<Region>.sum from located events; return its path.

    Also writes a sibling `<Region>.svicov.csv` carrying the per-event posterior covariance
    (cov_ee/cov_nn/cov_en/cov_zz, km^2) from the SVGD particle cloud, so a comparison notebook
    can draw oriented posterior ellipses (the .sum only stores scalar ERH/ERZ)."""
    out_dir = config.assert_writable(config.velmodel_dir(cfg, vmname))
    os.makedirs(out_dir, exist_ok=True)
    sum_path = config.sum_file(cfg, vmname)
    cov_path = os.path.join(out_dir, f"{cfg.region}.svicov.csv")
    n = 0
    cov_rows = ["id,cov_ee,cov_nn,cov_en,cov_zz,cov_ez,cov_nz"]
    with open(sum_path, "w") as f:
        f.write(_SUM_HEADER + "\n")
        for cuspid, loc, picks in results:
            if loc is None or not np.isfinite(loc["Hypocentre"][0]):
                continue
            f.write(_sum_row(cuspid, loc, picks, stations) + "\n")
            ee, nn, en, zz, ez, nz = posterior_cov(loc)
            cov_rows.append(f"{cuspid},{ee:.6g},{nn:.6g},{en:.6g},{zz:.6g},{ez:.6g},{nz:.6g}")
            n += 1
    if n == 0:
        raise RuntimeError(f"HypoSVI located 0 events — empty .sum at {sum_path}")
    with open(cov_path, "w") as f:
        f.write("\n".join(cov_rows) + "\n")
    return sum_path


# --------------------------------------------------------------- locator (load once, reuse)
def _resolve_device(cfg):
    """cfg.hyposvi_device wins; "auto" smoke-tests a USABLE GPU and falls back to CPU
    (e.g. a GPU NEWER than the installed CUDA wheel, where is_available() lies)."""
    device = getattr(cfg, "hyposvi_device", None)
    if not device or device == "auto":
        device = _auto_device()
    print(f"[hyposvi] device: {device}")
    return device


def make_locator(cfg, device=None) -> dict:
    """Build the HypoSVI locator ONCE: load the EikoNet P/S pair and construct the SVGD
    object. Returns a handle reusable across many LocateEvents() calls — this is the key
    to batch throughput (loading the EikoNet is the dominant per-call cost, so a 10k-event
    catalog run must load it once, not once per event). See `set_region_box` + `locate_batch`."""
    device = device or _resolve_device(cfg)
    p_ckpt, s_ckpt = _resolve_eikonet_paths(cfg)
    _ensure_on_path("HYPOSVI_DIR", "/path/to/HypoSVI")
    from HypoSVI.location import HypoSVI
    mp, meta_p = _load_eikonet(p_ckpt, device)
    ms, _ = _load_eikonet(s_ckpt, device)
    H = HypoSVI([mp, ms], Phases=["P", "S"], device=device)
    # RBF kernel width: a static value (km) curbs HypoSVI's dynamic over-dispersion of the
    # reported location uncertainty (Smith et al. 2022 sec 4.3). None keeps the dynamic default.
    rbf_sigma = getattr(cfg, "hyposvi_rbf_sigma", None)
    if rbf_sigma is not None:
        H.K.sigma = float(rbf_sigma)
    return dict(H=H, mp=mp, ms=ms, device=device, meta_p=meta_p,
                epochs=int(getattr(cfg, "hyposvi_epochs", 175)))


def set_region_box(loc, region_bounds, margin=0.15, zmax=25.0):
    """Restrict the SVGD initialisation box to a region (in the EikoNet's projected km).

    CRITICAL: the EikoNet is trained over all of Korea (~600x700 km), but SVGD seeds
    particles uniformly across H.xmin..H.xmax and cannot collapse a 600 km cloud onto a
    few-km source in ~175 steps (symptom: ±20-30 km "uncertainty", biased depth). Seed
    the box around the target region (+ margin) so particles start near the events. For a
    PER-EVENT batch run, call this with a tight box around each event's epicenter before
    every locate_batch() — the box is mutated in place on the shared locator."""
    H = loc["H"]
    lat0, lat1, lon0, lon1 = region_bounds
    (x0, y0) = H.projection(lon0 - margin, lat0 - margin)
    (x1, y1) = H.projection(lon1 + margin, lat1 + margin)
    H.xmin = [min(x0, x1), min(y0, y1), 0.0]
    H.xmax = [max(x0, x1), max(y0, y1), float(zmax)]


def locate_batch(loc, evts, stations, scratch_dir) -> list:
    """Locate a batch of events with a prepared locator. Returns [(cuspid_int, location, picks)].

    The caller is responsible for the SVGD box (call set_region_box first). Origin time is
    made self-consistent (org = pick - TT(hyp)) without touching the hypocentre."""
    H, mp, ms, device = loc["H"], loc["mp"], loc["ms"], loc["device"]
    os.makedirs(scratch_dir, exist_ok=True)
    H.LocateEvents(evts, stations, scratch_dir, epochs=loc["epochs"])
    results = []
    for cuspid_str, ev in evts.items():
        L = ev.get("location")
        if L is not None and np.isfinite(L["Hypocentre"][0]):
            ot = _consistent_origin_time(L["Hypocentre"], ev["Picks"], mp, ms, device)
            if ot is not None:
                L["OriginTime"] = str(ot)                # self-consistent OT (hypocentre unchanged)
        results.append((int(cuspid_str), L, ev["Picks"]))
    return results


# --------------------------------------------------------------- driver
def run_hyposvi(cfg, velmodels=None) -> dict:
    """HypoSVI equivalent of run_hypoinverse(): locate every event with the trained
    EikoNet pair and write a HYPOINVERSE-format .sum per requested velocity model.

    The EikoNet pair already encodes one velocity model; we write the .sum under
    each requested velmodel name (default: the meta's velmodel) so downstream
    stages that key off a specific velmodel name find it. Returns {vmname: sum_path}."""
    loc = make_locator(cfg)
    stations = _stations_df(cfg)
    evts, _idmap = _events_dict(cfg)
    if not evts:
        raise RuntimeError("HypoSVI: no events with picks found under waveforms dir.")
    set_region_box(loc, cfg.region_bounds,
                   margin=float(getattr(cfg, "hyposvi_box_margin_deg", 0.15)),
                   zmax=float(getattr(cfg, "hyposvi_depth_max_km", 25.0)))
    out_scratch = os.path.join(config.hyp_dir(cfg), "_hyposvi")
    results = locate_batch(loc, evts, stations, out_scratch)

    # Which velocity-model name(s) to write the .sum under.
    meta_p = loc["meta_p"]
    if velmodels is None:
        names = [meta_p.get("velmodel")] if meta_p.get("velmodel") else [v.name for v in cfg.velocity_models]
    else:
        names = list(velmodels)
    return {name: _write_sum(cfg, name, results, stations) for name in names}
