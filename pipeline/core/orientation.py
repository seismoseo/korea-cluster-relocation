"""Time-dependent horizontal orientation for S-on-T differential times.

Angles are clockwise misorientations of the recorded N axis. Wiring repairs are
applied before ObsPy rotation with ``ba = (station_to_source_baz - theta) % 360``.
No table is extrapolated: its author must explicitly define the usable epochs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_json(path, value):
    tmp = str(path) + f".tmp{os.getpid()}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _time(value, *, end=False):
    text = str(value).strip()
    if not text:
        raise ValueError("orientation epochs require explicit start and end")
    t = pd.Timestamp(text)
    if pd.isna(t):
        raise ValueError(f"invalid orientation time: {value!r}")
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t + pd.Timedelta(days=1) if end and _DAY.fullmatch(text) else t


def _flag(value, name):
    s = str(value).strip().lower()
    if s in ("", "0", "false", "no"):
        return False
    if s in ("1", "true", "yes"):
        return True
    raise ValueError(f"{name} must be true/false or 1/0, got {value!r}")


@dataclass(frozen=True)
class Epoch:
    network: str
    station: str
    sensor: str
    location: str
    start: pd.Timestamp
    end: pd.Timestamp
    theta: float
    flip_n: bool = False
    flip_e: bool = False
    swap_ne: bool = False


class OrientationTable:
    """Validated UTC epochs; starts inclusive, timestamp ends exclusive.

    Date-only ends include the complete UTC day. Empty location matches any
    location; overlapping wildcard and explicit location rows are rejected.
    """

    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        self.sha256 = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        df = df.rename(columns={old: new for old, new in (("net", "network"), ("sta", "station"))
                                if old in df and new not in df})
        required = {"network", "station", "sensor", "start", "end"}
        missing = required - set(df.columns)
        angle = "theta_deg" if "theta_deg" in df else "theta_M_apply"
        if missing or angle not in df:
            raise ValueError(f"{path}: orientation CSV needs network,station,sensor,start,end,theta_deg "
                             "(theta_M_apply is an accepted angle alias)")
        self.epochs = {}
        for i, row in df.iterrows():
            try:
                key = tuple(row[k].strip() for k in ("network", "station", "sensor"))
                if not all(key) or len(key[2]) != 2:
                    raise ValueError("network/station must be nonempty and sensor must be a two-character band")
                theta = float(row[angle])
                if not np.isfinite(theta):
                    raise ValueError("angle must be finite")
                start, end = _time(row.start), _time(row.end, end=True)
                if start >= end:
                    raise ValueError("start must precede end")
                loc = row.get("location", "").strip()
                loc = "" if loc == "*" else loc
                epoch = Epoch(*key, loc, start, end, theta,
                              *(_flag(row.get(k, ""), k) for k in ("flip_n", "flip_e", "swap_ne")))
                group = self.epochs.setdefault(key, [])
                for other in group:
                    if (not loc or not other.location or loc == other.location) and \
                            start < other.end and other.start < end:
                        raise ValueError(f"overlapping epochs for {'.'.join(key)}")
                group.append(epoch)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: row {i + 2}: {exc}") from exc

    def lookup(self, network, station, sensor, origin, location=""):
        t = _time(origin)
        return next((r for r in self.epochs.get((network, station, sensor), ())
                     if (not r.location or r.location == location) and r.start <= t < r.end), None)


def _source_signature(paths):
    return [(str(p), os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in paths]


def _coordinate(sac, name, bound):
    v = float(sac.get(name, -12345))
    if not np.isfinite(v) or abs(v) > bound:
        raise ValueError(f"rotation requires valid SAC {name}, got {v}")
    return v


def rotate_horizontal(n, e, theta, reference=None, *, flip_n=False, flip_e=False, swap_ne=False):
    """Return aligned R/T traces, geographic baz, effective angle; originals stay unchanged."""
    ident_n = (n.stats.network, n.stats.station, n.stats.location, n.stats.channel[:-1])
    ident_e = (e.stats.network, e.stats.station, e.stats.location, e.stats.channel[:-1])
    if ident_n != ident_e or not n.stats.channel.endswith("N") or not e.stats.channel.endswith("E"):
        raise ValueError("N/E must belong to the same network, station, location and sensor")
    sr = float(n.stats.sampling_rate)
    if sr <= 0 or not np.isclose(sr, e.stats.sampling_rate, rtol=0, atol=1e-8):
        raise ValueError("N/E sample rates differ")
    offset = float(e.stats.starttime - n.stats.starttime) * sr
    if abs(offset - round(offset)) > 1e-3:
        raise ValueError("N/E sample grids are not aligned")
    origin = n.stats.starttime - float(n.stats.sac.b)
    eorigin = e.stats.starttime - float(e.stats.sac.b)
    if abs(eorigin - origin) > 1e-4:
        raise ValueError("N/E SAC reference times differ")
    for hdr in ("a", "t0"):
        p, q = n.stats.sac.get(hdr, -12345), e.stats.sac.get(hdr, -12345)
        if p != -12345 and q != -12345 and abs(float(p) - float(q)) > 1e-4:
            raise ValueError(f"N/E {hdr} picks differ")
    start, end = max(n.stats.starttime, e.stats.starttime), min(n.stats.endtime, e.stats.endtime)
    if start >= end:
        raise ValueError("N/E have no common time span")
    n, e = n.slice(start, end).copy(), e.slice(start, end).copy()
    if n.stats.npts != e.stats.npts or abs(n.stats.starttime - e.stats.starttime) > 1e-6:
        raise ValueError("N/E samples do not align after trimming")
    s = n.stats.sac
    stla, stlo = _coordinate(s, "stla", 90), _coordinate(s, "stlo", 180)
    if reference is None:
        evla, evlo = _coordinate(s, "evla", 90), _coordinate(s, "evlo", 180)
    else:
        evla, evlo = map(float, reference)
        if not np.isfinite([evla, evlo]).all() or abs(evla) > 90 or abs(evlo) > 180:
            raise ValueError("rotation_reference must be a valid source (latitude, longitude)")
    baz = gps2dist_azimuth(stla, stlo, evla, evlo)[1]
    ba = (baz - float(theta)) % 360
    nd, ed = np.asarray(n.data, dtype=float), np.asarray(e.data, dtype=float)
    if swap_ne:
        nd, ed = ed, nd
    if flip_n:
        nd = -nd
    if flip_e:
        ed = -ed
    radial, transverse = rotate_ne_rt(nd, ed, ba)
    result = []
    for comp, data, az in (("R", radial, baz + 180), ("T", transverse, baz - 90)):
        tr = n.copy()
        tr.data = np.asarray(data, dtype=np.float32)
        tr.stats.channel = n.stats.channel[:-1] + comp
        tr.stats.sac.b = float(tr.stats.starttime - origin)
        tr.stats.sac.user0, tr.stats.sac.user1 = float(theta), float(ba)
        tr.stats.sac.kuser0 = "ROT"
        tr.stats.sac.cmpaz, tr.stats.sac.cmpinc = az % 360, 90.0
        # A pick present on only E is still valid for the rotated sensor.
        for hdr in ("a", "t0"):
            if tr.stats.sac.get(hdr, -12345) == -12345 and e.stats.sac.get(hdr, -12345) != -12345:
                tr.stats.sac[hdr] = e.stats.sac[hdr]
        result.append(tr)
    return (*result, baz, ba)


def prepare_rotation(cfg, common, events, out):
    """Create R/T next to gathered N/E and write auditable per-sensor provenance.

    Table-less T explicitly uses nominal north. A supplied table defaults to
    raising on missing coverage; ``zero`` warns, ``skip`` omits derived R/T.
    Returns configuration and per-event signatures for pair-cache invalidation.
    """
    table_path = getattr(cfg, "orientation_csv", None)
    table = OrientationTable(table_path) if table_path else None
    missing = getattr(cfg, "orientation_missing", "error")
    if missing not in ("error", "zero", "skip"):
        raise ValueError("orientation_missing must be error, zero or skip")
    reference = getattr(cfg, "rotation_reference", None)
    settings = dict(version=_VERSION, mode="derived", table_path=table.path if table else None,
                    table_sha256=table.sha256 if table else None,
                    missing=missing, reference=reference)
    signature = _digest(settings)
    dest = Path(out) / "orientation"
    previous_path = dest / "rotation.json"
    try:
        previous = json.loads(previous_path.read_text())
    except (OSError, ValueError):
        previous = {}
    records, event_signatures = [], {}
    for event in events:
        files = sorted(glob(os.path.join(common, event, f"{event}.*.*.*N.sac")))
        # Correlation selects one sensor per NET.CODE. Never silently select the
        # first of several sensors in a gathered event.
        seen = set()
        # P remains on Z; its source/pick changes also invalidate pair checkpoints.
        event_inputs = _source_signature(sorted(glob(os.path.join(common, event, f"{event}.*.*.*Z.sac"))))
        keep_outputs = set()
        for nf in files:
            parts = Path(nf).name.split(".")
            net, sta, channel = parts[1:4]
            identity = (net, sta)
            if identity in seen:
                raise ValueError(f"{event}/{net}.{sta}: multiple N/E sensors; select one before S-on-T")
            seen.add(identity)
            ef, rf, tf = nf[:-5] + "E.sac", nf[:-5] + "R.sac", nf[:-5] + "T.sac"
            paths = [nf] + ([ef] if os.path.isfile(ef) else [])
            srcsig = _source_signature(paths)
            event_inputs.extend(srcsig)
            n = read(nf)[0]
            if not np.isfinite(float(n.stats.sac.b)) or n.stats.sac.b == -12345:
                raise ValueError(f"{nf}: SAC b must define the origin reference")
            origin = n.stats.starttime - float(n.stats.sac.b)
            location = str(n.stats.location or "").strip()
            has_e = os.path.isfile(ef)
            epoch = table.lookup(net, sta, channel[:-1], str(origin), location) if table and has_e else None
            status = "measured" if epoch else "nominal"
            if table and epoch is None and has_e:
                label = f"{event}/{net}.{sta}.{channel[:-1]} at {origin}"
                if missing == "error":
                    raise ValueError(f"no orientation epoch for {label}")
                status = "missing_zero" if missing == "zero" else "missing_skip"
                if missing == "zero":
                    warnings.warn(f"no orientation epoch for {label}; using 0 degrees", RuntimeWarning)
            row = dict(event_id=event, network=net, station=sta, sensor=channel[:-1],
                       location=location, origin_utc=str(origin), status=status,
                       theta_deg=epoch.theta if epoch else 0.0,
                       flip_n=epoch.flip_n if epoch else False, flip_e=epoch.flip_e if epoch else False,
                       swap_ne=epoch.swap_ne if epoch else False,
                       epoch_start=str(epoch.start) if epoch else "",
                       epoch_end_exclusive=str(epoch.end) if epoch else "")
            if status == "missing_skip" or not os.path.isfile(ef):
                if not os.path.isfile(ef):
                    row["status"] = "no_e_component"
                for f in (rf, tf):
                    Path(f).unlink(missing_ok=True)
                records.append(row)
                continue
            token = _digest([signature, srcsig])
            old = previous.get("traces", {}).get(nf)
            if old and old.get("signature") == token and Path(rf).is_file() and Path(tf).is_file():
                row.update(baz_deg=old["baz_deg"], rotation_deg=old["rotation_deg"])
            else:
                e = read(ef)[0]
                radial, transverse, baz, ba = rotate_horizontal(
                    n, e, row["theta_deg"], reference,
                    **{k: row[k] for k in ("flip_n", "flip_e", "swap_ne")})
                for tr, path in ((radial, rf), (transverse, tf)):
                    tmp = path + f".tmp{os.getpid()}"
                    tr.write(tmp, format="SAC")
                    os.replace(tmp, path)
                row.update(baz_deg=baz, rotation_deg=ba)
            row["signature"], row["source_n"] = token, nf
            keep_outputs.update((rf, tf))
            records.append(row)
        # Missing/removed N/E must not leave an old T available to correlation.
        for comp in ("R", "T"):
            for f in glob(os.path.join(common, event, f"{event}.*.*.*{comp}.sac")):
                if f not in keep_outputs:
                    Path(f).unlink()
        event_signatures[event] = _digest(event_inputs)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(dest / "rotation.csv", index=False)
    manifest = dict(settings=settings, signature=signature, event_signatures=event_signatures,
                    traces={r["source_n"]: r for r in records if "source_n" in r})
    _atomic_json(previous_path, manifest)
    print(f"[rotation] {sum('source_n' in r for r in records)} sensors rotated; provenance {dest}")
    return manifest


def precomputed_signature(common, events, components=("T",)):
    """Track externally prepared T inputs without rewriting their samples or metadata.

    Legacy callers selecting T alone keep their exported components. Explicit
    ``rotate_for_xcorr`` or orientation settings request the derived path instead.
    """
    sources = {}
    for event in events:
        files = sorted({f for comp in {"Z", *components}
                        for f in glob(os.path.join(common, event, f"{event}.*.*.*{comp}.sac"))})
        sources[event] = _digest(_source_signature(files))
    return dict(signature=_digest(dict(version=_VERSION, mode="precomputed")),
                event_signatures=sources, precomputed=True)


def sync_pair_signature(out, xc, overrides, rotation=None):
    """Invalidate incompatible pair checkpoints before resume, then record the new inputs.

    Untracked legacy N/E and precomputed-T runs remain reusable. First derived T
    and later component/table/reference/repair changes invalidate prior pair files.
    """
    path = Path(out) / "xcorr_inputs.json"
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError):
        old = None
    signature = _digest(dict(xcorr=xc, overrides=repr(overrides),
                             rotation=rotation["signature"] if rotation else None))
    sources = rotation["event_signatures"] if rotation else {}
    invalidate_all = ((old is not None and old.get("signature") != signature) or
                      (old is None and rotation is not None and not rotation.get("precomputed")))
    changed = {e for e, s in sources.items() if old and old.get("events", {}).get(e) != s}
    n = 0
    if invalidate_all or changed:
        for sub in ("dt.cc_P", "dt.cc_S"):
            for p in (Path(out) / sub).glob(f"{sub}_*"):
                pair = p.name[len(sub) + 1:].split("_")
                if invalidate_all or set(pair) & changed:
                    p.unlink()
                    n += 1
        if n:
            print(f"[xcorr] invalidated {n} pair checkpoints after input changes")
    _atomic_json(path, dict(signature=signature, events=sources))
    return n
