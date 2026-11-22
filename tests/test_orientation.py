"""Offline scientific regressions for orientation-corrected S differential times."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from obspy import Trace, UTCDateTime, read
from obspy.core import AttribDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core import orientation, xcorr


def table(tmp_path, rows):
    path = tmp_path / "angles.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def row(**kw):
    return dict(network="KS", station="TEST", sensor="HH", start="2020-01-01",
                end="2020-12-31", theta_deg=0.0, **kw)


def trace(data, comp="N", origin="2020-01-01T00:00:00", start=-2.0):
    tr = Trace(np.asarray(data, dtype=np.float32))
    tr.stats.update(dict(network="KS", station="TEST", channel="HH" + comp,
                         sampling_rate=100.0, starttime=UTCDateTime(origin) + start))
    tr.stats.sac = AttribDict(b=start, a=0.5, t0=1.0, stla=0., stlo=0., evla=1., evlo=0.,
                             cmpaz=0 if comp == "N" else 90., cmpinc=90.)
    return tr


def write_event(common, eid, theta=0, shift=0):
    folder = Path(common) / eid
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(601) / 100 - 2
    pulse = np.exp(-((t - 1.0 - shift) / 0.10) ** 2) * np.cos(2 * np.pi * 8 * (t - 1.0 - shift))
    # baz=0: T=N*sin(-theta)-E*cos(-theta), with R=0.
    a = np.deg2rad(-theta)
    nd, ed = pulse * np.sin(a), -pulse * np.cos(a)
    for comp, data in (("N", nd), ("E", ed), ("Z", pulse)):
        tr = trace(data, comp, eid)
        tr.write(str(folder / f"{eid}.KS.TEST.HH{comp}.sac"), format="SAC")
    return pulse


def cfg(path=None, missing="error", reference=None):
    return SimpleNamespace(orientation_csv=str(path) if path else None,
                           orientation_missing=missing, rotation_reference=reference)


def test_epochs_network_sensor_dates_timestamps_and_alias(tmp_path):
    entries = [row(), {**row(), "network": "KG", "theta_deg": 90},
               {**row(), "sensor": "HG", "theta_deg": 45},
               {**row(), "start": "2021-01-01T00:00:00Z", "end": "2021-01-01T01:00:00Z", "theta_deg": -12}]
    p = table(tmp_path, entries)
    t = orientation.OrientationTable(p)
    assert t.lookup("KS", "TEST", "HH", "2020-12-31T23:59:59.999999Z").theta == 0
    assert t.lookup("KS", "TEST", "HH", "2021-01-01T00:00:00Z").theta == -12
    assert t.lookup("KS", "TEST", "HH", "2021-01-01T01:00:00Z") is None
    assert t.lookup("KG", "TEST", "HH", "2020-01-01").theta == 90
    assert t.lookup("KS", "TEST", "HG", "2020-01-01").theta == 45
    assert t.lookup("KS", "TEST", "HH", "2019-12-31") is None
    alias = pd.read_csv(p).rename(columns={"theta_deg": "theta_M_apply", "network": "net", "station": "sta"})
    alias.to_csv(p, index=False)
    assert orientation.OrientationTable(p).lookup("KG", "TEST", "HH", "2020-01-01").theta == 90


@pytest.mark.parametrize("changes", [dict(theta_deg=float("inf")), dict(theta_deg="bad"),
                                     dict(start=""), dict(end="2019-12-31"), dict(flip_n="perhaps")])
def test_malformed_epochs_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        orientation.OrientationTable(table(tmp_path, [{**row(), **changes}]))


def test_overlaps_including_location_wildcards_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlapping"):
        orientation.OrientationTable(table(tmp_path, [row(), {**row(), "location": "00"}]))
    p = table(tmp_path, [{**row(), "location": "00"}, {**row(), "location": "10", "theta_deg": 20}])
    assert orientation.OrientationTable(p).lookup("KS", "TEST", "HH", "2020-01-01", "10").theta == 20


def test_rotation_repairs_trim_preserves_absolute_picks_and_headers(tmp_path):
    n = trace(np.arange(100), "N")
    e = trace(-np.arange(100), "E")
    e.trim(e.stats.starttime + 0.1, e.stats.endtime)
    e.stats.sac.b = -1.9
    n0, e0 = n.data.copy(), e.data.copy()
    r, t, baz, ba = orientation.rotate_horizontal(n, e, 0, flip_e=True)
    assert (baz, ba) == (0, 0)
    np.testing.assert_allclose(r.data, -np.arange(10, 100))
    np.testing.assert_allclose(t.data, -np.arange(10, 100))
    assert abs(float(r.stats.starttime - r.stats.sac.b - (n.stats.starttime - n.stats.sac.b))) < 1e-6
    assert r.stats.sac.a == n.stats.sac.a and r.stats.sac.t0 == n.stats.sac.t0
    assert r.stats.sac.cmpaz == 180 and t.stats.sac.cmpaz == 270
    assert t.stats.sac.cmpinc == 90 and t.stats.channel == "HHT"
    np.testing.assert_array_equal(n.data, n0)
    np.testing.assert_array_equal(e.data, e0)
    p = tmp_path / "rotated.sac"
    r.write(str(p), format="SAC")
    reread = read(str(p))[0]
    assert abs(float(reread.stats.starttime - reread.stats.sac.b - (n.stats.starttime - n.stats.sac.b))) < 1e-5
    assert abs(float(reread.stats.starttime + reread.stats.sac.t0 - reread.stats.sac.b -
                     (n.stats.starttime + n.stats.sac.t0 - n.stats.sac.b))) < 1e-5


def test_swap_then_flip_order():
    n, e = trace([1, 2, 3]), trace([4, 5, 6], "E")
    r, t, _, _ = orientation.rotate_horizontal(n, e, 0, swap_ne=True, flip_n=True)
    np.testing.assert_array_equal(r.data, [4, 5, 6])
    np.testing.assert_array_equal(t.data, [-1, -2, -3])


def test_fixed_source_reference_and_nonzero_angle_convention():
    n, e = trace([1, 1, 1]), trace([0, 0, 0], "E")
    r, t, baz, ba = orientation.rotate_horizontal(n, e, 30, reference=(0, 1))
    assert baz == 90 and ba == 60
    np.testing.assert_allclose(r.data, -0.5)
    np.testing.assert_allclose(t.data, np.sqrt(3) / 2)


@pytest.mark.parametrize("fault", ["rate", "grid", "reference", "pick", "coordinates", "location", "sensor", "network"])
def test_incompatible_horizontals_rejected(fault):
    n, e = trace(np.ones(200)), trace(np.ones(200), "E")
    if fault == "rate":
        e.stats.sampling_rate = 50
    elif fault == "grid":
        e.stats.starttime += 0.005
    elif fault == "reference":
        e.stats.sac.b += 1
    elif fault == "pick":
        e.stats.sac.t0 += 1
    elif fault == "coordinates":
        n.stats.sac.stla = -12345
    elif fault == "location":
        e.stats.location = "10"
    elif fault == "sensor":
        e.stats.channel = "HGE"
    else:
        e.stats.network = "KG"
    with pytest.raises(ValueError):
        orientation.rotate_horizontal(n, e, 0)


def test_missing_policies_and_stale_t_removed(tmp_path):
    common, out = tmp_path / "waveforms", tmp_path / "dtcc"
    eid = "20200101000000"
    write_event(common, eid)
    p = table(tmp_path, [{**row(), "network": "KG"}])
    with pytest.raises(ValueError, match="no orientation"):
        orientation.prepare_rotation(cfg(p), common, [eid], out)
    with pytest.warns(RuntimeWarning, match="using 0"):
        orientation.prepare_rotation(cfg(p, "zero"), common, [eid], out)
    tf = common / eid / f"{eid}.KS.TEST.HHT.sac"
    assert tf.is_file()
    orientation.prepare_rotation(cfg(p, "skip"), common, [eid], out)
    assert not tf.exists()
    orientation.prepare_rotation(cfg(), common, [eid], out)
    assert tf.is_file()
    (common / eid / f"{eid}.KS.TEST.HHN.sac").unlink()
    orientation.prepare_rotation(cfg(), common, [eid], out)
    assert not tf.exists()


def test_multiple_sensors_rejected(tmp_path):
    common = tmp_path / "wf"
    eid = "20200101000000"
    write_event(common, eid)
    (common / eid / f"{eid}.KS.TEST.HGN.sac").write_bytes((common / eid / f"{eid}.KS.TEST.HHN.sac").read_bytes())
    with pytest.raises(ValueError, match="multiple"):
        orientation.prepare_rotation(cfg(), common, [eid], tmp_path / "out")


def test_n_only_station_does_not_need_orientation_coverage(tmp_path):
    common, out = tmp_path / "wf", tmp_path / "out"
    eid = "20200101000000"
    write_event(common, eid)
    (common / eid / f"{eid}.KS.TEST.HHE.sac").unlink()
    p = table(tmp_path, [{**row(), "network": "KG"}])
    orientation.prepare_rotation(cfg(p), common, [eid], out)
    records = pd.read_csv(out / "orientation" / "rotation.csv")
    assert records.status.tolist() == ["no_e_component"]
    assert not list((common / eid).glob("*T.sac"))


def test_subsecond_interpolation_cache_signature_changes(tmp_path):
    p = tmp_path / "trace.sac"
    p.write_bytes(b"one")
    xcorr._init_worker(None, None, None, {"interp_hz": 100}, None, None, None,
                       interp_cache_dir=str(tmp_path))
    before = xcorr._interp_cache_path(str(p), 5, 20)
    st = p.stat()
    import os
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 100))
    assert before != xcorr._interp_cache_path(str(p), 5, 20)


def test_reintroduced_event_invalidates_old_pairs(tmp_path):
    out = tmp_path / "out"
    (out / "dt.cc_S").mkdir(parents=True)
    xc = {"s_comps": ("T",)}
    rotation = dict(signature="settings", event_signatures={"e1": "original", "e2": "same"})
    orientation.sync_pair_signature(out, xc, {}, rotation)
    pair = out / "dt.cc_S" / "dt.cc_S_e1_e2"
    pair.write_text("old pair")
    orientation.sync_pair_signature(out, xc, {}, dict(signature="settings", event_signatures={"e2": "same"}))
    assert pair.exists()
    rotation["event_signatures"]["e1"] = "changed"
    assert orientation.sync_pair_signature(out, xc, {}, rotation) == 1
    assert not pair.exists()


def test_untracked_legacy_precomputed_pairs_remain_reusable(tmp_path):
    common, out = tmp_path / "wf", tmp_path / "out"
    events = ["e1", "e2"]
    for eid in events:
        folder = common / eid
        folder.mkdir(parents=True)
        for comp in ("Z", "T"):
            (folder / f"{eid}.KS.TEST.HH{comp}.sac").write_bytes(b"source")
    (out / "dt.cc_S").mkdir(parents=True)
    pair = out / "dt.cc_S" / "dt.cc_S_e1_e2"
    pair.write_text("existing precomputed T pair")
    rotation = orientation.precomputed_signature(common, events)
    assert orientation.sync_pair_signature(out, {"s_comps": ("T",)}, {}, rotation) == 0
    assert pair.exists()
    (common / "e1" / "e1.KS.TEST.HHT.sac").write_bytes(b"changed transverse source")
    rotation = orientation.precomputed_signature(common, events)
    assert orientation.sync_pair_signature(out, {"s_comps": ("T",)}, {}, rotation) == 1
    assert not pair.exists()


class SerialExecutor:
    """Use real production worker functions deterministically in this small fixture."""
    def __init__(self, *, initializer, initargs, **kw):
        initializer(*initargs)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
    def map(self, fn, items):
        return map(fn, items)


def test_xcorr_t_recovers_coherence_across_epoch_change_and_invalidates_resume(tmp_path, monkeypatch):
    common, out = tmp_path / "waveforms", tmp_path / "dtcc"
    events = ["20200523000000", "20200524000000"]
    # A 180 degree sensor reversal makes both raw N/E correlations negative.
    for eid, theta in zip(events, (20, 200)):
        write_event(common, eid, theta)
    p = table(tmp_path, [{**row(), "end": "2020-05-23", "theta_deg": 20},
                         {**row(), "start": "2020-05-24", "theta_deg": 200}])
    c = cfg(p)
    c.xcorr = dict(interp_hz=100, bandpass=(2, 20), pre=0.2, post=0.3,
                   margin=0.011, slide_step=0.01, shift_samp=2, cc_threshold=0.95,
                   s_comps=("N", "E"))
    c.xcorr_pair_overrides, c.num_cores, c.mainshock_event_id = {}, 1, None
    monkeypatch.setattr(xcorr.config, "dtcc_dir", lambda cfg: str(out))
    monkeypatch.setattr(xcorr.config, "assert_writable", lambda p: p)
    monkeypatch.setattr(xcorr.config, "waveforms_dir", lambda cfg: str(common))
    monkeypatch.setattr(xcorr.config, "sum_file", lambda cfg, vm: "unused")
    monkeypatch.setattr(xcorr.sumio, "read_sum", lambda p: pd.DataFrame({"id": [1, 2]}))
    from pipeline.core import evmap
    monkeypatch.setattr(evmap, "dir_of_cuspid", lambda cfg: {1: events[0], 2: events[1]})
    monkeypatch.setattr(xcorr, "ProcessPoolExecutor", SerialExecutor)
    xcorr.run_xcorr(c)
    assert " S\n" not in (out / "dt.cc_S_0.7").read_text()
    c.xcorr["s_comps"] = ("T",)
    result = xcorr.run_xcorr(c, resume=True)
    assert result["pairs"] == 1  # the old N/E pair was discarded
    lines = (out / "dt.cc_S_0.7").read_text().splitlines()
    assert len(lines) == 2 and lines[1].endswith("S")
    assert float(lines[1].split()[2]) > 0.999
    assert abs(float(lines[1].split()[1])) <= 0.011
    before = (common / events[0] / f"{events[0]}.KS.TEST.HHT.sac").stat().st_mtime_ns
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 0
    assert before == (common / events[0] / f"{events[0]}.KS.TEST.HHT.sac").stat().st_mtime_ns
    # Z changes must invalidate P checkpoints even when only T is derived.
    zpath = common / events[0] / f"{events[0]}.KS.TEST.HHZ.sac"
    z = read(str(zpath))[0]
    z.stats.sac.a = 0.6
    z.write(str(zpath), format="SAC")
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
    # Historical callers may already have corrected T: T-only configuration
    # consumes it verbatim even though the underlying N/E need correction.
    c.orientation_csv = None
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
    assert before == (common / events[0] / f"{events[0]}.KS.TEST.HHT.sac").stat().st_mtime_ns
    assert " S\n" in (out / "dt.cc_S_0.7").read_text()
    # Explicit derivation without a table means nominal north, rather than
    # consuming the externally prepared components.
    c.rotate_for_xcorr = True
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
    assert " S\n" not in (out / "dt.cc_S_0.7").read_text()
    c.orientation_csv = str(p)
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
    assert " S\n" in (out / "dt.cc_S_0.7").read_text()
    # Changing only table content must also discard completed T checkpoints.
    df = pd.read_csv(p)
    df.loc[1, "theta_deg"] = 20
    df.to_csv(p, index=False)
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
    assert " S\n" not in (out / "dt.cc_S_0.7").read_text()
    # Returning to the legacy component selection also cannot reuse old T files.
    c.xcorr["s_comps"] = ("N", "E")
    assert xcorr.run_xcorr(c, resume=True)["pairs"] == 1
