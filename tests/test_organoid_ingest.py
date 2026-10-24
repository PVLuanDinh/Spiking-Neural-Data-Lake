"""OG1 Stage 0 gates, each with a reachable red on a small synthetic HDF5 fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from snn_data_lake.sensorimotor import organoid_ingest as oi
from snn_data_lake.sensorimotor.organoid_ingest import IngestRefused

FREEZE = Path(r"D:\AI-project\lanes\organoid-og1-1023\og1_dandi_assets_freeze.json")


@pytest.fixture
def good(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "good.nwb")
    e = oi.fixture_entry(p)
    return p, e, oi.fixture_pin(e)


def test_green_control_converts_and_round_trips(good, tmp_path):
    p, e, pin = good
    r = oi.ingest_file(p, e, pin, tmp_path / "out")
    with np.load(tmp_path / "out" / oi.artifact_name(e), allow_pickle=False) as z:
        assert z["spike_times"].dtype == np.float64 and z["spike_times"].size == 6
        assert z["spike_times_index"].dtype == np.int64 and list(z["spike_times_index"]) == [
            3,
            5,
            6,
        ]
        assert float(z["duration_s"]) == 2.0005 and z["duration_s"].ndim == 0
        assert str(z["subject"]) == "HO1" and str(z["session"]) == "20250101T000000"
        assert str(z["source_sha256"]) == e["sha256"]
        assert z["x_pos"].dtype == np.float64 and list(z["x_pos"]) == [0.0, 100.0, 200.0]
        assert z["y_pos"].dtype == np.float64 and list(z["y_pos"]) == [0.0, 50.0, 100.0]
    assert r["devices"] == "none recorded in the file" and r["e_reproduce"]["mismatch_cells"] == 0


def test_a_license_empty_is_refused():
    with pytest.raises(IngestRefused, match=r"\(a\)"):
        oi.gate_license({"license": ""})
    with pytest.raises(IngestRefused, match=r"\(a\)"):
        oi.gate_license({"license": "spdx:Proprietary"})
    assert oi.gate_license({"license": "spdx:CC-BY-4.0"}) == "cc-by-4.0"


def test_b_flipped_byte_is_refused_before_h5py_opens(good, tmp_path, monkeypatch):
    p, e, pin = good
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 1
    bad = tmp_path / "flipped.nwb"
    bad.write_bytes(bytes(raw))

    def boom(path):
        raise AssertionError("h5py opened a file whose bytes were never verified")

    monkeypatch.setattr(oi, "_open_h5", boom)
    with pytest.raises(IngestRefused, match=r"\(b\)"):
        oi.ingest_file(bad, e, pin, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_b_wrong_size_is_refused(good):
    p, e, _ = good
    with pytest.raises(IngestRefused, match=r"\(b\)"):
        oi.gate_bytes(p, dict(e, size=e["size"] + 1))


def test_c_truncated_index_is_refused(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "c.nwb", index=[3, 5, 5])
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=r"\(c\).*truncated"):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")


def test_c_decreasing_and_empty_index_are_refused(tmp_path):
    spikes = np.arange(6.0)
    with pytest.raises(IngestRefused, match=r"decreases"):
        oi.gate_index(spikes, np.array([4, 2, 6]))
    with pytest.raises(IngestRefused, match=r"zero units"):
        oi.gate_index(np.array([]), np.array([], dtype=np.int64))


def test_c_uint16_wrap_reads_as_int64_then_refuses(tmp_path):
    import h5py

    p = oi.write_fixture_nwb(tmp_path / "w.nwb")
    with h5py.File(str(p), "a") as f:
        del f["units/spike_times_index"]
        f.create_dataset(
            "units/spike_times_index", data=np.array([3, 5, 70000 % 65536], dtype=np.uint16)
        )
    with h5py.File(str(p), "r") as f:
        spikes, idx = oi.read_units(f)
    assert idx.dtype == np.int64
    with pytest.raises(IngestRefused, match=r"\(c\)"):
        oi.gate_index(spikes, idx)


def test_d_times_x1000_is_refused(tmp_path):
    p = oi.write_fixture_nwb(
        tmp_path / "d.nwb", units=[[100.0, 500.0, 1200.0], [300.0, 1900.0], [700.0]]
    )
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=r"\(d\).*milliseconds"):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")


def test_d_samples_at_20khz_is_refused():
    with pytest.raises(IngestRefused, match=r"samples_20khz"):
        oi.gate_time_units(np.array([40000.0 * 3.0]), 3.0)


def test_d_no_recorded_span_is_refused(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "n.nwb", span=False)
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=r"no recording span"):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")


def test_refusal_carries_readings_and_writes_no_artifact(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "d.nwb", units=[[100.0, 1900.0]])
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused) as ei:
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    assert ei.value.readings["n_units"] == 1 and "source_sha256" in ei.value.readings
    assert not list((tmp_path / "out").glob("*.npz")) if (tmp_path / "out").exists() else True


def test_pin_shape_and_freeze_check():
    pin = oi.load_pin()
    a = pin["assets"]
    assert pin["dandiset"] == "001603" and pin["version"] == "0.260923.0623" and len(a) == 24
    assert pin["license"] == "spdx:CC-BY-4.0" and pin["citation"].startswith(
        "van der Molen, Tjitse"
    )
    assert {x["subject"] for x in a if x["split"] == "dev"} == {"HO1", "HO2", "HO3", "HO4"}
    assert {x["subject"] for x in a if x["split"] == "held-out"} == {"HO5", "HO6", "HO7", "HO8"}
    assert all("_ecephys" not in x["path"] and len(x["sha256"]) == 64 for x in a)
    assert len({oi.artifact_name(x) for x in a}) == 24
    assert oi.gate_license(pin) == "cc-by-4.0"


def test_pin_differing_from_freeze_is_refused():
    pin = oi.load_pin()
    freeze = [{"path": x["path"], "MB": x["size"] / 1e6, "sha": x["sha256"]} for x in pin["assets"]]
    assert oi.check_pin_against_freeze(pin, freeze) == 24
    freeze[3] = dict(freeze[3], sha="0" * 64)
    with pytest.raises(IngestRefused, match=r"pin != freeze"):
        oi.check_pin_against_freeze(pin, freeze)
    with pytest.raises(IngestRefused, match=r"pin != freeze"):
        oi.check_pin_against_freeze(pin, freeze[:-1])


@pytest.mark.skipif(not FREEZE.exists(), reason="freeze-time census lives outside the repo")
def test_pin_equals_the_real_freeze_census():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8-sig"))
    assert oi.check_pin_against_freeze(oi.load_pin(), freeze) == 24


def test_selfcheck_passes():
    assert oi._selfcheck() == 0


def _true(units=oi._FIXTURE_UNITS, bins=2000):
    fl = np.array([t for u in units for t in u])
    ix = np.cumsum([len(u) for u in units])
    return fl, ix, oi.rebuild_raster(fl, ix, bins, 1000.0).astype("u1")


def test_e_exact_reproduction_and_extremum_agreements():
    fl, ix, r = _true()
    out = oi.gate_reproduce(fl, ix, r, 0.0, 1000.0)
    a = out["agreement"]
    assert a["declared"] == 1.0 and out["mismatch_cells"] == 0
    for k in ("floor", "round", "floor_plus_half", "shift_plus1", "shift_minus1"):
        assert a[k] < 1.0, k


def test_e_plain_floor_raster_is_refused():
    """A1.1's old declaration: a raster built with plain floor no longer reproduces."""
    fl, ix, _ = _true()
    plain = oi.rebuild_raster(fl, ix, 2000, 1000.0, 0.0, "floor").astype("u1")
    with pytest.raises(IngestRefused, match=r"\(e'\) rebuilt raster != stored"):
        oi.gate_reproduce(fl, ix, plain, 0.0, 1000.0)


def test_e_rows_below_zero_are_dropped_and_row_zero_is_kept():
    spikes = np.array([0.0004, 0.0006])  # x = 0.4 -> row -1 (dropped); x = 0.6 -> row 0
    r = oi.rebuild_raster(spikes, np.array([1, 2]), 5, 1000.0)
    assert r.sum() == 1 and r[0, 1] and not r[:, 0].any()


def _t0_fixture(tmp_path, t0, pre=0):
    """The engine-arc panel's A-M4 probe as a fixture: 3 units after t0, plus `pre` spikes
    before t0."""
    rng = np.random.default_rng(0)
    units = [sorted(rng.uniform(max(t0, 0.0) + 0.002, max(t0, 0.0) + 1.6, 40)) for _ in range(3)]
    if pre:
        units[0] = sorted(list(rng.uniform(0.0, t0 - 0.01, pre)) + list(units[0]))
    p = oi.write_fixture_nwb(tmp_path / f"t0_{t0}.nwb", units=units, t0=t0)
    e = oi.fixture_entry(p)
    return p, e, oi.fixture_pin(e)


def test_d_span_starting_after_zero_is_refused(tmp_path):
    # A-M4: t0 > 0 with spikes in [0, t0) passed (c), (d) and (e') (rebuild_raster drops their
    # negative rows) and wrote the pre-recording spikes into the npz. Refused until (d) reads
    # t - t0.
    p, e, pin = _t0_fixture(tmp_path, 0.25, pre=25)
    with pytest.raises(IngestRefused, match="starts at t0"):
        oi.ingest_file(p, e, pin, tmp_path / "out")
    assert not (tmp_path / "out" / oi.artifact_name(e)).exists()


def test_d_span_starting_before_zero_is_refused(tmp_path):
    p, e, pin = _t0_fixture(tmp_path, -0.25)
    with pytest.raises(IngestRefused, match="starts at t0"):
        oi.ingest_file(p, e, pin, tmp_path / "out")


def test_d_span_starting_at_zero_still_converts(tmp_path):
    p, e, pin = _t0_fixture(tmp_path, 0.0)
    rows = oi.ingest_file(p, e, pin, tmp_path / "out")
    assert rows["span_start_s"] == 0.0 and rows["artifact"] == oi.artifact_name(e)


def test_e_times_x1000_is_refused():
    fl, ix, r = _true()
    with pytest.raises(IngestRefused, match=r"\(e'\) rebuilt raster != stored"):
        oi.gate_reproduce(fl * 1000, ix, r, 0.0, 1000.0)


def test_e_unit_index_off_by_one_is_refused():
    fl, ix, r = _true()
    with pytest.raises(IngestRefused, match=r"\(e'\)"):
        oi.gate_reproduce(fl, ix, np.roll(r, 1, axis=1), 0.0, 1000.0)


def test_e_single_bin_flip_is_refused_and_reports_the_cell():
    fl, ix, r = _true()
    r[777, 1] ^= 1
    with pytest.raises(IngestRefused, match=r"\(777, 1\)") as ei:
        oi.gate_reproduce(fl, ix, r, 0.0, 1000.0)
    assert ei.value.readings["mismatch_cells"] == 1 and ei.value.readings["mismatch_fraction"] > 0


def test_e_round_convention_stored_raster_is_refused():
    fl, ix, _ = _true()
    rnd = oi.rebuild_raster(fl, ix, 2000, 1000.0, 0.0, "round").astype("u1")
    with pytest.raises(IngestRefused, match=r"\(e'\)"):
        oi.gate_reproduce(fl, ix, rnd, 0.0, 1000.0)


def test_e_extremum_rule_refuses_when_the_data_cannot_tell_floor_from_declared():
    same = [[0.1006], [0.3006], [0.7006]]  # every remainder >= 0.5: floor == declared
    fl, ix, r = _true(same)
    with pytest.raises(IngestRefused, match=r"not decided by the data"):
        oi.gate_reproduce(fl, ix, r, 0.0, 1000.0)


def test_e_through_ingest_file_refuses_a_flipped_stored_raster(tmp_path):
    _, _, r = _true()
    r[1, 0] ^= 1
    p = oi.write_fixture_nwb(tmp_path / "f.nwb", raster=r)
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=r"\(e'\)") as ei:
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    assert not (tmp_path / "out").exists() and ei.value.readings["n_units"] == 3


def _rows(tmp_path, specs):
    rows = []
    for sess, units in specs:
        q = oi.write_fixture_nwb(tmp_path / f"{sess}.nwb", units=units)
        e = oi.fixture_entry(q, name=f"sub-HO1/sub-HO1_ses-{sess}.nwb")
        rows.append({"outcome": "ok", **oi.ingest_file(q, e, oi.fixture_pin(e), tmp_path / "o")})
    return rows


PIN = {"assets": [{"subject": "HO1", "split": "dev"}]}


def test_duplicates_identical_arrays_keep_the_lexicographically_first(tmp_path):
    u = oi._FIXTURE_UNITS
    rows = _rows(tmp_path, [("20250101T000009", u), ("20250101T000001", u)])
    man = oi.session_manifest(tmp_path / "o", rows, PIN)
    assert [(m["session"], m["duplicate_of"]) for m in man] == [
        ("20250101T000001", None),
        ("20250101T000009", "20250101T000001"),
    ]
    saved = json.loads((tmp_path / "o" / "og1_sessions.json").read_text(encoding="utf-8"))
    keys = (
        "npz subject session split source_sha256 units duration_s duplicate_of "
        "dup_timestamps out_of_order"
    ).split()
    assert set(saved[0]) == set(keys)
    assert saved[0]["split"] == "dev" and saved[0]["units"] == 3


def test_one_spike_moved_by_1e9_is_not_a_duplicate(tmp_path):
    u = oi._FIXTURE_UNITS
    moved = [
        [t + (1e-9 if (i, j) == (0, 0) else 0.0) for j, t in enumerate(v)] for i, v in enumerate(u)
    ]
    man = oi.session_manifest(
        tmp_path / "o", _rows(tmp_path, [("20250101T000001", u), ("20250101T000002", moved)]), PIN
    )
    assert [m["duplicate_of"] for m in man] == [None, None]


def test_equal_counts_but_different_spikes_are_not_duplicates(tmp_path):
    a = [[0.1006, 0.5004], [0.3009]]
    b = [[0.2006, 0.6004], [0.4009]]
    man = oi.session_manifest(
        tmp_path / "o", _rows(tmp_path, [("20250101T000001", a), ("20250101T000002", b)]), PIN
    )
    assert [m["duplicate_of"] for m in man] == [None, None]


def _refused(tmp_path, name, match, **kw):
    p = oi.write_fixture_nwb(tmp_path / name, **kw)
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=match):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_c_nan_spike_time_is_refused(tmp_path):
    _refused(tmp_path, "nan.nwb", r"\(c\).*not finite", units=[[0.1006, float("nan")], [0.3009]])


def test_c_infinite_spike_time_is_refused(tmp_path):
    _refused(tmp_path, "inf.nwb", r"\(c\).*not finite", units=[[0.1006, float("inf")], [0.3009]])


def test_c_negative_spike_time_is_refused(tmp_path):
    _refused(tmp_path, "neg.nwb", r"\(c\).*negative \(-0\.4", units=[[-0.4, 0.5004], [0.3009]])


def test_d_spikes_past_the_recorded_span_are_refused(tmp_path):
    _refused(
        tmp_path,
        "late.nwb",
        r"\(d\).*at or past the recorded span 1\.0005 s",
        units=[[0.1006, 1.9003], [0.3009]],
        bins=1000,
    )


def test_d_a_spike_exactly_at_the_span_is_refused_and_just_inside_is_not():
    with pytest.raises(IngestRefused, match=r"at or past"):
        oi.gate_time_units(np.array([0.5, 2.0]), 2.0)
    assert oi.gate_time_units(np.array([0.5, 1.999]), 2.0)["last_spike"] == 1.999


def _drop(path, key):
    import h5py

    with h5py.File(str(path), "a") as f:
        del f["units/" + key]


@pytest.mark.parametrize("key", ["x_pos", "y_pos"])
def test_c_units_table_without_positions_is_refused(tmp_path, key):
    p = oi.write_fixture_nwb(tmp_path / "np.nwb")
    _drop(p, key)
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=rf"\(c\).*no {key}"):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_c_positions_wrong_length_or_nan_are_refused(tmp_path):
    import h5py

    p = oi.write_fixture_nwb(tmp_path / "short.nwb")
    _drop(p, "x_pos")
    with h5py.File(str(p), "a") as f:
        f.create_dataset("units/x_pos", data=np.array([1.0, 2.0]))
    e = oi.fixture_entry(p)
    with pytest.raises(IngestRefused, match=r"x_pos has shape"):
        oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    q = oi.write_fixture_nwb(tmp_path / "nanpos.nwb")
    _drop(q, "y_pos")
    with h5py.File(str(q), "a") as f:
        f.create_dataset("units/y_pos", data=np.array([1.0, np.nan, 3.0]))
    e = oi.fixture_entry(q)
    with pytest.raises(IngestRefused, match=r"y_pos holds a non-finite"):
        oi.ingest_file(q, e, oi.fixture_pin(e), tmp_path / "out")


def test_positions_are_the_units_table_values_in_nwb_order(tmp_path):
    import h5py

    p = oi.write_fixture_nwb(tmp_path / "pos.nwb")
    _drop(p, "x_pos")
    with h5py.File(str(p), "a") as f:
        f.create_dataset("units/x_pos", data=np.array([700.5, 122.0, 3798.0]))
    e = oi.fixture_entry(p)
    oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    with np.load(tmp_path / "out" / oi.artifact_name(e), allow_pickle=False) as z:
        assert list(z["x_pos"]) == [700.5, 122.0, 3798.0] and list(z["y_pos"]) == [0.0, 50.0, 100.0]


def test_order_counts_skip_the_unit_boundary():
    st = np.array([1.0, 1.0, 0.5, 2.0, 0.1])  # unit 0 = first four, unit 1 = [0.1]
    assert oi.order_counts(st, np.array([4, 5])) == (1, 1)
    assert oi.order_counts(np.array([5.0, 1.0]), np.array([1, 2])) == (0, 0)
    assert oi.order_counts(np.array([1.0]), np.array([1])) == (0, 0)


def test_manifest_records_planted_duplicates_and_out_of_order(tmp_path):
    planted = [[0.1006, 0.1006, 0.7002, 0.5004], [0.3009, 1.9003]]
    clean = oi._FIXTURE_UNITS
    rows = _rows(tmp_path, [("20250101T000001", planted), ("20250101T000002", clean)])
    man = oi.session_manifest(tmp_path / "o", rows, PIN)
    assert [(m["dup_timestamps"], m["out_of_order"]) for m in man] == [(1, 1), (0, 0)]
    saved = json.loads((tmp_path / "o" / "og1_sessions.json").read_text(encoding="utf-8"))
    assert [(m["dup_timestamps"], m["out_of_order"]) for m in saved] == [(1, 1), (0, 0)]


def test_d_span_is_the_a21_raster_edge_n_rows_plus_half_over_rate(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "s.nwb", bins=1000)
    import h5py

    with h5py.File(str(p), "r") as f:
        assert oi.recorded_span(f)[0] == 1.0005


def test_d_spike_at_exactly_the_raster_edge_is_refused(tmp_path):
    _refused(
        tmp_path,
        "edge.nwb",
        r"\(d\).*at or past the recorded span 1\.0005 s",
        units=[[0.1006, 1.0005], [0.3009]],
        bins=1000,
    )


def test_d_spike_just_below_the_raster_edge_is_accepted(tmp_path):
    p = oi.write_fixture_nwb(tmp_path / "in.nwb", units=[[0.1006, 1.0004], [0.3009]], bins=1000)
    e = oi.fixture_entry(p)
    r = oi.ingest_file(p, e, oi.fixture_pin(e), tmp_path / "out")
    assert r["duration_s"] == 1.0005 and r["d_time_units"]["last_spike"] == 1.0004
