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
        assert float(z["duration_s"]) == 2.0 and z["duration_s"].ndim == 0
        assert str(z["subject"]) == "HO1" and str(z["session"]) == "20250101T000000"
        assert str(z["source_sha256"]) == e["sha256"]
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
    keys = "npz subject session split source_sha256 units duration_s duplicate_of".split()
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
