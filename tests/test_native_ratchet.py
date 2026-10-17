"""Guardrails for `tools/native_ratchet.py` (R0.3 meter): per-library importers may only fall.

Each test builds the thing the gate must catch (a rise, a new library, an unreadable Rust tree) and
runs the gate's own comparator / entry point on it.

Run: pytest tests/test_native_ratchet.py
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import native_ratchet as nr  # noqa: E402

from snn_data_lake.gates import rust_facts  # noqa: E402

BASE = {"numpy": 3, "zlib": 1, "struct": 0}
HITS = {
    "numpy": [("a.py", 1, False), ("a.py", 2, True), ("b.py", 1, False)],
    "zlib": [("a.py", 3, False)],
}


def test_census_counts_distinct_files_and_union():
    zones, union = nr.census(imported=HITS)
    assert {k: len(v) for k, v in zones.items()}["numpy"] == 2
    assert zones["scipy"] == []
    assert union == ["a.py", "b.py"] and nr.counts(zones)["zlib"] == 1


def test_union_excludes_stdlib_only_files():
    _, union = nr.census(imported={"zlib": [("z.py", 1, False)], "numpy": [("n.py", 1, False)]})
    assert union == ["n.py"]


def test_rise_is_red_and_fall_is_green():
    assert nr.compare(dict(BASE, numpy=4), BASE) == [("numpy", 4, 3)]
    assert nr.compare(dict(BASE, numpy=2), BASE) == []


def test_library_missing_from_baseline_is_red():
    assert nr.compare(dict(BASE, torch=1), BASE) == [("torch", 1, None)]


def test_update_baseline_refuses_a_rise(tmp_path, monkeypatch):
    path = tmp_path / "b.json"
    held = [5] * len(nr.LIBS)
    path.write_text(json.dumps(" ".join(map(str, held))), encoding="utf-8")
    monkeypatch.setattr(nr, "BASELINE_PATH", path)
    monkeypatch.setattr(nr, "census", lambda **_: ({k: ["f"] * 6 for k in nr.LIBS}, []))
    assert nr.main(["--update-baseline"]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == " ".join(map(str, held))


def test_real_tree_native_absent_line(capsys):
    assert nr.native_line(rust_facts.ROOT).startswith("native/ ABSENT")


def test_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch, capsys):
    (tmp_path / "native").mkdir()
    (tmp_path / "native" / "lib.rs").write_text("fn x() {}\n", encoding="utf-8")

    def boom(*a, **k):
        raise rust_facts.RustFactsUnavailable("no cargo")

    monkeypatch.setattr(rust_facts, "facts", boom)
    monkeypatch.setattr(nr, "NATIVE_ROOT", tmp_path)
    with pytest.raises(rust_facts.RustFactsUnavailable):
        nr.native_line(tmp_path)
    assert nr.main(["--assert"]) == 1


def test_self_check_passes():
    assert nr.self_check() == 0
