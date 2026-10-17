"""Guardrails for `gates/native_ratchet.py` (R0.3 meter): per-library importers may only fall.

Each test builds the thing the gate must catch (a rise, a new library, an unreadable Rust tree) and
runs the gate's own comparator / entry point on it.

Run: pytest tests/test_native_ratchet.py
"""

import json

import pytest

from snn_data_lake.gates import native_ratchet as nr
from snn_data_lake.gates import rust_facts

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


P = "src/snn_data_lake/"
PKG_HITS = {
    "numpy": [(P + "sleep/a.py", 1, False), (P + "sleep/b.py", 1, False), (P + "top.py", 1, False)],
    "torch": [(P + "sleep/a.py", 2, False)],
}


def test_live_counts_adds_union_and_per_package_keys():
    live = nr.live_counts(*nr.census(imported=PKG_HITS))
    assert live["numpy"] == 3 and live["union"] == 3
    assert live["sleep:numpy"] == 2 and live["sleep:torch"] == 1 and live["sleep:union"] == 2
    assert live["(root):numpy"] == 1 and "(root):zlib" in live


def test_one_package_rises_while_library_total_does_not():
    base = nr.live_counts(*nr.census(imported=PKG_HITS))
    moved = {
        "numpy": [
            (P + "sleep/a.py", 1, False),
            (P + "sleep/b.py", 1, False),
            (P + "sleep/c.py", 1, False),
        ],
        "torch": [(P + "sleep/a.py", 2, False)],
    }
    live = nr.live_counts(*nr.census(imported=moved))
    assert live["numpy"] == base["numpy"] == 3
    assert ("sleep:numpy", 3, 2) in nr.compare(live, base)


def test_new_package_with_imports_is_red_but_empty_one_is_not():
    base = {"numpy": 1, "sleep:numpy": 1}
    assert nr.compare({"numpy": 1, "sleep:numpy": 1, "lake:numpy": 0}, base) == []
    assert nr.compare({"numpy": 1, "sleep:numpy": 1, "lake:numpy": 1}, base) == [
        ("lake:numpy", 1, None)
    ]


def test_baseline_string_round_trips():
    live = nr.live_counts(*nr.census(imported=PKG_HITS))
    assert nr.decode(nr.encode(live)) == live and " " not in nr.encode(live)


def test_component_view_is_labelled_provisional():
    live = nr.live_counts(*nr.census(imported=PKG_HITS))
    view = "\n".join(nr.component_view(live))
    assert "provisional until G8 boundaries settle" in view and "sleep" in view


def test_rise_is_red_and_fall_is_green():
    assert nr.compare(dict(BASE, numpy=4), BASE) == [("numpy", 4, 3)]
    assert nr.compare(dict(BASE, numpy=2), BASE) == []


def test_library_missing_from_baseline_is_red():
    assert nr.compare(dict(BASE, torch=1), BASE) == [("torch", 1, None)]


def test_update_baseline_refuses_a_rise(tmp_path, monkeypatch):
    path = tmp_path / "b.json"
    held = {k: 5 for k in nr.COLS}
    path.write_text(json.dumps(nr.encode(held)), encoding="utf-8")
    monkeypatch.setattr(nr, "BASELINE_PATH", path)
    monkeypatch.setattr(nr, "census", lambda **_: ({k: ["f"] * 6 for k in nr.LIBS}, ["f"] * 6))
    assert nr.main(["--update-baseline"]) == 1
    assert nr.decode(json.loads(path.read_text(encoding="utf-8"))) == held


def test_native_absent_line(tmp_path):
    assert nr.native_line(tmp_path).startswith("native/ ABSENT")


def test_decode_rejects_a_row_of_the_wrong_width():
    with pytest.raises(ValueError):
        nr.decode("ALL=1,2,3")


def test_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch):
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


def test_empty_research_module_named_like_a_library_does_not_hide_a_src_importer():
    with nr._fixture_repo(shadows=("numpy", "zlib", "struct")):
        zones, union = nr.census()
    assert {k: v for k, v in nr.counts(zones).items() if v} == {"numpy": 1, "zlib": 1, "struct": 1}
    assert union == ["src/snn_data_lake/p/new.py"]


def test_control_names_every_library_a_dead_walk_reads_as_zero():
    with nr._fixture_repo() as src:
        dead = nr.control({lib: [] for lib in nr.LIBS}, src)
        assert nr.control(*nr.census()[:1], src) == []
    assert dead == [("numpy", 0, 1), ("zlib", 0, 1), ("struct", 0, 1)]


def test_dead_walk_is_red_even_though_every_count_reads_zero(tmp_path, monkeypatch, capsys):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(nr.encode({k: 5 for k in nr.COLS})), encoding="utf-8")
    monkeypatch.setattr(nr, "BASELINE_PATH", path)
    monkeypatch.setattr(nr, "census", lambda **_: ({k: [] for k in nr.LIBS}, []))
    with nr._fixture_repo() as src:
        monkeypatch.setattr(nr, "ROOT", src.parent)
        assert nr.main(["--assert"]) == 1
    out = capsys.readouterr().out
    assert "numpy: walk 0 != 1 line" in out and "gate OK" not in out


def test_the_tokenizer_control_reads_inline_imports_and_ignores_strings(tmp_path):
    """Codex r3 F-3: `if True: import numpy` and `x = 1; import zlib` are imports; `import struct`
    inside a docstring is not. The control must agree with the ast walk on valid code."""
    src = tmp_path / "src"
    (src / "snn_data_lake").mkdir(parents=True)
    (src / "snn_data_lake" / "m.py").write_text(
        '"""import struct is prose here."""\nif True: import numpy\nx = 1; import zlib\n',
        encoding="utf-8",
    )
    counts = nr.line_counts(src)
    assert (counts["numpy"], counts["zlib"], counts["struct"]) == (1, 1, 0), counts


def _rigged(monkeypatch, tmp_path, files, numpy_files, baseline_numpy=None):
    """A src tree under tmp_path, its census rigged to `numpy_files`; the baseline equals the live
    counts unless `baseline_numpy` overrides numpy."""
    for rel, text in files.items():
        p = tmp_path / "src" / "snn_data_lake" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    zones = {lib: (list(numpy_files) if lib == "numpy" else []) for lib in nr.LIBS}
    union = sorted(numpy_files)
    monkeypatch.setattr(nr, "ROOT", tmp_path)
    monkeypatch.setattr(nr, "census", lambda imported=None: (zones, union))
    live = nr.live_counts(zones, union)
    held = dict(live, numpy=baseline_numpy) if baseline_numpy is not None else live
    monkeypatch.setattr(nr, "_load_baseline", lambda: held)


def test_an_unreadable_src_file_fails_closed(tmp_path, monkeypatch):
    """Codex r4 N-4: a syntax-error file is skipped by the ast walk AND the tokenizer, so the
    control agrees with the omission; the gate must red, not read the lost importers as a fall."""
    _rigged(monkeypatch, tmp_path,
            {"m.py": "import numpy\n", "broken.py": "import numpy\nx = (\n"},
            ["src/snn_data_lake/m.py"])  # fmt: skip
    assert nr.main(["--assert"]) == 1


def test_numpy_zero_under_a_nonzero_baseline_is_an_instrument_failure(tmp_path, monkeypatch):
    """PREREG: 'a count of 0 for numpy is an instrument failure' (walk and control both dead)."""
    _rigged(monkeypatch, tmp_path, {"m.py": "x = 1\n"}, [], baseline_numpy=97)
    assert nr.main(["--assert"]) == 1
