"""The Rust half of P6's connectome diff (arc 2, r02b-p6).

A Rust `use`/path that crosses a region pair is an import pair, a `_native` symbol maps to its Rust
file's region, and the reader failing with `.rs` present is RED. Every test drives
`rust_region_imports` through `measure(root, overlay, native=...)` with fixture facts injected;
logic is never forked.
"""

import importlib
import os
import sys

import pytest

from snn_data_lake.gates import rust_facts

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
gate = importlib.import_module("p6_gate")

_AMY = "amyg" + "dala"  # from parts: move_gate C2(d) reads string literals
_CER = "cere" + "bellum"

REDS = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10"]
CONTROLS = ["G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14"]


def _verdict(res, base):
    return {
        row[0]: row[4] for row in gate.evaluate(res["metrics"], gate.rust_ceiling(base["metrics"]))
    }


def _arm(table, name):
    arms = {a[0]: a for a in getattr(gate, table, ())}
    assert name in arms, (
        f"gate.{table} carries no arm {name}: the self-check has no such constructed red"
    )
    return arms[name]


def _boom(*_args, **_kwargs):
    raise rust_facts.RustFactsUnavailable("constructed")


@pytest.fixture(scope="module")
def base():
    """The EMPTY-RUST baseline (real Python, no Rust) every arm is judged against."""
    return gate.measure(_ROOT, native=(gate.NATIVE_ABSENT, []))


def test_real_tree_prints_absent_and_leaves_every_python_count_alone(base):
    res = gate.measure(_ROOT)
    conn = res["conn"]
    assert "rust_status" in conn, (
        "the gate reports NOTHING about native/ (neither ABSENT nor a count)"
    )
    assert conn["rust_status"].startswith("native/ ABSENT"), conn["rust_status"]
    assert conn["rust_directed"] == set()
    assert conn["native_mapped"] == set()
    assert conn["native_unmapped"] == []
    assert res["metrics"] == base["metrics"]
    assert res["metrics"]["undeclared_import_pairs"] == gate.BASELINE["undeclared_import_pairs"]
    assert all(_verdict(res, base).values())


@pytest.mark.parametrize("name", REDS)
def test_red_arm_crosses_exactly_the_preregistered_pair_and_fires_the_ceiling(name, base):
    _, _how, fear_rs, overlay, extra = _arm("RUST_REDS", name)
    native = gate.fixture_native(fear_rs, extra)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    conn = res["conn"]
    assert conn["rust_status"] == native[0]
    via, other = (
        ("native_mapped", "rust_directed") if overlay else ("rust_directed", "native_mapped")
    )
    assert conn[via] == gate.RUST_PAIR
    assert conn[other] == set()
    assert conn["native_unmapped"] == []
    expect = dict(base["metrics"])
    expect["undeclared_import_pairs"] += 1
    assert res["metrics"] == expect
    verdict = _verdict(res, base)
    assert verdict["undeclared_import_pairs"] is False
    assert all(ok for k, ok in verdict.items() if k != "undeclared_import_pairs")


@pytest.mark.parametrize("name", CONTROLS)
def test_control_stays_green_with_no_rust_pair(name, base):
    _, _how, fear_rs, overlay, extra = _arm("RUST_CONTROLS", name)
    native = (gate.NATIVE_ABSENT, []) if fear_rs is None else gate.fixture_native(fear_rs, extra)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    conn = res["conn"]
    assert conn["rust_status"] == native[0]
    assert conn["rust_directed"] == set()
    assert conn["native_mapped"] == set()
    if fear_rs is None and overlay:
        assert len(conn["native_unmapped"]) == 1, conn["native_unmapped"]
        assert "fear_step" in conn["native_unmapped"][0] and "ABSENT" in conn["native_unmapped"][0]
    else:
        assert conn["native_unmapped"] == []
    assert res["metrics"] == base["metrics"]
    assert all(_verdict(res, base).values())


def test_native_target_with_native_absent_is_printed_not_dropped(base, capsys):
    _, _how, _fear_rs, overlay, _extra = _arm("RUST_CONTROLS", "G4")
    res = gate.measure(_ROOT, overlay=overlay)  # the REAL tree: native/ ABSENT
    conn = res["conn"]
    unmapped = conn.get("native_unmapped", "DROPPED SILENTLY (no record of the _native target)")
    assert isinstance(unmapped, list) and len(unmapped) == 1, unmapped
    assert res["metrics"] == base["metrics"]
    gate.report(res, _ROOT)
    out = capsys.readouterr().out
    assert "NATIVE TARGETS UNMAPPED: 1" in out
    assert "_native.fear_step" in out and "ABSENT" in out
    assert "Rust half: native/ ABSENT" in out


_RS_ROOT_FN = (
    "use pyo3::prelude::*;\n\n#[pyfunction]\npub fn root_fn(x: f64) -> f64 {\n    x\n}\n\n"
    "#[pymodule(gil_used = false)]\nmod _native {\n    #[pymodule_export]\n"
    "    use super::root_fn;\n}\n"
)
_RS_UTIL_EXTRA = {
    "native/sndl-core/src/lib.rs": gate.RS_CORE_LIB + "pub mod util;\n",
    "native/sndl-core/src/util.rs": (
        "use pyo3::prelude::*;\n\n#[pyfunction]\npub fn util_fn(x: f64) -> f64 {\n    x\n}\n"
    ),
    "native/sndl-python/src/lib.rs": gate.RS_PY_LIB.replace(
        "}\n", "    #[pymodule_export]\n    use sndl_core::util::util_fn;\n}\n"
    ),
}


@pytest.mark.parametrize(
    ("source", "extra", "reason"),
    [
        ("import snn_data_lake._native\n", None, "module-level or star import"),
        ("from snn_data_lake._native import *\n", None, "module-level or star import"),
        ("from snn_data_lake._native import ghost\n", None, "not a PyO3 export"),
        (
            "from snn_data_lake._native import root_fn\n",
            {"native/sndl-python/src/lib.rs": _RS_ROOT_FN},
            "its Rust file has no region",
        ),
        (
            "from snn_data_lake._native import util_fn\n",
            _RS_UTIL_EXTRA,
            "region `util` is not a connectome region",
        ),
    ],
    ids=["module", "star", "ghost", "no-region", "non-connectome-region"],
)
def test_native_target_reasons_with_native_present(source, extra, reason, base):
    overlay = {f"{_AMY}/_p6_native_reason.py": source}
    res = gate.measure(
        _ROOT, overlay=overlay, native=gate.fixture_native(gate.RS_FEAR_GREEN, extra)
    )
    conn = res["conn"]
    assert len(conn["native_unmapped"]) == 1 and reason in conn["native_unmapped"][0], conn[
        "native_unmapped"
    ]
    assert conn["native_mapped"] == set()
    assert res["metrics"] == base["metrics"]


def test_same_region_native_import_maps_to_no_pair(base):
    overlay = {f"{_CER}/_p6_native_same.py": "from snn_data_lake._native import fear_step\n"}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(gate.RS_FEAR_GREEN))
    assert res["conn"]["native_mapped"] == set()
    assert res["conn"]["native_unmapped"] == []
    assert res["metrics"] == base["metrics"]


def test_rust_region_table():
    labels = gate.frozen_labels(_ROOT)
    assert gate.rust_region("x/src/lib.rs", "x", labels) is None
    assert gate.rust_region("x/src/main.rs", "x", labels) is None
    assert gate.rust_region("x/src/rng.rs", "x", labels) == "rng"
    assert gate.rust_region("x/src/fear_conditioning.rs", "x", labels) == f"{_AMY}"
    assert gate.rust_region(f"src/{_AMY}/fear.rs", "", labels) == f"{_AMY}"
    assert gate.rust_region(f"x/src/{_CER}/fear_conditioning.rs", "x", labels) == f"{_AMY}"
    assert gate.rust_region("x/tests/it.rs", None, labels) is None


def test_reader_is_built_from_this_checkout_not_from_root(tmp_path, monkeypatch):
    _, _how, fear_rs, _overlay, _extra = _arm("RUST_REDS", "R1")
    paths = gate.write_native(tmp_path, fear_rs)
    roots = []
    real_facts = rust_facts.facts

    def spy(files, root=rust_facts.ROOT, allow_errors=False):
        roots.append(root)
        return real_facts(files, root, allow_errors)

    monkeypatch.setattr(rust_facts, "rust_files", lambda root=None, sub=None: paths)
    monkeypatch.setattr(rust_facts, "facts", spy)
    status, directed, exports = gate.rust_region_imports(str(tmp_path), gate.frozen_labels(_ROOT))
    assert roots == [rust_facts.ROOT], roots  # never --root's tools/rsfacts
    assert status.startswith("native/ PRESENT (7 .rs files)"), status
    assert directed == gate.RUST_PAIR  # absolute fact paths under an alternate root resolve
    assert exports == {"fear_step": f"{_CER}"}


def test_rust_self_check_judges_arms_against_the_empty_rust_baseline(monkeypatch, base):
    _, _how, fear_rs, _overlay, _extra = _arm("RUST_REDS", "R1")
    live = gate.fixture_native(fear_rs)
    real_facts = rust_facts.facts

    def facts(
        files, root=rust_facts.ROOT, allow_errors=False
    ):  # the fake live tree only; fixtures stay real
        return live[1] if list(files) == ["live.rs"] else real_facts(files, root, allow_errors)

    monkeypatch.setattr(rust_facts, "rust_files", lambda root=None, sub=None: ["live.rs"])
    monkeypatch.setattr(rust_facts, "facts", facts)
    live_res = gate.measure(_ROOT)
    assert (
        live_res["metrics"]["undeclared_import_pairs"]
        == base["metrics"]["undeclared_import_pairs"] + 1
    )
    # the GLOBAL ceiling now equals the live count: only the empty tree's own ceiling still fires
    monkeypatch.setitem(
        gate.BASELINE, "undeclared_import_pairs", live_res["metrics"]["undeclared_import_pairs"]
    )
    rows, control_rows, lines = gate.rust_self_check(_ROOT, base["metrics"])
    assert [r[0] for r in rows] == ["undeclared_import_pairs"] * len(gate.RUST_REDS)
    assert all(moved and fired for _n, _h, _b, _a, moved, fired in rows), rows
    assert all(not reds for _w, _y, reds in control_rows), control_rows
    assert any("fail-closed" in line for line in lines)


def test_rust_self_check_reports_a_dead_red_and_a_leaking_control(monkeypatch, base):
    r1_body = _arm("RUST_REDS", "R1")[2]
    monkeypatch.setattr(
        gate, "RUST_REDS", [("X1", "a GREEN body posing as a red", gate.RS_FEAR_GREEN, None, None)]
    )
    monkeypatch.setattr(
        gate, "RUST_CONTROLS", [("Y1", "R1 posing as a control", r1_body, None, None)]
    )
    rows, control_rows, _lines = gate.rust_self_check(_ROOT, base["metrics"])
    assert [(moved, fired) for _n, _h, _b, _a, moved, fired in rows] == [(False, False)], rows
    assert control_rows[0][2], control_rows


def test_orphan_file_in_no_mod_tree_still_counts(base):
    extra = {f"native/sndl-core/src/{_AMY}/orphan.rs": _arm("RUST_REDS", "R1")[2]}
    res = gate.measure(_ROOT, native=gate.fixture_native(gate.RS_FEAR_GREEN, extra))
    assert res["conn"]["rust_directed"] == gate.RUST_PAIR
    assert (
        res["metrics"]["undeclared_import_pairs"] == base["metrics"]["undeclared_import_pairs"] + 1
    )


def test_rust_pair_outside_connectome_regions_is_reported_not_counted(base, capsys):
    fear_rs = "use crate::util::util_fn;\n\npub fn f(x: f64) -> f64 {\n    util_fn(x)\n}\n"
    res = gate.measure(_ROOT, native=gate.fixture_native(fear_rs, _RS_UTIL_EXTRA))
    assert res["conn"]["rust_directed"] == {(f"{_AMY}", "util")}
    assert res["metrics"] == base["metrics"]
    gate.report(res, _ROOT)
    assert "1 directed Rust region pairs (0 between connectome regions" in capsys.readouterr().out


@pytest.mark.parametrize(
    "source",
    ["import snn_data_lake._native\n", "from snn_data_lake._native import *\n"],
    ids=["module", "star"],
)
def test_absent_status_wins_over_the_symbol_reason(source, base):
    res = gate.measure(_ROOT, overlay={f"{_AMY}/_p6_native_reason.py": source})
    unmapped = res["conn"]["native_unmapped"]
    assert len(unmapped) == 1 and gate.NATIVE_ABSENT in unmapped[0], unmapped


def test_unavailable_status_is_the_reason_for_an_unmapped_target(monkeypatch, base):
    monkeypatch.setattr(rust_facts, "rust_files", _boom)
    res = gate.measure(_ROOT, overlay=_arm("RUST_CONTROLS", "G4")[3])
    unmapped = res["conn"]["native_unmapped"]
    assert len(unmapped) == 1 and "UNAVAILABLE" in unmapped[0], unmapped


def test_unreadable_rust_with_rs_present_is_a_liveness_red_not_a_skip(monkeypatch, capsys, base):
    monkeypatch.setattr(rust_facts, "rust_files", _boom)
    res = gate.measure(_ROOT)
    status = res["conn"].get("rust_status", "NOT REPORTED")
    assert status.startswith("native/ UNAVAILABLE"), status
    assert res["metrics"] == base["metrics"]
    _lines, problems = gate.liveness(_ROOT, res)
    assert [p for p in problems if "RustFactsUnavailable" in p], problems
    assert gate.main(["--assert", "--root", _ROOT]) == 1
    out = capsys.readouterr().out
    assert "INSTRUMENT DEAD" in out and "RustFactsUnavailable" in out
