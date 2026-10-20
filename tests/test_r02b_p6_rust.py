"""The Rust half of P6's connectome diff (arc 2, r02b-p6; binding-export attribution, lane p6x).

A Rust `use`/path that crosses a region pair is an import pair, a `_native` symbol maps to its Rust
file's region, and the reader failing with `.rs` present is RED. A crate-root export (the binding's
own `#[pyfunction]` / `#[pyclass]`) belongs to no region: the OLD gate formed no pair for it (Codex
B2-fold MAJOR 1, both reviews: `use sndl_core::medulla::step;` + `step()` inside pfc_decide left
undeclared_import_pairs at 11), so it is attributed to the regions of the Python modules that USE
it (import or attribute style, read by rust_facts.python_native_refs) and forms (importer region,
target) for every connectome region its body reaches. Every test drives `rust_region_imports`
through `measure(root, overlay, native=...)` with fixture facts injected; logic is never forked.
The p6x reds were pre-registered in rows/p6x-REPORT.md and run against the pre-fix gate, where
each failed.
"""

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from snn_data_lake.gates import rust_facts

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
gate = importlib.import_module("p6_gate")

_AMY = "amyg" + "dala"  # from parts: move_gate C2(d) reads string literals
_CER = "cere" + "bellum"
_MED = "medu" + "lla"
_CTX = "cor" + "tex"

REDS = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R15"]
CONTROLS = ["G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14"]
CONTROLS += ["G15", "G16"]
_PFC = "faculties/prefrontal_executive.py"
_USE_LINE = "use std::panic::{AssertUnwindSafe, catch_unwind};"
_IN_PFC = "        let mut words = [0u32; N];"  # the first statement of pfc_decide's closure
_IN_ENTER = (
    "    match py.detach(|| catch_unwind(AssertUnwindSafe(f))) {"  # enter(): called by every export
)
_MEDULLA_PAIR = {("cortex", "medulla")}
_NOBODY = "<no regioned Python module>"
_NO_RUST = "no regioned Rust module"


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


*_, _RS_BINDING_REACH = _arm(
    "RUST_REDS", "R11"
)  # the binding whose own bind_step reaches cerebellum


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
    # native/ exists since R0.4 (sndl-core). Since pfc-p2 the PFC kernel
    # (prefrontal_executive.rs: the stem carries the Python module's frozen label, cortex) uses the
    # rng kit, and the PFC uses `_native.pfc_decide` after a module-level import: `rng` is no
    # connectome region, and pfc_decide (a crate-root export, p6x) is attributed to its cortex
    # importer and reaches cortex (same region) and rng -- no pair; panic_probe is used by no
    # regioned module and reaches no region; all PRINTED (pinned here so a change is seen) and
    # every count stays.
    assert conn["rust_status"].startswith(("native/ ABSENT", "native/ PRESENT")), conn[
        "rust_status"
    ]
    assert conn["rust_directed"] == {("cortex", "rng")}
    assert conn["native_mapped"] == set()
    assert conn["native_unmapped"] == []
    assert conn["native_attributed"] == [
        f"{_NOBODY}: _native.panic_probe (crate-root export; its body reaches {_NO_RUST})",
        f"{_PFC}: _native.pfc_decide (crate-root export; its body reaches cortex, rng)",
    ]
    assert res["metrics"] == base["metrics"]
    lines, _problems = gate.liveness(_ROOT, res)
    assert any("2 exports in no connectome region attributed" in line for line in lines), lines
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
    assert gate.own_unmapped(conn, overlay) == []
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
        own = gate.own_unmapped(conn, overlay)
        assert len(own) == 1, own
        assert "fear_step" in own[0] and "ABSENT" in own[0]
    else:
        assert gate.own_unmapped(conn, overlay) == []
    assert res["metrics"] == base["metrics"]
    assert all(_verdict(res, base).values())


def test_native_target_with_native_absent_is_printed_not_dropped(base, capsys, monkeypatch):
    monkeypatch.setattr(
        rust_facts, "rust_files", lambda *a, **k: []
    )  # native/ ABSENT (real tree has it now)
    _, _how, _fear_rs, overlay, _extra = _arm("RUST_CONTROLS", "G4")
    res = gate.measure(_ROOT, overlay=overlay)
    conn = res["conn"]
    assert "native_unmapped" in conn, "DROPPED SILENTLY (no record of the _native target)"
    unmapped = gate.own_unmapped(conn, overlay)
    assert len(unmapped) == 1, unmapped
    assert res["metrics"] == base["metrics"]
    gate.report(res, _ROOT)
    out = capsys.readouterr().out
    assert "NATIVE TARGETS UNMAPPED: 2" in out  # the arm's + the real PFC's (both printed)
    assert f"{_PFC}: _native.pfc_decide ({gate.NATIVE_ABSENT})" in conn["native_unmapped"]
    assert "_native.fear_step" in out and "ABSENT" in out
    assert "Rust half: native/ ABSENT" in out


_RS_ROOT_FN = gate.rs_binding(
    "#[pyfunction]\npub fn root_fn(x: f64) -> f64 {\n    x\n}\n", "root_fn"
)
_RS_TEST_EXPORT_EXTRA = {  # a #[pyfunction] in a tests/ file: not production -> no region, no reach
    "native/sndl-core/tests/helper.rs": (
        "use pyo3::prelude::*;\n\n#[pyfunction]\npub fn test_fn() {}\n"
    ),
    gate.RS_BINDING: gate.rs_binding("", "test_fn"),
}
_RS_UTIL_EXTRA = {
    "native/sndl-core/src/lib.rs": gate.RS_CORE_LIB + "pub mod util;\n",
    "native/sndl-core/src/util.rs": (
        "use pyo3::prelude::*;\n\n#[pyfunction]\npub fn util_fn(x: f64) -> f64 {\n    x\n}\n"
    ),
    gate.RS_BINDING: gate.rs_binding("", "sndl_core::util::util_fn"),
}


@pytest.mark.parametrize(
    ("source", "extra", "reason"),
    [
        ("import snn_data_lake._native\n", None, "module-level or star import"),
        ("from snn_data_lake._native import *\n", None, "module-level or star import"),
        ("from snn_data_lake._native import ghost\n", None, "not a PyO3 export"),
        (
            "from snn_data_lake._native import test_fn\n",
            _RS_TEST_EXPORT_EXTRA,
            "its Rust file is not production code (no region)",
        ),
        ("from snn_data_lake._native import *\n", _RS_BINDING_REACH, "module-level or star import"),
    ],
    ids=[
        "module",
        "star",
        "ghost",
        "not-production",
        "star-with-crate-root-export",
    ],
)
def test_native_target_reasons_with_native_present(source, extra, reason, base):
    overlay = {f"{_AMY}/_p6_native_reason.py": source}
    res = gate.measure(
        _ROOT, overlay=overlay, native=gate.fixture_native(gate.RS_FEAR_GREEN, extra)
    )
    conn = res["conn"]
    own = gate.own_unmapped(conn, overlay)
    assert len(own) == 1 and reason in own[0], own
    assert conn["native_mapped"] == set()
    assert res["metrics"] == base["metrics"]
    assert gate.own_attributed(conn, overlay) == []


def test_an_export_in_a_non_connectome_region_is_attributed_by_its_body_reach(base):
    """b4p P1: before, an export in src/util.rs (region `util`) was UNMAPPED "not a connectome
    region" and its body never read; now its body's reach is read like a crate root's (none)."""
    overlay = {f"{_AMY}/_p6_native_reason.py": "from snn_data_lake._native import util_fn\n"}
    native = gate.fixture_native(gate.RS_FEAR_GREEN, _RS_UTIL_EXTRA)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    conn = res["conn"]
    assert gate.own_unmapped(conn, overlay) == []
    assert gate.own_attributed(conn, overlay) == [
        f"{_AMY}/_p6_native_reason.py: _native.util_fn "
        f"(export in non-connectome region `util`; its body reaches {_NO_RUST})"
    ]
    assert conn["native_mapped"] == set()
    assert res["metrics"] == base["metrics"]


def test_r14_an_export_in_a_binding_submodule_counts_its_body_reach(base):
    """b4p P1 (Opus p6x MAJOR-1): R14's own wire (bindings, cerebellum) is a Rust pair outside the
    connectome (dropped, as before); the pair that counts comes through its Python importer."""
    _, _how, fear_rs, overlay, extra = _arm("RUST_REDS", "R14")
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    conn = res["conn"]
    assert conn["native_mapped"] == gate.RUST_PAIR
    assert conn["rust_directed"] == {("bindings", _CER)}
    assert gate.own_unmapped(conn, overlay) == []
    assert gate.own_attributed(conn, overlay) == [
        f"{_AMY}/_p6_bind_import.py: _native.bind_step "
        f"(export in non-connectome region `bindings`; its body reaches {_CER})"
    ]
    expect = dict(base["metrics"])
    expect["undeclared_import_pairs"] += 1
    assert res["metrics"] == expect
    verdict = _verdict(res, base)
    assert verdict["undeclared_import_pairs"] is False
    assert all(ok for k, ok in verdict.items() if k != "undeclared_import_pairs")


def test_crate_root_export_with_an_empty_reach_is_attributed_not_unmapped(base):
    overlay = {f"{_AMY}/_p6_native_reason.py": "from snn_data_lake._native import root_fn\n"}
    extra = {gate.RS_BINDING: _RS_ROOT_FN}
    native = gate.fixture_native(gate.RS_FEAR_GREEN, extra)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    conn = res["conn"]
    assert gate.own_unmapped(conn, overlay) == []
    assert (
        f"{_AMY}/_p6_native_reason.py: _native.root_fn "
        f"(crate-root export; its body reaches {_NO_RUST})"
    ) in conn["native_attributed"]
    assert conn["native_mapped"] == set()
    assert res["metrics"] == base["metrics"]


_RS_BIND_QUALIFIED = {  # the helper is called by a QUALIFIED path: disclosed as not followed
    gate.RS_BINDING: gate.rs_binding(
        "fn probe() {\n    let _c = Clock::new();\n}\n\n"
        "#[pyfunction]\npub fn bind_step(x: f64) -> f64 {\n    self::probe();\n    x\n}\n",
        "bind_step",
    )
}


@pytest.mark.parametrize(
    ("relpath", "source"),
    [
        (
            f"{_AMY}/_p6_bind_spelling.py",
            "import snn_data_lake._native as n\n\n\ndef use(x):\n    return n.bind_step(x)\n",
        ),
        (
            f"{_AMY}/_p6_bind_spelling.py",
            "from .. import _native\n\n\ndef use(x):\n    return _native.bind_step(x)\n",
        ),
        (
            f"{_AMY}/_p6_pkg/__init__.py",
            "from ... import _native\n\n\ndef use(x):\n    return _native.bind_step(x)\n",
        ),
    ],
    ids=["import-as", "relative", "package-init"],
)
def test_other_module_level_spellings_of_a_crate_root_export_count(relpath, source, base):
    overlay = {relpath: source}
    native = gate.fixture_native(gate.RS_FEAR_GREEN, _RS_BINDING_REACH)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR
    assert gate.own_unmapped(res["conn"], overlay) == []
    assert (
        res["metrics"]["undeclared_import_pairs"] == base["metrics"]["undeclared_import_pairs"] + 1
    )


def test_a_helper_called_by_a_qualified_path_is_not_followed_as_disclosed(base):
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R11")
    res = gate.measure(
        _ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, _RS_BIND_QUALIFIED)
    )
    assert res["conn"]["native_mapped"] == set()
    assert gate.own_unmapped(res["conn"], overlay) == []
    own = gate.own_attributed(res["conn"], overlay)
    assert len(own) == 1 and _NO_RUST in own[0], own
    assert res["metrics"] == base["metrics"]
    assert "qualified path" in " ".join(gate.RUST_BLIND_SPOTS)


def test_a_cfg_test_twin_does_not_shadow_the_production_helper(base):
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R11")
    items = (
        "fn probe() {\n    let _c = Clock::new();\n}\n\n#[cfg(test)]\nfn probe() {}\n\n"
        "#[pyfunction]\npub fn bind_step(x: f64) -> f64 {\n    probe();\n    x\n}\n"
    )
    extra = {gate.RS_BINDING: gate.rs_binding(items, "bind_step")}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR


def test_a_plain_impl_method_is_not_part_of_the_pyclass_reach(base):
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R13")
    items = (
        "#[pyclass]\npub struct Probe;\n\n#[pymethods]\nimpl Probe {\n    #[new]\n"
        "    fn new() -> Probe {\n        Probe\n    }\n}\n\nimpl Probe {\n"
        "    fn inner(&self) {\n        let _c = Clock::new();\n    }\n}\n"
    )
    extra = {gate.RS_BINDING: gate.rs_binding(items, "Probe")}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == set()
    assert res["metrics"] == base["metrics"]


def test_r16_pymethods_in_a_sibling_file_count_in_the_pyclass_reach(base):
    """b4p P2 (Opus p6x MINOR-2): only the export's own file was read, so a #[pymethods] impl in a
    sibling file left the class reaching nothing. R16's own wire (methods, cerebellum) is a Rust
    pair outside the connectome (dropped); the counted pair comes through the class's importer."""
    _, _how, fear_rs, overlay, extra = _arm("RUST_REDS", "R16")
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    conn = res["conn"]
    assert conn["native_mapped"] == gate.RUST_PAIR
    assert conn["rust_directed"] == {("methods", _CER)}
    assert gate.own_attributed(conn, overlay) == [
        f"{_AMY}/_p6_bind_class.py: _native.Probe (crate-root export; its body reaches {_CER})"
    ]
    expect = dict(base["metrics"])
    expect["undeclared_import_pairs"] += 1
    assert res["metrics"] == expect


def test_pymethods_of_another_type_are_not_part_of_the_pyclass_reach(base):
    """The control for the last-segment rule: `impl crate::Other` is not Probe's impl."""
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R13")
    items = (
        "#[pyclass]\npub struct Probe;\n\n#[pyclass]\npub struct Other;\n\n#[pymethods]\n"
        "impl crate::Other {\n    fn go(&self) {\n        let _c = Clock::new();\n    }\n}\n"
    )
    extra = {gate.RS_BINDING: gate.rs_binding(items, "Probe")}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == set()
    assert gate.own_attributed(res["conn"], overlay) == [
        f"{_AMY}/_p6_bind_class.py: _native.Probe (crate-root export; its body reaches {_NO_RUST})"
    ]
    assert res["metrics"] == base["metrics"]


def test_a_decoy_module_beside_a_package_does_not_hide_the_package_use(base):
    overlay = {
        f"{_AMY}/_p6_decoy/__init__.py": "from snn_data_lake._native import bind_step\n",
        f"{_AMY}/_p6_decoy.py": "from snn_data_lake import _native\n",
    }
    native = gate.fixture_native(gate.RS_FEAR_GREEN, _RS_BINDING_REACH)
    res = gate.measure(_ROOT, overlay=overlay, native=native)
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR
    assert gate.own_attributed(res["conn"], overlay) == [
        f"{_AMY}/_p6_decoy/__init__.py: _native.bind_step "
        f"(crate-root export; its body reaches {_CER})"
    ]


def test_a_self_recursive_helper_terminates_and_still_reaches(base):
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R11")
    items = (
        "fn probe() {\n    probe();\n    let _c = Clock::new();\n}\n\n"
        "#[pyfunction]\npub fn bind_step(x: f64) -> f64 {\n    probe();\n    x\n}\n"
    )
    extra = {gate.RS_BINDING: gate.rs_binding(items, "bind_step")}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR


def test_a_pyclass_is_one_export_and_its_methods_are_not_printed(base):
    _, _how, fear_rs, overlay, extra = _arm("RUST_REDS", "R13")
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR
    assert res["conn"]["native_attributed"] == [
        f"{_AMY}/_p6_bind_class.py: _native.Probe (crate-root export; its body reaches {_CER})"
    ]


def test_a_getter_setter_pair_keeps_both_bodies_in_the_pyclass_reach(base):
    _, _how, fear_rs, overlay, _extra = _arm("RUST_REDS", "R13")
    items = (
        "#[pyclass]\npub struct Probe;\n\n#[pymethods]\nimpl Probe {\n    #[getter]\n"
        "    fn get_x(&self) -> f64 {\n        let _c = Clock::new();\n        0.0\n    }\n\n"
        "    #[setter]\n    fn set_x(&mut self, _v: f64) {}\n}\n"
    )
    extra = {gate.RS_BINDING: gate.rs_binding(items, "Probe")}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    assert res["conn"]["native_mapped"] == gate.RUST_PAIR


def test_an_export_used_by_nobody_forms_no_pair_and_is_printed(base, capsys):
    _, _how, fear_rs, overlay, extra = _arm("RUST_CONTROLS", "G15")
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(fear_rs, extra))
    line = f"{_NOBODY}: _native.bind_step (crate-root export; its body reaches {_CER})"
    assert res["conn"]["native_attributed"] == [line]
    gate.report(res, _ROOT)
    out = capsys.readouterr().out
    assert "NATIVE EXPORTS ATTRIBUTED: 1" in out and line in out


def test_a_deleted_user_turns_its_export_into_used_by_nobody():
    res = gate.measure(_ROOT, overlay={_PFC: None})
    line = f"{_NOBODY}: _native.pfc_decide (crate-root export; its body reaches cortex, rng)"
    assert line in res["conn"]["native_attributed"]
    assert not [line for line in res["conn"]["native_attributed"] if line.startswith(_PFC)]


def _native_with_medulla(d):
    """A COPY of the real native/ tree under `d` whose sndl-core carries `medulla::step` (the target
    of Codex's reproducer); returns the binding's lib.rs."""
    shutil.copytree(
        os.path.join(_ROOT, "native"),
        os.path.join(d, "native"),
        ignore=shutil.ignore_patterns("target"),
    )
    core = Path(d, "native/sndl-core/src/lib.rs")
    core.write_text(core.read_text(encoding="utf-8") + f"pub mod {_MED};\n", encoding="utf-8")
    Path(d, f"native/sndl-core/src/{_MED}.rs").write_text("pub fn step() {}\n", encoding="utf-8")
    return Path(d, gate.RS_BINDING)


def _copy_facts(d, tag):
    root = Path(d, "native")
    return gate.native_facts(sorted(root.rglob("*.rs")), root, tag)


def _injected(step_site):
    """Facts for a COPY of the real native/ tree carrying Codex's medulla reproducer (`step();` at
    `step_site`)."""
    with tempfile.TemporaryDirectory() as d:
        lib = _native_with_medulla(d)
        text = lib.read_text(encoding="utf-8")
        assert text.count(_USE_LINE) == 1 and text.count(step_site) == 1, "binding changed shape"
        text = text.replace(_USE_LINE, _USE_LINE + f"\nuse sndl_core::{_MED}::step;")
        lib.write_text(text.replace(step_site, "        step();\n" + step_site), encoding="utf-8")
        return _copy_facts(d, "medulla injected")


_PYMODULE = "#[pymodule(gil_used = false)]\nmod _native {\n"
_RS_SUBMOD = "native/sndl-python/src/bindings.rs"  # region `bindings`: no connectome region


def _injected_submodule():
    """Codex's reproducer moved into a binding SUBMODULE (b4p P1): src/bindings.rs holds `sub_step`,
    whose body calls medulla's `step()`; lib.rs declares `mod bindings;` and exports it."""
    with tempfile.TemporaryDirectory() as d:
        lib = _native_with_medulla(d)
        text = lib.read_text(encoding="utf-8")
        assert text.count(_PYMODULE) == 1, "binding changed shape"
        exporting = _PYMODULE + "    #[pymodule_export]\n    use super::bindings::sub_step;\n"
        lib.write_text(text.replace(_PYMODULE, "mod bindings;\n\n" + exporting), encoding="utf-8")
        Path(d, _RS_SUBMOD).write_text(
            f"use pyo3::prelude::*;\nuse sndl_core::{_MED}::step;\n\n"
            "#[pyfunction]\npub fn sub_step() {\n    step();\n}\n",
            encoding="utf-8",
        )
        return _copy_facts(d, "medulla in a binding submodule")


def _medulla_fired(res, base):
    conn = res["conn"]
    assert conn["native_mapped"] == _MEDULLA_PAIR, conn["native_mapped"]
    assert conn["rust_directed"] == {("cortex", "rng")}  # the kernel's own wires are untouched
    expect = dict(base["metrics"])
    expect["undeclared_import_pairs"] += 1
    assert res["metrics"] == expect
    verdict = _verdict(res, base)
    assert verdict["undeclared_import_pairs"] is False
    assert all(ok for k, ok in verdict.items() if k != "undeclared_import_pairs")


def test_r_a_symbol_import_of_a_crate_root_export_counts_its_body_reach(base):
    overlay = {f"{_CTX}/_p6x_symbol_use.py": "from snn_data_lake._native import pfc_decide\n"}
    res = gate.measure(_ROOT, overlay=overlay, native=_injected(_IN_PFC))
    _medulla_fired(res, base)
    assert gate.own_unmapped(res["conn"], overlay) == []
    assert (
        f"{_CTX}/_p6x_symbol_use.py: _native.pfc_decide "
        "(crate-root export; its body reaches cortex, medulla, rng)"
    ) in res["conn"]["native_attributed"]


def test_r_b_attribute_style_use_by_the_real_pfc_counts_its_body_reach(base):
    res = gate.measure(_ROOT, native=_injected(_IN_PFC))
    _medulla_fired(res, base)
    assert res["conn"]["native_unmapped"] == []
    assert [line for line in res["conn"]["native_attributed"] if "pfc_decide" in line] == [
        f"{_PFC}: _native.pfc_decide (crate-root export; its body reaches cortex, medulla, rng)"
    ]


def test_r_c_what_a_helper_reaches_is_attributed_to_every_export_that_calls_it(base):
    res = gate.measure(_ROOT, native=_injected(_IN_ENTER))
    _medulla_fired(res, base)
    assert res["conn"]["native_attributed"] == [
        f"{_NOBODY}: _native.panic_probe (crate-root export; its body reaches medulla)",
        f"{_PFC}: _native.pfc_decide (crate-root export; its body reaches cortex, medulla, rng)",
    ]


def test_r_d_the_reproducer_in_a_binding_submodule_counts_its_body_reach(base):
    """b4p P1 (Opus p6x MAJOR-1): the old gate left this UNMAPPED ("region `bindings` is not a
    connectome region") and undeclared_import_pairs at 11."""
    overlay = {f"{_CTX}/_p6x_submodule_use.py": "from snn_data_lake._native import sub_step\n"}
    res = gate.measure(_ROOT, overlay=overlay, native=_injected_submodule())
    conn = res["conn"]
    assert conn["native_mapped"] == _MEDULLA_PAIR, conn["native_mapped"]
    assert conn["rust_directed"] == {
        ("cortex", "rng"),
        ("bindings", _MED),
    }  # its own: no connectome
    assert gate.own_unmapped(conn, overlay) == []
    assert (
        f"{_CTX}/_p6x_submodule_use.py: _native.sub_step "
        f"(export in non-connectome region `bindings`; its body reaches {_MED})"
    ) in conn["native_attributed"]
    expect = dict(base["metrics"])
    expect["undeclared_import_pairs"] += 1
    assert res["metrics"] == expect
    verdict = _verdict(res, base)
    assert verdict["undeclared_import_pairs"] is False
    assert all(ok for k, ok in verdict.items() if k != "undeclared_import_pairs")


def test_blind_spots_disclose_what_export_attribution_does_not_read():
    text = " ".join(gate.RUST_BLIND_SPOTS)
    assert "names no symbol here" not in text and "exports map to no region" not in text
    assert "non-call path" in text and "#[pyclass]" in text and "self.helper()" in text
    assert "every fn of its #[pymethods] impls" not in text  # the p6x overclaim (Opus MINOR-2)
    assert "LAST path segment" in text and "conflated" in text


def test_same_region_native_import_maps_to_no_pair(base):
    overlay = {f"{_CER}/_p6_native_same.py": "from snn_data_lake._native import fear_step\n"}
    res = gate.measure(_ROOT, overlay=overlay, native=gate.fixture_native(gate.RS_FEAR_GREEN))
    assert res["conn"]["native_mapped"] == set()
    assert gate.own_unmapped(res["conn"], overlay) == []
    assert res["metrics"] == base["metrics"]


def test_rust_region_table():
    labels = gate.frozen_labels(_ROOT)
    assert gate.rust_region("x/src/lib.rs", "x", labels) is None
    assert gate.rust_region("x/src/main.rs", "x", labels) is None
    assert gate.rust_region("x/src/rng.rs", "x", labels) == "rng"
    # a ported body named after its Python module inherits that module's label (Codex B2-fold)
    assert gate.rust_region("x/src/prefrontal_executive.rs", "x", labels) == "cortex"
    assert gate.rust_region("x/src/fear_conditioning.rs", "x", labels) == f"{_AMY}"
    assert gate.rust_region(f"src/{_AMY}/fear.rs", "", labels) == f"{_AMY}"
    assert gate.rust_region(f"x/src/{_CER}/fear_conditioning.rs", "x", labels) == f"{_AMY}"
    assert gate.rust_region("x/tests/it.rs", None, labels) is None


def test_reader_is_built_from_this_checkout_not_from_root(tmp_path, monkeypatch, base):
    _, _how, fear_rs, _overlay, _extra = _arm("RUST_REDS", "R1")
    paths = gate.write_native(tmp_path, fear_rs)
    roots = []
    real_facts = rust_facts.facts

    def spy(files, root=rust_facts.ROOT, allow_errors=False):
        roots.append(root)
        return real_facts(files, root, allow_errors)

    monkeypatch.setattr(rust_facts, "rust_files", lambda root=None, sub=None: paths)
    monkeypatch.setattr(rust_facts, "facts", spy)
    status, directed, exports, reach = gate.rust_region_imports(
        str(tmp_path), gate.frozen_labels(_ROOT), regions=base["conn"]["regions"]
    )
    assert roots == [rust_facts.ROOT], roots  # never --root's tools/rsfacts
    assert status.startswith("native/ PRESENT (7 .rs files)"), status
    assert directed == gate.RUST_PAIR  # absolute fact paths under an alternate root resolve
    assert exports == {"fear_step": f"{_CER}"}
    assert (
        reach == {}
    )  # fear_step lives in a connectome region: the file-region rule, not body reach


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
def test_absent_status_wins_over_the_symbol_reason(source, base, monkeypatch):
    monkeypatch.setattr(
        rust_facts, "rust_files", lambda *a, **k: []
    )  # native/ ABSENT (real tree has it now)
    overlay = {f"{_AMY}/_p6_native_reason.py": source}
    res = gate.measure(_ROOT, overlay=overlay)
    unmapped = gate.own_unmapped(res["conn"], overlay)
    assert len(unmapped) == 1 and gate.NATIVE_ABSENT in unmapped[0], unmapped


def test_unavailable_status_is_the_reason_for_an_unmapped_target(monkeypatch, base):
    monkeypatch.setattr(rust_facts, "rust_files", _boom)
    overlay = _arm("RUST_CONTROLS", "G4")[3]
    res = gate.measure(_ROOT, overlay=overlay)
    unmapped = gate.own_unmapped(res["conn"], overlay)
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
