"""r02b-instr: the Rust halves of instrument_guard (row 16) and replication_guard (row 4).

Every instrument_guard test drives `instrument_guard.scan`, the ONE function `main()`, `--assert`
and the self-check call, with ROOT pointed at a git-initialised fixture tree so the reader's
`rust_files` -> `facts` run for real; every replication_guard test drives `replication_guard.audit`.
Logic is never forked. Pre-registered in lanes/r0-arc2-1017/PREREG.md ("r02b-instr").
"""

import importlib
import pathlib
import subprocess
import sys

import pytest

from snn_data_lake.gates import rust_facts

_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

instrument_guard = pytest.importorskip("instrument_guard")
rg = importlib.import_module("snn_data_lake.gates.replication_guard")

LIB = "native/sndl-python/src/lib.rs"

IG_RED = """
use pyo3::prelude::*;
use std::env::set_var;

#[pyclass]
pub struct Loop { seed: u64 }

#[pymethods]
impl Loop {
    #[new]
    fn new(seed: u64) -> Self { fastrand::seed(seed); Loop { seed } }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    std::env::set_current_dir("..").ok();
    set_var("NORD_GPU", "1");
    set_seed(0);
    m.add_class::<Loop>()?;
    Ok(())
}

#[pymodule(gil_used = true)]
mod _legacy { #[pymodule_export] use super::Loop; }

fn set_seed(_s: u64) {}
"""

IG_GREEN = """
use pyo3::prelude::*;

#[pyclass]
pub struct Loop { rng: fastrand::Rng }

#[pymethods]
impl Loop {
    #[new]
    fn new(seed: u64) -> Self { Loop { rng: fastrand::Rng::with_seed(seed) } }
}

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Loop>()?;
    Ok(())
}

#[pyo3::pymodule(gil_used = false)]
mod _legacy { #[pymodule_export] use super::Loop; }

fn set_seed(_s: u64) {}
fn reseed(s: u64) { set_seed(s) }
"""

IG_RED_FINDINGS = [
    ("NATIVE-INIT-SEED", "fastrand::seed"),
    ("NATIVE-INIT-SEED", "set_seed"),
    ("NATIVE-PROCESS-MUTATION", "set_current_dir"),
    ("NATIVE-PROCESS-MUTATION", "set_var"),
    ("PYMODULE-GIL", "_legacy"),
    ("PYMODULE-GIL", "_native"),
]

RG_RED = """
use pyo3::prelude::*;

#[pyclass(name = "LiveBrain")]
pub struct NativeLoop;

pub struct Brain;

#[pyfunction]
#[pyo3(name = "AutonomousArm")]
fn arm() {}

#[cfg(test)]
mod tests { pub struct Brain; }

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeLoop>()?;
    m.add_function(wrap_pyfunction!(arm, m)?)?;
    Ok(())
}
"""

RG_GREEN = (
    RG_RED.replace('name = "LiveBrain"', 'name = "NativeLoop"')
    .replace("pub struct Brain;", "pub struct Core;")
    .replace('#[pyo3(name = "AutonomousArm")]\n', "")
)

RG_RED_REPLICAS = [
    ("AutonomousArm", f"{LIB}::arm"),
    ("Brain", f"{LIB}::Brain"),
    ("LiveBrain", f"{LIB}::NativeLoop"),
]

# Codex B1 r02b-instr MAJOR 1: the #[pyclass] and its #[pymodule_export] sit in DIFFERENT files
# (definition-only replica.rs reached from the #[pymodule] file); the finding names that file.
REPLICA = "native/sndl-python/src/replica.rs"
RG_XFILE_LIB = """
use pyo3::prelude::*;
mod replica;

#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::replica::NativeLoop;
}
"""
RG_XFILE_RED = {REPLICA: '#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'}
RG_XFILE_GREEN = {REPLICA: '#[pyclass(name = "NativeLoop")]\npub struct NativeLoop;\n'}
RG_XFILE_REPLICAS = [("LiveBrain", f"{REPLICA}::NativeLoop")]
# a second file exporting its own `LiveBrain`; both sites survive.
B_RS = "native/sndl-core/src/b.rs"
RG_COLLIDE = {
    **RG_XFILE_RED,
    B_RS: 'use pyo3::prelude::*;\n#[pyclass(name = "LiveBrain")]\npub struct Loop2;\n'
    "#[pymodule(gil_used = false)]\nmod _b { #[pymodule_export] use super::Loop2; }\n",
}
RG_COLLIDE_REPLICAS = sorted([*RG_XFILE_REPLICAS, ("LiveBrain", f"{B_RS}::Loop2")])
# the #[pymodule] only delegates to a helper in replica.rs that adds the renamed class.
RG_HELPER_LIB = """
use pyo3::prelude::*;
mod replica;

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    replica::register(m)
}
"""
_REG = (
    'use pyo3::prelude::*;\n#[pyclass(name = "LiveBrain")]\npub struct {c};\n'
    "pub fn {f}(m: &Bound<'_, PyModule>) -> PyResult<()> {{ m.add_class::<{c}>() }}\n"
)
RG_HELPER_RED = {REPLICA: _REG.format(c="NativeLoop", f="register")}
RG_HELPER_GREEN = {
    REPLICA: RG_HELPER_RED[REPLICA].replace('name = "LiveBrain"', 'name = "NativeLoop"')
}
# two helpers registering one `LiveBrain` through ONE #[pymodule]: each gets its own pair view.
RG_CHAIN_LIB = """
use pyo3::prelude::*;
mod a;
mod b;

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    a::reg_a(m)?;
    b::reg_b(m)
}
"""
A_RS, C_RS = "native/sndl-python/src/a.rs", "native/sndl-python/src/c.rs"
RG_CHAIN = {
    A_RS: _REG.format(c="LoopA", f="reg_a"),
    "native/sndl-python/src/b.rs": _REG.format(c="LoopB", f="reg_b"),
}
RG_CHAIN_REPLICAS = sorted(
    [("LiveBrain", f"{A_RS}::LoopA"), ("LiveBrain", "native/sndl-python/src/b.rs::LoopB")]
)
# two #[pymodule]s (two crates) each delegating to a helper named `register`: rust_facts resolves
# calls by last path segment, so the helpers must never share a view.
RG_TWO_ROOTS_LIB = RG_HELPER_LIB.replace("mod replica;", "mod a;").replace("replica::", "a::")
RG_TWO_ROOTS = {
    A_RS: _REG.format(c="LoopA", f="register"),
    "native/sndl-core/src/lib.rs": RG_HELPER_LIB.replace("mod replica;", "mod b;")
    .replace("replica::", "b::")
    .replace("fn _native", "fn _core"),
    "native/sndl-core/src/b.rs": _REG.format(c="LoopB", f="register"),
}
RG_TWO_ROOTS_REPLICAS = sorted(
    [("LiveBrain", f"{A_RS}::LoopA"), ("LiveBrain", "native/sndl-core/src/b.rs::LoopB")]
)
# an inert #[pymodule] in the definition file plus a later same-Rust-name decoy that exports itself
# unrenamed: the renamed class still resolves in the (replica.rs, lib.rs) pair view.
RG_INERT_DECOY = {
    REPLICA: '#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'
    "#[pymodule(gil_used = false)]\nmod _x {}\n",
    "native/sndl-python/src/zz.rs": "use pyo3::prelude::*;\n#[pyclass]\npub struct NativeLoop;\n"
    "#[pymodule(gil_used = false)]\nmod _zz { #[pymodule_export] use super::NativeLoop; }\n",
}
# the renamed #[pyclass] sits in ANOTHER #[pymodule] file; exported from b.rs.
RG_OTHER_ROOT_LIB = (
    'use pyo3::prelude::*;\n#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'
    "#[pymodule(gil_used = false)]\nmod _native {}\n"
)
RG_OTHER_ROOT = {
    B_RS: "use pyo3::prelude::*;\n#[pymodule(gil_used = false)]\n"
    "mod _b { #[pymodule_export] use crate::NativeLoop; }\n"
}
RG_OTHER_ROOT_GREEN = {**RG_OTHER_ROOT}
# only the whole-tree view: the chain runs through a helper that carries an inert #[pymodule] (so it
# is in no helper-chain view) into a third file holding the renamed class.
RG_DEEP_LIB = RG_CHAIN_LIB.replace("mod b;", "mod c;").replace(
    "    a::reg_a(m)?;\n    b::reg_b(m)", "    a::register(m)"
)
RG_DEEP = {
    A_RS: "use pyo3::prelude::*;\n#[pymodule(gil_used = false)]\nmod _inert {}\n"
    "pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {"
    " m.add_class::<crate::c::NativeLoop>() }\n",
    C_RS: '#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n',
}

CANON_SRC = {
    "engine/brain.py": "class Brain:\n    pass\n",
    "engine/live_brain.py": "class LiveBrain:\n    pass\n",
    "engine/autonomous_loop.py": "class AutonomousArm:\n    pass\n",
}


def _tree(tmp_path, text, extra=None):
    p = tmp_path / LIB
    p.parent.mkdir(parents=True)
    p.write_text(text, encoding="utf-8")
    for rel, src in (extra or {}).items():  # repo-relative paths, as p6_gate's fixture
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(src, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    return tmp_path


def _unavailable(monkeypatch):
    def raise_(paths, root=None, allow_errors=False):
        raise rust_facts.RustFactsUnavailable("constructed")

    monkeypatch.setattr(rust_facts, "facts", raise_)


# ------------------------------------------------------------------ instrument_guard (row 16)
def _ig(tmp_path, monkeypatch, text):
    monkeypatch.setattr(instrument_guard, "ROOT", _tree(tmp_path, text))
    return instrument_guard.scan()[0]


def test_ig_red_arm_has_exactly_the_six_preregistered_findings(tmp_path, monkeypatch):
    viol = _ig(tmp_path, monkeypatch, IG_RED)
    assert len(viol) == 6, viol
    assert {v[0] for v in viol} == {LIB}, viol
    assert sorted(
        (v[2], k) for v in viol for _r, k in IG_RED_FINDINGS if k in v[3] and v[2] == _r
    ) == (IG_RED_FINDINGS), viol


def test_ig_green_twin_has_zero_findings(tmp_path, monkeypatch):
    assert _ig(tmp_path, monkeypatch, IG_GREEN) == []


def test_ig_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch):
    monkeypatch.setattr(instrument_guard, "ROOT", _tree(tmp_path, IG_GREEN))
    _unavailable(monkeypatch)
    viol = instrument_guard.scan()[0]
    assert [v[2] for v in viol] == ["RUST-FACTS-UNAVAILABLE"], viol


def test_ig_rust_tags_are_rule_names():
    for tag, _k in IG_RED_FINDINGS:
        assert tag in instrument_guard.RULE_NAMES
    assert "RUST-FACTS-UNAVAILABLE" in instrument_guard.RULE_NAMES


@pytest.mark.parametrize(
    "tier,source",
    [
        ("tools", "from snn_data_lake import _native\n_native.set_seed(0)\n"),
        ("src", "from . import _native\n_native.set_seed(0)\n"),
        ("tools", "from snn_data_lake._native import set_seed\nset_seed(0)\n"),
    ],
    ids=["package-import", "relative-import", "from-import"],
)
def test_python_native_set_seed_at_module_scope_fires(tmp_path, monkeypatch, tier, source):
    (tmp_path / tier).mkdir()
    (tmp_path / tier / "probe.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(instrument_guard, "ROOT", tmp_path)
    viol = instrument_guard.scan()[0]
    assert [v[2] for v in viol] == ["MODULE-SCOPE-MUTATION"], viol
    assert "snn_data_lake._native.set_seed" in viol[0][3]


def test_python_native_set_seed_inside_a_function_is_not_flagged(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "probe.py").write_text(
        "from snn_data_lake import _native\n\n\ndef run():\n    _native.set_seed(0)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(instrument_guard, "ROOT", tmp_path)
    assert instrument_guard.scan()[0] == []


def test_ig_main_prints_the_native_status(capsys):
    instrument_guard.main([])
    out = capsys.readouterr().out
    assert instrument_guard._native()[0] in out
    assert "BLIND SPOTS" in out


def test_ig_selfcheck_fires_the_rust_reds():
    instrument_guard._selfcheck()


# ------------------------------------------------------------------ replication_guard (row 4)
def _pkg(tmp_path, replica):
    pkg = tmp_path / "snn_data_lake"
    files = dict(CANON_SRC, **({"replica.py": "class Brain:\n    pass\n"} if replica else {}))
    for rel, text in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return pkg


def test_rg_python_replica_red(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "_PKG_DIR", str(_pkg(tmp_path, replica=True)))
    a = rg.audit()
    assert a["replicas"] == [("Brain", "snn_data_lake.replica")], a
    assert a["missing"] == [], a


def test_rg_python_control_has_no_replica(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "_PKG_DIR", str(_pkg(tmp_path, replica=False)))
    a = rg.audit()
    assert a["replicas"] == [] and a["missing"] == [], a


def _rg(tmp_path, monkeypatch, text, extra=None):
    monkeypatch.setattr(rg, "ROOT", _tree(tmp_path, text, extra), raising=False)
    return rg.audit()


def test_rg_red_arm_has_exactly_the_three_preregistered_replicas(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_RED)
    assert sorted(a["replicas"]) == RG_RED_REPLICAS, a


def test_rg_green_twin_has_zero_replicas(tmp_path, monkeypatch):
    assert _rg(tmp_path, monkeypatch, RG_GREEN)["replicas"] == []


def test_rg_cross_file_export_is_attributed_to_the_pyclass_file(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_XFILE_LIB, RG_XFILE_RED)
    assert a["replicas"] == RG_XFILE_REPLICAS, a


def test_rg_cross_file_control_has_zero_replicas(tmp_path, monkeypatch):
    assert _rg(tmp_path, monkeypatch, RG_XFILE_LIB, RG_XFILE_GREEN)["replicas"] == []


def test_rg_renamed_pyclass_without_an_export_is_not_flagged(tmp_path, monkeypatch):
    assert _rg(tmp_path, monkeypatch, "mod replica;\n", RG_XFILE_RED)["replicas"] == []


def test_rg_two_files_exporting_one_name_both_survive(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_XFILE_LIB, RG_COLLIDE)
    assert sorted(a["replicas"]) == RG_COLLIDE_REPLICAS, a


@pytest.mark.parametrize("kept", [0, 1])
def test_rg_allowlisting_one_colliding_export_keeps_the_other(tmp_path, monkeypatch, kept):
    monkeypatch.setattr(rg, "PENDING_RETIREMENT", {RG_COLLIDE_REPLICAS[1 - kept]: "scheduled"})
    a = _rg(tmp_path, monkeypatch, RG_XFILE_LIB, RG_COLLIDE)
    assert a["replicas"] == [RG_COLLIDE_REPLICAS[kept]], a


def test_rg_helper_registered_export_is_resolved(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_HELPER_LIB, RG_HELPER_RED)
    assert a["replicas"] == RG_XFILE_REPLICAS, a


def test_rg_helper_registered_control_has_zero_replicas(tmp_path, monkeypatch):
    assert _rg(tmp_path, monkeypatch, RG_HELPER_LIB, RG_HELPER_GREEN)["replicas"] == []


def test_rg_helper_root_and_a_colliding_root_both_survive(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_HELPER_LIB, {**RG_HELPER_RED, B_RS: RG_COLLIDE[B_RS]})
    assert sorted(a["replicas"]) == RG_COLLIDE_REPLICAS, a


def test_rg_two_helpers_registering_one_name_through_one_pymodule(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_CHAIN_LIB, RG_CHAIN)
    assert sorted(a["replicas"]) == RG_CHAIN_REPLICAS, a


def test_rg_two_roots_with_same_named_helpers_both_survive(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_TWO_ROOTS_LIB, RG_TWO_ROOTS)
    assert sorted(a["replicas"]) == RG_TWO_ROOTS_REPLICAS, a


def test_rg_inert_pymodule_and_rust_name_decoy_do_not_hide_the_rename(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_XFILE_LIB, RG_INERT_DECOY)
    assert a["replicas"] == RG_XFILE_REPLICAS, a


def test_rg_pyclass_in_another_root_file_is_resolved(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_OTHER_ROOT_LIB, RG_OTHER_ROOT)
    assert a["replicas"] == [("LiveBrain", f"{LIB}::NativeLoop")], a


def test_rg_pyclass_in_another_root_file_control_has_zero_replicas(tmp_path, monkeypatch):
    lib = RG_OTHER_ROOT_LIB.replace('name = "LiveBrain"', 'name = "NativeLoop"')
    assert _rg(tmp_path, monkeypatch, lib, RG_OTHER_ROOT_GREEN)["replicas"] == []


def test_rg_chain_through_a_root_carrying_helper_needs_the_whole_tree_view(tmp_path, monkeypatch):
    a = _rg(tmp_path, monkeypatch, RG_DEEP_LIB, RG_DEEP)
    assert a["replicas"] == [("LiveBrain", f"{C_RS}::NativeLoop")], a


def test_rg_pending_retirement_allowlists_a_rust_replica(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "PENDING_RETIREMENT", {k: "scheduled" for k in RG_RED_REPLICAS})
    assert _rg(tmp_path, monkeypatch, RG_RED)["replicas"] == []


def test_rg_rust_facts_unavailable_with_rs_present_fails_main(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "ROOT", _tree(tmp_path, RG_GREEN), raising=False)
    _unavailable(monkeypatch)
    with pytest.raises(AssertionError, match="RustFactsUnavailable"):
        rg.main()


def test_rg_main_prints_the_native_status(capsys):
    rg.main()
    out = capsys.readouterr().out
    assert rg._native()[0] in out
    assert "BLIND SPOTS" in out


def test_rg_selfcheck_fires_the_reds():
    rg._selfcheck()
