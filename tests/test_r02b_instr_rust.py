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

# Codex B1 r02b-instr MAJOR 1: the #[pyclass] and its #[pymodule_export] sit in DIFFERENT files;
# exports resolve over the whole tree, and the finding names the export's own file.
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
RG_XFILE_RED = {"replica.rs": '#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'}
RG_XFILE_GREEN = {"replica.rs": '#[pyclass(name = "NativeLoop")]\npub struct NativeLoop;\n'}
RG_XFILE_REPLICAS = [("LiveBrain", f"{REPLICA}::NativeLoop")]

CANON_SRC = {
    "engine/brain.py": "class Brain:\n    pass\n",
    "engine/live_brain.py": "class LiveBrain:\n    pass\n",
    "engine/autonomous_loop.py": "class AutonomousArm:\n    pass\n",
}


def _tree(tmp_path, text, extra=None):
    p = tmp_path / LIB
    p.parent.mkdir(parents=True)
    p.write_text(text, encoding="utf-8")
    for name, src in (extra or {}).items():
        (p.parent / name).write_text(src, encoding="utf-8")
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
