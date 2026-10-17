"""Rust half of import_manifest (P3 cross-tier, R0.2 lane r02-edge): every `_native` name a governed
module references must be a real PyO3 export (E1, static) and must import + `hasattr` (E2, runtime).

Every arm goes through `native_findings` / `broken`, which `main()` calls with the real tree.

Run: pytest tests/test_import_manifest_native.py
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from snn_data_lake.gates import import_manifest, rust_facts

PYO3 = """
use pyo3::prelude::*;

#[pyfunction]
fn add(a: i64, b: i64) -> i64 { a + b }
%s
#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::add;
%s}
"""
LIF = "\n#[pyfunction]\nfn lif_step(v: f64) -> f64 { v * 0.9 }\n"
LIF_EXPORT = "    #[pymodule_export]\n    use super::lif_step;\n"

BRAIN = (
    "from snn_data_lake._native import lif_step\n"
    "from snn_data_lake import _native\n"
    "_native.lif_step(1.0)\n"
)


def _tree(tmp_path: Path, exports_lif: bool, py: str = BRAIN) -> Path:
    files = {
        "native/sndl-core/src/lib.rs": "pub mod lake;\n",
        "native/sndl-core/src/lake.rs": "pub fn append() {}\n",
        "native/sndl-python/src/lib.rs": PYO3 % ((LIF, LIF_EXPORT) if exports_lif else ("", "")),
        "src/snn_data_lake/__init__.py": "",
        "src/snn_data_lake/engine/__init__.py": "",
        "src/snn_data_lake/engine/brain.py": py,
        "src/snn_data_lake/reasoning/__init__.py": "",
        "src/snn_data_lake/reasoning/ungoverned.py": "from snn_data_lake._native import ghost\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    return tmp_path


def _refs(root: Path) -> dict:
    return import_manifest.native_refs(root / "src")


def _stub(*names):
    return lambda _name: types.SimpleNamespace(**dict.fromkeys(names, 1))


def test_refs_are_governed_only_and_cover_both_spellings(tmp_path):
    refs = _refs(_tree(tmp_path, True))
    assert refs == {"snn_data_lake.engine.brain": {"lif_step"}}, refs


def test_e1_red_arm_fires_exactly_once_through_the_entry_point(tmp_path):
    root = _tree(tmp_path, exports_lif=False)
    status, bad = import_manifest.native_findings(
        _refs(root), root, importer=lambda _n: pytest.fail("E2 must not re-report an E1 break")
    )
    assert status.startswith("native/ PRESENT"), status
    assert list(bad) == ["snn_data_lake._native:lif_step"], bad
    assert bad["snn_data_lake._native:lif_step"].startswith("BROKEN-STATIC"), bad


def test_green_twin_has_zero_findings(tmp_path):
    root = _tree(tmp_path, exports_lif=True)
    _status, bad = import_manifest.native_findings(
        _refs(root), root, importer=_stub("add", "lif_step")
    )
    assert bad == {}, bad


def test_e2_red_fires_through_the_real_broken_path(tmp_path, monkeypatch):
    root = _tree(tmp_path, exports_lif=True)
    stub = types.ModuleType("snn_data_lake._native")
    stub.add = 1  # the built extension is missing `lif_step`: static is clean, runtime is not
    monkeypatch.setitem(sys.modules, "snn_data_lake._native", stub)
    bad = import_manifest.broken({}, native=_refs(root), root=root)
    assert list(bad) == ["snn_data_lake._native:lif_step"], bad
    assert not bad["snn_data_lake._native:lif_step"].startswith("BROKEN-STATIC"), bad


def test_broken_without_native_computes_the_governed_refs_itself(tmp_path):
    root = _tree(tmp_path, exports_lif=False)
    bad = import_manifest.broken({}, root=root)
    assert list(bad) == ["snn_data_lake._native:lif_step"], bad


def test_broken_with_explicit_empty_native_skips_the_rust_half(tmp_path):
    root = _tree(tmp_path, exports_lif=False)
    assert import_manifest.broken({}, native={}, root=root) == {}


def test_refs_with_native_absent_are_broken(tmp_path, monkeypatch):
    root = _tree(tmp_path, exports_lif=True)
    for f in (root / "native").rglob("*.rs"):
        f.unlink()
    monkeypatch.delitem(sys.modules, "snn_data_lake._native", raising=False)
    status, bad = import_manifest.native_findings(_refs(root), root)
    assert status.startswith("native/ ABSENT"), status
    assert list(bad) == ["snn_data_lake._native"], bad


def test_no_refs_and_no_native_is_clean_and_says_absent(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    status, bad = import_manifest.native_findings({}, tmp_path)
    assert (status.startswith("native/ ABSENT"), bad) == (True, {})


def test_unparseable_rust_fails_closed(tmp_path):
    root = _tree(tmp_path, exports_lif=True)
    (root / "native/sndl-core/src/lib.rs").write_text("pub fn broken( {", encoding="utf-8")
    _status, bad = import_manifest.native_findings(_refs(root), root, importer=_stub("add"))
    assert list(bad) == ["native/"], bad
    assert "unparseable" in bad["native/"], bad


def test_real_tree_reports_native_absent():
    assert rust_facts.rust_files(rust_facts.ROOT) == [], "native/ exists now: re-count the gate"
    status, bad = import_manifest.native_findings(import_manifest.native_refs())
    assert status.startswith("native/ ABSENT") and bad == {}, (status, bad)
