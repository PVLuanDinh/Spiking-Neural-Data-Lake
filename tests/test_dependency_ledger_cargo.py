"""Cargo half of tools/dependency_ledger.py (P7, R0.2 lane r02-edge): UNUSED direction only.

Every arm goes through `crate_findings`, the function `main()` and `--self-check` call with the real
tree. UNDECLARED is the compiler's, so it is not tested here.

Run: pytest tests/test_dependency_ledger_cargo.py
"""

import pathlib
import subprocess
import sys

import pytest

_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

dependency_ledger = pytest.importorskip("dependency_ledger")


def _repo(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    return tmp_path


CARGO = '[package]\nname = "x"\nversion = "0.1.0"\n\n[dependencies]\nflate2 = "1"\nserde = "1"\n'
LIB_RED = "use serde::Serialize;\n"
LIB_GREEN = "use serde::Serialize;\nuse flate2::read::GzDecoder;\n"


def test_unused_crate_fires_exactly_once(tmp_path):
    root = _repo(tmp_path, {"crates/x/Cargo.toml": CARGO, "crates/x/src/lib.rs": LIB_RED})
    assert list(dependency_ledger.crate_findings(root)) == ["crates:flate2"]


def test_green_twin_has_zero_findings(tmp_path):
    root = _repo(tmp_path, {"crates/x/Cargo.toml": CARGO, "crates/x/src/lib.rs": LIB_GREEN})
    assert dependency_ledger.crate_findings(root) == {}


def test_lib_name_rename_workspace_dev_build_and_target_tables(tmp_path):
    root = _repo(
        tmp_path,
        {
            "Cargo.toml": '[workspace]\nmembers = ["m"]\n\n[workspace.dependencies]\n'
            'serde-json = { package = "serde_json", version = "1" }\nrand = "0.8"\n',
            "m/Cargo.toml": '[package]\nname = "m"\n\n[dependencies]\n'
            "serde-json = { workspace = true }\n"
            'my-crate = "1"\nrenamed = { package = "orig", version = "1" }\n'
            "\n[target.'cfg(unix)'.dependencies]\nlibc = \"0.2\"\n"
            "\n[dev-dependencies]\nrand = { workspace = true }\n"
            '\n[build-dependencies]\ncc = "1"\nunseen = "1"\n',
            "m/src/lib.rs": "use serde_json::json;\nuse my_crate::f;\nuse renamed::g;\n"
            "pub fn h() { libc::getpid(); }\n",
            "m/tests/t.rs": "use rand::Rng;\n",
            "m/build.rs": "extern crate cc;\nfn main() {}\n",
        },
    )
    assert list(dependency_ledger.crate_findings(root)) == ["crates:unseen"]


def test_dev_crate_used_only_in_src_tests_module_is_not_unused(tmp_path):
    root = _repo(
        tmp_path,
        {
            "Cargo.toml": '[package]\nname = "x"\n\n[dev-dependencies]\ntempfile = "3"\n',
            "src/lib.rs": "#[cfg(test)]\nmod t { fn f() { let _ = tempfile::tempdir(); } }\n",
        },
    )
    assert dependency_ledger.crate_findings(root) == {}


def test_use_numpy_in_rust_does_not_clear_pypi_numpy(tmp_path, monkeypatch, capsys):
    root = _repo(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname = "fixture"\ndependencies = ["numpy"]\n',
            "src/pkg/mod.py": "import json\n",
            "crates/x/Cargo.toml": '[package]\nname = "x"\n\n[dependencies]\nnumpy = "0.2"\n',
            "crates/x/src/lib.rs": "use numpy::PyArray1;\n",
        },
    )
    monkeypatch.setattr(dependency_ledger, "ROOT", root)
    monkeypatch.setattr(dependency_ledger, "SRC", root / "src")
    monkeypatch.setattr(dependency_ledger, "PYPROJECT", root / "pyproject.toml")
    monkeypatch.setattr(dependency_ledger, "EXEMPT", {})
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    assert dependency_ledger.crate_findings(root) == {}
    assert dependency_ledger.main(["--assert"]) == 1
    out = capsys.readouterr().out
    assert "numpy" in out.split("--- UNUSED")[1].split("--- MISDESCRIBED")[0], out
    assert "UNUSED: 1" in out, out


def test_stale_exempt_crates_row_reds(tmp_path, monkeypatch, capsys):
    root = _repo(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname = "fixture"\ndependencies = []\n',
            "src/pkg/mod.py": "import json\n",
            "crates/x/Cargo.toml": CARGO,
            "crates/x/src/lib.rs": LIB_GREEN,
        },
    )
    monkeypatch.setattr(dependency_ledger, "ROOT", root)
    monkeypatch.setattr(dependency_ledger, "SRC", root / "src")
    monkeypatch.setattr(dependency_ledger, "PYPROJECT", root / "pyproject.toml")
    monkeypatch.setattr(dependency_ledger, "EXEMPT", {})
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    monkeypatch.setattr(dependency_ledger, "EXEMPT_CRATES", {})
    assert dependency_ledger.main(["--assert"]) == 0
    monkeypatch.setattr(dependency_ledger, "EXEMPT_CRATES", {"ghost": "blesses nothing at all"})
    assert dependency_ledger.main(["--assert"]) == 1
    assert "ghost" in capsys.readouterr().out


def test_real_tree_day_one_count_is_the_preregistered_one():
    assert list(dependency_ledger.crate_findings()) == ["crates:proc-macro2"]
    assert "proc-macro2" in dependency_ledger.EXEMPT_CRATES


def test_self_check_covers_the_cargo_reds():
    assert dependency_ledger.self_check() == 0
