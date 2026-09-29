"""Guardrails for `tools/dependency_ledger.py` (P7, specs/010).

VERIFY THE VERIFIER (Standing Invariant 5): "A green gate that structurally cannot see the thing it
checks is worse than no gate, because it manufactures confidence." Every assertion below exists to
make a specific bucket of the gate FIRE on a fixture built to trip it. A test that only ran the gate
on the real tree would pass whether or not the gate could detect anything at all.

The import-name table is asserted explicitly because a wrong entry there manufactures a FALSE
UNDECLARED -- the dangerous direction, since the negative verdict is the one that pressures a real
dependency out of the tree.

Run: pytest tests/test_dependency_ledger.py
"""

import pathlib
import sys

import pytest

# ABSOLUTE, and derived from __file__. It was `sys.path.insert(0, "tools")` -- a RELATIVE entry,
# which stays on sys.path for the whole pytest process while other tests in this suite legitimately
# `monkeypatch.chdir(tmp_path)` (tests/test_selfchecks.py does it for every module). A relative
# sys.path entry is resolved at IMPORT time against whatever the cwd happens to be then, so the
# entry silently names a different directory after any chdir -- and an unrelated `tools/` under a
# temp directory would shadow this one. Same defect class as the module-scope side effects
# POLICIES.md records under the learn_daemon promotion.
_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

dependency_ledger = pytest.importorskip("dependency_ledger")


def test_import_name_table_maps_to_real_distribution_names():
    """`sklearn` is NOT a distribution; `scikit-learn` is. A wrong entry = a false UNDECLARED."""
    table = dependency_ledger.IMPORT_TO_DIST
    assert table["sklearn"] == "scikit-learn"
    assert table["PIL"] == "pillow"
    assert table["cv2"] == "opencv-python"
    for import_name, dist in table.items():
        assert import_name != dist, (
            f"{import_name!r} maps to itself -- entries that do not differ are noise and hide the "
            "ones that do"
        )
        assert dist == dependency_ledger._norm(dist), f"{dist!r} is not normalized"


def test_dist_of_strips_specifiers_and_extras():
    d = dependency_ledger._dist_of
    assert d("brian2>=2.10") == "brian2"
    assert d("mujoco==3.11.0") == "mujoco"
    assert d("spiking-neural-data-lake[arm,lake]") == "spiking-neural-data-lake"
    assert d("scikit_learn") == "scikit-learn", "normalization must fold _ to -"


def test_stdlib_is_never_reported():
    """`sys.stdlib_module_names` is the authority; a hand-maintained list rots."""
    assert "json" in sys.stdlib_module_names
    assert "pathlib" in sys.stdlib_module_names
    imported = dependency_ledger._imported(("src",))
    for stdlib_name in ("json", "pathlib", "math", "random", "zlib", "struct"):
        assert stdlib_name not in imported, f"{stdlib_name} is stdlib and must never be reported"


def test_first_party_and_repo_modules_are_never_reported():
    """The gate's first run called `training`, `eprop_lm` and `arcengine`-shaped locals third-party.

    Three of those were real repo modules imported flat. Resolution is by EXISTENCE anywhere in the
    repo, so this asserts the mechanism that fixed it rather than the specific names.
    """
    names = dependency_ledger._repo_module_names()
    assert "eprop_lm" in names, "training/eprop_lm.py exists and must resolve as first-party"
    assert "brain" in names, "src/snn_data_lake/engine/brain.py exists"
    imported = dependency_ledger._imported(("src",))
    assert "training" not in imported, "the repo's own training/ dir is not a PyPI package"
    assert "snn-data-lake" not in imported and "snn_data_lake" not in imported


def test_undeclared_bucket_FIRES_on_a_fixture(tmp_path, monkeypatch):
    """Make the UNDECLARED assert fire: an import of a package that is in no declaration."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "mod.py").write_text("import totally_not_a_real_distribution\n", encoding="utf-8")
    monkeypatch.setattr(dependency_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    found = dependency_ledger._imported(("src",))
    assert "totally-not-a-real-distribution" in found, (
        "the gate did not see an undeclared third-party import -- it cannot detect the thing it "
        "exists to detect"
    )


def test_guarded_vs_unguarded_is_distinguished(tmp_path, monkeypatch):
    """Severity depends on it: only an UNGUARDED module-top import breaks a clean install."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "top.py").write_text("import somepkg_a\n", encoding="utf-8")
    (src / "lazy.py").write_text(
        "def f():\n    import somepkg_b\n    return somepkg_b\n", encoding="utf-8"
    )
    monkeypatch.setattr(dependency_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    found = dependency_ledger._imported(("src",))
    assert found["somepkg-a"][0][2] is False, "a module-top import must be reported UNGUARDED"
    assert found["somepkg-b"][0][2] is True, "an import inside a function must be reported guarded"


def test_stale_exemption_fails_loudly():
    """An EXEMPT entry that blesses nothing must FAIL, not read as coverage.

    This is `dormancy_gate.EXEMPT`'s pattern and `citation_guard`'s lesson: line-keyed and
    name-keyed exemptions rot, and a rotted one silently widens the gate.
    """
    assert isinstance(dependency_ledger.EXEMPT, dict)
    for dist, reason in dependency_ledger.EXEMPT.items():
        assert reason and len(reason) > 20, (
            f"EXEMPT[{dist!r}] has no real reason -- 'it is fine' is not a disposition"
        )


def test_gate_runs_on_the_real_tree_and_reports_a_work_list():
    """The gate must run clean AND still be able to fail. Both halves matter.

    RE-MEASURED 2026-08-05 by running `python tools/dependency_ledger.py`: 1 UNDECLARED
    (`arcengine`, unguarded at `reasoning/arc_agi/play.py` and `play_brain.py`), 0 UNUSED,
    1 MISDESCRIBED (extra `[nir]` is in neither `[all]` nor ALL_EXCLUDES), 2 EXEMPT (`pillow`,
    `pyarrow`). The 2026-08-04 reading of this docstring omitted MISDESCRIBED entirely, and before
    that it said "a 4-item work list (arcengine, nir undeclared; pillow, pyarrow unused)" -- of
    which three items were wrong. Recorded as a DATED reading, not an invariant. This test does NOT
    assert the count -- pinning it would break every time someone fixes one, which is the wrong
    incentive. It asserts the gate executes end to end.
    """
    rc = dependency_ledger.main([])
    assert rc == 0, "report mode must always exit 0"


def test_assert_mode_returns_1_on_an_undeclared_import_and_0_when_the_ledger_balances(
    tmp_path, monkeypatch
):
    """A REAL verdict, both ways.

    This test used to end `assert dependency_ledger.main(["--assert"]) in (0, 1)`, which main
    cannot violate: it returns `1 if (do_assert and failed) else 0`. No input to this repository
    could redden it, so under F11 it is made reachable or deleted. It cannot be asserted against the
    LIVE tree either -- the gate legitimately returns 1 there today (arcengine + the [nir] extra) --
    so both verdicts are constructed over a fixture repo instead.
    """
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    mod = tmp_path / "src" / "pkg" / "mod.py"
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fixture"\ndependencies = []\n', encoding="utf-8")
    monkeypatch.setattr(dependency_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(dependency_ledger, "SRC", tmp_path / "src")
    monkeypatch.setattr(dependency_ledger, "PYPROJECT", pyproject)
    monkeypatch.setattr(dependency_ledger, "EXEMPT", {})

    mod.write_text("import totally_not_a_real_distribution\n", encoding="utf-8")
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    assert dependency_ledger.main(["--assert"]) == 1, "an undeclared import must FAIL assert mode"
    assert dependency_ledger.main([]) == 0, "report mode must exit 0 even with findings"

    mod.write_text("import json\n", encoding="utf-8")
    monkeypatch.setattr(dependency_ledger, "_REPO_MODULES", None)
    assert dependency_ledger.main(["--assert"]) == 0, "a balanced ledger must PASS assert mode"


# DELETED 2026-08-04: `test_the_gate_source_has_no_syntax_drift`. It re-parsed
# `dependency_ledger.__file__` with `ast.parse` and asserted nothing about the result. The module is
# already imported at the top of this file, so the source it re-read is source the interpreter has
# just compiled -- the assert was TRUE BY CONSTRUCTION and had NO REACHABLE RED: no input to this
# repository can make it fire while the import above succeeds, and if the import fails
# `importorskip` skips the whole file before this test runs. P8/F11: a criterion with no reachable
# red is DELETED, not weakened. The property it claimed to guard ("nothing else imports this tool,
# so nothing else notices a syntax error") is really held by the two `main()` calls above, which
# execute the gate end to end.
