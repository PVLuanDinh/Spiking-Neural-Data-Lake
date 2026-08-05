"""Guardrails for `tools/instrument_guard.py` (P4c -- the instruments are part of the brain).

VERIFY THE VERIFIER. This gate reports 0 violations on the live tree, because the six it was built
from were cleared in the same pass that built it. A gate that has only ever been seen green is
indistinguishable from a gate that cannot see anything, and this project has shipped three of those.
So every rule is fired against a fixture that reconstructs the defect it exists to catch, and the
REPAIRED shape is asserted NOT to fire -- a gate that reddens on its own remedy teaches the next
reader to stop applying the remedy.

This file also puts the gate inside the `pytest` gate command. `instrument_guard.py` is not in
`run_selfchecks.py` and not in `.github/workflows/ci.yml`; registering it in either is an edit to a
shared file. A test here is the honest way to make it assert on every run without one.

Run: pytest tests/test_instrument_guard.py
"""

import pathlib
import sys

import pytest

_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

instrument_guard = pytest.importorskip("instrument_guard")


def _scan_fixture(tmp_path, monkeypatch, source, name="probe.py", tier="tools"):
    """Run the gate over a one-file fake repo and return its violation rules."""
    (tmp_path / tier).mkdir(parents=True, exist_ok=True)
    (tmp_path / tier / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr(instrument_guard, "ROOT", tmp_path)
    return instrument_guard.scan()[0]


# --------------------------------------------------------------------------- each rule goes RED
def test_P4_construction_fires_on_a_direct_Brain_call(tmp_path, monkeypatch):
    """The real incident: `tools/arc_baseline.py` called `Brain()` inside `live_brain_solver()`
    while producing the published ARC "before" number. `substrate/replication_guard.py` cannot see
    this -- it asserts the class is DEFINED once, never that a consumer stopped calling it."""
    viol = _scan_fixture(
        tmp_path,
        monkeypatch,
        "from snn_data_lake.cortex.brain import Brain\n\n\ndef solve():\n    return Brain()\n",
    )
    assert [v[2] for v in viol] == ["P4-CONSTRUCTION"], viol
    assert "make_brain()" in viol[0][3], "the finding must name the sanctioned path"


def test_P4_construction_fires_on_the_QUALIFIED_spelling_too(tmp_path, monkeypatch):
    """`brain.Brain()` is the same violation and used to walk straight past the gate.

    MEASURED 2026-08-05 by neuter-A/B before the fix: this fixture returned `[]` from a scan that
    fired on the bare `Brain()` in the same run, because rule 1 keyed on `ast.Name`. A rule an
    import style can evade is not a rule -- and it is the rule this whole file exists to hold.
    """
    viol = _scan_fixture(
        tmp_path,
        monkeypatch,
        "from snn_data_lake.cortex import brain\n\n\ndef solve():\n    return brain.Brain()\n",
    )
    assert [v[2] for v in viol] == ["P4-CONSTRUCTION"], viol
    assert "brain.Brain(" in viol[0][3], "the finding must quote the spelling it found"


def test_an_unparseable_instrument_is_reported_not_skipped(tmp_path, monkeypatch):
    """A file `ast.parse` cannot read carries every other rule invisibly. It must be a finding."""
    viol = _scan_fixture(tmp_path, monkeypatch, "def broken(\n")
    assert [v[2] for v in viol] == ["SYNTAX"], viol


def test_cwd_dependent_syspath_fires_on_a_relative_literal(tmp_path, monkeypatch):
    viol = _scan_fixture(tmp_path, monkeypatch, 'import sys\nsys.path.insert(0, "src")\n')
    assert [v[2] for v in viol] == ["CWD-DEPENDENT-PATH"], viol


def test_module_scope_chdir_fires(tmp_path, monkeypatch):
    """P4c's own worked example: `run_selfchecks.py` runs part of the suite in ONE process."""
    viol = _scan_fixture(tmp_path, monkeypatch, 'import os\nos.chdir("/tmp")\n')
    assert [v[2] for v in viol] == ["MODULE-SCOPE-MUTATION"], viol


def test_module_scope_global_seeding_fires(tmp_path, monkeypatch):
    """The determinism invariant names this one: never module-level import-time seeding."""
    viol = _scan_fixture(tmp_path, monkeypatch, "import random\nrandom.seed(0)\n")
    assert [v[2] for v in viol] == ["MODULE-SCOPE-MUTATION"], viol
    viol = _scan_fixture(tmp_path, monkeypatch, "import numpy as np\nnp.random.seed(0)\n")
    assert [v[2] for v in viol] == ["MODULE-SCOPE-MUTATION"], viol


def test_the_tests_tier_is_scanned_too(tmp_path, monkeypatch):
    """`tests/` carried two of the five cwd-dependent entries, and it is the worse tier for them:
    the suite runs in ONE process and legitimately chdirs mid-run."""
    viol = _scan_fixture(
        tmp_path, monkeypatch, 'import sys\nsys.path.insert(0, "tools")\n', tier="tests"
    )
    assert [v[2] for v in viol] == ["CWD-DEPENDENT-PATH"], viol
    assert viol[0][0].startswith("tests/"), viol


# --------------------------------------------------------------------------- the fix stays green
def test_the_repaired_shapes_are_not_flagged(tmp_path, monkeypatch):
    """A __file__-derived path, a lazy chdir inside a function, and make_brain() must all pass.

    `os.environ.setdefault` at module scope is included deliberately: six tools use it to state a
    precondition before the first snn_data_lake import, and flagging it would have produced six
    findings that are all correct -- the allowlist shape that kills a gate under this repo's own
    adoption rule.
    """
    viol = _scan_fixture(
        tmp_path,
        monkeypatch,
        "import os\n"
        "import sys\n"
        "_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')\n"
        "sys.path.insert(0, _SRC)\n"
        "os.environ.setdefault('NORD_PERSIST', '0')\n"
        "def build():\n"
        "    import os\n"
        "    os.chdir('/tmp')\n"
        "    from snn_data_lake.cortex.brain import make_brain\n"
        "    return make_brain(seed=0, persist=False)\n",
    )
    assert viol == [], viol


def test_an_absolute_syspath_entry_is_not_flagged(tmp_path, monkeypatch):
    """Only RELATIVE literals are cwd-dependent. An absolute one is ugly, not broken."""
    viol = _scan_fixture(tmp_path, monkeypatch, 'import sys\nsys.path.insert(0, "/opt/whatever")\n')
    assert viol == [], viol


# --------------------------------------------------------------------------- the real tree
def test_the_live_tier_has_no_violations():
    """The gate that this whole feature exists to hold. If it goes red, an instrument stopped being
    governed -- and the failure message names the file, the line and the rule."""
    viol, _census = instrument_guard.scan()
    unexempt = [v for v in viol if f"{v[0]} -> {v[2]}" not in instrument_guard.EXEMPT]
    assert not unexempt, "\n".join(f"{v[0]}:{v[1]} {v[2]} -- {v[3]}" for v in unexempt)


def test_stale_exemptions_fail_loudly(monkeypatch, capsys):
    """`dormancy_gate.EXEMPT`'s pattern: an exemption that blesses nothing reads as coverage.

    THE STALE ENTRY IS CONSTRUCTED, because `EXEMPT` is empty today. The first version of this test
    iterated the real table and subtracted the real violations -- with `EXEMPT == {}` both of its
    assertions were true by construction, and no input to this repository could make either fire.
    That is the F11 shape ("a criterion with no reachable red is deleted, not weakened"), so the red
    is built here instead: a bogus key must be REPORTED as stale AND must fail `--assert`.
    """
    bogus = "tools/no_such_violation_here.py -> P4-CONSTRUCTION"
    monkeypatch.setattr(
        instrument_guard, "EXEMPT", {bogus: "a disposition long enough to be a real one"}
    )
    assert instrument_guard.main(["--assert"]) == 1, (
        "an EXEMPT entry that matches no real violation must FAIL the gate, not read as coverage"
    )
    assert "STALE EXEMPTIONS" in capsys.readouterr().out, "the stale entry must be named in output"


def test_an_exemption_without_a_disposition_fails_the_gate(tmp_path, monkeypatch, capsys):
    """ "fine" is not a disposition. Constructed, because EXEMPT is empty on this tree.

    The check lives in the GATE rather than only here: an exemption's quality is a property of the
    gate's table, and a rule that exists only in a test file is not enforced by the tool that ships.

    THE EXEMPTION IS MADE LIVE ON PURPOSE. A bogus key over the real tree would be caught by the
    STALE branch instead, so the test would pass with the disposition check deleted -- MEASURED: it
    did, in the first version of this test, and the A/B that removed `thin` from the verdict left it
    green. Exempting a violation that really exists is what isolates this branch, and the first half
    below doubles as the only coverage of the suppression path itself.
    """
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "red.py").write_text(
        'import sys\nsys.path.insert(0, "src")\n', encoding="utf-8"
    )
    monkeypatch.setattr(instrument_guard, "ROOT", tmp_path)
    key = "tools/red.py -> CWD-DEPENDENT-PATH"

    monkeypatch.setattr(
        instrument_guard,
        "EXEMPT",
        {key: "a real disposition: long enough to say why this site legitimately stays"},
    )
    assert instrument_guard.main(["--assert"]) == 0, "a live, dispositioned exemption must suppress"

    monkeypatch.setattr(instrument_guard, "EXEMPT", {key: "fine"})
    assert instrument_guard.main(["--assert"]) == 1, "an assurance is not a disposition"
    assert "NO DISPOSITION" in capsys.readouterr().out


def test_the_census_credits_a_tool_guarded_only_by_a_pytest_file():
    """The census's own first run called `red_reachability` and `dependency_ledger` SILENT while
    their pytest guards fire every bucket -- a false finding pointed at the best-verified tools in
    the tier. Asserted here because the fix is a judgement the gate makes, not a parse.

    REWRITTEN 2026-08-05. The first version asserted only `not c["has_selfcheck"]` -- the
    PRECONDITION for the branch, never the branch. It could not distinguish a census that credits a
    pytest-guarded tool from one that does not, which is the single thing its name claims. The
    judgement was unreachable from a test because it was a closure inside `main()`; it is
    `instrument_guard.guarded()` now, and this drives it.
    """
    _viol, census = instrument_guard.scan()
    by_file = {c["file"]: c for c in census}
    stems = {pathlib.Path(c["file"]).stem for c in census if c["tier"] == "tests"}
    for tool in (
        "tools/red_reachability.py",
        "tools/dependency_ledger.py",
        "tools/coverage_score.py",
    ):
        c = by_file[tool]
        assert not c["has_selfcheck"], (
            f"{tool} grew a self-check, so this test no longer isolates the pytest-guard branch "
            "-- re-point it at a tool that still lacks one, or it silently stops testing anything"
        )
        assert instrument_guard.guarded(c, stems), (
            f"{tool} has a pytest guard and was called SILENT"
        )
        assert not instrument_guard.guarded(c, set()), (
            f"{tool} was credited with NO test files in scope -- the census is crediting something "
            "other than the guard it claims to credit"
        )


def test_the_census_prefix_match_cannot_launder_an_unrelated_tool():
    """`test_arc_fix.py` must not credit `arc_fixdepth_control.py`: the prefix has to cover the
    whole stem plus a separator, or the census counts a tool as checked by a test of ANOTHER tool.
    """
    entry = {"file": "tools/arc_fixdepth_control.py", "has_selfcheck": False, "asserts": 0}
    assert not instrument_guard.guarded(entry, {"test_arc_fix", "test_arc_residual"})
    assert instrument_guard.guarded(entry, {"test_arc_fixdepth_control"})
    assert instrument_guard.guarded(entry, {"test_arc_fixdepth_control_extra"})


def test_assert_mode_returns_1_on_a_violation_and_0_when_clean(tmp_path, monkeypatch):
    """A REAL verdict, both ways.

    This asserted `main(["--assert"]) in (0, 1)` until 2026-08-05 -- a criterion with no reachable
    red, since `main` returns `1 if (do_assert and failed) else 0` and cannot produce anything else.
    It was the assert-true-by-construction shape this repo has shipped before, sitting in the file
    whose entire subject is gates that cannot see. Both verdicts are now constructed and run.
    """
    (tmp_path / "tools").mkdir()
    red = tmp_path / "tools" / "red.py"
    red.write_text('import sys\nsys.path.insert(0, "src")\n', encoding="utf-8")
    monkeypatch.setattr(instrument_guard, "ROOT", tmp_path)
    assert instrument_guard.main(["--assert"]) == 1, "a real violation must fail assert mode"
    assert instrument_guard.main([]) == 0, "report mode must exit 0 even with findings"

    red.write_text("import sys\n_ = sys\n", encoding="utf-8")
    assert instrument_guard.main(["--assert"]) == 0, "a clean tier must pass assert mode"
