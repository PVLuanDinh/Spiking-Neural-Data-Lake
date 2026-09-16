"""Guardrails for `tools/red_reachability.py` (P8, specs/010).

VERIFY THE VERIFIER (Standing Invariant 5). This gate currently reports NO_RED 0 and TAUTOLOGICAL 0
on the live tree -- **because the one real case was fixed before the gate existed**. A gate that has
only ever been seen green is indistinguishable from a gate that cannot see anything, and this
project has shipped three of those. So every bucket below is fired against a fixture that
reconstructs the defect, including the exact pre-fix shape of `consciousness_probe`.

Run: pytest tests/test_red_reachability.py
"""

import ast
import pathlib
import sys

import pytest

# ABSOLUTE, derived from __file__. It was `sys.path.insert(0, "tools")`; a relative sys.path entry
# is re-resolved against the cwd at import time, and this suite legitimately chdirs elsewhere
# (tests/test_selfchecks.py does it per module), so the entry can silently name a different
# directory -- or let an unrelated `tools/` under a temp dir shadow this one.
_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

red_reachability = pytest.importorskip("red_reachability")


PRE_FIX_CONSCIOUSNESS_PROBE = '''
"""The shape this gate exists to catch, reconstructed from consciousness_probe before 2026-08-04."""

def probe_binding():
    """Zero arguments: no state of the system can change the answer."""
    return True, "spiking 100% vs rate control 0%"

def probe_reentry():
    return True, "arithmetic over literals"

def main():
    results = {}
    for name, fn in [("BINDING", probe_binding), ("RE-ENTRY", probe_reentry)]:
        present, detail = fn()
        results[name] = present
    bnd, _ = probe_binding()
    assert bnd
    return 0
'''

REPAIRED_SHAPE = '''
def probe_access(brain=None):
    """Takes the system under test."""
    if brain is None:
        return None, "UNKNOWN"
    return True, "ok"

def probe_integration(seeds=(0, 1, 2)):
    return 0.5, "half"

def main():
    acc, _ = probe_access(None)
    frac, _ = probe_integration()
    assert acc is None
    assert 0.0 < frac < 1.0
    return 0
'''


def _scan_source(tmp_path, monkeypatch, source, name="fixture.py"):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr(red_reachability, "SRC", tmp_path / "src")
    monkeypatch.setattr(red_reachability, "ROOT", tmp_path)
    return red_reachability.scan()


def test_NO_RED_fires_on_a_zero_argument_criterion(tmp_path, monkeypatch):
    """The decisive check: a criterion taking no arguments is invariant to the whole system."""
    criteria, _ = _scan_source(tmp_path, monkeypatch, PRE_FIX_CONSCIOUSNESS_PROBE)
    no_red = [c["name"] for c in criteria if not c["reads_system"]]
    assert "fixture.probe_binding" in no_red, (
        "the gate did not flag a zero-argument criterion -- it cannot detect the defect it exists "
        "to detect (Kleiner & Hoel: invariant to the system = unfalsifiable at every level)"
    )
    assert "fixture.probe_reentry" in no_red


def test_TAUTOLOGICAL_fires_on_assert_of_a_computed_verdict(tmp_path, monkeypatch):
    """`bnd, _ = probe_binding(); assert bnd` -- a check that cannot go red by construction."""
    _, tautologies = _scan_source(tmp_path, monkeypatch, PRE_FIX_CONSCIOUSNESS_PROBE)
    assert tautologies, "the gate missed an assert on a verdict the same scope just computed"
    assert any(t["criterion"] == "probe_binding" for t in tautologies)


def test_the_REPAIRED_shape_is_NOT_flagged(tmp_path, monkeypatch):
    """The fix must not be punished as hard as the defect.

    `assert acc is None` and `assert 0.0 < frac < 1.0` assert FALSIFIABLE PROPERTIES of the result,
    not "it passed". An earlier version of this gate flagged both, which would have taught the next
    reader to stop asserting anything about a criterion's output at all.
    """
    criteria, tautologies = _scan_source(tmp_path, monkeypatch, REPAIRED_SHAPE)
    assert not tautologies, f"repaired self-checks were wrongly flagged: {tautologies}"
    assert all(c["reads_system"] for c in criteria), "repaired criteria take the system as input"


def test_tautologies_are_not_double_counted(tmp_path, monkeypatch):
    """`ast.walk` yields Module AND the FunctionDef inside it; the first run reported 8 for 4."""
    _, tautologies = _scan_source(tmp_path, monkeypatch, PRE_FIX_CONSCIOUSNESS_PROBE)
    keys = [(t["file"], t["line"], t["var"]) for t in tautologies]
    assert len(keys) == len(set(keys)), f"double-counted: {keys}"


def test_bare_truthiness_discrimination():
    """The single judgement this gate makes; asserted directly rather than through a fixture."""
    bare = red_reachability._is_bare_truthiness
    assert bare(ast.parse("assert x", mode="exec").body[0].test) is True
    assert bare(ast.parse("assert all(v)", mode="exec").body[0].test) is True
    assert bare(ast.parse("assert x is None", mode="exec").body[0].test) is False
    assert bare(ast.parse("assert 0.0 < f < 1.0", mode="exec").body[0].test) is False
    assert bare(ast.parse("assert a == b", mode="exec").body[0].test) is False


def test_stale_exemption_is_reported():
    """An exemption blessing nothing reads as coverage; it must fail loudly instead."""
    assert isinstance(red_reachability.EXEMPT, dict)
    for name, reason in red_reachability.EXEMPT.items():
        assert reason and len(reason) > 20, f"EXEMPT[{name!r}] carries no real disposition"


def test_gate_runs_on_the_real_tree():
    rc = red_reachability.main([])
    assert rc == 0, "report mode must always exit 0"


def test_assert_mode_returns_1_on_the_pre_fix_shape_and_0_on_the_repaired_one(
    tmp_path, monkeypatch
):
    """A REAL verdict, both ways -- through main(), not just through scan().

    This was `assert red_reachability.main(["--assert"]) in (0, 1)` until 2026-08-05: main returns
    `1 if (do_assert and failed) else 0`, so no input to this repository could make that assert
    fire. It is the assert-true-by-construction shape the gate below exists to hunt, sitting in the
    gate's own test file. Both verdicts are constructed and run here instead (F11: a criterion with
    no reachable red is deleted or made reachable, never left decorative).
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    probe = src / "fixture.py"
    monkeypatch.setattr(red_reachability, "SRC", tmp_path / "src")
    monkeypatch.setattr(red_reachability, "ROOT", tmp_path)

    probe.write_text(PRE_FIX_CONSCIOUSNESS_PROBE, encoding="utf-8")
    assert red_reachability.main(["--assert"]) == 1, "a zero-argument criterion must FAIL the gate"
    assert red_reachability.main([]) == 0, "report mode must exit 0 even with findings"

    probe.write_text(REPAIRED_SHAPE, encoding="utf-8")
    assert red_reachability.main(["--assert"]) == 0, "the repaired shape must PASS the gate"


def test_live_report_order_probe_reads_the_system():
    """The regression this feature exists to prevent, asserted against the real module.

    RE-POINTED 2026-09-16 (specs/022 T081, `evidence/t081-report-order.md` 2.6): this guarded
    `consciousness_probe.` criteria until that module was RETIRED (P2, T034's deferred clause --
    both replacement halves now exist: the report channel since T034, and this probe once its six
    reds fired). The live criteria it guards are `report_order_probe.`'s. `scan()` discovers a
    criterion by NAME (`probe_*` prefix, `_present`/`_verdict`/`_is_present` suffix, or one of the
    bare verdict names) and scores `reads_system` as "takes at least one real parameter", so the
    contract asserted is the same one: a criterion that reads no system cannot fail.

    The two "deleted (C08)" asserts that stood here -- `probe_binding` and `probe_reentry` must not
    reappear -- are DROPPED with the module they guarded: both names lived in
    `reasoning/consciousness_probe.py`, retired at T081, so nothing can re-add them there.
    """
    criteria, _ = red_reachability.scan()
    probes = {c["name"]: c for c in criteria if c["name"].startswith("report_order_probe.")}
    assert probes, "report_order_probe exposes no discoverable criteria"
    for name, c in probes.items():
        assert c["reads_system"], f"{name} takes no arguments -- the C09 defect has returned"
