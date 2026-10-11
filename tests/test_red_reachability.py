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
    return sum(seed % 2 for seed in seeds) / max(1, len(seeds)), "measured fraction"

def main():
    acc, _ = probe_access(None)
    frac, _ = probe_integration()
    assert acc is None
    assert 0.0 < frac < 1.0
    return 0
'''


IGNORED_TEN_INPUTS = """
def probe_mask(stage, mask, weights, host, seeds, controls, baseline, count, rate, reading):
    return "PASS"
"""


def test_assert_mode_rejects_ten_ignored_inputs(tmp_path, monkeypatch, capsys):
    _scan_source(tmp_path, monkeypatch, IGNORED_TEN_INPUTS)
    assert red_reachability.main(["--assert"]) == 1, (
        "ten ignored inputs returning PASS must fail --assert"
    )
    assert "NO_RED: 1" in capsys.readouterr().out
    assert red_reachability.main([]) == 0


@pytest.mark.parametrize(
    ("source", "reads", "defect"),
    [
        pytest.param(("def probe_x(state):\n    return True"), False, True, id="ignored-true"),
        pytest.param(
            ('def probe_x(state):\n    print(state)\n    return "PASS"'),
            True,
            True,
            id="logging-only",
        ),
        pytest.param(
            ("def probe_x(state):\n    unused = state\n    return True"),
            True,
            True,
            id="unused-assignment",
        ),
        pytest.param(
            ('def probe_x(state):\n    verdict = "PASS"\n    return verdict'),
            False,
            True,
            id="scalar-alias",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    verdict = True\n"
                "    next_verdict = verdict\n"
                "    print(state)\n"
                "    return next_verdict"
            ),
            True,
            True,
            id="alias-chain",
        ),
        pytest.param(
            ("def probe_x(state):\n    return True, state"), True, True, id="tuple-success"
        ),
        pytest.param(
            ('def probe_x(state):\n    return {"verdict": "PASS", "reading": state}'),
            True,
            True,
            id="dict-success",
        ),
        pytest.param(
            ('def probe_x(state):\n    return {"present": True, "reading": state}'),
            True,
            True,
            id="dict-present",
        ),
        pytest.param(
            'def probe_x(state):\n    return {"verdict": True, state: False}',
            True,
            False,
            id="dynamic-key-can-override-verdict",
        ),
        pytest.param(
            'def probe_x(state):\n    return {"verdict": True, **state}',
            True,
            False,
            id="dict-expansion-can-override-verdict",
        ),
        pytest.param(
            ('def probe_x(state):\n    if state:\n        return "PASS"\n    return "RED"'),
            True,
            False,
            id="input-selected-red",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    if state is None:\n"
                '        return None, "UNKNOWN"\n'
                '    return True, "PASS"'
            ),
            True,
            False,
            id="input-selected-refusal",
        ),
        pytest.param(
            ("def probe_x(state):\n    if state:\n        return True"),
            True,
            False,
            id="implicit-refusal",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    if not state:\n"
                '        raise ValueError("refused")\n'
                "    return True"
            ),
            True,
            False,
            id="explicit-refusal",
        ),
        pytest.param(
            ('def probe_x(state):\n    if state:\n        return True\n    return "PASS"'),
            True,
            True,
            id="both-branches-success",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                '    verdict = "PASS"\n'
                "    if not state:\n"
                '        verdict = "RED"\n'
                "    return verdict"
            ),
            True,
            False,
            id="input-selected-alias",
        ),
        pytest.param(
            ("def probe_x(state):\n    return verdict\n    verdict = True"),
            False,
            False,
            id="undominated-alias",
        ),
        pytest.param(
            "def probe_x(state):\n    match state:\n"
            "        case True:\n            return verdict\n"
            "    verdict = True\n    return verdict",
            True,
            False,
            id="unsupported-control-before-alias",
        ),
        pytest.param(
            ("def probe_x(state: state = state):\n    return 0.5"),
            False,
            True,
            id="annotation-default-not-loads",
        ),
        pytest.param(
            "def probe_x(state):\n    local: state = 0\n    return 0.5",
            False,
            True,
            id="local-annotation-not-load",
        ),
        pytest.param(
            "def probe_x(state):\n    def child(reading=state):\n        return reading\n"
            "    return child()",
            False,
            False,
            id="nested-default-access-unproven",
        ),
        pytest.param(
            "def probe_x(state):\n    @state\n    def child():\n        return 0.5\n"
            "    return child()",
            False,
            False,
            id="nested-decorator-access-unproven",
        ),
        pytest.param(
            ("def probe_x(state):\n    def child(state):\n        return state\n    return 0.5"),
            False,
            True,
            id="nested-parameter-shadow",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    def child():\n"
                "        state = 1\n"
                "        return state\n"
                "    return 0.5"
            ),
            False,
            True,
            id="nested-local-shadow",
        ),
        pytest.param(
            ("def probe_x(state):\n    def child():\n        return state\n    return child()"),
            False,
            False,
            id="captured-closure",
        ),
        pytest.param(
            ("def probe_x(state):\n    child = lambda state: state\n    return 0.5"),
            False,
            True,
            id="lambda-shadow",
        ),
        pytest.param(
            ("def probe_x(state):\n    child = lambda: state\n    return child()"),
            False,
            False,
            id="lambda-capture",
        ),
        pytest.param(
            ("def probe_x(state):\n    values = [state for state in (1, 2)]\n    return 0.5"),
            False,
            True,
            id="comprehension-shadow",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    values = [state for state in state]\n"
                "    return any(values)"
            ),
            True,
            False,
            id="comprehension-outer-iter",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    values = [state for item in (1, 2)]\n"
                "    return sum(values)"
            ),
            True,
            False,
            id="comprehension-free-load",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    values = {state: state for state in (1, 2)}\n"
                "    return 0.5"
            ),
            False,
            True,
            id="dict-comprehension-shadow",
        ),
        pytest.param(
            ("def probe_x(state):\n    values = (state for state in (1, 2))\n    return 0.5"),
            False,
            True,
            id="generator-shadow",
        ),
        pytest.param(
            ("def probe_x(state):\n    state = 0\n    return state"),
            True,
            False,
            id="rebound-unproven",
        ),
        pytest.param(
            ('def probe_x(state):\n    state = 0\n    return "PASS"'),
            False,
            True,
            id="rebound-fixed-success",
        ),
        pytest.param(
            ('def probe_x(state):\n    return locals()["state"]'),
            False,
            False,
            id="indirect-locals",
        ),
        pytest.param(
            ('def probe_x(state):\n    return eval("state")'), False, False, id="dynamic-eval"
        ),
        pytest.param(
            ('def probe_x(state):\n    exec("state = 0")\n    return "PASS"'),
            False,
            True,
            id="dynamic-fixed-success",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    verdict = True\n"
                '    exec("verdict = False")\n'
                "    return verdict"
            ),
            False,
            False,
            id="dynamic-alias-unproven",
        ),
        pytest.param(
            (
                "def probe_x(state):\n"
                "    verdict = True\n"
                "    def child():\n"
                "        nonlocal verdict\n"
                "        verdict = False\n"
                "    child()\n"
                "    return verdict"
            ),
            False,
            False,
            id="mutated-closure-alias",
        ),
        pytest.param(
            ("def probe_x(state):\n    return CURRENT_VERDICT"), False, False, id="global-backed"
        ),
        pytest.param(
            ("def probe_x(state):\n    return measure(state)"), True, False, id="call-backed"
        ),
        pytest.param(
            ('def probe_x(*args):\n    return "PASS" if args else "RED"'),
            True,
            False,
            id="variadic-positional",
        ),
        pytest.param(
            ('def probe_x(**kwargs):\n    return bool(kwargs["state"])'),
            True,
            False,
            id="variadic-keyword",
        ),
        pytest.param(
            ('def probe_x(*args, **kwargs):\n    return "PASS"'),
            False,
            True,
            id="ignored-variadics",
        ),
    ],
)
def test_supported_defects_and_unproven_shapes(tmp_path, monkeypatch, source, reads, defect):
    criteria, tautologies = _scan_source(tmp_path, monkeypatch, source)
    assert not tautologies
    assert len(criteria) == 1
    criterion = criteria[0]
    assert criterion["reads_system"] is reads
    assert criterion["no_red"] is defect
    assert criterion["analysis"] == ("NO_RED" if defect else "UNPROVEN")
    assert red_reachability.main(["--assert"]) == int(defect)
    assert red_reachability.main([]) == 0


@pytest.mark.parametrize("expression", ["CURRENT_VERDICT", "measure()"])
def test_zero_formal_external_state_is_unproven(tmp_path, monkeypatch, capsys, expression):
    criteria, _ = _scan_source(tmp_path, monkeypatch, f"def probe_x():\n    return {expression}")
    assert criteria[0]["has_formal_inputs"] is False
    assert criteria[0]["reads_system"] is False
    assert criteria[0]["analysis"] == "UNPROVEN"
    assert red_reachability.main(["--assert"]) == 0
    output = capsys.readouterr().out
    assert "UNPROVEN: 1" in output and "NO_FORMAL: 1" in output
    assert "exit 0 do not prove" in output
    assert "has a reachable red" not in output and "gate proves" not in output


def test_parameter_inventory_and_narrow_names(tmp_path, monkeypatch):
    criteria, _ = _scan_source(
        tmp_path,
        monkeypatch,
        "def probe_x(self, cls, positional, /, normal, *args, keyword, **kwargs):\n"
        "    return measure(positional, normal, args, keyword, kwargs)\n"
        "def ordinary_measurement(state):\n    return True\n",
    )
    assert len(criteria) == 1
    assert set(criteria[0]["params"]) == {"positional", "normal", "args", "keyword", "kwargs"}
    assert set(criteria[0]["input_loads"]) == set(criteria[0]["params"])


def test_repaired_fixture_has_constructed_input_dependent_readings(tmp_path, monkeypatch):
    _scan_source(tmp_path, monkeypatch, REPAIRED_SHAPE)
    namespace = {}
    exec(REPAIRED_SHAPE, namespace)
    assert namespace["probe_access"](None)[0] is None
    assert namespace["probe_access"](object())[0] is True
    assert namespace["probe_integration"]((0, 2))[0] == 0.0
    assert namespace["probe_integration"]((1, 3))[0] == 1.0
    assert red_reachability.main(["--assert"]) == 0


def test_dynamic_dictionary_key_constructs_a_real_refusal(tmp_path, monkeypatch):
    source = 'def probe_x(state):\n    return {"verdict": True, state: False}'
    _scan_source(tmp_path, monkeypatch, source)
    namespace = {}
    exec(source, namespace)
    assert namespace["probe_x"]("reading")["verdict"] is True
    assert namespace["probe_x"]("verdict")["verdict"] is False
    assert red_reachability.main(["--assert"]) == 0


def test_match_capture_cannot_prove_alias_success(tmp_path, monkeypatch):
    source = (
        "def probe_x(state):\n    verdict = True\n    match state:\n"
        "        case verdict:\n            pass\n    return verdict"
    )
    criteria, _ = _scan_source(tmp_path, monkeypatch, source)
    namespace = {}
    exec(source, namespace)
    assert namespace["probe_x"](True) is True
    assert namespace["probe_x"](False) is False
    assert criteria[0]["analysis"] == "UNPROVEN"
    assert red_reachability.main(["--assert"]) == 0


@pytest.mark.parametrize("condition", ["CURRENT_VERDICT", "measure()"])
def test_external_control_is_unproven(tmp_path, monkeypatch, capsys, condition):
    source = f"def probe_x():\n    if {condition}:\n        return True\n    return False"
    criteria, _ = _scan_source(tmp_path, monkeypatch, source)
    namespace = {"CURRENT_VERDICT": True}
    namespace["measure"] = lambda: namespace["CURRENT_VERDICT"]
    exec(source, namespace)
    assert namespace["probe_x"]() is True
    namespace["CURRENT_VERDICT"] = False
    assert namespace["probe_x"]() is False
    assert criteria[0]["analysis"] == "UNPROVEN"
    assert criteria[0]["has_formal_inputs"] is False
    assert red_reachability.main(["--assert"]) == 0
    output = capsys.readouterr().out
    assert "NO_RED: 0" in output and "NO_FORMAL: 1" in output


def test_later_comprehension_target_shadows_all_inner_loads(tmp_path, monkeypatch):
    source = (
        "def probe_x(state):\n"
        "    values = [state for item in (1,) for state in state]\n"
        "    return sum(values)"
    )
    criteria, _ = _scan_source(tmp_path, monkeypatch, source)
    assert criteria[0]["reads_system"] is False
    assert criteria[0]["analysis"] == "UNPROVEN"
    assert red_reachability.main(["--assert"]) == 0


def test_starred_comprehension_target_shadows_formal_input(tmp_path, monkeypatch):
    source = "def probe_x(state):\n    values = [state for *state, in [(1,)]]\n    return 0.5"
    criteria, _ = _scan_source(tmp_path, monkeypatch, source)
    namespace = {}
    exec(source, namespace)
    assert namespace["probe_x"](None) == 0.5
    assert namespace["probe_x"](object()) == 0.5
    assert criteria[0]["reads_system"] is False
    assert criteria[0]["no_red"] is True
    assert red_reachability.main(["--assert"]) == 1


def _scan_source(tmp_path, monkeypatch, source, name="fixture.py"):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr(red_reachability, "SRC", tmp_path / "src")
    monkeypatch.setattr(red_reachability, "ROOT", tmp_path)
    return red_reachability.scan()


def test_NO_RED_fires_on_a_zero_argument_criterion(tmp_path, monkeypatch):
    """Zero formal inputs with literal verdicts reconstruct the supported original defect."""
    criteria, _ = _scan_source(tmp_path, monkeypatch, PRE_FIX_CONSCIOUSNESS_PROBE)
    no_red = [c["name"] for c in criteria if not c["reads_system"]]
    assert "fixture.probe_binding" in no_red, (
        "the gate did not flag a zero-argument criterion -- it cannot detect the defect it exists "
        "to detect (literal result with no formal inputs)"
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
    assert all(c["reads_system"] for c in criteria), "repaired criteria load the system input"
    assert not any(c["no_red"] for c in criteria)


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
    bare verdict names). `reads_system` now records unshadowed formal-name loads, which is a
    stronger regression than parameter presence but still does not prove reachable RED.

    The two "deleted (C08)" asserts that stood here -- `probe_binding` and `probe_reentry` must not
    reappear -- are DROPPED with the module they guarded: both names lived in
    `reasoning/consciousness_probe.py`, retired at T081, so nothing can re-add them there.
    """
    criteria, _ = red_reachability.scan()
    probes = {c["name"]: c for c in criteria if c["name"].startswith("report_order_probe.")}
    assert probes, "report_order_probe exposes no discoverable criteria"
    for name, c in probes.items():
        assert c["reads_system"], f"{name} loads no formal input -- the C09 defect has returned"
