"""The e-prop BINDING CONSTRAINT, mechanized: a backprop trainer may not become reachable.

WHAT IS BEING HELD. `hippocampus/developmental_loop_brain.py` and `hippocampus/growable_readout.py`
train by `loss.backward()` + Adam. The cross-cutting invariant forbids that -- *"e-prop, not
backprop ... no BPTT, no weight transport"* -- and both are permitted anyway, on ONE stated ground:
they are TRAIN-TIER OFFLINE and no live root reaches them. Their own words:

  * `developmental_loop_brain.py`: *"BINDING CONSTRAINT: this module must never be wired to a live
    LEARNING consumer ... making this module reachable converts the permitted offline trainer into
    an e-prop violation that nothing would announce."*
  * `dormancy_gate.EXEMPT["developmental_loop_brain"]`: *"the specs/010 e-prop adjudication permits
    it ONLY while no live root reaches it"*.

The constraint is written in THREE places -- the module docstring, the EXEMPT reason, and
specs/010 -- and was asserted in NONE. "Nothing would announce it" was literally true: `import` it
from a live root and every gate stays green. MEASURED before this file existed, by adding one edge
from `cortex.brain` to `developmental_loop_brain` in the reachability graph:

    reachable()   199 -> 207   (+8, both Adam modules among them)
    stale_exempt    0 -> 8
    dormant         0 -> 0     <- dormancy_gate's ONLY assert, blind to all of it

THE SET IS DERIVED, NOT TYPED, because a hand-listed pair is a list that rots and the next Adam
module lands outside it. The derivation is the discriminator `POLICIES.md` itself used to adjudicate
`cortex/predictive_coding.py` -- *"THE DISTINGUISHING TEST IS WHAT THE TRANSPOSE FEEDS"*: a
transposed weight feeding a WEIGHT UPDATE is the violation, one feeding an ACTIVATION is not.
Applied to autograd: a `.backward()` whose function ALSO STEPS AN OPTIMIZER is a training path; one
that returns a gradient for comparison is the reference the invariant explicitly requires
("verified cosine > 0.5 against an independent reverse-mode BPTT gradient").

THAT DISTINCTION IS LOAD-BEARING AND WAS COUNTED BOTH WAYS. The naive rule -- any `.backward()` in a
governed module -- scores 3, and the third is `cortex/eprop_kernel.py`, which IS reachable. It would
have reddened the one instrument in the tree that exists to PROVE the invariant holds, on the
strength of the `_fwd_bwd` reference gradient. The counted split:

    rule A  any governed .backward()                       3 modules, 1 REACHABLE (eprop_kernel)
    rule B  .backward() + optimizer step in SAME function   2 modules, 0 reachable   <- adopted

Rule B reproduces the two modules the constraint names, exactly, without naming them.

Run: pytest tests/test_eprop_train_tier_unreachable.py
"""

import ast
import pathlib

import pytest

dormancy_gate = pytest.importorskip("snn_data_lake.gates.dormancy_gate")

_SRC = pathlib.Path(dormancy_gate.__file__).resolve().parents[1]
_PKG = dormancy_gate._PKG

#: The one module whose `.backward()` is the invariant's own verification control rather than a
#: trainer. Named here so the discriminator's job is legible, NOT used to filter -- rule B excludes
#: it structurally, and `test_the_reference_gradient_is_not_a_hazard` asserts that it does.
_REFERENCE_GRADIENT = f"{_PKG}.cortex.eprop_kernel"


def _trains_by_backprop(tree):
    """True if any ONE function both back-propagates and steps an optimizer.

    Same-function is the whole test. `eprop_kernel._fwd_bwd` calls `zero_grad` and `backward` and
    returns the gradient -- it never steps, so it is a measurement, not a learning rule.
    """
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = {
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        if "backward" in calls and "step" in calls:
            return True
    return False


def hazard_modules():
    """Governed modules that train by backprop -- derived from the tree on every run."""
    out = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        mod = f"{_PKG}." + path.relative_to(_SRC).as_posix()[:-3].replace("/", ".")
        if mod not in dormancy_gate.BRAIN_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        if _trains_by_backprop(tree):
            out.add(mod)
    return out


# --------------------------------------------------------------------- the criterion is not vacuous
def test_the_hazard_set_is_not_empty():
    """A criterion over an empty set passes for the wrong reason, forever.

    If the derivation silently stops matching -- a refactor renames `train_step`, `.backward()`
    moves behind a helper -- every assert below goes green while the constraint stops being held.
    That is the F11 shape this repo deletes criteria for, so the set's own non-emptiness is the
    first thing asserted rather than assumed.
    """
    hazards = hazard_modules()
    assert hazards, (
        "no governed module matches 'backward + optimizer step in one function'. Either the two "
        "train-tier Adam modules were retired -- in which case delete this file -- or the "
        "derivation stopped matching and the binding constraint is no longer held by anything"
    )


def test_the_reference_gradient_is_not_a_hazard():
    """`eprop_kernel` back-propagates to MEASURE, not to learn, and it is reachable.

    The naive rule (any governed `.backward()`) scores 3 and reddens exactly this module -- the
    instrument the invariant's own "cosine > 0.5 against an independent BPTT gradient" clause
    depends on. A gate that reddens the proof of compliance teaches the reader to delete the proof.
    """
    assert _REFERENCE_GRADIENT in dormancy_gate.BRAIN_MODULES, (
        "eprop_kernel left the governed tier -- re-point this test or it stops discriminating"
    )
    assert _REFERENCE_GRADIENT in dormancy_gate.reachable(), (
        "eprop_kernel is no longer reachable, so it can no longer demonstrate that the naive rule "
        "would have fired on a live module -- the counted split this file rests on is stale"
    )
    assert _REFERENCE_GRADIENT not in hazard_modules(), (
        "the discriminator broke: eprop_kernel's reference gradient is being read as a trainer"
    )


# ------------------------------------------------------------------------------ the constraint
def test_no_backprop_trainer_is_reachable_from_a_live_path():
    """THE BINDING CONSTRAINT. Permitted only while unreachable -- so assert unreachable.

    `dormancy_gate.main()` cannot catch this: its only assert is `not cov["dormant"]`, and a module
    going from unreachable to reachable makes `dormant` SMALLER. The gate that would have to notice
    is structurally pointed the other way.
    """
    reach = dormancy_gate.reachable()
    live = sorted(m.replace(_PKG + ".", "") for m in hazard_modules() & reach)
    assert not live, (
        "e-prop invariant: these modules train by loss.backward() + optimizer.step() and are "
        f"permitted ONLY while no live root reaches them -- {live} is/are now reachable. Either "
        "revert the wire, or port the consumer to the governed three-factor lr_scale "
        "(basal_ganglia/valence_stdp.py, neuromodulation/cortisol.py) and retire the Adam path. "
        "Do not silence this by editing EXEMPT: EXEMPT governs dormancy, not the e-prop invariant"
    )


def test_the_constraint_goes_RED_on_one_live_import(monkeypatch):
    """CONSTRUCTED RED. A gate only ever seen green cannot be told from one that cannot see.

    One edge -- the live root importing the loop -- is the entire defect, and it is how the real
    thing would arrive: `developmental_loop_brain` module-top-imports `growable_readout`, so a
    single wire converts BOTH Adam modules at once.
    """
    real_graph = dormancy_gate._graph
    target = f"{_PKG}.hippocampus.developmental_loop_brain"

    def wired():
        g = {k: set(v) for k, v in real_graph().items()}
        g.setdefault(dormancy_gate._LIVE_ROOTS[0], set()).add(target)
        return g

    monkeypatch.setattr(dormancy_gate, "_graph", wired)
    caught = hazard_modules() & dormancy_gate.reachable()
    assert target in caught, "the red did not fire on the input built to trip it"
    assert len(caught) >= 2, (
        f"one import must cascade to BOTH Adam modules via the module-top import; caught {caught}"
    )


def test_a_stale_exemption_fails_the_dormancy_gate(monkeypatch):
    """The same wire ALSO turns 8 EXEMPT entries stale, and that was printed as a NOTE and never
    asserted. `coverage()` has computed `stale_exempt` all along; `main()` asserted only `dormant`.

    This is the second, independent tripwire on the same cascade -- it fires on any exemption that
    stops being true, not only on these two modules.
    """
    real_graph = dormancy_gate._graph

    def wired():
        g = {k: set(v) for k, v in real_graph().items()}
        g.setdefault(dormancy_gate._LIVE_ROOTS[0], set()).add(
            f"{_PKG}.hippocampus.developmental_loop_brain"
        )
        return g

    assert not dormancy_gate.coverage()["stale_exempt"], "baseline must be clean or the red is moot"

    monkeypatch.setattr(dormancy_gate, "_graph", wired)
    cov = dormancy_gate.coverage()
    assert len(cov["stale_exempt"]) >= 2, cov["stale_exempt"]
    assert cov["dormant"] == [], (
        "dormant must stay EMPTY under the break -- that is precisely why the pre-existing assert "
        "could not catch this, and if it ever does catch it this test is measuring the wrong thing"
    )
    with pytest.raises(AssertionError, match="stale"):
        dormancy_gate.main()
