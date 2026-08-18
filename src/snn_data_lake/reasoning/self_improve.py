r"""
SAFE BOUNDED SELF-IMPROVEMENT -- the brain proposes changes to its OWN codebase, but ACCEPTS one only
if it passes every gate in isolation AND strictly improves a measured fitness, and NEVER touches its
own safety files. This is the controller that encodes, as code, the exact discipline used to build
and audit this project: propose -> isolate -> verify (all gates + behavioural probes) -> ratchet
(accept only if strictly better, no regression) -> else roll back.

HONEST SCOPE -- read this before believing any "self-improvement / singularity" claim (it mirrors
docs/SELF_IMPROVEMENT_RESEARCH.md, the project's own stated boundary):

  CAN mean: a continual loop where the brain reads its own architecture (the graphify self-map),
  proposes an ADDITIVE / refactoring change, tries it in an ISOLATED copy, runs the FULL verification
  harness (run_selfchecks + ruff + pytest + a behavioural probe battery), and merges it ONLY if every
  gate stays green AND a fitness metric strictly increases AND no probe regresses -- otherwise the
  change is discarded and the live tree is untouched. Monotonic: fitness never decreases. Corrigible:
  the safety files and the verifier itself are OUTSIDE the writable set, so the brain cannot weaken
  its own check (the cheap inverse of a Goedel machine).

  CANNOT mean: unbounded recursive self-improvement, an intelligence explosion, self-authored
  terminal values, or self-rewrite of the safety envelope. Two hard guardrails make that structural,
  not aspirational: (1) the protected set (brainstem safety files + this verifier) is refused before
  any change is even tried; (2) improvement is gated by a fitness function the brain does not author
  and cannot regress. "Gradually reach singularity" is NOT what this is -- it is bounded, verifiable,
  ratcheted enhancement within a proposal space, and it stays honest by construction: a change that
  cannot PROVE it helps (green gates + strictly higher fitness) is rejected, not merged on faith.

The self-check below proves the PROTOCOL deterministically on a toy system (no real repo mutation):
accept a strict-improvement-and-green change; reject a change that reddens a gate (a would-be
regression) and roll back; reject a green change that does not improve fitness; refuse a change that
touches a protected file; and show fitness is monotonic non-decreasing across a run. The real-mode
adapter (documented, not run in CI) wires verify -> subprocess(run_selfchecks + ruff + pytest +
probes), isolate -> a git worktree, apply -> the brain's proposal.

Deps: stdlib + (optional) the graphify self-map JSON.
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.self_improve

(the module path it is at TODAY. The previous line named `research/self_improve.py`, which has
not existed since v3.78 -- the same stale-path class the PROTECTED entry below was found to have
on 2026-08-01, in the one file whose whole job is to protect paths.)
"""

import copy
import json
import os
import sys

from snn_data_lake.substrate.safety_envelope import ENVELOPE as _ENVELOPE

#: Files this self-improver refuses to touch. The five-file safety envelope is IMPORTED, not retyped
#: (P2, 2026-08-18) -- it was one of three hand-maintained copies in src/, in two spellings, and
#: `substrate/safety_envelope` is now the one definition. What this set adds is its own: THIS
#: module, the proposer, so the brain cannot rewrite the thing that decides what it may rewrite.
#:
#: MOVED TO MODULE SCOPE in the same change, and that is not cosmetic. It was a local inside main(),
#: so `tests/test_self_mod_safety.py` had to PARSE it out of the source with a regex over
#: `PROTECTED = \{(.*?)\}` -- the test's own docstring called the parser "the price of checking it
#: at all". It is importable now, so the parser is deleted rather than kept beside it (P2: a merge
#: must reduce). The staleness check in main() that this set exists for is unchanged and still
#: derives its own side from the filesystem.
PROTECTED = frozenset((*_ENVELOPE, "reasoning/self_improve.py"))


class Change:
    """A proposed self-modification. `targets` are the files it would touch (checked against the
    protected set BEFORE it is tried); `apply` mutates an isolated copy of the system state."""

    def __init__(self, name, targets, apply, note=""):
        self.name = name
        self.targets = list(targets)
        self.apply = apply  # apply(state) -> mutates the isolated copy in place
        self.note = note


class SelfImprover:
    """Propose -> isolate -> verify -> ratchet -> accept-or-rollback, with corrigibility.

    verify(state)  -> bool : True iff EVERY gate is green AND no behavioural probe regressed.
    fitness(state) -> float: a measured metric (higher = better) the brain does NOT author.
    protected      -> set  : files that must never be modified (safety envelope + this verifier).
    """

    def __init__(self, state, verify, fitness, protected):
        self._state = copy.deepcopy(state)
        self._verify = verify
        self._fitness = fitness
        self._protected = set(protected)
        if not self._verify(self._state):
            raise ValueError("refusing to start: the baseline does not pass its own gates")
        self._fit = self._fitness(self._state)
        self.log = []  # (change name, outcome, reason, fitness-after) -- the audit trail

    @property
    def state(self):
        return copy.deepcopy(self._state)

    @property
    def fitness(self):
        return self._fit

    def propose(self, change):
        """Try ONE change. Returns (accepted: bool, reason). The live state changes ONLY on accept."""
        # (1) CORRIGIBILITY: a change that would touch a protected file is refused before it is tried.
        clash = self._protected.intersection(change.targets)
        if clash:
            return self._record(
                change, False, f"refused: touches protected file(s) {sorted(clash)}"
            )

        # (2) ISOLATE: work on a deep copy; the live state is never mutated during the trial.
        trial = copy.deepcopy(self._state)
        try:
            change.apply(trial)
        except Exception as e:  # a proposal that crashes on apply is discarded, live tree intact
            return self._record(change, False, f"rejected: apply raised {type(e).__name__}: {e}")

        # (3) VERIFY: every gate must stay green + no behavioural probe may regress.
        if not self._verify(trial):
            return self._record(
                change, False, "rejected: a gate went red (would regress the codebase)"
            )

        # (4) RATCHET: accept ONLY a STRICT improvement -- never merge a change that does not help.
        new_fit = self._fitness(trial)
        if new_fit <= self._fit:
            return self._record(
                change, False, f"rejected: fitness {new_fit} !> current {self._fit} (no gain)"
            )

        # (5) ACCEPT: merge the isolated copy, advance the monotonic fitness ratchet.
        self._state, self._fit = trial, new_fit
        return self._record(change, True, f"accepted: green + fitness {self._fit}")

    def run(self, changes):
        """Feed a stream of proposals; the fitness ratchet only ever climbs."""
        for change in changes:
            self.propose(change)
        return self._fit

    def _record(self, change, accepted, reason):
        self.log.append((change.name, accepted, reason, self._fit))
        return accepted, reason


def _self_pkg_relpath():
    """This module's own path RELATIVE TO THE PACKAGE ROOT, derived from `__file__` by walking up
    past `__init__.py` -- never written down.

    This exists so the PROTECTED check below can go RED. The obvious form of that check,
    `assert "reasoning/self_improve.py" in PROTECTED`, is true by construction: PROTECTED is a
    literal set in the same function, so the assert restates the line that created it and no input
    can make it fail. Deriving the path from the FILESYSTEM makes the two sides independent -- the
    literal is what a human typed, this is where the file actually is -- so renaming or moving this
    module reddens it. That is exactly the failure that shipped here once: the entry read
    `research/self_improve.py` after the tree moved, so the self-modifier was not protecting its own
    proposer, and nothing went red for it.
    """
    directory, base = os.path.split(os.path.abspath(__file__))
    parts = [base]
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory, name = os.path.split(directory)
        parts.append(name)
    parts.pop()  # drop the package root itself -- entries are package-RELATIVE
    return "/".join(reversed(parts))


def self_map(graph_path="graphify-out/graph.json"):
    """The brain's model of ITSELF: how many nodes/edges its own architecture graph has. Reading its
    own graphify graph is the 'knows its own codebase' half -- a proposal generator queries this to
    find under-connected / god-node hotspots to refactor. Returns {} if the graph is absent."""
    if not os.path.exists(graph_path):
        return {}
    with open(graph_path, encoding="utf-8") as f:
        g = json.load(f)
    # networkx node-link format stores edges under "links" (graphify's format), not "edges"
    return {"nodes": len(g.get("nodes", [])), "edges": len(g.get("links") or g.get("edges", []))}


# ---------------------------------------------------------------- the protocol self-check ----------


def main():
    print("=" * 96)
    print(
        "SAFE BOUNDED SELF-IMPROVEMENT -- propose -> isolate -> verify -> ratchet -> accept/rollback"
    )
    print("=" * 96)

    # A TOY 'codebase state': tunable params + an invariant every version must preserve (the 'gate').
    # fitness = a measured accuracy; the invariant models "the self-checks stay green".
    base = {"threshold": 0.5, "cache": True, "accuracy": 0.70, "invariant_ok": True}

    def verify(state):
        return bool(state["invariant_ok"])  # stands in for: 224/224 green + no probe regressed

    def fitness(state):
        return round(state["accuracy"], 6)

    imp = SelfImprover(base, verify, fitness, PROTECTED)
    start_fit = imp.fitness

    def tune(state):  # a genuine, safe improvement (green + higher accuracy)
        state["threshold"], state["accuracy"] = 0.55, 0.78

    def regress(state):  # a change that BREAKS an invariant (a would-be bug)
        state["accuracy"], state["invariant_ok"] = 0.99, False

    def noop_gain(state):  # green but no fitness gain -> must be rejected
        state["cache"] = False

    def touch_safety(state):  # would improve, but touches a protected safety file
        state["accuracy"] = 0.95

    def tune_more(state):  # a second real improvement, stacks on the first
        state["accuracy"] = 0.83

    a1 = imp.propose(Change("tune-threshold", ["cortex/readout.py"], tune))
    a2 = imp.propose(Change("faster-but-broken", ["substrate/binlog.py"], regress))
    a3 = imp.propose(Change("cosmetic-noop", ["cortex/readout.py"], noop_gain))
    a4 = imp.propose(Change("boost-via-reflex", ["brainstem/reflex.py"], touch_safety))
    a5 = imp.propose(Change("tune-more", ["cortex/readout.py"], tune_more))

    for name, (ok, why) in [
        ("tune-threshold", a1),
        ("faster-but-broken", a2),
        ("cosmetic-noop", a3),
        ("boost-via-reflex", a4),
        ("tune-more", a5),
    ]:
        print(f"  [{'ACCEPT' if ok else 'REJECT'}] {name:20s} -> {why}")
    print(f"\n  fitness ratchet: {start_fit} -> {imp.fitness}   (self-map: {self_map()})")

    # ---- gated self-check: the safety protocol holds ----
    assert a1[0] is True, "a green, strictly-improving change must be ACCEPTED"
    assert a2[0] is False and imp.state["invariant_ok"] is True, (
        "a change that reddens a gate must be REJECTED and rolled back (live invariant intact)"
    )
    assert imp.state["accuracy"] != 0.99, (
        "the rejected regression must NOT have leaked into the live state"
    )
    assert a3[0] is False, "a green change with NO fitness gain must be rejected (only ENHANCE)"
    assert a4[0] is False and "protected" in a4[1], (
        "a change touching a safety file must be refused (corrigibility)"
    )
    assert a5[0] is True and imp.fitness > start_fit, "improvements must stack; the ratchet climbs"
    # MONOTONICITY: fitness never decreased across the whole run
    fits = [start_fit] + [row[3] for row in imp.log]
    assert all(b >= a for a, b in zip(fits, fits[1:])), (
        "fitness must be MONOTONIC non-decreasing (no regression ever)"
    )
    # The verifier must be un-writable -- and this check has to be able to FAIL (P8 item 1).
    # It used to read `assert "reasoning/self_improve.py" in PROTECTED`, which is a tautology: the
    # literal on the left was typed into the literal set on the right fifty lines above, so no
    # input could redden it. The DERIVED form below compares the set against the filesystem, so
    # moving or renaming this module fires it -- which is precisely the state the tree was in
    # before 2026-08-01, when the entry named a `research/` path that no longer existed.
    #
    # THE RED WAS CONSTRUCTED AND RUN, 2026-08-05, not argued from the prose above. This module's
    # source was planted into a throwaway package tree at three locations and main() executed from
    # each:
    #   <pkg>/reasoning/self_improve.py    -> relpath 'reasoning/self_improve.py'    -> PASSED
    #   <pkg>/research/self_improve.py     -> relpath 'research/self_improve.py'     -> RED
    #   <pkg>/reasoning/self_improve_v2.py -> relpath 'reasoning/self_improve_v2.py' -> RED
    # The first is the positive control (the harness can pass at all); the second is the exact
    # pre-2026-08-01 state; the third is a rename the self-builder could itself propose.
    me = _self_pkg_relpath()
    assert me in PROTECTED, (
        f"the brain must not be able to rewrite its own verifier: this module is at {me!r}, which "
        f"is NOT in PROTECTED {sorted(PROTECTED)} -- the entry is stale, so the file is unprotected"
    )
    # TWO "RED CONTROL" asserts stood here until 2026-08-05 and were DELETED, not weakened (F11),
    # because neither could be reddened by any input:
    #   `assert "research/self_improve.py" not in PROTECTED` -- both operands are literals in THIS
    #     function, so it restates its own input. That is the same tautology shape as the assert it
    #     was added alongside to replace.
    #   `assert _self_pkg_relpath() != "research/self_improve.py"` -- DOMINATED. It can only fire
    #     when the relpath IS the research/ path, and in that state the assert above has already
    #     fired: proven by the planted-at-research/ run, whose AssertionError was the `me in
    #     PROTECTED` message, never this one.
    # Deleting them is the point of F11: a control that cannot fail is not a control, and leaving
    # it in place makes this block LOOK better covered than the one real assert it surrounds.

    print(
        "\nSELF-CHECK PASSED: bounded self-improvement -- accepts only green, strictly-improving, "
        "non-safety changes; rolls back regressions; monotonic fitness; refuses to touch its own "
        "safety envelope or verifier. Honest ceiling: verifiable ratcheted enhancement, NOT "
        "unbounded recursive self-improvement or 'singularity'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
