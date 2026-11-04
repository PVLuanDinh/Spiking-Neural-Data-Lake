"""The gradient-claim census must classify every rule, and must never bluff about one.

Six properties, each pinned by a CONSTRUCTED case rather than the happy path. The gate carries its
own `--self-check` (Standing Invariant 5), and these tests are deliberately NOT a copy of it: the
self-check proves the reds fire when the gate runs itself, these prove the same properties hold
through the module's public surface, where a later refactor could quietly move them.

Two defects found during implementation motivated half of this file, both measured 2026-08-17:

1. The first enumeration scored 242 governed rules and included `brainstem/reflex.py::Reflex.step`,
   because the verb list carried `step` and the weight-name test matched by PREFIX -- so `b` caught
   `bound_areas`, `_spectral_bound`, `buf` and `best`. A gate demanding a declaration inside the
   E-STOP path is a gate that demands editing the safety envelope, which FR-010 forbids.

2. `unclassified == 0` is satisfied by a DEAD WALK just as well as by a complete one. The liveness
   floor is the only thing separating them, and it is the easiest bound to omit.
"""

import importlib
import json

import pytest

gate = importlib.import_module("snn_data_lake.gates.gradient_claim_gate")

_PROBE = "faculties/zz_probe_rule.py"


def _states(rules, module_rel):
    return sorted({r["state"] for r in rules if r["module_rel"] == module_rel})


def test_a_local_rule_owes_nothing_and_the_reference_is_not_a_claimant():
    """A declared LOCAL rule owes no cosine; EXACT_GRADIENT is the reference, not a claimant.

    Both dispositions are exercised through a real walk of the real tree with ONE file overlaid, so
    the classification is derived from the tree rather than from a hand-maintained list -- which is
    FR-001's actual requirement. The REAL modules' dispositions are pinned when their declarations
    land (T031/T032); asserting them here would only pin the absence of a declaration.
    """
    local = (
        f'{gate.DECL_NAME} = {{"Stdp.learn": ("{gate.LOCAL}", "pair STDP, no claim")}}\n'
        "class Stdp:\n"
        "    def learn(self, pre, post):\n"
        "        self.w += pre * post\n"
    )
    rules, _, _ = gate.census(overlay={_PROBE: local})
    assert _states(rules, _PROBE) == [gate.CLASSIFIED_LOCAL], _states(rules, _PROBE)

    reference = (
        f'{gate.DECL_NAME} = {{"Net.grad_bptt": ("{gate.EXACT_GRADIENT}", "reverse-mode ref")}}\n'
        "class Net:\n"
        "    def grad_bptt(self, x):\n"
        '        """Reverse-mode gradient, computed exactly."""\n'
        "        self.w -= x\n"
    )
    rules, _, _ = gate.census(overlay={_PROBE: reference})
    got = _states(rules, _PROBE)
    assert got == [gate.CLASSIFIED_REFERENCE], got
    assert gate.CLASSIFIED_CLAIMANT not in got, (
        "the reference gradient must not be filed as a claimant"
    )


#: The RECALL CONTROL. `research.md` R1 asked for "a hand-labelled sample scored for precision AND
#: recall" and recorded it as not done; this is that control in mechanical form, restricted to the
#: cases the feature cannot function without. It exists because tuning PRECISION silently destroyed
#: RECALL once already: dropping `grad` from the verb list to cure a 242-rule over-count made
#: `plasticity/eprop_kernel.py` -- the e-prop reference gradient itself -- vanish from the census,
#: and every gate stayed green while it was invisible.
_MUST_BE_ENUMERATED = (
    "plasticity/eprop_kernel.py",  # the reference gradient FR-011 reconciles; 17 defs, all unseen
    "sensorimotor/eprop_shd.py",  # shipped encoder, guards itself on its own kappa-filtered form
    "sensorimotor/eprop_nmnist.py",  # shipped encoder
    "faculties/mlp_q.py",  # the feedback-alignment case SC-003 turns on
    "faculties/jepa_world_model.py",  # the closed transport violation
    "readout/predictive_coding.py",  # inference ran every tick while learning never did
    "reasoning/eprop.py",  # hand-rolled BPTT in pure loops; invisible to any .T-based walk
    "reasoning/gf.py",  # the owner-EXEMPT CONTROL that a two-value classifier would condemn
)


def test_every_load_bearing_module_stays_visible_to_the_census():
    """A rule the enumeration cannot see is a rule no bound can ever govern.

    This is the test that was missing when `plasticity/eprop_kernel.py` disappeared: the census
    reported 141 governed rules, `--assert` passed, `--self-check` passed, and the module the whole
    feature exists to measure was not in it. Nothing was red, because nothing was looking.
    """
    rules, _, _ = gate.census()
    seen = {r["module_rel"] for r in rules}
    missing = [m for m in _MUST_BE_ENUMERATED if m not in seen]
    assert not missing, (
        f"the enumeration lost load-bearing module(s): {missing}. A rule the census cannot "
        "see is a rule no bound governs, and every gate stays green while it is invisible."
    )
    # NEGATIVE CONTROL: membership must be able to answer NO, or the assertion above is satisfied by
    # a set that contains everything.
    assert "faculties/zz_not_a_real_module.py" not in seen, (
        "the census claims to contain a module that does not exist -- membership proves nothing"
    )
    # POSITIVE CONTROL: and it must answer YES for something outside the list, or `seen` is just the
    # list itself echoed back.
    assert len(seen - set(_MUST_BE_ENUMERATED)) > 0, (
        "the census sees only the modules this test names"
    )


def test_not_a_rule_is_reachable_only_by_declaration():
    """A fifth disposition is a drawer waiting to happen. This is the guard that it is not one.

    `UNDETERMINABLE` is ratcheted DOWN precisely because a state everything difficult can fall into
    stops meaning anything. `NOT_A_RULE` avoids that by construction rather than by a bound: the
    classifier cannot produce it. Only a human declaration carrying a reason can.
    """
    # The probe name MUST enumerate on its own, or the undeclared half of this test asserts over an
    # empty list. Found by mutation: `solve_unbounded` has no rule-verb and mutates no weight, so
    # undeclared it was never enumerated -- the declared half passed only because `enumerate_rules`
    # always includes a declared symbol, and the guard below was checking nothing.
    src = "class G:\n    def learn(self, n):\n        return n\n"
    rules, _, _ = gate.census(overlay={_PROBE: src})
    undeclared = [r["state"] for r in rules if r["module_rel"] == _PROBE]
    assert undeclared, "the undeclared probe was not enumerated -- this test would prove nothing"

    # `infer()` has exactly two possible answers, and NOT_A_RULE is not among them.
    for r in rules:
        assert r["inferred"] in ("CLAIM", "UNKNOWN"), f"infer() produced {r['inferred']}"

    # Undeclared, a symbol falls to UNDETERMINABLE or MISSING_DECLARATION -- never NOT_A_RULE.
    assert gate.CLASSIFIED_NOT_A_RULE not in undeclared, (
        f"the classifier swept a symbol into NOT_A_RULE without a declaration: {undeclared}"
    )

    # POSITIVE CONTROL: a DECLARED NOT_A_RULE must reach the state, or the check above is satisfied
    # by a value nothing can ever produce.
    declared = (
        f'{gate.DECL_NAME} = {{"G.learn": ("{gate.NOT_A_RULE}", "search, not a rule")}}\n'
    ) + src
    rules, _, _ = gate.census(overlay={_PROBE: declared})
    got = _states(rules, _PROBE)
    assert got == [gate.CLASSIFIED_NOT_A_RULE], got

    # And it must NOT be counted as debt: UNDETERMINABLE is ratcheted down, NOT_A_RULE is an answer.
    assert gate.NOT_A_RULE in gate.DISPOSITIONS
    assert gate._STATE_OF[gate.NOT_A_RULE] == gate.CLASSIFIED_NOT_A_RULE


def test_ng_snn_fit_is_direct_solve_and_not_lumped_in_as_local():
    """POLICIES.md forbids the lump-in that `LOCAL` would perform here.

    It adjudicates `growth/ng_snn.py::fit` as "not a violation ... a direct solve: no gradient
    is propagated through a network and no weight is transported. It IS non-local, and the policy
    should say so DISTINCTLY rather than lumping it in". `LOCAL` asserts locality, so using it would
    state something the authoritative document explicitly denies.
    """
    rules, _, _ = gate.census()
    fit = next((r for r in rules if r["qualname"] == "growth/ng_snn.py::NGSNN.fit"), None)
    assert fit is not None, "ng_snn::fit is not in the census at all"
    assert fit["state"] == gate.CLASSIFIED_DIRECT_SOLVE, fit["state"]
    assert fit["declared"] == gate.DIRECT_SOLVE, fit["declared"]

    # SCOPE GUARD: DIRECT_SOLVE must not become a general non-local bucket. predictive_coding stays
    # LOCAL -- POLICIES.md records its weight update as "a local Hebbian outer product ... No
    # transposed weight appears in the learning rule at all"; the w1.T is inference, not the rule.
    pc = [r for r in rules if r["module_rel"] == "readout/predictive_coding.py"]
    assert pc, "predictive_coding vanished from the census"
    assert all(r["declared"] == gate.LOCAL for r in pc), [r["declared"] for r in pc]

    # CONTROL: the two values must be distinguishable, or the distinction is decorative.
    assert gate.LOCAL != gate.DIRECT_SOLVE
    assert gate._STATE_OF[gate.LOCAL] != gate._STATE_OF[gate.DIRECT_SOLVE]


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_a_non_importing_module_is_still_classified():
    """SC-010. A classifier that skips what it cannot import silently UNDER-COUNTS.

    The non-importing set is DERIVED here, not hardcoded: the test imports every module carrying a
    rule and keeps the failures. If that set is ever empty the test fails loudly rather than passing
    vacuously, because then it would be asserting nothing.

    THE IMPORT SWEEP RUNS IN ONE SUBPROCESS, not in-process -- found 2026-08-20 by the T016 GIL
    assert: one rule-carrying module imports `pygenn._genn`, which has not declared free-threading
    support, so CPython RE-ENABLES THE GIL for the rest of the interpreter -- and there is no API
    to disable it again. In-process, this test silently de-parallelized every test that ran after
    it (each full suite run since the rule landed executed its tail GIL'd, invalidating whatever
    thread behaviour that tail measured) and, once make_brain refused GIL'd interpreters, it broke
    eight downstream constructions. The child interpreter absorbs the contamination and reports
    the same importable/not verdicts over stdout.
    """
    import subprocess
    import sys

    # del-reasoning 1b-1h deleted every real rule-carrying module that failed to import (arc_agi,
    # ng_snn -> arcengine / retired siblings), so the non-importing member is CONSTRUCTED: an
    # overlay module is in the census but absent from disk, hence unimportable by the child below.
    claimant = (
        "class Rule:\n"
        "    def learn(self, x):\n"
        '        """Approximates the gradient by backprop."""\n'
        "        self.w -= x\n"
    )
    rules, _, _ = gate.census(overlay={"reasoning/zz_probe_unimportable.py": claimant})
    mods = sorted({r["module_rel"] for r in rules})
    probe = (
        "import importlib, json, sys, contextlib, io\n"
        "bad = []\n"
        "for m, name in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        buf = io.StringIO()\n"
        "        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):\n"
        "            importlib.import_module(name)\n"
        "    except BaseException:\n"
        "        bad.append(m)\n"
        "print('NOT_IMPORTABLE ' + json.dumps(bad))\n"
    )
    pairs = json.dumps([[m, gate.modname(m)] for m in mods])
    proc = subprocess.run(
        [sys.executable, "-c", probe, pairs], capture_output=True, text=True, timeout=1200
    )
    line = next(
        (ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("NOT_IMPORTABLE ")),
        None,
    )
    assert line is not None, (
        f"the import-probe child produced no verdict (rc={proc.returncode}).\n"
        f"stderr tail: {proc.stderr[-400:]}"
    )
    not_importable = json.loads(line[len("NOT_IMPORTABLE ") :])

    assert not_importable, (
        "no module carrying a rule fails to import, so this test cannot demonstrate SC-010. "
        "Either the tree changed or the census stopped seeing the ungoverned subtree."
    )
    assert "reasoning/zz_probe_unimportable.py" in not_importable, not_importable
    for m in not_importable:
        assert any(r["module_rel"] == m for r in rules), (
            f"{m} does not import AND is absent from the census"
        )


def test_update_baseline_refuses_a_rise():
    """SC-011, demonstrated by CONSTRUCTION rather than asserted in prose.

    Measured on the existing `p6_gate`: a ceiling raised together with the regression it excuses
    passes BOTH `--assert` and `--self-check`. That is what separates a ratchet from a rubber stamp.
    """
    baseline = {"counts": {gate.MISSING_DECLARATION: 5, gate.UNDETERMINABLE: 9}}

    worse = {gate.MISSING_DECLARATION: 6, gate.UNDETERMINABLE: 9}
    assert gate._refuses_rise(worse, baseline) is True, (
        "a rise in MISSING_DECLARATION must be refused"
    )

    worse_u = {gate.MISSING_DECLARATION: 5, gate.UNDETERMINABLE: 10}
    assert gate._refuses_rise(worse_u, baseline) is True, (
        "a rise in UNDETERMINABLE must be refused too"
    )

    # CONTROLS: the refusal must be able to say NO as well as YES, or it is a constant.
    same = {gate.MISSING_DECLARATION: 5, gate.UNDETERMINABLE: 9}
    assert gate._refuses_rise(same, baseline) is False, "an unchanged count must be writable"
    better = {gate.MISSING_DECLARATION: 0, gate.UNDETERMINABLE: 0}
    assert gate._refuses_rise(better, baseline) is False, "a ratchet DOWN must be writable"


def test_a_dead_walk_fails_loudly():
    """SC-007 / FR-006. An empty census and a broken walk look identical from the counts alone.

    `unclassified == 0` is TRUE of a walk that found nothing, so the liveness floor is the one bound
    that separates a complete census from a dead one.
    """
    live, _, n_found = gate.census()
    floor = gate.counts(live)["rules_seen"]
    assert floor > 0 and n_found > 0, (
        "the control census is empty; everything below would be vacuous"
    )

    baseline = {"counts": {}, "rules_seen_floor": floor}
    assert gate.evaluate(gate.counts(live), baseline, n_found) == [], (
        "the unmutated tree must be clean"
    )

    # Delete the module contributing the most rules; rules_seen must FALL and the floor must fire.
    victim = max(
        {r["module_rel"] for r in live if r["governed"]},
        key=lambda m: sum(1 for r in live if r["module_rel"] == m),
    )
    dead, _, n_dead = gate.census(overlay={victim: None})
    c_dead = gate.counts(dead)
    assert c_dead["rules_seen"] < floor, f"deleting {victim} did not lower rules_seen"
    assert c_dead["unclassified"] == 0, (
        "a dead walk still reports zero unclassified -- that is the point"
    )
    findings = gate.evaluate(c_dead, baseline, n_dead)
    assert findings, "the liveness floor did not fire on a walk that lost a whole module"
    assert any("LIVENESS" in f for f in findings), findings


def test_an_ungoverned_finding_is_reported_but_never_asserted():
    """FR-001a. The census walks the WHOLE tree; the bounds assert on the GOVERNED tier only.

    Asserting over both would gate this feature on remediating `reasoning/` (~108 modules), a tree
    it does not own.
    """
    claimant = (
        "class Rule:\n"
        "    def learn(self, x):\n"
        '        """Approximates the gradient by backprop."""\n'
        "        self.w -= x\n"
    )
    rules, _, n_found = gate.census(overlay={"reasoning/zz_probe_ungoverned.py": claimant})
    hit = [r for r in rules if r["module_rel"] == "reasoning/zz_probe_ungoverned.py"]
    assert hit, "the ungoverned probe never reached the census -- the walk is not whole-tree"
    assert all(r["governed"] is False for r in hit), "a reasoning/ module must not be governed"
    assert any(r["state"] == gate.MISSING_DECLARATION for r in hit), [r["state"] for r in hit]

    gov = gate.counts(rules, governed_only=True)
    allr = gate.counts(rules, governed_only=False)
    assert allr[gate.MISSING_DECLARATION] > gov[gate.MISSING_DECLARATION], (
        "the ungoverned finding is invisible in the all-tier count, so it is not being REPORTED"
    )
    baseline = {
        "counts": {gate.MISSING_DECLARATION: gov[gate.MISSING_DECLARATION]},
        "rules_seen_floor": 0,
    }
    assert gate.evaluate(gov, baseline, n_found) == [], (
        "an ungoverned finding must not redden the gate"
    )


def test_a_ratchet_at_its_floor_reads_SATISFIED_not_merely_green():
    """SC-018 / FR-017a. A bound resting at its floor and a bound nobody exercised both print green.

    Standing Invariant 5 calls that shape worse than no gate: it manufactures confidence. The
    success state must be recognisable as success rather than as silence.
    """
    assert gate.ratchet_state(0, 0) == "SATISFIED", (
        "a count at its floor is the success state and must say so"
    )
    assert gate.ratchet_state(0, 7) == "SATISFIED", (
        "zero is the floor regardless of the recorded allowance"
    )

    # CONTROLS: the three states must be distinguishable, or the label carries no information.
    assert gate.ratchet_state(3, 7) == "GREEN", (
        "a live bound below its allowance is GREEN, not SATISFIED"
    )
    assert gate.ratchet_state(7, 7) == "GREEN", "at the allowance but above the floor is still live"
    assert gate.ratchet_state(8, 7) == "RED", "a rise must be RED"
    assert len({gate.ratchet_state(0, 0), gate.ratchet_state(3, 7), gate.ratchet_state(8, 7)}) == 3


def test_the_selfcheck_constructs_its_own_reds_and_says_so():
    """`--self-check` is half this module's guard and was reachable ONLY through the CLI.

    Measured before this test existed: coverage 73%, with the whole self-check body and `main()`
    uncovered. That is the shape the self-check itself exists to forbid -- a guard nothing verifies.
    Running it here puts it under pytest too, so a refactor that silently neuters a red is caught
    by the suite and not only by someone remembering to run the flag.
    """
    assert gate._selfcheck() is True, "the gate's own self-check does not pass on the current tree"


def test_main_exit_codes_across_every_mode(capsys, tmp_path, monkeypatch):
    """`main()` is the CI entry point; every branch of it decides whether a build goes red.

    Report mode must NEVER fail -- that is what makes it safe to run anywhere, and it is the sibling
    gates' contract too. The other modes must be able to return BOTH 0 and 1, or their exit code
    carries no information.
    """
    assert gate.main([]) == 0, "report mode must exit 0 regardless of findings"
    out = capsys.readouterr().out
    assert "GRADIENT-CLAIM CENSUS" in out
    assert "wall-clock" in out, "FR-013a requires the cost be PRINTED"
    assert "NEVER asserted" in out, "the output must say the cost gates nothing"

    assert gate.main(["--assert"]) == 0, "the current tree is within every bound"
    assert gate.main(["--self-check"]) == 0
    capsys.readouterr()

    # --assert with NO baseline must fail loudly rather than pass on an absent floor.
    monkeypatch.setattr(gate, "load_baseline", lambda *a, **k: None)
    assert gate.main(["--assert"]) == 1, "a missing baseline must not read as a pass"
    assert "no baseline" in capsys.readouterr().out


def test_update_baseline_writes_a_machine_generated_file_and_refuses_a_worse_one(
    capsys, tmp_path, monkeypatch
):
    """FR-018 through the REAL writer, not just the pure predicate.

    `test_update_baseline_refuses_a_rise` exercises `_refuses_rise` in isolation; this drives the
    path that actually writes, because a refusal that the writer never consults would pass that test
    and still launder a regression.
    """
    target = tmp_path / "gradient_claim_baseline.json"
    monkeypatch.setattr(gate, "_baseline_path", lambda *a, **k: str(target))

    monkeypatch.setattr(gate, "load_baseline", lambda *a, **k: None)
    # `--bootstrap` is REQUIRED here: with no baseline there is nothing to ratchet against, so this
    # creates one. See test_creating_a_baseline_from_nothing_demands_a_written_reason.
    assert gate.main(["--update-baseline", "--bootstrap", "test fixture"]) == 0
    assert target.exists(), "no baseline file was written"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["_generated_by"].endswith("--update-baseline"), payload
    assert set(payload["counts"]) == {gate.MISSING_DECLARATION, gate.UNDETERMINABLE}, payload
    assert payload["rules_seen_floor"] > 0, "a floor of zero would make the liveness bound vacuous"
    capsys.readouterr()

    # Now make the refusal FIRE and prove the writer honours it. Until 2026-08-17 this planted a
    # baseline of {0, 0} and relied on the live counts exceeding it; once both ratchets reached 0 --
    # their floor -- no rise could be built from the real tree and the test went green by absorbing
    # nothing. The property it exists for is the WIRING: `_refuses_rise` is CONSULTED and its answer
    # aborts the write. The predicate's own arithmetic is covered against real counts by
    # test_update_baseline_refuses_a_rise.
    before = target.read_text(encoding="utf-8")
    consulted = []
    monkeypatch.setattr(
        gate,
        "load_baseline",
        lambda *a, **k: {"counts": {gate.MISSING_DECLARATION: 0, gate.UNDETERMINABLE: 0}},
    )
    monkeypatch.setattr(gate, "_refuses_rise", lambda c, b: consulted.append((c, b)) or True)
    assert gate.main(["--update-baseline"]) == 1, "a rise must be REFUSED, not absorbed"
    assert consulted, "the writer never consulted _refuses_rise -- the refusal is decorative"
    assert "REFUSED" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8") == before, "the refused write still changed the file"

    # Control IN THE SAME TEST: flip the refusal to False and the identical argv must WRITE. Without
    # it, a `main()` that returned 1 unconditionally would satisfy every assertion above.
    monkeypatch.setattr(gate, "_refuses_rise", lambda c, b: False)
    assert gate.main(["--update-baseline"]) == 0, "a non-rise must be absorbed, not refused"
    assert target.read_text(encoding="utf-8") != before, "the accepted write changed nothing"
    capsys.readouterr()


def test_creating_a_baseline_from_nothing_demands_a_written_reason(capsys, tmp_path, monkeypatch):
    """The delete-and-re-derive seam: `_refuses_rise` is VACUOUS when there is nothing to compare.

    Deleting the file made the refusal a no-op, so a rise could be laundered in silence. Creating a
    baseline where none exists now requires `--bootstrap "<reason>"`, and the reason is stored. This
    does not prevent a determined bypass -- nothing short of signing would -- it removes the SILENT
    one, which is the difference between an invisible act and an attributable one.
    """
    target = tmp_path / "gradient_claim_baseline.json"
    monkeypatch.setattr(gate, "_baseline_path", lambda *a, **k: str(target))
    monkeypatch.setattr(gate, "load_baseline", lambda *a, **k: None)

    assert gate.main(["--update-baseline"]) == 1, "creating a baseline from nothing must be refused"
    assert not target.exists(), "the refused bootstrap still wrote a file"
    assert "--bootstrap" in capsys.readouterr().out

    assert gate.main(["--update-baseline", "--bootstrap", "census widened, not a regression"]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["_bootstrap_reason"] == "census widened, not a regression", payload
    capsys.readouterr()

    # `--bootstrap=X` must work too, or half the callers hit the refusal for a formatting reason.
    target.unlink()
    assert gate.main(["--update-baseline", "--bootstrap=equals form"]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["_bootstrap_reason"] == "equals form"
    capsys.readouterr()

    # The reason must SURVIVE a later ratchet-down, or it evaporates on the next honest write
    # and the file goes back to looking unexplained.
    monkeypatch.setattr(
        gate,
        "load_baseline",
        lambda *a, **k: {
            "counts": {gate.MISSING_DECLARATION: 9999, gate.UNDETERMINABLE: 9999},
            "_bootstrap_reason": "the original reason",
        },
    )
    assert gate.main(["--update-baseline"]) == 0, "a genuine ratchet DOWN must still be writable"
    assert (
        json.loads(target.read_text(encoding="utf-8"))["_bootstrap_reason"] == "the original reason"
    )


def test_a_repoless_walk_reports_empty_and_the_liveness_bound_catches_it(monkeypatch):
    """Under an installed wheel there is no repo root, so the walk can find nothing.

    The danger is that an empty census satisfies `unclassified == 0` perfectly. This pins that the
    empty result is REACHABLE and that the liveness bound is what turns it into a failure.
    """
    monkeypatch.setattr(gate, "_repo_root", lambda: None)
    trees, failures, n_found = gate.walk()
    assert (trees, failures, n_found) == ({}, [], 0), "a repoless walk must report empty, not guess"

    rules, _, n = gate.census()
    c = gate.counts(rules)
    assert c["rules_seen"] == 0 and c["unclassified"] == 0, (
        "the empty census looks perfect -- that is the trap"
    )
    findings = gate.evaluate(c, {"counts": {}, "rules_seen_floor": 1}, n)
    assert any("LIVENESS" in f for f in findings), findings


def test_every_failure_branch_of_evaluate_is_reachable():
    """The gate's own FAIL verdicts, driven one at a time. Found by reading COVERAGE, not code.

    At 92% module coverage the 29 uncovered lines were almost all failure branches -- uncovered for
    the worst possible reason: the tree is healthy, so nothing ever produced them. That is the shape
    `criteria measure the pathology` names. A gate whose RED message has never once been constructed
    could carry a wrong key, a swapped count or a stale word and no green run would ever show it.

    `evaluate()` is PURE -- no I/O -- so every branch is reachable from a constructed
    dict, and there is no excuse for leaving them unexercised. Each case perturbs exactly ONE
    field away from a passing baseline, so a finding can only come from the field under test.
    """
    ok = {
        "unclassified": 0,
        gate.MISSING_DECLARATION: 0,
        gate.UNDETERMINABLE: 0,
        "rules_seen": 149,
    }
    base = {
        "counts": {gate.MISSING_DECLARATION: 0, gate.UNDETERMINABLE: 0},
        "rules_seen_floor": 149,
    }

    # GREEN CONTROL FIRST. Without it, an `evaluate()` that returned a finding unconditionally would
    # satisfy every case below.
    assert gate.evaluate(ok, base, 379) == [], "the passing baseline must produce NO findings"

    cases = (
        ("unclassified", {"unclassified": 3}, 379, "unclassified", "ABSOLUTE"),
        ("missing rises", {gate.MISSING_DECLARATION: 1}, 379, gate.MISSING_DECLARATION, "RATCHET"),
        ("undeterminable rises", {gate.UNDETERMINABLE: 7}, 379, gate.UNDETERMINABLE, "RATCHET"),
        ("below the floor", {"rules_seen": 148}, 379, "rules_seen", "LIVENESS"),
        ("walk found nothing", {}, 0, "no files", "LIVENESS"),
    )
    for name, delta, n_found, needle, kind in cases:
        c = dict(ok, **delta)
        findings = gate.evaluate(c, base, n_found)
        assert findings, f"{name}: the branch produced NO finding -- its red is unreachable"
        named = f"{name}: no finding names {needle!r}: {findings}"
        assert any(needle in f for f in findings), named
        assert any(kind in f for f in findings), f"{name}: no finding says {kind!r}: {findings}"

    # A ratchet may FALL freely -- that is the whole point of it being a ratchet and not a bound.
    below = {"counts": {gate.MISSING_DECLARATION: 12}, "rules_seen_floor": 1}
    lower = gate.evaluate(ok, below, 379)
    assert lower == [], f"a count BELOW its baseline must not be a finding: {lower}"


def test_the_safety_envelope_is_classified_but_never_asked_to_change():
    """FR-010 / Standing Invariant 7, found by MEASUREMENT rather than review.

    The first census enumerated `brainstem/reflex.py::Reflex.step` as a learning rule. A
    MISSING_DECLARATION finding there has exactly one remedy -- edit the file -- and the five
    brainstem files are modified only with explicit approval. They stay COUNTED so it cannot
    vanish from the census, but they can never produce that finding.
    """
    rules, _, _ = gate.census()
    env = [r for r in rules if r["module_rel"] in gate._SAFETY_ENVELOPE]
    # NO real-tree assertion that the envelope is PRESENT. This test used to require it, on the
    # reasoning that the envelope "must not vanish from the census". That premise was wrong: the
    # envelope contains no learning rules, and its one enumerated symbol was
    # `serial_arm::assert_calibrated` -- a safety precondition check caught on the `calibrate`
    # substring, i.e. a false positive. Once `calibrate` was dropped on measured 0% precision the
    # envelope left the census, which is CORRECT for a learning-rule census. Requiring its presence
    # would have pinned a false positive in place as if it were coverage.
    assert all(r["state"] != gate.MISSING_DECLARATION for r in env), [
        r["qualname"] for r in env if r["state"] == gate.MISSING_DECLARATION
    ]

    # The overlay half below is what CARRIES this test, and it is unaffected by the envelope being
    # empty on the real tree -- it CONSTRUCTS the case rather than relying on one existing. That
    # matters twice: mutation testing showed the earlier real-tree version survived deleting the
    # exclusion entirely, because the envelope's one enumerated symbol was UNDETERMINABLE for an
    # unrelated reason. The exclusion is KEPT even though the envelope now has no enumerated symbol
    # at all: the enumeration is a heuristic that will widen again, and a guard that only matters
    # after the next widening still has to be proven to work BEFORE it.
    claimant = (
        "class Reflex:\n"
        "    def learn(self, x):\n"
        '        """Approximates the gradient by backprop."""\n'
        "        self.w -= x\n"
    )
    for path in gate._SAFETY_ENVELOPE:
        rules, _, _ = gate.census(overlay={path: claimant})
        hit = [r for r in rules if r["module_rel"] == path]
        assert hit, f"{path} produced no rule, so this case proves nothing"
        assert all(r["state"] != gate.MISSING_DECLARATION for r in hit), (
            f"{path}: an undeclared claimant in the safety envelope produced a finding whose only "
            f"remedy is editing an approval-gated file -- {[r['state'] for r in hit]}"
        )

    # CONTROL: the very same source OUTSIDE the envelope MUST produce the finding, or the assertion
    # above is satisfied by a classifier that flags nothing anywhere.
    rules, _, _ = gate.census(overlay={_PROBE: claimant})
    outside = [r["state"] for r in rules if r["module_rel"] == _PROBE]
    assert gate.MISSING_DECLARATION in outside, (
        f"the identical rule outside the envelope did NOT produce a finding ({outside}); "
        "the envelope result is therefore vacuous"
    )


def test_the_fr010_check_fires_on_a_constructed_violation():
    """FR-010a / T076. A check never shown to produce a finding cannot evidence an absence.

    This one's entire job is certifying that the feature's diff touched no safety-envelope file. An
    implementation returning `[]` unconditionally would certify that perfectly, forever, and this
    project has already had a script edit `brainstem/serial_arm.py` while the text beside it claimed
    the envelope was read and never modified. So the finding is CONSTRUCTED first.
    """
    # THE RED: each of the five, in both path spellings the tree actually uses.
    for env in gate._SAFETY_ENVELOPE:
        for planted in (f"src/snn_data_lake/{env}", rf"src\snn_data_lake\{env}"):
            found = gate.fr010_findings([planted])
            assert found, f"FR-010 did not fire on {planted!r} -- it cannot see the envelope"
            assert "SAFETY ENVELOPE" in found[0][1], found

    # THE GREEN CONTROL: ordinary paths, including this feature's own edits, must NOT fire. Without
    # it, a check that flagged everything would pass every assertion above.
    clean = [
        "src/snn_data_lake/gates/gradient_claim_gate.py",
        "src/snn_data_lake/plasticity/eprop_kernel.py",
        "tests/test_gradient_claim_measure.py",
        "specs/020-eprop-cosine-scope/spec.md",
        "src/snn_data_lake/engine/autonomous_loop.py",  # brainstem/, but NOT in the envelope
    ]
    assert gate.fr010_findings(clean) == [], gate.fr010_findings(clean)

    # And a failure to read git must NOT read as "nothing was touched".
    assert gate.fr010_changed_paths(root="/zz/no/such/repo") is None


def test_this_feature_touched_no_safety_envelope_file():
    """SC-016, run against the LIVE change set rather than against a remembered intention.

    **The first version of this test would have REDDENED CI, and the reason is worth keeping.** It
    ended with `assert changed, "an empty change set proved nothing"` as a positive control. On a
    clean checkout -- which is precisely what CI has -- `git status --porcelain` is EMPTY, so that
    assertion fails on a tree with nothing wrong with it. Verified by cloning this repo into a temp
    directory: 0 porcelain lines.

    The control was aimed at the wrong thing. An empty change set does not make the pass vacuous; it
    makes it TRUE and uninteresting -- nothing changed, so nothing touched the envelope. What must
    not be vacuous is the CHECK, and that is proven by
    `test_the_fr010_check_fires_on_a_constructed_violation`, which constructs a finding for all five
    envelope files in both path spellings. This test carries the LIVE half only.
    """
    changed = gate.fr010_changed_paths()
    if changed is None:
        return  # no git here; the constructed test above still pins the check itself
    findings = gate.fr010_findings(changed, approved=gate.fr010_approved(changed))
    assert not findings, f"this change set touches the safety envelope: {findings}"
    # When there IS something to look at, confirm the check actually looked: run it against the
    # live set plus a planted envelope path and require it to fire. That keeps a real assertion on
    # every run WITHOUT depending on the tree being dirty.
    # the same approval set: an APPROVED live envelope edit must not count as the planted hit
    planted = gate.fr010_findings(
        [*changed, "src/snn_data_lake/brainstem/reflex.py"], approved=gate.fr010_approved(changed)
    )
    assert len(planted) == 1, (
        f"the check did not fire on a planted envelope path among {len(changed)} live paths: "
        f"{planted}"
    )


def test_the_fr010_approval_path_has_its_own_reds(monkeypatch):
    """specs/022 T037a: the first envelope byte change since SC-016 landed showed this check had no
    notion of an APPROVED change. Approval is the project's own mechanized act (T021b): the baseline
    `tools/envelope_hashes.json` in the SAME change set AND `envelope_hash_gate.check()` green on
    the bytes on disk. Each half is a red on its own."""
    from snn_data_lake.gates import envelope_hash_gate as ehg

    env = next(iter(gate._SAFETY_ENVELOPE))
    planted = f"src/snn_data_lake/{env}"
    # (1) no baseline in the change set -> nothing is approved, the finding fires
    assert gate.fr010_approved([planted]) == set()
    assert gate.fr010_findings([planted], approved=gate.fr010_approved([planted]))
    # (2) baseline in the set but the bytes do NOT match it -> still a finding
    monkeypatch.setattr(
        ehg, "check", lambda baseline=None: (False, ["UNAPPROVED envelope byte change"])
    )
    both = [planted, "tools/envelope_hashes.json"]
    assert gate.fr010_approved(both) == set()
    assert gate.fr010_findings(both, approved=gate.fr010_approved(both))
    # (3) baseline in the set AND the gate green -> the envelope path is approved, no finding;
    #     an ordinary path in the same set is never "approved" (the set is envelope-only)
    monkeypatch.setattr(ehg, "check", lambda baseline=None: (True, []))
    approved = gate.fr010_approved(both + ["src/snn_data_lake/substrate/home.py"])
    assert approved == {planted}
    assert gate.fr010_findings(both, approved=approved) == []
    # the constructed red is untouched: the same planted path with NO approval still fires
    assert gate.fr010_findings([planted])


def test_a_phantom_declaration_reds_through_the_assert_composition():
    """PHANTOM_DECLARATION (owner task 2026-08-27): a declared qualname nothing provides.

    Runs the EXACT composition main()'s --assert path runs -- evaluate(governed_counts,
    ..., c_all=all_counts) -- for a GOVERNED phantom, an UNGOVERNED phantom (the ALL-scope
    bound, the one stated FR-001a deviation), the T050 wrong-class shape (a method-LOCAL
    variable in the wrong class must NOT validate the key), and the truthful shapes the
    first surface draft red-flagged (green controls).
    """
    phantom = (
        f'{gate.DECL_NAME} = {{"Ghost.gone": ("{gate.LOCAL}", "keys a symbol with no code")}}\n'
        "class Rule:\n"
        "    def learn(self, x):\n"
        "        self.w -= x\n"
    )
    incident = (
        f'{gate.DECL_NAME} = {{"Wrong.evaluate": ("{gate.LOCAL}", "the T050 shape")}}\n'
        "class Wrong:\n"
        "    def author(self, x):\n"
        "        evaluate = x + 1\n"
        "        return evaluate\n"
        "class Right:\n"
        "    def evaluate(self, x):\n"
        "        return x\n"
    )
    truthful = (
        f'{gate.DECL_NAME} = {{"Outer.Inner.learn": ("{gate.LOCAL}", "nested"), '
        f'"cond_learn": ("{gate.LOCAL}", "conditional"), '
        f'"Rule.learn": ("{gate.LOCAL}", "rebind")}}\n'
        "class Outer:\n"
        "    class Inner:\n"
        "        def learn(self, x):\n"
        "            return x\n"
        "if True:\n"
        "    def cond_learn(x):\n"
        "        return x\n"
        "def _impl(x):\n"
        "    return x\n"
        "class Rule:\n"
        "    pass\n"
        "Rule.learn = staticmethod(_impl)\n"
    )

    def assert_composition(overlay_path, src):
        rules, _, n = gate.census(overlay={overlay_path: src})
        gov = gate.counts(rules, governed_only=True)
        allc = gate.counts(rules, governed_only=False)
        return [f for f in gate.evaluate(gov, None, n, c_all=allc) if gate.PHANTOM_DECLARATION in f]

    # governed phantom (a real governed relpath overlaid) and ungoverned phantom both red
    assert assert_composition("readout/tempotron.py", phantom), "governed phantom must red"
    assert assert_composition(_PROBE, phantom), "ungoverned phantom must red (ALL-scope)"
    # the incident shape reds: a local variable is not a provided symbol
    assert assert_composition(_PROBE, incident), "the T050 wrong-class shape must red"
    # truthful shapes are NOT flagged
    assert assert_composition(_PROBE, truthful) == [], "truthful shapes must stay green"
    # and the live tree is phantom-free through the same composition
    live, _, n_live = gate.census()
    live_reds = [
        f
        for f in gate.evaluate(
            gate.counts(live), None, n_live, c_all=gate.counts(live, governed_only=False)
        )
        if gate.PHANTOM_DECLARATION in f
    ]
    assert live_reds == [], f"the live tree must carry zero phantoms: {live_reds}"
