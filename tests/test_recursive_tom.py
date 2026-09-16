"""
Edge tests for recursive_tom: N-agent, depth-k recursive Theory of Mind over VSA role-filler
propositions read out by a spiking LIF winner-take-all. Covers the behaviors the __main__ self-check
exercises -- false-belief transfer (an absent agent's belief stays stale), a divergent second-order
belief, non-location attitudes (wants/knows) with nesting, depth-3 nesting, and determinism.

(This list ended "and zero-dependency purity" until 2026-08-04. No test in this file has ever
asserted that, and P7 abolished the rule it named -- so the docstring was advertising coverage that
does not exist, for a property the project no longer holds. Both halves are wrong independently,
which is why it goes rather than gets a purity test written for it.)

Since 2026-09-16 it also carries T082's six reds (specs/022-unified-brain-arc/evidence/
t082-social-cognition.md 2.4), one per `test_r*`. They are committed RUNNING RED against 2.9
step 2's stubs and go green at step 4, where each one's MUTATED scorer is applied here and in
`recursive_tom.t082_reds()` -- the same constructed input in both, so a gate run that never
executes pytest still runs every red.

Run: pytest tests/test_recursive_tom.py
"""

import random

import pytest

from snn_data_lake.cortex import recursive_tom
from snn_data_lake.cortex.recursive_tom import RecursiveToM
from snn_data_lake.cortex.theory_of_mind import TheoryOfMind


def test_false_belief_transfer_and_world_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("B", "believes", "Y", witnesses={"B"})  # A absent for the move
    assert t.predict_first_order("A") == "X"  # false belief persists
    assert t.world_content("believes") == "Y"  # world moved on
    assert t.predict_first_order("A") != t.world_content("believes")


def test_absent_agent_belief_does_not_update():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A"})
    t.observe_event("B", "believes", "Y", witnesses={"B"})  # A not present -> untouched
    assert t.predict_first_order("A") == "X"
    assert t.predict_first_order("B") == "Y"


def test_second_order_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("A", "believes", "Y", witnesses={"A"})  # B absent for A's update
    assert t.predict_first_order("A") == "Y"
    assert t.predict_nested("B", "A") == "X"
    assert t.divergence("B", "A") is True


def test_true_belief_has_no_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("A", "believes", "Y", witnesses={"A", "B"})  # B present -> tracks the update
    assert t.predict_nested("B", "A") == "Y"
    assert t.divergence("B", "A") is False


def test_non_location_attitudes_store_and_nest():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("C", "wants", "cookie", witnesses={"C", "A"})
    t.observe_event("C", "knows", "secret", witnesses={"C"})
    assert t.predict_first_order("C", "wants") == "cookie"
    assert t.predict_first_order("C", "knows") == "secret"
    assert t.predict_nested("A", "C", "wants") == "cookie"
    assert t.predict_nested("A", "C", "knows") is None  # A didn't witness C learn it


def test_depth_three_nesting():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "P", witnesses={"A", "B", "C"})
    assert t.predict_chain(("C", "B", "A")) == "P"
    assert t.predict_chain(("B", "A")) == "P"
    assert t.predict_first_order("A") == "P"


def test_unheld_paths_return_none():
    t = RecursiveToM(n_agents=3, seed=0)
    assert t.predict_first_order("A") is None  # nothing observed yet
    t.observe_event("A", "believes", "X", witnesses={"A"})
    assert t.predict_nested("B", "A") is None  # B never modeled A


def test_determinism_and_noise_robustness():
    def build():
        t = RecursiveToM(n_agents=3, seed=0)
        t.observe_event("A", "believes", "X", witnesses={"A", "B"})
        t.observe_event("B", "believes", "Y", witnesses={"B"})
        return t

    a, b = build(), build()
    assert a.predict_first_order("A") == b.predict_first_order("A")
    assert a.predict_nested("B", "A") == b.predict_nested("B", "A")
    assert a.predict_first_order("A", jitter=2) == "X"  # population readout survives jitter


# =============================================================================
# T082 -- the six reds of 2.4. Each runs on its CONSTRUCTED input and, at 2.9 step 4, is
# reddened IN FACT by its own mutated scorer. Committed RED at step 3: `_need` names the
# instrument the step-4 commit lands, so every red fails on an ASSERTION rather than on an
# import error, which is what 2.4's "runs on its constructed input and FAILS its assertion"
# requires of the skeleton.
# =============================================================================
#: R1's three surface forms: the two the grammar accepts (2.1's script) and one it must refuse.
T082_OFF_GRAMMAR = "the weather is nice today."
T082_WITNESSED_FORM = "i put the marble in the basket."
T082_UNWITNESSED_FORM = "someone moved the marble to the box while i was away."


def _need(name):
    """The instrument `name`, or an ASSERTION naming the step that lands it. 2.9 step 2's
    skeleton carries stubs for what it can; the helpers below arrive at step 4, and a red that
    died on an AttributeError would not be a red that RAN."""
    fn = getattr(recursive_tom, name, None)
    assert fn is not None, (
        f"cortex.recursive_tom.{name} does not exist yet -- 2.9 step 2 landed the stubs and "
        f"step 4 lands the instrument. This red is RUNNING red, exactly as 2.4 requires."
    )
    return fn


def _as_int(label):
    """The recursive tracker's content is a LABEL; `TheoryOfMind` and the retired second-order
    semantics key on an integer location. R5 compares the same quantity across both."""
    return None if label is None else int(label)


@pytest.fixture(scope="module")
def scratch_brain():
    """THE ONE brain this module builds (2.9 step 3's cap): a scratch `make_brain(seed=0,
    persist=False)` under `throwaway_home()`, so the daemon's brain and the durable lake are
    unreachable from here. Module-scoped -- R1 is the only red that needs a live parse."""
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        from snn_data_lake.cortex.brain import make_brain

        yield make_brain(seed=0, persist=False)


def test_r1_the_parse_refuses_outside_the_grammar(scratch_brain):
    """R1 -- THE PARSE REFUSES OUTSIDE THE GRAMMAR. A statement the hand-specified surface parse
    does not match returns None from the LIVE `Brain._tom_update`, and (r) reads REFUSED on a
    constructed stream containing it; both script forms parse. The `parse_anything` mutation
    (a non-match returns an event) removes the refusal."""
    b = scratch_brain
    assert b._tom_update(T082_OFF_GRAMMAR) is None, (
        "a statement outside the grammar must be REFUSED, never guessed"
    )
    assert b._tom_update(T082_WITNESSED_FORM) is not None, "the witnessed script form must parse"
    assert b._tom_update(T082_UNWITNESSED_FORM) is not None, (
        "the unwitnessed script form must parse"
    )

    rows_fn = _need("t082_constructed_rows")
    mutated = _need("_t082_mutated_scorer")
    rec = rows_fn(seed=0, off_grammar_at=3)
    rows = rec["rows"]
    read = recursive_tom.t082_read(rows)
    assert read.get("n") == len(rows)
    assert read.get("parsed") == len(rows) - 1, (
        "exactly the off-grammar statement must fail to parse"
    )
    assert any("parse" in str(w) for w in read.get("refusals", [])), (
        "(r) must REFUSE a stream carrying a statement the grammar does not accept"
    )
    mut_rows = rows_fn(seed=0, off_grammar_at=3, parse=mutated("parse_anything"))["rows"]
    mut_read = recursive_tom.t082_read(mut_rows)
    assert mut_read.get("parsed") == len(mut_rows), "the mutation must parse everything"
    assert not mut_read.get("refusals"), (
        "the parse_anything mutation must REMOVE (r)'s refusal -- that is R1's red"
    )


def test_r2_the_control_is_a_world_model():
    """R2 -- THE CONTROL IS A WORLD MODEL. On a constructed 40-event witness script at 50 %
    absences the equal-capacity control's first-order prediction equals the world on 40/40 (it
    is the same class with the absence gating removed -- the C5 row's own dissociation input).
    The `control_keeps_witnesses` mutation (the control HONOURS absence) differs on the absent
    events."""
    feed = _need("t082_feed")
    mutated = _need("_t082_mutated_scorer")
    steps = recursive_tom.t082_script(0, recursive_tom.T082_R2_EVENTS)
    assert len(steps) == recursive_tom.T082_R2_EVENTS
    absent = [s for s in steps if s["unwitnessed"]]
    assert absent, "the script must carry absences for this red to be reachable"

    ctrl = RecursiveToM(n_agents=3, seed=0)
    hits = 0
    for st in steps:
        feed(ctrl, set(recursive_tom.T082_AGENTS), st["loc"])  # the control: everyone witnesses
        hits += ctrl.predict_first_order("user") == st["world"]
    assert hits == len(steps), (
        f"the equal-capacity control must read the WORLD on every episode, got {hits}/{len(steps)}"
    )

    keeps = mutated("control_keeps_witnesses")
    ctrl2 = RecursiveToM(n_agents=3, seed=0)
    mhits = 0
    for st in steps:
        keeps(ctrl2, st)
        mhits += ctrl2.predict_first_order("user") == st["world"]
    assert mhits < len(steps), (
        "a control that honours absence must DIFFER from the world on the absent events -- "
        "that is R2's red"
    )


def test_r3_the_dissociation_on_a_constructed_script():
    """R3 -- THE DISSOCIATION ON A CONSTRUCTED SCRIPT. Eight constructed MIXED scripts read (s1)
    MET and (s2) MET; eight ALL-WITNESSED scripts read (s1) NOT MET as a degenerate zero
    (acc_belief == acc_ctrl -- with nothing unwitnessed there is no belief to beat the world
    with). The `blind_scorer` mutation (the two sides swapped) reddens (s1) on the mixed set."""
    read_fn = _need("t082_constructed_read")
    mutated = _need("_t082_mutated_scorer")
    seeds = recursive_tom.T082_R_SEEDS

    mixed = [read_fn(seed=s) for s in seeds]
    got = recursive_tom.t082_contrasts(mixed)
    assert got.get("n_seeds") == len(seeds)
    assert got["s1"].met, f"(s1) must be MET on the mixed scripts: {got['s1']}"
    assert got["s2"]["met"], f"(s2) must be MET on the mixed scripts: {got['s2']}"

    allw = [read_fn(seed=s, all_witnessed=True) for s in seeds]
    got2 = recursive_tom.t082_contrasts(allw)
    assert all(d == 0.0 for d in got2["s1"].diffs), (
        f"an all-witnessed script leaves the belief model nothing to gain: {got2['s1'].diffs}"
    )
    assert not got2["s1"].met, "(s1) must read NOT MET on an all-witnessed script"

    blind = recursive_tom.t082_contrasts(mixed, score=mutated("blind_scorer"))
    assert not blind["s1"].met, (
        "swapping the belief and control arms must redden (s1) -- that is R3's red"
    )


def test_r4_the_charge_ab():
    """R4 -- THE CHARGE A/B. Recorded belief frames carrying false beliefs drive the brain's own
    `_tom_protect` fold to a caution trajectory that DIVERGES from the pinned-[0, 0] one, and the
    constant consumer (specificity) does not flip. The `consumer_ignores_payload` mutation (the
    fold reads 0 in place of the payload) removes the divergence, which is exactly the condition
    `record_charge` REFUSES on."""
    live, pinned_out, reads, spec = recursive_tom._measure_belief_charge()
    assert live is not None and pinned_out is not None, (
        "the charge instrument is still 2.9 step 2's stub (it returns a null reading)"
    )
    assert reads == recursive_tom.T082_R4_FRAMES, (
        f"the neuter must be READ on every recorded frame, got {reads}"
    )
    assert live != pinned_out, "the belief payload must move the caution trajectory"
    assert spec, "the constant consumer must NOT flip (specificity)"

    mutated = _need("_t082_mutated_scorer")
    m_live, m_pinned, m_reads, _ = recursive_tom._measure_belief_charge(
        consume=mutated("consumer_ignores_payload")
    )
    assert m_reads == recursive_tom.T082_R4_FRAMES, "the mutated consumer must still be read"
    assert m_live == m_pinned, (
        "a fold that ignores the payload must NOT diverge, so record_charge refuses -- "
        "that is R4's red"
    )


def test_r5_the_merge_is_lossless():
    """R5 -- THE MERGE IS LOSSLESS. On 200 random witness scripts (3 agents, 4 locations, 12
    events each) the SYMMETRIC feed into `RecursiveToM` reads `predict_nested(x, y)` equal to the
    retired second-order semantics for every ordered pair, and `predict_first_order` equal to
    `TheoryOfMind.predict_from_belief` for every agent. The `nested_reads_world` mutation reddens
    it.

    `affect_core.SecondOrderToM` is DELETED in the same commit that greens this red (2.1), so the
    second-order side is compared against `recursive_tom.t082_second_order_reference` -- the
    deleted class's `observe` bookkeeping restated over plain dicts. Recorded, not smoothed: the
    reference is a genuinely different mechanism (no VSA bundle, no spiking readout), which makes
    this a STRONGER equivalence than comparing against a class that shared `BeliefPopulation`
    with `TheoryOfMind`. The pre-deletion run against the live class is in the row's record."""
    ref_cls = _need("t082_second_order_reference")
    feed = _need("t082_feed")
    n_loc = recursive_tom.T082_R5_LOCATIONS
    agents = ("A", "B", "C")[: recursive_tom.T082_R5_AGENTS]

    for i in range(recursive_tom.T082_R5_SCRIPTS):
        rng = random.Random(recursive_tom.T082_R5_RNG_BASE + i)
        rt = RecursiveToM(n_agents=len(agents), seed=0)
        ref = ref_cls(n_locations=n_loc, seed=0)
        tom = TheoryOfMind(n_locations=n_loc, seed=0)
        for ev in range(recursive_tom.T082_R5_EVENTS):
            loc = rng.randrange(n_loc)
            # the opening event is witnessed by ALL, so every first-order and every ordered
            # pair is initialised and the comparison never rests on an unheld path
            present = (
                set(agents)
                if ev == 0
                else ({a for a in agents if rng.random() < 0.6} or {agents[0]})
            )
            feed(rt, present, str(loc))
            ref.observe(loc, sorted(present))
            tom.place(loc, sorted(present))
        for x in agents:
            assert _as_int(rt.predict_first_order(x)) == tom.predict_from_belief(x), (
                f"script {i}: first order for {x} diverged from TheoryOfMind"
            )
            for y in agents:
                if x == y:
                    continue
                assert _as_int(rt.predict_nested(x, y)) == ref.predict_second_order(x, y), (
                    f"script {i}: nested ({x},{y}) diverged from the retired second-order form"
                )

    mutated = _need("_t082_mutated_scorer")
    rng = random.Random(recursive_tom.T082_R5_RNG_BASE)
    rt = RecursiveToM(n_agents=len(agents), seed=0)
    ref = ref_cls(n_locations=n_loc, seed=0)
    reads_world = mutated("nested_reads_world")
    differs = 0
    for ev in range(recursive_tom.T082_R5_EVENTS):
        loc = rng.randrange(n_loc)
        present = (
            set(agents) if ev == 0 else ({a for a in agents if rng.random() < 0.6} or {agents[0]})
        )
        feed(rt, present, str(loc))
        ref.observe(loc, sorted(present))
    for x in agents:
        for y in agents:
            if x != y and _as_int(reads_world(rt, x, y)) != ref.predict_second_order(x, y):
                differs += 1
    assert differs > 0, (
        "a nested read that returns the WORLD must diverge from the retired second-order "
        "semantics on a stale pair -- that is R5's red"
    )


def test_r6_the_symmetric_feed_kills_the_spurious_divergence():
    """R6 -- THE SYMMETRIC FEED KILLS THE SPURIOUS DIVERGENCE (fact 4 reconstructed). After an
    unwitnessed then a witnessed statement `divergence("user", "brain")` is False and
    `predict_first_order("brain")` equals the world. The `feed_subject_only` mutation -- the OLD
    subject-centric call the live path made until this row -- reads the divergence True, which is
    the persistent spurious nested false belief fact 4 measured at 5 of 12 episodes."""
    feed = _need("t082_feed")
    mutated = _need("_t082_mutated_scorer")
    both, brain_only = {"user", "brain"}, {"brain"}

    rt = RecursiveToM(n_agents=3, seed=0)
    feed(rt, both, "basket")  # witnessed opening
    feed(rt, brain_only, "box")  # UNWITNESSED: the user is away
    assert rt.predict_first_order("user") == "basket", "the user's false belief must stand"
    assert rt.world_content() == "box"
    feed(rt, both, "drawer")  # witnessed again
    assert rt.divergence("user", "brain") is False, (
        "the symmetric feed must leave NO nested divergence after a witnessed statement"
    )
    assert rt.predict_first_order("brain") == rt.world_content() == "drawer"

    subject_only = mutated("feed_subject_only")
    rt2 = RecursiveToM(n_agents=3, seed=0)
    subject_only(rt2, both, "basket")
    subject_only(rt2, brain_only, "box")
    subject_only(rt2, both, "drawer")
    assert rt2.divergence("user", "brain") is True, (
        "the OLD subject-only feed must read a spurious nested divergence -- that is R6's red"
    )


def test_charge_evidence_is_committed_and_reproduces():
    """2.1's standing obligation on the declared wire: the attribution in
    `tools/region_charge_evidence.json` must RE-MEASURE to the same numbers, so the record
    cannot outlive the wire it describes (`tests/test_task_film.py`'s shape for the FiLM wire).
    The committed row is the CONSTRUCTED stream's A/B, which is why it reproduces at unit grain
    without building a brain; the live 60-episode recording's A/B is reported by the dry run."""
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    rec = read_attribution(recursive_tom.BELIEF_EDGE)
    assert rec is not None and rec.get("diverged") is True
    live, pinned_out, reads, spec = recursive_tom._measure_belief_charge()
    assert live != pinned_out and spec
    assert reads == recursive_tom.T082_R4_FRAMES
    assert rec.get("measured") == {"reads": reads, "first": list(live[0]), "last": list(live[-1])}


def test_the_charged_reading_is_re_derived_from_the_pin(tmp_path):
    """2.6's PASS branch (2026-09-16): `CONCLUSION['social_cognition'].reading` is not a free
    literal -- it equals 2.6's reduction of the committed pin (the mean over seeds of
    acc_belief - acc_ctrl) at the recorded precision, the MUTATED derivation (the sides swapped)
    differs, and a missing pin is a RED, never a skip (T080's and T081's shape)."""
    derived, recorded, mutated = recursive_tom.t082_pin_reading()
    assert derived == recorded == 0.4437, (derived, recorded)
    assert mutated != recorded, mutated
    with pytest.raises(AssertionError):
        recursive_tom.t082_pin_reading(tmp_path / "absent.json")


def test_the_live_false_belief_list_reaches_state_with_both_c5_sentences_beside_it(scratch_brain):
    """2.6's PASS branch: `Brain.state()["recursive_tom_false_beliefs"]` is the surface symbol the
    charged row names, its PROVENANCE carries BOTH C5 sentences plus "representation-level", the
    pin and the charge evidence, and the key is a LIST OF AGENT NAMES rather than the row's
    reading -- the confusion the provenance exists to prevent.

    The SCRATCH brain of this module's fixture, never the live lake. The two-statement sequence is
    deterministic whatever earlier reds left in the tracker: a witnessed statement puts every
    tracked agent on the world, so the list empties; the unwitnessed statement that follows moves
    the world with the user absent, so the user's first-order belief and the user's nested model of
    the brain both go stale."""
    from snn_data_lake.substrate import brain_telemetry
    from snn_data_lake.substrate.conclusion_protocol import CONCLUSION

    # The two sentences are read OFF THE ROW, never retyped here: a provenance line that
    # paraphrases them is a second copy that can drift.
    licensed, refused, _dis, reading, _date, symbols = CONCLUSION["social_cognition"]
    prov = brain_telemetry.PROVENANCE["recursive_tom_false_beliefs"]
    assert licensed in prov and refused in prov, "both C5 sentences ride beside the symbol"
    assert "representation-level" in prov
    assert "t082_social_cognition.json" in prov and "region_charge_evidence.json" in prov
    assert "BY CONSTRUCTION" in prov  # 2.10 (b)'s entailment, disclosed at the surface
    assert symbols == ("recursive_tom_false_beliefs",) and reading == 0.4437

    b = scratch_brain
    assert b._tom_update(T082_WITNESSED_FORM) is not None
    assert b.state()["recursive_tom_false_beliefs"] == [], "a witnessed turn leaves no divergence"
    assert b._tom_update(T082_UNWITNESSED_FORM) is not None
    live = b.state()["recursive_tom_false_beliefs"]
    # A LIST OF NAMES, not a number, and not the row's reading -- the first-order holder followed
    # by the ordered nested pair, exactly what `t082_false_beliefs` derives.
    assert live == ["user", "user->brain"], live
    assert all(isinstance(x, str) for x in live)
    assert live != reading and len(live) != reading
