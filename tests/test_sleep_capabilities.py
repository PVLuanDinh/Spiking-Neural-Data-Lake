"""specs/022 T058 -- sleep ownership (FR-024): per-capability neuters in one command.

The measured pre-T058 defect: every sleep-tick relax() returned None (no Benna-Fusi
cascade had ever been ensured), so `PlasticityRuleStack.consolidate()` was unreachable
on sleep ticks and `_plastic_relaxed` stayed 0 -- a silent no-op wearing a consolidation
lane's name. These arms drive the REAL Brain (temp NORD_HOME, hermetic) through wake
teaching + sleep ticks under each per-capability neuter (replay / benna_fusi /
structural) and the phase-level aggregate, in this one command, asserting each
capability's own observable moves under the full arm and goes flat under ITS neuter.
"""

import pytest

TEACH = [
    ("what is your goal today", "GOAL"),
    ("do you remember the bench", "MEMORY"),
    ("hello there brain", "GREET"),
    ("are you aware of yourself", "CONSCIOUS"),
]


def _brain(tmp_path, monkeypatch, sub):
    monkeypatch.setenv("NORD_HOME", str(tmp_path / sub))
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=0)
    assert b._plasticity is not None, "the sleep-ownership arms need the rule stack live"
    return b


def _teach(b, reps=3):
    from snn_data_lake.readout.interaction_learning import text_features

    for _ in range(reps):
        for text, intent in TEACH:
            b.brain._ixn.interact(text_features(text, b.brain._ixn.n_in), intent)


def _ixn_snapshot(b):
    return [list(n.w) for n in b.brain._ixn.neurons]


def test_sleep_capabilities_one_command(tmp_path, monkeypatch):
    # ---- FULL ARM: replay reaches consolidate(), rest settles, the slow clock reports ----
    full = _brain(tmp_path, monkeypatch, "full")
    _teach(full)
    full._sleep_tick()
    assert full._sleep_replayed > 0, "sleep must replay recorded wake interactions"
    assert full._sleep_consolidate_calls > 0, (
        "THE T058 POINT: consolidate() must be REACHED on the sleep tick itself -- the "
        "measured pre-state was zero reaches and every relax() returning None"
    )
    assert full._plastic_relaxed > 0, "rest must settle cascades the replay ensured"
    assert isinstance(full._structural_report, dict) and full._structural_report["budget"] > 0, (
        f"the slow clock must run and report ({full._structural_report})"
    )
    assert full._structural_refusals == 0

    # ---- REPLAY NEUTER: same wake teaching; the sleep tick reaches consolidate() ZERO
    # times (the reconstructed pre-T058 state) while rest still settles the WAKE-ensured
    # cascades -- replay's charge is the sleep-time deposits specifically.
    nr = _brain(tmp_path, monkeypatch, "nr").neuter_sleep("replay")
    _teach(nr)
    nr._sleep_tick()
    assert nr._sleep_replayed == 0 and nr._sleep_consolidate_calls == 0, (
        f"the replay neuter must cut the sleep-tick deposits "
        f"({nr._sleep_replayed}, {nr._sleep_consolidate_calls})"
    )
    assert nr._plastic_relaxed > 0, "wake-ensured cascades still rest without replay"

    # ---- BENNA-FUSI NEUTER: replay still deposits, but the rest phase never settles or
    # writes back -- benna_fusi's sleep charge is the rest half.
    nb = _brain(tmp_path, monkeypatch, "nb").neuter_sleep("benna_fusi")
    _teach(nb)
    nb._sleep_tick()
    assert nb._sleep_consolidate_calls > 0, "replay deposits survive the rest neuter"
    assert nb._plastic_relaxed == 0, (
        f"the rest neuter must leave no cascade settled ({nb._plastic_relaxed})"
    )

    # ---- STRUCTURAL NEUTER + THE CONSTRUCTED RED: register births with one sleep tick,
    # flip a live slot's sign (an identity violation, the DEEP-R trigger), sleep again.
    # Full arm: the flipped slot is PRUNED (zeroed) and the report counts it. Neutered
    # twin, same sequence: the flipped value survives untouched by the slow clock.
    def _flip_site(b):
        for _i, _n in enumerate(b.brain._ixn.neurons):
            for _c, _w in enumerate(_n.w):
                if _w > 1e-6:
                    return _i, _c
        raise AssertionError("no positive live slot to flip")

    st = _brain(tmp_path, monkeypatch, "st")
    _teach(st)
    st._sleep_tick()  # first sight: the rewirer records birth signs
    i, c = _flip_site(st)
    flipped_v = -abs(st.brain._ixn.neurons[i].w[c])
    st.brain._ixn.neurons[i].w[c] = flipped_v
    st.neuter_sleep("replay").neuter_sleep("benna_fusi")  # isolate the slow clock's write
    st._sleep_tick()
    assert st._structural_report["pruned"] >= 1, (
        f"a constructed sign flip must be pruned by the slow clock ({st._structural_report})"
    )
    # the violating value is GONE from the row (pruned to 0, or that vacancy re-seeded by
    # the budget-conserving regen -- either way the identity violation did not survive)
    assert st.brain._ixn.neurons[i].w[c] != flipped_v, (
        f"the flipped value must not survive the slow clock ({st.brain._ixn.neurons[i].w[c]})"
    )

    ns = _brain(tmp_path, monkeypatch, "ns")
    _teach(ns)
    ns._sleep_tick()
    i2, c2 = _flip_site(ns)
    flipped = -abs(ns.brain._ixn.neurons[i2].w[c2])
    ns.brain._ixn.neurons[i2].w[c2] = flipped
    ns.neuter_sleep("replay").neuter_sleep("benna_fusi").neuter_sleep("structural")
    ns._sleep_tick()
    assert ns.brain._ixn.neurons[i2].w[c2] == flipped, (
        "with the slow clock neutered the flipped slot must survive the sleep tick"
    )

    # ---- PHASE AGGREGATE (FR-024's phase-level control): all three off -> the sleep tick
    # leaves the interaction substrate byte-identical and every counter flat.
    ph = _brain(tmp_path, monkeypatch, "ph").neuter_sleep("phase")
    _teach(ph)
    before_w = _ixn_snapshot(ph)
    before_calls = ph._plasticity.consolidation_calls()
    ph._sleep_tick()
    assert _ixn_snapshot(ph) == before_w, "the phase neuter must leave the substrate untouched"
    assert ph._plasticity.consolidation_calls() == before_calls
    assert ph._sleep_replayed == 0 and ph._plastic_relaxed == 0
    assert ph._structural_report is None

    # ---- STALE-CARRY RED (the panel's blocker-shaped finding): a brain that has slept
    # successfully and is THEN neutered must read 0/None on the neutered tick -- the
    # previous tick's success must never stand where evidence belongs.
    sc = _brain(tmp_path, monkeypatch, "sc")
    _teach(sc)
    sc._sleep_tick()
    assert sc._sleep_replayed > 0 and sc._structural_report is not None  # success recorded
    sc.neuter_sleep("phase")
    sc._sleep_tick()
    assert sc._sleep_replayed == 0 and sc._sleep_consolidate_calls == 0, (
        "a phase-neutered tick must not inherit the previous tick's replay evidence"
    )
    assert sc._plastic_relaxed == 0 and sc._structural_report is None, (
        "a phase-neutered tick must not inherit the previous tick's rest/turnover evidence"
    )

    # ---- the neuter itself refuses an unknown element (a no-op neuter reads as passing)
    with pytest.raises(ValueError, match="unknown element"):
        ph.neuter_sleep("nonsense")

    # ---- the counters are SURFACED (structural_rewiring's record: state()-field
    # surfacing lands WITH the wire): a consumer can read the whole evidence row.
    row = sc.state()["sleep_consolidation"]
    assert set(row) == {
        "replayed",
        "consolidate_calls",
        "relaxed",
        "structural",
        "structural_refusals",
    }


def test_cascade_adoption_charge(tmp_path, monkeypatch):
    # THE TWO-WRITER CONFLICT'S CHARGE (T048/T051b-ii, closed at T058): with structural
    # AND benna_fusi both LIVE in one tick, a slow-clock edit must SURVIVE the same
    # tick's relax -- and with adopt_structural neutered (the deletion mutant the panel
    # measured surviving the whole suite), the deep store must re-assert the old value,
    # which is exactly the erasure the fix exists to stop.
    def _flipped_slot(b):
        for _i, _n in enumerate(b.brain._ixn.neurons):
            for _c, _w in enumerate(_n.w):
                if _w > 1e-6:
                    return _i, _c
        raise AssertionError("no positive live slot")

    def _run(sub, adopt_live):
        b = _brain(tmp_path, monkeypatch, sub)
        _teach(b)
        b._sleep_tick()  # births registered; cascades ensured by replay deposits
        i, c = _flipped_slot(b)
        b.brain._ixn.neurons[i].w[c] = -abs(b.brain._ixn.neurons[i].w[c])
        b.neuter_sleep("replay")  # isolate: slow clock + rest TOGETHER, no new deposits
        if not adopt_live:
            monkeypatch.setattr(type(b._plasticity), "adopt_structural", lambda self, *a, **k: 0)
        b._sleep_tick()  # rewire prunes the flip, then relax settles IN THE SAME TICK
        monkeypatch.undo()
        return b, i, c

    live, i, c = _run("adopt_live", adopt_live=True)
    assert live._structural_report["pruned"] >= 1
    w_live = live.brain._ixn.neurons[i].w[c]
    dead, j, d = _run("adopt_dead", adopt_live=False)
    assert dead._structural_report["pruned"] >= 1
    w_dead = dead.brain._ixn.neurons[j].w[d]
    # adopted: the cascade was re-anchored at the structural edit, so relax settles AT it;
    # unadopted: relax rebinds from the deep store and drags the row back toward the
    # pre-edit trace -- the silent erasure. The two must be measurably different.
    assert w_live != w_dead, (
        f"neutering adopt_structural must change the post-relax weight -- if it does not, "
        f"the cascade-adoption fix carries no charge ({w_live} vs {w_dead})"
    )
    assert abs(w_live) < abs(w_dead), (
        f"the unadopted row must be dragged back toward the deep store's pre-edit trace "
        f"(adopted {w_live}, unadopted {w_dead})"
    )


def test_production_feed_reaches_consolidation(tmp_path, monkeypatch):
    # THE PRODUCTION FEED (the panel's regime gap): the certifying arms above hand-call
    # interact(); this arm drives the brain's own conversational entry -- respond(), the
    # daemon's ingest route -- and asserts the ring fills and the sleep tick consolidates.
    # The autonomous drive loop ALONE never fills the ring (process-local, disclosed in
    # the sleep block); conversation is the ring's producer, and this proves that route.
    b = _brain(tmp_path, monkeypatch, "prod")
    for text in ("hello there brain", "what is your goal today", "do you remember me"):
        b.respond(text)
    assert len(b.brain._ixn._replay) > 0, (
        "the PRODUCTION conversational route (respond -> chat -> learn_turn -> interact) "
        "must feed the replay ring"
    )
    b._sleep_tick()
    assert b._sleep_replayed > 0 and b._sleep_consolidate_calls > 0, (
        "a sleep tick after production conversation must reach consolidate()"
    )


def test_replay_buffer_and_counter_units():
    # unit grain: the learner records wake interactions, replays without re-recording,
    # and the stack counter differences deposits exactly.
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    ixn = SpikingInteractionLearner(["A", "B"], n_in=8, seed=0)
    ixn.stack = PlasticityRuleStack(n_neurons=2, seed=0)
    assert ixn.replay(k=4) == 0, "an empty buffer replays nothing and says 0"
    ixn.interact([0.9, 0.1, 0.0, 0.2, 0.5, 0.0, 0.3, 0.1], "A")
    ixn.interact([0.1, 0.8, 0.4, 0.0, 0.0, 0.6, 0.0, 0.2], "B")
    assert len(ixn._replay) == 2
    before = ixn.stack.consolidation_calls()
    assert before > 0, "wake interacts deposit through the stack sandwich"
    n = ixn.replay(k=4)
    assert n == 4 and len(ixn._replay) == 2, "replay must not re-record itself"
    assert ixn.stack.consolidation_calls() > before, "replay must reach consolidate()"
