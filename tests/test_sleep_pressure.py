"""G6 / T114 part (c): the sleep-pressure harness (C12, A6, A7 C23/C24; clock arm == the clock).

Pre-registration: evidence/g6-sleep-durable-memory.md (2.5, 2.11, 2.16 A6, A7), specs/022.
Brain-backed cases build one short scratch brain each under throwaway_home (never the live lake).
"""

import pytest

from snn_data_lake.instruments import sleep_pressure as sp

GATE = ([False] * 6 + [True] * 4) * 5


def test_self_check_runs():
    sp.main()


def test_the_solve_arm_refuses_to_alias_the_oracle():
    """R3 refutation R2: no solved_sink ticks -> the solve arm is refused, not the oracle."""
    with pytest.raises(ValueError):
        sp.run_pressure_arm(None, "solve", 10, lambda b, t: None)


def test_a_request_advances_its_cycle_once_and_keeps_the_budget():
    r = sp.Rescheduler(sp.cycles_from_gate(GATE), len(GATE))
    assert r.request(2) and not r.request(3)
    assert not r.request(7)  # inside the advanced night: absorbed
    plan = r.plan()
    assert plan[2:6] == [True] * 4 and plan[6:10] == [False] * 4
    assert sp.budget(plan) == sp.budget(GATE) and r.absorbed == 2


def test_c12_refuses_an_arm_whose_budget_differs():
    extra = list(GATE)
    extra[0] = True
    ok, ref, bad = sp.check_budgets({"clock": GATE, "same": list(GATE), "extra": extra})
    assert not ok and set(bad) == {"extra"} and ref == (20, 5)


def test_a6_the_shuffle_moves_onsets_but_never_the_budget():
    r = sp.Rescheduler(sp.cycles_from_gate(GATE), len(GATE))
    for t in (3, 14, 33):
        r.request(t)
    for seed in range(5):
        sh = sp.shuffled_plan(sp.cycles_from_gate(GATE), len(GATE), r.advanced, seed)
        assert sp.budget(sh) == sp.budget(GATE)
    with pytest.raises(ValueError):
        sp.shuffled_plan(sp.cycles_from_gate(GATE), len(GATE), {1: 99}, 0)


def test_a7_cold_start_zero_baseline_and_late_admission():
    w = {"a": [0.0], "z": [0.0]}
    m = sp.PressureMeter({"a": lambda: w["a"], "z": lambda: w["z"]})
    for _ in range(sp.K_MIN):
        assert not m.request()
        for _ in range(4):
            w["a"] = [w["a"][0] + 1.0]
            m.observe_wake()
        m.wake_period_done()
    assert m.eligible() == ["a"]  # C23: 'z' has baseline 0 and is never a divisor
    for _ in range(9):  # a heavier wake than the baseline -> a request
        w["a"] = [w["a"][0] + 1.0]
        m.observe_wake()
    assert m.request()
    m.admit("late", lambda: [1e9])
    m.observe_wake()
    assert "late" not in m.eligible()  # C24: silent until its own K_MIN periods


def test_sleep_ticks_never_count_as_load():
    w = {"a": [0.0]}
    m = sp.PressureMeter({"a": lambda: w["a"]})
    m.observe_wake()
    w["a"] = [5.0]
    m.observe_sleep()
    m.observe_wake()  # first wake read after sleep re-anchors; the sleep-time move is not load
    assert m.c["a"]["load"] == 0.0


@pytest.fixture(scope="module")
def short_brain_runs():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import require_isolated_home, throwaway_home

    out = {}
    for arm, req in (("clock", None), ("oracle", [16, 41])):
        with throwaway_home():
            require_isolated_home()
            b = make_brain(seed=12201)
            gate = sp.clock_gate(b, 60)
            out[arm] = (
                gate,
                sp.run_pressure_arm(b, arm, 60, lambda brain, t: None, requests=req, seed=12201),
            )
            assert "consolidation_gate" not in vars(b.clock), "the gate is restored after the run"
    return out


def test_the_clock_arm_reproduces_the_clocks_own_gate(short_brain_runs):
    gate, run = short_brain_runs["clock"]
    assert run.plan == gate


def test_requests_move_onsets_and_keep_the_budget_on_a_real_brain(short_brain_runs):
    gate, run = short_brain_runs["oracle"]
    assert run.advanced and run.plan != gate
    assert sp.check_budgets({"clock": gate, "oracle": run.plan})[0]


@pytest.mark.parametrize("n", [0, 1, 7])
def test_decide_c_refuses_a_wrong_seed_count(n):
    seeds = [
        {"clock": 0.5 + 0.01 * i, "oracle": 0.6 + 0.01 * i, "pressure": 0.58, "shuffled": 0.5}
        for i in range(8)
    ][:n]
    assert sp.decide_c(seeds)[0] == "REFUSED"


def test_a7_every_wake_tick_of_learning_is_counted_across_sleep_and_restore():
    """Codex panel MAJOR: the first wake tick after sleep, restore and admission was dropped."""
    w = [0.0]
    m = sp.PressureMeter({"c": lambda: list(w)})

    def wake_tick():
        w[0] += 1.0
        m.observe_wake()

    wake_tick()
    wake_tick()  # construction snapshot: BOTH ticks count
    assert m.c["c"]["load"] == 2.0
    m.wake_period_done()
    w[0] += 50.0  # sleep-tick plasticity: never load
    m.observe_sleep()
    wake_tick()  # the first wake tick after sleep counts
    assert m.c["c"]["load"] == 1.0
    restored = sp.PressureMeter({"c": lambda: list(w)}, state=m.state())
    w[0] += 1.0
    restored.observe_wake()  # the first wake tick after a restore counts
    assert restored.c["c"]["load"] == 2.0
    late = [0.0]
    m.admit("late", lambda: list(late))
    late[0] += 4.0
    m.observe_wake()
    assert m.c["late"]["load"] == 4.0


def test_the_pressure_arm_counts_learning_done_on_the_tick_it_is_taught():
    """The meter reads AFTER the tick's teaching (real brain): total load == wake ticks x 1.0."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import require_isolated_home, throwaway_home

    probe = [0.0]

    def teach(brain, t):
        probe[0] += 1.0

    with throwaway_home():
        require_isolated_home()
        b = make_brain(seed=12201)
        run = sp.run_pressure_arm(
            b, "pressure", 60, teach, seed=12201, extra_circuits={"probe": lambda: list(probe)}
        )
    row = run.carrier["probe"]
    total = (
        row["baseline_load_per_wake"] * row["n_wake_periods"] + row["load_since_last_wake_onset"]
    )
    assert total == pytest.approx(sum(1 for a in run.plan if not a))


def _drive(m, w, script):
    """Replay a wake/sleep/onset script on meter `m` over weight cell `w`; return its trace."""
    trace = []
    for kind, dw in script:
        w[0] += dw
        if kind == "wake":
            m.observe_wake()
        elif kind == "sleep":
            m.observe_sleep()
        else:  # onset
            m.wake_period_done()
        trace.append((m.state(), m.request()))
    return trace


def test_a7_a_restored_meter_continues_exactly_like_the_uninterrupted_one():
    """Codex panel MAJOR (safety#4) REFUTED: a meter restored from `state()` over the same weights
    re-snapshots them (PressureMeter.__init__), so the next wake update is kept. Written red-able;
    it passed on its first run, so the claim is refuted and this stays as a regression pin.
    """
    script = [("wake", 1.0), ("wake", 1.0), ("onset", 0.0), ("sleep", 50.0), ("wake", 1.0)]
    tail = [("wake", 2.0), ("wake", 0.5), ("onset", 0.0), ("sleep", 9.0), ("wake", 3.0)]
    w1 = [0.0]
    whole = sp.PressureMeter({"c": lambda: list(w1)})
    _drive(whole, w1, script)
    full = _drive(whole, w1, tail)
    w2 = [0.0]
    head = sp.PressureMeter({"c": lambda: list(w2)})
    _drive(head, w2, script)
    restored = sp.PressureMeter({"c": lambda: list(w2)}, state=head.state())
    assert _drive(restored, w2, tail) == full


def test_a7_a_circuit_admitted_after_a_restore_carries_the_true_tick_not_zero():
    """The tick cursor is not in A7's carrier fields; the restorer supplies it (Codex panel safety#4)."""
    w = [0.0]
    m = sp.PressureMeter({"c": lambda: list(w)})
    for _ in range(5):
        m.observe_wake()
    restored = sp.PressureMeter({"c": lambda: list(w)}, state=m.state(), tick=m.tick)
    restored.admit("late", lambda: [0.0])
    assert restored.state()["late"]["admission_tick"] == 5
