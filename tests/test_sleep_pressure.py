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
    """R3 refutation R2: without G3's solved_sink request ticks the solve arm is refused, not run as the oracle."""
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
    from snn_data_lake.instruments.sleep_pruner_census import _assert_scratch_home
    from snn_data_lake.substrate.home import throwaway_home

    out = {}
    for arm, req in (("clock", None), ("oracle", [3, 30])):
        with throwaway_home():
            _assert_scratch_home()
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
    assert run.plan != gate or not run.advanced
    assert sp.check_budgets({"clock": gate, "oracle": run.plan})[0]
