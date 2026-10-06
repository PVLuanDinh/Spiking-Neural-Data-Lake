"""G6 / T114 part (b): the joint-mask search, `decide_b` and `mask_digest` (C7-C10, C22, C26).

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md (2.4, 2.11,
2.16 A3/A13/A14, the dated pre-data amendment 2026-10-03). Fixtures only: no brain, no lake.
"""

import pytest

from snn_data_lake.instruments import joint_mask_search as jms

ROW = {
    "sleep_id": "s1-n3",
    "tick": 7,
    "kept": ["u2", "u1"],
    "pruned": ["u3"],
    "synapses": [("feed", "x", "u1", 1), ("readout", "u3", "r0", 0)],
}


def test_self_check_runs():
    jms.main()


def test_mask_digest_is_order_invariant_and_flips_on_one_kept_bit():
    perm = {**ROW, "kept": ["u1", "u2"], "synapses": list(reversed(ROW["synapses"]))}
    assert jms.mask_digest(ROW) == jms.mask_digest(perm)
    for i in range(len(ROW["synapses"])):
        syn = [list(s) for s in ROW["synapses"]]
        syn[i][3] = 1 - syn[i][3]
        assert jms.mask_digest({**ROW, "synapses": [tuple(s) for s in syn]}) != jms.mask_digest(ROW)
    assert jms.mask_digest({**ROW, "tick": 8}) != jms.mask_digest(ROW)
    assert jms.mask_digest({**ROW, "kept": ["u1"], "pruned": ["u2", "u3"]}) != jms.mask_digest(ROW)
    assert len(jms.mask_digest(ROW)) == 64


def test_budgets_are_equal_and_every_evaluation_is_charged():
    fx = jms.CoverFixture(k=8, dup=4)
    res = jms.run_instance(fx, 3, budget=120, q=0.3)
    assert res["SA"].evaluations == res["random"].evaluations == res["_B"] == 120
    m = jms.Meter(fx, budget=2)
    m.evaluate((0,) * fx.n)  # an infeasible proposal still costs one evaluation
    m.evaluate((1,) * fx.n)
    with pytest.raises(jms.BudgetExhausted):
        m.evaluate((1,) * fx.n)


def test_sa_refuses_a_budget_below_four():
    fx = jms.CoverFixture()
    import random

    with pytest.raises(ValueError):
        jms.arm_sa(jms.Meter(fx, budget=3), fx.n, random.Random(0))


def test_c7_disabled_feasibility_lets_an_unsolving_winner_through_and_the_judge_reds():
    fx = jms.CoverFixture(k=8, dup=4)
    bad = jms.run_instance(fx, 0, feasibility=False, budget=200, q=0.3)
    assert not bad["SA"].future_ok
    good = jms.run_instance(fx, 0, budget=200, q=0.3)
    assert good["SA"].future_ok and good["SA"].retention_ok


def test_c8_sa_beats_random_on_the_planted_fixture():
    fx = jms.CoverFixture(k=8, dup=4)
    q = (fx.k + 1) / float(fx.n)
    runs = [jms.run_instance(fx, s, budget=200, q=q) for s in range(8)]
    assert all(r["SA"].cost == fx.minimum() for r in runs)
    assert jms._sup("C8", [r["random"].cost - r["SA"].cost for r in runs]).met


def test_c9_dropping_the_old_tasks_unit_breaks_retention():
    fx = jms.CoverFixture()
    assert fx.evaluate(tuple([1] * (fx.n - 1) + [0]))[0] is False
    assert fx.evaluate(tuple([1] * fx.n))[0] is True


def test_c10_a_zero_saving_mask_reads_net_below_zero():
    fx = jms.CoverFixture()
    full = tuple([1] * fx.n)
    r = jms.ArmResult("x", full, fx.cost(full), evaluations=3, work_per_tick=fx.work_per_tick(full))
    assert jms.net_savings(r, r, horizon=1000, eval_cost=0.5) < 0


def test_c26_a_judging_read_before_the_freeze_or_twice_is_refused():
    fx = jms.CoverFixture()
    m = jms.Meter(fx)
    with pytest.raises(jms.SelectionLeak):
        m.judge(tuple([1] * fx.n))
    m.freeze()
    m.judge(tuple([1] * fx.n))
    with pytest.raises(jms.SelectionLeak):
        m.judge(tuple([1] * fx.n))


@pytest.mark.parametrize(
    "over, outcome",
    [
        ({("SA", "cost"): jms._smaller}, "SA"),
        ({("random", "cost"): jms._smaller}, "RANDOM-WIN"),
        ({(a, "future_ok"): False for a in jms.COMPETITORS}, "UNBUILT"),
        ({("SA", "future_ok"): False}, "GREEDY"),
        ({(a, "future_ok"): False for a in ("SA", "greedy", "random")}, "INCUMBENT"),
    ],
)
def test_c22_decide_b_on_constructed_readings(over, outcome):
    assert jms.decide_b(jms.constructed_seeds(over)).outcome == outcome


def test_c22_a_size_win_never_adopts_past_a_failed_gate():
    assert (
        jms.decide_b(
            jms.constructed_seeds({("SA", "cost"): jms._smaller, ("SA", "future_ok"): False})
        ).outcome
        != "SA"
    )
    assert (
        jms.decide_b(
            jms.constructed_seeds({("SA", "cost"): jms._smaller, ("SA", "net"): -3.0})
        ).outcome
        != "SA"
    )
    assert jms.decide_b(jms.constructed_seeds(), instrument_gates_fired=False).outcome == "REFUSED"
    missing = jms.constructed_seeds()
    del missing[0]["SA"]
    assert jms.decide_b(missing).outcome == "REFUSED"
