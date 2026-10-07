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
    res = jms.run_instance(fx, jms.RED_SEEDS[3], budget=120, q=0.3)
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
    bad = jms.run_instance(fx, jms.RED_SEEDS[0], feasibility=False, budget=200, q=0.3)
    assert not bad["SA"].future_ok
    good = jms.run_instance(fx, jms.RED_SEEDS[0], budget=200, q=0.3)
    assert good["SA"].future_ok and good["SA"].retention_ok


def test_c8_sa_beats_random_on_the_planted_fixture():
    fx = jms.CoverFixture(k=8, dup=4)
    q = (fx.k + 1) / float(fx.n)
    runs = [jms.run_instance(fx, s, budget=200, q=q) for s in jms.RED_SEEDS]
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
    assert jms.net_savings(r, r, horizon=1000, eval_cost=0.5, restart_cost=0.0) < 0


def test_c26_a_judging_read_before_the_freeze_or_twice_is_refused():
    fx = jms.CoverFixture()
    m = jms.Meter(fx)
    with pytest.raises(jms.SelectionLeak):
        m.judge(tuple([1] * fx.n))
    m.freeze(tuple([1] * fx.n))
    m.judge(tuple([1] * fx.n))
    with pytest.raises(jms.SelectionLeak):
        m.judge(tuple([1] * fx.n))


def _decide(seeds):
    return jms.decide_b(seeds, instrument_gates_fired=True)


@pytest.mark.parametrize(
    "over, outcome",
    [
        ({("SA", "cost"): jms._smaller}, "SA"),
        ({("random", "cost"): jms._smaller}, "RANDOM-WIN"),
        ({(a, "future_ok"): False for a in jms.COMPETITORS}, "UNBUILT"),
        ({("SA", "future_ok"): False}, "GREEDY"),
        # R3 refutation R4: greedy is adopted over an INELIGIBLE incumbent (A3 step 3)
        ({(a, "future_ok"): False for a in ("incumbent", "SA", "random")}, "GREEDY"),
        ({(a, "future_ok"): False for a in ("SA", "greedy", "random")}, "INCUMBENT"),
    ],
)
def test_c22_decide_b_on_constructed_readings(over, outcome):
    assert _decide(jms.constructed_seeds(over)).outcome == outcome


def test_c22_a_size_win_never_adopts_past_a_failed_gate():
    assert (
        _decide(
            jms.constructed_seeds({("SA", "cost"): jms._smaller, ("SA", "future_ok"): False})
        ).outcome
        != "SA"
    )
    assert (
        _decide(jms.constructed_seeds({("SA", "cost"): jms._smaller, ("SA", "net"): -3.0})).outcome
        != "SA"
    )
    assert jms.decide_b(jms.constructed_seeds(), instrument_gates_fired=False).outcome == "REFUSED"
    missing = jms.constructed_seeds()
    del missing[0]["SA"]
    assert _decide(missing).outcome == "REFUSED"


def test_a_decision_without_its_gate_flag_is_a_type_error_not_a_pass():
    """R3 lead: `instrument_gates_fired` had a default of True; an omitted gate must not pass."""
    with pytest.raises(TypeError):
        jms.decide_b(jms.constructed_seeds())


def test_restart_cost_lowers_net_and_h_star_is_the_break_even():
    """2.4: net = saved x H - search - restart; H* = (search + restart) / saved."""
    full = jms.ArmResult("full", (1,) * 4, 24, 0, True, True, 10.0)
    res = jms.ArmResult("SA", (1, 0, 0, 0), 6, 20, True, True, 4.0)
    kw = {"eval_cost": 0.5}
    free = jms.net_savings(res, full, horizon=10, restart_cost=0.0, **kw)
    paid = jms.net_savings(res, full, horizon=10, restart_cost=7.0, **kw)
    assert free - paid == 7.0
    assert jms.break_even_horizon(res, full, restart_cost=5.0, **kw) == (20 * 0.5 + 5.0) / 6.0


def test_c9_the_retention_judgment_reds_a_mask_that_drops_the_old_unit():
    """C9 through Meter.judge (the judged retention), not the fixture's own predicate."""
    fx = jms.CoverFixture()
    m = jms.Meter(fx)
    dropped = tuple([1] * (fx.n - 1) + [0])
    m.freeze(dropped)
    _, retained = m.judge(dropped)
    assert retained is False


@pytest.mark.parametrize("n", [0, 1, 7, 9])
def test_a_decision_on_the_wrong_seed_count_is_refused(n):
    """Codex panel MAJOR: <8 (or 0) seeds ran through the 7 d.f. bar."""
    seeds = jms.constructed_seeds()
    seeds = (seeds * 2)[:n]
    assert _decide(seeds).outcome == "REFUSED"


def test_c26_judge_reads_only_the_frozen_winner():
    """Codex panel MAJOR: freeze() bound nothing, so any mask could be judged once."""
    fx = jms.CoverFixture()
    m = jms.Meter(fx)
    m.freeze(tuple([1] * fx.n))
    with pytest.raises(jms.SelectionLeak):
        m.judge(tuple([1] * (fx.n - 1) + [0]))
    m.judge(tuple([1] * fx.n))


def test_a13_a_budget_below_four_reports_random_and_sa_refused_and_decide_b_refuses():
    """Codex MINOR + R3: B < 4 REFUSES and REPORTS random and SA (A13) -- the instance returns
    with both arms absent and named in `_refused`, neither ran or was judged, decide_b reads the
    missing arm as REFUSED. A raise would kill a campaign loop instead of reporting the seed."""

    class One:
        n = 1
        evals = 0
        judges = 0

        def evaluate(self, mask):
            One.evals += 1
            return True, 1.0

        def cost(self, mask):
            return sum(mask)

        def judge(self, mask):
            One.judges += 1
            return True, True

        def work_per_tick(self, mask):
            return float(sum(mask))

    res = jms.run_instance(One(), 12001)
    assert "random" not in res and "SA" not in res
    assert res["_refused"] == ("random", "SA")
    assert One.evals == 0  # no random draw, no SA proposal
    assert One.judges == 3  # greedy, incumbent, no_prune only
    row = {"cost": 1, "future_ok": True, "retention_ok": True, "net": 1.0}
    seeds = [{a: row for a in jms.COMPETITORS if a in res} for _ in range(jms.N_SEEDS)]
    assert jms.decide_b(seeds, instrument_gates_fired=True).outcome == "REFUSED"


def test_one_active_unit_is_never_probed_for_removal_like_the_incumbent():
    """Codex panel MAJOR (p2p3#1, guard half): prune_unearned returns at <= 1 unit (ng_snn.py:359);
    the instrument's incumbent pass evaluated the removal too and charged two evaluations."""

    class Problem:
        n = 1

        def evaluate(self, mask):
            return True, 1.0

    meter = jms.Meter(Problem())
    mask, removed = jms._one_pass(meter, (1,))
    assert (mask, removed) == ((1,), 0)
    assert meter.spent == 0
