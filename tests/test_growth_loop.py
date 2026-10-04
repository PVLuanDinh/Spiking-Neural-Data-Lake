"""specs/022 T111 (G3) P3: the grow-attempt loop, the budget pause, the ledger and the unit journal.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.4 (every no-growth path
is finite and persisted), 2.6 (the loop, the attempt cap, whole-batch admission, the budget
mechanism, the retry rule, one stop rule, the ledger on GoalStore's _T_GOAL rows, the IMMORTAL unit
journal, the solve event), 2.9 C8/C13/C18 and 2.12a (an explicit isolated lake path; the live lake
refused). The learner here is a scripted fixture adapter (its diagnosis and attempt outcomes are set
by the test), because the taught learner's port lands after R2's P1; the loop's own mechanics are
what these tests pin. Every lake is pytest's tmp_path.
"""

import pytest

from snn_data_lake.growth.grown_code import GrownUnit
from snn_data_lake.growth.growth_loop import (
    BudgetValues,
    GrowthLoop,
    Ledger,
    LedgerError,
    retention_ok,
)
from snn_data_lake.growth.growth_stop import GrowthStop

VALUES = BudgetValues(first=52, beta=1.5, k_max=3)


class ScriptedLearner:
    """Fixture adapter: `diagnoses` is consumed per diagnose() call (the last value repeats); the
    problem solves on the attempt after `solve_after_units` admitted units (None = never)."""

    def __init__(self, diagnoses, solve_after_units=None):
        self.diagnoses = list(diagnoses)
        self.solve_after_units = solve_after_units
        self.units = []
        self.attempts = 0

    def diagnose(self):
        return self.diagnoses.pop(0) if len(self.diagnoses) > 1 else self.diagnoses[0]

    def admit(self, units):
        self.units.extend(units)

    def attempt(self, epochs):
        self.attempts += 1
        return self.solve_after_units is not None and len(self.units) >= self.solve_after_units


def _source(batch):
    counter = iter(range(10**6))

    def draw(n):
        return [
            GrownUnit(
                next(counter), (0.0,) * 4, 1.0, 12, 1 if i < max(1, batch - 1) else -1, (0.0,)
            )
            for i in range(n)
        ]

    return draw


def _loop(tmp_path, learner, batch=9, gated=True, problem="xor", **kw):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    return GrowthLoop(learner, problem, ledger, _source(batch), batch, gated=gated, **kw), ledger


# ---- the live lake is refused; the path is explicit (2.12a)


def test_ledger_refuses_a_missing_path():
    with pytest.raises(LedgerError):
        Ledger(None, VALUES)


def test_ledger_refuses_the_live_lake():
    from pathlib import Path

    with pytest.raises(LedgerError):
        Ledger(Path.home() / ".snn_data_lake" / "brain_lake.bin", VALUES)


# ---- the loop's terminal states
# ---------------------------------------------------------------------


def test_capacity_problem_grows_until_solved_and_emits_one_solve(tmp_path):
    seen = []
    loop, ledger = _loop(tmp_path, ScriptedLearner(["CAPACITY"], solve_after_units=18))
    loop.solved_sink = seen.append
    rec = loop.run()
    assert rec.state == "SOLVED" and rec.spend_units == 18 and rec.attempts == 2
    assert [r.problem_id for r in seen] == ["xor"]


def test_unsolvable_problem_pauses_at_the_budget_and_stays_open(tmp_path):
    # C8 (Codex b1 G3 #12): pause when the next WHOLE batch would exceed the budget, spend <= budget
    loop, _ = _loop(tmp_path, ScriptedLearner(["CAPACITY"]), batch=9)
    rec = loop.run()
    assert rec.state == "UNSOLVED" and rec.open
    assert rec.spend_units == 45 and rec.spend_units <= rec.budget
    (tmp_path / "s").mkdir()
    single, _ = _loop(tmp_path / "s", ScriptedLearner(["CAPACITY"]), batch=1)
    assert single.run().spend_units == 52


def test_literal_loop_with_the_diagnosis_disabled_still_pauses(tmp_path):
    # C8(a): the trigger misfires (DATA ignored) and the pause must still hold
    loop, _ = _loop(tmp_path, ScriptedLearner(["DATA"]), gated=False, diagnosis=False)
    rec = loop.run()
    assert rec.state == "UNSOLVED" and rec.spend_units == 45


def test_data_diagnosis_grows_nothing_and_keeps_the_problem_open(tmp_path):
    learner = ScriptedLearner(["DATA"])
    loop, _ = _loop(tmp_path, learner)
    rec = loop.run()
    assert rec.state == "CONTRADICTORY" and rec.open and rec.spend_units == 0
    assert learner.attempts == 0


def test_literal_loop_also_refuses_to_grow_on_data(tmp_path):
    learner = ScriptedLearner(["DATA"])
    loop, _ = _loop(tmp_path, learner, gated=False)
    rec = loop.run()
    assert rec.state == "CONTRADICTORY" and rec.spend_units == 0 and learner.attempts == 0


def test_separable_problem_that_never_solves_hits_the_attempt_cap(tmp_path):
    learner = ScriptedLearner(["SEPARABLE"])
    loop, _ = _loop(tmp_path, learner)
    rec = loop.run()
    assert rec.state == "LEARNING_RULE_DEFICIT" and rec.spend_units == 0
    assert learner.attempts == 3 and rec.open


def test_unresolved_problem_hits_the_attempt_cap(tmp_path):
    loop, _ = _loop(tmp_path, ScriptedLearner(["UNRESOLVED"]))
    rec = loop.run()
    assert rec.state == "UNRESOLVED" and rec.spend_units == 0 and rec.attempts == 3


def test_separable_task_solves_without_growth_in_the_gated_loop(tmp_path):
    loop, _ = _loop(tmp_path, ScriptedLearner(["SEPARABLE"], solve_after_units=0))
    rec = loop.run()
    assert rec.state == "SOLVED" and rec.spend_units == 0 and rec.attempts == 1


def test_literal_loop_grows_before_every_attempt(tmp_path):
    learner = ScriptedLearner(["SEPARABLE"], solve_after_units=0)
    loop, _ = _loop(tmp_path, learner, gated=False)
    rec = loop.run()
    assert rec.state == "SOLVED" and rec.spend_units == 9


# ---- the budget mechanism and the retry rule
# ---------------------------------------------------------


def test_budget_baseline_counts_only_positive_spend_solves(tmp_path):
    # D-G2-10's ledger unit check: solved spends {10, 20, 30} plus one zero-spend solve -> beta x 20
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    for i, spend in enumerate([10, 20, 30, 0]):
        pid = f"p{i}"
        ledger.open_problem(pid, n_codes=4)
        if spend:
            ledger.record_spend(pid, spend, evals=spend * 32)
        ledger.record_attempt(pid, solved=True)
    rec = ledger.open_problem("next", n_codes=4)
    assert rec.budget == 1.5 * 20 and rec.baseline_at_open == 20


def test_first_problem_budget_is_the_first_value(tmp_path):
    rec = Ledger(tmp_path / "lake.bin", VALUES).open_problem("xor", n_codes=4)
    assert rec.budget == 52 and rec.baseline_at_open is None


def test_open_refuses_a_budget_below_the_distinct_codes(tmp_path):
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "lake.bin", VALUES).open_problem("big", n_codes=53)


def test_new_teaching_draws_a_fresh_budget_at_most_k_times(tmp_path):
    loop, ledger = _loop(tmp_path, ScriptedLearner(["CAPACITY"]))
    assert loop.run().budgets_drawn == 1
    for expected in (2, 3):
        ledger.new_teaching("xor")
        assert loop.run().budgets_drawn == expected
    with pytest.raises(LedgerError):
        ledger.new_teaching("xor")
    rec = loop.run()
    assert rec.state == "UNSOLVED" and rec.spend_units == 3 * 45


def test_retry_after_sleep_draws_no_fresh_budget(tmp_path):
    loop, _ = _loop(tmp_path, ScriptedLearner(["CAPACITY"]))
    first = loop.run()
    again = loop.run()  # a retry with no new teaching: attempts only, no growth
    assert again.spend_units == first.spend_units and again.attempts == first.attempts + 1


# ---- one stop rule (2.6): the pause lives in GrowthStop
# -----------------------------------------------


def test_growth_stop_problem_key_verdict():
    stop = GrowthStop()
    stop.observe_problem("xor", solved=False, spend=36, budget=52, batch=9)
    assert stop.should_grow_problem("xor")
    stop.observe_problem("xor", solved=False, spend=45, budget=52, batch=9)
    assert not stop.should_grow_problem("xor")
    stop.observe_problem("xor", solved=True, spend=9, budget=52, batch=9)
    assert not stop.should_grow_problem("xor")


def test_growth_stop_problem_key_leaves_region_verdicts_alone():
    stop = GrowthStop()
    stop.observe_problem("cortex", solved=True, spend=0, budget=52, batch=9)
    assert stop.should_grow(
        "cortex"
    )  # an unobserved REGION still grows: the namespaces are disjoint


# ---- the ledger survives a restart (P4b; C13)
# ---------------------------------------------------------


def test_a_fresh_ledger_rebuilds_byte_equal(tmp_path):
    loop, ledger = _loop(tmp_path, ScriptedLearner(["CAPACITY"], solve_after_units=18))
    rec = loop.run()
    again = Ledger(tmp_path / "lake.bin", VALUES).ledger("xor")
    assert again == rec


def test_a_dropped_ledger_row_is_refused(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", n_codes=4)
    ledger.record_spend("xor", 9, evals=96)
    ledger._append_raw(
        {"kind": "problem", "problem_id": "xor", "seq": 99, "op": "attempt", "solved": False}
    )
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "lake.bin", VALUES).ledger("xor")


# ---- the unit journal (2.6, P4d)
# -----------------------------------------------------------------------


def test_every_admitted_unit_is_journaled_with_its_weights(tmp_path):
    loop, ledger = _loop(tmp_path, ScriptedLearner(["CAPACITY"], solve_after_units=9))
    loop.run()
    rows = Ledger(tmp_path / "lake.bin", VALUES).units()
    assert len(rows) == 9
    assert all(r["problem_id"] == "xor" and r["origin"] == "born" for r in rows)
    assert all(len(r["w_in"]) == 4 and "sign" in r and "batch_id" in r for r in rows)
    assert len({r["unit_id"] for r in rows}) == 9


def test_unknown_kinds_pass_through_and_sleep_save_is_kept(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.record_sleep_save({"unit_ids": [1, 2], "bout": 7})
    ledger._append_raw({"kind": "sleep_verdict", "unit_id": 1, "kept": True})
    fresh = Ledger(tmp_path / "lake.bin", VALUES)
    kinds = [r["kind"] for r in fresh.events()]
    assert kinds == ["sleep_save", "sleep_verdict"]


# ---- retention (2.9 C11 / D7)
# ---------------------------------------------------------------------------


def test_retention_ok_against_the_factual_bound():
    assert retention_ok(before=4, after=4, f_bound=0)
    assert not retention_ok(before=4, after=3, f_bound=0)
    assert retention_ok(before=4, after=3, f_bound=1)
