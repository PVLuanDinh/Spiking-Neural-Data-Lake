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

from snn_data_lake.growth import growth_loop
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
    problem solves on the attempt after `solve_after_units` admitted units (None = never).

    BOUNDED (lead 2026-10-04): more than MAX_ATTEMPTS attempts fail the test with an assertion in
    milliseconds, so a loop that lost its attempt cap is a reachable RED, never a hang until a
    runner timeout."""

    MAX_ATTEMPTS = 200  # above the 52 single-neuron attempts the longest legitimate test makes

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
        assert self.attempts <= self.MAX_ATTEMPTS, "unbounded attempts: the loop lost its cap"
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
    # through THE shared guard (R1, lead ruling 16), never a row-local check
    from pathlib import Path

    from snn_data_lake.substrate.home import LiveHomeRefused

    with pytest.raises(LiveHomeRefused):
        Ledger(Path.home() / ".snn_data_lake" / "brain_lake.bin", VALUES)


def test_ledger_refuses_a_symlink_alias_of_the_live_lake(tmp_path, monkeypatch):
    # Codex cp5 #1: an isolated directory's lake.bin that points at the live lake file must be refused;
    # the live home is a decoy dir (never the real one), the alias a real symlink (no skip: a platform
    # without symlinks errors loudly instead of reading green)
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.home import LiveHomeRefused

    live = tmp_path / "live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"")
    iso = tmp_path / "iso"
    iso.mkdir()
    (iso / "lake.bin").symlink_to(live / "brain_lake.bin")
    monkeypatch.setattr(home, "_default_live_home", lambda: str(live))
    with pytest.raises(LiveHomeRefused):
        Ledger(iso / "lake.bin", VALUES)
    Ledger(iso / "fresh.bin", VALUES)  # control: a fresh file in the same isolated dir is admitted


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


def test_a_retry_that_solves_is_reported_solved_with_its_spend_in_the_baseline(tmp_path):
    # Codex cp5 #2: pause at 45/52, then a no-fresh-budget retry solves; the solve must stand
    learner = ScriptedLearner(["CAPACITY"], solve_after_units=None)
    sunk = []
    loop, ledger = _loop(tmp_path, learner, solved_sink=sunk.append)
    assert loop.run().state == "UNSOLVED"
    learner.solve_after_units = 45
    rec = loop.run()
    assert rec.state == "SOLVED" and not rec.open and rec.spend_units == 45
    assert len(sunk) == 1 and ledger.baseline() == 45


def test_a_retry_after_an_unresolved_run_never_grows_on_the_unspent_remainder(tmp_path):
    # Codex cp5 #3: 9 units spent, then UNRESOLVED (remainder 43 of 52); after "sleep" the diagnosis
    # reads CAPACITY, but with no new teaching the retry attempts once and grows nothing (2.6)
    learner = ScriptedLearner(["CAPACITY", "CAPACITY", "UNRESOLVED"])
    loop, ledger = _loop(tmp_path, learner)
    first = loop.run()
    assert first.state == "UNRESOLVED" and first.spend_units == 9
    learner.diagnoses = ["CAPACITY"]
    again = loop.run()
    assert again.spend_units == 9 and again.attempts == first.attempts + 1
    ledger.new_teaching("xor")  # control: new teaching DOES license growth again
    assert loop.run().spend_units > 9


def test_a_source_returning_more_than_the_gated_batch_is_refused_before_any_write(tmp_path):
    # Codex cp5 #4: batch=1 passes the 45/52 gate, a source that returns 9 would cross the budget
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("b", 4)
    ledger.record_spend("b", 45, evals=0)  # b sits at 45/52
    draw = _source(9)
    with pytest.raises(LedgerError):
        GrowthLoop(ScriptedLearner(["CAPACITY"]), "b", ledger, lambda n: draw(9), 1).run()
    assert ledger.ledger("b").spend_units == 45 and not ledger.units()


def test_an_empty_source_ends_the_run_instead_of_looping(tmp_path):
    learner = ScriptedLearner(["CAPACITY"])  # bounded: a lost exit is an assertion, not a hang
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    rec = GrowthLoop(learner, "e", ledger, lambda n: [], 9).run()
    assert rec.state == "UNSOLVED" and rec.open and rec.spend_units == 0


def test_a_non_positive_batch_is_refused(tmp_path):
    for bad in (0, -9):
        with pytest.raises(ValueError):
            GrowthLoop(ScriptedLearner(["CAPACITY"]), "z", Ledger(tmp_path / "l.bin", VALUES), _source(9), bad)


def test_a_reused_unit_id_is_refused_before_any_birth_row_is_written(tmp_path):
    # Codex cp5 #5 + the 2026-10-04 amendment's red: re-admitting a journaled id is refused
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    units = _source(9)(9)
    ledger.record_units(units[:3], "p", 0)
    assert ledger.known_unit(units[0].unit_id) and not ledger.known_unit(10**9)
    n = len(ledger.units())
    with pytest.raises(LedgerError):  # across batches
        ledger.record_units([units[0]], "p", 1)
    with pytest.raises(LedgerError):  # within one batch, nothing written even for the fresh first id
        ledger.record_units([units[5], units[5]], "p", 1)
    assert len(ledger.units()) == n


def test_a_reconstructed_source_that_restarts_its_ids_stops_the_loop_with_nothing_spent(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    GrowthLoop(ScriptedLearner(["CAPACITY"]), "p", ledger, _source(9), 9).run()  # ids 0..44
    with pytest.raises(LedgerError):  # a fresh source counts from 0 again
        GrowthLoop(ScriptedLearner(["CAPACITY"]), "q", ledger, _source(9), 9).run()
    assert ledger.ledger("q").spend_units == 0


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


def test_a_g6_shaped_row_round_trips_through_the_one_writer(tmp_path):
    # G6's public entry point: no private _append_raw, no second GoalStore writer on the lake (P2)
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    row = {
        "kind": "sleep_verdict",
        "unit_id": 7,
        "problem_id": "xor",
        "sleep_id": 3,
        "verdict": "kept",
        "mask_digest": "ab" * 32,
        "synapses": [[7, 0], [7, 1]],
        "retention_label": "immortal",
    }
    ledger.record_event(row)
    assert Ledger(tmp_path / "lake.bin", VALUES).events() == [row]
    with pytest.raises(ValueError):
        ledger.record_event({"kind": "unit", "unit_id": 8})  # the ledger's own kinds have their API
    with pytest.raises(ValueError):
        ledger.record_event({"kind": "sleep_bout", "score": float("nan")})


# ---- the trace-trained candidate source (2.5; 2.9 C7) -------------------------------------------


def _xor_items():
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features

    labels = ["GREET", "GOAL", "ARM"]
    learner = SpikingInteractionLearner(labels, n_in=52, seed=9900, t_post=6)
    texts = ["cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun"]
    ys = [0, 1, 1, 0]
    items = []
    for t, y in zip(texts, ys, strict=True):
        ranks = tuple(learner.encode(text_features(t, 52)))
        drives = [learner._drive(n, list(ranks)) for n in learner.neurons]
        onehot = [1.0 if c == y else 0.0 for c in range(len(labels))]
        items.append((ranks, onehot, growth_loop.label_probabilities(drives)))
    return items


def test_trace_training_is_deterministic_per_seed():
    items = _xor_items()
    a = growth_loop.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9901)
    b = growth_loop.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9901)
    c = growth_loop.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9902)
    assert a == b and a != c


def test_trace_training_moves_the_row():
    items = _xor_items()
    w0 = (0.1,) * 52
    assert growth_loop.trace_train(w0, 1.0, 12, 8.0, items, 0.05, 20, seed=9901) != w0
    assert growth_loop.trace_train(w0, 1.0, 12, 8.0, items, 0.0, 20, seed=9901) == w0


def _gate_findings(source):
    import ast

    from snn_data_lake.gates.eprop_transport_gate import _findings_in

    tree = ast.parse(source)
    return [f for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) for f in _findings_in(fn)]


def test_the_trace_rule_carries_no_weight_transport():
    import inspect

    assert _gate_findings(inspect.getsource(growth_loop)) == []


def test_a_transposed_readout_in_the_trace_rule_is_flagged():
    # C7's red: L_j read through the transposed readout weights must be caught by the gate
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(growth_loop.trace_train))
    variant = src.replace(
        "L = sum(b * (yc - pc) for b, yc, pc in zip(feedback, y, p))",
        "L = (W_out.T @ err)[0]",
    )
    assert variant != src
    assert _gate_findings(variant), "the transport gate did not flag a transposed readout"
