"""G9 ROUND 2 / G3 2.6 and 2.8: first budget versus the unset SEQUENCE beta."""

import pytest

from snn_data_lake.growth.growth_loop import (
    BudgetValues,
    CanonicalGrowthAdapter,
    Ledger,
)
from snn_data_lake.instruments import taught_problems as tp

REFUSAL = "BETA is unset until the dated pre-data amendment enters G3's value"


def test_first_budget_and_restored_positive_spend_sequence_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "BETA", None)
    path = tmp_path / "lake.bin"
    values = BudgetValues(first=tp.FIRST_BUDGET, beta=None)
    ledger = Ledger(path, values)
    first = ledger.open_problem("ii.xor", 4)
    assert first.budget == 52 and first.baseline_at_open is None
    # Constructed ledger rows, not a measured learned solution or calibration.
    ledger.record_spend("ii.xor", 10, 10)
    ledger.record_attempt("ii.xor", True)
    restored = Ledger(path, values)
    assert restored.baseline() == 10
    before = path.read_bytes()
    with pytest.raises(ValueError) as error:
        restored.open_problem("i.S0C1", 8)
    assert str(error.value) == REFUSAL
    assert path.read_bytes() == before
    assert restored.problem_ids() == ["ii.xor"]


def test_zero_spend_solve_keeps_first_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "BETA", None)
    ledger = Ledger(tmp_path / "lake.bin", BudgetValues(first=52, beta=None))
    ledger.open_problem("zero", 4)
    ledger.record_attempt("zero", True)
    assert ledger.baseline() is None
    assert ledger.open_problem("ii.xor", 4).budget == 52


def test_readiness_keeps_sequence_debt_and_all_independent_debts(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "BETA", None)
    learner = tp.canonical_incumbent(9900)
    adapter = CanonicalGrowthAdapter(learner, tp.PROBLEMS["ii.xor"], seed=9900)
    ledger = Ledger(tmp_path / "lake.bin", BudgetValues(first=52, beta=None))
    assert adapter.readiness(ledger=ledger) == CanonicalGrowthAdapter._MISSING
    assert adapter.readiness(ledger=ledger, sequence=True) == (
        *CanonicalGrowthAdapter._MISSING,
        "source-owned BETA baseline",
    )
    ledger.open_problem("old", 4)
    ledger.record_spend("old", 10, 10)
    ledger.record_attempt("old", True)
    assert "source-owned BETA baseline" in adapter.readiness(ledger=ledger)
