"""G3 C13 restart reds on throwaway lakes, using the existing GoalStore codec."""

import os
import subprocess
import sys

import pytest

from snn_data_lake.growth.batch_reconciler import reconcile_batches
from snn_data_lake.growth.grown_code import GrownUnit
from snn_data_lake.growth.growth_loop import (
    BudgetValues,
    GrowthLoop,
    Ledger,
    LedgerError,
    make_admission,
)

VALUES = BudgetValues(52, 1.5)
ADMISSION = make_admission(
    site=(0.5 / 14, 1.5 / 4),
    centre_um=(0.0, 0.0),
    density_per_mm2=2.7e4,
    seed=9900,
)


def births(ledger, batch, ids):
    units = [GrownUnit(uid, (0.0,) * 52, 1.0, 12, 1, (0.0,)) for uid in ids]
    labels, _ = ADMISSION(
        units,
        "xor",
        batch,
        resolve_problem=lambda _: True,
        known_unit=ledger.known_unit,
    )
    ledger.record_units(units, labels, "xor", batch)


def fresh_process(path, home):
    env = dict(os.environ, NORD_HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "\n".join(
                [
                    "import sys",
                    "from snn_data_lake.growth.growth_loop import "
                    "Ledger, BudgetValues, LedgerError",
                    "from snn_data_lake.growth.batch_reconciler import reconcile_batches",
                    "ledger = Ledger(sys.argv[1], BudgetValues(52, 1.5))",
                    "try: print(reconcile_batches(ledger, 'xor'))",
                    "except LedgerError as exc: print(exc); sys.exit(3)",
                ]
            ),
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


@pytest.mark.parametrize("stage", ["partial_birth", "full_birth", "run"])
@pytest.mark.pool
def test_half_admitted_batch_is_refused_by_name_in_fresh_process(tmp_path, stage):
    path = tmp_path / "lake.bin"
    ledger = Ledger(path, VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0] if stage == "partial_birth" else [0, 1])
    if stage == "run":
        ledger.record_run("xor")
    before = path.read_bytes()
    child = fresh_process(path, tmp_path)
    assert child.returncode == 3, child.stderr
    assert "problem 'xor'" in child.stdout and "batch 0" in child.stdout
    assert "unit ids" in child.stdout and "restart refused" in child.stdout
    assert path.read_bytes() == before  # no repair journal, no deletion, no guessed spend


@pytest.mark.pool
def test_fully_charged_journal_survives_fresh_process_without_append(tmp_path):
    path = tmp_path / "lake.bin"
    ledger = Ledger(path, VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0, 1])
    ledger.record_run("xor")
    ledger.record_spend("xor", 2, 32)
    ledger.record_attempt("xor", False, 400)
    births(ledger, 1, [2])
    ledger.record_spend("xor", 1, 16)
    before = path.read_bytes()
    child = fresh_process(path, tmp_path)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "(0, 1)"
    assert path.read_bytes() == before


def test_retry_is_refused_before_any_attempt_or_append(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0])
    ledger.record_run("xor")
    before = ledger._path.read_bytes()

    class Untouched:
        def diagnose(self):
            raise AssertionError("diagnosis before reconciliation")

        def attempt(self, epochs):
            raise AssertionError("attempt before reconciliation")

    loop = GrowthLoop(
        Untouched(),
        "xor",
        ledger,
        lambda _: pytest.fail("source before reconciliation"),
        1,
        admission=ADMISSION,
        resolve_problem=lambda _: True,
    )
    with pytest.raises(LedgerError, match="unsettled batch 0"):
        loop.run()
    assert ledger._path.read_bytes() == before


def test_spend_cannot_cover_a_partial_birth_batch(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0])
    ledger.record_run("xor")
    ledger.record_spend("xor", 2, 32)
    with pytest.raises(LedgerError, match="spend 2 does not cover 1"):
        reconcile_batches(ledger, "xor")


def test_aggregate_counts_cannot_hide_a_missing_spend(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0])
    ledger.record_spend("xor", 2, 0)  # planted overcharge
    births(ledger, 1, [1])  # old aggregate unsettled() == 0
    assert ledger.unsettled("xor") == 0
    with pytest.raises(LedgerError, match="unsettled batch 0"):
        reconcile_batches(ledger, "xor")


def test_unrelated_sleep_rows_and_zero_unit_search_spend_pass_through(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", 4)
    ledger.record_run("xor")
    ledger.record_spend("xor", 0, 32)
    ledger.record_event({"kind": "sleep_save", "payload": {"untouched": True}})
    assert reconcile_batches(ledger, "xor") == ()
    assert ledger.events()[-1]["payload"] == {"untouched": True}


def test_corrupt_birth_digest_names_batch_and_unit(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    ledger.open_problem("xor", 4)
    births(ledger, 0, [0])
    ledger._store._events[-1]["threshold"] = 5.0  # planted corrupt body
    with pytest.raises(LedgerError, match="batch 0, unit ids \\[0\\].*digest mismatch"):
        reconcile_batches(ledger, "xor")


@pytest.mark.parametrize("module", ["batch_reconciler", "feed_readout"])
@pytest.mark.pool
def test_registered_selfcheck_entrypoint_runs_in_a_fresh_private_home(tmp_path, module):
    child = subprocess.run(
        [sys.executable, "-B", "-m", "snn_data_lake.growth." + module],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, NORD_HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE="1"),
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert module + ":" in child.stdout
