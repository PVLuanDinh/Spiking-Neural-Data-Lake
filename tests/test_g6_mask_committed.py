"""D20's real-lake false cases and committed-pair control; A12/D21 effect is unbuilt."""

import subprocess
import sys

import pytest

from snn_data_lake.growth.growth_loop import BudgetValues, Ledger
from snn_data_lake.growth.learning_progress import GoalStore
from snn_data_lake.instruments import joint_mask_search as jms
from snn_data_lake.instruments import sleep_mask_apply as sma
from snn_data_lake.substrate import home

VALUES = BudgetValues(first=52, beta=1.5, k_max=3)


@pytest.mark.parametrize("source", ["rows", "home"])
@pytest.mark.parametrize(
    ("case", "committed"),
    [
        ("prepared_only", False),
        ("rolled_back", False),
        ("different_digest", False),
        ("different_sleep", False),
        ("committed_pair", True),
    ],
)
def test_d20_real_lake_pair(tmp_path, monkeypatch, source, case, committed):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    lake = tmp_path / "brain_lake.bin"
    ledger = Ledger(lake, VALUES)
    prepared = sma.prepare("night-1", 7, kept=["u1"], pruned=["u2"])
    digest = jms.mask_digest(prepared)
    assert prepared["mask_digest"] == digest
    sma.write_prepared(ledger, prepared)
    if case == "rolled_back":
        # Construct A12's post-rollback input, not its still-unbuilt Brain restore.
        home.save_state("d20-restored-checkpoint", {"applied_mask_digest": sma.SENTINEL})
        ledger.record_event({"kind": "sleep_verdict", "sleep_id": "night-1", "mask_digest": digest})
        restored = home.load_state("d20-restored-checkpoint")
        assert restored["applied_mask_digest"] != digest
    elif case == "different_digest":
        other = sma.prepare("night-1", 7, kept=["u2"], pruned=["u1"])
        assert jms.mask_digest(other) != digest
        sma.write_commit(ledger, other, "ab" * 32)
    elif case == "different_sleep":
        # A forged other-night commit with the target digest must not count.
        sma.write_commit(ledger, {**prepared, "sleep_id": "night-2"}, "ab" * 32)
    elif case == "committed_pair":
        sma.write_commit(ledger, prepared, "ab" * 32)
    before = lake.read_bytes()
    rows = GoalStore(str(lake)).events()
    assert rows[0] == prepared  # replay from real persisted _T_GOAL rows
    if case == "rolled_back":
        assert [r["kind"] for r in rows] == ["sleep_mask_prepared", "sleep_verdict"]
    supplied = rows if source == "rows" else tmp_path
    assert sma.mask_committed(supplied, "night-1", digest) is committed
    assert lake.read_bytes() == before  # querying retains prepared/verdict/checkpoint bytes


def test_commit_alone_is_sufficient_and_generator_rows_work(tmp_path):
    lake = tmp_path / "brain_lake.bin"
    ledger = Ledger(lake, VALUES)
    prepared = sma.prepare("night-1", 7, kept=["u1"])
    sma.write_commit(ledger, prepared, "ab" * 32)
    rows = GoalStore(str(lake)).events()
    assert [r["kind"] for r in rows] == ["sleep_mask_commit"]
    assert sma.mask_committed(iter(rows), "night-1", jms.mask_digest(prepared)) is True
    assert sma.mask_committed(str(tmp_path), "night-1", jms.mask_digest(prepared)) is True


def test_committed_pair_replays_in_fresh_process(tmp_path):
    ledger = Ledger(tmp_path / "brain_lake.bin", VALUES)
    prepared = sma.prepare("night-1", 7, kept=["u1"], pruned=["u2"])
    sma.write_prepared(ledger, prepared)
    sma.write_commit(ledger, prepared, "ab" * 32)
    code = (
        "import sys; from snn_data_lake.instruments.sleep_mask_apply import mask_committed; "
        "assert mask_committed(sys.argv[1], 'night-1', sys.argv[2]) is True; "
        "assert mask_committed(sys.argv[1], 'night-2', sys.argv[2]) is False; "
        "print('fresh-process lake pair PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path), jms.mask_digest(prepared)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "fresh-process lake pair PASS"
