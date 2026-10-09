"""G6 / T114 P4: the `sleep_mask_prepared` producer and the applied-mask check (ruling 33, A12).

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md 2.16 A12,
the DATED PRE-DATA AMENDMENT 2026-10-03 item 3 (the `mask_digest` ships in `joint_mask_search`,
ruling 17a) and the dated note 2026-10-07 (D20). Every check here has a constructed, reachable
red: a check that cannot fire is deleted (P8). Brain-free and lake-file-free except where a row
is written through G3's ONE ledger writer on pytest's tmp_path.
"""

import pytest

from snn_data_lake.growth.growth_loop import BudgetValues, Ledger
from snn_data_lake.instruments import joint_mask_search as jms
from snn_data_lake.instruments import sleep_mask_apply as sma

VALUES = BudgetValues(first=52, beta=1.5, k_max=3)
ROW = sma.prepare("s1-n3", 41, kept=["u1", "u3"], pruned=["u2"], synapses=[("feed", 0, 1, True)])
COMMIT = {
    "kind": "sleep_mask_commit",
    "sleep_id": "s1-n3",
    "mask_digest": ROW["mask_digest"],
    "state_digest": "ab" * 32,
}
EVENT = {"applied_mask_digest": ROW["mask_digest"], "sleep_id": "s1-n3", "tick": 41}


def test_self_check_runs():
    assert sma.main() is None


def test_the_producers_digest_is_the_one_function_not_a_second():
    # ruling 17a: one `mask_digest`, imported; the prepared row is self-identifying by it
    assert ROW["mask_digest"] == jms.mask_digest(ROW) and len(ROW["mask_digest"]) == 64
    assert (
        sma.prepare("s1-n3", 41, kept=["u3", "u1"], pruned=["u2"], synapses=[("feed", 0, 1, True)])[
            "mask_digest"
        ]
        == ROW["mask_digest"]
    )
    assert (
        sma.prepare(
            "s1-n3", 41, kept=["u1", "u3"], pruned=["u2"], synapses=[("feed", 0, 1, False)]
        )["mask_digest"]
        != ROW["mask_digest"]
    )


def test_the_prepared_and_commit_rows_round_trip_through_the_one_writer(tmp_path):
    ledger = Ledger(tmp_path / "lake.bin", VALUES)
    sma.write_prepared(ledger, ROW)
    sma.write_commit(ledger, ROW, "ab" * 32)
    rows = Ledger(tmp_path / "lake.bin", VALUES).events()
    assert [r["kind"] for r in rows] == ["sleep_mask_prepared", "sleep_mask_commit"]
    assert rows[0]["mask_digest"] == ROW["mask_digest"] and rows[0]["tick"] == 41
    with pytest.raises(ValueError):
        sma.write_prepared(ledger, COMMIT)  # only A12's step 1 row is a prepared row


def test_the_sentinel_slot_gains_its_live_input_from_the_producer():
    # ruling 33: the slot R2's P2 shipped as "none" now has a value the producer computes
    assert sma.expected_mask([], "s1-n3", 41) == sma.SENTINEL == "none"  # nothing prepared yet
    assert (
        sma.expected_mask([ROW], "s1-n3", 41) == sma.SENTINEL
    )  # prepared, uncommitted: rolled back
    assert sma.expected_mask([ROW, COMMIT], "s1-n3", 41) == ROW["mask_digest"]
    assert sma.expected_mask([ROW, COMMIT], "s1-n3", 44) == sma.SENTINEL  # another tick


# ---- the O_mask red: three reachable refusals ------------------------------------------------
def test_o_mask_red_an_applied_digest_no_committed_mask_carries_is_refused():
    assert sma.check_applied_mask({**EVENT, "applied_mask_digest": "0" * 64}, [ROW, COMMIT]) == (
        sma.APPLIED_UNCOMMITTED
    )
    assert sma.check_applied_mask(EVENT, [ROW]) == sma.APPLIED_UNCOMMITTED


def test_o_mask_red_a_mask_prepared_at_another_tick_is_refused():
    other = sma.prepare("s1-n3", 44, kept=["u1"], pruned=[], synapses=[])
    rows = [ROW, other, COMMIT, {**COMMIT, "mask_digest": other["mask_digest"]}]
    assert sma.check_applied_mask({**EVENT, "applied_mask_digest": other["mask_digest"]}, rows) == (
        sma.APPLIED_NOT_PREPARED
    )


def test_o_mask_red_the_sentinel_while_a_committed_mask_was_prepared_is_refused():
    # the sleeping slot kept saying "none" although this tick's mask is applied state
    assert sma.check_applied_mask(
        {**EVENT, "applied_mask_digest": sma.SENTINEL}, [ROW, COMMIT]
    ) == (sma.APPLIED_MISSING)


def test_the_committed_case_is_honest_and_an_uncommitted_prepare_claims_nothing():
    assert sma.check_applied_mask(EVENT, [ROW, COMMIT]) is None
    assert sma.check_applied_mask({**EVENT, "applied_mask_digest": sma.SENTINEL}, [ROW]) is None
