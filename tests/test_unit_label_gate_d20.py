"""D20 / G6 P4: `check_saved` fed by G6's REAL resolver (`sleep_mask_apply.mask_committed`).

Spec: g6-sleep-durable-memory.md, DATED PRE-DATA AMENDMENT 2026-10-07 ("a false result is
PENDING, never REFUSED"; "Test owed at G6's P4: a red per false case, and a green for the
committed pair"). The resolver itself is covered on real lake rows by test_g6_mask_committed.py;
this file covers the WIRING over in-memory rows (no lake, no Brain). A mutant resolver (True for
everything) must fail the false-case checks, so they cannot pass on a stub.
"""

import pytest
from test_unit_label_gate import G, _good, _resolve, _saved

from snn_data_lake.instruments import sleep_mask_apply as sma

PREPARED = sma.prepare("night-1", 7, kept=["u1"], pruned=["u2"])
OTHER = sma.prepare("night-1", 7, kept=["u2"], pruned=["u1"])
DIGEST = PREPARED["mask_digest"]
COMMIT = {"kind": "sleep_mask_commit", "sleep_id": "night-1", "mask_digest": DIGEST}


def _outcome(rows, sleep_id="night-1", mask_digest=DIGEST, verdict="kept", resolver=None):
    rec = _good()
    resolver = resolver or (lambda s, m: sma.mask_committed(rows, s, m))
    return G.check_saved(
        _saved(rec, sleep_id=sleep_id, mask_digest=mask_digest, verdict=verdict),
        resolve_problem=_resolve,
        resolve_committed=resolver,
        first_sign={rec["unit_id"]: rec["sign"]},
    )


#: name -> (rows the resolver reads, sleep_id on the saved record, mask_digest on the saved record)
FALSE_CASES = {
    "unknown_sleep_id": ([COMMIT], "night-9", DIGEST),
    "digest_mismatch": ([COMMIT], "night-1", OTHER["mask_digest"]),
    "prepared_only": ([PREPARED], "night-1", DIGEST),
    "rolled_back": (
        [PREPARED, {"kind": "sleep_verdict", "sleep_id": "night-1", "mask_digest": DIGEST}],
        "night-1",
        DIGEST,
    ),
    "commit_of_other_digest": (
        [PREPARED, {**COMMIT, "mask_digest": OTHER["mask_digest"]}],
        "night-1",
        DIGEST,
    ),
    "commit_of_other_sleep": ([{**COMMIT, "sleep_id": "night-2"}], "night-1", DIGEST),
    "no_rows": ([], "night-1", DIGEST),
}


@pytest.mark.parametrize("case", sorted(FALSE_CASES))
def test_d20_false_case_is_pending_never_refused(case):
    rows, sleep_id, digest = FALSE_CASES[case]
    v = _outcome(rows, sleep_id, digest)
    assert v.outcome == "PENDING", (case, v)
    assert v.outcome != "REFUSED"


@pytest.mark.parametrize("verdict", ["kept", "pruned"])
def test_d20_committed_pair_gives_the_verdict(verdict):
    v = _outcome([PREPARED, COMMIT], verdict=verdict)
    assert (v.outcome, v.field, v.reason) == (verdict.upper(), None, None)


def test_d20_resolver_reads_only_the_exact_pair():
    seen = []

    def spy(s, m):
        seen.append((s, m))
        return True

    _outcome([], resolver=spy)
    assert seen == [("night-1", DIGEST)]


@pytest.mark.parametrize("case", sorted(FALSE_CASES))
def test_d20_mutant_true_resolver_fails_the_false_case(case):
    """A resolver that says True for everything turns each false case into KEPT: the PENDING
    assert above would fail on it, so the false-case tests do not pass on a stub."""
    rows, sleep_id, digest = FALSE_CASES[case]
    v = _outcome(rows, sleep_id, digest, resolver=lambda s, m: True)
    assert v.outcome != "PENDING"
