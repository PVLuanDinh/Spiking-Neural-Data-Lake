r"""G6 / T114 P4: the transactional MASK APPLICATION -- the `sleep_mask_prepared` producer, the commit row,
and the applied-mask check whose slot R2's `sleep` progress event carries.

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md, frozen at P0 and
amended 2026-10-03: section 2.16 A12 (transactional mask application), the DATED PRE-DATA AMENDMENT
2026-10-03 item 3 (`mask_digest` lives in `instruments/joint_mask_search.py`, ruling 17a, imported here
and never reimplemented) and the dated note 2026-10-07 (D20: `check_saved` reads committed verdicts).
DEPENDENCIES ruling 33: the CHECK (`check_applied_mask`) and its O_mask red land in the same arc as
THIS producer, so R2's `applied_mask_digest` sentinel gains its live input.

A12's protocol, per sleep that applies a mask:
  1. write `sleep_mask_prepared` {sleep_id, mask_digest, kept/pruned unit ids, synapse refs}   (`prepare`)
  2. write every domain unit's `sleep_verdict` row, each carrying `mask_digest`
  3. apply the mask to the active network                                                  (the consumer's)
  4. the learner checkpoint / `sleep` progress event records `applied_mask_digest`           (`expected_mask`)
  5. write `sleep_mask_commit` {sleep_id, mask_digest, post-apply state digest}             (`write_commit`)
Restore resolves a prepared mask WITHOUT a commit as ROLLED BACK, so an applied digest is only ever
honest when a commit row carries it -- which is what `check_applied_mask` enforces.

This module writes rows through G3's ONE lake writer (`Ledger.record_event`) and reads no active
network: the apply itself is the masked arm's, and the structural arm runs inside R3's owned campaign
(ruling 3), so nothing here runs a seed.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.sleep_mask_apply   (self-check, lake-free)
"""

from os import PathLike
from pathlib import Path

from snn_data_lake.instruments.joint_mask_search import mask_digest

#: The value a `sleep` event / checkpoint carries when NO mask was applied: R2's sentinel (ruling 33).
#: distinct-from: substrate/learner_events.py::NONE_SLOT (R2's own copy, on R2's branch) -- the same
#:   literal for the same slot; the integrator points both at one constant.
SENTINEL = "none"

# distinct-from: sleep/lake_consolidation.py::LakeConsolidation.consolidate, growth/growth_loop.py::Ledger.record_event,
#   gates/unit_label_gate.py::check_saved -- record pruning, another owner's row writer, and a saved-record
#   validator. This builds one mask row set and checks an applied-mask claim against it. Sites read.
APPLIED_UNCOMMITTED = "applied_mask_uncommitted"  # the event names a digest no commit row carries
APPLIED_NOT_PREPARED = (
    "applied_mask_not_prepared"  # the digest names a mask prepared at another tick
)
APPLIED_MISSING = (
    "applied_mask_missing"  # a committed mask was prepared for this tick; the slot still says none
)
APPLIED_EMPTY_MASK = (
    "applied_mask_empty"  # a claimed, committed mask with kept, pruned AND synapses all empty
)


def prepare(sleep_id, tick, *, kept=(), pruned=(), synapses=()):
    """A12 step 1's row: {kind, sleep_id, tick, kept, pruned, synapses, mask_digest}. The digest is
    `joint_mask_search.mask_digest` over THIS row (ruling 17a: one function, imported, never rewritten),
    so the row is self-identifying and the digest is order-invariant by construction."""
    if not kept and not pruned and not synapses:
        # EXACTLY the fully-empty shape (P2); a kept-only mask is NOT refused here
        raise ValueError(
            "prepare refuses the fully-empty mask: kept, pruned and synapses all empty"
        )
    row = {
        "kind": "sleep_mask_prepared",
        "sleep_id": sleep_id,
        "tick": int(tick),
        "kept": sorted(str(u) for u in kept),
        "pruned": sorted(str(u) for u in pruned),
        "synapses": [
            [str(kind), str(pre), str(post), int(bool(bit))] for kind, pre, post, bit in synapses
        ],
    }
    row["mask_digest"] = mask_digest(row)
    return row


def write_prepared(ledger, row):
    """A12 step 1: the prepared row, through G3's one writer (no private call, no second GoalStore)."""
    if row.get("kind") != "sleep_mask_prepared":
        raise ValueError(f"write_prepared takes a prepared row, got {row.get('kind')!r}")
    ledger.record_event(dict(row))
    return row


def write_commit(ledger, prepared, state_digest):
    """A12 step 5: the commit row {kind, sleep_id, mask_digest, state_digest}. Only a COMMITTED mask is
    applied state; a prepared row without one is rolled back on restore."""
    row = {
        "kind": "sleep_mask_commit",
        "sleep_id": prepared["sleep_id"],
        "mask_digest": prepared["mask_digest"],
        "state_digest": str(state_digest),
    }
    ledger.record_event(dict(row))
    return row


def _committed_digests(rows, sleep_id):
    return {
        str(r["mask_digest"])
        for r in rows
        if r.get("kind") == "sleep_mask_commit" and str(r.get("sleep_id")) == str(sleep_id)
    }


def mask_committed(rows_or_home, sleep_id, mask_digest) -> bool:
    """D20: a commit must match BOTH fields; preparation alone proves nothing.

    Accept event rows or an explicit home directory containing brain_lake.bin.
    Reuse GoalStore's lake codec and the existing commit-row reader (P2).
    """
    if isinstance(rows_or_home, (str, PathLike)):
        from snn_data_lake.growth.learning_progress import GoalStore

        rows_or_home = GoalStore(str(Path(rows_or_home) / "brain_lake.bin")).events()
    return str(mask_digest) in _committed_digests(rows_or_home, sleep_id)


def expected_mask(rows, sleep_id, tick):
    """A12 step 4's value: the `applied_mask_digest` the sleep event at (sleep_id, tick) must carry --
    the digest of the mask committed for that tick, or the SENTINEL when nothing was committed for it
    (a prepared-but-uncommitted mask is rolled back, so it must never be claimed as applied)."""
    committed = _committed_digests(rows, sleep_id)
    found = SENTINEL
    for r in rows:
        if (
            r.get("kind") == "sleep_mask_prepared"
            and str(r.get("sleep_id")) == str(sleep_id)
            and int(r.get("tick")) == int(tick)
        ):
            # lake order: the latest committed row for this tick wins, matching "fold = latest committed"
            found = str(r["mask_digest"]) if str(r.get("mask_digest")) in committed else SENTINEL
    return found


def _is_empty_mask_row(row):
    """A fully-empty mask: nothing kept, nothing pruned, no synapse ref. The digest covers all three
    fields, so every prepared row carrying one digest agrees on whether it is empty."""
    return not row.get("kept") and not row.get("pruned") and not row.get("synapses")


def check_applied_mask(event, rows):
    """Ruling 33's check (`learner_events.check_applied_mask` at integration): the applied-mask claim on
    one sleep progress event against the lake's mask rows. Returns the refusal reason, or None.

    `event` is any mapping carrying `applied_mask_digest`, `sleep_id` and `tick`; `rows` are the
    `sleep_mask_prepared` / `sleep_mask_commit` rows of the same lake. A planted inconsistency in any of
    the four ways below is the O_mask red -- a check that cannot fire is deleted, not kept (P8).

    ORDER IS LOAD-BEARING: the sentinel branch returns before any digest check, so the empty-mask refusal
    below sits only in the path where a digest is actually claimed. Run first, it would override the
    pinned APPLIED_MISSING outcome for a "none"/absent-key event on the very row it refuses."""
    applied = str(event.get("applied_mask_digest", SENTINEL))
    sleep_id, tick = event.get("sleep_id"), event.get("tick")
    if applied == SENTINEL:
        if expected_mask(rows, sleep_id, tick) != SENTINEL:
            return APPLIED_MISSING
        return None
    if applied not in _committed_digests(rows, sleep_id):
        return APPLIED_UNCOMMITTED
    if expected_mask(rows, sleep_id, tick) != applied:
        return APPLIED_NOT_PREPARED
    if any(
        r.get("kind") == "sleep_mask_prepared"
        and str(r.get("sleep_id")) == str(sleep_id)
        and int(r.get("tick")) == int(tick)
        and str(r.get("mask_digest")) == applied
        and _is_empty_mask_row(r)
        for r in rows
    ):
        # a forged fully-empty row (prepare() refuses it, so it can only be built by hand) with a valid
        # digest, committed and claimed at THIS (sleep_id, tick): proof of an applied mask that would
        # apply nothing. The (sleep_id, tick) match mirrors expected_mask above with the same coercion,
        # so a forged all-empty row at ANOTHER tick carrying a copied digest cannot refuse the honest
        # claim it would otherwise shadow (lens #716 Finding 1, upheld at #717).
        return APPLIED_EMPTY_MASK
    return None


def main():
    """Self-check (lake-free): the four O_mask reds each fire on a planted row set, the honest cases
    read None, the sentinel branch stays ahead of the empty-mask refusal, and the prepared row's digest
    is the one the check matches on."""
    good = prepare("s1-n3", 41, kept=["u1", "u3"], pruned=["u2"], synapses=[("feed", 0, 1, True)])
    commit_good = {
        "kind": "sleep_mask_commit",
        "sleep_id": "s1-n3",
        "mask_digest": good["mask_digest"],
        "state_digest": "ab" * 32,
    }
    rows = [good, commit_good]
    event = {"applied_mask_digest": good["mask_digest"], "sleep_id": "s1-n3", "tick": 41}
    assert check_applied_mask(event, rows) is None, "a committed mask at its own tick is honest"
    assert (
        check_applied_mask({**event, "applied_mask_digest": "0" * 64}, rows) == APPLIED_UNCOMMITTED
    )
    assert check_applied_mask(event, [good]) == APPLIED_UNCOMMITTED, "no commit, no applied mask"
    assert check_applied_mask({**event, "applied_mask_digest": SENTINEL}, rows) == APPLIED_MISSING
    other = prepare("s1-n3", 44, kept=["u1"], pruned=[], synapses=[])
    commit_other = {**commit_good, "mask_digest": other["mask_digest"]}
    both = [good, other, commit_good, commit_other]
    claim_other = {**event, "applied_mask_digest": other["mask_digest"]}
    assert check_applied_mask(claim_other, both) == APPLIED_NOT_PREPARED
    assert check_applied_mask({**event, "applied_mask_digest": SENTINEL}, [good]) is None, (
        "an uncommitted prepare claims nothing"
    )
    assert check_applied_mask(event, both) is None, (
        "the other tick's commit does not disturb this one"
    )
    # P2: prepare() refuses EXACTLY the fully-empty shape; a kept-only row still prepares
    try:
        prepare("s1-n3", 41)
        raise AssertionError("prepare must refuse the fully-empty mask")
    except ValueError:
        pass
    # P7 (corrected #662): the kept-only mask committed at ITS OWN tick stays accepted -- the new guard
    # is keyed on kept AND pruned AND synapses all empty, so it must not broaden to non-pruning masks
    claim_kept_only = {"applied_mask_digest": other["mask_digest"], "sleep_id": "s1-n3", "tick": 44}
    assert check_applied_mask(claim_kept_only, both) is None, (
        "a kept-only mask is not an empty mask"
    )
    # P3/P4: the forged bypass -- a fully-empty prepared row built BY HAND (prepare refuses it, so it
    # can only be forged), digested by the one imported function, committed and claimed at its own tick
    forged = {
        "kind": "sleep_mask_prepared",
        "sleep_id": "s1-n3",
        "tick": 41,
        "kept": [],
        "pruned": [],
        "synapses": [],
    }
    forged["mask_digest"] = mask_digest(forged)
    commit_forged = {
        "kind": "sleep_mask_commit",
        "sleep_id": "s1-n3",
        "mask_digest": forged["mask_digest"],
        "state_digest": "ab" * 32,
    }
    forged_rows = [forged, commit_forged]
    claim_forged = {"applied_mask_digest": forged["mask_digest"], "sleep_id": "s1-n3", "tick": 41}
    assert check_applied_mask(claim_forged, forged_rows) == APPLIED_EMPTY_MASK, (
        "a forged fully-empty committed mask is refused at its own tick"
    )
    # the guard is scoped to the (sleep_id, tick) the event names, like expected_mask: a forged all-empty
    # row at ANOTHER tick carrying a COPIED digest must not refuse the honest claim (lens #716 Finding 1,
    # upheld at #717)
    forged_stolen = {
        "kind": "sleep_mask_prepared",
        "sleep_id": "s1-n3",
        "tick": 44,
        "kept": [],
        "pruned": [],
        "synapses": [],
        "mask_digest": good["mask_digest"],
    }
    assert check_applied_mask(event, [good, commit_good, forged_stolen]) is None, (
        "a forged empty row at another tick cannot refuse the honest claim"
    )
    # P5: the SAME forged committed row with the string sentinel -- and with the key absent -- stays
    # APPLIED_MISSING: the sentinel branch returns before the empty-mask guard can fire
    assert (
        check_applied_mask({**claim_forged, "applied_mask_digest": SENTINEL}, forged_rows)
        == APPLIED_MISSING
    )
    assert check_applied_mask({"sleep_id": "s1-n3", "tick": 41}, forged_rows) == APPLIED_MISSING
    # P6: Python None is a separate claimed value (str(None) == "None"), not the string sentinel
    assert (
        check_applied_mask({**claim_forged, "applied_mask_digest": None}, forged_rows)
        == APPLIED_UNCOMMITTED
    )
    assert len(good["mask_digest"]) == 64
    print(
        "applied_mask check: O_mask reds reachable, the committed case honest, sentinel preserved"
    )


if __name__ == "__main__":
    main()
