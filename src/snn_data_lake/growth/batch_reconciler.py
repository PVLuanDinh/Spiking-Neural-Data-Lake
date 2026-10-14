"""G3 restart reconciliation of the existing immortal birth/spend journal.

Contract: specs/022-unified-brain-arc/evidence/g3-growth-loop.md, amendments
2026-10-06 (d)/(e), C13. Read the SAME Ledger/GoalStore codec in lake order.
Incomplete batches are refused by problem, batch and unit ids, before any loop
effect. Missing search cost or a partial batch is never silently re-derived;
immortal birth records are never removed. This is named-refusal reconciliation,
not an automatic recovery claim or an R2 learner-admission receipt.
"""

from pathlib import Path


def reconcile_batches(ledger, problem_id):
    """Return charged batch ids, or refuse an incomplete/corrupt journal by name.

    No writer, second store, new event kind, or repair append is created here.
    The old aggregate birth-minus-spend count missed a spent first batch followed
    by a half-written second batch on the retry path (which never calls _grow).
    """
    from snn_data_lake.growth.grown_code import GrownUnit, apply_unit_row
    from snn_data_lake.growth.growth_loop import LedgerError

    # ponytail: one ordered scan of the incumbent store, not another journal.
    rows = ledger._store.events()
    pending, charged, seen_units, seen_batches = [], [], set(), set()
    current = None
    opened = False
    seq = 0

    def refuse(reason, batch=None, units=()):
        raise LedgerError(
            f"problem {problem_id!r}: unsettled batch {batch!r}, "
            f"unit ids {list(units)}: {reason}; restart refused"
        )

    def count(value, name, batch=None):
        if type(value) is not int or value < 0:
            refuse(f"invalid {name}: {value!r}", batch)
        return value

    for row in rows:
        if row.get("problem_id") != problem_id:
            continue
        kind = row.get("kind")
        if kind == "unit":
            batch = count(row.get("batch_id"), "batch_id")
            uid = count(row.get("unit_id"), "unit_id", batch)
            if not opened:
                refuse("birth without problem open", batch, (uid,))
            if uid in seen_units:
                refuse("duplicate birth identity", batch, (uid,))
            if batch in seen_batches or (pending and batch != current):
                refuse("interleaved or reused batch identity", batch, (uid,))
            try:
                unit = GrownUnit(
                    uid,
                    tuple(row["w_in"]),
                    row["threshold"],
                    row["window"],
                    row["sign"],
                    tuple(row["out"]),
                )
                apply_unit_row((), {"op": "admit", "unit": unit})
                digest = ledger._birth_row(unit, row, problem_id, batch)["digest"]
            except (KeyError, TypeError, ValueError) as exc:
                refuse(f"invalid birth body: {exc}", batch, (uid,))
            if row.get("digest") != digest:
                refuse("birth digest mismatch", batch, (uid,))
            seen_units.add(uid)
            current = batch
            pending.append(uid)
        elif kind == "problem":
            if type(row.get("seq")) is not int or row["seq"] != seq:
                refuse(
                    f"ledger sequence gap: expected {seq}, got {row.get('seq')!r}", current, pending
                )
            seq += 1
            op = row.get("op")
            if op == "open":
                if opened:
                    refuse("duplicate problem open", current, pending)
                opened = True
            elif not opened:
                refuse("problem operation without open", current, pending)
            elif op == "spend":
                units = count(row.get("units"), "spend units", current)
                count(row.get("evals"), "spend evaluations", current)
                if units:
                    if units != len(pending):
                        refuse(
                            f"spend {units} does not cover {len(pending)} birth rows",
                            current,
                            pending,
                        )
                    charged.append(current)
                    seen_batches.add(current)
                    pending, current = [], None
            elif pending and op != "run":
                refuse(f"{op!r} before the batch's spend", current, pending)
    if pending:
        refuse(
            "birth rows have no spend row; batch size/search cost cannot be recovered",
            current,
            pending,
        )
    if opened:
        ledger.ledger(problem_id)  # reuse the incumbent fold, including its sequence red
    return tuple(charged)


def main():
    """The reachable restart red uses the actual lake and codec in a throwaway home."""
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.growth.growth_loop import BudgetValues, Ledger, LedgerError, make_admission
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home() as home:
        ledger = Ledger(Path(home) / "g3-reconcile.bin", BudgetValues(52, 1.5))
        ledger.open_problem("fixture", 4)
        assert reconcile_batches(ledger, "fixture") == ()
        units = [GrownUnit(0, (0.0,) * 52, 1.0, 12, 1, (0.0,))]
        admission = make_admission(
            site=(0.5 / 14, 1.5 / 4),
            centre_um=(0.0, 0.0),
            density_per_mm2=2.7e4,
            seed=9900,
        )
        labels, _ = admission(
            units,
            "fixture",
            0,
            resolve_problem=lambda _: True,
            known_unit=lambda _: False,
        )
        ledger.record_units(units, labels, "fixture", 0)
        try:
            reconcile_batches(ledger, "fixture")
        except LedgerError as exc:
            assert "batch 0" in str(exc) and "unit ids [0]" in str(exc)
        else:
            raise AssertionError("red: half-admitted batch was accepted")
    print("batch_reconciler: clean control passes; incomplete batch refused by name")


if __name__ == "__main__":
    main()
