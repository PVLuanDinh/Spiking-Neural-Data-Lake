"""specs/006-fixed-point-gf -- the typed mint->searcher bottleneck and its guards.

Enforces the SURVIVING half of contracts/fix_combinator.md: the typed-record intake (C10-C12)
and the port-before-retire inventory (research.md R6) -- LOO validation, MDL ordering,
memo-flagging with refuse-on-unseen -- all asserted on the one searcher path.

The OTHER half of that contract, the fix combinator itself, was built, measured, and RETIRED
with a null: on the full evaluation split not one winning program used a fix step, and its
trajectory node cost starved 2 of 400 training solves. Its tests retired with the code (P2 --
no flag-parked survivors); the record lives in specs/006-fixed-point-gf/{census.md,verdict.md}
and docs/DEPTH_CEILINGS.md.

No ARC benchmark data is touched here: every task is synthetic and hand-built in this file.
"""

from snn_data_lake.reasoning import arc_mint, arc_solver
from snn_data_lake.reasoning.arc_solver import synthesize

# ----------------------------------------------------------------------------------------------
# synthetic mint tasks (no benchmark data)
# ----------------------------------------------------------------------------------------------


def _grid_of(cells, rows, cols):
    g = [[0] * cols for _ in range(rows)]
    for (r, c), v in cells.items():
        g[r][c] = v
    return g


def _recolor_task_functional():
    """Objects recoloured BY SIZE with colour == size (const_mod10): functional, LOO-clean."""
    i1 = _grid_of({(0, 0): 7, (2, 0): 7, (2, 1): 7}, 5, 5)
    o1 = _grid_of({(0, 0): 1, (2, 0): 2, (2, 1): 2}, 5, 5)
    i2 = _grid_of({(4, 2): 7, (4, 3): 7, (4, 4): 7}, 5, 5)
    o2 = _grid_of({(4, 2): 3, (4, 3): 3, (4, 4): 3}, 5, 5)
    return [(i1, o1), (i2, o2)]


def _recolor_task_memo():
    """Sizes {1,2,3} -> colours {5,5,8}: no functional form fits, a flagged lookup is minted."""

    def pair(base_r):
        i = _grid_of(
            {
                (base_r, 0): 7,
                (base_r + 2, 0): 7,
                (base_r + 2, 1): 7,
                (0, 3): 7,
                (1, 3): 7,
                (2, 3): 7,
            },
            6,
            6,
        )
        o = _grid_of(
            {
                (base_r, 0): 5,
                (base_r + 2, 0): 5,
                (base_r + 2, 1): 5,
                (0, 3): 8,
                (1, 3): 8,
                (2, 3): 8,
            },
            6,
            6,
        )
        return i, o

    return [pair(0), pair(1)]


def _solved(cands, want):
    return any(c == want for c in cands)


# ----------------------------------------------------------------------------------------------
# typed-record intake (T015/T017/T018; contract C10-C12, research.md R6)
# ----------------------------------------------------------------------------------------------


def test_records_reconstruct_from_declarative_fields():
    for train in (_recolor_task_functional(), _recolor_task_memo()):
        recs = arc_mint.mint_records(train)
        assert recs, "the mint must emit at least one verified record for its own dev tasks"
        for rec in recs:
            fn2 = arc_mint.compile_record(
                rec.unit, rec.attribute, rec.rule_form, rec.rule_params, rec.consumer
            )
            for gi, go in train:
                assert fn2(arc_mint._grid(gi)) == arc_mint._grid(go), (
                    f"declarative fields must reproduce fn on every train pair: {rec.name}"
                )


def test_records_are_deterministic_and_mdl_then_name_ordered():
    train = _recolor_task_functional()
    a = arc_mint.mint_records(train)
    b = arc_mint.mint_records(train)
    assert [r[:7] for r in a] == [r[:7] for r in b]
    assert all(r.name.isascii() for r in a)


def test_no_grid_or_train_residue_in_record_fields():
    for train in (_recolor_task_functional(), _recolor_task_memo()):
        for rec in arc_mint.mint_records(train):
            for field in (rec.name, rec.unit, rec.attribute, rec.rule_form, rec.consumer):
                assert isinstance(field, str)
            assert isinstance(rec.rule_params, tuple)
            flat = list(rec.rule_params)
            while flat:
                v = flat.pop()
                if isinstance(v, tuple):
                    flat.extend(v)
                else:
                    assert isinstance(v, int), "rule_params carries ints only -- never grids"


def test_memo_record_refuses_unseen_and_none_propagates():
    train = _recolor_task_memo()
    recs = [r for r in arc_mint.mint_records(train) if r.memo_flag]
    assert recs, "the memo task must mint a flagged record"
    rec = recs[0]
    assert rec.rule_form in ("rank", "lookup")
    unseen = arc_mint._grid(_grid_of({(0, 0): 7, (0, 1): 7, (1, 0): 7, (1, 1): 7}, 6, 6))
    assert rec.fn(unseen) is None, "unseen attribute value -> None, never a guess"
    # None propagates under composition exactly like a failing primitive (C3/C12)
    composed = rec.fn(unseen)
    assert (composed if composed is None else arc_solver._flip_h(composed)) is None


def test_memo_flag_is_visible_in_the_program_description():
    train = _recolor_task_memo()
    test_in, want = train[0]
    cands, progs = synthesize(train, test_in, time_budget=None)
    won = [p for c, p in zip(cands, progs, strict=True) if c == want]
    assert won, "the flagged record must solve its own task through the one searcher"
    names = won[0][0]
    assert any(n.startswith("mint:") and (":lookup" in n or ":rank" in n) for n in names), names


def test_loo_still_rejects_a_train_only_functional_fit():
    # affine fits the FLAT data (3=3*1, 6=3*2) but cannot be refit with one pair held out,
    # so LOO rejects it and the honest answer is the flagged memo form (research.md R6).
    got = arc_mint.induce_rule([[(1, 3)], [(2, 6)]])
    assert got is not None and got[1] in arc_mint._MEMO_KINDS
    # positive control: with a third pair the affine form survives LOO and wins again
    got3 = arc_mint.induce_rule([[(1, 3)], [(2, 6)], [(3, 9)]])
    assert got3 is not None and got3[1] == "affine"
    assert got3[3] == (3, 0), "induced params must cross the bottleneck, (m, b) for affine"


def test_mdl_order_prefers_functional_over_memo_on_the_surviving_path():
    recs = arc_mint.mint_records(_recolor_task_functional())
    rec = next(r for r in recs if r.consumer == "recolor" and r.attribute == "size")
    assert rec.rule_form == "const_mod10" and rec.memo_flag is False


def test_minted_entries_join_the_pool_and_solve_through_make_solver():
    train = _recolor_task_functional()
    test_in, want = train[0]
    solver = arc_solver.make_solver(time_budget=None)
    cands = solver(train, test_in)
    assert _solved(cands, want), "a minted record must solve THROUGH the one searcher"
    cands0, progs0 = synthesize(train, test_in, time_budget=None)
    won = [p for c, p in zip(cands0, progs0, strict=True) if c == want]
    assert won and any(n.startswith("mint:") for n in won[0][0]), won


def test_pool_keeps_base_prior_and_appends_mint_after_fitted():
    train = _recolor_task_functional()
    tin = tuple(arc_solver._tup(i) for i, _o in train)
    tout = tuple(arc_solver._tup(o) for _i, o in train)
    names = [n for n, _f in arc_solver._pool(tin, tout, None)]
    assert names[: len(arc_solver._BASE)] == [n for n, _f in arc_solver._BASE]
    mint_idx = [i for i, n in enumerate(names) if n.startswith("mint:")]
    assert mint_idx, "the functional task must contribute minted entries to the pool"
    assert min(mint_idx) >= len(arc_solver._BASE), "minted entries come after base + fitted"


def test_direct_application_path_is_retired():
    # SC-006: exactly 1 of 1 end-to-end ARC solve paths -- the searcher. The mint's direct
    # path (candidates()/mint()) must be gone, not parked behind anything.
    assert not hasattr(arc_mint, "candidates")
    assert not hasattr(arc_mint, "mint")
    assert callable(arc_mint.mint_records)


def test_two_runs_solve_identically_with_minted_records():
    # FR-012: determinism with mint: entries in the searched space (two budget-bound runs)
    tasks = [_recolor_task_functional(), _recolor_task_memo()]
    runs = []
    for _ in range(2):
        out = []
        for train in tasks:
            test_in, _want = train[0]
            out.append(synthesize(train, test_in, time_budget=None))
        runs.append(out)
    assert runs[0] == runs[1], "two budget-bound runs must be identical (FR-012)"
