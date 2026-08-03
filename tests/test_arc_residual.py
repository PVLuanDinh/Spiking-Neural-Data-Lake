"""specs/006-fixed-point-gf DESIGN B -- the versioned descriptor alphabet and the induced selector.

docs/INVENTION.md section 3 widens an existing induction axis from an AUTHORED enumeration (8
(unit, attribute) tuples, 3 of arc_scene's public measurements) to an INDUCED CONJUNCTION over
measured descriptors. These tests hold the two things that widening is allowed to cost nothing:

  * the TYPED BOTTLENECK stays reconstructible -- a record's declarative fields, selector included,
    still rebuild its program with no mint-internal state crossing (contract C10/C11);
  * the selector actually SELECTS -- the pre-registered falsifier's clause (b). A record that a
    degenerate always-true conjunction reproduces has not shown the mechanism did anything.

No ARC benchmark data is touched here: every task is synthetic and hand-built in this file.
"""

from snn_data_lake.reasoning import arc_mint, arc_residual


def _g(rows):
    return tuple(tuple(int(v) for v in row) for row in rows)


def _ring_task():
    """Objects WITH a hole are recoloured; objects without one are left alone. The mint cannot emit
    this at all without a selector -- under the authored all-units skeleton the untouched object
    makes the task unexplainable."""
    i1 = _g(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0, 0],
            [0, 3, 0, 3, 0, 5],
            [0, 3, 3, 3, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    o1 = _g(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 8, 8, 8, 0, 0],
            [0, 8, 0, 8, 0, 5],
            [0, 8, 8, 8, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    i2 = _g(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 4, 4, 4, 0, 0],
            [0, 4, 0, 4, 0, 0],
            [0, 4, 4, 4, 0, 2],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    o2 = _g(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 8, 8, 8, 0, 0],
            [0, 8, 0, 8, 0, 0],
            [0, 8, 8, 8, 0, 2],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    return [(i1, o1), (i2, o2)]


# ----------------------------------------------------------------------------------------------
# the versioned enumeration (contract C10/C11 -- literals, never a pointer into mint state)
# ----------------------------------------------------------------------------------------------


def test_alphabet_is_a_dense_versioned_enumeration():
    ids = [pid for pid, _n, _f in arc_residual.DESCRIPTORS]
    assert ids == list(range(arc_residual.N_DESCRIPTORS)), "predicate ids must be dense"
    assert len(set(arc_residual.DESCRIPTOR_NAMES)) == arc_residual.N_DESCRIPTORS
    assert isinstance(arc_residual.DESCRIPTOR_VERSION, int)
    assert all(n.isascii() for n in arc_residual.DESCRIPTOR_NAMES)


def test_descriptors_are_integers_and_deterministic():
    g = _ring_task()[0][0]
    arc_residual.clear_cache()
    rows = arc_residual.descriptor_rows(g)
    arc_residual.clear_cache()
    assert arc_residual.descriptor_rows(g) == rows, "descriptor rows must be deterministic"
    for row in rows:
        for v in row:
            assert v is None or isinstance(v, int), (
                "a descriptor is an int or absent, never an object"
            )


def test_selector_carries_only_literal_ints():
    for rec in arc_mint.mint_records(_ring_task()):
        assert isinstance(rec.selector, tuple)
        for lit in rec.selector:
            assert isinstance(lit, tuple) and len(lit) == 2
            pid, val = lit
            assert isinstance(pid, int) and isinstance(val, int)
            assert 0 <= pid < arc_residual.N_DESCRIPTORS, (
                "a literal names a predicate in the alphabet"
            )


# ----------------------------------------------------------------------------------------------
# the induction itself
# ----------------------------------------------------------------------------------------------


def test_greedy_covering_stops_on_no_strict_improvement():
    # two identical rows on opposite sides: no literal can strictly reduce the negatives -> refuse
    g = _ring_task()[0][0]
    rows = arc_residual.descriptor_rows(g)
    assert arc_residual.induce_selector([rows[0]], [rows[0]]) is None
    # nothing to exclude -> the EMPTY conjunction, which is the always-true selector
    assert arc_residual.induce_selector(list(rows), []) == ()


def test_literal_must_hold_on_every_positive():
    g = _ring_task()[0][0]
    rows = arc_residual.descriptor_rows(g)
    sel = arc_residual.induce_selector([rows[0]], [r for r in rows[1:]])
    assert sel is not None
    assert arc_residual.selector_matches(rows[0], sel)
    assert not any(arc_residual.selector_matches(r, sel) for r in rows[1:])


def test_literal_cap_bounds_length_never_choice():
    g = _ring_task()[0][0]
    rows = arc_residual.descriptor_rows(g)
    for cap in (1, 2, arc_residual.SELECTOR_LITERAL_CAP):
        sel = arc_residual.induce_selector([rows[0]], list(rows[1:]), cap=cap)
        assert sel is None or len(sel) <= cap


# ----------------------------------------------------------------------------------------------
# the pre-registered falsifier's clauses, mechanised
# ----------------------------------------------------------------------------------------------


def test_selector_bearing_record_is_minted_and_reconstructs():
    train = _ring_task()
    recs = arc_mint.mint_records(train)
    assert recs, "a task where only SOME units are written must now mint a record"
    bearing = [r for r in recs if r.selector]
    assert bearing, "and that record must carry at least one induced literal"
    for rec in bearing:
        fn2 = arc_mint.compile_record(
            rec.unit, rec.attribute, rec.rule_form, rec.rule_params, rec.consumer, rec.selector
        )
        for gi, go in train:
            assert fn2(gi) == go, f"declarative fields must reproduce fn: {rec.name}"


def test_degenerate_always_true_selector_does_not_reproduce_the_record():
    # falsifier clause (b): if the empty conjunction reproduced the answer, the induction did zero
    train = _ring_task()
    bearing = [r for r in arc_mint.mint_records(train) if r.selector]
    assert bearing
    for rec in bearing:
        degenerate = arc_mint.compile_record(
            rec.unit, rec.attribute, rec.rule_form, rec.rule_params, rec.consumer, ()
        )
        assert not all(degenerate(gi) == go for gi, go in train), (
            f"an always-true selector reproduces {rec.name} -- the conjunction is doing no work"
        )


def test_selector_that_selects_nothing_returns_none():
    # a primitive that cannot act must FAIL like any other, never hand back its input as an answer
    train = _ring_task()
    rec = next(r for r in arc_mint.mint_records(train) if r.selector)
    empty = _g([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert rec.fn(empty) is None


def test_authored_object_attributes_survive_inside_the_alphabet():
    # every attribute the authored enumeration carries is also reachable as a measured descriptor,
    # so the two inductions are compared on the same vocabulary and not on different ones.
    for attr in ("shapecells", "shapeclass", "size", "side", "color", "nholes"):
        assert attr in arc_residual.DESCRIPTOR_NAMES, f"authored attribute unmeasurable: {attr}"


def _unselectable_task():
    """Some units change and some do not, and NO conjunction over the alphabet separates them: the
    changed units share no descriptor value that the unchanged unit lacks.

    This is the shape that caught the design's subsumption claim. The all-units induction handles it
    (an unchanged unit is just an observation mapping its attribute to the colour it already has);
    the induced conjunction REFUSES it. Neither mechanism contains the other, so retiring either one
    loses solves -- measured on evaluation task ae58858e, reproduced here without benchmark data."""
    i1 = _g([[1, 0, 2, 0, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    o1 = _g([[7, 0, 8, 0, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    i2 = _g([[0, 0, 0, 0, 0], [1, 0, 2, 0, 3], [0, 0, 0, 0, 0]])
    o2 = _g([[0, 0, 0, 0, 0], [7, 0, 8, 0, 3], [0, 0, 0, 0, 0]])
    return [(i1, o1), (i2, o2)]


def test_neither_recolor_induction_subsumes_the_other():
    # docs/INVENTION.md 3.4 item 2 claims the induced conjunction subsumes the authored enumeration,
    # so the six ("object", attr) tuples can go. MEASURED FALSE -- here is the counterexample.
    units = {u for u, _a, _f in arc_mint._RECOLOR_UNITS}
    assert units == {"region", "object"}, "the authored object enumeration must NOT be retired"

    train = _unselectable_task()
    pairs = [{"in": gi, "out": go} for gi, go in train]
    assert arc_mint._build_selected_recolor(pairs) == [], (
        "this task must be OUTSIDE the induced conjunction's reach -- otherwise it proves nothing"
    )
    recs = arc_mint.mint_records(train)
    assert any(r.consumer == "recolor" and r.unit == "object" for r in recs), (
        "the authored all-units path must still cover what the conjunction refuses"
    )


def test_the_two_inductions_are_distinct_consumers():
    # not one path behind a flag: they compile through different branches and the record says which
    train = _ring_task()
    bearing = [r for r in arc_mint.mint_records(train) if r.selector]
    assert bearing and all(r.consumer == "recolor_sel" for r in bearing)


def test_records_stay_deterministic_with_selectors():
    train = _ring_task()
    a = arc_mint.mint_records(train)
    b = arc_mint.mint_records(train)
    assert [r[:8] for r in a] == [r[:8] for r in b]
