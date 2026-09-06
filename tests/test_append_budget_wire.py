"""specs/022 L4 (FR-003): AppendBudget's first LIVE caller, and its constructed-overload red.

The budget carried a measured per-tag baseline from T010-i with NO caller at all -- reachable and
uncharged,
P3 in its plainest shape. These tests pin the wire end to end: home reports every append it knows
the tag
of, the brain owns the budget and closes it on the tick, a constructed overload TRIPS it on the live
path,
and a calm tick in the same test does not. The calm arm is the control: a budget that fires on
everything
would pass the red alone.
"""

import tempfile

import pytest

from snn_data_lake.substrate import home as home_mod
from snn_data_lake.substrate.lake_engine import APPEND_BASELINE, FLOOD_FACTOR, AppendBudget


@pytest.fixture
def scratch_home(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("NORD_HOME", td)
        home_mod.set_append_observer(None)
        yield td
        home_mod.set_append_observer(None)


def test_home_reports_every_append_whose_tag_it_knows(scratch_home):
    seen = []
    prev = home_mod.set_append_observer(lambda tag, n: seen.append((tag, n)))
    assert prev is None
    try:
        home_mod.save_named_state("l4probe", b"x" * 64)
        home_mod.KnowledgeLog().append({"cue": ["a"], "answer": "b.", "src": "chat"})
    finally:
        home_mod.set_append_observer(None)
    tags = {t for t, _ in seen}
    assert home_mod._T_STATE[0] in tags and home_mod._T_FACT[0] in tags, seen
    assert all(n > 0 for _, n in seen), seen


def test_a_raising_observer_never_breaks_a_durable_write(scratch_home):
    def boom(tag, n):
        raise RuntimeError("accounting exploded")

    home_mod.set_append_observer(boom)
    try:
        assert home_mod.save_named_state("l4probe", b"y" * 32) is True
        assert home_mod.load_named_state("l4probe") == b"y" * 32
    finally:
        home_mod.set_append_observer(None)


def test_a_constructed_overload_trips_the_budget_and_a_calm_tick_does_not(scratch_home):
    """THE RED, on the live path: the brain's own budget, fed by home's own observer, closed by the
    brain's
    own tick. The 0x07 ceiling comes from the recorded baseline, so the overload is sized from the
    measured
    p95 rather than from a number chosen here."""
    import json

    from snn_data_lake.cortex.brain import make_brain

    p95 = json.loads(APPEND_BASELINE.read_text(encoding="utf-8"))["per_tag_p95_bytes"]["7"]
    b = make_brain(seed=0, persist=True)
    assert b._append_budget is not None, "a persisting brain must own the budget"

    b.tick()
    calm_before = b.state()["append_floods"]
    home_mod.save_named_state("l4calm", b"z" * 128)  # far under the ceiling
    b.tick()
    assert b.state()["append_floods"] == calm_before, "a calm tick must not trip the budget"

    # OVERLOAD: strictly more than FLOOD_FACTOR x the recorded p95, in one tick, under
    # distinct names so dedup cannot silently swallow the writes.
    need = int(p95 * FLOOD_FACTOR) + 4096
    blob = b"q" * 8192
    written = 0
    i = 0
    while written < need:
        if home_mod.save_named_state(f"l4flood{i}", blob + bytes([i % 251])):
            written += len(blob)
        i += 1
    b.tick()
    st = b.state()
    assert st["append_floods"] > calm_before, (
        f"a {written} B burst against a {p95:.0f} B p95 "
        f"(x{FLOOD_FACTOR} bar) did not trip the budget"
    )
    assert "FLOOD" in st["append_flood_last"] and "0x07" in st["append_flood_last"], st


def test_the_budget_itself_still_refuses_an_unbaselined_tag():
    """The contract the live caller relies on: a tag with no recorded p95 never trips, so wiring the
    budget
    cannot invent a bound for a class nobody measured -- 0x01 in particular, which L6 left
    unbounded."""
    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 100.0}})
    bud.on_append(0x01, 10_000_000)
    assert bud.end_tick() == []
    bud.on_append(0x07, 10_000_000)
    assert bud.end_tick(), "a baselined tag far above its p95 must trip"
