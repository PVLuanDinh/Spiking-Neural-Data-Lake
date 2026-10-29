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

    from snn_data_lake.engine.brain import make_brain

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


def test_a_fresh_brains_first_night_flushes_do_not_flood_and_a_burst_still_does(scratch_home):
    """The recorded seed must describe what a fresh brain writes today. Measured at ee224e50
    (stdlib): every new home FLOODed on its first three 0x04 ticks and its first three night
    flushes, because the seeds were 113 B and 12,381 B (August) while the brain writes about 1 KB
    of 0x04 a tick and 3-16 MB of 0x07 a night flush (pa-p4bd-1/2 moved the nightly skills' npz
    into the lake). A false alarm on every new brain teaches its reader to ignore the real one.

    Seed 7 is held out: the baseline's recorded method measures seed 0, so the bar is never fitted
    to the seed it is judged on. POSITIVE CONTROL in the same brain: a planted burst above the live
    bar must still trip it, or this test would pass on a budget that can no longer fire."""
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=7, persist=True)
    assert b._append_budget is not None, "a persisting brain must own the budget"
    flushes = []
    real_flush = b._flush_prototypes

    def counting_flush():
        flushes.append(1)
        return real_flush()

    b._flush_prototypes = counting_flush
    ticks = 0
    while len(flushes) < 3 and ticks < 200:
        b.tick()
        ticks += 1
    assert len(flushes) >= 3, f"only {len(flushes)} night flushes in {ticks} ticks: none judged"
    st = b.state()
    assert st["append_floods"] == 0, (
        f"{st['append_floods']} flood(s) on ordinary writes in a fresh brain's first {ticks} ticks "
        f"(3 night flushes); last: {st['append_flood_last']}"
    )

    bar, source = b._append_budget.bar(home_mod._T_STATE[0])
    assert bar is not None, source
    blob = 4 << 20
    written = 0
    i = 0
    while written <= bar:
        if home_mod.save_named_state(f"burst{i}", bytes([i % 251]) * blob):
            written += blob
        i += 1
    b.tick()
    st = b.state()
    assert st["append_floods"] > 0, f"a {written} B burst against a {bar:.0f} B bar did not trip"
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


# ---- specs/022 L4b: the bar is the tag's OWN RECENT RATE, because the quantity is not stationary
# --------


def _measured_growth():
    """The MEASURED per-tick 0x07 series from a 1200-tick brain: 52,763 -> 77,998 B
    over 67 writing ticks, carried by whole frames that grow with the brain
    (growing_learners.json x2.3, coverage_cells x18.7)."""
    return [round(52763 * (77998 / 52763) ** (i / 66)) for i in range(67)]


def test_ordinary_growth_does_not_flag_seeded_or_unseeded():
    from snn_data_lake.substrate.lake_engine import RUNNING_WINDOW

    series = _measured_growth()
    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    flags = 0
    for n in series:
        bud.on_append(0x07, n)
        flags += len(bud.end_tick())
    assert flags == 0, f"ordinary growth flagged {flags} times"
    assert len(series) > RUNNING_WINDOW
    # PRECISION, because the first draft of this test overstated it: at 1200 ticks the series tops
    # out at
    # 77,998 B, still UNDER the retired fixed bar of 12,381 x 8 = 99,048 B, so THIS series would not
    # have
    # flagged under the old comparison either. What crosses it is the LIVE daemon at 160,470 B/tick
    # -- a
    # far older brain on the same monotone curve. The next test is where the two comparisons
    # diverge.
    fixed_bar = 12381.0 * FLOOD_FACTOR
    assert max(series) < fixed_bar < 160470, (max(series), fixed_bar)


def test_a_real_spike_still_flags_once_the_window_is_all_real():
    from snn_data_lake.substrate.lake_engine import RUNNING_WINDOW

    series = _measured_growth()
    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    for n in series[:RUNNING_WINDOW]:
        bud.on_append(0x07, n)
        bud.end_tick()
    assert bud.bar(0x07)[1] == "its own recent rate"
    bud.on_append(0x07, int(max(series[:RUNNING_WINDOW]) * FLOOD_FACTOR * 2))
    v = bud.end_tick()
    assert v and "FLOOD" in v[0], v


def test_a_stale_seed_costs_a_bounded_number_of_floods_and_a_sustained_flood_goes_quiet():
    """The disclosed cost of an adaptive bar, pinned so it cannot drift.

    A stale seed stops alarming after a few real ticks -- which is what ended the live
    daemon's stuck alarm -- and BY THE SAME MECHANISM a genuine sustained flood goes
    quiet after the same few. Detecting a sustained RAMP is a slope instrument's job;
    this is a flood assert and does not claim to be one."""
    from snn_data_lake.substrate.lake_engine import RUNNING_WINDOW

    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    flagged = 0
    for _ in range(40):
        bud.on_append(0x07, 160470)  # the live daemon's measured per-tick rate
        if bud.end_tick():
            flagged += 1
    assert 1 <= flagged <= RUNNING_WINDOW // 4, flagged
    assert bud.bar(0x07)[1] == "its own recent rate"


def test_a_tag_with_no_seed_and_no_history_still_never_trips():
    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 100.0}})
    assert bud.bar(0x01) == (None, "unbaselined")
    bud.on_append(0x01, 10_000_000)
    assert bud.end_tick() == []


def test_an_unseeded_tag_stays_unbaselined_until_it_has_a_FULL_window_of_its_own():
    """The contract L4b briefly inverted.

    end_tick creates a window for anything that writes, so without a minimum-sample gate an
    unrecorded tag got a live flood bar from ONE sample -- a budget nobody measured, which is
    exactly what 0x01 (the class L6 left unbounded) would have acquired."""
    from snn_data_lake.substrate.lake_engine import RUNNING_WINDOW

    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 100.0}})
    for _ in range(RUNNING_WINDOW - 1):
        bud.on_append(0x01, 500)
        assert bud.end_tick() == []
        assert bud.bar(0x01) == (None, "unbaselined")
    bud.on_append(0x01, 4_000_000)  # a huge tick while still unbaselined must NOT flag
    assert bud.end_tick() == []
    # ...and once it HAS a window of its own, it is bounded like any other tag.
    for _ in range(RUNNING_WINDOW):
        bud.on_append(0x01, 500)
        bud.end_tick()
    assert bud.bar(0x01)[1] == "its own recent rate"
    bud.on_append(0x01, 10_000_000)
    assert bud.end_tick()


def test_a_flood_that_ARRIVES_AS_A_RAMP_is_invisible_at_any_magnitude():
    """The disclosed cost, at its worst rather than its best.

    A 32-window p95 is the value from three writing ticks back, so the no-flag condition is
    r**3 <= FLOOD_FACTOR: any per-write ratio <= 2.0 never flags however large it becomes.
    Pinned so the disclosure cannot quietly drift back to the softer one."""
    bud = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    n, flags = 12381.0, 0
    for _ in range(67):
        n *= 2.0
        bud.on_append(0x07, int(n))
        flags += len(bud.end_tick())
    assert flags == 0, flags
    assert n > 1e23  # the magnitude it reaches while never flagging
    # THE CONTRAST that makes it a cost and not a bug: the same 8x delivered in ONE tick does flag.
    step = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    step.on_append(0x07, int(12381.0 * 8 * 2))
    assert step.end_tick()
