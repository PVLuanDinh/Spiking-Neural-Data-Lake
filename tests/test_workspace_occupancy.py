"""The live-brain occupancy harness the workspace docstrings owe.

`thalamus/global_workspace.py` and `CHANGELOG.md` quote live figures (occupancy {1: 117, 2: 3}
before the inhibitory rheobase, co-occupancy 0 -> 3 after the one-clock repoint) that until this
file had NO runnable source -- "there was no runnable harness for any of it" is the CHANGELOG's
own sentence. This is that harness, and it doubles as the standing guard on the rheobase's
deletion-on-null trigger: if live occupancy is ever pinned at exactly `capacity` on >= 95% of wake
ticks, the capacity bound has become a rank-order truncation with an inert formula beside it and
the kernel must be REVERTED, not tuned.

BASELINES, measured 2026-08-07 on this exact protocol (seed 0, 120 forced-DAY ticks, 52 ms/tick):
occupancy {1: 14, 2: 57, 3: 46, 4: 3} against capacity 4; pinned-at-capacity 3/120 (2%);
co-occupancy (>= 2 slots) 106/120; access-without-focus 1/120; slot census perception 185 /
coalition 93; head census perception 97 / coalition 23; `thought` 21 distinct / 117 changes.
Pre-rheobase the same protocol read occupancy {1: 117, 2: 3} and co-occupancy 3/120.

Asserts are DIRECTION AND FLOOR, never exact figures: CI runs a different torch build than the
local gate (`docs/POLICIES.md` P5), so a build-sensitive exact count may legitimately differ.
Floors sit far below the measured values and far above the pre-rheobase pathology.

distinct-from: measure_live_occupancy.py -- that scratchpad script is this file's untracked
measurement twin; THIS is the tracked, asserting instrument.
"""

from collections import Counter

from snn_data_lake.engine.brain import make_brain

N_TICKS = 120


def test_live_workspace_occupancy():
    b = make_brain(seed=0, refit_every=10, persist=False)
    ws = b.workspace
    # IDENTITY FIRST: every number below is about the arm's own bus. A copy would produce
    # plausible numbers about nothing.
    assert ws is b.brain.arm.workspace, "Brain.workspace must BE the arm's workspace instance"

    rows = []
    for _ in range(N_TICKS):
        b.clock.phase = 0.25  # force subjective DAY: wake ticks only
        b.tick()
        ac = getattr(b, "_access", None)
        slots = ws._slots  # the only surface carrying the source name
        rows.append(
            (
                len(slots),
                tuple(s for s, _c, _h in slots),
                bool(ac is not None and ac._ignited),
                bool(ac is not None and ac._focus),
            )
        )

    # GUEST, NOT OWNER -- and the state must be populated. `Brain._update_access` wraps its body
    # in a bare except that leaves empty state, so a silent failure looks like a quiet brain;
    # these two asserts are what stop that reading as green.
    ac = b._access
    assert ac is not None and ac._own_ws is False, "the faculty must be a GUEST on the arm's bus"
    assert b._access_state, "empty _access_state means a swallowed exception, not a quiet brain"

    occ = Counter(r[0] for r in rows)
    cap = ws.capacity

    # THE DELETION-ON-NULL TRIGGER, standing guard (direction: occupancy must NOT be a constant
    # at capacity). Measured 2% on the day the rheobase landed against the 95% trigger.
    pinned = sum(1 for r in rows if r[0] == cap)
    assert pinned < 0.95 * N_TICKS, (
        f"live occupancy pinned at capacity {cap} on {pinned}/{N_TICKS} wake ticks: the capacity "
        f"bound has become a rank-order truncation and the inhibitory rheobase must be REVERTED "
        f"per its own deletion-on-null clause, not tuned"
    )

    # THE WORKSPACE IS GENUINELY MULTI-OCCUPIED LIVE -- the property the rheobase exists to buy.
    # Floor 30 sits between the pre-rheobase 3/120 and the measured 106/120.
    co = sum(1 for r in rows if r[0] >= 2)
    assert co >= 30, (
        f"co-occupancy on only {co}/{N_TICKS} wake ticks -- the pre-rheobase pathology was 3/120 "
        f"and the rheobase baseline is 106/120; a collapse here is the annihilation returning"
    )
    assert len(occ) >= 3, (
        f"occupancy must take several values on a live stream, got {dict(sorted(occ.items()))}"
    )

    # BOTH producers hold slots and BOTH take the head -- the bus is shared in fact, not by
    # construction alone. Floors far below the measured 93 slot-ticks / 23 head-ticks.
    census = Counter(s for r in rows for s in r[1])
    heads = Counter(r[1][0] for r in rows if r[1])
    assert census.get("perception", 0) > 0 and census.get("coalition", 0) >= 10, (
        f"both live producers must hold slots (census {dict(census)})"
    )
    assert len(heads) >= 2, f"the head must be contested between producers, got {dict(heads)}"

    # The dissociation occurs live: admitted while not the head, at least once.
    awf = sum(1 for r in rows if r[2] and not r[3])
    assert awf >= 1, "access-without-focus must OCCUR on the live brain (baseline 1/120)"
