"""Pin the AUTONOMOUS SELF-BUILD ROUTE -- P0's headline avenue.

`docs/POLICIES.md` files self-building code under "What fires". Measured 2026-08-13: the cadence
DOES fire and it proposes ZERO candidates, so `SelfModHarness` is never constructed. That gap is
recorded in the policy text; these tests exist so the ROUTE cannot break silently underneath it.

WHAT THESE TESTS DELIBERATELY DO NOT ASSERT: that `proposed == 0`. Filling the proposer IS the open
P0 work, so an assert on the empty payload would go RED on the change it exists to encourage -- the
"criteria measure the pathology" failure this project has hit before. The payload is free to grow;
what is pinned is that the route still fires and that the opt-in still gates it.

Cost, stated rather than hidden: `test_selfbuild_route_fires` runs 190 ticks because the cadence is
`nights % SELFBUILD_EVERY == 0` and night 8 is the first hit -- reaching it any faster would mean
faking the constant, which is the thing being pinned. Measured on `.venv-lake`: **25.8 s / 26.3 s**
under the suite's own config (coverage on), 20.7 s with `--no-cov`. An audit measured ~47 s and that
did not reproduce here; it was taken while fifteen agents ran concurrently, so treat the figure as
load-dependent rather than either number as wrong. It spawns NO subprocess: verified that
`SelfBuildLoop.run([])` returns `[]` without invoking `gate_fn` (the gate shells out to
`run_selfchecks.py`, which would be multi-minute).

WHAT THIS FILE DOES NOT PIN, disclosed because the gap is easy to mistake for coverage: the PROPOSER
leg is near-vacuous here. A proposer that raises, or one deleted outright, still leaves these tests
green -- `Brain._maybe_autonomous_self_build`'s bare `except Exception: return None` converts the
first into a None result, which the load-bearing assert below DOES catch, but a proposer that merely
returns nothing is indistinguishable from a healthy one that found nothing. That is
deliberate: the empty payload is the open P0 work, so this file pins the ROUTE only. Anything
asserting the proposer's health belongs in a test of the proposer.

REACHABLE REDS, all three demonstrated by patching `src/` and reverting with `git checkout`:
  * cadence      `SELFBUILD_EVERY` 8 -> 9999  ->  fires the tick-count guard
  * opt-in gate  the `NORD_SELFMOD` guard made unreachable  ->  fires the control test
  * recorder     `propose_improvements` made to raise, so the bare `except` at
                 `Brain._maybe_autonomous_self_build` swallows it  ->  fires the LOAD-BEARING assert
                 with `assert None is not None`
The third was added after an audit found the first two never exercised the assert this file exists
for: they both fire one line earlier, on the tick-count guard. Two reds on a criterion is not the
same as a red on THE criterion.

IT DOES WRITE A LAKE, and an earlier revision of this docstring said it did not. Measured: 8 files
/ 19,291,290 B -- `brain_lake.bin` at 48,607 B plus the skill `.npz` set -- written into the
per-test `NORD_HOME` that `tests/conftest.py`'s autouse fixture isolates. `make_brain(seed=0)`
passes no `persist=`, so persistence resolves from the ambient environment, and conftest sets
`NORD_HOME` only (its own docstring: "deliberately NOT NORD_PERSIST=0"). The false claim was
produced by checking `./brain_lake.bin` in the repo root with `NORD_HOME` unset -- the wrong
location, so a miss was read as an absence. Recorded because it shipped inside the paragraph headed
"stated rather than hidden", which is the one place a false measurement must not sit.
"""

from snn_data_lake.cortex.brain import SELFBUILD_EVERY, make_brain


def test_selfbuild_route_fires(monkeypatch):
    """The night-8 cadence reaches the recorder. Pins the route, not the payload."""
    # monkeypatch.setenv restores on teardown -- this module's siblings have been bitten by a
    # leaked NORD_SELFMOD leaving every later in-process test running under a different gate.
    monkeypatch.setenv("NORD_SELFMOD", "1")

    b = make_brain(seed=0)
    for _ in range(190):
        b.tick()

    assert b._nights >= SELFBUILD_EVERY, (
        f"190 ticks reached only night {b._nights}; the cadence needs night {SELFBUILD_EVERY} to "
        f"fire, so this test can no longer reach the branch it exists to pin"
    )
    assert b._selfbuild_last is not None, (
        "the self-build cadence did not reach the recorder: _selfbuild_last is still None after "
        f"{b._nights} nights with NORD_SELFMOD=1. POLICIES.md files this route under 'What fires'"
    )
    # `_maybe_autonomous_self_build` OVERWRITES this dict on every fire, so it holds the LAST hit,
    # not the first. At 190 ticks night 8 is the only crossing, so the two coincide -- if the tick
    # count ever grows past night 16 this assert changes meaning and must be revisited.
    assert b._selfbuild_last["night"] == SELFBUILD_EVERY, (
        f"recorded night {b._selfbuild_last['night']!r}, expected the only cadence hit reachable in "
        f"190 ticks, at {SELFBUILD_EVERY}"
    )
    # Keys, not values: the counts are free to move when the proposer is filled.
    assert "proposed" in b._selfbuild_last and "kept" in b._selfbuild_last


def test_selfbuild_is_gated_off_by_default(monkeypatch):
    """The control: without the opt-in the route is inert.

    Without this, `test_selfbuild_route_fires` could pass on a brain that self-builds
    unconditionally -- i.e. the assert would be satisfied by the safety property being GONE.
    """
    monkeypatch.delenv("NORD_SELFMOD", raising=False)

    b = make_brain(seed=0)
    assert b._maybe_autonomous_self_build() is None, (
        "NORD_SELFMOD is unset, so the self-build must be a no-op; a non-None return means the "
        "operator opt-in has stopped gating self-modification"
    )
    assert b._selfbuild_last is None
