"""Pin the AUTONOMOUS SELF-BUILD ROUTE -- P0's headline avenue.

`docs/POLICIES.md` files self-building code under "What fires". Measured 2026-08-13: the cadence
DOES fire and it proposes ZERO candidates, so `SelfModHarness` is never constructed. That gap is
recorded in the policy text; these tests exist so the ROUTE cannot break silently underneath it.

WHAT THESE TESTS ASSERT ABOUT THE PAYLOAD, and why it changed. The route assertions deliberately
check KEYS, not counts -- filling the proposer IS the open P0 work, and a criterion that reddens on
the change it exists to encourage is the "criteria measure the pathology" failure this project has
hit before. But `proposed == 0` IS asserted, for a different reason: as a SAFETY tripwire. This test
runs the real `SelfBuildLoop` against the LIVE repository, and `promote()` writes before it gates,
so "nothing was kept" does not mean "nothing was written". The tripwire message says what
to do when it fires -- inject a scratch repo_root -- so it directs the fix, not blocks the work.
Distinguishing those two roles is the whole point: one assert would be pathology-measuring, the
other is the only bound that actually holds.

Cost, stated rather than hidden: `test_selfbuild_route_fires` runs 190 ticks because the cadence is
`nights % SELFBUILD_EVERY == 0` and night 8 is the first hit -- reaching it any faster would mean
faking the constant, which is the thing being pinned. Measured on `.venv-lake`: **25.8 s / 26.3 s**
under the suite's own config (coverage on), 20.7 s with `--no-cov`. An audit measured ~47 s and that
did not reproduce here; it was taken while fifteen agents ran concurrently, so treat the figure as
load-dependent rather than either number as wrong. It spawns NO subprocess: verified that
`SelfBuildLoop.run([])` returns `[]` without invoking `gate_fn` (the gate shells out to
`run_selfchecks.py`, which would be multi-minute).

WHAT THIS FILE DOES NOT PIN, disclosed because the gap is easy to mistake for coverage: the PROPOSER
leg is near-vacuous here, and MORE so than an earlier revision of this docstring admitted. A
proposer that RAISES still leaves these tests green: `Brain.propose_improvements` has its OWN
`except Exception: return []`, so the raise never reaches `_maybe_autonomous_self_build`'s handler
and `_selfbuild_last` is written normally. Measured -- patching the proposer to raise yields
`{'night': 8, 'proposed': 0, 'kept': 0}`, not None. The load-bearing assert catches only a raise
from `propose_improvements` ITSELF, which is what red 3 patches. A proposer that merely
returns nothing is likewise indistinguishable from a healthy one that found nothing. That is
deliberate: the empty payload is the open P0 work, so this file pins the ROUTE only. Anything
asserting the proposer's health belongs in a test of the proposer.

REACHABLE REDS, all three demonstrated by patching `src/` and reverting with `git checkout`:
  * cadence      `SELFBUILD_EVERY` 8 -> 9999  ->  fires the tick-count guard
  * opt-in gate  the `NORD_SELFMOD` guard made unreachable  ->  fires the control test
  * recorder     `propose_improvements` made to raise, so the bare `except` at
                 `Brain._maybe_autonomous_self_build` swallows it  ->  fires the LOAD-BEARING assert
                 with `assert None is not None`
The third was added after an audit found the first two never exercised the assert this file exists
for. Precisely: red 1 fires the tick-count guard one line ABOVE it, and red 2 fires the CONTROL
test, which has no tick-count guard at all and cannot affect the positive test (that test sets
`NORD_SELFMOD=1` itself). An earlier revision of this docstring said "they both fire one line
earlier", which is true of red 1 only. Two reds on a criterion is not the same as a red on THE
criterion -- and describing where a red lands is itself a claim that has to be checked.

IT DOES WRITE A LAKE, and an earlier revision of this docstring said it did not. Measured: 8 files
/ 19,291,290 B -- `brain_lake.bin` at 48,607 B plus the skill `.npz` set -- written into the
per-test `NORD_HOME` that `tests/conftest.py`'s autouse fixture isolates. `make_brain(seed=0)`
passes no `persist=`, so persistence resolves from the ambient environment, and conftest sets
`NORD_HOME` only (its own docstring: "deliberately NOT NORD_PERSIST=0"). The false claim was
produced by checking `./brain_lake.bin` in the repo root with `NORD_HOME` unset -- the wrong
location, so a miss was read as an absence. Recorded because it shipped inside the paragraph headed
"stated rather than hidden", which is the one place a false measurement must not sit.
"""

import hashlib
import os

import snn_data_lake.infra.self_build_loop as _sbl
from snn_data_lake.engine.brain import SELFBUILD_EVERY, make_brain

# Every file the one authored transform can reach. Digested before and after the run: this is what
# replaced `proposed == 0` as the safety bound (see the tripwire block below).
_REACHABLE = ("src/snn_data_lake/faculties/metacognitive_monitor.py",)


def _digest(root):
    """Content digest of every file a promote could legally rewrite in `root`."""
    h = hashlib.sha256()
    for rel in _REACHABLE:
        path = os.path.join(root, *rel.split("/"))
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def test_selfbuild_route_fires(monkeypatch, tmp_path):
    """The night-8 cadence reaches the recorder. Pins the route, not the payload."""
    # monkeypatch.setenv restores on teardown -- this module's siblings have been bitten by a
    # leaked NORD_SELFMOD leaving every later in-process test running under a different gate.
    monkeypatch.setenv("NORD_SELFMOD", "1")

    # SCRATCH repo_root -- this is the injection the retired `proposed == 0` tripwire asked for by
    # name ("give this test its own scratch repo_root before filling the proposer"). `brain.py`
    # imports SelfBuildLoop LAZILY inside _maybe_autonomous_self_build, so it resolves through this
    # module attribute at call time and patching here reaches it. The root is empty, so `promote()`
    # refuses every target as missing BEFORE it writes -- the live tree is unreachable by
    # construction rather than by the payload happening to be empty.
    from snn_data_lake.infra.self_mod_harness import _repo_root

    live_root = _repo_root()
    real_loop = _sbl.SelfBuildLoop
    seen = {}

    def _scratch_rooted(*a, **kw):
        kw["repo_root"] = str(tmp_path)
        seen["root"] = str(tmp_path)
        return real_loop(*a, **kw)

    monkeypatch.setattr(_sbl, "SelfBuildLoop", _scratch_rooted)

    # POSITIVE CONTROL for the digest itself, run BEFORE the brain does anything. A digest that
    # cannot tell two different byte-strings apart would make the "live tree untouched" assert below
    # vacuous, and vacuously-untouched is exactly what the retired assert was criticised for.
    before = _digest(live_root)
    assert before != hashlib.sha256(b"different").hexdigest(), "the digest is not content-sensitive"

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
        f"recorded night {b._selfbuild_last['night']!r}, expected the only cadence hit "
        f"reachable in 190 ticks, at {SELFBUILD_EVERY}"
    )
    # Keys, not values: the counts are free to move when the proposer is filled.
    assert "proposed" in b._selfbuild_last and "kept" in b._selfbuild_last

    # SAFETY, and it deliberately does NOT depend on the payload being empty.
    assert b._selfbuild_last.get("kept", 0) == 0, (
        f"a unit test APPLIED {b._selfbuild_last.get('kept')} self-modification(s): the scratch "
        f"repo_root injection above is not reaching SelfBuildLoop"
    )

    # THE TRIPWIRE, DISCHARGED AND REPLACED BY A STRICTLY STRONGER ONE -- 2026-08-17, specs/019
    # [B2.03]. What stood here was `proposed == 0`, with its own comment saying it SHOULD go red the
    # day the proposer was filled and that the correct response was to inject a scratch repo_root
    # rather than relax the bound. The proposer is now filled (one authored transform), so that day
    # arrived and the injection is above.
    #
    # The retired assert proved "nothing was written" only by proving NOTHING RAN, which is the
    # weakest form of the claim -- and it stops being available the moment the route does real work.
    # This replaces it with the claim it was standing in for: the live tree is BYTE-IDENTICAL after
    # a run in which the proposer really did propose. That is strictly stronger, because it survives
    # `promote()`'s apply-before-gate window: promote writes to `self.root` and rolls back on a
    # failure, so a proposal can be applied and reverted without ever reaching status "kept". With
    # `self.root` pointed at an empty scratch dir, the live tree is not merely unwritten -- it is
    # unreachable, and this digest is what proves it rather than assuming it.
    assert _digest(live_root) == before, (
        "the LIVE repository changed during a unit test: every file the authored transform can "
        "reach must be byte-identical after the run. promote() writes before it gates, so "
        "'kept == 0' does not mean nothing was written -- that is why this is a digest and not a "
        "count. Do not weaken this assert; fix the repo_root injection instead"
    )
    assert seen.get("root") == str(tmp_path), (
        "SelfBuildLoop was never constructed through the patched name, so the scratch root never "
        "took effect and the digest above proved nothing about a route that ran unrooted"
    )

    # The recorder is written by TWO branches and only the circuit-breaker one adds
    # this key. Without it the test passes on a run where SelfBuildLoop.run was never
    # called at all -- measurable today by forcing should_skip True.
    assert "skipped" not in b._selfbuild_last, (
        f"the cadence short-circuited instead of running the loop: {b._selfbuild_last!r}"
    )


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
