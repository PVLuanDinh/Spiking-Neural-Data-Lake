"""specs/022 T039h: the insula is a SECOND global-workspace specialist, and SC-001 moves off 0.

Exactly one specialist had ever drained the re-entrant broadcast -- `action`, inside
engine/brain.py, a SELF-read of that module's own instance. GWT's claim is that ignited content is
available to MANY specialists; this tree had one, in the owning module, which is why
`p6_gate`'s faculty-to-faculty count sat at 0 through six commits of P6 work.

P6's own honesty guard is the bar this file has to clear: "a route is not a conversation -- neuter
faculty A and show faculty B's own output changes; do not force-wire." So the counter moving is
NOT what is tested here. What is tested is that the reader's own output changes, that it is
byte-identical when detached, that the loop it closes does not run away, and that the control the
listener term depends on is still intact.

Run: pytest tests/test_interoceptive_specialist.py
"""

import os
import statistics
import sys

import pytest

from snn_data_lake.insula.interoceptive_integration import (
    BODY_CHANNELS,
    SPECIALIST,
    InteroceptiveIntegration,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKS = 40


# ================================================================================================
# SC-001's counter, measured through the gate itself rather than asserted about.
# ================================================================================================
def test_the_faculty_to_faculty_count_is_no_longer_zero():
    """SC-001: P6 faculty-to-faculty reads > 0, from a measured baseline of 0.

    Measured by running the gate, not by reading the source: the read must survive the walk's own
    attribution rules -- BOTH endpoints role FACULTY, and the instance resolved to the module that
    actually constructs it."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    p6_gate = pytest.importorskip("p6_gate")
    res = p6_gate.measure(ROOT)
    fac = res["fac_cross"]
    assert fac, "faculty-to-faculty reads is still 0 -- SC-001 is not met"
    mine = [c for c in fac if c[0].endswith("interoceptive_integration.py")]
    assert mine, f"the insula read is not among the faculty-to-faculty reads: {fac}"
    assert all(c[4] == "engine/brain.py" for c in mine), (
        f"the insula read resolved to the wrong owner: {mine}"
    )


def test_the_subscription_and_the_drain_use_ONE_name():
    """A subscription nobody drains and a drain nobody registered are the two halves of the same
    dormancy. One constant, read by both sides."""
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    try:
        assert SPECIALIST in b._reentrant._subs, (
            f"{SPECIALIST!r} is drained but never subscribed -- broadcast_to returns {{}} forever"
        )
    finally:
        b.close()


# ================================================================================================
# The mechanism, in isolation. No brain: the top-down term is what is under test.
# ================================================================================================
class _FakeWorkspace:
    """A stand-in bus. NOT the method under test -- it exists so the salience response to a
    top-down bias can be measured without the ignition loop feeding back into it."""

    def __init__(self, bias):
        self._bias = dict(bias)
        self.used = []

    def broadcast_to(self, name):
        return dict(self._bias)

    def report_use(self, name, content):
        self.used.append((name, content))
        return 0.0


class _FakeBrain:
    def __init__(self, bias):
        self.reentrant = _FakeWorkspace(bias)


def _settled(ii, n=300, **frame):
    out = 0.0
    for _ in range(n):
        out = ii.observe(**frame)
    return out


FRAME = dict(
    energy_deficit=0.1, curiosity_deficit=0.4, social_deficit=0.9, stress=0.15, caution=0.3
)
SPIKE = dict(FRAME, stress=0.9)


def test_detached_is_byte_identical_to_the_unattended_salience():
    """ADDITIVE FROM ZERO. An insula with no brain, or a brain whose workspace has ignited nothing,
    must compute exactly what it computed before this term existed -- otherwise the term's
    attribution is confounded by a change it makes everywhere."""
    a = InteroceptiveIntegration(channels=BODY_CHANNELS)
    b = InteroceptiveIntegration(channels=BODY_CHANNELS)
    b.attach(_FakeBrain({}))  # subscribed, nothing ignited -> broadcast_to returns {}
    for _ in range(50):
        assert a.observe(**FRAME) == b.observe(**FRAME)
    assert a.observe(**SPIKE) == b.observe(**SPIKE)


def test_the_top_down_term_raises_salience_and_reports_use():
    """The reader's OWN output moves, and it tells the bus it consumed -- the post-condition
    T039c's listener term has no writer without."""
    plain = InteroceptiveIntegration(channels=BODY_CHANNELS)
    attn = InteroceptiveIntegration(channels=BODY_CHANNELS)
    fb = _FakeBrain({"X": 0.2})
    attn.attach(fb)
    _settled(plain, **FRAME)
    _settled(attn, **FRAME)
    p, a = plain.observe(**SPIKE), attn.observe(**SPIKE)
    assert a > p, f"attention did not raise salience ({a:.6f} vs {p:.6f})"
    assert fb.reentrant.used, "the specialist consumed the bias and never reported use"
    assert fb.reentrant.used[0][0] == SPECIALIST


def test_the_bound_holds_under_an_absurd_bias():
    """THE REACHABLE RED FOR THE PLACEMENT. The term is applied to z BEFORE the squash, so the
    output stays in [0, 1) for any non-negative bias. Applied AFTER the squash it would leave the
    range and need a clamp -- and a clamp would hide an unbounded upstream bias rather than expose
    it. A huge bias must therefore saturate, never escape."""
    ii = InteroceptiveIntegration(channels=BODY_CHANNELS)
    ii.attach(_FakeBrain({"X": 1e6}))
    _settled(ii, **FRAME)
    for _ in range(20):
        s = ii.observe(**SPIKE)
        assert 0.0 <= s < 1.0, s


def test_a_negative_or_empty_bias_cannot_lower_salience():
    """A top-down term is attention, not suppression, in this build. Clamped at 0 so a malformed
    bus cannot silently invert the mechanism's direction."""
    ii = InteroceptiveIntegration(channels=BODY_CHANNELS)
    ii.attach(_FakeBrain({"X": -5.0}))
    plain = InteroceptiveIntegration(channels=BODY_CHANNELS)
    _settled(ii, **FRAME)
    _settled(plain, **FRAME)
    assert ii.observe(**SPIKE) == plain.observe(**SPIKE)


# ================================================================================================
# The live loop. This is the half that decides whether the wire is a conversation.
# ================================================================================================
def _run(detach):
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    if detach and b._insula is not None:
        b._insula._brain = None  # NEUTER: the specialist stops reading the bus
    rows = []
    for _ in range(TICKS):
        b.tick()
        rows.append(
            {
                "sal": float(getattr(b, "_feeling_salience", 0.0) or 0.0),
                "thr": float(b._reentrant.ignition_threshold),
                "win": b._reentrant._last.get("winner"),
                # last_top_down(), not _top_down(): the harness is a REPORTER, and the consuming
                # read's post-condition writes report_use -- a harness calling it contaminated
                # the attached arm's _use every tick (US1 policy audit, B4).
                "td": (0.0 if b._insula is None else b._insula.last_top_down()),
            }
        )
    b.close()
    return rows


@pytest.fixture(scope="module")
def attached():
    return _run(detach=False)


@pytest.fixture(scope="module")
def detached():
    return _run(detach=True)


def test_the_broadcast_actually_reaches_the_specialist_live(attached):
    """LIVENESS. If the live bus never delivers anything to this subscriber, every comparison
    below is between two identical arms and would pass vacuously."""
    td = [r["td"] for r in attached]
    assert max(td) > 0.0, (
        "the workspace never broadcast anything to this specialist over "
        f"{TICKS} ticks -- the subscription is registered and dead"
    )


def test_neutering_the_read_changes_the_READERS_own_output(attached, detached):
    """P6's charge bar, not its reachability bar: neuter the read and the reading faculty's OWN
    output must move. If the salience traces are identical, the route carries nothing and P8
    requires deleting it rather than reporting it."""
    a = [r["sal"] for r in attached]
    d = [r["sal"] for r in detached]
    assert len(a) == len(d) == TICKS
    differ = sum(1 for x, y in zip(a, d, strict=True) if abs(x - y) > 1e-12)
    assert differ > 0, (
        "the insula computes an identical salience with and without the broadcast -- "
        "reachability without charge, which P3 says is not charge"
    )


def test_the_loop_it_closes_does_not_run_away(attached):
    """THE RISK THIS BUILD TAKES ON, measured rather than argued. Salience lowers the ignition
    threshold (T039f) and ignition feeds the broadcast that raises salience. Bounded is not the
    same as stable: a term that pins salience at its ceiling would be a runaway that every
    'is it bounded' assert would pass."""
    sal = [r["sal"] for r in attached]
    assert max(sal) < 0.95, f"salience saturated (max {max(sal):.4f}) -- the loop is running away"
    tail = sal[-10:]
    assert len(set(round(x, 6) for x in tail)) > 1, (
        f"salience froze at {tail[0]:.6f} over the last 10 ticks -- a pinned signal, not a live one"
    )
    assert statistics.fmean(sal[-10:]) < 0.9


def test_the_language_control_is_still_intact():
    """The listener term T039c added is only measurable against a subscriber that keeps the base
    gain. `language` is that control: subscribed, never drained, never reporting. A third
    specialist that DOES report must not have quietly recruited it."""
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    try:
        for _ in range(20):
            b.tick()
        used = getattr(b._reentrant, "_use", {})
        assert not [k for k in used if k[0] == "language"], (
            f"language reported use and is no longer a never-reporting control: {used}"
        )
        assert [k for k in used if k[0] == SPECIALIST], (
            "the insula never reported use over 20 ticks -- the post-condition is dead"
        )
    finally:
        b.close()


# ================================================================================================
# US1 policy audit, finding B4: the instrument must not charge the listener term it reports.
# ================================================================================================
def test_last_top_down_reads_without_touching_the_bus():
    """The reporter accessor is side-effect-free, and it reports something real (liveness: a dead
    accessor returning a constant 0.0 would pass a pure no-mutation assert vacuously)."""
    ii = InteroceptiveIntegration(channels=BODY_CHANNELS)
    fb = _FakeBrain({"X": 0.2})
    ii.attach(fb)
    _settled(ii, **FRAME)
    n_used = len(fb.reentrant.used)
    td = ii.last_top_down()
    assert td > 0.0, "the applied term reads 0.0 after 300 attended ticks -- dead accessor"
    assert len(fb.reentrant.used) == n_used, (
        "last_top_down() reported use -- the reporter accessor must never write the bus"
    )


def test_the_telemetry_poll_does_not_charge_the_listener_term():
    """B4's liveness control, with the pre-fix code as the reachable red: snapshot() used to call
    _top_down(), whose post-condition writes report_use, so the cockpit poll was the largest
    single writer of the listener-use term (measured 37-40 of ~60 calls over 40 ticks) and the
    only 'consumer' on every sleep tick. Polled and unpolled runs must now be byte-equal in both
    the listener-use dict and the salience trace, warm-up window included."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.infra.brain_telemetry import snapshot

    def run(poll):
        b = make_brain(seed=0, persist=False)
        try:
            sal = []
            for _ in range(TICKS):
                b.tick()
                if poll:
                    snapshot(b)
                sal.append(float(getattr(b, "_feeling_salience", 0.0) or 0.0))
            return dict(getattr(b._reentrant, "_use", {})), sal
        finally:
            b.close()

    use_polled, sal_polled = run(poll=True)
    use_quiet, sal_quiet = run(poll=False)
    assert use_polled == use_quiet, (
        "the telemetry poll wrote the listener-use term -- the instrument is charging the "
        f"consumption record it reports (B4): polled={use_polled} quiet={use_quiet}"
    )
    assert sal_polled == sal_quiet, (
        "the telemetry poll moved feeling_salience -- the observer changed the measured"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
