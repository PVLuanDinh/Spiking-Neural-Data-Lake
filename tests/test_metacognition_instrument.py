"""The reds for T077 -- specs/022 Phase 4 / C4, the metacognition instrument
(evidence/t077-metacognition.md 2.4).

WRITTEN AND COMMITTED **RED**, before `track_error` / `replay_reliability` exist in
`cortex/metacognitive_monitor.py`, per the pre-registration's commit order (2.9 step 1). The import
below is the red: collection fails until the instrument lands. Each test constructs the input on
which the instrument must FAIL (or refuse), with the control that must hold in the same test.

  R0 REPLAY      -- the offline replay of the monitor on a recorded input stream reproduces its
                    read exactly; a recorded read perturbed by 1e-9 on one tick is CAUGHT.
  R1 DEGENERACY  -- 2 errors in 400: the stream's own permutation null cannot resolve tracking
                    (q95 above 0.5 + rho) -> REFUSED, no auroc2 emitted.
  R2 RELAY       -- a second-order read that IS the first-order input: the relay contrast is 0.
  R3 MARKER      -- the shuffled pairing sits inside the permutation band; the oracle outside it.
  R4 NOT RELAY-PROOF -- the monitor's own synthetic world with hold-gated commits: accuracy DEPENDS
                    on the gate, so the first-order-preserved equivalence reads NOT MET there.
  R5 NO COLLAPSE -- a lesioned read equal to the intact read: (b3)'s contrast is exactly 0.

Every red runs on constructed streams; none touches a lake or a brain. The campaign's live reading
(a scratch brain, seeds 0-7) is NOT here -- the pin is written once by `--t077-campaign`.
"""

import random

import pytest

from snn_data_lake.cortex.metacognitive_monitor import (
    RHO,
    T_BAR,
    MetacognitiveMonitor,
    replay_reliability,
    track_error,
)


def _tracker_stream(n, p_err, strength, seed):
    """A constructed (conf, correct) stream whose confidence tracks the outcome with `strength`:
    conf = uniform noise shifted by `strength` on correct items (overlapping classes; strength 0.15
    reads auroc2 about 0.65, never 1.0 -- a tracker the permutation band must still resolve)."""
    rng = random.Random(seed)
    correct = [0 if rng.random() < p_err else 1 for _ in range(n)]
    conf = [min(1.0, max(0.0, rng.random() * (1.0 - strength) + strength * y)) for y in correct]
    return conf, correct


def test_r1_a_degenerate_stream_is_refused_by_its_own_permutation_null():
    """2 errors in 400: the null's q95 sits near 0.84, above 0.5 + rho -> REFUSED, no auroc2. The
    control: 6 errors in 400 (q95 near 0.69) is readable. A single outcome class is refused."""
    conf, correct = _tracker_stream(400, 0.0, 0.9, seed=1)
    only_one = track_error(conf, correct, seed=0)
    assert only_one["readable"] is False and only_one["auroc2"] is None, only_one
    correct[7] = 0
    correct[311] = 0
    r = track_error(conf, correct, seed=0)
    assert r["readable"] is False, r
    assert r["auroc2"] is None, "a refused stream must emit NO auroc2"
    assert r["perm_q95"] > 0.5 + RHO, r
    for i in (100, 200, 250, 333):
        correct[i] = 0
    ok = track_error(conf, correct, seed=0)
    assert ok["readable"] is True and ok["perm_q95"] <= 0.5 + RHO, ok


def test_r5_a_lesion_that_does_not_collapse_reads_a_zero_contrast():
    """(b3) is intact - lesioned auroc2; a constructed lesioned run whose read EQUALS the intact
    read gives exactly 0 -- the in-vivo half of the dissociation can fail, and fails exactly."""
    conf, correct = _tracker_stream(400, 0.1, 0.15, seed=4)
    a = track_error(conf, correct, seed=0)["auroc2"]
    b = track_error(list(conf), list(correct), seed=0)["auroc2"]
    assert a - b == 0.0


def test_r3_the_shuffled_pairing_is_a_marker_and_the_oracle_is_not():
    conf, correct = _tracker_stream(400, 0.1, 0.15, seed=3)
    r = track_error(conf, correct, n_perm=200, seed=0)
    assert r["readable"] is True
    assert r["oracle"] == 1.0, "conf := outcome must read auroc2 1.0 when both classes are present"
    assert r["oracle"] > r["perm_q95"], "the positive control must sit outside the permutation band"
    # the dissociation input: the SAME confidences paired with shuffled outcomes sit inside the band
    assert r["shuffled_auroc2"] <= r["perm_q95"], r
    assert 0.5 < r["auroc2"] < 1.0, r
    assert r["auroc2"] > r["perm_q95"], "a weak tracker must still clear its own permutation band"


def test_r2_a_relay_reads_a_relay_contrast_of_exactly_zero():
    conf, correct = _tracker_stream(400, 0.1, 0.6, seed=2)
    second_order = list(
        conf
    )  # the relay hypothesis: the second-order read IS the first-order input
    a = track_error(second_order, correct, seed=0)["auroc2"]
    b = track_error(conf, correct, seed=0)["auroc2"]
    assert a == b, "a relay must read (b3)'s contrast as exactly 0 -- NOT MET, not a near miss"


def test_r0_the_replay_reproduces_the_monitor_and_catches_a_perturbed_input():
    rng = random.Random(0)
    stream = [
        {
            "confidence": rng.uniform(0.2, 0.9),
            "surprise": rng.uniform(0.0, 0.5),
            "learning_progress": rng.uniform(0.0, 0.1),
            "novelty": rng.uniform(0.0, 0.6),
            "conflict": rng.uniform(0.0, 0.3),
        }
        for _ in range(120)
    ]
    mon = MetacognitiveMonitor(base=0.4, seed=5)
    recorded = [mon.update(**row) for row in stream]
    replayed = replay_reliability(stream, seed=5, base=0.4)
    assert replayed == recorded, "the replay must reproduce the live read byte-identically"
    # the red: a recorded read that differs by 1e-9 on ONE tick is CAUGHT. (An INPUT perturbation
    # of 1e-9 is invisible by construction -- the read is a spike COUNT over 24 x 16 steps,
    # quantized at 1/384 -- measured 2026-09-15 before the freeze, so the red is on the recorded
    # read, not the input.)
    bad = list(recorded)
    bad[60] += 1e-9
    with pytest.raises(AssertionError):
        replay_reliability(stream, seed=5, base=0.4, expect=bad)
    # and an input perturbation LARGE enough to move one spike IS caught
    big = [dict(row) for row in stream]
    big[60]["surprise"] = 1.0
    with pytest.raises(AssertionError):
        replay_reliability(big, seed=5, base=0.4, expect=recorded)


def test_r4_the_synthetic_world_is_not_relay_proof_by_construction():
    """On the monitor's own world, abstaining on the unreliable RAISES committed accuracy (the
    module's claim 2): first-order accuracy is NOT preserved under the gate there, so (b4)'s
    equivalence must read NOT MET on this input -- the red for the behavioural half."""
    from snn_data_lake.cortex.metacognitive_monitor import _experiment

    r = _experiment(seed=0)
    n = r["committed"]
    a = r["commit_acc"]
    tol = (a * (1.0 - a) / max(1, n)) ** 0.5
    assert abs(r["commit_acc"] - r["always_acc"]) > tol, r


def test_the_bar_is_the_plan_s_bar():
    from snn_data_lake.cortex.prior_drift_harness import T_BAR as harness_bar

    assert T_BAR == 1.895
    assert T_BAR == harness_bar, (
        "one bar: the monitor restates the harness's constant, never a second value"
    )
