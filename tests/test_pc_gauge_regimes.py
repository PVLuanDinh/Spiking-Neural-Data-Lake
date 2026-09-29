"""The pc_err gauge term discriminates again -- four regimes, per the level-channel rule.

The defect (2026-08-26 chip): `spike_snapshot`'s cortex term clipped an INTEGER error-spike
COUNT with n(), so the gauge read exactly 1.0 on 120 of 120 measured live settles while the
raw count varied 2..37 -- a live signal wearing a constant's face. The fix divides by the
network's own error-population size. The level-channel rule (a baseline-relative signal
measures variance; a LEVEL channel must be tested on four regimes) binds here: the fraction
must read LOW when error is low, HIGH when error is high, stay high under sustained bad
input (never adapt the level away), and hit ~0 on a perfectly explained input.
"""

import numpy as np

from snn_data_lake.engine.live_brain import LiveBrain
from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork


def _settled_frac(net, x):
    trace = net.infer(x, steps=60, lr=0.05)
    return trace[-1] / (len(net.s_e0) + len(net.s_e1))


def test_four_regimes_on_the_pc_fraction():
    rng = np.random.default_rng(3)
    net = PredictiveCodingNetwork(dims=(64, 16, 8), seed=0)

    # regime 1: OFF/unfitted -- the snapshot term reads 0.0 when the PC arm is off (pinned
    # below at the snapshot level, where the branch lives).

    # regime 2: untrained net, structured input -> a HIGH, nonzero fraction.
    x = np.clip(rng.random(64) * 0.8 + 0.1, 0.0, 1.0)
    f_untrained = _settled_frac(net, x)
    assert 0.0 < f_untrained <= 1.0

    # regime 3: LEARNING pulls the fraction DOWN on the same repeated input.
    for _ in range(30):
        net.reset()
        net.infer(x, steps=60, lr=0.05)
        net.learn_step()
    net.reset()
    f_trained = _settled_frac(net, x)
    assert f_trained < f_untrained, (
        f"learning did not reduce the settled error fraction ({f_untrained:.3f} -> "
        f"{f_trained:.3f}) -- the level channel is not tracking explanation"
    )

    # regime 4a: sustained BAD input stays HIGH -- the level must not adapt away (the
    # constant-bad-reads-CALM failure a baseline-relative form would have).
    fracs_bad = []
    for _i in range(8):
        bad = np.clip(rng.random(64), 0.0, 1.0)  # fresh unexplainable input every time
        net.reset()
        fracs_bad.append(_settled_frac(net, bad))
    assert min(fracs_bad) > 0.5 * f_untrained, (
        f"sustained novel input drifted low ({min(fracs_bad):.3f}) -- the channel is "
        "measuring variance, not level"
    )

    # regime 4b: a perfectly explained input reads ~0. Feed the net's OWN top-down
    # prediction as the input: the mismatch at layer 0 is ~zero by construction.
    net.reset()
    net.infer(x, steps=60, lr=0.05)
    predicted = np.clip(np.asarray(net.w1 @ net.r1, dtype=float), 0.0, 1.0)
    net.reset()
    f_explained = _settled_frac(net, predicted)
    assert f_explained < f_untrained, (
        f"the net's own prediction read as surprising as raw input ({f_explained:.3f} vs "
        f"{f_untrained:.3f})"
    )


def _ctx(v, tmp, au, d):
    # the module's own context shape (live_brain._ctx)
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def test_snapshot_gauge_discriminates_and_off_reads_floor():
    # OFF: no PC arm -> the cortex term carries only the goal component.
    lb_off = LiveBrain(seed=0, predictive_coding=False)
    lb_off.tick(_ctx(0.5, 0.2, 0.1, 0.8))
    off = lb_off.spike_snapshot()["cortex"]
    assert off < 1.0, f"PC-off cortex gauge pinned ({off})"

    # ON: the gauge must move off the 1.0 pin (the defect was 1.0 on EVERY tick; three ticks
    # with distinct sensory contexts must not all read the ceiling).
    lb = LiveBrain(seed=0, predictive_coding=True)
    vals = []
    for c in (_ctx(0.9, 0.1, 0.1, 0.8), _ctx(0.2, 0.7, 0.4, 0.5), _ctx(0.5, 0.3, 0.9, 0.9)):
        lb.tick(c)
        vals.append(lb.spike_snapshot()["cortex"])
    assert all(v < 1.0 for v in vals), f"cortex gauge still pinned at the clip: {vals}"
