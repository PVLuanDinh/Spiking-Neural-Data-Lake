"""Tests for TimeCellClock (cortex/time_cells.py): a ramping-to-threshold spiking time-cell
population that tiles an interval. Covers the three faculty claims -- (a) decode() tracks true
elapsed steps monotonically, (b) the decoded SD scales up with longer intervals (Weber/scalar
timing emerges from accumulator noise), (c) learn_interval() reduces interval-reproduction error
via reward-modulated STDP -- plus per-instance determinism.

Run: pytest tests/test_time_cells.py
"""

from snn_data_lake.cortex.time_cells import TimeCellClock


def test_decode_tracks_elapsed_time_monotonically():
    clock = TimeCellClock(n_cells=64, seed=0)
    estimates = [clock.tick()[0] for _ in range(50)]
    assert all(estimates[i + 1] >= estimates[i] for i in range(len(estimates) - 1))
    assert abs(estimates[-1] - 50) < 5.0


def test_decoded_sd_grows_with_duration_weber_law():
    clock = TimeCellClock(n_cells=64, seed=0)
    sd_short = sd_long = 0.0
    for step in range(1, 51):
        _est, sd = clock.tick()
        if step == 10:
            sd_short = sd
        if step == 50:
            sd_long = sd
    assert sd_long > sd_short


def test_learn_interval_reduces_reproduction_error():
    learner = TimeCellClock(n_cells=64, seed=1)
    target = 25.5  # off-grid, between two integer thresholds
    i = min(
        range(learner.n_cells),
        key=lambda idx: abs(learner.thresholds[idx] / learner.drive - target),
    )
    error_pre = abs(learner.thresholds[i] / learner.drive - target)
    for _ in range(30):
        i_cell, _delta = learner.learn_interval(target, reward=1.0)
    error_post = abs(learner.thresholds[i_cell] / learner.drive - target)
    assert error_post < error_pre


def test_same_seed_is_deterministic():
    a = [TimeCellClock(n_cells=32, seed=7).tick() for _ in range(20)]
    b = [TimeCellClock(n_cells=32, seed=7).tick() for _ in range(20)]
    assert a == b
