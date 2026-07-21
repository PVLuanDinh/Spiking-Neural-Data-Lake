"""
Tests for reasoning/real_rpm_bench.py -- the HONEST held-out RPM benchmark that replaces the old
grammar self-test (which scored ~98% by solving puzzles its own generator produced).

What matters here is not a high score but a TRUSTWORTHY one: the matrices are hand-written and
held out from the solver's generator, each panel is RENDERED as a raster and re-PERCEIVED through
the spiking connected-component front-end (never handed in pre-parsed), and two broken controls
must sit near chance. A pipeline that leaks the answer key beats the controls; this suite fails if
it does.

Run: pytest tests/test_real_rpm_bench.py
"""

from snn_data_lake.reasoning.real_rpm_bench import (
    CHANCE,
    expected_cell,
    perceive_panel,
    render_panel,
    run_benchmark,
    scrambled_panel_control,
    shuffled_answer_control,
)


def test_accuracy_clears_chance_by_a_wide_margin():
    """Measured accuracy must sit far above the 1/8 chance floor (80% at time of writing)."""
    r = run_benchmark(seed=0)
    assert r["n_items"] >= 8, "the held-out set must not shrink below 8 matrices"
    assert r["chance"] == CHANCE
    assert r["accuracy"] >= 0.5, f"accuracy collapsed to {r['accuracy']:.3f} (chance {CHANCE:.3f})"
    assert r["accuracy"] > 3 * CHANCE, (
        "accuracy must clear chance by a wide margin, not squeak past"
    )


def test_strict_accuracy_never_exceeds_accuracy():
    """strict = correct AND the solver committed (known=True); it can only be <= accuracy."""
    r = run_benchmark(seed=0)
    assert r["strict_accuracy"] <= r["accuracy"] + 1e-9


def test_failures_are_kept_and_honest():
    """Real failures are retained rather than tuned away, and a failed item must NOT claim known."""
    r = run_benchmark(seed=0)
    wrong = [rec for rec in r["records"] if not rec["correct"]]
    assert wrong, "a benchmark nothing fails cannot detect failure -- keep the hard items"
    for rec in wrong:
        assert not rec["known"], (
            f"{rec['name']} answered WRONG while claiming known=True -- a confident wrong answer "
            "is worse than an honest gap"
        )


def test_broken_controls_sit_near_chance():
    """Shuffled-answer (broken key) and scrambled-panel (broken matrix) must both be unbeatable."""
    shuf = shuffled_answer_control(repeats=4, seed=101)
    scram = scrambled_panel_control(repeats=4, seed=202)
    assert shuf <= 0.30, f"shuffled-answer control beat chance ({shuf:.3f}) -- the key is leaking"
    assert scram <= 0.30, (
        f"scrambled-panel control beat chance ({scram:.3f}) -- structure is leaking"
    )


def test_perception_is_real_not_handed_in():
    """Panels are rendered to a raster and the attributes re-DERIVED from pixels by the spiking
    connected-component perceiver -- never handed in pre-parsed. This is what makes the benchmark
    a perception+reasoning test rather than a reasoning-only one."""
    from snn_data_lake.reasoning import real_rpm_bench as rb

    shape = next(iter(rb._RENDER))  # _RENDER keys are mixed-type; take the first in insertion order
    for spec in (((shape, 1, 0),), ((shape, 2, 0), (shape, 2, 4))):
        grid = render_panel(spec)
        assert isinstance(grid, tuple) and len(grid) > 1, "render_panel must produce a 2-D raster"
        assert perceive_panel(grid) == expected_cell(spec), (
            "the perceiver must recover the panel's attributes from the raster alone"
        )


def test_deterministic():
    a = run_benchmark(seed=0)
    b = run_benchmark(seed=0)
    assert a["accuracy"] == b["accuracy"]
    assert [r["answer_idx"] for r in a["records"]] == [r["answer_idx"] for r in b["records"]]
