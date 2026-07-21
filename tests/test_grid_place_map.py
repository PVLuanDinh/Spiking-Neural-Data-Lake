"""Tests for GridPlaceMap (hippocampus/grid_place_map.py): the grid population binds percepts to
locations, and both the firing criterion and the binding rule are DERIVED rather than configured.

Run: pytest tests/test_grid_place_map.py
"""

import statistics

from snn_data_lake.hippocampus.grid_place_map import GridPlaceMap

LABELS = ["digit3", "digit7", "digit1"]
ROUTE = [(-1.2 + 0.3 * i, -1.0 + 0.21 * ((i * 5) % 11)) for i in range(24)]


def _bound_map(k=6, seed=0):
    gpm = GridPlaceMap(seed=seed)
    pts = [(-1.2 + 2.4 * i / (k - 1), -1.2 + 2.4 * ((i * 7) % k) / (k - 1)) for i in range(k)]
    for i, p in enumerate(pts):
        gpm.observe(p, f"lm{i}")
    return gpm, pts


def test_identifies_every_bound_place():
    gpm, pts = _bound_map()
    assert all(gpm.place_at(p) == f"lm{i}" for i, p in enumerate(pts))


def test_stays_silent_where_no_place_was_bound():
    """Beating the alternatives is not the same as matching the template: at a far corner one
    landmark does stand out from the pack, and the map must still answer 'nowhere'."""
    gpm, _pts = _bound_map()
    far = (1.45, -1.45)
    acts = gpm.place_activation(far)
    best = max(acts.values())
    assert best > min(acts.values()), "this control needs a winner that leads the population"
    assert gpm.place_at(far) is None, f"a far corner must read as nowhere (winner {best:.3f})"


def test_firing_bar_is_derived_from_the_population_not_configured():
    """No place_threshold knob exists; the bar is the midpoint between a perfect self-match and the
    population's own off-field baseline, so it tracks the environment instead of being frozen."""
    gpm, _pts = _bound_map()
    assert not hasattr(gpm, "place_threshold"), "the configured threshold must be gone"
    acts = gpm.place_activation(_pts[0])
    best_p = max(acts, key=acts.get)
    bar = gpm._match_bar(acts, best_p)
    assert 0.3 < bar < 0.7, f"a sparse map's bar should land near 0.5 by derivation (got {bar:.3f})"


def test_one_label_at_many_places_does_not_destroy_the_place_code():
    """REGRESSION. The live caller binds whatever it perceives every tick, so a label recurs at many
    positions. Averaging those into one template drove the winner correlation to ~0.25 and made
    place_at answer None on nearly every tick. Distinct locations must open distinct place cells."""
    gpm = GridPlaceMap(seed=0)
    for i, p in enumerate(ROUTE):
        gpm.observe(p, LABELS[i % len(LABELS)])
    winners = [max(gpm.place_activation(p).values()) for p in ROUTE]
    answered = sum(gpm.place_at(p) is not None for p in ROUTE) / len(ROUTE)
    median = statistics.median(winners)
    assert median > 0.8, f"averaging across locations destroys the code (median {median:.3f})"
    assert answered > 0.9, f"the place code must be usable on the live path ({answered:.0%})"


def test_revisiting_consolidates_instead_of_growing_the_map():
    gpm = GridPlaceMap(seed=0)
    counts = []
    for _lap in range(3):
        for i, p in enumerate(ROUTE):
            gpm.observe(p, LABELS[i % len(LABELS)])
        counts.append(len(gpm._places))
    assert counts[0] == counts[-1], f"revisiting known places must not open new cells ({counts})"


def test_deterministic():
    a, pts = _bound_map(seed=0)
    b, _ = _bound_map(seed=0)
    assert [a.place_at(p) for p in pts] == [b.place_at(p) for p in pts]
    assert a.grid_code(pts[0]) == b.grid_code(pts[0])
