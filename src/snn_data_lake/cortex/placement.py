"""
Placement -- every grown neuron gets a PLACE, a LAYER and an immutable SIGN, from the H01 human prior.

WHAT THIS DOES. `cortex/neurogenesis_engine.py` grows units on the sleep tick and folds them to the
one durable lake. Until now a grown unit was a row of weights and nothing else: no position, no
layer, no sign. This gives each one a coordinate in the H01 cortical geometry, the layer that
coordinate falls in, and a Dale sign drawn from that layer's measured excitatory fraction.

WHAT THE PRIOR FIXES, AND WHAT IT EXPLICITLY DOES NOT. H01 fixes STRUCTURE -- the proportion of new
units per layer, the sign each one carries, the laminar order they occupy. **It does not cap COUNT,
and nothing here may be read as one.** Growth is unbounded: units accumulate into the one durable
lake and the population is meant to keep rising. P0 is explicit that a limit found in the world is
"a gap to close, not a boundary to respect", and P8 removes "that is not how it is done" as an
objection outright. This module supplies ORDER for unlimited growth; it is not a budget.

An earlier draft of this docstring got that backwards and called the biology a "ceiling", so the
correction is recorded rather than quietly edited. The relevant fact is real and is NOT a limit on
this brain: adult neurogenesis in humans is HIPPOCAMPAL (the dentate gyrus), so human cortex does
not add neurons in adulthood. That is a fact about humans. This brain is not bound by it, and the
engine's own artificial cap -- `l_max` fixed at 200 -- was already retired for exactly this reason.
What H01 contributes is the ANSWER TO "where does unit number 1,000,001 go, and what sign does it
carry", which is a question about architecture and stays answerable at any scale because the prior
is a distribution over layers, not a quota.

`_selfcheck` PROVES the no-cap claim rather than asserting it: the placed layer proportions hold at
100,000 units, so the prior scales without a ceiling appearing.

THE SIGN RULE IS THE MEASUREMENT, NOT A CONVENTION. `cortical_prior.LAYER_E_FRACTION` is derived
from H01: L1 0.0000, L2 0.6538, L3 0.6412, L4 0.7879, L5 0.7596, L6 0.8743. So an excitatory cell in
LAYER 1 is UNREPRESENTABLE here -- not by a special case, but because the measured fraction is
exactly zero. Layer 1 in real cortex is neuropil and interneurons with essentially no excitatory
somata, which is why two independent derivations of that fraction disagreed there by 0.1227 while
agreeing to 0.0049 everywhere else. `assign_sign` ALSO refuses it structurally, because a rule that
holds only while a number stays 0.0 is a rule waiting to be broken by an edit.

DALE'S PRINCIPLE: a cell's sign is fixed at birth and never changes. `place` returns it once; there
is no setter. A unit that could flip sign later is not obeying Dale, and the immutability is the
whole point of assigning at creation rather than at use.

DETERMINISM: one `random.Random(seed)` per call, never module-level seeding, so the same seed places
the same units in the same layers with the same signs -- on any machine. That matters because the
placement is PERSISTED with the substrate: a restart that re-derived different layers would be a
different brain wearing the same weights.

  python -m snn_data_lake.cortex.placement
"""

import math
import random
import sys

from snn_data_lake.cortex.cortical_prior import (
    LAYER_CIRCLES,
    LAYER_E_FRACTION,
    LAYER_OCCUPANCY,
)

EXCITATORY = 1
INHIBITORY = -1

_LAYER_NAMES = ("Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6")
WHITE_MATTER = "White matter"

#: Agreement between `layer_of` and H01's OWN published `cortical layer` labels, measured over all
#: 49,377 soma records: 45,404 / 49,377. Every disagreement is an ADJACENT layer (top confusions
#: L4->L3 971, L5->L6 537, L3->L2 501) -- boundary cells the circle fit places one band over. This
#: is the accuracy bound for anything derived from placement, and it is measured, not asserted.
LAYER_RULE_AGREEMENT = 0.9195


# distinct-from: cortex/is_snn_forward.py -- its "layer" is a LIF network layer with its own
#   membrane tau; this one is a CORTICAL LAMINAR BAND in micron geometry. Same English word,
#   unrelated jobs; site read, nothing to reuse in either direction.
def layer_of(x, y):
    """The cortical band containing (x, y) in MICRONS, by H01's own `fix_layer_mem` rule.

    THE ORDER IS NOT WHAT IT LOOKS LIKE, and misreading it put 33% of cells in Layer 1 (real ~10%):

      * `LAYER_CIRCLES[0]` is WHITE MATTER -- not Layer 1.
      * circles 1..5 are then tested in order and yield Layer **6, 5, 4, 3, 2** -- DESCENDING.
      * LAYER 1 IS THE REMAINDER outside every circle: the thin pial sliver, which is exactly what
        layer 1 is anatomically.

    A point is classified once (upstream: "don't reclassify points"), so the first containing circle
    wins. Coordinates are MICRONS; H01's soma coordinates are NANOMETRES and must be divided by 1000
    first -- the units trap `cortical_prior` names.
    """
    circles = LAYER_CIRCLES["value"]
    (cx, cy), r = circles[0]
    if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
        return WHITE_MATTER
    for i in range(len(circles) - 1):
        (cx, cy), r = circles[i + 1]
        if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
            return f"Layer {6 - i}"
    return "Layer 1"


def assign_sign(layer, rng):
    """EXCITATORY or INHIBITORY for a cell born in `layer`, drawn from H01's measured fraction.

    Refuses to return EXCITATORY for Layer 1 STRUCTURALLY, not merely because the fraction is 0.0.
    A rule that holds only while a number stays zero is a rule an edit can silently break, and this
    one encodes an anatomical fact: layer 1 has essentially no excitatory somata.
    """
    frac = LAYER_E_FRACTION["value"].get(layer)
    if frac is None:
        raise KeyError(
            f"no measured excitatory fraction for {layer!r} -- an unmeasured layer must not be "
            "given a default sign; that is how a made-up number acquires H01's authority"
        )
    if layer == "Layer 1":
        return INHIBITORY  # unrepresentable by measurement AND by construction
    return EXCITATORY if rng.random() < frac else INHIBITORY


def _depth_bands():
    """{layer: (lo, hi)} fractional depth, pia 0.0 -> white-matter border 1.0.

    Split by measured neuron occupancy, not equal thirds: a layer holding a third of the neurons
    owns a third of the depth axis. DERIVED, never typed in -- edit `LAYER_OCCUPANCY` and this moves.
    """
    occ = LAYER_OCCUPANCY["value"]
    # NORMALISED, because the published shares are rounded to 4 decimals and sum to 0.9999 -- so the
    # raw cumulative stops short of the white-matter border and the axis fails to tile. Caught by
    # this module's own tiling assert. Normalising is the fix; loosening the assert would have hidden
    # a 1-in-10,000 gap today and a real one later.
    total = math.fsum(occ[layer] for layer in _LAYER_NAMES)
    bands, lo = {}, 0.0
    for i, layer in enumerate(_LAYER_NAMES):
        hi = 1.0 if i == len(_LAYER_NAMES) - 1 else lo + occ[layer] / total
        bands[layer] = (lo, hi)
        lo = hi
    return bands


DEPTH_BANDS = _depth_bands()


# distinct-from: cortex/vsa_reasoning.py::hv -- that builds a +/-1 HYPERVECTOR spike code from a
#   crc32 seed; this draws a LAMINAR ASSIGNMENT from measured occupancy. Site read; the shared word
#   is "deterministic" and there is nothing to reuse either way.
def place(n_units, seed=0):
    """[(depth, layer, sign)] for `n_units` newly grown cells. Deterministic in `seed`.

    THE LAYER IS DRAWN FROM MEASURED OCCUPANCY, not by sampling the geometry. Layer 1 is the
    remainder OUTSIDE every circle, so uniform sampling in a bounding box over-produces it -- a first
    attempt put 33% of units in L1 against a real ~4% of neurons -- and correcting that would need a
    tissue outer radius nobody measured. `LAYER_OCCUPANCY` is a measurement; this uses it.

    POSITION IS FRACTIONAL CORTICAL DEPTH in [0, 1], pia to white matter. That is how laminar
    position is normally expressed, it is defined for every layer including L1, and it needs no
    invented constant -- unlike an absolute (x, y), which for L1 would require guessing how far the
    tissue extends beyond the outermost circle.

    WHITE MATTER IS NOT A DESTINATION for a grown unit: H01 puts 498 neurons there against 12,872
    cells, it being mostly myelinating glia. `LAYER_OCCUPANCY` already excludes it.
    """
    if n_units < 0:
        raise ValueError("n_units must be >= 0")
    rng = random.Random(seed)  # per-instance, never module-level (determinism invariant)
    occ = LAYER_OCCUPANCY["value"]
    layers = list(_LAYER_NAMES)
    weights = [occ[layer] for layer in layers]
    total = math.fsum(weights)
    out = []
    for _ in range(n_units):
        u = rng.random() * total  # inverse-CDF draw over the measured occupancy
        acc = 0.0
        chosen = layers[-1]
        for layer, w in zip(layers, weights):
            acc += w
            if u <= acc:
                chosen = layer
                break
        lo, hi = DEPTH_BANDS[chosen]
        out.append((lo + rng.random() * (hi - lo), chosen, assign_sign(chosen, rng)))
    return out


def layer_histogram(placed):
    """{layer: count} for a placement, in cortical order."""
    hist = dict.fromkeys(_LAYER_NAMES, 0)
    for _d, layer, _s in placed:
        hist[layer] = hist.get(layer, 0) + 1
    return {k: v for k, v in hist.items() if v}


def main():
    print("=" * 96)
    print("PLACEMENT -- grown units ordered into the H01 cortical geometry")
    print("=" * 96)
    placed = place(4000, seed=0)
    hist = layer_histogram(placed)
    total = len(placed)
    occ = LAYER_OCCUPANCY["value"]
    print(f"{total} units placed into cortex (white matter is not a destination)\n")
    print(
        f"{'layer':<9}{'depth band':>14}{'units':>7}{'share':>8}{'H01':>8}"
        f"{'exc':>6}{'inh':>6}{'E-frac':>9}{'H01':>9}"
    )
    for layer, n in hist.items():
        exc = sum(1 for _d, ly, s in placed if ly == layer and s == EXCITATORY)
        lo, hi = DEPTH_BANDS[layer]
        ref = LAYER_E_FRACTION["value"][layer]
        print(
            f"{layer:<9}{f'{lo:.3f}-{hi:.3f}':>14}{n:>7}{n / total:>8.1%}{occ[layer]:>8.1%}"
            f"{exc:>6}{n - exc:>6}{(exc / n if n else 0.0):>9.4f}{ref:>9.4f}"
        )
    print()
    print("depth 0.0 = pia, 1.0 = white-matter border; bands split by measured neuron occupancy.")
    print("Dale: every sign is assigned once at creation and never changes.")
    print("Layer 1 excitatory cells are UNREPRESENTABLE -- H01 measures that fraction at 0.0000.")
    print(
        f"layer rule agrees with H01's own labels on {LAYER_RULE_AGREEMENT:.2%} of 49,377 somata."
    )
    return 0


def _selfcheck():
    """P8 (1): construct the RED for every claim main() makes."""
    print("PLACEMENT self-check")

    a = place(500, seed=3)
    b = place(500, seed=3)
    assert a == b, "same seed placed different units -- the determinism invariant is broken"
    print(
        "  green  same seed -> identical placement (required: it is PERSISTED with the substrate)"
    )

    c = place(500, seed=4)
    assert a != c, "different seeds produced identical placement -- the seed is being ignored"
    print("  RED    a different seed moves the placement (the seed is not decorative)")

    # THE ANATOMICAL RULE, both ways.
    big = place(4000, seed=11)
    l1_exc = [p for p in big if p[1] == "Layer 1" and p[2] == EXCITATORY]
    assert not l1_exc, f"{len(l1_exc)} excitatory cells were placed in Layer 1 -- unrepresentable"
    print("  green  no excitatory cell in Layer 1, over 4000 draws")

    # OCCUPANCY: the placed distribution must track H01's measured neuron shares, or the prior is
    # decorative. This is the check that would have caught the first geometry misreading, which put
    # 33% of units in Layer 1 against a real 3.9%.
    hist = layer_histogram(big)
    for layer, share in LAYER_OCCUPANCY["value"].items():
        got = hist.get(layer, 0) / len(big)
        assert abs(got - share) < 0.03, (
            f"{layer} placed share {got:.4f} against H01's measured {share:.4f} -- the occupancy "
            "prior is not being followed"
        )
    print("  green  placed layer shares track H01's measured neuron occupancy (tol 0.03)")

    # Depth must be inside the layer's own band, or 'depth' is decorative.
    for d, layer, _s in big:
        lo, hi = DEPTH_BANDS[layer]
        assert lo <= d <= hi, f"{layer} unit at depth {d} outside its band {lo}-{hi}"
    assert abs(sum(hi - lo for lo, hi in DEPTH_BANDS.values()) - 1.0) < 1e-9, (
        "the depth bands do not tile [0,1] -- pia-to-white-matter must be covered exactly once"
    )
    print("  green  every unit's depth lies in its own layer band; the bands tile [0,1] exactly")
    try:
        assign_sign("Layer 1", random.Random(0))
        forced = assign_sign("Layer 1", random.Random(0))
        assert forced == INHIBITORY, "Layer 1 returned a non-inhibitory sign"
    except KeyError:  # pragma: no cover - Layer 1 is measured, so this cannot fire
        raise AssertionError("Layer 1 has a measured fraction and must not raise") from None
    print("  RED    assign_sign refuses EXCITATORY in Layer 1 structurally, not just statistically")

    # An unmeasured layer must not receive a default sign.
    try:
        assign_sign("Layer 42", random.Random(0))
        raise AssertionError("an unmeasured layer was given a sign instead of raising")
    except KeyError:
        pass
    print("  RED    an unmeasured layer raises rather than defaulting to a sign")

    # Signs must track the measured fractions, or the prior is decorative.
    placed = place(6000, seed=7)
    for layer in ("Layer 4", "Layer 6"):
        got = [p for p in placed if p[1] == layer]
        if len(got) < 200:
            continue
        frac = sum(1 for p in got if p[2] == EXCITATORY) / len(got)
        ref = LAYER_E_FRACTION["value"][layer]
        assert abs(frac - ref) < 0.06, (
            f"{layer} placed E-fraction {frac:.4f} against H01's {ref:.4f} -- the prior is not "
            "being used, or the draw is wrong"
        )
    print("  green  placed E-fractions track H01's measured values (L4, L6, tol 0.06)")

    assert place(0, seed=0) == [], "placing zero units must return empty, not raise"
    try:
        place(-1, seed=0)
        raise AssertionError("a negative unit count was accepted")
    except ValueError:
        pass
    print("  green  degenerate counts: 0 -> [], negative -> ValueError")

    # THE INVERSION, pinned. Reading the geometry the obvious way -- circle 0 = Layer 1, ascending
    # outward -- is wrong and put 33% of units in L1. White matter is the INNERMOST circle and
    # Layer 1 is the OUTER remainder, so these two assertions are the ones that catch a regression
    # back to the intuitive-but-wrong reading.
    assert layer_of(1e9, 1e9) == "Layer 1", (
        "a far coordinate must fall to LAYER 1 -- it is the remainder outside every circle (the "
        "pial sliver), not white matter"
    )
    (cx, cy), _r = LAYER_CIRCLES["value"][0]
    assert layer_of(cx, cy) == WHITE_MATTER, (
        "the centre of circle 0 must be WHITE MATTER -- circle 0 is the white-matter bound, not L1"
    )
    print(
        "  RED    the layer order is not the intuitive one: far -> Layer 1, circle 0 -> white matter"
    )

    # NO CEILING. The prior is a DISTRIBUTION over layers, not a quota, so it must place an
    # arbitrarily large population without the proportions drifting and without an upper bound
    # appearing anywhere. P0 treats a limit as a gap to close; a placement rule that silently capped
    # growth would put one back in.
    huge = place(100_000, seed=13)
    assert len(huge) == 100_000, f"placed {len(huge)} of 100,000 -- something is capping growth"
    hhist = layer_histogram(huge)
    worst = max(
        abs(hhist.get(layer, 0) / len(huge) - share)
        for layer, share in LAYER_OCCUPANCY["value"].items()
    )
    assert worst < 0.01, f"proportions drifted by {worst:.4f} at 100,000 units"
    assert min(hhist.values()) > 0, "a layer emptied at scale -- the distribution is degenerate"
    print(f"  green  NO CEILING: 100,000 units placed, worst proportion drift {worst:.4f}")

    print(f"self-check OK: {len(_LAYER_NAMES)} layers, 3 constructed reds, Dale enforced at birth")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
