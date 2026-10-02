"""
Cortical prior -- H01 HUMAN cortex reference data, the ONE place an externally sourced constant lives.

WHAT THIS IS. Measured values from H01 (Shapson-Coe et al.), a petavoxel reconstruction of a cubic
millimetre of HUMAN temporal cortex. Not a model, not a simulation, not something this brain learned:
a measurement of a real human brain, carried here with enough provenance that a reader can check it.

WHAT THIS IS NOT, and the distinction decides where it may be used. `reservoir/cortical_microcircuit.py`
is a Potjans-Diesmann circuit -- an excellent model, built from CAT and MACAQUE anatomy. It is not a
substitute for this file and this file is not a substitute for it. An earlier pass proposed deriving
these constants FROM that simulation "to avoid hardcoding"; that would have replaced human
measurement with model output while sounding like a P1 improvement. Recorded because the reasoning
was seductive and wrong.

P1 AND REFERENCE DATA. A value with `units`, `provenance` and an `accuracy_bound` is not a magic
constant -- it is a citation a reader can audit. What P1 forbids is a behaviour-arbitrating number
tuned until it worked; what this file holds is a number somebody measured in a human brain and
published. The four mandatory attributes exist to keep that distinction enforceable, and `validate`
fails an entry missing any of them.

P4b: THIS MUST NOT ENTER THE LAKE. The lake is what the brain LEARNED. This is what a microscope
saw. A prior persisted into brain_lake.bin would be indistinguishable from learned structure the day
someone reads it back.

DERIVED 2026-08-16 from github.com/ashapsoncoe/h01 at commit db223b8d769d8ae046f0026f12b251103f1dbab0.
The raw sources are ~13 MB and are NOT vendored -- regenerating is an offline step per specs/011
quickstart -- so byte counts and SHA-256 are recorded below and a re-derivation can prove it read the
same bytes.

EVERY FIGURE HERE WAS RE-DERIVED, NOT INHERITED from specs/011's Phase 0 reading, and all of that
reading's claims reproduced exactly: 49,377 soma records, 45,994/45,994 = 100% join coverage on
`agglo_seg`, 7 cortical layer values, 6 layer circles, 11 cell types.

  python -m snn_data_lake.growth.cortical_prior
"""

import sys

#: Upstream provenance. A re-derivation that reads different bytes than these is not a re-derivation.
H01_COMMIT = "db223b8d769d8ae046f0026f12b251103f1dbab0"
SOURCES = {
    "agglo_20200916c3_cell_data.json": (
        11186494,
        "5edc7f775617ecb35baffe5a966b65ce8f7fd0249262f56b6e9b9b44a940b7cc",
    ),
    "goog14r0s5c3_e_i_ratios_feb_2023.csv": (
        1998308,
        "b1e71a3bb387db2f36889788679600dca1c89e7256c1e97b19c3f5084116143c",
    ),
    "excitatory_axons_summary_df.csv": (
        7994,
        "e39d60392241c6883215a3f0aaaad2f598a9ee28356af0d2d8b33d7f58f6c845",
    ),
    "inhibitory_axons_summary_df.csv": (
        9230,
        "f4af2ca6e09c8a67a4809a9575d6a8866c44a13a89c54efe0ddca7a64961dab2",
    ),
    "cortical_bounds_circles.json": (
        496,
        "8c912730e0700110507367f01d8f5a4c419409fc9d93b19d566df826e0aeb449",
    ),
}

_REQUIRED = ("value", "units", "provenance", "accuracy_bound")


# distinct-from: substrate/home.py -- parses a LAKE RECORD's binary layout. distinct-from:
#   infra/brain_bundle.py -- validates a serialized BRAIN BUNDLE at a trust boundary.
#   distinct-from: infra/self_build_loop.py -- reads firewalled FILE BYTES. distinct-from:
#   faculties/is_snn_forward.py -- per-neuron FIRING RATES. distinct-from: reasoning/blimp_bench.py --
#   benchmark label/serve pairs. All five read; none checks reference-datum schema attributes, and
#   nothing here is reusable in either direction.
def entry(value, units, provenance, accuracy_bound):
    """One reference datum. Missing ANY of the four attributes makes it invalid."""
    return {
        "value": value,
        "units": units,
        "provenance": provenance,
        "accuracy_bound": accuracy_bound,
    }


# --------------------------------------------------------------------------------------------------
# LAYER GEOMETRY -- six circles, MICRONS. The units trap the data model names lives here: soma
# coordinates in the cell file are NANOMETRES and these are MICRONS, a 1000x error waiting to happen.
# --------------------------------------------------------------------------------------------------
LAYER_CIRCLES = entry(
    value=(
        ((462.2683543371059, 2805.973374970087), 1416.3922980940845),
        ((462.2683543371059, 2805.973374970087), 1731.0819866303507),
        ((1149.9313666484356, 2458.417758523231), 1476.134832192077),
        ((1149.9313666484356, 2458.417758523231), 1692.4091828120922),
        ((1158.1594071389497, 2411.7378929961988), 2135.2674700433477),
        ((1158.1594071389497, 2411.7378929961988), 2479.253415996829),
    ),
    units="microns; ((centre_x, centre_y), radius), ordered, first-match-wins",
    provenance="cortical_bounds_circles.json, all 6 records verbatim. Three distinct centres x two "
    "radii each (inner/outer bounds).",
    accuracy_bound="exact as published -- these ARE the boundary fit, not a sample of it. The fit's "
    "own error against true laminar boundaries is not quantified upstream and is not "
    "claimed here.",
)

# --------------------------------------------------------------------------------------------------
# PER-LAYER EXCITATORY FRACTION -- fractions, never counts (a count cannot be rescaled honestly).
#
# DERIVED TWICE, INDEPENDENTLY, and the disagreement IS the accuracy bound:
#   route A  the e_i CSV's own curated `cell type`, joined on `segment id`
#   route B  the cell_data JSON's `type`, joined on `agglo_seg`
# Layers 2-6 and white matter agree to <= 0.0049. LAYER 1 DISAGREES BY 0.1227, and that is the
# informative one: route A finds ZERO excitatory somata in L1, route B finds 85. Neuroanatomy is on
# route A's side -- layer 1 is neuropil and interneurons, with essentially no excitatory cell bodies
# -- so the 85 are near-certainly boundary misassignments from the circle geometry. L1 is published
# as 0.0 with a bound of 0.1227 rather than a tight number that would be false precision. This is
# exactly the "a fraction of exactly 0.0" degenerate case specs/011 demanded be handled explicitly.
#
# ONLY NEURONS CARRY A SIGN. Glia and vasculature are EXCLUDED, not counted as inhibitory: there are
# 20,140 oligodendrocytes here and folding them in would sink every fraction. Neurons of unknown sign
# (`unclassified neuron`) are excluded from the ratio and reported, never assigned.
# --------------------------------------------------------------------------------------------------
LAYER_E_FRACTION = entry(
    value={
        "Layer 1": 0.0000,
        "Layer 2": 0.6538,
        "Layer 3": 0.6412,
        "Layer 4": 0.7879,
        "Layer 5": 0.7596,
        "Layer 6": 0.8743,
        "White matter": 0.7731,
    },
    units="dimensionless fraction of SIGNED NEURONS that are excitatory, per layer",
    provenance="goog14r0s5c3_e_i_ratios_feb_2023.csv `cell type` x `cortical layer` (route A), "
    "cross-checked against agglo_20200916c3_cell_data.json `type` (route B). Signed "
    "neurons per layer: L1 540, L2 4469, L3 2851, L4 2998, L5 2092, L6 883, WM 498.",
    accuracy_bound={
        "Layer 1": 0.1227,  # the two routes genuinely disagree here -- see the block above
        "Layer 2": 0.0034,
        "Layer 3": 0.0015,
        "Layer 4": 0.0007,
        "Layer 5": 0.0010,
        "Layer 6": 0.0019,
        "White matter": 0.0049,
    },
)

# --------------------------------------------------------------------------------------------------
# DALE'S PRINCIPLE -- which cell types carry a sign at all.
# --------------------------------------------------------------------------------------------------
SIGNED_TYPES = entry(
    value={
        "excitatory": (
            "pyramidal neuron",
            "spiny stellate neuron",
            "excitatory/spiny neuron with atypical tree",
        ),
        "inhibitory": ("interneuron",),
    },
    units="H01 `type` / `cell type` string literals",
    provenance="the 11 distinct type values in agglo_20200916c3_cell_data.json; counts pyramidal "
    "8744, interneuron 4655, excitatory/spiny atypical 1208, spiny stellate 185.",
    accuracy_bound="classification is upstream's; no independent confirmation is claimed. The L1 "
    "disagreement above is the one place it is measurably uncertain.",
)

UNSIGNED_TYPES = entry(
    value=(
        "unclassified neuron",  # a NEURON of unknown sign -- excluded from ratios, reported
        "oligodendrocyte",
        "microglia/opc",
        "astrocyte",
        "blood vessel cell",
        "c-shaped cell",
        "unknown cell",
    ),
    units="H01 `type` string literals",
    provenance="the same 11 values; 1,275 unclassified neurons and 33,310 non-neuronal cells.",
    accuracy_bound="exact -- membership is a string match, not an estimate.",
)

SOURCE_VOLUME_NM = entry(
    value=(3.864e6, 2.362e6, 0.168e6),
    units="nanometres (x, y, z) -- 3.864 x 2.362 x 0.168 mm",
    provenance="specs/011 research D1, from the H01 publication's stated imaged volume.",
    accuracy_bound="as published; used only for the extrapolation sanity test, never as a scale "
    "factor on a learned quantity.",
)

# --------------------------------------------------------------------------------------------------
# PER-LAYER NEURON OCCUPANCY -- the share of CORTICAL NEURONS each layer holds.
#
# NEURONS ONLY, and the number that forces it: white matter holds 12,872 CELLS but only 498 NEURONS
# (3.5%), because it is mostly myelinating glia. Placing new units by ALL-CELL density would have put
# 28% of them in white matter, which is anatomically absurd. Glia are two thirds of this dataset.
#
# Used instead of re-deriving occupancy from LAYER_CIRCLES, because Layer 1 is defined as the
# REMAINDER outside every circle: uniform sampling in a bounding box over-produces it (the corners
# are all "Layer 1"), and choosing a tissue outer radius to correct that would be an invented
# constant. Occupancy is a measurement; use the measurement.
# --------------------------------------------------------------------------------------------------
LAYER_OCCUPANCY = entry(
    value={
        "Layer 1": 0.0390,
        "Layer 2": 0.3231,
        "Layer 3": 0.2061,
        "Layer 4": 0.2167,
        "Layer 5": 0.1512,
        "Layer 6": 0.0638,
    },
    units="dimensionless share of CORTICAL NEURONS (white matter excluded), sums to 1.0",
    provenance="goog14r0s5c3_e_i_ratios_feb_2023.csv, `cell type` restricted to the four neuronal "
    "types, grouped by `cortical layer`. 13,833 cortical neurons of 14,331 total "
    "(498 in white matter, excluded) out of 45,994 cells.",
    accuracy_bound="counting is exact; the LAYER ASSIGNMENT behind it is H01's own. An independent "
    "re-derivation of that assignment from LAYER_CIRCLES agrees on 45,404/49,377 = "
    "91.95% of somata, every disagreement an ADJACENT layer, so treat per-layer shares "
    "as +/- that boundary error rather than exact.",
)

#: Every entry in this module, for `validate` and for consumers that want to iterate.
ENTRIES = {
    "LAYER_CIRCLES": LAYER_CIRCLES,
    "LAYER_E_FRACTION": LAYER_E_FRACTION,
    "LAYER_OCCUPANCY": LAYER_OCCUPANCY,
    "SIGNED_TYPES": SIGNED_TYPES,
    "UNSIGNED_TYPES": UNSIGNED_TYPES,
    "SOURCE_VOLUME_NM": SOURCE_VOLUME_NM,
}

#: NOT DERIVED HERE, named so its absence is a decision rather than an oversight. GP_STRENGTH_E /
#: GP_STRENGTH_I (the N=1..20 uncensored axon-strength distributions) are in the two summary CSVs and
#: feed specs/011 Phase D, which specs/019 [B10.19] ranks LOWEST priority in that spec. Deriving them
#: needs the AIS-exclusion caveat carried into provenance, and a half-derived distribution with a
#: missing caveat is worse than an absent one.
NOT_YET_DERIVED = ("GP_STRENGTH_E", "GP_STRENGTH_I")


def validate(entries=None):
    """[(name, problem)] for every reference datum missing one of the four mandatory attributes.

    An entry that loses its `accuracy_bound` becomes a fake measurement -- a number with the
    authority of a citation and none of the checkability (specs/011 research D7).
    """
    entries = ENTRIES if entries is None else entries
    problems = []
    for name, e in sorted(entries.items()):
        if not isinstance(e, dict):
            problems.append((name, f"not an entry dict, got {type(e).__name__}"))
            continue
        for attr in _REQUIRED:
            if attr not in e:
                problems.append((name, f"missing mandatory attribute {attr!r}"))
            elif e[attr] is None or e[attr] == "":
                problems.append((name, f"attribute {attr!r} is empty"))
    return problems


def e_fraction(layer):
    """The excitatory fraction for `layer`, or None if this dataset does not cover it.

    None means NOT MEASURED. A caller must not substitute 0.5 or any other default: an unmeasured
    layer is a gap in the prior, and silently filling it is how a made-up number acquires H01's
    authority.
    """
    return LAYER_E_FRACTION["value"].get(layer)


def main():
    print("=" * 96)
    print("CORTICAL PRIOR -- H01 human cortex reference data")
    print("=" * 96)
    print(f"upstream commit {H01_COMMIT}")
    print(f"{len(SOURCES)} source files, {sum(b for b, _ in SOURCES.values()):,} bytes total")
    print()
    print(f"{'layer':<14}{'E-fraction':>12}{'+/- bound':>12}")
    for layer, frac in LAYER_E_FRACTION["value"].items():
        print(f"{layer:<14}{frac:>12.4f}{LAYER_E_FRACTION['accuracy_bound'][layer]:>12.4f}")
    print()
    print(
        f"{len(LAYER_CIRCLES['value'])} layer circles (microns), "
        f"{len(SIGNED_TYPES['value']['excitatory'])} excitatory + "
        f"{len(SIGNED_TYPES['value']['inhibitory'])} inhibitory signed types, "
        f"{len(UNSIGNED_TYPES['value'])} unsigned"
    )
    print(f"not yet derived: {', '.join(NOT_YET_DERIVED)} (specs/011 Phase D)")
    print()

    problems = validate()
    for name, why in problems:
        print(f"  INVALID {name}: {why}")
    print(
        "prior OK: every entry carries value + units + provenance + accuracy_bound"
        if not problems
        else f"prior FAIL: {len(problems)} invalid entry attribute(s)"
    )
    return 1 if problems else 0


def _selfcheck():
    """P8 (1): construct the RED for every claim main() makes."""
    print("CORTICAL PRIOR self-check")

    assert not validate(), f"the shipped entries are invalid: {validate()}"
    print(f"  green  all {len(ENTRIES)} entries carry the four mandatory attributes")

    for attr in _REQUIRED:
        broken = dict(LAYER_E_FRACTION)
        broken.pop(attr)
        assert validate({"BROKEN": broken}), (
            f"an entry missing {attr!r} was ACCEPTED -- the schema check is vacuous"
        )
    print(f"  RED    an entry missing any of {_REQUIRED} is rejected")

    hollow = dict(LAYER_E_FRACTION)
    hollow["accuracy_bound"] = ""
    assert validate({"HOLLOW": hollow}), "an emptied accuracy_bound was ACCEPTED"
    print("  RED    an entry whose accuracy_bound is emptied is rejected")

    assert e_fraction("Layer 1") == 0.0, "L1 must be exactly 0.0 -- the published degenerate case"
    assert e_fraction("Layer 99") is None, "an absent layer must return None, never a default"
    print("  green  degenerate cases: L1 is exactly 0.0; an absent layer returns None, not 0.5")

    b = LAYER_E_FRACTION["accuracy_bound"]
    others = [v for k, v in b.items() if k != "Layer 1"]
    assert b["Layer 1"] > max(others) * 10, (
        f"L1's bound {b['Layer 1']} is not visibly looser than the rest (max {max(others)}) -- the "
        "two derivation routes disagree there by 0.1227 and the bound must show it"
    )
    print(f"  green  L1's bound {b['Layer 1']} is >10x the worst other layer ({max(others)})")

    for layer, f in LAYER_E_FRACTION["value"].items():
        assert 0.0 <= f <= 1.0, f"{layer} fraction {f} outside [0,1]"
    print("  green  every published fraction is within [0,1]")

    print(f"self-check OK: {len(ENTRIES)} entries, 2 constructed reds, degenerate cases pinned")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
