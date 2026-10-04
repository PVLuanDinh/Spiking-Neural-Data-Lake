"""
Placement -- every grown neuron gets a PLACE, a LAYER and an immutable SIGN, from the H01 human prior.

WHAT THIS DOES. `growth/neurogenesis_engine.py` grows units on the sleep tick and folds them to the
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

T043 GENERALIZED THIS TO EVERY DECLARED REGION, without moving a byte of the above. `place` takes a
`region` and answers from a prior keyed by the region's declared (neuromere, zone) coordinate --
AT THIS GRAIN ONLY THE NEUROMERE IS LOAD-BEARING (the zone is carried and pinned inert; a
dorsoventral field is the missing half). Cortex keeps the H01 measured arm BYTE-IDENTICAL (the
additive-from-zero control -- a pinned golden asserts it), and every other region in
`REGION_COORDS` takes the GRN arm, whose label is the settled arealization output at that region's
own prosomeric coordinate and whose SIGN IS None -- refused, not defaulted, because no measured
E-fraction exists outside H01 cortex and a default is how a made-up number acquires a
measurement's authority. The same section composes the project's growers -- a grower -> region
routing table asserted to sit inside growth_program's owner-ratified GROWABLE partition --
through the 007 interface (`growth/growable.GrowthCoordinator`): the first `register_seam`
caller outside growable.py's own self-check (no live-tick caller: a sleep-tick wire was built
and measured a NULL on 2026-09-02 -- placement has no live reader yet -- and reverted under P3;
specs/022 evidence/s1-placement-null.md, pinned by tests/test_placement_charge_null.py), with
the 007 contract's C1.7 safety refusal AND T042's SCAFFOLD closure enforced as
one admission rule at the growth call site (`place_grown`).

  python -m snn_data_lake.growth.placement
"""

import math
import random
import sys
import zlib

from snn_data_lake.growth.arealization import (
    NEUROMERES,
    REGION_COORDS,
    discretize_sharp,
    emit_regions,
    graded_profiles,
    postmitotic_discretize,
    regime_of,
)
from snn_data_lake.growth.cortical_prior import (
    LAYER_CIRCLES,
    LAYER_E_FRACTION,
    LAYER_OCCUPANCY,
)
from snn_data_lake.growth.growable import GrowthCoordinator, is_safety_seam
from snn_data_lake.growth.growth_program import GROWABLE, SCAFFOLD

EXCITATORY = 1
INHIBITORY = -1

#: Public: consumers persist a layer by its INDEX here, so the stored form is compact and a rename
#: cannot silently re-map an already-written record to a different layer.
LAYER_NAMES = ("Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6")
_LAYER_NAMES = LAYER_NAMES  # internal alias kept so existing references stay valid
WHITE_MATTER = "White matter"

#: Agreement between `layer_of` and H01's OWN published `cortical layer` labels, measured over all
#: 49,377 soma records: 45,404 / 49,377. Every disagreement is an ADJACENT layer (top confusions
#: L4->L3 971, L5->L6 537, L3->L2 501) -- boundary cells the circle fit places one band over. This
#: is the accuracy bound for anything derived from placement, and it is measured, not asserted.
#:
#: APPENDED 2026-09-10, the text above kept VERBATIM because it is the record of what this constant
#: was taken to mean and rewriting it would destroy that record: THE SENTENCE "Every disagreement is
#: an ADJACENT layer" IS FALSE AS MEASURED. T062's H01 ingest re-measured the confusion matrix and
#: filed it as an open finding: "1,277 of the 3,973 disagreements are non-adjacent (519 of 3,016
#: after dropping the buckets), the largest being Layer 3 -> White matter 315 and, from id '0'
#: alone, Layer 1 -> White matter 246." Quoted from
#: `specs/022-unified-brain-arc/evidence/t062-h01-ingest.md` 3.5(15); nothing here is re-derived.
#: 'the buckets' in that parenthetical is the record's label for the id-'0'-only cut (dated line
#: appended to 3.5(15)); the 1,819-row exclusion below is the eight-id cut.
#: WHAT IS UNCHANGED: LAYER_RULE_AGREEMENT = 0.9195 stands exactly as recorded -- T062's fold made
#: that finding record-only, and two self-checks and a test assert on the value. WHAT CHANGES IS HOW
#: THE BOUND MAY BE READ: the disagreements are not all one band over, so "adjacent" may not be used
#: to bound the error of anything derived from placement. The same record measures a second reading
#: of the agreement itself, both kept: 45,404 / 49,377 = 0.9195 over all somata, and 45,129 / 47,558
#: = 0.9489 over the somata left after excluding the 1,819 bucket rows.
LAYER_RULE_AGREEMENT = 0.9195


# distinct-from: faculties/is_snn_forward.py -- its "layer" is a LIF network layer with its own
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


# distinct-from: faculties/vsa_reasoning.py::hv -- that builds a +/-1 HYPERVECTOR spike code from a
#   crc32 seed; this draws a LAMINAR ASSIGNMENT from measured occupancy. Site read; the shared word
#   is "deterministic" and there is nothing to reuse either way.
def place(n_units, seed=0, region="cortex"):
    # distinct-from: faculties/vsa_reasoning.py -- that builds a +/-1 HYPERVECTOR spike code from a
    #   crc32 seed; this draws a LAMINAR/REGIONAL ASSIGNMENT from a prior. Site read (see the
    #   module-level marker above); the shared word is "deterministic", nothing to reuse.
    """[(depth, layer, sign)] for `n_units` newly grown cells. Deterministic in `seed`.

    THE REGION SELECTS THE ARM (T043). The default, `region="cortex"`, runs the pre-T043 body
    below UNTOUCHED -- byte-identical output, pinned by the self-check's golden crc. Any other
    key of `REGION_COORDS` routes to the GRN arm (`_place_grn`): label from the settled
    arealization output at the region's declared coordinate (only the NEUROMERE is
    load-bearing at this grain -- region_prior's disclosure), depth uniform and DISCLOSED as
    uninformative, sign None -- refused, never defaulted. An unknown or non-string region
    raises. THIS IS THE PRIOR SURFACE, NOT A GROWTH ADMISSION: querying any region's prior
    (scaffold regions included -- main() prints all 16) is legitimate; admission is
    place_grown's job, where scaffold regions are refused.

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
    if region != "cortex":
        if not isinstance(region, str):
            raise ValueError(f"region must be a region name, got {type(region).__name__}")
        return _place_grn(n_units, seed, region_prior(region))
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
    """{layer: count} for a placement, in cortical order. H01-ARM PLACEMENTS ONLY: a GRN
    region label is not a cortical layer, and silently absorbing one here would report a
    region inside a laminar census (T043 panel) -- it raises instead."""
    hist = dict.fromkeys(_LAYER_NAMES, 0)
    for _d, layer, _s in placed:
        if layer not in hist:
            raise ValueError(
                f"{layer!r} is not a cortical layer -- layer_histogram takes the H01 arm's "
                "output only"
            )
        hist[layer] += 1
    return {k: v for k, v in hist.items() if v}


# ==================================================================================================
# T043 -- per-region priors keyed by the declared prosomeric coordinate, and the registered
# growers composed through the 007 interface
# (specs/007-whole-brain-neurogenesis/contracts/growth-consumption.md). P8 invention #7's first
# spec-backed consumer of the arealization GRN under specs/022 -- NOT a 007 C2 consumer: C2.1
# disqualifies a cortex-internal pair, and both modules sit in REGIONS["cortex"].
# distinct-from: faculties/grid_place_map.py -- its `_match_bar` midpoint is a place-cell firing
#   threshold; nothing here thresholds activity. Site read; the shared word is "midpoint".
# distinct-from: sensorimotor/arm_hardware.py -- its configurations are servo-rating arithmetic for
#   the physical arm; unrelated job. Site read; the shared word is "combinations".
# distinct-from: gates/gradient_claim_gate.py -- its "placement" is where a statistical bar
#   sits relative to a null; this places NEURONS. Site read; the shared word is "placement".
# distinct-from: growth/cortical_prior.py -- REUSED, not duplicated: it is this module's existing
#   import for the measured H01 fractions; nothing re-derives them here.
# distinct-from: growth/arealization.py -- REUSED, not duplicated: this section CONSUMES its
#   emit_regions/graded arm; the zero-repression comparator at its L567 is not re-implemented.
# ==================================================================================================

#: The presentation grid both arealization arms accept (even -- graded_profiles refuses odd).
_GRID_NODES = 16


# distinct-from: growth/arealization.py -- NOT distinct but REUSED: the binning below imports
#   discretize_sharp (the T043 panel's P2 finding killed the first draft's inline re-derivation).
# distinct-from: faculties/grid_place_map.py -- its midpoint is a place-cell firing bar;
#   unrelated job, site read; the shared word is "midpoint".
# distinct-from: reasoning/arc_mint.py -- its grid rules recolour ARC task cells; unrelated
#   job, site read; the shared word is "grid".
def _ap_node(neuromere):
    """The presentation-grid node for a neuromere: its midpoint on the 14-entry NEUROMERES
    partition, binned by the ONE sharp discretizer (`arealization.discretize_sharp`). Not an
    identity: the position comes from the 14-way partition and the bin from the 16-way grid,
    so the composition genuinely moves information between two DECLARED partitions."""
    ap = (NEUROMERES.index(neuromere) + 0.5) / len(NEUROMERES)
    return int(discretize_sharp([ap], _GRID_NODES)[0])


# distinct-from: sensorimotor/arm_hardware.py -- servo-rating arithmetic for the physical arm;
#   unrelated job, site read (also declared in the T043 header block above).
# distinct-from: faculties/theta_gamma_code.py -- its _ensemble draws a sparse LIF population per
#   gamma sub-cycle; nothing here draws populations. Site read; shared word is "which cells".
def region_prior(region):
    """The prior for `region`, keyed by its declared (neuromere, zone) coordinate -- which ARM
    answers "where does a grown unit go, and what identity does it carry". AT THIS GRAIN ONLY
    THE NEUROMERE IS LOAD-BEARING: the zone is carried and declared, and nothing downstream
    reads it yet (the self-check PINS that degeneracy so it cannot silently change) -- a
    dorsoventral field is what would make it load-bearing, and this build does not have one.

    H01 IS SELECTED BY NAME, NOT BY COORDINATE. cortex and hippocampus share ("sp", "alar") at
    this grain, and a coordinate-keyed lookup would hand H01's measured laminar occupancy to
    six regions it was never measured in. A measurement outranks the model at the one region
    where the measurement exists; everywhere else the GRN's settled output at the region's own
    coordinate is the prior, DISCLOSED as model-derived with zero tissue measurement behind it.

    DEGENERACY, DISCLOSED NOT SMOOTHED: `same_prior_regions` names the regions whose prior is
    IDENTICAL to this one -- the neuromere class over GRN-arm regions (the T043 panel caught
    the first draft disclosing the coordinate class instead, which wrongly listed cortex and
    wrongly omitted hypothalamus). Seven regions sit at ('sp','alar') and the grid cannot
    separate them; a prior that pretended to tell hippocampus from amygdala here would be
    invented resolution (T041's non-discrimination result).
    """
    coord = REGION_COORDS.get(region) if isinstance(region, str) else None
    if coord is None:
        raise ValueError(
            f"unknown region {region!r} -- not a declared prosomeric coordinate "
            f"(REGION_COORDS keys: {sorted(REGION_COORDS)})"
        )
    nm, _zone = coord  # the zone rides in `coord`; not load-bearing at this grain (pinned)
    if region == "cortex":
        return {
            "arm": "h01_measured",
            "coord": coord,
            "same_prior_regions": (),
            "disclosure": "measured laminar occupancy + E-fractions (H01); the one region "
            "with a tissue measurement -- selected by NAME, shared with no other region",
        }
    same = tuple(
        sorted(
            r for r, (n2, _z2) in REGION_COORDS.items() if r not in (region, "cortex") and n2 == nm
        )
    )
    node = _ap_node(nm)
    labels, states = emit_regions(_GRID_NODES)
    prior = {
        "arm": "grn_model",
        "coord": coord,
        "regime": regime_of(nm),
        "consensus_label": labels[node],
        "consensus_state": tuple(int(v) for v in states[node]),
        "same_prior_regions": same,
        "disclosure": "model-derived (the T040/T041 GRN settled at the declared coordinate); "
        "no tissue measurement; sign is REFUSED (None), not defaulted",
    }
    if prior["regime"] == "graded":
        prof = graded_profiles(_GRID_NODES)
        prior["graded_identity"] = tuple(round(float(v), 6) for v in prof[node])
        prior["label"] = str(postmitotic_discretize(prof)[node])
    else:
        # SHARP regime: the compartment IS the lineage, decided at birth (T041's two-regime
        # split) -- the label is the neuromere itself, stated directly rather than dressed up
        # as a computation.
        prior["label"] = nm
    return prior


def _place_grn(n_units, seed, prior):
    """The GRN arm: [(pos, label, None)] for `n_units` units of a non-cortical region.

    POS IS DRAWN UNIFORM in [0, 1] and disclosed UNINFORMATIVE: no measured within-region
    occupancy exists outside H01 cortex, and inventing bands would be a made-up number wearing
    a measurement's authority. LABEL is the prior's settled identity -- PER-COORDINATE, so
    every unit of a region carries the same label at this grain (disclosed; the GRN answers
    per coordinate, not per unit). SIGN IS None: Dale needs a measured E-fraction, none exists
    here, and None is the explicit absence marker a consumer must branch on -- `assign_sign`'s
    own rule ("an unmeasured layer must not be given a default sign") applied region-wide.
    """
    rng = random.Random(seed)
    label = prior["label"]
    return [(rng.random(), label, None) for _ in range(n_units)]


#: THE REGISTERED GROWERS (T043): a grower -> region routing table -- every grower this
#: project runs, keyed to the region whose prior places its grown units. Each name is a
#: growth seam registered through the 007 interface (compose_growers), and place_grown is
#: the placement route a grower's new units WILL take: today only function_preserving_growth
#: is driven through it, from this module's own __main__; neurogenesis_engine still calls
#: place() directly (open composition debt: the composed sleep-tick wire measured a NULL on
#: 2026-09-02 and was reverted, specs/022 evidence/s1-placement-null.md); ng_snn is registered
#: only. distinct-from: growth/growth_program.py::GROWABLE -- that is the owner-ratified
#: REGION partition (10 regions, pinned by equality in T042's self-check, counted by T046);
#: this is a 3-entry grower routing table, and the self-check asserts every route lands
#: INSIDE that partition, so the two are one rule rather than two.
GROWER_REGIONS = {
    "neurogenesis_engine": "cortex",  # the live sleep-tick grower; calls place() itself today
    "ng_snn": "hippocampus",  # torch tier -- registered; fit-driven traffic NOT claimed here
    "function_preserving_growth": "hippocampus",  # stdlib; carries the composed-path demo
}


# distinct-from: growth/arealization.py -- its L567 comparator is the zero-repression census
#   instrument; nothing here compares censuses. Site read; term matcher fired on generic words.
# distinct-from: sleep/hippocampal_memory.py -- a replay buffer over the durable lake;
#   nothing here buffers replays. Site read; unrelated job.
# distinct-from: faculties/interoceptive_integration.py -- a salience accessor; unrelated. Read.
# distinct-from: plasticity/r_stdp.py -- worker fork/join barriers; unrelated. Read.
# distinct-from: faculties/access_consciousness.py -- a workspace submit site; unrelated. Read.
def place_grown(grower, n_units, seed=0):
    """Placement for `grower`'s newly grown units -- THE GROWTH CALL SITE, enforcing the 007
    contract's C1.7 refusal AND T042's SCAFFOLD closure as ONE admission rule. The contract's
    own 2026-08-03 correction records that `is_safety_seam` had no caller outside its own
    module, so the refusal was structurally unreachable; this call site is the enforcement it
    demands, verified in the self-check by ATTEMPTING refusals and DISCRIMINATING the branch
    by message (the T043 panel's blocker: a bare except also caught the unknown-grower branch,
    leaving the C1.7 red unreachable). No live-tick caller reaches this: a sleep-tick wire was
    built and measured a NULL on 2026-09-02 (placement has no live reader; specs/022
    evidence/s1-placement-null.md, pinned by tests/test_placement_charge_null.py) and reverted.

    REFUSAL IS FIRST AND UNCONDITIONAL (the T042 lesson: a gate one arm can short-circuit is
    not a gate): a safety-seam-looking name is refused before the grower is even looked up, so
    no future branch can route around it.
    """
    if is_safety_seam(grower):
        raise ValueError(
            f"refusing growth placement for {grower!r}: the effector / actuator / reflex "
            "envelope is bounded by design and never grows (007 C1.7, enforced at the call "
            "site)"
        )
    region = GROWER_REGIONS.get(grower)
    if region is None:
        raise ValueError(
            f"unknown grower {grower!r} -- not in the grower set {sorted(GROWER_REGIONS)}"
        )
    if region in SCAFFOLD:
        raise ValueError(
            f"refusing growth placement for {grower!r}: region {region!r} is in the "
            "owner-ratified SCAFFOLD (T042's closure -- fixed regions never grow, and the "
            "T043 arm inherits that rule rather than opening a second path around it)"
        )
    return place(n_units, seed=seed, region=region)


#: The keys `labels_by` groups on (G7 pre-registration 2.2): the population keys G5's return path,
#: G6's census and per-batch attribution read. `origin` separates born from derived/legacy units.
LABEL_KEYS = ("region", "batch_id", "problem_id", "layer", "origin")


# distinct-from: growth/neurogenesis_engine.py -- its `placement` property RETURNS the stored
#   (depth, layer, sign) rows; this DRAWS them, role-conditioned. Site read (L542).
# distinct-from: engine/signal_ring.py, faculties/forward_model.py, faculties/wholistic_loop.py,
#   growth/h01_ingest.py -- a ring cursor, a sensory predictor, a metacognitive signal source and a
#   propagation probe; the guard matched docstring words only. All four sites read.
def _place_by_role(roles, seed):
    """[(depth, layer, sign)] with the SIGN FIXED BY THE ROLE and the layer drawn from H01's joint
    P(layer | sign), proportional to occupancy(layer) x P(sign | layer) -- the same measured
    E-fractions assign_sign reads. Keeps H01's CONDITIONAL; the E/I mixture is the caller's (G7 2.2)."""
    rng = random.Random(seed)
    occ, ef = LAYER_OCCUPANCY["value"], LAYER_E_FRACTION["value"]
    out = []
    for role in roles:
        sign = EXCITATORY if role == "E" else INHIBITORY
        weights = [occ[L] * (ef[L] if role == "E" else 1.0 - ef[L]) for L in _LAYER_NAMES]
        u, acc, chosen = rng.random() * math.fsum(weights), 0.0, None
        for layer, w in zip(_LAYER_NAMES, weights):
            if w <= 0.0:
                continue  # a zero-weight layer (E in Layer 1) is never drawn, not even at u == 0
            acc += w
            chosen = layer
            if u < acc:
                break
        lo, hi = DEPTH_BANDS[chosen]
        out.append((lo + rng.random() * (hi - lo), chosen, sign))
    return out


# distinct-from: place / place_grown -- those return bare (depth, layer, sign) tuples; this returns
#   the FULL admission record P4d's provenance check reads (G7 / T115). It delegates the default
#   draw to place() itself, so there is one laminar sampler, not two.
# distinct-from: growth/neurogenesis_engine.py, engine/signal_ring.py, faculties/forward_model.py,
#   faculties/wholistic_loop.py, growth/h01_ingest.py -- see _place_by_role above; all sites read.
def place_labeled(
    n_units, *, unit_ids, region, site, problem_id, batch_id, seed, centre_um, density_per_mm2,
    roles=None,
):
    """Full label records for one grown batch (G7 pre-registration 2.2, as built: unit_ids, region,
    centre_um and density_per_mm2 are the build's additions to the frozen signature, disclosed).

    roles=None: (depth, layer, sign) is EXACTLY place(n_units, seed=seed, region=region) -- the cortex
    golden does not move. roles given ("E"/"I" per unit): cortex only (no measured E-fraction exists
    elsewhere, O3), sign = role, layer from H01's P(layer | sign). Tangential offsets (x_um, y_um) are
    uniform in the disc that holds n_units at `density_per_mm2` around `centre_um`, from an RNG stream
    SEPARATE from the laminar one. The density has NO default: G4's Part B records it.
    """
    unit_ids = list(unit_ids)
    if len(unit_ids) != n_units:
        raise ValueError(f"{len(unit_ids)} unit_ids for {n_units} units")
    if not (isinstance(density_per_mm2, (int, float)) and math.isfinite(density_per_mm2)
            and density_per_mm2 > 0):
        raise ValueError(f"density_per_mm2 must be a finite positive number, got {density_per_mm2!r}")
    if roles is None:
        base = place(n_units, seed=seed, region=region)
    else:
        roles = tuple(roles)
        if region != "cortex":
            raise ValueError(
                f"roles refused for {region!r}: a role fixes a Dale sign, and no measured "
                "E-fraction exists outside H01 cortex (G7 O3)"
            )
        if len(roles) != n_units or any(r not in ("E", "I") for r in roles):
            raise ValueError(f"roles must be {n_units} of 'E'/'I', got {roles!r}")
        base = _place_by_role(roles, seed)
    radius_um = math.sqrt(n_units / (density_per_mm2 * math.pi)) * 1000.0
    off = random.Random(zlib.crc32(b"place_labeled/offsets") ^ (seed & 0xFFFFFFFF))
    cx, cy = centre_um
    ap, dv = site
    out = []
    for uid, (depth, layer, sign) in zip(unit_ids, base):
        r, theta = radius_um * math.sqrt(off.random()), 2.0 * math.pi * off.random()
        out.append({
            "unit_id": uid,
            "batch_id": batch_id,
            "position": (ap, dv, cx + r * math.cos(theta), cy + r * math.sin(theta), depth),
            "layer": layer,
            "sign": sign,
            "region": region,
            "problem_id": problem_id,
            "origin": "born",
        })
    return out


def label_of(records, unit_id):
    """The label record for `unit_id`, or None -- the one id-addressable reader (G5 D5)."""
    for rec in records:
        if rec.get("unit_id") == unit_id:
            return rec
    return None


def labels_by(records, key):
    """{value: [unit ids]} over `records` for one of LABEL_KEYS -- G5 and G6 read populations here
    and define no label of their own (P2)."""
    if key not in LABEL_KEYS:
        raise ValueError(f"labels_by key must be one of {LABEL_KEYS}, got {key!r}")
    out = {}
    for rec in records:
        out.setdefault(rec.get(key), []).append(rec.get("unit_id"))
    return out


# distinct-from: growth/growth_program.py -- its shared program decides GROW/QUIESCE proposals
#   from local fields; this coordinates seam WIDTHS into placements. Site read; adjacent by
#   design (T042/T043 are sibling tasks), no shared code.
# distinct-from: faculties/self_improvement_suite.py -- its _rung_growth measures duplicate_unit's
#   current-difference eps-invariant on a standalone layer; the self-check here demonstrates
#   gradmax insertion THROUGH the coordinator with placements. Site read; different check.
# distinct-from: reasoning/reasoning_compose.py -- routes reasoning sub-steps; unrelated. Read.
# distinct-from: infra/cockpit.py -- drives a GET through the HTTP handler; unrelated. Read.
def compose_growers(base_seed=0):
    """The registered growers composed through the 007 interface: one GrowthCoordinator with a
    seam per grower -- the first `register_seam` caller outside growable.py's own self-check
    (NOT live: the only callers are this module's own __main__; a sleep-tick caller was built,
    measured a NULL on 2026-09-02 -- no live reader of placement exists -- and reverted under
    P3, specs/022 evidence/s1-placement-null.md) -- each seam wired so a growth event RIPPLES
    into placements:
    `coord.grow(name, w)` appends exactly the delta units' placements to `ledger[name]`,
    deterministically. Event seed = crc32(name) ^ base_seed ^ old_width (crc32, never builtin
    hash -- the persistence invariant): a replay of the same growth SCHEDULE places
    identically; same-grower events can never collide (XOR is injective in old_width); and
    cross-grower separation is a MEASURED bound, not a "never" -- the pairwise crc32 XOR
    distances of the three names are >= 2.76e8, so a collision needs a grower at >= 2^28
    units, far above this repo's measured ~40k ceiling.

    TWO DISCLOSED DEPENDENCES. (1) Widths start at 0 AT COMPOSITION TIME: the seam tracks
    growth events THROUGH the coordinator from here forward, not a grower's prior history
    (the engine's existing units were placed by its own place() calls). (2) The ledger is
    growth-SCHEDULE-dependent: 0->5 in one event and 0->4->5 in two place differently,
    because the event seed folds old_width -- a consumer that persists the ledger must
    persist the schedule with it. Treat the returned ledger as READ-ONLY: the resize gates
    on its length, so hand-appending desynchronizes the seam. ng_snn is REGISTERED at width
    0 and its fit-driven traffic is NOT claimed: it is the torch tier, P3 honesty forbids
    reporting a charge nobody measured. Returns (coordinator, ledger).
    """
    coord = GrowthCoordinator()
    ledger = {}
    for name in sorted(GROWER_REGIONS):
        coord.register_seam(name, 0)
        ledger[name] = []

        def _resize(new_width, _name=name):
            old = len(ledger[_name])
            if new_width > old:
                ev_seed = (zlib.crc32(_name.encode()) ^ base_seed ^ old) & 0xFFFFFFFF
                ledger[_name].extend(place_grown(_name, new_width - old, seed=ev_seed))

        coord.connect(name, _resize)
    return coord, ledger


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

    print()
    print("T043 -- the (neuromere, zone)-keyed prior, per region")
    print(f"{'region':<18}{'coord':<17}{'arm':<14}{'regime':<9}{'label':<11}{'sign'}")
    for region in sorted(REGION_COORDS):
        pr = region_prior(region)
        if pr["arm"] == "h01_measured":
            row = ("h01_measured", "--", "(6 layers)", "measured")
        else:
            row = (pr["arm"], pr["regime"], pr["label"], "None (refused)")
        print(f"{region:<18}{str(pr['coord']):<17}{row[0]:<14}{row[1]:<9}{row[2]:<11}{row[3]}")
    print(
        "H01 is selected by NAME: seven regions share ('sp','alar') and only cortex has the "
        "measurement."
    )
    print(
        "GRN-arm sign is None -- REFUSED, not defaulted: no measured E-fraction exists outside "
        "the H01 cortex sample."
    )
    coord, ledger = compose_growers()
    print(
        f"registered growers (via 007 register_seam, first caller outside growable's own "
        f"self-check; no live-tick caller yet): {coord.seams()} -- ng_snn at width 0, its "
        "fit-driven traffic is the learning slice's claim, not this module's."
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

    # ================================ T043 checks =====================================
    # BYTE-IDENTITY CONTROL (additive-from-zero): the default arm and the explicit-cortex arm
    # are the same code path (disclosed self-referential), so the INDEPENDENT leg is the golden
    # crc pinned at a3191e3d, BEFORE the T043 edit. Reachable red measured on mutants: a reflow
    # of the cortical draw reds it; a sub-0.1% occupancy drift at n=12 does not (that magnitude
    # is the occupancy-share check's job above, tol 0.03).
    d = place(12, seed=5)
    assert d == place(12, seed=5, region="cortex"), (
        "region='cortex' diverged from the default arm -- the dispatch is not byte-identical"
    )
    assert zlib.crc32(repr(d).encode()) == 3377043314, (
        "the cortical draw was reflowed -- T043 was required to be additive-from-zero, and "
        "this crc was pinned against the pre-T043 tree at a3191e3d"
    )
    print("  green  T043: cortical arm byte-identical (dispatch equality + golden crc @ a3191e3d)")

    # THE GRN ARM: deterministic, sign REFUSED (None), depth in [0,1], unknown and non-string
    # regions raise.
    h1 = place(50, seed=3, region="hippocampus")
    assert h1 == place(50, seed=3, region="hippocampus"), "GRN arm is not deterministic in seed"
    assert h1 != place(50, seed=4, region="hippocampus"), "GRN arm ignores its seed"
    assert all(s is None for _p, _l, s in h1), (
        "a GRN-arm unit carries a sign -- no measured E-fraction exists outside H01 cortex, "
        "and a default sign is a fabricated measurement"
    )
    assert all(0.0 <= p <= 1.0 for p, _l, _s in h1), "GRN-arm position left [0,1]"
    assert all(s is not None for _d2, _l2, s in d), "a cortical unit lost its measured sign"
    for bad_region in ("ventricle", None, 42, ["cortex"]):
        try:
            place(3, seed=0, region=bad_region)
            raise AssertionError(f"an undeclared region {bad_region!r} was placed")
        except ValueError:
            pass
    print(
        "  RED    T043: sign is None on the GRN arm (refused, not defaulted); "
        "unknown / non-string regions raise"
    )

    # THE PRIOR KEY, measured honestly. cortex and hippocampus share ("sp","alar") and the arms
    # STILL differ -- H01 is selected by name. The identical-prior class is the NEUROMERE class
    # (the zone is carried, not load-bearing), asserted both ways: hypothalamus ('sp','basal')
    # gets the SAME prior as hippocampus and IS listed; cortex shares the coordinate and is NOT.
    pc, ph = region_prior("cortex"), region_prior("hippocampus")
    assert pc["coord"] == ph["coord"] == ("sp", "alar"), "the declared coordinates moved"
    assert pc["arm"] == "h01_measured" and ph["arm"] == "grn_model", (
        "a coordinate-keyed lookup handed H01 to a coordinate-mate -- the measurement must be "
        "selected by NAME"
    )
    assert ph["same_prior_regions"] == (
        "amygdala",
        "basal_ganglia",
        "claustrum",
        "hypothalamus",
        "insula",
        "olfactory",
    ), (
        "the identical-prior class is wrong -- it must be the NEUROMERE class over GRN-arm "
        "regions (hypothalamus in, cortex out), never the coordinate class"
    )
    # ZONE-DEGENERACY PIN (disclosed, not smoothed): same neuromere + different zone yields the
    # IDENTICAL prior today. This pin REDS the day the zone becomes load-bearing, which is the
    # moment same_prior_regions and the degeneracy disclosure must both change.
    pt, pe = region_prior("thalamus"), region_prior("epithalamus")
    assert pt["same_prior_regions"] == ("epithalamus",) and pe["same_prior_regions"] == (
        "thalamus",
    ), "the p2 identical-prior pair moved"
    keyed = lambda pr: {k: v for k, v in pr.items() if k not in ("coord", "same_prior_regions")}  # noqa: E731
    assert pt["coord"] != pe["coord"] and keyed(pt) == keyed(pe), (
        "the zone became load-bearing -- update same_prior_regions and the degeneracy disclosure"
    )
    # AGREEMENT IS BY CONSTRUCTION, kept as a REGRESSION GUARD: both readouts are monotone in
    # the same organizer field a(x) (the postmitotic label reduces to [a > 1/2], and the Boolean
    # consensus settle is a bijection of the same fgf8 input), so agreement corroborates
    # NOTHING -- what a disagreement proves is that one arm broke. Asserted across all 16
    # nodes; the reachable red is a neutered graded arm, not any natural input.
    prof16 = graded_profiles(_GRID_NODES)
    pm16 = postmitotic_discretize(prof16)
    lab16, _st16 = emit_regions(_GRID_NODES)
    assert [str(x) for x in pm16] == lab16, (
        "the graded postmitotic labels and the Boolean consensus labels diverged -- one arm "
        "broke (agreement is by construction while both are healthy)"
    )
    assert ph["regime"] == "graded" and ph["label"] == "anterior", "the sp prior label moved"
    # The declared prior surface carries what it says: the graded identity is anterior-high at
    # sp's node and the consensus state is the settled anterior pattern (read HERE so the two
    # fields are consumed and policed until T044+'s consumers land).
    assert ph["graded_identity"][0] == ph["graded_identity"][1] > 0.9, (
        "the graded identity at sp is not anterior-high -- the organizer field moved"
    )
    assert ph["consensus_state"] == (1, 1, 0, 0), (
        "the settled consensus state at sp is not the anterior pattern"
    )
    # The prior is NOT one constant along AP, and the sharp regime labels by lineage.
    pm = region_prior("medulla")
    assert pm["regime"] == "sharp" and pm["label"] == "r5", "sharp-regime lineage label moved"
    assert pm["consensus_label"] == "posterior" != ph["consensus_label"], (
        "the consensus label does not vary along the AP axis -- the prior degenerated to a constant"
    )
    print(
        "  green  T043: neuromere-keyed prior (zone carried, pinned not-load-bearing) -- "
        "H01 by name, identical-prior class disclosed, AP axis live"
    )

    # THE 007 COMPOSITION. C1.7 refusal AT THE CALL SITE, branch-DISCRIMINATED (the panel's
    # blocker: a bare `except ValueError` also caught the unknown-grower branch, so deleting
    # the safety gate left this green -- the message assert is what makes the red reachable).
    # Ordering is proven with an IN-SET safety name: only refusal-FIRST can refuse a name the
    # lookup would accept.
    for bad in ("arm_dof", "growing_arm_learner", "estop"):
        try:
            place_grown(bad, 1)
            raise AssertionError(f"a safety-named growth {bad!r} was placed instead of refused")
        except ValueError as e:
            assert "007 C1.7" in str(e), (
                f"{bad!r} was refused by the WRONG branch (unknown-grower, not the safety "
                "gate) -- the C1.7 red is unreachable"
            )
    try:
        place_grown("not_a_grower", 1)
        raise AssertionError("an unknown grower was placed instead of refused")
    except ValueError as e:
        assert "grower set" in str(e), "the unknown-grower branch lost its message"
    GROWER_REGIONS["estop_probe"] = "cortex"  # an IN-SET safety name: only refusal-FIRST reds
    try:
        place_grown("estop_probe", 1)
        raise AssertionError("refusal is not first -- an in-set safety name was placed")
    except ValueError as e:
        assert "007 C1.7" in str(e), "the in-set safety name was refused by the wrong branch"
    finally:
        del GROWER_REGIONS["estop_probe"]
    # T042's SCAFFOLD refusal is INHERITED (one admission rule, not two): a grower mapped to a
    # scaffold region is refused at this call site too, and the declared routing table itself
    # sits inside the owner-ratified growable partition.
    assert all(r in GROWABLE for r in GROWER_REGIONS.values()), (
        "a declared grower routes to a region outside growth_program.GROWABLE"
    )
    GROWER_REGIONS["scaffold_probe"] = "brainstem"
    try:
        place_grown("scaffold_probe", 1)
        raise AssertionError("a scaffold-region growth was placed -- T042's closure is bypassed")
    except ValueError as e:
        assert "SCAFFOLD" in str(e), "the scaffold refusal fired from the wrong branch"
    finally:
        del GROWER_REGIONS["scaffold_probe"]
    assert len(place_grown("function_preserving_growth", 2)) == 2, (
        "the positive control failed -- an in-set grower must place (else the refusal reds "
        "above prove only that everything is refused)"
    )
    print(
        "  RED    T043: C1.7 + SCAFFOLD refusal at the growth call site, branch-discriminated "
        "(safety / scaffold / unknown each by its own message; refusal-first proven in-set)"
    )

    # The coordinator seams: the first register_seam caller outside growable.py's own
    # self-check (NO live-tick caller -- the sleep-tick wire measured a NULL on 2026-09-02 and
    # was reverted, specs/022 evidence/s1-placement-null.md), the safety tripwire red THROUGH the
    # interface, and a growth event rippling into exactly the delta placements.
    coord, ledger = compose_growers(base_seed=0)
    assert coord.seams() == sorted(GROWER_REGIONS), "the grower set is not fully registered"
    try:
        coord.register_seam("motor_actions", 1)
        raise AssertionError("the 007 interface accepted a safety seam")
    except ValueError:
        pass
    from snn_data_lake.growth.function_preserving_growth import SpikingGrowthLayer

    layer = SpikingGrowthLayer(n_in=6, n_hidden=4, n_out=2, seed=2)
    wrng = random.Random(9)
    # amp 0.3, NOT 1.0: at amp 1.0 the readout saturates (37 spikes in 40 slots, measured) and
    # a NON-preserving growth with positive fan-out is undetectable (w_out=+50 read as
    # "preserved"). At amp 0.3 the rate is ~0.40 and a +1.0 perturbation flips the train --
    # the sensitivity control below proves that on every run.
    window = [[wrng.uniform(0.0, 0.3) for _ in range(6)] for _ in range(20)]
    _c0, out0, _h0 = layer.run(window)
    coord.grow("function_preserving_growth", layer.n_hidden)  # announce the initial 4
    gid = layer.add_unit_gradmax(window)
    coord.grow("function_preserving_growth", layer.n_hidden)  # the growth event: +1
    _c1, out1, _h1 = layer.run(window)
    assert out1 == out0, (
        "function_preserving_growth's zero-fanout insertion is not function-preserving -- "
        "the readout spike train moved"
    )
    # SENSITIVITY CONTROL (the criterion's own reachable red, run every time): a genuinely
    # non-preserving fan-out on the same window IS detected, so the assert above cannot be
    # saturation-blind without this control going red first.
    saved = layer.w_out[gid]
    layer.w_out[gid] = [1.0] * layer.n_out
    _c2, out2, _h2 = layer.run(window)
    layer.w_out[gid] = saved
    assert out2 != out0, (
        "the preservation assert is BLIND at this window -- a +1.0 fan-out perturbation did "
        "not move the readout, so 'preserved' means nothing here"
    )
    lg = ledger["function_preserving_growth"]
    assert len(lg) == layer.n_hidden == 5 and gid == 4, "the ripple did not place the delta"
    assert all(lab == "anterior" and s is None for _p, lab, s in lg), (
        "a rippled hippocampal placement lost its GRN label or acquired a fabricated sign"
    )
    # THE EVENT-SEED PROPERTIES, each policed (the panel measured all three mutants passing
    # before these asserts existed): replay identity; base_seed participation; per-grower
    # stream separation at equal widths; no constant seed (pairwise-distinct positions).
    coord2, ledger2 = compose_growers(base_seed=0)
    coord2.grow("function_preserving_growth", 4)
    coord2.grow("function_preserving_growth", 5)
    assert ledger2["function_preserving_growth"] == lg, (
        "a replayed growth sequence placed differently -- the event seed is not deterministic"
    )
    coord3, ledger3 = compose_growers(base_seed=1)
    coord3.grow("function_preserving_growth", 4)
    coord3.grow("function_preserving_growth", 5)
    assert ledger3["function_preserving_growth"] != lg, (
        "base_seed does not participate in the event seed -- two brains would share a stream"
    )
    coord4, ledger4 = compose_growers(base_seed=0)
    coord4.grow("function_preserving_growth", 3)
    coord4.grow("ng_snn", 3)  # synthetic seed-property probe, NOT fit-driven ng_snn traffic
    fpg3 = [p for p, _l, _s in ledger4["function_preserving_growth"]]
    ngs3 = [p for p, _l, _s in ledger4["ng_snn"]]
    assert fpg3 != ngs3, (
        "two growers at equal width drew the same stream -- crc32(name) is not separating them"
    )
    positions = [p for p, _l, _s in lg]
    assert len(set(positions)) == len(positions), (
        "duplicate positions in one ledger -- the event seed degenerated toward a constant"
    )
    assert ledger["ng_snn"] == [] and coord.width("ng_snn") == 0, (
        "ng_snn claimed traffic -- its fit-driven growth is the torch tier's, NOT claimed here"
    )
    print(
        "  green  T043: growers composed through 007 -- register_seam (first caller outside "
        f"growable's own self-check), growth ripples to {len(lg)} placements, preservation + "
        "its sensitivity control hold, seed properties policed"
    )

    # layer_histogram refuses a GRN label rather than absorbing it as a cortical layer.
    try:
        layer_histogram(h1)
        raise AssertionError("layer_histogram absorbed a GRN label as a cortical layer")
    except ValueError:
        pass
    print("  RED    T043: layer_histogram refuses non-cortical labels")

    print(
        f"self-check OK: {len(_LAYER_NAMES)} layers (H01, measured) + {len(REGION_COORDS) - 1} "
        "GRN-prior regions, Dale enforced at birth, the growers composed through 007"
    )


if __name__ == "__main__":
    # THE SELF-CHECK RUNS ON THE NO-ARG PATH TOO (T043 panel BLOCKER): run_selfchecks.py and
    # tests/test_selfchecks.py invoke this module with NO flag, so an argv-gated self-check is
    # a red nobody runs -- the exact shape the 007 contract's 2026-08-03 correction names. The
    # asserts execute in every gate run now (arealization's pattern); --self-check alone stays
    # supported for a quiet check without the report.
    _selfcheck()
    if "--self-check" not in sys.argv:
        sys.exit(main())
