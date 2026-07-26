"""
brain_geometry -- the 3D placement of EVERY snn_data_lake module, as pure data (no plotting).

WHAT THIS IS
    One function, `layout()`, that returns the whole brain as plain dicts: a coordinate for every
    module in the package (REGIONS + TORCH_TIER + the folded reasoning tier), a centre + centroid +
    module count per region, the region->region tracts, and the bounds of the space. Any renderer
    (a matplotlib figure, a web cockpit, a text dump) can consume it; this module draws nothing and
    imports nothing but the standard library plus the package's own region/connectome maps.

    It is the pure-stdlib extraction of the placement half of `integrations/chat_app/brain_activity_3d.py`
    (which additionally owns the matplotlib figure, the hull mesh, and the per-turn activity colouring).
    The literals, the placement rules and -- critically -- the RNG draw ORDER are carried over byte for
    byte, so both modules produce the same coordinates for the same package census.

THE HONESTY CONTRACT (read before trusting a coordinate)
    The coordinates are HAND-AUTHORED APPROXIMATE ANATOMY. They were typed by a human from a mental
    picture of a brain. They are:
      * NOT measured, and not traceable to any imaging study;
      * NOT MNI (or Talairach, or any registered stereotaxic frame) -- the units are dimensionless and
        the whole brain is normalised into roughly [-1, 1]^3;
      * NOT derived from the data, the connectome, or anything the brain computes. Nothing in the
        package produces a position; `REGIONS` / `TORCH_TIER` / `reasoning/*.py` supply only region
        MEMBERSHIP (which fixes each region's node COUNT, and hence its centroid), never a coordinate.
    They are good enough to make a brain-shaped picture legible, and they are worth nothing as anatomy.
    Treat the axes as an ordering (left/right, back/front, down/up), never as a measurement.

    The EDGES are the opposite: they are the REAL typed connectome. `edges` is derived from
    `snn_data_lake.connectome` -- each typed, directed, signed tract's endpoint nodes mapped to their
    region, self-edges and the region-less effector nodes dropped. There is no fallback tract list: if
    the connectome cannot be read this module raises, because a plausible-looking invented tract set is
    exactly the lie this contract exists to prevent. The ONE documented exception is
    `_CURATED_INFRA_EDGES` -- two hand-added wires for `substrate` and `plasticity`, which have no
    connectome node at all. They are listed separately below and are labelled curated, not measured.

    Three regions are not anatomy and say so in their `note` (`_LABEL_NOTE`): `substrate` is the spike
    data-lake / telemetry / infra tier (drawn as a separate cloud BELOW the brain), `plasticity` is not
    a nucleus (learning rules live distributed at synapses), and `reasoning` is the folded
    research/cognition tier, not a wired connectome region.

DETERMINISM (the draw order is load-bearing)
    A single `random.Random(1605)` draws every jitter, consumed while iterating the module census in
    dict order: REGIONS insertion order (with each region's TORCH_TIER modules appended to its list),
    then "reasoning" = the sorted glob of `reasoning/*.py`. Any change to module MEMBERSHIP -- adding,
    renaming or removing a module -- reshuffles every position drawn after it. That is by design (the
    layout is procedural, not stored), but it means coordinates are stable only for a fixed census.
    `main()` pins the first draw against a recorded constant so a silent seed/order change fails loudly.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.brain_geometry
"""

import glob
import math
import os
import random

SEED = 1605  # fixed -> deterministic jitter (a per-instance random.Random, never the global RNG)

# ------------------------------------------------------------------------------------------------
# region -> HAND-AUTHORED anatomical centre (x=L-/R+, y=post-/ant+, z=inf-/sup+). Regions in
# BILATERAL are paired: their modules are split L/R at x = +/- |x| (the x here is the magnitude).
# Approximate by construction -- see the honesty contract in the module docstring.
# ------------------------------------------------------------------------------------------------
REGION_CENTER = {
    "cortex": (0.00, 0.10, 0.55),  # whole cortical mantle -- placed over the outer upper hull
    "hippocampus": (0.35, -0.05, -0.30),  # medial temporal, bilateral
    "amygdala": (0.30, 0.20, -0.30),  # anterior to hippocampus, bilateral
    "basal_ganglia": (0.25, 0.10, 0.10),  # deep, lateral to thalamus, bilateral
    "thalamus": (0.00, -0.05, 0.05),  # central deep relay + GWT hub
    "hypothalamus": (0.00, 0.05, -0.18),  # below thalamus, anterior-inferior
    "epithalamus": (0.00, -0.22, 0.12),  # posterior-superior to thalamus (habenula/pineal)
    "midbrain": (0.00, -0.10, -0.20),  # superior brainstem
    "pons": (0.00, -0.15, -0.45),  # mid brainstem
    "medulla": (0.00, -0.15, -0.72),  # inferior brainstem (lowest)
    "brainstem": (0.00, -0.12, -0.45),  # the reflex/arm-control column
    "cerebellum": (0.00, -0.70, -0.45),  # posterior + inferior (behind + below)
    "olfactory": (0.00, 0.60, -0.42),  # anterior-inferior frontal base
    "claustrum": (0.45, 0.10, 0.05),  # lateral (insula/putamen border), bilateral
    "neuromodulation": (0.00, -0.10, -0.25),  # diffuse nuclei -- real sub-sources below
    "plasticity": (0.00, 0.00, 0.25),  # distributed (scattered across cortex + hippocampus)
    "substrate": (0.00, -0.35, -1.05),  # NON-anatomical infra cloud, BELOW the brain
    "reasoning": (0.00, 0.70, 0.40),  # folded research/cognition cloud, anterior (prefrontal)
}

BILATERAL = {"hippocampus", "amygdala", "basal_ganglia", "claustrum"}

# per-region jitter radius (scalar = isotropic; a 3-tuple = per-axis, for elongated structures)
_RADIUS = {
    "thalamus": 0.07,
    "hypothalamus": 0.06,
    "epithalamus": 0.05,
    "midbrain": 0.07,
    "pons": 0.06,
    "medulla": 0.06,
    "brainstem": (0.06, 0.08, 0.20),  # a vertical reflex/arm-control column
    "cerebellum": (0.30, 0.22, 0.16),  # a wide posterior-inferior lobe
    "olfactory": 0.07,
    "hippocampus": 0.11,
    "amygdala": 0.08,
    "basal_ganglia": 0.10,
    "claustrum": 0.06,
    "substrate": 0.22,
    "reasoning": (0.26, 0.20, 0.20),
}
_DEFAULT_RADIUS = 0.08

# the cortical modules whose function is prefrontal / executive / language / reasoning-cognition
# -- these bias ANTERIOR (+y) into the frontal lobe rather than spreading over the whole mantle.
COGNITIVE_CORTEX = {
    "language_brain",
    "nl_interface",
    "brain_reasoning",
    "spiking_lm_serve",
    "vsa_reasoning",
    "prefrontal_executive",
    "live_brain",
    "interaction_learning",
    "access_consciousness",
    "self_model",
    "metacognitive_monitor",
    "theory_of_mind",
    "reasoning_curriculum",
    "autotelic_goals",
    "quality_diversity",
    "self_improvement_suite",
    "wholistic_loop",
    "integration_measure",
    "agency_monitor",
}

# neuromodulation: place modules at their REAL diffuse source nuclei (locus_coeruleus handled
# separately because it is a paired/bilateral pontine nucleus). Anything not listed is a
# drive / motivation module -> clustered near the VTA/SNc dopamine source.
NEUROMOD_SOURCES = {
    "raphe_serotonin": (0.00, -0.15, -0.40),  # midline raphe (serotonin)
    "acetylcholine": (0.00, 0.30, -0.25),  # basal forebrain (ACh)
    "cortisol": (0.00, 0.08, -0.22),  # HPA / hypothalamic border (stress hormone)
    "astrocyte_modulation": (0.00, 0.00, 0.10),  # diffuse glia (large radius below)
}
NEUROMOD_DEFAULT = (0.00, -0.10, -0.20)  # VTA/SNc midbrain dopamine source

# honest labels for the non-anatomical / distributed tiers (a renderer MUST show these next to the
# region name -- they are the standing correction to "everything on this picture is a brain part").
_LABEL_NOTE = {
    "substrate": "data lake / infra (non-anatomical)",
    "plasticity": "distributed learning rules (not a nucleus)",
    "reasoning": "folded research / cognition tier",
}

# the space -- every module coordinate MUST fall inside this box (asserted in the self-check).
XLIM = (-1.05, 1.05)
YLIM = (-1.25, 1.15)
ZLIM = (-1.40, 0.90)

# The ONLY two edges that are NOT from the connectome. `substrate` and `plasticity` have no
# connectome node (one is infra, the other is distributed), so they would otherwise float unwired.
# These are CURATED by hand and must never be presented as measured tracts.
_CURATED_INFRA_EDGES = (
    ("hippocampus", "substrate"),  # curated: the lake working memory
    ("cortex", "plasticity"),  # curated: learning rules on cortex
)

_LAYOUT = None  # the cached layout() result (built once; treat the returned dict as read-only)


def _reasoning_modules():
    """The folded research/cognition tier: every module directly under snn_data_lake/reasoning/
    (excluding the package __init__ and any subpackage). Sorted, so the census is deterministic."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = []
    for p in glob.glob(os.path.join(pkg_dir, "reasoning", "*.py")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem != "__init__":
            names.append(stem)
    return sorted(names)


def _region_modules():
    """region -> [modules], covering the FULL package: REGIONS + the torch tier + the reasoning tier.
    The ITERATION ORDER of the returned dict is the RNG draw order -- do not sort or restructure it."""
    from snn_data_lake import REGIONS, TORCH_TIER  # lazy: keeps this module import-cycle free

    rm = {r: list(mods) for r, mods in REGIONS.items()}
    for r, mods in TORCH_TIER.items():  # torch-gated organs append to their own region's list
        rm.setdefault(r, []).extend(mods)
    rm["reasoning"] = _reasoning_modules()
    return rm


def _jitter(base, rad, rng):
    """A bounded uniform offset around `base`; `rad` is a scalar (isotropic) or a per-axis 3-tuple."""
    rx, ry, rz = (rad, rad, rad) if not isinstance(rad, (tuple, list)) else rad
    return (
        base[0] + rng.uniform(-rx, rx),
        base[1] + rng.uniform(-ry, ry),
        base[2] + rng.uniform(-rz, rz),
    )


def _cortex_point(module, rng):
    """A point on the OUTER UPPER hull (the cortical mantle). Cognitive/prefrontal modules bias
    anterior (+y) into the frontal lobe; the rest spread over the whole upper mantle."""
    ax, by, cz = 0.66, 0.86, 0.56  # inset half-axes -> nodes sit just inside the hull
    if module in COGNITIVE_CORTEX:
        theta = rng.uniform(math.pi * 0.30, math.pi * 0.70)  # anterior band (+y)
        phi = rng.uniform(0.35, 1.05)  # upper-front
        by = 0.90  # push a touch more anterior
    else:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(0.15, 1.50)  # whole upper mantle down to the sides
    st = math.sin(phi)
    x = ax * st * math.cos(theta) + rng.uniform(-0.03, 0.03)
    y = by * st * math.sin(theta) + rng.uniform(-0.03, 0.03)
    z = cz * math.cos(phi) + 0.12 + rng.uniform(-0.03, 0.03)  # +0.12 -> sit on the upper hull
    return (x, y, z)


def _neuromod_point(module, rng):
    """Place a neuromodulatory module at its real diffuse source nucleus."""
    if module == "locus_coeruleus":  # paired pontine LC (bilateral)
        side = 1.0 if rng.random() < 0.5 else -1.0
        return _jitter((0.10 * side, -0.15, -0.45), 0.05, rng)
    if module in NEUROMOD_SOURCES:
        rad = 0.18 if module == "astrocyte_modulation" else 0.05
        return _jitter(NEUROMOD_SOURCES[module], rad, rng)
    return _jitter(NEUROMOD_DEFAULT, 0.12, rng)  # drive/motivation -> near VTA/SNc


def _plasticity_point(rng):
    """Scatter faintly across cortex + hippocampus -- learning rules are distributed at synapses,
    NOT a discrete locus (hence the standing honesty note on this region)."""
    if rng.random() < 0.6:  # cortical synapses (upper mantle)
        return (rng.uniform(-0.55, 0.55), rng.uniform(-0.20, 0.50), rng.uniform(0.15, 0.55))
    side = 1.0 if rng.random() < 0.5 else -1.0  # hippocampal synapses (medial temporal)
    return _jitter((0.35 * side, -0.05, -0.30), 0.10, rng)


def _place(region, module, index, rng):
    """The position of one module: a per-region cluster with deterministic jitter. Three regions do
    NOT use their REGION_CENTER literal at all (cortex rides the hull, neuromodulation goes to its
    real source nuclei, plasticity scatters) -- read a position from the module, not from the centre."""
    if region == "cortex":
        return _cortex_point(module, rng)
    if region == "neuromodulation":
        return _neuromod_point(module, rng)
    if region == "plasticity":
        return _plasticity_point(rng)
    cx, cy, cz = REGION_CENTER[region]
    rad = _RADIUS.get(region, _DEFAULT_RADIUS)
    if region in BILATERAL:  # alternate L/R hemispheres by index
        side = 1.0 if index % 2 == 0 else -1.0
        return _jitter((side * cx, cy, cz), rad, rng)
    return _jitter((cx, cy, cz), rad, rng)


def _build_positions():
    """('region/module' -> (x,y,z), 'region/module' -> region) for EVERY module in the package.
    Deterministic: ONE per-instance random.Random(SEED) draws every jitter, in census order."""
    rng = random.Random(SEED)
    pos, region_of = {}, {}
    for region, mods in _region_modules().items():
        for i, m in enumerate(mods):
            key = f"{region}/{m}"
            pos[key] = _place(region, m, i, rng)
            region_of[key] = region
    return pos, region_of


def _connectome_pairs():
    """The REAL region->region tracts, and ONLY those: every typed edge of `snn_data_lake.connectome`
    with both endpoint nodes mapped to their region, the region-less effector nodes and the self-edges
    dropped. The curated infra edges are deliberately NOT mixed in here -- `layout()` unions them in
    separately so the two provenances stay countable. Raises if the connectome is unreadable; there is
    no invented fallback tract set (see the honesty contract)."""
    from snn_data_lake import REGIONS, connectome

    regset = set(REGIONS)  # the anatomical regions; 'reasoning' has no connectome node
    pairs = set()
    for e in connectome.EDGES:
        rs = connectome.NODES[e["source"]][0]
        rt = connectome.NODES[e["target"]][0]
        if rs in regset and rt in regset and rs != rt:
            pairs.add(tuple(sorted((rs, rt))))
    return pairs


def layout():
    """The whole brain as plain data. Built once and CACHED, so repeated calls are cheap and return
    the identical object -- treat the result as read-only. Shape:

        {"regions": [{"name", "x", "y", "z", "n_modules", "note"}],   # all 18 authored regions
         "modules": [{"key", "region", "module", "x", "y", "z"}],     # every module in the package
         "edges":   [[regionA, regionB], ...],                        # drawable tracts
         "space":   {bounds + frame + seed + provenance}}

    A region's `x/y/z` is its CENTROID (the mean of its members' positions) -- the label anchor and
    tract endpoint, which is what a renderer needs. It is NOT the declared `REGION_CENTER` literal, and
    the two differ, sometimes a lot: a bilateral region's members alternate L/R so its centroid
    collapses toward x~0, and cortex/neuromodulation/plasticity never use their declared centre at all.
    The one exception is a region with `n_modules == 0`, which has no centroid and falls back to the
    declared centre; `REGION_CENTER` remains public if you want the authored literal directly.

    `edges` are only the tracts whose BOTH endpoints have a centroid, i.e. both regions currently hold
    at least one module -- a region with zero modules cannot be drawn, so its tracts are not drawable.
    The dropped ones are not hidden: they are listed in `space["edges_skipped_no_centroid"]`. The two
    provenances stay countable in `space`: `n_edges_from_connectome` (real typed tracts) vs
    `n_edges_curated` (the hand-added infra wires). A renderer must not present the second as the first."""
    global _LAYOUT
    if _LAYOUT is not None:
        return _LAYOUT

    pos, region_of = _build_positions()

    counts, sums = {}, {}
    for key, region in region_of.items():
        counts[region] = counts.get(region, 0) + 1
        sx, sy, sz = sums.get(region, (0.0, 0.0, 0.0))
        px, py, pz = pos[key]
        sums[region] = (sx + px, sy + py, sz + pz)
    centroid = {
        r: (sx / counts[r], sy / counts[r], sz / counts[r]) for r, (sx, sy, sz) in sums.items()
    }

    tracts = _connectome_pairs()  # the real connectome, and nothing else
    curated = {tuple(sorted(p)) for p in _CURATED_INFRA_EDGES} - tracts  # genuinely hand-added only
    all_pairs = sorted(tracts | curated)
    drawable = [[a, b] for a, b in all_pairs if a in centroid and b in centroid]
    skipped = [[a, b] for a, b in all_pairs if a not in centroid or b not in centroid]
    n_curated_drawn = sum(1 for e in drawable if tuple(e) in curated)

    regions = []
    for name, declared in REGION_CENTER.items():
        # The region's render anchor is its CENTROID -- the mean of its members' positions -- because
        # that is what the label and the tract endpoints attach to, and it is often far from the
        # declared centre (a bilateral region's members alternate L/R so its centroid collapses toward
        # x~0; cortex/neuromodulation/plasticity never use their declared centre to place anything).
        # It falls back to the declared hand-authored centre in exactly one case: n_modules == 0, an
        # empty region that has no centroid but still needs a position. REGION_CENTER stays public for
        # anyone who wants the authored literal itself.
        ax, ay, az = centroid.get(name, declared)
        regions.append(
            {
                "name": name,
                "x": ax,
                "y": ay,
                "z": az,
                "n_modules": counts.get(name, 0),
                "note": _LABEL_NOTE.get(name),
            }
        )
    modules = [
        {
            "key": key,
            "region": region_of[key],
            "module": key.split("/", 1)[1],
            "x": x,
            "y": y,
            "z": z,
        }
        for key, (x, y, z) in pos.items()
    ]

    _LAYOUT = {
        "regions": regions,
        "modules": modules,
        "edges": drawable,
        "space": {
            "kind": "hand-authored approximate anatomy (NOT measured, NOT MNI, NOT derived from data)",
            "axes": "x = left(-)/right(+), y = posterior(-)/anterior(+), z = inferior(-)/superior(+)",
            "units": "dimensionless; the brain fits roughly [-1, 1]^3",
            "xlim": list(XLIM),
            "ylim": list(YLIM),
            "zlim": list(ZLIM),
            "seed": SEED,
            "n_modules": len(modules),
            "edges_source": (
                "snn_data_lake.connectome -- real typed tracts, endpoints mapped to region; plus "
                f"{len(curated)} CURATED (hand-added, NOT measured) infra edges: "
                + ", ".join(f"{a}<->{b}" for a, b in sorted(curated))
            ),
            "n_edges_from_connectome": len(drawable) - n_curated_drawn,
            "n_edges_curated": n_curated_drawn,
            "edges_skipped_no_centroid": skipped,  # real tracts we cannot draw (region has 0 modules)
        },
    }
    return _LAYOUT


# ================================================================================================
# The first module drawn, pinned. This is the SEED CONTRACT: it fails if the seed, the placement
# rules, or the head of the module census changes -- any of which silently moves every coordinate.
_SEED_PIN_KEY = "cortex/rank_order_coding"
_SEED_PIN_XYZ = (0.2656447451741262, 0.44153862227162194, 0.5522876185510223)


def main():
    lay = layout()
    mods, regions, edges = lay["modules"], lay["regions"], lay["edges"]

    # (a) the census must cover EVERY module: REGIONS + TORCH_TIER + the reasoning tier.
    from snn_data_lake import REGIONS, TORCH_TIER

    expected = (
        sum(len(m) for m in REGIONS.values())
        + sum(len(m) for m in TORCH_TIER.values())
        + len(_reasoning_modules())
    )
    assert len(mods) == expected, f"layout must place every module ({len(mods)} != {expected})"
    assert len(mods) >= 200, f"the package has >= 200 modules; layout placed only {len(mods)}"
    assert len(regions) == len(REGION_CENTER), f"every authored region must appear: {len(regions)}"

    # (b) every coordinate is finite and inside the stated bounds.
    for m in mods:
        x, y, z = m["x"], m["y"], m["z"]
        assert all(math.isfinite(v) for v in (x, y, z)), f"non-finite position for {m['key']}"
        assert XLIM[0] <= x <= XLIM[1] and YLIM[0] <= y <= YLIM[1] and ZLIM[0] <= z <= ZLIM[1], (
            f"module {m['key']} at {(x, y, z)} is outside the stated bounds"
        )

    # (c) every region anchor is finite, in-bounds, and really is the centroid of its members.
    by_region = {}
    for m in mods:
        by_region.setdefault(m["region"], []).append(m["x"])
    for r in regions:
        ax, ay, az = r["x"], r["y"], r["z"]
        assert all(math.isfinite(v) for v in (ax, ay, az)), (
            f"non-finite anchor for region {r['name']}"
        )
        assert XLIM[0] <= ax <= XLIM[1] and YLIM[0] <= ay <= YLIM[1] and ZLIM[0] <= az <= ZLIM[1], (
            f"region {r['name']} anchor {(ax, ay, az)} is outside the stated bounds"
        )
        members = [m for m in mods if m["region"] == r["name"]]
        assert len(members) == r["n_modules"], f"n_modules disagrees with members for {r['name']}"
        if members:  # the anchor must BE the centroid, not the declared centre
            for axis, got in zip(("x", "y", "z"), (ax, ay, az)):
                want = sum(m[axis] for m in members) / len(members)
                assert abs(got - want) < 1e-12, (
                    f"region {r['name']} anchor.{axis} is not the centroid"
                )
        else:  # only an EMPTY region may fall back to the declared hand-authored centre
            assert (ax, ay, az) == REGION_CENTER[r["name"]], (
                f"empty region {r['name']} must fall back"
            )
    for r in sorted(BILATERAL):
        xs = by_region.get(r, [])
        if len(xs) < 2:
            continue  # a 1-module bilateral region can only occupy one side
        assert any(v < 0 for v in xs) and any(v > 0 for v in xs), (
            f"bilateral region {r} must place members on BOTH sides, got x={xs}"
        )

    # (d) every edge endpoint is a known region, and the tracts are the real connectome.
    known = set(REGION_CENTER)
    for a, b in edges:
        assert a in known and b in known, f"edge {(a, b)} references an unknown region"
        assert a != b, f"self-edge {(a, b)} must have been dropped"
    n_real = lay["space"]["n_edges_from_connectome"]
    assert n_real >= 10, f"the REAL connectome must supply the major tracts (got {n_real})"
    assert lay["space"]["n_edges_curated"] == len(_CURATED_INFRA_EDGES), (
        "the curated infra edges must stay countable and separate from the connectome tracts"
    )

    # (e) determinism: the cache returns the identical object, AND a fresh build reproduces it.
    assert layout() is lay, "layout() must cache -- repeated calls return the identical object"
    fresh_pos, _ = _build_positions()
    for m in mods:
        fx, fy, fz = fresh_pos[m["key"]]
        assert (fx, fy, fz) == (m["x"], m["y"], m["z"]), f"non-deterministic placement: {m['key']}"

    # (f) the seed contract: a fresh process must reproduce the recorded first draw exactly.
    first = mods[0]
    assert first["key"] == _SEED_PIN_KEY, (
        f"the census head moved ({first['key']} != {_SEED_PIN_KEY}); every coordinate drawn after it "
        "has shifted -- re-pin _SEED_PIN_* deliberately, do not 'fix' it by accident"
    )
    for got, want in zip((first["x"], first["y"], first["z"]), _SEED_PIN_XYZ):
        assert abs(got - want) < 1e-12, (
            f"seed contract broken: first draw {(first['x'], first['y'], first['z'])} "
            f"!= pinned {_SEED_PIN_XYZ}"
        )

    n_note = sum(1 for r in regions if r["note"])
    empty = [r["name"] for r in regions if r["n_modules"] == 0]
    print("brain_geometry self-check OK:")
    print(
        f"  placed {len(mods)} modules across {len(regions)} regions "
        f"(REGIONS+TORCH_TIER+reasoning census = {expected})"
    )
    print(
        f"  coordinates are HAND-AUTHORED approximate anatomy (seed {SEED}); "
        "not measured, not MNI, not derived from data"
    )
    print(
        f"  edges: {len(edges)} drawable = {lay['space']['n_edges_from_connectome']} REAL connectome "
        f"tracts + {lay['space']['n_edges_curated']} CURATED infra edges"
    )
    print(
        f"  {len(lay['space']['edges_skipped_no_centroid'])} real tracts are not drawable and are "
        f"reported, not hidden (endpoint region has 0 modules: {empty})"
    )
    print(f"  {n_note} regions carry an honesty note (non-anatomical / distributed tiers)")
    print(f"  bounds x={XLIM} y={YLIM} z={ZLIM} -- every module in-bounds and finite")
    print(f"  seed contract pinned at {_SEED_PIN_KEY} = {_SEED_PIN_XYZ}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
