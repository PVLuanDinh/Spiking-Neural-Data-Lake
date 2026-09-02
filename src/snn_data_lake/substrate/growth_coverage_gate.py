"""
Growth-coverage gate -- specs/022 T046: the four instruments invention register row 7 owes,
mechanized and COUNTED on day one (the gate-before-rule invariant).

LOCATION: substrate/, NOT tools/ -- the T046 panel measured the tools/ draft REDDING the p4c
ratchet in the very CI job it wired itself into (the lint job dies at the ratchet step before
the gate's own steps run). Every future build lands governed from the start; the T022
region_charge_gate precedent already chose this zone. tasks.md's "in tools/" loses to P4c's
own gate, recorded here rather than smoothed.

WHAT IT ASSERTS (--assert == bare), and what it only REPORTS:

1. COVERAGE RED (ratchet, identity-pinned) + THE PARTITION APPLIED. regions-grown-LIVE over
   the owner-ratified GROWABLE partition (cortex/growth_program.py, 10 regions), measured by
   an import-RESOLVED AST census with per-site attribution. DAY-ONE COUNT: 1 of 10 (cortex,
   via neurogenesis_engine) -- BELOW FLOOR, said in bold every run. Asserted three ways:
   (a) ratchet: the identity-pinned baseline set {cortex} must stay contained (the set may
   only grow; when the measured set EXCEEDS the pin, a TIGHTEN-OWED line prints -- the pin
   is bumped by hand in a reviewed commit, never silently); (b) REFUSAL: any censused growth
   into SCAFFOLD (brainstem included -- the one most consequential catch), into
   CHARGED-NEVER-GROWN, or into a name outside REGION_COORDS reds the gate (the panel
   constructed a live brainstem placement PRINTING without failing in the first draft);
   (c) UNRESOLVED reds: a call whose region/grower is not a literal cannot be attributed,
   and the first draft silently attributed it to CORTEX -- fabricating the exact data point
   the ratchet pins; such sites now land in an <unresolved> bucket that reds the gate.

2. THREE-WAY PARTITION EXACT. GROWABLE (10) + SCAFFOLD (5) + {neuromodulation} ==
   REGION_COORDS keys exactly, pairwise disjoint. neuromodulation is dispositioned
   CHARGED-NEVER-GROWN -- live in the brain, in NEITHER set: not growable (no growth story
   for diffuse neuromodulatory nuclei at this grain) and not scaffold (it is not part of the
   fixed safety envelope's anatomy).

3. BOUNDARY SHARPNESS -- THE PRODUCTION LABELERS, MODAL LABELS, control in the same command.
   Both arms now run the REAL arealization symbols (the panel poisoned discretize_sharp with
   a raiser and the tools/ draft stayed GREEN -- it was policing an inline copy): the sharp
   arm calls arealization.discretize_sharp (n_comp DERIVED: the count of sharp-regime
   neuromeres), the graded arm perturbs graded_profiles' own countergradient layout and
   labels through arealization.postmitotic_discretize. Per-position MODAL labels across
   >= 7 seeds (the mode is computed, not implied -- the first draft said "modal" and
   counted distinct labels); width = positions where seeds are not unanimous, ALSO reported
   as a fraction (the count scales with resolution; the fraction is the physics). WHAT IS
   ASSERTED: the sharp arm's width is 0 BY CONSTRUCTION (lineage: discretize_sharp's
   signature takes positions only -- stated as construction, never sold as measurement) and
   the graded arm's width is >0 under noise and 0 without it (monotone, the reachable red);
   the REACHABLE regression red for the sharp arm is the production-symbol dependency
   itself, which the first draft did not have.

4. SCRAMBLED-FIELD SPATIAL NULL. The real profiles emit a CONTIGUOUS postmitotic partition
   (exactly 1 boundary -- CLOSED FORM: the organizer field is strictly monotone, so any
   threshold crosses once; the single edit that reds this is a non-monotone OrganizerField);
   spatially permuting the PROFILES (the field itself, not the labels -- the first draft
   permuted labels and was right only by the labeler's elementwise accident) fragments it
   (expectation 2k(n-k)/n = 64 at n=128, measured 51..75 over 50 seeds; bar 10 = ~6x below
   expectation, a declared slack bound).

5. ALLEN/PROSOMERIC EXTERNAL CONTROL, ELEMENTWISE + ARI. Our REGION_COORDS anatomy
   macro-grained through arealization.MACRO_OF_NEUROMERE (anatomy lives WITH the anatomy
   module -- the gate keeps only the EXTERNAL side) vs the Allen Developing Mouse Brain
   Atlas assignment typed below with provenance. Asserted ELEMENTWISE per region -- the
   panel proved ARI alone name-blind (midbrain is a singleton cluster, so relabeling it is
   partition-invariant: 1 of 48 single-region moves was invisible, and a wholesale
   forebrain<->hindbrain swap scores ARI 1.0). ARI + its 200-shuffle permutation null stay
   as the printed discrimination display. HONESTY: the comparison's OUR side is
   MACRO_OF_NEUROMERE, typed by the same author from the same textbooks -- what is external
   is the Allen table, and what is checked is the CONSISTENCY of our declared anatomy with
   it, not an independent measurement of the brain.

CENSUS SCOPE, DISCLOSED: src/ excluding reasoning/ (P4c counts it as brain, but it is
outside the three P3 gates and ruff; zero reasoning/ modules import placement today, so the
exclusion changes no number -- printed every run so it is never silent), excluding
placement.py itself, __main__ guards (either comparison order), and module-level harness
functions (main/self_check family, sync or async -- class methods excluded from the filter).
KNOWN BLIND SPOTS, measured absent today and stated: `import ...placement as X; X.place()`
module-attribute spellings are counted only when the alias is import-tracked;
`from ... import *` is invisible. Undercount reds LOUDLY (a missed region keeps the
coverage red redder); the fabrication direction is what the <unresolved> red closes.

  python -m snn_data_lake.substrate.growth_coverage_gate            # report (bare == assert)
  python -m snn_data_lake.substrate.growth_coverage_gate --assert   # CI form
  python -m snn_data_lake.substrate.growth_coverage_gate --self-check
"""

import ast
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from snn_data_lake.cortex.arealization import (
    MACRO_OF_NEUROMERE,
    NEUROMERES,
    REGION_COORDS,
    discretize_sharp,
    graded_profiles,
    postmitotic_discretize,
    regime_of,
)
from snn_data_lake.cortex.growth_program import GROWABLE, SCAFFOLD

REPO = Path(__file__).resolve().parent.parent.parent.parent

#: The identity-pinned coverage baseline: the regions grown from a LIVE path TODAY. A set,
#: not a count (a count survives a member swap); the ratchet allows growth only, and a
#: measured set EXCEEDING this pin prints TIGHTEN OWED rather than silently passing.
BASELINE_REGIONS_GROWN = frozenset({"cortex"})

#: CHARGED-NEVER-GROWN: live in the brain, in neither GROWABLE nor SCAFFOLD (disposition in
#: the module docstring). The three-way partition assert is what keeps this list from
#: silently absorbing a region someone forgot to classify.
CHARGED_NEVER_GROWN = frozenset({"neuromodulation"})

#: EXTERNAL REFERENCE (out-of-model): Allen Developing Mouse Brain Atlas, top-level ontology
#: (2013 release, textbook grain -- forebrain / midbrain / hindbrain assignment of gross
#: regions; the atlas's finer prosomeric segments are below this gate's grain). Typed in as
#: reference data with provenance, the cortical_prior pattern. This table is the EXTERNAL
#: side; the OUR side (arealization.MACRO_OF_NEUROMERE) deliberately lives with the anatomy
#: it describes, and the module docstring names the one-author limitation plainly.
ALLEN_MACRO = {
    "cortex": "forebrain",
    "hippocampus": "forebrain",
    "insula": "forebrain",
    "claustrum": "forebrain",
    "amygdala": "forebrain",
    "basal_ganglia": "forebrain",
    "olfactory": "forebrain",
    "hypothalamus": "forebrain",
    "epithalamus": "forebrain",
    "thalamus": "forebrain",
    "midbrain": "midbrain",
    "cerebellum": "hindbrain",
    "pons": "hindbrain",
    "medulla": "hindbrain",
    "brainstem": "hindbrain",
    "neuromodulation": "hindbrain",  # LC/raphe: rostral hindbrain tegmentum
}

#: Harness functions by declared convention (module level only; a CLASS METHOD named main is
#: real code and is NOT filtered): calls inside these are gate/self-check drives, never live
#: paths. Sync and async both (the panel constructed an async harness inflating coverage).
_HARNESS_FNS = {"main", "_main", "_selfcheck", "self_check", "_self_check"}

#: The census's placement-module spellings: exact, never a bare suffix (the panel constructed
#: `from mypkg.displacement import place` being COUNTED under endswith("placement")).
_PLACEMENT_MODULES = {"snn_data_lake.cortex.placement", "cortex.placement", "placement"}

UNRESOLVED = "<unresolved>"


def _is_placement_module(mod):
    return mod is not None and (mod in _PLACEMENT_MODULES or mod.endswith(".placement"))


def live_growth_census(src_root=None):
    """{region_or_<unresolved>: [call sites]} for every call that RESOLVES to
    placement.place / place_grown on a live path. Resolution, not name-matching, and
    fabrication-proof: a non-literal region/grower lands under <unresolved> (which reds the
    gate) -- the first draft defaulted it to CORTEX, manufacturing the pinned baseline's own
    data point. Scope and blind spots: module docstring."""
    root = Path(src_root) if src_root else REPO / "src"
    from snn_data_lake.cortex.placement import GROWER_REGIONS

    sites = {}

    def add(region, rel, lineno):
        sites.setdefault(region, []).append(f"{rel}:{lineno}")

    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root).as_posix()
        if rel.endswith("cortex/placement.py") or "/reasoning/" in f"/{rel}":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        imported = {}  # local name -> "place" | "place_grown"
        mod_aliases = (
            set()
        )  # names bound to the placement MODULE (import ... as / from cortex import placement)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if _is_placement_module(node.module):
                    for alias in node.names:
                        if alias.name in ("place", "place_grown"):
                            imported[alias.asname or alias.name] = alias.name
                elif node.module and node.module.endswith("cortex"):
                    for alias in node.names:
                        if alias.name == "placement":
                            mod_aliases.add(alias.asname or "placement")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_placement_module(alias.name):
                        mod_aliases.add(alias.asname or alias.name.split(".")[0])
        if not imported and not mod_aliases:
            continue
        skip_lines = set()
        for node in ast.walk(tree):
            is_main_guard = (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and (
                    (isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
                    or any(
                        isinstance(c, ast.Name) and c.id == "__name__"
                        for c in node.test.comparators
                    )
                )
            )
            if is_main_guard:
                for sub in ast.walk(node):
                    if hasattr(sub, "lineno"):
                        skip_lines.add(sub.lineno)
        for node in tree.body:  # harness filter: MODULE-LEVEL defs only
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in _HARNESS_FNS
            ):
                for sub in ast.walk(node):
                    if hasattr(sub, "lineno"):
                        skip_lines.add(sub.lineno)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or getattr(node, "lineno", None) in skip_lines:
                continue
            fn = node.func
            target = None
            if isinstance(fn, ast.Name):
                target = imported.get(fn.id)
            elif (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id in mod_aliases
                and fn.attr in ("place", "place_grown")
            ):
                target = fn.attr
            if target == "place":
                region_node = None
                for kw in node.keywords:
                    if kw.arg == "region":
                        region_node = kw.value
                    elif kw.arg is None:  # **kwargs: unattributable
                        region_node = ast.Name(id="_dynamic")
                if region_node is None and len(node.args) >= 3:
                    region_node = node.args[2]
                if region_node is None:
                    add("cortex", rel, node.lineno)  # the signature's default
                elif isinstance(region_node, ast.Constant):
                    add(region_node.value, rel, node.lineno)
                else:
                    add(UNRESOLVED, rel, node.lineno)
            elif target == "place_grown":
                grower_node = node.args[0] if node.args else None
                for kw in node.keywords:
                    if kw.arg == "grower":
                        grower_node = kw.value
                if isinstance(grower_node, ast.Constant):
                    region = GROWER_REGIONS.get(grower_node.value)
                    add(region if region else UNRESOLVED, rel, node.lineno)
                else:
                    add(UNRESOLVED, rel, node.lineno)
    return sites


def boundary_widths(n_seeds=7, n_pos=128, noise=0.03):
    """(sharp_width, graded_width, graded_fraction) via the PRODUCTION labelers and MODAL
    labels. Sharp: arealization.discretize_sharp over n_comp = the model's own count of
    sharp-regime neuromeres -- width 0 BY CONSTRUCTION (positions only; the reachable red is
    the production-symbol dependency: poison discretize_sharp and this gate dies). Graded:
    graded_profiles' countergradient layout perturbed by seeded field noise (a' = a + e
    means the profile moves by e * [1, 1, -1, -1] -- the layout identity, stated), labeled
    by arealization.postmitotic_discretize; width = positions where the >= 7 seeds are not
    UNANIMOUS around the computed modal label. noise=0.03: ~1.5x the postmitotic threshold
    band a 3%-of-range jitter crosses near the midline; the assert needs only >0 and
    0-at-zero-noise (monotone, checked in --self-check at 0.10)."""
    n_comp = sum(1 for nm in NEUROMERES if regime_of(nm) == "sharp")
    xs = (np.arange(n_pos) + 0.5) / n_pos
    prof_clean = graded_profiles(n_pos)
    layout = np.array([1.0, 1.0, -1.0, -1.0])
    sharp_runs, graded_runs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        e = rng.normal(0.0, noise, n_pos)
        sharp_runs.append(np.asarray(discretize_sharp(xs, n_comp)))
        graded_runs.append(postmitotic_discretize(prof_clean + np.outer(e, layout)))
    S = np.stack(sharp_runs)
    G = np.stack([np.asarray(g) for g in graded_runs])
    # the MODAL label, computed (not implied): per position, the majority label across seeds
    modal_g = np.array(
        [max(set(G[:, i].tolist()), key=G[:, i].tolist().count) for i in range(n_pos)]
    )
    sharp_w = int(sum(len(set(S[:, i].tolist())) > 1 for i in range(n_pos)))
    graded_w = int(sum((G[:, i] != modal_g[i]).any() for i in range(n_pos)))
    return sharp_w, graded_w, graded_w / n_pos


def contiguity_null(n_pos=128, seed=7):
    """(real_boundaries, scrambled_boundaries) through the production postmitotic labeler.
    The FIELD (profiles) is permuted, then labeled -- not the labels (the first draft
    permuted labels and was right only because this labeler happens to be elementwise).
    CLOSED FORMS, stated: the organizer field is strictly monotone so the real partition has
    exactly 1 boundary for any threshold; the scrambled expectation is 2k(n-k)/n (= 64 at
    n=128, k=64), measured 51..75 over 50 seeds."""
    prof = graded_profiles(n_pos)
    labels = np.asarray(postmitotic_discretize(prof))
    perm = np.random.default_rng(seed).permutation(n_pos)
    scrambled = np.asarray(postmitotic_discretize(prof[perm]))
    real_b = int((labels[1:] != labels[:-1]).sum())
    scram_b = int((scrambled[1:] != scrambled[:-1]).sum())
    return real_b, scram_b


def adjusted_rand(a, b):
    """Exact adjusted Rand index (verified against sklearn on 3000+ random partitions by the
    T046 panel; the degenerate max==expected branch matches sklearn's convention)."""
    n = len(a)
    assert n == len(b) and n > 1
    cats_a, cats_b = sorted(set(a)), sorted(set(b))
    table = np.zeros((len(cats_a), len(cats_b)), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        table[cats_a.index(x), cats_b.index(y)] += 1

    def c2(v):
        return v * (v - 1) // 2

    sum_ij = sum(c2(int(v)) for v in table.ravel())
    sum_a = sum(c2(int(v)) for v in table.sum(axis=1))
    sum_b = sum(c2(int(v)) for v in table.sum(axis=0))
    total = c2(n)
    expected = sum_a * sum_b / total if total else 0.0
    max_index = (sum_a + sum_b) / 2.0
    if math.isclose(max_index, expected):
        return 1.0 if math.isclose(sum_ij, expected) else 0.0
    return (sum_ij - expected) / (max_index - expected)


def our_macro_partition():
    out, missing = {}, []
    for r, (nm, _z) in REGION_COORDS.items():
        macro = MACRO_OF_NEUROMERE.get(nm)
        if macro is None:
            missing.append(nm)
        else:
            out[r] = macro
    return out, missing


def report(do_assert=False):
    del do_assert  # bare == assert: one return path, no rc-0 FAIL (the doc_budget lesson)
    failed = []
    print("=" * 96)
    print("GROWTH COVERAGE GATE -- register row 7's four instruments, counted (T046)")
    print("=" * 96)
    print(
        "census scope: src/ minus reasoning/ (disclosed; zero reasoning/ placement importers "
        "today, so the exclusion moves no number), minus placement.py itself, __main__ "
        "guards, module-level harness functions"
    )

    three_way = set(GROWABLE) | set(SCAFFOLD) | set(CHARGED_NEVER_GROWN)
    disjoint = (
        not (set(GROWABLE) & set(SCAFFOLD))
        and not (set(GROWABLE) & CHARGED_NEVER_GROWN)
        and not (set(SCAFFOLD) & CHARGED_NEVER_GROWN)
    )
    exact = three_way == set(REGION_COORDS)
    if not (disjoint and exact):
        failed.append("three-way partition not exact/disjoint over REGION_COORDS")
    print(
        f"partition: GROWABLE {len(GROWABLE)} + SCAFFOLD {len(SCAFFOLD)} + "
        f"CHARGED-NEVER-GROWN {len(CHARGED_NEVER_GROWN)} == REGION_COORDS "
        f"{len(REGION_COORDS)}: {'exact+disjoint' if disjoint and exact else 'BROKEN'}"
    )

    census = live_growth_census()
    for region in sorted(census):
        print(f"  {region:<14} <- {', '.join(census[region])}")
    if census.get(UNRESOLVED):
        failed.append(
            f"{len(census[UNRESOLVED])} unresolvable growth site(s): {census[UNRESOLVED]} "
            "-- a per-site census must not swallow an unattributable live growth"
        )
    named = frozenset(census) - {UNRESOLVED}
    grown = named & frozenset(GROWABLE)
    refused = sorted(named & (frozenset(SCAFFOLD) | CHARGED_NEVER_GROWN))
    unknown = sorted(named - frozenset(REGION_COORDS))
    if refused:
        failed.append(
            f"live growth into refused territory {refused} -- SCAFFOLD (the envelope's "
            "anatomy included) and CHARGED-NEVER-GROWN never grow"
        )
    if unknown:
        failed.append(f"live growth into undeclared region name(s) {unknown}")
    if not grown >= BASELINE_REGIONS_GROWN:
        failed.append(
            f"coverage ratchet BROKEN: baseline {sorted(BASELINE_REGIONS_GROWN)} not in "
            f"measured {sorted(grown)}"
        )
    if grown > BASELINE_REGIONS_GROWN:
        print(
            f"TIGHTEN OWED: measured coverage {sorted(grown)} exceeds the pinned baseline "
            f"{sorted(BASELINE_REGIONS_GROWN)} -- bump the pin in a reviewed commit"
        )
    print(
        f"COVERAGE RED (reported every run, asserted only as a ratchet + refusals): "
        f"{len(grown)} of {len(GROWABLE)} growable regions receive grown units from a LIVE "
        f"path -- **BELOW FLOOR: cortex-only placement** (identity-pinned baseline "
        f"{sorted(BASELINE_REGIONS_GROWN)}; the set may only grow)"
    )

    sharp_w, graded_w, graded_f = boundary_widths()
    if sharp_w != 0:
        failed.append(
            f"SHARP arm width {sharp_w} != 0 -- discretize_sharp stopped being a pure "
            "function of position (its construction, via the production symbol)"
        )
    if graded_w <= 0:
        failed.append(
            "the graded control read width 0 under noise -- the harness's noise is not "
            "reaching the postmitotic labeler"
        )
    print(
        f"boundary sharpness (7 seeds, noise 0.03, PRODUCTION labelers, modal labels): "
        f"sharp {sharp_w} (0 by construction) vs graded {graded_w} positions "
        f"({graded_f:.3f} of the axis)"
    )

    real_b, scram_b = contiguity_null()
    if not (real_b == 1 and scram_b >= 10):
        failed.append(
            f"spatial null did not dissociate: real {real_b} boundaries vs scrambled "
            f"{scram_b} (bar 10 sits ~6x under the 2k(n-k)/n = 64 expectation)"
        )
    print(
        f"spatial null (field permuted, then labeled): real -> {real_b} boundary "
        f"(closed form: monotone field, one crossing); scrambled -> {scram_b}"
    )

    ours_map, missing = our_macro_partition()
    if missing:
        failed.append(f"neuromere(s) {sorted(set(missing))} have no declared macro assignment")
    regions = sorted(ours_map)
    mismatches = sorted(r for r in regions if ours_map[r] != ALLEN_MACRO.get(r))
    if mismatches:
        failed.append(
            f"macro anatomy disagrees with the Allen reference at {mismatches} -- "
            "ELEMENTWISE, because ARI alone is name-blind (a singleton's move is invisible)"
        )
    ours = [ours_map[r] for r in regions]
    allen = [ALLEN_MACRO[r] for r in regions]
    ari = adjusted_rand(ours, allen)
    rng = np.random.default_rng(13)
    null_ari = float(
        np.mean([adjusted_rand(list(rng.permutation(ours)), allen) for _ in range(200)])
    )
    print(
        f"external control: ELEMENTWISE agreement {len(regions) - len(mismatches)} of "
        f"{len(regions)} regions (asserted); ARI {ari:.3f} vs permutation null "
        f"{null_ari:+.3f} over 200 shuffles (display -- ARI alone is name-blind, disclosed)"
    )

    print(
        "DISCLOSURE (reported every run): regions-grown is a census of who WRITES a placement "
        "record. No LIVE path reads one -- a brain with placement replaced by constant tuples "
        "is value-identical in state() and telemetry on every key but the wall clock "
        "(tests/test_placement_charge_null.py pins it; specs/022 evidence/s1-placement-null.md). "
        "The one built reader, cortex/genomic_bottleneck (FR-033), was wired behind its paired "
        "A/B (genomic_bottleneck.py's --ab entrypoint) and measured a NULL at 24 seeds -- it reads placement "
        "only under the non-default init_pool modes (genome, genome_shuffled) and is deleted in the "
        "next commit. A region "
        "added here without a live reader claims coverage of a write-only record. The 007 "
        "growth->placement wire was built, measured at 0 differing keys, and reverted under P3 "
        "(2026-09-02); retire this line the day that test reds."
    )
    print("=" * 96)
    if failed:
        for f in failed:
            print(f"gate FAIL: {f}")
        return 1
    print(
        "gate OK -- and OK means the RATCHET held, the refusals are empty, and the "
        "instruments discriminate; the coverage red above is STILL RED (1 of 10). Green "
        "never means coverage is adequate."
    )
    return 0


# distinct-from: substrate/manifest_gate.py -- its self-check fires reds through its own
#   verdict path; the SAME pattern is followed here, no code shared. Site read.
# distinct-from: substrate/selfcheck_cache.py -- P8 reds for the cache's keys; unrelated. Read.
# distinct-from: substrate/doc_budget_gate.py -- reds for budget statuses; unrelated. Read.
# distinct-from: thalamus/global_workspace.py -- a slot-load site; unrelated. Read.
# distinct-from: cortex/self_improvement_suite.py -- the rung census; unrelated. Read.
def _report_rc(mod):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.report()
    return rc, buf.getvalue()


def self_check():
    """Every criterion's constructed red, ROUTED THROUGH report() (the first draft's reds
    re-implemented each predicate beside the gate -- the control did not follow the read
    path), by monkeypatching the module's globals and restoring in finally. The PACKAGE-NAME
    module object is patched AND executed -- under `-m` the running copy is __main__ and
    `import` yields a second copy, so patching one while running the other reads green
    forever (this exact trap fired on the first run of this self-check)."""
    import snn_data_lake.substrate.growth_coverage_gate as G

    print("GROWTH COVERAGE GATE self-check")
    rc0, out0 = _report_rc(G)
    assert rc0 == 0, "the gate is not green on the real tree; fix that before mutating it"
    assert "BELOW FLOOR" in out0, "the coverage red stopped printing"
    assert "DISCLOSURE (reported every run)" in out0, "the write-only disclosure stopped printing"

    def mutant(name, attr, value, needle):
        saved = getattr(G, attr)
        try:
            setattr(G, attr, value)
            rc, out = _report_rc(G)
            assert rc == 1, f"{name}: report() stayed green under the mutation"
            assert needle in out, f"{name}: failed for the WRONG reason ({needle!r} absent)"
        finally:
            setattr(G, attr, saved)
        print(f"  RED    {name} -> rc=1 via report() (bare form, proving bare==assert)")

    census_real = live_growth_census()
    assert any("neurogenesis_engine" in s for s in census_real.get("cortex", [])), (
        "the census did not attribute cortex growth to neurogenesis_engine"
    )
    all_sites = [s for lst in census_real.values() for s in lst]
    assert not any("theory_of_mind" in s for s in all_sites), (
        "the census counted theory_of_mind's .place -- name collision, not placement"
    )
    assert not any("genomic_bottleneck" in s for s in all_sites), (
        "the census counted a harness-function drive as a live path"
    )
    print(f"  green  census: cortex <- {census_real['cortex']}; false-positive classes absent")

    mutant(
        "ratchet (baseline names an ungrown region)",
        "BASELINE_REGIONS_GROWN",
        frozenset({"cortex", "hippocampus"}),
        "coverage ratchet BROKEN",
    )
    mutant(
        "refusal (a censused SCAFFOLD growth)",
        "live_growth_census",
        lambda src_root=None: {**census_real, "brainstem": ["evil.py:9"]},
        "refused territory",
    )
    mutant(
        "refusal (an undeclared region name)",
        "live_growth_census",
        lambda src_root=None: {**census_real, "hippocampus_typo": ["evil.py:9"]},
        "undeclared region",
    )
    mutant(
        "unresolved bucket reds",
        "live_growth_census",
        lambda src_root=None: {**census_real, UNRESOLVED: ["evil.py:9"]},
        "unresolvable growth site",
    )
    mutant(
        "partition exactness (a category emptied)",
        "CHARGED_NEVER_GROWN",
        frozenset(),
        "not exact/disjoint",
    )
    mutant(
        "sharpness (production sharp arm broken)",
        "boundary_widths",
        lambda **kw: (3, 15, 0.117),
        "SHARP arm width",
    )
    mutant(
        "sharpness control (noise not reaching the labeler)",
        "boundary_widths",
        lambda **kw: (0, 0, 0.0),
        "graded control read width 0",
    )
    mutant(
        "spatial null (fragmentation lost)",
        "contiguity_null",
        lambda **kw: (5, 6),
        "spatial null did not dissociate",
    )
    mutant(
        "Allen elementwise (one region moved across macros)",
        "ALLEN_MACRO",
        {**ALLEN_MACRO, "cerebellum": "forebrain"},
        "disagrees with the Allen reference",
    )

    # the modal computation is a real mode: constructed 3-seed stack where the modal label
    # wins 2-1 -- unanimity fails, mode is the majority (not merely "distinct labels seen").
    G3 = np.array([["a", "a"], ["a", "b"], ["b", "b"]])  # 3 seeds x 2 positions, no ties
    modal = [max(set(G3[:, i].tolist()), key=G3[:, i].tolist().count) for i in range(G3.shape[1])]
    assert modal == ["a", "b"], "the modal computation is not a majority"
    print("  green  modal label is a computed majority (2-1 case constructed)")

    # graded monotonicity in noise -- the graded arm's own reachable red family.
    _s0, g0, _f0 = boundary_widths(noise=0.0)
    _s1, g1, _f1 = boundary_widths(noise=0.03)
    _s2, g2, _f2 = boundary_widths(noise=0.10)
    assert g0 == 0 and 0 < g1 < g2, f"graded width not monotone in noise ({g0}, {g1}, {g2})"
    print(f"  green  graded width monotone in noise: {g0} -> {g1} -> {g2} (0.0/0.03/0.10)")

    # ARI stays exact + negative-capable (display-only, but it must not lie as a display).
    assert math.isclose(adjusted_rand(["x", "x", "y", "y"], ["x", "x", "y", "y"]), 1.0)
    assert adjusted_rand(["x", "y", "x", "y"], ["y", "x", "y", "x"]) > 0.99  # relabel-invariant
    assert adjusted_rand(["x", "x", "y", "y"], ["x", "y", "x", "y"]) < 0.0  # negative-capable
    print("  green  ARI: exact, relabel-invariant (why it is display-only), negative-capable")

    print("self-check OK: every criterion red-constructed THROUGH report(), bare==assert proven")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(self_check())
    raise SystemExit(report(do_assert="--assert" in sys.argv))
