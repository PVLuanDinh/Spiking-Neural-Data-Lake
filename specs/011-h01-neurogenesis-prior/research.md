# Phase 0 Research: H01 as a beatable prior

**Feature**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06

Source read in full on 2026-08-06: all 50 files of `github.com/ashapsoncoe/h01` (branch `main`,
22,410,439 bytes), retrieved via scrapling, read by eight parallel readers with hierarchical
synthesis. Every figure below is from that reading. Where readers disagreed, the disagreement is
recorded rather than averaged.

---

## D0 — Is this repo usable at all by an outsider?

**Decision**: The *scripts* are a record of analyses run against Google infrastructure; the *data* we
need is separately committed and standalone.

**Rationale**: The repo's own README states it in two sentences — several scripts use BigQuery
databases that "are not publicly available and require a credentials file", with the mitigation that
the data is downloadable separately. The shared library `common_functions_h01.py` (2,118 lines) is
four things bolted together, of which only one matters to us: ~42 of its 49 functions are pure
computation, and the layer test we need is ~40 lines of numpy with no external calls.

**Alternatives considered**: Running the pipeline ourselves — rejected, it needs credentials we do
not have and infrastructure we would not reproduce. Re-deriving layers from geometry — rejected, see
D3.

**Incidental finding, reported not used**: `common_functions_h01.py:184` contains a **live Google API
key committed to a public repository**. We did not use it. Worth reporting upstream.

---

## D1 — Which four files, and what exactly is in them

**Decision**: Four committed files, ~2 MB after discarding what we do not need, zero Google
dependency.

| File | Size | What we take |
|---|---|---|
| `agglo_20200916c3_cell_data.json` | 11.2 MB | 49,377 soma records: `true_x/true_y/true_z` (**nanometres**, int), `type` (11 values), `soma_cubic_um`, join key `agglo_seg` |
| `goog14r0s5c3_e_i_ratios_feb_2023.csv` | 2.0 MB | the `cortical layer` column (7 values: Layer 1–6 + White matter) |
| `excitatory_axons_summary_df.csv` + `inhibitory_axons_summary_df.csv` | 17 KB | uncensored GP-strength distribution, N = 1..20 |
| `cortical_bounds_circles.json` | 496 B | six circles (centre + radius), **in microns** |

Join coverage on `agglo_seg` is **45,994 / 45,994 = 100.0%**. 15,471 neurons carry the complete
(xyz + layer + type + synapse counts) tuple; 14,176 have ≥100 synapses.

**Caveat that must travel**: `agglo_seg` is **not unique** — 49,377 records collapse to 45,994 ids;
1,336 ids repeat and id `0` appears 1,095 times. The source's own lookup is last-wins and silently
drops 3,383 records. Cell-type strings disagree between the two files on 4.58% of joined pairs,
almost entirely glia-vs-glia relabelling.

---

## D2 — Units: the four coexisting conventions

**Decision**: canonical internal unit is **nanometres**; convert the circle geometry from microns at
the boundary and assert the scale.

**Rationale**: Confirmed four independent ways — the source's own axis label `"Distance from soma
(nm)"`, its neuroglancer `CoordinateSpace(units='nm')`, the function name
`get_nm_dist_along_skel_path`, and a moment check (fitted spine offsets 402 / 1992 / 2260 land at the
correct sub-micron physical scale).

**Three voxel grids coexist and no file in the source reconciles them:**

| Grid | nm/voxel | Used by |
|---|---|---|
| A — cell table | 4 × 4 × ~30 (variable) | `x_vx/y_vx/z_vx` **only** |
| B — synapse / EM | 8 × 8 × 33 | synapse queries, the basal-dendrite CSV |
| C — skeletons | 32 × 32 × 33 | skeleton graphs |

**The trap that will actually bite is not 1000×, it is 2×.** Grids A and B are both "voxels", both in
this volume, and differ by **exactly 2× in x and y** — an error that survives a smell test where a
1000× one does not. **Rule: use `true_x/true_y/true_z` and never touch the `_vx` fields.** Verified:
`true_x == 4*x_vx` exactly for all 49,365 non-zero records.

**Z is not a uniform grid.** `true_z / z_vx` ranges 30.000 → 31.943 (median 30.9); `z_vx` is a
*section index* and `true_z` carries a per-section alignment correction of up to 10.2 µm. Corroborated
independently: 5,019 sections, thicknesses 30 nm (33.1%) / 33 nm (59.5%) / 40–80 nm (7.4%), summing
to 170.3 µm against a measured soma z-span of 167,670 nm — 1.6% agreement.

**The one micron file, and the silent failure it causes**: `cortical_bounds_circles.json` is in
microns. The source's own consumer converts with a bare, uncommented `/ np.array([1000,1000])`, and
the word "micron" appears **nowhere** in that file. Feed it nanometres-as-microns and every point
falls outside every circle and lands in **Layer 1 — which is the fall-through bucket, not an error**.
Silent, plausible, wrong. **This is why SC-002 exists and why its red is mandatory.**

---

## D3 — Depth is radial, and we use the source's labels rather than re-deriving

**Decision**: use the `cortical layer` column directly. Port the circle test only as a *validator*
and for placing units the brain grows.

**Rationale**: Depth is **radial in the x-y plane** from a curvature centre lying **outside the
imaged block** (cy ≈ 2400–2810 µm against a tissue y-max of 2362 µm). Layer boundaries are arcs of
1.4–2.5 mm circles because the cortical sheet is curved across the ~3.9 mm slab. **Z is not depth** —
`true_z` spans essentially the full 0–167,670 nm range in *every* layer (L1 median 53,760; L5 median
87,381 — no separation whatsoever). Anything thresholding a Cartesian axis misassigns cells at the
edges.

**The six circles are three concentric pairs**, tested first-match-wins in file order (white matter,
then peeling off L6, L5, L4, L3, L2, with the remainder falling through to L1):

- pair (0,1) brackets **L6** — thickness 314.690 µm exactly
- pair (2,3) brackets **L4** — 216.274 µm exactly
- pair (4,5) brackets **L2** — 343.986 µm exactly
- **L5, L3 and L1 are the gaps between pairs**, bounded by circles with *different* centres
  (|A−B| = 770.50 µm, |B−C| = 47.40 µm), so their thickness is **direction-dependent**: L5 mean ≈
  332–382 µm but **2.0 µm along some rays**; L3 mean ≈ 474–480 µm; L1 unbounded.

Empirical validation against 49,377 somata: L6 99.6% recall / 80.4% purity; L4 81.1% / 97.0%; L2
97.2% / 90.1%.

**Alternatives considered**: re-deriving layers ourselves — **rejected on measurement**. Three
independent computations over the same cells produced white-matter counts of 12,872 / 13,839 /
14,923. Absolute pool sizes are not reproducible better than ~10–20%; the **excitatory fraction is
robust to ±0.015** across the same three. **Do not port the source's cluster-ordering step** — it
sorts clusters by mean x, which works only because this specific sample is tilted. Port the output
circles, never the sort.

---

## D4 — Dale's Principle: the mechanism to port

**Decision**: adopt the source's cell-typed sign assignment verbatim. It appears identically in three
of its scripts, which is itself evidence it is the intended contract.

**Sign is a property of the CELL**, set by morphological type and inherited by every synapse it
makes:

- **Excitatory**: `pyramidal neuron`, `excitatory/spiny neuron with atypical tree`, `spiny stellate neuron`
- **Inhibitory**: `interneuron`
- **Unsigned** (excluded from the signed population, never defaulted): `blood vessel cell`,
  `unclassified neuron`, `microglia/opc`, `astrocyte`, `c-shaped cell`, `unknown cell`, `oligodendrocyte`

**Rationale**: this is structurally Dale-compliant by construction, and it is exactly the invariant
the consuming tree lacks (tree-wide, `is_inhibitory|e_i_ratio` matches zero files).

**Free second prior**: excitatory-pre → AIS-post is dropped by the source as an artifact — the axon
initial segment is an inhibitory target.

**The quiet error this decision must avoid**: the source's `e_prop` is the fraction of a neuron's
**incoming synapses** that are excitatory (its query filters on `post_synaptic_partner`). Reading
"76.9% for layer 5" as "77% of L5 neurons are excitatory" conflates two different quantities.
Cell-count E:I answers *how many of each sign to instantiate*; synapse-level E/I answers *what mix of
drive a cell receives*. **We need the first; the column named for it is the second.**

---

## D5 — Connection strength: what the numbers are, and what they are not

**Decision**: usable for Phase D, with three caveats carried in provenance. **No fitted
distributional form exists** — do not invent one.

**What the CSVs contain**: the uncensored distribution over **79,827,631 axon fragments**, split E vs
I. Verified element-for-element against the hardcoded vectors in the source's R script.

**Three things that must travel with the numbers:**
1. **"GP strength" is a per-AXON MAXIMUM** over that axon's partners (`MAX(pair_count) GROUP BY
   pre_seg_id`) — an upper-tail order statistic, **not** a per-connection synapse count. The raw
   per-connection multiplicity distribution *is* computed by the source and then never read again and
   never published.
2. **These are axon fragments in a cut volume**: 79.8M fragments against ~15k somata.
3. **The population excludes every axon synapsing onto an AIS**, removing chandelier cells — a
   heavily multi-synaptic inhibitory class — from exactly the tail where the E-vs-I asymmetry lives.

**The structural fact worth carrying even if nothing is seeded**: strong connections are an
**inhibitory** phenomenon. Above N=10 excitatory is (4,4,0,1,0,0,0,0,1,0) while inhibitory is
(144,82,58,54,28,19,14,11,7,5). Inhibitory axons are **1.741× more likely** to be multi-synaptic
(2.330% vs 1.338% at N≥2) and reach N=20 where E effectively dies at N=12. A uniform strong-connection
prior across both pools would misrepresent the tissue.

**No fitted form**: the R script's "expected" curve is an empirical geometric-simulation null, not a
Poisson/negbin/power-law fit, and the only statistic is a KS test. It rejects overwhelmingly
(D = 0.0470 / 0.0454 / 0.0514; √n·D = 419.7 / 354.3 / 223.0 against a 1.36 critical value) with a
mixture-shaped deviation. **Trust the obs/exp ratios only to about N=6** — the null vector saturates
at 1.0, so the 134× and infinite entries are quantisation artifacts.

**No weight units anywhere.** Nothing in the source has conductance, millivolts, PSP amplitude or
release probability. Multiplicity → weight needs a per-synapse strength constant from outside, which
makes any such weight **derived, not measured**.

---

## D6 — Delays: refused, and why that is a result

**Decision**: **out of scope, and not approximated.**

**Rationale**: seven readers independently grepped all 50 files for `veloc`, `conduct`, `m/s`, `ms`,
`g-ratio`, `diameter`, `caliber`, `time constant`, `latenc`. **Zero hits, all seven.** The word
`delay` appears five times and every one is HTTP request throttling. The only "timing" field in any
data file is human proofreading wall-clock in seconds.

**The numerator exists and is good**: `get_nm_dist_along_skel_path` returns true *summed arc length*
along the reconstructed skeleton in nanometres, not Euclidean displacement. Committed path-length data
amounts to n = 293 records, and those are between paired synapses on one neurite for validating
synapse merging — not an arbor survey.

**Two proxies exist and both are traps**: (a) myelination is a real per-node categorical class with
**no calibration to m/s anywhere**; (b) axon calibre is the edge `radius` attribute — **the one
spatial field the source library never unit-converts**, so its units are undetermined.

**The general statement**: an EM reconstruction measures **geometry, not dynamics**. It can say how
far a spike must travel to the nanometre and cannot say how fast. Closing the gap needs a
calibre-and-myelination → velocity relation imported from electrophysiology, which is **a modelling
assumption we bring, not a measurement H01 provides**. Any delay so derived must be labelled
literature-sourced or it carries the authority of a measurement and none of its provenance.

**One honest sanity bound derivable from the source alone**: the volume is 3.864 × 2.362 × 0.168 mm,
so the longest possible intra-volume path is ~4.5 mm. At cortical unmyelinated velocities of
0.1–0.5 m/s that is **9–45 ms**.

---

## D7 — Accuracy bounds every lifted number inherits

**Decision**: each constant ships with its bound in the same record. A number that loses its bound
becomes a fake measurement.

- **`fnr_i = 0.35`** — one in three inhibitory synapses is never detected; I-called synapses are 17%
  wrong. **The correction that repairs this has an operator-precedence ambiguity moving the
  whole-volume E:I between 1.70:1 and 2.04:1** — and E:I is the single number this feature most wants.
- **Layer pools ±10–20%** (see D3); the E-fraction ±0.015.
- **`measure_e_vs_i_accuracy.py` is a well-posed per-layer accuracy harness whose *both* output CSVs
  are absent from the repo.** The per-layer error bar exists only in the paper.
- **Connection strength**: the AIS exclusion, above.

---

## D8 — The two structural findings that change the design

1. **Layer 1 has zero excitatory somata but 540 interneurons** (plus ~1,576 astrocytes). The
   Potjans-Diesmann assumption "layer 1 has no cell bodies" is **half right** — no pyramidal somata,
   but a dense interneuron population. The consuming circuit inherits the wrong half. Resolved as
   plan decision **D2** (a new inhibitory-only L1 population).
2. **The source's scheme has 7 regions** (including white matter, holding 12,872 cells / 670 neurons)
   against the circuit's 8 populations across 4 layers. **The mapping is a decision, not a copy.**
   Resolved as plan decision **D2**.

---

## D9 — Scaling the prior beyond the measured population

**Resolved in [plan.md](./plan.md) decision D1**: the prior is a set of per-layer **proportions that
scale** — not absolute counts, and not a first-N seed.

Recorded here because the *evidence* for that choice is a research finding rather than a design
preference: the source's absolute pool sizes are not reproducible better than ~10–20% across three
independent re-derivations, while the fractions are robust to ±0.015. **The data itself says which
quantity survives re-derivation.** Seeding with counts would import the source's irreproducibility as
if it were signal.

---

## Open questions carried into implementation

1. **Which task the three-arm comparison (SC-005) is scored on.** The drift instrument needs a
   downstream outcome to compare seeded against random. Choosing a task that placement cannot
   plausibly affect would make the criterion vacuous in the "always passes" direction; choosing one
   that only placement affects would make it circular. Belongs to `/speckit-tasks`.
2. **What "a deliberately wrong prior" means concretely** — shuffled layer labels, inverted E/I, or
   uniform-across-layers. Each falsifies something different, and the choice determines what SC-005's
   second red actually proves.
3. **Whether grown positions are drawn from the measured soma positions or sampled from a fitted
   spatial density.** Drawing reuses real coordinates and cannot generalise past the measured volume;
   sampling requires a density model that D9's reasoning would then apply to.
