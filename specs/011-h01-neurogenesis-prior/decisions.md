# specs/011 decisions — the Phase 2 decisions that gate Phase B

Created 2026-09-09 under specs/022 T063 (`evidence/t063-prior-drift-harness.md`, §1–2.11 FROZEN).
T003 and T004 are decided here, each with the rejected candidates and the side on which each failed.
T005 (positions drawn vs sampled) is NOT decided here and is not touched by T063; it gates T018/T021,
which T063 does not run.

Governing text: `specs/011/spec.md`::SC-005, FR-008/FR-009/FR-010; `specs/011/plan.md`::D1, D4;
`specs/011/contracts/interfaces.md`::C5; `specs/011/data-model.md` §4; `specs/022/spec.md`::SC-006,
FR-022; `PLAN-FINAL-2026-09-02.md` Part F's preamble; `docs/POLICIES.md`::P2, P3, P8.

---

## T003 — the task the three-arm comparison is scored on

**DECIDED: a K-way drive-class discrimination read from the NON-DRIVEN units of the population,
with a least-squares linear readout.**

Concretely (the frozen protocol is §2.3 of the T063 pre-registration, which this row does not
restate): K drive classes, each a fixed 5 % subset of units drawn per seed and disjoint from every
other class's subset; a trial drives one class's subset for `T_trial` ticks with a deterministic
per-seed Poisson-shaped external current; the feature vector is the spike counts of the units that
belong to NO class subset; the readout is a least-squares linear map fit on previous blocks only;
the reward for a learning trial is +1 when the readout's prediction is correct and −1 otherwise.

**Why the acceptance bar is met.** T003's bar is a task that the seeding could *plausibly* affect but
that is *not defined in terms of* the seeding. The only path from a driven unit to a non-driven unit's
spike count is an edge of the graph, so connectivity plausibly affects the feature vector; and the task
is defined by the drive and the readout, which name no property of H01 — the same protocol scores a
randomly rewired table and a reversed-laminar table without a single change.

**The subject of the bar changed, and it is disclosed.** T003 was written when the arms were expected
to be *placement* priors. On the T062 kernel population the seeded object is the SYNAPSE GRAPH, not
placement, so the bar is read against the graph. That substitution is forced, not chosen: the object
brain's placement is a PINNED NULL (`tests/test_placement_charge_null.py` — a brain whose placement is
replaced by constant tuples is value-identical over 300 ticks), so three placement arms would produce
three identical outcomes and SC-005 would be vacuous in the always-passes direction, which is exactly
the failure T003's bar exists to prevent.

**The measurement T003 requires — recorded at pilot time, not assumed.** A DRIVE-ONLY population (the
same units, the same class subsets, the same drive, and NO EDGES) must score at chance on the held-out
trials. Without it, "the class code reaches the readout through the graph" is an assertion; with it,
the shortcut is measured absent. The pilot's reading is appended below.

**PILOT READING (2026-09-09, T063 pilot, git `a343a1f`, artifact `h01_c3_graph.npz` sha256
`a1d0491ac486bf194a479a8c6aab419df90c7f05d2ba643d6b16ed3c3d4a0549`, 14,215 units, 70,001 edges,
seed 0, K = 16 as the workload basis, 1 warm-up + 1 learning block).** The DRIVE-ONLY population —
the same units, the same class subsets, the same drive, and **no edges** — scores **0.0625** on the
held-out trials against a chance level of 1/K = **0.0625**: exactly chance, from **0** feature
spikes. The same protocol on the graph-bearing SEEDED arm reads **1.0000**. The shortcut is
measured absent, so the class code reaches the readout only through the graph. Command:
`PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.cortex.prior_drift_harness --pilot`;
every number is in `tools/t063_pilot.json`.

**AND THE SAME PILOT FOUND THE TASK SATURATED, which T003 must carry.** The no-learning RANDOM arm
scores **1.0000** at K = 8 and **1.0000** at K = 16 — and 1.0000 at every cell of a 13-configuration
sweep over class fractions 0.05 / 0.02 / 0.01 / 0.005 and K = 8 / 16 / 32 / 64, on both the SEEDED
and the RANDOM tables. A second instrument that fails differently agrees: nearest-centroid
classification of raw spike counts, fit and tested outside the harness with no readout at all,
reads **1.0000** for SEEDED and **1.0000** for RANDOM, while its own two controls hold exactly
(drive-only 0.1250 = chance with 0 feature spikes; label-shuffled 0.1250 for every arm). The task
as decided is therefore **too easy at this width for any wiring to be distinguished on it**: with a
class's reachable set a stable and essentially unique signature over ~14k units, every arm sits at
the ceiling and a positive control has nothing to beat. T063's pre-registered K rule — the ceiling
guard — has **no satisfying K**, and the campaign is blocked until the pre-registration is amended.
This paragraph records the finding; the amendment is the orchestrator's to make.

**Rejected candidates, and the side each failed on.**

| candidate | rejected because |
|---|---|
| next-category accuracy on the live percept stream (the L1a window) | UNREACHABLE, not merely wrong: the subject is a kernel `KernelState`/`Synapses` population that is off the live tick and has no percept stream, no NG-SNN and no category labels. A task the arms cannot be scored on at all. |
| C0's 20-item everyday battery | not in `src` (PLAN-FINAL Phase 4), and it scores a Brain, not a synapse population — the same unreachability. |
| SC-013's ARC-eval protocol | a different mechanism with its own criterion and its own reds; scoring T063 on it would make one criterion's null read as the other's. |
| the readout over ALL units' spike counts (the first draft of this task) | CIRCULAR-side failure: the driven subsets are themselves a class code, so a graph-free population is linearly decodable and the score stops being about wiring. Corrected to non-driven units only (Codex defect 6). |
| a task defined on the band-pair matrix itself (e.g. "recover the layer of a unit") | defined IN TERMS OF the prior — the circular side, exactly as T003 names it. |

---

## T004 — what "a deliberately wrong prior" concretely means

**DECIDED: the DIRECTION-REVERSED LAMINAR TABLE.** Every INTER-band edge of the seeded table has its
pre and post swapped; intra-band edges are unchanged; the edge count and the per-edge synapse-count
multiset are preserved; polarity is re-applied from the NEW source's cell sign (Dale); the fan-in
share is re-normalised so every unit's total absolute input is 1, as in every other arm.

**What it falsifies, and why it is the one that tests red 2's claim.** SC-005's second red is FOR the
claim that the prior's CONTENT — not merely its marginals — carries information that the task can use.
The reversal transposes the realised band-pair matrix exactly and swaps each unit's inter-band in- and
out-degree, so it destroys the laminar DIRECTION of the graph while holding the unit set, the cell
signs, the edge count, the synapse-count multiset and the fan-in normalisation fixed. If direction
carries nothing, the reversed table is just another draw and red 2 does not fire; if it carries
something, the reversed table is worse than a null that keeps the marginals and scrambles the rest.
DISCLOSED: the degree DISTRIBUTION is preserved, per-unit degrees are not.

**VALIDITY PRECONDITION (measured on the realised table BEFORE the campaign).** The reversed table's
band-pair weight-share matrix must be farther from SEEDED's — under the same PSI distance the drift
statistic S1 uses — than RANDOM's realised matrix is from SEEDED's. If it is not (a near-symmetric
laminar matrix), the WRONG arm is INVALID, red 2 is NOT READ, and the record states which asymmetry
was measured. This is the rule Codex defect 5 demanded: a corruption that provably changes the
intended information, or an invalidated control.

**PILOT READING (2026-09-09, same run): the WRONG arm is INVALID, and red 2 is NOT READ.** On the
realised tables at the full ingested width, PSI(WRONG, SEEDED) = **0.032389** against
PSI(RANDOM, SEEDED) = **2.281271** — the reversed table is **70x CLOSER** to the seeded band-pair
matrix than the marginal-matched null is, so the precondition fails.

**The asymmetry it measured, stated because the precondition requires it.** The H01 c3 band-pair
weight-share matrix carries **0.726520** of its mass on the DIAGONAL — intra-band edges, which the
reversal leaves untouched by construction — and **0.273480** off it; 20,739 of 70,001 edges
(29.6 %) are inter-band. Its total asymmetry is `sum |M - M^T| / 2` = **0.041883** of a matrix that
sums to 1, so transposing it moves about four per cent of the weight mass: PSI(M^T, M) =
**0.034053**, which agrees on a second path with the realised arm's 0.032389. This is exactly the
"near-symmetric laminar matrix" case the precondition was written for. The reversed-laminar prior
is not a wrong prior on this connectome — it is very nearly the same prior — and T004's decision
stands as a decision while its ARM is invalid on this artifact.

**Rejected candidates, and what each would have falsified.**

| candidate | what it would falsify | rejected because |
|---|---|---|
| INVERTED E/I | that the prior's SIGN assignment carries information | sign is EXCLUDED from drift by design (Dale; data-model §4, C5) and SC-004 asserts it survives learning. An inverted population changes the dynamics for a reason unrelated to the prior's wiring content, and it can fire red 2 while leaving every DriftRecord at distance 0 — which would also fire red 1, for the wrong cause. |
| SHUFFLED LAYER LABELS | that the layer ANNOTATION carries information | on a synapse-graph arm it changes no edge, so the dynamics are identical to SEEDED and red 2 is vacuous — a criterion that cannot fail is deleted, not weakened (P8 clause 1). |
| UNIFORM-ACROSS-LAYERS | that the per-layer proportions carry information | it destroys the degree sequence, so its question is inseparable from the RANDOM arm's; two arms asking one question is not two controls. |

---

## RE-DECISION 2026-09-09 (T063 pre-registration §2.12) — read this before either section above

Both sections above stand as the record of what was decided and measured FIRST. Neither is edited:
the amendment §2.12 of `specs/022-unified-brain-arc/evidence/t063-prior-drift-harness.md` re-decided
T003's INSTRUMENT and T004's ARM after the first pilot, and this section is what supersedes them.
Where the two disagree, this section is current and the sections above are history.

### T003 — the TASK is unchanged; its READOUT is not

The task's DEFINITION is exactly as decided above: K drive classes on fixed 5 % subsets drawn per
seed and disjoint, a deterministic per-seed Poisson-shaped drive, **features from NON-driven units
only**, a least-squares linear readout fit on previous blocks, `marginal_holdout` per class, and the
frozen per-trial protocol of §2.3. What changed is the readout's **CAPACITY**, and the guard that
sets it.

**Why.** The first pilot found the task saturated: with ~13,500 non-driven units against ~24
training trials per block, the least-squares readout is massively over-parameterised, so every
wiring interpolated its training block exactly and read **1.0000** held out — at K = 8 and K = 16,
at every cell of a 13-configuration sweep, and on a second instrument (nearest-centroid) that fits
no readout at all. The pre-registered K rule therefore selected **no K**.

**The replacement instrument.** The feature vector is a **FIXED per-seed random projection**
(Gaussian, orthonormalised) of the non-driven units' spike counts to **D** dimensions. **K = 8.**
**D is the LARGEST of {4, 8, 16, 32, 64, 128} at which the no-learning RANDOM arm scores below 0.9**
held out on the pilot seed — a capacity guard applied to the NULL, before any arm is compared. If no
D qualifies, the row is REFUSED ON MEASUREMENT with the finding that this population admits no
readout capacity at which its wiring can be separated from a marginal-matched null by this protocol.

**PILOT READING (2026-09-09, git `41c2705`, harness under amendment 2.12, same artifact
`h01_c3_graph.npz` sha256 `a1d0491a…4a0549`, 14,215 units, 70,001 edges, seed 0, K = 8, daemon PID
34316 LISTEN on 8137, settled = True from three 90 s CPU windows 120.297 / 121.109 / 112.016).**
The ladder reads **0.7500** at D = 4 and **1.0000** at D = 8, 16, 32, 64 and 128, so **D = 4** — the
largest under the ceiling, and the only candidate under it. DISCLOSED: the reading has 8 held-out
trials, so its grid is eighths; 0.7500 is 6/8 against a ceiling that admits up to 7/8. The guard is
COARSE at this block count, and it is satisfied as written.

**T003's required measurement, RE-MEASURED at the chosen D.** The DRIVE-ONLY population — the same
units, the same class subsets, the same drive, **no edges** — scores **0.1250** against a chance
level of 1/K = **0.1250**, while the same protocol on the graph-bearing SEEDED arm reads **0.6250**.
The shortcut is measured absent a second time: the projection opened no path around the graph.

**Rejected here, and the side each failed on.** OVERLAPPING class subsets — leaves the reachable-set
signature intact wherever the subsets differ, i.e. the same saturation one level down. A GRADED
regression target — changes the task's KIND after the numbers were seen, rather than its capacity.
The five candidates rejected in the T003 table above are unaffected and still stand.

### T004 — the WRONG PRIOR is now the BAND-PERMUTED-TARGETS table

**DECIDED: every edge keeps its pre and its synapse count, and its post is drawn from band
π(band(post)) at the same within-band degree rank, where π reverses cortical depth (Layer 1 ↔ Layer
6, Layer 2 ↔ Layer 5, Layer 3 ↔ Layer 4, White matter fixed); polarity is re-applied from the pre;
fan-in is re-normalised; the synapse-count multiset and the edge count are preserved.**

**Why the direction-reversed table was retired.** Not because it was wrong to decide — because it is
**structurally near-null on this connectome**, which the precondition existed to detect and did.
The band-pair weight-share matrix carries **0.726520** of its mass on the DIAGONAL, which a
transpose cannot touch; its total asymmetry is `Σ|M − Mᵀ|/2` = **0.041883**; PSI(Mᵀ, M) =
**0.034053**. The realised arm read PSI(WRONG, SEEDED) = **0.032389** against PSI(RANDOM, SEEDED) =
**2.281271** — invalid by 70×. **No transpose-based corruption can clear this precondition here.**

**What the permutation falsifies.** That the laminar **IDENTITY** of a synapse's target carries
information — a claim about the diagonal, which is where this connectome's mass actually is. It
moves each band's intra-band mass onto its depth-mirror, so it is far from SEEDED wherever band
masses differ (Layer 2's 4,709 units against Layer 6's 1,040).

**VALIDITY PRECONDITION, unchanged and MEASURED (same run).** PSI(WRONG, SEEDED) = **6.175870**
against PSI(RANDOM, SEEDED) = **2.281271** → the WRONG arm is **VALID**, 2.7× farther than the null.
**Off-band fallback edges: 0** — every edge landed in π(band(post)).

**TWO READINGS DISCLOSED, because the amendment's text admits them.** (1) "The same within-band
degree rank" between bands of different size is realised as the same rank **QUANTILE**: rank r of
n_b maps to ⌊r · n_π(b) / n_b⌋ — the only total and monotone reading of "same rank" across unequal
sets. Where the mirrored band is smaller the map is many-to-one, which is why this arm permits
repeated (pre, post) pairs and is not checked for multi-edges. (2) A remap that would create a
SELF-edge walks forward inside the target band; where that band is empty or holds only the pre, the
original post is kept and counted as an off-band fallback.

**T004's three earlier rejections (inverted E/I, shuffled layer labels, uniform-across-layers)
stand unchanged**, and the reversed-laminar table stays in the record above as the candidate this
connectome could not distinguish. Its readings are preserved verbatim as stage 1 of
`tools/t063_pilot.json`.

---

## What these two decisions do NOT close

- specs/011's original seeded statistics — layer occupancy, layer geometry, the excitatory CELL
  fraction — are PLACEMENT-side quantities that T063 does not measure (they are the pinned null
  above). FR-008's obligation on them is NOT claimed closed; T063's statistics are newly chosen
  graph-weight statistics, named as such in the pre-registration §2.4.
- T005 is untouched.
- Nothing here is a claim about the LIVE brain: the population is off the live tick, and an
  instrument that ticks a scratch population does not discharge the kernel's `wire` exemptions.
