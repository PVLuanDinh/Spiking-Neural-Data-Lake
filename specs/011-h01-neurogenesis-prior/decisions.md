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

**PILOT READING — NOT YET TAKEN.** This paragraph is a placeholder and is replaced, in the commit
that lands the pilot, by the measured drive-only held-out accuracy, the chance level 1/K, the same
protocol's SEEDED reading beside it, and the command that produced them. Until then T003 is decided
on its acceptance argument and its required measurement is OUTSTANDING, which is the honest state.

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

**PILOT READING — NOT YET TAKEN.** This paragraph is a placeholder and is replaced, in the commit
that lands the pilot, by the two measured PSI values (WRONG-vs-SEEDED and RANDOM-vs-SEEDED on the
realised table at the chosen width) and the resulting VALID / INVALID disposition of the WRONG arm.

**Rejected candidates, and what each would have falsified.**

| candidate | what it would falsify | rejected because |
|---|---|---|
| INVERTED E/I | that the prior's SIGN assignment carries information | sign is EXCLUDED from drift by design (Dale; data-model §4, C5) and SC-004 asserts it survives learning. An inverted population changes the dynamics for a reason unrelated to the prior's wiring content, and it can fire red 2 while leaving every DriftRecord at distance 0 — which would also fire red 1, for the wrong cause. |
| SHUFFLED LAYER LABELS | that the layer ANNOTATION carries information | on a synapse-graph arm it changes no edge, so the dynamics are identical to SEEDED and red 2 is vacuous — a criterion that cannot fail is deleted, not weakened (P8 clause 1). |
| UNIFORM-ACROSS-LAYERS | that the per-layer proportions carry information | it destroys the degree sequence, so its question is inseparable from the RANDOM arm's; two arms asking one question is not two controls. |

---

## What these two decisions do NOT close

- specs/011's original seeded statistics — layer occupancy, layer geometry, the excitatory CELL
  fraction — are PLACEMENT-side quantities that T063 does not measure (they are the pinned null
  above). FR-008's obligation on them is NOT claimed closed; T063's statistics are newly chosen
  graph-weight statistics, named as such in the pre-registration §2.4.
- T005 is untouched.
- Nothing here is a claim about the LIVE brain: the population is off the live tick, and an
  instrument that ticks a scratch population does not discharge the kernel's `wire` exemptions.
