# Gradient-claim census — baseline

**T038.** The measured answer to the question this feature exists to ask: *how many of this tree's
learning rules does the e-prop invariant's cosine clause actually govern?*

Every figure below is re-derivable. Do not quote one from here without re-running the command:

```bash
python -m snn_data_lake.substrate.gradient_claim_gate
```

Measured 2026-08-17 against the working tree. Governed tier = the ASSERT scope; `reasoning/` is
reported and never asserted.

## Headline (FR-007, SC-005)

| | count |
|---|---|
| files parsed | 379 |
| governed symbols enumerated | 149 |
| — of those, **not a learning rule at all** | 52 |
| **real governed learning rules** | **97** |
| — that **OWE a cosine** (`APPROXIMATES_GRADIENT`, the claimants) | **17** |
| — the **REFERENCE** they are measured against (`EXACT_GRADIENT`) | **1** |
| — **MIS-SCOPED**: real rules the clause would sweep in that owe no cosine | **79** |

**79 of 97 real governed learning rules are MIS-SCOPED** — rules the clause would sweep in and
wrongly condemn. That is the count FR-007 asks for. Distinct from **80 owe no cosine**, which
additionally counts the REFERENCE (`bptt_grads`): it owes none either, but the clause does not
wrongly condemn it — it is the thing the others are measured against. Conflating the two is how
`docs/POLICIES.md` came to state a partition that did not sum (17 + 79 = 96, not 97). It is not an argument that the invariant is wrong; it is the
measurement that the clause, read as a blanket requirement on anything that learns, would demand a
gradient comparison from four rules in five that have no gradient to compare against.

The 79 break down as **72 `LOCAL`** (spike-timing- or eligibility-local, no gradient propagated),
**5 `CONTROL`** (exist to be compared against, deliberately), **2 `DIRECT_SOLVE`** (non-local, but
propagating no gradient and transporting no weight — `ng_snn::fit`'s greedy matching pursuit with an
analytical beta per block, and the refit that drives it).

## The 17 claimants, and the 1 reference they are measured against

**Corrected 2026-08-17.** This section first said "the 18 that owe a cosine" and lumped
`EXACT_GRADIENT` in with the claimants. That is wrong, and wrong in the direction this whole feature
exists to prevent: `cortex/eprop_kernel.py::bptt_grads` is the INDEPENDENT REFERENCE the other
seventeen are compared *against*. Its cosine with itself is 1.0 by construction, so asking it to
clear a bar is asking an identity to pass a test. Measured: governed `CLASSIFIED_CLAIMANT` = 17,
`CLASSIFIED_REFERENCE` = 1. The mis-scoped count of 79 is unaffected — a reference is not a rule the
clause would wrongly condemn either.

```
EXACT_GRADIENT         cortex/eprop_kernel.py::bptt_grads
APPROXIMATES_GRADIENT  basal_ganglia/mlp_q.py::MlpQ.update
APPROXIMATES_GRADIENT  basal_ganglia/mlp_q.py::_train
APPROXIMATES_GRADIENT  cerebellum/jepa_world_model.py::JepaWorldModel.train_step
APPROXIMATES_GRADIENT  cerebellum/spiking_jepa.py::SpikingJEPA._readout_grad
APPROXIMATES_GRADIENT  cerebellum/spiking_jepa.py::SpikingJEPA.train_step
APPROXIMATES_GRADIENT  cortex/eprop_kernel.py::_TinyNet.eprop_grads
APPROXIMATES_GRADIENT  cortex/eprop_kernel.py::_apply_grads
APPROXIMATES_GRADIENT  cortex/eprop_nmnist.py::eprop_grads
APPROXIMATES_GRADIENT  cortex/eprop_nmnist.py::train_and_eval
APPROXIMATES_GRADIENT  cortex/eprop_shd.py::eprop_grads
APPROXIMATES_GRADIENT  cortex/eprop_shd.py::train_and_eval
APPROXIMATES_GRADIENT  cortex/kb_ranking.py::DistMult.train_epoch
APPROXIMATES_GRADIENT  hippocampus/developmental_loop_brain.py::_LinearReadout.train_step
APPROXIMATES_GRADIENT  hippocampus/growable_readout.py::GrowableReadout.train_step
APPROXIMATES_GRADIENT  hippocampus/growable_readout.py::LaminarColumns.train_step
APPROXIMATES_GRADIENT  neuromodulation/spiking_curiosity.py::_learner_step
APPROXIMATES_GRADIENT  neuromodulation/spiking_curiosity.py::_run_curriculum_learner
```

`_LinearReadout.train_step` is a **deliberate over-flag**, and whether a linear probe over detached
features genuinely owes the cosine is the open scoping question T074a settles in Phase 8. It is
declared `APPROXIMATES_GRADIENT` until that lands so the census errs toward demanding too much
rather than too little.

## Mis-scoped rules by region

| region | mis-scoped |
|---|---|
| cortex | 26 |
| plasticity | 22 |
| hippocampus | 9 |
| amygdala | 6 |
| neuromodulation | 6 |
| basal_ganglia | 4 |
| cerebellum | 2 |
| hypothalamus / midbrain / substrate / thalamus | 1 each |

## Ratchet state

```
MISSING DECLARATION  : 0   governed   (baseline 0, SATISFIED)
UNDETERMINABLE       : 0   governed   (baseline 0, SATISFIED)
unclassified         : 0
parse failures       : 0
rules_seen_floor     : 149
```

Both ratchets read SATISFIED rather than merely NO WORSE. **That still does not mean the scope is
correct** — a ratchet asserts the structure did not get worse, never that the policy is satisfied.
The gate prints that sentence on every run for the same reason.

`rules_seen_floor` is a **liveness bound, not a target**: a walk that enumerated nothing would report
zero unclassified symbols and zero missing declarations, and would look perfect. The floor is what
makes a dead walk fail instead of pass.

## What the enumeration is worth (T032b)

The 52 `NOT_A_RULE` labels are not tree damage — they are a precision measurement of the walk, and
they are what let it be tightened on evidence rather than by hand. Scored against all 157
hand-reviewed declarations:

| signal | precision |
|---|---|
| verb match only | 57% |
| weight-AugAssign only | 73% |
| both together | 94% |

Per verb, `calibrate` scored **0/5** and `plast` **0/3** — pure false-positive generators, now
dropped. Verified before landing: **1 symbol removed, 0 real rules lost**
(`brainstem/serial_arm::assert_calibrated`, a safety precondition caught on the `calibrate`
substring). `rules_seen` 150 → 149.

**Tightening carries a specific, already-realised hazard in this feature.** Dropping `grad` earlier
to cure an over-count removed `cortex/eprop_kernel` — the reference gradient, the one
`EXACT_GRADIENT` in the tree — from the census entirely, all 17 of its definitions invisible, while
every gate stayed green. `_GRADIENT_VERBS` and a permanent recall-control test
(`_MUST_BE_ENUMERATED`, 8 load-bearing modules) exist because of that, and any future tightening runs
the same before/after measurement.

## The safety envelope

The five `brainstem/` envelope files are **read by the census and never asked to change**. After the
tightening the envelope contributes zero enumerated symbols, which is correct rather than a gap: the
only one it ever contributed was a precondition assert that is not a learning rule.
`tests/test_gradient_claim_gate.py::test_the_safety_envelope_is_classified_but_never_asked_to_change`
pins this. The applier additionally carries a hard `_SAFETY_ENVELOPE` refusal, proven firing on all
five, after a script-driven edit modified `brainstem/serial_arm.py` during this feature's execution
(reverted; envelope verified at diff-lines 0).

## A recorded number that drifted from the code (T039a)

Found while writing the Phase 4 tests, by running the module instead of reading about it.

| site | figure |
|---|---|
| `specs/015-resolve-weight-transport/evidence/baseline-mlpq.txt` | `[-0.4722, +0.2785]` (the original measurement) |
| `basal_ganglia/mlp_q.py` (two sites, `:92` and `:178`) | `[-0.472, +0.279]` |
| **`docs/POLICIES.md`** e-prop invariant block | `[-0.472, +0.279]` -- **CORRECTED 2026-08-18**, now carries `[-0.4366, +0.1844]` with an inline note |
| **what `python -m snn_data_lake.basal_ganglia.mlp_q` prints today** | **`[-0.4366, +0.1844]`** |

Three sites carry an interval the code no longer produces. The **property** those sites are actually
relied on for is stable and still true — `|cos| > 0.5` in **0 of 8** draws, with the shipped
Kolen-Pollack path at **+0.9031** clearing the bar and the transport identity reading **1.000** — so
nothing downstream is wrong. The interval is what rotted.

`tests/test_gradient_claim_measure.py::test_the_feedback_alignment_case_falsifies_the_instrument_not_the_rule`
therefore asserts the PROPERTY and deliberately does not pin the interval. Pinning it would have
planted a fourth copy of a number already stale in three places.

**`docs/POLICIES.md` is deliberately NOT edited here.** T038's commit gates governing-document
edits, and the correction belongs in the same amendment as T072/T074 rather than as a drive-by.

## Phase 5 — the reference gradient reconciled (T048–T054)

`cortex/eprop_kernel.py::_TinyNet.eprop_grads` ran an **unfiltered** eligibility factorization while
both shipped encoders run the **kappa-filtered** one. The criterion was therefore attached to a form
nothing runs. FR-010's single named exception permits this one rule-implementation change.

Two things were wrong, and the second was asserted in prose:

1. The eligibility was never kappa-filtered — `ebar = kappa*ebar + psi*eps` was absent.
2. The readout gradient used the **raw** spike train, `err ⊗ z`. The loss sees
   `yint = kappa*yint + z @ W_out`, so the exact gradient is `err ⊗ zbar`. A comment nonetheless
   claimed the readout gradients were "IDENTICAL between e-prop and BPTT **by construction**".

**Every movement, both values (SC-008) — zero silently absorbed:**

| quantity | before | after |
|---|---|---|
| `cos_hidden` (the asserted criterion) | 0.9714 | **0.9993** |
| all-parameter cosine (reported, not asserted) | 0.8236 | **0.9993** |
| RED case (b), truncation regime | −0.2540 | **−0.1782** |
| `W_out` cosine vs BPTT | 0.986830 | **1.000000** |
| `W_out` relative L2 | 7.4233e-01 | **8.16e-08** |
| `torch.allclose(readout)` | **False** | **True** |
| readout share of all-4 squared norm | 90.7% | **81.2%** |
| seeded: sign-flipped learning signal (all-4) | 0.5859 | **0.6504** |
| seeded: zeroed input eligibility (all-4) | 0.7465 | **0.9214** |
| seeded: zeroed surrogate derivative (all-4) | 0.7398 | **0.9155** |

Every one of these matched the value `tasks.md` predicted before the work started.

**The hidden-only rationale is strengthened, not weakened.** All three seeded defects still pass an
all-4 criterion (0.6504 / 0.9214 / 0.9155, all above 0.5) while reddening hidden-only
(−0.9993 / +0.2588 / 0.0000). Making the readout exact moved every all-4 value **up**, so that
criterion is now even easier to pass while the mechanism is broken.

### T047's stated configuration does not reproduce

T047 specified `g_rec=50 → 0.7376 GREEN vs 0.4105 RED` as the verdict-separating case. That pair
appears **only** in `tasks.md`, `research.md` and `plan.md` — in no code, no evidence file, and with
no fixture recorded. Measured at that configuration the pair is **−0.1258 / +0.1928, both RED**.

Rather than pin an unreproducible number (the T039 defect again), a **144-point sweep** (3 seeds ×
8 recurrent gains × 3 input gains × 2 sequence lengths) located **12** genuinely separating
configurations. The test uses the largest — seed 11, `g_rec ×50`, `g_in 1.0`, T=24: reconciled
**+0.8465 GREEN** vs unfiltered **+0.1765 RED** — and states the fixture in full.

**Disclosed because it complicates the story:** 2 of those 12 separate the *other* way (seed 3 at
`g_rec ×5` and `×50`, `g_in 1.0`, T=96 — reconciled RED, unfiltered GREEN). The reconciled form is
what ships and is what makes the readout exact, but it is **not** uniformly higher on this metric at
every configuration.

### Scope, so it is not overread (T054)

This fixes an **inline test fixture, not the guard protecting the shipped encoders**. All three
mechanizations already run in the CI torch job and each encoder guards itself on its own
kappa-filtered form. Nothing shipped was mis-trained. What was wrong is that the **reference this
module offers** was not the rule the tree runs.

## Phase 6 — the nulls, the provenance, and a finding about the bar (T055–T062)

### The encoders were RUN, not transcribed (T060)

research.md R4 risk 1 recorded that the shipped encoders had never been executed for any figure
naming them — only read and transcribed. They have now been executed:

| module | hidden cosine | readout cosine |
|---|---|---|
| `cortex/eprop_shd.py` | **0.9992** | 1.000000 |
| `cortex/eprop_nmnist.py` | **0.9997** | 1.000000 |
| `cortex/eprop_kernel.py::_TinyNet` **after** T048 | **0.9993** | 1.000000 |
| `_TinyNet` **before** T048 | 0.9714 | 0.986830 |

This is independent confirmation of Phase 5's direction: both shipped encoders already measured the
readout at exactly 1.000000 and already describe it as exact by construction. They were right; the
reference fixture was the outlier, and it now sits in their regime rather than 0.028 away from it.

### The three reported-only nulls, and one of them reverses the verdict (FR-009c)

Measured on the anchor fixture, asserted on by nothing:

| null | mean | P(null > 0.5) |
|---|---|---|
| **asserted**: within-matrix permutation | +0.0214 | **0.0000** |
| isotropic matched-norm | +0.0012 | 0.0000 |
| within-**unit** permutation | +0.1020 | **0.1000** |
| **random relabel** (reference recomputed on shuffled targets; **N=1**, a single re-run, not a sampled null) | **+0.6500** | **1.0000** |

The untrained twin stays REJECTED as a control: it moves the net, the forward path and the reference
at once, so a movement cannot be attributed to any of them.

### FINDING FOR A SEPARATE DECISION — the bar is not moved here (FR-009b, T062)

**A gradient computed against SHUFFLED LABELS scores 0.6500 and clears the `0.5` bar on every
draw.** Against that null the bar does not discriminate at all: `P(null > bar) = 1.0000`.

This does not invalidate the aligned measurement — against the *mandated* null the separation is
real and large (aligned +0.9481, bar at +3.06 sd, `P(null > bar) = 0.0000`). What it shows is that
**which null you pick decides whether `0.5` means anything**, which is exactly why FR-009a had to
name one and why FR-009c requires the others be published rather than quietly dropped.

Per FR-009b the value is **recorded, not moved**. `git diff` on `docs/POLICIES.md` and
`.specify/memory/constitution.md` is **empty** — the bar is untouched. Whether `0.5` is well placed
is a separate decision with its own evidence burden, and moving it inside a feature that also
reconciled the reference gradient would leave any flipped verdict with two candidate causes.

### Staleness is derived, not decorative (T058/T059)

`measure_freshness()` recomputes the digest from the tree and returns one of three structurally
distinct answers — **ABSENT** (never run; must not read as a pass), **STALE** (ran against
declarations that have since changed), **CURRENT**. A digest that is never recomputed reads
identical to one that always matches, so the test drives all three plus an unreadable-file case.

## SC-012 — the 31 declared-vs-inferred disagreements, named (T080)

SC-012 asked for the disagreement count to fall to zero, or for every remaining one to be named with
the reason it is unresolved. **The count is 31 and it cannot reach zero**, so here is the list and
the reason.

**Why zero is unreachable by construction.** `infer()` may answer only `CLAIM` or `UNKNOWN` — it is
structurally incapable of returning `NO-CLAIM`, because structural inference measured ~31% error in
both directions and a confident negative from that signal is the verdict Standing Invariant 3 names
as the dangerous one. Measured across the governed census: `UNKNOWN` 101, `CLAIM` 48, `NO-CLAIM` 0.
So any real local rule — which by definition mutates a weight — is inferred `CLAIM` and declared
`LOCAL`, and that pairing is counted as a "disagreement" while being exactly the designed behaviour:
**the declaration adjudicates, the inference only recalls.**

A disagreement here therefore means "the recall net flagged it, a human declared otherwise", which is
the mechanism working, not failing. It is not a defect count and should not be ratcheted as one.

| declared | count | why it disagrees |
|---|---|---|
| `LOCAL` | 24 | a local rule mutates a weight, so `CLAIM` is the only answer `infer()` can give |
| `NOT_A_RULE` | 4 | named below |
| `CONTROL` | 2 | named below |
| `DIRECT_SOLVE` | 1 | named below |

**The 7 non-`LOCAL` disagreements, individually:**

| symbol | declared | why unresolved |
|---|---|---|
| `basal_ganglia/r_stdp.py::calibrate` | NOT_A_RULE | sets a scale from a measurement; the walk caught it on the `calibrate` substring |
| `cerebellum/spiking_jepa.py::main` | NOT_A_RULE | a module entrypoint that *drives* training; it calls a rule, it is not one |
| `hippocampus/function_preserving_growth.py::SpikingGrowthLayer.add_unit_gradmax` | NOT_A_RULE | inserts a unit chosen by a gradient-magnitude criterion; it reads a gradient to pick where to grow and updates no weight |
| `substrate/region_coherence.py::RegionCoherence.update` | NOT_A_RULE | maintains a running coherence statistic, no synapse involved |
| `hippocampus/ng_snn.py::NGSNN.fit` | DIRECT_SOLVE | the closed-form batch solve POLICIES.md adjudicates as non-local yet propagating no gradient — the disposition FR-002g exists for |
| `neuromodulation/naca.py::_off_path_byte_identity` | CONTROL | a byte-identity control proving an off-path change moves nothing |
| `neuromodulation/spiking_curiosity.py::_run_jump_learner` | CONTROL | the comparison arm of a curiosity experiment |

Every one is inferred `CLAIM` because it touches something weight-shaped; every one is declared
otherwise because a human read what it does. That is the intended division of labour, and the list
above is what SC-012 asked for in place of a bare number.

## Requirement traceability, and the four defects the audit found (2026-08-18)

Tasks reached 91/91 before this audit ran. **A task checkbox is not a requirement**, and the audit
found four things wrong with a feature already declared complete. Three were shipped in commits.

| # | defect | where |
|---|---|---|
| 1 | The stale interval `[-0.472, +0.279]` — **deferred to the Phase 8 amendment by this very document, then not fixed** | `docs/POLICIES.md` |
| 2 | The amendment claimed BOTH autograd copies were `declared APPROXIMATES_GRADIENT`; `reasoning/_loop_common::LinearReadout.train_step` **carried no declaration at all** | `docs/POLICIES.md` + T074a marked done |
| 3 | `coherence_ceiling()` — the function documenting the derivation — had **no production caller** | `gradient_claim_gate.py` |
| 4 | **`test_this_feature_touched_no_safety_envelope_file` would have REDDENED CI**: its positive control asserted the live change set is non-empty, and a clean checkout is empty | `tests/test_gradient_claim_gate.py` |

Defect 4 is the one worth re-reading. The control was aimed at the wrong thing: an empty change set
does not make the pass *vacuous*, it makes it **true and uninteresting**. What must not be vacuous is
the CHECK, and that is `test_the_fr010_check_fires_on_a_constructed_violation`'s job. Verified across
all four change-set states (empty / dirty / no-git / envelope-path-planted) after the fix.

Defect 2 slipped for a structural reason worth naming: **`reasoning/` is UNGOVERNED**, so the census
REPORTS `MISSING_DECLARATION` there and never asserts on it. A governing document can therefore
assert a disposition for an ungoverned symbol that the tree does not carry, and no gate objects.

### FR/SC ids with no trace in code

Of 66 ids in `spec.md`, 11 had no textual trace in code or record. `FR-002a`, `FR-002c` and
`FR-002d` are now tagged at the symbols implementing them (`resolve`, `read_declaration`,
`DECL_NAME`). The remainder are satisfied but not code-taggable, and are recorded here instead:

| id | where it is satisfied |
|---|---|
| FR-002e / FR-002f | `NOT_A_RULE` + its two drawer guards; pinned by `test_not_a_rule_is_reachable_only_by_declaration` |
| FR-008a | both governing surfaces amended in ONE commit — `06c6461a` |
| SC-013 | separation published above; bar value unchanged (`git diff` on both governing docs empty at Phase 8) |
| SC-014 | exactly one hand-maintained record — verified with a control: `EXEMPT` dict 0, `stale_exempt` 0, `stale_exemptions` 1 |
| SC-020 | the classifier cannot emit `NOT_A_RULE` without a declaration (constructed case) |
| SC-021 | the live-path-reachability clause is in `docs/POLICIES.md`; both autograd copies now share the disposition (defect 2) |
| **SC-004** | **not mechanizable** — "a reviewer can answer in under a minute" has no gate. Stated rather than silently counted as met |

### Known and deliberately not changed

`fr010_findings` / `fr010_changed_paths` are exercised only by tests. Those run in CI, so the check
is charged — but **`--assert` does not run FR-010**, and it should not: the envelope *may* be
modified with explicit approval, so a gate that refuses every envelope change would block approved
work. The constructed test is the guarantee; this is a scope statement, not a gap.

## The INSTRUMENT_FALSIFIED inversion, relocated here from docs/POLICIES.md

The Phase 8 amendment stated that `mlp_q`'s frozen feedback-alignment arm is the worked
`INSTRUMENT_FALSIFIED` case **because it misses the bar on every draw**. That is not what produces
that verdict, and as written it licensed "instrument falsified, rule left UNJUDGED" for **any** rule
that merely FAILS the cosine — the condemnation-avoidance the two-stage verdict exists to prevent.

**What actually triggers it**, measured against `gradient_claim_gate::verdict_for`:

| construction | coherence product | verdict |
|---|---|---|
| `(0.99, 0.99)`, aligned 0.9 | 0.980 | **INSTRUMENT_FALSIFIED** |
| `(0.10, 0.10)`, aligned −0.4366 (mlp_q FA worst draw) | 0.010 | `RULE_JUDGED` |
| mlp_q FA, all 8 draws, from `main()`'s own vectors | **~0.0001** | `RULE_JUDGED` **8 of 8** |

`verdict_for` keys the token **solely** on the coherence product reaching the bar — the condition
under which a within-matrix permutation stops being a null and the separation statistic inverts. A
low cosine at ordinary coherence is `RULE_JUDGED`: the RULE failed, not the instrument.

**`mlp_q` is not a registered fixture at all** (`_FIXTURES` holds one entry, the stdlib e-prop
anchor), so the gate emits no verdict for it in any mode. The `mlp_q` row in POLICIES.md is a PROSE
adjudication and the machine-read `GRADIENT_CLAIM["MlpQ.update"]` disposition is unchanged — that
declaration carried the same inversion in its reason string and was corrected in the same change.

The measurement IS degenerate for that arm — it reduces to asking whether the feedback is parallel
to the forward weights, which is the weight symmetry feedback alignment exists to deny — but that
is an argument about a specific rule, made in prose, not a verdict the instrument produced.

## Autograd trainers and the reachability hole (F09/F10, 2026-08-18)

`docs/POLICIES.md` states the SCOPE RULE; this is the enumeration behind it, kept here because a
closed list in governing text rots the moment a trainer is added.

**MEASURED (re-derived 2026-08-18, `src/` only, ast.walk over functions containing both a `.backward()` call and a `.step()` call): 21 symbols** — an earlier 22 did not reproduce; method-variant range 18-21, never 22. The list:

```
hippocampus/developmental_loop_brain.py::train_step
hippocampus/growable_readout.py::train_step (x2: GrowableReadout, LaminarColumns)
reasoning/_loop_common.py::train_step
reasoning/hjepa_arm_torch.py::train_step
reasoning/hjepa_torch.py::train_step
reasoning/jepa_scaffold_torch.py::train_step
reasoning/ng_snn/train_softhebb.py::sanity_linear
reasoning/rsnn_delays_torch.py::train_model
reasoning/snn_dreamer_6dof.py::update_world_model, ::update_actor_critic
reasoning/snn_motor_control.py::train_forward_model, ::model_based_reach, ::main
reasoning/snn_ssm_shd.py::run
reasoning/snn_world_model_rl.py::train_world_model, ::improve_policy
training/mnist.py::train_model
training/offload.py::train_with_offload
training/spiking_lm.py::train
training/trainer.py::train_model
```

The census
declares **17 governed claimants**. An earlier amendment named "the two genuine copies" as though
the set were closed; it is not, and the two it named are not the reached ones.

**There is NO train-tier exemption.** `snn_data_lake.training.cli` is an `_ENTRYPOINT` (one of
eight, alongside `substrate.demo`, `brainstem.live_arm`, `brainstem.closed_loop`,
`substrate.cockpit`, `substrate.web_ingest`, `substrate.nengo_bridge`, `substrate.region_wiring`),
so the `training/` tier is REACHED. The prior wording granted that tier a blanket permission and was
false when written.

**Why no gate decides this.** `dormancy_gate.reachable()` ends `return seen & BRAIN_MODULES` — it
masks to the governed tier and is structurally silent about `reasoning/` and `training/`, which is
where most of these live. So scope is a REVIEW JUDGMENT against the unmasked closure of
`_LIVE_ROOTS + _ENTRYPOINTS + _external_roots()` (POLICIES.md's SCOPE paragraph is authoritative for the definition), not a computed value. **Nothing detects a change in it**: a wiring
edit can move a trainer into scope with no gate, test or report firing. That is an OPEN HOLE, and
the over-flagging of affected symbols is a mitigation, not a fix.

## Phase 8 CLOSED (2026-08-18): the verification round's residue, recorded not fixed

Phase 8 ran NINE adversarial review rounds (three critics each: factual accuracy, internal
contradiction, overreach). Findings 20 -> 14 -> 20 -> 14 -> 13 -> 10 -> 11 -> 12 -> 11 -> 9;
BLOCKERs 5 -> 4 -> 4 -> 1 -> 1 -> 0 -> 1 -> 1 -> 1 -> **0**. From round 6 onward every worst
finding was a defect introduced by the PREVIOUS round's fixes -- the loop had become
self-referential (round 8's budget trim deleted the full path round 7 added; round 9 flagged it).

The exit condition, owner-set: a HANDS-OFF verification round against the frozen tree at
`f8554467` -- fix nothing unless a BLOCKER appears. It returned **9 findings, 0 BLOCKERs**, and
every figure in the amendment reproduced byte-exact for the FIFTH consecutive round. Per the rule,
the residue below is RECORDED, not fixed -- editing it would restart the oscillation this round
existed to break.

| # | sev | lens | finding (verbatim problem, abridged) |
|---|---|---|---|
| N01 | MINOR | FACTUAL ACCURACY | The amended supersession note in docs/POLICIES.md (the 'implement the gate before adopting the rule' bullet) says 'the P7/P8 gates ASSERT since 2026-08-05, per the enforcement rows' — but the P8 enforcement row carries NO date at all; only P7's row says 'both added 2026-08-05'. The constit |
| N02 | SERIOUS | INTERNAL CONTRADICTION | The constitution still asserts, twice, that the P7 and P8 gates are 'specified but not built', directly contradicting docs/POLICIES.md's enforcement rows — which the same amendment commit (f8554467) made the sole authority — and the amendment REMOVED the constitution's only in-file correct |
| N03 | SERIOUS | INTERNAL CONTRADICTION | The amendment's new P7 carve-out — 'an adopted trainer that PROPAGATES A GRADIENT ... owes > 0.5 ... (unless classified REFERENCE — see the SCOPE paragraph's self-declaration caveat)' — extends the REFERENCE classification to adopted-dependency trainers, but the SCOPE paragraph's validity  |
| N04 | MINOR | INTERNAL CONTRADICTION | POLICIES.md's amended GPU paragraph says offline-sweep 'gradient trainers there stay inside the e-prop SCOPE', naming hippocampus/growable_readout.py and hippocampus/developmental_loop_brain.py — but the SCOPE paragraph the sentence points at defines being 'inside this clause' by live-path |
| N05 | MINOR | INTERNAL CONTRADICTION | CLAUDE.md's P7 paragraph restates the adopted-trainer cosine clause WITHOUT the amendment's new REFERENCE carve-out — 'an adopted trainer that PROPAGATES A GRADIENT owes cosine > 0.5 against an independent one (SCOPED 2026-08-18 ...)' — so the gitignored surface now states a strictly broad |
| N06 | MINOR | INTERNAL CONTRADICTION | The constitution's amended 'Three provisional principles' paragraph keeps the claim 'P8's was the cheapest and the one with a named expected output, so it is the first that SHOULD stop being provisional' — but the SAME commit superseded-in-place the parallel sentence in POLICIES.md as stal |
| N07 | SERIOUS | overreach-and-underspeci | The wave-through channel the lens asks about is real, but it is not INSTRUMENT_FALSIFIED — it is false SELF-DECLARATION, and the amendment names it only in a parenthetical while giving the parallel reachability hole a bold OPEN HOLE header, blast radius, and closure lane. Scope classificat |
| N08 | SERIOUS | overreach-and-underspeci | LIVE-PATH REACHABILITY is defined tightly enough to act on, but nothing computes it, and the OPEN HOLE's recorded revisit trigger — 'the next promotion of a hazard-holding reasoning//training/ module' — does not cover the event class the hole is about. Promotion into the governed tier is t |
| N09 | MINOR | overreach-and-underspeci | The OPEN HOLE's closure is recorded as 'UNSCHEDULED (2026-08-18, deferred without an owner decision; no measurement says the risk is tolerable)'. By the repo's own standing rules an untaken owner decision is open work; the text is admirably honest about the deferral but records it as if di |

Full text (problems, evidence, fixes) is in the round-10 review output; the prescribed fixes are
retained there for whichever lane next touches these files. The known open residue in one line
each:

- The constitution's P7/P8 per-principle status lines still say "specified but not built" (a THIRD
  and FOURTH copy of the claim the amendment superseded twice elsewhere) -- the drift-class poster
  child, left for a consolidation lane rather than a tenth whack.
- The REFERENCE carve-out for ADOPTED trainers is review-only: no test or gate walks dependency
  source, so the "returns a gradient / never steps" line is mechanically drawn only in-repo.
- The self-declaration wave-through disclosure lives in a parenthetical; the promotion_sweep
  reminder is a print, not a gate.
- The OPEN HOLE's closure remains deliberately UNSCHEDULED with its reason and trigger recorded.

## Honesty notes

- The `~22 local rules` estimate in T032 was low by **3.3×**; the census found 72. Membership came
  from the census, which is why the task said to take it from there and not from a prose list.
- 25 ungoverned `MISSING_DECLARATION` (26 until 2026-08-18, when `reasoning/_loop_common.py` gained its declaration) and 48 ungoverned `UNDETERMINABLE` remain in `reasoning/`.
  These are reported and deliberately not asserted — `reasoning/` is outside the ratchet scope.
- The wall-clock figure is printed on every run and gates nothing (FR-013a, SC-019 verified by T034).
