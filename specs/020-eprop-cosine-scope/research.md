# Phase 0 Research: scope the e-prop cosine clause

**Date**: 2026-08-17 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

Five investigations against the real tree. **All five returned a blocking finding.** One of them
falsifies a premise in the spec's own "Why this exists" section, and it is recorded first because it
resizes a user story.

---

## R0. CORRECTION — the spec's US3 premise is measurably wrong

**Decision**: keep the reconciliation, but restate what it buys. It fixes an inline TEST FIXTURE, not
the guard protecting the shipped encoders.

**The claim that was wrong** (spec "Why this exists", item 3): *"The only mechanization the e-prop
invariant has validates a factorization neither shipped encoder runs."*

**Measured FALSE.** There are **three** cosine mechanizations in the governed tree, and all three are
registered in `run_selfchecks.py` TORCH_GATED, so all three run in the CI torch-tier job:

| site | asserts against | measured |
|---|---|---|
| `cortex/eprop_shd.py::run_smoke` | its OWN shipped `eprop_grads` (kappa-filtered) | 0.9992 |
| `cortex/eprop_nmnist.py::run_smoke` | its OWN shipped `eprop_grads` (kappa-filtered) | 0.9997 |
| `cortex/eprop_kernel.py::_self_check` | `_TinyNet.eprop_grads` (unfiltered) | 0.9714 |

The underlying observation still holds — the kernel's `_TinyNet` really does use a different
factorization from both encoders. What was wrong is the CONSEQUENCE: each encoder already guards
itself on its own form. The kernel's `_TinyNet` is an inline fixture, not the shipped encoders'
guard.

**Rationale for keeping the work anyway**: measured cost is 13 insertions / 7 deletions in one
function, all gates green, and it turns a **currently false in-file claim into a true one**.
`_self_check` states the readout gradients "are IDENTICAL between e-prop and BPTT by construction".
Measured false for the present form (cos_readout 0.986830, relative L2 error 7.423e-01 against
autograd); under the filtered form it becomes exactly true (cos 1.000000).

**Alternatives considered**: bringing the encoders to the kernel's form (rejected — the encoders are
the correct Bellec factorization); leaving it and disclosing (rejected by owner decision, FR-011).

**Risks**: (1) **Do not over-claim.** Writing this up as "the instrument guarding the shipped encoders
was wrong and is now right" overstates it. (2) **"No verdict moved" is a statement about SLACK, not
size** — healthy cells sit at cos_hidden 0.76–0.9998, i.e. 0.26–0.50 above the bar, so a ~0.01–0.02
factorization change cannot cross it regardless of importance. In the **truncation regime the forms
DO separate verdicts** (g_rec=50: 0.7376 GREEN vs 0.4105 RED), which is the reachable red.

**Evidence re-derivation cost** (7 recorded numbers rot, all inside comments): cos_hidden
0.9714→0.9993; cos_all 0.8236→0.9993; RED cosine −0.2540→−0.1782; the "90.7% of squared norm" figure
→ 81.2%; three seeded-defect rows 0.5859/0.7465/0.7398 → 0.6504/0.9214/0.9155 (the ARGUMENT survives
and strengthens). **`basal_ganglia/mlp_q.py` carries a SECOND copy of the 90.7% figure** — co-edit
required, or the claim lives in two places and gets corrected in one.

---

## R1. Static claim-detection — split verdict, and not where the spec put the split

**Decision**:
- **Declaration**: a module-scope constant read by `ast.literal_eval` on an `ast.Assign`, never
  imported. Shape `LEARNING_RULES = {"<Class>.<method>": ("<disposition>", "<reason>")}` with **FOUR**
  dispositions: `LOCAL` / `APPROXIMATES_GRADIENT` / `EXACT_GRADIENT` / `CONTROL`.
- **Inference**: DEMOTED from independent second opinion to a **recall net that may answer only CLAIM
  or UNKNOWN, never NO-CLAIM**.
- **UNDETERMINABLE** becomes a first-class classified state, ratcheted DOWN. *"Classified" must not
  mean "decided."*

**Rationale**: the half FR-013 worried about is SOUND — 378/378 files AST-parsed with 0 failures
(negative control proved the parser can reject), against the tree's recorded 25.7% import-failure
rate. **Coverage is not the risk.** The half the spec ASSUMED works is the one that fails: structural
inference measured **~31% error in BOTH directions**, with false negatives on precisely the landmark
cases — `mlp_q`, `jepa_world_model`, `gf.MLP`, `predictive_coding`. Three structural properties of
this tree defeat code-only inference:

1. **Backprop is routinely hand-rolled**, so there is no token to key on. `reasoning/gf.py::MLP`
   self-describes as trained by gradient descent and contains no `.backward()`, no autograd, no
   `torch.optim`. `mlp_q` says the rule is "backpropagated BY HAND". `reasoning/eprop.py::Net.grad_bptt`
   hand-rolls reverse-mode BPTT in pure `math` loops. All three read NO-CLAIM under a code-only pass.
2. **Controls live beside rules**, because P8 requires a control that is not the method itself.
   `eprop_kernel.py` contains the surrogate backward, the no-surrogate control, the independent
   autograd reference, and the e-prop rule — one file, four different reasons for a backprop signal.
3. **The claim and the update are in different functions.** `reasoning/eprop.py::Net.apply` is
   `self.Win[j][i] -= lr * dWin[j][i]` — byte-identical in shape to `oja_pca.OjaNeuron.learn`.

Mirror-image false positives are equally real: `ng_snn/train_softhebb.py` reads CLAIM off Adam +
`loss.backward()` while its actual rule is local SoftHebb (the torch machinery trains a linear
PROBE); `adex_neuron.py` reads CLAIM off `dw`, where `w` is the AdEx **adaptation current**, not a
synapse. Prose-only is worse: 45 of 120 rule sites mention gradients *in order to deny them*.

**Highest-value inference upgrade**: an intra-module **call graph**. In every claimant found, the
gradient is computed in one function and applied in another (`grad_eprop`/`grad_bptt` → `Net.apply`;
`eprop_grads` → `_apply_grads`). No mutation-site scan sees a claim; no claim-site scan sees a rule.

**Alternatives considered**: class attribute (LOST — cannot carry free-function rules such as
`reasoning/eprop.py::_batch_grad`); docstring marker (LOST — undiffable, and lives in the noisiest
medium); decorator (LOST — implies a runtime import in a subtree where 25.7% do not import).
Module-scope dict WON: precedent in three shipped gates (`dormancy_gate.EXEMPT`,
`method_dormancy_gate.EXEMPT_METHOD`, `eprop_transport_gate.EXEMPT`), all already read statically.

**BLOCKER**: FR-016's "zero unclassified, ALWAYS" as originally worded forces an undecidable rule out
as NO-CLAIM — a confident negative, which Standing Invariant 3 forbids. Measured wrong on ≥5 governed
modules, all load-bearing. **Resolved by defining "classified" to INCLUDE `UNDETERMINABLE`**, which
preserves the owner's absolute bound while honouring the invariant.

**Risks**: the enumeration is itself a heuristic with unproven recall — a hand-named control of 32
canonical rules hit 28. Every count (51 / 66 / 149) is a lower bound under one definition and an upper
bound under another. Revealed by a hand-labelled random sample of ~40 governed modules scored for
precision AND recall; not done.

---

## R2. Gate precedent — copy three gates, and copy NOTHING from p6_gate's baseline

**Decision**:
- **Walk + self-check skeleton** → `tools/p6_gate.py`. It is the only gate whose walk takes an
  in-memory **overlay** (`walk(root, overlay=None)`), letting `--self-check` construct a RED against
  the REAL tree with nothing touched on disk. That is exactly what "prove the criterion can
  DISCRIMINATE before it binds" needs. Its comparator `evaluate(metrics, baseline)` is PURE.
- **Classify semantics** → `substrate/method_dormancy_gate.py`, the only gate shipping a genuine
  third state (asserted / reported-not-asserted / standing-TODO).
- **Baseline storage** → `tools/p4c_ratchet.py` (machine-written, refuses a rise). **Never p6_gate.**

**Three name-resolution lessons the inference signal will hit**, each recorded from a measured failure
in `method_dormancy_gate`: import-scoped resolution (naive bare-attribute matching scored 0 failures
/ 47 UNCERTAIN — it could not catch its own motivating case); the `__main__` guard must follow ONE
LEVEL of indirection into `main()`'s body (marking only the literal block reported 0 dormant while the
motivating case sat dead); a function used as a VALUE is used (inspecting only `ast.Call` gave 5 false
positives of 12).

**BLOCKER — `tools/` has ZERO headroom.** `p4c_ratchet` reports `tools 30 / baseline 30 / +0`
(measured). A new file under `tools/` reddens CI **on its own commit**, and `--update-baseline`
REFUSES to absorb it. The gate must be a registered package module under
`src/snn_data_lake/substrate/` — the route all five sibling policy gates take.

**The p6_gate anti-pattern is WORSE than reported**: not only is there no mechanical refusal, but a
ceiling raised *together with the regression it excuses* passes BOTH the `--assert` and the
`--self-check` CI steps. Measured verdict: **LAUNDERED CLEAN**. FR-018 is therefore not optional
polish; it is the difference between a ratchet and a rubber stamp.

**Risk**: the liveness floor is the easiest thing to omit and the most expensive. Enforcement is
ABSOLUTE on "zero unclassified", and **a dead walk reports exactly zero unclassified**. Four gates in
this tree already fail that way.

---

## R3. Integration cost — declaration is free, the DOC is not, and one naming trap will bite

**Decision**: declaration as a module-level constant or class attribute, **never a method**. Checker
lives in `src/snn_data_lake/substrate/`, registered; the on-demand numeric half is a `--flag` path of
the SAME module (the `eprop_transport_gate` shape), not a second file.

**Measured — a declaration costs the P3 gates nothing**: `dormancy_gate` import edges +0,
`method_dormancy_gate` n_defs 66→66 / dormant 0→0, `import_manifest` targets +0 — **each with a
positive control in the same run that DID move** (+1 edge, n_defs 67 / dormant 1, +1 target). No
registration is needed for a declaration; REGIONS, `run_selfchecks` and `brain_atlas` all key on the
MODULE, never its contents.

**NAMING TRAP.** `method_dormancy_gate` matches `any(verb in name.lower())` over LEARN_VERBS
(learn, update, train, adapt, consolidate, reinforce, fit, plast, potentiate, depress, calibrate).
Measured hits on plausible API names: `declare_constraint` (con-**TRAIN**-t), `get_adapter`
(**ADAPT**-er), `plasticity_claim` (**PLAST**), `recalibrate`, `fit_claim`, `update_claim`. Any as a
METHOD on a class in a reachable module with no caller outside `__main__` becomes a DORMANT finding
and the gate asserts. **Measured clean**: `EPROP_GRADIENT_CLAIM`, `GRADIENT_CLAIM`,
`declare_gradient_claim`, `gradient_claim`, `classify`, `verify_claim`.

**BLOCKER — the doc amendment does not fit as a normal policy block.** `docs/POLICIES.md` has **541
words of headroom** (28,559 / 29,100). Real policy blocks in that same document measure 548, 623 and
747 words; a 21-line block measures 244. So the amendment must be condensed to **~250 words**, or the
ceiling raised in the same commit with the reason stated there. The other four budgeted docs have
23–108 words of headroom each, so none can absorb spillover.

**Highest-risk test**: `tests/test_eprop_train_tier_unreachable.py`. Its `_trains_by_backprop(tree)`
is "one function calls both `.backward()` and `.step()`", and
`test_the_reference_gradient_is_not_a_hazard` asserts `cortex.eprop_kernel` is in BRAIN_MODULES, is
reachable, and is NOT a hazard. The R0 reconciliation goes RED here if it puts an optimizer step in
the same function as `_fwd_bwd`'s `.backward()`.

---

## R4. The null — FR-009a is under-specified in a way that lets the implementer choose the answer

**Decision**: **NULL = within-matrix permutation of the CANDIDATE gradient.** Permute `W_in`
coordinates among themselves and off-diagonal `W_rec` coordinates among themselves, then cosine
against the UNCHANGED BPTT reference. It preserves norm, magnitude multiset and per-matrix scale, and
destroys only *which synapse got which magnitude* — exactly the directional claim the bar
adjudicates. It is a derangement of the method's OUTPUT, not the method (P8 guard 2), with in-tree
precedent at `tests/probe_015/criterion.py::permutation_null`.

**BLOCKER — "permuted or shuffled" are not interchangeable and give OPPOSITE verdicts on the same
gradients:**

| null | where 0.5 sits | P(null > 0.5) |
|---|---|---|
| within-matrix permutation | **+4.5 to +63 sd ABOVE** the null | ≤ 3e-5 |
| shuffled-label | **−0.32 to −2.55 sd BELOW** the null mean | 0.65 to 0.98 |

Whichever the implementer picks satisfies FR-009a as worded, and SC-013 accepts a single scalar that
hides which null produced it. **FR-009a must name the null exactly.** This does not touch the frozen
bar.

**Three further nulls REPORTED, never asserted**: isotropic matched-norm (flatters the bar — spread is
set by ambient d, so 0.5 reads +6.1 to +106.8 sd out; publish only to pre-empt the reader who computes
1/√d mentally); within-UNIT permutation (concedes the structure carrying the signal — median above 0.5
at every width, nearly the method itself); random relabel (a legitimate control for a DIFFERENT claim,
task specificity, on which 0.5 fails outright). **Untrained twin REJECTED** — moves net, forward path
and reference at once; uninterpretable.

**Scope**: measured on the 5 sites that actually assert this bar. `basal_ganglia/mlp_q.py` asserts
`|cos| > 0.5` on a DIFFERENT claim (Kolen-Pollack alignment, not gradient approximation) and is out of
scope. The ~22 local rules have no reference gradient, so the separation is **undefined** for them —
correctly, not as a gap.

**Risks**: (1) the shipped encoders were never EXECUTED — their `eprop_grads` was transcribed onto
`_TinyNet`, so a transcription error or a real-data regime could move the aligned tail. Revealed by
running both smoke paths and comparing to the transcription on matched geometry; **do this before
publishing any provenance number naming those modules.** (2) Every measurement is at
INITIALISATION; e-prop truncation error grows as recurrence strengthens, so a trained net could read
materially lower.

---

## Consolidated impact on the spec

| # | Finding | Spec change required |
|---|---|---|
| R0 | US3 premise false — three mechanizations exist, encoders self-guard | Restate "Why this exists" item 3; resize US3 to "fixes an inline fixture + makes a false in-file claim true" |
| R1 | Inference cannot adjudicate (~31% error) | FR-002a: inference demoted to CLAIM-or-UNKNOWN recall net; FR-002b re-scoped; add `UNDETERMINABLE` |
| R1 | FR-016 vs Standing Invariant 3 | Define "classified" to INCLUDE `UNDETERMINABLE`; ratchet it DOWN |
| R2 | `tools/` zone at zero headroom | Structure Decision: substrate/, registered — settled, not optional |
| R3 | POLICIES.md 541 words headroom | Amendment budget ≈250 words, or raise the ceiling in the same commit |
| R3 | LEARN_VERBS substring trap | Declaration must be a CONSTANT, and must avoid the six measured-unsafe names |
| R4 | FR-009a admits two nulls with opposite verdicts | Name the null exactly: within-matrix permutation |
