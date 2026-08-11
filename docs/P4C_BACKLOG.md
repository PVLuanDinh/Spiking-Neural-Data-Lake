# P4c backlog — the ungoverned build list

**Why this file exists.** A 161-module classification of `src/snn_data_lake/reasoning/` was produced
on 2026-08-06 and its *aggregate counts* were written into `CLAUDE.md`, but the per-module
enumeration was never committed. An adversarial pass then tried to re-derive "the subset that
duplicates a governed twin" and could not: `git grep -i 'MERGE.THEN.RETIRE'` returned nothing, and a
mechanical basename-collision proxy yields only 2 pairs, both of which turned out to be
classification-wrong. Its finding, verbatim in substance: **until the classification is written down,
that subset cannot be audited by anyone.**

That is a P4c failure about P4c work — *"a build the gates cannot see is not a build the project can
claim"* applies to a **disposition** as much as to a module. A verdict that exists only in a
conversation is a verdict the project cannot check, cannot inherit, and cannot falsify.

**Status.** The counts below are the authoritative ledger. The per-module rows are being reconstructed
from the measurements as each is settled; a row is written here when it is MEASURED, never when it is
merely assigned. An unwritten row is honest; a guessed one is not.

---

## Counts

Classified 2026-08-06 against a directory count of 161, reconciling exactly with zero gaps and zero
overlaps (six batches, `index % 6 == k`, membership confirmed row-by-row):

| verdict | count |
|---|---|
| PROMOTE | 107 |
| MERGE-THEN-RETIRE | 29 |
| RETIRE | 16 |
| EXEMPT-with-reason | 9 |
| **total** | **161** |

**45 of 161 (28%) end in deletion rather than promotion.** The subtree shrinks by more than a quarter
before one new capability is added.

Re-derive the live count with `python tools/p4c_ratchet.py` — it is ratcheted per zone and may only
fall. Do not quote a number from this file that the tool can produce.

---

## Settled

### Retired 2026-08-06 (`f299b631`) — 9 of the 16 RETIRE

`snn_mnist_dc`, `deep_stdp_readout`, `ng_snn/grow_head_nmnist`, `developmental_loop_deep`,
`ng_snn/train_mnist`, `snn_storage_core_snntorch`, `developmental_loop_lake`, `nengo_gui_devloop`,
`snn_moe_stdp_mnist`.

Zero registrations removed, because zero existed — **and that does NOT generalise.** `run_selfchecks.py`
registers `reasoning.fly_hash`, so a future retirement there WILL move the count. Measure per module.

### Retired 2026-08-07 — `deep_stdp_2stage_svm`, the 11th of the 16 RETIRE

Kept at `f299b631` with no recorded reason; the fresh derivation found none that survives. It is a
COMPLETED experiment whose full verdict is preserved in CHANGELOG v3.63 — the Kheradpisheh
global-pool recipe FAILS on digits (conv1-global 56.7%, 2-stage-global 63.9% vs conv1-spatial
93.8%), and the lever that did deliver is the RBF-SVM on conv1 SPATIAL features (96.2%, the then
new best) — so the harness's question is answered and its answer recorded. Zero importers, zero
registrations (run_selfchecks / REGIONS / EXEMPT / import_manifest / p6_gate KNOWN_* all clear),
zero doc command blocks, symbol sweep clear (`_conv1_both` / `_conv2_global` / `_svm_eval`
nowhere else). The fix-check: its one CORRECTED note is about its own header prose, not a code
correction any survivor needs. It consumed `deep_stdp_latency` and `deep_stdp_2stage`; deleting
it removes one consumer of each — their own dispositions are unaffected but their importer counts
move, noted for whichever lane settles them.

### Retired 2026-08-07 — `eth_mnist_bindsnet`, the 10th of the 16 RETIRE

The NO-GO was never the module, it was the CO-EDIT: sole `bindsnet` importer in `src/`, so
deleting it alone took `dependency_ledger --assert` from UNUSED 0 to 1 (measured at `f299b631`).
Retired together with the `bindsnet` entry in the `[gpu]` extra — a declared distribution whose
only importer is RETIRE-classified is not a capability. Zero registrations existed
(run_selfchecks / REGIONS / EXEMPT lists / import_manifest / p6_gate KNOWN_* / tests all clear);
the module name appears in no `.py` outside one comment pointer (co-edited). Docs co-edited:
`RUNNING.md` §5 (its runnable command was the section headline), `PERFORMANCE.md` (flag mention +
Covered scripts), `install.py`'s `[gpu]` description (which was ALREADY stale — it still named
`snntorch`, removed from that extra 2026-08-06). PORT-BEFORE-DELETE, the fix-is-not-a-capability
check: the file carried ONE unique correction — the `torch._six` compat shim (BindsNET <= 0.3
imports `torch._six`, removed in torch >= 2.0; shim = `types.ModuleType` with `container_abcs`,
`string_classes`, `int_classes` in `sys.modules`) — preserved as prose in `RUNNING.md` §5. Its
measured numbers (literature ladder to ~95%, 400/20k ~ 86% CPU) survive in the CHANGELOG history.

### Retired 2026-08-11 — `lake_skills`, `full_brain_experience` (batch-1 of the bulk lane)

Both measured by a two-stage adversarial pass (analyze + refute-by-default) and hand-verified
before deletion; the first rows of the #10 bulk settlement.

**`lake_skills`** — zero importers tree-wide (dynamic routes swept: importlib / `__import__` /
getattr / string dispatch / `_AliasFinder` / blimp_bench SERVES list, all clear), zero
registrations, and its own docstring recorded the disposition owed since 2026-08-04. Both halves
superseded ON LIVE PATHS: routing by `skill_builder.SkillRegistry.route()` over the durable lake
(P4b-correct where lake_skills was RAM-only; live via `nightly_skills` <- `cortex/brain.py:3069`,
gate-registered) and the symmetric Hebbian pair-counter by `concept_reflection.py:120-122` (live
via `developmental_core`). Fix-check: corrects nothing a survivor needs. Unique residue recorded
here, unconsumed, port-on-demand: (1) connected-component skill extraction over the pair-weight
graph (concept_reflection builds the identical graph, exposes only association_graph /
strongest_link); (2) the WIRED/AUTOMATIC dual-threshold effortful-to-automatic distinction; (3)
the design argument that raw 64-d crc32 signatures route by direct cosine, NOT through
encode_latency into ProximityIndex's Victor-Purpura metric (unmeasured claim, licenses nothing).
Its docstring's P2 question (this ↔ `lake_skill_library`, "exactly one survives") is settled by
this deletion; `lake_skill_library`'s own row is unwritten and unaffected.

**`full_brain_experience`** — the completed-experiment shape. Written v3.39 (2026-07-15) to
demonstrate "all torch tiers engage on real inputs"; a third of its subject no longer exists
(`cortex/language_center.SpikeGPTServe` — the SpikeGPT organ was SCRAPPED at v3.77, and the
module's bare `except` misreported the missing module as missing weights); the remaining two
tiers are exercised WITH asserts by governed code (growth `brain.py:3780`/`:2723`, respond
`brain.py` `__main__`, perceive_event `brain.py:1360`, `pc_cortex_serve.py:304`). The script
itself asserts NOTHING — `main()` returns 0 unconditionally with every tier wrapped in
`except Exception` — so it is un-promotable by the P8 bar (no reachable red) and proves nothing
where it stands. Port list EMPTY, verified: its one correction (direct `Brain()` -> `make_brain()`,
AUDIT_BACKLOG M22) survives in `make_brain` itself. Honest gap recorded, not papered over: NO
test covers the integrated checkpoint-loaded perception path (zero test hits for
`perceive_event`/`_pc_cortex`/`grown_units`); if that integration smoke is ever wanted it must be
rebuilt as an ASSERTING governed self-check, not preserved from this file.

### Retired 2026-08-11 — batch-2 of the bulk lane: `readout_boost`, `readout_ensemble`, `developmental_loop_unsupervised`, `developmental_loop_lake_slda`

Same method as batch-1 (two-stage adversarial pass, then hand-verified); all four zero-importer
(dynamic routes swept) and zero-registration.

**`readout_boost` and `readout_ensemble` — RETIRED on NEW grounds, not the struck ones.** Both
were reclassified OUT of RETIRE 2026-08-07 because their then-claimed supersessions were wrong
(different-question / inferior-imports-superior; rows above stand). These retirements rest on the
`deep_stdp_2stage_svm` precedent instead: COMPLETED experiments with verdicts durably recorded —
boost: CHANGELOG (ensemble +0.3, MLP −0.7, readout at ceiling, "next gain is the encoder" — and
that mandated Tier-3 encoder is BUILT and governed, `cortex/rsnn_encoder.py`); ensemble:
CHANGELOG v3.61/v3.62 (ensemble −1.9% vs best single, oracle ceiling 92.2%, "Accuracy
investigation CLOSED"). Port lists verified empty: boost's MLPHead measured NEGATIVE and its
`_softmax_scores` was a per-site copy of `language_readout_slda.scores` discriminant math
(deletion REMOVES a P2 duplicate); ensemble's reusable pieces are behavior-identically superseded
by `_loop_common` (LinearReadout / envi / batched cache_features). `laminar_readout:74`'s
importer note co-edited.

**`developmental_loop_unsupervised`** — completed experiment (CHANGELOG v3.67: 88.3% purity, NMI
0.614, intrinsic error 1.000→0.148); every mechanism it exercises is governed and live elsewhere
(spiking_curiosity, growing_learner, similarity_sized_growth, competitive_learning,
percept_surprise). Unconsumed residue recorded, port-on-demand: `_nmi()` (~15 stdlib lines) was
the tree's only clustering-NMI; if an NMI scorer is ever wanted, port it next to
`competitive_learning.purity` — wiring it to nothing today would be the fake consumer.

**`developmental_loop_lake_slda`** — verdict preserved (CHANGELOG v3.75: the 94.9%/95.4%
durable-lake SLDA loop), mechanism absorbed near-verbatim by `developmental_core:170-196`
(consumed with its own asserts by `autonomous_brain_recovered`). PORTED BEFORE DELETING, in the
retiring commit: `LINEAR_LAKE_REF = 0.902` + its hardened provenance block moved to
`developmental_loop_lake99.py` as ONE named constant, replacing the four scattered literals that
file carried (the exact disagreeing-literals defect lake_slda's own comment documented).
RECORDED AS NEVER-ANSWERED: its LEVER 3 (`NORD_PROTO` prototype-vs-raw replay) has no measured
answer anywhere in-tree — the question retires unanswered with the harness; re-open it as a
fresh measurement if the sign ever matters. Its two docstring citations in `autonomous_brain` /
`autonomous_brain_recovered` annotated with the retirement rather than left naming a dead path.

### P2 governed-twin settlements 2026-08-06 — 2 pairs

| retired | survivor | what ported first |
|---|---|---|
| `reasoning/gwt_reentry.py` | `thalamus/reentrant_workspace.py` | an exogenous top-down PRIOR (`expect`). The governed twin assembled coalition drive by `max()` with no exogenous channel, so 0.40 evidence + 0.30 expectation could not sum past the 0.55 rheobase — predictive completion was structurally unreachable. Verified by neuter-A/B: no-op'ing `expect` flips the ported claim True→False and leaves the other four rows unmoved. |
| `reasoning/ng_snn/model.py` | `hippocampus/ng_snn.py` | nothing owed. Same mechanism (verified: identical grown size and held-out accuracy on 3 of 3 seeds with `l_max` pinned on both); the governed copy carries strictly more (`prune_unearned`, growth-state seeding, `patience`, `return_state`, derived-not-fixed capacity). |

---

## STRUCK from the P2 list — the classification was wrong

Each was named as a governed-twin duplicate and is not one. Recorded so the pair is not re-proposed.

| pair | why it is not a P2 debt |
|---|---|
| `brain_adapters/executive_control` ↔ `cortex/executive_control` | **Basename collision only**, and the file says so at its own `:22-32`. The governed module is ACC conflict monitoring + EVC + a tree planner; the reasoning file is three adapters over a developmental competence vector. |
| `brain_adapters/lake_memory` ↔ `substrate/lake_memory` | Different classes answering different questions (proximity-addressable replay vs SHY prune-and-reconstruct). **And the direction is inverted**: the reasoning copy carries a P4b cross-process durability check the governed one lacks (`cross_process` 7 / `subprocess` 2 / `sys.executable` 3 against 0 / 0 / 0). Porting that INTO `substrate/lake_memory.py` is a new capability, not a settlement. |
| `fly_hash` ↔ `generalizing_novelty` + `minhash` | **Both governed modules disclaim the relationship in their own docstrings.** `generalizing_novelty.py:15-19` and `minhash.py:22-24` each say they answer a different question. P2 requires a benchmark that exists to call one better; none shows the governed pair better, so there is no deletion to make. |

The two failure shapes to check for, both already caught here:
**different-question-same-vocabulary** (`readout_boost` vs `readout_fair_compare` — MNIST vision vs
next-phoneme language, sole overlap the string "SLDA-RFF") and **inferior-imports-superior**
(`readout_ensemble` imports the module claimed to supersede it).

---

## Deferred, with what each owes

| item | owes |
|---|---|
| `snn_motor_control` / `snn_world_model_rl` / `snn_dreamer_6dof` ↔ `cerebellum/jepa_world_model` | A shared task and metric — today they run a rendered 2-factor world, a 6-joint planar reach in metres, and a 2-DOF tanh update, with **zero overlap**. Plus an e-prop formulation: `.backward()` sites are jepa 0 / spiking_jepa 0 against 3 / 2 / 3, so adopting any of them reopens the cross-cutting invariant jepa closed. They are also a **ladder, not three duplicates** — `snn_world_model_rl` imports `fk` from `snn_motor_control`, so retiring the "inferior" breaks its own superseder. |
| `gwt_deep` ↔ `global_workspace` + `reentrant_workspace` | **The owed benchmark is PAID and the binding half is CREDITED (2026-08-07, run in the exact form that retired the workspace `bind_*` family).** Pre-registered the OPPOSITE of that family's null, because `gwt_deep`'s 2026-08-04 correction fixed both halves of the defect the family died of (per-slot-only drive; F1 scorer; the pre-fix cross-product broadcast reconstructed as `_reds()` and run every self-check). Measured, Jaccard, 120 trials/K, seed 0: **spiking 1.0000 at every K** — strictly above its own all-fired-pairs strawman at K>=2 (1.0 vs 0.5000 / 0.2500), which is the separation the retired family never had — against rate 0.5083 / 0.1829; the K=4 margin (+0.8171) is 3.8x the control's own spread (0.2175). The K=2 margin sits just inside one control sd, consistent with the coin-flip softness `gwt_deep`'s own `_reds()` docstring already discloses; K=4 is the load-bearing separation. **The MERGE half is DONE (2026-08-11, the measured pass it owed).** The ignition/re-entry half is retired from `gwt_deep`: the `IGNITE_TH` gate, the `_REENTRY` sustain cycles, both demos and both self-check asserts — `thalamus/reentrant_workspace` is the ONE ignition/re-entry mechanism (it asserts both properties with its own reachable reds, and had already absorbed `gwt_reentry`'s ported `::expect`). The binding half stands as the tree's phase-binding demonstration. Measured on BOTH sides of the diff with `docs/coverage/artifacts/gwt_deep_split_ab.py` (the credited Jaccard form, 120 trials/K, seed 0): spiking 1.0000 at every K, strawman 0.5000/0.2500 at K=2/4, rate row identical to the digit — the surgery moved the binding benchmark by ZERO. All four `_reds()` rows still reachable; the two sub-threshold-ignition reds became sub-RHEOBASE reds (drive 0.2 < LEAK*V_TH = 0.25), same criteria, same deterministic red. **PROMOTED 2026-08-11 (the separate pass this row named): `reasoning/gwt_deep.py` -> `thalamus/gwt_deep.py`**, the workspace family's region. Registered in REGIONS + `run_selfchecks.py` CORE; dormancy-EXEMPT "permanent" as a benchmark demonstration (nothing on the live tick binds K objects; consumers = its `__main__` with 4 reachable reds + `tests/instruments/gwt_deep_split_ab.py`, repointed in the same commit). Self-check green at the new path; the A/B instrument re-run post-move reproduces the credited figures (spiking Jaccard 1.0000 at every K, strawman 0.5000/0.2500). The one other tree mention (`global_workspace.py` verified-unaffected list) repointed with its pre-promotion path noted. This row is SETTLED — the first PROMOTE-lane row to complete. |
| `research/ng_snn/model.py` — **SETTLED 2026-08-07** | The third NG-SNN mechanism copy is DELETED; `hippocampus/ng_snn.py` is the one survivor. All 7 drivers repointed (`grow_head_nmnist`, `nmnist_pipeline`, `nmnist_spatiotemporal`, `nmnist_spatiotemporal_3conv`, `push_mnist`, `train_mnist`, `validate_uci` — the last imports `current_encode` too, which the old row under-counted). Direction checked BEFORE the swap: the governed copy exports all three needed symbols, its `l_max` gate is `is None` (so the three drivers pinning `l_max=0` pin, not derive), no module-scope side effects in either copy, all 7 compile against the new import and the identical repoint already runs in `reasoning/ng_snn/`. `research/README.md` re-derived: the 2026-08-04 blockers it recorded (broken governed drivers) had already cleared. **Remaining P2 debt is the DRIVER-set duplication** between `research/ng_snn/` and `reasoning/ng_snn/` — its own measured pass; `research/ng_snn/data.py` is load-bearing for the live perception tier, so no blanket deletion. |
| the remaining RETIRE-class holdouts, each now with a MEASURED disposition (2026-08-07) | **`deep_stdp_2stage_svm` — RETIRED** (11th of 16; row below). **`ensemble_rbf` — RECLASSIFIED out of RETIRE**: it is the harness that produced the tree's 98.2% Kheradpisheh-class headline (CHANGELOG, the "data lever" arc), and it imports `LaminarColumns` as a MEMBER it measures — the same inferior-imports-superior inversion `f299b631` recorded for `readout_ensemble`; `brain_adapters/laminar_readout.py:74` itself names it a consumer to repoint before any move. Its eventual lane belongs to the `laminar_readout` settlement. **`readout_boost` and `readout_ensemble` — RECLASSIFIED out of RETIRE**, promoting `f299b631`'s findings from commit message to ledger: the first's claimed superseder answers a different question (MNIST vision vs next-phoneme language; sole overlap the string "SLDA-RFF"), the second imports its claimed superseder as a measured member. **`autonomous_brain` — KEPT**: the control half of a live A/B whose treatment half (`autonomous_brain_recovered`) exists only to explain it. **`snn_mnist_stdp_genn` — MANDATED BUILD (owner, 2026-08-07): FINISH THE PORT.** The finish-or-retire question was put to the owner and the answer is finish. It leaves the RETIRE class permanently — the 16 RETIRE resolves as: 11 retired, 3 reclassified (`readout_boost`, `readout_ensemble`, `ensemble_rbf`), 1 kept as an A/B control (`autonomous_brain`), 1 mandated build (this). What finishing owes, so the lane that picks it up starts from the ledger and not from archaeology: the `[genn]` extra stood up (pygenn + C++ toolchain + CUDA — no other consumer exists yet); the `burst_xtar_rule` GeNN codegen completed (the tree's only copy); then the full P4c entry bar — REGIONS + `run_selfchecks.py` registration, reachable from a live path or EXEMPT-with-reason, the module's own self-check with reachable reds, gates green. A real build of this size takes the spec-first path before implementation. **BUILT 2026-08-11 (specs/012, five commits `420f25b3..`): every owed item PAID.** The `[genn]` extra stands up (pygenn 5.4.0 compiled for the free-threaded ABI; the box's CUDA 13.3 install was INCOMPLETE — crt/, nvvm/cicc, cudart all absent — fixed with a local toolkit at `D:\AI-project\cuda-local` assembled from NVIDIA PyPI wheels, no admin); the codegen is complete and the module lives at `plasticity/snn_mnist_stdp_genn.py`, registered, EXEMPT-with-reason (train-tier opt-in), self-check with 7 reachable reds (an 8th assert was a derivation that round-trips and was DELETED per P8). Measured: toy-scale mirror EXACT (counts/theta/weights, 2 runs bitwise); parity +0.7/+1.7/+0.7pp across M=300-6k, M=300-60k, M=1600-60k — all inside the pre-registered ±3pp; GPU train 3.5-4.2x CPU; M=1600/60k = **0.846 GPU vs 0.839 CPU**, the rule family's best full-MNIST figure here. Honest scale finding: at M=300 BOTH sides collapse on 60k (0.55/0.53 — single-epoch theta accumulation, the rule's lane, not the port's). Evidence: `specs/012-finish-genn-port/evidence.md`. |
| `cortex/theta_gamma_code.py` dormancy (measured 2026-08-07) | **Zero non-self-check callers tree-wide**: the only consumer is `rose_language.order_by_phase` (lazy in-method import), whose only caller is `rose_language.main()` — its own self-check, in the tier pytest excludes. Every other tree-wide mention is a docstring distinct-from note or the REGIONS registration; `importlib`/`__import__`/string-dispatch swept, no hits. **That is a P3 dormancy question and it comes BEFORE any P2 merge question** — wire it into something whose output it changes, or retire it; deciding a merge for a buffer nothing consumes decides nothing. |
| `_settle_chunk` — one more inline LIF declaration (measured 2026-08-07) | `thalamus/global_workspace.py` imports no `LIFNeuron` (itertools/random/threading/time + region_wiring only) and declares its LIF dynamics inline; `thalamus/reentrant_workspace.py:234` carries ANOTHER inline kernel (still on its own `_G_INH = 2.4`, pre-rheobase); `thalamus/reticular_gate.py` next door composes `cortex/lif_neuron.LIFNeuron` properly, so the reuse pattern exists in-region. **MERGE candidate, never a bare ADOPT**: the primitive is exponential-leak / reset-to-value / floored, `_settle_chunk` is linear-leak / subtract-threshold / unfloored — and subtract-threshold is what makes `-_last_v` a linear drive-excess readout, which is what closed the 1,941-name-flip defect. A bare swap re-opens it; the merge owes an A/B and a `LIFNeuron` extension. |
| the three-way P2 debt: `GlobalWorkspace` binding ↔ `bind_*` family ↔ `ThetaGammaCode` | **REFUTED 2026-08-07, same day the trigger was recorded — the benchmark was run and returned the pre-registered null.** Jaccard, 120 trials/K: spiking 1.0000/0.5000/0.2500 at K=1/2/4 — EQUAL to the all-fired-pairs strawman at every K (the decoder returned the cross product; phase multiplexing recovered membership, never pairing); at K=2 the rate control was nominally better; no margin cleared the control's own spread. The `bind_*` family is DELETED (measurement preserved at the retirement record in `global_workspace.py`); there is no measured binding capability to merge, so the three-way merge is refuted, not deferred. |
| the `deep_stdp` lineage: `snn_mnist_stdp` -> `deep_stdp` -> `deep_stdp_latency` (measured 2026-08-11, batch-1) | **All three PROMOTE, in dependency order; no governed twin** — graph-exists + SRC_INVENTORY sweeps put the nearest governed neighbours (`plasticity/snn_mnist_stdp_genn`, `plasticity/inhibitory_stdp`, `basal_ganglia/r_stdp`) each on a different question, and the adversarial refute pass upheld both PROMOTEs. RETIRE is structurally impossible below the top: `deep_stdp_latency.py:99` and `deep_stdp_2stage.py:95` import `FcStdp/OneLayerNet/assign_labels/evaluate` from `deep_stdp`, and `snn_mnist_stdp` is a 15-importer hub (autonomous_brain:45, every `developmental_loop_*`, ensemble_rbf:31, fast_features:61, readout_boost:27, readout_ensemble:40, snn_mnist_stdp_fast:23, deep_stdp:71 itself) — **and it already has a GOVERNED lazy consumer: `plasticity/snn_mnist_stdp_genn.oracle()` imports it and `snn_mnist_stdp_fast`**, so the hub's promotion also closes a governed->reasoning reach. Ordered obligations: (1) `snn_mnist_stdp` first (else a governed `deep_stdp` imports from ruff-excluded reasoning/ — the import_manifest hazard); its unguarded snntorch+torchvision top-levels take the genn-precedent EXEMPT train-tier shape. (2) `deep_stdp` -> `plasticity/`; measured frictions: unguarded torch/snntorch at :66-68, stale `.venv-cu128` command at :58, `_env/_envf` duplicated from snn_mnist_stdp:63-68, `__main__` is a multi-minute train with a torchvision download — needs a toy-scale smoke path (specs/012 precedent) to be gate-runnable; its one assert (`acc1 >= 0.5`, :352) has a reachable red. (3) `deep_stdp_latency` last; its ~10 reasoning importers include the live perception tier (`autonomous_brain`), and its region call (cortex/ feature hierarchy vs plasticity/ rule) is NOT settled — settle at execution. `deep_stdp`'s ConvStdp half is a recorded honest negative (filter collapse, cos-sim ~0.96; CHANGELOG v3.49) licensing the latency machinery; any later within-lineage consolidation owes only the 4-symbol library port. |
| the physics-JEPA trio: `physics_jepa` / `physics_jepa_highdim` / `double_pendulum_jepa` ↔ `cerebellum/jepa_world_model` (measured 2026-08-11, batch-1) | **Settle in ORDER: the winner's MERGE first, then the two RETIREs.** `physics_jepa_highdim` = MERGE-THEN-RETIRE into the governed jepa lane (upheld; it reuses `JepaWorldModel` and adds the high-dim+noise probe — the honest positive its linear predecessor could not show). `physics_jepa` = RETIRE (honest negative preserved in CHANGELOG: learned latent does not beat a random encoder on a near-linear world) but import-BLOCKED today: `physics_jepa_highdim.py:31` imports `gen_dynamics, ACTIONS` from it (hand-verified — the ladder shape), and `physics_jepa:43` imports `skill_builder.signature`, coupling this trio to the skills family. `double_pendulum_jepa` = RETIRE (completed experiment, +60% on chaotic dynamics; sole non-history mention is a printed string at meta_skill_loop:302). DISTINCT from the `snn_motor_control`/`snn_world_model_rl`/`snn_dreamer_6dof` row above — different modules, same deferral logic. |
| the skills family: `content_signature` + `skill_builder` (PROMOTE), `meta_skill_loop` (MERGE-THEN-RETIRE), + the `nightly_skills`/`open_meta_loop` P2 already recorded in specs/010 (measured 2026-08-11, batch-1) | `skill_builder` PROMOTE -> `hippocampus/` (upheld; `SkillRegistry` routes over the durable lake, live via `nightly_skills.py:122` <- `cortex/brain.py:3069`) — but it imports `content_signature` (:44), so that moves first or same-commit (`spike_chat_lake:54` also imports it; repoint). `meta_skill_loop` MERGE-THEN-RETIRE (upheld) with THREE consumers to repoint first: `open_world_loop:46`, `open_meta_loop:37`, `nightly_skills:120` (hand-verified); its merge partner is the `open_meta_loop`/`nightly_skills` mechanism that specs/010's promotion-plans already recorded as "P2 UNSETTLED — the promotion is additive as it stands". The family settles TOGETHER — a per-module pass re-litigates the same imports five times. |
| `kb_ranking` — **PROMOTED 2026-08-11** (measured batch-1, executed same day) | **DONE: `reasoning/kb_ranking.py` -> `cortex/kb_ranking.py`.** The pre-flight CAUGHT what it asked for: `load_dataset` lazily imports `reasoning.world_kb_stream` (the offline fallback), so the move creates a governed->reasoning lazy reach — the exact shape `substrate/import_manifest` exists for, and the manifest DERIVES it (AST walk, no hand edit): 15 cross-tier targets post-move, all import. Registered REGIONS + `run_selfchecks` CORE; dormancy gate green with NO exemption (208 modules, 0 dormant). All four consumers repointed (`meta_skill_loop`, `nightly_skills`, `schema_induction`, `reasoning_cortex`). Self-check measured at promotion: 12.5s with cached FB15k-237, Hits@10 0.297 / MRR 0.136 = 431x/194x random, both asserts with reachable reds (sabotaged training ranks at random). One ruff fix at the governed bar (B007 `ep` -> `_ep`). Follow-on owed by whichever lane settles `world_kb_stream`: its row now carries a governed consumer. |

| `brain_adapters/brain_decision` (measured 2026-08-11, batch-2) | **PROMOTE -> `basal_ganglia/`, CONDITIONAL on landing on a live path** — sole importer is ungoverned `developmental_loop_brain.py:67`, so promoted alone it reds the dormancy gate and an EXEMPT would rebuild the parking lot; promote with/wired-from its consumer. Composition verified (only owned logic: minmax-norm, the alpha fusion, competence mapping, seeded tie-break; LatentQ and MetacognitiveMonitor composed, not re-implemented; no governed twin — graph-exists returns 2 files, both reasoning/). BONUS the row exists for: `method_dormancy_gate:113-121` records `LatentQ.update` as an OPEN "wire it and prove the charge" item, and `brain_decision.observe()` is the tree's only online trainer of it — promotion CLOSES that item. Frictions: P1 hardcodes (alpha 0.8/0.2, lr 0.1); the FUSION itself has no neuter-A/B (components proven to move, headline capability never proven to charge — owed at promotion). |
| `brain_adapters/growable_readout` (measured 2026-08-11, batch-2) | **PROMOTE -> `hippocampus/`** — no governed twin implements torch-tier function-preserving growth (Net2Net (1,0) split, Adam moments in lockstep) or freeze-isolation; RETIRE blocked by live reasoning importers (`autonomous_brain:49/:108` — the KEPT A/B control; `developmental_loop_brain:66/:244` opt-in honest negative). **The promotion doubles as the specs/007 T022 consolidation point** (tasks.md:100, unchecked): `laminar_readout:38/:188/:684` copied the freeze + split verbatim FROM this file — this is the origin copy the duplicated splits collapse into. Dormancy: gwt_deep EXEMPT precedent (consumed only by ungoverned code post-move). |
| `brain_adapters/hippocampal_memory` (measured 2026-08-11, batch-2) | **PROMOTE -> `hippocampus/`** — adapter-over-governed in the intended direction (per-digit governed ReplayBuffers; nothing governed measured as a member). Live consumer `developmental_loop_brain` uses all three methods on its main path, and that consumer is p6_gate-pinned (KNOWN_WRITERS), so ITS lane is promotion — settle both in one lane, else `method_dormancy_gate` flags store/consolidate until the loop follows. Self-check seconds-fast, ruff-clean (measured). |
| `brain_adapters/laminar_readout` ↔ `laminar_local_readout` (measured 2026-08-11, batch-2) | **MERGE-THEN-RETIRE** — the live path already runs the superseder (`cortex/brain.py:1382-1386` constructs LaminarLocalReadout; nothing governed touches this module), and the e-prop invariant makes PROMOTE-as-is the wrong direction (Adam/backprop through a reverse-mode tape on both classes). specs/010's promotion-plans records the ADOPT-deferred port-before-retire blockers (re-verified). Importers to repoint at execution: `developmental_loop_brain:73` (both classes, behind NORD flags, default arm _LinearReadout), `ensemble_rbf:33` (LaminarColumns; its lane already owed this settlement per the RETIRE-holdouts row). Couples to the `growable_readout` promotion (T022 origin-copy consolidation). |
| `language_self_build` (measured 2026-08-11, batch-2) | **PROMOTE -> `cortex/` — LIVE ON THE DAEMON'S SLEEP TICK, runtime-proven**: py-spy of the live daemon shows `_positional_code` on the stack via brain.tick -> _enter_sleep -> nightly_skills.night -> meta_skill_loop.study. Seven reasoning-tier importers + `rose_language:475` lazy-imports it on a path `cortex/brain.py` reaches; specs/010's rose_language pre-flight already records that rose's promotion gives this file its first governed importer. RETIRE structurally impossible. Frictions, measured: `__main__` takes 324 s (needs a toy-scale smoke path before CORE registration — specs/012 precedent); module-top imports of `corpus_stream` + `brain_adapters/lake_memory` mean those promote first/same lane. **This row merges into the skills/language family settlement above** — the family keeps growing: corpus_stream, lake_memory, language_readout_slda, readout_fair_compare's ports all converge there. |
| `readout_fair_compare` (measured 2026-08-11, batch-2) | **RETIRE with PORT-FIRST obligations — defers to the language-family settlement.** Completed experiment (CHANGELOG "READOUT STUDIES"), zero importers, harness-over-governed (shape 2: measures SpikingInteractionLearner + SLDA_RFF_np as subjects). What must port before deletion: (1) `_ngram_acc(train, test, k)` — the equal-budget n-gram baseline; `language_self_build._bigram_baseline` fits uncapped and has no trigram; (2) the equal-budget assert discipline — `language_readout_slda:184` compares SLDA capped at per=6000 against an UNCAPPED bigram, and the corrected same-budget comparison exists ONLY in this file; (3) do NOT port `rstdp_degrades` (it pins a pathology of governed interaction_learning — keep as the CHANGELOG finding only). |

### The language/skills FAMILY — fully measured 2026-08-11 (batch-3 closes the set); ONE settlement pass

Batch-3 measured the nine remaining members (two-stage adversarial pass, all upheld). With the
batch-1/2 rows above, every member now has a verdict; the settlement is one ordered pass:

| member | verdict (batch) |
|---|---|
| `content_signature` — **PROMOTED 2026-08-11 (C1, executed)** | DONE: `reasoning/` -> `hippocampus/`. Both promotion obligations paid: pinned known-good bins assert added (dog->61, cat->40, COMPUTED then pinned — the persistence-stable red is now reachable: swapping crc32 for builtin hash() or changing SIG reds it), and `SIG = 64` carries the persistence-format comment (P1 exception 3 shape: changing it orphans every stored key). SIBLING DEBT recorded, not silently adopted: NO single named authority for the tree's N=64 channel width exists (multimodal_encoder's 4x16 bands, streaming_hub's `channel < 64`, esp32 path all carry it as literals) — consolidating that is its own P1 pass. Three importers repointed (skill_builder, spike_chat_lake, nightly_skills), run_selfchecks entry moved RESEARCH -> CORE at the new path, REGIONS registered, dormancy green (reachable, no exemption). (3) |
| `corpus_stream` — **PROMOTED 2026-08-11 (C2a, executed)** | DONE: `reasoning/` -> `cortex/` (language-specific ingestion). Verified at REAL scale at the move (126,052 CMUdict entries / 128,577 tokens / 458,529 phoneme pairs — the module's own "NOT toy scale" contract held). TWO asserts DELETED per F11, verified true-by-construction by hand before deleting: ARPABET membership (load_cmudict filters to ARPABET at source) and the context<=k bound (the slice arithmetic asserting itself) — red-unreachable from any input. Nine importers repointed (8 reasoning + `training/ar_spiking_lm.py:360`), REGIONS + run_selfchecks CORE, dormancy green no exemption, manifest 15/15. Its `nightly_skills` consumer edge sits inside `brain.py:3079`'s bare except — the silent-kill hazard closes fully when nightly_skills itself promotes (C9). (3) |
| `world_kb_stream` — **PROMOTED 2026-08-11 (C2b, executed)** | DONE: `reasoning/` -> `cortex/` beside kb_ranking (the KB-completion lane's two halves reunited governed). kb_ranking's governed->reasoning reach is CLOSED (its lazy fallback now governed->governed; docstring updated). Disclosed and derived: world_kb_stream's own main() lazily imports `reasoning.language_readout_slda` + `reasoning.language_self_build` — two NEW cross-tier reaches (manifest 15 -> 16, all import), both closed by C5/C6. Four importers repointed (rose_language, spike_chat_lake, kb_ranking, tools/lang_coverage), REGIONS + run_selfchecks CORE (0.4s, one real assert vs the per-relation most-frequent baseline — P8's correct control shape, measured 50.0% vs 38.9%), dormancy green no exemption. (3) |
| `brain_adapters/lake_memory` — **PROMOTED+RENAMED 2026-08-11 (C3, executed): `hippocampus/proximity_replay.py`** | DONE, renamed at the move because brain_atlas asserts basename uniqueness across REGIONS and `substrate/lake_memory` holds the name (different question per the STRUCK row; class stays `LakeMemory`). THREE fixes landed with the move, each verified: (1) the P4b cross-process child's `-m` target was a HARDCODED old dotted path — now derived from `__spec__.name` (`type(self).__module__` was tried first and is WRONG under `-m`, where it reads `__main__`; caught by the check itself before commit); (2) the dead `durable=` sink + `rebuild_from_durable()` DELETED per P3 (zero of 14 constructor sites ever passed it; writer retired 2026-07-29 — `home.py`'s orphan note closed in the same commit; the durable-MEMORY gap they addressed stays recorded, rebuilding it starts from the one lake); (3) 12 importers repointed. run_selfchecks CORE (0.5s: 100% proximity recall vs 25% chance, close/reopen durability, P4b cross-process readback with BOTH constructed reds firing). The port-INTO-substrate question (giving `substrate/lake_memory` the cross-process check) remains its own row, unowned here. (3) |
| `skill_builder` | PROMOTE -> hippocampus/ after content_signature (1) |
| `language_readout_slda` | PROMOTE -> cortex/ — inherits readout_fair_compare's equal-budget assert port. (3) |
| `language_self_build` — **PROMOTED 2026-08-11 (C6, executed out of recorded order — the order was WRONG)** | DONE: `reasoning/` -> `cortex/`. **Order correction, measured not assumed**: the recorded sequence put skill_builder before this module, but skill_builder imports it (and language_readout_slda) at module TOP — promoting skill_builder first would have created governed->reasoning module-top imports. Corrected order: C6 -> C5 -> C4. Gate runtime honestly, per the not-toy-scale directive: corpus/mechanism/controls/asserts all UNCHANGED and real-scale; the 324s was ~1.1M spiking respond() calls across six ~91k-pair held-out evals, and the ONE bound added is `NORD_EVAL_N` (default 2000, `0` = full tail) on the eval SAMPLE — measured post-move: **17.7s with every figure matching the recorded full-tail values** (separations 56.9/52.1 identical, phonetics 7.4% vs shuffle 1.9%). Also landed: the Windows-only `TEMP` cache-path defect fixed (same fix as lake99/autonomous_brain), 7 importers repointed, world_kb_stream's self_build reach closed (manifest 16 -> 15), REGIONS + run_selfchecks CORE, dormancy green (live on the sleep tick). (2) |
| `open_world_loop` | PROMOTE -> hippocampus/ — fix its inline `% 64` duplication of content_signature's binning at the move. (3) |
| `nightly_skills` | PROMOTE -> cortex/ — the sleep-tick entry `brain.py:3069` reaches; last of the promotes (imports half the family). (3) |
| `meta_skill_loop` | MERGE-THEN-RETIRE (1) — merge partner is the open_meta_loop/nightly_skills mechanism |
| `open_meta_loop` | MERGE-THEN-RETIRE (3) — the specs/010 "P2 UNSETTLED, additive" pair resolves in the same merge |
| `readout_fair_compare` | RETIRE after its three ports (2) |
| `lake_skill_library` | RETIRE (3) — with `lake_skills` already deleted, the "exactly one survives" pair resolves to NEITHER: the routing capability's live home is `skill_builder.SkillRegistry` over the durable lake; verify that claim at execution before deleting. |

Order: content_signature -> {corpus_stream, world_kb_stream, lake_memory} -> skill_builder ->
{language_readout_slda + fair_compare ports} -> language_self_build -> open_world_loop ->
{meta_skill_loop + open_meta_loop merge} -> nightly_skills -> lake_skill_library retire.
Each step at the full entry bar, gates green per commit; the family's row details live in the
batch-1/2 rows above and the batch-3 digests (session scratchpad, superseded by execution rows
as each lands).

**Instrument defect, recorded 2026-08-11:** `specs/010-policy-compliance-audit/promotion-plans.json`
is TRUNCATED at exactly 400,000 bytes — unparseable JSON, the last entry cut mid-string, and the
17-module blocked-promotion record it carries is this lane's prior art. Its content also repeats
(the same modules appear 2-3x, some with empty blocker arrays). Regenerate or repair before the
PROMOTE lane leans on it; until then it is quotable only via lenient text extraction.

---

## Deletion hazards — check ALL of these, the obvious list is incomplete

Before deleting any `reasoning/` module, grep it in every one of:

`run_selfchecks.py` · `src/snn_data_lake/__init__.py` (REGIONS / TORCH_TIER / `_MOVED`) ·
`substrate/dormancy_gate.py` EXEMPT · `substrate/method_dormancy_gate.py` EXEMPT_METHOD ·
`substrate/import_manifest.py` · `pyproject.toml` · `tests/` · **`tools/p6_gate.py`** ·
`docs/` command blocks

**`tools/p6_gate.py` is the one the obvious checklist misses, and it ASSERTS IN CI.** It hard-codes
`reasoning/` paths with writer floors — `KNOWN_WRITERS["reasoning/brain_adapters/executive_control.py"] = 3`
at `:271`, consumed at `:1328-1345` where a known writer producing zero writes appends a problem, plus
a dedicated criterion at `:1381`. Same for `train_of_thought` (4), `rpm_solver` (1),
`spike_chat_lake` (1), `developmental_loop_brain` (1), and `KNOWN_READERS` for two of them. **Deleting
one of those files without editing `p6_gate.py` in the same commit reds the build.**

Also non-obvious, and each has actually fired: `substrate/brain_geometry.py:170-179` globs
`reasoning/*.py` into an RNG draw order (a deletion shifts module coordinates in the cockpit's 3D map);
`tools/doc_command_guard.py` reddens if a documented `python -m` command names a deleted module;
`tools/dependency_ledger.py` reddens if a deletion strands the last importer of a declared
distribution (this is why `eth_mnist_bindsnet` is a NO-GO — it is the sole `bindsnet` importer).

**Two instruments read the git INDEX, not the disk** — `tools/p4c_ratchet.py` and
`tools/citation_guard.py`. A lane that deletes from the filesystem without staging cannot make either
of them move, so a retirement is verifiable only by the disk-reading instruments until someone stages.

---

## Rules learned the hard way

- **A fix is not a capability, so it survives no capability audit.** `snn_storage_core_snntorch` had
  corrected a snnTorch idiom (`init_hidden=False`) that its superseder never knew was wrong; deleting
  it left the defective version as the only surviving copy. Ask of every retiree: did it CORRECT
  something, not just DO something.
- **Check the direction before believing a supersession.** A verdict claimed the `src/` ng_snn copies
  were ahead of the `research/` ones; diffing showed the opposite.
- **An import census tells you which name failed FIRST, never what a module needs to LOAD.** "28
  failures share one root cause" was overstated 7x; removing the cause unblocked 6.
- **"Never imported" from AST tooling is a hypothesis.** Sweep `importlib`, `__import__`, `pkgutil`,
  `getattr`, string-name dispatch and the `_AliasFinder` layer, and match at SYMBOL level with word
  boundaries — a caller can import a symbol without naming the module, and a substring match falsely
  ties `foo` to a surviving `foo99`.
- **A gate metric that moves on a deletion is not automatically a finding.** `red_reachability` went
  13→12 on a name-shaped false positive (`_present` matching `CRITERION_SUFFIXES`).
