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
| `gwt_deep` ↔ `global_workspace` + `reentrant_workspace` | **The owed benchmark is PAID and the binding half is CREDITED (2026-08-07, run in the exact form that retired the workspace `bind_*` family).** Pre-registered the OPPOSITE of that family's null, because `gwt_deep`'s 2026-08-04 correction fixed both halves of the defect the family died of (per-slot-only drive; F1 scorer; the pre-fix cross-product broadcast reconstructed as `_reds()` and run every self-check). Measured, Jaccard, 120 trials/K, seed 0: **spiking 1.0000 at every K** — strictly above its own all-fired-pairs strawman at K>=2 (1.0 vs 0.5000 / 0.2500), which is the separation the retired family never had — against rate 0.5083 / 0.1829; the K=4 margin (+0.8171) is 3.8x the control's own spread (0.2175). The K=2 margin sits just inside one control sd, consistent with the coin-flip softness `gwt_deep`'s own `_reds()` docstring already discloses; K=4 is the load-bearing separation. **The MERGE half of the split verdict is now ACTIVE**: retire the ignition/re-entry half that duplicates `reentrant_workspace`, keep the binding half as the tree's honest phase-binding demonstration. Its own measured pass — the file's self-check interleaves both halves, so the split is surgery, not deletion. |
| `research/ng_snn/model.py` — **SETTLED 2026-08-07** | The third NG-SNN mechanism copy is DELETED; `hippocampus/ng_snn.py` is the one survivor. All 7 drivers repointed (`grow_head_nmnist`, `nmnist_pipeline`, `nmnist_spatiotemporal`, `nmnist_spatiotemporal_3conv`, `push_mnist`, `train_mnist`, `validate_uci` — the last imports `current_encode` too, which the old row under-counted). Direction checked BEFORE the swap: the governed copy exports all three needed symbols, its `l_max` gate is `is None` (so the three drivers pinning `l_max=0` pin, not derive), no module-scope side effects in either copy, all 7 compile against the new import and the identical repoint already runs in `reasoning/ng_snn/`. `research/README.md` re-derived: the 2026-08-04 blockers it recorded (broken governed drivers) had already cleared. **Remaining P2 debt is the DRIVER-set duplication** between `research/ng_snn/` and `reasoning/ng_snn/` — its own measured pass; `research/ng_snn/data.py` is load-bearing for the live perception tier, so no blanket deletion. |
| the remaining RETIRE-class holdouts, each now with a MEASURED disposition (2026-08-07) | **`deep_stdp_2stage_svm` — RETIRED** (11th of 16; row below). **`ensemble_rbf` — RECLASSIFIED out of RETIRE**: it is the harness that produced the tree's 98.2% Kheradpisheh-class headline (CHANGELOG, the "data lever" arc), and it imports `LaminarColumns` as a MEMBER it measures — the same inferior-imports-superior inversion `f299b631` recorded for `readout_ensemble`; `brain_adapters/laminar_readout.py:74` itself names it a consumer to repoint before any move. Its eventual lane belongs to the `laminar_readout` settlement. **`readout_boost` and `readout_ensemble` — RECLASSIFIED out of RETIRE**, promoting `f299b631`'s findings from commit message to ledger: the first's claimed superseder answers a different question (MNIST vision vs next-phoneme language; sole overlap the string "SLDA-RFF"), the second imports its claimed superseder as a measured member. **`autonomous_brain` — KEPT**: the control half of a live A/B whose treatment half (`autonomous_brain_recovered`) exists only to explain it. **`snn_mnist_stdp_genn` — MANDATED BUILD (owner, 2026-08-07): FINISH THE PORT.** The finish-or-retire question was put to the owner and the answer is finish. It leaves the RETIRE class permanently — the 16 RETIRE resolves as: 11 retired, 3 reclassified (`readout_boost`, `readout_ensemble`, `ensemble_rbf`), 1 kept as an A/B control (`autonomous_brain`), 1 mandated build (this). What finishing owes, so the lane that picks it up starts from the ledger and not from archaeology: the `[genn]` extra stood up (pygenn + C++ toolchain + CUDA — no other consumer exists yet); the `burst_xtar_rule` GeNN codegen completed (the tree's only copy); then the full P4c entry bar — REGIONS + `run_selfchecks.py` registration, reachable from a live path or EXEMPT-with-reason, the module's own self-check with reachable reds, gates green. A real build of this size takes the spec-first path before implementation. **BUILT 2026-08-11 (specs/012, five commits `420f25b3..`): every owed item PAID.** The `[genn]` extra stands up (pygenn 5.4.0 compiled for the free-threaded ABI; the box's CUDA 13.3 install was INCOMPLETE — crt/, nvvm/cicc, cudart all absent — fixed with a local toolkit at `D:\AI-project\cuda-local` assembled from NVIDIA PyPI wheels, no admin); the codegen is complete and the module lives at `plasticity/snn_mnist_stdp_genn.py`, registered, EXEMPT-with-reason (train-tier opt-in), self-check with 7 reachable reds (an 8th assert was a derivation that round-trips and was DELETED per P8). Measured: toy-scale mirror EXACT (counts/theta/weights, 2 runs bitwise); parity +0.7/+1.7/+0.7pp across M=300-6k, M=300-60k, M=1600-60k — all inside the pre-registered ±3pp; GPU train 3.5-4.2x CPU; M=1600/60k = **0.846 GPU vs 0.839 CPU**, the rule family's best full-MNIST figure here. Honest scale finding: at M=300 BOTH sides collapse on 60k (0.55/0.53 — single-epoch theta accumulation, the rule's lane, not the port's). Evidence: `specs/012-finish-genn-port/evidence.md`. |
| `cortex/theta_gamma_code.py` dormancy (measured 2026-08-07) | **Zero non-self-check callers tree-wide**: the only consumer is `rose_language.order_by_phase` (lazy in-method import), whose only caller is `rose_language.main()` — its own self-check, in the tier pytest excludes. Every other tree-wide mention is a docstring distinct-from note or the REGIONS registration; `importlib`/`__import__`/string-dispatch swept, no hits. **That is a P3 dormancy question and it comes BEFORE any P2 merge question** — wire it into something whose output it changes, or retire it; deciding a merge for a buffer nothing consumes decides nothing. |
| `_settle_chunk` — one more inline LIF declaration (measured 2026-08-07) | `thalamus/global_workspace.py` imports no `LIFNeuron` (itertools/random/threading/time + region_wiring only) and declares its LIF dynamics inline; `thalamus/reentrant_workspace.py:234` carries ANOTHER inline kernel (still on its own `_G_INH = 2.4`, pre-rheobase); `thalamus/reticular_gate.py` next door composes `cortex/lif_neuron.LIFNeuron` properly, so the reuse pattern exists in-region. **MERGE candidate, never a bare ADOPT**: the primitive is exponential-leak / reset-to-value / floored, `_settle_chunk` is linear-leak / subtract-threshold / unfloored — and subtract-threshold is what makes `-_last_v` a linear drive-excess readout, which is what closed the 1,941-name-flip defect. A bare swap re-opens it; the merge owes an A/B and a `LIFNeuron` extension. |
| the three-way P2 debt: `GlobalWorkspace` binding ↔ `bind_*` family ↔ `ThetaGammaCode` | **REFUTED 2026-08-07, same day the trigger was recorded — the benchmark was run and returned the pre-registered null.** Jaccard, 120 trials/K: spiking 1.0000/0.5000/0.2500 at K=1/2/4 — EQUAL to the all-fired-pairs strawman at every K (the decoder returned the cross product; phase multiplexing recovered membership, never pairing); at K=2 the rate control was nominally better; no margin cleared the control's own spread. The `bind_*` family is DELETED (measurement preserved at the retirement record in `global_workspace.py`); there is no measured binding capability to merge, so the three-way merge is refuted, not deferred. |

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
