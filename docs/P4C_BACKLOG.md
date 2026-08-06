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
| `gwt_deep` ↔ `global_workspace` + `reentrant_workspace` | **Split verdict.** Its ignition/re-entry half genuinely duplicates `reentrant_workspace`; its headline — theta-gamma phase multiplexing to bind K feature pairs, with a spiking-vs-rate control — does not, and neither `cortex/theta_gamma_code.py` nor `cortex/crossmodal_binder.py` answers it. Owes a K-pair binding benchmark, then MERGE: retire the duplicated half, keep the binding half. |
| `research/ng_snn/model.py` | A **third** copy of the NG-SNN mechanism, formatting-only different from the retired reasoning copy. P2 says one survives and there are two. Owes: repoint 7 bare `from model import NGSNN` drivers, update `research/README.md:41-49`, delete. Left for its own pass — 8 mechanical edits at the end of a lane is how an asserted port gets recorded as discharged. |
| the remaining 7 of the 16 RETIRE | `readout_boost`, `readout_ensemble`, `autonomous_brain`, `eth_mnist_bindsnet`, `snn_mnist_stdp_genn`, `deep_stdp_2stage_svm`, `ensemble_rbf` — see the retirement commit for the per-module reason each is a NO-GO. |

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
