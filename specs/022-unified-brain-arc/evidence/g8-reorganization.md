# G8 / T116 -- the repository reorganized by engine function (pre-registration)

Row: plan Part D, G8 / T116, "reorganize the repository to the engine": packages follow the engine's components (the
brain and its grower, reservoir, readout, sleep, lake, infrastructure), never brain anatomy; the brain's regions or
columns live on grown neurons (G7). Red: Part F's G-R8. Rule: POLICIES P4d, "the repository is reorganized to the
engine's components (plan G8); each move re-points every path-keyed rule in the same commit". The owner's words
(2026-09-25, verbatim): "Weights transport is allowed as long as the engine can work, the repo should be reorganized to
fit the new structure as well. 2 networks are allowed as long as it fits the new engine structure"; "The brain regions
should be neurons and not coded like the current work tree structure on github, the engine grows the neurons and
attaching them over time, it could be either cortical columns or anatomically accurate brain, I don't know which
structure to consider since we need to do more research on the brain design philosophy through web grab". The owner's
four G8 decisions are quoted in section 2.1.

Status: **PRE-REGISTERED -- FROZEN 2026-09-25 at the design read's fold (the commit that lands this text); dated
amendments only after this line.** Nothing of G8 has been built or run. Section 2 is frozen from that commit, before
S0's first gate run; section 3 records every stage.

Source: the G8 design draft, a read-only Claude Opus design measured at `4b237c6` (archived verbatim in the memory
store, `archive/engine-research-audit-2026-09-25/g8_design_draft.md`). This file re-derives every count and hazard the
draft relies on at HEAD `68db184` (section 1), states what did not survive the re-derivation (section 1.11) and carries
the design forward with those corrections (section 2). The design read of this pre-registration (18 findings, F1-F18;
the orchestrator reproduced every claim at `f416296`, whose `src/` equals `68db184`'s, and judged each fix separately
from its claim) is folded into sections 1 and 2 and ledgered in section 2.13, with the orchestrator's rulings at the
fold. Codex was not used (the owner's hold of 2026-09-25). The slot is
a session call under the 2026-09-05 authorization: after G1(b)'s selection record, its losing-port deletion and the
G1(a)+G1(b)+docs stack's battery and push (done: `68db184`), before G1(c).

## 1. Territory, re-verified at HEAD `68db184`

Every instrument below was first run on the draft's own tree (`git archive 4b237c6`, exported to a scratch directory)
and had to print the draft's figures there before a figure at `68db184` was believed. It did, except where section 1.11
names an instrument-definition difference. The blocks B1-B3 in section 1.12 print the numbers of sections 1.2-1.9 and
1.11 on the checkout; the gates and the searches listed in 1.12 print the rest. Read-only throughout: no index write, no
live lake (B2 creates its own temporary `NORD_HOME` and asserts `home()` resolves under it before anything is built),
no daemon.

### 1.1 The tree

- HEAD `68db184` = `origin/main` (local ref); `git status --short` prints nothing.
- 24 commits since the draft's `4b237c6` (`git log --oneline 4b237c6..68db184`). In `src/` they touch five modules and
  one registry comment (`git diff --stat 4b237c6 68db184 -- src`): `cortex/recurrent_pool.py` (the losing ports g2-g4
  deleted at `440a6a5`), `cortex/pool_selection.py`, `cortex/memory_diagnostic.py`, `substrate/doc_budget_gate.py`,
  `substrate/dormancy_gate.py` (two EXEMPT reasons reworded, no key changed) and a comment in `__init__.py`. No module
  was added or removed. The rest is documents: POLICIES.md (P4d), the plan, spec, tasks, the constitution, AGENTS.md,
  AGENT_COLD_START.md, POLICIES_EVIDENCE.md, the G1(b) record and two new evidence files (tracked files 1,246 -> 1,248).

### 1.2 Modules, the registry and the destinations (unchanged)

- `git ls-files src/`: 454 files = 438 `.py` (24 `__init__.py` + 414 modules) + 16 data files under `brainstem/so101/`
  (13 STL meshes, two MuJoCo XML files, the licence). Per package: cortex 94, reasoning 92 + `reasoning/ng_snn` 9 +
  `reasoning/arc_agi` 6, substrate 87, hippocampus 26, plasticity 22, neuromodulation 13, brainstem 12, basal_ganglia
  10, cerebellum 10, thalamus 8, training 6, midbrain 4, hypothalamus 3, amygdala 2, lakehouse 2, medulla 2, top level
  2, claustrum, epithalamus, insula and pons 1 each, olfactory 0.
- The registry, parsed by AST from `src/snn_data_lake/__init__.py`: `REGIONS` 286 modules under 18 keys, `TORCH_TIER`
  7, `TRAIN_TIER` 4, so 297 governed; 117 ungoverned (reasoning 107, training 6, lakehouse 2, top level 2). No registry
  entry lacks its file, no two of the 414 modules share a basename, and one basename is also a package name: `claustrum`.
- Section 2.12's table is the draft's, re-verified row by row: 414 rows, each module exactly once (0 missing, 0 extra, 0
  duplicated), 0 tier mismatches against the registry, 253 moves and 161 stays. Every move stays one package deep, keeps
  its basename and keeps its governance status. Destinations: faculties 92, sensorimotor 31, growth 26, gates 23,
  instruments 23, infra 14, sleep 11, readout 9, engine 7, reservoir 7, substrate 6 in, plasticity 4 in. Stays:
  plasticity 14, substrate 25, the five envelope files and the 117 ungoverned modules. 75 rows carry a judgment flag, 69
  of them moves.
- The moves empty 15 package directories: the 14 anatomy packages and the already empty `olfactory`. **All 14 anatomy
  packages empty only at M11**, because each holds at least one faculty. All 23 subpackage `__init__.py` files are
  docstring-only, so nothing is re-exported through them.
- **An emptied directory is not empty on disk** (the design read, F4, at `f416296`): 21 package directories hold an
  ignored `__pycache__/` (`.gitignore:2`), so removing a package's tracked files leaves a directory behind.
  `claustrum/` holds `__pycache__/`, `claustrum.py` and `__init__.py`; the already empty `olfactory/` holds
  `__init__.py` and `__pycache__/`.

### 1.3 The live column (unchanged)

The table's `live` column, re-derived with the dormancy gate's own walk (`dormancy_gate._graph()`, closed from
`_LIVE_ROOTS` for L and from `_ENTRYPOINTS` for E): L 253, E 22, `-` 139, and 0 mismatches against the draft's column
at both heads. `python -m snn_data_lake.substrate.dormancy_gate` prints brain modules 297, reachable 238, exempt 59,
dormant 0, no stale exemption and 7 open (non-permanent) dispositions. Two governed modules are reached only through
`_external_roots()` (the root scripts and `integrations/`): `substrate/learn_daemon.py` and `substrate/selfcheck_cache.py`.

### 1.4 The path-keyed registries (unchanged except where marked)

| registry | at `68db184` | re-pointed at |
|---|---|---|
| `dormancy_gate.EXEMPT` (basename keys) | 59 rows (52 permanent, 7 wire); 0 name a module absent from `_CANON` | none (the basename invariant) |
| `dormancy_gate._LIVE_ROOTS` | 4: `cortex.brain`, `cortex.live_brain`, `brainstem.autonomous_loop` (engine); `thalamus.reentrant_workspace` (faculties) | M10, M11 |
| `dormancy_gate._ENTRYPOINTS` | 8: `live_arm`, `closed_loop`, `web_ingest` (M3); `demo`, `cockpit`, `nengo_bridge`, `region_wiring` (M4); `training.cli` stays | M3, M4 |
| `method_dormancy_gate.EXEMPT_METHOD` | 13 keys, 8 path-keyed, 7 re-keyed (M6 1, M7 1, M11 5); `plasticity.espp_rule._SpyESPP.train_step` stays | M6, M7, M11 |
| `import_manifest.GOVERNED` | 16 names; lacks `insula` and `pons` (and `olfactory`, which holds no module); **also carries `lakehouse`, which `REGIONS` does not govern**; a listed name with no directory is skipped without a word (`import_manifest.py:173`; the design read, F15) | each stage that creates or deletes a package (2.5 item 6); the scan delta declared at M11b |
| `replication_guard.CANON` | 3, all to `engine` | M10 |
| `manifest_gate.REGISTRANTS` | 22 dotted: M3 2 (the afferent and efferent boundaries), M5 2, M7 1 (`r_stdp`), M10 3, M11 14 | per stage |
| `run_selfchecks.py` quoted `<pkg>.<mod>` names | 366 (364 distinct; 3 are comment mentions); 254 name moved modules (M1 23, M2 23, M3 31, M4 14, M5 26, M6 7, M7 13, M8 11, M9 6, M10 7, M11 93), one of them a comment, so **253 runner strings** | per stage |
| `brian2_backend` `MIGRATED_CALL_SITES` + `CUSTOM_KERNELS` | 32 entries, **20** name moved modules (2 + 18); `MODULE_NAME` = `snn_data_lake.substrate.brian2_backend` is the `-m` target of its isolated-simulator subprocesses (the free-threading boundary) | M2, M5, M6, M7, M11 |
| `selfcheck_cache` self-check subjects | `"substrate.signal_ring"`, `"substrate.dormancy_gate"`; a stale subject silently falls back to another | M10, M1 |
| `growth_coverage_gate._PLACEMENT_MODULES` | `snn_data_lake.cortex.placement`, `cortex.placement`, `placement` (the bare name masks a stale pair) | M5 |
| `tests/pin/capture.py` | keys the ignored local pins by the string `snn_data_lake.cortex.eprop_shd` (it also imports `eprop_shd` and `eprop_nmnist`; its docstring names `cortex.eprop_kernel`) | M3 (the docstring M7) |
| `pyproject.toml [project.scripts]` | 6; `snn-arm`, `snn-loop`, `snn-fetch-corpus` (M3), `snn-demo`, `snn-nengo-launch` (M4); `snn-train` stays | M3, M4 |
| `.github/workflows/ci.yml` | 16 `python -m snn_data_lake.substrate.*` calls, all moved: 15 gates (M1), `brian2_backend` (M5); the `brainstem/so101/scene.xml` path stays | M1, M5 |
| `integrations/packaging` | `snn-ui.spec` excludes `snn_data_lake.hippocampus.ng_snn` (M5); `snn_ui_main.py` imports `substrate.cockpit` (M4) | M4, M5 |
| root `learn_daemon.py` shim | imports `substrate.learn_daemon` | M10 |
| POLICIES.md | 57 citations of moved modules by a path-or-dotted regex (51 path form): M1 24, M2 1, M3 2, M4 7, M5 7, M6 1, M7 3, M10 7, M11 5; the e-prop register's 5 sites (4 re-pointed, `reasoning/gf.py::MLP.train` stays) | per stage, with `policy_manifest.json` regenerated |
| documented `python -m` commands (doc_command_guard SCOPE) | 66 commands in the 62 scoped documents; 35 name moved modules, in 12 documents: M1 7, M2 1, M3 17, M4 2, M5 1, M9 2, M10 1, M11 4 | per stage |
| `tools/p4c_ungoverned_baseline.json` | total 202; the `tools` zone 30 | none (governed moves only) |
| `tools/envelope_hashes.json`, `safety_envelope.ENVELOPE`, `self_mod_harness._PROTECTED`, `godel_machine._SAFETY_KERNEL` | unchanged; the harness's and the proposer's own entries are derived from `__file__` | none |
| string constants naming a moved module, live `.py` | 343 in 22 files | per stage (C2(d)) |
| `tools/rule_census.json` + `rule_composer._CENSUS_ADJUDICATED` (the design read, F6, at `f416296`) | 104 rows under `rules`, keyed `<pkg>/<mod>.py::<qualname>`; 18 adjudicated entries, keyed by path; the regen refuses an adjudicated entry that matches no live row; B1 classes the JSON as frozen text | each stage that moves a rule-bearing module (2.5 item 6) |

The fold splits S1 into S1a-S1c and M11 into M11a-M11e (section 2.4 distributes this table's S1 and M11 items). The
further path-keyed code sites the design read found (F1, F5, F7, F16, F17) are ledgered with their reproductions in
section 2.13.

No new package name matches a `.gitignore` rule (`git check-ignore`; positive control: `__pycache__/` is reported
ignored). The registry's direct readers are `brain_atlas`, `dormancy_gate`, `region_charge_gate`, `atlas_node_gate`,
`brain_geometry`, `region_wiring` and `run_selfchecks.py`; `tests/test_arm_vocabulary.py` reads it through
`brain_atlas.REGIONS`, `tools/p4c_ratchet.py` and `tools/promotion_sweep.py` parse it by AST, and `selfcheck_cache`
anchors on its `REGIONS = {` text. `live_brain.spike_snapshot()` keys a hard-coded tuple of 17 anatomical names, not the
registry, so G8 does not touch it.

### 1.5 `__file__` depth

- 154 lines in 87 files use `__file__`; 59 of them count directory levels on the same line (34 `parents[3]`, 2
  `parents[2]`, one four-`.parent` chain, 12 triple-`dirname`, 10 double-`dirname`), unchanged.
- **59 is a same-line lower bound.** At least three more lines count levels from a name derived from `__file__`
  earlier: `doc_budget_gate._ROOT` (four `dirname` of `_HERE`), the repo in `dormancy_gate._external_roots()` and
  `import_manifest._ROOT_DIR` (two `dirname` of `_PKG_DIR`); `self_mod_harness._repo_root()` falls back to
  `join(here, "..", "..", "..")`. The depth invariant (section 2.3) covers them because it is asserted per module, not
  per line.
- In the 92 faculty-bound modules: 14 `__file__` lines in 9 modules (the draft's figure), of which 4 lines in 4 modules
  count levels (`percept_surprise`, `recursive_tom`, `time_cells`, `wm_capacity`). `rose_language._region()` prints its
  own directory's name as its "region" and will print `faculties` (a printed label only).

### 1.6 The back-compat alias layer's limits (unchanged; measured at run time)

- `_MOVED` holds 285 flat names derived from `REGIONS` (`claustrum` excluded by the `m != r` guard; no `TORCH_TIER` or
  `TRAIN_TIER` name). The `_AliasFinder` is the last entry of `sys.meta_path`.
- `find_spec` returns a spec for a flat name and `None` for a two-level path (`snn_data_lake.cortex.lif_neuron`), a
  package name and a torch-tier name. `_AliasLoader` defines `create_module` and `exec_module` only; nothing synthesizes a
  package. Identity holds: the flat alias is the regional module object.
- `python -m snn_data_lake.lif_neuron` in a fresh process exits 1 with `AttributeError: '_AliasLoader' object has no
  attribute 'get_code'`. In a process that has already imported the alias, `runpy` fails differently: `ImportError:
  loader for snn_data_lake.cortex.lif_neuron cannot handle snn_data_lake.lif_neuron`. The real `SourceFileLoader` checks
  the name it is asked for, so a `get_code` delegation must ask for the target's own name.
- In-tree flat-alias imports: 14 statements in 13 files by AST (for example `interpreter.selfcheck`'s
  `from snn_data_lake import signal_loop`).

### 1.7 The G0(c) checkpoint, measured

The facts in the code: `interaction_learning._encode_state` writes `"__cls__": "<module>.<qualname>"` for every package
object in the chat learner's tree; `_merge_state` refuses a stored name that differs from the fresh instance's class
("checkpoint class mismatch") and any lazy class outside `_G0C_LAZY_ATTRS` (which names `plasticity.bcm_plasticity.
BCMNeuron` and `plasticity.benna_fusi.BennaFusiSynapse`, both staying). `Brain.__init__` (the `persist` branch) wraps the
restore in `except Exception` and reports "restore refused; learner stays fresh" to stderr; `Brain.
_flush_interaction_learner` then saves the fresh learner at close or at the end of a night. The measurements (B2, a
fresh process, 16 s wall):

- **The default learner's checkpoint** (the configuration `make_brain` builds; here `make_brain(seed=100,
  persist=False)` taught with the frozen `_g0c_schedule`) holds 17 classes and 472 objects. **Three of the 17 move:** `SpikingInteractionLearner` (row 41, readout, M7),
  `RSTDPNeuron` (row 50, plasticity, M7) and **`LIFNeuron` (row 21, growth, M5; 9 objects, one inside each
  `RSTDPNeuron`)**. The other 14 live in `plasticity/` and stay, among them six private stack classes
  (`plasticity_rule_stack._BCMMetaGate`, `_BennaFusiConsolidation`, `_EIBalance`; `homeostatic_plasticity_stack.
  _HomeostaticScaling`, `_IntrinsicPlasticity`, `_ShortTermGain`).
- **The M5 red, constructed without any move:** the checkpoint restores unmodified into a fresh learner; with only the
  `LIFNeuron` class string rewritten to its M5 path it is refused, "checkpoint class mismatch". The default learner
  therefore breaks first at M5, not M7.
- **The opt-in `readout_zoo` learner** adds 10 classes, all neuron models that move at M5 (rows 9, 10, 12, 14, 23, 25,
  26: `AdEx`, `ALIFNeuron`, `AutapseNeuron`, `DendriticNeuron`, four `neuron_zoo` cells, `ResonateFire`, `SRMNeuron`).
  Its checkpoint restores only into a zoo learner whose pools already exist (after one `respond()`); a fresh zoo learner
  and the default learner refuse it ("checkpoint attribute mismatch" on `RSTDPNeuron`, because `_zoo_pool` is created
  lazily). `make_brain` never builds a zoo learner, so this arm is learner-level, not a brain construction.
- **With a `delta_sink` attached, `state_dict()` refuses** ("unsupported checkpoint value: builtins.function"), as the
  frozen G0(c) protocol pins (`tests/test_g0c_learner_persists.py::test_frozen_protocol_does_not_silently_exclude_other_state`).
  No checkpoint carrying a `SynapseDelta` can exist today.
- The probe process left a 15 KB `brain_lake.bin` in its temporary home although no arm persisted, so the per-arm home
  isolation is load-bearing even for the non-persisting arms.
- The daemon: no answer on `127.0.0.1:8137` at this writing (`curl -m 5` exit 7); the owner stopped it on 2026-09-24.

### 1.8 The envelope pins (unchanged)

The five envelope files import outside `brainstem/` in four places, each at use time: `so101_arm._latch_path` (the
durable E-STOP latch) imports `substrate.home.home`; `So101Arm.view` imports `substrate.arm_view.ArmView`;
`so101_arm.main` and `serial_arm.main` import `substrate.home.throwaway_home`. `interpreter.selfcheck` reaches
`signal_loop` only through the flat alias. Their other imports are `snn_data_lake.brainstem.*`, which stays.
`python -m snn_data_lake.substrate.envelope_hash_gate --assert` prints "OK: all 5 envelope files match their approved
hashes"; `git ls-files --eol` shows `serial_arm.py` and `so101_arm.py` LF in the working tree, `reflex.py` CRLF.
The envelope's comments cite moving modules on 8 lines of `so101_arm.py` (241, 243, 602-607; the design read, F11);
they stay byte-identical, and section 2.6 names them.

### 1.9 G1(b)'s R-b0 after the losing ports' deletion

`recurrent_pool.check_r_b0()` now runs one check, `check_r_b0_lsm`, which lazily imports one source,
`snn_data_lake.reasoning.spiking_ssm`, and carries its own red (`red_leak`). The frame tracing by code object that the
draft cites (`sys.settrace`, `frame.f_code is code`) belonged to the deleted ports and is gone. `reasoning/` does not
move, so the check is expected identical before and after every stage (C4).

### 1.10 What a pair-keyed rewrite strands silently (found by this re-verification)

The draft's catch-all (C2(d)) counts dotted string constants in live code and names none of these; the template, the
path-form literal and the frozen runnable citations sit outside its rewrite rule (an exact `(package, module)` map) or
its scope:
- **Neuter keys and lazy-contract asserts** on `snn_data_lake.cortex.kb_ranking` (row 230, M11): `reasoning/
  schema_induction.py` constructs its confidence-gate red by setting `sys.modules[_KB] = None`, and `reasoning/
  reasoning_cortex.py` asserts three times that the key is not in `sys.modules`. With a stale key the red stops firing
  and the asserts pass vacuously; `schema_induction`'s own comment records the same defect at the 2026-08-11 promotion.
  Both files are ungoverned, so C2(d) must scan the ungoverned tiers.
- **Subprocess self-launch strings**: `interaction_learning` (3), `memory_diagnostic`, `neurogenesis_engine`,
  `pool_selection`, `everyday_battery` and `brian2_backend.MODULE_NAME` launch `-m snn_data_lake.<old path>`; after
  their stage they would run only through the alias's `get_code`.
- **A templated battery command**: PLAN A.4's battery line runs `-m snn_data_lake.substrate.$m` over the 16 gate names
  (plus two literal `policy_manifest` calls). The pair map cannot see the template, and `specs/` is outside
  doc_command_guard's SCOPE, so nothing asserts it. The same line lives in two copies outside the repo (the design
  read, F12): the memory store's `archive/t062-t063-2026-09-09/t066_battery_v2.sh` and
  `D:\AI-project\omnigent-sndl\SNDL-PROTOCOL.md` section 5.
- **Path-form literals**: `brain_native_selfbuild`'s firewall self-check lists `src/snn_data_lake/substrate/
  self_mod_harness.py` as "the harness at its REAL path"; after M4 it is silently a not-at-that-path entry (the check
  stays green through basename widening and stops exercising the real path).
- **Runnable citations inside frozen documents** (citation_guard's asserted STRICT scope: a shell fence or a
  Reproduce/Usage heading): `specs/005-language-organ/quickstart.md` lines 75-76 (`cortex/language_acquisition.py`,
  `cortex/spiking_lm_serve.py`, `cortex/brain.py`) and `specs/022-unified-brain-arc/evidence/t081-report-order.md` line
  272 (`cortex/report_order_probe.py`, `thalamus/ignition_instrument.py`). Left as frozen history, these 5 citations
  stop resolving at M2, M3, M10 and M11, and `tools/citation_guard.py` goes red (lint CI and the battery). The design
  read (F10) told them apart: t081's line 272 is prose that the `reproduc` heading rule catches (a false positive, not
  a command); the quickstart's lines 75-77 are one `git diff --quiet HEAD -- <paths>` command, which a stale path
  passes vacuously. Beside them, `docs/POLICIES_EVIDENCE.md` lines 410 and 476 are `python -m` commands naming
  `substrate.import_manifest` and `substrate.method_dormancy_gate`, in doc_command_guard's SCOPE, which resolves `-m`
  by file and rejects a flat alias.

### 1.11 What moved since `4b237c6`, and what the draft got wrong

Counts that moved (the instrument reproduced the draft's figure at `4b237c6` unless the middle column says otherwise):

| count | draft at `4b237c6` | re-derived at `4b237c6` | at `68db184` |
|---|---|---|---|
| import statements naming moved modules (stage-attributed) | 1,124 in 370 files | 1,124 in 370 | **1,123 in 369** (M7 60 -> 59; 47 -> 46 files) |
| live-document citations of moved modules | 1,265 (1,180 path + 85 dotted; 60 / 32 files) | 1,264 (1,180 + 84; 60 / 31) | **1,278** (1,193 + 85; 60 / 31) |
| frozen-document citations of moved modules | 1,426 path + 131 dotted (133 / 46 files) | same | **1,442** + 131 (135 / 46) |
| `.py` path citations of moved modules | 1,460 | 1,460 | **1,468** |
| path citations of any module, non-`.py` text | 3,890 in 222 files (another extension set) | 3,846 in 222 | **3,881 in 224** |
| POLICIES.md citations of moved modules | 57 (M1 26) | 56 (M1 24) | **57** (M5 6 -> 7) |
| R-b0 checks / sources | four / four | -- | **one / one** (1.9) |
| tracked files | -- | 1,246 | 1,248 |

Everything in sections 1.2-1.6 and 1.8 is unchanged. What the draft got wrong, most consequential first:

1. **The checkpoint census.** Its class table omits `LIFNeuron`, which the production default checkpoint holds and which
   moves at M5 (1.7); its R-L1 red ("refuses at M5 (zoo arm) and at M7 (all arms)") is wrong for the default arm, which
   refuses at M5. The plan's G8 row and T116 name the same incomplete list ("`interaction_learning`, `r_stdp`, the
   plasticity stack or `rule_composer`"). S0 lands before every move, so the design holds; the red and the hazard
   ranking change.
2. **R-L1's arms cannot all be built as written.** The `delta_sink` arm cannot write a checkpoint (its `state_dict()`
   refuses), and the zoo arm cannot be restored by constructing the brain (1.7). Section 2.7 re-designs both.
3. **S1's positive null cannot pass as written.** "The S0 -> S1 census diff must be EMPTY" contradicts S1 adding
   `move_gate`: a governed module, a new `REGIONS` key, a dormancy EXEMPT row, a runner name and an `import_manifest`
   entry. Sections 2.4 and 2.6 declare that delta instead.
4. **Five runnable citations in frozen documents** turn citation_guard's STRICT scope red at M2, M3, M10 and M11
   (1.10). The draft keeps frozen documents untouched and also asserts STRICT green (C7); both cannot hold.
5. **The A.4 battery template** is invisible to the draft's pair-keyed rewrite (1.10), though the draft has M1 rewrite
   A.4's block.
6. **C2(d)'s scope** must include the ungoverned tiers, tests, path-form strings and `sys.modules` keys (1.10).
7. **C4 is stale:** one R-b0 check with one source and no frame tracing (1.9); the draft anticipated the deletion.
8. **The `get_code` delegation** must pass the target's name (1.6).
9. `import_manifest.GOVERNED` also carries `lakehouse`; the "new package list" must keep scanning it.
10. Minor figures: `brian2_backend`'s two tuples hold 20 moved entries, not 21 (the 21st moved-module string is
    `MODULE_NAME`); the "54 in-tree flat-alias imports" did not reproduce (14 by AST; no definition was recorded); "14
    `__file__` root lines in 9 faculty modules" are 14 `__file__` lines, 4 of them root-counting (1.5); the 59
    level-counting lines are a same-line lower bound (1.5); the 366 runner names include 3 comment mentions; the frozen
    `docs/coverage/artifacts/` hold 16 `.py` files, 15 of which make 32 package imports (the draft: 15 files, 31).

### 1.12 The commands that re-derive section 1

From the repo root, with the repo's venv. The `git` facts: `git rev-parse HEAD origin/main`, `git status --short`,
`git log --oneline 4b237c6..68db184`, `git diff --stat 4b237c6 68db184 -- src`, `git ls-files src/ | wc -l`,
`git ls-files 'src/*.py' | wc -l`, `git ls-files 'src/*__init__.py' | wc -l`,
`git ls-files src/snn_data_lake/brainstem/so101 | wc -l`, `git ls-files --eol src/snn_data_lake/brainstem`,
`git check-ignore -v src/snn_data_lake/<new package>/x.py` for each new package name. The gates:
`python -m snn_data_lake.substrate.dormancy_gate`, `python -m snn_data_lake.substrate.envelope_hash_gate --assert`. The
daemon: `curl -s -m 5 http://127.0.0.1:8137/api/state; echo $?` (7 = no answer). The searches behind sections 1.5 and
1.10: `git grep -n -E "dirname\(os.path.dirname\(_(HERE|PKG_DIR)|join\(here, .\.\..," -- src/snn_data_lake/substrate`;
`git grep -n "snn_data_lake.cortex.kb_ranking" -- src/snn_data_lake/reasoning`;
`git grep -n -E "MODULE_NAME = |\"snn_data_lake\.cortex\.(interaction_learning|memory_diagnostic|neurogenesis_engine|pool_selection|everyday_battery)\"" -- src`;
`git grep -n -F 'snn_data_lake.substrate.$m'`; `git grep -n "the harness at its REAL path"`. The draft's figures at
`4b237c6` were reproduced by the same instruments run over a scratch export of `git archive 4b237c6` (a directory walk
in place of `git ls-files`, this file's table copied in, and `PYTHONPATH` pointed at the export's `src` for the live
column).

B1, static (no package import; section 1.2, 1.4, 1.5 and the moved counts in 1.11; about 4 s):

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import ast, os, re, subprocess
from collections import Counter
DOC = "specs/022-unified-brain-arc/evidence/g8-reorganization.md"
ls = subprocess.run(["git", "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
P = "src/snn_data_lake/"
mods = [f[len(P):] for f in ls if f.startswith(P) and f.endswith(".py") and not f.endswith("__init__.py")]
def lit(path, names):  # module-level literal tables, parsed, never imported
    out = {}
    for n in ast.parse(open(path, encoding="utf-8").read()).body:
        t = n.targets[0] if isinstance(n, ast.Assign) else getattr(n, "target", None)
        if isinstance(t, ast.Name) and t.id in names:
            out[t.id] = n.value
    return out
tabs = {k: ast.literal_eval(v) for k, v in lit(P + "__init__.py", {"REGIONS", "TORCH_TIER", "TRAIN_TIER"}).items()}
tier = {f"{r}/{m}.py": {"REGIONS": "C", "TORCH_TIER": "T", "TRAIN_TIER": "TR"}[k] for k, t in tabs.items() for r, ms in t.items() for m in ms}
print("modules", len(mods), "| governed", len(tier), {k: sum(map(len, t.values())) for k, t in tabs.items()}, "| REGIONS keys", len(tabs["REGIONS"]), "| ungoverned", len(mods) - len(tier))
print("basename collisions", [b for b, c in Counter(m.rsplit("/", 1)[-1] for m in mods).items() if c > 1],
      "| basenames equal to a package", sorted({m.rsplit("/", 1)[-1][:-3] for m in mods} & {m.split("/")[0] for m in mods if "/" in m}))
row = re.compile(r"^\| (\d+) \| `([^`]+)` \| (= \(stays\)|`[^`]+`) \| ([^|]*) \| (\w+) \| ([LE-]) \| ([^|]*) \|$")
rows = [m.groups() for m in map(row.match, open(DOC, encoding="utf-8").read().splitlines()) if m]
cur = [r[1] for r in rows]
new = {r[1]: r[2].strip("`") for r in rows if r[2] != "= (stays)"}
print("table rows", len(rows), "| missing", sorted(set(mods) - set(cur)), "| extra", sorted(set(cur) - set(mods)), "| duplicated", len(cur) - len(set(cur)),
      "| tier mismatches", [r[1] for r in rows if r[4] != tier.get(r[1], "U")])
print("moves", len(new), "| stays", len(rows) - len(new), "| not one level deep or renamed",
      [k for k, v in new.items() if k.count("/") != 1 or v.count("/") != 1 or v.split("/")[1] != k.split("/")[1][:-3]],
      "| governance changed", [k for k, v in new.items() if (k in tier) == (v.split("/")[0] in ("reasoning", "training", "lakehouse"))])
print("destinations", dict(sorted(Counter(v.split("/")[0] for v in new.values()).items())), "| judgment flags", sum(r[6].startswith("J") for r in rows),
      "(moves", sum(r[6].startswith("J") for r in rows if r[1] in new), ") | live column", dict(Counter(r[5] for r in rows)))
left = Counter(k.split("/")[0] for k in mods if "/" in k and k not in new)
print("package dirs emptied", sorted(d for d in os.listdir(P) if os.path.isdir(P + d) and d != "__pycache__" and not left[d]))
STAGE = dict(gates="M1", instruments="M2", sensorimotor="M3", infra="M4", growth="M5", reservoir="M6", plasticity="M7", readout="M7", sleep="M8", substrate="M9", engine="M10", faculties="M11")
stage = {k: STAGE[v.split("/")[0]] for k, v in new.items()}
order = lambda c: dict(sorted(c.items(), key=lambda kv: int(kv[0][1:])))
ff, lv = [], []
for m in mods:
    for i, line in enumerate(open(P + m, encoding="utf-8").read().splitlines(), 1):
        if "__file__" in line:
            ff.append(m)
            k = [f"parents[{x}]" for x in re.findall(r"\.parents\[([2-9])\]", line)] or (["4x.parent"] if ".parent.parent.parent.parent" in line else []) or ([f"{line.count('dirname(')}x dirname"] if line.count("dirname(") >= 2 else [])
            if k:
                lv.append((m, k[0]))
fac = [m for m in ff if new.get(m, "").startswith("faculties/")]
print("__file__ lines", len(ff), "in", len(set(ff)), "files | level-counting", len(lv), dict(Counter(k for _, k in lv)),
      "| in faculty-bound modules:", len(fac), "lines in", len(set(fac)), "modules,", sum(new.get(m, "").startswith("faculties/") for m, _ in lv), "level-counting")
D = P + "substrate/"
ex = [ast.literal_eval(k) for k in lit(D + "dormancy_gate.py", {"EXEMPT"})["EXEMPT"].keys]
canon = {m for t in tabs.values() for ms in t.values() for m in ms}
em = [ast.literal_eval(k) for k in lit(D + "method_dormancy_gate.py", {"EXEMPT_METHOD"})["EXEMPT_METHOD"].keys]
emp = [k for k in em if f"{k.split('.')[0]}/" in {m.split("/")[0] + "/" for m in mods}]
print("dormancy EXEMPT rows", len(ex), "| phantom rows", [k for k in ex if k not in canon], "| EXEMPT_METHOD keys", len(em), "path-keyed", len(emp),
      "moved", sum(f"{k.split('.')[0]}/{k.split('.')[1]}.py" in new for k in emp))
gov = ast.literal_eval(lit(D + "import_manifest.py", {"GOVERNED"})["GOVERNED"])
print("import_manifest.GOVERNED", len(gov), "| packages on disk it lacks", sorted({m.split("/")[0] for m in mods if "/" in m} - set(gov) - {"reasoning", "training"}),
      "| names it carries that REGIONS does not", sorted(set(gov) - set(tabs["REGIONS"])))
regs = [ast.literal_eval(e)[0] for e in lit(D + "manifest_gate.py", {"REGISTRANTS"})["REGISTRANTS"].elts]
dk = lambda d: "/".join(d.split(".")[1:3]) + ".py"
print("REGISTRANTS", len(regs), order(Counter(stage[dk(d)] for d in regs if dk(d) in stage)))
bb = lit(D + "brian2_backend.py", {"MIGRATED_CALL_SITES", "CUSTOM_KERNELS", "MODULE_NAME"})
bbl = [ast.literal_eval(e).split("::")[0] for k in ("MIGRATED_CALL_SITES", "CUSTOM_KERNELS") for e in bb[k].elts]
print("brian2_backend tuples: moved entries", sum(dk(d) in new for d in bbl), "of", len(bbl), "| MODULE_NAME", ast.literal_eval(bb["MODULE_NAME"]))
rs = open("run_selfchecks.py", encoding="utf-8").read()
q = [f"{a}/{b}.py" for a, b in re.findall(r"[\"']([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)[\"']", rs) if os.path.isdir(P + a)]
print("run_selfchecks quoted names", len(q), "| naming moved modules", sum(k in new for k in q), order(Counter(stage[k] for k in q if k in new)))
live_py = [f for f in ls if f.endswith(".py") and not f.startswith("docs/coverage/artifacts/") and (f.startswith(("src/", "tests/", "tools/", "integrations/", "research/", "training/")) or "/" not in f)]
dot = {"snn_data_lake." + k[:-3].replace("/", "."): s for k, s in stage.items()}
pk = {}
for k in stage:
    pk.setdefault(k.split("/")[0], set()).add(k.split("/")[1][:-3])
imp, files, strs, sfiles = Counter(), {}, 0, set()
pat = re.compile(r"^(?:snn_data_lake\.)?([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)(?:[.:][\w.:]*)?$")
for f in live_py:
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        continue
    for n in ast.walk(tree):
        s = set()
        if isinstance(n, ast.Import):
            s = {dot[a.name] for a in n.names if a.name in dot}
        elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
            s = {dot[n.module]} if n.module in dot else set()
            p = n.module.split(".")
            if len(p) == 2 and p[0] == "snn_data_lake":
                s |= {stage[f"{p[1]}/{a.name}.py"] for a in n.names if a.name in pk.get(p[1], ())}
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            m = pat.match(n.value)
            if m and f"{m.group(1)}/{m.group(2)}.py" in new:
                strs += 1
                sfiles.add(f)
        for x in s:
            imp[x] += 1
            files.setdefault(x, set()).add(f)
print("import statements naming moved modules", sum(imp.values()), "in", len(set().union(*files.values())), "files |", order(imp), "| files", order({k: len(v) for k, v in files.items()}))
print("string constants naming moved modules", strs, "in", len(sfiles), "files")
frozen = lambda f: ("/evidence/" in f and f.startswith("specs/")) or re.match(r"specs/0(0\d|1\d|2[01])-", f) or f in ("docs/POLICIES_EVIDENCE.md", "CHANGELOG.md") or f.startswith("docs/coverage/artifacts/") or (f.startswith("tools/") and f.endswith(".json") and f != "tools/policy_manifest.json")
text = [f for f in ls if not f.endswith(".py") and f.lower().endswith((".md", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".json", ".bat", ".sh", ".ps1", ".nir", ".spec", ".vbs", ".rst", ".html", ".csv", ".xml", ".in"))]
PR = re.compile(r"(?<![\w])([a-z_][a-z0-9_]*)/([A-Za-z_][A-Za-z0-9_]*)\.py\b")
DR = re.compile(r"snn_data_lake\.([a-z_][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
def cites(fs, rx):
    c, hit = Counter(), set()
    for f in fs:
        for m in rx.finditer(open(f, encoding="utf-8", errors="replace").read()):
            k = f"{m.group(1)}/{m.group(2)}.py"
            if k in stage:
                c[stage[k]] += 1
                hit.add(f)
    return c, len(hit)
for label, fs in (("LIVE docs", [f for f in text if not frozen(f) and f != DOC]), ("FROZEN docs", [f for f in text if frozen(f) and f != DOC])):
    (cp, fp), (cd, fd) = cites(fs, PR), cites(fs, DR)
    print(label, "citing moved modules: path", sum(cp.values()), "in", fp, "files, dotted", sum(cd.values()), "in", fd, "files | per stage", order(cp + cd))
cp, fp = cites(live_py, PR)
print(".py path citations of moved modules", sum(cp.values()), "in", fp, "files |", order(cp))
import json, sys, tomllib
sys.path.insert(0, "tools")
import citation_guard as cg, doc_command_guard as dcg
J = lambda e: "".join(x.value for x in e.values if isinstance(x, ast.Constant)).lstrip(".") if isinstance(e, ast.JoinedStr) else ast.literal_eval(e)
where = lambda d: stage.get("/".join(d.replace("snn_data_lake.", "").split(".")[:2]) + ".py", "stays")
roots = lit(D + "dormancy_gate.py", {"_LIVE_ROOTS", "_ENTRYPOINTS"})
print("_LIVE_ROOTS", {J(e): where(J(e)) for e in roots["_LIVE_ROOTS"].elts}, "| _ENTRYPOINTS", {J(e): where(J(e)) for e in roots["_ENTRYPOINTS"].elts})
canon = lit(D + "replication_guard.py", {"CANON"})["CANON"]
print("CANON", {ast.literal_eval(k): where(J(v)) for k, v in zip(canon.keys, canon.values)})
print("pyproject scripts", {k: where(v.split(":")[0]) for k, v in tomllib.load(open("pyproject.toml", "rb"))["project"]["scripts"].items()})
print("ci.yml snn_data_lake.substrate calls", len(re.findall(r"snn_data_lake\.substrate\.\w+", open(".github/workflows/ci.yml", encoding="utf-8").read())))
scope = list(dcg._iter_docs())
cmds = [(str(doc), m.group("mod")) for doc in scope for line in doc.read_text(encoding="utf-8", errors="replace").splitlines() for m in dcg.CMD.finditer(line)]
mv = [c for c in cmds if where(c[1]) != "stays"]
print("SCOPE documents", len(scope), "| commands", len(cmds), "| naming moved modules", len(mv), "in", len({c[0] for c in mv}), "documents", order(Counter(where(c[1]) for c in mv)))
base = json.load(open("tools/p4c_ungoverned_baseline.json", encoding="utf-8"))
print("p4c baseline total", base["total"], "| tools zone", base["zones"]["tools"])
pol = open("docs/POLICIES.md", encoding="utf-8").read()
pc = Counter(stage[f"{a}/{c}.py"] for a, c in re.findall(r"(?<![\w/.])([a-z_][a-z0-9_]*)[./]([a-z_][a-z0-9_]*)(?:\.py)?\b", pol) if f"{a}/{c}.py" in stage)
print("POLICIES.md citations of moved modules", sum(pc.values()), order(pc), "| path form", sum(1 for a, c in PR.findall(pol) if f"{a}/{c}.py" in stage))
art, ai = [f for f in ls if f.startswith("docs/coverage/artifacts/") and f.endswith(".py")], []
for f in art:
    for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
        if any(x.startswith("snn_data_lake") for x in names):
            ai.append(f)
print("frozen artifacts", len(art), ".py | importing the package", len(set(ai)), "files,", len(ai), "statements")
run = []
for f in ls:
    if f.endswith(".md") and f != DOC:
        lines = open(f, encoding="utf-8", errors="replace").readlines()
        run += [(f, i, m.group(1)) for i in sorted(cg._runnable_lines(lines)) for m in cg._CITE.finditer(lines[i - 1]) if "/".join(m.group(1).split("/")[-2:]) in stage]
print("citation_guard runnable-line citations of moved modules", len(run), run)
flat, fa, readers = {m for r, ms in tabs["REGIONS"].items() for m in ms if m != r}, [], set()
for f in [x for x in ls if x.endswith(".py")]:
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "snn_data_lake" and not n.level:
            fa += [f] if any(a.name in flat for a in n.names) else []
            if {a.name for a in n.names} & {"REGIONS", "TORCH_TIER", "TRAIN_TIER"}:
                readers.add(f)
        elif isinstance(n, ast.Import):
            fa += [f for a in n.names if a.name.count(".") == 1 and a.name.startswith("snn_data_lake.") and a.name.split(".")[1] in flat]
print("flat-alias import statements", len(fa), "in", len(set(fa)), "files | direct registry readers", sorted(readers))
rp = open(P + next(m for m in mods if m.endswith("/recurrent_pool.py")), encoding="utf-8").read()  # wherever it lives
print("R-b0:", re.findall(r"def (check_r_b0\w*)", rp), "| reasoning imports", re.findall(r"from snn_data_lake\.reasoning import (\w+)", rp), "| settrace in the module", "settrace" in rp)
sc = open(D + "selfcheck_cache.py", encoding="utf-8").read()
print("selfcheck_cache subjects", [s for s in ('"substrate.signal_ring"', '"substrate.dormancy_gate"') if s in sc],
      "| _PLACEMENT_MODULES", sorted(ast.literal_eval(lit(D + "growth_coverage_gate.py", {"_PLACEMENT_MODULES"})["_PLACEMENT_MODULES"])))
outside = {f: sorted(set(re.findall(r"snn_data_lake\.[a-z_]+\.[a-z_0-9]+", open(f, encoding="utf-8").read()))) for f in ("tests/pin/capture.py", "integrations/packaging/snn-ui.spec", "learn_daemon.py")}
print("outside src, moved names:", {f: {d: where(d) for d in v if where(d) != "stays"} for f, v in outside.items()},
      "| snn_ui_main imports", re.findall(r"from snn_data_lake\.(\w+) import (\w+)", open("integrations/packaging/snn_ui_main.py", encoding="utf-8").read()))
EOF
```

B2, the checkpoint (section 1.7; builds one scratch brain in a temporary home, never the live lake; about 16 s):

```bash
PYTHONUTF8=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import json, os, tempfile
os.environ["NORD_HOME"] = tempfile.mkdtemp(prefix="g8_b2_")  # before any package import
from collections import Counter
from snn_data_lake.substrate.home import home
assert os.path.abspath(home()).startswith(os.path.abspath(os.environ["NORD_HOME"])), home()
from snn_data_lake.cortex import interaction_learning as ixn
from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.cortex.live_brain import _IXN_INTENTS
from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
def census(v, out):
    if isinstance(v, dict):
        if "__cls__" in v:
            out[v["__cls__"]] += 1
        for x in v.values():
            census(x, out)
    elif isinstance(v, list):
        for x in v:
            census(x, out)
    return out
def learner(**kw):
    return ixn.SpikingInteractionLearner(list(_IXN_INTENTS), 52, seed=100, t_post=6, stack=PlasticityRuleStack(len(_IXN_INTENTS), seed=100), **kw)
b = make_brain(seed=100, persist=False)
try:
    ixn._g0c_schedule(b, 100)
    ck = json.loads(json.dumps(b.brain._ixn.state_dict(), allow_nan=False))
finally:
    b.close()
a = census(ck["tree"], Counter())
print("default arm:", len(a), "classes,", sum(a.values()), "objects")
for k, n in sorted(a.items()):
    print(f"  {n:4d} {k}")
raw = json.dumps(ck)
for label, text in (("unmodified", raw), ("LIFNeuron at its M5 path", raw.replace("snn_data_lake.cortex.lif_neuron.", "snn_data_lake.growth.lif_neuron."))):
    try:
        learner().load_state_dict(json.loads(text))
        print("restore", label, "-> RESTORED")
    except ValueError as exc:
        print("restore", label, "-> REFUSED:", exc)
z = learner(readout_zoo=True)
for i in range(6):
    z.learn_turn(f"probe utterance {i}", _IXN_INTENTS[i % len(_IXN_INTENTS)])
zk = json.loads(json.dumps(z.state_dict(), allow_nan=False))
print("readout_zoo arm adds:", sorted(set(census(zk["tree"], Counter())) - set(a)))
for label, f in (("fresh zoo learner", learner(readout_zoo=True)), ("zoo learner after one respond()", learner(readout_zoo=True)), ("default learner", learner())):
    if label.endswith("respond()"):
        f.respond([0.1] * 52)
    try:
        f.load_state_dict(zk)
        print("zoo checkpoint into", label, "-> RESTORED")
    except ValueError as exc:
        print("zoo checkpoint into", label, "-> REFUSED:", exc)
d = learner()
d.delta_sink = lambda payload, grouped: None
try:
    d.state_dict()
    print("delta_sink attached: state_dict() SUCCEEDED")
except ValueError as exc:
    print("delta_sink attached: state_dict() REFUSED:", exc)
h = os.environ["NORD_HOME"]
print("left in the temporary home:", {f: os.path.getsize(os.path.join(h, f)) for f in os.listdir(h)})
EOF
```

B3, imports the package (sections 1.3, 1.6, 1.8; no brain, no lake), then the fresh-process `-m` check:

```bash
PYTHONUTF8=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -W ignore - <<'EOF'
import ast, importlib, os, re, runpy, sys, tempfile
os.environ["NORD_HOME"] = tempfile.mkdtemp(prefix="g8_b3_")  # insurance: nothing here opens a lake
from collections import Counter
from snn_data_lake.substrate import dormancy_gate as d
g = d._graph()
def closure(roots):
    seen = {r for r in roots if r in g}
    stack = list(seen)
    while stack:
        for dep in g.get(stack.pop(), ()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen
L, E, X = closure(d._LIVE_ROOTS), closure(d._ENTRYPOINTS), closure(tuple(d._external_roots()))
row = re.compile(r"^\| \d+ \| `([^`]+)` \| (?:= \(stays\)|`[^`]+`) \| [^|]* \| \w+ \| ([LE-]) \| [^|]* \|$")
rows = [m.groups() for m in map(row.match, open("specs/022-unified-brain-arc/evidence/g8-reorganization.md", encoding="utf-8").read().splitlines()) if m]
dotted = lambda k: "snn_data_lake." + k[:-3].replace("/", ".")
live = {k: "L" if dotted(k) in L else "E" if dotted(k) in E else "-" for k, _ in rows}
print("live column", dict(Counter(live.values())), "| mismatches with the table", [k for k, v in rows if live[k] != v],
      "| reached only through _external_roots()", sorted(k for k in live if live[k] == "-" and dotted(k) in X))
B = "src/snn_data_lake/brainstem/"
for f in ("reflex", "serial_arm", "so101_arm", "arm_config", "interpreter"):
    tree = ast.parse(open(f"{B}{f}.py", encoding="utf-8").read())
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("snn_data_lake") and not (n.module or "").startswith("snn_data_lake.brainstem"):
            owner = [g_.name for g_ in fns if g_.lineno <= n.lineno <= g_.end_lineno]
            print(f"envelope import: {f}.{owner[-1] if owner else '<module>'} -> {n.module} import {', '.join(a.name for a in n.names)}")
import snn_data_lake as S
fin = next(x for x in sys.meta_path if type(x).__name__ == "_AliasFinder")
print("flat aliases", len(S._MOVED), "| torch/train names aliased", [m for t in (S.TORCH_TIER, S.TRAIN_TIER) for ms in t.values() for m in ms if m in S._MOVED],
      "| 'claustrum' aliased", "claustrum" in S._MOVED, "| finder is last on sys.meta_path", sys.meta_path[-1] is fin)
for name in ("snn_data_lake.lif_neuron", "snn_data_lake.cortex.lif_neuron", "snn_data_lake.cortex", "snn_data_lake.eprop_kernel"):
    print("find_spec", name, "->", type(fin.find_spec(name)).__name__)
print("loader methods", sorted(k for k in vars(type(fin.find_spec("snn_data_lake.lif_neuron").loader)) if not k.startswith("_")))
a, b = importlib.import_module("snn_data_lake.lif_neuron"), importlib.import_module("snn_data_lake.cortex.lif_neuron")
print("identity", a is b)
try:
    runpy.run_module("snn_data_lake.lif_neuron", run_name="g8_probe")
except Exception as exc:
    print("runpy after import ->", type(exc).__name__, exc)
EOF
PYTHONPATH=src python -m snn_data_lake.lif_neuron; echo "exit $?"   # AttributeError ... 'get_code', exit 1
```

## 2. Pre-registration (PRE-REGISTERED -- FROZEN 2026-09-25 at the design read's fold, the commit that lands this text; dated amendments only after this line)

### 2.1 The owner's four decisions, verbatim as decided

As recorded in the plan's G8 row, *DATED 2026-09-25, the owner's decisions (after the G8 design draft; each the
recommended option)*: "(1) the proposed package names are kept: engine, growth, reservoir, readout, plasticity (kept),
sleep, substrate (kept, the lake), sensorimotor, infra, gates, instruments, faculties; (2) the hand-coded region
faculties move to ONE flat `faculties/`, none retired, each later getting its own P2 row, their old region kept as data
in `LEGACY_REGIONS`; (3) `brainstem/` is UNTOUCHED: the five envelope files and the SO-101 assets stay byte-identical,
and `substrate/home.py` and `substrate/arm_view.py` stay where they are because the E-STOP latch imports them; (4)
region-keyed gates read a FROZEN pre-G8 label table (`LEGACY_*`), and re-targeting them to grown-neuron labels is later
G7/G4 work." (Measured, section 1.8: the latch, `so101_arm._latch_path`, imports `substrate.home`; `So101Arm.view`
imports `substrate.arm_view`.)

### 2.2 The target layout

All packages sit one level under `src/snn_data_lake/`. Counts are section 2.12's.

| package | job | modules | of which move in |
|---|---|---|---|
| `engine/` | the brain's composition and tick: `Brain`/`make_brain`, `LiveBrain`, `AutonomousArm`, the drive loop (the daemon), the faculty ring, the typed port manifest, the report channel | 7 | 7 |
| `growth/` | the grower (neurogenesis engine, NGSNN, growth stop and program, recruiters, Net2Net), attachment and placement labels (placement, arealization, the H01 prior) and the neuron models the grower instantiates (the zoo, the Brian2 LIF authority) | 26 | 26 |
| `reservoir/` | the liquid-reservoir candidates and recurrent dynamics (the G1(b) pool, the column microcircuit, neural sampling, the criticality pool, the deterministic kernel state and tick, the OU bootstrap) | 7 | 7 |
| `readout/` | readout neurons and decoders (the chat readout, the percept readout, the grown online readout, the growable readout, the tempotron, the predictive-coding net, classifiers) | 9 | 9 |
| `plasticity/` (kept) | the tuning rules the brain tunes neurons with; its 14 stayers hold every plasticity-stack checkpoint class | 18 | 4 |
| `sleep/` | the scheduler input (circadian clock), replay, consolidation, the prune passes, nightly skill formation | 11 | 11 |
| `substrate/` (kept = the lake) | the durable lake the brain lives inside: store (home, binlog, segments, index, engine, working memory, sketches) and recall (lake recall, proximity index and replay, content signature, paradigm-B query) | 31 | 6 |
| `sensorimotor/` | encoders, ingest, the afferent and efferent boundaries, and the non-envelope arm I/O | 31 | 31 |
| `infra/` | the cockpit, telemetry and geometry, device routing, the daemon's wiring registry, the self-modification harness and self-build loop, shared helpers, export and launch entrypoints | 14 | 14 |
| `gates/` | every policy and correctness gate the battery or CI runs with `--assert`, plus G8's `move_gate` (new at S1a) | 23 (+1) | 23 |
| `instruments/` | governed measurement harnesses, probes, benchmarks and offline trainers | 23 | 23 |
| `faculties/` | the hand-coded region faculties, all live dependencies, moved not retired, flat, each pending its own P2 row | 92 | 92 |
| `brainstem/` (untouched) | the safety envelope only: `reflex`, `serial_arm`, `so101_arm`, `arm_config`, `interpreter` + the `so101/` assets | 5 | 0 |
| `reasoning/`, `training/`, `lakehouse/` (untouched) | the ungoverned tiers; the P4c program drains them, G8 does not | 115 | 0 |
| top level (untouched) | `__init__.py` (the registry and the alias layer), `brain_atlas.py`, `connectome.py` | 2 | 0 |

Region and column labels on grown neurons are G7/G3/G4 lake data, written by `growth/placement` and its neighbours; no
`columns/` or `regions/` code package exists or is created, and the frozen `LEGACY_*` tables (the code layout's history)
are never merged with the grown-neuron label vocabulary.

### 2.3 The four invariants (each asserted at every commit by C6)

1. **Governance invariance.** No module changes governance status: the 297 governed modules stay governed and the 117
   ungoverned do not move. Moving an ungoverned module into a governed package is a PROMOTION (the e-prop block), not
   a move. This keeps the p4c ratchet, `reasoning/gf.py`'s CONTROL grant, the OPEN HOLE trigger and R-b0's source out
   of play.
2. **Depth.** Every module stays exactly one package below `snn_data_lake/`, asserted per module, which covers the 59
   same-line level counters and the multi-line ones (1.5).
3. **Basename.** No module is renamed; the 414 basenames stay unique. The basename-keyed registries (dormancy EXEMPT,
   the flat alias map, the wiring manifest's `declaring_module`) and `brain_native_selfbuild`'s basename-widened
   firewall need no re-keying.
4. **Pins.** The five envelope files and the SO-101 assets stay byte-identical and are never opened for writing;
   `substrate/home.py` and `substrate/arm_view.py` stay (1.8); `signal_loop` is reached by the envelope only through the
   flat alias, which follows `REGIONS`.

### 2.4 The stages: one commit each, every commit leaving the tree fully consistent

"Fully consistent" means the cheap half (section 2.8) green at that commit; the full battery runs once, on the stacked
head. No stage is built before the previous stage's cheap half is green.

| stage | content | path-keyed items re-pointed in the same commit (besides `REGIONS`, imports, runner names and live docs) | planted un-re-pointed entry (must red) |
|---|---|---|---|
| **S0** | Checkpoint class-path translation at restore: in `interaction_learning._merge_state`, before the class comparison and before the lazy lookup, the stored module is mapped through the legacy table (identity when not legacy); the `_G0C_LAZY_ATTRS` lookup runs on the translated name; unknown names still refuse (fails closed); the lake is never rewritten and the schema stays `g0c.v1`. At S0 only its test injects a planted map. Red: R-L0 (2.7). | none | R-L0's empty-map control |
| **S0b** | The dormancy gate's phantom-EXEMPT fix: `coverage()` reports `phantom_exempt` (EXEMPT keys absent from `_CANON`), `main()` fails on a non-empty list, and the gate's self-check plants a row naming a module that does not exist, which must red. Counted first: 0 phantom rows at `68db184` (1.4), so it lands green with its red shown. | none | the planted phantom row |
| **S1a** | The G-R8 instrument `move_gate` in a new `gates/` package (registered: a `gates` key, a runner name, a permanent dormancy EXEMPT row), together with what keeps the label consumers off that key: `LEGACY_REGIONS`, `LEGACY_TORCH_TIER`, `LEGACY_TRAIN_TIER` as byte-identical frozen copies of the S0-head tables, placed after the live tables; `package_of()`/`module_path()`; the label consumers switched to `LEGACY_*` (`brain_atlas.region_of`, `region_charge_gate`, `atlas_node_gate`, `brain_geometry`, `region_wiring`), taking LABELS and MEMBERSHIP from `LEGACY_*` and module PATHS from `module_path()`, with `brain_atlas._region_files` iterating `module_path()` files, never `dirname(pkg.__file__)` (F5); `selfcheck_cache`'s roster anchor becomes the regex `(?m)^REGIONS = \{` (F17). The pre-S1a census: 2.6. | `import_manifest.GOVERNED` gains `gates` (F15) | an EXEMPT-row deletion (C1); F5's join red: in a temporary copy, one module moved with the joins in place leaves both region censuses unchanged, and the same move with the path taken from `LEGACY_*` (the draft's form) reds (reachability lost) |
| **S1b** | The extended alias layer (2.5): `_LEGACY` specs carrying the target's `origin` and `has_location=True`, their `get_code` delegating under the target's name, the flat alias unchanged (F2); `_LEGACY_PKGS` computed from the tables (F4); the `SNN_ALIAS_STRICT=1` tripwire (F3(c)); in `move_gate`, C8 and C9 resolving by file (F3(a)) and C1 comparing NAME-valued registries raw (F3(b)); the R-L1 baseline written at this head (2.7). Its C1 diff must be EMPTY: the positive null (2.11 Q2). | none | a synthetic legacy package and the `claustrum` collision built in a temporary copy (C2(b)), so those reds are reachable before M11a; a planted legacy import raises under strict (F3(c)'s positive control); a legacy `-m` of a gate that reads `__file__` at import, that gate moved in a temporary copy, exits exactly as the real path does, the flat alias's `AttributeError` and the `get_code`-only `TypeError` its controls (F2) |
| **S1c** | POLICIES.md P3 "ANATOMY IS DATA" reads "...until specs/022's G7 and G4 re-key them; G8 freezes both on the pre-G8 `LEGACY_*` table (owner decision, 2026-09-25)" or an equivalent wording (F9). The owner's decision in chat outranks POLICIES; this is POLICIES owing the fix. | `tools/policy_manifest.json` regenerated; the word delta measured against `doc_budget_gate` and declared before the gate runs (33 words of headroom at `f416296`; reworded to net zero if short) | an undeclared extra word in POLICIES.md (C1's doc-budget diff) |
| **M1** | `gates/` <- 23 substrate gates; `policy_manifest.token_to_argv` and `cross_check` derive from the registry: a `<pkg>/<mod>.py` token naming a governed module becomes `-m snn_data_lake.<pkg>.<mod>` and `in_selfchecks = stem in mods`, `tools/` tokens unchanged (F1) | ci.yml's 15 gate calls; PLAN A.4's battery line (the templated `-m snn_data_lake.substrate.$m` and its two literal `policy_manifest` calls), the commit message naming its two out-of-repo copies (F12); 7 SCOPE commands, among them `POLICIES_EVIDENCE.md` lines 410 and 476, each with its historical path in a dated note beside it (F10; the notes' word delta declared against `doc_budget_gate`, 28 words of headroom at `f416296`, reworded to net zero if short); `selfcheck_cache`'s `substrate.dormancy_gate` subject; POLICIES.md gate paths (24) with `policy_manifest.json` regenerated | `selfcheck_cache`'s subject (C1, C2(d); the cache falls back silently); F1's: a `gates/` token must be in_ci and PASS under `policy_manifest --audit`, and the pre-fix code reds on it in a temporary copy |
| **M2** | `instruments/` <- 23 | the G1(a)/G1(b) harness paths in the live plan text (frozen evidence keeps them); `brian2_backend.MIGRATED_CALL_SITES` (2); the self-launch strings of `memory_diagnostic`, `pool_selection`, `everyday_battery`; a line-keyed citation_guard EXEMPT row for `t081-report-order.md` line 272, which is prose and stays unrewritten (F10; not earlier: a stale exemption reds); `GOVERNED` + `instruments` | `MIGRATED_CALL_SITES`' `plasticity.deep_stdp` (C1, C2(d)) |
| **M3** | `sensorimotor/` <- 31, including the five non-envelope `brainstem/` modules | pyproject `snn-arm`, `snn-loop`, `snn-fetch-corpus`; `_ENTRYPOINTS` `live_arm`, `closed_loop`, `web_ingest`; REGISTRANTS (2); 17 SCOPE commands; `tests/pin/capture.py`; the specs/005 quickstart's line 76 path `spiking_lm_serve`, its historical path in a dated note (F10); `GOVERNED` + `sensorimotor` | `_ENTRYPOINTS`' `brainstem.live_arm` (C1) |
| **M4** | `infra/` <- 14 | pyproject `snn-demo`, `snn-nengo-launch`; `_ENTRYPOINTS` `demo`, `cockpit`, `nengo_bridge`, `region_wiring`; `snn_ui_main.py`; `brain_native_selfbuild`'s REAL-path firewall literal; `GOVERNED` + `infra` | `snn_ui_main.py`'s import (C9, by file) |
| **M5** | `growth/` <- 26 | `snn-ui.spec`'s exclude; the e-prop register site `hippocampus/ng_snn.py::fit`; ci.yml's `brian2_backend` call; `brian2_backend.MODULE_NAME` and 13 `CUSTOM_KERNELS`; REGISTRANTS `growth_stop`, `learning_progress`; `growth_coverage_gate._PLACEMENT_MODULES`; `growth_coverage_gate.py:188`'s `endswith("cortex")`, which becomes the package of `module_path("placement")` (F7); `neurogenesis_engine`'s self-launch string; `GOVERNED` + `growth`. **R-L1's first decisive stage** (`LIFNeuron`; the zoo classes) | `snn-ui.spec`'s exclude (C9, by file; PyInstaller itself says nothing) |
| **M6** | `reservoir/` <- 7 | `EXEMPT_METHOD` `spontaneous_activity`; `CUSTOM_KERNELS` `kernel_tick`; the G1(c) migration-host text; `GOVERNED` + `reservoir` | the `EXEMPT_METHOD` key (C1, C2(d)) |
| **M7** | `plasticity/` <- 4 tuning rules (`r_stdp`, `valence_stdp`, `eprop_kernel`, `naca`); `readout/` <- 9 (including `interaction_learning`) | the e-prop register site `cortex/predictive_coding.py`; `EXEMPT_METHOD` `predictive_coding`; REGISTRANTS `r_stdp`; `CUSTOM_KERNELS` `eprop_kernel`, `r_stdp`; `interaction_learning`'s three self-launch strings; `GOVERNED` + `readout`. **R-L1's second decisive stage** | the POLICIES.md register site (C7) |
| **M8** | `sleep/` <- 11 | `GOVERNED` + `sleep`; nothing else beyond the defaults | one runner name (C1, C8) |
| **M9** | lake recall into `substrate/` <- 6 | none (`proximity_replay` launches `-m type(self).__module__`) | one live-doc citation of `hippocampus/lake_recall.py` (C7) |
| **M10** | `engine/` <- 7 | `replication_guard.CANON` (3); `_LIVE_ROOTS` (3); the root `learn_daemon.py` shim; POLICIES.md P4's `cortex.brain.Brain`/`make_brain()` names with `policy_manifest.json`; REGISTRANTS `brain`, `autonomous_loop`, `report_channel`; `selfcheck_cache`'s `substrate.signal_ring` subject; the specs/005 quickstart's line 76 path `brain.py`, its historical path in a dated note (F10); `tools/p6_gate.py:1733,1811`'s `os.path.join(source_dir(root), "cortex", ...)`, re-pointed through `module_path()` (F7); `GOVERNED` + `engine` | `CANON`'s `Brain` (C8, by file) |
| **M11a** | `faculties/` <- 1: `claustrum` alone (the module named like its package, 1.2); deletes `claustrum/` with its `__pycache__/` | `GOVERNED` + `faculties`, - `claustrum` (F15); `_LEGACY_PKGS` gains `claustrum`, which rule (c) keeps out of `_MOVED` (2.5) | `claustrum` put back in `_MOVED` (C2(b)); `claustrum/` left holding only its `__pycache__/` (C6, the namespace shadow) |
| **M11b** | `faculties/` <- 12 from nine packages: `amygdala` 2, `hypothalamus` 2, `medulla` 2, `brainstem` 1 (`cpg_oscillator`), `epithalamus` 1, `insula` 1, `plasticity` 1 (`wm_capacity`), `pons` 1, `substrate` 1 (`region_coherence`); deletes `amygdala/`, `epithalamus/`, `hypothalamus/`, `insula/`, `medulla/`, `pons/` and the already empty `olfactory/` with its `REGIONS` key, each with its `__pycache__/` | REGISTRANTS (4: `interoceptive_arousal`, `homeostatic_drives`, `interoceptive_integration`, `region_coherence`); `GOVERNED` - `amygdala`, `epithalamus`, `hypothalamus`, `medulla` (F15) | the REGISTRANTS row `snn_data_lake.insula.interoceptive_integration` (C8: no file at that path; the alias raises under strict) |
| **M11c** | `faculties/` <- 22: `neuromodulation` 12, `hippocampus` 10; deletes both directories | `EXEMPT_METHOD` `neuromodulation.developmental_envelope.DevelopmentalEnvelope.update`; REGISTRANTS (2: `raphe_serotonin`, `astrocyte_modulation`); 1 SCOPE command (`docs/RUNNING.md`, `spiking_storage_prototype`); `GOVERNED` - both (F15) | the `EXEMPT_METHOD` key (C1, C2(d)) |
| **M11d** | `faculties/` <- 24: `cerebellum` 9, `thalamus` 6, `basal_ganglia` 5, `midbrain` 4; deletes the four directories | `_LIVE_ROOTS` `reentrant_workspace`; `EXEMPT_METHOD` (4: `spiking_jepa`'s two `train_step` keys, `latent_q`'s two `update` keys); POLICIES.md's 5 citations, among them the e-prop register sites `jepa_world_model` and `mlp_q`, with `policy_manifest.json` regenerated; REGISTRANTS (5: `evidence_accumulation`, `interoceptive_relay`, `dopamine_nuclei`, `global_workspace`, `reentrant_workspace`); `CUSTOM_KERNELS` `spiking_pid`, `gwt_deep`; 2 SCOPE commands (`docs/GAP_CLOSURE.md`, `docs/JEPA.md`); `atlas_node_gate.py:165` (`SRC / "thalamus"`) and `:199-203` (`REGIONS.get("cerebellum")`, `SRC / "cerebellum"`), its self-check corpus built from `module_path()` and asserting at least 1 file scanned (F7); `thalamus/global_workspace.py`'s own local `REGIONS` tuple left alone (2.5 item 2); `GOVERNED` - the four (F15) | `_LIVE_ROOTS`' `thalamus.reentrant_workspace` (C1) |
| **M11e** | `faculties/` <- 33: `cortex` whole, the one sub-stage above 25 (a partial package is never moved); deletes `cortex/` | `reasoning_cortex`'s three `sys.modules` keys derived with `module_path("kb_ranking")`, and `schema_induction`'s `_KB` neuter key (F16); the specs/005 quickstart's line 75 path `language_acquisition`, its historical path in a dated note (F10); REGISTRANTS (3: `percept_surprise`, `task_set`, `recursive_tom`); 1 SCOPE command (`docs/coverage/Language.md`, `rose_language`); `GOVERNED` - `cortex` (F15) | `schema_induction`'s `_KB` (C2(d)); its own confidence-gate red and `reasoning_cortex`'s asserts must still fire in this sub-stage's cheap half (F16) |
| **D1** | The A.5 old-to-new map (253 rows, generated from `LEGACY_*`); the content rewrites of the index documents (ANATOMY, OVERVIEW, CODEBASE_DIGEST, README "Architecture", AGENT_COLD_START section 1 under its SYNC DUTY); POLICIES.md's `REGIONS` wording if not already landed | `policy_manifest.json` regenerated; the doc-budget word counts declared before the gate runs | a planted dangling live citation (C7) |

**The M11 split** (the design read, F4; 2.11 Q3). Generated from section 2.12's table with B1's row parser: the 92
`faculties` moves come from 17 source packages. `claustrum` is M11a alone. The other 16, in descending order of their
faculty-bound modules (ties by name), are cut into consecutive groups of at most 25 modules, and the groups run smallest
first: M11b 12, M11c 22, M11d 24, M11e 33. A sub-stage holds whole source packages only and deletes every directory it
empties. `cortex` holds 33 faculty-bound modules in one package, so M11e is the one sub-stage above 25 (the
orchestrator's ruling at the fold: cortex stays whole). The already empty `olfactory/` goes at M11b with its `REGIONS`
key. The table above distributes M11's path-keyed items and planted reds to the sub-stages that hold them. The stages
are 21 commits: S0, S0b, S1a, S1b, S1c, M1-M10, M11a-M11e, D1.

Stage sizes at `68db184` (B1). The M11a-M11e rows are B1's own instruments with the `faculties` destination keyed by
source package through the grouping above; they sum to the M11 row, which is their positive control, and M1-M10
reproduce unchanged:

| stage | moves | import statements (files) | live-doc citations | frozen-doc citations | `.py` path citations | runner names | SCOPE commands |
|---|---|---|---|---|---|---|---|
| M1 gates | 23 | 51 (39) | 130 | 249 | 227 | 23 | 7 |
| M2 instruments | 23 | 99 (52) | 47 | 96 | 107 | 23 | 1 |
| M3 sensorimotor | 31 | 104 (72) | 103 | 137 | 101 | 31 | 17 |
| M4 infra | 14 | 115 (95) | 51 | 130 | 64 | 14 | 2 |
| M5 growth | 26 | 139 (90) | 84 | 146 | 148 | 26 | 1 |
| M6 reservoir | 7 | 23 (14) | 16 | 24 | 21 | 7 | 0 |
| M7 tuning + readout | 13 | 59 (46) | 49 | 119 | 88 | 13 | 0 |
| M8 sleep | 11 | 35 (20) | 28 | 23 | 25 | 11 | 0 |
| M9 lake recall | 6 | 20 (15) | 10 | 9 | 8 | 6 | 2 |
| M10 engine | 7 | 196 (95) | 361 | 319 | 274 | 7 | 1 |
| M11a faculties (`claustrum`) | 1 | 2 (2) | 2 | 1 | 1 | 1 | 0 |
| M11b faculties (nine packages) | 12 | 35 (16) | 36 | 25 | 22 | 12 | 0 |
| M11c faculties (`neuromodulation`, `hippocampus`) | 22 | 65 (36) | 55 | 33 | 43 | 22 | 1 |
| M11d faculties (`cerebellum`, `thalamus`, `basal_ganglia`, `midbrain`) | 24 | 66 (46) | 110 | 98 | 153 | 25 | 2 |
| M11e faculties (`cortex`) | 33 | 114 (66) | 196 | 164 | 186 | 33 | 1 |
| M11 faculties, the five together | 92 | 282 (120) | 399 | 321 | 405 | 93 | 4 |
| **total** | **253** | **1,123 (369)** | **1,278** | **1,573** | **1,468** | **254** | **35** |

M11d's 25 runner names include the one comment mention (1.4). No import statement names modules of two M11
sub-stages. `tools/rule_census.json` is counted in the frozen column here (B1's class); from S1a it is a live,
regenerated registry (F6), and section 3 re-derives both columns per stage.

### 2.5 What every move commit carries

1. `git mv` of the stage's modules, which change only in import lines and path strings (git sees renames).
2. Imports, dotted strings and path strings rewritten in every live consumer: `src/` including the 48 `reasoning/`
   files that import moved modules, `tests/` (102 files), `tools/` (15), the root scripts, `integrations/`, the
   top-level `research/` (10) and the repo-root `training/`. The rewrite is a map lookup on exact `(old package,
   module)` pairs, never a bare-word substitution (`thalamus/global_workspace.py` has its own local `REGIONS` tuple).
3. **Beyond the pair map, named per stage in 2.4:** templated forms (A.4's `snn_data_lake.substrate.$m`),
   `sys.modules` keys and `__name__` comparisons (derived with `module_path()` where one module keys another, F16),
   subprocess `-m` self-launch strings (rewritten, or derived from the module's own `__spec__.name`), path-form
   literals, and split path parts and package-name tests (`os.path.join` and `Path /` operands, `endswith`/`startswith`,
   `REGIONS.get("<pkg>")`), re-pointed through `module_path()` (F7).
4. `REGIONS`/`TORCH_TIER`/`TRAIN_TIER` re-keyed for the stage; `LEGACY_*` never edited after S1a.
5. The runner's names in `run_selfchecks.py` (253 strings, 1 comment; section 1.4). A stale name no longer fails under
   `runpy` once S1b lets a `_LEGACY` name run (F2): it is caught by C1's raw comparison of the runner's name set
   (F3(b)), and it raises under `SNN_ALIAS_STRICT=1` (F3(c)).
6. The stage's path-keyed registries (2.4). `import_manifest.GOVERNED` gains each new package in the stage that
   creates it (S1a `gates`, M2 `instruments`, M3 `sensorimotor`, M4 `infra`, M5 `growth`, M6 `reservoir`, M7 `readout`,
   M8 `sleep`, M10 `engine`, M11a `faculties`), keeps `lakehouse`, and loses each emptied package in the M11 sub-stage
   that deletes its directory, because `import_manifest.py:173` skips a listed name with no directory without a word
   (F15). **The rule census (F6):** in each stage that moves a rule-bearing module, the pair map is applied to
   `rule_composer._CENSUS_ADJUDICATED` first, then `python -m snn_data_lake.plasticity.rule_composer --regen-census`
   runs in the same commit (the regen refuses an adjudicated entry that matches no live row); `tools/rule_census.json`
   is a live registry, not frozen text, and the mover re-derives each stage's moving-row count into section 3.
7. Live documents: every citation of the stage's modules, and every `python -m` command in doc_command_guard's SCOPE.
   **A runnable command inside a frozen document is rewritten at its stage, its historical path kept in a dated note
   beside it** (the plan's G8 row, dated amendment of 2026-09-25; F10), outside the runnable lines (a `.py` path on a
   runnable line must resolve): `docs/POLICIES_EVIDENCE.md` lines 410 and 476 at M1 (doc_command_guard resolves `-m`
   by file and rejects a flat alias), and the specs/005 quickstart's `git diff --quiet` paths, line 75 at M11e and line
   76 at M3 and M10 (a stale path passes vacuously). A note's words in a budgeted document are declared against
   `doc_budget_gate` before the gate runs, reworded to net zero if the headroom is short. `t081-report-order.md` line
   272 is prose that citation_guard's `reproduc` heading rule catches: it is not rewritten, and a line-keyed
   citation_guard EXEMPT row lands in M2's commit (not earlier: a stale exemption reds). All other frozen text keeps its
   historical paths, resolved through the A.5 map.
8. `ruff check --fix --select I` and `ruff format` on the touched files (longer package names re-sort and re-wrap).
9. Line endings preserved per file; the envelope files are never opened for writing.
10. Any package directory the stage empties is deleted in the same commit, with its ignored `__pycache__/` (F4): all at
    M11a-M11e, and the already empty `olfactory/` at M11b. `_LEGACY_PKGS` is computed from the tables, never from the
    filesystem, so a directory left behind is C6's red, not a silent shadow.
11. The stage's G-R8 census JSON is recorded in section 3, beside its declared delta (2.6).

The mover is a deterministic script driven by the frozen map. Its source is archived verbatim with the arc (the memory
store's `archive/g8-2026-09-25/`), beside its hash in section 3, so any stage can be rebuilt for attribution (2.8;
F18). Its P4c disposition: an out-of-tree operator artifact like the battery script of record, never committed to
`tools/` (a new `tools/*.py` would raise the p4c `tools` zone, baseline 30, and red the ratchet); `move_gate` is its
governed checker.

**The extended alias layer (S1b).** `_LEGACY` maps each old `pkg.mod` to the current `pkg.mod`, derived from `LEGACY_*`
and the live registry, holding only the pairs that differ (empty at S1b, growing by itself). `_LEGACY_PKGS` is every
`LEGACY_*` package key minus every live package key of the three tiers, computed from the tables and never from the
filesystem (empty until M11a; F4). `find_spec` order: (a) a name in `_LEGACY_PKGS` gets a synthetic package spec
(`is_package=True`, empty `submodule_search_locations`, a docstring naming G8); (b) a `_LEGACY` path gets an identity
alias whose spec carries the target's `origin` and `has_location=True`, with a loader whose `get_code(fullname)`
delegates to the target's real loader **under the target's name** (1.6; `runpy` sets `__file__` from `spec.origin`,
F2); (c) flat names exactly as today, with every `_LEGACY_PKGS` name removed from `_MOVED` (`claustrum`). The flat
alias serves imports only and is never run with `-m`: `doc_command_guard`'s premise that a flat alias cannot run under
`-m` stays true, and nothing in G8 needs one (F2). **The tripwire (F3(c)):** with `SNN_ALIAS_STRICT=1` the finder raises
on any `_LEGACY` or synthetic-package resolution and never on a flat alias, which the envelope's own self-check uses
(`interpreter.py` imports `snn_data_lake.signal_loop` inside it); 2.8 names the checks that run with it on. The finder
stays appended, so a real file always wins (C6 asserts no `.py` at a legacy path and no legacy package directory).
Identity is the contract: `sys.modules[legacy] is sys.modules[new]`. It gives one object two names, so it is not a
second mechanism (P2); live code may not use legacy names (C1, C2(d)); it serves the lake's old class strings, frozen
evidence and callers outside the repo.

### 2.6 G-R8 and its controls

G-R8 (Part F, as amended): red when a gate's output changes by more than the delta declared in the move's
pre-registration before its first gate run, an import path or back-compat alias breaks, an envelope hash changes, a
live-lake checkpoint no longer restores, or G1B's R-b0 source-parity check cannot find its sources; a red reverts the
move. Control: the pre-move tree's gate outputs, re-derived; the pre-move checkpoint restore; a planted un-re-pointed
path-keyed entry, which must red.

**The instrument** is `gates/move_gate` (governed, landing at S1a; F3's parts at S1b): `--census <out.json>`; `--diff
<pre.json> <post.json> --declared <delta>`; and `--self-check`, which builds every red below in a temporary copy of the
tree, never in the checkout. OUTCOME sets (reachable sets, verdicts, counts) are mapped through `LEGACY_*` before
comparison; NAME-valued registries are compared RAW against the post-stage set generated from the 2.12 table (F3(b),
C1). A pure move therefore has an empty delta, and a stale name is a red even where the alias would import it.

**The pre-S1a census.** `move_gate` does not exist before S1a, so S1a's own `move_gate` takes the pre-stage census by
running over an export (`git archive`) of the S0b head; this is how S1a's delta is still declared before the first gate
run. Wherever `move_gate` itself changes (S1b), the parent's census is re-taken the same way, so both sides of every
diff come from one instrument.

**The declared deltas, pre-registered here (before S0's first gate run):**
- S0: none.
- S0b: the dormancy report gains `phantom_exempt`, empty.
- S1a: `move_gate`'s own registration and nothing else: dormancy total 297 -> 298 and exempt 59 -> 60 (reachable 238
  unchanged), `REGIONS` keys 18 -> 19 (`gates`), one more runner name, one more atlas module, `import_manifest.
  GOVERNED` + `gates`. The region-charge and atlas-node censuses are unchanged (they read `LEGACY_*` from this commit).
  This replaces the draft's "the S0 -> S1 diff must be EMPTY", which S1's own content makes impossible (1.11 item 3).
  The label consumers switch in this same commit because a `gates` key they still read from the live table adds a
  `gates: UNREACHABLE` row to the region-charge census (16 -> 17 rows, reproduced; 2.13, the S1 re-cut).
- S1b: EMPTY, the positive null (2.11 Q2): the alias layer, the tripwire, `_LEGACY_PKGS` and the R-L1 baseline change
  no census.
- S1c: POLICIES.md's word delta as declared before the gate runs, and nothing else.
- M1-M10: none, except that the live key sets of `REGIONS`, `TORCH_TIER` and `TRAIN_TIER` become exactly those the
  section 2.12 table implies after the stage (generated from the table, never hand-written; for `REGIONS` the new keys
  are M2 instruments, M3 sensorimotor, M4 infra, M5 growth, M6 reservoir, M7 readout, M8 sleep and M10 engine, while
  M1 and M9 add none), `GOVERNED` gains the stage's new package (2.5 item 6), and citation_guard's report-only BROAD
  count may rise by at most the stage's frozen-document citations (2.4). The BROAD count in LIVE files may not rise,
  except by the envelope carve-out below.
- **The envelope carve-out (F11).** `so101_arm.py` is never opened for writing, so its comment citations of moving
  modules dangle from their stages on: citation_guard's BROAD count in live files may rise by exactly these lines and
  by nothing else, M1 2 (604, 605), M5 1 (243), M9 1 (241), M11d 3 (602, 603, 607) and M11e 1 (606). The owner is
  told: a comment-only envelope edit is theirs to approve.
- M11a-M11e: the same key-set rule. `REGIONS` gains `faculties` at M11a and loses each anatomy key in the sub-stage
  that empties its package: M11a `claustrum`; M11b `amygdala`, `epithalamus`, `hypothalamus`, `insula`, `medulla`,
  `pons` and the already empty `olfactory`; M11c `hippocampus`, `neuromodulation`; M11d `basal_ganglia`, `cerebellum`,
  `midbrain`, `thalamus`; M11e `cortex` (13 keys remain after M11e). `GOVERNED` follows 2.5 item 6. `import_manifest`'s
  scanned set gains `insula/interoceptive_integration` and `pons/mossy_relay` at M11b (the pre-existing gap, closed as
  a declared delta).
- D1: documents only; the doc-budget word counts are declared in section 3 before the gate runs.
- Every stage from S1b's baseline on: R-L1's readings identical (no delta is ever declared for the lake).

| control | asserts | the constructed red |
|---|---|---|
| **C1 gate census diff** | Two classes, taken with `SNN_ALIAS_STRICT=1` (F3(c)). OUTCOME sets, timings stripped and paths mapped through `LEGACY_*`: the dormancy reachable set and EXEMPT verdicts; the method-dormancy lists; import_manifest targets and broken list; replication sites; the region-charge census; atlas phantoms and exemptions; the manifest rebuild equal to the committed NIR bytes; envelope findings; the gradient-claim and transport censuses; growth coverage; p4c zone counts; p6 counts; citation STRICT and BROAD; doc_command; the doc-budget word counts; brain_atlas; the rule census's dispositions, old keys mapped to new (F6); `policy_manifest --audit`'s per-command verdicts (F1); the exit codes of determinism, replay, capability, immortal, conclusion and policy_manifest. NAME-valued registries, read RAW and never through `LEGACY_*`, against the post-stage set generated from the 2.12 table (F3(b)): the dormancy `EXEMPT` keys, `_LIVE_ROOTS`, `_ENTRYPOINTS`, `EXEMPT_METHOD`, `import_manifest.GOVERNED` (2.5 item 6), `replication_guard.CANON`, `REGISTRANTS`, the runner's name set, `MIGRATED_CALL_SITES`, `CUSTOM_KERNELS`, `MODULE_NAME`, the `selfcheck_cache` subjects, `_PLACEMENT_MODULES`, `tests/pin/capture.py`'s pin key, the pyproject script targets, ci.yml's `-m` targets, `snn-ui.spec`'s hidden imports and excludes, `snn_ui_main.py`'s and the root shim's imports, the envelope-name registries (`envelope_hashes.json`, `ENVELOPE`, `_PROTECTED`, `_SAFETY_KERNEL`) and `tools/rule_census.json`'s keys (F6); documents stay C7's. The diff must equal the declared delta. | Each stage's planted un-re-pointed entry (2.4) applied in a temporary copy must red; a planted EXEMPT-row deletion must red at S1a while the real diff equals the declared delta. |
| **C2 import and alias** | (a) the importable set is unchanged: a fresh-process import of all 414 modules by their current path, in a scratch `NORD_HOME`; the baseline at `f416296` is 414 attempted and 13 failing (lakehouse 2, reasoning 11; none in a moving package), the full list re-recorded at S0's head in section 3 (F13); (b) every `_LEGACY` path, synthetic package and flat alias `is` the current module, run on the real checkout as well as in the temporary copy, because an ignored `__pycache__/` exists only there (F4); (c) `python -m <legacy gate> --assert` exits as the new path does, for the 16 battery gates and `brian2_backend`, through a `_LEGACY` name (the flat alias is never run with `-m`; F2); (d) zero live in-tree imports or strings naming a legacy path, statically, in every live `.py` of every tier (governed and ungoverned, tests, tools, integrations, research, root scripts), dotted and path forms, plus the templated forms, plus split path parts (string operands of `os.path.join` and `Path /`), `endswith`/`startswith` package tests and package-prefix strings, AST-scoped to code (F7); (e) the frozen `docs/coverage/artifacts/*.py` import through the alias. (b), (c) and (e) run with `SNN_ALIAS_STRICT` off. | Drop one `_LEGACY` entry: (b) reds. Leave one test importing `snn_data_lake.cortex.brain`: (d) reds. Put `claustrum` back in `_MOVED`: (b) reds (built at S1b in a temporary copy, again at M11a for real). (d)'s positive control: run with the complete map over the `f416296` tree, it must find F7's six known sites (`p6_gate.py:1733` and `:1811`; `growth_coverage_gate.py:188`; `atlas_node_gate.py:165`; `:199-203`'s `REGIONS.get("cerebellum")` and `SRC / "cerebellum"`) before its zero is believed. |
| **C3 envelope** | `envelope_hash_gate --assert` green; `git diff --stat` empty for the five files and `tools/envelope_hashes.json`; the `safety_envelope` document check green; in a fresh process `so101_arm._latch_path()` resolves through the real `substrate.home` with a file loader, not an alias. | The gate's own self-check (its positive control and three reds); a planted `git mv` of `brainstem/reflex.py` in a temporary worktree gives "envelope file MISSING on disk". EOL caution: a fresh worktree checks `serial_arm.py` and `so101_arm.py` out CRLF and reds with no change; the documented `git -c core.autocrlf=false checkout --` remedy comes first. |
| **C4 R-b0 source parity** | `recurrent_pool.check_r_b0()` (one check, `check_r_b0_lsm`, one source, 1.9) at the pre and post heads: identical verdict and numbers. | Its own `red_leak`; or `reasoning/spiking_ssm.py` renamed in a temporary copy with no alias: ModuleNotFoundError. |
| **C5 lake continuity** | R-L1 (2.7). | The translation disabled; `lif_neuron` or `r_stdp` moved without the map. |
| **C6 static invariants** | Depth unchanged for every module; basenames unique; no `.py` at any legacy path; no directory for any `_LEGACY_PKGS` name on the REAL checkout, ignored files included (F4); `LEGACY_*` AST-equal to the S0 head's tables; no governed module in an ungoverned package; the three registry tables still AST literals (p4c's `_governed` depends on it). | A planted `growth/models/lif_neuron` module: the depth red. A planted `cortex/brain` stub: the shadow red. `claustrum/` left holding only its `__pycache__/` after M11a's move, in a temporary copy: the namespace-shadow red. |
| **C7 documents** | `doc_command_guard --assert` and `citation_guard` (STRICT) green; zero new BROAD dangling citations in LIVE files after mapping, except the envelope carve-out's named lines (F11); the frozen rise within its declared bound; the frozen runnable commands rewritten with their dated notes (F10). | Skip rewriting one live document; leave one runnable frozen citation unrewritten: STRICT reds. |
| **C8 registry parity** | Every name resolved by FILE, never through an alias: it must map to an existing `src/snn_data_lake/<pkg>/<mod>.py`, and an alias spec counts as red (F3(a)); brain_atlas map equal to the filesystem; the runner's tiers agree with the registry; `CANON` resolves; the wiring manifest byte-identical; p4c zone counts identical. Run with `SNN_ALIAS_STRICT=1`. | A stale `REGISTRANTS` row: no file at its path, and the alias raises under strict. |
| **C9 packaging and CI** | Resolved by FILE as in C8 (F3(a)): every pyproject script target, every `snn-ui.spec` hidden import and exclude and every ci.yml `-m` target maps to an existing module file, and each `-m` target has a `__main__`. Run with `SNN_ALIAS_STRICT=1`. | `snn_data_lake.hippocampus.ng_snn` left in the spec's excludes. |

### 2.7 The lake reds, re-designed to the measured checkpoint

- **R-L0 (S0, the mechanism).** In a fresh process with its own temporary home: `make_brain(seed=100, persist=True)`,
  taught with the frozen `_g0c_schedule`, flushed. The saved checkpoint's `__cls__` strings for **all three moving
  classes** (`SpikingInteractionLearner`, `RSTDPNeuron`, `LIFNeuron`) are rewritten to planted legacy names and a test
  map from those names to the current ones is injected; the restore runs in a second fresh process. PASS iff the
  restore succeeds, the `_g0c_walk` digest equals the writer's, the answers on a fixed probe set are equal and no
  refusal is reported. Controls: the same checkpoint with an EMPTY map must refuse "checkpoint class mismatch" (shown at
  design time for `LIFNeuron` alone, 1.7); an unmapped unknown class must refuse; a name mapped to a missing target must
  refuse.
- **R-L1 (the baseline at S1b's head, then at S1c's and every M head).** Arm (i), default: checkpoint home H written by
  a brain at S1b's head (`persist=True`, a temporary home, the frozen schedule); at each later head a fresh process
  constructs the brain over a fresh COPY of H. Arm (ii), zoo: a learner-level checkpoint (`readout_zoo=True`, the six
  fixed turns of B2) written at S1b's head; at each later head a fresh process restores it into a zoo learner that has
  built its pools with one `respond()` on a fixed input, the only target that restores it today (1.7). Check (iii),
  `delta_sink`: not a restore arm; its `state_dict()` refusal is re-asserted at every head. PASS iff (a) no refusal (the
  probe patches `_report_ixn_refusal` to raise, so the swallow cannot hide one), (b) the restored walk equals the
  pre-move walk: the S1b walk JSON is stored beside H and its class strings are mapped old->new through `LEGACY_*`
  before comparison (a digest cannot be mapped, and `_g0c_walk` carries `"<module>.<qualname>"` class strings; F14),
  and (c) the probe answers are identical.
  - Reachable red: with the translation disabled, arm (i) refuses at M5 (`LIFNeuron`) and again at M7; arm (ii) refuses
    at M5. A planted stage moving `lif_neuron` without the map refuses.
- **Hygiene.** One subprocess per arm (`throwaway_home` sets the process-global `NORD_HOME`); `home()` asserted under
  the temporary directory before any construction; H copied, never reused (a restore followed by a flush writes to
  it); the live lake is never opened. The probe leaves a lake file in its home even when nothing persists (1.7). R-L1's
  probe asserts that the restore did not refuse BEFORE any flush can run: a refused restore is swallowed and the flush
  then saves the fresh learner over the taught one (F8; 2.10).
- **A second lake string, no action:** `learning_progress.GoalStore` rows store an `achievement_function_ref` dotted
  string (`snn_data_lake.cortex.brain.predict_accuracy` in its self-check), stored and never resolved; a future
  resolver must translate it through `LEGACY_*`.

### 2.8 The per-commit cheap half, and ONE battery on the stacked head

- **Per commit, before the next stage is built:** C1 (census and diff against the declared delta), C2 (a)-(e), C3, C4,
  C5 (R-L1's arms, from S1b's baseline on), C6, C7, C8, C9, `policy_manifest --audit` (it executes the gate commands;
  F1) and the stage's planted red; `ruff check` and `ruff format --check` on the touched files. C1, C8, C9 and the
  stage's gate runs execute with `SNN_ALIAS_STRICT=1`; C2 (b), (c) and (e) with it off (F3(c)). M11e's cheap half also
  runs `reasoning_cortex`'s and `schema_induction`'s own self-checks, whose neutered reds must still fire (F16). A
  stage whose cheap half is not green is not built upon.
- **ONE battery** (PLAN A.4's literal block, opening with the record assert that halts on red) on the stacked G8 head,
  after the three-lens panel and its fold, per the owner's rule of 2026-09-02. Every commit message names that battery
  ("verified by the arc's battery, named in the closing record commit"). The record states which form of A.4's
  templated gate line the battery ran, and the orchestrator updates its two out-of-repo copies to `gates.$m` at the
  push (SYNC DUTY; M1's commit message names them; F12). A red on the head is attributed by re-running the failing
  stage per commit in a throwaway worktree with the archived mover (2.5; F18), after the EOL remedy and the
  ignored-assets junctions of the protocol's 2026-09-22 note; an unattributable red does not push. A full battery per
  commit (about 21 x 80 min) is not run.

### 2.9 The daemon, the cost, the slot

- **The daemon stays stopped through G8** (the owner stopped it on 2026-09-24; no answer on its port at this writing,
  1.7). It relaunches itself on new commits when running (protocol section 4), so a restart during G8 would run each
  stage's moved code against the live lake. Its start or stop is the owner's word only.
- **The cost is measured on ONE stage before the rest.** Known unit costs: B1 4 s, B2 16 s per process. Before M1 is
  built, the full cheap half's wall time at S1c's head is recorded in section 3; M1's build and cheap half are timed
  before M2 is built; the per-stage estimate is re-stated from those two readings before M2. If the projection exceeds
  two working sessions, the plan is re-cut before M2 by a dated amendment. The draft's unit estimates stand until then,
  re-stated for the 21 commits (S0, S0b, S1a-S1c, M1-M10, M11a-M11e, D1): 15-30 min of build and review and 10-15 min
  of cheap half per commit, about 9-16 h in all, plus one battery of 65-95 min and a panel of about 25 min.
- **The slot:** after G1(b)'s selection record, its losing-port deletion and the stack's battery and push (done at
  `68db184`), before G1(c) pre-registers; never inside a single-head campaign (a move inside one forces a re-record,
  because G1(a)'s revision binding refuses a replay at a different head).

### 2.10 Dispositions

- **A C1 diff outside the declared delta, a C2, C6, C8 or C9 red, or an R-L1 refusal:** the stage's commit is reverted,
  never amended to absorb the difference; the mover or the map is fixed and the stage rebuilt and re-checked. A
  baseline is re-recorded only by the re-point itself, never to absorb a rise (the plan's G8 row).
- **A planted red that does not fire:** the instrument is blind to that registry; G8 HALTS before the next stage, and
  nothing moves on a blind instrument.
- **A C3 red:** G8 halts; the envelope is never edited to make it green; the owner is told.
- **A C4 red:** the stage is reverted.
- **A battery red on the stacked head:** attributed per stage per commit; an unattributable red does not push.
- **After the push (F8):** no stage that moved a checkpoint class is reverted, and no pre-G8 build runs against a
  post-G8 lake, unless a reverse map lands first; the daemon stays stopped until the owner decides. The underlying P4b
  defect -- a refused restore erases the taught learner within one night (reproduced) -- is filed as its own arc
  (carried openings, 2026-09-25). The reverts in the first bullet are pre-push.
- **Another checkout (F4, disclosed):** a checkout that pulls G8 keeps its ignored `__pycache__/` directories and gets
  the namespace shadow; C6 is that checkout's detector. This machine has one checkout; the WSL path is the same
  directory.
- **Never done in G8:** a module retired, renamed, merged or re-governed; `brainstem/` or `tools/envelope_hashes.json`
  touched; the live lake opened or a stored checkpoint rewritten; a new `tools/*.py`; the daemon started. Moving an
  ungoverned module is out of scope (a promotion). No G8 result is a claim about the engine: G8 moves code; the engine's
  claims are G9's.

### 2.11 Answered by the design read (2026-09-25)

1. **The live checkpoint's contents** (was: a read-only census of the live `interaction_learner.json` before M5?). No
   live-lake census before M5: S0's translation maps by MODULE over every `LEGACY_*` pair, so it does not depend on
   which classes the live record holds, and `_G0C_LAZY_ATTRS` names only classes that stay. If the owner asks for one,
   it is taken frame by frame, read-only, on a COPY of the lake, never through `BinLog`, whose torn-tail repair can
   truncate a lake (the comment at `substrate/home.py::_T_CALIB`).
2. **S1's positive null** (was: the declared-delta form, or `move_gate` first so the next commit diffs to EMPTY?). S1
   splits in three (2.4): `move_gate` lands first, in S1a, with `LEGACY_*` and the label consumers' switch that keep its
   `gates` key out of the region censuses (the S1 re-cut, 2.13); S1b must diff to EMPTY; S1c carries F9.
3. **The riskiest alias paths first live at M11** (was: is S1's temporary-copy red enough, or should M11 be split?).
   M11 splits by whole source packages, `claustrum` alone first (M11a-M11e, 2.4; F4).
4. **C2(d) as a static scan** (was: does zero legacy strings suffice?). No: F3's file resolution, raw registry
   comparison and strict tripwire, and F16's self-check re-runs in M11e.
5. **Frozen runnable citations** (was: rewrite five runnable lines, or a line-keyed EXEMPT row per line?). F10: the
   EXEMPT row for t081's prose line 272 at M2; the quickstart's and `POLICIES_EVIDENCE.md`'s commands rewritten at
   their stages with dated notes, under the plan's dated amendment (2.5 item 7).
6. **The judgment calls** (75 rows, 69 of them moves, section 2.12). Unchanged: the design read reproduced no defect in
   them.
7. **`import_manifest.GOVERNED` after G8.** F15: each new package added in the stage that creates it, `lakehouse`
   kept, each emptied package dropped in the sub-stage that deletes it (2.5 item 6).

### 2.12 Every module's destination (414 rows, re-verified at `68db184`)

`tier`: C `REGIONS`, T `TORCH_TIER`, TR `TRAIN_TIER`, U ungoverned. `live`: L reached from the four `_LIVE_ROOTS`, E
only from the eight `_ENTRYPOINTS`, `-` neither (EXEMPT, reached only from an outside root such as the root
`learn_daemon.py` shim, or ungoverned). `judgment call`: J names the alternative home. The checkpoint-bearing rows are
21 (`LIFNeuron`, M5), 41 and 50 (M7), and the zoo rows 9, 10, 12, 14, 23, 25 and 26 (M5).

| # | current `src/snn_data_lake/...` | new | engine function | tier | live | judgment call |
|---|---|---|---|---|---|---|
| 1 | `brainstem/autonomous_loop.py` | `engine/autonomous_loop` | engine: composition+tick | C | L |  |
| 2 | `cortex/brain.py` | `engine/brain` | engine: composition+tick | C | L |  |
| 3 | `cortex/live_brain.py` | `engine/live_brain` | engine: composition+tick | C | L |  |
| 4 | `substrate/learn_daemon.py` | `engine/learn_daemon` | engine: composition+tick | C | - | J: brief lists daemon under infra/ |
| 5 | `substrate/report_channel.py` | `engine/report_channel` | engine: composition+tick | C | L | J: sensorimotor/ |
| 6 | `substrate/signal_ring.py` | `engine/signal_ring` | engine: composition+tick | C | L | J: infra/ |
| 7 | `substrate/wiring_manifest.py` | `engine/wiring_manifest` | engine: composition+tick | C | L | J: gates/ |
| 8 | `basal_ganglia/learning_progress.py` | `growth/learning_progress` | growth: grower | C | L | J: faculties/ (also owns the _T_GOAL store) |
| 9 | `cortex/adex_neuron.py` | `growth/adex_neuron` | growth: neuron model | C | L |  |
| 10 | `cortex/alif_neuron.py` | `growth/alif_neuron` | growth: neuron model | C | L |  |
| 11 | `cortex/arealization.py` | `growth/arealization` | growth: attach/place/label | C | L |  |
| 12 | `cortex/autapse_neuron.py` | `growth/autapse_neuron` | growth: neuron model | C | L |  |
| 13 | `cortex/cortical_prior.py` | `growth/cortical_prior` | growth: attach/place/label | C | L |  |
| 14 | `cortex/dendritic_neuron.py` | `growth/dendritic_neuron` | growth: neuron model | C | L |  |
| 15 | `cortex/growth_program.py` | `growth/growth_program` | growth: attach/place/label | C | L |  |
| 16 | `cortex/growth_stop.py` | `growth/growth_stop` | growth: grower | C | L |  |
| 17 | `cortex/h01_ingest.py` | `growth/h01_ingest` | growth: attach/place/label | C | - |  |
| 18 | `cortex/heterogeneous_microcircuit.py` | `growth/heterogeneous_microcircuit` | growth: neuron model | C | - | J: instruments/ (EXEMPT worked example) |
| 19 | `cortex/hodgkin_huxley.py` | `growth/hodgkin_huxley` | growth: neuron model | C | L |  |
| 20 | `cortex/izhikevich_neuron.py` | `growth/izhikevich_neuron` | growth: neuron model | C | L |  |
| 21 | `cortex/lif_neuron.py` | `growth/lif_neuron` | growth: neuron model | C | L |  |
| 22 | `cortex/neurogenesis_engine.py` | `growth/neurogenesis_engine` | growth: grower | T | L |  |
| 23 | `cortex/neuron_zoo.py` | `growth/neuron_zoo` | growth: neuron model | C | L |  |
| 24 | `cortex/placement.py` | `growth/placement` | growth: attach/place/label | C | L |  |
| 25 | `cortex/resonate_and_fire.py` | `growth/resonate_and_fire` | growth: neuron model | C | L |  |
| 26 | `cortex/srm_neuron.py` | `growth/srm_neuron` | growth: neuron model | C | L |  |
| 27 | `hippocampus/function_preserving_growth.py` | `growth/function_preserving_growth` | growth: grower | C | L |  |
| 28 | `hippocampus/growing_learner.py` | `growth/growing_learner` | growth: grower | C | L |  |
| 29 | `hippocampus/ng_snn.py` | `growth/ng_snn` | growth: grower | T | L |  |
| 30 | `substrate/basic_components.py` | `growth/basic_components` | growth: neuron model | C | - | J: engine/ |
| 31 | `substrate/brian2_backend.py` | `growth/brian2_backend` | growth: neuron model | C | E | J: reservoir/ (lens C) or engine/ |
| 32 | `substrate/growable.py` | `growth/growable` | growth: grower | C | L |  |
| 33 | `substrate/similarity_sized_growth.py` | `growth/similarity_sized_growth` | growth: grower | C | L |  |
| 34 | `cortex/cortical_microcircuit.py` | `reservoir/cortical_microcircuit` | reservoir | C | - | J: growth/ (column template for G4) |
| 35 | `cortex/neural_sampling.py` | `reservoir/neural_sampling` | reservoir | C | - | J: faculties/ |
| 36 | `cortex/recurrent_pool.py` | `reservoir/recurrent_pool` | reservoir | C | - |  |
| 37 | `plasticity/criticality_regulator.py` | `reservoir/criticality_regulator` | reservoir | C | L | J: stays plasticity/ |
| 38 | `substrate/kernel_state.py` | `reservoir/kernel_state` | reservoir | C | - | J: engine/ (the tick kernel) |
| 39 | `substrate/kernel_tick.py` | `reservoir/kernel_tick` | reservoir | C | - | J: engine/ (the tick kernel) |
| 40 | `substrate/spontaneous_activity.py` | `reservoir/spontaneous_activity` | reservoir | C | L | J: growth/ |
| 41 | `cortex/interaction_learning.py` | `readout/interaction_learning` | readout | C | L |  |
| 42 | `cortex/laminar_local_readout.py` | `readout/laminar_local_readout` | readout | C | L |  |
| 43 | `cortex/language_readout_slda.py` | `readout/language_readout_slda` | readout | C | L | J: faculties/ (language) |
| 44 | `cortex/predictive_coding.py` | `readout/predictive_coding` | readout | C | L | J: sensorimotor/ (a perception front end today) |
| 45 | `cortex/snn_classifier.py` | `readout/snn_classifier` | readout | C | L | J: sensorimotor/ (recognizer) |
| 46 | `cortex/snn_moe_classifier.py` | `readout/snn_moe_classifier` | readout | C | L | J: sensorimotor/ (recognizer) |
| 47 | `cortex/tempotron.py` | `readout/tempotron` | readout | C | L |  |
| 48 | `hippocampus/growable_readout.py` | `readout/growable_readout` | readout | C | - |  |
| 49 | `hippocampus/online_readout.py` | `readout/online_readout` | readout | T | L |  |
| 50 | `basal_ganglia/r_stdp.py` | `plasticity/r_stdp` | tuning rule | C | L |  |
| 51 | `basal_ganglia/valence_stdp.py` | `plasticity/valence_stdp` | tuning rule | C | L |  |
| 52 | `cortex/eprop_kernel.py` | `plasticity/eprop_kernel` | tuning rule | T | L | J: readout/ |
| 53 | `neuromodulation/naca.py` | `plasticity/naca` | tuning rule | C | L | J: faculties/ |
| 54 | `plasticity/bcm_plasticity.py` | = (stays) | tuning rule | C | L |  |
| 55 | `plasticity/benna_fusi.py` | = (stays) | tuning rule | C | L | J: sleep/ (lens C) |
| 56 | `plasticity/competitive_learning.py` | = (stays) | tuning rule | C | - |  |
| 57 | `plasticity/eligibility_algebra.py` | = (stays) | tuning rule | C | L |  |
| 58 | `plasticity/espp_rule.py` | = (stays) | tuning rule | C | L |  |
| 59 | `plasticity/homeostatic_plasticity_stack.py` | = (stays) | tuning rule | C | L |  |
| 60 | `plasticity/homeostatic_scaling.py` | = (stays) | tuning rule | C | L |  |
| 61 | `plasticity/inhibitory_stdp.py` | = (stays) | tuning rule | C | L |  |
| 62 | `plasticity/intrinsic_plasticity.py` | = (stays) | tuning rule | C | L |  |
| 63 | `plasticity/meta_plasticity.py` | = (stays) | tuning rule | C | L |  |
| 64 | `plasticity/oja_pca.py` | = (stays) | tuning rule | C | L |  |
| 65 | `plasticity/plasticity_rule_stack.py` | = (stays) | tuning rule | C | L |  |
| 66 | `plasticity/rule_composer.py` | = (stays) | tuning rule | C | L |  |
| 67 | `plasticity/short_term_plasticity.py` | = (stays) | tuning rule | C | L | J: reservoir/ (lens C) |
| 68 | `cortex/nightly_skills.py` | `sleep/nightly_skills` | sleep | C | L | J: faculties/ |
| 69 | `hippocampus/episode_consolidation.py` | `sleep/episode_consolidation` | sleep | C | L |  |
| 70 | `hippocampus/experience_replay.py` | `sleep/experience_replay` | sleep | C | L |  |
| 71 | `hippocampus/hippocampal_memory.py` | `sleep/hippocampal_memory` | sleep | C | - | J: faculties/ |
| 72 | `hippocampus/lake_consolidation.py` | `sleep/lake_consolidation` | sleep | C | L | J: substrate/ (lake) |
| 73 | `hippocampus/sleep_consolidation.py` | `sleep/sleep_consolidation` | sleep | C | L |  |
| 74 | `hippocampus/sleep_replay.py` | `sleep/sleep_replay` | sleep | C | L |  |
| 75 | `hypothalamus/circadian_clock.py` | `sleep/circadian_clock` | sleep | C | L |  |
| 76 | `plasticity/developmental_pruning.py` | `sleep/developmental_pruning` | sleep | C | - |  |
| 77 | `plasticity/lifelong_consolidation.py` | `sleep/lifelong_consolidation` | sleep | C | L |  |
| 78 | `plasticity/structural_rewiring.py` | `sleep/structural_rewiring` | sleep | C | L |  |
| 79 | `cortex/paradigm_b_engine.py` | `substrate/paradigm_b_engine` | lake: recall | C | E | J: instruments/ |
| 80 | `cortex/paradigm_b_matcher.py` | `substrate/paradigm_b_matcher` | lake: recall | C | L | J: instruments/ |
| 81 | `hippocampus/content_signature.py` | `substrate/content_signature` | lake: recall | C | L |  |
| 82 | `hippocampus/episodic_recall.py` | `substrate/episodic_recall` | lake: recall | C | L | J: faculties/ |
| 83 | `hippocampus/lake_recall.py` | `substrate/lake_recall` | lake: recall | C | L |  |
| 84 | `hippocampus/proximity_replay.py` | `substrate/proximity_replay` | lake: recall | C | L |  |
| 85 | `substrate/arm_view.py` | = (stays) | sensorimotor: arm I/O | C | L | J: sensorimotor/ once the owner approves the envelope import edit |
| 86 | `substrate/binlog.py` | = (stays) | lake: store | C | L |  |
| 87 | `substrate/chat_memory.py` | = (stays) | lake: store | C | L |  |
| 88 | `substrate/data_quality.py` | = (stays) | lake: store | C | - | J: gates/ |
| 89 | `substrate/frequency_moment.py` | = (stays) | lake: store | C | L |  |
| 90 | `substrate/home.py` | = (stays) | lake: store | C | L |  |
| 91 | `substrate/lake_bridge.py` | = (stays) | lake: store | C | L |  |
| 92 | `substrate/lake_engine.py` | = (stays) | lake: store | C | L |  |
| 93 | `substrate/lake_index.py` | = (stays) | lake: store | C | - |  |
| 94 | `substrate/lake_memory.py` | = (stays) | lake: store | C | - |  |
| 95 | `substrate/lake_migrate.py` | = (stays) | lake: store | C | L |  |
| 96 | `substrate/lake_segments.py` | = (stays) | lake: store | C | L |  |
| 97 | `substrate/lake_working_memory.py` | = (stays) | lake: store | C | L |  |
| 98 | `substrate/minhash.py` | = (stays) | lake: store | C | L |  |
| 99 | `substrate/morris_counter.py` | = (stays) | lake: store | C | L |  |
| 100 | `substrate/proximity_index.py` | = (stays) | lake: recall | C | L |  |
| 101 | `substrate/reservoir_sampling.py` | = (stays) | lake: store | C | L |  |
| 102 | `substrate/signal_calibrator.py` | = (stays) | lake: store | C | L | J: infra/ |
| 103 | `substrate/sliding_window_count.py` | = (stays) | lake: store | C | L |  |
| 104 | `substrate/spike_cardinality.py` | = (stays) | lake: store | C | L |  |
| 105 | `substrate/spike_stream_sketch.py` | = (stays) | lake: store | C | L |  |
| 106 | `substrate/spike_telemetry_hub.py` | = (stays) | lake: store | C | L | J: infra/ (telemetry) |
| 107 | `substrate/stream_sketch_panel.py` | = (stays) | lake: store | C | L |  |
| 108 | `substrate/streaming_hub.py` | = (stays) | lake: store | C | L |  |
| 109 | `substrate/streaming_quantile.py` | = (stays) | lake: store | C | L |  |
| 110 | `brainstem/arm_bridge.py` | `sensorimotor/arm_bridge` | sensorimotor: arm I/O | C | E |  |
| 111 | `brainstem/arm_hardware.py` | `sensorimotor/arm_hardware` | sensorimotor: arm I/O | C | L |  |
| 112 | `brainstem/closed_loop.py` | `sensorimotor/closed_loop` | sensorimotor: arm I/O | C | E |  |
| 113 | `brainstem/live_arm.py` | `sensorimotor/live_arm` | sensorimotor: arm I/O | C | E |  |
| 114 | `brainstem/signal_loop.py` | `sensorimotor/signal_loop` | sensorimotor: arm I/O | C | L |  |
| 115 | `cerebellum/wavefront_planner.py` | `sensorimotor/wavefront_planner` | sensorimotor: arm I/O | C | E |  |
| 116 | `cortex/corpus_stream.py` | `sensorimotor/corpus_stream` | sensorimotor: encode/ingest | C | L |  |
| 117 | `cortex/eprop_nmnist.py` | `sensorimotor/eprop_nmnist` | sensorimotor: encode/ingest | T | L |  |
| 118 | `cortex/eprop_shd.py` | `sensorimotor/eprop_shd` | sensorimotor: encode/ingest | T | L |  |
| 119 | `cortex/gesture_recognition.py` | `sensorimotor/gesture_recognition` | sensorimotor: encode/ingest | C | E |  |
| 120 | `cortex/lca_sparse_coding.py` | `sensorimotor/lca_sparse_coding` | sensorimotor: encode/ingest | C | L | J: faculties/ |
| 121 | `cortex/learned_matcher.py` | `sensorimotor/learned_matcher` | sensorimotor: encode/ingest | C | L | J: readout/ |
| 122 | `cortex/multimodal_encoder.py` | `sensorimotor/multimodal_encoder` | sensorimotor: encode/ingest | C | L |  |
| 123 | `cortex/nl_interface.py` | `sensorimotor/nl_interface` | sensorimotor: encode/ingest | C | L | J: faculties/ (language) |
| 124 | `cortex/pc_cortex_serve.py` | `sensorimotor/pc_cortex_serve` | sensorimotor: encode/ingest | C | L | J: readout/ |
| 125 | `cortex/population_encoding.py` | `sensorimotor/population_encoding` | sensorimotor: encode/ingest | C | L |  |
| 126 | `cortex/rank_order_coding.py` | `sensorimotor/rank_order_coding` | sensorimotor: encode/ingest | C | L |  |
| 127 | `cortex/recognizer_bank.py` | `sensorimotor/recognizer_bank` | sensorimotor: encode/ingest | C | L | J: readout/ |
| 128 | `cortex/rsnn_encoder.py` | `sensorimotor/rsnn_encoder` | sensorimotor: encode/ingest | C | - |  |
| 129 | `cortex/spike_preprocessing.py` | `sensorimotor/spike_preprocessing` | sensorimotor: encode/ingest | C | L |  |
| 130 | `cortex/spiking_lm_serve.py` | `sensorimotor/spiking_lm_serve` | sensorimotor: encode/ingest | C | E | J: faculties/ (language) |
| 131 | `cortex/trained_brain.py` | `sensorimotor/trained_brain` | sensorimotor: encode/ingest | C | L | J: infra/ |
| 132 | `cortex/world_kb_stream.py` | `sensorimotor/world_kb_stream` | sensorimotor: encode/ingest | C | L | J: substrate/ |
| 133 | `substrate/afferent_boundary.py` | `sensorimotor/afferent_boundary` | sensorimotor: boundary | C | L |  |
| 134 | `substrate/efferent_boundary.py` | `sensorimotor/efferent_boundary` | sensorimotor: boundary | C | L |  |
| 135 | `substrate/esp32_encoder_path.py` | `sensorimotor/esp32_encoder_path` | sensorimotor: encode/ingest | C | L |  |
| 136 | `substrate/hf_stream.py` | `sensorimotor/hf_stream` | sensorimotor: encode/ingest | C | - |  |
| 137 | `substrate/ingest_guard.py` | `sensorimotor/ingest_guard` | sensorimotor: encode/ingest | C | L |  |
| 138 | `substrate/make_sensor_dataset.py` | `sensorimotor/make_sensor_dataset` | sensorimotor: encode/ingest | C | - | J: instruments/ |
| 139 | `substrate/sensor_csv_ingest.py` | `sensorimotor/sensor_csv_ingest` | sensorimotor: encode/ingest | C | - |  |
| 140 | `substrate/web_ingest.py` | `sensorimotor/web_ingest` | sensorimotor: encode/ingest | C | L |  |
| 141 | `basal_ganglia/brain_native_selfbuild.py` | `infra/brain_native_selfbuild` | infra | C | L | J: faculties/ |
| 142 | `basal_ganglia/self_build_loop.py` | `infra/self_build_loop` | infra | C | L | J: faculties/ |
| 143 | `substrate/_mathutils.py` | `infra/_mathutils` | infra | C | L |  |
| 144 | `substrate/brain_bundle.py` | `infra/brain_bundle` | infra | C | L |  |
| 145 | `substrate/brain_geometry.py` | `infra/brain_geometry` | infra | C | E |  |
| 146 | `substrate/brain_telemetry.py` | `infra/brain_telemetry` | infra | C | L |  |
| 147 | `substrate/cockpit.py` | `infra/cockpit` | infra | C | E | J: sensorimotor/ (P4c: the sensorimotor surface) |
| 148 | `substrate/demo.py` | `infra/demo` | infra | C | E | J: sensorimotor/ |
| 149 | `substrate/device.py` | `infra/device` | infra | C | L |  |
| 150 | `substrate/drift_monitor.py` | `infra/drift_monitor` | infra | C | L | J: instruments/ |
| 151 | `substrate/nengo_bridge.py` | `infra/nengo_bridge` | infra | C | E | J: sensorimotor/ |
| 152 | `substrate/nir_export.py` | `infra/nir_export` | infra | C | - | J: sensorimotor/ |
| 153 | `substrate/region_wiring.py` | `infra/region_wiring` | infra | C | L | J: engine/ |
| 154 | `substrate/self_mod_harness.py` | `infra/self_mod_harness` | infra | C | L |  |
| 155 | `substrate/amdahl_gate.py` | `gates/amdahl_gate` | gate | C | - |  |
| 156 | `substrate/atlas_node_gate.py` | `gates/atlas_node_gate` | gate | C | - |  |
| 157 | `substrate/capability_bar.py` | `gates/capability_bar` | gate | C | - | J: instruments/ |
| 158 | `substrate/conclusion_protocol.py` | `gates/conclusion_protocol` | gate | C | L | J: instruments/ |
| 159 | `substrate/determinism_gate.py` | `gates/determinism_gate` | gate | C | - |  |
| 160 | `substrate/doc_budget_gate.py` | `gates/doc_budget_gate` | gate | C | - |  |
| 161 | `substrate/dormancy_gate.py` | `gates/dormancy_gate` | gate | C | L |  |
| 162 | `substrate/envelope_hash_gate.py` | `gates/envelope_hash_gate` | gate | C | L |  |
| 163 | `substrate/eprop_transport_gate.py` | `gates/eprop_transport_gate` | gate | C | - |  |
| 164 | `substrate/gradient_claim_gate.py` | `gates/gradient_claim_gate` | gate | C | L |  |
| 165 | `substrate/growth_coverage_gate.py` | `gates/growth_coverage_gate` | gate | C | - |  |
| 166 | `substrate/immortal_retention.py` | `gates/immortal_retention` | gate | C | - | J: instruments/ |
| 167 | `substrate/import_manifest.py` | `gates/import_manifest` | gate | C | - |  |
| 168 | `substrate/lake_rebuild_gate.py` | `gates/lake_rebuild_gate` | gate | C | - |  |
| 169 | `substrate/manifest_gate.py` | `gates/manifest_gate` | gate | C | L |  |
| 170 | `substrate/method_dormancy_gate.py` | `gates/method_dormancy_gate` | gate | C | L |  |
| 171 | `substrate/perf_baseline.py` | `gates/perf_baseline` | gate | C | - | J: instruments/ |
| 172 | `substrate/policy_manifest.py` | `gates/policy_manifest` | gate | C | - |  |
| 173 | `substrate/region_charge_gate.py` | `gates/region_charge_gate` | gate | C | - |  |
| 174 | `substrate/replay_gate.py` | `gates/replay_gate` | gate | C | - |  |
| 175 | `substrate/replication_guard.py` | `gates/replication_guard` | gate | C | - |  |
| 176 | `substrate/safety_envelope.py` | `gates/safety_envelope` | gate | C | L |  |
| 177 | `substrate/selfcheck_cache.py` | `gates/selfcheck_cache` | gate | C | - | J: infra/ |
| 178 | `cortex/attention_schema.py` | `instruments/attention_schema` | instrument | C | - |  |
| 179 | `cortex/deep_stdp_latency.py` | `instruments/deep_stdp_latency` | instrument | TR | - |  |
| 180 | `cortex/everyday_battery.py` | `instruments/everyday_battery` | instrument | C | - |  |
| 181 | `cortex/integration_measure.py` | `instruments/integration_measure` | instrument | C | L | J: faculties/ |
| 182 | `cortex/memory_diagnostic.py` | `instruments/memory_diagnostic` | instrument | C | - |  |
| 183 | `cortex/nef_compute.py` | `instruments/nef_compute` | instrument | C | E | J: faculties/ |
| 184 | `cortex/neuronal_avalanches.py` | `instruments/neuronal_avalanches` | instrument | C | L | J: reservoir/ |
| 185 | `cortex/pool_selection.py` | `instruments/pool_selection` | instrument | C | - |  |
| 186 | `cortex/prior_drift_harness.py` | `instruments/prior_drift_harness` | instrument | C | - |  |
| 187 | `cortex/report_order_probe.py` | `instruments/report_order_probe` | instrument | C | L |  |
| 188 | `hippocampus/developmental_loop_brain.py` | `instruments/developmental_loop_brain` | instrument | TR | - | J: faculties/ |
| 189 | `plasticity/deep_stdp.py` | `instruments/deep_stdp` | instrument | TR | - |  |
| 190 | `plasticity/snn_mnist_stdp.py` | `instruments/snn_mnist_stdp` | instrument | TR | - |  |
| 191 | `plasticity/snn_mnist_stdp_genn.py` | `instruments/snn_mnist_stdp_genn` | instrument | C | - |  |
| 192 | `substrate/energy.py` | `instruments/energy` | instrument | C | - |  |
| 193 | `substrate/episode_driver.py` | `instruments/episode_driver` | instrument | C | - | J: sensorimotor/ |
| 194 | `substrate/neuro_chip.py` | `instruments/neuro_chip` | instrument | C | - |  |
| 195 | `substrate/neuro_chip_stream.py` | `instruments/neuro_chip_stream` | instrument | C | - |  |
| 196 | `substrate/paired_contrast.py` | `instruments/paired_contrast` | instrument | C | L |  |
| 197 | `substrate/port_charge_probe.py` | `instruments/port_charge_probe` | instrument | C | L | J: gates/ |
| 198 | `substrate/world_red_probe.py` | `instruments/world_red_probe` | instrument | C | - |  |
| 199 | `thalamus/binding_instrument.py` | `instruments/binding_instrument` | instrument | C | - |  |
| 200 | `thalamus/ignition_instrument.py` | `instruments/ignition_instrument` | instrument | C | - |  |
| 201 | `amygdala/affective_appraisal.py` | `faculties/affective_appraisal` | faculty (interim) | C | L |  |
| 202 | `amygdala/fear_conditioning.py` | `faculties/fear_conditioning` | faculty (interim) | C | L |  |
| 203 | `basal_ganglia/action_gating.py` | `faculties/action_gating` | faculty (interim) | C | L |  |
| 204 | `basal_ganglia/brain_decision.py` | `faculties/brain_decision` | faculty (interim) | C | - |  |
| 205 | `basal_ganglia/evidence_accumulation.py` | `faculties/evidence_accumulation` | faculty (interim) | C | L |  |
| 206 | `basal_ganglia/latent_q.py` | `faculties/latent_q` | faculty (interim) | C | L |  |
| 207 | `basal_ganglia/mlp_q.py` | `faculties/mlp_q` | faculty (interim) | C | L |  |
| 208 | `brainstem/cpg_oscillator.py` | `faculties/cpg_oscillator` | faculty (interim) | C | L |  |
| 209 | `cerebellum/forward_model.py` | `faculties/forward_model` | faculty (interim) | C | L |  |
| 210 | `cerebellum/gain_adaptation.py` | `faculties/gain_adaptation` | faculty (interim) | C | L |  |
| 211 | `cerebellum/hjepa_serve.py` | `faculties/hjepa_serve` | faculty (interim) | C | L |  |
| 212 | `cerebellum/jepa_world_model.py` | `faculties/jepa_world_model` | faculty (interim) | C | L |  |
| 213 | `cerebellum/microcircuit.py` | `faculties/microcircuit` | faculty (interim) | C | - |  |
| 214 | `cerebellum/polychronization.py` | `faculties/polychronization` | faculty (interim) | C | L |  |
| 215 | `cerebellum/spiking_jepa.py` | `faculties/spiking_jepa` | faculty (interim) | C | L |  |
| 216 | `cerebellum/spiking_pid.py` | `faculties/spiking_pid` | faculty (interim) | C | E |  |
| 217 | `cerebellum/synfire_chain.py` | `faculties/synfire_chain` | faculty (interim) | C | L |  |
| 218 | `claustrum/claustrum.py` | `faculties/claustrum` | faculty (interim) | C | L |  |
| 219 | `cortex/access_consciousness.py` | `faculties/access_consciousness` | faculty (interim) | C | L |  |
| 220 | `cortex/affect_core.py` | `faculties/affect_core` | faculty (interim) | C | L |  |
| 221 | `cortex/agency_monitor.py` | `faculties/agency_monitor` | faculty (interim) | C | L |  |
| 222 | `cortex/autotelic_goals.py` | `faculties/autotelic_goals` | faculty (interim) | C | L |  |
| 223 | `cortex/competence_executive.py` | `faculties/competence_executive` | faculty (interim) | C | - | J: instruments/ |
| 224 | `cortex/counterfactual_sim.py` | `faculties/counterfactual_sim` | faculty (interim) | C | L |  |
| 225 | `cortex/crossmodal_binder.py` | `faculties/crossmodal_binder` | faculty (interim) | C | L | J: sensorimotor/ |
| 226 | `cortex/divisive_normalization.py` | `faculties/divisive_normalization` | faculty (interim) | C | L | J: sensorimotor/ |
| 227 | `cortex/executive_control.py` | `faculties/executive_control` | faculty (interim) | C | L |  |
| 228 | `cortex/generative_self_model.py` | `faculties/generative_self_model` | faculty (interim) | C | L |  |
| 229 | `cortex/is_snn_forward.py` | `faculties/is_snn_forward` | faculty (interim) | C | L | J: reservoir/ |
| 230 | `cortex/kb_ranking.py` | `faculties/kb_ranking` | faculty (interim) | C | L | J: readout/ |
| 231 | `cortex/language_acquisition.py` | `faculties/language_acquisition` | faculty (interim) | T | L |  |
| 232 | `cortex/language_self_build.py` | `faculties/language_self_build` | faculty (interim) | C | L |  |
| 233 | `cortex/learned_attention.py` | `faculties/learned_attention` | faculty (interim) | C | L |  |
| 234 | `cortex/learned_evc.py` | `faculties/learned_evc` | faculty (interim) | C | L |  |
| 235 | `cortex/metacognitive_monitor.py` | `faculties/metacognitive_monitor` | faculty (interim) | C | L |  |
| 236 | `cortex/percept_surprise.py` | `faculties/percept_surprise` | faculty (interim) | C | L | J: growth/ (its surprise feeds LP -> growth_stop) |
| 237 | `cortex/prefrontal_executive.py` | `faculties/prefrontal_executive` | faculty (interim) | C | L |  |
| 238 | `cortex/quality_diversity.py` | `faculties/quality_diversity` | faculty (interim) | C | L |  |
| 239 | `cortex/reasoning_curriculum.py` | `faculties/reasoning_curriculum` | faculty (interim) | C | L | J: growth/ (ReasoningLearner grows) |
| 240 | `cortex/recursive_tom.py` | `faculties/recursive_tom` | faculty (interim) | C | L |  |
| 241 | `cortex/rose_language.py` | `faculties/rose_language` | faculty (interim) | C | L |  |
| 242 | `cortex/second_order_access.py` | `faculties/second_order_access` | faculty (interim) | C | L |  |
| 243 | `cortex/self_improvement_suite.py` | `faculties/self_improvement_suite` | faculty (interim) | C | L | J: infra/ |
| 244 | `cortex/self_model.py` | `faculties/self_model` | faculty (interim) | C | L |  |
| 245 | `cortex/task_set.py` | `faculties/task_set` | faculty (interim) | C | L |  |
| 246 | `cortex/theory_of_mind.py` | `faculties/theory_of_mind` | faculty (interim) | C | L |  |
| 247 | `cortex/theta_gamma_code.py` | `faculties/theta_gamma_code` | faculty (interim) | C | L | J: sensorimotor/ |
| 248 | `cortex/time_cells.py` | `faculties/time_cells` | faculty (interim) | C | L |  |
| 249 | `cortex/vsa_reasoning.py` | `faculties/vsa_reasoning` | faculty (interim) | C | L |  |
| 250 | `cortex/vsa_structure_mapping.py` | `faculties/vsa_structure_mapping` | faculty (interim) | C | L |  |
| 251 | `cortex/wholistic_loop.py` | `faculties/wholistic_loop` | faculty (interim) | C | L | J: engine/ |
| 252 | `epithalamus/habenula.py` | `faculties/habenula` | faculty (interim) | C | L |  |
| 253 | `hippocampus/affective_memory.py` | `faculties/affective_memory` | faculty (interim) | C | L |  |
| 254 | `hippocampus/allocentric_map.py` | `faculties/allocentric_map` | faculty (interim) | C | L |  |
| 255 | `hippocampus/contrastive_episode_analysis.py` | `faculties/contrastive_episode_analysis` | faculty (interim) | C | L |  |
| 256 | `hippocampus/grid_cells.py` | `faculties/grid_cells` | faculty (interim) | C | L |  |
| 257 | `hippocampus/grid_place_map.py` | `faculties/grid_place_map` | faculty (interim) | C | L |  |
| 258 | `hippocampus/open_world_loop.py` | `faculties/open_world_loop` | faculty (interim) | C | L | J: sleep/ |
| 259 | `hippocampus/ring_attractor.py` | `faculties/ring_attractor` | faculty (interim) | C | E |  |
| 260 | `hippocampus/skill_builder.py` | `faculties/skill_builder` | faculty (interim) | C | L | J: sleep/ |
| 261 | `hippocampus/spiking_grid_can.py` | `faculties/spiking_grid_can` | faculty (interim) | C | L |  |
| 262 | `hippocampus/spiking_storage_prototype.py` | `faculties/spiking_storage_prototype` | faculty (interim) | C | L |  |
| 263 | `hypothalamus/homeostatic_drives.py` | `faculties/homeostatic_drives` | faculty (interim) | C | L |  |
| 264 | `hypothalamus/set_points.py` | `faculties/set_points` | faculty (interim) | C | L |  |
| 265 | `insula/interoceptive_integration.py` | `faculties/interoceptive_integration` | faculty (interim) | C | L |  |
| 266 | `medulla/interoceptive_arousal.py` | `faculties/interoceptive_arousal` | faculty (interim) | C | L |  |
| 267 | `medulla/vital_nuclei.py` | `faculties/vital_nuclei` | faculty (interim) | C | L |  |
| 268 | `midbrain/dopamine_nuclei.py` | `faculties/dopamine_nuclei` | faculty (interim) | C | L | J: plasticity/ (lens C: the R-STDP third factor) |
| 269 | `midbrain/multisensory_integration.py` | `faculties/multisensory_integration` | faculty (interim) | C | L |  |
| 270 | `midbrain/sc_deep_layers.py` | `faculties/sc_deep_layers` | faculty (interim) | C | - |  |
| 271 | `midbrain/superior_colliculus.py` | `faculties/superior_colliculus` | faculty (interim) | C | - |  |
| 272 | `neuromodulation/acetylcholine.py` | `faculties/acetylcholine` | faculty (interim) | C | L |  |
| 273 | `neuromodulation/affect_integrator.py` | `faculties/affect_integrator` | faculty (interim) | C | L |  |
| 274 | `neuromodulation/astrocyte_modulation.py` | `faculties/astrocyte_modulation` | faculty (interim) | C | L |  |
| 275 | `neuromodulation/cortisol.py` | `faculties/cortisol` | faculty (interim) | C | L |  |
| 276 | `neuromodulation/developmental_envelope.py` | `faculties/developmental_envelope` | faculty (interim) | C | L |  |
| 277 | `neuromodulation/drive_reduction.py` | `faculties/drive_reduction` | faculty (interim) | C | L |  |
| 278 | `neuromodulation/empowerment.py` | `faculties/empowerment` | faculty (interim) | C | L |  |
| 279 | `neuromodulation/generalizing_novelty.py` | `faculties/generalizing_novelty` | faculty (interim) | C | L |  |
| 280 | `neuromodulation/intrinsic_motivation.py` | `faculties/intrinsic_motivation` | faculty (interim) | C | L |  |
| 281 | `neuromodulation/locus_coeruleus.py` | `faculties/locus_coeruleus` | faculty (interim) | C | L |  |
| 282 | `neuromodulation/raphe_serotonin.py` | `faculties/raphe_serotonin` | faculty (interim) | C | L |  |
| 283 | `neuromodulation/spiking_curiosity.py` | `faculties/spiking_curiosity` | faculty (interim) | C | L |  |
| 284 | `plasticity/wm_capacity.py` | `faculties/wm_capacity` | faculty (interim) | C | L |  |
| 285 | `pons/mossy_relay.py` | `faculties/mossy_relay` | faculty (interim) | C | - |  |
| 286 | `substrate/region_coherence.py` | `faculties/region_coherence` | faculty (interim) | C | L |  |
| 287 | `thalamus/connectome_router.py` | `faculties/connectome_router` | faculty (interim) | C | L |  |
| 288 | `thalamus/global_workspace.py` | `faculties/global_workspace` | faculty (interim) | C | L | J: engine/ (the broadcast) |
| 289 | `thalamus/gwt_deep.py` | `faculties/gwt_deep` | faculty (interim) | C | - |  |
| 290 | `thalamus/interoceptive_relay.py` | `faculties/interoceptive_relay` | faculty (interim) | C | L |  |
| 291 | `thalamus/reentrant_workspace.py` | `faculties/reentrant_workspace` | faculty (interim) | C | L | J: engine/ (a dormancy live root) |
| 292 | `thalamus/reticular_gate.py` | `faculties/reticular_gate` | faculty (interim) | C | L |  |
| 293 | `brainstem/arm_config.py` | = (stays) | safety envelope | C | L |  |
| 294 | `brainstem/interpreter.py` | = (stays) | safety envelope | C | E |  |
| 295 | `brainstem/reflex.py` | = (stays) | safety envelope | C | L |  |
| 296 | `brainstem/serial_arm.py` | = (stays) | safety envelope | C | L |  |
| 297 | `brainstem/so101_arm.py` | = (stays) | safety envelope | C | L |  |
| 298 | `reasoning/_loop_common.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 299 | `reasoning/analogy_bench.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 300 | `reasoning/ar_spiking_lm_serve.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 301 | `reasoning/arc_agi/brain_player.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 302 | `reasoning/arc_agi/env.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 303 | `reasoning/arc_agi/play.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 304 | `reasoning/arc_agi/play_brain.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 305 | `reasoning/arc_agi/rule_inducer.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 306 | `reasoning/arc_agi/run_arc.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 307 | `reasoning/arc_bench.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 308 | `reasoning/arc_diagnostic.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 309 | `reasoning/arc_mint.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 310 | `reasoning/arc_residual.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 311 | `reasoning/arc_scene.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 312 | `reasoning/arc_solver.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 313 | `reasoning/autonomous_brain.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 314 | `reasoning/autonomous_brain_recovered.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 315 | `reasoning/autonomous_enrich.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 316 | `reasoning/bayesian_snn.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 317 | `reasoning/blimp_bench.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 318 | `reasoning/burst_multiplexing.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 319 | `reasoning/cellular_neural_network.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 320 | `reasoning/concept_reflection.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 321 | `reasoning/deep_stdp_2stage.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 322 | `reasoning/delay_learning.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 323 | `reasoning/dense_associative_memory.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 324 | `reasoning/derivation.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 325 | `reasoning/developmental_core.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 326 | `reasoning/developmental_loop.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 327 | `reasoning/developmental_loop_lake99.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 328 | `reasoning/developmental_loop_slda.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 329 | `reasoning/developmental_loop_wm.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 330 | `reasoning/dishbrain_learning.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 331 | `reasoning/eprop.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 332 | `reasoning/eprop_lm_alif_serve.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 333 | `reasoning/eprop_lm_serve.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 334 | `reasoning/event_quadtree_compression.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 335 | `reasoning/fast_features.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 336 | `reasoning/fly_hash.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 337 | `reasoning/free_energy_control.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 338 | `reasoning/gf.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 339 | `reasoning/godel_machine.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 340 | `reasoning/hjepa_arm_torch.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 341 | `reasoning/hjepa_torch.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 342 | `reasoning/hopfield_baseline.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 343 | `reasoning/impact_analysis.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 344 | `reasoning/index_compression.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 345 | `reasoning/jepa_scaffold_torch.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 346 | `reasoning/lake_reason.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 347 | `reasoning/loop_latency.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 348 | `reasoning/make_results_plot.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 349 | `reasoning/mcr_hdc.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 350 | `reasoning/model_builder.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 351 | `reasoning/nef_integrator.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 352 | `reasoning/nengo_gui_brain.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 353 | `reasoning/neural_heterogeneity.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 354 | `reasoning/ng_snn/data.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 355 | `reasoning/ng_snn/extend_3conv.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 356 | `reasoning/ng_snn/nmnist_pipeline.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 357 | `reasoning/ng_snn/nmnist_spatiotemporal.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 358 | `reasoning/ng_snn/nmnist_spatiotemporal_3conv.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 359 | `reasoning/ng_snn/push_mnist.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 360 | `reasoning/ng_snn/softhebb_mnist.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 361 | `reasoning/ng_snn/train_softhebb.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 362 | `reasoning/ng_snn/validate_uci.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 363 | `reasoning/nmnist_ingest.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 364 | `reasoning/paradigm_b_genn.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 365 | `reasoning/pc_abduction.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 366 | `reasoning/pc_active_inference.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 367 | `reasoning/pc_cortex_torch.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 368 | `reasoning/pragmatics.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 369 | `reasoning/random_boolean_network.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 370 | `reasoning/raven_bench.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 371 | `reasoning/real_rpm_bench.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 372 | `reasoning/reasoning_compose.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 373 | `reasoning/reasoning_cortex.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 374 | `reasoning/reservoir_eprop.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 375 | `reasoning/resonator_network.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 376 | `reasoning/resume_learning.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 377 | `reasoning/rpm_solver.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 378 | `reasoning/rsnn_delays_torch.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 379 | `reasoning/schema_induction.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 380 | `reasoning/self_improve.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 381 | `reasoning/semantic_memory.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 382 | `reasoning/sn_p_system.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 383 | `reasoning/snn_dreamer_6dof.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 384 | `reasoning/snn_mnist_stdp_fast.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 385 | `reasoning/snn_motor_control.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 386 | `reasoning/snn_ssm_shd.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 387 | `reasoning/snn_world_model_rl.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 388 | `reasoning/sorn.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 389 | `reasoning/sparse_distributed_memory.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 390 | `reasoning/spike_bloom.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 391 | `reasoning/spike_chat_lake.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 392 | `reasoning/spike_compression.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 393 | `reasoning/spike_kg.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 394 | `reasoning/spiking_som.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 395 | `reasoning/spiking_ssm.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 396 | `reasoning/stats_bounds.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 397 | `reasoning/stdp_wta_bayes.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 398 | `reasoning/temporal_coding_storage.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 399 | `reasoning/test_prototype.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 400 | `reasoning/train_of_thought.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 401 | `reasoning/triplet_stdp.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 402 | `reasoning/vsa_synthesize.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 403 | `reasoning/web_ingest_gate.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 404 | `reasoning/weight_perturbation.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 405 | `training/cli.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 406 | `training/mnist.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 407 | `training/mnist_genn.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 408 | `training/offload.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 409 | `training/spiking_lm.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 410 | `training/trainer.py` | = (stays) | research tier (ungoverned) | U | E |  |
| 411 | `lakehouse/delta_demo.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 412 | `lakehouse/medallion.py` | = (stays) | research tier (ungoverned) | U | - |  |
| 413 | `brain_atlas.py` | = (stays) | research tier (ungoverned) | U | L |  |
| 414 | `connectome.py` | = (stays) | research tier (ungoverned) | U | L |  |

### 2.13 The design read and its fold

The design read of this pre-registration filed 18 findings: 3 BLOCKER, 6 MAJOR, 9 MINOR. The orchestrator reproduced
every claim at `f416296` (whose `src/` equals `68db184`'s) and judged each fix separately from its claim; the
reproduction column is that record and is not re-run here. The fix column carries the orchestrator's adoption: adopted,
adopted NARROWED, or adopted in another form, with the reason.

| F | severity | the claim | the reproduction (the orchestrator, at `f416296`) | the fix adopted, and why | landed in |
|---|---|---|---|---|---|
| F1 | BLOCKER | `policy_manifest` derives a gate's `-m` command and its CI standing only for tokens under `substrate/`: after M1 a `gates/<gate>.py` token becomes a file-path command and is looked up among the `tools/` scripts, so the moved gates lose their manifest standing. | Code read: `policy_manifest.token_to_argv` and `cross_check` branch on the literal prefix `substrate/`; `policy_manifest --audit` exists (`run_audit`). | Adopted: both derive from the registry (a `<pkg>/<mod>.py` token naming a governed module becomes `-m snn_data_lake.<pkg>.<mod>`, `in_selfchecks = stem in mods`; `tools/` tokens unchanged). M1's planted red: a `gates/` token is in_ci and PASSes, and the pre-fix code reds on it in a temporary copy. The cheap half gains `policy_manifest --audit`. | 2.4 M1; 2.6 C1; 2.8 |
| F2 | BLOCKER | A legacy name cannot run under `python -m`: the alias loader has no `get_code`, and a `get_code` delegation alone still fails, because `runpy` sets `__file__` from `spec.origin`, which an alias spec leaves `None`; C2(c) could not pass as drafted. | Runtime: `python -m snn_data_lake.envelope_hash_gate --assert` (today's flat alias) -> `AttributeError: '_AliasLoader' object has no attribute 'get_code'`; with `get_code` delegation only -> `TypeError ... not 'NoneType'`; with `origin=<target origin>` and `has_location=True` -> "OK: all 5 envelope files match", exit 0. | Adopted NARROWED: only the `_LEGACY` specs carry the target's origin and `has_location=True`, with `get_code(fullname)` delegating to the target's loader under the target's name. The flat alias stays exactly as today, because it serves imports only, `doc_command_guard`'s premise that a flat alias cannot run under `-m` stays true, and nothing in G8 needs a flat `-m`. Planted red: a legacy `-m` of a gate that reads `__file__` at import exits exactly as the real path does, the three readings its positive and negative controls. | 2.4 S1b; 2.5 (the alias layer); 2.6 C2(c) |
| F3 | BLOCKER | An identity alias makes a stale name import, so a registry entry, runner name or packaging target left on a legacy name keeps working, and every check that resolves names by import (C1 after mapping, C8, C9) stays green on an un-re-pointed entry; a static scan (2.11 Q4) does not close that. | Runtime: `import snn_data_lake.r_stdp` succeeds as `snn_data_lake.basal_ganglia.r_stdp` while `src/snn_data_lake/r_stdp.py` does not exist. | Adopted, three parts, at S1b: (a) C8 and C9 resolve every name by FILE, an alias spec counting as red; (b) C1 compares every NAME-valued registry of 1.4 RAW against the post-stage set generated from the 2.12 table, only OUTCOME sets mapped through `LEGACY_*`; (c) the `SNN_ALIAS_STRICT=1` tripwire, never on the flat alias (the envelope's own self-check reaches `signal_loop` through one), ON for C1, C8, C9 and the stage's gate runs, OFF for C2 (b), (c) and (e), a planted legacy import its positive control. | 2.4 S1b; 2.5; 2.6 (the instrument, C1, C2, C8, C9); 2.8; 2.11 Q4 |
| F4 | MAJOR | An emptied package directory is not empty on disk: removing its tracked files leaves the ignored `__pycache__/`, a directory Python imports as a namespace package ahead of the appended alias finder, so a `_LEGACY_PKGS` defined as "no directory on disk" never fires and the synthetic package is shadowed. | 21 package directories hold an ignored `__pycache__/` (`.gitignore:2`); `claustrum/` holds `__pycache__`, `claustrum.py`, `__init__.py`. | Adopted, plus a disclosure: `_LEGACY_PKGS` = every `LEGACY_*` package key minus every live one (three tiers), from the tables; the mover deletes each emptied directory with its `__pycache__/` in the stage that empties it; C6 asserts no legacy package directory on the REAL checkout, and C2(b) runs there too; M11 splits by whole source packages, `claustrum` alone first. Disclosure: another checkout that pulls G8 keeps its `__pycache__/` and gets the shadow; C6 is its detector (one checkout here; the WSL path is the same directory). | 1.2; 2.4 M11a-M11e; 2.5 item 10 and the alias layer; 2.6 C2(b), C6; 2.10 |
| F5 | MAJOR | The label consumers use the region table for both the label and the import path, so a consumer switched to `LEGACY_*` (the draft's S1) builds a moved module's old path and loses its reachability. | Code read: `region_charge_gate.py:83`, `atlas_node_gate.py:128`, `brain_atlas.py:37-38` use the region table for both the label and the import path. | Adopted: labels and membership from `LEGACY_*`, module paths from `module_path()`; `brain_atlas._region_files` iterates `module_path()` files, never `dirname(pkg.__file__)`. Planted red: in a temporary copy one module moved with the joins in place leaves both region censuses unchanged, and the same move with the path taken from `LEGACY_*` reds. Landed at S1a, not S1b (the S1 re-cut, below). | 2.4 S1a; 2.6 (S1a's delta) |
| F6 | MAJOR | `tools/rule_census.json` (rows keyed `<pkg>/<mod>.py::<qualname>`) and `rule_composer._CENSUS_ADJUDICATED` (entries keyed by path) sit outside the draft's registry census, and B1 classes the JSON as frozen text: a move leaves them stale, and the regen then refuses on an adjudicated entry that matches no live row. | `tools/rule_census.json` holds 104 rows under `rules`; `_CENSUS_ADJUDICATED` holds 18 entries; the regen refuses ("REFUSED: ADJUDICATED entries matching zero live rows"). | Adopted: the census joins 1.4's registries and leaves the frozen class; in each stage that moves a rule-bearing module the pair map is applied to `_CENSUS_ADJUDICATED` first, then `--regen-census` runs in the same commit; C1 maps old keys to new and requires identical dispositions. The per-stage moving-row counts are re-derived by the mover into section 3, not written here, because they were not reproduced. | 1.4; 2.4 (the stage-size note); 2.5 item 6; 2.6 C1 |
| F7 | MAJOR | Six code sites build a package path from parts or test a package name, which neither the pair map nor C2(d)'s dotted scan sees: after their stage each quietly reads a directory that no longer holds its modules. | Code read: `tools/p6_gate.py:1733,1811` (`os.path.join(source_dir(root), "cortex", ...)`), `growth_coverage_gate.py:188` (`endswith("cortex")`), `atlas_node_gate.py:165` (`SRC / "thalamus"`), `:199-203` (`REGIONS.get("cerebellum")`, `SRC / "cerebellum"`). | Adopted: re-pointed at M10 (`p6_gate`, through `module_path()`), M5 (`growth_coverage_gate`, the package of `module_path("placement")`) and M11d, the sub-stage that empties `thalamus` and `cerebellum` (`atlas_node_gate`, its self-check corpus built from `module_path()`, asserting at least 1 file scanned). C2(d) is extended to split path parts, package tests and package-prefix strings, AST-scoped to code, these six sites its positive control. | 2.4 M5, M10, M11d; 2.5 item 3; 2.6 C2(d) |
| F8 | MAJOR | A refused restore erases the taught learner within one night: `Brain` swallows the refusal and `_flush_interaction_learner` then saves the fresh learner over the checkpoint, so reverting a checkpoint-moving stage after the push, or running a pre-G8 build against a post-G8 lake, destroys what the chat learner was taught. | Runtime, a throwaway lake, the real `save_state`/`consolidate`: taught flags `[True, False]` -> `[False]`; code read: `brain.py`'s restore swallows the refusal ("restore refused; learner stays fresh"), `_flush_interaction_learner` has no refusal guard. | Adopted as a disposition; the defect is NOT fixed in G8, because it is P4b's own arc: after the push no checkpoint-moving stage is reverted and no pre-G8 build runs against a post-G8 lake unless a reverse map lands first; the daemon stays stopped until the owner decides; the defect is filed in the carried openings (2026-09-25); R-L1's probe asserts no refusal before any flush can run. | 2.7; 2.10 |
| F9 | MAJOR | POLICIES P3 "ANATOMY IS DATA" says the region rules stay code hygiene "until specs/022's G7 and G8 re-key them", while the owner's decision (4) freezes them on `LEGACY_*` in G8 and leaves the re-keying to G7/G4. | `docs/POLICIES.md` P3, "ANATOMY IS DATA": "Both rules stay code hygiene until specs/022's G7 and G8 re-key them". | Adopted, as its own stage S1c: "...until specs/022's G7 and G4 re-key them; G8 freezes both on the pre-G8 `LEGACY_*` table (owner decision, 2026-09-25)" or an equivalent wording; `policy_manifest.json` regenerated; the word delta declared against `doc_budget_gate` before the gate runs, net zero if the headroom is short. The owner's decision in chat outranks POLICIES; this is POLICIES owing the fix. | 2.4 S1c; 2.6 (S1c's delta) |
| F10 | MINOR | The draft's one rule for frozen runnable text (2.5 item 7) treats three unlike cases alike and outruns the plan's "frozen evidence keeps its historical paths": `t081-report-order.md:272` is prose that citation_guard's `reproduc` heading rule catches, not a command; the specs/005 quickstart's lines are one `git diff --quiet` command that a stale path passes vacuously; and `POLICIES_EVIDENCE.md`'s two frozen `python -m` commands cannot ride the alias, because `doc_command_guard` resolves `-m` by file and rejects a flat alias. | `specs/005-language-organ/quickstart.md:75-77` is a `git diff --quiet HEAD -- <paths>` command (a stale path exits 0); `docs/POLICIES_EVIDENCE.md:410,476` are `python -m snn_data_lake.substrate.{import_manifest, method_dormancy_gate}`; `tools/doc_command_guard.py` resolves `-m` by FILE (`_module_file`) and rejects flat aliases. | Adopted with a plan amendment (the commit before this one): a line-keyed citation_guard EXEMPT row for t081's line 272 lands in M2's commit (not earlier: a stale exemption reds); the quickstart's line 75 (M11e) and line 76 (M3, M10) and `POLICIES_EVIDENCE.md`'s lines 410 and 476 (M1) are rewritten at their stages, the historical path in a dated note beside each. | 1.10; 2.4 M1, M2, M3, M10, M11e; 2.5 item 7; the plan's G8 row |
| F11 | MINOR | `so101_arm.py`, never opened for writing, cites moving modules in 8 comment lines; each dangles from its stage on and raises citation_guard's BROAD count in live files, which 2.6 forbade. | Code read: comment citations at `so101_arm.py` lines 241, 243, 602, 603, 604, 605, 606, 607. | Adopted: a named envelope carve-out in 2.6, the BROAD count in live files rising by exactly these lines at their stages (M1 604, 605; M5 243; M9 241; M11d 602, 603, 607; M11e 606) and by nothing else; the owner is told (a comment-only envelope edit is theirs to approve). | 1.8; 2.6 (the carve-out, C7) |
| F12 | MINOR | The templated battery line also lives in two copies outside the repo, which no in-repo rewrite and no gate reaches. | The memory store's `archive/t062-t063-2026-09-09/t066_battery_v2.sh` and `D:\AI-project\omnigent-sndl\SNDL-PROTOCOL.md` section 5 carry `-m snn_data_lake.substrate.$m`. | Adopted: M1's commit message and the closing record name both copies; the orchestrator updates them to `gates.$m` at the push (SYNC DUTY); the record states which form the battery ran. | 1.10; 2.4 M1; 2.8 |
| F13 | MINOR | C2(a) ("import all 414 modules") cannot pass as drafted: 13 fail today, none in a moving package. | Runtime, a scratch `NORD_HOME`, 0 files written: 414 attempted, 13 failed (lakehouse 2, reasoning 11). | Adopted: C2(a) asserts that the importable set is unchanged, this reading its baseline; the full list is re-recorded at S0's head in section 3. | 2.6 C2(a) |
| F14 | MINOR | R-L1 (b) compared walk DIGESTS "once class strings are mapped", but a digest cannot be mapped: the walk carries `"<module>.<qualname>"` class strings, so every move changes it. | Code read: `interaction_learning._g0c_walk` returns `{"class": "<module>.<qualname>", ...}`. | Adopted: the S1b walk JSON is stored beside H and its class strings are mapped old->new through `LEGACY_*` before comparison. | 2.7 |
| F15 | MINOR | `import_manifest` skips a listed name with no directory without a word, so a new package missing from `GOVERNED` goes unscanned silently, and a name listed without its directory scans nothing. | Code read: `import_manifest.py:173` skips a missing directory silently. | Adopted: `GOVERNED += <new package>` in the stage that creates it (S1a `gates`, M2 `instruments`, M3 `sensorimotor`, M4 `infra`, M5 `growth`, M6 `reservoir`, M7 `readout`, M8 `sleep`, M10 `engine`, M11a `faculties`), `lakehouse` kept (Q7). Ruled at the fold: each emptied package leaves `GOVERNED` in the M11 sub-stage that deletes its directory. | 1.4; 2.4; 2.5 item 6; 2.6 C1; 2.11 Q7 |
| F16 | MINOR | `reasoning_cortex`'s three `sys.modules` asserts on the `kb_ranking` key pass vacuously once the key is stale, and no static scan shows that `schema_induction`'s neutered red still fires. | Code read: `reasoning/reasoning_cortex.py:1157,1177,1185` assert `"snn_data_lake.cortex.kb_ranking" not in sys.modules`. | Adopted: in M11e, the sub-stage that moves `kb_ranking`, the three keys are derived with `module_path("kb_ranking")`, and `reasoning_cortex`'s and `schema_induction`'s self-checks run in its cheap half (their neutered reds must still fire). | 2.4 M11e; 2.5 item 3; 2.8; 2.11 Q4 |
| F17 | MINOR | `selfcheck_cache` finds the roster with `init_text.index("REGIONS = {")`, a substring that `LEGACY_REGIONS = {` also contains. | Code read: `selfcheck_cache.py:899-901` anchors with `init_text.index("REGIONS = {")`. | Adopted NARROWED to one fix: the anchor becomes the line-anchored regex `(?m)^REGIONS = \{` in the commit that introduces `LEGACY_REGIONS` (S1a after the re-cut); the table placement is unchanged (`LEGACY_*` after the live tables). | 2.4 S1a |
| F18 | MINOR | 2.5 kept the mover out of the tree with only its hash recorded, while 2.8 attributes a battery red by rebuilding a stage per commit, which needs the mover's source. | Section 2.5's last paragraph against 2.8's attribution. | Adopted: the mover's source is archived verbatim with the arc (the memory store's `archive/g8-2026-09-25/`), beside its hash in section 3; its P4c disposition is an out-of-tree operator artifact like the battery script of record, never committed to `tools/` (which would raise the p4c `tools` zone, baseline 30). | 2.5; 2.8 |

**The S1 re-cut (the orchestrator's ruling at the fold, option δ).** The design read answered Q2 with `move_gate`
landing first so the next commit diffs to EMPTY, and the fold's first cut put `LEGACY_*`, F5's consumer switch and
F17's anchor in that second commit (S1b). That order cannot hold: S1a's `gates` key would reach the label consumers
while they still read the live table. Reproduced in-process at `f416296` with the key planted, by the writer and again
by the orchestrator: the region-charge census went from 16 to 17 rows, a `gates: UNREACHABLE` row added (control: the
key removed, the census equal again). That is a change outside S1a's declared delta, and S1b's switch to the pre-G8
`LEGACY_*` would reverse it, so S1b could not diff to EMPTY; `brain_geometry` and `region_wiring` read the live table
the same way (code read). Adopted: S1a lands `move_gate` together with `LEGACY_*`, `package_of()`/`module_path()`, F5's
switch and joins and F17's anchor, under the draft's S1 delta word for word; S1b lands the alias layer (F2), F3's three
parts, F4's `_LEGACY_PKGS` and the R-L1 baseline, and must diff to EMPTY; S1c lands F9. Rejected: declaring the rows at
S1a and their reversal at S1b (it loses the positive null); excluding `gates` inside three consumers at S1a (code that
goes dead after S1b); copying `LEGACY_*` at S1a's head (a G8 package in the pre-G8 table, against the owner's decision
4). The pre-S1a census is taken by S1a's own `move_gate` over a `git archive` export of the S0b head (2.6).

**The fold's other rulings (the orchestrator, 2026-09-25).** `cortex` stays whole at 33 modules in M11e (a partial
package is never moved); emptied package names leave `GOVERNED` at the sub-stage that deletes their directory (F15);
F10's dated notes and F9's rewording carry a declared word delta against the measured headroom (`POLICIES_EVIDENCE.md`
28 words and `POLICIES.md` 33 at `f416296`), reworded to net zero if short; `olfactory/` goes at M11b with its `REGIONS`
key; 2.5 item 5 no longer relies on `runpy` failing on a stale runner name (F2 lets it run) and names C1's raw
comparison and the tripwire instead.

## 3. Results

Empty. The results begin at S0.
