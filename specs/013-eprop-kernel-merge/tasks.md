# Tasks: E-prop Kernel MERGE + Surrogate Perception Tier Retirement

**Feature**: `013-eprop-kernel-merge` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

**Base**: HEAD `47555a74`, tree clean. Gate env `.venv-lake\Scripts\python.exe`.

## Execution model — read before starting

**This lane is strictly sequential.** `[P]` appears on very few tasks, and that is not an
oversight:

1. Every commit rewrites line numbers the next commit cites (FR-020).
2. The gates are **stateful** — `import_manifest`'s floor, the p4c ratchet baseline, and the p6
   ratchets each carry a number that a prior commit moved.
3. Two commits editing `cortex/brain.py` (C1, C2, C3) cannot overlap at all.

`[P]` is used only inside a single commit's *measurement* block, where the probes are read-only and
touch disjoint files.

**Phase = commit.** Each phase begins with pre-commit measurements, ends with the gate batch, the
cite re-derivation, and exactly one commit.

**The gate batch** (referenced as *GATE BATCH* below) is the 10 commands in
[quickstart.md](quickstart.md) §0, plus the untracked-residue check. It MUST include
`ruff format --check`.

---

## Phase C0 — Evidence only (no behaviour change)

**Goal**: produce the two artifacts that *license* later commits. Nothing here changes runtime
behaviour, and that is deliberate: evidence that gates a deletion must not land in the commit that
performs it.

**Independent test**: the audit table and the pin files exist; the tree behaves identically
(`run_selfchecks.py` output unchanged).

- [X] T001 [P] Enumerate the 19 `surrogate_snn`-only definitions with the AST census in [quickstart.md](quickstart.md) C0-1; confirm the count is exactly 19 and record the list in `specs/013-eprop-kernel-merge/audit-surrogate-capability.md`
- [X] T002 [P] For each of the 17 routine definitions, write one disposition — covered / ported / dropped-with-reason — with its evidence, in `specs/013-eprop-kernel-merge/audit-surrogate-capability.md`
- [X] T003 Adjudicate `SpikingConvNet` in `specs/013-eprop-kernel-merge/audit-surrogate-capability.md`: it is a CONVOLUTIONAL spiking architecture where `src/snn_data_lake/cortex/eprop_nmnist.py` is recurrent fully-connected. Decide PORT or DROP-with-reason. **A DROP must state what capability the tree loses.** If it is judged load-bearing and unported, mark the deletion BLOCKED and stop the lane here
- [X] T004 Adjudicate `_heaviside_no_surrogate` in `specs/013-eprop-kernel-merge/audit-surrogate-capability.md`: it is a surrogate-OFF control, the shape P8 guard (2) values. Decide PORT or DROP-with-reason on the same terms as T003
- [X] T005 Adjudicate `ng_snn_data_file` and `_repo_root` in `specs/013-eprop-kernel-merge/audit-surrogate-capability.md`, noting their out-of-tree referents at `research/ng_snn/data.py:19`, `research/README.md:54`, `research/neuromorphic_torch/neuro_chip_gpu.py:52`
- [X] T006 Write the pin harness `tests/pin/capture.py` — fixed seed, synthetic input, no dataset and no `tonic`; serializes `forward.logits_sum`, `forward.latent_sum`, `eprop_grads.grads`, `eprop_grads.loss`, `bptt_grads.grads` for both consumers, recording torch version and device in the same dict
- [X] T007 Write the comparator `tests/pin/compare.py` using `torch.equal` (exact). **Do NOT use `allclose`** — a tolerance wide enough to survive a build difference is wide enough to hide the reshape error this exists to catch (research.md D-3)
- [X] T008 Capture the BEFORE side: run `capture.py` producing `tests/pin/nmnist-before.pt` and `tests/pin/shd-before.pt`
- [X] T009 Verify the harness is SENSITIVE before trusting it: perturb one input element, confirm `compare.py` reports a difference, then discard the perturbed capture. A comparator that has never reported a difference is not a control
- [X] T010 Run the GATE BATCH; confirm unchanged from HEAD baseline (p4c 205, manifest 8 targets, p6 5/5, self-checks 298, pytest 91.00%)
- [X] T011 Re-derive every `file:line` cite in the C0 artifacts after the last edit (FR-020), then commit C0

**Checkpoint**: if T003 or T004 returned BLOCKED, the lane stops and the spec's DELETE ruling is
revisited. That is a legitimate outcome, not a failure.

---

## Phase C1 — Retire the surrogate-BPTT perception tier (US1)

**Goal**: exactly one perception encoder survives.

**Independent test**: brain constructs, `perception_tier == "eprop-snn"` where the e-prop checkpoint
exists, zero live references remain, gates green.

### Pre-commit measurements

- [X] T012 [P] [US1] Sweep `tests/` for `models/surrogate_nmnist.pth` as a PATH STRING (prior sweeps covered class names only) per [quickstart.md](quickstart.md) C1
- [X] T013 [P] [US1] Re-verify the `models/surrogate_nmnist.pth` checkpoint holds only `['model','config']`, so `surro_acc` in `src/snn_data_lake/reasoning/pc_cortex_torch.py` is always `None` and the `:481` branch never fires — i.e. that block was never an A/B

### Deletions

- [X] T014 [US1] Delete `training/surrogate_snn.py`
- [X] T015 [US1] Delete the `else:` fallback branch at `src/snn_data_lake/cortex/brain.py:638-649` (the `SurrogateEncoder` import, the `_load_enc` call, and the `perception_tier = "torch-snn"` assignment)
- [X] T016 [US1] Delete the dead surrogate-ceiling block at `src/snn_data_lake/reasoning/pc_cortex_torch.py:427-446` and `:481-490`

### Co-edits — ALL in this same commit

- [X] T017 [US1] Lower the manifest floor at `src/snn_data_lake/substrate/import_manifest.py:287` from `>= 7` to `>= 6`. **This step is licensed by T014's target removal landing in the same commit** — a floor lowered without its removal is silencing a check (Standing Invariant 2)
- [X] T018 [US1] Update the stale comment at `src/snn_data_lake/substrate/import_manifest.py:109` and `:111-113` which names `training.surrogate_snn` as a live root target
- [X] T019 [US1] Fix the now-false telemetry comment at `src/snn_data_lake/cortex/brain.py:3301` (`"torch-snn" (hybrid surrogate-SNN encoder loaded) or "stdlib"`) and the tier description at `:563`
- [X] T020 [US1] ~~Remove `models/surrogate_nmnist.pth` from the checkpoint-probe tuple at `brain.py:641`~~ **SUBSUMED by T015** — that tuple sat *inside* the `else:` block T015 deleted, so it went with the block (and `models/nmnist_encoder.pt` with it, correctly: it was a second checkpoint name for the same retired `SurrogateEncoder`). Recorded rather than ticked as separate work.
- [X] T021 [US1] Update the `[train]` extra comment at `pyproject.toml:203`, which names `training/surrogate_snn.py` as the hybrid N-MNIST perception encoder
- [X] T022 [US1] Remove the four stale references in the governed twin `src/snn_data_lake/cortex/eprop_nmnist.py:6`, `:411`, `:424`, `:434`
- [X] T023 [US1] Update `training/eprop_shd.py:417` ("the sibling `training/surrogate_snn.py` already pins this") and `training/ar_spiking_lm.py:23` ("CONTRACT (mirrors `training/surrogate_snn.py`)")
- [X] T024 [P] [US1] Update `research/ng_snn/data.py:19`, `research/README.md:54`, `research/neuromorphic_torch/neuro_chip_gpu.py:52`
- [X] T025 [P] [US1] Update `docs/coverage/Perception.md:195`, `docs/coverage/SpatialTemporal.md:27`, `docs/COVERAGE_TO_100.md:68`, `docs/AUDIT_BACKLOG.md` (3 sites), `docs/P4C_BACKLOG.md:269`
- [X] T026 [US1] Regenerate `tools/p4c_ungoverned_baseline.json` (it lists `training/surrogate_snn.py` at `:242`); confirm total 205 → 204 and `training` zone 6 → 5

### Verification

- [X] T027 [US1] Confirm zero live references outside `CHANGELOG.md` (SC-006) with the sweep in [quickstart.md](quickstart.md) C1
- [X] T028 [US1] Confirm `python -m snn_data_lake.substrate.import_manifest` reports **7** targets and passes
- [X] T029 [US1] Run the GATE BATCH
- [X] T030 [US1] Re-derive every `file:line` cite AFTER the last edit above (FR-020); read the self-check count from output rather than inferring it; then commit C1

**Commit body MUST NOT claim**: that e-prop is more accurate (p=1.0000 is a tie); that the surrogate
path was unreachable on a fresh clone (neither checkpoint ships); or that this costs nothing (a
machine holding only `surrogate_nmnist.pth` loses its encoder entirely).

---

## Phase C2 — Split the perception try block (US2)

**Goal**: an audio import failure can no longer roll back a completed vision tier, and a mis-typed
import in C3 fails loudly.

**Independent test**: block `training.eprop_shd` at import; the vision tier survives.

### Pre-commit measurements

- [X] T031 [US2] Record the CURRENT recorder observable at **>= 12 ticks** (never 0 — a 0-tick comparison is vacuous; at 12 ticks a rollback collapses `/cognition/attention_real` from a 3-key dict to a scalar, 237 → 235 leaf paths)
- [X] T032 [US2] Run `tools\p6_gate.py --assert` and record the 5 ratchet values, so the post-split delta is measured rather than inferred

### Implementation

- [X] T033 [US2] Split the single `try` at `src/snn_data_lake/cortex/brain.py:597` (closed by the blanket `except Exception:` at `:679`) into an independent vision `try` covering `:630-663` and an audio `try` covering `:664-678`, each with its own handler and its own tier rollback

### Verification

- [X] T034 [US2] With the audio encoder blocked at import via a `meta_path` blocker, confirm the vision tier survives and `perception_tier != "stdlib"` (SC-008)
- [X] T035 [US2] Re-measure the recorder observable at >= 12 ticks; confirm the leaf-path count is unchanged from T031 (SC-007)
- [X] T036 [US2] Re-run `tools\p6_gate.py --assert`; confirm zero delta against T032
- [X] T037 [US2] Run the GATE BATCH
- [X] T038 [US2] Re-derive cites after the last edit (FR-020); commit C2

**Commit body MUST NOT claim** this fixes the off-root/CWD shape — off-root, the relative checkpoint
paths at `brain.py:630`/`:664` fail their `os.path.exists` guards and the imports are never
attempted at all. That is NG-004.

---

## Phase C3 — Promote `eprop_shd` into the governed tree (US3)

**Goal**: the audio trainer is governed. **Highest gate risk in the lane — four gates move at once.**

**Independent test**: all gates green, auditory tier still loads from `models/eprop_shd.pt`.

### Pre-commit measurements

- [ ] T039 [US3] Sweep `training/eprop_shd.py` for depth-counted path logic a move would silently retarget (`training/surrogate_snn.py:337-366` had exactly that shape; eprop_shd's post-move path behaviour was measured by nobody)
- [ ] T040 [US3] Confirm the self-check gate at `training/eprop_shd.py:920` (`smoke = args.smoke or len(sys.argv) == 1`) will still fire under `-m` invocation

### The move — ALL SEVEN co-edits in this same commit

- [ ] T041 [US3] `git mv training/eprop_shd.py src/snn_data_lake/cortex/eprop_shd.py`
- [ ] T042 [US3] Register `"eprop_shd"` in `TORCH_TIER["cortex"]` in `src/snn_data_lake/__init__.py` — without this, `cortex` becomes a new p4c zone with implicit allowance 0 and goes RED
- [ ] T043 [US3] Register `"cortex.eprop_shd"` in `run_selfchecks.py` TORCH_GATED, near the existing `cortex.eprop_nmnist` entry
- [ ] T044 [US3] Rewrite `src/snn_data_lake/cortex/brain.py:668` to `from snn_data_lake.cortex.eprop_shd import EpropAudioEncoder` — the `_PKG` guard at `substrate/dormancy_gate.py:353` gives a bare-root `training.*` import no graph edge, so the dormancy gate goes RED without this
- [ ] T045 [US3] Rewrite `docs/TRAINING.md:23` from `.venv-lake/Scripts/python training/eprop_shd.py` to the dotted `-m` form matching its sibling at `:22` — it is a STRICT `citation_guard` line and goes RED on the move
- [ ] T046 [US3] Set `UNGOVERNED_ROOT = ()` at `src/snn_data_lake/substrate/import_manifest.py:126` and lower the floor at `:287` from `>= 6` to `>= 5`; both are required or `:306` fires on `unmatched=['training']`
- [ ] T047 [US3] Delete the stale comment at `src/snn_data_lake/substrate/import_manifest.py:111-113` naming the two root targets
- [ ] T048 [US3] Regenerate `tools/p4c_ungoverned_baseline.json`; confirm total 204 → 203 and `training` zone 5 → 4

### Verification

- [ ] T049 [US3] Confirm `.venv-lake\Scripts\python.exe -m snn_data_lake.cortex.eprop_shd` reaches its self-check and exits 0
- [ ] T050 [US3] Confirm the dormancy gate sees the module as reachable
- [ ] T051 [US3] Confirm `python -m snn_data_lake.substrate.import_manifest` reports **6** targets and passes
- [ ] T052 [US3] Run the GATE BATCH
- [ ] T053 [US3] Re-derive cites after the last edit (FR-020); commit C3

**Commit body MUST carry** the `UNGOVERNED_ROOT = ()` caveat: it records "no AST-visible root-shape
imports", not "no governed → `training/` edges". A surviving invisible one:
`src/snn_data_lake/reasoning/eprop_lm_serve.py:261-264` does `sys.path.insert` then
`import eprop_lm as trainer`, a bare-name import `_targets()` cannot see.

**Commit body MUST NOT describe this as "the merge"** — a move relocates duplication into the
governed tree; it does not retire it.

---

## Phase C4 — Extract the one kernel (US4) — the commit that closes the lane

**Goal**: exactly one definition of each shared symbol tree-wide.

**Independent test**: numeric outputs bit-identical before and after.

### Pre-commit measurements

- [ ] T054 [US4] Re-run the `flatten_frames` subsumption A/B: prove `eprop_shd`'s body reproduces `eprop_nmnist`'s outputs bit-exactly on every shape `eprop_nmnist` receives, and that `eprop_nmnist`'s body raises on 3-D audio input. **If subsumption fails on any shape, STOP — the collapse is wrong and the kernel design changes**
- [ ] T055 [US4] Re-capture the BEFORE pin at the current HEAD (C3 moved the file; the C0 capture predates the move)

### Implementation

- [ ] T056 [US4] Create `src/snn_data_lake/cortex/eprop_kernel.py` per [contracts/eprop_kernel.md](contracts/eprop_kernel.md): the 5 byte-identical module-scope definitions, `EpropLIFBase` (`psi`, `offdiag_mask`, `forward`, one concrete `flatten_frames`), `EpropEncoderBase` (`__init__`), the 5 same-valued LIF constants, and `EPROP_BPTT_COSINE_FLOOR = 0.5`
- [ ] T057 [US4] Write the merged `forward` docstring describing BOTH input shapes (FR-008) — do not silently adopt one modality's wording
- [ ] T058 [US4] Write the kernel `__main__` self-check: build a minimal concrete subclass INLINE (importing a consumer is a cycle, FR-005), assert `_cosine(...) > EPROP_BPTT_COSINE_FLOOR`, print the actual cosine, ASCII-only, `random.Random(seed)` per instance
- [ ] T059 [US4] Delete the 9 duplicated definitions from `src/snn_data_lake/cortex/eprop_nmnist.py` and import them from the kernel
- [ ] T060 [US4] Delete the 9 duplicated definitions from `src/snn_data_lake/cortex/eprop_shd.py` and import them from the kernel
- [ ] T061 [US4] Replace the hand-written bare `0.5` at `src/snn_data_lake/cortex/eprop_shd.py:830` with the imported `EPROP_BPTT_COSINE_FLOOR` (FR-003)
- [ ] T062 [US4] Register `"eprop_kernel"` in `TORCH_TIER["cortex"]` (`src/snn_data_lake/__init__.py`) and `"cortex.eprop_kernel"` in `run_selfchecks.py` TORCH_GATED
- [ ] T063 [US4] **DELETE** the P2 debt note at `src/snn_data_lake/__init__.py:415-417` — delete, not amend (FR-015); a surviving debt marker for discharged debt is a false claim

### Verification — the pin is the acceptance step, NOT the cosine assert

- [ ] T064 [US4] Run `tests/pin/compare.py`; every tensor must be BIT-IDENTICAL via `torch.equal`. Any difference blocks the lane (SC-003). **Do not substitute the cosine assert** — `bptt_grads` computes its reference through the same `forward`, so a `flatten_frames` defect moves both sides together and the cosine stays high (Standing Invariant 5)
- [ ] T065 [US4] Demonstrate the kernel self-check's reachable RED (FR-019, SC-010): negate the broadcast error term feeding the eligibility product, confirm the cosine falls below the floor, record the observed value. **If nothing reddens it, REMOVE the assert rather than softening it**, and report that outcome
- [ ] T066 [US4] Demonstrate P3 charge: with `snn_data_lake.cortex.eprop_kernel` neutered in `sys.modules`, confirm BOTH consumers fail to import (per [contracts/eprop_kernel.md](contracts/eprop_kernel.md))
- [ ] T067 [US4] Re-run the AST survivor census (SC-001): every symbol in [data-model.md](data-model.md) §1 has exactly ONE definition site tree-wide
- [ ] T068 [US4] Confirm the geometry constants remain at TWO sites with DISTINCT values — `NUM_CLASSES` 10/20, `DEFAULT_T` 15/100, `DEFAULT_HIDDEN` 64/128. Merging them is a defect, not an improvement
- [ ] T069 [US4] Run the GATE BATCH
- [ ] T070 [US4] Re-derive cites after the last edit (FR-020); commit C4

---

## Phase C5 — Polish & cross-cutting

- [ ] T071 Verify the SAFETY ENVELOPE is untouched: `git diff 47555a74..HEAD --name-only -- src/snn_data_lake/brainstem/` returns **zero lines**. Any file there changing is a plan error, not an acceptable side effect
- [ ] T072 Append the lane's CHANGELOG entry, quoting only figures read from output — never written by increment. A prior arc shipped four commit messages whose self-check counts were incremented rather than read
- [ ] T073 Update `docs/P4C_BACKLOG.md`: the blocked tier loses `eprop_shd` and `surrogate_snn`; 3 blocked entries remain (`schema_induction`, `snn_moe_classifier`, `spike_chat_lake`)
- [ ] T074 Run `graphify update .` to keep the knowledge graph current after the moves
- [ ] T075 Final full GATE BATCH on the complete lane; record the closing figures (expected p4c 203, manifest 6 targets, `UNGOVERNED_ROOT = ()`)

---

## Dependencies

```
C0 (evidence) ──> C1 (retire) ──> C2 (split) ──> C3 (move) ──> C4 (kernel) ──> C5 (polish)
     │                                                             ▲
     └── T006-T009 pin harness ───────────────────────────────────┘  (re-captured at T055)
```

**Hard orderings and why** (do not resequence without re-measuring):

- **C0 before C1** — evidence that gates a deletion must not ride in the commit that performs it.
- **C2 before C3** — the blanket `except Exception:` at `brain.py:679` would swallow a mis-typed
  import rewrite in C3 *while `import_manifest` went green*, because the target simply vanished.
- **C1 and C3 separate** — removing both `training.*` manifest targets in one commit fires TWO
  asserts (`:287` at `len 6 < 7`, and `:306` on `unmatched=['training']`).
- **C3 before C4** — the kernel's consumers must both be in `cortex/` before they can share it.
- **T055 after C3** — the C0 capture predates the file move.

## Parallel opportunities

Deliberately few. Genuinely parallel-safe:

- **T001, T002** — read-only AST census over one file.
- **T012, T013** — read-only probes over disjoint files.
- **T024, T025** — documentation-only edits in disjoint trees, no gate is stateful on them.

Everything else is sequential. Three phases edit `cortex/brain.py`; two edit
`substrate/import_manifest.py`; every phase moves a stateful gate number.

## MVP scope

**C0 + C1 is the minimum shippable increment.** It discharges half the P2 debt (one perception
mechanism instead of two), removes a module, lowers p4c by one, and leaves the tree gate-green and
strictly better. C2–C4 complete the merge but C1 stands alone.

**C0 alone ships nothing** and is not an increment — it is the evidence that licenses C1, and it may
BLOCK it (T003/T004).

## Task count

| phase | tasks |
|---|---|
| C0 evidence | 11 (T001–T011) |
| C1 retire (US1) | 19 (T012–T030) |
| C2 split (US2) | 8 (T031–T038) |
| C3 move (US3) | 15 (T039–T053) |
| C4 kernel (US4) | 17 (T054–T070) |
| C5 polish | 5 (T071–T075) |
| **total** | **75** |
