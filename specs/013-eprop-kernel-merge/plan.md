# Implementation Plan: E-prop Kernel MERGE + Surrogate Perception Tier Retirement

**Branch**: `013-eprop-kernel-merge` | **Date**: 2026-08-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/013-eprop-kernel-merge/spec.md`

## Summary

Two copies of one e-prop engine exist (`cortex/eprop_nmnist.py` governed, `training/eprop_shd.py`
ungoverned), holding **9 byte-identical definitions**; and two perception encoders run behind a
checkpoint-existence flag at `cortex/brain.py:631`/`:638-649`. P2 requires exactly one mechanism per
job. This lane extracts `cortex/eprop_kernel.py` as the one e-prop mechanism, promotes the audio
trainer into the governed tree, deletes the surrogate-BPTT tier, and splits a blanket
`except Exception:` that would otherwise hide the work's own failures.

**Technical approach**: five commits, each independently gate-green, ordered so that the commit
which makes failures *visible* (the try-split) lands before the commit with the highest gate risk
(the move). A prerequisite commit (Commit 0) produces evidence only — the P2 unique-capability
dispositions and the bit-exact numeric pin — because both are required *before* the deletions they
gate, and neither changes behaviour.

**The governing design constraint is R-001, not any gate.** The kernel is safe wherever it rests on
byte-identity. It rests on judgment in exactly one place (`flatten_frames`), and the existing cosine
assert structurally cannot detect an error there, because `bptt_grads` computes its reference
through the same `forward` — both sides move together. Standing Invariant 5 names this shape
directly: *a green gate that structurally cannot see the thing it checks is worse than no gate.*
The bit-exact pin is therefore the verification mechanism for Commit 4, and the cosine assert is
explicitly NOT.

## Technical Context

**Language/Version**: Python 3.14.6 free-threading (`requires-python = ">=3.14"`)

**Primary Dependencies**: `torch` 2.11.0+cu128, `numpy`, `mujoco==3.11.0` (unconditional core).
`snntorch` present in the gate env. **`tonic` ABSENT** — anything needing real N-MNIST/SHD ingest
beyond the committed packs (`models/nmnist_pack.npz`, `models/shd_pack.npz`) is out of reach without
an install. No new dependency is introduced by this lane; per the constitution's note, that fact is
recorded as a fact and is *not* offered as evidence of anything.

**Storage**: N/A for this lane. `brain_lake.bin` is untouched; no P4b surface is involved. Model
checkpoints are gitignored artifacts (`.gitignore:143` `models/*`), not stores.

**Testing**: `pytest` (>=70% line coverage; currently 91.00%) + `run_selfchecks.py` per-module
`__main__` self-checks. The kernel joins the TORCH_GATED set.

**Target Platform**: Windows local gate + Linux/Windows CI matrix. **CI has no GPU and installs CPU
torch**, so the local and CI torch builds differ by construction.

**Project Type**: single Python package (`src/snn_data_lake/`) plus ungoverned root trees being
drained (`training/`, `reasoning/`).

**Performance Goals**: none for this lane. The self-check cost added by the kernel and by the
promoted `eprop_shd` must stay at parity (~1.6 s marginal, measured **[F]**); the lane must not
grow the in-process pytest wall time materially.

**Constraints**:
- One change per commit, full gate green at *every* commit.
- The suite must stay runnable with **no GPU**.
- **The FR-017 bit-exact pin runs LOCALLY ONLY** and is not committed as a CI assert — see
  *Decision D-3* in [research.md](research.md). Bit-exact float equality across two different torch
  builds (local cu128 vs CI CPU) is not a contract this project can hold, and asserting it in CI
  would manufacture a red that says nothing about the merge.

**Scale/Scope**: 2 source files merged into 3 (kernel + 2 consumers); 1 file deleted; 1 file moved;
~20 co-edit sites; 5 commits; 4 asserting gates move.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design — see bottom.*

### Core Principles

| principle | verdict | basis |
|---|---|---|
| **P0** autonomous self-improvement | **NEUTRAL** | This lane neither adds nor removes a self-improvement avenue. It does not touch `self_build_loop`, `self_mod_harness`, or `godel_machine`. Recorded so the lane is not credited with P0 progress it does not make. |
| **P1** brain-native / no-hardcode | **ADVANCES** | `eprop_shd.py:830`'s bare `0.5` becomes the imported `EPROP_BPTT_COSINE_FLOOR` (FR-003). One magic constant retired. The five LIF constants are consolidated to one definition each, not re-hardcoded. |
| **P2** adopt-or-merge, non-additive | **THE POINT OF THE LANE** | Both routes are used and both are legitimate: MERGE for the e-prop engine (build the one kernel, retire both copies of the shared definitions), ADOPT for the perception tier (e-prop becomes THE path, the surrogate is deleted). Survivor count is SC-001 and it is checked, not asserted. |
| **P3** no dormancy | **RISK, MITIGATED** | The kernel must not land dormant. Its consumers are two governed modules that import it directly, so it is reachable by construction; but *reachable is not charged*, and the honesty guard forbids faking a consumer. Charge is real here — the kernel's definitions are the execution path, not a side channel — and Commit 4's bit-exact pin doubles as the charge evidence (neuter the kernel import and both consumers fail to import at all). |
| **P4 / P4b** one brain / one lake | **NEUTRAL** | No `Brain` construction path is added. `brain.py` edits are import-site and try-structure only. No durable store is touched. |
| **P5** GPU | **NEUTRAL** | No compute-shape change. The lane must not break the no-GPU path; CI covers that. |
| **P6** faculty-to-faculty | **PROTECTED, NOT ADVANCED** | FR-004 puts the kernel in `cortex/` specifically because a cortex→other-region import drives `undeclared_import_pairs` 11→12 and reddens the ratchet **[F]**. This is a *do no harm* choice; the lane claims no P6 progress. |
| **P7** dependencies | **N/A** | No dependency added or removed. Per the constitution, this is stated and nothing is inferred from it. |
| **P8** invent the method | **BINDS on the new self-check** | The kernel's `__main__` is a new criterion, so it owes guard (1) a reachable RED (FR-019), guard (2) a control that is not itself, and guard (4) disclosure. See *Standing Invariant 5* below — these are the same obligation seen from two sides. |

### Standing Invariants

| # | invariant | verdict |
|---|---|---|
| 1 | count a gate's output before a rule leans on it | **MET** — every gate figure this plan sequences on was run at HEAD, not read: p4c 205 (zones enumerated), manifest 8 targets / floor 7, p6 5/5 ratchets, citation_guard OK. |
| 2 | **never silence a check to make it pass** | **APPLIES DIRECTLY, AND THE PLAN MUST NOT LOOK LIKE IT.** This lane lowers `import_manifest.py:287` twice (7→6→5) and empties `UNGOVERNED_ROOT`. That is the *opposite* of silencing only if each step follows a genuine removal of a target. Discipline: each floor step lands in the same commit as the removal that justifies it, never ahead of it, and never batched. A floor lowered without its removal is silencing and is a lane failure. `:302-304` pre-authorises the empty tuple, so `UNGOVERNED_ROOT = ()` is a sanctioned terminal state, not a dodge. |
| 3 | **the negative verdict is the dangerous one — it retires a working faculty** | **APPLIES DIRECTLY. This is the sharpest constitutional constraint on the lane.** The lane retires a *working* 92.58% encoder on an A/B that returned **p = 1.0000 at n=256** — an underpowered probe. The invariant says a probe that cannot distinguish outcomes must report **UNKNOWN**, not a confident negative. **Resolution**: the deletion is NOT justified on the accuracy verdict, which is recorded as UNKNOWN/tie and explicitly forbidden from the commit message (spec §"three claims"). It is justified on three grounds that are *not* underpowered: (a) P2 forbids two live mechanisms for one job regardless of which is better; (b) Standing Invariant 8 — e-prop, not backprop — makes the surrogate-BPTT path the one that loses a tie by rule; (c) measured speed 7.8–10.6× and 0.56× params. FR-018 then guarantees nothing unique is lost. |
| 4 | absence of a warning is not evidence of absence | **MET** — the plan schedules positive sweeps (path-string sweep, depth-counted-path sweep) rather than inferring safety from green gates. |
| 5 | **verify the verifier** | **APPLIES DIRECTLY — this is R-001.** The cosine assert structurally cannot see a `flatten_frames` error. The plan does not lean on it. FR-017's pin is the verifier, and FR-019 requires the *new* criterion demonstrate its own reachable red. |
| 6 | runtime evidence outranks static reading | **MET** — every [F] claim gating a commit is re-run as a scheduled pre-commit task, not trusted. |
| 7 | **safety envelope untouched** | **MET, and checkable**: this lane touches zero files under `brainstem/`. Any task proposing one is a plan error. |
| 8 | **e-prop, not backprop** | **ADVANCES** — the lane deletes the tree's surrogate-BPTT *perception* path. Care: it must NOT delete a BPTT *reference gradient*, which is what the invariant depends on. Measured: `surrogate_snn.py` contains zero `bptt_grads`/`_cosine`/`_apply_grads`/`_SpikeFn`; both e-prop modules keep their own `bptt_grads`. The reference survives in the kernel, where both consumers inherit it. |

### Disclosure (P8 guard 4 — unprompted)

**The constitution's own Governance table is stale on P6 and contradicts `docs/POLICIES.md`.** It
records P6 as *"HALF — `tools/p6_gate.py` (topology; manual, NON-asserting, not in CI)"*, while
POLICIES.md and `CLAUDE.md` record that `p6_gate.py` has asserted in CI with `--assert` +
`--self-check` since 2026-08-06, and I ran `--assert` green at HEAD. Governance says this file
"MUST NOT contradict" POLICIES.md. **Out of scope for this lane and not fixed here** — flagged
because a plan that silently relied on the stale row would be leaning on an uncounted gate.

**Verdict: PASS.** No unjustified violation. Complexity Tracking below carries the one addition.

## Project Structure

### Documentation (this feature)

```text
specs/013-eprop-kernel-merge/
├── plan.md              # This file
├── spec.md              # Feature specification (20 FR, 5 NG, 10 SC, 4 R)
├── research.md          # Phase 0 output — the decisions and their evidence
├── data-model.md        # Phase 1 output — what the kernel holds vs what stays
├── quickstart.md        # Phase 1 output — the runnable gate + pin procedure
├── contracts/
│   └── eprop_kernel.md  # Phase 1 output — the kernel's public contract
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── cortex/
│   ├── eprop_kernel.py        # NEW (Commit 4) — the ONE e-prop mechanism
│   ├── eprop_nmnist.py        # EDIT — loses 9 defs, imports the kernel
│   ├── eprop_shd.py           # MOVED HERE (Commit 3), then loses 9 defs (Commit 4)
│   └── brain.py               # EDIT — :638-649 deleted (C1); try split (C2); :668 rewritten (C3)
├── substrate/
│   └── import_manifest.py     # EDIT — floor 7->6 (C1), UNGOVERNED_ROOT=() + 6->5 (C3)
├── reasoning/
│   └── pc_cortex_torch.py     # EDIT — dead surrogate-ceiling block deleted (C1)
└── __init__.py                # EDIT — TORCH_TIER regs (C3, C4); P2 debt note DELETED (C4)

training/
├── surrogate_snn.py           # DELETED (Commit 1)
└── eprop_shd.py               # MOVED OUT (Commit 3)

run_selfchecks.py              # EDIT — TORCH_GATED regs (C3, C4)
tools/p4c_ungoverned_baseline.json  # regenerated as counts fall
docs/                          # TRAINING.md:23 (C3) + coverage/backlog co-edits (C1)
```

**Structure Decision**: the kernel lands in `src/snn_data_lake/cortex/` — not `plasticity/`, where
e-prop belongs anatomically — because both consumers are cortical sensory encoders and a
cortex→other-region import reddens the p6 `undeclared_import_pairs` ratchet (11→12) **[F]**.
Rationale and the rejected alternative are in [research.md](research.md) D-1.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| **A third module** (`eprop_kernel.py`) where two exist today | P2 MERGE is defined as "build the ONE mechanism that supersedes both, then retire both originals". The file count rises 2→3 while the *mechanism* count falls 2→1, which is the reduction P2 asks for. Net source reduction ~100 lines. | **Fold the shared code into `eprop_nmnist` and have `eprop_shd` import from it** — rejected: it makes the vision module the implicit owner of the audio module's engine, so the audio consumer cannot be understood or tested without the vision one, and any future third consumer inherits N-MNIST framing. It also leaves the "which is authoritative" ambiguity P2 exists to kill. **Do nothing / keep two copies** — rejected: that is the P2 debt this lane exists to discharge, and it is already recorded as debt at `__init__.py:415-417`. |
| **Commit 0 produces artifacts, not behaviour** | FR-017 (bit-exact pin) and FR-018 (unique-capability dispositions) must both exist *before* the commits they gate. Folding them into those commits would mean the evidence and the change it licenses land together, unreviewable. | Folding the audit into Commit 1 — rejected: it would put a "should we delete this at all" finding inside the commit that deletes it, which is the shape where an inconvenient finding gets quietly dropped. |

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 artifacts (`research.md`, `data-model.md`, `contracts/eprop_kernel.md`,
`quickstart.md`):

- **P2 survivor count** — `data-model.md` enumerates exactly one definition site per shared symbol
  after Commit 4. Verified as a table, not asserted prose. **PASS.**
- **P3 charge** — `contracts/eprop_kernel.md` records that neutering the kernel import makes both
  consumers fail to import, which is charge evidence rather than a claimed consumer. **PASS.**
- **P8 reachable red** — `contracts/eprop_kernel.md` names the concrete input that reddens the
  kernel self-check (a sign-flipped learning signal driving cosine negative). The criterion is
  therefore falsifiable and survives FR-019. **PASS.**
- **Standing Invariant 3** — the accuracy verdict is recorded as UNKNOWN in `research.md` D-2 and
  the deletion rests on rule + speed + params, never on the tie. **PASS.**
- **Standing Invariant 5** — `quickstart.md` makes the bit-exact pin the Commit 4 acceptance step
  and states in-line that the cosine assert is not sufficient for it. **PASS.**

**No new violations introduced by the design.**
