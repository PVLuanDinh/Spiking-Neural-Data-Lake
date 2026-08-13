# Feature Specification: E-prop Kernel MERGE + Surrogate Perception Tier Retirement

**Feature Branch**: `013-eprop-kernel-merge`

**Created**: 2026-08-12

**Status**: Draft

**Input**: User description: "Merge the duplicated e-prop training kernel into one governed mechanism and retire the surrogate-BPTT perception tier."

**Measurement basis**: HEAD `47555a74`, tree clean. Every figure in this spec was produced by an
adversarially-verified measurement pass (5 dimensions x refute-by-default skeptics, 48 findings
surviving of 58) and the load-bearing structural claims were then re-derived independently by the
main agent. Figures NOT re-derived by the main agent are marked **[F]**. Nothing here is relayed
from the 2026-08-04/05 promotion-plans record, whose cites predate the `eprop_nmnist` promotion and
are stale.

## Why this feature exists

P2 requires that exactly ONE mechanism survive for one job. This tree currently violates that twice
over in the e-prop perception path:

1. **One engine, two copies.** `src/snn_data_lake/cortex/eprop_nmnist.py` (governed, TORCH_TIER,
   vision/N-MNIST) and `training/eprop_shd.py` (ungoverned, audio/SHD) hold **nine byte-identical
   definitions** — verified at HEAD, line-exact:

   | definition | eprop_nmnist.py | eprop_shd.py | lines |
   |---|---|---|---|
   | `class _SpikeFn` | L352-363 | L344-355 | 12 |
   | `bptt_grads` | L366-387 | L358-379 | 22 |
   | `_cosine` | L390-395 | L382-387 | 6 |
   | `_apply_grads` | L526-536 | L515-525 | 11 |
   | `evaluate` | L539-553 | L528-542 | 15 |
   | `psi` | L222-224 | L188-190 | 3 |
   | `offdiag_mask` | L230-231 | L198-199 | 2 |
   | encoder `__init__` | L426-428 | L408-410 | 3 |
   | `_SpikeFn.forward` | L353-357 | L345-349 | 5 |

   Plus `forward` (A L233-255 / B L201-223) at ratio **0.9939**, differing on **exactly one line** —
   the docstring shape annotation (`A:235` event frames vs `B:203` audio frames) — and five
   byte-identical, same-valued LIF constants (`DEFAULT_ALPHA` 0.9, `DEFAULT_V_TH` 0.6,
   `DEFAULT_GAMMA` 0.5, `DEFAULT_DAMP` 0.3, `DEFAULT_KAPPA` 0.9). Roughly **102 lines per file**.

2. **Two encoders behind a flag.** `cortex/brain.py:631` (`if os.path.exists(_eprop)`) selects the
   e-prop encoder, and `:638-649` (`else:`) selects `training/surrogate_snn.py`'s surrogate-BPTT
   encoder. P2 names this shape explicitly: a merge that keeps both paths behind a flag is
   *additive with extra steps*.

A third defect sits in the same code and is fixed here because it makes the rest of the work
unsafe to verify: `cortex/brain.py:597` opens ONE `try` closed by a blanket `except Exception:` at
`:679` **with no `as name`**, spanning both the vision half (`:630-663`) and the audio half
(`:664-678`), resetting the tier to `"stdlib"` at `:680-681`. An audio import failure therefore
rolls back an already-completed vision tier **[F]**, and any mis-typed import rewrite fails
silently.

## User Scenarios & Testing *(mandatory)*

The "user" of this feature is the brain itself and the gate set that governs it. Each story is an
independently landable commit that leaves the full gate green.

### User Story 1 - Retire the surrogate-BPTT perception tier (Priority: P1)

The brain stops carrying a second, non-brain-native perception encoder. `training/surrogate_snn.py`
is deleted along with its only importer (`cortex/brain.py:638-649`) and the dead surrogate-ceiling
block in `reasoning/pc_cortex_torch.py`.

**Why this priority**: It is the largest single reduction, it is independent of the kernel work,
and it removes one of the two `training.*` entries from the import manifest — which must happen in
its own commit (see FR-012).

**Independent Test**: Delete the module and its co-edit surface; run the full gate. The brain still
constructs, the e-prop vision tier still loads where its checkpoint exists, and no gate reddens.

**Acceptance Scenarios**:

1. **Given** a machine with `models/eprop_nmnist.pt` present, **When** the brain is constructed,
   **Then** `perception_tier == "eprop-snn"` exactly as before this change.
2. **Given** the tree after deletion, **When** `grep -r "surrogate_snn\|SurrogateEncoder\|torch-snn"`
   is run over everything except `CHANGELOG.md`, **Then** zero live references remain (historical
   CHANGELOG entries are left untouched).
3. **Given** the tree after deletion, **When** `python -m snn_data_lake.substrate.import_manifest`
   runs, **Then** it reports 7 targets and passes, with the floor at `:287` lowered 7 -> 6 in this
   same commit.
4. **Given** `reasoning/pc_cortex_torch.py` after deletion, **When** it is run, **Then** it produces
   the same output it produced before — because the removed block's `surro_acc` was always `None`
   and its `:481` branch never fired **[F]**.

---

### User Story 2 - Split the perception try block (Priority: P1)

The vision and audio perception tiers get independent `try` blocks, so a failure in one cannot
silently roll back the other.

**Why this priority**: This is an independent latent defect, and it must land BEFORE Story 3
because the blanket `except Exception:` at `:679` would swallow a mis-typed import rewrite in that
story — while the import manifest would go *green*, since the target simply vanished.

**Independent Test**: Block `training.eprop_shd` at import (meta_path blocker), construct the brain,
and confirm the vision tier survives.

**Acceptance Scenarios**:

1. **Given** an importable vision encoder and a deliberately unimportable audio encoder, **When**
   the brain is constructed, **Then** `perception_tier` reflects the loaded vision encoder and is
   NOT `"stdlib"`.
2. **Given** the split, **When** the recorder observable is compared before and after **at >= 12
   ticks**, **Then** the leaf-path count is unchanged. A zero-tick comparison is VACUOUS and must
   not be used: at 12 ticks a rollback collapses `/cognition/attention_real` from a 3-key dict to a
   scalar, 237 -> 235 leaf paths **[F]**.

---

### User Story 3 - Promote the audio e-prop trainer into the governed tree (Priority: P2)

`training/eprop_shd.py` becomes `src/snn_data_lake/cortex/eprop_shd.py`, registered and gated.

**Why this priority**: It has a live importer (`brain.py:668`) and drives the auditory tier, so it
must be ported rather than deleted. It carries the highest gate risk in the lane — four gates move
at once.

**Independent Test**: After the move, all four asserting gates plus the self-check suite pass, and
the auditory tier still loads from `models/eprop_shd.pt`.

**Acceptance Scenarios**:

1. **Given** the move, **When** the dormancy gate runs, **Then** the module is reachable — which
   requires `brain.py:668` rewritten to `from snn_data_lake.cortex.eprop_shd import
   EpropAudioEncoder` in the SAME commit, because `dormancy_gate.py:353`'s `_PKG` guard gives a
   bare-root `training.*` import no graph edge.
2. **Given** the move, **When** `tools/citation_guard.py --assert` runs, **Then** it passes —
   requiring `docs/TRAINING.md:23` rewritten from
   `.venv-lake/Scripts/python training/eprop_shd.py` to the dotted `-m` form matching its sibling
   at `:22`, which carries no `/` and no `.py` and therefore matches nothing.
3. **Given** the move, **When** `tools/p4c_ratchet.py --assert` runs, **Then** it passes —
   requiring `"eprop_shd"` added to `TORCH_TIER["cortex"]`, without which `cortex` becomes a new
   zone with implicit allowance 0 and goes RED.
4. **Given** the move, **When** `python -m snn_data_lake.cortex.eprop_shd` is run, **Then** its
   self-check executes and passes.

---

### User Story 4 - Extract the one shared e-prop kernel (Priority: P1)

`src/snn_data_lake/cortex/eprop_kernel.py` is created holding the shared definitions; both e-prop
modules import from it; the duplicated definitions are deleted from both.

**Why this priority**: This is the commit that actually discharges the P2 debt. Stories 1-3 are
prerequisites; only this one leaves exactly one mechanism standing.

**Independent Test**: Numeric outputs of both consumer modules are bit-identical before and after.

**Acceptance Scenarios**:

1. **Given** pinned pre-merge values, **When** the merge lands, **Then**
   `forward.logits_sum`, `forward.latent_sum`, `eprop_grads.grads`, `eprop_grads.loss` and
   `bptt_grads.grads` are **BIT-EXACTLY equal** on BOTH modules.
2. **Given** the kernel, **When** `python -m snn_data_lake.cortex.eprop_kernel` runs, **Then** it
   builds a minimal concrete subclass inline (importing nothing from its consumers) and asserts
   `cosine > EPROP_BPTT_COSINE_FLOOR`.
3. **Given** the merge, **When** `src/snn_data_lake/__init__.py:415-417` is inspected, **Then** the
   P2 debt note is **absent** — deleted, not amended.

### Edge Cases

- **A machine holding only `models/surrogate_nmnist.pth`** goes from a working 92.58% encoder to
  **no encoder at all**. This is the correct P2 outcome but it is a real cost and must be stated,
  not glossed.
- **A fresh clone or CI** loads NEITHER encoder: `.gitignore:143` is `models/*` with negations only
  for the two packs, so neither checkpoint ships. The e-prop path is equally unreachable there.
- **Off-root / non-repo CWD**: the relative checkpoint paths at `brain.py:630` and `:664` fail their
  `os.path.exists` guards, so the imports are never attempted at all **[F]**. Splitting the try does
  NOT fix this; root-anchoring those paths is out of scope.
- **`SNN_EPROP_CKPT` redirect**: a documented env override reaches the `else:` branch even when the
  default checkpoint exists **[F]**.
- **A mis-typed import in Story 3** would be swallowed by `:679` and would make the manifest go
  *green*. Story 2 exists to make this visible.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The tree MUST contain exactly ONE definition of each of the nine byte-identical
  e-prop definitions listed above, in `src/snn_data_lake/cortex/eprop_kernel.py`.
- **FR-002**: The kernel MUST hold a shared LIF/e-prop base class (carrying `psi`, `offdiag_mask`,
  `forward`, `flatten_frames`), a shared encoder base (carrying `__init__`), the five LIF
  constants, and `EPROP_BPTT_COSINE_FLOOR = 0.5`.
- **FR-003**: `eprop_shd.py`'s hand-written bare `0.5` at `:830` MUST be replaced by the imported
  `EPROP_BPTT_COSINE_FLOOR`, eliminating a P1-adjacent magic constant.
- **FR-004**: The kernel MUST live in `cortex/`, not `plasticity/`. A cortex -> other-region import
  drives `p6_gate`'s `undeclared_import_pairs` ratchet from 11 to 12 and reddens it **[F]**; both
  consumers are cortical sensory encoders, so cortex -> cortex adds nothing.
- **FR-005**: The kernel MUST NOT import from `eprop_nmnist` or `eprop_shd` (import cycle). Its
  `__main__` self-check MUST build a minimal concrete subclass inline.
- **FR-006**: The kernel MUST be registered in `TORCH_TIER["cortex"]` and in `run_selfchecks.py`
  TORCH_GATED.
- **FR-007**: `flatten_frames` MUST collapse to a single body — `eprop_shd`'s
  `x.reshape(x.shape[0], x.shape[1], self.n_in)`, which subsumes `eprop_nmnist`'s on every shape
  either class receives **[F]**. The per-modality seam is `self.n_in`, already a per-subclass
  attribute. Keeping two bodies where one suffices is additive with extra steps.
- **FR-008**: The merged `forward` docstring MUST describe BOTH input shapes. Silently adopting one
  modality's wording is not acceptable.
- **FR-009**: `encode` and `classify` (ratios 0.960 / 0.947 **[F]**) MUST NOT be hoisted into the
  kernel on a similarity ratio alone. A ratio cannot separate material from cosmetic difference.
  They may be hoisted only after the same bit-identity A/B that justifies FR-007.
- **FR-010**: `training/surrogate_snn.py` MUST be deleted, together with `cortex/brain.py:638-649`
  and `reasoning/pc_cortex_torch.py:427-446` + `:481-490`.
- **FR-011**: `training/eprop_shd.py` MUST be moved (`git mv`) to
  `src/snn_data_lake/cortex/eprop_shd.py`, with `brain.py:668` rewritten in the SAME commit.
- **FR-012**: The two `training.*` manifest targets MUST be removed in SEPARATE commits, each
  lowering the `import_manifest.py:287` floor by one in its own commit. Removing both at once fires
  TWO asserts: `:287` at `len 6 < 7`, and `:306` on `unmatched=['training']`.
- **FR-013**: When the LAST `training.*` target is removed, `import_manifest.py:126` MUST be set to
  `UNGOVERNED_ROOT = ()`, which `:302-304` pre-authorises, and the stale comment at `:111-113`
  naming the two root targets MUST be deleted.
- **FR-014**: `cortex/brain.py`'s single perception `try` (`:597`-`:679`) MUST be split into an
  independent vision `try` and audio `try`.
- **FR-015**: The P2 debt note at `src/snn_data_lake/__init__.py:415-417` MUST be DELETED once the
  lane closes. A surviving debt marker for discharged debt is a false claim.
- **FR-016**: Every co-edit site listed in **Key Entities -> Co-edit surface** MUST be updated in
  the commit that invalidates it. A co-edit slipping to a later commit reddens the dormancy gate or
  the citation guard.
- **FR-017**: Before FR-001 lands, the numeric outputs named in User Story 4 scenario 1 MUST be
  pinned bit-exactly on both modules, and asserted equal afterwards.
- **FR-018** *(P2 honesty guard — "any unique capability of the retired mechanism ported first")*:
  Before `training/surrogate_snn.py` is deleted, EACH of its **19 surrogate-only definitions** MUST
  receive an explicit disposition — (a) proven already covered by `cortex/eprop_nmnist.py`,
  (b) PORTED, or (c) recorded as deliberately dropped WITH its reason. Silent loss is a P2
  violation regardless of how the A/B came out. The 19: `SpikingConvNet`, `LeakyLIF`,
  `SurrogateEncoder`, `_FastSigmoid`, `_fast_sigmoid`, `_first_layer_grad`, `_flat_dim`, `_fwd_bwd`,
  `_heaviside_no_surrogate`, `_prepare`, `_repo_root`, `_self_check`, `_self_check_body`,
  `_synthetic_frames`, `assert_supported_reset`, `init_leaky`, `inner`, `lif_layers`,
  `ng_snn_data_file`. **Two are flagged as probable REAL unique capability and must not be waved
  through**: `SpikingConvNet` is a CONVOLUTIONAL spiking architecture where `eprop_nmnist` is a
  recurrent fully-connected LIF — a different architecture, not a worse copy of the same one; and
  `_heaviside_no_surrogate` is a surrogate-OFF control, which P8 guard (2) values as "a control that
  is not the method itself". `ng_snn_data_file` and `_repo_root` additionally have out-of-tree
  referents (`research/ng_snn/data.py:19`, `research/README.md:54`,
  `research/neuromorphic_torch/neuro_chip_gpu.py:52`).
- **FR-019** *(P8 — "what an invented method owes", guard 1)*: the kernel's new `__main__`
  self-check MUST have a **reachable RED**. Construct and record the input that makes
  `cosine > EPROP_BPTT_COSINE_FLOOR` FAIL. If no such input exists, the criterion is **deleted, not
  weakened**. A self-check that cannot fail is not a gate, and this lane's whole value is that the
  kernel is gated.
- **FR-020** *(cross-cutting invariant)*: every file:line citation written into a commit message,
  CHANGELOG entry, or doc by this lane MUST be re-derived AFTER the last edit of the commit that
  touches that file. POLICIES.md records the project already paying for this — a cited file was
  edited between two reads in one session and its line numbers moved by one, and the standing note
  is that *"a line citation into an actively-developed file is stale on a timescale of minutes, not
  releases."* This lane rewrites both cited files, so every cite in this spec is a candidate for
  rot the moment Story 1 lands.

### Non-Goals (explicit scope limits)

- **NG-001**: The kernel serves **TWO** consumers, not four, and MUST NOT be documented otherwise.
  The 4-arg `_SpikeFn` returns `grad_out * psi, None, None, None` (`eprop_nmnist.py:363`) — no
  gradient to the threshold argument. Harmless where `v_th` is a float constant; NOT harmless for
  `training/eprop_lm_alif.py:103`, whose threshold is a live adapted tensor (measured: the 1-arg
  form gives `||d spikes/d a|| = 1.0727`, the 4-arg form gives `None`) **[F]**.
- **NG-002**: Unifying the other in-tree `autograd.Function` surrogates —
  `reasoning/rsnn_delays_torch.py:91`, `training/eprop_lm.py:78`, `training/eprop_lm_alif.py:103`,
  `training/ar_spiking_lm.py:74` — is a SEPARATE lane owing its own A/B.
- **NG-003**: The `reasoning/ng_snn/` <-> `research/ng_snn/` duplicated subtree is out of scope; it
  is a repoint-then-retire lane of its own, and `research/ng_snn/` also holds ~47 MB of checkpoints
  needing separate disposition **[F]**.
- **NG-004**: Root-anchoring the relative checkpoint paths at `brain.py:630` / `:664` is out of
  scope.
- **NG-005**: `src/snn_data_lake/reasoning/eprop.py` is **NOT** a third copy of the kernel and is out
  of scope. Measured: it is a pure-stdlib e-prop reference (378 lines, `math`/`zip`, no torch, no
  `autograd`), it shares only four generic names with `eprop_nmnist` (`__init__`, `_cosine`,
  `forward`, `main`), and its `_cosine` (L282-286, `math.sqrt`) is a DIFFERENT implementation from
  `eprop_nmnist`'s (L390-395, `torch.norm`) — different substrates, not duplication. It is the
  verification anchor POLICIES.md cites for the whole e-prop invariant (`:38-40` states the cosine
  bar; `grad_bptt` at `:196` "shares NO code with grad_eprop"). Flagged, not folded in: it sits in
  the ungoverned `reasoning/` 109, so the invariant's own anchor is outside the gates — a real P4c
  item owed its own lane.

### Key Entities

- **The e-prop kernel** — `src/snn_data_lake/cortex/eprop_kernel.py`. The one surviving mechanism.
  Holds the surrogate spike function, the independent BPTT reference, the cosine control, the Adam
  step, the evaluation loop, the shared LIF base, the shared encoder base, and the shared constants.
- **The vision consumer** — `cortex/eprop_nmnist.py`. Keeps `eprop_grads`, `encode`, `classify`,
  `_prep`, `load`, `config`, `main`, and its own geometry (`NUM_CLASSES` 10, `DEFAULT_T` 15,
  `DEFAULT_HIDDEN` 64).
- **The audio consumer** — `cortex/eprop_shd.py` after promotion. Same shape, different geometry
  (`NUM_CLASSES` 20, `DEFAULT_T` 100, `DEFAULT_HIDDEN` 128). These values differ, so they MUST NOT
  move into the kernel.
- **Co-edit surface** (swept at HEAD; the measurement pass named only two of these):
  `pyproject.toml:203`, `tools/p4c_ungoverned_baseline.json:242`,
  `cortex/eprop_nmnist.py:6/:411/:424/:434`,
  `cortex/brain.py:563/:641/:646/:648/:649/:3301`, `substrate/import_manifest.py:109/:111-113`,
  `training/eprop_shd.py:417`, `training/ar_spiking_lm.py:23`, `research/ng_snn/data.py:19`,
  `research/README.md:54`, `research/neuromorphic_torch/neuro_chip_gpu.py:52`,
  `docs/TRAINING.md:23`, `docs/coverage/Perception.md:195`, `docs/coverage/SpatialTemporal.md:27`,
  `docs/COVERAGE_TO_100.md:68`, `docs/AUDIT_BACKLOG.md` (3 sites), `docs/P4C_BACKLOG.md:269`.
  `brain.py:641` additionally checks a SECOND checkpoint name (`models/nmnist_encoder.pt`).

## The ruling on `training/surrogate_snn.py`

**DELETE — conditional on FR-018 clearing first.** Recorded here with its evidence AND its limits,
because the limits are what keep the commit message honest.

**The condition is not a formality.** P2's honesty guard requires "any unique capability of the
retired mechanism **ported first**", and the A/B below does not discharge it: an A/B compares the
two mechanisms *on the job they share*, and says nothing about what only one of them can do.
`surrogate_snn` holds 19 definitions `eprop_nmnist` does not, of which `SpikingConvNet` (a
convolutional spiking architecture, against `eprop_nmnist`'s recurrent fully-connected LIF) and
`_heaviside_no_surrogate` (a surrogate-off control) are probable real unique capability. If FR-018
finds that either is genuinely load-bearing and unported, the deletion is **blocked** until it is
ported or explicitly and reasonedly dropped. Discovering that mid-lane is a legitimate outcome, not
a failure of the plan.

**It is not the BPTT control.** Measured at HEAD: `surrogate_snn.py` contains **zero** occurrences
of `bptt_grads`, `_cosine`, `_apply_grads` or `_SpikeFn`; its only backward path is
`loss.backward()`. Both e-prop modules carry their own independent reference (`bptt_grads` at
`eprop_nmnist.py:366` and `eprop_shd.py:358`). The e-prop invariant was measured at cosine
**0.9997** with `training.surrogate_snn` made unimportable by a `meta_path` blocker **[F]**.
Deleting it therefore does not touch the cosine > 0.5 invariant.

**The A/B P2 demands EXISTS and has been run** **[F]**: e-prop 238/256 = **92.97%** vs surrogate
237/256 = **92.58%** on the N-MNIST test split; McNemar exact **p = 1.0000**; 95% CI on the paired
difference **[-2.77, +3.55] pp**; e-prop **7.8-10.6x faster** on cpu and ~5x on cuda; **152,714 vs
273,806 params** (0.56x); untrained-weights control 8-13%.

**The only defensible commit sentence**: accuracy indistinguishable at n=256 (p=1.0000, CI
[-2.8,+3.6] pp — underpowered, not proven equal); e-prop 7.8-10.6x faster on cpu at 0.56x the
parameters; and the surrogate branch is dormant on any machine where the e-prop checkpoint exists.

**Three claims the work MUST NOT make**:

- NOT "e-prop is more accurate" — a one-sample gap at p=1.0000 is a tie.
- NOT "the surrogate path is unreachable on a fresh clone" — neither checkpoint ships, so the
  e-prop path is equally unreachable; three states that DO reach `:646` were measured **[F]**.
- NOT "this costs nothing" — see Edge Cases.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The count of byte-identical definitions shared between the two e-prop consumer
  modules falls from **9 to 0**, measured by the same AST+difflib census used to produce this spec.
- **SC-002**: All four asserting gates (`p4c_ratchet`, `p6_gate`, `import_manifest`,
  `citation_guard`) plus `red_reachability` and `dependency_ledger` are green at every commit in
  the lane, not merely at the end.
- **SC-003**: The numeric outputs named in FR-017 are **bit-exactly identical** before and after
  the merge on both consumer modules. Any difference blocks the lane.
- **SC-004**: `p4c_ratchet` ungoverned total falls from **205** to **203** (`training` zone 6 -> 4);
  the ratchet only fires on a rise, so a fall is always safe.
- **SC-005**: The import manifest falls from **8** targets to **6**, with `UNGOVERNED_ROOT` reduced
  to `()` and its floor stepped down one commit at a time.
- **SC-006**: Zero live references to `surrogate_snn` / `SurrogateEncoder` / `torch-snn` remain
  outside `CHANGELOG.md`.
- **SC-007**: The perception tier observable is unchanged for a machine holding
  `models/eprop_nmnist.pt`, measured at **>= 12 ticks** (a 0-tick measurement is vacuous).
- **SC-008**: Blocking the audio encoder no longer rolls back the vision tier.
- **SC-009**: All **19** surrogate-only definitions carry a written disposition (covered / ported /
  dropped-with-reason) before the deletion commit lands. Count checked, not asserted: 19 of 19.
- **SC-010**: The kernel's self-check has a **recorded reachable RED** — a named input that makes it
  fail, demonstrated once. A criterion with no demonstrated red is deleted rather than kept.

## Assumptions

- The gate environment is `.venv-lake` (python 3.14.6 free-threading, torch 2.11.0+cu128, cuda
  True, snntorch present, **tonic absent**). Anything requiring `tonic` is out of reach without an
  install.
- `models/eprop_nmnist.pt` and `models/eprop_shd.pt` exist on this box and are gitignored, so local
  gate results differ from CI by construction; CI loads neither encoder.
- The A/B figures marked **[F]** were produced by the measurement pass at n=256 and are carried into
  the spec as-is. They are sufficient to justify the deletion under P2 (which requires an A/B to
  exist, not a win) but they are NOT sufficient to claim an accuracy result. A better-powered rerun
  at n=10,000 would most likely buy a better-powered null (power calc gives p~0.13) **[F]**.
- One change per commit with the full gate green, per the project's cross-cutting invariant.
- `CHANGELOG.md` historical entries are never rewritten; only live references are updated.

## Risks

- **R-001 (top risk, and it is silent rather than red)**: the kernel is safe wherever it rests on
  byte-identity. It rests on JUDGMENT in exactly one place — the FR-007 `flatten_frames` collapse.
  If that call is wrong, a wrong tensor shape reaches `eprop_grads`, and **the cosine assert at
  `eprop_nmnist.py:783` would very likely still PASS**, because `bptt_grads` computes its reference
  through the SAME `forward` — both sides wrong together. *A control that compares a mechanism
  against itself cannot detect a change in that mechanism.* FR-017's bit-exact pin is the only
  mitigation, and it is therefore not optional.
- **R-002**: `brain.py:679`'s blanket `except Exception:` has no `as name`. A mis-typed import
  rewrite in Story 3 deletes the audio tier silently, and the import manifest goes *green* because
  the target vanished. Story 2 is sequenced ahead specifically to make this visible.
- **R-003**: Collapsing Stories 1 and 3 into one "retire the training/ e-prop stuff" commit is the
  intuitive move and it fires two asserts. They must stay separate (FR-012).
- **R-004**: `UNGOVERNED_ROOT = ()` records "no AST-visible root-shape imports", NOT "no governed ->
  `training/` edges". A surviving invisible one was measured:
  `src/snn_data_lake/reasoning/eprop_lm_serve.py:261-264` does `sys.path.insert` then
  `import eprop_lm as trainer`, a bare-name import `_targets()` cannot see. This caveat belongs in
  the commit body.
