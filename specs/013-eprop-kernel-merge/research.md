# Phase 0 Research — E-prop Kernel MERGE

All measurements at HEAD `47555a74`, tree clean. Gate env
`D:\AI-project\spiking-neural-data-lake\.venv-lake\Scripts\python.exe` (python 3.14.6
free-threading, torch 2.11.0+cu128, cuda True, snntorch present, **tonic absent**).

Claims re-derived by the main agent are unmarked. Claims carried from the measurement pass and NOT
independently re-run are marked **[F]**; each one that gates a commit is scheduled as a pre-commit
re-measurement in [quickstart.md](quickstart.md) rather than trusted here.

---

## D-1 — The kernel lands in `cortex/`, not `plasticity/`

**Decision**: `src/snn_data_lake/cortex/eprop_kernel.py`.

**Rationale**: anatomically e-prop is a plasticity rule and `plasticity/` is the honest home. It is
rejected on a measured gate consequence: `tools/p6_gate.py` ratchets `undeclared_import_pairs` at
**11**, and a `cortex → plasticity` import drives it to **12**, reddening the ratchet **[F]**. Both
consumers (`eprop_nmnist`, `eprop_shd`) are cortical sensory encoders, so `cortex → cortex` adds no
pair at all.

**Alternatives considered**:
- `plasticity/eprop_kernel.py` + declare the `cortex → plasticity` pair in `connectome.py` first.
  Rejected *for this lane*, not on principle: it is a second, independent change to the connectome
  and P6's declared-route surface, and bundling it would break "one change per commit". It is the
  correct follow-up if the kernel later acquires a non-cortical consumer.
- Leave the shared code in `eprop_nmnist.py` and import from it. Rejected — see plan.md Complexity
  Tracking; it makes the vision module the implicit owner of the audio engine.

**Consequence recorded**: the kernel's placement is a *gate-driven* compromise against anatomy.
That is worth saying out loud rather than presenting `cortex/` as the natural home.

---

## D-2 — Deleting `training/surrogate_snn.py`: the verdict is UNKNOWN, the decision is DELETE

**Decision**: delete, conditional on FR-018 clearing. **The accuracy comparison is recorded as
UNKNOWN and is not part of the justification.**

**Why this framing rather than "e-prop won"**: Standing Invariant 3 — *"the negative verdict is the
dangerous one, because it is the one that retires a working faculty… where a probe cannot
distinguish, it MUST report UNKNOWN rather than a confident negative."* The A/B **[F]** returned:

| | e-prop | surrogate |
|---|---|---|
| accuracy (N-MNIST test, n=256) | 238/256 = **92.97%** | 237/256 = **92.58%** |
| McNemar exact | **p = 1.0000** | |
| 95% CI, paired difference | **[-2.77, +3.55] pp** | |
| speed (cpu) | **7.8–10.6× faster** | baseline |
| speed (cuda) | ~5× faster | baseline |
| parameters | **152,714** | 273,806 (0.56×) |
| untrained-weights control | 8–13% | 8–13% |

A one-sample gap at p = 1.0000 is a **tie**, and n=256 is underpowered — a power calculation puts
p≈0.13 even at n=10,000 **[F]**, so a larger rerun most likely buys a better-powered null, not a
win. Calling that "e-prop is more accurate" would be exactly the confident negative the invariant
forbids.

**What the deletion actually rests on** — three grounds, none of which is underpowered:

1. **P2 forbids the coexistence regardless of which is better.** `brain.py:631` (if) vs `:638-649`
   (else) is two live mechanisms for one job selected by a flag. P2's merge-specific guard: *"if two
   implementations of the same job are still reachable when you are done, it was not a merge."*
2. **Standing Invariant 8 breaks the tie by rule.** *E-prop, not backprop.* The surrogate path is
   surrogate-gradient BPTT (`loss.backward()` at `:11, 387, 453, 517, 524`). When two mechanisms tie
   on evidence, the one the invariant names is the one that survives.
3. **Measured cost asymmetry**: 7.8–10.6× faster at 0.56× the parameters.

**The BPTT-control worry does not apply** (measured, not relayed): `surrogate_snn.py` contains
**zero** occurrences of `bptt_grads`, `_cosine`, `_apply_grads`, `_SpikeFn`. Both e-prop modules
carry their own independent `bptt_grads` (`eprop_nmnist.py:366`, `eprop_shd.py:358`). The e-prop
invariant was measured at cosine **0.9997** with `training.surrogate_snn` made unimportable by a
`meta_path` blocker **[F]**. Deleting it does not remove the control the invariant depends on.

**The condition that is NOT discharged by any of the above** — P2's honesty guard requires *"any
unique capability of the retired mechanism ported first"*, and an A/B compares only the *shared*
job. `surrogate_snn` has **19** definitions `eprop_nmnist` lacks. Two are probable real unique
capability and are called out so they cannot be waved through:

- **`SpikingConvNet`** — a **convolutional** spiking architecture. `eprop_nmnist` is a recurrent
  fully-connected LIF. This is a different architecture, not an inferior copy of the same one.
- **`_heaviside_no_surrogate`** — a surrogate-OFF control, the kind P8 guard (2) specifically
  values ("a control that is not the method itself").

Also carrying out-of-tree referents: `ng_snn_data_file` and `_repo_root`
(`research/ng_snn/data.py:19`, `research/README.md:54`,
`research/neuromorphic_torch/neuro_chip_gpu.py:52`).

**Open until Commit 0**: if FR-018 finds either load-bearing and unported, the deletion is
**blocked** until it is ported or reasonedly dropped. That is a legitimate outcome.

**Cost, recorded not glossed**: deleting `:638-649` removes the only `"torch-snn"` tier. A machine
holding only `models/surrogate_nmnist.pth` goes from a working 92.58% encoder to **no encoder**.

---

## D-3 — The bit-exact pin runs LOCALLY ONLY and is not a CI assert

**Decision**: FR-017's pin is a local pre/post procedure recorded in the commit body. It is **not**
committed as a pytest assert.

**Rationale**: the local gate runs torch 2.11.0+cu128 on a free-threaded 3.14 interpreter; **CI has
no GPU and installs CPU torch**. CLAUDE.md states the consequence plainly: *"a build-sensitive
numeric assert can be green in one and red in the other."* Bit-exact float equality across two torch
builds is not a contract this project can hold. Committing it as a CI assert would manufacture a red
that says nothing about whether the merge preserved behaviour — which is Standing Invariant 5's
failure mode pointed the other way (a gate that fires for reasons unrelated to what it checks).

**What makes this safe rather than a dodge**: the pin's *job* is to catch a `flatten_frames`
mistake made by this lane, on this machine, between two commits minutes apart — same interpreter,
same build, same seeds. That is precisely the comparison where bit-exactness is meaningful and
cheap. It is a **verification step**, not a **regression gate**, and the plan says so rather than
promoting it to a gate it cannot be.

**Alternative considered**: a tolerance-based (`allclose`) assert committed to CI. Rejected — a
tolerance wide enough to survive a CPU/GPU build difference is wide enough to hide a real
`flatten_frames` reshape error, which produces plausible-magnitude but wrong values. A control that
cannot fail on the defect it exists to catch is deleted, not weakened (P8 guard 1).

**What IS committed**: the kernel's own `__main__` cosine self-check (build-tolerant, `> 0.5`), and
the pre-existing consumer self-checks. Those run in CI.

---

## D-4 — `flatten_frames` collapses to one body

**Decision**: the kernel carries **one** concrete `flatten_frames`, `eprop_shd`'s body
`x.reshape(x.shape[0], x.shape[1], self.n_in)`. Not an abstract method, not two overrides.

**Rationale**: measured to subsume `eprop_nmnist`'s `x.reshape(*x.shape[:-3], self.n_in)` on every
shape either class receives — bit-identical outputs for `forward`, `eprop_grads`, `bptt_grads`
across 4 configs, while the `eprop_nmnist` body *raises* `RuntimeError` on 3-D audio input **[F]**.
The genuine per-modality seam is `self.n_in`, already a per-subclass attribute
(`eprop_nmnist.py:202`, `eprop_shd.py:173`). Two bodies where one suffices is additive with extra
steps.

**This is the lane's only judgment call and it is where R-001 lives.** It is therefore the one
decision gated by the bit-exact pin rather than by a gate. Re-run the subsumption A/B as a
pre-commit task (quickstart step C4-1) — do not carry the **[F]** into the commit.

**Explicitly NOT extended to `encode` / `classify`** (ratios 0.960 / 0.947 **[F]**). A similarity
ratio cannot separate material from cosmetic difference; hoisting on a ratio is the same vacuity
that a skeptic already overturned once in this lane's measurement pass. FR-009 bars it until the
same bit-identity evidence exists.

---

## D-5 — Why the try-split lands *before* the move

**Decision**: Commit 2 (split) precedes Commit 3 (move), though the move does not depend on it.

**Rationale**: `cortex/brain.py:679` is `except Exception:` with **no `as name`**, closing a `try`
opened at `:597` that spans both the vision half (`:630-663`) and the audio half (`:664-678`), and
resetting to `"stdlib"` at `:680-681`. Two consequences, both measured or structural:

1. A mis-typed import rewrite in Commit 3 fails **silently** — the tier just becomes `"stdlib"`.
2. Worse, `import_manifest` would go **GREEN** on that failure, because the target simply vanished
   from the scan. The gate that exists to catch a broken cross-tier import cannot catch this one.

Splitting first makes Commit 3's failure mode visible. The lane can proceed without it; it is
simply riskier, and the risk is of the invisible kind.

**Independently, the split fixes a real defect**: blocking `training.eprop_shd` rolls back a vision
tier that had **already completed**, leaving 256 frames loaded while `_perceiving_real=False` **[F]**.

**Scope limit**: the split does **not** fix the off-root/CWD shape. Off-root, the *relative*
checkpoint paths at `brain.py:630` and `:664` fail their `os.path.exists` guards and the imports are
never attempted at all **[F]**. Root-anchoring those is NG-004.

---

## D-6 — Floor stepping: one removal, one step, one commit

**Decision**: `import_manifest.py:287`'s floor goes 7→6 in Commit 1 and 6→5 in Commit 3. Never
batched, never ahead of the removal.

**Rationale**: Standing Invariant 2 — *never silence a check to make it pass*. Lowering a floor is
structurally identical to silencing unless each step is licensed by a genuine target removal landing
in the same commit. The module's own comment at `:275-278` records this discipline and the reason
(one-target slack). Measured consequence of batching: removing both `training.*` targets in one
commit fires **two** asserts — `:287` at `len 6 < 7` and `:306` on `unmatched=['training']`.

`UNGOVERNED_ROOT = ()` at `:126` in Commit 3 is a **sanctioned terminal state**, not a dodge:
`:302-304` pre-authorises it in so many words ("if `training/` is ever promoted INTO the package,
removing the name is the correct edit and must not leave an assert that can never again be
satisfied").

**Honest caveat that belongs in Commit 3's body**: `UNGOVERNED_ROOT = ()` records "no **AST-visible**
root-shape imports", not "no governed → `training/` edges". A surviving invisible one was measured:
`src/snn_data_lake/reasoning/eprop_lm_serve.py:261-264` does `sys.path.insert` then
`import eprop_lm as trainer` — a bare-name import `_targets()` cannot see.

---

## D-7 — The kernel serves TWO consumers, not four

**Decision**: scope the kernel to `eprop_nmnist` + `eprop_shd`. Document the limit in the contract.

**Rationale**: the 4-arg `_SpikeFn` returns `grad_out * psi, None, None, None`
(`eprop_nmnist.py:363`, verified) — **no gradient to the threshold argument**. Harmless where `v_th`
is a float constant, which is true for both intended consumers. NOT harmless for
`training/eprop_lm_alif.py:103`, whose threshold is a live adapted tensor: measured, the 1-arg form
gives `||d spikes / d a|| = 1.0727` and the 4-arg form gives **`None`** **[F]**.

Unifying the remaining in-tree surrogates is NG-002, a separate lane owing its own A/B. Verified
in-tree census (excluding `vendor/`): `cortex/eprop_nmnist.py:352`, `training/eprop_shd.py:344`,
`reasoning/rsnn_delays_torch.py:91`, `training/eprop_lm.py:78`, `training/eprop_lm_alif.py:103`,
`training/ar_spiking_lm.py:74` (`_FastSigmoidSpike`), `training/surrogate_snn.py:109`
(`_FastSigmoid`) — 7 `autograd.Function` surrogates in 2 families.

**The trap this avoids**: advertising the kernel as "the one surrogate for every e-prop trainer"
would be true-sounding and false, and would silently zero the ALIF adaptation gradient the day
someone adopted it — while reddening nothing, because that module's cosine assert stays above 0.5
**[F]**.

---

## D-8 — `reasoning/eprop.py` is not a third copy (NG-005)

**Decision**: out of scope, and *not* because it is small — because it is a different thing.

**Measured**: 378 lines, pure stdlib (`math`, `zip`), **no torch, no `autograd`**. Shares only four
generic names with `eprop_nmnist` (`__init__`, `_cosine`, `forward`, `main`). Its `_cosine`
(L282-286, `math.sqrt`/`zip`) is a **different implementation** from `eprop_nmnist`'s (L390-395,
`torch.norm`/`@`) — different substrate, not duplication.

It is the verification anchor POLICIES.md cites for the whole e-prop invariant: `:32-40` states the
gold-standard check and the `> 0.5` bar, and `grad_bptt` at `:196` documents that it *"shares NO code
with grad_eprop"*.

**Flagged, not folded in**: that anchor sits in the ungoverned `reasoning/` 109, so the invariant's
own reference implementation is outside the gates. That is a genuine P4c item and it is somebody's
lane — just not this one.

---

## Open questions carried into Phase 1

None blocking. Two are scheduled as pre-commit measurements rather than resolved here, because
resolving them by reading would be exactly the static-over-runtime inversion Standing Invariant 6
forbids:

1. Does `python -m snn_data_lake.cortex.eprop_shd` reach its self-check after the move? It gates on
   `smoke = args.smoke or len(sys.argv) == 1` (`:920` **[F]**). Under `-m` it should hold — verify,
   do not assume.
2. Does `eprop_shd.py` contain depth-counted path logic that a move silently retargets?
   `surrogate_snn.py:337-366` has exactly that shape; `eprop_shd`'s post-move path behaviour was
   measured by nobody.
