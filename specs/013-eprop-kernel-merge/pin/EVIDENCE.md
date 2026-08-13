# Pin evidence — T009 sensitivity, and the R-001 proof

Measured at HEAD `47555a74`, `.venv-lake` (torch 2.11.0+cu128), device cpu, seed 20260812.
`training/eprop_shd.py` has not moved yet (C3 pending); payload records `shd_module`.

---

## 1. Determinism — the pin is stable

`capture.py --tag before` then `--tag recap`, compared: **12 tensors checked, 0 differ.** Exit 0.

A pin that drifts between two identical runs would report the merge as broken every time; it does
not.

---

## 2. T009a — one-element perturbation: SENSITIVE, but incompletely

`capture.py --tag probe --perturb` (shift one input element by 1.0):
**12 checked, 8 differ.** Exit 1.

**Disclosed rather than reported as a clean pass (P8 guard 4):** the 4 that did NOT move are
`nmnist forward.logits_sum`, `nmnist forward.latent_sum`, `nmnist eprop_grads.loss` and
`nmnist eprop_grads.spikes`. Cause: those forward outputs are spike **counts**, and a one-element
nudge is subthreshold — it never flips a spike. So this probe leaves a third of the pinned surface
unexercised, and calling the comparator "proven sensitive" on it alone would overclaim.

That is why T009b exists.

---

## 3. T009b — the ACTUAL defect class: ALL 12 MOVE

A wrong `flatten_frames` does not nudge a value; it changes **which input element lands on which
input unit**. The honest sensitivity test is therefore a permutation of the flattened feature axis,
not an epsilon.

`capture.py --tag reshape --reshape-defect`: **12 checked, 12 differ.** Exit 1.

| module | tensor | max abs delta |
|---|---|---|
| nmnist | `forward.logits_sum` | 36.9682 |
| nmnist | `forward.latent_sum` | 15 |
| nmnist | `eprop_grads.grads.W_in` | 8.43427 |
| nmnist | `eprop_grads.loss` | 1.34628 |
| nmnist | `eprop_grads.spikes` | 0.00489584 |
| nmnist | `bptt_grads.grads.W_in` | 6.86902 |
| shd | `forward.logits_sum` | 538.071 |
| shd | `forward.latent_sum` | 100 |
| shd | `eprop_grads.grads.W_in` | 225.976 |
| shd | `eprop_grads.loss` | 2.16965 |
| shd | `eprop_grads.spikes` | 0.0280508 |
| shd | `bptt_grads.grads.W_in` | 158.234 |

The comparator catches the defect it exists to catch, on every pinned tensor.

---

## 4. THE R-001 PROOF — the cosine assert is blind to this defect

Same reshape defect, measured against the criterion the lane would otherwise have trusted:

| module | cosine CLEAN | cosine WITH the reshape defect | floor | does the cosine catch it? |
|---|---|---|---|---|
| nmnist | 0.995375 | **0.997168** | 0.5 | **NO** |
| shd | 0.994357 | 0.993720 | 0.5 | **NO** |

**Both stay far above the floor while every one of the 12 pinned tensors changed**, one of them by
538. On `nmnist` the cosine did not merely fail to fall — it **rose**, so a reader watching only
that number would conclude the defect had made things slightly *better*.

**Mechanism**: `bptt_grads` computes its reference through the same `forward`, therefore the same
`flatten_frames`. The defect moves both sides of the comparison together, and a cosine between two
identically-displaced vectors is unchanged. A control that compares a mechanism against itself
cannot detect a change in that mechanism.

**Status change**: R-001 was written into the spec as a reasoned risk. It is now a **measured
fact**. Standing Invariant 5 — *"a green gate that structurally cannot see the thing it checks is
worse than no gate, because it manufactures confidence"* — applies literally and by measurement,
not by analogy.

### What this obliges

1. **FR-017's pin is not belt-and-braces. It is the only verifier of the FR-007 collapse.** The
   Commit 4 acceptance step is `compare.py`, and the cosine assert must never be cited for it.
2. The kernel's own self-check (FR-019) still owes a reachable RED, and **the cosine assert cannot
   be the whole of it**. The surrogate-off control ported from `surrogate_snn`
   (audit item #11) matters more than it looked: it reddens on a different axis.
3. This measurement belongs in the Commit 4 body verbatim. It is the reason the lane is safe.

### Reproduce

```powershell
.venv-lake\Scripts\python.exe tests\pin\capture.py --tag before
.venv-lake\Scripts\python.exe tests\pin\capture.py --tag reshape --reshape-defect
.venv-lake\Scripts\python.exe tests\pin\compare.py --a before --b reshape
```

The R-001 cosine measurement was produced by a throwaway script outside the repo (it monkeypatches
an instance and edits nothing); its logic is the `_reshape_defect` helper in `capture.py`.

---

## 5. A gate caught the plan, not just the code

The harness was first written to `specs/013-eprop-kernel-merge/pin/`. `tools/p4c_ratchet.py --assert`
immediately went **RED**:

```
specs: 2 > baseline 0  (+2)
    NEW (candidate): specs/013-eprop-kernel-merge/pin/capture.py
    NEW (candidate): specs/013-eprop-kernel-merge/pin/compare.py
GATE FAIL: the ungoverned count ROSE.
```

P4c counts any `.py` outside the governed tree as an ungoverned **build**, and `specs/` had no
baseline zone, so allowance defaulted to 0. **The plan had not accounted for its own evidence
tooling being P4c-countable.**

The gate also refuses the easy exit in so many words — *"Do NOT run `--update-baseline` to absorb
it: that path refuses a rise by design"* — which is Standing Invariant 2 mechanized.

**Resolution, without touching any gate**: moved to `tests/pin/`. p4c already excludes `tests/**`
as *"the instrument that measures a build, not a build"*, and that is exactly what this harness is.
`tests/**` 62 → 64; total stays **205**. Neither the ratchet nor the baseline was edited.

The alternative — adding a `specs/` exclusion — would have widened a check to admit the artifact
that tripped it. That is the shape Standing Invariant 2 forbids, and it was available and declined.
