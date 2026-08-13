# FR-018 — Unique-capability audit of `training/surrogate_snn.py`

**Task**: T001–T005. **Gates**: the Commit 1 deletion. **Measured at** HEAD `47555a74`.

P2's honesty guard: *"any unique capability of the retired mechanism ported first."* The A/B does
not discharge this — an A/B compares the two mechanisms **on the job they share** and says nothing
about capability only one of them has. Every definition below therefore gets one of three
dispositions with its evidence: **COVERED** / **PORT** / **DROP-with-reason**.

Census (T001): `surrogate_snn.py` holds **19** definitions absent from
`src/snn_data_lake/cortex/eprop_nmnist.py`. Count reproduced by AST diff.

---

## Verdict summary

| disposition | count |
|---|---|
| COVERED by the surviving mechanism | 9 |
| **PORT** (must land before deletion) | **3** |
| DROP with stated loss | 7 |

**LANE STATUS: CLEAR — conditional on the 3 PORTs landing in or before Commit 1.**
No BLOCKED verdict. One DROP carries a real, stated capability loss (T003).

---

## T004 — `_heaviside_no_surrogate` → **PORT** (and it improves the kernel)

**This is the most valuable thing in the module and it must not die with it.**

```python
def _heaviside_no_surrogate(x):
    """A TRUE Heaviside: no custom backward, so it detaches the graph. Negative control only."""
    return (x > 0).float()
```

Live as a **negative control** at `surrogate_snn.py:609-624`: it swaps the surrogate out
(`lif.spike_grad = _heaviside_no_surrogate`) and asserts the measured gradient collapses — *"the
same measurement must collapse. If it does not, C2 is not..."*. Its harness `_fwd_bwd` (`:513-525`)
checks `loss.requires_grad` **rather than assuming it**, precisely so that "no gradient" arrives as
a **measurement** and not as a swallowed exception.

**Nothing else in the tree provides this.** Swept every Heaviside site: `rsnn_delays_torch.py:92`,
`ar_spiking_lm.py:75`, `eprop_lm.py:79`, `eprop_lm_alif.py:104`, `surrogate_snn.py:110` are all
*Heaviside-forward-with-surrogate-backward* — that is **the method**, not a control on it.
`cortex/rsnn_encoder.py:32,140` are hard-Heaviside forward paths with no gradient story at all.

This is exactly the asset P8 guard (2) names: *"a control that is not the method itself."*

**Disposition: PORT into `cortex/eprop_kernel.py`** as the kernel's negative control, alongside
`_first_layer_grad` and `_fwd_bwd` adapted to the kernel's LIF base.

**Consequence for the design — this is a genuine improvement, not just preservation.** FR-019 owes
the kernel self-check a *reachable RED*. The plan proposed sign-flipping the broadcast error term.
The surrogate-off swap is a **second, independent** red and a better-attested one, because it is a
control this repo already runs and trusts. Recommend the kernel self-check carry **both**:
sign-flip (tests the learning signal) and surrogate-off (tests that the surrogate is load-bearing).

*This is what P2 MERGE is supposed to look like: the retiring mechanism's best asset is absorbed by
the survivor rather than discarded with it.*

---

## T003 — `SpikingConvNet` → **DROP, with a real and stated capability loss**

**What it is**: `Conv2d(in_ch,12,5) → MaxPool2d(2) → LIF → Conv2d(12,32,5) → MaxPool2d(2) → LIF →
Linear(flat,hidden) → LIF → Linear(hidden,num_classes) → LIF` (`:179-270`). A surrogate-gradient
**convolutional** spiking network with spatial pooling.

**What survives**: `cortex/eprop_nmnist.py`'s `EpropNet` — a **recurrent fully-connected** LIF. No
convolution, no spatial structure.

**WHAT THE TREE LOSES, stated plainly as the disposition requires**: after this deletion the
project has **no surrogate-gradient convolutional spiking network**. Verified — `:199` and `:202`
are the only `nn.Conv2d` sites in the tree outside the `ng_snn` subtrees, and those use
`HebbConv2d` (`research/ng_snn/softhebb_mnist.py:56`, `reasoning/ng_snn/softhebb_mnist.py:51`), a
**SoftHebb Hebbian** convolution — a different learning rule, not a gradient-trained spiking CNN.
Those subtrees are also duplicated and ungoverned (NG-003).

**Why DROP is nonetheless the correct call, on measurement rather than convenience**:

1. The A/B is the relevant evidence *for this task*: on N-MNIST (34×34 event frames — a task where
   a convolutional spatial prior should help if it helps anywhere), the recurrent FC encoder ties
   it, **92.97% vs 92.58%, p = 1.0000, at 0.56× the parameters**. Convolution is not buying
   measurable accuracy here.
2. The contracted capability is the **encoder interface** (`encode` / `classify` / `load`), and
   `EpropEncoder` provides it — `eprop_nmnist.py:411` documents itself as mirroring
   `SurrogateEncoder`. Architecture is *how* the job was done, not a second job.
3. Standing Invariant 8 — e-prop, not backprop — means a BPTT-trained architecture does not get
   preserved on architectural novelty alone.

**Recorded as a real loss, not waved through.** If the project later wants convolutional spiking
perception under a local rule, that is a **new build** (P0/P8 territory: a conv e-prop encoder),
not a resurrection of this file. The loss is stated here so that decision is made knowingly rather
than discovered later.

**FLAGGED FOR OWNER**: this is the one disposition in the audit that destroys a capability rather
than relocating it. It is reversible only via `git revert`.

---

## T005 — the out-of-tree referents

| def | referents | disposition |
|---|---|---|
| `ng_snn_data_file` (`:364-366`) | `research/ng_snn/data.py:19`, `research/README.md:54` | **COVERED** — `eprop_nmnist.py` provides `_ng_snn_dir` + `_load_data_module` for the same job. Both referents are **prose**, not imports; update them in Commit 1 (T024). |
| `_repo_root` (`:337-361`) | `research/neuromorphic_torch/neuro_chip_gpu.py:52` | **COVERED** — that referent is a **comment** citing it as a *pattern* ("Same reasoning as..."), not an import. `eprop_nmnist._ng_snn_dir` carries the same walk. Re-point the comment (T024). |

Neither is an import edge, so neither blocks. Both are citation-rot sites that FR-020 covers.

---

## Full disposition table (all 19)

| # | definition | lines | disposition | basis |
|---|---|---|---|---|
| 1 | `SurrogateEncoder` | 277-331 | COVERED | `EpropEncoder` (`eprop_nmnist.py:415`) — same `encode`/`classify`/`load` interface, documented as mirroring it |
| 2 | `SpikingConvNet` | 179-270 | **DROP** | T003 — real loss, stated above |
| 3 | `_flat_dim` | 212-218 | DROP | conv-shape helper; dies with #2, meaningless without it |
| 4 | `lif_layers` | 220-221 | DROP | accessor over #2's four snntorch LIFs |
| 5 | `assert_supported_reset` | 223-234 | DROP | guards a **snntorch** `reset_mechanism_val`; the kernel's LIF base is hand-rolled and has no snntorch reset field to mismatch. The guard has no referent after #2 dies |
| 6 | `LeakyLIF` | 135-171 | COVERED | `EpropLIFBase` in the kernel fills the LIF role. Different substrate (hand-rolled vs snntorch), same function |
| 7 | `init_leaky` | 161-163 | COVERED | `LeakyLIF` state init; subsumed by #6 |
| 8 | `_FastSigmoid` | 109-125 | COVERED | kernel's `_SpikeFn` is the surviving surrogate. **Noted**: it is *triangular*, not fast-sigmoid — a different surrogate shape, deliberately, since the triangular form is what both e-prop consumers already use |
| 9 | `_fast_sigmoid` | 128-132 | COVERED | wrapper over #8 |
| 10 | `inner` | 129-130 | COVERED | closure inside #9 |
| 11 | **`_heaviside_no_surrogate`** | 503-505 | **PORT** | T004 — the tree's only surrogate-off control |
| 12 | **`_first_layer_grad`** | 508-510 | **PORT** | the control's readout; adapt `net.conv1.weight.grad` → the kernel base's first parameter |
| 13 | **`_fwd_bwd`** | 513-525 | **PORT** | the control's harness; its `loss.requires_grad` check is the reason "no gradient" is a measurement rather than an exception |
| 14 | `_synthetic_frames` | 528-545 | COVERED | **Corrected during T006.** Initially marked PORT-to-pin-harness. It is not needed: *both* surviving consumers already carry their own `_make_synthetic` (`eprop_nmnist.py:642`, `eprop_shd.py:641`) with per-modality signatures, and the pin harness uses those. Recorded rather than silently amended — the first verdict was written before the harness was built |
| 15 | `_prepare` | 306-316 | COVERED | `eprop_nmnist._prep` does the encoder-side input prep |
| 16 | `_repo_root` | 337-361 | COVERED | T005 — `_ng_snn_dir` carries the same walk |
| 17 | `ng_snn_data_file` | 364-366 | COVERED | T005 |
| 18 | `_self_check` | 548-568 | DROP | the module's own self-check; dies with the module. Its **negative-control content is preserved** via #11–#13 |
| 19 | `_self_check_body` | 571-689 | DROP | as #18 |

---

## What this audit changes in the lane

1. **Commit 1 acquires a prerequisite.** The 4 PORTs (#11–#14) must exist before
   `training/surrogate_snn.py` is deleted. Options: land them in Commit 0's pin harness (#14) and
   carry #11–#13 forward into Commit 4's kernel, **or** stage them into a holding module. The
   simplest honest route: **#14 into the pin harness now; #11–#13 into the kernel at Commit 4, and
   Commit 1's message records that the control is in flight with its landing commit named.**
   *That is a deliberate, disclosed temporary gap and it must be written down, not left implicit.*
2. **FR-019 gets a better red.** The kernel self-check should carry the surrogate-off control as a
   second reachable red beside the sign-flip.
3. **One capability is genuinely destroyed** (T003) and the owner should see that before Commit 1.

## Honest limits of this audit

- Dispositions are based on **reading and tree-wide capability search**, not on runtime A/B per
  definition. For 15 of 19 that is proportionate (helpers, closures, accessors). For `SpikingConvNet`
  the runtime evidence that matters — the accuracy A/B — exists and is cited.
- The A/B behind #2's DROP is **[F]** (n=256, from the measurement pass) and was not re-run here.
  It is underpowered and is recorded as a **tie**, never as a win. The DROP does not rest on it
  alone; it rests on P2 + Standing Invariant 8 + the parameter cost, with the tie as supporting
  evidence.
- `assert_supported_reset` (#5) is dropped on the argument that its referent dies with #2. If
  `LeakyLIF` were ported instead of covered, this verdict would change.
