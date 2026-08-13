# Contract — `snn_data_lake.cortex.eprop_kernel`

**Status**: to be created in Commit 4. This document is the contract it must satisfy.

**One-line**: the single e-prop training kernel for the project's **torch, fixed-threshold** spiking
encoders. It is not a general surrogate library, and the limit is part of the contract.

---

## Public surface

### Constants

| name | value | note |
|---|---|---|
| `DEFAULT_ALPHA` | `0.9` | membrane decay |
| `DEFAULT_V_TH` | `0.6` | **fixed** threshold — see *Scope limit* |
| `DEFAULT_GAMMA` | `0.5` | surrogate width |
| `DEFAULT_DAMP` | `0.3` | surrogate damping |
| `DEFAULT_KAPPA` | `0.9` | eligibility-trace decay |
| `EPROP_BPTT_COSINE_FLOOR` | `0.5` | the e-prop invariant's bar, from POLICIES.md |

Consumers MUST import `EPROP_BPTT_COSINE_FLOOR` rather than hand-writing `0.5` (FR-003).

### Functions

```
_SpikeFn(v, v_th, gamma, damp)  ->  spikes            # autograd.Function, triangular surrogate
bptt_grads(net, xs, tgt)        ->  grads             # INDEPENDENT reverse-mode reference
_cosine(a, b)                   ->  float             # torch comparator, flattens both sides
_apply_grads(params, grads, ...) -> None              # Adam step, in place
evaluate(net, loader, ...)      ->  accuracy
```

### Classes

```
EpropLIFBase          # psi, offdiag_mask, forward, flatten_frames
  - subclass MUST define: n_in, n_rec, num_classes, T  (the per-modality geometry)
  - subclass MUST define: eprop_grads                  (the per-modality gradient path)
EpropEncoderBase      # __init__ only; subclass defines encode / classify / load / config
```

### Import direction (FR-005)

`eprop_kernel` imports **nothing** from `eprop_nmnist` or `eprop_shd`. The dependency is strictly
`consumer -> kernel`. A back-import is a cycle and a contract violation.

---

## Scope limit — this contract binds TWO consumers, not four

`_SpikeFn.backward` returns `grad_out * psi, None, None, None`: **no gradient flows to the threshold
argument.**

That is correct and harmless for both intended consumers, whose `v_th` is a float constant. It is
**wrong** for a consumer with an adapted (tensor) threshold. Measured on
`training/eprop_lm_alif.py:103`, whose threshold is `A = V_TH + BETA*a`:

```
1-arg form (eprop_lm_alif.py:103):  ||d spikes / d a|| = 1.072687029838562
4-arg form (this kernel):           ||d spikes / d a|| = None
```

**Therefore**: this module MUST NOT be documented as "the one surrogate for every e-prop trainer".
Adopting it in `eprop_lm_alif` would silently zero the ALIF adaptation gradient **while reddening
nothing** — that module's own cosine assert stays above 0.5 **[F]**. Widening the contract requires
a backward that returns a real `v_th` gradient, and that is NG-002's lane with its own A/B.

---

## Self-check contract (`__main__`)

Runs under `python -m snn_data_lake.cortex.eprop_kernel` via `run_selfchecks.py` TORCH_GATED.

**MUST**:

1. Build a **minimal concrete subclass inline** — importing a consumer would be a cycle (FR-005).
2. Assert `_cosine(eprop_grads, bptt_grads) > EPROP_BPTT_COSINE_FLOOR`.
3. Print the actual cosine (ASCII only — Windows cp1252).
4. Be deterministic: `random.Random(seed)` per instance, never module-level import-time seeding.

**P8 guard (1) — the reachable RED (FR-019, SC-010).** A criterion that cannot fail is deleted, not
weakened. The red for this check is a **sign-flipped learning signal**: negate the broadcast error
term feeding the eligibility product, and the e-prop gradient anti-aligns with the BPTT reference,
driving the cosine negative and below the floor. This MUST be demonstrated once and the observed
value recorded in the commit body. If it turns out no input reddens the check, the assert is
**removed**, not softened — and that outcome is reported, not buried.

**P8 guard (2) — a control that is not the method itself.** `bptt_grads` is the control, and it is
a genuine one for the *gradient* claim: it back-propagates adjoints including the recurrent
off-diagonal and reset terms that e-prop deliberately drops, so it shares no code with the
eligibility computation.

**P8 guard (4) — disclosure, stated because it is the thing most likely to be glossed.** The control
is independent in the *gradient* path but **not** in the *forward* path: both `eprop_grads` and
`bptt_grads` consume the same `forward`, therefore the same `flatten_frames`. A defect introduced
upstream of the split moves **both** sides together and the cosine stays high. **This check
therefore cannot verify the `flatten_frames` collapse (R-001), and must never be cited as if it
does.** The verification for that is the bit-exact pin in [quickstart.md](../quickstart.md) C4-1/C4-2.

---

## P3 charge evidence (not a claimed consumer)

Reachability is not charge. The kernel's charge is structural and is demonstrated rather than
asserted: **neuter the kernel import and both consumers fail to import at all**, because the kernel
holds their spike function, their gradient reference and their base class. There is no
"imported-but-inert" state available to this module — which is the honest form of P3 satisfaction,
as opposed to wiring a dead consumer to make a gate go green.

Record the neuter result in the Commit 4 body:

```powershell
# expect: BOTH fail with ModuleNotFoundError / ImportError
.venv-lake\Scripts\python.exe -c "import sys; sys.modules['snn_data_lake.cortex.eprop_kernel']=None; import snn_data_lake.cortex.eprop_nmnist"
.venv-lake\Scripts\python.exe -c "import sys; sys.modules['snn_data_lake.cortex.eprop_kernel']=None; import snn_data_lake.cortex.eprop_shd"
```

---

## Registration contract

Landing the module without both registrations reddens gates (see data-model.md §4):

1. `src/snn_data_lake/__init__.py` → `TORCH_TIER["cortex"]` — else `cortex` becomes a new p4c zone
   with implicit allowance 0 → **RED**.
2. `run_selfchecks.py` → TORCH_GATED, near the existing `cortex.eprop_nmnist` entry.

Both land in the same commit as the file.
