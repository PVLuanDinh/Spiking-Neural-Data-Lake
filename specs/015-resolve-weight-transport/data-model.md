# Phase 1 Data Model — Resolve the e-prop Weight Transport at `mlp_q`

Shapes read from `MlpQ.update`'s actual arithmetic, not inferred.

---

## E-1 — The three feedback pathways, side by side

All three produce `dh`, the hidden-layer update direction. They differ in ONE vector.

```
err   = target - Q(z, a)                       scalar, identical in all three
pre,h = forward(z)                             identical in all three
mask  = (pre[j] > 0.0)                         relu gate, identical in all three

  TRANSPORT (incumbent, the violation)
      dh[j] = err * W2[a][j] * mask            reads the FORWARD weight, transposed

  FIXED-RANDOM B (vanilla FA, already benchmarked, NOT adopted)
      dh[j] = err * B[a][j] * mask             B drawn once, never moves

  CONVERGING B (Kolen-Pollack, the candidate)
      dh[j] = err * B[a][j] * mask             B receives the SAME increment as W2
      B[a][j] += lr * err * h[j]   (+ decay)   -> B tracks W2 without ever reading it
```

**The whole difference is whether `B` can move, and toward what.** The transport pathway sets
`B := W2` by fiat every step. Vanilla FA freezes `B` at a draw. Kolen-Pollack lets `B` *learn* `W2`
from quantities already present at the synapse — `err` and `h[j]` are both already computed by the
existing read-out step.

**Why this is not a second mechanism (P2/FR-008)**: the candidate REPLACES the feedback vector inside
the one existing update. Survivor count after adoption is 1. A constructor flag selecting between
transport and `B` would be the additive failure.

| field | type | note |
|---|---|---|
| `B` | `list[list[float]]`, shape `(n_actions, hidden)` | same shape as `W2`; init from the module's existing `uniform(-1,1)/sqrt(hidden)` scale — no new scale constant |
| decay | float or absent | **OPEN (P1)**. Either derived from `lr` or omitted with a small init. Decided by measurement, not assumed. See research.md D-1. |
| rng | `random.Random(bseed)` | **per-instance**, never module-level |

---

## E-2 — The criterion, before and after

| | quantity actually compared | can a fixed random `B` pass? | can it separate good from bad? |
|---|---|---|---|
| **before** | `cos(W2[a], B[a])` — `err` and the mask divide out | **no, by construction** at any reasonable width | **no** |
| **after** | `cos(dh_rule, dh_transport)` on the same input and same pre-update weights, read **per epoch** | yes, if `B` converges | **must be demonstrated** — see contracts/criterion.md |

**Disclosed, not hidden**: under this module's arithmetic the new quantity remains a monotone
function of the old one, because `err` is a shared scalar and the mask zeroes the same coordinates on
both sides. **The repair is not a different formula.** It is (a) reading it per-epoch so alignment
DEVELOPMENT is visible, and (b) pairing it with a feedback that can actually move. A criterion whose
subject cannot change was the defect; a criterion whose subject can change is the fix.

---

## E-3 — The trajectory record

One row per `(variant, bseed, epoch)`. This is the artifact the decision consumes.

| field | note |
|---|---|
| `variant` | `transport` \| `fixed_random` \| `converging` \| `permutation_null` |
| `bseed` | 1..8 — the module's existing sweep |
| `epoch` | 1..30 — the module's existing pass count |
| `cosine` | `cos(dh_rule, dh_transport)` at that epoch |
| `acc_nonlinear` | held-out on the XOR fixture |
| `acc_linear` | held-out on the linearly-separable fixture |

**Both accuracies travel with the cosine** so the two failure modes the spec names are visible in one
table: *aligns but stops learning* (cosine rises, accuracy falls) and *learns but does not align*
(accuracy holds, cosine flat). Reporting cosine alone cannot distinguish either.

**Ranges, not central values** (FR-006): 8 draws are reported as a range. A single mean is
unreproducible — the module already records that its cosine "is a property of the DRAW".

---

## E-4 — The permutation null (the control)

`dh_transport` with its coordinates shuffled by a per-instance `random.Random`.

**Identical magnitude, identical element multiset, different direction.** So a criterion that scores
it near the real thing is measuring magnitude, not direction — which is the failure mode being
guarded against.

**Fixed in Phase 1, before any candidate runs** (R-001). It does not depend on which rule wins, so
the criterion cannot be tuned toward a wanted answer. It is a control that is not the method itself,
which is P8 guard (2).

---

## E-5 — What is NOT modelled here

- **`cortex/predictive_coding.py`** — this lane produces its adjudication only, never its code
  (FR-012). No entity above applies to it, and a `mlp_q` result does not transfer to it (R-004).
- **Any lake record** — this lane persists nothing.
- **Any spiking quantity** — `MlpQ` has no membrane, threshold or spike, which is exactly why SFDFA
  does not transfer (research.md D-1).
