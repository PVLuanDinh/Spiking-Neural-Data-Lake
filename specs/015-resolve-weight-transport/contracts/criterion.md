# Contract — The Repaired Alignment Criterion

The one interface this feature exposes to the rest of the project: a number the e-prop invariant is
judged against. Fixed here **before** any candidate rule is run.

## What it compares

```
cos( dh_rule , dh_transport )
```

Both computed on **the same input** and **the same pre-update weights**, so they differ in the
feedback vector alone. `dh` is the hidden-layer update direction — the thing the invariant is
actually about, and the thing `MlpQ.update` actually applies.

Read **at every training pass**, not once at the end.

## What it must satisfy before it may judge anything

| # | requirement | how it is shown |
|---|---|---|
| 1 | scores the transport gradient against itself at **1.0** | identity; if not, the harness is wrong |
| 2 | scores the **permutation null** low | same magnitude, same element multiset, shuffled direction |
| 3 | has a **demonstrated reachable red** | construct the failing input and record the output |
| 4 | its degeneracy is **disclosed** | see below |

**A criterion failing (1) or (2) is deleted, not weakened.** If it cannot separate the permutation
null from the real gradient, it is measuring magnitude and this feature reports that and stops.

## Disclosed degeneracy

Under `MlpQ`'s arithmetic this remains a **monotone function of `cos(W2[a], B[a])`** — `err` is a
shared scalar that divides out and the relu mask zeroes the same coordinates on both sides.

**This is stated, not smoothed.** The repair is not a new formula. It is:

1. reading the quantity **per epoch**, so alignment development is visible rather than an endpoint;
2. pairing it with a feedback vector that **can move**.

The old criterion's defect was that its subject was frozen by construction, so no alignment rule
could ever pass. A rule whose feedback converges can. Anyone reading this contract should be able to
predict that a fixed-random draw still scores near zero here — that is correct behaviour, not a
residual bug.

## Threshold

`|cos| > 0.5`, the project's existing cross-cutting bar. **This feature does not amend it.** If a
rule learns well while failing it, that is an owner decision with the numbers attached (non-goal 4),
not something this lane may resolve.

## Reporting shape

- per `(variant, bseed, epoch)`, never a single mean — the module already records that this quantity
  "is a property of the DRAW"
- 8 draws reported as a **range**
- held-out accuracy on **both** fixtures beside every cosine, so *aligns-but-stops-learning* and
  *learns-but-does-not-align* are both visible

## What this criterion may NOT be used for

- Judging `cortex/predictive_coding.py`. Different mechanism, different objective; a result here does
  not transfer in either direction (R-004).
- Justifying an amendment to the cross-cutting invariant.
- Deciding adoption on accuracy alone — accuracy is a co-report, not the criterion.
