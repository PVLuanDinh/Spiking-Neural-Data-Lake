# DECISION POINT — 015, Phase 5 (T025–T027)

Compared against [research.md](../research.md) **D-3 as written**, not a revised version.

## The comparison

| variant | D-3 predicted | MEASURED | |
|---|---|---|---|
| transport | 1.0 every epoch by identity | **1.000000000000**, all 240 rows | ✓ |
| fixed-random `B` | flat, no trend, within ±0.35; \|cos\| < 0.5 on **8 of 8** | [0.1604, 0.3379]; **0 of 8** over bar | ✓ |
| converging `B` | \|cos\| > 0.5 on **≥ 6 of 8** by pass 30 | [0.8785, 0.9121]; **8 of 8** | ✓ exceeds |
| permutation null | flat near 0; \|cos\| < 0.5 on 8 of 8 | **0.0000**; 0 of 8 | ✓ |

**Accuracy** (D-3: clears > 0.85 held-out non-linear and > 0.30 over the linear control on every
draw, within ~±0.02 of the FA twin's 0.990 / 0.993):

- converging XOR **[0.990, 0.993]** — clears 0.85; within 0.003 of the FA twin
- converging linear **[0.983, 0.987]** — within 0.010 of the FA twin
- margin over the linear control: 0.990 − 0.467 = **0.523**, far above 0.30

**Neither refutation condition fired.** D-3 said adoption is refuted by failing 6-of-8, or by
clearing it while dropping below the shipped accuracy gates on any draw. Neither happened.

## Per-draw attribution (T027)

| bseed | converging \|cos\| | over bar | acc XOR | acc linear | fixed-random \|cos\| |
|---|---|---|---|---|---|
| 1 | 0.8872 | YES | 0.993 | 0.987 | 0.2292 |
| 2 | 0.9110 | YES | 0.990 | 0.987 | 0.2204 |
| 3 | 0.9019 | YES | 0.993 | 0.983 | 0.2454 |
| 4 | 0.9003 | YES | 0.993 | 0.987 | 0.2172 |
| 5 | 0.8785 | YES | 0.993 | 0.983 | 0.1604 |
| 6 | 0.9069 | YES | 0.993 | 0.983 | 0.3379 |
| 7 | 0.9121 | YES | 0.993 | 0.983 | 0.3318 |
| 8 | 0.8898 | YES | 0.993 | 0.987 | 0.2320 |

**8 of 8, every draw, with its accuracy beside it.** Against the noise floor established in C2
(a single draw clears 0.5 by chance 3.8% of the time; 6-of-8 by chance is 7.9e-08), this is not a
tail event — and the values sit at ~0.90, not marginally over the bar.

## WHERE THE PREDICTION WAS WRONG, recorded rather than smoothed

D-3 predicted converging feedback would **"rise monotonically in trend from ~0"**. It does not.

| epoch | converging | fixed-random |
|---|---|---|
| 1 | **0.9231** | 0.1823 |
| 3 | 0.9378 | 0.1589 |
| 10 | 0.9193 | 0.1613 |
| 30 | 0.8985 | 0.2468 |

`B` is at **0.92 after a single epoch** and drifts slightly DOWN thereafter. It starts from the same
random draw as `fixed_random` (~0.18) and aligns within the first pass, because epoch 1 applies
`lr*err*h[j]` to `B` on all 800 updates while `W2` receives the identical increment. **Alignment is
established almost immediately and then tracked — not developed over training.**

The prediction's DIRECTION was right; its model of the MECHANISM was wrong. Recording that, because
"as predicted" would have been the easy and false summary.

Second, smaller deviation: `fixed_random` **drifts upward** (0.18 → 0.25). A frozen `B` cannot move,
so this is `W2` rotating slightly toward it. Still nowhere near the bar, and it makes the endpoint
range slightly wider than D-3's ±0.35 guess at the top end (0.3379).

## VERDICT

**GO — Phase 6a (adopt).** The bar and the epoch count came from D-3 and were not renegotiated.

## T018 — the decay term (P1)

| decay | final \|cos\| range | over bar | acc XOR |
|---|---|---|---|
| **0.0 (decay-free)** | **[0.8785, 0.9121]** | **8 of 8** | [0.990, 0.993] |
| lr/10 = 0.01 | [0.2176, 0.8296] | 7 of 8 | [0.983, 0.997] |
| lr = 0.1 | [0.4289, 0.4880] | **0 of 8** | [0.970, 0.977] |

**Decided: decay-free.** Decay actively hurts here — at `lr` it destroys alignment entirely. P1 is
satisfied by **omission**, not derivation: the adopted rule introduces **no new constant at all**.
