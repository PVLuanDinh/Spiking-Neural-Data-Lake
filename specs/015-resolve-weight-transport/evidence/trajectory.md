# Trajectory -- 015 discriminating measurement (T019-T022)

Fixtures are the module own: 400 train / 300 held-out, hidden=32, lr=0.1, 30 passes,
model seed 0, B from random.Random(bseed) as uniform(-1,1)/sqrt(hidden), bseed 1..8.
Rows: 960 = 4 variants x 8 draws x 30 epochs (asserted).
CONTROL: transport reads 1.000000000000 at every epoch, else the probe is broken.

## Final epoch, ranges over 8 draws (never a mean -- the module records this quantity
## is a property of the DRAW)

| variant | final abs(cos) range | draws over 0.5 | acc XOR | acc linear |
|---|---|---|---|---|
| transport | [1.0000, 1.0000] | **8 of 8** | [0.993, 0.993] | [0.983, 0.983] |
| fixed_random | [0.1604, 0.3379] | **0 of 8** | [0.980, 0.997] | [0.990, 0.993] |
| converging | [0.8785, 0.9121] | **8 of 8** | [0.990, 0.993] | [0.983, 0.987] |
| permutation_null | [0.0000, 0.0000] | **0 of 8** | [0.467, 0.467] | [0.520, 0.980] |

## Trajectory, mean abs(cos) across draws

| epoch | converging | fixed_random |
|---|---|---|
| 1 | 0.9231 | 0.1823 |
| 2 | 0.9372 | 0.1686 |
| 3 | 0.9378 | 0.1589 |
| 5 | 0.9325 | 0.1440 |
| 8 | 0.9238 | 0.1474 |
| 10 | 0.9193 | 0.1613 |
| 15 | 0.9126 | 0.1999 |
| 20 | 0.9082 | 0.2324 |
| 25 | 0.9029 | 0.2416 |
| 30 | 0.8985 | 0.2468 |

## Per-draw attribution at pass 30 (T027)

| bseed | converging abs(cos) | over bar | acc XOR | acc linear | fixed_random abs(cos) |
|---|---|---|---|---|---|
| 1 | 0.8872 | YES | 0.993 | 0.987 | 0.2292 |
| 2 | 0.9110 | YES | 0.990 | 0.987 | 0.2204 |
| 3 | 0.9019 | YES | 0.993 | 0.983 | 0.2454 |
| 4 | 0.9003 | YES | 0.993 | 0.987 | 0.2172 |
| 5 | 0.8785 | YES | 0.993 | 0.983 | 0.1604 |
| 6 | 0.9069 | YES | 0.993 | 0.983 | 0.3379 |
| 7 | 0.9121 | YES | 0.993 | 0.983 | 0.3318 |
| 8 | 0.8898 | YES | 0.993 | 0.987 | 0.2320 |

## T018 -- the decay term, MEASURED (P1)

| decay | final abs(cos) range | over bar | acc XOR |
|---|---|---|---|
| **0.0 decay-free** | **[0.8785, 0.9121]** | **8 of 8** | [0.990, 0.993] |
| lr/10 = 0.01 | [0.2176, 0.8296] | 7 of 8 | [0.983, 0.997] |
| lr = 0.1 | [0.4289, 0.4880] | **0 of 8** | [0.970, 0.977] |

DECIDED: decay-free. Decay HURTS -- at lr it destroys alignment entirely (0 of 8).
P1 is satisfied by OMISSION, not derivation: there is no new constant.
