# Local SHD pilot ledger

Completed November 4, 2026, America/Chicago. Owner authorized Torch repair, SHD preparation and local real-data pilot.

- Completed: 16 runs across four previously exposed development seeds, 178,332 training exposures over the same 6,369 unique training recordings. Development: 1,787 recordings from speakers 10 and 11; all 20 classes present in each split. No official test data downloaded or read.
- Source: pinned upstream commit `c4c10b285f1741aa798d5ee9c5296ae371fabecf`. No upstream source edits and no Git branch created. All 483 original export files matched pinned GitHub blob hashes. The five imported files were reverified after cache recovery; every run logs matching imported-source hashes.
- Runtime repaired: CPython 3.14.6t, Torch 2.15.0 CPU, NumPy 2.5.4, GIL off, one Torch thread. Final three fresh-process tensor checks passed in 0.76–1.11 seconds. Working bytecode cache and exact recovery inputs now persist outside temporary storage.
- Independent final analysis verified all 144 prediction checkpoints, raw-logit argmaxes, IDs/targets, confusion and task scores, matching paired initial weights and streams, and equal replay/repeat updates and exposures. Durable scoring metadata supports rescoring without the temporary dataset cache.
- Completed-cohort experiment-worker wall time: 34.1 minutes, excluding setup, recovery, interrupted work and calendar downtime. Maximum completed-worker RSS: 834.5 MiB. No algorithm failures. One infrastructure interruption is retained with 5,736 last-recorded exposures and nine prediction checkpoints; its same-seed retry reproduced all shared arrays and model hashes bit for bit.

| Arm | Final macro accuracy (%) | Sample SD (pp), n=4 |
|---|---:|---:|
| joint | 41.16 | 3.20 |
| sequential | 6.22 | 1.83 |
| repeat | 7.21 | 2.30 |
| replay | 12.68 | 2.57 |

Replay minus matched repeated-current updates: 5.47 pp mean, 0.46 pp sample SD. This is descriptive variation across four exposed seeds, not a powered confirmation.

Earlier-class final macro accuracy was 6.03% with replay versus 0% with repeated-current updates; latest-class accuracy was similar (72.50% versus 72.09%). Substantial forgetting remains. The secondary peak-relative forgetting metric and the final retention endpoint answer different questions; the canonical analysis corrects the collector's clipping of later improvements, retaining original summaries unchanged.

Results and figures: `reports/pilot-report.html`, `reports/pilot-results.json`. Protocol: `PROTOCOL.md`. Deviations and infrastructure loss/recovery: `DEVIATIONS.md`. Reproduction: `RUNBOOK.md`. Closing checks: `reports/closing-verification.json`.

The real-data audio component pilot is complete. The full engine still refuses execution with G9_A_UNREADY (retention selection, grown-source/admission, G6 mask/progress and durable final-retention bindings; sequence baselines may also require BETA). No growth, integrated reservoir, recursive self-improvement, official-test benchmark, transformer superiority, or learned-lake persistence was measured or claimed.
