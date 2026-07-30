# Coverage evidence -- Language

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| ar_spiking_lm_serve | 0 | src/snn_data_lake/reasoning/ar_spiking_lm_serve.py:53;training/ar_spiking_lm.py:48 | - | auditor:correction (COVERAGE_TO_100.md:70-74) | 2026-07-29 |
| category_table_never_committed | UNSCORED | - | - | auditor:Language (only a category total, 35, reached COVERAGE_BASELINE.md:19) | 2026-07-28 |

- `ar_spiking_lm_serve` scores **0** under rubric rule 4. The trainer exports to
  `models/spiking_lm.npz` (`training/ar_spiking_lm.py:48`) and the server loads
  `models/ar_spiking_lm.npz` (`:53`); neither half can see the other's file, so the load precondition
  never fires. BLiMP's `n/a` was read as an untrained model and is at least partly a path mismatch.
- The category's BLiMP figure (**51.0%** against a 50.0% chance floor, COVERAGE_TO_100.md:24) *is*
  held-out and could back a row -- but there is no recorded artifact for it in the tree, so it cannot
  be entered. That is rubric rule 3 doing its job, not the instrument being unable to see it.
