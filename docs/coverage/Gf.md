# Coverage evidence -- Gf

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| arc_mint_promotion | 0 | src/snn_data_lake/reasoning/arc_mint.py:558 | - | auditor:AUDIT_BACKLOG.md:109 (M31) | 2026-07-28 |
| category_table_never_committed | UNSCORED | - | - | auditor:Gf (only a category total, 34, reached COVERAGE_BASELINE.md:20) | 2026-07-28 |

- `arc_mint_promotion` scores **0**: 1,601 lines of `arc_scene` + `arc_mint` terminate in `mint()`,
  whose only caller is the module's own `main()`. Self-declared, and P3-exempt -- but exempt from the
  dormancy gate is not a band, and rubric rule 4 scores what runs on a live tick.
- ARC-AGI **2.75%** held-out against a 0.5% trivial-solver noise floor (COVERAGE_TO_100.md:26) is real
  held-out evidence and would be admissible as a `heldout` artifact. It is not in the tree as one.
