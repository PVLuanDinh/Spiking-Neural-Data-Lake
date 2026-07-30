# Coverage evidence -- SpatialTemporal

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| category_table_never_committed | UNSCORED | - | - | auditor:SpatialTemporal (only a category total, 36, reached COVERAGE_BASELINE.md:18) | 2026-07-28 |

The 2026-07-28 sweep ran one auditor per category and each was asked for "the constituent faculties
with `file:line` and the band each one earned" (COVERAGE_RUBRIC.md:85-91). Those per-faculty tables
were never committed -- only the category integer survived, in COVERAGE_BASELINE.md's result table.
The mean's operands are therefore unavailable, so the category is UNSCORED rather than 36. This row
exists so the category stays in the denominator: silent omission is the inflation route.
