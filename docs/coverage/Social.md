# Coverage evidence -- Social

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| category_table_never_committed | UNSCORED | - | - | auditor:Social (only a category total, 38, reached COVERAGE_BASELINE.md:17) | 2026-07-28 |
| pragmatics_implicature | 25 | src/snn_data_lake/reasoning/spike_chat_lake.py:997 | - | auditor:correction (COVERAGE_TO_100.md:57-60) | 2026-07-29 |

COVERAGE_BASELINE.md:32 records a faculty in this category scored **80** on a single A/B. No artifact
for it exists in the tree, so under rubric rule 2 it cannot be entered here at all -- an 80 with no
recorded A/B is exactly the claim this instrument refuses. `pragmatics_implicature` is entered at 25
because the baseline's "zero callers" finding (gap 3) was corrected: `spike_chat_lake.py:997-1003`
does call `implicature(text)` and emits a reply candidate, but there is no A/B and no caller in
`cortex/brain.py`, which is band 21-40 by definition.
