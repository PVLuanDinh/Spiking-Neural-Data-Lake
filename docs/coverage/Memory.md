# Coverage evidence -- Memory

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| recall_drive_source | 20 | src/snn_data_lake/cortex/brain.py:1532 | recall_60.txt | fable-5 / tools/charge_probe.py recall 60 | 2026-07-29 |
| chat_memory_write_only | 25 | src/snn_data_lake/substrate/chat_memory.py:143 | - | auditor:AUDIT_BACKLOG.md:219 | 2026-07-28 |
| chat_memory_uncalled_method | 15 | src/snn_data_lake/substrate/chat_memory.py:237 | - | auditor:AUDIT_BACKLOG.md:218 | 2026-07-28 |

- `recall_drive_source` is the strongest negative in this set and the reason the exit-code rule is
  worth having. The source is genuinely **eligible** -- 14 learned (recall, action) pairs, so it fired
  -- and it still produced 0/60 act-hold flips and 0/60 trajectory changes. CONSUMED-BUT-NOT-DECISIVE
  (exit 1) is the rubric's band 21-40 in one word, and no row citing this artifact can claim more.
