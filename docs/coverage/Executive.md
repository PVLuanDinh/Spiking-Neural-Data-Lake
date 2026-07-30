# Coverage evidence -- Executive

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| drive_arbitration | 61 | src/snn_data_lake/cortex/brain.py:1532 | drive_60.txt | fable-5 / tools/charge_probe.py drive 60 | 2026-07-29 |
| place_drive_source | 20 | src/snn_data_lake/cortex/brain.py:1532 | place_60.txt | fable-5 / tools/charge_probe.py place 60 | 2026-07-29 |
| metacognitive_arbitration | 35 | src/snn_data_lake/cortex/brain.py:261 | - | auditor:AUDIT_BACKLOG.md:90 (M12) | 2026-07-28 |
| pfc_plan_method | 15 | src/snn_data_lake/cortex/prefrontal_executive.py:333 | - | auditor:AUDIT_BACKLOG.md:218 | 2026-07-28 |
| arc_library_learning | 0 | src/snn_data_lake/reasoning/arc_solver.py:880 | - | auditor:AUDIT_BACKLOG.md:106 (M28) | 2026-07-28 |

- `drive_arbitration` is the **only** band-41+ claim in this prototype that the instrument accepts,
  and it is accepted because `drive_60.txt` records exit 0 with 22/60 trajectory changes -- the same
  control COVERAGE_BASELINE.md:90 cites to prove the harness is not degenerate. It reproduced exactly.
- `place_drive_source` scores **20** because its probe returned NOT-ELIGIBLE (exit 2, empty value
  table). The artifact is attached anyway: an honest negative is evidence, and attaching it is what
  lets the scorer refuse a later attempt to re-badge the same run as a charge.
- `arc_library_learning` is **0** under rubric rule 4: `ArcLibrary.learn()` is guarded by
  `library is not None` and every scored path passes `library=None`, so the reported ARC pass@2 was
  measured with library learning off.
