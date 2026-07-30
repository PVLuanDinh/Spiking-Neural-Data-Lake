# Coverage evidence -- Consciousness

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| workspace_reentry_key | 0 | src/snn_data_lake/thalamus/reentrant_workspace.py:271;src/snn_data_lake/thalamus/reentrant_workspace.py:289 | - | auditor:AUDIT_BACKLOG.md:264 | 2026-07-28 |
| ignition_novelty_source | 20 | src/snn_data_lake/cortex/brain.py:1532 | novelty_60.txt | fable-5 / tools/charge_probe.py novelty 60 | 2026-07-29 |
| global_availability | UNSCORED | src/snn_data_lake/thalamus/reentrant_workspace.py:143 | - | auditor:refutation (COVERAGE_BASELINE.md:15) | 2026-07-28 |
| pci_over_nights | UNSCORED | - | - | auditor:refutation (COVERAGE_BASELINE.md:15) -- no file:line was committed | 2026-07-28 |

- `workspace_reentry_key` scores **0** under rubric rule 4, which names this exact shape ("a key space
  that never collides"): the write key at `:271` and the lookup key at `:289` do not match, so the
  re-entry path learns to 1.0 and delivers a constant. A 1.9x effect was measured when the key spaces
  were made to coincide -- that is a *potential*, and the rubric scores what runs.
- `ignition_novelty_source` scores **20**, not 0: its probe returned UNKNOWN (exit 2), because an
  undirected source has no value table and eligibility cannot be established from outside. UNKNOWN is
  not evidence of inertness, and the artifact says so in its own words.
- `pci_over_nights` is the honest shape of an unbacked claim: COVERAGE_BASELINE.md asserts PCI took
  one value over 25 nights, and committed no citation. It stays in the denominator as UNSCORED.
