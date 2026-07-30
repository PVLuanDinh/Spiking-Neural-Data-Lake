# Coverage evidence -- Perception

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| attention_goal_modulation | 0 | src/snn_data_lake/cortex/learned_attention.py:139;src/snn_data_lake/cortex/brain.py:1080 | - | auditor:Perception (COVERAGE_TO_100.md:61-64) | 2026-07-29 |
| attention_lif_gain | 20 | src/snn_data_lake/cortex/learned_attention.py:64 | - | auditor:AUDIT_BACKLOG.md:91 (M13) | 2026-07-28 |
| eprop_snn_visual_tier | 40 | src/snn_data_lake/cortex/brain.py:545 | - | auditor:Perception (COVERAGE_BASELINE.md:48-53) | 2026-07-28 |
| eprop_snn_audio_tier | 40 | src/snn_data_lake/cortex/brain.py:577 | - | auditor:Perception (COVERAGE_BASELINE.md:48-53) | 2026-07-28 |
| perceived_digit_to_place | 40 | src/snn_data_lake/cortex/brain.py:1282 | - | auditor:re-check (COVERAGE_BASELINE.md:80) | 2026-07-28 |
| multimodal_binding | UNSCORED | src/snn_data_lake/cortex/multimodal_encoder.py:198 | - | auditor:refutation (COVERAGE_BASELINE.md:11) | 2026-07-28 |

Notes on the two zeros and the UNSCORED, because a low band with no reason is not a measurement:

- `attention_goal_modulation` scores **0** under rubric rule 4 (a precondition that never fires).
  `learned_attention.py:91-92` returns `np.zeros(n_bands)` when `goal=None` and `:139` multiplies by
  `(1.0 + alpha_goal * goal_gain)`, so `alpha_goal` is a multiplicative identity; the only live call
  site passes `goal=None` (`brain.py:1080` -- the baseline's `:1068` was inside a docstring).
- `multimodal_binding` is **UNSCORED**, not 0: the auditor's neuter was non-isolating
  (COVERAGE_BASELINE.md:11), so no valid A/B exists in either direction. Unmeasured is not absent.
- The three `40`s are the band ceiling for evidence that is live-but-unprobed. All three are also
  gated on `models/` binaries that are gitignored, so on a fresh clone they are 0 (COVERAGE_BASELINE.md:45-58).
