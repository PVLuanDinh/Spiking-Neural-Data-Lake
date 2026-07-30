# Coverage evidence -- WholeBrain

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| faculty_charge_registry | 0 | src/snn_data_lake/substrate/region_wiring.py:34 | - | auditor:WholeBrain (COVERAGE_BASELINE.md:82) | 2026-07-28 |
| godel_machine_selfbuild | 0 | src/snn_data_lake/reasoning/godel_machine.py:159 | - | auditor:correction (COVERAGE_TO_100.md:81-86) | 2026-07-29 |
| category_table_never_committed | UNSCORED | - | - | auditor:WholeBrain (only a category total, 29, reached COVERAGE_BASELINE.md:21) | 2026-07-28 |

- `faculty_charge_registry` scores **0** under rubric rule 4, which names this shape verbatim ("a
  registry with no registrations"). `_WIRED = []` at `region_wiring.py:34` and nothing calls `wire()`
  outside the module's own self-check, so `charge_all()` iterates an empty list every cycle and the
  daemon logs `adapters=0`. The module's own docstring says so.
- `godel_machine_selfbuild` scores **0**: `attempt_on_repo` (`godel_machine.py:159`) has zero call
  sites tree-wide; the only reference to the module outside itself is its filename as a string in the
  self-build's protected-paths list. POLICIES.md listed it among the modules that "fire every 8 nights".
- COVERAGE_BASELINE.md:32 records a faculty here scored **70** on a single A/B. No artifact exists.
