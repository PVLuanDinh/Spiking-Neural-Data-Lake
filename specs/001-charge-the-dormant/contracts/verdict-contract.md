# Contract: verdicts, exit codes, and exemption keys

Three interfaces this feature must not break, because recorded evidence and CI both depend on them.

## 1. `tools/charge_probe.py` — verdict and exit code

```
BEHAVIOURAL CHARGE OK          exit 0   changed a decision / trajectory       rubric band 61-80
INTERNAL-READ CHANGED          exit 1   changed a value the brain later reads rubric band 41-60
CONSUMED-BUT-NOT-DECISIVE      exit 1   consumed, no measured change
NOT-ELIGIBLE                   exit 2   the source never activated
UNKNOWN                        exit 2   the probe cannot ask this question
```

**Invariant — exit 0 is the >40 gate.** `coverage_score.py`'s F8 demands an exit-0 artifact for any
band above 40. Returning 0 for anything weaker than a decision change would let 41–60 evidence
satisfy a 61–80 claim. `INTERNAL-READ CHANGED` therefore exits 1 deliberately.

**Invariant — stdout ratios are fingerprint observables.** `_fingerprint` harvests every bare `N/M`
from a probe's stdout. Any new line printed as a ratio appends an observable to EVERY recorded
artifact's fingerprint and fails them all under `--rerun`. New diagnostics print `N of M`. This is
not a style preference; it is the difference between adding a diagnostic and invalidating the
ledger.

## 2. `tools/coverage_score.py` — artifact and citation formats

- Artifact header keys are fixed: `kind`, `command`, `exit`, `commit`, `recorded_at`,
  `recorded_by`, `python`. `record` writes them; `_artifact` parses them.
- Citations are `path:line` or `path:line#anchor`, where the anchor is the enclosing `def`/`class`.
  The bare form must keep resolving — every legacy row uses it.
- `--rerun` compares recorded exit code AND observable fingerprint. Both must match.
- **F8**: band > 40 requires an artifact whose recorded exit code passed. Unchanged by this feature.

## 3. `substrate/method_dormancy_gate.py` — exemption keys

Current (after `f181b484`): `EXEMPT_METHOD` accepts `module.method` and falls back to bare `method`.

**Required by this feature**: also accept `module.Class.method`, preferred over both.

```
lookup order:  module.Class.method  →  module.method  →  method
```

**Invariant — an exemption must not be able to suppress a finding it does not name.** A bare
`"learn"` key exempts that verb across all 194 modules, which would hide the very defect class this
gate exists to catch. Existing bare keys are retained for compatibility but no NEW bare key may be
added.

**Invariant — `audit()` returns `{dormant, uncertain, open_exempt, n_defs}`** and the gate's
self-check asserts on `dormant` only. `uncertain` is REPORTED, never asserted: it is a list of
questions, not a list of failures. Changing that would turn 16 open questions into a red gate and
create pressure to silence them.

## Compatibility rules

1. No recorded artifact may stop reproducing because of a change made here. Check with `--rerun`
   after any change to a probe's output.
2. No existing exemption key may stop working.
3. `score --check` must stay exit 0, and the headline number must not move unless a band changed
   for a stated, measured reason.
