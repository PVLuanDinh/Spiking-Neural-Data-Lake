# Phase 1 Data Model: Charge the Dormant

No database and no new persisted schema. The entities here are the units this feature reasons
about; two of them already exist on disk as file formats.

## Rule under adjudication

The unit of work. Nine instances.

| field | type | notes |
|---|---|---|
| `module` | str | dotted, e.g. `basal_ganglia.latent_q` |
| `cls` | str | enclosing class, e.g. `LatentQ` — **required**, because two rules share a module and a method name |
| `method` | str | e.g. `update` |
| `lineno` | int | current definition line; volatile, so never the identity |
| `verdict` | enum | `NEVER_CALLED` \| `CHANGED_DECISION` \| `CHANGED_INTERNAL_READ` \| `CHANGED_NOTHING` \| `UNKNOWN` |
| `tier` | enum | `eprop-snn` \| `stdlib` — part of the verdict, not context |
| `disposition` | enum | `WIRED` \| `RETIRED` \| `EXEMPT` \| `OPEN` |
| `artifact` | str | filename under `docs/coverage/artifacts/`, or `-` |

**Identity** is `module.cls.method`. Not `module.method`: `latent_q` defines `update` on both
`LatentQ` and `_TabularQ`, with opposite correct dispositions. Not the line number: it drifts, and
the drift is what produced a finding reported against the wrong class.

**Validation**
- `disposition = RETIRED` requires a recorded port of any unique capability (P2).
- `disposition = EXEMPT` requires `(disposition_kind, reason)` and MUST be keyed by the full
  identity, so it cannot suppress a different rule.
- `disposition = WIRED` requires an artifact whose verdict is `CHANGED_DECISION` or
  `CHANGED_INTERNAL_READ`. A wire that produces `CHANGED_NOTHING` is a dead consumer and is
  forbidden by P3.
- `verdict = UNKNOWN` is legal and terminal-for-now; it MUST NOT be silently converted to
  `CHANGED_NOTHING`.

**State transitions**

```
OPEN ──measure──> verdict assigned
  ├─ NEVER_CALLED ──port check──> RETIRED
  ├─ NEVER_CALLED ──real call site exists──> WIRED ──A/B──> (must charge, else revert)
  ├─ NEVER_CALLED ──correct to be dead──> EXEMPT(permanent, reason)
  ├─ CHANGED_NOTHING ──fix the disjoint wiring──> re-measure ──> WIRED
  └─ UNKNOWN ──> stays OPEN, recorded as unknown
```

## A/B artifact *(existing format — `docs/coverage/artifacts/*.txt`)*

Machine-written by `tools/coverage_score.py record`. Header keys: `kind`, `command`, `exit`,
`commit`, `recorded_at`, `recorded_by`, `python`, then `--- stdout ---` and the verbatim run.

**Validation**: must re-run to the same exit code and the same observable fingerprint under
`--rerun`. A hand-written transcript survives `score` and dies here — that is the point of the
format.

**Extension for this feature**: the header's `recorded_by` MUST name the perception tier, because
`AffectiveAppraisal.learn` charges in one tier and not the other, and a tier-free verdict is not a
verdict.

## Disposition record

Lives in one of two places depending on kind:

- `EXEMPT` → `substrate/method_dormancy_gate.py::EXEMPT_METHOD`, keyed `module.Class.method`,
  value `(kind, reason)` where `kind ∈ {permanent, wire, retire}`. A non-`permanent` kind is an
  OPEN exemption and the gate reports it as a standing TODO rather than hiding it.
- `WIRED` / `RETIRED` → the commit that lands it, plus the ledger row's `measured_by` cell.

## Ledger row *(existing format — `docs/coverage/<Category>.md`)*

Columns: `faculty | band | citation | artifact | measured_by | measured_at`. Citations may carry
`#anchor`. This feature may only change a row when a measurement here bears on it, and the change
MUST cite that measurement (FR-007).

## Relationships

```
Rule 1──0..n A/B artifact          (a rule may be measured more than once, e.g. per tier)
Rule 1──1    Disposition
Ledger row 0..n──0..n Rule          (a band may rest on several rules; a rule may back several rows)
```

The many-to-many at the bottom is why re-banding is scoped: changing one rule's verdict does not
mechanically change one band.
