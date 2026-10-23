# P8 null-scale amendment: gate pre-registration (frozen before the gate is built)

Owner, 2026-10-23 (verbatim, in order): "Draft a P8 amendment: every null records the scale it was measured at (grown
units, ticks, seeds), and deletion on a null requires a re-test at a larger scale first. Draft only; I decide whether it
lands." -- "Scale factor: 10x minimum on the scaling axis." -- "Expiry of a scale-retest exemption: before the next
pushed arc." -- "Harms: a measured harm is enough to delete; no scale re-test required." -- "Retroactivity: option B,
forward plus a one-time list of past small-scale deletions as re-test candidates." -- "Disputed scaling axis: the
pre-registration author decides, with a panel review." -- "Decision-C parts: yes, deleting a decision-C part whose
deciding test reds also waits for the 10x re-test." -- "Go: start the landing arc for the P8 null-scale amendment, gate
first, as the draft's header lists."

Draft text the gate enforces: `D:\AI-project\lanes\policy-drafts\P8-null-scale-amendment-DRAFT.md` (lands in
`docs/POLICIES.md` P8 in this arc). Cross-cutting invariant: the gate is built and its output COUNTED before the rule
lands. Sections 0-4 are frozen at the commit that adds this file; later changes are dated addenda in section 6.

## 0. Records the gate reads

Scope: tracked `specs/**/evidence/*.md` and `docs/*.md`.

```
NULL-RECORD: <method> | units=<int|n/a> | ticks=<int|n/a> | seeds=<int> | mde=<number|unknown> | retest=<R>
R := REF:<repo path>#<method> | PENDING since <full 40-hex sha> | STRUCTURAL: <reason, >= 20 chars>
RETEST-RECORD: <method> | axis=<units|ticks|seeds> | <axis>=<int> | result=<null|charges> | of=<repo path of the NULL-RECORD>
```

- `n/a` is allowed for `units` (a rule acting on no unit population) and `ticks` (an offline measurement), never for `seeds`.
- `unknown` is allowed for `mde` only inside `specs/022-unified-brain-arc/evidence/null-scale-triage.md` (the one-time
  retroactive list, option B); a forward NULL-RECORD states its mde.
- A **delete-on-null statement** is a line matching, case-insensitively,
  `(delet(ed|ion|e)|retir(ed|e)|removed) on (its|a|the|their) (own )?null` in an evidence file
  (`specs/**/evidence/*.md`). Policy restatements in `docs/` and spec plans are not statements about a deletion and are
  out of this scope.

## 1. What the gate reds (each one has a reachable red in section 3)

1. **R-FIELD** -- a NULL-RECORD or RETEST-RECORD line with a missing field, an unknown key, or a value of the wrong type.
2. **R-UNRECORDED** -- a delete-on-null statement in an evidence file whose markdown section (from the nearest heading
   above it to the next heading of the same or higher level) holds no NULL-RECORD line, unless the statement is
   GRANDFATHERED (section 2).
3. **R-RETEST-REF** -- `retest=REF:...` whose target has no RETEST-RECORD for the same method with `of=` pointing back.
4. **R-RETEST-SCALE** -- a RETEST-RECORD whose `<axis>` value is less than 10x the original NULL-RECORD's value on that
   axis (if the original's value is `n/a`, the RETEST-RECORD may not use that axis).
5. **R-EXPIRED** -- `retest=PENDING since <S>` where `<S>` is not an ancestor of HEAD, or where the commits `S..HEAD`
   contain a closing-record commit (subject starting `chore(record): closing execution record`) that is NOT HEAD itself.
   Meaning: the entry may ride in the arc that creates it up to and including that arc's closing record; once any commit
   lands after that record (the next arc being built), the entry is expired. A missing history (shallow clone, unknown
   sha) is RED with "history unavailable", never a pass.
6. **R-STALE-GRANDFATHER** -- a GRANDFATHERED entry that matches no current line (the list may shrink only by deleting the
   entry in the same commit).
7. **R-EXEMPT** (in `gates/dormancy_gate.py`, via the one shared check in `null_scale_gate`): an `EXEMPT` entry with the
   new disposition `"scale-retest"` whose reason does not name `NULL-RECORD:<repo path>` and `since=<full sha>`, or whose
   since is expired by rule 5. `dormancy_gate` imports the check (P2: one mechanism), never re-implements it.

Not red: harms (owner answer 3; a harm needs no re-test), STRUCTURAL retests (the reason is printed for review; the gate
cannot judge an argument and says so in its output), records in files outside the scope.

## 2. Grandfathering and the count that comes first

`GRANDFATHERED` is a constant in `gates/null_scale_gate.py`: one entry per existing delete-on-null statement, keyed by
(repo path, sha256 of the stripped line text, first 12 hex) so a line move does not break it and a reword does.

**Count first, before the POLICIES.md text lands, on this branch's base (89f3af14):**
- expected: the 31 statements of the 2026-10-23 census (17 evidence files) are found, all 31 grandfathered;
  NULL-RECORD lines = 0; R-* reds = 0. The landing commit's message prints the real counts.
- If the gate finds a number other than 31, the census or the regex differs: the gate's own count wins, the difference
  is listed line by line in section 6 before anything lands.

## 3. Reachable reds (tests, `tests/test_null_scale_gate.py`) and the self-check

Each R-* gets a test that constructs the failing input in a temporary tree (or an injected history function for
R-EXPIRED) and asserts the gate reds with that code, plus one test that the clean input passes. In addition:
- **R-EXPIRED** is tested at four points of an arc's life: before the record (green), at the record as HEAD (green), one
  commit after the record (red), unknown sha (red).
- **Mutants** (run by the builder, logged, not committed): removing each check from the gate makes its test red.
- `python -m snn_data_lake.gates.null_scale_gate` = `--assert` (exit 1 on any red, prints counts); `--self-check` runs
  the planted reds in-process and fails if any does not red (a gate seen only green is indistinguishable from one that
  cannot see).

## 4. Landing (one arc, one battery)

Lanes build on branches cut from `claude/p8-nullscale-1023`; the integrator composes onto it, then onto int3:
1. gate + tests + `dormancy_gate` disposition (lane `p8-gate`);
2. `null-scale-triage.md`, the option-B list (lane `p8-triage`, read-only research, writes only that file);
3. `docs/POLICIES.md` P8 text + P4d sentence (both places) + `docs/AGENT_COLD_START.md` + `run_selfchecks.py` / package
   registry / `ci.yml` step with full-history checkout (integrator);
4. panel (3 Opus lenses), fold, ONE battery on the stacked head, closing record LAST, push, read back.

## 5. Dispositions

- If the counted output is not clean on the base (a non-grandfatherable finding), the rule does not land in this arc;
  the finding goes to the owner.
- A panel MAJOR that the gate cannot be made to catch is disclosed in the closing record, not hidden.

## 6. Addenda (dated, appended only)

1. 2026-10-23, before any lane started and before the gate exists (no data). Retroactive records often do not state
   every field. Inside `null-scale-triage.md` ONLY, `unknown` is allowed for `units`, `ticks`, `seeds` and `mde` (not
   only `mde`). Forward records are unchanged: `seeds` is always an int, `mde` a number. The triage file cites each
   statement by `path:line` and its grandfather key and never quotes the statement's wording, so the file itself
   carries no delete-on-null statement for R-UNRECORDED to read.
