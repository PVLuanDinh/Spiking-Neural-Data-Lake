# CLI Contract: `python -m snn_data_lake.substrate.policy_manifest`

One module: six flag modes plus the bare run_selfchecks invocation. ASCII-only output (Windows cp1252). Exit code 0 = the mode's honest green;
nonzero = red OR derivation failure (derivation failure exits before printing rows — FR-007).

## Modes

| invocation | does | exit 0 when |
|---|---|---|
| `--manifest [--json]` | derive + print the manifest; runs nothing | derivation reconciled (headings vs table, R1) |
| `--manifest --write` | regenerate the committed copy `tools/policy_manifest.json` (the ONLY edit path) | write succeeded + derivation reconciled |
| `--audit [--json]` | derive, then run every row's commands (subprocess); print per-policy verdicts + `NOT_IN_CI` / `UNCLAIMED_GATE` drift findings | derivation ok AND no FAIL/CANNOT_RUN (NO_GATE and REVIEW_ONLY rows print but do not redden `--audit`; NO_GATE and unclaimed-gate drift redden `--certify`, REVIEW_ONLY never does) |
| `--certify [--full]` | audit + write `tools/policy_certification.json`; `--full` also runs the three suite-level gates | certified == true (no FAIL / CANNOT_RUN / NO_GATE, ZERO unclaimed-gate drift, derivation reconciled; REVIEW_ONLY disclosed, never blocking) |
| `--assert` | CI mode: committed `tools/policy_manifest.json` must MATCH a fresh derivation (sync assert), derivation must reconcile, AND every `absolute`/`ratchet` command must be claimed by a policy and present in CI (drift ratchet — the counts may improve, never worsen vs the pinned baseline inside the module) | no derivation failure, no NEW drift finding |
| `--self-check` | verify-the-verifier: constructed reds for FAIL, CANNOT_RUN, NO_GATE, empty walk, reconciliation mismatch — each must produce its verdict (SI 5 / P8 reachable red) | every constructed red reads red |
| *(bare, no flags)* | what `run_selfchecks.py` executes (it passes NO argv): identical to `--assert` — the sibling-gate precedent (`safety_envelope`'s bare main asserts; `gradient_claim_gate`'s in-module comment records shipping a bare-report path once and correcting it) | same as `--assert` |

## Contract details

- **CI wiring**: `--assert` runs as one lint step per commit; `--certify` runs as a separate job
  on every push to main, publishing the certification JSON as a job artifact, refusal loud in
  the job summary — non-blocking initially, flip-to-required a recorded owner decision
  (Clarifications 2026-08-18). **`--assert` itself does NOT execute other gates** — CI already runs
  them; running them twice doubles lint time for zero information. It asserts the MAP is honest,
  not the territory green.
- **`--audit` treats NO_GATE/REVIEW_ONLY as report-not-red** so the audit is runnable on today's
  tree (four review-only policies at 2026-08-18). `--certify` reddens on NO_GATE and on
  UNCLAIMED_GATE drift; REVIEW_ONLY is disclosed in the certificate and NEVER blocks
  (Clarifications 2026-08-18). SC-002's reachable red: `--certify` on today's tree MUST refuse —
  driven by the UNCLAIMED_GATE census (three members at analysis time) until their
  enforcement-table rows land post-count.
- **Baseline for `--assert`**: pinned SET inside the module (same pattern as every ratchet here),
  updated only with a stated reason in the diff. The initial baseline is whatever the FIRST
  COUNTED RUN reports — measured expectation at analysis time: `{safety_envelope,
  doc_budget_gate, ingest_guard}`, an observation, never the assert. The instrument's own gate
  gets a NAMED self-disposition printed in every report. Clarifications: every member's claiming
  table row lands post-count with US4's doc lane, then the baseline ratchets to EMPTY.
- **Bare must be able to fail**: the bare path exists so a run_selfchecks entry can go RED —
  registration (tasks T003) is DEFERRED until `--assert` and `--self-check` are implemented,
  because a registered module whose bare path cannot exit nonzero is a PASS that can never
  fail (the defect `gradient_claim_gate.py`'s comment records shipping for one commit).
- **Output discipline**: counts are printed, never restated in docs; prose that needs the numbers
  says "re-run the gate" (Phase 8 drift-class rule).
- **JSON output** (`--json`): one object, schema mirroring data-model.md; stdout only, no file
  written except by `--certify`.

## Failure-message contract

- Derivation failure names ALL identity terms (`parents=N table_only=K table_rows=M`, with
  N + K != M spelled out) — a bare "mismatch" is not actionable.
- `CANNOT_RUN` includes the argv and the OSError/exit context; `FAIL` includes argv + exit code +
  last stderr line. The two are never collapsed (FR-005).
- Empty walk prints the zero explicitly ("0 policies parsed — refusing: a dead walk and a clean
  tree look identical") and exits nonzero (FR-007).
