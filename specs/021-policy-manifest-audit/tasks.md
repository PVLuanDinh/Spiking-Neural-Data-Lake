# Tasks: Policy Manifest + Release Audit

**Input**: Design documents from `/specs/021-policy-manifest-audit/`

**Prerequisites**: plan.md, spec.md (with Session 2026-08-18 clarifications), research.md,
data-model.md, contracts/cli.md, quickstart.md

**Tests**: INCLUDED — the spec mandates them (SC-004 "demonstrated by a constructed failure",
`--self-check` constructed reds per SI 5 / P8; an instrument whose reds are unreachable is the
defect class this feature audits for).

**Organization**: by user story. Standing constraints binding EVERY task: zero `brainstem/` edits
(FR-013 — verify with `git diff --stat` before each commit); no policy CONTENT change; POLICIES.md
never grows past 29,750 words (FR-011); ASCII-only output; counts printed by the tool, never
restated in prose (Phase 8 drift rule); one change per commit with the full gate green.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

**Purpose**: baseline capture — probe before act.

- [X] T001 Record the pre-feature gate state: run `.venv-lake/Scripts/python.exe -m ruff check .` and `.venv-lake/Scripts/python.exe -m pytest -q` (exit codes captured SEPARATELY), plus `.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.doc_budget_gate` (bare `python` on this machine is the hermes venv and cannot import the package — measured); note results in specs/021-policy-manifest-audit/manifest-baseline.md (create the file with a "Pre-feature state" section). A polluted baseline reads clean — this is the control for every later "nothing broke" claim.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the derivation core every story consumes. No story starts before T007 passes.

- [X] T002 Create src/snn_data_lake/substrate/policy_manifest.py skeleton: module docstring (what it derives, what it never does — no policy edits), dataclasses Policy, GateCommand, ManifestRow, AuditResult, Certification per data-model.md, argparse main with the flag modes from contracts/cli.md stubbed to exit 2 "not implemented" (bare invocation == `--assert` per the contract — which is why registration, T003, waits until it exists).
- [X] T003 Register the module — EXECUTES AFTER T016, see Dependencies: add to REGIONS in src/snn_data_lake/__init__.py, to run_selfchecks.py, AND a `dormancy_gate.EXEMPT["policy_manifest"]` ("permanent", reason) entry modelled on the gradient_claim_gate entry (dormancy_gate.py:354 — every sibling gate carries one). Bare invocation must already equal `--assert` and be able to exit nonzero: registering a module whose bare path cannot fail is the defect gradient_claim_gate.py's in-module comment records shipping for one commit. The CI `--assert` step added in US2 is its live path.
- [X] T004 Implement the HEADING parse in src/snn_data_lake/substrate/policy_manifest.py: `^## P\d` parents + the GENERAL sub-clause pattern `^### P\d[a-z]*( amendment)?` (anchored alternation — the enumerated `^### P4b|P4c` form had a precedence bug matching 16 lines and silently dropped `P3 amendment` and `P5a`), roll-up under parent P<digit> per research.md R1; refuse an empty walk loudly (FR-007: "0 policies parsed — refusing").
- [X] T005 Implement the TABLE parse in src/snn_data_lake/substrate/policy_manifest.py: BOTH head-matter tables, located by structural anchor never line numbers — the enforcement table (header containing `mechanized in CI`) for per-row enforcement kind per research.md R5 (`absolute`/`ratchet`/`review-only`; e-prop PARTIALLY splits into per-command kinds plus a review-only remainder), and the gate-output table (header `| gate | output | in CI? |`) as the command source for gates the enforcement row does not spell out (P4c's `tools/p4c_ratchet.py` lives ONLY there — measured, R1). Command accept-filter: a backticked token is a command iff it matches `^(tools|substrate)/[a-z0-9_]+\.py( --[a-z-]+)*$`; `tools/x.py` → `[python, tools/x.py, <flags>]`, `substrate/x.py` → `[python, -m, snn_data_lake.substrate.x, <flags>]`; everything else (`main()`, `.backward()`, `mlp_q`, line-cites) is ignored as citation. Sub-clauses are first-class output rows with their OWN kind, default `review-only` when nothing names their gate, NEVER inheriting the parent's command (research.md R5); excluded from the reconciliation identity. Exclude rows self-labelled "not a policy".
- [X] T006 Implement reconciliation in src/snn_data_lake/substrate/policy_manifest.py: the identity `parent_headings + len(TABLE_ONLY_IDS) == table_policy_rows` with `TABLE_ONLY_IDS = {"e-prop"}` pinned beside a comment (e-prop has a table row and NO heading — measured, R1); failure message naming ALL terms (`parents=N table_only=K table_rows=M`), nonzero exit before any row prints (FR-003).
- [X] T007 [P] Write tests/test_policy_manifest.py part 1 — parsers: the identity parents + table_only == table rows holds on the REAL docs/POLICIES.md (live positive — 9 + 1 == 10 at HEAD; assert the IDENTITY, not the frozen numbers); constructed fixture with a missing table row → reconciliation red, and a fixture where a TABLE_ONLY id gains a heading → red (double-count direction); empty-document fixture → FR-007 red; accept-filter negative control — a fixture cell holding `dormancy_gate.py:71`, `main()` and `.backward()` yields ZERO commands. Run: `.venv-lake/Scripts/python.exe -m pytest tests/test_policy_manifest.py -q`.

**Checkpoint**: derivation core proven on the real tree with reachable reds.

---

## Phase 3: User Story 1 — Show me, honestly, which policies nothing enforces (P1) — MVP

**Goal**: one command prints every policy with enforcement kind + command-or-NONE; counts reconcile;
review-only shown prominently as not-machine-decided, never green.

**Independent test**: quickstart.md step 1 — `--manifest` on the real tree, row count equals
heading count after roll-up; step 2's honesty check (an all-green first run = blind instrument).

- [X] T008 [US1] Implement `--manifest` and `--manifest --json` in src/snn_data_lake/substrate/policy_manifest.py: text table + JSON per data-model.md; review-only rows rendered prominently as NOT MACHINE-DECIDED (never green — SC-002 wording); sub-clause rows render with their OWN kind (P4b: NOT MACHINE-DECIDED, research.md R5).
- [X] T009 [US1] Implement the liveness cross-check in src/snn_data_lake/substrate/policy_manifest.py per research.md R2: parse `.github/workflows/ci.yml` `--assert` steps and run_selfchecks.py registrations; populate GateCommand.in_ci / in_selfchecks; DEFINE the gate predicate in the module (run_selfchecks registers ~50 modules, most not gates — the predicate is stated and tested, never implicit); compute drift findings NOT_IN_CI and UNCLAIMED_GATE; record a NAMED self-disposition for policy_manifest's own gate, printed in every audit report (a silently self-exempting auditor is the finding class).
- [X] T010 [US1] Implement `--manifest --write` in src/snn_data_lake/substrate/policy_manifest.py: regenerate tools/policy_manifest.json (the ONLY edit path; UTF-8, sorted keys, trailing newline for stable diffs), then create the committed copy by running it once.
- [X] T011 [P] [US1] Extend tests/test_policy_manifest.py — manifest: kind derivation spot-checks (P3 per its row's wording, P4c/P6 ratchet — P4c's command sourced from the gate-output table, P1 review-only, P4b review-only-not-inherited — the overclaim control, e-prop split); the UNCLAIMED_GATE census on the REAL tree is non-empty and contains `safety_envelope`, `doc_budget_gate` and `ingest_guard` (the live reds — this test pins the findings until US4 claims the rows, then flips to assert an EMPTY census: mark it with a comment saying exactly that); constructed NOT_IN_CI fixture; SC-007 fixture — copy docs/POLICIES.md, append a synthetic `## P9` heading AND its enforcement-table row, assert the derived manifest gains exactly one row with the correct kind and command-or-NONE and still reconciles, with ZERO code edits (a hardcoded policy list fails this).
- [X] T012 [US1] Record the FIRST COUNTED RUN in specs/021-policy-manifest-audit/manifest-baseline.md: full `--manifest` output verbatim, dated, at the named commit — this is the FR-012 count that unblocks any later policy-text edit. Do NOT restate the numbers anywhere else; point here.

**Checkpoint**: US1 delivers standalone value — the "what enforces this?" question is answerable
from one command (SC-003).

---

## Phase 4: User Story 2 — A build cannot be certified without being audited (P2)

**Goal**: `--audit` runs every row's command; `--certify` refuses on FAIL / CANNOT_RUN / NO_GATE /
unclaimed-gate drift, discloses REVIEW_ONLY, records what it ran; `--assert` guards CI per commit;
certify job runs on every push to main.

**Independent test**: quickstart.md steps 2–4 — audit runs on the real tree; certify REFUSES on
today's tree naming every unclaimed census member; every constructed red fires.

- [X] T013 [US2] Implement `--audit` in src/snn_data_lake/substrate/policy_manifest.py: subprocess execution per row (never import — state isolation, 020's FR-013b lesson), verdict mapping per data-model.md, CANNOT_RUN carrying argv + OSError context DISTINCT from FAIL carrying argv + exit + last stderr line (FR-005), REVIEW_ONLY neutral-listed, exit semantics per contracts/cli.md.
- [X] T014 [US2] Implement `--certify [--full]` in src/snn_data_lake/substrate/policy_manifest.py: writes tools/policy_certification.json per data-model.md (tree hash + dirty flag, commands_run with durations, review_only disclosure list); certified true ONLY on zero FAIL/CANNOT_RUN/NO_GATE/unclaimed-drift + reconciled derivation; `--full` additionally runs the three suite gates.
- [X] T015 [US2] Implement `--assert` in src/snn_data_lake/substrate/policy_manifest.py: committed tools/policy_manifest.json must byte-match a fresh `--write` rendering (sync assert); derivation reconciles; drift ratchet vs pinned baseline SET inside the module — initialized from T009's FIRST COUNTED RUN, never from spec prose (analysis-time expectation `{safety_envelope, doc_budget_gate, ingest_guard}` is an observation, not the assert), with a comment naming the post-count claim plan (clarification 1, as corrected 2026-08-18); the set may shrink, never grow.
- [X] T016 [US2] Implement `--self-check` in src/snn_data_lake/substrate/policy_manifest.py: five constructed-red fixtures — FAIL (a command that exits 1), CANNOT_RUN (nonexistent executable), NO_GATE (fixture policy with absolute kind and no command), empty walk, reconciliation mismatch — each must produce its verdict; exit 0 only when every red is reachable (SI 5).
- [X] T017 [P] [US2] Extend tests/test_policy_manifest.py — audit/certify: verdict mapping incl. the FAIL vs CANNOT_RUN distinction asserted on two constructed rows in one run; certify refusal on the real tree (names every unclaimed census member; `certified: false` in the JSON); CERTIFY-LEVEL constructed reds independent of the census — a fixture row whose command exits 1, and a second whose executable is missing: each makes the certify decision refuse, naming that policy and its verdict (SC-004's demonstration must survive US4 clearing the census); certification JSON schema + dirty-flag honesty; REVIEW_ONLY rows present and never blocking; green-path positive control — an all-stub-pass fixture (every command exits 0, no drift) asserts `certified: true` is REACHABLE with `commands_run` non-empty (durations recorded) and the review_only disclosure list populated: the certificate must be demonstrably issuable, not only refusable.
- [X] T018 [US2] Wire CI in .github/workflows/ci.yml: (a) one lint-job step `python -m snn_data_lake.substrate.policy_manifest --assert` after the existing gate steps; (b) a `certify` job on `push` to `main` running `--certify`, uploading tools/policy_certification.json as an artifact, `continue-on-error: true` with the refusal printed in the job summary (non-blocking initially — FR-014; flip-to-required is a recorded later owner decision, not this feature's).
- [X] T019 [US2] Run the full local gate battery per quickstart.md step 5 (run_selfchecks.py, pytest, ruff — exit codes captured separately) and append the audit + certify first-run outputs verbatim to specs/021-policy-manifest-audit/manifest-baseline.md.

**Checkpoint**: certification exists, refuses honestly on today's tree, and every landing build
gets audited by CI.

---

## Phase 5: User Story 3 — Every process that generates work points at the policy (P3)

**Goal**: templates carry the runnable check; dispatch delivers regenerated manifest output.

**Independent test**: quickstart.md step 6 — generate a feature from the templates; the policy
check appears as a runnable command, not a description.

- [X] T020 [P] [US3] Edit .specify/templates/plan-template.md: Constitution Check section gains the runnable line `python -m snn_data_lake.substrate.policy_manifest --audit` plus a pointer to docs/POLICIES.md as the authority — pointer + command ONLY, zero restated policy content (Phase 8 drift-class rule).
- [X] T021 [P] [US3] Edit .specify/templates/spec-template.md, tasks-template.md, constitution-template.md and checklist-template.md: one pointer line each naming docs/POLICIES.md and the manifest command (FR-009 — measured 2026-08-18: ZERO of the five templates references `POLICIES.md`; the spec's earlier "constitution-template mentions it" was a substring artifact).
- [X] T022 [US3] Extend the existing POLICIES guard hook — the file is C:\Users\vinhl\.claude\hooks\sndl-policies-guard.ps1 (SessionStart; NOT sndl-policies-per-task.ps1, the per-prompt guard installed beside it) — to inject freshly regenerated manifest output at session start by running `& "D:\AI-project\spiking-neural-data-lake\.venv-lake\Scripts\python.exe" -m snn_data_lake.substrate.policy_manifest --manifest` with the repo as working directory and passing stdout through: a THIN CALLER embedding zero policy content of its own (clarification 2). On any failure the hook emits a visible one-line marker, never a silent skip. Disclose in the hook file's comment that its user-scope location is outside repo gates and why that is safe (nothing in it can rot).
- [X] T023 [US3] Verify US3 end-to-end: run `.specify/scripts/powershell/setup-plan.ps1` against a scratch feature dir (SPECIFY_FEATURE_DIRECTORY pointed at a temp specs dir), confirm the generated plan carries the runnable check; delete the scratch dir; verify the HOOK path too — execute the hook's command line directly, assert its output byte-matches a fresh `--manifest` run, and grep the hook file to assert it contains no policy text of its own (the thin-caller property, mechanized); record both verifications in specs/021-policy-manifest-audit/manifest-baseline.md.

**Checkpoint**: new work is generated pointing at the manifest; dispatch delivers data, not prose.

---

## Phase 6: User Story 4 — The governing document gets smaller (P4) — LAST, gated on T012

**Goal**: POLICIES.md measurably below 28,559 words (SC-006: ≥1,191 words recovered from 29,750),
ceiling LOWERED in the same commit, nothing lost — every relocated case byte-exact in the
companion file and reachable from a pointer.

**Ordering gate**: no task in this phase starts before T012's counted run is recorded (FR-012).
Byte-preserving rule throughout: RELOCATION, never rewording (clarification 3).

- [X] T024 [US4] Count relocatable mass per section of docs/POLICIES.md with a throwaway script run from the session scratchpad, NEVER saved into the tree (an untracked .py anywhere non-ignored enters the p4c ratchet corpus — run `python tools/p4c_ratchet.py --assert` before this phase's first commit as the control); the script IMPORTS `doc_budget_gate`'s own word counter rather than reimplementing it (two counters disagree by 502 words on this file — the gate is the arbiter). Words per `## `/`### ` section, incident-forensics blocks identified by their headers; record the ranked table in specs/021-policy-manifest-audit/manifest-baseline.md. The e-prop block is touched ONLY if the non-e-prop total cannot reach 1,191 words (clarification 3) — the recorded table decides, not judgment.
- [X] T025 [US4] Create docs/POLICIES_EVIDENCE.md (header: what it holds, that POLICIES.md is authoritative, that entries are byte-exact relocations) and register it in doc_budget_gate.BUDGETS in src/snn_data_lake/substrate/doc_budget_gate.py with a ceiling counted from its initial content — SAME commit (research.md R6). This commit is NOT standalone: it lands together with T026's FIRST relocation batch (R6's one-commit shape), so the ceiling is counted from real relocated content, never from an empty header that the next commit immediately blows past.
- [X] T026 [US4] Relocate the counted sections from docs/POLICIES.md to docs/POLICIES_EVIDENCE.md in ranked order until the SC-006 target is met: each move byte-exact, a one-line pointer left at the old site; verify per move with a scratchpad script asserting the moved text appears verbatim in the destination (no rewording possible if the bytes must match). Each batch commit ALSO: regenerates tools/policy_manifest.json via `--manifest --write` and re-runs `--assert` (a POLICIES.md edit with a stale committed manifest reddens T015's sync assert in CI), adjusts POLICIES_EVIDENCE.md's ceiling in the same commit as the batch that grows it, and pre-checks that no relocated block contains a backticked gate command inside a parsed region — if the manifest diff is non-empty, STOP and re-scope the batch (relocation must never silently change the derived manifest).
- [X] T027 [US4] LOWER the POLICIES.md ceiling in src/snn_data_lake/substrate/doc_budget_gate.py to the new measured size (+small stated slack) in the SAME commit as the final relocation, with the reason in the constant's comment — the consolidation-debt note there is retired by this edit, say so.
- [X] T028 [US4] Add enforcement-table rows in docs/POLICIES.md for EVERY member of the counted unclaimed census (POST-count per clarification 1; `safety_envelope` as a policy-enforcement row, `doc_budget_gate`/`ingest_guard` as `*(not a policy — repo hygiene)*` rows matching the citation_guard precedent; the words come out of the relocation budget so FR-011 holds), then ratchet the module baseline census to EMPTY in src/snn_data_lake/substrate/policy_manifest.py, regenerate tools/policy_manifest.json via `--manifest --write`, and flip the T011 live-red test to assert an empty census (updating its comment).
- [X] T029 [US4] Convert the constitution's per-principle P7/P8 status sentences — the two occurrences of "PROVISIONAL and review-enforced: its gate is specified but not built" in .specify/memory/constitution.md (near :218 and :250; locate by SENTENCE, both false since 2026-08-05) — to pure pointers at docs/POLICIES.md's enforcement rows, the same treatment specs/020 Phase 8 applied to the e-prop and P7/P8 table rows; then grep constitution.md for any remaining enforcement-status restatement that contradicts POLICIES.md and convert those too (measure, do not assume the two are the last); record the sweep in specs/021-policy-manifest-audit/manifest-baseline.md. This discharges FR-011's "duplication with the constitution is removed" clause, which previously had zero covering tasks.
- [X] T030 [US4] Re-run `.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.doc_budget_gate` + `--certify`; append the post-consolidation outputs verbatim to specs/021-policy-manifest-audit/manifest-baseline.md (SC-006 evidence; certify may now go green — if it does, confirm the T016 self-check still proves the reds reachable, so green is not blindness).

**Checkpoint**: the governing document is smaller, nothing lost, ceiling ratcheted down.

---

## Phase 7: Polish & Cross-Cutting

- [X] T031 [P] Run `graphify update .` to refresh graphify-out/ for the new module.
- [X] T032 Full-gate final pass: run_selfchecks.py, pytest (whole suite, coverage >= 70%), ruff — exit codes separate; `git diff --stat` proves zero brainstem/ lines across the whole feature (FR-013); `--assert` green; record in specs/021-policy-manifest-audit/manifest-baseline.md. After the feature's first push to main: verify the certify JOB actually RAN (nonzero duration — this repo has recorded "all green" while CI executed zero jobs) and uploaded tools/policy_certification.json as an artifact; quote the job-summary text in manifest-baseline.md.
- [X] T033 Audit the completed feature against docs/POLICIES.md IN FULL (owner rule: no build is declared complete until audited against POLICIES.md) — P2 one-mechanism check (the enforcement table is still the single hand-edited authority; the manifest is derived), P3 (module wired + charged via CI), P4c (nothing outside the governed tree), P8 debts (reds reachable, control independent), FR-011/FR-013 held; record verdicts per policy in specs/021-policy-manifest-audit/manifest-baseline.md.

---

## Dependencies

```text
T001 -> T002..T007 (foundation)  -> US1 (T008-T012)
US1 -> US2 (T013-T019)   [audit consumes manifest + cross-checks]
US1 -> US3 (T020-T023)   [templates/hook point at a manifest that must exist]
T012 (counted run) ==FR-012 gate==> US4 (T024-T030)  [also after US2's T015 baseline exists]
T003 (registration) executes AFTER T016 — bare == --assert must exist and be able to fail
   (contracts/cli.md bare row; the gradient_claim_gate shipped-defect precedent)
US2, US3, US4 -> Polish (T031-T033)
```

US2 and US3 are independent of each other and may interleave after US1. US4 is strictly last
among stories.

## Parallel opportunities

- T007 alongside T004–T006 review (test file, different from module file, TDD-adjacent).
- T011 alongside T010; T017 alongside T014–T016 (test file vs module file).
- T020 and T021 in parallel (different template files); both parallel with US2 tasks.
- T031 parallel with T032.

## Implementation strategy

MVP = Phase 1 + 2 + US1 (T001–T012): the honest enforcement map, standalone value, FR-012's
counted output banked. Then US2 (certification), US3 (generation surfaces), US4 (consolidation)
as separately committable increments — one change per commit, full gate green each time, every
increment reaching main per the standing rule.
