# Implementation Plan: Policy Manifest + Release Audit

**Branch**: `021-policy-manifest-audit` | **Date**: 2026-08-18 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/021-policy-manifest-audit/spec.md` (audited against
HEAD `f61e4a94` 2026-08-18 — see the spec's "Audit against tree state" section; the pinned counts
from 2026-08-17 are superseded inline there).

## Summary

Build ONE governed instrument that (a) DERIVES a machine-readable manifest of every policy in
`docs/POLICIES.md` — id, binds-what, enforcement kind, runnable gate command or NONE — and (b) runs
a release audit over it that refuses to certify while any policy is missing a command, red, or
unrunnable, reporting `CANNOT_RUN` distinctly from `FAIL` and `REVIEW_ONLY` as a first-class honest
kind. Then point the spec-kit templates and agent dispatch at the manifest (US3), and pay the
recorded consolidation debt so POLICIES.md gets smaller (US4). Per FR-012, the manifest and audit
are built and their output COUNTED before any policy text is edited — US4 is LAST and is gated on
US1's counts. Five clarifications recorded in spec.md (Session 2026-08-18) bind this plan:
safety_envelope claimed POST-count (baseline UNCLAIMED_GATE:1 first); FR-010 = guard hook as
thin regenerating injector + repo-tracked command in templates; US4 relocation is count-first,
byte-exact, e-prop only if otherwise unreachable; REVIEW_ONLY is certifiable-with-disclosure;
--certify runs as a non-blocking CI job on every push to main.

## Technical Context

**Language/Version**: Python 3.14 free-threaded (`.venv-lake`, the ONE gate env)

**Primary Dependencies**: stdlib only for the instrument (`re`, `subprocess`, `json`, `argparse`).
No new dependency is needed; P7 accepts any, but nothing here justifies one.

**Storage**: manifest derivation is the source of truth; a committed copy
`tools/policy_manifest.json` exists for review with a sync assert in `--assert` (the spec's
recorded assumption + the p4c-baseline pattern); certification records are JSON under `tools/` (measured in specs/020: a `.json` under `tools/`
costs the p4c ratchet nothing — its corpus is `*.py` only).

**Testing**: pytest (`tests/test_policy_manifest.py`), plus the module's own `--self-check`
(verify-the-verifier: constructed RED for every verdict kind, per SI 5 / P8).

**Target Platform**: Windows + Linux CI (ASCII-only output; `zlib.crc32` if any hashing).

**Project Type**: single governed CLI module inside the existing package.

**Performance Goals**: default audit completes in under ~2 minutes — it runs the per-policy gate
commands (each seconds), NOT the full 30-minute suite; `--full` exists for the release lane.

**Constraints**: FR-013 — no policy CONTENT changes, no `brainstem/` files touched, no gate
relaxed. FR-011 — POLICIES.md must not grow from 29,750 words. Doc-budget gate binds any doc edits.

**Scale/Scope**: ~12 policy headings, ~12 enforcement-table rows, 5 CI `--assert` gates today —
all counts DERIVED at runtime, never pinned in code or prose (the spec's own pinned counts went
stale within one day; that is the finding).

## Constitution Check

*GATE: evaluated against `.specify/memory/constitution.md` + `docs/POLICIES.md` (authoritative).*

| gate | verdict | why |
|---|---|---|
| P2 adopt-or-merge, non-additive | PASS with design constraint | POLICIES.md's head matter (the enforcement table + the gate-output table, located by header anchor) is a hand-maintained proto-manifest. The manifest DERIVES from it rather than standing beside it; the table remains the single hand-edited authority, the manifest is its parse plus reconciliation. No second hand-edited copy is created. |
| P3 no dormancy | PASS | new module registers in `__init__.py` REGIONS + `run_selfchecks.py` + `dormancy_gate.EXEMPT` (permanent, reasoned — every sibling gate carries one), with registration DEFERRED until the bare path == `--assert` can exit nonzero (T003 after T016; a bare-stub registration is the defect `gradient_claim_gate.py`'s comment records shipping once); live path is the CI lint job (`--assert`). |
| P4c every build is part of the brain | PASS | lands in `src/snn_data_lake/substrate/` from the first commit; nothing goes to `reasoning/` or the repo root. |
| P8 invented-instrument debts | PASS (planned) | reachable RED: `--self-check` constructs a failing gate, a missing command, and an empty policy walk, and asserts each verdict (SC-004, FR-005, FR-007). Control that is not the method: the reconciliation count comes from policy HEADINGS while rows come from the TABLE — two independent parses that must agree. Deletion on a null is not applicable (the audit is expected non-green on first run; an all-green first run is itself treated as the failure, per SC-002). |
| Count-first invariant (FR-012) | PASS by ordering | US1 (manifest + counts) completes and its output is recorded BEFORE US4 touches any policy text. Three prior proposals died at this bar; this plan sequences around it rather than arguing with it. |
| Safety envelope | PASS | zero `brainstem/` edits anywhere in this feature (FR-013). The audit RUNS existing gates; it modifies none. |
| e-prop invariant | N/A | no gradient anywhere in this feature. |
| Doc budget | BINDS US4 | POLICIES.md at 29,750/29,750. US4 shrinks the FILE and LOWERS the ceiling in the same commit; a shrink that leaves the ceiling standing re-opens the headroom the budget denies. |

**Post-design re-check (Phase 1)**: unchanged — no design decision below adds a mechanism beside an
existing one, touches the envelope, or grows POLICIES.md.

## Project Structure

### Documentation (this feature)

```text
specs/021-policy-manifest-audit/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/cli.md     # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/snn_data_lake/substrate/policy_manifest.py   # THE instrument: derive + audit + certify,
                                                 # --manifest / --audit / --certify / --assert / --self-check
tests/test_policy_manifest.py                    # pytest coverage incl. constructed reds
tools/policy_manifest.json                       # committed manifest copy (review surface;
                                                 # sync-asserted against fresh derivation)
tools/policy_certification.json                  # certification record (FR-006); .json under tools/
                                                 # is ratchet-free (measured, specs/020 plan)
.specify/templates/plan-template.md              # FR-009: Constitution Check points at the manifest
.specify/templates/spec-template.md              # FR-009: pointer line
.specify/templates/tasks-template.md             # FR-009: pointer line
.specify/templates/constitution-template.md      # FR-009: pointer line (ALL FIVE templates
.specify/templates/checklist-template.md         #   measured blind at HEAD — none refs POLICIES.md)
docs/POLICIES_EVIDENCE.md                        # US4 companion file for relocated forensics
                                                 # (budgeted in doc_budget_gate from its first commit)
.github/workflows/ci.yml                         # lint step (--assert) + certify job on push
                                                 # to main (non-blocking, artifact-publishing)
```

**Structure Decision**: single governed module in `substrate/` (the specs/020 precedent:
`tools/*.py` reddens the p4c ratchet on its own commit, so the gate module lives governed and only
its JSON artifact lives under `tools/`). US3 edits are template text only; US4 edits are doc moves
with pointers, executed LAST under FR-012.

## Complexity Tracking

No constitution violations to justify. The one deliberate scope refusal: the manifest does NOT
attempt to machine-decide review-only policies (P1's "brain-native" is not machine-decidable);
`REVIEW_ONLY` is reported honestly rather than faked green — that is FR-008, not a gap.
