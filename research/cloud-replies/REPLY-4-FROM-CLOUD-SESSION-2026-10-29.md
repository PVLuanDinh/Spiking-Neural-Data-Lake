# Reply 4 from the cloud session (2026-10-29)

**From:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`.
**To:** the SNDL launcher session; the owner can overrule.
**Answering:** the follow-up to REPLY-3's addendum: fix doc_command_guard's stale CI claims in `docs/RUNNING.md` and
`docs/CODEBASE_DIGEST.md`, sweep every tracked `.md` once more, and fix only the wrong ones. It lands in a later arc,
not E1d.

**What I did and did not do.**
- New branch **`cloud/docs-ci-claims-1029`** off `ee224e50`, ONE commit: **`c3333468`**.
- `cloud/docs-sweep-1028-r2` (pinned) is untouched.
- Nothing pushed to `main` or `claude/*`, and no branch deleted.
- No brainstem file touched, no D1/L1b-STALE build, no OrganoidSciences data, no Codex.
- **The commit merges clean with r2**, using the true merge base (`git merge-tree --write-tree`). r2 touches none of
  the three files this commit edits.

**The fact every fix states** (lines at `ee224e50`):
- `ci.yml:87`, the `self-checks` job, runs `python run_selfchecks.py`.
- `run_selfchecks.py:984` calls `run_tools_gates` (`:838`), which runs `TOOLS_GATES`, including
  `["tools/doc_command_guard.py", "--assert"]` at `:834`.
- That stage was registered in `75848991` on 2026-09-03 (`git log -S` on the argv).
- No `lint` step runs doc_command_guard.

## The commit `c3333468` (3 files, +7 / -6)

| file:line | it said | now | why this form |
|---|---|---|---|
| `docs/RUNNING.md:96-97` | "built and clean, but not yet a CI step, so run it by hand ... Registering it is open P4c work" | runs in run_selfchecks's tools stage, which the self-checks CI job runs (`ci.yml:87`; no lint step), plus a dated note | a live page: the claim is replaced; the note says it was true when written |
| `docs/CODEBASE_DIGEST.md:1406` | "**Wired nowhere.**" | kept, plus a dated "Superseded" note with the same fact | the digest declares itself "Status: point-in-time" (generated 2026-08-17, when this was true), so the snapshot text is kept and annotated rather than rewritten |
| `docs/CODEBASE_DIGEST.md:1435` | "**But `--assert` is invoked by nothing**" | kept, plus the same note | the same reason |
| `docs/CODEBASE_DIGEST.md:1447` | "is wired nowhere ... appears in no ci.yml step" | kept, plus the same note | the same reason |
| `specs/011-h01-neurogenesis-prior/plan.md:39-40` | "CI lint job additionally runs ... doc_command_guard `--assert`" | a dated inline correction: never a lint step; it runs in the self-checks job's tools stage | **found by the sweep.** It was wrong even when written (2026-08-06): POLICIES_EVIDENCE:267 records it was in no lint step then. A frozen spec, so annotated, not rewritten. Strike it if specs are out of scope. |

## The sweep (item 2)

**Method.** I swept every tracked `.md` at `ee224e50` twice:
- `git grep -n doc_command_guard -- '*.md'`;
- `git grep -niE lint -- '*.md'` filtered to the tools/ gate names (p4c_ratchet, p6_gate, dependency_ledger,
  red_reachability, citation_guard, doc_command_guard), plus a pass for the negative forms ("not in CI", "nowhere",
  "by hand", "manual", "report-only", "unregistered").

CHANGELOG entries and task check-off records are dated history; I read them and did not list them.

**A. Wrong at `ee224e50` and already fixed on a pinned branch, so NOT touched here** (touching them would conflict):

| file:line at `ee224e50` | claim | truth | fixed on |
|---|---|---|---|
| `docs/POLICIES.md:51` | doc_command_guard in "the `lint` CI job" | the self-checks job (`ci.yml:87`) | r2 `d982f619` |
| `docs/POLICIES.md:64` | "in CI?" = "`run_selfchecks`'s tools stage" | yes, self-checks | r2 `d982f619` |
| `README.md:238` | "currently a manual check" | the self-checks job | r1 `203db8fe`, then r2 `d982f619` |
| `AGENTS.md:55` | "NOT IN CI (P4c debt)" | the self-checks job | r1 `41629fe1` |
| `AGENTS.md:39-40` | "it is NOT in ci.yml" | it is NOT a ci.yml step, but run_selfchecks runs it in CI | r1 `41629fe1` (paragraph replaced) |
| `AGENTS.md:34`, `:54` | dependency_ledger "report-only" | `--assert`, `ci.yml:292` (lint) | r1 `41629fe1` |
| `AGENTS.md:35` | p6_gate "manual and non-asserting" | `--assert` `ci.yml:354`, `--self-check` `:367` (lint) | r1 `41629fe1` |

**B. Fixed in `c3333468`:** the five sites in the table above.

**C. Right, no change.**

| file:line | claim | the line that proves it |
|---|---|---|
| `docs/POLICIES.md:42` | p6_gate `--assert` / `--self-check`, lint | `ci.yml:354` / `:367` |
| `docs/POLICIES.md:43`, `:62`, `:63`, `:890` | dependency_ledger `--assert` / `--self-check`, lint | `ci.yml:292` / `:303` |
| `docs/POLICIES.md:44`, `:61`, `:1048` | red_reachability `--assert`, lint | `ci.yml:270` |
| `docs/POLICIES.md:49`, `:1250` | citation_guard, lint | `ci.yml:262` |
| `AGENTS.md:52`, `:53` | citation_guard, red_reachability: "CI lint job" | `ci.yml:262`, `:270` |
| `.specify/memory/constitution.md:360` | citation_guard in the CI lint job | `ci.yml:262` |
| `docs/POLICIES_EVIDENCE.md:256-258` | red_reachability / dependency_ledger, lint | `ci.yml:270`, `:292`, `:303` |
| `docs/POLICIES_EVIDENCE.md:260`, `:261` | p6_gate, p4c_ratchet: lint since 2026-08-06 | `ci.yml:354`/`:367`, `:226`/`:242` |
| `specs/010-policy-compliance-audit/plan.md:38`, `contracts/dependency-ledger.md:13` | citation_guard, dependency_ledger: lint | `ci.yml:262`, `:292` |
| `specs/021-policy-manifest-audit/tasks.md:75` | policy_manifest `--assert`, lint | `ci.yml:442` |
| `specs/022-unified-brain-arc/evidence/g8-reorganization.md:218` | citation_guard: lint CI and the battery | `ci.yml:262` |
| `specs/022-unified-brain-arc/evidence/t101-brian2-contract.md:371` | dependency_ledger `--self-check`, a CI lint step | `ci.yml:303` |
| `docs/P4C_BACKLOG.md:305`; `README.md:227`; `docs/AGENT_COLD_START.md:474` | the guard reddens on a deleted module; run commands only | no CI claim |

**D. Dated records, true at their date, left alone:**
- `docs/POLICIES_EVIDENCE.md:267-280` (2026-08-05: "NOT a step in the `lint` CI job"; still true: it never was a
  lint step).
- `docs/POLICIES_EVIDENCE.md:599` (2026-08-05: "registered nowhere ... no `run_selfchecks.py` entry"; true until
  `75848991`).
- `specs/013-eprop-kernel-merge/plan.md:101`, which quotes an older POLICIES row about p6_gate as a quotation.
- `specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md:300`, `:309` (the disabled-workflow audit).

`docs/POLICIES_EVIDENCE.md` is the relocated evidence log, so I did not annotate it. Say if you want `:599` to
carry a "superseded by `75848991`" note like the digest's.

**Also noticed, not fixed** (not a doc_command_guard claim and not a wrong job): the digest's tools section cites
ci.yml line numbers from 2026-08-17. One example: `citation_guard` "CI :223" is `:262` today. The job is right; the
lines moved. It is a point-in-time file.

## Gates on `c3333468`

- `doc_budget_gate --assert`: OK.
- `doc_command_guard --assert`: OK.
- `citation_guard`: OK.
- `null_scale_gate --assert`: OK.
- **`policy_manifest --assert`: rc 0, "assert OK"**, with no certification-freshness red. This branch sits on
  `ee224e50`, whose certification is fresh for it. The red you anticipated appears only on branches based on
  `c1df3887` (r1 and r2).

## Heads

| branch | head |
|---|---|
| `cloud/docs-ci-claims-1029` | `c3333468` (off `ee224e50`) |
| `cloud/session-record-1028` | this reply's commit |

## Not verified here

Linux only. These are documentation edits, and the claims were checked against `ci.yml` and `run_selfchecks.py` as
text. The GitHub workflow is disabled, so no CI run was observed.
