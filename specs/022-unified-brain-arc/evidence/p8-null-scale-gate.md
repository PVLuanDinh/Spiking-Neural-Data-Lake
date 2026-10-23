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
2. 2026-10-23, same point (no lane started, no data). Two more `retest` values, allowed inside `null-scale-triage.md`
   ONLY: `CANDIDATE` (a past small-scale deletion the owner may pick for a re-test; no expiry, because no code is
   unwired by it) and `NOT-NEEDED: <harm|at-scale|restatement> <reason, >= 20 chars>`. Elsewhere either value is
   R-FIELD.
3. 2026-10-23, same point. This file quotes the owner's words, and line 4 matches the delete-on-null regex: it states
   the RULE, not a deletion. R-UNRECORDED therefore skips exactly two files BY NAME, this pre-registration
   (`p8-null-scale-gate.md`) and the triage list (`null-scale-triage.md`, which addendum 1 already keeps free of quoted
   statements). The exclusion list is a named constant, printed on every `--assert` run, and a test asserts that a
   statement placed in any other evidence file still reds. R-FIELD and the record checks still read both files.
4. 2026-10-23, COUNTED LANDING OUTPUT (section 2), before the POLICIES.md text lands. Run on the stacked head (origin/main
   cb294e57 + this arc's prereg, triage list, gate and dormancy commits), `python -m snn_data_lake.gates.null_scale_gate
   --assert`: files scanned 128; statements 35; grandfathered 34; NULL-RECORDs 13 (all in the triage list); RETEST-RECORDs
   0; R-* reds 0; three STRUCTURAL arguments printed for review; rc 0. **The gate's count (34 grandfathered) wins over the
   spec's 31.** Line by line, the census of 2026-10-23 ran at 2ff11373; three statements arrived afterwards with the
   INSULA arc's evidence files: `insula-level-arms-production.md:171`, `insula-p8-engine-reach.md:27` and `:75`. The 35th
   statement is this file's own line 4, excluded by name (addendum 3). The lane `p8-triage` counted the same 35 by an
   independent census. Red-first record: 73 of 82 tests failed against a stub before the gate existed; 19 mutants, all
   killed (`lanes/p8-gate-1023-row/logs/`).
5. 2026-10-23, PANEL FOLD (item #4498: red team #4519, integration #4520, policy #4528) and an owner ruling.
   - **Owner ruling on B-M1 (verbatim):** "B-M1: strict. No deletion without the 10x re-test: when a scale-retest entry
     expires, the next arc's battery stays red until a RETEST-RECORD (or a checked STRUCTURAL argument) exists. Remove the
     'delete it with not re-tested at scale disclosed' exit." POLICIES item 3 now says so; the gate already reds an expired
     entry until it resolves, so no gate change follows from the ruling.
   - Gate-code findings go to lane `p8-gatefold` (thread P8-GATEFOLD): statements matched per paragraph (A-M1, which also
     changes the census and the grandfather keys); expiry dated from the commit that added the record, not the written sha
     (A-M2); R-EXEMPT must name a file holding a PENDING record (A-M3); merge commits do not expire an entry (A-M4); a
     deletion whose re-test charged reds (A-M5); setext headings, with the spec-literal section scope disclosed (A-M6);
     `NOT-NEEDED: harm` allowed in every file per owner answer 3 (B-M2); a closing record is a commit that modifies
     `tools/policy_certification.json`, not a subject string (B-M3); verb variants added to the statement regex (A-m1).
     Section 0's regex and section 1's R-UNRECORDED text are AMENDED by those items; their final counts go in addendum 6.
   - Integrator folds: full-history checkout on every CI job that runs the gate (self-checks, test, certify; A-M4,
     C-M1); POLICIES item 2 drops the seeds axis and item 4 restores "until a run with enough seeds is done" (B-M4);
     triage block 11 (Oja) back to CANDIDATE because L1b set no pre-registered harm bar (B-M5); numeric units for the
     units-axis candidates (B-M6); underpowered notes on W1 and T069; doc_budget ceiling +103 measured.
   - The write set of section 4 also included `src/snn_data_lake/gates/doc_budget_gate.py` and `tools/policy_manifest.json`
     (C minor-3; both were disclosed in their commit messages).
   - Disclosed, not fixed: one NULL-RECORD covers every statement in its section, and a section runs to the next heading of
     the same or higher level (spec-literal, printed in the gate's LIMIT line); the battery's `run_selfchecks` runs the gate
     bare, while `--self-check` runs in the lint CI job (C minor-4, the policy_manifest precedent).
6. 2026-10-23, GATE FOLD COUNTS (lane `p8-gatefold`, commit "null_scale_gate panel fold F1-F10", task P8-GATEFOLD scored
   total=1) plus one integrator fold, before the re-panel.
   - **Final `--assert` on the folded head:** files scanned 128; statements 34; grandfathered 33; NULL-RECORDs 14 (all in
     the triage list); RETEST-RECORDs 0; R-* reds 0; three STRUCTURAL arguments printed; rc 0. These replace addendum 4's
     35 / 34 / 13, which predate the folds.
   - **Section 0's regex, as amended (F10):** `(delet(ed|ion|e|ing|es)|retir(ed|e|ing|es)|remov(ed|e|es|ing|al)) on
     (its|a|the|their) (own )?null`, case-insensitive. It found no new statement in the tree (census unchanged by F10).
   - **Section 1's R-UNRECORDED, as amended (F1 + the integrator fold):** the regex runs over each prose paragraph
     (consecutive non-blank lines that are not heading or fence lines, joined by single spaces, located at its first line),
     and over each heading line and each fenced line on its own. The integrator fold restores the last half: F1 alone
     dropped headings and fenced lines, so `t069-cerebellar-rule.md:109` (a heading that records a deletion) left the
     census; four tests were red on F1 and are green on the fold (`lanes/p8-nullscale-1023-row/R1-heading.red.txt`).
     Known limit, disclosed: a numbered or bulleted list with no blank lines is ONE paragraph, so editing any item of such a
     list rewords its key (R-UNRECORDED plus R-STALE-GRANDFATHER until the key is regenerated in the same commit); a heading
     or fenced statement wrapped across lines is still missed, as before the join.
   - **New statement found by the paragraph census:** one, `trajectory-audit-2026-09-01.md:558-559` (a plan item wrapped
     across two lines, keyed on its list paragraph at :454); triage block 36, class restatement. Census arithmetic: the
     line census's 35 statements, minus two pairs that now share one paragraph (`l-exit.md` blocks 11+12, `s5-paging-null.md`
     blocks 26+27), plus that one = 34. The triage list keeps line keys; `check_triage.py` (line census) stays OK at 35
     statements, 36 blocks.
   - **GRANDFATHERED diff (e6e10b0c -> folded head):** 34 -> 33 keys; kept 4 (`l1b-synaptic-online.md`@8a7ac8989836,
     `s3-neuron-pager.md`@6ef10f902e5c, `s4-lake-compactor.md`@0cc96a9e8077, `t069-cerebellar-rule.md`@948a3b350b92); 30
     removed and 29 added, every one a re-key of the same statement onto its joined paragraph (full list in
     `lanes/p8-gatefold-1023-row/logs/F1.keydiff.txt` plus `lanes/p8-nullscale-1023-row/R1-regen.txt`).
   - **Other semantics the folds changed (each with red-first tests and a killed mutant, `lanes/p8-gatefold-1023-row/`):**
     R-EXPIRED dates an entry from the first commit that added its record (`git log -S`), counts closing records from that
     commit INCLUSIVE, treats a closing record as a non-merge commit that modifies `tools/policy_certification.json`, and
     ignores merge commits (F2, F4, F8); R-EXEMPT requires the named path to hold a `PENDING since` record (F3); a new code
     R-RETEST-CHARGED reds a deletion whose re-test charged (F5); setext headings bound sections (F6);
     `NOT-NEEDED: harm` is allowed in every in-scope file (F7).
   - **Panel MINORs still open, disclosed for the closing record:** false reds on a BOM, on bold or numbered record lines, on
     a `~~~` fence and on backslash paths; untracked files in the scan (red team m2, m3).
7. 2026-10-23, RE-PANEL FOLD (item #4561: red team #4562, policy #4565; lane `p8-gatefold2`, task P8-GATEFOLD2 scored
   total=1; three commits "null_scale_gate re-panel fold G1", "G2 G4 G5 G7", "G3 G6"), before the battery.
   - **Section 1 rule 5 (R-EXPIRED), as amended.** F2's creation-commit lookup is removed (it never expired a record written
     with other spacing or moved to another file, and falsely redded a record whose method appeared earlier in its file;
     real history reproduced the last). For `retest=PENDING since <S>` of method m (and a `scale-retest` entry of module x):
     S must be an ancestor of HEAD (missing history stays red); if the tree of S already holds a PENDING record of m (or a
     `scale-retest` row for x), the `since` was moved forward past its own record and reds; otherwise the entry expires at
     the arc's closing record C: a non-merge commit in S..HEAD that modifies `tools/policy_certification.json` with
     "closing" in its subject, counted only when a non-merge commit descends from C on the ancestry path to HEAD. The
     mid-arc re-certifications 0a5e4b13 and 4d5be853 are not closing records (real-history test: since=4be0d691 is green at
     its arc's closing record fbb1f38e and red at the next arc's first commit). A `since` must therefore be the arc's base
     (the previous closing record) or later; POLICIES' mechanization sentence says the same.
   - **Section 0 / rule 4:** the re-test axis is `units` or `ticks`; `axis=seeds` is R-FIELD (POLICIES item 2 and addendum 5
     dropped the seeds axis; the gate now enforces it).
   - **R-RETEST-CHARGED** reds a delete-on-null statement when ANY in-scope RETEST-RECORD with the same method, `of=` its
     file and `result=charges` exists, whatever the null's own retest value. **R-EXEMPT** additionally requires the named
     file to hold a record whose retest is `PENDING since` the entry's own `since`.
   - **R-UNRECORDED reads** prose paragraphs with leading block-quote markers stripped and the statement regex run with `*`
     and `_` removed (keys stay on the joined text). This re-keyed two existing block-quoted statements (`s3-neuron-pager.md`
     6ef10f902e5c -> ff26b6b3cdaf, `s4-lake-compactor.md` 0cc96a9e8077 -> 5b5edb284961) and found no new statement.
   - **Every `NOT-NEEDED: harm` record is printed for review**; the gate cannot judge whether the harm passed its bar.
   - **Counts on the folded head:** files scanned 130 (the learning-rules evidence file and OG1's evidence file joined the
     scope); statements 34; grandfathered 33; NULL-RECORDs 14; RETEST-RECORDs 0; R-* reds 0; three STRUCTURAL arguments and one
     harm record printed; rc 0. The 13 -> 14 NULL-RECORD change since addendum 4 is the triage fold that split block 11 (Oja
     harm kept, BCM moved to U4).
   - **Restack (panel C minor-1).** The arc was restacked onto int3 so every commit is green alone; the integrator's
     per-commit check (brain_atlas, policy_manifest --assert, dormancy_gate, doc_budget_gate, citation_guard,
     doc_command_guard, this gate and its tests when present, test_policy_manifest when POLICIES/manifest/CI changed, ruff on
     changed .py files) found three reds in a first restack and none in the second: the gate and its dormancy consumer are one
     commit with the gate's REGIONS lines; the live-tree test's NULL-RECORDs == 0 pin is dropped where the triage list already
     carries records; the POLICIES claim carries its CI steps. Old -> new sha map: the closing record lists it.
   - **Known limits, disclosed:** a fenced same-method PENDING example in the tree of S would read as a forward edit (none
     exists); a shallow clone reds every PENDING entry; a heading or fenced statement wrapped across lines is still missed;
     the panel MINORs listed in addendum 6 remain.
8. 2026-10-23, RE-PANEL 2 FOLD (item #4625: red team p8-rp2-a #4627, policy + integration p8-rp2-b #4626), before the
   battery. The restack was found clean (14 of 18 pairs patch-id identical, the 4 others exactly the named surgeries, the
   whole-arc patch-id equal); the per-commit check covers all 23 commits, each green alone.
   - **R-EXPIRED, completed (rp2-a M1).** Addendum 7's rule read only the tree of `since`, so an expired entry survived
     the next arc when one intermediate commit removed the record, flipped it to STRUCTURAL or moved it out of scope and
     the next restored it PENDING since that commit. The gate now also reads the tree of the newest closing record HEAD is
     past: a record (or `scale-retest` row) already PENDING there has ridden an arc and is expired, whatever `since` says.
     Integrator fold, red first (6 tests red on the previous head); rp2-a's own 40-case suite re-run against it: the five
     laundering cases now red, every case rp2-a held stays as rule. The closing subject is matched as the WORD "closing";
     backticks join `*` and `_` as emphasis stripped before the statement regex.
   - **Addendum 7 corrected (rp2-b m6):** POLICIES' mechanization sentence says `since` must predate its own PENDING
     record; "base or later" is this pre-registration's reading of item 3's "the arc's base on origin/main", not a POLICIES
     text. **Section 0's grammar** still lists `axis=<units|ticks|seeds>`; it is frozen, and addendum 7 governs it (seeds is
     R-FIELD).
   - **Dispositions of the remaining MINORs (no change; each stays review-enforced):** a closing-record subject without
     the word "closing" expires nothing (fail-open; every closing record since 2026-10-14 says "closing execution record",
     and the lane playbook names it); a closing record made as a merge commit, or lost in a squash merge, is unseen (not this
     repository's style: 0 of the last 300 commits); a next arc forked at the old base without the closing record passes
     until it is integrated; a second closing record in the same arc reds that arc's own entries at the second record
     (the protocol commits the record last, after every review); a method resolved by REF and then given a new PENDING
     record reads as expired at the next arc (fail-closed); R-EXEMPT ties the entry to its record by `since`, not by method;
     HTML emphasis inside the phrase and a respelled method escape; any non-merge commit after a closing record starts the
     next arc, so a riding entry reds at it (item 3's "expires before the next pushed arc", read strictly).
