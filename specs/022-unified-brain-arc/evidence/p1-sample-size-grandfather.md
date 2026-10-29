# P1 amendment: the grandfathered pre-registrations (the CLOSED list, named before the gate)

The P1 amendment's gate specification (`docs/POLICIES.md`, "P1 amendment", Gate specification) says the
pre-registrations frozen before the amendment landed are "a CLOSED list, grandfathered by name when the gate is
built". This file names that list now, so membership is fixed before the gate exists (policy-enforcer m2, board #4979).
It lists; it judges no file. A file on this list is exempt from the "Sample size (P1)" check only for the text it held
at the tree below; a pre-registration section added to it later is not grandfathered.

**Tree:** origin/main `c9d0c097` (the amendment's landing; closing execution record of engine arc 1).

## 1. How the list is found (reproducible; run from any checkout that has `c9d0c097`)

R1, a markdown heading that names a pre-registration (whole files and frozen sections):

```
git grep -n -i -E "^#+ .*(pre-?registration|prereg)" c9d0c097 -- '*.md'
```

R2, a pre-registration in an evidence file that declares itself in body text, not in a heading (the files R2 finds
that R1 does not):

```
git grep -l -i -E "pre-?regist|prereg" c9d0c097 -- 'specs/*/evidence/*.md'      # minus the R1 files
```

R1 gives 53 files (79 heading lines), R2 adds 23 evidence files: **76 files in all**. R1's heading lines include
sub-sections about a pre-registration (its deviations, a defect, a verdict "in its words"); they are listed with their
file, not separately counted. Mentions of the word outside `specs/*/evidence/` with no heading (plans, task lists,
`CHANGELOG.md`, `docs/POLICIES*.md`: 34 files, `git grep -l ... c9d0c097 -- '*.md'` minus R1 minus R2) are references
to pre-registrations, not pre-registrations, and are not on the list.

## 2. Controls

- **Positive:** R1 finds G3 T0 (`g3-t0-baseline.md` L1, `# G3 T0: baseline capture ... -- PRE-REGISTRATION`,
  frozen at `f1c59aa0`) and OG1 A8 (`og1-organoid-stream.md` L713, `## Amendment A8, 2026-10-23 (Stage 0b
  pre-registration ...; FROZEN)`, frozen at `f6fcc802` on main, cherry-picked from `553ad9c8`, the sha #4979 names). Both are rows below.
- **Negative:** the insula Stage 0b pre-registration, `specs/022-unified-brain-arc/evidence/insula-p8-sham-0b.md`,
  lives on branch `claude/insula-sham-1023` (`586fe1b5`), not in the tree: `git ls-tree -r --name-only c9d0c097 |
  grep -c sham-0b` prints 0. The instrument would see it: R2 run on that branch
  (`git grep -l -i -E "pre-?regist|prereg" claude/insula-sham-1023 -- 'specs/*/evidence/*.md' | grep -c sham-0b`)
  prints 1. So it is absent from this list because it is absent from the tree, and it is not grandfathered: it was
  frozen after the amendment, and a merge of its branch does not add it here (the closed list is by name, not by
  ancestry).

## 3. The list (74 grandfathered = 71 rows below + the 3 executed rows of 4b; two DRAFT rows moved to section 4)

| # | file | rule | grandfathered text (R1: the heading lines at `c9d0c097`) |
|---|---|---|---|
| 1 | `docs/AGENT_COLD_START.md` | R1 | L650 `### 8.8 Pre-registration: what an invented method owes, operationally` |
| 2 | `specs/011-h01-neurogenesis-prior/decisions.md` | R1 | L128 `## RE-DECISION 2026-09-09 (T063 pre-registration §2.12) — read this before either section above` |
| 3 | `specs/022-unified-brain-arc/evidence/c0-everyday-battery.md` | R1 | L40 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 4 | `specs/022-unified-brain-arc/evidence/g0a-chat-correction.md` | R1 | L32 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 5 | `specs/022-unified-brain-arc/evidence/g0b-reply-consumes-route.md` | R1 | L42 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 6 | `specs/022-unified-brain-arc/evidence/g0c-learner-persists.md` | R1 | L52 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 7 | `specs/022-unified-brain-arc/evidence/g1a-causal-contract.md` | R1 | L79 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 8 | `specs/022-unified-brain-arc/evidence/g1b-pool-selection.md` | R1 | L111 `## 2. Pre-registration (frozen at the commit that lands the final text)` |
| 9 | `specs/022-unified-brain-arc/evidence/g1c-carried-state.md` | R1 | L1 `# G1(c) / T109 part (c) -- carried state through the selected pool (pre-registration and record)`; L47 `## 2. Pre-registration (frozen at the commit that lands this text; later changes are dated amendments made ...` |
| 12 | `specs/022-unified-brain-arc/evidence/g3-t0-baseline.md` | R1 | L1 `# G3 T0: baseline capture for the one-grower merge -- PRE-REGISTRATION` |
| 13 | `specs/022-unified-brain-arc/evidence/g4-wiring-prior.md` | R1 | L1 `# G4 / T112 -- the anatomical wiring prior: the H01 distance kernel first (G7's T3), then the attachment ru...`; L72 `## 2. Pre-registration: T3 -- does H01 favour a Gaussian distance kernel?`; L407 `## 3. Parts B-E (outlines; each owes a dated pre-registration amendment before its first run)`; L792 `### Dated pre-data amendment 2026-10-07 (c) to the Part B amendment -- the panel's folds (Codex gpt-6.1-sol...` |
| 14 | `specs/022-unified-brain-arc/evidence/g5-handoff-repair-2026-10-11.md` | R1 | L9 `## Preregistration and rejected plan` |
| 16 | `specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md` | R1 | L1 `# G6 / T114 -- sleep and durable memory (pre-registration, wave 1; frozen by the lead's P0 commit; section ...`; L77 `## 2. Pre-registration` |
| 17 | `specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md` | R1 | L1 `# G7 / T115 -- the unit-label gate (P4d's provenance check, red G-R7) and the designs of T1 and T4 (pre-reg...`; L61 `## 2. Pre-registration` |
| 18 | `specs/022-unified-brain-arc/evidence/g8-reorganization.md` | R1 | L1 `# G8 / T116 -- the repository reorganized by engine function (pre-registration)`; L575 `## 2. Pre-registration (PRE-REGISTERED -- FROZEN 2026-09-25 at the design read's fold, the commit that land...`; L3205 `### 3.27 G8-3 pre-registration: M11a-M11e and D1 (Claude, work loop, 2026-10-02)` |
| 19 | `specs/022-unified-brain-arc/evidence/gates-for-the-ratified-rows.md` | R1 | L1 `# The gates for X1, L5 and C5 (2026-09-06) — pre-registration`; L64 `## 2. Pre-registration (written before any gate code)`; L251 `### 3.4 Deviations from the pre-registration, disclosed` |
| 20 | `specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md` | R1 | L50 `## 2. Pre-registration (written before any cell runs; the build follows it)` |
| 21 | `specs/022-unified-brain-arc/evidence/l1c-online-readout.md` | R1 | L47 `## 2. Pre-registration (written before any cell runs; the build follows it)` |
| 22 | `specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md` | R1 | L31 `## 2. Pre-registration`; L339 `### 6.3 A defect in the pre-registration itself, recorded because it nearly decided this` |
| 23 | `specs/022-unified-brain-arc/evidence/l4-resume-and-the-append-ceiling.md` | R1 | L84 `## 6. Pre-registration: AppendBudget's first live caller` |
| 24 | `specs/022-unified-brain-arc/evidence/l4b-append-ceiling-nonstationary.md` | R1 | L82 `## 5. Pre-registration` |
| 25 | `specs/022-unified-brain-arc/evidence/l6-immortal-vs-consolidation.md` | R1 | L28 `## 2. Pre-registration` |
| 26 | `specs/022-unified-brain-arc/evidence/og1-organoid-stream.md` | R1 | L713 `## Amendment A8, 2026-10-23 (Stage 0b pre-registration: the sorted 10x re-test source; FROZEN)`; L780 `### A8.2 Stage 1 clause (P8 item 2; the integrator authored A4.9 and the Stage 1 pre-registration; this pan...` |
| 27 | `specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md` | R1 | L1 `# P8 null-scale amendment: gate pre-registration (frozen before the gate is built)` |
| 28 | `specs/022-unified-brain-arc/evidence/q2-surprise-context.md` | R1 | L127 `## 3. Results — 12 cells, 3000 ticks, 20:17–20:29 UTC, at f1a9bf8 (the pre-registration commit's head;` |
| 30 | `specs/022-unified-brain-arc/evidence/r2-typed-replay-hydration.md` | R1 | L1 `# R2 / T103 -- sleep replay hydrated from typed lake records through the incumbent interaction learner (DRA...`; L76 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 32 | `specs/022-unified-brain-arc/evidence/s1-unit-blocks.md` | R1 | L30 `## 2. Pre-registration (written before unit_blocks.py existed)` |
| 33 | `specs/022-unified-brain-arc/evidence/s2-exit.md` | R1 | L16 `## 0. Pre-registration -- the rules, written and COMMITTED BEFORE any command below was run` |
| 34 | `specs/022-unified-brain-arc/evidence/s3-neuron-pager.md` | R1 | L88 `## 2. Pre-registration (written before neuron_pager.py existed)`; L381 `### 3.4 Deviations from the pre-registration, each with its reason` |
| 35 | `specs/022-unified-brain-arc/evidence/s4-lake-compactor.md` | R1 | L99 `## 2. Pre-registration (written before lake_compactor.py existed)`; L390 `### 3.4 Deviations from the pre-registration, each with its reason` |
| 36 | `specs/022-unified-brain-arc/evidence/s5-paging-null.md` | R1 | L82 `## 2. Pre-registration (written before the run)`; L461 `### 3.4 The verdict, in the pre-registration's words`; L529 `### 3.6 Deviations from the pre-registration, each with its reason` |
| 37 | `specs/022-unified-brain-arc/evidence/s6-kernel-wire.md` | R1 | L68 `## 2. Pre-registration (written before the instrument exists)`; L419 `### 3.5 Deviations from the pre-registration, each with its reason` |
| 38 | `specs/022-unified-brain-arc/evidence/sc-012a-width.md` | R1 | L43 `## 2. Pre-registration (written before any criterion arm ran; section 2.5 written before any arm ran too)` |
| 39 | `specs/022-unified-brain-arc/evidence/scoring-path-abstention.md` | R1 | L23 `## 2. Pre-registration (written before the change)` |
| 40 | `specs/022-unified-brain-arc/evidence/t062-h01-ingest.md` | R1 | L69 `## 2. Pre-registration`; L486 `### 3.5 Deviations from the pre-registration, each with its reason (dated 2026-09-09)` |
| 41 | `specs/022-unified-brain-arc/evidence/t063-prior-drift-harness.md` | R1 | L46 `## 2. Pre-registration`; L480 `### 3.5 Deviations from the pre-registration, each with its reason (dated 2026-09-09)` |
| 42 | `specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md` | R1 | L40 `## 2. Pre-registration (written before cf_ltd_rule.py existed)` |
| 43 | `specs/022-unified-brain-arc/evidence/t070-integration-kernel.md` | R1 | L57 `## 2. Pre-registration`; L207 `### defect of this pre-registration, and its fix (the same statistic, a different algorithm, equality-tested)` |
| 44 | `specs/022-unified-brain-arc/evidence/t077-metacognition.md` | R1 | L60 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 45 | `specs/022-unified-brain-arc/evidence/t078-attention-schema.md` | R1 | L77 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 46 | `specs/022-unified-brain-arc/evidence/t079-agency.md` | R1 | L111 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 47 | `specs/022-unified-brain-arc/evidence/t080-temporal-binding.md` | R1 | L69 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 48 | `specs/022-unified-brain-arc/evidence/t081-report-order.md` | R1 | L87 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 49 | `specs/022-unified-brain-arc/evidence/t082-social-cognition.md` | R1 | L68 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 50 | `specs/022-unified-brain-arc/evidence/t090-conclusion-plate.md` | R1 | L43 `## 2. Pre-registration (frozen at the commit that lands this file)`; L237 `### 3.2 PRE-REGISTRATION OF THE SECOND READING (2026-09-16 23:18 local, clock read at write time), frozen B...` |
| 51 | `specs/022-unified-brain-arc/evidence/t101-brian2-contract.md` | R1 | L55 `## 2. Pre-registration (frozen at the commit that lands this file)` |
| 52 | `specs/022-unified-brain-arc/evidence/trainer-retirement.md` | R1 | L1 `# The four backprop trainers: port the capability, then retire (pre-registration and record)` |
| 53 | `specs/022-unified-brain-arc/quickstart.md` | R1 | L30 `## S2 — H01 prior + paging *DATED NOTE 2026-09-09: ...* [the note's delete-on-null wording is elided here: a quote of it would read as a statement to the P8 null-scale gate; see the file]` |
| 54 | `specs/022-unified-brain-arc/evidence/docs-only-battery.md` | R2 | whole file |
| 55 | `specs/022-unified-brain-arc/evidence/encoder-path-root.md` | R2 | whole file |
| 56 | `specs/022-unified-brain-arc/evidence/engine-structure-2026-09-25.md` | R2 | whole file |
| 57 | `specs/022-unified-brain-arc/evidence/g7-organizing-unit.md` | R2 | whole file |
| 58 | `specs/022-unified-brain-arc/evidence/insula-level-arms-production.md` | R2 | whole file |
| 59 | `specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md` | R2 | whole file |
| 60 | `specs/022-unified-brain-arc/evidence/jepa-collapse.md` | R2 | whole file |
| 61 | `specs/022-unified-brain-arc/evidence/jepa-live-floor.md` | R2 | whole file |
| 62 | `specs/022-unified-brain-arc/evidence/l-exit.md` | R2 | whole file |
| 63 | `specs/022-unified-brain-arc/evidence/l-parallel-arms.md` | R2 | whole file |
| 64 | `specs/022-unified-brain-arc/evidence/l4c-ignition-is-coalition-size.md` | R2 | whole file |
| 65 | `specs/022-unified-brain-arc/evidence/learning-rules-1019.md` | R2 | whole file |
| 66 | `specs/022-unified-brain-arc/evidence/null-scale-triage.md` | R2 | whole file |
| 67 | `specs/022-unified-brain-arc/evidence/s1-genome-fanin-ab.md` | R2 | whole file |
| 68 | `specs/022-unified-brain-arc/evidence/s1-placement-null.md` | R2 | whole file |
| 69 | `specs/022-unified-brain-arc/evidence/s2-real-percept-stream.md` | R2 | whole file |
| 70 | `specs/022-unified-brain-arc/evidence/s2b-growing-learner-lake-record.md` | R2 | whole file |
| 71 | `specs/022-unified-brain-arc/evidence/s2c-rotating-stream-refused.md` | R2 | whole file |
| 72 | `specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md` | R2 | whole file |
| 73 | `specs/022-unified-brain-arc/evidence/w-exit.md` | R2 | whole file |
| 74 | `specs/022-unified-brain-arc/evidence/w1-goal-curriculum.md` | R2 | whole file |
| 75 | `specs/022-unified-brain-arc/evidence/w2-context-latents.md` | R2 | whole file |
| 76 | `specs/022-unified-brain-arc/evidence/w3-world-baselines.md` | R2 | whole file |

## 4. Owes a "Sample size (P1)" section at freeze (2; NOT grandfathered)

Arc-2 panel, lens B MAJOR-5 (board #5196), fixed by the integrator 2026-10-24 (G2 has since moved to 4b, see its row): P1 grandfathers only pre-registrations
"frozen before this amendment landed". These two rows declare themselves DRAFT / not frozen in their own text at
`c9d0c097`: G3 has not run, and G5's Parts A-D ran (2026-10-07) but its Part E was added as a DRAFT on 2026-10-27, so
freezing them later with their c9d0c097 text would have let the next engine rows run with a habitual seed count. Each must carry the
section's fields when it freezes. The gate's deferral date for each is a forcing date,
renewable only by a dated integrator commit giving the reason (each renewal an enforcer-reviewed build, D-74). The gate's reds (when built) add this case: a pre-registration on this list frozen
without the section reds. Row numbers are those of the original 76-file list.

| # | file | rule | text at `c9d0c097` |
|---|---|---|---|
| 11 | `specs/022-unified-brain-arc/evidence/g3-growth-loop.md` | R1 | L1 `# G3 / T111 -- growth where G2's capacity cell licenses it (DRAFT pre-registration, wave 1; not frozen)`; L60 `## 2. Pre-registration (DRAFT; frozen by the orchestrator; later changes are dated strengthenings before an...` |
| 15 | `specs/022-unified-brain-arc/evidence/g5-readout-variants.md` | R1 | L1 `# G5 / T113 -- readout variants against the incumbent OnlineReadout (pre-registration DRAFT)` |

### 4b. Executed rows whose heading still says DRAFT (3; grandfathered)

policy-enforcer #5319 m1: r1 and r3 were first moved to section 4 by the same DRAFT grep, but both say "DRAFT; becomes
frozen at the commit that lands it" and both landed and ran (each file has its Results section; R3 / T104 is ticked, R1 carries
a 2026-10-15 pre-data amendment), so they were frozen before this amendment and stay grandfathered; the DRAFT heading is
stale history.

| # | file | rule | text at `c9d0c097` |
|---|---|---|---|
| 29 | `specs/022-unified-brain-arc/evidence/r1-bounded-recall.md` | R1 | L1 `# R1 / T102 -- bound historical recall (pre-registration DRAFT and record)`; L97 `## 2. Pre-registration (DRAFT; becomes frozen at the commit that lands it; later changes are dated strength...` |
| 10 | `specs/022-unified-brain-arc/evidence/g2-taught-problems.md` | R1 | moved here 2026-10-29 (policy-enforcer #6592, integrator ruling D-73): grandfathered because it was frozen before the landing, shown by execution (a dated read in its own text); a commit reference alone is not evidence of a freeze. The read: section 5.1 is a confirmatory read, 2026-10-07; the commit 6bc97430 (2026-10-03) is history. G3 stays in section 4: never run, every amendment PRE-DATA. |
| 31 | `specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md` | R1 | L1 `# R3 / T104 -- eligible replay composed with the incumbent learner and grower; earned growth on held-out da...`; L84 `## 2. Pre-registration (DRAFT)` |
