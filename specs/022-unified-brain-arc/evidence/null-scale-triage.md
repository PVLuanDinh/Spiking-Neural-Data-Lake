# P8 null-scale retroactive triage list (option B, one-time)

Lane p8-triage, 2026-10-23. Read-only research; nothing here restores or removes anything. The owner picks from the ranked table at the end.

Spec: `p8-null-scale-gate.md` sections 0-4 and addenda 1-3 (`unknown` allowed for units, ticks, seeds, mde; `retest=CANDIDATE` and `retest=NOT-NEEDED:` allowed here only). This file cites each statement by `path:line` and key and never quotes a statement's wording.

## Census

Regex of spec section 0 over `specs/**/evidence/*.md` at base 89f3af14 plus the pre-registration commits: **35 statements** (the spec expected 31; the difference is disclosed in REPORT.md). The pre-registration's own line (`p8-null-scale-gate.md:4`) is counted because the integrator's checker counts it, and is triaged as a restatement. Key = `<repo path>@<first 12 hex of sha256(stripped line)>`.

Integrator addendum, 2026-10-23 (after gate fold F1): the gate now reads paragraphs (headings and fenced lines kept one each), so its census is 34 statements; block 36 is the one statement it sees that the line census did not (wrapped across two lines). Blocks 1-35 keep their line keys; the gate's keys for multi-line paragraphs differ from them (pre-registration addendum 6 lists the change).

## Today's capacity per axis (the reference for 'more than 10x below')

| axis | capacity today | source |
|---|---|---|
| units | 16,384 units run through the brain in one 226-cell campaign; 14,215 signed units ingested (H01, T062); the engine's measured binding width is 222,193 units at a 64 MB budget | `s5-paging-null.md:403`; `s2-exit.md:104`; `s2-state-per-unit.md:55-58` |
| ticks | 9,600 ticks in one single-campaign run, which landed at 644.1 s (review #4486 m3; the lane's draft called it planned); kernel throughput 23,823 ticks/s | `t078-attention-schema.md:379-381`; `g1b-pool-selection.md:65` |
| seeds | S = 31 in one registered contrast (power-driven, not a scaling axis for most mechanisms) | `g1c-carried-state.md:74` |

Thresholds used: units below 1,638 (16,384 / 10); ticks below 960 (9,600 / 10). A candidate must fall under one of them on the axis named in its block. Seeds are never the axis alone.

## Blocks

### 1. specs/022-unified-brain-arc/evidence/c0-everyday-battery.md:7  key=specs/022-unified-brain-arc/evidence/c0-everyday-battery.md@f823f1fef028
class: restatement
evidence: Policy list in the arc preamble (c0-everyday-battery.md:5-8); the file measures an instrument and records no removal.

### 2. specs/022-unified-brain-arc/evidence/g5-readout-variants.md:690  key=specs/022-unified-brain-arc/evidence/g5-readout-variants.md@8b94a63cd542
class: restatement
evidence: Pre-registered disposition for Part D (g5:689-691): states which parts are owner-drawn and never removed; no Part D cell has run (Part D waits on G1(c)).

### 3. specs/022-unified-brain-arc/evidence/g5-readout-variants.md:1056  key=specs/022-unified-brain-arc/evidence/g5-readout-variants.md@4fe8589dc323
class: small-scale
NULL-RECORD: categorical coupling (counts-matrix relabelling prior on the readout) | units=256 | ticks=800 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Verdict at g5:1054-1058: B-R1 NOT MET (mean +0.0021, t +0.444) on confirmatory seeds 11100-11107; the module and arms were removed in the next commit. Scale: pool N=256 (g5:579), 800-tick cells (g5:77, g5:131), 8 seeds; the bar is a t-threshold, not an mde. AXIS = ticks: the counts matrix accumulates one event per readout error, so evidence grows with run length; 800 is 12x below the 9,600-tick capacity (units, 256 vs 16,384, is 64x below and agrees).

### 4. specs/022-unified-brain-arc/evidence/g5-readout-variants.md:1269  key=specs/022-unified-brain-arc/evidence/g5-readout-variants.md@3057ca1110fe
class: restatement
evidence: Part D.7 repeats the P4d rule (g5:1268-1269); no Part D result exists in this file.

### 5. specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md:243  key=specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md@b2b6465f1f03
class: restatement
evidence: Pre-registered outcome table for the simulated-annealing pruner (g6:241-244). The Results section says none measured (g6:846), so no removal happened.

### 6. specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md:371  key=specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md@622b1d1a619a
class: restatement
evidence: Section 2.9 lists what a G6 null would remove and what it keeps (g6:369-374); results are empty (g6:846), so nothing was removed.

### 7. specs/022-unified-brain-arc/evidence/insula-level-arms-production.md:171  key=specs/022-unified-brain-arc/evidence/insula-level-arms-production.md@2fbda974ff4c
class: restatement
evidence: Owner adjudication paragraph (insula-level-arms-production.md:166-171) restates what P8 asks of the insula charge; it says nothing is measured yet.

### 8. specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md:27  key=specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md@b6838817cbf7
class: restatement
evidence: Orientation note (insula-p8-engine-reach.md:25-27) states which P8 branch applies to the insula; no result is attached to the line.

### 9. specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md:75  key=specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md@fd339a7f9e38
class: restatement
evidence: Pre-registered reading for REACH-NULL (insula-p8-engine-reach.md:73-75). The recorded reading is REACHES (line 119), so this branch did not fire and nothing was removed.

### 10. specs/022-unified-brain-arc/evidence/jepa-live-floor.md:614  key=specs/022-unified-brain-arc/evidence/jepa-live-floor.md@ceb93e50d738
class: harm
NULL-RECORD: reservoir build of the live latent floor (commit D, not landed) | units=unknown | ticks=unknown | seeds=3 | mde=unknown | retest=NOT-NEEDED: harm the planted k=1 order read failed on every seed, 1.61 / 5.87 / 18.9 at 10 cycles
evidence: R4 NOT MET at jepa-live-floor.md:603-605: the planted k=1 read fails in the in-tree test (seed 0) and on all three harness seeds; R5 also broke two self-checks (jepa-live-floor.md:606-611). The build was archived, not landed (jepa-live-floor.md:612-614). Why harm and not null (integrator fold of reviews #4485/#4486, MINOR): R4 is a pre-registered NON-REGRESSION read (jepa-live-floor.md:129, :203, the in-tree tests stay green at k=1), and the build turned one red while its reservoir read 1.61 / 5.87 / 18.9 where the incumbent ring buffer reads 0.069 (jepa-live-floor.md:450-452): a measured regression against the incumbent. A read that only misses its bar would be a null. Ticks and units are not stated in these lines.

### 11. specs/022-unified-brain-arc/evidence/l-exit.md:20  key=specs/022-unified-brain-arc/evidence/l-exit.md@6a99ed3ae91e
class: small-scale
NULL-RECORD: Oja local rule on the grown units' input rows (L1b) | units=38 | ticks=800 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Row L1b (l-exit.md:20) records Oja at t -1.96 vs frozen rows, 8 seeds x 800 ticks on the synthetic stream. Panel fold (#4528 B-M5): this is NOT a measured harm under POLICIES P8 amendment item 4 ("past its pre-registered bar in the harm direction"), because L1b's only pre-registered bar was "beats the control" (l1b-synaptic-online.md:113, :229) and the harm reading was a mirrored read (:266), the same ruling the E1 `_plan_w` signal got; its stated cause was later found wrong (:367, M-1). So Oja is a CANDIDATE, like BCM (U4). units=38 is the top of the recorded 2-38 grown-unit range (l2-grow-on-plateau.md:165), so a 10x re-test means at least 380 units (panel B-M6). AXIS = units.

### 12. specs/022-unified-brain-arc/evidence/l-exit.md:22  key=specs/022-unified-brain-arc/evidence/l-exit.md@7e97ad3d8373
class: small-scale
NULL-RECORD: plateau trigger inside the growth gate (L2) | units=38 | ticks=1200 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Row L2 (l-exit.md:22): outcome effect +0.625 units/seed at t 0.141, readout-neuter effect t 1.135; 8 seeds x 1200 ticks on SyntheticPerceptStream (l2-grow-on-plateau.md:256, :327). The record itself says the stream is near ceiling for the incumbent, so little exists for a trigger to prevent (ledger, 2026-09-06 entry). Grown units per seed run from 2 to 38 (l2-grow-on-plateau.md:165); units=38 records the top of that range so a 10x re-test means at least 380 units (panel #4528 B-M6). AXIS = units: the trigger acts on the unit count; 1200 ticks is only 8x below 9,600, so ticks does not qualify.

### 13. specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md:327  key=specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md@3a870391338a
class: restatement
evidence: Commit-table row (l1b-synaptic-online.md:327) naming the commit that removed the rule; the measurements are triaged at the l-exit row for L1b.

### 14. specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md:350  key=specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md@8a7ac8989836
class: restatement
evidence: Panel ledger header (l1b-synaptic-online.md:348-350) records that the three review lenses agreed the disposition stands; it adds no new removal.

### 15. specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md:4  key=specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md@9202bf9204a9
class: restatement
evidence: Preamble (l2-grow-on-plateau.md:3-4) points back at the L1b removal that changed this row's trigger.

### 16. specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md:9  key=specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md@719e8f98fb86
class: restatement
evidence: Territory paragraph (l2-grow-on-plateau.md:8-10) cites the L1b removal as why the row's wording no longer applies.

### 17. specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md:4  key=specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md@67680ff9a025
class: restatement
evidence: The pre-registration quotes the owner stating the rule (p8-null-scale-gate.md:3-5); this is the rule itself, which addendum 3 names as a rule-restating line.

### 18. specs/022-unified-brain-arc/evidence/r1-bounded-recall.md:109  key=specs/022-unified-brain-arc/evidence/r1-bounded-recall.md@9297ddb1d509
class: restatement
evidence: Pre-registered KEEP branch (r1-bounded-recall.md:108-110); the Results section says nothing recorded (r1:754).

### 19. specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md:36  key=specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md@0edb6ea1a6fe
class: restatement
evidence: Territory item 3 (r3-replay-growth-composition.md:36-37) cites the L2 removal as standing; no new removal.

### 20. specs/022-unified-brain-arc/evidence/s0-exit.md:176  key=specs/022-unified-brain-arc/evidence/s0-exit.md@d4914f4163e5
class: structural
NULL-RECORD: faculty-to-faculty workspace route (SC-001) | units=n/a | ticks=60 | seeds=1 | mde=unknown | retest=STRUCTURAL: the language subscriber's broadcast was byte-identical to the action one on 20 of 20 ticks and has zero call sites
evidence: s0-exit.md:150-176. The refusal rests on a code fact, not an effect size: the owning module's payload is listener-blind (s0-exit.md:171-174, byte-identical on 20/20 ticks, make_brain(seed=0), 60 wake ticks) and the only candidate subscriber has zero call sites (s0-exit.md:161-164). More ticks cannot change a payload that is computed once from the winner.

### 21. specs/022-unified-brain-arc/evidence/s0-exit.md:210  key=specs/022-unified-brain-arc/evidence/s0-exit.md@c326a826a698
class: small-scale
NULL-RECORD: interoceptive charge route into the workspace (T039b) | units=unknown | ticks=120 | seeds=unknown | mde=unknown | retest=CANDIDATE
evidence: s0-exit.md:196-210: the producer's max output 0.2798 (arousal) and 0.2142 (drives) sat below the bus admission floor 0.5505, and pinning it left the brain byte-identical on 120 of 120 ticks (s0-exit.md:197-205). Later evidence at 1000 ticks x 5 seeds found the insula reaching two deciding columns on 4 of 5 seeds (insula-p8-engine-reach.md:119-121, :164), which contradicts 'never competes' at a larger run. AXIS = ticks: reach is a rate over a run; 120 is 80x below 9,600.

### 22. specs/022-unified-brain-arc/evidence/s3-neuron-pager.md:3  key=specs/022-unified-brain-arc/evidence/s3-neuron-pager.md@6ef10f902e5c
class: restatement
evidence: Banner (s3-neuron-pager.md:3) points to s5-paging-null.md for the measured removal of the pager; the measurement is triaged at its s5 disposition block.

### 23. specs/022-unified-brain-arc/evidence/s4-lake-compactor.md:3  key=specs/022-unified-brain-arc/evidence/s4-lake-compactor.md@0cc96a9e8077
class: structural
NULL-RECORD: lake compactor and block tier (collateral of the pager removal) | units=n/a | ticks=n/a | seeds=unknown | mde=unknown | retest=STRUCTURAL: the compactor's pager-dependent halves had no non-pager consumer, checked by a positive-controlled grep
evidence: Banner at s4-lake-compactor.md:3. The compactor was removed because its merge and reorder had no consumer once the pager went (s5-paging-null.md:513-514; the consumer search with its positive control is at s5-paging-null.md:757). No effect size was measured for it, so no scale applies.

### 24. specs/022-unified-brain-arc/evidence/s4-lake-compactor.md:319  key=specs/022-unified-brain-arc/evidence/s4-lake-compactor.md@b198632d46ad
class: restatement
evidence: Pre-registered disposition table row (s4-lake-compactor.md:316-322); the compactor's later removal is triaged at the banner line.

### 25. specs/022-unified-brain-arc/evidence/s5-paging-null.md:158  key=specs/022-unified-brain-arc/evidence/s5-paging-null.md@a1ad953913a7
class: restatement
evidence: Pre-registered disposition table (s5-paging-null.md:156-158) written before the run; the applied outcome is triaged at the 3.5 block.

### 26. specs/022-unified-brain-arc/evidence/s5-paging-null.md:304  key=specs/022-unified-brain-arc/evidence/s5-paging-null.md@a977fd8e7138
class: restatement
evidence: Second pre-registration amendment, outcome text (s5-paging-null.md:303-308), written before the campaign.

### 27. specs/022-unified-brain-arc/evidence/s5-paging-null.md:306  key=specs/022-unified-brain-arc/evidence/s5-paging-null.md@8de96c3725db
class: restatement
evidence: Continuation of the same amendment text (s5-paging-null.md:306-308); not an applied removal.

### 28. specs/022-unified-brain-arc/evidence/s5-paging-null.md:512  key=specs/022-unified-brain-arc/evidence/s5-paging-null.md@bf906c5f281f
class: at-scale
NULL-RECORD: activation-driven neuron pager (residency layer) | units=16384 | ticks=unknown | seeds=8 | mde=unknown | retest=NOT-NEEDED: at-scale ran 226 cells at 16,384 units (above the 14,215-unit ingest) and failed its cost bound by 11.7x
evidence: s5-paging-null.md:403 (226 cells, 16,384 units, S = 8) and :475 (raw tick-cost ratio 35.2x against a 3.0x bound). The recorded units equal the largest run-through-the-brain scale in the capacity table, so nothing is 10x below. The measured memory win (s5-paging-null.md:461-475) is part of the record. This class depends on the capacity chosen (review #4486 m3): against the engine's 222,193-unit binding width (s2-state-per-unit.md:56) the run would sit 13.6x below; run-through capacity is used because it is what the brain has actually run.

### 29. specs/022-unified-brain-arc/evidence/s5-paging-null.md:761  key=specs/022-unified-brain-arc/evidence/s5-paging-null.md@c95b15ab106e
class: restatement
evidence: Fold commit table (s5-paging-null.md:755-761) lists the execution of the removal triaged at the 3.5 block; it adds no measurement.

### 30. specs/022-unified-brain-arc/evidence/sc-012a-width.md:271  key=specs/022-unified-brain-arc/evidence/sc-012a-width.md@b4005869fac4
class: restatement
evidence: Disposition item 2 (sc-012a-width.md:271-273) says the L2 removal stands and cites the width measurement; it restates the L2 removal triaged at its l-exit row.

### 31. specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md:109  key=specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md@948a3b350b92
class: small-scale
NULL-RECORD: cerebellar climbing-fibre LTD rule (cf_ltd_rule) | units=32 | ticks=900 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: t069-cerebellar-rule.md:157-163: taught vs shuffled teaching signal t 1.242 (p 0.133; row at :163) on held-out seeds 100-107, 900 steps; the plant's tissue sizes were n_mossy 4-32 (t069:19). The gain was uniform-dominated (best uniform +0.0188 vs rule +0.0189), so a larger tissue could still open headroom the uniform scale cannot take. AXIS = units: the rule's job is credit assignment across mossy channels; 32 is 512x below 16,384. Underpowered (panel #4528 B-M4): the record gives 29% power at S=8 (t069-cerebellar-rule.md:261), so under POLICIES P8 amendment item 4 this result was INCONCLUSIVE; its re-test must first reach power .8 on seeds, then run at 10x units.

### 32. specs/022-unified-brain-arc/evidence/trainer-retirement.md:67  key=specs/022-unified-brain-arc/evidence/trainer-retirement.md@7edd5f76fa51
class: restatement
evidence: Pre-registered null disposition for the trainer port (trainer-retirement.md:65-68); it states the rule, no result.

### 33. specs/022-unified-brain-arc/evidence/trainer-retirement.md:242  key=specs/022-unified-brain-arc/evidence/trainer-retirement.md@2d523a869ab2
class: small-scale
NULL-RECORD: mnist_genn one-epoch smoke (retired module) | units=unknown | ticks=unknown | seeds=unknown | mde=unknown | retest=CANDIDATE
evidence: trainer-retirement.md:242-250: chance after one epoch on the full set (test 0.0785); the lines say the four numbers are unreproduced and the removal rests on the owner's decision 17c and the module's self-declared UNVERIFIED status (trainer-retirement.md:247-249). Integrator fold of reviews #4485 (MAJOR-1) and #4486 (M1): the lane's draft classed this at-scale because a 3-epoch budget is within 10x of one epoch, which misreads the rule; the rule asks whether the null was measured within 10x of what can run today, and this record states no scale at all, while a 10-epoch re-test is runnable. AXIS = training budget (epochs, the ticks axis). Lowest value: the deletion rests on decision 17c, not only on this smoke.

### 34. specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md:435  key=specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md@d7287e22638d
class: restatement
evidence: Audit prose praising the deletion discipline (trajectory-audit-2026-09-01.md:434-436); it records no deletion.

### 35. specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md:689  key=specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md@3c47b4420774
class: restatement
evidence: Audit finding quoting the policy for the spiking half of gf.solve_series (trajectory-audit-2026-09-01.md:689); it proposes, and no file in the evidence tree records the removal.

### 36. specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md:454  key=specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md@f1104c223cfa
class: restatement
evidence: Plan item 5 of the audit's numbered list (trajectory-audit-2026-09-01.md:558-559, wrapped across two lines, so only the gate's paragraph census after fold F1 sees it; the key is the whole list paragraph starting at :454) proposes a fix to the answering path and cites the policy; it records no removal. Added by the integrator, 2026-10-23.

## Untracked sources (memory ledger `project-snn-refuted-and-negatives.md`)

### U1. ledger  key=untracked:w1-goal-curriculum
class: small-scale
NULL-RECORD: goal curriculum for the arm (basal_ganglia/curriculum.py, W1/T076) | units=n/a | ticks=400 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Ledger section 'W1 / T076' (project-snn-refuted-and-negatives.md:172-200) records the retirement by owner decision after t -0.34 / -0.52 / -0.50 on 400 ticks x 8 seeds; the same ledger states a 10% effect needs about 96 seeds and the observed effect needs ~437 (matches w1-goal-curriculum.md:490). The tracked file records the measurement but its wording sits outside the census regex, so it has no census key. AXIS = ticks: curriculum progress is a long-horizon quantity; 400 is 24x below 9,600. Unit-dependence is not the question, so units=n/a. Underpowered (panel #4528 B-M4): the record says a 10% effect needs about 96 seeds, so under POLICIES P8 amendment item 4 the result was INCONCLUSIVE; its re-test must first reach power .8 on seeds, then run at 10x ticks.

### U2. ledger  key=untracked:l1b-ach-gate
class: small-scale
NULL-RECORD: acetylcholine plasticity gate wire (removed with L1b) | units=unknown | ticks=800 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Ledger entry for L1b (project-snn-refuted-and-negatives.md:368-390): the gate level is alive but its last-200-tick mean is 0.026-0.053; the wire was withdrawn and later removed in the fold (l1b-synaptic-online.md:153 shows the 0.030 tail). The tracked record does not carry a NULL-RECORD for the gate alone. AXIS = ticks (a tail magnitude over a short run); 800 is 12x below 9,600. Low value: the gate only matters if a rule that needs it is re-tested.

### U3. ledger  key=untracked:hjepa-linear-v3.13
class: structural
NULL-RECORD: linear two-level H-JEPA core module (v3.13) | units=unknown | ticks=unknown | seeds=unknown | mde=unknown | retest=STRUCTURAL: both levels use spectrally contractive linear predictors that saturate to a fixed point under multi-step rollout
evidence: Ledger section 'Hierarchy / world-model negatives' (project-snn-refuted-and-negatives.md:94-102): the module was removed uncommitted after L0 fast-roll MSE 0.35 beat L1 coarse-roll 0.48; the stated mechanism is a property of linear contractive predictors, not of run size, and the ledger names the fix (nonlinearity on all three axes). No scale is recorded.

### U4. ledger  key=untracked:l1b-bcm
class: small-scale
NULL-RECORD: BCM local rule on the grown units' input rows (L1b) | units=38 | ticks=800 | seeds=8 | mde=unknown | retest=CANDIDATE
evidence: Split from block 11 by the integrator's fold of review #4486 (m2). Row L1b (l-exit.md:20) records BCM at t -1.36 vs frozen rows, which did not fall past the bar, 8 seeds x 800 ticks; grown units stay in the single digits to low tens (l2-grow-on-plateau.md:165, sc-012a-width.md:188). The record's mechanism for the L1b fall is a readout held stale by the closed growth gate (l1b-synaptic-online.md:268-277), and L1c (co-adapting readout) has since landed, so that cause is now testable. AXIS = units: the rule acts on the grown units' input rows, 2-38 units vs 16,384.

## Candidates ranked by how load-bearing the removed job was (P8's transferable rule)

| rank | key | method | axis | recorded scale | why it matters |
|---|---|---|---|---|---|
| 1 | untracked:w1-goal-curriculum | goal curriculum (W1/T076) | ticks | 400 ticks x 8 seeds | The only mechanism aimed at autonomous goal-driven exploration; the record itself says the effect was unresolvable at 8 seeds (needs ~96). |
| 2 | specs/022-unified-brain-arc/evidence/l-exit.md@6a99ed3ae91e + untracked:l1b-bcm | Oja (block 11) and BCM (U4) local rules on grown input rows (L1b) | units | 8 seeds x 800 ticks, 2-38 units | Online synaptic learning is a named brain job; the stated cause of the L1b fall (stale readout under a closed gate) is gone since L1c landed. Neither rule's fall is a pre-registered harm (panel #4528 B-M5). |
| 3 | specs/022-unified-brain-arc/evidence/s0-exit.md@c326a826a698 | interoceptive charge route (T039b) | ticks | 120 ticks | A later 1000-tick x 5-seed read already shows the insula reaching two deciding columns, so the early null may not hold. |
| 4 | specs/022-unified-brain-arc/evidence/l-exit.md@7e97ad3d8373 | plateau growth trigger (L2) | units | 8 seeds x 1200 ticks, 2-38 units | Growth decisions are the engine's core loop; the record names a missing stream where growth substitutes for learning. |
| 5 | specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md@948a3b350b92 | cerebellar LTD rule (T069) | units | n_mossy 4-32, 900 steps, 8 seeds | Credit assignment in the cerebellum; the null came from a plant whose headroom was uniform, so the plant, not the rule, limited the test. |
| 6 | specs/022-unified-brain-arc/evidence/g5-readout-variants.md@4fe8589dc323 | categorical coupling of the readout (G5 Part B) | ticks | 800 ticks, N=256, 8 seeds | A readout prior; the readout stays below the persistence floor, but the test used a single short stream. |
| 7 | untracked:l1b-ach-gate | acetylcholine plasticity gate wire | ticks | 800 ticks, 8 seeds | Only matters if a rule needing a third factor is re-tested. |
| 8 | specs/022-unified-brain-arc/evidence/trainer-retirement.md@2d523a869ab2 | mnist_genn one-epoch smoke (retired module) | ticks (training budget) | 1 epoch on the full set; units, ticks and seeds not recorded | The deletion rests on owner decision 17c and the module's UNVERIFIED status; a 10-epoch re-test is runnable. Lowest value. |

*DATED 2026-10-28 (cloud audit; nothing above is edited):* rank 2's "the stated cause of the L1b fall (stale readout under a closed gate) is gone since L1c landed" holds for the LIVE prediction only. `growth/neurogenesis_engine.py` keeps `_batch_beta`, the readout of the last OPEN-gate solve (L1c Amendment 3), and scores every closed-gate refit row of the growth gate's evidence with it, so the stale reader is still there for the gate. The owed co-adaptation control is pre-registered as L1b-STALE (T121; record `evidence/d1-readout-staleness.md`, on the integrator's branch at the time of writing).

Nothing in this list is restored by it. The axis choice is the author's (owner answer 5) and is for the panel to review.
