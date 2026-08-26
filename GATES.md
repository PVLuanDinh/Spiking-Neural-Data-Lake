# Gates: specs/022-unified-brain-arc — remaining tasks (T051a → T087)

OWNS: src/**, tests/**, specs/**, docs/**, tools/**, run_selfchecks.py

Scope: every remaining specs/022 task row lands [X] under the SNDL per-task rhythm (build → 3-lens panel → 13-stage battery ALL GREEN → row with measured evidence → commit → push → daemon poll), and the terminal gates prove the whole tree green with zero open rows.

Evidentiary basis, stated once: each per-task gate's CHECK reads the task's `- [X]` row in specs/022-unified-brain-arc/tasks.md. That row is written ONLY after the task's full battery verdict is ALL GREEN and the commit is pushed (the rhythm), so the row is the durable completion record — the gate verifies the record exists, the rhythm is what makes the record honest. G0 is the checker's own sensitivity control. This file is deliberately UNTRACKED (a working ledger, not a spec artifact).

- [x] G0: row-checker sensitivity control — the regex matches a known-done row (T050) and does NOT match a known-open row (T051a)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');const done=/^- \[X\] T050 /m.test(t);const open=!/^- \[X\] T051a /m.test(t)||/^- \[ \] T051a /m.test(t);if(done&&open)console.log('CONTROL-OK both-directions');else process.exit(1)"
  EXPECT: CONTROL-OK both-directions
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\AI-project\spiking-neural-data-lake; path=f3f15cf8c5ab/38 entries; EXPECT=matched; output-sha256=a5538f49a2ae589d7678127707026d78e727bc802c924096d32d7fded78c9583; output-bytes=27

- [ ] T004: the three cross-cutting invariant texts landed in docs/POLICIES.md
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T004 /m.test(t))console.log('ROW-DONE T004');else process.exit(1)"
  EXPECT: ROW-DONE T004
  EVIDENCE: pending

- [ ] T051a: composer kernel — typed read/write sets, N rules via the T050 algebra, naive-sum red, per-rule neuter attribution
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T051a /m.test(t))console.log('ROW-DONE T051a');else process.exit(1)"
  EXPECT: ROW-DONE T051a
  EVIDENCE: pending

- [ ] T051b: incumbent discharges through the composer — LPL extraction, T048 two-writer adjudication, producers×manifest
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T051b /m.test(t))console.log('ROW-DONE T051b');else process.exit(1)"
  EXPECT: ROW-DONE T051b
  EVIDENCE: pending

- [ ] T051c: FR-018 sweep over the governed rule census (a wave, per-rule rows)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T051c /m.test(t))console.log('ROW-DONE T051c');else process.exit(1)"
  EXPECT: ROW-DONE T051c
  EVIDENCE: pending

- [ ] T051: parent record — lands only when T051a+T051b+T051c are all [X]
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');const ok=['T051a','T051b','T051c'].every(id=>new RegExp('^- \\[X\\] '+id+' ','m').test(t))&&/^- \[X\] T051 /m.test(t);if(ok)console.log('ROW-DONE T051-family');else process.exit(1)"
  EXPECT: ROW-DONE T051-family
  EVIDENCE: pending

- [ ] T052: ESPP adoption + N-MNIST/SHD modality manifest rows (discharges T036's S1 disposition)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T052 /m.test(t))console.log('ROW-DONE T052');else process.exit(1)"
  EXPECT: ROW-DONE T052
  EVIDENCE: pending

- [ ] T053: iSTDP charged on the tick (dimensionality-collapse neuter control)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T053 /m.test(t))console.log('ROW-DONE T053');else process.exit(1)"
  EXPECT: ROW-DONE T053
  EVIDENCE: pending

- [ ] T054: cerebellar tissue build (granule/PF/Purkinje/MLI + pn relay) — also the dated consumer of T048's parked structural-rewiring wire
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T054 /m.test(t))console.log('ROW-DONE T054');else process.exit(1)"
  EXPECT: ROW-DONE T054
  EVIDENCE: pending

- [ ] T055: WM attractor build (Mongillo populations + STP)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T055 /m.test(t))console.log('ROW-DONE T055');else process.exit(1)"
  EXPECT: ROW-DONE T055
  EVIDENCE: pending

- [ ] T056: K derivation (K = T_max/t_s) + the five guards
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T056 /m.test(t))console.log('ROW-DONE T056');else process.exit(1)"
  EXPECT: ROW-DONE T056
  EVIDENCE: pending

- [ ] T057: recall contract (recall(cue) → k episodes with margins)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T057 /m.test(t))console.log('ROW-DONE T057');else process.exit(1)"
  EXPECT: ROW-DONE T057
  EVIDENCE: pending

- [ ] T058: sleep ownership L5.5 (consolidation reaches Benna-Fusi; per-capability neuters)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T058 /m.test(t))console.log('ROW-DONE T058');else process.exit(1)"
  EXPECT: ROW-DONE T058
  EVIDENCE: pending

- [ ] T060: executive faculty port (task-set FiLM, FR-034)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T060 /m.test(t))console.log('ROW-DONE T060');else process.exit(1)"
  EXPECT: ROW-DONE T060
  EVIDENCE: pending

- [ ] T061: US2 exit certification (seed→multi-region micro-brain evidence)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T061 /m.test(t))console.log('ROW-DONE T061');else process.exit(1)"
  EXPECT: ROW-DONE T061
  EVIDENCE: pending

- [ ] T062: H01 ingest, cheapest slice first
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T062 /m.test(t))console.log('ROW-DONE T062');else process.exit(1)"
  EXPECT: ROW-DONE T062
  EVIDENCE: pending

- [ ] T063: three-arm drift harness (seeded/random/wrong-prior)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T063 /m.test(t))console.log('ROW-DONE T063');else process.exit(1)"
  EXPECT: ROW-DONE T063
  EVIDENCE: pending

- [ ] T064: unit blocks + block-aligned lake records
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T064 /m.test(t))console.log('ROW-DONE T064');else process.exit(1)"
  EXPECT: ROW-DONE T064
  EVIDENCE: pending

- [ ] T065: binding-width derivation (resident bytes > host RAM, measured)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T065 /m.test(t))console.log('ROW-DONE T065');else process.exit(1)"
  EXPECT: ROW-DONE T065
  EVIDENCE: pending

- [ ] T066: activation-driven paging (P8 invention #1) — also the dated consumer of percept_surprise's paging disposition
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T066 /m.test(t))console.log('ROW-DONE T066');else process.exit(1)"
  EXPECT: ROW-DONE T066
  EVIDENCE: pending

- [ ] T067: growth delta tier + background merge (FreshDiskANN shape)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T067 /m.test(t))console.log('ROW-DONE T067');else process.exit(1)"
  EXPECT: ROW-DONE T067
  EVIDENCE: pending

- [ ] T068: paging null-clause measurement at a derived comparison width
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T068 /m.test(t))console.log('ROW-DONE T068');else process.exit(1)"
  EXPECT: ROW-DONE T068
  EVIDENCE: pending

- [ ] T069: cerebellar rule on the built tissue (also discharges T037's efferent consumer disposition)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T069 /m.test(t))console.log('ROW-DONE T069');else process.exit(1)"
  EXPECT: ROW-DONE T069
  EVIDENCE: pending

- [ ] T070: live integration instrument
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T070 /m.test(t))console.log('ROW-DONE T070');else process.exit(1)"
  EXPECT: ROW-DONE T070
  EVIDENCE: pending

- [ ] T071: US3 exit certification
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T071 /m.test(t))console.log('ROW-DONE T071');else process.exit(1)"
  EXPECT: ROW-DONE T071
  EVIDENCE: pending

- [ ] T072: internal-only proposer (FR-025)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T072 /m.test(t))console.log('ROW-DONE T072');else process.exit(1)"
  EXPECT: ROW-DONE T072
  EVIDENCE: pending

- [ ] T073: evaluator cascade (P8 invention #5)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T073 /m.test(t))console.log('ROW-DONE T073');else process.exit(1)"
  EXPECT: ROW-DONE T073
  EVIDENCE: pending

- [ ] T074: archive-not-champion + MAP-Elites cells
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T074 /m.test(t))console.log('ROW-DONE T074');else process.exit(1)"
  EXPECT: ROW-DONE T074
  EVIDENCE: pending

- [ ] T075: circuit-breaker + self-build cockpit rows
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T075 /m.test(t))console.log('ROW-DONE T075');else process.exit(1)"
  EXPECT: ROW-DONE T075
  EVIDENCE: pending

- [ ] T076: curriculum as the curiosity consumer (|LP| bandit; the GoalStore's live writer arrives here or is re-adjudicated)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T076 /m.test(t))console.log('ROW-DONE T076');else process.exit(1)"
  EXPECT: ROW-DONE T076
  EVIDENCE: pending

- [ ] T077: metacognition instrument (extends the incumbents)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T077 /m.test(t))console.log('ROW-DONE T077');else process.exit(1)"
  EXPECT: ROW-DONE T077
  EVIDENCE: pending

- [ ] T078: attention schema (adopt + P8 invention #8)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T078 /m.test(t))console.log('ROW-DONE T078');else process.exit(1)"
  EXPECT: ROW-DONE T078
  EVIDENCE: pending

- [ ] T079: agency instrument (efference-copy mismatch red)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T079 /m.test(t))console.log('ROW-DONE T079');else process.exit(1)"
  EXPECT: ROW-DONE T079
  EVIDENCE: pending

- [ ] T080: temporal-binding instrument (membership-matched pairing)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T080 /m.test(t))console.log('ROW-DONE T080');else process.exit(1)"
  EXPECT: ROW-DONE T080
  EVIDENCE: pending

- [ ] T081: report-order probe (also the consciousness_probe retirement point)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T081 /m.test(t))console.log('ROW-DONE T081');else process.exit(1)"
  EXPECT: ROW-DONE T081
  EVIDENCE: pending

- [ ] T082: social cognition composed (recursive_tom as an L3 faculty)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T082 /m.test(t))console.log('ROW-DONE T082');else process.exit(1)"
  EXPECT: ROW-DONE T082
  EVIDENCE: pending

- [ ] T083: US4 exit certification (SC-008/SC-009 evidence)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T083 /m.test(t))console.log('ROW-DONE T083');else process.exit(1)"
  EXPECT: ROW-DONE T083
  EVIDENCE: pending

- [ ] T084: cockpit completion sweep (all instrument rows with PROVENANCE)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T084 /m.test(t))console.log('ROW-DONE T084');else process.exit(1)"
  EXPECT: ROW-DONE T084
  EVIDENCE: pending

- [ ] T085: per-tissue human-scale derivation
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T085 /m.test(t))console.log('ROW-DONE T085');else process.exit(1)"
  EXPECT: ROW-DONE T085
  EVIDENCE: pending

- [ ] T086: consistency sweep (every SYNTHESIS §6 row executed or carried dated)
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T086 /m.test(t))console.log('ROW-DONE T086');else process.exit(1)"
  EXPECT: ROW-DONE T086
  EVIDENCE: pending

- [ ] T087: quickstart end-to-end on the pinned host
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');if(/^- \[X\] T087 /m.test(t))console.log('ROW-DONE T087');else process.exit(1)"
  EXPECT: ROW-DONE T087
  EVIDENCE: pending

- [x] GX1: fast gate composite green on the final tree (manifest, p4c, p6, red-reachability, gradient-claim, citation, policy-manifest)
  CHECK: node -e "const {execSync}=require('child_process');const py='.venv-lake/Scripts/python.exe';const cmds=[[py,['-m','snn_data_lake.substrate.manifest_gate','--assert']],[py,['tools/p4c_ratchet.py','--assert']],[py,['tools/p6_gate.py','--assert']],[py,['tools/red_reachability.py','--assert']],[py,['-m','snn_data_lake.substrate.gradient_claim_gate','--assert']],[py,['tools/citation_guard.py']],[py,['-m','snn_data_lake.substrate.policy_manifest','--assert']]];const {spawnSync}=require('child_process');for(const [c,a] of cmds){const r=spawnSync(c,a,{env:{...process.env,PYTHONPATH:'src',PYTHONUTF8:'1'},stdio:'pipe'});if(r.status!==0){console.error('GATE-RED',a.join(' '));process.exit(1)}}console.log('FAST-GATES-GREEN 7/7')"
  EXPECT: FAST-GATES-GREEN 7/7
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\AI-project\spiking-neural-data-lake; path=f3f15cf8c5ab/38 entries; EXPECT=matched; output-sha256=3e17a2ef69fbd43727169d8332a5bc03ddff64c238b1d5a9ce0e34e934548d9b; output-bytes=21

- [ ] GX2: zero open task rows remain in specs/022 tasks.md
  CHECK: node -e "const t=require('fs').readFileSync('specs/022-unified-brain-arc/tasks.md','utf8');const open=(t.match(/^- \[ \] T\d/gm)||[]).length;if(open===0)console.log('ZERO-OPEN-ROWS');else{console.error('open rows:',open);process.exit(1)}"
  EXPECT: ZERO-OPEN-ROWS
  EVIDENCE: pending

- [ ] GX3: full self-check suite green on the final tree (the battery's heaviest stage, run to completion)
  CHECK: node -e "const {spawnSync}=require('child_process');const r=spawnSync('.venv-lake/Scripts/python.exe',['run_selfchecks.py','--carry'],{env:{...process.env,PYTHONPATH:'src',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'},stdio:'pipe',timeout:3600000});if(r.status===0)console.log('SELFCHECKS-GREEN rc=0');else process.exit(1)"
  EXPECT: SELFCHECKS-GREEN rc=0
  EVIDENCE: pending

- [ ] GX4: final state pushed — working tree clean and local main not ahead of origin/main (manual: verified against `git status -sb` at the end of the run, with the last battery number and HEAD SHA recorded here)
  EVIDENCE: pending

<!--
Rhythm reminder: a task row turns [X] ONLY after its own 13-stage battery is ALL
GREEN and the commit is pushed. These gates verify the records and the tree; the
rhythm is what makes the records honest. GX3 is heavy (~15 min with carry) — run it
at checkpoints, not per row-flip. This ledger is untracked on purpose.
-->
