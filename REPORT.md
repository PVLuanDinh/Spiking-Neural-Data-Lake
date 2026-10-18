# Phase 1a report: collapse the full-lake scans in Python (lane rust-p1a, 2026-10-17)

Branch `claude/rust-p1a-1017` at base `06744945` (= origin/main). Worktree `D:\AI-project\lanes\rust-p1a-1017`.
Pre-registered criteria: `D:\AI-project\sndl-rust-port-phase0-report-2026-10-17.md` section 4 "Phase 1a", quoted
verbatim in section 3 below. Nothing was loosened. Lake snapshot: live `~/.snn_data_lake` copied once (minus
`telemetry*`, `*.bak*`, `*.retired`, `*.lock`, `selfcheck_cache*`) to `D:\AI-project\lanes\rust-p1a-1017-data\home_src`,
sha256 `695d01cfd0f3ae2c8cfe98ce44be4e78ec0da09e827c23f9e8d643350b84af8c`, 3,456,242,033 B (the Phase 0 snapshot,
unchanged: live mtime 2026-10-15 15:17:30). Every run used a fresh copy of it, the daemon stayed down, the live lake
was only ever read by the copy. All copies are deleted at the end (section 9).

## 0. Verdict in one line

Boot 7 -> 2 disk passes, night 9.02 -> 0.015 lake-reads, consolidation tick 7 passes -> 0 disk passes + the rewrite,
lake byte-identical in 7/7 runs, red fires, per-consumer loads equal; criterion (c) (tick median <= 13 s) is a
MISS as pre-registered (median 19.5 s against 27.3 s baseline) with the variance located in the 3.47 GB rewrite's
I/O, not in the scans (section 5).

## 1. The change (3 files, one mechanism, no flag, no dual path)

1. `src/snn_data_lake/substrate/binlog.py` (claimed #3633). `BinLog.__init__` already materialises
   `payloads = list(self.replay())` and keeps every payload in `_seen` (dedup). It now keeps that same list, in
   append order, as `self._payloads` (one pointer per record: 64 MB on 8.0M records; `None` for `dedup=False`, where
   keeping it would add 3.46 GB). `append`/`extend` append written payloads to it, `rewrite` replaces it with the
   survivors. `replay()` yields the resident list when `stale()` is False (the same stat `home._log` already trusts
   to hand out the cached instance) and falls through to the unchanged disk scan otherwise (stale instance,
   `dedup=False`, bare `BinLog.__new__`). Construction's two passes (`_repair_torn_tail` + replay) are untouched.
2. `src/snn_data_lake/substrate/home.py` (HELD by claude-homecensus-1017: worktree copy edited, exact patch at
   `patches/home.py.patch` against 06744945, NOT committed). `KnowledgeLog.consolidate` makes ONE pass over the
   lake: FACT records are decoded for the immortal guard exactly as `replay()` decodes them, every other tag is
   collected raw for the pass-through. Same `lost`, same `foreign`, same `ImmortalRefusal`; the `_prune_state_history`
   call and the `_snn_state_index` pop are unchanged. `BinLog.rewrite` semantics untouched.
3. `src/snn_data_lake/engine/brain.py` (claimed #3634). `_enter_sleep` builds ONE `KnowledgeLog` and ONE
   `records = list(klog.replay())` for both the nights%4 schema block and the nights%8 lake block (each used to build
   its own KnowledgeLog, a fact-count pass, and replay the facts again). `records` stays exactly what the lake block's
   own replay would have yielded: a schema-induction row that actually lands (`log.append` returned True) is appended
   as `unpack_fact(pack_fact(rec))`, the dict a replay returns. Gating is equivalent (`records is not None` <=>
   persist and nights%4==0 and the KnowledgeLog built).
4. `tests/test_binlog_resident_replay.py` (claimed #3636): the one runnable check -- resident replay equals a
   `dedup=False` disk read, reads nothing (`open` monkeypatched to raise), follows `rewrite`, and an external
   writer / truncation makes the same instance read the disk again.

`sleep/lake_consolidation.py` (held) was not changed. No new def/class under `src/`: `graph-exists.js` was run with
`--root` for the three capabilities ("resident payload list in BinLog", "consolidate single pass", "shared sleep
read"); the exact hits were `BinLog` and `KnowledgeLog`, which were edited in place (reuse), the rest were
distinct-from (other stores: `lake_memory`, `self_model` spike-lake trace, `lake_bridge` raw iteration).

## 2. Instruments

`D:\AI-project\lanes\rust-p1a-1017-tools\` (outside the worktree; results in `results\`):

- `run1a.ps1`: one brain run on a fresh copy; env set INSIDE Windows (`PYTHONPATH=<repo>\src`, `NORD_HOME=<copy>`,
  `NORD_GPU=1`, `PYTHONDONTWRITEBYTECODE=1`), Phase 0 `cpucheck.py --wait` quiet gate (system CPU < 12% on two
  consecutive 3 s samples; the post-gate sample is logged per run), sha256 of the lake after. Both arms run with
  cwd = `D:\AI-project\spiking-neural-data-lake` (section 7, finding 2). Every script prints
  `snn_data_lake.__file__` and asserts it is under `--repo`.
- `prof1a.py`: Phase 0 `prof_tick.py` mode=time with `--repo` and the `__file__` assert (220 ticks, warm 10).
- `scan1a.py`: Phase 0 `scan_count.py` minus cProfile/PFC fixture, plus: per-`replay()` read-byte delta (a full
  replay is a DISK pass when the process read counter moved >= 0.5 lake sizes during it, else resident), the
  per-consumer boot loads of criterion (f), and a tag census. Positive controls run first and passed in both arms:
  scratch-log construct = {construct 1, repair 1, replay 1}, explicit full / early-close partial, and one explicit
  RAW replay of the copy moved the read counter by exactly 3,456,242,033 B (ratio **1.000** in CNT_B and CNT_A: the
  control re-run the brief asked for).
- `red1a.py`: Phase 0 `red_1a.py` with `--repo` (consolidation's foreign pass-through silently drops `_T_SELFBUILD`).
- Timed arms interleaved baseline/after: B1 (after CNT_B), then A1 B2 A2 B3 A3 (after CNT_A), then RED.

Baseline = main checkout `D:\AI-project\spiking-neural-data-lake` at 06744945 (its `src/` is clean; only
`.gitattributes`/`AGENTS.md` are locally modified). After = the worktree.

## 3. Pre-registered criteria, baseline vs after

| criterion (Phase 0 section 4, verbatim) | baseline (this lane) | after | verdict |
|---|---|---|---|
| (a) consolidation tick <= 1 full pass + the rewrite; counter view AND read-byte view <= 1.1 lake sizes excl. the rewrite's write | 7 disk passes + rewrite; read-byte 7.005 | **0 disk passes** + rewrite; read-byte **0.000**; `replay()` was called 4 times on the tick, all served resident (0 bytes) | PASS (see disclosure D1) |
| (b) night total <= 2.1 lake-reads | 9.021 (9 disk passes) | **0.015** (0 disk passes; the 0.015 is tick 15's `language_acq.npz`-class reads, identical in baseline) | PASS |
| (c) consolidation tick median of 3 quiet-gated runs <= 13 s | 24.50 / 29.06 / 27.28 -> median 27.28 s | 19.53 / 20.23 / 8.42 -> **median 19.53 s** | **MISS** (section 5) |
| (d) post-220-tick lake sha256 == 6c9375e0...bbca8 (3,455,394,651 B) | 6c9375e0 in CNT_B, B1, B2, B3 | 6c9375e0 in CNT_A, A1, A2, A3 (3,455,394,651 B) | PASS, 7/7 |
| (e) wake mean within 112.45-113.38 ms +/- 5% (106.83-119.05) | 108.70 / 129.42 / 112.11, mean of means 116.74 | 106.88 / 114.74 / 134.88, mean of means **118.83** | PASS on the mean of means (by 0.2 ms); single runs on BOTH arms left the band today (B2 129.4, A3 134.9): day noise, section 5 |
| (f) per-consumer boot loads equal baseline | table below | identical | PASS |
| boot passes 7 -> <= 3 (construction's two untouched) | 7 disk passes, read-byte 7.02 | **2 disk passes** (repair + construction replay), read-byte **2.02**; `replay()` called 6 times, 5 resident | PASS |
| unprofiled construction <= 106.45 s | 99.10 / 98.59 / 89.27 | 84.07 / 81.09 / **101.57** (max) | PASS |
| RED `red_1a.py` must change the lake hash | Phase 0: 42a9879b | **42a9879b5f1b40f474942287297617bea4d13c7ab66bb3df9343aa92468b21b7**, 3,447,033,966 B, `prune_calls=1 selfbuild_dropped=53666 errs=0` | FIRES |
| control: read-byte counter calibrated 1.000 on one explicit replay | 1.000 (CNT_B) | 1.000 (CNT_A) | re-run, holds |

### (f) per-consumer boot loads (scan1a.py wrappers; recorded in the baseline run BEFORE the change)

| consumer | baseline CNT_B | after CNT_A |
|---|---|---|
| `_restore_episodes` -> `load_episodes(limit=64)` | 64 episodes, sha c839742401ecca99 | 64, c839742401ecca99 |
| `_r2_boot` -> `scan_learner_events` (strict all-kinds `load_episodes`) | 784,868 episodes (sha 22d7c8c28fb32a08); 123,338 events, 0 refusals, session 2 | identical |
| `load_growth` (`_make_neuro`) | 5,715 B blob, sha e773171b53d8c7d4 | identical |
| named states (`_state_index` at boot) | 8 names (coverage_cells, growing_learners.json, interaction_learner.json, nightly_skills.json, reasoning_curriculum.json, recall_trace.json, signal_calib.json, time_cells.json), index sha b62c301c10ae8b0a; rebuilt after consolidate: 8, 8280e2c1dbf030ec | identical |
| coalition ring `base_ordinal` | 6,606,150 | 6,606,150 |
| `KnowledgeLog()` fact count (t91, t189) | 30,452 | 30,452 (one construction per tick instead of two at t189) |
| end of run: `_lake_kept/_lake_pruned/_lake_night/_lake_refused/_schemas/_nights` | 30452 / 0 / 8 / 0 / 0 / 9 | identical |
| lake tag census (records) | 01:30,452 03:86,095 04:784,868 05:53,666 07:25 08:437,729 0a:6,606,150 (7,998,985) | same snapshot |

The only differing number is the R2 receipt sha (f5f2ba05 vs 9eff1ce7): the receipt embeds the lake file's
`mtime_ns`/inode stat records, which differ per fresh copy; its counts (events, refusals, session) and the final
lake hash are identical.

## 4. Timed runs (n=3 per arm, interleaved, quiet-gated; `results\<tag>.json`)

| run | order | arm | post-gate CPU % (wait s) | construct s | wake mean ms | wake median | wake p90 | tick 189 s | lake sha |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 1 | baseline | 17.2 (6) | 99.10 | 108.70 | 114.63 | 128.02 | 24.50 | 6c9375e0 |
| A1 | 2 | after | 5.8 (6) | 84.07 | 106.88 | 114.21 | 124.83 | 19.53 | 6c9375e0 |
| B2 | 3 | baseline | 6.0 (18) | 98.59 | 129.42 | 135.68 | 150.99 | 29.06 | 6c9375e0 |
| A2 | 4 | after | 10.7 (6) | 81.09 | 114.74 | 121.55 | 134.23 | 20.23 | 6c9375e0 |
| B3 | 5 | baseline | 5.6 (6) | 89.27 | 112.11 | 116.72 | 129.86 | 27.28 | 6c9375e0 |
| A3 | 6 | after | 51.6 (9) | 101.57 | 134.88 | 133.05 | 174.65 | 8.42 | 6c9375e0 |

Post-gate CPU: B1's 17.2% was a Claude-side `python.exe` (pid 48192); A3's 51.6% was `vmmemWSL` at 665% (the WSL
VM, my own side) -- A3 is the FAST after run, so the co-tenant rule does not rescue (c): A1 (5.8%) and A2 (10.7%)
were quiet and both exceed 13 s.

Instrumented runs (not timing arms): CNT_B tick 189 = 29.66 s (7 disk replays 3.5-6.2 s each, rewrite 5.37 s,
`_state_index` rebuild 2.65 s); CNT_A tick 189 = **7.72 s** (4 resident replays 0.53 + 0.59 + 0.82 + 0.41 s,
rewrite 3.47 s). Boot (instrumented): CNT_B 139.6 s, CNT_A 97.4 s; resident passes at boot: coalition count 0.73 s,
`_state_index` 0.89 s, `load_growth` 6.51 s (growth-chain hashing), `_restore_episodes` 3.27 s, strict
`scan_learner_events` 22.35 s (per-episode sha256 -- the consumer, not the scan, as Phase 0 said).

## 5. The (c) miss, located

The after tick is 8.4 s (A3), 7.7 s (CNT_A) when it is fast and ~20 s (A1, A2) when it is slow; the resident
scans are 2.3 s in every run, so the swing is in the 3.47 GB `BinLog.rewrite` (write + fsync + `os.replace`) and
whatever the OS is doing to the disk when it lands -- each timed run starts with a fresh 3.4 GB `Copy-Item` of the
lake, whose write-back is still draining when tick 189 fires ~4 min later on a fast boot. That is the rewrite the
brief says is NOT this lane's to change (compaction semantics, owner decision), and the "<= 5 s" target was already
recorded as unreachable without it. ATTRIBUTION BATCH (not the pre-registered n=3, run afterwards with a
disk-write-quiet gate added to the CPU gate and the rewrite timed inside the tick): see section 5a.

## 5a. Attribution batch (A4 A5 B4 A6: CPU gate + disk-write gate; `rewrite_s` recorded)

Same harness plus `cpucheck.py --disk` (system disk writes < 20 MB/s on two consecutive 3 s samples, after the CPU
gate) and a one-line wrapper timing `BinLog.rewrite` inside the tick (both arms). NOT the pre-registered n=3; it
does not replace section 4's verdict.

| run | arm | gate wait s (CPU post-gate %) | construct s | wake mean ms | tick 189 s | rewrite s | tick minus rewrite s | lake sha |
|---|---|---|---|---|---|---|---|---|
| A4 | after | 135.3 (15.7) | 109.54 | 132.23 | 8.55 | 4.00 | 4.55 | 6c9375e0 |
| A5 | after | 72.0 (20.9) | 89.73 | 165.63 | 37.11 | **33.15** | 3.96 | 6c9375e0 |
| B4 | baseline | 123.0 (8.0) | 100.06 | 118.78 | 28.92 | 9.17 | 19.75 | 6c9375e0 |
| A6 | after | 18.0 (7.7) | 82.68 | 107.74 | 7.15 | 3.52 | 3.63 | 6c9375e0 |

Read it this way: the part of the consolidation tick that this change owns (everything but the rewrite) is
**3.6-4.6 s after** against **19.8 s baseline** (B4) -- inside the ~4.5 s the Phase 0 criterion allotted for "one pass
+ other". The rewrite itself swung 3.5 / 4.0 / 9.2 / **33.1 s** across four back-to-back runs on this box today (Phase 0
measured 7.7 s), and the disk-write gate did not remove the swing (A5 had waited 72 s for a quiet disk). So (c) is a
MISS on the pre-registered median because the 3.47 GB rewrite's I/O time is not reproducible on this machine today,
not because the scans were not collapsed; whether the rewrite goes incremental is the owner's compaction decision
the Phase 0 report already names. Lake hash identical in all four (11/11 overall).

Timing token: `resource:timing` was held by this lane from 10:36 through every run; the lease renewed at 11:52
(#3657, 2 h) expired at 13:52 and `r02-reader` took it over at 13:53 (#3670/#3671), after this lane's last brain run
exited at 12:08 -- no overlap. Lake copies deleted at 15:25.

## 6. Tests (worktree code, cwd = main, rc captured directly)

- `python -m snn_data_lake.substrate.binlog` rc=0; `python -m snn_data_lake.substrate.home` rc=0;
  `python -m snn_data_lake.sleep.lake_consolidation` rc=0 (`results\tests1a.log`).
- `pytest --no-cov -p no:cacheprovider` on tests/test_binlog_resident_replay.py test_home_guard.py
  test_home_pytest_tripwire.py test_episode_consolidation.py test_g9_live_home_boot.py
  test_learn_daemon_telemetry_home.py test_lake_bridge.py test_lake_engine_recovery.py test_lake_recall.py
  test_episode_durability.py test_growth_delta_chain.py test_brain_growth_restart.py test_g0c_learner_persists.py
  test_r2_sleep_bout.py test_selfchecks.py -k "not test_selfchecks or home or binlog or lake_consolidation":
  **150 passed, rc=0**. (The first pass, with cwd = worktree and the repo's 70% coverage threshold on a subset,
  reported rc=1: one real red, finding 2 below, and the coverage threshold; both resolved as stated.)
- `ruff check` and `ruff format --check` clean on the three files.

## 7. Findings and disclosures

1. **D1, the counter view after the change.** The Phase 0 counter wrapped `BinLog.replay()` and called an
   exhausted call a full pass; after this change a cached instance's `replay()` exhausts without touching the
   disk. `scan1a.py` therefore records, per call, whether the process read counter moved (the calibrated control),
   and reports both: on tick 189 `replay()` is called 4 times (ctor count, records, consolidate's one pass,
   `_state_index` rebuild), 0 of them disk; at boot 6 times, 1 disk. The read-byte view (the independent control the
   criterion names) is what shows the collapse is real; the raw call count is reported so nobody has to trust the
   classifier.
2. **cwd-relative untracked data in the main checkout changes the brain's trajectory.** The first pytest pass
   reddened `test_episode_durability::test_recall_reaches_past_working_memory` on the worktree only. Probe
   (`rust-p1a-1017-tools\probe_recall.py`, 1600 fresh-lake ticks, deterministic in 2/2 repeats per arm): main cwd ->
   2,904 lake records, first episode `where='6'`, test passes; worktree cwd (same main code or worktree code) ->
   2,865 records, `where='None'`, window-tier recall returns a 0.6-confidence wrong episode; **worktree code with
   cwd = main -> 2,904, `where='6'`, passes.** Cause: `Brain._place_here` (grid-place binding) depends on something
   read relative to cwd that exists untracked under main (`0/`, `docs/knowledge/`, `.claude/`, ...) and not in a
   clean worktree. Not this lane's defect; a determinism hazard for the lead: a fresh clone or a worktree is not the
   same brain as the operator's checkout. All measurement here runs both arms with cwd = main.
3. The after boot is 15-18 s faster unprofiled (84/81 s vs 99/99 s on the first two pairs) and A3 101.6 s: all under
   the 106.45 s bar; not a claim, n=3 and the day was noisy.
4. Wake-mean noise today was ~25 ms peak-to-peak on BOTH arms (Phase 0 saw 1%); the paired delta after - baseline is
   +2.1 ms (+1.8%) on means of means; no wake-path code changed except one `list.append` per lake append.
5. Memory: +64 MB resident for the dedup instance (pointers only; the bytes were already in `_seen`);
   `dedup=False` instances unchanged. Concurrency note: a resident reader holds no file handle, so on Windows a
   concurrent `rewrite()`'s `os.replace` can no longer collide with a reader's open handle (that collision raised
   before); a reader mid-iteration keeps the old list, the same "whole old or whole new" the docstring promises.
6. The resident list is a CACHE over the disk (P4b wording): validity is `stale()`, the one stat `_log()` already
   trusts; a same-size external rewrite is invisible to it exactly as it was invisible to `_seen`/`_count` before.

## 8. Protocol steps that could not apply (named per the no-skip rule)

- Reload item 0 `i-have-adhd`: not installed in this lane's skill set (checked `~/.claude/skills`); output shape
  rules followed from the brief's own text.
- agent-fleet "open the TUI tab" (Desktop rule): terminal lane, no Desktop terminal tool; `fleet.py` not used (the
  lane rule forbids listing/spawning sessions).
- `graphify update .`: the graph is main's commit and `graphify-out/` lives only in the main checkout; the worktree
  change is folded by the integrator, who runs the update after promotion. `graph-exists.js --root <worktree>` was run
  before every edit (three queries; hits disposed in section 1).
- Reload item 9 advisor tool: not present in this session. The repo `verify` skill (`.claude/skills/verify`) targets the
  cockpit/daemon, which the brief orders DOWN; the equivalent proof here is the 220-tick lake byte-identity x7 and
  the selfchecks.
- Per-arc rhythm panel / battery / push / daemon poll: reserved to the integrator by the brief ("never push, promote
  or battery"); the branch is delivered for review.
- No AskUserQuestion/TaskCreate was needed; no step took a documented default in place of an answer.

## 9. Deliverables, claims, cleanup

- Commits on `claude/rust-p1a-1017`: `f2534a94` perf(lake): the change (binlog.py, brain.py, the test, the home.py
  patch); the next commit adds this REPORT.md. Nothing pushed, promoted or batteried (lane rule).
- Memory handoff: claude-mem #17476 (project `spiking-neural-data-lake/rust-p1a-1017`); the lane hook refuses writes
  under `.claude`, so no memory FILE was written -- the lead folds this report into the SNDL memory store.
- Patches for held paths: `patches/home.py.patch` (against 06744945). `lake_consolidation.py`: no patch needed.
- Board: thread RUST-P1A, status #3638; result + request to lead posted at the end.
- Claims to release: binlog.py #3633, brain.py #3634, tests/test_binlog_resident_replay.py #3636, resource:timing.
- Lake copies under `D:\AI-project\lanes\rust-p1a-1017-data\` deleted after the last run (`home_src`, `home_run`).
- Scripts and results: `D:\AI-project\lanes\rust-p1a-1017-tools\` (kept; not part of the repo).
