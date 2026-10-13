# R1 / T102 -- bound historical recall (pre-registration DRAFT and record)

Plan row: PLAN-FINAL R workstream, R1 / T102 (FR-038, SC-016), and Part F's R-R1 ("nonresident exact match is lost, a false
match is trusted, corrupt/stale index hides records, rebuild differs or cost bound fails | canonical exact scan,
absent/ambiguous cues and valid rebuilt index at the same watermark"). tasks.md T102: "preserve attractor/window/deep tiers,
verify candidates in source slot space, cover old/ambiguous/absent cues and restart; preregister workload splits and costs,
compose a rebuildable historical index or bounded-memory scan only if it earns replacement of the incumbent scan".
P4d: the brain "recalls anything in the lake because it lives there". Base `e1b3ae3c`, branch `claude/r1`.

**STATUS: REVIEWED, READY TO FREEZE (2026-10-04).** Lead read, advisor and Codex b3 critique folded (2.14, 2.15); it
becomes frozen at the lead's P0 commit on `claude/r1`. Every
number below is MEASURED (with its conditions) or marked PROJECTION.

## 1. Territory, re-taken at `e1b3ae3c` (static reads named by symbol; runtime probes in 1.4)

1.1 **The three tiers** (all in `engine/brain.py::Brain.recall_episode_durable`, in this order; the first answer wins):
  1. WORKING -- `sleep/episode_consolidation.py::EpisodeConsolidator.recall_episode`: k-WTA attractor completion over the
     resident assemblies (capacity derived by `capacity_for`; 64 in every probe below). Confidence = completion x support
     over spike assemblies. **It is not slot-verified**: the returned episode is never compared to the cue's slots. Its
     encoder hashes the whole episode, so a partial cue misses here by its own disclosed limit (tests/test_lake_recall.py).
  2. WINDOW -- `substrate/lake_recall.py::LakeRecall` attached to `Brain._episode_window` (the last `capacity` episodes
     restored from the lake at boot by `_restore_episodes`, extended by the wake tick's own appends). LAESA k-NN as a
     filter, then **verified in slot space** (`verify_slots`: matched = present and str-equal, contradicted = present and
     different, a missing slot is neither); confidence = matched / len(cue); per-arity measured null for the margin.
  3. DEEP -- `home.load_episodes()` (a full replay of `brain_lake.bin` per call), newest first, exact sub-cue:
     `all(str(ep.get(k)) == str(v) for k, v in cue.items())`, confidence 1.0, margin None. Wrapped in `except Exception:
     pass`, so a read failure is reported as `source="miss"`.
  Sources returned: "working", "lake" (window OR deep -- the label does not say which), "miss", "none" (no consolidator).

1.2 **Consumers.** One live caller: `Brain.respond`'s T057 lake fallback, `recall_episode_durable(cue, min_conf=1.0)`,
reached only when the reconstruction misses its floor. The default `min_conf=0.5` has no live caller (tests only).

1.3 **Derived-state incumbents (P2 candidates to adopt, never to duplicate).**
  - `substrate/binlog.py::BinLog` owns framing ([magic][u32 len][payload]...). Construction runs `_repair_torn_tail` and one
    full replay to fill `_seen` (the dedup set holds EVERY payload in RAM). `stale()` compares size only and deliberately
    reports False after the instance's own `rewrite()`; its docstring says a caller with a derived index "has to drop it
    itself". `home._log()` caches one instance per path.
  - `substrate/lake_bridge.py::located_records` is the one forward reader (`iter_records` delegates to it); its docstring:
    the bridge "NEVER reimplements framing (P2: one mechanism per job)". It constructs a SECOND `BinLog(path,
    dedup=False)`, which runs tail repair: the writer-vs-probe hazard its own docstring and `lake_rebuild_gate` warn about.
    For BinLog records `seq` is None ("a caller that needs one there uses the record's forward-scan ordinal").
  - `substrate/lake_index.py::LakeIndex` (T007): sqlite WAL, `records(tag, key, container, seq)` PRIMARY KEY (tag, key),
    `put_batch` with the applied watermark in the same transaction, `get(tag, key)` only (no range read),
    `logical_dump_digest`. No live writer: the T007 audit found no on-disk index over brain_lake.bin and none is built live.
  - `gates/lake_rebuild_gate.py` (T008): build -> digest -> delete -> rebuild -> equal, planted-row red, lake-tracking
    control; keys every record (tag, ordinal) with seq = ordinal.
  - `home.KnowledgeLog.consolidate` rewrites the whole lake as `foreign + facts` (sleep compaction): every record offset
    moves; episodes keep their relative order.
  - The chat's `LakeMemory(path=None)` in `reasoning/spike_chat_lake.py` is "REBUILT each boot from the durable knowledge" --
    a FACT-recall index, not one of T102's episode tiers; it is the in-tree precedent for a derived index rebuilt from the
    lake at boot (P4b: a cache, never the source of truth).

1.4 **Runtime probes, 2026-10-03** (out of tree: scratchpad `r1/probe_tiers.py`, `r1/cost_probe.py`; throwaway NORD_HOME
under %TEMP%, deleted after; `.venv-lake` 3.14.6t; the live lake was never read):
  - (a) **Deep-tier false match at the live bar.** Lake of 203 episodes, three lacking `affect`. Cue `{"when": "u1",
    "affect": "None"}` -> `source=lake, confidence=1.0`, returning the episode that HAS NO `affect` slot
    (`str(ep.get("affect")) == "None"`). The window tier had refused it (0.5 < 1.0); the deep tier trusted it.
    `LakeRecall._is_subcue` carries the same `str(ep.get(k))` shape (it gates only the calibration null's rejection
    sampling).
  - (b) Window tier at the non-live 0.5 bar: a one-slot-off full cue of a resident episode -> `source=lake,
    confidence=0.8, contradicted=1`. Reported only (no live caller uses 0.5).
  - (c) Working tier, same one-slot-off cue: confidence 0.2, not the resident episode (no false match on that probe; one
    probe is not a rate -- the audit measures it).
  - (d) **Restart collapses distinct events.** Two boots at seed 0, 100 ticks each, persist: lake episodes 10 after boot 0
    and STILL 10 after boot 1. Episodes carry `when = str(self._wake_ticks)`, and `_wake_ticks = 0` at construction, so the
    second boot formed byte-identical episodes and BinLog dedup kept one record each. The wake tick's block appends to
    `_episode_window` even when `save_episode` returned False, so in boot 1 the window held 20 entries over 10 records:
    the window diverges from the lake it claims to mirror.
  - (e) **Cost unit at real scale** (design seed 14900; warm page cache -- the lake was written seconds before):
    513.3 MB lake = 121,000 foreign 4,200 B records (tag 0x7E) + 50,000 live-shaped episodes, interleaved.

    | quantity (warm cache) | measured |
    |---|---|
    | `BinLog` construction, cold process | 0.46 s; dedup set 513 MB of payload resident |
    | `load_episodes()` full | 0.25 s, 50,000 episodes; tracemalloc peak 40.6 MB |
    | `load_episodes(limit=64)` (the boot restore) | 0.27 s (a full pass for 64 records) |
    | `make_brain(seed=0, persist=True)` on this lake | 7.2 s (2.15 s on a 203-episode lake) |
    | `recall_episode_durable` exact nonresident, min_conf 1.0 | 0.53 / 0.30 / 0.30 s (source lake) |
    | `recall_episode_durable` absent, min_conf 1.0 | 0.30 / 0.32 / 0.38 s (source miss: a correct abstention costs a full pass) |
    | prototype token->offset index, one pass | 0.29 s; 50,024 tokens; tracemalloc peak 14.6 MB |
    | prototype query p50 / p95 / max (300 cues per class) | exact-old 0.32 / 0.62 / 1.11 ms; partial 0.07 / 0.11 / 0.15 ms; absent 0.001 ms |
    | prototype == deep-scan answer (20 per class) | 20/20 exact, partial, absent; **0/20 None-trap** (the deep scan returns the false match, the index refuses) |

    The prototype used its own frame parser OUT OF TREE ONLY; the in-tree design does not (2.12 P3, constraint C1).
    PROJECTION, cold cache: `home._log`'s docstring measured BinLog construction at 1316 / 1420 / 1307 ms on the live lake at
    513.1 MB / 121,177 records (2026-08-05) against 0.46 s here, a factor of about 2.9. The live `brain_lake.bin` is
    3,318,607,701 B (a stat only, taken for the board's mtime safety check: last write 2026-09-27 01:19, untouched by these
    probes; its episode count is unknown). The deep tier is linear in lake bytes, so a live deep-tier call is PROJECTED at
    ~1.9 s warm and ~5.6 s cold, paid on every chat-path miss and on every correct abstention.
  - **Safety:** every probe set NORD_HOME to a fresh %TEMP% dir before importing `home` (board PROTOCOL 4b); the live
    home's newest mtime after the probes is still 2026-09-27 01:19 (checked 2026-10-03 20:08).

1.5 **Scope read.** T102's tiers are the EPISODE tiers. P4d's "anything in the lake" also spans facts (KnowledgeLog) and
chat (DurableChatMemory; the per-boot LakeMemory). R1 measures and bounds episode recall; widening is owner question Q2.
`load_episodes()` excludes `kind == "teach"` by default, so the deep tier cannot recall taught episodes (Q1).

## 2. Pre-registration (DRAFT; becomes frozen at the commit that lands it; later changes are dated strengthenings)

### 2.1 Question, and what a result licenses

- **Q (audit).** At a fixed lake watermark, what are the incumbent chain's coverage, trusted false matches, abstention,
  latency and memory per tier and per cue class -- through the real consumer, not a copy?
- **Q (successor).** Does a derived historical index rebuilt from the lake reproduce the deep tier's exact-match guarantee
  (slot present AND str-equal, newest first) on every cue class, survive corruption/staleness/compaction/restart without
  hiding a record, and meet the frozen cost bounds -- so that it earns replacement of the O(lake) scan (P2 ADOPT)?
- **A PASS licenses only** "the successor replaces the deep scan on this workload at these bounds". Never a claim about
  facts/chat recall, about the working tier's quality, or that recall improves learning (R3's question).
- **A null** keeps the incumbent scan (with its stacked defect fixes, 2.7 F1-F3); the successor's code is reverted (P8
  deletion on a null). It never deletes recall, the three tiers, LakeRecall, LakeIndex or any lake record: recall is an
  owner-drawn behaviour of the engine (P4d), and a null on one implementation is not a null on the part.

### 2.2 The workload (frozen generator `instruments/recall_audit.py::make_workload(seed, watermark)`)

- **Episodes**, live-shaped: who in {self (p .9), owner, arm}; what in 8 actions; where in 10 places; affect in 5;
  `when = str(5 * i)` restarting at each simulated boot (B = 3 boots per lake), so genuine cross-boot duplicates arise
  the way 1.4 (d) measured them. Plus, per lake: 20 episodes lacking `affect` (None-trap class), 10 `kind="teach"`
  episodes, 2 crc32 cell-collision pairs for `when` (`lake_recall._find_cell_collision`'s construction), and foreign
  filler records (tag 0x7E) interleaved.
- **Watermarks** (a watermark = the exact lake bytes; its `lake_bridge.stream_digest` is recorded): W_c (correctness):
  4,000 formed episodes + 2,000 x 4 KB filler (~8 MB); W_real (cost): 50,000 episodes + 121,000 x 4,200 B filler
  (513 MB, the live composition last measured; the 50,000 episode count is a PROJECTION); W_live (scale): the live
  lake's byte size by stat, 3.32 GB, at W_real's record mix and episode fraction, on two seeds only.
- **Schema:** episodes written after F3 carry the shared session slot (2.7 F3); W_c mixes legacy (no session slot) and
  session-bearing episodes so both schemas are covered.
- **Recallable set:** the oracle and every arm use ONE predicate for "an autobiographical episode the recall surface
  covers", imported from `home`, never restated: today `load_episodes`' inline `kind != "teach"`; from R2's P1 commit on,
  `home.is_autobiographical_episode(episode) -> bool` over `home.LEARNER_EVENT_KINDS = frozenset({"teach", "turn",
  "sleep", "percept"})` (R2, D5; learner events are `_T_EPISODE` records with a `kind` slot, and the predicate lands in
  the same commit as the first writer of a new kind). W_c therefore also writes learner-kind records as distractors.
- **Cue classes per seed** (counts at W_c; expected answer = the ORACLE's match set, 2.3):
  C1 resident-exact 100 | C2 window-partial 100 | C3 nonresident-exact 200 | C4 nonresident-partial (`when` + 1-2 slots)
  200 | C5 ambiguous (no `when`, 2-3 slots, >= 2 matching episodes) 200 | C6 absent 200 = 100 novel values + 50 None-trap
  (`{when of a slot-lacking episode, affect: "None"}`) + 25 collision ghosts + 25 extra-slot (an exact cue plus a slot no
  episode has) | C7 cross-boot duplicate 50 | C8 non-autobiographical kinds 40 (exact cues of teach and learner-kind
  records; expected MISS under the imported predicate -- barred as a FALSE if answered; Q1 can change the predicate in
  `home`, never this bar) | C9 corrupt/stale scenarios
  (successor arms, 2.4 B4) | C10 restart (every cue re-asked in a NEW process over the same lake).
- **Splits (frozen before any tuning):** dev seeds tune nothing that is barred; the bars in 2.4 are frozen in this text.

### 2.3 The control that is not the method: the generator-side ORACLE

The oracle is the workload generator's own truth table: the set of DISTINCT episodes it wrote (dedup by the sorted
(str(slot), str(value)) tuple -- `pack_episode` stringifies every value, so `{"when": 5}` and `{"when": "5"}` are one
record -- computed in the generator, never by BinLog), with presence-checked matching (`k in ep and
str(ep[k]) == str(v)`). It imports no retrieval or storage mechanism (`binlog`, `lake_recall`, `brain`, home's
readers); its one import from `home` is the recallable-set predicate (2.2), a scope definition, and the generator's own
per-episode recallability label is asserted equal to that predicate on every written episode (a teach episode is the
positive case). Every answer is scored against it: VERIFIED-CORRECT (the returned episode is in the cue's match set),
FALSE (returned, not in the set), MISS. **Its positive control runs first** (2.7 I1): on the trap-free classes it agrees
with the incumbent deep scan on every cue, and a planted wrong answer is scored FALSE. (Precedent, out of tree: the cost
unit's prototype, which applies the oracle's presence-checked rule, agreed with the deep scan on 60 of 60 trap-free cues
and disagreed on 20 of 20 None-trap cues.)

Beside it: the incumbent chain unchanged; a SCAN-ONLY arm (the deep tier alone); per-tier NEUTER arms (working neutered by
an instance patch returning a miss; window neutered by an empty `_episode_window`; deep neutered by a `load_episodes`
patch returning []), giving per-tier attribution by neuter rather than by the total.

### 2.4 Endpoints and bars (primary at `min_conf=1.0`, the live caller's bar; 0.5 reported beside it)

Per path P and seed s: L = present cues answered MISS (lost); F = cues answered FALSE at >= min_conf (trusted false);
A = absent cues answered MISS / absent cues (correct abstention); per-class latency p50 / p95 / max per tier and end to end;
per-call tracemalloc peak (a separate pass); standing memory (the index's resident size; BinLog's dedup set reported as
context). **Correctness is deterministic, so there is no statistic: each bar is a count that must hold on EVERY
confirmatory seed (a conjunction; no majority rule).**

The successor chain S (working + window + successor deep tier) replaces the scan only if, on each of 14100-14107:
- **B1 exactness:** L = 0 and F = 0 over C1-C8; A = 1 on C6 and C8.
- **B2 guarantee reproduction:** on C3-C6 the successor deep tier returns EXACTLY the fixed scan's episode (the same
  record, newest-first tie-break), not merely a member of the match set.
- **B3 rebuild:** delete the derived index -> rebuild from the lake -> logical digest equal; a planted row moves the digest;
  one appended episode moves the rebuilt digest (the `lake_rebuild_gate` pattern, both controls in the same run). Across a
  compaction offsets legitimately move, so the cross-compaction comparison is LOGICAL: the (token -> episode content, in
  episode order) map before `KnowledgeLog.consolidate` equals the map rebuilt after it (consolidate preserves episodes
  and their order: `home`'s self-check, "consolidate resequenced or dropped an episode").
- **B4 corruption / staleness never hides a record:** under each of {index file truncated; bytes garbled; a planted
  wrong-offset row; one episode's token rows deleted; index watermark ahead of the lake; the lake appended behind the
  index; the lake rewritten under the index by `KnowledgeLog.consolidate`; a torn lake tail}, every present cue is answered
  VERIFIED-CORRECT or the call refuses visibly (`source="refused"` with a reason) -- never MISS, never FALSE.
- **B5 restart:** a new process over the same lake returns identical (episode, tier) answers on every cue; the window
  rebuilt at boot equals the last `capacity` lake episodes.
- **B6 cost at W_real (warm cache, the measured condition of 1.4 (e)):** successor deep-path p95 <= 25 ms AND <= the
  incumbent deep-path p50 / 10, both measured interleaved (ABAB) in the same process; per-call traced peak <= 1 MB;
  standing index RAM <= 1 KB per episode; `make_brain` wall-time delta vs the incumbent <= +1.0 s; scale (two seeds):
  p95(W_live) / p95(W_real) <= 2 for a lake 6.5x larger (the incumbent's ratio reported beside it; PROJECTED about 6).
- **B7 load witness (VOID, never pass/fail):** a fixed CPU spin is timed before and after each cost arm; a drift beyond 50%
  VOIDS that seed's cost reading; it is re-run once on the same seed; a second VOID halts and reports (no reserve swap for
  load). Under the G1(c) campaign's load a VOID is the expected failure shape (memory: a fixed-epsilon timing control reds
  under host load).

The incumbent AUDIT records the same L / F / A / latency / memory table for the incumbent chain, the scan-only arm and each
neuter, on dev seeds 14000-14003 (P1) and on the confirmatory seeds at the deciding run (paired, same lake bytes and same
cue list per seed). Its expected findings are pre-stated so they cannot be relabelled after the fact: F > 0 on the C6
None-trap (deep tier); B5's window-equality red on C10 (the window appends deduped episodes); a cross-boot collapse count
> 0 on C7.

### 2.5 Decision rule (P2)

- **ADOPT** the successor iff B1-B6 hold on all eight confirmatory seeds (VOIDs re-run per B7), every red in 2.7 fired in
  its build arc, and the Codex panel leaves no unrefuted blocker. The deep scan's code is then DELETED (its unique
  capability -- exact sub-cue over the whole lake, newest first -- carried by the successor and proven by B2); the
  successor's cold path (a rebuild from the lake) is its own fallback, not a second answering mechanism (P2's
  merge-must-reduce guard: one deep mechanism survives). Part F's control, the canonical exact scan, survives as the
  instrument-side reference in `recall_audit.py` (it answers no live query), so B2 stays checkable after adoption.
- **KEEP** otherwise: the fixed incumbent scan stays THE deep tier; the successor's modules, rows and interfaces with no
  other consumer are reverted; this file records which bar failed and by how much (the measured disposition the row asks
  for). The audit instrument stays (P3 disposition 3: benchmark harness, exempt with reason).
- The working and window tiers are not candidates for replacement here (the row: "preserve the three recall tiers").

### 2.6 Null disposition (each is a causal claim; the instrument that owes its positive control is named)

- B1/B2/B4 fail -> the named cause is the successor's completeness/staleness logic (tail fold, compaction detection). The
  instruments are the ORACLE (I1) and the rebuild check (S7); both owe their positive controls BEFORE the deciding run. If
  either control fails, the reading is a harness defect, not the successor's null.
- B6 fails -> the named cause is the successor's per-query I/O path. The latency harness (I3) and load witness (I5) owe
  their positive controls first; a VOID is never read as a null.
- An incumbent finding (2.4's expected list or any other) is a claim about the incumbent only after I1 and I2 passed.

### 2.7 Checks, each with a constructed, reachable red (a red that does not fire halts its arc)

Instrument (P1): **I1** oracle -- agreement with the incumbent on trap-free classes; red: a planted wrong episode scored
FALSE. **I2** tier attribution (wraps, no brain.py edit: `LakeRecall.recall` and `home.load_episodes` counting wrappers)
-- a nonresident exact cue attributes `deep`, a window-partial `window`, a resident exact `working`; red: a wrap on the
wrong attribute must read 0 on a known-deep cue and fail its own check. **I3** latency -- a stub tier with a planted 50 ms
sleep reads p50 >= 50 ms; red: a span timed outside the call. **I4** memory -- a known 10 MB allocation reads within 10%.
**I5** load witness -- artificial spinners must VOID. **I6** determinism -- the same seed gives a byte-identical lake digest
and cue list, another seed does not. **I7** throwaway-home guard -- every arm runs inside the incumbent
`home.throwaway_home()` (board PROTOCOL 4b; not a second guard), and the instrument additionally refuses any NORD_HOME
ending `.snn_data_lake`; red: that path is refused; the live home's newest mtime is recorded before and after each run.

Incumbent fixes (P2, stacked commits, each with its red, BEFORE the equivalence run -- P2's honesty guard needs the A/B on
equivalent semantics): **F1** the deep tier's `str(ep.get(k))` becomes presence-checked (and `LakeRecall._is_subcue`'s same
shape); red: 1.4 (a)'s cue at confidence 1.0 (fires today). **F2** the window append is gated on `save_episode`'s return;
red: 1.4 (d)'s window over-count (fires today). **F3** episode identity across boots (a P4b loss: distinct events
collapsed): the wake tick's episode carries a `session` slot from ONE per-lake session counter shared with R2 (P2: R2
computes it once at construction as 1 + max session over learner events AND episodes -- today G0(c)'s `_teach_session` --
and exposes one Brain attribute; R1's writer reads it; R1 adds no second counter); red: 1.4 (d)'s two-boot run leaves
10 lake episodes where 20 were formed (fires today); fixed -> 20. Matching stays sub-cue: a cue that omits `session`
matches every session (C7 becomes an AMBIGUOUS class, answered by any member), a cue that names it is verified on it.
F3 waits for R2's counter attribute or is built by whichever row lands first, never twice.

Successor (P3): **S1** reader -- `replay_located` offsets equal independently summed frame lengths and `read_at(off)`
equals the payload; red: an offset taken at the payload instead of the frame. **S2** `replay()` delegating to the located
reader keeps its contract (append order, stop at a torn frame, NO repair side effect -- R2's reader and R-R2 depend on it)
and leaves `lake_bridge`'s round-trip digest unchanged; reds: a reader dropping one record moves the digest; a reader that
continues past a torn frame or truncates the file is caught by a torn-tail fixture whose bytes are hashed before and after. **S3** tail fold;
red: fold disabled loses the newest nonresident records under B1. **S4** verification; red: candidates returned
unverified plus a planted wrong-offset row produce a FALSE caught by B1. **S5** compaction detection (an anchor checksum of
the record at the watermark, or a generation); red: detection disabled + `KnowledgeLog.consolidate` -> wrong or absent
answers caught by B4. **S6** refuse visibly; red: a variant returning MISS on an sqlite error is caught by B4. **S7**
rebuild equality with planted-row red and lake-tracking control. **S8** the cost bar is reachable; red: the index
bypassed (a full pass per query) fails B6 at W_real. **S9** no second BinLog; the builder reads through `home._log()`'s
cached instance; red: a builder constructing `BinLog(path)` is caught by a construction-count wrap.

### 2.8 Seeds (R1's block 14000-14999, SEEDS.md)

| set | use | status |
|---|---|---|
| 14000-14009 | dev: harness development, the P1 incumbent audit on 14000-14003 | EXPOSED |
| 14100-14107 | confirmatory, S = 8, read once at the deciding run | RESERVED |
| 14108-14115 | reserve: replaces a confirmatory seed only for a harness defect (named in the record) | RESERVED |
| 14200-14239 | planted reds, one seed per red arm | EXPOSED |
| 14300-14307 / 14308-14315 | fresh confirmatory / reserve block for a SECOND successor implementation only, after the first is judged on 14100-14107 (2.14, Codex 12) | RESERVED |
| 14900-14909 | design measurements; **14900 used 2026-10-03** (1.4 (e)) | EXPOSED |

Pairing: every arm on a seed reads the same lake bytes and the same cue list. Grep (the Grep tool, 2026-10-03, whole
worktree, json/npz/bin excluded): zero seed-context hits for `14\d{3}` (`seed` within 40 characters); positive control,
the same regex shape on 4242|203 -> 13 hits in 3 files. Disclosure: a Bash `grep` of the same block returned 0 with its
positive control ALSO at 0, and a later Bash grep mangled its regex; the command-rewriting hook is suspected, so every count
cited here comes from the Grep tool.

### 2.9 Dependencies

Upstream for the deciding run: none (DEPENDENCIES.md). Interfaces shared with R2 (T103), all agreed in section 3: R2
relies on `BinLog.replay()`'s contract, which R1's reader change must keep (S2; R2 needs no offsets, D2); R1 imports R2's
`home.is_autobiographical_episode` once R2's P1 lands (D5) and R2's session counter attribute for F3 (D4); `engine/brain.py`
passes between the rows by claim handoff through main (D1). F3 and the C8 learner-kind distractors wait for R2's P1;
nothing else does. Downstream (plan): A1/T105 records the workload after R1; N4/T099 needs accepted R1-R3 if it picks
durable recall; G6's saved problem data and G9's engine loop end at "lake -> recall (R1/R2)".

### 2.10 Cost at real scale (measured 1.4 (e); the rest PROJECTION)

Per confirmatory seed: W_c write < 1 s; ~1,400 cues x (incumbent, scan-only, three neuters, successor) through one
`make_brain` per arm (2 s each) on an 8 MB lake -> about 2 min. W_real: write 0.4 s; incumbent cost arm 40 cues per class
x 4 classes x 0.3 s ~ 50 s; successor 300 cues per class < 1 s; index build 0.3 s. W_live (two seeds, 3.32 GB each):
write ~3 s, incumbent 40 x 4 x ~1.9 s ~ 5 min each. Whole deciding run PROJECTED ~40 min wall, ONE process at a time
(host load; never beside a pytest file), ~11 GB written sequentially and deleted per seed. Throwaway homes go on D:
(834.6 GB free on 2026-10-03; C: had 77 GB); the run's preflight refuses to start below 15 GB free on its target drive.

### 2.11 Arc split and the files each part touches (claims on the board; shared hot files claimed at their commit)

- **P0 docs:** this file; the plan's R1 dated note (orchestrator-only).
- **P1 instrument:** NEW `src/snn_data_lake/instruments/recall_audit.py` (workload, oracle, attribution wraps, latency /
  memory / load witness, JSON report; refuses the live home), NEW `tests/test_recall_audit.py`; registration in the same
  commit: `src/snn_data_lake/__init__.py` (REGIONS), `run_selfchecks.py` (bare self-check), `gates/dormancy_gate.py`
  (EXEMPT: benchmark harness) -- shared hot files, claimed at that commit. Then the incumbent audit on 14000-14003,
  recorded in section 3.
- **P2 incumbent fixes:** `engine/brain.py` (R2 holds the claim; claim handoff through main between commits, D1) for
  F1 (`recall_episode_durable`), F2 and F3 (the wake tick's episode block); `substrate/lake_recall.py` (`_is_subcue`) for
  F1. One commit each, red first. R2's brain.py hunks (construction block, `_record_teaching`,
  `_flush_interaction_learner`, `_sleep_tick`'s replay block) are disjoint by symbol.
- **P3 successor** (built only after P1's record; the design measurement on 14901-14903 runs first):
  (a) `substrate/binlog.py`: an additive `replay_located(start)` -> (offset, payload) and `read_at(offset)` on the
  cached instance, `replay()` delegating with its contract unchanged (ONE framing implementation; R2 reads only
  `replay()`, D2); (b)
  `substrate/lake_index.py`: an additive range read (R2 edits neither file, D0); (c) the episode-token rows -- PROPOSED
  inside the one derived index (`lake_index.db`, a reserved tag, seq = byte offset) so one derived index has one rebuild
  gate (Q6). The token-row BUILDER lives in NEW `substrate/episode_index.py` and takes a BinLog INSTANCE, never a path:
  live, the brain passes `home._log()`'s cached instance (C2); `lake_rebuild_gate` gains an arm that calls the same
  builder with the instance it owns on a temp or quiesced root (its `--root` restriction stands). `build_index`'s
  pointer rows and `lake_bridge.located_records` stay as they are: `located_records` yields `seq=None` for BinLog records
  and constructs a second BinLog, so it can neither supply offsets nor run live. The same module answers queries
  (posting intersection -> `read_at` -> slot verification; tail fold from the watermark; compaction detection; refuse
  visibly), with NEW `tests/test_episode_index.py`;
  (d) the deciding run on 14100-14107; (e) ON ADOPT ONLY, `engine/brain.py`: the deep tier calls `episode_index`, the scan
  is deleted, `_restore_episodes` takes its slice from the index's boot pass. The in-RAM posting-array variant (the
  prototype's shape) is built only on a MEASURED LakeIndex failure of B6 (the plan: "use incumbent lake_index/lake_recall
  mechanisms where suitable, with a measured disposition if they do not fit").
  **Constraints carried into P3:** C1 no second frame parser (framing is BinLog's, P2); C2 no second BinLog construction
  (the writer-vs-probe hazard: read through `home._log()`); C3 compaction moves every offset and `stale()` cannot see it
  (S5 is mandatory); C4 an approximate candidate never carries confidence -- every answer is verified against the record
  bytes read from the lake.
- Each arc: registrations in the commit that adds a module, the census after every symbol-adding commit, the
  orchestrator's Codex gpt-6.1-sol panel before the deciding run, a fold stated as re-panelled or battery-verified only,
  ONE battery on the stacked head (orchestrator), closing record last.

### 2.12 Scope

Built: 2.11. Not built: facts/chat recall bounds (Q2); any change to the working tier's confidence (Q5); the session
counter itself (R2's, shared); BinLog's resident dedup set (Q7); a similarity (non-exact) deep-history index -- the window
tier keeps similarity.

### 2.13 Open questions (for the owner via the orchestrator)

- **Q1** `kind == "teach"` episodes are excluded by `load_episodes()`' default, so no tier recalls them. P4d says
  "anything in the lake". Include them in the recall surface? (R1 reports C8; it changes no semantics unasked.)
- **Q2** Widen R1 to a report-only coverage census of fact and chat recall (P4d's "anything"), no successor?
- **Q3** The live lake's episode count (R1 never reads its contents; its size, 3.32 GB, came from the mtime stat): a
  count from a quiesced backup would replace the 50,000-episode PROJECTION in W_real.
- **Q4** Cross-boot identity (1.4 (d)) is fixed by R1's F3 with R2's shared session counter (D4). Confirm that adding a
  `session` slot to new episodes (legacy episodes keep none) is acceptable: it changes episode bytes from the fix onward.
- **Q5** The working tier is not slot-verified. Keep as is (the row says preserve) unless the audit measures a trusted
  false match at min_conf 1.0 -- then a stacked verification commit like F1?
- **Q6** One derived index file (`lake_index.db` + token rows + one rebuild gate) or a separate episode index file? And
  may a derived file live in the home at all? P4b's sidecar "family" (`home.save_state`'s docstring) was STATE beside
  `brain_lake.bin`; a rebuildable index is a cache, which P4b allows, but it would be the first derived file there. No
  gate enumerates the home's permitted contents (Grep tool, 2026-10-03, over src/; the same pattern fired on
  `home.save_state`'s "beside brain_lake.bin" docstring as its positive). The alternative needing no file is the in-RAM
  index rebuilt at boot (the chat LakeMemory precedent; 0.29 s and 14.6 MB at W_real measured), which P3's design
  measurement (14901-14903) costs against LakeIndex before either is built.
- **Q7** BinLog keeps every payload resident for dedup (513 MB at W_real): outside R1; a row of its own (A1/T105)?

### 2.14 Amendments to 2.2-2.11 (lead read 2026-10-03; Codex b3 critique folded 2026-10-04)

Source: R1's self-defects D-a..D-e (board `p0_reports\R1.md`), the lead read L1-L5 there, lead ruling 16, and the Codex
b3 critique's R1 findings 1, 10, 11, 12, 25 (and finding 13, filed on R2, which binds R1's F3). Each item amends the
section it names; where it conflicts with the earlier text, this text wins.

- **Codex 1, BLOCKER (amends 2.3 scoring, 2.4 B1/B4, D-a, F4).** Adding `source="refused"` made the outcome set
  incomplete: a refusal is neither MISS nor FALSE, so a successor refusing a present deep-required cue (e.g. C7) escaped
  L and F, and B2 does not cover C7. Fixed: the oracle scores FOUR outcomes -- VERIFIED-CORRECT, FALSE, MISS, REFUSED --
  and any other result (an exception escaping the call, no output, a malformed result) is scored ERROR. On an INTACT
  workload (C1-C8 and C10; every arm except B4's planted faults), **L counts every present cue answered MISS, REFUSED or
  ERROR**, and an absent cue answered REFUSED or ERROR counts against A (A = absent cues answered MISS / absent cues).
  B1 on an intact workload therefore requires every present deep-required cue to be VERIFIED-CORRECT and every absent
  cue to be MISS. A visible refusal is PERMITTED only inside B4's explicitly faulted arms (where it is the required
  alternative to VERIFIED-CORRECT). The same rule binds the incumbent after F4: a REFUSED on an intact workload is a lost
  record for the incumbent too. Red: a successor variant that refuses one present C7 cue on an intact W_c fails B1
  (L = 1); the same variant inside a B4 faulted arm passes B4.
- **D-a (amends 2.4 B1/B2, 2.6).** The working tier is preserved, not
  replaced, and is not slot-verified (1.1), so a working-tier FALSE at confidence 1.0 must not fail the successor for a
  cause outside it. B1's L/F counts and B2 are barred on DEEP-ATTRIBUTED answers (I2's attribution) plus the cues the deep
  tier should have answered (every present cue the working and window tiers returned MISS on); the chain-level L/F/A table
  is reported beside it for every arm. A FALSE attributed to the working (or window) tier is an incumbent finding routed
  to Q5, identical across the incumbent and successor arms by construction (same lake, same cue list, same tiers above the
  deep tier), and is never read as a successor null. Red: a planted working-tier FALSE (instance patch returning a wrong
  episode at confidence 1.0 on one cue) leaves B1 on deep-attributed answers unchanged and shows in the chain-level table.
- **D-b -> F4 (amends 2.7 incumbent fixes, 2.5, 2.11 P2).** The deep tier's
  `except Exception: pass` in `Brain.recall_episode_durable` reports a lake read failure as `source="miss"`: a hidden
  record (R-R1, "corrupt/stale index hides records", applied to the incumbent's own read). **F4**, a stacked incumbent fix
  beside F1-F3 and landed BEFORE the equivalence run so the A/B compares equivalent semantics (the successor's S6 already
  refuses visibly): a read failure returns `source="refused"` with the exception's type and message as the reason, never
  "miss". Red: a planted read error (a `home.load_episodes` patch that raises) answers "miss" today (fires today) and
  "refused" after F4. Disclosed beside it, not fixed under F4: `_restore_episodes` swallows the same way and sets
  `_episodes_restored = 0` silently. The audit's restore census (Codex 25, MINOR, folded: the lake's TOTAL count was the
  wrong reference, since the restore is capacity-limited) compares `_episodes_restored` with the EXPECTED count
  min(capacity, the oracle's recallable distinct episodes at the watermark), and compares the restored slice's CONTENTS
  (the attractor's stored episodes and `_episode_window`, in order) with the oracle's newest-first recallable episodes
  truncated to capacity -- the oracle, not `load_episodes`, so the census does not grade the reader with itself. A
  mismatch is a reported finding; any fix is a separate stacked commit with its own red, proposed after the P1 audit.
  Red: a correct 64-episode restore from W_c reads no mismatch; a planted `load_episodes` raise at boot reads restored 0
  against expected 64 and is flagged.
  F4 touches `engine/brain.py` (`recall_episode_durable` only), under the D1 claim handoff.
- **D-c (amends 2.4 B5, 2.2 C10; schedule corrected by Codex 10, MAJOR).** B5 as written compares boot to boot; both
  boots restore from the same lake, so it checks determinism only and misses the wake-time divergence F2 fixes (1.4 (d)).
  Added arm **B5w**, on the 1.4 (d) schedule: TWO boots over one throwaway home, each `make_brain(seed=0, persist=True)`
  followed by the SAME N = 100 wake ticks and a persist; after EACH boot's wake ticks, the in-process `_episode_window`
  must equal the newest `capacity` recallable episodes of the lake as the oracle reads the lake bytes (newest first),
  checked in a NEW process; then B5 as before. Codex 10's point is correct: a single fresh-home boot never makes
  `save_episode` return False, so it cannot exercise F2. A further subtlety it implies: once F3 lands, boot 2's episodes
  carry a new session and are no longer duplicates, so the F2 red needs duplicates by construction. **F2's red therefore
  runs B5w with F3 neutered** (the session slot omitted -- exactly the pre-F3 1.4 (d) condition) **and F2 reverted**:
  boot 2's window must then over-count (20 window entries over 10 lake records) and fail B5w. The live B5w arm runs with
  all fixes and must pass after both boots. A second red: B5w with F3 neutered and F2 FIXED passes (the window does not
  over-count even when duplicates are formed), so F2 is attributed by its own revert, not by F3's.
- **D-d + L2 (amends 2.4 B6/B7, 2.10).** The deciding run (P3 (d)) and every
  cost arm (B6, B7, the W_real and W_live passes) are GATED on the lead's "campaign DONE" message AND an explicit host
  release by the lead; the run is one process at a time (~40 min, ~11 GB written sequentially). Before the first cost arm,
  B7's load witness runs as a pre-check; a VOID there means the host is not released and the run does not start. The P1
  incumbent audit's correctness arms (W_c, ~2 min per seed) are also gated on "campaign DONE"; nothing in this file runs
  while a campaign is recording.
- **D-e (amends 2.7 I7; implemented by the shared guard below).** I7 refuses an UNSET or empty `NORD_HOME` (the
  default home is the live lake) as well as any `NORD_HOME` resolving to `C:\Users\vinhl\.snn_data_lake` (resolved path,
  case-insensitive, not a suffix match). Reds: an unset `NORD_HOME` is refused; a path ending `.snn_data_lake` is refused;
  a fresh `throwaway_home()` is accepted (the positive).
- **I7 shared guard (amends 2.7 I7, 2.11 P1; lead ruling 16, shared with G5 / Codex #14 on G5; NOT seen by the Codex
  b3 critique -- written after that batch was sent; see the Review record).** A prefix check on `selfcheck_home_` is not isolation: a prefixed junction or symlink can
  alias the live lake. ONE guard, in `substrate/home.py`, built by R1 in its P1 `home.py` commit (after R2's new kinds,
  OWNERS.md landing order), adopted by every row (P2; no row writes a second one):
  (1) `assert_not_live_home(path, live_home=None) -> Path`: refuses an empty or None path; resolves `path` with
  `strict=True` and `live_home` (default `~/.snn_data_lake`) with `strict=False`; after `os.path.normcase` refuses equality
  or containment in EITHER direction (`os.path.commonpath`); returns the resolved path.
  (2) `throwaway_home(live_home=None, parent=None)` creates its temp dir under `parent` (`TemporaryDirectory(dir=parent)`;
  None keeps today's system-TEMP behaviour for every existing caller), calls (1) on it before setting `NORD_HOME`, and
  sets the RESOLVED path, so every existing caller is guarded by the one mechanism. `parent` is Codex 11's fix (below).
  (3) `require_isolated_home(live_home=None) -> Path` for harnesses that set `NORD_HOME` themselves: refuses an unset or
  empty `NORD_HOME` (the default home is the live lake), then applies (1); called before any lake-binding import or
  Brain construction. R1's `recall_audit` calls it at entry; its own suffix check above is replaced by it.
  Every refusal raises ONE named type, `home.LiveHomeRefused(ValueError)` (catchable as either; the message names the
  offending resolved path and the rule: equal / contains / contained / unset), so an adopting harness prints it as a
  refusal (G5 confirmed the contract and names, debate D7).
  Reds (all on a DECOY live dir passed as `live_home`; no test touches the real live path): a `selfcheck_home_*` junction
  (Windows `mklink /J`) or symlink (POSIX) to the decoy is refused; a subdirectory of the decoy is refused; the decoy's
  parent is refused; an unset `NORD_HOME` is refused by (3); a fresh `throwaway_home()` is accepted (the positive).
  **Golden check:** `throwaway_home()` is code the G1(c) campaign executed, so this change triggers 2.11's golden check at
  integration; the expected result is BYTE-IDENTICAL campaign output (the guard only refuses, `parent=None` keeps the
  campaign's temp location, and on an accepted temp dir the change is only the resolved path it sets; if resolution changes the string -- e.g. an 8.3 short name under
  %TEMP% expanding -- that is named in the integration record, and any non-identical golden output is a red that halts
  the landing).
- **L3 + Codex 11, MAJOR (amends 2.10, I7).** Throwaway homes for W_real and W_live go on D: ONLY, never on C:. Codex 11
  is correct: `throwaway_home()` calls `TemporaryDirectory` with no parent, and this host's TEMP is on C:, so "use the
  incumbent helper" silently put a multi-GB home on C:. Fixed through the ONE helper (no second): `recall_audit` calls
  `throwaway_home(parent=<a fixed D: scratch root named in the run's config>)`, and the preflight checks the ACTUAL
  created home -- the drive of the RESOLVED path `throwaway_home` set in `NORD_HOME` must be `D:`, and `shutil.disk_usage`
  of that resolved path must show >= 15 GB free -- not an intended target variable. Reds: `parent=None` on this host
  (TEMP on C:) is refused by the preflight for a W_real/W_live arm; a D: parent with a planted free-space reading below
  15 GB is refused; a D: parent with room is accepted. W_c correctness arms (~8 MB) may use the default parent.
- **Codex 12, MAJOR (amends 2.8 seeds, 2.11 P3).** Successor SELECTION is separated from confirmation. (i) The
  implementation (LakeIndex token rows vs the in-RAM posting array, Q6) is chosen and FROZEN on design seeds 14901-14903
  and dev seeds 14000-14009 only, before any confirmatory seed is read; P3's design measurement already costs both shapes
  against B6 there. (ii) The confirmatory block 14100-14107 judges ONLY the frozen choice, once. (iii) If the frozen
  choice fails any bar on 14100-14107, that block is spent: the disposition is KEEP for that implementation (recorded
  with the failing bar), and no reading from it guides a second build. A second implementation (e.g. the RAM variant after
  a LakeIndex B6 fail) may then be built only by a dated strengthening of this file that is selected on dev/design seeds
  and judged on the FRESH predeclared block **14300-14307** (reserve 14308-14315), never on 14100-14115. Grep (the Grep
  tool, 2026-10-03, whole worktree): zero seed-context hits for `14\d{3}` (2.8), which covers 14300-14315; the positive
  control fired on 4242.
- **Codex 13, MAJOR (filed on R2; binds R1's F3 and D4).** The shared session counter is ONE definition: 1 + the maximum
  `session` over ALL session-bearing records -- learner events AND autobiographical episodes -- as already agreed in D4;
  R1 consumes R2's attribute and adds no counter. Added to F3's reds: two boots that write autobiographical episodes and
  NO learner events must get distinct sessions (boot 2's session = boot 1's + 1) and leave 20 lake episodes, not 10; a
  counter computed over learner events only fails it. R1 has told R2 (board message) that R1's F3 bar depends on that
  definition.
- **L4, L5 (recorded, no amendment).** F3's "built by whichever row lands first, never twice" is decided by the lead at
  integration. Q1-Q7 go to the owner batch (board `OWNER_QUESTIONS.md`).
- **Ruling 9 (recorded).** R1's bars are deterministic counts (2.4: a conjunction over seeds, no statistic), so the
  degenerate-contrast convention has no paired contrast to apply to; if any R1 endpoint ever becomes a paired contrast, a
  zero-variance contrast reads NOT MET.

### 2.15 Review record

- **Lead read:** 2026-10-03, board `p0_reports\R1.md`, section "=== LEAD READ ===" (L1-L5); lead rulings 9-11 and 16 in
  board `DEPENDENCIES.md`.
- **Advisor read:** CLI lead session, Fable, 2026-10-03 (consulted by the lead).
- **Codex critique:** model gpt-6.1-sol, batch b3; reply file `scratchpad\codex_p0\b3.reply.txt` (S =
  `C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\161935bb-36f0-4e8d-8a38-0411dcc64d0f\scratchpad`), sha256
  `1c8abc4a99bfa18c7e1d97d042bb1240c8b4cf8f893f88277bec5c38fae52e57`; findings rendered in `codex_p0\b3.critique.md`.
  Static review; no code ran. R1 tally: BLOCKER 1, MAJOR 3, MINOR 1, plus finding 13 (filed on R2) binding R1's F3.
- **Dispositions (all folded in 2.14, none refuted):** 1 BLOCKER -- fixed (REFUSED/ERROR scored; L counts them on an
  intact workload; refusal allowed only in B4's faulted arms; red registered). 10 MAJOR -- fixed (B5w = two boots on the
  1.4 (d) schedule; F2's red runs with F3 neutered so duplicates exist; F2 attributed by its own revert). 11 MAJOR --
  fixed (`throwaway_home(parent=)` in the one helper; preflight on the actual resolved home's drive and free space).
  12 MAJOR -- fixed (selection frozen on design/dev seeds; 14100-14107 judges once; a second implementation goes to the
  fresh block 14300-14315). 25 MINOR -- fixed (restore census against min(capacity, oracle count) and the oracle's
  slice). 13 MAJOR (R2's) -- the one counter over learner events AND episodes stated; a no-learner-event two-boot red
  added to F3.
- **Not seen by Codex (disclosed):** the I7 shared-guard item (lead ruling 16; G5 debate D7) was written AFTER batch b3
  was sent, so Codex did not review it; the fold's own text (this 2.14 as folded) is likewise unreviewed. Both are
  battery-verified only unless the lead re-panels them.

### 2.16 Dated amendment 2026-10-04 (pre-data; after the freeze at f9f7c998) -- the shared session counter's call site

A strengthening; no data has been read under this file. Amends F3 (2.7), Codex 13 (2.14) and D4. The counter is R2's
`home.next_session()` = 1 + the maximum `session` slot over ALL session-bearing `_T_EPISODE` records (learner events AND
autobiographical episodes); R1 imports it and defines none. Because it reads the lake, a second call after the boot's
first write returns a higher number, so it is called ONCE per boot, at Brain construction, and stored in
`Brain._session`; R1's wake-tick episode writer and R2's learner-event writer both read `Brain._session` and never call
`next_session()` during a boot. The episode slot is named `session`. Added to F3's reds (R2 registers the same two in its
own 2.12): within one boot, learner event -> episode -> learner event carry ONE session (red: an arm calling
`next_session()` per write fails it); boot 1 writing autobiographical episodes only gives boot 2 a strictly greater
session. Agreed with R2 (debate D8).

### 2.17 Dated amendment 2026-10-04 (pre-data; disclosed deviation, lead ruling 20) -- I5's red uses planted timings

Additive; no data has been read under this file. 2.7 I5 registered "artificial spinners must VOID". The built red
(`tests/test_recall_audit.py::test_load_witness_voids_on_drift_and_passes_when_steady`, and the module's bare
self-check) instead feeds PLANTED witness timings to the one verdict function (`recall_audit.load_verdict`): a
1.0 s -> 1.6 s drift and a 1.0 s -> 0.4 s drift read VOID, 1.0 s -> 1.2 s reads OK. Reason: on this 28-core host a
reliable live drift needs enough spinner processes to saturate every core, which is itself the shared-host load the
board's PROTOCOL 5 forbids, and a spinner count short of that makes the red flaky rather than reachable. What the
planted red does not prove, said plainly: that `spin_witness()` itself slows under real contention. That half is read
at the deciding run instead: B7's pre-check (2.14 D-d) records the witness on the released host, and any confirmatory
VOID is reported with both readings. The verdict rule, the 50 % tolerance and B7's re-run-once-then-halt are
unchanged.

### 2.18 Dated amendment 2026-10-06 (pre-data on 14100-14107; cp4 fold, F3b, workload actor probability)

Additive; no line above is changed. Nothing on 14100-14107 or any reserved seed has been run.
- **cp4 fold (Codex gpt-6.1-sol panel 2026-10-04, `codex_p0\cp4.final.txt`; nine items name R1: #1 BLOCKER, #2/3/5/6/8/10
  MAJOR, #14/15 MINOR; all reproduced, none refuted).** Landed as nine commits on claude/r1, each red first:
  72fbb8a4 (#1 output root through the shared guard + confined to the active home), 7a1ce292 (#8 live-home stamp before/after,
  exception path included, the gate's own `_live_home_stamp`), 74e7088c (#2 B5w graded on lake bytes read by a NEW
  process), a5039fce (#3 B5 compares the returned episode), 34e0bf08 (#5 I1/I2/I6 real controls, a failure raises
  `ControlFailed` before any finding), 3708fde3 (#6 ERROR partition), 5f5dc88a (#10 outcome-by-tier), 9b564fa9 (#14), 200aee0d (#15).
- **#14 (P(self) = 0.9).** 2.2 said 0.9; the build drew uniformly from a 9-of-11 list (0.818). Now `rng.random() < 0.9`;
  every seed's generated stream changed. No 14100-14107 data had been read.
- **F1-F4 landed** (6f6c4a66 + eb9befcd F1, 3fcb2ba5 F4, 1df6cd61 F2, 858e0167 F3), each red first.
- **F3b (5a1c8f0d) CHANGES RECALL CONFIDENCE SEMANTICS.** F3's `session` slot was a cue-matching slot: a same-boot neighbour
  matched it for free, so a cue quoting a stored episode scored 3/6 = 0.5 against a wrong window episode (the default
  `min_conf`) and `test_episode_durability.py::test_recall_reaches_past_working_memory` failed after F3 (measured: 96 passed,
  1 failed over seven recall, durability and guard test files). `LakeRecall.recall` now excludes `PROVENANCE_SLOTS` =
  {`session`} from the cue's verified fraction (`matched / len(content)`) and returns [] for a cue of provenance alone;
  the deep tier's exact sub-cue test still sees every slot. R2 is asked to refute this specifically (lead, 2026-10-06).
- **Dev smoke, seed 14000 (exposed dev seed; not the deciding run).** I1/I2/I6 pass; at `min_conf` 1.0 L = F = 0 in every
  class for the incumbent and scan-only, B5 identical, restore 64/64, B5w persisted = window on both boots (10, 20); at 0.5
  the window tier answers FALSE on C3 183/200, C4 175/200, C5 80/200, C6 56/200, C7 45/50 (reported, untouched).

- **R2's refutation pass on F1-F4/F3b (2026-10-06, read-only, no BREAK of F1-F4), folded.** (1) The deep tier answered a
  provenance-only cue at confidence 1.0 while `recall()` returned []: 1c7e0f07 makes `is_subcue` refuse an empty or
  provenance-only cue (session still must AGREE beside content). (2) STATED, not changed: "the session slot is not recall
  evidence" holds for `confidence` only -- `margin` / `retrieval` still encode the whole cue and calibrate on `len(cue)`,
  and the null's slot census includes `session`; margins before/after F3b were NOT measured and no claim is made that they
  are unchanged; the min_conf gate reads only confidence. (3) The durable copy is now `_durable`, so the working `_ep` is
  never stamped (no behaviour change). R2 did not reproduce the 96/1 test count.

### 2.19 Dated amendment 2026-10-06 (pre-data; step (0) of P3: the Q6 costing, derived index file vs in-RAM)

Additive; no line above is changed. Costing from the 1.4 (e) measurements and the tree at f580a6ff; no new run, no
14100-14107 or 14901-14903 data read. The numbers are the measured ones (50,000 episodes, 513 MB lake, warm cache) or
stated arithmetic on them; every projection is marked.
- **In-RAM prototype (measured).** One pass 0.29 s; 50,024 tokens; tracemalloc peak 14.6 MB = about 292 B per episode, inside
  B6's 1 KB per episode; query p50 / p95 0.32 / 0.62 ms. It passes B6 on its face at W_real.
- **Why in-RAM is not free (arithmetic).** It is rebuilt from the lake at EVERY boot, so boot cost is O(lake): the pass is
  0.29 s at 513 MB, PROJECTED about 1.9 s warm / 5.6 s cold at the live 3.32 GB (the same factor 2.15 -> 7.2 s of
  `make_brain` already shows). B6's `make_brain` delta (<= +1.0 s) is read at W_real, so it passes there and would fail
  at W_live. The boot restore already makes one full pass (0.27 s, `load_episodes(limit=64)`), so a shared pass only adds
  tokenisation, but that pass itself is the O(lake) cost the row exists to bound.
- **Derived file (lake_index.db) (design, UNMEASURED).** Boot cost is O(tail): fold the records after the watermark. Two
  facts from the tree constrain it: `LakeIndex` is PRIMARY KEY (tag, key) with one (container, seq) pointer per key
  (lake_index.py, `records` table), so a token's posting list cannot be one row; each posting is a row keyed
  `token|offset` and queries read a key range (the additive range read in 2.11 P3 (b)); and an episode carries about 5
  tokens, so rows ~ 5 x episodes (250,000 at 50,000 episodes). Query cost is posting-range reads plus `read_at`
  verification; no number exists yet. The 14901-14903 design measurement reads it BEFORE any deciding run.
- **R2's point (b), costed.** The lake holds learner events (teach / turn / sleep / percept / ctxread, `_T_EPISODE`
  payloads) beside lived episodes; the builder and the tail fold must DECODE a record to test `is_autobiographical_episode`,
  and learner records advance the watermark without a token row. 1.4 (e): `load_episodes()` decodes 50,000 episodes in 0.25 s
  (about 5 us each), so the decode is cheap; R2's sleep events add roughly 420 B per learner-affecting sleep tick, so the
  fold's cost is bytes, not rows.
- **Disposition.** The derived file stays the default (2.11 P3 (c)) because it is the only variant whose boot cost does
  not grow with the lake; the in-RAM posting array is built only if the design measurement shows the posting-range query
  misses B6's p95 <= 25 ms. This amends nothing in 2.4-2.5; it records which variant is built first and why.

### 2.20 Dated amendment 2026-10-06 (pre-data; the successor's build choices and the cost harness's unregistered details)

Additive; no line above is changed. Nothing on 14100-14107 or 14901-14903 has been run; the only successor data in hand
is a dev smoke on 14000 (W_c, exposed: B1 B2 B5 true, index build ~0.1 s) and a harness smoke at fixture scale
(`W_cost_smoke`, labelled not a measurement). Choices the frozen text left open, fixed here BEFORE any measurement:
- **B2 is read on the successor DEEP tier alone** (the `successor_deep` arm: working and window neutered, then the index),
  against the `scan_only` arm, per cue on C3-C6 by returned-episode identity. In the full chain (`successor` arm) a window
  answer legitimately differs from the scan's newest-first pick (C5 is ambiguous), so B1 and B5 read the full chain and B2
  reads the deep tier, matching 2.4's wording ("the successor deep tier returns EXACTLY the fixed scan's episode").
- **B4's eight faults** are tested at fixture scale on a NEW instance over the damaged state (each answer equals the lake's
  own scan answer, or `source="refused"`): index truncated, garbled, wrong-offset rows, one episode's rows deleted,
  watermark ahead of the lake, lake appended behind, lake rewritten by `BinLog.rewrite`, torn lake tail. Defences added
  to meet them: a row must agree with its own key (else refuse), the posting-row count is checked once per process (a
  mismatch rebuilds), progress ahead of the lake counts as a stale anchor (rebuild), the index file opens lazily so a
  damaged file refuses inside `recall`. Stated limit: compaction is detected on the LAST folded frame only.
- **The design measurement (14901-14903, W_real)**: cost cues are 40 per class over C3, C4, C7 and C6 (20 novel + 20
  None-trap), 160 cues; ONE child process; each cue is asked of the incumbent deep path (working and window neutered) and
  of the successor in alternating order (the pair order flips every cue and every round), 2 rounds; per-call traced peak
  over 40 sampled cues (successor only); the B7 spin witness before and after the child. **Standing RAM** = the process
  working-set (RSS) delta across building and re-opening the index (psutil), divided by the recallable episodes, a coarse
  upper bound that includes sqlite's page cache. **make_brain delta** is read as the warm re-open of the persisted index
  (a fold of nothing); the COLD build is reported beside it, never folded into the bar (it is paid once per lake, or once
  per compaction). The scale bar needs W_live and belongs to the deciding run. The harness writes the lake in bulk
  (`BinLog.extend`, one fsync per 5,000 records, byte-identical to `save_episode`'s framing) and deletes it per seed.
- A design measurement does not decide ADOPT; it reads the posting-range query's latency against B6 and decides whether
  the in-RAM variant is built (2.19).

### 2.21 Dated amendment 2026-10-07 (pre-data; the P3 panel's lens 1 folded)

Additive; no line above is changed, no bar in 2.4-2.5 moves, no seed is touched. Nothing on 14100-14107 or 14901-14903 has
been run; every reproduction below used fixture-scale lakes under a fresh `NORD_HOME` on D: (the live `brain_lake.bin` size
and mtime were identical before and after). Source: Codex gpt-6.1-sol, lens 1 (correctness + P8) on the frozen clone of
50602b34, board `codex_panels_1006\r1_p3\out_correctness_p8.txt`: 2 BLOCKER and 5 MAJOR, ALL reproduced by the lead's
instruction before any change (brain-free fixtures), none refuted. Lenses 2 (P2/P3) and 3 (safety/determinism) are not in.
- **M5 (MAJOR, `recall_audit.bars`) -- 0128f921.** B1 was read on the chain-level table, so a working- or window-tier FALSE
  failed the successor; 2.14 D-a bars exactly that. B1 now reads the `by_tier` cells and skips the preserved tiers.
  Implements D-a as written.
- **M4 (MAJOR, `_cost_run`) -- 20f004ea.** A VOID child result was scored (all four bars True). B7 as frozen: re-run the child
  once on the same lake; a second VOID raises `CostVoid` (no scoring, no reserve seed).
- **B1 (BLOCKER) -- eac9c86c.** A ghost episode framed inside a non-episode record's payload, with one posting redirected,
  was answered at confidence 1.0; a posting redirected to another genuine frame passed too. Posting keys are now
  `tok:offset:chk` (8 hex of the frame's payload) and recall refuses when the bytes read do not match; the row format is
  versioned (`ep_fmt`), an older index rebuilds. Stated limit: an actor who rewrites key, check and seq consistently is outside
  the accident model.
- **M3 (MAJOR) -- ff0930e0.** The once-per-process row check compared counts; a flipped key byte kept the count and hid a
  present record. The fold keeps `ep_digest` (sum mod 2**64 of a hash of every (key, seq)) beside `ep_rows`; the check streams
  the rows (additive `LakeIndex.iter_range`) and rebuilds on a mismatch. **Cost, named in advance:** one O(index rows) pass
  at the first sync of a process (about 250,000 rows at W_real); it lands in B6's make_brain delta (`index_warm_open_s`) and
  is read at the 14901-14903 design run. **Fallback and its trigger (not built):** if that delta exceeds +1.0 s, per-token
  row counts replace the global digest; the trigger is that measurement and nothing else.
- **M7 (MAJOR) -- d2e3bad9.** Deleting a record's rows through another connection after the first verification hid it until
  restart. `sync()` reads SQLite's `data_version` (additive `LakeIndex.data_version`); another connection's commit voids the
  cached verdict, our own commits do not. Stated limits: tampering through the module's own connection, and a file swapped
  under an open handle, are not modelled.
- **B2 (BLOCKER) -- c0b4fbfe. This SUPERSEDES the 2.20 sentence "Stated limit: compaction is detected on the LAST folded
  frame only".** [F20, A, F20, Z] rewritten as [A, Fn, Z] moves A and leaves Z (the last folded frame) in place, so the anchor
  passed and A's unique cue answered MISS. `BinLog.rewrite` replaces the file, so the lake's (device, inode) is kept with the
  anchor (`ep_lake_id`) and a different id rebuilds; `binlog.py` is not edited. Stated residual, with its red: an id reused
  by two rewrites between syncs AND a last frame left at the same offset passes (the test uses a constant-identity variant to
  show the anchor alone is blind to the first case); a filesystem whose id reads 0 falls back to the anchor alone. The
  alternative that closes the reuse residual (a generation counter bumped inside `BinLog.rewrite`) is a shared-file edit and
  is not taken.
- **M6 (MAJOR, S8's red is vacuous) -- follows (item 7).** The test passed with only `b6_bars` exposed. A real
  full-scan bypass arm (`succ_bypass`) joins the cost child; a deterministic unit red counts frames read per query (the
  index path reads few frames, the bypass reads the whole fixture lake); the MEASURED bypass red (p95 against the 25 ms and
  scan/10 bars) is read at the 14901-14903 design measurement, which waits for the machine.
- **Found on the way (pre-existing red, disclosed) -- f678fc1b.** `test_ratified_row_gates.py::test_the_retention_label_is_
  consistent_and_every_reclaimer_is_accounted_for` was RED at 50602b34 (verified on a `git archive` of that commit): the
  retention census flagged `recall_audit._cost_run` (names `brain_lake.bin`, called `shutil.rmtree` inline). Deletion now goes
  through `_drop_scratch`, which re-runs the shared guard and refuses the root itself, a sibling or an escape; the census is
  unchanged.
- **Review status of this fold:** unit-tested red first (each red is stated with its commit), NOT re-panelled by Codex; battery
  verification is the lead's. Run `tests/test_recall_audit.py` and `tests/test_episode_index.py` one per call: given together
  through `run_tests.ps1`, pytest resolved `snn_data_lake` from the main checkout's `src` (observed, cause not investigated).

### 2.22 Dated amendment 2026-10-07 (pre-data; the P3 panel's safety + determinism lens folded)

Additive; no line above is changed, no bar in 2.4-2.5 moves, no seed is touched. As in 2.21 nothing on 14100-14107 or
14901-14903 has run; reproductions used fixture-scale lakes and a DECOY live home planted through the guard's own seam
(`home._default_live_home`), with the real `brain_lake.bin` size and mtime identical before and after. Source: Codex
gpt-6.1-sol, lens 3 (safety + determinism) on 50602b34, board `codex_panels_1006\r1_p3\out_safety_determinism.txt`:
2 BLOCKER and 10 MAJOR. Reproduced on the decoy: S1, S2, S12; in memory: S8, S5; by cited code reading: S3, S4, S6, S7, S11.
- **S1 (BLOCKER) 32e56479.** `--out` and every child spec's `out` were opened unguarded; `--out <live>/brain_lake.bin`
  overwrote the decoy lake. `report_path` refuses an empty path, a non-.json name, a directory that is or lies inside the
  live home, an existing target in it, a symlink, a non-regular file and a HARDLINK (`st_nlink > 1`: it writes through to
  the lake); the CLI checks `--out` before the heavy run and runs the run and the write inside `_stamped`. Stated limit:
  a fresh report path whose directory is not in the live home is trusted (no race-free open is attempted).
- **S2 (BLOCKER) bc4ba184.** `--parent` was created before `throwaway_home` refused it. `make_guarded_dir` guards the nearest
  existing ancestor first. A parent that is an ANCESTOR of the live home is refused too (the shared guard's containment is
  symmetric); reports go to D:.
- **S12 af69fe3f.** `SEED_BLOCK` 14000-14999 enforced (`check_seed`, `seed_range`) in the CLI, `_cost_run`, `_run_seed` and the
  three seed-bearing children; `make_workload` stays free for unit fixtures. Enforces 2.8.
- **S4 f40c9cac.** The resolved cost home must be on `COST_DRIVE = "D:"` (a D: junction to C: is C:). Enforces 2.10. Tests
  run with the constant monkeypatched to None except the S4 test (disclosed in the commit body).
- **S3 9a1bb2de / S5 211dcb53.** The cost run owns a `mkdtemp` directory and removes it in `finally` (timeout and VOID halt
  included); `_drop_scratch` (introduced in f678fc1b, which refuses the root, a sibling or an escape) also drops the
  `home._LOG_CACHE` entries of the deleted lake.
- **S6 12d3283b.** The lake digest is hashed in 1 MiB chunks.
- **S7 b5b5f947.** Recall walks `LakeIndex.iter_range` (additive `reverse=`) lazily under `closing`; the posting list is
  never loaded whole.
- **S8 95160c87.** Reset voids `_checked` first and clears rows and markers in one transaction (additive `LakeIndex.clear`).
- **S9 5b205bbc.** A corruption verification exposes (`IndexCorrupt`) rebuilds the INDEX once and retries; a second one
  refuses visibly. The lake is never written by the heal (the red patches `append`, `extend` and `rewrite` to raise and
  compares the lake bytes and mtime). Earlier folds had already turned the self-consistent wrong pointer into a refusal or a
  rebuild (eac9c86c, ff0930e0); this makes the warm instance answer. LAYERING DISCLOSURE: with the frame check and the heal on,
  removing the compaction anchor alone no longer hides a record, so the B4 'ahead' defence-red now removes the visibility
  layer together with the anchor, and the B2 red asserts the identity check itself (`_anchor_ok`).
- **S11 815795a9. A stronger, additive qualification, acceptable only because nothing on 14901-14903 or 14100-14107 has
  run.** The cost child reports each arm's round-0 answer per cue; the parent scores them against the generator oracle. The
  SUCCESSOR must be VERIFIED on every present cue and MISS on every absent one, else `CostInvalid` and no B6 bar is
  computed; the incumbent's faults are reported as `scan_faults` (2.14 D-a: an incumbent finding, not a successor null).
- **S10 (partly refuted) 46576d0c + 20f004ea.** The retry-once-then-halt and 'never score a VOID' are done. A missing or
  non-OK `load` now counts as VOID. NOT adopted: a parent-side spin witness bracketing `_spawn`. B7 asks for a spin timed
  before and after each cost arm; the arms are ABAB-interleaved in ONE child process, so the child's before/after bracket
  IS that bracket, and a parent bracket would time child start-up rather than the measured loop.
- **Review status:** every item is unit-tested (red first, recorded in its commit) and NOT re-panelled by Codex; the lens 2
  (P2/P3) retry reads the head as of its freeze. Tests ran through the fixed `run_tests.ps1` (its `IMPORT_ROOT` line refuses
  a wrong import root).

### 2.23 Dated amendment 2026-10-08 (pre-data; the P3 panel's lens 2, P2/P3, folded)

Additive; no bar, seed or clause changes; nothing on 14100-14107 or 14901-14903 has run. Source: Codex gpt-6.1-sol lens 2 read
the code at ade9bdf3 (board `codex_panels_1006` r1_p3 `out_p2p3_retry2.txt`): 1 BLOCKER, 1 MAJOR, 2 MINOR, all reproduced or
confirmed by tool before the fix, none refuted. Each is unit-tested red first and NOT re-panelled.
- **BLOCKER 3e10530b (+ style d872e593).** `substrate.episode_index` was governed (REGIONS) and in no `run_selfchecks` tier, so
  the runner's registration assertion failed at import. Added to CORE; the pending-ADOPT dormancy exemption is unchanged.
- **MAJOR 388b8632.** The anchor now also requires `ep_watermark == anchor_offset + 4 + len(anchor_payload)`; progress that
  disagrees with its anchor rebuilds (an appended record could be hidden by a watermark corrupted to the new lake end).
- **MINOR f73e7406 (P2).** `LakeIndex.delete_tag`, `delete_meta`, `range_seqs` and `range_rows` (all R1 additions) are deleted;
  `clear` and `iter_range` are the one delete and the one range read.
- **MINOR bd7f8624.** The `recall_audit` dormancy exemption names `--r1-cost` -> `_cost_child` as the second Brain-touching
  path and is classified permanent (benchmark harness) rather than wire.
- **Measured elsewhere, disclosed.** `tests/test_selfchecks.py` on this tree: three failures not caused by these folds -- the
  rule census lags R2's `readout/interaction_learning.py` rule claims (artifact regeneration is the integrator's),
  `gates.policy_manifest` wants a battery-start certification, and `gates.envelope_hash_gate` fails on a pristine
  `git archive` of origin/main 9a080a46 too (R1 changes nothing under brainstem/). Every other CORE self-check
  passed, including `substrate.episode_index` and `instruments.recall_audit`.

### 2.24 Dated note 2026-10-08 (integration)

R1's integration is blocked on owner question #2 (X1) via R2's P1 commit db610a2b (which moved the interaction-learner checkpoint
to r2.v1); R1 depends on `home.LEARNER_EVENT_KINDS`, `home.is_autobiographical_episode`, `home.next_session` and `Brain._session`,
all introduced by db610a2b. Lead decision 2026-10-08: option (3), integration waits on the owner's X1 answer; no R1 code changes,
and the 14901-14903 design measurement (which needs no integration) runs on this branch. Six further R2 commits ride the same
dependency but R1 uses nothing of theirs (8f064a31, 58ceb314, 483309a1, d899f8d1, 5b03990c, 7460a5cf).

### 2.25 Dated note 2026-10-08 (the 14901-14903 design measurement, read once)

Additive; no line above is changed, no bar moves. Run by the lead on this branch at 0a1dc3f4: one invocation
`recall_audit --r1-cost --seeds 14901-14903 --size W_real`, `NORD_HOME` and the parent on D: (fresh), RC 0, 392 s, free RAM
never below 3.09 GB; the live `brain_lake.bin` size and mtime ticks (3318607701 | 639260867593316840) were identical before
and after. An earlier stopped run had reported 14901 only; this run reproduces it byte-exact (lake sha256 1e8e4c4a...,
513,226,097 B, 49,863 episodes) and adds 14902 / 14903.

| seed | succ p50 / p95 ms | scan p50 ms | warm open s | cold build s | index RSS B/episode | bypass p95 ms | answers differ | B7 |
|---|---|---|---|---|---|---|---|---|
| 14901 | 3.44 / 5.07 | 289 | 0.261 | 1.80 | 71 | 423 | 0 | OK |
| 14902 | 3.40 / 4.94 | 287 | 0.262 | 1.82 | 48 | 412 | 0 | OK |
| 14903 | 3.38 / 4.95 | 286 | 0.260 | 1.77 | 44 | 415 | 0 | OK |

Reading against 2.19 and B6 at W_real: the derived-file successor's p95 (<= 5.07 ms) is inside 25 ms and inside the incumbent
p50 / 10 (>= 28.6 ms) on all three seeds; latency, peak, RAM and boot bars read True; warm re-open (which carries 2.22 M3's
digest pass) <= 0.262 s, inside +1.0 s, so M3's fallback trigger does not fire; the S8 bypass arm fails B6 on every seed
(its red is real, 2.22 M6). Per 2.19's disposition the in-RAM posting array is NOT built; the derived file stays. The scale
bar needs W_live and stays with the deciding run (14100-14107). A design measurement decides no ADOPT.

## 3. Debate log

| # | partner | challenge / question | answer | resolution |
|---|---|---|---|---|
| D0 | R1 -> R2 | binlog.py + lake_index.py claimed; one offset reader; brain.py hunks; identity; seeds | R2 agreed R1 owns both files and edits neither | resolved |
| D1 | R2 | brain.py is R2's claim; R1's hunks (F1-F3, the index swap) | R2: disjoint from its symbols; claim handoff through main, whoever builds first holds it, release between commits | resolved (2.11 P2) |
| D2 | R2 | does R2 need offsets / its own frame parser? | R2: no parser, no offset cursor; a LOGICAL watermark (session, seq) over one streaming pass of the cached `home._log().replay()`; requires `replay()` to keep append order, torn-frame stop, no repair side effect; an offset start is a later optimization only | resolved: S2 carries R2's contract as reds; the reader stays R1's |
| D3 | R2 | compaction moves offsets; does R2 need a detector? | R2: no -- a logical key survives `consolidate`, which preserves episodes and their order (home self-check). R1 still needs S5 (offsets move); R1 tells R2 if consolidate or the S4 compactor ever drops or reorders `_T_EPISODE` | resolved: S5 is R1-only; B3's cross-compaction check is logical |
| D4 | R2 | episode identity (1.4 (d)) in R2's scope? | R2: not in its row (autobiographical records stay off the learner path); the fix is R1's; P2: ONE per-lake session counter for both writers, computed by R2 over learner events and episodes, one Brain attribute | conceded: F3 is R1's, using R2's counter. R1 rebutted "recall should ignore identity slots" (matching stays sub-cue; ignoring a slot the cue names would itself be a false match; the audit measures the session slot's effect on LakeRecall's per-arity null); R2 ACCEPTED the rebuttal | resolved |
| D5 | R1 -> R2 | R2 changes `load_episodes`' default exclusion to a learner-kind registry; one importable predicate? learner events as `_T_EPISODE`? | R2 ACCEPTED: `home.LEARNER_EVENT_KINDS` + `home.is_autobiographical_episode`, the one definition, used by `load_episodes`' default and imported by R2's reader; learner events are `_T_EPISODE` records with a `kind` slot; ORDERING: the predicate and exclusion land in the same commit as the first writer of a new kind (R2's P1) | resolved (2.2 recallable set; C8 now barred) |
| D6 | R2 | seeds | disjoint (14000-14999 vs 15000-15999) | resolved |
| D7 | G5 -> R1 | one live-home isolation guard for G5 (Codex #14 on G5) and R1's I7? a prefix check is aliasable by a junction | R1: one guard inside `home.py` (`assert_not_live_home`, called by `throwaway_home`; `require_isolated_home` for explicit NORD_HOME), built by R1 after R2's kinds, decoy-dir reds; G5 adopts it | resolved by lead ruling 16 (2.14 I7 shared guard) |
| D8 | R2 <-> R1 (2026-10-04, Codex 13) | one session counter: definition, slot name, call site | R2: `home.next_session()` over learner events AND episodes, slot `session`; R1: it reads the lake, so per-write calls split one boot into several sessions -> call once at construction into `Brain._session`, both writers read it | resolved (2.16); R2 froze the same text and reds in its 2.12 |

## 4. Results

None. Nothing has been recorded under this pre-registration.
