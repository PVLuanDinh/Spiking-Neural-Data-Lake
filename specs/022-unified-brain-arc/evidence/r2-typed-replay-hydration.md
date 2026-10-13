# R2 / T103 -- sleep replay hydrated from typed lake records through the incumbent interaction learner (DRAFT pre-registration)

Row: plan Part D "R workstream", R2 / T103 (PLAN-FINAL:1196-1226, with the DATED 2026-09-27 extended-sleep note); red
R-R2 (Part F, PLAN-FINAL:2227); FR-038, SC-016; P4b (the lake is the source of truth, a cache is not). Status: P0
pre-registration, folded after the lead read, the advisor and the Codex critique (Review record at the end); frozen at
the lead's commit that lands this file.
Every number in this file that is not marked MEASURED is a PROJECTION.

## 1. Territory, read at the base `e1b3ae3c` before the design

1. The learner and its inputs (the inventory the plan asks for first). `readout/interaction_learning.py::
   SpikingInteractionLearner` (:320) learns only through `interact` (:554), reached in production only from
   `learn_turn` (:621), which `engine/live_brain.py` calls from `chat` (:978, self-labelled: the firewall's reading of
   the reply), `teach` (:938) and `correct` (:952, authoritative). `Brain.respond` (brain.py:4421) calls `chat` on
   every turn, the daemon's transcript ingest included (`human=False`); `Brain.correct` (:4330) calls `correct`.
   A ring entry is `(text_features(utterance, 52), label, key)` with `key = " ".join(utterance.lower().split())`
   (:291). Since `text_features` splits `text.lower()` (:831-841), `text_features(key, 52) == text_features(utterance,
   52)` exactly: the stimulus is RE-DERIVABLE from the key under the declared encoder, so an event needs no vector.
2. Sleep consumers of the learner, each run by `Brain._sleep_tick` (:3254) on every sleep tick, each neuterable by
   `neuter_sleep` (:3602): (a) `replay(k=8)` (:587) draws uniformly WITH replacement from the ring by the learner's
   own `random.Random`; (b) the structural slow clock, `Brain._rewirer = StructuralRewiring(seed)` built on the first
   sleep tick of a PROCESS (:3322-3323), holding `_rng`, `_signs`, `_budget`, `_cycles` (sleep/structural_rewiring.py:
   171-178); (c) the Benna-Fusi rest, `relax` over the stack. G0(c) measured that the learner changes on every sleep
   tick and on no wake tick (g0c-learner-persists.md 1.3).
3. What G0(c) already carries (landed; g0c-learner-persists.md 2.6). The checkpoint `interaction_learner.json`
   (schema `g0c.v1`) is the learner's whole object tree INCLUDING the ring and the RNG, saved by `_enter_wake` and
   `close` (brain.py:3883, :5256, :5212-5222). Authoritative teaching is a `_T_EPISODE` event of kind `teach` with
   R2's fields (brain.py:5186-5210); `rebuild_authority` (:670) rebuilds the authority from them at construction
   (brain.py:349-362) and rewrites the ring. Open for R2 by that file's own boundary: replay hydrated from typed lake
   events beyond the in-memory ring; event identity for self-labelled turns; a watermark or cursor; the other learners'
   replay.
4. Self-labelled turns have NO lake identity today. Only `teach` events exist; a `chat` turn's ring entry lives only
   in the process and in the next checkpoint. A crash between two checkpoints loses every turn since the last one:
   its deposit, its ring entry and its existence.
5. The reader. `home.load_episodes` (home.py:1058) is a full streaming pass over `BinLog.replay` (binlog.py:258) and
   excludes kind `teach` from the autobiographical default (:1070). `BinLog` dedups on payload bytes (binlog.py:150),
   so two genuinely repeated events stay distinct only by their identity slots. `BinLog.rewrite` exists (:300):
   byte offsets are NOT stable identities, so a watermark is a logical `(session, seq)` key, never an offset.
6. MEASURED 2026-10-03 (scratch probe, out of tree: `scratchpad/r2probe/probe_sleep_restore.py`, dev seed 15000, a
   scratch `NORD_HOME`, worktree at `e1b3ae3c`): 42 turns (40 `respond` + 2 `correct`), tick to the end of a night,
   snapshot the home, continue as the twin; a fresh process on the snapshot restores and ticks to its next night. Over
   the next night's first 6 sleep ticks the restored learner's weights and RNG equal the twin's on every tick. The
   rewirer is rebuilt fresh in the restored process (`_cycles` 1..6 against the twin's 4..9) but its `_signs`, `_rng`
   and the weights stayed equal because the structural pass did nothing (turnover 0.0, 0 pruned, 0 regenerated on
   every tick; R3 reports the same on its seed 16000). So the rewirer is a real carrier that this schedule does not
   exercise: it diverges only when a live weight changes sign between nights (a prune for the twin, an adopted sign
   for the fresh rewirer) or a regeneration draws its `_rng`. Its red must be CONSTRUCTED (section 2.4, O_rw); my
   static prediction that the incumbent would red through the sleep tick was wrong on this schedule.
7. Cost, MEASURED on the same host: import 0.78 s; `make_brain(persist=True)` 2.1-2.6 s; 42 turns 3.25 s (77 ms per
   turn); a wake/sleep tick 51 ms; the scratch lake after the night 227,557 B (checkpoint included). A learner event
   packed by `pack_episode` is 266 B; `save_episode` on a scratch lake 0.57 ms median, 0.69 ms p90 (fsync per append).
   A synthetic 261,837,008 B lake (1,200 state-like frames of 200 KB + 60,000 learner events): a raw `BinLog.replay`
   pass 0.056 s (warm cache) and a typed-event filter pass 0.403 s, i.e. about 6.7 us per learner event unpacked.
   PROJECTION for the live lake (3.32e9 B, event count unknown, cold cache unmeasured): one pass of order seconds,
   dominated by bytes read cold and by the event count; R2 adds NO pass (section 2.1, one shared read).
8. Learners outside this row's mechanism, with their state today (inventory, dispositions in 2.9):
   `EpisodeConsolidator` is already refilled from the lake at construction (brain.py:3514-3520);
   `sleep/experience_replay.py::ReplayBuffer` (brain.py:1235, :2501, :3437; `(unit, action, reward)` into the arm's
   q-table) is process-local; `growth/neurogenesis_engine.py::NeurogenesisSelfImprove`'s `(percept_{t-1}, category_t)`
   buffer `_buf_x/_buf_y` (:243-244) is process-local and outside `growth_state_bytes` (:608), and no percept or
   transition record exists in the lake (home.py tags 0x01-0x0b).
9. The grower's inputs (part P3, assigned to R2 by main on 2026-10-03, board DEPENDENCIES.md). `Brain._wake_tick`
   hands every categorized wake percept to `NeurogenesisSelfImprove.observe(res["percept"], int(res["unit"]))`
   (brain.py:2478-2481; refusals counted in `_neuro_feed_refusals`). `observe` validates width and category
   (neurogenesis_engine.py:308-313), buffers the pair `(percept_{t-1}, category_t)` up to `buffer_cap=1000`
   (:315-319), scores the live instrument and, in production, moves the retained model's beta in place through
   `OnlineReadout.learn` (:337-338; the Brain builds the grower with `online_readout="delta"`, brain.py:1474). The
   sleep `tick` refits from the buffer every `refit_every` ticks (:348-353). `growth_state_bytes` (:608) saves the
   model at `_flush_growth` (the night's persist block and `close`), not the buffer, `_prev`, `_categories` or the
   live counters. The category is the arm `GrowingLearner`'s prototype id: append-only (growing_learner.py:84-95),
   flushed by `_flush_prototypes` at the same two boundaries (brain.py:5138-5170), so a prototype recruited after the
   last flush is lost on a crash and its id is reused by a different prototype after the restart. MEASURED: a
   percept event packed by `pack_episode` with the 64-dim percept as base64 float64 is 940 B and round-trips exactly;
   decoding one costs 17 us.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The mechanism (one mechanism survives: the ring becomes a cache over the lake)

- **Every learner input is a typed lake event, written BEFORE its deposit (write-ahead).** `learn_turn` calls one
  event sink for every turn, self-labelled or authoritative. The persisting `Brain` attaches it (it replaces the
  G0(c) teach sink, generalized; one sink, one counter). Slots, all strings, schema `r2.v1`:
  - `kind` (`teach` for authoritative, unchanged; `turn` for self-labelled), `event_id` (`<session>-<seq>`),
    `stream` (`chat`), `session` (ONE per-lake boot counter, `home.next_session()`: one past the highest `session`
    slot among ALL session-bearing `_T_EPISODE` records, i.e. learner events AND R1's autobiographical episodes; the
    same function serves both writers -- Codex 13, 2.12), `seq` (one per-session counter shared by every learner kind:
    `teach`, `turn`, `sleep`, `percept`);
  - `source` (the utterance key; the stimulus is re-derived), `target` (the label OBSERVED at the turn: the firewall's
    intent for `turn`, the person's intent for `teach`; never the authority-resolved label), `provenance`
    (`self` / `authoritative`), `encoder` (`text_features n_in=52`, the frozen segment), `grown_version` (G3's
    grown-segment version when G3's port is attached, else absent; see the next bullet), `schema` (`r2.v1`), `valid`
    (`1`), `split` (`train` unless an instrument or an evaluator writes `heldout`; G6's problem validation items are
    written `heldout`).
  Eligibility for redo AND replay keys on `encoder`. G3's grown segment (debate row 4) is a pure function of the
  frozen code and the admitted units at a recorded version, and G3 journals unit admissions as immortal `_T_GOAL`
  `unit` rows in the same ordered lake: REDO re-deposits each event under the grown version current at its original
  lake position by interleaving those rows (G3 supplies the pure row-apply; an event whose version cannot be
  reconstructed at its position is refused for redo and counted); REPLAY goes through the current units and records
  the version it used.
  A failed event write REFUSES the durable effect (Codex 2; G0(c)'s "the deposit goes on" rule is NOT kept): no
  deposit, no authority update, no ring entry for that turn. The reply is still given, and the refusal is counted
  (`unjournaled_refused`) and reported. Invariant: the weights, the authority and the ring are a pure function of
  the journaled events, so nothing after a failed write depends on experience that hydration cannot rebuild.
- **One reader, `substrate/learner_events.py`** (new): `read_learner_events(kinds, *, path=None,
  include_sessions=None, exclude_sessions=None, allow_instrument=False) -> (events, refusals)` (the complete shared
  signature, Codex 26: `path=None` resolves to the brain's own home lake; `allow_instrument=True` raises `ValueError`
  when the resolved path is the production home). One streaming pass in lake order. It returns
  only ELIGIBLE events and a refusal tally by reason:
  - not a learner kind (every autobiographical who/what/where/when/affect episode, never returned);
  - a required slot missing (refused, never padded);
  - an unknown `schema` or `encoder` (refused; the upcaster maps `g0c.v1` teach events, whose slot set is already
    complete, to `r2.v1`);
  - `valid != "1"` or `split == "heldout"` (refused);
  - a repeated `event_id` (the first wins; the repeat is refused: idempotence on identity, not on bytes);
  - a `(session, seq)` regression within the lake order (refused and counted; the lake order is authoritative);
  - provenance `instrument` unless the caller passes `allow_instrument=True`, which itself is refused (a visible
    `ValueError`) when the resolved path is the production home; the Brain's hydration never passes it.
  Session filters apply BEFORE anything downstream selects (R3 imports this reader and writes none of its own).
- **One definition of "autobiographical" in `home.py`:** `LEARNER_EVENT_KINDS = frozenset({"teach", "turn",
  "sleep", "percept"})` and `is_autobiographical_episode(episode) -> bool`; `load_episodes`' default uses it, R1's
  oracle, deep tier and index import it, and `learner_events.py` imports the registry from `home.py`. The consolidator
  refill, durable recall and `LakeRecall` see exactly the episodes they see today. ORDERING: the predicate lands in the
  same commit as the first writer of any new kind, never after (agreed with R1, debate row 7).
- **The checkpoint becomes schema `r2.v1`:** the learner tree WITHOUT the ring (declared never-saved, like
  `_authority`), plus the watermark, which is ONE cursor over all learner kinds (the `(session, seq)` of the last
  `teach`/`turn`/`sleep` event whose effect the saved state contains; Codex 4, 2.12) and the bout state, the per-event replay counts of the ring's events and the total, and the Brain's `_rewirer` state (section
  1 item 6). Saved at the same two boundaries, `_enter_wake` and `close`.
- **Hydration at construction (persisting brains only), ONE read shared with the authority rebuild** (no added lake
  pass): restore the checkpoint; read the learner events once; rebuild the authority from the eligible `teach` events
  at or before the watermark; set the ring to the last `_REPLAY_CAP` eligible events at or before the watermark, each
  entry `(text_features(source, 52), authority.get(source, target), source, event_id)`, with its saved replay count;
  then REDO every eligible event after the watermark, in lake order, through `learn_turn`'s own path with the sink
  suppressed (the same deposit, the same authority update and ring rewrite a live turn makes; no new event, no RNG
  draw, no tick). Redo restores wake learning that happened once; it is counted apart from replay (`redo_count`) and
  is not replay and not new evidence. The READER is idempotent by `event_id`; a second
  HYDRATION in one process is refused (2.11 D5).
- **Why this equals an uninterrupted twin.** In production the ring is exactly the last 64 `learn_turn` inputs with
  labels rewritten to the latest authority (`interact` :564-566, `learn_turn` :638-642, `rebuild_authority`
  :683-685), so it is a pure function of the event sequence; wake ticks do not touch the learner (G0(c) 1.3); deposits
  draw no RNG. Section 2.3 tests this claim rather than trusting it.
- **Replay is unchanged as a policy** (the plan's baseline uniform/recency policy: the 64-event recency window,
  uniform draws with replacement, k = 8 per sleep tick). Each draw increments its event's replay count and records the
  drawn `event_id`s for the tick. Replay writes no lake event, advances no wake chronology (`Brain._step`, `_nights`,
  the grower's `observed`/`live_n`) and has no motor path.
- **Legacy dispositions.** A `g0c.v1` checkpoint is upcast: weights, stack and RNG restored; its ring DROPPED and the
  count reported (its self-labelled entries have no lake identity, and inventing one would fabricate session and
  sequence); no watermark, so nothing is redone and post-checkpoint `teach` events only rebuild the authority, as under
  G0(c). Retention: learner events are `_T_EPISODE` records, never pruned (the authority needs every `teach`; R3's
  selection reads all eligible experience). Any other schema is refused visibly and the learner stays fresh.
- **Envelope.** No `brainstem/` file changes; no actuator route; nothing reads outside the brain's own home.

### 2.2 The instrument (`instruments/replay_hydration.py`, new; every home a fresh temporary `NORD_HOME`)

Per seed, every process builds `make_brain(seed=seed, persist=True)` on its own home. The schedule: a seeded shuffle
of `_CONV_TRAIN` through `Brain.respond(human=False)` (self-labelled), two `Brain.correct` lines, then ticks to the
end of a night (the declared boundary). The kill points:
- **K0 (close):** the schedule, then `close()`; a fresh process restores.
- **K1 (end-of-night crash):** the schedule, the tick that runs `_enter_wake`, then `os._exit(0)`.
- **K2 (mid-wake crash):** as K1, then `m = 6` post-boundary turns (4 `respond`, 1 `correct`, 1 re-`correct` of an
  utterance already in the ring), then `os._exit(0)`. Only redo can carry these.
- **K3 (mid-night crash; part P2, section 2.8):** as K2, then ticks into the next night and `os._exit(0)` after
  `j = 3` sleep ticks.

The twins are persisting processes on their own homes that run the same schedule and are never killed (persistence
adds writes, not learner dynamics; the restored process and its twin then share the session numbering). After the
kill point, with NO new user turns, every restoring process and its twin tick until `S = 6` sleep ticks of the next
night have run (K3: the rest of the night, then 6).

Recorded in every restoring process and its twin, read off the LIVE objects, never through the codec:
- before the first post-restart sleep tick: the learner tree by G0(c)'s independent walker (`_g0c_walk`, reused, a
  second implementation), the ring's `event_id`s and labels in order, the RNG state, the watermark, the replay and
  redo counts, the rewirer's `_signs`, `_budget`, `_cycles` and `_rng`;
- after each of the S sleep ticks: every neuron's weights, the drawn `event_id`s and labels, the RNG state, the replay
  counts, the rewirer state;
- around hydration and replay: `Brain._step`, `_nights`, the grower's `observed`/`live_n`, and the count of calls
  reaching `So101Arm.apply()` (the instrument wraps the live instance's bound method in its own process; no envelope
  file is read for edit or changed).

### 2.3 The verdict rules (written first; equality has no variance, so no statistic and no majority-of-seeds)

- **R2-c1, the restart equals the twin (R-R2's main clause).** In K0, K1 and K2, on EVERY confirmatory seed, the
  restoring process equals its twin on every record of section 2.2, with no tolerance.
- **R2-c2, the mechanism is load-bearing.** The incumbent arm I_K2 (G0(c)'s restore: hydration and redo neutered by
  the instrument) differs from its twin on at least one record on EVERY seed. If it equals its twin on a seed, K2 did
  not exercise redo there and that seed is INVALID (the construction is wrong, not the mechanism right). I_K1 equals
  its twin on every seed (the valid-restart control: the harness can say "equal").
- **R2-c3, every constructed red fires** (section 2.4): each fault arm differs from its twin, or trips its guard's
  refusal check, on at least one seed. A red that fires on no seed halts the arc (P8: it is deleted, never weakened).
- **R2-c4, the controls hold.** The replay-OFF twin (`neuter_sleep("replay")`) differs from the replay-ON twin in the
  post-night weights on EVERY seed (replay moves the observable, so equality is not vacuous); the comparator flags a
  planted 1-ulp weight change and one swapped ring entry (run first, on a dev seed).
- **R2-c5, the boundary (pytest, `tests/test_r2_replay_hydration.py`), each a separate check:** a `persist=False`
  brain writes no event and no checkpoint; hydration twice equals hydration once; `load_episodes()` returns exactly
  the autobiographical episodes with learner events present; a `g0c.v1` checkpoint upcasts as 2.1 states, with the
  dropped ring counted; an unknown schema leaves the learner fresh and says so; the reader returns each planted fault
  of 2.4 as a refusal with the right reason; hydration, redo and replay change none of the chronology counters and
  make zero `apply()` calls; an autobiographical episode is never returned by the reader.
- **R2-c6 (part P2), the interrupted bout.** In K3, on EVERY seed, the restoring process equals its twin from its
  first post-restart tick on every record of 2.2, the night's remaining sleep ticks included.
- **Disposition.** R2-c1 to R2-c5 all PASS -> P1 ships on. Any RED deletes P1's change in a stacked commit, G0(c)'s
  checkpoint path stays the incumbent, and the row records the null. R2-c6 is P2's own verdict and R2-c7 (2.10) is
  P3's, each with the same rule over its own part only (a P3 red reverts P3, never P1).
- **Scope.** These seeds, this schedule, kills at K0-K3, same-code restarts, the interaction learner and the sleep
  carriers that touch it. Arbitrary-crash recovery outside wake time is P2's claim only. No accuracy, retention or
  growth claim (that is R3, R-R3).

### 2.4 Checks and their constructed, reachable reds

Each red is a fault the instrument builds in the restoring process only, or plants in the restored home's lake before
the restore; production has no switch for any of them.
| red | construction | must fire on |
|---|---|---|
| O_wm (watermark dropped) | restore without the watermark: redo either every event or none | R2-c1 records (K2) |
| O_rng / O_w / O_stack | the G0(c) omission arms, kept for attribution | R2-c1 records |
| O_rw (rewirer omitted) | the twin's and the restoring process's live weight `w[0][0]` is driven through zero identically after the boundary (instrument, both processes), then the rewirer is left fresh | weights / rewirer after the structural pass |
| O_hyd (no hydration) | ring left empty, redo off | ring, draws, weights |
| DUP (duplicate queue hydration) | plant a byte-distinct record repeating a ring event's `event_id`; arm: the idempotence guard neutered | ring, draws |
| REORDER | plant an event whose `(session, seq)` regresses; arm: hydrate in shuffled order | ring order, draws |
| HELDOUT | plant an eligible-looking event with `split=heldout`; arm: the split guard neutered | ring contains a held-out id (explicit check) |
| MISSING | plant an event without `target`; arm: the reader pads it with the firewall label | refusal tally / ring |
| CHRONO / MOTOR | arm: hydration ticks the brain once | `_step` / `_nights` check |
Per-carrier attribution is reported (which records each omission moves, on how many seeds), with no pass condition,
as in G0(c) 2.3.

### 2.5 The null disposition (a causal claim; its instrument owes its positive control first)

If R2-c1 fails, the registered cause is "a continuation carrier is missing from the checkpoint or the hydration". The
instruments that carry that claim owe their positive controls BEFORE the confirmatory run: the walker and the
comparator (R2-c4's planted 1-ulp and swapped-entry checks) and the omission arms (each must move its records on a dev
seed). If a positive control fails, the disposition is HELD and the instrument is repaired first. A null deletes R2's
P1 implementation (the `turn` events' writer, the reader's use for hydration, redo, the `r2.v1` checkpoint) and keeps
G0(c) untouched. It deletes no owner-drawn part (P4d): the lake and the recall path on the engine drawing stay; only
this row's implementation is reverted, and the owner is told.

### 2.6 Seeds (R2's block 15000-15999; the grep for the whole block fired only on non-seed numbers: arXiv ids
2306.15364 and 2401.15453, a byte count 15625, a daemon PID 15928, a weight sum 84.15141 and byte counts 15168/15984/
15967 at PLAN-FINAL:927; positive control: the same grep finds 7000 and 5001 where SEEDS.md says they are)

- dev (exposed; development, positive controls): 15000-15007 (15000 used by section 1 item 6's probe);
- P1 confirmatory (read once, S = 8): 15010-15017; reserve 15020-15027;
- P2 confirmatory: 15030-15037; reserve 15040-15047;
- P3 confirmatory: 15050-15057; reserve 15060-15067. 15068-15999 unassigned.
The constructed reds run on the confirmatory seeds against the same twins (attribution needs the same seeds), as in
G0(c). Paired-seed rule: every arm of a seed shares that seed's schedule and twin.

### 2.7 Cost at real scale

MEASURED per unit in section 1 item 7. PROJECTION for the P1 campaign: per seed, 3 kill points x (persisting +
restoring) + 3 twins + I_K1 + I_K2 + 2 replay-OFF processes + 10 fault arms, about 24 processes of 10-20 s; 8 seeds in
4 workers, about 12-25 minutes. The production cost: one 266 B event and about 0.6 ms (fsync) per turn; the
checkpoint grows by the rewirer state (PROJECTION, a few KB) and loses the ring's 64 stimulus vectors; boot hydration
adds no lake pass and about 7 us per learner event to the pass G0(c) already makes, plus the redo of the turns since
the last boundary (77 ms per turn measured for a full `respond`; a redo deposit alone is cheaper, unmeasured).

### 2.8 Arc split and the files each part touches (claims filed on the board)

- **P0 (docs):** this file.
- **P1 (build + campaign):** `substrate/learner_events.py` (new); `readout/interaction_learning.py` (the sink for
  every turn, ring entries with `event_id`, replay counts and draws, hydration and redo, the `r2.v1` codec and the
  `g0c.v1` upcaster, its `GRADIENT_CLAIM` rows); `engine/brain.py` (HOT, minimal: the generalized sink, the one read at
  construction, the rewirer in the checkpoint); `substrate/home.py` (`load_episodes`' default exclusion only);
  `instruments/replay_hydration.py` (new); `tests/test_r2_replay_hydration.py` (new);
  `tests/test_g0c_learner_persists.py` (adapted to the ring-free checkpoint; claimed); a register row (FR-031).
  Write order inside P1: the `home.py` predicate and exclusion land in the same commit as the first `turn` writer.
- **File sequencing (main sequences the claims):** `readout/interaction_learning.py` R2 P1 then G3's port step;
  `growth/neurogenesis_engine.py` R2 P3, then R3 (`replay_rows`), then G7 (label arrays); `engine/brain.py` R2,
  then R1 (recall hunks), then R3, then G6 (P4 pressure input); `substrate/home.py` R2's new kinds with its first
  new-kind writer, R1's import after. RULED by main 2026-10-03 (board OWNERS.md "Hot-file landing order").
- **P2 (the resumable bout; the DATED 2026-09-27 clause):** a sleep-progress event (kind `sleep`) at the END of
  every learner-affecting sleep tick, slots: `event_id`, `stream` (`sleep`), `session`, `seq`, `night`, `tick`,
  `stage_index`, `stage_name`, `presentations_spent`, `drawn` (the replayed `event_id`s), `rng_digest`,
  `weights_digest`, `rewirer_digest`, `pending` (G6's pending structural candidates, G6's serialization), `schema`,
  `valid`. At restart: P1's redo rebuilds the start-of-night state from the end-of-previous-night checkpoint plus the
  day's events (no extra checkpoint), then the night's completed ticks are re-executed in order (replay draws,
  structural pass, relax), each VERIFIED against its event's digests, a mismatch refused visibly. Declared carriers
  for a staged bout (G6 declares its own in the format agreed in debate row 6): stage, budgets (replay k, ticks,
  presentations, structural candidates), participating populations, pressure baseline (G6's named state
  `sleep_pressure.json` at `_enter_wake`/`close`), pending candidates, RNG, cursor, learner traces. Today's night is
  one stage; G6's runner (`instruments/staged_sleep_bout.py`) maps onto the same event. Files: `engine/brain.py`
  (`_sleep_tick`, the redo at construction), `substrate/learner_events.py`, the instrument and tests above.
- **P3 (the grower's percept-transition record; assigned to R2 by main 2026-10-03; R3's grower family F7 waits on
  it):** section 2.10. Files: `growth/neurogenesis_engine.py` (claimed: a default-None `transition_sink` called in
  `observe` after validation and before buffering; a pure `hydrate_transitions(events)`), `engine/brain.py` (attach
  the sink and hydrate after `restore_growth_state`, :1494-1496), `substrate/learner_events.py` (kind `percept`), the
  instrument and tests above.

### 2.9 The other learners' replay (G0(c)'s open item), dispositions proposed

- `EpisodeConsolidator`: already hydrated from the lake (1 item 8); no R2 change.
- `ReplayBuffer` -> the arm's q-table: process-local `(unit, action, reward)`; not a learner input this row's events
  describe; carried to T093's census with its own fresh-process red, not built here.
- `NeurogenesisSelfImprove`'s transition buffer: ASSIGNED to R2 (main, 2026-10-03); part P3, section 2.10.

### 2.10 Part P3 -- the grower's percept-transition record (assigned by main; R3's F7 waits on it)

- **Schema.** `_T_EPISODE`, kind `percept`, `r2.v1`, all strings: `event_id`, `stream` (`wake`), `session`, `seq`
  (the same per-lake counter as 2.1), `x` (the percept, base64 of little-endian float64, exact), `category`,
  `n_protos` (the arm `GrowingLearner`'s `n_units` when written), `encoder` (`arm.percept gained width=64`; an
  instrument stream writes `synthetic.stream.k<k>` with provenance `instrument` and `n_protos` = its category count),
  `provenance` (`self`), `valid`, `split`. 940 B per event, MEASURED (1 item 9).
- **Writer (the grower's training step: where a percept enters its training buffer).** A default-None
  `transition_sink` on `NeurogenesisSelfImprove`, called in `observe` AFTER the width/category validation and BEFORE
  the buffering (write-ahead), so a refused percept never becomes an event. The persisting `Brain` attaches it next
  to `restore_growth_state` (brain.py:1494-1496). None -> byte-identical, so G3's `growth_fingerprint` identity check
  stays green.
- **Reader and hydration.** `read_learner_events(("percept",), ...)`. Training PAIRS `(x_{t-1}, category_t)` follow
  the ORIGINAL percept adjacency, not the filtered list (Codex 3). Every `percept` event carries `pseq`, a per-session
  counter of percepts that passed `observe`'s validation. It increments on every write ATTEMPT, so a failed write leaves
  a visible gap. A pair is admitted only when both endpoints are ELIGIBLE, share one session and have `pseq_t =
  pseq_{t-1} + 1`. A refused, held-out, missing or failed-write endpoint BREAKS the chain, and no bridging pair is ever
  formed across it. The pairing is never joined across a restart either (PLAN-FINAL:1206, "never join the end of one
  episode to another"). A failed `percept` write refuses that observation's buffering and resets `_prev` to None, so
  the live learner breaks the chain exactly as hydration does (Codex 2's invariant applied to P3). Extra eligibility: for `arm.percept` events, `category` must be below the
  restored arm learner's `n_units` (an unflushed recruit's id is reused after a crash; such events are refused and
  counted); the width must equal the engine's. At construction, after `restore_growth_state`, a pure
  `hydrate_transitions(events)` refills `_buf_x`/`_buf_y` with the last `buffer_cap` within-session pairs and
  `_categories` with every eligible category. It does NOT touch `_prev`, `_prev_cat`, `observed`, `live_n` or
  `live_correct` (chronology; the first live percept after a restart has no predecessor). Replay into the grower is
  R3's selection over these events; it writes no event.
- **The restart test (R2-c7), CPU device (CUDA refits are not claimed deterministic).** Per seed, a persisting brain
  runs wake/sleep ticks past at least one refit; kill K1g at the declared boundary (the tick after `_flush_growth`
  in the night's persist block, `os._exit(0)`). The restoring process and the uninterrupted twin then run ONLY the
  grower's sleep step (`neuro.tick()` until the next refit, no new percepts; the Brain's other carriers are T093's).
  Bar, on EVERY seed, no tolerance: the hydrated `_buf_x`/`_buf_y`/`_categories` equal the twin's; the model after
  the next refit (W, b, beta, units, trajectory row) equals the twin's.
- **K2g, mid-wake grower kill: REPORT-ONLY.** Production's `online_readout="delta"` moves beta on every wake
  transition. G5 (debate row 12): the incumbent delta rule draws no RNG and its beta update is a pure function of
  (beta, m, S, category, mu, sign), and `growth_state_bytes` carries `beta`, `m_out` and `_batch_beta`
  (neurogenesis_engine.py:633-639, :689), so a redo of the post-watermark transitions through `observe` could be exact
  for the incumbent; but the redo must seed `_prev` with the killed session's last pre-watermark percept and clear it
  after the redo (no cross-restart pair), and a trace challenger (if G5 ships one) must put its state in the blob.
  Reported, no bar; promotion to a bar is a dated amendment before any confirmatory run.
- **Reds (each must fire):** CROSS (the pairing joins across the session boundary: one pair more than the twin's);
  O_buf (no hydration: the refit differs); DUP, REORDER, HELDOUT, MISSING (as 2.4, on `percept` events); WIDTH (a
  planted 63-dim percept event, refused); VOCAB (a planted `category >= n_units`, refused); INSTRUMENT (a planted
  provenance `instrument` event in a persisting brain's lake, refused by hydration). Positive control: a refit on the
  twin's buffer shuffled across sessions differs from the twin's (the buffer content reaches the observable).
- **Seeds.** P3 confirmatory 15050-15057; reserve 15060-15067; dev 15000-15007.
- **Cost (PROJECTION):** one 940 B event (and ~0.6 ms fsync, MEASURED for a 266 B event) per categorized wake tick;
  the live daemon's 0x07 appends are about 160 KB per tick (G0(c) 1.2), so the added bytes are under 1% of them; the
  daemon's wake-tick rate is unmeasured while it is stopped. Hydration decodes at most `buffer_cap + 1` events per
  session tail (17 us each) inside the one shared pass.

### 2.11 Amendments from the lead read (2026-10-03), Codex-reviewed -- amend 2.1-2.10 and section 4

Source: R2's self-found defects D1-D5 (board `p0_reports\R2.md`), the lead read L1-L5 there, and board ruling 9.
Where a bullet below contradicts 2.1-2.10, the bullet governs. Section 2.12 governs over both.

- **D1 / L1, the incumbent arms run at the base head (amends 2.2, 2.3 R2-c2, 2.4).** Under `r2.v1` the checkpoint holds
  no ring, so "R2's code with hydration and redo neutered" is O_hyd (empty ring), not G0(c)'s restore, and I_K1 could
  never equal its twin. Ruled: I_K1, I_K2 AND THEIR OWN TWINS run at the base `e1b3ae3c` (G0(c)'s real `g0c.v1` code), in
  a separate read-only detached checkout the lead creates at build time. The instrument module does not exist at that
  head, so for these four arms per seed it runs as a FILE outside the package with `PYTHONPATH=<base checkout>\src`,
  the I arm and its twin both there. No copy of the `g0c.v1` codec is kept inside the instrument (P2: one codec after
  Q5's retirement). Each incumbent arm is compared ONLY to its own twin at the same head; no record is compared across
  heads. Disclosed: R2-c2 is a claim about the base head's mechanism on R2's schedule, and the R2-head arms (R2-c1,
  R2-c3, R2-c4) are a claim about R2's head. The schedule (2.2) uses only calls that exist at both heads
  (`make_brain`, `Brain.respond`, `Brain.correct`, `Brain.tick`, `close`, `os._exit`); the build checks this by
  running the base-head twin before any confirmatory seed.
- **D2, a stale-prototype guard (amends 2.10 Reader, Reds). Deviation from the lead read, flagged.** The lead read says
  `proto_digest`. On reading `growth/growing_learner.py::GrowingLearner.observe`, a prototype is REFINED in place
  (`(1 - lr) * w + lr * x`) on every familiar pattern. A digest of the vector at write time would therefore differ from
  the restored vector for every legitimately flushed prototype that was refined afterwards, and it would refuse good
  events. Proposed instead: a flushed-count identity. Two slots on every `percept` event: `proto_boot` (the arm
  learner's `n_units` restored at this process's boot, constant for the session) and `proto_flushed` (its `n_units` at
  the latest `_flush_prototypes` in this process; equal to `proto_boot` before the first flush). An event `(s, c)` is
  ELIGIBLE only if prototype `c` is PROVEN durable, i.e. `c < F`, where `F` is either the largest `proto_flushed` among
  later events of session `s`, or `proto_boot` of session `s + 1` when the hydrating process is reading session `s + 1`'s
  events (or is itself session `s + 1`, using its own restored count). Without that evidence the event is refused as
  `stale_proto` and counted, which is conservative: it loses at most the tail after the last flush of a crashed
  session. `proto_boot` of a session later than `s + 1` is never used, because an intermediate session may have
  recruited and flushed a new prototype under the reused id.
  Red STALE-PROTO, which must fire: session A recruits prototype `c`, writes percept events with category `c`, then
  crashes before the flush. Session B recruits a new prototype under the same id `c`, flushes and closes. Session C
  hydrates. The arm with the guard neutered puts A's `c`-events into the buffer (they differ from the twin built
  without A's tail); the guard refuses them. VOCAB (`c >= n_units`) stays as the cheaper first check.
- **D3 / L3, the 0x04 append budget.** `tools/lake_append_baseline.json` seeds 0x04 at a p95 of 113 B per tick. R2's
  writers add 266 B per learner turn and 940 B per categorized wake percept, both under 0x04, which is about 10x the
  seed per wake tick. Without a re-pin, `AppendBudget`'s FLOOD assert fires. That re-measure is the INTEGRATOR's
  (board ruling 7): R2's P1 commit message states the new per-tick 0x04 bytes, MEASURED under the baseline's own method
  (`make_brain(seed=0)` x120 ticks, scratch `NORD_HOME`, p95 over nonzero per-tick sums), for P1 and again for P3. The lead re-measures and
  re-pins at integration, with the constructed flood red still firing. R2 does not edit the baseline file.
- **D4, INVALID seeds (amends 2.3 R2-c2 and 2.6).** A confirmatory seed on which I_K2 equals its twin is INVALID. It is
  replaced ONCE by the next unused seed of that part's reserve, in order. More than 2 INVALID seeds in one part
  halts the arc, and the construction is repaired before any reserve beyond them is read. The same rule applies to
  P2's and P3's load-bearing arms.
- **D5, wording and one check (amends 2.2, 2.3 R2-c5, section 3).** (i) 2.2's "the restored process and its twin then
  share the session numbering" is wrong: the restoring process opens session N+1, while the twin stays in session N. It
  is harmless for R2-c1 because no new learner event is written after the kill point, so every compared `event_id`
  predates the kill. (ii) Hydration is construction-only. A second hydration call in one process is REFUSED
  (`RuntimeError`, counted), because a second redo would deposit the post-watermark events twice. This is added as an R2-c5
  check. (iii) The debate log's rows are reordered by number (16 had sat after 18).
- **Ruling 9 (degenerate contrasts).** R2 registers no paired-contrast statistic: every bar is exact equality or exact
  difference over all seeds. Adopted for consistency: any "must differ" bar (R2-c2, R2-c3, R2-c4, the P3 positive
  control) on which both sides are identical reads NOT MET, never vacuously met.
- **G6 D21, `applied_mask_digest` (amends 2.1's checkpoint and 2.8 P2; G6's A12 transactional mask; board rulings
  17/17a). NOT CODEX-REVIEWED: this bullet was added after the Codex batch-3 send.** R2 has no
  checkpoint per sleep tick (Q4), so the field lives in TWO places: (i) a slot `applied_mask_digest` on P2's `sleep`
  progress event, which is written at the END of each learner-affecting sleep tick and therefore AFTER that tick's mask
  apply (G6 step 3) and BEFORE G6's `sleep_mask_commit` (step 5). Its value is G6's digest of the last mask applied to the active
  network, and empty if none. (ii) The same field in the `r2.v1` checkpoint at `_enter_wake` and `close`. G6's restore
  rule reads the digest from the latest progress event of the interrupted night, or else from the checkpoint. Redo across a mask
  apply inside a night: P2 re-executes only the ticks that have a progress event. Each re-executed apply is taken from
  G6's `sleep_mask_prepared` row for `(sleep_id = "<session>-n<night>", tick)`, never re-decided. Its digest, computed
  by G6's one importable `mask_digest(prepared_row)` (R2 never reimplements it), is VERIFIED against the event's
  `applied_mask_digest`; a mismatch is refused visibly. D21 was accepted by the lead; G6's dated A12
  amendment follows campaign DONE. `mask_digest` ships in G6's P2 (`instruments/joint_mask_search.py`, fixtures only,
  no wait), so R2's P2 imports it from there, with no wait on G6's P4. This breaks the cycle in which G6's P4 lands after
  R2 on `engine/brain.py` (pending the lead's ruling on G6's proposal). A tick with no progress event is not re-executed. So a crash
  between apply and progress event leaves the restored network at its pre-mask state, which agrees with G6's rollback.
  A crash between the progress event and the commit restores the applied mask, and G6 completes the commit. Carrier
  (G6's format): {class: progress-event + checkpoint field, where: `sleep` event / `r2.v1` checkpoint, boundary: end
  of each learner-affecting sleep tick / `_enter_wake`, `close`, owner: R2 (field) / G6 (value), omission_red: with
  the field omitted, a crash between the progress event and `sleep_mask_commit` makes G6's C18 red (restore rolls
  back a mask that P2's redo re-applied)}. Verdict: G6's C18 owns it; R2-c6 includes the field in its equality records.
- **L4.** 2.10's K2g stays REPORT-ONLY; promotion to a bar happens only by a dated amendment before any confirmatory run.
- **L5 / section 4.** Q1, Q2 (reworded below), Q4, Q5 and Q6 go to the owner batch; Q3 is resolved; Q7 is added.

### 2.12 Codex critique folds (batch 3, gpt-6.1-sol, 2026-10-03) -- govern over 2.1-2.11

- **Codex 2 (BLOCKER), failed journal writes. FIXED.** 2.1 now refuses the durable effect of a turn whose event
  write failed: no deposit, no authority update, no ring entry, counted `unjournaled_refused`. Fault arm FAILWRITE
  (R2-c1 family, K2 schedule), with the failure injected identically in the restoring process and its twin:
  1. A `correct` whose event write fails.
  2. A later successful `respond` on the same key.
  3. The `_enter_wake` checkpoint.
  4. A restart.
  Bar: equal to the twin. Its red runs the same schedule under the OLD rule (deposit and authority update despite the
  failed write) in BOTH the pre-kill process and its twin. The red's checkpoint then holds the unjournaled
  correction's deposit and the later turn's authority-resolved label. After the restart, the rebuilt authority lacks
  the correction, so the ring labels and the following replay differ from the never-killed twin's. The red must fire
  (R2-c3).
- **Codex 3 (BLOCKER), manufactured adjacency. FIXED** in 2.10. Pairs follow the original `pseq` adjacency, and any
  refused endpoint breaks the chain. A percept written `split=heldout` is not buffered by the live learner either, and
  breaks its live chain, so live and hydrated pairing agree. Production writes only `train`; held-out percepts come
  only from instruments and G2/G6 evaluators. Added R2-c7 checks (pytest), one per refusal reason (HELDOUT, DUP,
  MISSING, WIDTH, `stale_proto`, a failed write): an endpoint interposed between two train percepts A and B of one
  session yields NO pair `(x_A, category_B)`. Their red is the arm that pairs the filtered list, which forms that pair.
  R2-c7's twin-equality bar stands for the confirmatory schedules (no refusals in them by construction). On a
  schedule with refusals, the hydrated buffer must equal the twin's pairs minus every pair with a refused endpoint,
  exactly.
- **Codex 4 (BLOCKER), checkpoint vs sleep-progress boundary. FIXED.** The checkpoint's watermark is ONE cursor over
  every learner kind that changes the saved state (`teach`, `turn`, `sleep`; all share the per-session `seq` and lie in
  lake order), together with the bout state (`night`, `tick`, `stage_index`, `phase` in {wake, sleep}).
  - **Recovery:** redo every eligible event after the cursor in lake order. A `teach`/`turn` goes through `learn_turn`;
    a `sleep` event re-executes its tick (replay draws, structural pass, relax, G6's mask from the prepared row), and the
    result is verified against the event's digests. Events at or before the cursor are never re-executed, so nothing a
    checkpoint already contains is applied twice.
  - **Commit ordering at `_enter_wake`:**
    1. The night's last `sleep` progress event is appended (fsync).
    2. The checkpoint is written with the cursor = that event.
    3. The checkpoint replaces the previous one only when complete. The build verifies that the store's write is
       replace-on-complete; if it is not, the build makes it so, with a torn-write pytest red.
  - **Crashes at each point:**
    - between steps 1 and 2: the old cursor is the previous `_enter_wake`, and recovery redoes the day's wake events,
      then night N's ticks, each verified, reaching the end-of-night state once;
    - after step 3: nothing is redone;
    - during step 2: the old checkpoint stands (as between 1 and 2).
  - **New kill point K4 (P2, R2-c6):** `os._exit` between the end-of-night progress-event append and the checkpoint
    write, reached by an instrument hook in the restoring process's run only. Bar: equal to the twin. Red O_cursor:
    recovery ignores the cursor's sleep component and re-executes from the last wake event, so night N is applied twice
    (or a digest refusal fires); the red must differ from the twin or refuse.
- **Codex 13 (MAJOR), one session counter. FIXED** in 2.1: `home.next_session()` = one past the highest `session`
  among ALL session-bearing `_T_EPISODE` records (learner events and R1's autobiographical episodes). It lands in
  `home.py` with R2's first writer, and R1 imports it (R1's F3/D4 definition, now the single one). New R2-c5 check: boot
  1 writes autobiographical episodes only (no chat, no percept); boot 2's session is strictly greater. Its red is the
  arm counting learner events only, which reuses boot 1's number. (Amended: see the DATED PRE-DATA AMENDMENT
  2026-10-04 at the end of this file.)
- **Codex 14 (MAJOR), grown MEMORY units. FIXED as a gate.** Until an ordered context-read journal (episode id, turn
  order; schema owner R2, read producer G3) AND its reconstruction test exist by a dated amendment, hydration refuses
  EXACT recovery when any admitted unit at an event's `grown_version` has G3's memory/state role. Redo stops at the first
  such event, and the stop is counted (`memory_unit_unreconstructible`) and reported, never silent. The ring is still
  rebuilt; no deposit is invented. Capacity units (stimulus-pure) keep the claim. The deciding-test dependency is
  registered: any R2 verdict that covers memory units waits on G3's state-branch pre-registration (board
  DEPENDENCIES.md; reported to the lead). New R2-c5 check: a planted unit row with the memory role makes hydration
  report the refusal.
- **Codex 26 (MINOR), reader signature. FIXED** in 2.1 with the complete shared signature, `allow_instrument=False`
  included, and its production-home refusal. Sent to R3.
- Not reviewed by Codex: 2.11's G6 D21 bullet (added after the send; lead rulings 17/17a). It is re-panelled with R2's
  P2 arc or said plainly to be battery-verified only.

## 3. Debate log

| # | from | challenge / question | answer | resolution |
|---|---|---|---|---|
| 1 | R3 (aba549e6c29954a0b) | which records does R2 read, which learner consumes them; does R2 journal grower transitions; what call does R3 import; do replays carry provenance and ids | interaction-learner events only (kinds teach/turn); grower journal not claimed, assignment asked of main; R3 imports `read_learner_events`; drawn `event_id`s exposed, replay writes nothing and advances no chronology | R3 adopted the reader and registered its grower arm as waiting (its reply); signature changes are to be messaged to R3 |
| 2 | R3 | the process-local `_rewirer` breaks a restored structural arm | measured (1 item 6): real carrier, not exercised on seed 15000 (turnover 0); O_rw is constructed; the rewirer joins the `r2.v1` checkpoint | folded into 2.1 and 2.4; number sent to R3 |
| 3 | G5 (a2a188231e63cd173) | does hydration call `OnlineReadout.learn` or read the grower's beta; any edit to online_readout.py, rule_composer census rows or the gradient-claim gate | no: interaction_learning.py has no reference to OnlineReadout/predict_next/_beta and those files none to it (grep); R2 edits only interaction_learning.py's own `GRADIENT_CLAIM` dict | no collision; G5 told when P3 was assigned (row 12) |
| 4 | G3 (aab6003013a8b76f6) | G3's default-OFF grown-segment port on the learner: does the ring store a width the port changes; does R2 edit `__init__`/`state_dict`/the G0(c) pin test | ring re-derived from the key, width unchanged; YES R2 edits all of them, so one upcaster chain, R2 P1 first, G3's attribute as the next schema step | ACCEPTED by G3 |
| 5 | G3 | REFUTATION: version-locking eligibility to the grown version starves old-task replay after the first growth | CONCEDED: `encoder` (frozen segment) gates eligibility; `grown_version` is a separate slot; redo interleaves G3's `_T_GOAL` unit rows at their lake positions via G3's pure `apply_unit_row`; replay goes through current units | folded into 2.1; G3 confirmed unit rows are written synchronously at admission (write-ahead) and admitted units' input weights freeze |
| 6 | G6 (a37039ad149d21ac3) | the hydrated path's call; who owns staged-bout persistence and in what format; is the pressure baseline R2's carrier; the event schema | construction-time hydration (no mid-life call); R2 owns persistence + the interrupted-bout check, G6 declares carriers as {class, where, boundary, owner, omission_red}; the baseline is G6's named state; schema given | ACCEPTED by G6 |
| 7 | G6 | refinements: draws != presentations ("rem" draws more than once); add `bout_rng_draws` and `pause_reason` slots | CONCEDED both; R2 corrected its own claim that P2 needs an `_enter_sleep` checkpoint (P1's redo rebuilds the start-of-night state) | folded into 2.8 P2; G6 renamed its `_T_GOAL` record `sleep_bout`, joined on (session, night) |
| 8 | R1 (a918e715474c451ac) | one offset-carrying reader in binlog.py; compaction detector; brain.py hunk delivery; event identity for autobiographical episodes (a second boot's episodes dedup into the first's: 10 records, not 20) | R2 has no offset cursor (logical watermark survives consolidate, which keeps `_T_EPISODE` order, home.py:1414-1417); R2 needs only `replay()`'s contract; brain.py by claim handoff; episode identity is R1's, using R2's ONE per-lake session counter | ACCEPTED; R1's S2 reds on `replay()`'s contract |
| 9 | R1 | rebuttal: recall must not ignore a session slot the cue names | CONCEDED | closed |
| 10 | R1 | D5: one importable "autobiographical" predicate; ordering constraint | ACCEPTED: `home.LEARNER_EVENT_KINDS`, `home.is_autobiographical_episode`; they land with the first new-kind writer | folded into 2.1; R1 bars an exact cue of a learner-kind record to MISS (its C8) |
| 11 | R3 | (a) instrument streams have no arm `n_protos`; (b) three writers on neurogenesis_engine.py; is `_prev` hydrated | (a) CONCEDED as a consumer flag `allow_instrument`, refused on the production home; (b) order R2, R3, G7 (G3 now touches no line of that file), main sequences the claim; `_prev` NOT hydrated | ACCEPTED by R3 |
| 12 | G5 | (reply to the P3 question) | the incumbent delta has no RNG and is pure in (beta, m, S, cat, mu, sign); a trace challenger carries an EMA and would go into `growth_state_bytes`; K2g report-only is right; G5's pilot shows the trace losing to delta | folded into 2.10 K2g |
| 13 | G7 (a80b14a934d639732) | overlap on neurogenesis_engine.py's restore | none: G7 adds per-unit label arrays only, never buffer-like state | order R2, R3, G7 agreed |
| 14 | G3 | addendum: G2's delayed-cue problems feed context through READ calls; grown MEMORY units (state branch) make a deposit depend on the episode's prior reads | open interface item: redo under memory units needs the context reads journaled (episode id, turn order); capacity units stay stimulus-pure | OPEN until G3's state branch is pre-registered |
| 15 | main | ruling: the grower's percept-transition record is assigned to R2 | added as part P3 (2.10) | done |
| 16 | main | rulings: hot-file landing order (2.8); tasks.md's "R1 -> R2 -> R3" is an ORDER, not a dependency, so R2 does not wait on R1; owner items queued in board OWNER_QUESTIONS.md; the advisor read is the orchestrator's after the wave | folded into 2.8 and section 4 | done |
| 17 | R3 | what is the default ring hydration after a restart (recency, uniform, per-session quota; seeded?) | recency: the last 64 eligible teach/turn events in lake order across sessions, labels rewritten to the latest authority; deterministic, no RNG; only replay's draws use the restored learner RNG | confirmed to R3 (its F6/C12 control reads as intended) |
| 18 | G6 | does `route` write a learner event; would journaled reads of G2's validation/future splits carry split=heldout | `LiveBrain.route` (live_brain.py:954-961) -> `respond` (:547-549) is pure inference and writes nothing; the sink is in `learn_turn` only; TRAP: `Brain.respond` (brain.py:4421) -> `chat` -> `learn_turn` WRITES a `turn` event. R2 commits: reads get no event; any future read kind is its own kind, never replay-eligible; any journaled validation/future presentation carries split=heldout | confirmed to G6 |
| 19 | G6 (D21, lead-approved) | the learner checkpoint records `applied_mask_digest` for G6's A12 transactional mask; does it fit redo across a mask apply inside a night | AMEND: there is no per-tick checkpoint (Q4), so the field rides P2's end-of-tick `sleep` progress event (after the apply, before the commit) plus the `r2.v1` checkpoint; redo re-applies masks from G6's verdict rows, verified by the digest, never re-decided; an unevented tick is not redone, so the restored state is pre-mask, matching G6's rollback | ACCEPTED by G6 and the lead (rulings 17/17a); in 2.11, not Codex-reviewed; `mask_digest` imported from G6's P2 |
| 20 | Codex batch 3 (gpt-6.1-sol) | findings 2, 3, 4 (BLOCKER), 13, 14 (MAJOR), 26 (MINOR) | all six fixed in text (2.12), none refuted | folded 2026-10-04 |

## 4. Open questions (for the owner, via main)

1. A live `g0c.v1` checkpoint's ring (up to 64 self-labelled entries with no lake identity) is dropped at the first
   R2 boot, counted and reported, rather than given a fabricated session and sequence. Accept?
2. Every chat turn, the daemon's transcript ingest included, becomes a 266 B immortal lake event, and every
   categorized wake percept a 940 B one (P3). Accept the growth (the daemon's rates are unmeasured while it is
   stopped; against its ~160 KB of 0x07 per tick both are under 1%)?
3. RESOLVED by main 2026-10-03 (debate row 16): tasks.md:266's "R1 -> R2/T103 -> R3/T104" is an order, not a
   dependency; R-R2 names no R1 result, so R2 does not wait on R1.
4. The mid-night resume (P2) redoes the night's sleep operations from the end-of-previous-night checkpoint plus the
   day's events instead of a checkpoint per sleep tick (each about 200 KB, which tripped the 0x07 flood detector in
   G0(c) 3). Accept the redo design?
5. P2 (one mechanism survives): R2 RETIRES G0(c)'s ring-in-checkpoint path (the ring becomes a cache over the lake).
   G0(c)'s recorded PASS stands as history; its campaign instrument's O_ring arm loses its meaning. Accept the
   retirement rather than keeping both?
6. Scope of the "other learners" (2.9): the arm's `ReplayBuffer` -> q-table stays process-local and goes to T093's
   census, not to R2. Accept?

Added from the lead read (2026-10-03), Codex-reviewed:
- Q2 reworded (L3): every learner turn, the daemon's transcript ingest included, adds a 266 B IMMORTAL `_T_EPISODE`
  event, and every categorized wake percept adds a 940 B one (P3). Retained immortal bytes per day = 266 x (learner
  turns per day) + 940 x (categorized wake percepts per day), never reclaimed. Both daily rates are UNMEASURED while
  the daemon is stopped; the first daemon day after integration measures them. The earlier "under 1% of 0x07" framing
  is withdrawn: 0x07 bytes are compactable and these are not. Accept the immortal growth?
- Q7 (interface, from debate row 14): a redo under G3's grown MEMORY units needs the episode's context reads journaled
  (episode id, turn order). It stays open until G3 pre-registers its state branch, and is GATED meanwhile (2.12 Codex
  14); capacity units are unaffected.

## Review record

- Lead read: 2026-10-03, board `p0_reports\R2.md` ("=== LEAD READ ===", L1-L5 and the L1 mechanics); rulings 9, 10, 13
  (the flushed-count identity replaces `proto_digest`) and 17/17a (G6 D21) in board DEPENDENCIES.md.
- Advisor read: the CLI lead session, Fable, 2026-10-03 (the L1 mechanics in the lead read).
- Codex critique: batch 3, model gpt-6.1-sol, reply file
  `C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\161935bb-36f0-4e8d-8a38-0411dcc64d0f\scratchpad\codex_p0\b3.reply.txt`,
  sha256 `1c8abc4a99bfa18c7e1d97d042bb1240c8b4cf8f893f88277bec5c38fae52e57`. R2's findings: 2, 3, 4 (BLOCKER), 13, 14
  (MAJOR), 26 (MINOR), all fixed (2.12). The fold itself is unreviewed text: it is re-panelled with R2's P1 arc, or said
  plainly to be battery-verified only. 2.11's G6 D21 bullet was not in the Codex send.

## DATED PRE-DATA AMENDMENT 2026-10-04 (with R1's D8; no data read)

Amends 2.12's Codex 13 bullet, which is frozen at `b6812e3f`. No confirmatory or reserve seed of R2's block has been run.
Not Codex-reviewed.
- `home.next_session()` is called ONCE per boot, at `Brain` construction, and the result is stored on one attribute,
  `Brain._session`. Every writer (R2's learner-event sink, R1's autobiographical episodes) reads that attribute and
  never calls `next_session()` again during the boot. Reason: the function reads the lake, so a second call after the
  boot's first write would return a higher number and split one boot across sessions (raised by R1, its D8).
- Added R2-c5 check: within one boot, a learner event, then an autobiographical episode, then a learner event all
  carry the same session, and boot 2's session is strictly greater. Red: the arm in which each writer calls
  `next_session()` per write; it splits the boot across sessions and must fire.

## DATED PRE-DATA AMENDMENT 2026-10-06 (with G3; no confirmatory or reserve data read)

Amends 2.1 and 2.12 (ruling 18, Codex 14). Not Codex-reviewed.
- A fifth learner kind, `ctxread`, joins `LEARNER_EVENT_KINDS` and `KIND_SPECS`: the context-read journal of G3's state-branch amendment (`g3-growth-loop.md`, claude/g3 98d92203, conditions accepted at 21ca1bc7). One row per stepped turn of a live memory unit, written BEFORE the step.
- Slots: the identity slots plus `episode_id`, `turn_index`, `grown_version` (ASCII decimals), `encoded_ranks_digest` (lowercase sha256 hex, the one token slot) and `encoder` (`text_features n_in=52`). The row inherits `split` and `valid` from the turn it journals; a heldout turn's row is heldout.
- It is written through `Brain._record_ctxread`, which shares the boot session and the one `_learner_seq` with teach/turn. G3 calls the sink and never writes the lake. The teach/turn reader never returns it; R1's recall excludes it through `LEARNER_EVENT_KINDS`.
- Reader refusals added: a non-decimal slot or a malformed digest is `bad_identity`; a missing slot, heldout, and invalid keep their reasons. A swapped `turn_index` or a dropped row is G3's `reconstruct_state` red, not a per-event refusal.
- Until G3's `reconstruct_state` test lands, memory-unit deposits stay excluded from redo eligibility and counted (ruling 18 stands).
- The extra immortal row per stepped turn joins the B13 claim-with-a-check: the measured B/turn is reported at the first daemon restart.

## DATED PRE-DATA AMENDMENT 2026-10-06 (P2 build decisions; lead-approved 2026-10-06; no confirmatory or reserve data read)

Amends 2.8 P2 and 2.11 G6 D21. Not Codex-reviewed. Dev seeds 15000-15007 only until the lead's written go.
- **One in-order redo.** Hydration's redo is ONE pass over the eligible `teach`, `turn` and `sleep` events after the
  checkpoint cursor, ordered by (session, seq), because a chat turn can land between two sleep ticks (`respond` and
  `tick` are separate public methods; no lock). A `sleep` event dispatches to the brain's `_sleep_learner_step` with the
  event sink suppressed: no clock tick, no new event, no extra RNG draw beyond the step's own.
- **The `sleep` event** is appended AFTER a learner-affecting sleep tick's replay, structural pass and relax, only when
  the brain persists and has a boot session. Slots: the identity slots, `sleep_id` (`<session>-n<night>`), `night`,
  `tick` (the boot's `_sleep_ticks` at that tick), `stage_index` 0, `stage_name` `night`, `presentations_spent`, `drawn`
  (the replayed event ids, comma-joined), `rng_digest`, `weights_digest` (sha256 of the learner's encoded state tree:
  weights, stack, replay counts, cursor), `rewirer_digest`, `pending` (empty until G6) and `applied_mask_digest`.
  It shares `Brain._session` and the one `_learner_seq`, and advances the learner's cursor so the next checkpoint
  covers the night.
- **Redo key** `(sleep_id, tick)`: a tick with an event is re-executed once, a tick without one is not. A duplicate is
  refused by the reader (`duplicate_id`), an out-of-order one by `seq_regression`.
- **Verification and atomicity.** Each re-executed tick's three digests must equal its event's. A mismatch stops the
  redo, restores the learner and the rewirer to their state before the first re-executed sleep event (the checkpoint
  plus the day's events), and is reported in `_ixn_restore` (`sleep_refused`); nothing is half-applied.
- **`applied_mask_digest`** is a slot now, the sentinel `none` while no mask producer exists (the reader treats an
  empty slot as `missing_slot`). Lead ruling: the check that compares it with `mask_digest(prepared_row)` and its
  `O_mask` red are NOT shipped here, because a check whose only live input is the sentinel would be dormant (P3).
  They land in the same arc as G6's P4 producer (`sleep_mask_prepared`), written by R2 in `learner_events.py` once G6's
  producer commit exists. Debt row: "R2+G6: applied_mask check lands with G6 P4" (the lead records it on the board).
- **Reds, constructed and red first:** `O_sleepskip`, `O_cursor` (K4), `O_digest` (atomic), `O_order`, `O_dup`,
  `O_interleave`, `O_nopreevent` (a kill after the replay and before the event leaves that tick un-redone).
- **Cost, a claim with a check (B13).** One more immortal row per learner-affecting sleep tick; the commit that lands the
  writer states the measured rows/night and B/night on a dev seed, and the lead re-pins `lake_append_baseline.json`.
- **c6 reads** K3 (kill mid-night after k >= 1 learner-affecting ticks) and K4 (kill between the end-of-night event
  and the checkpoint write): the restoring process equals its never-killed twin on every record of 2.2, no
  tolerance, every seed. A red reverts P2 only.

## INCIDENT ADDENDUM 2026-10-06 -- a test of R2's wrote the live lake (owner stop; no data of this arc read)

Not a pre-registration change. Hard rule 2 (never write `C:\Users\vinhl\.snn_data_lake`) was breached by R2's new,
UNCOMMITTED `tests/test_r2_sleep_redo.py`.
- **What ran.** The file's two tests K4 and O_nopreevent called `monkeypatch.undo()` mid-test. `undo()` reverts every
  monkeypatch of the test, including `tests/conftest.py`'s `_hermetic_home` `NORD_HOME` setenv, so the next
  `make_brain(persist=True)` (the `restored_brain()` helper) opened the DEFAULT home, the live lake. The file was run
  twice (the red run, then a `-x` green attempt) on 2026-10-06 in the 15:3x-15:4x window; the last write is the lake's
  mtime 2026-10-06 15:42:17.996. Each such Brain restored from the live lake's old checkpoint and its `close()` flushed
  a new one.
- **What was seen.** `brain_lake.bin` 3318607701 -> 3318650467 bytes (+42,766) and mtime 2026-09-27 01:19:19.331684 ->
  2026-10-06 15:42:17.996148. A read-only scan of the last 120 KB found one new `interaction_learner.json` record,
  schema `r2.v1`, `"_cursor": null` (a Brain's closing flush), and none of this arc's `sleep`, `ctxread` or digest
  records. Only the tail was inspected, so a second such flush is not excluded.
- **How it was noticed.** The K4 test failed on `restored._ixn_restore["checkpoint"] is None` because a checkpoint
  existed; the live lake has one, no test home had. R2 then compared the lake mtime and stopped.
- **Why it was not caught earlier.** The earlier R2 tests never call `undo()`; this file was the first, and nothing
  asserted that `NORD_HOME` was a private home before a persisting Brain was built.
- **State.** The daemon was stopped and the arm untouched. R2 did not repair, revert or read-write the lake after
  the stop; the lead verified the size and mtime and ruled an owner stop (no test that builds a Brain until the owner
  decides).
- **Fix, tests only, not yet run.** The two `undo()` calls are scoped (`with monkeypatch.context()`), and every Brain
  in the file is built through `_brain()`, which refuses unless `NORD_HOME` is a conftest tmp home and not the live
  one. `grep -rn monkeypatch.undo` over tests/test_r2_replay_hydration.py, tests/test_g0c_learner_persists.py and
  tests/test_r2_sleep_redo.py finds no other call.

## DATED ADDENDUM 2026-10-07 -- B13's cost claim, measured (panel m4; the writer commit 88002162 omitted it)

Additive; no pre-registered clause changes. Instrument: the lake append baseline's own method (BinLog.append wrap,
NORD_HOME scratch, p95 over nonzero per-tick byte sums), 120 ticks, live lake size and mtime unchanged.
- **Rows/night, B/night (dev seed 15000).** One 0x04 record per learner-affecting sleep tick, 610-613 B each. Full
  nights 1-4: 13, 13, 13, 14 rows; 7,950 / 7,969 / 7,969 / 8,582 B. Night 0 (the short first night): 3 rows, 1,830 B.
- **Append budget.** 0x04 per-tick p95 613 B, max 613, 72 nonzero of 120 ticks (seed 0 and seed 15000 identical); no
  tick above 8x the old 113 B seed (904 B). The lead re-pinned 	ools/lake_append_baseline.json 0x04 only, 113 -> 613
  B, every other tag carried; the old seed predates R2's typed learner events (P1 teach/turn and P2 sleep both write
  0x04).
- **Verified.** The baseline's consumers (test_append_budget_wire, test_growth_delta_chain, test_ratified_row_gates,
  test_synapse_delta, test_synapse_delta_writer) pass; python -m snn_data_lake.gates.immortal_retention OK.