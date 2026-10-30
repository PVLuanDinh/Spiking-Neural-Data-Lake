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

## DATED PRE-DATA AMENDMENT 2026-10-15 (R2 row-plan 1.0 against fbb1f38e; no confirmatory or reserve data read)

Amends 2.1, 2.2, 2.3 (R2-c3, R2-c4), 2.4, 2.8, 2.10, 2.11 (D5(ii)) and 2.12 by the clauses below; nothing above
this heading is edited.
Not Codex-reviewed. Rounds 2 and 3 fold reviews #2710 and #2711 (both CHALLENGE) and Codex #2719. Round 4 folds
rv-r2amend-b #2736 (CHALLENGE) and Codex #2740 (CHALLENGE). Round 5 folds rv-r2amend-b #2795 (CHALLENGE), the
integrator reply #2796 and integrator ruling v3 #2779, which supersedes ruling #2714 (retracted by #2737) and the
interim ruling #2738, before this lands. Round 6 folds R2INST round 2 #2873 (claim=fail), Codex #2885 and the
integrator retraction #2886 into A3 only (pre-data; dev seeds 15000-15007 seen, nothing else). Round 7 folds
rv-r2amend-a #2902 and the Codex review of #2898 (relayed #2903) into A3's RED-PARTIAL and A2's R2-c3 row (same
disclosure: dev seeds only).
Basis: board #2689 (integrator disposition), reviewed by #2690 and #2691 (both AGREE), evidence
`rows/r2-mechanism-diff-2026-10-15/REPORT.md`. Every symbol below was re-checked by `git.exe grep`/`show` on fbb1f38e
(the main head this amendment is written against) by the amendment lane, not copied from REPORT.md. Symbols are cited
by name, never by line. The only text hit for the base hash `e1b3ae3c` in src, tests and tools at fbb1f38e is
`tests/test_candidate_lifecycle.py`, a comment about `ng_snn.py` (2026-10-04); it is unrelated to R2 and is not
evidence for any clause here.

### A1. Precondition

No confirmatory or reserve seed of R2's block (15010-15067) has been read. Checked 2026-10-15 by a textual scan of
`rows/` and `specs/` for those numbers: the only hits are the registered allocations (2.6, 2.10), copies of them, and
float or byte substrings (`"bytes": 15022`); reviews #2690 and #2691 repeated the scan and agree. Reason it holds: the
campaign instrument `instruments/replay_hydration.py` never existed on the main lineage (absent at fbb1f38e; 753 lines
only at claude/r2 56031ab9). Codex #2719 repeated it over 4,252 text artifacts of `rows/` and both pinned trees:
allocations, declarations, hashes and unrelated floats only.
**The licence is QUALIFIED.** It rests on a TEXT audit of `rows/`, `specs/` and both trees. Binary artifacts and other
execution locations were not audited, and the old instrument on claude/r2 accepted arbitrary `--seed`/`--campaign`
inputs, so main-lineage absence alone does not prove non-exposure. A dated PRE-DATA amendment is licensed on that
qualified basis, not as categorical proof. Only dev seeds 15000-15007 are exposed until A5.

### A2. The mechanism as landed, per changed clause

- **2.1 writer.** `Brain._append_learner_event` is the one writer of every kind in `learner_events.KIND_SPECS`
  (`posed_problem`, `teach`, `turn`, `ctxread`, `sleep`, `sleep_bout`); it refuses `ctxread` ("R2 effect/producer is
  not bound") and consumes `Brain._r2_seq` before the append. Wake turns reach it through
  `Brain._record_learner_event`, attached as `SpikingInteractionLearner.event_sink`. Write-ahead, one boot session
  (`Brain._session`) and one sequence are kept; the registered "sink in `learn_turn`" is this callback.
- **2.1 failed write.** CHANGED by owner #1037: section A3 supersedes it. The counter `unjournaled_refused` now
  exists as `SpikingInteractionLearner._unjournaled_refused`, incremented by `Brain._record_learner_event`.
- **2.1 reader and receipt.** `learner_events.read_learner_events` is kept; boot uses
  `learner_events.scan_learner_events(kinds, strict=True, return_receipt=True)` (all kinds, no session filter, no
  instrument events), which returns the eligible events and one `receipt`. `SpikingInteractionLearner.hydrate(events,
  *, receipt, sleep_step, checkpoint_prefix_digest, effect_actor)` refuses ("R2 incomplete or mismatched same-pass
  receipt") unless the receipt is complete and its `eligible_ids` equal the events' ids. The one shared read stands;
  the receipt is a new contract.
- **2.1 session.** `home.next_session` is called once per boot (through `scan_learner_events`) and stored on
  `Brain._session`: the 10-04 amendment is unchanged.
- **2.1 kinds.** `home.LEARNER_EVENT_KINDS` = `teach`, `turn`, `sleep`, `sleep_bout`, `percept`, `ctxread`,
  `posed_problem`; `home.is_autobiographical_episode` is kept. `percept` is a registry row only: it has no
  `KIND_SPECS` schema and no writer (see A6). `posed_problem` is G9's kind; R2 registers no check for it here.
- **2.1 checkpoint.** `r2.v1` carries exactly `schema`, `tree`, `rewirer`, `sleep_observations`, `prefix_digest`.
  `prefix_digest` binds the checkpoint to the canonical event prefix its cursor covers
  (`learner_events.prefix_digest`, advanced per append by `advance_prefix_digest`). 2.1 names one refused class
  (an unknown schema). Main refuses three: (1) an unknown schema or key set; (2) a cursor not in the eligible
  prefix ("R2 checkpoint cursor is not in the eligible prefix"); (3) NEW, a cursor with no `prefix_digest` binding
  ("R2 checkpoint canonical prefix binding missing") or with a digest that differs from `prefix_digest(events,
  cursor)` ("R2 checkpoint canonical prefix digest mismatch"). Each is refused before any cache or cursor install.
- **2.1 legacy `g0c.v1` upcast. CORRECTS REPORT.md ("apparently unchanged").** `load_state_dict` and
  `learner_events._upcast` still upcast, but `Brain._r2_boot` refuses a checkpoint with no `prefix_digest` before
  `hydrate` runs, so a `g0c.v1` checkpoint is REFUSED and latched at boot, never upcast with its ring dropped.
  Section 4's Q1 (drop the ring?) is therefore answered by refusal; it is an owner OPENING, not a check.
- **2.1 hydration and redo.** `SpikingInteractionLearner.hydrate` redoes, in lake order after the cursor: `teach` and
  `turn` through `_apply_turn(..., cache_only=<at or before cursor>)`; `sleep` through `sleep_step["step"]`, which is
  `Brain._r2_sleep_step`; `sleep_bout` through `instruments.staged_sleep_bout.learner_bout`; `posed_problem` through
  `effect_actor`. Each redone `sleep` or `sleep_bout` is verified against its `SLEEP_DIGESTS` and `SLEEP_PROGRESS`
  slots and refused on mismatch. The Brain installs the staged learner only if `Brain._r2_may_rearm(report)` (complete
  and `processed_ids == eligible_ids`), else the latch stays. D5(ii) CHANGED in type only: a second hydration raises
  `ValueError("R2 hydration is once per staged learner")`, not the registered `RuntimeError`.
- **2.8 P2 dispatch and report.** `_sleep_learner_step`, `_ixn_restore` and `sleep_refused` are absent. In place:
  `Brain._r2_sleep_step(ixn, rewirer)` (structural rewiring only; returns `(rewirer, {"_structural_report"})`),
  `Brain._r2_sleep_learning` (writes the `sleep` event), `Brain._report_ixn_refusal(what, exc)` (stderr, ASCII, no
  stored attribute) and `Brain._r2_fault` (`{"phase", "cause"}`). Failure is latch-then-rollback: the `except` block of
  `Brain._record_learner_event`, `Brain._r2_run_bout` and `Brain._r2_sleep_learning` calls `Brain._r2_close` FIRST,
  then `Brain._r2_restore` returns learner, rewirer, observations and posed actors to their `Brain._r2_snapshot`
  pre-effect state; a restore that raises replaces the fault with phase "rollback". Any check that read
  `_ixn_restore["sleep_refused"]` must read `_r2_fault` and the
  stderr receipt instead. The `sleep` event slots match 2.8 (`stage_index` "0", `stage_name` "night", digests via
  `learner_events.sleep_digests`); `pending` and `applied_mask_digest` are "none" and `hydrate` refuses anything
  else ("R2 deciding mask carrier is not bound").
- **2.3 R2-c4 replay-OFF control (owner #1039 / R3 retirement; fact reported by lane r2inst, #2702, re-verified).**
  `Brain.neuter_sleep` raises "retired sleep capability" for `replay`, `benna_fusi` and `attractor`, so the registered
  replay-OFF twin cannot be built on main. The `sleep` event's `drawn` slot is written by `Brain._r2_sleep_learning`
  as `",".join(ixn._last_drawn) or "none"`: it is "none" unless `SpikingInteractionLearner.replay` was explicitly
  invoked (public, not retired; it sets `_last_drawn`, and nothing resets it before a sleep tick), and `hydrate`
  compares a redone `sleep` event against that same live `_last_drawn`. `presentations_spent` is "0". The
  `sleep_bout` event's `drawn` IS live: `instruments.staged_sleep_bout.bout_progress` writes
  `json.dumps(report.carriers)`. R2-c4's must-differ control becomes, on 2.4's O_rw driven schedule:
  (i) precondition: `ixn.stack` is present (with `stack` None, `Brain._r2_sleep_step` returns `rewirer` and
  `{"_structural_report": None}` unchanged, so the twins are equal by construction);
  (ii) the post-night `ixn.neurons[i].w` rows of the structural-ON twin and of the `neuter_sleep("structural")` twin
  (which sets `_sleep_caps["structural"]` False) DIFFER in at least one row on EVERY seed; a rewirer object that
  exists but moved no weight does not satisfy it, and None-versus-object is not a difference;
  (iii) `rewirer.report()` turnover is reported separately, no bar. A seed where the precondition fails, or the rows
  are equal, reads NOT MET (ruling 9), never skipped. The planted 1-ulp and swapped-ring-entry comparator checks are
  unchanged. The I_K1/I_K2 arms run at base `e1b3ae3c` (D1), where replay still exists, so only they keep a replay
  `drawn`.
- **2.4 red O_rng (RNG carrier omitted at restore) and 2.3 R2-c3. Integrator ruling v3 #2779, which
  supersedes #2714 and the interim #2738; found by rv-r2amend-b #2736.** History: lane r2inst found on dev seed 15000
  (#2713) that O_rng did not fire in an instrument that never calls replay, and ruling #2714 then made it a
  conditional P8 deletion.
  #2714 was RETRACTED (#2737) because it was wrong: `SpikingInteractionLearner.replay(k)` is public and not retired
  ("explicit invocation remains"), it is that learner's only `self.rng` draw after `Random(self.seed)` in
  `__init__`, and main's G0C role runner (`G0C_ROLES` carries `O_rng`) calls `learner.replay(k=8)` and records
  `draws` and `weights_after`. "Fires on no seed" was a property of an instrument that never calls replay, not of
  the mechanism, so the deletion would have weakened a reachable red. Interim ruling #2738 (superseded by #2779
  below), verbatim: "O_rng stays a LIVE red for
  P1/P2 on main. The R2 instrument's O_rw driven schedule gains ONE explicit SpikingInteractionLearner.replay(k)
  step after restore, run identically in the restoring process and its twin, recording RNG state and drawn ids per
  2.2; O_rng (RNG carrier omitted at restore) must then fire on every dev seed 15000-15007, and per R2-c3 a red that
  fires on no seed halts the arc (no deletion, no weakening)."
  #2779 (v3), verbatim: "O_rng stays a LIVE red for P1/P2 on main. Construction (as amendment A2 (ii) and the
  instrument #2749 already implement): the persisting process AND its twin each call
  SpikingInteractionLearner.replay(k) BEFORE the boundary tick (so the checkpointed RNG has advanced) and again AFTER
  restore, inside the learner effect boundary; the O_rng arm omits only the RNG carrier at restore. It must FIRE
  (RNG state and drawn ids differ from the twin, no reconciliation fault) on EVERY dev seed 15000-15007, and the
  instrument's verdict must require all 8 (a 1-of-8 fire does not pass); per R2-c3 a red firing on no seed halts
  the arc. Explicit replay writes no lake event: this red tests the restored RNG carrier, not a durable effect."
  Consequences: O_rng is neither deleted nor weakened; it counts in R2-c3's "every constructed red fires"; if it
  fires on no seed the arc halts (P8). O_w and O_stack are unchanged. At base `e1b3ae3c` it stays a live red on the
  I_K1/I_K2 arms.
  Construction facts checked on fbb1f38e, which the ruling does not state:
  (i) `SpikingInteractionLearner.replay` calls `_require_effect_boundary`, which raises "persistent learner effect
  requires the common R2 boundary" when `event_sink` is bound and `_effect_depth` is 0. On a persisting Brain the
  step runs inside `ixn._effect_depth += 1 ... -= 1`, as `Brain._r2_sleep_learning` does. Main's role runner calls
  `learner.replay(k=8)` bare; the instrument lane confirms, per role, whether that learner has an `event_sink`.
  (ii) The after-restore step alone cannot make the red fire. Nothing else draws the RNG, so a checkpoint written
  before any replay holds a never-drawn `Random(seed)` state, and an omitted restore leaves the fresh `Random(seed)`
  that `Brain._r2_fresh_learner` builds: equal states, identical draws in the omitted-restore process, its twin and
  the correct restore. `rng`, `_last_drawn` and `_replay_counts` ride in the `r2.v1` tree (`_state_attrs` does not
  exclude them). So the O_rng arm family also runs the same explicit `replay(k)` BEFORE the boundary tick that
  writes the checkpoint, identically in the persisting process and its twin, then K1, then the ruled after-restore
  step in both. The pre-boundary step is not part of the registered K0-K4 equality schedule, whose records are
  unchanged. Its effects are not journaled and live only in that checkpoint: K0-K3 kill after it is written, K4
  does not and is not an O_rng record. The pre-boundary step first appeared as this lane's verified correction of
  #2738's wording and is now part of ruling #2779 itself; if the red fires on no dev seed, R2-c3 halts the arc and
  no deletion is licensed.
  (iii) Disclosure: explicit replay writes no lake event and advances no wake chronology (its docstring), so the
  red tests the restored RNG carrier, not a durable effect.
- **2.3 R2-c3 TIGHTENED (pre-data, 2026-10-15; from rv-r2amend-b #2795, accepted in integrator reply #2796; this
  clause is in A2's R2-c3 row, not a new subsection).** Verbatim from #2796: "R2-c3 TIGHTENED (pre-data,
  2026-10-15): every 2.4 red must fire on EVERY seed of the run (dev 15000-15007 first, then each
  confirmatory/reserve set); a seed whose arm record is incomplete (crashed/missing) is INVALID, and any INVALID
  seed makes R2-c3 NOT MET for that run (never PASS, never skipped); the registered weaker bar (fires on no seed
  halts) is superseded, not deleted -- a red firing on no seed still halts the arc." The gate is therefore
  all-seeds, not any-seed: a red that fires on 1 of 8 seeds is NOT MET (#2795: the instrument's `verdict()` used
  `any_` over valid seeds only, so one firing seed gave PASS and an INVALID seed was excluded from the count; that
  is not the registered bar and is wrong under this tightening). INVALID seeds count against the run. This
  applies to every 2.4 red, O_rng included (#2779 states the same all-8 requirement for it).
  Round 7 (2026-10-15, pre-data; Codex on #2898, relayed #2903; dev seeds 15000-15007 only seen): an INVALID seed
  from ANY cause makes R2-c3 NOT MET, whether its arm record is incomplete (crashed/missing) OR the arm is
  complete but INVALID by its incumbent control (I_K2 equals its twin, D4). The instrument's `verdict()` must not
  count only valid-plus-incomplete seeds (a synthetic run with one control-INVALID seed returned PASS). D4's
  once-only replacement of an INVALID confirmatory seed does NOT apply to R2-c3: replacing the seed would drop the
  INVALID one from the count, which is the "never skipped" case. R2-c3 stays NOT MET for the run.

### A3. Owner #1037: the latch, and the PROSPECTIVE FAILWRITE schedule it requires

Board #1037 (`owner-answer`, relayed by codex-sndl-coord-1010, 2026-10-11), by speaker, each verbatim:
- The owner's own words: "Adopt these three decisions".
- The choice put to the owner, which that answer adopts: "R2 stops all learning writes after a write failure until
  restart, with its old failed-write test updated in a dated amendment".
- The relayer's application (codex-sndl-coord-1010), NOT the owner's words: "Applied to original question#1: R2: stop
  all common-writer learning-event kinds after a write failure until restart; update the old FAILWRITE schedule/test
  in a dated prospective amendment before new checks."
- The coordinator's record (`rows/owner-dispositions-adopted-2026-10-11.md`), NOT the owner's words, R2 #1: "all kinds
  on the common R2 learner-event writer latch after an uncertain failed write until successful restart
  reconciliation; write the dated prospective amendment of the old FAILWRITE schedule before implementing or running
  its new checks."
The owner's authority is the adopted choice; the "before new checks" ordering is the relayer's and coordinator's
application of it, and this amendment treats it as binding.

**ORDERING BREACH, disclosed.** #1037 required this amendment BEFORE the latch was implemented or its checks run.
That did not happen. Main already implements the latch (`Brain._append_learner_event`,
`Brain._record_learner_event`, `Brain._r2_boot`, `Brain._r2_close`) and carries tests of it:
`tests/test_g0c_learner_persists.py` (comments "owner1037-1039/FAILWRITE"),
`tests/test_r2_replay_hydration.py::test_failed_b_latches_and_os_restart_uses_actual_durable_prefix`,
`tests/test_r2_sleep_redo.py::test_sleep_append_failure_restores_full_state_and_keeps_latch` and
`::test_checkpoint_failure_preserves_actual_event_prefix_and_reconciles_complete_b`. This amendment is pre-DATA (no
campaign result of the new arm exists; no confirmatory or reserve seed was read) but POST-implementation. It does not
claim to precede the implementation. The docstring of `tests/test_r2_replay_hydration.py` says UNRUN; this lane
ran nothing and claims no result of those tests. Its schedule is built on the prospective brief
`rows/r2-owner-adopted-predata-amendment-2026-10-11.md` (claim #1048), never before folded into this file; each
state transition there was re-verified on fbb1f38e. Where main's tests differ, the difference is named below.

**This section supersedes 2.12's FAILWRITE arm (Codex 2) and 2.1's per-turn "refuses the durable effect of THAT
turn" rule.** The old text above is not edited. 2.12's bar "equal to the twin, with the failure injected
identically in the restoring process and its twin" is withdrawn: the twin would carry the same fault and never
reconcile (board #1063), and the old red is not retained as an arm.

**State machine (all verified).** `READY` is `Brain._r2_fault is None`. A process starts `pending`
(`Brain._r2_boot`) and reaches READY only through a complete reconciliation (`_r2_may_rearm`). Any failed or
unconfirmed common-writer append (raise, or `save_episode` returning False) calls `Brain._r2_close`
(latching ALL kinds) BEFORE `Brain._r2_restore` rolls the survivor back to its pre-effect state; a restore that
raises replaces the phase with "rollback". Nothing in-process clears the latch, including the
fault injector being removed; only a fresh `Brain` construction that reconciles does. A latched `teach`/`turn`
returns pure inference and increments `_unjournaled_refused`; a latched `sleep`, `sleep_bout`, `posed_problem` or
checkpoint call is refused before its append and effect and is reported by `Brain._report_ixn_refusal` or an
exception, not counted in `_unjournaled_refused`. `Brain._r2_seq` advances for the failed B (even when the frame
landed) and never again while latched. `ctxread` and `percept` have no live producer on main: they are named gaps,
not claims.

**FAILWRITE arm (R2-c1 family; dev seeds 15000-15007 first, then the P1 seeds of 2.6).** Per seed, in one private
persisting home:
1. A: a successful `teach` of key K1, then the `_enter_wake` checkpoint. Retain the checkpoint bytes CP_A.
2. B: one `correct`/`teach` of key K2 with the fault injected at `substrate.home.save_episode` (the writer's one
   append). Three arms by what the lake holds afterwards: `absent` (nothing written), `landed` (the complete B frame
   is written, then the call raises), `torn` (a frame header and a payload one byte short, then False). The torn
   tail differs by B: 255 B for `teach`, 1,663 B for `sleep_bout` (dev seeds 15000-15007, Codex #2885; an earlier
   reading of 255 B for both was wrong).
3. C, all in the survivor and all after B: (i) `Brain.respond` on K2 (the old step 2: a later turn on the same
   key; `Brain.respond` calls `chat`, which calls `learn_turn(text, intent)` with no feedback: kind `turn`,
   through `event_sink`, so latched it counts n_i = 1 refusal; if `chat` raised, `respond` falls back to
   `converse`, which writes nothing, n_i = 0. n_i is read, not assumed: it is the number of `turn` events the
   never-faulted reference appends for the same `respond` call); (ii) a `teach` of a new key;
   (iii) a direct `learn_turn`; (iv) one learner-affecting sleep tick (`Brain._r2_sleep_learning`); (v) one
   `Brain._r2_run_bout` call; (vi) the checkpoint, by calling `Brain._flush_interaction_learner` directly and
   observing ITS return (`Brain._enter_wake` returns None and is not the observable); (vii) one
   `posed_problem` source (`Brain._g9_apply_source`, a value the instrument builds to pass `posed_source`; if it
   cannot, C(vii) reads UNVERIFIED, never skipped). Then the fault is removed and (i)-(vii) are repeated.
4. Kill; restart in a fresh OS process with no fault; record.

**Expected durable behaviour and the pass bar (equality, no tolerance, every seed).**
- After B: `Brain._r2_fault["phase"]` is "append" (teach/turn) or "bout_append_or_preview" (`sleep_bout`, A4(d));
  the survivor's full learner snapshot
  (`Brain._r2_snapshot`, with exactly `_redo_count` and `_unjournaled_refused` popped from
  `learner.state.tree.attrs`, as main's `_semantic` helper does) equals A's; `_r2_seq` is A's + 1.
- After every C call, including the repeats: `_r2_seq` unchanged; the lake's raw bytes (length and sha256) equal
  the after-B bytes; the checkpoint equals CP_A; the learner snapshot (same two counters popped) equals A's.
  Return values: C(iv) returns False, C(v) raises `RuntimeError` naming the latch, C(vi) returns False with one
  stderr line "R2 checkpoint refused by latch", C(vii) raises `PosedRefused` ("common all-learning latch is
  closed"). C(vi) with the latch removed DOES change the checkpoint and the lake's raw bytes (observed on dev seeds,
  #2873; this section's earlier text said the checkpoint would stay equal to CP_A and is superseded). The red fires
  on the True return of `Brain._flush_interaction_learner` and on the missing "R2 checkpoint refused by latch" line;
  the checkpoint and raw-bytes bars fail with them but are not what the red rests on.
- Excluded counters, each asserted on its own: `_unjournaled_refused` is read AFTER `Brain._r2_restore` has
  returned (rollback restores it to its pre-effect value; `Brain._record_learner_event` increments it afterwards)
  and equals b (B's own count) plus one per C call that reaches `Brain._record_learner_event`: C(i) counts n_i
  (above), C(ii) `teach` 1, C(iii) `turn` 1 (`learn_turn` with no feedback); C(iv)-(vii) count 0. b is 1 when B is a
  `teach`/`turn` and 0 when B is `sleep_bout` (`Brain._r2_run_bout` increments no counter). Totals: b + n_i + 2 after
  the first pass and b + 2(n_i + 2) after the repeats: 4 and 7 for a teach/turn B with n_i = 1 (3 and 5 with n_i = 0);
  3 and 6 for a `sleep_bout` B with n_i = 1 (2 and 4 with n_i = 0).
  `_redo_count` is unchanged from its value after A in the survivor. The raw tree is retained. No other field is
  dropped from the comparison.
- Restart: `_r2_fault is None`, `_hydration_count == 1`, and the learner snapshot equals an INDEPENDENT reference:
  for `absent` and `torn` the state after A alone; for `landed` the state after A then B, captured after B returns
  and the cursor has advanced. Each reference is a separate, never-faulted persisting process on its own home that
  runs the same calls (a reference built inside the faulting `save_episode` is a pre-cursor snapshot and is wrong).
  The other arm's reference must differ from the restart (identical sides read NOT MET, ruling 9). For `torn`, the
  before-repair raw tail and the complete-prefix length and hash are retained, and the after-repair prefix is the
  last complete frame.
- Observer rule (all arms, every seed; Codex #2885, retraction #2886): every measurement taken between C calls is
  NON-MUTATING. Raw bytes are hashed once and the checkpoint, prefix and learner records are decoded from that same
  immutable snapshot; the instrument never constructs a repairing `BinLog` and never calls `home.load_state` (or
  `load_named_state`, `_log`) on the test home while the survivor is under test, because that path runs
  `BinLog.__init__` -> `_repair_torn_tail` -> truncate and measures a file the observer itself shortened (the round-2
  `torn` failure, 8 of 8 seeds, was this). A repair of the torn tail is licensed only where the restarted process
  does it, and is then bound by the `torn` restart bar above. The strict raw-bytes bar is UNCHANGED for every arm:
  length and sha256 after every C call equal the after-B bytes.
- Rearm: after a successful restart, one separately injected failure latches again.
- Report, no bar: the `phase` strings reached by each C kind.

**Constructed reachable REDs (each must fire; a red that fires on no seed is deleted, never weakened, P8).**
- **RED-KIND:** a build that latches only the failing turn's kind and lets the next kind through. ONE frozen
  construction: run B in the correct build; after B returns and before C, set `Brain._r2_fault` to None (the latch
  reopened) and wrap `Brain._record_learner_event` so B's kind (`teach`) returns pure inference and increments
  `_unjournaled_refused` with no append and no `Brain._r2_close`. Raising instead goes through `Brain._r2_close` and
  relatches ALL kinds, which is the correct build, and a mutation installed before B fails the B bars first. C(ii)
  is refused as in the correct build, so the red fires only through a non-teach call: C(iii) `learn_turn` appends a
  `turn` event (the raw-bytes and `_r2_seq` checks fail there at the latest, whatever n_i is), and C(iv)-(vii)
  append a `sleep`, `sleep_bout`, checkpoint or `posed_problem` record. For B = `sleep_bout` wrap
  `Brain._r2_run_bout` to raise before its `try`, as its own latch check does, with no `Brain._r2_close`.
- **RED-RETRY:** the latch cleared in-process by a later successful append (or by the injector's removal). The
  repeats of C append; the same checks fail.
- **RED-GATE:** `Brain._r2_may_rearm` forced True over an incomplete report. Construction at the gate itself: in
  the restarting process wrap `SpikingInteractionLearner.hydrate` to return the real report with `complete` set
  False (or one id dropped from `processed_ids`) and neuter `Brain._r2_may_rearm` to return True. Malformed events
  or receipts are NOT the injection: `hydrate` refuses them earlier ("R2 incomplete or mismatched same-pass
  receipt"), before the gate is reached. The red build rearms (`_r2_fault` None) where the correct build latches
  phase "reconcile", and the next real write is caught by the raw-bytes check.
- **RED-ORACLE:** the A-only reference used for `landed` (and the pre-cursor snapshot) must each fail a named field
  (`_cursor`, ring or authority) against the restart.
- **RED-COUNT:** a build that does not count the refused `teach`/`turn` calls fails the exact-delta check.
- **RED-CKPT:** the latch not covering the checkpoint path. Construction: `Brain._r2_fault` is set None for the
  C(vi) call only (a build whose `_flush_interaction_learner` has no latch check behaves so). It must fail on
  the True return and the missing "R2 checkpoint refused by latch" line, not on bytes (see the bar).
- **RED-POSED:** the latch not covering `posed_problem`. Construction: `Brain._r2_fault` is set None for the
  C(vii) call only (a build without the `_need(self._r2_fault is None, ...)` line of `Brain._g9_apply_source`).
  A `posed_problem` frame is appended, so the raw-bytes and `_r2_seq` checks fail.
- **RED-PARTIAL (FW-PARTIAL; new 2026-10-15, registered BEFORE the instrument implements it; Codex #2885):** a
  latched build that writes a PARTIAL learning record. Construction, in every arm and on every seed: at C(iii) (a
  direct `learn_turn`), with the latch closed, a wrapper appends to the lake file a length header plus a learner-event
  payload shorter than the header declares, leaves `_r2_seq` and the learner snapshot unchanged, and returns the
  expected refusal (pure inference, `_unjournaled_refused` incremented as in the correct build). In the absent and
  landed arms the complete prefix and the decoded learner records stay identical, so a prefix comparison alone
  returns no difference there. That is FALSE in the torn arms: the torn tail is one byte short of a frame, the
  partial append completes it, and the complete prefix moves (rv-r2amend-a #2902 measured prefix 17 -> 61 on
  constructed bytes). The red therefore rests on the strict raw-bytes bar (length and sha256 equal to after-B) in
  EVERY arm; it must fire in all six arms (teach and bout x absent, landed, torn) on every seed, and the
  instrument records per arm whether the prefix moved, without using that as the bar. A red that fires on no seed
  is deleted, never weakened (P8); an INVALID seed makes it NOT MET (R2-c3 as tightened above).
  Round 7 (2026-10-15, pre-data; rv-r2amend-a #2902 and Codex on #2898, relayed #2903): this correction follows
  a dev-seed observation (15000-15007; the instrument built the red on teach/absent only). No confirmatory or
  reserve seed has been read.

**Disagreements with what main's tests assert (fbb1f38e), each named.**
1. No main test calls `Brain.respond` on the failed key. In `tests/test_g0c_learner_persists.py`,
   `test_teaching_event_write_failure_latches_correction_until_restart` re-issues `correct`;
   `test_failed_b_latches_and_os_restart_uses_actual_durable_prefix` uses `teach` and `learn_turn`.
   C(i) is new.
2. `sleep_bout` and `posed_problem` as C calls under a latched state: only `tests/test_r2_sleep_bout.py` asserts
   the bout refusal ("sleep_bout refused by latch"); `_g9_apply_source` under a latch is not exercised by a FAILWRITE
   test. C(v) and C(vii) are new; C(vii) is the posed call and carries the bar above.
3. Main restarts compare to `_reference(include_b)` (an independent persisting process), which agrees with the
   schedule; the registered twin-with-identical-injection does not appear in main's tests and is withdrawn above.
4. Main's tests run `make_brain(seed=0)` on one key pair; the arm runs the 2.2 schedule on the dev and P1 seeds.
5. I found no main test for these crash points of #1048: after a successful event but before the checkpoint
   (partly covered by the sleep redo test), after a complete checkpoint frame before success is reported, after a
   successful checkpoint before exit, and after restart before the first new append. The instrument covers them or
   reports each UNVERIFIED.
6. Main's `test_failed_b_latches_and_os_restart_uses_actual_durable_prefix` asserts raw bytes, CP_A, the snapshot
   and `_r2_seq` ONCE, after the first C pair (`teach` then `learn_turn`; refused == 3), then only
   `_r2_sleep_learning() is False` and `_flush_interaction_learner() is False`; not after EACH C call. It has no
   repeat-after-fault-removal pass and no rearm check (the restart asserts `fault is None` only; no second failure
   is injected). It asserts `_r2_fault is not None`, not `phase == "append"`; that string is asserted only in
   `test_g0c_learner_persists.py::test_teaching_event_write_failure_latches_correction_until_restart`. Its reds
   (`test_latch_and_rollback_neuters_have_named_reachable_reds`) are latch-neuter and rollback-neuter; RED-KIND,
   RED-RETRY and RED-COUNT are new, and RED-ORACLE is only partly there (other-arm reference must differ).
7. STRICTER than main: A3 requires the raw bytes after every C call to equal the after-B bytes, and for `torn` the
   after-repair prefix length and hash. Main compares only `frozen.read_bytes() == raw_prefix` (its own pre-restart
   copy against the same variable), never after-B against after-C and never an after-repair prefix.
8. Main has no named red for the latch missing from the checkpoint path (`Brain._flush_interaction_learner`) or from
   `posed_problem`. RED-CKPT and RED-POSED are added above, constructed.

### A4. Owner #2106: `sleep_bout`

#2105 asked (owner question, R2 x G6 x F5): journal each staged sleep bout as a new R2 event kind and loosen F5's
K10c after the data was read (A), or run it on a detached non-durable learner copy (B, the integrator's
recommendation). #2106, verbatim: "Owner answer relayed verbatim by single integrator claude-cont-1013 from the
Claude chat (2026-10-13, AskUserQuestion on #2105): selected option "A: journal the bout" -- "Add a new R2 event
kind with deterministic restart redo, and a dated post-data amendment loosening K10c to allow exactly that write.
Bout effects become durable brain learning.""

Where K10c's dated post-data amendment lives: `specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md`
(its head, "POST-DATA amendment, 2026-10-13, owner-question #2105 / owner-answer #2106"). It is NOT in this file.
`specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md` still carries only the original K10c row
(its 2026-10-08 F5 amendment: "no lake write during the bout") with no 2026-10-13 text; the loosening is not
restated there.

The mechanism: `Brain._r2_run_bout` (the only caller is `instruments.staged_sleep_bout.run_bout`; there is no
live-tick caller) predicts the bout on a preview learner through `learner_bout`, appends ONE `sleep_bout` event
(slots `bout`, the three digests, `drawn`, `presentations_spent`) through `Brain._append_learner_event`, then
runs the bout. After the effect it refuses on a digest/progress mismatch or on any change of the ring or authority
cache, rolls back and latches. Restart redo goes through `instruments.staged_sleep_bout.learner_bout`.

What R2's checks must now cover for kind `sleep_bout` (new check R2-c8, dev seeds first, equality, no tolerance):
- (a) restart equals an independent never-killed reference on every 2.2 record after a bout, for a kill after the
  event and before the effect, after the effect and before the checkpoint, and after the checkpoint;
- (b) a planted `sleep_bout` event with one altered digest or progress slot is refused by redo by name and latches;
  its red is a redo that skips the verification;
- (c) exactly one `sleep_bout` record per bout, and no other lake write during the bout (the K10c reading as
  amended); ring, ring ids and authority are unchanged by a bout (main enforces this: `Brain._r2_run_bout`
  compares `ixn._snapshot()["cache"]`, which holds `_replay`, `_replay_ids` and `_authority`). Replay counts
  are NOT compared by main; R2-c8 reads `_replay_counts` before and after, a new check;
- (d) the FAILWRITE arm of A3 runs with `sleep_bout` as the failing kind B and as a C call. For B = `sleep_bout`
  the fault is at its append in `Brain._r2_run_bout`: `phase` is "bout_append_or_preview" (not "append"); no
  `_unjournaled_refused` increment (b = 0); `_r2_seq` still advances by one; the normal-chat totals are 3 after the
  first pass and 6 after the repeats (A3's formula, n_i = 1), or 2 and 4 if `respond` emits no turn.

### A5. The campaign instrument

`instruments/replay_hydration.py` was never ported (not among #1779's 20 operations: 17 Edits and 3 Writes). Main's
`tests/test_r2_replay_hydration.py` is not that instrument and not merely a helper: it holds seven test functions
(complete restart, failed-B arms, partial-B rollback, the latch/rollback reds, strict same-pass, failed hydration,
one-hydration/cache alias) plus helpers imported by five modules. It covers P1/P2 recovery mechanisms; it does not
supply the registered campaign. Its docstring says UNRUN. The instrument is REBUILT on main in its own lane
(claude-sonnet-r2inst-1015) against THIS amendment: kill points K0-K4 and twins as
registered, I_K1/I_K2 at the base head per D1, the FAILWRITE arm per A3, R2-c4 per A2, R2-c8 per A4. R2-c1..c7 and
2.5's positive controls run on dev seeds 15000-15007 first. No confirmatory seed is read before the rebuilt
instrument and this amendment are both reviewed and on the stacked head.

### A6. P3

Owner, 2026-10-15 (chat): "Start the R2 arc: docs amendment (#1037 latch, #2106 sleep_bout) plus build the
replay_hydration campaign instrument and P3, as parallel lanes. Get #2689 reviewed first." P3 was never built at
either head: `transition_sink`, `hydrate_transitions`, `pseq` and `stale_proto` appear in no file of src, tests or
tools at 56031ab9 or fbb1f38e (#2690, #2691, Codex #2719). `percept` is a registry row in
`home.LEARNER_EVENT_KINDS` only. Nothing was measured, so this is not a null. P3 is built in its own
lane (claude-sonnet-r2p3-1015) to 2.10 as registered (with 2.11 D2 and 2.12 Codex 3); any deviation lands as a
further dated pre-data amendment before seeds 15050-15057.
P3 does not block P1/P2 readiness: 2.3 and 2.10 give P1, P2 and P3 separate verdicts (a P3 red reverts P3, never P1),
so P1/P2 work proceeds on its own readiness. Literal execution "as registered" still waits on A2-A4 and the rebuilt
instrument. One consequence of A3, stated for that lane to confirm: a `percept` write goes through
`Brain._append_learner_event`, so its failure latches every kind, not only that observation's buffering and `_prev`.

### A7. Codex-14 gate `memory_unit_unreconstructible`

Absent on main (none in src, tests or tools at fbb1f38e; at 56031ab9 it is in
`readout/interaction_learning.py` and `tests/test_r2_replay_hydration.py`; at fbb1f38e only this file carries the
registered text) and no successor found: a search of the Brain, the learner and
`substrate/` for `memory unit`, `unreconstructible` and a memory/state role finds only the `ctxread` schema comment.
`ctxread` is a schema with no live producer (the writer refuses it). So the gate is a carried OPENING, unverified
on main, NOT a satisfied check. The instrument lane reports it UNVERIFIED. Any R2 verdict that covers memory
units waits on it (2.12, Codex 14).

### A8. B13 opening

Opening, no diagnosis. B13's registered cost (the 2026-10-06 P2 amendment, "Cost, a claim with a check (B13)";
2.11 D3 is the 0x04 append budget it is measured against; the 2026-10-07 addendum: per-tick p95 613 B, re-pinned
113 -> 613 B) is UNVERIFIED on main: neither confirmed nor contradicted. The live daemon's /api/state
(2026-10-15, read by the integrator, one live workload, not a dev-seed measurement) reports `append_floods` 36
and a last flood "FLOOD: tag 0x04 (IMMORTAL) appended 5631 B this tick > 8.0x ... (625 B)". Measurement of the
flood source is owed before or with the instrument arc, under the
baseline's own method, and may re-pin the cost clause inside a further dated amendment. One 5631 B live tick is not a
contradiction of a p95 (`tools/lake_append_baseline.json` still seeds tag 4 at 113 B and says "not a ceiling"); the
clause stays UNVERIFIED pending that measurement.

### A9. Panel fold 2026-10-15 (pre-data)

Panel A1/B1 identifies an over-refusal in 2.11 D2: a durable percept pair disappears on
session 3 after session 2 records chat/teach but no percept. Register both failing inputs:
the endpoints already carry `proto_boot=2, proto_flushed=2` (A1), or session 1 records
three percepts with zero boot/flushed counts before its successful prototype flush (B1).
Session 2 restores the prototypes, teaches without observing percepts, and closes; session 3
must retain the same within-session pairs. These three-session regressions are registered REDs.

Corrected evidence rule: admit a percept on its OWN durable-prototype evidence, category
strictly below its `proto_boot` or `proto_flushed`, as well as the existing later same-session
flush or immediately-next-session boot proof. Refuse `stale_proto` when these records cannot
prove that category durable, including the original unflushed crashed tail whose id was reused.
Today's count several sessions later is never proof for that tail. To preserve the B1 proof
without another percept, the common writer records the restored `proto_boot` and successfully
saved `proto_flushed` counts on ordinary `teach`/`turn` lake records too (optional metadata
accepted by their shipped schemas). Their boot count proves ONLY their own session's inherited
categories and the immediately preceding session's flush; their flushed count proves only
their own session. P3 consumes these eligible records through the SAME shared recovery pass,
alongside percepts, with no second read or store. Missing legacy metadata invents no proof.

Panel A2: O_cursor must keep a valid canonical prefix binding for its mutated cursor and
reach hydration's redo/state comparison. A prefix-binding refusal cannot count as this red.
Panel A3/B2: P3 tests exercise the shipped percept schema and hydrate branch, with no fixture
or restart-child repairs; removing either integration must produce a demonstrated refusal.

Exposure: development inputs/seeds only (15000-15007); no confirmatory or reserve seed read.
This addendum precedes code changes and new panel-fold measurements; the panels' reported
development reproductions are disclosed above, not presented as untouched data.

A9 clarification (2026-10-15, development verification): when BOTH durable counts are zero,
ordinary teach/turn records retain their existing exact event format. Zero proves no category
durable, so omitting that optional metadata changes no admission proof. Positive counts still
travel in those same records. This addresses the G0(c) event-format regression found by the
affected suite, without changing its test or weakening the reused-id refusal.

## A10. Development instrument attribution, 2026-10-16 (K4FIX)

Development seeds 15000 and 15001 on 9f5a9d58 plus PANELFOLD reproduce a K4
cursor-only inequality: respectively (1,25)/(1,26) and (1,27)/(1,28), restart/twin.
The K4 hook exits before the first interaction checkpoint in `_enter_wake`;
the reused K1 twin finishes that tick, including `_wake_tick` and its P3 percept.
That extra event is after K4's declared cut (2.8 and the 2026-10-06 K4 clause),
not an omitted restart event. No compared field, tolerance, or cursor is excluded.
The instrument will use a dedicated never-killed K4 twin: sample its LIVE `before`
record at the same pre-checkpoint hook, then delegate the real flush and continue
the ordinary schedule and all six sleep records. Retain its later post-boundary
record as diagnostic evidence, and compare K0/K1 to their original K1 twin.
This aligns the measurement with K4's registered cut rather than treating a
future wake percept as part of the crashed prefix. A planted cursor difference
must still fail the unchanged exact comparator.

The CTXREAD plant previously chose its new identity after only teach/turn/sleep,
ignoring P3's later percept on the shared sequence. The reader therefore refused
a duplicate ID before the intended unbound-effect guard could run. All constructed
plants will allocate after the full eligible typed prefix; the guard-on CTXREAD
arm must reach `R2 unbound effect: ctxread`, and its guard-off arm must bypass
that exact guard and complete. This retains the control with a constructible input.
It does not establish a live ctxread producer, memory-unit reconstruction, or
Codex-14 success: the A7 opening remains UNVERIFIED. No confirmatory/reserve data
were read; the new regression tests are red before the instrument repair.

A10 diagnostic clarification, 2026-10-16: the later post-boundary diagnostic
retained by the implementation is its `post_boundary_cursor`, not a second full
learner record. The full live `before` record is sampled at K4's pre-checkpoint
cut, and all six full registered sleep observations remain compared exactly.

## A11. Re-panel 2 fold, 2026-10-16 (FOLD3, pre-fix; integrator claude-cont-1013)

Codex lens A (re-panel 2) reported three MAJOR defects on the FOLD2 tree; the integrator reproduced
all three with the lens's own script before this amendment was written. No campaign data, no
confirmatory or reserve seed, is read by the fold. Each fix must ship red-first: a regression that
fails on the FOLD2 tree and passes after the fix, plus a control that is not the fix itself.

1. Witness authority (P3 hydration). A session whose prototype-durability witnesses disagree (any
   two teach/turn/percept witnesses carrying different boot or flushed counts, in ANY order) is
   refused as a whole: none of its witnesses may serve as durability proof for any earlier
   session's pairs, and the refusal is counted by name. Red: false-then-truthful and
   truthful-then-false orderings, for teach and for turn, must both restore zero stale pairs. Control:
   an honest history with agreeing witnesses still restores its expected pairs.
2. O_cursor attribution (instrument). The cursor red counts only when a redo step COMPLETED and the
   intended digest/progress comparison then disagreed (or a carrier-dependent divergence was shown);
   an exception raised inside the redo step, of any type other than the intended mismatch, makes the
   arm INVALID, never fired. Red: lens A's unrelated-OSError construction must not score O_cursor
   fired. Control: the intended mismatch still fires.
3. Scratch-home ownership (instrument, P4b). `run_process` binds a home only when the instrument
   itself created it fresh (a private, empty root created by the instrument, marked as owned) and
   it is not, does not contain, and is not inside the live home; any other caller-supplied
   directory is refused before any read or write, and `_plant_bout` re-checks the ownership
   mark and the lake-file path before a destructive open. Red: lens A's existing non-temporary
   directory must be refused. Control: the instrument's own fresh scratch root still runs.

A11.1 correction, 2026-10-16 (integrator, pre-merge; no campaign data read). A11 item 1 as written
("different boot or flushed counts ... refused") contradicts the registered successful-flush control
(`test_stale_proto_an_unflushed_crashed_tail_is_refused_not_hydrated`): within one session the
flushed count legitimately RISES after a landed flush. FOLD3-W found this before integration and did
not weaken the control. Corrected rule: (a) every witness of one session must carry the SAME boot
count; any boot disagreement, in any order, refuses the whole session as durability proof, counted as
`session_boot_conflict`. (b) Flushed counts of one session, read in sequence order, must be
non-decreasing; a decrease refuses the whole session, counted as `session_flushed_conflict`; a rise is
a landed flush and is admitted. Reds: lens A's false-then-truthful and
truthful-then-false boot orderings for teach, turn and a first percept; a flushed decrease. Controls:
the successful-flush rise control unchanged, and the honest agreeing history.

A11.2 correction, 2026-10-16 (integrator, after Codex lens A re-panel 3 on FOLD3-I; no campaign data read).
Item 2 generalized: attribution is by EXCEPTION TYPE, never by message text. The intended comparison
failure is raised only as one dedicated exception class defined by the instrument at the comparison
itself; EVERY omission arm (O_cursor, O_wm and each sibling) fires only on a completed redo followed
by that class (or a demonstrated carrier-dependent divergence). Any other exception, from the step,
any callback (effect_actor included), or anywhere else, makes the arm INVALID, never fired, even if
its message imitates the mismatch. Reds: lens A's effect_actor raising ValueError with the mismatch
text; O_wm with the unrelated OSError step. Control: the intended mismatch fires each arm.
Item 3 narrowed, with its structural reason (disclosed, not hidden): the ownership check defends
MISDIRECTION and STALE REUSE, not deliberate forgery by a caller holding the instrument's own
privileges -- such a caller can open the lake file directly, so no in-process check can bind it. Rule:
the parent (`run_seed`) creates each arm home with `tempfile.mkdtemp` and a fresh per-launch nonce
(`secrets.token_hex`), writes the owner mark, and hands the nonce to its child; `run_process` refuses
any root whose mark is missing, names another root, or carries a nonce other than the one the
current launch issued, and still refuses the live home in every case. Reds: an unmarked existing
directory; a marked directory reused from an earlier launch (stale nonce). Control: the instrument's
own fresh root runs end to end through `run_process` (a real subprocess, not a mock).

A11.3 correction, 2026-10-16 (integrator, after Codex lens A re-panel 3 on FOLD3-W; no campaign data read).
(1) Malformed witness metadata is never skipped. A teach/turn/percept witness whose `proto_boot` or
`proto_flushed` is ABSENT is legacy and proves nothing (fails closed, as before); one whose field is
PRESENT but not a non-negative decimal integer string (e.g. "unknown", "", None) invalidates its
whole session as durability proof, counted by name as `session_witness_malformed`. Red: lens A's
false-boot witness followed by a truthful-boot witness with malformed flushed metadata, for teach and
turn, restores zero stale pairs. (2) Refusal accounting: every percept discarded by a session-level
conflict is counted under the reason it is assigned, in addition to the session-level counters.
(3) Disclosed narrowing, with its structural reason. A flushed-count RISE is admitted as written:
the brain writes `proto_flushed` only after `save_state(PROTOTYPES_STATE)` returns True
(`engine/brain.py`, the `_r2_proto_flushed = durable_units` advance in the prototype flush), so a
well-formed rise with no landed save has no brain-producible path; it requires a forged lake record,
and a writer of well-formed lake records can write the prototypes themselves -- the hydrator cannot
authenticate the one lake against itself. Session-bound save evidence would need a new field in the
prototype state record (P2/P4b work, not this fold) and is carried as a disclosed follow-up. The
earlier boot-conflict refusal (A11.1) is kept as cheap defence in depth; it, too, was raised by a
planted record. Stopping rule for this arc's reviews: a finding with no path the brain itself can
produce (crash, torn write, ordering, stale state, schema drift) is DISCLOSED, not folded.

A11.4 addition, 2026-10-16 (integrator, after Codex lens A re-panel 4 on FOLD3-I; no campaign data read).
A child result counts only if it was produced by THIS launch: `run_seed` removes any existing output
file for the arm before launching its child, and accepts the arm's JSON only when the child exited 0
and wrote that file afresh; a non-zero exit, a missing file, or an unparsable file makes the arm an
error (INVALID), recording the exit code. Red: lens A's reused output directory with a child that
exits 37 without writing must yield an error, not the earlier result. Control: a successful child
still yields its result.

## Mechanism re-diff fbb1f38e..c4c10b28, 2026-10-29, pre-data; no confirmatory or reserve data read

Row plan 1.0: "before any Phase-2 seed, diff each campaign pre-registration's named mechanisms against the bytes; a changed
mechanism owes a dated pre-data amendment first." The 2026-10-15 amendment above diffed R2 against fbb1f38e. origin/main
is now c4c10b28 (621 commits later). This addendum repeats the diff for that range. It edits nothing above this heading,
is documentation only, and amends no clause: it names what changed and what the next dated pre-data amendment must state.
Lane r2-mechdiff (static analyst). No code, test, instrument or brain was run (campaign mode: no pytest, no Windows python,
no GPU); every verdict below is a reading of `git.exe show`/`diff` bytes, by `ast` comparison of each named symbol. The full
mechanism list with spec line, file and line pair (fbb1f38e -> c4c10b28) is
`lanes/engine-1024/r2-mechdiff-row/MECHS.md` (out of git). Not Codex-reviewed.

### R1. Mechanism list (item 1, summary)
Every mechanism, function, file and constant named in 2.1-2.12 and A1-A11 was resolved: the learner
(`readout/interaction_learning.py`), the lake events (`substrate/learner_events.py`), the Brain writer, latch, bout and
checkpoint methods (`engine/brain.py`), the lake and codec (`substrate/home.py`, `substrate/binlog.py`), the grower
(`growth/neurogenesis_engine.py`, `growth/growing_learner.py`), the bout runner (`instruments/staged_sleep_bout.py`), the
campaign instrument (`instruments/replay_hydration.py`), `tools/lake_append_baseline.json`, the five `brainstem/` files, and
the seed block 15000-15067.

### R2. The diff (item 2, summary)
- UNCHANGED or MOVED (same bytes, new line): the whole latch, rollback, bout and checkpoint machinery A3 and A4 describe
  (`_record_learner_event`, `_r2_close`, `_r2_restore`, `_r2_snapshot`, `_r2_may_rearm`, `_r2_fresh_learner`, `_r2_sleep_step`,
  `_r2_sleep_learning`, `_r2_run_bout`, `_flush_interaction_learner`, `_report_ixn_refusal`, `_g9_apply_source`,
  `neuter_sleep`, `_enter_wake`, `tick`, `respond`, `correct`); the learner except `hydrate`; `scan_learner_events`,
  `read_learner_events`, `prefix_digest`, `SLEEP_DIGESTS`, `SLEEP_PROGRESS`; `save_episode`, `load_episodes`,
  `next_session`, `LEARNER_EVENT_KINDS`, `_log`, `save_named_state`, the frame codec (`pack_episode`, `_MAGIC`, `_LEN`),
  `BinLog._repair_torn_tail`; every def of `instruments/staged_sleep_bout.py`; the `brainstem/` files, `engine/live_brain.py`,
  `sleep/experience_replay.py`, `sleep/sleep_replay.py` and `tools/lake_append_baseline.json` (no diff at all).
- CHANGED-COSMETIC: `home.save_state`, `home.load_state`, `home.state_path` (docstrings only, AST equal);
  `growth.growing_learner.GrowingLearner` (one comment); the `learner_events.py` module docstring; `tests/test_g0c_learner_persists.py`
  (+12: the injected fault is scoped by `monkeypatch.context()` where it used `monkeypatch.undo()`; same assertions).
- CHANGED (behaviour), with commit:
  1. `percept` is live (9f5a9d58, d5a32337, b24ea8e7, 60490871). `KIND_SPECS["percept"]` exists;
     `SpikingInteractionLearner.hydrate` has an `elif kind == "percept":` branch that only moves `_cursor` (before: `raise
     ValueError("R2 unbound effect: percept")`; `ctxread` still raises); `Brain._record_percept` and
     `_attach_percept_transitions` are new; `NeurogenesisSelfImprove.observe` calls `transition_sink` before buffering and
     raises `RuntimeError("percept transition write refused; observation not buffered")` when it does not land;
     `hydrate_transitions`, `restore_transitions`, `pack_percept` are new; `Brain._r2_boot` hands the one pass's events on
     (`_r2_percept_events`); `Brain._append_learner_event` maps `percept` to stream `wake` and encoder
     `arm.percept gained width=<dim>` and adds `proto_boot`/`proto_flushed` to `teach`/`turn` events when either count is
     nonzero (A9). `Brain.neuro` is a property over `Brain.grower` (`growth/grower.py`, G3-T1); `_wake_tick` calls
     `grower.observe`, `_sleep_tick` calls `grower.tick()`.
  2. `substrate/binlog.py` (f2534a94, 47955b4e, 1ea6b303, ec18e9c5, 080f967a, Phase 1a): a dedup `BinLog` keeps its payloads
     resident and serves `replay()` from memory while `stale()` (size, mtime_ns, inode, and the last frame read from disk)
     says the file is unchanged; the lock is re-entrant; `rewrite()` writes a private `mkstemp` file and refuses an empty
     payload; `append`/`extend` stat before and after. The bytes `append`, `extend` and `rewrite` write, and the frame
     format, are unchanged.
  3. P4b persistence (bebdd0e2, 01802dd4, da936776, ddac65d2, 63ced7d5, e1baa9d2, 8c81961d, 697f802b):
     `Brain._flush_prototypes` writes `arm.q`, the affect SOM and the lifelong learner into `growing_learners.json`, keeps
     refused sub-records in `growing_learners.refused.<store>.<id>.json` (`_keep_refused`), and `Brain.__init__` restores
     them; the language organ and nightly skills are saved into the lake (`Brain._enter_sleep` and `Brain.close` call
     `lm.save()` with no path; `Brain.__init__` reads `load_named_state(STATE_NAME)`); `home.load_named_state` now RENAMES a
     legacy sidecar on a hit (`_retire_sidecar`).
  4. `Brain._sleep_tick` (f2534a94): one `KnowledgeLog` and one `replay` serve the schema-induction and consolidation
     blocks (perf; written to give the same records the second replay gave).
  5. `instruments/replay_hydration.py` now exists (2166 lines; 28f0d5d0, d5a32337, b24ea8e7). `DEV_SEEDS =
     tuple(range(15000, 15008))` at :54; a seed outside it is refused at :230 ("no confirmatory licence").
  A2 (2026-10-15) said "`percept` is a registry row only: it has no `KIND_SPECS` schema and no writer" and A6 said P3 was
  never built. Both are true of fbb1f38e and false of c4c10b28; A9-A11 (written against later trees) already govern the P3
  evidence rules and are consistent with the code read here (`stale_proto`, `session_boot_conflict`, `session_flushed_conflict`,
  `session_witness_malformed` are all in `hydrate_transitions`).

### R3. Disposition per CHANGED item (item 3) - does the registered prediction still test the same thing?
1. `percept` live. R2-c1 to c6 are unchanged in meaning: the restart and its twin run one schedule, so the percept events of
   wake ticks enter `(session, seq)`, `_r2_seq`, the cursor and the event ids of both (2.1 already shares one `seq` over
   `teach`, `turn`, `sleep`, `percept`; A10 handled the K4 cut). R2-c7 (P3) is the claim that now has code. Gaps the amendment
   must state, with no new design: (a) the 2.2 record set contains no `_r2_pseq`, `_r2_proto_boot`, `_r2_proto_flushed` or
   grower buffer, and `Brain._r2_snapshot` (unchanged) holds none of them, so P1/P2 equality says nothing about them and
   only R2-c7 covers them; (b) `hydrate` redoes a post-checkpoint `percept` by moving the cursor only, so its grower effect is
   restored by `restore_transitions`, not by redo, and R2-c7 must be the check that names that; (c) a teach/turn event carries
   two extra string slots once a prototype count is nonzero, so the byte sizes in 2.7 (266 B, 940 B) and A3 (255 B and 1,663 B
   torn tails, measured on dev seeds on earlier trees) are descriptive figures that no bar rests on: the bars are
   length-and-sha equality, and the sizes must be re-measured before they are quoted.
2. `BinLog`. The raw-bytes bars (A3: length and sha256 equal after every C call) and the independent-reference restart bar
   still test the same thing: they read the file or run in a fresh OS process, and the bytes written are unchanged. What
   changes is the survivor: its `replay()` can now come from memory, and `stale()` is stricter than the size-only test the
   observer rule (A3) was written against, so any raw edit behind a cached instance (the injected `landed` and `torn`
   writes, `_plant_bout`'s truncate) rebuilds it, and the rebuild runs `_repair_torn_tail`. The amendment must restate the
   observer rule as binding for the survivor's own reads as well as the instrument's (no C call may reach the lake through
   `home._log`, `load_state`, `load_named_state` or `Brain.__init__` between the torn injection and the kill), and
   must record that every dev-seed observation in A3-A11 (#2873, #2885, the K4 cut, the 255 B and 1,663 B figures) was taken
   on a tree that predates 2026-10-17, so those observations are dev-seed evidence for the instrument's construction, not
   observations of c4c10b28. No bar was derived from them except those figures.
3. P4b persistence. The interaction-learner claim (R2-c1 to c6) does not name `arm.q`, the affect SOM, the lifelong
   learner, the language organ or the nightly skills; a restart now restores more of them than before, which cannot make a
   2.2 record differ but is outside the scope sentence of 2.3 ("the interaction learner and the sleep carriers that touch
   it") and must not be read into an R2 verdict. `Brain._enter_sleep` and `Brain.close` now write the language organ into
   the lake when `_acq_lm` was built (a `respond` builds it): a lake write that is not a learner event. The A3 C-calls and
   the A4 bout path call none of `_enter_sleep`, `close`, `_flush_prototypes`, so the bars are unaffected; R2-c8(c) "no other
   lake write during the bout" is a statement about the bout and still holds on the unchanged `_r2_run_bout`, but any
   lake-byte comparison taken across a sleep entry or `close` (K0, K1 around `_enter_wake`) now includes these records and
   the larger prototype record. The amendment must say so. `_flush_prototypes` advances `_r2_proto_flushed` only when
   `save_state` returns True; `save_state` returns False for a byte-identical re-save, so the count then fails closed
   (A11.3 item 3 states "only after save_state returns True": consistent, not a defect).
4. `_sleep_tick` single replay. Intended output-equivalent; not verified by this lane. The sleep event digests
   (`rng_digest`, `weights_digest`, `rewirer_digest`) cover the learner and rewirer, which this edit does not touch. No
   amendment text is owed beyond listing it.
5. `Brain.neuro` property and `Grower`. grower.py documents byte-identical forwarding and the diff shows pure forwarding;
   the P3 sink is set on the engine through the property. No R2 prediction depends on the attribute path. Not verified by
   running.
6. The instrument. A5 said no confirmatory seed is read before the rebuilt instrument and this amendment are both
   reviewed and on the stacked head. The instrument is on the main lineage at c4c10b28; its review status is the
   integrator's to state, not this lane's. Its dev-seed guard (:54, :230) is consistent with A1.

### R4. Data-exposure check (item 3)
No confirmatory or reserve seed of R2's block (15010-15067) was read, and none of this lane's checks read a result. Searched
2026-10-29, regex `(^|[^0-9.])150(1[0-7]|2[0-7]|3[0-7]|4[0-7]|5[0-7]|6[0-7])([^0-9]|$)` unless noted:
- `git.exe log --all --oneline -S15010`: 13 commits (substring search); `-G` with the full block regex: 9 commits. Every
  hit is one of: R2's own registered allocations and refusals (b6812e3f, 42f35b4c, b38c023a, ccf6d297, 28f0d5d0,
  9f5a9d58 - `rh.run_seed(15010, ...)` in `tests/test_replay_hydration_instrument.py` is a refusal test), or an unrelated
  number (a `15057.2257` float in `tools/g4_t3_campaign.json` 9588545f/cbd8ce14, `0.15044`/`0.15062` floats 9acc7308, and
  afa611a0, 27c90b7b, 9c872e8e, cb33b8dc, 963799e8, where `15010` occurs only inside a longer token: no standalone match in
  their diffs).
- `git.exe grep` of c4c10b28 (all tracked files): the R2 spec, `tests/test_replay_hydration_instrument.py`,
  `tests/test_r2_p3_percept_transitions.py`, `tools/g4_t3_campaign.json` (a float), `native/sndl-core/tests/fixtures_rng.json`
  (a hex float), `specs/.../r3-conf-F1-records.sha256` (a hash), three PDFs under `research/`.
- The board `sndl-board-live` (msgs and codex_p0, 32 files): pids (`"pid": 15056`), hash substrings, R2's allocations, and
  review posts that state no seed >= 15010 was read (#2689, #2690, #2691, #2710, #2711, #2719, #2731, #2736, #2764, #2843,
  #2844, #2879, #2881, #2882, #2895, #2902, #2903). Memory notes `project-snn-engine-row-plan-1013.md` and
  `project-snn-session-handoff-1014.md`: allocation lines only.
- Lane folders and rows/ (text files .json/.jsonl/.csv/.txt/.log/.md outside src, tests, native, tools, specs, docs, models), seed-context regex `[Ss]eeds?[^0-9A-Za-z]{0,30}150(1[0-7]|...|6[0-7])\b` and `"seed": 150xx`: 47 files, all briefs, review and lane logs (codex-*.log, panel-*.log, REPORT.md, BRIEF.md) quoting R2's allocations, the `rh.run_seed(15010, tmp_path)` refusal test, or "no seed >= 15010 read". No result file (results/, *.json records) holds a seed of the block. The grep exited 2 (unreadable entries skipped); a wider number-only scan of the same tree was cut at its time limit and is not claimed.
This is a text search. Binary artifacts and any process that never wrote a file were not examined, so the qualification
of A1 stands: this is non-exposure by search, not proof.

## DATED PRE-DATA AMENDMENT 2026-10-29 -- DRAFT (not frozen; no confirmatory or reserve data read)

Row plan 1.0: "a changed mechanism owes a dated pre-data amendment first." The addendum above ("Mechanism re-diff
fbb1f38e..c4c10b28") named what changed. This section states, for each CHANGED item, which registered prediction reads
through it and whether it still tests the same thing. It is a DRAFT for the integrator and the policy-enforcer: it is not
frozen until the integrator lands it, and it changes no bar, no seed block, no decision rule and no design. Lane r2-amend,
documentation only; nothing was run (campaign mode: no pytest, no Windows python, no GPU). Every cite below is to this
file's own sections (2.x, A1-A11, R1-R4) or to `lanes/engine-1024/r2-mechdiff-row/MECHS.md` (out of git, "MECHS"). Not
Codex-reviewed. Three outcomes only: UNAFFECTED, RE-STATED (exact new wording), OWED (a run or a statement).

### D1. Disposition per CHANGED item

**1. `percept` is live (9f5a9d58, d5a32337, b24ea8e7, 60490871).**
- R2-c1 to R2-c6 (2.3) -- UNAFFECTED. R2-c1 and R2-c6 compare the restoring process with its twin "on every record of
  section 2.2" (2.3); the 2.2 record set holds the learner tree, ring ids and labels, RNG, watermark, replay and redo
  counts, rewirer state, weights, `_step`, `_nights`, the grower's `observed`/`live_n`, and the `apply()` call count. It holds
  no `_r2_pseq`, `_r2_proto_boot`, `_r2_proto_flushed` or grower buffer, and `Brain._r2_snapshot` (unchanged, MECHS A3)
  holds none of them. Twin and restoring process run one schedule, so the percept events of wake ticks enter `(session, seq)`,
  `_r2_seq`, the cursor and the event ids of both (2.1 shares one `seq` over `teach`, `turn`, `sleep`, `percept`). The
  comparison is the same comparison.
- R2-c7 (2.10) -- UNAFFECTED in bar, RE-STATED in two sentences. The bar (hydrated `_buf_x`/`_buf_y`/`_categories` equal the
  twin's, and the model after the next refit equals the twin's, on every seed, no tolerance) is unchanged, and the code now
  matches 2.10's schema, writer and reader text (MECHS A5). New wording for the two statements the code falsifies or sharpens:
  (i) A2's sentence "`percept` is a registry row only: no `KIND_SPECS` schema and no writer" and A6's "P3 was never built"
  are true of fbb1f38e only; at c4c10b28 `KIND_SPECS["percept"]` exists, `Brain._record_percept` writes it, and
  `NeurogenesisSelfImprove.observe` calls `transition_sink` before buffering. (ii) A post-checkpoint `percept` event is redone
  by `SpikingInteractionLearner.hydrate` moving the cursor only (`elif kind == "percept":`); its grower effect is restored
  by `restore_transitions`/`hydrate_transitions`, not by redo. R2-c7 is the only registered check that reaches it, and
  P1/P2 equality says nothing about the grower buffer, `_r2_pseq` or the prototype counts.
- Event byte sizes (940 B per percept event, 2.10 and 2.7; 266 B, 2.7; 255 B and 1,663 B torn tails, A3) -- RE-STATED: they
  are descriptive figures measured on earlier trees, and no bar rests on them (the bars are length-and-sha equality). A
  teach/turn event now gains two decimal-string slots when a prototype count is nonzero (MECHS A3, A9). They are not to be
  quoted for c4c10b28 until re-measured.
- OWED: the R2-c7 reds listed in 2.10 (CROSS, O_buf, DUP, REORDER, HELDOUT, MISSING, WIDTH, VOCAB, INSTRUMENT) and its
  positive control run on dev seeds against the c4c10b28 instrument. See D3.

**2. BinLog resident replay and `stale()` (f2534a94, 47955b4e, 1ea6b303, ec18e9c5, 080f967a).**
- A3's raw-bytes bars (length and sha256 equal after every C call) and the fresh-OS-process independent-reference restart
  bar -- UNAFFECTED: they read the file bytes or run in a fresh process, and `append`, `extend`, `rewrite` and the frame
  format write the same bytes (MECHS A4).
- The observer rule (A3) -- RE-STATED. New wording: "No C call may reach the lake through `home._log`, `home.load_state`,
  `home.load_named_state` or `Brain.__init__` between a torn or landed injection and the kill, and the same holds for the
  surviving process's own reads. A dedup `BinLog` serves `replay()` from resident payloads while `stale()` (size, mtime_ns,
  inode and the last frame read from disk) says the file is unchanged; a raw edit behind a cached instance (the injected
  `landed` and `torn` writes, `_plant_bout`'s truncate) makes it stale, and the rebuild runs `_repair_torn_tail`." The rule's
  force and the bars are unchanged; only the mechanism it guards is stricter than the size-only test A3 was written against.
- The dev-seed observations in A3-A11 (#2873, #2885, the K4 cut, the 255 B and 1,663 B figures) -- RE-STATED as dev-seed
  evidence for the instrument's construction on trees that predate 2026-10-17, not as observations of c4c10b28. No bar was
  derived from them except those two byte figures (see item 1 above).
- OWED: a dev-seed run showing the torn and landed injection arms (R2-c3's reds) still fire and still leave the file the
  instrument expects under the survivor's resident cache. See D3.

**3. P4b persistence (bebdd0e2, 01802dd4, da936776, ddac65d2, 63ced7d5, e1baa9d2, 8c81961d, 697f802b).**
- R2-c1 to R2-c7 -- UNAFFECTED. The registered claims name the interaction learner and the sleep carriers that touch it
  (2.3 Scope); none names `arm.q`, the affect SOM, the lifelong learner, the language organ or the nightly skills. A restart
  now restores more than before, which cannot make a 2.2 record differ and is outside the scope sentence; it is not to be read
  into an R2 verdict.
- R2-c8(c), "no other lake write during the bout" (A4) -- UNAFFECTED: it is a statement about `_r2_run_bout`, which is
  byte-unchanged, and the A3 C-calls and the A4 bout path call none of `_enter_sleep`, `close`, `_flush_prototypes`.
- Lake-byte comparisons taken across a sleep entry or `close` (K0; K1 around `_enter_wake`) -- RE-STATED. New wording: "A
  lake-byte comparison across `Brain._enter_sleep` or `Brain.close` includes the language organ (`lm.save()` writes
  `STATE_NAME` into the lake when `_acq_lm` was built, which a `respond` does), the nightly skills, the larger prototype
  record (`growing_learners.json` now carries `q`, `affect`, `lifelong`) and any `growing_learners.refused.<store>.<id>.json`
  (`_keep_refused`); `home.load_named_state` now renames a legacy sidecar on a hit." The fail-closed count is consistent with
  A11.3 item 3: `_r2_proto_flushed` advances only when `save_state` returns True, and it returns False for a byte-identical
  re-save.
- OWED: none beyond D3's dev-seed run (the K0 and K1 lake-byte records must be re-taken at c4c10b28 before any is quoted).

**4. `Brain._sleep_tick` single replay (f2534a94, performance).**
- R2-c1 and R2-c6 -- UNAFFECTED as comparisons: twin and restoring process both run the head's `_sleep_tick`, so equality is
  read between two runs of one code. The sleep event digests (`rng_digest`, `weights_digest`, `rewirer_digest`) cover the
  learner and rewirer, which this edit does not touch (MECHS A3).
- R2-c2 (I_K2 differs from its twin; I_K1 equals it) and R2-c4 (replay-OFF differs from replay-ON in post-night weights) --
  they read `_sleep_tick`'s output as an observable, so they test the same thing only if the edit is output-equivalent.
  The commit says it intends to give the records the second replay gave; r2-mechdiff did not verify it. OWED: an
  output-equivalence check on a dev seed (the parent of f2534a94 against the head, same schedule, the 2.2 post-sleep-tick
  records equal), or an integrator statement that the perf lane already holds one. See D3.

**5. `Brain.neuro` property and `Grower` (60490871, G3-T1).**
- R2-c1 to R2-c6 -- UNAFFECTED: no prediction depends on the attribute path (MECHS A3 and A5); grower.py documents
  byte-identical forwarding and the diff shows pure forwarding.
- R2-c7 and 2.2 -- RE-STATED, wording only: wherever 2.10 writes `neuro.tick()` and 2.2 reads "the grower's `observed`/
  `live_n`", read "`Brain.grower.tick()` and `Brain.grower`'s `observed`/`live_n`, reached through the `Brain.neuro`
  read-through property (`grower.percept.engine`)"; the P3 sink is set on the engine through that property. The bar is
  unchanged. 2.10's cite "brain.py:1494-1496" is stale; the attach is `Brain._attach_percept_transitions()` after the
  prototype restore (MECHS A3).
- OWED: none beyond item 1's R2-c7 dev run, which exercises the forwarding.

**6. The rebuilt instrument (`instruments/replay_hydration.py`, 28f0d5d0, d5a32337, b24ea8e7).**
- A5's precondition ("no confirmatory seed is read before the rebuilt instrument and this amendment are both reviewed and on
  the stacked head") -- UNAFFECTED as text; the instrument is on the main lineage at c4c10b28, and its dev-seed guard
  (`DEV_SEEDS = range(15000, 15008)` at :54; a stray seed refused at :230, "no confirmatory licence") is consistent with A1.
- OWED: an integrator statement of the instrument's review status (which panel read which commit, and whether this amendment
  was reviewed alongside it), and of whether the confirmatory-seed licence is still withheld. This lane does not state it.

### D2. The P1 question: is R2 grandfathered or does it owe a "Sample size (P1)" section?

`docs/POLICIES.md` at c4c10b28, lines 172-207, the P1 amendment (owner directive 2026-10-23, relayed on the board as #4876;
review-enforced, PROVISIONAL until its gate is built and counted). Its item 1: "Every pre-registration carries a 'Sample
size (P1)' section: the measured SD the target comes from (a prior run's paired SD or a pre-registered pilot), the effect the
claim needs, and the fixed-sample target at power .8 ... after the pre-registration's own multiplicity correction across
endpoints." Its item 3: "A fixed count only for a one-shot run, its power stated before data; a result at an underpowered
count is INCONCLUSIVE (P8 item 4)." The CLOSED-list sentence, at :206 (the line continues from :205):

> "the pre-registrations frozen before this amendment landed are a CLOSED list, grandfathered by name when the gate is built
> (ancestry is not the trigger: a branch merged after the landing would slip past it)."

Facts, stated without a ruling:
- R2's registration is "frozen at the commit that lands this file" (section 2, line 76), and the amendments dated 2026-10-04,
  2026-10-06, 2026-10-15 and 2026-10-16 (A1-A11) all predate the 2026-10-23 directive. The seed blocks (2.6: P1 15010-15017,
  P2 15030-15037, P3 15050-15057, S = 8 each, plus reserves) and the verdict rules (2.3) are in the frozen text.
- The gate is not built (the amendment is PROVISIONAL), so the closed list has no names in it yet. [Superseded by D-68 / D2.1:
  R2 goes on the closed list by name; claude/p1-gate-1029's GRANDFATHERED names this file.] Whether "frozen before this
  amendment landed" covers R2 turns on whether a dated amendment (this one, and A1-A11 before it) re-freezes a registration.
  This draft changes no bar, no seed block and no decision rule, so on that reading R2 stays in the pre-landing class and is
  to be named in the list when the gate is built; on the other reading it owes the section before any 15010-15067 seed.
  Which reading holds is the integrator's and the policy-enforcer's to decide, and it is not decided here. RULED 2026-10-29 (integrator, accepted by launcher D-68): see D2.1.
- If a section is owed, the variance source the P1 text asks for does not exist in R2's design: section 2.3 states that
  "equality has no variance, so no statistic and no majority-of-seeds". Every R2 verdict (R2-c1 to R2-c7) is exact equality
  on every seed, and R2-c2 and R2-c4 are "differs on EVERY seed". There is no paired SD to measure, and the dev seeds
  15000-15007 are exposed (2.6), so a pilot SD would have to come from a new pre-registered pilot, not from them.
  What a section could state instead is the one-shot reason (P1 item 3) and the seed-count arithmetic: with S = 8 seeds per
  block and zero failures observed, the one-sided 95% upper bound on the per-seed failure probability is 1 - 0.05^(1/8) =
  0.31 (arithmetic only, not run). Whether a bar that cannot reject a per-seed failure rate below that is acceptable for an
  exact-equality mechanism claim is the P1 question the owner's directive leaves to the reviewers.

### D3. Owed before any 15010-15067 seed (ordered)

1. Integrator statement: the P1 reading in D2 (grandfathered by name, or a "Sample size (P1)" section owed, with the variance
   source), and the instrument's review status (D1 item 6). Statements, no run. The P1 reading is RULED (D-68, see D2.1); the instrument's review status (D1 item 6, the OWED
   line in D1) stays OWED.
2. Review of this amendment (policy-enforcer, static) and its landing as a frozen amendment dated before any confirmatory read.
   Until then it is a draft and the confirmatory licence stays withheld.
3. A dev-seed run (15000-15007 only) of the FIXED instrument (claude/r2-instr-fix-1029@b311e78d until it lands, then its landed sha), under the heavy-run lock after the campaign: the R2-c4
   positive controls first (planted 1-ulp weight change, one swapped ring entry, the omission arms each move their records),
   per 2.5, because a failed positive control holds the disposition. The same run carries the R2-c2/R2-c4 observables at head. The instrument itself runs no schedule at two heads (`--head base` is the e1b3ae3c I arms, `BASE_ROLES` only, never the parent of
   f2534a94), but the dev-seed RUNBOOK of this run carries D1 item 4's `_sleep_tick` output-equivalence check (lanes/engine-1024/r2-devprep-row
   RUNBOOK.md item 3 step 4, r2dev-item3.ps1: the head's instrument with `--base-src` = the tree of f2534a94's parent on the equivalence
   roles, compared by `r2dev_read.py equiv` with records_equal), which is route (b) below once it runs (enforcer #6892 MINOR-2). D1 item 4 is
   discharged by either (a) an integrator statement that cites, by commit, a check the perf lane already holds
   (the parent of f2534a94 against the head, one dev seed, one schedule, the 2.2 post-sleep-tick records equal),
   or (b) a two-head comparison tool named by commit and run on a dev seed under the heavy-run lock. Until one
   lands, D1 item 4 stays OWED, and R2-c2/R2-c4 are read as testing the head's `_sleep_tick` (MAJOR-5 of #6735).
4. A dev-seed run of the constructed reds of 2.4 and of R2-c7's reds (2.10), including the torn and landed arms under the
   survivor's resident BinLog cache (D1 item 2): each must fire on a dev seed; a red that fires on none halts the arc (R2-c3, P8).
5. Re-measure the lake-byte records of K0 and K1 and the event sizes (D1 items 1 and 3) at c4c10b28 before any is quoted.
5a. The finding (iii) red-first test, named by test id, is SHOWN failing on the parent of the D-79 fix for its stated reason (the
   restored brain's clock state or sleep/wake differs from the uninterrupted twin's), by the RECORDED pytest log of that test id on
   claude/r2-instr-fix-1029 at the commit that lands it (6fd1344e, or its landed sha), run BEFORE claude/circadian-p4b-1029 stacks on
   it (first line = the command). The record of choice is the same test run with --runxfail, its log matching either assertion message
   ('restored clock is not the twin's at the cut', :128, or 'the restored brain's day diverged from the twin's', :129). A strict XFAIL alone is weaker: an AssertionError raised anywhere inside
   run_brain or its fixtures also logs XFAIL (enforcer #7114, #7141 NIT). The circadian branch has no commit
   where the red can fail: it first appears there after the first fix (enforcer #7054 MINOR-2). A stdlib simulation stands only
   as the pre-run prediction (enforcer #7018 MINOR-1); the b311e78d test does not meet this at seed 15000 (D1.4). Then the launcher's D-79 fix (claude/circadian-p4b-1029) has
   landed with its review, together with 5c, before any seed outside the dev block (15010-15067); then R2's K3 arms assert the
   resumed phase (D1.2). (Enforcer #6948 MINOR-1.)
5b. The K1g (R2-c7, P3) arms run on dev seeds 15000-15007 under the heavy-run lock and each 2.10 red fires, before any P3 seed (15050-15067).
5c. Lands with 5a's fix, never after it: `replay_hydration.chrono_clean` (b311e78d:1759-1769, which requires construction `step == 0`),
   C1, and option (i)'s `twin_K3` sleep-tick count are amended for a brain that RESTORES its clock: each compares the restoring arm's
   clock against the restored sub-record or the twin, never against 0. The exact condition comes from circadian-p4b round 2's REPORT
   item 4; the build is r2-instr-fix round 4. Then a dev-seed run (15000-15007, heavy-run lock) shows CHRONO and MOTOR fire only when
   their defect is planted. A red that fires on clean code is not a red (enforcer #6948 MAJOR-1; panel #6910 MAJOR-1). Until then
   the restoring arms' chronology reads are OWED.
6. Only then the confirmatory reads, in the registered order and block (P1 15010-15017, P2 15030-15037, P3 15050-15057), once
   each; reserves 15020-15027, 15040-15047, 15060-15067 only per the registered reserve rule.

Data-exposure: no confirmatory or reserve seed of R2's block was read or run in writing this section (search record: R4).

### D2.1 Detection power at S = 8 (D-68)

D-68 accepted the integrator's P1 ruling: R2 stays on the P1 closed list by name, and this bar-neutral amendment does not
re-freeze it. Before any data, this subsection states each verdict's shape and its detection power at S = 8. It changes no
bar, no seed block and no decision rule. Arithmetic only; nothing was run. Cites are to section 2.3 (R2-c1 to R2-c6), 2.4
(reds), 2.10 (R2-c7) and the R2-c3 tightening in A2 (#2796, "every 2.4 red must fire on EVERY seed").

**No measured SD exists.** Every R2 verdict is exact equality or exact difference on a record (2.3: "equality has no
variance"), so there is no paired SD in R2's own dev rows to cite (D2 above). Wherever a count or a test would need an SD,
the reading is: no measured SD: underpowered reading = INCONCLUSIVE (P8 item 4). R2 has no count bar and no statistical test,
so that sentence is stated once and applies to none of the rows below except as noted.

**Shape A, per-seed pass/fail, any failure = FAIL** (the restart equals the twin, on every seed). Power to catch a fault that
shows on a seed with probability p is 1-(1-p)^8:

| p (per-seed manifestation of a real fault) | 0.10 | 0.25 | 0.50 |
|---|---|---|---|
| power at S = 8 | 0.570 | 0.900 | 0.996 |

Power reaches .8 at p >= 0.182 (1-(1-p)^8 >= .8 <=> p >= 1-0.2^(1/8) = 0.182). Below about 18% of seeds the detection power is under .8;
below about 8.3% (1-0.5^(1/8)) a fault is missed more often than not. A pass at S = 8 with no failure bounds the per-seed failure rate only below 0.31
(one-sided 95%, 1-0.05^(1/8); D2).

| verdict | shape | read |
|---|---|---|
| P1 R2-c1 (K0, K1, K2 restoring = twin, every record) | A | power per the table |
| P2 R2-c6 (K3 interrupted bout, restoring = twin from first tick) | A | power per the table |
| P3 R2-c7 bar (hydrated `_buf_x`/`_buf_y`/`_categories` and post-refit model = twin) | A | power per the table |
| I_K1 control (equals its twin on every seed) | A | power per the table, for a harness that wrongly says "different" |

**Shape B, differs on EVERY seed** (a control or red whose firing is itself the pass condition). A pass needs all 8 seeds to
fire, so the pass probability for an arm that fires with per-seed probability q is q^8: 0.430 at q = 0.90, 0.100 at q = 0.75,
0.004 at q = 0.50. The cost of this shape is false NOT MET, not a missed fault. A seed where the arm does not fire is INVALID
(R2-c2: "the construction is wrong, not the mechanism right"; R2-c3 tightening: any INVALID seed makes R2-c3 NOT MET for the run).

| verdict | shape | read |
|---|---|---|
| P1 R2-c2 (I_K2 differs from its twin on every seed) | B | pass probability q^8 as above; a miss is INVALID, not a mechanism null |
| P1 R2-c3 (each 2.4 red fires on every seed, tightened) | B | q^8 per red, for every red in the 2.4 table (O_rng, O_w, O_stack included) |
| P1 R2-c4 replay-OFF twin differs in post-night weights on every seed | B | q^8 as above |
| P3 R2-c7 reds (CROSS, O_buf, DUP, REORDER, HELDOUT, MISSING, WIDTH, VOCAB, INSTRUMENT) | B, if "each must fire" means every seed | 2.10 says "each must fire" without naming a seed count; the tightening names the 2.4 reds. Under an any-seed reading the shape is A-like: power 1-(1-q)^8 to see the red fire at least once. Which reading holds is the integrator's statement, not made here |

**Shape C, deterministic planted check, no seed power.** R2-c4's planted 1-ulp weight change and swapped ring entry run first
on one dev seed and are deterministic: each either moves the comparator or does not. S = 8 power does not apply. R2-c5
(pytest, `tests/test_r2_replay_hydration.py`) is a set of separate boolean checks with no seeds: no power statement applies.

**Where the detection is weak.** For each shape-A verdict, a restart defect that appears on fewer than about 18% of seeds
(for example a defect that needs a rare schedule branch) is detected with power under .8 at S = 8, and below about 8.3%
it is more likely to pass than to fail. R2's seeds are 8 seeded shuffles of
`_CONV_TRAIN` plus the fixed kill points, so the exercised schedules are 8 per block. A pass is therefore a statement about
those 8 schedules and the 2.3 scope clause, and not about a per-seed failure rate below 0.31. No result at S = 8 is reported as
a population rate.

### D1.2 K3 alignment and finding (iii) (2026-10-29, integrator; launcher D-77; pre-data)

R2-c6 registers K3 as "the rest of the night, then 6". A restoring process starts on a fresh clock (`chrono_clean` asserts
`step == 0` and `nights == 0`; `Brain.tick` derives sleep/wake from `_step`, engine/brain.py:3903-3920), so the restart and its twin
would count different ticks to the boundary. Ruled (option (i), the registered observable kept exactly): the twin_K3 arm
measures its remaining sleep-tick count and every K3 restoring arm reads that count (claude/r2-instr-fix-1029, round 2). **The
harness carries circadian phase across the restart; the brain does not (finding iii).** This sentence is owed unprompted in R2's
method AND in its results record. Finding (iii) is a P4b-shaped restart defect of its own: before any seed outside the dev block 15000-15007 (15010-15067, confirmatory and reserve) a red-first
test (xfail strict, raises=AssertionError) that a restart mid-night resumes the night exists; fixing or deferring (iii) is the
launcher's call (D-77). Route A (per-seed K1g arms for R2-c7, launcher D-77) keeps 2.10's "on EVERY seed" with no amendment.

### D1.3 Instrument at claude/r2-instr-fix-1029@b311e78d (2026-10-29, pre-data)

RECORD only. Nothing here sets or changes a bar, seed block or decision rule. Source: that branch's REPORT.md (rounds 1 and 2) and
board #6793, #6854. It is STATIC work: the instrument at b311e78d was built, not run, and none of its tests has been run. Its review
status (D1 item 6) reads: BLOCK-1 built (route A), MAJOR-1..4 and -6 fixed, MAJOR-2 (b)-(d) OWED, MINOR-1 and MINOR-4 OWED, at
claude/r2-instr-fix-1029@b311e78d (re-review owed).

(a) The confirmatory-seed guard. The one `check_seeds` now runs first in every entry point: `run_process` (the `--proc` entry),
`run_failwrite` (before its catch-all, so a refusal is raised and never written as an INVALID arm), `verdict()`, and (already)
`run_seed` and `run_campaign`. A seed outside the dev block 15000-15007 is admitted only when `R2_CONFIRMATORY_GO` names exactly ONE
part and every seed is in that part's block or reserve: `P1` = 15010-15017 or reserve 15020-15027; `P2` = 15030-15037 or reserve
15040-15047; `P3` = 15050-15057 or reserve 15060-15067 (2.6, 2.10). So the go must name part AND block: a bare `=1` licenses
nothing, seeds 15068-15999 are refused, and any mix of blocks (dev plus confirmatory included) is refused. `verdict()` takes the part
from `check_seeds` and reads only that part's lines. The D4 halt (more than 2 INVALID) counts per part.

(b) Route A: P3's roles and arms for R2-c7 (all CPU, 2.10 "CPU device"; 21 processes per seed), each against its 2.10 line.

| role | parent | what it does | 2.10 line it reads through |
|---|---|---|---|
| pre_K1g0 | none | session 1: ticks to a completed night, clean `close()` | Reds, CROSS: a session boundary must exist or CROSS is vacuous; "never joined across a restart" |
| pre_K1g | pre_K1g0 | session 2: ticks until a refit of its own, then `os._exit(0)` right after the night's `_flush_growth` returns | The restart test: "kill K1g at the declared boundary (the tick after `_flush_growth` in the night's persist block, `os._exit(0)`)" |
| twin_K1g | pre_K1g0 | same schedule, never killed; snapshots its buffer at that point, then ONLY `neuro.tick()` to the next refit | The restart test: the uninterrupted twin "run ONLY the grower's sleep step (`neuro.tick()` until the next refit, no new percepts)" |
| res_K1g | pre_K1g | the restore: hydrated buffer, then the same grower step | The restart test bar, on EVERY seed: hydrated `_buf_x`/`_buf_y`/`_categories` and the model after the next refit (W, b, beta, units, trajectory row) equal the twin's |
| K1g_POS | pre_K1g0 | the twin with `_buf_y` rotated by half (shuffled across sessions), then the refit | Positive control: the refit on the shuffled buffer differs from the twin's |
| K1g_CROSS | pre_K1g | restore with the pairing guard (`prev[0] == session and prev[1] + 1 == pseq`) removed by an AST rewrite of `hydrate_transitions`, refused if the guard's syntax changed | Red CROSS: one pair more than the twin's |
| O_buf | pre_K1g | a real restart whose `restore_transitions` is a no-op (hydration neutered) | Red O_buf: "no hydration: the refit differs" |
| K1g_{DUP,REORDER,HELDOUT,MISSING,WIDTH,VOCAB,INSTRUMENT}_{on,off} | pre_K1g | one planted percept event in the copied lake; `_off` neuters that kind's guard in the shared reader AND in `_percept_refusal` | Reds DUP, REORDER, HELDOUT, MISSING (as 2.4), WIDTH (63-dim), VOCAB (`category >= n_units`), INSTRUMENT (provenance `instrument`) |

`c7_seed` reads, per seed: K1g = `res_K1g` equals `twin_K1g`, no fault, `_prev` None, `observed` 0, no transition refusals (2.10 "does NOT
touch `_prev` ... `observed`"); CROSS and O_buf fire when unlatched and unequal; each planted kind reads guard_on_refused, guard_on_ok,
guard_off_flagged; the control's W or beta differs. `PART_SEEDS["P3"]` = 15050-15057 + 15060-15067; `PART_BARS["P3"]` = R2-c7, read on
EVERY counted seed (2.3 Disposition). A P3 seed is INVALID only by its own K1g arms (2.3 "a P3 red reverts P3, never P1"; A6; A11.4); an
INVALID P3 seed reads R2-c7 NOT MET; a K1g red that fires on no seed halts P3. This keeps 2.10's "on EVERY seed" (D1.2). The branch
discloses these readings, each one the instrument's choice on a point 2.10 leaves open, stated here so no later reader mistakes them
for the registration; none moves a bar:
- pre_K1g0 (a cleanly closed session 1) is an addition: without it CROSS has no boundary to cross.
- "Until the next refit" is read literally: the grower ticks until a tick with `ticks % refit_every == 0`. The P3 pytest instead ticks
  until the trajectory grows. Under the literal reading O_buf (empty buffer: `_refit` returns early) reads as a moved trajectory.
- The twin is its own process, not the killed process continuing as in the pytest.
- guard_on_ok for a percept red accepts a visible latch OR an exact restore; which guard (reader or hydration) refuses each kind is a
  dev-run fact, not registered. VOCAB's planted event carries `proto_flushed` = category + 1 so the stale-prototype proof does not
  also refuse it with only the vocab check off.
- Whether O_buf is P3's D4 load-bearing arm (the I_K2 analogue) is OWED (see (c)); today an equal O_buf reads as a red that did not fire.

(c) R2-c3's reds by part (MAJOR-2 (a), RESOLVED from the registration at 155d1319, `_RED_PART`):
- P1: 2.4's table (O_wm, O_hyd, O_rng, O_w, O_stack, O_rw, CHRONO, MOTOR and the planted DUP, REORDER, HELDOUT, MISSING), because 2.3
  Disposition makes R2-c1..c5 P1's and R2-c3 is "every constructed red fires (section 2.4)"; O_rng's lone-replay control goes with O_rng;
  `FW:*` is P1's (A3: FAILWRITE runs "then the P1 seeds of 2.6").
- P2: O_sleepskip and O_cursor (K4), by the 2026-10-06 P2 amendment ("Reds, constructed and red first: `O_sleepskip`, `O_cursor` (K4)
  ..."; "A red reverts P2 only"; 2.3 Disposition: R2-c6 is "P2's own verdict").
- P3: the R2-c7 reds of (b), by 2.3 Disposition and 2.10.
- The halt (a red that fires on no seed) follows the same partition.
Still OWED, no line assigns them: (b) R2-c8 (A4, `sleep_bout`) names no part, and O_bout, SLEEPDIGEST, CTXREAD, BOUTDIGEST and
BOUTPROGRESS have no assigning line, so all of these join BOTH P1 and P2 for now (stricter, never weaker); (c) per-part role subsets
(every seed runs every role, so a P2 read also runs P1's arms and P3's); (d) INVALID for P2 (D4 says "the same rule applies to P2's and
P3's load-bearing arms" but names neither's arm; P2 keeps the I_K2-based rule). Each is an integrator ruling, not made here.

(d) K3 alignment as built (option (i), D1.2). `twin_K3` records the sleep ticks left in its cut night (until `_nights` advances), then S = 6,
and reports `k3_rest`. `run_seed` passes that count to every K3 restoring arm (res_K3, O_sleepskip, SLEEPDIGEST_on, SLEEPDIGEST_off) as
`--proc`'s 8th argument and brings `twin_K3` into any K3 selection; each records `S + k3_rest` sleep ticks and refuses to run without
the count. Disclosed coupling: those four arms are no longer independent processes; a failed `twin_K3` makes all four incomplete
(INVALID, never a fire). The harness carries circadian phase across the restart; the brain does not (finding iii).
Finding (iii)'s red is `tests/test_restart_circadian.py` (b311e78d), `xfail(strict=True, raises=AssertionError, reason="finding (iii):
restart forgets circadian phase (D-77)")`: a dev-seed brain cut after its first sleep tick and restored (throwaway homes) must tick the
same as an uninterrupted twin ("asleep"). The static first failure at f68b2e37: the restore's `_step` is 0 (brain.py:656) and
`light_cycle(period)(0)` is lights-on (circadian_clock.py:65-70), so the first tick reads "awake" and the assert raises AssertionError. A
twin whose night ended after one tick fails through `pytest.fail`, so it can never read as the xfail. It is NOT fixed, and it has not been
run; the expected reading is XFAIL.

(e) Owed before any 15050 seed (read with D3, which is not edited here): the K1g dev-seed run. On dev seeds 15000-15007 only, CPU, under the heavy-run lock after
the campaign: `--seed <dev seed> OUT --roles <P3 roles>`. Every K1g red (CROSS, O_buf, DUP, REORDER, HELDOUT, MISSING, WIDTH, VOCAB,
INSTRUMENT) and the K1g_POS control must fire on the dev seeds; one that fires on none halts P3 (A5's order; P8 item 1). The run also
shows the brain's actual guard causes (MINOR-1 OWED) and whether `observed`/`live_n` at construction are twin-relative (MINOR-4 OWED).
It sits in D3 item 4's run (R2-c7's reds, 2.10); this item adds nothing to the bar, only names the P3 roles and the seed block that run
uses. Also still owed, unchanged: the re-review of the instrument at b311e78d, and the integrator's statement on (c) above.

Data-exposure: no confirmatory or reserve seed of R2's block was read or run in writing this section.

### D1.4 Correction of D1.2's premise and of D1.3(d) (2026-10-29, integrator; enforcer #6947, #6948, #6952; pre-data; append-only)

1. D1.2's cited mechanism is wrong: "`Brain.tick` derives sleep/wake from `_step`". `Brain.tick` takes sleep/wake from the clock's
   `consolidation_gate()`, which is phase >= 0.5 (engine/brain.py:3927-3934 @4b699b33; sleep/circadian_clock.py:149-152); light only
   nudges the phase through the PRC. Option (i)'s ruling still stands on the corrected mechanism: a restoring process builds a fresh
   clock (phase drawn from `Random(seed)`), so it and its twin can count different ticks to the boundary. Only the cited reason changes.
2. D1.3(d)'s expected reading is wrong. It says the restore's `_step` is 0 and `light_cycle(period)(0)` is lights-on, so the first tick
   reads awake and the reading is XFAIL. In fact `Random(15000).random()` = 0.8209, which is night. The enforcer's stdlib simulation of
   `CircadianClock` gives first ticks S S S w, so the twin's expected reading is asleep and the UNFIXED restored brain is also asleep: the
   strict xfail XPASSes, and the run reads FAILED, not XFAIL. The b311e78d red does not discriminate at seed 15000 (enforcer #6935 P-2;
   panel #6910 BLOCK-1). "It is NOT fixed" in (d) holds as of b311e78d only.
3. The replacement red has posted (circadian-p4b round 2, #6998 @1d87efd4): `tests/test_restart_circadian.py::test_a_restored_brain_resumes_its_twins_clock`,
   held out of git at lanes/engine-1024/circadian-p4b-row/r2-instr-fix-red/test_restart_circadian.py with xfail(strict=True,
   raises=AssertionError). It REPLACED b311e78d's file on claude/r2-instr-fix-1029 at 6fd1344e (round 4, item F); git blob
   1370e287c5376ad091853f74540da2944cbb3167, byte-identical to the held copy (git hash-object, the integrator, 2026-10-29). Its
   assertion (lines :128-129 in the pinned blob 1370e287; :122-124 in circadian's copy at 1d87efd4, before the 5-line xfail marker): a brain restored at a MID-DAY cut after the first night has (_step,
   clock phase, sha256 of the rng state) equal to the uninterrupted twin's at construction, and its next 200 tick() sleep/wake
   readings equal the twin's. The clock triple is NOT re-compared after the 200 ticks, so a clock whose phase drifts while the
   200 readings stay equal passes; R2's K3 reads sleep/wake readings and their transition sequence, which the second assert
   covers (enforcer #7054 MINOR-1). D3 5a requires its recorded pytest failure on 5a's named base.
4. D1.3's review status for b311e78d omitted #6898 (enforcer CHALLENGE, 3 MINOR, open at b311e78d): the K1g procedure duplicated in the
   tests (p3t:965-1023 vs rh:1260-1351); K3's count aligned but not its wake/sleep transition sequence; the run_seed table ignored the
   part. So (d)'s "K3 alignment as built" is not settled at b311e78d: #6898 MINOR-2 is open on that point. r2-instr-fix round 3
   (@1e929e2c, #6926) reports MINOR-2 and MINOR-3 fixed and MINOR-1 partly. Its re-read posted (#6987, addendum #6991): MINOR-2 and
   MINOR-3 RESOLVED; MINOR-1 PARTLY; one residual MINOR (run_campaign :2681 runs the base head on P3 seeds) and one more MINOR (the
   kill point still reads 'tick after' at :43 and :243). RULED (integrator, 2026-10-29): the instrument is R2-c7's ONLY reading;
   the duplicated K1g procedure is merged into the instrument's, tree-checked (r2-instr-fix round 4, items B-E), and the dev-seed
   RUNBOOK item 4 stops reading the pytest as R2-c7 (r2-devprep round 4).
5. Disclosed (enforcer #6952 A-1): at 4b699b33 the clock is flushed only after `_flush_growth` in the night's persist block, so R2's
   K1g kill at pre_K1g leaves the previous session's close-time phase on disk. res_K1g then restores a phase up to one day staler than
   its growth. circadian-p4b round 2 moves the clock flush first (its brief, item 7). Until that lands, K1g chronology reads are OWED
   with 5c.
6. Ranges: D2's sentence and D3's heading now read 15010-15067 (enforcer #6948 MINOR-2; the old 15010-15057 left the reserves
   15020-15027, 15040-15047 and 15060-15067 outside the owed list).
7. K3 phase check (integrator ruling #7111, accepted by launcher D-84; enforcer #7103). twin_K3 records its own sleep-onset step;
   each K3 restoring arm's resumed step must EQUAL it (the bound rests on tick order at circadian@1d87efd4: _step += 1, then
   _enter_sleep, whose persist block flushes the clock first). The K1g chronology reads a FOUR-cell table keyed on the code under
   test, read harness-side (ruling #7186, superseding #7142; enforcer #7141, #7183): (pre_ flushed, restore carries) = scored;
   (flushed, dropped) = chrono FAIL; (none flushed, restore carries) = chrono FAIL 'a restored record with no flushed lineage' (an
   isolation defect: a stale or shared home, or the live lake); (none, none) = refused by name. One red per FAIL cell (r2-instr-fix
   round 5 item 6 and round 6).
   'pre_ flushed' means WROTE: a truthy _flush_circadian return (r2-instr-fix round 7 @aa238ad7, enforcer #7204). As built in
   round 8 (@1ab73b72, #7268; enforcer AGREE #7296), two PRECEDENCE rows are read before the four cells, for EVERY restoring arm:
   (a) no pre_ flush wrote AND at least one FAILED (a raised write or a failed keep; the failure counter bumped) -> its own named
   FAIL: CHRONO_FLUSH_FAILED for every restoring arm (K1G_CHRONO_FLUSH_FAILED for K1g; K3_phase False with k3_phase_flush_failed
   for K3), never 'the restore dropped it'; (b) no pre_ flush wrote and every False was persist-off or an unchanged record ->
   REFUSED by name (K1G_CHRONO_UNWRITTEN / K3_PHASE_UNWRITTEN), neither pass nor fail. The OWED line of enforcer #7255 MINOR-1 is
   CLOSED by round 8 (enforcer #7297). chrono_clean's step term compares with the step the parent pre_ process
   last flushed (harness-side). Until claude/circadian-p4b-1029 lands, K3's alignment holds BY CONSTRUCTION because the harness
   carries the phase (option (i), D-77 (i)), and K3_phase, keyed on the same harness-side bit, is refused by name as
   not-applicable (never NOT MET; #7183 NIT). It becomes a real cross-check of
   the brain only after the D-79 fix lands (launcher D-84).
