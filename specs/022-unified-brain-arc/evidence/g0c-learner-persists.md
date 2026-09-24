# G0(c) / T108 part (c) -- the chat learner survives a restart: its continuation state and its authoritative teaching

Row: plan Part D "G workstream", G0 (c); red G-R0's persistence clause, "a fresh process does not reproduce taught
routing and the next learning/replay step" (Part F); FR-041, SC-019; P4b. Parts (a) and (b) are
`g0a-chat-correction.md` and `g0b-reply-consumes-route.md`. This arc is the interaction learner's slice of R2/T103
(durable, restart-safe replay); section 2.6 states the boundary. Status: RESULTS RECORDED (2026-09-24). Section 2 is frozen at the
commit that landed this file; section 3 records the runs and every deviation from section 2.

## 1. Territory, read at HEAD `a36cd34` before the design

1. The learner is not persisted. `cortex/interaction_learning.py::SpikingInteractionLearner` has no serialization,
   and neither `LiveBrain` nor `Brain` saves or restores it. Measured on a scratch home at seed 100 with
   `persist=True`: after G0(a)'s arm-A schedule and `close()`, a fresh process on the same home routes only 1 of the
   20 `_CONV_TEST` lines as the taught brain did, its weight digest differs (`631829bf` against `607bb5d0`), and its
   RNG is back at its seed state. So the incumbent reaches this row's red on its own.
2. The carriers. The learner holds one `basal_ganglia/r_stdp.py::RSTDPNeuron` per intent, an instance
   `random.Random` (replay sampling), the replay ring (`_replay`: stimulus, label, utterance key) and the authority
   map (`_authority`). `Brain.__init__` attaches a `plasticity/plasticity_rule_stack.py::PlasticityRuleStack`.
   - The schedule changed eleven stack fields on all nine neurons: `_short._syn`, `_short._last`, `_homeo._cell`,
     `_homeo._target`, `_intr._ip`, `_meta._bcm` (built lazily), `_ei._ctrl`, `_ei._t`, `_cons._cas` (built
     lazily), `_cons._prev` and `_cons.calls`. None of these classes has a `state_dict` today.
   - A walk of the whole tree after the schedule found 17 classes (364 `BennaFusiSynapse` objects) and about 12,000
     floats. The non-JSON kinds: each `plasticity/inhibitory_stdp.py::InhibitorySTDP` holds a recording deque, and
     its `plasticity/rule_composer.py::RuleComposer` holds rule specifications and a name set (built by its
     constructor) and a thread lock. The stack's `_pre`/`_post` lists share their objects with
     `_short`/`_homeo`/`_intr`.
   - Encoded as JSON the tree is about 155 KiB and takes 5 ms. The live daemon already appends about 160 KB of 0x07
     per tick (`substrate/lake_engine.py`'s measured rate).
   - Not every carrier reaches an observable of the next step: `_IntrinsicPlasticity.stabilize` changes the
     stabilizer's rate, which `HomeostaticPlasticityStack.stabilize` discards and `_drive` never reads.
3. The incumbent persistence path. `substrate/home.py::save_state` / `load_state` store one JSON named state in the
   ONE lake (`_T_STATE`, 0x07; IMMORTAL, with superseded frames pruned; an unchanged state writes nothing). The
   night's persist block in `Brain._enter_sleep` saves `reasoning_curriculum.json` and calls `_flush_growth`,
   `_flush_prototypes` and `_flush_coverage`; `Brain.close` calls the same three flushes (not the curriculum save).
   `Brain.__init__` restores named states when it persists. `Brain.close` restores the previous append observer
   BEFORE its flushes, so a save made in `close` is not seen by the brain's append budget.
   `Brain.tick` runs `_enter_sleep` and then `_sleep_tick` in the same call, and `_enter_wake` then `_wake_tick`.
   Measured at seed 100 over 40 ticks: the learner changed on every sleep tick (its replay) and on no wake tick. So
   the end of a night (`_enter_wake`) is an exact boundary for the learner; the start of one is not.
4. Episodes. `home.py::save_episode` appends a role=filler slot map as a `_T_EPISODE` (0x04) record, deduplicated on
   its packed bytes, and `load_episodes` returns every one. Its readers do not filter by kind: `Brain`'s episode
   refill takes the most recent `capacity`, durable recall matches any episode on its cue slots, and
   `hippocampus/lake_recall.py::LakeRecall.refresh` indexes every episode.
5. `SpikingInteractionLearner.replay` passes each ring entry's stored label straight to `interact`; it does not
   consult the authority map. The ring's labels are therefore authoritative only because G0(a)'s rewrite changed
   them at correction time.
6. R2/T103 (planned, unbuilt) will hydrate bounded replay from typed lake events, persisting stable event identity,
   stream/session identity, sequence, schema/encoder version, a source reference, the target, provenance and a
   validity mask, and the "learner continuation state at the declared boundary". Its red R-R2 kills and restarts a
   scratch brain and requires the next replay step to match the uninterrupted twin.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The mechanism

- **The continuation checkpoint.** `SpikingInteractionLearner.state_dict()` / `load_state_dict()` capture and
  restore the learner's whole object tree: neurons, RNG, replay ring and the attached stack with every sub-object.
  One instance-state codec does this for the whole tree rather than a bespoke `state_dict` per class.
  - It records each object's class name and attribute dict, recursing into lists, tuples, dicts and the tree's
    objects. `random.Random` is stored as its state; tuples, sets and deques are tagged; every float is exact.
  - DECLARED as never saved, and kept from the freshly constructed object: thread locks; a `RuleComposer`'s rule
    specifications and name set (built by its constructor); the authority map (its own source, next bullet); and the
    teaching sink (runtime wiring the brain re-attaches). Any other value it cannot encode is a refusal at save
    time, never a silent skip.
  - Restore MERGES the saved attributes into the freshly constructed tree, recursing into existing objects in place,
    so shared references stay shared. An object the fresh tree lacks (a lazily built `BCMNeuron` or cascade) is
    created without its constructor and given the saved attributes. A saved tree whose class names or attribute sets
    differ from the running code is refused visibly and the learner stays fresh.
- **Where it is saved.** `Brain` saves the checkpoint as the named state `interaction_learner.json` through
  `home.save_state` (about 155 KiB, only when changed) at the END of each night, in `_enter_wake`, after the night's
  last replay, AND in `close`. It restores it in `__init__` after the stack is attached, only when the brain
  persists. A `persist=False` brain
  writes nothing and restores nothing.
- **Authoritative teaching is an event in the lake.** Every authoritative label (`learn_turn(..., feedback=...)`,
  reached by `Brain.correct` and `LiveBrain.teach`) is appended at once as a `_T_EPISODE` record through a sink the
  persisting `Brain` attaches to the learner. Its slots carry R2's event fields now:
  - kind "teach", a stable event id, stream "chat", the session (this brain's process start), a sequence number;
  - the source reference (the utterance key), the target (the intent), provenance "authoritative";
  - the encoder ("text_features n_in=52"), the schema version, and a validity mask.
- **The authority is rebuilt, and the ring reconciled.** At construction the persisting brain rebuilds the
  authority map from the teaching events (latest per utterance wins, as in G0(a)); the checkpoint carries no
  authority. It then applies G0(a)'s ring rewrite to the restored ring: every entry whose key has an authoritative
  label takes that label. So a correction made after the last checkpoint survives a crash in both the authority
  and the replay ring.
- **Teaching events are not autobiographical memories.** `load_episodes` excludes kind "teach" by default, and the
  authority rebuild asks for it by name. The consolidator refill, durable recall and `LakeRecall` therefore see
  exactly the episodes they saw before.
- **Envelope and safety.** No `brainstem/` file changes. Nothing adds an actuator route or reads outside the brain's
  own home.

### 2.2 The instrument (seeds 0-7; every home a fresh temporary `NORD_HOME`)

`g0c_*` functions in `cortex/interaction_learning.py`. Per seed, every process builds `make_brain(seed=seed,
persist=True)` on its own copy of the relevant home and runs G0(a)'s arm-A schedule where stated. The NEXT STEP is
`learn_turn(<first CORRECTED line>, "GREET")` (a self-label on a corrected utterance, so supersession needs the
authority) then `replay(k=8)`.

The persisting processes, one per production hook:
- **P1 (close):** the schedule, then `close()`. The checkpoint is written only by the production `close`.
- **P1n (crash after a night):** the schedule, then ticks until the tick that runs `_enter_wake` after a completed
  night, then `os._exit(0)` without `close`. The checkpoint is written only by the production end-of-night hook.
- **P1x (a correction after the checkpoint):** as P1n, then `Brain.correct` re-corrects the SECOND corrected line
  to a different intent, then `os._exit(0)`. The checkpoint predates the re-correction; only its teaching event is in
  the lake. A run whose checkpointed ring holds no entry for that utterance cannot test the reconciliation and is
  INVALID.

The uninterrupted twins, which never restart and never persist:
- **T:** the schedule, then the next step. **Tn:** the schedule, the same ticks as P1n, then the next step.
  (P1x has no twin: a crash after the re-correction loses that correction's deposit by design, since the
  checkpoint is the boundary, so an uninterrupted process is not the reference. R-c1x states what must hold.)

The restoring processes, each on a fresh copy of its P1 home, by arm:
- **A (close) on P1's home, A_night on P1n's, A_x on P1x's:** full restore, then the next step.
- **O_rec on P1x's home:** full restore except the ring reconciliation.
- **N on P1's home:** no restore at all.
- **One omission arm per carrier on P1's home:** O_w (neuron weights), O_stack (the stack), O_rng (the RNG),
  O_ring (the replay ring), O_auth (the authority rebuild). The instrument removes that carrier in the restoring
  process only; the brain has no switch.

Recorded in every restoring process and its twin, read off the LIVE objects and never through the codec:
- **Before the next step:** the whole learner tree, dumped by the instrument's own walker (a second implementation,
  not the codec), with locks, rule specifications and the sink excluded; and the routes over the 20 `_CONV_TEST`
  lines.
- **After the next step:** every neuron's weights, the utterance keys and labels the replay drew, and the routes.

Cost: about 20 s per process and 15 processes per seed; 120 processes in 4 workers, about 12 minutes.

### 2.3 The verdict rules (written first)

- **R-c1, the restart reproduces the learner (G-R0's persistence clause).** In arms A and A_night, on EVERY seed,
  the restoring process equals its twin (T, Tn) on all of these, with no tolerance:
  - the dumped tree before the next step;
  - the routes before the step;
  - the weights, the replay's drawn keys and labels, and the routes after the step.
- **R-c1x, a correction after the checkpoint survives a crash.** In arm A_x, on EVERY seed: the rebuilt authority
  holds the re-correction's intent for that utterance; every ring entry for it carries that intent; and every entry
  the next step's replay draws for it carries that intent.
- **R-c2, the reds.** N must fail R-c1 on every seed. O_rec must fail R-c1x on at least one seed.
- **Per-carrier attribution (reported; no pass condition).** For each omission arm, which of R-c1's parts fail and
  on how many seeds. An omission that fails only the tree comparison (as `_intr._ip` is expected to) is reported as a
  carrier the next step does not exercise.
- **R-c3, the lake is the source of the authority (pytest).** A restart on a home with the teaching events but no
  checkpoint still supersedes a self-label on a corrected utterance. A restart on a home with a checkpoint but no
  teaching events does not.
- **R-c4, the boundary (pytest).** Each is a separate check:
  1. a `persist=False` brain writes no checkpoint and no teaching event;
  2. a codec refusal leaves the learner fresh and says so;
  3. `load_episodes()` returns exactly the autobiographical episodes with teaching events present, so the
     consolidator refill, durable recall and `LakeRecall` are unchanged;
  4. the codec refuses an unknown value rather than skipping it.
- **Reported, not barred:** the checkpoint's size and save time; the append-budget flood count in each P1 home,
  with the caveat that a save in `close` is outside the brain's budget (section 1 item 3).
- **Disposition:** R-c1 to R-c4 all PASS -> the persistence ships on. Any RED deletes the change in a stacked commit,
  and the row records the null.
- **Scope:** this corpus, these seeds, this schedule, this next step, and a same-code restart. A restart across a code
  change that alters the tree's classes is refused by design.

### 2.4 Controls and disclosures

- N is the incumbent and the omission arms are constructed faults; none is the method.
- The tree comparison uses the instrument's own walker and the observables are read off live objects, so a carrier
  the codec forgot cannot make a restoring process and its twin agree.
- The twins are separate processes; they rely on the determinism of a seeded schedule, which G0(a) and G0(b)
  measured byte for byte.
- A register row lands with the mechanism (FR-031).

### 2.5 Draft reads folded before this freeze

Codex (gpt-6-astra) read the first draft. Its findings, all folded:
- the production hooks are proven separately (P1, P1n), instead of an explicit flush that could mask a missing save;
- R-c2 has one verdict, and the omission arms are attribution only;
- the tree comparison catches carriers no observable reads (`_intr._ip`), and the sink is excluded from the codec;
- the ring is reconciled with the rebuilt authority, with its own clause (R-c1x) and red (O_rec);
- teaching events are kept out of the autobiographical readers;
- the R2 event fields are defined now;
- section 1's two errors are corrected (the curriculum save is in `_enter_sleep` only; the deque is
  `InhibitorySTDP`'s).

A second, narrower read of the reworked draft found one blocking defect, folded: a checkpoint saved in
`_enter_sleep` precedes that tick's `_sleep_tick` replay, so a restore could never equal a twin that ticked on.
The checkpoint moved to `_enter_wake` (section 1 item 3 measures why that boundary is exact). The same read found
no unreachable red and confirmed that `BinLog.append` flushes and fsyncs before `os._exit`.

### 2.6 The R2/T103 boundary

This arc's checkpoint is the "learner continuation state at the declared boundary" R2 names, for this one learner,
and its teaching events carry R2's event fields. R2 remains open for:
- hydrating replay from typed lake events beyond the in-memory ring;
- event identity for self-labelled turns;
- a watermark or cursor;
- the other learners' replay.

## 3. Results (2026-09-24, at `cb64431`)

The command: `python -m snn_data_lake.cortex.interaction_learning --g0c-campaign <out-dir>` (out-dir outside the
repo). 8 seeds x 15 roles = 120 processes, every one valid, on a clean tree and one head. A second instrument,
written from section 2.3 rather than the module's code, recomputed every comparison and rule from the raw per-role
records and agreed with `g0c_verdict`. Both are archived outside the repository under the memory store,
`memory/archive/g0c-2026-09-24/` (`campaign-1-cb64431/`, `recheck_c.py`). The live lake was untouched
(3,318,586,733 bytes, unchanged since 01:28:37 CDT).

**Verdict: PASS.**

- **R-c1: PASS.** A equals its twin T, and A_night equals Tn, on all five records on every seed: the dumped tree
  before the next step, the routes before it, and after it the weights, the replay's draws and the routes.
- **R-c1x: PASS.** In A_x on every seed the rebuilt authority holds the re-correction's intent, all five checkpointed
  ring entries for that utterance carry it, and every replay draw for it carries it.
- **R-c2: PASS.** N fails R-c1 on every seed, on all five records. O_rec fails R-c1x on every seed: its ring keeps the
  stale intent on every seed (the authority is right), and its replay drew that stale entry on some seeds.
- **R-c3 and R-c4: PASS.** The pytest file (`tests/test_g0c_learner_persists.py`) passes at `cb64431`.

**Per-carrier attribution (reported).** Every omission arm failed on every seed, each on the records its carrier
drives:

| arm | differ from T on every seed | differ on some seeds only |
|---|---|---|
| O_w (neuron weights) | tree, routes before, weights after, routes after | none |
| O_stack (the stack) | tree, weights after | routes after (routing before a deposit is unchanged) |
| O_rng (the RNG) | tree, weights after, draws | routes after |
| O_ring (the replay ring) | tree, weights after, draws | routes after |
| O_auth (the authority rebuild) | tree, weights after, routes after | none |

No carrier was unexercised by this next step, and the tree comparison added coverage for any state no observable
reads.

**Reported, not barred.**
- **The checkpoint.** 172,802 to 234,353 bytes by seed (its size grows with the lazily built cascades); the last save
  took 7 to 20 ms. P1 saved exactly once, through `close` (its night saves were suppressed so that only the close hook
  could produce its checkpoint). P1n and P1x saved four times each, the last at the end-of-night boundary.
- **The append budget.** The brain's flood detector recorded 2 or 3 floods in every P1n and P1x run, against 1 in P1,
  whose night saves were suppressed. So a night's checkpoint write trips the 0x07 flood detector on some nights. This
  is a measured cost, carried as an opening (the write cadence, or the 0x07 baseline) for its own arc. A save in
  `close` is outside the budget (section 1 item 3).
- **The crash path.** P1n needed 2 to 24 ticks after the schedule to complete a night; P1x's checkpointed ring held 5
  entries for the re-corrected utterance on every seed, so R-c1x was reachable everywhere.

**The panel and its fold.** Three Codex (gpt-6-astra) lenses read the arc at `e45ba7b`. The instrument lens found
none. Two MAJOR findings were reproduced from the code and folded in `4146f18`, each with a test that fails without it:
- this section's "any restore failure leaves a working brain" was not yet true: the checkpoint READ sat outside the
  restore handler, so a failing lake read escaped construction. It is inside the handler now.
- a frozen build (no source) refused every trained checkpoint, because lazily built objects were validated by
  parsing their constructors' source at restore. The two classes now carry an explicit attribute schema, pinned to
  the constructors by a test, and a no-source round trip restores exactly.
The campaign re-ran at `4146f18` (`campaign-2-4146f18/`): PASS, all 120 processes valid, and every recorded value of
every role identical to the first run. That fold is code, read by the closing report, not re-panelled. The
integration lens's MINOR (the plan, tasks and cold-start doc still called part (c) open) is folded in the same
commit as this paragraph.

**Deviations from section 2.** None in the protocol. Implementation points, disclosed:
- P1's night saves were suppressed by the instrument, so that its checkpoint could only come from `close`, which is
  what section 2.2 requires P1 to prove.
- Four orchestrator fixes landed with the mechanism (`cb64431`), each with a test that fails without it: any restore
  failure leaves a working brain; the teaching session is a per-lake counter, not the wall clock; a lazy class with no
  source is a visible refusal; a failed teaching-event write does not break the correction.
