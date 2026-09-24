# G0(a) / T108 part (a) -- an external correction path into the chat intent learner, with provenance

Row: plan Part D "G workstream", G0 (a); red G-R0's correction clauses (Part F); FR-041, SC-019. T108 lands its three
parts as three arcs: (a) this file, (b) the reply consuming the learned route, (c) the learner's persisted state.
Status: RESULTS RECORDED (2026-09-24). Section 2 is the frozen pre-registration; section 3 records the runs and every
deviation from section 2.

## 1. Territory, read at HEAD `09f4028` before the design

1. The chat path. Cockpit `POST /api/chat {"text"}` queues a ticket (`substrate/cockpit.py::Turns.submit`); the tick
   thread's `Turns.pump` calls `brain.respond(text, human=True)`; `cortex/brain.py::Brain.respond` calls
   `self.brain.chat(message, human=human)` with no `feedback`; `cortex/live_brain.py::LiveBrain.chat` replies via
   `converse` and then calls `self._ixn.learn_turn(text, _reply_to_intent(reply), feedback=feedback)`.
2. `cortex/interaction_learning.py::SpikingInteractionLearner.learn_turn` uses `feedback` when given, else the
   firewall's label, for ONE `interact` deposit. Nothing records which it was.
3. `interact` appends `(stimulus, label)` to the sleep-replay ring (`_REPLAY_CAP` = 64) with no provenance, and the
   sleep tick calls `self.brain._ixn.replay(k=8)` on every sleep tick.
4. No external correction exists: `do_POST` serves `/api/act` and `/api/chat {"text"}` only, and no caller in `src/`
   passes `feedback` to `LiveBrain.chat` (the self-checks and tests do).
5. `LiveBrain.converse` has a stop-word branch ("stop", "halt", "freeze", "abort", "estop") that acts whatever
   `human` says. A correction must therefore never pass through `converse`.
6. `make_brain` takes keyword arguments only.
7. Measured on scratch brains (seed 0, `throwaway_home()`, `persist=False`): construction 11.4 s; one
   `Brain.respond` turn 0.77 s; one wake+sleep cycle = 25 ticks (12 wake, 13 sleep) in about 1.8 s; each night
   replays about 13 x 8 = 104 recorded turns.
8. The corpus: `_CONV_TRAIN` (5 intents x 4 keyword-bearing lines) and `_CONV_TEST` (5 intents x 4 novel
   paraphrases) in `cortex/interaction_learning.py`. A design-time generator check (seeds 0-2, `converse`, which
   does not learn) labelled all 20 test lines GREET, identically on every seed. So the 4 GREET lines are correct
   and the other 16 are BLIND. Frozen here: CORRECTED = the first two lines of GOAL, MEMORY, CONSCIOUS and ARM
   (8 lines); HELD-OUT = the last two lines of every intent (10 lines: 8 blind, 2 GREET).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The mechanism

- **Provenance in the learner.** `learn_turn(utterance, firewall_intent, feedback)` treats a given `feedback` as
  AUTHORITATIVE for that utterance (normalized: lower-cased, whitespace-collapsed) and records it in a bounded map (a
  memory bound, disclosed under P1). Two parts, each with a neuter switch used only by the instrument and tests:
  - SUPERSESSION: a later self-labelled turn on the same utterance uses the authoritative label, not the firewall's;
  - RING REWRITE: the replay ring's older entries for the same utterance take the authoritative label (ring entries
    carry the utterance key).
- **The channel.** `LiveBrain.correct(text, intent)` validates `intent` against `_IXN_INTENTS` and calls
  `learn_turn(text, None, feedback=intent)`: no `converse`, no reply, no actuator path. `Brain.correct(message,
  intent)` is the canonical entry (P4). The cockpit keeps each ticket's text (a bounded map) and accepts
  `POST /api/chat {"correct": <ticket id>, "intent": <label>}`; the tick thread's pump applies it through
  `Brain.correct`, never through `respond`. The page gets one control per reply: pick the intent meant, and send.
- **Envelope and safety.** No `brainstem/` file changes. A correction teaches intent RECOGNITION only.

### 2.2 The instrument and the arms (one process per seed and arm, seeds 0-7, scratch brains only)

The instrument is a set of `g0_*` functions in `cortex/interaction_learning.py` (the t082 pattern). Every result
records `head` and `dirty`; a run on a dirty tree is not a result. Per run:
1. `make_brain(seed=seed, persist=False)` inside `throwaway_home()`.
2. Warm-up: every `_CONV_TRAIN` line through `Brain.respond(u, human=False)`, 2 passes, order shuffled by
   `random.Random(seed)`.
3. Firewall labels for every `_CONV_TEST` line via `_reply_to_intent(converse(u, human=False))`. A run whose labels
   differ from section 1's frozen ones is INVALID and reported.
4. PRE-SEED: each CORRECTED line once through `Brain.respond(u, human=False)`, so the ring holds a conflicting
   self-label for it before any correction. Routing snapshot R0.
5. Correction, by arm, once per CORRECTED line with its true intent:
   - **A (the method):** `Brain.correct(u, true)`, both parts on;
   - **B0:** `Brain.correct` with both parts neutered (a one-shot correction, today's learner behaviour);
   - **B1:** `Brain.correct` with SUPERSESSION neutered;
   - **B2:** `Brain.correct` with RING REWRITE neutered;
   - **C (the incumbent):** no correction.
   Routing snapshot R1.
6. Challenge, 3 rounds: every CORRECTED line through `Brain.respond(u, human=False)` (self-labelled), then 25 ticks
   (one day cycle with its sleep replay). Routing snapshots R2-R4. Each night's replay count must be above zero,
   or the run is INVALID.
7. A spy in the instrument's own process logs every teaching event on a CORRECTED utterance, wake or replay:
   (utterance, label used, phase, before or after its correction). It changes no behaviour.
8. Read out with `LiveBrain.route` (no plasticity): M1 = the fraction of CORRECTED lines routed to their true intent
   at R4; M2 = HELD-OUT accuracy; M3 = `_CONV_TRAIN` accuracy.

`human=False` throughout, so no arm intent ever actuates (T037a). Cost: about 110 s per run; 40 runs in 4 worker
processes, about 20 minutes.

### 2.3 The statistic and the verdict rules (written first)

- **R-a1, "a corrected misrouted prompt does not improve later routing" (G-R0):** per seed d = M1(A) - M1(C). PASS
  needs Part F's paired one-sided t > 1.895 (7 d.f., alpha = 0.05), ties counted, AND an exact one-sided sign-flip
  p <= 0.05 over the 256 flips. The sign-flip p assumes each seed's d is symmetric about zero under the null. If
  every d is equal (t undefined), all d > 0 reads PASS and all d <= 0 reads RED, and the file says so. Otherwise
  RED.
- **R-a2, "a self-label overwrites an authoritative correction" (G-R0), read on LABELS, not on routing:** in arm A,
  any teaching event on a CORRECTED utterance after its correction whose label is not the authoritative one, or any
  end-of-run ring entry for a CORRECTED utterance with another label, is RED. There is no tolerance.
- **R-a3, the safety boundary (a unit test, in pytest):** a correction whose ticket text is a command ("stop", and
  an arm command) is dispatched without any call to `Brain.respond`, `LiveBrain.chat`, `LiveBrain.converse`, the
  NL firewall's command path, `So101Arm.apply` or the E-STOP latch. Any call is RED.
- **Secondary, reported and not barred:** M2, M3, the R0-R4 trajectories, per-seed values with mean and sd, the
  invalid seeds. A drop in M3 is disclosed with its size.
- **Scope of inference:** this corpus, these 8 seeds and this schedule only.

### 2.4 The reds are reached by constructed faults, run first

The mechanism, its two neuter switches and the instrument land together. Before arm A is read, the neutered arms
must show their parts' reds:
- B1 must log, on at least one seed, a wake teaching event after the correction carrying the self-label
  (supersession's red);
- B2 must log, on at least one seed, a replay teaching event after the correction carrying a pre-seeded self-label
  (ring rewrite's red).

A part whose neutered arm shows no red has no reachable red. It is deleted, not kept (P8: the criterion is deleted,
not weakened), and the file says so. R-a1's scorer is reddened by a mutated input (C against itself, d = 0) in a
unit test.

### 2.5 Controls and disclosures

- C is the incumbent and B0-B2 are neutered twins; none is the method.
- Training dose: the correction's own ring entry is replayed in every correction arm, the same dose in A and B0-B2.
- The corpus is the module's own, written before this arc; this file freezes the splits and their labels.
- A register row for the provenance rule lands in `docs/INVENTION_REGISTER.md` with the mechanism (FR-031).
- The design was read by Codex before this freeze; its points (the neutered arms, the pre-seed, label-level R-a2,
  R-a3, the sign-flip p, the territory corrections) are folded above.

## 3. Results (re-established at `38d9db7` after the panel fold; first run at `b6cd92c`; identical numbers)

The command: `python -m snn_data_lake.cortex.interaction_learning --g0-campaign <out-dir>` (out-dir outside the
repo). Both runs: a clean tree, one common head, 40 of 40 runs valid, about 6-7 minutes. A second instrument
recomputed M1 from the raw R4 routes, the teaching labels, the ring audits and R-a1's statistics without
`g0_verdict`, and agreed with it on every number; the re-run at `38d9db7` reproduced the first run exactly.

**Deviations from section 2, folded from the G0(a) panel (`38d9db7`) and disclosed here.**
- The authority map has NO bound: section 2.1's "bounded map" let a self-label override a correction after 4096
  others (a behaviour-arbitrating eviction, P1). The map lives for the process; G0(c) persists it.
- The scorer requires all 8 registered seeds valid in every arm and one common head. The first scorer allowed 6
  seeds, against section 2.3's "7 d.f." and "256 flips"; no run was invalid, so no reported number changed.
- The spy counts and logs a teaching call only after it completes; a raising call makes the run invalid.
- `LiveBrain.teach` now goes through `learn_turn(feedback=...)`, keyed and authoritative like `Brain.correct`.
- Part F's permutation false-positive rate is now printed (below), beside the observed sign-flip p.

**2.4 first -- the reds are reachable.** B1 (supersession neutered) logged 192 post-correction WAKE teaching events
carrying the firewall's self-label. B2 (ring rewrite neutered) logged 295 post-correction REPLAY events carrying a
pre-seeded self-label. Both parts' reds fire, so both parts stay.

| arm | M1 per seed (8) | M1 mean | R4 routed back to GREET | post-correction wrong labels (wake / replay) | ring mismatches | M2 | M3 |
|---|---|---|---|---|---|---|---|
| A | 1.0 on every seed | 1.000 | 0 | 0 / 0 | 0 | 0.875 | 0.969 |
| B0 | .25 .5 .625 .375 .625 .625 .75 .5 | 0.531 | 22 | 192 / 897 | 256 | 0.525 | 0.887 |
| B1 | .625 .75 .75 .375 .75 .875 .875 1.0 | 0.750 | 13 | 192 / 602 | 192 | 0.600 | 0.900 |
| B2 | 1.0 on every seed | 1.000 | 0 | 0 / 295 | 64 | 0.888 | 0.969 |
| C | .375 .5 .5 .5 .375 0 .625 .625 | 0.438 | 26 | -- (no correction) | 256 | 0.512 | 0.894 |

Every run's minimum replay count per round was 96 (never zero). R4 = routing after the third challenge round.

- **R-a1: PASS.** d = M1(A) - M1(C) per seed = 0.625, 0.5, 0.5, 0.5, 0.625, 1.0, 0.375, 0.375; mean 0.562, sd
  0.200, t = 7.94 (bar 1.895), exact one-sided sign-flip p = 1/256 = 0.0039. Part F's permutation false-positive
  rate, conditional on the observed magnitudes (the complete R-a1 rule re-run on every one of the 256 sign flips of
  the observed d): 9/256 = 0.035.
- **R-a2: PASS.** Arm A: no post-correction teaching event with a non-authoritative label, wake or replay, and no
  end-of-run ring entry for a corrected utterance with another label, on any seed.
- **R-a3: PASS** in pytest (`tests/test_g0_correction.py`): a correction on "stop" and on an arm command reaches no
  `respond`, `chat`, `converse`, `So101Arm.apply` or E-STOP latch.
- **Secondary.** M2 (held-out paraphrases) 0.512 -> 0.875, A vs C: corrections generalize. M3 (the training lines)
  0.894 -> 0.969, with no seed dropping.

**Attribution, stated plainly.** Supersession carries the routing effect: without it (B1), 13 corrected routes fall
back to GREET; a one-shot correction (B0) loses 22 of 64. The ring rewrite protects the teaching RECORD, not the
routing on this schedule: without it (B2), 295 replays re-teach the stale self-label, yet routing stays at 1.0 on every
seed, because supersession and the correction's own dose outweigh them. It stays under 2.4's rule (its red is reachable),
and its routing effect here is zero.

**Disposition.** G0(a) meets its pre-registered criteria; G-R0's correction clauses read PASS on this corpus, these 8
seeds and this schedule only. G-R0's routing-consumption and fresh-process clauses belong to G0(b) and G0(c). The
executor's rewrite of the module's own self-check was not taken (`4474ada` guards the bare run instead).
