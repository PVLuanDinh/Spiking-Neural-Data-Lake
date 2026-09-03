# Rank 2a: the persisted grower fed by the brain's own percept stream (2026-09-02)

Audit §8 rank 2, first commit of three: *"Replace the synthetic `_mode_features` input with the brain's
own percept/episode stream (`tick(features)`)."* Owner directive: "Do rank 2."

## What was wired

- `cortex/neurogenesis_engine.NeurogenesisSelfImprove.observe(features, category)` — called on every
  WAKE tick from `Brain._wake_tick` with the arm loop's own percept, `res["percept"]`: the
  attention-gained context vector its `GrowingLearner` categorized (the unified real vision/audio
  channels, the arm's real proprioception, the SAFE distance → `encode_multimodal` → attention gain →
  [0,1]^64), and the brain's own category of it (`res["unit"]`, that learner's prototype id; the brain
  holds three `GrowingLearner` instances and this is the arm's, whose categories drive action). The
  engine records (percept_{t−1} → category_t): from the previous percept, predict the brain's NEXT
  perceptual category, in the brain's own vocabulary. The reflex path returns `unit=None` and feeds
  nothing. Refusals are counted on `state()["neurogenesis"]["feed_refusals"]`, never swallowed.
  (The first draft fed the RAW pre-attention pattern; the panel measured it differing from the
  categorized vector by up to 0.5 on every tick. `AutonomousArm.step` now returns the vector it
  categorized.)
- `tick()` is now the SLEEP step only: refit every `refit_every` sleep ticks on the buffered stream. The
  engine synthesizes nothing; an engine shown no percept refits nothing, grows nothing and reports
  accuracy 0.0 (its self-check's claim 0).
- The 5-mode Markov chain the engine used to synthesize for itself on every tick is RETIRED from the live
  path (P2: exactly one stimulus survives — the brain's) and survives only as
  `SyntheticPerceptStream`, the module's self-check stimulus and this A/B's positive control, driven
  through the same `observe()`/`tick()` the brain uses.
- `state()["neurogenesis"]` reports, beside the top-level `predict_accuracy`, the two floors it must
  clear on the slice it was measured on: `baseline_majority` (answer the training majority every time)
  and `baseline_persistence` (answer "the same as the last category"); `None` until a refit measured
  them, and they ride in the growth blob with the fit they belong to, so a restored accuracy never sits
  beside a fabricated floor. The cockpit renders both under its `predict acc` row and
  `brain_telemetry.PROVENANCE` tags `predict_accuracy` with the instruction to read it against them. On a
  bursty stream persistence is the honest floor.

## Protocol v1 — PRE-REGISTERED before the first run (this section is written first; results follow)

Paired by seed, S = 8 seeds (0–7), each arm `make_brain(seed, persist=False)` under `throwaway_home()`
(no lake, no daemon), N = 800 brain ticks (the clock alternates ≈12 wake / ≈13 sleep ticks, so ≈32
cycles, ≈400 sleep ticks → 20 refits; each refit adds ≈19 rows against a 60-row held-out slice, so
consecutive refits' held-out slices overlap ≈70% and the last-third mean is a smoothed estimate, not
six independent measurements — the paired-over-seeds t is unaffected). The engine's growth gate rides
as wired.

Arms (the wire is neutered at the feed, nothing else changes):

| arm | what `observe` receives | what it establishes |
|---|---|---|
| LIVE | the brain's percept + its own category, as wired | the measurement |
| NEUTER-PERCEPT | a constant vector + the real category | whether the PERCEPT carries information beyond the category sequence's own inertia |
| SHUFFLE-CATEGORY | the real percept + a category drawn uniformly (seeded) from the categories seen so far | the structure-free null: destroys temporal structure, keeps the vocabulary |
| SYNTH (positive control) | `SyntheticPerceptStream(seed)` through the same feed at the same cadence, brain otherwise untouched | the instrument's sensitivity: the chain the engine used to synthesize (expected ≈ 0.7 vs chance 0.2) |

Attribution arms (P8 clause 3, which band of the percept carries what):

| arm | what `observe` receives |
|---|---|
| NEUTER-PROPRIO | the percept with the temperature (proprioception) band zeroed |
| NEUTER-EXTERO | the percept with the visual and audio bands zeroed |

Statistics per arm and seed, from the engine's own refit trajectory: `acc` = mean held-out accuracy over
the LAST THIRD of refits; `pers` and `maj` = the persistence and majority baselines on the same slices,
same window; `Δpers = acc − pers`; `units` = grown units at the end; `n_modes`; `observed`.

**Verdict rule, with a significance level (the rank-1b lesson).** The real percept stream CHARGES the
grower iff, paired over the 8 seeds, one-sided paired t-tests give p < 0.05 (t > 1.895 at 7 d.f.) for
BOTH: (i) LIVE Δpers > SHUFFLE-CATEGORY Δpers, and (ii) LIVE Δpers > NEUTER-PERCEPT Δpers. The
statistic is Δpers, not raw accuracy, because on a bursty stream raw accuracy is mostly the stream's
inertia. The positive control must clear its own bar (SYNTH acc − SYNTH chance 0.20 > 0.3 on every
seed) or the run is discarded as a dead instrument. The rule's false-positive rate under exchangeability
is computed on the real data by permuting arm labels within seed and reported beside the verdict.

**Pre-registered disposition.** A WIN: the wire stands as built. A NULL: the wire ALSO stands — the
synthetic chain was an experiment about growth, not the brain growing on what it perceives (P0's honesty
guard: "growth is not improvement", and a fabricated stimulus is not experience), so the honest number
on the organism's real stream replaces the flattering number on a synthetic one; the finding is recorded
here as the measured reason rank 2c (a richer real stream) exists, and the engine's `predict_accuracy`
is read against its baselines from now on. What a NULL would NOT license is restoring the chain as a
fallback: that is the retired mechanism behind a flag (P2).

## Result of the FIRST run (protocol v1; 8 seeds × 6 arms × 800 ticks; superseded — kept as the record)

**Read this section as what the panel reviewed, not as the landing's measurement.** The three-lens
review of this run found (a) the wire fed the engine the RAW pre-attention context pattern while the
category was of the attention-GAINED percept (max |difference| 0.5 on every tick), so the engine learned a
map from a space the categories were never drawn in — fixed in the code before the re-run below;
(b) the v1 shuffle control drew categories uniformly and so flattened the class marginal as well as the
temporal structure (0.21 of the LIVE − SHUFFLE gap was the class prior); (c) the attribution finding was
stated above its own bar (t 1.81 < 1.895); (d) "on every seed" was false for one tie. The re-run under
protocol v2 (next section) is the measurement of record; v1's numbers stay here because they were the
review's input.

Per seed, last-third-of-refits means (units = grown units at the end):

| seed | LIVE acc / pers / maj (units) | NEUTER-PERCEPT acc (units) | SHUFFLE acc / pers | SYNTH acc | proprio-only acc (units) | extero-only acc (units) |
|---|---|---|---|---|---|---|
| 0 | 0.389 / 0.961 / 0.400 (3) | 0.400 (43) | 0.208 / 0.228 | 0.689 | 0.519 (3) | 0.367 (71) |
| 1 | 0.744 / 0.956 / 0.597 (2) | 0.597 (48) | 0.267 / 0.244 | 0.742 | 0.483 (1) | 0.597 (10) |
| 2 | 0.292 / 0.928 / 0.386 (1) | 0.386 (38) | 0.231 / 0.194 | 0.639 | 0.369 (2) | 0.372 (23) |
| 3 | 0.494 / 0.986 / 0.364 (2) | 0.364 (25) | 0.203 / 0.194 | 0.744 | 0.000 (2) | 0.364 (6) |
| 4 | 0.864 / 0.978 / 0.706 (1) | 0.706 (1) | 0.286 / 0.206 | 0.719 | 0.706 (2) | 0.706 (3) |
| 5 | 0.428 / 0.653 / 0.181 (2) | 0.181 (3) | 0.233 / 0.214 | 0.694 | 0.431 (2) | 0.156 (12) |
| 6 | 1.000 / 1.000 / 0.667 (2) | 0.500 (5) | 0.178 / 0.186 | 0.661 | 1.000 (2) | 0.125 (4) |
| 7 | 0.178 / 0.947 / 0.178 (1) | 0.178 (30) | 0.183 / 0.219 | 0.644 | 0.178 (1) | 0.178 (11) |

Arm means: LIVE acc 0.549 (pers 0.926, maj 0.435, **1.75 units**); NEUTER-PERCEPT 0.414 (24.1 units);
SHUFFLE 0.224 (pers 0.211; 10.5 units); SYNTH 0.692 (pers 0.17, chance 0.20; 13.9 units);
proprio-only 0.461 (1.9 units); extero-only 0.358 (17.5 units). The brain showed the engine 4.6
categories on average (the arm's learner recruits 4–5 prototypes in 800 ticks); the majority category
holds 40–79% of the buffer.

**Verdict under the pre-registered rule: NULL.** (i) LIVE Δpers − SHUFFLE Δpers = −0.390 ± 0.093,
t = −4.22, 1–7 — FAILS. (ii) LIVE Δpers − NEUTER-PERCEPT Δpers = +0.135 ± 0.065, t = +2.06, 5–2 —
passes. The rule needed both. Its false-positive rate under exchangeability on this data (arm labels
permuted within seed, 4,000 draws): 0.012. Positive control: SYNTH 0.64–0.74 vs chance 0.20 on every
seed (bar 0.3 cleared 8/8) — the feed, the engine and the differencer are live.

**Disclosure (P8 clause 4): comparison (i) was mis-specified when it was pre-registered, and the rule
is NOT being repaired after the fact.** Δpers subtracts each arm's OWN persistence floor, and the
shuffle arm destroys exactly that floor (0.926 → 0.211), so (i) compares the live arm's deficit against
a floor of 0.93 with the shuffle arm's deficit against a floor of 0.21 — different floors, not
comparable. Raw accuracy across arms is reported below as a POST-HOC comparison, labelled as such —
and it is not "the fair one" either: this run's shuffle control drew categories UNIFORMLY, which flattens
the class marginal as well as the temporal order (its majority floor 0.227 vs LIVE's 0.435), so about
0.21 of the raw LIVE − SHUFFLE gap is the class prior; the marginal-preserving comparisons are LIVE
against its own majority floor and against the constant-percept neuter. The rank-1b lesson is that
changing the rule after seeing the data runs against the method, whatever the direction, so the
pre-registered verdict stands as NULL and the corrected rule is re-registered as protocol v2 for a
fresh run rather than applied to this one.

Post-hoc, fair comparisons (raw accuracy, paired, one-sided t at 7 d.f.):

| comparison | mean ± SE | t | wins–losses |
|---|---|---|---|
| LIVE − SHUFFLE | +0.325 ± 0.099 | +3.27 | 7–1 |
| LIVE − NEUTER-PERCEPT | +0.135 ± 0.065 | +2.06 | 5–2 |
| LIVE − its own MAJORITY floor | +0.114 ± 0.050 | +2.27 | 5–2, one float tie (seed 7: +3e-9) |
| LIVE − its own PERSISTENCE floor | **−0.377 ± 0.098** | **−3.86** | **0–7, one tie (seed 6: 1.0 vs 1.0)** |
| LIVE − extero-only (proprio neutered) | +0.191 ± 0.105 | +1.81 | 6–1 |
| LIVE − proprio-only (extero neutered) | +0.088 ± 0.073 | +1.20 | 3–3 |

What the organism's own stream measures:

1. **There is structure, and the grower catches some of it.** Above shuffle (t 3.3) and above the
   constant-percept neuter (t 2.1), and above its own majority floor (t 2.3): the percept predicts the
   next own-category better than "answer the commonest category".
2. **It never reaches the stream's own inertia.** Persistence — "the next category is this one" —
   scores 0.93 on average and beats the grower on 7 of 8 seeds (a tie at 1.0/1.0 on seed 6, whose
   held-out slices were single-category runs) by 0.38 on average. The stream is bursty and
   nonstationary at the buffer's timescale (one category holds 40–79% of the buffer; the held-out slice
   is the most recent third, often a single run), and a batch refit on the earlier window predicts the
   earlier regime. The synthetic chain the engine used to feed itself was STATIONARY and ergodic —
   precisely the property the real stream lacks — which is why ~0.7 there meant nothing about here.
3. **Attribution (P8 clause 3) — SUGGESTIVE, below the document's own bar in this run.** Zeroing the
   proprioception band costs 0.19 (t 1.81 < 1.895, 6–1 with one tie) and the extero-only arm grows
   **17.5 units at 0.358**; zeroing the visual+audio bands is a null (t 1.20, 3–3 with two ties) and the
   proprio-only arm grows 1.9 units at 0.461. The unit-count contrast is the robust half; the accuracy
   half does not clear the bar, and seed 3 is a direct counter-example (proprio-only 0.000 over its last
   third — a model answering a stale category on a single-run held-out slice — while LIVE reached
   0.494). Precondition disclosed: on this machine the gitignored perception checkpoint arms the real
   vision/audio channels (`perception_unified` True); on a fresh clone the extero bands are a period-3
   synthetic rotation, so "i.i.d. digit/word frames" describes this host, not the suite.
4. **Growth on the real stream is minimal: 1.75 units in 800 ticks** (against 13.9 on the synthetic
   chain, 24 on a CONSTANT percept, 17.5 on extero noise). The earn-your-place bar is doing its job on
   a low-entropy stream — and the constant-percept arm exposes an instrument finding worth its own
   look: with every input identical the greedy admission still admits 24 units (up to 48). The panel
   reproduced it directly on `NGSNN.fit` with a constant X and located BOTH halves: (a) at some row
   counts the fit runs to its derived `l_max` on float-noise projections (N=60 → 43 units; N=20/40/240 →
   1–3), and (b) `prune_unearned`'s never-prune-to-nothing guard keeps EVERY duplicate when all are
   individually redundant, so grow-on-top accumulates one unit per refit (the archived arm: 1, 9, 10,
   11, 12, 47, 48 …). Post-LIF columns on a constant input have exactly zero variance, so it is not a
   harness artefact. Recorded, not fixed here (it lives in `hippocampus/ng_snn.py`, not in this wire);
   the honest prune keeps ONE of an all-redundant set.

**Disposition (pre-registered): the wire STANDS.** A NULL was pre-registered to keep the wire, and the
reason holds harder now than when it was written: the honest number on the organism's own stream
(0.55 against a 0.93 floor, 1.75 units) replaces a flattering number on a stimulus the engine made up
(~0.7 against 0.20; the module's own self-check prints 0.67–0.71, the audit's "0.77" was one run). The chain is not restored as a fallback (P2). `predict_accuracy` is now reported
beside its floors on `state()["neurogenesis"]`, so the cockpit cannot show a 1.0 that is inertia.

What this measurement changes downstream:

- **Rank 2c's premise needs a correction.** A rotating real N-MNIST stream will not make
  next-category more predictable — i.i.d. frames are what the extero arms already show to be noise —
  so 2c's criterion stays what the audit pre-registered (the |LP| facade leaving its 1-distinct-value
  floor), and no claim about THIS grower rides on it.
- **The lever this exposes is the LEARNER, not the stimulus** (an invention candidate under P8, not this
  commit): a batch classifier refit on a stationary window is the wrong predictor for a bursty,
  nonstationary category stream; a recency-weighted or transition-conditioned predictor that at least
  reaches the persistence floor is the first thing worth pre-registering. Its red is already written:
  LIVE − persistence ≥ 0 on a majority of seeds, or it is deleted.
- **Rank 2b (persist `GrowingLearner`) is now LOAD-BEARING for this wire, not just for the audit's
  blocker** *[WITHDRAWN as an accuracy claim by 2b's own measurement — `evidence/s2b-growing-learner-lake-record.md`,
  Result: the attribution protocol could not reach the question; 2b's claim is P4b's]*: the engine's categories are the arm learner's prototype ids, and the persisted NG-SNN
  readout is keyed to them; a learner that re-grows from zero on every daemon relaunch re-keys the
  vocabulary the substrate was fit on.

Reproduce: the harness (`rank2_ab.py`) and every per-arm JSON are archived with the session's panel
record beside the memory store (`archive/audit-2026-09-01/rank2a/`); the pinned test
`tests/test_real_percept_stream.py` carries the wire's reds.

## Protocol v2 — PRE-REGISTERED after the panel, BEFORE the re-run (this section written first)

Changes from v1, each traced to a panel finding, none to the v1 numbers' direction:

1. **The wire feeds the vector the learner categorized.** `AutonomousArm.step` now returns
   `res["percept"]` (the attention-gained context vector its `GrowingLearner.observe` ran on) and
   `Brain._wake_tick` hands exactly that to `observe()`. The attribution arms neuter bands of THAT
   vector.
2. **The verdict statistic is raw held-out accuracy, paired over seeds**, with each arm's floors reported
   as within-arm questions. The v1 statistic subtracted a per-arm floor that a control destroys (obs
   #105); it is not used again.
3. **The shuffle control preserves the class marginal**: the replacement category is drawn from the
   running list of category OCCURRENCES (with multiplicity), so it keeps the prior and destroys only
   the temporal order.
4. **Two arms added.** ORACLE-PREVCAT: the percept is the one-hot of the PREVIOUS category — what the
   same batch learner scores when handed exactly the information persistence uses (lens A's control: if
   it reaches the persistence floor, the deficit is the percept→category relation, not the batch refit).
   NEUTER-FEED: `observe()` is a no-op — the engine sees nothing — and the WHOLE final `Brain.state()` is
   recorded for every arm, so the brain-level charge question (P3: does the wire change any outcome
   OUTSIDE the engine?) is answered by the flattened key diff LIVE vs NEUTER-FEED per seed, the rank-1
   differencer.
5. **Preconditions recorded per arm**: `perception_unified`, real vision, real audio (on this machine
   the gitignored perception checkpoint arms the real channels; on a fresh clone the extero bands are a
   period-3 synthetic rotation).

**Verdict rule v2 (with alpha):** the real percept stream CHARGES the grower iff paired one-sided t >
1.895 (7 d.f., p < 0.05) for BOTH LIVE − SHUFFLE(marginal-preserving) and LIVE − NEUTER-PERCEPT on raw
held-out accuracy (last third of refits). Reported beside it, never in it: LIVE − own majority, LIVE − own
persistence, ORACLE − own persistence, the attribution arms with their t against the same bar (a
sub-bar t is "suggestive", never "carries"), the brain-level charge keys, and the positive control
(SYNTH − 0.20 > 0.3 on every seed, else the run is discarded). Ties are counted as ties.
**Disposition unchanged and pre-registered for either outcome:** the wire stands (P0/P2: the fabricated
stimulus is not restored); the brain-level charge is reported under P3's vocabulary
(CONSUMED-BUT-NOT-DECISIVE → UNCERTAIN) and does not by itself revert the wire, because reverting it
restores a fabricated stimulus.

## Result of the re-run (protocol v2) — THE MEASUREMENT OF RECORD

8 seeds × 8 arms × 800 ticks, every arm its own process, `perception_unified` True on all 64 live arms
(this host's checkpoint arms the real vision/audio channels), 0 feed refusals anywhere. Last-third
means (units = grown units at the end):

| seed | LIVE acc / pers / maj (u) | NEUTER-PERCEPT (u) | SHUFFLE acc / pers / maj (u) | SYNTH | ORACLE-PREVCAT (u) | proprio-only (u) | extero-only (u) |
|---|---|---|---|---|---|---|---|
| 0 | 0.417 / 0.961 / 0.400 (2) | 0.400 (43) | 0.397 / 0.367 / 0.611 (7) | 0.689 | 0.419 (2) | 0.519 (2) | 0.322 (44) |
| 1 | 0.956 / 0.956 / 0.597 (2) | 0.597 (48) | 0.683 / 0.700 / 0.786 (1) | 0.742 | 0.597 (2) | 0.956 (2) | 0.597 (2) |
| 2 | 0.247 / 0.928 / 0.386 (2) | 0.386 (38) | 0.367 / 0.478 / 0.636 (7) | 0.639 | 0.386 (2) | 0.219 (2) | 0.386 (2) |
| 3 | 0.000 / 0.986 / 0.364 (3) | 0.364 (25) | 0.500 / 0.381 / 0.503 (31) | 0.744 | 0.653 (2) | 0.097 (2) | 0.158 (2) |
| 4 | 0.978 / 0.978 / 0.706 (3) | 0.706 (1) | 0.658 / 0.789 / 0.889 (15) | 0.719 | 0.978 (7) | 0.519 (1) | 0.706 (1) |
| 5 | 0.458 / 0.653 / 0.181 (2) | 0.181 (3) | 0.419 / 0.331 / 0.408 (16) | 0.694 | 0.456 (3) | 0.256 (2) | 0.331 (2) |
| 6 | 1.000 / 1.000 / 0.667 (2) | 0.500 (5) | 0.303 / 0.458 / 0.361 (5) | 0.661 | 1.000 (1) | 1.000 (1) | 1.000 (2) |
| 7 | 0.178 / 0.947 / 0.178 (1) | 0.178 (30) | 0.561 / 0.508 / 0.625 (30) | 0.644 | 0.156 (1) | 0.178 (1) | 0.178 (4) |

Mean units: LIVE 2.1, NEUTER-PERCEPT 24.1, SHUFFLE 14.0, SYNTH 13.9, ORACLE 2.5, proprio-only 1.6,
extero-only 7.4, NEUTER-FEED 0 (the engine saw nothing).

**Verdict under protocol v2: NULL.** LIVE − SHUFFLE (marginal-preserving) = +0.043 ± 0.138, t = 0.31,
5–3; LIVE − NEUTER-PERCEPT = +0.115 ± 0.101, t = 1.14, 5–2 with one tie. Neither clears 1.895. Rule
false-positive rate under exchangeability on this data: 0.010. Positive control: SYNTH 0.64–0.74 vs
chance 0.20 on every seed (8/8). The grower's next-category prediction from the brain's own percept
carries nothing measurable beyond the class prior.

Within-arm floors and the oracle (reported, not part of the verdict):

| comparison | mean ± SE | t | wins–losses–ties |
|---|---|---|---|
| LIVE − its own MAJORITY floor | +0.094 ± 0.092 | +1.03 | 6–2–0 |
| LIVE − its own PERSISTENCE floor | −0.397 ± 0.140 | −2.83 | 2–5–1 (AT the floor on seeds 1, 4, 6) |
| ORACLE-PREVCAT − its own PERSISTENCE floor | −0.345 ± 0.098 | −3.53 | at the floor on 2 seeds, below on 6 |
| ORACLE-PREVCAT − its own MAJORITY floor | +0.146 ± 0.056 | +2.61 | — |
| LIVE − ORACLE-PREVCAT | −0.051 ± 0.099 | −0.52 | 3–3–2 |
| LIVE − extero-only (proprio neutered) | +0.069 ± 0.065 | +1.08 | 4–2–2 |
| LIVE − proprio-only (extero neutered) | +0.061 ± 0.066 | +0.93 | 3–2–3 |

What the organism's own stream measures, corrected wire:

1. **The stimulus is not the limit; the learner's admission window is.** The persistence oracle — the
   percept IS the previous category, one-hot, so an identity map scores the persistence floor — reaches
   that floor on only 2 of 8 seeds and stops at 2.5 units. NGSNN admits a unit only if it lowers error on
   the held-out slice, and the slice is the most RECENT third of a bursty stream: usually a single run
   of one category, so a unit that would predict any rarer category never earns its place, and the
   readout answers the majority. The live percept does no worse than the oracle (t −0.52). (The panel's
   3-seed oracle at 400 ticks reached 0.93 on seed 0 — a different window composition — which is the
   same mechanism seen from the other side.)
2. **With the corrected wire the grower reaches the persistence floor on 3 of 8 seeds** (1, 4, 6; v1
   reached it on none but a tie) and clears its majority floor on 6 of 8 — suggestive (t 1.03), not
   established. The failures (seeds 0, 2, 3, 7) are the regime-shift shape: seed 3 scores 0.000 over its
   last third because the held-out run is a category the fit answers with a stale label.
3. **Attribution: both bands are nulls at this power** (t 1.08 and 0.93). What survives is the unit
   count: with proprioception neutered the grower fits the exteroceptive channels into 7.4 units at
   0.36 (v1: 17.5); with the exteroceptive channels neutered, 1.6 units at 0.46. v1's "proprioception
   carries the signal" is withdrawn as a claim; "the extero channels are where the units go" stands as a
   count, not a test.
4. **Brain-level charge (P3), measured with the whole `state()`:** LIVE vs NEUTER-FEED differ on
   `growth_gate.cortex.{grow, reason}`, `self_state.{grown_units, confidence, surprise}`,
   `self_surprise_gen` on every seed and `reliability` on two — all self-model and growth-gate keys fed
   by `grown_units`/`predict_accuracy`; no action, access, novelty or arm key moves. That is P3's
   CONSUMED-BUT-NOT-DECISIVE bucket: the wire is UNCERTAIN as a charge on behaviour, never claimed as one.
   It stands because reverting it restores a fabricated stimulus (P0/P2), not because it charges.
5. **Growth on the real stream: 2.1 units in 800 ticks** (constant percept 24, shuffle 14, synthetic
   chain 13.9). The instrument finding from v1 (units admitted on a constant input) is unchanged.

**Disposition (pre-registered for either outcome): the wire STANDS.** The honest number on the organism's
own stream replaces the flattering one on a stimulus the engine made up; the chain is not restored.
`predict_accuracy` travels with its floors on `state()`, in the growth blob, on the cockpit and in
`brain_telemetry.PROVENANCE`.

What this measurement changes downstream:

- **The lever is the admission rule's validation window, with the oracle as its control** (a P8
  candidate, not this commit): a held-out slice that covers the category marginal (stratified or
  recency-mixed) instead of the most recent run. Its red is written now: ORACLE-PREVCAT must reach its
  persistence floor on a majority of seeds under the new window, and LIVE must not fall; if the oracle
  still fails, the window was not the limit and the finding is recorded. "Recency weighting" (v1's
  guess) is withdrawn as the first candidate.
- **Rank 2b is load-bearing for this wire** *[withdrawn as an accuracy claim by 2b's measurement; see
  `s2b-growing-learner-lake-record.md`]*: the engine's categories are the arm learner's prototype
  ids; a learner that re-grows from zero on every daemon relaunch re-keys the vocabulary the persisted
  readout was fit on (the panel measured the OPEN-gate path re-keying at the first refit and the
  CLOSED-gate path mis-keying — accuracy 0.0 rows — until 2b lands).
- **Rank 2c's premise is weakened, not corrected**: the attribution arms are nulls, so nothing here says a
  rotating real stream would or would not change next-category predictability; 2c's own criterion (the
  |LP| facade leaving its floor) is the only claim it carries.
- **Expected on the daemon's relaunch onto this commit** (pre-registered for the post-commit poll): the
  restored synthetic-era substrate (22 units in the live lake) is mostly pruned at the first refit on the
  real stream (the panel measured 24 → 8 on a HEAD blob), and `predict_accuracy` reads the restored
  0.7-ish figure beside `None` floors until that refit. Record the observed drop, do not smooth it.

Reproduce: `rank2_ab_v2.py` and the 65 per-arm JSONs (`v2_arms/`) are archived with the session's panel
record beside the memory store (`archive/audit-2026-09-01/rank2a/`), with the three lens reports; the
pinned test `tests/test_real_percept_stream.py` carries the wire's reds.
