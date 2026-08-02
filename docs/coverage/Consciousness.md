# Coverage evidence -- Consciousness

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| access_strength_graded | 42 | src/snn_data_lake/thalamus/reentrant_workspace.py:322#access_strength; src/snn_data_lake/cortex/brain.py:2216#_observe_self_wake; src/snn_data_lake/cortex/brain.py:1919#_update_metacog | cons_refute_strength_iso_120.txt | opus-5 REFUTER / consciousness_refute_probe.py strength_iso 120 (55 REFUTED) | 2026-07-30 |
| second_order_access | 42 | src/snn_data_lake/cortex/second_order_access.py:176#hold; src/snn_data_lake/cortex/second_order_access.py:142#observe; src/snn_data_lake/cortex/brain.py:3064#respond | consciousness_second_120.txt | opus-5 REFUTER / consciousness_refute_probe.py second_thin,c1,ws,av,st 120 (50 REFUTED) | 2026-07-30 |
| self_model | 38 | src/snn_data_lake/cortex/self_model.py:274#confidence; src/snn_data_lake/cortex/brain.py:2193#_observe_self_wake | - | opus-5 / scratchpad measure_consciousness2.py + 4.py | 2026-07-30 |
| reentry_sustain | 35 | src/snn_data_lake/thalamus/reentrant_workspace.py:218#step; src/snn_data_lake/thalamus/reentrant_workspace.py:257#step | - | opus-5 / scratchpad measure_consciousness3.py, 62 wake ticks | 2026-07-30 |
| ignition | 30 | src/snn_data_lake/cortex/brain.py:2329#_update_access; src/snn_data_lake/cortex/brain.py:1413#_route_percept_to_action | ignition_60.txt | opus-5 / tools/charge_probe.py ignition 60 | 2026-07-30 |
| reentry_reroute | 28 | src/snn_data_lake/thalamus/reentrant_workspace.py:301#_deliver; src/snn_data_lake/cortex/brain.py:1600#_route_percept_to_action | reroute_60.txt | opus-5 / tools/charge_probe.py reroute 60 | 2026-07-30 |
| global_availability | 25 | src/snn_data_lake/thalamus/reentrant_workspace.py:315#_deliver; src/snn_data_lake/cortex/access_consciousness.py:224#step | - | opus-5 / scratchpad measure_consciousness3.py, 62 wake ticks | 2026-07-30 |
| integration_measure_pci | 22 | src/snn_data_lake/cortex/brain.py:2412#_sleep_tick; src/snn_data_lake/cortex/brain.py:2916#state | - | opus-5 / scratchpad measure_consciousness2.py, 200 ticks / 8 nights | 2026-07-30 |
| arm_workspace_broadcast | 21 | src/snn_data_lake/brainstem/autonomous_loop.py:1163#_workspace_broadcast; src/snn_data_lake/cortex/live_brain.py:92#__init__ | - | opus-5 / scratchpad measure_consciousness4.py, 240 ticks | 2026-07-30 |
| reentrant_predictive_completion | 15 | src/snn_data_lake/reasoning/gwt_reentry.py:50#run_loop; src/snn_data_lake/reasoning/consciousness_probe.py:102#probe_reentry | - | opus-5 / static reachability + live respond() probe | 2026-07-30 |
| reportability_of_access_content | 12 | src/snn_data_lake/cortex/live_brain.py:365; src/snn_data_lake/reasoning/spike_chat_lake.py:929#converse | - | opus-5 / live respond() on the three consciousness phrases | 2026-07-30 |
| binding_of_broadcast_content | 10 | src/snn_data_lake/thalamus/global_workspace.py:335#bind_broadcast_spiking; src/snn_data_lake/thalamus/global_workspace.py:605#main | - | opus-5 / static call-site search (self-check only) | 2026-07-30 |

**Category score: 26.7** = 320 / 12, **down from the audited 28.4 after an adversarial pass in which
both band-41+ rows fell** (55 -> 42 and 50 -> 42). Two rows still clear 40 on recorded neuter-A/Bs, but
each now sits at the FLOOR of 41-60 rather than the middle, and each rests on a different artifact than
the audit cited. **No row reaches 61**, so this category contains **no claim that a decision, action,
or learning update depends on a consciousness faculty** -- and the refutation strengthened that: on the
isolated A/B the lifelong learn-gate admissions changed 0/120 and the arm trajectory 0/120.

The two refutations, in one line each, both measured not argued:

* `access_strength_graded` 55 -> 42. The audit's neuter had **three** live consumers, not one. Pinning
  `access_strength()` to 1.0 also drove the region-coherence "access" raster (`brain.py:2040`) and the
  second-order monitor's strength input (`brain.py:2843`), where the no-workspace default is **0.0**,
  not the 1.0 the audit called "the documented default". Isolating the claimed consumer removes the
  entire reply/hedge evidence: **29/30 turns -> 0/30, hedges 8-vs-2 -> 8-vs-8.**
* `second_order_access` 50 -> 42. Removing each of the monitor's four inputs in turn shows the abstain
  decision rides on **one** of them -- the availability bit that this file's own `global_availability`
  row proves cannot differ from ignition. Strength contributes 0/120, `_metacog_rel` contributes 0/120.

Full method, numbers and the two attacks that FAILED are under "Adversarial pass" below.

---

## How this was measured

Interpreter `.venv-lake` (py3.11.15), `PYTHONPATH=src`, HEAD `8436b5e9`. Every number below comes from
a live brain built the Policy-4 way, `make_brain(seed=0, persist=False)`, ticked and then read -- not
from a grep.

**Harness control.** `tools/charge_probe.py drive 60` reproduced exactly: `22/60` trajectory changes,
exit 0, the figure `COVERAGE_BASELINE.md:90` cites. So a zero in this file is a zero in the brain, not
a broken probe.

**Two probes, and why a second one exists.** `tools/charge_probe.py` neuters a key of `Brain._drive_w`,
so it can only reach faculties wired into the arm's drive vector -- `ignition` and `reroute` here. The
two faculties consumed elsewhere (the graded access readout, which feeds a self-model term; and the
second-order hold, which is read only inside `respond()`) needed a probe of their own, so
`docs/coverage/artifacts/consciousness_ab_probe.py` is tracked alongside its artifacts. It mirrors
`charge_probe`'s method and exit-code contract, and it runs arm A twice as a determinism guard before
reporting anything.

**The first version of that probe was wrong, in the dangerous direction.** It pinned
`Brain._reentrant_strength` before `tick()` -- but `_w8_tick` (`brain.py:1671`) recomputes that
attribute from the workspace *inside* the tick, before `_observe_self_wake` (`brain.py:1821`) consumes
it. So the neuter leaked and the probe reported `confidence() changed 0/120 ticks`: a live faculty
reading as dead, which is the verdict that gets a mechanism retired. Patching the producing method
(`ReentrantWorkspace.access_strength`) instead gives **89/120**. This is `charge_probe.py:54-67`'s
`_PinnedWeights` lesson recurring in a new place, and it is the reason the leaky number is recorded here
rather than quietly dropped.

Everything not covered by those two probes was measured by instrumented live runs whose scripts are
untracked scratchpad files -- see "what I could not measure".

**A third probe was added by the adversarial pass.** `docs/coverage/artifacts/consciousness_refute_probe.py`
re-measures the two rows above 40 with per-consumer attribution rather than a single verdict, and it is
what lowered both. Its method, its seven artifacts and the two of its own attacks that failed are in
"Adversarial pass" below.

---

## Row-by-row: what was verified

### access_strength_graded -- 42 (audited 55, REFUTED; the audit's own text follows, then the refutation)
The 32-unit heterogeneous LIF readout population added to de-saturate access strength. **It works and
it is decisive on an internal read.** Live it takes 14-15 distinct values (0.0 to 1.0) over 60-200
ticks, so the pinned-at-1.000 defect its own docstring documents is genuinely fixed. It is then
consumed, not parked: `brain.py:2136` combines it with `_access_ema` under `min()` and passes the result
as the self-model's `access_index`, one of the three **multiplicative** factors of
`self_model.confidence()` (`self_model.py:274`). `min()` selected the workspace strength -- not
`_access_ema` -- on **169 of 200** ticks.

The A/B (`consciousness_strength_120.txt`, exit 0) neuters it to **1.0, the value `brain.py:2134-2135`
documents as the no-workspace default**, so arm B is exactly the pre-de-saturation brain rather than an
artificial zero (pinning it to 0.0 would collapse confidence to 0 and "prove" a charge trivially):

    self_model.confidence() changed by the signal : 89/120 ticks
    metacog HOLD gate changed by the signal       : 1/120 ticks
    arm trajectory changed by the signal          : 0/120 ticks
    REPLY TEXT changed by the signal              : 29/30 turns

**55, not 61+**, and the reason is the third line. Band 61-80 needs a changed decision, action, or
learning update; the arm trajectory was untouched on every one of 120 ticks and the metacognitive hold
gate flipped once. What this faculty demonstrably changes is a **self-model term and a reply** -- which
is the rubric's own wording for 41-60. It sits in the upper half of that band because 89/120 is a
strong, repeatedly-sampled divergence on the quantity the brain's own confidence is built from.

**REFUTED to 42.** The audit invited exactly this attack ("show that the 1.0 neuter is not the right
counterfactual") and the attack lands, for a reason stronger than correlation:
`_reentrant_strength` has **three** live consumers and the A/B neutered all three at once.

    cortex/brain.py:2137   self-model access_index          <- the row's claimed consumer, getattr default 1.0
    cortex/brain.py:2040   region-coherence "access" raster <- getattr default 0.0
    cortex/brain.py:2843   SecondOrderAccess.observe(..., STRENGTH, ...)  <- getattr default 0.0

So "1.0 is the value `brain.py:2134-2135` documents as the no-workspace default" is true at **one** of
the three sites and is the **opposite extreme** at the other two: with no workspace those two read 0.0.
Arm B is therefore not "exactly the pre-de-saturation brain" -- it is a brain whose cross-region access
raster is pinned at maximum on every tick. And that raster is not inert: it runs through the synfire
binding gate into `_reentrant.ignition_threshold` (`brain.py:2063-2070`), i.e. **back into the ignition
of the workspace being probed.** Measured (`cons_refute_strength_paths_120.txt`, exit 0, 120 ticks):

    region-coherence value changed   : 116/120 ticks
    synfire BINDING flag changed     :  31/120 ticks
    workspace IGNITION THRESHOLD chg :  48/120 ticks
    workspace availability changed   :  22/120 ticks
    second-order HOLD changed        :  24/120 ticks

Those last two lines mean the audit's `strength` A/B silently neutered three *other* rows of this same
category (`ignition`, `global_availability`, `reentry_sustain`) and then credited the whole divergence
to this one.

`cons_refute_strength_iso_120.txt` (exit 0, reproduces under `--rerun`) runs the counterfactual the row
actually claims and nothing else: `access_strength()` is left live, so `:2040` and `:2843` still see the
true graded value, and only the self-model's `access_index` is replaced by `_access_ema` -- which *is*
`min()` as the identity, i.e. precisely what `brain.py:2134-2135` documents. The result splits the
audit's evidence in two:

    self_model.confidence() changed      : 89/120 ticks   (identical to the audit -- SURVIVES)
    metacog reliability changed          : 48/120 ticks   (a real internal read -- SURVIVES)
    confidence BUCKET changed            :  0/120 ticks   (the episode-ring / durable-lake write)
    metacog HOLD gate changed            :  1/120 ticks
    lifelong learn-gate admissions       :  0/120 ticks
    arm trajectory changed               :  0/120 ticks
    REPLY TEXT changed                   :  0/30 turns    (audit reported 29/30 -- FALLS)
    hedged replies A vs B                :  8 vs 8        (audit reported 8 vs 2 -- FALLS)

**The reply evidence was not this faculty's.** All 29 reply divergences and all 6 lost hedges came from
the two contaminating paths, and the hedge collapse specifically is the second-order monitor's hold
flipping on 24/120 ticks -- the mechanism the *next* row is scored on. That is one cause counted twice
across two rows, the defect that cost Executive 9 points, and it is why this row cannot keep the upper
half of its band on "a self-model term **and a reply**".

What survives is genuine and is why the row stays above 40 rather than dropping into the twenties:
confidence moves on 89/120 ticks, and `brain.py:1852` feeds `self_model.state()["confidence"]` into the
metacognitive monitor, whose reliability then differs on 48/120 ticks and is read by the hold gate, the
explore gate and the lifelong learn-gate. That is "changes an internal read", so 41-60 -- at the
**floor**, because every downstream consumer of that changed read declined to change: the bucket the
self-episode ring stores is identical on 120/120, the learn-gate admits the same tasks, the hold gate
flips once in 120 (0.8%), and the arm never moves.

### second_order_access -- 42 (audited 50, REFUTED)
The spiking metacognition of access (`SecondOrderAccess`), read in `respond()` and cached as
`_second_order_hold`. `brain.py:2949` branches the reply on `metacog_hold or second_order_hold`, so
whenever the second-order monitor holds and the metacognitive monitor does not, this faculty alone
selects a different reply. Two independent measurements agree on the size of the effect.

First, the joint distribution over 240 ticks with a chat turn every 4th tick (60 turns):

    (second_order_hold, metacog_hold) = (False,False) 52 | (True,False) 7 | (False,True) 1

**7 of 60 turns took the hedged branch because of this faculty and nothing else.** Second, the A/B
(`consciousness_second_120.txt`, exit 0) pins `hold()` to False so the monitor never abstains:

    REPLY TEXT changed by the signal        : 22/30 turns
    hedged replies  A(wired) vs B(neutered) : 8 vs 1
    confidence / metacog gate / trajectory  : 0/120 ticks each

`8 - 1 = 7` hedges are attributable to the second order, which reproduces the joint-distribution figure
exactly. **Read the 22/30 with care and do not quote it as decisiveness**: once one reply diverges, the
reply path mutates state (`_tom_update`, `introspect`, `reason`), so later turns differ by cascade. 7 is
the honest per-turn number.

**50, not higher**, on three counts: it changes a reply and nothing else (confidence, the metacog gate
and the trajectory are all 0/120); it lives on the **chat** path only, so a bare tick never evaluates
it; and its `calibration()` reads 1.0000 only because its ground truth `ignited and availability >= 0.5`
was almost always true -- that 1.0 is degenerate, not skill. It is also fed an `availability` that is a
duplicate of the `ignited` bit passed alongside it (see `global_availability`).

**REFUTED to 42.** The audit's neuter removes the whole monitor (`hold() -> False`) and so cannot say
*which* of the monitor's four inputs the abstain decision is made of. `observe(availability,
access_strength, ignited, first_order_conf)` sums them into one drive (`second_order_access.py:142-150`),
three of the four are workspace quantities and the fourth is Executive's `_metacog_rel`
(`brain.py:2845`). I removed each side's **information** while keeping its scale -- each input replaced
by its own running mean, the P1-shaped neuter -- and re-ran the same 120 ticks / 30 turns. Four recorded
A/Bs, all reproducible, and the decomposition is stark:

    input whose information was removed   hold flips   hedges A vs B   replies changed   exit
    first_order_conf  (= _metacog_rel)      0/120         8 vs 8            0/30           1
    access_strength   (the graded readout)  0/120         8 vs 8            0/30           1
    availability      (the ignited bit)    24/120         8 vs 2            6/30           0
    availability + access_strength         24/120         8 vs 2            6/30           0

**The abstain decision is a function of one input.** Availability alone reproduces the entire footprint;
the graded access strength and the first-order reliability contribute exactly nothing. And this file's
own `global_availability` row establishes that availability in this wiring **cannot** differ from the
ignited bit -- `reached / len(self._subs)` with all-or-none delivery is 0.0 or 1.0. So the
"second-order metacognition of access" that runs live is a 24-neuron LIF re-transduction of a single
first-order bit already scored elsewhere in this category, and its `calibration()` of 1.0 is degenerate
for the same reason. Band 41-60 still holds by the rubric's own wording -- the output demonstrably
changes a reply -- but at the **floor**: one bit of information, 7 of 60 turns, nothing else moving.

**Two attacks on this row FAILED and the failures are recorded because a refuter's negatives are
evidence too.** (1) The audit suggested the row might be "Executive's signal leaking through a
second-order wrapper that adds nothing of its own". It is not: removing `_metacog_rel`'s information
changed the hold on **0/120** ticks (`cons_refute_second_c1_120.txt`, exit 1). Whatever this monitor is,
it is not a metacog_rel repeater. (2) I expected the hedge to be a canned 37-character prefix on an
otherwise identical narrator sentence, which would have been band-floor-or-below. It is not: of the 22
diverging turns, **0** satisfy `A == HEDGE + B` and all 22 have a genuinely different tail
(`cons_refute_second_thin_120.txt`). The hedged branch produces substantively different text, so the
reply change is real. Those two findings are why this row lands at 42 and not in the twenties.

**Double-count booked once, deliberately.** My measurement shows `global_availability`'s readout is not
telemetry-only after all: removing its information changes 6/30 replies, which on the band table alone
would read 41+. Raising it would score the **same chain** twice (availability -> second-order hold ->
hedged reply). The chain's charge is booked here, on the row whose neuter is the faculty itself, and
`global_availability` stays at 25 with its behavioural route explicitly attributed to this row. That
choice costs the category ~1.3 points relative to raising both, and it is the reason the roster was not
merged instead: the two neuters are *not* behaviourally indistinguishable (8->1 vs 8->2 hedges, 28 vs 24
hold flips), so the Executive precedent of collapsing indistinguishable probe points into one row does
not strictly apply, and collapsing them would have **raised** the mean to 26.8 by shrinking the
denominator. Both readings are in "Adversarial pass" below.

### self_model -- 38
Confidence took 49 distinct values over 200 ticks and the narrator's introspective text took 20
distinct forms over 60 turns, so the self-model reaches the reply and varies with state. Two specific
findings:

* **The `model_conf == clamp01(acc)` identity is really gone.** Recomputing the exact expression at
  `self_model.py:272` on live signals: the identity held on **1 of 200 ticks** (39 distinct
  `model_conf` values). The prior audit's "197/197 -> 1/197" claim is confirmed at 200 ticks. The
  learned generative self-model (`GenerativeSelfModel`, e-prop) is what supplies the real prediction
  error -- 18 distinct surprise values live -- and it is the reason the identity broke.
* **A new dead constant in its place.** `confidence()` is `decision_conf * model_conf * access_conf`,
  and `decision_conf` is `0.5 if decision_margin was never observed`. `decision_margin` is a declared
  self-model signal but **`_observe_self_wake` never passes it**, so `decision_conf` measured **0.5 on
  200/200 ticks**. One of the three factors is inert. (The BG decision margin *is* consumed -- but at
  `brain.py:1846`, into the metacognitive monitor, by a separate baseline-relative computation. The
  self-model's own margin channel is the one that is dead.)

### reentry_sustain -- 35
The closed re-entrant loop: `reentrant_workspace.py:218` adds the decaying sustain current to the
incumbent coalition's drive, `:257` fades it. Live it is **not** a constant -- 19 distinct
`_g_reentry` values over 62 wake ticks: pinned at 0.88 (= `_SUSTAIN` 1.10 x `_REENTRY_DECAY` 0.80) on
41 ticks where bottom-up support re-charges it every tick, then a genuine decay tail
(0.704 ... 0.0159) through a dropout. It alters the next tick's competition, which is an internal read.
Scored 35 rather than higher because the property the module's self-check demonstrates in isolation --
"re-entry sustains the content across a bottom-up dropout" -- **did not manifest live**: the workspace
went dark for 8 consecutive wake ticks while `g_reentry` decayed, i.e. the loop produced a fading tail
rather than a held content. The incumbent does change (`winner` was `ctx` 42 ticks, `self` 12, `None`
8), so the slot is genuinely contested; the hold is just weak.

### ignition -- 30, and this is the row with the pre-declared ceiling on it
`tools/charge_probe.py ignition 60` returned **exit 0, `BEHAVIOURAL CHARGE OK`, 47/60 trajectory
changes** (`ignition_60.txt`). On the band table alone that reads as 61-80. It is scored **30**, for
two independent reasons, and both need to be on the record:

1. **The rubric pre-declares that this route cannot raise this category.** `COVERAGE_RUBRIC.md:99-102`
   says wiring into `_route_percept_to_action` moves a three-way arm decision and "cannot raise
   Language, Imagination, Gf, Consciousness or WholeBrain -- crediting those to a wiring wave is the
   theater P0 forbids". `elig["ignition"]` at `brain.py:1354` is exactly that wiring. Taking the 61
   would be scoring around a ceiling that was written down before scoring started.
2. **The signal the probe neutered is a constant.** Measured over 200 ticks, `_access_state["ignited"]`
   was True on **197** of them; the only dark ticks are 0, 1 and 2, before the novelty baseline warms:

       ignited trace: 000111111111111111 ... 111  (3 dark, then never again)

   So `elig["ignition"]` is 1.0 on 197/200 ticks. Zeroing `_drive_w["ignition"]` therefore removes a
   near-constant term from the drive vector, and the 47/60 divergence measures the **magnitude of that
   term**, not information carried by ignition. This is the repo's own recorded failure mode --
   `charge_probe.py:124-131` warns that a *moving* weight proves nothing; the mirror case is a
   *constant* eligibility with a load-bearing weight. The all-or-none gate that `access_consciousness.py`
   exists to provide does not gate on the live path after the third tick.

   The re-entrant workspace is different and better: its ignition **failed on 8 of 62 wake ticks**
   (13%), so *that* gate does gate. Two live ignition mechanisms, opposite verdicts -- see the P2 note.

### reentry_reroute -- 28 (this row REPLACES the prior `workspace_reentry_key` 0, and the 0 was stale)
The prior file scored `workspace_reentry_key` **0** under rule 4, on the ground that the Hebbian write
key `(source, content)` and the read key `(subscriber, content)` were disjoint. **That defect is
fixed** -- `reentrant_workspace.py:289-304` is the v4.78 key-space fix, which aggregates the learned
association over the sources that potentiated the ignited content. Verified live: `_assoc` holds
`('percept','ctx')` and `_reroute_bias` measured 23 distinct values in 0.0-0.16, where 0.16 is
`_REENTRY_GAIN` 0.08 x (1 + assoc) with assoc at ~1.0. So the learned term is read, and it doubles the
base gain. The bias then reaches `elig["reroute"]` and the per-action drive vector (`brain.py:1533`).

Scored 28, not higher: the value is concentrated at its ceiling (0.16 on 17 of 62 wake ticks, 0.0 on
the 9 dark ones), so the fix converted a constant 0.08 into a mostly-constant 0.16 -- and
`tools/charge_probe.py reroute 60` returned **exit 2, `UNKNOWN`, 0/60 divergence** (`reroute_60.txt`).
The artifact is attached even though it is a negative: an honest negative is evidence, and attaching it
stops the same run being re-badged as a charge later. `UNKNOWN` is not evidence of inertness -- an
undirected source has no learned value table, so eligibility cannot be established from outside -- but
it is certainly not a charge.

### global_availability -- 25
Both availability readouts are **exact restatements of the ignited bit, structurally, not by
coincidence.**

* `reentrant_workspace.py:315` returns `reached / len(self._subs)`, and `_deliver` increments `reached`
  for *every* subscriber whenever ignition occurred. With the two live subscribers (`action`,
  `language`, registered at `brain.py:710-711`) the value can only be 0.0 or 1.0. Measured over 62 wake
  ticks: 2 distinct values, and the availability trace is **character-for-character identical** to the
  ignited trace (54 ignited / 8 dark).
* `access_consciousness.py:200` computes `received / self.n_taps` over 4 taps that all receive the same
  broadcast object, so it is 1.0 or 0.0 for the same reason. Measured 57/3 over 60 ticks, matching
  `ignited` exactly.

This confirms and sharpens the prior claim. The prior wording was "1.0 on all 60 ignited ticks"; the
stronger true statement is that availability **cannot** differ from ignition in this wiring, so it adds
no information -- including to `_second_order.observe(availability, strength, ignited, ...)` at
`brain.py:2842`, which receives both the availability and the bit it duplicates. A knock-on: because
`availability == gate == 1` whenever ignited, `access_index = gate * availability * integration`
collapses to `integration`, and the two were measured with identical value sets (18 distinct values
each over 60 ticks). The three-factor access index is a one-factor index live. 25 = live, consumed, and
provably redundant.

### integration_measure_pci -- 22
`PerturbationalComplexity` on the sleep path, every 4th night. Its precondition does fire, so this is
not a rule-4 zero. But: over 200 ticks spanning **8 nights** and 3 recomputations it took **exactly one
value, 0.4068** -- the prior audit's "one value over 25 nights" reproduces. And it has no consumer at
all: `_pci` appears at `brain.py:417` (init), `:2332` (write) and `:2801` (`state()`). Live, constant,
telemetry-only.

### arm_workspace_broadcast -- 21
The original spiking GWT (`thalamus/global_workspace.py`) is live inside the arm --
`live_brain.py:92` passes `workspace=True`, so `_workspace_broadcast` steps it on every tick. Its own
docstring calls it a "READ-ONLY annotation". Measured: the broadcast content took **2 distinct values
over 240 ticks** (`0` on 237, `None` on 3), and `_ignitions` -- the only quantity that escapes, via
`substrate/brain_telemetry.py:267` -- reached **1**. The broadcast content itself is returned as
`res["workspace"]` and **grep finds no reader for that key anywhere in `src/`**. Band 21, the floor of
"live, not consumed", because it is below even the telemetry case the band describes: the content has
no reader, and the one derived number that does reach telemetry moved once in 240 ticks.

### reentrant_predictive_completion -- 15
`reasoning/gwt_reentry.py` is the only implementation of re-entrant **predictive completion** (weak
bottom-up evidence + a matching top-down prior crossing ignition together) with an explicit
no-hallucination guard -- a capability the live `ReentrantWorkspace` does not have. It is not
registered in `REGIONS`, so `dormancy_gate` never looks at it. It is reachable, but only thinly:
`consciousness_probe.py:102` imports `run_loop`, and `consciousness_probe` is imported by
`reasoning/spike_chat_lake.py`, which `live_brain.py:450` loads. So its live reach is a boolean printed
inside one fixed-phrase chat reply. It changes nothing. 15 = built, effectively not live.

### reportability_of_access_content -- 12
Reportability is the defining criterion of access consciousness, so it is scored. What runs:

* The dominant intent is a **hardcoded string**. `live_brain.py:360-370` matches
  `{conscious, sentient, aware, awake, sapient}` and returns a fixed paragraph. Verified live:
  `respond("are you conscious")` and `respond("are you aware")` both return that constant. Its framing
  is honest and agnostic -- it refuses to fake a measure -- but as a *faculty* it reads zero live access
  variables.
* The state-reading variant is narrower and does not read the brain either.
  `respond("do you have experience")` reaches `spike_chat_lake.py:929` and returned
  `access=True, binding=True, integration=True, re-entry=True (4/4)`. Those four booleans come from
  `consciousness_probe.probe_*`, which **construct fresh objects and re-run the module self-checks**
  (`consciousness_probe.py:52-62`). The brain is reporting unit-test outcomes about its module class
  definitions, not the state of its own workspace. It is a P0 honesty hazard in the making: a 4/4
  self-report that no live measurement backs.
* What *is* genuinely reportable is one bit, via the narrator: measured reply text includes
  `"Right now ignited on"`. The workspace **content** (`_access_hold["content"]`, e.g. `digit-7`) never
  reaches any reply.

12 = one bit of real reportability, on a path whose headline answer is a constant.

### binding_of_broadcast_content -- 10
`global_workspace.py:321-396` implements binding-by-synchrony (theta-gamma phase multiplexing) with a
rate control, and `:605` makes "spikes beat rate at K=2" a permanent assertion. Every one of those
functions is called only from `main()` and from `consciousness_probe.probe_binding`. Nothing live binds
multiple contents into the broadcast. Note the overlap: `COVERAGE_RUBRIC.md:28` assigns "multimodal
binding" to **Perception**; this row scores only the binding of *workspace broadcast* contents, and a
Perception auditor scoring `CrossModalBinder` is scoring a different mechanism.

---

## Capability inventory (Policy 2: what each file uniquely provides)

Deliberately NOT a markdown table: `tools/coverage_score.py::_table` treats every pipe-prefixed line
after the evidence header as an evidence row, so a second table in this file makes the scorer refuse the
whole thing (F3). One line per file -- capability, then whether it is live.

* `thalamus/global_workspace.py` -- spiking LIF ignition + shared-inhibition WTA + persistence/decay;
  binding-by-synchrony. LIVE twice (inside the arm, and inside `AccessConsciousness`).
* `thalamus/reentrant_workspace.py` -- coalitions, ignition that can FAIL, closed re-entry, subscriber
  re-route, the graded 32-unit access readout. LIVE (`Brain._reentrant`).
* `cortex/access_consciousness.py` -- VSA bundling of grown neurons into one content + the integration
  proxy. LIVE (`Brain._access`).
* `cortex/second_order_access.py` -- metacognition OF access (reliability of the access report). LIVE on
  the chat path only.
* `cortex/self_model.py` -- signal store, the confidence product, narrator introspection. LIVE.
* `cortex/generative_self_model.py` -- learned (e-prop) forward model of the brain's own state, which is
  what supplies a real self-surprise. LIVE.
* `cortex/integration_measure.py` -- perturbational complexity / normalized LZ. LIVE on sleep, every 4th
  night.
* `reasoning/gwt_reentry.py` -- re-entrant predictive completion + the no-hallucination guard. NOT live
  except as a boolean inside one chat probe.
* `reasoning/consciousness_probe.py` -- a 4-way present/absent self-check bundle. Chat path only, and it
  re-runs self-checks rather than reading the live brain.
* `docs/coverage/artifacts/consciousness_ab_probe.py` -- neuters the two non-drive-source faculties
  (`access_strength`, `hold`). One verdict per faculty; no per-consumer attribution, which is how it
  credited three consumers' divergence to one row.
* `docs/coverage/artifacts/consciousness_refute_probe.py` -- the same two faculties with every other live
  consumer logged, an isolated single-consumer counterfactual, and running-mean input neuters that
  decompose which input a decision is actually made of. Unique capability: it can tell "this faculty
  charges" from "this attribute has three readers". Should be merged into `tools/charge_probe.py` (gap 3).

**Two P2 observations, reported not acted on.** (1) There are **three** live global-workspace
instances -- `AutonomousArm.workspace`, `AccessConsciousness._ws`, and `Brain._reentrant` -- with two
different ignition behaviours (never fails vs fails 13%). P2 says exactly one mechanism should survive.
(2) `gwt_reentry.run_loop` and `ReentrantWorkspace` both implement re-entry; the live one lacks the
prototype's predictive-completion capability, so a straight retirement would lose a capability -- P2's
port-before-retire case.

---

## The rows a refuter should attack first -- WHAT THE AUDIT ASKED FOR (executed; outcome below)

`access_strength_graded` (55) and `second_order_access` (50) are the only rows above 40 and together
they carry 3.4 of the category's 28.4. Both rest on a probe **I wrote for this audit**, which is a
weaker provenance than `tools/charge_probe.py`; attack it first.

* **access_strength_graded**: the A/B is exit 0 on 89/120, but ask whether `self_model.confidence()` is
  itself consequential. The gate it feeds (`_metacog_hold`) flipped 1/120, and the arm never moved. If
  "changes an internal read" is true but nearly weightless, argue it to ~42 -- the floor of the band
  rather than the upper half. A stronger refutation: show that the 1.0 neuter is not the right
  counterfactual because `_access_ema` and the strength readout are correlated, so arm B is not "the
  pre-de-saturation brain" but a different brain.
* **second_order_access**: 7 hedges out of 60 turns is small, and the monitor's inputs are an
  availability that duplicates `ignited` plus `metacog_rel`, which is an **Executive** faculty. Argue
  that what actually varies is Executive's signal leaking through a second-order wrapper that adds
  nothing of its own, and that the row belongs at ~30 in this category.
* **the probe itself**: it patches instance attributes (`access_strength`, `hold`) rather than using a
  pinned-container subclass. I checked the one leak I found and recorded it, but a refuter should look
  for another, and should re-run both artifacts with `--rerun`.

`ignition` (30) can be attacked from the *other* side: its A/B is exit 0 with 47/60. The defence is the
two paragraphs under that row -- the pre-declared ceiling, plus a 197/200-constant eligibility -- and if
a refuter rejects both, that row goes to 61 and the category to ~31.0.

---

## Adversarial pass (refutation): 28.4 -> 26.7, both band-41+ rows fell

An independent refuter re-measured every claim above band 40, defaulting to refuted. **Both rows fell**:
`access_strength_graded` 55 -> 42 and `second_order_access` 50 -> 42. Sum 341 -> 320 over the same
twelve rows, so the category goes **28.4 -> 26.7**. Rubric rule 5: this is the instrument working, not a
regression in the brain. No row was deleted, no row was added, and the denominator is still 12.

**The forgery check passed, so neither row fell on provenance.** Both audit artifacts reproduce:
`--rerun consciousness_strength_120` gives recorded exit 0 / now 0 with observables
`(89/120, 1/120, 0/120)` unchanged, and `--rerun consciousness_second_120` gives exit 0 / now 0 with
`(0/120, 0/120, 0/120)`. The audit's `drive 60` harness control also still reproduces at 22/60, exit 0.
Nothing here was hand-written. The rows fell on **attribution**, which is the harder check and the one
`--rerun` cannot perform: a probe can reproduce perfectly and still be measuring more than one thing.

**What the refutation ran.** `docs/coverage/artifacts/consciousness_refute_probe.py`, tracked next to
the audit's probe, same construction path (`make_brain(seed=0, persist=False)`), same 120 ticks / 30
turns / same message, same arm-A-twice determinism guard, same exit-code contract. Six recorded
artifacts, every one written by `coverage_score.py record`:

    cons_refute_strength_paths_120   exit 0   the audit's neuter + every other consumer logged
    cons_refute_strength_iso_120     exit 0   ONLY the claimed consumer neutered (also --rerun verified)
    cons_refute_second_thin_120      exit 0   hold() -> False, plus the hedge-prefix test
    cons_refute_second_c1_120        exit 1   Executive input's information removed
    cons_refute_second_ws_120        exit 0   both workspace inputs' information removed
    cons_refute_second_av_120        exit 0   availability alone
    cons_refute_second_st_120        exit 1   graded strength alone

**Finding 1 -- the strength A/B was a three-consumer neuter.** `_reentrant_strength` is read at
`brain.py:2137` (the claimed consumer), `:2040` (the region-coherence access raster, which reaches
`_reentrant.ignition_threshold` through the synfire binding gate) and `:2843` (the second-order
monitor). The audit's justification for pinning it to 1.0 -- "the documented no-workspace default" --
holds only at `:2137`; at the other two sites the no-workspace default is 0.0. Under the audit's neuter
the workspace's own ignition threshold changed on 48/120 ticks and its availability on 22/120.

**Finding 2 -- the strength row's reply evidence belonged to other rows.** Isolated to the claimed
consumer, `REPLY TEXT changed` goes 29/30 -> **0/30** and the hedge count goes 8-vs-2 -> **8-vs-8**. The
confidence divergence (89/120) is untouched by the isolation, so it is genuinely this faculty's; the
reply divergence is not. The lost hedges are the second-order monitor's hold flipping (24/120) -- the
next row's mechanism.

**Finding 3 -- the confidence divergence stops early.** It changes `_metacog_rel` (48/120), which the
brain reads, so the row stays above 40. But the confidence **bucket** -- the value that reaches the
self-episode ring and the durable self-lake at `self_model.py:239,256`, i.e. the "memory write" the
rubric names -- is identical on **120/120** ticks; lifelong learn-gate admissions 0/120; hold gate
1/120; arm 0/120. Floor of the band, not the middle.

**Finding 4 -- the second-order monitor's decision is one bit.** Of its four inputs only `availability`
matters (24/120 hold flips, hedges 8-vs-2); `access_strength` and `_metacog_rel` each move it 0/120. And
availability is provably the ignited bit in this wiring. The faculty is live and does change a reply, so
it is not below 41 -- but "second-order metacognition of access" measured as a re-transduction of one
first-order bit does not hold the middle of the band.

**Two of my own attacks failed and are recorded as such:** the Executive-leakage hypothesis (0/120) and
the canned-prefix hypothesis (0 of 22 diverging turns). Both are written up under
`second_order_access`. Recording a refuter's misses is the same discipline as recording an auditor's
honest negatives; without it the next pass re-runs them.

**What I did NOT change, and why.** (a) `ignition` stays at 30. Its A/B is exit 0 at 47/60 and the band
table alone would license 61, but the audit's two defences are both correct and independently checkable
-- `COVERAGE_RUBRIC.md:99-102` pre-declares that `_route_percept_to_action` wiring cannot raise this
category, and a 197/200-constant eligibility means the A/B removed a weight, not information. A refuter's
job is not to raise scores the ceiling forbids. (b) The roster stays at 12 rows. Nothing was trimmed:
the low rows are load-bearing denominator and dropping any of them would *raise* the mean, which is the
inflation direction. (c) `global_availability` stays at 25 rather than rising to 41 on my own new
evidence -- see the double-count note under `second_order_access`.

**The sensitivity, stated instead of buried.** Three defensible readings of the same measurements:
**26.7** as scored (12 rows, chain charged once on the monitor); **26.8** if `global_availability` and
`second_order_access` are merged into one chain row (11 rows -- note the merge *raises* the number, which
is why it was declined); **28.0** if `global_availability` is independently raised to 41 on the 6/30
reply divergence, which double-counts the chain. The scored figure is the lowest of the three, which is
the correct default when the judgment call is genuinely open.

**Not re-measured, and it bounds this pass.** Every figure is seed 0, one message text, 120 ticks; the
`_metacog_explore` gate was not logged separately (the arm's 0/120 makes it moot for band purposes but
not for completeness); and the eight rows at or below 40 were left exactly as audited -- I attacked what
the rubric requires a refuter to attack, which is the claims above 40, and a low row I did not verify is
not a low row I endorse.

---

## Ranked gaps (distance_to_100 x feasibility)

1. **De-saturate the ignition that feeds action.** `_access_state["ignited"]` is True on 197/200 ticks.
   The relative-baseline salience at `brain.py:2245` was written to make access lapse, and it does not.
   Highest leverage, cheapest fix, and it would make the existing `ignition_60` A/B mean what it claims.
2. **Make `global_availability` capable of a value other than 0 or 1.** With all-or-none delivery to
   every subscriber it is arithmetically the ignited bit. Partial broadcast (per-subscriber gating on
   the learned association, which is already computed) would make availability an independent signal
   and would un-collapse `access_index` from `integration`.
3. **Fold `consciousness_ab_probe.py` and `consciousness_refute_probe.py` into
   `tools/charge_probe.py`.** Nine neuter targets across two probes now live in
   `docs/coverage/artifacts/` instead of the tracked tool, which means the evidence standard for these
   rows does not travel with the tool -- the exact defect `charge_probe.py:9-12` was created to fix
   when the probe lived in gitignored `scratchpad/`. Instrument work, not brain work, and cheap. Fold in
   the refuter's **per-consumer attribution** while doing it: a probe that prints one verdict per
   neutered signal cannot distinguish "this faculty charges" from "this attribute has three readers",
   and that distinction is what moved both scores in this category.
4. **Give the self-report something to report.** Route `_access_hold["content"]` and the live
   availability/strength into the consciousness intent, and replace `probe_*`'s self-check re-runs with
   reads of the running brain. Currently the brain answers a question about itself from unit tests.
5. **Wire `decision_margin` into `self_model.observe`** -- one of three confidence factors is a constant
   0.5 live.
6. **Make PCI vary or retire the number.** One value over 8 nights, no consumer.
7. **Port predictive completion into `ReentrantWorkspace`, then retire `gwt_reentry`** (P2).
8. **Decide what `_reentrant_strength` is allowed to mean, in one place.** Three live readers take it
   with three different fallbacks -- 1.0 at `brain.py:2137`, 0.0 at `:2040`, 0.0 at `:2843` -- and the
   `:2040` reader routes it through the synfire binding gate into the workspace's own
   `ignition_threshold` (`:2069`). So the graded access readout partly sets the ignition threshold of
   the workspace that produces it. That feedback is not obviously wrong as brain design, but it makes
   every A/B on this signal non-isolating by construction, and the disagreeing fallbacks mean
   "no workspace" is simulated as maximum access at one site and zero access at another. Added by the
   adversarial pass; it is the defect that cost this category most of its lost points.

---

## What I could not measure, and what is static-only

* **Eleven artifacts, three provenances.** `ignition_60.txt` (exit 0) and `reroute_60.txt` (exit 2) come
  from `tools/charge_probe.py`; `consciousness_strength_120.txt` and `consciousness_second_120.txt`
  (both exit 0) come from a probe written for this audit and tracked next to them; the seven
  `cons_refute_*` artifacts come from the refuter's probe, also tracked. All reproduce under
  `python tools/coverage_score.py --rerun <name>`. The audit pair is weaker evidence than the
  `charge_probe` pair purely on provenance -- same hand wrote the probe and the claim -- which is gap 3,
  and it is the reason the refuter re-derived the strength row from a probe of its own rather than
  re-reading theirs. Note what that bought: both audit artifacts reproduced byte-for-byte and the claim
  above them still fell, so `--rerun` is a forgery check and not an attribution check.
* **Ten rows sit at or below 40.** Two of them carry a `charge_probe` artifact anyway; the other eight
  have none, and rubric rule 2 does not require one below 41. Those eight were measured by instrumented
  live runs whose scripts are **untracked scratchpad files**, so `--rerun` cannot check them. Anything in
  this file outside the four artifacts is re-derivable but not currently re-derived.
* **`binding_of_broadcast_content` and `reentrant_predictive_completion` are static-only findings** --
  call-site searches, not runtime. Per rubric rule 1 that is exactly why they are scored low and
  labelled: this repo has had static dormancy claims refuted on runtime A/B before, so treat both as
  provisional.
* **Phenomenal consciousness is deliberately not a row.** `COVERAGE_RUBRIC.md:31` defines this category
  as global availability / ignition / re-entry / access / self-model -- all access-side. Adding a row
  for phenomenal experience would score 0 or `UNSCORED`, and either would assert a settled negative
  answer that no measurement here supports. Every module in the inventory carries the same agnostic
  framing and a banned-vocabulary assertion in its self-check; this file keeps that line. Nothing here
  claims phenomenal experience, and nothing here denies it.
* **The chat path is under-sampled.** `second_order_access` and `reportability_of_access_content` only
  run inside `respond()`. 60 turns at one message text is not a distribution; a different message mix
  could move both rows.
* **One prior row was re-identified, not deleted.** `ignition_novelty_source` (band 20) cited the
  `ignition` eligibility line but attached `novelty_60.txt`, i.e. it was measured with
  `charge_probe.py novelty` -- the wrong source. The `ignition` row above supersedes it with the correct
  probe (`charge_probe.py ignition`, exit 0). The `novelty` source itself is a **Perception** faculty and
  `novelty_60.txt` (exit 2, UNKNOWN) remains in `artifacts/` for whoever scores it there.

---

## Relation to the withdrawn 5.0

The prior first-pass number for this category was **5.0** over four rows: `workspace_reentry_key` 0,
`ignition_novelty_source` 20, and two `UNSCORED`. **26.7 is not a 5.3x improvement in the brain**, and
comparing the totals directly would repeat the mistake `docs/coverage/README.md:42-62` warns about. The
movement decomposes as:

* `workspace_reentry_key` 0 -> `reentry_reroute` 28: the **only genuine capability change**. The
  key-space defect that justified the 0 was fixed in v4.78 and the fix is verified working live. The 0
  was stale when this sweep started.
* two `UNSCORED` -> `global_availability` 25 and `integration_measure_pci` 22: **measurement, not
  capability.** Both were unmeasured; both are now measured and both turned out to be what the prior
  auditor suspected (availability is the ignited bit; PCI is one value). They score in the twenties
  because "live and reaching a consumer" is worth the twenties on this rubric, not because they got
  better.
* eight new rows: **roster expansion**, which `README.md:64-70` names as the largest known hole in the
  instrument -- so here is the arithmetic instead of a reassurance. The four carried-over faculties
  (`reentry_reroute` 28, `ignition` 30, `global_availability` 25, `integration_measure_pci` 22) average
  **26.25**. The eight new rows, **after refutation** (42, 42, 38, 35, 21, 15, 12, 10), average
  **26.9**. So had only the prior roster been scored the category would read 26.3, and expansion now
  accounts for **+0.45 points** of the 26.7 -- down from +2.2 before the refutation, because the two rows
  the expansion added at 55 and 50 are the two that fell. Roster expansion is where a refuter trimming
  the roster would aim; on these numbers there is almost nothing left there to trim, and trimming a
  *low* row would raise the mean.

A like-for-like read on the three comparable faculties that were 0 or unmeasured is `0, 0, 0` ->
`28, 25, 22`. A downgrade would have been just as reportable; this happens to be an upgrade, and it is
mostly the instrument catching up to code that shipped two commits ago. The one thing here that is *not*
catch-up is `access_strength_graded` at **42**: that is the first recorded evidence in this category that
a consciousness faculty changes something the brain itself reads -- and the refutation narrowed what that
something is. It is `_metacog_rel`, on 48 of 120 ticks, and it goes no further: not the memory write, not
the learn gate, not the arm, not the reply.
