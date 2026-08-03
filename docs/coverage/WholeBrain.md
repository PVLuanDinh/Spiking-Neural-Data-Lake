# Coverage evidence -- WholeBrain

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

The category claims four things (`COVERAGE_RUBRIC.md:38`): inter-region wiring, oscillatory
coordination, homeostasis, self-improvement. The roster below is grouped in that order.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| homeostatic_drive_to_action | 40 | src/snn_data_lake/cortex/brain.py:1531#_route_percept_to_action; src/snn_data_lake/cortex/brain.py:1677#_drive_for | wb_drive_60.txt | opus-5 WholeBrain REFUTER (was 65) / rubric ceiling COVERAGE_RUBRIC.md:99-102 + wb_refute_onset_60.txt | 2026-07-30 |
| circadian_wake_sleep_gate | 35 | src/snn_data_lake/cortex/brain.py:2655#_gate; src/snn_data_lake/cortex/brain.py:2709#tick | - | opus-5 WholeBrain REFUTER (was 40) / unrecorded in-process run, recordable and not recorded | 2026-07-30 |
| cpg_rhythm_to_action_preference | 35 | src/snn_data_lake/brainstem/autonomous_loop.py:676#_cpg_bias; src/snn_data_lake/brainstem/autonomous_loop.py:665 | - | opus-5 WholeBrain REFUTER (was 40) / unrecorded in-process run, recordable and not recorded | 2026-07-30 |
| interoceptive_caution_to_selection | 35 | src/snn_data_lake/brainstem/autonomous_loop.py:955#_interocept; src/snn_data_lake/brainstem/autonomous_loop.py:748#_select; src/snn_data_lake/brainstem/autonomous_loop.py:1355#step | - | opus-5 WholeBrain REFUTER (was 40) / unrecorded in-process run, recordable and not recorded | 2026-07-30 |
| cross_region_coherence_ignition_gate | 30 | src/snn_data_lake/cortex/brain.py:2159#_c1_tick; src/snn_data_lake/substrate/region_coherence.py:132#RegionCoherence | - | opus-5 WholeBrain REFUTER (was 35) / outcome measured INVARIANT at every gain incl. 0.0 | 2026-07-30 |
| neurogenesis_growth | 35 | src/snn_data_lake/cortex/brain.py:2369#_sleep_tick; src/snn_data_lake/cortex/brain.py:2408#_sleep_tick | - | opus-5 WholeBrain auditor / live run, 60 ticks (grown_units) | 2026-07-30 |
| connectome_spike_routing | 30 | src/snn_data_lake/cortex/brain.py:2100#_c1_tick; src/snn_data_lake/cortex/brain.py:2877#state | - | opus-5 WholeBrain auditor / live run, 60 ticks + reader grep | 2026-07-30 |
| gwt_reentry_to_action_drive | 30 | src/snn_data_lake/cortex/brain.py:1612#_route_percept_to_action; src/snn_data_lake/cortex/brain.py:1693#_route_percept_to_action | wb_reroute_60.txt | opus-5 WholeBrain auditor / tools/charge_probe.py reroute 60 + eligibility measured from inside | 2026-07-30 |
| criticality_edge_of_chaos_regulation | 30 | src/snn_data_lake/cortex/brain.py:2500#_enter_sleep; src/snn_data_lake/cortex/brain.py:2421#_sleep_tick | - | opus-5 WholeBrain auditor / live run, sigma sampled | 2026-07-30 |
| wholistic_integration_capstone | 22 | src/snn_data_lake/cortex/brain.py:354#__init__ | - | opus-5 WholeBrain auditor / live run at birth | 2026-07-30 |
| agency_controllability_monitor | 15 | src/snn_data_lake/brainstem/autonomous_loop.py:1658#step; src/snn_data_lake/cortex/agency_monitor.py:2#module | - | opus-5 WholeBrain auditor / live run, agency() sampled at 60 ticks | 2026-07-30 |
| homeostatic_plasticity_stack | 10 | src/snn_data_lake/cortex/brain.py:246#__init__; src/snn_data_lake/cortex/interaction_learning.py:200#_deposit; src/snn_data_lake/cortex/brain.py:2394#_sleep_tick | - | opus-5 WholeBrain auditor / live instrumented run, pre_gain/stabilize/relax counted | 2026-07-30 |
| gated_self_modification | 10 | src/snn_data_lake/cortex/brain.py:2647#_maybe_autonomous_self_build; src/snn_data_lake/cortex/brain.py:2696#_enter_wake | - | opus-5 WholeBrain auditor / live run, _selfbuild_last after 2 nights | 2026-07-30 |
| knowledge_acquisition_web_ingest | 12 | src/snn_data_lake/reasoning/autonomous_enrich.py:40#module; src/snn_data_lake/reasoning/spike_chat_lake.py:1572#boot | - | opus-5 WholeBrain auditor / caller grep over brain.py + cockpit.py | 2026-07-30 |
| brain_native_selfbuild_proposer | 5 | src/snn_data_lake/cortex/brain.py:1793#propose_improvements | - | opus-5 WholeBrain auditor / live run, propose_improvements(budget=6) | 2026-07-30 |
| region_faculty_charge_registry | 0 | src/snn_data_lake/substrate/region_wiring.py:34#module; src/snn_data_lake/substrate/region_wiring.py:59#charge_all | - | opus-5 WholeBrain auditor / live run, region_wiring.wired()/coverage() | 2026-07-30 |
| godel_machine_selfbuild | 0 | src/snn_data_lake/reasoning/godel_machine.py:162#attempt_on_repo | - | opus-5 WholeBrain auditor / repo-wide call-site search | 2026-07-30 |

**Mean over the 17 rows: 374 / 17 = 22.0.** **Zero faculties above 40.** Zero UNSCORED.

**REFUTED, 2026-07-30, by an independent adversarial pass. The auditor's number was 24.6 with one
band-41+ row; that row fell, and four more moved down. See "Adversarial pass" below for the before/after
and for the two attack lines that FAILED (the artifact is real and it reproduces; the flip rate is not
degenerate).** The auditor's own note on rule 5 still holds and is worth keeping: this is lower than the
withdrawn baseline's 29 because the roster went from 3 rows to 17 and the bar was then applied to all of
them. Two downgrades are substantive rather than procedural -- the plasticity stack and the sleep
writeback -- and one prior claim the auditor expected to confirm turned out wrong in the other direction.

## Adversarial pass (refuter), stated first because it changes the headline

**Before 24.6, after 22.0. One band-41+ claim existed and it fell.** Five rows moved: the drive charge
65 -> 40, three unrecorded in-process rows 40 -> 35, and the coherence gate 35 -> 30. Nothing was
deleted; every row and both of the auditor's artifacts are still here, plus one new recorded artifact.

**1. The refutation that carries the number: `COVERAGE_RUBRIC.md:99-102` pre-declares that this credit
cannot be given to this category.** Verbatim: *"Wiring a faculty into `cortex/brain.py::_route_percept_to_action`
moves a three-way arm decision (ENGAGE / RELEASE / HOLD). It can raise Executive, Affect, SpatialTemporal
and the action-relevant part of Perception. It **cannot** raise Language, Imagination, Gf, Consciousness
or WholeBrain -- crediting those to a wiring wave is the theater P0 forbids."* Both of the band-65 row's
citations are **inside that function**: `_route_percept_to_action` is defined at `brain.py:1329` and the
next `def` is `_w8_tick` at `brain.py:1631`, so `:1502` (`elig["drive"]`) and `:1600` (`drive_a`) are its
body, and the vector they build is handed to the arm at `:1616` (`self.brain.arm._ext_drive = ext`). The
faculty is real, the wire is real, the A/B is real -- and it is exactly the route the rubric excluded
from this category **in advance, in a section headed "stated before scoring so nobody scores around
them."** The auditor's file, which engages the rubric closely everywhere else, never cites this clause.
Band 40 is the ceiling that clause imposes: the row keeps the best evidence in the file and stops being
the thing that lifts the category.

**2. Ownership, and the double-count the auditor half-found.** The auditor disclosed an overlap with
`Executive.md:9` claiming band 61 for `drive_arbitration` on the same neuter. **That disclosure is now
stale in a way that matters:** `Executive.md:9` is `task_set_sr_mapping` (61, `taskset_60.txt`), and
`Executive.md:167-172` records why -- `drive` was re-scoped out of Executive as *"the dominant homeostatic
need -- interoceptive motivation, an Affect faculty"*. So the two-categories-one-A/B problem did not
resolve in WholeBrain's favour; it resolved by a peer auditor assigning the faculty to **Affect**, which
is one of the four categories the rubric says an arm-decision wire **can** raise. There is a home for
this charge where it is creditable, and it is not this file.

**3. `wb_drive_60.txt` is a re-record of the sweep's shared control.** Its stdout is byte-identical to
`drive_60.txt` (same command `tools/charge_probe.py drive 60`, same 9 learned pairs, same 0.0495 weight,
same 0/60 and 22/60), which is used as the harness control by `Memory.md:61`, `Social.md:92`,
`Imagination.md:47` and `Executive.md:107`. Re-recording it at a newer commit is legitimate and it does
reproduce -- but "it reproduces the known-good control exactly, so the harness is not degenerate" is not
independent corroboration of the claim when the control **is** the claim's only run.

**4. Two attack lines FAILED, and the refuter reports that too.** `coverage_score.py --rerun wb_drive_60`
was re-executed here: recorded exit 0 / now 0, observables `(('0','60'), ('22','60'))` / now identical,
`reproduces`. Not a hand-written transcript. And the flip rate is not degenerate -- see the next item,
which found *more* charge than the artifact showed.

**5. New recorded evidence: `wb_refute_onset_60.txt` (kind ab, exit 0), from
`docs/coverage/artifacts/wb_refute_probe.py drive 60`,** which imports `charge_probe._PinnedWeights`
rather than copying it, so the neuter is identical by construction. It answers the three questions
`charge_probe` does not:

- **Self-control A vs A: 0/60, 0/60, 0/60** on stop, trajectory and action. `charge_probe` runs no
  second A, so nothing in `wb_drive_60.txt` ruled out the 22/60 being nondeterminism. It is not.
- **The `22/60` is ONE event.** First differing tick **39**, equal->differ **onsets 1**, differ->equal
  re-convergences **0**. 60 - 39 + 1 = 22. The trajectory splits once and never re-converges, so the
  count is a divergence-**onset** artifact of the `map_position` observable, not 22 independent flips.
  This independently re-derives what `Executive.md:113-116` measured for `taskset`, and it explains why
  `taskset_60.txt` and `wb_drive_60.txt` carry **byte-identical** observable lines from two unrelated
  sources: any perturbation that first bites at tick 39 prints 22/60. The same figure appears in four
  category files and is one piece of arithmetic, not four effect sizes.
- **The selected action DOES change: 17/60 ticks, 2 onsets, first at tick 39**, over 7 distinct actions
  including ENGAGE and HOLD. `charge_probe`'s observables are `arm_stopped` and `map_position`
  (`brain.py:2785`, the allocentric map dead-reckoned from the real end-effector stream at
  `autonomous_loop.py:1595`); it never looks at `_last_res["action"]`. **So the recorded artifact
  understated the faculty**: this is a decision change, band 61-80 in kind. It is capped at 40 here
  anyway, by category and not by effect size, and that distinction is the whole finding. If the sweep
  adjudicates the scope question the other way, the row's evidence supports 61 in Affect.

**6. Four of these 17 rows are four additive terms in one argmax.** `autonomous_loop.py:654-677`:
`bias` takes the interoceptive term, `td` takes `_ext_drive` (which carries **both** `drive` and
`reroute`) at `:662`, then the CPG phase at `:665`, and one `max(ACTIONS, key=...)` at `:675` consumes
all of it. That is not automatically one faculty -- these are independent sources with separately learned
weights, and the observable does discriminate between them (`drive` 22/60 vs the STN brake's 7/60 vs
`pfcbias`'s 57/60 in `Executive.md:86-91`) -- but the auditor ran no separability test, so for
`cpg_rhythm_to_action_preference` and `interoceptive_caution_to_selection` the fingerprint question that
cost Executive 9 points is **open, not closed**. It is also why 41% of this file's band mass sits on one
arm-preference vector, which is the shape the rubric's ceiling clause exists to catch.

**7. The "instrument gap" premise is false, so the three rows resting on it move 40 -> 35.** The auditor
holds three rows at exactly 40 on the stated ground that no recordable A/B *can* exist for them --
*"no band above 40 is reachable for them today no matter what the code does"* and *"`tools/**` was
off-limits this session, so they are recorded as work, not done"*. `coverage_score.py record` runs **any
command**, and on the same day six peer auditors recorded `kind: ab` artifacts from probes they wrote
into `docs/coverage/artifacts/` -- the directory this auditor was free to write:
`affect_ab_probe.py` (-> `affect_cortisol_caution_60.txt`, exit 0, and it includes the A-vs-A control
`charge_probe` lacks), `executive_ab_probe.py`, `consciousness_ab_probe.py`, `social_tom_ab_probe.py`,
`spatiotemporal_pathway_ab.py`, `imagination_ab.py`, `gf_reply_ab.py`. This refuter's own probe above is
the seventh. So the ceiling those rows sit on was self-imposed, not instrumental; the measurement was
available and not taken. Under `Executive.md:31-33`'s scale -- which this file should be comparable to --
**35** is "consumed into the decision path, counterfactual not isolated in a reproducible run" and **40**
is reserved for a *measured* decision change held at a genuine instrument ceiling. These three are the
former. The claim that they are "probably 61-80" is also contradicted by the nearest measured analogue in
the tree: `Executive.md:93-99` neutered `plan_sequence`, a ~0.14 additive term in the **same** `td`
vector, and got **0/60 actions and 0/60 trajectory** -- fully live, fully consumed, behaviourally inert.
A +/-0.23 nudge to `td` is not evidence the argmax moved.

**8. `cross_region_coherence_ignition_gate` 35 -> 30 on the auditor's own measurement.** Their prose
states that `region_coherence.py:132-137` records ignition firing on 100% of wake ticks at **every** gain
in [0.0, 1.0] including 0.0, and concludes "a moved threshold whose outcome is invariant is not a
charge". Agreed -- and that is `Executive.md:34`'s definition of **30** ("live and consumed, and measured
**not** to change any decision: only an internal scalar moves"), not of 35 (a *thin* measured effect).
The band should follow the sentence.

**What survived unchanged.** Every row at 30 and below, and the reasoning behind them. Spot-checked and
confirmed: `criticality_edge_of_chaos_regulation` at 30 is consumed beyond telemetry (`_criticality`
feeds `cpl` at `brain.py:2331`, not only `state()` at `:2680`), so the row is not padding;
`wholistic_integration_capstone` at 22 was considered for a rule-based cut to <=20 -- band 21-40 says
"on the wake/sleep tick" and a once-at-birth call is not on the tick -- and left alone, because it is
imported and executed by a live path and 22 is already the bottom of its band. The four low
self-improvement rows (0, 5, 10, 10) were checked for the reverse defect, one cause split across rows to
pad the roster downward; they are three distinct modules with distinct capabilities, and merging them
would *raise* the mean, so the split stands.

## The instrument gap as the auditor stated it -- premise refuted, see item 7 above

`tools/charge_probe.py` is the only *tracked* runtime A/B in `tools/`, and it can neuter exactly
one thing: a key of `Brain._drive_w`. Of WholeBrain's four sub-areas, **only homeostasis has a
faculty that reaches action through `_drive_w`.** Inter-region wiring (apart from the GWT re-entry
source), oscillatory coordination and self-improvement have no `_drive_w` key at all. The auditor
concluded from this that no band above 40 was reachable for them; **that conclusion does not follow**
-- `record` executes any command, and an auditor-written probe under `docs/coverage/artifacts/` is a
tracked A/B, as six peers demonstrated the same day. What remains true and useful is the per-row
statement of what a probe would have to pin, which is a ready-made work list rather than an excuse:

- `circadian_wake_sleep_gate`: force `clock.consolidation_gate()` to a constant and compare tick kinds.
- `cpg_rhythm_to_action_preference`: pin `AutonomousArm.cpg = None` (the `_cpg_bias` guard at
  `autonomous_loop.py:560` already handles it) and diff the selected action.
- `interoceptive_caution_to_selection`: pin `AutonomousArm.intero = None` (guard at `:864`) and diff
  the act/hold decision.

`tools/**` was indeed owned by other agents this session, so no *extension of `charge_probe`* was
written -- but none was needed. Each of the three pins above is expressible in an auditor-owned probe
under `docs/coverage/artifacts/`, which is where `wb_refute_probe.py` lives. They are work not done,
not work that could not be done.

## The charge as the auditor stated it (band 65), and why 40 replaced it

Refuted on scope, not on substance -- items 1-3 and 5 above. Everything below is the auditor's text and
it is accurate; the recorded evidence in it is real and, per item 5, it *understated* the faculty. What
it cannot do is raise this category.

`homeostatic_drive_to_action` was the only band-41+ claim here. `wb_drive_60.txt` records exit 0 with
**22/60 arm-trajectory changes and 0/60 act/hold changes**, which reproduces the known-good control at
`COVERAGE_BASELINE.md:90` exactly, at a later commit, from a separate run -- so the harness is not
degenerate. `coverage_score.py --rerun wb_drive_60` was then executed and reported `reproduces`
(recorded exit 0 / now 0; recorded observables `(('0','60'), ('22','60'))` / now identical), so the
artifact survives the forgery check as well as the score. Supporting live measurements over 30 ticks: `HomeostaticDrives.deficit()` returned 30
distinct triples, the dominant drive rotated across all three modelled needs (energy / curiosity /
social), the dominant need's deficit spanned 0.05-0.30+, and 8 `(drive, action)` values had been
learned from dopamine. So the signal is graded and live, not a constant wearing a signal's clothes.

**The overlap** (stale as written -- see item 2 above; `Executive.md:9` is now `task_set_sr_mapping` and
`drive` was re-scoped to Affect, so the pair to adjudicate is WholeBrain vs **Affect**)**:** the auditor
wrote: `Executive.md:9` claims band 61 for `drive_arbitration` against the same neuter. It is
the same pin on the same key -- one A/B is being credited in two categories. I am flagging it rather
than deleting my row, because silent omission from a denominator is the inflation route the rubric
names and because the neuter is arguably *more* specific to this claim than to Executive's: pinning
`_drive_w["drive"]` and clearing `_drive_value` kills precisely the two terms fed by
`HomeostaticDrives` (`brain.py:1502` undirected urgency, `brain.py:1600` the drive-conditioned value),
and nothing else. If the sweep adopts a single-count rule, this is the pair to adjudicate; the honest
resolution is a decision about scope, not a quiet edit to either file.

## Rule-4 zeros, all four re-verified at runtime

`region_faculty_charge_registry` is **0**, and rubric rule 4 names this shape verbatim ("a registry with
no registrations"). Verified live, not from the docstring: on a `make_brain(seed=0, persist=False)`
instance, `region_wiring.wired()` returned `[]` and `coverage()` returned
`regions_wired 0 / regions_total 17, n_wired_faculties 0`, with all 17 anatomical regions in
`unwired_regions`. `charge_all()` therefore iterates an empty list every daemon cycle.

`godel_machine_selfbuild` is **0**: a repo-wide search for `attempt_on_repo` returns its own `def` and
its own docstring, three backlog/doc mentions, and nothing else -- no `.py` in `src/`, `tests/`, `tools/`
or the root calls it.

`homeostatic_plasticity_stack` is **10, and this is a new downgrade found by instrumenting rather than
reading.** `brain.py:225-236` documents the stack as wired onto the live R-STDP learner and quotes an
A/B (drive 18.97 -> 3.01, max|w| 6.64 -> 0.98). The attachment is real -- `brain._ixn.stack` is not None
on a live brain. But over 12 wake + 12 sleep ticks I counted **`pre_gain` 0 calls, `stabilize` 0 calls,
zero interaction-learner weights changed, and zero Benna-Fusi cascades created.** The cause: the hooks
fire inside `SpikingInteractionLearner.learn_turn`, whose only caller is `LiveBrain.teach`
(`live_brain.py:475`), whose only callers are `live_brain.py`'s own self-check at `:850` and `:862`.
Neither `Brain.tick()` nor `Brain.respond()` nor the cockpit reaches it. So the stack is charged in a
gate run and never on the brain's autonomous path. Two corrections fall out: the in-code A/B at
`brain.py:232-236` is a comment, not a recorded artifact, so it cannot support a band above 40 either
way; and the comment's citations to `interaction_learning.py:206/:229` have drifted -- the real hook
lines are **`:200` (`pre_gain`) and `:223` (`stabilize`)**.

The **sleep synaptic-consolidation writeback** (`brain.py:2304-2307`) is the same defect's downstream
half and is folded into that row rather than scored separately, because splitting one cause into two
rows would be padding even in the downward direction. Measured: `relax()` was called **84 times (7
neurons x 12 sleep ticks) and returned None every single time**, so `_plastic_relaxed` stayed 0 and
nothing was ever written back. `_BennaFusiConsolidation.relax` returns None when `self._cas[i] is
None`, and the cascade is created lazily on the first deposit -- which never happens. The comment at
`brain.py:2298-2299` ("they must be written back or the rest phase does nothing") is correct about the
mechanism and describes a rest phase that currently does nothing.

## What the recent connectome fix actually bought

The lead I was given says `connectome_router`'s return "was discarded until fixed this week (now
consumed at `brain.py:2010`)". Verified, and the honest band is still **30**: `_connectome_propagation`
is written at `brain.py:2015` and its **only** reader anywhere in `src/` is `brain.py:2752`, inside
`state()`. Rubric band 21-40 is exactly "output reaches only telemetry, `state()`, or a log". The fix
turned a dropped return into a telemetry key, which is real progress against a dropped return and is
not a behaviour. It is live and graded -- 6 distinct values over 60 ticks (0.0, 0.075, 0.225, 0.45,
0.6, 0.7) -- so it is not the constant-in-signal's-clothing defect; it simply has no consumer.
`connectome_reverberation` is not scored at all: the code labels it structural (`brain.py:2018-2023`),
it is computed once, and I measured it at the constant 0.25, which matches that label.

`cross_region_coherence_ignition_gate` is **30** (the auditor wrote 35; refuted in item 8 above, on the
auditor's own measurement) on mixed evidence pointing in opposite directions. In
favour: over 60 ticks `_region_coherence_val` took 30 distinct values and
`_reentrant.ignition_threshold` took **12 distinct values in 0.361-0.399 against a base of 0.55** -- so
the coherence signal demonstrably writes a number the workspace itself later reads, which is the shape
of band 41-60. Against: the module's own calibration block (`region_coherence.py:132-137`) states that
in the sessions measured the workspace ignited on **100% of wake ticks at every gain in [0.0, 1.0],
including gain 0.0**, i.e. no value of the constant changed what the brain did. A moved threshold whose
outcome is invariant is not a charge, and I have no A/B, so it cannot exceed 40 regardless.

`gwt_reentry_to_action_drive` is **30**, and the artifact is an honest negative I am attaching on
purpose. `wb_reroute_60.txt` records exit 2 (UNKNOWN) with 0/60 act/hold and 0/60 trajectory changes.
The probe reports UNKNOWN rather than a negative verdict because `reroute` is an undirected source with
no learned value table, so it cannot establish eligibility from outside -- but I established it from
inside: over 30 ticks `_reroute_bias` was **non-zero on 26/30 ticks across 9 distinct values**, and
`arm._ext_drive` was a populated vector on 27/30 ticks with 15 distinct values. So the GWT re-entry
source does fire, does reach the per-action drive vector (not merely telemetry), and neutering it
changes nothing measurable. That is "consumed but not decisive" with the eligibility question closed,
which is a stronger and worse finding than the probe's own UNKNOWN. Attaching the exit-2 artifact also
stops a later attempt to re-badge this run as a charge.

## Self-improvement: the avenue is armed, empty, and off

Three findings compound, and the order matters.

`gated_self_modification` is **10**. `_maybe_autonomous_self_build` returns at `brain.py:2545` unless
`NORD_SELFMOD == "1"`, and nothing in the tree sets that except `self_build_loop.py`'s own self-check.
Verified live: after 2 completed nights `_selfbuild_last` was None. A strict reading of rule 4 makes
this 0; I scored 10 instead because the guard *can* fire by operator action and the daemon documents
the flag, so "built, not live" is the more accurate band than "absent". Stating both readings so the
choice is visible rather than buried.

`brain_native_selfbuild_proposer` is **5**, and it is why the previous paragraph matters less than it
looks: `propose_improvements(budget=6)` returned **`[]`** on a live brain. So even with the flag set,
the loop would be handed nothing to sandbox-verify. This is an honest null, and the proposer's own
design note records the cause (cognition thresholds encode global calibration that a per-instance
running baseline cannot reconstruct). The self-writing-code avenue is therefore not merely gated off;
its generator is empty.

`neurogenesis_growth` is **35** and is the one self-improvement avenue that actually runs: measured
`grown_units` 0 -> 3 over 60 ticks, sleep-gated as documented. Its output reaches
`self_model.observe(grown_units=...)` at `brain.py:2358`, which is a self-model term -- band 41-60's own
example -- but with no A/B it caps at 40, and I held it at 35 because of a separate confirmed defect:
**`neurogenesis_engine.py` contains no persistence path of any kind.** A search of the module for
`def save`, `def load`, `state_path`, `json.dump`, `np.savez`, `torch.save` and `pickle` returns zero
matches, and no caller persists `grown_units` either. Growth is real and does not survive a restart.

`knowledge_acquisition_web_ingest` is **12**. P0 names knowledge acquisition as a self-improvement
avenue, so omitting it would understate the category, but it is not on the brain's path: neither
`cortex/brain.py` nor `substrate/cockpit.py` references `spike_chat_lake` or the ingest gate at all.
Its live entrypoints are separate processes (`autonomous_enrich.py --forever`) and offline corpus
trainers. The sleep tick does write durable knowledge (schema induction, `brain.py:2420-2448`), but
that is internal consolidation of episodes the brain already had, not acquisition, and it belongs to
Memory's roster rather than being double-counted here.

`wholistic_integration_capstone` is **22**: `WholisticLoop` runs exactly once, at birth
(`brain.py:354`), and `_integration_competence` / `_integration_gain` / `_integration_retention` reach
only `state()`. Measured at birth: competence 0.8333, gain 0.1667, `_integration_ok` True. A once-per-
lifetime readout that nothing consumes is a birth diagnostic, which is what the surrounding comment
already calls it.

`agency_controllability_monitor` is **15** for two independent reasons, either of which alone would put
it below the telemetry band. First, zero readers: `observe()` runs every tick at
`autonomous_loop.py:1561` and nothing in `src/` reads `agency()`, `controllability()`, `status()` or
`snapshot()`, and no `agency` key reaches `state()`. Second, and this is the part I measured rather
than inherited: **`agency()` returned `n_scored: 0` after 60 live ticks** -- 6 channels seen, maximum
per-channel `n` of 10 against `warmup=120` -- so the readout is not merely unread, it is empty. The
module's own status block (`agency_monitor.py:2-19`) reaches the same conclusion from a 600-tick run and
puts the requirement at roughly 4,000 ticks. Its citation of the call site as `autonomous_loop.py:1498`
has drifted; the call is at `:1561`.

## A lead that was wrong, and the correction is the point

I nearly filed `theta_gamma_code` as a dormant oscillatory-coordination faculty on the strength of a
grep showing its only importer to be `reasoning/rose_language.py`, plus a second grep showing
`rose_language` had no callers. **The second grep was wrong.** `cortex/brain.py:3419` imports
`RoseLanguage`, via `_rose_serve()`, which is called on two live paths: the sleep tick's nightly refit
(`brain.py:2450`) and `respond()` when a message starts with `parse` or `diagram` (`brain.py:2899`). So
the module is live-on-demand, not dormant. I then excluded it from this roster on different grounds --
a phase-nesting code inside a language pipeline is Language's faculty, not whole-brain oscillatory
coordination, and borrowing it would be padding of the worst kind. Recording the near-miss because it
is the exact failure mode `COVERAGE_RUBRIC.md:65-68` warns about, it happened inside this audit, and
the tool that produced the wrong answer was a shell grep whose output was being filtered.

## Capability inventory (Policy 2)

- `substrate/region_wiring.py` -- the per-cycle charge-adapter registry. Unique capability: a plug
  point that adds a faculty to the daemon loop without editing the daemon. Currently zero registrations.
- `thalamus/connectome_router.py` -- signed, delayed LIF spike propagation over the `connectome.py`
  schematic; reverberation over signed loops. The only module that makes the wiring graph dynamic.
- `substrate/region_coherence.py` -- Kuramoto phase order + a tractable integration proxy over live
  per-region activity; the only cross-region synchrony measure with an ignition-gating output.
- `cerebellum/synfire_chain.py` -- feedforward synchrony survival; supplies the all-or-none threshold
  that turns continuous coherence into a binding event.
- `hypothalamus/circadian_clock.py` -- the only oscillator that gates which tick body runs.
- `brainstem/cpg_oscillator.py` -- Matsuoka CPG; the only rhythm generator with a path into the
  action-preference vector.
- `plasticity/criticality_regulator.py` -- the only closed-loop branching-ratio regulator (reuses
  `cortex/neuronal_avalanches` for the measurement).
- `hypothalamus/homeostatic_drives.py` + `medulla/interoceptive_arousal.py` -- the two homeostasis
  signals with proven or measured paths into selection.
- `plasticity/plasticity_rule_stack.py` (over `homeostatic_plasticity_stack.py`) -- composes seven
  plasticity mechanisms behind one `pre_gain`/`stabilize`/`relax` contract. Unique, and currently
  uncharged on the autonomous path.
- `cortex/neurogenesis_engine.py` -- growth of units that predict the brain's own next perception. No
  persistence.
- `reasoning/self_mod_harness.py` + `self_build_loop.py` + `godel_machine.py` -- sandbox-verify,
  ratchet, byte-exact rollback, and the immutable safety kernel. The safety machinery is the unique
  capability here and it is intact; the driver above it is off and its proposer is empty.
- `cortex/agency_monitor.py` -- per-channel sustained-controllability detection with hysteresis;
  genuinely distinct from `cerebellum/forward_model.py`'s transient error, and worth keeping unwired
  rather than wiring on 129 observations of a single channel.

## Gaps, ranked by distance_to_100 x feasibility

1. **Record the three A/Bs that were skipped, in an auditor-owned probe.** Not "extend `charge_probe`"
   -- that framing is what produced three unrecorded rows (item 7). A generic "pin this attribute to a
   constant and diff `arm_stopped` / `map_position` / `_last_res["action"]`, with an A-vs-A control"
   probe under `docs/coverage/artifacts/` settles `circadian_wake_sleep_gate`,
   `cpg_rhythm_to_action_preference` and `interoceptive_caution_to_selection` in one file;
   `wb_refute_probe.py` is 100 lines and already does it for one source. The refuter's prior is that
   they land at 30-40, not 61-80, because the nearest measured analogue in the same `td` vector
   (`plan_sequence`, `Executive.md:93-99`) is inert -- but that is a prior, and the point of the probe
   is that nobody has to take it. Report the separability fingerprints while there, since item 6 leaves
   that question open. Largest movement per unit of work in this category, in either direction.
2. **Adjudicate the scope question the refutation exposes.** The `drive` A/B is creditable in Affect and
   barred in WholeBrain by `COVERAGE_RUBRIC.md:99-102`; `Affect.md` is still a single UNSCORED stub, so
   the charge is currently credited **nowhere** and the row here sits at the ceiling with its artifact
   attached. That is the ceiling working, not an omission -- but the sweep should place the credit
   deliberately rather than by whichever auditor writes last. The general form is a rubric-level fix:
   the ledger's own "roster padding" hole (`docs/coverage/README.md:64-70`) and this cross-category
   ownership hole are the same missing artifact, a **tracked roster that says which category owns which
   faculty.**
3. **Make the interaction learner deposit on the brain's own path**, or retire the plasticity-stack
   attachment. Right now `brain.py:246` reads as a wire and measures as nothing. Either is honest;
   leaving it is the P3 violation.
4. **Give the connectome propagation a consumer.** It is live and graded and reaches only `state()`.
5. **Persist grown units.** Neurogenesis is the one self-improvement avenue that runs, and it resets
   every restart, so no amount of uptime accumulates.
6. **Fix the self-build's empty proposer before arming the flag.** Turning `NORD_SELFMOD` on today
   changes nothing, because `propose_improvements()` returns `[]`.
7. **Register something in `region_wiring`, or retire it.** A registry with no registrations has been
   read as a working plug point for its whole life.
8. **Long-horizon measurement.** The category's headline claim -- that the brain improves itself over
   uptime -- is not measurable in a 60-tick audit at ~2.3 s/tick, and the two faculties that would show
   it (`agency_controllability_monitor` at ~4,000 ticks, `gated_self_modification` at 8-night cadence)
   both sit past that horizon. I did not add an UNSCORED row for it, because it would double-count
   faculties already scored above; the honest statement is that this audit bounds the short horizon
   only.
