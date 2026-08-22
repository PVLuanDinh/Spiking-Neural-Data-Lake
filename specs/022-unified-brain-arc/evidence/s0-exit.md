# S0 (US1) exit certification — specs/022 T039

Certified on `main` at the commit that lands this file. Every number below was produced by the
command printed beside it, on the pinned measurement host (`LUAN_AND_ANNIE`), and re-derivable by
running that command again. **This file certifies what was measured, including what did not pass.**

> **AMENDED 2026-08-22 by T039f — see §8. The sections below are left EXACTLY as certified**,
> because a certification's numbers are the record of what was true at the commit that landed it and
> editing them in place would destroy the only evidence that the gap existed. §8 states what has
> moved since; where §8 and an earlier section disagree, §8 is current.

## Verdict

| # | Criterion (spec.md) | Measured | Status |
|---|---|---|---|
| 1 | Integer floor of charged regions met, identity pinned by FR-015 | 7 of 16 CHARGED, floor recorded at 7 | **MET as an integer; identity 5 of 6 — see §1** |
| 2 | SC-001: P6 faculty-to-faculty reads > 0, neuter-verified | 0 | **UNMET — see §2 and §6a** |
| 3 | SC-002: wake-tick p50/p95 ratchets vs the T001 baseline | p50 162.156 vs 183.517 | GREEN |
| 4 | SC-003: G5 green — identical spike trains, RNG stream positions asserted | all four legs | GREEN |
| 5 | SC-002: lake rebuild-from-log proves logical-dump equality | rebuild equality + planted red | GREEN |

## 1. Region charge floor

```
python -m snn_data_lake.substrate.region_charge_gate --assert
```

```
region charge: 7 of 16 CHARGED (7 evidence records; 0 exempt, ceiling 0)
  basal_ganglia CHARGED      amygdala        REACHABLE-UNCHARGED    insula     UNREACHABLE
  cortex        CHARGED      brainstem       REACHABLE-UNCHARGED    olfactory  UNREACHABLE
  hypothalamus  CHARGED      cerebellum      REACHABLE-UNCHARGED    pons       UNREACHABLE
  medulla       CHARGED      claustrum       REACHABLE-UNCHARGED
  midbrain      CHARGED      epithalamus     REACHABLE-UNCHARGED
  neuromodulation CHARGED    hippocampus     REACHABLE-UNCHARGED
  thalamus      CHARGED
  floor holds: 7 >= 7
```

**Day one (T022) was 0 of 16.** Seven charges, each carried by a measured replay-twice attribution
in `tools/region_charge_evidence.json`, every one `diverged=True` — a non-diverged record charges
nothing, which is measurement rather than declaration:

| attribution | charges | landed |
|---|---|---|
| `raphe_serotonin.five_ht_out → evidence_accumulation` | neuromodulation | T025 |
| `interoceptive_arousal.arousal_out → interoceptive_in` | medulla | T026 |
| `homeostatic_drives.drive_deficit_out → interoceptive_in` | hypothalamus | T026 |
| `astrocyte_modulation.glial_gain_out` | neuromodulation | T027 |
| `percept_surprise.surprise_out → brain.salience_surprise_in` | cortex | T028 |
| `dopamine_nuclei.da_out → r_stdp.dopamine_in` | midbrain, basal_ganglia | T029 |
| `reentrant_workspace.reroute_out → brain.reroute_in` | **thalamus**, cortex | **T039a** |

**THE INTEGER FLOOR IS 7 AND IS RECORDED IN `tools/region_charge_floor.json` BY THIS COMMIT.** It
is a floor, not an arrival: from here the gate asserts `charged_count >= 7` on every run, and the
red is reachable — bumping the artifact to 8 returns
`REGION CHARGE GATE RED: 7 charged < floor 8`, rc=1 (run, and the artifact restored, while writing
this).

**THE IDENTITY FR-015 PINNED IS 5 OF 6, AND THIS CERTIFICATION SAYS SO RATHER THAN AROUND IT.**
FR-015 reads: *"floor := the count of regions charged by T025–T038's enumerated registrants — at
minimum neuromodulation, medulla, hypothalamus, midbrain, thalamus, insula"*. Measured:

* charged **and** named — neuromodulation, medulla, hypothalamus, midbrain, **thalamus** (T039a);
* named and **NOT** charged — **insula**;
* charged and not named — basal_ganglia, cortex.

Before T039a the integer would have read 6 purely because two unnamed regions charged in place of
two named ones. That is why the floor artifact carries an `identity_gap` block: a count standing in
for an identity is exactly the substitution this gate exists to prevent.

**Insula is the one gap, and it is a different KIND of gap.** It is the only region in the tree for
which `charged − reachable` is non-empty: its two `body_in` edges already carry diverged measured
attributions, so nothing about it is unmeasured, and the gap is reachability alone. The consumer
that `dormancy_gate`'s exemption row promised for T030/T031 — both `[X]` — did not land, and §6b
measures WHY rather than re-dating the promise: **the coalition sink is out of range**, so that
particular wire is a measured null and was refused. A sink that does work was found and is scoped
there.

## 2. SC-001 — P6 faculty-to-faculty reads

```
python tools/p6_gate.py --assert
```

```
P6 GATE OK: all 5 ratchets hold.
  [OK] unimplemented_declared_pairs  24 <= 24
  [OK] unread_writer_modules          7 <=  7
  NOT ASSERTED, deliberately: faculty-to-faculty reads = 0.
```

The gate refuses to assert this count and states why: *"A 'must not FALL' ratchet on a count that
is already 0 has NO reachable red — nothing can drive it below zero. P8: a criterion with no
reachable red is DELETED, not weakened. It becomes a floor the day it is non-zero."*

SC-001 requires this count **> 0 on genuinely charged signals (neuter-verified), from a measured
baseline of 0**. It is 0. The route was searched and refused on measurement — §6a.

## 3. Wake-tick ratchets (SC-002)

```
python -m snn_data_lake.substrate.perf_baseline --assert
```

```
wake p50 162.156 vs baseline 183.517 (ambient-ratio) -> OK
p95 274.050 vs 225.768 -> reported-HIGH (tail is ambient-dominated here)
duty now 0.508, baseline 0.508
```

GREEN on the binding leg. The ratchet is the **ambient-ratio** form: p50 binds and the tail is
REPORTED, because the p95 is dominated by ambient load on a developer machine rather than by the
brain. The duty cycle is reported beside it, as SC-002 requires, and is unchanged from baseline.

## 4. G5 (SC-003)

```
python -m snn_data_lake.substrate.replay_gate
```

```
replay_gate wiring-check OK: snapshot->restore continuation byte-identical;
omit-positions refused; re-seeded control diverged; parameter mismatch refused
```

GREEN, and note the two controls: **omitting the RNG stream positions is refused** (the criterion
is positions-not-seeds) and **a re-seeded control diverges**, so the byte-identity is evidence of a
restored stream position rather than of a determinism that would hold anyway.

Ignition baseline recorded at `tools/ignition_baseline.json`: `theta=0.55`,
`theta_hat_antimode=0.595`, `theta_hat_logistic=0.585`, `fraction_ignited=0.5673`, verdict OK.

## 5. Lake rebuild proof (SC-002)

```
python -m snn_data_lake.substrate.lake_rebuild_gate
```

```
lake_rebuild_gate self-check OK: rebuild equality; planted-divergence red fired;
lake-tracking control moved the digest
```

GREEN, with its own reachable red (a planted divergence fires) and a liveness control (the digest
moves when the lake changes), so an equality result cannot be produced by a dead comparison.

## 6. Disposition of what is UNMET

Recorded rather than resolved by weakening the criteria, per the cross-cutting invariant that a
gate asserts the structure did not get worse and never that the policy is satisfied. Neither item
below was force-wired to make a number move, and that restraint is the finding, not an omission.

### 6a. SC-001 (faculty-to-faculty > 0) — UNMET, and the route was searched before it was refused

`fac_cross` counts a read of a workspace INSTANCE where BOTH the reading module and the OWNING
module are role FACULTY. The count is 0 for a structural reason rather than an oversight: an
adversarial search over all five eligible owners found exactly one candidate — `cortex/brain.py`
owns a `ReentrantWorkspace` and registers a subscriber literally named `"language"` whose
`broadcast_to("language")` has zero call sites, so the language faculty could drain a channel built
and named for it.

**It was refused on its own measurement.** Instrumented over the proposal's own protocol
(`make_brain(seed=0)`, 60 wake ticks):

* the winning coalition content was the perceived digit on **57 of 60** ticks and `None` on 3;
  `"self"` never won and `"salience"` never won;
* the salience coalition was submitted 31 times with content `"novel"` on **31 of 31** and
  `"threat"` on **0** (consistent with the already-recorded finding that `threat()` returns 0.0 on
  every call);
* `broadcast_to("language")` was **byte-identical to `broadcast_to("action")` on 20 of 20 ticks**.

So the payload carries nothing listener-specific, and the flip the neuter test was built around has
measured probability 0 at that seed. Wiring it would have moved the counter while changing no
outcome — P3's dormancy and P6's own "a route is not a conversation" in a single move. **P8 requires
deletion on a null, and reporting the null is a complete result.** `tools/p6_gate.py` reaches the
same verdict about itself.

**The prerequisite is already recorded in the tree**: `cortex/brain.py` states that the re-entrant
bias payload is computed once from the winning content and is therefore listener-blind, and that
this must be fixed BEFORE a specialist read means anything. That ordering is the honest next step;
SC-001 becomes satisfiable after it, not before.

### 6b. Insula charge — UNMET, and the promised route is a MEASURED NULL (T039b)

Insula is the only region in the tree that is CHARGED-BUT-UNREACHABLE: its two `body_in` edges
already carry diverged measured attributions, so nothing about it is unmeasured. The entire gap is
one static live-path import, which is why `dormancy_gate.EXEMPT` has promised since T021a that the
`interoception_out` coalition consumer would land with T030/T031 and retire the row. Both are `[X]`
and it did not land. **This certification measured WHY, rather than re-dating the promise a third
time.**

**The coalition sink is out of range — the producer cannot reach the consumer's admission floor.**

| quantity | measured | command |
|---|---|---|
| live bus admission band | **[0.5505, 0.7450]**, floor = the rheobase | `GlobalWorkspace.band_claim` over 30 draws; `ignition_threshold` = 0.55 |
| `feeling_salience`, vary=arousal | min 0.0000, **max 0.2798** | `port_charge_probe.interoceptive_challenge_frames("arousal")` |
| `feeling_salience`, vary=drives | min 0.0000, **max 0.2142** | same, `("drives")` |

The producer's strongest possible output falls ~0.27 short of the floor the bus admits on, so an
interoceptive bid never competes. Two independent adversarial measurements drove the real feed into
the real bus and agree: `interoception` appears in **zero of 120** slot and head censuses, and
**pinning `feeling_salience` to its own live mean leaves the brain BYTE-IDENTICAL on 120/120 ticks**
— identical slots, broadcasts, thought, and ignition count. A 12-of-120 difference that looked like
movement is reproduced exactly by a constant-mean arm and eliminated by a constant-zero arm, so it
is buffer-order jitter, not consumption.

That is reachability without charge, which P3 says is not charge, and it is the same null SC-001
hit in §6a. **P8 requires deletion on a null; the route is refused and the measurement recorded so
it is not re-attempted.** The `EXEMPT` row now states this measurement instead of a prediction.

**The sink that does work, and what it will owe.** The producer's own docstring does not say
"compete" — it says *a body far from baseline should find it EASIER TO IGNITE the workspace*. That
is threshold modulation, and the door already exists: T032's `set_base_rheobase` on
`thalamus/reentrant_workspace.py` (range-checked, provenance-recorded), whose module is a declared
pair with insula and whose `reroute_out` edge T039a just charged. Measured as graded and sharp — a
body coalition at 0.45–0.52 ignites **0/30** at θ = 0.55 and **30/30** once θ reaches its level.

The feed is live: `cortex/live_brain.py` constructs the arm with `interoception=True`, and the live
arm carries a real per-tick `{stress, caution}`. Three obligations that build carries, none of them
bookkeeping:

1. a public accessor for the arm's CACHED medullary reading — `stress` has none today, and it must
   never trigger a second `sense()` (that forks the replay stream, per the organ's own rule);
2. a **P1-derived** modulation law, not a tuned gain — the whole point is that the constant is not
   picked;
3. P8's full invention burden on that law: a reachable red, a control that is not the method
   itself, per-item attribution, and deletion if it nulls.

Its blast radius is the central ignition path, including a tracked 120-tick occupancy harness whose
baselines would move. It is a task, not a subtask, and it is the honest next one.

## 7. What this certification does and does not claim

It claims: three criteria measured green with their own controls, an integer floor recorded from a
measured census with a reachable red, and two criteria measured UNMET with the reason each was not
forced. It does not claim that US1 is satisfied — a gate asserts that the structure did not get
worse, never that the policy is met, and two of five criteria are open with named next steps.

## 8. Amendment 2026-08-22 (T039f) — criterion 1's identity gap is CLOSED

| # | Criterion | At certification | Now |
|---|---|---|---|
| 1 | charged-region floor + FR-015 identity | 7 charged, identity **5 of 6** (insula missing) | **8 charged, identity 6 of 6** |
| 2 | SC-001 faculty-to-faculty reads | 0 | 0 — still open, see below |

**§6b's route was taken and it is not a null.** The insula terminus is fed the declared `body_in`
concat from the two producers already live in `cortex/brain.py`, and its `feeling_salience` joins
the EXISTING ignition law rather than opening a second one (P2 — the tree already modulates this
threshold through `region_coherence.ignition_bias`). `tools/region_charge_floor.json` is raised
7 → 8 and `identity_gap.named_and_NOT_charged` is now empty.

Measured, seed 0: salience mean 0.235 / max 0.355 / 198 distinct over 200 ticks; the live threshold
moves on 198 of 200 ticks, **135 of them ticks where the synfire binding gate is shut** and the
coherence term contributes exactly zero. Neuter A/B over 40 ticks: threshold differs on 35, the
workspace **winner** differs on 21 — and, disclosed, the binary `ignited` flag differs on **0**, so
what moved is which content wins rather than whether anything ignited at all.

**Two claims THIS DOCUMENT made are refuted by that build, and they are corrected here rather than
quietly dropped:**

* §6b warned the 120-tick occupancy harness baselines would move. They do not. That harness
  measures the **arm's GlobalWorkspace**; driving the `ReentrantWorkspace` rheobase 0.55 → 0.40
  leaves every field of it byte-identical (occupancy, pinned-at-capacity, co-occupancy, both
  censuses, access-without-focus). The blast radius was smaller than this document priced.
* The T039 tasks entry recorded that the importer "must be a THALAMUS module … so the only home is
  `global_workspace.py` at 255 KB", on the grounds that `undeclared_import_pairs` is zero-slack.
  Measured: cortex is a **declared** connectome partner of insula (`insula → ctx_pfc`), so the
  import from `cortex/brain.py` held that ratchet at 11 and moved `unimplemented_declared_pairs`
  **24 → 23**. It implemented a declared route. The architectural cost that entry priced is not owed.

**SC-001 remains 0 and criterion 2 remains UNMET.** §6a's premise — that the re-entrant payload is
listener-blind — no longer holds after T039c, re-measured here: `broadcast_to("action")` and
`broadcast_to("language")` now differ on 48 of 60 ticks at seed 0 (18/60, 51/60, 51/60, 55/60 at
seeds 1–4), and with `report_use` neutered they differ on **0 of 60 at every seed**, reproducing
this document's original "byte-identical" null exactly. So the payload objection is answered and the
counter still did not move, for a different and structural reason: the only `broadcast_to` reader is
`cortex/brain.py` reading **its own** instance, which is a SELF-read and excluded from `cross`.
`fac_cross` needs a *different* faculty module. Also re-derived and NOT reproducing: this document's
"the perceived digit won 57 of 60 and `self` never won" — at HEAD, seed 0 gives `self` 27, digits 32,
None 1 (harness control: `persist=False` vs the default is byte-identical, so that is not the cause).
The half that DOES reproduce is `salience` winning 0 of 60 on all five seeds.

### 8a. T039g — the SC-001 instrument was blind to clean code

Recorded here because it changes what "SC-001 is 0" means. Measured over four overlay arms:
reading the re-entrant workspace through `brain._reentrant` (the owner's PRIVATE attribute) scored
a faculty-to-faculty read; reading the SAME object through a public accessor produced **no site at
all — byte-identical to a negative-control arm reading an attribute nothing constructs.** So the
only route this gate rewarded was the one that reaches past a class into its private state, and
§6a's "the route was searched and refused" was true of a search the instrument had narrowed.

T039g fixes the walk (zero delta at HEAD: fac_cross 0, cross 2, unattributable 11, before and
after) and adds `Brain.reentrant`. Three false attributions had to be closed on the way, each one
found by the fix breaking something that already worked — the hop is a fallback and never an
override, it applies only to an attribute access and never a bare name, and it refuses any name
that is also a construction token somewhere. Without those, the fix manufactured three
faculty-to-faculty reads out of reads of the ACTION-owned shared bus.

**SC-001 is still 0 and this commit claims nothing about it.** What changed is that a consumer can
now be written the clean way and be seen.

### 8b. T039h — SC-001 is MET, and US1's five criteria are all green

| # | Criterion | At certification | Now |
|---|---|---|---|
| 1 | charged-region floor + FR-015 identity | 7 charged, identity 5 of 6 | **8 charged, identity 6 of 6** (T039f) |
| 2 | SC-001 faculty-to-faculty reads | 0 | **1, neuter-verified** (T039h) |
| 3 | wake-tick ratchets | GREEN | GREEN |
| 4 | G5 | GREEN | GREEN |
| 5 | lake rebuild equality | GREEN | GREEN |

The read: `insula/interoceptive_integration.py::_top_down` draining
`brain.reentrant.broadcast_to("interoception")`, owner `cortex/brain.py`, both endpoints role
FACULTY, through the public accessor rather than past it.

**The counter is not the evidence; the neuter A/B is.** Same seed, 40 ticks, the only difference
being whether the specialist reads the bus: the reading faculty's OWN output differs on **35 of
40** ticks from tick 5, with the top-down term nonzero on 37 of 40 (max 0.1572) so the comparison
is not between two identical arms.

**What this does NOT establish, stated because the table above would otherwise be read as more than
it is:** the workspace **winner differs on 0 of 240 ticks** and `ignited` on 0 of 40. The wire
changes the reading faculty's output and the ignition threshold it feeds. It does not change which
content wins. P6's bar is the consumer's own output and that bar is met; the stronger claim is not
made here.

Two further measurements the build owed rather than assumed:

* **The loop does not run away.** Salience lowers the threshold and ignition raises salience. Over
  240 ticks salience rose on 87 of 239 ticks (a runaway is ~100%), peaked at 0.3964 against a bar
  of 0.95, and the top-down term saturates at 0.2261. The saturation is the WINNER settling, not
  the mechanism failing to grade — by window, 11 distinct winners give 17 distinct term values,
  2 give 30, 1 gives 11, 1 gives 1.
* **It cost no workspace diversity.** The detached arm collapses to a single winner in exactly the
  same windows (11/2/1/1, 12 distinct overall, winner differs 0 of 240), so the settling is the
  workspace's own behaviour rather than something this loop imposed.

`language` remains subscribed-and-never-drained on purpose: it is the negative control that makes
T039c's listener term measurable. Measured `_use` after 40 ticks — interoception 9 keys, action 9,
language 0.

**§6a is now closed and §2's verdict is superseded.** Its premise (a listener-blind payload) fell
at T039c; its structural blocker (the only reader was the owning module) fell here; and its
instrument blocker — that the gate could only see a private poke — fell at T039g.

### 8c. T039i — the poll found the surface could not say what the insula measures

Not a criterion change; US1's five stay as §8b records them. Recorded here because it is the third
time the post-task poll has caught the same shape and the pattern is the useful part.

On the daemon running T039h, `how does your body feel right now?` returned an abstain (route `Gf`)
and an answer about wake phase, while `/api/all` carried `feeling_salience` 0.2114 and
`interoceptive_attention` 0.1 with PROVENANCE rows on both. The brain measured its body, the
cockpit drew it, and the brain could not say it — P4c dormancy on a sensorimotor surface, exactly
what T039e fixed for the coverage pair.

T039i adds the report. It reads the salience from the faculty that produced it rather than
recomputing (recomputing would advance the running baseline and report a different number than the
one the ignition law used — verified byte-equal, and the reply is asserted not to advance `_seen`),
and it refuses to translate the distance into a mood: the number says how unusual the body is for
itself, carries no valence, and decays toward zero as the baseline follows a persistent state.

**The transferable rule, now three for three: every quantity this project adds to the ignition or
sensorimotor path needs the brain to be able to SAY it, not only the cockpit to draw it.** T038 →
T039e for coverage, T039f/T039h → T039i for interoception. The poll is what turns that from a
principle into a caught defect.

### 8d. T039j — the three open tract items, and one correction to how they were described

§8b and §8c left three items open in writing. All three are closed, and one of them was described
wrongly when it was named.

| item | state |
|---|---|
| `interoception_out` had no typed manifest edge | **closed** — `region_coherence.interoception_in`, charged |
| the manifest shortcut medulla → insula | **closed** — the tract now relays through `thal_relay` |
| `(thalamus, insula)` / `(medulla, thalamus)` "unimplemented" | **the description was wrong** — see below |

**The correction.** Those two pairs were called unimplemented, and the count did move 23 → 21, but
`specs/BACKLOG_RANKED.md` row 59 had already adjudicated exactly this: *do not "fix" FINDING 1 by
adding region-to-region imports* — `p6_gate`'s `unimplemented_declared_pairs` measures **import
topology, not traffic**, and those routes were already load-bearing through
`thalamus/connectome_router.py`, which turns every declared connectome edge into a delayed LIF
projection with zero direct imports. So nothing dead was brought to life. The earlier framing
("a runtime loop closed, the import graph didn't move") implied a gap that the repo had already
measured away, and this record supersedes it.

**What was genuinely wrong, and is now fixed**, is at the TYPED layer, which is a different layer
from the anatomy graph and always was: the manifest declared a medulla → insula edge the connectome
declares nowhere, and one OUT port sourced no edge at all. Both are gone.

**P2 was checked rather than assumed.** `connectome_router` is the incumbent and it is live. It
propagates abstract graded drive over the anatomy graph and contains zero references to `stress`,
`caution` or the terminus — measured, and pinned by a test that fails the day it grows into the
typed path, because on that day these two are one job and must merge rather than sit beside one
another.

**Moving an edge moves its charge, and the gate said so before I did.** Medulla fell to
REACHABLE-UNCHARGED and the region count went 8 → 7, red against its own floor, because the
medulla's charge lived on the edge that moved. The relay now carries that per-edge attribution;
the record citing the retired edge was deleted rather than left beside the new one.
