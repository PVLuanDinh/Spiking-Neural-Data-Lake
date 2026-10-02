# Brain-Faculty Coverage — Baseline (2026-07-28, v4.78)

> **WITHDRAWN 2026-07-29 as unreproducible. Do not cite the 37.5% below.**
> The rubric forbids any band above 40 without a runtime A/B, and **there was not one recorded A/B
> transcript in the repository** — all 602 tracked files searched, `BEHAVIOURAL CHARGE` present in
> three (the rubric, POLICIES.md, and `tools/charge_probe.py` itself) and `act/hold decision changed`
> in one (the probe). So the maximum re-derivable total was **40%**, and 37.5% sits at 94% of that
> ceiling while the text below documents multiple rule-4 zeros. Worse, a category score is *defined*
> as the mean over its faculties and no per-faculty band table was ever committed, so **0 of the 11
> scores can be re-derived** — the rubric asked each auditor for exactly that table (`:85-91`) and the
> deliverable was produced and discarded.
> The instrument that ends this is now `tools/coverage_score.py` with the ledger in
> [`coverage/`](coverage/); read [`coverage/README.md`](coverage/README.md) first. The number it
> currently prints is a **floor on a partial roster** and is NOT comparable to 37.5% — treating them as
> comparable is the same mistake `COVERAGE_RUBRIC.md:14-16` records for the earlier 22.8% and ~31%.
> Everything below is kept as the record of what was claimed and how it was reasoned, which is still
> the best available inventory of where the gaps are.

**Equal-weighted total: 26.3%** — the current, re-derivable number. Run `python tools/coverage_score.py
score` to reproduce it from the tracked ledger; `--check` parses the line above and refuses if this file
and the evidence disagree. It is a FIRST MEASUREMENT, not a fall from 37.5%: the 37.5% is withdrawn as
unreproducible (see the banner), so the two do not sit on a common scale and no delta between them means
anything. The withdrawn table is kept below as history.

> **26.4% -> 26.3% on 2026-08-13, and the 0.1 is the honest part.** For an unknown stretch before
> this date the ledger could not produce a number AT ALL: 175 citations had drifted out of
> resolution, and `--check` refuses to print a total while any claim is unsupported. So the 26.4%
> above was a stored headline that the evidence had stopped being able to confirm or deny. The
> anchors were repaired (151 by `--repin`'s blame walk, 19 by unique-symbol AST lookup, 5 by hand),
> `--check` computed a total again, and it disagreed with the stored one. The delta is ONE row:
> `Consciousness.binding_of_broadcast_content` moved band 10 -> UNSCORED, because its own citation
> column had been arguing since 2026-08-07 that "the capability is honestly ABSENT, not low-band"
> while the band column still scored it 10. UNSCORED counts 0, so the mean fell. Nothing about the
> brain changed on 2026-08-13; what changed is that the number can be checked again.

Scored against [`COVERAGE_RUBRIC.md`](COVERAGE_RUBRIC.md) by eleven independent read-only auditors, one
per category. Every category scoring above 40 was then handed to a separate agent whose only job was to
**refute** it.

## Result

| category | auditor | final | Δ | what moved it |
|---|---:|---:|---:|---|
| Perception | 46 | **43** | −3 | non-isolating neuter on the multimodal encoder; two constants inside the top faculty (`goal=None` at `brain.py:1068` makes `alpha_goal` a multiplicative identity; `gain` passed but never read) |
| Executive | 51 | **42** | −9 | the four top faculties are four probe points on **one serial chain** — the conflict-neuter and evc-neuter action traces are byte-identical on 122/122 wake ticks |
| Memory | 40 | **40** | 0 | not verified (≤40) |
| Imagination | 39 | **39** | 0 | not verified (≤40) |
| Consciousness | 48 | **38** | −10 | `global_availability` is 1.0 on all 60 ignited ticks — a restatement of the ignited bit; PCI took one value over 25 nights |
| Affect | 38 | **38** | 0 | not verified (≤40) |
| Social | 38 | **38** | 0 | not verified (≤40) |
| SpatialTemporal | 36 | **36** | 0 | not verified (≤40) |
| Language | 35 | **35** | 0 | not verified (≤40) |
| Gf | 34 | **34** | 0 | not verified (≤40) |
| WholeBrain | 29 | **29** | 0 | not verified (≤40) |

**Equal-weighted total: 37.5%** (412/11). Pre-refutation it was 39.5%; adversarial verification cost
2.0 points.

## Read this number as an upper bound

Only categories scoring above 40 were adversarially verified — 3 of 11 — and **all three fell**, by 3,
9 and 10 points. A refutation rate of 3/3 among the checked categories is the load-bearing fact here.
The other eight were exempted from verification *by their own low score*, which is a selection effect
and not a clean bill of health; several of them contain individual faculties scored 61–80 on a single
A/B each — Social 80, Memory 80, WholeBrain 70, Imagination 68 ×2, Affect 68 — which are exactly the
shapes that fell elsewhere.

If the unverified categories refute at the rate the verified ones did, the honest figure is low-to-mid
thirties. That extrapolation is deliberately **not** taken as the number; it is stated so the 37.5% is
read with its error bar attached.

## This does not compare to 22.8% or ~31%

Those were scored against a rubric that was never committed, with a different denominator and a
different evidence standard. Any delta against them would be arithmetic across incompatible scales.
**The next re-sweep compares to 37.5%, and to nothing before it.** ~~Struck 2026-07-30~~: 37.5% was
withdrawn as unreproducible, so nothing compares to it either. The comparison base is the re-derived
26.3% at the top of this file (26.4% until 2026-08-13; see the note there -- the anchors were repaired
so the ledger could compute a total again, and one row moved to UNSCORED).

## The artifact caveat

Seven of Perception's thirteen faculties — including all four highest — are live only through binaries
that are **not in the repo**: `models/` is gitignored (`.gitignore:76`) and `git ls-files models/`
returns nothing. Verified empirically by redirecting `SNN_EPROP_CKPT` / `SNN_EPROP_AUDIO_CKPT`
(`brain.py:545`, `:577`) to absent paths: the default gives `perception_tier=eprop-snn`,
`perceiving_real=True`; without the checkpoints it gives `tier=stdlib`, `perceiving_real=False`, and
`LearnedAttention.attend` is never called at all. The 92.97% figure is measured on the same 256 frames
cycled forever.

The weights are gitignored for good reason — they are large binaries — and `sensorimotor/eprop_nmnist.py` (promoted from training/ 2026-08-12)
and `training/eprop_shd.py` regenerate them. That is now written down in
[`TRAINING.md`](TRAINING.md); until this baseline it was not, anywhere, which made the perception tier
exactly the kind of unreproducible claim the rubric exists to prevent.

## Highest-leverage gaps — two of the top four did not survive re-checking

Before acting on any of these I re-measured them myself. **Gap 1's finding was confirmed but its
proposed fix rejected; gap 2 was refuted outright.** That is a 50% failure rate among the top-ranked
"cheap wires", from a sweep that had already been through one adversarial pass — which is the strongest
argument in this document for never wiring straight off an audit's ranked list.

Note especially how gap 2 failed: the same auditor who correctly found that `models/` ships empty then
measured a *downstream consequence of that same emptiness* and reported it as an independent defect.
One cause, counted twice. The Executive category fell 9 points for the identical reason at a different
scale — four probe points on one serial chain, scored as four faculties.


Cheap wires first. The ceiling applies throughout: wiring a faculty into the three-way arm decision at
`engine/brain.py::_route_percept_to_action` can raise **Executive, Affect, SpatialTemporal and
action-relevant Perception**, and cannot raise Language, Imagination, Gf, Consciousness or WholeBrain.

| # | gap | target |
|---|---|---|
| 1 | ~~**The amygdala can never learn**~~ — **finding confirmed, disposition REJECTED.** `threat()` is indeed a measured constant: called 38× per 80 ticks, returns exactly 0.0 every time, and that zero propagates into every episode's arousal tag, the LC salience quantile and the workspace threat bid. But the cause is not a defect. `pair()`'s one live call site (`autonomous_loop.py:1108`) sits **inside the SAFETY-FIRST early return** at `:1102` — it fires only on real contact, which also trips E-STOP. That is correct fear-conditioning semantics. With `SAFETY_CALIBRATED = False` and no physical arm, there are no aversive events, and 0.0 is the honest output of a brain that has never been hurt. The proposed wire — bind `pair()` to the negative-reward signal — would condition the amygdala on **disappointment rather than nociception**, a semantic change to a safety-adjacent faculty made to move a number. Disposition: P3 **exempt, awaiting a real sensor**; it starts learning with no code change once the arm exists | none — see `faculties/fear_conditioning.py` header |
| 2 | ~~**`GridPlaceMap` never observes**~~ — **REFUTED on runtime evidence.** Re-measured over 120 live ticks with the checkpoints present: `perception_tier=eprop-snn`, `perceiving_real=True`, `_perceived_digit` takes **eight distinct values (0–7)**, and `place_here` populates (`6`, `6@1`, `6@2`). The guard at `brain.py:1282` fires. The auditor almost certainly measured in the no-checkpoint state — which is the artifact caveat above, not a second independent defect. **Two findings, one cause.** | none |
| 3 | **`Pragmatics` has zero callers** outside its own module | Social |
| 4 | The faculty-charge registry has zero registrations (`region_wiring.py:34`) — and is honesty-trapped: its own source warns that registering telemetry adapters would fake liveness | `infra/region_wiring.py` |

## Tested and rejected

`salience_novelty` is the one gap from this sweep that was implemented and then **reverted**. Passing
it makes `arm._novelty` and `spikes.midbrain` go from one distinct value over 60 ticks to 32 each — and
a 200-tick neuter-A/B then showed 0/200 workspace-winner flips, 0/200 act/hold decisions and 0/200
trajectory changes, at +8.4% tick cost (49.0 → 53.1 ms). The control that makes that trustworthy: the
same harness shows the `drive` source flipping 22/60 trajectories, so the observable is not degenerate.

The faculty is inert in fact, not merely unwired. P3 says a wire that does not charge is reverted, so
it was. The measurement is recorded at `engine/brain.py:197-215` and in `PROVENANCE`, so the next
person does not re-run it.
