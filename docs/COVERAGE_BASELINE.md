# Brain-Faculty Coverage — Baseline (2026-07-28, v4.78)

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
**The next re-sweep compares to 37.5%, and to nothing before it.**

## The artifact caveat

Seven of Perception's thirteen faculties — including all four highest — are live only through binaries
that are **not in the repo**: `models/` is gitignored (`.gitignore:76`) and `git ls-files models/`
returns nothing. Verified empirically by redirecting `SNN_EPROP_CKPT` / `SNN_EPROP_AUDIO_CKPT`
(`brain.py:545`, `:577`) to absent paths: the default gives `perception_tier=eprop-snn`,
`perceiving_real=True`; without the checkpoints it gives `tier=stdlib`, `perceiving_real=False`, and
`LearnedAttention.attend` is never called at all. The 92.97% figure is measured on the same 256 frames
cycled forever.

The weights are gitignored for good reason — they are large binaries — and `training/eprop_nmnist.py`
and `training/eprop_shd.py` regenerate them. That is now written down in
[`TRAINING.md`](TRAINING.md); until this baseline it was not, anywhere, which made the perception tier
exactly the kind of unreproducible claim the rubric exists to prevent.

## Highest-leverage gaps, all verified

Cheap wires first. The ceiling applies throughout: wiring a faculty into the three-way arm decision at
`cortex/brain.py::_route_percept_to_action` can raise **Executive, Affect, SpatialTemporal and
action-relevant Perception**, and cannot raise Language, Imagination, Gf, Consciousness or WholeBrain.

| # | gap | target |
|---|---|---|
| 1 | **The amygdala can never learn.** `pair()` is gated on `distance < CONTACT` while the live tick feeds `SAFE_DISTANCE = CONTACT * 3.0` (`brain.py:71`), so `threat()` is a measured constant 0.0 over 60/60 ticks. One dead constant also zeroes every stored episode's arousal tag, the LC salience quantile, and the workspace threat bid | `brainstem/autonomous_loop.py:1373` |
| 2 | **`GridPlaceMap` never observes** — the guard at `brain.py:1282` requires `_perceived_digit`, measured `None` on 120/120 ticks, so every episodic trace carries `where='None'` | `cortex/brain.py:1282` |
| 3 | **`Pragmatics` has zero callers** outside its own module | Social |
| 4 | The faculty-charge registry has zero registrations (`region_wiring.py:34`) — and is honesty-trapped: its own source warns that registering telemetry adapters would fake liveness | `substrate/region_wiring.py` |

## Tested and rejected

`salience_novelty` is the one gap from this sweep that was implemented and then **reverted**. Passing
it makes `arm._novelty` and `spikes.midbrain` go from one distinct value over 60 ticks to 32 each — and
a 200-tick neuter-A/B then showed 0/200 workspace-winner flips, 0/200 act/hold decisions and 0/200
trajectory changes, at +8.4% tick cost (49.0 → 53.1 ms). The control that makes that trustworthy: the
same harness shows the `drive` source flipping 22/60 trajectories, so the observable is not degenerate.

The faculty is inert in fact, not merely unwired. P3 says a wire that does not charge is reverted, so
it was. The measurement is recorded at `cortex/brain.py:197-215` and in `PROVENANCE`, so the next
person does not re-run it.
