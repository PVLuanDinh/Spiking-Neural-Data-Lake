# Coverage to 100 — the program, and what it will and will not buy

**Owner decision, 2026-07-29.** Four things were authorized in one turn: (a) treat 100% brain-faculty
coverage as a multi-session program rather than a one-session push, (b) remove the zero-dependency
rule's remaining machinery and add dependencies freely, (c) make the brain genuinely use CPU **and**
GPU **and** disk, and (d) widen the live populations so the GPU pays on measurement rather than forcing
narrow ones onto CUDA. This file is the program. It is tracked so the next session inherits the
findings instead of re-deriving them — the same reason [`COVERAGE_RUBRIC.md`](COVERAGE_RUBRIC.md)
exists.

## What 100% actually means, stated before any work

Per [`COVERAGE_RUBRIC.md`](COVERAGE_RUBRIC.md), band **81–100 is "generalizes"** — the faculty holds
outside the developer-constructed case, on held-out data or an unrehearsed distribution. A category's
score is the **evidence-weighted mean over its constituent faculties**, not the maximum. So 100% is not
"everything wired". It is *every faculty in all eleven categories proven to generalize.*

Two consequences, both load-bearing:

1. **Wiring cannot get there.** The rubric pre-declares the ceiling: wiring a faculty into
   `cortex/brain.py::_route_percept_to_action` moves a three-way arm decision and can raise Executive,
   Affect, SpatialTemporal and action-relevant Perception. It **cannot** raise Language, Imagination,
   Gf, Consciousness or WholeBrain. Crediting those to a wiring wave is the theater P0 forbids.
2. **Two categories are at measured architectural ceilings.** BLiMP **51.0%** against a 50.0% chance
   floor on a fully-trained 808k-parameter e-prop LM, and ARC-AGI **2.75%** held-out against a 0.5%
   trivial-solver noise floor, with the failure mode proven to be DSL coverage rather than search
   budget. Points there require capability, not wiring.

The honest read of the 37.5% baseline is that it is an **upper bound**: only the 3 categories scoring
above 40 were adversarially verified and **all three fell** (by 3, 9 and 10 points). The other eight
were exempted by their own low score, which is a selection effect, not a clean bill of health.

## The instrument gap — fix this first, because nothing else is checkable without it

**The 37.5% is assembled entirely by hand.** No `.py` file in this repo reads `COVERAGE_RUBRIC.md` or
`COVERAGE_BASELINE.md`; a repo-wide search matches only markdown. The three code paths named
"coverage" measure unrelated things:

| path | what it actually measures |
|---|---|
| `pyproject.toml:101` `--cov-fail-under=70` | pytest **line** coverage |
| `substrate/dormancy_gate.py:346-361` `coverage()` | module **reachability** — `{total, reachable, exempt, dormant}` |
| `substrate/region_wiring.py:67-88` `coverage()` | registered **charge adapters** — currently **0**; the live daemon logs `adapters=0` |

None of them is the faculty percentage. This is precisely the failure `COVERAGE_RUBRIC.md:8-9` was
written to end: *"A percentage nobody can re-derive is not a measurement."* The rubric ended the
problem for the *standard* and left it open for the *score*.

Until a scorer exists, "coverage went up" is an assertion. So the first deliverable is the scorer, and
per the standing rule for adding a gate, **its output is counted before any rule leans on it.**

## Corrections to COVERAGE_BASELINE.md found while planning this (2026-07-29)

Recorded here rather than silently fixed, because the baseline is the comparison point for every later
sweep and a stale entry inflates the next delta.

- **"`Pragmatics` has zero callers outside its own module" (gap 3) is stale.**
  `reasoning/spike_chat_lake.py:220-226` constructs `Pragmatics(seed=seed)` and `:997-1003` calls
  `implicature(text)` and emits a reply candidate. The true statement is narrower: it has no caller in
  `cortex/brain.py` and no A/B — wired to the chat path only.
- **Citation drift on the attention finding.** The baseline cites `brain.py:1068` for the `goal=None`
  site; `:1068` is inside a docstring and the real call is `brain.py:1080`. The finding itself holds —
  `learned_attention.py:91-92` returns `np.zeros(n_bands)` for `goal=None` and `:139` multiplies by
  `(1.0 + alpha_goal * goal_gain)`, so `alpha_goal` is a multiplicative identity.
- **The artifact caveat is machine-dependent, and on this machine it is milder than written.**
  `models/` is gitignored with 0 tracked files, so a fresh clone has no perception — that stands. But
  locally it **is** populated: `eprop_nmnist.pt`, `eprop_shd.pt`, `nmnist_pack.npz`, `shd_pack.npz`,
  `surrogate_nmnist.pth`, `pc_cortex_nmnist.json`, `eprop_lm.npz`, `eprop_lm_alif.npz`, `spikegpt/`.
  The one artifact absent when this was written was `ar_spiking_lm.npz`; **it now exists** (39.5 MB,
  trained later the same day) — see below.
- **BLiMP's `n/a` was a path mismatch, not only an untrained model — SINCE FIXED (`bcd53611`).**
  Both halves now resolve `models/ar_spiking_lm.npz` (`training/ar_spiking_lm.py:51`,
  `reasoning/ar_spiking_lm_serve.py:53`). Kept as the record of the defect, in the past tense it earned;
  the description below is how it stood before the fix. A second, independent cause was found later and
  is separate: `blimp_bench` never imported the AR serve at all.
  `training/ar_spiking_lm.py:48` exports to `models/spiking_lm.npz`; `reasoning/ar_spiking_lm_serve.py:53`
  loads `models/ar_spiking_lm.npz`. Neither half can see the other's file, and `:455` *prints* the
  trainer's path while the loader uses the other one — which is how it stayed invisible. The trainer's
  docstring also still calls the file `training/spiking_lm.py`.

## POLICIES.md self-claims corrected the same day

A policy file that describes a codebase which no longer exists is the failure its own standing caution
names. Three claims were checked and were false:

- **`godel_machine` was listed among the self-build modules that "fire every 8 nights."** Its
  `reasoning/godel_machine.py::attempt_on_repo` has **zero call sites** tree-wide, and the only reference
  to the module outside itself is its *filename as a string* in the self-build's protected-paths list
  (`basal_ganglia/brain_native_selfbuild.py`). What does fire, CORRECTED 2026-08-13: the `self_build_loop` route reaches its recorder but is MEASURED EMPTY -- 190 ticks under `NORD_SELFMOD=1` leave `Brain._selfbuild_last == {'night': 8, 'proposed': 0, 'kept': 0}`, so `self_mod_harness` is NEVER CONSTRUCTED. The claim that it is `reached via Brain.self_improve`
  is FALSE: that method has ZERO call sites tree-wide (its def is `Brain.self_improve`, and `docs/POLICIES.md`'s own P0 row records the same). CITED BY SYMBOL because all five line numbers in this paragraph were re-derived 2026-08-13 and every one had rotted -- `godel_machine.py:159` lands on a path string (the def is `attempt_on_repo`), `brain.py:1708` on an R-STDP comment, `:2520` on a table row, `:3485` on `"kill",`, and `brain_native_selfbuild.py:73` on a `__spec__` comment. The route is `cortex/brain.py::Brain._enter_wake` -> `Brain._maybe_autonomous_self_build` -> `basal_ganglia/self_build_loop.SelfBuildLoop.run`. Wiring the Godel-machine adapter is open P0 work, and so is FILLING THE PROPOSER: a call-site wire into a loop that proposes nothing satisfies a grep and changes no outcome.
- **"67 such swallows in `cortex/brain.py` alone"** did not reproduce, and neither did the audit's
  correction of it ("112 / 203"), because neither said what it counted. Re-counted by `ast` with the
  definition stated: `brain.py` has **111** bare `except Exception:` handlers, **39** of which wrap a
  `try` body containing an import; across 245 governed-tier files, **142** bare / **49** import-wrapping.
- **The neuter-A/B probe was cited as `scratchpad/smoke_drive_charge.py`, which does not exist.** The
  real instrument is `tools/charge_probe.py`, and it is tracked on purpose — this repo has twice cited
  an instrument that was gitignored.

## The compute story, and why widening is the honest route to the GPU

P5's measurements are real and they say the live tick does not belong on CUDA **at current widths**:
eager torch-CUDA loses to numpy below ~32,768 elements; CUDA graphs beat eager CUDA everywhere and drop
the crossover to **~4,096**; live populations are **24–240**, still ~17x short; and forcing it measured
**0.01x** at width 38. Repeating that experiment a third time is not a plan.

P5 also names the number that would change the answer: **population WIDTH.** And that width now exists
on the growth tier — with the earn-your-place stop corrected, grown capacity reaches **4,512 units at
24,000 rows**, past the crossover. So the route is: *make the brain genuinely wide, then move it,
against a benchmark on the real workload.* Not: move a narrow brain and call `cuda.is_available()` a
speedup.

Also unresolved and prior to any of this: the live tick is **891 ms** after two unbounded O(n^2) growths
were removed (from 90,000 ms), and that 891 ms is **unattributed**. `substrate/learn_daemon.py` still claims
`~33 ms`. The project's own rule applies — *neuter it and measure the delta; a cProfile ranking once
sent a whole optimisation plan after the wrong function.*

Structural fact that shapes the design: the live tick's cost centre contains **zero torch and zero
numpy**. `brainstem/autonomous_loop.py::AutonomousArm.step()` is 480 lines and ~60 narrow subsystem
calls with no array library at all.

## Dependency policy — the rule was already removed; the machinery was not

`pyproject.toml:8-13` has said since 2026-07-20 that the zero-dependency rule is removed project-wide,
and `dependencies = ["torch", "numpy"]` is unconditional. What survived is the machinery that still
acts as though it were in force, and it is being removed:

All of the following were **removed 2026-07-29** and are listed as the record of what the rule's
machinery consisted of:

- ~~`tests/conftest.py:29` forces `NORD_TORCH=0` for the whole pytest run~~ — file emptied to a
  rationale-only docstring
- ~~`.github/workflows/ci.yml:15-33` runs a 6-cell `torchfree-self-checks` matrix that installs **only**
  numpy~~ — job deleted, its 2x3 OS/Python matrix moved onto the real self-check job
- ~~`tests/test_counterfactual_sim.py:79-86` is the last executable purity assert~~ — removed
- ~~`src/snn_data_lake/training/__init__.py:3-7` still describes exporting weights "back into the
  zero-dep core"~~ — reframed as the serving-format argument it always was
- plus the flag itself at `cortex/brain.py` (two sites), `brain_atlas.py`, `run_selfchecks.py`,
  `substrate/cockpit.py`, `substrate/brain_telemetry.py`, and the frozen-exe entry point

`run_selfchecks.py`'s `TORCH_GATED` / `NENGO_GATED` lists are **kept**: they are still hand-maintained,
but they now mean "this module needs an optional extra", which is a real distinction, not a
zero-dependency claim.
- P5 honesty guard (c) makes the **CPU** env authoritative — `.venv-lake` (torch 2.13.0+cpu, numpy 2.4.6)
  against `.venv-cu128` (torch 2.11.0+cu128, numpy 1.26.4). Two different torch *and* numpy majors gate
  the same tree.

**Kept deliberately:** determinism comes from seeding (`random.Random(seed)`, `zlib.crc32`), not from
hardware absence. Removing the CPU-only gate must not remove reproducibility.

**One P2 duplicate found:** `src/snn_data_lake/training/device.py` says it *"Mirrors
research/torch_device.py"* — two device selectors for one job. `reasoning/torch_device.py::resolve_device`
is the survivor candidate (honours arg → `NORD_DEVICE` → `NORD_GPU=1`); note all ~25 of its callers are
in `reasoning/`, none in `cortex/`, `brainstem/` or `substrate/`.

## The ladder

Each phase ends with a measurement, and a phase that moves nothing is reported as moving nothing.

| # | phase | exit criterion |
|---|---|---|
| 1 | **Instrument** | `tools/coverage_score.py` re-derives a per-category % from tracked evidence files; refuses band-41+ without an A/B artifact and band-81+ without held-out evidence; the count of currently-unbacked category scores is published |
| 2 | **Dependency + env** | one environment runs daemon, self-checks, pytest and ruff; zero-dep machinery gone; the added dependency list is recorded with a justification each |
| 3 | **Attribute the tick** | where the 891 ms goes, by neuter-and-measure, not by profiler ranking; `substrate/learn_daemon.py` made true |
| 4 | ~~**Widen + fuse**~~ **REFUTED, done differently** | The widening premise was wrong and the measurement said so: the whole live brain holds **132** LIF cells, all per-tick neuron state fuses to ~1,700–3,500 elements (*inside* the crossover, not past it), and `AutonomousArm.step()` is **0.00%** of the profile — so fusing neurons would have optimised nothing. The real hot path was `coincidence`, a Python loop run once per stored episode per tick at **74.5%**. Fused bit-exactly via `(D + a·b)/2`: `complete()` 59.9 → 0.9 ms at N=2,000, and linear-in-N → **flat**. Shipped on CPU; this GEMV's measured crossover is ~1,600–2,000 rows, so a captured CUDA graph is a *loss* at our widths, and numba beats one by 16× on fused LIF. See P5. |
| 5 | **Charge the dormant** — **FIRST PASS DONE 2026-07-31** (`specs/001-charge-the-dormant`) | the open P3 findings resolved as wired-and-charged / retired / exempt-with-reason, each with a neuter-A/B. Never a dead consumer to silence a gate. **Outcome: 9 rules adjudicated at runtime. 2 permanent exemptions (dormant AND correct to be dormant); 4 OPEN exemptions carrying a disposition + port finding; 3 measured called-but-inert. NOTHING was wired or retired — every port check found a live class or an unported capability, and P2 forbids deleting either. No band moved; all five affected rows were already ≤40, so nine inert rules had inflated nothing.** Instrument fixes it forced: class-granular exemption keys (`n_defs` 56→58, two defs had been overwriting each other) and a graded observable in `charge_probe` (the flip counts were argmax tests over eleven sources and reported live faculties as inert). Still open: `IntrinsicPlasticity.adapt` measures 0 invocations at 400 ticks, contradicting an earlier CALLED_BUT_DISCARDED verdict — both readings recorded, neither deleted. |
| 6 | **Capability** — **SATISFIED AS A NULL, 2026-07-31** | the two ceiling categories attacked where they are actually blocked: a real-scale AR spiking LM for Language, perceptual predicate induction for Gf. Held-out numbers or an honest null. **Both halves were attacked and both returned measured results below the bar, which is what this criterion asks for — it says "held-out numbers OR an honest null", not "a capability".** LANGUAGE: a real 10.3M-parameter AR spiking LM trained on 119.6M chars of WikiText-103, held-out disjointness PROVEN not assumed (`tools/ar_lm_disjoint.py`, 0 of 3350 BLiMP sentences occur in the exposure); BLiMP did not move — every model row sits between the 29.4% length prior and the 53.0% char-level bar, at or below the 50% chance line (`docs/DEPTH_CEILINGS.md`). Gf: the ARC solver on the **evaluation split, 400 held-out tasks, 12/400 = 3.00%** (`gf_arc_eval_lib_off.txt`), and the cross-task library null is stronger than an equal score — `lib=on` solves **the identical twelve task IDs** as `lib=off` (`gf_arc_eval_lib_on.txt`), so the library changes not just how many but *which*, which is nothing. Banded 10 and 0 respectively. **NO CAPABILITY IS CLAIMED BY THIS ROW.** Closing phase 6 means the ceiling was measured, not raised: the honest Gf figure is ~3% held-out and the honest Language figure is chance. What would raise them is a training run measured in GPU-hours, and that is the next arc, not this one. |
| 7 | **Bound the growing substrate** — **MEASURED 2026-07-31, ANSWER IS NO** | whether the brain can run continuously *at all* — see below. Ranks with phase 1, not after phase 6. **Measured on the live daemon over 15.3 h of continuous operation: `brain_lake.bin` 75.07 → 151.13 MB, mean 4.98 MB/h (recent window 3.64), i.e. 119 MB/day and 31–43 GB/year.** A compaction path exists and is live (`KnowledgeLog.consolidate`, `substrate/home.py:152`, confirmed charged in the Phase 5 adjudication) but bounds only HALF the lake — its own docstring says it replaces the durable FACT records while "preserving any non-FACT (spiking) records in the one lake untouched". Symbolic facts are pruned and merged; spiking records and the `_T_GROWTH` blobs written every sleep tick are append-only with no prune path. **So the brain cannot presently run indefinitely, and the reason is not that a store grows — it is that exactly one half of the one lake has a prune policy.** The policy for the other half is specified in `specs/002-capability-and-bounding/spec.md` and deliberately NOT built here: changing what the brain forgets needs its own A/B, and the growth substrate is restored at construction, so a careless prune would delete state the brain later reads. |

## Phase 7 — the growth this plan originally missed

Every unbounded-growth defect chased so far was a **record store**: the two O(n²) tick growths (90,000 →
891 ms), `_episodes` (the fused scan is flat in N, but memory still grows ~30 MB at N=15,000),
`telemetry.jsonl` (69 → 160 MB in one session), the durable lake's facts (19,525 → 19,775), and the
always-novel `when` filler at `brain.py:1680` (`str(self._wake_ticks)`, so every episode adds a codebook
entry that can never match a future cue). Those are all **logs**.

Nothing here has measured growth in the **learned substrate** — neurogenesis adding units every sleep
night, weight matrices widening, state accumulating across nights. That omission is worth naming
plainly, because **P0 requires the brain to grow continuously**, so unbounded weight growth is the
failure mode this project's own purpose pushes toward, and no gate covers it.

**What is measured (cheap, from the daemon's own log).** Across 733 samples spanning nights 41–8,004,
`grown_units` stays in **8..120**, mean 46.7, with no upward trend. That is consistent with the prune
machinery working: `attractor.sleep(learn=True)` + `downscale_prune()` (`brain.py:2289-2294`),
Benna-Fusi metaplasticity, and `lake_memory`'s `SALIENT_CAP=128` / `TRACE_HARDCAP=4096` /
`RESERVOIR_K=64`.

**Two reasons that reassurance is not yet worth anything:**

1. **It may be the wrong field.** I do not know from the log whether `grown_units` is a *current
   population* or a *per-interval delta*. If it is a delta, the cumulative total is unmeasured and the
   8..120 range says nothing. Reading its emit site is the first five minutes of this phase.
2. **The night counter RESETS.** Observed in sequence: `364 → 41 → 81 → 122`. The daemon exits and
   relaunches whenever a new commit lands, and ~20 commits landed in a single session — so "8,004
   nights" is not 8,004 *continuous* nights of accumulated substrate, and the long-horizon behaviour
   has rarely been exercised at all.

**The decisive measurement is deliberately not the one above.** In-process pruning says nothing about
on-disk accumulation. Snapshot the size *and shape* of every persisted weight artifact — `models/*.npz`,
`brain_data/*.npz`, prototypes, `language_acq.npz`, the skills/curriculum JSON — at successive restarts
and plot against cumulative nights. Plateau = bounded; monotone = not. Then check whether any weight
**dimension** grows with no prune counterpart: `cortex/neurogenesis_engine.py`,
`hippocampus/growing_learner.py`, `hippocampus/function_preserving_growth.py`,
`substrate/similarity_sized_growth.py`.

**The template for a principled grower already exists**: NG-SNN's earn-your-place stop, capacity
monotone in data, 4,512 units at 24,000 rows — and note it only became trustworthy after the instrument
grading it was fixed. **Do not add a cap.** P1 forbids a magic ceiling, and the honest fix for an
unbounded grower is usually a prune rule rather than a limit; this repo's own record is that
capacity constants tuned on synthetic tasks were wrong three times, always in the direction of stopping
too early.

**The general lesson, since it caught us twice today:** a value that *moves* is not a value that is
*bounded*, and a faculty that prunes in-process is not a substrate that is bounded on disk.

### The question inverts: growth is not unbounded, it is DISCARDED

Raised by the owner, and verified: **the brain has one durable lake and the grown substrate is not in
it.** `cortex/neurogenesis_engine.py` has **no `save`, `load`, `persist`, `restore` or `resume` method
and writes no npz/json** — `grown_units` appears once, in a state dict at `:202`. `Brain._make_neuro`
constructs the engine fresh on every start. So grown units do not survive a restart.

`grown_units` was also confirmed to be a **population** attribute (`self.neuro.grown_units`, read at
`brain.py:2100`, `:2198`, `:2355`, `:2620`), not a per-interval delta — which resolves the caveat above
and makes the in-process 8..120 bound real. But it makes the restart finding worse, not better: a
bounded live population that is rebuilt from nothing is not a substrate that compounds.

**The daemon's persistence claim is true of what it names and false of what a reader will assume.**
`substrate/learn_daemon.py` reports "brain durable save + chat lake flushed" and `:20` states that state
"survives the restart via persist/reload". That holds for the durable lake, the chat store and the npz
weights. It does not hold for the grown neural substrate, because the owning engine has nowhere to save
to. Same shape as every other defect in this document: a claim accurate about its component and wrong
about the thing it appears to cover.

**Why this matters more than the bound.** The daemon exits and relaunches on every new commit — ~20
times in a single session — and the night counter resetting (`364 → 41 → 81 → 122`) is that happening.
So P0's "neural growth — live" avenue is real *per process* and resets constantly, meaning across 7,963
nights it has never accumulated. Growth that is discarded is not growth, and this is the census-counted-
as-capability failure P0's own honesty guard names, one level deeper than the census.

**The work, in order:**

1. **Enumerate what actually persists** across a restart versus what is rebuilt — the lake, chat, npz
   weights, prototypes, curriculum, *and* the grown units — and write it down. The claim at
   `substrate/learn_daemon.py` needs to become specific or go.
2. **Give the grown substrate a home in the one durable lake.** `_T_FACT` and `_T_SPIKE` already exist in
   the BinLog container (`substrate/home.py:45-46`), `_T_SPIKE` has just been freed by `SpikeLake`'s
   retirement and is kept as a reserved tag, and the format's foreign-tag skip/preserve behaviour is now
   self-checked — so an additive record type is the cheap, contract-respecting route. One lake, per the
   design, rather than a new sidecar file.
3. **Only then ask whether growth is bounded.** The disk snapshot described above answers a different
   question once growth actually survives; measuring accumulation across restarts is meaningless while
   the substrate resets at every one.

**Sequencing note, stated so it is not lost:** this reorders phase 7. Persisting growth comes *before*
bounding it — and it may well create the unbounded-growth problem this phase was originally written to
look for. That is the correct order anyway: a brain that cannot accumulate cannot be measured for
accumulation, and a prune rule designed against a substrate that resets daily would be fitted to noise.

## The standing honesty constraints for this program

- A faculty that cannot be scored is **visible in the denominator**, never silently omitted — silent
  omission is the inflation route.
- A wire that does not charge is **reverted**, not kept. Precedent: `salience_novelty` made two signals
  go from 1 distinct value to 32 each, then measured 0/200 workspace flips, 0/200 act/hold decisions and
  0/200 trajectory changes at +8.4% tick cost. It was reverted, and the control that makes that
  trustworthy is that the same harness shows `drive` flipping 22/60 trajectories.
- **Runtime evidence outranks static reading**, in both directions. A coverage audit once flagged PFC
  and amygdala as dormant and both were refuted on runtime A/B.
- Growth is not improvement. A grown unit that changes no outcome is dormant, not progress, and the
  census of what the brain has is not a measure of what it can do.
- **P6 (added 2026-07-29) is the constraint that explains the ceiling above, so read it before planning
  any wave.** Every build must be able to reach every other and actually use it — reachable from another
  *faculty* through a declared route, and consumed by another faculty in a way that changes *that
  consumer's* output. The reason wiring has a ceiling at all is precisely a P6 failure: faculties are
  wired into the single three-way action gate at `cortex/brain.py::_route_percept_to_action` and not into
  each other, so a wave can charge the arm decision and still leave Language, Imagination, Gf,
  Consciousness and WholeBrain — **five categories, 45% of the weight** — untouched. Those five are the
  ones that need faculties talking to faculties, which is why the ceiling is structural rather than a
  budget problem.
  **The 48.5% figure this bullet used to quote is WITHDRAWN** (corrected 2026-07-29): it was computed
  from the *retired* 37.5% per-category roster, which this document forbids comparing against, and it
  rested on a second error — "only 4 of 11 reachable", when 11 − 5 excluded leaves **six** (the rubric
  names four explicitly; Memory and Social are unstated). Re-derived from what the instrument actually
  reports: six non-excluded categories at band 70 gives **38.6%**, four gives **25.9%**, both provisional
  on a 30-row partial roster. See [`coverage/README.md`](coverage/README.md), where the number sits next
  to its evidence. The conclusion is unchanged — wiring cannot reach 100% — and P6 does not depend on
  which figure is right, only on the five categories being structurally out of wiring's reach.
  Note P6 is **provisional and review-enforced**: its gate (diff the real faculty→faculty consumption
  graph against `connectome.py`'s declared edges) is specified but NOT built, and this file's own bar says
  that matters.
- **A route is not a conversation.** The failure P6 attracts is a telemetry bus — wire every faculty into
  a shared dict, watch `adapters` climb from 0, and call it harmony. `substrate/region_wiring.py`'s own
  source warns that registering telemetry adapters would fake liveness. A P6 claim needs P3's evidence
  aimed one level over: neuter faculty A, show faculty B's own output changes. Adding a reader that
  stores a value is not consumption.
