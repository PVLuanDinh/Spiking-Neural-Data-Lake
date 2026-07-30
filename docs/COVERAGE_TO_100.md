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
  The one genuinely absent artifact is `ar_spiking_lm.npz` — see below.
- **BLiMP's `n/a` is a path mismatch, not only an untrained model.**
  `training/ar_spiking_lm.py:48` exports to `models/spiking_lm.npz`; `reasoning/ar_spiking_lm_serve.py:53`
  loads `models/ar_spiking_lm.npz`. Neither half can see the other's file, and `:455` *prints* the
  trainer's path while the loader uses the other one — which is how it stayed invisible. The trainer's
  docstring also still calls the file `training/spiking_lm.py`.

## POLICIES.md self-claims corrected the same day

A policy file that describes a codebase which no longer exists is the failure its own standing caution
names. Three claims were checked and were false:

- **`godel_machine` was listed among the self-build modules that "fire every 8 nights."** Its
  `attempt_on_repo` (`godel_machine.py:159`) has **zero call sites** tree-wide, and the only reference
  to the module outside itself is its *filename as a string* in the self-build's protected-paths list
  (`brain_native_selfbuild.py:73`). What does fire: `brain_native_selfbuild` (`brain.py:1708`) and
  `self_build_loop` (`brain.py:2520`), with `self_mod_harness` reached via `Brain.self_improve`
  (`brain.py:3485`). Wiring the Gödel-machine adapter is now explicit open P0 work.
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
were removed (from 90,000 ms), and that 891 ms is **unattributed**. `learn_daemon.py:61` still claims
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
| 3 | **Attribute the tick** | where the 891 ms goes, by neuter-and-measure, not by profiler ranking; `learn_daemon.py:61` made true |
| 4 | **Widen + fuse** | live populations past 4,096; per-region elementwise updates fused into one batched kernel captured as a CUDA graph; shipped only on a measured win against the CPU path, regressions reported |
| 5 | **Charge the dormant** | the open P3 findings resolved as wired-and-charged / retired / exempt-with-reason, each with a neuter-A/B. Never a dead consumer to silence a gate |
| 6 | **Capability** | the two ceiling categories attacked where they are actually blocked: a real-scale AR spiking LM for Language, perceptual predicate induction for Gf. Held-out numbers or an honest null |

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
