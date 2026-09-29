# P6 INTERCONNECT -- measured topology, and the program that follows from it

**Status: MEASURE AND DESIGN. No wire built. No file under `src/`, `tests/`, `specs/` or
`run_selfchecks.py` edited. Nothing at all under `src/snn_data_lake/reasoning/**`.**

Every number below was produced by running an instrument or reading a line today
(2026-08-03), not quoted from a prior document. Where a prior document, a survey or a
review is wrong, the correction is stated with the line that settles it.

---

## 0. The measured P6 topology today

`tools/p6_gate.py`, run under `.venv-lake`, 399 of 399 files parsed:

```
writes  : 42 across 10 modules      reads : 9 across 5 modules
subscribe registrations: 10         workspace classes tracked: 3
25 of 42 writes are inside the 2 bus modules' own self-checks -> real broadcast is 17 of 42
self-reads   : 8 of 9    (a module reading a workspace instance it constructed itself)
cross-module : 1 of 9    engine/live_brain.py:287  self.arm.workspace.broadcast()
                         <- instance owned by engine/autonomous_loop.py; reader role = tick driver
FACULTY-to-FACULTY reads: 0 of 9
named subscribers: 4 registered, 2 of 4 drained
writer modules read by nobody: 7 of 10
instantiation sites: 26 across 10 modules; 11 of 26 outside the bus self-checks, across 8 modules
```

| module | role | W | R | sub |
|---|---|---|---|---|
| thalamus/global_workspace.py | BUS | 14 | 3 | 4 |
| thalamus/reentrant_workspace.py | BUS | 11 | 2 | 4 |
| reasoning/train_of_thought.py | FACULTY | 4 | 2 | 0 |
| engine/brain.py | FACULTY | 3 | 1 | 1 |
| cortex/competence_executive.py (reasoning/brain_adapters/executive_control.py until 2026-08-12) | FACULTY | 3 | 0 | 0 |
| engine/autonomous_loop.py | ACTION | 3 | 0 | 0 |
| cortex/access_consciousness.py | FACULTY | 1 | 0 | 1 |
| reasoning/rpm_solver.py | FACULTY | 1 | 0 | 0 |
| reasoning/spike_chat_lake.py | FACULTY | 1 | 0 | 0 |
| instruments/developmental_loop_brain.py (reasoning/ until 2026-08-12) | FACULTY | 1 | 0 | 0 |
| engine/live_brain.py | ACTION | 0 | 1 | 0 |

### How this differs from the 2026-07-31 reading, and why

The prior reading was **36 writes across 8 modules, 2 reads across 2**. All four deltas
are **instrument defects, not tree movement** -- `git blame` dates every affected line to
2026-07-21, ten days before that reading.

1. **`engine/brain.py` was missing from the writer list entirely.** 3 writes at `:1733`
   `:1738` `:1742` (`self._reentrant.submit`). The receiver attribute is named
   `_reentrant`, not `workspace`/`ws`/`_ws`, and the walk keyed off the name. "8 writer
   modules" was really 9 on that same tree. Largest single correction, and the same blind
   spot the current resolver still has (see W0.1).
2. **`engine/brain.py:1610` is a real, consuming read.** `self._reentrant.broadcast_to("action")`
   feeds `elig["reroute"]` into the three-factor drive-weight update. "2 reads" was really 3.
3. **`reasoning/developmental_loop_brain.py:295` is a WRITE, not one of the two reads.**
   The receiver is `WorkspaceBroadcast` (RETIRED 2026-08-12 -- retirement record in
   `cortex/competence_executive.py`; at this reading it lived at
   `reasoning/brain_adapters/executive_control.py`),
   whose `broadcast(source, content, salience)` called `_ws.submit()` then `_ws.step()`.
   `GlobalWorkspace.broadcast()` takes zero args, so a 3-arg `.broadcast(` was a submit
   wearing a reader's name. The gate split them on arity while the class lived; dlb now
   calls `submit()` under its own name.
4. **Two more writes of that same wrapper shape** inside the adapter module itself (cited at
   this reading as `executive_control.py:231`/`:234`; that bare basename now resolves to the
   OTHER module, cortex/executive_control -- the adapter is cortex/competence_executive since
   2026-08-12, and the line numbers are a dated snapshot).

Net: the prior headline was low on both sides -- it lost a writer module and a read, and
counted a write as a read. Two errors in opposite directions netting to a small number is
the silent-degradation shape. The verdict nonetheless survives the correction: every read
but one is a module reading an instance it constructed itself, and the one cross-module
read is the tick driver.

**One fact the ratio hides, and it is larger than the ratio.** 11 private workspace objects
across 8 modules *(a dated snapshot; the 2026-08-12 WorkspaceBroadcast retirement removed the
two `WorkspaceBroadcast.self._ws` entries below and ratcheted p6_gate's
private_instance_modules ceiling 7 -> 6 -- re-derive with the gate, not this list)*.
`train_of_thought.self.ws`, `access_consciousness.self._ws`, `rpm_solver`
a function-local `ws`, `spike_chat_lake` a function-local `ws`, `WorkspaceBroadcast.self._ws`
twice, `brain.self._reentrant`, `autonomous_loop.self.workspace`. Faculty A's `submit()` is
not merely unread by faculty B -- it is **invisible** to it.

---

## 1. Four structural facts, each of which kills or reorders proposals

These were found by three adversarial passes over the surveys' proposals. None is visible
from inside a single region. Each was re-verified against source today.

### F1 -- the drain is content-blind, so the bus cannot carry "which faculty spoke"

`thalamus/reentrant_workspace.py:282-312`, `_deliver`:

```python
assoc = max((v for (_src, c), v in self._assoc.items() if c == winner), default=0.0)
bias = self._REENTRY_GAIN * (1.0 + assoc) if self.reentry else 0.0
s["bias"] = {winner: bias} if bias > 0.0 else {}
```

`bias` was computed **once per tick** and assigned to every subscriber unchanged. There was
no name-dependent term anywhere in `_deliver`. The v4.78 key-space fix comment above it said
so deliberately: *support for the ignited content is a property of the content, shared by
every subscriber receiving it.* Therefore `broadcast_to("attention")` and
`broadcast_to("memory")` returned dicts **byte-identical to `broadcast_to("action")`**, which
`brain.py` already drains -- measured 35 of 35 broadcasts.

And the one live drain collapses it further: `self._reroute_bias = max(rb.values())`. A
single float. *Which* coalition ignited is discarded at the only read site in the tree.

**Consequence (as written, and it held until 2026-08-22).** Every proposal of the form
"subscribe faculty X and drain `broadcast_to('X')`" added a **duplicate reader of one scalar
that is already read**. The gate counts read *sites*, not distinct payloads, so it scored
this as +1 while zero new information moved. This killed both wires the prior program named
as its gate-movers, and it is why specs/022 SC-001 had no honest satisfier.

**FIXED IN PART BY specs/022 T039c -- the premise, not the caution.** The payload is no
longer name-blind: `ReentrantWorkspace.report_use(name, content)` gives the
`(subscriber, content)` association the writer it never had (v4.78 deleted the *read* because
nothing wrote the key; this adds the writer instead), and `_deliver` adds that per-listener
term. Measured on the same 40-tick schedule: with nobody reporting, the two channels stay
identical on 35/35 and byte-identical to the pre-change value `0.15999660973063193` -- so
every workspace in the tree that does not report is unchanged; with `action` reporting the
use it genuinely makes, they differ on 34/35 (action `0.22608` against language `0.15999`),
and the privilege decays back toward the shared base once the use stops (`0.22601` ->
`0.17493` thirty ticks later). The listener axis lives in its **own dict** (`_use`), because
`_assoc_support` aggregates by content alone and a shared dict leaked a listener's use into
the term every subscriber receives and into `association_level()`, the declared
graded-delivery source -- constructed at source support `0.10` against listener use `0.90`,
both returned `0.90`, and `max()` hides it whenever source support is near 1.0.

**THE CAUTION SURVIVES INTACT, and it is the half to keep reading.** A drain that reports no
use still receives the base gain, so *subscribing a faculty and draining its channel remains
a duplicate reader unless that faculty actually consumes what it gets and says so*. The
payload can now carry a listener distinction; it does not do so for free. The gate still
counts sites rather than payloads, so a wire is still worth exactly the information it
moves -- which is the sentence this section existed to write down.

**AND THE FIX HAS A MEASURED CEILING, stated here so the next reader does not over-read it.**
The listener term settles at a fixed **~0.833x** multiple of the shared `(source, content)`
association, because a broadcast a faculty consumes arrives on the same tick set that
potentiates that association at the same rate. So the payload separates *a subscriber that
consumes* from one that does not -- it does **not** grade DEGREE of use, and two subscribers
that consume alike receive byte-identical dicts again. The graded version was built and
measured and is a **null**: gating the report on the re-route being material
(`elig["reroute"] > mean_e`, the loop's own baseline, no new threshold) fires on **0 of 76**
live winner-ticks, because the re-route is always a below-average source -- the same fact
`tools/charge_probe.py` reports as 34 of 40 internal reads changed and 0 of 40 decisions.
Grading this honestly requires the re-route to matter more first; that is a different task
from giving the key a writer, and F1's SPEAKER half (the payload still carries no submitter
identity) is untouched by either.

### F2 -- `_deliver`'s availability is all-or-none by construction, and a live comment claims otherwise

Same function. `ignited` is loop-invariant, so `reached` is 0 or `len(self._subs)` and
`availability` is in `{0.0, 1.0}` for **any** subscriber count. `engine/brain.py:711-712`
asserts the opposite in as many words: *"Subscribing also makes `availability` a GRADED
fraction-of-subscribers-reached rather than a binary, de-saturating global access."* That
claim is false in current source. A comment asserting a mechanism the code does not
implement is the drift pattern POLICIES keeps flagging, and this is a live instance.

It is not telemetry-only. `_reentrant_availability` reaches
`SecondOrderMonitor.observe()` at `brain.py:2967` (inside `respond()`), whose
`_second_order_hold` gates two `respond()` branches at `:3074` and `:3080`. So a subscriber
added today reweights a consciousness-adjacent number the moment anyone fixes the graded
loop -- and "2 distinct" is the signature of the all-or-none loop, not headroom.

### F3 -- nine of fifteen candidate wires terminate in the single action gate

`_route_percept_to_action` is `engine/brain.py:1398`. AST-resolved enclosing function for
every line below, so this is not eyeballed:

```
brain.py:1516  cue = {"where": place, "affect": self._emotion}   -> _route_percept_to_action
brain.py:1610  rb = self._reentrant.broadcast_to("action")       -> _route_percept_to_action
brain.py:1635  self._drive_w["reroute"] * elig["reroute"]        -> _route_percept_to_action
brain.py:1694  self.brain.arm._ext_drive = ext                   -> _route_percept_to_action
autonomous_loop.py:687  return [td[i] + ext[i] ...]              -> _route_signals -> selection
```

The transitive consumer closure of the fifteen ranked proposals lands: **9 of 15 at the
action gate, 2 of 15 in a `respond()` string, 2 of 15 off-tick, 1 of 15 in a telemetry
dict, 1 of 15 (partly) in a durable store that has not yet fired.**

This is the defect the phase exists to avoid, and the prior program walked into it while
banning it by name. K-GATE forbids "a new `_drive_for` term / `elig[...]`" --
and `elig["reroute"]` *is* the workspace's only read. Every chain that lands on
`ignition_threshold` lands on a `_drive_for` term through a rename.

### F4 -- `.venv-lake` has no snntorch, so every content-distinctness number is NOT-ELIGIBLE

Verified today: `snntorch False, torch True` in `.venv-lake`; `snntorch True` in
`.venv-cu128`. So `_perceiving_real=False` and `_perceived_digit is None` always, which is
why `brain.py:1735` submits the literal fallback `"ctx"`.

Sharper than the survey stated: `brain.py:1742` submits salience with content
`("threat" if threat > 0.3 else "novel")`, and `threat` is ~0 live, so that content is the
constant string `"novel"` **in both venvs**. The bus's entire live content alphabet is
`{percept-or-"ctx", "self", "novel"}` -- three keys, one of which is a placeholder.

`brain.py:2320` also pins the access path: `real = 1.0 if (self._perceiving_real and
self._perceived_digit is not None) else 0.0` feeds the ignition strength, so the "real
percept bump" term is dead in `.venv-lake` and the ignition test reduces to pure
novelty-vs-baseline. Any access-consciousness baseline measured in `.venv-lake` has one of
its two terms pinned off.

---

## 2. The headline metric is replaced

The prior program's target was `FACULTY-to-FACULTY reads: 0 of 9 -> 2 of 11`. **That table
is withdrawn.** Both wires that moved it read `engine/brain.py`'s own `_reentrant`, whose
only writer is `brain.py` itself (`:1733` `:1738` `:1742`, all brain.py's own computed
values). They would have counted as faculty-to-faculty only because the gate's role table
tags `engine/brain.py` FACULTY. Substantively: the god object talking to itself through a
bus, read by two of its own submodules, delivering a payload F1 proves identical to the one
already read. Not one of the 7 orphaned writers is connected by either wire.

Optimising that number is the same failure as wiring into the single action gate, one level
up. The gate's own output already says so:

> Adding a reader is NOT sufficient while each writer holds a PRIVATE instance -- see the
> instance census above; there is no shared object for a reader to read.

**The metric this program is judged on instead, all three reported together:**

| figure | today | what it means |
|---|---|---|
| **distinct non-gate terminals** | **1 of 15** | of the candidate closures, how many end somewhere other than `_drive_for`/`_ext_drive`/`_pfc_bias`/`q[unit]`, a telemetry dict key, or a `respond()` string |
| private workspace instances | 11 | mutually invisible buses outside the bus self-checks |
| orphaned faculty writers | 7 of 10 | writer modules read by nobody |
| faculty-to-faculty reads | 0 of 9 | kept, reported, **not targeted** |

The single non-gate terminal today is NM2's eviction path, and it has not yet fired
(61 of 256 slots). A wave that raises the read count while leaving distinct non-gate
terminals flat has, by this repo's own honesty guard, built a telemetry bus.

---

## 3. Wave 0 -- the instrument, before any wire

Seven items. This is the bulk of the phase, and that is the honest shape of it: three
adversarial passes found more defects in the instruments than wires worth building.

**W0.1 -- the resolver, validated on a fixture and not on the headline.**
`tools/p6_gate.py:411-428` resolves a receiver's last token against modules that *construct*
an instance bound to that token. A consumer holding an injected `self._ws` matches two
owners (`access_consciousness.py:92`, `executive_control.py:102`), goes ambiguous, returns
`None`, and the read is filed as a self-read. This is the same class of blind spot that lost
`engine/brain.py` from the 2026-07-31 walk.
Two rules on the fix. (a) Fix the resolver, not the attribute name: naming an injected
attribute `_reentrant` because that token resolves uniquely is Goodharting the instrument
built to catch Goodharting. (b) **Prefer a dynamic resolver** -- wrap the instance and record
the calling module at read time, the `tools/charge_probe.py` idiom this repo already trusts
-- over any static one, because a static affordance (annotation, parameter class) is the same
failure one step out.
**Self-check, corrected.** The prior program pre-declared *"the headline must read 0 of 9
unchanged; if the instrument fix alone moves it, the fix is wrong."* **Rejected as written**
-- if the resolver mis-files real cross-module reads, the correct outcome of fixing it is
that the number moves, and requiring it to stay put pressures the fix toward preserving the
prior. Replace with: validate on a synthetic fixture containing one known-correct injected
cross-module read and one known-correct self-read; publish a hand audit of all 9 current
reads; require the fixed resolver to reproduce the hand count, whatever it is.

**W0.2 -- bound the subscriber inbox, and stop the gate calling it drained.**
`reentrant_workspace.py:288/307` appends to `s["inbox"]` every tick for every subscriber and
nothing ever trims it. `broadcast_to()` (`:140-146`) returns `dict(s["bias"])` and **never
touches `inbox`**. So the two currently-"drained" subscribers are leaking lists; at ~28 ms
per tick that is roughly 3M entries per subscriber per day on a long-running daemon.
`tools/p6_gate.py:546-547` marks a subscriber drained on a `broadcast_to` call, so the
instrument certifies the leak as healthy.
Fix both: `deque(maxlen=...)` in `subscribe()` (or pop in `_deliver`), and make the gate's
drained test require an actual `inbox` read. **Pre-declared self-check: the reported
"2 of 4 drained" must go DOWN.** An instrument fix that raises its own headline is the fix
being wrong.

**W0.3 -- derive the jitter from the content, not from a shared RNG stream.**
`reentrant_workspace.py:153-158` draws `self._rng.uniform(0, _JITTER)` lazily on first sight
of a content, and `submit()` records `self._order[c] = len(self._order)`, a monotone
never-cleared registry that `_simulate` sorts by and the ignition tie-break reads. **Change
the arrival order of contents and every subsequent content's jitter changes.** This is
dormant only because of F4: today the percept content is the literal `"ctx"`. The moment the
census moves to `.venv-cu128`, content keys become data-dependent and arrival order becomes
data-dependent with them. It is not seed noise, so "sweep 7 seeds" does not control for it,
and it will make a dead wire look alive on a majority of seeds.
Fix: `zlib.crc32(str(content).encode())` scaled into `[0, _JITTER)` -- already this repo's
mandated convention (POLICIES: *"`zlib.crc32` never builtin `hash()`"*). **Pre-declared
self-check: with crc32 jitter and no wires, a 300-tick run must produce the identical winner
sequence to a run where the same contents are pre-registered in a different order.** Today
it will not. **This must land before W0.6, because W0.6 activates the hazard.**

**W0.4 -- the action-clamped A/B in `tools/charge_probe.py`. Highest-value item in the phase.**
Every A/B proposed anywhere in these surveys neuters writer A and looks for a change in
reader B's output. On this tree, `A -> gate -> action -> next-tick percept -> B` is a live
path on **every** wake tick, so a neuter that changes the selected action changes B's output
whether or not any direct edge exists. Nine of fifteen wires are exposed. `charge_probe.run(
source, n, neuter)` neuters a `_drive_w` entry through a `_PinnedWeights` dict subclass and
has no facility to control for this.
Add a mode that **replays a fixed action sequence** in both arms (force `res["action"]` from
a recorded baseline trace; the reflex/E-STOP path stays intact). Then: neuter A with the
action clamped, and if B's output still differs it is a genuine faculty-to-faculty edge; if
B differs only in the unclamped run, the edge is routed through the action gate and the wire
is a spoke. Run it on the two existing reads (`elig["reroute"]`, and whatever W0.1 surfaces)
before building anything. It retro-validates the `:969` and Claustrum reverts for free.
**Nothing downstream of this is interpretable without it.**

**W0.5 -- broaden the threat/salience drive, using the in-tree precedent.**
`threat` is pinned at ~0 live and it is the load-bearing evidence for three separate
proposals (NM1's incumbent `1.0 + thr` is 1 distinct on 120 of 120; NM2's `arousal=threat`
is 0.0 on 61 of 61; HP3's competitor is 0.0 on 80 of 80). Four downstream repairs were being
planned around one upstream pin. This repo already fixed the identical pin once, at
`engine/brain.py:1059-1069`:

> LC PART 1 -- BROADEN THE DRIVE: LC fires on SALIENT events, but it was fed threat only, and
> threat ~ 0 in the idle loop, so its level sat at a dead tonic constant. Feed it the real
> salience signals already computed this tick -- threat OR novelty OR predictive surprise

Apply the same template and re-run the three pinned-argument censuses afterwards.
**P1-safe:** it replaces a single-source feed with signals already computed in the same
scope; it tunes no constant. **I reject the prediction that NM1 and HP3 "discharge for
free"** -- that is an untested hypothesis, and the whole point of W0.5 is that the censuses
get re-run before anyone claims it.

**W0.6 -- re-run the variance census in `.venv-cu128` (F4), including the ACCESS path.**
Affected: every content-distinctness number, plus `_access_ema` / `access_index` /
`_access_state`, whose `real` term is pinned off in `.venv-lake` (`brain.py:2320`). Must run
**after** W0.3.

**W0.7 -- build the connectome-diff half of the gate.**
POLICIES specifies the instrument as: *derive the actual faculty-to-faculty consumption
graph, diff it against `connectome.py`'s declared edges, and report faculties whose
in-degree and out-degree are both zero outside an entrypoint.* `tools/p6_gate.py:92-96` says
outright it does not do this. The consequence is not cosmetic: **twelve of fifteen wires
move no gate number at all**, and after the reclassifications below, nine of them are
covered by no aggregate instrument whatsoever. A program whose majority is unmeasurable by
any instrument it plans to build is not mechanized.
`connectome.py` verified today: **81 edges, 40 nodes.** Two counts, both cheap from the
existing AST walk: *declared edges with a live call-graph realization, N of 81*, and
*faculties with in-degree and out-degree both zero outside an entrypoint*. The non-bus wires
are exactly connectome-shaped edges -- NM1 is neuromodulation-to-amygdala, NM2 is
neuromodulation-to-hippocampus, CX1 is amygdala-to-cortex-to-hippocampus. If it returns a
rubber-stamp-sized list, that kills the gate and is a complete result.

**Also recorded in Wave 0 as declared defects, not wiring opportunities:** the closed
`novelty` port on `MetacognitiveMonitor` (see F-KILL below); `'language'` registered at
`brain.py:714` and never drained; `brain.py:711`'s false graded-availability claim (F2);
`connectome_router`'s "FIXED" incident row, whose return is consumed into `state()` at
`brain.py:2877`, i.e. into telemetry, which is the failure the same POLICIES section defines.

---

## 4. Ranked wires

Ranking is **ordinal**. No measurement supports a cardinal ratio and manufacturing one would
be the P1 defect this program audits. Region IDs in one namespace: **CP**=cortex-perception,
**CX**=cortex-executive, **HP**=hippocampus, **NM**=neuromodulation, **TH**=thalamus/BG,
**RC**=reasoning/cerebellum.

| # | id | producer -> consumer | terminal | wave |
|---|---|---|---|---|
| 1 | **NM2** | AffectIntegrator arousal -> `AffectiveEpisodicStore.store(arousal=)` | `respond()` note + store eviction (non-gate) | **W1** |
| 2 | **RC2** | `ForwardModel.residual` -> `pc_surprise` | action gate (twice) | W2, split, gated on W0.4 |
| 3 | **RC3** | `arc_solver.components()` -> `rpm_solver.solve()` | off-tick | W2, blocked by scope |
| 4 | **CP1** | `PcCortexServe.generate/infer` -> `brain.imagine()` | `respond()`; **P6(a) fails** | W2, Imagination ledger |
| 5 | **CP2** | real percept -> `NeurogenesisEngine` (contract change) | telemetry + self-model | W3 |
| 6 | **CX4** | `TaskSetLayer.active_set` -> spiking curiosity context key | action gate | W3, gated |
| 7 | **CX1** | `_appraised` -> `AffectCore.recognize()` -> episodic | action gate | W3, re-scoped |
| 8 | **NM1** | Cortisol `learn_rate_scale` -> `AffectiveAppraisal.learn(neuromod=)` | action gate | **P1 ledger** |
| 9 | **CX3** | computed stakes -> `LearnedEVC.allocate(stakes=)` | action gate | **P1 ledger** |
| - | TH3, HP2', TH2, NM3, TH1, RC1, HP1, CX2, HP3, CX5, CP3, TH4 | see section 5 | -- | **KILLED** |

### W1.1 -- NM2 -- the one Wave-1 wire

- **Producer.** `neuromodulation/affect_integrator.AffectIntegrator.integrate()` arousal,
  26 distinct over 120 ticks.
- **Consumer.** `hippocampus/affective_memory.AffectiveEpisodicStore.store(fragment,
  valence, arousal, reward)` at `engine/autonomous_loop.py:1647`.
- **Signal.** Replace `arousal=threat` with the fused core-affect arousal `aa`.
- **Consumer output that changes.** Which episode `recall_by_emotion(v, a, k=1)` returns --
  measured counterfactual on the live 61-episode store moves top-1 from tick 47 to tick 26.
  It also restores one of three terms in `_importance()`, which governs eviction from the
  256-slot store. The recall terminal is `_somatic_recall()` inside `respond()`; the
  eviction terminal is durable and is the one non-gate terminal in the whole candidate set.
- **P1 story.** Removes a constant, adds none. `arousal=threat` is 0.0 on 61 of 61 stored
  episodes, so `recall_by_emotion`'s `|da|` term adds an identical constant to every
  candidate -- **the arousal half of somatic-marker retrieval is provably a no-op today.**
  `store()` takes the raw `pattern`, so nothing else in the call shifts.
- **Pre-declared falsifier.** A/B on the **argument**, not the producer (see below):
  `arousal=threat` vs `arousal=aa`, same seed, 200 ticks. Dead if the `tick` field of
  `_somatic_recall`'s returned marker is identical on every tick a marker is returned **and**
  the surviving episode set is identical. Opposite-sided falsifier (see section 6): dead also
  if the returned marker's distinct count collapses to 1, or its longest run exceeds the
  7-seed baseline spread. **NOT-ELIGIBLE branch:** if the run never reaches capacity and
  never returns a marker, report NOT-ELIGIBLE, not a dead wire.
- **Correction to the survey that proposed it.** The survey claimed
  `AffectIntegrator.integrate()` arousal is "consumed by nobody today." **False.**
  `brain.py:2031-2035` does `av, aa = integrate(...)` then `observe(av, aa)` then
  `_emergent_emotion = label()`, which reaches telemetry at `:2870` and a `respond()` string
  at `:3228`. This does not kill the wire -- the incumbent `arousal=threat` is still pinned,
  and the consumer is a different faculty -- but it **does** invalidate the producer-neuter
  A/B, because pinning `integrate()` to `(v, 0.0)` collapses the arousal axis of the emotion
  prototype space at the same time. Hence: neuter the argument.
- **Honest caps, shipped with it.** The recall terminal is a `respond()` note, so this is
  P6-clean and P3-weak -- the same disposition P6 itself assigns to
  `MetacognitiveMonitor.hold()`. The eviction terminal has not fired (61 of 256). There is a
  one-tick lag: the store write is in the arm's `step()`, the integrator runs later in
  `_c1_tick`.
- **Wave 1.** It is the only wire in the program that does not terminate at the action gate.

### W2.1 -- RC2 -- reafference cancellation, two changes, gated on W0.4

- **Producer.** `ForwardModel.residual` (cerebellum).
- **Consumer.** `pc_surprise`, drained at `brain.py:1730-1731` into the reentrant percept
  coalition's salience, and separately into `elig["surprise"]`.
- **Signal.** Two separate changes, each A/B'd alone. (i) Wire `ForwardModel.update()` on the
  arm's realised post-gate command so `a_hat`/`b_hat` stop being the literals `-0.5`/`1.0`.
  (ii) Only then subtract the reafference so `pc_surprise` reflects externally-caused change.
- **Consumer output that changes.** `pc_surprise` -> reentrant ignition -> which content wins
  the broadcast. **This is the only proposal in the set whose consumer sensitivity was
  measured before the proposal existed:** a `pc_surprise` neuter changed `reentrant_strength`
  on 59 of 60 ticks.
- **P1 story.** (i) discharges two numeric literals arbitrating a learned model's predictions.
- **Pre-declared falsifier.** Ignition count and broadcast *winner* identical on 300 of 300.
  Second falsifier, the one expected to bite: `detect()` firing on 0 of 300 or 300 of 300 is
  a constant carrying no information -- the `:969` shape, and the same trap as `blocked`
  firing on acceleration. Third: a **scramble control** (payload replaced by a shuffled
  value of identical shape and entropy) must differ from the real arm; if it does not, the
  wire injects noise, not information.
- **Terminal, stated plainly.** Action gate, twice (`elig["reroute"]` via ignition, and
  `elig["surprise"]` directly). It is ranked 2 on evidence quality, not on P6 merit, and it
  **must not land before W0.4** returns a verdict on whether the reentrant path can carry
  anything to a non-gate consumer.

### W2.2 -- RC3 -- arc_solver components into rpm_solver

- **Producer / consumer.** `reasoning/arc_solver.components()` -> `reasoning/rpm_solver.solve()`.
- **Signal.** Promote a bench-only consumption to a declared route: the pipeline already
  exists and is already asserted inside `real_rpm_bench.py:418-427`, real consumption living
  inside a script's `main()` -- the identical shape as `nightly_skills.route()`.
- **Consumer output that changes.** `rpm_solver` answers matrices given as rasters, so its
  `known` flag and answer index change on perceived input.
- **P1 story.** None; it adds no constant and removes none.
- **Pre-declared falsifier.** `raven_bench` and `real_rpm_bench` `solved` and `known_frac`
  unchanged, and no winning rule sourced from a perceived cell.
- **Honest cap.** Both endpoints are off the live tick (`brain.py` imports neither), so this
  closes P6's *"Gf isolated from Perception"* incident row and **does not raise P3**.
- **SCOPE BLOCK.** Both files are under `src/snn_data_lake/reasoning/**`, which HARD
  CONSTRAINT 1 forbids touching this pass. Re-scoped to **measurement-only** for this phase,
  or handed off after the concurrent agent lands. The prior program scheduled it without
  noticing.

### W2.3 -- CP1 -- grounded imagination, on the Imagination ledger not the P6 ledger

- **Producer.** `reasoning/pc_cortex_torch.PcCortexServe`, a loaded 6.4 MB checkpoint
  (`dims [2312,128,64,32]`, `test_acc 0.748`) with zero in-edges and zero out-edges;
  `surprise`, `generate`, `predict`, `make_efe_value`, `reconstruct` have zero call sites.
- **Consumer.** `brain.imagine()` at `brain.py:3602`, today a `PredictiveCodingNetwork(dims=
  (8,4,2))` clamped to a crc32-random 2-d cause.
- **Signal.** Class-prototype top cause `z_c` = running mean of `infer(frame)` per perceived
  class -> `generate(z_c)` -> a 2312-d observation -> `classify(x_hat)`.
- **Consumer output that changes.** A grounded event-frame image instead of an 8-d toy
  vector; 6 of 10 classes re-perceive as themselves, pairwise cosine min -0.699 / mean 0.284,
  against an incumbent that collapses to **4 distinct classes of 10**.
- **P1 story.** No new constant -- the prototype is a mean, the gate is the model's own
  `classify`. Zero tick cost: `imagine()`'s only caller is `brain.py:3010`, behind a literal
  `imagine`/`picture`/`visualize` first-word test.
- **Pre-declared falsifier, CORRECTED.** The survey's falsifier -- *"the reply is
  byte-identical on 200 of 200 cues"* -- **will fire while the imagery works perfectly.**
  `brain.py:3010-3013` renders the result as `{perc[:6]}`, the first six elements. Today
  `perc` is 8-d so six elements carry the signal; a 2312-d N-MNIST frame's first six pixels
  are corner background and near-certainly `0.0` for every class. Given this program counts
  exemptions as deliverables, a false exemption is the most expensive error it can make.
  Replace with: `classify(generate(z_cue))` independent of the cue (<=1 of 10 hits, or
  all-pairs cosine >0.95); **or** the imagine branch fires 0 times over a 200-turn transcript;
  **or** the *label* is identical on 200 of 200 cues. And before building: print
  `[round(v,4) for v in generate(z_c)[:6]]` for all 10 prototypes; if they collide, change
  the readout to the label plus a frame statistic.
- **Second contract break to fix.** `imagine()`'s two producers return different
  dimensionalities (8 vs 2312) depending on whether the checkpoint loaded, i.e. on F4. That
  is a silent contract fork on an environment difference. Pin the branch and have the reply
  name the producer, or F4 becomes a heisenbug in the chat path.
- **Why it is not a P6 wire.** P6 (a) requires reachability *"through a typed, declared route
  rather than only from an entrypoint or a chat handler."* `imagine()` has exactly one
  caller, in `respond()`, on a keyword match. POLICIES' own incident table already contains
  this shape (`reasoning/pragmatics.py` -- *"wired to chat, not to the brain"*). Honest
  disposition: **P6(b) satisfied, P6(a) chat-handler-only, blocked on the same input-free
  interval as the WM attractor** (section 7). Ship it as an Imagination/P3 quality fix; the
  evidence is strong and the 10-digit cap is well stated.

### W3.1 -- CP2 -- real percept into NeurogenesisEngine, under a named contract change

- **Producer / consumer.** The live percept stream -> `cortex/neurogenesis_engine.tick()`,
  which before 2026-09-02 took no argument and drew features from `_mode_features(self.current_mode)`
  with `self._chain = random.Random(seed)`; `brain.py` called it bare -- while its
  docstring claimed it grows "to predict the brain's own next perception."
  **BUILT 2026-09-02 (specs/022 audit s8 rank 2a) under a different contract than the one below:**
  `NeurogenesisSelfImprove.observe(percept, category)` on every wake tick, target = the brain's
  OWN next category (the arm learner's prototype id), not the digit class -- the "current class"
  target proposed below would duplicate the encoder's own classification (P2). Measured: NULL
  under its pre-registered rule (v2, after a panel): the grower carries nothing beyond the class
  prior, reaches the stream's persistence floor (0.93) on 3 of 8 seeds, and a persistence oracle
  does no better -- the earn-your-place validation window on a bursty stream is the limit, not
  the stimulus; attribution by band is a null. The chain is retired from the live path. Record:
  `specs/022-unified-brain-arc/evidence/s2-real-percept-stream.md`.
- **Signal, with the contract change named not sneaked.** On the live pack the *next*-mode
  contract is unlearnable (0.180 in-sample over 255 transitions with 10 states, ~2.5
  samples/cell, an overfit upper bound, against a 0.100 floor), so feeding it the real stream
  under that contract would correctly grow nothing. The learnable target is the **current**
  class (0.917).
- **Consumer output that changes.** `grown_units`, `predict_accuracy`, `growth_state_bytes()`
  -- read at `:2288` (access-assembly size) and `:2190`/`:2444` (self-model).
- **P1 story.** Replaces a synthetic seeded chain with real data; adds no constant.
- **Pre-declared falsifier.** Held-out `predict_accuracy` at chance 0.100, **or** the
  `(tick, n_units, acc)` trajectory matching the synthetic-chain run refit-for-refit.
- **Blocker stated with it.** `neurogenesis_engine` has no persistence path (POLICIES
  incident row), so grown units die every restart and this wire compounds nothing until that
  is fixed. Terminal is mostly telemetry plus the self-model.

### W3.2 -- CX4 -- task set into curiosity, consumer corrected, gated

- **Producer.** `TaskSetLayer.active_set`.
- **Consumer, CORRECTED.** The survey named `neuromodulation/intrinsic_motivation.CuriosityDrive`.
  **That class is not on the live path:** `brain.py:199` passes `spiking_curiosity=True`, so
  `autonomous_loop.py:332-333` constructs `SpikingCuriosityDrive` and the rate class is never
  built live. The wire must target the spiking drop-in or it is a wire to a `None`.
- **Signal.** Partition learning-progress statistics by active task set.
- **Consumer output that changes.** `_lp_mean`.
- **Mandatory pre-check.** Count set shifts over 500 live wake ticks. If `active_set` is 0
  shifts on 500 of 500 the wire is dead by construction and must be abandoned, not defended.
- **Pre-declared falsifier.** `_lp_mean` identical to 4 dp on 500 of 500. Honest caveat:
  partitioning will initially *lower* measured LP after each shift -- a legitimate result to
  report, not a regression to hide.
- **Terminal.** Action gate. `_lp_mean` reaches `brain.py:1024` -> `elig["novelty"]`, plus
  `tgt_v` at `:1986` and EVC `reward` at `:2169`. And one of its four consumers is
  `learn_gate -> _ll`, which F-KILL proves saturated. Wave 3, and the claim must rest on the
  other three.

### W3.3 -- CX1 -- learned appraisal into the affect label, re-scoped

- **Producer.** The learned `(valence in [-1,1], arousal in [0,1])` at `brain.py:1980`.
- **Consumer.** `AffectCore.recognize()` at `brain.py:2739`, today hand-assembled from
  `self_state["valence"]` and `abs(arm._rpe)`. Range- and semantics-exact; no adapter, no
  constant.
- **P1 story.** Discharges a hand-assembled cue in favour of the learned one.
- **Terminal, AST-resolved.** `_emotion` is read at `brain.py:1516` -- **inside
  `_route_percept_to_action`** -- building the recall cue whose `_recall_conf` becomes
  `elig["recall"]` at `:1533`. Also at `:1761` as an episode affect tag, and `:2808`
  telemetry. So "three faculties downstream" is really one action-gate term, one episode tag,
  one telemetry key.
  **I reject one reviewer's claim that `:1516` sits inside `_tick_ensemble`.** AST resolution
  puts it in `_route_percept_to_action` (`:1398-...`). The correction makes CX1 worse, not
  better: it is not merely a tick-lagged memory cue, it is an action-gate term.
- **The prerequisite that is cheaper than the wire.** `recognize()` is called inside
  `def state(self)` (`:2720`), *"Telemetry for /api/state"*, which is also called out of band
  by `infra/brain_telemetry.py:212`, `engine/learn_daemon.py` and
  `infra/nengo_bridge.py:307`. So the affect-to-memory route executes only because
  something polls, and its lag is a full tick. Move the `recognize()` call into `_c1_tick`
  immediately after `:1981`, one line, and A/B that alone: `_emotion` sequence, `rec["key"]`,
  `_recall_conf` over 300 ticks. **If moving the call alone changes nothing, CX1 has no
  headroom and is abandoned before it is built.** Then assert `state()` side-effect-free as
  an invariant rather than leaving it a hope.
- **Pre-declared falsifier.** Identical label on 200 of 200; **or** label differs but
  `rec["key"]` and `_recall_conf` are identical on 200 of 200 -- the cue is discretized to 8
  prototypes and the WTA can swallow a small shift. That second condition is the real risk
  and is the number to report.
- **Wave 3, re-scoped.** A quality improvement to the recall cue. It becomes a P6 claim only
  if a second consumer of `_emotion` is found off the `_route_percept_to_action` path;
  `EpisodeConsolidator.store()` at `:1761` is the candidate, and `_episode_retained` must be
  shown to change independently of `elig["recall"]` before "two faculties" is claimed.

### P1 ledger -- two real defects, removed from the P6 count

**NM1 -- Cortisol `learn_rate_scale` into `AffectiveAppraisal.learn(neuromod=)`.**
`brain.py:1989` passes `neuromod=(1.0 + thr)`, measured **1 distinct on 120 of 120** because
`threat` is 0.0 on 80 of 80: a three-factor rule running with its third factor pinned. The
producer is named in `affective_appraisal.py:25` and `:119` and was never plugged in. Real
P1 defect, worth fixing. **Not a P6 wire, and four separate hazards must be in the writeup:**

1. *Terminal is the action gate.* `_appraised` -> `brain.py:2131` affect raster ->
   `RegionCoherence` -> `_bound` -> `ignition_threshold` -> ignition ->
   `broadcast_to("action")` -> `elig["reroute"]` -> `_drive_for`.
2. *It is a sign-asymmetric rectifier, not a gain.* `cortisol.py:46-49` returns
   `1.0 + maxmod*level` iff `signed < 0`, else exactly 1.0 -- **a no-op by construction on
   66 of 120 ticks**. `affective_appraisal.py:126-140` applies `gate = max(0.0, neuromod)` to
   the positive branch, the negative branch **and the arousal branch**, so the arousal pool's
   update rate is chosen by the sign of the *valence* error. With bounded potentiation and
   decay, a rule whose gain is larger for `err_v < 0` parks `v` systematically below `tgt_v`.
   That is a ratchet. And `brain.py:2131` takes `abs()` of it, so drift toward -1 pushes the
   coherence raster monotonically toward 1.0: **the declared success signal fires, then
   saturates**, and at 200 ticks saturation is indistinguishable from success. Run 5,000
   ticks and report the `_bound == True` fraction in the first 500 versus the last 500.
3. *The chain passes through a boolean.* `brain.py:2153-2158`:
   `thr = ignition_bias(base) if self._bound else float(base)`. A 9%-mean learning-rate
   change reaching a boolean either flips it or is discarded; there is no graded transmission
   at that node. Instrument `_bound` directly and report its flip count as the **primary**
   number, with `_appraised` distinctness secondary. `_appraised` is rounded to 3 dp and may
   not clear that quantum anyway.
4. *The A/B must neuter the argument, not the producer.* Pinning `cort.level` to 0.0 also
   disables `autonomous_loop.py:1530` -- `self.q[unit][action] += self.lr *
   self.cort.learn_rate_scale(r) * (...)`, the live Q-learning rate -- plus `caution_bias()`
   at `:746` and `:815`. The neuter changes the selected action, and any `_appraised`
   difference measured that way is unattributable.

Also: the argument must be the appraisal error `tgt_v - v` (negative on 54 of 120), not
`tgt_v`, which is rectified and never negative and would make the wire dead by construction.

**CX3 -- computed stakes into `LearnedEVC.allocate(stakes=)`.**
`brain.py:2168` reads `allocate(conflict, 0.5)`: a numeric literal arbitrating behaviour, and
worse -- the *learned* controller is fed a constant while the hardcoded one beside it gets a
computed value. Genuine P1 debt. **Not a P6 wire:** the only consumer of `LearnedEVC`'s
`cost`/`value` is `brain.py:2178 arm._learned_ctl` -> `_pfc_bias` (`autonomous_loop.py:1015`)
-> `evc()` -> `gain` -> top-down preference over `ACTIONS`. The survey's stated consumer
output -- *"the learned cost and value trajectories diverge, which is LearnedEVC's own
output"* -- is circular: a faculty's own state changing is the producer side of P6 (b).
Three things the fix must carry:

- **Use the BG `_decision_margin`, not `_metacog_rel`, and the reason is ordering.** AST
  resolution: `_c1_tick` (which contains `:2168`) runs **before** `_update_metacog` (which
  writes `_metacog_rel` at `:1934`). So `stakes = 1 - _metacog_rel` reads tick t-1 while
  `conflict` is tick t -- two factors of one product from different ticks -- and
  `_metacog_rel = 0.0` at `:262` puts tick 0 at stakes 1.0, the top of the range, for the
  first three-factor update. `_decision_margin` is set inside `brain.tick()` and is same-tick
  fresh. The two producers are **not** interchangeable.
- **Chatter falsifier.** `engaged = (gain > 0.0)` is a hard relay inside a delayed feedback
  loop (stakes -> gain -> engaged -> learn -> `_learned_ctl` -> arm -> margin -> stakes).
  "Dead if `engaged` identical on 300 of 300" passes cleanly while the controller oscillates
  at period 2. Require run-length distribution and lag-1 autocorrelation inside the 7-seed
  baseline spread, and sample `signature()` at ticks 50, 150 and 300 -- a limit cycle returns
  to the same weights, so an end-of-run match proves nothing.
- Report it under P1, and drop it from the "pinned arguments" figure, which then reads
  **2 today -> 0 after the P1 pass**, not 3 -> 0.

---

## 5. KILLED -- decorative, or the action gate in disguise

This section is worth as much as the plan. Twelve proposals die here, four of them the prior
program's own headline items.

### Killed by F1 -- the bus cannot carry them

**TH3 (was ranked #2, was one of two gate-movers): reentrant ignited content ->
`ReticularGate.attention()`. KILLED THREE WAYS.**

1. *F1.* `broadcast_to("attention")` returns a dict byte-identical to `broadcast_to("action")`,
   which is already drained. A mandatory pre-check would have caught it and is cheaper than
   the build: over 300 live ticks, assert `broadcast_to("attention") != broadcast_to("action")`
   on at least one tick. Under current source that fails on 300 of 300. Inventing a
   per-subscriber term to fix it would re-create the axis v4.78 removed **because it was
   never learned**.
2. *The identical experiment is already in the tree, reverted.* `engine/autonomous_loop.py:
   906-917`, a comment headed *"MEASURED DEAD, and left dead deliberately -- do not 'fix' this
   cut without reading this"*: wiring `Claustrum.broadcast()` into `_attn_gain` changed that
   gain on 25 of 200 -- *"but a neuter-A/B moved NOTHING downstream: selected action 0/200,
   arm act/hold 0/200, trajectory 0/200, recognized units 4 vs 4, grown units 6 vs 6.
   CONSUMED-BUT-NOT-DECISIVE, so it was reverted."* Same consumer variable, same range. The
   structural reason is pre-declared in the same comment: *"even fed a reachable bar, a gain
   change in [0.5,1.0] has no leverage on a 4-unit prototype learner's vigilance test"* --
   and `_attn_gain` is constructed at `:887-891` as `0.5 + 0.5*(m/mx)`, bounded to [0.5,1.0]
   by construction. The prior program cited `:969` (the *other* revert, sixty lines down the
   same file) as its worked example and missed the on-target one.
3. *Its stated consumer output does not exist.* TH3 claimed the changed output is
   *"the percept written into the lake."* `_attn_gain`'s only application is
   `autonomous_loop.py:1284-1292`, producing `percept`, which reaches `learner.observe()` and
   `z_lq` for `latent_q` (itself EXEMPT, not constructed live). The **raw `pattern`** is what
   goes to `_lake_write(pattern, goal, action)` (`:1115`/`:1397`), to `_affmem.store(pattern,
   ...)` and to the amygdala. Nothing writes `percept` to the lake.
   And `_attended`, its other stated output, has exactly one reader in the whole tree:
   `brain.py:2921`, `getattr(self.brain.arm, "_attended", -1)` inside a telemetry dict.
   Making a 3-distinct telemetry field into a 4-distinct telemetry field is `:969` verbatim.

TH3's own falsifier could not have caught this: it declared dead if *"the encoded percept is
byte-identical on 200 of 200,"* but a multiplicative gain change makes `percept` floats
differ by construction, so it can never fire -- while the Claustrum run proves action and
trajectory can be 0 of 200 anyway. **A falsifier the tree already contains a counterexample
to.** The correct disposition: `_attn_gain` is EXEMPT-with-reason and no producer can rescue
it. `ReticularGate.attention()` remains a declared port with zero live callers -- a real
finding, recorded as a dead port on a consumer chain measured non-decisive, not as a wiring
opportunity.

**HP2' (was the second gate-mover): reentrant `broadcast_to("memory")` ->
`EpisodicRecall.encode()` sixth slot. KILLED.** Same F1 kill -- the payload is identical to
the one `brain.py:1610` already drains. Two further reasons it should not be rescued:
its terminal is `_recall_conf` -> `elig["recall"]` -> `_drive_for`, the action gate one hop
earlier than CX1; and its falsifier (*"pin the drained value to a constant; dead if
`complete()` returns the same key every tick in both arms"*) cannot distinguish information
transfer from noise injection -- a sixth binding of *any* varying payload perturbs `S`,
shifts the thresholds and shuffles recall, so it would pass identically with
`random.random()`. If it is ever revived it needs a three-arm A/B with a
shape-and-entropy-matched scramble control, and a prior check that `EpisodicRecall.complete()`
has any consumer other than `_recall_conf`.

### Killed as the action gate under another name

**TH2** (BG `_decision_margin` -> `ignition_threshold`) and **NM3** (Cortisol `caution_bias`
-> `ignition_threshold`) and **TH1** (TRN gain -> reentrant percept salience). All three,
plus NM1, resolve to the same node: `elig["reroute"]` in `undirected` at `brain.py:1635`,
one float added equally to both non-HOLD actions (the comment at `:1626-1629` says so:
*"these say 'act', not which act"*). The prior program's F3 treated this as an **attribution**
problem and prescribed one writer per A/B window; sequencing them makes each individually
attributable and leaves all four terminating at the same node.
Two further kills. TH2's cited headroom -- *"`_reentrant_availability` only 2 distinct
today"* -- is a **misdiagnosis**: 2 distinct is `{0.0, 1.0}`, the all-or-none signature of
F2, and no threshold value makes an all-or-none loop graded. NM3's A/B is already run and is
marginal: `metacog_hold` moved on 3 of 100 while the loud `metacog_rel` 93 of 100 is a
telemetry field. TH1 pre-declares its own collapse risk (the factor is 0.5 on 154 of 200).
Also note the lag nobody costed: `_w8_tick` (which calls `_reentrant.step()`) runs **before**
`_c1_tick` (which writes `ignition_threshold` at `:2158`), so every write applies to the
*next* tick's ignition -- and `rasters["access"] = _reentrant_strength` at `:2129` closes
that loop at the same lag. These wires add gain to an existing delayed feedback loop.

**HP3** (familiarity -> the `0.6` perception salience at `autonomous_loop.py:1157`). Killed
by its own mandatory pre-check on data already in hand: `threat` is 0.0 on 80 of 80 and the
GlobalWorkspace broadcast is 1 distinct on 200 of 200. Perception wins at 0.6 against a
competitor pinned at zero; **no value in [0,1] changes the outcome.** The finding is the
degenerate competition, not the wire. (This is the reasoning the rest of the program should
have applied to TH3.)

**CP3** (`CrossModalBinder.recall()` -> `respond()`). 49 of 200, ~24.5%. A wire that launders
a three-in-four-wrong guess into a percept.

**TH4** (BG margin -> TRN `contrast`). TRN is saturated into WTA live (`_attn_gain`
effectively one-hot 1.0/0.5); modulation wiggles the 0.5 floor and cannot move the argmax.

**K-GATE, named once so it is not re-proposed piecewise:** anything into
`_route_percept_to_action`, `elig[...]`, a new `_drive_for` term, or a second STN brake. Three
regions proposed variants of this independently. It is the defect this phase exists to avoid,
and per F3 the bus's only drain is already one of these sites.

### Killed by a closed port or a saturated terminal

**HP1** (recall confidence -> `MetacognitiveMonitor` `novelty`). **Double kill.**
(a) *The port is closed.* `metacognitive_monitor.py:174` stores `nov` and `:176-181` computes
`drive` from `base + w_conf*c + w_lp*lp - w_surprise*s*(1-lp)` -- `nov` appears nowhere in it.
Two independent 0.0-to-1.0 sweeps found `reliability`, `hold`, `learn_gate` bit-identical.
(b) *The terminal is saturated.* Its only weight-changing consumer is `brain.py:2489-2493`:
`gate >= self._lg_base` compares the gate to an EMA of itself and is open on 200 of 200;
`len(_make_tasks(seed=0)) == 3` and `_ll_i` reaches 3 of 3 in every condition tested. Nothing
downstream of `learn_gate()` can change after tick 3 -- and that block is inside
`_enter_sleep`, not the wake tick, which constrains it further still.

**CX2** (`GSM.predict()` `next_hat[1]` -> metacog learning progress). Same saturated
terminal. Falsified before building.

**CX5** (ToM divergence -> `AffectCore.empathize`). `_false_beliefs` is assigned only inside
`respond()` (`:3373`, `:3392`) and decays 0.9x/tick, so it is ~0 on silent ticks. Keep the
*pre-condition test* -- it audits the existing `_social_risk` -> arm wire, which is equally
dead for the same reason.

**K-LC** (LC `gain()`/`mode()` -> anything). `level` = 1.0 on 75 of 80 pins `gain()` at
`GAIN_MAX`. A **prerequisite defect** (rescale LC's phasic accumulation), not a wire. Note
that W0.5 is the same class of repair with a working precedent.

**K-ROSE** (repair `rose_language._build_readout`'s `("continent","phase","class")` triple).
Real defect -- 0 of 84 live triples pass, `nightly_refit` returns `None` -- but
`competence()`/`comprehend()` have zero consumers, so a working refit tops out at a telemetry
field. P1 plus a dead-route finding. Fix the consumer chain first.

**K-EXEMPT** (`RecognizerBank`, `ContrastiveEpisodeAnalyzer.advice()`, `DevelopmentalEnvelope`,
`DriveReductionLearner`, `spike_kg`, `spiking_jepa`, `latent_q`/`mlp_q`). Disposition is
EXEMPT-with-reason. `ContrastiveEpisodeAnalyzer` is the trap: the most conspicuous dead organ,
therefore the most tempting, and every candidate consumer takes it into a baseline-relative
gate that a mean shift cannot move.
**One caveat that must be honoured before these are recorded.** Several are exempted on
static grounds ("no importers", "not constructed live"). `src/snn_data_lake/brain_atlas.py:
37/83/92/109` imports every module in every region via `importlib`, so "no importers" is a
**hypothesis** for anything under `snn_data_lake.<region>` -- this repo has been burned by
exactly that before. Re-derive all seven against a runtime import trace before recording them.

### RC1 -- not killed, demoted to a labelled negative-result experiment

`arc_mint` -> `arc_solver._pool`. arc_mint's own header pre-declares zero held-out solves
today. Build only if the phase wants the number; gate admission on `not rec.memo_flag` so
`rank`/`lookup` forms cannot inflate train-fit. Report the zero. **Also scope-blocked**:
both files are under `reasoning/**`.

---

## 6. Falsifiers: the class defect, and the two-sided rule

Every falsifier proposed across all six surveys had the same shape: *"identical between arms
-> dead."* Not one could fail toward **pathological**. Three pathologies these wires can
produce all read as success:

| pathology | example | why the one-sided falsifier passes |
|---|---|---|
| latch / entropy collapse | TH3 | `_attended` differs from baseline -- it is just frozen |
| period-2 chatter | CX3, TH2/NM3 | `engaged` differs on many ticks -- it is oscillating |
| monotone drift to saturation | NM1 | the terminal moves, then pins |

This is the `:969` failure with the sign flipped: `:969` made a constant vary and changed
nothing; these make a variable *stop* varying, or vary wrongly, and the instrument reports
motion.

**Rule for every wire in this program.** Ship a second, opposite-sided falsifier. For the
consumer's output sequence over 7 seeds, record distinct count, Shannon entropy, longest run,
lag-1 autocorrelation, and pre-declare **both** directions: *identical to baseline = revert*
**and** *entropy falls, or longest run grows, beyond the 7-seed baseline spread = revert*.

**Second rule: neuter the argument, not the producer, for a pinned-argument repair.** The
producer-pin discipline is correct for a *new* wire and wrong for this class, because both
pinned-argument producers already have other live consumers (cortisol drives the live
Q-learning rate at `autonomous_loop.py:1530`; `AffectIntegrator` arousal drives the emotion
prototypes at `brain.py:2033`). Same seed, two runs, one argument swapped. One flag, no
collateral.

**Third rule: no wire may land while the live-wire graph contains an unmeasured cycle.** F3
guards one scalar; the loop this program can close is on different variables. RC2 gives
`action(t) -> ForwardModel residual -> pc_surprise(t+1) -> sal -> submit("percept") ->
winner(t+1)`, and `step()` already latches (`if reentry and bottom_up_supported: _g_reentry =
_SUSTAIN`). External positive feedback into a latch gives a workspace that ignites one
content forever -- and per the table above, that reads as success. After each stage, 1,000
ticks; record the winner sequence's distinct count, entropy and longest run; collapse or
run-length growth past the 7-seed baseline = revert the second wire, not the first.

---

## 7. A rising read count is necessary and NOT sufficient

**Key 1 -- the gate.** A read must appear in `tools/p6_gate.py`'s output. It is a static AST
walk: it proves a read *exists*, never that it *changes the reader's output*. The gate
deliberately does not assert the P6 status line, because asserting "0 faculty reads" would go
green forever by measuring nothing and red the moment someone did the right thing. F1 shows
key 1 is weaker still than "static": two reads can both be counted while returning the same
bytes.

**Key 2 -- the neuter-A/B, aimed one level over, with the action clamped.** Disable the
WRITER faculty and show the READER faculty's **own output** changes. This is POLICIES' own
requirement for a P6 claim. It is not sufficient on its own either, and the reason is F3:
`A -> gate -> action -> next-tick percept -> B` is live on every wake tick, so an unclamped
neuter that changes the selected action will change B's output whether or not a direct edge
exists. **The sufficient evidence is the action-clamped A/B (W0.4): B's output must still
differ when the action sequence is held fixed in both arms.** If it differs only unclamped,
the edge is routed through the action gate and the wire is a spoke, not a conversation.

A wire that turns key 1 and fails key 2 is `engine/autonomous_loop.py:969` again --
`_wm_spikes` 1 distinct to 22 distinct, and held goal / selected action / arm act-hold /
trajectory each changed 0 of 200. It is reverted, and the faculty is EXEMPTed with a reason,
exactly as under P3.

**The companion figure, and where it belongs.** *Pinned arguments remaining: 2 today -> 0
after the P1 pass* (`neuromod` 1.0 on 120 of 120; `arousal` 0.0 on 61 of 61) plus *2 closed
ports* (metacog `novelty`; `broadcast_to("language")` never drained). It is harder to satisfy
vacuously than a read count -- but it is still gameable in exactly the `:969` way, and both
of the original three sat on action-gate chains, so unpinning them produces varying values
feeding the same gate. **It is reported under P1, with no P6 claim attached.**

**The honest success criterion for this phase is not a number in any table.** It is
*N wires shipped with a passed two-key falsifier, and M faculties EXEMPTed with a measured
reason* -- and a phase that ends with more exemptions than wires has succeeded.

---

## 8. Blocked on a genuine input-free interval (`autonomous_loop.py:1001`)

**No wire in Waves 1-3 depends on this**, which is why the program can proceed.

The structural reason, quoted from `autonomous_loop.py:986-995`: *"a working-memory attractor
exists to hold information across a GAP IN INPUT, and this loop has no gap"* -- `update()`/
`plan()` re-deliberate and re-commit the primary goal on **every** wake tick, and
`_gate_primary` resets that frame's drive to 1.0 whenever the margin clears, *"which live it
always does."* The only ways to make the pop fire are to lower `recurrence`/`hold_threshold`
or stop re-committing: retuning the module's own constants until a number moves, which P1
forbids.

Blocked:

1. `PrefrontalExecutive.maintain()` -- built, measured, **reverted** (0 of 200 on every
   outcome; `_wm_spikes` 1 distinct -> 22 distinct).
2. `PrefrontalExecutive.push_subgoal()` -- blocked harder: a sub-goal is a task
   *decomposition* and this loop is flat and reactive, so there is no honest trigger at all.
3. `top_down_bias()`'s multi-frame summation, `max_goals`, the HELD-parent pop-back, and
   `_wm_spikes` -- all dead live for the same one reason.
4. `cerebellum/spiking_jepa` -> `PrefrontalExecutive.plan` latent rollout -- blocked twice:
   already EXEMPT with a measured reason (`encode()` does not vary across 24 distinct live
   width patterns), and a rollout needs a deliberation phase the tick does not have.
5. `wavefront_planner.collides()` -> `PrefrontalExecutive.plan()` -- same shape via
   `deliberate_ahead`, whose only callers are `live_brain.main()`.
6. **Autonomous, non-chat-triggered imagination.** CP1 is *not* blocked here because
   `imagine()` runs on a chat turn -- but that is exactly why CP1 fails P6 (a). The moment
   anyone proposes running imagination on the wake tick to fix (a), it lands in this list: an
   imagination faculty that runs while input keeps arriving is just another percept transform.

**Do not conflate CX5 with this family.** ToM's `_tom_protect` is written only inside
`respond()` and decays 0.9x/tick, so it is ~0 on silent ticks. That is *"no conversational
traffic on the wake tick"* -- the **opposite** precondition, needing traffic rather than
quiet. Bundling them would hide that the existing `_social_risk` -> arm wire is equally dead.

**Constructive half, with the false candidate ruled out first.** Two quiescence-shaped
mechanisms exist and **neither is an input-free interval**:

- `autonomous_loop.py:1577` -- `if not grew: # quiescent (familiar) tick` is a *novelty*
  predicate behind `reward_fn is not None`. Input still arrives, the percept is still
  delivered, `_gate_primary` still re-commits. Calling this the precondition would be a false
  equivalence.
- `sleep/sleep_consolidation.SleepConsolidator.quiescent()` (`:182`) -- the right
  **shape** (a phase, not a tick predicate), and `reasoning/lake_skill_library.py:251` records
  outright that *"quiescent() is never called."* Its only callers are its own demo and the
  offline adapter `hippocampal_memory.py:133`.

So the precondition may be a wiring job on an existing mechanism rather than new
infrastructure -- a Wave-4 design item, deliberately out of scope for a measure-and-design
phase, and it needs its own falsifier before anyone builds on it: **if a quiescent interval
is introduced and `maintain()` still changes 0 of 200 outcomes, the WM attractor is EXEMPT
permanently, not blocked.**

---

## 9. Wave 4 -- what actually de-fragments the bus, and in which order

Two candidate infrastructure changes were proposed. They are **not** interchangeable and the
order matters.

1. **Make the drain content-sensitive** (fix F1): give `_deliver` a per-subscriber term, or
   give the one drain access to *which* coalition ignited rather than `max(rb.values())`.
2. **Share one workspace instance** across the 7 orphaned writers (fix the 11-instance
   census).

**(1) strictly precedes (2), and the prior reconciliation had it backwards.** Sharing one
instance among faculties whose `broadcast_to` returns an identical content-blind payload,
drained at one site that collapses it to a float, gains nothing: eleven private buses is a
smaller defect than one bus whose single reader throws away which faculty spoke. Do (1),
re-measure, then do (2).

A constraint on (1): the per-subscriber axis v4.78 deliberately removed was removed **because
it was never learned**. Re-adding it to make two wires look alive would be re-introducing a
known defect to satisfy a metric. Any fix must derive the per-subscriber distinction from
something that exists -- the subscriber's own declared interest, or the winner's source --
not from a new learnable weight nobody trains.

Also in Wave 4, and cheap: `reentrant_workspace.py:399-409` ships `_Specialist.emit()` --
`bias = ws.broadcast_to(self.name)` added to its next output -- a working example of one
faculty consuming another's broadcast. It lives only inside the bus's own self-check. The
pattern P6 asks for is already written, already passing, and used by zero faculties.

---

## 10. Build order

**W0.1 resolver (dynamic, fixture-validated) -> W0.2 bound the inbox + fix the drained test
-> W0.3 crc32 jitter + order-invariance check -> W0.4 action-clamped charge_probe -> W0.5
broaden the threat/salience drive -> W0.6 `.venv-cu128` re-census incl. the ACCESS path ->
W0.7 connectome-diff half -> [re-baseline] -> W1.1 NM2 -> P1 pass (NM1 with `_bound`
instrumented and 5,000 ticks; CX3 with the margin and the chatter test) -> W2.1 RC2 (i) ->
W2.1 RC2 (ii) -> W2.3 CP1 (Imagination ledger, prototype readout fixed first) -> W3.3 CX1
(move `recognize()` out of `state()` first; abandon if that alone changes nothing) -> W3.1
CP2 -> W3.2 CX4 (gated on the shift pre-check) -> [Wave 4: fix the drain, then share the
instance].**

W0.3 must precede W0.6. W0.4 must precede every wire. Each wire is one change, one two-sided
A/B, one verdict. Never two wires into one consumer in one window; never a wire whose
falsifier has already fired; never a second wire while the live-wire graph has an unmeasured
cycle.

**Per-tick cost: no objection found, and it is stated rather than manufactured.**
`ReticularGate.mask()` already runs 640 LIF steps every tick; `LearnedEVC._population_gain`
already runs 2,400 inner iterations every tick -- both candidate changes are to an
*argument*, adding no call. `ReentrantWorkspace._simulate` gains no coalition from any
surviving wire. CP1's 2312-d forward pass is confined off-tick behind the chat keyword test.
The real cost risks are the unbounded inboxes (W0.2) and the 5,000-tick runs the drift and
loop checks now require.

---

## 11. Critique dispositions -- what was rejected, and why

Applied in full: the Claustrum-revert kill of TH3; the content-blind-drain kill of TH3 and
HP2'; the binary-availability finding; the terminal census replacing the read-count table;
the reclassification of NM1 and CX3 to P1; CP1's P6(a) failure; the `perc[:6]` false-negative;
the unbounded inbox and the gate's drained miscount; the order-dependent jitter; the
producer-vs-argument neuter correction; the read-before-write ordering on CX3; the `_bound`
boolean; the `ignition_threshold` one-tick lag; the sign-asymmetric-rectifier and `abs()`
ratchet on NM1; the two-sided falsifier class fix; the action-clamped A/B; the scramble
control; the connectome-diff half; the ACCESS path in the F4 census; the `reasoning/**` scope
block on RC3 and RC1; the runtime-import re-derivation of the K-EXEMPT list; the LC-precedent
salience broadening.

Rejected, with reasons:

1. **"`brain.py:1516` sits inside `_tick_ensemble`."** REJECTED. AST resolution puts it inside
   `_route_percept_to_action` (`:1398`). The correction strengthens the underlying finding
   rather than weakening it -- CX1's terminal is an action-gate term, not merely a tick-lagged
   memory cue -- so the disposition (re-scope CX1) is unchanged and the stated reason is
   corrected.
2. **"The Wave-0 headline must read 0 of 9 unchanged after the resolver fix."** REJECTED. If
   the resolver mis-files real cross-module reads, the correct outcome of fixing it is that
   the number moves. Requiring it to stay put pressures the fix toward preserving the prior --
   Goodharting the instrument, in the paragraph that coins the phrase. Replaced with a
   fixture plus a published hand audit.
3. **"`AffectIntegrator` arousal is consumed by nobody today."** REJECTED (a claim of the
   original program, not a critique). `brain.py:2031-2035` consumes it into the emotion
   prototypes. NM2 survives -- the incumbent is still pinned and the consumer is still a
   different faculty -- but its A/B method changes.
4. **"Broadening the threat drive discharges NM1 and HP3 for free."** REJECTED as stated. It
   is an untested prediction. W0.5 is kept; the "for free" claim is not, and the censuses are
   re-run before anything is discharged.
5. **"Promote instance-sharing to Wave 1."** REJECTED in that order. Sharing an instance on a
   bus whose payload is content-blind and whose one drain is `max()` connects nothing. The
   drain fix precedes it (section 9).
6. **"Pinned arguments 3 -> 0 is a useful companion figure for P6."** PARTIALLY REJECTED. The
   figure is kept because it is harder to game than a read count, but it moves to the P1
   ledger with no P6 claim, and it reads 2 -> 0 after CX3 is reclassified.
7. **"Wave 1 is the cheapest wires that make an existing bus consumer read something."**
   REJECTED as a frame. Six regions produced 21 proposals and exactly one was a bus read; F1
   then killed it. Wave 1 contains one wire and it is not a bus read. That is the finding, not
   a shortfall: the bus is not where this brain's information lives.

---

## CORRECTION 2026-08-03 -- the payload fix is NECESSARY AND NOT SUFFICIENT, measured

Everything above stands as written, and its FIRST RECOMMENDATION is now known to be insufficient on
its own. This section was added after the fact by the coordinator; do not read the wave ordering
above without it.

The plan concluded: fix the content-blind payload (F1) first, then share the 11 private instances.
An invention workflow then proposed a ledger carrying speaker identity through that payload, and
prescribed a cheap Step 0 before writing any code -- census the per-tick (speaker x listener)
activity and report the rank of the coincidence matrix. Its author predicted the census would come
back RED and cancel their own design. It did.

MEASURED, 300 brain ticks, instrumenting `submit()` and `step()` per tick:

    ticks observed                        : 146
    distinct (speakers, listeners) per tick: 1 of 146
    speakers = ('percept', 'salience', 'self')   listeners = ('action', 'language')   x146

**Rank 1.** `engine/brain.py:1741-1757` submits all three sources unconditionally inside one `try`
block every tick, and the two subscribers are registered once at `:727`. The speaker set and the
listener set are CONSTANT. An identity channel added to the payload would therefore carry a constant,
and a receiver that learned to weight speakers would learn one weight vector and never revise it.

**This is a deeper reason for `faculty-to-faculty 0 of 9` than the one this document gives.** The
payload discarding the speaker is real and worth fixing. But even a shared bus with a
speaker-addressed payload connects nothing while every faculty speaks on every tick: there is no
differential activity to learn from, nothing to attribute, and nothing to vote about.

**What that changes about the wave ordering.** The prerequisite is not payload-then-share. It is:

  W-1. DIFFERENTIATION. Some faculty must sometimes NOT speak, or speak differently, for reasons that
       depend on its own state. Until that exists, W1 and W2 are building an addressing scheme for a
       message everyone always sends. This is measurable with the census above and it currently
       reads 1 of 146.
  W1/W2 (payload, then instance sharing) stay in this order, behind W-1.

The falsifier for the whole program stands unchanged and is now joined by a cheaper one: if the
census still reports 1 of 146 distinct speaker sets after a wave lands, that wave did not create
differentiation, whatever else it did.

A first probe that killed a design in minutes, before a line was written, is the cheapest result in
this document. It is recorded here because the temptation is to fix the payload -- it is a real
defect, it is satisfying to fix, and on its own it would have changed nothing measurable.

---

## CORRECTION 2026-08-03 (second) -- the differentiation blocker was MISREAD, and the wave order is backwards

The CORRECTION above concluded from `distinct (speakers, listeners) per tick : 1 of 146` that no
faculty-to-faculty selection was happening, and re-ordered the waves to put **W-1 DIFFERENTIATION**
ahead of the payload fix. **The census number is real and reproduces. The conclusion drawn from it
is wrong, and the re-ordering it motivated is backwards.**

**What the census actually counts.** It counts who is PRESENT in the submission buffer each tick,
not who WINS. Presence is constant because the same three sources submit every tick -- that is the
design, not a defect. Selection happens downstream of submission, and it varies.

**Measured, seed 0, 300 ticks, brain built via `make_brain(seed=0, persist=False)`:**

```
distinct argmax keys over broadcasts  : 2        (self 105, ctx 39)
```

So the winning faculty identity DOES vary tick to tick. Differentiation is not missing. Reading
constant presence as absent selection is the presence-is-not-participation error run in mirror
image -- the usual form mistakes presence for participation; this one mistook constant presence for
absent participation. Both are the same failure to distinguish the buffer from the outcome.

**The real blocker is one line, and it is a CONSUMER defect.** `engine/brain.py:1623`:

```python
rb = self._reentrant.broadcast_to("action")
self._reroute_bias = round(float(max(rb.values())), 4) if rb else 0.0
```

`max(rb.values())` takes the MAGNITUDE and discards the KEY. The key is the identity of the winning
faculty. The comment two lines above claims "a globally-available coalition CHANGES behaviour"; what
crosses the wire is a scalar that could have come from anywhere.

**The decisive A/B -- permute the keys, preserve the values.** Same multiset of magnitudes, winner
identity maximally scrambled. If identity reached anything, this must move something:

```
                          ARM A (intact)      ARM B (keys permuted)
reroute_bias              0.16                0.16
full Brain.state() sha    d7a6d81fc43d83f9    d7a6d81fc43d83f9
```

Byte-identical. The winner's identity provably carries ZERO information downstream. It is already
being handed to the reader; the reader throws it away.

**Consequence for the plan.** W-1 DIFFERENTIATION is deprioritized, not promoted. Manufacturing more
differentiation upstream of a consumer that discards identity cannot move any measured outcome -- the
brain already produces variation this wire refuses to carry. **Fix the consumer first: make the
reader take the key, not just the value.** Only after identity survives the wire does adding
differentiation upstream have anywhere to land.

**Standing falsifier, unchanged in spirit and corrected in target.** The old one asked whether the
census still reports 1 of N distinct speaker sets. That question is now known to be the wrong one.
The replacement: **permute the keys of `broadcast_to()`'s return while preserving its values; if
`Brain.state()` is still byte-identical, the wire still carries no identity and the wave did not
land, whatever else it changed.**
