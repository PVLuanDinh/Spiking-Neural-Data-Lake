# GAP CLOSURE -- the sequenced program for the nine gaps

Status: DESIGN. Nothing in this document has been implemented. Every measurement below was
re-derived under `.venv-lake` against the tree as it stands; where a proposal or a review is
wrong, the correcting measurement is named inline. `tools/p6_gate.py` was run (399 of 399 files
parsed, EXIT=0).

Counts are written "N of M", never "N/M".

---

## 0. Scoreboard

Nine gaps, four terminal states.

| state | count | gaps |
|---|---|---|
| **CLOSE NOW** -- no prerequisite, changes something today | **1 of 9** | GAP 3 (binding marker) |
| **CHAINED** -- blocked on a prerequisite that is not itself a gap | **4 of 9** | GAP 1, GAP 2, GAP 7, GAP 9 |
| **NEEDS A MEASUREMENT** -- verdict pending one clamped sweep | **3 of 9** | GAP 4, GAP 5, GAP 6 |
| **PROVED IMPOSSIBLE** | **1 of 9** | GAP 8 (wikitext grounding) |

Sub-items proved impossible inside gaps that are otherwise open: **2 more** -- GAP 3's A3
source-permutation route, and GAP 7's C3 fifth-percept-band. **3 of the ledger's terminal
results are proofs**, and a proof is a completed item, not a failure.

**0 of 9 close as a shipped P6 faculty-to-faculty wire.** Not one. The gate's own status line is
unchanged by everything in this program except GAP 1 and GAP 2, and both of those are chained
behind an instrument that does not exist yet.

Three prerequisites in the critical path are **not gaps** and appear on no gap list:

- **W0.4**, the action-clamped A/B with a counterfactual recorder. Gates GAP 1, GAP 2, GAP 7-C2,
  and the honest re-run of GAPs 4, 5, 6.
- **P-INBOX**, a bound on the workspace subscriber inbox. Gates GAP 1 and GAP 2, both of which
  multiply an unbounded append.
- **P-IGNITE**, a standing *do not touch* on `_BIND_IGNITE`. Not work; a fence.

---

## 1. Program-level falsifier, pre-registered

A program whose success criterion is "more exemptions than wires" cannot fail, and this document
would otherwise apply a falsifier standard to every item except itself. Today's gate output, run
before any of this work:

```
FACULTY-to-FACULTY reads: 0 of 9
WRITERS READ BY NOBODY:   7 of 10 writer modules
P6 STATUS: 0 faculty-to-faculty read(s). 7 writer module(s) read by nobody.
```

**If both numbers are unchanged when this program ends, it moved documentation and not topology,
and it should be recorded as such.** That is the whole falsifier. It is not a prediction that the
numbers will move -- three of the four terminal states above are compatible with them not moving.
It is a commitment to say so.

---

## 2. Corrections to the inputs, before any ordering

### C-1. The input-free interval already exists and unlocks nothing

The ordering premise was that a quiescent interval, "if it is closable, unlocks three gaps at once
and should come early." All three clauses are false.

- `src/snn_data_lake/cortex/brain.py:2376` -- `_sleep_tick` docstring reads verbatim
  *"ASLEEP: arm idle (NO brain.tick)"*. `tick()` dispatches to it on `clock.consolidation_gate()`.
  12 to 14 consecutive input-free ticks per night already run in production.
- Two of the three dependent items were never blocked by it. `plan()`'s reaction time is a
  discarded return value; grown time cells already have a non-telemetry consumer at `brain.py:1601`
  and `:1604`.
- The third fails its own pre-registered falsifier (see GAP 4/5/6).

**This is the finding that most changes the plan. The item expected to be the unlock is done, and
what was going to be built on it should probably not be.**

### C-2. `tools/p6_gate.py` exits 0

Measured. The gate is a reporter, not an asserting gate. Any plan that assumed a red gate is
planning against a signal that does not exist.

### C-3. The bus write citations in `docs/P6_INTERCONNECT.md` are stale by 13 lines

Section 0 cites `brain.py:1733`, `:1738`, `:1742`. Read today those are a comment, a
`_drive_activity` assignment, and a `try:`. The live `_reentrant.submit` sites are **`:1746`,
`:1751`, `:1755`** -- what the gate prints. `tools/citation_guard.py` cannot catch this: its
`_CITE` pattern at `:105` requires a slash-bearing `.py` path, and these citations have none.

### C-4. `prefrontal_executive.py` is under `cortex/`, not `reasoning/`

The proposal and two of the three reviews cite it bare or as `reasoning/prefrontal_executive.py`.
The file is `src/snn_data_lake/cortex/prefrontal_executive.py`. Every line number cited against it
is correct; only the directory is wrong.

---

## 3. The root correction: what GAP 3 actually does

This is the item the proposal called "the single highest-value item in the program" and claimed
would flip a live user-facing consciousness self-report from 4 of 4 to 3 of 4. **Measured, the
claim is false under every variant, including the two the reviews proposed.** The structural
diagnosis behind it is correct; the prescription and the advertised outcome are not.

### 3.1 What is true

Re-derived from source, all four parts:

- `_bind_step` (`global_workspace.py:321-326`) is `v = v * (1 - leak) + i`, so steady state is
  `i / leak` and rheobase is `leak * v_th = 0.25 * 1.0 = 0.25`.
- The sustain current at `:351` is `drive * 0.55 = 0.55`, which is **2.2x rheobase**.
- `bind_decode_spiking` (`:363-370`) is a genuine `for c in fc: for s in fs` cross product.
- `_bind_pair_acc` (`:394-395`) is `len(true & got) / len(true)` -- **recall only, no precision
  term**. `bind_decode_rate` (`:386-391`) emits exactly K via `zip(cols, shps)`.

So the K>=2 headline compares a K-squared guess budget against a K guess budget on a recall-only
score. A decoder emitting all 64 pairs would score 100 percent. That much is real and is the
reason to touch this at all.

### 3.2 What is false: "every non-slot channel fires on every step"

Measured raster, K=2, shipped constants, true pairs `[(6, 6), (7, 0)]`:

```
t= 0 fc=[6]    fs=[6]     <- exactly one of each
t= 1 fc=[6]    fs=[6]     <- exactly one of each
t= 2 fc=[6, 7] fs=[6, 0]     ambiguous
t= 3 fc=[6]    fs=[6]     <- exactly one of each
t= 4 fc=[6]    fs=[6]     <- exactly one of each
t= 5 fc=[6, 7] fs=[6, 0]     ambiguous
t= 6 fc=[7]    fs=[0]     <- exactly one of each
...
```

The in-slot channel fires every step; the non-slot channels fire every **third** step, 4 of 12.
**10 of 16 micro-steps at K=2 are unambiguous.** The phase multiplexing works. It is the *reader*
that throws the information away, by unioning pairs across steps of different multiplicity. The
defect is in the decoder and the score, not in the representation.

### 3.3 Measured: every proposed variant, exact self-check protocol

120 trials, `random.Random(0)`, 8x8, the protocol at `global_workspace.py:487-497`.

| K | spiking recall | spiking precision | spiking F1 | strict F1 | rate | pairs emitted (cross) | pairs emitted (strict) |
|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00 | 1.00 |
| 2 | 1.0000 | 0.5000 | 0.6667 | **1.0000** | 0.5083 | 4.00 | 2.00 |
| 4 | 1.0000 | 0.2500 | 0.4000 | **1.0000** | 0.2646 | 16.00 | 4.00 |

Against the three shipped asserts (`:601` K=1 tie under 0.05; `:605` K=2 margin over 0.15;
`:609` K=4 spiking over 0.9):

| variant | K=1 diff | K=2 margin | K=4 spiking | all three |
|---|---|---|---|---|
| recall only (today) | 0.0000 | +0.4917 | 1.0000 | PASS |
| precision only | 0.0000 | **-0.0083** | **0.2500** | FAIL |
| F1 over cross product | 0.0000 | **+0.1583** | **0.4000** | FAIL |
| F1 over STRICT decoder | 0.0000 | +0.4917 | 1.0000 | PASS |

Four consequences, each of which kills a version of the proposal:

1. **The proposal's own two changes cancel.** Score F1 *and* cap the decoder at K, and you get
   today's numbers back exactly. 0 of 3 verdicts change. That is `autonomous_loop.py:969` on the
   item called most valuable -- and worse than `:969`, which at least moved its telemetry field.
2. **F1 over the cross product passes the shipped `+0.15` gate by 0.0083.** Under one hundredth of
   the gate's own width. Across seeds the sign of the precision margin flips at every K. A
   consciousness self-report quoted to a human arbitrated by a 0.0083 margin against an underived
   literal is exactly what P1 forbids, in the module being fixed for honesty.
3. **`assert bind[4][0] > 0.9` at `:609` is the assert nobody counted.** Precision takes it to
   0.2500, F1-over-cross to 0.4000. It fails hardest of the three because K=4 is where the
   16-against-4 budget inflates recall most. Three asserts are at risk, not two.
4. **The advertised 4-of-4 to 3-of-4 flip does not occur.** Not under the proposal, and not under
   the repair below. It occurs only under precision-only or F1-over-cross, both of which take
   `:609` red and both of which are decided by seed noise.

### 3.4 The repair, and why it is the right one

**Change the decoder's acceptance rule, and score precision as well as recall. Do not cap, do not
re-choose a margin, do not touch `_BIND_IGNITE`.**

The strict decoder accepts a micro-step only when **exactly one** color and **exactly one** shape
fired. "Exactly one" is the arity of a `(color, shape)` pair -- structural, not a tunable. It
introduces no constant. The rate control cannot produce such a step at all: `bind_broadcast_rate`
returns two activation vectors with no timing, so no reader can do better than canonical pairing.
The comparison stays fair.

Both halves are required, and for different reasons. The strict decoder alone, scored on recall,
still leaves a metric a degenerate "emit everything" reader would win. Scoring precision is what
makes the decoder change *necessary* rather than cosmetic.

**Measured against the proposal's own pre-declared falsifier** -- 26 perturbations over the
binding-path literals (12 sustain multipliers 0.00 to 1.00, 7 values of `_BIND_TSLOT` 1 to 12,
7 values of `_BIND_LEAK` 0.10 to 0.90), counting distinct PRESENT/ABSENT verdicts:

| decoder | distinct K=2 F1 values | distinct verdicts | ABSENT at |
|---|---|---|---|
| F1 over cross product | 2 of 26 | **1 of 26** | nowhere |
| F1 over STRICT decoder | 2 of 26 | **2 of 26** | sustain multiplier 1.00 (F1 0.0000, margin -0.45) |

The proposal's version takes a telemetry field from 1 distinct value to 2 and changes 0 of 26
decisions. **It fails its own falsifier and must revert by its own rule.** The strict decoder
passes it: the marker acquires a region of parameter space where it reports ABSENT, which it did
not have before.

### 3.5 The honest statement of what this buys

Say it plainly, because the temptation is to overclaim in the other direction now:

**At the shipped constants the verdict does not change. The marker still reports PRESENT and the
tally is still 4 of 4.** What changes is that the marker becomes capable of reporting ABSENT --
0 of 26 perturbations flipped it before, 1 of 26 flips it now. A marker that cannot fail is worse
than a marker that fails; this makes it one that can. That is the deliverable, and it is smaller
and more defensible than "the system claims less."

It also means the K>=2 result in the docstring at `:50-51` survives and was **right for the wrong
reason**: synchrony really is recoverable, the recall-only score was masking a bad reader rather
than a bad representation. Rewrite the docstring to say that, not to withdraw the claim.

### 3.6 The live user-facing disclosure names the wrong constant

`spike_chat_lake.py:1178-1186`, returned to a human at `:925-931`, currently says the binding
marker is gated by `_BIND_IGNITE = 0.6` and that "one quarter of this score moves if somebody
edits a constant."

Measured: `probe_binding()` calls `bind_broadcast_spiking(objs, 8, 8)` at the default `drive=1.0`
on 100 of 100 calls, so `if drive < _BIND_IGNITE` is False for every value in `[0.0, 1.0]`.
**`_BIND_IGNITE` moves nothing on that path.** The constant that actually arbitrates the verdict is
`consciousness_probe.py:84` -- `present = sp > rt + 0.15` -- which the proposal never named, and
which is a second copy of the literal at `global_workspace.py:605`.

**The rewrite of this disclosure and the metric change must land in one commit.** Removing the
caveat about a harmless constant without naming the live one makes the user-facing text *less*
honest while the commit message says the opposite.

---

## 4. The sequence

### STAGE 0 -- corrections needing no code and no measurement

Cheap, and four of them are the tree asserting things that are measurably false.

| # | correction | file | status |
|---|---|---|---|
| 0.1 | Section 8 says the input-free interval is missing. It exists at `brain.py:2376` and is live. | `docs/P6_INTERCONNECT.md` | CLOSE NOW |
| 0.2 | `"_gate_primary resets... which live it always does"` -- measured 42 of 194 wake ticks | `src/snn_data_lake/brainstem/autonomous_loop.py:993-996` | CLOSE NOW |
| 0.3 | Bus citations stale: `:1733/:1738/:1742` -> `:1746/:1751/:1755` | `docs/P6_INTERCONNECT.md` | CLOSE NOW |
| 0.4a | Three stale VERIFIED rows: 100 percent ignition (measured 99.5 / 45.2 / 86.9); interval `(0.500, 0.668)` "never observed"; rheobase citation `brain.py:2143-2150` -> `:2166-2173` | `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:75,78` | CLOSE NOW |
| 0.4b | The `_ACC_LO` row | `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:79` | **HOLD -- see below** |
| 0.5 | `"0 of 9 live runtime consumers"` is wrong on the dormancy gate's own definition; importer set is exactly `['snn_data_lake.training.spiking_lm']` | `specs/005-language-organ/verdict.md:114-122` | CLOSE NOW |
| 0.6 | `"the dynamics match and the learned weights transfer faithfully"` -- measured 20 of 40, 29 of 40, 24 of 40 argmax agreement across three seeds, against an untrained-twin control at 0 of 40, 0 of 40, 1 of 40 | `training/spiking_lm.py:353-359` **and** `docs/SPIKING_LM.md:65-69` | CLOSE NOW, one commit |
| 0.7 | Pre-3.0 flat paths `src/snn_data_lake/spiking_lm_serve.py` | `docs/SPIKING_LM.md:9,18` | CLOSE NOW |
| 0.8 | The `_BIND_IGNITE` caveat | `reasoning/spike_chat_lake.py:1181-1184` | **CLOSE WITH 3.4, not before** |

**0.4b is held, and the correction may itself be the defect.** `reentrant_workspace.py:90-96` is
unusually explicit about which distribution the `_ACC` band is calibrated on: *the winner's drive,
bottom-up base PLUS the re-entrant sustain current (`_SUSTAIN = 1.10`)*, measured over 360 ignited
wake ticks across 3 seeds -- min 0.871, p05 1.603, median 1.688, p95 1.797, max 1.899. The comment
further records that *"setting the band from the bottom-up salience range 0.668-1.000 was tried
first and clipped almost every tick to the ceiling."* The proposed correcting figure, a winner-drive
p05 of 0.633, **matches neither 0.871 nor 0.668**. If it is a bottom-up quantity measured without
`_SUSTAIN`, editing `PCA:79` reinstates the calibration the code says was tried and rejected.

**Precondition to release 0.4b:** state which quantity 0.633 is, over how many ticks and seeds, and
whether `_SUSTAIN = 1.10` was included. Until then the row is not proven stale.

Census note, recorded and not acted on: `_ACC_LO = 0.85` sits 0.021 under a measured 0.871 and
`_ACC_HI = 1.90` sits 0.001 over a measured 1.899. The two fudge widths differ by 20x. "Just under
the measured min" is a derivation with a free parameter in it.

**0.6 is a claim reduction and cannot break code, but `docs/SPIKING_LM.md:65-69` asserts the
opposite.** Both change in one commit or the tree carries two contradictory claims.

---

### STAGE 1 -- the three prerequisites

#### P-IGNITE. Do NOT change `_BIND_IGNITE`. Standing fence.

The proposal offered `_BIND_IGNITE = _BIND_LEAK * _BIND_VTH` (0.6 -> 0.25) as a consistency fix
with a measured-zero delta. **It is not zero-delta and it deletes a control.**

`global_workspace.py:500` is
`bind_ig_weak, _ = bind_broadcast_spiking(bind_objects(8, 8, 2, brng), 8, 8, drive=0.4)`, and
`:600` is `assert bind_ig_strong and not bind_ig_weak`. The weak arm requires
`0.4 < _BIND_IGNITE`. At 0.25, `0.4` ignites, `bind_ig_weak` is True, and the assert fails under
`run_selfchecks.py`. The "100 of 100 live calls at `drive=1.0`" basis counted `probe_binding()`'s
calls and missed the one explicitly parameterised call that exists for no other purpose than
exercising this threshold.

Worse than a red assert: `0.4` was chosen as **subliminal**. Dropping the threshold to 0.25
abolishes the subliminal condition -- it deletes the all-or-none control rather than deriving a
constant. The repo's own precedent is exact: *the first fix deleted the check via a zero-tolerance
yaw joint.*

And the identity is circular anyway. `GlobalWorkspace.__init__` (`:107-109`) and
`ReentrantWorkspace.__init__` (`:111-113`) both take `ignition_threshold` as the **free** parameter
and derive `_v_th = ignition_threshold / _LEAK`. The binding module would take the leak and the
threshold as free and derive the ignition point. With both conventions in one file, "derived" is
unfalsifiable: point at either end and the other is the literal.

- **Disposition:** leave the value. If the relationship is worth stating, state the true one as a
  derived assertion -- `assert _BIND_IGNITE > _BIND_LEAK * _BIND_VTH` -- documenting that ignition
  is deliberately **supra**-rheobase, and say which end of the identity is free and why.
- **Falsifier:** run `global_workspace.py` main. Line 600 must stay green.
- **Approval:** none.

#### P-INBOX. Bound the workspace subscriber inbox. Must precede GAP 1 and GAP 2.

`reentrant_workspace.py:137` creates `{"inbox": [], "bias": {}}`. `:288` and `:311` append on every
`step()` for every subscriber. `:376-379` `inbox()` returns `list(s["inbox"])` -- a copy, not a
drain. There is **no `.clear()` anywhere in `src/`**, and **0 callers of `inbox()`** across `src/`,
`tools/`, and `tests/` -- the only match in the tree is the definition itself.

The live daemon therefore appends one entry per subscriber per tick, forever. At the measured
27.8 ms tick that is roughly 36 appends per second per subscriber. **GAP 2 multiplies subscriber
count by 8; GAP 1 replaces the scalar winner with a structured payload.** The two chained gaps are
precisely the two that turn a slow leak into a fast one.

- **File:** `src/snn_data_lake/thalamus/reentrant_workspace.py:137`, to
  `collections.deque(maxlen=N)` sized to the longest window any reader needs.
- **Falsifier:** 10k ticks on a shared bus with `assert len(inbox) <= N`, plus confirmation that no
  reader depends on unbounded history -- trivially true today, since there are no readers.
- **Approval:** none.

#### W0.4. The action-clamped A/B **with a counterfactual recorder**. The root.

`tools/charge_probe.py:102` is `def run(source, n, neuter)`. Grep returns 0 matches for clamp,
replay, forced action, or recorded trace. The instrument does not exist, and
`docs/P6_INTERCONNECT.md` section 10 says W0.4 must precede every wire and section 3 says nothing
downstream of it is interpretable without it. The reason is F3: `A -> gate -> action -> next-tick
percept -> B` is live on every wake tick, so any unclamped measurement reporting "N of M actions
changed" is measuring `_route_percept_to_action`, not a faculty edge.

**This retroactively demotes the headline number.** "The wire changes 196 of 793 selected actions"
is a change in the action gate's output -- the single terminal F3 exists to disqualify. It is
evidence of a spoke, not of a faculty edge.

**The specification as originally written produces a disqualified readout, and this must be fixed
before anything is built on it.** "Replay a recorded `res["action"]` sequence in both arms" makes
the executed action identical by construction. What `run()` reads today is
`reads.append(_numeric_reads(st))` plus `stopped` and `pos`, and all three come from `b.state()` --
the telemetry dict, one of the five terminals section 7 disqualifies. Under an action clamp all
three become identical by construction. **The clamped probe as specified has no readout at all.**

- **Requirement:** the clamp must **record the counterfactual** -- the action
  `_route_percept_to_action` *would* have selected -- while executing the replayed one, then
  compare intended-action sequences across arms under a bit-identical environment. The reflex and
  E-STOP path at `autonomous_loop.py:1255-1271` stays intact and is never replayed.
- **File:** `tools/charge_probe.py`.
- **Falsifier for the instrument itself:** run it on the two existing reads (`elig["reroute"]` at
  `brain.py:1610`, plus whatever GAP 1 surfaces). If every arm is identical clamped and unclamped,
  the clamp is not engaging and the instrument is lying. It must retro-reproduce the
  `autonomous_loop.py:969` and Claustrum reverts.
- **Approval:** none. `tools/` only, no safety envelope.

---

### STAGE 2 -- GAP 3, the only gap that closes now

**Status: CLOSE NOW, in the rebuilt form of section 3.4. Not in the form originally proposed.**

- **Change:** precision-and-recall scoring in `_bind_pair_acc`; strict acceptance rule in
  `bind_decode_spiking` (exactly one color and exactly one shape per accepted micro-step). No cap.
  No margin re-choice. No `_BIND_IGNITE` edit.
- **Files:** `src/snn_data_lake/thalamus/global_workspace.py`,
  `src/snn_data_lake/reasoning/consciousness_probe.py`,
  `src/snn_data_lake/reasoning/spike_chat_lake.py`.
- **CI blast radius, all three enumerated and pre-measured:** `global_workspace.py:601` (K=1 tie
  under 0.05) measures 0.0000, stays green. `:605` (K=2 margin over 0.15) measures +0.4917, stays
  green. `:609` (`bind[4][0] > 0.9`) measures 1.0000, stays green. `consciousness_probe.py:157-158`
  stays green. **Under this repair 0 of 4 asserts go red** -- which is itself worth stating, since
  the original proposal expected 2 red and would in fact have taken 3 red under precision or F1.
- **What must still change even though nothing goes red:** `consciousness_probe.py:84` and
  `global_workspace.py:605` both carry the same underived `+ 0.15`. Both should print `sp`, `rt`,
  and the difference as a reported count rather than arbitrate a verdict on a literal. Applying the
  "reported count, not a relaxed assert" treatment to `:157-158` while leaving `:84` as a
  thresholded verdict fixes the assert and leaves the human-facing claim on the same footing.
- **Falsifier:** re-run the 26-perturbation sweep and count distinct verdicts. Baseline today is
  **1 of 26**. This repair measures **2 of 26**. If it comes back 1 of 26, this moved a telemetry
  field and no decision, and it reverts.
- **Approval:** **user-visible, though the count does not change.** `spike_chat_lake.py:1173-1193`
  is the one path where a consciousness claim reaches a human. The tally stays 4 of 4; the
  *disclosure text* changes, and it must name `consciousness_probe.py:84` rather than
  `_BIND_IGNITE`. Ship 0.8 in the same commit. Not an arm-envelope approval, but it should not ship
  silently.

**Scope note, adopted from review:** the argument for this repair -- a marker that cannot report
ABSENT is worse than one that can -- does not stop at binding. `consciousness_probe.py:95` gates
INTEGRATION with `(pci_i > 3 * pci_n) and (pci_i > 0.25) and ...`, three more underived literals in
the same four-marker tally. Either scope the claim to the binding marker explicitly in the
disclosure, or add a census row for the other three markers so the next pass knows the tally is not
clean. Do not silently imply the tally is now honest.

---

### STAGE 3 -- verdicts pending a measurement, gated on W0.4

#### GAP 4, GAP 5, GAP 6 -- the WM attractor.

**Status: NEEDS MEASUREMENT, 7 seeds, ACTION-CLAMPED. Honest prior: EXEMPT under the shipped
constants, with one structural blocker that is permanent.**

The wire is one line with zero new constants. `self.brain.arm.pfc` is the correct attribute path,
with the in-tree None-guard idiom at `brain.py:2204-2205`; each `_sleep_tick` block is
`try`/`except`-wrapped and the insert must follow that shape. `maintain()` and `_run_wm_pool`
(`cortex/prefrontal_executive.py:315-348`) draw **zero** random numbers, so the A/B carries no RNG
drift. That part of the design is sound.

Four things point the other way, and they are not equally strong. Sorting them is the point:

| leg | kind | force |
|---|---|---|
| `top_down_bias` (`:350-357`) is `bias[frame[0]] += strength` and **never reads `frame[1]`** -- the decayed drive reaches no consumer at all | **structural** | permanent, no constant involved |
| the pop needs `0.98^n < 0.2`, i.e. n >= 80 consecutive delay ticks against a 12-to-14 tick night | arithmetic on two constructor defaults (`recurrence=0.98`, `hold_threshold=0.2`, `cortex/prefrontal_executive.py:88-89`, unoverridden at `autonomous_loop.py:278`) | strong, but `recurrence=0.85` gives n = 11 |
| the pre-declared falsifier already fired on seed 2 -- 8 pops, 0 of 794 actions, 0 of 794 trajectory, grown units 7 vs 7 | measurement | strong |
| the scramble control beats it on coverage on 2 of 3 seeds (theta coverage, arms A, B, C: seed 0 gives 89, 229, 312; seed 1 gives 193, 213, 250; seed 2 gives 697, 697, 658) -- a coin explores more than the WM's own time constant | measurement | strong; suggests the RELEASE does the work, not the memory |

**Therefore the verdict word must be "EXEMPT under the shipped constants, plus one structural
blocker", not "EXEMPT PERMANENTLY".** Only the `top_down_bias` leg earns "permanently"; the
arithmetic leg rests on two unarbitrated defaults, and the honest reason it may not be retuned is
already written at `autonomous_loop.py:991-993`: *the only ways to make the pop fire are to lower
recurrence or hold_threshold, i.e. to retune the module's own constants until a number moves.* P1
forbids that. It does not make the blocker structural.

- **Files if it ships:** `src/snn_data_lake/cortex/brain.py`, `docs/P6_INTERCONNECT.md`.
- **Fourth falsifier, rated highest and added here:** seed 0's baseline is **latched** -- action
  entropy 1.29 bits, longest run 72, lag-1 repeat 0.838, against seed 1 at 2.16 bits, run 11,
  repeat 0.557 and seed 2 at 2.25 bits, run 19, repeat 0.557.
  If effect size tracks how latched the baseline is, the finding is a latch defect in the live loop
  and the WM wire is being credited for fixing it.
- **Do NOT scale by K.** K=40 moves 121 of 194 actions at 400 ticks, but picking K because K=1
  moved nothing is P1 laundering, and there is no principled conversion in the tree between a
  circadian hour-step and a WM delay tick.
- **The horizon must be pre-registered with its arithmetic, and both horizons reported.** The
  derivation is clean and was left implicit: `brain.py:168` sets `day_period=200`; the pop needs
  `ceil(ln(0.2) / ln(0.98)) = 80` `maintain()` calls; `maintain()` runs only inside `_sleep_tick`
  at 12 to 14 ticks per night, so roughly 7 nights minimum; 1600 ticks is 8 nights with one night
  of headroom. Print those four inputs. **Report 400 and 1600 in the same table.** Presenting the
  horizon where the effect is nonzero as mandatory and the horizon where it is zero as forbidden,
  without the arithmetic, is horizon selection after seeing the result -- the same defect as
  choosing K=40, two paragraphs from where that is correctly refused.
- **The distinction that makes this legitimate, stated once:** horizons are measured by counting,
  gains are chosen. Report the observed `maintain()`-call count, not a tick budget, and the
  objection dissolves.
- **A real CI blind spot worth recording whichever way this lands:** `tests/test_time_cells.py:435-451`
  is the only long live-brain test and runs 300 ticks. At 400 the wire is `:969` verbatim, so CI is
  byte-identical and blind either way.
- **Approval:** none. `maintain()` mutates only `pfc._stack` and its LIF membranes, calls no
  `arm.apply`, and runs inside `_sleep_tick`, which by its own docstring never calls `brain.tick()`
  or `arm.step()`. No safety-envelope contact.

#### GAP 4a -- `plan()`'s spiking reaction time.

**Status: NOT SCHEDULED. Precondition: name the consumer.**

`plan()` returns `(goal, sum(rts) / len(rts))` at `cortex/prefrontal_executive.py:297`;
`autonomous_loop.py:1047` and the call beginning at `:1058` are bare expression statements on both
live JEPA paths. The reaction time is computed every wake tick and discarded. That much is
verified.

But this was scheduled as CLOSE NOW alongside the one item that actually changes behaviour, and it
has **no named destination**. A wire proposed without a consumer *is* the `:969` pattern, before it
is built. Two compounding problems: any use of a reaction time to modulate behaviour needs an
rt-to-gain mapping, which is a new literal arbitrating behaviour; and by F3 whatever it modulates
flows into `_route_percept_to_action`, so the terminal is the action gate -- the spoke this whole
program spends its first section disqualifying.

- **Precondition before it may be scheduled at all:** name the consumer and its falsifier. If
  `mean_rt` lands only in `state()`, it is a telemetry wire and does not count. **Assigning the
  value is not the deliverable; a consumer whose output changes is.**
- **If it is scheduled:** pre-commit to a running-baseline form (`substrate/signal_calibrator.py`,
  `POLICIES.md:52`) rather than a threshold, or the P1 defect recurs in a new module.
- **If no honest consumer exists:** record it as EXEMPT with the measured reason -- *computed every
  wake tick, no honest consumer exists in a flat reactive loop* -- and remove it from the closure
  count rather than carrying it as pending work.
- **Approval:** none.

#### GAP 4b -- grown time cells.

**Status: NEEDS MEASUREMENT. Reframed from a wiring job to a neuter A/B.**

The consumer already exists: `brain.py:1601` reads `_elapsed_time` into the timed-Go window and
`:1604` feeds `learn_interval`. The open question is not "wire a consumer" but "does growth beyond
width 64 change *that* consumer's output". Run the neuter A/B on `_time_clock`. The reframing is
the whole correction.

---

### STAGE 4 -- GAP 1 then GAP 2, both gated on W0.4 and P-INBOX

Payload before instance sharing. `docs/P6_INTERCONNECT.md` section 9 states the reason -- *eleven
private buses is a smaller defect than one bus whose single reader throws away which faculty
spoke* -- and today's gate output makes the order tighter: the payload fix touches **one** module
with its own self-check, while instance sharing touches **8** modules' constructors across 26
instantiation sites, 11 of them outside the bus self-checks. Payload-first is cheaper, more
reversible, and it is what makes the sharing measurable.

#### GAP 1 -- bus payload. Status: CHAINED behind W0.4 and P-INBOX. Design-ready.

- **The design constraint that must not be violated** (section 9): the per-subscriber axis v4.78
  removed was removed *because it was never learned*. Any fix must derive the distinction from
  something that exists -- the subscriber's declared interest, or the winner's source -- **not** a
  new learnable weight nobody trains.
- **Files:** `src/snn_data_lake/thalamus/reentrant_workspace.py` (`_deliver`, `:282-312`),
  `src/snn_data_lake/cortex/brain.py:1610`, the one live drain.
- **Falsifier, and it is sharp:** F3 records that `brain.py:1610` resolves into
  `_route_percept_to_action`. A content-sensitive payload therefore delivers richer information
  **to the action gate**. If after the fix the judged metric -- distinct non-gate terminals, today
  **1 of 15** -- is still 1 of 15, the fix enriched a telemetry path and must revert. **A rising
  read count is explicitly not sufficient**; the gate counts read sites, not distinct payloads.
- **Approval:** none.

#### GAP 2 -- instance sharing. Status: CHAINED behind GAP 1.

The gate prints its own precondition: *adding a reader is NOT sufficient while each writer holds a
PRIVATE instance -- there is no shared object for a reader to read.* Sharing is necessary; it is
not sufficient; doing it first buys nothing.

- **Files, the 8-module instance census, verified against gate output:**
  `brainstem/autonomous_loop.py:230`, `cortex/access_consciousness.py:92`, `cortex/brain.py:708`,
  `reasoning/brain_adapters/executive_control.py:102/230/233`,
  `reasoning/developmental_loop_brain.py:265`, `reasoning/rpm_solver.py:305`,
  `reasoning/spike_chat_lake.py:817`, `reasoning/train_of_thought.py:92/97`.
- **Falsifier:** *orphaned faculty writers, today 7 of 10* must fall, **and** the two-sided
  pathology check from section 6 must pass. One shared bus into a latching `step()` (`_SUSTAIN` on
  `bottom_up_supported`) is the documented route to a workspace that ignites one content forever,
  and per the section-6 table that reads as success on any distinct-count instrument.
- **A free asset nobody has used:** `reentrant_workspace.py:399-409` already ships
  `_Specialist.emit()` -- one faculty consuming another's broadcast -- passing inside the bus's own
  self-check, used by 0 faculties. The pattern P6 asks for is already written.
- **Approval:** none.

---

### STAGE 5 -- GAP 7, blocked on owner approval and on a concurrency redesign

**Status: SPLIT three ways. C3 proved impossible; C2 needs engineering *and* approval; C1 needs
approval, engineering, and a recovery policy that does not exist.**

#### C3 -- fifth percept band. PROVED IMPOSSIBLE. Do not attempt.

`_context_pattern` uses the fixed `encode_multimodal` (4 bands x 16, `B = 4` confirmed at
`PCA:83`); switching to `add_modality` resizes the Amygdala, the growing learner, JEPA and the PC
net. And there is no free band: `brain.py:1166-1177` shows `temperature` already carries
proprioception. Structural, no constant in the argument.

#### C2 -- disturbance channel. CHAINED behind a concurrency redesign, a clamp, AND approval.

The mechanism exists (`so101_arm.py:198-214`), and the percept route is dead and measurable
(`deviation` 1 distinct value, `trajectory_breach()` 0 of 200 fires). Its acceptance evidence is
the action-clamped A/B, and the null was pre-declared correctly: *`deviation` gains distinct values
and `trajectory_breach()` starts firing, but no consumer's output changes* is `:969` again.

Three blockers, two of which were not named and one of which invalidates the measurement itself:

1. **The `arm.stopped` guard is mandatory and the code does not have it.** `inject_disturbance`
   (`so101_arm.py:210-214`) writes `self.data.qpos[...]` with no `stopped` check, so an HTTP client
   could move an emergency-stopped arm and the measured bitwise-freeze property at
   `so101_arm.py:15-18` becomes false. **This guard is correct with or without the measurement and
   should not wait on W0.4.**
2. **It also has no joint-limit and no finiteness check, and the illegal pose persists.**
   `:212` is `self.data.qpos[spec.qpos_index] += deg * _DEG` with no range check -- while
   `probe_contact` at `:110-114` explicitly validates `m.jnt_range` and returns `OUT_OF_LIMIT`. The
   counterfactual body validates limits; the live body does not. Because it runs `mj_forward` and
   not `mj_step`, no constraint solver runs, so `state()` reports the out-of-limit pose; and if the
   arm is latched, `apply()` returns before `mj_step` (`:266-268`), so the illegal pose is the
   arm's reported state indefinitely. Downstream, `brain.py:1169-1177` folds `theta_deg` into
   proprioception inside a bare `except Exception: pass`, and `min(1.0, max(0.0, nan))` returns
   `0.0` -- proprioception silently pins to a constant, the repo's own "a constant where a computed
   value belongs" tell, reached without raising.
   **Guard must be `if self.stopped or not math.isfinite(deg): return self.state()`, plus a clamp
   to `arm_config.JOINT_LIMITS`. Falsifier: POST a 10 000-degree disturbance; `probe_contact` on
   the resulting qpos must not return `OUT_OF_LIMIT`.**
3. **A second POST route violates the cockpit's written no-lock contract, and the race erases C2's
   own measurement.** `cockpit.py:37-43` states it: *there is no lock anywhere in the brain stack
   ... calling that from an HTTP handler while the daemon ticks the same object is a data race on a
   brain that is also writing to disk. `Turns` therefore hands the text to whoever owns the tick
   loop.* The corruption argument needs no claim about MuJoCo's GIL behaviour: `apply()` is a
   multi-statement Python sequence, and the tick thread can be preempted between
   `d.ctrl[spec.qpos_index] = got` (`:315`) and the `mj_step` loop (`:330-331`), and again before
   `self._settled = np.array(d.qpos[: m.nu], dtype=float)` (`:333`). An interleaved qpos write lands
   inside the gate's critical section and `_settled` then records the disturbed pose as settled --
   **erasing the very deviation C2 exists to create.** C2's measurement is self-defeating under
   concurrency, in the direction that produces a **false null**. A trap for the implementer:
   `brain.py:2714` `tick()` says *"Called under a lock -- returns fast"*, which is the daemon's
   single-instance process lock, not a mutex, and reads as if mutation were already serialized.
   **Repair: route the disturbance through the existing `Turns` queue, or a sibling queue drained by
   the tick owner. Never call `inject_disturbance` from a request thread. Falsifier: compare
   `threading.get_ident()` at the apply site against the tick owner's; if it can execute on a
   handler thread, it is not fixed.**

**C2 needs owner approval.** The claim that only C1 does is wrong -- see the folded question below.

#### C1 -- object and proximity channel. Blocked on approval AND on a recovery policy.

`brain.py:1158-1163` states the invariant C1 relaxes, twice: the docstring says *"distance pinned
SAFE so the E-STOP never trips"* and the body comment says *"`distance` -- the ONLY E-STOP channel
-- is deliberately untouched, so the reflex path is byte-identical and the safety envelope is
unchanged."* `CONTACT = 0.04` at `arm_config.py:27`.

**Beyond approval, C1 as specified creates an unrecoverable E-STOP deadlock, and it was not
named.** `_CLEARS_LATCH = ("HOME", "RETRACT_ALL")` (`so101_arm.py:75`), and `apply()` at `:266-269`
returns `state()` **without stepping** for every other command while `stopped`. Only those two
commands can ever un-latch the arm. Every issue site in `src/` is a self-check or a demo --
`so101_arm.py:461/:465`, `cerebellum/spiking_pid.py:220`, `brainstem/cpg_oscillator.py:74`,
`brainstem/serial_arm.py:102/:162` -- plus the separate `interpreter.py:40` reflex map used by
`signal_loop`, and the `nl_interface.py:43-44` chat mapping. **`AutonomousArm.step()` issues
neither.** Today this is inert only because `brain.py:1161-1163` pins `distance` SAFE. C1 relaxes
exactly that pin: the first real proximity trip latches the arm, `apply()` then refuses every
subsequent command, `_wake_tick` keeps running, the brain keeps selecting actions, `_settled` never
updates, and learning continues against a body that cannot move. The failure is **silent** -- events
look normal and only `state()["stopped"]` shows it.

**And the obvious recovery is documented-forbidden.** `autonomous_loop.py:1377-1386` already
records both halves: *"ArmSim.apply("HOME") clears `stopped`, so firing this while E-STOPPED would
un-freeze an emergency-stopped arm. It must never run in that state"*, and separately that HOME was
measured NET NEGATIVE over 5 seeds at 600 ticks (84.4 to 80.2 cells) and replaced by a local
`_A_POS` lift. Compounding it, `_home()` (`:173-183`) calls `mujoco.mj_resetData` **and** HOME is in
`_CLEARS_LATCH`, so one HOME deletes the scene object and un-freezes the arm in the same call --
the arm resumes into a world whose hazard the sim just removed.

- **Engineering prerequisite:** a bounded-latch recovery policy -- a re-arm path the brain can
  reach (an explicit `RETRACT_ALL` in the action vocabulary, gated on N consecutive clear-distance
  ticks), plus `stopped` in the loop's own assertion set. **Falsifier:** drive the arm into a latch
  and count ticks-to-recovery; if unbounded, C1 does not ship.
- **Constraint on that recovery:** it must not `mj_resetData`, or `_home()` must re-place the scene
  object. **Falsifier:** place an object, latch, recover, assert the object survives in `d.qpos`.
- **Two more that ship a lie if omitted from the same commit as C1:** an OBSTACLE verdict in
  `_verdict()` -- measured, an object at the gripper gives body-id pairs (8,6) and (8,7), no body 0,
  so the arm reports SELF-collision when a user hits it; and a decision on `mj_resetData` silently
  un-placing the object on `_home()`, which is also the autonomous postural-recovery command at
  `autonomous_loop.py:1389`.

#### The approval questions

None of these is a subagent's or a coordinator's to grant.

| | question |
|---|---|
| **A1** | May a measured proximity drive the E-STOP channel -- relaxing `brain.py:1161-1163`? |
| **A2** | **May a local HTTP client cause arm motion or trip the E-STOP?** One yes or one no. |
| **A3** | May the vendored MuJoCo scene be augmented programmatically? (`so101/SO-ARM100-LICENSE` unread -- confirm it permits this.) |
| **A6** | **May the brain clear its own E-STOP?** |

**A2 is deliberately one question, not three.** `cockpit.py:186-188` 404s every POST but
`/api/chat`; `cockpit.py:52` records *"loopback-only binding makes this low-risk, not no-risk --
anything local can POST"*; `/api/chat` only enqueues text; and `brain.py:26-31` guarantees the chat
path *"never sends HOME / RETRACT_ALL / JOINT_* / any motor command."* So the true statement is
sharper than the three separate items it replaces: **today no HTTP request can cause arm motion at
all.** C1 and C2 together make loopback HTTP the system's first actuation surface. Splitting that
across a second POST route, a TEST-ONLY promotion, and an E-STOP reachability question lets it be
approved piecemeal, which is how the envelope gets crossed without anyone deciding to cross it.

**A6 is new and is a strictly larger authority than A1 or A2.** It follows from the deadlock: C1
without a re-arm path is unshippable, and every re-arm path is the brain clearing its own emergency
stop.

---

### STAGE 6 -- GAP 9, chained behind a pin and a restatement

**Status: CHAINED. And the item as proposed closes a different gap than the one stated.**

There are **three** serve modules, not one:

| module | status today |
|---|---|
| `cortex/spiking_lm_serve.py` | already executed by `run_selfchecks.py:83` |
| `reasoning/ar_spiking_lm_serve.py` | already executed by `run_selfchecks.py:339`, already consumed by `reasoning/blimp_bench.py:171` (the `SERVES` table), already covered by `tests/test_blimp_bench_serve.py` |
| `reasoning/eprop_lm_serve.py` | already executed by `run_selfchecks.py:340` |

So "the serve is dormant" is not true of any of them. The proposed fix -- `.[dev,train]` plus a
transfer test -- tests the torch-to-zero-dep **weight bridge**, which is a different claim from the
gap. **Rename the item to what it is: a control-relative regression test for the training-to-serve
weight bridge**, move it out of the gap ledger into ordinary test work, and restate GAP 9 with the
three modules distinguished. It may already be closed, or may not exist.

The test itself is worth building and its shape is right:

- **Change:** add the extra to the CI test job; one new test asserting the transfer
  **control-relatively** -- same-weights agreement must strictly exceed the untrained-twin control.
  No literal arbitrates; the control does. Measured margin 20 to 29 of 40 versus 0 to 1 of 40.
- **Pin the dependency.** `pyproject.toml:67` is `train = ["torch", "torchvision", "snntorch>=0.9",
  "numpy"]` -- a **floor** -- while `:66` is `gpu = [..., "snntorch==1.0.0", ...]` -- a **pin**. The
  file's own comment at `:79` insists ruff is pinned and not floored, and records the institutional
  lesson from exactly this failure mode. An unpinned snntorch lets any minor release move the
  argmax-agreement count the new test asserts on: CI determinism would depend on a third party's
  release schedule. **Pin `snntorch==1.0.0` in whichever extra CI installs.**
- **Name the seed before running.** `SpikingLM(cfg)` init is currently unseeded and agreement swung
  35 of 40 to 25 of 40 across two unseeded runs. A 10-point swing means the seed is choosable after
  seeing the result. State the value in the spec, and assert the control margin over at least 3
  seeds so a single lucky draw cannot carry it. `40` should be stated as the eval-set size it is,
  not a tuned n.
- **CI install lines, corrected:** `.github/workflows/ci.yml:42` is `pip install -e .` on the
  self-checks matrix job (py 3.11/3.12/3.13); `:85` and `:110` are the two `.[dev]` jobs. There are
  three install lines, not one, and the item must say which it means.
- **Falsifier:** substitute the untrained twin's weights; the assert must drop to 0 to 1 of 40. If
  it does not, the test asserts nothing `run_selfchecks.py:83` does not already prove and should be
  dropped rather than kept green. Second falsifier: if `importorskip("snntorch")` still skips in the
  CI log after the install change, **the gate does not exist** -- a skipping test is not a gate, and
  the honest fallback is a P3 case-3 EXEMPT with a measured reason, not a green skip.
- **Do NOT wire the serve into the live brain.** 0 matches in `docs/P6_INTERCONNECT.md` and
  `tools/`, no artifact to serve, and `docs/coverage/Language.md:243-249` pre-declares Language
  unreachable by wiring.
- **Approval:** none.

**The coverage warning attached to this item is false and should be dropped.**
`pyproject.toml:141` is `omit = ["*/reasoning/*", "*/training/*"]`, so `training/spiking_lm.py` is
**already outside** the `--cov-fail-under=70` denominator; adding a test that imports it cannot move
the gate. The adjacent risk that is real and was not named: `cortex/spiking_lm_serve.py` is **not**
omitted, so a test importing the cortex serve does add statements to the denominator. Measure that
one, and pre-commit to the disposition before seeing the number -- if an honest test would breach
70 percent, the gate moves or the file is excluded, and the test lands. Deciding that order after
seeing the number is how the test quietly disappears.

---

## 5. Findings, not tasks

Items that turned out to be completed results. A proof is a deliverable.

### GAP 8 -- wikitext as a grounding source. PROVED IMPOSSIBLE.

The proof rests on **one** leg, not two. Both were originally presented as independent structural
proofs; only the second is extractor-independent.

- **Leg 1, demoted to a measurement.** 363 of 1,870,332 rough sentences contain a full KB triple,
  0.019 percent -- an upper bound on **exact-symbol co-occurrence**, not on triple-asserting
  sentences under any extractor. A paraphrase-tolerant extractor is not excluded by it. Still
  decisive against the swap on its own terms: the incumbent `load_kb()` has **605 curated triples**,
  so the hybrid yields fewer scenes than the source it would replace. 486 of 493 single-token KB
  symbols do occur; the symbols are present, the scenes are not. The parquet footer settles intent:
  one column, `{"text": {"dtype": "string"}}` -- no scene, no annotation, no alignment.
- **Leg 2, and it carries the proof.** The restriction rule's job (`research.md:94-95`) is to keep
  the organ off pairs it could not have acquired. Under a 338,235-type lexicon it admits
  **41,567 of 67,000 pairs and 65 of 67 paradigms** -- overshoots of 208x and 8x against the
  pre-registered floors. **A control that admits everything is not a control.** This disqualifies
  the swap independently of pre-registration, and independently of any extractor. It is also
  robust to the floors being wrong: 208x is far outside any plausible re-derivation.
- **Amend `verdict.md`, which currently reads as if pre-registration were sufficient.** R6 *did*
  pre-register wikitext. It fails on the second test, not the first.
- **Falsifier, with its borrowed constant removed.** The original ">= 200 verified sentences" is the
  BLiMP 200-pair floor wearing a different job's clothes: `research.md:64` sets that floor and
  `:67-91` derives it from the standard error of a paired difference across a 2.6-point per-char
  logprob band, on pair lists both arms score identically under contract B1. A wikitext sentence
  count has no paired structure, no discordance rate, no scoring convention and no 2.6-point band --
  every clause of the derivation is absent. **The constant-free falsifier is the incumbent: the swap
  must produce more verified scenes than 605.** The measured co-occurrence ceiling of 363 already
  fails it with no threshold at all.
- **Files:** `specs/005-language-organ/research.md` (R6, line 409 -- cite the number instead of the
  judgement), `specs/005-language-organ/verdict.md`.
- **The live route costs one probe, not a feature:** an arm-state renderer's **lexicon only**, run
  through `tools/lang_coverage.py` exactly as the KB lexicon was at `:309`, before any organ is
  built. Prerequisite genuinely absent -- 0 matches for scene, describe, or caption across
  `src/snn_data_lake/brainstem/`. Predicted 3 to 8 of 67 paradigms, at or below the floor. **If it
  also misses, the conclusion is not "try a third corpus"** -- it is that BLiMP is the wrong
  instrument, and `docs/POLICIES.md:155` applies: build the benchmark first.
- **Why to record this loudly:** the risk runs the other way. Adopting wikitext would produce a
  **green** A/B that is a corpus-size result wearing an assembly-calculus label. This repo has the
  worked case in `probe_fluid.py`, saturated and now banned as a signal. A NOT-P2-ELIGIBLE record is
  honest; a mislabelled green is not.

### GAP 3's A3 route -- source-permutation surrogate. PROVED IMPOSSIBLE, structurally.

`brain.py:1746`, `:1751`, `:1755` post exactly three submits per tick with disjoint contents:
`("percept", digit-or-"ctx")`, `("self", "self")`, `("salience", "threat"-or-"novel")`. That is a
constant bijection, so **every coalition has exactly one supporter structurally, not merely on the
seeds tested**. `reentrant_workspace.py:212` is
`base[content] = min(1.2, base[content] + self._COAL_BONUS * (len(sources[content]) - 1))` with
`sources` keyed by content, so `_COAL_BONUS = 0.06` multiplies 0 on every live tick. A3's surrogate
is the identity map, `log(p_live / p_surr) == 0` on every bin, and no sign change exists to find. A
dead constant cannot smuggle anything.

Second, independent blocker: the recorded state is insufficient. 193 of 582 drive samples are
recoverable, and they are the right tail only (winner min 0.515 versus all-drives min 0.078).

**Do not build the H_live / H_surr ring buffer.** The cheap assertion
`sorted(live_drives) != sorted(surr_drives)` goes red immediately today.

**And do not build the named prerequisite either.** "A post topology where at least one content has
more than one supporter" means arranging for two different sources to emit the *same content
string*, purely so a dead constant becomes non-zero -- which makes `_COAL_BONUS` vary and moves the
winner drive into the action gate. **Manufacturing a coincidence to animate a constant is the
failure mode, not the route out of it.** State it as a finding: the coalition mechanism is
unreachable under the live posting topology and is EXEMPT until some faculty independently has
reason to post a content another faculty also posts. **Forbid engineering the collision.**

### GAP 7's C3 -- fifth percept band. PROVED IMPOSSIBLE.

See Stage 5. Structural, no constant in the argument.

### Small corrections of record

- **The non-slot binding channels fire 4 of 12 steps, not every step.** Asserted without
  measurement in a document that demands measurement. The conclusion about the K-squared budget is
  unaffected; the raster in section 3.2 is the correction.
- **`trajectory_breach()` does not have zero callers.** Callers exist at
  `cerebellum/spiking_pid.py:240`; `inject_disturbance` has callers at
  `hippocampus/ring_attractor.py:163`, `cerebellum/spiking_pid.py:239`, `brainstem/live_arm.py:401`.
  They look like self-check and demo paths, but "zero callers" as written is false. The live-percept
  claim ("0 of 200 fires on the wake path") stands; the call-site claim does not.
- **`run_selfchecks.py:83` is `cortex.spiking_lm_serve`, not the binding self-check.**
  `thalamus.global_workspace` is at `:165`. The stage that corrects everyone else's citations
  carried the same defect on its own flagship.

---

## 6. Explicitly rejected

Recorded so they are not re-proposed.

**R-1. Rejected: "the most valuable single deliverable makes the system claim less, 4 of 4 to
3 of 4."** Measured false under every variant. The proposal's own two changes cancel each other and
leave 0 of 3 verdicts changed; the strict-decoder repair also leaves the verdict PRESENT at the
shipped constants. Only precision-only (K=2 margin -0.0083, K=4 0.2500) or F1-over-cross (+0.1583,
0.4000) flip it, and both take `global_workspace.py:609` red and are decided by a seed band wider
than the margin. **The honest deliverable is a marker that can fail, not a smaller count.**

**R-2. Rejected: capping `bind_decode_spiking` at K pairs.** It is the counter-fix -- it restores
today's numbers exactly -- and it smuggles a selection rule. The function returns an unordered
`set`; capping it at K requires a rule for *which* K, and every pair is equally synchronous, so the
rule is either a ranking that does not exist or the iteration order of a Python set. It also
promotes `_BIND_TSLOT = 6` and the `sustain=4` default into arbiters of a user-facing verdict; under
the strict decoder both stay inert (strict F1 is 1.0000 across `_BIND_TSLOT` 1 through 12 and
`_BIND_LEAK` 0.10 through 0.90).

**R-3. Rejected: "precision-scored, spiking and rate are tied at every K, so report the identity and
conclude synchrony buys nothing at K>=2."** The measurement behind it is right -- cross-product
spiking precision is 0.5000 / 0.2500 against a rate control at 0.5083 / 0.2646, a tie -- but the
conclusion is about the *reader*, not the representation. Under the strict decoder spiking precision
is 1.0000 at K=1, 2 and 4 while the rate control is 1.0000 / 0.5083 / 0.2646. **They are not tied.**
Concluding that synchrony is worthless from a decoder that unions across steps of different
multiplicity is drawing a structural claim from a defect in the instrument -- the exact error this
program exists to catch.

**R-4. Rejected: `_BIND_IGNITE = _BIND_LEAK * _BIND_VTH` as a zero-delta consistency fix.** See
P-IGNITE. It breaks `global_workspace.py:600` via the `drive=0.4` arm at `:500`, and it deletes the
subliminal control rather than deriving a constant.

**R-5. Rejected: the `_ACC_LO` row of Stage 0.4.** See 0.4b. The module's own comment records that
the bottom-up calibration was tried and rejected, and the correcting figure matches neither measured
quantity. Held pending a statement of what 0.633 is.

**R-6. Rejected: GAP 4a as a CLOSE NOW item.** Reclassified to NOT SCHEDULED with a named
precondition. A wire with a verified source, a verified discard, and no named destination is the
`:969` pattern before it is built. Not reclassified all the way to EXEMPT, which is premature -- a
consumer may exist; nobody has named one.

**R-7. Rejected: the coverage-denominator warning on GAP 9.** `pyproject.toml:141` omits
`*/training/*`, so the named module cannot move the gate. The adjacent real risk
(`cortex/spiking_lm_serve.py`, not omitted) replaces it.

**R-8. Rejected: the ">= 200 verified wikitext sentences" falsifier.** Borrowed from the BLiMP
paired-difference floor, whose entire derivation is absent here. Replaced by the incumbent's 605
triples, which needs no threshold.

**R-9. Rejected: "EXEMPT PERMANENTLY" for GAPs 4/5/6 as a whole.** Only the `top_down_bias` leg is
structural. The rest rests on two constructor defaults, and P1 is the reason not to move them --
which is a policy blocker, not a permanent one. Say which.

**R-10. Rejected: "a phase that ends with more exemptions than wires has succeeded" as this
program's success criterion.** It is unfalsifiable and it exempts the program from the standard it
applies to every item. Replaced by the pre-registered gate numbers in section 1.

---

## 7. Build order

Dependency-ordered. Items at the same number are order-independent with respect to each other.

1. **W0.4 with the counterfactual recorder** (`tools/charge_probe.py`). The root: four items are
   uninterpretable without it, and building it as originally specified would poison all four.
2. **P-INBOX** (`reentrant_workspace.py:137`), one line, 0 readers, gates GAP 1 and GAP 2.
   **P-IGNITE**, a fence, costs nothing.
3. **GAP 3 rebuilt** (strict decoder + precision-and-recall + the disclosure rewrite from 0.8).
   The only gap that closes now.
4. Stage 0 corrections, minus 0.4b and 0.8.
5. **GAPs 4/5/6** clamped sweep, both horizons reported. Expected outcome: EXEMPT with a measured
   reason.
6. **GAP 1**, then **GAP 2**. Payload before sharing.
7. **GAP 9** renamed and rescoped, snntorch pinned, seed named.
8. **GAP 7** -- nothing until A1, A2, A3, A6 are answered. C2 additionally needs the queue-routed
   mutation and the limit clamp; C1 additionally needs a bounded-latch recovery policy.

---

## 8. Is "all of them" achievable? No, and part of that is proved rather than asserted.

- **1 of 9 is PROVED IMPOSSIBLE.** GAP 8's disqualifying leg is a degenerate control admitting
  41,567 of 67,000 pairs -- extractor-independent and robust to the floors being wrong. "All of
  them" is false by proof, not by opinion.
- **2 more sub-items are proved impossible**, GAP 3's A3 route and GAP 7's C3, both structurally.
- **3 of 9** (GAPs 4, 5, 6) carry a structural blocker no measurement can remove: `top_down_bias`
  discards the decayed drive. The remaining legs are blocked by P1, which is a policy, not physics
  -- so the honest verdict is EXEMPT under the shipped constants, not EXEMPT permanently.
- **1 of 9** (GAP 7) is gated on authority nobody in the engineering loop holds, and one of its four
  approval questions -- may the brain clear its own E-STOP -- exists only because a deadlock was
  found while writing this.
- **2 of 9** (GAPs 1, 2) are the only items that could move the gate's status line, and both are
  chained behind an instrument that does not exist.
- **0 of 9 close as a shipped P6 faculty-to-faculty wire.**

The single most consequential finding is not on any gap list: **the root is a missing instrument,
not a missing wire.** `tools/charge_probe.py` has no action clamp, so every "N of M actions changed"
number in the inputs is denominated in the one terminal F3 exists to disqualify. One missing
instrument gates four gaps, and its first specification could only have read a disqualified
terminal.

The second is that **the flagship deliverable does less than advertised and is still worth
shipping.** It does not make the system claim less. It makes a marker that was structurally
incapable of reporting ABSENT capable of it -- 0 of 26 perturbations flipped it before, 1 of 26
flips it now -- on the one code path where a consciousness claim reaches a human. That is a smaller
claim than the one it replaces, and it is one that measurement supports.
