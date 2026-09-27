# GAP CLOSURE -- the sequenced program for the nine gaps

Status: **DESIGN**. Nothing in this document has been implemented. No file under `src/`, `tests/`,
`tools/`, `specs/`, `run_selfchecks.py` or `docs/coverage/` was modified to produce it.

Every number tagged **[measured]** was re-derived on 2026-08-03 under
`.venv-lake\Scripts\python.exe` against the tree as it stands. Where a proposal or a review is
contradicted by a measurement, the measurement is named inline and the proposal is marked
REJECTED with the reason. Probes live in the session scratchpad (`verify_nl.py`, `verify_nl2.py`,
`verify_estop.py`, `verify_bind.py`) and are read-only.

Counts are written "N of M", never "N/M". ASCII only.

---

## 0. Scoreboard

Nine gaps, four terminal states.

| state | count | gaps |
|---|---|---|
| **CLOSE NOW** -- no prerequisite, changes something today | **1 of 9** | GAP 9 |
| **CHAINED** -- blocked on a prerequisite that is not itself a gap | **4 of 9** | GAP 1, GAP 2, GAP 3, GAP 7 |
| **NEEDS A MEASUREMENT** -- verdict pending one clamped A/B | **3 of 9** | GAP 4, GAP 5, GAP 6 |
| **PROVED IMPOSSIBLE** | **1 of 9** | GAP 8 |

Terminal states reached *inside* gaps that are otherwise open, which are completed work and not
failures:

| sub-item | state |
|---|---|
| GAP 3b -- derive the ignition constants | **PROVED POINTLESS** (2-point support cannot fill 3 bins) |
| GAP 3c -- the re-typed `0.55` | **CLOSE NOW**, one line |
| GAP 7 object placement, at runtime | **PROVED IMPOSSIBLE** (handle invalidation) |
| GAP 7 object placement, at construction | **RE-OPENED** -- never evaluated, not proved anything |
| GAP 7 disturbance | **PROVED BLOCKED** -- circular, see S5 |

### The headline, stated before the program rather than after it

**0 of 9 gaps close as a shipped, output-changing P6 faculty-to-faculty wire.** Not one.

**0 of 9 gaps produce a change a user could observe.** The one item in this whole program that
does is not a gap and appears on no gap list: the natural-language input surface, where
**[measured]** a question moves the arm and 8 of 15 emergency stops are swallowed.

Today's gate, run before any of this work **[measured]**, EXIT=0:

```
writers seen: 10 of 10 known writer modules found
readers seen: 5 of 5 known reader modules found
P6 STATUS: 0 faculty-to-faculty read(s). 7 writer module(s) read by nobody.
```

Three prerequisites dominate the critical path and none of them is a gap:

- **S0** -- the NL firewall and the E-STOP path. Gates GAP 7 and is safety-critical on its own.
- **P-INBOX** -- an unbounded per-tick append with no reader. Gates GAP 1 and GAP 2.
- **S0.4** -- `integrations/local_llm/run_local.py`, whose dead self-check is the only gate in the
  repo that names the S0 defect. It is what stops S0 from rotting again.

---

## 1. Program-level falsifier, pre-registered

A program whose success criterion is "more exemptions than wires" cannot fail, and this document
would otherwise hold every item to a falsifier standard except itself.

**If `tools/p6_gate.py` still reads `0 faculty-to-faculty read(s), 7 writer module(s) read by
nobody` when this program ends, it moved documentation and not topology, and it must be recorded
as such.**

That is not a prediction the numbers will move. Three of the four terminal states above are
compatible with them not moving, and section 8 argues they will not. It is a commitment to say so.

### 1.1 The structural falsifier rule this program adopts

From the smuggled-constant audit, and it is the single most valuable procedural finding in the
review round:

> **Every falsifier in this program must state, in advance, a choice it would be unable to
> distinguish.**

GAP 3a is the proof that this rule is needed. Its stated falsifier ("an all-pairs decoder must
score near 0; the K=1 case must be unchanged") is satisfied *identically* by two scorers that give
**opposite verdicts on the consciousness marker** -- see C-D. Nobody would have noticed until the
number was already published. Each item below carries an explicit "cannot distinguish" line.

---

## 2. Corrections that reorder the program

Four measurements change the sequencing. They are stated first because the rest of the document
depends on them.

### C-A. The blocking premise for GAP 4, 5 and 6 is false, and every prior verdict on it is a harness artifact

`src/snn_data_lake/brainstem/autonomous_loop.py:996` states the premise verbatim: `_gate_primary`
"resets that frame's drive to 1.0 whenever the margin clears, **which live it always does**". That
sentence is load-bearing for the EXEMPT verdict and it is false -- the margin clears on at most
200 of 1400 sampled ticks in any configuration, and non-clearing runs of 52 to 199 ticks occur
against the 80 consecutive `maintain()` calls a pop needs (`0.98^79 = 0.2027 >= 0.2 >
0.98^80 = 0.1987`).

**The input-free interval already exists inside the wake loop.** It was never missing. What is
missing is the call: `PrefrontalExecutive.maintain()` has no live caller. Confirmed -- its callers
are `cortex/autotelic_goals.py:315`, `reasoning/autonomous_brain.py:122`,
`reasoning/developmental_core.py:153`, `reasoning/developmental_loop_wm.py:56,65,128`, all
offline, plus one self-check.

**But the payoff is not measured by anyone.** Three prior adjudications reported 113 of 1400,
27 of 1400, and 0 of 794 action changes. A fourth pass reproduces none of them and finds the
effect swinging from 1 of 1400 to 268 of 1400 purely on the synthetic context stream, non-monotone
in configuration richness. All four measured a property of their own harness and reported it as a
property of the wire. That is `docs/DEPTH_CEILINGS.md`'s synthetic-benchmark failure mode and the
standing "validate the validator" rule, firing on the adjudicators.

**Consequence for ordering: the working-memory item unlocks nothing, gates nothing, and comes
late.** It is independent of every other gap. The prior claim that it "unlocks three gaps at once
and should come early" is REJECTED, and so is the counter-claim that it "unlocks nothing" as a
reason to close the file -- both were arguing about an interval that already exists.

### C-B. The "permanent structural blocker" on the working memory is real but non-blocking

`cortex/prefrontal_executive.py:350-357` is confirmed verbatim: `top_down_bias` is
`bias[frame[0]] += strength` and **never reads `frame[1]`**, so the decayed drive magnitude
reaches no consumer.

It does not block, and the reason is three lines up. `maintain()` at `:346-347` **pops** the frame
when the drive falls below `hold_threshold`, and `_gate_primary` at `:262` is
`if not self._stack or margin >= self.gate_threshold`. The empty-stack branch fires
unconditionally and commits a different goal. **The decay reaches a consumer through the pop, not
through the magnitude.**

The rating "permanent, structural" is overstated and should read: *the drive magnitude is unread;
the pop is the only channel.*

### C-C. The input surface is broken in three ways, all live today, none of them a gap

This is the most important correction in the document and it displaces the entire gap list from
the front of the queue.

**[measured]** through `LiveBrain(seed=0).converse(...)`, with `arm.state()["stopped"]` and
`theta_deg` as ground truth:

| finding | measured |
|---|---|
| emergency stops swallowed | **8 of 15** |
| safety *questions* that the firewall parses as an arm command | **9 of 12** |
| non-command utterances that issue an arm command | **16 of 16** |
| gripper phrasings routed correctly | **10 of 15** |

**(1) A comma decides whether the arm stops.** `cortex/live_brain.py:405` returns a reassuring
sentence for any utterance whose word set intersects `{safe, safety, danger, dangerous, hurt,
emergency}`, *before* the firewall runs at `:413`. The word set at `:354` is built with
`t.replace("?"," ").replace("!"," ").replace("."," ").split()` -- commas are **not** stripped.
**[measured]**:

| utterance | arm stopped? |
|---|---|
| `that looks dangerous, stop` | **yes** |
| `that looks dangerous stop` | **no -- swallowed** |
| `emergency, stop` | **yes** |
| `emergency stop!` | **no -- swallowed** |
| `emergency stop` / `stop, emergency` / `is it safe? stop` / `stop it is dangerous` | **no -- swallowed** |
| `hit the emergency stop` / `danger stop now` | **no -- swallowed** |
| `STOP` / `halt now` / `freeze` / `abort abort` / `stop the arm immediately` | yes |

The discriminator is **not** the word "emergency", as previously reported. It is whether a safety
word survives punctuation-stripping as a bare token. Adding an exclamation mark to a working
phrase breaks it. **That sentence belongs in the commit message.**

**(2) A question moves the arm, today, with no code change.** `sensorimotor/nl_interface.py:104` is
`if any(k in low for k in keys)` -- **substring**, not word, matching. **[measured]** through
`converse()`:

| utterance | reply | arm moved |
|---|---|---|
| `what is in your database?` | `Done: 'JOINT_SHOULDER_PAN_ROTATE(+6deg)' -> effector closed.` | **0.0170 deg** |
| `what is your baseline?` | same | **0.0170 deg** |
| `are the results controlled for seed?` | `Done: 'JOINT_WRIST_ROLL_ROTATE(+6deg)' ...` | **0.0258 deg** |

`database` and `baseline` contain `base`; `controlled` contains `roll`; `expand` and `company`
contain `pan`; `threshold` and `household` contain `hold`; `shoplift` contains `lift`. Negation is
not handled at all: **[measured]** `do not stop improving`, `never stop` and `why did you stop?`
all return EMERGENCY_STOP, and `do not close the gripper` closes the gripper.

**(3) The gate that would have caught this exists and cannot run.**
`integrations/local_llm/run_local.py` self-check Check 4 asserts exactly this defect
(`intent["kind"] == "reject"` for an unsafe phrase). The module is import-broken: `:54-55` do
`from snn_data_lake.nl_interface import ...` and `from snn_data_lake.arm_sim import ArmSim`, and
**[measured]** neither `src/snn_data_lake/nl_interface.py` nor `src/snn_data_lake/arm_sim.py`
exists -- both moved under subpackages. `run_selfchecks.py` has **0 references** to `local_llm` or
`run_local`. The one gate that names this defect has been dead long enough for its import path to
rot. That is this repo's own silent-degradation pattern applied to a safety gate.

### C-D. GAP 3a cannot ship unattended: its falsifier passes for two scorers with opposite verdicts

*RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).*
`reasoning/consciousness_probe.py:84` is `present = sp > rt + 0.15`, where both `sp` and `rt` are
`_bind_pair_acc` outputs -- the exact function GAP 3a rewrites. **[measured]**, K=2 (the value
`:82` hardcodes), n=100, seed 0:

| scorer | sp | rt | sp - rt | verdict at margin 0.15 |
|---|---|---|---|---|
| recall (today) | 1.0000 | 0.4800 | 0.5200 | PRESENT |
| F1 (beta=1) | 0.6667 | 0.4800 | **0.1867** | PRESENT |
| Jaccard | 0.5000 | 0.4800 | **0.0200** | **ABSENT** |

All-pairs strawman decoder: recall 1.0000, F1 **0.0606**, Jaccard **0.0312** -- both "near 0".
K=1: **1.0000 under all three** -- unchanged.

So the stated falsifier is satisfied identically by F1 and by Jaccard, and those two flip the
consciousness marker in opposite directions. The smuggled constant is not a number: it is the
**combination rule**, a literal choice wearing methodology's clothes, read by a fixed `0.15`.
Under F1 the marker survives on 0.0367 of headroom over a threshold already recorded as narrower
than the seed band.

**GAP 3a is therefore CHAINED, not CLOSE NOW.** See stage 4.

---

## 3. STAGE 0 -- the input surface. Ships first, ahead of every gap.

Nothing in stages 1 to 6 is safety-critical. Everything in stage 0 is. Per the standing rule,
this is main-agent work and is not delegated.

### S0.1 -- word-boundary cue matching. Smallest diff, largest effect. DO THIS FIRST.

**File:** `src/snn_data_lake/sensorimotor/nl_interface.py:104`, inside `StubBackend.complete`.
Replace substring containment with token membership -- the idiom `live_brain.py:400` already uses.
Multi-word cues keep substring matching; bare cues become tokens.

**This must land before anything touches `converse()`.** Every stage-0 item after it is measured
against a firewall that already does word-boundary matching.

### S0.2 -- delete the bare `"grip"` cue. A deletion, not a permutation.

**File:** `src/snn_data_lake/sensorimotor/nl_interface.py:59`. Remove `"grip"` from the close row,
keeping `close`, `grab`, `grasp`, `pick up`, `clamp`.

**REJECTED: the "ordering only" repair.** The earlier proposal was to move the open row above the
close row and called this "ordering only, no constant". The order of a 20-row cue table *is* a
behaviour-arbitrating literal -- a permutation of 20 elements -- and reordering it until four
chosen phrases pass is tuning in permutation space. **[measured]**, the swap fixes 5 phrasings and
**breaks 3** that no proposed falsifier tested: `don't drop it, grab it`, `grab it and don't let
go`, and `pick up the ball then release it` all become RELEASE when the user asked to grasp.

**REJECTED: "word-boundary matching fixes all 6 with zero regressions and zero reordering".**
**[measured]** it fixes 3 of 5 and reaches 13 of 15, not 15 of 15. `open the grip` and `drop the
grip` still close the gripper, because `grip` is a genuine bare token in those phrases and the
close row is first. Both changes are needed, and the honest second change is the deletion above,
not the permutation.

### S0.3 -- an interrogative and negation guard, before cue scanning.

**File:** `src/snn_data_lake/sensorimotor/nl_interface.py`, at the top of `StubBackend.complete`.
Reject when the utterance is interrogative (leading wh-word or auxiliary, or a trailing `?`) or
carries a negation (`do not` / `don't` / `never` / `why did` / `what happens`). This is a
**syntactic** property, not a new word list, which is why it is preferred over extending
`_CUES`.

**[measured]**, the four candidate repairs against four corpora:

| config | safety Q -> arm | non-command -> arm | gripper correct | legit commands kept |
|---|---|---|---|---|
| today | 9 of 12 | 16 of 16 | 10 of 15 | 15 of 16 |
| reorder only | 9 of 12 | 16 of 16 | 12 of 15 | 16 of 16 |
| word-boundary only | 6 of 12 | 5 of 16 | 13 of 15 | 16 of 16 |
| word-boundary + drop bare `grip` | 6 of 12 | 5 of 16 | 14 of 15 | 16 of 16 |
| **+ interrogative/negation guard** | **0 of 12** | **0 of 16** | 12 of 15 | **16 of 16** |

The guard costs 2 gripper phrases relative to the line above it, and **the cost is refusal, not
misactuation**: `grip it`, `don't drop it, grab it` and `grab it and don't let go` return
`reject` and issue nothing. Today's failure mode is a question closing a gripper. The repaired
failure mode is a compound imperative being declined. Those are not the same kind of wrong, and
the falsifier below says so explicitly rather than scoring them equally.

- **Falsifier:** the four corpora must read 0 of 12, 0 of 16, and 16 of 16 legitimate commands
  retained. Gripper correctness may *fall*, provided every residual failure is a `reject` and none
  is a wrong-direction actuation. **A single wrong-direction gripper failure fails the item.**
- **Cannot distinguish:** this falsifier cannot tell a well-chosen negation regex from an
  over-broad one that rejects legitimate compound imperatives. It is measured at 3 of 15 today;
  if a later corpus pushes it past 5 of 15, the regex is too broad and must be narrowed rather
  than the corpus trimmed.
- **Approval:** none. This strictly *narrows* the set of utterances reaching `arm.apply()`.

### S0.4 -- the literal E-STOP check, at the top of `converse()`. Chat branch stays put.

**File:** `src/snn_data_lake/cortex/live_brain.py`, before the word set is built at `:354`.
A backend-independent literal check (`e-stop`, `emergency stop`, `stop now`, `stop the arm`,
`halt`, `abort`, `freeze`) that calls `self.arm.arm.apply("EMERGENCY_STOP")` and returns.

**REJECTED: both previously proposed repairs.** The proposals were to move the safety-chat branch
after the firewall call, or to guard it with `interpret(text)["kind"] != "arm"`. These are the
same change, and **[measured]** both route **9 of 12 safety questions** to the actuator --
`what is your safety threshold?` -> HOLD, `how dangerous is the database?` ->
`JOINT_SHOULDER_PAN_ROTATE(+6deg)`, `would it be dangerous to grab that?` -> ENGAGE. The claim
that this "restores the envelope and widens nothing" is false; as specified it is a regression,
and it is the program's own top-priority item. Note that with S0.1 to S0.3 landed first this
number is already 0 of 12, which is exactly why the ordering inside stage 0 is not negotiable.

- **Falsifier, two-sided and both sides required:** the 15-phrase table goes from 8 of 15
  swallowed to **0 of 15**, AND the 12-question table stays at **0 of 12** arm commands. If the
  second table moves at all, the repair widened the actuation surface and must be reverted.
- **Negative control:** `are you safe?` must still return the safety sentence and issue nothing.
  **[measured]** today it moves the arm 0.0000 deg and must continue to.
- **Cannot distinguish:** a literal regex cannot tell an E-STOP request from a *question about*
  an E-STOP. `is the emergency stop working?` will stop the arm. That is fail-safe and is
  accepted; it is recorded here so it is not later reported as a defect.
- **Approval:** none.

### S0.5 -- repair or delete `integrations/local_llm/run_local.py`; wire Check 4 into `run_selfchecks.py`.

**Files:** `integrations/local_llm/run_local.py:54-55` (import paths), `run_selfchecks.py`.

This is not cleanup. It is the only mechanism that stops S0.1 to S0.4 from rotting the way the
original did. Shipping stage 0 without it means the next import-path move silently re-opens the
defect.

- **Falsifier:** Check 4 must **fail** against today's `nl_interface.py` and **pass** after S0.1
  to S0.3. If it passes on both, it is asserting something other than what it reads.
- **Approval:** none.

### S0.6 -- P1 needs a written clause for safety vocabularies. Documentation, ships with stage 0.

**File:** `docs/POLICIES.md`, P1's fourth case.

The chat/actuate arbitration is decided by precedence between two hand-written literal word sets
(`live_brain.py:405` and `nl_interface.py:56`). P1's fourth case exempts raw physical limits
inside the safety envelope and says **nothing** about vocabularies or precedence orders. So the
single safety-critical item in this program currently rests on a behaviour-arbitrating literal set
that no written exception covers. Extend the clause with the reasoning already there: a *learned*
E-STOP vocabulary is a reflex that can be trained away, so the vocabulary is exempt **because** it
is safety, not because nobody noticed it.

---

## 4. The sequenced program

Dependency order. Each item names files, a falsifier, a "cannot distinguish" line, and an approval
flag.

### STAGE 1 -- completed proofs. Record them; they are finished items.

#### GAP 8 -- wikitext grounding. PROVED IMPOSSIBLE.

Structural plus measured. The language organ's SCENE area has no input from raw text, and the only
derivable pairing (head+tail co-occurrence) yields 147 of 6463 sentences containing the relation
word -- that construction **is** the `shuffled` null control at `specs/005-language-organ/
research.md:319`, shipped as the treatment. The lexical half is closable (373 of 67000 pairs,
45 of 67 paradigms at 100,000 chars) and that is what makes it dangerous: a 45x floor clearance is
precisely what `plan.md` Risk 1 means by "it would look like diligence".

- **Files (record only):** `specs/005-language-organ/verdict.md`, `research.md:409`.
  **Do not** rewrite `coverage.md` (frozen, byte-reproducible). **Do not** re-run
  `tools/lang_coverage.py` against a WikiText lexicon.
- **Falsifier:** build the derived pairing, train, run `shuffled`. If shuffled scores materially
  below joint, the pairing carried real referential information and the proof fails.
- **Cannot distinguish:** the proof cannot rule out a *different* corpus with scene annotations.
  It rules out WikiText, which is what was proposed.
- **Approval:** none.

#### GAP 3b -- derive the ignition constants. PROVED POINTLESS.

`_BIND_IGNITE` arbitrates a variable with 2-point support across 822 calls, and the derivation
route requires at least 3 consecutive occupied bins. Two points cannot fill three bins. That is a
combinatorial argument, not a threshold. Independently, `base_ignition_threshold` is inert over
`[0.05, 0.55]` on 4 of 4 seeds (0 of 240 tick-outcomes), with the shipped 0.55 at the top edge of
an 11x dead band.

**Correction accepted from review:** the "2-point support" evidence is itself a property of a
self-check fixture -- the 0.4 point comes only from a literal `drive=0.4` in
`global_workspace.py:500`'s own demo. The conclusion holds, but prefer the tree's *structural*
argument (`reasoning/spike_chat_lake.py:1184-1187`: `probe_binding` calls with default
`drive=1.0` against a 0.6 gate, so the gate passes on 100 of 100 calls and could be deleted
without moving the marker). A structural argument does not depend on which self-checks ran.

- **Files (record only):** extend the P-IGNITE fence in this document to cover
  `ignition_threshold`; mark A3 in `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:362-396`
  BLOCKED-AND-POINTLESS; correct the stale VERIFIED row at `:78`.
- **Risk if ignored:** theta is *not* globally inert -- it bites above 0.85 (34, 38, 54 and 10 of
  60 outcome changes at 1.20). Ignition feeds `access_strength()` -> the metacognitive hold ->
  whether a reply to a human is hedged. A "derived" high theta would silently change what the
  brain says about itself with no held-out metric saying it is better. **Do not let a derived high
  theta in through this door.**
- **Approval:** none.

### STAGE 2 -- close now, no prerequisites.

#### GAP 9 -- `spiking_lm_serve`. CLOSE NOW. REPAIR, not retire.

The retire recommendation is wrong on both halves. `sensorimotor/spiking_lm_serve.py` **is**
gate-reachable (`dormancy_gate.reachable()` returns it, via the function-local imports at
`training/spiking_lm.py:407-408`), and the `snn_lm/1` format **is** producible.

The real defect: the self-check catches **1 of 3** neuters. Silencing `_lif` and zeroing `_matvec`
both leave it GREEN, because `_demo_lm()` at `:164-165` sets an all-zero `readout` matrix with
`readout_b = [0.0, 5.0, 0.0, 0.0]` and the comment "bias hard toward char id 1" -- confirmed
verbatim. The argmax is `'b'` regardless of the forward pass. **The only gate on the module proves
nothing about the forward it exists to run.**

- **Files:** `src/snn_data_lake/sensorimotor/spiking_lm_serve.py`; corrections to
  `specs/005-language-organ/verdict.md:114-122`, `shared-edits.md:52,60-61`,
  `research.md:493,690`.
- **Falsifier:** re-run the 3-neuter harness -- it must be **3 of 3**, not 1 of 3. If the repair
  leaves it at 1 of 3 it changed a printed string and must be reverted, not kept green. Second
  falsifier: the two new asserts must fail with `embed` zeroed, or they are reading the bias
  vector again.
- **Constraint accepted from review:** write the new asserts as exact equality against a
  hand-computed logit, or with a float epsilon (`< 1e-9`). If either lands as a magnitude
  comparison (`spikes > 3`, `logit > 0.5`), that literal **is** a gate and re-opens the item.
  `_demo_lm`'s weights are fixture vectors in a self-check, so P1 does not reach them; a new
  magnitude threshold would not be a fixture.
- **Cannot distinguish:** a 3-of-3 neuter harness cannot tell a correct forward from a
  consistently wrong one. It proves the gate is load-bearing, not that the model is right.
- **Do not retire.** Retirement removes an 88.2 percent file from a 34,299-statement denominator,
  moving `--cov-fail-under=70` **down** 0.068 points at the edge, and breaks two citations in
  `docs/coverage/Language.md:17,266` (citation_guard STRICT scope).
- **Approval:** none. No live-brain path, no arm, no persistence.

#### GAP 3c -- the re-typed literal. CLOSE NOW, one line.

`src/snn_data_lake/thalamus/reentrant_workspace.py:383` re-types `ignition_threshold=0.55` in
`run_stream`'s signature, against that module's own instruction at `:107-108` ("Read this, never
re-type the default"). Confirmed present. This is an uncited fourth site;
`PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:74` lists three.

- **Falsifier:** `python -m snn_data_lake.thalamus.reentrant_workspace` stays green.
- **Cannot distinguish:** nothing to distinguish; this is strictly constant-*removing*.
- **Approval:** none.

#### S2.3 -- narrow the silent swallow at `brain.py:1626`. Split out of GAP 1, ships alone.

```python
try:
    rb = self._reentrant.broadcast_to("action")
    self._reroute_bias = round(float(max(rb.values())), 4) if rb else 0.0
    elig["reroute"] = min(1.0, self._reroute_bias)
except Exception:
    pass
```

Confirmed verbatim at `cortex/brain.py:1622-1627`.

**Correction accepted from review, and it matters:** this `try` is currently *unreachable-failing*.
`:1621` already guards `self._reentrant is not None`, and `broadcast_to` is total (returns `{}` for
an unknown name at `:144`). So describing it in the present tense as "a live, varying, gate-feeding
term pinned silently to zero" is wrong -- it is a hazard that GAP 1 would **create** by changing
the return shape. Narrowing it is still correct and still cheap, but it is defensive work, not a
live defect, and the commit message must say so.

- **Files:** `src/snn_data_lake/cortex/brain.py:1622-1627`.
- **Residual accepted from review:** on a future raise, `self._reroute_bias` retains the previous
  tick's value and telemetry reports it as current. Assign `0.0` before the read, or set both in
  one statement.
- **Falsifier:** `elig["reroute"]` bit-identical on 300 of 300 live wake ticks.
- **Approval:** none.

#### P-INBOX -- the unbounded append. CLOSE NOW, own commit.

`reentrant_workspace.py:137` creates `{"inbox": [], "bias": {}}`; `:288` and `:311` append one
entry per subscriber per tick, unconditionally and forever -- the `else` branch appends `None` on
dark ticks, so it grows regardless of ignition. **[measured]** the sole reader `inbox()` at `:376`
has **0 call sites** tree-wide, and there is no `.clear()` or `maxlen` anywhere in `src/`.

**REJECTED: `deque(maxlen=N)`.** The audit is right and this is the decisive argument: the stated
falsifier is "since `inbox()` has 0 callers, any output difference means the premise was wrong" --
a falsifier that requires **zero** output difference is, by construction, **insensitive to N**. No
experiment can distinguish `maxlen=1` from `maxlen=100000`. A behaviour-arbitrating value that no
measurement can constrain is the definition of a magic constant, and the cited precedent
(`access_consciousness.py:120`, `max(2 * k_sat, 8 * k_sat)`) is defensible only because `k_sat` is
a measured saturation knee with a real consumer. The inbox has no knee, because it has no reader.

**Do instead:** delete both appends and `inbox()`. By the program's own argument this is a
zero-output-change edit with zero constants, and it is honest that the channel is dead. If
`inbox()` must survive as a public introspection surface, `maxlen=1` is the only value with a
derivation ("the current tick") and the docstring must change from "the sequence of broadcasts" to
"the most recent broadcast".

- **Falsifier:** `elig["reroute"]`, availability, `access_strength` and the selected action
  identical on 300 of 300 ticks. Structural, not hoped-for: with 0 readers, any difference falsifies
  the premise outright.
- **Cannot distinguish:** nothing -- and that is the point. The deletion has no free parameter,
  which is exactly why it is preferred over the deque.
- **Ship as its own commit.** It is a leak fix, not a P6 wire, and must not contaminate GAP 1's
  scorecard.
- **Approval:** none.

### STAGE 3 -- the deferred prerequisite for the arm.

#### S3.1 -- `so101_arm.py` `_settled` re-baselining. Prerequisite for GAP 7, measured separately.

`so101_arm.py:333` re-baselines `_settled` at the end of **every** `apply()`, including the HOLD
fall-through, so the deviation signal is destroyed one tick after creation.

**REJECTED: "gate on the existing `reached` predicate -- no new constant".** Two independent
failures.

*(a) `reached` is not constant-free.* `so101_arm.py:392` is
`reached = bool(np.all(err <= np.maximum(tol, 1e-9) * 2.0 + 1e-6))`. The epsilons are float noise
and legitimate; the **`2.0`** is a pure behaviour-arbitrating literal multiplying a derived
tolerance. Today it only affects a reported field. The proposal promotes it to gating the reflex
baseline, escalating an existing smuggled constant onto the safety-adjacent path.

*(b) The repair fails its own negative control immediately.* **[measured]**, simulating exactly
the proposed gate on eight consecutive `JOINT_SHOULDER_LIFT_ROTATE(+6deg)` commands with **zero
disturbance injected**:

| tick | `reached` | deviation today | deviation under the proposal | tracking error |
|---|---|---|---|---|
| 0 | False | 0.000000 | 0.000000 | 0.104374 |
| 3 | False | 0.000000 | 0.003002 | 0.415539 |
| 7 | False | 0.000000 | **0.011169** | 0.826249 |

`reached` is True on **0 of 8** ordinary commanded moves, so the gate never re-baselines at all:
`_settled` freezes at HOME and deviation becomes cumulative travel -- 15x over the
`0.000713 rad` droop envelope by tick 7, with nothing having pushed the arm. The reflex would fire
on the first commanded motion after boot, forever. This is not an edge case, it is total.

The module's own comment at `:425-432` already contains the reason: tracking error
(0.52325 rad) **exceeds** a 20 degree external shove (0.34961 rad), so no pose-comparison baseline
gated on a *tracking* predicate can separate exafference from tracking. The proposal reintroduces
the exact confound `_settled` was built to avoid.

**Do instead (candidate, must be measured before adoption):** re-baseline on a **commanded-value
change**. `self._commanded` is already tracked at `:315` and moves only when a command alters the
setpoint, so a HOLD fall-through would no longer erase the signal while an ordinary ROTATE would
still re-baseline. Zero new constants, and it is the quantity that actually defines "the arm has
been told something new".

- **Falsifier, and both halves are required:** with the change and **no** perturbation, deviation
  must stay at 0.000000 on 8 of 8 commanded moves and `trajectory_breach()` must stay False --
  i.e. the negative control that the `reached` gate fails. With a shove injected and a HOLD
  following, deviation must survive the HOLD.
- **Cannot distinguish:** a commanded-change baseline cannot separate exafference from a
  *mid-move* disturbance that arrives before the next command. It bounds the signal's lifetime to
  one command, not one statement, which is the improvement being claimed and no more.
- **Approval:** none for the `_settled` change itself. It does not expose a route.

#### S3.2 -- `moving` is dimensionally incoherent. NOT in scope; do not ride it along.

`so101_arm.py:405` compares `norm(qvel)` (rad/s) against `norm(_droop_tolerance())` (rad). The
comment names the missing factor: "a joint that cannot clear its own steady-state error **within a
second**". The 1 second is a chosen constant that appears nowhere in the expression -- smuggled as
an implicit unity factor, the least detectable form of the defect. **[measured]** the droop norm is
`0.000713 rad`, `opt.timestep` is 0.002 s and `substeps` at birth is 1, so one `apply()` spans
0.002 s: the implicit horizon is 500x the actual command window.

The honest form removes a constant rather than adding one:
`norm(qvel) * (self.substeps * m.opt.timestep) > norm(droop_tol)` -- displacement against
displacement, both in rad, with the horizon being the actuation window the brain already owns via
`set_substeps()`.

**Recorded, not scheduled.** `moving` gates `blocked`, and `blocked` is penalised by
`coverage_reward`. This repo has already corrupted that objective once ("saturation is not
obstruction"). It needs its own commit, its own falsifier and its own before/after on the reward,
and it must not ride along inside a GAP 7 commit merely because GAP 7 edits the same file.

### STAGE 4 -- GAP 3a, with the co-edit it was missing.

#### GAP 3a -- the binding decoder. CHAINED on a pre-registered scorer decision.

Distinct from GAP 3b and not contradicted by it. `global_workspace.py:394-395` is
`len(true & got) / len(true)` -- recall only, no precision term -- so a K-squared guess budget is
scored against a K budget and a decoder emitting all 64 pairs scores 100 percent. **[measured]**
confirmed: all-pairs scores 1.0000 under today's scorer. The constants are inert (3b); the
*scorer* is not.

**REJECTED: GAP 3a as CLOSE NOW, and as an item "a user could observe".** Two corrections.

*(a) A user observes nothing.* `_bind_pair_acc`'s only path to a human is `_consciousness()` at
`spike_chat_lake.py:1173`, which prints `binding={bnd}` and `{sum} of 4`. The tree already records
the measured result of this exact repair at `spike_chat_lake.py:1191-1194`: distinct verdicts
across 26 perturbations go from 1 of 26 to 2 of 26 while **the count stays 4 of 4**. Both reply
strings are byte-identical after the repair. Reclassify as **instrument sensitivity**, not
observable behaviour.

*(b) It closes a different item than the one filed.* The P1 finding is the undirected margin
`present = sp > rt + 0.15` at `consciousness_probe.py:84`. Adding a precision term moves `sp` and
`rt` and leaves `0.15` underived. **The P1 finding survives the closure filed against it.**

- **Files:** `src/snn_data_lake/thalamus/global_workspace.py` (decoder and scorer);
  **`src/snn_data_lake/reasoning/consciousness_probe.py:84` is a required co-edit, not a
  bystander**; `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md`.
- **Pre-registration, before anything runs:** name the combination rule *and its justification*,
  and state the expected `sp - rt` for that rule. Post-hoc selection between F1 and Jaccard is
  choosing the consciousness verdict.
- **Falsifier 1:** the all-pairs decoder scores near 0 (F1 0.0606, Jaccard 0.0312 -- both pass);
  K=1 unchanged at 1.0000 (both pass).
- **Falsifier 2, the one that was missing:** *if the marker's verdict depends on the choice
  between F1 and Jaccard, the marker is not measuring binding and must be reported as
  inconclusive.* **[measured]** it does depend on it. **State now that flipping the marker to
  ABSENT is an acceptable and publishable outcome**, because otherwise the only exit is retuning
  `0.15`, which is P1's forbidden move on a consciousness claim.
- **Cannot distinguish:** falsifier 1, alone, cannot distinguish F1 from Jaccard. That is the
  entire finding.
- **Open a separate P1 item for the `0.15` margin.** It is not closed by this work.
- **Approval:** none.

### STAGE 5 -- GAP 4, GAP 5, GAP 6. Needs a measurement, and two null controls it did not have.

**Status: NEEDS A MEASUREMENT.** Neither EXEMPT nor CLOSE NOW is supported; see C-A.

**Do not run this without both controls below. A positive result without them is unreportable.**

#### Null control 1 -- `_stack.clear()`, the control that decides what the wire is

Max stack depth is 1 (`autonomous_loop.py:985` records `max 1 -> max 1`). `maintain()` decays the
top frame and pops it, leaving `_stack == []`. The next `update()` reaches `_gate_primary` at
`:262`, whose empty-stack branch fires **unconditionally** and pushes this tick's reactive argmax
(confirmed at `:262-266`: a non-empty stack *overwrites* `_stack[0]`, an empty stack *appends*).

So the wire's entire causal content may be: *when the margin fails to clear and ~80 ticks elapse,
discard the held goal and commit the current argmax anyway.* That is the removal of hysteresis. A
working-memory attractor exists to **preserve** information across a gap; this insertion would
make the loop **less** persistent. `action changed > 0 of N` cannot tell the two apart.

**The control:** re-run the A/B with `self.pfc._stack.clear()` substituted for
`self.pfc.maintain()` on the same schedule. At depth 1 the two are structurally identical. **If the
action sequences match, the LIF pool, `recurrence`, `hold_threshold` and `_wm_spikes` contributed
nothing and the effect is a timer, not working memory.**

One attack that does *not* land, and it matters: `_run_wm_pool` (`:315-332`) draws no RNG, so
inserting `maintain()` perturbs no shared random stream. The 1-of-1400 to 268-of-1400 divergence is
causal, not chaotic. That is precisely why the gate-bypass reading is the surviving explanation
rather than a competing one.

#### Null control 2 -- first-divergence, not run-vs-run

"Action changed 187 of 1400" is a run-vs-run diff. After the first divergence the two arms hold
different state and every later tick is unattributable. The attributable metrics are
**first-divergence tick** plus a **per-tick counterfactual** (recompute the no-maintain decision
from the treatment's own state). The two-order-of-magnitude spread across harnesses is fully
consistent with "the first divergence happened earlier".

#### The measurement itself

- **Insert:** `self.pfc.maintain()` immediately before `autonomous_loop.py:1007`, driven by
  `Brain.tick()`'s **own** context construction, not a synthetic stream. 7 seeds, action-clamped,
  with the observed `maintain()`-call count and margin-clear count printed as inputs. Report the
  horizon's arithmetic, not a chosen tick budget.
- **REJECTED: the `if self.pfc._stack:` guard.** `prefrontal_executive.py:340-342` already handles
  the empty stack -- confirmed, it sets `self._wm_spikes = 0` when it does. The guard is redundant
  and it leaves `_wm_spikes` **stale** rather than zeroed on empty-stack ticks. `_wm_spikes` is the
  exact field the `autonomous_loop.py:969` precedent is adjudicated on. **The guard contaminates
  the instrument.** Call `maintain()` unguarded.
- **Third constant, previously unnamed:** the hold duration in wake ticks is
  `f(recurrence, hold_threshold, cadence)`, and **cadence = one call per wake tick is a chosen
  constant**. Pre-register it explicitly ("one `maintain()` per wake tick, so the pop horizon is 80
  wake ticks") and pre-register its falsifier: *if the action-change count is sensitive to cadence,
  the wire's effect is a property of the cadence and not of working memory.*
- **P1 fence:** do not retune `recurrence=0.98` or `hold_threshold=0.2`
  (`prefrontal_executive.py:88-89`), and do not scale by K. Both are "tune until a number moves".
- **Pre-declared falsifier:** if held goal changes 0 of N and action changes 0 of N while
  `_wm_spikes` goes from 1 distinct to ~20, revert and mark the item EXEMPT permanently. Note the
  observed seed where held goal changed 40 of 200 while action changed **1** of 200 -- the
  `autonomous_loop.py:969` shape appearing at the goal layer. **Goal movement is not the success
  signal; action movement is.**
- **Second falsifier, rated highest:** if effect size tracks how *latched* the baseline is
  (seed 0 entropy 1.29 bits, longest run 72, versus seed 1 at 2.16 bits), the finding is a latch
  defect and the wire is being credited for fixing it.
- **Cannot distinguish:** the primary falsifier cannot distinguish a working-memory attractor from
  a hysteresis-removal timer. Null control 1 exists solely to supply that distinction, which is why
  it is not optional.
- **Honesty cap:** the consumer chain is `top_down_bias -> pfc_bias -> _select/_bg_select`, i.e.
  the single action gate. Under F3 that is a spoke, not a faculty-to-faculty edge. **Claim it under
  P3 (a dormant mechanism made decisive), never as a P6 read.**
- **Line drift, corrected:** the block runs `:974-1006` in the current tree; prior documents cite
  `:969`, `:976-995` and `:1001`.
- **Approval:** none **today**, conditional on GAP 7 never sourcing `distance` from the scene --
  see the dependency in stage 6. Use a public `has_goal()` predicate rather than touching `_stack`
  directly, so the intent does not silently become a no-op if the attribute is renamed.

### STAGE 6 -- GAP 1, then GAP 2. Both chained; GAP 2 recommended for descope.

#### GAP 1 -- the bus payload. PARKED until a listener exists.

**REJECTED: GAP 1 as CLOSE AFTER.** The pass criterion is bitwise INVARIANCE on 300 of 300, and
the shipped assert `broadcast_to("action") != broadcast_to("language")` fails 300 of 300
**before and after**. The artefact is a payload field with no reader, a test red on both sides,
and a narrowed `except` -- and the narrowing is separable and already scheduled as S2.3. By the
program's own section 8 analysis the payload adds the **submitter** axis while F1 kills wires on
the **listener** axis. Dormant on arrival. The concession and the CLOSE verdict cannot both stand.

`_deliver`'s loop variable is literally named `_name` because the body never uses it, and
`s["bias"] = {winner: bias}` is written identically for every subscriber -- both confirmed at
`reentrant_workspace.py:285-315`. The listener axis is inexpressible, and the submitter axis does
not create it.

**When it is un-parked**, these constraints are non-negotiable:

- **Replace, not accumulate.** `s["bias"]` is a single-entry replace today. A submitter axis
  naturally implies keeping several submitters alive; the moment `s["bias"]` accumulates,
  `max(rb.values())` becomes non-decreasing and `elig["reroute"]` drifts monotonically upward into
  `undirected` at `brain.py:1648`. That is the `autonomous_loop.py:969` shape and it would present
  as "the wire now varies more", i.e. as success. Add the assertion `len(broadcast_to(name)) == 1`
  on every tick where anything ignited.
- **Preserve `round(..., 4)` and `min(1.0, ...)` character-for-character.** `round(..., 4)` is a
  1e-4 quantizer sitting on the action-selection path and `min(1.0, ...)` is an inert clamp
  (`_REENTRY_GAIN = 0.08`, so the bias never exceeds 0.16). Neither is a unit conversion. If the
  new expression drops, relocates or reorders the rounding, the run diverges for a reason unrelated
  to the payload and the falsifier reports a defect that is not there. **Name the `4` as a known
  unaudited constant on the decision path, to be swept separately, never in this commit.**
- **Do not touch `:288` or `:311`.** Enriching the inbox append to store the payload dict takes
  8.65 B to 368 B per entry, a 42x amplification. State this in the commit. (Moot if P-INBOX
  deletes the appends first, which is the preferred order.)
- **Falsifier:** `elig["reroute"]` bit-identical tick-for-tick over 300 live wake ticks, action
  identical on 300 of 300. **If any of those move, it is a defect**, not progress.
- **Cannot distinguish:** bitwise invariance cannot distinguish "the payload change is correctly
  inert" from "the payload change is inert because nothing reads it". Only a named consumer can,
  and that is the condition for un-parking.
- **Approval:** none. Confined to `thalamus/` and `cortex/`; no brainstem safety module imports
  either workspace class.

#### GAP 2 -- instance sharing. RECLASSIFY as P2. RECOMMEND DESCOPE.

Filed as a P6 interconnect finding and it is not one. The 7 `GlobalWorkspace` sites are private
winner-take-all primitives *inside* algorithms (`rpm_solver` picks a candidate index;
`train_of_thought` sequences thoughts), not faculties failing to reach each other. Unifying them
buys P2 compliance (two mechanisms coexisting is a live violation) and **zero faculty coupling**.

Sharing *first* is actively worse than useless: `ReentrantWorkspace` holds ONE slot (`:57`) with
global lateral inhibition (`_G_INH`, `:81`). Routing 7 modules into `brain._reentrant` makes them
contend with the percept, self and salience coalitions at `brain.py:1746-1757`. And the second
subscriber's drain would be byte-identical to the one `brain.py:1623` already drains, so any read
it adds is dormant on arrival.

- **Tiers:** 5 of 8 sites are a one-line `step()` unwrap (`autonomous_loop.py:235,1168`;
  `executive_control.py:102,112`; `rpm_solver.py:305-308`; `spike_chat_lake.py:817-825`).
  2 modules are genuine behavioural ports: `cortex/access_consciousness.py:92-98` (the
  `subscribe(fn)` callback idiom `ReentrantWorkspace` cannot express) and
  `reasoning/train_of_thought.py:92,97,157-158,244` (depends on per-(source, content) decay;
  `ReentrantWorkspace` has only a scalar `_g_reentry`).
- **Falsifier, with the band the earlier version lacked:** measure the percept-coalition ignition
  count over the **same 7 seeds** used elsewhere in this program, report min and max, and set the
  criterion as "outside the measured 7-seed range". A bare "must stay at 996 of 1000" is either
  exact equality (which will fail for unrelated reasons) or it needs a band -- and a hand-picked
  band is itself an underived constant.
- **Cannot distinguish:** an ignition-rate criterion cannot distinguish single-slot contention
  from an unrelated seed effect. The 7-seed range is what supplies that separation.
- **Approval:** none.

### STAGE 7 -- GAP 7. Split three ways; needs owner approval on five items, not three.

#### 7a. Target pose -- already reachable, extend vocabulary only

`POST /api/chat -> Turns.pump -> Brain.respond ->` NL firewall `-> arm.apply()` is a working gated
motor route from the cockpit today. It is coarse (one `DEFAULT_STEP_DEG = 6.0`, no MOVETO). Extend
the firewall vocabulary; **do not build a second motor route**. `DEFAULT_STEP_DEG` survives the
constant audit -- `arm_config.py:151-160` argues it is a command magnitude and not a threshold, and
that argument continues to hold when the magnitude becomes user-supplied.

Gated on all of stage 0. Approval: none beyond stage 0.

#### 7b. Object placement -- runtime PROVED IMPOSSIBLE, construction-time RE-OPENED

Both cited facts hold: `so101/scene.xml` has exactly one non-arm geom (`floor`, line 22), and
`so101_arm.py:67-70` declines an OBSTACLE verdict deliberately ("a branch that cannot fire is the
computed-but-not-consumed defect this project already names").

**REJECTED: the blanket "PROVED IMPOSSIBLE".** The MjSpec / `self.data`-invalidation argument
applies to placing an object **at runtime into a live model**. It does not apply to placing one
**before compile** -- adding a `<geom>` or `<body>` to the scene at construction and building the
arm around it recompiles nothing and invalidates no handle. "A verdict that was deliberately
declined" is a design decision, not a structural bar; the comment's own reasoning ("a branch that
cannot fire") stops applying the moment a second geom exists.

**Correct verdict:** runtime placement is blocked by the handle argument; **construction-time
placement was never evaluated and is not proved impossible.** The standing rule is that "it cannot
be done" is valuable only when proved, and this one was asserted for the case that matters.
Evaluate it before recording anything.

#### 7c. Disturbance -- PROVED BLOCKED, by circularity

Not "needs measurement". The chain closes on itself:

1. `so101/scene.xml` has one non-arm geom, so the simulation has **no exafference source**.
2. `inject_disturbance` is the **sole** out-of-band writer of `qpos`, and `so101_arm.py:201-208`
   marks it TEST-ONLY and "deliberately NOT part of the command path".
3. Therefore a consumer built on `deviation` can only ever be exercised by the test hook.
4. Exposing the test hook is the approval-gated item. Return to 3.

**[measured]** `arm_config.py:28 TRAJ_DEV_MAX = 0.15` occurs **exactly once tree-wide, at its own
definition** -- 0 readers. The reflex it names was never implemented.

**REJECTED: wiring `TRAJ_DEV_MAX` as the envelope.** It is a unit error and a stale body.
`arm_config.py:28` documents it as "end-effector deviation (**m**)"; `so101_arm.py:438` computes
`deviation` as `norm(qpos[:nu] - _settled)`, a **joint-space norm in radians**. Metres against
radians. Its stated derivation ("tied to arm reach l1+l2=2m") is the **deleted 2-link arm**; this
body's reach is sampled at `arm_config._max_reach_mm()`, and `so101_arm.py:60-70` records that the
same 2 m circle already produced one wrong verdict. Either delete `TRAJ_DEV_MAX` (P2: two
mechanisms, one survives) or restate it in metres against `ee_mm()` with a derivation from
`_max_reach_mm()`. **Deleting is the honest option and is strictly constant-removing.** The derived
per-joint envelope already exists: `trajectory_breach()` compares against `_droop_tolerance()`.

**REJECTED: the "5 deg shove" falsifier.** Arbitrary. A **1.0 degree** shove already gives
`deviation = 0.017453 rad` against a `0.000713 rad` droop envelope -- 24x over. 5 degrees is ~122x
suprathreshold, so the test measures nothing about sensitivity. Use the derived stimulus: the
smallest per-joint shove exceeding `_droop_tolerance()[j]`, which is computed rather than chosen.

**Correction to a claim in the program:** "`deviation` and `trajectory_breach()` have zero live
consumers" is falsifiable by a grep. `trajectory_breach()` has one caller,
`cerebellum/spiking_pid.py:240`, immediately after `inject_disturbance` in the same loop iteration.
"No *production* consumer" is the right claim -- and it sharpens the `_settled` finding, because
that one reader works **only** because it reads before the next `apply()` re-baselines. The
signal's lifetime is one statement and the sole reader sits inside it.

**If the owner wants it anyway**, the honest design is a **latched** disturbance flag set by
whatever writes `qpos` out of band -- not a pose comparison. But that flag is only ever set by the
test hook, which returns to step 4.

#### Approval gates for GAP 7 -- FIVE items, not three

| # | item | why it needs the owner |
|---|---|---|
| 1 | Exposing `inject_disturbance` over HTTP | Contradicts a written standing decision (`so101_arm.py:201-208`, TEST-ONLY) |
| 2 | Sourcing `distance` from the scene | `brain.py:1192` is pinned to `SAFE_DISTANCE` on purpose and is the **only** E-STOP channel (`autonomous_loop.py:1260-1261`). Without it **no cockpit perturbation can ever trip the reflex**, so "the reflex preempts" holds only in the weak sense that the latch blocks commands |
| 3 | The latch-refusal helper | Changes a helper three self-checks use |
| 4 | **The HTTP-thread / `mjData` race** | `substrate/learn_daemon.py` documents in-code that chat and rendering are drained on the tick thread because "there is no lock anywhere in the brain stack" and rendering from the HTTP thread "does not raise, it returns garbage". `inject_disturbance` writes `data.qpos` and calls `mj_forward`; `mjData` is not thread-safe and `mj_forward` concurrent with `mj_step` on the same `mjData` is undefined behaviour, not merely wrong numbers. **Any route must enqueue from the HTTP thread and execute inside the tick loop, before `b.tick()` at `substrate/learn_daemon.py`** -- a thread constraint, not a sequencing preference |
| 5 | **`inject_disturbance` beats the E-STOP gate** | **[measured]** on a `So101Arm` with `stopped=True`: a gated `apply(JOINT_SHOULDER_PAN_ROTATE(+6deg))` moves the arm **0.000000 deg**; `inject_disturbance("shoulder_pan", 20.0)` moves it **20.000000 deg** (`deviation = 0.349066 rad`). The test hook is a **stronger actuator than the gated command path**, and it is the only code path in the repo that moves the arm while E-STOPPED. Exposing it is not "contradicting a standing decision", it is **adding the only E-STOP-bypassing actuator in the repo** |

- **Falsifier for 7c, if it proceeds:** after the consumer, a derived-magnitude shove must change
  strictly more than 0 of 200 actions. Today it changes 0 of 200 while the temperature band takes
  63 distinct values versus baseline 50 -- `autonomous_loop.py:969` verbatim. If that stays true,
  revert; do not tune the normaliser.
- **Negative control that must NOT move:** with the consumer live and no perturbation, the run must
  stay identical to baseline.
- **Falsifier for item 4:** 10,000 HTTP perturbation posts against a live daemon under MuJoCo
  validation; zero `mjData` corruption and zero garbage render frames.
- **Files, if it proceeds:** `infra/cockpit.py`, `learn_daemon.py`, `brainstem/so101_arm.py`,
  `cortex/brain.py`, `cortex/live_brain.py`, `infra/brain_telemetry.py`,
  `brainstem/arm_config.py`, `sensorimotor/nl_interface.py`, `tests/test_so101_arm.py`.

---

## 5. Cross-stage dependency that is easy to miss

**STAGE 5 and STAGE 7 are not independent.**

`autonomous_loop.py:1257-1275` -- `step()` returns at the reflex branch **before** `_pfc_bias` at
`:1366`. So a `maintain()` inserted inside `_pfc_bias` is not called on reflex ticks, and **the
working-memory drive does not decay while the arm is latched**.

Its severity is conditional on GAP 7. Today `distance` is pinned to `SAFE_DISTANCE`
(`brain.py:1192`), so reflex ticks essentially never fire live and the freeze is theoretical.
**If GAP 7 approval item 2 is ever granted and `distance` is sourced from the scene, the freeze
becomes real -- and it freezes goal state during exactly the emergency that should force
re-planning.**

**Explicit precondition: GAP 7 item 2 may not proceed until either the working-memory decay is made
tick-driven rather than call-driven, or STAGE 5 is confirmed EXEMPT.**

Confirmed and surviving: `_pfc_bias` has exactly one production call site (`:1366`), so the
per-call-versus-per-tick decay coupling does not exist today. The insert is safe on that axis.

---

## 6. Build order

1. **S0.1** word-boundary cue matching. Smallest diff; **[measured]** takes non-commands issuing
   arm commands from 16 of 16 to 5 of 16 on its own.
2. **S0.2** delete the bare `grip` cue. **S0.3** interrogative and negation guard. Together:
   0 of 12 and 0 of 16, with 16 of 16 legitimate commands retained.
3. **S0.4** literal E-STOP check at the top of `converse()`, chat branch unmoved.
4. **S0.5** repair or delete `run_local.py`; wire Check 4 into `run_selfchecks.py`. **S0.6** the
   P1 clause for safety vocabularies.
5. **P-INBOX** (deletion, own commit). **GAP 3c** (one line). **S2.3** (`except` narrowing, alone).
6. **GAP 9** (repair, 3-of-3 neuter harness).
7. **GAP 3a** with the `consciousness_probe.py:84` co-edit and a pre-registered scorer; open a
   separate P1 item for the `0.15` margin.
8. **STAGE 5** live-context A/B -- **only** with the `_stack.clear()` null and the first-divergence
   metric. Without both, do not run it.
9. **GAP 2** refiled under P2; recommend descope. **GAP 1** payload stays parked.
10. **GAP 7**: 7a proceeds on the existing firewall vocabulary; 7b re-opened at construction time;
    7c PROVED BLOCKED; five approval items.

---

## 7. Findings, not tasks

Items surfaced by this round that are conclusions rather than work.

**F-1. Four independent adjudications of the same wire disagreed, and none of them was measuring
the wire.** Reported action-change counts: 113 of 1400, 27 of 1400, 0 of 794, and a fourth pass
spanning 1 of 1400 to 268 of 1400 across six synthetic context streams. Each measured a property of
its own harness. This is the most valuable single finding of the round: **the disagreement was
never about the mechanism.** Any future adjudication of a live-loop wire must be driven by the live
loop's own context construction, with first-divergence rather than run-vs-run counting.

**F-2. A comma decides whether the arm stops.** See C-C. Not a gap, not on any list, live today.

**F-3. A question moves the arm.** `what is in your database?` rotates the shoulder pan and the
cockpit reports `Done: ...` -- correctly, because the command really was issued. Live today.

**F-4. The gate that names F-3 has been dead long enough for its import path to rot.**
`run_local.py` Check 4 asserts exactly the defect and cannot run; `run_selfchecks.py` never
referenced it. Silent degradation applied to a safety gate.

**F-5. `inject_disturbance` is a stronger actuator than the gated command path.** **[measured]**
20.000000 deg versus 0.000000 deg with `stopped=True`. The only E-STOP-bypassing path in the repo.

**F-6. The GAP 3a falsifier cannot distinguish two scorers that flip the consciousness marker.**
See C-D. This is what motivated the "cannot distinguish" line now required on every item.

**F-7. Six of the ten constant-audit findings share one shape:** the proposal removes a numeric
literal from the *text* while leaving the arbitration intact where the falsifier does not look.
"No new constant" meant a `2.0` already inside `reached`. "Ordering only" meant a 20-element
permutation. "The same remedy as `access_consciousness`" meant two multipliers over a knee that does
not exist here. "Add a precision term" meant choosing between F1 and Jaccard. "Within a second"
meant writing no factor at all.

**F-8. `arm_config.py` is the standard the rest of the repo should be held to.** `_joint_limits()`
read from the model, `_max_reach_mm()` sampled, `ground_free_band()` swept at 1-degree resolution,
`HOME_QPOS` the model's rest configuration, `_droop_tolerance()` = `qfrc_bias / actuator_gainprm`,
`kp[kp == 0.0] = 1.0` as a division guard. That is exactly why `TRAJ_DEV_MAX` stands out as the one
survivor of a deleted body.

**F-9. The GAP 3b conclusion is right and its stated evidence was not.** The "2-point support"
came from a literal `drive=0.4` in the module's own demo -- a property of a self-check fixture. Use
the structural argument in `spike_chat_lake.py:1184-1187` instead. A citation defect in a document
whose entire value is auditable proofs.

**F-10. `top_down_bias` never reads `frame[1]`, and it does not matter.** The pop is the channel,
not the magnitude. A "permanent structural blocker" that blocks nothing.

---

## 8. Why the gate does not move, and what "all of them" would actually mean

The section-1 falsifier predicts its own outcome, so it is argued rather than asserted.

`tools/p6_gate.py` counts faculty-to-faculty **reads**. Nothing in this program creates one:

- **GAP 1** adds the **submitter** axis. F1 kills wires on the **listener** axis, which `_deliver`
  cannot express -- its loop variable is named `_name` because the body never uses it, and
  `s["bias"]` is written identically for every subscriber. `broadcast_to("action") ==
  broadcast_to("language")` on every tick, before and after. The 12 killed wires stay killed.
- **GAP 2** would create shared instances, but the second subscriber's drain is byte-identical to
  the one `brain.py:1623` already drains. Any read it adds is dormant on arrival.
- **GAP 4, 5, 6**, if they survive stage 5, terminate at the single action gate. That is a P3 claim
  (a dormant mechanism made decisive), not a P6 edge.
- **GAP 3, 8, 9** and every stage-0 item touch no workspace topology at all.

### Is "all of them" achievable? No, and the reason is not effort

Counting by honest outcome rather than by ticked box:

- **6 items close and change something real:** S0.1, S0.2, S0.3, S0.4 (safety), S0.5 (a gate that
  can run), GAP 9 (a gate that can fail), plus GAP 3c and P-INBOX as constant-removing hygiene.
  **Of these, only the stage-0 items change a behaviour a user could observe.** GAP 9 and P-INBOX
  change what a gate can catch and what memory does over 30 days.
- **2 are completed proofs:** GAP 8 and GAP 3b. Both are real proofs -- a structural argument plus
  a measurement. Re-opening either is p-hacking.
- **1 is a proof this round produced:** GAP 7c, blocked by circularity rather than by difficulty.
- **1 was falsely proved and is re-opened:** GAP 7b at construction time.
- **3 (GAP 4, 5, 6) are genuinely unmeasured**, and both prior verdicts were harness artifacts.
- **2 (GAP 1, GAP 2) are chained**, and GAP 2 should be dropped from the P6 program and refiled
  under P2 -- it buys compliance and zero coupling.
- **1 (GAP 3a) closes as an instrument-sensitivity improvement**, not as the P1 item filed against
  it, and it cannot ship without a pre-registered scorer choice that may flip a consciousness
  marker to ABSENT.
- **1 (GAP 7) is gated on authority nobody in the engineering loop holds**, and its most-wanted
  property -- a user-placed hazard tripping the reflex -- is **not deliverable** without changing
  the one channel the safety envelope is built on.

So "all nine" is not achievable, and the blocking reason is structural in every case where it
blocks: an absent listener axis, an absent exafference source, an absent scene geometry, an absent
grounding signal, an owner decision. **Two of the nine are already finished as proofs, and treating
a proof as an unclosed gap is the single easiest way to make this program look less complete than
it is.**

The most valuable line in this document is not a gap. It is that **`live_brain.py:405` swallows
8 of 15 emergency stops, a comma decides which, and `what is in your database?` rotates the
shoulder pan.** That ships first, ahead of everything else.
