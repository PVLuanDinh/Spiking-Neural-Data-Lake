# The scoring path answers a class the stream never emits — the fix, and what it moves (2026-09-07)

**Opened by SC-012a's own panel** (`evidence/sc-012a-width.md` §6), which retracted that arc's verdict.
This row is the arc §6.4 named: *"a scoring path that cannot answer a class absent from the training
targets ... is its own arc with its own red, not a fix folded into this one."*

## 1. The defect, re-derived here (not quoted from §6)

`NGSNN._rates` returns `cnt / T` — spike counts over T — and is therefore **non-negative**. A held-out row
on which no fitted unit fires has an exactly-zero rate vector, so its score vector `S @ beta` is exactly
zero and `argmax` resolves the tie to **column 0**. `fit` sets `self.m = int(y.max()) + 1`, so a column 0
exists; but the engine buffers `percept_{t-1} -> category_t`, so the targets are the categories that FOLLOW
one another and **class 0 need never appear among them** (measured train classes: 1..5, 1..4, 1..3).

The result is not a weak prediction. It is a prediction of a class that **cannot be correct**, made by
tensor layout rather than by any decision rule, and recorded as an accuracy.

**Two callers make the same decision, and that is a P2 defect in its own right:**
`hippocampus/ng_snn.py::NGSNN.predict` (the open-gate path) and an **inline copy** of the argmax in
`cortex/neurogenesis_engine.py::_refit`'s closed-gate branch. Two mechanisms for one job. The fix leaves
ONE.

## 2. Pre-registration (written before the change)

### 2.1 The rule

A score vector that is exactly zero carries **no evidence**. The instrument answers the **training-majority
class** on such a row — the prior — and **counts** it.

**Why the prior and not an abstention-with-exclusion.** Excluding unanswered rows from the denominator
would move seed 0's CAP1 accuracy 0.233 -> 0.933 (measured in `evidence/sc-012a-width.md` §6.2): an
instrument change that manufactures a four-fold "improvement" is exactly the shape this project refuses.
Answering the prior keeps the denominator fixed and makes an uninformative readout score **at its own
majority floor**, which is the comparison every accuracy in this tree is already read against. A readout
that learned nothing then reads as "learned nothing" instead of "worse than chance".

### 2.2 ONE mechanism, two callers (P2/P4 shape)

`NGSNN.decide(S, beta)` becomes the single decision rule. `NGSNN.predict` calls it; the engine's
closed-gate branch calls it instead of its inline argmax. The inline copy is DELETED, not left beside it.

### 2.3 The reachable RED (P8 clause 1)

Construct a fit whose held-out rows produce all-zero rates, with a training set whose classes exclude 0.
- **Pre-fix behaviour must be reconstructible**: the bare argmax answers 0 on every such row.
- **Post-fix**: every such row answers the training-majority class, and the abstention count equals the
  number of all-zero rows.
- If no input can make the two differ, the fix changes nothing and is **deleted, not weakened**.

### 2.4 The control that is not the method (P8 clause 2)

**Rows where at least one unit fires must be BIT-IDENTICAL before and after.** A correct instrument fix
moves ZERO of the predictions it was not built to change (the standing lesson: an instrument fix that moves
a number it should not have touched is a second defect wearing a repair's clothes). Asserted as an exact
tensor equality on a fit with no silent rows, in the same command as the red.

### 2.5 Attribution and disclosure (P8 clauses 3 and 4)

`last_abstentions_` is recorded per fit and surfaced, so the fraction of any reported accuracy that came
from the fallback is a printed number rather than an assumption. **This CHANGES measured accuracies
anywhere units are silent** — the L-row records are re-read against it in section 4 and any moved number is
reported, not smoothed.

### 2.6 Disposition on a null

If the red cannot be constructed, or the control shows the fix moves firing rows, the change is REVERTED
and the defect is recorded as unfixable by this route.

## 3. The change

`NGSNN.decide(S, beta=None)` is the ONE decision rule. `NGSNN.predict` delegates to it, and the engine's
closed-gate branch calls it instead of the inline argmax it used to carry — **two mechanisms became one**,
which is the P2 half of this row and was a defect in its own right.

The rule: `argmax(S @ beta)`, except that a row whose **RATE** vector is exactly zero answers
`prior_class_` — the training-majority class, recorded at `fit` from the targets actually present rather
than from the width of the one-hot (`self.m` spans `0..max(y)` and can carry columns for classes the fit
never saw; that width is the defect, so it may not also be its fix).

**The test is read on the RATE row, not on the score.** A zero SCORE can also arise from a row that fired
into a zero beta column, and that IS an answer the readout made; overriding it would be this fix reaching
past its own defect.

**An unfitted model invents no prior.** `prior_class_` is `None` until a fit runs, and `decide` then falls
back to the bare argmax rather than manufacturing a class. Absence is a fact.

### 3.1 The red and the control, in one command (`tests/test_ng_snn_decide.py`, 5 pins)

- **RED, reconstructed rather than described:** the retired bare argmax is rebuilt in the test file and
  asserted to answer **0 on every silent row** — because a gate only ever seen green is indistinguishable
  from one that cannot see. `decide` answers the prior instead, and the two are asserted to DIFFER.
- **CONTROL:** on rows where any unit fires, `decide` is **bit-identical** to the retired rule
  (`torch.equal`), including a MIXED matrix where only the silent rows may move.
- **P2:** the engine source is read and asserted to contain `m.decide(S_te, beta_inst)` and NOT the retired
  `@ beta_inst).argmax(` — a duplicate is a property of the tree, not of a run.

### 3.2 A defect this arc's own control caught

The first form of this change initialised `prior_class_` inside `prune_unearned` rather than `__init__` —
so an unfitted model had no attribute at all, and every prune would have **reset the prior**. The unfitted
pin failed immediately and named it. Recorded because the fix for an instrument defect is itself an
instrument, and it got one wrong on the first pass.

## 4. What the fix MOVES — measured, not asserted (the section 2.5 disclosure)

Live engine, 800 ticks, same seeds and same cells as `evidence/sc-012a-width.md`:

| seed | CAP1 before | CAP1 **after** | its majority floor | CAP16 before | CAP16 after |
|---|---|---|---|---|---|
| 0 | 0.2111 | **0.9528** | 0.7444 | 0.8278 | 0.8361 |
| 1 | 0.0000 | **0.7389** | 0.7389 | 0.9583 | 0.9583 |
| 2 | 0.6944 | 0.6944 | 0.6944 | 0.9472 | 0.9472 |
| 3 | 0.2889 | **0.4000** | 0.3194 | 0.7722 | 0.7722 |

Two things this table says, and both were pre-registered:

1. **The degenerate arm moves to its prior and stops.** CAP1 now sits AT or above its own majority floor on
   every seed (exactly at it on seeds 1 and 2). An uninformative readout reads as uninformative instead of
   as worse-than-chance. It does **not** jump to the 0.933 that dropping the rows from the denominator
   would have manufactured — that was the alternative section 2.1 refused, and the refusal holds.
2. **The real arm barely moves.** CAP16 changes on ONE of four seeds, by +0.008; at 11-14 units rows are
   almost never silent. That is the system-level form of the control: the fix reaches the arm it was built
   for and leaves the other alone.

**The consequence for SC-012a, stated because it is large and it is the point of this arc:** conjunct (a),
`acc(16) − acc(1)`, was **+0.465 (t 5.024)** on the contaminated instrument. On these four seeds it is
about **+0.18 and NEGATIVE at seed 0**. Most of the pre-fix effect was the artefact. SC-012a is now
readable for the first time, and re-reading it is the next commit in this arc — not this one, because a
change to the instrument and a verdict measured with it are two changes.

## 5. THE PANEL'S FOLD (2026-09-07) — five findings, two of them BLOCKERs on this arc's own claims

The arc was panelled at `924d706` on three lenses (25 agents, 15 findings surviving adversarial
verification, **0 changing the arc's conclusion**). What follows is the fold. The pattern held for a
fifth day: **every one of the serious findings was a defect in this arc's own apparatus**, not in the
tree it audited.

### 5.1 BLOCKER — the P2 claim was FALSE: there were THREE decision sites, not two

Section 1 and the fix commit both state that `NGSNN.predict` and the engine's closed-gate inline
argmax "were two mechanisms for one job". A **third** survived, on the LIVE path:
`neurogenesis_engine.py`'s per-tick instrument inside `observe()` carried its own
`int((S @ m.beta).argmax(dim=1)[0])`, feeding `live_correct` -> `live_accuracy`. Its trailing comment
`# == m.predict(xp), one rollout` was TRUE before the fix and FALSE the moment `predict` began
routing through `decide` — **a comment that outlived its body, on the very line that needed to move.**

It now calls `m.decide(S)`. The consolidation the commit claimed is true only with this line
included; it is now.

**Why the SC-012a verdict is untouched by it, traced rather than asserted:** the criterion's three
conjuncts read only `acc`/`maj`, which come from `trajectory`; `trajectory` is appended at exactly two
sites and BOTH were already on the fixed path. `live_acc` is a parallel reporting statistic read only
by L1c-R1, and the contaminated `pred` dead-ended in `live_correct` — the online readout's `learn`
never receives it, so it could not feed back into beta, the substrate, or any refit.

### 5.2 BLOCKER — the fix did not survive persistence, on the one path that matters most

`prior_class_` was set only in `fit`. `restore_growth_state` rebuilds the model by assigning tensors
and never fits, so **a restored brain carried `prior_class_ = None` and `decide` silently reverted to
the bare argmax** — the exact defect this arc retired, returning on every persisted construction, i.e.
every daemon relaunch. It now rides in the blob as an OPTIONAL array (the additive contract, no format
bump), with `-1` meaning "no prior recorded" so an old blob restores as None rather than as class 0.

Pinned by `test_the_prior_survives_the_engine_persistence_round_trip`, whose control is an old blob
with the key stripped: it must restore as None and COUNT its unanswerable rows, so the red stays
reachable. **Every SC-012a cell ran `persist=False`, so no reported number was scored with the
reverted rule** — verified, not assumed.

### 5.3 The disclosure counter read healthiest exactly where the defect was live

`last_abstentions_` was set from the silent-row count BEFORE the guard that decides whether the prior
is applied at all. On a model with no prior — an unfitted one, or (until 5.2) a RESTORED one — it
reported N abstentions while `decide` had in fact answered column 0 N times.

It now counts only rows ANSWERED FROM THE PRIOR, and a second counter, `last_unanswerable_`, carries
the rows it could not answer even from the prior — the retired defect's own counter, so that case can
never hide inside the abstention number.

**It was also DORMANT (P3):** section 2.5 pre-registered that the fallback fraction would be
"a printed number rather than an assumption", and the first form of the fix never built it — zero
consumers outside its own tests. `state()` now surfaces `decide_prior_class`, `decide_abstentions`,
`decide_unanswerable` and `decide_heldout_n`, recorded PER REFIT (`_record_decide`) rather than per
call, because the live instrument overwrites the model attribute every wake tick.

### 5.4 The arc's own P2 pin could only see the site it had already fixed

The pin grepped one literal, `"@ beta_inst).argmax("` — the LOCAL VARIABLE NAME of the branch it had
just migrated. The surviving third copy binds `m.beta`, so the assertion was **structurally incapable
of firing on it**, and it passed with the survivor in place. A guard that can only see the fixed site
is not a guard. It is now a census — any argmax over a beta in that file — and it is anchored on
`__file__` rather than the cwd, because the old form raised `FileNotFoundError` from any other
directory, erroring instead of failing.

### 5.5 The fix BLINDED an existing red, and that red had to be rebuilt

L-R4's aliveness pin asserted `|acc - majority| <= 1e-3` on the constant-percept arm. Before this arc
a DEAD unit was silent, answered class 0, scored ~0.0, and sat far from the floor — so the gap
discriminated. After `decide`, **a dead unit answers the prior and lands AT the majority floor too**:
the assertion now passes for alive and dead alike.

The repair stops routing aliveness through accuracy. A live unit answers some held-out rows itself; a
dead one abstains on every one. The pin now reads `abstentions >= heldout_n` as death, asserts
`unanswerable == 0` (a silent row answered by tensor layout is the retired defect firing), and keeps
the floor relation only as a loose companion. **This is the shape worth remembering: a fix that makes
a system better-behaved can erase the very contrast an existing red depended on.**

### 5.6 The L-row re-read section 2.5 pre-registered and section 4 did not do

Section 2.5 promised "the L-row records are re-read against it ... and any moved number is reported,
not smoothed". Section 4 re-read only CAP1 and CAP16. Done now, against
`l1b-synaptic-online.md`'s recorded incumbent (same configuration; the refit accuracy is the batch
instrument on both, which is what makes them comparable):

| | recorded (retired path) | re-read (fixed path) |
|---|---|---|
| units | [2, 2, 2, 7, 6, 3, 3, 2] | **identical** |
| majority floors | 8 values | **identical** — the control: floors are label bookkeeping and the fix must not move them |
| acc | [0.3111, 0.9167, 0.825, 0.6556, 0.6278, 0.8333, 0.8667, 0.9472] | moves on **3 of 8** seeds: s0 +0.478, s5 +0.056, s6 +0.011 |
| **(acc − majority)** | t 1.487, p 0.0898, 7/1/0 — **does NOT clear** | **t 4.799, p 0.0039, 8/0/0 — CLEARS** |
| (acc − persistence) | t −2.898 — does not clear | t −3.791 — **still does not clear** |

**A landed record's headline statistic FLIPS.** `l1b-synaptic-online.md`'s incumbent was recorded as
failing to clear its own majority floor; on the fixed instrument it clears it. Corrected in that file.
It does **not** resurrect L1b: L1b was deleted because its rule lost to the frozen rows, a different
comparison, and nothing here touches that. Nothing reaches persistence on either instrument.

### 5.7 What the panel did NOT overturn

The lenses were pointed at the most self-serving shape available — an instrument change that flips a
verdict toward firing. Three separate agents attacked it and none found a mechanism by which
answering the prior on a silent row advantages the wide arm. The structural argument stands: on a
silent row the retired rule scored **exactly 0** (class 0 is never a target) and the fix scores the
majority rate, so `acc(1)` can only rise, `acc(16)` rises far less (it is rarely silent), and
`acc(16) − acc(1)` can only SHRINK. Measured: +0.316 against +0.002. The fix made firing harder by
construction, not by luck.
