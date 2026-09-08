# Phase 2 exit — learning in the substrate: what landed, what was deleted, and what the phase actually taught (2026-09-07)

*The plan of record, Phase 2: "**Exit:** `evidence/l-exit.md`."* No structure was prescribed, so this
file states the disposition of every row, then the two findings that outlived the rows. Every number
is copied from the task record or commit it names — re-derive from those, never from this file.

**The phase's headline, and it is not the one it was scoped for: Phase 2 REMOVED more mechanism than
it landed.** Three of its build rows were built, measured and removed. Two (L1b, L2) fell to clean
pre-registered nulls — P8 clause 5 working as designed, "the licence working, not the licence
failing". The third (T069) was **refused on measurement rather than nulled**: its deciding contrast
was underpowered, a ratified control was never run, and its apparatus never entered version control,
so it does not carry the same weight and this file does not give it any (§1, and
`evidence/t069-cerebellar-rule.md` §4).

## 1. Every row, with its disposition

| row | disposition | where |
|---|---|---|
| **L1a** the admission window | **LANDED** 2026-09-05 `3979001` — one marginal-covering split for all three grower windows; landed on the oracle's reading, the session's call | `evidence/l-parallel-arms.md` |
| **L1b** synapses learn online | **BUILT → DELETED ON ITS NULL** 2026-09-05 — Oja and BCM on the grown units' input rows both lost to the frozen rows (t −1.96 / −1.36); Oja FELL past the bar, a measured harm | `evidence/l1b-synaptic-online.md` |
| **L1c** the readout learns online | **LANDED** 2026-09-06 — the delta rule on `NGSNN.beta`, +0.143 live over the stale incumbent (t 4.13); CHARGED in the cells, DORMANT on the deployed brain | `evidence/l1c-online-readout.md` |
| **L2** grow on plateau | **BUILT → DELETED ON ITS NULL** 2026-09-06 `5f602e3` (the deletion; `a3117b2` is its battery record) — moved its own outcome by +0.625 units/seed at t 0.141, and neutering the readout did not shift the growth rate (t 1.135) | `evidence/l2-grow-on-plateau.md` |
| **L3** the admission instrument | **LANDED** 2026-09-05 `308f6db` — the constant percept admits exactly 1 unit on 8 of 8 (was 13–51) | plan row |
| **L4** persist what makes a relaunch resume | **HALF BUILT, HALF RE-OPENED** 2026-09-06 — the append wire stands; the persistence half re-opens on one leg (17 of 52 moving `state()` keys identical in a second life) | `evidence/l4-resume-and-the-append-ceiling.md` |
| **L4b** the append ceiling | **DONE** 2026-09-06 — the ceiling was NON-STATIONARY, not stale; the bar is now the tag's own recent rate | `evidence/l4b-append-ceiling-nonstationary.md` |
| **L4c** the ignition asymmetry | **DONE** 2026-09-06 — ignition is COALITION SIZE, not the relaunch; retracts L4 §8.2 | `evidence/l4c-ignition-is-coalition-size.md` |
| **L5** the IMMORTAL class | **RATIFIED + GATE BUILT** 2026-09-06 — `substrate/immortal_retention.py` reads `lake_engine.RETENTION` rather than declaring a second class | plan row |
| **L6** immortal vs the sleep tick | **LANDED** 2026-09-06 — the cap had NEVER bound (26,037 learned facts against 250,000), so the fix costs nothing observable today | `evidence/l6-immortal-vs-consolidation.md` |
| **T069** the cerebellar rule on the built tissue | **BUILT → REMOVED, REFUSED ON MEASUREMENT** 2026-09-07 — the shuffled signal learned as well as the graded one (t 1.242), but that contrast carries ~29 % power at S=8, the ratified Bouvier control was never run, and the module never entered version control so no number is re-derivable. Removed, but NOT a clean null | `evidence/t069-cerebellar-rule.md` §4 |
| **SC-011** a stream with learnable temporal structure | **NEVER RAN — BLOCKED** | Phase 1 delivered no stream with measured succession, and Part C forbids starting a growth task before its stimulus is shown to carry the structure its red needs |
| **SC-012a** does width buy accuracy | **FIRED** 2026-09-07, on its SECOND reading — the first was retracted when `acc(1)` proved an artefact of the scoring path | `evidence/sc-012a-width.md` |

**Count: 7 landed, 3 built-and-removed on their own measurements, 1 half-open, 1 blocked, 1 criterion fired = 13 rows.**
*(Corrected by the arc's panel: the first form said "6 landed" while enumerating seven, and its
total came to 12 against a 13-row table. The seven are L1a, L1c, L3, L4b, L4c, L5, L6.)*

## 2. What the phase established about learning in this substrate

**Online learning works at the READOUT and not at the input rows.** L1c's delta rule on the grown
units' readout is the phase's one surviving learning mechanism (+0.143 live, t 4.13). L1b's two
candidate input-weight rules both lost to frozen rows, and the mechanism it identified — the input
rows moving under a readout the closed growth gate holds stale — is why L1c had to precede any
second attempt. That ordering was a finding, not a plan.

**Width buys accuracy, but not more than the brain already takes.** SC-012a fired: acc(16) − acc(1)
clears (t 2.596) and acc(16) clears its majority floor (t 4.011). But no single doubling clears,
8→16 is **−0.013**, and forcing width to 9–14 units does NOT beat the incumbent's own 2–7
(t −0.141). Nothing reaches the persistence floor at any width. The lever exists and is nearly
exhausted where the brain already sits.

**Nothing in this phase reached the persistence floor.** L1b, L1c and SC-012a all report accuracies
below "answer the previous category" on the live stream. That is the phase's most durable negative
and it bounds what Phase 3's width can be expected to buy.

## 3. The finding that outlived the rows: the apparatus was the story

Across the phase's arcs the worst defect was, repeatedly, in the measuring apparatus rather than in
the tree being measured. The phase's deletions are only trustworthy because the instruments were
attacked as hard as the mechanisms:

- A **shared scoring path answered a class the stream never emits** — an all-zero rate row's argmax
  resolved to column 0, deciding 45 of 60 held-out rows on one seed. It invalidated SC-012a's first
  verdict, which was RETRACTED the same day it was produced, and it moved a statistic RECORDED IN L1b's
  evidence file from "does not clear" to "clears" once fixed. L1b itself landed nothing: the moved
  number is the INCUMBENT's, measured during L1b's arc, and the correction does not resurrect the
  deleted rule.
- A **pre-registered same-rows guard was unpassable by construction** (it demanded identical floors
  across a null arm whose whole purpose is to move them), and would have reported no verdict at all.
- An **exogenous cap was inert** — `l_max` never bound because the pursuit's RMSE target stopped it
  first, so three of a five-point grid's arms were bit-identical.
- A **fix BLINDED an existing red**: after the scoring path stopped answering an impossible class, a
  dead unit answered the prior and landed at the majority floor, where L-R4's aliveness pin could no
  longer tell it from a live one.
- T069's **red was unreachable while wearing the name of the fix** — the rule pinned to the tissue's
  window, but the relay hops that create the latency difference sit outside the tissue.

**The transferable rule this phase paid for:** a null is only as good as the instrument that produced
it, and an instrument is only trustworthy once it has been shown to be able to produce the other
answer. L1b's and L2's removals each rest on a red that was constructed and seen to fire. T069's does
not, which is exactly why it is recorded as a refusal rather than a null — and its own arc adds the
rule it broke: **commit the mechanism BEFORE deleting it, so a null stays re-derivable.** L1b, W1 and
T044 all did; T069 did not, and its numbers are gone with it.

## 4. Open at the exit, each measured and named

- **SC-011 has no stimulus.** It needs a stream with measured temporal succession; Phase 1 measured
  the tracked packs to carry none. Blocked, not skipped.
- **Part C's reporting rule may now lift** — "until SC-012a has fired" is satisfied. Deliberately not
  taken in the firing arc: it is a separate docs commit, it edits
  `tests/test_units_admitted_wording.py`, and the reading is qualified by a control that fails at
  N=16. Owner-visible.
- **SC-012a's conjunct (c) has NO REACHABLE RED** on the fixed instrument — the veto could not have
  vetoed, so the fire rests on (a) and (b). Replacing a ratified conjunct is a spec change whose
  replacement null owes its gate first.
- **L4's persistence half** stands on one leg, with a committed probe and no red currently firing.
- **L1c is DORMANT on the deployed brain** — one silent unit, charge 0.0 over 929 percepts — until a
  unit fires on the live stream.
- **Ignition is structurally unreachable on a matured brain** (L4c), handed to the C-rows.
- **`spec.md:69` (US3 AC-3) stays RATIFIED and UNDISCHARGED.** It requires the Bouvier GPE/G_E
  control pair; T069 substituted its own green/red pair without disclosing the substitution at the
  time, and the Bouvier pair is never defined anywhere in this repo.

## 5. What Phase 3 inherits

A substrate whose readout learns online, whose admission instrument is honest (L3), whose lake has a
declared immortal class with a gate (L5/L6) and a non-stationary append bar (L4b) — and a measured
ceiling: width past ~8 units buys nothing on today's stream, and no arm at any width beats
persistence. Phase 3's constructed width (T065) is therefore capacity ahead of consumption, exactly
as the plan's own O5(f) disclosure says, and this exit does not soften that.
