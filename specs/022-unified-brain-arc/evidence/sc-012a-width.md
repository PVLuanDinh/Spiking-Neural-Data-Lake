# SC-012a — is width worth anything on this stream? (2026-09-07)

The plan row, quoted (`PLAN-FINAL-2026-09-02.md` Phase 2; `spec.md:183`, ratified under OD-5):

> **SC-012a (new test, same file):** admission CAPPED exogenously at N ∈ {1, 2, 4, 8, 16}, S = 8, held-out
> next-category accuracy under L1a's window; paired t of acc(16) − acc(1) > 1.895 AND acc(16) above its own
> majority floor at the bar; null: the same grid with labels shuffled marginal-preserving must not rise.
> Reported beside it, not a bar: the marginal gain per doubling, acc(2N) − acc(N) at each N (the efficiency
> signal Part C names).

Until this criterion fires, owner-facing surfaces report a unit count as units **admitted**, never as growth
(Part C; gate `tests/test_units_admitted_wording.py`).

**Why now** (the session's ordering call under the 2026-09-05 authorization): L1b and L1c were both refused on
a 2–7-unit live substrate and each pointed at WIDTH as the binding constraint. SC-012a is the plan's own
measurement of whether width buys next-category accuracy on this stream at all — the fact L2's
growth-on-plateau needs before it is rebuilt. **A measurement row: no mechanism lands; the criterion fires or
it does not.**

**Provenance of this file.** A complete draft of this arc (harness, statistics, in-tree test patch, and
sections 1–2) was written 2026-09-06 and archived at `memory/archive/sc012a-2026-09-06/`. It was **never
run**. Section 1 below RE-VERIFIES its territory against the current head, and section 2.5 records an
amendment that the re-verification forced. The draft harness is superseded, not deleted: its reading is
reported in section 3.1 because it is the reading this arc would have published.

## 1. Territory — re-verified at `2fddea8` (clean tree), not inherited from the draft

The draft's section 1 was written at the L1c refusal head and asserted the incumbent path "byte-identical to
`ea69088`". Seven arcs landed after that (L1c LANDED, L2 built-measured-deleted, L4/L4b/L4c). Every claim was
re-checked against the tree; the map does not outrank the territory.

| claim | status at `2fddea8` |
|---|---|
| `NGSNN.fit` skips its held-out split when `l_max` is set | **CONFIRMED** — `ng_snn.py`: `n_val = 0 if self.l_max is not None else max(8, int(0.2 * N))`; the earn-your-place stop is guarded "Only when no preset `l_max` was given" |
| the engine passes `_l_max` to every refit's `NGSNN` | **CONFIRMED** — two construction sites, both `l_max=self._l_max` |
| `prune_unearned` still runs under a cap | **CONFIRMED** — `_refit` calls it after `fit` on every open-gate refit |
| `observe(features, category)` is the single feed point | **CONFIRMED** — exactly one live caller, `cortex/brain.py` `self.neuro.observe(res["percept"], int(res["unit"]))`, with `_neuro_feed_refusals` counted beside it. The monkeypatch therefore intercepts the whole stream |
| `trajectory` rows are `(ticks, n_units, acc)`; `baselines` are `(ticks, majority, persistence)` | **CONFIRMED** — the stats script's `r[1]`/`r[2]` indices are right |
| the measured accuracy is the BATCH instrument, not L1c's online readout | **CONFIRMED, and it is the claim most at risk from L1c.** Both append sites use batch weights: the closed-gate branch scores with `self._batch_beta`, the open-gate branch with the fresh fit's `model.predict`. L1c's online readout mutates `m.beta`, which feeds `live_accuracy` **only**. SC-012a's statistic is uncontaminated; `live_*` is reported, never barred |
| **the growth gate is live in the cells** | **CHANGED — the draft never mentioned it.** `make_brain` wires `growth_gate=_growth_gate_verdict` and `online_readout="delta"`. Under a CLOSED gate `_refit` returns early after scoring the retained model and does **not** refit, so a cap gets no chance to bind on that tick. Measured: 20 refit rows on every 800-tick cell, so the gate did not suppress the grid — but `trajectory` cannot distinguish an open-gate refit from a closed-gate score row, so the gate state is now recorded directly (section 2.4) |
| the uncapped incumbent stops at 2–7 units | **CONFIRMED** and re-measured here: 2, 3, 4, 5 units at seeds 0–3 under a cap of 16 |

## 2. Pre-registration (written before any criterion arm ran; section 2.5 written before any arm ran too)

### 2.1 The cells — 11 arms × 8 seeds × 800 ticks, 4 at a time, two stages

Stage A (REAL): `CAP1, CAP2, CAP4, CAP8, CAP16` with `eng._l_max = N` **and** `eng._eps_tol = 0.0`
(section 2.5). Stage B (SHUF): the same five caps, fed a `random.Random(seed + 500)` permutation of the
seed's recorded `CAP1_REAL` category sequence — the marginal exactly preserved, the percept→category pairing
destroyed. Beside both, `INCUMBENT`: neither `_l_max` nor `_eps_tol` touched, the configuration the live
brain actually runs, so the regime gap section 2.5 discloses is reported rather than implied.

Every cell records units at the consumer **and** at the source, `units_max`, `cap_bound`, the refit accuracy
and floors (last third), the live instrument, the growth-gate state, the category sequence, W/beta hashes,
`eps_tol_used`/`l_max_used` read back off the engine, head and timings. torch threads pinned to 1.

### 2.2 The criterion — each conjunct quoted, marked MET / NOT MET one by one

Statistic (the plan's): S = 8, paired one-sided t > 1.895 AND exact sign-flip p ≤ 0.05, ties at |d| ≤ 1e-6,
DEGENERATE below 5 non-tied seeds; the random-sign false-positive rate printed beside every verdict.

- **(a)** "paired t of acc(16) − acc(1) > 1.895" — paired over seeds, last-third refit accuracy.
- **(b)** "AND acc(16) above its own majority floor at the bar" — paired (acc(16) − maj(16)) clears.
- **(c)** "the same grid with labels shuffled marginal-preserving must not rise" — paired
  (acc_SHUF(16) − acc_SHUF(1)) does NOT clear. If it clears, the rise is the instrument's and SC-012a does
  not fire whatever (a) and (b) read.
- **Reported, not barred:** the marginal gain per doubling acc(2N) − acc(N) at N ∈ {1,2,4,8}; effective units
  per cap; acc(N) − persistence; the live instrument; the INCUMBENT arm.

**No conjunct is relabelled.** All three are barred; (a) and (b) are the PASS conditions and (c) is a veto.
Anything else in this file is reported, and is said to be reported.

### 2.3 What fires, and what follows (pre-registered)

**SC-012a FIRES iff (a) AND (b) are MET AND (c) holds.**

- **If it fires:** the Part C reporting rule lifts — owner-facing surfaces may say "growth" once the headline
  is re-worded under `tests/test_units_admitted_wording.py` (a separate docs commit, after the arc); L2's
  growth-on-plateau gains a measured reason to exist.
- **If it does not fire:** the reporting rule STAYS; L2 is re-scoped to its attribution red on the existing
  gate (no new admission trigger is built) and the width lever moves to Phase 3's constructed width (T065).

**The not-fired branch's causal sentence is BOUNDED BY THE WIDTH ACTUALLY REALIZED, not by the cap's
label.** An earlier form of this section said "buys no next-category accuracy at any width up to 16". That
asserts more than the instrument earns: the criterion caps admission at 16, but the pursuit saturates below
the cap on most seeds and `prune_unearned` cuts further, so the realized width is what the sentence may
speak about. The disposition is therefore written against MEASURED effective units (section 3), and the
control that licenses it — the cap having actually varied width — is reported per cap rather than assumed.
Without that bound this is the 2026-09-05 shape: a P8-clause-5 deletion executed on a null the instrument
was never required to have earned.

**The null disposition is itself a causal claim, and it owes a positive control** (the 2026-09-05 lesson).
The claim behind "it did not fire ⇒ width buys nothing on this stream" is that the cap actually varied the
width. That control runs BEFORE the verdict is read, in section 2.4 — and on the draft harness it FAILED.

### 2.4 Instrument controls — run and read before any conjunct

**What these controls DO and DO NOT decide** (fixed 2026-09-07 while the shuffled stage was still
running, so conjunct (c) was unknown when the rule was written). The criterion is the plan's three
conjuncts — (a) AND (b) AND NOT (c) — and nothing else. The controls below are **instrument-validity**
checks: they decide whether that reading can be TRUSTED, and they QUALIFY it in the verdict. They do not
veto a ratified criterion. Letting controls of the session's own invention gate a ratified criterion would
put the verdict in the hands of whichever controls the session chose to write. A failed control is
reported beside the verdict, named, with what it does and does not undermine.

1. **The cap binds.** `cap_bound` (fit maximum ≥ N) must hold on a majority of seeds for N ≥ 4. **This is the
   control the draft harness failed** (section 3.1).
2. **Reach, downward.** `CAP1` must read exactly 1 unit on 8 of 8 — the cap can bind at all.
3. **Distinctness.** `CAP4`, `CAP8`, `CAP16` must not be bit-identical at a seed (W hash). Identical arms make
   acc(16) − acc(1) a two-point comparison wearing a five-point grid's name.
4. **Two counters that fail differently** (obs #2106). Effective width is read at the CONSUMER
   (`eng.grown_units`, post-prune) and at the SOURCE (the last refit row's own unit column), in one execution.
   A cell whose counters disagree is reported, never averaged.
5. **Same rows.** Floors and category sequences identical across a seed's arms; a divergent seed is excluded
   and counted.
6. **The SHUF null behaves.** `CAP1_SHUF` sits at or below its majority floor — a shuffled pairing gives a
   one-unit readout nothing but the prior.

### 2.5 THE AMENDMENT — the exogenous cap was INERT, and the reach control is what found it

Written before any criterion arm ran, after the section 2.4 controls were run on the drafted harness.

**The defect.** The draft set only `eng._l_max = N`. `NGSNN.fit`'s greedy pursuit has two stops before the
cap: the train-RMSE target (`if E.pow(2).mean().sqrt() < self.eps_tol: break`) and saturation (`if not keep:
break`). On the brain's own buffer the RMSE target is met almost immediately — measured at seed 0, train RMSE
0.257 against the engine's default `eps_tol = 0.28` — so the pursuit breaks at **2 units** and `l_max` never
binds. Measured across seeds 0–3, CAP16 admitted **2, 3, 4, 5** units (fit maxima 3, 3, 9, 11) — squarely
inside the uncapped incumbent's own 2–7 band.

**The dissociation that decided the cause** (`why_cap_never_binds.py`, seed 0; `l_max = 16` held fixed,
`eps_tol` swept, in one command with its own positive control):

| eps_tol | 0.28 (default) | 0.10 | 0.02 | 0.001 | 0.0 |
|---|---|---|---|---|---|
| units admitted | **2** | 16 | 16 | 16 | 16 |

with `l_max = 1 → exactly 1 unit` in the same command. **The cap is REACHABLE; it was PRE-EMPTED.** An
in-code comment states the opposite for a different regime — *"which is exactly why the old eps_tol never
fired"* — and reading it instead of measuring is what produced the draft. Runtime evidence outranks static
reading, as the invariant says.

**The amendment.** A capped arm sets BOTH `_l_max = N` and `_eps_tol = 0.0`. `eps_tol` is then CONSTANT
across every capped arm, so it cannot confound acc(N) against acc(M); N is the only thing that varies, which
is what "capped exogenously at N" requires.

**Disclosed, and it is a real cost.** The grid therefore measures width in a regime the LIVE brain does not
run in: the live engine keeps `eps_tol = 0.28` and stops at 2–7 units on its own. SC-012a's answer is about
whether width, once forced, buys accuracy — not about whether this brain would ever reach that width
unaided. The `INCUMBENT` arm runs beside the grid so the gap is a printed number rather than a footnote.
`eps_tol` is not tuned to move a verdict: it is driven to zero to remove a stop, held at one value for every
arm, and reported.

## 3. Results — 88 cells at `2fddea8`, 11 arms x 8 seeds x 800 ticks

Provenance guard PASS (one head, matching the repo's; wall times and accuracies vary; `eps_tol_used`
and `l_max_used` read back correct on every capped cell). Same rows: `cat_seq` byte-identical across all
eleven arms on all 8 seeds — **the engine's width does not feed back into the brain's own stream**, which
is what makes the grid comparable at all. 0 seeds excluded; 0 counter disagreements; 0 feed refusals; 0
permutation overruns.

### 3.1 What the DRAFT harness would have published, and why it is recorded here

Run first, at seed 0, as section 2.4's controls:

| arm | cap | units | fit max | acc | W hash |
|---|---|---|---|---|---|
| CAP1 | 1 | 1 | 1 | 0.2111 | `266e2da8…` |
| CAP2 | 2 | 2 | 2 | 0.9278 | `307357d4…` |
| CAP4 | 4 | **2** | 3 | **0.9278** | `21b71863…` |
| CAP8 | 8 | **2** | 3 | **0.9278** | `21b71863…` |
| CAP16 | 16 | **2** | 3 | **0.9278** | `21b71863…` |

`CAP4`, `CAP8` and `CAP16` are **bit-identical** — one W hash, one accuracy. The five-point grid collapsed
to two distinct substrates. On that grid acc(16) − acc(1) = **+0.717**, and (b) clears its 0.744 majority
floor at 0.928: **SC-012a would have FIRED**, lifting Part C's "units admitted, never growth" rule and
licensing L2's growth-on-plateau — on a comparison of 1 unit against 2, with three of five arms the same
run. It is recorded because it is the reading this arc would have published, and because a control that
only ever runs after the verdict is not a control.

### 3.2 The grid, amended (means over 8 seeds; floors on the same slice)

| arm | units (final) | fit max | cap bound | acc | acc − majority | acc − persistence |
|---|---|---|---|---|---|---|
| CAP1 | 1 (8/8) | 1 | 8/8 | 0.3420 | −0.273 (t −2.82) | −0.612 (t −6.90) |
| CAP2 | 2 (8/8) | 2 | 8/8 | 0.6740 | +0.059 (t 1.38) | −0.280 (t −3.53) |
| CAP4 | 4 (8/8) | 4 | 8/8 | 0.7167 | +0.101 (t 1.55) | −0.237 (t −3.64) |
| CAP8 | 6–8 | 7–8 | 6/8 | 0.8069 | **+0.192 (t 6.78)** | −0.147 (t −2.70) |
| CAP16 | 9–14 | 14–16 | **2/8** | 0.8066 | **+0.191 (t 3.94)** | −0.147 (t −2.65) |
| INCUMBENT | 2–7 | — | n/a | 0.7479 | — | — |

### 3.3 The criterion, conjunct by conjunct

- **(a) MET.** acc(16) − acc(1): t **5.024**, p 0.0039, mean **+0.4646**, 8 wins / 0 losses, 0 ties,
  random-sign false-positive rate 0.035.
- **(b) MET.** acc(16) − majority(16): t **3.936**, p 0.0039, mean **+0.1913**, 8 wins / 0 losses.
- **(c) VIOLATED — the veto fired.** The SAME grid on marginal-preserving shuffled labels ALSO rose:
  t **1.928** against the 1.895 bar, p **0.0469** against 0.05, mean +0.1635, 6 wins / 1 loss / 1 tie.

**SC-012a DOES NOT FIRE.** Section 2.2 pre-registered this exactly: "If it clears, the criterion's rise
is the instrument's, and SC-012a does not fire whatever (a) and (b) read." It clears **narrowly**, and
that margin is recorded as a disclosure, not as grounds to discount a rule written before the data
existed — the 2026-09-05 defect was a failed conjunct of a pre-registered PASS relabelled "secondary".

### 3.4 Why the null rose — the attribution (P8 clause 3)

| grid | CAP1 | CAP2 | CAP4 | CAP8 | CAP16 | its own majority floor |
|---|---|---|---|---|---|---|
| REAL | 0.342 | 0.674 | 0.717 | 0.807 | 0.807 | 0.615 |
| SHUF | 0.362 | 0.551 | 0.428 | 0.543 | 0.525 | 0.591 |

The shuffled grid rises off CAP1 and then oscillates around 0.52–0.55, **never once clearing its own
majority floor at any width**. The mechanism: a **1-unit readout cannot represent the label marginal** on
this 5-class skewed stream (seed 0's buffer is {1: 270, 2: 94, 3: 11, 4: 6, 5: 11}), so adding units
recovers the prior — with or without a percept→category pairing. (c) is carried by seeds 0 and 7, where
CAP1_SHUF read 0.144 and 0.000. **Conjunct (a) uses acc(1) as its baseline, so (a) and (c) are
contaminated by the SAME degeneracy**: part of (a)'s +0.465 is prior-recovery that the null reproduces.
The plan's own null caught a defect in the plan's own contrast.

**What survives the null, stated as a dissociation rather than as SC-012a firing:** REAL clears its
majority floor from CAP8 (t 6.78) while SHUF never clears its floor at any width. That contrast does not
rest on acc(1), and it is the honest form of "the pairing is learnable at width".

### 3.5 Reported beside the criterion, not barred

- **Marginal gain per doubling** — the efficiency signal Part C names: 1→2 **+0.332** (t 2.99, clears);
  2→4 +0.043 (t 0.72); 4→8 +0.090 (t 1.07); 8→16 **−0.0003** (t −0.007). **Only the first doubling pays.**
- **Forced width does not beat the brain's own.** CAP16 (9–14 units) − INCUMBENT (2–7 units) = +0.0587,
  t 0.719, p 0.2734 — does NOT clear. Driving `eps_tol` to zero to force ~12 units buys nothing over what
  the engine admits unaided.
- **Nothing reaches the persistence floor at any width.** acc − persistence stays negative on 8 of 8 seeds
  at every cap (CAP16 t −2.65). Consistent with L1b and L1c: the readout still does not beat "answer the
  previous category".

### 3.6 Controls, including the one that FAILED

| # | control | result |
|---|---|---|
| 1 | the cap binds (fit max >= N) | **NOT MET at N=16 (2/8)**; MET at N=1,2,4 (8/8) and N=8 (6/8) |
| 2 | CAP1 reads exactly 1 unit | MET, 8/8 |
| 3 | CAP4/8/16 are distinct substrates | MET, 8/8 (the defect of 3.1 is gone) |
| 4 | the two width counters agree | MET, 0 disagreements |
| 6 | CAP1_SHUF at or below its majority floor | MET (mean −0.229) |

**Control 1 failed at N=16 and it is not smoothed.** Above ~14 units the pursuit saturates before the cap
and `prune_unearned` cuts further, so "N=16" realizes **9–14** units. Per section 2.4 the controls qualify
the reading, they do not veto a ratified criterion — and this failure does not undermine (a): widths 1 and
9–14 genuinely differ (control 3, 8/8). What it bounds is the LANGUAGE: every claim here is about width up
to the **realized 9–14**, never "up to 16".

## 4. Disposition

**SC-012a did not fire, so section 2.3's not-fired branch executes**, and the licence for it is earned:
the cap did vary width (controls 2, 3, 4 MET; 1 partial), so this is a measured null and not an instrument
failure.

1. **Part C's reporting rule STAYS.** Owner-facing surfaces continue to say units **admitted**, never
   growth. `tests/test_units_admitted_wording.py` is unchanged and still binding.
2. **L2 stays deleted and is NOT rebuilt as an admission trigger.** L2 was already deleted on its own null
   (T092); this removes the remaining reason to re-propose it — width buys no next-category accuracy over
   the incumbent at any realized width up to 9–14, and the marginal gain per doubling is zero past N=2.
3. **The width lever moves to Phase 3's constructed width (T065)**, where paging and the prior need it for
   their own reasons.
4. **NOT claimed:** that width is useless in general, or that a wider substrate cannot pay on a stream with
   more structure. The measurement is about THIS stream, whose persistence floor (0.954) no arm reached at
   any width. Part C's "world before the wire" principle is the standing reading: the limiter is the world.

**In-tree:** `tests/test_real_percept_stream.py::test_width_buys_next_category_accuracy_and_the_shuffled_grid_does_not`,
a STRICT xfail carrying these numbers. Only the three conjuncts fail through `AssertionError`; every
instrument check (same rows, the reach control, the width-varied control, degenerate ties) fails through
`pytest.fail` -> `Failed`, which is **not** a subclass of `AssertionError` and therefore cannot be absorbed
by the xfail — an instrument that broke can never be filed as the pre-registered null.

## 5. What this arc got wrong in its own apparatus

Recorded because the 2026-09-06 through-line was that every arc's worst defect was in the measuring
apparatus, and this arc is four for four.

1. **The exogenous cap was inert.** `l_max` alone never bound above ~2 units; three of five grid arms were
   bit-identical. Found by running section 2.4's reach control BEFORE the criterion, and fixed by driving
   `eps_tol` to zero with the cap (section 2.5).
2. **The same-rows guard could not pass.** It required identical floors across REAL and SHUF arms, but a
   shuffled arm's floors MUST move — that movement is the null working. It excluded all 8 seeds and read
   as "the experiment failed" rather than "the check is wrong". Corrected to compare `cat_seq` across all
   arms and floors within each label-regime. An independent review raised the same defect as a BLOCKER and
   isolated it causally: neutralizing only the SHUF floor fields takes `keep` from `[]` to all 8 seeds.
3. **The verdict logic conflated my controls with the plan's criterion.** The first form of the stats
   script made firing conditional on controls of the session's own invention, which would let the verdict
   be steered by which controls I chose to write. Corrected 2026-09-07 **while the shuffled stage was
   still running**, so conjunct (c) was unknown when the logic changed.
4. **The record overclaimed its own null.** Section 2.3 said "no accuracy at any width up to 16" when the
   cap realized 9–14. Bounded to the width measured.

A fifth, found by the review and fixed in the in-tree test rather than the harness: a strict xfail
declared `raises=AssertionError` absorbs any `assert`-based instrument check, so a broken instrument would
have been filed as the pre-registered null. Instrument checks now raise `Failed`, which the xfail cannot
absorb.
