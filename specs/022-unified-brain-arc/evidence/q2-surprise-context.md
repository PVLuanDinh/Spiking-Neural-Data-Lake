# Q2 — W2's surprise half, its own probe (2026-09-05)

Owner instruction, in chat ("Do all this", 2026-09-05): Q2 gets its own probe rather than closing by default.
The plan's W2 sentence asked for the encoder's surprise in the context; W2 refused the id lane on measurement
(evidence/w2-context-latents.md §4) and its §5 asked whether a surprise-only context change — one scalar, no
alphabet fragmentation — is worth a probe of its own, or W2 closes whole. This record answers by measurement.

Lessons carried from the live-floor arc, applied here before any cell runs: every bar is CHOSEN and marked so;
every control's expected reading on THIS gauge is computed from existing data before it is a conjunct (obs
#2073); a bar set from the probe's own seeds cannot be scored MET on them (obs #2074) — the probe's verdict is
PROCEED / REFUSED at n = 3, and the build's reds (if any) are read at S = 8 on seeds the probe never saw.

## 1. Territory (code at HEAD; W2 §1 and the live-floor arc's numbers)

- The signal: `Brain._surprise_signal` ∈ [0, 1], the Oja/Sanger residual of the vision encoder, set in
  `_perceive_real` from `PerceptSurprise.observe(percept)` every wake tick a real frame is perceived. W2 §1b:
  72 distinct values at 4 dp over 294 rows, mean 0.082, range [0, 1] — alive.
- Its consumers today (P2 census, TWO — an earlier draft said three: `self_state.surprise` is the
  generative self-model's own e-prop error `_self_surprise`, a different quantity; the record panel's
  INT-2): `LearningProgress.consume_surprise` (T046a, the |LP| third factor; brain.py ~1708) and the
  action gate's `surprise` eligibility feature (loop C2, brain.py ~1910). It does NOT reach the arm's 64-d
  context, so the arm's GrowingLearner (categories → the neuro engine's stream), the world model (→
  curiosity, Empowerment), the amygdala's context threat and the unit-keyed Q-state never see it. The
  scalar delivered on tick t is the residual of tick t−1's percept — `_safe_ctx` runs at the top of
  `_wake_tick`, `_perceive_real` after `brain.tick` — the band's own one-tick lag (INT-9).
- The context: `_safe_ctx` returns `visual` and `audio` as 4-vectors pooled from the real stream
  (near-constant: visual means [0.074, 0.098, 0.035, 0.035], stds ≤ 0.035), `temperature` = [proprio],
  `distance` pinned SAFE (the E-STOP channel, never touched). `encode_multimodal` averages four Gaussian bumps
  per band into 16 cells.
- Delivery for a probe with NO tree change: wrap `_safe_ctx` so `visual[3] = clamp(_surprise_signal)` — one
  scalar replaces the fourth (near-constant) element; the band keeps its shape, three of its four bumps, and
  gains one moving bump whose position IS the surprise. With the id lane refused there is no category bump to
  confound (W2 §3.0 deviation 1), and the band today carries no category (W2 §1c). Vision is the band the
  surprise comes from. **Receivers of the wrapped ctx, found by the record panel (INT-3):** besides the
  arm's pattern, `_wake_tick` reads the same ctx into the L5 generalizing-novelty pseudo-count (which feeds
  BACK into `_safe_ctx`'s attention gain on the next tick, and into LC salience, the time-clock reset and
  the affect targets), into `_c1_tick`'s 9-d affect feature (visual[3] is feature 3), and into the
  predictive-coding front end (→ metacog hold, LC, re-entrant salience); `_tick_ensemble` does not read it.
  A no-tree-change splice at `_safe_ctx` therefore wires four receivers, not one; an on-tree lane would
  owe a lane per receiver or a splice at `_context_pattern`.

## 2. Protocol — pre-registered before any cell runs

Every cell: `throwaway_home()`, `make_brain(seed, persist=False)`, 3000 ticks, marks 1500 / 3000, one arm per
subprocess, 4 workers; seeds 0–2; instance monkeypatches only; the live daemon left running; JSON outside the
repo. Harness: `memory/archive/q2-2026-09-05/q2_surprise_probe.py`. Per wake tick the probe logs the 64-d
pattern the arm built, the surprise scalar delivered, the arm's unit and action, perr, raw_cur, raw_emp; at
marks the engine summary, the arm learner's and the vision/audio learners' unit counts.

Arms:
- **incumbent** — untouched (the instrument control: must reproduce the live-floor arc's Stage 0b incumbent
  at summary level — `raw_emp` zero-fraction 0.310 / 0.284 / 0.448, perr last full block 0.00453 / 0.00541 /
  0.00348 — else the harness is wrong).
- **surprise** — `visual[3] = s_t`.
- **surprise_shuffled** — `visual[3]` = a seeded uniform draw from the surprise values seen so far (the
  marginal kept, the timing destroyed — the control that is not the method).
- **surprise_pinned** — `visual[3]` = the running mean of s (the bump's energy, no information).

Gates, in force before the first cell (each bar CHOSEN, each control's expected reading stated):
- **G-INFO (the premise, plan Part C):** on the incumbent arm, a ridge readout (λ = 1e-3) from the 64-d pattern
  to s_t, fit on wake rows 1–1000 and scored on the rest, reads held-out R² < 0.5 on ≥ 2 of 3 seeds (CHOSEN).
  Expected: the pattern's vision/audio bands are near-constant (W2 §1c) and the proprio band is the body's
  own walk, so the context should carry little of the encoder's residual — R² near 0. If R² ≥ 0.5 the context
  already carries the surprise and the lane is redundant: REFUSED.
- **G-FRAG (the plan's own condition):** the arm learner's unit count at 3000 under `surprise` ≤ incumbent + 2
  on 3 of 3 seeds (CHOSEN; incumbent today 7 / 4 / 5 at 3000 — Stage 0b). The vision/audio learners observe
  the PERCEPT, not the context, and cannot change — reported as the sanity row (must equal the incumbent's).
- **G-CONSUMER (the charge, at probe power):** on ≥ 2 of 3 seeds, at least one of —
  (i) the engine's last-third held-out accuracy under `surprise` > under `surprise_shuffled` AND > its own
  persistence floor (both floors printed; the incumbent's reading is the reference);
  (ii) the `raw_emp == 0` fraction after wake row 300 under `surprise` differs from `surprise_shuffled` by
  more than the incumbent's seed-to-seed range (0.284–0.448 → 0.164; CHOSEN as the noise scale);
  (iii) the arm's actions USE the scalar: Cramér's V of action × {s above / below the arm's own median}
  after wake row 300 under `surprise` exceeds the same statistic under `surprise_shuffled` (the null: same
  marginal, no timing) on the same seed. (An earlier draft's "the unit series differs from the pinned arm on
  > 10 % of rows" is withdrawn before any run: two brains with any input difference diverge, so it could not
  fail.)
- **PROCEED TO BUILD iff G-INFO AND G-FRAG AND G-CONSUMER.** Otherwise Q2 closes WHOLE, by measurement: the
  record says which gate failed, the plan's Part H Q2 reads CLOSED BY MEASUREMENT, W2's row stays REFUSED,
  and no lane is declared (P2: nothing additive lands).

**Corrections to §2's descriptions, found by the record panel AFTER the read (disclosed, not rewritten):**
`surprise_pinned` delivers the running mean — a RAMP from 0 to 0.155 over the run, identical to the
other two treated arms for the first 77 wake rows (the surprise is exactly 0 there) and 0.087–0.156 after
row 300: a near-constant fourth element at the band's low edge, not "the bump's energy" (INT-4, I-9).
`surprise_shuffled` keeps the marginal only approximately — drawing from the running prefix over-weights
the early low regime (delivered mean 0.113–0.121 against the surprise's 0.1545; 230–244 distinct values
against 503–505) (I-6). The surprise is exactly 0.0 on 62 % of wake ticks, so "the arm's own median" is
0.0 and (iii)'s split is nonzero-vs-zero (I-7); V is computed against the RAW surprise on every arm (the
delivered value would make the pinned arm's split an early/late time split) (I-5). G-FRAG's bar had no
pre-computed null spread on the unit COUNT — the obs #2073 rule was applied to G-INFO and not to G-FRAG
(I-2, INT-7). (i)'s persistence conjunct was unreachable for every arm INCLUDING the incumbent at this
horizon (W2 §4.5 and the live-floor arc's Stage 0b already showed 0.18 / 0.84 / 0.55 against 0.97 / 0.99
/ 0.98) — a conjunct with no reachable green, a pre-registration defect (PH-2, I-8).

### 2.3 If the probe passes — the build and its reds at S = 8 (so the bar exists before the data)
`_safe_ctx` delivers s as the fourth visual element on the tree; the lane declared
(`percept_surprise.surprise_out → brain.context_surprise_in`; census + `tests/test_wiring_manifest.py` in the
same commit); reds per seed over S = 8 (seeds 3–10, never seen by the probe) against `surprise_shuffled`:
paired one-sided t > 1.895 AND exact sign-flip p ≤ 0.05 on the engine's held-out accuracy (W-R2b's form) and
on whichever G-CONSUMER statistic carried the probe; the pinned arm must lose to the wired arm on the same
statistic (P3 charge: the information, not the bump). Positive control in the same command (a planted paired
effect on the observed tie pattern clears the bar; a null sample does not). G-FRAG re-read at S = 8.

### 2.4 Disposition on a null
Q2 closes whole; the failing gate named; no lane; the surprise keeps its three consumers.

### 2.5 Amendment 1 — G-INFO re-measured with an ALIGNED instrument (written after the record panel, BEFORE
the re-run cells)

The record panel (§5) found the G-INFO instrument misaligned: `make_brain` runs `predictive_coding=True`, so
`LiveBrain.tick` calls `arm._context_pattern` a SECOND time per wake tick (the PC front end's re-encode) and
the harness logged two patterns per surprise value, then sliced the doubled list against the surprise series
— pairing tick i//2's pattern with s_i. The archived R² values (§3) are that misaligned statistic; the surprise
arm's negative rows, which WERE the readout's positive control (the delivery is in that pattern), failed and
were explained away. G-INFO as first scored is therefore UNMEASURED. The harness now keeps one pattern per
`_safe_ctx` call and drops the PC re-encode (counted). Re-run, pre-registered: incumbent × seeds 0–2 (the
gate) and surprise × seeds 0–2 (the positive control), 3000 ticks, marks 1500/3000. **G-INFO MET iff the
aligned incumbent held-out R² < 0.5 on ≥ 2 of 3 (CHOSEN, as before) AND the positive control — the surprise
arm's own pattern → s — reads R² > 0.5 on ≥ 2 of 3 (CHOSEN; the panel's aligned seed-0 reading was 0.93
brain-free and 0.93 in-brain).** If the positive control fails, the readout cannot see a signal it is given
and G-INFO stays UNMEASURED. Beside it, the instrument witness the panel asked for (INT-6): the visual band's
lit cells under delivery must extend past the incumbent's cells 0–4. The re-run also records V on BOTH
splits (raw surprise; the delivered value) with a within-arm permutation null (I-4/I-5) and the surprise's
zero mass (I-7) — reported, since the verdict does not rest on (iii).

## 3. Results — 12 cells, 3000 ticks, 20:17–20:29 UTC, at `f1a9bf8` (the pre-registration commit's head;
`probe/` in `memory/archive/q2-2026-09-05/`)

| arm | seed | arm units (vis / aud learners) | engine acc last third / persistence / majority | `raw_emp` zero-frac after 300 | Cramér's V (action × surprise-half, raw split) | G-INFO held-out R² — **MISALIGNED, invalid (§2.5 Amendment 1; superseded by §3.2)** | perr last full block |
|---|---|---|---|---|---|---|---|
| incumbent | 0 | 7 (9 / 15) | 0.183 / 0.973 / 0.335 | 0.310 | 0.1045 | **−53.25** (y var 0.0832) | 0.00453 |
| incumbent | 1 | 4 (9 / 15) | 0.839 / 0.995 / 0.843 | 0.284 | 0.0929 | **−0.008** | 0.00541 |
| incumbent | 2 | 5 (9 / 15) | 0.553 / 0.981 / 0.555 | 0.448 | 0.1500 | **0.009** | 0.00348 |
| surprise | 0 | **9** (+2) | 0.140 / 0.864 / 0.335 | 0.260 | 0.0735 | (−1.43, the delivery in the pattern) | 0.00566 |
| surprise | 1 | **6** (+2) | 0.263 / 0.975 / 0.253 | 0.319 | 0.1108 | (−0.36) | 0.00563 |
| surprise | 2 | **8** (+3) | 0.021 / 0.980 / 0.346 | 0.358 | 0.1077 | (−0.45) | 0.00495 |
| surprise_shuffled | 0 | 8 (+1) | 0.581 / 0.954 / 0.377 | 0.290 | 0.0686 | (−232.8) | 0.00555 |
| surprise_shuffled | 1 | 6 (+2) | 0.419 / 0.930 / 0.668 | 0.246 | 0.1036 | (−0.10) | 0.00484 |
| surprise_shuffled | 2 | 7 (+2) | 0.212 / 0.983 / 0.520 | 0.274 | 0.1198 | (−0.06) | 0.00403 |
| surprise_pinned | 0 | 7 (+0) | 0.886 / 0.989 / 0.611 | 0.293 | 0.1270 | (−1.64) | 0.00387 |
| surprise_pinned | 1 | 6 (+2) | 0.321 / 0.975 / 0.460 | 0.274 | 0.0861 | (−1.42) | 0.00646 |
| surprise_pinned | 2 | 8 (+3) | 0.156 / 0.981 / 0.056 | 0.393 | 0.1183 | (−0.14) | 0.00505 |

- **Instrument control (incumbent vs the live-floor arc's Stage 0b at 3000):** `raw_emp` zero-fractions 0.310 /
  0.284 / 0.448 and perr 0.00453 / 0.00541 / 0.00348 reproduce that arc's incumbent digit for digit; arm units
  7 / 4 / 5 likewise (`stage0b/incumbent_s*.json`, `arm_units`). The delivery fired on every wake tick
  (spliced 1481–1485; perception unified on all but the first), the band's length is 4, and the vision/audio
  learners (9 / 15 prototypes) are identical across arms — which proves only that the percept path was
  untouched by the wrapper, true by construction (INT-6); the witness that the delivered scalar reached the
  64-d pattern is §3.2's lit-cell count (the visual band lights cells past the incumbent's 0–4 under
  delivery).
- **The surprise scalar is alive — and it is ONE realization:** 1481–1485 rows, mean 0.1545, 503–505 distinct
  values at 4 dp, exactly 0.0 on 62 % of ticks (the 300-tick smoke's 0.013 was the run's first minute; W2
  §1b's 0.082 over 294 rows is the same early regime). The sequence is byte-identical across the four arms
  and across seeds (`PerceptSurprise` is built with seed 0 and the pack pointer advances per wake tick), so
  the twelve cells sample one stimulus; the seeds vary only the brain (INT-5).

### 3.2 The aligned G-INFO re-run (§2.5 Amendment 1; incumbent and surprise × seeds 0–2; `probe2/`)

| arm | seed | aligned held-out R² (pattern → s) | aligned? / PC re-encodes dropped | visual-band lit cells: max / mean after row 300 | V raw split / delivered split | permutation null mean ± sd, p | surprise zero-frac, median |
|---|---|---|---|---|---|---|---|
| incumbent | 0 | **-0.019** | True / 1483 | 7 / 5.44 | 0.1045 / None | 0.098 ± 0.02, p 0.38 | 0.62, 0.0 |
| incumbent | 1 | **-0.010** | True / 1481 | 7 / 5.39 | 0.0929 / None | 0.1 ± 0.019, p 0.63 | 0.621, 0.0 |
| incumbent | 2 | **-0.036** | True / 1485 | 7 / 5.33 | 0.15 / None | 0.096 ± 0.02, p 0.0 | 0.62, 0.0 |
| surprise | 0 | **0.930** | True / 1483 | 13 / 6.76 | 0.0735 / 0.0735 | 0.101 ± 0.019, p 0.925 | 0.62, 0.0 |
| surprise | 1 | **0.937** | True / 1481 | 13 / 6.74 | 0.1108 / 0.1108 | 0.099 ± 0.019, p 0.26 | 0.621, 0.0 |
| surprise | 2 | **0.927** | True / 1485 | 13 / 6.8 | 0.1077 / 0.1077 | 0.1 ± 0.021, p 0.32 | 0.62, 0.0 |

- Every cell reproduces the first run's brain to the digit (units, engine numbers, drives; the harness change
  touched logging only); each cell dropped one PC re-encode per wake tick, so the pattern log is aligned
  (`aligned = True`).
- **G-INFO, aligned (§2.5 Amendment 1): MET.** The incumbent's pattern predicts the surprise with held-out R²
  -0.019 / -0.010 / -0.036 (< 0.5 on 3 of 3) AND the positive control — the surprise arm's own
  pattern → s — reads 0.930 / 0.937 / 0.927 (> 0.5 on 3 of 3): the readout sees a signal it is given and
  the incumbent context carries none of it. The lane WOULD add information — a true premise closed by G-FRAG.
- The instrument witness (INT-6): under delivery the visual band lights 13 / 13 / 13 cells at most against the
  incumbent's 7 / 7 / 7 — the delivered scalar reached the 64-d pattern.
- V's within-arm permutation null (200 shuffles of s against the actions) is ≈ 0.10 ± 0.02 on every cell; the
  surprise arm's readings sit inside it (permutation p above); on the delivered split the surprise arm reads
  the same as on the raw split by construction (its delivered value IS s).

## 4. Verdict — each gate against §2's wording, MET / NOT MET

- **G-INFO (the premise), as FIRST scored: UNMEASURED** — the instrument was misaligned (§2.5 Amendment 1);
  the archived −53.25 / −0.008 / 0.009 pair tick i//2's pattern with s_i, and the surprise arm's rows
  (−1.43 / −0.36 / −0.45), which were the readout's positive control, FAILED and were wrongly explained as
  "the delivery in the pattern". **As re-measured with the aligned instrument (§3.2): MET — incumbent held-out R² -0.019 / -0.010 / -0.036 (< 0.5 on 3 of 3) and the positive control 0.930 / 0.937 / 0.927 (> 0.5 on 3 of 3): the context does not carry the encoder's surprise, so a lane WOULD add information; the premise is true and the probe closes on G-FRAG.**
- **G-FRAG (the plan's own condition, "no alphabet fragmentation"): NOT MET by the letter.** The arm
  learner's units under `surprise` are 9 / 6 / 8 against the incumbent's 7 / 4 / 5 — +2 / +2 / **+3**; the bar
  was ≤ +2 on 3 of 3 (CHOSEN), and seed 2 exceeds it by one unit. Read against the same bar, the shuffled
  control (+1 / +2 / +2) is MET and the pinned control (+0 / +2 / +3) is NOT MET — the control that is not
  the method passes the bar the method failed, and a zero-information ramp confined to 0.09–0.16 fails it
  on the same seed. The between-arm spread at a fixed seed is ±1 unit and no noise scale was pre-computed
  for the count (a pre-registration defect, disclosed), so the count cannot attribute the fragmentation to
  the lane, to the bump, or to divergence through the four receivers the splice wired (§1) — an earlier
  draft's "the bump's, not the information's" is an exclusion these counts cannot support and is withdrawn
  (I-2, INT-7, PH-5). What stands: the bump ALONE already fails the bar; whether the information adds
  fragmentation (+7 total vs +5 / +5; seed 0: +2 vs +0) is not resolvable at n = 3.
- **G-CONSUMER: MET BY THE LETTER — via (iii) on 2 of 3; (i) 0 of 3; (ii) 0 of 3.** The letter is defective
  and the gate is NOT load-bearing: PROCEED fails on G-FRAG alone (PH-3). (i) The engine's last-third
  accuracy under `surprise` (0.140 / 0.263 / 0.021) is below the shuffled arm's (0.581 / 0.419 / 0.212) on 3
  of 3 — UNSCORED at n = 3: the gauge swings by seed and by any perturbation (the pinned control moves it
  +0.70 / −0.52 / −0.40 against the incumbent with no information delivered), and the persistence-floor
  conjunct is unreachable for every arm including the incumbent, so (i) fails by construction, not by
  measurement. An earlier draft wrote "the lane HARMS the grower" — withdrawn as unearned (PH-1, I-3); the
  "more categories" mechanism fails on seed 1 where all three treated arms hold 6 units. Post hoc and
  unregistered, reported not scored: the surprise arm's accuracy is the lowest of the four arms on every
  seed (P = 1/64 under exchangeability). (ii) `raw_emp` zero-fraction differences vs the shuffled arm are
  0.030 / 0.073 / 0.084, inside the 0.164 noise scale. (iii) Cramér's V under `surprise` (raw split) exceeds
  the shuffled arm's on seeds 0 and 1 (0.0735 vs 0.0686; 0.1108 vs 0.1036) and not on seed 2 (0.1077 vs
  0.1198): 2 of 3 by SIGN, margins 0.005–0.007; on the DELIVERED split the shuffled arm reads 0.1096 /
  0.0632 / 0.1183 and the count is 1 of 3 (I-5); the independence null for a 13 × 2 table at n ≈ 1183 is
  V ≈ 0.10 ± 0.02 and every surprise-arm reading sits inside it (permutation p 0.92 / 0.25 / 0.33; the
  incumbent, with no delivery, reads above both delivery arms, so the statistic rides the percept path —
  I-4, PH-8). P(≥ 2 of 3 by sign under the null) = 0.5. **Pre-registration defect, disclosed:** (iii)
  carried no margin, so it cannot fail informatively at n = 3.
- **PROCEED requires G-INFO ∧ G-FRAG ∧ G-CONSUMER: NOT MET (G-FRAG). Q2 CLOSES BY MEASUREMENT — for THIS
  delivery.** The failing gate fails on the delivery: `visual[3] = s ∈ [0, 1]`, 4–30× the band's native
  range, and the zero-information pinned control fails the same bar on the same seed. A RANGE-MATCHED
  delivery (s scaled into the band's ~[0, 0.1]) was not probed and is the reopening an owner could name;
  whether it would be visible at σ = 1/15 is not measured here (PH-4). No lane is declared; the surprise
  keeps its two consumers; W2's row stays REFUSED; the plan's Part H Q2 reads CLOSED BY MEASUREMENT for this
  delivery with G-FRAG named. Commit `c31900e`'s subject ("fragments the alphabet and harms the grower")
  over-states what §4 now says; the subject stands in history, this record corrects it.
- **Reported, not scored:** the engine's last-third accuracy swings by seed and by any perturbation (incumbent
  0.18 / 0.84 / 0.55; pinned 0.89 / 0.32 / 0.16) — at n = 3 it could not have resolved (i) in either
  direction; a build's S = 8 reds would have needed the paired form. Recorded as a limitation of the probe's
  power, not of the verdict (G-FRAG fails on a unit COUNT, which does not swing).
- **Owed onward:** the range-matched delivery, if an owner names it (PH-4). The surprise signal's existing
  consumers are the action gate's eligibility feature (loop C2) and `LearningProgress.consume_surprise`; a
  third would need a red of its own.

## 5. The record panel — three lenses on `f1a9bf8..c31900e`, and the fold (commit 4)

Three read-only lenses (instrument / policy-honesty / integration; 108 tool calls; every finding reproduced by
a command, one lens re-ran an incumbent cell and reproduced the archive byte for byte; archived as
`record_panel.txt`). Verdicts: yes-with-the-fold ×3. The closure stands by the pre-registered letter (G-FRAG
NOT MET; PROCEED is a conjunction); the fold changes one gate's status (G-INFO: MET → UNMEASURED, then
re-measured aligned in §3.2), strips two causal headlines, and discloses the null scales.

| id | finding (short) | disposition |
|---|---|---|
| I-1 / INT-1 | G-INFO measured by a misaligned instrument (the PC front end's second `_context_pattern` call per tick doubled the pattern log; sliced 2:1 against s); the surprise arm's rows were a failed positive control | ACCEPTED: UNMEASURED as first scored; §2.5 Amendment 1 pre-registered the aligned re-run with its positive control; §3.2 carries it |
| I-2 / INT-7 / PH-5 | G-FRAG's causal reading unsupported: the pinned control fails the same bar on the same seed, the shuffled control passes it; no noise scale on the count | ACCEPTED: NOT MET by the letter; attribution withdrawn; per-arm bar readings stated; defect disclosed |
| I-3 / PH-1 / PH-2 / I-8 | "HARMS the grower" unearned (the gauge swings under any perturbation; the persistence conjunct unreachable for every arm) | ACCEPTED: struck from §4, Part H and the W2 pointer; the commit subject's over-statement stated |
| PH-3 | G-CONSUMER never stated as a whole gate; post-read qualifiers on a passed gate | ACCEPTED: MET by the letter, defect disclosed, not load-bearing |
| PH-4 | the closure attributed to the delivery yet closed WHOLE | ACCEPTED: closed for THIS delivery; the range-matched delivery named as the owner-nameable reopening |
| INT-2 | the consumer census named a non-consumer (`self_state.surprise` is `_self_surprise`) | ACCEPTED: two consumers |
| INT-3 | the wrapped ctx reached four receivers, not one | ACCEPTED: §1; G-FRAG hedged; §2.3's lane note |
| I-4 / I-5 / I-7 / PH-8 | V's independence null ≈ 0.10 ± 0.02; raw vs delivered split (1 of 3 on the delivered); the median is 0.0 (62 % zeros); the statistic rides the percept path | ACCEPTED, each at its bullet |
| INT-4 / I-9 / I-6 / INT-5 | the pinned arm is a ramp (identical to the others for 77 rows); the shuffled marginal only approximately kept; one stimulus realization across 12 cells | ACCEPTED: §2's post-read corrections; §3 |
| INT-6 / INT-10 | the 9 / 15 match proves only the percept path untouched; cite the Stage 0b JSON for 7 / 4 / 5 | ACCEPTED |
| INT-8 / PH-6 | three stale readers (the plan's W2 row, tasks.md's W2 row, the W2 record's §6); Part H did not name G-FRAG; the W2 pointer omitted G-CONSUMER | ACCEPTED: all pointed here |
| INT-9 | the one-tick lag unstated | ACCEPTED |
| PH-7 | a negative held-out R² over-read as "nothing predicts" | superseded by the re-run |
| PH-9 / INT-10 / I-10 | pre-registration preceded the cells (commit 20:17:34Z; cells from 20:17:34Z); gate text unchanged; every table digit reproduces; docs-only, tree clean | recorded |

Not changed by the panel: the closure (G-FRAG NOT MET by the letter), the twelve cells' digits, the plan's
W2 row status (REFUSED).
