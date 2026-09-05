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
- Its consumers today (P2 census, three): `LearningProgress.consume_surprise` (T046a); the action gate's
  `surprise` feature (loop C2, an undirected salience like novelty); `self_state.surprise` on state(). It
  does NOT reach the arm's 64-d context, so the arm's GrowingLearner (categories → the neuro engine's
  stream), the world model (→ curiosity, Empowerment), the amygdala's context threat and the unit-keyed
  Q-state never see it.
- The context: `_safe_ctx` returns `visual` and `audio` as 4-vectors pooled from the real stream
  (near-constant: visual means [0.074, 0.098, 0.035, 0.035], stds ≤ 0.035), `temperature` = [proprio],
  `distance` pinned SAFE (the E-STOP channel, never touched). `encode_multimodal` averages four Gaussian bumps
  per band into 16 cells.
- Delivery for a probe with NO tree change: wrap `_safe_ctx` so `visual[3] = clamp(_surprise_signal)` — one
  scalar replaces the fourth (near-constant) element; the band keeps its shape, three of its four bumps, and
  gains one moving bump whose position IS the surprise. With the id lane refused there is no category bump to
  confound (W2 §3.0 deviation 1), and the band today carries no category (W2 §1c). Vision is the band the
  surprise comes from.

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

## 3. Results — 12 cells, 3000 ticks, 20:17–20:29 UTC, at `f1a9bf8` (the pre-registration commit's head;
`probe/` in `memory/archive/q2-2026-09-05/`)

| arm | seed | arm units (vis / aud learners) | engine acc last third / persistence / majority | `raw_emp` zero-frac after 300 | Cramér's V (action × surprise-half) | G-INFO held-out R² (incumbent only) | perr last full block |
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
  7 / 4 / 5 likewise. The delivery fired on every wake tick (spliced 1481–1485; perception unified on all but
  the first), the band's length is 4, and the vision/audio learners (9 / 15 prototypes) are identical across
  arms, as they must be (they see the percept, not the context).
- **The surprise scalar is alive:** 1481–1485 rows, mean 0.1545, 503–505 distinct values at 4 dp (the 300-tick
  smoke's 0.013 was the run's first minute; W2 §1b's 0.082 over 294 rows is the same early regime).

## 4. Verdict — each gate against §2's wording, MET / NOT MET

- **G-INFO (the premise): MET, 3 of 3.** The incumbent's 64-d pattern predicts the surprise scalar with
  held-out R² −53.25 / −0.008 / 0.009 (the −53 is the ridge extrapolating a near-constant pattern; the sign
  says nothing predicts): the context does not carry the encoder's surprise, so a lane WOULD add information.
- **G-FRAG (the plan's own condition, "no alphabet fragmentation"): NOT MET.** The arm learner's units under
  `surprise` are 9 / 6 / 8 against the incumbent's 7 / 4 / 5 — +2 / +2 / **+3**; the bar was ≤ +2 on 3 of 3
  (CHOSEN), and seed 2 exceeds it. The `surprise_shuffled` (+1 / +2 / +2) and `surprise_pinned` (+0 / +2 / +3)
  arms fragment too: moving the band's fourth element from ~0.035 to the surprise's range fragments the
  alphabet whatever the element carries — the fragmentation is the bump's, not the information's.
- **G-CONSUMER: NOT MET on (i) and (ii); (iii) met by sign only.** (i) The engine's last-third accuracy under
  `surprise` (0.140 / 0.263 / 0.021) is BELOW the shuffled arm's (0.581 / 0.419 / 0.212) on 3 of 3 and below
  its own persistence floor on 3 of 3 — the lane HARMS the grower (more categories, a less learnable stream);
  0 of 3. (ii) `raw_emp` zero-fraction differences vs the shuffled arm are 0.030 / 0.073 / 0.084, all inside
  the 0.164 noise scale; 0 of 3. (iii) Cramér's V under `surprise` exceeds the shuffled arm's on seeds 0 and 1
  (0.0735 vs 0.0686; 0.1108 vs 0.1036) and not on seed 2 (0.1077 vs 0.1198): 2 of 3 by SIGN, with margins
  0.005 and 0.007 against a between-seed spread of 0.03–0.05 and an incumbent (no delivery at all) reading
  0.09–0.15 — P(≥ 2 of 3 by sign under the null) = 0.5. **Pre-registration defect, disclosed:** (iii) carried
  no margin, so it cannot fail informatively at n = 3; it is recorded as met by the letter and uninformative,
  and it does not rescue the gate on its own reading.
- **PROCEED requires G-INFO ∧ G-FRAG ∧ G-CONSUMER: NOT MET (G-FRAG). Q2 CLOSES WHOLE, BY MEASUREMENT.** No lane
  is declared; the surprise keeps its three consumers; W2's row stays REFUSED; the plan's Part H Q2 reads CLOSED
  BY MEASUREMENT with G-FRAG named.
- **Reported, not scored:** the engine's last-third accuracy swings by seed and by any perturbation (incumbent
  0.18 / 0.84 / 0.55; pinned 0.89 / 0.32 / 0.16) — at n = 3 it could not have resolved (i) in either
  direction; a build's S = 8 reds would have needed the paired form. Recorded as a limitation of the probe's
  power, not of the verdict (G-FRAG fails on a unit COUNT, which does not swing).
- **Owed onward:** none from Q2. The surprise signal's existing consumers are the action gate's feature (loop
  C2), `LearningProgress.consume_surprise` and `self_state.surprise`; a fourth consumer would need a red of
  its own.
