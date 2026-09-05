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
