# Insula level arms on the production perception tier

Lane insula-lvl-1022 (origin cade96cf), 2026-10-22. Owner pick "1 2", arc 1. Board thread INSULA.

## 1. Protocol (pre-registered 2026-10-22, committed BEFORE any arm is run)

**Why.** The 2026-10-19 level arms (constant salience at 2x / 0.5x / 0.0 of the live mean; D_out from
live 23 / 5 / 28) and the reading "the insula's outcome charge is its LEVEL; tick-to-tick variation is
outcome-inert (P3 dormancy shape 5)" were measured on the NO-ENCODER perception tier and are WITHDRAWN
(`tests/test_interoceptive_ignition.py`, PREREGISTERED test docstring; `encoder-path-root.md`). On the
production tier (both encoders) seed 0 gave N=6 C=9 S=20. This arc re-measures on that tier, over five
seeds, so the owner can adjudicate P3 shape 5 on data from the brain the daemon runs.

**Question.** On the production tier, is the insula's outcome charge carried by its LEVEL, by its
tick-to-tick VARIATION (timing), by both, or by neither?

**Harness semantics.** Identical to `_run` in `tests/test_interoceptive_ignition.py`, with `seed` as a
parameter: `make_brain(seed, persist=False)` under `throwaway_home()`, OUTCOME_TICKS = 200, the
insula's `observe` wrapped to replace the salience the brain consumes, rows per tick = (ignited,
winner, live_thr, salience, observed, tier). Production tier pinned by absolute path to the four
`SNN_*` variables. One arm per subprocess; OMP_NUM_THREADS=4; arms sequential.

**Arms, per seed s in 0..4.**

| arm | definition |
|---|---|
| live | the insula as built |
| live2 | a second live arm (repeat pair) |
| neutered | `_insula = None` |
| const (C) | constant at the live mean over CONSULTED ticks |
| shuf (S) | the live consulted trace permuted by `random.Random(s)` |
| x2 | constant at 2.0 * that mean |
| x05 | constant at 0.5 * that mean |
| x0 | constant at 0.0 |

**Statistics.** D_out(x) = ticks on which (ignited, winner) differs from live. D_thr(x) = ticks on
which `live_thr` differs from live. N = D_out(neutered), C = D_out(const), S = D_out(shuf),
L2 = D_out(x2), L05 = D_out(x05), L0 = D_out(x0). Also Z = D_out(x0 vs neutered) and R = D_out(live2).

**Controls (not the method).**
1. Repeat pair: R must be 0 for outcome AND threshold on every seed. Any seed with R > 0: the arm
   design is invalid on this tier; STOP, report, do not interpret that seed (or any).
2. The injected signal reaches the consumer: `_assert_control_arms_real` checks (same observed
   schedule as live, no replay exhausted, injected trace as designed, D_thr > 0 from live, recorded
   `live_thr` equals the law applied to the arm's own salience within 1e-6). The x2 and x05 arms get
   the same checks with their own expected trace; the x0 arm has D_thr checked against its law only
   (D_thr from live > 0 is required of all injected arms, x0 included).
3. Z reported, expected 0 (a constant zero is removal). Z > 0 is a finding about the injection, stated.
4. Tier gate: every row records (`_perception_tier`, `_audio_enc is not None`, `_perception_device`).
   A run not ("eprop-snn", True, device) is REFUSED, never reported; all arms of a seed share one device.
   Red shown with `SNN_EPROP_CKPT` set to a nonexistent absolute path.

**Pre-registered readings** (applied to the five-seed table; "most seeds" = at least 4 of 5):
- **LEVEL-ONLY**: max(C, S) = 0 on at least 4 seeds AND N > 0 on at least 4 seeds.
- **VARIATION matters**: max(C, S) > 0 on at least 4 seeds. (The 2026-10-20 seed-0 numbers C=9 S=20
  already lean here; that is why the bar is written for five seeds and not one.)
- **BOTH**: VARIATION matters AND the level also matters, where "level matters" = max(L2, L05) > 0
  on at least 4 seeds (a constant off the live mean moves the outcome, i.e. level is not free).
  If VARIATION matters and max(L2, L05) = 0 on at least 4 seeds the reading is VARIATION-ONLY.
- **NEITHER**: N = 0 on at least 3 seeds (the insula does not move the outcome on the production
  tier; P8 deletion-on-a-null is the owner's to adjudicate).
- **INDETERMINATE**: none of the above holds (reported as that, with the per-seed table; no
  reading is forced).
- Per-seed bar check also reported: the 2026-10-20 bar N > max(C, S) holds on how many of 5 seeds.
No retuning after data. A null is reported as measured. Conditions are read literally; if more than
one holds (e.g. NEITHER with VARIATION) all are reported, none is chosen between.

**Disposition.** The evidence file records the reading the data select. The OWNER adjudicates P3
shape 5. This arc is docs-only: no `src/`, `tests/` or `tools/` change. If a code or test change
seems needed, it goes in the row's REPORT.md and the lane stops.

**Unmeasured, stated up front.** Encoder device is whatever the checkout resolves (cuda on the
2026-10-20 runs); whether cpu reads the same numbers is not measured by this arc.

## 2. Results (2026-10-22)

Run: `driver.py` (row folder `D:\AI-project\lanes\insula-lvl-1022-row`), 5 seeds x 8 arms, 40 brains, one
subprocess each, rc 0 on all 40. Tier of every arm: `("eprop-snn", True, "cuda")`; all arms of a seed
share the device (cuda); the driver asserts it. Tier gate red shown (`SNN_EPROP_CKPT` set to a
nonexistent path): the harness REFUSED with rc 3 (`results/tier-gate-red.log`).

**Controls.** R = 0 (outcome) and D_thr(live2) = 0 on all 5 seeds, so the arm design is valid on this
tier. Z = D_out(x0 vs neutered) = 0 on all 5 seeds. Injected arms: same observed schedule as live, no
replay exhausted, D_thr from live > 0 (const 197-200, shuf 184-200, x2 197-200, x05 197-200, x0 195-198),
max |live_thr - law(own salience)| = 0.0 on every arm.

**Harness vs `_run`** (closing-review fold, 2026-10-22). `arm.py` is not byte-for-byte `_run`: it drops the
organ-sense spy, the three extra `interoception()` reads per tick and the `stress=-999` copy poison (accessor
checks the test covers, not the outcome), and it builds the brain under `throwaway_home()` where the test uses
conftest's `NORD_HOME` fixture. `driver.py` PRINTS the controls above and asserts only one tier per seed; it
does not assert R = 0 or the trace checks. The Opus reviewer and the lead each re-derived every control from
the raw JSON with their own code (trace equality and shuffle-not-identity included, error 0.0); seed 0
reproduces 6/9/20 through the same `random.Random(0)`, runtime evidence the arms match `_run`'s semantics.
The rc 3 of the tier-gate red is the harness's refusal exit (arm.py); the log holds the message, not the rc.

**Table** (D_out from live, 200 ticks):

| seed | N | C | S | x2 | x05 | x0 | Z | R | N > max(C,S) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 9 | 20 | 50 | 3 | 6 | 0 | 0 | no |
| 1 | 98 | 33 | 16 | 28 | 17 | 98 | 0 | 0 | yes |
| 2 | 112 | 48 | 48 | 27 | 18 | 112 | 0 | 0 | yes |
| 3 | 21 | 3 | 2 | 7 | 18 | 21 | 0 | 0 | yes |
| 4 | 0 | 45 | 45 | 43 | 3 | 0 | 0 | 0 | no |

Seed 0 reproduces the 2026-10-20 reading (N=6 C=9 S=20).

**What every count is** (closing-review fold MAJOR-1, re-derived by the lead). Every D_out tick above is a
WINNER change: the `ignited` sequence is identical across all 8 arms on every seed (seed 0 first ignites at
tick 29, seeds 2-4 at 26, seed 1 never), so no arm changes WHETHER ignition happens, only which coalition wins.
The injected arms' differences are transient: their last differing tick is 61 (seed 0), 47 (1), 53 (2), 48 (3),
47 (4). Removal is not: neutered (= x0) keeps differing until tick 176 on seed 1 and 199 on seed 2. An
ignition bit that never varies across arms has not been shown to be a property of the brain rather than of
this probe (the instrument rule); it is reported, not interpreted.

**Pre-registered readings, applied literally.**
- LEVEL-ONLY: does NOT hold (max(C,S) = 0 on 0 of 5 seeds).
- VARIATION matters: HOLDS (max(C,S) > 0 on 5 of 5 seeds).
- BOTH: HOLDS (max(L2,L05) > 0 on 5 of 5 seeds), so the pre-registered label is BOTH, not VARIATION-ONLY.
  Confound, stated as the lead flagged: x2 and x05 are constants, so they also destroy timing; whenever
  VARIATION matters, max(L2,L05) > 0 is expected regardless of level. The "level matters" part of BOTH is not
  separated from timing by the pre-registered design.
- NEITHER: does NOT hold (N = 0 on 1 of 5 seeds, bar is 3).
- Bar N > max(C,S) holds on 3 of 5 seeds (1, 2, 3); fails on seeds 0 and 4.

**Surprises.** N varies from 0 to 112 across seeds, so the insula's removal effect on the winner record is
seed-dependent. Seed 4, data only (closing-review fold MAJOR-2, re-derived by the lead; no cause claimed):
removing the insula moves the threshold on 198 of 200 ticks yet changes no winner (N = 0); const and shuf give
the SAME winner record (D_out(const, shuf) = 0, differing from live on the same ticks 3-47), so on seed 4 they
are one effect, not two; x2 differs from const on 3 ticks; x05 differs from live only around ticks 24-26.
D_out(const, shuf) on the other seeds: 11, 17, 5, 3. On every seed the live trace is 0.0 on its first two
consulted ticks; on seeds 1-4 the first consulted tick is 0, so live and neutered share threshold 0.55 there
while the constant arms start lower (0.446-0.458); on seed 0 the first consulted tick is 3 and all arms share
0.55 at tick 0.
x0 equals neutered on every seed (Z = 0), so the injection control is clean.

### 2a. POST-HOC (not pre-registered; added after data collection began, at the lead request)

The level test free of the timing confound: x2 and x05 vs const. All three are constants (no timing); only
the level differs. `posthoc.py`, same cached rows.

| seed | D_out(x2, C) | D_thr(x2, C) | D_out(x05, C) | D_thr(x05, C) |
|---|---|---|---|---|
| 0 | 43 | 197 | 8 | 197 |
| 1 | 36 | 200 | 37 | 200 |
| 2 | 28 | 200 | 30 | 200 |
| 3 | 6 | 200 | 21 | 200 |
| 4 | 3 | 200 | 45 | 200 |

The winner record changes with the constant's level, timing held out: both comparisons are > 0 on 5 of 5
seeds. Qualified (closing-review fold MINOR-4): winner only, early ticks only (see "What every count is"),
and not monotone in level (seed 4, D_out from live: x0 0, x05 3, C 45, x2 43); with no small-perturbation
control this shows the record depends on the constant's value, not a dose-response. D_thr is saturated
(about every tick) and carries no information here. This reading was not pre-registered and
does not replace the pre-registered one above. The OWNER adjudicates P3 shape 5.

**Unmeasured.** cpu device; seeds beyond 0-4; ticks beyond 200.

## 3. Owner adjudication (2026-10-22)

Owner ruling, in chat, on the pre-registered and post-hoc results above: **not P3 dormancy shape 5.** The
insula's tick-to-tick variation changes the outcome on the production tier (VARIATION matters on 5 of 5 seeds),
so "a constant wearing a signal's clothes" does not describe it.

**P8 question opened** (owner pick "Open a P8 question"). The charge measured here is narrow: winner-only (the
ignition sequence never differs across arms), the injected arms' effects end by tick 47-61, and the removal
effect is seed-dependent (N from 0 to 112). Whether that charge is load-bearing for the engine's function is not
settled by this probe. It is to be judged by the engine's function (owner 2026-09-30: a brain-tuned readout
recovering relational content on held-out data, against controls that break the path), and it owes what P8 asks
of an invented method: a reachable red, a control that is not the method itself, per-item attribution, and
deletion on a null. Not yet pre-registered; nothing measured.

## 4. Addendum (2026-10-23): what removal changes

Section 2's sentence "Every D_out tick above is a WINNER change: the `ignited` sequence is identical across all 8 arms" read
`_access_state["ignited"]`, a near-constant bit. The reentrant workspace's own ignition shows something different for REMOVAL:
of N's ticks, the workspace did not ignite at all under neutered (winner None) on 1 of 6 (seed 0), 61 of 98 (seed 1),
92 of 112 (seed 2) and 0 of 21 (seed 3); x0 matches neutered. The injected arms (const, shuf, x2, x05) change workspace
ignition on no seed, so their counts are winner changes, as written. Removal's charge is therefore partly a LEVEL effect (the
insula's salience keeps the workspace igniting on seeds 1-2), and the injected arms' charge is which coalition wins. The owner's
ruling in section 3 (not P3 shape 5) rests on the injected arms' VARIATION result, which this does not change; section 3's
phrase "winner-only (the ignition sequence never differs across arms)" holds for the injected arms, not for removal. Re-derived
by the lead from `D:\AI-project\lanes\insula-lvl-1022-row\results`; found by the Opus closing review of
`insula-p8-engine-reach.md` (section 5, item 4).
