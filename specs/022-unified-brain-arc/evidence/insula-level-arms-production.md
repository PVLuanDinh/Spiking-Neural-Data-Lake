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
