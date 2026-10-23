# G3 T0: baseline capture for the one-grower merge -- PRE-REGISTRATION

Status: FROZEN before any brain ran (lane g3-t0, 2026-10-23). Base commit: `20e1e742` (origin/main, the P8 arc landed).
Harness: `tests/test_g3_t0_baseline.py` (`capture()` is invoked explicitly; its fast tests recompute the verdicts below from the
committed `g3-t0-baseline.json`). Design: research-1016 DESIGN.md section 4, row T0. A results section is appended in a LATER commit;
this section is not edited after the freeze.

## 1. Question

The G3 merge (T1) must move no value on the percept path. "No difference" is worth only as much as the differencer's power, so T0 fixes
(a) what two identical runs already disagree on (the nondeterminism baseline) and (b) that the instrument sees a change made in the
SAME stage the merge touches (`_refit`). T1 and later rows diff against this capture, never against a fresh run.

Deciding statement: the harness is usable as T1's control iff, at EVERY seed in section 3, all of:

- H0a. the two `live` arms differ on NO key outside `{snap.t}` (the wall clock).
- H0b. `growth_off` moves `state.grown_units`, and the fit leaves `grown_units == 0`.
- H0c. `engine_seed_shift` moves the placement record AND moves at least one of {`growth_fingerprint`, `state.grown_units`,
  `state.predict_accuracy`} at at least one seed.
- H0d. `engine_admission_shift` (`_eps_tol = 0.05`) moves `state.predict_accuracy` or `state.grown_units`.
- H0e. `reverse_tiebreak` (`candidate_lifecycle.select_block` patched to `reverse=True`) moves `growth_fingerprint` at at least one seed.

Failure handling, fixed now:
- H0a fails: the baseline is more than the wall clock; T0 records the extra keys by name and T1's "0 keys beyond baseline" becomes
  "0 keys beyond this named set". Reported, not widened silently.
- H0c fails at every seed: the seed shift is not a usable same-stage control; the same-stage control becomes
  `engine_admission_shift` + `reverse_tiebreak`, and the evidence says so.
- H0d AND H0e both fail: the harness is BLIND to `_refit`; T1 must not start (report to the integrator).
- A control that does not fire is reported as a FAILED CONTROL with the arm's actual diff list, never reworded.
- Fire-first rule: the red arms are read before any T1 diff is accepted.

## 2. Perception tier

Every arm records `state()["perception_tier"]` and `state()["perceiving_audio"]` at construction and after the last tick. Expected
`eprop-snn`. Any arm reporting `stdlib` STOPS the capture (a launch fault, not a result). A tier change between the two `live` arms of
one seed is an H0a failure. The tier is printed in every table row; no number is quoted without it. T0 claims nothing about the
brain's competence: it measures the harness.

## 3. Seeds, ticks, arms

- Brain seeds 0, 1, 2. Ticks 300 per arm (`TICKS` of `tests/test_placement_charge_null.py`).
- Arms per seed (7 brains, 21 total): `live_a`, `live_b`, `growth_off` (`b.neuro.tick = lambda: None`), `engine_seed_shift`
  (`b.neuro.seed` 0->1, 1->2, 2->0, never a no-op), `engine_admission_shift` (`b.neuro._eps_tol = 0.05`), `neuter_place` (constant
  tuple `(0.5, "Layer 2", 1)`; captured as the existing pinned null, NOT a pass criterion), `reverse_tiebreak` (`select_block`
  patched to `reverse=True`).
- Per arm: `make_brain(seed=s, persist=False)` inside `throwaway_home()`; one Brain at a time; the live lake and daemon untouched.
- Bare-engine pins: the 8 `STREAM`/`TRAJ` and 6 `ONLINE` values of `tests/test_candidate_lifecycle.py` are re-captured on this commit
  and compared to the committed values. If any differs, T1's byte-identity target is the NEW value, recorded here.

## 4. Recorded per arm

(1) the flattened `state()` + `brain_telemetry.snapshot(b)` view (`_flatten` of the existing test; all keys, JSON-safe);
(2) `growth_fingerprint(b.neuro)`, `crc32(json.dumps(b.neuro.trajectory))`, `grown_units`, `predict_accuracy`;
(3) `list(b.neuro.placement)`; (4) perception tier, audio flag, `neuro.device`, `neuro.seed`, ticks, wall seconds;
(5) `b.neuro.state()`. Run-wide: base commit, interpreter, `torch.__version__`, host.
Reuse (no new mechanism): `_flatten`/`_diff` imported from `tests/test_placement_charge_null.py`, `growth_fingerprint` from
`growth/neurogenesis_engine.py`. New code: a driver `capture()` and the verdict recomputation, in the test module, none under `src/`.

## 5. Controls that must fire

| red | how | must |
|---|---|---|
| R-T0-1 growth_off | `b.neuro.tick = lambda: None` | `grown_units == 0` and `state.grown_units` in the diff vs `live_a` |
| R-T0-2 seed shift | `b.neuro.seed` shifted | placement record differs from live's AND fingerprint/`grown_units`/`predict_accuracy` moved at >= 1 seed |
| R-T0-3 admission shift | `_eps_tol = 0.05` | `state.predict_accuracy` or `state.grown_units` in the diff |
| R-T0-4 reversed tie-break | patch `select_block` | `growth_fingerprint` differs at >= 1 seed |
| R-T0-5 baseline | `live_a` vs `live_b` | diff `== {snap.t}` |
| R-T0-6 blind-harness plant | the differencer fed two identical views, one value edited by +1 ulp | the diff names that key |

R-T0-6 is a fast test that needs no brain. The verdict tests recompute H0a-H0e from the committed JSON.

## 6. Cost and scope

Estimate (not a measurement): ~10 min on a quiet host, ceiling 2 h loaded. Runs under `resource:timing` (claimed `--ttl 3h`, released
after). No GPU; refits route to CPU at this width.

P8: T0 asserts no null. If the harness is declared BLIND (H0d and H0e fail) that concerns THIS width (<= 2 refits per arm at 300
ticks); the named scale axis is `ticks` (re-test at 3000, ten times) before T1 is declared un-gateable. Underpowered = inconclusive.
P5b: Python (no kernel added).

## 7. Does NOT establish

- No claim the merged grower is identical: T0 only makes the claim testable; T1 is the claim.
- No claim about competence or the value of growth. Held-out accuracy appears only as a differencer key.
- Seeds 0-2 at 300 ticks reach one or two grown units; a fingerprint over so few catches a changed tie-break only if it changes which
  of a few candidates is admitted. The bare-engine pins carry the strong half of T1's gate.
- Nondeterminism across processes, GPU or a differently loaded host is not sampled.
