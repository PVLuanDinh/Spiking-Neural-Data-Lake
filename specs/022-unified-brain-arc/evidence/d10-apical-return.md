# D10 apical-compartment entry for the readout's return: build evidence

Binding pre-registration: `lanes\tickets\DESIGN-ROWS-1106\D10\PREREG.md` sha256 4fc3ff89c2e9 (incl. A1-A3);
reviewer-4 GO #8709, enforcer AGREE #8722 (conditions #8683 (a)-(c)). Worktree `b2-d10-1106`, base int3
`2f5a9c22`; control worktree `b2-d10ctl-1106` detached at `2f5a9c22` (read-only). No campaign or pilot run here.

## 1. TDD RED (before the implementation), 2026-11-06

Head `2f5a9c22dd305da70651921a5a1ce9c896d7142e` + the new tests only (grown_code.py / posed_problem.py
unmodified). `snn_data_lake.__file__` = `D:\AI-project\lanes\b2-d10-1106\src\snn_data_lake\__init__.py`.
Command (worktree root, PYTHONPATH=<worktree>/src, PYTHONUTF8=1):
`python -m pytest --no-cov -p no:cacheprovider -q -s tests/test_grown_code_apical.py
"tests/test_g9_posed_actor.py::test_apical_return_changes_brains_own_next_answer_untrained_and_random_do_not"`
Result: 11 failed, 2 passed. Each red, quoted:
- `test_apical_alone_never_fires` (R1): `assert gc.latency(R1_UNIT, [0], TAU, afferents=R1_AFF) is None` -> `assert 1 is None`.
- `test_apical_gains_subthreshold_basal` (R2 unit): drive == gate -> `assert 1.3237453538768933 == (0.8824969025845955 * (1.0 + 0.3799489622552249))`.
- `test_inhibitory_return_shunts`: `assert gc.latency(unit, [0], TAU, afferents=aff) is None` -> `assert 1 is None`.
- `test_planted_disconnected_apical_is_caught` (R3): `AssertionError: gain=1 plant was not caught`.
- `test_column_latencies_uses_apical`: `assert {0: 1, 1: 1} == {0: None, 1: None}`.
- `test_nonpositive_target_threshold_refused[0.0|-1e-18|-1.0]`: `Failed: DID NOT RAISE ValueError`.
- `test_chat_context_refuses_nonpositive_threshold_before_any_drive`: `Failed: DID NOT RAISE PosedRefused`.
- Live-brain replacement: `AssertionError: the return alone flipped the answer` / `assert 'GREET' == 'INTEROCEPT'`
  (isolation probe; the additive entry fires a zero-basal unit). Fixture: i=51 j=28 W=3, B/theta=0.9000000000000001,
  A/theta=87.80999640936221 (tanh saturated: a fixed ~2x step, as disclosed).
- `test_no_afferents_path_byte_identical`: red was a TEST bug (`AttributeError: 'int' object has no attribute 'hex'`,
  an empty sum is int 0); fixed to compare `repr`. It is a regression guard: green on the unmodified code by design.
- Green pre-implementation by design: `test_additive_fires_on_isolation_fixture` (R4 reference arm) and
  `test_zero_apical_is_byte_identical` (A=0 regression guard).

## 2. gain=1 plant RED on the replacement live-brain positive (enforcer #8683 (a)), BEFORE the green run

2026-11-06T18:24-0600, worktree head `2f5a9c22dd305da70651921a5a1ce9c896d7142e` + D10 working-tree changes
(implementation in). Command: `bash lanes/tickets/DESIGN-ROWS-1106/D10/run_plant_r4.sh` (section PLANT):
`PYTHONPATH="<wt>/src;lanes/tickets/DESIGN-ROWS-1106/D10" python -m pytest --no-cov -p no:cacheprovider
-p plant_gain1 -q -s "tests/test_g9_posed_actor.py::test_apical_return_changes_brains_own_next_answer_untrained_and_random_do_not"`
(`plant_gain1.py` forces `grown_code._apical_gain` to 1.0 for the test). Output: RED, rc=1:
`AssertionError: {'intact': 'SAFETY', 'target': 'GREET', 'neutered': 'SAFETY', 'baseline': 'SAFETY', ...}` --
intact == neutered == baseline. Fixture line: `D10 fixture: i=51 j=28 W=3 B/theta=0.9000000000000001`;
`D10 probe: A/theta=87.80999640936221 intact=SAFETY baseline=SAFETY`.

## 3. R4: the additive-era test 437, archived and runnable on the 2f5a9c22 control worktree (#8683 (b))

Re-run command (anyone): from `D:\AI-project\lanes\b2-d10ctl-1106` (detached `2f5a9c22dd305da70651921a5a1ce9c896d7142e`),
`PYTHONPATH=D:/AI-project/lanes/b2-d10ctl-1106/src PYTHONUTF8=1 python -m pytest --no-cov -p no:cacheprovider -q
"tests/test_g9_posed_actor.py::test_learned_return_changes_brains_own_next_answer_untrained_and_random_do_not"`
(also `run_plant_r4.sh` section R4). `snn_data_lake.__file__` = `D:\AI-project\lanes\b2-d10ctl-1106\src\snn_data_lake\__init__.py`.
Result 2026-11-06T18:24-0600: 1 passed, rc=0 -- the additive entry's return alone flips the brain's answer.

## 4. GREEN

2026-11-06, after sections 2-3. Command: `bash lanes/tickets/DESIGN-ROWS-1106/D10/run_green.sh` (worktree
PYTHONPATH, `snn_data_lake.__file__` = the worktree's), files: tests/test_grown_code_apical.py,
tests/test_g9_posed_actor.py, tests/test_grown_code.py, tests/test_column_batch.py, tests/test_g9_growth_adapter.py,
tests/test_g9_live_home_boot.py. Result: rc=0, 80 passed (progress dots; addopts -q suppresses the summary line),
0 failed. Replacement positive: `D10 fixture: i=51 j=28 W=3 B/theta=0.9000000000000001`,
`D10 probe: A/theta=87.80999640936221 intact=GREET baseline=SAFETY` (neutered == baseline, isolation query at its
baseline, both controls and the in-test gain=1 plant at baseline).

## 5. P3 runtime-use check (PREREG section 3; reviewer-4 #8870 blocking item 1), 2026-11-06

Probe (outside the repo): `lanes/tickets/DESIGN-ROWS-1106/D10/p3_probe.py` (sha256 1819417a027d...).
Command, from that dir, after the lock check + `lightrun_guard.sh` = `GUARD: clear` + board LIGHTRUN start/end:
`PYTHONPATH=D:/AI-project/lanes/b2-d10-1106/src PYTHONUTF8=1 python p3_probe.py D:/AI-project/lanes/b2-d10-1106
lanes/tickets/DESIGN-ROWS-1106/D10/p3_probe_result.json` -> rc=0, verdict OK. Result JSON:
`lanes/tickets/DESIGN-ROWS-1106/D10/p3_probe_result.json` sha256 07aa9abb81a9b08939aea317de0cacc7591a7e7df65f2cbbbf739ed27b5db739.
`snn_data_lake.__file__` = the worktree's src. Throwaway NORD_HOME via `substrate.home.throwaway_home()` (a temp dir;
it reported "left behind after cleanup" -- a Temp dir, never the live lake). make_brain(seed=0, persist=True); 800
`tick()`; `_bind` (test_g9_posed_actor fixture, problem `ii.xor` from PILOT_PROBLEMS per A3); one
`learn_part_d_episode`; one `read_part_d_episode` (label-free turns, masks off, as test_g9_posed_actor.py:160-167).
sys+threading setprofile, cumulative call counts per phase:

| phase | grown_code._drive | grown_code._apical_gain | GrownReturnPort.context | criticality_regulator.py |
|---|---|---|---|---|
| after make_brain | 0 | 0 | 0 | 1595 |
| after 800 ticks | 0 | 0 | 0 | 5211 |
| after bind | 0 | 0 | 1 | 5211 |
| after teach | 36 | 36 | 3 | 5211 |
| after read | 64 | 64 | 5 | 5211 |

- Known positives: criticality_regulator = 5211 calls (non-zero); GrownReturnPort.context after the teach = 2
  (non-zero). `_drive` and `_apical_gain` = 36 after the teach, 64 after the read. Not VOID.
- `tick()` alone reaches the apical entry 0 times: the return path is EVENT-driven (posed teach/read), as PREREG
  section 3 states. A tick-only profile would read 0 on a live path.
- Theta range of the bound units' birth rows: n = 2, min = median = max = 0.5. These rows come from the
  `_bind` fixture (constructed GrownUnit theta = 0.5), NOT from organic growth; the real grown-unit theta range is
  owed by the pilot (PREREG :109/:334).
- A/theta (max over each target's window, after teach + read): 0.3115 for both targets -> gain 1 + tanh = 1.302.
  At this fixture scale (fixture weights 0.2, theta 0.5) the gate is GRADED, unlike the live-brain positive's
  saturated A/theta = 87.8. Return weights after the read: 18/18 non-zero, 0.2 .. 0.20000000000000234.
- Wall 95.17 s; peak working set 1514.9 MB.
- First attempt rc=1 was a PROBE bug, not a finding: the read passed labelled turns and `posed_source` refused it
  ("OFF reads must be label-free; teaching label is probe-only"); fixed to label-free turns and re-run.

## 6. Disclosures and deviations (reviewer-4 #8870 non-blocking a-f)

- (a) The live-brain positive's first assert (`intact == target and neutered == baseline`) ALSO passes on the
  additive code (the RED run failed later, at the isolation probe). Apical-vs-additive discrimination in that test
  comes from the isolation probe (the same learned return alone must not flip a zero-basal query) and the gain=1
  plant, not from the intact/neutered pair.
- (b) DEVIATION: PREREG section 6.3 / A2.C1 say the byte-identity tests compare against recorded pre-change
  fixtures. `test_zero_apical_is_byte_identical` compares the afferents={}/empty path with the afferents=None path of
  the new code, and `test_no_afferents_path_byte_identical` compares with a test-local base-only reference and checks
  latencies + `_drive` only (not segment_psp / column_latencies). Low risk: the afferents=None branch is textually
  unchanged (`if afferents is None: return base`).
- (c) DEVIATION: section 6 says every test is seen red before the implementation. Three were green by design:
  `test_additive_fires_on_isolation_fixture` (R4 reference arm), `test_zero_apical_is_byte_identical` and
  `test_no_afferents_path_byte_identical` (regression guards). Recorded in section 1.
- (d) Sign behaviour change: with a NEGATIVE basal drive, an excitatory return now multiplies it (up to 2B, more
  negative), where the additive entry would have raised it. No crossing results (theta > 0 is enforced), but the
  sign behaviour of the drive differs.
- (e) A NaN theta gets the apical refusal message (`apical_targets_ok`: NaN > 0 is False) before `_check_unit`'s
  finiteness message. Cosmetic; both refuse.
- (f) A/theta ~ 88 in the live-brain positive makes it a fixed ~2x step; gradedness is shown at unit level
  (`test_apical_gains_subthreshold_basal`, A/theta ~ 0.40) and in the P3 run above (A/theta 0.31, gain 1.30 with the
  `_bind` fixture). The real-growth A/theta scale stays owed by the pilot.
