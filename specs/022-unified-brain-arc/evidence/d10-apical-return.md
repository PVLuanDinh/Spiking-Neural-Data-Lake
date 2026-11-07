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
