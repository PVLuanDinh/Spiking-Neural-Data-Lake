# hot-homestat REPORT (2026-10-19)

Route: investigate (lane hotfix). Branch claude/hot-homestat-1019, base d639bb87.

## Change
`src/snn_data_lake/substrate/home.py::_scan_home_entries`: `os.stat(e.path)` is now taken ONCE per file entry
before the `any(os.path.samestat(...))`, instead of once per live-home file. Refusal semantics unchanged
(alias handling, recursion, entry cap, hardlink refusal and message).

## Counts (os.stat calls from `_scan_home_entries`, scratch home N=40 files, live home M=25 files)
- before (HEAD function): 1000 (= N*M)
- after: 40 (= N)
Script: `.lane-tmp/count.py` (not committed).

## Tests (OMP/MKL threads=1, `--no-cov -p no:cacheprovider`, one file at a time)
Passing before AND after: test_home_guard (26), test_home_pytest_tripwire (8), test_world_red_probe_guard (3),
test_growth_loop (66), all rc=0. These include 3 existing hardlink-refusal tests in test_home_guard.py
(lines 192, 212, 239), so no new test was added (brief: add one only if none exists).
Failing identically before and after (FAILED/ERROR sets diff-equal, per file): test_g9_live_home_boot,
test_sleep_pressure, test_g5_readout_cells, test_r2_p3_percept_transitions, test_r2_replay_hydration,
test_r2_sleep_redo, test_r3_structural_hydration, test_replay_hydration_instrument, test_sleep_plasticity_ab,
test_sleep_pruner_census, test_staged_sleep_bout, test_taught_campaign (rc=1).
Cause seen: `ImportError: cannot import name '_native' from 'snn_data_lake'` at faculties/prefrontal_executive.py:129
(`_native.*` is gitignored and not built in this checkout or in main's src).

## Could not verify
- Those 12 files' Brain-level tests never reach `require_isolated_home` success paths here because `_native` is not
  built, so they do not exercise the change; I only proved the failing sets are unchanged. I did not diff failure
  messages beyond the FAILED/ERROR test ids.
- No wall-clock claim (battery running). Counts only.
- The pre-change run swapped the old file in temporarily (git stash is denied by a hook) and restored the new one;
  final file equals the committed fix.
