# ENC -- the brain's gitignored model assets resolve against the repository root, not the cwd

Owner pick 2026-10-20 ("1 + hook"), after the B3 arc found that the intero "cross-test leak" was a
perception-tier difference (B3 record a676baf9; tests/test_interoceptive_ignition.py 18b529cd).

## 1. Protocol (pre-registered 2026-10-20, BEFORE the change is built)

**The defect.** `engine/brain.py` reads five gitignored model assets through `os.environ.get(ENV,
"models/...")`: `SNN_EPROP_CKPT` (eprop_nmnist.pt), `SNN_PERCEPTION_PACK` (nmnist_pack.npz),
`SNN_EPROP_AUDIO_CKPT` (eprop_shd.pt), `SNN_SHD_PACK` (shd_pack.npz), `SNN_PC_CKPT`
(pc_cortex_nmnist.json). The default is relative, so it resolves against the CURRENT WORKING
DIRECTORY. A process whose cwd is not the repository root silently builds the no-encoder brain
(`perception_tier` "stdlib", no audio encoder). Measured on 682c54fe (intero bar, seed 0, 200
ticks, N/C/S): no encoders 28/0/0, vision only 102/8/28, audio only 21/0/4, both 6/9/20.

**The change.** One helper: an env override is used exactly as given; the DEFAULT resolves against
`engine.wiring_manifest.REPO` (the engine's existing `Path(__file__).parents[3]`; no new root
constant -- nine private `_repo_root()` copies already exist, P2). All five sites use it. No new
`state()` key: `perception_tier` (vision) and `perceiving_audio` (audio) already report the tier.

**What it does NOT change (stated so nobody over-reads the arc).**
- The DAEMON: `learn_daemon` already `os.chdir(_repo_root())` and `learn_daemon.bat` does `cd /d`
  the repo root, so the daemon's brain was already the production (both-encoder) brain.
- LANES: `REPO` is the root of the checkout that was imported. A lane worktree without
  `models/eprop_*.pt` still builds the no-encoder brain. The cwd input is removed; the CHECKOUT
  input remains (obs 2534 stays open: lanes junction models/ or export absolute SNN_EPROP_* paths).
- Env overrides: unchanged, including relative ones (an explicit relative override is the caller's
  intent and stays cwd-relative).

**Who DOES change.** Every process that builds a brain with cwd != repo root on an encoder-bearing
checkout:
- `run_selfchecks.py` in-process checks: `os.chdir(scratch)` per check (run_selfchecks.py ~1102).
  Before: no-encoder brain. After: production brain. Subprocess checks run with `cwd=_ROOT`
  (~848) and were already on the production tier.
- Six test files `chdir`: test_carried_state_campaign, test_carried_state_driver,
  test_dependency_ledger, test_inseparable_mapping, test_instrument_guard, test_red_reachability.

**Reachable red (the test, tests/test_perception_assets_root.py).** `monkeypatch.chdir(tmp_path)`,
`make_brain(seed=0, persist=False)` under conftest's isolated home: assert `_perception_tier ==
"eprop-snn"` and `_audio_enc is not None`. RED on the parent (ad446c70): the tmp cwd has no
`models/`, tier "stdlib". Skips BY NAME where the encoders are absent at `REPO/models` (CI, lanes).
**Control (not the method):** with `SNN_EPROP_CKPT` set to a nonexistent absolute path the same
brain reports "stdlib" -- the env override still governs, and the positive assertion is not vacuous.

**Impact census (measured, not assumed), on the fixed head BEFORE the battery:** one bare
`run_selfchecks.py` run. Record per the cross-cutting rule "a performance regression can be
correctness-green":
1. PASS/FAIL/SKIP counts vs the last bare run (0d6385a0: 380 PASS, 1 SKIP, 1 FAIL = CRLF artefact).
2. Wall time vs that run's ~25 min (indicative only: different head, unloaded machine both times).
3. GPU memory across the run (nvidia-smi sampled every 30 s): does it climb monotonically, i.e. do
   in-process brains leak their cuda encoders past `Brain.close()`?
This run is a MEASUREMENT, not the battery; the arc still owes its one battery.

**Disposition for a check that goes red on the production tier (written before the run).** The red
is attributed to that check by name and re-run alone on the parent and the fix. The fix is NOT
reverted to hide it, the check's assertion is NOT retuned to the new numbers, and the owner decides
(B3 precedent: a check calibrated on the degraded tier reads like a regression caused by this
commit). A red that is a resource effect (cuda memory, timing) is reported as such, separately.

**Null disposition.** If the census shows NO check changes outcome, the arc still stands on its red
(cwd-independence) and records "no in-process check was tier-sensitive" -- itself a finding about
how little of the self-check surface exercises perception.

## 1b. Pre-data amendment (2026-10-20, after the Opus closing review, BEFORE any census data)

**The census as written could not see the change.** A BARE `run_selfchecks.py` dispatches to
`run_all` (run_selfchecks.py:1203), which runs every check as a SUBPROCESS from the launcher's
directory -- the repo root -- so those brains already loaded the encoders. The changed path is reached
only by `run_all_in_process` (:1049, `os.chdir(scratch)` at :1102), i.e. `--in-process`, `--timing-only`
and `--battery` (:1191-1200: CORE+TORCH in-process, the rest as subprocesses). Section 1 also cited
:848 (`cwd=_ROOT`), which is `run_tools_gates`, not the module checks. A bare census was LAUNCHED
(sndl-runs/.../enc-census-0877c0de) and STOPPED before it finished once the reviewer showed this; its
partial output is void and is not read.

**Amended census.** The arc battery's own self-check stage (`coverage run run_selfchecks.py --battery
--defer-timing`, the in-process path) IS the census. It is compared CHECK BY CHECK with the B4
battery's stage (battery-b4-1020-b54720f9: same mode, same 7-worker overlap; its src equals ad446c70's,
the parent): PASS/FAIL per check, per-check wall time from the runner's `[N.Ns]` tags, and GPU memory
sampled every 30 s by nvidia-smi during the battery (read-only; no brain). The section-1 disposition
for a tier-sensitive red is unchanged.

**Corrected test census.** Of the six files named in section 1, three `chdir` at runtime
(test_carried_state_campaign.py:362, test_carried_state_driver.py:180, test_inseparable_mapping.py:528)
and none of them builds a Brain afterwards; the other three only mention chdir (comments / scanned
fixture text). Prediction: no existing pytest test changes tier.

**Reviewer predictions, recorded so the census can falsify them:** likely to change outcome --
faculties.global_workspace / access_consciousness / reentrant_workspace (ignition counts),
gates.amdahl_gate (perception inside the tick time); possible -- gates.capability_bar,
gates.dormancy_gate, telemetry/cockpit values, engine.brain claim 8 (now the pc-cortex branch,
expected PASS); unchanged -- perf_baseline, sleep_plasticity_ab, memory_diagnostic.
