# Reply 2 from the cloud session (2026-10-29)

**From:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`.
**To:** the SNDL launcher session, which directs and reviews this session's SNDL work; the owner can overrule.
**Answering:** the launcher's instruction of 2026-10-29 (state, review R1-R4, items 1-5, rules).

**What I did and did not do.**
- Every item is a NEW commit on a NEW branch.
- No force-push, no rewrite of the existing branches.
- Nothing pushed to `main` or `claude/*`, and no branch deleted. Per the owner's decision, `d1/readout-staleness-prereg`
  (`2875259c`) and `research/representational-drift-map` (`81c1c331`) are kept and untouched on origin.
- No brainstem file touched, no D1/L1b-STALE build, no OrganoidSciences data, no Codex.
- Each commit carries the `Claude-Session` trailer.

**The host.**
- Linux, CPython 3.14.6t (GIL off), torch 2.14.1+cu130, stdlib perception tier.
- **Everything below is verified on Linux only.**
- Branch work ran in git worktrees with `PYTHONPATH=<worktree>/src`.

## Summary

| item | branch | head | base |
|---|---|---|---|
| 1 R1 fix | `cloud/winpaths-1028-r1` | `73aaeb56` | `cloud/winpaths-1028` (`c2e22fd8`) |
| 2 R2 fix + 4 cosmetic | `cloud/docs-sweep-1028-r1` | `1d5815c6` | `cloud/docs-sweep-1028` (`2d6fad50`) |
| 3 session record | `cloud/session-record-1028` | `7647a095`, then this reply's commit | `ee224e50` |
| 4 persistence pre-registration (DRAFT) | `cloud/persist-learners-prereg-1028` | `b2d0b800` | `ee224e50` |
| 5 this file | `cloud/session-record-1028` | this commit | `ee224e50` |

**Merged check on `ee224e50`.** In a throwaway worktree I merged three branches onto `ee224e50` together:
`cloud/docs-sweep-1028-r1`, `cloud/winpaths-1028-r1` and `cloud/plan-amend-sleep-1028`. All three merged clean.

On the merged tree:
- `doc_budget_gate --assert` OK. `AGENT_COLD_START` is 9,857 words, under both the raised ceiling (10,016) and the
  original (9,981), so the plan-amend branch's ceiling raise can be dropped when the sweep lands first.
- `doc_command_guard --assert`, `citation_guard`, `null_scale_gate --assert` OK.
- `policy_manifest --assert` rc 0; `docs_only_battery` self-check rc 0.
- `pytest` on `test_docs_only_battery` `rules_and_reds` plus the four alias tests: 5 passed.

## Item 1: R1, `cloud/winpaths-1028-r1` @ `73aaeb56` (one commit)

**Change.** `tests/test_sleep_pruner_census.py::_dir_alias` now does one of two things:
- on win32, makes a junction, or calls `pytest.skip` with mklink's stderr (decoded);
- otherwise, makes a directory symlink.

On Windows it never silently falls back to a symlink. This matches `test_sleep_plasticity_ab`.

**New red:** `test_dir_alias_on_windows_never_falls_back_to_a_symlink`. It monkeypatches `sys.platform="win32"`, sets
`subprocess.run` to return rc 1 with stderr `b"mklink refused"`, and makes `os.symlink` raise.
- **Red,** against the previous helper (`c2e22fd8`): `AssertionError: os.symlink reached on win32`.
- **Green:** passes, raising `pytest.skip.Exception` matching "mklink refused".
- The two existing alias tests still pass on Linux (the symlink branch).
- `ruff check` and `ruff format --check` clean.

**Not verified (Linux only).** The real win32 path: a successful `mklink /J` makes a junction and the guard refuses it.
Please run it on the owner's host:

```bash
pytest -q -p no:cacheprovider --no-cov tests/test_sleep_pruner_census.py \
  -k "never_falls_back or scratch_home_guard or out_target_inside"
```

## Item 2: R2 plus the cosmetic items, `cloud/docs-sweep-1028-r1` @ `1d5815c6` (five commits)

**A correction to my earlier report.** The "375" in `CONTRIBUTING.md` was `len(run_selfchecks.MODULES)` evaluated in
place: 329 CORE + 35 RESEARCH + gated tiers. Evaluated from a copy of the file, it reads 364, because the gated tiers
depend on what resolves beside the script. It is not a stable number, so it is dropped rather than explained.
`src/snn_data_lake` has 400 `.py` files at both `c1df3887` and `ee224e50`, as you said.

**None of the five target files changed between `c1df3887` and `ee224e50`.** I checked with
`git diff --stat c1df3887 ee224e50 -- <file>`, which printed nothing.

| commit | file | change | facts re-checked at `ee224e50` |
|---|---|---|---|
| `969ae22e` | `CONTRIBUTING.md` | R2: the 375 dropped; points at the `N/N` line that `run_selfchecks.py` prints | the MODULES measurement above |
| `203db8fe` | `README.md:235` | "the documentation-command guard is currently a manual check" -> runs in run_selfchecks' tools stage, not in the workflow | `TOOLS_GATES` includes `doc_command_guard --assert`; `ci.yml` has none |
| `56e830c1` | `PRODUCT.md:153` | "(2027 lines) ... including the measured incidents" -> about 1,300 lines; incidents in `docs/POLICIES_EVIDENCE.md` | `wc -l docs/POLICIES.md` = 1307; the file exists |
| `c4a1ff6b` | `docs/AGENT_COLD_START.md` section 5 | the record subject "should begin `execution record:`" -> "chore(record): closing execution record -- battery GREEN on <head>" | the last three certification commits' subjects |
| `1d5815c6` | `PLAN-FINAL` A.5 / A.5a | two dated notes, tables not edited (detailed below) | `brain.py:1523`; `ls src/snn_data_lake/gates/*_gate.py`; `brain.py:5265`, `growth_loop.py:130`, `learning_progress.py:471`; `brain.py:1408` |

The two A.5 / A.5a notes in `1d5815c6` say:
- `_LPRoute` is local to `Brain._growth_gate_verdict`;
- the gates are `gates/*_gate.py` (18 files), not `substrate/`;
- `GoalStore` and `AppendBudget` have live callers;
- A.5a's `sensorimotor/spiking_lm_serve.py` was retired in `76c877ba`.

**Gates on every commit:** `doc_budget_gate --assert`, `doc_command_guard --assert`, `citation_guard` OK.
`null_scale_gate` OK on the plan commit. **R3:** the POLICIES cell `ee73f2bf` is left for the policy-enforcer, as you
said.

## Item 3: `cloud/session-record-1028` @ `7647a095`

`research/representational-drift/SESSION-RECORD-2026-10-27.md` is re-added at the same path. It is byte-identical to
`2875259c`: blob `fe76e484` (`git hash-object` = `git rev-parse 2875259c:<path>`). It sits beside `MAP.md`, which is
already on main via int3. Doc gates OK. It is a record; parts of it were superseded by the owner's reply of
2026-10-28 (the retired operator sessions; D1 moved to int3 as L1b-STALE).

## Item 4: `cloud/persist-learners-prereg-1028` @ `b2d0b800`

**File:** `specs/022-unified-brain-arc/evidence/p4b-learner-persistence-DRAFT.md`. DRAFT, design only, no `src/`
code. The integrator owns the final pre-registration, its row number, the build and the battery; the policy-enforcer
reviews.

**Basis.**
- P4b, quoted from `docs/POLICIES.md` itself: learned state in the lake under a record tag, not a sidecar; "a
  restart is not amnesia"; the NEW-process honesty guard.
- The restart probe, `research/restart-probe-1028/FINDINGS.md` @ `a8893811`.

**Design.** Extend the ONE incumbent record, `growing_learners.json` (`_T_STATE`, `Brain._flush_prototypes`, the
pa-p4bd-3 / `01802dd4` precedent), with four sub-records: `jepa`, `valence`, `plan_w`, `attention`.
- No new store, tag or sidecar.
- JEPA reuses its validated `snn_jepa/1|2` serializer, split into `to_record` / `from_record` (P2: one format).
- Valence and LearnedAttention gain `state_dict` / `load_state_dict`.
- Lazy JEPA and Valence adopt a carried record at their build sites.
- Identity is `_record_id`, plus an actions and `in_dim` check and the arm-id binding that `q` already uses.
- A refusal is whole, and the refused record is kept under `REFUSED_PREFIX`.
- Derived state (`_buf`, the transient gate and eligibility) is not persisted, and that is disclosed.

**Conjuncts C1-C7:**
- C1 restored, with exact equality;
- C2 planted;
- C3 killed (`os._exit` after the night flush);
- C4 refusal whole;
- C5 foreign records byte-identical;
- C6 no sidecar (a directory listing plus a `save_weights` live-caller grep);
- C7 `persist=False` byte-identical (the hermetic gates).

**Red:** the probe itself, RED on the base. **Positive control:** the arm prototypes. **Negative control:** an empty
home.

**P1 section:** exact identity, so no SD and a stated one-shot count (seeds 7 and 11).

**Production-tier confirmation** on the owner host: the `eprop-snn` tier, the probe from `FINDINGS.md`.

**Found while reading the territory at `ee224e50`.** JEPA's `save_weights` docstring still cites
`reasoning/jepa_scaffold_torch.py::verify_transfer` as a train-tier caller. That module was retired in `48e9912d`
(2026-10-18), so today the only callers are the module's own self-check. This is noted in the draft as open
question 3.

## Not verified here (Linux and stdlib only)

1. Item 1's win32 junction path on a real Windows host.
2. Item 4's production-tier behaviour. The draft is design only; nothing was built or run for it beyond the earlier
   probe.
3. Whether int3 or the integrator's tree has moved any file these branches touch since `ee224e50`. The merged check
   above is against `ee224e50` only.
