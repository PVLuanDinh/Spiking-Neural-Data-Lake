# Agents — Spiking Neural Data Lake

**A clone has no `CLAUDE.md` (it is gitignored): read `docs/AGENT_COLD_START.md` before running anything
below.**

**Core dependencies:** `torch`, `numpy`, `pyarrow`, `mujoco==3.11.0`, `brian2==2.10.1`, `psutil`,
`threadpoolctl`, `scipy` and `networkx` (`pyproject.toml` `[project] dependencies`; the first five were the whole
list until 2026-10, corrected 2026-10-28). Installing also needs a **Rust toolchain**: `snn_data_lake._native` is
compiled through maturin at install time (`docs/INSTALL.md`). MuJoCo and Brian2 are pinned for reproducible physics/neural semantics.
Brian2 is the equation/event reference; Torch executes the parity-checked
`growth/brian2_backend.py` bridge. Custom kernels remain. Brian2 2.10.1 enables the GIL under
CPython 3.14t: real reference simulations run in isolated subprocesses; the parent stays free-threaded.
The zero-dependency-core contract was removed project-wide on 2026-07-20. **There is no torch-free
mode: the `NORD_TORCH=0` off-switch was REMOVED on 2026-07-29** together with the CI matrix job that
exercised it — if torch is absent the install is broken, not degraded. **Per P7 (2026-08-04) any
dependency that helps this become a phenomenal brain is accepted, and "it would be another
dependency" is not a valid reason to reject anything.**
*(Corrected 2026-08-04: this said `NORD_TORCH=0` "gives a torch-free brain". It gives an
`ImportError`. Grep says so — `engine/brain.py` carries the removal note at two sites.)*

**Governing policies:** `docs/POLICIES.md` — **P0 through P8: nine numbered policies, fourteen entries**
(P4b, P4c *and* P4d are sub-clauses of P4, P5a *and* P5b of P5; amendments uncounted), not P0–P5. **P5b**
(owner, 2026-10-17) is Python by default, Rust only where a six-criterion pre-registration measures it as better.
*(Recounted 2026-10-28: "thirteen" missed P5b.)* P0 is autonomous self-improvement and outranks the
rest; **P4b** the durable lake is part of the brain; **P4c** EVERY build is part of the brain — there
is no "outside the brain", and "it is only a helper / only the UI / only a benchmark" is not a
disposition; **P4d** (PROVISIONAL) the engine, two networks that grow to solve inside the one lake;
P7 is the dependency policy above; P8 is invent any method, but a success criterion with
no reachable RED is deleted, not weakened.
*(Recounted 2026-09-25: "eleven" was right when written and missed P5a, which landed after it.)*
*(Corrected 2026-08-04: this line said "P4b is a sub-clause" and omitted **P4c** entirely — the
policy under which the daemon and the cockpit are parts of the brain, and the one most likely to be
missed by an agent reading only this page.)*

**Mechanization, re-derived against `.github/workflows/ci.yml` and `run_selfchecks.py` on 2026-10-28:**
`run_selfchecks.py` runs every registered module's self-check, then seven tools gates (`TOOLS_GATES`):
`p4c_ratchet --assert`, `p6_gate --assert` (P6), `dependency_ledger --assert` (P7), `red_reachability --assert`
and `--self-check` (P8), `citation_guard` and `doc_command_guard --assert`. The **lint** job also runs
`native_ratchet`, `growth_coverage_gate`, `gradient_claim_gate`, `policy_manifest`, `null_scale_gate` (P8's
amendment) and `p5b_gate` (P5b), each `--assert` plus `--self-check`; the self-checks job runs the determinism,
replay, manifest, envelope-hash, region-charge and atlas-node gates. **The GitHub Actions workflow has been
disabled since 2026-08-20** (its last four runs failed), so none of this runs on GitHub today: the operator's
local battery (plan A.4 step 5) is the verification of record.
*(Corrected 2026-10-28, cloud audit: the paragraph this replaces said `dependency_ledger` was report-only and
that "only P6 is still gate specified, not built"; both gates assert, `p6_gate` since 2026-08-06, including its
connectome diff.)*
*(Earlier corrections of this paragraph, dated 2026-08-04 and 2026-08-05, are condensed out; read them with
`git show c1df3887:AGENTS.md`.)*

## Gates (all must pass)

```bash
python run_selfchecks.py                        # every module's self-check, then the seven tools gates
pytest                                          # >=70% line coverage gate
python -m ruff check .                          # rules E,F,I,UP,B
python -m ruff format --check src tests tools   # this exact scope is what CI runs
```

The full battery, with every gate's `--assert`, is the operator's block in plan A.4 step 5; `ci.yml` lists the
lint-job gates. *(Corrected 2026-10-28: this block listed `dependency_ledger` as report-only and
`doc_command_guard` as "NOT IN CI (P4c debt) ... unregistered"; `run_selfchecks.py` has run
`doc_command_guard --assert` in its tools stage since 2026-09-03, and `dependency_ledger --assert` is a tools
gate and a lint step.)*

Until 2026-08-05 this page was gitignored and untracked, so it existed in no clone and the gates that scan
`git ls-files` never saw it; it has been tracked since `bcad89e`. *(Condensed 2026-10-28 from the measured account,
whose document and command counts had gone stale; the full text is in `git show c1df3887:AGENTS.md`.)*
*(`CLAUDE.md` stays gitignored by owner decision: P4c's one named exception.)*

## Safety envelope (modify with care + re-test)

- `brainstem/`: `reflex.py`, `serial_arm.py`, `so101_arm.py`, `arm_config.py`, `interpreter.py`
- All motor commands → the `So101Arm.apply()` gate (`so101_arm.py`, `class So101Arm` → `def apply`;
  the arm is **6-DOF**, not the retired `arm_sim.py`'s 2). Real servos are additionally refused by
  `serial_arm.assert_calibrated` while `arm_config.SAFETY_CALIBRATED` is `False`.
  [SECURITY.md](SECURITY.md) has the re-test commands.
- *(Corrected 2026-08-04: this list named the non-existent `arm_sim.py` and omitted the live
  `so101_arm.py`, and the gate was named `ArmSim.apply()` — a class absent from `src/`. Envelope
  code unchanged; only its description was wrong.)*
- Run a module self-check with its FULL dotted path (`python -m snn_data_lake.brainstem.reflex`); the
  flat back-compat alias resolves imports but cannot execute under `-m`.

## Adding a module

Register it in **both** `src/snn_data_lake/__init__.py` `REGIONS` and `run_selfchecks.py`, then make it
reachable from a live path or add it to `gates/dormancy_gate.py` `EXEMPT` with a reason — Policy 3
fails CI otherwise.

**Determinism:** `random.Random(seed)`, `zlib.crc32()` for persisted hashes, ASCII-only output.

See `CLAUDE.md` (local-only, gitignored) for full onboarding details, layout, and optional extras.
