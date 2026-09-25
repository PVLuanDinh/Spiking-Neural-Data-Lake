# Agents — Spiking Neural Data Lake

**A clone has no `CLAUDE.md` (it is gitignored): read `docs/AGENT_COLD_START.md` before running anything
below.**

**Core dependencies:** `torch`, `numpy`, `pyarrow`, `mujoco==3.11.0`, and `brian2==2.10.1`
(`pyproject.toml`). MuJoCo and Brian2 are pinned for reproducible physics/neural semantics.
Brian2 is the equation/event reference; Torch executes the parity-checked
`substrate/brian2_backend.py` bridge. Custom kernels remain. Brian2 2.10.1 enables the GIL under
CPython 3.14t: real reference simulations run in isolated subprocesses; the parent stays free-threaded.
The zero-dependency-core contract was removed project-wide on 2026-07-20. **There is no torch-free
mode: the `NORD_TORCH=0` off-switch was REMOVED on 2026-07-29** together with the CI matrix job that
exercised it — if torch is absent the install is broken, not degraded. **Per P7 (2026-08-04) any
dependency that helps this become a phenomenal brain is accepted, and "it would be another
dependency" is not a valid reason to reject anything.**
*(Corrected 2026-08-04: this said `NORD_TORCH=0` "gives a torch-free brain". It gives an
`ImportError`. Grep says so — `cortex/brain.py` carries the removal note at two sites.)*

**Governing policies:** `docs/POLICIES.md` — **P0 through P8: nine numbered policies, thirteen entries**
(P4b, P4c *and* P4d are sub-clauses of P4, P5a of P5; amendments uncounted), not P0–P5. P0 is autonomous self-improvement and outranks the
rest; **P4b** the durable lake is part of the brain; **P4c** EVERY build is part of the brain — there
is no "outside the brain", and "it is only a helper / only the UI / only a benchmark" is not a
disposition; **P4d** (PROVISIONAL) the engine, two networks that grow to solve inside the one lake;
P7 is the dependency policy above; P8 is invent any method, but a success criterion with
no reachable RED is deleted, not weakened.
*(Recounted 2026-09-25: "eleven" was right when written and missed P5a, which landed after it.)*
*(Corrected 2026-08-04: this line said "P4b is a sub-clause" and omitted **P4c** entirely — the
policy under which the daemon and the cockpit are parts of the brain, and the one most likely to be
missed by an agent reading only this page.)*

**Mechanization, re-derived against `.github/workflows/ci.yml` on 2026-08-05:** the three P3 gates
and the P4 replication guard run in `run_selfchecks`. The **lint** job additionally runs
`tools/citation_guard.py`, **`tools/red_reachability.py --assert` (P8 — ASSERTING)** and
`tools/dependency_ledger.py` (P7 — report-only pending specs/010 T021/T062). **Only P6 is still
"gate specified, not built"** — `tools/p6_gate.py` exists, runs and self-checks, but is manual and
non-asserting, and its connectome-diff half is genuinely absent.
*(Corrected 2026-08-04: this page said "P6, P7 and P8 are review-enforced and PROVISIONAL — gates
specified, not built" while P8's gate was already asserting in CI. **Corrected again 2026-08-05:
that same fix added `tools/doc_command_guard.py` to this list under the heading "re-derived against
ci.yml", and it is NOT in ci.yml.** Grepping `doc_command_guard` across `.github/workflows/`,
`run_selfchecks.py`, `.pre-commit-config.yaml` and `pyproject.toml` returns zero matches. The gate
is built and clean; nothing runs it. A sentence claiming the verification that would have caught it
is the most expensive kind of wrong in this file.)*

## Gates (all must pass)

```bash
python run_selfchecks.py                        # every registered module's __main__ self-check
pytest                                          # >=70% line coverage gate
python -m ruff check .                          # rules E,F,I,UP,B
python -m ruff format --check src tests tools   # this exact scope is what CI runs
python tools/citation_guard.py                  # CI lint job: cited .py files are tracked
python tools/red_reachability.py --assert       # CI lint job: P8, no unfalsifiable criterion
python tools/dependency_ledger.py               # CI lint job: P7 ledger (report-only today)
python tools/doc_command_guard.py --assert      # NOT IN CI (P4c debt): documented `python -m` runs
```

The last one is **built, clean and unregistered** — run it by hand until it is a CI step. Under P4c
a build no gate can see is not a build this project can claim, so registering it is open work.

**And that sentence applied to THIS PAGE, which is why the paragraph above could not see itself.**
Until 2026-08-05 `AGENTS.md` was gitignored — measured `git check-ignore -v AGENTS.md` ->
`.gitignore:76:/AGENTS.md`, and `git ls-files --error-unmatch AGENTS.md` -> *"did not match any
file(s) known to git"*: ignored **and** untracked, so this document existed in no clone. Two of the
gates listed above were blind to it as a direct consequence. `tools/citation_guard.py` scans
`git ls-files` only, so it has never checked one citation on this page. `tools/doc_command_guard.py`
names `AGENTS.md` **literally in its own `SCOPE` tuple**, so it has been counting a document no
clone has — re-run 2026-08-05, it prints `scanned 55 documents; 51 python -m command(s)` here, of
which this page supplies 1 document and 1 command: the dotted module self-check quoted in the
safety-envelope section below. (The three `ruff` invocations in the gate block are not counted —
the gate's `CMD` regex matches the dotted-package form only. **This sentence deliberately does not
re-quote that command**: writing it a second time made the gate print *52*, so a paragraph
describing a count silently changed it.) **A correction that fell out of re-running it:** `docs/POLICIES.md`
gives the clean-clone figure as *54 / 50*; the command count is right and the document count is off
by one, because **`CLAUDE.md` is in that same `SCOPE` tuple and is likewise gitignored and
untracked** (0 counted commands). The clean-clone figure is **53 / 50**.
The ignore rule is now removed, and this page has been tracked since `bcad89e` (2026-08-05).
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
reachable from a live path or add it to `substrate/dormancy_gate.py` `EXEMPT` with a reason — Policy 3
fails CI otherwise.

**Determinism:** `random.Random(seed)`, `zlib.crc32()` for persisted hashes, ASCII-only output.

See `CLAUDE.md` (local-only, gitignored) for full onboarding details, layout, and optional extras.
