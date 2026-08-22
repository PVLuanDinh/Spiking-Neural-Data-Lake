# Security & Safety

This project simulates and can drive a physical **6-DOF SO-101** robot arm. The arm control path is
safety-critical: a defect can command real servos. This document describes the safety envelope
and how to change it responsibly.

## Safety envelope

Five modules form the envelope. Every motor command flows through the `So101Arm.apply()` gate
(`src/snn_data_lake/brainstem/so101_arm.py`, `class So101Arm` → `def apply`), which enforces the
emergency-stop latch, joint limits, and command grammar before anything reaches a servo.

*(Corrected 2026-08-04, twice over. First: this line named `ArmSim.apply()`, a class that does not
exist anywhere in `src/`, so the security document's one-line summary and its own table disagreed
about which code gates every motor command. Second, in the same pass: the opening sentence still
said **2-DOF**, a fact about the retired `arm_sim.py` — `so101_arm.py` asserts `len(theta_deg) == 6`
in its own self-check. The `:144` / `:268` line numbers were also dropped in favour of the SYMBOLS,
because a line citation in a safety document rots silently on the next edit and nothing in CI
resolves it. Nothing in the envelope changed — only its description; that is its own hazard, since
this is the paragraph telling a reader which code is approval-gated.)*

| File | Role |
|------|------|
| `src/snn_data_lake/brainstem/interpreter.py` | validates and parses commands (allow-list grammar) |
| `src/snn_data_lake/brainstem/so101_arm.py`   | the body and its gate: MuJoCo SO-101, every motion passes through `apply()` |
| `src/snn_data_lake/brainstem/reflex.py`      | emergency-stop reflex (imminent contact / trajectory deviation) |
| `src/snn_data_lake/brainstem/serial_arm.py`  | real-hardware serial interface |
| `src/snn_data_lake/brainstem/arm_config.py`  | calibration constants and joint limits |

**The E-STOP latch is durable across restarts** (specs/022 T037a; owner decision OD-4, delegated
2026-08-19). The daemon restarts on every commit, so a latch that lived only in the process was
cleared by a routine automated event. On the E-STOP edge `So101Arm.apply()` also writes a sidecar
(`so101_arm.ESTOP_LATCH` in the persistence home -- the name lives in the envelope so an edit
outside the approval gate cannot move or void it; beside the lake, never in it: safety state must
be readable when the lake is not). A `So101Arm` built by a later process reads it at construction
and comes up stopped, reporting `latch_source == "restored"` and the writer
(`latch_restored_from`) in its state and on the cockpit's E-STOP row. Who may clear what is decided
at the gate (`apply(cmd, deliberate=...)`) and nowhere else: a stop this process tripped clears on
`HOME` / `RETRACT_ALL` from anyone, as before; a RESTORED stop clears only on a deliberate reset --
`deliberate=True`, passed by the cockpit's actuation ticket and by its chat route ONLY when a person
typed the message (`Brain.respond(text, human=True)`; the daemon's transcript ingest calls the same
route with the default and an ingested arm intent never actuates at all -- the review found ordinary
prose such as 'reset the counter' reaching `HOME` through the NL firewall) and by nothing automated --
so the loop's self-clear, a `SerialArm` mirror's construction `HOME` and the reflex's `RETRACT_ALL`
are refused on it, the arm stays frozen, and the refusal is counted (`latch_refusals`). The sidecar is owned: a body that never held the stop leaves it untouched.
Fail-closed: an unreadable home or corrupt sidecar is a stop; a sidecar that cannot be written
leaves the in-process latch standing and is counted (`latch_persist_errors`). Scope: honoured across
processes (a per-process random token is recorded -- not the pid, which a reboot reuses); a body
rebuilt inside the same process is a new body, not a restart. Every self-check that drives the gate
runs under `substrate.home.throwaway_home()`, so the commands below never write the live brain's
home.

The real serial interface (`serial_arm.SerialArm`) refuses to drive hardware until
`arm_config.SAFETY_CALIBRATED` is set `True`, which must only happen after the placeholder
limits are replaced with the specific arm's measured values. There is no runtime bypass of
this gate.

## Changing the envelope

- Changes to any of the five files above require review and a re-run of their module
  self-checks plus the arm test suite. Use the FULL dotted path — the flat back-compat alias
  (`python -m snn_data_lake.reflex`) raises
  `AttributeError: '_AliasLoader' object has no attribute 'get_code'`, because the alias layer
  resolves imports but cannot execute a module as `-m`:

  ```bash
  PYTHONPATH=src python -m snn_data_lake.brainstem.reflex
  PYTHONPATH=src python -m snn_data_lake.brainstem.so101_arm
  PYTHONPATH=src python -m snn_data_lake.brainstem.interpreter
  PYTHONPATH=src python -m snn_data_lake.brainstem.serial_arm
  PYTHONPATH=src python -m snn_data_lake.brainstem.arm_config
  pytest tests/test_so101_arm.py tests/test_reflex.py tests/test_interpreter.py tests/test_arm_config.py tests/test_serial_arm.py tests/test_arm_vocabulary.py
  ```
- Do not weaken an assertion, remove a limit check, or add a bypass path to make a demo pass.
- The simulated body's gate (`So101Arm.apply()`) must stay at least as strict as the hardware path
  — the sim is the safety mirror.

## Reporting a vulnerability

Report security or safety issues privately to the maintainer at the address on the GitHub
profile (github.com/Aighluvsekks) rather than opening a public issue. Please include a
reproduction and the affected version.
