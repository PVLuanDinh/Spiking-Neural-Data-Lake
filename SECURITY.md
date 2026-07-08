# Security & Safety

This project simulates and can drive a physical 2-DOF robot arm. The arm control path is
safety-critical: a defect can command real servos. This document describes the safety envelope
and how to change it responsibly.

## Safety envelope

Five modules form the envelope. Every motor command flows through the `ArmSim.apply()` gate,
which enforces the emergency-stop latch, joint limits, and command grammar before anything
reaches a servo.

| File | Role |
|------|------|
| `src/snn_data_lake/interpreter.py` | validates and parses commands (allow-list grammar) |
| `src/snn_data_lake/arm_sim.py`     | physics simulator; mirrors and gates the live arm |
| `src/snn_data_lake/reflex.py`      | emergency-stop reflex (imminent contact / trajectory deviation) |
| `src/snn_data_lake/serial_arm.py`  | real-hardware serial interface |
| `src/snn_data_lake/arm_config.py`  | calibration constants and joint limits |

The real serial interface (`serial_arm.SerialArm`) refuses to drive hardware until
`arm_config.SAFETY_CALIBRATED` is set `True`, which must only happen after the placeholder
limits are replaced with the specific arm's measured values. There is no runtime bypass of
this gate.

## Changing the envelope

- Changes to any of the five files above require review and a re-run of their module
  self-checks (`python -m snn_data_lake.<mod>`) plus the arm test suite
  (`pytest tests/test_arm_sim.py tests/test_reflex.py tests/test_interpreter.py ...`).
- Do not weaken an assertion, remove a limit check, or add a bypass path to make a demo pass.
- The simulator's gate must stay at least as strict as the hardware path — the sim is the
  safety mirror.

## Reporting a vulnerability

Report security or safety issues privately to the maintainer at the address on the GitHub
profile (github.com/Aighluvsekks) rather than opening a public issue. Please include a
reproduction and the affected version.
