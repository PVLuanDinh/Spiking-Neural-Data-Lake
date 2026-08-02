# Command-vocabulary sites (T050)

**Status 2026-08-01.** The spec called the vocabulary "a closed string set defined in three
places that must agree". The survey found **eight**, and the three the spec named were not the
three that mattered most — two of the unlisted ones were producers that would have gone on
emitting retired tokens with nothing failing.

Every site below now derives its vocabulary from `brainstem/arm_config.py`. That module owns
`JOINTS` (the table), `ROTATABLE` (the addressable names), `TOKENS` / `MOVETO_TOKENS` (the
strings) and `command_for()` / `moveto_for()` (the formatters).

## Producers — sites that EMIT a command string

| # | Site | Was | Now |
|---|---|---|---|
| 1 | `brainstem/interpreter.py` `COMMANDS` | two hand-written entries, magnitudes `+15` / `-10` its own comment called placeholders | comprehension over `arm_config.ROTATABLE` at `DEFAULT_STEP_DEG` |
| 2 | `brainstem/signal_loop.py` `COMMANDS` | five-element literal list | `[*arm_config.TOKENS, "ENGAGE", "RELEASE", "HOME"]` |
| 3 | `brainstem/autonomous_loop.py` `_MOTOR` | f-string over `("A", "B")` | `arm_config.command_for` over `ROTATABLE` — 4 tokens became 10, `ACTIONS` 7 → 13 |
| 4 | `cortex/nl_interface.py` `ARM_LABELS` | two joint labels + English cues naming only "shoulder" / "elbow" | one label per rotatable joint, cues ordered most-specific-first |
| 5 | `brainstem/cpg_oscillator.py` | `JOINT_A_ROTATE` in an f-string | **NOT YET MIGRATED — C5.** Still drives its own `ArmSim`, so it is self-consistent |

## Validators — sites that ACCEPT or REJECT a command string

| # | Site | Was | Now |
|---|---|---|---|
| 6 | `brainstem/serial_arm.py` `_JOINT` | `JOINT_[AB]_ROTATE\(...\)` | alternation of `re.escape`d names from `ROTATABLE`, both verbs |
| 7 | `brainstem/so101_arm.py` `_JOINT_TOKEN` | — (new) | regex is **syntax**; the name group is checked against `_ROTATABLE`, which is **vocabulary** |
| 8 | `brainstem/arm_sim.py` | `JOINT_[AB]` | **deliberately unchanged.** Retired in Phase 7; changing it now would make the incumbent accept tokens its own body cannot execute |

## Why the two regexes are not one shared regex

`serial_arm` builds its allow-list from `arm_config` **independently of the mirror**. That
redundancy is not duplication to be factored out — it is what caught a malformed token the
mirror had accepted (`...ROTATE(+5deg)EMERGENCY_STOP`, admitted by a `.match` that should have
been `.fullmatch`, which performed the rotation and silently dropped the stop). Two independent
readings of one source; not one reading trusted twice.

## What a partial migration would have done

Left half-done, this set fails silently in both directions:

- a six-joint token against the 2-DOF body → the regex rejects it, `apply()` returns unchanged
  state, and the caller sees a successful no-op;
- a two-joint token against the 6-DOF body → `JOINT_A_*` matches no entry in `ROTATABLE` and is
  dropped the same way.

Neither raises. That is why the vocabulary flip, the body swap and the state-key migration are
one change rather than three.

## Open

- **T047** — `ACTIONS` is now *built* from `arm_config`, but the autonomous path still passes
  its chosen token to `apply()` without running it through the validator the chat path uses.
  Building from the right source and *checking against* it are different properties; only the
  first is done.
- **T049 / C5** — `cpg_oscillator.py`.
- **T055 / C6–C8** — `live_arm.py`, `spiking_pid.py`, `ring_attractor.py`,
  `wavefront_planner.py`, `demo.py`, `snn_motor_control.py` still construct `ArmSim` directly
  and are internally consistent on it. Note for whoever migrates them: `theta_deg[0]` was joint
  A (a **pitch** joint) on the old body and is `shoulder_pan` (a **yaw** joint) on the new one,
  so an unchanged `[0]` index silently reads a different joint. `shoulder_lift` is index 1.
