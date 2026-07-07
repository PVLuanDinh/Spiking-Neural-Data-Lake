# Robot-Arm Hardware Specification & Scope

This document is the authoritative statement of what the Spiking Neural Data Lake
robot-arm subsystem CAN and CANNOT do, as implemented in code today (repo version
2.53.2, `pyproject.toml:11`). Every claim below cites `file:line`. Where the code
was not read closely enough to be certain, that is flagged explicitly rather than
asserted. This is a safety-critical system - if this document and the code ever
disagree, the code is correct and this document is stale.

## CAN-do / CANNOT-do summary

| Capability | CAN | CANNOT |
|---|---|---|
| Arm geometry | 2-link planar arm, forward kinematics (`arm_sim.py:45-49`) | 3D arm, >2 joints, inverse kinematics solver |
| Motion | Discrete relative joint rotations, HOME/RETRACT, gripper open/close (`interpreter.py:27-33`) | Continuous/velocity control, path planning around obstacles |
| Collision | Detect ground-plane and one point obstacle after the fact (`arm_sim.py:51-57`) | Route around an obstacle; it only halts on penetration |
| Force/torque | Trip a danger reflex on out-of-range current/force channels (`reflex.py:21-24`) | Closed-loop force/torque control or compliant motion |
| Sensing | Ultrasonic distance (m) + IR temperature (K) via ESP32 (`esp32_encoder_path.py:14-16`) | Vision/video or audio (bands held at zero, `esp32_encoder_path.py:16`, `:229-241`) |
| Safety stop | Hardwired reflex fires in <=1 step on severe breach, freezes wire on E-STOP (`reflex.py:69`, `serial_arm.py:72-74`) | Guarantee hard real-time response (Python + serial, no RTOS) |
| Real hardware | Drive a real arm via `SerialArm` once calibrated (`serial_arm.py:57-66`) | Move a real arm with placeholder limits - gate refuses (`serial_arm.py:41-49`) |
| Physics | Optional pybullet rigid-body backend, else exact kinematic stdlib model (`arm_sim.py:8-13`, `:119-145`) | Full contact dynamics without pybullet installed |
| Multi-arm | One `ArmSim`/`SerialArm` instance = one arm | No multi-arm coordination anywhere in this codebase |

---

## 1. Arm model

### Geometry
- **2-link planar arm**: `ArmSim.__init__(self, l1=1.0, l2=1.0, use_pybullet=False)`
  (`arm_sim.py:33`, docstring "2-joint planar arm" `arm_sim.py:31`). Link lengths
  `l1`, `l2` default to 1.0 unit each; **2 revolute joints** `theta[0]`/`theta[1]`
  (joint A, joint B, `arm_sim.py:35`); a **gripper** scalar 0=open..1=closed
  (`arm_sim.py:36`).

### Forward kinematics
- `ee()`: `x = l1*cos(t0) + l2*cos(t0+t1)`, `y = l1*sin(t0) + l2*sin(t0+t1)`
  (`arm_sim.py:45-49`), `t0 = theta[0]`, `t01 = theta[0]+theta[1]`.
- **Forward kinematics only.** No function in `arm_sim.py` computes the inverse
  (target pose -> joint angles); confirmed by reading the whole file
  (`arm_sim.py:1-244`) - no `ik(`, `inverse`, or solver routine exists.

### Units
- **Joint angles**: radians internally (`self.theta`, `arm_sim.py:35`,
  `DEG = math.pi/180.0` at `arm_sim.py:22`); **command language + reporting are
  degrees** - `JOINT_A_ROTATE(+15deg)` (`arm_sim.py:27`), `state()["theta_deg"]`
  (`arm_sim.py:113`).
- **Positions/distances**: metres (`FLOOR_Y=-0.20`, `OBSTACLE=(0.4,1.3)`,
  `OBSTACLE_R=0.30`, `arm_sim.py:23-25`; sensor distances in metres,
  `esp32_encoder_path.py:14`, `arm_config.py:9,17-19`).
- **Temperature**: Kelvin (`T_LO_K=293.15`, `T_HI_K=313.15`,
  `esp32_encoder_path.py:28-29`).

### Workspace / reach
- Defaults `l1=l2=1.0`: HOME (`theta=[0,0]`) -> end-effector `(2.0, 0.0)`, per the
  module's own self-check `assert home["ee"] == (2.0, 0.0)` (`arm_sim.py:174-175`).
  Max reach = `l1+l2`; min reach (folded) = `|l1-l2|` (0 when equal) - geometric
  consequences of the FK formula, not a coded limit. **No explicit joint-angle
  range clamp exists in `arm_sim.py`** (confirmed by full-file read); limits are
  only enforced indirectly via collision detection and whatever the real servo
  hardware itself refuses to do.

### Physics backend
- **stdlib**: exact 2-link planar FK + geometric collision, always available,
  zero deps ("the verified core"). **pybullet** (optional): real rigid-body
  physics + contact collision, headless `p.DIRECT`, CI-safe - docstring
  `arm_sim.py:8-13`.
- `use_pybullet=True` -> `_init_pybullet()` (`arm_sim.py:119-145`); **any
  exception (including pybullet absent) falls back to stdlib** -
  `except Exception: self._p = None` (`arm_sim.py:144-145`). `self.backend`
  reports which is active (`arm_sim.py:38,143`).
- `pybullet_contacts()` returns real contact counts on pybullet, always `None`
  on stdlib (`arm_sim.py:153-159`).

---

## 2. Command set

All commands are plain ASCII strings, produced by `interpreter.py` and consumed by
`ArmSim.apply()` / `SerialArm.apply()`. The full recognized vocabulary (the "wire
allow-list") is defined by `_FIXED` and the `_JOINT` regex in `serial_arm.py:31-32`:

| Command | Effect | Step size / parameter | Source |
|---|---|---|---|
| `JOINT_A_ROTATE(+Ndeg)` / `(-Ndeg)` | Rotates joint A by a **relative** delta in degrees | Interpreter default: `+15deg` (`interpreter.py:28`); arbitrary float N accepted by the regex `[+-]?\d+(?:\.\d+)?deg` (`arm_sim.py:27`) | `arm_sim.py:60-71` |
| `JOINT_B_ROTATE(-Ndeg)` | Rotates joint B by a relative delta in degrees | Interpreter default: `-10deg` (`interpreter.py:29`) | `arm_sim.py:60-71` |
| `GRIPPER_CLOSE` | Sets `gripper = 1.0` (fully closed) | n/a (binary) | `arm_sim.py:72-73` |
| `GRIPPER_OPEN` | Sets `gripper = 0.0` (fully open) | n/a (binary) | `arm_sim.py:74-75` |
| `HOME` | Resets `theta=[0,0]`, `gripper=0.0`, clears `stopped` | n/a | `arm_sim.py:76-80` |
| `RETRACT_ALL` | Same effect as `HOME` (both hit the same branch) | n/a | `arm_sim.py:76-80` |
| `EMERGENCY_STOP` | Sets `stopped = True`; freezes all further motion until `RETRACT_ALL`/`HOME` | n/a | `arm_sim.py:61-63` |
| `HOLD` | No-op: does not match any branch, so no joint motion and `expected_ee` is NOT recommitted | n/a | `arm_sim.py:81-85` (comment), confirmed by self-check `arm_sim.py:211-212` |

Notes:
- **Relative, not absolute**: deltas add to the current angle
  (`self.theta[j] += float(m.group(2)) * DEG`, `arm_sim.py:70`) - no "rotate to
  absolute angle N" command exists.
- **Freeze semantics**: once `stopped` is `True`, `apply()` short-circuits to
  `return self.state()` for anything but `RETRACT_ALL`/`HOME` (`arm_sim.py:64-65`).
  `SerialArm` enforces the same freeze **on the physical wire**, independent of the
  mirror: `_ALLOWED_WHEN_STOPPED = {"EMERGENCY_STOP","HOME","RETRACT_ALL"}`
  (`serial_arm.py:33`), gated in `apply()` (`serial_arm.py:72-74`).
- **Unrecognized/unmapped commands** fall through with `moved=False` - no motion,
  no expected-pose change (`arm_sim.py:66-67,81-85`).
- Interpreter priority: **reflex event > AVOID instinct veto > confident match**
  (`interpreter.py:9-13`, implemented `interpreter.py:45-61`):
  1. `"reflex"` key present -> `REFLEX_CMD` lookup (`STOP`->`EMERGENCY_STOP`,
     `WITHDRAW`->`RETRACT_ALL`), reward `-1.0` (`interpreter.py:34,46-47`).
  2. `ev.get("instinct") == "AVOID"` -> forces `"HOLD"`, reward `-0.3`
     (`interpreter.py:52-53`) - this **vetoes** whatever was matched.
  3. Otherwise the recognized gesture label is mapped via the `COMMANDS` /
     `SENSOR_COMMANDS` dict (`interpreter.py:27-33`, `live_arm.py:37-39`).

---

## 3. Sensors

### Hardware
- **Board**: ESP32 Dev Module driving an ultrasonic rangefinder (HC-SR04-style -
  code treats it as a single distance-in-metres channel, `Sensor_1` per
  `arm_config.py:9`) and an MLX90614 IR thermometer (`Sensor_2`, `arm_config.py:9-10`).
  **Flagged uncertainty**: no literal "HC-SR04" or "MLX90614" string appears in
  the Python source read for this doc - that part identification comes from
  `arm_config.py`'s comments ("distance in metres (ultrasonic / Sensor_1)", "IR =
  Sensor_2") and project memory, not a source-code string match.
- **Wire contract**: 115200 baud default throughout - `serial_arm.py:57`
  (`SerialArm.__init__(self, port=None, baud=115200, ...)`), `live_arm.py:57`,
  `esp32_encoder_path.py:124,134` (docstring: "the ESP32 firmware rate").
- **Line format**: plain-text `"distance_m, temp_k"` (two comma-separated floats),
  parsed by `parse_esp32_line()` (`esp32_encoder_path.py:32-57`).

### Distance (ultrasonic)
- Units: **metres**. `distance_m / 3.0`, clamped to `[0,1]` - the **3 m band is
  the object-present gate** (`esp32_encoder_path.py:14,71-72`).
- Separately, `arm_config.py` defines the recognition-side distance bins in
  metres: `NEAR_MAX = 0.30`, `MID_MAX = 0.60`, `IN_RANGE_MAX = 2.0`
  (`arm_config.py:17-19`) - note this is a **different threshold (2.0 m) than the
  encoder path's 3.0 m gate**; they are two independent modules with their own
  in-range cutoffs. Do not conflate them.
- **Reflex contact distance**: `CONTACT = 0.04` m (4 cm) - below this the
  imminent-contact reflex fires `EMERGENCY_STOP` (`arm_config.py:22`, used at
  `live_arm.py:106-112`).

### Temperature (IR, MLX90614-class sensor per `Sensor_2`)
- Units: **Kelvin**, band 293.15 K (20 C) to 313.15 K (40 C), linearly normalized
  to `[0,1]` and clamped (`esp32_encoder_path.py:27-29,74-80`).
- **Idle sentinel**: firmware sends `temp_k = -1` when `distance > 3.00 m` (no
  object in range); any `temp_k <= 0` is treated as **missing** (`temp01 = None`),
  which the multimodal encoder drops to an all-zero band rather than encoding a
  bogus reading (`esp32_encoder_path.py:9-10,74-76`, verified by self-check
  `esp32_encoder_path.py:250-264`).
- IR fusion for *presence gating* (distinct from the temperature reading above) is
  **disabled by default and uncalibrated**: `IR_PRESENT = None` in
  `arm_config.py:43` - "distance alone decides presence, exactly as before,"
  because "IR polarity/scale is sensor-specific and UNCALIBRATED"
  (`arm_config.py:39-42`).

### Video and audio - NOT implemented
- Four encoder bands: visual `[0:16]`, temperature `[16:32]`, audio `[32:48]`,
  distance `[48:64]` (`esp32_encoder_path.py:14-16`; confirmed by self-check
  `feats_obj[0:16] == [0.0]*16` and `feats_obj[32:48] == [0.0]*16`,
  `esp32_encoder_path.py:228-241`).
- Docstring: **"Visual [0:16] and audio [32:48]: absent (None) -> zeros until
  video module lands"** (`esp32_encoder_path.py:16`). No camera or microphone
  input path exists in any module read for this document. Treat video/audio as
  **unimplemented, not partially working**.

---

## 4. Safety envelope - what you CAN rely on

### Reflex fast-path (hardwired, unlearned)
- `Reflex` (`reflex.py:27-73`), "fast, unlearned, sign-aware" (`reflex.py:6-11`):
  runs on **raw signed samples** before any classification, catching danger in
  either direction.
- Default channel rules (`reflex.py:21-24`): channel 6 (motor current) valid
  range **[-50.0, 800.0]**, breach -> `"STOP"`; channel 7 (force/torque) valid
  range **[-300.0, 300.0]**, breach -> `"WITHDRAW"`.
- Firing logic (`reflex.py:45-73`): scale-free severity (`over/(hi-lo)`,
  `reflex.py:60-61`); fires when leaky membrane `v` crosses `threshold=0.10`
  (default, `reflex.py:36`) **or** after `persist=10` consecutive breach samples
  even if too mild to cross the threshold alone (`reflex.py:66-69`) - closes the
  dead-band where a sustained mild overload could be silently ignored. A severe
  breach fires in **one step** (self-check `reflex.py:111-112`).
- **Missing-channel reporting**: an out-of-range rule channel is recorded once in
  `self.missing` with a stderr warning - inactive for that frame, but surfaced,
  not hidden (`reflex.py:50-54`, self-check `reflex.py:129-132`).

### NaN / non-finite fail-CLOSED (as of v2.53.x)
- Non-finite reading (`NaN`/`inf`) on a monitored channel = **maximum severity**
  (`sev = float("inf")`, `reflex.py:57-58`), guaranteeing the reflex fires instead
  of silently passing (`NaN < x`/`NaN > x` are both `False` in Python, which would
  otherwise fail open) - asserted for both channels (`reflex.py:136-139`).
- Same principle in the live loop: a non-finite distance sample trips the contact
  reflex instead of falling through `d < CONTACT` -
  `if not math.isfinite(d) or d < CONTACT:` (`live_arm.py:106`), regression
  assertion at `live_arm.py:208-211`.

### Collision detection
- Ground plane: end-effector `y < FLOOR_Y` (`FLOOR_Y=-0.20`) -> `"GROUND"`
  (`arm_sim.py:23,52-54`). Point obstacle: within `OBSTACLE_R=0.30` m of
  `OBSTACLE=(0.4,1.3)` -> `"OBSTACLE"` (`arm_sim.py:24-25,55-56`).
- Collision is **detected after the fact** - the check reports on the next
  `state()`/`collision()` call; there is no pre-motion avoidance in `arm_sim.py`.

### EMERGENCY_STOP freeze-until-retract
- `ArmSim`: `apply("EMERGENCY_STOP")` sets `stopped=True`; every subsequent
  `apply()` except `RETRACT_ALL`/`HOME` returns state unchanged
  (`arm_sim.py:61-65`, self-check `arm_sim.py:188-194`).
- `SerialArm`: freeze enforced **on the wire itself**, not just the mirror -
  `blocked_by_stop = self.mirror.stopped and command not in
  _ALLOWED_WHEN_STOPPED` (`serial_arm.py:72-74`); a blocked command touches only
  the mirror, "nothing goes to the servos" (`serial_arm.py:69-71`). Verified:
  commands during the freeze never reach the fake serial's sent-lines
  (`serial_arm.py:162-168`).

### Calibration gate (no bypass)
- `SAFETY_CALIBRATED = False` by default (`arm_config.py:32`) - a **placeholder
  tuned to the sim, not any real rig** (`arm_config.py:29`).
- `assert_calibrated()` raises `RuntimeError` unless `SAFETY_CALIBRATED` is `True`
  or `allow_uncalibrated=True` is passed explicitly (`serial_arm.py:41-49`).
- Runs on **every** `SerialArm` construction, including an injected `writer=` -
  hardened so "an injected `writer=` no longer bypasses it" (`serial_arm.py:11-13,
  57-60`). Verified: injected fake writer with no opt-in still refused
  (`serial_arm.py:136-141,170`); opening a real port while uncalibrated also
  refused (`serial_arm.py:142-147,171`).
- `live_arm.py --serial ... --actuate ...` also calls `assert_calibrated()`
  before opening the real-hardware path (`live_arm.py:166`).

### Command allow-list
- Only two shapes reach the wire: `_FIXED = {"EMERGENCY_STOP","HOME",
  "RETRACT_ALL","GRIPPER_CLOSE","GRIPPER_OPEN","HOLD"}` and the
  `JOINT_[AB]_ROTATE(...)` regex (`serial_arm.py:31-38`). Anything else is
  dropped, never transmitted (`serial_arm.py:73-74`, self-check with
  `"DROP_EVERYTHING"`, `serial_arm.py:127,168`).

### Trajectory-deviation comparator
- `ArmSim` tracks `expected_ee` - where the EE should be given the last
  **commanded** motion (`arm_sim.py:40,85`). `deviation()` = Euclidean distance
  actual vs. expected (`arm_sim.py:101-105`).
- `trajectory_breach(bound=TRAJ_DEV_MAX)` is `True` when `deviation() > bound`,
  `TRAJ_DEV_MAX = 0.15` m by default (`arm_config.py:23-26`, imported
  `arm_sim.py:20`), checked at `arm_sim.py:107-109`.
- Catches **external perturbations** moving the arm without a command
  (`perturb()`, `arm_sim.py:91-99`). A no-op command (`HOLD`, unknown token, or a
  gripper command) does **not** clear a breach; only real motion recommits
  `expected_ee` (`arm_sim.py:81-85`, self-check `arm_sim.py:205-218`).

### Cortisol-modulated vigilance
- `cortisol.py`'s `Cortisol` state biases the instinctive `AVOID` decision:
  `v_action, v = valence.act(window, bias=cortisol.caution_bias())`
  (`signal_loop.py:404`). Per the graph report, cortisol lowers the reflex
  threshold under stress (hypervigilance) and biases toward `AVOID`. **Flagged
  uncertainty**: `signal_loop.py` was read only at this integration call site,
  not `cortisol.py` in full - treat exact cortisol numeric constants as
  unverified here; read `src/snn_data_lake/cortisol.py` directly before relying
  on specific thresholds.

---

## 5. Driving REAL hardware

To move a physical arm instead of the simulator:

1. **Install the `arm` extra** (pyserial): `arm = ["pyserial"]`
   (`pyproject.toml:26`). Imported **lazily**, only inside `SerialArm.__init__`
   when `writer is None` (`serial_arm.py:61-63`) and inside `read_serial_lines()`
   (`esp32_encoder_path.py:140-145`) - core stays zero-dependency until a real
   port is opened.
2. **A serial port and baud rate**: `SerialArm(port, baud=115200, ...)`
   (`serial_arm.py:57`); default baud matches the ESP32 firmware
   (`esp32_encoder_path.py:134`).
3. **Calibration - mandatory, no bypass**: `arm_config.SAFETY_CALIBRATED` ships
   `False` (`arm_config.py:32`) with placeholder `CONTACT`/joint-limit values
   "tuned to the sim, NOT your rig" (`arm_config.py:29`). Replace those constants
   with the rig's measured limits and flip `SAFETY_CALIBRATED = True` before
   `SerialArm` will drive real servos (`serial_arm.py:41-49`). **This is
   explicitly the integrator's responsibility** - the code refuses to guess safe
   limits for unseen hardware; no calibration wizard or auto-detection exists in
   any module read for this document.
4. **Entry points**: `snn-arm` -> `snn_data_lake.live_arm:main`
   (`pyproject.toml:38`); run with `--serial COMx` (sensor-only, sim actuation) or
   `--serial COMx --actuate COMy`/`--actuate COMx` (real actuation, same or
   separate UART) - `live_arm.py:11-13,161-182`.
5. **Fail-safe on serial errors**: any write exception re-raises as
   `RuntimeError`, "so the caller stops the loop rather than driving blind"
   (`serial_arm.py:20,75-79`, self-check `serial_arm.py:149-155,172`).

---

## 6. What the arm CANNOT do (scope ceilings)

- **Only a 2-link planar arm.** No 3D motion, no more than 2 joints, no general
  IK solver. `ArmSim` hardcodes a 2-joint model (`arm_sim.py:33-35`) and only
  implements forward kinematics (`arm_sim.py:45-49`) - confirmed by full-file
  read: no routine accepts a target `(x,y)` and returns joint angles.
- **No closed-loop force/torque control.** A force/current-like channel is used
  only as a **danger trip** in `Reflex` (`reflex.py:21-24`) - it stops or
  withdraws, it does not modulate torque/grip force to a setpoint. No
  force-feedback controller exists anywhere in the modules reviewed.
- **No vision or audio.** Confirmed unimplemented (Section 3); visual `[0:16]`
  and audio `[32:48]` bands are always zero (`esp32_encoder_path.py:16,228-241`).
- **No path planning / obstacle avoidance.** `arm_sim.py` detects collision after
  the EE has already entered the ground plane or obstacle sphere
  (`arm_sim.py:51-57`); no pre-motion planner routes around it. The only
  response to a collision is to stop (reflex/E-STOP), not reroute.
- **The simulator is kinematic by default.** Real dynamics (mass, inertia,
  contact forces) exist only if the optional `pybullet` backend initializes
  (`arm_sim.py:8-13,119-145`); the always-available stdlib path is pure
  geometry - angles in, position out, no dynamics simulated.
- **Sim-to-real gap is not modeled or bounded.** `SerialArm` mirrors `ArmSim`
  purely for state/gripper bookkeeping (`serial_arm.py:53-56,65-66`) - it does
  not read back real encoder positions to correct for drift or backlash. The
  proprioceptive "slip watchdog" (`live_arm.py:41,51-54,122-127`) is explicitly
  **read-only**, issuing no correction (self-check `live_arm.py:268-282` proves
  it never changes the arm's actual state).
- **No multi-arm support.** `ArmSim`/`SerialArm`/`Reflex`/`Interpreter` are all
  single-instance-per-arm; nothing coordinates two or more arms.
- **No hard real-time guarantee.** Plain Python over a standard serial link with
  software timeouts (`timeout=1` in `serial_arm.py:63`) - no RTOS, no
  bounded-latency scheduling; `live_arm.py`'s reconnect-with-backoff
  (`live_arm.py:57-73`) shows the loop tolerates multi-second serial dropouts,
  incompatible with a hard real-time claim.
- **Ships uncalibrated by default.** `SAFETY_CALIBRATED=False` and placeholder
  `CONTACT`/`TRAJ_DEV_MAX` values out of the box (`arm_config.py:22-32`) - a
  deliberate fail-safe default, but the system does nothing useful on real
  hardware until an integrator completes calibration (Section 5).
- **Opt-in joint-drive primitives are reactive, not planned.**
  `SpikingPID.drive_to()` drives joint A only, via a population-coded PID
  (`spiking_pid.py:26-47,50-78`); `cpg_sweep()` drives a rhythmic bounded
  oscillation on joint A only (`cpg_oscillator.py:44-61`). Both go through the
  gated `apply()` (collision/E-STOP still apply), but neither is a general
  motion planner and both are single-joint as implemented.

---

## Sources read for this document

Full read: `arm_config.py`, `arm_sim.py`, `serial_arm.py`, `reflex.py`,
`interpreter.py`, `live_arm.py`, `esp32_encoder_path.py`, `spiking_pid.py`,
`cpg_oscillator.py` (all in `src/snn_data_lake/`). Grep-only, not read in full:
`ring_attractor.py` (`ProprioceptiveEstimator`/`command()`/`mismatch_deg()`
signatures, lines 78/93/103), `signal_loop.py` (`Cortisol` call site, line 404),
`pyproject.toml` (version/`arm` extra/console scripts). Orientation:
`graphify-out/GRAPH_REPORT.md`. Grep-only sources are flagged inline above
(cortisol numeric constants, ring_attractor internals, HC-SR04/MLX90614 part
numbers).
