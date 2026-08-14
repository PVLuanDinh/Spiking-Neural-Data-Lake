# Feature Specification: Live Simulated Sensory Transduction

**Feature Branch**: `014-live-sensor-transduction`

**Created**: 2026-08-13

**Status**: Draft

**Input**: Owner directive — the cockpit should solve sensory transduction, simulating a live environment in which the brain translates Arduino-class electrical signals into spike signals; the daemon acts as producer and consumer; the feed reaches the safety envelope.

---

## OWNER AUTHORIZATION — read this before anything else

`docs/POLICIES.md` and the constitution both state that the five `brainstem/` safety-envelope files
are **modified only with explicit approval**. The owner gave that approval for this feature, in
these terms:

> "Simulate sensor feed reach the safety envelope. The cockpit gain an afferent sensor channel.
> Follow POLICIES.md. The daemon should use it as a producer and consumer."

**What that authorizes**: a simulated sensor feed that reaches the reflex arc; a new afferent
channel on the cockpit; the daemon producing and consuming a simulated environment.

**What it does NOT authorize, and this spec treats as unchanged**: relaxing the E-STOP, altering
existing reflex thresholds, calibrating hardware, or any general licence to edit the envelope in
future work. This approval is scoped to this feature and expires with it.

---

## Why this exists

The brain's only live sensory transduction today is `cortex/eprop_nmnist.py` (N-MNIST event digits)
and `cortex/eprop_shd.py` (SHD spoken digits). Measured this session under the sanctioned harness,
that pipeline genuinely runs — `perception_tier="eprop-snn"`, `_perceiving_real=True`,
`_perceiving_audio=True`. **But what it transduces is benchmark content.** A brain that classifies
digits has a working pipeline, not a sense.

`docs/POLICIES.md` P4c names the cockpit **the brain's SENSORIMOTOR SURFACE** — "where the brain
meets the world: chat and controls in are AFFERENT … it is the boundary of the organism." Afferent
transduction belongs there. Today the cockpit's entire afferent surface is `/api/chat`; there is no
sensor channel at all.

Meanwhile the transduction machinery **already exists and is already governed**:
`substrate/esp32_encoder_path.py` carries `parse_esp32_line` → `normalize(distance_m, temp_k)` →
`encode_reading` → `spikes_for(feats)`, registered in `__init__.py` and self-checked in CORE. It has
**zero consumers** — measured, no references to `spikes_for` or `encode_reading` anywhere outside
the module — and is dormancy-EXEMPT `"permanent"` with the reason *"AWAITING HARDWARE: real-sensor
adapter; pyserial absent and arm_config SAFETY_CALIBRATED=False"*.

**That exemption is broader than the blockage.** `pyserial` is genuinely absent (measured:
`ModuleNotFoundError`), but only `read_serial_lines` needs it. The `parse → encode → spikes` chain
consumes **text lines** and needs no hardware whatsoever. A simulated environment awaits nothing.

So: the producing half is built, the consuming half is absent, and an exemption has been standing in
for the missing half.

## User Scenarios & Testing *(mandatory)*

The "user" is the brain, its owner at the cockpit, and the gate set. Each story is an independently
landable increment leaving the full gate green.

### User Story 1 — The daemon simulates an environment the brain can sense (Priority: P1)

The daemon generates a stream of sensor readings representing a simulated physical environment
(proximity and temperature), in the same textual form a real ESP32/Arduino emits, and feeds them
through the existing transduction chain into the brain on its own tick.

**Why this priority**: it is the whole point — it converts a dormant, exempted transduction path
into a live sense, and it is the half that does not touch the safety envelope.

**Independent Test**: run the daemon; observe that readings are produced, encoded to spikes, and
that the brain's state reflects them.

**Acceptance Scenarios**:

1. **Given** a running daemon with the simulator enabled, **When** it ticks, **Then** sensor
   readings pass through `parse_esp32_line` → `encode_reading` → `spikes_for` and the resulting
   spikes reach the brain on the tick thread.
2. **Given** the simulator is disabled, **When** the daemon ticks, **Then** behaviour is byte-identical
   to today — no reading is produced and nothing in the brain changes.
3. **Given** any produced reading, **When** it is stored, reported, or acted on, **Then** it is
   marked as SIMULATED at every one of those points (FR-009).

---

### User Story 2 — The cockpit accepts an afferent sensor reading (Priority: P1)

The cockpit gains a sensor endpoint so a reading can be injected from outside the daemon's own
simulator — the owner poking the environment, or an external simulator standing in for hardware.

**Why this priority**: it is the owner's stated requirement and it is what makes the cockpit the
sensorimotor surface in fact rather than in docs.

**Independent Test**: POST a reading; observe it reach the brain on the next tick without the
request thread ever touching the brain or the arm.

**Acceptance Scenarios**:

1. **Given** a cockpit served WITHOUT a sensor queue, **When** a reading is POSTed, **Then** it is
   refused, exactly as `/api/act` returns 404 when served without an act queue.
2. **Given** a malformed or out-of-contract reading, **When** it is POSTed, **Then** it is rejected
   with an error and nothing is queued.
3. **Given** a well-formed reading, **When** it is POSTed, **Then** the request thread only validates
   and enqueues — it never calls into the brain, the reflex, or the arm.

---

### User Story 3 — The feed reaches the safety envelope, and can only ever add caution (Priority: P1)

A simulated proximity reading can trip the reflex arc, producing a protective action.

**Why this priority**: this is the authorized capability and the highest-risk part of the feature.
It is listed last of the P1s because it must land only once Stories 1 and 2 are green.

**Independent Test**: drive a simulated obstacle to close range; observe a protective action;
then confirm by construction and by test that no simulated value can suppress an existing stop.

**Acceptance Scenarios**:

1. **Given** a simulated proximity reading inside the danger limit, **When** the reflex steps,
   **Then** a protective action fires.
2. **Given** ANY simulated reading whatsoever, **When** an existing channel (motor current, force)
   is simultaneously in breach, **Then** the existing breach still fires — the simulated channel
   cannot delay, mask, or cancel it (FR-004).
3. **Given** the feature is fully enabled, **When** the existing reflex rules are inspected, **Then**
   channels 6 and 7 still carry motor current and force/torque with unchanged bounds (FR-005).
4. **Given** a non-finite simulated reading, **When** the reflex steps, **Then** it is treated as a
   sensor fault at maximum severity, matching the existing contract for real channels.

---

### User Story 4 — The exemption is settled honestly (Priority: P2)

`esp32_encoder_path`'s dormancy exemption is either lifted (it now has a live consumer) or its
reason is corrected to name only the part that is genuinely hardware-blocked.

**Why this priority**: it is bookkeeping, but it is the bookkeeping that stops a standing exemption
from covering for a missing consumer — which is what it has been doing.

**Acceptance Scenarios**:

1. **Given** the consumer lands and CHARGES, **When** the dormancy gate runs, **Then** the module is
   reachable and the exemption is removed.
2. **Given** the consumer lands and does NOT charge, **When** the disposition is written, **Then**
   the wire is REVERTED and the exemption restored with a corrected reason (FR-012).

### Edge Cases

- **A reading arrives while the brain is mid-tick** — the queue pattern makes this a non-event; the
  request thread never touches brain state.
- **The simulator floods the queue** — bounded queue, oldest dropped, drop count reported. A sensory
  surface that silently drops is a surface that lies about what it sensed.
- **A reading names a channel the frame does not carry** — the existing reflex behaviour already
  covers this: report once on stderr as INACTIVE, never silently skip.
- **`pyserial` becomes available later** — nothing in this feature changes; `read_serial_lines`
  remains the hardware path and remains unused.
- **The simulated environment is boring** — a simulator that never approaches a limit exercises the
  wire but never the reflex. The charge probe must drive it to the limit, not merely run it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The daemon MUST be able to produce a stream of simulated sensor readings in the same
  textual form the existing `parse_esp32_line` already accepts. No new parsing format.
- **FR-002**: The daemon MUST consume those readings on its own tick thread, through the EXISTING
  `parse_esp32_line` → `encode_reading` → `spikes_for` chain. This feature adds no second
  transduction path (P2: exactly one mechanism per job).
- **FR-003**: The cockpit MUST expose an afferent sensor endpoint following the `/api/act` pattern
  EXACTLY: validate on the request thread, enqueue a ticket, never touch brain/reflex/arm from the
  request thread, and refuse when served without a queue.
- **FR-004** *(safety, structural)*: A simulated reading MUST only ever be able to ADD a stop
  condition. This MUST be structural, not careful: the simulated value enters as an ADDITIONAL
  reflex channel whose severity is summed into the existing membrane current alongside the existing
  channels. Because severity is non-negative and the fire condition is a threshold on an
  accumulating sum, an added channel can raise the membrane and can never lower it, and the
  most-severe-action selection can never deselect a more severe existing breach.
- **FR-005** *(safety)*: Reflex channel indices 6 (motor current) and 7 (force/torque) MUST retain
  their current meaning and bounds. New channels take HIGHER indices. A test MUST pin 6 and 7,
  because the rules are index-keyed and a frame-layout shift would silently re-map them onto the
  wrong data — a safety regression no existing check would catch.
- **FR-006** *(safety)*: The E-STOP path MUST be unchanged. No existing threshold, rule, or action
  is altered by this feature.
- **FR-007** *(safety)*: `arm_config.SAFETY_CALIBRATED` MUST remain `False` and
  `serial_arm.assert_calibrated()` MUST continue to refuse real servos. This feature must not
  calibrate hardware and must not appear to.
- **FR-008** *(safety, P1 exception (d))*: Any new proximity/temperature danger limit is a RAW
  PHYSICAL LIMIT INSIDE THE SAFETY ENVELOPE and MUST stay a fixed constant. P1's fourth exception
  applies verbatim: "a reflex threshold that drifts with a running baseline is a reflex that can be
  trained away, and the whole point of the envelope is that it cannot."
- **FR-009** *(honesty)*: Every simulated reading MUST be marked as simulated at every point it is
  stored, reported, or acted upon — in the brain's state report, in any lake record, and in any
  cockpit display. A simulated proximity threat MUST NOT be presentable as a real one.
- **FR-010** *(P3, and the feature's real bar)*: The consumer MUST be proven to CHARGE by a
  neuter-A/B in the style of `tools/charge_probe.py` — neuter the sensor feed and show that a
  brain-level outcome changes. Reachability is not charge, and P3 forbids faking a consumer.
- **FR-011** *(P8)*: Every criterion this feature adds MUST carry a demonstrated reachable RED — the
  input that makes it fail, constructed and recorded. A criterion with no reachable red is DELETED,
  not weakened.
- **FR-012** *(deletion on a null)*: If FR-010's probe shows the feed does NOT change any outcome,
  the wire MUST be reverted and `esp32_encoder_path` re-exempted with a corrected reason. Shipping
  an inert wire to clear an exemption is the defect this feature exists to remove, not to commit.
- **FR-013** *(P4b)*: Anything persisted goes into `brain_lake.bin` under its own record tag. The
  daemon MUST NOT gain a second durable store.
- **FR-014**: The simulator MUST be default-OFF, or otherwise byte-identical in behaviour to today
  when disabled, so that no existing measurement silently changes meaning.

### Non-Goals

- **NG-001**: Real hardware bring-up. No ESP32 is connected, read, or assumed.
- **NG-002**: Installing `pyserial`. `read_serial_lines` stays the (unused) hardware path.
- **NG-003**: Calibrating the arm or flipping `SAFETY_CALIBRATED`.
- **NG-004**: Changing any existing E-STOP or reflex threshold.
- **NG-005**: Retiring the N-MNIST/SHD perception tiers. They remain the vision/audio encoders; this
  feature adds a *third*, non-benchmark sense rather than replacing them.

### Key Entities

- **Simulated reading** — a proximity distance and a temperature, in the ESP32 text form, carrying
  an explicit SIMULATED marker that survives every downstream hop.
- **Sensor queue** — the cockpit→tick handoff, mirroring the existing act queue: bounded, drop-counted,
  drained on the tick thread.
- **Simulated environment** — the daemon-side producer; a scripted or stochastic world that can be
  driven to the danger limit on demand (the charge probe needs that, per Edge Cases).
- **Additional reflex channel** — a new, higher-index nociceptor channel carrying proximity, additive
  to `DEFAULT_RULES`, never displacing 6 or 7.

## Success Criteria *(mandatory)*

- **SC-001**: With the simulator disabled, the brain's observable state is IDENTICAL to before this
  feature, measured at ≥12 ticks (a 0-tick comparison is vacuous — established at specs/013).
- **SC-002**: With the simulator enabled and an obstacle driven to the danger limit, a protective
  action fires.
- **SC-003**: Reflex channels 6 and 7 carry unchanged meanings and bounds — asserted by a test that
  fails if either is displaced.
- **SC-004**: For every existing breach scenario, the outcome with the simulated channel present is
  the same as, or more protective than, without it. Never less protective. Demonstrated across the
  breach cases, not argued.
- **SC-005**: The neuter-A/B shows a brain-level outcome that differs with the feed versus without
  it — or FR-012's revert is executed and recorded.
- **SC-006**: Every new criterion has a recorded, demonstrated reachable RED.
- **SC-007**: `SAFETY_CALIBRATED` is still `False` and real-servo entry points still refuse.
- **SC-008**: `esp32_encoder_path`'s dormancy disposition matches reality — exemption lifted if it
  charges, or restored with a corrected reason if it does not.
- **SC-009**: Full gate green at every commit: self-checks, pytest ≥70% coverage, ruff check AND
  `ruff format --check`, p4c, p6, citation_guard, red_reachability, dependency_ledger,
  import_manifest.

## Assumptions

- The simulated environment models **proximity and temperature** because those are exactly what
  `normalize(distance_m, temp_k)` already accepts. Inventing new modalities would mean a second
  transduction path, which P2 forbids.
- The reflex is the right consumer for proximity because `cortex/brain.py` already describes
  `_perceiving_real` as "the arm's sensory channels are DRIVEN by real perception" — the causal path
  from distance to caution already exists in intent.
- Default-OFF is assumed for the simulator (FR-014) so that existing coverage numbers, self-check
  outputs, and recorded measurements do not silently change meaning.
- The gate environment is `.venv-lake`; CI has no GPU and no hardware. Nothing in this feature may
  require either.

## Risks

- **R-001 (highest)**: A simulated threat is mistaken for a real one somewhere downstream — in the
  cockpit display, a lake record, or a coverage claim. This is the honesty risk, and it is why
  FR-009 requires the marker to survive *every* hop rather than exist at the entry point.
- **R-002**: Index drift silently re-maps safety rules. `DEFAULT_RULES` is index-keyed and nothing
  today pins the layout; FR-005's test exists solely to make this loud.
- **R-003**: The consumer does not charge, and the temptation is to wire it somewhere livelier
  rather than accept the null. FR-012 pre-commits to the revert precisely to remove that choice
  from the moment it would be made under pressure.
- **R-004**: A prior arc measured that `blocked` fired on *acceleration*, so the objective paid the
  arm to stand still. A proximity→caution wire is the same family: the intended behaviour (stop when
  close) and a degenerate one (never move) are hard to tell apart from the reward alone. Any claim
  that the brain "responds to obstacles" must distinguish these.
