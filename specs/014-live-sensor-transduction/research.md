# Phase 0 Research — Live Simulated Sensory Transduction

All measurements at HEAD on branch `016-tier-consistency-check` (parent `462da205`), gate env
`.venv-lake`. Claims re-derived by the main agent; nothing here is relayed from memory.

---

## D-1 — THE CONSUMER ALREADY EXISTS. Nothing is wired; something is fed.

**Decision**: do not build a consumer. Feed the one that is already there.

**Measured**:

- `brainstem/autonomous_loop.py:1469` — `AutonomousArm.step(self, visual, temperature, audio,
  distance, reward_fn=None, homeostatic_load=0.0)`
- `:1473` — `if distance is not None and (not math.isfinite(distance) or distance < CONTACT):`
- `:1479` — `am.pair(pat)  # US present: bind this context to threat`
- `arm_config.py:27` — `CONTACT = 0.04  # d < CONTACT -> imminent-contact reflex EMERGENCY_STOP`
- `cortex/brain.py:71` — `SAFE_DISTANCE = CONTACT * 3.0  # perception-only distance band; never in a
  reflex/danger zone`
- `autonomous_loop.py:2269` — the live caller passes `distance=safe_d`

**Why `threat()` is exactly 0.0, and it is not a bug.** The distance the arm is fed today is derived
from `SAFE_DISTANCE`, which is documented as *never* in the danger zone. The contact branch is
therefore **unreachable by construction**, not broken. `fear_conditioning.py` states the same
conclusion from the other side: 38 calls per 80 wake ticks, 0.0 every time, and "a threat readout of
0.0 is the HONEST output of a brain that has never been hurt."

**So the feature is small and the leverage is large**: supply a distance that CAN enter the danger
zone, and an entire faculty starts working with no code change — exactly as the module predicted
("When the arm is built and the sensor is calibrated, this faculty starts learning with no code
change at all").

**Alternatives considered and rejected**:

- *Invent a new consumer (perception / attention / workspace)* — rejected. P3 forbids faking a
  consumer, and a new sink built to receive this signal would be precisely that. A real one exists.
- *Bind `pair()` to the reward signal* — rejected, and it was rejected before this lane:
  `fear_conditioning.py` records that proposal and calls it "theater P0's honesty guard exists to
  refuse", because it conditions the amygdala on DISAPPOINTMENT rather than nociception.

---

## D-2 — The owner's ruling, and the line it draws

**Decision**: simulated contact drives `pair()`. **Nothing may be scored on it.**

The module's stated prerequisite is "a real proximity sensor **and a real body**". A simulated
sensor is neither, and `amygdala_fear_conditioning` sits at **band 0**. Making `threat()` non-zero
would move a coverage number on simulated evidence — the exact shape the disposition refuses. Asked
directly, the owner ruled: yes to the learning, no to the scoring.

**What makes this honest rather than theatrical, stated so it can be checked:**

1. **The semantics are unchanged.** The amygdala still conditions on CONTACT — nociception — not on
   disappointment. The refused wire changed *what the faculty means*; this one changes *what the
   world contains*.
2. **Every derived value is marked SIMULATED** (FR-009), at every hop, so no downstream reader can
   mistake it.
3. **No coverage band rises in this lane.** Pinned as an explicit check, not an intention.
4. **The DISPOSITION note is amended, not overtaken.** A safety-adjacent faculty's stated
   prerequisite is being changed; that is recorded in the file with the date and the authorization,
   so the next reader sees a decision rather than a drift.

**The honest limit, disclosed unprompted (P8 guard 4):** a brain that has only ever been hurt in
simulation has learned about a simulation. `threat()` becoming non-zero is evidence that the
mechanism works, NOT evidence that the brain has a validated threat model of the world.

---

## D-3 — Reflex channel layout: proximity at index 8

**Decision**: the simulated proximity channel is **index 8**. Indices 6 and 7 are untouched.

**Rationale, measured from `brainstem/reflex.py`**: `DEFAULT_RULES = {6: (-50.0, 800.0, "STOP"),
7: (-300.0, 300.0, "WITHDRAW")}` is **index-keyed into a raw sample list**. Any change that shifts
frame layout silently re-maps motor-current and force/torque rules onto the wrong data. Nothing in
the tree pins that layout today, so the failure would be silent — a safety regression no existing
check catches. New channels therefore go strictly ABOVE the existing ones, and FR-005's test pins
6 and 7 so a future shift is loud.

**Why adding a channel can only add caution — the structural argument, from the arithmetic**:

- severity is `over / max(1e-9, hi - lo)` when out of bounds, else `0.0` → **non-negative, always**
- `current += sev` accumulates across channels → an added channel can only **raise** `current`
- `self.v = self.v * self.decay + current` → membrane can only rise relative to the same run without
  the channel
- firing is `self.v >= self.threshold or self._breach >= self.persist` → a higher membrane fires no
  later, never later
- action selection takes the **most severe** channel, so a new channel can only replace the chosen
  action with a MORE severe one, never deselect a worse existing breach

That is a property of the arithmetic, not of care. FR-004 is satisfied structurally; SC-004 still
requires it demonstrated as a table, because a structural argument that was never run is still an
argument.

**Non-finite handling is inherited, not re-implemented**: `reflex.py` already treats a non-finite
reading as `inf` severity — "sensor fault = max danger, never ignore". A simulated NaN is therefore
maximally dangerous automatically. No new code, and it matches the existing contract for real
channels.

---

## D-4 — Simulator dynamics

**Decision**: deterministic per-instance `random.Random(seed)`, **default-OFF**, and drivable to the
danger limit on demand.

- **Per-instance seeding, never module-level.** The project's determinism invariant says
  `random.Random(seed)`, and specs/015 has just repaired a process-global reseed that leaked across
  `run_selfchecks`'s single process (measured: `[0.028451, …]` → `[0.022726, …]`). Reintroducing one
  here would undo that lane in the same week.
- **Default-OFF** (FR-014) so existing self-check output, coverage figures and recorded measurements
  do not silently change meaning.
- **Drivable to the limit** because the charge probe must exercise the reflex, not merely run the
  wire. An environment that never approaches `CONTACT` would exercise the pipe and prove nothing —
  the vacuity trap named in the spec's Edge Cases.
- **No new safety constant**: the simulator's danger limit IS `arm_config.CONTACT`. P1's fourth
  exception keeps it fixed, and reusing it means there is no second number to drift.

---

## D-5 — Commit sequence, envelope last

1. **C1 — simulated environment producer** (daemon). No envelope. Default-OFF.
2. **C2 — cockpit afferent endpoint**. No envelope. Mirrors `/api/act` exactly: validate on the
   request thread, enqueue, never touch brain/reflex/arm from it, refuse when served without a queue.
3. **C3 — tick-side consumer**: feed the produced/POSTed reading through the EXISTING
   `parse → encode → spikes` chain and into `AutonomousArm.step`'s `distance` argument. **This is the
   commit where the amygdala can first fire**, so the SIMULATED marker and the amendment to
   `fear_conditioning.py`'s disposition land here, in the same commit.
4. **C4 — reflex channel 8** (the safety envelope). Lands only after C1–C3 are green.
5. **C5 — dispositions**: dormancy exemptions settled per the probe result; coverage band pinned.

Envelope last because C1–C3 are reversible and carry no safety risk; C4 is the one commit where a
mistake reaches a protective mechanism.

---

## D-6 — The SIMULATED marker (R-001, the honesty risk)

**Decision**: the marker is a field on the reading itself, carried by every hop, and its absence is
a test failure rather than a default.

The risk is not that someone lies; it is that a marker set at the entry point gets dropped one hop
later and a simulated threat reads as real in the cockpit or a lake record. So the marker travels
WITH the value rather than being re-derived from "is the simulator on?" — a global flag can be true
while a specific reading came from elsewhere, and that ambiguity is the defect.

Checked at three hops: the brain's state report, any lake record, and the cockpit display. The test
constructs a simulated reading and asserts the marker survives to each; stripping it at any hop
fails.

---

## Open questions carried into Phase 1

None blocking. One is deferred to measurement rather than resolved by argument, per Standing
Invariant 6:

- **Does the feed actually CHARGE?** FR-010's neuter-A/B decides. `tools/charge_probe.py` is the
  template and carries an external positive control (47/60 flips with all drive sources neutered),
  which must be honoured — a null from an insensitive harness is not a null. If nothing charges,
  FR-012's revert executes and both exemptions are restored with corrected reasons. That outcome is
  pre-committed here, before the number is known.
