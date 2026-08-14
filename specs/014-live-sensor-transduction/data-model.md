# Phase 1 Data Model — Live Simulated Sensory Transduction

Shapes measured by running the chain in `.venv-lake`, not read off the source.

---

## E-1 — The reading, and its two consumers

One parsed line feeds **two different consumers with two different values**, and conflating them is
the design's main trap.

```
"0.030, 298.1"                      the wire format: "distance_m, temp_k", two comma-separated floats
   |
   +-- parse_esp32_line  -> (0.03, 298.1)        RAW, metres + kelvin
   |      |
   |      +-- normalize  -> (0.01, 0.2475)       NORMALIZED [0,1], temp None when missing
   |             |
   |             +-- encode_reading -> 64 floats  --> spikes_for -> 7 (time, feature) events
   |                                                   [SPIKE PATH: perception]
   |
   +-- RAW distance_m (0.03) ------------------------> AutonomousArm.step(distance=...)
                                                       [REFLEX/AMYGDALA PATH: safety]
```

**Measured** (`.venv-lake`, chain run end to end):

| input line | parse | normalize | feats | spikes |
|---|---|---|---|---|
| `"0.030, 298.1"` | `(0.03, 298.1)` | `(0.01, 0.2475)` | 64 | 7 |
| `"0.500, 298.1"` | `(0.5, 298.1)` | `(0.1667, 0.2475)` | 64 | 8 |
| `"-1, -1"` | `(-1.0, -1.0)` | `(0.0, None)` | 64 | 3 |
| `"STOPPED"` | `None` | — | `None` | — |
| `"garbage"` | `None` | — | `None` | — |

**The safety path takes the RAW metres, never the normalized value.** `CONTACT = 0.04` is expressed
in metres; `distance01` is `distance_m / 3.0` clamped to `[0,1]`, so the entire danger zone occupies
the bottom **1.333%** of the normalized range. Comparing a normalized value against a metres
threshold would put the trip point at 12 cm instead of 4 cm — a silent 3× loosening of a safety
limit. The two values travel together and are never substituted for each other.

---

## E-2 — THE IDLE SENTINEL IS A REAL HAZARD, AND IT IS THIS FEATURE'S

**Found by running the chain, not by reading it. The naive wire ships a defect.**

`parse_esp32_line`'s docstring: the firmware sends `-1` for distance when **nothing is within 3 m**
— the sentinel means *farthest*, i.e. maximally safe. But:

```
raw:         -1.0 < CONTACT (0.04)          -> True
             math.isfinite(-1.0)            -> True
  => AutonomousArm.step's branch  `distance is not None and (not isfinite or distance < CONTACT)`
  => True. THE CONTACT BRANCH FIRES ON "NOTHING DETECTED".

normalized:  -1.0 / 3.0 = -0.333 -> clamped -> 0.0   (measured above)
  => in feature space the sentinel reads as MAXIMUM PROXIMITY.
```

`normalize()` guards the sentinel for **temperature** (`temp_k <= 0 -> None`, an explicit missing
modality) and **does not** for distance — it clamps. Both paths therefore invert the sentinel's
meaning, and on the safety path that means **fear conditioning on an empty room**: the amygdala
would learn that "nothing there" is the threat, and `threat()` would go non-zero for the most
misleading possible reason on the very first tick.

**Requirement (new, promoted out of this measurement):** the consumer **rejects a sentinel/negative
distance before it reaches `step()`**. It is a missing reading, exactly like `temp_k <= 0`, not a
close one. A test constructs `"-1, -1"` and asserts the contact branch does NOT fire.

**Why this is not fixed inside `normalize()`**: changing `normalize`'s clamping semantics would alter
the feature vector every existing caller receives, and this feature's non-goals forbid rewriting the
governed chain (P2: reuse, do not modify). The guard belongs in the new consumer, where the raw
metres are read.

**Disclosed rather than smoothed (P8 guard 4):** this defect exists in the tree *today* and is
latent only because nothing ever calls `step()` with a real sensor value. This feature is what makes
it reachable, so this feature owns it.

---

## E-3 — Reflex frame layout

Rules are index-keyed into a raw sample list (`brainstem/reflex.py`).

| index | meaning | rule | status |
|---|---|---|---|
| 6 | motor current | `(-50.0, 800.0, "STOP")` | **UNCHANGED — pinned by test** |
| 7 | force / torque | `(-300.0, 300.0, "WITHDRAW")` | **UNCHANGED — pinned by test** |
| **8** | **simulated proximity (metres)** | **added, `"WITHDRAW"`** | new |

Invariants:

- New channels take indices **strictly above** the existing ones. Never insert, never renumber.
- A frame shorter than index 8 is already handled: an absent rule channel is reported once on stderr
  as INACTIVE, never silently skipped. No new code, and the simulator being default-OFF means the
  normal frame stays exactly as long as it is today.
- Severity is `over / max(1e-9, hi - lo)`, non-negative; `current` sums across channels; membrane is
  `v = v * decay + current`; firing is `v >= threshold or _breach >= persist`. **An added channel can
  only raise the membrane, so it fires no later and never later.** See research.md D-3.
- Non-finite reading → severity `inf`, inherited unchanged ("sensor fault = max danger").

---

## E-4 — The SIMULATED marker

The marker is a **field on the reading**, not a global "is the simulator on?" flag. A global can be
true while a specific reading came from elsewhere; that ambiguity is the defect R-001 names.

| hop | carries the marker as | test |
|---|---|---|
| producer (daemon simulator) | set at construction, never defaulted | constructed reading has it |
| cockpit POST | required field; a reading without it is **rejected**, not defaulted | POST without it → 4xx |
| tick-side consumer | read from the reading, passed on | — |
| brain state report | explicit `simulated` field | present and `True` |
| lake record (if persisted) | inside the record body under its own tag | round-trip preserves it |
| cockpit display | rendered next to the value | string contains the marker |

**Absence is a failure, not a default.** Defaulting a missing marker to `False` would make a dropped
marker read as REAL — the single worst outcome. Defaulting to `True` would be safe but would let a
real reading be mislabelled once hardware exists. Rejecting is the only option that cannot silently
mislabel in either direction.

---

## E-5 — Simulator state

| field | type | note |
|---|---|---|
| `enabled` | bool | **default False** (FR-014) |
| `rng` | `random.Random(seed)` | **per-instance**, never module-level (specs/015 just fixed exactly that leak) |
| `distance_m` | float | current simulated distance, metres |
| `temp_k` | float | current simulated temperature, kelvin |
| `forced` | float or None | test/probe hook: drive straight to a value so the charge probe can reach the danger zone |

The simulator emits the **wire format** (`"distance_m, temp_k"`) so it enters through
`parse_esp32_line` like any other line. It does not construct feature vectors directly — that would
be a second transduction mechanism and a P2 violation. **The simulator's danger limit is
`arm_config.CONTACT` itself**; no new safety constant is introduced.
