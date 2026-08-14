# Implementation Plan: Live Simulated Sensory Transduction

**Branch**: `014-live-sensor-transduction` | **Date**: 2026-08-13 | **Spec**: [spec.md](spec.md)

**Input**: [spec.md](spec.md) — 14 FRs, 5 non-goals, 9 success criteria, 4 risks, plus a recorded
owner authorization to let a simulated sensor feed reach the safety envelope.

## Summary

The transduction chain exists and is governed (`substrate/esp32_encoder_path.py`); it has zero
consumers and a standing dormancy exemption doing duty for the missing half. The cockpit — which
P4c calls the brain's sensorimotor surface — has no afferent sensor channel. This feature adds a
simulated environment produced by the daemon, an afferent cockpit endpoint, and a tick-side consumer
that runs the EXISTING chain, and lets the result reach the reflex arc.

**The consumer turned out to already exist, and that is the plan's central finding.**
`brainstem/autonomous_loop.AutonomousArm.step` already calls the amygdala's `pair()` inside its
safety-first contact branch — `if distance is not None and (not math.isfinite(distance) or distance
< CONTACT)`. A distance reading below `CONTACT` starts fear conditioning **with no new wiring at
all**. There is nothing to invent; there is something to feed.

## Technical Context

**Language/Version**: Python 3.14.6 free-threading. **Gate env `.venv-lake`** (torch 2.11.0+cu128;
**`tonic` ABSENT**, **`pyserial` ABSENT** — both measured). CI has no GPU and no hardware; nothing
in this feature may require either.

**Primary Dependencies**: none added. The simulator is stdlib; the transduction chain already ships.

**Storage**: `brain_lake.bin` only, under its own record tag (P4b). No second store.

**Testing**: `pytest` + `run_selfchecks.py`. New self-checks join CORE.

**Constraints**: default-OFF simulator (FR-014); safety envelope touched **only** in
`brainstem/reflex.py`, and only to ADD a channel rule.

**Scope**: 3 P1 stories + 1 P2; ~5 commits; one safety-envelope file.

## THE OWNER DECISION THAT SHAPES EVERYTHING BELOW

`amygdala/fear_conditioning.py` carries, in its own docstring:

> **DISPOSITION: P3 "exempt — awaiting a prerequisite" (a real proximity sensor and a real body),
> NOT "wire it."** … a semantic change to a safety-adjacent faculty, made to move a coverage number.
> **That is the theater P0's honesty guard exists to refuse.**

Measured there: `threat()` runs 38 times per 80 wake ticks and returns **exactly 0.0 every time** —
"the HONEST output of a brain that has never been hurt."

A simulated sensor is not "a real proximity sensor and a real body". Feeding it would make `threat()`
non-zero for the first time, and `amygdala_fear_conditioning` currently scores **band 0**. Asked
explicitly, the owner ruled:

> **YES — the simulated feed may reach the amygdala, but NOTHING MAY BE SCORED ON IT.**

**Binding consequences, and they are not negotiable later in this lane:**

1. Simulated contact drives `pair()`. The semantics survive: the amygdala conditions on **contact**,
   which is nociception, not on disappointment. That distinction is exactly what the module's note
   was protecting, and it is preserved.
2. **Every threat value derived from a simulated event is marked SIMULATED** at every hop.
3. **The coverage band stays 0.** No band, anywhere in `docs/coverage/`, may rise on simulated
   evidence in this lane. This is the condition the authorization was given under.
4. The module's DISPOSITION note is **amended, not silently overtaken** — it must say that a
   simulated body satisfies the prerequisite for LEARNING but not for SCORING, and that the owner
   authorized that distinction on 2026-08-13.

## Constitution Check

*GATE: must pass before Phase 0. Re-checked after Phase 1 — see bottom.*

| principle | verdict | basis |
|---|---|---|
| **P0** self-improvement | **ADVANCES** | A brain that has never been hurt cannot learn aversion. This gives it an environment in which aversive events exist at all — a channel that was built, exempted, and inert. |
| **P1** no-hardcode | **EXCEPTION (d) APPLIES, DELIBERATELY** | The proximity danger limit is a raw physical limit inside the safety envelope and stays FIXED. P1: "a reflex threshold that drifts with a running baseline is a reflex that can be trained away." The simulator reuses the EXISTING `arm_config.CONTACT`; it introduces no new safety constant. |
| **P2** non-additive | **PASS** | Exactly one transduction mechanism: the existing `parse → encode → spikes` chain. Writing a second parser would be the violation. |
| **P3** no dormancy | **THE BAR OF THIS FEATURE** | Two exemptions are in play: `esp32_encoder_path` ("awaiting hardware") and `fear_conditioning` ("awaiting a prerequisite"). Neither may be cleared by a wire that does not CHARGE. FR-010's neuter-A/B decides; FR-012's revert is pre-committed. |
| **P4 / P4b** one brain, one lake | **PASS** | The daemon constructs the one brain via `make_brain()` and drives it; it gains no cognition and no store. Anything persisted goes to `brain_lake.bin` under its own tag. |
| **P4c** every build is the brain | **THE FEATURE'S PREMISE** | The cockpit is the sensorimotor surface; afferent transduction belongs there. This closes the gap between that claim and a cockpit whose only afferent channel is chat. |
| **P5** GPU | **N/A** | No compute-shape change; stdlib simulator. |
| **P6** faculty-to-faculty | **DO NO HARM** | This feature adds no cross-region import; the p6 ratchets must be unchanged, measured not assumed. |
| **P7** dependencies | **N/A** | None added. `pyserial` stays absent (NG-002). |
| **P8** invent the method | **BINDS** | Every new criterion owes a demonstrated reachable red; the "only adds caution" claim owes a measured table, not an argument; a null result triggers deletion (FR-012). |

### Standing Invariants

| # | verdict |
|---|---|
| 1 | **MET** — the "only adds caution" table is the gate's output, counted before the claim is made. |
| 2 **never silence a check** | **APPLIES.** Two exemptions get touched. Removing an exemption because a wire exists — rather than because it charges — is silencing. FR-010 gates both. |
| 3 **the negative verdict is dangerous** | **APPLIES INVERTED.** Here the risky verdict is a POSITIVE one: declaring the amygdala "now learning" on simulated evidence. The owner's no-scoring condition is the mitigation. |
| 4 absence of warning ≠ absence | **MET** — the caution table sweeps breach scenarios rather than inferring from one. |
| 5 **verify the verifier** | **APPLIES.** `tools/charge_probe.py` carries an external positive control (47/60 flips) proving the harness is sensitive at all. FR-010's probe must use it, or demonstrate its own sensitivity, before any null is believed. |
| 6 runtime > static | **MET** — every claim here is a measurement. |
| 7 **safety envelope** | **AUTHORIZED, SCOPED.** Only `brainstem/reflex.py`, only to ADD a rule. The other four files are untouched and the plan must verify that with a diff, not a promise. |
| 8 e-prop not backprop | **N/A** — no learning rule changes. |

**Verdict: PASS**, with the amygdala decision recorded above as an owner ruling rather than an
engineering choice.

## Project Structure

```text
specs/014-live-sensor-transduction/
├── plan.md          # this file
├── spec.md
├── research.md      # Phase 0 — the decisions and their measurements
├── data-model.md    # Phase 1 — reading shape, channel layout, marker propagation
├── contracts/
│   └── sensor-endpoint.md
├── quickstart.md    # Phase 1 — the runnable gate + probe procedure
└── tasks.md         # Phase 2 (/speckit-tasks)

src/snn_data_lake/
├── substrate/
│   ├── esp32_encoder_path.py   # UNCHANGED (P2: reuse, do not rewrite)
│   ├── cockpit.py              # + afferent sensor endpoint (the /api/act pattern)
│   ├── learn_daemon.py         # + simulated environment producer, + tick-side consumer
│   └── dormancy_gate.py        # exemption disposition settled
├── brainstem/
│   ├── reflex.py               # + ONE channel rule. The only envelope file touched.
│   └── autonomous_loop.py      # consumer ALREADY EXISTS (AutonomousArm.step contact branch)
└── amygdala/
    └── fear_conditioning.py    # DISPOSITION note amended per the owner ruling
```

**Structure Decision**: the simulator lives in the daemon because P4c makes the daemon the brain's
DRIVE — the thing that gives it continuity. An environment the brain lives in belongs to its drive
loop, not to a trainer.

## Complexity Tracking

| Violation | Why needed | Simpler alternative rejected because |
|---|---|---|
| Touching a safety-envelope file | The authorized capability *is* "the feed reaches the safety envelope". It cannot be delivered without adding a reflex rule. | **Keep the feed out of the reflex** — that was offered to the owner as the narrower option and explicitly declined in favour of reaching the envelope. |
| A simulated signal drives a safety-adjacent faculty | Same: without it the amygdala remains a faculty that cannot learn because nothing ever hurts. | **Wait for real hardware** — the arm is not built; this would defer the capability indefinitely. The owner's no-scoring condition is what makes the simulated path honest rather than theatrical. |

## Post-Design Constitution Re-Check

- **P3 / Standing Invariant 2** — [research.md](research.md) D-1 records that BOTH exemptions are
  gated on the same neuter-A/B, and that a null reverts rather than re-homes the wire. **PASS.**
- **P8** — [quickstart.md](quickstart.md) schedules a demonstrated red for each new criterion, and
  the caution table as a measured matrix. **PASS.**
- **Owner condition (no scoring)** — [data-model.md](data-model.md) makes the SIMULATED marker a
  field on the reading that every downstream hop carries, and quickstart pins the coverage band at
  0 as an explicit check rather than an intention. **PASS.**
- **Safety envelope scope** — quickstart's final step diffs `brainstem/` and fails on any file other
  than `reflex.py`. **PASS.**

**No new violations introduced by the design.**
