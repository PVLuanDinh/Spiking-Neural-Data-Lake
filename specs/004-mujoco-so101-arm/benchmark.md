# P2 Benchmark Verdict: MuJoCo SO-101 vs ArmSim

**Run**: 2026-08-01 · `tools/arm_benchmark.py` · `mujoco==3.11.0` · commit `2a6b4555`
**Requirement**: FR-001 (a benchmark must exist before any deletion), FR-010 (a null must be
recordable), FR-019 (torque must be reportable), SC-001, SC-002.

## Verdict

**The replacement is LICENSED under P2.** The two mechanisms do the same job, and the candidate
can report a quantity the incumbent structurally cannot. That is not a better score on a shared
metric — it is a capability that does not exist in the incumbent at all.

This is a licence to proceed with the replacement, **not** a claim that the arm is finished or
that the physical build is viable. Those are separate findings, and one of them is negative.

## What each body is

| | ArmSim (incumbent) | SO-101 (candidate) |
|---|---|---|
| degrees of freedom | 2 | 6 |
| units | dimensionless (`l1=l2=1.0`) | metres |
| integration | none — a command teleports the joint | 0.002 s step |
| reach | 2.0 dimensionless | **0.4784 m** horizontal (sampled) |
| holding torque | **cannot be measured** | per-joint, N·m |

The two are not commensurable, so no single "winner" number is reported. Each row states what
each arm can report and names what it cannot. The capability delta is the finding.

## The capability delta

`ArmSim` reports torque as `None`, not `0.0`, and the distinction is the point: zero would be a
claim about physics — that no torque is needed to hold a pose — while `None` states the quantity
does not exist in a body with no mass, no gravity and no timestep. Conflating them is how a
kinematic toy comes to look like it passed a torque check. A test pins this.

## The build notes' actual question, answered

*Can STS3215 servos hold 500 g at reach?* **No — three joints exceed their continuous rating.**

Per-joint hold torque with 0.5 kg at the tool centre point, worst case over the reachable
envelope (20 000 sampled poses):

| joint | worst N·m | kg·cm | rated kg·cm | × rated | % of stall |
|---|---|---|---|---|---|
| shoulder_pan | 0.000 | 0.00 | 7.5 | 0.00× | 0% |
| **shoulder_lift** | 2.873 | 29.29 | 7.5 | **3.91×** | 86% |
| **elbow_flex** | 1.896 | 19.33 | 4.2 | **4.66×** | 57% |
| **wrist_flex** | 0.899 | 9.16 | 4.2 | **2.21×** | 27% |
| wrist_roll | 0.037 | 0.38 | 3.2 | 0.12× | 1% |
| gripper | 0.004 | 0.04 | 2.0 | 0.02× | 0% |

**Over continuous rating**: `shoulder_lift`, `elbow_flex`, `wrist_flex`.
**Over stall**: none.

A joint above rated but below stall will move, then heat, then trip thermal cutoff (~70 °C per
the build notes). "It moved in the simulation" is not "the servo holds it".

### The trap this nearly walked into

The model's `actuator_forcerange` is **3.350 N·m = 34.1 kg·cm**, which is **stall** torque — it
matches the notes' "~30 kg·cm stall at 12 V" for the 1:345 gear. The first version of this
benchmark compared against that figure and reported `shoulder_lift` at 86% utilisation, which
reads as comfortable headroom. Against **rated** torque, the same load is **3.91× over**.

The build notes state the rule themselves: *"Size servos to rated torque, not stall torque.
Rated is roughly a quarter of stall."* Both columns are now shown, and the rated column is the
one that decides the purchase.

## Independent corroboration

Two unrelated routes agree:

- `arm_hardware`'s analytic 300 mm lever: shoulder needs **15.0 kg·cm** vs 7.5 rated → `ok=False`
- MuJoCo full rigid-body model at worst pose: shoulder needs **29.29 kg·cm** vs 7.5 rated → 3.91×

These are **different quantities** — the table is payload-only against a fixed lever, the model
includes the arm's own mass at full extension — so equality would be suspicious. Agreement in
direction, and the model's larger magnitude, is the corroboration. The repo's existing figure was
not wrong; it was a lower bound.

## Method notes

- The payload is applied **analytically**, `τ = qfrc_bias + Jᵀ·F` at the `gripperframe` site. The
  vendored model is never mutated (FR-012), and a test asserts body count and total mass are
  unchanged after the calculation.
- Worst case is **sampled**, not solved — there is no closed form at 6 DOF, and a corner search
  over joint limits understates it because the maximum is interior (corner search gave 0.3961 m
  reach against the sampled 0.4784 m).
- The benchmark runs against a **bare** MuJoCo body and the existing `ArmSim`, with no dependency
  on the replacement gate — so it could license the work before the work existed.

## What this verdict does NOT say

- It does not say the arm is safe to drive. E-STOP, clamping and the collision contract are gate
  properties; the bare body has none, and the benchmark says so rather than omitting the row.
- It does not say the physical build is viable. It says the opposite for three joints, at 500 g.
- It does not license deleting `arm_sim.py`. P2 also requires port-before-retire, which is US5.
