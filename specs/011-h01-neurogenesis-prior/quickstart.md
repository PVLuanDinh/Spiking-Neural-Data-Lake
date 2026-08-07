# Quickstart: validating 011

**Feature**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06

How to prove this feature works, and — more importantly — how to prove each check can **fail**. Every
scenario below names its red. A check whose red has never been run is not evidence.

## Prerequisites

```bash
.venv-lake/Scripts/python.exe -c "import sys, sysconfig; print(sys.version, sysconfig.get_config_var('Py_GIL_DISABLED'))"
```

Expect Python 3.14.x and `Py_GIL_DISABLED = 1`. The package refuses to import on a GIL build.

Source data is **not** vendored wholesale — only the derived constants are, in
`cortex/cortical_prior.py`. Regenerating them from the upstream files is an offline step, never a
live path.

## The gates (all must stay green, every commit)

```bash
.venv-lake/Scripts/python.exe -m ruff check .
```

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest -q
```

Then the asserting instruments — note `p4c_ratchet` in particular, because this is the first feature
planned after it went asserting and **this feature must not raise the ungoverned count**:

```bash
.venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert
```

```bash
.venv-lake/Scripts/python.exe tools/p6_gate.py --assert
```

## Scenario 1 — the circuit is drivable (Phase A)

Present two distinguishable inputs to the layered circuit; confirm the responses differ. Present one
input twice; confirm the responses are identical.

**Red**: remove the input port and re-run. The first half must fail. If it still passes, the check is
measuring something other than the port.

## Scenario 2 — the drift instrument works before there is anything to bless (Phase B)

Run the three-arm comparison — seeded / random control / deliberately wrong prior — against the
*existing unseeded* growth. The instrument must distinguish the arms.

**Red, direction 1**: pin the seeded values so learning cannot move them. Drift reads zero and the
check **fails**, naming pinning as the cause.

**Red, direction 2**: seed from a deliberately incorrect prior. The seeded arm underperforms the
random control and the check **fails**, naming the prior.

Both reds are mandatory. A criterion that can only fail one way is not measuring what this feature
claims.

## Scenario 3 — placement, and the trap that produces a plausible wrong answer (Phase C)

Grow a population from a fixed seed. Confirm every unit carries a layer and a position, that the
distribution spans more than one layer, and that the fall-through layer holds no more than the stated
fraction.

**Red — run this one, it is the whole reason SC-002 exists**: supply the boundary geometry in the
wrong length unit (nanometres where microns are expected). **Every** unit lands in the fall-through
layer. The check must fail. Without this red, a units error ships looking like a valid assignment.

## Scenario 4 — sign is per cell and survives learning (Phase C)

Inspect every cell's outgoing connections: all carry one sign. Run learning to convergence: no sign
changed, magnitudes did.

**Red**: construct a cell with mixed outgoing signs — the check fails and names it. Separately, a run
in which **no magnitude changed** must also fail, so the check cannot pass vacuously on a brain that
never learned.

## Scenario 5 — placement charges behaviour (Phase C)

Neuter placement: pin positions and layers to a single value. Compare the brain's **own output**
against the placed run.

**Red**: identical outputs mean placement is inert, and the check fails. This is the P3 bar — a census
of correctly placed neurons is not evidence.

## Scenario 6 — persistence across a process boundary (Phase C)

Grow, exit the process, start a **new** one, read the placement state back from the lake. Positions,
layers and signs match.

**Red**: an in-memory round-trip that never crosses a process boundary must **not** count as passing.
"It persists" is a claim about a new process.

## Scenario 7 — provenance survives (all phases)

Every externally sourced value reported anywhere carries its units, origin and accuracy bound.

**Red**: strip the bound from one constant — the check fails and names it.

## Scenario 8 — the inhibitory tail (Phase D, if it ships)

Compare seeded excitatory and inhibitory strength distributions. The inhibitory tail is heavier.

**Red**: seed both from one pooled distribution — the check fails.

## What "done" means for a partial delivery

- **A only** — the layered circuit can be driven. Real, standalone, and needed regardless of H01.
- **A + B** — plus an instrument that can tell whether any future seeding earns its place.
- **A + B + C** — placed, signed, drift-measured neurogenesis. **This is the feature's core result and
  it is complete on its own terms.**
- **D** may never ship. Its bias is the least correctable, and the feature does not depend on it.
