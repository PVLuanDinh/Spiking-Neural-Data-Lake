# Phase 1 Quickstart — validating this feature

**Feature**: 020-eprop-cosine-scope | **Date**: 2026-08-17

Runnable scenarios that prove the feature works. Entities are in [data-model.md](./data-model.md);
the CLI surface is in [contracts/cli.md](./contracts/cli.md). No implementation code here.

## Prerequisites

```bash
cd D:\AI-project\spiking-neural-data-lake
```

Python is `.venv-lake\Scripts\python.exe` (3.14.6 free-threading, torch 2.11.0+cu128). The static
half needs no GPU and no torch. The `--measure` half needs torch and runs on CPU.

---

## Scenario 1 — the census reports, and the counts reconcile (US1)

```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate
```

**Expect**: a census whose per-state counts sum to `rules seen`, `unclassified == 0`, and a non-zero
`rules seen`. A known local rule (any STDP variant) appears as `CLASSIFIED_LOCAL`; the e-prop
reference appears as `CLASSIFIED_REFERENCE`, not as a claimant.

**Fails if**: totals do not reconcile, or `rules seen` is 0 — an empty census and a broken walk look
identical, which is why the floor exists.

---

## Scenario 2 — the gate asserts, and its self-check constructs real reds (US1, Invariant 5)

```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --assert
```
```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --self-check
```

**Expect**: `--assert` exits 0 on a clean tree. `--self-check` exits 0 having CONSTRUCTED each red
listed in the contract — including deleting a file from the walk to fire the liveness floor, and a
GREEN control proving the repaired shape is not flagged.

**Fails if**: `--self-check` only runs the happy path. A self-check that never constructs a failure
verifies nothing — this repo has four gates that pass that way today.

---

## Scenario 3 — the baseline refuses to launder a regression (FR-018)

```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --update-baseline
```

**Expect**: refuses with exit 1 when the current count is worse than the recorded baseline, naming
the offending counts.

**Why this scenario exists**: measured on the existing `p6_gate`, a ceiling raised *together with the
regression it excuses* passes BOTH `--assert` and `--self-check` — verdict **laundered clean**. This
is the scenario that separates a ratchet from a rubber stamp.

---

## Scenario 4 — a rule is never condemned by an instrument that cannot judge it (US2, SC-003)

```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --measure
```

**Expect**: every judged rule reports a PAIR (aligned + null), never a lone number. Reconstructing the
`mlp_q` feedback-alignment configuration yields `INSTRUMENT_FALSIFIED`, **not** a rule failure — that
case has a known correct answer already reached by hand, so it is the reachable red for the checker
itself.

**Fails if**: any rule is reported as failing on a criterion that no constructed input could make it
pass.

---

## Scenario 5 — where 0.5 actually sits, and the bar does not move (US4, SC-013)

Included in `--measure` output.

**Expect**: the null is `within_matrix_permutation`; `0.5` reported as sd-above-null with
P(null > 0.5); the three reported-only nulls present and gating nothing. **`git diff` shows the
numeric bar unchanged.**

**Why the null is named**: measured, a permutation null puts `0.5` +4.5 to +63 sd above it
(P ≤ 3e-5), while a shuffled-label null puts it 0.32–2.55 sd BELOW the null mean (P = 0.65–0.98).
Opposite verdicts on the same gradients — an unnamed null lets the implementer choose the answer.

---

## Scenario 6 — the reconciliation, and what it must not break (US3)

```bash
.venv-lake\Scripts\python.exe -m snn_data_lake.cortex.eprop_kernel
```
```bash
.venv-lake\Scripts\python.exe -m pytest tests/test_eprop_train_tier_unreachable.py tests/pin -q
```

**Expect**: kernel self-check green with BOTH its reds intact; both encoder self-checks byte-identical
to baseline; 5 tests pass.

**Highest-risk failure**: `test_the_reference_gradient_is_not_a_hazard` asserts `cortex.eprop_kernel`
is in BRAIN_MODULES, is reachable, and is NOT a "hazard" (one function calling both `.backward()` and
`.step()`). The reconciliation reds this if it puts an optimizer step alongside `_fwd_bwd`'s
`.backward()`.

**Also required**: the seven recorded numbers inside `eprop_kernel` comments change and must be
re-derived (cos_hidden 0.9714→0.9993, the "90.7%" figure→81.2%, three seeded-defect rows). **`mlp_q`
carries a second copy of the 90.7% figure** — co-edit, or the claim lives in two places and gets
corrected in one.

---

## Fast loop — between tasks (72s measured)

```bash
.venv-lake\Scripts\python.exe -m ruff check . ; .venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --assert ; .venv-lake\Scripts\python.exe -m snn_data_lake.substrate.gradient_claim_gate --self-check ; .venv-lake\Scripts\python.exe -m pytest tests/test_gradient_claim_gate.py --no-cov -q
```

**Expect**: four exit codes, all 0. Measured 72s against the full gate's ~1795s — **25x cheaper**, and
it covers everything one task can plausibly break. What it does NOT see is a cross-module regression
or the coverage floor, which is why the full gate below still runs at every PHASE boundary and before
any commit touching `src/`.

## Full gate before commit

```bash
.venv-lake\Scripts\python.exe -m ruff check . ; .venv-lake\Scripts\python.exe run_selfchecks.py ; .venv-lake\Scripts\python.exe -m pytest -q
```

Capture each exit code separately — a chained run reports only the last, which has burned this project
before. Plus `python -m snn_data_lake.substrate.doc_budget_gate --assert` if `docs/POLICIES.md`
changed: headroom is **541 words**, and real policy blocks in that file run 548–747 words, so the
amendment must land near **250 words** or raise the ceiling in the same commit with its reason.
