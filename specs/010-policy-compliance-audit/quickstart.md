# Quickstart: validating Policy Compliance (P0-P8)

How to prove this feature works end to end. Every scenario below states what it looks like
**before** the fix and **after**, because a validation guide whose steps pass either way validates
nothing.

## Prerequisites

```bash
cd D:/AI-project/spiking-neural-data-lake
```

Gate environment (CPU, deterministic): `.venv-lake/Scripts/python.exe`.
GPU environment (equivalent, optional): `.venv-cu128/Scripts/python.exe`.
CI has no GPU — every command here must work in `.venv-lake`.

## Scenario 1 — the two new gates fail on the unfixed tree

This is the first thing to run, and it must FAIL. A green result here means the instruments are not
measuring what they claim (plan Gate 2, Standing Invariant 5).

```bash
.venv-lake/Scripts/python.exe tools/dependency_ledger.py
```

**Before**: reports `UNDECLARED` (≥1 distribution, unguarded top-level), `UNUSED` (≥2), and a
self-description mismatch. **After**: all buckets empty or exempted-with-reason; `--assert` exits 0.

```bash
.venv-lake/Scripts/python.exe tools/red_reachability.py
```

**Before**: reports `NO_RED` ≥ 2, and flags one `RED_REACHABLE` criterion whose invocation pins a
passing seed. **After**: `NO_RED` is 0; deleted criteria are gone rather than re-thresholded.

Expected first-run counts are recorded in [contracts/](./contracts/). If either gate returns an
allowlist-sized list instead of a work list, **drop the gate, not the tree.**

## Scenario 2 — a clean install actually works (SC-004)

The defect this catches breaks arrival, and no existing gate sees it.

```bash
python -m venv /tmp/sndl-clean
/tmp/sndl-clean/Scripts/python.exe -m pip install .
/tmp/sndl-clean/Scripts/python.exe -c "import snn_data_lake, importlib, pkgutil; [importlib.import_module(m.name) for m in pkgutil.walk_packages(snn_data_lake.__path__, 'snn_data_lake.')]"
```

**Before**: `ModuleNotFoundError` on an undeclared third-party package.
**After**: completes with no error.

Note: some modules import optional-extra packages by design. Those must fail *gracefully* (guarded
import) or be declared in an extra — an unguarded top-level import of an undeclared package is the
defect.

## Scenario 3 — the brain stops answering a question it cannot measure (SC-001, SC-003)

The headline scenario. Run the markers against a healthy brain and a degraded one:

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.consciousness_probe
```

**Before**: prints `4 of 4 PRESENT`, and prints it identically no matter what the brain is doing —
the probe functions accept no brain at all. Deleting the `Brain` class would not change the output.

**After**: the surviving criteria take the brain as an argument, so a healthy and a degraded brain
give different results; criteria with no reachable red are absent, not re-thresholded; the
surviving multi-seed criterion reports its distribution rather than a pinned passing seed.

Then the live path:

```bash
.venv-lake/Scripts/python.exe learn_daemon.py     # then ask "are you conscious" in the cockpit chat
```

**Before**: a confident 4-of-4 verdict.
**After**: a reply stating what is measured and what is not. **This is the point of the feature**,
and it is not a retreat from the sentience goal — it removes a painted finish line so the real one
stays findable. Per P8: the brain may report what it measures, never what it cannot.

## Scenario 4 — the cockpit marks what cannot vary (SC-007)

```bash
.venv-lake/Scripts/python.exe learn_daemon.py     # http://127.0.0.1:8137/
```

For each displayed number, ask whether any brain state changes it.

**Before**: at least one headline value cannot vary and carries no provenance tag, sitting directly
above two values that do carry one — so the untagged number reads as the more trustworthy, which is
backwards. **After**: every displayed value either varies or is labelled as unable to.

## Scenario 5 — observing does not become the load (SC-008)

```bash
.venv-lake/Scripts/python.exe tools/charge_probe.py --ticks 120     # baseline, no cockpit polling
```

Compare tick rate with the cockpit open and polling versus closed. **Before and after**: no
significant difference. This scenario is a regression guard, not a fix — a ~1 s census once sat
inside `snapshot()` and starved the tick loop while every gate stayed green, so it is checked
explicitly rather than assumed.

## Scenario 6 — the three gates, at every commit (SC-009)

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```
```bash
.venv-lake/Scripts/python.exe -m pytest
```
```bash
.venv-lake/Scripts/python.exe -m ruff check .
```
```bash
.venv-lake/Scripts/python.exe tools/citation_guard.py
```

All must be green at **every** commit this feature produces, not merely at the end — the
cross-cutting invariant is one change per commit so a regression isolates to its own change.

**Note on the self-check count**: it is 278 in `.venv-cu128` and 277 in `.venv-lake` (nengo present
in the CUDA env runs `NENGO_GATED` instead of skipping). Both are correct; a changed count is not.

## Scenario 7 — the safety envelope is untouched (SC-010)

```bash
git diff --stat HEAD -- src/snn_data_lake/brainstem/reflex.py src/snn_data_lake/brainstem/serial_arm.py src/snn_data_lake/brainstem/so101_arm.py src/snn_data_lake/brainstem/arm_config.py src/snn_data_lake/brainstem/interpreter.py
```

**Expected: empty output.** Any change fails SC-010. These five files are report-only in this
feature; findings against them are recorded and left for explicit owner approval.

## What "done" looks like

- Scenario 1 gates both green under `--assert`, having been red first.
- Scenario 2 completes with no import error.
- Scenario 3 gives *different* answers for different brain states, or says it cannot measure.
- Scenarios 4–7 pass.
- Every applied change traces to a finding with a recorded verification verdict (FR-011), and every
  refuted finding is retained as refuted rather than dropped.
