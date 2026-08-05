# Contract: `tools/red_reachability.py`

**Purpose**: assert every registered success criterion, marker, probe verdict or scored gate has a
**recorded input under which it reports FAIL**. Satisfies spec SC-001, SC-002, SC-003 (P8).

This is the executable form of falsifier F11 —
*"Any success criterion ... turns out to have no reachable red → That criterion is deleted, not
weakened."* — and of P8's honesty guard.

## Interface

```bash
python tools/red_reachability.py            # report; exit 0
python tools/red_reachability.py --assert   # exit 1 on any NO_RED without an exemption
```

Registered in `.github/workflows/ci.yml`'s **lint** job. Not in `run_selfchecks.py` (see plan
Structure Decision).

## What it checks, in order of decisiveness

1. **`reads_system`** — does the criterion take the system under test as a parameter? A criterion
   with a zero-argument signature is invariant to *every* change to the system, at every level of
   the computational hierarchy, and is therefore unfalsifiable in the strict sense of Kleiner & Hoel
   2020 (research R1). **This check alone is sufficient for `NO_RED`** and is the cheapest, so it
   runs first.
2. **`red_input` recorded** — is there a registered input under which the criterion reports FAIL?
3. **`red_demonstrated`** — has that input actually been executed and observed to fail? A recorded
   but never-run red is `UNKNOWN`, not `RED_REACHABLE`.
4. **`consumers`** — who reads the verdict. Drives severity, per P8: *an invented method's debt is
   proportional to how load-bearing its output is.* A verdict reaching a human outranks one read by
   nothing.

## Output format

```
==============================================================================
red_reachability -- a criterion that cannot fail is not a measurement
==============================================================================
scanned <N> criteria across <M> modules

--- NO_RED: no input exists that makes these report FAIL ---
  reasoning.consciousness_probe.probe_binding      zero-arg; 1.000/300 sets
      consumed by: reasoning/spike_chat_lake.py:929  -> REACHES A HUMAN
  reasoning.consciousness_probe.probe_reentry      zero-arg; arithmetic over literals

--- UNKNOWN: could not construct a failing input (hand-adjudicate) ---
  ...

--- RED_REACHABLE ---
  reasoning.consciousness_probe.probe_integration  ABSENT on 10/20 seeds
      WARNING: invocation pins seed=0, which is a passing seed

NO_RED: <n>   UNKNOWN: <n>   RED_REACHABLE: <n>   EXEMPT: <n>
```

## Expected FIRST RUN output (counted before adoption)

**This gate MUST fail on today's tree.**

| bucket | expected | note |
|---|---|---|
| NO_RED | **≥ 2** — `probe_binding`, `probe_reentry` | both zero-arg; BINDING measures 1.000 across 300 random sets, RE-ENTRY is `0.40 < 0.55 < 0.70` over literals |
| RED_REACHABLE but mis-invoked | **1** — `probe_integration` | reads ABSENT on ~10 of 20 subnet seeds; the shipped call pins `seed=0` |
| UNKNOWN | unknown a priori | this is the number that decides adoptability |

**Adoption rule**: if UNKNOWN dominates — the shape that killed a prior proposal at
0 failures / 47 UNCERTAIN — **the gate is dropped, not the tree**. A gate that mostly shrugs
manufactures confidence without measuring (Standing Invariant 5).

## Registration

A criterion opts in by being discoverable, not by being hand-listed (hand-maintained manifests rot —
the `_external_roots()` lesson). Discovery is by AST over functions whose name or return matches the
criterion shapes (`probe_*`, `*_present`, functions returning `(bool, str)` verdicts, scored gate
entrypoints), scoped to modules the module gate already proves reachable.

**Scope discipline, from the method-dormancy-gate precedent**: widening the scan beyond
already-reachable modules re-introduces a rubber-stamp-sized list. Keep it narrow; a narrow gate
that holds beats a broad one that gets exempted into meaninglessness.

## FIRST RUN, ACTUAL (2026-08-04) — counted before adoption

**Final on the live tree: NO_RED 0, TAUTOLOGICAL 0, READS_SYSTEM 13, EXEMPT 0.**

**A green first run is normally the failure signal, and here it is not — but only because the
reason is documented and independently proven.** The one real case (`consciousness_probe`'s
`probe_binding` / `probe_reentry`, plus `main()` asserting its own verdicts) was fixed under T012–T015
*before* this gate existed. The gate is therefore confirming that fix from the outside rather than
finding nothing.

That distinction is worthless unless demonstrated, so `tests/test_red_reachability.py` fires every
bucket against a fixture that **reconstructs the pre-fix `consciousness_probe` verbatim**:

- `test_NO_RED_fires_on_a_zero_argument_criterion` — flags both deleted probes
- `test_TAUTOLOGICAL_fires_on_assert_of_a_computed_verdict` — flags `bnd, _ = probe_binding(); assert bnd`
- `test_the_REPAIRED_shape_is_NOT_flagged` — the fix must not be punished as hard as the defect
- `test_live_consciousness_probe_now_reads_the_system` — regression guard on the real module

### Two instrument defects found by counting, both fixed

1. **Double counting.** First run reported 8 tautologies for 4 real ones: `ast.walk` yields `Module`
   *and* the `FunctionDef` inside it, and the scan walked both.
2. **It flagged its own repair.** The first version treated any assert *referencing* a computed
   verdict as tautological, which caught `assert acc is None` and `assert 0.0 < frac < 1.0` — the
   repaired self-checks. Narrowed to **bare truthiness** (`assert v`, `assert all(v)`) versus a
   comparison against an expected value. This is a false-positive correction, not a narrowing to
   dodge a true positive: `assert present` cannot go red, `assert 0.0 < frac < 1.0` demonstrably can.

   It also cleared `midbrain/dopamine_nuclei.py:223`, which was a **false positive**: the assert is
   `other.phasic(o_naive["REWARD"]) == naive_rew_rate`, a determinism check that merely references
   the value. Verified by reading the file, not by trusting the count drop.

### Known false-positive class in discovery, recorded not silenced

`VERDICT_NAMES` contains `present`, which collides with `DopamineNuclei.present(stimulus)` and
`competitive_learning.present(...)` — methods meaning "present a stimulus to the network", not
"is it present". They land harmlessly in READS_SYSTEM. **Not narrowed away**, per Standing
Invariant 2: narrowing the pattern to dodge them would also hide real criteria named `present`.

**Adoption status: BUILT, COUNTED, TESTED, and clean on the live tree.** Safe to register as
asserting in CI — unlike the dependency ledger, it has no outstanding findings to disposition first.

## Correctness risks

1. **A verb-match false positive** — a helper named `probe_*` that is not a criterion. Exempt by
   name with a reason, do NOT narrow the pattern; narrowing to dodge a true positive is forbidden by
   Standing Invariant 2, and would also hide real criteria.
2. **`UNKNOWN` silently treated as pass.** The single worst failure this instrument could have: it
   would make the gate green while criteria that cannot fail keep shipping. UNKNOWN must be reported
   distinctly and counted in the summary line.
3. **Running the criterion to test its red has side effects.** Criteria are executed in a subprocess
   with a seeded RNG; anything that writes to the lake is excluded from execution and reported
   `UNKNOWN` rather than run.

## Verify-the-verifier (Standing Invariant 5)

`tests/test_red_reachability.py` MUST make each assert FIRE:
- a fixture criterion with a zero-arg signature → reported `NO_RED`;
- a fixture criterion with a demonstrated failing input → reported `RED_REACHABLE`;
- a fixture whose red cannot be constructed → reported `UNKNOWN`, **not** `NO_RED`;
- an `EXEMPT` entry matching no real violation → gate FAILS loudly.

## What this gate deliberately does NOT do

It does not judge whether a criterion measures the *right* thing — only whether it can fail. A
criterion can be red-reachable and still be measuring something irrelevant. **No gate can tell you a
criterion is meaningful**, which is why P8's guard is review-enforced and this instrument is only
its cheapest half.
