# Quickstart: validating the Brain-Native Assembly-Calculus Language Organ

**Feature**: `specs/005-language-organ` | **Contracts**:
[language_organ.md](./contracts/language_organ.md), [lang_ab.md](./contracts/lang_ab.md)

Runnable scenarios, in the order they become meaningful. Each names the requirement it validates and what
a failure means. Nothing here is a substitute for the gates; it is what you run to believe the result.

> ## 2026-08-02 — ONLY THE FIRST SCENARIO IS RUNNABLE. The rest describe things that were never built.
>
> The feature stopped at the T011 hard stop with **`NOT-P2-ELIGIBLE`** — see [`verdict.md`](./verdict.md)
> and [`coverage.md`](./coverage.md). `tools/lang_coverage.py` exists and reproduces byte-identically;
> `src/snn_data_lake/cortex/language_organ.py` and `tools/lang_organ_ab.py` **do not exist**, and
> `cortex/language_acquisition.py` is present and unmodified.
>
> The two commands that work today, both from the repo root:
>
> - `.venv-lake\Scripts\python.exe tools/lang_coverage.py` — exits **2** with `NOT-P2-ELIGIBLE`, which
>   is the tool doing its job.
> - `.venv-lake\Scripts\python.exe -m pytest tests/test_lang_coverage.py -q --no-cov` — 11 of 11.
>
> Everything below that names the organ or the A/B harness is kept as the interface a future attempt
> must satisfy, not as instructions. Do not read an unrunnable scenario here as a missing step.

## Prerequisites

- Interpreter: `.venv-lake\Scripts\python.exe`. **Not** the `python` on PATH.
- `PYTHONPATH=src` for anything invoking `-m snn_data_lake.*`.
- BLiMP data at `data/lang/blimp/` (67 `.jsonl` files). Absent, the harnesses SKIP cleanly at exit 0 -
  **and a skip is never a pass**.
- The always-on daemon may be running. Every probe below isolates `NORD_HOME`; if you run something by
  hand that does not, its number is noise.

Below, `PY` means `.venv-lake\Scripts\python.exe`.

## The four gates (green before and after every step)

```
PY run_selfchecks.py
PY -m pytest
PY -m ruff check .
PY -m ruff format --check src tests tools
```

`pytest` enforces `--cov-fail-under=70`. **No assertion may be loosened and no check silenced to make
any of these pass** (Standing Invariant 2). If a gate goes red, the change is wrong, not the gate.

---

## Scenario 1 - the instrument can fail, on the exact set the verdict will use (FR-002, SC-001)

```
PY tools/lang_coverage.py
```

**Expect**: the rule printed verbatim; `P of 67` paradigms and `N of M` pairs; the three degenerate
scorers run on the restricted set with `random_scorer` within 3 points of 50%, `constant_scorer` at 0%,
and `length_scorer`'s leakage printed as a number.

**Exit 2 is a legitimate outcome.** It means a floor was missed (fewer than 8 of 67 paradigms, or fewer
than 200 pairs) or a control passed where it must fail. Either way the answer is `NOT-P2-ELIGIBLE` and
the feature stops. **It is not a signal to widen the corpus and re-run** - choosing the pair set to get a
verdict is p-hacking with extra steps, and R1 states the floors before the measurement precisely so they
cannot be moved after it.

**If `random_scorer` drifts off 50% on the restricted set**, the restriction introduced a leak that the
full set averaged out. The set is disqualified. This is the check that stops a subset from silently
becoming a green light that cannot go red.

---

## Scenario 2 - the baseline is real and reproducible (FR-003, SC-002)

```
git diff --quiet HEAD -- src/snn_data_lake/cortex/language_acquisition.py \
    src/snn_data_lake/sensorimotor/spiking_lm_serve.py src/snn_data_lake/engine/brain.py \
    src/snn_data_lake/reasoning/blimp_bench.py tools/
git rev-parse HEAD
PY tools/blimp_baseline.py 25
```

**Expect**: a clean diff (a dirty tree **aborts** the capture), the commit recorded in `baseline.txt`,
and the three-scorer table. The historical figures on the full 67 x 25 set are acquired-LM **51.0%**,
char-bigram **52.8%**, ROSE **no signal** (`docs/DEPTH_CEILINGS.md:118-127`) - if your numbers differ
materially, something changed and that is the finding.

**Why the abort matters**: `specs/003` records a research pass that measured an uncommitted rewrite and
labelled it the baseline. HEAD is moving in this repo right now because a second agent is committing, so
this check is live, not ceremonial.

### G8 M3 path note (2026-09-27)

The original baseline command named
`src/snn_data_lake/cortex/spiking_lm_serve.py`; its runnable path is now
`src/snn_data_lake/sensorimotor/spiking_lm_serve.py`.

### G8 M10 path note (2026-09-29)

The original baseline command named
`src/snn_data_lake/cortex/brain.py`; its runnable path is now
`src/snn_data_lake/engine/brain.py`.

---

## Scenario 3 - the organ is deterministic (FR-018, SC-012)

```
PY -m pytest tests/test_language_organ.py -k determinism -q
```

**Expect**: two organs built with the same seed and given the same exposure in the same order produce
identical output. **Failure means** a module-level seed, or a tie in the cap broken by a fresh random
draw - the second is the one that hides, because it only bites when two units are exactly equally driven.

---

## Scenario 4 - scoring does not teach the model its own test (FR-015, C2.1, C5.1)

```
PY -m pytest tests/test_language_organ.py -k readonly -q
```

**Expect**: every weight is byte-identical before and after a full scoring pass. **Failure means** the
benchmark taught the model, and every number downstream of it is void.

---

## Scenario 5 - the schedule is learned, not written (C4.2, plan Risk 5)

```
PY -m pytest tests/test_language_organ.py -k schedule_changes -q
```

**Expect**: the inter-area disinhibition schedule after exposure differs from the schedule at
construction. **Failure means** the grammar was written into the schedule and the organ is reproducing
it, which produces correct-looking output for the wrong reason. This is the subtlest trap in the
architecture and the accuracy number will not reveal it.

---

## Scenario 6 - no literal arbitrates behaviour (FR-007, C0.3, C3.2)

```
PY -m pytest tests/test_language_organ.py -k derived -q
```

**Expect**: `n`, `k`, `p` and `beta` each satisfy their asserted derivation; the projection loop
terminates on convergence; the guard bound is derived and reaching it increments a **reported** refusal
counter. **Failure means** either a literal at the centre of the organ (P1) or - worse - a silently
accepted iteration cap, which is the silent-degradation pattern: a stage fails, swallows it, writes a
placeholder, exits 0.

---

## Scenario 7 - the grounding is load-bearing (FR-004, SC-004, US3)

```
PY tools/lang_organ_ab.py --arms joint,shuffled,frozen
```

**Expect**: `joint` above the most-frequent-role baseline on held-out items; `shuffled` and `frozen` at
it. Both numbers printed for every arm.

- **`shuffled` matching `joint`** means the organ learned word statistics and the grounding is
  decorative. The grounded-acquisition claim is unfounded and must be withdrawn.
- **`frozen` scoring well** means the random synapse graph and the readout are carrying the result, not
  the learning. *Movement is not activation*, in its most flattering form.

Either outcome is a real finding and is reported. Neither is a reason to adjust and re-run.

---

## Scenario 8 - joint versus staged (FR-006, SC-005)

```
PY tools/lang_organ_ab.py --arms joint,staged
```

**Expect**: the contrast printed with direction and magnitude, whichever way it falls. If `staged`
matches or beats `joint`, the summary says the joint claim (arXiv 2406.11977) **is not supported here**.
That is a publishable result about this organ, not a failure to hide.

---

## Scenario 9 - the verdict (FR-016, SC-006, US4)

```
PY tools/coverage_score.py record --kind ab --name lang_organ_ab -- tools/lang_organ_ab.py
PY tools/coverage_score.py --rerun lang_organ_ab
```

**Expect**: both arms, both scorings, four admission controls, all in one output; a summary line whose
first token is `ADOPT`, `NULL` or `NOT-P2-ELIGIBLE`; and a `--rerun` that reproduces it.

**Read the summary, not just the exit code.** Exit 0 means a verdict was legitimately produced, not that
it was ADOPT. `NULL` exits 0 deliberately: a null is not an error, and a nonzero exit invites someone to
"fix" it.

**`--rerun` failing while `record` succeeded** is the specific forgery the ledger exists to catch - most
often a `python -c` one-liner, because `rerun` splits the recorded command on whitespace.

---

## Scenario 10 - the honest costs (FR-021, FR-022, SC-009, SC-010)

Read the two blocks `lang_organ_ab.py` prints at the end:

- **Density**: `H`, the derived `M`, `rho_floor = H_b^{-1}(log2 M / H)`, `rho_measured`, and the plain
  statement. Expect the statement to say that **compute is dense regardless of activity sparsity** and
  that the low floor here follows from the organ's small `M` - from it doing less than a language model -
  so scaling toward open-domain prediction raises the floor and removes the sparsity.
  `rho_measured < rho_floor` is an **anomaly**, not a win: the organ would be carrying less information
  than it needs.
- **Latency**: `Brain.respond` before and after, same units, with the delta. **Expect a regression** - a
  dense matrix product with a top-k, iterated to convergence, per word, replaces a 34x88 matrix-vector
  product. The number is the point; the regression is not a bug to hide.

---

## Scenario 11 - after ADOPT, exactly one generator survives (FR-010, SC-007, US5)

```
PY tools/citation_guard.py
grep -rn "language_acquisition" src/ tools/ tests/ run_selfchecks.py
PY run_selfchecks.py
```

**Expect**: no reachable reference to the deleted module; exactly one reply generator reachable from
`Brain.respond`; the self-check registry updated in the correct tier with the module-count change
**stated**; all four gates green with nothing loosened.

**Counting survivors is the test.** If two implementations of the reply-generation job are still
reachable - behind a flag, a mode switch, or an `if` - it was not a merge, it was additive with extra
steps, and FR-010 is unmet.

---

## Scenario 12 - after NULL, nothing changed (FR-005, SC-008)

```
git status --porcelain src/snn_data_lake/cortex/
```

**Expect**: `cortex/language_acquisition.py` present and unmodified; the organ not wired into
`respond()`; `verdict.md` recording the null with its numbers; `docs/coverage/Language.md` unchanged or
explicitly noted as unchanged.

**The feature is COMPLETE, not blocked.** A null result is a valid, successful end state. It does not
license a re-chosen pair set, a substituted metric, a parameter sweep, or a retry - and the harness has
no parameter that could deliver one anyway (B8).

---

## Scenario 13 - the dormant CORE module is adjudicated either way (FR-012, SC-011)

```
grep -rn "spiking_lm_serve\|snn_lm" src/ tools/ tests/ run_selfchecks.py
```

**Expect**: a recorded disposition - wired, retired, or **exempt with a stated reason** - and a consumer
count reported as `N of M`. This runs regardless of the organ's verdict, because the module is dormant
either way: it is registered CORE at `run_selfchecks.py:83`, its self-check passes, and it serves a
hand-built four-token demo that generates `'abbbbbb'` because no trained `snn_lm/1` weight file exists
anywhere in the repo.

**Before retiring, note**: it is the only torch-free `snn_lm/1` reader, so deleting it orphans
`src/snn_data_lake/training/spiking_lm.py`'s export path. Exempt-with-reason is a permitted P3
disposition. Leaving it undecided is not.
