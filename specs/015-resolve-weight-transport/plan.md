# Implementation Plan: Resolve the e-prop Weight Transport at `mlp_q`

**Branch**: `015-resolve-weight-transport` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

**Input**: 18 FRs, 9 success criteria, 4 user stories, 5 non-goals, 4 risks. Tier 0 item #2 — the
last open one.

## Summary

`mlp_q` self-declares weight transport and was left unresolved because its refutation of the
alternative rests on a criterion that **cannot discriminate**: the cosine reduces to
`cos(W2[a], B[a])`, near zero by construction for a fixed random draw, so it asks whether the
feedback is parallel to the forward weights — the weight symmetry feedback alignment exists to deny.

The plan is therefore: **repair the instrument, then measure, then decide** — and the repair lands
before the measurement it will judge, so it cannot be retro-fitted to a wanted answer.

**The central Phase 0 finding**: the owner's proposed SFDFA does **not** transfer to this module —
`MlpQ` is a rate MLP with no spikes, and SFDFA's contribution is precisely the spiking part. But its
*essential* property (feedback that is ESTIMATED and CONVERGES toward the forward weights, rather
than transported or frozen) has a rate-domain mechanism — **Kolen-Pollack**, which `MlpQ.update`
already contains every term for. See [research.md](research.md) D-1.

## Technical Context

**Language/Version**: Python 3.14.6 free-threading, gate env `.venv-lake`. CI has no GPU.

**Primary Dependencies**: **none, and none may be added.** `mlp_q` is stdlib with hand-written
backprop — no torch, no autograd, no numpy in the update path.

**Storage**: none. This lane writes no lake record.

**Testing**: `pytest` + `run_selfchecks.py`. The module's `main()` self-check re-derives its numbers
on every gate run; what this lane adds there is deliberately minimal (research.md D-4).

**Constraints**: `brainstem/` untouched — this lane carries **no** safety-envelope approval.
One change per commit, full gate green at every commit, **no docs-only carve-out**.

**Scope**: 3 P1 stories + 1 P2; 5–6 commits, one of which is conditional on a measurement.

## Constitution Check

*GATE: must pass before Phase 0. Re-checked after Phase 1 — see bottom.*

| principle | verdict | basis |
|---|---|---|
| **P0** self-improvement | **ADVANCES** | Removes a false "measured and refuted" from the record. P0's honesty guard is about not lying to ourselves; a refutation resting on a criterion that cannot fail is exactly that. |
| **P1** no-hardcode | **BINDS, one open item** | The Kolen-Pollack decay term would be a new bare constant. research.md D-1 leaves it explicitly open with two routes (derive from `lr`, or run decay-free from a small init) and requires the choice be MEASURED. The feedback scale itself is inherited from the module's existing `uniform(-1,1)/sqrt(hidden)`. |
| **P2** non-additive | **THE BAR OF THE ADOPTION COMMIT** | If a rule is adopted, exactly ONE credit-assignment path may remain reachable. FR-008. The measurement lives in a scratch probe precisely so a second live head is never created (D-4) — the same discipline the module already applied to the FA twin. |
| **P3** no dormancy | **N/A** | No new module; no exemption touched. |
| **P4 / P4b** one brain, one lake | **PASS** | No brain construction, no durable store. |
| **P4c** every build is the brain | **PASS** | The scratch probe lives in `tests/probe_015/` and is an instrument, not a build. **CORRECTED during implementation 2026-08-14**: it was first placed under `specs/`, and `p4c_ratchet` reddened — a `.py` there opens a NEW ZONE (`specs: 1 > baseline 0`) and the ratchet refuses a rise by design. `tests/**` is already excluded as "the instrument that measures a build, not a build", which is exactly what this is. It must not be left in `src/`. |
| **P5** GPU | **N/A** | Stdlib arithmetic on width-32 vectors. |
| **P6** faculty-to-faculty | **DO NO HARM** | No cross-region import added. The p6 ratchets must be unchanged — measured, not assumed. |
| **P7** dependencies | **PASS, and constrained** | None added. FR-014. |
| **P8** invent the method | **THE HEAVIEST CLAUSE HERE** | A repaired criterion is a new criterion: it owes a demonstrated reachable RED, a control that is not the method itself (the permutation null, D-2), per-item attribution across 8 draws, unprompted disclosure of the residual degeneracy, and **deletion on a null** (FR-007, pre-committed in D-3). |

### Cross-cutting invariants

| invariant | verdict |
|---|---|
| **e-prop, not backprop** | **THE SUBJECT.** This lane exists to resolve one of its four sites. It may not relax the invariant; if the measurement forces that question it becomes an owner decision (non-goal 4). |
| safety envelope | **UNTOUCHED.** No approval carried. A task touching `brainstem/` is an error in this plan. |
| verify-first | **MET** — probe → act → prove → commit, and C1–C3 land before any behaviour change. |
| runtime evidence > static reading | **MET** — D-1's infeasibility finding came from reading `MlpQ.update`'s actual arithmetic, and the adoption decision consumes a measured trajectory. |
| one change per commit, full gate | **MET** — D-5, and the gate batch runs on docs commits too. |
| **adding a policy means building its gate and counting its output FIRST** | **APPLIES TO THE CRITERION.** C2 lands the repaired criterion and its red BEFORE C3 measures anything with it. |

**Verdict: PASS.** One open P1 sub-decision (the decay term), deliberately deferred to measurement
rather than assumed.

## Project Structure

```text
specs/015-resolve-weight-transport/
├── plan.md          # this file
├── spec.md
├── research.md      # Phase 0 — the six decisions and their evidence
├── data-model.md    # Phase 1 — the criterion, the feedback pathway, the trajectory record
├── contracts/
│   └── criterion.md # what the criterion compares, and what it must separate
├── quickstart.md    # Phase 1 — the runnable procedure, with controls
│   (probe lives in tests/probe_015/ -- see the P4c row: a .py under specs/ reddens p4c)
└── tasks.md         # Phase 2 (/speckit-tasks)

src/snn_data_lake/basal_ganglia/
└── mlp_q.py         # the ONLY source file this lane may change
```

**Structure Decision**: the measurement is a scratch probe in `tests/probe_015/`, not a module addition and NOT under `specs/`.
`main()` re-derives on every gate run, so a 30-pass × 8-draw × 2-variant × per-epoch sweep landed
there is a cost every future commit pays to answer a one-time question (research.md D-4).

## Complexity Tracking

| Violation | Why needed | Simpler alternative rejected because |
|---|---|---|
| Adopting a mechanism the owner did not name (Kolen-Pollack, not SFDFA) | SFDFA is defined over spikes and `MlpQ` has none; its contribution is the spiking part. KP is the rate mechanism with the property that made SFDFA attractive. | **Implement SFDFA as written** — impossible here without inventing a spiking `MlpQ`, which is a different and much larger feature. **Report infeasibility and stop** — permitted by the spec, but leaves the degenerate criterion in place and Tier 0 open, when a testable candidate exists. |
| A new feedback matrix in a module whose point is having one head | It is not a second head: it replaces the transport term inside the existing head's single update. Survivor count stays 1 (FR-008). | **Retire `MlpQ` for the linear `LatentQ`** — the other named resolution, and it costs the non-linear capability outright (XOR 0.993 → 0.467). P2's honesty guard requires porting that first, which is a larger lane than this one. |

## Post-Design Constitution Re-Check

- **P8** — [contracts/criterion.md](contracts/criterion.md) fixes what the criterion compares and what
  it must separate BEFORE any candidate runs; [quickstart.md](quickstart.md) schedules the reachable
  red and the permutation null. **PASS.**
- **P2** — the survivor count is an explicit quickstart step, not an intention. **PASS.**
- **P1** — the decay decision is carried as an open item with two routes and a measurement, not
  resolved silently. **PASS, with the item named.**
- **R-001 (criterion tuned to the answer)** — mitigated structurally: the known-bad reference is a
  permutation null fixed in Phase 1, independent of which rule wins, and D-3's prediction is written
  before any run. **PASS.**
- **Safety envelope** — quickstart's final step diffs `brainstem/` and fails on any entry. **PASS.**

**No new violations introduced by the design.** The one honest residue, disclosed rather than
smoothed: the repaired criterion remains a monotone function of `cos(W2, B)` under this module's
arithmetic (research.md D-2). It becomes informative because `B` can now MOVE, not because the
algebra changed — and that is stated in the contract rather than left for a reviewer to find.
