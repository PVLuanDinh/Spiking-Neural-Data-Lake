# Implementation Plan: Grow the Live Populations (Autonomous LIF Growth)

**Branch**: `003-grow-the-live-populations` | **Date**: 2026-08-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-grow-the-live-populations/spec.md`

**Baseline commit for every figure in this plan**: `c6ca8c8d`, working tree verified clean for
`cortex/time_cells.py` and `cortex/brain.py` before measurement.

## Summary

Make one live population — `TimeCellClock`, hardcoded at `n_cells=64` (`brain.py:744`) — grow its own
LIF cells, and establish that an added cell can earn its place before letting the brain grow
continuously.

Phase 0 changed the approach in three ways the spec did not anticipate:

1. **The readout must be replaced before anything can be grown.** Today's `decode()` collapses to the
   ramp `v`, so growth is not merely unmeasured but provably worthless — RMSE is *identical* at widths
   64, 128, 256 and 512. The replacement is the geometric-mid population readout, chosen because it is
   the only candidate that both reads the population (fails a degenerate-ladder control) and falls with
   width. It costs **+45% RMSE at the live width**, closing to +4% by n=512. That cost is the price of
   measurability and is reversible via the off-switch.
2. **Growth must be interior insertion at a reset boundary, not append-at-top.** Append-at-top is
   measurably inert below saturation under every readout tested (ties 20 of 20), because a cell above
   the ramp never fires.
3. **Execution order differs from the spec's priority order**, because slice-1 evidence gathered before
   the readout fix is an FR-006 clamp-widening artefact, and the readout fix changes its magnitude.

Two latent defects are fixed on their own merits along the way: `decode()` never divides by `drive`
while `learn_interval` does, and `learn_interval` can already break the ascending-threshold invariant
the whole class depends on.

## Technical Context

**Language/Version**: Python 3.11 / 3.12 / 3.13 (CI matrix, `.github/workflows/ci.yml:27-29`)

**Primary Dependencies**: stdlib only for the affected modules. `cortex/time_cells.py` imports `math`
and `random` and nothing else; this feature adds no import. `numpy`/`torch` are unconditional project
dependencies but nothing in this plan needs them — which keeps the work green on the `NORD_TORCH=0`
path.

**Storage**: `home.save_state` / `load_state` atomic JSON (`time_cells.json`). Deliberately **not** the
one lake — see research R4.

**Testing**: `python run_selfchecks.py`; `pytest` (testpaths `tests`, `--cov-fail-under=70`);
`python -m ruff check .`; `ruff format --check src tests tools`. All four are CI gates.

**Target Platform**: ubuntu-latest and windows-latest, CPU only. CI has no GPU.

**Project Type**: single Python package (`src/snn_data_lake/`), 10 anatomical regions, plus `tools/`
probes and a tracked coverage-artifact ledger.

**Performance Goals**: no regression to live tick cost. The affected path runs once per tick;
geometric-mid adds one `sqrt` and one multiply per `decode()` call. Persistence must not add lake-scan
cost — `load_state` is 0.110 ms against `load_growth`'s measured 0.400 s / 187.5 MB resident.

**Constraints**: no new tuned constant (P1); determinism under a fixed seed
(`tests/test_time_cells.py:46`); no assertion may be loosened; live populations are 24–240 wide so the
tick stays off the GPU (P5); the feature must have an off-switch under which live behaviour is
unchanged.

**Scale/Scope**: one population, 64 → 65 cells initially. The four other hardcoded widths are
explicitly out of scope until the pattern is proven on one.

## Constitution Check

*GATE: must pass before Phase 0 research. Re-checked after Phase 1 design — the result below is the
post-design re-evaluation.*

| Principle | Assessment | Verdict |
|---|---|---|
| **P0** autonomous self-improvement | Neural growth is a named P0 avenue and this closes it for one live population. The honesty guards are load-bearing rather than decorative here: growth that changes no outcome is dormant, and Phase 0 measured exactly that condition holding today. Ordering is built so nothing is credited before it is measured — *"build the channel, then fill it, then measure it — and do not credit it until the third step"* (`docs/POLICIES.md:60-71`). | **PASS** |
| **P1** brain-native / no-hardcode | Removes `n_cells=64` as a fixed behaviour-arbitrating value. Every introduced quantity is derived from existing state: the inserted threshold is `sqrt(t[j]*t[j+1])`, the extend-at-top threshold is `n_cells+1`, the split ratio is ported from `ng_snn.py:178`, the order-preservation bounds are the neighbours themselves. No new numeric literal arbitrates behaviour. | **PASS** |
| **P2** adopt-better or merge, retire the inferior | Exactly one earn-your-place standard survives: the discipline stated in `ng_snn.py:146-200` is ported, not duplicated. The old `decode()` branch is **replaced**, not kept behind a flag — the off-switch gates *growth*, not two coexisting readouts. | **PASS** |
| **P3** no dormancy | The risk is real and named: a `grow()` callable only under a disabled flag is dormant by the letter of the gate. Mitigation is that the readout change and the `learn_interval` order fix are unconditionally live, and the growth path is exercised by its own self-check and tests. The gate's treatment of a flag-gated method must be **confirmed during implementation, not assumed**. | **PASS, with a check owed** |
| **P4** one brain / no replication | No new Brain, no new store, no re-implemented composition. Construction still goes through `make_brain()`. | **PASS** |
| **P5** GPU where it measurably helps | Untouched. The population is 64–65 wide, three orders below the measured CUDA crossover; the tick stays on CPU. | **PASS** |
| **P6** every build reaches every other and is used | The readout is consumed by three existing readers (`brain.py:1559`, `:2797`, `:3197`), so the changed quantity genuinely changes a consumer's output. **Growth events themselves have no second-faculty consumer yet** — an honest gap, recorded rather than papered over, and the reason this feature does not claim P6 satisfaction for the growth signal. | **PARTIAL — declared** |
| **SI-1** count a gate's output before leaning on it | The degenerate-ladder control was implemented and run *before* any readout was adopted, and it disqualified the leading candidate. | **PASS** |
| **SI-2** never silence a check to make it pass | No assertion is loosened. The plan *adds* a control the leading candidate failed rather than relaxing one it failed. | **PASS** |
| **SI-3** the negative verdict is the dangerous one | The instrument exits 2 `NOT-ELIGIBLE` when its readout cannot see the population, rather than reporting a confident null. Restore refuses and records rather than half-restoring. | **PASS** |
| **SI-4** absence of a warning is not absence | Tie counts printed unconditionally; 10 of 30 paired differences measured as exactly 0.0. | **PASS** |
| **SI-5** verify the verifier | The governing rule for this feature. The degenerate-ladder control exists solely to satisfy it, and it changed the outcome. | **PASS** |

**No violations requiring justification.** Complexity Tracking is therefore empty.

## Project Structure

### Documentation (this feature)

```text
specs/003-grow-the-live-populations/
├── spec.md               # feature specification
├── plan.md               # this file
├── research.md           # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/            # Phase 1 output
│   └── time_cells.md     # the population's public contract
├── checklists/
│   └── requirements.md
└── tasks.md              # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── cortex/
│   ├── time_cells.py     # decode() readout replacement; learn_interval order-preservation;
│   │                     #   grow()/refuse; growth + refusal counters; state serialization
│   └── brain.py          # restore hook in its OWN try (never the :736-752 handler);
│                         #   state() exports width + growth + refusal counters
└── substrate/
    └── home.py           # unchanged — save_state/load_state already exist

tools/
├── timing_heldout.py     # NEW, tracked: readout-agnostic paired-seed held-out A/B,
│                         #   degenerate-ladder control, null-control arm
└── coverage_score.py     # unchanged — used to record the artifact

tests/
└── test_time_cells.py    # extended: non-destructive growth, order preservation,
                          #   degenerate-ladder control, restore round-trip

docs/coverage/artifacts/  # the recorded kind:heldout artifact, if a verdict is reached
```

**Structure Decision**: single project, existing layout. The feature touches two source modules
(`cortex/time_cells.py`, `cortex/brain.py`), adds one tracked tool, and extends one test module. No new
package, no new directory, no new dependency. The harness goes in `tools/` rather than `scratchpad/`
because `scratchpad/` is gitignored and `tools/citation_guard.py:320` forbids a tracked file citing an
untracked `.py` as a reproduction path.

> **P7 VOID (annotated 2026-08-04).** The sentence above certifies "no new dependency" as a constitution check. **That is not a criterion under P7** (`docs/POLICIES.md`), which states that *"it would be another dependency" is NOT a valid reason to reject anything* and requires a rejection to cite a measurement on the real workload or a structural argument. No library was evaluated here, so the check scored nothing. Left in place rather than deleted because this file records a decision that was made; annotated because P7 names these certifications as live residue of the zero-dependency rule abolished 2026-07-20. `specs/010/plan.md` is the first plan to decline the certification on purpose.

## Execution Order

Deliberately not the spec's priority order — see research.md, "Cross-cutting".

| # | Work | Why here | Live behaviour change |
|---|---|---|---|
| 1 | `learn_interval` order-preservation (refuse, not clamp) | Pure bug fix. FR-012 is unsatisfiable without it; 9 of 12 seeds invert today. | Yes — drops order-breaking updates |
| 2 | `decode()`: geometric-mid readout, divide by `drive`, drop the saturation clamp | Defect fix owed regardless. Must precede growth or all growth evidence is a clamp artefact. | Yes — +45% RMSE at n=64; fixes drive-scaling and the 292.9 RMSE saturation catastrophe |
| 3 | `grow()` / refusal, interior insertion at a reset boundary | Now demonstrable, because step 2 made the readout population-sensitive. | No — off by default |
| 4 | `tools/timing_heldout.py` + degenerate-ladder control + null-control arm | The instrument FR-002 requires. Self-refuses if its readout cannot see the population. | No |
| 5 | Persistence via `save_state`; observability via `state()` | Required before growth may be continuous (FR-007, FR-011). | No — off by default |
| 6 | Autonomous growth behind the off-switch | Gated on step 4 returning a benefit. | Only when enabled |

Steps 1 and 2 are independently shippable and worth shipping whether or not anything is ever grown.

## Risks

- **The +45% accuracy cost at the live width is a real regression** to a quantity the timed Go window
  consumes. It shrinks as the population grows, which is the point, but it is paid up front for a
  benefit that arrives later and may not arrive at all.
- **P3 dormancy on a flag-gated `grow()`** must be confirmed against the actual gate, not assumed.
- **`state()` additions may break digests.** `ab_method.py`'s behavioural digest and any golden
  comparison over `state()` need enumerating before keys are added.
- **The achievable ledger movement is small.** `record` writes no `metric:`/`baseline:` keys, so F10 caps
  a freshly recorded `kind: heldout` artifact's band at 40 — one point above the row's current 35. The
  honest deliverable may be a correct instrument, a real number, and a barely-moved row.
- **The attribution question is open.** Geometric-mid is admissible because it reads only the population,
  not because it is more accurate. Whether a point-estimate improvement can be attributed to how well the
  population *tiles* the interval — rather than to how recently any cell fired — remains unresolved and
  must not be claimed.

## Complexity Tracking

No Constitution Check violations. Section intentionally empty.
