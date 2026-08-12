# Phase 0 Research: whole-brain neurogenesis

**Feature**: specs/007-whole-brain-neurogenesis | **Date**: 2026-08-03

Every number below was produced by a command that was run. Four probes, each adjudicated against an
adversarial re-run; where a refuter corrected a finding, the corrected result is what is recorded.
Claims that could not be backed by a run are marked UNVERIFIED.

Environment for every quoted command: repo root, `.venv-lake\Scripts\python.exe`, `PYTHONPATH=src`.

---

## The finding that reframes the feature

**Function-preserving growth has a consumption signal of exactly ZERO at growth time, by
construction.** This is not a defect in any mechanism. It is what "function-preserving" means, and
the tree says so in its own docstring:

```python
# src/snn_data_lake/substrate/growable.py
def grow_rows(mat, new_rows, ncols=None, fill=0.0):
    # rows are all `fill` (zero), so they emit 0 until trained and the EXISTING outputs are unchanged.
```

New units emit 0. Existing outputs are untouched. Measured on the coordinator's own ripple path:

```
consumer output BEFORE        : [-0.59, 1.31]
consumer output AFTER ripple  : [-0.59, 1.31]
CONSUMER OUTPUT DELTA         : 0.0
new producer channels emit    : [0.0, 0.0, 0.0]
new consumer weights on them  : [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

**Consequence: "grow, then read the width" can never satisfy SC-001, for any mechanism.** The
consumption this feature requires cannot come from the act of widening. It can only come from grown
units subsequently LEARNING nonzero values, and those values reaching a consumer that branches on
them. Any plan whose consumption story is "the consumer sees a bigger population" is dead on
arrival, and the measured 88-of-93-identical baseline is the arithmetic consequence, not a bug to be
found and fixed.

This single fact removes an entire family of designs, which is why it leads.

---

## Decision 1: Which growth mechanism survives, and which are retired (P2)

**Decision**: **No survivor is selected in Phase 0, and nothing is retired yet.** This is an honest
null, not a deferral for its own sake: the choice of growth primitive provably cannot move the
consumption number (see the finding above), so consolidating mechanisms cannot satisfy this
feature's acceptance criterion. Selection happens after one decision gate, below.

Approved immediately, needing no survivor decision -- these are dead code, not candidates:

1. `reasoning/readout_ensemble.py:63` `grow` -- body is a bare `return`.
2. `reasoning/developmental_loop_brain.py:101` `grow` -- body is `return  # linear: no hidden layer to grow`.
3. Four duplicated `(1,0)` readout splits (`growable_readout.py:176`, `laminar_readout.py:205`
   and `:483`, `laminar_local_readout.py:149`) which the source documents as "REUSED verbatim" from
   one another. Collapse to one. [RESOLVED 2026-08-12, T022: collapsed 4 -> 2 -- the one torch
   helper `_extend_zero_fanout` in hippocampus/growable_readout.py plus the numpy mirror in
   laminar_local_readout.py; laminar_readout.py's two sites died with the file at the laminar merge.]

**The decision gate**: register the two seams that already grow (`growing_learner.py:112`,
`multimodal_encoder.py:198`), wire ONE real producer/consumer pair across a module boundary, and A/B
whether the consumer's output moves for a reason other than width. `docs/AUDIT_BACKLOG.md` M25
already prescribes this and states the verdict rule -- `0 -> retire GrowthCoordinator`. Run it and
honour the answer either way.

**Rationale**: the survey nominated `substrate/growable.py` on a claimed "signal-to-noise infinite".
That ratio is 0 over 0 -- UNDEFINED. The `1.8` signal appeared only after the probe hand-wrote
`B["W"][0][2] = 1.0` and hand-populated a producer row. The probe measured its own hand.

`growable` is also not live in any sense that matters:

```
grep -rl "is_safety_seam\|GrowthCoordinator\|register_seam" --include=*.py src/ tests/ tools/
  -> src/snn_data_lake/substrate/growable.py          (1 of 1 files; zero in tests/)

grep -rn "\.grow_dim(" --include=*.py src/ tests/
  -> growing_learner.py:101   (its own definition)
  -> growing_learner.py:198   (a __main__ demo)
```

A registry that is never registered. Its only consumer is a lambda inside its own `main()`, in the
same module as the producer, which does not change its output for any reason -- it only resizes.
`docs/AUDIT_BACKLOG.md:103` already files it under the "registry never registered" defect pattern.
Promoting a module the repo has queued for deletion would be the wrong direction.

**Alternatives considered**:

| Mechanism | Measured reason it is not (yet) the survivor |
|---|---|
| `substrate/growable.py` + `GrowthCoordinator` | Bit-exact preserving (`max\|d\|=0` on 8 of 8 probes) -- but that exactness IS zero consumption at growth time. Ripple consumer delta 0.0. Zero live participation. No earn gate. Queued for retirement by M25. |
| `cortex/time_cells.py` `grow_if_earned` | The only grower with a working halting condition and the only one measured end to end. Perturbs exactly 2 downstream fields, not a global noise floor. Not general: its floor is specific to a threshold ladder. |
| `hippocampus/growing_learner.py` | The de facto shared mechanism -- 4 regions grow only by embedding it. Recruitment is inline in `observe`, not in a `grow*` method, so name-based scans undercount it. |
| `substrate/similarity_sized_growth.py` | Never observed growing inside a running brain; standalone only. |

**What would falsify this**: if the M25 gate shows a registered cross-boundary consumer whose output
moves for a reason other than magnitude, `growable` becomes a real candidate and this null is wrong.

---

## Decision 2: What consumes grown capacity, and what P1 targets

**Decision**: P1 targets the time-cell population -- the only grower that earns -- but **NOT** the
existing `elig["timing"]` route. P1 must build a consumer **outside cortex** that **branches** on
the clock's float estimate, rather than summing it into a common-mode scalar.

**The genuinely new fact: the channel is not dead, it is SUB-RESOLUTION.** Disbelieving the null and
instrumenting raw internals with a matched A/A control:

```
A/A control  arm._ext_drive  0 of 300   (max |delta| 0.000000000)
             arm._conflict   0 of 300   (max |delta| 0.000000000)
A/B test     arm._ext_drive  differs on 211 of 300, max |delta| 0.000099808
             arm._conflict   differs on  23 of 300, max |delta| 0.000100000
             acc_conflict (state(), 3dp)                     0 of 300
             arm_stopped / dominant_drive / metacog_explore   0 of 300 each
```

Growth reaches the cortex-to-brainstem hand-off on 211 of 300 ticks. It arrives about three orders
of magnitude too small to survive rounding, and is then absorbed into a summation. Every one of the
23 raw conflict differences is exactly `1.0e-04` -- one unit in the last place of `_conflict`'s own
`round(..., 4)`.

**The prerequisite defect, and it is the same shape as the P6 blocker.** `cortex/brain.py:1603`:

```python
self._learned_latency = max(1, int(round(el)))
```

The clock produces a refined float; the consumer casts it to an int. In the hermetic run
`_learned_latency` differs on **0 of 300** ticks -- the quantiser destroys **all** of the
refinement, not most of it. Growth's entire product is discarded at the moment of consumption.

This is now the third instance of one defect class in this codebase: `max(rb.values())` discards a
winning faculty's identity, `int(round(el))` discards timing refinement, and function-preserving
`fill=0.0` emits nothing until trained. In each case a producer emits information a consumer
coarsens away. That recurrence is the real subject of this feature.

**Three independent disqualifiers for the existing route** (any one is fatal):

1. **Same region.** `REGIONS["cortex"]` contains `time_cells`, `brain` AND `executive_control`. The
   only code branching on the clock is `brain.py:1601-1616`. `autonomous_loop.py:696` is
   `return [td[i] + ext[i] for i in range(len(td))]` -- a blind add of a pre-computed vector. The
   brainstem never sees the clock. That is transport, not consumption.
2. **No identity.** `elig["timing"]` enters via `undirected`, which `brain.py:1639-1643`'s own
   comment says is added to both non-HOLD actions EQUALLY. FR-005 cannot be satisfied by a quantity
   applied equally to every option.
3. **The consumer is provably permutation-invariant.** `executive_control.py:80-98` `conflict()` is
   documented "Scale- and permutation-invariant" and its self-check at `:263-266` ASSERTS that
   shuffling the same drive values cannot change the output.

**What would falsify this**: if a consumer outside cortex can be shown to branch on the float
estimate and its output moves in an A/B with a matched A/A control at 0, P1 is on the right target.
If no such consumer can be built without inventing an artificial branch, this feature's P1 is not
achievable as specified and the spec must be amended rather than the test weakened.

---

## Decision 3: What demand signal earns growth without a magic constant (P1)

**Decision**: keep `resolvable()` (`time_cells.py:423-427`, backed by `first_passage_sd()` at
`:413`) as the halting condition -- it is the only measured self-extinguishing, genuinely
constant-free quantity in the tree. **Honest null on the entire benefit-test family.** Ship one
targeted fix: `widest_log_gap()` must select the widest log gap **among resolvable gaps**.

**The benefit gate contributes nothing. Measured:**

```
ARM A  baseline, both arms live       n_cells   68  grown   4  refused 196
ARM B  resolvable() -> always True    n_cells  264  grown 200  refused   0
ARM C  would_earn() -> always -1.0    n_cells   68  grown   4  refused 196
```

ARM C is byte-identical to ARM A. The held-out benefit arm contributes **0 refusals**. Every refusal
in the system comes from the resolution floor.

This corrects the record on specs/003, which I committed earlier this session. Its "verdict earns"
result is real -- the gate does discriminate, 4 grown against 196 refused -- but the discrimination
is entirely `resolvable()`. The held-out benefit test that the spec presented as the earn criterion
is inert.

**A benefit test cannot terminate growth. Measured, not argued** -- run as the sole gate:

```
growth     0  n_cells    65  realized delta -7.424491e-04
growth   999  n_cells  1064  realized delta -1.016490e-06
growth  1999  n_cells  2064  realized delta -3.979251e-07
PAYMENT went red 0 times in 2000 growths; final width 2064
resolvable gaps at width 2064: 0 of 2063
```

On a noise-free tiling metric the sequence improves monotonically forever, so "a monotone
non-improving sequence halts" is vacuous. Note what DID terminate that run: the supply floor again,
at 0 of 2063 resolvable.

A prospective/realized distinction does not rescue it: `would_earn()` builds the candidate on a
copy, and `grow()` makes the population become that copy, so the two are the same float --
`prospective == realized on 40 of 40 growths`.

**Rejected**: a top-p running-quantile demand arm. A top-p test requires `p`, and `p` is the magic
constant the policy forbids, merely relocated.

**The shippable fix.** `widest_log_gap()` argmaxes over gaps equal to within `8.882e-16` (63 of 63
gaps within 1e-15 of the max on the initial ladder), so it can select a gap the floor then refuses,
wasting the attempt. Selecting among resolvable gaps makes site selection and the halting condition
agree.

**What would falsify this**: a demand signal that is baseline-relative, terminates on a measured
run, and is not a relabel of the resolution floor. None was found; one may exist.

---

## Decision 4: What bounds growth, and how determinism survives 15 regions

**Decision -- bound**: a two-stage per-region gate in this ORDER: (1) resolution floor as the
halting condition, (2) marginal held-out benefit as a necessary-but-not-sufficient filter.
**Rescoped from "the general mechanism" to "the mechanism for 1 of 16 seams."** For the other 15 the
floor is UNBUILT and UNMEASURED. **Until a region's floor is built and A/B'd, that region does not
grow.** "This region has no derivable floor, so it does not grow" is a correct accepted outcome, not
a blocker -- and it is the clause that prevents this feature from shipping 15 dormant growers.

The floor is the bound and the benefit test is not:

```
both halves live                   -> halts at width 68  (grown 4)
resolvability half neutered        -> reaches width 3064, still growing at the attempt limit
benefit-only replicate             -> 40 of 40 attempts grew, 0 refusals, 64 -> 104
```

The last line lands exactly on the "600 ticks, 64 -> 104, 0 refusals" the module docstring records
from live running, which is a useful independent corroboration.

**Decision -- determinism**: four clauses.

- **R1. A growth decision must consume ZERO random numbers.** MEASURED: `rng.getstate()` is
  bit-identical across 50 `grow_if_earned` attempts (4 grown, 46 refused). This is what decouples
  "did it grow" from the noise stream -- the grown-vs-ungrown A/B shows the internal ramp `v`
  BIT-IDENTICAL after 60 ticks while the decoded output differs on 48 of 60.
- **R2. Growth may fire only at a state the region reaches from its OWN seeded dynamics** (the reset
  boundary) -- never on a wall clock, a tick count, or an external schedule.
- **R3. No growth mechanism may read mutable external state during construction or tick unless
  behind its own explicit opt-in flag; when enabled it must ADOPT WHOLESALE OR REFUSE**, never
  merge, truncate or pad. **This clause is violated in the live tree today**, which is the strongest
  argument for writing it down.
- **R4. Ripple order must be a deterministic total order.** `GrowthCoordinator` satisfies this today
  by accident (`_consumers` is insertion-ordered, `seams()` returns `sorted()`); a set iteration
  over region names would break it under hash randomization. Pin it explicitly. REASONED, not
  measured.

**A published measurement was withdrawn.** The survey's "21x saving purely from gate ORDER, 0.618 ms
vs 13.303 ms" is not an order measurement -- `grow_if_earned` returns at the floor without ever
calling `would_earn`, so at the two widths quoted it provably never reaches the benefit test. The
comparison is refusal-versus-evaluation. The timing rig was also unwarmed at 5 reps and
non-monotonic (cost falling as width grew 16x), which is a rig artifact.

**What would falsify this**: a region whose floor cannot be derived but which nonetheless benefits
measurably from growth would show the "no floor, no growth" rule is too strict.

---

## Open questions carried into the design

1. **Can a legitimate cross-region consumer of timing be built at all?** Decision 2 says the
   existing route cannot work and names what a replacement needs (outside cortex, branches on the
   float, order-sensitive). Whether such a consumer exists NATURALLY -- rather than being invented
   to pass the test -- is unresolved. Cheapest experiment: enumerate every consumer of any
   continuous cortex output that already branches (produces a boolean or categorical), and check
   whether any could take the timing estimate without an artificial branch being added.
2. **Do the other 15 seams have derivable resolution floors?** Unmeasured. Cheapest experiment: for
   one mid-sized region, attempt an empirical dispersion estimator and see whether it
   self-extinguishes.
3. **Is R3's live violation load-bearing?** The clause is violated today; whether fixing it changes
   any behaviour is unmeasured.

## What this research REFUTED

These are the designs now NOT being built, each killed by a number:

- **"Pick the best growth primitive and consolidate."** Cannot move the consumption number for any
  primitive, because function-preserving growth emits zero by construction.
- **"`growable`/`GrowthCoordinator` is the mature shared mechanism."** A registry never registered;
  16 occurrences in 1 file; its consumer is a lambda in its own `main()`. Its headline
  signal-to-noise was 0 over 0 with a hand-written weight supplying the signal.
- **"The `elig["timing"]` route just needs a bigger signal."** Three independent disqualifiers, one
  of which is an ASSERT in the consumer's own self-check that permutation cannot change its output.
- **"A held-out benefit test is the earn criterion."** 0 refusals in 200 attempts; byte-identical to
  disabling it; never red in 2000 growths reaching width 2064.
- **"Gate order gives a 21x saving."** Refusal-versus-evaluation, on an unwarmed non-monotonic rig.
- **"`acc_conflict` moves on 2 of 300 ticks, so a cross-region consumer already exists."** Does not
  reproduce hermetically: 0 of 300, with the A/A control at 0 of 102.

Two false positives were caught by matched A/A controls that would otherwise have been reported as
findings. Any A/B in this feature that ships without a matched A/A control is not evidence.
