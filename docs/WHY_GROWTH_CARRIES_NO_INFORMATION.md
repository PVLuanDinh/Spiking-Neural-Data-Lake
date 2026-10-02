# Why growth carries no information in this brain

**Measured 2026-08-03.** Written after two producers failed specs/007's acceptance criterion with
the same shape, and after three of my own explanations for it were falsified by measurement.

## The answer

**Growth is not broken. The interfaces are scalar-only.**

Growth produces *distinctions* -- 12 categories where there was 1, a finer timing ladder where there
was a coarse one. Every module boundary in this brain transmits a *magnitude* -- a sum, a max, a
rounded scalar, a permutation-invariant aggregate. **A distinction cannot survive a magnitude-only
channel.** That is the whole finding, and every measurement below is a consequence of it.

## What was falsified on the way, and why it matters

Three plausible explanations, each killed by its own measurement. Recording them because each would
have sent the build somewhere expensive and wrong.

**1. "The input stream is degenerate."** The hypothesis was that `_safe_ctx` pins the context, so
percepts barely vary and growth has nothing to encode. **False:**

```
percepts 448, dim 36    distinct percepts: 146 of 146 (on the 64-dim instance)
per-dim sd mean 0.180233192   max 0.386060840   dims with sd == 0: 0 of 64
```

Every percept is distinct with real per-dimension spread.

**2. "Cosine saturates on non-negative vectors, so nothing looks novel."** **False, and backwards:**

```
PAIRWISE cosine between unrelated percepts:
  mean 0.410289   p05 0.219259   p50 0.290452
  -> at vigilance 0.85, 87.2% of unrelated pairs look NOVEL and would recruit
```

The gate should recruit aggressively, not refuse.

**3. "The stream is temporally smooth, so a per-sample novelty test can never fire."** The sharpest
of the three, and the easiest to test -- shuffle the arrival order, change nothing else. **False:**

```
units, ARRIVAL order (live) : 12
units, SHUFFLED order       : [12, 12, 12, 12, 12]   ratio 1.0x
```

Bit-identical. Order is irrelevant.

## What is actually true: growth saturates because it is finished

```
final prototypes            : 12
percepts covered at >= 0.85 : 448 of 448
best-match cosine   min 0.9678   p05 0.9761   p50 0.9842
```

Twelve prototypes cover the entire percept space with margin. The learner recruits until coverage is
adequate and then stops. **That is correct behaviour, not a defect.** The producer is doing its job.

The same reading applies to the time cells: growth halts at width 68-69 because `resolvable()`
refuses further sites, which is a working self-extinguishing floor.

**So the information exists at the producer. It is destroyed at the boundary.**

## The six boundaries, all measured this session

Every one independently coarsens its input. They are in different regions, written at different
times, by different mechanisms -- which is what makes this architectural rather than incidental.

| # | Site | What it discards | Evidence |
|---|---|---|---|
| 1 | `engine/brain.py:1603` | `max(1, int(round(el)))` -- an int cast over a refined float | `_learned_latency` differs on 0 of 300 ticks: 100% of the refinement |
| 2 | `engine/brain.py:1623` | `max(rb.values())` -- keeps the magnitude, drops the winning faculty's KEY | permuting keys while preserving values leaves `Brain.state()` byte-identical at `d7a6d81fc43d83f9` |
| 3 | `engine/autonomous_loop.py:696` | `[td[i] + ext[i] ...]` -- blind add into a common-mode scalar | the timing term is added to both non-HOLD actions EQUALLY, per the code's own comment |
| 4 | `engine/autonomous_loop.py` `q[unit]` | unit IDENTITY | relabelling unit ids leaves 0 of 300 actions changed, on 10 of 10 seeds |
| 5 | `faculties/executive_control.py:80-98` `conflict()` | scale AND permutation | documented invariant, with an assert at `:263-266` enforcing it |
| 6 | `growth/growable.py` `fill=0.0` | new units emit 0 until trained | the docstring says so; ripple consumer delta measured `0.0` |

Sites 1 and 3 are the same defect twice on one path. `faculties/action_gating.py:114` does
`range(1, steps+1)` -- a seventh instance of the int-cast form, in a third region.

## Why this reframes specs/007 and P6 as one problem

specs/007 asks every region to be a producer AND a consumer. P6 asks every build to reach every
other and actually be used. Both are currently blocked, and **they are blocked by the same thing**:

- P6's blocker is `max(rb.values())` discarding the winning faculty's identity.
- specs/007's blocker is that no consumer can use a finer population.

These are one defect: **the channels carry scalars, and neither identity nor distinction is a
scalar.** Fixing them separately would mean building two incompatible widenings of the same
interfaces. Fixing the channel fixes both.

## What follows, and what does not

**Does NOT follow**: probe a third producer. Two producers failed identically, and the measurements
above show the failure is not in either of them.

**Does follow**:

1. **The consumer fixes are the work.** `brain.py:1603` and `brain.py:1623` are standalone P1
   defects on their own merits -- a hardcoded quantiser and a discarded key -- independent of any
   feature.
2. **Growth is not the frontier; interface width is.** Adding neurogenesis to 13 more regions
   before the channels can carry a distinction would produce 13 more producers whose output dies at
   the first boundary. specs/007's per-region floor gate already prevents this, which is the one
   part of that spec these measurements strengthen rather than undermine.
3. **The vigilance constants are a separate real defect.** 0 of 31 `vigilance =` occurrences derive
   from dispersion. That is a P1 violation whether or not anything consumes growth.

## Provenance

Measurements 1, 2, 3, the saturation table, and boundaries 1, 2 and 6 were run directly in the main
session and are reproducible from the commands in the git history. Boundaries 4 and 5 come from
subagent probes; 5 is corroborated by an assert in the source itself, 4 was reproduced by two
independent probes with an instrument check confirming the relabelling shim fired, but I did not
re-run it myself and it is marked accordingly.

The unifying claim -- that these six are one architectural defect rather than six coincidences -- is
an INFERENCE from the six measurements, not itself a measurement. The prediction it makes, and the
way to falsify it: a consumer that takes structure rather than a scalar should show growth moving a
downstream outcome. If one is built and growth still changes nothing, this explanation is wrong.

---

# CORRECTION 2026-08-03: the scalar-channel inference is WRONG. The brain starts past the knee.

The document above named its own falsifier: *"build a consumer that takes structure rather than a
scalar, and if growth still moves nothing, this explanation is wrong."* The owner proposed exactly
that -- multidimensional arrays and sparse tensors. Testing it falsified the inference, and produced
a better answer.

## The sparse channel is not better than the scalar one

Task: recover the true elapsed interval (10 classes, chance 0.100) from the clock's state, comparing
the SCALAR channel the brain consumes today (the decoded `(est, sd)` pair) against a SPARSE
population code (which cells have fired, dim = n_cells), nearest-neighbour, held-out halves:

```
width   scalar-channel acc   sparse-channel acc
   32        0.7950               0.7400
   64        0.9800               0.9450
  128        0.9900               0.9850
  256        0.9900               0.9850
  512        0.9800               0.9800
```

**The scalar decode is as good as, or better than, the sparse population code at every width.** Two
floats carry the interval information that 512 sparse dimensions carry. The channel is not the
bottleneck for this quantity.

The int cast was measured separately and is also not the bottleneck in principle -- it costs 0.0400
at width 64 and 0.0050 at width 128.

## What the same experiment revealed instead

Read the first column again. **Growth DOES carry information**: accuracy rises 0.7950 -> 0.9900 as
width goes 32 -> 128. The earlier finding that "growth does not improve the clock" was measured on
`sd`, and **`sd` was the wrong observable.** On a task the clock actually performs, growth helps a
lot.

And then it stops helping, because the curve saturates by 128 and is already at 0.9800 by 64.

**`engine/brain.py:757` constructs `TimeCellClock(n_cells=64, seed=self._seed)`.**

The brain is initialised at 64, where the task is already 98% solved. Growth from 64 to 69 -- the
measured live behaviour -- moves along the flat top of the curve. It carries no information *because
there is none left to carry at that width*, not because a channel destroys it.

That also makes `n_cells=64` a P1 violation of a particularly consequential kind: a hardcoded
constant that determines whether an entire self-improvement avenue can do anything at all.

## The corrected answer

**Growth carries no information because the populations are constructed above the knee of their own
utility curve.** Not because interfaces are scalar. The scalar-channel story was an inference from
six coarsening boundaries, and it was wrong -- those boundaries are real, but they are not what is
suppressing growth.

Predicted, falsifiable: initialise the clock at 32 instead of 64 and growth from 32 to 64 should
move interval discrimination by about +0.185, which is 4x the entire loss attributable to the int
cast. If that does not reproduce in the live brain, this explanation is wrong too.

## Where structure DOES still help, honestly measured

The owner's proposal is not dead -- it is aimed at a different quantity. Structure preserves
IDENTITY, which no scalar can carry. Measured over 140 live workspace broadcasts:

```
KEY   (identity)  distinct 11   entropy 1.5555 bits
VALUE (magnitude) distinct 45   entropy 3.1807 bits
PAIR  (both)      distinct 58   entropy 3.5007 bits
-> information discarded by dropping the key: 0.3201 bits per broadcast
```

Real, and modest. A vector- or dict-valued channel would carry it; `max(rb.values())` cannot. So the
recommendation splits cleanly:

- **Precision** (the timing case): a scalar carries it fine. Structure buys nothing. Fix the WIDTH,
  not the channel.
- **Identity** (the P6 case): a scalar cannot carry it at all. Structure is the only fix, worth
  0.3201 bits per broadcast.

## What this costs, and the honest accounting

Three explanations were falsified before this one, and now a fourth -- my own -- has been falsified
by the owner's suggestion. The pattern in all four: I measured a quantity that was easy to reach
(`sd`, cosine saturation, arrival order, channel shape) rather than the quantity the mechanism is
FOR (can it tell intervals apart?). Every wrong answer came from an observable chosen for
convenience.

The doc above is kept unedited rather than rewritten, because the sequence of wrong answers is more
useful than a clean final one -- particularly the part where a "flat" curve was flat only because it
was measured at 64 and above.
