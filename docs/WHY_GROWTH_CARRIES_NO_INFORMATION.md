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
| 1 | `cortex/brain.py:1603` | `max(1, int(round(el)))` -- an int cast over a refined float | `_learned_latency` differs on 0 of 300 ticks: 100% of the refinement |
| 2 | `cortex/brain.py:1623` | `max(rb.values())` -- keeps the magnitude, drops the winning faculty's KEY | permuting keys while preserving values leaves `Brain.state()` byte-identical at `d7a6d81fc43d83f9` |
| 3 | `brainstem/autonomous_loop.py:696` | `[td[i] + ext[i] ...]` -- blind add into a common-mode scalar | the timing term is added to both non-HOLD actions EQUALLY, per the code's own comment |
| 4 | `brainstem/autonomous_loop.py` `q[unit]` | unit IDENTITY | relabelling unit ids leaves 0 of 300 actions changed, on 10 of 10 seeds |
| 5 | `cortex/executive_control.py:80-98` `conflict()` | scale AND permutation | documented invariant, with an assert at `:263-266` enforcing it |
| 6 | `substrate/growable.py` `fill=0.0` | new units emit 0 until trained | the docstring says so; ripple consumer delta measured `0.0` |

Sites 1 and 3 are the same defect twice on one path. `basal_ganglia/action_gating.py:114` does
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
