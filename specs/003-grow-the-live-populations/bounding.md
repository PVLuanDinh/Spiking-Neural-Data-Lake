# Durable footprint of autonomous growth (T049, FR-014, SC-008)

Reproduce:

```bash
python tools/timing_heldout.py --bounding --bounding-ticks 600
python tools/timing_heldout.py --bounding --bounding-ticks 2000
```

Live brain via `make_brain(persist=True)` with `NORD_TIME_CELLS_GROW=1`, against a throwaway
`NORD_HOME` so the operator's durable state is never touched.

| ticks | width | grown | refused | failed | durable bytes |
|---|---|---|---|---|---|
| 600 | 64 → 69 | 5 | 35 | 0 | 1422 |
| 2000 | 64 → 69 | 5 | 36 | 0 | 1422 |

## Verdict: BOUNDED, on both axes, and one of the two bounds had to be built

**The store is bounded because overwrite IS the prune policy.** `home.save_state` writes one atomic
JSON blob and replaces it; there is no append, so there is no retention constant to tune (P1) and no
tail to prune. The footprint is `O(n_cells)` — 1422 bytes at width 69 — and it does not depend on how
long the brain runs. This is the reason the clock's state does not go in the lake: `_T_GROWTH` is
180.0 of 187.5 MB (96.0%) of the real lake, `specs/002`'s prune policy for it is unbuilt, and
`load_growth` costs 0.400 s and 187.5 MB resident on *every* Brain construction against
`load_state`'s 0.110 ms.

**The population is bounded because the halt fires** — and the first version of it could not.

The original earn-your-place rule was "grow only if held-out error strictly falls". MEASURED: 600
live ticks, **width 64 → 104, 40 growths, 0 refusals**. The rule never said no, and could not: on a
noise-free tiling metric, splitting the widest gap ALWAYS lowers held-out error, so `delta < 0` is
satisfied by every insertion forever. It is a green light that cannot go red — the same
Standing-Invariant-5 failure this project keeps finding inside its own instruments, this time in the
thing that was supposed to be the safety bound.

The bar that actually binds is the accumulator's own noise. For a Wiener process with drift `drive`
and per-step variance `noise_sd^2`, the first-passage dispersion at level `theta` is
`sqrt(theta * noise_sd^2 / drive^3)` — the result the module header already cites and that `decode()`
already reports empirically as its `sd`. A bracket narrower than that dispersion cannot be resolved
by this population however finely it is tiled, so a cell that only subdivides below the floor buys
precision the substrate cannot deliver. `TimeCellClock.resolvable()` is derived entirely from
existing state, so no threshold constant was introduced.

With the floor in place growth stops at **width 69 after 5 insertions**, and stays there: identical
at 600 and 2000 ticks. The halt is OBSERVABLE rather than inferred — `time_cells_refused` climbs
while `time_cells_grown` stands still, both exported from `state()`. An absence of growth events
cannot tell "stopped because it stopped helping" from "never ran"; a rising refusal count can.

## What is still not bounded, stated rather than implied

Nothing bounds the *number of reset boundaries* the brain reaches, so the refusal counter grows
without limit. That is a counter, not a population and not a store: 36 refusals at 2000 ticks is
four bytes of `int`, and it is the intended cost of making the halt visible.
