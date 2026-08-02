# Saturation regime census (T018, SC-003)

Every claim this feature makes must name the regime it was measured in. The readout's
saturated branch is a RANGE CEILING, not an estimate: once every cell has fired the
population holds no further information, so an effect measured inside that branch is an
artefact of where the ceiling sits and not of how the population tiles time.

Live brain via `make_brain(persist=False)`, 3 seeds x 600 ticks.

| seed | width | unsaturated ticks | saturated ticks | unsaturated share |
|---|---|---|---|---|
| 0 | 64 | 576 of 600 | 24 | 0.9600 |
| 1 | 64 | 554 of 600 | 46 | 0.9233 |
| 2 | 64 | 600 of 600 | 0 | 1.0000 |
| **all** | | **1730 of 1800** | **70** | **0.9611** |

**Verdict**: 96.1% of live clock-advancing ticks are unsaturated. The held-out
instrument evaluates the ladder noise-free across the population's own gaps, so it reports
the UNSATURATED regime; nothing here licenses a claim about the saturated one.
