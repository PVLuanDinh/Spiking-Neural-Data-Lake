# Contract — Cockpit Afferent Sensor Endpoint

`POST /api/sensor`

Mirrors `/api/act` (`substrate/cockpit.py:360-372`) structurally, because that pattern is the one
already reviewed and trusted for a request thread that must not touch the brain.

## The three properties inherited from `/api/act`, verbatim in intent

1. **The request thread never touches the brain, the reflex, or the arm.** It validates and enqueues.
   All consumption happens on the tick thread.
2. **404 when the queue is absent.** A cockpit served without a sensor queue cannot inject sensor
   readings at all — the capability is absent, not merely disabled. This is what makes "default-OFF"
   structural rather than a matter of configuration.
3. **Validate against an explicit allow-list before enqueuing**, never after.

## Request

```json
{ "distance_m": 0.03, "temp_k": 298.1, "simulated": true }
```

| field | required | validation |
|---|---|---|
| `distance_m` | yes | finite float; **negative/sentinel rejected** (see data-model E-2) |
| `temp_k` | yes | float; `<= 0` accepted and treated as the missing-modality sentinel the chain already defines |
| `simulated` | **yes** | must be present and `true`. **Absent → 400.** Never defaulted. |

`simulated` is required and must be `true` because this endpoint has no way to originate a real
reading — real readings arrive over serial, and `pyserial` is absent. An endpoint that accepted
`"simulated": false` would be offering to mislabel.

## Responses

| status | when |
|---|---|
| `202` | accepted and enqueued (the tick thread has not consumed it yet — do not report it as applied) |
| `400` | malformed body, missing field, non-finite value, negative distance, or `simulated` absent/false |
| `404` | no sensor queue was supplied to `serve()` — the capability is not present |
| `503` | queue full (back-pressure; the tick thread is behind) |

`202`, not `200`: the reading has been *queued*, not *acted on*. Returning `200` would let the
cockpit claim an effect the brain has not had yet — the precise shape P4c's "what the cockpit
REPORTS must be what the brain MEASURES" forbids.

## What this endpoint may NOT do

- May not write to the durable lake directly (P4b — the tick thread owns persistence).
- May not construct or reach a `Brain` (P4).
- May not bypass `parse_esp32_line` by accepting a pre-built feature vector (P2 — one transduction
  mechanism; accepting feats would be a second one).
- May not actuate. It is afferent only; `/api/act` remains the sole efferent path.

## Reachable red (P8)

Each validation rule owes a request that triggers it:

| rule | red request |
|---|---|
| `simulated` required | `{"distance_m": 0.03, "temp_k": 298.1}` → 400 |
| sentinel rejected | `{"distance_m": -1, "temp_k": -1, "simulated": true}` → 400 |
| non-finite rejected | `{"distance_m": "NaN", "temp_k": 298.1, "simulated": true}` → 400 |
| queue absent → 404 | serve without a sensor queue, POST anything → 404 |
