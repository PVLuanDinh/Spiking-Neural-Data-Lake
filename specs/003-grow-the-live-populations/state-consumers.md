# Consumers of `Brain.state()` — enumerated before adding keys (T027)

`state()` is `cortex/brain.py:2678`, a flat ~105-key dict. Four keys are added by this feature
(`time_cells_width`, `time_cells_grown`, `time_cells_refused`, `time_cells_failed`). Adding a key to
a dict that something hashes, diffs, or counts is a silent behaviour change in that something, so
every reader was enumerated first rather than after a gate went red.

| Consumer | How it reads `state()` | Effect of new keys |
|---|---|---|
| `docs/coverage/artifacts/ab_method.py:90` — behavioural digest | `repr([st.get(k) for k in _BEHAVIOURAL])`, and `_BEHAVIOURAL` is the fixed tuple `("arm_stopped", "map_position")` (`:38`) | **None.** The digest is over two named keys, not over the dict. This was the risk the plan flagged; it does not materialise. |
| `docs/coverage/artifacts/ab_method.py:91` — numeric-leaf READS census | `_leaves(st)`, which walks every numeric leaf | **Denominator moves.** A census that read `0 of 120` reads `0 of 124`. The *count* of changed leaves is unaffected while the clock is not growing; the totals in previously recorded artifacts are now stale by four. Not a defect, but any re-run will legitimately print different totals. |
| `substrate/brain_telemetry.py:212` — cockpit snapshot | `{k: _num(v) for k, v in st.items()}` | **None.** Generic pass-through; the four keys appear in `/api/state` and are ints, which `_num` handles. |
| `substrate/cockpit.py:173` — `/api/state` | serves `snap["cognition"]` verbatim | **None.** The page reads named fields; unknown keys are ignored by the renderer. |
| `tests/` — `test_learned_evc.py`, `test_second_order_access.py`, `test_wavefront_planner.py`, `test_so101_arm.py`, `test_serial_arm.py` | all read named keys | **None.** Grepped for `set(st)` / `len(st)` / `sorted(st)` / `.keys()` — no test asserts an exact key set. |
| the other 13 `docs/coverage/artifacts/*_probe.py` harnesses | named keys, or `ab_method`'s helpers | **None** beyond the census denominator above. |

## Verdict

No consumer breaks. One consumer — the numeric-leaf census in `ab_method.py` — changes its reported
denominator, which is recorded here so a future reader does not mistake `0 of 124` against an older
`0 of 120` for a change in the brain.
