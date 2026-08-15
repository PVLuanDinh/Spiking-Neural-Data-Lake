# Quickstart — Validating Live Simulated Sensory Transduction

Gate env is `.venv-lake`. Run everything from the repo root. Use **PowerShell**, not Bash (a
context-mode hook intercepts network-touching Bash calls).

---

## 0. The gate batch, run at EVERY commit

`ruff format --check` is in the batch deliberately — a prior arc shipped formatting drift because
the local batch ran only `ruff check`.

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py && .venv-lake/Scripts/python.exe -m pytest -q && .venv-lake/Scripts/python.exe -m ruff check . && .venv-lake/Scripts/python.exe -m ruff format --check . && .venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert && .venv-lake/Scripts/python.exe tools/p6_gate.py --assert && .venv-lake/Scripts/python.exe tools/citation_guard.py --assert && .venv-lake/Scripts/python.exe tools/red_reachability.py --assert && .venv-lake/Scripts/python.exe tools/dependency_ledger.py --assert && .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.import_manifest
```

---

## 1. The chain still works (baseline, before any change)

Confirms the existing transduction path and the shapes data-model.md records.

Expected, measured: `"0.030, 298.1"` → parse `(0.03, 298.1)` → normalize `(0.01, 0.2475)` → 64 feats
→ 7 spike events. `"STOPPED"` and `"garbage"` → `None`.

---

## 2. The sentinel guard (data-model E-2 — the defect this feature owns)

**This is the check most likely to be skipped and the one that matters most.**

- Feed `"-1, -1"` (firmware "nothing within 3 m").
- Assert the contact branch does **NOT** fire and `pair()` is **not** called.
- **Reachable red**: remove the guard and re-run — the branch fires, because `-1.0 < 0.04` and
  `isfinite(-1.0)` are both true. Record the red output; a guard that has never been seen to fail is
  not a guard.

---

## 3. "Only ever adds caution" — the measured table (SC-004)

Not an argument. A matrix, run and recorded.

For each row, compare the reflex outcome **with** the simulated channel present against **without**
it, on the same input sequence and seed:

| ch6 | ch7 | ch8 (simulated) | required outcome |
|---|---|---|---|
| safe | safe | safe | identical to baseline |
| **breach** | safe | safe | identical to baseline — a simulated-safe channel may not mask a real breach |
| safe | **breach** | safe | identical to baseline |
| **breach** | **breach** | safe | identical to baseline |
| safe | safe | **breach** | fires (new stop added) |
| **breach** | safe | **breach** | fires no later than baseline; action same or more severe |
| **breach** | **breach** | **breach** | fires no later than baseline |
| any | any | **non-finite** | fires — sensor fault is max danger |

Sweep ch8 across safe → danger rather than testing one value (Standing Invariant 4: absence of a
warning in one case is not absence in general).

The claim is **"same or more protective, never less"**. Any row where the with-channel run fires
LATER, or selects a LESS severe action, than baseline fails the feature.

---

## 4. Channel 6 and 7 are pinned (SC-002)

Assert `DEFAULT_RULES[6] == (-50.0, 800.0, "STOP")` and `DEFAULT_RULES[7] == (-300.0, 300.0,
"WITHDRAW")`, and that a frame's index 6 and 7 still carry motor current and force/torque.

**Reachable red**: shift the frame layout by one and confirm the test fails. Nothing in the tree
pins this today, so without the red this test is just a restatement of the source.

---

## 5. The SIMULATED marker survives every hop (SC-005, R-001)

Construct a simulated reading, drive one tick, then assert the marker is present in: the brain state
report, the lake record if one was written, and the cockpit display string.

**Reachable red**: strip the marker at each hop in turn, one at a time, and confirm the test fails
at each. Three separate reds — a single end-to-end assertion can pass while an intermediate hop
drops and a later hop re-derives it.

Also assert the endpoint contract's four reds (contracts/sensor-endpoint.md).

---

## 6. THE DECIDING MEASUREMENT — does the feed CHARGE? (FR-010, P3)

Everything above proves the wire is correct. **This decides whether it stays.**

Use `tools/charge_probe.py` as the template. It carries an external positive control (47/60 flips
with all drive sources neutered); honour it, or demonstrate this probe's own sensitivity first.
**A null from a harness that has never reported a difference is not a null** (Standing Invariant 5).

Procedure:

1. Run the brain with the simulator ON, driven into the danger zone. Record the outcome.
2. **Neuter the sensor feed** (readings produced but not consumed). Same seed. Record the outcome.
3. The feed CHARGES iff an outcome the brain would otherwise reach actually changes.

Per-item attribution (P8 guard 3): report separately whether the charge appears in the **reflex**
path, the **amygdala** path, or the **spike/perception** path. "Something changed" is not
attribution.

### If it charges

The exemptions are settled per what was actually proven — `esp32_encoder_path`'s reason is corrected
to name the real blockage (only `read_serial_lines` needs `pyserial`; `parse → encode → spikes`
never did), and `fear_conditioning`'s DISPOSITION note is amended per the owner ruling.

### If it does NOT charge

**FR-012 executes: the wire is REVERTED and both modules are re-exempted with corrected reasons.**
This is pre-committed in research.md before the number was known, so it cannot be renegotiated after
seeing one. A null here is the licence working.

---

## 7. The owner's condition, checked rather than intended

- **Coverage band stays 0.** Diff `docs/coverage/` across the lane and assert **no band rises**.
  A band that rose on simulated evidence violates the condition the authorization was given under.
- **Every threat value derived from a simulated event is marked SIMULATED** — covered by step 5.

---

## 8. Safety-envelope scope, verified by diff not by promise

```bash
git diff --name-only main -- src/snn_data_lake/brainstem/
```

**Must list `reflex.py` and nothing else.** `serial_arm.py`, `so101_arm.py`, `arm_config.py`,
`interpreter.py` are untouched. Also assert `arm_config.SAFETY_CALIBRATED is False` and that
`serial_arm.assert_calibrated()` still refuses.

Run this **before** the envelope commit is pushed, not after.
