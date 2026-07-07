# Agents — Spiking Neural Data Lake

**Core invariant:** zero-dependency core (`src/snn_data_lake/`) — stdlib only, no runtime deps.

## Three gates (all must pass)

```bash
python run_selfchecks.py   # Smoke test every core module
pytest                     # >=70% coverage
python -m ruff check .     # Rules E,F,I,UP,B
```

## Safety envelope (modify with care + re-test)

- `reflex.py`, `serial_arm.py`, `arm_sim.py`, `arm_config.py`, `interpreter.py`
- All motor commands → `ArmSim.apply()` gate

**Determinism:** `random.Random(seed)`, `zlib.crc32()` for persisted hashes, ASCII-only output.

See `CLAUDE.md` for full onboarding details, layout, and optional extras.
