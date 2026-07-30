# Contributing

## Setup

Python 3.11+ (CI runs 3.11 / 3.12 / 3.13). `pip install -e .` pulls the core dependencies
(`torch`, `numpy`); the dev tooling and optional integrations are extras on top.

```bash
python -m venv .venv-lake
.venv-lake/Scripts/activate      # POSIX: source .venv-lake/bin/activate
pip install -e ".[dev]"          # pytest, ruff, pre-commit
```

## The four gates

Every change must keep all four green; CI enforces them on {ubuntu, windows} x {3.11, 3.12, 3.13}.

```bash
python run_selfchecks.py                  # every registered module's __main__ self-check (277)
pytest                                    # unit suite, >=70% line coverage gate
python -m ruff check .                    # lint (rules E, F, I, UP, B)
python -m ruff format --check src tests   # formatting -- this exact scope is what CI runs
```

## Invariants

- **Dependencies.** `torch` and `numpy` are core deps; the zero-dependency-core contract was removed
  project-wide on 2026-07-20. Modules may import them freely. Anything heavier goes behind an
  optional extra (`arm`, `lake`, `gpu`, ...). There is no torch-free path: the `NORD_TORCH=0`
  off-switch was removed on 2026-07-29, so do not add a new one, and do not gate a module on
  torch's absence.
- **Determinism.** Seed with `random.Random(seed)` per instance — never module-level import-time
  seeding. Use `zlib.crc32()` for persisted hashes, not the builtin `hash()`.
- **ASCII-only output.** Windows cp1252 crashes on non-ASCII in `print()`. Keep console output ASCII.
- **Formatting.** The tree is `ruff format`-clean; run `ruff format src tests` before a gate. E501
  stays relaxed on `src/**` because `ruff format` does not wrap long lines.

## Adding a module

1. Give it an `if __name__ == "__main__"` self-check: assert-based and deterministic. State the
   honest baseline/control it is measured against — no rigged comparisons.
2. Register it in **both** `src/snn_data_lake/__init__.py` `REGIONS` **and** `run_selfchecks.py`
   (the `CORE` list for a product module, `RESEARCH` for a `snn_data_lake.reasoning.*` build).
   `brain_atlas` asserts filesystem↔map parity for every `REGIONS` directory, so an unregistered
   module there fails the gate.
3. Make it reachable from a live path, or add it to `substrate/dormancy_gate.py` `EXEMPT` with a
   reason — Policy 3 fails CI otherwise.
4. Run the four gates.

## Safety-critical code

The robot-arm control path is safety-critical — see [SECURITY.md](SECURITY.md). Changes to the
five envelope files require review and a re-run of their self-checks.
