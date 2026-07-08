# Contributing

## Setup

Python 3.11+ (CI runs 3.11 / 3.12 / 3.13). The core has no runtime dependencies; only the
dev tooling and optional integrations need installing.

```bash
python -m venv .venv-lake
.venv-lake/Scripts/activate      # POSIX: source .venv-lake/bin/activate
pip install -e ".[dev]"          # pytest, ruff, pre-commit
```

## The three gates

Every change must keep all three green; CI enforces them on {ubuntu, windows} x {3.11, 3.12, 3.13}.

```bash
python run_selfchecks.py         # runs every registered module's __main__ self-check
pytest                           # unit suite, >=70% line coverage
python -m ruff check .           # lint (rules E, F, I, UP, B)
python -m ruff format --check tests
```

## Invariants

- **Zero-dependency core.** `src/snn_data_lake/` is standard-library only — `dependencies = []`
  in `pyproject.toml` stays empty. Integrations that need third-party packages go behind an
  optional extra (`arm`, `lake`, `gpu`, ...) or in `research/` (ruff-excluded, not in CI's core gate).
- **Determinism.** Seed with `random.Random(seed)` per instance — never module-level import-time
  seeding. Use `zlib.crc32()` for persisted hashes, not the builtin `hash()`.
- **ASCII-only output.** Windows cp1252 crashes on non-ASCII in `print()`. Keep console output ASCII.
- **Line length.** E501 is intentionally relaxed on `src/**` (the core is hand-aligned). Do not reflow.

## Adding a module

1. Give it an `if __name__ == "__main__"` self-check: assert-based, deterministic, zero-dependency.
   State the honest baseline/control it is measured against — no rigged comparisons.
2. Register it in `run_selfchecks.py` — the `CORE` list for a reusable library primitive
   (importable as `snn_data_lake.<mod>`, ruff-enforced), or the `RESEARCH` list for a
   `research/` module.
3. Run the three gates.

## Safety-critical code

The robot-arm control path is safety-critical — see [SECURITY.md](SECURITY.md). Changes to the
five envelope files require review and a re-run of their self-checks.
