# Contributing

## Setup

CPython **3.14 free-threaded** (`3.14t`, PEP 703): `pyproject.toml` declares `requires-python = ">=3.14"`, the
package refuses a GIL build at import (`snn_data_lake.require_free_threaded`), and CI pins `3.14t`. A **Rust
toolchain** is also required: `pip install -e .` compiles the `snn_data_lake._native` extension through maturin at
install time (see `docs/INSTALL.md`). `pip install -e .` pulls the core dependencies (`pyproject.toml`
`[project] dependencies`); the dev tooling and optional integrations are extras on top.

```bash
uv venv --seed --python 3.14t .venv-lake   # any way of getting a 3.14t interpreter works
.venv-lake/Scripts/activate      # POSIX: source .venv-lake/bin/activate
pip install -e ".[dev]"          # pytest, ruff, pre-commit
```

*(Corrected 2026-10-28, cloud audit: this section said "Python 3.11+ (CI runs 3.11 / 3.12 / 3.13)" and named no
Rust toolchain; both were wrong since the cp314t requirement and the `_native` extension build of 2026-10-18.)*

## The core gates

Every change must keep these green. They are the core of the battery, not all of it: `AGENTS.md`'s gate block
and `.github/workflows/ci.yml` list the rest (the tools gates that `run_selfchecks.py` runs, and the lint job's
policy gates).

```bash
python run_selfchecks.py                        # every registered module's __main__ self-check, then the tools gates
pytest                                          # unit suite, >=70% line coverage gate
python -m ruff check .                          # lint (rules E, F, I, UP, B)
python -m ruff format --check src tests tools   # formatting -- this exact scope is what CI runs
```

*(Corrected 2026-10-28, cloud audit: this section was titled "The four gates", said CI enforces them on
{ubuntu, windows} x {3.11, 3.12, 3.13}, counted 277 self-checks (375 modules today) and gave the format scope as
`src tests`; CI runs `src tests tools` on 3.14t. The GitHub Actions workflow has been disabled since 2026-08-20, so
the operator's local battery is the verification of record.)*

## Invariants

- **Dependencies.** `torch` and `numpy` are core deps; the zero-dependency-core contract was removed
  project-wide on 2026-07-20. Modules may import them freely. Anything heavier goes behind an
  optional extra (`arm`, `gpu`, `nir`, ...). There is no torch-free path: the `NORD_TORCH=0`
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
3. Make it reachable from a live path, or add it to `gates/dormancy_gate.py` `EXEMPT` with a
   reason — Policy 3 fails CI otherwise.
4. Run the four gates.

## Safety-critical code

The robot-arm control path is safety-critical — see [SECURITY.md](SECURITY.md). Changes to the
five envelope files require review and a re-run of their self-checks.
