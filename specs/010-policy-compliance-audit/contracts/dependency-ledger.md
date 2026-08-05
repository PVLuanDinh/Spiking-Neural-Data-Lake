# Contract: `tools/dependency_ledger.py`

**Purpose**: assert the declared-dependency set and the imported-package set agree in both
directions. Satisfies spec SC-004 and SC-005 (P7).

## Interface

```bash
python tools/dependency_ledger.py            # report; exit 0
python tools/dependency_ledger.py --assert   # exit 1 on any unexempted violation
```

Registered as a step in `.github/workflows/ci.yml`'s **lint** job, beside `citation_guard.py`.
Deliberately NOT in `run_selfchecks.py` — see plan Structure Decision.

## Output format

```
==============================================================================
dependency_ledger -- every import declared, every declaration used
==============================================================================
scanned <N> files under src/, <M> distributions in pyproject

--- UNDECLARED: imported under src/ but absent from pyproject ---
  scikit-learn        (unguarded, top-level)
    src/snn_data_lake/reasoning/deep_stdp_2stage_svm.py:80
    ...

--- UNUSED: declared in pyproject, imported nowhere ---
  pillow              [hf]
  pyarrow             [lake]

--- SELF-DESCRIPTION ---
  pyproject.toml:1  says "torch + numpy"; core dependencies are torch, numpy, mujoco

UNDECLARED: <n>   UNUSED: <n>   EXEMPT: <n>
```

## Expected FIRST RUN output (counted before adoption)

Per Standing Invariant 1 and the project's rule that a gate's output is counted before any rule
leans on it. **This gate MUST fail on today's tree.** A first run that passes would mean the
instrument is not measuring what it claims.

| bucket | expected | note |
|---|---|---|
| UNDECLARED | **≥ 1 distribution** (`scikit-learn`, ~5 sites; check `nir`) | unguarded top-level, breaks a clean install |
| UNUSED | **≥ 2** (`pillow`, `pyarrow`) | declared, imported nowhere in `src/`, `tests/`, `tools/` |
| self-description | **1** | `pyproject.toml:1` and `:27` say "torch + numpy"; `mujoco==3.11.0` is a third core dep |

**Adoption rule**: if the run returns an allowlist-sized list (say >30 entries needing blanket
exemption), **the gate is dropped, not the tree** — precedent is the broad citation rule that died
at 475 violations while its narrow form survived at 21. A work-list-sized result is adoptable.

## FIRST RUN, ACTUAL (2026-08-04) — counted before adoption, per Standing Invariant 1

**Final: UNDECLARED 2, UNUSED 2, EXEMPT 0. A four-item work list — adoptable.**

| bucket | dist | detail |
|---|---|---|
| UNDECLARED | `arcengine` | **2 UNGUARDED sites** — `reasoning/arc_agi/play.py:37`, `play_brain.py:39`. Exists nowhere in the repo. **The 48-agent audit did not find this one**; it only named `scikit-learn` and `nir`. |
| UNDECLARED | `nir` | 1 guarded (lazy) site — `substrate/nir_export.py:121`. Cannot break a clean install; real but low severity. |
| UNUSED | `pillow` | declared `[hf]`, imported nowhere |
| UNUSED | `pyarrow` | declared `[lake]`, imported nowhere |

**The counting is the story, and it is why this bar exists.** The run sequence was
**13 → 6 → 14 → 5 → 4**, and every drop was a fix to the INSTRUMENT, never a narrowing of the tree:

1. **13** — the naive version reported `arcengine`, `model`, `env`, `play`, `data`, `rule-inducer`,
   `softhebb-mnist`... as undeclared PyPI packages. All were local modules imported flat.
   *Allowlist-shaped — under this project's rule that kills the gate or fixes the instrument.*
2. **6** — added sibling-directory resolution. `training` (the repo's own dir) still reported, and
   `pytest`/`ruff`/`pre-commit`/`pyinstaller` reported as UNUSED though they are tools by design.
3. **14** — scanning `research/` and `integrations/` for UNDECLARED made it *worse*. That produced
   the scoping insight below.
4. **5** — scoped UNDECLARED to `src/` (the only tier that ships in the wheel) while keeping USED
   broad. `eprop_lm`, `eprop_lm_alif`, `ar_spiking_lm` still misreported.
5. **4** — resolution by EXISTENCE anywhere in the repo, not by location. Those three live in
   `training/` and are imported flat by serve scripts in a different top-level dir, so no ancestor
   walk from the importer could ever have found them.

**The asymmetric scope is the load-bearing design decision.** UNDECLARED is `src/`-only because
`pyproject.toml` ships only `src/snn_data_lake`; USED is repo-wide because a dependency imported
from `integrations/` is being used. Folding them together gave 14 entries where 2 are actionable.

**Adoption status: BUILT, COUNTED, TESTED — NOT YET ASSERTING.** `--assert` exits 1 today on the
four above. It is registered in CI only once they are resolved; turning it on now would redden the
build for findings that have not been dispositioned, which is not the same as enforcing a rule.

## Correctness risks, stated because they are this instrument's failure modes

1. **Import name ≠ distribution name.** `sklearn`→`scikit-learn`, `PIL`→`pillow`,
   `cv2`→`opencv-python`, `yaml`→`pyyaml`. A wrong mapping yields a **false UNDECLARED**, which is
   the dangerous direction (Standing Invariant 3). The mapping is an explicit table, tested, never
   inferred from the string.
2. **Stdlib detection.** Use `sys.stdlib_module_names`; a hand-maintained list rots.
3. **Extras are not installed in CI.** The gate reads `pyproject.toml` statically; it must NOT
   conclude "unused" from an ImportError at runtime.
4. **Optional imports inside try/except are still imports.** They are declared-required or
   declared-in-an-extra, but they are not exempt from the ledger. The `guarded` flag records the
   distinction; it does not excuse it.

## Verify-the-verifier (Standing Invariant 5)

`tests/test_dependency_ledger.py` MUST contain a case that makes each assert FIRE:
- a fixture importing a package absent from metadata → gate reports UNDECLARED;
- a fixture declaring a package imported nowhere → gate reports UNUSED;
- an `EXEMPT` entry that matches no real violation → gate FAILS loudly, so an exemption that blesses
  nothing cannot read as coverage (the `citation_guard.EXEMPT` pattern).

## Exemptions

`EXEMPT` entries follow `dormancy_gate.EXEMPT`'s pattern: each carries a **disposition and a
reason**, and the gate asserts every entry still matches a real violation. Legitimate reason shapes:
a build-time-only dependency (e.g. `hatchling`), or a package imported only by an excluded tier.
"It is fine" is not a reason.
