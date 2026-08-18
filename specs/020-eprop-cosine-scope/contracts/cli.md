# Phase 1 Contracts — CLI surface

**Feature**: 020-eprop-cosine-scope | **Date**: 2026-08-17

This project's gates expose a CLI, not an API. Two surfaces, one module (the `eprop_transport_gate`
shape: the on-demand half is a `--flag` path of the same module, not a second file — `tools/` has
**zero** ratchet headroom, so a second file there reddens CI on its own commit).

**Home**: `src/snn_data_lake/substrate/` — registered in `__init__.py` REGIONS and
`run_selfchecks.py`. A registered substrate module costs the P4c ratchet nothing (`compare -> []`); an
unregistered one costs `[('src/snn_data_lake/substrate', 1, 0)]`.

---

## `python -m snn_data_lake.substrate.gradient_claim_gate` — report mode (default)

Prints the census, then APPLIES THE BOUNDS and exits 1 on a finding -- the same shape every sibling
gate ships.

**CORRECTED 2026-08-17 by measurement.** This section previously said "exits 0 regardless of findings.
Report mode never fails; that is what makes it safe to run anywhere and is consistent with the sibling
gates." The consistency claim was FALSE: `dormancy_gate` and `method_dormancy_gate` carry no
`--assert` flag at all and simply assert in `main`; `eprop_transport_gate` and `doc_budget_gate` have
the flag AND assert in `main`. All four read exit 0 only because the tree is clean -- `dormancy_gate`
bare exited **1** earlier that day, when `gradient_claim_gate` was itself dormant.

Being the outlier had a cost that was not theoretical. `run_selfchecks.py` invokes each registered
module through `runpy.run_module(..., run_name="__main__")` with NO argv, so a never-failing default
path made this module's registration a `PASS` that could not fail on a finding -- "a gate wired in
BARE is a log line wearing a gate's clothes", the lesson `ci.yml` records three times. Verified after
the fix by planting a real finding: `runpy` raises `SystemExit=1` with it and `SystemExit=0` without.

```
GRADIENT-CLAIM CENSUS
  rules seen           : <n>          <- liveness floor, see --assert
  classified local     : <n>
  classified claimant  : <n>
  classified reference : <n>
  classified control   : <n>
  MISSING DECLARATION  : <n>  (ratchet, baseline <b>)
  UNDETERMINABLE       : <n>  (ratchet, baseline <b>)
  unclassified         : <n>  <- must be 0
```

## `--assert` — the CI gate

Exit **0** only if all hold; exit **1** naming which failed:

| bound | kind | rationale |
|---|---|---|
| `unclassified == 0` | ABSOLUTE | a hole in the census supports no claim about scope |
| `missing_declaration <= baseline` | RATCHET | green means "no worse", never "correct" — and the output must SAY so |
| `undeterminable <= baseline` | RATCHET | else the honesty state becomes a sweep-it-here drawer |
| `rules_seen >= <floor>` | **LIVENESS** | a dead walk reports zero unclassified and looks perfect. Four gates in this tree already fail exactly this way. |
| `parse_failures == 0` | ABSOLUTE | 378/378 parse today; a new failure is a silent census hole |

## `--self-check` — verify the verifier

Constructs its own REDs against the **real tree** using an in-memory overlay (the `p6_gate.walk(root,
overlay=...)` pattern), touching nothing on disk. Must construct and confirm at minimum:

1. a rule with no declaration and a `CLAIM` inference → `MISSING_DECLARATION`, count rises, ratchet fires
2. a declaration removed from a claimant → does NOT silently become `LOCAL`
3. a deleted file (overlay `None`) → `rules_seen` falls and the **liveness floor fires**
4. an unparseable module → `parse_failures` rises and the absolute bound fires
5. a **GREEN control**: the repaired shape is NOT flagged (else the checker cannot tell fixed from broken)
6. determinism: two runs over the same tree produce an identical CLASSIFICATION -- the same
   qualnames in the same order with the same states, and identical counts. NOT byte-identical
   stdout: FR-013a prints a wall-clock figure that varies by design, so a raw stdout diff is
   guaranteed to fail and would be the wrong check. Verified 2026-08-17: classification
   identical, stdout identical once the cost line is removed.

## `--update-baseline`

Writes the machine-generated baseline to **`tools/gradient_claim_baseline.json`**. **MUST refuse to
record a worse value** and exit 1 with the offending counts. No hand-editable literal anywhere.

**Home decided by measurement, 2026-08-17.** `p4c_ratchet`'s corpus is
`git ls-files --cached --others --exclude-standard -- *.py` (`p4c_ratchet.py:114`, pathspec at `:171`),
so a `.json` under `tools/` costs that zone nothing — proven by `tools/p4c_ungoverned_baseline.json`
sitting there today while the zone reads 30/30 green. It is tracked and un-ignored, so it functions as
a ratchet CI can read, and staying outside `src/snn_data_lake/` keeps it out of the wheel
(`[tool.hatch.build.targets.wheel] packages = ["src/snn_data_lake"]`, no filter).

**Reuse, do not re-author.** `p4c_ratchet.py:441-449` is the working refuse-a-rise writer: it compares
offered counts against the baseline, prints each risen zone and exits 1. `_load_baseline` (`:255-268`)
reads with `utf-8-sig` deliberately, so a hand-edit's BOM does not crash the gate.

**CORRECTED 2026-08-17 during implementation.** This paragraph recommended `doc_budget_gate.py:49-50`
as the repo-root precedent — a four-level counted `__file__` climb. **That recommendation was wrong
and the tree already said so**: `cortex/laminar_local_readout._repo_root` records that a counted depth
"encodes the module's then-current nesting", and moving the file one level silently resolved to the
WRONG directory, making a gate go green while testing synthetic data instead of the real pack. The
implementation therefore REUSES `substrate/self_mod_harness._repo_root` (anchored on
`run_selfchecks.py` + `src`) rather than adding a fourth resolver. The objection raised against reuse
— "it drags heavy imports into a stdlib-tier gate" — was measured FALSE: 47 ms / 111 modules with
torch NOT loaded, against `dormancy_gate`'s 25 ms / 58. **The wheel limit still holds** and is stated
in the module docstring: with no repo root, `_repo_root()` returns None, the walk reports an empty
census, and the LIVENESS bound turns that into a loud failure instead of a silent pass.

---

## `--measure` — the on-demand instrument (NOT in CI)

Never asserts in CI. Runs the numeric half: per claimant, the aligned cosine, the within-matrix
permutation null, the discriminating-power pair, and the separation report.

```
DISCRIMINATING POWER   null=within_matrix_permutation  draws=<int>  seed=<int>
  <qualname>
    aligned            : <float>          (<float> sd above null)
    coherence cand/ref : <float> / <float>
    null p50 / p95     : <float> / <float>
    0.5 sits           : <float> sd above null,  P(null > 0.5) = <float>
    verdict            : RULE_JUDGED | INSTRUMENT_FALSIFIED | UNAVAILABLE
```

`draws` and `seed` are on the header line because without them the block is not reproducible on an
unchanged tree (FR-009e). `coherence` is on its own line because it is the PRECONDITION for the two
lines under it: above the coherence ceiling the permutation is not a null, `0.5 sits` goes negative,
and the verdict is `INSTRUMENT_FALSIFIED` regardless of what `aligned` reads (FR-009d).

Gradients are obtained by running each fixture in a SUBPROCESS (FR-013b). The gate imports no
rule-carrying module in any mode; a fixture whose subprocess fails to import is reported
`UNAVAILABLE`, which is what that verdict means.

Exit codes: **0** measured; **2** NOT-ELIGIBLE / UNAVAILABLE (distinct from a failed measurement —
FR-005). Output is recorded to an artifact so "nobody ran it lately" is visible rather than
indistinguishable from "it passed" (FR-015).

**Reported-not-asserted**: the isotropic, within-unit and random-relabel nulls appear in this output
and gate nothing (FR-009c).

---

## Declaration contract (what a rule module carries)

```python
GRADIENT_CLAIM = {
    "<Class>.<method>": ("APPROXIMATES_GRADIENT", "<one-line reason>"),
}
```

Module-scope constant, `ast.literal_eval`-able, **never imported** by the checker. Keyed by qualified
symbol. **Six** dispositions: `LOCAL` / `APPROXIMATES_GRADIENT` / `EXACT_GRADIENT` / `CONTROL` /
`NOT_A_RULE` / `DIRECT_SOLVE`.

`DIRECT_SOLVE` (FR-002g) is for a genuine rule that owes no cosine and is **not local** —
`ng_snn::fit`'s closed-form batch solve. `LOCAL` asserts locality; `DIRECT_SOLVE` explicitly does not,
which is the distinction POLICIES.md requires be kept "distinctly rather than lumping it in".

`NOT_A_RULE` was added after the first real declaration pass measured four as insufficient (FR-002e):
a diagnostic, a search, two symbolic-grammar operations and a combinator could not be described
truthfully by any of the original values. It is **declaration-only** — the inference cannot emit it —
and its count reports enumeration imprecision, not tree health.

**Name constraint, measured**: must not be a method, and must avoid any `LEARN_VERBS` substring —
`declare_constraint` (con-**TRAIN**-t), `get_adapter` (**ADAPT**-er), `plasticity_claim` (**PLAST**)
all become DORMANT findings. `GRADIENT_CLAIM` is measured clean.

**Cost, measured with positive controls in the same run**: adding this constant moves
`dormancy_gate` import edges +0, `method_dormancy_gate` n_defs 66→66 / dormant 0→0, `import_manifest`
targets +0 — while the controls moved (+1 edge, n_defs 67 / dormant 1, +1 target). No registration is
required for a declaration.
