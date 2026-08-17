# Phase 1 Contracts — CLI surface

**Feature**: 020-eprop-cosine-scope | **Date**: 2026-08-17

This project's gates expose a CLI, not an API. Two surfaces, one module (the `eprop_transport_gate`
shape: the on-demand half is a `--flag` path of the same module, not a second file — `tools/` has
**zero** ratchet headroom, so a second file there reddens CI on its own commit).

**Home**: `src/snn_data_lake/substrate/` — registered in `__init__.py` REGIONS and
`run_selfchecks.py`. A registered substrate module costs the P4c ratchet nothing (`compare -> []`); an
unregistered one costs `[('src/snn_data_lake/substrate', 1, 0)]`.

---

## `python -m snn_data_lake.substrate.<module>` — report mode (default)

Prints the census and exits **0 regardless of findings**. Report mode never fails; that is what makes
it safe to run anywhere and is consistent with the sibling gates.

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
6. determinism: two runs over the same tree produce identical output

## `--update-baseline`

Writes the machine-generated baseline. **MUST refuse to record a worse value** and exit 1 with the
offending counts. No hand-editable literal anywhere.

---

## `--measure` — the on-demand instrument (NOT in CI)

Never asserts in CI. Runs the numeric half: per claimant, the aligned cosine, the within-matrix
permutation null, the discriminating-power pair, and the separation report.

```
DISCRIMINATING POWER
  <qualname>
    aligned            : <float>
    null p50 / p95     : <float> / <float>
    0.5 sits           : <float> sd above null,  P(null > 0.5) = <float>
    verdict            : RULE_JUDGED | INSTRUMENT_FALSIFIED | UNAVAILABLE
```

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
symbol. Four dispositions: `LOCAL` / `APPROXIMATES_GRADIENT` / `EXACT_GRADIENT` / `CONTROL`.

**Name constraint, measured**: must not be a method, and must avoid any `LEARN_VERBS` substring —
`declare_constraint` (con-**TRAIN**-t), `get_adapter` (**ADAPT**-er), `plasticity_claim` (**PLAST**)
all become DORMANT findings. `GRADIENT_CLAIM` is measured clean.

**Cost, measured with positive controls in the same run**: adding this constant moves
`dormancy_gate` import edges +0, `method_dormancy_gate` n_defs 66→66 / dormant 0→0, `import_manifest`
targets +0 — while the controls moved (+1 edge, n_defs 67 / dormant 1, +1 target). No registration is
required for a declaration.
