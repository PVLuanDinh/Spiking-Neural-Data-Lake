# D-434: push per part, not per arc (PROVISIONAL), evidence record, 2026-11-07

The binding rule is in `docs/POLICIES.md`, cross-cutting invariants, "AMENDED 2026-11-07, owner: PUSH PER PART —
PROVISIONAL", and its row in the enforcement table. This file carries what the rule rests on and what it does not say.

## Owner words (verbatim, D-434)

"push build upon positive testing and integration of parts to the main build to avoid long wait time for batteries"

## Incident

In ARC6 and ARC7 every row waited hours for one arc battery before anything reached `main`. ARC7 composed and gated
ROUTES (0a9bc178), D7 c1 (7940301a) and D9 (53895374) onto the integration branch by 2026-11-07 16:15 with nothing
pushed since 847706a7 (the ARC6 push).

## Rationale

A part that is gated, reviewed and tested on the tests that read what it changed can reach `main` without waiting
for rows it does not touch. The 2026-09-02 per-arc rule bought whole-battery coverage per push; this rule trades
that for earlier integration and keeps the arc-end full coverage (D-348) as the backstop.

## What the rule does not relax

- One change per commit.
- A bad push is undone by a new revert commit on top, never a force-push.
- The push go is the owner's: per push, or a standing go the owner gives in words.
- Lane branches are never pushed; only the integration head is.
- The panel before the arc closes, the arc-end full shard coverage and closing record (D-348), and the safety envelope.

## Post-push read-back (part of every per-part push)

After the push: `git fetch` then `git rev-parse origin/main`; it must equal the pushed head (full 40-hex). Negative
control: the pre-push `origin/main` sha is shown NOT equal to the pushed head, so the check can fail.

## Selector notes (why the rule is PROVISIONAL)

- No change-scoped selector exists yet. Until it does, the selection is made by hand and says so.
- Unit 1's hand selection (847706a7..unit head) is the union of (a) the static import closure of the changed src files
  (the integrator's ast-closure script, sha12 82b5290683dc), (b) the test files that read the changed docs and evidence
  by path, and (c) the Brain-building tests that reach the changed modules by string, subprocess or importlib (D-442
  item 3, reviewer-2 #9962): 151 test files, list sha12 675ea30c6dea = 134 + 9 + 8. Known positive: a direct importer of
  a changed module is selected; known negative: a test that imports the package but reaches no changed module is not.
- When `engine/brain.py`'s closure is hit, the selection is near-full: change-scoped is not small for such parts.

## The selector's gate specification (POLICIES bar: a provisional owner rule carries its gate's spec)

Full spec: reviewer-2's SELECT-UNIT-SPEC v2, sha12 d26704a1ed7c (ARC8 build). It extends `gates/docs_only_battery.py`'s
reader map to source files, lives in `src` with `--self-check` and `--assert`, and ports the ast closure above.
- Inputs: base and head full shas and a clean detached tree; a dirty or mismatched tree is refused.
- Four reader passes, reported separately: (a) the static import closure; (b) literal path readers of changed docs and
  data; (c) dynamic reachers (importlib, `__import__`, subprocess / `python -m`, module-name strings, `make_brain` when
  `engine/brain.py`'s closure is hit); (d) global-influence files: a changed `conftest.py` selects every test under its
  directory, and a change to the pytest configuration, `src/snn_data_lake/__init__.py` or `run_selfchecks.py` selects
  the full suite. Changed test files are always selected.
- Output: the selected test files with a reason tag each, a count line, the static doc-gate stage whenever a `.md` or a
  gate file changes, and a known positive and known negative printed on every run.
- Planted reds, each must fire (`--self-check`, 'reds fired 6 of 6'): RED-1 missed direct importer; RED-2 missed path
  reader; RED-3 missed dynamic reacher; RED-4 an unrelated test selected (or a reader-less changed file selecting
  something); RED-5 a dirty or mismatched tree accepted; RED-6 a planted `conftest.py` change not selecting its directory,
  or a pytest-config change not selecting the full suite.
- `--assert`: the shard run refuses unless its selection list equals the selector's output on that exact (base, head).
- Acceptance, counted before PROVISIONAL is removed: on base 847706a7 and head 105e382e the output is a SUPERSET of the
  151-file hand selection (675ea30c6dea) with every extra file explained by a reason tag; a hand file it omits is a tool
  miss (RED) or a ruled hand over-selection. Exact equality is a happy outcome, not the bar.
- Kill rule: if a planted red does not fire or a hand-selected reader is missed, it is not adopted and the hand
  selection stays, labelled as such; if its selection is rubber-stamp-sized on real units, that is reported and the
  owner rules whether it earns its place.

## Instrument notes from unit 1 (script shas)

- D-443: an acked compose_gate fold is not an installed fold. The D10 revert gate (a90a5698bcc9) and the v2 docs gate
  (ca02b472c8e5) ran under compose_gate 122c458692e1 / compose_row 7b4c189b796e, before the D-432 fold
  (07c9e07932e1 / 28299c23cf11) was installed; reviewer-3 ruled NO-RERUN (#9981, #9986).
- D-445: the first unit-1 shard run (runner c30d607bda03) was VOID by instrument: Start-Process joined the pytest
  marker unquoted, and a shard function returned a log line with its process. A doc-budget red on the same run was
  real (POLICIES.md 16,930 > 16,416). Fixed in the unit-1 shard runner, sha12 7a53b37faa7f,, which carries a mandatory `-Smoke`.
- The docs-only compose set does not run `doc_budget_gate`; a diff touching a standing doc needs it (#9949).
- POLICIES.md had 0 words of headroom at 8576b47f; the binding text's 215 words (187 + a 28-word table row) are paid by a measured ceiling raise.
