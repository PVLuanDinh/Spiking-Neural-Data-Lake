# Shared-file edits requested by `specs/005-language-organ`

This agent implemented `specs/005-language-organ` concurrently with a second agent implementing
`specs/006-fixed-point-gf` in the same working tree. Files that both agents could touch were declared
off-limits to both. Every edit this feature needs in one of those files is written here for the
coordinating agent to apply, rather than applied.

**Verdict context**: the outcome is `NOT-P2-ELIGIBLE` (see [`verdict.md`](./verdict.md)). No source
file under `src/` was created or modified, so **most of the shared edits this feature originally
anticipated are not needed**. Only the two below are.

---

## 1. `docs/coverage/Language.md` — REQUIRED (tasks.md T011, T058, FR-025, SC-013)

**Nothing about the band changes.** The organ was never built, so no row's evidence moved. What is
required is a recorded note that the swap was *measured* and found not P2-eligible, so the next reader
does not re-propose it and does not read the silence as "untried".

Suggested addition, to be placed wherever the file records adjudications rather than inside a scored
row (exact wording is the coordinating agent's call; the content is what matters):

```markdown
### 2026-08-02 -- assembly-calculus language organ: NOT-P2-ELIGIBLE, no band change

`specs/005-language-organ` proposed replacing `cortex/language_acquisition.py` with a brain-native
Assembly-Calculus language organ (arXiv 2306.15364 / 2507.11788), decided by an A/B on
`reasoning/blimp_bench.py`. The swap was refused at the instrument stage, before any organ code was
written: the exposure plan's lexicon covers **0 of 67** BLiMP paradigms and **0 of 67000** pairs
against floors of 8 and 200, because `world_kb_stream.load_kb()` renders proper nouns and chemical
symbols and BLiMP's minimal pairs are built out of function words the KB does not contain (**39 of
2637** BLiMP word types are in the lexicon; the best per-pair coverage reached is 50.0%).

Artifact: `specs/005-language-organ/coverage.md`, `tools/lang_coverage.py` exit 2, commit
`95c3f806f19fe3c1dc704336581ade75541578de`.

**No row changes and no row is deleted.** `generation_acquired_lm` and the other Language rows stand
exactly where they were; the incumbent was not measured on a frozen set, so nothing about it is newly
known. The blocker is the exposure source, not the architecture.
```

**Constraint that must survive whatever wording is used**: a row is never deleted to raise a score
(FR-025), and this addition must not read as a downgrade of any existing row — nothing was
re-measured, so nothing moved.

---

## 2. `run_selfchecks.py` — RECOMMENDED, not required by this verdict (tasks.md T057, FR-012)

`cortex/spiking_lm_serve.py` is adjudicated **independently of the organ's verdict** because it is
dormant either way. The recommendation is **RETIRE**, with the counted evidence in
[`verdict.md`](./verdict.md#t057--cortexspiking_lm_servepy): **0 of 9** references are live runtime
consumers, and there are **0** trained `snn_lm/1` weight files in the tree, so its passing CORE
self-check is a green light that cannot go red.

Retiring it is **not a one-line edit** and must not be done as one. It is the only torch-free reader
of the `snn_lm/1` format, so deleting it alone orphans `training/spiking_lm.py::export_lm` and
`verify_transfer()`. The two permitted resolutions (P3):

- **Retire both ends together** — delete `src/snn_data_lake/cortex/spiking_lm_serve.py`, remove
  `"cortex.spiking_lm_serve"` from `run_selfchecks.py:83`, remove `"spiking_lm_serve"` from
  `src/snn_data_lake/__init__.py:46` and `src/snn_data_lake/substrate/brain_geometry.py:117`, and
  remove or rewrite `training/spiking_lm.py::verify_transfer()` (`:401`-`:412`, the only importer) and
  the `--export`/transfer-check text in `training/cli.py:48`, `:133`. **State the module-count change**
  rather than letting the self-check count shrink silently (SI-2).
- **Exempt with a stated reason** — keep it, and record in `substrate/dormancy_gate.EXEMPT` or the
  coverage ledger why a torch-free reader of an export format with no trained artifacts is worth
  keeping.

**This feature does not choose between them**, because both touch shared files and the second is a
policy call about the training tier that is outside this feature's evidence. What it does provide is
the count, so the choice is made against numbers. Leaving it undecided is the one thing FR-012 forbids.

### Registration edits this feature does NOT need

Listed explicitly so their absence is not mistaken for an oversight:

- **No new module to register.** `cortex/language_organ.py` was never created (T028-T034 did not run),
  so there is no `REGIONS` entry, no `run_selfchecks.py` CORE entry, and no `dormancy_gate.EXEMPT`
  entry to add.
- **No module to deregister.** `cortex/language_acquisition.py` is present and unmodified.
  `run_selfchecks.py:369` stays exactly as it is (P2: deletion is ADOPT-only).
- **No `pyproject.toml` change.** No new dependency; `tools/lang_coverage.py` imports only numpy-free
  stdlib plus the existing `reasoning.blimp_bench` and `reasoning.world_kb_stream`.
- **No `.github/` change.** The two files added (`tools/lang_coverage.py`,
  `tests/test_lang_coverage.py`) are picked up by the existing `pytest` and `ruff` gates without
  configuration.

> **P7 VOID (annotated 2026-08-04).** The sentence above certifies "no new dependency" as a constitution check. **That is not a criterion under P7** (`docs/POLICIES.md`), which states that *"it would be another dependency" is NOT a valid reason to reject anything* and requires a rejection to cite a measurement on the real workload or a structural argument. No library was evaluated here, so the check scored nothing. Left in place rather than deleted because this file records a decision that was made; annotated because P7 names these certifications as live residue of the zero-dependency rule abolished 2026-07-20. `specs/010/plan.md` is the first plan to decline the certification on purpose.

---

## 3. `git add tools/lang_coverage.py` — REQUIRED before `tools/citation_guard.py` goes green

Not a content edit, but it is the one thing standing between this feature and a green T060, and this
agent is forbidden to stage.

`tools/citation_guard.py` currently reports **1 unexempted STRICT violation**:

```
specs/005-language-organ/contracts/lang_ab.md:22 -> tools/lang_coverage.py
```

The guard resolves citations against `git ls-files`, so an existing-but-unstaged file reads as
untracked. `tools/lang_coverage.py` is on disk, runs, and is cited as the reproduction path for
`coverage.md` — which is correct and is the behaviour the guard wants. **Staging the file clears it; do
not add an EXEMPT entry.** An exemption here would be an allowlist covering a file that genuinely
should be tracked, which is the shape `citation_guard.py`'s own docstring says kills gates.

For the record, the guard's output before this pass had **3** such violations, all in
`contracts/lang_ab.md`. Two of them were `tools/lang_organ_ab.py`, which was never built because the
feature stopped, and they were fixed properly rather than exempted: section B of that contract now
carries a NOT BUILT banner and states its interface unfenced, so it no longer presents a nonexistent
file as a runnable reproduction path. `BROAD` scope is report-only and non-asserting, as its docstring
requires.

---

## Files this agent created or modified (none shared)

| path | status |
|---|---|
| `tools/lang_coverage.py` | NEW |
| `tests/test_lang_coverage.py` | NEW, 11 of 11 passing |
| `specs/005-language-organ/coverage.md` | NEW, machine-written, exit 2 |
| `specs/005-language-organ/baseline.txt` | NEW |
| `specs/005-language-organ/verdict.md` | NEW |
| `specs/005-language-organ/shared-edits.md` | NEW (this file) |
| `specs/005-language-organ/research.md` | MODIFIED — R10 added (SNNLP) |
| `specs/005-language-organ/spec.md` | MODIFIED — FR-027..FR-030, SC-014..SC-016 added (SNNLP) |
| `specs/005-language-organ/tasks.md` | MODIFIED — T061-T070 added, checkboxes marked |
| `specs/005-language-organ/contracts/lang_ab.md` | MODIFIED — section C added (SNNLP arms) |
| `docs/DEPTH_CEILINGS.md` | MODIFIED — one line under Ceiling 2 recording the refusal (T059) |

---

## Adjudication by the coordinating agent, 2026-08-02

**1. `docs/coverage/Language.md` -- APPLIED**, in the wording proposed, as a dated adjudication
section. No band changed, no row deleted. `coverage_score --check` still exits 0 and the total still
prints 26.4%.

**3. `git add tools/lang_coverage.py` -- APPLIED.** It and `tests/test_lang_coverage.py` are now
tracked; `citation_guard` is back to `gate OK`. Not exempted, as instructed.

**2. `run_selfchecks.py` / `cortex/spiking_lm_serve.py` retire -- DECLINED, and the recommendation
was wrong on its facts.**

The report gave "0 of 9 references are live consumers". Checked directly, two things contradict it:

- `src/snn_data_lake/training/spiking_lm.py:407-408` imports and calls it:
  `from snn_data_lake.cortex.spiking_lm_serve import generate, load_lm`. That is the trained-weights
  round-trip -- the check that a model trained in the training tier serves identically in the core.
  Not the live brain tick, but a real caller, and retiring the module breaks it.
- `docs/coverage/Language.md:221` already records the port-before-retire blocker: it is the **only
  torch-free `snn_lm/1` reader**, so deleting it orphans that format.

Also worth separating, because the names invite the mistake: `reasoning/ar_spiking_lm_serve.py` is a
DIFFERENT module, is live, and is registered as "the BLiMP lever". Most references matching a loose
`spiking_lm_serve` grep belong to it, not to the cortex module.

Retiring an empty pipe is usually right. Retiring one that still has the only reader for a persisted
format, and a caller that proves train/serve equivalence, is a port-before-retire question and not a
side effect of an unrelated verdict. Left in place, unexempted, unchanged.
