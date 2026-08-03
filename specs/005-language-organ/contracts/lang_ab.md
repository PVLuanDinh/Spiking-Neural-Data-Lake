# Contract: `tools/lang_coverage.py` and `tools/lang_organ_ab.py`

**Feature**: `specs/005-language-organ` | **Spec**: [../spec.md](../spec.md) | **Phase 0**:
[../research.md](../research.md)

Two tracked probes. Together they are the whole evidentiary apparatus of this feature: one freezes the
pair set, the other issues the verdict. Both live in `tools/` and are tracked - `scratchpad/` is
gitignored and `tools/citation_guard.py` gates a tracked file citing an untracked `.py` as a runnable
reproduction path.

---

# A. `tools/lang_coverage.py`

Computes the lexically-covered BLiMP subset from the **exposure plan**, freezes it, and runs the
admission controls on it. Runs **before** any organ exists (R9), which is the only order in which
"fixed before either arm is scored" (FR-023) is achievable.

## Usage

```
python tools/lang_coverage.py [--blimp DIR] [--out specs/005-language-organ/coverage.md]
```

## Contract obligations

- **A1** The lexicon is computed from the **exposure plan** - the word inventory of
  `reasoning/world_kb_stream.py::load_kb()` rendered into sentences - **not** from a trained organ. A
  lexicon read off a trained organ is a lexicon that could have been tuned.
- **A2** The rule is stated in the output verbatim: a pair is IN when every word of both its sentences is
  in the exposure lexicon. The rule is a property of the pair set and the exposure; **no arm's score is
  an input**.
- **A3** Reports `P of 67` paradigms and `N of M` pairs. Never `N/M` - the artifact fingerprint harvests
  bare ratios positionally, and adding one changes every artifact's fingerprint.
- **A4** **Floors, checked and enforced by exit code:**
  - `>= 8 of 67` paradigms with at least one covered pair
  - `>= 200` covered pairs
  Missing either exits **2** with `NOT-P2-ELIGIBLE` and the reason. Exit 2 is not a failure of the tool;
  it is the tool doing its job.
- **A5** Emits the frozen `pair_ids` list to `coverage.md`, which is **committed before any scoring**.
  Both arms later score exactly this list - not "the same rule applied separately", the same list.
- **A6** Runs the four admission controls on the restricted set, in this invocation:
  `blimp_bench.random_scorer` (must land within 3 points of 50%), `constant_scorer` (must score 0%),
  `length_scorer` (leakage printed, not thresholded - its value IS the size of the leak, and a threshold
  would hide a growing one). The fourth, the unexposed-organ control, runs in `lang_organ_ab.py` because
  it needs an organ.
- **A7** If `random_scorer` misses 50% by more than 3 points or `constant_scorer` scores above 0 on the
  restricted set, the restriction has introduced a leak: exit **2**, set disqualified. This is the check
  that stops a subset from becoming an accidental green light.
- **A8** Absent BLiMP data prints `SKIP lang_coverage: ...` and exits **0**, matching `blimp_bench`'s
  behaviour - and the skip is recorded such that it can never be read downstream as a passed A/B.
- **A9** Re-running at the same commit with the same inputs reproduces the same `pair_ids` byte for byte.
- **A10** Isolates `NORD_HOME` to a per-invocation temp directory. The always-on daemon writes
  `~/.snn_data_lake` continuously and ten of the ledger's artifacts were once irreproducible for exactly
  this reason.

## Exit codes

| code | meaning |
|---|---|
| 0 | set frozen and admissible, or a clean SKIP for absent data |
| 2 | `NOT-P2-ELIGIBLE` - a floor missed, or a control passed where it must fail |
| 1 | tool error |

---

# B. `tools/lang_organ_ab.py` — SPECIFIED, NEVER BUILT

> **This file does not exist and no command below is runnable.** The feature stopped at the T011 hard
> stop with `NOT-P2-ELIGIBLE` (see [`../verdict.md`](../verdict.md)), so the verdict harness was never
> written. Section B is kept as the interface a future attempt must satisfy, **not** as a reproduction
> path. The command lines are therefore given inline and unfenced, so `tools/citation_guard.py` is not
> told that a reader who clones this repo can run a file that is not there — which is the exact defect
> that guard exists to catch, and it would be a poor look for this document to commit it.

The four arms, the head-to-head, and the verdict.

## Interface (as specified; nothing implements it)

Invocation would be `tools/lang_organ_ab.py` with optional
`--coverage specs/005-language-organ/coverage.md`, `--seeds N` and `--arms ...`.

Recording would go through the ledger, which is how the band ceiling comes from an exit code rather
than a claim: `tools/coverage_score.py record --kind ab --name lang_organ_ab -- <the harness>`.

## The arms

| arm | construction | expectation |
|---|---|---|
| `joint` | the shipped organ | the feature |
| `staged` | semantics to convergence, then syntax | worse than joint (2406.11977) |
| `shuffled` | scenes permuted against sentences; same items, order, split, seed | at the frequency baseline |
| `frozen` | identical construction, plasticity never applied | at the frequency baseline |
| `incumbent` | `cortex/language_acquisition.py` as it stands | **unknown - measured at T014, never quoted.** The 51.0% figure is withdrawn (`docs/DEPTH_CEILINGS.md:555`); no current trustworthy figure exists for this module specifically (research.md R2a) |
| `bigram` | char bigram fitted on identical exposure text | 52.8% on the full set; **beats the incumbent on the last comparison where both were measured together** |

## Contract obligations

- **B1** Every arm scores **exactly** the frozen `pair_ids`. The harness asserts the list it loaded
  matches the one in `coverage.md`, and refuses otherwise.
- **B2** Both scorings reported side by side for every arm - unnormalized sum-logprob and per-character -
  exactly as `blimp_bench.main()` already does. Reporting one silently makes every previously recorded
  number incomparable.
- **B3** All four admission controls print **inline with the headline numbers, in this invocation**. Not
  a separate run whose result could be from a different subset (FR-002).
- **B4** Scoring is read-only in every arm. The harness snapshots organ weights before and after the
  scoring pass and asserts equality; a mismatch is a hard failure, because an arm that learned during
  evaluation was taught its own test.
- **B5** An arm that cannot score a sentence returns `None`; `blimp_bench` sentinels it and the pair ties;
  ties count as wrong. Credit is never awarded for having no opinion (FR-014).
- **B6** **ADOPT requires all of:**
  1. `joint` > `incumbent` under **both** scorings, and
  2. `joint` > `bigram` under **both** scorings, and
  3. all four admission controls failing as specified - **including `length_scorer`, named explicitly**
     (FR-026): it must be printed with the headline, and a `joint` advantage present under sum-logprob
     but absent under per-character normalization is a **length-bias correction, not a language win**,
     and is reported as one. About six points are available from fixing that bias alone - more than the
     entire 2.6-point per-char competence band (research.md R2a) - so this is the most likely way this
     harness manufactures a fake capability, and
  4. `frozen` and `shuffled` at the frequency baseline.
  Any one unmet is **NULL**. Criterion 1 compares against T014's **measured** incumbent number on the
  frozen set; no quoted figure - 51.0% or any other - may stand in for it.
- **B7** `NULL` exits **0** with `NULL` in the summary. **A null is a successful, terminal outcome**
  (FR-005). It does not exit nonzero, because a null is not an error and a nonzero exit invites someone
  to "fix" it.
- **B8** The harness has **no parameter that changes the verdict**. No threshold flag, no tolerance, no
  "retry with". `--seeds` widens the sample and `--arms` selects which to print; neither can turn a NULL
  into an ADOPT.
- **B9** The joint-versus-staged contrast is printed with direction and magnitude **whichever way it
  falls**. If `staged` matches `joint`, the summary says the joint claim is not supported (R4).
- **B10** Prints the held-out role accuracy against the most-frequent-role baseline the same harness
  computes, both numbers, as `N of M` correct.
- **B11** Prints the density report: `H`, the derived `M`, `rho_floor = H_b^{-1}(log2 M / H)`,
  `rho_measured`, and the plain statement that compute is dense regardless of activity sparsity, and that
  the low floor follows from the organ's small `M` - i.e. from it doing less than a language model
  (FR-021, R7). `rho_measured < rho_floor` is reported as an **anomaly**, not a success.
- **B12** Prints `Brain.respond` latency for both arms in the same units, and the delta, whichever way it
  falls (FR-022). A regression appears in the summary, not only an appendix.
- **B13** Every `Brain` is constructed through `make_brain()` (P4). Isolates `NORD_HOME`.
- **B14** ASCII only; counts as `N of M`.
- **B15** The recorded command is `--rerun`-able. `coverage_score rerun` does `a["command"].split()`, so
  a `python -c` one-liner records fine and dies on rerun - the forgery the exit-code mechanism exists to
  catch, dressed as compliance. The command is a plain script invocation.
- **B16** Deterministic at a fixed seed; `--seeds N` reports mean and spread across seeds, and the spread
  is printed so a one-draw result is not mistaken for a stable one. `docs/coverage` already records
  thirteen claims that stood on one draw.

## Exit codes

| code | meaning | band consequence |
|---|---|---|
| 0 | verdict issued - `ADOPT` or `NULL`, stated in the summary | the recorded exit code is what a band > 40 claim reads |
| 2 | not eligible - frozen set missing, floors unmet, or an admission control passed where it must fail | no band claim admissible |
| 1 | tool error | no band claim admissible |

**Exit 0 does not mean ADOPT.** It means a verdict was legitimately produced. Anything reading this
artifact must read the summary line, not merely the code, and the summary line's first token is the
outcome.

---

# C. The SNNLP additions (arXiv 2401.17911)

Added 2026-08-02 by owner scope change. Reasoning in [../research.md](../research.md) R10; requirements
FR-027 to FR-030.

**C0 - the constraint that makes this safe, and it is not optional.** The BLiMP verdict is computed
from the BLiMP block **alone**. The `PHON` encoder A/B and the sentiment block are printed **after** the
verdict line and are structurally incapable of reaching it. **If BLiMP returns `NULL`, the verdict is
`NULL`, whatever sentiment says.** A test forces a `NULL` BLiMP block alongside a maximal sentiment
block and asserts the verdict is still `NULL` (SC-015). Prose promising this is not enough - B8 already
forbids any parameter that changes the verdict, and this is the same guarantee for a second input.

## C1. The `PHON` encoder A/B

Four arms, identical exposure, order, split and seed, all feeding an otherwise identical `PHON`:

| arm | what it is | why it is here |
|---|---|---|
| `snnlp` | SNNLP's proposed text-to-spike encoding | the proposal |
| `poisson` | Poisson rate-coding | **the paper's own comparison**, which is what makes this a P2 adopt question and not a preference |
| `graphemefold` | `cortex/language_acquisition.py::_encode_ctx`'s position-tagged grapheme bag | what this repo does today. Omitting the incumbent would make "SNNLP is better" a comparison against nothing this tree uses |
| `rbfpop` | `cortex/population_encoding.py::PopulationEncoder` | the repo's existing population code. Note it encodes a **scalar**, so this arm must state which scalar embedding of the word form it used, because that choice is then doing part of the work |

- **C1.1** The winner is whichever measured best on the organ's own task. **An existing encoding
  winning is a legitimate and expected outcome**, is adopted, and SNNLP is then recorded as
  measured-and-not-adopted (FR-028).
- **C1.2** The ~13% SNNLP-over-Poisson figure and the 32x/60x energy figures are **CITED, UNVERIFIED**
  (research.md R10). No criterion may reference them. The A/B's own number is the only admissible one.
- **C1.3** No energy or sparsity figure is an acceptance criterion here, exactly as FR-021 already
  requires of the organ. This repo has no hardware to measure them on, and quoting the paper's would be
  the decorative-citation pattern R7 refuses.
- **C1.4** `theta_gamma_code.py` is **not** an arm. It encodes multi-item ordering within a cycle and is
  a `SEQ` candidate, not a text-to-spike encoder. Named so it is not mistaken for one.

## C2. The sentiment block

- **C2.1** Corpus is **SST-2**. The one-time download command lives in the loader's module docstring and
  the data is **not committed** - `data/` is gitignored (`.gitignore:43`) and `reasoning/blimp_bench.py`
  sets this precedent exactly. Follow it; do not invent a second data convention.
- **C2.2** The loader lives beside the organ, in `cortex/`. Not `reasoning/`.
- **C2.3** **A bag-of-words control prints inline with the sentiment number.** Sentiment is largely
  lexically solvable, so a bag of words scores well while knowing no grammar at all. If the organ does
  not beat it, the output must say **in the same block** that the sentiment number is not evidence of
  anything - not omit the control and not drop the number.
- **C2.4** The sentiment block never appears above the verdict line, and no sentiment quantity is an
  input to any ADOPT criterion (C0).
- **C2.5** Absent SST-2 prints a `SKIP` with a stated reason and exits 0, and the skip is recorded such
  that it can never be read downstream as a passed capability - the same treatment A8 gives absent
  BLiMP data.
- **C2.6** **If the download is unavailable or the licence is unclear, STOP and report.** A
  self-authored substitute corpus is forbidden by name: a corpus authored alongside the mechanism it
  scores is the weakest evidence available, and research.md refuses it twice already (R1, R6).

## C3. What this section does NOT change

Stated because the value of the whole apparatus is that it did not move: the frozen `pair_ids`, the
restriction rule, the four admission controls, the char-bigram bar, the ADOPT conjunction in B6, the
`NULL`-exits-0 rule in B7, and the no-verdict-changing-parameter rule in B8 are all **byte-for-byte
unchanged** by the SNNLP addition.
