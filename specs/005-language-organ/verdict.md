# Verdict: `specs/005-language-organ`

**Outcome: `NOT-P2-ELIGIBLE`** — recorded at tasks.md **T011**, the hard stop stated before the work
began.

**Commit**: `95c3f806f19fe3c1dc704336581ade75541578de`
**Artifact**: [`coverage.md`](./coverage.md), machine-written by `tools/lang_coverage.py`, exit **2**.

This is a **successful terminal state**, not a failure and not a blocker. `tasks.md` states it in
advance: *"if `tools/lang_coverage.py` exits 2, the outcome is `NOT-P2-ELIGIBLE`, T010 records it, and
the feature stops at T011. It does not proceed to a wider corpus and a re-run."* Three outcomes were
declared successes at authoring time; this is the first of them, and it cost one probe and no source
change to the brain.

---

## The numbers

Measured by `tools/lang_coverage.py` at the pinned commit, over the full BLiMP universe (67 paradigms
x 1000 pairs, no `--limit`):

| quantity | measured | floor (research.md R1) | verdict |
|---|---|---|---|
| paradigms with at least one covered pair | **0 of 67** | >= 8 of 67 | **missed** |
| covered pairs | **0 of 67000** | >= 200 | **missed** |
| BLiMP word types inside the exposure lexicon | **39 of 2637** | — | diagnostic |
| best per-pair word coverage reached by any single pair | **50.0%** | rule requires 100% | diagnostic |
| exposure lexicon size | **508 words** from **605** KB triples | — | — |

The rule, verbatim from the artifact and fixed before any arm was scored (FR-023, A2):

> A pair is IN the restricted set when every word of BOTH its sentences is in the exposure lexicon.
> Membership is a function of the rule and the lexicon ONLY; no arm's score is an input.

The lexicon source, verbatim (A1):

> `reasoning/world_kb_stream.py::load_kb()` — every `(head, relation, tail)` triple rendered into the
> word sequence the organ hears, split on whitespace, `_` and `-`, lowercased, stripped of
> surrounding punctuation. **NOT read off a trained organ.**

**A9, reproducibility, checked rather than claimed**: `tools/lang_coverage.py` was re-run at the same
commit with the same inputs and its output is **byte-identical** to the committed `coverage.md`.

## Why the miss is structural rather than marginal

Not a near miss to be closed by a slightly bigger corpus. `load_kb()` returns 605 real
`(head, relation, tail)` triples whose rendered word inventory is 508 words of **proper nouns, chemical
symbols, taxonomic categories and integers** — `abuja`, `antimony`, `amphibian`, `27`. The renderer
introduces no words of its own: no determiners, no copula, no auxiliaries, no complementisers, no
wh-words.

BLiMP sentences are ordinary English. The most frequent out-of-lexicon tokens in the first paradigm
alphabetically (`adjunct_island`, grammatical member of the first 200 pairs) are `who` (108), `what`
(92), `before` (56), `while` (52), `after` (48), `without` (44), `had` (40), `has` (30), `is` (26),
`was` (25), `to` (24), `about` (22), `the` (20). **The
exposure plan contains no function words at all**, and BLiMP's minimal pairs are built out of them —
that is what makes the pair minimal. The best coverage any single pair reached was **50.0%** of its
words, against a rule requiring 100%.

So the gap is not a scale gap. A KB of triples and a benchmark of grammatical minimal pairs do not
share a vocabulary in kind, and no amount of additional triples of the same kind changes that.

## What was NOT done, by rule

- **The corpus was not widened.** `research.md` R1 names this move: *"Not 'widen the corpus until it
  passes' — that is choosing the pair set to get a verdict, which is the p-hacking the rule exists to
  prevent."* `plan.md` Risk 1 ranks it the highest-damage silent failure in the feature *because it
  would look like diligence*.
- **The 100%-coverage rule was not relaxed.** R1 records a derived coverage fraction as the next
  candidate **if** it is derived, committed and justified before scoring. It cannot be reached for
  here: relaxing it now, after seeing 0, is trying rules until one passes. At 50.0% best-pair
  coverage, no defensible fraction would admit a pair anyway.
- **No source file under `src/` was modified.** `cortex/language_acquisition.py` is present and
  unmodified; the organ was never built, so there is nothing to wire and nothing to delete (P2: the
  incumbent is deleted **only** on ADOPT).
- **No metric was substituted.** The SNNLP sentiment work folded into this feature mid-pass is
  explicitly non-deciding and is blocked behind the same stop — see below.

## Admission controls

`A6` requires the controls to run **on the restricted set, in the same invocation**. The restricted
set is empty, so they are **not computable on it**, and the artifact says exactly that. An empty set
cannot demonstrate that the instrument can go red on it, which is by itself sufficient to refuse a
verdict (A7 path).

Because a record that only said "not computable" would leave a reader unable to tell whether the
machinery works at all, `tools/lang_coverage.py` additionally runs the same three controls over the
full universe and labels the block **"NOT the verdict set, decides nothing"**:

| control | full universe (67000 pairs) | required behaviour on a verdict set |
|---|---|---|
| `random_scorer` | **49.9%** (33425 of 67000) | within 3 points of 50.0 — would have **passed** the check |
| `constant_scorer` | **0.0%** (0 of 67000) | exactly 0.0, ties count as wrong — would have **passed** |
| `length_scorer` | **30.7%** (20591 of 67000) | leakage printed, never thresholded (FR-026) |

The fourth control, the unexposed organ, needs an organ. No organ exists, so it did not run, and that
is stated rather than counted as a pass.

`length_scorer` at 30.7% against `docs/DEPTH_CEILINGS.md`'s recorded **29.4%**: the recorded figure is
25 pairs per paradigm (1675 pairs), this one is all 1000 (67000 pairs). A sample-size difference, not
drift. Recorded so the next reader does not chase it.

## Dispositions that run regardless of the verdict

### T057 — `cortex/spiking_lm_serve.py`

**Disposition: RETIRE (recommended), not executed by this pass.** Consumers **counted, not asserted**,
over `src/`, `tools/`, `tests/`, `run_selfchecks.py` and `learn_daemon.py` at the pin. Note the name
collision that makes this easy to get wrong: `reasoning/ar_spiking_lm_serve.py` is a **different**
module, it *is* live (`blimp_bench.SERVES`), and it is not what is being adjudicated here.

| role | count | who |
|---|---|---|
| references outside its own file | **9 of 10** (the 10th is its own docstring, `spiking_lm_serve.py:22`) | see below |
| of those, real **imports** | **2 of 9** | `training/spiking_lm.py:407` and `:408`, both inside `verify_transfer()` |
| **live runtime consumers** | **0 of 9** | none — no live path builds it; `verify_transfer()` is a CLI transfer check |
| registry / metadata entries | **3 of 9** | `run_selfchecks.py:83` (CORE), `src/snn_data_lake/__init__.py:46` (REGIONS), `substrate/brain_geometry.py:117` |
| docstring-only mentions | **4 of 9** | `training/spiking_lm.py:9`, `:352`, `:400`, and `training/cli.py:48` |
| producers of the `snn_lm/1` format it reads | **1 of 1** | `training/spiking_lm.py::export_lm` |
| trained `snn_lm/1` weight files anywhere in the tree | **0** | — the four `models/*.npz` belong to other serves |

It is the only torch-free reader of `snn_lm/1`, so retiring it alone orphans that export path; the
permitted resolutions are retiring both ends together or **exempt with a stated reason** (P3's third
disposition). The decision touches `run_selfchecks.py`, which this agent is forbidden to edit, so the
recommendation and its count are recorded here and the edit is written to
[`shared-edits.md`](./shared-edits.md) for the coordinating agent. **What is not permitted is leaving
it undecided, and it is not left undecided.**

### T051, T052, T053, T055 — not applicable

Port-before-delete, persistence, the `predictive_coding.py` consumer count and the deletion itself are
all **ADOPT-only**. There is no ADOPT. Nothing was ported, nothing was deleted, and
`cortex/language_acquisition.py` is untouched.

### T060 — `tools/citation_guard.py` and the gates

`citation_guard.py` STRICT scope went **3 unexempted violations -> 1**, all three in
`contracts/lang_ab.md`:

- **2 fixed properly, not exempted**: `tools/lang_organ_ab.py` was cited twice inside fenced command
  blocks and was never built. Section B now carries a NOT BUILT banner and states its interface
  unfenced, so a reader who clones this repo is no longer told to run a file that is not there. Fixing
  the citation is the right disposition; an exemption would have been an allowlist for a real defect.
- **1 remaining**: `tools/lang_coverage.py`. The guard resolves against `git ls-files` and this agent is
  forbidden to stage. **Staging the file clears it** — see [`shared-edits.md`](./shared-edits.md) item 3.
  Do not exempt it.

Gate status at the pin and after, recorded in full in [`baseline.txt`](./baseline.txt): `ruff check` and
`ruff format --check` green, `run_selfchecks.py` **277 of 277** green, `pytest` **one failure** —
`tests/test_selfchecks.py::test_core_selfcheck_in_process[brainstem.arm_hardware]` — which predates
every file this feature added, passes when run alone, and passes as a subprocess under
`run_selfchecks.py`. Another agent modified `brainstem/arm_hardware.py` (+308 lines) and
`substrate/cockpit.py` in this working tree while that pytest run was in flight, and the tree was
clean at the T001 pin, so both edits landed after it. Not this feature's, in a file this feature never
touches, and recorded rather than rounded to green.

## The SNNLP scope addition, and where it stands

A mid-pass scope change added arXiv **2401.17911** (SNNLP) to this feature: its spike encoding as the
`PHON` area's text-to-spike encoder with an A/B against Poisson rate-coding, plus SST-2 sentiment as a
**second, non-deciding** capability metric with a bag-of-words control.

- **SNNLP was considered as a replacement for BLiMP as the arbiter and rejected.** The reasoning is
  recorded in `research.md` **R10** so nobody re-proposes it.
- **BLiMP remained the arbiter and the verdict logic did not move.** The ADOPT/NULL/NOT-P2-ELIGIBLE
  criteria, the frozen pair set, the four admission controls and the char-bigram bar are byte-for-byte
  what they were.
- **The new tasks T061-T070 did not execute.** They all sit downstream of the T011 stop: the encoder
  A/B needs a `PHON` area and the sentiment arm needs an organ to score, and neither exists. They are
  specified with IDs and dependencies in `tasks.md` and are blocked, not skipped.
- **No SST-2 data was downloaded and no sentiment corpus was authored.** The constraint was explicit
  that a self-built substitute corpus is not acceptable; with the feature stopped, downloading a
  corpus for a harness that cannot run would be the additive move.

## What the spec got wrong

Recorded because `specs/003` got three of its expected figures wrong and recording that was the
valuable part.

1. **`research.md` R1 treats the coverage question as genuinely open — "whether the covered subset
   clears the floor is unmeasured" — and sizes a 200-pair floor with a careful paired-difference
   argument.** The measured answer is **0**. The floor derivation was doing arithmetic on a quantity
   that turns out to have no support at all. The derivation is not wrong; it was answering the wrong
   question. The question that decided this feature was *"do the two vocabularies overlap in kind?"*
   and it is answerable in one grep, without any statistics.
2. **The spec's Edge Cases say "BLiMP's vocabulary is far larger than any modest grounded corpus" and
   predict "an organ that returns UNSCORABLE for most pairs".** *Most* understates it. The number is
   **all** — 0 of 67000 — and the reason is not size but **kind**: the KB has no function words, and
   minimal pairs are made of function words.
3. **Assumption 3 and Q1 both assume a covered subset exists and only its size is in doubt.** Q1's
   assumption reads *"yes, on a lexically-covered subset"*. The correct answer to Q1 is **no**, for
   this exposure source. Q1 was nonetheless the right question to ask first, and asking it first is
   what made the feature cost one probe.
4. **`tasks.md` T011 instructs the agent to update `docs/coverage/Language.md`.** That file is shared
   state a concurrently running agent also reads, so the edit is written to `shared-edits.md` instead
   of applied. The task should have named the shared-file boundary; it did not.

## What would make this feature P2-eligible later

Not proposed as a next step, only recorded so the finding is actionable rather than merely negative.
The blocker is the **exposure source**, not the architecture and not the instrument:

- A grounded exposure source whose rendered sentences are ordinary English — the `specs/004` arm/MuJoCo
  environment is named in `research.md` R6 as the strongest follow-on, and a physically grounded verb
  comes with the function words a sentence needs.
- Or a different instrument, which P2 permits (*"build the benchmark first"*) and `research.md` R1
  argues against for this feature, since a benchmark authored alongside the mechanism it scores is the
  weakest evidence available.

Either is a **separate decision taken before a restriction is computed**, never in response to this
exit. That distinction is the whole content of R1.
