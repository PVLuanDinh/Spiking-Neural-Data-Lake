# The two hard depth ceilings — measured, not asserted

The v3.78 release notes named two ceilings honestly but without numbers:

> Two depth ceilings stand: (1) fluent open-domain language absent (ROSE = compositional structure +
> grounded semantics, NOT GPT-level generation); (2) general fluid intelligence still ~absent
> (no test-time program synthesis for novel out-of-grammar problems).

An unmeasured ceiling is an opinion. This document replaces both with numbers from **real, external,
public benchmarks** — chosen because they cannot be gamed by a template bank, a taught-triple traversal,
or a fluent-sounding demo, and because they come with published human and machine anchors.

Both statements turned out to be **wrong in their diagnosis**, in opposite directions: the Gf ceiling was
described as an absence that is in fact present-but-domain-blind, and the language ceiling was described as
a matter of eloquence when it is a total absence of measurable grammatical competence.

## The instruments

Two harnesses make the ceilings permanently measurable. Both are registered RESEARCH-tier self-checks and
both SKIP-with-notice when their dataset is absent, so the zero-dep CI matrix stays green either way.

| harness | file | benchmark |
|---|---|---|
| `arc_bench` | `src/snn_data_lake/reasoning/arc_bench.py` | ARC-AGI-1, 400 train + 400 eval tasks, exact match, pass@2 |
| `blimp_bench` | `src/snn_data_lake/reasoning/blimp_bench.py` | BLiMP, 67 paradigms × 1000 minimal pairs |

Both ship controls that must **genuinely fail**, so a real score is believable:

```
arc_bench   : identity 0/200 = 0.0% | constant 1/200 = 0.5% | oracle 50/50 = 100.0%   -> OK
blimp_bench : random 49.7% | constant 0.0% | length 30.4% | oracle 100.0%             -> OK
```

The ARC trivial-solver noise floor is therefore **0.5%**, and BLiMP chance is **50.0%**. Sentence length alone
scores *below* chance on BLiMP (30.4%), confirming the pairs are genuinely minimal — no surface leak.

---

## Ceiling 1 — general fluid intelligence

**Benchmark.** ARC-AGI-1 (Chollet, *On the Measure of Intelligence*, 2019). Every task needs a different
solution program, and that program is in no taught grammar, so it must be synthesised at test time. This is
the structural bar for Gf credit: developer-aware generalisation, zero gradient descent at test time.

### Before

`tools/arc_baseline.py`, evaluation split, first 100 tasks, pass@2:

| interface | score |
|---|---|
| `gf-direct` — gf.py's inducer, hand-fed pre-parsed grids (deliberately generous) | **0/100 = 0.0%** |
| `reasoning-cortex` — `build_default_cortex` routed directly | **0/100 = 0.0%** |
| `live-brain` — `Brain.respond`, as a user would ask | **0/100 = 0.0%** |

Zero through every interface, including one rigged in the brain's favour. Against the 0.5% trivial floor,
the brain was **at or below noise**.

### The release notes were wrong about why

They said there is "no test-time program synthesis." False. `src/snn_data_lake/reasoning/gf.py` contains
real test-time program synthesis — `induce_program` (iterative-deepening enumeration returning the smallest
program consistent with the observations), `OpenGf` (self-extending, with library learning), and
`consistent_hypotheses` / `ActiveLearner` (hypothesis search that breaks underdetermination by experiment).
It is also **already wired live**, as `NumberSeriesExpert`, `LetterExpert` and `RavenExpert` in
`reasoning/reasoning_cortex.py`.

The real defect is narrower: that machinery synthesises programs over **1-D numeric and letter sequences**.
ARC is about **2-D grids and the objects inside them**, and the brain had no perception of that domain — so
there was nothing for the inducer to induce over. A working synthesis engine, blind to the domain.

### After — `reasoning/arc_solver.py`

Object perception (connected components with colour/bbox/size/symmetry/uniqueness attributes, reusing
`arc_agi/rule_inducer.py`'s `parse_objects`) + a 13-primitive grid-transformation DSL + iterative-deepening
search for the **smallest program consistent with every training pair** (MDL bias, following `gf.py`'s own
idiom) + a cross-task library. Wired live as `ArcExpert` in `reasoning_cortex.py`.

| split | score |
|---|---|
| training (developed against) | **40/400 = 10.00%** |
| **evaluation (held out, never inspected)** | **11/400 = 2.75%** |

Controls that genuinely fail: identity-only DSL scores 0/3 on synthetic tasks and **0/40 on training**.
Determinism verified (two budget-bound runs solve the identical set).

**Library learning was an exact null.** Training split with the library ON and OFF produced **identical**
scores and **identical solved-ID sets** (40/400 both ways), despite the library learning 6 composites. It
demonstrably helps in a constructed depth-1 case in the self-check, and demonstrably does nothing at scale.
Reported as measured, not quietly dropped.

### What is honest to claim

The measured move is **0.0% → 2.75%** on held-out ARC-AGI-1, roughly 5× the trivial floor. Human performance
is ~85%. What was built is **test-time program synthesis over an author-chosen grid vocabulary**: it solves
the geometry / colour-mapping / object-selection slice and returns an honest empty candidate list everywhere
else. The DSL is a developer-supplied prior, and under Chollet's developer-aware clause that prior is charged
against the score. **This is not general fluid intelligence, and the gf.py wall is unmoved.**

An 8-design adversarial panel (8 independent designs, 3 refutation lenses each) reached the same verdict
independently and more strongly: **7 of 8 designs were refuted**, and the single survivor states in its own
thesis that the Gf ceiling *cannot be closed, only moved* — because every published RPM/analogy success,
spiking or not, assumes a hand-specified attribute grammar, which the developer-aware clause charges against
the score. No design in the surveyed space earns general Gf credit.

### Addendum (2026-08-03) -- fixed-point depth NULLED, typed mint intake ADOPTED (specs/006-fixed-point-gf)

The next architectural hypothesis after this measurement was tested and settled: **variable-depth
computation via a fixed-point combinator** (iterate one pooled primitive to exact-equality
convergence, classify every trajectory converged / cycle / diverged / budget -- no iteration cap,
no tolerance constant), plus the standing P2 payment of merging `arc_mint`'s verified primitives
into the one searcher through a **typed record bottleneck** (unit, attribute, rule form, rule
params, consumer, memo-flag -- nothing else crosses).

Three-arm record at identical budgets (node_budget=4000, time_budget=5.0, pass@2; full logs and
per-ID accounts in `specs/006-fixed-point-gf/`):

| arm | evaluation | training |
|---|---|---|
| baseline at `95c3f806` (re-measured; the documented 11/40 had drifted) | 12 of 400 = 3.00% | 43 of 400 = 10.75% |
| fix combinator alone (census run) | 12 of 400 -- byte-identical ID set | -- |
| fixed-K control (K = mean winning-fix depth, C18) | SKIP: K underivable -- 0 of 0 winning fix trajectories | -- |
| **final: mint intake adopted, combinator retired** | **18 of 400 = 4.50%** | **53 of 400 = 13.25%** |

**The combinator is an exact null with a cost, reported as measured.** The trajectory census
(500918 of 500918 classified, zero non-termination) shows genuine multi-step contractions exist
(5699 converged trajectories at depths 3..19) -- but not one winning program anywhere used a fix
step, and the trajectories' node cost starved 2 of 400 training solves (both recovered on
retirement). Same framing as the library-learning null above: the tasks this DSL misses lack
VOCABULARY, not depth -- `arc_diagnostic`'s coverage verdict, independently reconfirmed. Per the
adopt-or-retire policy the combinator was REMOVED, not flag-parked.

**The mint merge earned its adoption.** All 16 gained IDs are depth-1 minted records through the
one searcher; zero solved IDs lost on either split in the final state. Honesty qualifiers: 3 of
the 6 evaluation gains (00576224, 00dbd492, 0a2355a6) sit in the mint's 20-task dev subset of
this split and are charged as developed-on; the 3 clean gains include `84f2aca1` -- recoloured
regions by an induced AFFINE rule on size, the first FUNCTIONAL (non-memorised) held-out ARC
solve this project has produced. Memo-flagged wins print their flag in the program description.
Exactly 1 of 1 end-to-end ARC solve paths remains (`synthesize()`; the mint's direct application
path is deleted).

Reproduce: `tools/arc_score.py evaluation on|off` and `training on|off` at the current HEAD for
the final figures; `tools/arc_census.py` and `tools/arc_fixdepth_control.py` document the
combinator measurement and reproduce it at the feature-measurement commit (both SKIP-print the
record at HEAD -- a retired mechanism yields no fresh figure).

---

## Ceiling 2 — fluent open-domain language

**Benchmark.** BLiMP (Warstadt et al., TACL 2020). 67 paradigms × 1000 minimal pairs differing in exactly one
grammatical feature. Automatic, needs **no LLM judge** — it cannot be passed by sounding fluent, nor by a
template bank, because both sentences share nearly every word.

Anchors: chance **50.0%**, **word-level** 5-gram (83M words) **61.2%**, GPT-2 large **86.0%**, human **88.6%**.
For the **character**-level models in this document the right yardstick is not 61.2% but a **char** 5-gram
on this repo's own exposure: **53.0%** at held-out char PPL **4.56**. See *The language number was two
defects, not one* below — and read every BLiMP number here as an **unnormalized sum-logprob** number.

### Before

`tools/blimp_baseline.py`, 67 paradigms × 25 pairs = 1675. Scoring is strictly read-only — no weight
update during evaluation, or the benchmark would teach the model its own test.

| scorer | score |
|---|---|
| char-bigram fitted on the same exposure text | **52.8%** |
| acquired-LM — `cortex/language_acquisition.py`, the online locally-plastic learner | **51.0%** |
| ROSE `comprehend` — `reasoning/rose_language.py` | **no signal** |

Three findings, all worse than the release-note framing:

1. The language faculty is **at chance**. 51.0% against a 50.0% floor is not a weak grasp of syntax.
2. It is **beaten by a character bigram table** on identical text.
3. **Its entire lifetime language exposure was 2,428 characters.**

**ROSE.** `comprehend` returns `{'known': False, 'reason': 'ungrounded'}` for *every* ordinary English
sentence tested, including grammatical ones like "the cat sat on the mat". It emits no grammaticality signal
and cannot separate fluent English from word salad. This is not a failure against its own spec — ROSE is a
constituency-and-grounding mechanism over a small world KB and says so. But it sharpens the honest claim.

**The core's "spiking LM" is an empty pipe.** `cortex/spiking_lm_serve.py` is registered CORE and its
self-check passes, but there is **no trained weight file anywhere in the repo**: it serves a hand-built
4-token demo model (`vocab=4, d_model=4, 1 block`) that generates `'abbbbbb'`. `run_selfchecks.py` records
that SpikeGPT was scrapped in v3.77. There is no trained language model in this project.

### The mechanism, established by measurement

Corpus scaled from 2,428 characters to **10,000,000 characters** of WikiText-103 (4,119×):

```
  chars_seen |  acq-LM |  bigram |  5-gram |   acq PPL |   2g PPL |   5g PPL
        1000 |    54.8 |    51.2 |    47.9 |     17.77 |    15.97 |    30.12
       10000 |    55.5 |    52.2 |    49.9 |     17.67 |    12.25 |    20.14
      100000 |    53.5 |    53.2 |    51.3 |     15.62 |    10.93 |     9.56
     1000000 |    53.4 |    53.0 |    52.7 |     16.21 |    10.71 |     5.31
    10000000 |    53.2 |    53.5 |    53.4 |     15.81 |    10.69 |     4.38
```

Four candidate causes were **falsified** by ablation:

| candidate | verdict | evidence |
|---|---|---|
| homeostatic decay erases learning | falsified | removing `W *= (1-1e-5)` entirely: 53.4% → 53.1% |
| vocabulary folding destroys the distinctions | falsified | **0/1675** pairs collapse; 100% stay separable |
| `ctx=8` too short | falsified **and reversed** | ctx 2→32 gives BLiMP 57.3→53.4, PPL 11.91→18.69 |
| features are lossy | falsified | 0.65% collision over 117,859 contexts; each position 72–84% decodable |

What is left, and it is confirmed twice over:

- **The readout is linear over a summed bag.** `W` is 34×88 = **2,992 parameters regardless of corpus size**.
  A linear map on a superposition can only learn an additive per-position vote; next-character prediction
  needs conjunctions, which are not linearly separable from a summed bag. This is why *shorter* context wins.
- **The learning rule is not the cap.** Fitting the same architecture full-batch to convergence with no decay
  — the convex optimum of the shipped design — reaches **57.0%** BLiMP. The online local rule reaches 53.4%.
  It underfits its own architecture by ~3 points; the architecture is the wall.
- **Naive width expansion is refuted.** A random sparse expansion to 8192 dims (278,528 synapses) made it
  *worse* (BLiMP 52.3–53.2%, PPL 26.75–30.92).

### The finding that reframes the whole ceiling

**Prediction quality and grammaticality are decoupled here.** Better language modelling does not buy BLiMP.

A count-based character n-gram swept from bigram to 11-gram (up to 5.2M nonzero parameters) on 10M characters:

```
   n-gram   nonzero params   BLiMP%    char PPL
        2              838     52.3      10.692
        5          256,688     53.4       4.379      <- best PPL region
        8        2,228,645     54.1       4.835
       11        5,208,719     53.1       7.026
```

Character perplexity improves 2.6× and BLiMP moves inside a ~2-point band. The best character model in the
entire arc reaches **54.7%**.

Adding capacity and nonlinearity to the readout, under the same online local rule, does not help either:

| model | learned params | BLiMP | held-out PPL |
|---|---|---|---|
| shipped 88-dim linear, online | 2,992 | 53.4% | 16.21 |
| shipped 88-dim linear, **convex optimum** | 2,992 | **56.1%** | 13.82 |
| SLDA-RFF over the spiking PC latent (the repo's own nonlinear readout) | 1,176,784 | 55.9% | 13.90 |
| conjunctive features, ctx≤7 wide | 10,289,692 | 52.3% | **4.69** |

A **3,400× larger** conjunctive readout improves perplexity 3.5× and scores **worse** on BLiMP than the tiny
linear one. Across 12 variants, correlation(held-out PPL, BLiMP) = **+0.452** — a *positive* r, meaning lower
perplexity goes with *lower* BLiMP.

Changing the unit from characters to words does not rescue it either. On 15,523,489 word tokens
(194,760 types, 30k vocab), 12 configurations up to 4-grams with 13M parameters:

```
  best by held-out PPL: word 3-gram, Witten-Bell
    -> BLiMP 52.5% mean | 50.6% on the clean fully-in-vocab subset (n=694 pairs) | word PPL 569.4
```

The clean-subset number is **chance**. (Reported with its confound: BLiMP has an 11.16% OOV rate against this
vocabulary, which is why the in-vocab subset is quoted separately rather than the headline being taken at
face value.)

**So the readout was never the cap, and neither was the corpus, the context window, the vocabulary fold, or
the learning rule.** Every one of those was tested and eliminated by ablation.

### Why — the mechanism, not just the correlation

A per-paradigm trace explains the flat ~50%. The word 3-gram is **not uniformly weak; it is bimodal**:

```
  principle_A_domain_1              100%      wh_vs_that_with_gap_long_distance     4%
  principle_A_case_1                100%      coordinate_structure_constraint       8%
  wh_questions_subject_gap           96%      matrix_question_npi                   8%
```

The token trace shows the cause: **BLiMP's hard pairs are reorderings of the same words.** The ungrammatical
"What *was* Alicia approaching senators" contains the frequent bigram `what was` (−2.5); the grammatical
"What *senators* was Alicia..." contains the rare `what senators` (−11.9). So an n-gram-class model is not
guessing on these items — it is **systematically wrong**, confidently preferring the ungrammatical sentence.
The two modes cancel to chance. Capacity, context and data all move perplexity without touching the
cancellation, because none of them supply long-distance structure.

That also explains the otherwise baffling inversion: the aliased 88-dim bag scores *higher* on BLiMP
(55.9–56.1%) than models that predict text far better (52–54%). It is a worse predictor whose errors happen
to be less anti-correlated with grammaticality.

Nothing in this family — count-based or locally learned, character or word, 3K or 130M parameters — clears
~57%, while the published word 5-gram anchor is 61.2% and GPT-2 large is 86.0%. Closing that gap requires a
representation of **long-distance dependency**, not more capacity. (Honest caveat: the 61.2% anchor used
83M words, 5.4× the largest clean exposure reached here, so the word-level arm is not a like-for-like
refutation of that number.)

### Integrity notes on these measurements

Four bugs were found and fixed during this study, three of which had produced invalid numbers: a missing
unigram level in the backoff chain (harmless at V=34, drove BLiMP *below chance* at V=30k); fixed add-alpha
collapsing on singleton word contexts (held-out PPL *rose* with context); UNK acting as a free ride that
rewarded OOV-heavy, often ungrammatical sentences; and **held-out contamination above 1.25e8 exposure
characters** — verified directly by probing, with the affected run retained on disk as
`scratchpad/lang_word_big_out.VOID.txt` and a `assert_heldout_disjoint` gate added so it cannot recur.
The contaminated numbers are not used anywhere in this document.

### What is honest to claim

- The language ceiling is **not** "structure without eloquence". There is **no measurable grammatical
  competence** anywhere in the stack, no trained language model, and the acquired learner's own architecture
  provably caps at 57.0%.
- **Corpus scale is not the lever** — proven, not assumed: 4,119× more real text moved BLiMP by −1.6 points,
  and at the convex optimum train PPL (13.650) and held-out PPL (13.822) are equal, so the model is
  capacity-limited, not data-limited.
- **Capacity is not the lever either** — 3,400× more parameters scored *worse*.
- **The unit is not the lever at this scale** — word n-grams over 15.5M tokens sit at chance on the clean
  in-vocab subset, and a plastic local-rule readout with **129,584,388 parameters** at matched exposure
  scores 50.8–53.9%, i.e. it merely matches the count model.
- **The failure is systematic, not noisy** — on the hard paradigms these models confidently prefer the
  *ungrammatical* sentence, because the pairs are word reorderings and the ungrammatical order contains the
  commoner n-gram.
- The acquired-LM's BLiMP number **should not be cited as evidence of language ability** anywhere.
- All four language designs put to the adversarial panel were **refuted**, several on measured wall-clock
  grounds (one encoder path benchmarked at 96 hours; one serving architecture at 25 days for a full sweep).

### 2026-08-02 — an assembly-calculus organ was proposed and refused at the instrument stage. **There is no "After".**

`specs/005-language-organ` proposed replacing `cortex/language_acquisition.py` with a brain-native
Assembly-Calculus language organ (arXiv 2306.15364 / 2507.11788), to be decided by an A/B on
`reasoning/blimp_bench.py`. **No "After" row is added to this ceiling, because there is no after**: the
swap was refused before any organ code was written, so nothing was measured against the figures above
and none of them moved.

The refusal is a lexical-coverage one and it is structural, not marginal. The proposed grounded exposure
source is `reasoning/world_kb_stream.py::load_kb()`, whose 605 triples render to a 508-word inventory of
proper nouns, chemical symbols, taxonomic categories and integers. Against a rule requiring every word
of both sentences of a pair to be in that inventory:

| quantity | measured | floor |
|---|---|---|
| paradigms with at least one covered pair | **0 of 67** | >= 8 of 67 |
| covered pairs | **0 of 67000** | >= 200 |
| BLiMP word types inside the exposure lexicon | **39 of 2637** | — |
| best per-pair word coverage reached | **50.0%** | rule requires 100% |

The exposure plan contains **no function words at all**, and BLiMP's minimal pairs are built out of
them — that is what makes a pair minimal. So the gap is one of kind, not of scale, and no quantity of
further triples of the same kind closes it.

Artifact: `specs/005-language-organ/coverage.md`, `tools/lang_coverage.py` exit **2**
(`NOT-P2-ELIGIBLE`), commit `95c3f806f19fe3c1dc704336581ade75541578de`. Full record and the three
things the spec got wrong: `specs/005-language-organ/verdict.md`.

One incidental figure from the same run, recorded so it is not mistaken for drift: `length_scorer` over
the **full** 67000 pairs scores **30.7%**, against the **29.4%** recorded above at 25 pairs per
paradigm (1675 pairs). Sample size, not disagreement.

---

## Bottom line

Both ceilings **still stand**. One moved measurably, one did not move at all:

| ceiling | before | after | status |
|---|---|---|---|
| fluid intelligence | ARC-AGI eval **0.0%** | ARC-AGI eval **2.75%** | moved off the floor; not closed |
| open-domain language | BLiMP **51.0%** (chance 50.0) | unchanged | not moved; cause now proven architectural |

The honest gain is not the 2.75%. It is that both ceilings are now **quantified against external benchmarks
with failing controls**, both have a **mechanism identified by ablation rather than by assertion**, and two
comfortable stories the project was telling itself have been retired: that Gf lacked a synthesis engine (it
has one), and that language merely lacked eloquence (it lacks grammar, and more data will not supply it).

Neither number is evidence about consciousness in either direction, and nothing here settles that question.

---

## Wave-1 depth addition — spiking RAVEN (Gf) + an order-sensitive AR language serve

The next wave attacked each ceiling at the mechanism this document had just identified. Both additions are
built, wired live, and gate-green (256/256 self-checks, ruff clean, E-STOP trio intact); neither closes its
ceiling, and both are reported with the same developer-aware honesty as above.

### Gf — a second measured point: RAVEN, not ARC

New instruments: `reasoning/rpm_solver.py` (`RpmSolver` + live `RpmExpert`) and `reasoning/raven_bench.py`.
The solver is genuinely spiking-native: role-filler object codes bound with `cortex/vsa_reasoning.py`'s
bipolar HD algebra (the repo's NVSA substrate), **example→rule induction** over the NVSA rule set
(distribute-three via row-product+unbind, progression via `gf.py`'s shift inducer, constant, arithmetic)
executed as spike XOR / permute / cleanup ops, with the winning rule and answer selected through
`basal_ganglia.action_gating.BasalGangliaGate.select` + `thalamus.global_workspace.GlobalWorkspace` ignition.

```
raven_bench (synthetic 8-choice):  RpmSolver 98.3% (59/60), known 96.7%
controls:  random 11.7%  |  constant 15.0%   (chance 12.5%)   cited anchor NVSA-RAVEN ~88%
live route:  12/12 'raven:' queries reach rpm-raven via reasoning_cortex.route() with the correct answer
```

**What is honest to claim.** This closes the *RAVEN-abstraction slice* the way `arc_solver` closed the
geometry slice — no more, and for the same reason it is no more. The attribute codes (type/size/color over a
small integer pool) are a **developer-supplied grammar**, and under Chollet's developer-aware clause that
prior is charged against the score. So 98.3% on synthetic RPM is **not general fluid intelligence**; it is
test-time rule induction over an author-chosen attribute vocabulary. This is precisely the outcome the lone
surviving adversarial design predicted — every published RPM success, spiking or not, assumes a hand-specified
attribute grammar. The untested gap is **real** I-RAVEN with a perception front-end extracting attributes from
pixels (where NVSA's cited ~88% lives); `raven_bench` already loads such a dataset if present, and scores the
same solver honestly against it. The `gf.py` wall — novel *out-of-grammar* synthesis — remains unmoved.

### Language — the order-sensitive instrument the diagnosis demanded

The mechanism section above proved the cap is **order-invariance**: the SLDA-RFF readout decodes a summed bag,
so on BLiMP's hard pairs (word reorderings) it is not guessing but *systematically wrong*. This wave installs
the architectural cure, forward-serve half first: `reasoning/ar_spiking_lm_serve.py` (numpy, forward-only) +
`training/ar_spiking_lm.py` (torch [train] tier). It is a SpikeGPT-style Spiking-RWKV char LM (Zhu et al.,
arXiv:2302.13939) — LIF-gated linear recurrence token-mixer, no softmax — scored by **full-sentence
autoregressive log-probability**, which is order-sensitive by construction. Trained offline with BPTT +
fast-sigmoid surrogate gradients, frozen to numpy, forward-served with `logprob(sentence)`; wired into
`blimp_bench` as an opt-in real scorer.

```
order proof:  logprob(grammatical) != logprob(shuffled)   random-init diff 1.40 ;  after tiny inline train  167.19
blimp_bench:  AR spiking-LM -> n/a  (no frozen artifact at models/ar_spiking_lm.npz)
```

> **SUPERSEDED.** That `n/a` had a second cause this line did not name: `blimp_bench` never imported
> `ar_spiking_lm_serve` at all, so no artifact could have moved it. Both causes are fixed and the model
> is now measured — see *The language number was two defects, not one* below.

**What is honest to claim.** BLiMP is **unchanged** — the serve degrades honestly to `n/a` because no
real-scale artifact exists, and it fabricates no number. What is new is the **first representation in the
stack that can encode word order at all**: unlike every bag/n-gram model measured above, shuffling the input
changes the score. A toy model sits at n-gram ~57–60%; moving BLiMP off that floor needs the real-scale GPU
train (~GPT-2-small, 10–100M words) exported to `models/ar_spiking_lm.npz`. This wave ships the **architecture
and the full train→freeze→forward-serve pipeline**, not a result. The acquired-LM's ~53% must still not be
cited as language ability, and no BLiMP gain is claimed here.

### Bottom line, restated

| ceiling | status after Wave-1 |
|---|---|
| fluid intelligence | RAVEN slice now solved (98.3% synthetic, developer-aware-charged); **general Gf still not earned**; `gf.py` out-of-grammar wall unmoved |
| open-domain language | order-invariance cure now **installed as architecture** (order-sensitive AR logprob, proven); **BLiMP unmoved** — awaits the real-scale GPU train |

Both ceilings still stand. The wave converted each from "cause identified" to "cure built and instrumented",
and left the honest number exactly where the evidence puts it.

---

## The language number was two defects, not one

Two independent defects made every BLiMP number above unreadable. They are fixed, and the bench now
prints what it actually measures.

**Defect 1 — the AR model was never wired in.** `reasoning/blimp_bench.py` loaded `eprop_lm_alif_serve`,
then `eprop_lm_serve`, and **never imported `ar_spiking_lm_serve`**. So the Wave-1 `n/a` above named only
one of two causes: even with a frozen artifact present, no code path could score it. (The other cause, a
trainer/serve artifact-path split, was fixed separately in `bcd53611`.) All three frozen char LMs are now
scored, **each in its own row** — the old block preferred ALIF and silently dropped the others.

**Defect 2 — the scorer had a length bias larger than any effect ever reported here.** Scoring was
**unnormalized sum-logprob**. BLiMP's grammatical member is usually the *longer* one, and a weak char LM
pays a roughly constant cost per character, so the sum systematically prefers the shorter, ungrammatical
member. The size of the leak, measured: **sentence length ALONE scores 29.4%** — 20.6 points off chance,
on the wrong side. A control that far from 50% is the whole reason to distrust an unnormalized column.

### Measured — every model the bench can load, both scorings (67 × 25 = 1675 pairs)

| scorer | sum-logprob | per-char | delta |
|---|---|---|---|
| length only (control) | **29.4%** | constant → all ties → 0% | — |
| AR spiking-RWKV, `models/ar_spiking_lm.npz`, 10,339,425 params | 47.7% | 49.0% | +1.3 |
| AR spiking-RWKV, **untrained twin** — identical shape, random init | **49.5%** | 48.4% | −1.1 |
| e-prop ALIF-LM, `models/eprop_lm_alif.npz`, 812,256 params | 46.9% | 51.9% | +5.0 |
| e-prop flat LM, `models/eprop_lm.npz`, 112,475 params | 48.3% | 50.4% | +2.1 |
| char 5-gram on the same exposure (add-alpha backoff, untuned) | **53.0%** | not run | — |

Normalizing turns the length control into a **constant** — the same value for every sentence, so every
pair ties and ties score 0. That is the mechanism made visible: dividing by length is exactly what
removes the surface term the sum was leaking.

### What this does to the record

- **The sub-chance 46.9% was not a broken model.** Length-normalized it is 51.9%. Below chance was the
  scorer's bias, not an anti-grammatical model.
- **The documented 51.0% incumbent is not a property of a model either.** It is one scoring convention's
  output. Every historical BLiMP number in this document is an **unnormalized** number and is not
  comparable to a normalized one. The bench therefore prints **both** columns rather than switching:
  a silent switch would have made this document's own history incomparable without saying so.
- **The 61.2% anchor was the wrong yardstick.** It is a **word**-level 5-gram on 83M words; every model
  in the table is **character**-level. Measured on this repo's own exposure through the *same*
  byte-addressed reader and the *same* proven-disjoint held-out slice, a char 5-gram (add-alpha backoff,
  no tuned parameters, fit on 12M chars) scores **53.0%** (888/1675) at held-out char **PPL 4.5577**.
  **53.0% is the honest char-level bar.** Every prior comparison of a char-level model to 61.2% compared
  different units, and the gap it implied was partly a unit conversion.

### The AR model's result is an honest null, and that is a complete result

The frozen artifact is real: 10,339,425 params, dim 512, 3 layers, ctx 128, trained on 119,618,292 ASCII
chars of WikiText-103, checkpointed on **held-out** char PPL (never on BLiMP) at step 1399 of a planned
14,000 — **10% of its schedule**, with the run logged out to step 1700 (12%). Its checkpoint PPL is
**7.2016**.

Against its own untrained twin it measures **−1.8 points** under sum-logprob (47.7 vs 49.5) and **+0.7**
under per-char (49.0 vs 48.4). At 1675 pairs the standard error is ~1.2 points, so **neither delta is
distinguishable from zero, and the sign of "did training help?" flips with the choice of scorer.** That
flip is the finding: at this point in the schedule the scorer moves the answer more than the training
does. Nothing here is improved by picking the flattering column.

It also loses to the char 5-gram on **both** axes at once — 47.7/49.0% vs 53.0% on BLiMP, and held-out
char PPL 7.20 vs 4.56. A 10.3M-parameter spiking AR model at 10% of schedule is a **worse character
predictor** than a count-based 5-gram, so its BLiMP number needs no separate explanation.

**Contamination.** These numbers sit on a split whose disjointness is *proven*, not asserted — the lesson
of `scratchpad/lang_word_big_out.VOID.txt` mechanized as integer byte intervals rather than an estimated
chars-to-bytes ratio: exposure = bytes [0, 120,000,000), held-out = bytes [200,000,000, 200,400,000),
an 80,000,000-byte gap; the trainer's guard verifiably refuses an exposure that reaches the held-out
offset; 100/200/400-char verbatim probes from the held-out slice occur **nowhere** in the exposure; and
**0 of 3350** BLiMP sentences (67 × 25, good *and* bad) occur verbatim in the 119.6M-char exposure.

**Still unmoved.** The ceiling stands. The best number in the table is a count-based 5-gram at 53.0%, no
model shows grammatical competence, and this section changes only what the numbers *mean*, not what they
are: it removes a bias that was making models look worse than chance and an anchor that was making the
gap look wider than it is.

## Reproduce

> **The instruments now survive a clone; the datasets still do not. Both halves matter.**
>
> **What was fixed (2026-07-30).** Every script this section used to cite lived in `scratchpad/`, which is
> gitignored (`.gitignore:88`), so on any machine but the one that produced the numbers this page listed
> commands for files the reader did not have. Eight of them are now **tracked in `tools/`** — alongside
> `charge_probe.py` and `capacity_curve.py`, which live there precisely because a cited instrument must
> survive a clone: `arc_baseline.py`, `arc_score.py`, `blimp_baseline.py`, `ar_lm_ngram_control.py`,
> `ar_lm_disjoint.py`, `lang_scale.py`, `lang_diag.py`, `lang_probe.py`. Each takes its inputs as an
> explicit path (`--data` / `--corpus` / `--blimp`) instead of a hardcoded local one, and each **SKIPs
> with a message and exit 0** when its input is absent rather than reporting a number computed from no
> data. `tools/citation_guard.py` asserts this from now on, so the defect cannot come back silently.
>
> **What is still NOT reproducible from a clone, and is stated rather than papered over.** The
> `data/lang/` corpus, the `data/arc/` tasks and the `models/*.npz` artifacts are gitignored local-only
> data (the download commands are at the end of this section; the `.npz` has to be retrained). So a fresh
> clone gets the instrument and a clean SKIP, not the number. That is a strictly weaker claim than
> "reproducible", and it is the claim a static check can support: the reader is pointed at a file that
> exists.
>
> **One command was deleted rather than promoted.** `ar_lm_untrained_control.py` — the random-init twin
> behind the 49.5% baseline — **does not exist on disk at all** any more, so there was nothing to promote.
> It is also redundant: the canonical section above builds the untrained twin inside
> `reasoning.blimp_bench` rather than citing a separate script, and that path IS tracked. The 49.5% stays
> in the table as a **recorded result**, reproducible through the bench, not through this deleted file.
>
> **Not covered by the promotion, and unchanged.** The `scratchpad/ar_lm_*_out.txt` logs holding the
> training provenance (corpus size, PPL trace, step count) are still gitignored, and the `.npz` metadata
> still records format/vocab/dims but none of that provenance. The round-trip deviations and the incumbent
> re-score are still recorded results. Fixing that means writing provenance into the trainer's metadata,
> and it is not fixed here.

```bash
cd D:/AI-project/spiking-neural-data-lake

# harness self-checks (both skip cleanly if the data is not downloaded)
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.arc_bench
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.blimp_bench
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.arc_solver

# Wave-1 additions (Gf RAVEN + order-sensitive AR language serve)
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.raven_bench
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.rpm_solver
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.ar_spiking_lm_serve          # fast stdlib+numpy order proof
PYTHONPATH=src .venv-cu128/Scripts/python.exe -m snn_data_lake.reasoning.ar_spiking_lm_serve --train  # strict trained order proof (torch)

# before-state baselines (tracked in tools/ since 2026-07-30; each SKIPs cleanly without its dataset)
.venv-lake/Scripts/python.exe tools/arc_baseline.py 100
.venv-lake/Scripts/python.exe tools/blimp_baseline.py 25

# the ARC solver, both splits, library on/off
.venv-lake/Scripts/python.exe tools/arc_score.py evaluation on
.venv-lake/Scripts/python.exe tools/arc_score.py training off

# the two-defect language section: the bench prints the whole table (both scorings, all three LMs,
# length control, and it BUILDS the untrained twin) and needs data/lang/blimp + the gitignored
# models/*.npz artifacts
PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.blimp_bench
.venv-lake/Scripts/python.exe tools/ar_lm_ngram_control.py 12e6   # the char 5-gram bar (53.0%)
.venv-lake/Scripts/python.exe tools/ar_lm_disjoint.py             # the disjointness proof

# the language scale + mechanism study
.venv-lake/Scripts/python.exe tools/lang_scale.py 1e3 1e4 1e5 1e6 1e7
.venv-lake/Scripts/python.exe tools/lang_diag.py     # ablations D1-D7
.venv-lake/Scripts/python.exe tools/lang_probe.py    # decodability D8

# each of the eight takes its input as an explicit path, e.g. a corpus somewhere else entirely:
.venv-lake/Scripts/python.exe tools/lang_scale.py 1e3 --corpus /data/wikitext/train0.txt
```

Datasets (all gitignored under `data/`):

```bash
curl -sL -o arc.zip   https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip     # -> data/arc/
curl -sL -o blimp.zip https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip   # -> data/lang/blimp/
# WikiText-103: HF parquet wikitext-103-raw-v1/train-00000-of-00002.parquet -> data/lang/wikitext/
```

---

## Appendix (2026-07-29) — the AR spiking-LM path split, a first real train, and three corrections to how every number above must be read

This appendix records a defect fix and a set of measurements made on 2026-07-29. Every number below
was measured this session; the scripts are in `scratchpad/ar_lm_*.py` and their outputs in
`scratchpad/ar_lm_*_out.txt`.

### The defect, and what the fix does and does not change

`training/ar_spiking_lm.py` exported to `models/spiking_lm.npz`; `reasoning/ar_spiking_lm_serve.py`
loaded `models/ar_spiking_lm.npz`. Neither half could ever see the other's file, and the serve
self-check *printed* the trainer's path while its loader used the other one. Fixed (commit
`bcd53611`): both halves now agree on `models/ar_spiking_lm.npz`. Audit of the same shape across the
tree: `eprop_lm`/`eprop_lm_serve` and `eprop_lm_alif`/`eprop_lm_alif_serve` both agree and load — no
further trainer/serve path mismatches (`scratchpad/ar_lm_pathaudit.py`).

**Correction 1 — the `n/a` had TWO causes and one remains.** `blimp_bench.py` never imports
`ar_spiking_lm_serve` at all: its opt-in scorer block (`blimp_bench.py:151-166`) loads
`eprop_lm_alif_serve`, then `eprop_lm_serve`. The Wave-1 text above says the AR LM was "wired into
`blimp_bench` as an opt-in real scorer" — as of this appendix that wiring **does not exist**. Until
`blimp_bench` (or a successor) actually consumes `SpikingLMServe`, no AR-spiking-LM artifact can move
the reported number, path fix or no path fix.

**The "BYTE-FOR-BYTE" claim, measured** (`scratchpad/ar_lm_roundtrip.py`,
`ar_lm_parity_artifact.py`): literally false, materially true. torch-f64 vs numpy-f64 logits max
|dev| 1.776e-15 (float op-ordering); the real round trip (f32-trained weights served in f64) max
|dev| 1.92e-6; **0 of 23,232 spikes differ** under either dtype pairing; full-sentence logprob dev
2.84e-14. On the shipped `.npz` re-imported to torch: 1.07e-14, 0 argmax disagreements.

### Correction 2 — the 51.0% incumbent is stale, and the live number is below chance for a localized reason

Measured live (67 paradigms x 25 pairs = 1675, `scratchpad/ar_lm_incumbent.py`): what `blimp_bench`
actually loads today scores

```
e-prop ALIF-LM (preferred) : 46.9% sum-logprob | 51.9% per-char-normalized
e-prop flat LM (fallback)  : 48.3% sum-logprob | 50.4% per-char-normalized
```

Below chance is not a weak model; it is a systematic bias, and it is **localized**: the good member
of a BLiMP pair is usually the longer one (`length_scorer` = 29.4%, i.e. preferring the shorter
sentence is wrong ~70% of the time), and an unnormalized sum-logprob pays ~constant cost per
character, so a weak LM systematically prefers the shorter = bad sentence. Dividing by sentence
length flips both models back to ~chance (+5.0 / +2.1 points). Any future sum-logprob BLiMP score
must be read with this bias in mind, and 51.0% must stop being cited as the current incumbent.

### Correction 3 — the 61.2% anchor is WORD-level and was the wrong yardstick

The 61.2% "5-gram" anchor (Warstadt et al., Table 3) is a **word**-level model trained on 83M words.
A char-level model cannot be compared to it. Measured on this repo's own corpus
(`tools/ar_lm_ngram_control.py`, char 5-gram, add-alpha backoff, nothing tuned): **51.4%** at 3M
chars, **53.0%** at 12M chars (held-out char PPL 5.39 / 4.56). **53.0% is the honest char-level
n-gram bar**, and prior comparisons of char models to 61.2% (including the "toy sits at n-gram
~57-60%" lines above) were against the wrong yardstick.

### The first real train of the AR spiking-LM — an honest null, with the cause instrumented

> **Reconciliation note, added when these two accounts met in a merge.** This appendix and
> *"The language number was two defects, not one"* above were written independently, hours apart, by two
> passes working from the same artifacts — so they cover the same ground and git merged both cleanly
> without noticing. **The section above is canonical**: it carries all three models scored under BOTH
> scorers plus the length-only control in one table, and it verifies the untrained-twin baseline by
> building the twin rather than citing it. Where the two differ, prefer it.
>
> **"Canonical" does NOT extend to the PPL comparison, corrected 2026-07-29.** The canonical section
> quotes held-out char PPL **7.20 vs 4.56**, which compares across *different* slices; this appendix's
> **6.54 vs 4.88** is the protocol-MATCHED pair (identical windowed protocol, same held-out slice), and on
> that specific number the appendix is right and the canonical section is not. A blanket "prefer the
> section above" was too broad, and saying so is cheaper than letting a mismatched comparison inherit
> authority from a label.
>
> **Two numbers genuinely differ**, not one. First, this appendix says
> the run reached **12%** of schedule, which is the last *logged* step (1700 / 14,000). The **checkpoint
> that produced every measured number is step 1399 = 10%**. Both are true of the run; only the second is
> true of the artifact, and the artifact is what was scored.
>
> Kept rather than deleted because it holds the training provenance the canonical section does not:
> corpus size, the PPL trace, and the byte-interval disjointness proof. That provenance lives ONLY in a
> gitignored log (`scratchpad/ar_lm_full_out.txt`), and the `.npz` metadata records format/vocab/dims
> but no corpus size, step count or PPL — so on a fresh clone these numbers are unreproducible from the
> artifact alone. That is worth fixing in the trainer's metadata, and is not fixed here.
> **Update 2026-07-30:** the TRAINER was fixed (`852ae1d7` added the provenance block, documented in
> `docs/TRAINING.md`). THIS ARTIFACT was not, and cannot be — see the log-sourced note in the canonical
> section below. The mechanism exists; this file predates it.

Setup (all in commit `bcd53611`): byte-addressed WikiText reader with a hard held-out guard
(exposure `[0, E)` refuses to run when `E >= 200,000,000`, the offset `tools/lang_scale.py` holds out —
the `lang_word_big_out.VOID.txt` lesson mechanized). Disjointness **proven, not assumed**
(`tools/ar_lm_disjoint.py`): 100/200/400-char held-out probes absent from the exposure, and 0 of
3350 BLiMP sentences (good AND bad members) occur verbatim in the 119.6M-char training text. Model
selection on held-out PPL, never on BLiMP.

Run: dim=512, layers=3, ctx=128, batch=28 — **10,339,425 params** (SpikeGPT-style Spiking-RWKV,
BPTT + fast-sigmoid surrogate, RTX 5070) on the first 120,000,000 bytes (119.6M ASCII chars ~ 20M
words) of WikiText-103. **Stopped deliberately at step 1700 of 14000** (coordinator halt for the
repo gate); the shipped artifact is the best held-out checkpoint, step 1399, held-out char PPL
7.2016 (32k-char eval slice) / **6.5423** over the full 200k-char held-out slice under the windowed
protocol.

> **LOG-SOURCED, NOT ARTIFACT-VERIFIED (checked 2026-07-30).** `models/ar_spiking_lm.npz` carries **no
> `provenance` block** — `SpikingLMServe().provenance_status()` returns `"unrecorded"` and all ten
> `PROVENANCE_KEYS` read `"unknown"`. It was exported before the schema landed (`852ae1d7`). So every
> training number in this paragraph — step 1399, 14000 planned, the PPLs, the corpus size — is sourced
> to the gitignored log `scratchpad/ar_lm_full_out.txt` and **cannot be checked against the file**.
> What the artifact does testify to is what it holds: **10,339,425 params**, vocab 97, dim 512,
> layers 3, d_ff 2048, `trained=True`. Those match, so the paragraph is not contradicted — it is
> unverified, which is a different and weaker status.
>
> **This is permanent for this file, not a to-do.** No checkpoint survives to re-export from, and not
> one PROVENANCE_KEY is derivable from weights. Retraining would produce a DIFFERENT model and
> silently invalidate the BLiMP numbers recorded in `docs/coverage/` against this one, so it is not a
> fix. The next real-scale run inherits the fixed exporter and will testify for itself.

BLiMP, 67 x 25 = 1675 pairs, scored through the frozen numpy serve (`scratchpad/ar_lm_blimp.py`):

```
TRAINED   sum-logprob        : 47.7%  (799/1675)
TRAINED   per-char normalized: 49.0%  (821/1675)
UNTRAINED same architecture  : 49.5%  (829/1675)   <- the control that gives the row above meaning
controls  random 51.1% | length-only 29.4%         (chance 50.0)
```

**BLiMP did not move: the learned delta over random init is -1.8 points (sum) / +0.7 (per-char).**
(The per-char figure read **-0.5** until 2026-07-29. That was a scorer mix-up: it subtracted the
*sum-logprob* untrained baseline 49.5 from the *per-char* trained score 49.0. The untrained twin scores
**48.4** per-char, so the correct per-char delta is **+0.7**. This is not cosmetic -- two negatives
destroyed the canonical section's headline finding, which is that **the SIGN of "did training help?"
flips with the choice of scorer**. Found by audit; the reconciliation note above said "one number
genuinely differs" and this was a second.)
This is a null result and it is reported as one. What the instruments say about why:

1. The model had not yet learned even its own training data's local statistics: under an IDENTICAL
   windowed protocol on the SAME held-out slice, the spiking LM's char PPL is **6.54** vs **4.88**
   for a char 5-gram fitted on the same exposure (`scratchpad/ar_lm_final_eval.py`). A model still
   above the n-gram PPL bar has no business beating the n-gram's 53.0% BLiMP, and it does not.
   The bottleneck at this scale is optimization time (the scored checkpoint is step 1399 = 10% of
schedule; 12% was the last LOGGED step, and only the checkpoint was scored), not the
   benchmark.
2. The per-paradigm scatter is bimodal (100%/100%/100% next to 0%/0%/4%; 27/67 paradigms above
   chance), the signature of systematic surface preferences, not grammar — a random scatter would
   concentrate near 50%.
3. Even the fully-converged char-level ceiling on this data is ~53% (correction 3). Char-level
   models at this corpus scale sit within ~3 points of chance; the docstring's honest-scale
   statement (~GPT-2-small on 10-100M WORDS for 70-80%) still stands, and word/subword units are
   likely a precondition for the next honest increment, not more steps alone.

**What this appendix claims:** the pipeline is now genuinely round-trip-verified at real scale, the
eval split is proven disjoint, and the first real artifact exists at the path the serve loads.
**What it does not claim:** any BLiMP movement. The language ceiling stands exactly where the
mechanism section put it, and the two yardsticks it was being measured against (51.0% incumbent,
61.2% n-gram) were both wrong in the flattering direction.
