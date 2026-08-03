# Feature Specification: The Brain-Native Assembly-Calculus Language Organ

**Feature Directory**: `specs/005-language-organ`

**Created**: 2026-08-02

**Source**: Owner directive, 2026-08-02, on the research basis in `research.md` (arXiv 2306.15364,
2507.11788, 2406.11977, 2607.26648). Governed by `docs/POLICIES.md` (P0-P6), which is the authoritative
policy text; `.specify/memory/constitution.md` is its governance surface and defers to it by its own rule
- *"Where the two ever disagree, `docs/POLICIES.md` wins and this file is the defect"*
(`constitution.md:36-38`). The Standing Invariants (`constitution.md:101-114`) apply throughout.

**Status**: Draft - ready for `/speckit-plan`. Three questions recorded unresolved in
[Open questions](#open-questions), each with a stated assumption the plan proceeds on.

**Working-tree note**: HEAD moved during authoring (`5599e6ab` -> `30d8e76a`) because a second agent is
committing to this repo concurrently. **No figure in this document is attributed to a commit on the
strength of that observation.** Every measured number quoted below is quoted from the tracked document
that recorded it (`docs/DEPTH_CEILINGS.md`, `docs/coverage/Language.md`), and T001 re-pins the real
baseline commit before any measurement is taken. Attributing a number to a commit nobody verified is the
defect `specs/003`'s own baseline note exists to prevent.

## Problem

The brain's language faculty is at chance and the cause is architectural, not a matter of scale, tuning,
or exposure. This has been established by measurement, not argument, and the measurements are already in
the tree.

`cortex/language_acquisition.py` is the live language tier. `Brain._lm_serve` builds it
(`brain.py:3499`), `Brain._language` calls its `generate()` (`brain.py:3541`), and `Brain.respond`
consumes the result on the Gf branch (`brain.py:3077`). It **does not clear the 50.0% chance floor**,
and it is **beaten by a character bigram table fitted on the identical text (52.8%)**
(`docs/DEPTH_CEILINGS.md:118-127`).

**The often-quoted 51.0% is WITHDRAWN and must not be cited as the current incumbent score.**
`docs/DEPTH_CEILINGS.md:555` - *"Correction 2 - the 51.0% incumbent is stale, and the live number is
below chance for a localized reason"* - retires it. The live figures there, over 67 paradigms x 25
pairs = 1675, are **46.9% sum-logprob / 51.9% per-char** (e-prop ALIF-LM) and **48.3% / 50.4%** (e-prop
flat LM). Below chance is a length bias, not a weaker model, and the whole reason both scorings are
mandatory here - see F3a. **What the incumbent scores on this feature's frozen pair set is measured at
T014, not quoted from any document**, and that measured number, not 51.0%, is what the verdict compares
against.

Scaling its exposure 4,119x - from 2,428 characters to 10,000,000
characters of WikiText-103 - moves it to **53.2%**, where a char 5-gram on the same text sits at 53.4%
(`docs/DEPTH_CEILINGS.md:147-155`). Four candidate causes were falsified by ablation: homeostatic decay
(53.4 -> 53.1 with the decay removed entirely), vocabulary folding (0 of 1675 pairs collapse), short
context (ctx 2 -> 32 makes it *worse*, 57.3 -> 53.4), and feature loss (0.65% collision over 117,859
contexts).

What is left is the architecture, confirmed twice: **the readout is a linear map over a summed bag**, 34
x 88 = **2,992 parameters regardless of corpus size**. A linear map on a superposition can learn an
additive per-position vote; next-character prediction needs conjunctions, which are not linearly
separable from a summed bag. Fitting the *same architecture* full-batch to convergence - the convex
optimum of the shipped design - reaches **57.0%**. The online local rule reaches 53.4%. It underfits its
own architecture by about three points, and **the architecture is the wall**.

This is not a bug to fix inside the existing module. There is no parameter to turn. The mechanism cannot
represent the structure language has, so no amount of the right kind of exposure will teach it that
structure.

There is a mechanism that can. The Assembly Calculus line of work (Papadimitriou / Vempala /
Mitropolsky; arXiv 2306.15364, 2507.11788) builds language out of six ingredients that are all already
this project's ingredients - excitatory neurons, brain areas, random synapses, Hebbian plasticity, local
inhibition (a per-area cap), and **inter-area inhibition** - and from a tabula rasa, on a modest number
of GROUNDED sentences, learns word semantics, syntactic role (verb versus noun), word order, and can
generate novel sentences. No backprop. No pretrained weights. No corpus-gradient descent. It is the
architecture the existing module's docstring *describes itself as* and structurally is not.

So this feature replaces the language architecture. Per P2 that means exactly one survivor.

## Findings already in hand

Verified against tracked source and tracked measurement documents. Each finding names where it came from
so the next reader can re-derive it rather than take it.

**F1 - The live language path is three functions long, and the organ replaces the middle one.**
`Brain._lm_serve` (`brain.py:3499`) lazily builds and caches the char learner and reloads
`language_acq.npz` when `self._persist`; `Brain._language` (`brain.py:3541`) calls `generate()` with a
fixed decode configuration and applies a coherence guard; `Brain.respond` (`brain.py:3077`) takes the
result on the Gf-fluid branch, falling back to `narrator.introspect(...).summary` when the generation is
empty or fails the guard. The organ is a substitution at `_lm_serve`, not a new branch. Writes at
`brain.py:2607` and `brain.py:3656` persist the learner's state.

**F2 - The existing module is measurably at the chance floor, and the cause is proven architectural.**
See Problem. The decisive datum is that the convex optimum of the same architecture is 57.0%: the
learning rule is not the cap, the representation is.

**F3 - A grammaticality instrument already exists, it is not gameable by fluency, and it already ships
its own admission controls.** `reasoning/blimp_bench.py` scores 67 paradigms x N minimal pairs; a pair's
two sentences share nearly every word, so a template bank and a fluent-sounding generator both fail it,
and no LLM judge is involved. It ships three degenerate scorers that must fail: `random_scorer`
(`:106`, must land near 50%), `constant_scorer` (`:111`, no opinion; ties count as wrong, so it must
score 0%), and `length_scorer` (`:116`, pure surface heuristic, exposing length leakage). It reports
**both** unnormalized sum-logprob and per-character normalized scoring side by side, because switching
silently would make every prior number incomparable. Unscorable sentences become the `UNSCORABLE`
sentinel and always LOSE their comparison rather than winning one. **The metric this feature will be
judged on therefore already has an admission control, and it already fires.**

**F3a - The 51.0% incumbent figure is WITHDRAWN, and the replacement numbers are below chance for a
measured, localized reason.** `docs/DEPTH_CEILINGS.md:555`, *"Correction 2 - the 51.0% incumbent is
stale, and the live number is below chance for a localized reason"*, ends: *"51.0% must stop being cited
as the current incumbent."* Live over 1675 pairs: e-prop ALIF-LM **46.9% sum-logprob / 51.9% per-char**,
e-prop flat LM **48.3% / 50.4%**.

The mechanism, because "below chance" invites the wrong conclusion: BLiMP's grammatical member is usually
the **longer** sentence, and an unnormalized sum-logprob pays a roughly constant cost per character, so a
weak LM systematically prefers the **shorter** - wrong - member. Measured, not inferred: `length_scorer`,
which does nothing but prefer the shorter sentence, scores **29.4%**. Normalizing by length returns both
models to roughly chance (+5.0 and +2.1 points).

Three consequences bind this feature. **(1)** Both scorings are mandatory (FR-013) and this is why - a
single-convention number here is uninterpretable. **(2)** About six points are available from fixing the
length bias alone, which is larger than the entire per-character competence band of 2.6 points, so **a
scorer that merely handled length correctly would look like a language win under sum-logprob**; FR-026
and control four exist for that. **(3)** The withdrawn 51.0% was `tools/blimp_baseline.py`'s acquired-LM
arm while the live figures are `blimp_bench.SERVES`' e-prop serves - different harnesses, different
models - so **no current trustworthy figure exists for `cortex/language_acquisition.py` specifically**,
and T014 measures it on the frozen set rather than quoting anything. Full derivation in `research.md`
R2a.

**F4 - The data is present.** `data/lang/blimp/` holds the 67 `.jsonl` paradigm files and
`data/lang/wikitext/` the scale corpus. `blimp_bench.main()` SKIPs cleanly (exit 0) when the directory is
absent, so a clone without the download degrades to a skip rather than a failure - which means **an
absent-data skip must never be read as a passing A/B**, and the tasks assert on the pair count.

**F5 - `tools/blimp_baseline.py` is the recorded "before" harness and it already contains the
non-degenerate control.** It scores three arms on the same text: the acquired LM, ROSE `comprehend`, and
**a char bigram fitted on the identical exposure text**. That bigram is the control that makes "anything
would have helped" refutable, and it is the arm that currently *wins*. It was promoted out of gitignored
`scratchpad/` on 2026-07-30 precisely because a baseline whose instrument does not survive a clone is not
a baseline.

**F6 - No Language faculty can currently produce a tracked A/B artifact, and this is recorded.**
`tools/charge_probe.py language 60` exits 2 - *"unknown drive source 'language'"* - because the probe
neuters a key of `Brain._drive_w` (`brain.py:633-648`) and that dictionary contains no language source at
all (`artifacts/lang_no_ab_source.txt`; `docs/coverage/Language.md`, finding R8). The rubric makes a
recorded-exit-code A/B the sole gate for band > 40, so **the instrument that would score this feature's
result is structurally blind to it in both directions**. Three Language rows already have runtime
evidence meeting the band-41-60 definition and are capped at 40 for exactly this reason. Closing this is
in scope: it is the difference between a measured win and an unreportable one.

**F7 - Language coverage is 22.2 over eleven rows, and `generation_acquired_lm` is the top row at band
40.** `docs/coverage/Language.md`. The organ inherits that row. A null result must leave the number where
it is, and the ledger's own rule is that a row is never deleted to raise a score.

**F8 - `cortex/spiking_lm_serve.py` is an empty pipe, is registered CORE, and its self-check passes.**
`run_selfchecks.py:83`. There is no trained `snn_lm/1` weight file anywhere in the repo; it serves a
hand-built four-token demo (`vocab=4, d_model=4`, one block) that generates `'abbbbbb'`
(`docs/DEPTH_CEILINGS.md:140-143`). It has **no live consumer**: its only references are
`src/snn_data_lake/training/spiking_lm.py`'s transfer check and its own self-check
(`docs/coverage/Language.md`, capability inventory). It is not a competitor to the organ - it is a
separate P3 disposition that this feature must adjudicate rather than leave dangling, because a green
self-check on a module that changes nothing is the "green light that cannot go red" pattern this repo
keeps shipping. Note before deleting: it is the **only torch-free `snn_lm/1` reader**, so retiring it
orphans that export format.

**F9 - Registration tiers differ and the swap must respect them.** `cortex.spiking_lm_serve` is at
`run_selfchecks.py:83`, plain CORE. `cortex.language_acquisition` is at `run_selfchecks.py:369`, in the
numpy-gated CORE tier (it imports numpy). `reasoning.blimp_bench` is RESEARCH (`:351`). A replacement
module must be registered in the tier its imports actually require, and the CI count must not silently
shrink.

**F10 - Grounding already exists in the tree and does not need inventing.**
`reasoning/world_kb_stream.py::load_kb()` returns a deterministic knowledge base of real
`(head, relation, tail)` triples across geography, chemistry, biology, physics, astronomy, computing,
mathematics and commonsense, and `reasoning/rose_language.py::ground()` (`:155`) already resolves a word
to a KB referent or returns `None` for an honest lexical gap. The organ needs (sentence, scene) pairs; a
triple **is** a scene with two arguments and a relation. Building a second grounded corpus when one is in
the tree would be the additive move P2 forbids.

**F11 - The existing exposure corpus is copyright-safe, self-authored, and ungrounded.**
`language_acquisition.py`'s `CORPUS` is plain English about the brain itself, repeated four times, for a
total lifetime exposure of **2,428 characters**. It carries no scene, so it can teach word statistics and
cannot teach word meaning. Whatever the organ is exposed to must be paired with a referent or the
grounded-acquisition claim is unfounded.

**F12 - Artifact fingerprinting is positional, and prints are ASCII.** The reproduction check harvests
every bare `N/M` ratio in a probe's stdout, so adding one ratio to a shared harness changes every
artifact's fingerprint. Counts are reported as `N of M`. Output is ASCII-only (Windows cp1252).

**F13 - Two tests already touch this area.** `tests/test_blimp_bench_serve.py` and
`tests/test_ar_lm_provenance.py`. There is **no** test that imports `cortex.language_acquisition`. A
module carrying the live reply path with zero test coverage is part of why its failure went unmeasured
for as long as it did.

**F14 - Recording a held-out artifact through the sanctioned path is currently impossible.**
`tools/coverage_score.py record` writes only `kind/command/exit/commit/recorded_at/recorded_by/python`,
but F10 of the scorer refuses any `kind: heldout` artifact lacking `metric:` and `baseline:`. So band >
80 is unreachable through the sanctioned path today (`docs/coverage/Language.md`). This feature does not
need band 80 and must not pretend otherwise; it is recorded here so the plan does not budget for a claim
the ledger cannot accept.

## User Scenarios & Testing

Written as what an operator can point at and check. Every scenario is a measurement.

### User Story 1 - The metric can see what it claims to measure, before anything is built (Priority: P1)

The operator runs the grammaticality instrument on the pair set this feature will be judged on, and
watches the degenerate scorers FAIL on that exact pair set - not on the full 67 paradigms, on the subset
the comparison will actually use. If a restricted pair set lets `random_scorer` drift away from chance,
or lets `length_scorer` carry most of the signal, then the restriction has introduced a leak and the
comparison is void before it starts.

**Why this priority**: this is the blocking slice. This project's recurring failure is a gate that is a
green light unable to go red, and a *subsetted* benchmark is the classic way one is created by accident.
Standing Invariant 1 - count a gate's output before any rule leans on it - makes this first, not
diligent.

**Independent Test**: run the three degenerate scorers on the restricted pair set. `random_scorer` lands
near 50%, `constant_scorer` scores 0%, `length_scorer` reports its leakage explicitly. If any of the
three passes where it must fail, the pair set is disqualified.

**Acceptance Scenarios**:

1. **Given** the pair set the A/B will use, **When** the three existing degenerate scorers are run on it,
   **Then** each fails in its specified direction, and the numbers are printed alongside the headline
   score in the same invocation - never in a separate run whose result could be from a different subset.
2. **Given** the restricted pair set, **When** its size is reported, **Then** it is reported as
   `N of M` pairs over `P of 67` paradigms, so a shrinking denominator is visible rather than silent.
3. **Given** BLiMP data absent from disk, **When** the harness runs, **Then** it SKIPs and the skip is
   **not** counted as a passed A/B anywhere downstream.
4. **Given** a proposed restriction rule, **When** it is applied, **Then** it is applied identically to
   both arms and is derived from a property of the pair set, not chosen after seeing either arm's score.

### User Story 2 - A baseline exists that a later claim can be measured against (Priority: P2)

The operator can name a commit, a command, and a number for what language competence was before anything
changed, and re-run the command to get the number back.

**Why this priority**: an "improvement" without a captured before is a claim, not a measurement. It is
second only because the instrument must be shown honest first - a baseline taken with a blind instrument
is a number, not evidence.

**Independent Test**: `specs/005-language-organ/baseline.txt` names a commit; re-running the recorded
commands at that commit reproduces the recorded figures.

**Acceptance Scenarios**:

1. **Given** a clean tree for the language sources, **When** the baseline is captured, **Then** it
   records the commit SHA, the three gates' status, the BLiMP score of the existing acquired LM under
   both scorings, the char-bigram control on identical text, the restricted-set sizes, and the tick and
   reply latency of `Brain.respond`.
2. **Given** a dirty tree for any language source, **When** baseline capture is attempted, **Then** it
   ABORTS. `specs/003` recorded the failure this prevents: a research pass measured an uncommitted
   rewrite and labelled it the baseline.
3. **Given** the baseline, **When** any later figure in this feature is quoted, **Then** it names the
   baseline it is a delta from.

### User Story 3 - The organ acquires language from grounded exposure, jointly (Priority: P3)

The operator exposes the organ to grounded sentences - each a word sequence paired with the scene it
describes - and can show, on held-out sentences, that it has acquired word meaning, syntactic role, and
word order together.

**Why this priority**: this is the feature. It is third because the two slices above are what make its
result believable.

The joint requirement is not stylistic. arXiv 2406.11977 finds syntactic and semantic bootstrapping are
the same mechanism, and both effects are strongest when syntax and semantics are learned SIMULTANEOUSLY,
mutually constraining each other's hypothesis spaces. A curriculum that teaches words first and syntax
second is a different, weaker mechanism wearing the same name. So the staged version is not merely
avoided - it is **built and run as a control arm**, and if staged matches joint, the joint claim is
withdrawn rather than asserted.

**Independent Test**: on held-out grounded sentences never exposed, the organ assigns the correct
syntactic role (noun versus verb) and the correct referent above the frequency baseline that the same
harness computes.

**Acceptance Scenarios**:

1. **Given** grounded exposure, **When** a held-out sentence is presented, **Then** the organ's
   role assignment beats the most-frequent-role baseline computed on the same exposure - the harness
   reports both, so a win against nothing is visible.
2. **Given** the same exposure with **scenes randomly permuted** against sentences, **When** the organ is
   trained identically, **Then** it does NOT reach the main arm's score. If it does, the organ learned
   from word statistics alone and the grounding is decorative.
3. **Given** the same exposure with **plasticity disabled**, **When** the organ is evaluated, **Then** it
   scores at the frequency baseline. A frozen organ that scores well means the readout, not the
   learning, is carrying the result.
4. **Given** a **staged** arm (semantics to convergence, then syntax), **When** it is compared to the
   joint arm on identical exposure and identical held-out data, **Then** the difference is reported
   whichever way it falls, and a joint-is-better claim is made only if it is measured.
5. **Given** the organ, **When** it is constructed twice from the same seed, **Then** the two are
   identical - the random synapse graph comes from a per-instance `random.Random(seed)`, never from
   module-level seeding.

### User Story 4 - The A/B decides, and it is allowed to say no (Priority: P4)

The operator runs the head-to-head on the existing instrument and gets a verdict that can come back
negative. If it does, the organ is not adopted, nothing is deleted, and the feature stops.

**Why this priority**: it depends on all three above.

**Independent Test**: one command emits both arms' scores under both scorings on the identical pair set,
with the degenerate controls in the same output, and an exit code that distinguishes ADOPT from NULL.

**Acceptance Scenarios**:

1. **Given** both arms, **When** the A/B runs, **Then** both are scored on the identical pair set with
   the identical scoring functions, and the score is reported under both sum-logprob and per-character
   normalization.
2. **Given** an organ that does not beat the incumbent, **When** the A/B completes, **Then** the recorded
   outcome is NULL, the incumbent is not deleted, the organ is not wired into `respond()`, and the
   feature is reported complete. **A null result is a successful end state**, and this specification
   treats it as one.
3. **Given** an organ that does not beat the **char-bigram control**, **When** the A/B completes,
   **Then** the outcome is NULL even if it beats the incumbent. The incumbent already loses to a bigram
   table; replacing it with something that also loses to a bigram table is not progress.
4. **Given** a verdict, **When** it is recorded, **Then** it is recorded through
   `tools/coverage_score.py record --kind ab` so the band ceiling comes from the recorded exit code and
   not from the claim, and it is `--rerun`-able.
5. **Given** any arm that cannot score a sentence, **When** it is scored, **Then** the pair is a tie and
   a tie counts as wrong. Credit is never awarded for having no opinion.

### User Story 5 - Exactly one language generator survives (Priority: P5)

After an ADOPT verdict, an operator asking "which mechanism produces the brain's replies?" gets one
answer, and the other is gone from the tree.

**Why this priority**: it is the P2 obligation, and it is last because deleting before proving is the
thing P2's honesty guard forbids.

**Independent Test**: after the retire, a tree-wide search finds no reachable second next-token generator
on the reply path, and the three gates are green with no assertion loosened and no self-check count
quietly reduced.

**Acceptance Scenarios**:

1. **Given** an ADOPT verdict, **When** the retire lands, **Then** `cortex/language_acquisition.py` is
   **deleted** - not flag-gated, not kept behind a mode switch. Counting survivors is the test: if two
   implementations of the same job are still reachable, it was not a merge.
2. **Given** the retire, **When** it is prepared, **Then** every unique capability of the retired module
   is either ported or explicitly recorded as dropped with the reason. The online `expose()` from live
   conversation and the `language_acq.npz` persistence are the two that must be adjudicated by name.
3. **Given** the retire, **When** the self-check registry is updated, **Then** the replacement is
   registered in the correct tier and the total module count does not shrink without being stated.
4. **Given** a NULL verdict, **When** the feature closes, **Then** nothing is deleted and this scenario
   does not execute.
5. **Given** `cortex/spiking_lm_serve.py`, **When** its disposition is decided, **Then** it is one of
   P3's three - wired, retired, or exempt with a reason - and the decision names its consumer count as
   `N of M`. It is adjudicated regardless of the organ's verdict, because it is dormant either way.

### Edge Cases

- **The organ's lexicon does not cover BLiMP.** BLiMP's vocabulary is far larger than any modest grounded
  corpus. An organ that returns UNSCORABLE for most pairs ties them, ties count as wrong, and it loses to
  the incumbent by construction rather than on merit. This is the single largest threat to the feature's
  validity and Q1 records it; the restriction rule in US1 exists to answer it, and the rule must be fixed
  before either arm is scored.
- **A restriction rule that flatters the organ.** Choosing paradigms after seeing scores is p-hacking with
  extra steps. The rule must be a property of the pair set (lexical coverage), applied identically, fixed
  before scoring.
- **The organ produces no comparable score at all.** Assembly convergence is not a log-probability. If a
  monotone acceptability score cannot be derived from the organ, the two mechanisms cannot be compared on
  the existing benchmark, and per P2 the honest answer is *the swap is not P2-eligible*, not *pick the
  more fashionable one*.
- **The firing rate does not fall.** arXiv 2607.26648 bounds a recurrent spiking language model's density
  from below: `rho >= H_b^{-1}(log2 M / H)`, with the floor RISING with memory load and task difficulty.
  The organ is recurrent and carries state across a sentence. **This organ will not also be sparse or
  cheap**, and the specification says so here rather than letting an implementer discover it. No sparsity
  or energy target is an acceptance criterion, and none may be added later without re-deriving the bound.
- **Latency regression on the live tick.** Each assembly projection is a dense random-matrix product with
  a top-k, iterated to convergence, per word. Replacing a 34x88 matrix-vector product with this will make
  `respond()` slower. The cost must be measured and stated, not discovered.
- **A fixed round count smuggled in as a constant.** "Project five times" is a numeric literal
  arbitrating behaviour, which P1 forbids. Convergence must be detected, not counted to.
- **Area size and cap chosen by hand.** `n`, `k`, `beta`, and the synapse density are exactly the
  constants the AC literature states as literals. Every one must be derived from existing state or
  learned, or P1 is violated in the feature's core data structure.
- **Two brains.** Any harness that constructs a `Brain` outside `make_brain()` violates P4. The organ is
  built in the existing `_lm_serve` slot.
- **A benchmark that teaches the model its own test.** Scoring must be strictly read-only; the incumbent
  harness already enforces this and the organ's scorer must too. An organ whose Hebbian weights update
  during evaluation has been taught the test.
- **The daemon is running.** Probes read `~/.snn_data_lake` while the always-on daemon writes it.
  `coverage_score record`/`rerun` isolate `NORD_HOME` per invocation; a probe run outside that path must
  do the same or its number is noise.
- **An empty-lake probe.** A probe assuming accumulated durable state sees an empty lake under an
  isolated `NORD_HOME`.
- **Non-ASCII generation.** `respond()` already forces `core.encode("ascii", "ignore")`. The organ's own
  prints must be ASCII at the source, not rescued downstream.

## Requirements

- **FR-001**: The comparison metric MUST be an instrument that already exists in this repository. Per P2,
  if the two mechanisms cannot be compared by an existing benchmark, the swap is not P2-eligible and the
  answer is to build the benchmark first. The instrument is `reasoning/blimp_bench.py`.
- **FR-002**: The metric MUST be shown able to FAIL on degenerate input, on the exact pair set the
  verdict will use, in the same invocation that produces the verdict. A metric whose admission controls
  were verified on a different pair set has not been verified.
- **FR-003**: A baseline MUST be captured BEFORE any implementation task modifies any source file, MUST
  be attributable to a named commit, and MUST abort if any language source is dirty at capture time.
- **FR-004**: The evaluation MUST include a NULL/control arm that receives the identical treatment minus
  the mechanism under test. At minimum: plasticity-disabled and grounding-shuffled. An improvement that
  the control also achieves is not attributable to the organ.
- **FR-005**: A NULL result MUST be a valid, successful terminal state. It MUST NOT trigger a retry with
  adjusted parameters, a re-chosen pair set, or a substituted metric. It stops the feature, nothing is
  deleted, and the outcome is recorded.
- **FR-006**: The organ MUST learn syntax and semantics SIMULTANEOUSLY from grounded (sentence, scene)
  exposure in a single pass over each item. A staged curriculum MUST NOT be the shipped path, and MUST be
  implemented as a control arm so the joint claim is measured rather than assumed.
- **FR-007**: No numeric literal may arbitrate the organ's behaviour (P1). Area size, per-area cap,
  synapse density, plasticity rate, and the number of projection rounds MUST each be derived from
  existing state, from the exposure the organ has actually received, or learned. Projection MUST
  terminate on a detected convergence condition, never on a fixed count. Each derived quantity MUST name
  its derivation at its definition site.
- **FR-008**: Learning MUST be local. Hebbian and error-modulated-Hebbian updates only. No
  `loss.backward()`, no autograd, no batch corpus gradient descent, no weight transport. Per the standing
  rule, any construction that reaches for backprop pivots to e-prop instead.
- **FR-009**: The organ MUST be constructed through the existing `Brain._lm_serve` slot and the brain
  through `make_brain()` (P4). No second `Brain`, no second durable store.
- **FR-010**: On an ADOPT verdict the inferior mechanism MUST be DELETED, not flag-gated (P2). If two
  implementations of the reply-generation job remain reachable when the feature closes, it was not a
  merge and the requirement is unmet.
- **FR-011**: Every unique capability of a retired module MUST be ported before deletion, or recorded as
  deliberately dropped with the reason. `expose()`-from-live-conversation and `language_acq.npz`
  persistence MUST each be adjudicated by name.
- **FR-012**: `cortex/spiking_lm_serve.py` MUST be resolved to one of P3's three dispositions - wired,
  retired, or exempt with a stated reason - independently of the organ's verdict, and the decision MUST
  report its consumer count.
- **FR-013**: Both scoring conventions MUST be reported side by side (unnormalized sum-logprob and
  per-character normalized), as the incumbent harness already does. Reporting only one silently makes
  every prior recorded number incomparable.
- **FR-014**: An arm that cannot score a sentence MUST tie that pair, and a tie MUST count as wrong.
  Credit MUST NOT be awarded for having no opinion.
- **FR-015**: Scoring MUST be strictly read-only. No plasticity may fire during evaluation in any arm.
- **FR-016**: The verdict MUST be recorded through `tools/coverage_score.py record --kind ab` so the band
  ceiling derives from the recorded exit code, and the recorded command MUST be `--rerun`-able (a
  `python -c` one-liner is not, because `rerun` splits on whitespace).
- **FR-017**: The recording harness MUST live in `tools/` and be tracked. `scratchpad/` is gitignored and
  `tools/citation_guard.py` gates a tracked file citing an untracked `.py` as a runnable reproduction
  path.
- **FR-018**: Two organs constructed with the same seed MUST produce identical output. Randomness MUST
  come from a per-instance `random.Random(seed)` or a per-instance numpy `Generator`; module-level
  seeding is forbidden.
- **FR-019**: All counts MUST be printed as `N of M`. All output MUST be ASCII.
- **FR-020**: The three gates MUST be green before and after every slice - `python run_selfchecks.py`,
  `pytest` at `--cov-fail-under=70`, `python -m ruff check .` and `ruff format --check src tests tools` -
  with no assertion loosened and no check silenced to make it pass (Standing Invariant 2).
- **FR-021**: The organ's live firing density MUST be measured and reported against the arXiv 2607.26648
  bound `rho >= H_b^{-1}(log2 M / H)` for the organ's actual memory load. No sparsity, density, or energy
  target may be an acceptance criterion for this feature.
- **FR-022**: The change in `Brain.respond` wall-clock latency MUST be measured and reported whichever
  way it falls. A regression is a result, not a reason to hide the number.
- **FR-023**: The pair-set restriction rule MUST be a stated property of the pair set, fixed and
  committed BEFORE either arm is scored, and applied identically to both arms.
- **FR-024**: The organ MUST be on the live reply path if adopted (P3). A generator that changes no reply
  is dormant, not progress; if it cannot pass the coherence guard, the honest report is that it was not
  adopted.
- **FR-025**: The Language coverage ledger MUST be updated to reflect the measured outcome, including a
  DOWNGRADE or an unchanged number if that is what was measured. A row MUST NOT be deleted to raise a
  score.
- **FR-026**: `length_scorer` MUST be reported alongside the other admission controls in the same
  invocation as the verdict, and MUST be named explicitly in the ADOPT criteria. Rationale, from the
  measured mechanism (F3a): BLiMP's grammatical member is usually the longer one and `length_scorer`
  scores 29.4%, so roughly six points of apparent sum-logprob gain are available from correcting the
  length bias alone - more than the entire per-character competence band. **A scorer that merely handled
  length correctly would therefore look like a language win under sum-logprob.** An organ whose
  sum-logprob advantage over the incumbent is not also present under per-character normalization MUST NOT
  be reported as a language improvement; it is a length-bias correction and MUST be named as one.

### Added 2026-08-02 by owner scope change - SNNLP (arXiv 2401.17911). Full reasoning in `research.md` R10.

- **FR-027**: **BLiMP remains the sole arbiter.** SNNLP MUST NOT replace `reasoning/blimp_bench.py` as
  the deciding benchmark. The `ADOPT` / `NULL` / `NOT-P2-ELIGIBLE` criteria, the frozen pair set, the
  four admission controls and the char-bigram bar are unchanged by this addition. Rationale, recorded
  so it is not re-proposed: SNNLP measures a different thing, its downstream task **cannot fail an
  admission control on syntax** because sentiment is largely lexically solvable, and substituting it
  would orphan every recorded BLiMP figure in `docs/DEPTH_CEILINGS.md` Ceiling 2.
- **FR-028**: The `PHON` area's text-to-spike encoding MUST be decided by a measured A/B, not by a
  citation. The arms are: **SNNLP's proposed encoding**, **Poisson rate-coding** (the paper's own
  comparison, which is what makes this a P2 adopt question rather than a preference), and the two
  encodings that already exist in this tree - `cortex/language_acquisition.py::_encode_ctx`'s
  position-tagged grapheme fold and `cortex/population_encoding.py::PopulationEncoder`. All arms feed
  an otherwise identical `PHON`, on identical exposure, order, split and seed. **If an existing
  encoding wins, it is kept and SNNLP is recorded as measured-and-not-adopted.**
- **FR-029**: Sentiment MUST be a **second, non-deciding capability metric**. **If the BLiMP A/B
  returns `NULL`, the verdict is `NULL` whatever sentiment says**, and this MUST be enforced in the
  harness - the verdict is computed from the BLiMP block alone and the sentiment block is printed
  after the verdict line, never as an input to it. Sentiment is reported as a demonstrated capability
  and MUST NOT be reported as evidence about syntax.
- **FR-030**: The sentiment arm MUST print a **bag-of-words control inline**, for the same reason the
  char-bigram control is mandatory on BLiMP: sentiment is largely lexically solvable, so an organ that
  does not beat a bag of words has demonstrated nothing and its sentiment number is not evidence of
  anything either. That MUST be stated in the output rather than the number quietly dropped. The
  corpus is **SST-2**; the one-time download command lives in the loader's module docstring and the
  data is NOT committed, following `reasoning/blimp_bench.py`'s precedent exactly (`data/` is
  gitignored, `.gitignore:43`). The loader lives beside the organ in `cortex/`. **If the download is
  unavailable or the licence is unclear, the work STOPS and reports; a self-authored substitute
  corpus is forbidden.**

## Key Entities

- **Area** - a population of excitatory units with a per-area cap (local inhibition selecting the top-k
  most-driven units), recurrent random synapses within it, and random synapses to and from other areas.
  Carries a plastic weight structure updated by Hebbian coincidence only.
- **Assembly** - the stable set of units that remains active in an area after projection converges. The
  organ's representation of a word, a role, or a scene element. Identified by its unit set, not by a
  label.
- **Projection** - firing an assembly in one area and letting the cap select a response assembly in
  another, with Hebbian strengthening on the co-active synapses, repeated until the response assembly
  stops changing. The single primitive the whole organ is built from.
- **Inter-area inhibition schedule** - which areas are allowed to project into which at each step of
  processing a sentence. This is the sixth principle of arXiv 2507.11788 and is what makes word order and
  syntactic role learnable rather than stipulated.
- **Grounded exposure item** - a (word sequence, scene) pair. The scene is a referent structure drawn
  from the existing knowledge base, not a new corpus.
- **Held-out split** - grounded items never exposed, used for the role and referent measurements.
  Positional and deterministic, so the split does not depend on a shuffle nobody recorded.
- **Verdict record** - the `tools/coverage_score.py` artifact carrying kind, command, exit code, and
  commit. The exit code, not the prose, is what a band ceiling reads.

## Success Criteria

- **SC-001**: On the pair set used for the verdict, `random_scorer` lands within 3 points of 50%,
  `constant_scorer` scores 0%, and `length_scorer`'s leakage is printed - all in the same invocation as
  the headline number.
- **SC-002**: `specs/005-language-organ/baseline.txt` names a commit and every recorded command
  reproduces its recorded figure when re-run at that commit.
- **SC-003**: The restricted pair set's size is reported as `N of M` pairs over `P of 67` paradigms, and
  the rule producing it is committed before either arm is scored.
- **SC-004**: On held-out grounded items, the organ's syntactic-role accuracy exceeds the
  most-frequent-role baseline the same harness computes, and the plasticity-disabled and
  grounding-shuffled arms do not.
- **SC-005**: The joint-versus-staged contrast is reported with its direction and magnitude, whichever
  way it falls.
- **SC-006**: The head-to-head emits both arms under both scorings on one pair set with the controls
  inline, and exits with a code that distinguishes ADOPT from NULL.
- **SC-007**: If ADOPT: `cortex/language_acquisition.py` no longer exists, exactly one reply generator is
  reachable from `Brain.respond`, and the three gates are green with nothing loosened.
- **SC-008**: If NULL: nothing is deleted, the organ is not wired into `respond()`, the outcome is
  recorded as a null, and the feature is reported COMPLETE rather than blocked.
- **SC-009**: The organ's measured firing density is reported alongside the derived 2607.26648 floor for
  its memory load, and the report states plainly whether the organ is dense.
- **SC-010**: `Brain.respond` latency before and after is reported in the same units, and any regression
  is stated in the summary rather than only in an appendix.
- **SC-011**: `cortex/spiking_lm_serve.py` has a recorded disposition with a consumer count.
- **SC-012**: Two same-seed organs produce identical output, asserted by a test.
- **SC-013**: `docs/coverage/Language.md` reflects the measured outcome, up, down, or unchanged.
- **SC-014**: The `PHON` encoder A/B reports all four encodings on identical exposure with the winner
  named, and the winner is whichever measured best - including when that is an encoding this repo
  already had.
- **SC-015**: The sentiment block prints below the verdict line, prints the bag-of-words control
  inline, and the harness demonstrably cannot turn a `NULL` into an `ADOPT` - asserted by a test that
  forces a `NULL` BLiMP block alongside a maximal sentiment block and checks the verdict is still
  `NULL`.
- **SC-016**: If SST-2 is absent, the sentiment block SKIPs with a stated reason and exit 0, and the
  skip can never be read downstream as a passed capability - the same treatment absent BLiMP data
  gets.

## Assumptions

1. **The scene source is the existing knowledge base.** `reasoning/world_kb_stream.py::load_kb()`
   triples supply referents and `reasoning/rose_language.py::ground()` supplies the word-to-referent
   resolution, rather than a new corpus. A triple is a two-argument scene. Recorded as Q2.
2. **The organ is word-level, the incumbent is character-level.** BLiMP scoring is defined over
   sentences, so both can produce a sentence-level score; the comparison is at the sentence level and
   the internal granularity difference is not itself disqualifying.
3. **The restriction rule is lexical coverage.** A pair is in the set when every word of both its
   sentences is in the organ's acquired lexicon. This is a property of the pair set and of exposure, not
   of either arm's score. Recorded as Q1.
4. **"Better" means strictly higher accuracy on the restricted set under BOTH scorings, and above the
   char-bigram control.** A win under one scoring and a loss under the other is not a win.
5. **The organ replaces `_lm_serve`'s payload, not `respond`'s structure.** The Gf-fluid branch, the
   coherence guard, and the narrator fallback all stay. This keeps the change surgical and keeps the
   fallback that makes a bad generation harmless.
6. **`reasoning/rose_language.py` is not retired by this feature.** It provides constituency parse and
   grounding, which the organ consumes rather than duplicates; it is a supplier, not a competitor.
   Recorded as Q3.
7. **CPU only.** The organ's areas at the scale a modest grounded corpus supports are far below this
   repo's measured CUDA crossover (N ~ 24k). P5 is satisfied by staying on CPU and saying why.
8. **The venv is `.venv-lake\Scripts\python.exe`**, not the `python` on PATH.
9. **BLiMP data stays on disk.** If it is absent the harness SKIPs, and a skip is never a pass.
10. **The daemon may be running.** Every recorded probe isolates `NORD_HOME`.

## Out of Scope

- Fluent open-domain generation. The organ's literature basis learns nouns, verbs, meanings, order, and
  novel sentence generation from modest grounded exposure. That is not GPT-level fluency and this feature
  does not claim a path to it.
- Retiring `reasoning/rose_language.py`, `reasoning/pragmatics.py`, `cortex/interaction_learning.py`, or
  the e-prop LM serves. Each is a separate P2 or P3 adjudication with its own evidence.
- Deleting `models/spikegpt/SpikeGPT-216M.pth` (861,716,013 bytes, no `.py` in the tree references it).
  Recorded here because it is dead weight found while working; it is not this feature's to remove.
- Any sparsity or energy optimization. See FR-021.
- Band > 80 coverage claims. F14 makes them unreachable through the sanctioned path.
- Scaling the organ beyond what a grounded corpus in this repo supports. That is a follow-on, and the
  honest framing is that scale is the next question, not this one's answer.

## Dependencies

- `reasoning/blimp_bench.py` and `data/lang/blimp/` - the instrument and its data. Present.
- `tools/blimp_baseline.py`, `tools/lang_scale.py`, `tools/lang_diag.py`, `tools/lang_probe.py` - the
  recorded baseline and diagnosis harnesses. Present and tracked.
- `reasoning/world_kb_stream.py` and `reasoning/rose_language.py::ground()` - grounding. Present.
- `tools/coverage_score.py record`/`rerun` - the artifact ledger. Present; F14 limits it to `kind: ab`.
- `tools/charge_probe.py` - has no language-eligible source (F6). Adding one is in scope for making the
  result reportable above band 40; the feature's verdict does not depend on it.
- `cortex/predictive_coding.py` - consumed by the incumbent. Retiring the incumbent must not orphan it;
  it has other consumers and this must be confirmed, not assumed.
- Another agent is authoring `specs/006-*` concurrently and HEAD is moving. Every figure is pinned at
  T001; nothing outside `specs/005-language-organ/` is touched by this specification.

## Open questions

Recorded verbatim, with the assumption the plan proceeds on. None blocks planning.

**Q1 - Can the organ be scored on BLiMP at all, given that its lexicon comes from modest grounded
exposure and BLiMP's vocabulary does not?**

*Assumption*: yes, on a lexically-covered subset, with the restriction rule fixed before scoring and
applied identically to both arms (Assumption 3). **If the covered subset is too small to carry a verdict
- the plan sets the floor at a stated minimum pair and paradigm count - then per P2 the swap is not
P2-eligible and the correct outcome is to say so, not to substitute a metric that happens to fit.** This
is the question most likely to end the feature, and it is answered in Phase 0 before any organ code is
written.

**Q2 - Is the knowledge-base triple set a sufficient scene source, or does grounded acquisition need
scenes with attributes and events rather than binary relations?**

*Assumption*: the triple set is sufficient for the noun/verb and word-order claims, because a
`(head, relation, tail)` triple already provides two arguments and a relation - the minimal structure the
verb-versus-noun distinction needs. If it proves insufficient, the honest fallback is a smaller claim, not
a hand-written corpus that flatters the organ.

**Q3 - Is `reasoning/rose_language.py` a P2 competitor to the organ, or a supplier to it?**

*Assumption*: supplier. It provides constituency and KB grounding; the organ provides acquisition and
generation. But its `comprehend()` returns `{'known': False, 'reason': 'ungrounded'}` for every ordinary
English sentence tested, and the organ will also produce a comprehension judgement - which is two
mechanisms for one job and therefore a P2 question. It is deferred rather than answered, and deferring it
is itself recorded so the next reader does not mistake silence for resolution.
