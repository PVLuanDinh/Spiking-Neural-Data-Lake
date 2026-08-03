# Phase 0 Research: The Brain-Native Assembly-Calculus Language Organ

**Feature**: `specs/005-language-organ` | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

Every figure quoted from this repository is quoted from the tracked document that recorded it, with the
citation. Nothing here was re-measured during Phase 0, and nothing here is attributed to a commit that
was not verified - HEAD moved during authoring because a second agent is committing concurrently, so
T001 re-pins the baseline before any implementation figure is taken. Where this document says MEASURED,
it means measured by a prior recorded pass and cited; where it says DERIVED, it means computed here from
cited figures; where it says OPEN, it means nobody has measured it and the plan schedules the
measurement.

---

## R1. Can the existing benchmark express the organ at all?

**This is the question that can end the feature, so it is first.**

P2's guard is explicit: *"'Better' requires a measurement that exists. If the two mechanisms cannot be
compared by any benchmark in this repo, the swap is not P2-eligible and the answer is build the
benchmark first, not pick the more fashionable one."* The worked case in `docs/POLICIES.md` is the
Thousand Brains proposal to replace the Global Workspace, refused because no consensus benchmark existed.
An assembly-calculus organ replacing a character LM is the same shape of proposal and gets the same test.

### What the instrument is

`reasoning/blimp_bench.py`. 67 paradigms x up to 1000 minimal pairs; the two sentences of a pair differ
in exactly one grammatical feature and share nearly every word. It is automatic, needs no LLM judge,
cannot be gamed by fluency, and cannot be passed by a template bank. `data/lang/blimp/` holds all 67
`.jsonl` files. Anchors it states itself: chance 50.0%, human agreement 88.6%, word-level 5-gram 61.2%,
GPT-2 large 86.0%, and - the right yardstick for character-level models on this repo's own exposure - a
**char 5-gram at 53.0%, held-out char PPL 4.56**.

### The two structural facts that decide the question

1. **A scorer is a function from sentence to a real number.** `score()` (`blimp_bench.py:73`) takes any
   callable. It does not require a log-probability, only a monotone acceptability ordering. So an organ
   that emits a convergence-quality number rather than a logprob is admissible in principle.
2. **An unscorable sentence loses.** `UNSCORABLE = -1e18` is the sentinel; `cached_logprob` maps a `None`
   return to it, and `per_char` passes it through *unnormalized* on purpose, so a pair both of whose
   sentences are unscorable stays a tie, and ties count as wrong. The docstring states the reason:
   normalizing the sentinel would award credit for having no opinion, decided by sentence length.

Fact 2 is the danger. BLiMP's vocabulary is thousands of words. An organ whose lexicon comes from modest
grounded exposure will return nothing for most pairs, tie them, and lose to the incumbent **by
construction rather than on merit**. That is not a fair A/B; it is an A/B the organ cannot win no matter
how good it is.

### Decision

**Score both arms on a lexically-covered subset, with the restriction rule fixed and committed before
either arm is scored, and refuse the comparison if the subset is too small.**

The rule: a pair is IN the set when every word of both its sentences is in the organ's acquired lexicon.
This is a property of the pair set and the exposure - not of either arm's score - so it cannot be tuned
after the fact. It is applied identically to both arms: the incumbent character LM is scored on exactly
the same pairs, not on the full 67 paradigms.

**The refusal floor, stated before the measurement so it cannot be moved afterwards:**

| quantity | floor | rationale |
|---|---|---|
| paradigms with at least one covered pair | **>= 8 of 67** | a verdict resting on fewer than eight grammatical phenomena is a verdict about one phenomenon |
| covered pairs total | **>= 200 of the restricted universe** | see the derivation below - it is set on the PAIRED difference, not on a single proportion |
| degenerate-control behaviour on the restricted set | all four must fail as specified | R2 below |

**Derivation of the 200-pair floor, re-derived against the live numbers (see R2a) rather than the
withdrawn 51.0%.** The differences this instrument must resolve, in the **per-character** convention
where the length bias is removed, are: e-prop flat LM **50.4**, e-prop ALIF-LM **51.9**, char bigram
**52.8**, char 5-gram **53.0**. That is the real working band - **four mechanisms inside 2.6 points**.

Two corrections to how this floor was first justified, both of which change the reasoning and neither of
which changes the number:

1. **The relevant statistic is the paired difference, not an independent proportion.** The first draft
   cited "standard error ~3.5 points at n=200", which is `sqrt(0.25/200)` - the SE of a *single*
   proportion measured independently. That was the wrong statistic and it was also internally
   inconsistent, since 3.5 points cannot resolve a 2.6-point band. Both arms score the **identical**
   frozen pair list (contract B1), so the comparison is paired and the relevant quantity is the SE of
   the *difference*, which depends on the discordance rate and is substantially smaller than the
   independent-proportion SE. 200 pairs is set on that basis. **The harness must report the paired
   discordance count as `N of M`**, because the difference's SE cannot be read off the pair count alone
   and a difference reported without it is not interpretable.
2. **The floor must not be sized on the sum-logprob spread.** Under sum-logprob the same four
   mechanisms span 46.9 to 52.8 - about 6 points, looking easier to resolve. Most of that spread is
   length leakage (R2a), so sizing the floor on it would be sizing the instrument on its own bias. The
   per-char band is the honest one and is what the floor is set against.

The floor is therefore **scoring-convention-dependent by design**: it is derived from the per-char band
and applies to both conventions, because a set too small to resolve the bias-free difference is too
small regardless of which number is printed first.

**If either floor is missed, the outcome is: the swap is not P2-eligible, recorded as such, feature
stops.** Not "widen the corpus until it passes" - that is choosing the pair set to get a verdict, which
is the p-hacking the rule exists to prevent. Widening exposure is a legitimate separate decision, taken
before the restriction is computed and never in response to it.

### Rationale

The alternative readings were each worse:

- **Score on all 67 paradigms.** Honest but uninformative: it measures lexicon size, not grammar, and
  guarantees a null for reasons unrelated to the architecture. It would produce exactly the wrong
  conclusion - "assembly calculus does not work" - from an experiment that never tested it.
- **Build a new benchmark.** P2 permits this ("build the benchmark first"), but a benchmark authored
  alongside the mechanism it scores is the weakest possible evidence, and this repo has the worked case:
  `probe_fluid.py` saturated and is now banned as a signal. Using the existing instrument on a stated
  subset keeps the instrument adversarial.
- **Use the grounded held-out metrics as the deciding benchmark.** They do not exist yet, so per P2 they
  cannot decide. They are supporting evidence (R3) and are labelled as such everywhere.

### Alternatives considered for the restriction rule

- *Paradigms hand-picked as "syntactic rather than lexical"* - rejected: a judgement call made by whoever
  wants the result.
- *Pairs where the differing word is in the lexicon, other words substituted* - rejected: substitution
  changes the pair, and the whole value of BLiMP is that the two sentences differ in exactly one feature.
- *Coverage at 90% of words rather than 100%* - rejected as a first choice because it introduces a
  threshold that arbitrates the result, which is the P1 pattern. **OPEN**: if 100% coverage misses the
  floor, a coverage fraction derived from the exposure's own type-token statistics is the next candidate,
  and it must be derived, committed, and justified before scoring - not tried until something passes.

### Open

Whether the covered subset clears the floor is unmeasured. It is T004, before any organ code exists.

---

## R2. The metric's admission control, on the restricted set specifically

**MEASURED, in the sense that the controls already exist and already fire** - `blimp_bench.py` ships
three degenerate scorers whose failure is the point:

| control | site | must |
|---|---|---|
| `random_scorer` | `:106` | land near 50%. One seeded `random.Random(0)` bound at definition time, so the control is reproducible across calls - the `noqa: B008` on that line documents that the mutable default is deliberate |
| `constant_scorer` | `:111` | score **0%**. No opinion; ties count as wrong |
| `length_scorer` | `:116` | expose length leakage. Returns `-len(sentence)`; BLiMP's grammatical member is usually longer, so this reveals how much of any score is surface |

### Decision

**Run all three on the restricted pair set, in the same invocation that produces the verdict, and print
them inline with the headline number.** Add one organ-specific control: **an organ constructed with the
same seed but never exposed** (no plasticity ever applied) must score at chance. That control catches the
failure mode this repo names *movement is not activation* in reverse - a readout that scores well from
the random synapse graph alone, before any learning, means the graph and not the acquisition is carrying
the result.

Verifying the controls on the *full* set and then reporting a headline from the *restricted* set would be
the classic accidental green light. A subset can reintroduce a length correlation the full set averages
out. So the controls run on the exact set the verdict uses, and `length_scorer`'s number on the
restricted set is printed - not asserted to a threshold, printed, because its value IS the size of the
leak and a threshold would hide a growing one.

### Rationale

Standing Invariant 1: count a gate's output before any rule leans on it. The rule that will lean on this
gate is FR-001's "better on the existing benchmark", and it must not lean on a benchmark whose honesty
was checked somewhere else.

## R2a. The 51.0% incumbent is withdrawn, and why the replacement number is below chance

Recorded here so the next reader does not re-derive it, and so the ADOPT bar is stated against the right
number.

`docs/DEPTH_CEILINGS.md:555` is titled *"Correction 2 - the 51.0% incumbent is stale, and the live number
is below chance for a localized reason"* and ends: *"51.0% must stop being cited as the current
incumbent."* Measured live over 67 paradigms x 25 pairs = 1675, what `blimp_bench` actually loads today
scores:

```
e-prop ALIF-LM (preferred) : 46.9% sum-logprob | 51.9% per-char-normalized
e-prop flat LM (fallback)  : 48.3% sum-logprob | 50.4% per-char-normalized
```

### The mechanism, because "below chance" invites the wrong conclusion

Below chance is **not** a weaker-than-random model. It is a systematic, localized bias:

1. BLiMP's **grammatical** member is usually the **longer** sentence.
2. An unnormalized sum-logprob pays a roughly constant cost per character.
3. So a weak LM systematically prefers the **shorter** member - which is the **wrong** one.

The size of the effect is measured, not inferred: `length_scorer` - which does nothing but prefer the
shorter sentence - scores **29.4%**, i.e. preferring the shorter sentence is wrong about 70% of the time.
Dividing by sentence length flips both models back to roughly chance (+5.0 and +2.1 points). A
sub-chance sum-logprob score is therefore evidence of a length-dominated scorer, not of a model that has
learned anti-grammar.

### Two consequences this feature is bound by

- **Both scorings are mandatory** (FR-013), and this is why. A single-convention report of any of these
  numbers is uninterpretable.
- **A scorer that merely handled length correctly would look like a language win under sum-logprob.**
  Roughly six points are available from fixing the bias alone - larger than the entire per-char
  competence band of 2.6 points. This is the most likely way this feature manufactures a fake capability,
  and it is what FR-026 and control four exist to catch.

### One precision worth keeping, since the two figures come from different harnesses

The withdrawn 51.0% was `tools/blimp_baseline.py`'s **acquired-LM** arm - `cortex/language_acquisition.py`
itself. The live figures above are the **e-prop serves** in `blimp_bench.SERVES`, which is a different
harness scoring different models. The withdrawal is authoritative regardless, and the document states it
without qualification. But it means **no current, trustworthy figure exists for
`cortex/language_acquisition.py` specifically** - which is exactly why **T014 measures the incumbent arm
on the frozen pair set rather than quoting any document**, and why no task in this feature is permitted to
substitute a quoted number for that measurement.

### Two scorings, both reported

Non-negotiable and already the incumbent harness's behaviour. An unnormalized sum-logprob carries a
length bias; per-character normalization removes it but changes what every previously recorded number
means. `blimp_bench.main()` prints both side by side, and the module docstring says switching silently
"would make every number this bench printed before incomparable without saying so". Both arms, both
scorings, one table. A win under one and a loss under the other is **not** a win (spec Assumption 4).

---

## R3. The architecture, and what it maps onto in this tree

### Source

- arXiv **2306.15364**, *The Architecture of a Biologically Plausible Language Organ*: stylized neurons,
  synapses, brain areas, and plasticity, plus simplified sensory perception; learns nouns, verbs, and
  their meanings from a modest number of GROUNDED sentences, by Hebbian plasticity, with no
  backpropagation.
- arXiv **2507.11788**, *Simulated Language Acquisition in a Biologically Realistic Model of the Brain*:
  the same lineage (Assembly Calculus; Papadimitriou / Vempala / Mitropolsky). Six principles -
  excitatory neurons, brain areas, random synapses, Hebbian plasticity, local inhibition, and
  **inter-area inhibition**. From tabula rasa, in any language, learns word semantics, syntactic role
  (verb vs noun), word order, and generates novel sentences.

### Decision: one primitive, four areas, one schedule

**The primitive is `project`.** Fire an assembly in a source area; every unit of a target area receives
the summed weight from active presynaptic units; local inhibition keeps only the top-k most-driven units
(the **cap**); Hebbian plasticity strengthens synapses whose two endpoints were both active; repeat.
Convergence is when the selected unit set stops changing. Everything the organ does - binding a word to a
meaning, assigning a role, encoding order, generating - is projection under a different schedule of which
areas may fire into which.

**Local inhibition is the cap.** It is not a separate mechanism to build. This matters for P1: the cap is
`k`, and `k` is therefore a behaviour-arbitrating quantity that must be derived, not literal (R5).

**Inter-area inhibition is the schedule.** It is the sixth principle and the one that makes syntax
learnable rather than stipulated: at each step of reading a sentence, only some areas are disinhibited,
and which ones are open is what the organ learns. A design that hardcodes the schedule has hardcoded the
grammar and is not acquiring anything - this is the single most important honesty trap in the
architecture, because a hand-written schedule produces correct-looking output for the wrong reason.

**Four areas, and no more without evidence.** Following the lineage and the ponytail rule that the best
code is the code never written:

| area | role | grounded by |
|---|---|---|
| `PHON` | the word form as presented; the sensory entry point | the word string, folded to the existing inventory |
| `LEX` | the acquired lexical assembly - what the organ knows a word to be | `LEX <-> PHON` and `LEX <-> SCENE` reciprocal projection |
| `SCENE` | the referent structure of what the sentence is about | `reasoning/world_kb_stream.py` triples (R6) |
| `SEQ` | position and role; where word order and verb-vs-noun live | the disinhibition schedule over the sentence |

A fifth area is added only if a measurement shows the four cannot express something. Adding areas until
the number improves is fitting the architecture to the benchmark.

### What already exists in this tree that the organ must NOT duplicate (P2)

- **Local Hebbian / R-STDP update rules** - `cortex/language_acquisition.py` has
  `dW = lr * outer(prediction_error, presynaptic_latent)`; `reasoning/language_self_build.py` has an
  R-STDP next-char readout; `cortex/interaction_learning.py` has an R-STDP spiking intent router. The
  organ must reuse a rule, not author a fourth. Which one is a plan-time decision recorded in
  `contracts/language_organ.md`.
- **Spike-polarity hypervectors and role-filler binding** - `cortex/vsa_reasoning.py::hv` and
  `reasoning/rose_language.py`'s `bind(WORD, ENTITY)`. The organ's `SCENE` grounding is a consumer of
  this, not a reimplementation.
- **The neuron zoo** - `cortex/neuron_zoo.py` is the repo's one heterogeneous population behind a single
  contract, and it *"does NOT re-implement any dynamics -- it IMPORTS"*. An area's units go through it
  unless a measurement shows they cannot.

### Alternatives considered

- **Keep the character LM and add areas around it.** Rejected: additive, and it preserves the linear-map
  bottleneck that is the proven cause (`docs/DEPTH_CEILINGS.md`).
- **Widen the existing readout.** Already refuted by measurement: a random sparse expansion to 8192 dims
  (278,528 synapses) made it *worse* - BLiMP 52.3-53.2%, PPL 26.75-30.92.
- **Train a spiking transformer.** Violates the standing e-prop rule and the project's own framing; also
  `models/spikegpt/SpikeGPT-216M.pth` (861,716,013 bytes) already sits on disk from the last time this
  was tried, with no `.py` in the tree referencing it, and `run_selfchecks.py:362` records SpikeGPT as
  scrapped in v3.77.

  **Citation drift found while verifying this, and corrected here rather than propagated**:
  `docs/coverage/Language.md` and `docs/DEPTH_CEILINGS.md` both cite this record at
  `run_selfchecks.py:343`. That line is now `"reasoning.godel_machine"`. The SpikeGPT record is at
  **`:362`**. Two drifted citations in the very documents this feature's evidence rests on is the
  defect `tools/citation_guard.py` exists to report, and T060 re-runs it; the drift is noted here so
  the next reader of those two files is not misled by a line number that resolves to something else.

---

## R4. Joint inference: why the staged version is a control arm and not a fallback

### Source

arXiv **2406.11977**, *Reframing linguistic bootstrapping as joint inference*: syntactic bootstrapping
(using syntax to learn word meaning) and semantic bootstrapping (using meaning to learn syntax) are the
SAME mechanism, and both effects are strongest when syntax and semantics are learned SIMULTANEOUSLY,
each constraining the other's hypothesis space.

### Decision

**One pass per exposure item updates lexical and syntactic structure together.** No stage boundary, no
"first learn the words, then learn the grammar" curriculum, no freeze-then-thaw.

**And the staged version is built as an arm and run**, because "we did the joint thing and it worked" is
not evidence that joint was why. Four arms, all on identical exposure and identical held-out data:

| arm | what it is | what it tests | expected |
|---|---|---|---|
| **joint** | the shipped organ | the feature | best |
| **staged** | semantics to convergence, then syntax | 2406.11977's prediction | worse than joint |
| **shuffled** | scenes randomly permuted against sentences | is the grounding load-bearing, or is this word statistics? | at the frequency baseline |
| **frozen** | identical construction, plasticity disabled | is the readout or the learning carrying it? | at the frequency baseline |

`shuffled` and `frozen` are the null/control arms the directive requires. `staged` is a scientific
contrast, not a control.

**The reporting rule that makes this honest: if `staged` matches `joint`, the joint claim is WITHDRAWN,
not explained.** The spec is written so that outcome is reportable rather than embarrassing. This repo's
own worked precedent is `docs/coverage/README.md`'s main-versus-null contrast table, which reports
`main better 12 of 12` and the null's `mean d = -0.000002` in the same block - the null's number is
printed whether or not it flatters the main arm.

### Rationale

This is the same discipline the coverage ledger already enforces and the same one `specs/003` used when
it reported a **+45% RMSE regression** at the live width as "the accepted, documented cost of
measurability" rather than omitting it.

### Open

Whether a staged arm can even be constructed fairly - "semantics to convergence" needs a convergence
criterion that does not itself smuggle in syntax. The plan schedules this as a design task, and if it
cannot be constructed fairly, that fact is reported instead of a contrast.

---

## R5. P1: every constant that arbitrates behaviour, and where it comes from

The Assembly Calculus literature states its parameters as literals - area size `n`, cap `k`, synapse
density `p`, plasticity rate `beta`, and a fixed number of projection rounds. **Copying them in would put
five magic numbers at the centre of the organ**, which is precisely what P1 forbids: *no numeric literal
may arbitrate behaviour*. `docs/POLICIES.md` allows a derived initial value - it is a starting point that
becomes a learned one once the population's rule moves it - and forbids a tuned constant sitting where a
computed value belongs.

### Decision table

| quantity | source | derivation |
|---|---|---|
| **cap `k`** | the exposure the organ has received | derived from the lexicon size the organ has actually acquired, so the cap grows as the organ learns rather than being sized for a corpus nobody has shown it. The repo's template for an earn-your-place size is `hippocampus/ng_snn.py`, whose capacity is monotone in data (4,512 units at 24,000 rows) |
| **area size `n`** | `k` and a separability requirement | derived from `k` and the number of distinct assemblies the area must keep distinguishable, by the standard birthday-collision argument. Not chosen; computed, and the computation is asserted |
| **synapse density `p`** | `n` and `k` | derived so the expected input drive to a unit is what the cap can discriminate. A density that makes every unit receive nearly identical drive makes the cap a coin flip |
| **plasticity rate `beta`** | observed convergence | derived from the organ's own measured convergence behaviour, in the manner the repo already uses for `substrate/signal_calibrator.py`-style self-calibration. **OPEN**: the exact derivation is a plan-time decision and must be written at the definition site |
| **projection rounds** | **not a number at all** | the loop terminates when the selected assembly stops changing. A fixed round count is a literal arbitrating behaviour and is forbidden by FR-007. A bounded maximum is permitted **only** as a non-termination guard, must be derived, and reaching it must be reported as a refusal rather than silently accepted as convergence |

### The trap this table exists to close

`docs/POLICIES.md` records the failure mode directly: a bounded maximum that is silently treated as
success is the *silent degradation* pattern - a stage fails, swallows it, writes a placeholder, exits 0,
and the tell is a constant where a computed value belongs. So the guard round-count must increment a
**refusal counter that is reported**, exactly as `specs/003`'s growth refusals are.

### Rationale, and the honest limit

P1 compliance here is real but partial, and saying so now is better than being caught later: the
*derivations* themselves encode choices (which separability requirement, which convergence criterion).
The standard the repo actually applies is that a quantity must be computed from state the system has
rather than typed in by a human who tried a few - and each derivation must name itself at its definition
site so the next reader can attack it. That is the bar, and it is met. Claiming the organ has "no
constants" would be the fake-compliance move.

---

## R6. Where the grounded scenes come from

The organ needs (word sequence, scene) pairs. The incumbent's `CORPUS` is 2,428 characters of
self-authored copyright-safe English about the brain, repeated four times, and it carries **no scene** -
it can teach word statistics and cannot teach word meaning. That is the whole point of a grounded organ.

### Decision

**Use `reasoning/world_kb_stream.py::load_kb()`.** It returns a deterministic, offline, curated set of
real `(head, relation, tail)` triples across geography, chemistry, biology, physics, astronomy,
computing, mathematics and everyday commonsense, with bounded-tail relations enumerated in `_RELATIONS`
and broader facts in `_extra_facts()`. A triple **is** a two-argument scene: `head` and `tail` are the
nouns, `relation` is the verb-like element. That is the minimal structure the verb-versus-noun
distinction requires, which is exactly the distinction 2507.11788 reports the organ learning.

Word-to-referent resolution is `reasoning/rose_language.py::ground()` (`:155`), which already resolves a
word to a KB referent or returns `None` for an **honest lexical gap - never faked**.

### Rationale

Authoring a second grounded corpus when one is in the tree is the additive move P2 forbids, and a corpus
authored alongside the mechanism it validates is the weakest possible evidence. Reusing the KB also means
the organ's lexicon is auditable: the operator can enumerate exactly which words it can ever have
acquired, which is what makes R1's coverage computation possible at all.

### Alternatives considered

- *WikiText-103* (`data/lang/wikitext/`, already on disk) - ungrounded. It is the right corpus for the
  scale question and the wrong one for the grounding question. Useful later as an exposure-scale arm; not
  the acquisition corpus.
- *A hand-written scene corpus* - rejected, above.
- *The arm/MuJoCo environment as a scene source* (`specs/004-mujoco-so101-arm`) - genuinely attractive,
  because a physically grounded verb ("lift", "move") is better grounding than a KB relation. Rejected
  for *this* feature as scope: it couples the language organ's verdict to the arm's availability. Named
  here as the strongest follow-on.

### Open

Q2 from the spec: whether binary relations suffice, or whether acquisition needs scenes with attributes
and events. Unmeasured. The stated fallback if they do not suffice is a **smaller claim**, not a
hand-written corpus that flatters the organ.

---

## R7. The sparsity ceiling: what it does and does not forbid

### Source

arXiv **2607.26648**, *The Sparsity Ceiling*. A recurrent spiking language model cannot go below roughly
50% firing rate; the recurrent state must stay active to carry information. The bound:

```
rho >= H_b^{-1}( log2(M) / H )
```

where `H` is the number of recurrent units, `M` the number of states the recurrent state must
distinguish, `H_b` the binary entropy function. The floor RISES with memory load and with task
difficulty. It is an informational limit, not an architectural one - no cleverer architecture evades it,
because it is a counting argument about how many distinguishable patterns a population at density `rho`
has.

### The plain statement this spec is required to make

**Two separate things are commonly conflated here, and both must be said.**

**(1) Compute is dense, unconditionally, and this organ will be slower.** A projection is a dense
random-matrix product over an area followed by a top-k selection, iterated to convergence, once per word,
per area pair open in the schedule. Sparse *activity* does not make that cheap: the matrix is dense
regardless of how few units end up selected. The incumbent it replaces is a 34x88 matrix-vector product -
2,992 parameters. **Replacing that with this will make `Brain.respond` slower, and the number must be
measured and stated (FR-022), not discovered by whoever notices the daemon lagging.** There is no
sparsity or energy target in this feature's acceptance criteria (FR-021), and none may be added later
without re-deriving the bound.

**(2) The activity-density floor is low here only because the organ carries little.** DERIVED, from the
bound: the floor scales with `log2(M) / H`. For an open-domain next-token model, `M` is enormous and the
paper's ~50% follows. For this organ, `M` is the number of sentence states it must distinguish - roughly
lexicon size to the power of sentence length, times the role assignments - which is many orders of
magnitude smaller, while `H` (units per area) is large. So `log2(M)/H` is small, `H_b^{-1}` of it is
small, and a cap-k density well under 1% is **permitted by the bound**.

That is not a free win. It is the same fact stated from the other side: **the organ can be sparse
precisely because it is doing less than a language model.** Anyone who later scales it toward open-domain
prediction raises `M`, raises the floor, and loses the sparsity - and will be tempted to report the
sparsity anyway because it was true at the smaller scale.

### Decision

**Compute the floor and check it, rather than quoting the paper.** FR-021 requires the organ's measured
firing density to be reported alongside `H_b^{-1}(log2 M / H)` computed for the organ's actual `M` and
`H`. This turns the bound from a decorative citation into a live check that can fail: if measured density
sits *below* the derived floor, the organ is losing information it needs and the anomaly is real, not a
success.

### Rationale

This repo's recurring failure is a citation that cannot go red. A bound quoted in a docstring changes
nothing; a bound computed each run and printed next to the measurement is an instrument.

---

## R8. The P2 retire decision, and what must be ported first

### The candidates, and what each uniquely provides

From `docs/coverage/Language.md`'s capability inventory, which was written for exactly this purpose -
*"Policy 2: what provides what, before anything is retired"*:

| module | unique capability | live consumer | disposition |
|---|---|---|---|
| `cortex/language_acquisition.py` | online spiking predictive-coding char LM; `expose()` learns from live turns; `generate()` produces the reply | **`Brain._lm_serve` / `_language`** | **the competitor.** ADOPT-or-NULL on R1's A/B |
| `cortex/spiking_lm_serve.py` | the only torch-free `snn_lm/1` reader | **none** - only `training/spiking_lm.py`'s transfer check and its own self-check | **dormant regardless.** Separate P3 adjudication (FR-012) |
| `cortex/self_model.py::SelfNarrator` | deterministic state-grounded generation; the only always-coherent generator | `brain.py:2932,2951,2961` | **not a competitor** - it is the fallback that makes a bad generation harmless. Keep |
| `reasoning/rose_language.py` | VSA constituency, KB-grounded `comprehend`, SLDA head | `parse` only | **supplier, not competitor** - deferred, Q3 |

### Decision on the competitor

**`cortex/language_acquisition.py` is DELETED on an ADOPT verdict, and untouched on a NULL verdict.**

Not flag-gated. `docs/POLICIES.md`'s merge-specific guard: *"'Merging' that keeps both code paths behind
a flag, a mode switch, or an `if` is additive with extra steps... Count the survivors: if two
implementations of the same job are still reachable when you are done, it was not a merge."*

**Two capabilities must be ported or explicitly dropped before the deletion (FR-011):**

1. **`expose()` from live conversation.** Every turn's text is fed back so the brain keeps learning from
   talking. The organ's equivalent is exposure to the turn plus whatever scene the turn grounds to - and
   most conversational turns ground to nothing, so this is a genuine capability question, not a mechanical
   port. If the organ cannot learn from ungrounded conversation, that must be **recorded as a dropped
   capability with the reason**, not quietly lost.
2. **`language_acq.npz` persistence** (`brain.py:2607`, `:3656`, reloaded in `_lm_serve` under
   `self._persist`). The organ needs its own persisted state or it forgets on every daemon relaunch - and
   `specs/003` measured that the daemon relaunches **~20 times in a working session**. Weights stay STATE,
   and the one durable store rule applies: exactly one `brain_lake.bin`. `specs/003` deliberately used
   `home.save_state` / `load_state` atomic JSON rather than the lake, and that precedent applies here.

Also to confirm before deletion, not assume: `cortex/predictive_coding.py` is imported by the incumbent.
It has other consumers - `PredictiveCodingNetwork.infer_batch` was promoted into `src` and
`tools/lang_scale.py` reads it - but "has other consumers" must be counted, not asserted.

### Decision on `cortex/spiking_lm_serve.py`

**Default: RETIRE. Adjudicated independently of the organ's verdict, because it is dormant either way.**

It is registered CORE (`run_selfchecks.py:83`) with a passing self-check, and it serves a hand-built
four-token demo model (`vocab=4, d_model=4`, one block) that generates `'abbbbbb'` because **there is no
trained `snn_lm/1` weight file anywhere in the repo**. A green self-check over a module that changes
nothing is the green-light-that-cannot-go-red pattern in its purest form.

The one thing that stops this being mechanical: it is the **only torch-free reader of the `snn_lm/1`
format**, so deleting it orphans `src/snn_data_lake/training/spiking_lm.py`'s export path. So the retire
task must count readers and producers and report `N of M`, and the permitted outcomes are: retire both
ends of the format together, or **exempt with a stated reason** (P3's third disposition). What is not
permitted is leaving it as-is with no recorded decision.

### The verdict-recording problem

**F6/R8 of `docs/coverage/Language.md`: no Language faculty can currently produce a tracked A/B
artifact.** `tools/charge_probe.py language 60` exits 2 - the probe neuters a key of `Brain._drive_w`
(`brain.py:633-648`) and that dictionary has no language source at all; its observables are `arm_stopped`
and `map_position`. The rubric makes a recorded-exit-code A/B the sole gate for band > 40, so three
Language rows already sit capped at 40 while carrying runtime evidence that meets the 41-60 definition.

**Decision**: the feature's verdict is recorded through `tools/coverage_score.py record --kind ab` on the
feature's own tracked harness under `tools/` (FR-016, FR-017), which is `--rerun`-able and needs no
change to `charge_probe.py`. Adding a language-eligible observable to `charge_probe.py` is the *right*
fix for the capped rows and is **explicitly not** smuggled into this feature: the coverage auditor already
recorded that doing so is "a change to the instrument, not an audit finding". It is named as follow-on
work.

Note also F14: `record` writes no `metric:`/`baseline:` header, so `kind: heldout` artifacts always fail
the scorer's F10 and **band > 80 is unreachable through the sanctioned path**. This feature is nowhere
near 80 and must not budget for it.

---

## R9. Execution order

The spec's priority order and the execution order coincide, unlike `specs/003`. Stated explicitly because
that coincidence is worth checking rather than assuming:

1. **Instrument honesty first (US1)**, because a baseline taken with a blind instrument is a number and
   not evidence, and because R1's coverage floor can end the feature before any code is written. This is
   the cheapest possible place to discover the feature is not P2-eligible.
2. **Baseline second (US2)**, at a pinned commit on a clean tree.
3. **The organ third (US3)**, with its four arms.
4. **The verdict fourth (US4)**.
5. **The retire last (US5)**, and only on ADOPT.

The one ordering subtlety: **the restriction rule depends on the organ's lexicon, which depends on
exposure, which is decided before the organ is built.** So step 1 computes coverage from the *exposure
plan* (the KB's word inventory), not from a trained organ. That keeps the rule fixed before either arm is
scored (FR-023) and is the only order in which that guarantee is possible.

---

## R10. SNNLP (arXiv 2401.17911): adopted as the PHON encoder question, REJECTED as the arbiter

Added 2026-08-02 by an owner scope change taken mid-implementation, after the paper was read. Recorded
in full - including the rejection - so nobody re-proposes the part that was refused.

### Source, and what it actually is

arXiv **2401.17911**, *SNNLP: Energy-Efficient Natural Language Processing Using Spiking Neural
Networks* (Knipper, Mishty, Sadi, Karmaker Santu, Jan 2024). **It is not a benchmark.** It is a
**text-to-spike ENCODING** paper: it compares ways of turning text into spike trains, proposes one, and
reports that the proposed encoding beats Poisson rate-coding by roughly **13%** on its downstream task,
plus hardware energy figures of roughly **32x** at inference and **60x** at training. Its downstream
task is **sentiment analysis**.

**Provenance note, applying this document's own standard.** Those three figures are quoted from the
owner directive that introduced the paper, not from a copy of the paper read in this repository and not
re-measured here. They are therefore **CITED, UNVERIFIED** - the weakest evidence class this document
uses - and no task below is permitted to lean on them. What the tasks lean on is the A/B this repo runs
itself, which is the point of running one.

### Decision 1: REJECTED as a replacement for BLiMP as the arbiter

The verdict logic does not move. `ADOPT` / `NULL` / `NOT-P2-ELIGIBLE`, the frozen pair set, the four
admission controls and the char-bigram bar are untouched. Three reasons, in the order of how much
damage the swap would do:

1. **It measures a different thing.** BLiMP measures grammatical acceptability on minimal pairs that
   differ in exactly one syntactic feature. Sentiment measures a document-level label. This feature's
   whole claim - stated in its title and in FR-006 - is about **syntax and semantics acquired jointly**.
   A sentiment score is silent about syntax.
2. **Its downstream task cannot fail an admission control on syntax.** This is the decisive reason and
   it is the same one that set the char-bigram bar. **Sentiment is largely lexically solvable**: a bag
   of words scores well on SST-2 while knowing no grammar at all. An instrument a bag of words can pass
   cannot go red on the thing being claimed, and R2/SI-1 refuse a verdict from an instrument that
   cannot go red. BLiMP's minimal pairs share almost every word between the two members, which is
   exactly what makes them immune to that.
3. **Swapping breaks comparability with the recorded baseline.** `docs/DEPTH_CEILINGS.md` Ceiling 2 is
   a BLiMP record - chance 50.0, char bigram 52.8, char 5-gram 53.0, the withdrawn 51.0, the live
   46.9/51.9 and 48.3/50.4 (R2a). Substituting the metric orphans every one of those numbers and
   leaves the feature with a "before" it cannot subtract from. `blimp_bench.py`'s own docstring makes
   the same argument about silently switching scoring conventions, for the same reason.

The tempting counter-argument - *"sentiment is a task the organ can actually do, and BLiMP is a task it
demonstrably cannot"* - is precisely why the swap must be refused. Choosing the metric the mechanism
can pass, after learning it cannot pass the other one, is the same move as widening the corpus in R1.
It is p-hacking at the level of the instrument rather than the sample.

### Decision 2: ADOPTED as the PHON area's encoder, decided by an A/B and not by the citation

The spec hand-waved how `PHON` turns a word form into spikes (R3's table says only *"the word string,
folded to the existing inventory"*). SNNLP fills that hole with something the paper itself compares, so
it is a **real P2 adopt question** rather than a preference: **SNNLP encoding versus Poisson
rate-coding, both feeding an otherwise identical `PHON`, scored on the organ's own task.**

What already exists in this tree and must not be duplicated (P2):

| existing mechanism | what it encodes | relation to `PHON` |
|---|---|---|
| `cortex/population_encoding.py::PopulationEncoder` | one continuous scalar -> Gaussian RBF receptive fields -> spike counts | a **candidate third arm**, not a competitor to retire. It is a scalar front-end; a word form is not a scalar, so using it means choosing a scalar embedding of the word first, and that choice would then be doing the work |
| `cortex/theta_gamma_code.py` | multi-item ordering within a theta cycle | a candidate for the `SEQ` schedule, **not** for text-to-spike. Named here so it is not mistaken for one |
| `cortex/language_acquisition.py` | char -> position-tagged bag of graphemes -> predictive-coding latent (`_encode_ctx`, `:83`) | the **incumbent encoding**, and therefore the honest fourth arm: it is what the repo does today |

**The A/B decides, not the citation.** If the incumbent `_encode_ctx` folding or the RBF population
code scores better on the organ's own task, that one is kept and SNNLP is recorded as measured-and-not-
adopted. This is the same rule the organ itself is held to.

**Honest limit, stated now.** The encoder A/B is a comparison **within** the organ, so it is only
meaningful once an organ exists and can be scored. It is not a fallback verdict for the feature and
cannot substitute for the BLiMP A/B.

### Decision 3: sentiment is a SECOND, NON-DECIDING capability metric, and cannot rescue a NULL

**The hard constraint, written into the harness and not only into prose: if BLiMP returns `NULL`, the
verdict is `NULL`, whatever sentiment says.** Sentiment is reported as a demonstrated capability, never
as evidence about syntax. `tools/lang_organ_ab.py` computes the verdict from the BLiMP block alone and
prints the sentiment block after it, below the verdict line.

- **Corpus: SST-2**, small, standard and free. `data/` is gitignored (`.gitignore:43`) and
  `reasoning/blimp_bench.py` sets the precedent exactly: the one-time download command lives in the
  module docstring, the loader ships, the data does not. That pattern is followed rather than invented.
- **The loader lives beside the organ** (`cortex/`), not in `reasoning/`.
- **A bag-of-words control is mandatory and printed inline**, for the same reason the char-bigram
  control is mandatory on BLiMP: sentiment is largely lexically solvable, so an organ that does not
  beat a bag of words has demonstrated nothing, and the sentiment number is then not evidence of
  anything either. It is reported as such rather than dropped.
- **If the download is unavailable or the licence is unclear: STOP and report.** Substituting a
  self-authored sentiment corpus is forbidden by name - a corpus authored alongside the mechanism it
  scores is the weakest evidence available, and this document already refuses it twice (R1, R6).

### Status at implementation time

**All of R10's measurable content is blocked behind the T011 hard stop and none of it ran.** The
encoder A/B needs a `PHON` area and the sentiment arm needs an organ; the feature stopped at
`NOT-P2-ELIGIBLE` before either existed. The rejection in Decision 1 is the part of R10 that is
*already* decided and does not need a measurement, which is why it is worth keeping on its own.

---

## Summary of decisions

| # | decision | status |
|---|---|---|
| R1 | Lexically-covered BLiMP subset, rule fixed before scoring, floors 8 of 67 paradigms and 200 pairs | floors stated; **MEASURED 2026-08-02: 0 of 67 paradigms, 0 of 67000 pairs -> NOT-P2-ELIGIBLE** (`coverage.md`) |
| R2 | Three existing degenerate controls plus an unexposed-organ control, all on the restricted set, inline with the verdict | decided |
| R3 | Four areas, one `project` primitive, learned inter-area schedule; reuse existing local rules and VSA binding | decided; which local rule is a contract-time choice |
| R4 | Joint acquisition shipped; staged / shuffled / frozen arms all built and reported | decided; staged-arm fairness OPEN |
| R5 | `k` from lexicon, `n` from `k` + separability, `p` from `n`,`k`, `beta` from convergence, rounds by convergence-detection with a reported refusal counter | decided; `beta` derivation OPEN |
| R6 | `world_kb_stream.py` triples as scenes, `rose_language.ground()` for resolution | decided; sufficiency OPEN (Q2) |
| R7 | Compute is dense and slower - stated up front; activity floor computed each run and checked, not quoted | decided |
| R8 | `language_acquisition.py` deleted on ADOPT after porting `expose()` and persistence or recording them dropped; `spiking_lm_serve.py` retire-by-default with a counted consumer report | decided |
| R9 | Instrument, baseline, organ, verdict, retire - in that order | decided |
| R10 | SNNLP (2401.17911) **rejected** as the arbiter (BLiMP stays); **adopted** as the `PHON` encoder question, decided by an A/B against Poisson rate-coding and the two existing encoders; SST-2 sentiment added as a **non-deciding** second capability metric with a mandatory bag-of-words control | rejection decided; the A/B and the sentiment arm are **blocked behind the T011 stop and unmeasured** |
| R8 | `spiking_lm_serve.py` consumer count | **MEASURED 2026-08-02: 0 of 9 references are live runtime consumers; 0 trained `snn_lm/1` files in the tree** (`verdict.md` T057) |
