# The coverage ledger — status, and how to read the number in it

`tools/coverage_score.py` re-derives the brain-faculty percentage from the eleven category files in
this directory. Run it:

```bash
python tools/coverage_score.py                 # re-derive the total, or refuse and say why
python tools/coverage_score.py --rerun drive_60  # re-execute a recorded artifact and compare
python tools/coverage_score.py record --kind ab --name drive_60 -- tools/charge_probe.py drive 60
```

## Why this exists

`COVERAGE_RUBRIC.md:8-9` says *"a percentage nobody can re-derive is not a measurement"* — and then the
baseline it governs was assembled entirely by hand. No `.py` in the tree read either document. The
rubric ended the problem for the **standard** and left it open for the **score**. This is the score
half.

The scorer has **no opinion about what band a faculty deserves.** An auditor writes that in a category
file. What the scorer does is *refuse* to total up claims the rubric's own evidence standard does not
support: band > 40 needs an A/B artifact whose **recorded exit code** says the A/B passed, band > 80
needs held-out evidence naming its metric and baseline, every scored row cites a `file:line` that
resolves, and all eleven categories must be present so the denominator cannot quietly shrink.

The anti-forgery core is that a band ceiling comes from the **recorded exit code**, not from the claim.
`tools/charge_probe.py place 60` exits 2 (NOT-ELIGIBLE); a row claiming band 65 against that artifact is
refused, and editing the row cannot change the exit code. It is not airtight — a determined forger can
hand-write a transcript, which survives `score` and dies on `--rerun` — but it removes the *accidental*
inflation this project keeps hitting.

### `record`/`rerun` now isolate `NORD_HOME` themselves, and two artifacts cannot be reproduced at all

Established 2026-07-30 by re-executing **all 82 artifacts** after the `_fingerprint` regex was widened
(see the comment above `_fingerprint` in `tools/coverage_score.py`). Three results worth having in one
place:

1. **70 of 82 reproduced with the operator's live `learn_daemon.py` running; 80 of 82 reproduce once
   `NORD_HOME` points at a scratch directory.** The ten-artifact gap is entirely the daemon: it writes
   `~/.snn_data_lake` continuously, most probes read it, and `consciousness_refute_probe.py` is
   well-behaved enough to *detect* the resulting noise and exit 2 with
   `UNKNOWN: arm A is not reproducible at seed=0 -- no verdict from a noisy harness` rather than emit a
   verdict. **FIXED 2026-07-30: `record` and `rerun` now run every probe against a fresh empty
   `NORD_HOME`** (a per-invocation temp dir), so the ledger no longer depends on what the operator's
   brain happened to be doing. An explicit `NORD_HOME` in the caller's environment still wins, which
   keeps a frozen-lake snapshot possible without editing the instrument. Confirmed by re-running all 82
   under isolation: **80 reproduce, 2 fail** (the pair in point 2), and **not one artifact that
   reproduced ambiently broke under isolation** — the risk worth checking first was that a cold lake
   would turn real evidence into null evidence, and it did not, so nothing was re-recorded to reach 80.
2. **`cons_refute_strength_iso_120` and `exec_refute_marg_taskset_given_pfcbias` are not reproducible in
   principle**, and neither was re-recorded. Both are deterministic *given* a lake, and both report a
   `metacog reliability` counter that reads that mutable durable state: `48/120` recorded, `48/120` on an
   empty lake but with the reply count moving `0/30 -> 1/30`, and `50/120` against a frozen copy of the
   live lake; the Executive one reads `3/60`, `5/60` and `22/60` across the same three conditions.
   Re-recording would capture one more snapshot of a moving number and call it evidence, which is the
   failure this instrument exists to prevent. Stated as a known limit instead.
3. **The regex widening caused none of this.** Every artifact was scored under both the old and the new
   regex from the same execution: the same 70 pass and the same 12 fail either way, so no recorded
   artifact was invalidated by the change. It did add one catch — under an isolated `NORD_HOME`,
   `cons_refute_strength_iso_120`'s only divergence is a `N/M turns` line, which the old regex could not
   see and would have passed as reproducing.

### Citation rot: 23 of this ledger's citations point at a comment or a blank line

Added 2026-07-31, and **counted before any rule leans on it**, per the standing invariant for adding a
gate. `_cite_ok` verified only that the cited FILE existed and that the line number was INSIDE it. A
cited file that gains lines above the cited point therefore leaves the number in range and pointing at
something else, and `score --check` stays green while every affected claim quietly stops naming its
evidence. This is not hypothetical: `SpatialTemporal.md` cited `brain.py:1325` for the place binding,
commit `2454355f` added 14 lines above it, and 1325 became a comment. A human caught it; the instrument
could not.

Two mechanisms now exist:

1. **`#anchor`** — a citation may be written `path:line#symbol`, where `symbol` is the `def`/`class` the
   line sits in (or `module`). F5 REFUSES it if the line no longer sits in that symbol. Anchored to the
   symbol rather than to the line's text on purpose: reformatting a line whose meaning never changed
   must not fail the gate, because these citations are claims about WHICH CODE, not which characters.
   The bare `path:line` form still resolves, so no existing row was invalidated.
2. **W6** — a WARNING (never fatal) naming every citation that points at a comment or blank line.

**The census was 23 rows across 10 of the 11 categories**, overwhelmingly into `cortex/brain.py`, the
most edited file in the tree. **All 23 are now repaired and anchored: the census stands at 1.**

Each re-point was derived by locating the cited line's ORIGINAL text at the baseline commit and finding
where that byte-identical text lives now (the offsets are uniform per region — `+26`, `+40`, `+52` —
because the drift is insertions above, not rewrites), then reviewed by a second pass told that accepting
a wrong citation is worse than leaving a rotted one. No citation was snapped to a nearby plausible line
to silence a warning, and no row was found whose mechanism had disappeared.

**The 1 remaining warning is correct and expected.** `Executive.md:20`'s citation of
`autonomous_loop.py:969` (`:878` when this was written; re-anchored by `--repin` on 2026-08-02 after
the loop grew ~600 lines) really does name a comment — deliberately: that comment is the documentation of
why PFC working memory is *unreachable* from the loop, which is the faculty's finding. A negative result
cited to the prose that records it. It keeps warning, and that is better than an exemption list.

**W6 fires for ANCHORED citations too, and the first version did not.** It skipped them, so adding an
anchor SILENCED the census instead of fixing anything — a suppression path built into the check whose
whole purpose is to end silent suppression. Fixed the same day it shipped; the Executive row is the
proof, since it would otherwise now read 0.

### `--repin` is blind to a ledger with uncommitted edits, and `--repin-from` is the fix

Added 2026-08-02, the day it was needed. `--repin` finds the text a citation ORIGINALLY named by
`git blame`-ing the ledger row for the commit that wrote it. A row edited but not committed blames to
the all-zero sha, which `--repin` correctly refuses to read -- and then counts as `unblamable` and
moves on.

That is the normal state right after a repair. MEASURED: repairing this ledger and then editing
`cortex/brain.py` in the same session drifted 36 citations, and **84 of 88 rows were unblamable**, so
`--repin` reached 16 of them and printed a confident summary. A tool that fixes 16 of 36 and says
nothing about the other 20 is worse than one that refuses.

`--repin-from SHA` ignores blame and reads every cited line at a named pre-drift commit. From
`51827048` it moved 114 with **0 unblamable**. The function's KNOWN LIMIT paragraph had already
described this recovery path in prose -- *"has to be recovered from an explicit pre-drift commit
instead"* -- without implementing it, which is its own small lesson about documented workarounds.

### The census UNDERCOUNTS, and by how much is not yet known

Repairing the 23 surfaced a harder class the comment/blank heuristic is structurally blind to: **a
citation that drifted onto a DIFFERENT LINE OF CODE.** Confirmed instances found while repairing, none
of which W6 can see — `brain.py:596` (was the `_audio_enc` assignment, now the import two lines above
it), `:1254` (was in `_consume_perception`, now in `_perceive_audio`), `:1112` (was in `_safe_ctx`, now
in `_attend_real_stream`), plus `:1464`, `:2322` and one that should read `:1620#_drive_for`. Two of
those keep the SAME enclosing function, so even an anchor would pass them.

**Now mechanized: `python tools/coverage_score.py --repin [--write]`.** It `git blame`s each ledger row
to find the commit that WROTE the citation, reads the cited line's text at that commit, and locates that
byte-identical text at HEAD. A unique match is a re-point; absent or ambiguous is reported and never
guessed. Dry run by default.

**The first run found 81 drifted citation references — W6 had been able to see 23 of them.** Also 180
already correct, 0 whose text had vanished, 16 ambiguous (the line text occurs more than once at HEAD,
so no unique target exists). All 126 citation references now carry `#anchor`.

**The ordering trap, recorded because it caught this repair.** `git blame` is per-LINE, so editing a
ledger ROW resets provenance for every OTHER citation on that row. The 23 hand-repairs therefore erased
the blame trail for their row-mates, and `--repin` then read their already-drifted text as the original
and pinned the drift as correct — with an anchor on it, which made it look verified. Four citations were
recovered from an explicit pre-drift commit instead (`brain.py:596→598`, `1254→1280`, `1112→1138`,
`1464→1504`). **Run `--repin` BEFORE hand-editing rows, never after.**

Two known gaps remain, stated rather than implied fixed: the 16 ambiguous references have no unique
target and still need a human, and an anchor proves only that a citation sits in the right function —
drift WITHIN a function passes both checks. A green `--check` means the numbers re-derive; it has never
meant, and still does not mean, that every citation names its evidence.

### The neuter probe now reports a GRADED observable, because its flip counts were masking live faculties

`charge_probe`'s two observables — act/hold changed, trajectory changed — are ARGMAX tests over eleven
competing drive sources. Removing one voice almost never moves the winner, so a source that genuinely
shifts the decision variable scored `0/60` and read as inert. `SpatialTemporal.md` measured the shape
exactly: all-eleven neuter gives **47/60** trajectory flips, every single-source neuter gives **0/60**.
The pathway is decisive; no single voice is, because ten others carry the sum. Phase 5 of
`COVERAGE_TO_100.md` asks for a neuter-A/B per dormant faculty — run on that probe it would have
produced masked negatives across the board, and a negative verdict is the one that retires a faculty.

The probe now also reports **`internal reads changed by the signal: N of M ticks`**, plus the top state
fields by total |delta| — the values the brain itself later reads. New verdict tier between the two
existing ones: **INTERNAL-READ CHANGED**, which maps to the rubric's 41–60 band ("changes an internal
read") while 61–80 ("changes a decision") still requires a flip.

**It is deliberately NOT exit 0.** Exit 0 is the >40 gate in `coverage_score` (F8); returning it here
would let 41–60 evidence satisfy a 61–80 claim, which is the inflation this instrument exists to stop.

Measured immediately on the ledger's own known-good control: `drive 60` under the corrected harness
gives **0/60 flips but 55/60 ticks of changed internal reads**, moving `acc_conflict`, `evc_value`,
`evc_cost` and `reliability`. The control that looked dead is consumed — with evidence.

The new line prints `N of M`, not `N/M`, on purpose: `_fingerprint` harvests every bare ratio in a
probe's stdout, so a ratio here appended a third observable to every artifact's fingerprint and made
all of them fail `--rerun` — an instrument change invalidating the evidence it exists to check. The two
flip counts reproduce byte-identically either way, which is what proves this addition is not
behavioural.

### Phase 5, first pass: 17 learning-verb methods adjudicated at RUNTIME (2026-07-31)

`method_dormancy_gate.audit()` reports 0 dormant but **16 UNCERTAIN** — a call with a matching NAME
exists on a live module, but the gate cannot prove the receiver is that class. Name-matching is weaker
evidence than reachability, and P3's guard is already "reachability is not charge". Each was resolved by
following the receiver and instrumenting a real tick run, then adversarially reviewed.

**8 LIVE with charge proven** — `LearnedAttention.learn`, `ValenceLearner.learn`,
`JepaWorldModel.train_step`, `PrefrontalExecutive.update`, `Cortisol.learn_rate_scale`,
`DriveReductionLearner.train`, `NGSNN.fit`, `KnowledgeLog.consolidate`.

**6 DORMANT (never called on a live path)** — `LatentQ.update` (module imported, every method cold),
`_TabularQ.update` (correct: a self-check control), `ForwardModel.update`, `DevelopmentalEnvelope.update`
(dormant twice over — never called, and discarded even under its opt-in flag),
`PredictiveCodingNetwork.learn` (correct: the offline wrapper over the live `learn_step`),
`AffectiveEpisodicStore.consolidate`.

**3 CALLED_BUT_DISCARDED** — called on the right receiver, changing nothing:
- **`AffectiveAppraisal.learn`** — the first pass called this LIVE with a confirmed chain; the reviewer's
  neuter-A/B found every behavioural hash byte-identical, only telemetry moving. The pool that charges
  (`w_aro`) and the component that is consumed (`valence`, index `[0]`) are **disjoint**. Worse, it is
  tier-dependent: it charges in the stdlib tier and is inert in the eprop-snn tier the daemon runs.
- **`TimeCellClock.learn_interval`** — called ~66 times per 400 ticks and learns nothing; its plastic
  threshold is a de facto fixed constant, contradicting the module's own headline claim to have replaced
  a hand-coded interval constant. Also a P1 finding.
- **`IntrinsicPlasticity.adapt`** — live call site, proven receiver, no charge on the route taken.

**Two instrument defects found in the gate itself.** It reports the `_TabularQ` finding at
`latent_q.py:165` while the real class method is `LatentQ.update` at `:102` — the lineno names a
different class than the finding is about, which nearly got the wrong class adjudicated. And
`EXEMPT_METHOD` is keyed by BARE METHOD NAME, so the two exemptions this pass earned cannot be written:
`"learn"` and `"update"` would blanket-exempt the commonest learning verbs across all 194 modules and
hide the very defect the gate exists to catch — the original `PredictiveCodingNetwork` case was itself a
method named `learn`. The granularity is the defect; the key needs to become `module.Class.method`.
Until then those two stay UNCERTAIN with their dispositions recorded here rather than silenced.

### AffectiveAppraisal.learn charges a pool nothing reads (measured, `affect_appraisal_ab_eprop.txt`)

Phase 5 / T013. The rule is called 61 times in 120 ticks on the correct receiver and updates real
weights. Neuter it and the behavioural digest is **byte-identical** (`eb13865b2f0b8b56` both arms).
What moves is `affective_appraisal[1]` — **arousal** — total |delta| 1.178 across 19 of 120 ticks.

`_appraised` has exactly one non-telemetry consumer: `brain.py:2081`,
`"affect": abs(float((self._appraised or (0.0, 0.0))[0]))`, which reads index **`[0]` — valence**.
`brain.py:2794` is `state()` telemetry. **The pool that charges and the component that is consumed
are disjoint**, so the learning is real and reaches nothing. Verdict: CALLED-BUT-DISCARDED in the
eprop-snn tier. It is tier-dependent — the prior adjudication measured it charging in the stdlib
tier — so the artifact's `recorded_by` names the tier, and this verdict does not generalise.

**Two limits of the harness, stated because the first version tripped one of them.**
`docs/coverage/artifacts/ab_method.py` originally walked only int/float/bool/dict leaves and reported
`0 of 120` for a rule whose one effect is a `(valence, arousal)` TUPLE — a null that was an artifact
of the measuring device's own blindness, which is the exact defect class this ledger exists to catch,
reproduced inside the instrument on its first run. Fixed to expand sequences.

Second, still open: the harness reads `state()`, which mixes values the brain later CONSUMES with
values only telemetry shows. It therefore cannot award the rubric's 41–60 band by itself; its output
says so and requires a non-telemetry reader to be named first. That is why this finding is
CALLED-BUT-DISCARDED rather than "changes an internal read".

Also note the artifact's fingerprint is empty: the harness prints `N of M`, never `N/M`, so
`--rerun` verifies its exit code and not its observables. Deliberate — a ratio here would append an
observable to every other artifact's fingerprint — but it means this artifact's reproduction check is
weaker than a tick-counting probe's.

### TimeCellClock.learn_interval: 32 calls, zero movement (`time_cells_ab_eprop.txt`)

Phase 5 / T014 + T017. Called 32 times in 120 ticks on the correct receiver. Neutered, the
behavioural digest is identical AND **0 of 120 ticks show any state change** — a stronger null than
`AffectiveAppraisal`, which at least moved a telemetry field.

That null alone would have been a half-truth, because `state()` never exposes what this rule writes:
`learn_interval` mutates `self.thresholds[i]` inside `TimeCellClock`. So the A/B proves the effect
does not PROPAGATE, and says nothing about whether the threshold moves. Measured directly instead:

```
learn_interval calls       : 32
n_cells                    : 64
distinct threshold vectors : 1        <- across all 120 ticks
start == end               : True
max |delta| over run       : 0.0
first 4 thresholds         : [1.0, 2.0, 3.0, 4.0]   (start AND end)
```

**The "plastic" threshold is the initial ladder, unchanged.** 32 reward-modulated STDP updates moved
it by exactly zero.

**This is a P1 finding as much as a P3 one.** `cortex/time_cells.py`'s own header claims the module
REPLACES a hand-coded interval constant with a learned one. Measured, the learned value is a de facto
fixed constant — the "constant sitting where a computed value belongs" that P1 names as this
project's signature defect, wearing a learning rule's clothes. The claim in the module header is
contradicted by its own runtime behaviour and should not be quoted until the rule is fixed.

> **SUPERSEDED, 2026-08-02 (specs/003).** Everything above was true when it was measured and is kept
> as the record of *why* the rule was fixed, but it must not be quoted as current. Two things have
> changed underneath it:
>
> - **The integer ladder `[1.0, 2.0, 3.0, 4.0]` no longer exists.** The constructor tiles
>   geometrically, so the first four thresholds at n=64 are `1.0000, 1.0672, 1.1389, 1.2154`. The
>   integer ladder was itself the reason the rule could not learn: with `float(i+1)` at `drive=1.0`
>   every nominal latency was exactly an integer tick, so the actual spike time equalled the nominal
>   one, `dt` was identically 0.0, and the threshold was unlearnable in principle rather than merely
>   in practice. `time_cells_ab_eprop.txt` predates that change and the two others that followed it
>   (`t_pre` taken from `spike_tick` rather than the nominal, and the order-preserving refusal), so
>   the artifact is stale by design, not by neglect.
> - **The rule now moves thresholds.** MEASURED over 4000 ticks x 12 seeds in the reset-driven
>   regime: 36.58 of 64 thresholds move. The "32 calls, zero movement" figure describes the retired
>   rule.
>
> What is NOT superseded is the methodological point, which still holds and is why the row was not
> re-banded on the strength of a code change: `state()` exposing nothing about the clock's internals
> meant `ab_method.py` was structurally blind to this rule, so its null was uninformative rather than
> negative. That gap is closed separately — see `time_cells_width` / `_grown` / `_refused` /
> `_failed` in `state()`.

### IntrinsicPlasticity.adapt: measured NEVER CALLED, contradicting the earlier verdict

Phase 5 / T015 (`intrinsic_plasticity_ab_eprop.txt`). On the `make_brain()` live path, seed 0,
eprop-snn tier: **0 invocations at 120 ticks and 0 at 400 ticks**. Behavioural digest identical, 0 of
400 ticks changed. The harness exits 2 (NEVER-CALLED) rather than returning a charge verdict, because
neutering something that never ran proves nothing.

**This contradicts the 2026-07-31 adjudication**, which recorded CALLED_BUT_DISCARDED with "a live
call site and a proven receiver". The call site is real — `homeostatic_plasticity_stack.py:140`,
`ctx["rate"] = self._ip[i].adapt(y)` — so the disagreement is not about whether the code exists. It
is about whether that stack is reached on the path `make_brain()` actually takes.

**Both readings are recorded and neither is deleted.** Either the earlier verdict measured a
different entry point (calling the stack directly, or via `autonomous_loop` rather than the brain
tick), or the stack is unreachable from the live tick and the rule's true disposition is
wire-or-retire rather than a charge-strength question. That distinction changes which of P3's three
dispositions applies, so it is worth resolving properly rather than by picking the more recent number.

Left OPEN with the contradiction stated. Recording a confident verdict here would mean choosing
between two measurements on the basis of which one ran last.

### Phase 5 re-band (T028): five bands examined, none moved, each restated

Five ledger rows cite the nine rules' files. Every one was re-examined against the Phase C/D
measurements. **None changed**, and SC-005 is satisfied by restating why rather than by silence:

| row | band | why the measurement does not move it |
|---|---|---|
| `Affect.learned_affective_appraisal` | 22 | Band 21–40 IS "live, not consumed". The measurement — charges arousal, sole consumer reads valence — is precisely a not-consumed finding, so it CONFIRMS the band instead of weakening it. |
| `SpatialTemporal.interval_timing_time_cells` | 35 | The band rests on the interval TIMER (`_interval_timer`, ramping every wake tick), not on `learn_interval`. The measurement shows the PLASTICITY half is inert; the timing readout it was banded for is untouched. **The row must not be read as evidence that the interval is learned.** |
| `Imagination.efference_copy_forward_model` | 10 | Band 1–20 is "built, not live". `ForwardModel.update` never called on a live path — exactly that. Confirmed, not contradicted. |
| `Affect.somatic_marker_recall` | 40 | Rests on the store/recall path. `AffectiveEpisodicStore` is constructed live at `autonomous_loop.py:1548` and `.store()` is called; only `.consolidate()` is dead. The dead method is not what the band cites. |
| `Memory.affective_content_retrieval` | 10 | Same store, same reasoning, and 1–20 already claims no more than "built". |

**The pattern is worth naming**: every one of these bands was ALREADY at or below 40 — "live, not
consumed" or "built, not live". The rules measured dead this phase were never carrying a charge claim
in the first place. That is the ledger's conservatism working: nine rules found inert, and not one
band was inflated by them. It is also why this phase moved no number, which is the correct outcome
and not a missing deliverable.

## The state of this ledger, stated plainly

**It is a first pass, not the adversarially-verified sweep the rubric specifies.** The rubric's
procedure is one independent auditor per category plus a refuter for every score above 40
(`COVERAGE_RUBRIC.md:85-95`). What is here was populated in a single pass while building the
instrument. Treat the per-faculty bands as **provisional and re-adjudicable**; treat the *refusals* as
binding, because those are mechanical.

The roster is also incomplete: 30 faculty rows across eleven categories, 10 of them `UNSCORED`. Several
categories have a single row. A category with one row is not a measured category.

## `37.5%` is WITHDRAWN, and it is not replaced by this number

`COVERAGE_BASELINE.md` records an equal-weighted total of 37.5%. That figure is withdrawn as
**unreproducible**, for a reason that does not depend on anybody's judgment about bands:

> **There was not one recorded A/B transcript in the repository.** All 602 tracked files were searched:
> `BEHAVIOURAL CHARGE` appeared in three (this rubric, POLICIES.md, and `charge_probe.py` itself) and
> `act/hold decision changed` in one (the probe). The rubric forbids any band above 40 without a
> runtime A/B. **So the maximum re-derivable total was 40%** — and 37.5% sits at 94% of that ceiling,
> which requires nearly every faculty in the brain to sit at the very top of "live, not consumed" while
> the baseline's own text documents multiple rule-4 zeros (amygdala `threat()` constant,
> `region_wiring`'s empty registry, the reentrant-workspace key mismatch).

A category score is *defined* as the mean over its constituent faculties, and no per-faculty band table
was ever committed for any category, so **0 of the 11 recorded scores can be re-derived at all**. The
rubric asked each auditor for exactly that table (`:85-91`); the deliverable was produced and discarded.

**The number this ledger currently prints is therefore a floor on a partial roster, not a re-measurement
of the brain.** Do not compare it to 37.5% — the two are not on the same scale, which is precisely the
mistake `COVERAGE_RUBRIC.md:14-16` warns about for the earlier 22.8% and ~31% figures. The next
adversarial sweep populates these files properly and *that* becomes the first comparable baseline.

## The hole the instrument cannot close

**Roster padding.** The category score is the mean over rows the auditor chooses, so adding ten
trivially-live band-40 sub-faculties lifts a category from 20 to ~35 with no new capability. The scorer
cannot refuse this because the rubric never fixes the faculty roster. Closing it needs a **tracked,
fixed roster per category, changed only with a stated reason** — a rubric-level change, so it is flagged
here rather than quietly implemented.

Lesser known holes, all real: a chosen tick count (`drive 4` exits 1, `drive 60` exits 0 — same faculty,
same seed, opposite verdicts; the artifact records `n`, so it is visible but not refused); a long run
until one flip appears, since `charge_probe`'s contract is `flips > 0` (surfaced as W4); and a citation
that resolves to a line saying something else, which no cheap check can catch.

## What the ceiling looks like, so nobody plans past it

The rubric pre-declares that wiring into `cortex/brain.py::_route_percept_to_action` moves a three-way
arm decision and **cannot** raise Language, Imagination, Gf, Consciousness or WholeBrain. That is
**five categories excluded — 45% of the weight** at 9.09 points each — which move only on capability.

**Corrected 2026-07-29, twice, and both errors were mine.** This paragraph said "only 4 of the 11
categories are reachable", which does not add up: 11 − 5 excluded leaves **six**. The rubric explicitly
names four that wiring can raise (Executive, Affect, SpatialTemporal, action-relevant Perception) and
explicitly excludes five; **Memory and Social are simply unstated**, and quietly filing them under
"unreachable" made the arithmetic wrong in the direction that flattered the plan.

It also claimed a **48.5%** wiring ceiling. That number is **withdrawn**: it was computed from the
*retired* 37.5% per-category roster, and this file forbids comparing anything to 37.5% two sections
above. Quoting a ceiling derived from a withdrawn baseline is the same category error, one level up.

Re-derived from the evidence this instrument actually reports (Perception 23.3, Memory 20.0, Executive
26.2, Consciousness 5.0, Social 12.5, the rest 0.0): taking all **six** non-excluded categories to band
70 yields **38.6%**; taking only the four named yields **25.9%**. Both figures are **provisional on a
30-row partial roster** and will move once the ledger is populated properly — which is exactly why the
number belongs here, next to the evidence, rather than in a plan document as a headline.

The shape of the conclusion is unchanged and is the part that matters: **wiring cannot reach 100%, and
the five excluded categories need capability.** That is why `COVERAGE_TO_100.md` puts capability last
and largest, not first and cheapest.

---

## One inserted time cell EARNS its place (`time_cells_growth_heldout.txt`)

specs/003, T039–T040. `python tools/timing_heldout.py --seeds 12`, exit 0, recorded and verified
`reproduces`. This is the verdict the whole growth feature was gated on, and it was permitted to be
a null.

```
main (widest log-gap)    mean d = -0.004970  sd 0.001050  se 0.000303
                         improves 12 of 12   ties 0 of 12   refusals 0 of 12   |mean/se| 16.40
null-control             mean d = -0.000002  sd 0.000003  se 0.000001
                         improves 10 of 12   ties 0 of 12   refusals 0 of 12   |mean/se|  3.07
main - null (contrast)   mean d = -0.004967  sd 0.001050  se 0.000303
                         main better 12 of 12   ties 0 of 12                   |mean/se| 16.39
```

**The held-out split**: the interval universe is derived from the population's OWN ladder — 10 points
per gap at `lo + (hi-lo)*(p+0.5)/10`, ordered gap-major, deterministic positional 1-in-5 stride
(`index % 5 == 4` → TEST). 630 intervals → 504 of 630 teach, 126 of 630 test, disjoint, with **63 of
63 gaps** represented in the test set. The 20% ratio is the discipline ported from
`reasoning/ng_snn.py:178-201`; the POSITION is not portable — a literal trailing-20% split leaves
only 4 of 63 gaps able to move the metric.

**The control**: a second cell inserted at the *lowest-teach-residual* gap — where the population
already fits best. Its purpose is to answer "would any extra cell have done this?", and it says no:
the same insertion elsewhere buys 0.04% of the benefit.

**The regime**: unsaturated. `specs/003-grow-the-live-populations/regimes.md` measures **96.11% of
live clock-advancing ticks unsaturated** (3 seeds x 600 ticks on `make_brain(persist=False)`). The
readout's saturated branch is a range ceiling rather than an estimate, so nothing here licenses a
claim about the saturated regime; only `grow("extend")` addresses that, and its benefit is
clamp-widening under FR-006 and is never reported as earning.

### Three things this result is not

1. **It is not a claim that the brain times better.** The instrument evaluates the ladder noise-free,
   which isolates how well the population TILES the interval and deliberately excludes the ramp's
   own noise. Whether a point-estimate improvement can be attributed to tiling rather than to how
   recently any cell fired remains open — geometric-mid was adopted because it reads only the
   population, not because it is more accurate. Against true elapsed it is in fact **+45% RMSE worse
   at the live width** (1.0536 vs 0.7276), closing to +4% by n=512. That cost is the price of
   measurability and is stated wherever the benefit is.
2. **It is not a big number.** 0.004970 of held-out RMSE. It is significant because the paired
   dispersion is tiny, not because the effect is large.
3. **The verdict rule was changed mid-measurement, and here is why.** The first version required the
   control to be indistinguishable from zero. It scored exactly +0.000000, ties 12 of 12 — and that
   was an artefact: the lowest-residual gap held **zero test intervals**, so nothing measured it. The
   fix restricts the control to gaps a test interval actually falls in, at which point a
   noise-free readout guarantees *some* movement and "indistinguishable from zero" becomes
   unreachable by construction. The verdict now rests on the paired **contrast** between the arms at
   the same seeds, which is what FR-006 was asking all along: is the benefit attributable to
   placement? `tests/test_timing_heldout.py::test_null_control_gap_is_measurable` pins the repair so
   the fake zero cannot come back.

**Fingerprint is empty**, as with the artifact above: the harness prints `N of M`, never `N/M`, so
`--rerun` checks the exit code and not the observables. Deliberate, and it means this artifact's
reproduction check is weaker than a tick-counting probe's.

**Ledger effect**: `record` writes no `metric:`/`baseline:` keys, so F10 caps a freshly recorded
`kind: heldout` artifact at band 40. `SpatialTemporal.interval_timing_time_cells` sits at 35. The
honest deliverable of this feature is a correct instrument, a real number, and at most five points.

### 2026-08-03 -- seven citations re-anchored to a FUNCTION HEAD, which is weaker and says so

The fifth re-anchor of the day. The safety work (cockpit actuation, the NL firewall, the E-STOP
intent ordering, the latch on `inject_disturbance`) rewrote lines in six source files, and
`--repin-from 5868e1c1` fixed 55 mechanically. Ten survived as ambiguous, and four of those were
disambiguated by the rule used earlier: read the cited line's text at the pre-drift commit, find its
occurrences at HEAD, keep only the one whose enclosing symbol matches the anchor already named.

**Seven could not be fixed that way, because their original line TEXT no longer exists** -- the edits
did not move those lines, they replaced them (the false "GRADED availability" comment among them).
For those the citation was re-anchored to the head of the function the anchor already names:

  Perception.eprop_event_vision_tier      brain.py:1207#_perceive_real
  Perception.eprop_event_audio_tier       brain.py:1281#_perceive_audio
  Affect.dimensional_valence              brain.py:2198#_observe_self_wake
  Affect.affective_contagion_empathy      brain.py:1658#_drive_for
  Social.empathy_help_drive_to_action     brain.py:1411#_route_percept_to_action
  Gf.sleep_schema_induction               brain.py:2468#_enter_sleep
  WholeBrain.gwt_reentry_to_action_drive  brain.py:1411#_route_percept_to_action

**This is a REAL LOSS OF PRECISION and is recorded rather than smoothed.** The original citations
named a specific line someone chose deliberately; these name a function. Two of the seven now cite
the SAME line (`brain.py:1411`), so W3's duplicate-fingerprint reasoning cannot tell those two rows
apart by citation any more. The claim each row makes is unchanged and no band moved -- what got
weaker is the evidence link, and a weaker link that is labelled is better than a broken one that is
not.

**The standing cost, now measured five times in one day:** any commit touching a cited source file
re-drifts this ledger. Re-anchoring is part of landing such a commit. Two tooling lessons from doing
it badly first: `--repin --write` must run ONCE from a clean base -- repairing on top of a partial
repin reads post-repin line numbers against a pre-repin commit and produces nonsense (a citation
"matching" 194 lines) -- and the pre-drift line numbers must come from the ledger AS COMMITTED, not
from the working tree.
