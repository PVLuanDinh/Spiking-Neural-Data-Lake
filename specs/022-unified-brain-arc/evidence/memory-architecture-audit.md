# Memory architecture: the literature's checklist against this tree (audit at 621742a, 2026-09-04)

## What this is, and what it is not

An external source set on human memory was measured against this tree and against
`PLAN-FINAL-2026-09-02.md`. **The source is the citation page of a Kurzgesagt science video, "How
Are Memories Stored Inside Your Brain?"** — expert-reviewed by four memory researchers (Wilma
Bainbridge, Sheena Josselyn, Lukas Kunz, Michel van den Oever) and backed by 126 primary citations.

It is named plainly because provenance is part of the evidence: this is **popular-science framing
over real literature**, so the *citations* are the evidence and the narration is only a pointer to
them. Nothing here is adopted because a video says it. P8 governs: the literature is a starting
point, not a boundary, and "the paper does it this way" is never itself a reason to build.

**This file is an AUDIT, not a plan.** It records what the tree had at this HEAD against what the
literature names. The ranked work at the end is the audit's consequence, not a commitment.

## Method, and the one thing that makes the negatives trustworthy

Ten mechanisms, one adjudicating agent each, every one required to: run the duplicate-capability
existence check in at least three phrasings; read `graphify-out/SRC_INVENTORY.md` rather than
walking the tree; grep the capability vocabulary AND the cited authors' surnames; prove liveness at
RUNTIME rather than from source; and — before reporting ABSENT — **prove its detector fires on a
known positive.** Every non-`PRESENT` verdict was then attacked by a second agent whose only job
was to find the incumbent the first one missed, because this project has declared something absent
and been wrong about it three times.

Three adjudications were corrected by that attack step, and two of the corrections were mine:
* An attacker found `hippocampus/growing_learner.GrowingLearner` rewrites a matched stored trace
  toward the cue. Checked directly: it does — but in `observe()` (encoding), while `classify()`
  "never grows". So the *dilution dynamic* exists, driven by re-EXPERIENCE rather than re-CALL. The
  adjudicator was wrong that nothing rewrites; the attacker was wrong that retrieval does.
* I nearly refuted the correct ACh finding myself: my grep matched `LocusCoeruleus.mode()`, a
  different class with an identically-named method. The name collision, not the finding, was wrong.

## Verdicts

| # | mechanism | verdict | the finding |
|---|---|---|---|
| 1 | Hippocampus stores an INDEX, not the content | **PRESENT** | the tree is *stricter* than the claim: P4b makes the lake the sole source of truth and the in-memory index a cache. `LakeRecall` is the charged incumbent |
| 2 | Pattern completion from a partial cue | **PRESENT** | four charged tiers; 100% held-out slot recovery. A fifth would be the duplication P2 exists to prevent |
| 3 | Vividness/confidence does not track accuracy | **PRESENT** | already found and killed here independently — `lake_recall`'s own docstring records that its first build "scored wrong answers at 1.0". Live: retrieval strength 1.000 against verified confidence 0.750 with 1 contradicted slot |
| 4 | Retrieval TRANSFORMS the trace | **PARTIAL** | **a risk to live claims, not a gap** — see below |
| 5 | Reconsolidation: recall makes a trace labile | **PARTIAL** | retrieval is a **pure function**: 32 recalls on the live path left the lake byte-identical; 50 recalls left two stores' digests identical. `RecallTrace` exists with **inverted polarity** — retrieval *protects* a record from the sleep prune |
| 6 | Synaptic tagging AND CAPTURE | **PARTIAL** | the TAG exists (`r_stdp` eligibility, e-prop traces); the CAPTURE step — a limited shared resource that tagged synapses compete for — does not |
| 7 | Novelty/emotion gate encoding strength | **PARTIAL** | novelty is live and habituating (1.0 → 0.378) but **never reaches a write**; its only consumer is attention gain |
| 8 | Winning the competition gates PLASTICITY | **PARTIAL** | winning gates the ROUTE (`ReentrantWorkspace._hebb_update`), not the plasticity. Every source learns whenever RPE ≠ 0, attended or not |
| 9 | Decay is the default; retention is earned | **PARTIAL** | the synaptic half is charged (`AttractorNet._normalize`, `downscale_prune`); nothing forgets once a trace is IN the lake |
| 10 | Sleep replay consolidates, dose-dependently | **PARTIAL** | replay machinery is real and partly charged; the cortical-vs-hippocampal DISSOCIATION has no incumbent, and there is no replay-DOSE axis wired to an outcome |

**All ten are absent from `PLAN-FINAL-2026-09-02.md`**, verified with detector controls run on the
plan file itself before the negatives were believed.

## RISK BEFORE WORK: mechanism 4 threatens claims this tree already publishes

This is listed first because it is not a feature request — it is a possible defect in live metrics.

The cited results (Xiao et al., *Transformed Neural Pattern Reinstatement*; Karlsson Wirebring et
al., *Lesser Neural Pattern Similarity across Repeated Tests Is Associated with Better Long-Term
Retention*; Zhuang et al. on reorganization predicting retention AND false memory) say recall does
NOT reinstate the encoding pattern verbatim, and that **LOWER** similarity across repeated
retrievals predicts **BETTER** long-term retention.

This tree scores recall by similarity-to-the-original in at least seven places, and it is charged:
`EpisodeConsolidator.retained` is published live as `episode_retained` and sits off its ceiling.
**If the literature holds, a metric this tree treats as "higher is better" is at best ambiguous and
at worst inverted.**

What it owes, and it is a measurement rather than a build: take the existing replay harness, score
retention against a HELD-OUT probe rather than against similarity-to-stored, and test whether
similarity and retention move together or apart across repeated replays. **Reachable red:** they
move apart. That result would not break the brain, but it would retire a published number, and
finding it ourselves is cheaper than not finding it.

## Ranked backlog

Ranked by capability bought per unit of risk, with P3 fixes ahead of builds — charging something
this tree already has beats adding something it does not.

**1. Charge the ACh encode/retrieve lability gate (a P3 fix, not a build).**
`neuromodulation/acetylcholine.Acetylcholine` already implements Hasselmo's literal
encode-versus-retrieve gate. Measured tree-wide: `plasticity_scale`, `feedforward_gain` and
`recurrent_gain` have **ZERO readers** outside their own module (control: `make_brain`, 74 readers —
the detector fires). Only `.level` is read, as a prototype-rate gain in `Brain._wake_tick`.
This is the cheapest real capability in the audit AND it is the gate mechanism 5 needs: a brain
that cannot distinguish an encoding tick from a retrieval tick cannot make retrieval do anything
different. **Owes:** a neuter-A/B showing the gate changes an outcome, not merely that it is
called — the distinction this project's P3 guard exists to enforce.

**2. Mechanism 5, reconsolidation, but only after (1).** The largest citation block in the source
set (22 of 126). Two honest cautions carried from the adjudication: `RecallTrace` already couples
retrieval to durable survival with the OPPOSITE sign, so this is a P2 adjudication against an
existing mechanism and not empty ground; and `GrowingLearner` already dilutes a stored prototype
toward new input, so the dilution dynamic must not be built twice. **Owes:** a reachable red where
a retrieved-then-restored trace measurably differs from the stored one, plus a BOUNDARY condition
(the literature is explicit that not every recall destabilizes), plus a control that is not the
method itself.

**3. Mechanism 6, the capture half of synaptic tagging.** The most concretely implementable item —
the cited paper (Lehr, Luboeinski & Tetzlaff) does it in a spiking network. The tree has the tag;
what is missing is a limited shared resource a later strong event manufactures and earlier tagged
synapses compete for. **Owes, and this is the P2 question that decides whether it is built at
all:** is an e-prop eligibility trace the same mechanism or a different one? Answer that with a
measurement before writing anything.

**4. Mechanism 7, couple novelty to the write.** Novelty is already computed and already habituates
on the live tick; it simply never reaches `KnowledgeLog.append`. Cheap, and it makes what the lake
retains reflect what mattered. **Owes:** proof that a novelty-gated write changes what is
RECALLABLE later, not merely what is stored.

**5. Mechanism 8, gate plasticity on workspace access.** Buys attentional credit assignment — today
the brain writes about things it never attended to. **Owes:** a measured reduction in interference,
with the un-gated arm as its control.

**6. Mechanism 9, durable forgetting.** The largest STRUCTURAL divergence from the biology and the
one that is least obviously a defect: this project deliberately keeps a durable lake, and some tags
are IMMORTAL by policy. **Owes a decision before any code** — whether "the brain forgets" should
apply to `brain_lake.bin` at all. Recorded as a question, not queued as work.

**7. Mechanism 10, the replay-DOSE axis.** `SleepConsolidator.wake(skill_id, assemblies, reps)`
already carries `reps`, which is the dose axis. The claim is dose-response ("sleep less, forget
more"), and the tree has no measurement relating replay dose to retention. **Owes:** the
dose-response curve, with the zero-dose arm as its control.

## What the audit says about the tree, stated plainly

The three `PRESENT` verdicts are not consolation. On the architectural core the literature
emphasizes most — an index into a durable store, completion from a partial cue, and confidence that
is not merely strength — **this tree already matches the literature, and in two of the three it is
stricter than the claim.** The gaps are real but they are gaps in a structure that is largely right,
and the single most valuable item in the whole audit is charging a mechanism the tree already has.
