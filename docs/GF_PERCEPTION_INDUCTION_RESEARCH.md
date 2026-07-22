# SNDL Perception-Driven Primitive Induction — Research Program Roadmap

**Status: SCOPING ONLY. Nothing here is built.** Constraint: no-LLM (symbolic / spiking / numpy;
torch allowed in the `[train]` tier only). This is the genuine remaining fluid-intelligence (Gf) move,
scoped as a multi-session research program.

## Why this program exists (the honest wall)

Measured this session: SNDL's honest held-out Gf number is **ARC-AGI evaluation ~3%** (12/400;
`reasoning/arc_bench.score(solver, 'evaluation')`). The `scratchpad/probe_fluid.py` 11/11 is
**saturated / probe-fit** (banned as a success signal — its items map 1:1 to hand-authored `gf.py`
primitives). Every *within-closure* Gf lever is **falsified**:

- **Depth / search** — a depth-lift probe (`scratchpad/probe_depth_lift.py`) showed depth-5 (100k
  nodes, 12 s) rescues **0/40** eval tasks that depth-3 misses. The wall is **expressiveness**, not depth.
- **Neural-guided re-ranking** — `arc_diagnostic` recorded `misranked=0`; ranking cannot move it.
- **Corpus abstraction (DreamCoder "sleep" / Stitch / babble)** — compression-based invention is
  closure-preserving *by construction* (independently confirmed: Alford DreamCoder-on-ARC). It lifts
  depth-within-closure, never expressiveness.

The failing ~97% need concepts **outside the ~70-primitive DSL closure**. The only unfalsified lever is
inducing **new primitives from raw perception** — the human "DSL design" step, automated.

## 1. THE THESIS

**An MDL-scored perceptual-predicate mint.** Bridge a small, pre-registered set of domain-general
**measurement generators** (connectivity variants, symmetry-group membership, periodicity / tiling,
cardinality, adjacency / containment / alignment, non-contiguous same-shape grouping — ARGA-style
competing abstractions) to the existing synthesizer: enumerate candidate predicates over the train
pairs, score by **two-part MDL** (Ferré's portable principle: cost of the predicate + cost of the pairs
given it), and promote winners as **new *typed* terminals** into `synthesize()`'s grammar — using the
OpenGf promotion pattern (`gf.py:426 _learn`) but with terminals built from *measurements*, not from
compositions of old primitives.

**Why this and not the alternatives:**
- **Not grammar-widening.** Hand-porting Hodel's ~90 missing primitives would lift eval score while
  performing *zero* concept invention — it re-does the human offline DSL-design step the program exists
  to automate. Highest-scoring available move, and pure theater w.r.t. Gf.
- **Not probe-fitting.** `probe_fluid.py` is banned; the metric is held-out ARC-eval delta only.
- **Honest fine print (stated up front):** a generator set is itself a closure — a *meta-closure*. This
  program does **not** escape closure-boundedness (nothing finite does). The falsifiable bet is narrower:
  *a small closure of domain-general measurements, composed by the existing synthesizer, covers far more
  of ARC-eval's missing concepts per author-decision than a closure of hand-authored grid operations.*
  Phase 2's probe measures exactly this bet and can kill it.

**SNDL substrate grounding:** `components()` (`arc_solver.py:106`) already produces connected-component
objects; the wall is its **closed** five-key attribute schema `{cells,color,size,bbox,shape}` (`:150`)
and the **architectural disconnect** — `arc_solver` imports stdlib only, with zero wiring to any
perceptual machinery. The program is concretely: open the schema, build the bridge, let MDL decide what
gets named. (`OpenGf` at `gf.py:358` is the codebase's one genuine grammar-*growing* mechanism, but 1-D
integer only + re-packs compositions; `VsaSynthesizer` synthesizes rule-programs but consumes *parsed*
attributes — "no perception"; `StructureMappingSolver` maps *given* propositions, never induces a
relation.)

## 2. PHASES (each gated by a falsify-first probe)

### Phase 1 — Ceiling measurement: oracle-primitive injection  (1–2 sessions)
- **Goal:** measure the program's absolute ceiling *before building anything* + produce a concept-gap
  taxonomy of what is actually missing.
- **De-risk probe (this IS the phase):** on a **development half** of the 40 depth-3-missed eval tasks
  (lock the other half, never inspect it), hand-identify the missing concept and hand-write the *one*
  minimal generic primitive per task (Hodel's 160-primitive vocabulary as reference taxonomy). Inject
  into `_BASE` (`arc_solver.py:528`); run existing `synthesize()` at existing budgets.
- **Falsifier:** oracle injection rescues **< ~15%** of dev misses → the "one missing primitive" form of
  the expressiveness thesis is wrong (tasks need many concepts + structural change) → the program dies /
  re-scopes *here*, for the cost of one probe.
- **Build:** none. Probe script only (`scratchpad/probe_oracle_inject.py`).
- **Honest metric:** dev-split rescue rate (= the ceiling every later phase is bounded by) + the concept-
  class histogram (grouping vs counting vs relations vs other).
- **Anti-theater rule:** an oracle primitive must be generic + parameterized (bounded AST, no task-
  specific grids, plausibly fires elsewhere). A `solve_task_X` primitive rescues trivially, proves nothing.

#### Phase 1 RESULT (measured — GO)

Executed: 44 eval tasks scanned → **41 depth-3 misses**; split 20 **dev** (inspectable) / 21 **locked**
(`scratchpad/oracle_locked.txt`, never opened). Four Fable-5 agents hand-wrote one **generic** missing-
concept primitive per dev task; each passed an AST genericity/safety audit (no literal-grid smuggling, no
banned calls, no task-specific constants); injected per-task into `_BASE`, run through unchanged depth-3
`synthesize()`.

**Rescue rate = 19/20 = 95%** (well above the 15% kill line → **GO**). Concept-class histogram of the
rescues: geometry 4, symmetry 3, region_fill 3, relation 3, color_logic 2, object_transform 2, counting 1,
grouping 1 — a small, diverse set of *measurement-expressible* concepts (the Phase-2 generator-set target).

**Honest reading (do not overclaim):** this is the **oracle ceiling** — it proves the expressiveness wall
is *breachable by generic measurement-based primitives* (the missing concepts ARE expressible), NOT that
SNDL's ARC-eval improved (it is unchanged ~3%; the oracle prims live in `scratchpad/`, are **not** committed
to `_BASE` — committing hand-authored primitives is exactly the grammar-widening theater guard #1 forbids).
The *realized* gain requires the Phase-3 **mint** to DISCOVER these concepts from perception (the unsolved
no-LLM bottleneck); per the ceiling calibration, expect it to realize ~1/3 of this. Two primitives honestly
encode a *measurement* + rely on the existing `_fit_colormap` closer (the mapping-*type* limitation Phase-3's
type extension must address). The 21 locked tasks remain reserved for a future held-out check; cross-task
generalization of the concepts is a Phase-4 question, untested here.

### Phase 2 — Open the perceptual schema: pre-registered measurement generators  (2–3 sessions)
- **Goal:** replace the closed schema with an extensible attribute/relation registry populated by a
  **fixed, pre-registered** generator set (frozen in a committed manifest *before* any eval inspection
  beyond the Phase-1 dev split).
- **De-risk probe (before the build):** paper/toy **coverage probe** — for each Phase-1 taxonomy entry,
  show (on paper or a 50-line script) the missing concept is a composition of the proposed generators.
- **Falsifier:** the taxonomy's mass sits *outside* the generator set (simulation-like physics, denoising,
  arithmetic-on-structure) → the generator set is wrong / the approach is → redesign or stop before
  writing the registry.
- **Build:** refactor `components()` output to registry form; implement generators (numpy/stdlib,
  deterministic, cached like `_SEG_CACHE`); ARGA-style multiple competing abstraction lattices per task.
- **Honest metric:** static representational coverage — % of Phase-1 missing concepts expressible in the
  predicate space (no search yet → nothing to inflate).
- **Honesty note ("spiking"):** this phase is symbolic/numpy. Do NOT wrap it in a spiking veneer to look
  brain-native; wiring `rank_order_coding`/`lca_sparse_coding` is a *later, optional* experiment that must
  earn its place by measured contribution.

### Phase 3 — The mint: MDL-scored hypothesis generation + typed promotion  (3–5 sessions; the bottleneck)
- **Goal:** close decomposition gap **(ii)** — the unsolved one — by generating out-of-closure operation
  hypotheses *from* the Phase-2 predicate space, scoring by two-part MDL, promoting winners as typed
  terminals into `synthesize()` per task.
- **De-risk probe (before the build):** **enumerate-and-check** — on the Phase-1 oracle-rescued tasks,
  does blind enumeration of the predicate space reach the known-sufficient predicate within a realistic
  node/time budget? Record the rank of the correct predicate in MDL ordering.
- **Falsifier:** correct predicates rank too deep to find within budget → the combinatorial cost has
  beaten the approach honestly → stop / redesign the prior (narrowing the space back into hand-authoring
  is not allowed).
- **Build:** the bridge module (the missing perception→synthesis wiring); the **type-system extension**
  — `count`, `objectset`, `relation` as first-class sorts (a minted predicate returning a count is
  unusable in a grid→grid-only grammar). This is a named sub-project, not a footnote.
- **Honest metric:** **held-out ARC-eval delta** (locked split) — tasks solved end-to-end with a *minted*
  primitive; plus a per-primitive **novelty audit** (old closure at depth 5 cannot reproduce the minted
  predicate's extension on probe grids).

### Phase 4 — Cross-task reuse: a library of minted concepts  (2–3 sessions)
- **Goal:** make minted predicates persistent + reusable *without* re-importing the falsified within-
  closure library trap.
- **De-risk probe:** measure Phase-3 minted-set cross-task firing on `re-arc` procedurally-generated
  tasks (development data, never eval).
- **Falsifier:** no minted primitive ever fires off its birth task → minting is memorization dressed as
  induction → stay per-task.
- **Build:** behavioral dedup (OpenGf pattern); anti-unify constants→parameters; admission gate ≥2
  distinct tasks or leave-one-out pass.
- **Honest metric:** locked-eval delta library ON vs OFF (ablation) + reuse rate.

### Phase 5 (optional, high-risk) — Sub-symbolic escape hatch: CompressARC + symbolization readout  (3–4 sessions)
- **Goal:** for tasks where symbolic minting fails, a per-task equivariant compressor (`[train]`-tier
  torch) manufactures structure with no DSL at all; then attempt readout of what it found into the
  Phase-2 predicate vocabulary (the open problem the CompressARC authors left).
- **De-risk probe:** reproduce CompressARC on a task sample; intersect its solved set with SNDL's missed
  set.
- **Falsifier:** the compressor only solves tasks SNDL already solves → no complementarity → dead before
  the hard readout part.
- **Honest metric:** incremental locked-eval delta on tasks *neither* system solved alone.
- **Honest label:** research-grade; the readout step has no existing solution. Budget as likely-null.

## 3. HONEST CEILINGS

| Phase | Ceiling | Why |
|---|---|---|
| 1 | Defines the program ceiling. Realistic guess: oracle injection rescues 20–50% of misses | Many misses plausibly need *multiple* new concepts + deeper composition (out of scope) |
| 2 | Bounded by Phase-1 taxonomy coverage; expect 50–80% of labeled concepts expressible | Gestalt / texture / figure-ground concepts resist small generator sets |
| 3 | Realizes maybe a third of Phase 1's ceiling | Search cost: knowing a sufficient predicate exists ≠ finding it under budget |
| 4 | Small multiplier on Phase 3, possibly zero | ARC-eval tasks are deliberately conceptually disjoint; reuse may be rare |
| 5 | A few tasks at most | Complementarity unproven; readout unsolved anywhere |

**Overall honest expectation:** ARC-eval ~3% → **maybe 5–8% absolute if everything goes well; zero is a
live outcome.** Calibration: Ferré (multi-year, full-time, the closest existing line) reached 7% private;
Icecuber needed 142 primitives + heavy C++ search for 20%. SNDL will not approach the leaderboard — that
is not the goal. The goal is a working no-LLM instance of the *mechanism* (perception → hypothesis →
verified named concept), which none of the cited systems fully has.

**Null-outcome value (why scope it anyway):** every phase-gate falsification is a real result in SNDL's
Gf theory (like this session's depth-lift probe that killed three attempts cheaply). Phase 1's taxonomy
is a publishable-grade characterization of *what concept classes gate SNDL's Gf* even if nothing else is
built. Phase-2/3 substrate work (open attribute schema, typed values, perception–synthesis bridge)
removes the two hard architectural walls regardless of eval movement. **High-risk / low-expected-yield
and worth scoping:** it is the only unfalsified Gf lever left, its first gate costs one session, and its
failure modes are all informative.

## 4. THEATER TRAPS (program-wide) + GUARDS

1. **Hand-smuggled Hodel primitives** (grammar-widening as induction) → generator manifest frozen +
   committed before eval inspection; provenance log per minted primitive; no generator edits from eval.
2. **Renamed-composite fake novelty** (the falsified library trap by the back door) → mandatory
   out-of-closure audit (old `_BASE` at depth 5 must fail to reproduce the minted predicate's extension).
3. **Overfit-to-inspected-eval** → hard split: dev half (Phase 1 may inspect) vs locked half (reported
   metric only, never opened); `re-arc` generators for all development iteration.
4. **Explosion "solved" by narrowing** (restrict the space until only anticipated answers fit = hand-
   authoring) → always report enumeration-space size + rank-of-found-predicate; a trivially small space
   is a red flag, not a win.
5. **Birth-task memorization** → Phase-4 admission gate (≥2 tasks or leave-one-out).
6. **Saturated-probe scoring** → `probe_fluid.py` banned; held-out ARC-eval delta only.
7. **Verification false comfort** (checks pass because the generator was pre-loaded upstream) →
   verification claims are only as honest as the pre-registration trail; audit both together.
8. **Oracle-probe self-deal** (Phase-1 "primitives" that are task solvers) → genericity constraints
   (bounded AST, parameterized, no literal grids).
9. **Spiking-washing** (numpy symbolic machinery wrapped in spike vocabulary to satisfy the brain-native
   rule) → label the induction machinery honestly as symbolic; spiking encoders enter only via measured
   ablation gains.

## 5. FIRST CONCRETE STEP (next session)

Write **`scratchpad/probe_oracle_inject.py`**:
1. Split the 40 depth-3-missed eval tasks: 20 **dev** (inspectable), 20 **locked** (write the IDs to a
   file, do not open).
2. For each dev task: hand-identify the missing concept; write one *generic* primitive (genericity
   constraints above); tag it with a concept class.
3. Inject into `_BASE` (`arc_solver.py:528`); run `synthesize()` at unchanged budgets; record rescue
   yes/no per task.
4. Output: rescue rate (= program ceiling) + concept-class histogram (= Phase-2 generator-set
   requirements).

**Kill condition, stated in advance:** rescue rate **< 15%** → the single-missing-primitive expressiveness
thesis is falsified for these tasks; do not proceed to Phase 2; report the null and re-scope. Same
discipline that killed three attempts this session, applied before the first line of program code.

## SOTA references (no-LLM lens; from the survey)

- **Icecuber** (2020 Kaggle winner, ~20%) — 142 hand-crafted C++ grid functions, greedy DAG search depth
  4; zero induction. Confirms the wall = where the primitive set ends. `arxiv.org/abs/2412.07322`
- **Hodel `arc-dsl`** — 160 hand-crafted primitives cover all 400 ARC-train; the existence proof that
  "DSL design" IS the concept-invention step to automate. `github.com/michaelhodel/arc-dsl` (+ `re-arc`
  generators).
- **ARGA** (Xu, Khalil & Sanner, AAAI-23) — object-centric graph abstraction; multiple competing
  abstraction schemes per task (the portable idea for Phase 2). `arxiv.org/abs/2210.09880`
- **Ferré ARC-MDL / MADIL** — two-part-MDL object-centric *models* (not transformation programs); new
  patterns emerge from compression; 7% private. The MDL scoring principle for the mint (Phase 3).
  `arxiv.org/abs/2311.00545`, `arxiv.org/abs/2505.01081`, `github.com/sebferre/ARC-MDL`
- **CompressARC** (Liao & Gu, 20% public, no pretraining) — per-task equivariant compression, no DSL;
  Phase 5's escape hatch (open problem: symbolizing what it found).
- **DreamCoder-on-ARC** (Alford) — independently confirms corpus abstraction is closure-preserving.
  `arxiv.org/abs/2110.11536`
- **Apperception Engine** (Evans) — the strongest existing no-LLM template for minting *new predicates +
  objects* from raw sensory input (ASP/Datalog + Kantian unity constraints); follow-up adds a neural
  perception front-end (`doi:10.1016/j.artint.2021.103521`).
- **ILP predicate invention** (Metagol/MIL; POPPI, `arxiv.org/abs/2104.14426`) — invent new predicate
  *symbols* from few examples, but *closure-relative* unless fed new perceptual background relations.
- **Playgol / Knorf** (Cropper; Dumančić et al.) — self-supervised curriculum + library refactoring to
  grow an ILP library without a corpus of human solutions (the play/sleep pattern for Phase 4).

## SNDL substrate anchors

`reasoning/arc_solver.py` — `components()` :106, attribute dict :150 (the closed schema to open), `_BASE`
:528, `_fitted_prims` :675, `ArcLibrary` :726, `synthesize()` :770. `reasoning/gf.py` — `OpenGf` :358,
`_learn` :426 (the promotion pattern). `reasoning/arc_bench.py` — `score('evaluation')` = the only honest
metric. `reasoning/arc_diagnostic.py` — the depth-≤3 coverage proof this program's Phase-1 probe extends.
