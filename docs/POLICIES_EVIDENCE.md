# POLICIES evidence -- relocated incident forensics

`docs/POLICIES.md` is the authoritative policy text; this file holds worked cases and
incident forensics RELOCATED from it byte-exact (specs/021 US4; extended specs/022 T003).
Entries are moves, never rewrites: each body below appears exactly as it stood in
POLICIES.md at relocation time, and a pointer remains at the original site. Budgeted in
`doc_budget_gate.BUDGETS` from its first commit.

## e-prop invariant -- relocated adjudication forensics (specs/022 T003)

### mlp_q weight-transport resolution (015): why both stated routes were wrong

**WHY BOTH STATED RESOLUTIONS WERE WRONG TO CHOOSE FROM, and this is the transferable part.**
Route (a) was framed as necessary because the FA cosine "misses the bar in both directions". That
cosine **could not discriminate**: the module's own docstring records it as *"essentially
`cos(W2[a], B[a])` under the relu mask (`err` cancels)"*, and for a FIXED RANDOM `B` that is near
zero **by construction** — it asked whether the feedback is PARALLEL to the forward weights,
which is the weight symmetry feedback alignment exists to deny. An instrument no alignment rule
could ever pass is not evidence against alignment rules, so it never justified amending the
invariant. Route (b) was never needed: nothing had to be retired and no capability paid for.

**MEASURED** (`specs/015-resolve-weight-transport/evidence/trajectory.md`, 960 rows,
per-epoch transport control 1.0 or void): converging `B` clears on **8 of 8** draws against a pre-registered >= 6-of-8 bar;
frozen random feedback **0 of 8**; permutation null 0.0000. **No capability lost** (XOR and
linear margins unchanged) and **no new constant** (P1) — the weight-decay pairing was swept
and killed on measurement. Full numbers: trajectory.md, not here.

### gf.py::MLP.train control adjudication (2026-08-15): the full reasoning

**Why it is nonetheless not a violation — on the invariant's own scope, not by analogy.** This
rule governs how the BRAIN learns; code whose JOB is to be measured against is not one of the
brain's learning rules. The invariant concedes the category itself: its verification clause
*requires* "an independent reverse-mode BPTT gradient", so backprop is obliged to exist in this
tree as an instrument. **CORRECTED 2026-08-15, same day, because the first version of this
sentence overstated and was challenged:** it read *"`MLP` is that category"*, which conflates two
different things. The instrument the invariant NAMES is `reasoning/eprop.py::Net.grad_bptt`, it
already exists, and it shares no code with `grad_eprop` — `MLP` is NOT that instrument. It is a
control of a different kind: a Gc BASELINE, not a gradient reference. The general principle
carries the verdict; the specific claim did not, and reading it as "MLP is the gradient the
invariant requires" would have been false. What `MLP` has is: **zero references outside `gf.py`** (measured over `src/`,
`tests/`, `tools/`), ONE construction site (`main_sl()`), and `main_sl` is a self-check reached
only from `main()` under `__main__`. No faculty consumes it; it is on no tick.

**What it is FOR is the load-bearing half: it exists to LOSE.** `main_sl()` asserts
`mdl_ok < ind_ok` — the learned model MUST fail the held-out family the program-inducer solves —
beside `in_acc >= 0.5`, which requires it to genuinely learn. It is the Gc arm of a Gc-vs-Gf
dissociation. **Porting it to e-prop would VOID the experiment, and that structural argument is
what this rejection ships in place of a measurement (P7 guard (b)):** a weakened learner would
satisfy `mdl_ok < ind_ok` because it learns worse, not because learning is not synthesis — P8's
contaminated control, the same failure as the `mlp_q` FA cosine that no alignment rule could
ever pass.

**TWO THINGS THIS VERDICT DOES NOT DISCHARGE.** (1) The module did not self-declare: its
docstring said "every weight comes from gradient descent" — true, and naming neither the
transport nor this invariant, where `mlp_q` named both. Disclosure added at the site 2026-08-15
(P8 clause 4). (2) The gate consequence — **CLOSED 2026-08-15: the control exemption is GRANTED
by the owner**, recorded in the invariant block's exemption paragraph.

## P7 — Any dependency that helps this become a phenomenal brain is accepted

### Real past incidents, each re-verified against the tree 2026-08-04

The bar requires these, and requires them re-checked rather than inherited. Every line below was read
from the file on the date stated. The 2026-07-20 abolition is what makes them evidence rather than
history: **the rule is gone and the tree is still producing this code.**

| incident | evidence |
|---|---|
| **Dependency-aversion manufactured a P2 violation in the safety envelope — the sharpest one, and it is CLOSED, which is what makes it good evidence.** A private 2-link arm carried *"a third implementation of the safety gate, with a second E-STOP that no test ever exercised"* | `substrate/demo.py:74-88`, whose retirement marker states the cause in the project's own words: *"It existed so this demo could run without dependencies; the zero-dependency rule was removed project-wide in 2026-07 and mujoco is a core dependency now, so that reason no longer describes anything."* Retired under specs/004 |
| **Three independent hand-rolled linear solvers**, none importing numpy *(historical incident, all three since fixed: nef_compute's `_solve` DELETED — the module is `cortex/nef_compute.py` since 2026-08-12 and solves via `np.linalg.solve`; the other two are thin numpy wrappers with the incident recorded in-file)* | at the time: `reasoning/nef_compute.py:32` (Gaussian elimination, partial pivoting), `reasoning/pc_active_inference.py:85` (Gauss-Jordan, partial pivoting), `reasoning/sorn.py:263` (Gaussian elimination, partial pivoting) — against `cerebellum/wavefront_planner.py:212`, which solves the same shape of problem in **one line**: `dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(3), err)` |
| **The duplication the rule forced has DIVERGED and can no longer be merged** | `substrate/_mathutils.py:16-23` — *"Copies whose formula DIFFERS were deliberately left in place (not merged)"*, listing seven, including that `burst_multiplexing._pearson` / `stdp_wta_bayes.pearson` *"use zero-denominator guards, so they are NOT byte-equivalent"*. A correctness difference, not a style one |
| **A hand-written PNG encoder, while the library that does it is already a declared dependency nothing imports** | `substrate/arm_view.py::encode_png` — IHDR/IDAT/IEND chunking, CRC, filter-type-0 scanlines, docstring *"Stdlib only."* Meanwhile `pillow` is declared in the `[hf]` extra and imported **nowhere** in `src/`, `tests/` or `tools/`. **STILL OPEN 2026-08-04, and now tracked** — `tools/dependency_ledger.py` carries `pillow` as an EXEMPT (it is a genuine transitive pin for `datasets`' image decoding) whose reason names this exact incident and specs/010 T021 as the owner. Adopting pillow here and DELETING the hand-rolled encoder is the P7+P2 move, and it needs the measurement of what it buys. *(Cites converted to symbols: `arm_view.py:46-74` had drifted — `encode_png` is at `:55` — and `pyproject.toml:73` is now a comment about the `[gpu]`/`[train]` extras.)* |
| **Nengo was never permissible for the core loop under the old rule**, so population coding was re-implemented | `docs/PITCH.md:275-277` — the arm loop *"borrows the NEF idea of population-coded representations for the spiking PID controller, but implements it directly rather than depending on Nengo"*; the `[nengo]` bridge exists only as an opt-in extra |
| **P0's open cortical-column avenue IS a hand-rolled dependency incident.** `cortex/cortical_microcircuit.py` reimplements Potjans-Diesmann 2014 from scratch — the Table 5 connection-probability matrix, the fixed-in-degree connectivity draw, the variance-preserving `w/sqrt(scale)` correction, ring-buffer delays, clock-driven per-neuron integration — and its **only imports are `math` and `random`** | `cortical_microcircuit.py:47-48` (imports), `:58-67` (CONN_PROB), `:160-191` (`_build_connectivity`), `:193-246` (`simulate`). This is the same file P0 lists as its clearest open avenue, and NEST/Brian2/GeNN each ship this exact model |

**Residue status, RE-MEASURED 2026-08-04. Two of the three items this paragraph named are now
CLOSED, and it was still asserting all three.** The rule was abolished on 2026-07-20; here is what
the tree actually holds today, each line re-derived rather than inherited:

- **CLOSED — the purity asserts are GONE.** This said *"Three `assert "torch" not in sys.modules`
  purity asserts survive"* at three sites. Grep for the statement across `src/`: **zero live
  occurrences.** The only remaining hit is `reasoning/vsa_structure_mapping.py:39`, which is *prose
  describing the retired artefact*, and `laminar_local_readout.py` carries its own removal note
  ("Two `assert "torch" not in sys.modules` purity asserts were [removed]... P7 supersedes it").
- **CLOSED — the package docstring.** This said `src/snn_data_lake/__init__.py:1` *"still opens 'a
  zero-dependency neuromorphic library'"*. It now opens *"snn_data_lake -- a neuromorphic library
  organized by brain anatomy."*
- **CLOSED — `scikit-learn`.** The gate specification below said the ledger *"fails today on
  scikit-learn... absent from `pyproject.toml` entirely"*. It is now declared in the `[gpu]` and
  `[train]` extras with a dated note. **CLOSED FURTHER, re-run 2026-08-05:** this line then read
  *"`tools/dependency_ledger.py` reports **UNDECLARED: 1**, and it is `arcengine` (two unguarded
  imports under `reasoning/arc_agi/`)"*, which was overtaken by the `[arc]` declaration in the same
  day's floor-raise pass. Re-derived, the tool reports **UNDECLARED: 0** — `arcengine` is declared
  and its marker is gone. The two unguarded imports are unchanged and still real
  (`reasoning/arc_agi/play.py` and `play_brain.py`, `from arcengine import GameAction, GameState` at
  module top in both); what changed is that pyproject now names the distribution they need.
- **OPEN — the spec-plan certifications, and there are EIGHT, not six.** Re-counted by grep for
  "no new depend" across `specs/`: `001/plan.md` (**twice**), `003/plan.md`, `006/plan.md`,
  `007/plan.md`, `009/plan.md` — **plus the two this paragraph missed**, `005/shared-edits.md`
  and `008/spec.md`. None evaluated a library, so the scoring habit is running on nothing.
  `specs/010/plan.md` is the one that gets it right and says so explicitly. Each of the eight now
  carries a P7 void-notice at the site (added 2026-08-04) rather than only being counted here —
  **verified 2026-08-05: eight `P7 VOID` notices, one per certification, and the certification text
  is left in place beneath each.** *(Line numbers dropped in the same check. The pass that inserted
  the notices wrote `001/plan.md:19` and `:61` — and inserting a notice at `:22` pushed the second
  certification to `:63`, so the citation was invalidated by the edit that accompanied it. A cite
  into a file you are editing in the same pass is stale before the pass ends.)*
- **OPEN — the hand-rolled incidents themselves** (the solvers, the PNG encoder, the microcircuit)
  are all still in the tree. Those are P2 work items, not residue of the rule.

**The lesson this paragraph is now itself an instance of:** it was the section arguing "the rule is
gone and the tree is still producing this code", and it went on asserting three cleanups that had
already landed. A residue list has to be re-derived on every edit or it becomes a list of work
someone already did — which discredits the items that are still real.

**Two honest counter-examples, recorded so this policy is not read as a blank cheque.** Both are
rejections that P7 would **not** overturn, and both are the correct shape:

- **spikingjelly and cupy** were rejected *"on a structural reason rather than a benchmark — their
  fused multi-step kernels collapse the **time** axis `T`, and the live tick has `T=1` by
  construction"* (P5, above). Category (b). It stands. **And the record needs a correction that an
  audit produced on 2026-08-04:** spikingjelly was not "rejected twice" — it was *adopted* on
  2026-07-02 into `research/neuromorphic_torch/`, and the later structural finding is about the live
  tick, not about the library's admissibility.
- **numba** won its benchmark decisively — *"for fused LIF at our real widths numba beats a CUDA graph
  by ~16×"* — and was still not adopted, because no live population reaches the width at which the win
  exists. A measured win on a workload the brain does not run is not a reason to adopt. Note the
  reason for this one has itself been re-measured once (P5's `0.00%` amendment), which is why it is
  stated here in its durable form rather than by quoting the retracted number.

## P8 — Invent the method. The literature is a starting point, not a boundary

### Real past incidents, each re-verified against the tree 2026-08-04

| incident | evidence |
|---|---|
| ~~**LIVE DEFECT, and the sharpest justification for this policy: the brain answers "are you conscious" from a probe that never reads the brain.**~~ **CLOSED 2026-08-04 (specs/010 C08+C09) — struck through, not deleted, because this file keeps the record of what it claimed.** `reasoning/spike_chat_lake.py` still routes the literal strings `("are you conscious", "are you aware", "do you have experience")` to `self._consciousness()`, but the probe behind it was REBUILT | `reasoning/consciousness_probe.py` — its own header now records the fix. **BINDING and RE-ENTRY were DELETED, not re-thresholded**, per falsifier F11: BINDING scored exactly 1.000 across 300 random object sets against a rate control whose docstring conceded the pairing was *"right only by luck"*, and RE-ENTRY was arithmetic over three literals (0.40 < 0.55 < 0.70) with no brain input and no RNG. What survives: **GWT ACCESS now reads the brain** (`probe_access(brain)` drives `Brain._access` and returns UNKNOWN, not a verdict, without one), and **INTEGRATION reports the DISTRIBUTION across seeds** instead of the one hardcoded `seed=0` that passed. Verified 2026-08-04: `tools/red_reachability.py --assert` lists both surviving probes under READS_SYSTEM and reports **NO_RED 0**. *(Line cites `:925` / `:929` / `:84` / `:90-95` all dropped — every one had drifted; `spike_chat_lake.py:925` now resolves to an unrelated `return` inside the ignition helper.)* |
| **A metric rejected for being arithmetic rather than function** | `specs/009/data-model.md:55` — `can_degrade`: *"MUST be able to get WORSE. Any metric that improves monotonically with parameter count is arithmetic, not function."* `mean_assembly_coupling()` was rejected on it: it sums surviving weights, so thinning moves it whether or not recall degrades — it measures the **manipulation**, not the function |
| **An oracle metric — the score leaked the answer** | `specs/009/data-model.md:57` — `not_an_oracle`: assembly-derived wiring on `AttractorNet` is an ORACLE, because *"with two disjoint assemblies the co-activity mask IS the ground-truth partition"*. It scored 1.000 untrained. Recorded UNRESOLVED pending a clean metric, not closed |
| **A null through a rounding instrument is not a null** | `specs/008/research.md:100-108` — the deep unrounded fingerprint found A/B differing at **1,713 of 98,320**, while the same three runs read through `Brain.state()` gave **0 of 149**. The tree's own wording is *"A null measured through `Brain.state()` is not a null"* (`specs/007/spec.md:595`) |
| **A saturated success signal, banned and never lifted** | `probe_fluid.py` is recorded SATURATED and BANNED as a Gf signal in five files across `docs/` and `specs/`, with the honest replacement being a held-out ARC-eval number: `specs/006/verdict.md:16` — **18 of 400 = 4.50%** eval, 53 of 400 = 13.25% train |
| **An invention that measured NULL and was DELETED rather than parked — P8 working exactly as intended** | the fixed-point combinator. `specs/006/verdict.md:13` — combinator-only scored *"12 of 400 -- byte-identical ID set"*; `:61` — *"The combinator's only measured effect was negative: 2 of 400 training solves starved"*; `:67` — *"null result; feature stops here"* |
| **The same invention, load-bearing twice, validated once — the debt-is-proportional-to-load rule, violated** | the `(a·b)/D` fusion identity. In `cortex/vsa_reasoning.py` it carries its algebra and a live check. Reused in `cortex/access_consciousness.py::_coherence`, it claims *"FUSED, bit-exactly (v4.80)"* (`:154`) and *"bit-exact: `_coherence` is a pure function of `codes`"* (`:205`) with **no assertion anywhere comparing the fused branch against the scalar one** — the bit-exactness rests on a prose algebra argument, in the module that feeds the consciousness telemetry |
| **Killing your own proposal before building it is the licence succeeding** | `docs/INVENTION.md:6-7` — of four proposals, three *"were killed by a property of a file that anybody could have opened, and one was killed by the algebra of its own headline metric"* |

## P5 -- relocated GPU measurement forensics (US1 policy audit, 2026-08-25)

**CROSSOVER RE-MEASURED 2026-07-28, and it MOVED — this policy asked for exactly that ("re-measure
when populations grow; the crossover is a number, not a doctrine").** The range recorded below was
optimistic. Measured again on the RTX 5070, torch 2.11.0+cu128, 300 iterations per point:

| N | numpy | torch-CUDA | winner |
|---:|---:|---:|---|
| 240 | 0.0022 ms | 0.0627 ms | numpy, 34x |
| 4,096 | 0.0028 ms | 0.0139 ms | numpy, 5x |
| **24,576** | 0.0069 ms | 0.0136 ms | **numpy, 2x** |
| **32,768** | 0.0075 ms | 0.0133 ms | **numpy, 1.8x** |
| 262,144 | 0.5499 ms | 0.0133 ms | CUDA, 41x |

numpy still wins at BOTH ends of the recorded 24,576-32,768 band, so the true crossover is somewhere
between 32,768 and 262,144. The run was under machine load, which penalises numpy more than CUDA, so
this errs in the conservative direction. The practical consequence is unchanged and now has more
headroom: live populations of 24-240 are nowhere near it.

**RE-MEASURED AT TARGET SCALE 2026-08-03, AND THE PRACTICAL CONSEQUENCE INVERTS.** Every
measurement above is correct and every one of them is about populations of 24-240. The owner
corrected the premise on 2026-08-03: **the live tick is meant to grow past 24,576 and far beyond,
and the neuron population should already be at least 1,000,000.** Under that target the sentence
"live populations of 24-240 are nowhere near it" describes the DEFECT, not the design. Measured on
the RTX 5070 (12.8 GB), torch 2.11.0+cu128, elementwise LIF tick (decay, threshold, reset):

| N | numpy CPU | torch CPU | torch CUDA | CUDA vs torch-CPU |
|---:|---:|---:|---:|---|
| 240 | 0.003 ms | 0.013 ms | 0.037 ms | 0.3x -- CPU wins, the toy regime |
| 24,576 | 0.017 ms | 0.055 ms | 0.033 ms | 1.7x |
| 1,000,000 | 4.761 ms | 0.265 ms | 0.088 ms | 3.0x (and **54x** vs numpy) |
| 4,000,000 | 18.639 ms | 2.310 ms | 0.189 ms | 12.2x |
| 16,000,000 | 77.674 ms | 15.711 ms | 1.552 ms | 10.1x |

**SPARSE IS NOT AN OPTIMISATION AT THIS SCALE, IT IS THE ONLY REPRESENTATION THAT FITS.** Synaptic
connectivity at fanout 100:

| N | dense weight matrix | sparse COO | sparse matvec |
|---:|---:|---:|---:|
| 100,000 | 40 GB | 0.120 GB | 13.1 ms |
| 1,000,000 | **4000 GB** | **1.200 GB** | 11.3 ms |

A dense 1M x 1M matrix is 4000 GB against 12.8 GB of VRAM -- impossible by ~300x. And spike vectors
at biological firing rates compress hard: at 1 percent firing, COO is 33.3x smaller than dense
(0.12 MB vs 4.00 MB); at 2 percent, 16.7x.

**What this changes, and what it does not.** The live tick at its DESIGN width is a GPU target, and
`specs/008-vectorize-scalar-paths` carries that as a requirement rather than an option. What does
NOT change: the deterministic CPU gate stays authoritative, `run_selfchecks` and `pytest` MUST
remain runnable without a GPU because CI has none, and GPU reduction order is not deterministic in
general. If reproducibility at scale and determinism cannot be reconciled, the answer is a CPU gate
at small width plus a separately-validated GPU path at scale -- never a weakened determinism
requirement. This entry is the re-measurement this policy asked for, applied to the target
population rather than the current one.

**CUDA GRAPHS TESTED 2026-07-28 — they move the crossover 6-8x, and the conclusion still holds.**
The measurements above are EAGER torch, where each tiny op pays a separate kernel launch. CUDA graphs
capture a whole op sequence and replay it with a single `cudaGraphLaunch`, which is aimed exactly at
the CPU-dispatch overhead that made the width-38 test read 0.01x. Measured on the RTX 5070, 200
sequential LIF-shaped ops (`v = v*decay + in; s = v >= theta; v -= theta*s`), best-of-20 after warmup:

| N | numpy | CUDA graph | winner |
|---:|---:|---:|---|
| 24 | 0.578 ms | 1.118 ms | numpy 0.52x |
| 240 | 0.613 ms | 1.127 ms | numpy 0.54x |
| 1,024 | 0.788 ms | 1.129 ms | numpy 0.70x |
| 3,072 | 1.088 ms | 1.224 ms | numpy 0.89x |
| **4,096** | 1.233 ms | 1.225 ms | **graph 1.01x** |
| 24,576 | 4.010 ms | 1.223 ms | graph 3.28x |
| 65,536 | 8.975 ms | 1.425 ms | graph 6.30x |

Two results, both worth keeping. **Graphs beat EAGER CUDA everywhere** (2.2 ms vs 7.4-26.6 ms on the
same sequence) — the launch-overhead hypothesis is confirmed, and any future GPU path here should
capture rather than dispatch. And **the crossover drops from ~24,576-32,768 to ~4,096**, because
replay cost is flat in N while numpy's is linear.

But live populations are **24-240**, still ~17x below even the improved crossover, so **the
conclusion is unchanged: the live tick does not belong on the GPU.** This is the strongest technique
available tested against the real shape, and it loses by 2x at our widths. Recording it under guard
(b): "we tried CUDA graphs here and it was 0.54x" is a result, and it should stop this experiment
being repeated a third time. The number that would change the answer is population WIDTH — at 4,096
the GPU starts paying, so a genuinely wide population is the thing to look for, not a faster kernel.
