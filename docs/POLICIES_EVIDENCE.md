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
| ~~**LIVE DEFECT, and the sharpest justification for this policy: the brain answers "are you conscious" from a probe that never reads the brain.**~~ **CLOSED 2026-08-04 (specs/010 C08+C09) — struck through, not deleted, because this file keeps the record of what it claimed.** `reasoning/spike_chat_lake.py` still routes the literal strings `("are you conscious", "are you aware", "do you have experience")` to `self._consciousness()`, but the probe behind it was REBUILT | `reasoning/consciousness_probe.py` — its own header now records the fix. **BINDING and RE-ENTRY were DELETED, not re-thresholded**, per falsifier F11: BINDING scored exactly 1.000 across 300 random object sets against a rate control whose docstring conceded the pairing was *"right only by luck"*, and RE-ENTRY was arithmetic over three literals (0.40 < 0.55 < 0.70) with no brain input and no RNG. What survives: **GWT ACCESS now reads the brain** (`probe_access(brain)` drives `Brain._access` and returns UNKNOWN, not a verdict, without one), and **INTEGRATION reports the DISTRIBUTION across seeds** instead of the one hardcoded `seed=0` that passed. Verified 2026-08-04: `tools/red_reachability.py --assert` lists both surviving probes under READS_SYSTEM and reports **NO_RED 0**. *(Line cites `:925` / `:929` / `:84` / `:90-95` all dropped — every one had drifted; `spike_chat_lake.py:925` now resolves to an unrelated `return` inside the ignition helper.)* *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).* |
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

---

## POLICIES.md constitution split -- relocated evidence (owner directive, 2026-08-30)

*(Each block below is relocated BYTE-EXACT from `docs/POLICIES.md` on 2026-08-30, the day
POLICIES.md became a pure constitution: rules, honesty guards, verdicts and derivation pins
stay there; dated measurements, censuses, correction narratives and incident forensics live
here. The source label above each block names the section it came from. Everything inside a
block is historical -- line numbers, counts and statuses describe the trees they were
written against. Re-derive with the named gate before relying on any figure. Exception to
byte-exactness, disclosed here and marked in place: where constitution text travelled inside a
relocated block, the duplicated sentences were trimmed post-relocation so exactly one statement
survives -- each trim is a bracketed note at its site.)*

### Head matter: enforcement corrections, counted gate output 2026-08-05, provenance of the 2026-07 audits (relocated 2026-08-30)

**Corrected 2026-08-04, and the direction of this drift is the one this file keeps warning about.** The
two rows above said *"its gate is specified but NOT built"* while **both gates existed, ran, and were
registered in `.github/workflows/ci.yml`** — P8's `--assert`ing. This is the same failure recorded
under P6 ("a claim that was already false when it was last written"), now with three instances, so it
is worth naming as a pattern rather than an incident *(the normative rule distilled from this —
re-run the gate before writing a sentence about the gate — stays in `docs/POLICIES.md`; trimmed
here post-relocation)*. Counted output on this tree, **re-run 2026-08-05 in
`.venv-lake`** (every row below was executed, not copied):

| gate | output | in CI? |
|---|---|---|
| `tools/red_reachability.py --assert` | 400 files scanned, 13 criteria; **NO_RED 0, TAUTOLOGICAL 0, READS_SYSTEM 13, EXEMPT 0** — exit 0 | yes, `lint`, `--assert` |
| `tools/dependency_ledger.py --assert` | 400 files scanned, **29** distributions; **UNDECLARED 0, UNUSED 0, MISDESCRIBED 0**, EXEMPT 2, ALL_EXCLUDES 8 — prints `gate OK`, exit 0. **This row said `UNDECLARED 1` / `MISDESCRIBED 1` / `ALL_EXCLUDES 7` and "prints `gate FAIL`, exits 0 (report-only)"; re-run 2026-08-05 in `.venv-lake`, all three counts are 0/0/8 and the tool prints `gate OK`.** The parenthetical *"exits 0 (report-only)"* was the more damaging half and it was not a stale measurement but a structural fact nobody had checked: `main()` returns 1 only under `if do_assert and failed`, so the CI step — which invoked the tool with no flag — could not have reddened the build on ANY finding. Both are fixed together: the flag is added and the counts are re-derived | yes, `lint`, **`--assert`** |
| `tools/dependency_ledger.py --self-check` | 5 MISDESCRIBED criteria mutated against a string copy of `pyproject.toml`: **RED OK 5, NO RED 0, BROKEN 0**; unmutated tree reports 0 findings — `self-check OK: 5/5 criteria proven falsifiable`, exit 0 | yes, `lint` (**registered 2026-08-05**; it had run nowhere, so its own "5/5 proven falsifiable" line was a claim no gate re-derived) |
| `tools/doc_command_guard.py --assert` | 55 documents, 51 `python -m` commands (+2 EXEMPT = 53 matched); **UNRUNNABLE 0** — after 19 broken commands were CLEARED, not exempted. **54 / 50 on a clean clone**: `AGENTS.md` is in the gate's SCOPE and is gitignored, so one in-scope document and one command exist only on a working machine | **NO — see below** |
| `tools/p6_gate.py` | **ASSERTING in CI since 2026-08-06** (`--assert` + `--self-check`, `lint` job); a RATCHET, so green means "no worse", never "P6 satisfied". See P6 for the re-derived reading | yes |
| `tools/p4c_ratchet.py` | **ASSERTING in CI since 2026-08-06** (`--assert` + `--self-check`, `lint` job). Mechanizes P4c's "every future build lands inside the governed tree from the start", which was review-enforced only — and review is what let 259 builds accumulate outside it. Per-ZONE, not on the total: a total-only ratchet lets one zone launder against another (retire 5 in `reasoning/`, spend the headroom on 5 new `tools/` scripts, gate green, tree no better). Baseline is machine-generated by `--update-baseline`, never hand-typed, and that flag REFUSES to write a rise | yes |

**Two corrections to the block above, both found by re-running the gates on 2026-08-05 rather than
by reading what the previous pass wrote about them. Both are the same failure this section names,
committed in the section that names it.**

1. **`tools/doc_command_guard.py` is NOT a step in the `lint` CI job, and the row above said it
   was.** Measured: `.github/workflows/ci.yml`'s `lint` job runs `ruff check .`,
   `ruff format --check src tests tools`, `tools/citation_guard.py`,
   `tools/red_reachability.py --assert` and `tools/dependency_ledger.py` — five checking steps, and
   none of them is this gate. **That enumeration was invalidated later the same day by the pass that
   wrote it**, which added `--assert` to the ledger step and registered
   `tools/dependency_ledger.py --self-check` beside it; RE-DERIVED by parsing the YAML, the `lint`
   job now has **9 steps, 6 of them checking steps** (the other three are `checkout`,
   `setup-python` and `pip install -e ".[dev]"`). The conclusion is unchanged and re-measured —
   `doc_command_guard` is still in none of them — but a count of CI steps written in a pass that
   edits CI goes stale before the pass ends, and this block is where that lesson is supposed to live. Grepping `doc_command_guard` across `.github/workflows/`, `run_selfchecks.py`,
   `.pre-commit-config.yaml` and `pyproject.toml` returns **zero matches**. The gate is real, runs
   clean (`--assert`, exit 0), and carries a self-check whose red is reachable — but nothing runs it
   automatically, so under **P4c** it is *a build the gates cannot see, and therefore not a build
   this project can claim*. Registering it is open work; the CI edit is escalated because
   `.github/**` is outside this pass's write scope.
2. **The dependency ledger reports a THIRD direction the block omitted** — `MISDESCRIBED`. When this
   correction was written its measured line was
   `UNDECLARED: 1   UNUSED: 0   MISDESCRIBED: 1   EXEMPT: 2   ALL_EXCLUDES: 7`, and the live finding
   was *`ALL-EXTRA  extra [nir] is neither in [all] nor in ALL_EXCLUDES -- pip install ".[all]" does
   not install it`*. Recording only the two directions that were zero-or-known turned a finding into
   an absence. **CLOSED, re-run 2026-08-05 in `.venv-lake`:** `nir` is now a member of `[all]` and
   `arcengine` is declared, so the line reads
   `UNDECLARED: 0   UNUSED: 0   MISDESCRIBED: 0   EXEMPT: 2   ALL_EXCLUDES: 8`. The point survives
   the fix and is why this entry is amended rather than deleted: the omission was of a *direction*,
   not of a number, and a report that names only the directions it expects to be clean will hide the
   next one exactly as it hid this one.

**Provenance of the 2026-07 amendments.** Three independent audits mapped this codebase against
external frameworks — predictive coding, the Thousand Brains Theory, and quantum-consciousness
theories. They proposed five policies between them; **one** survived adversarial review, and the
rejections were as instructive as the survivor. Two proposed gates were *implemented before adoption*
and killed by their own output: one scored 0 failures / 47 UNCERTAIN (it could not catch its own
motivating case), the other produced 58 failures where 1 was claimed (adoption would have meant a
58-entry rubber-stamp allowlist). That bar is now itself a rule — see *Adding a policy* below.

What the audits mostly returned was not governance but **defects**, and they clustered into one
recognisable family, recorded under P3.

### P0: the avenue status table as of 2026-08-16 (statuses are point-in-time; the avenue LIST lives in POLICIES.md) (relocated 2026-08-30)

**The avenues, and their honest status today.** "By every available avenue" is only meaningful if we
say which are open — so each row states what fires, what does not, and what specifically blocks it:

| avenue | status |
|---|---|
| **self-building code** | **ROUTE FIRES, PAYLOAD EMPTY (measured 2026-08-13).** This row led with a bare **"What fires"** until 2026-08-13, and `specs/010-policy-compliance-audit/findings.md:370` forbids exactly that: it directs **(a)** *"Do NOT promote the SelfBuildLoop route into 'What fires'. State it honestly"* **and (b)** *"Give the measurement beside the claim"*. **Both halves are now applied** — (b) first, and (a) one commit later in the same lane, after an audit caught that the first pass had quoted only (b) and written that `:370` "was not applied until now" while the forbidden heading was still leading the row. A half-quoted directive reads as a discharged one. Route re-derived 2026-08-04 and cited by SYMBOL, because the line cites in this row had drifted AGAIN within one day of last being "corrected": `cortex/brain.py::Brain._maybe_autonomous_self_build` -> `basal_ganglia/self_build_loop.py`, every 8 nights, **only** under `NORD_SELFMOD=1` (enabled in `learn_daemon.bat`), plus `basal_ganglia/brain_native_selfbuild.py`. **MEASURED 2026-08-13, and the measurement is the whole point: the cadence FIRES and the payload is EMPTY.** 190 ticks of `make_brain(seed=0)` under `NORD_SELFMOD=1` **and `NORD_HOME` pointed at a scratch directory** (without it the run appends to the operator's LIVE lake -- two auditors hit exactly that while checking this row; `tests/conftest.py` already defends the suite, this prose copy was the undefended one) leave `Brain._selfbuild_last == {'night': 8, 'proposed': 0, 'kept': 0}` — the night-8 cadence triggers exactly once, the loop is entered, and it proposes **ZERO** candidates, so `SelfModHarness` is never constructed and no self-modification is ever attempted. (`git diff --stat` is empty after the run: nothing writes to the tree, because the per-proposal body is never entered.) This row has said "What fires" with no number beside it, which is exactly the shape the honesty guard ten lines below forbids — *"a capability reported without measurement is fake"*. It fires; it does nothing. (`findings.md:370`'s half (b); its half (a) is discharged by this row's new heading, above.) **It also re-sizes the `godel_machine` clause later in this row**: wiring `attempt_on_repo` into a loop that proposes nothing would satisfy a call-site grep and change no outcome, which is the force-wire this policy's own guard forbids. Filling the proposer is the open P0 work here, and it is a different and larger job than adding a call site. **`self_mod_harness.py` was listed here as reached through `Brain.self_improve` — it is NOT. `cortex/brain.py::Brain.self_improve` has ZERO call sites tree-wide** (re-verified 2026-08-04: the only tree-wide match for the name is its own `def`). An entrypoint with no callers listed under "What fires" is the census-counted-as-capability failure this row corrects for `godel_machine` one clause later, committed in the same sentence. **What does NOT fire:** `reasoning/godel_machine.py` — its `attempt_on_repo` has **zero call sites** tree-wide, confirmed against `importlib` / `__import__` / getattr and string-name dispatch rather than from a static read alone (re-counted 2026-08-05 over `src/`, `tests/`, `tools/`: **four** matching lines, all non-calls — its own `def`, **two** lines of its own docstring, and a docstring in `tests/test_self_mod_safety.py`; the previous pass wrote "three matches" and the enumeration it gave was right in kind and one short in count). The module IS imported live by `substrate/self_mod_harness.py` and registered in `run_selfchecks.py`, so "the only reference outside itself is its filename as a string" does not hold. It is imported-but-never-invoked — P3 shape 1, not absence — which is a different defect and a shorter fix. Wiring both is open P0 work. **PATHS CORRECTED 2026-08-04:** this row said `reasoning/self_mod_harness.py` and an undirectoried `self_build_loop.py`; both had been PROMOTED out of `reasoning/` (to `substrate/` and `basal_ganglia/` respectively) by the P4c program, and the row followed neither |
| **knowledge acquisition** | live — chat ingest → the one durable lake (`substrate/chat_memory.py`), `substrate/web_ingest.py`, reasoning-curriculum growth during sleep |
| **neural growth** | **RETENTION IS BUILT — this row asserted the opposite until 2026-08-04 and was wrong on every one of its three load-bearing claims.** Growth runs **only during sleep**, and what it grows is now KEPT — **cited by SYMBOL, 2026-08-05, because every line cite in this clause had rotted within a day of being written**: `cortex/neurogenesis_engine.py` assigns **`self._model = model`** at the end of a successful refit, seeds each refit from the prior model instead of starting cold (the `EXTENDS it instead of replacing it` block), and prunes on earn-your-place (the `EARN-YOUR-PLACE PRUNE` block). The serialization the row once said did not exist is `growth_state_bytes()` / `restore_growth_state()` / `substrate_is_retained()`, and it is **CALLED on two live paths** — the sleep tick and construction in `cortex/brain.py` — writing into the ONE lake under the growth tags (`substrate/home.py`: `_T_GROWTH_DELTA` chain records since specs/022 T010-ii; `_T_GROWTH` holds the legacy whole-blob snapshots and still takes raw non-npz blobs), which is P4b-correct rather than a sidecar. *(Cites into `cortex/brain.py` and `home.py`'s trailing line numbers were DROPPED 2026-08-04: `brain.py:3780` was `except Exception:`, `:2722` was blank, and `home.py:231`/`:280` were `return path` and a docstring. **The `neurogenesis_engine.py` cites were KEPT in that pass with the words "all three re-verified exact" — and re-resolved on 2026-08-05 they were not.** Measured: `:211` is a comment (`self._model = model` is at `:215`), `:250` is a comment (`def growth_state_bytes` is at `:254`), `:279` is `ticks=np.array(...)` (`def restore_growth_state` is at `:283`), `:321` is `self._acc = acc` (`def substrate_is_retained` is at `:325`), and `:180-183`/`:189` land on the explanatory comments above the code they name. Every one is off by exactly four lines, because a concurrent edit inserted four lines above them — **which is the point**: the assertion "re-verified exact" was true when written and false the same day, so the durable form is the symbol, and this row now uses it. The companion cite `substrate/similarity_sized_growth.py:78` was likewise off — `self.W` is at `:81` — and is named by symbol below.)* **What this row got wrong, recorded because the failure mode is the point:** it claimed "retains NOTHING", "never assigned to `self` (verified: zero matches for `self.*model*=`)" and "no `save`/`load` at all". The grep was the instrument and the instrument was wrong — the assignment is `self._model = model`, which `self.*model*=` does not match. **The file contradicted itself 18 lines later**, at the honesty-guard bullet that cites `neurogenesis_engine.substrate_is_retained()` by name as the site naming its own gap; a row and its own guard disagreeing is the drift this file keeps suffering. **`substrate/similarity_sized_growth.py` DOES retain** (`self.W`, allocated in `__init__` beside the `self._W0` init snapshot that exists to PROVE the weights moved). `hippocampus/{growing_learner,function_preserving_growth}.py` **UNVERIFIED** — a narrow attribute scan found no retained weight array, which is not the same as proving there is none. **THE REMAINING GAP IS NARROWER THAN THIS ROW CLAIMED, corrected 2026-08-16.** It read *"no cross-process restart has been demonstrated… demonstrating restart-with-grown-substrate is the open P0 work"*. **A cross-process restart IS demonstrated, at the ENGINE level:** `neurogenesis_engine._cross_process_restart` genuinely spawns a `subprocess`, is driven by the module's own self-check, and its child half runs under the `__main__` guard. Stating otherwise pointed P0 effort at work that shipped. **THE BRAIN-LEVEL RESTART IS NOW DEMONSTRATED TOO, AND THE SUSPECTED FLUSH WAS REAL — specs/019 `[B2.09]` CLOSED 2026-08-16 as a defect fix, which is what this row said it would be if the flush was real.** `cortex/brain.py::close()` never flushed growth, so growth reached the lake ONLY on the sleep cadence, and that save LAGS the growth that caused it. Measured, seed 7, two processes: a brain ticked until `grown_units` rose and then exited kept **0 of 2 units and wrote ZERO `_T_GROWTH` records**; one given 200 further ticks exited holding 3 and restored **5**, because the last cadence save had caught a PRE-PRUNE snapshot. The lake held a DIFFERENT brain than the one that exited, in both directions at once — too little and too much — which is P4b's *"a restart is not amnesia"* failing while every existing check stayed green. Fixed by `cortex/brain.py::Brain._flush_growth`: ONE mechanism with TWO callers (the sleep tick and `close()`), keeping the `substrate_is_retained()` gate rather than `grown_units > 0`, because those two came apart for the whole life of this feature. Proven by `tests/test_brain_growth_restart.py`, which spawns a real interpreter over the same lake. **The lesson outlives the fix and is why this row is worth its length:** the first restart test, at 260 ticks, PASSED AGAINST THE LIVE DEFECT — over that many ticks a save always lands after the last growth, so exit-state and save-state agree BY LUCK. `test_growth_survives_an_immediate_exit` is the one carrying the reachable red. A test can pin exactly the right property and still be satisfied by a coincidence, and only a constructed red separates the two |
| **cortical column construction** | **NOT live — but NOT for the reason this row gave until 2026-08-16.** `cortex/cortical_microcircuit.py` is a faithful 8-population Potjans-Diesmann circuit, EXEMPT-permanent under the `"cortical_microcircuit"` key in `substrate/dormancy_gate.py` `EXEMPT`. **This row said *"it has no input port… building that port is the work"*. That is FALSE, and had been since 2026-08-04.** `CorticalMicrocircuit.thalamic_drive(thalamic_hz)` IS the port — its own docstring calls it *"THE INPUT PORT… the signal a caller injects"* — and `simulate(t_ms, settle_ms, thalamic_hz)` carries a caller's drive into it. It is **anatomically targeted**: `CONN_PROB_TH` is nonzero for **L4e/L4i/L6e/L6i only**, the real thalamocortical projection, so a caller cannot inject into L2/3 or L5 and the stimulus must travel the intracortical route. It contributes **exactly 0.0 at 0 Hz** (an undriven run stays bit-identical to the pre-port module) and **refuses a negative rate**. **The TRUE reason it is not live: NOTHING DRIVES IT** — measured 2026-08-16, zero references to `thalamic_drive` or `CorticalMicrocircuit` anywhere outside its own module. That is P3 dormancy, not an absent capability, and it resizes this avenue from *design and build a port* to *wire a caller to an interface that already exists and self-checks*. **What IS still absent, stated precisely so the next reader is not sent to build the wrong thing:** the port carries a **scalar rate**, so a caller may modulate drive INTENSITY but cannot inject a spatial PATTERN. `specs/011` Phase A is scoped as *"the layered circuit gets an input port"* against a port that exists; a patterned input is the honest remaining work there. Wiring a live driver is specs/019 `[B2.08]`, and the exemption is retired the day it lands. *(Cited by SYMBOL: this row's line cite has now been wrong twice — `dormancy_gate.py:83` pointed at a comment about a deleted `reaction_diffusion_compute`, and its 2026-08-04 replacement `:100-104` has since rotted onto `gwt_deep`.)* |
| **self-recalibration** | live — `substrate/signal_calibrator.py`, running-baseline forms per P1 |
| **future avenues** | **open by construction.** This row is not a placeholder — it is the directive above made structural. A channel of self-improvement that is identified later joins this table and is built, on the same terms as the rows above it: named blocker, then channel, then fill, then measurement. The five rows above are what has been *thought of*, not what is *possible*, and treating the table as complete is the failure this row exists to prevent |

### P1: worked case 2026-08-07 (the _G_INH derivation and the defence that no value could fix) (relocated 2026-08-30)

**WORKED CASE, 2026-08-07 — a hardcode retired by discovering it was already a derivation, and a
second one that could NOT be retired by tuning.** Both are in `thalamus/global_workspace.py` and
together they are what the corollary above means in practice.

- **`_G_INH = 2.4`, the "TOTAL lateral-inhibition budget", was `_W_SELF * capacity`** — an EXACT
  float identity (`2.4 / 4 == 0.6 == _W_SELF`), i.e. the per-slot excitation/inhibition balance
  multiplied back up by the capacity it was about to be divided by again. The constant was the round
  trip written down as a number. Retired; the shipped workspace is byte-identical without it, and a
  workspace at any OTHER capacity now stays E/I balanced instead of inheriting one capacity's
  arithmetic. **The tell was not the value, it was that the value only ever appeared divided.**
- **The incumbent's defence could not be fixed by any value, and that is the corollary's other
  half.** `self_boost + hysteresis = 0.25` exceeded the 0.20-wide band it operated in, so no
  compliant specialist could take the workspace head from any other. Displaceability needs
  `defence < (1 - lo)/2`; the same quantity is what `capacity = floor(1/defence)` divides, so every
  value satisfying it widens the workspace past the point where the bottleneck binds. Swept: the
  admissible window `(0.2000, 0.2250)` costs churn 9 → 15 and retention 47/47 → 45/47 at EVERY value
  in it, and the ignition-threshold axis has no admissible value at all. **Churn-clean and
  displaceable are disjoint.** (The sweep's baselines are the pre-inhibitory-rheobase kernel's, its
  own date; the same day's later change gave the shared inhibitory pool a rheobase and the stream's
  churn baseline is 4 since — the window argument stands, its absolute figures are that morning's.) The blocking constraint was the bug exactly as this policy says: ONE subsidy was doing
  two jobs (sustaining an assembly, and deciding which assembly is broadcast), which want opposite
  magnitudes. Splitting them cost nothing on either side. A constant that resists every value is not
  a tuning problem, and sweeping it is how you prove that rather than assume it.

### P1: the 2026-08-04 line-to-symbol citation conversion in the safety-constants paragraph (relocated 2026-08-30)

*(Citations in this paragraph converted from lines to SYMBOLS, 2026-08-04, after all three drifted:
`arm_config.py:22` resolved to `NEAR_MAX = 0.30` (`CONTACT` is at `:27`), `arm_config.py:29` resolved
to a comment about the trajectory-deviation reflex, and `serial_arm.py:46-52` resolved to the
`_FIXED` / `_ALLOWED_WHEN_STOPPED` token sets — not the calibration refusal, which is
`assert_calibrated()` some ten lines below. `reflex.py:22` was the one that still resolved. **Three of
four wrong, in the paragraph that tells a reader which safety constants are deliberately fixed** —
which is the argument for symbols over lines in the envelope's documentation generally, and the same
argument `SECURITY.md` acted on in the same pass.)*

### P2: the open two-workspace debt, measured 2026-07-28 (adjudication summary stays in POLICIES.md) (relocated 2026-08-30)

**OPEN P2 DEBT, measured 2026-07-28: two workspaces run additively.** `thalamus/global_workspace.py`
and `thalamus/reentrant_workspace.py` are both live and both fed every tick. Measured over 40 wake
ticks on a real brain: **72 submits each**, with 3 content values reaching both. Their public surfaces
are near-identical — `submit(source, content, salience)`, `subscribe`, `current`, `step` on both — and
`ReentrantWorkspace` adds `broadcast_to`, `association`, `access_strength`, `inbox`. On interface
alone it is a superset. That is exactly the shape this policy says must resolve to one survivor.

It is **not** resolved here, for the reason this policy already gives: *"better" requires a measurement
that exists*, and SNDL still has no consensus benchmark — no ambiguous multi-source evidence with a
known right answer — so neither can be shown better. Retiring `GlobalWorkspace` on interface elegance
alone would be the fashion-treadmill move this guard exists to stop, and it has **8 importers**
(`access_consciousness`, `executive_control`, `consciousness_probe`, `concept_reflection`,
`autonomous_loop`, …) against ReentrantWorkspace's 3.

What DID get fixed is a defect that had to be cleared before any comparison could be fair: the
`AccessConsciousness` wrapper was **constructed inside the wake tick**, so a fresh `GlobalWorkspace`
was built every tick — measured **25 constructions over 40 ticks** against ReentrantWorkspace's **1**.
Every instance took exactly one step, making incumbent defence, hysteresis and decay structurally
unreachable, while a comment eighteen lines below promised "ignited content is HELD across ticks, not
rebuilt from scratch each tick". Now persisted: **25 → 2** constructions, `access_index` still varying
across 25 distinct values. Any future A/B between the two workspaces would previously have compared a
persistent mechanism against one that was destroyed before it could express itself.

**The benchmark that would settle it**, stated so the debt is actionable rather than perpetual: a set
of ambiguous multi-source inputs with a known correct winner, scored on which workspace's ignited
content matches it. Until that exists, both stay and this entry is the honest record of why.

### P3 region-charge rule: the T039f census update (landing census 0/16 -> 8/16) (relocated 2026-08-30)

**UPDATED 2026-08-22 (specs/022 T039f): the count above is the LANDING census and is kept as
that record; it is no longer the current one. Insula is now REACHABLE AND CHARGED -- its US1
port landed, `cortex/brain.py` feeds the terminus the declared `body_in` frame and its
`feeling_salience` joins the ignition law -- so the census reads 8 of 16 CHARGED and the
UNREACHABLE set is olfactory and pons, both of which still own no modules. The floor artifact
carries the identity: `tools/region_charge_floor.json` is 8 with `named_and_NOT_charged`
empty. Re-derive with `region_charge_gate --assert` rather than quoting either number.**

### P3: the cross-tier import-manifest census narrative (mechanism statement stays in POLICIES.md) (relocated 2026-08-30)

*(Mechanism statement and derivation pin live in `docs/POLICIES.md` (P3, the cross-tier edge);
trimmed here post-relocation so one statement survives. The dated census:)* **Re-counted
2026-07-29, with the definition stated, because the old figure was unreproducible and so was its
correction:** `cortex/brain.py` has **111** bare `except Exception:` handlers (no `as name`), **39** of
which wrap a `try` body containing an import; across the 245 governed-tier files it is **142** bare and
**49** import-wrapping. This text said "67 such swallows in `cortex/brain.py` alone" and the audit that
flagged it said "112 / 203" — neither states what it counted, which is why neither reproduces. The
counter is `ast`-based and stated here so the next reader can re-derive it rather than inherit it. Delete or break one of those targets and every gate stays green while the
brain sets the faculty to `None` and ticks on without it — including `self_build_loop` and
`self_mod_harness`, which *are* P0's self-building-code avenue, and `rose_language`, which is language.
The canary derives the manifest by AST walk (never hand-maintained — the `_external_roots()` lesson) and
asserts each target still imports. Re-derive the census with
`python -m snn_data_lake.substrate.import_manifest` rather than quoting it from here — this sentence
carried "15 modules / 17 call sites" while the gate printed 7 / 9 (US1 audit, M7). It proves
import-ability, not charge; that is deliberate, because
import-ability is precisely what a deletion breaks and what the swallows hide.

### P3: the dormancy family -- the seven observed shapes, original worked-example table (relocated 2026-08-30)

### The dormancy family — seven observed shapes of "live but inert"

The module gate asks one question: *is this module imported from a live root?* Every defect below
answered **yes** and was still dead. They are the same failure wearing different clothes, and naming
them is most of what makes the next one findable. All seven are real, all were found by audit rather
than CI, and all are cited from this tree:

| # | shape | worked example |
|---|---|---|
| 1 | **method never called** — the module is live, one of its methods is not | `PredictiveCodingNetwork.learn()` — inference ran every tick, learning had one caller: its own `__main__`. Fixed v4.71 |
| 2 | **return value discarded** — a live call whose result goes nowhere | `connectome_router.step()`; its only export reached telemetry alone, measured constant 0.250 over 122 ticks. **SINCE FIXED** (`51bf5253`, v4.80): the return is consumed at `cortex/brain.py:2010`. The old cite `:2001` had also drifted onto an unrelated line — corrected 2026-07-29, because this table is the reference readers copy from |
| 3 | **key-space mismatch** — a learned path structurally unreachable | `thalamus/reentrant_workspace.py` potentiates on `(source, winner)` at `:271`, looks up `(subscriber, winner)` at `:289`; disjoint live, so the learned re-route took 2 values in 122 ticks |
| 4 | **write-only consumer** — a faculty fed every tick, never read | `cortex/agency_monitor.py` — `observe()` runs live; nothing reads `agency()` / `controllability()` / `status()`, and no key reaches telemetry |
| 5 | **constant wearing a signal's clothes** — present, plausible, never varies | `spikes.midbrain` (its novelty term is never enabled); `integration_competence` (assigned only in `Brain.__init__`); the `gated` trio (never *computed*, not merely never applied) |
| 6 | **silent default** — a lookup table missing the cases that matter | `_ALLO_MOVE` had no entry for the four `JOINT_*_ROTATE` tokens, so `.get(action, (0.0, 0.0))` meant the commands that actually move the arm contributed **zero** to path integration. Fixed v4.72 |
| 7 | **the registry nobody registered with** — a plug point, called every cycle, with zero plugs | `substrate/region_wiring.py` is *the* documented per-cycle faculty-charge registry, and its docstring asserted that "the daemon imports `wired_faculties` (which registers every adapter) at boot". **There is no `wired_faculties` module** (re-verified 2026-08-04: zero files of that name anywhere in the tree), `substrate/learn_daemon.py` imports only `region_wiring`, and nothing anywhere calls `wire()` outside the self-check — which clears `_WIRED` first, so it proves the mechanism and says nothing about production. `charge_all()` iterates `_WIRED`, which is initialised `[]` and never appended to on a live path, and the daemon logs `adapters={cov['n_wired_faculties']}` = 0. Docstring corrected; **the registry is still empty as of 2026-08-04, and filling it is open P3 work.** *(Path corrected: this said `learn_daemon.py:291`, and the root `learn_daemon.py` is now a 32-line shim — the daemon was promoted to `substrate/learn_daemon.py` under P4c and this row, like five other documents, kept citing line numbers into a file that no longer has them.)* |


### P3 amendment: the predictive-coding worked case and the 2026-07-31 runtime re-adjudication (relocated 2026-08-30)

**Why module granularity is not enough — a worked case, since fixed.** `dormancy_gate.py:71` keys on
the module basename, so a module passes P3 clean with half its capability dead.

`cortex/predictive_coding.py` was exactly that. The brain settled its 3-level Rao-Ballard net on every
wake tick (`cortex/live_brain.py:161`) with the error-neuron architecture implemented faithfully — while
`PredictiveCodingNetwork.learn()` had **one** call site in the tree, inside the module's own `__main__`
self-check. The live path called `reset()` + `infer()` only, against weights seeded random at
construction that never changed (verified: `w1` was `array_equal` before and after 40 further wake
ticks). Inference wired, learning never called, and the module-level gate saw a live, imported,
fully-compliant module.

**That case is now closed** (v4.71): the per-sample rule was extracted as `learn_step()`
(`predictive_coding.py:244`) and is called every tick at `cortex/live_brain.py:185`. Measured over 400
ticks — prediction-error energy −67.8% with the call, +9.1% without it (weights frozen).

Fixing it does not retract the argument; it completes it. The blind spot is no longer hypothetical: a
half-dead module demonstrably passed the module gate, for as long as that feature had existed, and was
found by an audit rather than by CI. That is the evidence this amendment rests on.

**RE-ADJUDICATED 2026-07-31 at RUNTIME** (`specs/001-charge-the-dormant`, and see
`docs/coverage/README.md` for the per-rule evidence). Seventeen learning-verb methods were resolved by
following the RECEIVER and instrumenting real tick runs, then adversarially reviewed: 8 LIVE with
charge proven, 6 never called on a live path, 3 called on the right receiver and changing nothing.
Dispositions: 2 permanent exemptions (dormant AND correct to be dormant), 4 OPEN exemptions each
carrying a disposition and a port finding, 3 measured inert. **Nothing was wired or retired** — every
port check found either a live class or an unported capability, and P2 forbids deleting either.
Two instrument defects were fixed en route: exemption keys are now class-granular (`module.Class.method`;
`n_defs` 56→58, because two same-named defs in one module had been overwriting each other and the
finding was reported against the wrong class), and `charge_probe` gained a graded observable after its
argmax flip counts were shown to report live faculties as inert.

### P3 amendment: candidate-set history, the three load-bearing implementation details, adjudications (relocated 2026-08-30)

The candidate set
below was produced by the import-scoped pass described next and has NOT been re-adjudicated since the
predictive-coding fix — treat it as a starting list to verify, not a verdict.

**Mechanization: BUILT — `substrate/method_dormancy_gate.py`, in `run_selfchecks` CORE.** A pure
static AST pass (~1 s, no runtime cost). Re-derive the census with
`python -m snn_data_lake.substrate.method_dormancy_gate` rather than quoting a count from here —
this paragraph carried "56 scanned / 16 UNCERTAIN / 5 exemptions, one standing TODO" for weeks
while the gate printed 66 / 20 / 9 with three standing TODOs (US1 audit, M7). What made it
adoptable is the SHAPE, which is stable: dormant asserts at 0, an earlier draft produced
59 failures — the rubber-stamp shape that killed a different proposal above.

**Three details are load-bearing, each found by implementing the naive version first and watching it
fail.** The first two were predicted when this section was written; the third and the correction to
the second were found by building it:

1. **Import-scoped resolution.** Matching bare `ast.Attribute` call names across all reachable modules
   scores 0 failures / 47 UNCERTAIN — the name `learn` resolves in 20 unrelated modules, so it cannot
   even catch its own motivating case. A call site counts only if its module imports the defining
   module.
2. **Same-module callers COUNT, except under `if __name__ == "__main__"`.** Excluding the defining
   module produces a real false positive (`plasticity/meta_plasticity.py:261` legitimately drives
   `_Unit.learn` at `:131`). The `__main__` guard is precisely what isolates the genuine case.
   **Correction found by building it:** that guard block is almost always a single line — `main()`.
   The self-check's real body lives in a top-level `def main()`, so marking only the literal `if`
   counts every assertion inside `main()` as a live caller. The first run scored 0 dormant *while its
   own motivating case sat there dead*. The functions the guard calls must be marked too.
3. **A function used as a VALUE is used.** `reasoning/arc_mint.py:224-228` registers its fitters in a
   dispatch table — `("affine", 3, _fit_affine)` — so a pass inspecting only `ast.Call` reported all
   five as dead: 5 false positives out of 12. Bare `Name`/`Attribute` references count as use.

**Scope, and it is the difference between adoptable and useless.** The gate audits a learning-verb
**method** (inside a class) on a module the **module gate already proves reachable**. Widening either
half re-introduces the 59-failure list — mostly module-level helpers inside `reasoning/` and
`training/` files that the module gate deliberately scopes out.

**Adjudication of the candidate set, now done.** The five-candidate list this section used to carry
was stale, as it warned. Re-run under the built gate: `cortex/predictive_coding.py` wired and
charge-proven (v4.71); `plasticity/meta_plasticity.py:131` and `neuromodulation/drive_reduction.py:196`
confirmed **non-findings** — both have real non-`__main__` callers and reach the live path through the
birth-time `census()` chain (`cortex/brain.py:379-385`), so they belong to the charge tier, not the
call-site tier. `neuromodulation/empowerment.py:86` `reward()` is genuinely dead — its only call site
is inside its own `main()`, and the live path calls the sibling `step_reward()` (`:91`), which
recomputes rather than delegating; it is a **retire** candidate, and it is out of this gate's scope
because it computes a signal rather than applying an update. `hippocampus/affective_memory.py:173`
`consolidate()` is **confirmed dormant by the gate itself** and held as an open `wire` exemption.
The gate also found one the list never had: `neuromodulation/drive_reduction.py:238 learned_map()`,
a readout the verb match catches on the word "learn" — exempted as a named false positive rather than
narrowed away, because narrowing to dodge it would also hide accessors that genuinely should be read.

### P3 amendment: what it would not have caught + the standing caution (relocated 2026-08-30)

**Stated honestly: this amendment would have caught no *earlier* incident.** The audited history before
it is *consumption* gaps (a producer with callers whose output nothing reads) and an unsupplied
constructor argument — a call-site check passes all of them. It earns its place on the predictive-coding
case above: found by audit, confirmed by measurement, and invisible to the module gate throughout.

**Standing caution for this section.** It was written while its motivating case was still open, and went
stale within two commits when that case was fixed — the text asserted dead code that had become live.
An amendment justified by a live example has to be re-read whenever that example changes, or the policy
file starts describing a codebase that no longer exists.

### P4b: the episode_consolidation measurement that produced the policy (relocated 2026-08-30)

**The measurement that produced this policy.** `hippocampus/episode_consolidation` formed real
autobiographical episodes into an in-memory attractor for months of development. Measured
2026-08-04: **59 episodes stored over 600 ticks with `persist=True` added 0 records to the lake**,
and **149 episodes over 1,500 ticks were recalled 0 times**. The brain was forming memories into a
second store that nothing read and nothing kept, and every one died with the process. It looked
healthy from every angle except the one that mattered -- `retained()` reported 1.000 throughout,
because the store was perfectly recalling episodes it was about to lose entirely.

### P4c: the dead-E-STOP measurement that produced the policy (relocated 2026-08-30)

**The measurement that produced this policy.** `substrate/cockpit.py::serve()` accepted an `acts`
queue and then did not forward it when constructing its handler, so **every arm control on the one
frontend was inert, including the E-STOP**. The page rendered, the buttons pressed, nothing moved.
It survived review because the cockpit was filed as a frontend rather than as the brain's motor
surface: no one ran P3's reasoning over a UI control, because P3 was understood to be about faculties.
Under P4c that button is a faculty, and the defect is a dormancy failure that three gates should have
caught. (Fixed 2026-08-04, specs/010 C10, with a regression test proven red without the fix.)

### P4c: the measured non-compliance debt list (CLAUDE.md exception rule stays in POLICIES.md) (relocated 2026-08-30)

**Measured non-compliance at the moment this policy lands** — stated as debt, not as a settled state.
This is a BUILD LIST in P0's sense: every item is work that will be done, never a boundary to accept.

- **`learn_daemon.py` — CLOSED 2026-08-04.** It was 370 lines at the repo root, in no REGION, in
  neither `run_selfchecks.py` list, not a `dormancy_gate` entrypoint, with **no self-check and not
  one assert**: the most load-bearing process in the project and the least verified file in it. Now
  `substrate/learn_daemon.py`, registered and self-checked, with a thin root shim preserving the
  documented command. The move immediately exposed three latent bugs that being ungoverned had
  hidden — `os.chdir()`, `sys.path.insert()` and `os.environ.setdefault("NORD_GPU", "1")` all ran at
  MODULE SCOPE, and `run_selfchecks.py` runs part of the suite in ONE process, so importing the
  daemon under the gate would have changed the working directory and the GPU preference for every
  module after it. Being outside the gates is not a neutral state; it is where bugs go unmeasured.
- **`substrate/cockpit.py` — governed as code, was mis-CLASSIFIED.** The layout map called
  `substrate/` "infra (NOT a brain region)", which is the exact framing that let a dead E-STOP
  through review. Corrected: infra by directory, brain by function.
- **~165 modules remain in `src/snn_data_lake/reasoning/`** — inside the package but outside the
  three P3 gates, the coverage denominator and the ruff rules. Under P4c every one of them is part
  of the brain and belongs in a governed region or retired. A promotion program is running against
  this (12 promoted as of 2026-08-04); the count is debt, and the target is zero.
- **`research/`, `training/`, `integrations/`, `gcp/`, `infra/`** are ruff-excluded tiers holding
  further builds. Each is P4c debt on the same terms.
- **`CLAUDE.md` is UNGOVERNED BY CHOICE — the one named exception to P4c, owner decision 2026-08-05.**
  P4c says a file is part of the brain or retired, with no third category. This is a third category,
  and it is written here rather than left silent because a documented exception is auditable and an
  undocumented one is just a gap. The file is gitignored as local-only and is NOT published to the
  public repo; tracking it would be a disclosure decision, not a hygiene fix, and the owner declined
  it. **The accepted cost, stated exactly: its 18 `file.py:NNN` citations are checked by nothing.**
  `tools/citation_guard.py` scans `git ls-files`, so an untracked file is invisible to it by
  construction — the same property that makes the guard trustworthy on tracked files makes it blind
  here. Anyone editing `CLAUDE.md` owns the accuracy of its citations personally, and this project
  has already found that hand-maintained line numbers rot faster than anything else in it.
  The GATE-side half of this was fixed rather than accepted: `tools/doc_command_guard.py` globs the
  filesystem instead of git, so it had been scanning `CLAUDE.md` and reporting **55 documents on the
  owner's machine against 54 in CI and in every clone**. A gate whose corpus depends on who runs it
  cannot compare two runs, so `CLAUDE.md` is out of its `SCOPE`. Measured delta: 1 document,
  0 commands.
- **`docs/coverage/artifacts/` holds 16 EXECUTABLE evidence artifacts and not one of them is
  governed** — measured 2026-08-05. All 16 are `.py` files that produce the numbers the coverage
  bands are scored on, and **16 of 16 contain zero `assert` statements** and **16 of 16 carry
  module-scope side effects** (`sys.path.insert` / `os.environ.setdefault` / `os.chdir` /
  `random.seed` at import). That is the exact triple the `learn_daemon` promotion is recorded above
  as having exposed, in sixteen files that no REGION, no `run_selfchecks.py` list and no P3 gate can
  see — while their output is cited as evidence throughout `docs/coverage/`. Under P4c a benchmark
  is not a third category. Promoting them requires edits to `src/snn_data_lake/__init__.py` and
  `run_selfchecks.py`, so it is scheduled work rather than a docs fix.
- **`tools/doc_command_guard.py` is built, clean, and registered nowhere** (2026-08-05) — no CI
  workflow, no `run_selfchecks.py` entry, no pre-commit hook. It is the newest instance of this
  policy's own honesty guard: *a build the gates cannot see is not a build the project can claim*,
  and it was written into three documents as a CI step before it was one.

### P4c amendment: the 18-commits-behind measurement that produced the amendment (relocated 2026-08-30)

**The measurement that produced this amendment, and it is this policy's own honesty guard failing in
a new direction.** On 2026-08-16, `origin/main` was **18 commits behind** a branch that had been
fully pushed for days. Every commit was green — `run_selfchecks` 301/301, `pytest` clean, ruff clean.
What `main` was missing: **two safety-path fixes** (the ESP32 idle sentinel binding a fear
association to a context where nothing was there, on every idle sample; a 3x units-contract gap on
the distance threshold), **a determinism-invariant fix** (`torch.Generator(device=...)` meant the
same seed grew a DIFFERENT brain on a machine with a card than on CI), and
`substrate/eprop_transport_gate.py` — **the only mechanization the invariant had until 2026-08-18**,
when `substrate/gradient_claim_gate.py` joined it: a second CI-asserting e-prop gate that
classifies every governed learning rule as owing a cosine or not, ratcheting MISSING_DECLARATION
and UNDETERMINABLE, plus the absolute ALL-scope PHANTOM_DECLARATION bound (2026-08-27)
(`--assert` + `--self-check` in the lint job).
Anyone cutting from `main` got none of it, and nothing in this file said that was a defect.

**Why the existing guard did not catch it.** P4c's guard is *"a build the gates cannot see is not a
build the project can claim"*, and every gate COULD see this work — on the branch. The gates run
where the work is. Being invisible to `main` is a different failure from being invisible to the
gates, and it is the one that decides what the project actually has.

### P4c amendment: how the closing merge was verified on origin/main (relocated 2026-08-30)

**Honesty guard:** "it is pushed" is a claim about a BRANCH, not about the project — the same shape
as P4b's *"'it persists' is a claim about a NEW process"*. Prove it by reading the artifact back out
of `origin/main` after the merge, not by inspecting the branch you merged from. The merge that
closed this incident was verified that way: the guard, the CPU-bound generators and the gate file
each confirmed present ON `origin/main`, with a negative control (the temporary `RED PROOF` marker
used to prove a test's red) confirmed ABSENT, so the check could distinguish a real merge from a
grep that would have matched anything.

### P5a: the .venv-cu128 retirement sequence and the CUDA wheel hunt (relocated 2026-08-30)

**`.venv-cu128` IS RETIRED, 2026-08-05, and the gate env `.venv-lake` now carries CUDA itself.**
Measured on the unified env: `python 3.14.6 free-threading build` / `_is_gil_enabled() False` /
`torch 2.11.0+cu128` / `cuda.is_available() True` / `NVIDIA GeForce RTX 5070`. One environment with
both capabilities supersedes two with half each, which is P2 (adopt-or-merge, then retire the
inferior) applied to environments rather than modules.

**WHY IT HAD TO BE RETIRED RATHER THAN KEPT, and the sequence is recorded because it was a
self-inflicted break.** `requires-python` went to `">=3.14"` and `snn_data_lake/__init__.py` gained a
runtime refusal on GIL-enabled interpreters (both 2026-08-05, free-threading is now REQUIRED).
`.venv-cu128` is **Python 3.11.15**, so from that moment the package could neither be installed into
it (pip refuses on the floor) nor imported there (the guard raises). The GPU became UNREACHABLE, and
the commit that did it claimed the adoption cost nothing — the claim was written without checking.
The fix was a wheel that had to be looked for rather than assumed:
`torch-2.11.0+cu128-cp314-cp314t-win_amd64.whl`, a CUDA build for the FREE-THREADED ABI, from
`https://download.pytorch.org/whl/cu128`. The default PyPI index serves CPU-only torch on Windows,
which is why a plain `pip install -e .` had silently produced `2.13.0+cpu` with `cuda False`.


### P5a: the open venv-vs-requires-python floor disagreement, measured 2026-08-05 (relocated 2026-08-30)

**OPEN, 2026-08-05 — THE CPU GATE ENV NO LONGER CONFORMS TO THE PROJECT'S OWN FLOOR, and it is
recorded here because this is the section that names `.venv-lake` as the gate env.** `pyproject.toml`
raised `requires-python` from `">=3.11"` to `">=3.12"` on 2026-08-05 (the reason and the full
measurement live in the comment block beside that field). `.venv-lake` is **Python 3.11.15** —
measured twice, `pyvenv.cfg` reads `version = 3.11.15` and `.venv-lake/Scripts/python.exe --version`
prints `Python 3.11.15`. pip enforces the field, so `pip install -e .` into `.venv-lake` now FAILS:
constructed and run in that venv against two wheels differing only in that metadata line, the
`>=3.11` control printed `Would install` and exited 0 while the `>=3.12` case printed
`ERROR: Package 'reqpy-floor312' requires a different Python: 3.11.15 not in '>=3.12'` and exited 1.
**Every measurement recorded in this file was taken in that venv.** It is deliberately NOT recreated
as part of the floor change — recreating it is an owner decision and it re-bases the environment
behind every number in `docs/`. Until it is recreated on 3.12 the gate env and the declared floor
disagree, and an already-installed `.venv-lake` keeps running (the check is at install time, not at
import time), which is precisely why this could sit unnoticed if it were not written down.

### P5a: width/crossover measurement history, the 66x GEMV fusion, profiler lessons, numba/spikingjelly verdicts (relocated 2026-08-30)

**Update (2026-07-28): that width now exists, on the growth tier.** With the NG-SNN earn-your-place
stop corrected (it had been scoring rows the solve already fit — see
[PERFORMANCE.md](PERFORMANCE.md)), grown capacity is monotone in data and reaches **4,512 units at
24,000 rows**, past the 4,096 crossover, on plain MNIST at 8x8. The earlier reading — that capacity
saturates near 2,000-2,600 and that GPU-relevant width would need a harder problem — was an artifact
of the broken criterion, not a property of the task.

This does **not** move the live tick: those populations are still 24-240 and the conclusion above
stands unchanged. What it changes is the `[train]`/`[gpu]` tier, which now belongs on the card by
WIDTH and not merely by fit cost. It also relocates the thing worth re-measuring as the brain grows:
this capacity curve, not the kernel. And it is a second instance of the guard-(b) lesson — the first
saturation claim was refuted by fixing the instrument, exactly as the first two "the live tick should
be on CUDA" claims were refuted by measuring it.

**The live-tick CUDA perception path is measured at last, and it does not pay.** `cortex/brain.py:170`
defaults `device="cuda"` and `_perceive_real()` runs a per-frame batch-of-1 forward pass every wake
tick -- a GPU path that shipped without the benchmark honesty guard (a) demands. On the real live shape
(1 frame x 15 timesteps x 2x34x34 -> 128 -> 64): **torch-CPU 0.073 ms, torch-CUDA 0.073 ms, 1.00x.**
Not a regression like the 0.01x at width 38, but not a gain either -- it holds VRAM and adds a
host-device transfer dependency for nothing. The default is LEFT AS IS pending a bit-exactness check,
because guard (d) cuts both ways: moving a live numeric path OFF CUDA carries the same burden as moving
one onto it. Flipping it is a follow-up with a controlled A/B, not a free win.

**The original measurement, still valid.** For SNDL's elementwise
spiking-update kernels, eager torch-CUDA on this card beats numpy-on-CPU only between
**N ≈ 24,576 and 32,768** elements. Every population on the live tick is **24–240** wide — 100× to
1300× short of that. Consequences, all measured:

- Vectorizing `cortex/prefrontal_executive.py::_decide_spiking` at its real width of 38 **regresses**:
  0.36× on numpy, 0.05× on torch-CPU, **0.01× on torch-CUDA**. Kernel-launch latency alone exceeds the
  whole per-call budget.
- `cortex/lif_neuron.py::step` looks like the hot path (244,768 calls per 10 ticks) and is not: neutering
  it entirely moves an awake tick 43.6 → 41.2 ms, a **6% ceiling**. Moving it to CUDA cannot beat that
  ceiling and would pay launch latency for every one of ~24.5k tiny calls.
- The real win on that hot path was **CPU-side batching**, not hardware: `AttractorNet.accuracy` folded
  36 independent recall trials into one array pass for a measured **6.93×**, bit-exact (v4.68).
- **And it happened again, larger, 2026-07-29 — with the widening plan refuted on the way.** This
  policy's standing invitation ("the moment a live population crosses the measured threshold, put it on
  the GPU") was taken up: the plan was to widen the live populations past the ~4,096 CUDA-graph
  crossover and move the tick. A runtime width inventory killed it. The whole live brain holds **132**
  per-object LIF cells, and *all* live per-tick neuron state fused into one flat vector is
  **~1,700–3,500** elements — inside the crossover band, not past it. `py-spy` on the live daemon then
  showed why widening would have optimised nothing: **`AutonomousArm.step()` is 0.00% of the profile**
  (`_c1_tick` 0.10%, `_tick_ensemble` 0.01%), while **74.5%** of samples sit in
  `cortex/vsa_reasoning.py::coincidence` — a Python loop run once per stored episode per tick, entered
  via `hippocampus/episodic_recall.py`. Fusing that as one GEMV, using the exact integer identity
  `coincidence(a,b) == (D + a·b)/2` for ±1 spike codes, measured **`complete()` 59.946 → 0.909 ms at
  N=2,000 (66×)**, bit-exact and tie-identical. The load-bearing part is not the factor: the scalar
  cost was **linear in the episode count** (fitted `ms/tick = 56.60 + 0.0804·N_episodes`) and the fused
  cost is **flat** (0.61–0.91 ms across N=60–2,000), so the growth term is gone rather than reduced.
  **Amended 2026-07-30 — the `0.00%` above is stale, and the way it went stale is the lesson.** Once the
  74.5% term was removed, the intercept was attributed by wrapped timing (sum 56.71 ms against 56.64
  measured): `AutonomousArm.step()` is **20.1 ms, 36%** of it, of which `_pfc_bias → pfc.plan →
  deliberate` is 15.8. Nothing about the arm changed — a profile share is a share *of what else is
  running*, so removing a dominant term promotes whatever sat underneath it, and a percentage measured
  before a fix cannot be quoted after one. The intercept's two largest terms were then removed
  bit-exactly (`AccessConsciousness._coherence` fused by the same `(a·b)/D` identity, and a discarded
  coherence no longer computed), taking the instrumented tick **56.64 → 27.78 ms**. Still open and
  deliberately not taken: `commit_races=25` is now ~46% of the remaining tick, but cutting it is
  **behavioural**, needs a decision-distribution A/B, and is a fixed constant in a decision path (P1).
  Two further results worth keeping so they are not re-litigated: for this GEMV the measured CPU→GPU
  crossover is **~1,600–2,000 rows**, so it ships on the **CPU** and a captured CUDA graph is a *loss*
  below that; and for fused LIF at our real widths **numba beats a CUDA graph by ~16×**. `spikingjelly`
  and `cupy` were rejected on a structural reason rather than a benchmark — their fused multi-step
  kernels collapse the **time** axis `T`, and the live tick has `T=1` by construction, so the axis
  those kernels exist to collapse is absent. Net new dependencies for all of the above: **none**.
  **Reframed 2026-08-04 under P7, because that last sentence reads as a score and is not one.** Every
  optimisation above was correctly decided on its measurement, and adding nothing was the right
  outcome each time — but "net new dependencies: none" is not evidence *for* those decisions and must
  never be quoted as though it were. P7 makes the dependency count a non-criterion in both directions.
  The one item here that P7 puts back on the table is **numba**, which won its benchmark by ~16x and
  was declined only because no live population reaches the width where the win exists: that is a
  measurement about WIDTH, so it is exactly the kind of finding P5 says to re-measure as the brain
  grows, and P0's target population makes the re-measurement inevitable rather than optional.

**Where the GPU therefore pays today, and should be used without hesitation:** the `[train]` / `[gpu]`
tiers — surrogate-gradient training, `hippocampus/ng_snn.py`, `cortex/neurogenesis_engine.py`, the
torch readout studies (promoted out of `reasoning/brain_adapters/` across the 2026-08-11/12 lanes;
now `hippocampus/growable_readout.py` and `hippocampus/developmental_loop_brain.py` — the
directory itself was deleted 2026-08-12), and any offline sweep over a real dataset
(a COMPUTE regime endorsement; gradient trainers there stay inside the e-prop SCOPE).
Wide, batched, repeated — the regime the crossover favours. The daemon
already sets `NORD_GPU=1` and runs under `.venv-cu128` so this tier lands on the card by default.

**Where it does not pay today:** the live wake/sleep tick, at current population widths. That is a
statement about widths, not a prohibition — **the moment a live population crosses the measured
threshold, this policy says put it on the GPU.** Re-measure when populations grow; the crossover is a
number, not a doctrine.

### P5: the two gate environments measured against each other + the nengo/ruff environment defects (relocated 2026-08-30)

**The two gate environments, measured against each other 2026-07-29 — and why neither is "retired".**
The instruction was to consolidate onto one environment and drop the CPU gate's veto. Measured, that
turns out to be half right, and the half that is wrong matters: **CI has no GPU.** Every job runs on a
GitHub runner, so a gate that requires this card cannot be the gate. What guard (c) below was really
protecting is not the CPU *environment*, it is the property that the suite is runnable without one.

So both are now verified equivalent rather than one being privileged. On a clean checkout at
`598b07dc`, `.venv-cu128` (torch 2.11.0+cu128, numpy 1.26.4): **278/278** self-checks, **473 passed**,
coverage **90.80%**. `.venv-lake` (torch 2.13.0+cpu, numpy 2.4.6): **277/277**, **473 passed**,
**91.11%**. The 278-vs-277 is nengo being installed in the CUDA env, so `NENGO_GATED` runs instead of
skipping. The coverage gap is the worktree having no `models/` (gitignored), so perception falls to the
stdlib tier and less code executes — not a behavioural difference.

**Two environment defects had to be fixed before that comparison was even possible, and both are worth
knowing because neither is visible from reading `pyproject`:**

1. **`nengo` ships a pytest plugin whose hook signature is invalid under pytest 9.1.1** —
   `pytest_report_collectionfinish(config, startdir, items)`. Autoload aborts collection with
   `PluginValidationError` before a single test runs, in any environment where nengo is installed.
   It presents as a total test failure with nothing to do with this code. `-p no:nengo` is now in
   `addopts`; it is inert where nengo is absent. **This is the actual reason the CPU env had been the
   only place pytest could start** — that was never a policy preference, it was a broken plugin.
2. **`ruff` was FLOORED at `>=0.15.20`, so installing the dev extra elsewhere resolved to 0.16.0** and
   created a third formatting authority disagreeing with both `.pre-commit-config.yaml`'s pinned rev
   and the 422 files formatted in-tree — arriving as a mysterious `ruff format --check` failure. Now
   pinned. A formatter's version is part of the format.

### P6: the 2026-07-31 first quantitative reading, the refuted remedy, and the census drift tables (relocated 2026-08-30)

**FIRST QUANTITATIVE P6 READING, 2026-07-31.** P6's gate is specified but not built, so until now it had
no number. The global workspace is the one structure in this brain shaped like a mutual-consumption bus
(the role a transformer's residual stream plays: many writers, and later computation READS it), so the
checkable proxy is its write:read topology.

Measured by AST over `src/snn_data_lake/` — **36 workspace writes across 8 modules, 2 reads across 2**.
The breakdown is the finding, not the ratio: **25 of the 36 writes are inside the two workspace modules
themselves** (`global_workspace` 14, `reentrant_workspace` 11), so real faculty broadcast is **11 writes
from 6 faculties** — `train_of_thought`, `autonomous_loop`, `access_consciousness`, `rpm_solver`,
`spike_chat_lake`, `executive_control`. The two readers are `cortex/live_brain.py:287` and
`reasoning/developmental_loop_brain.py:295` (the module is `hippocampus/developmental_loop_brain.py` since 2026-08-12; the cited site is a 3-arg WRITE per P6_INTERCONNECT's correction), and **`live_brain` is the tick driver** — the action path.

So the bus exists and is used as a **broadcast sink whose only live consumer is the action path**: the
single action gate wearing a bus's clothes. Not one faculty reads another faculty's contribution, which
is zero-way at the faculty level and precisely what this policy forbids.

~~**The cheap remedy this implies**: faculties should READ the workspace, not only `submit()` to it.
That is what makes one faculty's broadcast change another faculty's output — condition (b) — and it
needs no new router, only consumers on a bus that already exists.~~

**THE REMEDY ABOVE IS REFUTED, 2026-08-04 — there is no bus.** `tools/p6_gate.py` (which exists, runs
clean, and self-checks — see the enforcement correction below) reports **31 workspace instantiation
sites across 9 modules**, of which **16 sit outside the bus modules' self-checks, across 7 modules:
that many PRIVATE, MUTUALLY INVISIBLE workspaces.** Every faculty that "broadcasts" constructs and
holds its *own* `GlobalWorkspace`. `cortex/live_brain.py` is the single module that reaches into
another module's instance; every other reader would be reading only its own writes.

**Re-derived 2026-08-04 in `.venv-lake`, and the figures had ALREADY drifted from the ones this
paragraph was written with the same day** — recorded because this policy's own instruction is
"re-derive with `tools/p6_gate.py` any time", and the first person to follow it found different
numbers:

| reading | as written here | re-derived 2026-08-04 |
|---|---|---|
| instantiation sites / modules | 26 / 10 | **31 / 9** |
| PRIVATE instances / modules | 11 / 8 | **16 / 7** |
| writes / modules | 36 / 8 *(2026-07-31)* | **46 / 10** |
| reads / modules | 2 / 2 *(2026-07-31)* | **9 / 5** |
| **faculty-to-faculty reads** | **0** | **0** |
| **writer modules read by nobody** | **7** | **7** |

**The two rows that carry the policy's argument did not move**; every row that moved is a census. That
is the useful shape of this table: the finding is robust and the counts are perishable, so quote the
finding and re-run the gate for the counts. Note also that the private-instance count went **UP** while
the write count went up — which is precisely the "a rising write count is not progress" reading the
gate's own verdict line warns about.

The gate states the consequence in its own output: *"Adding a reader is NOT sufficient while each
writer holds a PRIVATE instance — there is no shared object for a reader to read."*

So the corrected reading is worse than "a bus with no readers": **it is a private bus per writer.**
Re-derived 2026-08-05: **16 PRIVATE instances across 7 modules**, against 10 writer modules of which
**7 are read by nobody**. *(This sentence said "eight separate buses with one writer each" — a figure
from the superseded 2026-07-31 census, left standing when the table two paragraphs above was
re-derived. Stated as a shape rather than a count now, because the count is the perishable half and
this file has now recorded that lesson three times while re-breaking it twice.)* The real remedy is
therefore NOT cheap and must be stated honestly — a shared
workspace instance has to exist before a reader can mean anything, and that is a construction-path
change (P4's `make_brain()` owning one workspace and handing it down), not a one-line consumer.
Recording this because the retracted sentence would have sent the next reader to add readers that
provably could not carry anything, and "we added consumers and the count did not move" is exactly the
force-wiring P3 forbids.

Re-derive with `tools/p6_gate.py` any time; a rising read count against a flat write count is what P6
progress looks like — but **a rising read count against a tree of private instances is not progress at
all.** *(Was "against 11 private instances"; the gate reported **16** when re-run on 2026-08-05, and
the table above had already been corrected to 16/7 while this line kept the old number — the exact
"a claim living in two places, only one corrected" failure this policy's own incident table names.
The claim is now stated in the form that cannot rot; re-run `tools/p6_gate.py` for the count.)*

### P6: real past incidents table (incl. the retracted neurogenesis row) (relocated 2026-08-30)

**Real past incidents it would have caught** — required by the bar below, and there is no shortage. Each
row was **re-verified against the tree on 2026-07-29**, because several were fixed or reverted in the same
session that wrote this policy, and a rule citing an already-closed defect as live is the drift this file
keeps suffering. Rows marked FIXED are kept deliberately: the bar asks what the policy *would have caught*,
and a closed incident is the better evidence precisely because its cause is known.

| incident | shape |
|---|---|
| `region_wiring` registry: 0 registrations, `adapters=0` live | the plug point nobody plugged into |
| `thalamus/connectome_router.step()` return discarded, export a constant 0.250 — **since FIXED** (`51bf5253`, v4.80); the return is now consumed at `brain.py:2010` | computed, reached no faculty |
| `cortex/agency_monitor.py` `observe()` every tick; `agency()`/`controllability()`/`status()` zero readers | write-only organ |
| `claustrum/claustrum.py` `broadcast()` (defined `:118`) zero live call sites — its only caller is `:193`, inside its own `__main__` | a broadcast hub that broadcasts to nobody |
| `reasoning/pragmatics.py` reachable only from `spike_chat_lake`, no `brain.py` caller | wired to chat, not to the brain |
| `reasoning/arc_solver.py` — "imports stdlib only, with zero wiring to any perceptual machinery" | Gf isolated from Perception |
| `reasoning/nightly_skills.py` forms and persists a skill every night; `route()` called only from its own `main()` | the brain never uses a skill it formed |
| ~~`cortex/neurogenesis_engine.py` has no persistence path, so grown units die on every restart~~ — **RETRACTED 2026-08-04, the claim was false when written and stayed false here after the P0 row was corrected.** Retention is at `:211`, serialization at `:250`/`:279`, and it is called on two live paths into the lake under `_T_GROWTH`. Kept struck-through rather than deleted because a policy file that quietly removes its own wrong incidents loses the record of having been wrong | *(was: a faculty that cannot even reach its own future self)* |

The last one **was** the sharpest, and it is now **RETRACTED** (2026-08-04): the grown substrate IS in
the one durable lake, under `_T_GROWTH` (since specs/022 T010-ii: under the `_T_GROWTH_DELTA`
chain), written from the sleep tick and from construction. The
sentence that stood here — that the avenue "resets ~20 times in a working session and never
compounds" — described a tree that had already been fixed. Two lessons are worth more than the
incident was: a grep (`self.*model*=`) missed an assignment written `self._model = model` and the
null it returned was published as a finding; and this row survived a *later* edit to the P0 avenue
table because the same claim lived in two places and only one was corrected. **When a policy claim
appears twice, fixing one copy is not fixing it.**

### P6: mechanization history -- the two gate defects, the rejected repair, disclosed blind spots (relocated 2026-08-30)

**Mechanization: HALF BUILT — corrected 2026-08-04, this paragraph said "NOT BUILT" while the tool
existed, ran, and self-checked.** *"Adding a policy: implement the gate before adopting the rule… its
output counted, before the policy lands."* Two prior proposals died there. P6 landed
**review-enforced and provisional** on the owner's instruction; the honest current status is three
parts, not one:

1. **BUILT** — `tools/p6_gate.py` derives the workspace write:read topology by AST over
   `src/snn_data_lake/`, parses 400 of 400 files, and carries a real self-check ("is this walk still
   alive?": writer/reader modules found, classes constructed, arity split, roles re-derived).
2. **ASSERTING AND IN CI as of 2026-08-06** — `--assert` (five criteria) and `--self-check` are both
   CI steps. It ships as a **RATCHET, not an arrival**: at landing `faculty-to-faculty > 0` was RED
   (it moved 0 → 1 at T039h, 2026-08-22 — the floor now lives in
   `tests/test_interoceptive_specialist.py` with the read's identity pinned), and a gate that must
   be disabled on the day it lands is not a gate. Baselines were this walk's own output
   at `4fc5fb1c`, every one ±1 from red; re-derive current values from the gate, never from here.
3. **BUILT** — the connectome-diff half derives the faculty→faculty consumption graph and diffs it
   against `connectome.py`'s declared edges, in both directions: `undeclared_import_pairs` (traffic on
   no declared route) and `unimplemented_declared_pairs` (a declared route with no traffic).

**HOW THE SHAPE WAS DECIDED, and it is the reusable part.** A six-commit history sweep with the
instrument held constant at the new file, measuring each candidate across `6b33a45d → 4fc5fb1c`,
then pinning each at N−1 and asking whether it reddens N. **Private-instance SITES fires on
`c7e1b816` (16→18) and `acd4627a` (18→22) — the two commits that are this project's actual P6 work.
Rejected by measurement: a gate that reddens the repair is worse than no gate.** `faculty-to-faculty
must not FALL` was also rejected — a count cannot fall below 0, so it has **no reachable red at all**,
and F11 says deleted, not weakened; it is printed, and becomes a floor the day it is non-zero.

**The flat rows are the finding twice over: three commits of P6 work moved writes +24 and reads +45
and moved STRUCTURE BY ZERO.**

**TWO DEFECTS IN THE GATE ITSELF, found by adversarial review and fixed at zero baseline cost.** Both
are the same class and neither was visible from reading:
- **`fac_cross` was ONE-SIDED.** It filtered `role_of(reader) == FACULTY` and never checked the
  OWNER's role, so a faculty reading `self.arm.workspace` — the ACTION gate's bus — moved the headline
  number 0 → 1 while a both-endpoints-FACULTY filter stayed at 0. **The gate's own headline was
  satisfiable by the exact move P6 verbatim calls "the DEFECT P6 names, not a way to satisfy it."**
- **`unread_writer_modules` reddened the FIX.** Writes were attributed to the writing module rather
  than to the instance written INTO, so a new faculty submitting into the shared bus that
  `cortex/live_brain.py:313` reads — output genuinely consumed by another module — took it 7 → 8 and
  exited 1. **Missed because the criterion had only ever been run against REGRESSIONS, never against
  a FIX.** That is the general lesson: a candidate gate must be run against the change it exists to
  encourage, not only against the change it exists to catch.

Also corrected in the walk: the alias hop resolved only the local form (`ws = ...`) and not the
attribute form (`self._ws = self.arm.workspace`); `region_imports` ignored relative imports entirely.
A repair was **rejected by measurement** and recorded so it is not re-proposed — resolving an
ambiguous receiver by CLASS takes `cross_module_reads` 1 → 4, and three of the four "cross" reads it
invents are local objects inside a self-check attributed on a naming coincidence. *"This walk does not
know"* is the correct answer to a real ambiguity.

**KNOWN AND DISCLOSED, not fixed:** a faculty handed its bus by dependency injection writes through a
receiver this static walk cannot attribute and is still counted as unread (1 of the 7 today,
`reasoning/spike_chat_lake.py`) — strict was chosen over lenient because excluding unknowns lets any
new writer evade the ceiling by taking its bus as a parameter, and the count is PRINTED so it is never
mistaken for a finding about the tree. **If a future commit hits this red, the answer is to spell the
wiring attributably or to say so in the commit — not to raise the ceiling.** Separately, **241 of the
400 `.py` files** sit in directories `connectome.py` declares no region for, so a cross-region wire
routed through any of them is invisible to `undeclared_import_pairs`; closing that is a
`connectome.py` change. Dynamic `importlib`/string-name imports are invisible to the walk.

**How this went stale is the lesson, and it runs the opposite way to the usual drift.** The tool
landed 2026-08-03 (`5868e1c1`); this file was committed *again afterwards* still saying the gate did
not exist. The failure was not a claim rotting as the tree moved on — it was a claim that was already
false when it was last written, because nobody re-read the sentence they were editing around. Its own
output is what corrected it, and its output also **refuted this policy's prescribed remedy** (the
private-instance finding above), which is the strongest possible argument for the gate-first rule this
paragraph exists to discuss.

### P7: mechanization correction history and counted ledger output (relocated 2026-08-30)

*(Corrected 2026-08-04: this said "NOT BUILT" while the gate existed, ran, and was wired into CI.
**Corrected again 2026-08-05, and this paragraph is now the third instance of the drift the top of
this file names.** It read "BUILT, REPORT-ONLY … It stays **report-only**, deliberately and with the
reason stated in the workflow: it exits non-zero on findings nobody has dispositioned yet … Flip to
`--assert` as the last step of specs/010 T021/T062." Two separate defects in those two sentences:*

1. *The flag was flipped earlier the same day, and the row in the enforcement table at the top of
   this file was updated while this paragraph — P7's own mechanization statement — was not. A claim
   living in two places with only one corrected is the failure P6's table names, and here both
   places are in this one file.*
2. *The justification was **structurally false when it was written**, not merely stale. The bare
   invocation could never exit non-zero on anything: `main()` returns 1 only under
   `if do_assert and failed`, and `do_assert` is `"--assert" in sys.argv`. RUN rather than read —
   `main()` driven against a mutated string copy of `pyproject.toml` (`nir` removed from `[all]`,
   which re-opens the ALL-EXTRA finding recorded as CLOSED above) returns **rc=1 with `--assert` and
   rc=0 bare**, printing the identical `gate FAIL: the dependency ledger does not balance` in both.
   So "reddening the build for an undecided finding is not the same as enforcing a rule" described a
   trade-off that was not on the table: the step was a log line either way.)*

Both steps re-run in `.venv-lake` after the flip: `--assert` exits 0 (`gate OK`) and `--self-check`
exits 0 (`5/5 criteria proven falsifiable`). **Each was also proven able to FAIL, which is what the
2026-08-04 version of this paragraph never established for the step it described.** `--assert`:
rc=1 on the mutated copy above. `--self-check`: rc=1 on **both** of its failure paths — an anchor
that no longer matches prints `BROKEN` (mutating `arm = ["pyserial"]` to single quotes) and a
criterion whose mutation cannot fire prints `NO RED` (dropping `mujoco==3.11.0` from
`[project.dependencies]`, after which neither `CORE-PROSE` red can light). A gate that has only ever
been seen green is indistinguishable from one that cannot see.

**Counted output on this tree, RE-RUN 2026-08-05 in `.venv-lake` after the `arc` declaration and the
`requires-python` floor raise** — which is what the gate-first bar asks for:

```
scanned 400 files under src/, 29 distributions in pyproject
UNDECLARED: 0   UNUSED: 0   MISDESCRIBED: 0   EXEMPT: 2   ALL_EXCLUDES: 8
gate OK: every import is declared, every declaration is used, prose matches lists
```

*(Re-run 2026-08-05 and CORRECTED TWICE. First: this block read `UNDECLARED: 1   UNUSED: 0
EXEMPT: 2` and listed only `arcengine`. The gate emits five counters, not three, and the
`MISDESCRIBED` one was **1, not 0** — a live finding nobody had dispositioned. Quoting a gate's
output means quoting all of it; dropping the counters that are non-zero is how a work list becomes a
clean bill of health. Second, later the same day: the block was then left asserting
`28 / 1 / 1 / 7` and the two findings below it — `arcengine` UNDECLARED and the `[nir]` ALL-EXTRA —
**after both were closed in this same file**, once in the gate table near the top and once in the
"Two corrections" entry beside it. Three copies of one measurement, two re-derived and one not, and
this was the copy captioned "Counted output on this tree", i.e. the one a reader would trust most.
The durable fix is not a better caption: **quote a gate by re-running it, or do not quote it.**)*

*Its work-list character is not lost by the counts going to zero — what makes the ledger a work list
is that `EXEMPT` and `ALL_EXCLUDES` carry written dispositions that must each still match a real
violation, and the gate fails loudly on one that blesses nothing.*

**Of the three properties the specification below predicted would fail, two now PASS.** Recorded
because the specification is evidence about what the gate was for, and its predictions being
overtaken is the gate working:

1. *"every declared dependency is imported somewhere, or carries a reason"* — predicted to fail on
   `pillow` and `pyarrow`; **passes**, both carry reasons.
2. *"every third-party module imported under `src/` is declared"* — predicted to fail on
   `scikit-learn` and `nir`; **now passes.** scikit-learn is declared in `[gpu]`/`[train]`, `nir` in
   `[nir]` and in `[all]`, `arcengine` in `[arc]`. *(This item read "now fails only on `arcengine`"
   until 2026-08-05, when the floor raise declared it; re-derived, `UNDECLARED` is 0. Note what the
   green does and does not prove — the ledger's own docstring is explicit that a distribution
   declared only in an EXTRA is green here while a plain `pip install .` still raises
   ModuleNotFoundError, which is exactly `arcengine`'s shape.)*
3. *"`pyproject.toml`'s own self-description matches its dependency list"* — predicted to fail because
   `:1` and `:27` said *"torch + numpy"* while `mujoco==3.11.0` is a third core dependency;
   **fixed in `pyproject.toml` itself**, whose `:1` and `description` now name all three. **This
   property IS checked by the built gate — corrected 2026-08-05.** The paragraph here said it was
   *"a prose-vs-data check the ledger does not implement, so it is review-enforced"*; the ledger's
   own docstring quotes this specification verbatim and states that the `MISDESCRIBED` direction
   was added on 2026-08-04 to satisfy it. Run 2026-08-05: `tools/dependency_ledger.py --self-check`
   MUTATES `pyproject` to construct a red for every criterion and reports **`5/5 criteria proven
   falsifiable`** (`CORE-PROSE` x2, `CORE-ONLY`, `ALL-EXTRA`, `STALE-EXCLUDE`), which is P8 burden 1
   discharged. Marking a mechanized check "review-enforced" is the same drift as marking a built gate
   "NOT BUILT", pointing the other way: it costs the gate its credit and invites someone to build
   it twice, which is the additive shape P2 forbids.

Note what this gate can and cannot do: it checks the **ledger**, never the judgement. No gate can
tell you a framework was refused for a bad reason, which is why guard (b) is enforced by review.


### P8: counted red_reachability output 2026-08-04 + specification-prediction notes (relocated 2026-08-30)

**Counted output on this tree, 2026-08-04:**

```
scanned 400 files under src/, found 13 criteria
NO_RED: 0   TAUTOLOGICAL: 0   READS_SYSTEM: 13   EXEMPT: 0
```

**It asserts from a green start, which is normally the shape that makes a gate untrustworthy, and the
reason it is acceptable here is worth keeping.** A gate that has only ever been seen green is
indistinguishable from one that cannot see. This one has been proven able to fire:
`tests/test_red_reachability.py` reconstructs the **pre-fix** `consciousness_probe` and checks every
bucket lights up. That is the general rule this entry contributes — *if a gate lands green, ship the
reconstruction of the defect it was built for* — and it is the same discipline `citation_guard`'s
exemption-rot assert uses.

**The specification below predicted "at least two failures on the first run — BINDING and RE-ENTRY".
Both were deleted before the gate landed**, so the prediction was right about the defects and wrong
about the ordering: the fix arrived first and the gate ratchets it. Recorded because the gate-first
bar exists to stop a rule being adopted on an uncounted promise, and the honest report here is that
the count was taken *after* the cleanup rather than before it.

*(The specification's claim that `substrate/brain_telemetry.py:637` "asserts four of those tags
survive" did not resolve on re-check — `:637` is a bare `break`. `PROVENANCE` itself is at `:124`.
The instrument half of the claim stands; the line does not.)*

### e-prop invariant: the 2026-08-15 transport-gate widening (weight-name predicate, INDEX-SWAP rule, discarded drafts) (relocated 2026-08-30)

  **WIDENED 2026-08-15, at the owner's instruction, and limit (a) below is now PARTLY closed.** Two
  defects were found in the gate itself, both by asking what it could not see rather than by reading
  it. **(1) Its weight-name predicate missed the names it most needed:** `Wrec` / `Win` / `Wout` --
  `reasoning/eprop.py`'s actual weights -- were not weight-ish at all, and **`W2` was rejected as a
  CONSTANT** because `"W2".isupper()` is True, `W2` being the exact name in `mlp_q`'s retired
  transport that this row cites as the gate's miss. The constant exclusion is NARROWED rather than
  removed (a real constant is caps-and-underscores with two or more trailing caps, so `W_MAX` still
  excludes and `W2` no longer does); **re-counted before adopting, the shipped rule scores GOVERNED
  0 / UNGOVERNED 1 under both predicates, so widening cost zero findings.** **(2) A SECOND RULE,
  INDEX-SWAP, now catches the loop spelling:** an explicit transpose is an ARRAY-LIBRARY idiom, and
  `reasoning/eprop.py` hand-rolls reverse-mode BPTT in pure `math`/`random` loops with no numpy and
  no `.T` anywhere, so that file was invisible to the walk **entirely** -- not a finding, not a
  census entry. The rule keys on the forward order being the one the module ASSIGNS to, so a read in
  the reversed order is a transpose, and a module that only COMPARES both orders (`neural_sampling`
  raises unless `W[i][j] == W[j][i]`, ENFORCING symmetric coupling) establishes no forward order and
  drops out by construction rather than by an allowlist. **Counted before adopting: 1 finding**,
  `reasoning/eprop.py`'s `Wrec`, which is NOT a violation -- `Net.grad_bptt` is the independent
  reverse-mode gradient this invariant's own cosine clause REQUIRES -- and is ungoverned, so it
  reports. The governed half asserts, so a module hand-rolling backprop in loops no longer passes
  silently on the day it is promoted. **Two earlier drafts were discarded at this bar and the
  discards are the useful part:** "both index orders appear anywhere" scored 2 with the symmetry
  validator as a false positive AND missed its own motivating case; bolting the new rule onto the
  existing terminal test ("the transport must reach a weight update in the SAME function") was
  rejected on design, because `grad_bptt` RETURNS gradients and `apply()` performs the update.

### e-prop invariant: the full per-site adjudication narratives (verdict register stays in POLICIES.md) (relocated 2026-08-30)

  - ~~`cerebellum/jepa_world_model.py` is a **real violation of the locality principle, on the live
    wake tick.**~~ **CLOSED 2026-08-04.** The transport half is gone: `_feedback_gz()` is now the ONE
    backward definition for both the encoder path and the rollout path (P2 — one mechanism, not two),
    and it returns `list(gp)` — the identity/skip half alone, which is a broadcast error signal of
    exactly the shape e-prop asks for. **And the remedy THIS POLICY PRESCRIBED was built first and
    REFUTED, which is the part worth keeping.** Feedback alignment (a fixed random `Bz`) was
    benchmarked over 1200 train / 350 held-out transitions, 5 seeds: held-out invariance MSE ratio to
    the transport incumbent was **0.657 for FA against 0.654 for simply dropping the recurrent term**,
    FA won on only 2 of 5 seeds, and six independent `Bz` draws all landed in 0.726–0.739 — so the
    random matrix contributed nothing measurable and *the entire ~34% gain is the removal of the
    transport term, not feedback alignment.* Shipping `Bz` would have bought a new matrix, a new seed
    stream and a new constant with a null (P2: a merge must REDUCE; P8 guard (d): report the null).
    **Recorded rather than shipped — a policy prescribing a fix does not make the fix correct, and
    the measurement outranks the prescription.** Cosine(FA, transport) was 0.971, far above the 0.5
    e-prop bar, which is what makes the null trustworthy rather than an artifact of a broken control.
  - `hippocampus/ng_snn.py::fit` is **not** a violation. Greedy matching pursuit with an
    analytical beta per block is a direct solve: no gradient is propagated through a network and no
    weight is transported. It IS non-local, and the policy should say so distinctly rather than
    lumping it in — "non-local" and "backprop" are different claims, and only the second is what this
    rule forbids. Disposition: keep, and stop citing it as a pending violation.
  - **OPEN QUESTION, added 2026-08-13, NOT adjudicated: the `dh` step inside `basal_ganglia/mlp_q.py::MlpQ.update`** (cited by SYMBOL, per this file's own rule below — the line cite `:190` written here on 2026-08-13 was broken THE SAME DAY by a +6-line docstring edit in that very file, which moved the statement to `:196`). `dh = [err *
    w2i[j] ...]` reads the FORWARD read-out weights `W2[a]` transposed to carry error backward into
    the shared hidden layer. The module **self-declares this** in its own module docstring (the "OPEN e-PROP VIOLATION, NAMED HERE BECAUSE THIS MODULE PERFORMS IT" paragraph) as an open e-prop violation
    and names this policy — yet `mlp_q` appeared **nowhere in this file** until this entry added it, so the site had been
    disclosed in the code and invisible in the policy record since it was written. It is **not on the
    live tick** (`AutonomousArm` defaults `mlp_q=False`) but is one constructor argument from it, and
    it executes on every gate run via self-check 19, so it is latent rather than absent. **The
    measurements already exist** (the module docstring's "MEASURED, so the fix is a decision rather than a guess" block). `main()` RE-DERIVES the transport and LatentQ columns and the cosine sweep on every gate run; the FEEDBACK-ALIGNMENT ACCURACY row is NOT re-derived -- that twin is a SCRATCH benchmark the module deliberately does not ship, and says so in the same block: feedback
    alignment clears both of the module's shipped gates on all 8 draws and *gains* 1.0 point on the
    linear rule, but its hidden-gradient cosine sweeps to **[-0.4366, +0.1844] with |cos| > 0.5 in 0
    of 8** — so swapping trades one failed clause of this invariant for another, which is not a fix.
    (**Interval corrected 2026-08-18**; census-baseline.md. The PROPERTY — 0 of 8 clear — is what
    the argument rests on; the test asserts it, pins no interval.) The
    two honest resolutions the module states: **(a)** accept FA and amend what "cosine > 0.5" means
    for a non-gradient *alignment* rule, or **(b)** retire the head under P2 for the linear `LatentQ`,
    which has no hidden layer and therefore no backward pass at all — at a measured cost of the
    non-linear capability (**0.467 vs 0.993 on XOR**), which P2's honesty guard requires be ported
    first. Recorded here so the decision is made against the measurements rather than rediscovered.

    **RESOLVED 2026-08-14 (specs/015-resolve-weight-transport) — and by NEITHER of those two
    routes.** The transport is RETIRED. `dh` now reads a separate feedback matrix `B` that receives
    the SAME increment as the forward read-out (`B[a][j] += lr*err*h[j]`), so `B` TRACKS `W2`
    without ever reading it — `err` and `h[j]` are already local to that synapse. This is
    **Kolen-Pollack** (Kolen & Pollack 1994), NOT the Spiking Forward DFA that prompted the lane:
    SFDFA estimates its feedback by propagating directional derivatives as SPIKE GRADES, and `MlpQ`
    is a rate MLP with no membrane, threshold, spike or time index, so SFDFA's actual contribution
    has nothing here to attach to. What transferred is its ESSENTIAL property — feedback ESTIMATED
    and CONVERGING rather than transported or frozen. Naming the shipped rule "SFDFA" would have
    been a citation defect.

    **The transferable rule is in `docs/POLICIES.md` (e-prop verdict register); the full
    route-rejection narrative and the 015 measurements are earlier in THIS file (specs/022 T003
    relocation)** *(sentence rewritten post-relocation: it pointed outward at this file)*: an
    instrument no alignment rule could ever pass is not evidence against alignment rules (the FA
    cosine reduced to `cos(W2[a], B[a])`, near zero BY CONSTRUCTION), and route (b) was never
    needed. Headline numbers only: converging `B` 8/8 vs frozen 0/8, permutation null 0.0000, no
    capability lost, no new constant.

    **The invariant was NOT amended TO RETIRE THIS TRANSPORT, and did not need to be** -- route
    (a) was unnecessary because Kolen-Pollack clears the bar. **Scoped 2026-08-18:** the SCOPE
    paragraph DID later amend this clause, on the different question of which rules owe a cosine
    at all; it classifies this arm `RULE_JUDGED` — the gate's coherence-keyed token, meaning THE
    NULL SURVIVED, not a finding that the arm failed. The 2026-08-14 non-discrimination argument
    is unamended and still carries route (a)'s rejection.
    `main()`'s check (5) separates three ways on every gate
    run (transport identity / shipped learned `B` / frozen control) — re-run it for the values. **One prediction was wrong and is recorded rather than smoothed**: the spec
    predicted `B` would "rise monotonically from ~0"; it is at 0.92 after ONE epoch and drifts
    slightly DOWN. Alignment is established almost immediately and then tracked, not developed over
    training — the direction was right, the mechanism model was not.
  - **OPEN QUESTION, added 2026-08-13, NOT adjudicated, and the LIVE one: `cortex/predictive_coding.py:200-201`.**
    `g1 = self.w1.T @ s_e0` / `g2 = self.w2.T @ s_e1` — the forward weight array, transposed, carrying
    error backward. Unlike `mlp_q` this **runs on every wake tick** (`cortex/brain.py:235` constructs
    the brain with `predictive_coding=True`), which is the exact property that made the jepa site a
    violation. **CORRECTED 2026-08-14: it is NOT undisclosed in its own module, and this entry said it was.** `cortex/predictive_coding.py` writes the transpose out in its own prose FOUR times -- the module docstring (`r1 += lr * ( W1^T @ s_e0 - s_e1 )`), `infer_step`'s docstring, the inline comments on the two cited lines themselves, and again in the batch path. What was undisclosed is the POLICY RECORD, not the code -- which is precisely what the finding cited below concludes: its own heading reads "TITLE OVERSTATES CONCEALMENT". This entry cited that finding approvingly while asserting the framing it retracts. Half-quoting a source to support the claim it corrects is the same defect the P0 row above records against `findings.md:370`.
    **The question is real in both directions and must not be settled by
    analogy.** For: exact weight symmetry by construction — the same array, transposed — is the
    objection this invariant exists to name. Against: in Rao-Ballard, `W1` is the GENERATIVE weight
    (`x_hat = W1 @ r1`), so `W1^T @ s_e0` is one matrix in its two structural roles rather than a
    separate forward pathway being copied, which is the standard locality argument for predictive
    coding. `specs/010-policy-compliance-audit/findings.md:826` already reached this site, called it
    *"an undisclosed VIOLATION SITE, not concealed code"*, and observed that this verdict "covers
    exactly two sites" with `predictive_coding.py` appearing in neither — **that finding was never
    carried into this file, which is how a live site stayed unadjudicated for nine days.**

    **ADJUDICATED 2026-08-14, at the owner's instruction: NOT a violation of this invariant.** The
    verdict rests on a mechanism read out of the module, not on the Rao-Ballard analogy, and not on
    the Predictive E-prop preprint — which addresses BPTT and has **0** full-text hits for weight
    transport / weight symmetry / feedback alignment, so its silence is silence.

    **THE DISTINGUISHING TEST IS WHAT THE TRANSPOSE FEEDS.** In both sites this verdict ruled
    violations, a transposed forward weight fed a WEIGHT update — jepa's `Wz^T gp` entered the
    predictor's gradient, and `mlp_q`'s `W2^T` produced `dh`, which updated `W1`. That is credit
    assignment to weights through a transported weight, which is exactly what "e-prop, not backprop"
    forbids. `predictive_coding` does not do it:

    - `g1 = self.w1.T @ s_e0` feeds `self.r1 = self.r1 + lr * (g1 - s_e1)` — an **ACTIVATION**.
      It is INFERENCE, settling a transient latent state, not credit assignment.
    - the **weight** update is `self.w1 += lr_w * np.outer(self.cum_e0 / self.n_steps, self.r1)` —
      a **local Hebbian outer product** of a local error rate and a local cause. **No transposed
      weight appears in the learning rule at all.**

    The invariant as this file states it governs LEARNING RULES — "local eligibility trace ×
    broadcast learning signal, verified cosine > 0.5 against an independent BPTT gradient". There is
    no gradient propagated through a network here and no weight transported into an update, so there
    is nothing for the cosine clause to be measured against. Same disposition as `ng_snn::fit`, and
    for the same structural reason.

    **WHAT THIS VERDICT DOES NOT SAY, stated so it is not read as broader than it is.** `self.w1.T`
    IS exact weight symmetry, on every wake tick, and a physical synapse is unidirectional — so
    realising that transpose still requires a symmetric backward pathway. That is a real
    biological-plausibility question. **It is simply not the question this invariant asks**, and
    widening an invariant is an owner decision, not something an adjudication may do on its own
    authority. Recorded here as OPEN SCOPE, distinct from the closed violation question: if the
    invariant is ever extended from learning rules to inference pathways, this site is the first
    thing it catches.

  - **ADJUDICATED 2026-08-15, at the owner's instruction: `reasoning/gf.py::MLP.train` performs
    weight transport, and is NOT a violation of this invariant — it is a CONTROL.** For where this finding surfaces in the
    gate's output, RE-RUN IT — restated tallies rotted here twice.

    **The transport is real, not arguable.** `gh = (g @ self.w2.T) * (1 - h**2)` carries error
    through the transposed forward read-out into `gw1 = xb.T @ gh`, then `self.w1 -= lr * gw1`, so
    **the transpose feeds a WEIGHT update** — the jepa/`mlp_q` shape, the opposite of
    `predictive_coding`, under the distinguishing test above. Hand-written reverse-mode with no
    autograd, which is why `tools/promotion_sweep.py`'s marker grep cannot see it. **Scope
    MEASURED, so this verdict does not enumerate where it appears to survey:** `gf.py` contains
    exactly THREE transposes, all inside `MLP.train`; `h.T` and `xb.T` transpose an ACTIVATION and
    an INPUT, and exactly ONE (`self.w2.T`) transposes a weight. There is no second site in the file.

    **The verdict's kernel is in `docs/POLICIES.md` (e-prop verdict register); the full reasoning
    is earlier in THIS file (specs/022 T003 relocation)** *(sentence rewritten post-relocation: it
    pointed outward at this file)*: not a violation on the invariant's own scope —
    code whose JOB is to be measured against is not one of the brain's learning rules; `MLP` is a
    Gc BASELINE (not the gradient instrument, which is `reasoning/eprop.py::Net.grad_bptt`) that
    **exists to LOSE** (`mdl_ok < ind_ok`), and porting it to e-prop would VOID the experiment —
    P8's contaminated control. Zero references outside `gf.py`, on no tick. Its missing
    self-declaration was corrected 2026-08-15 (P8 clause 4).

    **THE EXEMPTION, GRANTED 2026-08-15, and what was built so that it cannot rot.**
    `eprop_transport_gate` asserts on the GOVERNED tier, so promoting `gf.py` — backlog row 40 calls
    it the "pure leaf" that moves FIRST — would have converted this report into a governed RED.
    The exemption carries the verdict above as its reason and — Phase 7 having retired the
    gate's `EXEMPT` dict — lives in `reasoning/gf.py::GRADIENT_CLAIM["MLP.train"]`, read
    through the registry, so the ARC promotion cluster is unblocked. **Disclosed: this DECENTRALIZES the channel**
    — a module can self-grant by naming the gate in its declaration reason. `stale_exemptions`
    catches a grant blessing nothing; one blessing a REAL finding, only review catches. The owner
    grant remains the rule, mechanized as the marker (closure lane: an exemption-count
    ratchet, unbuilt). **Adding it NARROWED what a mechanized
    gate asserts, which is why it required the owner and not an adjudication.**

    Staleness is asserted BOTH ways, matched against ALL findings — the grant PRECEDES the
    promotion it unblocks, so a governed-only match would read STALE the day it was correctly
    written — and both directions were PROVEN by making them fire: a planted exemption matching
    nothing reds the gate (re-run it for the exact wording; a verbatim quote rotted here once). The bar for a future entry is stated at the site: a CONTROL is the instrument
    this invariant's own verification clause requires; **a faculty is not, however far off the tick
    it sits** — `mlp_q` was one constructor argument from the live tick and its transport was
    RETIRED, not exempted.

  *(Line citations in this bullet — `jepa_world_model.py:275-278`/`:277` and `ng_snn.py:137` — were
  dropped 2026-08-04 after all three failed to resolve to what they claimed. `ng_snn.py:137` is
  `mu = S.mean(dim=0)`. Worse, the jepa file was edited BETWEEN two reads in the same audit session,
  and its line numbers moved by one under me: a line citation into an actively-developed file is
  stale on a timescale of minutes, not releases.)*

### Cross-cutting: the third proposal (untracked-instrument citations) -- counted broad reject, adopted strict scope (relocated 2026-08-30)

- **A third proposal died at this bar, 2026-07-30: "a tracked file may not cite an untracked `.py`
  instrument."** It had the best motivation of any proposal here — three real incidents, two of them in
  one session: `POLICIES.md` citing a `scratchpad/smoke_drive_charge.py` that never existed, the same
  dead citation in `dormancy_gate.py`, and `DEPTH_CEILINGS.md`'s AR appendix citing nine gitignored
  `scratchpad/ar_lm_*.py` scripts as its evidence *hours after* the first two were corrected. It was
  built anyway, per this rule, and counted: **474 violations across 126 files, 195 in `CHANGELOG.md`
  alone**, from 1,431 citations. Allowlist-shaped, eight times worse than the 58-failure proposal above,
  so it is **not adopted**. The count also explained itself: a changelog citing a since-retired file is
  correct, because the entry describes a tree that existed. The instrument is kept as a report at
  `tools/citation_guard.py`, and the surviving narrower hypothesis is recorded there — the real defect is
  an untracked script presented as a *runnable reproduction path*, not a filename in prose. Building the
  naive version first is what distinguished those two, and it cost one file instead of a policy.
  **The narrow version was then built and counted too, and it SURVIVED at 21 violations across 5 files**
  (`tools/citation_guard.py`, STRICT scope — a `.py` cited inside a shell-fenced block or under a
  Reproduce/Usage heading). 21 was a work list, not an allowlist, so the rule was adoptable in that
  scope. What it found was worse than the incident that motivated it: **`docs/RUNNING.md`, the project's
  how-to-run page, had five commands pointing at files absent from a clone.**
  **FINAL STATE, same day: broad REJECTED at 475 across 127 files (report only, still non-asserting);
  strict ADOPTED and ASSERTING at 0 unexempted violations with exactly 1 exemption.** The 21 were
  cleared, not exempted, and the dispositions differed as predicted:
  - **10 were stale paths.** `research/` became `src/snn_data_lake/reasoning/` in v3.78 (`627b2e2b`), and
    `RUNNING.md` (×5), `JEPA.md` (×3), `PERFORMANCE.md` (×1) plus `HARDWARE_SETUP.md`'s sketch path were
    never updated — the files had *moved*, not gone. Corrected to `python -m snn_data_lake.reasoning.<mod>`.
    One was a genuine retirement: `spike_knowledge_graph.py` was merged into `spike_kg.py` (`17067032`).
  - **8 were promoted into `tools/`** from gitignored `scratchpad/` — real evidence for real published
    numbers, now tracked, with inputs taken as explicit paths and a clean SKIP when a dataset is absent.
    Each was run; `lang_scale.py`, `lang_diag.py` and `lang_probe.py` reproduce their doc tables exactly
    (53.4→53.1 decay ablation, 0/1675 fold collisions, 0.65% over 117,859 contexts, 2,992 params).
  - **1 was deleted, not promoted:** `ar_lm_untrained_control.py` no longer exists on disk, and the
    tracked bench builds the untrained twin anyway. Its command is gone and the doc now says the 49.5% is
    a recorded result, not reproducible from that file.
  - **1 exemption, `docs/HARDWARE_SETUP.md:144`.** The cited file IS tracked, at
    `integrations/robot arm/sensor_reading.py`; the SPACE in the directory name truncates the guard's
    path token, so it cannot resolve. Fixing that in the resolver was rejected because it would equally
    resolve the stale `robot arm/...` form the line carried until it was corrected — the tokenizer fix
    would hide a live defect class, so the false positive is exempted where the reason is readable.
    `EXEMPT` follows `dormancy_gate.EXEMPT`'s pattern and, because line-keyed entries rot, the gate also
    asserts every entry still matches a real violation: an exemption that blesses nothing fails loudly
    instead of reading as coverage. Both asserts were tested by making them fire.
  Registered as a step in `.github/workflows/ci.yml`'s lint job, not in `run_selfchecks.py`: the latter
  only executes `python -m snn_data_lake.<mod>` package modules, so qualifying would mean moving it into
  the package and churning the canonical 277 count quoted in five documents — to gate repo hygiene, which
  is not a brain-faculty self-check. Recorded here because "the broad rule died at 475 and the narrow one
  now holds the line at 0+1" is the result, and the only reason either number exists is that this bar
  demands counting before adopting.

### Cross-cutting: the 2026-08-04 docs-wide line-citation audit behind the cite-the-symbol rule (relocated 2026-08-30)

- **A `file.py:NNN` citation is a claim, and it is the least durable claim this project makes.**
  Added 2026-08-04 after a docs-wide audit resolved **every** `path.py:LINE` citation in `docs/`,
  `specs/`, `README.md`, `SECURITY.md` and `AGENTS.md` against the tree. Measured: **2,714 resolved,
  90 pointing at a file that does not exist, 27 pointing past end-of-file.** Worse than the miss rate
  is *where* the misses were — in this file, in the P0 avenue table, in the P1 safety-envelope
  paragraph (**three of its four cites wrong**), in the e-prop invariant, and in `SECURITY.md`'s
  description of the arm. The P0 growth row's cites, corrected the day before, had already drifted
  again.

  **`tools/citation_guard.py` does not catch this and cannot be blamed for it** — it resolves the
  FILE, deliberately and for a counted reason, and every finding above has a file that exists. Two
  further failure modes it also cannot see, both found in the same pass: a citation whose *file* is
  ambiguous (`executive_control.py` names two different modules, 411 and 645 lines), and a documented
  command whose file is tracked and which still exits 1 (`docs/RUNNING.md` had six —
  `tools/doc_command_guard.py` now gates that one, at 0 unexempted after 19 were cleared).


### P3 honesty guard: the four-faculty dormancy acquittal (relocated 2026-08-30; found dropped by the split review panel)

(History: a coverage-audit once flagged PFC / amygdala / cortical-column / synfire as dormant;
every one was REFUTED on runtime A/B. Static grep proves import-absence, not behavioural
inertness -- mark unproven cases UNCERTAIN, not dormant.)
