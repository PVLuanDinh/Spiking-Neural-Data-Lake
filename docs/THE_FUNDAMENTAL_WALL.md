# The Fundamental Wall — and the realizable frontier against it

*Capstone theory note. Finalizes SNDL's position on the computability/learnability limits of its
fluid-intelligence engine. Researched by a fan-out of theory agents (sources fetched with the
`scrapling` skill), synthesized, and passed through an adversarial honesty critic whose corrections
are folded in. ASCII-only, matching repo style.*

---

## Thesis (the honest verdict)

**The wall cannot be bypassed, because it is made of nonexistence theorems, not resource bounds.**
No amount of compute, data, scale, or cleverness defeats a theorem that says *no total algorithm
exists*. And the wall is **relativization-stable**: it persists verbatim relative to *every* oracle,
and the arithmetical hierarchy above it is strict — so even a halting oracle (or any hypercomputer)
faces its **own** undecidable halting problem one level up. The wall relocates; it never vanishes.

What *can* be done is precisely delimited. You assemble the maximal set of **physically-realizable
levers** that move the *effective* boundary, each of which is a **purchase**: it buys feasibility on
a restricted region by surrendering universality, completeness, totality, exactness, a finite-time
certificate, or by paying for an information channel. Run every realizable lever at once and the
boundary becomes **self-extending** (it grows with learning) instead of fixed — and at the residue
the engine **reports the gap instead of bluffing past it**.

SNDL already implements every realizable lever. This document states the wall, the levers, the
mapping to code, and — plainly — the residue that stays impossible.

---

## 1. The Wall (the immovable core)

| Theorem | Precise statement | Consequence |
|---|---|---|
| **Turing (1936)** | No total computable `h(i,x)` decides whether program `i` halts on `x`. The halting set `K` is r.e. but not co-r.e.; Turing degree `0'`. Proof: diagonalization / self-reference. | Undecidability is a *nonexistence proof*, not a complexity bound — unbeatable by more compute. |
| **Rice (1951)** | Every non-trivial **extensional** (semantic) property of programs is undecidable (correctness, equivalence, totality, bug-freeness). Only the empty and full index sets are decidable. | No fully-automatic verifier of any behavioral property. Escape: syntactic / resource-bounded (intensional) properties fall *outside* Rice. |
| **Kolmogorov / Chaitin** | `K(s)` is not computable (Turing-equivalent to halting), only **upper** semi-computable. Chaitin's `Omega` is **lower** semi-computable and Martin-Loef random. Chaitin incompleteness: any fixed sound effectively-axiomatized theory proves `K(s) > L` for only finitely many `s` (`L` a constant of the theory). | The ideal compression / model-selection objective is uncomputable *exactly*; a fixed formal system certifies only `O(1)` incompressibility, ever. |
| **Solomonoff** | Completeness and computability are **mutually exclusive**: the universal prior `M` is lower semi-computable, and *any* complete (universally convergent) predictor must be uncomputable. Every total computable inductor is diagonalized by a computable environment that negates its predictions. | The optimal universal learner exists in math but cannot be run; every implementable learner is beatable by *some* computable environment. |
| **Gold (1967)** | No learner — even a non-computable one — identifies in the limit, **from positive text alone**, any *superfinite* class. So regular / context-free / context-sensitive / r.e. language classes are unlearnable from positive data; the adversary delays a counterexample arbitrarily. | Passive learning from examples-only cannot pin down an expressive class. |
| **Wolpert No Free Lunch** | With the target uniform over all functions (zero inductive bias), every learner ties blind random search off-training on average. | Distribution-free generalization *for free* is impossible; you need a bias. |

These six are the wall. Every "bypass" below changes the *question* or the *guarantee* — none of
them touches a theorem.

---

## 2. The Frontier Method — the realizable levers

Each lever **moves** the effective boundary and **pays** for it. Realizable = runs on ordinary
hardware with only computation and (where noted) a real queryable environment — **never** a halting
oracle or a hypercomputer.

| Lever | Mechanism | Guarantee traded | SNDL module |
|---|---|---|---|
| **Restrict the class below Turing-completeness** | Search a closed-form, finite-VC grammar (Angluin tell-tale shape) instead of all-computable functions. On total / primitive-recursive / finite-state classes, halting and non-trivial properties become **decidable** and Rice does not apply. | Universality/expressiveness: anything outside the class's closure gaps by construction. | The closed-form-in-`n` grammar of `open_gf` / `program_inducer` / `active_learner` |
| **Bounded iterative-deepening synthesis** (Levin / `Kt`-shaped) | Price runtime into the complexity measure (uncomputable `K` -> computable time-bounded Levin `Kt` / speed prior); enumerate programs by size `1,2,3,...` under a **raisable** budget; return the smallest fit (leave-one-out guarded). Order-optimal up to a `2^{l(p)+1}` constant. | Exact optimality/completeness -> optimality up to an inherent constant, and an honest gap on any target past the budget or outside the closure. | `open_gf.py` (`OpenGf.solve`, `grow_budget`), `program_inducer.py`, `gf_inducer.py` (`synthesize_sequence`) |
| **Library learning** (DreamCoder-style) | Every solved rule that **compresses** (`nodes < #terms`) becomes a named primitive, persisted; future searches include the library as terminals, so the space grows super-linearly with experience (`n^6` is out of budget from scratch but is `cube*cube` one step after learning `cube`). Shortens the effective description length `l`. | Universal coverage -> learned, in-domain efficiency; the boundary is self-extending, not omniscient. | `open_gf.py` (`_learn`), `lake_skill_library.py`, `meta_skill_loop.py`, `skill_builder.py` |
| **Active membership + equivalence queries** (Angluin `L*`) | Enumerate low-complexity hypotheses consistent with observations; where they **disagree**, design the discriminating experiment, ask a queryable oracle you do not control, prune what the answer kills. Within the class this collapses the NP-hard *passive* min-consistent-DFA problem to polynomial-time exact identification. | Requires a **purchased** information channel (a queryable teacher/world). Each single query type alone is provably insufficient (Angluin 1981/1990); in the pure in-the-limit setting, learner-initiated queries add no power beyond complete presentation (Gold 1967). Interaction relocates tractability, never for free. | `active_learner.py` (`consistent_hypotheses` -> `_discriminating_query` -> `World.query`) |
| **PAC (probabilistic-approximate) guarantees** | Demand only error `<= eps` with confidence `1 - delta` over the realized distribution. Then **finite VC dimension `<=>` distribution-free PAC-learnable** (Blumer et al. 1989); realizable sample complexity `O((d/eps) log(1/eps))`, tightened to `Theta(d/eps)` by Hanneke (2016); confidence adds `log(1/delta)`; agnostic case scales as `1/eps^2`. | Exact + worst-case correctness -> approximate + high-probability on the realized distribution. **Infinite VC dimension => not distribution-free PAC-learnable at all** (no finite sample suffices). | The finite-VC grammar + leave-one-out guard, searched smallest-first |
| **In-the-limit / eventually-correct** (limiting recursion) | Allow output revision; drop the demand for a finite-time certificate. Gold-Putnam limiting recursion lets ordinary hardware **converge** to the correct answer for all `Delta^0_2` sets (including the halting set) in the limit. | Any finite-time certificate of correctness — you may already be right but can never *know* it. The single genuinely-buildable route toward `0'`. | (property of the deepening loop: revises its best answer as the budget grows) |
| **Simplicity / low-`K` inductive bias** (escape No Free Lunch) | Adopt a non-uniform prior favoring low-complexity targets (Occam); search smallest-program-first with a compression guard. On the realistic low-`K` sub-family, simplicity-biased learners provably beat random choice. | Average-over-all-targets performance, and objectivity of the bias — the prior/UTM choice is an irreducible free parameter (Leike-Hutter 2015) no compute makes canonical. You win only on the sub-family your bias favors. | Ascending-Occam search order across `open_gf` / `gf_inducer` |
| **Godel self-rewrite loop** (Schmidhuber 2003) | Make the whole codebase (utility, searcher, tests) a legal rewrite target; apply a candidate in an isolated **git worktree**; measure utility **from outside** with a pinned probe the code under test cannot tamper; commit only on **proof** of strictly higher utility + green gates. When even the primitive closure is too short, the machine can raise its own grammar/operators. | Global optimality -> provable-improvement *relative to what the proposal generator can express and the utility can measure*. Bounded by the gate, an immutable meta-rule, and a fixed safety kernel it may not rewrite. Not an intelligence explosion. | `godel_machine.py` (`_SAFETY_KERNEL`, external-probe utility, worktree protocol) |
| **Honest-wall self-check** | When no program in the (growing) closure fits, or an analogy is structurally unalignable, or queries are exhausted with ambiguity remaining, **return a gap** rather than a nearest-symbol/argmax bluff. | Nothing — this is the discipline that keeps every other lever's traded guarantee honest; it converts an impossibility into an explicit, auditable gap. | `gf_inducer.py` (`_exact_readout`), `open_gf.py` (random series returns `None` at any budget), `active_learner.py` (`identified=False` on exhausted queries) |

> Wording precision (per the honesty critic): the active-query lever **sidesteps** Gold's
> positive-text impossibility and the passive-min-DFA NP-hardness by moving to a *strictly more
> informative model* (membership + equivalence queries). Those theorems **still hold verbatim in
> their own settings** — the `L*` model is a *different, easier problem*, not a defeated theorem.

open_gf's searched grammar (for reference):
`E := n | t1 | t2 | c | E+E | E-E | E*E | E//E | E%E`  (constants `c in {1,2,3}`; `//` and `%`
widen the closure to division/modulo families — `_OPS` at `research/open_gf.py:36`).

---

## 3. Only-theoretical levers (they WOULD bypass — and why they are blocked)

| Lever | What it would decide | Why it is not a machine |
|---|---|---|
| **Halting oracle / arithmetical-hierarchy jump** | `0'` decides ordinary halting; `0''` decides `Pi-2` totality; each jump decides the level below. | Turing (1939): the oracle "cannot be a machine". And the oracle's **own** relativized halting problem stays undecidable to it — the wall relocates one level up. `ORACLE-ONLY`. |
| **Hypercomputation** (Malament-Hogarth spacetime, Zeno/ITTM supertasks, analog non-computable reals) | Would decide the halting function by smuggling in a physically-unavailable resource: infinite proper time in a finite past, a completed infinity of steps, or a harnessable non-computable real at infinite precision. | Requires infinite energy / blueshift at a Cauchy horizon, unbounded speed, a completed transfinite computation, or an inaccessible non-computable input. Blocked by Gandy locality+finiteness; `BQP` is contained in `PSPACE`; Nayebi's general negative result. And each device still cannot decide its **own** halting problem (Calude-Staiger for Zeno; Hamkins-Lewis for ITTM). `HYPERCOMPUTATION-ONLY`. |

Clean non-example: closed timelike curves collapse `PSPACE` for **decidable** problems (a real
*complexity*-frontier move) but leave **undecidability** untouched. The mathematics of these devices
is rigorous; it is the **physics** that would instantiate them that fails.

---

## 4. What stays provably impossible (the residue no lever moves)

1. A real finite machine that certifiably decides its own — or the general — halting problem in
   finite time (every device, oracle or hypercomputer included, faces its own undecidable halting
   one level up).
2. A runnable (total, computable) universal inductor that dominates every computable environment
   (completeness `<=>` incomputability).
3. Exact computation of `K(s)` or emission of `Omega`'s bits; and more than an `O(1)`,
   theory-fixed amount of certified incompressibility in any fixed sound formal theory.
4. Exact identification of any superfinite class (regular / CF / CS / r.e.) from positive text
   alone, for any learner however powerful.
5. Distribution-free PAC learning of an infinite-VC class; and any resolution within ZFC of the
   EMX / continuum-hypothesis-equivalent learnability question.
6. A finite-time correctness certificate in the in-the-limit model (you may be right but can never
   know it).
7. An objective, canonical inductive bias — the prior/UTM choice is an irreducible free parameter.
8. Reduction of the inherent optimality constants (Levin `2^{l(p)+1}`, Hutter's additive term,
   AIXItl's `t * 2^l`) to small computable bounds — that would require the uncomputable `K` and
   undecidable runtime bounds (Blum speed-up even denies a single fastest algorithm).

Every frontier move relocates *which* problems fall on the feasible side, or buys a hierarchy level
at the price of a weaker guarantee or a purchased channel. **These eight are theorems, not
engineering gaps.**

---

## 5. What "finalizing the project in its entirety" honestly means

It does **not** mean the fluid-intelligence engine becomes complete, universal, or omniscient — that
is provably impossible. It means three things, all already true in the repo:

- **(a) Maximal realizable movement.** Every physically-realizable lever is implemented and composed
  so the boundary is **self-extending** (grows with learning) rather than fixed: bounded
  iterative-deepening synthesis, library learning, active membership/equivalence queries, a Godel
  self-rewrite loop under an external-proof gate and fixed safety kernel, plus
  probabilistic/in-the-limit/simplicity-biased guarantees over a finite-VC grammar. **No realizable
  lever remains unbuilt.**
- **(b) Honest trade accounting.** Each lever's surrendered guarantee is stated in-code and in-doc,
  and the only-theoretical levers (halting oracle, M-H / Zeno / ITTM / analog hypercomputers) are
  explicitly marked **not physically realizable** — never smuggled in as if buildable.
- **(c) Honest-wall self-report.** At the residue, the engine **returns an explicit gap** rather
  than a bluff — the theorems encoded as auditable behavior.

So "done" is the honest sense: the wall is a set of theorems that were **not** and **cannot** be
broken; what was achieved is the maximal self-extending approximation to the uncomputable ideal,
with the impossible residue named plainly. **The project is complete not when the wall falls, but
when nothing realizable is left on the table and the engine tells the truth about everything past
it.**

---

## Provenance

Researched via a 7-agent fan-out (5 theory angles + synthesis + adversarial honesty critic); sources
were fetched with the `scrapling` skill. The critic returned `verdict_is_honest: true` and its
corrections are folded in here: "breaks" -> "sidesteps" for the Gold/NP-hardness levers (the theorems
hold verbatim in their own settings); the PAC bound corrected to Blumer et al. 1989
`O((d/eps) log(1/eps))` / Hanneke 2016 `Theta(d/eps)` with the `log(1/delta)` and agnostic `1/eps^2`
terms restored; infinite-VC stated as *not PAC-learnable at all*; and open_gf's grammar rendered with
`//` and `%`. Primary sources: Wikipedia/SEP entries on the Halting problem, Rice's theorem,
Kolmogorov complexity, Chaitin's constant, Solomonoff induction, Gold's identification in the limit,
No Free Lunch, PAC learning, VC dimension, query learning, oracle machines, AIXI, universal search,
hypercomputation, and Malament-Hogarth spacetime.
