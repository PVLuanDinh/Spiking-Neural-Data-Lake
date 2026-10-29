# A1 / T105 -- annealing for lake access: pre-registration (DRAFT)

> **Status: DRAFT, NOT FROZEN, NOTHING RUN.** Written 2026-10-29 by lane `a1-prereg` (docs builder, thread A1-PREREG)
> against base `c4c10b28` on `claude/a1-prereg-1029`. No code, test, pilot, simulation of the design or brain run
> produced anything below: every number is either copied from a cited file or marked PROJECTION or OWED. It freezes only
> after the "Owed before freeze" list (section 9) is closed and the review panel and the policy-enforcer AGREE.
>
> - **Row:** tasks.md T105 [A1] (`tasks.md:234`), FR-040 (`spec.md:166`), SC-018 (`spec.md:209`). T105's wording is the
>   checklist; where this draft and that wording differ, the row binds.
> - **Order:** R1/T102 -> A1/T105 -> A2/T106 -> A3/T107 (`tasks.md:274`; PLAN-FINAL `:1114`). T102 is still `[ ]`
>   (`tasks.md:231`): see section 9, item 1.
> - **Scope:** this file pre-registers the experiment T106 will run. It does not pre-register T107's promotion, and it
>   licenses no autonomous promotion (section 8).

## 0. Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth. T105 is a pre-registration; what decides whether T106 can ever say anything is the evaluator, not the
  optimizer. The ambitious move is to make the evaluator an instrument an adversary cannot game.
- **Ladder.**
  - **R1:** register exactly what T105 lists: workload, constraints, objective J, schedule, horizon, trace splits.
  - **R2:** also register the outside judges and the faults they must catch before any benefit is read: a known-optimum
    landscape the annealer must solve, a sealed test split with a leakage canary, a cost meter the candidate cannot
    self-report to, and a drift segment. Derive the seed count from a measured variance and run it sequentially (P1).
  - **R3:** also serve the candidate in shadow beside the incumbent on the live lake for a cycle, and let a passing
    candidate switch itself in.
- **Pick: R2.**
  - **R3 is ruled out** by P4b (`POLICIES.md:406-435`: the live lake is part of the brain, so a shadow run on it is a
    production read of the live lake, which this plan bars: PLAN-FINAL `:1335-1336`), by T107's own limits (`tasks.md:236`:
    autonomous promotion needs T073/T074, Phase 4's exit and FR-025/OD-2), and by the standing lane rules (the live lake
    `C:\Users\vinhl\.snn_data_lake` is never touched).
- **Range** (methods borrowed, not personas).
  - Database physical design: tune on a training workload with what-if costing, then judge on a later, held-out workload.
  - Group-sequential clinical trials: named looks, an alpha-spending function, a cap with power at the cap (P1 amendment).
  - Adversarial (red team of the optimizer): plant test-set leakage, hidden search cost, an infeasible low-J candidate and
    a broken RNG, and require each to be caught.
- **Outside judge** (none is the method itself):
  - the canonical scan oracle of R1 (`r1-bounded-recall.md:112-141`, `:142`): the set of distinct episodes the generator wrote;
  - `lake_bridge.stream_digest` (`substrate/lake_bridge.py:91`) of the canonical lake before and after every candidate;
  - `home.lake_snapshot` / `home.live_lake_untouched` (`substrate/home.py:602`, `:633`) on the live lake, as a no-touch check;
  - a known-optimum synthetic landscape (section 4, A-R1 d);
  - an external wall-clock and `tracemalloc` meter that the candidate and the search code cannot write to.

## 1. What A1 pre-registers, and what it holds fixed

A1 pre-registers an **offline, sleep-time, discrete search over DERIVED lake-access configuration** on a frozen,
representative recorded query trace (PLAN-FINAL `:1280-1284`). A derived structure is a cache over the lake and is rebuilt
from it (P4b); a missing or discarded cache is "not resident", never amnesia (`POLICIES.md:406-435`).

**Held fixed (T105, `tasks.md:234`; PLAN-FINAL `:1280-1281`, `:1325-1330`):**
- canonical records and their order (`brain_lake.bin`, BinLog frames; `substrate/binlog.py:33`);
- replay selection (the first experiment isolates storage effects);
- durability and fsync policy, including `LakeIndex`'s own `synchronous` setting (`substrate/lake_index.py:60-75`);
- IMMORTAL records, facts and labels; compaction retention; the neuron pager (retired); a second learned-memory store.

Co-recall grouping may reorder DERIVED lookup structures, never the canonical event sequence (PLAN-FINAL `:1281`).

### 1.1 The incumbent primitives (graph-exists run at `c4c10b28` with `--root` on the worktree; every hit disposed)

| Query | Hit | Disposition |
|---|---|---|
| "lake cache" | `substrate/proximity_index.py` (`ProximityIndex`, `:80`): knobs `n_medoids`, `disk_threshold`, `cache_cap`; the bounded OrderedDict LRU page cache (`_pattern`, `:104-123`); `nearest_exact` (`:251`) | **REUSE.** This is the search space's main substrate and the LRU control (section 4). Not rebuilt. |
| | `substrate/lake_recall.py` (`LakeRecall`, `:132`; `refresh`, `:174`; builds `ProximityIndex(q=self.q)` with defaults, `:192`) | **REUSE.** The rebuild cost (`refresh` is "~0.2-0.3 s at 64 episodes", docstring `:174-183`) is the `rebuild_cost` term. The knobs are NOT plumbed through `LakeRecall` today: owed to T106 (section 9, item 7). |
| | `substrate/lake_index.py` (`LakeIndex`, `:60`) | **REUSE as the exact-pointer index.** Rebuildable (`lake_rebuild_gate`); its durability pragmas stay fixed. |
| | `substrate/binlog.py` (`BinLog`, `:33`: resident dedup payloads, `stale()` `:70`, `replay()` `:357`) | **REUSE** as canonical reader and the physical-read counter's anchor. Unchanged. |
| | `substrate/home.py` (`_log` path-keyed instance cache `:720-758`, `_state_index`) | **DISTINCT-FROM.** A correctness structure with a staleness check, not a tunable cache. Not searched. |
| | `sleep/hippocampal_memory.py`, `reasoning/spike_chat_lake.py` (per-boot scratch index `:282-286`, `_kb_index` `:726`), `gates/selfcheck_cache.py`, `reasoning/arc_residual.py::clear_cache`, `reasoning/blimp_bench.py::cached_logprob` | **DISTINCT-FROM.** Replay buffers, wake-path chat scratch and unrelated per-task caches; the wake path is out of scope (A-R3 refuses wake-path optimization, PLAN-FINAL `:1321`). |
| "index" | no lexical match; source inspection found `LakeIndex`, `ProximityIndex` (above) | UNKNOWN is not absence; inspected, no third index. |
| "access pattern" | no match. The only access hook is WRITE-side: `home.set_append_observer` / `observe_append` (`substrate/home.py:995`, `:1017`) | **GAP, not a duplicate.** No read-side query-trace recorder exists. T106 builds one (section 3). It reuses the observer's guarded-call idiom and adds no second write observer. |
| "simulated annealing" | no match; PLAN-FINAL `:1654`: "No simulated annealing exists in the tree" | New by construction. T106 runs graph-exists again with `--root` before adding any `def`. |

P2: no merge is needed because nothing yet does this job. If annealing wins, the tuned configuration becomes the one path
and the incumbent defaults are retired (T107); if it does not, the search code leaves the live path under P8 item 3
(`dormancy_gate.EXEMPT`, disposition `scale-retest`, with a NULL-RECORD). P3: the searcher either runs in the sleep driver
or carries that exemption.

## 2. The non-destructive WORKLOAD

### 2.1 What is searched: the configuration x

Only DERIVED cache, index and access configuration (T105). x has four parts, every one rebuildable from the lake:

1. `n_medoids` from a pinned grid (default `max(1, int(sqrt(N)))`, `proximity_index.py:137`): the index fan-out.
2. `disk_threshold` and `cache_cap` from pinned grids (defaults 4096 and 256, `proximity_index.py:80`): where the index
   goes out-of-core and how many decoded patterns stay resident.
3. **Cache membership M:** a set of resident record groups, bounded by the incumbent's resident-byte budget. A group is a
   co-recall cluster of episode ids found on the TRAINING trace only.
4. Lookup order inside the derived structures (co-recall grouping). The canonical event sequence is never touched.

`nearest_exact` is the only query path scored (`proximity_index.py:251`): `nearest(probe)` is approximate and can miss the
true neighbour (`:239`), so `probe` is not a searched knob, and a configuration may not trade recall for speed.

### 2.2 Hard feasibility constraints (checked BEFORE scoring and acceptance; no temperature accepts an infeasible x)

PLAN-FINAL `:1294-1296`. Each is a constraint, never a penalty term.

| # | Constraint | Judge |
|---|---|---|
| F1 | Exact retrieval: every trace query returns the same recallable set and verdict (VERIFIED/MISS/REFUSED) as the canonical scan | R1's oracle (`r1-bounded-recall.md:142`) |
| F2 | Retention: canonical lake bytes and `stream_digest` identical before and after build and switch | `lake_bridge.stream_digest` |
| F3 | Rebuild correctness: derived state rebuilt from scratch is logically equal to the candidate | rebuild-and-compare |
| F4 | Bounded staleness: candidate watermark equals the recorded lake watermark at the switch point | watermark record |
| F5 | Resource limits: peak resident bytes <= M0, query p95 <= L0, append p95 <= A0 + A/A band (all incumbent-derived, section 2.3) | external meter |
| F6 | Source-event order: canonical offsets and replay order unchanged | `stream_digest` + offset list |

### 2.3 The normalized objective

```text
J(x) = wL*p95_latency/L0 + wM*peak_bytes/M0 + wI*physical_reads/I0 + wW*write_bytes/W0 + wC*rebuild_cost/C0
```

- Denominators L0, M0, I0, W0, C0 are measured on the INCUMBENT configuration on the TRAINING trace and fixed; they are
  never taken from a candidate (PLAN-FINAL `:1291-1292`). Measured values are OWED to the pilot.
- Weights are nonnegative and sum to 1. **Primary: equal weights (0.2 each).** The choice is a declaration; the per-weight
  sensitivity (each weight doubled, renormalized) is reported descriptively and is not a decision input.
- J is the SEARCH objective. The **primary endpoint is not J** (section 5): it is the net time saved over the horizon, so
  that search and build cost enter the claim.

### 2.4 Search, cooling and neighbourhood schedule (finite, seeded; pinned on train and validation only)

- **Neighbourhood moves** (three types): step one scalar knob by +-1 grid cell; swap one cache group in or out of M under
  the byte budget; swap the order of two adjacent groups. A proposal is generated, then checked against F1-F6; an
  infeasible proposal is rejected and **consumes no acceptance draw** (invariant tested by A-R1).
- **Initial temperature T0:** set so that the median worsening Delta-J, measured from a fixed sample of random feasible
  moves on the training trace, is accepted with probability 0.8. Derived, not a constant. The probability 0.8 and the
  sample size are pinned at freeze from the pilot (OWED).
- **Cooling:** geometric, T_k = T0 * r^k, with r chosen so that T_end gives 0.01 acceptance of the median worsening at
  the last evaluation. **No reheat** in the first experiment (reheating needs its own pre-registration).
- **Budget:** a fixed number B of J evaluations, identical for annealing, greedy and random (equal-budget). B is derived
  from the pilot as the count at which the pilot's random-search best-so-far curve has flattened; OWED.
- **Stopping rule:** budget exhausted, or no improvement of the best feasible J in 20% of B (pinned).
- **Best feasible** is kept separately from the current state, because the last accepted state may be worse
  (PLAN-FINAL `:1296-1298`). The candidate scored in T106 is the best feasible configuration on VALIDATION.
- **Seeds:** a search seed per (workload seed, arm). The RNG, the schedule and the move generator are one object whose
  state is persisted with the result (lineage for T107).
- **Acceptance rule:** accept a worsening feasible proposal with probability `exp(-(J_new - J_cur)/T)`; accept any
  improving feasible proposal.

### 2.5 Amortization horizon

The claim is a net benefit over a declared deployment horizon H (PLAN-FINAL `:1304-1305`).
- Cost_total(x) = (wall-clock to search) + (wall-clock to build and switch) + (sum of per-query latency over H queries).
- **H = q_day * D**, where q_day is the query rate MEASURED on the recorded trace, and **D = 30 days** is the declared
  horizon. D is a declaration. D = 7 and D = 90 are reported descriptively. The break-even horizon is always reported.
- **The effect floor delta\*** (section 5) is derived from the break-even: the smallest relative saving that repays
  search and build cost within H. It is not chosen to make the claim easy.

## 3. The query traces

### 3.1 Where each trace comes from

A trace is a RECORDED sequence of (time, cue, answering tier, bytes read). It is recorded by running the engine on a
**throwaway home** (`home.throwaway_home`, `substrate/home.py:647`), never by reading or writing the live lake (P4b; the
live lake is protected by `live_lake_untouched`, `:633`).

| Trace | Source | What it can and cannot support |
|---|---|---|
| **T-gen** | R1's frozen generator `instruments/recall_audit.py::make_workload(seed, watermark)` (`r1-bounded-recall.md:112-141`): cue classes C1-C10 over W_c (4,000 episodes, ~8 MB) and W_real (50,000 episodes, 513 MB, a PROJECTION for the episode count), with popularity and drift laid over the cue stream | Ground truth is known, so it supports A-R1 and the correctness side of A-R2. **It is synthetic. It cannot by itself support a claim about this brain's access pattern.** |
| **T-rec** | A read-side recorder (OWED, section 9 item 4) logging `Brain.recall_episode_durable` calls during a real engine run on a throwaway home over a W_real-shaped lake | The only trace type that supports the primary claim. |

The **primary held-out benefit is judged on T-rec**. T-gen carries the planted controls (known popularity, planted drift).

### 3.2 Splits (chronological; no held-out query informs the search)

- **Train** = first 50% of the trace by time; **validation** = next 20%; **test** = last 30%. Never shuffled.
- The test file is hashed (`sha256`) BEFORE the search starts and handed only to the evaluator process. The search process
  has no read path to it (the leakage canary, section 4).
- Warm-up for a warm run is drawn only from the train tail.

### 3.3 Required strata (in T105's words: cold/warm, rare/old and drift)

| Stratum | Definition | Why |
|---|---|---|
| Cold | fresh process, empty derived caches, first N_cold test queries | restart and boot cost |
| Warm | after a warm-up of N_warm train-tail queries | steady state |
| Rare/old | queries whose target lies in the oldest 20% of offsets AND has train frequency <= 1 (including never seen in train) | where cache membership cannot help; guards against an optimizer that buys head-latency with tail-latency |
| Drift | the test segment, plus a **constructed drift arm** on T-gen: the popularity ranking of the test segment is permuted relative to train. Natural drift is measured as the Jensen-Shannon divergence between train and test key frequencies and reported | the "regression after workload drift" fail condition of A-R2 |

N_cold, N_warm and the strata sizes are pinned from the pilot (OWED).

## 4. T106's controls and the reds it must show

### 4.1 Controls (named now; unchanged, LRU/recency, greedy, equal-budget random)

All four are scored on **identical (trace, seed) pairs** and the same end-to-end costs, including search and build cost.

| Control | Definition |
|---|---|
| **Unchanged** | the incumbent defaults: `ProximityIndex` with `n_medoids=None`, `disk_threshold=4096`, `cache_cap=256` (`proximity_index.py:80`) |
| **LRU/recency** | the same resident-byte budget M0 filled by a measured LRU/recency policy over the train trace (the incumbent page cache is already an OrderedDict LRU, `:104-123`) |
| **Greedy** | best-improvement hill climb over the SAME neighbourhood and feasibility checks, accepting only improving moves, at the same evaluation budget B |
| **Equal-budget random** | B feasible draws from the same neighbourhood; the best feasible by validation J is kept |

### 4.2 A-R1: constructed faults the instrument must reject (each is a planted red; the gate comes before the rule)

| # | Constructed fault | Required outcome |
|---|---|---|
| a1 | An **infeasible candidate with a far lower J** (violates each of F1..F6 in turn: 6 plants) | refused before scoring and before any RNG draw; the RNG draw count equals the number of FEASIBLE WORSENING proposals |
| a2 | **Acceptance fault:** accept-always and accept-never | flagged by the acceptance-rate monitor and by the Metropolis check: on a two-state toy landscape the observed accept frequency matches `exp(-d/T)` within a binomial tolerance fixed at freeze |
| a3 | **RNG fault:** constant RNG; seed ignored | same seed reproduces the same trajectory byte-for-byte; different seeds differ; a constant stream is flagged |
| a4 | **Schedule fault:** non-monotone cooling; an unregistered reheat; budget overrun | refused |
| a5 | **Best-feasible fault:** a run whose final state is worse than its best | the reported candidate is the best feasible, not the last state |
| a6 | **Temperature limit:** T -> 0 must reproduce greedy's trajectory on the same seed | equal |
| d | **Search-power control (known optimum):** a synthetic landscape with a planted global optimum, in which the default cache is undersized relative to a hot set | the annealer finds the optimum within a pinned tolerance at budget B on the pinned fraction of seeds; if it cannot, the search is blind and any null on the lake is VOID, not a null |

### 4.3 A-R2: the held-out benefit bar (T106 / PLAN-FINAL `:1303-1310`)

**A PASS requires all of:**
1. The primary endpoint (section 5) beats **each** of the four controls on the held-out test split, with search and build
   cost included, in a sequential design that stops only on a written boundary (section 6).
2. No secondary bound regresses: F1-F6 hold on every test query; each of p95 latency, peak resident bytes, physical reads,
   write bytes and rebuild cost is non-inferior to the incumbent within its pre-registered margin (margins from the A/A
   noise floor, OWED).
3. No regression on the rare/old stratum or on the drift arm beyond its bound.
4. The four fail checks below all fire on their plants.

**A-R2 fails on, and each has a planted red:**
- *No benefit.*
- *Hidden search cost:* a planted evaluator that omits search time must be flagged. Costs come from the external meter, not
  from candidate-supplied fields.
- *Trace leakage:* a planted search that reads the test split must be flagged by the split hash and the process isolation.
- *Regression after workload drift:* the drift arm.

**A lower training objective is not a demonstrated lake improvement** (PLAN-FINAL `:1309-1310`). A null is recorded with its
scale and is **never** answered by weakening a criterion (T106, `tasks.md:235`).

## 5. Endpoints and claimed effect

- **Unit:** one workload seed = one (lake, T-rec trace) pair, plus one search seed per arm.
- **Primary endpoint, per seed and per control c:** `R_c = 1 - Cost_total(annealed) / Cost_total(c)` on the test split, over
  horizon H (section 2.5). Paired by seed.
- **Claim (one-sided):** for every control c in {unchanged, LRU, greedy, random}, the mean of R_c is at least **delta\***, the
  break-even floor of section 2.5. The conjunction over four controls is an **intersection-union test**: each control is
  tested at one-sided alpha = .05 and the claim needs all four, so no alpha is split across controls. Power is computed for
  the conjunction.
- **Secondary bounds** (section 4.3, item 2) are non-inferiority checks; they are conjunctive with the primary.
- **delta\* is OWED** (pilot: search and build wall-clock, per-query saving). It is derived from the break-even, not set by
  hand; if delta\* is so large that no configuration in the grid could reach it, the pre-registration is void before data.

## 6. Sample size (P1)

Per the P1 amendment (`POLICIES.md:172-207`) and the P8 null-scale amendment item 4 (`POLICIES.md:944-995`). **Nothing in
this section has been measured.**

### 6.1 Variance source (OWED, named)

A **pilot** on a DEV seed block (disjoint from the test block; ranges allocated by the integrator before the pilot runs) of
n_pilot seeds, run after the campaign under the heavy-run lock. It reports, per control c:
- SD of the paired per-seed difference of R_c, split into between-workload and within-search (search-seed) components;
- the A/A noise floor (incumbent vs incumbent) for every primary and secondary measure;
- the natural drift (JS divergence), the query rate q_day, per-evaluation cost, search/build cost and the correlation
  across controls (for the conjunction).

If the within-search variance dominates, the design uses m search seeds per workload seed (m from the pilot); otherwise 1.

### 6.2 Fixed-sample target

With one-sided alpha = .05 and power .80: `n_fix = ceil((1.645 + 0.842)^2 * (SD / delta*)^2) = ceil(6.19 * (SD/delta*)^2)`
for the hardest control, corrected for the t distribution and for the conjunction by simulation. The multiplicity
correction across endpoints is the intersection-union structure of section 5 (no alpha split). SD and delta\* are OWED, so
n_fix is OWED.

### 6.3 Sequential design

- **Looks (named):** L1, L2, L3 at information fractions **0.50, 0.75 and 1.00 of the cap** (seed blocks added in order).
- **Alpha-spending:** Lan-DeMets O'Brien-Fleming-type, one-sided total .05: `alpha*(t) = 2 - 2*Phi(1.96/sqrt(t))`.
  Cumulative alpha at the looks (computed with `math.erf`): **.00557 at t = .50, .02363 at t = .75, .05000 at t = 1.0**.
  The z boundaries are NOT yet computed (they need the joint distribution across looks): OWED, with a stdlib script.
- **Futility:** non-binding. By P1 item 2 it is taken only at a look whose mde (at the design's alpha, from the
  pre-registered SD) is at or below delta\*. At information fraction f of the cap the mde is `delta*/sqrt(f*R)`, which is
  at or below delta\* only when `f >= 1/R`. With R about 1.02-1.03 (PROJECTION for three looks), L1 and L2 cannot stop for
  futility, so **the only futility look is the final look, where it coincides with the cap.** L1 and L2 may stop only for
  efficacy.
- **Cap:** `n_cap = ceil(n_fix * R)`, R the simulation-verified inflation factor (PROJECTION 1.02-1.03), so power at the
  cap is at least .80.
- **Simulation before data (OWED):** with the pilot's SD and correlations, simulate (a) a planted null: the false-positive
  rate of the full conjunction stays at or below .05; (b) a planted effect of size delta\*: it crosses with power at least
  .80 at the cap; (c) the probability of every outcome, INCONCLUSIVE included.
- **Rule:** seeds added after an unregistered look void the reading. A futility stop, or a cap reached without crossing,
  is **not a null**: it reports its mde, and P8 item 4 decides (INCONCLUSIVE unless mde <= delta\*).
- **Affordability check (OWED):** if n_cap evaluations at W_real scale exceed the host budget the integrator names, the
  pre-registration is returned for a smaller delta-scale claim before data, not run underpowered.

## 7. Disposition, fixed in advance

| Disposition | Condition |
|---|---|
| **MET** | the sequential design stops on a written efficacy boundary (or the cap is crossed), every A-R1 red fired and all four A-R2 plants fired, all four controls beaten, every secondary bound held |
| **NOT MET, powered** | the cap is reached without crossing (or a futility stop at the final look) with mde <= delta\*, all planted reds fired and the known-optimum control (A-R1 d) passed. Records a NULL-RECORD line (P8 item 1): units = lake episodes, ticks = n/a, queries per trace, seeds, mde, re-test state. The 10x re-test axis is **lake size** (W_real to about 10x W_real, or W_live at 3.3 GB), named here by the author, because a derived-cache benefit scales with the lake |
| **NOT MET, underpowered** | mde > delta\* at the seed count reached (including a stop on the host budget). **INCONCLUSIVE, not a null**, no deletion at any scale (P8 item 4) |
| **VOID** | any planted red did not fire; the known-optimum control failed (the search is blind); a leak, hidden-cost or live-lake touch is detected (`live_lake_untouched` not ok); an unregistered look or added seed; the T-rec trace does not exist (only T-gen) |

A null on W_c alone cannot be a powered null for the lake claim: at W_c the whole lake fits in the page cache.
**Honest prior (not evidence):** the incumbent index is already O(sqrt N) LAESA with a bounded LRU, so a small or null
benefit is a live possibility. That is why the design is powered to register a null rather than to find a win.

## 8. What T107 needs before any promotion (T107's own list, `tasks.md:236`; PLAN-FINAL `:1311-1322`)

Not part of this experiment. Listed so that A1's result is not mistaken for a promotion licence.
- candidates written only to disposable derived state against a recorded lake watermark;
- logical equality and source digests; restart; interrupted-build and interrupted-switch tests (fault-injected, canonical
  lake unchanged); atomic recoverable activation at a safe boundary; watermark reconciliation or retry, never a stale
  switch; the prior configuration kept recoverable;
- bounded, sleep-only execution; persisted lineage (accepted config, workload and evaluator hashes, RNG and search state,
  costs) in the existing lake state; A-R3 refuses wake-path optimization, unbounded search and self-reported scores;
- **autonomous promotion additionally needs T073/T074, Phase 4's exit and FR-025/OD-2**; no canonical deletion, fsync
  reduction, protected-surface change or neuron-pager revival. Autonomous promotion is OUT of scope.
- A MET here earns the right to build T107; it does not promote anything.

## 9. Owed before freeze

1. **R1/T102 accepted** (`tasks.md:231` is `[ ]`), or this file freezes on `make_workload` at a named commit with the
   dependency recorded. A1 "records the workload after R1" (`r1-bounded-recall.md:277`).
2. **The pilot** (section 6.1): SD (between and within), A/A noise floor, natural drift, q_day, per-evaluation, search and
   build cost, cross-control correlation. Seed blocks allocated by the integrator; a heavy run, under the lock, after the campaign.
3. **The design simulation** (section 6.3): z boundaries per look, the inflation factor R, planted-null false-positive
   rate, planted-effect power, outcome probabilities. A stdlib script is enough; the numpy path waits for the campaign.
4. **The read-side query-trace recorder** (section 3.1, T-rec). graph-exists found none; `observe_append` is write-side.
   Without it the primary claim has no admissible trace and the file is VOID by section 7. Its design must make the recorder
   guarded (a failing recorder never breaks a recall) and must write only to a throwaway home.
5. **Fixed numbers from the pilot:** L0, M0, I0, W0, C0, A0, delta\*, B, T0's acceptance probability and sample size,
   N_cold, N_warm, stratum sizes, the secondary non-inferiority margins, the Metropolis tolerance and the known-optimum
   tolerance.
6. **Lake-size scale:** confirm W_real's 50,000-episode count (a PROJECTION in R1) and whether a W_live-scale run is
   affordable; fix the primary scale.
7. **Knob plumbing:** `LakeRecall.refresh` builds `ProximityIndex(q=self.q)` with defaults (`lake_recall.py:192`), so
   `n_medoids`, `disk_threshold` and `cache_cap` are not reachable from the recall path. T106 must plumb them without a
   second index mechanism (P2) and re-run graph-exists with `--root` before adding any `def`.
8. **The P1 gate fields** (`POLICIES.md:203-207`): measured SD, effect, target, looks, spending function, futility, cap
   must be present as fields at HEAD when this freezes.
9. **Seed blocks and branches:** dev and test blocks allocated; the frozen trace and evaluator hashes recorded.
10. **Review:** panel and policy-enforcer AGREE; every challenge resolved or recorded; then drop "(DRAFT)" in a dated freezing commit.
11. **Task registration:** whether T105 is closed by this file's freezing commit or T106 carries the open items, decided by the integrator.

*Motivation only, not a prediction* (PLAN-FINAL `:1327-1329`): annealed data-placement research
(https://research.ibm.com/publications/improving-server-performance-on-transaction-processing-workloads-by-enhanced-data-placement).
