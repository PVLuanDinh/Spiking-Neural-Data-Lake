# A1 / T105 -- annealing for lake access: pre-registration (DRAFT)

> **Status: DRAFT, NOT FROZEN, NOTHING RUN.** Written 2026-10-29 by lane `a1-prereg` (docs builder, thread A1-PREREG)
> against base `c4c10b28` on `claude/a1-prereg-1029`. No code, test, pilot, simulation of the design or brain run
> produced anything below: every number is either copied from a cited file or marked PROJECTION or OWED. It freezes only
> after the "Owed before freeze" list (section 9) is closed and the review panel and the policy-enforcer AGREE.
>
> - **Row:** tasks.md T105 [A1] (`tasks.md:234`), FR-040 (`spec.md:166`), SC-018 (`spec.md:209`). T105's wording is the
>   checklist; where this draft and that wording differ, the row binds.
> - **Order:** R1/T102 -> A1/T105 -> A2/T106 -> A3/T107 (`tasks.md:274`). T102 is still `[ ]`
>   (`tasks.md:231`): see section 9, item 1.
> - **Fold 1 (2026-10-29):** the 3-lens panel and the policy-enforcer CHALLENGED this draft. Lens A's BLOCK holds at
>   `c4c10b28`: the cache knobs are inert on the live recall path. The scored path is now the whole-lake standalone
>   `LakeRecall(path=...)` mode (section 2.0), which is NOT the live path and NOT T107's promotion target. Several fixes
>   need an integrator ruling (section 9, items 12-17). The changelog is at the end of the file.
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
representative recorded query trace (PLAN-FINAL `:1280-1281`). A derived structure is a cache over the lake and is rebuilt
from it (P4b); a missing or discarded cache is "not resident", never amnesia (`POLICIES.md:406-435`).

**Held fixed (T105, `tasks.md:234`; PLAN-FINAL `:1283`, `:1323-1325`):**
- canonical records and their order (`brain_lake.bin`, BinLog frames; `substrate/binlog.py:33`);
- replay selection (the first experiment isolates storage effects);
- durability and fsync policy, including `LakeIndex`'s own `synchronous` setting (`substrate/lake_index.py:60-75`);
- IMMORTAL records, facts and labels; compaction retention; the neuron pager (retired); a second learned-memory store.

Co-recall grouping may reorder DERIVED lookup structures, never the canonical event sequence (PLAN-FINAL `:1282`).

### 1.1 The incumbent primitives (graph-exists run at `c4c10b28` with `--root` on the worktree; every hit disposed)

| Query | Hit | Disposition |
|---|---|---|
| "lake cache" | `substrate/proximity_index.py` (`ProximityIndex`, `:80`): knobs `n_medoids`, `disk_threshold`, `cache_cap`; the bounded OrderedDict LRU page cache (`_pattern`, `:104-123`); `nearest_exact` (`:251`) | **REUSE.** This is the search space's main substrate and the LRU control (section 4). Not rebuilt. |
| | `substrate/lake_recall.py` (`LakeRecall`, `:132`; `refresh`, `:174`; builds `ProximityIndex(q=self.q)` with defaults, `:192`) | **REUSE.** Its standalone `path=` mode is the scored path (section 2.0). The rebuild cost (`refresh` is "~0.2-0.3 s at 64 episodes", docstring `:174-183`) is the `rebuild_cost` term. The knobs are NOT plumbed through `LakeRecall` today: owed to T106 (section 9, item 7). |
| | `substrate/proximity_replay.py` (`LakeMemory._ensure_index`, `:240`, also builds `ProximityIndex(q=self.q)`) | **HELD FIXED.** The replay and chat path; replay is held fixed (T105). Item 7's plumbing stays per-instance, and the class defaults of `ProximityIndex` do not change, so this consumer is untouched. |
| | `substrate/lake_index.py` (`LakeIndex`, `:60`) | **REUSE as the exact-pointer index.** Rebuildable (`lake_rebuild_gate`); its durability pragmas stay fixed. |
| | `substrate/binlog.py` (`BinLog`, `:33`: resident dedup payloads, `stale()` `:70`, `replay()` `:357`) | **REUSE** as canonical reader and the physical-read counter's anchor. Unchanged. |
| | `substrate/home.py` (`_log` path-keyed instance cache `:720-758`, `_state_index`) | **DISTINCT-FROM.** A correctness structure with a staleness check, not a tunable cache. Not searched. |
| | `sleep/hippocampal_memory.py`, `reasoning/spike_chat_lake.py` (per-boot scratch index `:282-286`, `_kb_index` `:726`), `gates/selfcheck_cache.py`, `reasoning/arc_residual.py::clear_cache`, `reasoning/blimp_bench.py::cached_logprob` | **DISTINCT-FROM.** Replay buffers, wake-path chat scratch and unrelated per-task caches; the wake path is out of scope (A-R3 refuses wake-path optimization, PLAN-FINAL `:1319`). |
| "index" | no lexical match; source inspection found `LakeIndex`, `ProximityIndex` (above) | UNKNOWN is not absence; inspected, no third index. |
| "access pattern" | no match. The only access hook is WRITE-side: `home.set_append_observer` / `observe_append` (`substrate/home.py:995`, `:1017`) | **GAP, not a duplicate.** No read-side query-trace recorder exists. T106 builds one (section 3). It reuses the observer's guarded-call idiom and adds no second write observer. |
| "simulated annealing" | no match; PLAN-FINAL `:1654`: "No simulated annealing exists in the tree" | New by construction. T106 runs graph-exists again with `--root` before adding any `def`. |
| "random search with greedy hill climb baseline over configurations" (Fold 1) | no lexical match; source inspection found `faculties/quality_diversity.py` `random_search` (`:305`) and `greedy_hillclimb` (`:318`) | **DISTINCT-FROM.** They search spiking-policy weights for behaviour-cell coverage (MAP-Elites controls), not a derived lake configuration under hard feasibility. Their shape (equal-budget random, a (1+1) climb) is the precedent the A1 controls follow; T106 re-checks REUSE of the loop skeleton before adding a `def`. |
| "accept only if better ratchet search" (Fold 1) | `reasoning/self_improve.py` (`:1`); `faculties/executive_control.py::plan_sequence` (`:179`) | **DISTINCT-FROM.** `self_improve` ratchets CODE changes behind gates, and `plan_sequence` searches action rollouts. Neither searches a cache configuration. |
| "query trace recorder for recall calls" (Fold 1) | `sensorimotor/recognizer_bank.py::StorageRecognizer` (`:223`); `substrate/episodic_recall.py::EpisodicRecall` (`:263`) | **DISTINCT-FROM.** Both are memories that answer recalls; neither logs the cue stream. The read-side recorder stays a GAP (item 4). |

P2: no merge is needed because nothing yet does this job. If annealing wins, the tuned configuration becomes the one path
and the incumbent defaults are retired (T107); if it does not, the search code leaves the live path under P8 item 3
(`dormancy_gate.EXEMPT`, disposition `scale-retest`, with a NULL-RECORD). P3: the searcher either runs in the sleep driver
or carries that exemption.

## 2. The non-destructive WORKLOAD

### 2.0 The scored access path (Fold 1, lens A BLOCK)

The live recall path cannot be the scored path, because the knobs are inert on it (read at `c4c10b28`):
- `Brain.recall_episode_durable` builds `LakeRecall(episodes=self._episode_window)` (`engine/brain.py:3581-3587`). The
  window is restored with `load_episodes(limit=self._episode_consolidator.capacity)` (`:3537-3541`) and trimmed to that
  capacity (`:2493-2498`). The capacity is `capacity_for(rate, EPISODE_RETENTION_DAYS=53)` (`:1259`, `:160`): window
  scale, "hundreds of episodes" (`lake_recall.py:66`).
- `ProximityIndex.build` keeps the in-RAM path while `n <= disk_threshold` (`proximity_index.py:172-175`; default 4096).
  The LRU `_cache`, `cache_cap` and `_disk_reads` act only in disk mode (`:80-123`). So on the window `cache_cap` changes
  nothing and `physical_reads` is 0 for every x. Only `n_medoids` moves a term there (the LAESA pruning count).
- The DEEP tier is a whole-lake replay scan with exact sub-cue matching and no index (`brain.py:3599-3611`). No knob
  of x reaches it. Indexing it would be a new mechanism, which this fold does not invent.

**The scored path is the whole-lake standalone mode `LakeRecall(path=<lake>)`.** It exists in-tree: `refresh` reads
`load_episodes(self._path)` (`lake_recall.py:180`) and builds one `ProximityIndex` over every episode (`:191-194`). Its
callers today are the module's self-check (`:333`, `:353`, `:461-478`) and `tests/test_lake_recall.py` (`:41-72`). At
W_real (50,000 episodes, more than 4,096) the index is in disk mode at the defaults, so `cache_cap`, `disk_threshold` and
`_disk_reads` all bite. This mode is **not the live recall path**, so a MET is a claim about this mode only. It is
**not T107's promotion target**: putting a whole-lake index on the live path would be its own pre-registered change.

Known costs of this path, disclosed:
- Standalone `refresh` replays the lake on every staleness check, which is O(lake) per `recall` (`lake_recall.py:74-75`,
  honest limit 3). That replay is the same for every x. Whether the endpoint includes it or scores only the
  index query after one refresh is **OWED to the integrator** (section 9, item 12).
- `refresh` calibrates margins with per-arity nulls (`:209-234`), so it costs more than the build alone.

**Each knob must move a measured term (pilot gate).** The pilot (6.1) varies each knob alone, from the incumbent, on the
scored path. A knob whose every grid value leaves all five J terms inside the A/A band is reported as INERT and leaves
x before freeze. **Every J denominator must be positive.** A zero L0, M0, I0, W0 or C0 (for example I0 = 0 when the
pilot lake is below `disk_threshold`) returns the file for redesign before data. It is never patched by a constant.

### 2.1 What is searched: the configuration x

Only DERIVED cache, index and access configuration (T105). x has four parts, every one rebuildable from the lake:

1. `n_medoids` from a pinned grid (default `max(1, int(sqrt(N)))`, `proximity_index.py:137`): the index fan-out.
2. `disk_threshold` and `cache_cap` from pinned grids (defaults 4096 and 256, `proximity_index.py:80`): where the index
   goes out-of-core and how many decoded patterns stay resident.
3. **Cache membership M:** a set of resident record groups, bounded by the incumbent's resident-byte budget. A group is a
   co-recall cluster of episode ids found on the TRAINING trace only.
4. Lookup order inside the derived structures (co-recall grouping). The canonical event sequence is never touched.

**Parts 3 and 4 have no incumbent hook (Fold 1).** `ProximityIndex`'s cache is a pure OrderedDict LRU (`:104-123`), with
no pinned-membership set and no lookup-order knob. Searching M or the order would need a second cache mechanism (P2), so
they cannot be searched until the integrator rules (section 9, item 13): either drop them from x, or add them under a P2
disposition with graph-exists. Until that ruling, x is parts 1-2 (three scalar knobs).

`nearest_exact` is the only query path scored (`proximity_index.py:251`): `nearest(probe)` is approximate and can miss the
true neighbour (`:239`), so `probe` is not a searched knob, and a configuration may not trade recall for speed.

### 2.2 Hard feasibility constraints (checked BEFORE scoring and acceptance; no temperature accepts an infeasible x)

PLAN-FINAL `:1296-1299`. Each is a constraint, never a penalty term.

| # | Constraint | Judge |
|---|---|---|
| F1 | Exact retrieval: every trace query returns the same recallable set and verdict (VERIFIED/MISS/REFUSED) as the canonical scan | R1's oracle (`r1-bounded-recall.md:142`) |
| F2 | Retention: canonical lake bytes and `stream_digest` identical before and after build and switch | `lake_bridge.stream_digest` |
| F3 | Rebuild correctness: derived state rebuilt from scratch is logically equal to the candidate | rebuild-and-compare |
| F4 | Bounded staleness: candidate watermark equals the recorded lake watermark at the switch point | watermark record |
| F5 | Resource limits: peak resident bytes <= M0 * (1 + m_M), query p95 <= L0 * (1 + m_L), append p95 <= A0 * (1 + m_A). L0, M0, A0 are incumbent-derived (section 2.3). m_M, m_L, m_A are the A/A noise-floor margins of section 4.3 item 2 (OWED, item 5), so a re-measured incumbent is feasible | external meter |
| F6 | Source-event order: canonical offsets and replay order unchanged | `stream_digest` + offset list |

**What F1-F6 can catch on this space (Fold 1, disclosed).** `nearest_exact` is exact for any `n_medoids`
(`proximity_index.py:259-266`), and recall is byte-identical across RAM and disk modes (`:22-27`). So F1 holds for
every x of parts 1-2. F2 and F6 hold whenever only derived state changes, and F4 holds on a frozen recorded lake. On
this space only F5 binds. The planted reds (A-R1 a1) are the only evidence that F1-F4 and F6 fire.

### 2.3 The normalized objective

```text
J(x) = wL*p95_latency/L0 + wM*peak_bytes/M0 + wI*physical_reads/I0 + wW*write_bytes/W0 + wC*rebuild_cost/C0
```

- Denominators L0, M0, I0, W0, C0 are measured on the INCUMBENT configuration on the TRAINING trace and fixed; they are
  never taken from a candidate (PLAN-FINAL `:1296`). Measured values are OWED to the pilot. Each must be positive
  (section 2.0).
- Weights are nonnegative and sum to 1. **Primary: equal weights (0.2 each).** The choice is a declaration; the per-weight
  sensitivity (each weight doubled, renormalized) is reported descriptively and is not a decision input.
- **Reported secondaries outside J (PLAN-FINAL `:1284-1285`):** p50 query latency, and checkpoint/restart cost (time to
  bring the derived state back after a process restart). Both are reported per arm and do not enter J or the decision.
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
- **Search-only constants (P1 census, Fold 1).** 0.8 and 0.01 (cooling acceptance), "20% of B" (stall stop) and the
  equal J weights (2.3) shape only the SEARCH. They change which candidate reaches the test split. They cannot change
  how the test split, the endpoint or the boundaries are computed, so they are declared, not derived. That is not true
  of the split fractions 50/20/30 (3.2) or the rare/old cut "oldest 20%, train frequency <= 1" (3.3): those change what
  the test set contains, so each owes a P1 case or a derivation (section 9, item 5).
- **Best feasible** is kept separately from the current state, because the last accepted state may be worse
  (PLAN-FINAL `:1300-1301`). The candidate scored in T106 is the best feasible configuration on VALIDATION. Selection
  reads validation J only. The validation strata (3.3) are reported and are not selection inputs.
- **Seeds and streams:** a search seed per (workload seed, arm). It seeds two separate streams: a **proposal stream**
  (move generator) and an **acceptance stream** (the Metropolis uniform draws only). A-R1 a1 counts the acceptance stream
  alone. Both stream states and the schedule are persisted with the result (lineage for T107).
- **Acceptance rule:** accept a worsening feasible proposal with probability `exp(-(J_new - J_cur)/T)`; accept any
  improving feasible proposal.

### 2.5 Amortization horizon

The claim is a net benefit over a declared deployment horizon H (PLAN-FINAL `:1305-1306`).
- Cost_total(x) = (wall-clock to search) + (wall-clock to build and switch) + (sum of per-query latency over H queries).
- **H = q_day * D.**
  - **q_day** comes from a named record pinned BEFORE the pilot. It is never the rate of the driver that records T-rec,
    because a faster driver would shrink H's search-cost share. Which record is OWED to the integrator (section 9,
    item 14).
  - **D = 7 days is primary (Fold 1).** This is the shortest declared horizon, so the author's choice of D cannot make
    MET easier. D = 30 and D = 90 are reported descriptively. D may be replaced before freeze only by a MEASURED horizon,
    with the integrator's ruling (item 14). The options are: the span over which the tuned configuration's advantage
    survives natural drift (the JS-divergence trajectory on T-rec), or the re-tune cadence the sleep driver actually runs.
  - The break-even horizon is always reported.
- **Break-even is R_c = 0.** Cost_total already contains search and build cost, so "the saving repays search and build
  within H" is exactly R_c > 0. The test of section 5 encodes it. **The design effect delta_d** (section 5) is separate
  from break-even, and it is not derived from break-even. (Fold 1 replaces the draft's "delta\* from the break-even",
  which counted search cost twice.)

## 3. The query traces

### 3.1 Where each trace comes from

A trace is a RECORDED sequence of (time, cue, answering tier, bytes read). It is recorded by running the engine on a
**throwaway home** (`home.throwaway_home`, `substrate/home.py:647`), never by reading or writing the live lake (P4b; the
live lake is protected by `live_lake_untouched`, `:633`).

| Trace | Source | What it can and cannot support |
|---|---|---|
| **T-gen** | R1's frozen generator `instruments/recall_audit.py::make_workload(seed, watermark)` (`r1-bounded-recall.md:112-141`): cue classes C1-C10 over W_c (4,000 episodes, ~8 MB) and W_real (50,000 episodes, 513 MB, a PROJECTION for the episode count), with popularity and drift laid over the cue stream | Ground truth is known, so it supports A-R1 and the correctness side of A-R2. **It is synthetic. It cannot by itself support a claim about this brain's access pattern.** |
| **T-rec** | A read-side recorder (OWED, section 9 item 4) logging `Brain.recall_episode_durable` calls during a real engine run on a throwaway home over a W_real-shaped lake. The recorded cue stream is then re-posed to the scored path (section 2.0) under each arm | The only trace type that supports the primary claim. |

The **primary held-out benefit is judged on T-rec**. T-gen carries the planted controls (known popularity, planted drift).

**T-rec's two sources must be named and pinned before the pilot (Fold 1; OWED to the integrator, section 9 item 15):**
- **Lake content:** what fills the W_real-shaped lake. It is generated on the throwaway home and is never copied from
  `C:\Users\vinhl\.snn_data_lake`.
- **Cue driver:** what poses the recalls during the engine run: a frozen script or battery, named at a commit.
- **Representativeness check:** the comparison that makes T-rec "representative" (PLAN-FINAL `:1280-1281`), written
  before the pilot.
- **Claim scope:** if the cues come from `make_workload`, T-rec is T-gen run through the engine. The claim is then
  scoped to that synthetic driver and says nothing about "this brain's access pattern".
- **Cost:** one T-rec per workload seed means n_cap engine runs. These count in the affordability check (6.3).

### 3.2 Splits (chronological; no held-out query informs the search)

- **Train** = first 50% of the trace by time; **validation** = next 20%; **test** = last 30%. Never shuffled.
- The test file is hashed (`sha256`) BEFORE the search starts and handed only to the evaluator process. **A hash detects
  modification, not reading (Fold 1).** So the leakage control that can fire is **OWED to the integrator** (section 9,
  item 16). The options are:
  - **(a) a canary:** test-only episode ids are planted, and the run is flagged if any of them enters the move log or M;
  - **(b) commit-reveal:** the test split is generated or decrypted only after the search commits its candidate hash.
- Warm-up for a warm run is drawn only from the train tail.

### 3.3 Required strata (in T105's words: cold/warm, rare/old and drift)

| Stratum | Definition | Why |
|---|---|---|
| Cold | fresh process, empty derived caches, first N_cold test queries | restart and boot cost |
| Warm | after a warm-up of N_warm train-tail queries | steady state |
| Rare/old | queries whose target lies in the oldest 20% of offsets AND has train frequency <= 1 (including never seen in train) | where cache membership cannot help; guards against an optimizer that buys head-latency with tail-latency |
| Drift | the test segment, plus a **constructed drift arm** on T-gen: the popularity ranking of the test segment is permuted relative to train. Natural drift is measured as the Jensen-Shannon divergence between train and test key frequencies and reported | the "regression after workload drift" fail condition of A-R2 |

The strata are defined the same way on the VALIDATION segment as on test, since T105 names "train/validation/test query
traces" with these cases (Fold 1). On validation they are reported only; selection reads validation J (2.4).

N_cold, N_warm and the strata sizes are pinned from the pilot (OWED).

## 4. T106's controls and the reds it must show

### 4.1 Controls (named now; unchanged, LRU/recency, greedy, equal-budget random)

All four are scored on **identical (trace, seed) pairs** and the same end-to-end costs, including search and build cost.

| Control | Definition |
|---|---|
| **Unchanged** | the incumbent defaults: `ProximityIndex` with `n_medoids=None`, `disk_threshold=4096`, `cache_cap=256` (`proximity_index.py:80`) |
| **LRU/recency** | the same resident-byte budget M0 filled by a measured LRU/recency policy over the train trace (the incumbent page cache is already an OrderedDict LRU, `:104-123`). **Fold 1: as written this IS the Unchanged control.** The definition is OWED to the integrator (section 9, item 17), with three options: (a) a distinct LRU policy, for example LRU sized to M0 in BYTES rather than `cache_cap` entries; (b) LRU over co-recall record groups, which depends on item 13 reinstating M; (c) drop it to three controls and disclose the identity |
| **Greedy** | **first-improvement** stochastic hill climb on the SAME seeded proposal stream (2.4), with the same feasibility checks: it accepts the first improving feasible proposal and rejects every worsening one, at the same evaluation budget B (Fold 1: was best-improvement, which a correct annealer at T -> 0 cannot reproduce) |
| **Equal-budget random** | B feasible draws from the same neighbourhood; the best feasible by validation J is kept |

**Per-part attribution (P8 item 3, Fold 1).** On the winning x, one **revert-to-incumbent arm per searched knob** is
scored with the primary, on the same (trace, seed) pairs. Each arm resets one knob to its incumbent value: `n_medoids`,
`disk_threshold`, `cache_cap`, plus M and order if item 13 reinstates them. The report gives each arm's R against the
winner. A gain that no single part carries is reported as a finding, not explained away.

### 4.2 A-R1: constructed faults the instrument must reject (each is a planted red; the gate comes before the rule)

| # | Constructed fault | Required outcome |
|---|---|---|
| a1 | An **infeasible candidate with a far lower J** (violates each of F1..F6 in turn: 6 plants) | refused before scoring and before any acceptance draw; the ACCEPTANCE-stream draw count (2.4) equals the number of FEASIBLE WORSENING proposals. Proposal-stream draws are not counted |
| a2 | **Acceptance fault:** accept-always and accept-never | flagged by the acceptance-rate monitor and by the Metropolis check: on a two-state toy landscape the observed accept frequency matches `exp(-d/T)` within a binomial tolerance fixed at freeze |
| a3 | **RNG fault:** constant RNG; seed ignored | same seed reproduces the same trajectory byte-for-byte; different seeds differ; a constant stream is flagged |
| a4 | **Schedule fault:** non-monotone cooling; an unregistered reheat; budget overrun | refused |
| a5 | **Best-feasible fault:** a run whose final state is worse than its best | the reported candidate is the best feasible, not the last state |
| a6 | **Temperature limit:** at T -> 0 the annealer accepts no worsening move, and its trajectory equals the first-improvement Greedy's (4.1) on the same seeded proposal stream | equal |
| d | **Search-power control (known optimum):** a synthetic landscape with a planted global optimum, in which the default cache is undersized relative to a hot set | the annealer finds the optimum within a pinned tolerance at budget B on the pinned fraction of seeds; if it cannot, the search is blind and any null on the lake is VOID, not a null |
| d-red | **Blind searcher (Fold 1):** a planted random walk at budget B, with no acceptance rule, run on d's landscape | must FAIL d's tolerance. If it passes, d cannot tell a blind search from a working one, and d is VOID |

### 4.3 A-R2: the held-out benefit bar (T106 / PLAN-FINAL `:1303-1310`)

**Scope:** every bar below reads the scored path of section 2.0.

**A PASS requires all of:**
1. The primary endpoint (section 5) beats **each** of the four controls (three if item 17 rules (c)) on the held-out test split, with search and build
   cost included, in a sequential design that stops only on a written boundary (section 6).
2. No secondary bound regresses: F1-F6 hold on every test query; each of p95 latency, peak resident bytes, physical reads,
   write bytes and rebuild cost is non-inferior to the incumbent within its pre-registered margin (margins from the A/A
   noise floor, OWED).
3. No regression on the rare/old stratum or on the drift arm beyond its bound.
4. The four fail checks below all fire on their plants.

**A-R2 fails on, and each has a planted red:**
- *No benefit:* plant the **incumbent submitted as the candidate** (an A/A run through the full pipeline). It must NOT
  pass (Fold 1).
- *Hidden search cost:* a planted evaluator that omits search time must be flagged. Costs come from the external meter, not
  from candidate-supplied fields.
- *Trace leakage:* a planted search that reads the test split must be flagged by the leakage control that item 16 picks.
  A hash and "process isolation" alone cannot flag a read (Fold 1).
- *Regression after workload drift:* the drift arm. Its plant (Fold 1) is a **train-top-K membership candidate** scored
  on the permuted-popularity drift arm, and it must trip the drift bound. This plant needs part 3 (M). If item 13 does
  not reinstate M, the drift plant is OWED to the integrator.

**A lower training objective is not a demonstrated lake improvement** (PLAN-FINAL `:1310`). A null is recorded with its
scale and is **never** answered by weakening a criterion (T106, `tasks.md:235`).

## 5. Endpoints and claimed effect

- **Unit:** one workload seed = one (lake, T-rec trace) pair, plus one search seed per arm.
- **Primary endpoint, per seed and per control c:** `R_c = 1 - Cost_total(annealed) / Cost_total(c)` on the test split, over
  horizon H (section 2.5). Paired by seed.
- **Claim (one-sided; Fold 1):** for every control c in {unchanged, LRU, greedy, random}, H0_c: mean R_c <= 0 is
  rejected. R_c is net of search and build cost (2.5), so R_c > 0 IS break-even. The conjunction over the controls is an
  **intersection-union test**: each control is tested at one-sided alpha = .05 and the claim needs all of them, so no
  alpha is split across controls. Power is computed for the conjunction.
- **Design effect delta_d (Fold 1):** the smallest NET saving of interest on R_c. It sizes n_fix (6.2), and it is the bar
  that P8 item 4 compares the mde against (section 7). It is separate from break-even and derived, not set by hand. The
  derivation source is **OWED to the integrator** (section 9, item 14b), with two options: (a) the A/A noise floor of R_c
  from the pilot; (b) T107's own build and switch cost amortized over H. If delta_d is so large that no configuration in
  the grid could reach it, the pre-registration is void before data. (This replaces the draft's "delta\*", which came
  from break-even and so was 0 or counted search cost twice.)
- **Secondary bounds** (section 4.3, item 2) are non-inferiority checks; they are conjunctive with the primary.

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

With one-sided alpha = .05 and power .80: `n_fix = ceil((1.645 + 0.842)^2 * (SD / delta_d)^2) = ceil(6.19 * (SD/delta_d)^2)`
for the hardest control, corrected for the t distribution and for the conjunction by simulation. delta_d is the design
effect of section 5 (Fold 1: it was delta\*, which was 0 on the net endpoint, leaving n_fix undefined). The multiplicity
correction across endpoints is the intersection-union structure of section 5 (no alpha split). SD and delta_d are
OWED, so n_fix is OWED.

### 6.3 Sequential design

- **Looks (named):** L1, L2, L3 at information fractions **0.50, 0.75 and 1.00 of the cap** (seed blocks added in order).
- **Alpha-spending:** Lan-DeMets O'Brien-Fleming-type, one-sided total .05: `alpha*(t) = 2 - 2*Phi(1.96/sqrt(t))`.
  Cumulative alpha at the looks (computed with `math.erf`): **.00557 at t = .50, .02362 at t = .75, .05000 at t = 1.0**
  (Fold 1: t = .75 was misrounded as .02363; the value is .0236226).
  The z boundaries b_1, b_2, b_3 are NOT yet computed (they need the joint distribution across looks): OWED, with a stdlib
  script.
- **Stopping rule for the conjunction (Fold 1).** At look k, the design stops for efficacy if and only if BOTH hold at
  that same look:
  - every control's R_c test crosses b_k;
  - every non-inferiority bound of 4.3 item 2 holds, each tested one-sided against the same b_k (IUT, so no split).
  A control that crossed at an earlier look does NOT stay crossed: it must cross again at look k. If only some controls
  cross, the trial continues to the next look. The design simulation (below) runs exactly this rule.
- **Futility:** non-binding. By P1 item 2 it is taken only at a look whose mde is at or below delta_d. Fold 1 states the
  mde basis. At look k, with information fraction f_k and n_k = f_k * n_cap seeds,
  `mde_k = (b_k + 0.842) * SD / sqrt(n_k)`: one-sided, at that look's boundary b_k, power .80, with the pre-registered SD.
  - At L3, n_3 = n_cap = n_fix * R with R chosen so power at the cap is at least .80, so mde_3 <= delta_d by construction.
  - At L1 and L2, n_k < n_cap and b_k > b_3, so mde_k > delta_d.
  - So **the only futility look is the final look, where it coincides with the cap.** L1 and L2 may stop only for efficacy.
- **Cap:** `n_cap = ceil(n_fix * R)`, R the simulation-verified inflation factor (PROJECTION 1.02-1.03), so power at the
  cap is at least .80.
- **Simulation before data (OWED):** with the pilot's SD and correlations, and the stopping rule above, simulate:
  - (a) a planted null: the false-positive rate of the full conjunction stays at or below .05;
  - (b) a planted effect of size delta_d: it crosses with power at least .80 at the cap;
  - (c) the probability of every outcome, INCONCLUSIVE included.
- **Rule:** seeds added after an unregistered look void the reading. A futility stop, or a cap reached without crossing,
  is **not a null**: it reports its mde, and P8 item 4 decides (INCONCLUSIVE unless mde <= delta_d).
- **Affordability check (OWED):** if n_cap evaluations at W_real scale exceed the host budget the integrator names, the
  pre-registration is returned for a smaller delta-scale claim before data, not run underpowered. The n_cap T-rec engine
  runs (3.1), the per-arm attribution arms (4.1) and the whole-lake `refresh` cost (2.0) all count in that budget.

## 7. Disposition, fixed in advance

| Disposition | Condition |
|---|---|
| **MET** | the sequential design stops on a written efficacy boundary (or the cap is crossed), every A-R1 red fired (d-red included) and all four A-R2 plants fired, every control of 4.1 beaten at the same look (6.3 stopping rule), every secondary bound held |
| **NOT MET, powered** | the cap is reached without crossing (or a futility stop at the final look) with mde <= delta_d, all planted reds fired and the known-optimum control (A-R1 d) passed. Records a NULL-RECORD line (P8 item 1): units = lake episodes, ticks = n/a, queries per trace, seeds, mde, re-test state. The 10x re-test axis is **lake size in episodes**, named here by the author, because a derived-cache benefit scales with the lake. **Fold 1:** the re-test lake holds **>= 500,000 episodes (>= 10x W_real's 50,000)**, generated on a throwaway home (`home.throwaway_home`). It is never `C:\Users\vinhl\.snn_data_lake` or a copy of it. (The draft's "W_live at 3.3 GB" is 6.4x W_real's 513 MB, below 10x, and it named the live lake.) |
| **HARM (Fold 1)** | the annealed configuration passes a pre-registered secondary bound (4.3 item 2) or the rare/old or drift bound (4.3 item 3) in the harm direction, at the design's alpha. A measured harm: the search code is deleted without a re-test (P8 null-scale item 4, `POLICIES.md:968`), and the incumbent stays |
| **NOT MET, underpowered** | mde > delta_d at the seed count reached (including a stop on the host budget). **INCONCLUSIVE, not a null**, no deletion at any scale (P8 item 4) |
| **VOID** | any planted red did not fire; the known-optimum control failed (the search is blind); a leak, hidden-cost or live-lake touch is detected (`live_lake_untouched` not ok); an unregistered look or added seed; the T-rec trace does not exist (only T-gen); a J denominator is zero or a searched knob is inert on the scored path (2.0) |

A null on W_c alone cannot be a powered null for the lake claim: W_c (4,000 episodes) is below `disk_threshold`
(4,096), so at the defaults the index stays in RAM and `cache_cap` is inert (section 2.0).
**Honest prior (not evidence):** the incumbent index is already O(sqrt N) LAESA with a bounded LRU, so a small or null
benefit is a live possibility. That is why the design is powered to register a null rather than to find a win.

## 8. What T107 needs before any promotion (T107's own list, `tasks.md:236`; PLAN-FINAL `:1312-1321`)

Not part of this experiment. Listed so that A1's result is not mistaken for a promotion licence.
- candidates written only to disposable derived state against a recorded lake watermark;
- logical equality, source digests and continuation (PLAN-FINAL `:1313`); restart; interrupted-build and interrupted-switch tests (fault-injected, canonical
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
   guarded (a failing recorder never breaks a recall) and must write only to a throwaway home. **P3 disposition owed now
   (Fold 1):** it lands in `src`, so it is either charged to a live region or EXEMPT as an instrument, named before it is
   built.
5. **Fixed numbers from the pilot:** L0, M0, I0, W0, C0, A0 (each positive, section 2.0), delta_d, B, T0's acceptance
   probability and sample size, N_cold, N_warm, stratum sizes, the margins m_L, m_M, m_A and the other secondary
   non-inferiority margins, the Metropolis tolerance and the known-optimum tolerance. The pilot also runs the knob-mobility
   gate (2.0). **P1 case or derivation owed (Fold 1):** the split fractions 50/20/30 (3.2) and the rare/old cut "oldest
   20%, train frequency <= 1" (3.3), because they change what the test set contains (2.4).
6. **Lake-size scale:** confirm W_real's 50,000-episode count (a PROJECTION in R1), and whether the >= 500,000-episode
   re-test lake (section 7) is affordable. Fix the primary scale. It must exceed `disk_threshold`'s default 4,096
   (section 2.0).
7. **Knob plumbing:** `LakeRecall.refresh` builds `ProximityIndex(q=self.q)` with defaults (`lake_recall.py:192`), so
   `n_medoids`, `disk_threshold` and `cache_cap` are not reachable from the recall path. T106 must plumb them without a
   second index mechanism (P2) and re-run graph-exists with `--root` before adding any `def`. The plumbing is
   per-instance: `ProximityIndex`'s class defaults stay as they are, so `proximity_replay.LakeMemory` (`:240`) and the
   live window tier are unchanged (Fold 1).
8. **The P1 gate fields** (`POLICIES.md:203-207`): measured SD, effect, target, looks, spending function, futility, cap
   must be present as fields at HEAD when this freezes.
9. **Seed blocks and branches:** dev and test blocks allocated; the frozen trace and evaluator hashes recorded.
10. **Review:** panel and policy-enforcer AGREE; every challenge resolved or recorded; then drop "(DRAFT)" in a dated freezing commit.
11. **Task registration:** whether T105 is closed by this file's freezing commit or T106 carries the open items, decided by the integrator.
12. **Integrator ruling: the endpoint on the scored path (Fold 1, lens A BLOCK).** Standalone `refresh` replays the
    whole lake on every `recall` (2.0). The options are:
    - (a) score end-to-end, replay included. This is honest about the mode, but the x-invariant replay dilutes R_c;
    - (b) score the index query (`_nearest_d` after one refresh per test segment), with the replay reported separately.
    If the integrator judges that the standalone mode is not a fit target at all, **A1 is BLOCKED**: the only other
    knob-moving path needs a new mechanism (an index under the deep tier), which this fold does not invent.
13. **Integrator ruling: x parts 3-4 (membership M, lookup order).** There is no hook in `ProximityIndex` (2.1). The
    options are: (a) drop them, so x is three scalar knobs; (b) add them as a new mechanism under a P2 disposition, with
    graph-exists. Items 17 (b) and the drift plant (4.3) depend on this ruling.
14. **Integrator ruling: q_day's source and D (2.5).** Name the record q_day comes from, pinned before the pilot. D = 7
    is primary unless a measured horizon replaces it before freeze.
    14b. **Integrator ruling: delta_d's derivation (section 5).** The options are: (a) the A/A noise floor of R_c;
    (b) T107's build and switch cost amortized over H.
15. **Integrator ruling: T-rec's lake source, cue driver and representativeness check (3.1)**, pinned before the pilot.
16. **Integrator ruling: the leakage control (3.2).** The options are: (a) a canary; (b) commit-reveal.
17. **Integrator ruling: the LRU control (4.1).** The options are: (a) a distinct policy, LRU sized in bytes;
    (b) group LRU, which needs M; (c) three controls, with the identity disclosed.

*Motivation only, not a prediction* (PLAN-FINAL `:1327-1330`): annealed data-placement research
(https://research.ibm.com/publications/improving-server-performance-on-transaction-processing-workloads-by-enhanced-data-placement).

## Fold 1 changelog (2026-10-29, lane `fold-a1`)

Fold 1 covers the 3-lens panel (A #6400, B #6398, C #6404) and the policy-enforcer (#6392, #6408) on `1c139cf9`. Every
code cite was read at `c4c10b28`. The per-finding dispositions are in `lanes/engine-1024/fold-a1-row/FOLD.md`. Each
change, by section:

- **Header:** a Fold 1 note. The `PLAN-FINAL :1114` cite is dropped, because that line is the architecture diagram and
  the order is `tasks.md:274`.
- **1.1:**
  - three graph-exists rows (search, ratchet, trace recorder), each disposed DISTINCT-FROM;
  - a HELD-FIXED row for `proximity_replay.LakeMemory` (`:240`);
  - `LakeRecall`'s standalone mode is named as the scored path.
- **2.0 (new):** the BLOCK (lens A), verified. The knobs are inert on the live window tier, and the deep tier is an
  unindexed replay scan. The scored path is whole-lake `LakeRecall(path=...)`, which is not live and not T107's target.
  Also new: a knob-mobility pilot gate, and every J denominator must be positive.
- **2.1:** parts 3-4 (M, order) have no hook. They cannot be searched until item 13 is ruled.
- **2.2:** F5 gets A/A margins m_L, m_M, m_A. A disclosure that only F5 binds on this space.
- **2.3:** p50 latency and checkpoint/restart cost are added as reported secondaries.
- **2.4:**
  - the P1 census of search-only constants; the split fractions and the rare/old cut are owed a P1 case;
  - selection reads validation J only;
  - separate proposal and acceptance RNG streams.
- **2.5:**
  - D = 7 days is primary (30 and 90 descriptive);
  - q_day comes from a named record pinned before the pilot, never from the T-rec driver;
  - break-even is R_c = 0, and delta\* is replaced by a separate delta_d.
- **3.1:** T-rec's lake source, cue driver, representativeness check, claim scope and cost (item 15).
- **3.2:** a hash cannot detect a read; the leakage control's options go to item 16.
- **3.3:** the strata are also defined on validation (reported only).
- **4.1:**
  - the LRU control as written equals Unchanged (item 17);
  - Greedy is now first-improvement on the same proposal stream;
  - per-knob revert-to-incumbent attribution arms.
- **4.2:** a1 counts acceptance-stream draws only; a6 is restated against first-improvement Greedy; new red d-red
  (a blind searcher must fail d).
- **4.3:**
  - plants for "no benefit" (the incumbent as candidate) and for drift (a train-top-K M candidate, which needs item 13);
  - the leakage plant is tied to item 16;
  - the scope line;
  - PLAN-FINAL cite `:1310`.
- **5:** H0_c: mean R_c <= 0; delta_d (derivation in item 14b).
- **6.2:** n_fix is sized on delta_d.
- **6.3:**
  - the conjunction stopping rule (all controls and all NI bounds at the same look, with no carry-over);
  - the mde basis is stated per look;
  - t = .75 spend corrected to .02362;
  - the affordability budget names the T-rec runs, the attribution arms and the whole-lake refresh.
- **7:**
  - re-test lake >= 500,000 episodes, generated on a throwaway home, never the live lake;
  - a new HARM row (P8 item 4);
  - MET and VOID updated;
  - the W_c note now cites `disk_threshold`.
- **8:** "continuation" restored (PLAN-FINAL `:1313`).
- **9:** items 4 (P3), 5, 6 and 7 extended; new integrator-ruling items 12-17. **Item 12 decides whether A1 is BLOCKED.**
- **PLAN-FINAL line drift fixed:** `:1282`, `:1283`, `:1296`, `:1296-1299`, `:1300-1301`, `:1305-1306`, `:1310`,
  `:1312-1321`, `:1319`, `:1323-1325` and `:1327-1330`.
