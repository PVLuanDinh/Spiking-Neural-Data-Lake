# INVENTION -- the merged design

Four inventions were proposed (APERTURE, THE DISCREPANCY MINT, SLOT, ARREARS) and three
adversarial reviews were run against them. This document is the survivor. It is not an average.
Most of what was proposed is dead, and the deaths are the more useful half of the result: three
of the four proposals were killed by a property of a file that anybody could have opened, and one
was killed by the algebra of its own headline metric.

Every line citation below was opened and read while writing this document. Where a proposal's
citation was wrong, the correction is stated.

---

## 0. What survived, what died, and why

### 0.1 Died -- mechanisms

**APERTURE's resonator register. Dead three times over.**

1. *The physics claim is false in the cited file.* `reasoning/resonate_and_fire.py:32-39` is
   explicit forward Euler -- the `y` update reads the OLD `x`:

   ```
   self.x = x + (self.b * x - self.omega * y + current) * self.dt
   self.y = y + (self.omega * x + self.b * y) * self.dt
   ```

   With `b = 0` the per-step gain is exactly `sqrt(1 + (omega*dt)^2) > 1`. The register does not
   conserve norm through a silent gap; it grows geometrically. "Exactly norm-preserving, no
   tuned decay, structural not chosen" was the entire P1 credit for the maintain step and it is
   void. Line 37 then zeroes both components on threshold crossing, so the map is not invertible
   either, and the units holding the most content erase first.
2. *The derived frequency collapses.* `omega_i = -ln(decay_a)` with
   `decay_a = math.exp(-1.0 / tau_a)` (`growth/dendritic_neuron.py:65`) equals `1/tau_a`, and
   `tau_a` is ONE scalar. All `d/2` pairs would rotate at the same frequency. The "bank" is one
   2-D subspace replicated, effective capacity 2 rather than `d`.
3. *It already exists.* `faculties/theta_gamma_code.py` holds multiple items simultaneously in
   phase slots, re-fires them every theta cycle (active maintenance), and reads them back BY
   SLOT. That is an S-item register with slot-addressed retrieval. `growth/autapse_neuron.py`
   is the delayed-match-across-a-silent-gap harness with an ablation to chance. Building a third
   register is not invention.

**APERTURE's OVERFLOW headline. Dead by its own algebra.**
`OVERFLOW = (partial-report accuracy x S) - (whole-report item count)`, and whole report is
approximately the bus rank `k`. The design's own N5 row says reverting the bus to a rank-1
payload collapses whole report to about 1 item while partial report survives. So OVERFLOW is
maximised by making the report channel WORSE -- it is monotone decreasing in the one quantity the
invention exists to raise. The cheapest large green number is a deliberately crippled aperture.
That is the "guaranteed by construction" failure arriving through arithmetic instead of through a
gradient.

**APERTURE's `c_max = argmax PCI`. Dead on policy.**
Setting a live coupling parameter to the argmax of the repo's perturbational complexity index is
optimising toward the closest thing this repo has to a consciousness marker. Two-sidedness blocks
monotone rigging; it does not block selection toward the metric, which is what an argmax is.
Separately, the harness does not exist: `cortex/integration_measure.PerturbationalComplexity`
constructs its own internal LIF pool and has no API to accept an external population, and
`lz76_complexity` is pure-Python and roughly O(n^2), so the proposed 12-point x 7-seed sweep is
tens of hours, not "hours". **No PCI value appears anywhere in this design, deliberately.**

**ARREARS's delta. Dead as written -- identically 0.0 on every tick, every seed.**
Verified: subscribers are exactly `("action", "language")` (`engine/brain.py:726-727`);
submitters are exactly `"percept"`, `"self"`, `"salience"` (`brain.py:1746`, `:1751`, `:1755`);
`_assoc` is keyed on the SUBMITTER (`reentrant_workspace.py:271`, `:278-280`). The intersection
is 0 of 2. ARREARS computes `sup.get(k, 0.0) - self._assoc.get((k, winner), 0.0)` over
subscribers, so both terms miss and every listener receives exactly `0.0 - 0.0`. This is the same
namespace disjointness that the v4.78 KEY-SPACE FIX comment at `reentrant_workspace.py:289-300`
records and deletes. ARREARS listed the fix ("the subscribers must also submit") as one of five
prerequisites. It is not a prerequisite; it is the whole mechanism, and it is unbuilt.
Secondarily, `_HEBB_LR = 0.25` drives the association toward 1 within about ten co-ignitions
while measured salience sits in 0.668-1.000, so at steady state the delivered payload would be
uniformly NEGATIVE on the winner -- every listener pushed away from the ignited content, every
tick.

**ARREARS's D-gated afference interval. Dead.**
"Close when D rises, reopen when D stops falling" is a strict float comparison against the
previous tick, which is a threshold at exactly zero and the most noise-sensitive one available,
with no hysteresis. With a live content alphabet of 3 keys it reopens on the first flat tick, so
the modal gap is about 1 tick, which cannot charge anything. And the design computes an
autocorrelation window offline and then states it is never used for a live gate -- a
computed-and-not-consumed quantity inside a proposal whose thesis is consuming computed signals.

**SLOT's VSA layer. Dead on cold start and on capacity.**
`keyacc` initialises to zeros, so `key[c] = sign(keyacc[c])` is the all-zero vector for every
subscriber; `bind(msg, 0-vector)` is `[-1] * 512` for everyone, so on tick 0 SLOT delivers
byte-identical payloads -- the exact F1 defect it exists to fix. `sign(0) = 0` also fails the
module's own `is_spike_code` two-level check. Fixing cold start with crc32-seeded keys then makes
the reads near-random per subscriber, at which point SLOT's only bus falsifier (permute the keys,
count changed decisions) passes on pure noise, because SLOT ships a permutation control and no
scramble control. Cleanup SNR at `D = 512` falls as `1/sqrt(n_slots)` against a noise floor of
about 0.044, and SLOT's own text warns that skeletons may end up with 3 or more of 5 fields
slotted.

**SLOT's Falsifier 0. Dead as a falsifier.**
`generalize(fill(S,k1), fill(S,k2)) == S` cannot go red for any implementation matching the
one-line specs given: `fill` writes keys into slots and leaves literals alone, so two terms it
produces differ exactly at the slot positions, and `generalize` returns exactly `S`. The test's
inputs are generated by the function under test. It was described as "the strongest kind
available here"; it is an algebraic identity.

**MINT's forward-holdout admission. Dead -- selection on the test set.**
"Re-run the solver on a forward hold-out of tasks not yet attempted; admit iff solves rise." In a
sequential pass over the evaluation split, "not yet attempted" ARE the remaining evaluation tasks
-- the same tasks the headline (18 of 400) is measured on. Admission and falsifier become the
same computation and the headline is non-decreasing by construction.

**MINT's FlyHash + ART spiking layer. Dead on three independent counts.**
(a) Zero importers. All hits for `fly_hash` outside its own file are prose:
`faculties/action_gating.py:16` and `:77`, `faculties/is_snn_forward.py:77`,
`sensorimotor/lca_sparse_coding.py:14`, `faculties/generalizing_novelty.py:15`. The claim "live:
action_gating, is_snn_forward, lca_sparse_coding" was made from a name grep.
(b) Degenerate on the proposed descriptor: `FlyHash(d=32, expansion=7, wta=0.05)` gives m = 224
units and k = 11 winners (not 12), with `samples` defaulting to 6, so activations over a
~32-boolean descriptor take about 7 distinct values across 224 units and the top-11 cut is
decided mostly by the index tiebreak. The proudest P1 claim in MINT -- vigilance as the equal
error point of two measured distributions -- then has no equal error point to find.
(c) It duplicates `reasoning/schema_induction.SchemaInducer`, which is the same ART-1 with the
same intersection learning and the same crc32-seeded sparse signatures.
Its own null test could not fail either: an exact-signature dict over 32 booleans matches
essentially nothing, so "the spiking layer differs from the dict" was guaranteed.

**MINT's derived vocabulary cap. Dead by substitution into itself.**
`M` solves `B/(P+M) - 1 >= (depth-1 survivor count)`. With MINT's own numbers (B = 4000,
P = 75, survivors = 52) this gives `75 + M <= 75.47`, i.e. **M <= 0** -- no primitive may ever be
admitted. The escape is that the true survivor count is far below 52, because `synthesize`
appends to `nxt` only after two filters (`gs is None`, and `ns in seen`) at
`arc_solver.py:866-875` -- but if survivors are far below the affordable parent count, the budget
never binds and the `dParents/dP = -B/P^2` argument that motivated restricting capture to the
depth <= 1 shell is void. The cost model and the cap cannot both be load-bearing.

**MINT's "retirement must fire" and "total nodes must not rise". Both dead.**
The first is satisfied by loosening admission until junk gets in, and a mechanism with perfect
admission scores red -- the criterion is inverted. The second cannot rise: nodes are capped per
task by the truncation at `arc_solver.py:850-855`, so adding primitives makes tasks hit the cap
sooner and the total can only stay flat or fall.

**The `burst_multiplexing` head-to-head, proposed by three of the four. Dead as a category
error.** That module encodes two scalars into ONE neuron's spike times, stochastically, over 80
bins per window, decoded by temporal grouping. Its rank is 2 by construction and it is a
per-neuron spike-coding primitive, not a bus format. It has zero importers. Racing it against a
population bus on a 27.8 ms tick and deleting the loser retires a working self-checked primitive
on a comparison it was never in. P2 requires that one mechanism survive A COMPARISON; this is not
one.

**`Brain._enter_sleep` as the offline host, proposed by MINT and SLOT. Dead.**
`brain.py:2713-2715` -- `tick()`, docstring "Called under a lock -- returns fast" -- calls
`self._enter_sleep()` synchronously at `:2722`. It is on the tick. MINT hangs 20-40 s per
candidate there and SLOT hangs 3-6 s.

### 0.2 Died -- claims about the tree

Three citations were wrong and each mattered:

* `_Specialist.decide` at `reentrant_workspace.py:409-410` (SLOT's bus consumer) does not exist.
  The method is `emit` at `:408`, and the class docstring at `:399-402` says it is used ONLY by
  the self-check. SLOT's key-permutation arm would have been measured on a test double.
* `assoc_weight` at `:320` (SLOT) does not exist. The public reader is `association(source,
  content)` at `:318`.
* "11 faculties, 110 ordered pairs" (APERTURE, SLOT, ARREARS). There are **2 of 2** registered
  subscribers and **1 of 2** is ever read: `broadcast_to("action")` at `brain.py:1623` is the
  only live read in `src/`; `broadcast_to("language")` is never read. The denominator is 2
  ordered pairs, of which 1 is measurable.

### 0.3 Survived

* **The diagnosis of F1.** It is correct and it had not been said this sharply: the source axis
  is LEARNED and DISCARDED inside the same function. `_hebb_update` maintains a real
  `(source, content)` matrix with a local gradient-free rule (`:268-280`); `_deliver` collapses
  it with `max(...)` at `:301-304` and assigns the resulting scalar identically to every
  subscriber at `:308`. `brain.py:713-725` already documents both defects correctly. F1 is not a
  bandwidth problem. It is a `max()`.
* **rho, the rank surplus instrument** (from ARREARS). Row-centre the (listener x content) matrix
  of delivered payloads; report `rho = sum_{i>=2} sigma_i / sigma_1`. Today rho = 0 as a THEOREM
  about the source, not as a noisy reading, because every row is identical. And rho is invariant
  under any `payload_k(t) -> alpha(t)*payload(t) + beta(t)` -- every "turn up the gain" and every
  "make the shared scalar vary more" proposal scores exactly zero. It retrodicts the reverted
  Claustrum `_attn_gain` wire (a multiplicative gain on one pathway) as a guaranteed zero, before
  the 0-of-200-decisions measurement. This is the best instrument in the four documents and it is
  worth shipping even if both designs below die.
* **The ordered-pair permutation metric** (from APERTURE), with a mandatory invariance companion
  it did not have. Permuting WHO spoke, with content and salience byte-identical, must change a
  listener's output -- and on trials where speaker identity is NOT informative, must not.
  Without the second half a fixed random per-faculty map scores full marks.
* **The scramble control** (from ARREARS): same multiset, wrong assignment. It is the only
  control in the four that separates "the bus carries an address" from "the bus carries
  per-listener variance".
* **The matched-vocabulary control arm** (from SLOT): hand-add the same number of authored
  primitives and compare. It is the only arm anywhere that separates "learned vocabulary" from
  "more vocabulary", and it is the arm a design would most want to skip.
* **MINT's budget arithmetic**, verified against `synthesize` line by line: `nodes += 1` once per
  (frontier state, primitive) at `:849`, so depth 1 costs exactly the pool size and the marginal
  cost of an added primitive is `-B/P^2` affordable depth-2 parents. This explains the retired
  fix-combinator's measured 2-of-400 training regression rather than describing it.
* **MINT's "3 of 22"**, verified: `arc_mint.py:46` imports exactly `object_holes, period,
  shape_class`; `arc_scene.py` defines 22 public measurements plus `main`. The master lever --
  consume the computed-but-not-consumed -- lands on a real 19-function gap.
* **MINT's `ArcLibrary` deletion**, the only P2 retirement in the four with four independent
  measured nulls behind it and a causal mechanism for the null.
* **MINT's "the trap" paragraph.** Mint count, prototype count and allocation rate are telemetry
  and are barred from every success claim, in advance. Adopted verbatim in spirit below.
* **APERTURE's posture in its own limits sections.** "Mechanistically, yes -- that is exactly
  what it is" about the bigger-buffer objection, and the flat statement that no version of the
  experiment discriminates "experienced" from "buffer with a late read port". Adopted.

---

## 1. One design or two

Two. The argument is not convenience.

1. **Schedule.** The access mechanism must complete inside a 27.8 ms tick. The Gf mechanism runs
   many-to-one over a corpus and needs at least two contributing tasks before it may mint
   anything. There is no host for the second on the tick -- `_enter_sleep` is called
   synchronously from `tick()` under a lock (verified above). Merging them forces either minting
   from one instance (overfitting, already measured here) or deferring the broadcast to a
   consolidation window (no tick).
2. **Evidence class, and it is asymmetric.** The Gf filler is verified against ground truth --
   `arc_mint.solves(fn, pairs)` reproduces this task's train pairs exactly, and it can REFUSE.
   The access filler has no ground truth: there is no train pair for "which faculty should have
   been listened to". Its admission evidence is strictly weaker. **The two falsifiers are
   reported separately and are never summed.** Summing them is how a green on the Gf arm pays for
   a red on the access arm.
3. **Failure independence.** The access arm's most likely red (faculty activity is not
   differential, so the payload matrix is rank 1 for a structural reason) and the Gf arm's most
   likely red (the residual capture surface is small, or every new solve is memo-flagged) share
   no cause. A merged design lets one arm's null be explained away by the other's blocker.
4. **The unification is not yet earned.** SLOT is right that `compile_record` is `fill` on the Gf
   side, and the observation that both defects are "a term with no variable positions" is real.
   But its day-one falsifier is an algebraic identity, so nothing tests that the shared object is
   shared. Under P2, a shared abstraction that has not been compared against two separate
   implementations is an add-beside, not an adopt-and-retire. **The unification is pre-registered
   as a third and later question** (section 6, step 8), with a test that can actually fail: for
   every real corpus pair, `fill(generalize(t1, t2), keys_of(t1)) == t1`, over terms nobody
   generated with `fill`.

---

## 2. DESIGN A -- LEDGER

**The bus payload gets an address, and the address is a count.**

### 2.1 The defect, at the line

`faculties/reentrant_workspace.py:282-315`. Inside the per-subscriber loop, `assoc` is computed
from a generator that does not mention the subscriber at all:

```
assoc = max((v for (_src, c), v in self._assoc.items() if c == winner), default=0.0)
bias  = self._REENTRY_GAIN * (1.0 + assoc) if self.reentry else 0.0
s["bias"] = {winner: bias} if bias > 0.0 else {}
```

Every subscriber therefore receives a byte-identical dict, and `reached += 1` fires inside the
branch taken for every subscriber, so `reached / len(self._subs)` is 0.0 or 1.0 and nothing else.

### 2.2 The mechanism

The v4.78 lesson is NOT "never add a per-subscriber axis". It is "never add a per-subscriber axis
with no learning rule and no reader". Nothing currently in the workspace is (speaker x listener)
shaped -- `self._subs[name]` holds only `inbox` and `bias` -- so this design ADDS that axis
rather than pretending to derive it from existing state, and pays for it with a learning rule
that has no rate and no decay, plus a reader, plus a falsifier that can go red.

**The ledger.** `C[s, k]` = the number of ignited ticks on which speaker `s` supported the
winning content AND listener `k` also submitted something. Both endpoints are already computed:
`sources[winner]` is built in `step()` and thrown away, and `self._buffer` is still populated
when `_deliver` runs (`_deliver` is called at `:262`, `self._buffer = []` at `:264`). This is a
pure coincidence count -- an integer, incremented on co-occurrence. No learning rate, no decay,
no gain, no bootstrap problem: it needs no bias to have had an effect before it can grow.

**The payload.** `bias_k = mean over s in sources(winner) of ( C[s,k] / T[s] )`, where `T[s]` is
the count of ignited ticks on which `s` supported the winner. A mean of frequencies of counts, in
[0, 1], zero for every pair on tick 0, and different per listener exactly when different speakers
have historically co-occurred with different listeners. `_REENTRY_GAIN` is deleted.

**The named degeneracy, stated before the build.** If every faculty submits on every tick then
`C[s,k] == T[s]` for every `k`, every column of the delivered matrix is identical, and **rho = 0
for a structural reason**. That is today's submission pattern: `brain.py:1746-1757` submits all
three sources inside one `try` block on every tick. Step 0 of the build order measures this
before a line is written, and a rank-1 answer cancels Design A.

### 2.3 Numbered steps, state, complexity

Symbols: `N` = faculties that both speak and listen (today 2 subscribers of 2 and 3 submitters of
3, disjoint; the merge target is a single namespace of N); `|C|` = live content alphabet (3
today: the perceived digit or `"ctx"`, `"self"`, and `"threat"` or `"novel"`).

| # | Step | Operation | Cost per tick |
|---|---|---|---|
| A1 | Thread `sources` into `_deliver` | one extra argument; the dict already exists | 0 |
| A2 | Active set | `act = {src for (src, _c, _s) in self._buffer}` | O(buffer), about 14 |
| A3 | **Read before write** | for each listener k: `bias_k = mean_s C[(s,k)] / T[s]` over `s in sources(winner)` | N * |sources| dict lookups; 6 at N=2, 121 at N=11 |
| A4 | Deliver | `s["bias"] = {winner: bias_k} if bias_k > 0.0 else {}` | O(N) |
| A5 | Write | `T[s] += 1` for each speaker; `C[(s,k)] += 1` for each (speaker, active listener) | same as A3 |
| A6 | Graded availability | count of listeners with `bias_k > 0` of `len(self._subs)`; **report only** | O(N) |
| A7 | Record for the instrument | `self._delivered = {k: bias_k}` | O(N) |
| A8 | rho accumulate | `M_acc[(k, winner)] += bias_k`, count++ | O(N) |
| A9 | rho (OFFLINE, per window) | row-centre `M` (N x |C|, at most 11 x 11), `numpy.linalg.svd`, `rho = sum(sigma[1:]) / sigma[0]` | about 1.3k flops, once per window |

Read-before-write in A3/A5 is load-bearing: tick `t`'s payload is a function of ticks `1..t-1`
only, so an arm is attributable and a listener's own current activity cannot write its own bias.

**State added:** `self._ledger` (dict, at most N^2 = 121 integer entries), `self._tsup` (dict, at
most N integers), `self._delivered` (at most N floats), `self._M_acc` (at most N * |C| = 121
pairs). Under 2 KB at N = 11. Bounded by construction -- deliberately unlike `s["inbox"]`, which
is documented as growing without bound.

**Complexity at 24-240 widths.** The bus arithmetic is **width-independent**. Population width
enters only through the salience each faculty already computes, which is unchanged. At N = 11 the
per-tick cost is about 250 integer operations against a 27.8 ms tick; one 240-unit recurrent
population update is 240^2 = 57600 multiply-adds, so the mechanism is about 0.4 percent of a
single population and about 0.004 of the tick. It also DELETES an
O(|sources| * |contents|) generator scan (the `max(...)` at `:301`). Nothing here approaches the
measured 262144 numpy-vs-CUDA crossover; the live tick stays on CPU, in pure Python, with no
numpy on the delivery path. rho is offline and is the only numpy call.

### 2.4 What it retires (P2 -- adopt and retire, not add beside)

1. **`_REENTRY_GAIN = 0.08`** (`:85`) -- deleted. The delivered magnitude becomes a frequency of
   two counts.
2. **The `max(...)` source-axis collapse** (`:301-304`) -- deleted.
3. **The uniform `s["bias"] = {winner: bias}` assignment** (`:308`) -- deleted. `broadcast_to`
   keeps its signature and its `{content: magnitude}` return type, so the one live consumer at
   `brain.py:1623-1625` keeps working unchanged. Port before retire: with a single listener the
   new payload must reproduce the old ordering of decisions, asserted by a unit test.
4. **`availability = reached / len(self._subs)`** (`:315`) -- deleted, not left beside. Replaced
   by A6, which is report-only.
5. **`_assoc` as a delivery-path object.** `_hebb_update` and `association()` survive (there is a
   public reader and a self-check), but they leave the delivery path. `_HEBB_LR` and
   `_HEBB_DECAY` are thereby DEMOTED from load-bearing to introspection-only. ARREARS promoted
   them; this demotes them.

**Explicitly NOT retired, named as surviving offenders:** `_SUSTAIN = 1.10`,
`_REENTRY_DECAY = 0.80`, `_COAL_BONUS = 0.06`, `_JITTER = 0.02`, `_G_INH = 2.4`, `_LEAK = 0.5`,
`_W_SELF = 0.6`, `_TAU_SELF = 0.6`, `_K_INH = 0.30`, `_LEAK_INH = 0.5`,
`ignition_threshold = 0.55`, `_ACC_N = 32`, `_ACC_LO = 0.85`, `_ACC_HI = 1.90`. This design does
not clean them and must not claim to.

### 2.5 P1 audit

| Quantity | Derivation | Status |
|---|---|---|
| `C[s,k]` | integer co-occurrence count | clean -- a count |
| `T[s]` | integer count of ignited ticks with s supporting the winner | clean -- a count |
| `bias_k` | mean over speakers of `C/T` | clean -- a frequency of counts, no rate, no decay, no gain |
| ledger init | empty; every pair starts at exactly 0.0 | clean -- silence by default |
| graded availability | count of listeners with a nonzero payload of the subscriber count | clean -- a count, and report-only |
| rho | `sum_{i>=2} sigma_i / sigma_1` of the row-centred delivered matrix | clean -- a ratio of singular values of a count-derived matrix |
| significance of a decision change | permutation and scramble nulls over the same ticks | clean -- empirical nulls, no cutoff |
| determinism | `random.Random(seed)` per instance, existing | clean |

**Declared offenders, this design's own:**

* **`_JITTER = 0.02` is a blocking prerequisite, not merely inherited.** It is drawn from a shared
  RNG stream keyed on content arrival order. LEDGER makes content and speaker identity
  load-bearing, which ACTIVATES that hazard: the mechanism could look alive on a majority of
  seeds for a reason unrelated to the ledger. crc32-keyed per-content jitter must land first.
* **The clamp at `brain.py:1625`** (`elig["reroute"] = min(1.0, self._reroute_bias)`) is an
  existing constant on the consumer side. `bias_k` is in [0, 1] by construction so the clamp
  should never bind, but the fraction of ticks on which it binds must be reported as a health
  check. A payload flattened by saturation is a payload that carries nothing, and this repo has
  already shipped one check that fired on saturation and was read as obstruction.
* **Thirteen inherited workspace literals**, listed in 2.4 above, untouched.

**Net: 1 arbitrating constant deleted, 2 demoted from load-bearing to introspection-only, 0
introduced, 13 inherited and named.** That is a genuine P1 improvement and it is small.

### 2.6 Dissociation, falsifiers, control arms

Every ablation is a pinned dict-subclass neuter that RAISES on write. A value that moves is not a
value that matters; a neuter that merely zeroes a copy leaves the quantity reachable.

Pre-registered: 7 seeds, at least 300 ticks per arm, same seed across arms, and the
action-clamped `charge_probe` arm must exist first (below) or nothing here is attributable.

| Arm | What it pins | Predicted |
|---|---|---|
| **N0** none | -- | baseline |
| **N1 address neuter** | every listener receives the column MEAN of the payload; magnitude preserved, identity destroyed | outputs CHANGE relative to N0 |
| **N2 magnitude neuter** | every payload replaced by the population mean magnitude, per-listener RANKING preserved | outputs UNCHANGED |
| **N3 scramble** | the delivered multiset permuted across listeners -- same values, same entropy, wrong assignment | outputs CHANGE relative to N0 |
| **N4 speaker permutation** | which faculty is labelled as speaker permuted, content and salience and every seed byte-identical | at least 1 of the measurable ordered pairs changes |
| **N5 revert** | payload forced back to `_REENTRY_GAIN * (1 + max assoc)` | rho = 0 exactly |

**The dissociation, and it is the entire empirical content.** The required crossing is **N2
unchanged and N3 changed**: the ADDRESS matters and the MAGNITUDE does not. A bus that is merely
delivering a bigger number predicts the opposite pattern -- N2 moves and N3 does not, because
scrambling a set of numbers preserves their distribution. **If both move, or neither moves, the
design is refuted and the module is deleted, not demoted to a cockpit field.**

**Primary falsifiers, each independently able to kill:**

1. **rho.** Red if rho = 0 after the change. NECESSARY, NOT SUFFICIENT -- rho can be raised by
   injecting independent per-listener noise, which is what N3 exists to catch.
2. **Behavioural charge.** `tools/charge_probe.py` with the ledger payload as the named source,
   at least 200 ticks x 7 seeds, observables `arm_stopped` and `map_position`. Verdict contract
   preserved: `BEHAVIOURAL CHARGE OK` (exit 0) or `CONSUMED-BUT-NOT-DECISIVE` (exit 1). Exit 1
   means revert -- the same standing rule that reverted the `_wm_spikes` wire at
   `engine/autonomous_loop.py:969`, which took a telemetry field from 1 distinct value to 22,
   fired on 195 of 200 ticks, and changed 0 of 200 decisions.
3. **Ordered-pair permutation WITH its invariance companion.** N4's count alone is gameable by a
   fixed random per-faculty map. The pass requires BOTH: on trials where speaker identity IS
   informative about the correct action, the listener's accuracy must RISE; on trials where it is
   NOT informative, the listener's output must be UNCHANGED. Only the conjunction counts.
4. **Count-matched control.** A control arm delivers a nonzero payload on the same NUMBER of
   ticks, scheduled by `zlib.crc32` of the tick index and independent of the ledger. Red if the
   downstream outcome matches -- only the quantity of biasing mattered.
5. **Action clamp.** Replay a fixed action sequence in both arms. Red if every difference
   vanishes -- the effect is routed through the single action gate and is a spoke, which is the
   defect P6 names rather than its fix.
6. **Degeneracy.** Red if the delivered matrix has rank 1 over 300 ticks (see 2.2), or if the
   `min(1.0, ...)` clamp at `brain.py:1625` binds on a majority of ticks.

**Explicitly deleted from the proposals: the availability distinctness gate.** ARREARS proposed
"availability must exceed 2 distinct values". `access_strength()` already ships a graded readout
with 144 distinct values measured over 360 ignited ticks. Moving `availability` from 2 distinct
to N+1 distinct while changing no consumer's output is exactly the `_wm_spikes` shape. Graded
availability is reported and is never a pass criterion.

---

## 3. DESIGN B -- RESIDUAL

**Induce which cells get written, not just what value goes in them.**

### 3.1 The defect

`arc_solver._fit_colormap` (`:713-727`) is a single degenerate repair: one colour-to-colour
lookup keyed on the cell's own colour, fitted cellwise across all pairs, discarded per node. The
mint can already invent PARAMETERS (`_fit_const_modk`, `_fit_offset_mod10`, `_fit_affine`,
`_fit_rank`, `_fit_lookup`), and it can already invent WRITE SETS -- but only over authored
enumerations: `_PAINT_CELLSETS` has exactly **2 of 2** entries (`perimeter`, `allholes`) at
`arc_mint.py:560`, and `_RECOLOR_UNITS` has exactly **8 of 8** authored (unit, attribute) pairs
at `:443-452`. Meanwhile `arc_mint.py:46` imports **3 of 22** of the measurements `arc_scene`
already computes.

So this is not a new kind of invention. It is **widening an existing induction axis from an
authored enumeration to an induced conjunction**, and that framing matters: it inherits a working
MDL-order plus leave-one-out discipline (`induce_rule`, `:252`) and a typed-record bottleneck
that is already auditable by reconstruction.

### 3.2 The mechanism

`TypedPrimitiveRecord` (`:512`) gains ONE field, `selector`, and `compile_record` (`:563`) gains
one branch:

```
apply(g):  U = units(unit, g)
           T = { v in U : selector(v, g) }     # INDUCED conjunction over measured descriptors
           for v in T: write rule(key_attribute(v)) into cells(v)
```

`_fit_colormap` is the instance `selector = all cells, key = the cell's own colour, form =
lookup`. **Port before retire: the general induction must reproduce every solve `_fit_colormap`
currently earns before `_fit_colormap` is deleted.**

**The typed bottleneck is preserved by a versioned enumeration, not by hope.** MINT's
open-endedness clause (an admitted selector becomes a new measurement available to the next
round) would break contract C10/C11: a record referencing measurement #17 is only reconstructible
against the store that defines #17, so mint-internal state crosses the boundary by reference.
Repair adopted here: the descriptor alphabet is a **versioned enumeration**; a record carries the
literal tuple `(predicate-id, args)` from that enumeration and never a store pointer; a version
bump invalidates every stored record. Reconstruction stays the audit.

**No clustering layer, and no credit layer.** MINT proposed FlyHash + ART for approximate
cross-task recurrence and R-STDP for proposal ordering, each with a kill switch. Both kill
switches passed by construction (0.2 above), and both mechanisms are optional. So neither is
built. Recurrence is EXACT: a residual class mints when its induced selector plus value rule
validates on at least 2 contributing tasks. Whether APPROXIMATE matching admits records that
exact matching misses, and whether those records earn solves, is a separate, later, failable
question -- and if it is asked, the null is a nearest-neighbour matcher with its Hamming radius
SWEPT over the full range, not a single exact-signature dict.

### 3.3 Numbered steps, state, complexity

Symbols: `C <= 900` cells (`_MAXS = 30`), `n` = 3 to 5 train pairs, `P` = pool size, `B = 4000`
node budget, `F` = descriptor features, `L` = selector literals, `T` = 400 tasks.

Steps B1-B3 ride the existing solve. Steps B4-B7 run OFFLINE IN A SEPARATE PROCESS against a
snapshot -- not in `_enter_sleep`, which is on the tick.

| # | Step | Where | Cost |
|---|---|---|---|
| B1 | **Capture.** In `close()` (`arc_solver.py:822`), compute the residual mask and wanted values against `train_out`. Only defined where shapes match -- see the gate below. | `arc_solver.py` | O(P*n*C), same cellwise loop `_fit_colormap` already runs |
| B2 | **Describe.** Build the F-vector from `arc_scene` measurements, per residual cell and per unit. | new `arc_residual.py` | O(P*n*(C + objs*F)) |
| B3 | **Stream out.** Append to a snapshot file. Nothing durable is written on the solve path. | `arc_solver.py` | O(1) amortised |
| B4 | **Induce the selector.** Greedy sequential covering over the descriptors, one literal at a time, stopping when the next literal does NOT strictly reduce misclassified cells summed over all train pairs. | offline | O(L*F*C*n) per candidate |
| B5 | **Induce the value rule.** `induce_rule` / `_build_recolor` UNCHANGED: functional forms in MDL order, leave-one-out validated; `rank` / `lookup` accepted only if nothing functional validates, and flagged memo, refuse-on-unseen. | `arc_mint.py:252` | O(units*7*n) |
| B6 | **Verify.** `arc_mint.solves(fn, pairs)` (`:497`) must reproduce EVERY train pair of EVERY contributing task. Then compute the equivariance type: the largest subgroup of D4 x background-fixing colour permutations under which `f(g.x) = g.f(x)`. A selector that commutes with nothing is a memorisation tell. | offline | O(|H|*n*C), about 29k |
| B7 | **Admit.** Re-run the solver at FROZEN budgets on a hold-out drawn ONLY from the 400 TRAINING tasks, with and without the candidate. Admit iff solves rise, or nodes-to-first-solve falls, or a task moves out of `arc_diagnostic`'s `coverage_proven` bin. **Evaluation is run ONCE, at the end, with the record set frozen.** Report `H`, the number of candidates tested, alongside the solve count. | offline | seconds to minutes per candidate, sequential-greedy |

**Complexity on the live tick: ZERO.** Design B touches no tick. It has no per-tick cost at any
population width, and no relationship to the 24-240 band or to the 262144 crossover -- it is
symbolic Python over grids.

**Durable state:** the versioned descriptor enumeration, the admitted record fields, and the
ledger of nodes-charged versus nodes-saved per record. Under 100 KB. It goes in the one lake
(`substrate/lake_memory.py` / `substrate/home.py`), not a new sidecar.

**Retirement of records -- opportunity-matched.** A record is retired when it has been proposed
at least as many times as the MEDIAN admitted record that has earned a solve, has appeared in 0
of its proposals' winning programs, and its ledger is negative. A record that is never proposed
is never retired and costs nothing -- correct, since it cannot be blamed. The bar is the
population's own median, not a constant.

### 3.4 What it retires (P2)

**BUILT AND MEASURED 2026-08-03. One of these three retirements happened; two are withdrawn, and
one of the two cost a solve before it was caught.** Corrections are inline below.

1. **`ArcLibrary` (`arc_solver.py:740-760`) -- deleted outright.** DONE, with its `cap = 64`,
   `_defs`, `_seen`, the `lib:` pool branch, and the `library=` parameter of `make_solver`
   (`:885`). A self-check now fails if it reappears (`arc_solver.py:1007`), because P2 retirement
   means deleted, not flag-parked.

   **Two numbers in the original text are corrected.** The null is real and was re-confirmed
   independently before anything was deleted -- library ON and OFF return **byte-identical**
   solved-ID sets on both splits while learning 4 composites. But the counts quoted here, 12 of
   400 evaluation and 43 of 400 training, were stale by two arcs: the measured pre-Design-B
   baseline is **18 of 400 evaluation and 53 of 400 training**. And **the "115 s faster" claim is
   withdrawn** -- re-measured at the current commit the two arms were within noise (215 s versus
   217 s on evaluation). The P2 case never needed the timing claim and now rests where it should,
   on the 0-solve delta alone.

2. **`_fit_colormap` / `_apply_map` (`:713-731`) -- WITHDRAWN, not retired.** The claim was that
   the general induction subsumes it. It does not: `_fit_colormap` is a whole-grid closer and
   nothing in the widened alphabet ports that behaviour. It stays.

3. **The hand-assigned MDL integers -- WITHDRAWN for now, and named as surviving P1 offenders.**
   `_STRUCT_RECORDS`' `mdl` column (2, 3, 4) and `3 + (2 if memo else 0)` in `mint_records` are
   still hardcoded. Replacing them changes pool ORDER, which would confound the one thing on
   trial in the same commit. That is a scheduling reason, not an exemption -- P1 still owes this,
   and section 3.5 below now records it as a surviving offender rather than a completed cleanup.
   `SELECTOR_LITERAL_CAP = 4` is declared alongside it as a resource guard, and the descriptor
   alphabet itself remains a developer-authored boundary -- charged against the 19 of 400, not
   excused by it.

**The retirement this section got most wrong is one it did not list: `_RECOLOR_UNITS`.** Section
3.2 above argues the induced conjunction widens the authored 8-entry enumeration, with the
implication that the authored tuples become dead weight. **Measured false.** Retiring the six
object tuples lost clean evaluation ID `ae58858e` -- an intermediate run scored 18 of 400 with
1 gained and 1 lost, and the net-zero total is exactly the shape that would have read as "no
regression" if only the count had been checked instead of the ID set.

The cause is structural, not a tuning miss. `induce_selector` returns `None` on that task's
3-of-7, 3-of-9, 2-of-6 and 1-of-6 train splits, because the all-units path absorbs UNCHANGED
units as observations mapping a value to the colour it already has -- an identity mapping that
needs no selector at all. Neither mechanism contains the other. The tuples are restored and the
counterexample is pinned as a synthetic regression test, so the next attempt to "simplify" this
axis fails loudly instead of quietly costing a solve.

### 3.5 P1 audit

| Quantity | Derivation | Status |
|---|---|---|
| Selector stopping rule | strict improvement in misclassified cells | clean -- no threshold |
| Value rule form | existing MDL order plus leave-one-out | clean -- inherited discipline |
| Minimum contributing tasks | at least 2 -- recurrence is not definable on 1 | clean -- definitional |
| Admission bar | held-out solves, nodes-to-first-solve, `coverage_proven` movement | clean -- all measured, all on the TRAINING split |
| Record ordering | ledger rank; fallback = counted free parameters | **NOT CLEAN as built** -- the hand-assigned MDL integers SURVIVE; see 3.4 item 3 |
| Retirement bar | population median proposals-per-successful-record | clean -- a median of counts |
| Equivariance type | the largest subgroup under which the record commutes | clean -- computed, not chosen |

**Declared offenders:**

* **The descriptor alphabet F.** Assembled by a developer from `arc_scene`'s 22 public
  measurements. This is the honest residue of the whole design: it moves the authored boundary
  FROM primitives TO measurements and grows compositionally above it. It does not remove the
  boundary. That prior stays charged against the score, exactly as `DEPTH_CEILINGS.md` charges
  the current DSL.
* **The vocabulary cap. NOT DERIVED.** MINT's inequality gives M <= 0 at the stated operating
  point (0.1 above). The honest position: the cap is a DECLARED RESOURCE GUARD in the same class
  as `node_budget` and `time_budget`, and its effect is reported as a CURVE -- solves versus pool
  size -- so no single value arbitrates the verdict. If the cap ever changes which primitive
  wins rather than how large the pool is, it is an arbiter and must be re-derived.
* **The selector literal bound L.** Same class: a resource guard, reported as a curve.
* **Frozen and never to be raised:** `node_budget = 4000`, `time_budget = 5.0`, `attempts = 2`,
  `max_depth = 3`, `_MAXS = 30`. `verdict.md` already records raising the budget as the forbidden
  fix. A gain bought with compute is not a gain.
* **Not touched, and named so nobody thinks they were cleaned:** `SchemaInducer`'s
  `vigilance = 0.7`, `alpha = 0.5` (the ART choice-function tie-breaker, an arbiter on that
  module's decision path), `min_support = 2`, `replay_passes = 2`, `k_rel = 16`, `k_tail = 16`,
  `k_head = 8`, `max_prototypes = 256`; the whole R-STDP constant block; `arc_scene`'s internal
  constants, which are unaudited.

**Net: 4 arbitrating literals retired (the three MDL integers plus `ArcLibrary`'s `cap`), 0
introduced as arbiters, 2 declared as resource guards reported as curves, and one authored
alphabet named as the residue.**

### 3.6 Falsifiers and control arms

**Primary, and it can go red.** ARC-AGI-1 EVALUATION solves through `arc_solver.make_solver()` at
frozen budgets, `tools/arc_score.py` semantics, currently **18 of 400**. **Red at <= 18 of 400.**

A bare increase is not sufficient, and the consequence of a qualified red is pre-registered.
4 of the current 6 evaluation wins are memo-flagged and 3 of 6 sit inside the contaminated
20-task `scratchpad/oracle_dev.json` dev subset. The mechanism counts as having invented
something only if **at least 1 of the new solves** is earned by a record that is (a) a FUNCTIONAL
rule form, (b) carries at least 1 induced selector literal, and (c) is OUTSIDE the dev subset.
Contaminated and clean counts are reported separately. **If solves rise and 0 of the new solves
meet (a) and (b) and (c), the selector induction is RETIRED and only the ported `_fit_colormap`
equivalent survives.**

**Control arms, all mandatory:**

1. **Matched-vocabulary arm.** Hand-add the same number of authored primitives and re-run. A
   learned mechanism that loses to hand-written DSL operations has not earned its complexity.
2. **Pool-order arm.** Shuffle the pool order at the same pool size. Separates "the new primitive
   solved it" from "the pool order changed and a different program won first". At depth 3 the
   node count scales with the cube of the pool size, so a mechanism that adds 2 solves while
   tripling the branching factor can lose solves net at a fixed budget. Pool size and
   nodes-to-first-solve are reported alongside every result.
3. **Label-shuffled selector arm.** Induce selectors from label-shuffled residuals. The admitted
   count must be near zero. If junk selectors admit at the same rate, the admission bar is not a
   bar.
4. **Coverage arm.** `arc_diagnostic`'s `coverage_proven` count must strictly FALL. Solves up
   with coverage flat means ranking or luck, not vocabulary. This bin is budget-invariant, so it
   cannot be gamed by raising the budget.
5. **Node-efficiency arm.** Nodes-to-first-solve on tasks solved in BOTH arms, plus the count of
   tasks hitting the budget cap. This replaces MINT's "total nodes must not rise", which cannot
   rise because nodes are capped per task.
6. **Retirement precision/recall.** Inject `k` records induced from label-shuffled residuals;
   require retirement of most of them within a matched proposal count AND retirement of 0 of the
   records that earned solves. This replaces MINT's "at least one record must be retired", which
   was satisfiable by loosening admission.

**The trap, named in advance and adopted verbatim in spirit.** The most likely outcome is that
minted-record counts rise beautifully, admitted selectors rise beautifully, and evaluation stays
at 18 of 400. That is `engine/autonomous_loop.py:969` again. **Mint count, admitted-record
count, and selector-length distribution are TELEMETRY, not outcome, and none may appear in a
success claim.**

---

## 4. A null is a successful outcome

This is stated as a section rather than a clause because both designs are more likely to return
red than green, and reporting the red is the deliverable.

* **If step 0 returns rank 1** -- faculty activity is not differential, so no (speaker x
  listener) object on this workspace can exceed rank 1 -- then the repo learns something it does
  not currently know: the 11 private workspaces and the rank-1 bus are not merely
  under-connected, there is **nothing differential to connect**, and the P6 bandwidth program is
  chasing bandwidth for an address that does not exist. That is a genuine architecture-level
  negative and it costs one afternoon.
* **If rho stays 0 after the change**, the payload is still consensus and the change is cosmetic.
  Report it.
* **If the dissociation fails to cross** (N2 and N3 both move, or neither), then the bus is
  carrying a bigger number and not an address. The module is deleted.
* **If the depth-1 survivor census shows the budget never binds**, MINT's cost model was
  decoration and the depth <= 1 capture shell has no justification. Report it and drop the shell
  restriction.
* **If the residual is defined on only a small fraction of depth <= 1 states**, that fraction is
  the mechanism's ceiling and it should be published before the descriptor pipeline is written.
* **If evaluation stays at 18 of 400** after admission, the vocabulary hypothesis has been tested
  on this substrate and refused, the same way the variable-depth hypothesis was tested and
  refused (5699 converged trajectories at depths 3 to 19 won 0 of 12 solves). That is a result.

A red result here is more informative than a green one, and it costs the same to obtain.

---

## 5. What this CANNOT establish

**Nobody may finish this document believing the brain has been made conscious. Nothing in either
design is evidence of consciousness, and neither design is intended to produce any.**

* **Not consciousness, not phenomenality, not experience.** Every quantity Design A produces is a
  statement about a joint distribution over outputs. `rho` is a ratio of singular values of a
  matrix of at most 11 x 11 integer-derived payloads. `bias_k` is a frequency of two counts. A
  dissociation between "which listener is addressed" and "how large the number is" is a
  functional fact about a message bus. It is not a claim about anything being like anything.
* **Not Phi, not an approximation of Phi, not comparable to Phi, not comparable to PCI 0.41.**
  PCI does not appear anywhere in either design, deliberately: the one forbidden move across the
  four proposals was setting a live parameter to the argmax of the complexity index, and the way
  to be sure that move is not made is to have no PCI in the design at all. No number produced
  here may be printed beside PCI 0.41 in a way that invites a conflation.
* **Nothing is optimised toward any marker.** The ledger is a count. It is not tuned, and there
  is no parameter to tune. Design B is admitted on training-split solves and evaluated once.
* **It does not fix the binding marker.** *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).* `reasoning/consciousness_probe.probe_binding`'s
  `present = sp > rt + 0.15` (`:84`) is untouched, and its 1-of-26 insensitivity is untouched.
  Fixing the availability payload does not fix binding, and saying otherwise would be exactly the
  marker-conflation this project forbids.
* **It does not create an input-free interval, and working-memory attractors still cannot
  charge.** This is declared OUT OF SCOPE with a reason: every gate proposed across the four was
  either a rhythm importing five constants (`MatsuokaCPG`, which additionally imports the SO-101
  arm at module level in `faculties/cpg_oscillator.py:17-18`) or a zero-threshold comparison with
  no hysteresis and a modal gap of about 1 tick. Naming the blocker as unaddressed is more honest
  than shipping a gate that fires often and changes nothing.
* **It does not move `tools/p6_gate.py`'s shipped static count.** That gate counts read SITES by
  AST walk. Design A adds no read site; it changes what one existing read returns. **The shipped
  gate will report the same static number before and after.** A green on the new dynamic
  permutation metric must never be reported as movement on the shipped one, and the real
  denominator is 2 ordered pairs of which 1 is measurable -- not 110.
* **Design B is not a depth result and not fluid intelligence in Chollet's developer-aware
  sense.** The descriptor alphabet is authored. The mechanism moves the authored boundary from
  primitives to measurements; it does not remove it.
* **Design B is not a fix for memorisation.** Refuse-on-unseen plus leave-one-out plus
  equivariance typing raise the bar; they do not prove generalisation. With 4 of 6 current wins
  memo-flagged, the correct prior is that the next win is memo-flagged too.
* **Not comparable to ARC-AGI-2 or ARC-AGI-3 or to any LLM-harness number.** Different
  instrument, different axis. Never place them beside 18 of 400.
* **Neither arm's result may be summed with or substituted for the other's.** A green on Design B
  is evidence about vocabulary and says nothing about the bus. A green on Design A is evidence
  about addressing and says nothing about ARC.
* **A green on both does not establish that they are one object.** The unification is a separate
  question with a separate test (section 6, step 8), and if that test passes while both arms have
  gone red, the honest conclusion is that the unification is real and useless.

---

## 6. Build order -- cheapest falsifiable step first

Each step can cancel what follows it, and cancellation is a legitimate outcome. Nothing below
step 3 is written until step 0 and step 2 have both returned.

**Step 0 -- the differential-activity census. Minutes. No code on a live path. Can cancel Design
A entirely.**
Over at least 300 wake ticks, log `(sources(winner), set of submitters this tick)` -- both already
computed and discarded in `ReentrantWorkspace.step()`. Build the (speaker x listener) coincidence
matrix and report its rank and the per-faculty activity fractions as "N of M". **Predicted red:
`brain.py:1746-1757` submits all three sources inside one `try` block on every tick, so the
matrix is rank 1 today.** If red, the fallback is to make submission conditional on a faculty
having something to say -- which changes `base`, `sources`, and the coalition bonus, therefore
changes which coalition wins, therefore requires a full re-baseline of winner distribution,
availability, `access_strength`, and the `charge_probe` reference. That re-baseline is a
deliverable in its own right and must be reported before any Design A arm is interpreted.

**Step 1 -- extract rho into `tools/p6_gate.py`. Minutes. Independent of everything.**
Report `rho` with its provable-zero baseline and the invariance statement. It is an instrument,
not a mechanism, and it is worth shipping even if both designs die. Add nothing to the static
read-site count.

**Step 2 -- the residual census. An afternoon. Can cancel Design B's shell restriction and its
cost model.**
Two numbers, both from instrumenting `synthesize` across the 400 training tasks: (i) the
distribution of `len(nxt)` after depth 1, which decides whether the node budget binds at all and
therefore whether the depth <= 1 capture shell has any justification; (ii) the fraction of depth
<= 1 states that admit a residual at all -- `_fit_colormap` returns `None` immediately on a shape
mismatch (`:718-719`), so crop, scale, unscale, tile, half, split and materialize prefixes have
no residual. That fraction is the mechanism's ceiling.

**Step 3 -- the crc32 jitter fix and the action-clamped `charge_probe` arm.**
Both are prerequisites for interpreting any Design A measurement. `_JITTER` currently draws from
a shared RNG keyed on content arrival order; Design A makes content identity load-bearing.
`charge_probe` currently has no action clamp, and withholding or biasing anything changes the
selected action, which changes the next percept, which changes everything.

**Step 4 -- Design A, minimal.** Make the registered subscribers also submit; add the ledger;
rewrite `_deliver`; delete `_REENTRY_GAIN`, the `max()` collapse, and
`reached / len(self._subs)`. Run N0 through N5 and falsifiers 1 through 6.

**Step 5 -- Design B, minimal.** `arc_residual.py` with the versioned descriptor enumeration and
greedy selector induction; the `selector` field and the `compile_record` branch; the offline
snapshot admission process; `ArcLibrary` deleted. Admission on the training split only.

**Step 6 -- Design B's control arms**, especially the matched-vocabulary arm, BEFORE evaluation
is run. Evaluation is run once, at the end, with the record set frozen, and `H` is reported.

**Step 7 -- report.** Two verdicts, side by side, never summed. Contaminated and clean counts
separately. Telemetry quantities in a clearly labelled telemetry section that no success claim
may reference.

**Step 8 -- only if both arms return green: the unification test.** Over real corpus terms nobody
generated with `fill`: for every pair `(t1, t2)`, assert
`fill(generalize(t1, t2), keys_of(t1)) == t1` and likewise for `t2`. Then the cross test: a
skeleton induced on the Gf side must be fillable by the bus-side `fill` and validate under the
bus-side predicate, and the reverse. If the two sides need different `fill` implementations in
practice, the unification is nominal and must be reported as such.

---

## 7. Files touched

**Design A**

* `src/snn_data_lake/faculties/reentrant_workspace.py` -- `_ledger` and `_tsup` state; `_deliver`
  rewritten and given the `sources` argument; `_REENTRY_GAIN` deleted; the `max()` aggregate
  deleted; `reached / len(self._subs)` deleted; `step()` records the tally before
  `self._buffer = []`. `submit`, `subscribe`, `broadcast_to`, `association`, `access_strength`
  keep their signatures.
* `src/snn_data_lake/engine/brain.py` -- the registered faculties must both `submit()` and
  `subscribe()`; `_reentrant_availability` at `:1758` becomes the ledger-derived report-only
  value; the comment at `:713-725` must be UPDATED and must not be made false again -- it is
  currently correct.
* `tools/p6_gate.py` -- new `rho` section and a new DYNAMIC permutation section. The static
  read-site count is not touched and does not move.
* `tools/charge_probe.py` -- the action-clamped arm (step 3).
* new `tests/test_ledger_rank.py`.

**Design B**

* new `src/snn_data_lake/reasoning/arc_residual.py` -- descriptor enumeration, residual
  descriptor construction, greedy selector induction. The only new module.
* `src/snn_data_lake/reasoning/arc_mint.py` -- one `selector` field on `TypedPrimitiveRecord`
  (`:512`), one branch in `compile_record` (`:563`), the versioned descriptor enumeration,
  counted free parameters replacing the hand-assigned `mdl` integers.
* `src/snn_data_lake/reasoning/arc_solver.py` -- residual sink in `close()` (`:822`); store-fed
  `_pool`; **`ArcLibrary` deleted** with its pool branch and `make_solver`'s `library=`
  parameter.
* new `tools/residual_census.py` -- step 2's two numbers.
* new `tests/test_arc_residual.py`.
* **NOT touched:** `engine/brain.py:_enter_sleep` -- it is on the tick. The offline pass is a
  separate process against a snapshot.

**Reused unchanged:** `reasoning/arc_scene.py`, `reasoning/arc_diagnostic.py`,
`reasoning/arc_bench.py`, `tools/arc_score.py`, `tools/arc_census.py`,
`substrate/lake_memory.py`.

**Explicitly not built, and why:** any resonator or theta-slot register (already exists in
`faculties/theta_gamma_code.py` and `growth/autapse_neuron.py`); any PCI sweep (optimising toward a
marker, and the harness does not exist); any VSA bus layer (cold-start identity collapse and
`D = 512` capacity); any FlyHash or ART clustering layer (zero importers, degenerate on the
descriptor, duplicates `SchemaInducer`, and its null could not fail); any R-STDP credit layer for
record ordering (its null could not fail); any afference gate (no P1-clean generator and a modal
gap of about 1 tick); any `burst_multiplexing` head-to-head (category error).
