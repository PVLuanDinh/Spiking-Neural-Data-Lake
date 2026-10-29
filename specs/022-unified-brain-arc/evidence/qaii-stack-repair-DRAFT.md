# Q-AII stack repair: make the canonical learner learn A_ii without relaxing its stability (DRAFT pre-registration)

**Status: DRAFT, design only.** No `src/` code. Written by the cloud session (`session_01TeA2yYimjAiHiGFzw6hqHX`) on
the SNDL launcher's instruction of 2026-10-29, at `origin/main` `ee224e50`.
- **It decides nothing.** The integrator owns the final pre-registration, the row (if the owner makes this one), every
  run, the panel and the battery. The policy-enforcer reviews it.
- **Why it exists.** It is the first link of the G9 path, and it has no tasks row
  (`research/g9-critical-path-1029.md` section 4 step 0, on `cloud/g9-critical-path-1029`).
- **Measurements in this file.** Six cells on exposed dev seed 8000 reproduce the recorded attribution and time a
  cell (section 1.4). Everything else not marked MEASURED is a projection.

## 0. The owner's ruling this answers

Quoted from `evidence/g2-taught-problems.md` (the 2026-10-13 note, integrator `claude-cont-1013`, lines 650-652). The
board items themselves are not visible to this session:

> Q-AII is ANSWERED by the owner (board #2134/#2135, 2026-10-13, "Fix the stack first"): A_ii and (ii) stay parked until
> a separately pre-registered repair of the E/I balance x Benna-Fusi consolidation x homeostatic scaling interaction
> makes the CANONICAL brain learn A_ii (measured, reachable red, no safety relaxation); (ii) then runs on the repaired
> brain under this file's frozen 2.9 reading.

**Three binding words follow from it:**
- **CANONICAL:** the learner as the live brain builds it.
- **Measured, reachable red.**
- **No safety relaxation:** switching a mechanism off is not a repair.

## 1. Territory (at `ee224e50`)

### 1.1 What A_ii is

- **Definition:** `instruments/delayed_cue.py:235-247`. Four texts over the words `ram`, `cup`, `emu`, `cod` (buckets 15,
  16, 20, 26, disjoint from (i) and (ii)):
  - `ram cup emu cod`
  - `ram cup cod emu`
  - `cup ram emu cod`
  - `cup ram cod emu`
- **Labels:** "GREET iff 'ram' precedes 'cup'" (`label_aii`, u == 0); otherwise GOAL.
- **Its role:** it is (ii)'s word-order construction on disjoint words, labelled separably. It is G3's curriculum task A
  for B = (ii) (`g2-taught-problems.md` 2.2, lines 99-102). By registration the incumbent must solve it ("the incumbent
  solves `A_ii` on every seed, B5's form").
- **The solved predicate and stability:** `taught_problems.STABLE` (`taught_problems.py:49`) means solved at each of the
  last 10 epoch-end reads. The harness is `taught_campaign.attempt` (`taught_campaign.py:211`), driven by
  `taught_campaign.run_arm` (`:309`), with `EPOCHS = 400` (`:47`).

### 1.2 The canonical learner and the stack

- **The factory:** `taught_problems.canonical_incumbent` (`taught_problems.py:386-398`) builds
  `SpikingInteractionLearner(_IXN_INTENTS, n_in=N_IN, seed, t_post=T_POST)` and sets
  `unit.stack = PlasticityRuleStack(n_neurons=9, seed=seed)`.
- **It is the live brain's learner.** `taught_campaign.py:62-64` names it "the ONE factory (P2); B8 below proves it equal
  to the Brain's learner". The live wire is `src/snn_data_lake/engine/brain.py:353-379`: `self.brain._ixn.stack = PlasticityRuleStack(...)`.
- **The stack's measured justification,** `src/snn_data_lake/engine/brain.py:360-363`, quoted: "unstabilized R-STDP saturates -- peak drive 18.97,
  max|w| 6.64 -- and becomes RIGID (it then fails to learn conflicting material at all). With the stack: drive 3.01,
  max|w| 0.98 ... it did NOT improve held-out intent accuracy (5/14 either way), so it is wired for stability, not for
  recognition quality."
- **Where the learner consumes the stack:** `readout/interaction_learning.py:534-560`. PRE: `d *= stack.pre_gain(...)`.
  POST: `n.w = stack.stabilize(...)`.

**The stack's six switches.** `PlasticityRuleStack.__init__`, `plasticity_rule_stack.py:266-308`, plus the base
`HomeostaticPlasticityStack.__init__`, `homeostatic_plasticity_stack.py:181-212`:

| switch | class, file:line | phase | what it does to a weight |
|---|---|---|---|
| `metaplastic` | `_BCMMetaGate`, `plasticity_rule_stack.py:87-148` | PRE | scales the deposit gain into [floor 0.3, 1.0] by the BCM theta |
| `short_term` | `_ShortTermGain`, `homeostatic_plasticity_stack.py:78-104` | PRE | Tsodyks-Markram gain on the deposit |
| **`homeostatic`** | `_HomeostaticScaling`, `homeostatic_plasticity_stack.py:106-140` | POST | multiplicative scaling only when drive exceeds a ceiling set ONCE at `ceiling_factor` 3.0 x the neuron's first-seen drive (`:116-138`) |
| `intrinsic` | `_IntrinsicPlasticity`, `homeostatic_plasticity_stack.py:142-162` | POST | tunes gain and bias; weights unchanged |
| **`ei`** | `_EIBalance`, `plasticity_rule_stack.py:150-189` | POST | every step, `w *= 1 - min(max_trim 0.05, k 0.5 x inhibition)` (`:177-189`); the iSTDP controller's inhibition grows while a drive proxy exceeds `rho0` 0.05 |
| **`consolidation`** | `_BennaFusiConsolidation`, `plasticity_rule_stack.py:194-255` | POST | each step, deposits the weight CHANGE into a 6-beaker cascade initialised at equilibrium on the first-seen weight, diffuses once, and returns the fast beaker (`:217-240`), so a change seeps toward an anchor set at initialisation |

**The POST order:** base (homeostatic, then intrinsic), then E/I, then Benna-Fusi, then BCM observe
(`stabilize`, `:318-328`). The `learning_step` form is `:383-403`.

### 1.3 How A_ii fails today (recorded)

From `g2-taught-problems.md` 2.17 and 2.17.1 (dev seeds 8000-8003, lines 534 and 558-575):
- **The canonical incumbent never stable-solves A_ii at E 400.** The per-epoch accuracy is mostly 0.5, with isolated 1.0
  reads; means 0.491 / 0.492 / 0.497 / 0.491.
- **The bare learner** (no stack) stabilises at 118 / 133 / 140 / 141 epochs.
- **The six-switch attribution** (dev seeds, no code changed):
  - "the E/I balance, the Benna-Fusi consolidation and the homeostatic scaling each block alone (only-one-on), no single
    or pair switched off restores it, and the three together off restore it (174 / 156 / 179 / 141)";
  - "The metaplastic gate and intrinsic plasticity do not block; short-term gain only slows it."
- **The confirmatory campaign** (2026-10-07, seeds 8200-8207, read once, G2 5.1) read `inc.aii 0/8`.

**The shape this implies (my reading):** an **OR of three sufficient blockers**, not an interaction in the
statistical sense. Each of the three alone stops A_ii, and only removing all three restores it. The owner's word
"interaction" is kept as the ruling's wording.

### 1.4 Reproduced here (MEASURED, dev seed 8000 only, already exposed)

- Run through `taught_campaign.run_arm(("x", "aii", "incumbent", "solves", False), 8000, epochs=400)`.
- The switch set was varied by wrapping `PlasticityRuleStack`'s keyword arguments in a scratch script. No repository code
  was changed.
- Stdlib, CPython 3.14.6t, Linux.

| configuration | epochs run | stable | mean accuracy | wall-clock |
|---|---|---|---|---|
| canonical (all six on) | 400 | no | 0.491 | 1.3 s |
| E/I, Benna-Fusi and homeostatic off ("off3") | 174 | yes (to_stable 164) | 0.674 | 0.3 s |
| bare (no stack) | 118 | yes | 0.602 | 0.2 s |
| only E/I on (of the three) | 400 | no | 0.497 | 1.1 s |
| only Benna-Fusi on | 400 | no | 0.516 | 1.5 s |
| only homeostatic on | 400 | no | 0.572 | 0.8 s |

**This reproduces the recorded numbers exactly** (0.491, 174, 118) and the only-one-on blocks. **One cell costs about 1
second.**

### 1.5 What must not move (the safety and regression floor)

**The repair must keep every one of these:**
1. **Stability:** `plasticity_rule_stack._stability_selfcheck` (`:557-611`).
   - "the superset must keep the synapses BOUNDED (drive < 20.0)";
   - "the unstabilized control must DIVERGE past the bounded rule (c_drive > 3 x s_drive)";
   - rate homeostasis within 0.08 of mu.
2. **Retention:** `_retention_selfcheck` (`:464-517`).
   - The rule stack learns A and the new B (> 0.9 each);
   - it retains A at the measured horizon (>= 0.80);
   - it retains A at least 0.30 above the base stack.

   This is Benna-Fusi's anti-forgetting guarantee, and **the core tension**: the anchor that retains A is the anchor that
   slows learning A_ii.
3. **Metaplasticity:** `_adaptive_lr_selfcheck` (`:519-555`): the gate stays in [0.3, 1.0], with the cold start at 1.0.
4. **Equivalence and determinism:** `_equivalence_and_determinism_selfcheck` (`:613-669`). With the extras off, the stack
   is byte-identical to the base; the same seed gives a byte-identical trajectory.
5. **Live charge:** `_tick_charge_selfcheck` (`:671-788`): each stack rule CHARGED on the real host path, iSTDP's
   INERT control kept.
6. **The brain-level stability claim** at `src/snn_data_lake/engine/brain.py:360-363` (bounded drive and |w| on the live path, still able to learn
   conflicting material). **This is not currently an asserted test.** The integrator should decide whether the repair
   arc first turns it into one (owner question 4).
7. **G2's other cells:** the (i) cells, A_c1d and C1-d dispositions in G2 5.1 must be unchanged ("The (i) cells, C1-d and
   A_c1d do not depend on it", 2.17.1).
8. **Persistence:** the learner's persisted state. G0(c) checkpoints the stack's lazily built classes with fixed
   attribute sets (`interaction_learning.py:140-147`, including `benna_fusi.BennaFusiSynapse`). A repair that changes a
   persisted attribute set owes its migration disposition in the same commit (tasks.md discipline). R2's replay
   hydration and sleep's `relax` consume the same stack, so their tests are in the floor.

## 2. Hypotheses

**H-shared, the timescale hypothesis (primary).** All three blockers contract the learning neuron's weights at the
per-step timescale:
- the E/I trim, up to 5% a step;
- the homeostatic ceiling, frozen at 3x the FIRST-seen drive;
- Benna-Fusi's diffusion toward an anchor fixed at initialisation.

A_ii is slow to learn: the bare learner needs 118-141 epochs. Its per-epoch weight gain is smaller than any one
blocker's per-epoch contraction, so the discriminating weights never accumulate.

**What H-shared predicts in the pilot** (section 3):
- **(a)** For each blocker alone, there is a strength threshold below which A_ii stabilises.
- **(b)** That threshold moves with the learning rate (a larger lr tolerates a stronger blocker).
- **(c)** Re-referencing each blocker to a running baseline, or moving it to a slower clock, restores A_ii while every
  item of section 1.5 holds. Candidates for (c):
  - E/I trim gated on SUSTAINED above-target rate rather than per step;
  - the homeostatic ceiling tracking a running baseline rather than the first-seen drive;
  - Benna-Fusi's anchor moving at the deep-beaker rate rather than staying at the initial weight.

**The per-mechanism alternatives, each refuting H-shared for that mechanism if it holds:**
- **H-EI:** the trim's drive proxy (`exc = 0.15 + 0.02 x drive`, `:181`) saturates the iSTDP controller on A_ii's
  inputs whatever the timescale.
- **H-HS:** A_ii's solution needs a winning drive above 3x the first-seen drive, so any fixed ceiling factor blocks it.
- **H-BF:** alternating GREET / GOAL deposits on shared words cancel in the deep store, so the fast beaker returns to the
  anchor whatever the coupling.

**The null:** no setting of the three mechanisms inside the section 1.5 floor makes the canonical learner stable-solve
A_ii at the registered E. Section 6 disposes of it.

**P2.** Every candidate edits the incumbent class in place (`_EIBalance`, `_HomeostaticScaling`,
`_BennaFusiConsolidation`). No second stack, no new mechanism, and no switch turned off.

## 3. The pilot that bounds the cost (dev seeds only; it decides the candidate, never the claim)

**Seeds.**
- G2's exposed development seeds 8000-8003.
- If the integrator allocates them, more from G2's development range 8004-8099 (`g2-taught-problems.md:267`), checked
  against the board's seed ledger, which this session cannot see.
- No confirmatory or reserve seed is read.

**The cells.** Every cell is one `run_arm` on A_ii at E 400, through the canonical factory with one parameter changed.

| block | cells | count |
|---|---|---|
| P0, the reproduction | canonical, off3, bare (the section 1.4 cells) x 4 seeds | 12 |
| P1, dose per blocker | only-that-one-on, at 5 strengths each, including production and 0: `ei_max_trim` {0.05, 0.02, 0.01, 0.005, 0}; `ceiling_factor` {3, 5, 10, 20, inf}; Benna-Fusi via `bf_n_levels` {6, 4, 2, 1} and a deposit-scale grid the integrator fixes before the run | (5 + 5 + 5) x 4 seeds = 60 |
| P2, lr interaction | canonical at lr x {1, 2, 4} | 12 |
| P3, candidate (c) shapes | 3 candidates (one per blocker) and their union, each in the full canonical stack | 4 x 4 seeds = 16 |
| P4, the floor | for every P3 candidate, the five `plasticity_rule_stack` self-check functions of section 1.5 (items 1-5) with that candidate's parameters | seconds each, deterministic |

**Size and wall-clock.**
- About 100 learner cells at a measured 0.2-1.5 s each (section 1.4): **about 2-3 minutes of CPU on stdlib**, plus the
  P4 self-checks.
- **The pilot is cheap. Its cost is the build (an instrument that varies the stack through declared parameters, its
  reds, a panel), not compute.**
- **Projection:** one arc to build the pilot instrument with its reds and run it; one arc to fold its reading into this
  pre-registration and freeze it.

**What the pilot outputs** (not a claim): the feasible region where A_ii is stable on all dev seeds AND every P4 check
passes. The candidate it names is frozen into section 4 before any confirmatory seed. **If the region is empty,** the
confirmatory campaign does not run; the record goes to the owner with the pilot's map (section 6).

## 4. The confirmatory campaign (the claim)

**Arms, paired on the same seeds:**
- **CANON:** today's canonical learner.
- **REPAIRED:** the canonical learner with the frozen candidate.

**Primary endpoint (G2's own predicate, unchanged):** stable-solves A_ii within E = 400 (`STABLE`: solved at each of the
last 10 epoch-end validation reads). It is binary per seed.

**Co-primary (intersection-union; both must hold, so no alpha split):**
- (1) REPAIRED beats CANON on the primary endpoint (section 5's test);
- (2) every section 1.5 floor item passes with the REPAIRED parameters (deterministic checks, each pass or fail).

**Secondary, descriptive:** epochs-to-stable on the solving seeds, against off3's dev values (174 / 156 / 179 / 141).

### 4.1 The red that must fire on the base

On the confirmatory seeds, CANON fails the primary endpoint. This is reachable and recorded: 0 of 4 dev seeds, 0 of 8
confirmatory seeds 8200-8207, and seed 8000 re-measured here.
- **If CANON solves A_ii on a majority of the confirmatory seeds,** the premise is false. The campaign reports it as such
  and claims nothing.

### 4.2 Positive controls (not the method)

- **Learnability instrument:** off3 and bare stable-solve A_ii on every pilot seed (MEASURED on 8000; recorded on
  8000-8003). This proves the harness can read a solve on these seeds.
  - They are never repairs: they switch mechanisms off.
  - **off3 is the instrument's "a solve is possible" control. If REPAIRED equals off3's behaviour, the floor checks
    (item 2 above) decide whether it is a repair or a relaxation.**
- **The stability instrument's own red:** `_stability_selfcheck`'s unstabilized control must diverge (c_drive > 3 x
  s_drive) in the same run as REPAIRED's bounded drive. If the control fails to diverge, the stability reading is void.
- **lr = 0** (G2's A_c1d lr-0 control, 2.17.1, recorded): an untaught learner fails. This rules out a harness that
  "solves" without learning.

## 5. Sample size (P1)

**Measured inputs:**
- CANON solves A_ii on 0 of 12 seeds (dev 0/4; confirmatory 0/8).
- off3 and bare solve on 4 of 4.
- off3's epochs-to-stable SD is 17.4 epochs (174 / 156 / 179 / 141; descriptive only).

**The test.** A one-sided exact sign test on discordant pairs (REPAIRED solves and CANON fails, against the reverse).

**The claimed effect:** REPAIRED solves on at least 75% of seeds. With CANON failing everywhere, a discordant pair is a
REPAIRED solve.

**The fixed-sample target** (computed exactly): at q = 0.75, n = 8 gives power 0.886 at one-sided alpha .05. For
reference, q = 0.6 needs n = 10 and q = 0.9 needs n = 6.

**The sequential design** (P1 item 2):
- **Looks:** at n = 8 and n = 16.
- **Boundary:** a per-look one-sided alpha of 0.03 (Bonferroni-style split across the two looks). That means 8 of 8
  discordant favourable at look 1 (7 of 7 if one seed is concordant), and 13 of 16 at look 2.
- **Futility, non-binding, at look 1 only:** 3 or fewer REPAIRED solves of 8. Look 1 is the fixed-sample target, so its
  mde is the claimed effect, as P1 requires.
- **Cap:** 16, an inflation factor of 2 over the fixed target. The discrete test needs n = 10 for q = 0.6, so the cap
  keeps power at least .8 for an effect somewhat smaller than claimed.

**Simulation** (200,000 runs each, written for this draft):

| planted truth | cross at 8 | cross at 16 | futility at 8 | cap |
|---|---|---|---|---|
| null, discordance 20% each way | 0.0008 | 0.0035 | 0.944 | 0.052 |
| null, discordance 50% each way | 0.004 | 0.009 | 0.363 | 0.624 |
| q = 0.75 (claimed) | 0.677 | 0.296 | 0.027 | 0.000 |
| q = 0.6 | 0.315 | 0.508 | 0.175 | 0.002 |

- The false-positive rate is at most 0.013 under both nulls.
- Power at the claim is 0.973.
- A futility stop or the cap without a crossing is INCONCLUSIVE unless the mde at that count is at or below the claim
  (P1 item 2 and P8 item 4).

**Multiplicity.** The co-primary floor checks are deterministic pass-or-fail, so they take no alpha.

**Seeds.** The confirmatory and reserve block is allocated by the integrator from the board ledger. It is disjoint from
8000-8099 and from 8200-8207 (already read once).

## 6. Null disposition (P8, including the 2026-10-23 null-scale amendment)

**What a null would be about.** A null here is a null on the REPAIR CANDIDATE, not on any mechanism.
- Nothing in the stack is removed or disabled on any outcome. The three mechanisms are wired for stability
  (`src/snn_data_lake/engine/brain.py:360-363`), and the owner ruled out safety relaxation.
- The candidate's code is reverted. It was never live: it lands only on a crossing.

**The scale.** Every null names it. The axis is **epochs**: a learning rule scales with ticks (P8 amendment item 2), and
A_ii is a slow-learning task.
- The record states the units (9 readout neurons), the epochs (400), the seeds, the mde and the re-test state, in the
  format the amendment's section 0 fixes.
- **The 10x re-test is the same arms and predicate at E = 4,000.** G2 already ran its ii.twin read at 4,000 epochs
  (2.17.1), so the harness supports it. A second null at 4,000 closes the candidate.

**Two other outcomes:**
- **Pilot region empty** (no candidate passes the floor and solves the dev seeds): this is not a campaign null.
  - The pilot map goes to the owner, with the Q-AII options G2 recorded (owner question 1).
  - A_ii and (ii) stay parked.
- **Underpowered** (cap reached, mde above the claim): INCONCLUSIVE. It licenses nothing.

**What this does NOT claim on any outcome:**
- anything about (ii), growth, the reservoir or the engine (G2 5.1's own sentence);
- that the stack helps recognition (`src/snn_data_lake/engine/brain.py:363`: it does not);
- that a repair at the learner level holds in the brain without the brain-level checks of section 7.

## 7. The production-tier question

**On stdlib (here or anywhere):**
- the whole learner-level programme: pilot P0-P4, the confirmatory campaign and the floor self-checks.
- The incumbent A_ii path is pure Python: `taught_campaign`, `taught_problems`, `delayed_cue`, `inseparable_mapping`,
  `interaction_learning` and the stack import no torch on that path. torch enters `taught_campaign` only in the presized
  control (`:125-138`), which this programme does not use.
- The perception tier changes the brain's percept front-end, not this learner or its taught front-end.
- **MEASURED:** the stdlib numbers in section 1.4 equal G2 2.17.1's recorded numbers (0.491, 174, 118). 2.17.1 does not
  name its host; G2's 5.1 campaign ran on the owner host (its `D:\` paths). This is consistent with tier independence on
  this path; it is not a cross-host proof.

**Only on the owner host, at the production tier:**
- **(a)** The brain-level consequences:
  - the live intent learner under the REPAIRED stack in a persisting scratch brain;
  - the bounded-drive and still-plastic claim of `src/snn_data_lake/engine/brain.py:360-363` (section 1.5 item 6);
  - G0(c)'s checkpoint round trip;
  - R2's replay hydration and sleep's `relax` on the repaired stack.
- **(b)** The battery.
- **(c)** G2's (ii) taught campaign afterwards, which the ruling licenses only on the repaired brain.

The brain's live front-end and its daemon cadence are the production tier's. The repair is not claimed for the live brain
until (a) passes there.

## 8. Order and ownership

- **Lands after the E1d campaign.** It changes brain code outside the pc harness (`plasticity/plasticity_rule_stack.py`
  and possibly `plasticity/homeostatic_plasticity_stack.py`). E1d may cite its calibration only while that code equals
  `1b6e53ab` (the launcher's ruling for the P4b row, which applies by the same reasoning; my inference).
- **Then:**
  1. this pre-registration, finalised by the integrator;
  2. the pilot instrument with its reds;
  3. the pilot reading folded and frozen;
  4. the red on the base;
  5. the build;
  6. the confirmatory campaign;
  7. the brain-level checks on the owner host;
  8. panel, fold and battery.
- **Rows it unblocks:** G2's (ii) (T110), then G3's retention reads (`g3-growth-loop.md` A6, "readable A_ii and #2135
  release").

## 9. For the owner and the integrator

1. **If the pilot region is empty, which Q-AII option applies?** G2's record keeps the original options; this session
   has not read them (board #2134/#2135).
2. Should this become a tasks row on the G9 path (the critical-path file's owner question 3)?
3. **Is the claimed effect right?** "at least 75% of seeds" is this draft's choice. G2's registered incumbent
   requirement is "solves on every seed", which is stricter. Choosing it would make the per-seed predicate 100% and the
   sign-test design above conservative.
4. **Turn `src/snn_data_lake/engine/brain.py:360-363` into an asserted test first?** It is a measured comment today. As an asserted test it would
   make section 1.5 item 6 a reachable red in the repair's own arc.
