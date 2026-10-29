# Representational drift and the SNDL engine: a research map

Prepared 2026-10-27 for Luan Dinh (owner). This is a research map, not a pre-registration. It proposes
no rule and claims no result. Every engine statement below cites a file or evidence record at
`6612ce7`. Line numbers go stale; the symbol names are the durable reference. Paths under `evidence/`
are in `specs/022-unified-brain-arc/evidence/`.

## 0. Bottom line

1. **Nothing in the tree measures representational drift.** I searched src/, tests/ and tools/ for
   population-vector correlation across sessions, tuning correlation, CKA, RSA, Procrustes,
   null-space and decoder-across-days. All searches came back empty. Existing uses of the word "drift" mean other
   things:
   - weight statistics moving off the H01 prior (`instruments/prior_drift_harness.py`, `DriftRecord`);
   - input distribution shift (`infra/drift_monitor.py`, `psi`).
2. **The engine was designed to hold representations still.**
   - Grown units' input rows are frozen after admission (`growth/grown_code.py`, `GrownUnit.w_in`).
   - The NG-SNN keeps inherited rows byte-identical (`growth/neurogenesis_engine.py`, grow-on-top).
   - The chat rank code is pinned (`tests/test_grown_code.py`, "C1: old codes unchanged").
   - The perception encoder is frozen offline e-prop.
   - The reservoir is a FIXED `KernelLsmPool` after its adaptation window
     (`reservoir/recurrent_pool.py`, `frozen()`).

   So the textbook setting, a hidden code that drifts under a stable readout, exists in only a handful
   of places (section 1).
3. **The engine's one measured drift pathology is exactly the literature's central problem.** In L1b,
   Oja and BCM rules moved the grown units' input rows. The readout, which the growth gate held stale,
   broke: OJA vs frozen rows gave t = -1.959, and seed 7 fell from 0.983 to 0.267 within one refit.
   The record's own conclusion, quoted verbatim (`evidence/l1b-synaptic-online.md:312`): *"the
   READOUT must co-adapt with any moving input"*. The rule was deleted. The control the record says is
   owed before that negative generalises, a re-solve-beta-on-every-refit control, has never been run.
   The drift literature has instruments for this question: null-space decomposition, self-healing
   readouts and latent realignment. The engine does not.
4. **The highest-value move is to build the instrument first and validate it on mouse data.** Then
   point it at the engine's four live drift substrates. The first engine study should be the L1b debt
   (section 4, D1), because a negative there is already on record and is waiting for its control.
5. **Some things are out of scope.** This map proposes no new plasticity rule, no pruning strategy
   and no sleep-benefit claim. Each would collide with the re-proposal guard
   (`docs/AGENT_COLD_START.md` §10) or with P4d's "Not claimed" list (`docs/POLICIES.md:619`)
   (section 7).

## 1. The engine's drift anatomy

**Drift sources** (S) are places where the response to a fixed input can change over time. **Drift
readers** (R) are places where something downstream reads a changing code.

| id | where | what changes for a fixed input | live? | today's reader |
|---|---|---|---|---|
| **S1** | `growth/growing_learner.py` `observe` (ART prototypes; `lr` 0.2, vigilance 0.85 on the arm learner) | Prototypes refine toward recent inputs, and recruitment adds categories, so the same percept can be **relabelled** over time | LIVE (arm, vision, audio) | Prototype ids are the targets for the next stage (`engine/brain.py` ~:2577) |
| **S2** | `growth/ng_snn.py` `_rates`, `prune_unearned`; refit on every 20th sleep tick (`neurogenesis_engine.py` `_refit`) | **Population membership** turns over through admissions and prunes. A surviving unit's response is identical by construction, because the rows are frozen and the layer is stateless per percept | LIVE (sleep refit); deployed charge ~0 | S3 |
| **S3 / R1** | readout beta: `OnlineReadout("delta")` on every wake percept, plus batch `_solve_beta` at open-gate refits | The reader itself drifts, recency-shaped (L1c) | LIVE | behaviour |
| **S4** | `readout/predictive_coding.py` W1/W2, `learn_step()` on every wake tick (`make_brain` sets `predictive_coding=True`) | Settled causes `r2` for a re-presented stimulus. Over 400 ticks, energy fell 67.8% with learning vs +9.1% frozen | LIVE | Only the scalar `pc_surprise` reads it. **The `r2` code itself is unread** |
| **S5** | chat R-STDP neurons + `PlasticityRuleStack` (BCM pre-gain, homeostatic scaling, intrinsic plasticity, iSTDP, Benna-Fusi deposit/diffuse) | Weights move continuously (max\|dw\| 15.30 with routing held at 1.000, `plasticity/homeostatic_scaling.py:16-20`) | LIVE on chat | Representation = readout (single layer): **no null space exists here** |
| **S6** | `GrownReturnPort` (`growth/grown_code.py`): plastic R-STDP afferents onto grown units | Latencies of grown units for the same text: a true hidden-code drift | G9 scratch homes only (`require_isolated_home`) | `OwnedPartDReader` (`readout/posed_problem.py`) |
| **S7** | `sleep/episode_consolidation.py` `replay(steps=60)`: covariance STDP on a 256-unit episode attractor | Attractor patterns across sleep | LIVE (sleep) | `retained()`; no cross-session correlation |
| R2 | `readout/laminar_local_readout.py` on the frozen encoder latent | The input is fixed, so only the reader moves | LIVE | behaviour |

The following do **not** drift today:
- the grown input rows (L1b deleted the rules);
- the 52-dimensional chat code;
- the encoder;
- the reservoir. It is plastic only in the blocked E4 row, and carried state was not useful: G1(c)
  gave D = -0.0074, t = -2.14.

The sleep tick at HEAD is narrower than the plan's text suggests. The wholesale Benna-Fusi
`relax()` rebind, `interaction_replay_deposit` and `attractor_sleep_learning` were **retired on
2026-10-13** (`engine/brain.py:4032-4034`). Plan line 1634's "the sleep tick overwrites those weights
(rewire, relax)" is superseded. Benna-Fusi's pull toward the deep store now happens only at wake, one
`diffuse(1)` per deposit.

### Stimuli the lake already holds

`percept` events (`substrate/learner_events.py`, `_T_EPISODE`, an IMMORTAL record class) carry the
exact float64 x, category, prototype count and session. They are an exact, permanent bank of probe
stimuli for "the same stimulus on a different day."

The frame pointer `_real_ptr` restarts at 0 on every Brain construction, and the packs are
256-periodic. So the same frames also recur within a session and after every relaunch.

### What the lake does not hold

The lake does not hold representation history. Prototypes and the chat learner are `_T_STATE` frames
whose superseded copies may be dropped (`substrate/lake_engine.py`). Growth snapshots
(`_T_GROWTH_DELTA`) are COMPACTABLE. PC weights have no lake record that I found. Reservoir records
are still owed (G-R9 (vi)).

Two consequences follow:
- Any drift study runs on **scratch brains that snapshot explicitly**.
- Past chat states are reconstructible only by deterministic redo, and an API for that is
  unconfirmed.

## 2. What the engine has already measured that bears on drift

| record | measured | what it means for drift |
|---|---|---|
| `evidence/l1b-synaptic-online.md` | Oja on grown input rows: t = -1.959 vs frozen (harm past the bar); BCM t = -1.358. Oja also changed admissions: [6,5,4,6,15,4,6,2] vs [2,2,2,7,6,3,3,2] | Code drift under a stale reader harms. Drift also changed **growth decisions**. Owed: the re-solve-every-refit control and the derivation of μ. `null-scale-triage.md` rows 11/U4 classify it as "NOT a measured harm" under P8's amendment, as a retest CANDIDATE with mde unknown |
| `evidence/l1c-online-readout.md` | On a drifting chain, batch 0.445 vs online delta 0.638. On a stationary chain the online rule costs 0.04-0.09. In-brain, delta vs incumbent t = 4.132, +0.143 | A co-adapting reader wins when the world drifts and costs a little when it does not. It is recency-shaped: on the marginal-covering held-out slice it scores -0.253 / -0.111 / +0.003 / -0.067 |
| `evidence/g5-readout-variants.md` | DELTA - BATCH +0.1509, t = 4.551 (replicated). The trace rule on "drift" and "burst" chains passed 1 of 12 configurations and replicated 0, so it was deleted | Tracking a drifting world is solved at the reader by delta. A temporal-trace reader was not |
| `evidence/s2b-growing-learner-lake-record.md` | "The readout re-keys after restart" read NULL (2-0-4) but **vacuous**: the children replayed identical streams | A cross-session drift question that is open and owed |
| `evidence/r3-replay-growth-composition.md` | Task A 1.000 → 0.250 after teaching B, → 0.500 after 10 sleep ticks. Sleep plasticity F1 NULL; F2 NOT READ (turnover 0 on 8/8 seeds); F3 NULL; extended sleep F4 credited (C9 t = 5.465) | Interference and partial recovery across sleep are measurable on this harness |
| `evidence/g6-sleep-durable-memory.md` | Structural rewiring turnover 0.0 over 102 cycles; downscale-prune null | `sleep/structural_rewiring.py` has no writer that crosses zero, so it is not a drift source today |
| `evidence/t063-prior-drift-harness.md` | The kernel learner lost to its frozen twin on every seed (t = -13.0154) and was deleted | Weight drift on the H01 prior is "not measurable" (no writer). FR-008/009 stay OPEN |

## 3. Literature → engine: where each finding lands

Citations marked [V] were verified against an indexed record. [V\*] means one secondary listing only.
The full list is in section 9.

| # | finding | engine substrate | what it predicts here | feasibility |
|---|---|---|---|---|
| L1 | **Drift in coding-null directions; part of it is not null and degrades a fixed decoder** (Rule, O'Leary & Harvey 2019 [V]; Rule et al. 2020 [V]; Keinath 2022 [V]; Xia 2021 [V]) | S2/S3, and L1b's moving rows | L1b's harm should be explained by the **row-space fraction** of the row drift relative to beta. A re-solved beta should absorb it. | High: replays L1b's own arms (D1) |
| L2 | **Self-healing readout**: Hebbian plus homeostasis tracks drift without error feedback (Rule & O'Leary 2022 [V]) | R1 delta readout; the chat stack S5 | The delta reader is already a supervised tracker. An unsupervised self-healing reader would be a new rule, which this map does not propose | Measurement only |
| L3 | **Coordinated drift**: under noisy Hebbian/anti-Hebbian learning, receptive fields random-walk while RSA geometry stays fixed (Qin et al. 2023 [V]) | **S4 predictive coding** (local learning, unread code) | `r2` tuning for re-presented percepts walks while the population RSA matrix stays correlated with day 0. Diffusion should scale with the learning rate | High: live, cheap, needs only snapshots (D3) |
| L4 | **Geometry**: drift lies along high-variance, within-class directions, and that pattern matches **dropout** specifically (Aitken et al. 2022 [V]) | S2 turnover (admit/prune ≈ structured dropout) | Turnover-driven population change should be aligned with within-class variance. A Gaussian-noise control should not be | Medium: needs growth snapshots (D2) |
| L5 | **Frequent stimuli drift less**; stopping exposure speeds drift back up (Schoonover 2021 [V]; Pashakhanloo & Koulakov 2023 [V]; Morales 2025 [V\*]) | S1 prototypes (lr-weighted refinement) | Prototypes of frequent categories move less per presentation but get more presentations. The net sign is an empirical question that the 256-periodic stream makes easy to test | High (D2) |
| L6 | **Time vs experience**: rate drift follows time, tuning drift follows experience (Geva 2023 [V\*]; Khatib 2023 [V\*]) | S1/S4 vs S2 | Tuning-like drift (S1, S4) should scale with presentations. Participation drift (S2) should scale with sleep refits | Medium (D2/D3 design) |
| L7 | **Sleep stages push drift in opposite directions**: NREM accelerates assembly drift and REM counters it (Bollmann 2025 [V]). Sleep ripples pre-tune the next day's fields (Maboudi 2024 [V]) | S7 episode attractor; G6 staged bout (`instruments/staged_sleep_bout.run_bout`) | Drift per cycle differs between staged and flat bouts. Replay content predicts next-session tuning better than the previous session does | Medium; **licensed sentence only about representation, never benefit** (D4) |
| L8 | **Replay-based continual learning keeps old tasks while representations drift; weight anchoring freezes them** (arXiv 2608.16141, 2026 [V\*]; van der Veldt et al. 2025 [V]) | S5 Benna-Fusi cascade; R3's A-then-B harness | A cascade should allow code drift while retaining task A. **No paper measures drift under a Benna-Fusi cascade (Benna & Fusi 2016 [V])**: a genuine literature gap | Medium; S5 is single-layer, so measure on R3's harness (D6) |
| L9 | **Assembly membership protects against drift**; drift arrives as abrupt switches, not diffusion (Gonzalez 2019 [V]; Kossio 2021 [V]; Micou & O'Leary 2024 preprint [V]) | S2 (admission and prune are literally switches); S7 | Unit fate (pruned or kept) should be predicted by co-activation with the rest of the population | Medium |
| L10 | **Latent realignment** keeps decoders working across months (Gallego 2020 [V]; Degenhart 2020 [V]; Karpowicz 2025 [V]) | G9 restart (G-R9 (vi): the rebuilt reservoir reproduces earlier answers) | Procrustes realignment as a diagnostic: is a restart failure a rotation (recoverable) or a loss of structure? | Later (D5) |
| L11 | **Behaviour and sampling confounds** masquerade as drift (Sadeh & Clopath 2022 [V]; Stringer 2019 [V]; Liberti 2022 [V]; Yuan 2026 [V\*]) | Input stream: attention-gained `_context_pattern`, so the arm's own attention changes what it samples | Apparent drift must be measured on **fixed lake-stored probes**, never on the live stream. That is also the §9 lesson on synthetic tasks | Design constraint |
| L12 | **Stimulus-dependent drift**: gratings are stable, natural movies drift (Marks & Goard 2021 [V]) | Part A validation target | The instrument must reproduce this direction on DEV data before it reads the engine | Part A (D0) |

## 4. A ranked research programme

Each study is listed with the plan row it naturally attaches to. The owner decides row ids. A new
red-family letter is needed, since A, B, C, G, H, L, N, R, S, U, W and X are taken. `D-` would be
free.

### D0. The instrument, validated on synthetic and mouse data (prerequisite)

**New module.** Pure measurement, a `permanent` dormancy exemption in the `prior_drift_harness` shape:
`instruments/representational_drift.py`. Functions:
- `pv_correlation`: population-vector correlation on a fixed neuron × stimulus slice;
- `tuning_correlation`, with circular differencing for orientation;
- `decoder_across_days`: closed-form ridge/lstsq as `DIRECT_SOLVE`, **capacity-guarded**, because
  T063's over-parameterised readout read 1.0000 on every wiring;
- `readout_null_split(delta, W)`;
- `procrustes_realign`, using `scipy.linalg.orthogonal_procrustes` (scipy is already core);
- a `DriftReading` result type, deliberately not named `DriftRecord`.

**Reds, each run before the claim is read:**
- A/A identical sessions read exactly 0 (the K7 sighted-detector pattern in `sleep_plasticity_ab`).
- A planted rotation is recovered at graded doses.
- Planted drift confined to null(W) reads a fraction ≈ 1, and row-space drift reads ≈ 0. A
  random-readout control of matched rank is included.
- A planted shift in preferred orientation across the 0/180 wrap reads its true size.
- An over-parameterised decoder is refused by the capacity guard.
- A non-orthogonal planted drift is not "recovered" by Procrustes.

**Mouse validation (Part A).**
- **Reproduce** Marks & Goard's gratings-vs-movies direction and Deitch et al.'s preserved
  representational geometry on DEV mice, then read a sealed HELD-OUT set.
- If the instrument fails to reproduce them, **the instrument is BLIND** (repair or delete). It is
  never "mice do not drift."

**Data route.**
- Neuropixels NWB units: **extend** `sensorimotor/organoid_ingest.py`. It already reads
  `units/spike_times` with byte, licence, index, time-unit and reproduction gates. P2 forbids a
  second NWB units reader.
- 2-photon ΔF/F and the Dryad deposit need a new `sensorimotor/ophys_ingest.py`. Use `h5py`, as
  OG1 did with "no pynwb".
- AllenSDK, if needed at all, runs only in a subprocess venv (`tools/rd1_fetch.py`), because it is
  unlikely to install on cp314t.
- Data goes to `data/drift/` (gitignored, never the lake: "what a microscope saw").
- Pins go in `tools/*_pin.json`.

**Attaches to:** a new row. Statistics come from `instruments/paired_contrast.py` (mice or sessions
as the paired unit).

### D1. L1b's owed control, read as drift (first engine study; highest value)

**Question.** When grown input rows move, what fraction of their effect on hidden rates lies in beta's
row space? Does re-solving beta at every refit absorb it?

**Arms.** These are L1b's own arms on scratch homes:
- FROZEN rows;
- OJA rows with a stale beta;
- OJA rows with **beta re-solved at every refit** (the owed control);
- PLANTED: a known rotation of the rows confined to null(beta), and one in row(beta).

**Prediction (L1).** Harm scales with the row-space fraction. The re-solve arm removes the harm. The
planted null-space arm shows none.

**Why first.** It pays an existing debt (`l1b-synaptic-online.md:312`; G5/T113's G-R5 requires "L1b's
co-adaptation controls") and needs no new learner, since the arms are L1b's.

**Not proposed:** re-landing the Oja rule. If the control clears it, that is a separate
pre-registration under the §10 guard.

### D2. Prototype and population drift on lake-stored probes (S1, S2)

**Question.** Re-present IMMORTAL `percept` events to snapshots of the grower taken across a run.
How fast do labels change (S1)? How does population geometry change under admit/prune (S2), with
surviving units held fixed as the slice?

**Predictions:**
- Frequency dependence (L5).
- Presentations vs refits (L6).
- Turnover aligned with within-class variance (L4).
- Unit fate predicted by co-activation (L9).

**Attaches to:** G3 / T111, T111a (G-R3 "old codes change", "the old mapping regresses past its
bound"), and s2b's owed restart re-keying test scored after the streams diverge.

**Must report:**
- added units separately (§10: a unit count is not growth);
- executed / attempted / effective-change counts before any reading. S2's deployed charge is ~0,
  so a zero there is a structural null.

### D3. Coordinated drift in predictive coding (S4)

**Question.** For re-presented percepts, does `r2` tuning random-walk while its RSA geometry stays
put (Qin 2023)?

**Arms:**
- learning on;
- a frozen twin forked from the same snapshot (`gates/replay_gate.py` RNG positions);
- planted rotation;
- learning-rate doses.

**Why it matters for the engine.** `r2` is computed every wake tick and read by nothing except a
scalar. If its geometry is stable, it is a candidate representation for a future reader. If it
collapses, that is a finding about an incumbent readout rule. It is also the cleanest test of a
headline theoretical result (Qin et al.) in a local-learning spiking-adjacent network.

**Attaches to:** G5 (predictive coding is one of the owner's named readout rules).

### D4. Drift across sleep vs wake (S7, staged bout)

**Question.** Per cycle, does the episode-attractor code drift more across sleep than across wake?
Do staged and flat bouts differ in drift direction (L7)? Does replay content predict next-session
codes (Maboudi)?

**Controls.** R3's FLAT/OFF equal-compute bouts and the stage shuffle.

**Licensed sentence.** Only about representation change. "Sleep improves the brain" is refused
(P4d; R3's F1 and F3 are recorded NULLs).

**Attaches to:** G6 / T114 (e), R3 / T104 (G-R6, R-R3).

### D5. Grown-code drift under the return port and restart realignment (S6, G9)

**Question.** Once `GrownReturnPort` learns, how far do grown units' latencies move for fixed text?
After a restart, is any failure of G-R9 (vi) a rotation that Procrustes recovers, or lost structure?

**Attaches to:** G9 / T117 (G-R9 (v)/(vi)). It is gated on G9's maturity and runs in isolated homes
only.

### D6. Drift under a Benna-Fusi cascade (literature gap)

**Question.** On R3's A-then-B harness, does the representation of task A drift while A is retained
(L8)?

**Note.** S5 is single-layer, so this is measured on the harness's hidden or attractor state, not
the chat neurons. It is the most publishable item: no published work measures drift under a
cascade.

**Attaches to:** R3 / T104, and `docs/ACADEMIC_RESEARCH_PLAN.md`'s retention criterion (re-scoring
earlier tasks after each new one). Drift answers *how* retention holds.

## 5. Governance checklist (condensed)

Sources: `docs/AGENT_COLD_START.md` §6, §8.8, §9; `docs/POLICIES.md` P1, P2, P4b/c/d, P7, P8; the
OG1, T063 and R3 pre-registrations.

1. **Existence check.** Record the three negative searches. Every new symbol carries
   distinct-from markers against `prior_drift_harness`, `DriftRecord` and `infra/drift_monitor`.
2. **Pre-registration committed alone, before any red or data byte.** It goes in `evidence/<row>.md`
   with sections Territory / Pre-registration / Record, and dated append-only amendments. Part B is
   frozen only after Part A's verdict.
3. **Conjuncts of PASS** read MET / NOT MET / NOT READ. **One constructed red per criterion**, or the
   criterion is deleted (`tools/red_reachability.py --assert`; name criteria `*_verdict` so the scan
   sees them).
4. **Controls that are not the method:** FROZEN twin, PLANTED, marginal-preserving shuffle and
   random-readout. Each runs in the mechanism's own regime.
5. **P1 sample size:**
   - measured SD and its source;
   - the claimed effect;
   - the fixed-sample target at power .8 after multiplicity correction;
   - a sequential design with looks, named alpha spending, non-binding futility and an inflated cap;
   - simulated operating characteristics.

   The sequential machinery exists only inside `instruments/rpe_charge.py` (`seq_boundaries`,
   `seq_decision`, `_mde`). Reusing it is a P2 **lift** into a shared module, in its own commit with
   R-S1..R-S5 as the equivalence oracle.
6. **Null handling.**
   - Format: `NULL-RECORD: method | units= | ticks= | seeds= | mde= | retest=`.
   - Deletion waits for a ≥10x re-test on **units or ticks**, the only legal axes
     (`gates/null_scale_gate.py` `AXES`).
   - Underpowered means INCONCLUSIVE.
7. **Licensed and refused sentences**, per study. The refused sentence always includes "the engine
   drifts like mouse cortex". There is no registry red for cross-species likeness.
8. **Registration in one commit:** `REGIONS`, `run_selfchecks.CORE` (self-check < 60 s, no
   download), a `dormancy_gate.EXEMPT` row, `GRADIENT_CLAIM` for the decoders (`DIRECT_SOLVE`),
   tests, the 800-line soft ceiling, ASCII output, and `random.Random(seed)` per instance.
9. **Engine runs** use `throwaway_home()` and `make_brain(seed=, persist=False)`, one subprocess per
   arm and seed, with `require_scratch_home` / `guard_out`. The live lake is digested before and
   after.

## 6. Decisions only the owner can make

1. **Row id and red-family letter** (proposal: `D-R*`), and where the PLAN-FINAL row goes.
2. **Null axis for Part A.** The null-scale gate admits only `units` and `ticks`. "More mice",
   "more sessions" or "more days" is not a legal 10x axis, so a Part A failure must be ruled up front
   as one of:
   - INSTRUMENT BLIND (repair or delete, as in T063 2.9);
   - a NULL-RECORD on a newly admitted axis.
3. **Licences.** `sensorimotor/ingest_guard.license_ok` is deny-by-default. "Allen Institute Terms of
   Use" is not on the allowlist, so it is refused unless the DANDI copy carries an SPDX id
   (unconfirmed). The Dryad deposit is probably CC0 (unconfirmed).
4. **More than 40 paired units.** `paired_contrast` refuses n > 40 (`MITM_MAX_N`, never
   approximated). If mice or sessions exceed 40, either block the data or approve an approximate
   FPR path.
5. **Lifting the sequential-design code** out of `rpe_charge` (a P2 merge).
6. **Dependency route:** extend the h5py reader (recommended; no new dependency) vs an AllenSDK
   subprocess venv.

## 7. What this map deliberately does not propose (re-proposal guard)

- **No Hebbian, BCM or Oja rule on grown input rows.** D1 measures L1b's arms and does not re-land
  the rule (§10, L1b row).
- **No pruning or annealing guided by drift.** Pruning strategies were neutral or negative in
  v3.43-v3.48, and annealing is restricted to G6's joint-mask validation.
- **No sleep-benefit sentence.** That is P4d's "Not claimed", with R3 F1/F3 NULL and G6's
  downscale-prune null.
- **No learner on the H01 prior.** T063's learner was deleted on the frozen-twin criterion.
- **No unit count read as growth.** Admissions in D2 are reported separately.
- **No measurement on the live stream or the deployed daemon.** Its E-STOP has been latched since
  2026-08-31, it has one silent unit, and it replays its early stream at every relaunch. Synthetic
  tasks have misled three times (§9), and the stream has no first-order succession (OG1 §1).

## 8. Open checks I could not confirm

- Whether the live daemon runs `persist=True`, and whether any `_T_GROWTH_DELTA` prefix has already
  been compacted on the real lake. These were not reachable from a cloud clone.
- Whether an API exists for deterministic redo of the chat learner's past states.
- Whether `readout/tempotron.py` is reached on the live tick.
- Dataset details: DANDI ids and versions for the Allen sets, the Dryad file format, and the
  published numbers to pre-register for REPRODUCE.
- Where seed blocks are allocated. G3 cites a board `SEEDS.md` that is not in the tree.
- `CONTRIBUTING.md:5` says "Python 3.11+ (CI runs 3.11/3.12/3.13)", which contradicts the cp314t
  requirement. This is a stale-doc defect for the operator.

## 9. Bibliography

[V] verified against an indexed record; [V\*] single secondary listing; [U] unverified.

- Aitken K, Garrett M, Olsen S, Mihalas S (2022). The geometry of representational drift in natural and artificial neural networks. *PLOS Comput Biol* 18(11):e1010716. doi:10.1371/journal.pcbi.1010716 [V]
- Attardo A, Fitzgerald JE, Schnitzer MJ (2015). Impermanence of dendritic spines in live adult CA1 hippocampus. *Nature* 523:592-596. doi:10.1038/nature14467 [V]
- Benna MK, Fusi S (2016). Computational principles of synaptic memory consolidation. *Nat Neurosci* 19:1697-1706. doi:10.1038/nn.4401 [V]
- Bollmann L, Baracskay P, Stella F, Csicsvari J (2025). Sleep stages antagonistically modulate reactivation drift. *Neuron* 113:1446-1459. doi:10.1016/j.neuron.2025.02.025 [V]
- Climer JR, Davoudi H, Oh JY, Dombeck DA (2025). Hippocampal representations drift in stable multisensory environments. *Nature* 645:457-465. doi:10.1038/s41586-025-09245-y [V]
- Degenhart AD et al. (2020). Stabilization of a brain-computer interface via the alignment of low-dimensional spaces of neural activity. *Nat Biomed Eng* 4:672-685. doi:10.1038/s41551-020-0542-9 [V]
- Deitch D, Rubin A, Ziv Y (2021). Representational drift in the mouse visual cortex. *Curr Biol* 31(19):4327-4339.e6. doi:10.1016/j.cub.2021.07.062 [V]
- Delamare G, Zaki Y, Cai DJ, Clopath C (2024). Drift of neural ensembles driven by slow fluctuations of intrinsic excitability. *eLife* 12:RP88053. doi:10.7554/eLife.88053 [V]
- de Vries SEJ et al. (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nat Neurosci* 23:138-151. doi:10.1038/s41593-019-0550-9 [V\*]
- de Vries SEJ, Siegle JH, Koch C (2023). Sharing neurophysiology data from the Allen Brain Observatory. *eLife* 12:e85550. doi:10.7554/eLife.85550 [V]
- Driscoll LN, Pettit NL, Minderer M, Chettih SN, Harvey CD (2017). Dynamic reorganization of neuronal activity patterns in parietal cortex. *Cell* 170:986-999. doi:10.1016/j.cell.2017.07.021 [V]
- Driscoll LN, Duncker L, Harvey CD (2022). Representational drift: emerging theories for continual learning and experimental future directions. *Curr Opin Neurobiol* 76:102609. doi:10.1016/j.conb.2022.102609 [V\*]
- Gallego JA, Perich MG, Chowdhury RH, Solla SA, Miller LE (2020). Long-term stability of cortical population dynamics underlying consistent behavior. *Nat Neurosci* 23:260-270. doi:10.1038/s41593-019-0555-4 [V]
- Geva N, Deitch D, Rubin A, Ziv Y (2023). Time and experience differentially affect distinct aspects of hippocampal representational drift. *Neuron* 111(15):2357-2366.e5. doi:10.1016/j.neuron.2023.05.005 [V\*]
- Gonzalez WG, Zhang H, Harutyunyan A, Lois C (2019). Persistence of neuronal representations through time and damage in the hippocampus. *Science* 365:821-825. doi:10.1126/science.aav9199 [V]
- Jensen KT, Kadmon Harpaz N, Dhawale AK, Wolff SBE, Ölveczky BP (2022). Long-term stability of single neuron activity in the motor system. *Nat Neurosci* 25:1664-1674. doi:10.1038/s41593-022-01194-3 [V]
- Karpowicz BM et al. (2025). Stabilizing brain-computer interfaces through alignment of latent dynamics. *Nat Commun* 16:4662. doi:10.1038/s41467-025-59652-y [V]
- Keinath AT, Mosser CA, Brandon MP (2022). The representation of context in mouse hippocampus is preserved despite neural drift. *Nat Commun* 13:2415. doi:10.1038/s41467-022-30198-7 [V]
- Khatib D et al. (2023). Active experience, not time, determines within-day representational drift in dorsal CA1. *Neuron* 111(15):2348-2356.e5. doi:10.1016/j.neuron.2023.05.014 [V\*]
- Kossio YFK, Goedeke S, Klos C, Memmesheimer RM (2021). Drifting assemblies for persistent memory: neuron transitions and unsupervised compensation. *PNAS* 118(46):e2023832118. doi:10.1073/pnas.2023832118 [V]
- Liberti WA et al. (2016). Unstable neurons underlie a stable learned behavior. *Nat Neurosci* 19:1665-1671. doi:10.1038/nn.4405 [V]
- Liberti WA, Schmid TA, Forli A, Snyder M, Yartsev MM (2022). A stable hippocampal code in freely flying bats. *Nature* 604:98-103. doi:10.1038/s41586-022-04560-0 [V]
- Maboudi K, Giri B, Miyawaki H, Kemere C, Diba K (2024). Retuning of hippocampal representations during sleep. *Nature* 629:630-638. doi:10.1038/s41586-024-07397-x [V]
- Marks TD, Goard MJ (2021). Stimulus-dependent representational drift in primary visual cortex. *Nat Commun* 12:5169. doi:10.1038/s41467-021-25436-3. Data: doi:10.25349/D9M606 [V]
- Micou C, O'Leary T (2023). Representational drift as a window into neural and behavioural plasticity. *Curr Opin Neurobiol* 81:102746. doi:10.1016/j.conb.2023.102746 [V\*]
- Micou C, O'Leary T (2024). Statistics of cortical representational drift can enable robust readout. *bioRxiv*. doi:10.1101/2024.09.26.614914 [V]
- Mongillo G, Rumpel S, Loewenstein Y (2017). Intrinsic volatility of synaptic connections. *Curr Opin Neurobiol* 46:7-13. doi:10.1016/j.conb.2017.06.006 [V]
- Morales GB, Muñoz MA, Tu Y (2025). Representational drift and learning-induced stabilization in the piriform cortex. *PNAS*. doi:10.1073/pnas.2501811122 [V\*]
- Pashakhanloo F, Koulakov A (2023). Stochastic gradient descent-induced drift of representation in a two-layer neural network. *ICML, PMLR* 202:27401-27419 [V]
- Qin S et al. (2023). Coordinated drift of receptive fields in Hebbian/anti-Hebbian network models during noisy representation learning. *Nat Neurosci* 26:339-349. doi:10.1038/s41593-022-01225-z [V]
- Ratzon A, Derdikman D, Barak O (2024). Representational drift as a result of implicit regularization. *eLife* 12:RP90069. doi:10.7554/eLife.90069 [V]
- Rubin A, Geva N, Sheintuch L, Ziv Y (2015). Hippocampal ensemble dynamics timestamp events in long-term memory. *eLife* 4:e12247. doi:10.7554/eLife.12247 [V]
- Rule ME, O'Leary T, Harvey CD (2019). Causes and consequences of representational drift. *Curr Opin Neurobiol* 58:141-147. doi:10.1016/j.conb.2019.08.005 [V]
- Rule ME et al. (2020). Stable task information from an unstable neural population. *eLife* 9:e51121. doi:10.7554/eLife.51121 [V]
- Rule ME, O'Leary T (2022). Self-healing codes. *PNAS* 119(7):e2106692119. doi:10.1073/pnas.2106692119 [V]
- Sadeh S, Clopath C (2022). Contribution of behavioural variability to representational drift. *eLife* 11:e77907. doi:10.7554/eLife.77907 [V]
- Schoonover CE, Ohashi SN, Axel R, Fink AJP (2021). Representational drift in primary olfactory cortex. *Nature* 594:541-546. doi:10.1038/s41586-021-03628-7 [V]
- Sheintuch L et al. (2017). Tracking the same neurons across multiple days in Ca2+ imaging data. *Cell Rep* 21:1102-1115. doi:10.1016/j.celrep.2017.10.013 [V]
- Siegle JH et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature* 592:86-92. doi:10.1038/s41586-020-03171-x [V]
- Stringer C et al. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science* 364:eaav7893. doi:10.1126/science.aav7893 [V]
- van der Veldt S, van de Ven GM, Moorman S, Etter G (2025). Learning continually with representational drift. arXiv:2512.22045 [V]
- Xia J, Marks TD, Goard MJ, Wessel R (2021). Stable representation of a naturalistic movie emerges from episodic activity with gain variability. *Nat Commun* 12:5170. doi:10.1038/s41467-021-25437-2 [V]
- Ziv Y et al. (2013). Long-term dynamics of CA1 hippocampal place codes. *Nat Neurosci* 16:264-266. doi:10.1038/nn.3329 [V]
- (2026). Continual-learning rules shape representational drift. arXiv:2608.16141 [V\*; authors U]

Open datasets beyond the Allen and Marks & Goard sets:
- Driscoll 2017 PPC: Dryad doi:10.5061/dryad.gqnk98sjq [V]
- Keinath 2022 CA1: Dryad doi:10.5061/dryad.2z34tmpp9 [V]
- Allen Visual Behavior 2P (multi-session familiar/novel; Garrett et al. bioRxiv 10.1101/2023.02.14.528085) [V]

---

## Amendment 2026-10-29 to D0: one drift home (integrator G5 S.5 row 5, launcher D-70)

Source: `g5-readout-variants.md` fold 1 F.6 (claude/g5-settle-1029@2157ee27) and the G9 draft's owed item 3
(claude/g9-prereg-1029@a404bada). Appends only; nothing above is edited, and nothing is built.

**Names `instruments/representational_drift.py` owns:**
- from D0 as written: `pv_correlation`, `tuning_correlation`, `procrustes_realign`, `DriftReading`;
- added by F.6: `rate_correlation` and `rsa_correlation` (F.2), and `read_drift(pre, post, *, probe_ids, units=None, metric)`, a thin
  dispatcher over the four readers that returns a `DriftReading` (fields `values`, `mean`);
- the F.1 unit rule (compare the INTERSECTION unit set) and the F.5 guard (flat response is flagged, not 0.0) apply to all four;
- **`rsa_drift` moves in** (G9 owed item 3). It now lives in `infra/_mathutils.py` (added by `f0f50385`, test first in `fd987f3f`,
  both on the G9 branch only). When the module is built it is folded into the module as the body of `rsa_correlation`, and the
  `_mathutils` copy is removed in the same change, so exactly one RSA function exists.

**Import rule for Part E and G9:**
- Part E's alignment module imports `procrustes_realign` from this module and keeps only the anchors and the readout.
  P-E0 uses `pv_correlation` alone.
- G9's criterion 1 imports the readers (`rate`, `tuning`, `rsa`, `pv`) from here.
- Neither defines a drift-reading or Procrustes function of its own (P2). Correlations come from `infra/_mathutils.pearson`.

---

## Correction 2026-10-29 to the D0 amendment above: D0 imports the built section 2.4 (integrator, after policy-enforcer #6757)

Source: policy-enforcer #6757 (MAJOR-1, MINOR-3) and the integrator's correction of G5 S.5 row 5. Appends only; nothing above is edited,
and nothing is built. It **replaces the first sentence of the amendment above** ("`instruments/representational_drift.py` owns...").

**Why.** The drift home is the BUILT section 2.4 code, `src/snn_data_lake/instruments/readout_staleness.py:269-300` (c0f90736):
`survivor_pairs`, `rate_drift`, `readout_projector`, `visible_fraction`. F3's rule is "whichever builds first owns".

**D0's owned names now:**
- `readout_null_split(delta, W)` is **not a new function**. It IS `readout_projector` + `visible_fraction` in `readout_staleness.py`. D0
  imports them; a `(delta, W)` convenience is at most a one-line wrapper over that pair, never a second SVD or projector.
- `survivor_pairs` and `rate_drift` stay in `readout_staleness.py` and are imported. `rate_drift` (a norm ratio) is distinct from
  `pv_correlation`, `tuning_correlation`, `rate_correlation` and `rsa_correlation` (scale-free correlations): different quantity, so no
  merge.
- `decoder_across_days` is D0's, but is **a placement, not a new home**: it goes in the drift section of `readout_staleness.py`, or in
  the merged drift module described next. It is not built anywhere yet.
- `procrustes_realign`, `pv_correlation`, `tuning_correlation`, `rate_correlation`, `rsa_correlation`, `DriftReading`, `read_drift`:
  unbuilt. When built they go into the same drift section, or the whole section moves into one drift module in ONE P2 merge commit with
  its tests. Never a second module beside it. The module name `instruments/representational_drift.py` is the target name of that merged
  module; until then `readout_staleness.py` is the home.
- `rsa_drift`: built on the unlanded branch claude/g9-gaps-1029 (f0f50385), at `infra/_mathutils.py:186`. At the ARC6 landing it is
  placed in (or imported by) the same drift section; the `_mathutils` copy is not kept alongside. Exactly one RSA function.

**Import rule (restated).** Part E's alignment module imports `procrustes_realign` from the drift home and keeps only its anchors and
readout. P-E0 uses `pv_correlation` alone. G9's criterion 1 imports the readers from the drift home. Neither defines a drift-reading or
Procrustes function of its own (P2). The F.1 unit rule and the F.5 guard apply to every reader.

**Named P2 debt (MINOR-3, restored from g5-readout-variants.md F.5/F.6).** Three flat-vector semantics are live in the tree today
(`infra/_mathutils.py:44` `pearson` raises ZeroDivisionError; `faculties/grid_place_map.py:49` `_pearson` returns 0.0;
`faculties/agency_monitor.py:1118` `_t079_pearson` returns None), and `read_drift`'s explicit flag would be a fourth. Owner: the lane that
builds the drift readers (the integrator assigns it). X.3 row 8: a caller census of `_mathutils.pearson`, then either one guard in the
shared primitive or the consolidation of the guarded copies. The shared helper does not change in this arc. A full census at c0f90736
finds five `pearson` definitions (also `instruments/rpe_charge.py:343`, `growth/h01_ingest.py:894`); `grid_place_map._pearson` is a
population-vector correlation that `pv_correlation` must dispose of.
