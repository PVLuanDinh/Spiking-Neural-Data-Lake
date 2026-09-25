# G1(b) / T109 part (b) -- the pool-selection benchmark

Row: plan Part D "G workstream", G1 / T109: "Before (b) runs, G1 builds and pre-registers a pool-selection benchmark on
prospective live-percept prediction, with matched information, tuning budgets and resource bounds; the chosen pool
demonstrates the determinism contract and its constructed red." Arm (b) is "ONE recurrent pool promoted from the four
research-tier reservoirs under P2, carrying state"; arm (c) is "the same pool reset each percept". FR-041: the pool
"MUST be attributed by carried-versus-reset state under a frozen causal information contract". G-R1 (Part F, amended
2026-09-25 with this file) is G1(c)'s red; this arc selects the pool G1(c) tests. The harness is G1(a)'s
(`evidence/g1a-causal-contract.md`, `cortex/memory_diagnostic.py`). Status: PRE-REGISTERED. Two Codex design reads and one
Claude Opus read are folded (section 2.12); the owner paused Codex on 2026-09-25. Section 2 is frozen at the commit
that lands this file. Section 3 records the build, the runs and every deviation from section 2.

## 1. Territory, read at HEAD `fd89e54` before the design

The self-check logs, the design probes and their outputs are archived under `archive/g1b-2026-09-25/` in the memory
store. Every number below is printed by one of them: `selfcheck_*.log`, `selfcheck_timings.out`, `probe_territory.out`,
`probe_contrast_sd.out`, `probe_rb6_positive_control.out`, `probe_rb6_reference.out`, `probe_rb6_lp.out` or
`probe_ridge_frozen.out`.

1. **The four research-tier reservoirs** (no importer outside `reasoning/`; plan "What the tree already has"; none has
   a measured row in `docs/P4C_BACKLOG.md`, which writes a row only when measured). Each module's own self-check passed
   at `fd89e54`, printing:
   - `reasoning/spiking_ssm.py`: a fixed random sparse spiking reservoir (Maass LSM). N = 100, fan-in 8, recurrent
     weights U(-0.5, 0.5), leak 0.7, threshold 1.0, and a reset of h to 0 on a spike. The input is a SCALAR
     (`w_in[i] * x`); `run_reservoir` starts every sequence at h = 0, s = 0 and returns mean rates. Pure Python.
     Printed: 96.4% +/- 5.9% on its 3-class task (chance 33%), 701 SynOps per sequence.
   - `reasoning/reservoir_eprop.py`: builds its recurrent layer FROM `spiking_ssm.build_reservoir`, densifies it
     `[pre, post]`, zeroes the diagonal and rescales the spectral radius to RHO = 0.90. The LIF is
     `v = ALPHA v - V_TH z_prev + cur` with ALPHA 0.90 and V_TH 0.6 (the soft reset), and `cur = x @ win + z_prev @ wrec`.
     Arm (i) freezes the reservoir and trains an e-prop readout; arm (ii) trains `win`, `wrec` and the readout by e-prop
     on the category targets. Printed (66 s): hybrid 54.4% +/- 3.8% vs pure e-prop about 55%. Only its FIXED reservoir
     can be a pool here: arm (ii)'s learning reads the category channel, which G1(a) section 2.1 forbids a pool. Its
     full-scale trainer is `cortex/eprop_shd.py`.
   - `reasoning/sorn.py`: N_E = 100 binary excitatory and N_I = 20 inhibitory threshold units; sparse plastic E->E
     (P_EE 0.10, rows normalized to 1), fixed E->I (0.20) and I->E (0.30); thresholds T_E ~ U(0, 0.5), T_I ~ U(0, 0.8).
     STDP, synaptic normalization and intrinsic plasticity (target H_IP 0.10) run over 800 six-symbol words (4,800
     steps; SHARED_LEN 4, 8 symbols); each symbol drives a disjoint block of 8 E units at 1.0, plus 4 seeded noise units
     per step; the readout drops a 20-step washout. (Its introductory prose still describes four-symbol words.)
     Printed: 89.1% vs a static twin's 84.4% (+4.7 points).
   - `reasoning/random_boolean_network.py`: N binary nodes, K inputs each, random truth tables with bias p, and a
     synchronous autonomous update (`step`). It has NO input port. Printed: Derrida slopes 0.48 / 0.99 / 1.50 against
     the derived 0.5 K.
2. **Governed recurrent structures, each with a different job.** The inventory below is what `graph-exists` and a
   source search found. It is not claimed to be exhaustive.
   - `plasticity/criticality_regulator.py::CriticalityRegulator`: a cascade-regime homeostat (n = 240, sparse
     excitatory out-edges, `LIFNeuron` units, activity-proportional inhibition, a knob moved toward branching ratio
     0.99). `cortex/brain.py` builds one (n = 64) and reports its sigma as `criticality`. It has no input port, and each
     `_cascade` resets every membrane.
   - `cortex/cortical_microcircuit.py::CorticalMicrocircuit`: Potjans-Diesmann, scalable. Its one port,
     `thalamic_drive`, is a SCALAR rate into L4/L6, and `simulate` restarts its membranes, delay buffers and noise.
     Nothing drives it (P3 row).
   - `cortex/eprop_kernel.py::EpropLIFBase.forward`: the governed implementation of the SAME soft-reset recurrent LIF
     equation as `reservoir_eprop` (`v = alpha v - v_th z + cur`, off-diagonal `W_rec`). It runs whole sequences from
     v = 0, z = 0, in torch.
   - `cortex/autapse_neuron.py::AutapseNeuron`: single-neuron delayed self-recurrent memory.
   - `cortex/prefrontal_executive.py::_lif_pool_run` (goal persistence), `midbrain/sc_deep_layers.py`'s saccade line
     attractor, `reasoning/nef_integrator.py::NEFIntegrator` (integration).
3. **The substrate tick.** `substrate/kernel_tick.py::KernelTick` is the governed vectorized LIF step (equation owner
   `cortex/lif_neuron.py`). It computes v = decay * v + (integer deliveries + external drive) / 2^frac_bits in float32,
   fires at threshold outside refraction, and resets to a CONSTANT `v_reset`; delays are >= 1, `decay` must lie in
   (0, 1], and weights are quantized to integers. It keeps integrating a refractory membrane, whereas
   `LIFNeuron.step` clamps it and discards the input. It is not on the live tick (S6). Measured at N = 256, one thread
   (`_bench_kernel`, two runs): 24,922 and 23,823 ticks/s at fan-out 8 (P = 1), 1,953 and 1,695 at fan-out 256, i.e.
   0.4-6 s for the 9,480 pool steps of one replay at T_in = 8. It cannot express a soft reset, a zero decay (SORN's
   memoryless threshold units) or truth tables.
4. **The percept and the learner.** x_t is 64 floats in [0, 1] (seed 100: mean 0.092, sd 0.196, L2 norm 1.645-1.747,
   no exact zeros); the learner's input contract is "normalized features [0, 1]" (`hippocampus/ng_snn.py`). A replay at
   width 256 costs under 1 s (G1(a) section 1.10). G1(a) fixed N = 256 for the pools (its section 2.2).
5. **Threads.** numpy 2.5.1 runs on OpenBLAS 0.3.33 with 24 threads when no thread variable is set
   (`probe_threads_default.out`); `threadpoolctl` and scipy are installed.
   `memory_diagnostic.replay` sets torch to one thread and refuses any other torch policy; `_policy` records no BLAS
   setting.
6. **The grower's seed variance** (`probe_contrast_sd.py` on the exploratory G1(a) pilot, seeds 100-103, per-seed means
   over the three scored segments): e1 - e1-0 sd 0.135, d - d0 sd 0.106, e2 - e2-0 sd 0.067, f64 - a sd 0.078.
7. **The revision binding.** A G1(a) replay is refused unless it runs at its recording's head and dirty flag (G1(a)
   section 3.7).
8. **Design-time observation: in this protocol the incumbent grower did not use history that a ridge readout used on
   every seed.** Proving the positive control on its generator before the freeze (obs #2198) used:
   - `probe_rb6_positive_control.py` for the grower, and `probe_rb6_reference.py` and `probe_ridge_frozen.py` (the
     frozen estimator of section 2.2, unrounded) for the ridge;
   - the source SORN task at SHARED_LEN 2: the words (0, 2, 3, 4) and (1, 2, 3, 5), each symbol a disjoint 8-dim pattern
     with the percepts' norm;
   - 1,200 events with one sleep tick after each, events 257-1,200 scored;
   - expected ceilings of 0.75 without memory and 0.875 with two percepts of history (expectations, not finite-sample
     bounds).
   On seeds 4242-4247 the results were:
   - the grower's (e1) - (e1-0): +0.003, -0.040, +0.021, -0.004, +0.160 and -0.005;
   - the grower on (f64), handed the current category, read 0.53-0.77;
   - the ridge's (e1) 0.872-0.894 against its (e1-0) 0.748-0.788, i.e. +0.106 to +0.124 on every seed;
   - R-b6's constructed red was proven on the same generator: on the history-free twin family (section 2.7) the
     ridge's (e1) - (e1-0) was -0.0064 to 0.0 on seeds 4242-4245 (`probe_rb6_red.out`).
   These are protocol-specific observations, not a proof that the grower cannot use history, because the two learners
   differ at once. The ridge refits on all past pairs, while the grower caps its buffer, withholds rows, gates growth,
   builds spiking features and prunes. The disjoint patterns and the one-sleep-per-event schedule are disclosed test
   conditions. Constant |LP| values (0.0, 0.05; `probe_rb6_lp.py`) changed nothing, but a constant cannot exercise
   `GrowthStop`'s rising-excursion branch, so the growth gate's dynamics were not tested.
9. **Design-time observation: the grower's final models were small.** It held 2-4 units at the end of every pilot
   replay read (seeds 100-103; arms a, f64, e1, e1-0; one transient maximum of 51 units, pruned back), and its
   per-refit score swung 0.38-1.00. It held 3-4 units on the synthetic stream. Capacity as the cause of section 1.8, or
   of section 1.6's variance, is a HYPOTHESIS: no capacity intervention was run.
10. **Design-time observation: on the exploratory pilot, these particular history encodings added no material average
    gain through this ridge protocol** (`probe_ridge_frozen.py`, seeds 100-103, unrounded per-seed means over the scored
    segments). (e1) - (e1-0) had mean +0.0006 and sd 0.0067; (e2) - (e2-0) had mean -0.0031 and sd 0.0043; (f64) - (a)
    had mean +0.0460 and sd 0.0436. On seed 100 the ridge on (a) alone equals the persistence floor: both are
    0.9856873 as means over the three segments (`probe_ridge_frozen.out`, `probe_territory.out`). Nonlinear
    history, other horizons and transition-conditioned predictability were not tested, and segment means can hide a
    local effect. "The stream lacks short-range structure" is therefore an unproven explanation for a future null.

## 2. Pre-registration (frozen at the commit that lands the final text)

### 2.1 Eligibility, what is compared, and the governing contract

- **The candidates are the four research-tier reservoirs** that FR-041 and T109 name. A governed structure is not a
  candidate unless its job is this one: a stimulus-driven pool carrying state from percept to percept. None of
  section 1.2's is. CriticalityRegulator monitors a regime; the microcircuit's streaming and vector input are its own
  open work (specs/019 [B2.08]); EpropLIFBase is the equation owner that candidate g2 follows; the others do other jobs.
- **What is ranked** is a complete pipeline: the shared encoder, each candidate's input rule, and its recurrent pool.
  Differences between candidates are not attributed to recurrent dynamics alone. The forms of one candidate share one
  encoder, so G1(c)'s contrast is within one pipeline.
- **The governing contract, stated once:**
  - G1(a)'s contract is extended, not altered: its arms keep the incumbent grower, and R-b8 proves their readings
    unchanged.
  - Every G1(b) contrast (D, D_task and the eligibility constraints) is read through the reference readout (section
    2.2). The grower's reading of every contrast is reported beside it.
  - The reset twin is Settled (section 2.2); literal Reset is reported.
  - The plan's G-R1 is amended in the same stack to name the decoder, the Settled twin and the blind-harness tests
    (section 2.6).
  - A G1(c) PASS licenses only "carried state improves prospective prediction through this fixed ridge decoder": never
    an incumbent-grower benefit, never adoption.

### 2.2 The pool contract (identical for every candidate)

- **Units.** Exactly N = 256 units of state, every unit counted; phi_t reads all 256 and nothing else.
- **Encoder.** One shared projection W_in (256 x 64, entries N(0, 1) / 8, `numpy.random.default_rng(seed)` with seed
  `recording_seed * 31 + 7`), identical for every candidate on a recording; u_t = W_in x_t in float64. Each
  candidate applies its own input rule (section 2.3). The pool's view declares ONLY x.
- **Stepping.** On each wake percept the pool runs T_in counted internal steps with u_t held; phi_t is each unit's
  mean activity (spike or state) over those T_in steps, in [0, 1]. Pools do not step on sleep ticks.
- **The adaptation window.** For the pool's first 256 percepts, counted by the pool itself (never from acquisition
  records, which no arm reads), every form runs as CARRIED. SORN's plasticity runs only inside this window. At the
  boundary every learned parameter is fingerprinted; the fingerprints must be identical across a candidate's forms
  (Front-end excepted, below) and unchanged to the end. So the forms of one candidate share the same learner state at
  the boundary and differ only afterwards.
- **Forms, after the window.**
  - **Carried:** the state persists from percept to percept.
  - **Reset:** every mutable activity carrier (section 2.3) returns to its declared initial value; then T_in counted
    steps. Reported only.
  - **Settled** (the primary reset twin): as Reset, then S_settle = 64 uncounted steps on u_t, then T_in counted steps.
    It removes a startup transient without carrying anything across percepts. No monotone relation between settling
    and accuracy is claimed. A Carried-minus-Settled gain licenses nothing about cross-percept information until G1(c)'s
    within-block time shuffle.
  - **Front-end** (reported, never an eligibility input): a lesion applied to a copy of the common adapted state at the
    boundary, stepped as Carried:
    - g1 and g2 have every recurrent synapse zeroed, but the membrane leak and reset stay, so the control is
      zero-synapse but STATEFUL, not memoryless;
    - g3 has W_EE, W_EI and W_IE zeroed, which leaves x_E = Theta(g u - T_E), x_I = Theta(-T_I);
    - g4's non-input nodes hold their boundary state, and its input nodes apply only their input bits.
    Its expected parameter differences are recorded separately from freeze violations.
- **The startup diagnostic (reported).** For every configuration and realization, 16 fixture percepts are used. They
  come from one generator, `numpy.random.default_rng(777)`: 16 successive draws r = `random(64)`, each fixture
  `clip(v * 1.73 / |v|, 0, 1)` with v = r ** 4.
  Each is driven from reset for 256 steps to obtain its long-run mean activity r_j (the mean over steps 257-320). The
  transient at S in {0, 16, 64} is the maximum over fixtures of the mean over units of |phi_{j,S} - r_j|. A
  configuration whose transient at S = 64 exceeds 0.05 is flagged non-convergent (periodic or irregular), not
  refused. Its D is reported with that flag.
- **Realizations.** A candidate's random structure comes from seed `recording_seed * 31 + 11 + index` (index 1-4 in
  section 2.3's order). Each recording gets a fresh realization, so the benchmark selects a mechanism and its settings,
  not one draw.
- **Arithmetic (the pools).** float64 numpy. No BLAS product (`@`, `dot`, `matmul`, `einsum`) in any pool's per-step
  or per-percept computation. (The reference readout's prediction `x @ W` and its refit solve are BLAS/LAPACK calls;
  every replay pins them to one BLAS thread and records it, so a replay is deterministic, and R-b4 checks that.)
  Every reduction in a pool is a SEQUENTIAL sum in a fixed, declared order: `np.cumsum`'s last element along
  the declared axis, or an explicit loop over terms vectorized across units. NumPy's ufuncs and `cumsum` run
  single-threaded, so the result does not depend on the thread count. `np.sum`, `np.add.reduce` and `np.mean` sum
  pairwise on float axes and are not used on a state-producing path. Where R-b0 claims bit parity, the order is the
  source's (section 2.3). Construction-time linear algebra (g2's `eigvals`) runs inside
  `threadpoolctl.threadpool_limits(1)`. The encoder's u = W_in x is a sequential sum over the 64 inputs.
- **Two learners read every replay.**
  - **The incumbent grower** is G1(a)'s: `NeurogenesisSelfImprove` from the recorded constructor arguments, CPU, delta
    online readout, width 256, CAUSAL |LP|.
  - **The reference readout** (section 1.8 is why it exists) is an online ridge, specified exactly:
    - at event t >= 2 it predicts c_t from x = [phi_{t-1}, 1] as the argmax of x @ W over W's columns, which are the
      categories present at the last refit, sorted ascending (the first maximum wins, so an exact tie goes to the
      lowest id); a category first seen after the last refit cannot be predicted before the next;
    - it predicts None before its first fit, and at event 1 (there is no phi_0);
    - after predicting it adds the pair: XtX += `np.multiply.outer(x, x)` and Xty[c_t] += x, accumulated in event order
      in float64;
    - after every 20th pair it refits W = solve(XtX + P, [Xty[k] for k in the sorted seen categories]), with P = 1.0 * I
      except a zero at the bias (an unpenalized bias), by LAPACK at one BLAS thread, recorded;
    - its state (XtX, Xty, W) is fingerprinted separately from the grower's, and it never touches the grower;
    - the design probes' estimator is this one (`probe_ridge_frozen.py`), and section 1.10's numbers come from it.
  - **The readout's policy disposition.** It is a governed benchmark component with no live deployment (P4c: it is
    part of the brain's tree and gated with it). Its direct solve is the class the e-prop invariant's standing verdict
    already admits (non-locality is not backprop; no gradient is transported through time). P1 does not bind lambda =
    1.0 and the refit period 20, because the readout arbitrates no behavior: they are fixed estimator settings of a
    measurement, disclosed here.

### 2.3 The candidates (exact rules; tiny hand-computed fixtures in the tests)

For every candidate, the realization seed is section 2.2's, and the initial state before the window is the reset state
below unless stated. Each item gives the drawing, the per-step update, the reset state and the grid.

- **g1, LSM (`spiking_ssm`).** Carriers v (float64), s (bool).
  - Drawing: `random.Random(realization_seed)` draws exactly as `spiking_ssm.build_reservoir` does. The N draws
    `uniform(-1, 1)` of its `w_in` are consumed first and unused; then, per unit in order, k pairs
    (`randrange(N)`, `uniform(-w, w)`) (the source's w is 0.5).
  - Per step: rec_i is a sequential accumulation into 0.0 over the unit's stored incoming list, adding w where s_j.
    Then v_i <- (lambda v_i + g u_i) + rec_i; s_i <- v_i >= 1; v_i <- 0 where s_i.
  - Reset: v = 0, s = 0.
  - Grid: k {8, 32}; w {0.25, 0.5, 1.0}; lambda {0.5, 0.7, 0.9}; g {1, 2, 4, 8}; T_in {1, 2, 4, 8}.
- **g2, ESN-LSM (`reservoir_eprop`'s fixed reservoir, EpropLIFBase's equation; soft reset only).** Carriers v, z.
  - Drawing: W_rec is g1's drawing at k = 8 and w = 0.5 (the source's), densified `[pre, post]`, zero diagonal, and
    rescaled to spectral radius rho, with `eigvals` under `threadpool_limits(1)`.
  - Per step: cur_i = (sequential sum over pre, in index order, of z_prev[pre] * W_rec[pre, i]) + g u_i; then
    v <- ALPHA v - V_TH z_prev + cur and z <- v >= V_TH, with ALPHA 0.90 and V_TH 0.6.
  - Reset: v = 0, z = 0.
  - Grid: rho {0.5, 0.9, 1.2}; g {1, 2, 4, 8}; T_in {1, 2, 4, 8}.
- **g3, SORN (`sorn`).** 213 E + 43 I units. Carriers x_E, x_I; learned parameters W_EE and T_E, frozen after the window.
  - Drawing: `random.Random(realization_seed)` draws exactly as `sorn.build_net` does at N_E = 213 and N_I = 43. That is
    its call order, including the input-block shuffle (consumed, unused), with the source's P_EE 0.10, P_IE 0.20,
    P_EI 0.30, threshold ranges and normalized rows.
  - Per step: the source's `_step` and, inside the window, `self_organize`'s STDP, synaptic normalization (a sequential
    sum) and intrinsic plasticity, with no noise units. The source's drive (1.0 into an 8-unit block per symbol)
    becomes the E units' drive g * u[0:213]: a dense signed drive, not a symbolic one (disclosed). phi is ordered E
    (213) then I (43).
  - Initial state before the window: all zeros, as the source's `self_organize` starts.
  - Reset: one binary state drawn once by `random.Random(realization_seed + 2000)`, each unit active with probability
    H_IP.
  - Grid: plasticity {off, (eta_stdp, eta_ip) in {0.001, 0.004} x {0.001, 0.01}}; H_IP {0.05, 0.1, 0.2};
    g {1, 2, 4, 8}; T_in {1, 2, 4, 8}.
  - "off" is the static twin. The result speaks only for this short-adaptation variant: at most 2,048 plastic steps,
    against the source's 4,800. Parameter movement across the window is reported.
- **g4, RBN (`random_boolean_network`).** 256 nodes. Carrier: the state bits.
  - Drawing: `random.Random(realization_seed)` draws exactly as `build_rbn` does, with K and p. After it, the input
    nodes are the first round(f * 256) of a permutation drawn by `random.Random(realization_seed + 1000)`.
  - Per step: the synchronous update of every node from its K inputs' previous values. Then, on the input nodes, the
    input bits b_i = [u_i > tau] are applied by XOR or by CLAMP.
  - Reset and initial state: one bit state drawn once by `random.Random(realization_seed + 2000)` (fair bits).
  - There is no gain: only tau acts on the bits.
  - Grid: K {1, 2, 3}; p {0.5, 0.7}; f {0.1, 0.25, 0.5}; mode {xor, clamp}; tau {-0.1, 0, 0.1}; T_in {1, 2, 4, 8}.

### 2.4 The manifest, the budget and the resource bound

- **The manifest.** Each grid is enumerated in the order written in section 2.3, with the last factor varying fastest.
  The grids as written contain no two configurations with identical effect, so there is no deduplication. The draw is
  a without-replacement sequence from `random.Random(2026 + index)`: `sample(grid, len(grid))`, read in order. The
  first 32 draws that survive the resource screen are the candidate's configurations; a whole grid smaller than that
  is disclosed. The manifest's crc32 is taken over `json.dumps(manifest, sort_keys=True, separators=(',', ':'))`. It
  is recorded in section 3 before any tuning replay, and the build computes it from this text.
- **The budget is an equal EVALUATION count** (32 configurations x 2 forms on each tuning recording and synthetic
  stream), not equal search coverage: the grids differ in size.
- **The cost scalar and the bound.** The cost of a configuration is the median, over three repetitions, of the
  per-percept wall time of the pool alone in the Carried form (projection included). It is measured over percepts
  257-end of seed 200's recording, at one thread on the CPU. Initialization, the adaptation window, Settled's cost and
  end-to-end replay times are reported separately. A drawn configuration whose cost exceeds 25 ms is dropped,
  disclosed with its number, and the next draw takes its place; the bound is intentionally a seed-200 pool bound. (A
  live wake tick costs about 55 ms, G1(a) section 1.2.) The tie-break uses the same scalar, and costs within 10% of each
  other are tied. Costs are CPU-only; no P5 speed claim is made.

### 2.5 The statistic and the rule

- **D** is the reference readout's accuracy(Carried) minus accuracy(Settled), where a None prediction is a miss. On a
  live recording it is the mean over the recording's available scored segments (G1(a) section 2.3); a recording with
  none is a harness defect and halts the campaign. D for a phase is the mean over its four recordings.
- **D_task** is the same difference on the synthetic family of R-b6 (section 2.7), taken over each stream's 944 scored
  events and then averaged over four seeds: 5001-5004 in tuning, 5005-5008 in selection. A synthetic stream's seed
  stands for the recording seed in section 2.2's W_in and realization seeds. When the live D values are near-tied,
  which section 1.10 predicts, the ranking selects for this one synthetic task family.
- **Eligibility.** A configuration (in tuning) or a tuned candidate (in selection) is eligible only if all of these
  hold:
  - every job it needs completed OK;
  - non-inferiority on the phase's LIVE recordings only: the mean of its Carried accuracy is at least that of (e1-0),
    through the same readout, minus 0.05;
  - on no live recording of the phase is its Carried phi degenerate over the scored rows. Degenerate means at least 90%
    of units constant (one value on every scored row), or fewer than two distinct phi rows.
  Front-end never enters eligibility.
- **The ordering** (used in tuning and in selection alike):
  1. Rank by D. Entries within band_D = max(0.005, sigma_dev) of the best D are near-tied. sigma_dev is section 2.6's;
     sigma_dev / sqrt(4) is one standard error of a four-recording mean, and 2 * sigma_dev / sqrt(4) = sigma_dev.
  2. Among the near-tied, rank by D_task. Entries within 0.005 of the best D_task are still tied.
  3. Among those, the lower cost wins, with costs within 10% of each other tied.
  4. A remaining tie goes to manifest order (tuning) or to the lower index (selection).
  The ranking is descriptive: no inferential tie or equivalence is claimed.
- **Tuning** applies the ordering, per candidate, to its eligible configurations on 200-203 (with D_task on 5001-5004).
- **Selection** applies it to the tuned candidates on 204-207 (with D_task on 5005-5008). A candidate with no
  eligible configuration is refused, and if none is eligible nothing is selected.
- **Reported beside D, never used to select:**
  - Carried, Reset, Settled and Front-end accuracy per seed and segment, through both learners;
  - the grower's D and D_task;
  - coverage, prediction entropy, the prior-fallback rate;
  - the costs;
  - (e1-0) and (e1) at width 256 through both learners;
  - the degeneracy and startup diagnostics;
  - the input-response statistics: each candidate's driven-unit mask, and the mean and sd of its per-unit drive;
  - SORN's parameter movement.

### 2.6 G1(c)'s instrument, decision rule and power (frozen here, before any selection data exist)

- **Instrument.** G1(c) reads its primary endpoint, Carried minus Settled, through the reference readout, with the
  grower's reading secondary and descriptive (section 2.1). G1(c)'s own pre-registration may only strengthen this.
- **Decision.** The endpoint passes when its one-sided paired t over S seeds exceeds the alpha = 0.05 critical value at
  S - 1 d.f., i.e. a gain detectably above zero. The null boundary is 0. Delta = 0.02 is the design effect, the smallest
  gain worth detecting; it is not an acceptance threshold. The estimate and its one-sided 95% lower bound are reported
  against Delta.
- **The blind-harness tests** (G-R1's amended REFUSED clause) are:
  - the reference readout fails R-b6's history positive control on G1(c)'s synthetic seed, 6000;
  - the persistence oracle (f64) does not beat the incumbent arm (a) through the readout, by the same paired t.
- **Power.** S is the smallest S >= 8 with (t_{0.95,S-1} + t_{0.80,S-1}) * sigma_plan / Delta <= sqrt(S), capped at 64.
  - sigma_plan = max(0.0436, sigma_dev). 0.0436 is a planning PROXY: the largest per-seed SD of a width-matched
    contrast through the frozen readout on the exploratory pilot (section 1.10, unrounded). It is labeled as an
    assumption, not shown to be conservative for the reservoir contrast.
  - sigma_dev is the largest per-seed SD of Carried minus Settled through the readout, across the four candidates'
    first surviving manifest configurations. It is measured on the development seeds 100-103, re-recorded at the build
    head, after the build and before any tuning.
  - At sigma_plan = 0.0436 the rule gives S = 31; a binding cap is disclosed as reduced power.
  - Sensitivity (the same rule):
    - sigma 0.02: Delta 0.01 gives 27, Delta 0.02 gives 8, Delta 0.03 gives 8;
    - sigma 0.0436: Delta 0.01 gives more than 64, Delta 0.02 gives 31, Delta 0.03 gives 15;
    - sigma 0.08: Delta 0.01 or 0.02 give more than 64, Delta 0.03 gives 46.

### 2.7 Checks, each with a constructed red and a witness

Mechanism checks (R-b0, R-b0x, R-b3, R-b5) run on development fixtures, over every manifest configuration where they
apply, BEFORE any tuning. Every red asserts its specified observable divergence.

- **R-b0, parity with the sources.** g1, g3 and g4 reproduce their sources bit for bit in float64; g2 is as stated:
  - g1: `spiking_ssm.run_reservoir`'s rates and SynOps (N = 100, scalar drive, reset between sequences);
  - g2: identical spike trains and membranes within 1e-12 of `reservoir_eprop`'s reservoir forward with the readout
    inert (numpy float64, whose `@` is a BLAS product with its own association). Also, the port run in float32 must
    give spike trains identical to `EpropLIFBase.forward` (W_out zero) on a declared fixture whose membranes stay at
    least 1e-4 from V_TH;
  - g3: `sorn._step` states and `sorn.self_organize` weights and thresholds on the source's stream (noise included for
    this fixture only);
  - g4: `random_boolean_network.step` with no input.
  Each red perturbs one constant that its fixture provably exercises (a leak, a threshold, a truth-table entry the
  fixture addresses, a plasticity rate used inside the fixture window) and must break parity.
- **R-b0x, adapter fixtures.** Independent hand-computed fixtures (N <= 8) cover what the sources never had: vector
  drive, continuous stepping across percepts, each reset form, T_in averaging, RBN XOR and CLAMP timing and input-node
  selection, the window boundary, the Front-end lesions, and the reference readout (a hand-computed fit, the
  seen-so-far rule, the bias treatment). Reds: each fixture with its rule deliberately wrong. A future-only category
  mutation must never be predicted before its first appearance.
- **R-b1, causality.** G1(a)'s R-a2 on every candidate's Carried and Settled forms at its tuned configuration, on seed
  203 (a tuning recording, so no selection-seed artifact exists before selection), at the three cuts, for both
  learners. Red: (leak) changes at every cut.
- **R-b2, channels.** Every pool runs through a view that declares only x. Red: a (forbidden) pool asking for c raises.
- **R-b3, determinism by construction** (a pool-only instrument, outside `replay`, which pins torch). Each manifest
  configuration is driven by the startup diagnostic's 16 fixture percepts cycled to 512 percepts (256 through the
  window, 256 after), in its Carried and Settled forms, under `threadpoolctl` at BLAS 1 and 24 threads and torch 1 and
  4. The loaded backend and the effective thread counts are recorded. The internal activity sequence of every step and
  the complete continuation state must be identical. That equality runs on every manifest configuration. The reds
  run once per mechanism, on a declared witness configuration whose fixture is asserted to exercise the fault:
  - g1 and g2 (float membranes; order matters only when a unit receives at least three presynaptic spikes at once):
    the recurrent sum taken in an order permuted by an unseeded source must differ in the continuation state. The
    witness is g1 with k 32, w 1.0, lambda 0.9, g 8, T_in 8, and g2 with rho 1.2, g 8, T_in 8. The fixture must be
    asserted to contain at least 100 unit-steps with three or more simultaneous presynaptic spikes.
  - g3 and g4 (binary state, where a one-ulp reordering almost never flips a bit): an asynchronous in-place update must
    differ in the continuation state.
  A candidate whose unmodified configuration fails the equality is refused or repaired before tuning; there is no
  single-thread waiver.
- **R-b4, replay determinism.** Two fresh-process replays give identical phi fingerprints, both learners'
  fingerprints and both learners' predictions, for:
  - each candidate's Carried form at its tuned configuration, on seed 203;
  - the winner's Carried and Settled forms, on all four selection recordings, after selection.
  Red: (pool-noise), which adds unseeded `os.urandom` noise to one float membrane (g1, g2) or flips one bit chosen by
  `os.urandom` (g3, g4). It must change the continuation state AND the phi fingerprints.
- **R-b5, the forms do what they claim.**
  - (i) History erasure: after the window, two different random prefixes followed by the same Reset or Settled and the
    same percept give identical phi and continuation state. Red: a reset that skips one carrier must fail.
  - (ii) History persistence: on a fixture configuration known to carry state (g1 with k 8, w 0.5, lambda 0.9, g 2,
    T_in 1), Carried after two different prefixes must give different continuation states. Red: a "carried" form that resets every percept must
    fail. Each configuration's own persistence is reported.
  - (iii) Freeze: every learned parameter at the end equals its boundary fingerprint. Red: SORN with plasticity left
    on after the window must fail.
- **R-b6, the instrument can see history** (a positive control that is not the method, proven on its generator at
  design time, sections 1.8 and 1.10; the generator is `archive/g1b-2026-09-25/probe_rb6_positive_control.py::
  synthetic_stream`). The synthetic stream family uses the source SORN task at SHARED_LEN 2:
  - whole words are appended, each `WORDS[rng.random() < 0.5]` with rng = `random.Random(seed)` and
    WORDS = ((0, 2, 3, 4), (1, 2, 3, 5)), until there are at least 1,200 symbols, then truncated to 1,200;
  - symbol k is the disjoint block `dims[8k : 8k + 8]` of dims = `numpy.random.default_rng(seed).permutation(64)`,
    valued 1.73 / sqrt(8), and its category is k;
  - event i has tick 2i + 1 and one sleep tick at 2i + 2, and every |LP| is a recorded None;
  - the first 256 events are labeled warm-up and the rest "unseen-order";
  - the grower's recorded configuration is seed = the stream seed, refit_every 20, buffer_cap 1000, l_max None,
    eps_tol 0.28, T 15.
  On seed 4242 (and, in G1(c), seed 6000), the reference readout's (e1), holding three past percepts, must beat its
  (e1-0) by at least 0.05. At design time it did: +0.124. If it does not, the harness is blind to history and nothing
  is selected. The grower's reading on the same stream is recorded, not required. Red: on the history-free twin family
  (WORDS = ((0, 2, 3, 4), (1, 2, 3, 4)), where the next symbol never depends on the past beyond the current one), the
  same comparison must give (e1) - (e1-0) < 0.05.
- **R-b7, binding.** Every result carries the recording's identity, the candidate, the form, the configuration hash,
  the realization, W_in and topology fingerprints, the initialization, the numeric policy (with BLAS) and the revision.
  Coverage is asserted unique and complete. Reds: a replay from a wrong configuration, a duplicate seed, and a
  tuning-seed replay in the selection table must each be refused.
- **R-b8, the extension changed no G1(a) measurement.** It has two halves, each with its red:
  - Pre-build: seed 100's recording, re-recorded at the build head, has the same `_source_digest` as the G1(a) pilot
    recording at `829fb40` (archived in `archive/g1a-2026-09-24/pilot3_full.tar.gz`). Every G1(a) scored arm's CAUSAL
    replay gives predictions, phi fingerprints and learner fingerprints equal to that pilot's pinned replays. Red: a
    one-element change to `_feature("e1")` must turn this half red.
  - With and without the readout: the same replays with the reference readout on and off give identical grower
    predictions, phi fingerprints and learner fingerprints. Red: a reference readout that writes into the grower's
    input must change them.

### 2.8 Failures and repairs

Every check can fail in two ways: its green fails, or its red does not fire. Each is classified by the part it
implicates.
- **The shared instrument fails.** This covers any failure in the reference readout, the scorer, the harness, the
  view, the binding, the comparators or the recorder:
  - R-b0x's readout fixtures;
  - a red that does not fire;
  - R-b1, R-b2, R-b6, R-b7 or R-b8 in either mode;
  - an R-b4 difference in the readout's or the grower's state.
  Selection HALTS with nothing selected. A broken test never selects a different winner, and never the runner-up.
- **A candidate fails.** This is a failure of one pool rule's green: its R-b0, its R-b0x pool fixtures, its R-b3
  equality, its R-b4 phi difference, or its R-b5. That candidate alone is refused, and only once the shared instrument
  has shown itself sighted: every shared red fired.
- **Jobs.** A FAIL is a defect: the campaign aborts, the defect is fixed, and every recording and replay re-runs at the
  new head (section 1.7). A TIMEOUT makes that configuration ineligible; seeds are never averaged over what completed.
  No retries.
- **Inspection and repair.** Selection outcomes are "inspected" as soon as any REPLAY of a selection recording
  (204-207) or of a selection stream (5005-5008) has been written. The recordings themselves hold no outcome. R-b1 and
  the per-candidate R-b4 run on tuning seed 203, so they write none. A repair after inspection, including one forced by the winner's post-selection R-b4, exposes
  204-207 and 5005-5008. Selection then re-runs on the reserved 208-211 and 5009-5012. A second such repair halts
  G1(b).

### 2.9 Dispositions

- **The winner** is G1(c)'s pool at its tuned configuration. Nothing here is confirmatory: the selection licenses no
  claim that carried state helps (G1(c)'s primary endpoint) and no adoption.
- **The losing ports** are deleted from the pool module after the selection record; the record names the commit
  that held them. The four research SOURCES are not touched in this arc. Each one's disposition (retire, promote or
  exempt, with its unique capability, consumer and regression proof) is a separate P2 row, filed as an opening.
- **The governed host of an LIF winner.** If g1 or g2 wins, G1(c) passes a migration gate before its confirmatory run.
  g1's host is `KernelTick`; g2's is EpropLIFBase's soft-reset equation run as a streaming step. The gate:
  - parity fixtures in the host's arithmetic against the benchmark port (for g1: refraction, the threshold boundary,
    delay 1, weight quantization and float32 membranes; for g2: float32 and the torch product's association);
  - D re-measured on 204-207 with the hosted form, within 0.005 of the port's;
  - R-b3 and R-b4 on the hosted form.
  Failing the gate, G1(c) runs the port and the migration is an opening. KernelTick cannot express g3 or g4, and for
  them the disposition says so.
- **P3.** The pool module is registered in `REGIONS` and `run_selfchecks`; its consumers are the harness's pool arms,
  the campaign entrypoints and the tests; its dormancy row is EXEMPT as a governed benchmark component until G1(c)'s
  disposition.

### 2.10 Seeds, the run matrix and the budget

- **Seeds.**
  - Live recordings: tuning 200-203, selection 204-207, reserved 208-211; G1(c) from 300.
  - Synthetic: R-b6 on 4242 (and G1(c)'s on 6000); D_task tuning 5001-5004, selection 5005-5008, reserved 5009-5012.
  - Development readings only (sigma_dev and R-b8): 100-103, re-recorded at the build head.
  - Exposed, and never used for a selection or confirmatory reading: 0-7, 100-103, and 4242-4247 (the design probes).
- **One head.** Every recording, replay and check of this arc runs at one clean head (section 1.7). A code change
  after the recordings re-records.
- **Order.**
  1. The recordings: 100-103 and 200-207.
  2. The mechanism checks on fixtures: R-b0, R-b0x, R-b3, R-b5.
  3. R-b6 and its red, and R-b8.
  4. sigma_dev on 100-103, and the resource screen on 200.
  5. The manifest's crc32, recorded in section 3.
  6. Tuning on 200-203 and 5001-5004.
  7. R-b1 and the per-candidate R-b4 on 203.
  8. Selection on 204-207 and 5005-5008.
  9. The winner's R-b4 on 204-207.
  The first replay of a selection seed happens at step 8.
- **Matrix.**
  - 12 recordings (8 live, 4 development), about 130 s each; the resource screen, at most 128 configurations; the
    mechanism checks on fixtures.
  - Tuning: 4 candidates x 32 configurations x 2 forms x (4 recordings + 4 synthetic streams) = 2,048 replays.
  - Selection: 4 x 4 forms x 4 recordings, plus 4 x 2 forms x 4 synthetic streams, plus the references.
  - Checks: R-b1 4 x 2 x 3 = 24 on seed 203; R-b4's twins (203, then the winner's on 204-207); R-b6 and its red;
    R-b8's two halves.
  Each replay takes about 1-15 s.
- **Limits.** 4 worker processes; torch and BLAS at one thread (except R-b3); a 900 s timeout per process;
  temporary homes removed; `dirty` beside `head` in every artifact.

### 2.11 Scope

- **Built here:** one governed pool module (the four candidates as streaming pools, the encoder, the forms, the
  startup diagnostic, the manifest, and R-b0 to R-b7's machinery); the reference readout in the harness; the harness's
  pool arms; the campaign entrypoints (record, screen, tune, select, check); tests; and the module's self-check, which
  constructs every red on synthetic fixtures.
- **Not built here:** any live wiring; any change to the four sources, to `CriticalityRegulator`, to `EpropLIFBase`
  or to `KernelTick`; the within-block time shuffle (G1(c)).

### 2.12 The design reads and their folds

**Read 1.** Codex (`gpt-6-astra`, high effort, read-only, via claude-octopus) read the draft at `fd89e54` and returned
24 findings: 3 BLOCKER, 20 MAJOR, 1 MINOR. The orchestrator reproduced the decisive code facts first: SORN's six-symbol
task and sizes; the pure arm's trained `win`; the trainer's path; EpropLIFBase's soft-reset equation; the replay's torch
pin; the missing BLAS policy; OpenBLAS at 24 threads. The fold:
- F1 (BLOCKER) -> 2.9: no source retires here.
- F2 -> 1.2 and 2.1: CriticalityRegulator is not a candidate.
- F3 -> 2.1: the eligibility rule.
- F4 -> 1.2 and 2.3: the inventory, and g2 follows EpropLIFBase.
- F5 -> 1.1: corrections.
- F6 (BLOCKER) -> 1.3 and 2.9: the KernelTick migration gate.
- F7 (BLOCKER) -> R-b3.
- F8 -> 1.5 and R-b3.
- F9 -> 2.3 and R-b5.
- F10 -> 2.2's window.
- F11 -> 2.2's Settled.
- F12 -> 2.3.
- F13 -> 2.1 and 2.2's Front-end.
- F14 -> 2.3 and 2.4.
- F15 -> 2.3.
- F16 -> 2.5.
- F17 -> 2.5.
- F18 -> 2.6.
- F19 -> 2.4.
- F20 and F23 -> 2.8.
- F21 -> R-b0x.
- F22 -> R-b0, R-b4, R-b5, R-b6 and 2.5.
- F24 -> R-b7.

**Read 2** (the same channel, on the fold) marked 18 of the 24 CLOSED and 6 NOT CLOSED (F7, F11, F13, F18, F19, F22),
and filed 15 new findings: 2 BLOCKER, 13 MAJOR. All are folded here:
- N1 (BLOCKER) -> 2.1's governing contract, 2.6, and the plan's G-R1 amendment in the same stack.
- N2 -> 2.2's policy disposition.
- N3 -> 1.8-1.9: bounded observations and hypotheses.
- N4 -> 1.10.
- N5 -> 2.2's exact estimator, R-b0x, and `probe_ridge_frozen.py` re-run unrounded.
- N6 -> 2.5 and 2.10: fresh synthetic seeds and the exposed list.
- N7 and F18 -> 2.6: the proxy labeled, sigma_dev added, sensitivity published.
- N8 -> 2.6's decision rule and t-based power.
- N9 and F11 -> 2.2: Settled without a conservativeness claim; the startup diagnostic; the shuffle requirement.
- N10 and F13 -> 2.2's Front-end.
- N11 (BLOCKER) and F7 -> R-b3's per-mechanism reds.
- N12 -> 2.7's order and 2.8's repair rule.
- N13 -> 2.5's reductions.
- N14 and F19 -> 2.4's cost scalar.
- N15 -> 2.3: g2 soft reset only.
- F22's remaining faults -> R-b5 (ii) and (iii).
Both reviews are archived as `archive/g1b-2026-09-25/design_read.txt` and `design_read2.txt`.

**Read 3.** The owner paused Codex after read 2, so a Claude Opus reviewer (read-only, `model: 'opus'`) read the fold.
It marked every read-2 item CLOSED except F7 and N11 (R-b3's order red cannot fire for SORN) and a residual of N6
(4242 missing from the exposed list). It filed 12 findings: 1 BLOCKER, 7 MAJOR, 4 MINOR. The orchestrator
reproduced the checkable claims first: read 2's tally really was 18 and 6; SORN's binary state; the missing archived
outputs, now produced; seed 100's floor. The fold:
- F1 (BLOCKER) -> R-b3's per-mechanism witness reds, and R-b4's bit-flip noise for g3 and g4.
- F2 -> 2.5's band_D.
- F3 -> R-b8's pre-build half.
- F4 -> 2.8's classification and the move of R-b1 and R-b4 to seed 203.
- F5 -> 2.9's gate for either LIF winner, and R-b0's g2 leg.
- F6 -> R-b6's red, generator and grower configuration, and 2.10's seed lists. The red was proven on its generator
  (section 1.8).
- F7 -> the section 1 header, the archived outputs, and 1.3, 1.5 and 1.10 restated from them.
- F8 -> 2.2 and 2.3's exact draws, sums, states and seeds, and 2.4's manifest.
- F9 -> 2.5's degeneracy test.
- F10 -> this tally.
- F11 -> the plan note ("G1(b) and G1(c)").
- F12 -> 2.4's cost ties and replacement draws.
Two orchestrator items were folded with them: sequential sums in the sources' order (2.2), and g2's tolerance parity
and pinned construction (2.3 and R-b0). The review is archived as `archive/g1b-2026-09-25/design_read3.txt`.

### 2.13 Design-time findings and the redesign they forced (disclosed)

Proving R-b6's positive control on its generator before the freeze (obs #2198) produced sections 1.8-1.10. Read through
the incumbent grower alone, this benchmark, and G1(c) after it, would have measured an instrument that did not
extract history in this protocol, and whose seed variance (sd 0.135) swamps any plausible pool effect. The redesign
keeps the grower as G1(a) froze it and adds the reference readout (section 2.2). The selection reads D through that
readout, with D_task as the tie-breaker (section 2.5); R-b6 is re-based on the readout; and G1(c)'s instrument,
decision rule and power follow (section 2.6). Filed as openings (carried-openings file, 2026-09-25): the grower's
behavior in sections 1.8-1.9, for the grower's own arc and the plan's G2/G3 rows; Part F's blind-harness clause; the
live-stream observation of section 1.10 as G1(c)'s design input; and the four sources' P2 rows.

## 3. Results

### 3.1 The build, part A: the pool module (`fd953ec`)

`cortex/recurrent_pool.py` (1,614 lines, over the 800-line soft ceiling; its docstring gives the reason: one cohesive
component) and `tests/test_recurrent_pool.py` (61 tests). Built by a Claude Opus builder from a bounded brief (the owner
paused Codex); verified by the orchestrator: the self-check (5.9 s), the 61 tests, ruff, format, both dormancy gates
and the dependency ledger, plus a read of the four step rules, the forms and the window against section 2.3 and the
sources. `threadpoolctl` became a declared core dependency: the ledger reddened, because it had been installed only
through scikit-learn.
- **R-b0 at build time:**
  - g1, g3 and g4 are bit for bit. SORN's states over 240 steps match, and so do its weights and thresholds after
    13,404 STDP events.
  - g2 matches the numpy source with a weight gap of 0.0, a membrane gap of 1.8e-15 and identical spikes.
  - g2's float32 port is spike-identical to `EpropLIFBase.forward` with a 4.4e-4 margin.
- **The draw orders' crc32s:** lsm 1691104335, esn 176457545, sorn 489683113, rbn 959998414. The grid sizes are 288, 48,
  240 and 432.

**Deviations and declarations (section 2 left each open, or the source overrode its words):**
1. **SORN's row normalization is CPython's builtin `sum()`**, which has been compensated (Neumaier) since 3.12. It is
   not the "sequential sum" section 2.3 wrote. A plain running sum differs on about half of the rows and breaks R-b0,
   so the port reproduces `sum()` exactly, and a test pins it. The orchestrator confirmed: `sum([0.1] * 10 + [1e16,
   1.0, -1e16])` gives 2.0, where the running sum gives 0.0.
2. **Silent terms.** g1 and g3 add +0.0 where the sources skip, and g2 sums only the spiking rows. Neither changes a
   partial sum.
3. **Draws the spec left open:**
   - g4's input-node permutation is `random.Random(realization_seed + 1000).sample(range(256), 256)`.
   - SORN's reset state draws `rng.random() < H_IP` per unit, E units first; RBN's draws `randrange(2)` per node.
4. **g4's Front-end:** every node holds its state, then its input bits act by the configured mode.
5. **What is fingerprinted.** The boundary fingerprint covers every parameter, a superset of the learned ones.
   Front-end's lesion fingerprint and the parameters it changed are recorded apart from freeze violations. The
   continuation fingerprint includes the percept count.
6. **How R-b0's g2 leg reads the sources.** It reads their per-step v and z by `sys.settrace`; the sources are not
   edited. Its float32 fixture was declared from a scan of 18 choices (all spike-identical; 15 met the 1e-4 margin):
   g 4, the first 4 fixture percepts each held 8 steps, rho 0.9, seed 100.
7. **Witnesses the spec named only by mechanism.**
   - R-b3, g3: {plasticity (0.004, 0.01), H_IP 0.1, g 2, T_in 1}.
   - R-b3, g4: {K 2, p 0.5, f 0.25, xor, tau 0, T_in 1}.
   - Both use an os.urandom-ordered in-place update, counted by reads of neighbors already changed that step.
   - The R-b5 (iii) freeze red uses g3's R-b3 witness.
   - The erasure prefixes are `default_rng(1)` and `default_rng(2)`, 8 percepts each. The check reports whether the two
     branches diverged, so a vacuous pass is visible.
8. **R-b5 (i)'s red fires for every carrier but one.** The exception is SORN's x_i in the Reset form at T_in 1: the old
   x_i is overwritten before it is read, so skipping its reset changes nothing. That is correct, not blindness. The red
   is witnessed on SORN's x_E.
9. **The startup diagnostic** runs on copies of the pool with plasticity off. On the g1 persistence witness it flags
   transient(64) = 0.170: at T_in 1, a binary phi cannot equal a fractional long-run mean. Configurations with T_in 1
   and binary states will be flagged, which is reported and not refused (section 2.2).
10. **The self-check uses reduced sizes** for R-b3 and R-b5 (64 percepts, window 32) to stay under 20 s; the tests use
    section 2's sizes. tau keeps the grid's literal `0`, an int, which the crc32s include.
11. **Campaign order** (a deviation in the orchestrator's brief for part B, recorded here). Section 2.10's order lists
    sigma_dev and R-b3 before the resource screen, but both need its survivors ("first surviving manifest
    configurations", "every manifest configuration"). The campaign therefore runs the screen, then R-b3's equality,
    then sigma_dev, all before any tuning, as section 2.7 requires.
