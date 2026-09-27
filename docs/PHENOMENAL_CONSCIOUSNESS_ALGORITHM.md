# ACS-LOOP: one algorithm from six theories, and the eleven things it cannot do yet

Design pass 2026-08-02. Successor to `docs/CONSCIOUSNESS_METHODS.md`, which queued a single build
(Rolls' higher-order syntactic thought) and named the dormancy it would close. This document
generalises that pass across six theories, collapses them into one mechanism, and then spends most
of its length on why the mechanism is not yet buildable here.

**Nothing here is built. Nothing here claims the brain is conscious. Nothing here can be made to
claim it.**

---

## 0. What this is, and what it is not

### What it is

A **precision-gated control loop with a separately-parameterized model of its own control state**,
plus the ablation protocol that would test whether that model is load-bearing. Three things the
brain would gain if it were built and if the ablation came back positive:

1. **Endogenous attention control** -- a top-down goal in the attention gate, which
   `cortex/brain.py:1128` currently passes as `goal=None` with a comment saying the goal vector does
   not exist yet.
2. **Per-step credit assignment on a multi-step plan** -- the capability
   `docs/CONSCIOUSNESS_METHODS.md` queued, which the brain lacks entirely.
3. **A calibrated confidence signal with one named source**, replacing three unranked ones.

Every one of those is worth building with the consciousness framing deleted. That is the test of
whether the framing is doing honest work: **remove it and the build still pays for itself.**

### What it is not

- **It is not a consciousness detector, and no result it produces is evidence of phenomenal
  experience.** Every quantity in this document is about access, control, or report. That is not a
  modesty caveat added at the end; it is each source theory's own stated position. Rolls concedes
  his support is a plausibility appeal and the evidence "is, and I think may remain, circumstantial."
  Wiese and Friston require their framework to admit systems that implement the computations
  *without* being conscious. Butlin and Long decline necessity, sufficiency and weighting, and call
  their list a rubric of candidacy. Graziano's own answer is "yes and no." Lamme identifies
  phenomenality with a state that by construction produces no report.
- **It does not compute Phi and must never be described as approximating one.**
  `instruments/integration_measure.py` computes a perturbational complexity index. PCI has never been
  compared against Phi on any object, because Phi has never been computed for any system large
  enough to have a PCI. No ordering of PCI values licenses a Phi claim in any direction.
- **It does not satisfy IIT, and cannot.** Under IIT as written, the units actually doing the
  causing here are transistors behind a von Neumann bottleneck, so this substrate is disqualified
  regardless of every number below. That statement does not deny machine consciousness -- other
  theories disagree with IIT precisely at that point -- and it is more useful than a green marker.
- **It does not implement HOT.** See section 3.2. The build deliberately declines HOT's aboutness
  commitment, and nothing here may be cited as a HOT test.
- **It is not currently implementable.** Eleven steps are blocked on this substrate for reasons
  given in section 7. Two of the blocks can cancel half the program, and cancelling it would be a
  legitimate outcome.

### The rule this document is written under

Do not deny the possibility. Do not fake the evidence. **A mechanism whose success criterion cannot
go red is worthless.** This repo has shipped three such gates in one feature and caught all three by
disbelieving an output. The first draft of this algorithm shipped three more -- they are listed in
section 10 with the fixes -- and the reason they were caught is that three independent adversarial
reads were run against the tree rather than against the draft's own citations.

---

## 1. Verification status

Everything the algorithm attaches to was read in the tree on 2026-08-02. Line numbers are from that
read.

| Claim | Status | Evidence |
|---|---|---|
| `PREC0 = 1.0`, `PREC1 = 1.0` gate the PC error drive | VERIFIED | `cortex/predictive_coding.py:88-89`, used `:192-193`, `:237-238` |
| `THETA = 0.5`, `LEAK = 0.02` set the error-neuron operating point | VERIFIED | `cortex/predictive_coding.py:86-87` |
| `ignition_threshold=0.55` is the default in three constructors | VERIFIED | `thalamus/reentrant_workspace.py:103`, `thalamus/global_workspace.py:93`, `cortex/access_consciousness.py:84` |
| **`brain.py:2149` is the clobber, not a hook** | VERIFIED | `cortex/brain.py:2143-2150` recomputes `ignition_threshold` from `base_ignition_threshold` every wake tick, inside `try:` / `except Exception: pass` at `:2115`/`:2151`. It also writes `_v_th = thr / _LEAK` at `:2150`. |
| `base_ignition_threshold` is documented immutable | VERIFIED | `thalamus/reentrant_workspace.py:105-109`, "kept immutable ... Read this, never re-type the default" |
| The gate that decides access reads `self.ignition_threshold` | VERIFIED | `thalamus/reentrant_workspace.py:244` |
| **The workspace is saturated: 100% ignition at every gain** | VERIFIED | `substrate/region_coherence.py:132-137`. Winning drives span [0.668, 1.000] median 0.815; masked drives span [0.086, 0.500] median 0.121, p75 0.167 (`:115-117`). The interval (0.500, 0.668) has never been observed. |
| **The graded access readout is calibrated at the shipped theta** | VERIFIED | `thalamus/reentrant_workspace.py:89-101`. `_ACC_N=32`, `_ACC_LO=0.85`, `_ACC_HI=1.90`, band measured over 360 ignited wake ticks *at theta=0.55*; consumed at `:355-374`. |
| `IGNITION_GAIN = 0.5` scales the coherence discount on the rheobase | VERIFIED | `substrate/region_coherence.py:103`, consumed `:146`, `:298-303` |
| A hardcoded `"drive": 0.5` is one of six inputs to that coherence | VERIFIED | `cortex/brain.py:2124` |
| The endogenous attention port exists and is unused | VERIFIED | `cortex/learned_attention.py:88-101 _goal_gain`, `:116 attend(features, goal=None)`; sole live call `cortex/brain.py:1128` passes `goal=None` |
| **B = 4, not 8** | VERIFIED | `cortex/brain.py:814` -- `LearnedAttention(dim=64, n_bands=4, seed=...)` |
| **`brain.py:1128`'s documented hole is a FEATURE-SPACE goal** | VERIFIED | `cortex/brain.py:1125-1127`. A band-preference vector takes the *other* `_goal_gain` branch and does not close it. |
| The attention gate is computed FROM the content | VERIFIED | `cortex/learned_attention.py:126-140` -- `sal` is per-band mean absolute feature magnitude; gate = learned synapse * salience * goal * surprise |
| `attention_focus` is exported and unconsumed | VERIFIED | `cortex/brain.py:2865` |
| **`state()['pci']` is a synthetic surrogate, stale 3 nights in 4** | VERIFIED | `cortex/brain.py:2405-2419`: fresh `PerturbationalComplexity(n=10, coupling=cpl, noise=0.45)` under `if self._nights % 4 == 0`, inside `try:` with bare `except: pass` at `:2419`. Surfaced at `:2916`. |
| **`PerturbationalComplexity` is a closed toy with no `system` port** | VERIFIED | `instruments/integration_measure.py:199-231` -- `_run` builds its own `LIFNeuron` list from its own `self.w` and pre-draws its own noise. No injection point exists. |
| `lz76_complexity` is a pure-Python nested scan | VERIFIED | `instruments/integration_measure.py:82-113` |
| The regime factories are a `weight_mode` switch | VERIFIED | `instruments/integration_measure.py:296-316` |
| `push_subgoal()` has callers only in its own self-check | VERIFIED | def `cortex/prefrontal_executive.py:299`; callers `:469`, `:630` only |
| `maintain()` has SIX call sites across four modules, all offline | VERIFIED | `cortex/autotelic_goals.py:315`, `reasoning/autonomous_brain.py:122`, `reasoning/developmental_core.py:153`, `reasoning/developmental_loop_wm.py:56/:65/:128`. Plus self-checks at `prefrontal_executive.py:383/389/475/574` and `autonomous_loop.py:2092`. The enumeration at `brainstem/autonomous_loop.py:970-974` is correct and is the authority. |
| **`_new_frame` stores an integer, not a proposition** | VERIFIED | `cortex/prefrontal_executive.py:311-313` returns `[index, 1.0, [0.0] * self._n_wm]` |
| `deliberate()` exists | VERIFIED -- **the draft said "NOT VERIFIED"** | `cortex/prefrontal_executive.py:230-235` |
| `recurrence=0.98`, `hold_threshold=0.2`, `max_goals=4`, `commit_races=25` | VERIFIED | `cortex/prefrontal_executive.py:88-96` |
| `_n_wm = 6`, `_wm_steps = 8` | VERIFIED | `cortex/prefrontal_executive.py:132-133` |
| **`Brain.tick(self)` takes no arguments** | VERIFIED | `cortex/brain.py:2690`; `_wake_tick(self)` at `:1808` builds its own context |
| `charge_probe._PinnedWeights` is a **dict subclass** | VERIFIED | `tools/charge_probe.py:54-74`, applied `:106` to `b._drive_w`, a `{source: float}` map |
| The prior leaky neuter fired on 12/60 ticks | VERIFIED | `tools/charge_probe.py:58-66` |
| `charge_probe`'s observables are actuator-mediated | VERIFIED | `tools/charge_probe.py:112-113` -- `arm_stopped`, `map_position` |
| VSA `D = 512`, `VOCAB = 128`, `THRESH = 0.95` | VERIFIED | `cortex/vsa_reasoning.py:48-51`; `record_recovery` `:140-151` |
| `eprop.py` is a standalone demo; `grad_eprop` is bound to `Net` | VERIFIED | `reasoning/eprop.py:105-235`; `_cosine` `:282`; `assert r["cosine"] > 0.5` `:365` |
| `GenerativeSelfModel.KAPPA = 0.5` is a literal | VERIFIED | `cortex/generative_self_model.py:70`; trace form `:101-105`, update `:144-145` |
| Instrumented tick is 27.78 ms | VERIFIED | `docs/POLICIES.md:496` |
| `_BIND_IGNITE = 0.6` sits under a "used only by the self-check" header | VERIFIED | `thalamus/global_workspace.py:311-318`, branched `:339`, `:376` |
| **...but it reaches a live user-facing consciousness self-report** | VERIFIED -- **new finding, section 8 offender 4** | `reasoning/spike_chat_lake.py:34` imports `probe_binding`; `reasoning/consciousness_probe.py:72-85` calls `bind_broadcast_spiking`, which branches on `_BIND_IGNITE`; `spike_chat_lake.py:1173-1182` tallies it into a reply; that reply is returned on the live chat path at `:925-931` for "are you conscious". **[DEAD ROUTE, 2026-08-07: `consciousness_probe.py`'s call went with its F11 deletion on 2026-08-04, and the `bind_*` family itself -- `_BIND_IGNITE` included -- was retired 2026-08-07 when the honest precision-bearing benchmark returned its pre-registered null. The row is kept as the finding it was; nothing it cites still exists on a live path.]** *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).* |

---

## 2. Triage across the six theories

### 2.1 Agreements -- no adjudication needed

**AG-1. Precision is an inferred quantity, not a constant.** Hohwy makes it the definition of
attention; GWT's ignition threshold *is* a precision gate under another name; HOT's confidence is a
precision estimate over first-order states; AST's schema is a model of how precision is allocated.
Nobody argues for a constant. This tree ships `PREC0`, `PREC1`, and three separate `0.55`s.

**AG-2. A second-order read that is separately parameterized and therefore capable of being wrong.**
HOT requires misrepresentation. AST requires impoverishment. Type-2 AUROC requires the confidence
source to be able to decorrelate from correctness. A read computed as a deterministic function of
the first-order state satisfies none of them, and every existing "second-order" module here
(`cortex/metacognitive_monitor.py`, `second_order_access`) is exactly such a function.

**AG-3. Dissociation, not markers.** All six converge on the same instrument shape: ablate, then
show one capability survives while another collapses. This repo already owns the instrument.

**AG-4. None of it bears on phenomenal consciousness.** Stated inside each theory, not added here.

### 2.2 The stack

The theories are not rival descriptions of one thing. They describe levels of one control loop, and
data dependency forces the order:

```
L0  first-order inference       RPT local recurrency + PP hierarchy
     -> content + error
L1  precision allocation        PP precision inference; AST's controlled variable
     -> THE CONTROL STATE
L2  access selection            GWT ignition + re-entrant sustain
     -> availability + sustain trace
L3  model OF the control state  AST schema + PP precision-over-precision
     -> band preference / blame / confidence
L4  credit assignment on a plan HOST symbolic repair
```

**The synthesis claim is that L3 is one computation, not two.** AST's attention schema and PP's
precision-over-precision are the same object: a separately-parameterized, content-blind, predictive
read of a control variable, fed back into the controller of that variable. P2 (exactly one mechanism
survives a comparison) therefore forbids shipping them separately. The algorithm ships one -- the
**Attention-Control-State reader (ACS)** -- with three heads.

The name is deliberate. The draft called it CSM, which a skimmer reads as "Conscious State Model."
It is a controller input. Nothing in the name may suggest otherwise.

### 2.3 HOT is declined, not implemented

The draft claimed HOT's higher-order reader collapses into the same object. **It does not, and the
claim is withdrawn.** HOT's higher-order state is *about the first-order perceptual state*; that
aboutness is what makes it higher-order in HOT's sense and what makes misrepresentation possible. A
model of an attentional control variable is higher-order with respect to *attention*, not with
respect to content. The collapse is legitimate as engineering and illegitimate as theory
attribution.

The consequence propagates. Type-2 AUROC from a control-state-only confidence head can beat chance
because control state correlates with trial difficulty. **That is difficulty estimation, and it
would survive in a system with no higher-order representation of content whatsoever.** It is
reported as difficulty estimation.

If a HOT arm is wanted later it needs a separate content-facing reader, built and named as such.

### 2.4 Contradictions -- picks and the experiments kept

| # | Contradiction | Pick | Experiment kept |
|---|---|---|---|
| K1 | GWT: breadth is the mechanism. Rolls: breadth cannot identify the *kind* of computation. | GWT's mechanism, threshold derived -- **conditional on gate G1** | **DX1**: sweep theta, measure held-out repair accuracy. Flat = consistent with Rolls' objection to breadth-as-mechanism. Peaked = consistent with GWT's computational prediction. **Neither adjudicates either theory's claim about phenomenal consciousness, which DX1 does not address.** |
| K2 | AST: the higher-order model must be content-blind. HOST: it must bind to the first-order symbols. | Content-blind (AST), **conditional on gate G4** | **DX2**: two readers, identical capacity and parameter count, one fed step symbols, one fed only control-state segments. Compare held-out blame accuracy against the episode's own 1/K. |
| K3 | GWT: tighter workspace -> more access. Measured PCI work: tighter workspace -> lower complexity. | Neither as a marker; both reported as predictions | **DX3**: the same sweep measures behavioural access charge and live PCI. **Written down in advance: a falling PCI under a tighter workspace is the PREDICTED result, not a regression.** |
| K4 | RPT: perceptual organization completes without attention. AST: the model of attention is the locus. | Defer RPT | Deferred with the prerequisite named: a figure-ground readout not derivable from the category readout, plus a feedforward-only mode inside the perception tier. Built today it degenerates into "recurrence is present," which cannot go red. |
| K5 | Friston/Deane: counterfactual depth *is* consciousness. Seth/Hohwy: property-mappings. | Property-mappings | **No discriminating experiment exists, and that is the finding.** Depth >= 1 holds for `plan(horizon=1)` and for a thermostat; no author names a required depth; supplying one would be a literal arbitrating a consciousness verdict. Deane's `SV = -H[sigma(-G)]` is always finite and positive, has no red state, and would sit next to `access_index` and `pci` looking like a third consciousness readout. Not built. |
| K6 | HOSS derives ignition from a higher-order posterior. HOST disqualifies confidence monitoring. | HOSS as instrument, HOST as mechanism | **DX4**: correlate the HOSS present-vs-absent KL asymmetry against ignition, with an N=1 branching control that must make the asymmetry vanish. **Conditional on gate G1** -- see section 7. |
| K7 | `GlobalWorkspace` and `ReentrantWorkspace` are both live; the latter's docstring argues the former is broken. | Resolve before instrumenting | An unexecuted P2 verdict sitting in a comment. Step 0. |
| K8 | IIT is anti-functionalist; this substrate is disqualified. | State it, do not test it | Not an experiment. A standing caveat. |

---

## 3. Blocking gates -- measurement only, before any build

Five of these can cancel large parts of the program. **Cancellation is a legitimate result and is
far better than the alternative, which is running an experiment whose instrument is at its rail and
reporting the rail as a theory verdict.** Every gate is pure measurement, and every one has a stated
red.

| Gate | What is measured | RED, and what it cancels |
|---|---|---|
| **G1 -- saturation** | On the task, over `S` seeds: the coalition-drive histogram and the per-tick ignition rate at the shipped `theta = 0.55`. | If ignition does not fail on a non-trivial fraction of ticks -- fraction threshold derived from the measured overlap of the winning [0.668,1.000] and masked [0.086,0.500] drive distributions, **not chosen** -- then **theta is not a controlled variable and DX1, DX3, DX4 are CANCELLED.** `region_coherence.py:132-137` says the current answer is 100% ignition at every gain in [0,1]. Assume the gate fires until measured otherwise. |
| **G2 -- generator assertions** | Six properties of the episode generator, on the generator alone, before any brain touches it. See section 5.2. | Any one red invalidates the task. Most important: a decoder trained on gap-window input must not recover the target above its own shuffle band. Generators leak. |
| **G3 -- task binding** | With no ACS at all: mutual information between the held goal at end-of-gap and the committed first action. | At the shuffle floor, **stop**. `brainstem/autonomous_loop.py:986-995` diagnoses the reverted WM wire as "this loop has no gap." Adding a gap does not make the action path *consume* what is held; consumption is a wiring property. Without G3, an all-null ablation matrix gets misread as "the ACS is inert" when the truth is "the task never bound." |
| **G4 -- content leakage** | Train a decoder of target identity from `u_t` alone, held out, with a label-shuffle band. Symmetrically, the direct-content decode at matched capacity. | If `u_t` decodes target identity at accuracy statistically indistinguishable from the direct content decode, **content-blindness is a naming convention, AST's specific commitment is untestable here, and the L3 result is VOID -- not weakened.** |
| **G5 -- deterministic replay** | Two *unperturbed* runs of the whole brain from the same snapshot with all RNG streams at the same position: assert bit-identical rasters. | If it cannot be made to pass, **B1 (real PCI) is not runnable on this substrate and must be reported as such.** See section 7 block 5 for why the failure inflates PCI rather than obviously breaking. |

---

## 4. State

```
# L1 precision                                                     dim
sigma2[l]          EMA of squared PC error per level                L=2
alpha[l]           EMA rate, from measured autocorrelation time     L
pi_floor[l]        LIF-derived minimum gain (see A2)                L
tau_u              autocorrelation time of u_t                      1

# L2 access threshold
H_live[G]          running histogram of posted coalition drive      G
H_surr[G]          running histogram of surrogate-post drive        G
theta_t            derived rheobase (see A3)                        1

# L3 ACS reader -- ONE mechanism, own RNG stream, own weights
u_ring[W][2B+L+5]  content-blind control-state ring buffer
zbar[C]            leaky eligibility trace                          C
W_pred[C][2B+L+5]  next-control-state predictor        (three-factor)
W_g[C][B]          HEAD A: endogenous band preference  (three-factor)
W_b[C+1][1]        HEAD B: per-step blame logit        (three-factor)
W_c[C][1]          HEAD C: calibrated confidence       (three-factor)
acs_rng            crc32 child seed, SEPARATE from the planner's stream

# L4 plan
plan_code[D]       VSA bundle of the current one-off plan           D=512
step_codes[K_e][D] bound (position, action) codes
trace[K_e]         (s_i, a_i, s_hat_i+1, s_i+1, r_i, d_i, u_window_i)

# calibrators
cal_blame          SignalCalibrator running quantile (substrate/signal_calibrator.py)
cal_fail           running episode failure base rate

# log
trials[]           per-episode JSONL record
```

**`C` must be recomputed from the constructors, not from the symbol table.** At the shipped
`n_bands=4` and `L=2`, one frame of `u_t` is `2B + L + 5 = 15`, and `C = 15 * (W+1)`. The draft
assumed `C ~ 300`. It is an order of magnitude smaller, and the consequence is in section 9.2.

---

## 5. The algorithm

### 5.1 Inputs and outputs

**Inputs.** Live percept latent; PC per-level errors; coalition posts `(source, content, salience)`;
the arm's measured backlash and repeat-command position-dispersion distributions; the episode
generator; seed.

**Outputs.** Per tick: `pi_hat[l]`, `ticks_at_floor[l]`, `theta_t`, `u_t`, `g_t`,
`drive_margin = drive_winner - theta_t`, `ignited`. Per episode: `blame_posterior`, `blamed_step`,
`repair_success`, `confidence`, `correct` in {0,1}, `report_arm`. Per sweep: the tables in 5.4.

Note `drive_margin` and not `access_strength`. See section 7 block 2.

### 5.2 Step 2 is not "build a task" -- it is a stimulus port

`Brain.tick(self)` takes no arguments (`cortex/brain.py:2690`) and `_wake_tick` builds its own
context from `_safe_ctx()`. There is no external stimulus port. The task requires per-tick control
of percept-band contents, a cue/gap schedule, an injected slip on a designated step, an
information-only probe action, and a removable report channel. **None of that reaches the brain
today.** This is an input-port addition to a 3979-line module on the live path, and it is the
largest and riskiest single item in the plan.

Budget it as three things: a `tick(stimulus=None)` port, an `EpisodeDriver` that owns the schedule,
and the G5 determinism gate. Land it alone. **The draft called the per-trial log the
highest-value item; that is wrong only in ordering -- there is nothing to log per trial until a
trial exists as an object the brain can be driven through.**

**TASK: occluded cued multi-step reach with an injected slip.** Per episode `e`:

1. **Cue** `t in [0,c)`: a target is cued; a higher-salience distractor appears elsewhere. **The
   band-to-target mapping is randomized per episode** so that target identity is not band identity
   -- otherwise `onehot(focus_band)` inside `u_t` carries the target in the clear and G4 fires by
   construction.
2. **Gap** `[c,g)`: the cue is removed; all bands carry noise only.
3. **Plan** `t=g`: the executive emits a `K_e`-step sequence, `K_e` drawn per episode.
4. **Execute** `[g, g+K_e)`: on step `j_e` the transition slips by a magnitude drawn from the arm's
   **measured** backlash distribution. `j_e` is ground truth, is never an input to any reader, and
   **is never a training target** (section 5.3 A7).
5. **Probe**: one action reveals which slip regime is active and makes zero task progress.
6. **Outcome**: binary, tolerance = the arm's **measured** repeat-command position dispersion.
7. **Report arms**: `r1` includes an action emitting the target identity; `r2` removes it.

**G2's six assertions, run on the generator alone, each with a red:**

| # | Assertion | Red |
|---|---|---|
| 1 | Gap: a decoder trained on gap-window input does not recover the target above its shuffle band | it does -> there is no gap |
| 2 | Off-salience: distractor salience exceeds cued-band salience on >= a stated fraction of cue windows | below it -> holding the cue is not against salience |
| 3 | One-off: count distinct `(plan, j_e)` pairs; report train/eval overlap | overlap above a preregistered maximum -> add a plan-disjoint split |
| 4 | Ground truth: a classifier for `j_e` fit on the observable stream performs at chance | above chance -> `j_e` is visible and blame is trivial |
| 5 | Outcome: base failure rate lands in a stated interval | 0% or 100% -> every downstream AUROC is undefined |
| 6 | Epistemic: the probe measurably sharpens the regime posterior | it does not -> it is a no-op action, not an information-only one |

**Gap length is a swept independent variable, not a derived constant.** `recurrence=0.98` and
`hold_threshold=0.2` hard-cap any WM-held gap at `log(0.2)/log(0.98) ~ 80` maintain() calls from
drive 1.0. Report results as a function of gap length with 80 named as the ceiling. (See section 10
for why the "set the gap to 60% of the measured decay" repair was rejected.)

### 5.3 Part A -- the online loop

---

**A0. Resolve P2 between the two workspaces. GATE, runs once.**
Compare `thalamus/global_workspace.py` against `thalamus/reentrant_workspace.py` on the task. Retire
the loser; port what it uniquely provides. The reentrant docstring already argues the case; execute
it rather than instrumenting both.
*Cost:* one `S x E` sweep. *Blocks:* everything.

---

**A1. Emit the first-order error.** Unchanged. `cortex/predictive_coding.py errors()`.
*O(L d)* per tick.

---

**A2. Infer precision.** Replaces `PREC0` / `PREC1`.

```
tau_l     = first zero-crossing lag of autocorr(e_l)        # warmup, then rolling re-estimate
alpha_l   = 1 / tau_l
sigma2_l  = (1-alpha_l)*sigma2_l + alpha_l*e_l^2            # from errors() at UNIT GAIN, asserted
pi_l      = 1 / (sigma2_l + eps)
pi_hat_l  = max(pi_floor_l, L * pi_l / sum_l' pi_l')        # allocation, floored
drive_l   = pi_hat_l * e_l
pwse      = sum_l pi_hat_l * |e_l|^2
```

`pwse` is a **precision-weighted squared error**. It is **not** variational free energy: there is no
complexity term and no `-0.5 * sum(log pi)`. The draft's comment called it "actual variational free
energy," which smuggles back the constitutive FEP branch that K5 explicitly rejects. The variable
and its comment carry the disclaimer permanently.

Three things the draft got wrong here, all now fixed:

- **`PREC0 = PREC1 = 1.0` are identity multipliers.** A2 is therefore a **new mechanism**, not a
  de-hardcoding, and must be justified on that basis. The literal is not removed; it is relocated
  into `L` (the zero-sum normalizer keeps `mean_l(pi_hat_l) == 1.0` for every input, forever) and
  into `THETA = 0.5`, which sets the absolute operating point and which A2 does not touch.
- **The zero-sum falsifier is an identity.** "Raising precision on one channel must degrade
  competitors at fixed total" is a unit test of the normalizer. It is deleted as a theory test and
  re-filed as a normalizer assertion. The real Hohwy test is whether the *allocation tracks
  manipulated reliability*: inject known-variance noise into one level and require `pi_hat` to move
  in the predicted direction. Red = it does not track.
- **`pi_floor_l` prevents a lock-out.** Without it: `pi_hat_0` falls -> fewer `s_e0` spikes -> the
  `r1` update moves less -> `e0` stays large -> `sigma2_0` grows -> `pi_0` falls further. At
  `LEAK = 0.02` a silent error population is effectively absorbing. The floor is derived from the
  LIF's own rheobase given `THETA` and `LEAK` -- the value producing >= 1 spike -- so it is not a
  new literal. **`ticks_at_floor[l]` is a first-class output: a level floored most of the time
  refutes A2 rather than tuning it.**

*Attach:* `cortex/predictive_coding.py:88-89` (delete both), `:192-193`, `:237-238`.
*O(L)* per tick; *O(L tau)* per re-estimate.

---

**A3. Derive the ignition threshold.** Replaces `0.55`. **Blocked on gate G1 and on the contract
change in section 7 block 1.**

Two running histograms over one bin grid: `H_live` from posted coalition drives, `H_surr` from a
surrogate stream.

```
theta_t = the drive at the first sign change of log(p_live(d) / p_surr(d))
          sustained over >= 3 consecutive bins each with n >= 30
```

Two draft errors fixed. **Exact equality between two empirical histograms is satisfied by no bin
almost surely**, so the draft's `max{d : p_live(d) == p_surr(d)}` would have returned its own red on
every run for numerical reasons. And **the crc32 surrogate does not transfer**:
`substrate/region_coherence.py:189-194` scatters a *phase* for a silent region so it reads as
incoherent. Scattering a post's content label leaves salience untouched, so the drive marginal is
unchanged and `p_live == p_surr` on every bin. The surrogate must destroy what theta detects:
**permute which source posted which content across ticks**, at matched per-source marginal salience,
then recompute the coalition drive.

Three reds, all reachable:

1. No sustained sign change on the grid -> theta is underivable, DX1 reports a null. **Restoring
   0.55 with a comment is the forbidden response.**
2. A crossing landing inside the empirically empty interval (0.500, 0.668) -> theta is unidentified
   and DX1 is cancelled.
3. Theta unstable across a decade of bin-grid `G` -> the derivation is reported as unstable.

**This supplies *an* operational threshold. It does not answer Rolls' question.** Rolls asks how
global a broadcast must be for consciousness to be present; a likelihood-ratio crossing answers "at
what drive does posting become diagnostic of non-scrambled structure." Those are different
questions, and the draft's "this is Rolls' yardstick supplied operationally" is withdrawn.

*Attach:* the new `set_base_rheobase()` contract (section 7 block 1), never `brain.py:2149`.
*O(G)* per update, amortizable to *O(1)*.

---

**A4. Emit the control-state vector.** Content-blind by construction, and measured for it.

```
u_t = concat( gate[0..B-1],                    # LearnedAttention gate,   B = 4
              pi_hat[0..L-1],                  # from A2,                 L = 2
              onehot(focus_band | NONE),       #                          B+1 = 5
              focus_strength, ignited, drive_margin, sustain )   # 4
                                               # total 2B + L + 5 = 15
```

Constraints, enforced structurally and then **verified by measurement**:

- **Content-blind by type**: a pure function of gate, precision, focus, ignition and their history.
  Constructor assert plus a test that two different latents producing the same control state produce
  a bit-identical `u_t`.
- **Content-blind in information**: gate G4. This is the one that matters. `learned_attention.py`
  computes `sal` from the content and the gate from `sal`, so **`u_t` is a coarse, content-derived
  code, not an orthogonal channel.** Type-level content-blindness is a code-organization property.
  Only a *measured* lossiness makes K2 a live question.
- **Impoverished**: no weights, no salience, no surprise, no reward, no estimate of the S/N benefit
  attention confers.
- **Read back**: `u_t` is an input to A5, not an entry in `state()`. A vector that is only exported
  is what `brain.py:2865` already has, and it is not a schema.

*O(2B + L + 5)* per tick.

---

**A5. ACS: predict the next control state.** The one L3 mechanism.

```
kappa   = exp(-1 / tau_u)                   # rolling re-estimate, not warmup-frozen
zbar    = kappa * zbar + u_t
u_hat   = W_pred @ zbar
err     = u_{t+1} - u_hat                   # spiking ON/OFF LIF population
dW_pred = lr * outer(err, zbar)             # three-factor, local, no BPTT
```

Form follows `cortex/generative_self_model.py:101-105` and `:144-145`, with one deviation named:
that module's `KAPPA = 0.5` is a literal (`:70`) and this one is estimated.

**The draft's `divergence_from_first_order` criterion is deleted.** It could not go red. The ACS is
content-blind by constructor assert; the planner's value ranking is a function of content; a read
that structurally cannot see content cannot be a deterministic function of a content-derived
ranking. Divergence > 0 was guaranteed before a single weight was trained. Worse, the metric has no
direction -- maximum divergence is what a randomly initialised reader produces, so the pass
condition and the signature of a dead layer were the same number.

**Replacement, which has a red:** hold out episodes; require the ACS blame ranking to beat *the
planner's own value ranking* as a predictor of `true_faulty_step`. **If the planner's value ranking
predicts the faulty step as well as the ACS does, the ACS adds nothing.** No divergence measure can
support a misrepresentation claim, because divergence is what noise produces.

*O(C (2B+L+5))*.

---

**A6. HEAD A -- endogenous band preference.**

```
g_t   = relu(W_g @ zbar)                              # length B = 4
gated = learned_attn.attend(latent, goal=g_t)         # the n_bands branch
dW_g  = lr * r * (elig - mean(elig)),  elig = outer(zbar, g_t)
```

The `n_bands`-length branch at `cortex/learned_attention.py:94-97` is reachable at
`n_bands=4, dim=64` and needs no signature change. This is the cleanest item in the plan: a wire,
not a build.

**But it does not close the documented hole.** `cortex/brain.py:1125-1127` refuses a goal because "a
real goal vector *in feature space* is not available yet." A `B`-length band preference takes the
*other* `_goal_gain` branch (`:98-101`). The feature-space hole stays open, and the draft's claim
that A6 closes it is withdrawn.

**Step gate (see section 10, applied critique 13):** landing Head A makes `u_t` consumed, which is
P3 satisfaction, not evidence. The gate is an outcome-change count fixed in advance from the
measured seed-to-seed spread: **`g_t` must change the selected band on >= N of 200 ticks AND move
ticks-on-target versus `goal=None` by more than the seed spread.** Both, or Step 6 is reverted the
way the WM wire was. The chance line for a 4-band hold endpoint is 0.25, stated here and not
discovered in analysis.

*O(C B)*.

---

**A7. HEAD B -- blame on a one-off plan, then targeted repair.**

*Plan encoding at episode start* (`cortex/vsa_reasoning.py`, D=512):

```
step_codes[i] = bind(permute(POS, i), hv(a_i))
plan_code     = bundle_i step_codes[i]
readback(i)   = cleanup(bind(permute(POS,i), plan_code), codebook)
```

*Blame at episode end -- and the VSA record is CONSUMED:*

```
b_i = W_b @ [ zbar[window_i] ; sim(readback(i), hv(a_i)) ]
p   = softmax(b)
dW_b = lr * (r - r_bar) * zbar[window_i]     # r = correct in {0,1}, running baseline
```

Two draft errors fixed, both fatal as written.

**(a) The VSA layer was decorative.** Blame read only control-state segments; repair rebuilt the
plan from `step_codes[]` already held in `trace[K_e]`; `readback(i)` was defined and never consumed;
and `_new_frame(index)` stores an integer (`prefrontal_executive.py:311-313`), so pushing
`plan_code` into a frame changes no downstream read. Delete every VSA call and the draft's numbers
were bit-identical -- reproducing `autonomous_loop.py:978-982` exactly, a constant telemetry field
turned into a varying one with 0/200 outcome change. Worse, the draft preregistered "drop `permute`;
if repair is unchanged, HOST's syntax claim is refuted" -- and since `permute` only affected the
unconsumed readback, **that tie was guaranteed by the code, manufacturing a refutation of Rolls out
of a wiring gap.** Consuming `sim(readback(i), hv(a_i))` gives the arm a real red: overload the
record past the capacity `vsa_reasoning.record_recovery(k_pairs, seed)` (`:140-151`) already
measures, and blame accuracy must fall toward `1/K_e`.

**(b) The teacher did not exist.** The draft's `dW_b = lr * (target - p_i) * zbar` uses the one-hot
of `j_e`, contradicting the task's own "never visible to any reader" and contradicting HOST, whose
higher-order system *infers* the faulty step. The only third factor available at episode end is the
binary outcome. **`j_e` is an evaluation-only variable that never enters any update.** The
consequence is honest and expensive: blame becomes REINFORCE-shaped single-scalar credit assignment
across `K_e` steps, variance grows with `K_e`, and separating from `1/K_e = 0.25` at a defensible
effect size needs order 1e3 to 1e4 episodes per condition. **Held-out blame accuracy is therefore
demoted to a secondary endpoint behind repair-success-versus-re-plan, which uses the same signal at
far lower variance.**

*Repair:* rewrite only the blamed step; `a'` ranked by `cortex/counterfactual_sim.py imagine` and
committed through the spiking WTA at `prefrontal_executive.py:230`. (`deliberate()` exists; the
draft said it had not checked.)

*Repair/abandon:* `cal_blame`, a running quantile of the brain's own recent blame confidences.
**Its quantile level is a stated convention, not a derivation** -- see section 8 offender 8.

*O(K_e D)* encode, *O(A D)* readback, *O(C K_e)* blame.

---

**A8. HEAD C -- calibrated confidence.**
`c_t = sigmoid(W_c @ zbar)`. **This is the confidence source, named here, and nothing else may be
cited as confidence for the duration of the program.** `cortex/metacognitive_monitor.py`'s
reliability and `second_order_access` are demoted to descriptive telemetry by name, in writing, at
the baseline commit. The draft stated this anti-accommodation rule and then left the blank unfilled,
which is the accommodation.

Reported as **difficulty estimation**, per section 2.3.
*O(C)*.

---

**A9. Per-trial log.**

```
{trial_id, seed, condition, confidence, correct, u_trace, latent_features,
 blamed_step, true_faulty_step, repair_success, report_blocked, theta_t,
 drive_margin, mean_firing_rate, gap_len, capacity_scale, noise_sigma, split_tag}
```

Split by episode id **before any learning**; assert `len(eval) < len(all)` and that no eval id
appears in train. The prior burn's tell was `result count == input count`. **Episode-id splitting
does not prevent plan leakage** -- see section 8 offender 6.
*O(1)* per episode.

### 5.4 Part B -- offline measurement

**B1. Real PCI.** **Blocked on G5.** See section 7 block 5. Once unblocked: snapshot, run branch A
and branch B from the same snapshot with the same noise realization, B receiving a supra-threshold
pulse; `causal_response` differences them so spontaneous noise cancels; serialize time-major;
`normalized_lz` cross-checked with `gzip_complexity`; report `source_entropy` as the ceiling and
**do not divide by it**; force 0 on degenerate entropy; report `mean_firing_rate` alongside.

**dPCI is rate-matched or it is not reported.** LZ76 complexity is strongly rate-dependent, and
correct and incorrect trials will differ in rate -- that is what "the arm did something different"
means. Subsample into equal-rate bins before differencing. Primary = rate-matched; unmatched =
descriptive. Null = shuffle `correct` within rate bins and require dPCI into the shuffle band.
**dPCI's sign convention and disconfirming outcome are stated in the preregistration or dPCI is
dropped by name**, the way GBI was.

**B2. Expander control.** Degree-3 XOR expander at matched unit count and matched firing rate:
globally integrated by construction, computationally trivial by construction. It slots into the
existing `weight_mode` factory pattern at `integration_measure.py:296-316` in about ten lines.
**Report its PCI beside every PCI figure the repo emits, whichever way it comes out.** The existing
controls (silent, independent-noise, synchronous) are all *low*-integration and can only confirm.

**This is the first item to build.** It is independent of G5 -- it runs inside the toy simulator,
against the toy PCI, and tests whether that PCI rewards integration without computation. It is the
only control in the plan that can make an existing published number look bad.

**B3. Theta sweep -- DX1 + DX3 + DX4 in one pass.** **Blocked on G1.** Grid theta on **empirical
drive quantiles, never uniformly on [0,1]**, so points land where the distribution has mass. Stage
it: 5 screening points at `S=7`, refine only where a crossing appears, confirm at `S=20` on the
surviving region. Run the section 9.2 control matrix at a **single** theta, not across the sweep.
The compute arithmetic goes in the preregistration (section 7 block 8).

**B4. Noise titration.** Arms: broadcast wired vs bypassed, plus steelman arms (dropout plus weight
decay, train-time noise augmentation, matched non-ACS bottleneck, and **ACS-off but workspace-read**).
The fourth is the one that matters: the prior measurement found robustness persisting with the
self-model off, which falsifies the attractive story that routing through a compressed self-summary
buys robustness.

**B5. Report-blocked decoding.** Linear decoder over internal features at report-blocked steps,
minus shuffled-label AUC on the same CV splits. **Expanded from the draft's unexplained "dNRS"
acronym; if its disconfirming outcome cannot be stated, it is dropped by name.**

**B6. GBI -- DROPPED BY NAME.** `ReentrantWorkspace` is single-slot. At `K_s = 1` the participation
coefficient is undefined; at `K_s = 2` it is identically 0 for every possible system. A marker
returning the same value regardless of the system cannot go red. Same for the capacity sweep: only
levels 1 and 0 are reachable.

**B7. Preregistration.** Per marker: prediction, confirming outcome, **disconfirming outcome**,
interpretation of each, structural relevance, boldness, baseline commit, and **the tick arithmetic**.
A marker whose disconfirming outcome cannot be stated in advance does not get run. **B7 is applied
to the output list in 5.1, not only to the named markers** -- that is how dPCI and B5 nearly shipped
without reds.

---

## 6. Complexity

Per tick, at the real constructor values `B=4`, `L=2`, `C = 15(W+1)`:

| Step | Cost | Against a 27.78 ms tick |
|---|---|---|
| A1 error | `O(L d)` | existing |
| A2 precision | `O(L)`; `O(L tau)` per re-estimate | negligible |
| A3 theta | `O(G)`, amortizable `O(1)` | negligible |
| A4 `u_t` | `O(15)` | negligible |
| A5 predictor | `O(15 C)` | ~1e3-1e4 MACs even at generative_self_model.py's pure-Python idiom -- well under 1 ms |
| A6 head A | `O(C B)` | negligible |
| A7 head B | `O(C K_e)` per episode + `O(K_e D)` encode + `O(A D)` readback | once per episode, D=512, A=7: negligible |
| A8 head C | `O(C)` | negligible |

**The per-tick cost is not the problem, and no step needs numpy, a wider population, or a frozen
LLM.** The problems are elsewhere: B1's LZ76 is `O(n^2)` pure Python and B1 raises `n` by roughly
600x (section 7 block 6), and the sweep matrix is multiplicative (block 8).

---

## 7. NOT IMPLEMENTABLE ON THIS SUBSTRATE

Eleven blocks. Each names the reason and what unblocks it. **Two of them can cancel large parts of
the program, and that is the correct behaviour.**

---

**Block 1 -- A3 has no durable write target.**
`cortex/brain.py:2143-2150` recomputes `ignition_threshold` from `base_ignition_threshold` on every
wake tick, and the docstring at `reentrant_workspace.py:105-109` declares the base immutable. A
theta written to `ignition_threshold` survives zero ticks; the comparison that actually gates access
(`:244`) never sees it. The block also writes `_v_th = thr / _LEAK`, which the draft never mentioned,
so even a correct write would half-apply. The whole thing sits inside `try: ... except Exception:
pass`, so a raising write leaves the previous theta silently in place.

**The failure is silent, and its silent form scores as a theory result.** DX1 would run, produce a
flat curve across every grid point, and K1 preregisters "flat = Rolls." A disconnected setter would
be published as a verdict.

*Unblock:* an explicit `set_base_rheobase(theta, provenance=...)` that asserts theta lands inside
the LIF's usable range given `_LEAK`, reads back and asserts the write took, and records provenance.
Narrow the bare `except`. Then a positive control that cannot be skipped: set theta = 0.0 and
theta = 1.1, assert ignition reads 100% and 0%. **If either fails, no point on the sweep may be
reported.**

---

**Block 2 -- the graded access readout is not a sweepable instrument.**
`reentrant_workspace.py:89-101` sets `_ACC_N=32`, `_ACC_LO=0.85`, `_ACC_HI=1.90` from a drive
distribution measured *at theta = 0.55* over 360 ignited wake ticks. Sweep theta and the winner-drive
distribution translates off the ladder: low theta puts everything at the ceiling, high theta leaves
no drive to read. `access_strength` clips to a constant across most of the grid.

This is instrument saturation on the primary readout, and `access_strength` is not peripheral -- it
was an input to `u_t` and therefore to all four heads, *and* the behavioural charge DX3 correlates
against PCI.

*Unblock:* either re-derive the three constants per grid point from a warmup at that theta (21 extra
warmups, cheap, but in the plan rather than discovered mid-run), or **drop the population readout
for sweep purposes and report `drive_margin = drive_winner - theta_t`**. A4 above takes the second
option. The three constants go on the offender list either way.

---

**Block 3 -- the workspace is saturated, so theta may not be a controlled variable at all.**
`region_coherence.py:132-137`: "the workspace ignited on 100% of wake ticks at EVERY gain in
[0.0, 1.0], including gain 0.0." Winning drives [0.668, 1.000], masked [0.086, 0.500]. **The
interval (0.500, 0.668) has never been observed**, so a uniform grid puts several points in a dead
zone and yields a step function -- an outcome K1's preregistered space {flat, peaked, no-crossing}
does not contain, and which would be scored "flat." DX4 is worse: it correlates a KL asymmetry
against a comparator that is a constant 1, which returns NaN or 0 depending on the code path, and
the N=1 branching control is a control on the marker, not on the comparator.

*Unblock:* gate G1. **If the task does not break saturation, DX1, DX3 and DX4 are cancelled.** That
is a legitimate result and is far better than three NaNs read as findings.

---

**Block 4 -- there is no external stimulus port.** Section 5.2. `Brain.tick(self)` takes no
arguments.

---

**Block 5 -- B1 needs a deterministic-fork engine that does not exist.**
The draft said the PCI pipeline exists and only the system under test is wrong. **The pipeline does
not exist.** `PerturbationalComplexity._run` (`integration_measure.py:199-231`) constructs its own
`LIFNeuron` list from its own `self.w` and pre-draws its own noise from its own seed. There is no
protocol, no injection point, no `system` parameter. The class cannot measure anything but itself.

The causal-footprint trick -- the thing that makes PCI un-riggable -- requires two runs of the whole
brain sharing the exact same noise realization. On this substrate that is not a seed; it is every
RNG stream at the same position in both branches, across dozens of per-module `random.Random`
instances, numpy generators and crc32 child seeds. Two that will break it immediately:
`ReentrantWorkspace._jitter_for` memoizes a fresh `self._rng.uniform` draw the first time a content
is seen, so a perturbation surfacing one new content desynchronizes every subsequent jitter; and
`_stable_commit` runs `commit_races = 25` noisy races (`prefrontal_executive.py:237-245`), so any
branch difference in commit count consumes a different number of draws.

**Once the streams diverge the footprint becomes near-all-ones and the failure INFLATES PCI. It will
not look like a bug; it will look like a spectacular result.**

*Unblock:* a `system` protocol (`step(currents)`, `snapshot()`, `restore()`) plus one designated
brain raster, and then gate G5. **If G5 cannot be made to pass, B1 is not runnable and must be
reported as such.**

*Do not couple the cheap fix to this.* Renaming `state()['pci']` to `pci_surrogate` and narrowing
the bare `except` at `brain.py:2419` costs nothing and stops a wrong number being reported. That
number is also **stale on three nights in four**, because the computation is under
`if self._nights % 4 == 0` while the field is read every tick -- the repo's documented
silent-degradation shape sitting on top of the synthetic-surrogate problem.

---

**Block 6 -- `lz76_complexity` is `O(n^2)` pure Python and B1 raises `n` by ~600x.**
Current use is `n = 10 units x 57 post-stimulus steps = 570` characters, effectively free. A live
raster at populations 24-240 over tens of ticks pushes `n` to 1e3-1e4, and the quadratic term turns
each trial from milliseconds into minutes, times trials x sites x conditions. `gzip_complexity` is
already C-speed but cannot substitute: the reported number is normalized-LZ, and swapping the
estimator silently would change what every prior comparison means.

*Unblock:* cap `N_u * T_w` and justify the cap from the LZ estimator's own convergence, not from a
runtime budget; or vectorize/compile it and re-run the existing regime-ordering self-check
(integrated > independent-noise > silent ~ synchronous ~ 0) to prove it is the same function.

---

**Block 7 -- the neuter idiom does not transfer to matrix weights.**
`tools/charge_probe.py:54` is a `dict` subclass overriding `__setitem__`, specialized to
`b._drive_w`, a `{source: float}` map. The ACS weights are matrices. In numpy, `W += dW` mutates the
buffer in place and calls nothing; in nested lists, `W[j][k] = v` invokes the *inner* list's
`__setitem__`. **The pin would be a silent no-op and the "airtight" arm would be the live arm.**

The direction matters. `charge_probe.py:60-66` records the previous leaky neuter firing on 12/60
ticks and warns that a partial neuter biases verdicts toward "inert." **A no-op neuter biases the
other way -- toward "the ACS is load-bearing," which is the result the plan wants.**

*Unblock:* for arrays, `W.flags.writeable = False` so writes **raise and fail the run** rather than
leaking, with the pin on the read side. Then the verification the draft lacked: assert the pinned
weights are bit-identical at episode start and end, and report no ablation number until that check
has passed on every arm. **A neuter is verified by measurement, not asserted by construction** --
the same rule this document applies to everything else.

---

**Block 8 -- the sweep matrix is multiplicative against a 27.78 ms tick.**
B3 alone at `Q=21` x `S=20` x `E=200` x `T=60` is 5.04M ticks, about 39 core-hours for one arm.
Section 9.2 lists seven controls plus the primary; the report arms double it; that is roughly 620
core-hours. B4 adds ~120. Section 9.3's baselines add more. A7's variance analysis pushes `E` up,
not down. Order 1e3 core-hours before B1. It parallelizes across seeds, so this is not impossible --
but at 16 cores it is weeks of wall clock, and B3 is not "one pass."

*Unblock:* the staging in B3, plus the arithmetic in the preregistration.

---

**Block 9 -- RPT / DX5 has no readout.** RPT's dissociation needs a figure-ground readout
independent of the category readout. The claustrum assembly detector is documented unreachable by
construction (0/200 measured, re-measured 0), and `CrossModalBinder` is cross-modal superadditivity,
not spatial grouping. `ReentrantWorkspace(reentry=False)` is the *global* arm only; there is no
feedforward-only mode inside the perception tier. Built today, RPT degenerates into "recurrence is
present," which cannot go red.

---

**Block 10 -- the e-prop / BPTT cosine check is vacuous for the ACS, and is dropped by name.**
`reasoning/eprop.py` is a standalone demo: `grad_eprop(self, xs, tgt)` is a method on `Net`, bound to
`Net`'s specific `Win`/`Wrec`/`Wout` shape. There is no reusable rule to point at the ACS. Worse,
**the ACS as specified has no plastic recurrent weights** -- `zbar = kappa*zbar + u_t` is a fixed
leaky trace and `u_hat = W_pred @ zbar` is a linear readout, so there is no recurrent credit path to
approximate and the eligibility gradient and BPTT compute the same expression. The cosine would be
1.0 by construction: a green marker that cannot go red.

*Unblock:* make `kappa` plastic, at which point the check becomes meaningful and must be run. Until
then it is dropped, and the reason is stated in the spec rather than left as a passing assert.

---

**Block 11 -- no genuine no-report arm exists for this task.**
The task's outcome is "did the end-effector reach the target" under an occluded cue -- so **reaching
the correct target is itself a report of the cue identity. The action is the report.** `r1`/`r2`
remove the *explicit* report and leave the implicit one intact. They are renamed
**explicit-report** and **reduced-report**, and the draft's "may not be cited for any theory" is
deleted because it implies a certification the design cannot issue.

This is the same problem the draft correctly diagnosed for `charge_probe`, whose observables
(`arm_stopped`, `map_position`, `tools/charge_probe.py:112-113`) are actuator-mediated -- meaning
**every behavioural-charge verdict in the coverage docs is an explicit-report measurement with no
reduced-report control.** That finding stands and is uncomfortable.

*Unblock:* a passive trial type -- cue, gap, no action permitted, marker measured, ground truth known
only to the experimenter.

---

## 8. P1 audit

### Derived from measured state

| Quantity | Derivation |
|---|---|
| `pi_hat[l]` | measured error variance; removes `PREC0`/`PREC1` -- but see offender 2 |
| `pi_floor[l]` | the LIF's own rheobase given `THETA`, `LEAK` |
| `alpha[l]`, `kappa` | measured autocorrelation time, rolling re-estimate |
| `theta_t` | sustained log-likelihood-ratio sign change, live vs source-permuted surrogate -- but see offender 3 |
| reach tolerance | measured repeat-command position dispersion of the arm |
| slip magnitude | measured backlash distribution of the arm |
| AUROC chance line | label shuffle on the same trials, never asserted as 0.5 |
| blame chance line | `1/K_e`, the episode's own drawn plan length |
| G1's ignition-failure fraction | measured overlap of the winning and masked drive distributions |
| effective dimensionality | participation ratio of the ACS output covariance |

### Offenders -- this algorithm is NOT P1-clean

1. **`"drive": 0.5` at `cortex/brain.py:2124`.** A hardcoded literal is one of six inputs to the
   cross-region coherence measure; coherence feeds `ignition_bias()`, which modulates the rheobase.
   One sixth of the input to the thing that modulates theta is a constant, and one region is by
   construction never doing anything. **A3's derivation is not meaningful until this is fixed or
   coherence-gating is removed from the theta path.** First on the list, correctly.

2. **`THETA = 0.5` and `LEAK = 0.02` at `predictive_coding.py:86-87`, plus the normalizer `L`.**
   `PREC0`/`PREC1` are identity multipliers at the shipped values, so A2 relocates the constant
   rather than removing it: `L` pins `mean(pi_hat)` at 1.0 forever, and `THETA` sets the absolute
   operating point untouched. The zero-sum normalization also silently substitutes for the absent
   `-0.5 * sum(log pi)` term. That is a design choice, not a derivation, and it is recorded as one.

3. **Four new constants behind theta**: the bin grid `G`, the histogram window / forgetting rate,
   the surrogate permutation scheme, and the crossing rule. Every one moves theta, theta gates
   access, access gates the headline. **The draft's audit had a hole exactly where its flagship
   derivation lived.** Sensitivity requirement: theta stable across a decade of `G`, or the
   derivation is reported as unstable.

4. **`_BIND_IGNITE = 0.6` at `global_workspace.py:318` -- RECLASSIFIED, NOT REMOVED.** One review
   argued it is a self-check constant and inflates the audit. The header at `:311-313` does say
   "used only by the self-check," and it arbitrates no brain *behaviour*. **But the loose end that
   review flagged is real and I chased it.** `reasoning/spike_chat_lake.py:34` imports
   `probe_binding`; `consciousness_probe.py:72-85` calls `bind_broadcast_spiking`, which branches on
   `_BIND_IGNITE` at `:339`; `spike_chat_lake.py:1173-1182` tallies the result into a reply; and that
   reply is returned on the live chat path at `:925-931` when a user asks "are you conscious." So a
   hardcoded literal arbitrates one of four markers in a **user-facing consciousness self-report**.
   The reply text itself is honest ("I neither claim nor deny it"), but under this document's own
   framing a literal gating a reported consciousness marker on a live path is a **more** serious
   offender than one gating behaviour, not a less serious one. It stays on the list, reclassified.
   **[CLOSED 2026-08-07 by deletion, not by derivation: the probe's call to `bind_broadcast_spiking`
   went with the F11 deletion of the BINDING marker (2026-08-04), and the `bind_*` family --
   `_BIND_IGNITE` with it -- was retired 2026-08-07 when the honest benchmark returned its
   pre-registered null (the spiking decoder scored identically to an all-fired-pairs strawman at
   every K). The literal no longer exists, and neither does the marker it gated.]**

5. **`_ACC_N = 32`, `_ACC_LO = 0.85`, `_ACC_HI = 1.90`** at `reentrant_workspace.py:99-101`, and
   `_SUSTAIN = 1.10`, which sits inside the measured band. Block 2.

6. **`max_goals = 4`** at `prefrontal_executive.py:96`. Two separate problems. As a P1 offender it
   bounds `K_e`, so any capacity-degradation test measures a constant we chose. **The draft's
   proposed derivation is dropped by name**: "the K at which VSA cleanup recovery falls to the
   surrogate floor" is a function of `D = 512` and `VOCAB = 128` (`vsa_reasoning.py:48-49`), i.e. a
   different hardcode wearing a derivation's clothes -- and at D=512 the floor lands well above K=4,
   so the derived value would not equal 4 and there is no stated response to that. `max_goals` is a
   declared reporting parameter, matching B6's treatment of GBI. As a *validity* threat it is worse:
   with `K_e <= 4` and `A = 7` the plan space is at most 2,800 strings, so a table keyed on
   `(K_e, first action, outcome)` competes directly with credit assignment. Episode-id splitting
   does not prevent plan leakage. Report distinct `(plan, j_e)` counts and train/eval overlap;
   above a preregistered maximum, add a plan-disjoint split and report both.

7. **`recurrence = 0.98` and `hold_threshold = 0.2`** at `prefrontal_executive.py:88-89`. They cap
   any WM-held gap at ~80 ticks -- the plan's central manipulation, hard-bounded by two constructor
   defaults. Neither appeared in the draft's list. `_n_wm = 6` and `_wm_steps = 8` (`:132-133`) bound
   what the pool can hold at all.

8. **`cal_blame`'s quantile level -- a convention, not a derivation.** The draft derived it from
   `cal_fail`, "the system abandons at the rate the world actually fails." That identity mapping is
   *a* policy; the optimum depends on repair cost versus replan cost, neither of which is measured.
   **And G2 assertion 5 requires the base failure rate to be tuned into a stated interval, which
   makes `cal_fail` itself a tuned quantity -- so deriving from it would launder a tuned constant
   through a memorable sentence.** Declared a convention.

9. **`GenerativeSelfModel.KAPPA = 0.5`** (`:70`) in the module A5 borrows its form from. A5 estimates
   `kappa` instead; the deviation is named so "reuses the exact form" is not claimed.

10. **ACS learning rate `lr`.** A stability parameter that nonetheless shapes learned blame.
    Mitigation, not a fix: require the dissociation to hold across a swept `lr` decade and report the
    curve, not one point.

11. **`eps` in `1/(sigma2+eps)`.** Divide guard, acceptable only under an assert that no branch
    reads it. State the assert.

12. **The 75% in the noise-titration reporting.** Convention. Acceptable only if nothing branches on
    it. Any gate of the form "robust iff L75 > x" is explicitly forbidden.

13. **`cosine > 0.5`** at `eprop.py:365`. A verification threshold on the existing demo, not a
    behaviour arbitrator. Named for completeness; dropped for the ACS per block 10.

14. **`commit_races = 25`** (`prefrontal_executive.py:95`). Not a P1 offender -- a stability
    parameter -- but listed because it is a primary RNG-desynchronization source for G5.

15. **Seed counts (20, 7).** Statistical power conventions, not behaviour arbitrators.

---

## 9. The dissociation test

### 9.1 Primary -- pin the ACS

Matrix-safe neuter only (block 7), verified by readback, or no number is reported.

**MUST SURVIVE (first-order, matched seed):**

- Plan **execution** on first-time-correct trials.
- **Exogenous** capture: ticks-to-refocus after an off-focus surprise spike, unchanged.
- Free-energy settling on unoccluded percepts.

**The draft required the surviving trajectory to be BIT-IDENTICAL. That is unachievable and the
draft's own protocol therefore returned REFUTED with probability 1.** Head A is live in the
perceptual gate: A6 replaces `goal=None` at `brain.py:1128`, `gated` feeds `chan` at `:1129-1130`,
and everything downstream. Pinning the ACS changes `g_t`, which changes the gate on every tick,
which changes the trajectory on every trial including correct-first-time. Under the draft's own rule
("both sides move -> refuted"), the design self-destructs -- and the predictable response after
seeing that is to relax "bit-identical" post hoc, which is an accommodation hole in the primary
result.

*Fix, both parts:*
(a) A preregistered **equivalence test** with the margin fixed before any run: TOST on end-effector
RMSE against the arm's measured repeat-command dispersion. Note in the preregistration that this is
the same measured quantity as the reach tolerance, so the survive-limb test is exactly as sensitive
as the outcome test -- defensible, but stated rather than discovered.
(b) A **Head-A-only ablation arm**, so the gate-path confound is separable from the blame and
confidence heads.

**MUST COLLAPSE:**

- Held-out repair rate -> the re-plan-from-scratch baseline. *(Primary endpoint -- lowest variance.)*
- Endogenous hold: ticks-on-target on the cued off-salience band -> the no-controller baseline.
  Chance line 0.25 at B=4.
- Held-out blame accuracy -> `1/K_e`. *(Secondary -- see A7(b) on variance.)*
- Type-2 AUROC -> into the shuffle band. *(Reported as difficulty estimation, not metacognition.)*

**It is one ablation, four measures, and three preserved-capability controls.** The draft called it
a "quadruple dissociation," which is a methods-term misuse -- a double dissociation requires two
manipulations, and the four collapse measures are three heads sharing one `zbar`, one `u_t` and one
pin. They will not fail independently.

**Per-head ablations are first-class arms, not sensitivity analysis.** Pin `W_g` alone, `W_b` alone,
`W_c` alone, `W_pred` alone, plus the joint. **Preregistered discriminator: the one-computation
claim of section 2.2 requires that each single-head pin collapses only its own capability AND that
pinning `W_pred` collapses all three. If pinning `W_pred` leaves the heads intact, L3 is three
independent readouts and section 2.2 is refuted.** Under the joint pin alone, one head collapsing
drags the aggregate and the write-up reads "the ACS is load-bearing" -- the design would have been
arranged so the synthesis claim wins under both the true and false versions of itself.

**Both sides move** -> the ACS was a shared resource, not a higher-order layer. Refuted.
**Neither moves** -> the ACS is inert. **Null, and a successful terminal state.**

### 9.2 Two tables, not one

The draft put the CONTENT arm inside a table headed "each must reproduce the collapse, or the
primary is void." The CONTENT arm is not supposed to reproduce the collapse -- its *tie* is the
result. As written, a tie satisfied the validity rule and a win made "the primary is void" available
as a reading, so the document's own most-likely-to-fire falsifier had an escape hatch built into the
table it lived in.

**Controls -- each must reproduce the collapse, or the primary is void:**

| Arm | What it kills |
|---|---|
| zeros | the primary |
| matched-variance Gaussian | "an active pathway mattered, not its content" |
| cross-episode permutation of `u_t` | right marginal, wrong pairing |
| random `g_t` from the same distribution | "the controller learned nothing" |

**Dropped by name:** matched-effective-dim PCA of the ACS's own inputs, and matched-dim random
projection. At `dim(u_t) = 15` a matched-dim PCA is close to the identity and a matched-dim random
projection is close to a rotation. Both tie by construction, and a tie would read as "any bottleneck
would do" when it actually means "there is no bottleneck." Reinstate only if `W` is deliberately
widened, and say why.

**Comparison arms -- the outcome IS the result, not a validity check:**

| Arm | Both outcomes, written before the run |
|---|---|
| **CONTENT arm** (K2 / DX2): a reader fed attended content at identical dimensionality and parameter count | **Tie** -> "a model of attention" collapses into "an extra informative input"; AST's specific commitment is idle for this system and task. **Win for the ACS** -> content-blindness is doing work. Neither is embarrassing. |
| **Symbols arm** (HOST's side of K2): a reader fed the bound step codes | Interpreted against the control-state arm; both are genuinely higher-order, so this isolates content-blindness alone. |

**This arm is only interpretable if G4 has passed.** If `u_t` decodes target identity at accuracy
indistinguishable from the direct content decode, the two arms are information-matched twins and any
tie is a task-design artifact, not a theory verdict.

### 9.3 Baselines that must be able to win -- run FIRST

Four of these are expected to fire. A baseline that wins is a result, not a disqualification.

1. **Re-plan from scratch** (`cortex/executive_control.py:149 plan_sequence`). Most likely to tie on
   a short-horizon task. Match on success rate *and* steps-to-success.
2. **Re-plan with one randomly chosen step forbidden.** Most of the value of knowing *which* step
   may be captured by being forced to change *some* step.
3. **Random-step blame**, >= 7 seeds. Seed discipline is not optional: the Executive table records a
   signal that flipped 17/60 outcomes at seed 0 and 0/60 at seeds 1 and 2 while measurably present
   at full strength.
4. **Episode-length-conditioned prior**: `argmax_i P(j = i | K_e)` fit on train. If `j_e` is
   non-uniform this beats chance with no higher-order anything. Random-step blame does *not* cover
   this -- random is uniform, the prior is not.
5. **Latch-and-hold**: record the band cued during `[0,c)`, output a one-hot, no learning, no `u_t`.
   Ties on any generator where the cue is visible during the cue window, which the task requires.
6. **Logistic regression on `[sigma2_0, sigma2_1, K_e]`** for Type-2 AUROC. Two scalars A2 already
   computes, no ACS. The per-step raw-trace baseline does not cover this: it is per-step and this is
   per-episode.
7. **Raw per-step `(d_i, r_i)` trace** instead of `u_t`. If equal, there is no thought-about-thought
   -- it is a first-order eligibility trace, and `eprop.py` already computes those.
8. **Unordered bundle** (drop `permute`). **Only interpretable after A7(a)'s fix**; before it, the
   tie is guaranteed by the code.

Preregister the margin each must lose by, and **derive the margin from the measured seed-to-seed
spread**. Measure the spread first, write the margin second.

### 9.4 Rigging check

Verify `(gate, pi_hat, ignition)` is **in principle recoverable** from what the pinned arm still
receives. Wilterson pre-empted exactly this: spotlight location was recoverable from the clean patch
in the noisy field, which made the no-schema arm information-sufficient. **If the control state is
not recoverable, the ablation is rigged and the result is void, not positive.** Note this check
passes trivially if band structure is task structure -- which is why step 1 of the task randomizes
the band-to-target mapping, and why G4 asks the *reverse* question.

### 9.5 A null is a successful outcome

Stated plainly, and preregistered as such. If the ACS is pinned and nothing moves, the finding is
that a content-blind model of the control state is not load-bearing in this system on this task.
That is publishable, it closes a question, and it costs less than the alternative -- which is to
retune the mechanism's own constants until a number moves. This repo has the record of doing exactly
that and reverting it: `brainstem/autonomous_loop.py:976-995`, where wiring PFC working memory made
`_wm_spikes` go from 0 to 22 distinct values while held goal, selected action, arm act/hold and
trajectory each changed 0/200. **A mechanism that becomes live and changes nothing is the failure
mode, and the honest response is a null.**

---

## 10. What would change our mind

Falsifiers for the whole approach, not for individual markers. Each is an observation that would end
or redirect the program.

| # | Observation | What it falsifies |
|---|---|---|
| **F1** | **G4 fires**: target identity decodes from `u_t` at accuracy indistinguishable from the direct content decode. | Content-blindness is a naming convention here. AST's specific commitment is untestable on this substrate, and the L3 result is void. **This is the single most likely fatal observation**, because the gate is computed from the content at `learned_attention.py:126-140`. |
| **F2** | **The CONTENT arm ties**, with G4 passed. | "A model of attention" reduces to "an extra informative input." AST's specific commitment is idle for this system and task. The build keeps its engineering value and loses its theory claim. |
| **F3** | **G1 fires**: the task does not break ignition saturation. | Theta is not a controlled variable. DX1, DX3 and DX4 are cancelled, K1 is unanswerable here, and the derived-threshold contribution reduces to "we replaced a literal with a procedure and could not test whether it matters." |
| **F4** | **G3 fires**: mutual information between the held goal at end-of-gap and the first committed action is at the shuffle floor. | The task does not bind. Every downstream null is uninterpretable and must not be reported as "the ACS is inert." |
| **F5** | **G5 fails**: two unperturbed runs from one snapshot do not replay bit-identically. | B1 is not runnable. The repo keeps a surrogate PCI and must say so, and no PCI-based claim about this brain can be made at all. |
| **F6** | **Pinning `W_pred` leaves all three heads working.** | Section 2.2's "L3 is one computation" is refuted. The heads are three independent readouts and the P2 collapse was packaging, not a finding. |
| **F7** | **The planner's own value ranking predicts `true_faulty_step` as well as the ACS does.** | The second-order layer adds nothing. HOT-adjacent and HOST-adjacent claims both go. |
| **F8** | **Any baseline in 9.3 ties within the preregistered margin**, in particular re-plan-from-scratch or latch-and-hold. | Per-step credit assignment, or endogenous control, bought nothing. This is the outcome `docs/CONSCIOUSNESS_METHODS.md` already named as a null and blessed. |
| **F9** | **The dissociation holds at one `lr` and not across the swept decade.** | The result is a property of a tuning choice, not of the mechanism. |
| **F10** | **A marker moves only in the explicit-report arm.** | It is a report artifact. Given block 11, "survives the reduced-report arm" is not a certification either -- it merely fails to disqualify. |
| **F11** | **Any success criterion in this document turns out to have no reachable red.** | That criterion is deleted, not weakened. Three were deleted from the draft on exactly this ground (`divergence_from_first_order`, the zero-sum precision falsifier, the e-prop cosine); two more (dPCI, B5) are conditional on stating one. |
| **F12** | **A theory-relevant result is produced by a step that was measured to be a no-op.** | The whole engineering layer is untrustworthy and every number must be re-derived. Blocks 1, 2, 5 and 7 are each capable of producing this, and each produces it *silently*. |

**F12 is the class of error this document exists to prevent.** It is worse than a marker that cannot
go red, because it produces a marker that goes red -- or green -- for a reason unrelated to the
claim. The draft contained four instances: a theta write to a clobbered attribute, a saturated
comparator ladder, a dict neuter on matrix weights, and a VSA layer nothing read.

---

## 11. Critiques applied and rejected

Three independent adversarial reads were run against the tree. Everything below was checked in the
source before being accepted or rejected.

### Applied in full

Blocks 1-11 and offenders 1-15 above are the applied critiques. Summarized by what they changed:

1. Theta's attach point was the clobber, not a hook -> new `set_base_rheobase()` contract + a
   positive control that cannot be skipped (block 1).
2. The workspace is saturated -> **gate G1, which can cancel three of four discriminating
   experiments** (block 3).
3. The graded access readout is calibrated at the shipped theta -> replaced with `drive_margin`
   (block 2). *No other review caught this.*
4. `PerturbationalComplexity` has no `system` port and the RNG fork will inflate PCI -> B1 is
   blocked on G5; the cheap rename is split out (block 5).
5. Head B's teacher did not exist -> binary-outcome REINFORCE; `j_e` is evaluation-only; blame
   demoted to a secondary endpoint (A7b).
6. The VSA layer was decorative and baseline 8's tie was guaranteed -> `sim(readback(i), hv(a_i))`
   is consumed (A7a).
7. `_PinnedWeights` is a dict subclass -> matrix-safe neuter with readback verification (block 7).
8. `divergence_from_first_order` could not go red -> replaced with the planner-ranking comparison
   (A5).
9. The task leaked target identity into `u_t` via `onehot(focus_band)` -> band-to-target mapping
   randomized per episode; gate G4 (task step 1, F1).
10. "Quadruple dissociation" is a methods-term misuse and the bit-identical survive limb was
    unachievable -> renamed; TOST with a preregistered margin; Head-A-only arm (9.1).
11. The joint pin hid the refutation -> per-head ablations as first-class arms with a preregistered
    discriminator (9.1, F6).
12. "Actual variational free energy" -> `pwse`, with the disclaimer permanent (A2).
13. Step 6's gate was a wiring fact -> restated as an outcome-change count against the seed spread
    (A6).
14. HOT's aboutness commitment is not satisfied by a content-blind reader -> HOT declined explicitly
    (2.3).
15. `THETA`, `LEAK`, `recurrence`, `hold_threshold`, the ACC ladder, and four theta-behind constants
    were missing from the audit -> added (section 8).
16. A2's zero-sum falsifier was an identity, and precision could lock out a level -> falsifier
    re-filed; LIF-derived floor; `ticks_at_floor` as a first-class output (A2).
17. Exact histogram equality and the misapplied crc32 phase surrogate -> sustained log-ratio sign
    change; source-permutation surrogate (A3).
18. dPCI was rate-confounded and unfalsifiable; "dNRS" was an unexpanded acronym -> rate-matching
    required; B7 applied to the whole output list (B1, B5).
19. Four baselines reachable by a lookup table or one-line rule were missing -> added as 9.3 items
    4-7.
20. `max_goals`'s capacity derivation laundered into `D`/`VOCAB` -> dropped by name; the validity
    threat kept separately (offender 6).
21. The e-prop cosine was vacuous for a non-plastic trace -> dropped by name (block 10).
22. `dim(u_t) = 15`, not ~300 -> `C` recomputed; two control arms dropped by name (5.4, 9.2).
23. `Brain.tick(self)` has no stimulus port -> step 2 rescoped and reordered ahead of the log
    (block 4, 5.2).
24. The CONTENT arm sat in the "must reproduce the collapse" table -> two tables (9.2).
25. Theory-adjudication language ("Flat = Rolls") -> rewritten to name computational sub-claims only
    (K1, K3).
26. There is no genuine no-report arm -> renamed explicit-report / reduced-report; the certification
    claim deleted (block 11).
27. Warmup-derived-then-frozen `tau`/`kappa` -> rolling re-estimate (A2, A5).
28. `deliberate()` exists; `maintain()` has six offline call sites; A6 does not close the documented
    feature-space hole -> all three corrected (section 1, A6, A7).
29. Placement: the honest section was sixth of seven -> moved to section 0, above everything.
30. "CSM" reads as "Conscious State Model" -> renamed ACS.
31. B2 (the expander control) promoted to first: independent of G5, ~10 lines, and the only control
    that can make an existing published number look bad.
32. Compute arithmetic and LZ76's quadratic term -> staging plan and a justified cap (blocks 6, 8).

### Rejected, with reasons

**R1. "`_BIND_IGNITE` is a self-check constant and should be demoted off the offender list."**
*Partially rejected.* The header at `global_workspace.py:311-313` does read "used only by the
self-check," and the review was right that listing pure self-check constants inflates an audit. But
the review named a loose end -- `spike_chat_lake.py:34` imports from `consciousness_probe` -- and
chasing it shows the constant reaches `spike_chat_lake.py:925-931`, the live chat reply to "are you
conscious." **A literal gating a reported consciousness marker on a user-facing path is a more
serious offender than one gating behaviour, not a less serious one.** Reclassified, kept.

**R2. "Derive the task gap length from the measured `_wm_spikes` decay; set the gap to ~60% of
where hold falls to the popped baseline."** *Rejected.* The 60% is an unnamed constant, and the
decay it is 60% *of* is governed by `recurrence = 0.98` and `hold_threshold = 0.2` -- two literals
the same review had just added to the offender list. The "derivation" is `f(0.98, 0.2, 0.6)`, which
is exactly the laundering that review correctly objects to for `max_goals`. **Gap length is a swept
independent variable** with the ~80-tick cap reported as a hard ceiling, and results are reported as
a function of it.

**R3. "If content decodes from `u_t` above the shuffle band, the run stops."** *Rejected as a binary
stop; accepted as a graded gate.* Once the band-to-target mapping is randomized per episode, content
leakage becomes a quantity task design can *reduce*, not a yes/no property. The gate is: the CONTENT
arm comparison is interpretable only if the `u_t` decode is below the direct-content decode by a
preregistered margin. Full stop only if they are statistically indistinguishable. Stated as G4 and
F1.

**R4. "The four lookup-table-reachable endpoints show the success criteria are unsound."**
*Rejected as framing; the four baselines are accepted in full.* A baseline that wins is a result --
it says the capability is cheaper than claimed. Treating "a simple rule ties" as a disqualification
rather than a finding is the accommodation move in reverse, and `docs/CONSCIOUSNESS_METHODS.md`
already blesses exactly this outcome as a null.

**R5. "G2 assertion 5 (base failure rate in a stated interval) is a clean generator gate."**
*Accepted as a gate, rejected as compatible with deriving `cal_blame`.* If the failure rate is
tuned into an interval, then `cal_fail` is a tuned quantity and deriving the abandon quantile from
it launders a tuned constant. The two critiques collide; the resolution is offender 8 -- `cal_blame`'s
quantile is a declared convention.

**R6. "The equivalence margin should be the arm's measured repeat-command dispersion."** *Accepted
with a disclosure the critique did not make.* That is the same measured quantity already used as the
reach tolerance, so the survive-limb equivalence test is exactly as sensitive as the outcome test.
Defensible, but it must be stated in the preregistration rather than discovered in review.

### Not a critique, but the one thing all three reads missed

`_BIND_IGNITE` reaching a live user-facing consciousness self-report (offender 4, verified above).
`reasoning/spike_chat_lake.py:1173-1182` answers "are you conscious" with a four-marker tally, and
at least one of those four markers is gated by a hardcoded literal. The reply's own text is honest
and neither claims nor denies experience -- but the tally it reports is arbitrated by a number
nobody derived, on the one path in this repo where a consciousness claim actually reaches a human.

---

## 12. Build order

| # | Step | Gate cleared | Blocked by |
|---|---|---|---|
| 0 | **B2 expander control** | the only refutable PCI control; ~10 lines; independent of everything | -- |
| 1 | Rename `state()['pci']` -> `pci_surrogate`; narrow the bare `except` at `brain.py:2419` | stops a wrong, and stale, number being reported | -- |
| 2 | **Stimulus port + `EpisodeDriver` + G5 determinism gate** | the true critical path; also B1's precondition | -- |
| 3 | **G2**: six generator assertions, generator alone | can invalidate the task before any brain runs | 2 |
| 4 | **G1**: saturation measurement on the task | **can cancel DX1/DX3/DX4** | 2, 3 |
| 5 | **G3**: task-binding pilot, no ACS | protects every downstream null from misattribution | 2, 3 |
| 6 | Per-trial log (A9) | unblocks five measurements | 2 |
| 7 | Fix `"drive": 0.5` (`brain.py:2124`) | makes theta's derivation meaningful | -- |
| 8 | Matrix-safe neuter + readback verification | blocks all of section 9 | -- |
| 9 | Precision inference (A2) with LIF floor and `ticks_at_floor` | removes two identity multipliers; adds a real mechanism | -- |
| 10 | `set_base_rheobase()` contract + range assert, then derived theta (A3) | removes `0.55`; three reachable reds | 4, 7 |
| 11 | `_ACC_LO`/`_ACC_HI` re-derivation per theta, or `drive_margin` | blocks B3 | 10 |
| 12 | `u_t` + ACS + Head A together (A4-A6), **after G4** | Head A must land *with* the reader, against an outcome-change count | 5, 6, 9 |
| 13 | Heads B/C with the binary-outcome third factor and the consumed VSA record (A7-A8) | the HOST capability | 12 |
| 14 | Ablation matrix, per-head arms, report arms (section 9) | **the dissociation result -- access, control and report only** | 6, 8, 12, 13 |
| 15 | Theta sweep -> DX1/DX3/DX4 | the discriminating experiments | 4, 10, 11, 14 |
| -- | B1 real PCI | gated on G5; **if G5 cannot pass, report B1 as not runnable** | 2 |
| -- | GBI, capacity sweep, `max_goals` derivation, matched-dim PCA / random projection, e-prop cosine on the ACS | **dropped by name** | see sections 5.4, 8, 9.2, 7 |
| -- | RPT / DX5 | **deferred**; no figure-ground readout, no feedforward-only perception mode | block 9 |

**None of these steps produces evidence about phenomenal consciousness. See section 0.**

The highest-value item is step 2, and steps 3-5 are the cheapest way to find out whether the rest is
worth building. Two of them can end the program. **Ending it on a measurement is the good outcome;
the bad outcome is a green marker produced by an instrument at its rail.**

---

## 13. A published framework that proposes the opposite: NCAC (arXiv 2405.02370)

Reviewed 2026-08-02 at the owner's request. **Neuromorphic Correlates of Artificial Consciousness**,
Anwaar Ulhaq, Central Queensland University, May 2024, 13 pages, 8 figures, 6,673 words.

Recorded here because it is a sincere attempt at the same goal that lands on the method this
document exists to refuse, and because the next person to find it should not have to re-derive why.

NCAC proposes four phases -- Quantification, Simulation, Adaptation, Implementation -- and states its
method in four assumptions (section 3.1). Paraphrased minimally:

1. Neural correlates of consciousness correlate with high **Phi** values as defined by IIT.
2. Simulating those correlates with SNNs is feasible.
3. **"Employing a gradient descent-like algorithm can efficiently optimize NNACs by adjusting the
   parameters (weights) of the SNNs ... to minimise the loss function"** -- the loss being the gap
   between the artificial system's Phi and the biological Phi.
4. That optimisation converges, and the resulting similarity indicates "the potential for artificial
   consciousness".

### Why we do not adopt it

**It optimises toward the metric it uses as evidence.** Train a system to minimise the distance
between its Phi and a target Phi, and a Phi close to the target is guaranteed by construction. The
number cannot come out wrong, so it cannot be evidence. This is the same defect this repo caught
three times inside one feature on 2026-08-02 -- a success criterion that cannot go red -- arriving
here as the central method rather than as an oversight.

**It requires Phi.** This project computes a perturbational complexity index in
`instruments/integration_measure.py` and forbids calling it Phi. Phi is not tractable for a system of any
size, and the paper does not address tractability while making Phi an optimisation target.

**It requires gradient descent on SNN weights.** Standing rule here is e-prop over backprop: local
plasticity on the live path, never `loss.backward()`. Adopting NCAC's method means abandoning that,
and the cost of the rule is already known and paid (a measured ~6-point accuracy gap on N-MNIST,
disclosed in `docs/DEPTH_CEILINGS.md`).

**Its qualia proposal is verification by report.** The paper suggests validating qualia by giving
machines "expressive artificial intelligence infrastructure" so they can share "the sensation of
redness". Section 0 of this document is about exactly why report is not experience, and every source
theory cited there agrees.

### What it does not contain

`falsif*` appears **0 times in 6,673 words**. No equations, no benchmark, no dataset, no results, no
implementation: "algorithm" 5 times, "we implement" 0, "results show" 0. It is a position paper and
says so honestly -- "this work may raise eyebrows and invite criticism" -- so this is a description
of its genre, not an accusation of overreach.

### What it is good for

As a **negative control on our own framing**. NCAC and this document start from the same premise --
neuromorphic substrate, IIT in the background, artificial consciousness as a serious target -- and
diverge at exactly one decision: whether the consciousness metric is something you OPTIMISE or
something you REFUSE TO TREAT AS EVIDENCE. We took the second branch before reading it. That the
first branch is where a published framework independently landed is the best argument available that
the branch point is real and worth stating as loudly as section 0 states it.
