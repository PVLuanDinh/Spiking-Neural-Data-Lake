r"""
Spiking learning-progress curiosity + self-paced curriculum -- the ONE live self-improvement
primitive (neuromodulation/intrinsic_motivation.py CuriosityDrive + neuromodulation/curriculum.py
Curriculum) rebuilt so its core signal is GENUINELY SPIKING: the learning-progress value is read
off EMITTED SPIKE COUNTS of leaky integrate-and-fire pools, never off a rate EMA or a membrane
potential. Roadmap item SI-1 "SPIKE-LP".

Honest framing (read this first). This is a PURITY conversion, NOT a new capability. On a stationary
or decaying task it behaves like the rate CuriosityDrive it replaces -- the win is that the mechanism
now lives in spikes, so the live-loop self-improvement primitive stops being a rate-ANN drift inside
an otherwise spiking brain. The measured margin is printed, not faked (a standing "the spiking margin
is HONESTLY SMALLER than the rate version's" sentence here described the two earlier forms; the
2026-08-26 EMA rebuild measures at parity on the explorer protocol -- see the behavioural paragraph
-- while its flat-jitter residual stays 1.2-1.6x the rate sibling's). (An earlier sentence here attributed the flat-noise residual
to "rectification of two noisy spike-rate estimates" -- MEASURED WRONG 2026-08-26: the dominant flat
bias of the then-shipped two-pool form was the rheobase mismatch documented in the mechanism section
below, up to +0.53 LP per observe at flat e=0.5, not a small rectification residue. A same-day
intermediate claim that the residual was "<= +0.002 at every level" was ALSO wrong -- measured on an
unjittered protocol while every shipped flat test is jittered, where the true warm residual is
3.4-4.6e-3/observe; the honest statement of both residuals, with their protocols, is in the
mechanism section.) It holds no world model and makes no action decision -- it is the CONSUMER of a
surprise signal.

WHAT THE DROP-IN COSTS BEHAVIOURALLY -- RE-MEASURED 2026-08-26 against the shipped EMA form
(the 2026-08-04/05 figures in parentheses are the TWO-POOL form's, kept for the record). The
protocol is `intrinsic_motivation._run_explorer` -- this module's self-check has no exploration
test while the rate sibling's does -- run as `_run_explorer(Drive(seed=s), 48, greedy=True)` for
s in 0..11, splits written learnable/mastered/noisy:

    chases the learnable frontier   rate 11/12 seeds     spiking 12/12 (two-pool was 10/12)
    seed-0 visit split              rate 46/1/1          spiking 46/1/1 (two-pool 27/6/15)
    MEAN split over the 12 seeds    rate 41.8/1.5/4.7    spiking 44.0/1.5/2.5 (two-pool 24.1/6.8/17.1)
      curiosity-blind random control (greedy=False), same 12 seeds, run on the SPIKING
      drive's own rng: mean 16.3/14.9/16.8

The measured split now MATCHES the rate drive's (12/12 vs 11/12, mean within noise of each
other on 12 seeds) -- PARITY is claimed, superiority is NOT (12 seeds, one world). The honest
negative this paragraph used to carry (the two-pool form's weak, noisy-inflated preference) was
a symptom of the flat-level bias this rebuild removed; what remains honest to say is that the
margin is no longer smaller than the rate version's on this protocol, while the flat-jitter
residual (mechanism section, residual (a)) is still 1.2-1.6x the rate sibling's.

*(Two figures were CORRECTED here on 2026-08-05 by re-running them. The random-control split was
recorded as "13/19/16"; that split occurs at no seed in 0..7 for either drive class, and the
protocol behind it was not recorded -- it is replaced above by the 12-seed mean, whose protocol is.
A sentence claiming "the spiking margin ran 6.7x-248x (mean 61.6x) against the rate drive's flat
18.2x" was DELETED: it recorded no accumulation protocol, so it is not re-derivable, and every
best-faith reconstruction is dominated by a near-zero denominator on the noisy arm and returns
values spanning ten orders of magnitude. The margin this module can honestly quote is the one its
own self-check PRINTS on every run at (2) below -- re-derived by the gate rather than by a comment;
a frozen copy of that number once sat in this sentence and drifted, so no copy sits here now.)*

P2 STATUS, stated because two mechanisms for one job is the defect P2 names. Both drives are live:
`brainstem/autonomous_loop.py` selects between them with a `spiking_curiosity` flag and
`cortex/brain.py` sets it True, so the canonical brain runs this one while the rate original stays
constructible with its own importers. P2 calls a flag that keeps both paths "additive with extra
steps". This is NOT resolved here -- the resolution spans a file outside this module, and the
measurement above does not support retiring EITHER (the spiking one is not better on the behaviour,
the rate one is not spiking, and spiking-ness is a project thesis rather than a benchmark result).
Recorded as open P2 debt with its evidence attached, which is what P2 asks for when neither
mechanism dominates. (EVIDENCE RE-DATED 2026-08-26: an earlier version of this paragraph said
"the spiking one is not better on the behaviour" -- written against the two-pool form's 10/12.
The EMA rebuild measures at PARITY on the explorer protocol (12/12 vs 11/12, means within
noise), so behaviour no longer argues for either side; the live reasons to retire neither are
now (a) the spiking form's flat-jitter residual, still 1.2-1.6x the rate sibling's, and (b)
spiking-ness being a project thesis rather than a benchmark result -- unchanged. A second P2
surface was CREATED by the rebuild and is recorded rather than silent: the fast/slow EMA
arithmetic in observe() is a deliberate 4-line verbatim copy of the rate sibling's estimator
(only the input differs -- normalised LIF counts vs raw error). Extracting the shared
estimator would touch the ADOPTED rate incumbent (the T046a facade's dependency) for zero
behavioural gain, so the copy is named, cross-referenced at the site, and carried with this
same P2 debt: whichever pass resolves the two-drive flag resolves the four lines with it.)

The key idea (unchanged from the rate version). Rewarding raw surprise (high prediction error)
gets permanently stuck on unpredictable NOISE -- a screen of static has maximal error forever (the
"noisy-TV problem"). Developmental curiosity rewards LEARNING PROGRESS = the RATE at which a
context's prediction error is FALLING:
  - HIGH where the model is still improving      (learnable novelty -- the frontier),
  - ~0  where the context is already mastered     (error low and flat -- boring),
  - ~0  where error is high but FLAT / unlearnable (noise -- so it does NOT get stuck).

The SPIKING mechanism (this is the whole point) -- REBUILT 2026-08-26, and the reason is a
measured refutation of the design it replaces, kept here in full:

  THE RETIRED TWO-POOL FORM AND ITS MEASURED DEFECT. The original design kept a FAST (tau 2)
  and a SLOW (tau 16) LIF pool per context, both calibrated ONCE at _E_REF = 0.9, and read
  LP = max(0, slow_rate - fast_rate). Its docstring claimed "when the error is FLAT (mastered-low
  OR noisy-high) both pools settle to the same rate -> LP ~ 0". MEASURED FALSE below the
  calibration level (300 flat observes, 8 seeds, mean steady-state LP per observe):
      e=0.1 +0.040   e=0.2 +0.196   e=0.3 +0.323   e=0.4 +0.444   e=0.5 +0.525 (the peak)
      e=0.6 +0.404   e=0.7 +0.199   e=0.8 +0.106   e=0.9 -0.001   e=1.0  0.000
  CAUSE: a LIF's rheobase scales with its leak -- the fast pool needs drive ~0.39/thr where the
  slow pool needs ~0.06/thr -- so at flat mid-level error the fast pool is fully SUB-THRESHOLD
  (rate exactly 0.0 for e <= 0.4) while the slow pool still fires; one calibration at 0.9 cannot
  equate two different f-I nonlinearities anywhere but at 0.9. Consequence: max(0, rs - rf) paid a
  large PERPETUAL curiosity reward on flat mid-level error -- the exact noisy-TV attractor the
  design exists to prevent (accumulated intrinsic_reward on a flat-0.5 stream, 300 observes:
  139-177 across seeds). A leak-scaled two-pool repair (per-neuron injection scaled by 1-decay,
  equalising the rheobase) was built and MEASURED: it fixes flat (~0 at every level) but destroys
  the signal (learnable accumulated LP 0.018 vs the required > 3.0; explorer 0/12) -- the lag the
  signal needs was carried by the very asymmetry that broke flatness. REFUTED, not shipped.

  A SECOND FORM WAS BUILT, SHIPPED FOR HOURS, AND REFUTED BY ITS OWN REVIEW PANEL (2026-08-26,
  same day -- kept because its failure modes are the instructive ones): ONE tau-2 pool + a SHORT
  and LONG sliding window over its counts. Its flat green was a COLD-WINDOW ARTIFACT: for the
  first _WINDOW observes the two windows hold identical items (difference structurally 0), and
  the calibration reset left the pool phase-synchronised for ~150 more -- the 90-observe test
  protocol lived entirely inside that suppression, while the WARM steady state paid ~5.5e-3
  LP/observe on jittered flat FOREVER (2x the rate sibling's own). Its asserts were seed-lucky
  (6 of 20 seeds red); its tau-2 pool was BLIND below e ~ 0.5 (a falling 0.40 -> 0.05 stream
  registered exactly 0.0 -- a regression vs the two-pool form, whose slow pool saw low levels --
  and the e=0.3 flat assert passed vacuously, certifying the blindness); periodic error paid
  11.6x its own learnable signal (rate: 1.6x); and its learnable signal was mostly WINDOW-DRAIN
  ECHO of the initial drop (a step cliff paid 38% more than genuine exponential learning).

  THE SHIPPED FORM (third, measured against every prior failure): ONE pool whose membrane
  time-constants SPAN [_TAU_LO, _TAU_HI] log-spaced (population coding -- smooth f-I from
  e ~ 0.15 up, closing the rheobase blindness), and the rate sibling's OWN fast/slow EMA pair
  (alpha / alpha_slow, the constructor's existing knobs) run over the pool's normalised emitted
  spike counts (downstream on the defect's own protocol: flat-0.5 accumulated intrinsic_reward
  over 300 observes falls 139-177 -> 0.44-0.64, ~300x):
     learning_progress = max(0, slow_EMA - fast_EMA)   of   count / normaliser.
  Falling error -> the fast EMA drops first -> positive LP; flat error at ANY level -> both EMAs
  estimate the same stationary mean -> ~0, with the two HONEST RESIDUALS measured and stated:
  (a) JITTERED flat rectifies window-sampling noise to a small perpetual positive -- 3.4-4.6e-3
  LP/observe warm steady-state at e=0.5-0.7, which is 1.2-1.6x the rate sibling's own 2.8e-3 on
  the identical streams (a family property of rectified LP, not a spiking-only defect; the
  two-pool form paid up to 0.53 -- two ORDERS more); (b) UNJITTERED flat leaves ~2e-3/observe of
  deterministic mode-locking residue where the rate sibling is exactly 0 (the per-observe count
  settles into a short periodic pattern the EMA pair rectifies; jitter and heterogeneity shrink
  it, the floor is disclosed rather than hidden). Two further FAMILY exposures, stated as
  properties with one recorded-protocol example each because no finite sweep names THE worst
  (a draft of this paragraph quoted its own grid's maximum as "the worst" and a wider grid
  promptly beat it by 19% -- the re-verify lens's blocker; the file's own history already
  deleted one unre-derivable figure, so these are recorded properly or not at all):
  (c) PERIODIC/oscillating unlearnable error pays MULTIPLES of the learnable signal on BOTH
  drives -- example, recorded protocol: square wave 0.1 <-> 1.0, period 60, 300 observes,
  worst of seeds 0..4: spiking 54.7 accumulated LP (2.7x its own learnable-stream signal),
  rate sibling 45.2 (2.4x its own) on the identical stream. Compressed from the window-pair
  form's 11.6x-relative, and within ~15% of the incumbent's exposure -- a shared property of
  rectified LP, unbounded over stream families, disclosed rather than swept. (d) A STEP CLIFF
  in the error pays slightly MORE than smooth genuine learning on BOTH drives -- example,
  recorded protocol: [1.0] then 89 flat-tail observes vs 0.9**t, seeds 0..7: cliff/genuine
  ratio 1.009-1.011 spiking, 1.020 rate -- an EMA-estimator property the spiking form
  inherits at marginally BETTER fidelity than the incumbent (the window-pair form's 38%
  overshoot was the defect; ~1% is the family floor). Measured on the module's own asserted
  protocols: learnable stream accumulated LP ~20 (rate ~19), low-band falling 0.40 -> 0.05
  registers 4.2 (was 0.0), 12-seed explorer 12/12 with mean split 44.0/1.5/2.5 (rate
  41.8/1.5/4.7) -- parity claimed, superiority NOT (12 seeds). First sighting seeds both
  EMAs at the first sample: no transient (LP starts 0).

  The self-paced CURRICULUM likewise replaces its integer low-progress streak with a LIF ACCUMULATOR
  neuron: each low-LP tick injects a fixed "evidence" current and steps the accumulator; a high-LP
  tick RESETS it (still learning -> no premature advance). When the accumulator SPIKES (sustained
  low-LP evidence crosses threshold) the stage is judged LEARNED. The injection is derived from
  `patience` so the accumulator spikes after exactly `patience` consecutive low-LP ticks, while a
  single one-off dip leaks / gets reset and sustained HIGH LP never spikes -- "starting small" made
  spiking.

Both classes duck-type their rate originals: identical constructor signatures and public methods, so
they drop in wherever CuriosityDrive / Curriculum are used (the loop wiring is done by the caller).

Refs: Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational approaches"
(2007); Schmidhuber, "Formal Theory of Creativity, Fun, and Intrinsic Motivation" (2010); Elman,
"Learning and development in neural networks: the importance of starting small" (Cognition, 1993).

Deps: math, random, cortex.lif_neuron.LIFNeuron,
infra._mathutils.mean (numpy/torch allowed project-wide); ASCII-only prints, deterministic (per-instance random.Random(seed);
LIF dynamics are deterministic; spike counts are integers).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.spiking_curiosity
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.infra._mathutils import mean

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_learner_step": (
        "APPROXIMATES_GRADIENT",
        "one frontier-gated gradient step of the shared competence w toward `target`, with the "
        "effective gain decaying as exp(-(gap/_REACH)**2). It is a gradient step and says so",
    ),
    "_run_curriculum_learner": (
        "APPROXIMATES_GRADIENT",
        "the self-paced curriculum harness that drives _learner_step stage by stage; carries the "
        "same claim as the step it applies",
    ),
    "_run_jump_learner": (
        "CONTROL",
        "the jump-to-hardest CONTROL: the SAME learner and SAME budget spent entirely on the hardest "
        "target with no curriculum. It exists to LOSE to the curriculum arm, which is what makes the "
        "comparison mean anything",
    ),
    "SpikingCuriosityDrive._calibrate": (
        "NOT_A_RULE",
        "measures a pool mean spike count at the reference error, then resets. A calibration "
        "measurement",
    ),
    "SpikingCuriosityDrive.learning_progress": (
        "NOT_A_RULE",
        "returns rectified spiking learning progress, max(0, slow_EMA - fast_EMA) over one "
        "pool's normalised emitted counts. A MEASUREMENT of how fast learning is going",
    ),
    "main": (
        "NOT_A_RULE",
        "the self-check harness: drives the declared learner/control arms and measures their "
        "outputs; every gradient step it exercises carries its own declaration above",
    ),
}

# ---- SPIKING learning-progress pool constants (tuned so a decaying error yields clearly positive ----
# ---- accumulated LP while a constant-high-mean+jitter stream yields near-zero LP; NON-saturated,  ----
# ---- NON-mode-locked -- an honest finite anti-noisy-TV margin, not a saturation/mode-lock zero).  ----
_POOL_N = 12  # neurons per pool (averaging shrinks the rectification bias on flat noise)
_SUB = 10  # sub-ticks the error current is injected per observe (finer spike-count resolution)
_TAU_LO = 2.0  # bottom of the pool's membrane time-constant SPAN (was _TAU_FAST, the fast pool)
_TAU_HI = 16.0  # top of the SPAN (was _TAU_SLOW, the slow pool). The retired fast/slow PAIR's
#              two values became the ONE pool's log-spaced population range, 2026-08-26: two
#              narrow single-tau channels (each with its own rheobase) became population coding
#              with a smooth f-I from e ~ 0.15 up. Not new constants -- the same two incumbents,
#              re-purposed with the measured derivation in _make_pool's docstring. The LAG now
#              lives in the fast/slow EMA pair over the pool's counts (the rate sibling's own
#              alpha / alpha_slow), never in a second membrane and never in a sliding-window
#              pair (_WINDOW, 12, retired 2026-08-26 with the refuted window-pair form -- its
#              flat green was a cold-window artifact; see the mechanism section).
_TAU_JIT = 0.22  # per-neuron tau_m heterogeneity (+/- fraction) -- desynchronises, breaks mode-lock
_THR_JIT = 0.16  # per-neuron threshold heterogeneity (+/-) -- smooths the pool rate vs drive
_INJECT_GAIN = (
    0.65  # error -> injected current scale (keeps both pools in the sensitive, non-saturated band)
)
_E_REF = 0.9  # reference error the pool's normaliser is measured at (~1 on flat error there;
#              SCALE only in the EMA form -- the same normaliser feeds both EMAs, so it cannot
#              re-create the two-pool form's cross-calibration defect)
_WARMUP = 40  # calibration observes used to measure the pool's flat-error normaliser
_TAU_ACC = (
    50.0  # curriculum LIF-accumulator membrane time-constant (slow leak -> integrates evidence)
)
_ACC_SAFETY = 1.000002  # tiny margin so the accumulator crosses threshold reliably AT `patience`, not one late

# --- transfer-proof world (ported from the retired rate curriculum.py; the "starting small" control) ---
_TARGETS = (0.25, 0.50, 0.75, 1.00)  # increasing-difficulty milestones of the SHARED competence w
_REACH = 0.30  # frontier of learnability: a target much farther than this yields ~no gradient
_LR = 0.30  # base learning rate (fraction of the in-reach gap closed per tick)


class SpikingCuriosityDrive:
    """A GENUINELY SPIKING learning-progress curiosity drive -- a drop-in for CuriosityDrive.

    observe(k, e) injects the prediction error into ONE heterogeneous-tau LIF pool and folds its
    normalised EMITTED SPIKE COUNT into a fast and a slow EMA (the rate sibling's own pair);
    novelty(k) is a habituating [0,1] recency signal (identical to the rate version);
    learning_progress(k) = max(0, slow_EMA - fast_EMA) is the anti-noisy-TV signal read from spike
    counts (flat at ANY level -> ~0, residuals measured and disclosed; see the module docstring for
    the measured refutations of the two forms this replaced); intrinsic_reward(k, e) observes THEN
    returns the curiosity reward; curiosity_value(k) adds a bootstrapping novelty bonus for a
    curiosity-greedy explorer. Same public interface and constructor defaults as CuriosityDrive.
    Deterministic; per-instance random.Random(seed)."""

    def __init__(
        self,
        alpha=0.2,
        novelty_decay=0.6,
        slow_ratio=0.2,
        novelty_gain=0.5,
        explore_bonus=0.05,
        seed=0,
    ):
        # alpha / alpha_slow keep drop-in signature parity with the rate CuriosityDrive AND are
        # LOAD-BEARING (2026-08-26): the fast/slow separation IS the rate sibling's own EMA pair,
        # run at these exact rates over the ONE pool's normalised emitted spike counts. No membrane
        # pair, no window pair, no second calibration.
        self.alpha = float(alpha)  # the fast EMA rate (the rate sibling's own knob)
        self.alpha_slow = float(alpha) * float(
            slow_ratio
        )  # the slow EMA rate (alpha * slow_ratio, identically the rate sibling's)
        self.novelty_decay = float(
            novelty_decay
        )  # geometric habituation factor per observe, in (0,1)
        self.novelty_gain = float(
            novelty_gain
        )  # how strongly novelty multiplicatively boosts the reward
        self.explore_bonus = float(
            explore_bonus
        )  # additive novelty bonus in curiosity_value (bootstrap only)
        self.rng = random.Random(seed)  # per-instance rng (pool heterogeneity + selector tie-break)
        self._ctx = {}  # context_key -> spiking state (pool, normaliser, fast/slow EMAs)

    # distinct-from: neuromodulation/locus_coeruleus.py -- the LC is an AROUSAL readout
    #   pool (salience -> phasic burst -> downstream gain); this pool transduces a
    #   prediction ERROR into a count for the LP pair. Different signal, different
    #   consumer; term overlap is "pool of LIF neurons" only. Site read.
    def _make_pool(self):
        """Build the ONE pool of _POOL_N LIF neurons whose membrane time-constants SPAN
        [_TAU_LO, _TAU_HI] log-spaced, plus per-neuron tau/threshold jitter.

        WHY A SPAN AND NOT ONE TAU (measured, 2026-08-26): a LIF's rheobase scales with its
        leak, so a single-tau pool has a NARROW sensitive band -- the tau-2.0 pool of the
        refuted window-pair form was fully sub-threshold below e ~ 0.5, blind to all learning
        in the low-error regime (a falling 0.40 -> 0.05 stream registered exactly 0.0), and
        its e=0.3 flat assert passed VACUOUSLY (adversarial panel F3). Log-spacing the taus
        across the band the retired fast/slow PAIR once occupied gives the population a
        smooth f-I from e ~ 0.15 up (0 / 5 / 7 / 26 / 35 / 52 counts at e = 0.1 / 0.2 /
        0.3 / 0.5 / 0.7 / 0.9, measured) -- population coding where the pair had two narrow
        channels. The sensitivity FLOOR ~ 0.15 remains and is disclosed: mastered-low
        contexts (e ~ 0.05) read count 0, which is the correct verdict for them."""
        pool = []
        for i in range(_POOL_N):
            frac = i / (_POOL_N - 1)
            base_tau = _TAU_LO * (_TAU_HI / _TAU_LO) ** frac  # log-spaced across the band
            tau = base_tau * (
                1.0 + self.rng.uniform(-_TAU_JIT, _TAU_JIT)
            )  # per-neuron membrane-tau jitter
            thr = 1.0 + self.rng.uniform(-_THR_JIT, _THR_JIT)  # per-neuron threshold jitter
            pool.append(LIFNeuron(tau_m=tau, v_threshold=thr))
        return pool

    def _drive_pool(self, pool, pred_error):
        """Inject current (_INJECT_GAIN * pred_error) into every neuron for _SUB sub-ticks and return the
        total EMITTED SPIKE COUNT (sum of the 1/0 threshold crossings). THIS is the spiking quantity the
        learning-progress signal is built from -- an integer count of threshold crossings, never a membrane value."""
        current = _INJECT_GAIN * pred_error
        spikes = 0
        for nrn in pool:
            for _ in range(_SUB):
                spikes += nrn.step(current)  # LIFNeuron.step -> 1 (spiked) or 0 (did not)
        return spikes

    # distinct-from: neuromodulation/drive_reduction.py -- its pool spike COUNT proves
    #   growth under constant drive; this measures a NORMALISER for the LP estimator.
    #   Generic-vocabulary overlap only. Site read.
    def _calibrate(self, pool):
        """Measure THE pool's mean spike count at the reference error _E_REF (a flat drive), then reset the
        membranes for a clean start. In the EMA form this normaliser is SCALE ONLY: the same value divides
        the count feeding BOTH EMAs, so it cannot re-create the two-pool form's defect (one calibration
        trying to equate two different f-I nonlinearities -- the measured bias curve in the module
        docstring); it just puts LP in ~unit range (count/norm ~ 1 at _E_REF)."""
        total = 0
        for _ in range(_WARMUP):
            total += self._drive_pool(pool, _E_REF)
        for nrn in pool:
            nrn.reset_state()  # discard the calibration transient; start from rest
        return max(total / _WARMUP, 1e-6)  # mean spikes/observe at _E_REF (guard tiny/zero)

    # distinct-from: neuromodulation/intrinsic_motivation.py -- the RATE sibling this
    #   class duck-types (the recorded open P2 debt in the module docstring). The EMA
    #   PAIR below deliberately imports that sibling's fast/slow EMA STRUCTURE -- run at
    #   the sibling's own alpha/alpha_slow -- while the SIGNAL stays emitted spike
    #   counts of a LIF pool, which the rate sibling has none of.
    def observe(self, context_key, pred_error):
        """Fold one world-model prediction error for context_key into its spiking state: on first sighting build
        + calibrate the ONE heterogeneous-tau LIF pool; every call drive the pool with the error and fold its
        normalised EMITTED SPIKE COUNT into the fast and slow EMAs, and bump the visit count. Returns None."""
        e = float(pred_error)
        st = self._ctx.get(context_key)
        if st is None:
            pool = self._make_pool()  # ONE pool, taus SPANNING the band (see _make_pool)
            st = {
                "n": 0,
                "pool": pool,
                "norm": self._calibrate(pool),  # flat-error normaliser (scale only)
                "fast": None,  # fast/slow EMAs of the normalised count -- the rate
                "slow": None,  # sibling's own pair, fed spikes instead of raw error
            }
            self._ctx[context_key] = st
        st["n"] += 1
        r = self._drive_pool(st["pool"], e) / st["norm"]  # normalised EMITTED count
        if st["fast"] is None:
            st["fast"] = r  # first sighting seeds BOTH EMAs at the first sample:
            st["slow"] = r  # no transient (LP starts exactly 0)
        else:
            # These two lines are a DELIBERATE verbatim copy of the rate sibling's estimator
            # update (intrinsic_motivation.CuriosityDrive.observe) -- the P2 STATUS paragraph
            # carries the adjudication: the copy resolves with the two-drive flag, not before.
            st["fast"] += self.alpha * (r - st["fast"])
            st["slow"] += self.alpha_slow * (r - st["slow"])

    def novelty(self, context_key):
        """Novelty in [0,1]: 1.0 for a never-seen context, HABITUATES as novelty_decay**count with each
        observe() (identical to the rate CuriosityDrive -- novelty is bookkeeping, not the spiking learner)."""
        st = self._ctx.get(context_key)
        if st is None:
            return 1.0
        return self.novelty_decay ** st["n"]

    # distinct-from: neuromodulation/intrinsic_motivation.py -- the rate sibling's
    #   max(0, slow_EMA - fast_EMA) in ERROR units; this one runs the SAME estimator pair
    #   over SPIKE-COUNT rates (the duck-typed drop-in, the recorded open P2 debt). Read.
    # distinct-from: basal_ganglia/learning_progress.py -- the T046a |LP| FACADE, which
    #   READS the adopted rate drive (rectified() delegates); no mechanism there to
    #   duplicate. Site read.
    # distinct-from: plasticity/competitive_learning.py -- its fast/slow ACTIVITY traces
    #   are WTA adaptation state, not an LP estimator; term overlap only. Site read.
    def learning_progress(self, context_key):
        """Rectified SPIKING learning progress = max(0, slow_EMA - fast_EMA) of the ONE pool's normalised
        EMITTED SPIKE COUNTS -- the rate sibling's exact estimator pair (its own alpha / alpha_slow), fed
        spikes instead of raw error. POSITIVE while the context's error is FALLING (the fast EMA has already
        dropped to the new lower count level while the slow EMA still remembers the old higher one); ~0 once
        the error is FLAT at ANY level -- both EMAs then estimate the same stationary mean. Honest residuals,
        both measured and disclosed in the module docstring: window-sampling noise on JITTERED flat streams
        rectifies to a small perpetual positive (1.2-1.6x the rate sibling's own on identical streams), and
        deterministic mode-locking leaves ~2e-3/observe on UNJITTERED flat where the rate sibling is exactly
        0. 0.0 before any observe."""
        st = self._ctx.get(context_key)
        if st is None or st["fast"] is None:
            return 0.0
        return max(0.0, st["slow"] - st["fast"])  # slow > fast only while the count level FALLS

    def intrinsic_reward(self, context_key, pred_error):
        """The curiosity reward the loop ADDS to any extrinsic reward. Calls observe() internally (pass each
        fresh prediction error exactly ONCE), then returns learning_progress * (1 + novelty_gain * novelty) --
        identical composition to the rate version, so a mastered or noisy context (lp ~0) earns ~0 no matter how
        novel. Deterministic."""
        self.observe(context_key, pred_error)
        return self.learning_progress(context_key) * (
            1.0 + self.novelty_gain * self.novelty(context_key)
        )

    def curiosity_value(self, context_key):
        """Expected intrinsic reward of visiting context_key NEXT (no new error yet):
        learning_progress * (1 + novelty_gain * novelty) + explore_bonus * novelty. The additive novelty bonus
        makes an unseen context worth a first taste (bootstrap); once habituated only real spiking learning
        progress sustains value -- so a curiosity-greedy selector chases the learnable frontier, never the noise."""
        lp = self.learning_progress(context_key)
        nov = self.novelty(context_key)
        return lp * (1.0 + self.novelty_gain * nov) + self.explore_bonus * nov

    def most_curious(self, candidates):
        """Return the context_key of maximal curiosity_value among `candidates` (a curiosity-greedy pick).
        Deterministic: the per-instance rng only shuffles first so exact ties are broken without positional
        bias -- same seed + same history -> same choice."""
        cand = list(candidates)
        self.rng.shuffle(cand)
        return max(cand, key=self.curiosity_value)


class SpikingCurriculum:
    """A self-paced developmental curriculum whose advance gate is a SPIKING LIF ACCUMULATOR -- a drop-in for
    Curriculum. observe(lp) feeds the current stage's learning-progress and returns True on the tick it ADVANCES:
    each low-LP tick injects a fixed evidence current into the accumulator neuron and steps it; a high-LP tick
    RESETS it (still learning -> hold); when the accumulator SPIKES (sustained low-LP evidence, ~`patience` ticks)
    the stage is LEARNED. stage / mastered / done / difficulty() / review_stage() are identical to Curriculum.
    Deterministic; per-instance random.Random(seed) (only review_stage draws it)."""

    def __init__(self, n_stages, mastery_threshold=0.02, patience=8, seed=0):
        self.n_stages = max(1, int(n_stages))  # stages indexed 0..n_stages-1
        self.mastery_threshold = float(
            mastery_threshold
        )  # learning_progress at/below this counts as "not learning"
        self.patience = max(
            1, int(patience)
        )  # consecutive low-LP ticks needed to judge a stage LEARNED
        self.rng = random.Random(seed)  # per-instance rng (interleaved-review picks / determinism)
        self._mastered = 0  # number of stages completed (0..n_stages)
        # LIF accumulator replaces the integer low-LP streak. Solve the injection so a fresh accumulator crosses
        # threshold EXACTLY on the `patience`-th consecutive low-LP tick: with leak d = exp(-1/tau), the membrane
        # after k evidence injections is inj*(1-d^k)/(1-d); setting that == threshold at k = patience gives inj.
        d = math.exp(-1.0 / _TAU_ACC)
        self._inject = (
            (1.0 - d) / (1.0 - d**self.patience) * _ACC_SAFETY
        )  # spikes at exactly `patience` low ticks
        self._acc = LIFNeuron(
            tau_m=_TAU_ACC, v_threshold=1.0
        )  # the evidence-accumulating gate neuron
        self._acc.reset_state()

    def observe(self, learning_progress):
        """Feed the CURRENT stage's learning-progress for this tick. Returns True on the tick the curriculum
        ADVANCES. A low-LP tick injects evidence into the accumulator LIF and steps it; if the accumulator SPIKES
        the stage is LEARNED -> advance (mastered++, reset the accumulator). A high-LP tick RESETS the accumulator
        (still learning -> never a premature advance). No-op once done (returns False)."""
        if self.done:
            return False
        lp = float(learning_progress)
        if lp <= self.mastery_threshold:
            if self._acc.step(
                self._inject
            ):  # sustained low-LP evidence crossed threshold -> LEARNED
                self._mastered += 1  # advance to the next (harder) stage
                self._acc.reset_state()  # clear the evidence for the new stage
                return True
            return False
        self._acc.reset_state()  # still learning -> dump the evidence (no premature advance)
        return False

    @property
    def stage(self):
        """Current stage index in 0..n_stages-1 (clamped at the hardest once done)."""
        return min(self._mastered, self.n_stages - 1)

    @property
    def mastered(self):
        """Number of stages completed so far (0..n_stages)."""
        return self._mastered

    @property
    def done(self):
        """True once every stage has been mastered."""
        return self._mastered >= self.n_stages

    def difficulty(self):
        """Current-stage difficulty in [0,1] = stage/(n_stages-1); 0.0 for a single-stage curriculum."""
        if self.n_stages <= 1:
            return 0.0
        return self.stage / (self.n_stages - 1)

    def review_stage(self):
        """Return a random already-mastered stage index for interleaved REVIEW (spaced practice against
        forgetting), or None if nothing is mastered yet. Uses the per-instance rng -> deterministic given the
        seed. Optional; the core advance rule does not consume it."""
        if self._mastered == 0:
            return None
        return self.rng.randrange(self._mastered)


# =====================================================================================================
# Self-check: six deterministic control groups. Every stream is fixed (or drawn from a seeded
# rng) and the LIF dynamics + spike counts are deterministic, so the tail numbers reproduce byte-for-byte.
# Each claim has a CONTROL that MUST fail. The MEASURED margin is printed, NOT asserted to any fixed
# multiple (a "margin is smaller than the rate version's" rule stood here through two retired forms;
# the EMA form measures at or above the rate sibling's margin on most seeds -- printed, never assumed).
# =====================================================================================================

_EPS = 3.5  # ABSOLUTE bound on 90-observe flat/noisy accumulated LP. RE-DERIVED 2026-08-26:
#             the previous 0.5 was calibrated on the window-pair form's cold-window-SUPPRESSED
#             noise (its own panel measured the warm steady state 6x over it) and sat ON the
#             measurement (5.46e-3/obs x 90 = 0.49). Derivation: 20-seed measured max noisy
#             accumulation for the shipped EMA form = 2.69 (the rate sibling measures 1.04 on
#             the IDENTICAL protocol -- the family's own rectified-noise floor, not zero);
#             bound = measured max 2.69 + CHOSEN round headroom (3.5 is a picked ceiling, said
#             as such -- the derivation fixes the floor it must clear, not the exact value).
#             The absolute bound alone cannot carry discrimination; _MARGIN_FLOOR below does.
#             NOTE the two residual ratios in this module are DIFFERENT statistics, both true:
#             1.2-1.6x = WARM steady-state per-observe drift vs the rate sibling; 2.69-vs-1.04
#             = COLD 90-observe accumulation vs the rate sibling on this assert's protocol.
_MARGIN_FLOOR = 5.0  # per-seed learnable/noisy MARGIN floor. Derivation: 20-seed measured
#             minimum per-seed margin = 8.3x (min learnable 19.4, per-seed noisy 0.59-2.69);
#             floor set below the measured minimum with disclosed headroom.
_POS = 3.0  # a genuinely learnable (falling-error) stream must accumulate clearly-positive LP
#             well above this (20-seed measured min 19.4 -- large headroom, kept from the
#             original where it was honest)


def _learner_step(w, target):
    """One frontier-gated gradient step of the shared competence w toward `target`. The effective gain
    decays as exp(-(gap/_REACH)**2): ~1 when the target is within reach, ~0 when it is far -- a stylized
    vanishing-gradient / sparse-reward frontier (progress is only possible NEAR current competence).
    Returns the new w. This is why a far ("hard") target cannot be learned cold, but a nearby one can."""
    gap = target - w
    gain = math.exp(-((gap / _REACH) ** 2))
    return w + _LR * gain * gap


def _run_curriculum_learner(budget, seed):
    """Self-paced SPIKING curriculum learner: train the SHARED competence on the CURRENT stage's target,
    and advance to the next (harder) target when the SpikingCurriculum LIF accumulator detects mastery
    (learning progress dried up). The competence w PERSISTS across stages -> easy stages walk w forward
    into reach of the hard one (transfer). Feeds the per-tick error REDUCTION as the learning-progress
    signal. Returns (final competence w, stages mastered, steps used)."""
    cur = SpikingCurriculum(len(_TARGETS), seed=seed)
    w = 0.0
    steps = 0
    while steps < budget and not cur.done:
        steps += 1
        target = _TARGETS[cur.stage]
        before = abs(target - w)
        w = _learner_step(w, target)
        lp = max(0.0, before - abs(target - w))  # learning progress = error reduction this tick
        cur.observe(lp)
    return w, cur.mastered, steps


def _run_jump_learner(budget):
    """Jump-to-hardest control: the SAME learner and SAME budget, spent ENTIRELY on the hardest target
    with no curriculum. The hardest target starts out of reach (gap 1.0 >> _REACH) so the gradient is
    ~0 and the competence barely moves. Returns the final competence w."""
    w = 0.0
    for _ in range(budget):
        w = _learner_step(w, _TARGETS[-1])
    return w


def _stages_reached(w, tol=0.05):
    """How many milestone targets the competence w has actually attained (within tol)."""
    return sum(1 for t in _TARGETS if w >= t - tol)


def main():
    # ---- (2) ANTI-NOISY-TV AT THE SPIKE LEVEL (the critical control) -- MULTI-SEED.
    # The window-pair predecessor's single-seed asserts were green at their shipped seed
    # and red on 6 of 20 seeds (its panel's F2 -- the same seed-luck class the T052-i
    # panel caught the same day). Five seeds run IN-GATE, deliberately including 7 and 8,
    # two of the predecessor's failing seeds; the 20-seed sweep the bounds derive from is
    # recorded beside _EPS/_MARGIN_FLOOR.
    seed_rows = []
    learn_per = []
    learn_stream = [0.9**t for t in range(90)]  # genuinely learnable: error decays, masters
    srng = random.Random(1234)
    noisy_stream = [
        0.9 + srng.uniform(-0.05, 0.05) for _ in range(90)
    ]  # constant-high mean + jitter, NO downward trend
    for _s in (0, 1, 2, 7, 8):
        dl = SpikingCuriosityDrive(seed=_s)
        dn = SpikingCuriosityDrive(seed=_s)
        a = b = 0.0
        for le, ne in zip(learn_stream, noisy_stream):
            dl.observe("learnable", le)
            dn.observe("noisy", ne)
            lv = dl.learning_progress("learnable")
            a += lv  # accumulate the SPIKING learning-progress ...
            if _s == 1:
                learn_per.append(lv)
            b += dn.learning_progress("noisy")  # ... for the learnable and the noisy context
        seed_rows.append((_s, a, b))
    learn_lp = next(a for _s, a, _b in seed_rows if _s == 1)
    noisy_lp = next(b for _s, _a, b in seed_rows if _s == 1)
    margin = learn_lp / max(noisy_lp, 1e-9)  # the HONEST measured spiking margin (printed)

    # THE LEVEL SWEEP (the reachable red for the retired two-pool design, which held
    # flat->~0 ONLY at the calibration level: at flat e=0.5 it accumulated ~40 LP on this
    # exact protocol -- and its fast pool sat fully sub-threshold for e <= 0.4, which made
    # the window-pair successor's 0.3 arm pass VACUOUSLY; the heterogeneous span fires at
    # 0.3 -- count ~7/observe -- so this arm is now live). WORST seed of five is bound.
    level_lp = {}
    for _lev in (0.3, 0.5, 0.7):
        worst_lev = 0.0
        for _s in (0, 1, 2, 7, 8):
            dz = SpikingCuriosityDrive(seed=_s)
            zrng = random.Random(4321)
            acc_lev = 0.0
            for _ in range(90):
                dz.observe("flat", _lev + zrng.uniform(-0.05, 0.05))
                acc_lev += dz.learning_progress("flat")
            worst_lev = max(worst_lev, acc_lev)
        level_lp[_lev] = worst_lev

    # THE LOW-BAND RED (the window-pair predecessor's F3: its tau-2 pool was BLIND below
    # e ~ 0.5 -- this exact falling 0.40 -> 0.05 stream registered 0.0, indistinguishable
    # from mastered, so a curriculum gate would advance off genuine low-regime learning).
    d_low = SpikingCuriosityDrive(seed=1)
    low_lp = 0.0
    for _t in range(90):
        d_low.observe("low", max(0.05, 0.40 * 0.95**_t))
        low_lp += d_low.learning_progress("low")

    # The former "swap CONTROL" (relabel the streams, assert LP follows the signal) is
    # DELETED under P8, not kept: context_key is an opaque dict key, so the relabeled run
    # is arithmetically THE SAME COMPUTATION -- its numbers were identical to the (2) arms
    # to the last digit and no implementation could fail it while (2) passed (the re-verify
    # lens measured the tautology). A control with no reachable red is deleted, per this
    # module's own precedent.

    # ---- (1) STRUCTURAL-SPIKING: learning_progress IS a function of EMITTED SPIKE COUNTS ----
    # The EMA state holds floats, so integer-ness is asserted where the integers LIVE: at
    # the _drive_pool boundary. The instance's _drive_pool is wrapped (log + delegate)
    # AFTER the calibrating first observe; replaying the logged counts through the same
    # normalise-and-EMA arithmetic must reproduce learning_progress -- LP is a pure
    # function of the count log.
    dstruct = SpikingCuriosityDrive(seed=1)
    dstruct.observe("hot", 0.9)  # first observe builds + calibrates (not logged)
    st = dstruct._ctx["hot"]
    _f0, _s0 = st["fast"], st["slow"]  # the replay's starting point
    _count_log = []
    _orig_drive = dstruct._drive_pool

    def _logging_drive(pool, e2):
        c2 = _orig_drive(pool, e2)
        _count_log.append(c2)
        return c2

    dstruct._drive_pool = _logging_drive
    for _ in range(29):
        dstruct.observe("hot", 0.9)  # drive hard so the pool actually spikes
    counts_are_int = bool(_count_log) and all(isinstance(c, int) for c in _count_log)
    pools_spiked = sum(_count_log) > 0  # the pool crossed threshold (spike_count > 0)
    _rf, _rs = _f0, _s0
    for _c in _count_log:
        _r = _c / st["norm"]
        _rf += dstruct.alpha * (_r - _rf)
        _rs += dstruct.alpha_slow * (_r - _rs)
    lp_from_spikes = max(0.0, _rs - _rf)  # recompute LP purely from the integer counts ...
    lp_matches = (
        abs(lp_from_spikes - dstruct.learning_progress("hot")) < 1e-12
    )  # ... it equals learning_progress()
    # CONTROL that LP reads SPIKE COUNTS and not membranes. The previous control here asked whether
    # the membranes were ints (`all(isinstance(v, int) for v in [nrn.v ...])`) and asserted they
    # were not. That had NO reachable red -- LIFNeuron's v_reset/v_rest default to 0.0 so a membrane
    # list is float on every input, and the pool is never empty (12 units) so `all()` cannot go
    # vacuously True either. Deleted under P8 falsifier F11 and replaced by a NEUTER: scramble every
    # membrane WITHOUT touching the recorded spike-count windows, and require LP to be bit-identical.
    #
    # Two properties of the neuter are load-bearing and were BOTH got wrong on the first attempt
    # (the red would not fire), so they are stated rather than left implicit:
    #   * it must run on a FALLING error stream, where LP is genuinely non-zero. On the flat 0.9
    #     drive above LP is ~0 BY DESIGN, and a neuter applied to a zero signal cannot move it
    #     under any implementation -- it would have looked like a pass while testing nothing.
    #   * the scrambled values must be WILD (far from any real membrane trajectory), so a
    #     membrane-reading LP variant cannot accidentally reproduce its pre-neuter value.
    #     (The two-pool ancestor's version of this note required driving the two pools APART;
    #     with ONE pool the equivalent hazard is writing values a real trajectory could hold.)
    neut = SpikingCuriosityDrive(seed=1)
    for _t in range(20):
        neut.observe("falling", 0.9**_t)  # a live falling-error signal to move
    _nst = neut._ctx["falling"]
    lp_pre_neuter = neut.learning_progress("falling")
    for _i, _nrn in enumerate(_nst["pool"]):
        _nrn.v = 9.0 + 0.01 * _i  # wildly off any real trajectory (thr band ~0.84-1.16)
    lp_post_neuter = neut.learning_progress("falling")
    membrane_neuter_moved = lp_post_neuter != lp_pre_neuter

    # ---- (4) DETERMINISM (curiosity): two seed=7 drives, same stream -> identical LP + reward ----
    det_stream = [("x", 0.9), ("x", 0.7), ("x", 0.55), ("x", 0.4), ("x", 0.3), ("x", 0.25)]
    da, db = SpikingCuriosityDrive(seed=7), SpikingCuriosityDrive(seed=7)
    sig_a = [(da.learning_progress(k), da.intrinsic_reward(k, ev)) for (k, ev) in det_stream]
    sig_b = [(db.learning_progress(k), db.intrinsic_reward(k, ev)) for (k, ev) in det_stream]
    curiosity_det = sig_a == sig_b

    # ---- (3) SPIKING CURRICULUM: LIF-accumulator advance + two controls that MUST fail to advance ----
    ca = SpikingCurriculum(4, patience=8, seed=0)
    adv_tick = None
    for t in range(1, 40):
        if ca.observe(0.0):  # sustained low LP -> accumulator SPIKES
            adv_tick = t
            break
    cb = SpikingCurriculum(3, patience=8, seed=0)
    for _ in range(100):
        cb.observe(0.3)  # CONTROL (b): sustained HIGH LP -> never spikes
    high_mastered = cb.mastered
    cc = SpikingCurriculum(3, patience=8, seed=0)
    dip_advanced = False
    for x in [0.3] * 5 + [0.0] + [0.3] * 20:  # CONTROL (c): a single one-off dip -> leaks/reset
        dip_advanced = cc.observe(x) or dip_advanced
    dip_mastered = cc.mastered
    cd = SpikingCurriculum(5, patience=8, seed=0)
    d_start = cd.difficulty()  # fresh -> stage 0 -> difficulty 0.0
    for _ in range(500):
        cd.observe(0.0)  # master every stage
    d_end = cd.difficulty()  # done -> hardest stage -> difficulty 1.0

    # ---- (4) DETERMINISM (curriculum): two seed=7 curricula, same LP trace -> identical advances/reviews ----
    csrng = random.Random(1234)
    lp_trace = [0.3 + csrng.uniform(-0.05, 0.05) for _ in range(10)] + [
        0.0
    ] * 30  # seeded noisy-then-low LP trace
    cca, ccb = SpikingCurriculum(5, seed=7), SpikingCurriculum(5, seed=7)
    seq_a = [cca.observe(x) for x in lp_trace]
    seq_b = [ccb.observe(x) for x in lp_trace]
    review_a = [cca.review_stage() for _ in range(12)]
    review_b = [ccb.review_stage() for _ in range(12)]
    curriculum_det = (seq_a == seq_b) and (review_a == review_b)
    curr_masters = cca.mastered
    curr_ticks = [i for i, a in enumerate(seq_a) if a]

    # ---- (6) TRANSFER -- curriculum-HELPS via starting small (the HARD, un-riggable control, ported ----
    # from the retired rate curriculum.py). The SPIKING curriculum must keep the property: mastering
    # nearby stepping-stone targets walks the shared competence into reach of the hard one, so within the
    # SAME step budget it reaches FAR higher final competence than a learner thrown straight at the hardest.
    budget = 300
    cur_w, cur_mastered, cur_steps = _run_curriculum_learner(budget, seed=0)
    jump_w = _run_jump_learner(budget)
    cur_stages = _stages_reached(cur_w)
    jump_stages = _stages_reached(jump_w)
    learner_repro = _run_curriculum_learner(budget, seed=0) == _run_curriculum_learner(
        budget, seed=0
    )

    # ---- (5) PURITY vs the rate CuriosityDrive (report only, NOT asserted -- different mechanisms) ----
    from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive

    rate = CuriosityDrive(seed=1)
    rate_peak = 0.0
    rate_plateau = 0.0
    for le in learn_stream:
        rate.observe("L", le)
        rate_peak = max(rate_peak, rate.learning_progress("L"))
    for _ in range(60):
        rate.observe("L", learn_stream[-1])
        rate_plateau = rate.learning_progress("L")

    print("=" * 92)
    print(
        "SPIKING CURIOSITY -- learning-progress from EMITTED SPIKE COUNTS (SI-1 SPIKE-LP; anti-noisy-TV)"
    )
    print("=" * 92)
    print(
        f"pool: N={_POOL_N} sub={_SUB} taus log-spaced {_TAU_LO}..{_TAU_HI}  EMA pair = the rate "
        f"sibling's alpha/alpha_slow  gain={_INJECT_GAIN}  acc tau={_TAU_ACC}"
    )
    print("-" * 92)
    print(
        f"(1) structural  : LP from INTEGER spike counts={counts_are_int}  pools spiked={pools_spiked}  "
        f"LP==spike-count value={lp_matches}  (NEUTER: scrambling every membrane moved LP="
        f"{membrane_neuter_moved} -> LP ignores .v)"
    )
    print(
        f"(2) anti-noisyTV: learnable spiking-LP {learn_lp:.3f}  vs  noisy {noisy_lp:.3f}   "
        f"MEASURED margin {margin:.1f}x  (seed 1; 5-seed rows "
        f"{[(s, round(a, 1), round(b, 2)) for s, a, b in seed_rows]}, margin floor {_MARGIN_FLOOR}x)"
    )
    print(
        f"    level sweep : flat accumulated LP worst-of-5-seeds at e=0.3/0.5/0.7 = "
        f"{level_lp[0.3]:.3f}/{level_lp[0.5]:.3f}/{level_lp[0.7]:.3f} (each <= {_EPS}; the "
        f"retired two-pool form measured ~40 at 0.5); low-band 0.40->0.05 registers "
        f"{low_lp:.3f} (window-pair form: 0.0)"
    )
    print(
        f"(3) curriculum  : sustained-low advance @tick {adv_tick} (=patience 8); sustained-high mastered "
        f"{high_mastered}; single-dip advanced={dip_advanced}; difficulty {d_start:.2f}->{d_end:.2f}"
    )
    print(
        f"(4) determinism : curiosity identical={curiosity_det}  curriculum identical={curriculum_det}  "
        f"(masters {curr_masters}/5 @ {curr_ticks})"
    )
    print(
        f"(5) purity      : spiking learnable-LP peak {max(learn_per):.3f}/plateau {learn_per[-1]:.2e}  vs  "
        f"rate peak {rate_peak:.3f}/plateau {rate_plateau:.2e}  (same shape; win is PURITY)"
    )
    print(
        f"(6) transfer    : spiking curriculum reaches competence {cur_w:.3f} (mastered {cur_mastered}/4, "
        f"{cur_stages} milestones, {cur_steps} steps) vs jump-to-hard {jump_w:.3f} ({jump_stages}) "
        f"reproducible={learner_repro}"
    )
    print("=" * 92)

    # ---- (1) STRUCTURAL-SPIKING asserts (un-riggable: LP is literally a spike-count function) ----
    assert counts_are_int, "learning_progress must be computed from INTEGER emitted-spike counts"
    assert pools_spiked, (
        "the spiking pools must actually cross threshold and spike when driven (spike_count > 0)"
    )
    assert lp_matches, (
        "learning_progress must EQUAL the spike-count-derived value (LP IS a function of spike counts)"
    )
    # NEUTER control: scrambling every membrane must NOT move LP (it is a spike-count function).
    # RED constructed and run (re-derived 2026-08-26 on the ONE-POOL state; the earlier record's
    # variant read the retired fast/slow pool pair) against a membrane-reading LP variant --
    # `mean(nrn.v for nrn in st["pool"])` substituted for the windowed counts. The variant is
    # STATED because without it the figures are not re-derivable: the scramble moves its reading
    # 0.225195 -> 9.055000, so a membrane-reading LP FIRES this assert, while the shipped
    # spike-count LP moves by exactly 0.0 from a live 0.212332.
    assert lp_pre_neuter > 0.0, (
        f"the neuter must be applied to a LIVE signal, else it cannot fail under any "
        f"implementation (LP {lp_pre_neuter:.6f})"
    )
    assert not membrane_neuter_moved, (
        f"LP must be a function of SPIKE COUNTS only -- scrambling the pool membranes moved it "
        f"{lp_pre_neuter:.6f} -> {lp_post_neuter:.6f}"
    )

    # ---- (2) ANTI-NOISY-TV -- learnable clearly positive, noisy bounded ABSOLUTE and by
    # MARGIN, on EVERY in-gate seed (single-seed asserts were the predecessor's F2) ----
    for _s, _a, _b in seed_rows:
        assert _a > _POS, (
            f"learnable (falling error) must accumulate clearly POSITIVE spiking LP at every "
            f"seed (seed {_s}: {_a:.3f})"
        )
        assert _b <= _EPS, (
            f"high-but-flat noise must stay bounded at every seed (seed {_s}: {_b:.3f} <= {_EPS})"
        )
        assert _a > _MARGIN_FLOOR * max(_b, 1e-9), (
            f"the learnable/noisy MARGIN must hold at every seed (seed {_s}: "
            f"{_a:.3f} vs {_b:.3f}, floor {_MARGIN_FLOOR}x)"
        )
    for _lev, _acc in level_lp.items():
        assert _acc <= _EPS, (
            f"flat error at level {_lev} must stay bounded (worst-of-5-seeds {_acc:.3f} <= "
            f"{_EPS}) -- the retired two-pool design accumulated ~40 at level 0.5 on this "
            f"protocol (the noisy-TV attractor; measured curve in the module docstring)"
        )
    assert low_lp > 1.0, (
        f"a falling 0.40 -> 0.05 stream must REGISTER ({low_lp:.3f} > 1.0) -- the window-pair "
        f"predecessor measured exactly 0.0 here (blind below its rheobase, its panel's F3)"
    )
    # (the swap-control asserts that stood here are deleted with their arm above -- a
    # tautology over opaque keys, no reachable red)

    # ---- (3) SPIKING CURRICULUM asserts (advance via LIF accumulator; both controls must NOT advance) ----
    assert adv_tick == 8, (
        f"sustained low LP must SPIKE the accumulator and advance at exactly patience=8 ({adv_tick})"
    )
    assert high_mastered == 0, (
        "CONTROL: sustained HIGH LP must never spike the accumulator (no premature advance)"
    )
    assert not dip_advanced and dip_mastered == 0, (
        "CONTROL: a single one-off low dip must leak / get reset by the next high tick -> no spike, no advance"
    )
    assert d_start == 0.0 and d_end == 1.0, (
        "difficulty must rise 0 -> 1 as the curriculum is mastered end to end"
    )

    # ---- (4) DETERMINISM ----
    assert curiosity_det, (
        "two SpikingCuriosityDrive(seed=7) fed the same stream must give identical LP + reward"
    )
    assert curriculum_det, (
        "two SpikingCurriculum(seed=7) fed the same LP trace must give identical advances + reviews"
    )

    # ---- (6) TRANSFER -- the HARD, un-riggable control (ported): starting small must FAR exceed jump-to-hard ----
    assert cur_w > 0.9, (
        "the self-paced spiking curriculum must reach HIGH final competence within budget"
    )
    assert jump_w < 0.1, (
        "jump-to-hardest must stay stuck (the hard target starts out of the learnability frontier)"
    )
    assert cur_w > 5.0 * jump_w, (
        f"curriculum competence {cur_w:.3f} must FAR exceed jump-to-hard {jump_w:.3f} (transfer via starting small)"
    )
    assert cur_mastered > jump_stages and cur_stages > jump_stages, (
        "the spiking curriculum learner must master MORE milestones than the jump-to-hardest control on the same budget"
    )
    assert learner_repro, (
        "the seeded spiking curriculum learner must reproduce its (competence, mastered, steps) exactly"
    )

    print(
        "self-check OK: GENUINELY SPIKING -- learning_progress read from INTEGER spike counts (a membrane .v "
        "readout fails the int-count control);"
    )
    print(
        f"  ANTI-NOISY-TV preserved at the spike level: learnable LP {learn_lp:.3f} >> noisy {noisy_lp:.3f} "
        f"(MEASURED margin {margin:.1f}x, noisy bounded absolute + by margin floor on 5 seeds);"
    )
    print(
        f"  curriculum advances via a LIF ACCUMULATOR at tick {adv_tick}=patience (sustained-high + single-dip both "
        f"hold at mastered 0); deterministic (mean learnable LP {mean(learn_per):.3f});"
    )
    print(
        f"  on a stationary/decaying task it matches the rate CuriosityDrive in shape (peak {max(learn_per):.3f} vs "
        f"{rate_peak:.3f}) -- the win is spiking PURITY, not new capability;"
    )
    print(
        f"  TRANSFER (hard): starting-small spiking curriculum reaches competence {cur_w:.3f} ({cur_stages}/4 "
        f"milestones) vs jump-to-hard {jump_w:.3f} ({jump_stages}/4) on the same {budget}-step budget."
    )


if __name__ == "__main__":
    main()
