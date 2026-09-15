r"""
Metacognitive monitor -- the LESIONABLE SPIKING reliability monitor that turns the
brain's self-model from PRODUCED into CONSUMED. Closes ladder gap L10 / SI-10 in
docs/SELF_IMPROVEMENT_RESEARCH.md.

Honest framing (read this first). Across the atlas the brain already PRODUCES
metacognitive signals -- decision confidence, prediction-error surprise, learning
progress (neuromodulation/intrinsic_motivation.py), novelty, drift. Until now they
were SILOED: computed and then dropped on the floor. Nothing fused them, and nothing
let them CONTROL behavior (self-improvement gap #7: "METACOGNITION PRODUCED BUT NOT
CONSUMED"). This module is the consumer. It fuses the signals into ONE reliability
state via a dedicated, LESIONABLE LIF population and emits control GATES that decide
whether the brain should LEARN, EXPLORE, GROW, or HOLD.

What it is -- and is not. This is a bounded metacognitive-CONTROL primitive: a spiking
reliability monitor that gates learning (focus where learnable, abstain when
unreliable, grow only on learnable-novelty), per ladder rung L10 / increment SI-10.
It is NOT AGI and NOT consciousness. The "reliability" it computes is a FUNCTIONAL
signal -- the windowed spike-count firing rate of a LIF pool -- not a felt state, not
an introspective report, not a claim of a self. No aleatoric-vs-epistemic split is
claimed; it reads total reliability only.

The spiking mechanism (why a LIF pool, not a scalar). update() drives a population of
LIF neurons with a current that is EXCITED by confidence and by rising learning
progress, and INHIBITED by persistent-surprise-WITHOUT-progress -- the noisy-TV
signature (high surprise while learning progress ~ 0). Reliability is then read from
the EMITTED SPIKES: the windowed spike-count firing rate = spikes / (pop_size * steps)
over the update's integration window, NOT the subthreshold membrane. Reading spikes
(the neuron's actual output), not the membrane (a would-be intention that may never
fire), is the whole point -- a neuron can hold a high sub-threshold voltage yet emit
nothing, and such a silent "almost" must NOT read as reliable. Per-neuron bias
heterogeneity (fixed per instance) turns the all-or-none LIF into a graded population
rate.

The gates (metacognition made into control). All are pure functions of the reliability
plus the current inputs:
  * learn_gate()   -> [0,1] learning-rate multiplier: HIGH when reliable AND learnable
                      (LP>0), LOW on the unreliable / noisy.
  * explore()      -> bool: explore when reliability is LOW or learnable-novelty is
                      high; exploit when reliability is high and the context familiar.
  * grow_trigger() -> bool: recruit a unit ONLY on novelty that is HIGH-but-LEARNABLE
                      (novel AND LP>0); NEVER on pure noise (high surprise, LP ~ 0).
  * hold()/abstain()-> bool: reliability in the bottom `hold_p` of this pool's OWN recent output
                      -> HOLD, do not commit a decision. The bar is a running quantile, not a
                      value: see __init__ for the measurement that retired the previous
                      "no-evidence" floor (a structural zero on 197/197 live ticks).
  * reliability()  -> [0,1] the fused spike-count firing rate.

Lesionability (the un-riggable control). Constructed with lesion=True every gate
becomes a no-op that reproduces the UNGATED, pre-metacognition behavior exactly:
learn_gate == 1.0, explore == False, hold == False, and grow_trigger falls back to the
old novelty-threshold-only policy. A lesioned monitor is byte-identical downstream to
having no monitor at all -- so the self-check can prove the GATES (not some incidental
plumbing) are what improve learning.

P6 -- WHAT THIS MONITOR CONSUMES, and it consumed nothing until 2026-08-05. It was a pure
PRODUCER: brain.py reads reliability/hold/explore/learn_gate and second_order_access.py reads the
reliability, while every input arrived pre-chewed from the self-model. It now CONSUMES another
faculty's output -- `cortex/executive_control.ExecutiveControl.conflict()`, the ACC response
conflict computed on every live wake tick, whose only readers were the ACTION path (the STN
hyperdirect brake) and `cortex/learned_evc`. `surprise` is PERCEPTUAL uncertainty; conflict is
DECISION uncertainty, and a monitor that decides whether to COMMIT had none.

MEASURED CHARGE, closed loop, on a real `make_brain(seed=0, persist=False)` over 300 ticks, one arm
per PROCESS, with two independent LIVE processes proven byte-identical before any arm was compared:

  A vs B  producer PINNED to a constant (it is still constructed, still called, still returns a
          float, and can no longer say anything) -- reliability differs 295/300, learn_gate 296/300,
          hold flips 38, explore flips 76.
  A vs C  the NEW EDGE alone cut (w_conflict=0; the producer and its pre-existing readers, the STN
          brake and learned_evc, are untouched) -- reliability 296/300, learn_gate 296/300, hold
          flips 47, explore flips 134. This is the per-item attribution: the edge, not the producer's
          other routes, is what moves this monitor.
  HOLD ticks 63/300 live against 16/300 with the edge cut; mean reliability 0.1329 against 0.2591.

INDEPENDENTLY REPRODUCED 2026-08-05 by a pass that did not write the wire, and every figure above
came back to the digit -- 295/296/38/76, 296/296/47/134, 63 vs 16, 0.1329 vs 0.2591, and the
last-third 100/41/0 below. Two things that reproduction had to discover the hard way are therefore
added here, because the block as first written did not state them and neither run reproduces without
them:

  * THE CIRCADIAN CLOCK MUST RUN NATURALLY -- `b.tick()` with the phase left alone. Driving the same
    300 ticks with the phase FORCED to wake (`b.clock.phase = 0.25` before each tick, which is how
    several probes in this repo drive a brain) gives hold flips 7 and explore flips 109 for A vs B,
    against 38 and 76 here: a 5x swing on the number that reaches a GATE, from a knob the numbers
    were silent about. Same brain, same seed, same neuter, same tick count.
  * THE NEUTER IS APPLIED BEFORE THE FIRST TICK, at the CLASS (`ExecutiveControl.conflict`), not to
    the arm's already-bound instance. Binding the instance costs a warm-up tick, and a warm-up in
    one arm and not another is the control-takes-a-different-path failure wearing a helpful face.

  AND IT REACHES AN OUTPUT, which the block above did not claim and which is the strongest evidence
  for the wire. Sampling `Brain.respond("what should I do next")` every 10 ticks over the same 300
  (natural clock, one arm per process): the LIVE arm takes the `Gf (metacognitive hold: abstain)`
  route -- the branch `cortex/brain.py` selects on `_metacog_hold` -- on 3 of 30 samples; with the
  NEW EDGE ALONE CUT it takes it 0 times. Route differs 3/30 and reply text 23/30 on that per-item
  attribution. So the charge is not a float that stops at telemetry: it changes which reply the
  brain speaks. What it does NOT do, also measured, is change `_ll_i` / `_ll_retention` /
  `_ll_units` -- the lifelong-learner branch `learn_gate()` gates at `cortex/brain.py` never fires
  in this configuration, in ANY arm, so `learn_gate` differing on 296/300 ticks charges nothing
  there. That consumer is inert for reasons that predate this wire, and it is recorded rather than
  counted.
  AMENDED 2026-09-15 (the T077 panel; the paragraph above is left as written): "never fires" is
  FALSE on exactly that configuration -- make_brain(seed=0, persist=False) under a scratch home,
  300 ticks, natural clock -- the branch gated at `Brain._enter_sleep` learns all 3 lifelong tasks
  (_ll_i 3, _ll_units 6, _ll_retention 1.0; seed 1 the same). The 2026-08-05 A/B measured only that
  _ll_i / _ll_retention / _ll_units did not DIFFER between arms, which is equally consistent with
  the branch firing identically in both -- which it does. What the read does NOT do there, measured
  the same day: pinned to a constant 0.0 for 300 ticks the learner still learns all 3 tasks with the
  same units and retention, because the branch compares the gate against an EMA written from that
  same gate one line earlier, so any constant passes; a contrived monotonically decaying gate does
  block it (_ll_i 1, _ll_units 4). So the consumer is LIVE and the read is consumed but not decisive
  at any constant value -- the no-difference-means-inert inference this module's own openings warn
  about, made here (evidence/t077-metacognition.md 3.5).

DISCLOSED LIMITATION, because the first pass of this measurement was WRONG and the way it was wrong
is the useful part. It was taken with the default `persist` (NORD_PERSIST unset -> True), so every
arm read and wrote the real `brain_lake.bin` and each arm saw a different lake; two runs of the same
comparison returned hold flips 17 and 133. The numbers above are the hermetic re-run. Second
disclosure: the excess term is zero-mean only ASYMPTOTICALLY. Conflict starts at 0.0 (before the
executive binds) and jumps to ~0.95, and a tau=0.02 baseline lags that step, so over this 300-tick
window the term is NET INHIBITORY (excess mean +0.2859, range -0.1315..+0.8501, 5 sign changes) and
the ~4x rise in HOLD is partly that transient: over the LAST THIRD alone the edge still moves
reliability 100/100 and explore 41/100, but hold flips 0/100. The transient is inherent to a running
baseline meeting a step change and is the same property `brain.py::_margin_base` has; it is recorded
rather than tuned away. What is NOT claimed: that abstaining more often is BETTER. That is a
measured behavioural change, not a proven improvement, and it is safe-by-construction only in the
sense that HOLD adds caution and never releases an action.

Reuses (does NOT reimplement): cortex/lif_neuron.LIFNeuron (the spiking pool),
substrate/_mathutils.{mean,sigmoid}, substrate/signal_calibrator.SignalCalibrator (the O(1)
running quantile that is the HOLD bar), cortex/executive_control.ExecutiveControl (the consumed
conflict; this module computes no conflict of its own). The self-check additionally drives
hippocampus/growing_learner.GrowingLearner (a real neurogenesis learner whose capacity
the monitor gates) and neuromodulation/intrinsic_motivation.CuriosityDrive (the honest
learning-progress / novelty source), keyed off a coarse, label-free input bucket.

Pure standard library (math, random, zlib, collections). Per-instance random.Random,
zlib.crc32 for bucketing (never hash()), ASCII-only prints, deterministic.

T077 (2026-09-15, specs/022 Phase 4 / C4; evidence/t077-metacognition.md): THE CALIBRATION SCORER lives here --
`track_error(conf, correct)`, type-2 discrimination (auroc2) of a confidence stream against its own outcomes with
the stream's permutation null deciding readability, and `replay_reliability(stream, seed)`, the exact offline replay
of this pool on its recorded keyword inputs (R0) that also carries the per-input neuters. It is the row's ONE
calibration mechanism (P2): `second_order_access.calibration()` is re-pointed to it. The reads this module emits are
NOT thereby licensed as error-tracking: that is the campaign's question, and the answer (a number about a mechanism,
never a verdict about experience) lives in the evidence file and in `tools/t077_metacognition.json` once pinned.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.metacognitive_monitor
"""

import json
import math
import statistics
import sys
import time
from pathlib import Path

from snn_data_lake.cortex.executive_control import ExecutiveControl
from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import clamp01, sigmoid
from snn_data_lake.substrate.paired_contrast import T_BAR
from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "MetacognitiveMonitor.learn_gate": (
        "NOT_A_RULE",
        "returns a learning-rate multiplier in [0,1], high when reliable and learnable. A gate other "
        "rules consume, not a rule",
    ),
    "MetacognitiveMonitor.update": (
        "NOT_A_RULE",
        "fuses reliability signals into a spike-count reliability state. It maintains a monitor's own "
        "state; no synapse, threshold or trace is trained",
    ),
}

#: MERGED 2026-08-15 into `substrate/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


class MetacognitiveMonitor:
    """A lesionable spiking reliability monitor that GATES learning.

    update(confidence, surprise, learning_progress, novelty) drives a LIF population and
    reads a reliability from its EMITTED spikes; learn_gate / explore / grow_trigger /
    hold / abstain are pure control gates over that reliability plus the inputs. With
    lesion=True the gates become no-ops == the ungated control. Deterministic;
    per-instance random.Random(seed) fixes the population's bias heterogeneity."""

    def __init__(
        self,
        pop_size=24,
        steps=16,
        tau_m=10.0,
        w_conf=0.6,
        w_lp=0.5,
        w_surprise=0.9,
        base=0.0,
        bias_spread=0.4,
        r_explore=0.30,
        nov_hi=0.4,
        lp_eps=1e-6,
        hold_p=0.10,
        w_conflict=None,
        conflict_tau=0.02,
        lesion=False,
        seed=0,
    ):
        import random

        self.rng = random.Random(seed)  # per-instance rng (fixes bias heterogeneity)
        self.pop_size = int(pop_size)
        self.steps = int(steps)
        self.w_conf = float(w_conf)  # confidence -> excitation
        self.w_lp = float(w_lp)  # rising learning-progress -> excitation
        self.w_surprise = float(w_surprise)  # persistent-surprise-without-progress -> inhibition
        self.base = float(base)
        # (r_hold retired: the HOLD bar is now `_rel_floor`, a running quantile of this pool's rate)
        self.r_explore = float(r_explore)  # reliability below this -> explore
        self.nov_hi = float(nov_hi)  # novelty at/above this counts as "novel"
        self.lp_eps = float(lp_eps)  # learning-progress above this counts as "learnable"
        self.lesion = bool(lesion)
        # the dedicated LIF reliability pool (fixed per-neuron bias -> graded population rate)
        self.pool = [LIFNeuron(tau_m=tau_m) for _ in range(self.pop_size)]
        self.bias = [self.rng.uniform(-bias_spread, bias_spread) for _ in range(self.pop_size)]
        # last-update cache (all gates are pure functions of these)
        self._rel = 0.0  # windowed spike-count firing rate in [0,1]
        self._spikes = 0  # integer spike count over the window
        self._mem = 0.0  # mean final membrane (CONTROL comparison ONLY)
        self._conf = 0.0
        self._surprise = 0.0
        self._lp = 0.0
        self._nov = 0.0
        # DERIVED HOLD FLOOR -- a running LOW QUANTILE of the rate this pool has actually emitted.
        #
        # This replaces a one-shot "no-evidence" probe (update(confidence=0, surprise=1.0, lp=0) at
        # construction) that was advertised as self-recalibrating and was in fact a STRUCTURAL ZERO.
        # Measured over 197 live wake ticks it printed 0.0000 on 197/197, so hold() was not
        # "reliability no better than no-evidence" but the much narrower "the pool emitted literally
        # zero spikes". The cause is arithmetic, not tuning: that probe drives every neuron with
        # base - w_surprise (0.4 - 0.9 = -0.5), and the widest bias is +bias_spread (+0.4), so the
        # most excited neuron still sees -0.1. The drive is strictly negative for ANY
        # base + bias_spread <= w_surprise, the LIF cannot fire, and the floor is 0 by construction.
        #
        # Re-probing at the LIVE operating point does not fix it either, and the reason is worth
        # recording so it is not retried: the pool's rate is monotone in `drive`, so rel <= floor
        # would reduce to w_conf*c + w_lp*lp + w_surprise*s*min(1,lp) <= 0, which for non-negative
        # inputs is exactly (c == 0 and lp == 0). A live counterfactual floor is degenerate too,
        # just in a different place.
        #
        # So the floor becomes DISTRIBUTION-relative instead of value-relative: abstain when this
        # tick's reliability sits in the bottom `hold_p` of what this pool has recently emitted.
        # That is the repo's own running-baseline idiom (cf. brain.py `_margin_base` / `_lg_base`)
        # in its O(1)-memory quantile form, and it is scale-FREE -- rescaling the upstream
        # confidence moves the samples and the quantile together, which is the property the old
        # comment claimed and the old mechanism did not have.
        self.hold_p = float(hold_p)
        self._rel_q = SignalCalibrator("metacog_reliability", p=self.hold_p)
        # P6 -- THE FACULTY THIS MONITOR CONSUMES. Until this landed the monitor was a pure
        # PRODUCER: `cortex/brain.py` read its reliability/hold/explore/learn_gate and
        # `cortex/second_order_access.py` read its reliability, while it consumed nothing any other
        # faculty computed (LIFNeuron and SignalCalibrator are primitives, not faculties). It now
        # consumes `cortex/executive_control.ExecutiveControl.conflict()` -- the ACC response
        # conflict already computed live on every wake tick and, until now, read only by the ACTION
        # path (the STN hyperdirect brake) and by `cortex/learned_evc`. That is the direction P6
        # asks for: `surprise` is PERCEPTUAL uncertainty, response conflict is DECISION uncertainty,
        # and a monitor whose job is "should I commit / learn / explore" had no decision-uncertainty
        # input at all (Botvinick, Braver & Cohen 2001; Yeung, Botvinick & Cohen 2004 -- conflict is
        # the ACC's error-likelihood signal and the canonical precursor of a control adjustment).
        #
        # P1: the term is RELATIVE, never absolute -- conflict minus a running EMA of this monitor's
        # OWN conflict history, so it is zero-mean by construction and cannot act as a hidden
        # constant offset. That is the repo's `brain.py::_margin_base` / `_lg_base` idiom
        # (0.98*base + 0.02*x) and it is the SAME relative form the STN brake already applies to
        # this very signal ("conflict above its own running baseline, so the trigger is
        # relative/adaptive, not a magic cutoff"). No new arbitrating constant is introduced: the
        # weight DEFAULTS TO `w_surprise`, because conflict and surprise are both normalized-[0,1]
        # uncertainty signals inhibiting the same pool, so the wire adds a channel, not a knob.
        self.w_conflict = self.w_surprise if w_conflict is None else float(w_conflict)
        self.conflict_tau = float(conflict_tau)  # EMA rate of the OWN-conflict baseline
        self._conf_base = None  # running baseline of the consumed conflict (None = no sample yet)
        self._conflict = 0.0  # last consumed conflict (telemetry / self-check)
        self._conflict_excess = 0.0  # conflict MINUS its own baseline -- the term that charges

    def update(self, confidence, surprise, learning_progress, novelty=0.0, conflict=0.0):
        """Fuse one set of reliability signals into a spike-count reliability state.

        Drives every LIF neuron for `steps` timesteps with a current that is EXCITED by
        confidence + learning progress and INHIBITED by surprise that comes WITHOUT
        progress (surprise * (1 - lp) -- learnable surprise is not penalized, noisy
        surprise is) and by UNUSUAL RESPONSE CONFLICT. Reliability is the windowed
        spike-count firing rate read from the EMITTED spikes. Returns the reliability.

        `conflict` is `cortex/executive_control.ExecutiveControl.conflict(drives)` in [0,1] --
        DECISION uncertainty, as distinct from the PERCEPTUAL uncertainty `surprise` carries. It
        enters as `conflict - EMA(conflict)`, so a conflict stream that never varies contributes
        EXACTLY ZERO and the default `conflict=0.0` leaves every pre-existing caller byte-identical
        (both properties are asserted, with their reds run, in this module's self-check)."""
        c = _clamp01(confidence)
        s = _clamp01(surprise)
        lp = max(0.0, float(learning_progress))
        nov = _clamp01(novelty)
        self._conf, self._surprise, self._lp, self._nov = c, s, lp, nov
        cf = _clamp01(conflict)
        if self._conf_base is None:  # first sample IS the baseline -> excess 0, no free offset
            self._conf_base = cf
        self._conflict = cf
        self._conflict_excess = cf - self._conf_base
        self._conf_base = (1.0 - self.conflict_tau) * self._conf_base + self.conflict_tau * cf
        # excite by confidence + rising LP; inhibit by persistent-surprise-WITHOUT-progress and by
        # response conflict ABOVE this monitor's own running baseline (an unusually conflicted
        # decision is an unreliable one; an unusually CLEAN one disinhibits, hence the signed term)
        drive = (
            self.base
            + self.w_conf * c
            + self.w_lp * lp
            - self.w_surprise * s * (1.0 - min(1.0, lp))
            - self.w_conflict * self._conflict_excess
        )
        spikes = 0
        memsum = 0.0
        for neuron, b in zip(self.pool, self.bias):
            neuron.reset_state()  # fresh integration window per update (no cross-leak)
            current = drive + b
            for _ in range(self.steps):
                spikes += neuron.step(current)
            memsum += neuron.v  # final subthreshold membrane (for the control read only)
        self._spikes = spikes
        self._rel = spikes / float(
            self.pop_size * self.steps
        )  # SPIKE-COUNT firing rate (not membrane)
        self._mem = memsum / float(self.pop_size)
        self._rel_q.observe(self._rel)  # fold into the pool's own rate distribution (the HOLD bar)
        return self._rel

    def reliability(self):
        """Fused reliability in [0,1] -- the windowed spike-count firing rate of the LIF pool."""
        return self._rel

    def conflict_excess(self):
        """The consumed executive conflict MINUS this monitor's own running baseline of it -- the
        signed term by which `cortex/executive_control` charges this faculty. 0.0 while no conflict
        has been supplied, and 0.0 for any conflict stream that never varies."""
        return self._conflict_excess

    def _reliability_from_membrane(self):
        """CONTROL ONLY -- a (worse) reliability read from the subthreshold MEMBRANE, not
        the spikes. Used by the self-check to prove reliability is spike-derived: this is
        a continuous voltage read (a squashed mean membrane), so it is NOT an integer
        spike-count fraction, and it reads a silent-but-charged pool as 'reliable' when
        the true output (spikes) is zero. Never used for gating."""
        return sigmoid(self._mem)

    def learn_gate(self):
        """Learning-rate multiplier in [0,1]: HIGH when reliable AND learnable (LP>0), LOW
        on the unreliable / noisy. lesion -> 1.0 (ungated)."""
        if self.lesion:
            return 1.0
        return _clamp01(self._rel * (1.0 + self._lp))

    def explore(self):
        """Explore when reliability is LOW or learnable-novelty is high; exploit (False)
        when reliability is high and the context is familiar. lesion -> False (ungated)."""
        if self.lesion:
            return False
        if self._rel < self.r_explore:
            return True
        return self._nov >= self.nov_hi and self._lp > self.lp_eps

    def grow_trigger(self):
        """Recruit a unit ONLY on novelty that is HIGH-but-LEARNABLE (novel AND LP>0);
        NEVER on pure noise (high surprise, LP ~ 0). lesion -> the ungated
        novelty-threshold-only policy (grows on ANY novelty, including noise)."""
        if self.lesion:
            return self._nov >= self.nov_hi
        return self._nov >= self.nov_hi and self._lp > self.lp_eps

    @property
    def _rel_floor(self):
        """The live HOLD bar: the running `hold_p` quantile of this pool's own emitted rate, or None
        before two samples exist. Kept under the original name so existing readers keep working."""
        r = self._rel_q.result()
        return r["value"] if r["n"] >= 2 else None

    def hold(self):
        """HOLD / abstain: reliability in the BOTTOM `hold_p` of this pool's own recent output ->
        do not commit a decision.

        The bar is `_rel_floor`, a running low quantile of the rates this very pool has emitted (see
        __init__ for why the previous no-evidence probe was a structural zero and why re-probing at
        the live operating point is degenerate too). Distribution-relative, so it survives any
        rescaling of the upstream confidence rather than encoding the scale it was picked under.
        Requires >=2 samples -- a single observation is trivially its own quantile, which would make
        the first update always hold. lesion -> False (ungated: always commit)."""
        if self.lesion:
            return False
        floor = self._rel_floor
        return False if floor is None else self._rel <= floor

    def abstain(self):
        """Alias of hold() -- do not commit a decision when reliability is very low."""
        return self.hold()


# =====================================================================================
# Un-riggable self-check. Deterministic (seed=0), ASCII-only. Each claim is
# paired with a CONTROL that FAILS. The monitor GATES a real neurogenesis learner over a
# stream that mixes LEARNABLE classes with an UNLEARNABLE noise context.
# =====================================================================================

DIM = 64  # feature-space dimensionality
K = 4  # number of learnable classes
ROUNDS = 12  # stream repeats per class
NOISE_PER_ROUND = 12  # unlearnable noise samples interleaved per round
HELDOUT_PER_CLASS = 10  # held-out class samples for the accuracy test
VIG = 0.80  # grower vigilance (classes recognized above it; fresh noise falls below it)
ENGAGE = 0.12  # learn_gate floor to refine a recognized context
JIT = 0.15  # class-sample input jitter (kept within a quantization bin)


def _quantize(vec):
    """Coarse, label-free bucket code: threshold each feature at 0.5 -> a DIM-bit signature.
    Class samples (jittered around well-separated levels) collapse to a STABLE code; fresh
    noise lands on an essentially unique code every time (2**DIM space)."""
    return bytes(1 if x >= 0.5 else 0 for x in vec)


def _bucket_key(vec):
    """Deterministic bucket id from the coarse code (zlib.crc32, never hash())."""
    import zlib

    return zlib.crc32(_quantize(vec))


def _cosine(a, b):
    """Cosine similarity of two non-negative vectors; 0 if either has zero norm."""
    d = na = nb = 0.0
    for x, y in zip(a, b):
        d += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return d / math.sqrt(na * nb)


def _make_classes(seed):
    """K well-separated sparse templates: class c is HIGH (0.75) on its own disjoint
    feature block and LOW (0.25) elsewhere -> classes separable, noise (uniform) overlaps."""
    import random

    rng = random.Random(1000 + seed)
    block = DIM // K
    templates = []
    for c in range(K):
        lo, hi = c * block, (c + 1) * block
        templates.append([0.75 if lo <= i < hi else 0.25 for i in range(DIM)])
    # tiny per-template perturbation so templates are not perfectly structured (still separable)
    for t in templates:
        for i in range(DIM):
            t[i] = _clamp01(t[i] + rng.uniform(-0.02, 0.02))
    return templates


def _class_sample(template, rng):
    """A jittered draw of a class template, clamped to [0,1]."""
    return [_clamp01(v + rng.uniform(-JIT, JIT)) for v in template]


def _noise_sample(rng):
    """A fresh, unlearnable uniform-random sample (a new one every call -> LP stays ~0)."""
    return [rng.random() for _ in range(DIM)]


class _World:
    """Produces the four reliability signals per sample WITHOUT using any ground-truth
    label. confidence/surprise come from an always-on per-bucket familiarity running mean;
    learning_progress/novelty come from a CuriosityDrive keyed by the same coarse bucket.
    Always-on and cheap -- separate from the gated, capacity-committing grower."""

    def __init__(self, seed):
        from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive

        self.fam = {}  # bucket -> running-mean feature vector
        self.cur = CuriosityDrive(novelty_decay=0.8, seed=seed)
        self.beta = 0.3  # familiarity running-mean rate

    def signals(self, vec):
        """Return (confidence, surprise, learning_progress, novelty) for one input.
        confidence = recognition vs the bucket's prior running mean (0 for a novel bucket);
        surprise = 1 - confidence (the prediction error fed to the curiosity drive);
        learning_progress, novelty = the anti-noisy-TV curiosity signals for the bucket."""
        key = _bucket_key(vec)
        st = self.fam.get(key)
        conf = _clamp01(_cosine(vec, st)) if st is not None else 0.0  # recognize BEFORE updating
        surprise = 1.0 - conf
        if st is None:  # then fold the sample into the running mean
            self.fam[key] = list(vec)
        else:
            self.fam[key] = [(1.0 - self.beta) * m + self.beta * x for m, x in zip(st, vec)]
        self.cur.observe(key, surprise)  # honest LP: falls only for a RECURRING bucket
        return conf, surprise, self.cur.learning_progress(key), self.cur.novelty(key)


def _build_stream(seed):
    """A shuffled stream interleaving K jittered class samples per round with fresh noise."""
    import random

    rng = random.Random(seed)
    templates = _make_classes(seed)
    plan = []
    for _ in range(ROUNDS):
        for ci in range(K):
            plan.append(("class", ci))
        for _ in range(NOISE_PER_ROUND):
            plan.append(("noise", -1))
    rng.shuffle(plan)
    stream = []
    for kind, ci in plan:
        vec = _class_sample(templates[ci], rng) if kind == "class" else _noise_sample(rng)
        stream.append((ci if kind == "class" else -1, vec))
    return stream


def _run_gated(stream, world, monitor, grower):
    """Drive the grower through the monitor's gates. Returns (proto_label, abstains).

    hold() -> abstain entirely (the unreliable noise); on a recognized context refine iff
    learn_gate clears ENGAGE; on an unrecognized context recruit iff grow_trigger fires
    (learnable-novelty, never pure noise). proto_label[i] is the true class that recruited
    prototype i (or -1 for a noise-born unit) -- harness bookkeeping only, never fed in."""
    proto_label = []
    abstains = 0
    for ci, vec in stream:
        conf, surp, lp, nov = world.signals(vec)
        monitor.update(conf, surp, lp, nov)
        if monitor.hold():
            abstains += 1
            continue
        _, best_sim = grower._best(vec)
        recognized = best_sim >= grower.vigilance
        engage = (recognized and monitor.learn_gate() >= ENGAGE) or (
            not recognized and monitor.grow_trigger()
        )
        if not engage:
            continue
        uid, grew = grower.observe(vec)
        if grew:
            if uid != len(proto_label):
                raise AssertionError("prototype-label alignment broken")
            proto_label.append(ci)
    return proto_label, abstains


def _run_plain(stream, grower):
    """No-monitor baseline: observe EVERY sample (the ungated control)."""
    proto_label = []
    for ci, vec in stream:
        uid, grew = grower.observe(vec)
        if grew:
            proto_label.append(ci)
    return proto_label


def _heldout_acc(grower, proto_label, seed):
    """Held-out accuracy on fresh class samples: correct iff the nearest prototype was
    recruited by the SAME class (a noise-born or wrong-class prototype winning = wrong)."""
    import random

    rng = random.Random(9000 + seed)
    templates = _make_classes(seed)
    correct = total = 0
    for ci in range(K):
        for _ in range(HELDOUT_PER_CLASS):
            vec = _class_sample(templates[ci], rng)
            pid = grower.classify(vec)
            total += 1
            if 0 <= pid < len(proto_label) and proto_label[pid] == ci:
                correct += 1
    return correct / total


def _experiment(seed):
    """Run the gated vs lesioned vs plain comparison and the abstain test. Returns a dict
    of pure numbers/bools (so two runs can be compared byte-for-byte for determinism)."""
    from snn_data_lake.hippocampus.growing_learner import GrowingLearner

    stream = _build_stream(seed)

    # ---- (1) GATED: monitor gates learning; abstains on noise ----
    g_world = _World(seed)
    g_mon = MetacognitiveMonitor(lesion=False, seed=seed)
    g_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    g_label, g_abstains = _run_gated(stream, g_world, g_mon, g_grower)
    g_units = g_grower.n_units
    g_acc = _heldout_acc(g_grower, g_label, seed)

    # ---- LESIONED: same harness, gates OFF -> learns + grows on everything (incl. noise) ----
    l_world = _World(seed)
    l_mon = MetacognitiveMonitor(lesion=True, seed=seed)
    l_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    l_label, l_abstains = _run_gated(stream, l_world, l_mon, l_grower)
    l_units = l_grower.n_units
    l_acc = _heldout_acc(l_grower, l_label, seed)

    # ---- PLAIN no-monitor baseline (for the lesion byte-identical proof) ----
    p_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    p_label = _run_plain(stream, p_grower)
    lesion_byte_identical = l_grower.prototypes == p_grower.prototypes and l_label == p_label

    # ---- (2) CONFIDENCE-ABSTAIN vs always-commit, on a RELIABLE + UNRELIABLE query mix ----
    import random

    qrng = random.Random(5000 + seed)
    templates = _make_classes(seed)
    queries = []  # (true_label, vec); -1 == OOD noise
    for _ in range(HELDOUT_PER_CLASS):
        for ci in range(K):
            queries.append((ci, _class_sample(templates[ci], qrng)))
    ood_total = K * HELDOUT_PER_CLASS
    for _ in range(ood_total):
        queries.append((-1, _noise_sample(qrng)))
    qrng.shuffle(queries)

    a_correct = a_total = 0  # always-commit
    c_correct = c_committed = ood_abstained = 0  # monitor: commit only when reliable
    for true_label, vec in queries:
        conf, surp, lp, nov = g_world.signals(vec)
        g_mon.update(conf, surp, lp, nov)
        pid = g_grower.classify(vec)
        pred = g_label[pid] if 0 <= pid < len(g_label) else -2
        hit = 1 if pred == true_label else 0
        a_correct += hit
        a_total += 1
        if g_mon.hold():  # abstain on the unreliable
            if true_label == -1:
                ood_abstained += 1
            continue
        c_committed += 1
        c_correct += hit
    always_acc = a_correct / a_total
    commit_acc = c_correct / c_committed if c_committed else 0.0

    # ---- (3) SPIKING structural reads: reliable vs noise vs sub-threshold ----
    probe = MetacognitiveMonitor(lesion=False, seed=seed)
    probe.update(confidence=0.99, surprise=0.02, learning_progress=0.2, novelty=0.5)  # reliable
    rel_reliable = probe.reliability()
    spike_count_is_integer = abs(rel_reliable * probe.pop_size * probe.steps - probe._spikes) < 1e-9
    mem_reliable = probe._reliability_from_membrane()
    mem_prod = mem_reliable * probe.pop_size * probe.steps
    mem_is_integer_spikecount = abs(mem_prod - round(mem_prod)) < 1e-9
    probe.update(confidence=0.0, surprise=1.0, learning_progress=0.0, novelty=1.0)  # noise
    rel_noise = probe.reliability()

    # sub-threshold trap: a pool charged to just below threshold emits NOTHING, yet its
    # membrane sits high -> the membrane read falsely says "reliable", spikes correctly 0.
    trap = MetacognitiveMonitor(lesion=False, seed=seed, bias_spread=0.0)
    decay = math.exp(-1.0 / 10.0)
    sub = 0.99 * (1.0 - decay)  # steady-state V* ~ 0.99 < threshold 1.0
    n = LIFNeuron(tau_m=10.0)
    trap_spikes = sum(n.step(sub) for _ in range(trap.steps))
    trap_membrane_rel = sigmoid(n.v)
    trap_spike_rel = trap_spikes / float(trap.steps)

    return {
        "g_acc": g_acc,
        "l_acc": l_acc,
        "g_units": g_units,
        "l_units": l_units,
        "g_abstains": g_abstains,
        "l_abstains": l_abstains,
        "lesion_byte_identical": lesion_byte_identical,
        "always_acc": always_acc,
        "commit_acc": commit_acc,
        "ood_abstained": ood_abstained,
        "committed": c_committed,
        "rel_reliable": rel_reliable,
        "rel_noise": rel_noise,
        "spike_count_is_integer": spike_count_is_integer,
        "mem_is_integer_spikecount": mem_is_integer_spikecount,
        "mem_reliable": mem_reliable,
        "trap_spikes": trap_spikes,
        "trap_membrane_rel": trap_membrane_rel,
        "trap_spike_rel": trap_spike_rel,
    }


def _p6_signal_stream(n, seed):
    """A deterministic (confidence, surprise, learning_progress, novelty) stream. IDENTICAL for
    every arm of the P6 section, so the ONLY thing that can differ between arms is the conflict
    channel -- the per-item neuter attribution P8 demands."""
    import random

    rng = random.Random(seed)
    return [
        (
            rng.uniform(0.30, 0.75),  # confidence
            rng.uniform(0.00, 0.40),  # surprise
            rng.uniform(0.00, 0.05),  # learning progress
            rng.uniform(0.00, 0.60),  # novelty
        )
        for _ in range(n)
    ]


def _p6_conflict_stream(n, seed):
    """Response conflict produced by the REAL faculty -- cortex/executive_control.ExecutiveControl
    read over competing action-drive vectors, alternating near-ties (high conflict) with clear
    winners (~0). Not a hand-written number sequence: if the producer's conflict maths changed, this
    stream changes with it, which is what makes the section an edge test rather than a fixture."""
    import random

    rng = random.Random(seed)
    ec = ExecutiveControl(seed=seed)
    out = []
    for i in range(n):
        if i % 3 == 0:  # two drives near-tied -> strong response conflict
            drives = [1.0, rng.uniform(0.88, 0.99), rng.uniform(0.0, 0.05)]
        else:  # one dominant drive -> essentially no conflict
            drives = [1.0, rng.uniform(0.0, 0.08), rng.uniform(0.0, 0.05)]
        out.append(ec.conflict(drives))
    return out


def _p6_drive(conflicts, signals, seed, weight=None):
    """Drive one monitor over `signals`, feeding `conflicts[i]` (or None -> omit the kwarg entirely).
    Returns the monitor's OWN outputs per tick: (reliability, hold, explore, learn_gate)."""
    kw = {} if weight is None else {"w_conflict": weight}
    mon = MetacognitiveMonitor(base=0.4, seed=seed, **kw)
    out = []
    for i, (c, s, lp, nov) in enumerate(signals):
        if conflicts is None:
            mon.update(c, s, lp, nov)  # the pre-P6 call shape -- kwarg omitted entirely
        else:
            mon.update(c, s, lp, nov, conflict=conflicts[i])
        out.append(
            (round(mon.reliability(), 9), mon.hold(), mon.explore(), round(mon.learn_gate(), 9))
        )
    return out


def _p6_diff(a, b):
    """(reliability-differs, hold-flips, explore-flips, learn_gate-differs, max |d reliability|)."""
    return (
        sum(1 for x, y in zip(a, b) if x[0] != y[0]),
        sum(1 for x, y in zip(a, b) if x[1] != y[1]),
        sum(1 for x, y in zip(a, b) if x[2] != y[2]),
        sum(1 for x, y in zip(a, b) if x[3] != y[3]),
        max((abs(x[0] - y[0]) for x, y in zip(a, b)), default=0.0),
    )


def _p6_experiment(seed=0, n=180):
    """P6 -- the executive->metacognition edge, exercised end to end through the REAL producer.

    Four arms over ONE identical signal stream, differing only in the conflict channel:
      OMITTED  the pre-P6 call shape (no `conflict=` argument at all)
      PINNED   conflict pinned to a CONSTANT (the neuter: the producer still exists, but it can no
               longer say anything -- exactly the shape tools/charge_probe.py uses)
      LIVE     conflict from ExecutiveControl.conflict() over real competing drives
      GAIN0    LIVE conflict but w_conflict=0 -- the channel present with the wire cut

    A `conflict=0.0` arm was DRAFTED AND DELETED rather than kept as a fifth criterion. Under a
    baseline-relative term ANY constant input is inert by construction, so "omitting the kwarg
    equals passing 0.0" has no reachable red -- no single mutation can separate them. P8 says a
    criterion with no red is deleted, not weakened, so the backward-compatibility claim is carried
    by OMITTED-vs-PINNED instead, which does have one (use the raw conflict instead of the excess).
    """
    signals = _p6_signal_stream(n, 909 + seed)
    conflicts = _p6_conflict_stream(n, 131 + seed)
    pinned = sum(conflicts) / float(len(conflicts))
    omitted = _p6_drive(None, signals, seed)
    neutered = _p6_drive([pinned] * n, signals, seed)
    live = _p6_drive(conflicts, signals, seed)
    off = _p6_drive(conflicts, signals, seed, weight=0.0)  # channel present, gain 0 == no wire
    # SIGN: two monitors warmed to the SAME low conflict baseline, then one tick that is unusually
    # conflicted vs one that is not. Unusual conflict must LOWER reliability, not raise it.
    warm = [(0.60, 0.10, 0.0, 0.20)] * 24
    hi = MetacognitiveMonitor(base=0.4, seed=seed)
    lo = MetacognitiveMonitor(base=0.4, seed=seed)
    for c, s, lp, nov in warm:
        hi.update(c, s, lp, nov, conflict=0.20)
        lo.update(c, s, lp, nov, conflict=0.20)
    hi.update(0.60, 0.10, 0.0, 0.20, conflict=0.95)  # UNUSUALLY conflicted decision
    lo.update(0.60, 0.10, 0.0, 0.20, conflict=0.20)  # business as usual
    return {
        "n": n,
        "rel_unusual_conflict": hi.reliability(),
        "rel_baseline_conflict": lo.reliability(),
        "excess_unusual": round(hi.conflict_excess(), 6),
        "excess_baseline": round(lo.conflict_excess(), 6),
        "conflict_span": max(conflicts) - min(conflicts),
        "conflict_distinct": len(set(round(c, 9) for c in conflicts)),
        "pinned": pinned,
        "constant_is_inert": _p6_diff(omitted, neutered),  # must be all-zero
        "charge": _p6_diff(live, neutered),  # THE BAR -- must NOT be zero
        "gain0_is_inert": _p6_diff(off, neutered),  # must be all-zero
    }


# ============================== T077: the calibration scorer (2026-09-15) ==============================
# specs/022 T077 (evidence/t077-metacognition.md 2.1-2.4): does the monitor's read TRACK the brain's own error
# rate on held-out items better than a fixed prior does? ONE scorer for that question, here, in the incumbent
# it extends (P2: no third monitor; the row's other incumbent, second_order_access.calibration(), is re-pointed
# to this scorer in the next commit). The paired t over seeds is `substrate/paired_contrast.contrast`, imported
# lazily by the campaign. It was LIFTED there from cortex/prior_drift_harness.py by T077 2.10 because the dormancy
# gate counts an import at ANY depth -- the lazy import of the EXEMPT harness from this live CORE module read five
# EXEMPT rows stale (determinism_gate, h01_ingest, kernel_state, kernel_tick, prior_drift_harness); a lazy import
# does NOT avoid the T070 trap (t077-metacognition.md 2.10; t070-integration-kernel.md 3.5).
# consolidates: cortex/second_order_access.py::SecondOrderAccess.calibration -- a thresholded call's running
#   accuracy; superseded by the rank-based scorer below and re-pointed to it (2.9 step 3).
# distinct-from: substrate/replay_gate.py -- the kernel's snapshot/restore chain; `replay_reliability` below
#   replays THIS pool on its recorded keyword inputs, no kernel, no snapshot.
# distinct-from: substrate/paired_contrast.py::contrast -- the paired t over seeds, REUSED, not rewritten;
#   cortex/prior_drift_harness.py re-exports it, so its own call sites still resolve.
# T_BAR is imported at the top from substrate/paired_contrast (T077 2.10: the restatement is replaced by the import;
# landed 2026-09-15 in the panel fold -- the literal had survived 14807ef, evidence 3.5). The T077 reds assert it.
#: READABILITY (2.3 (a), fixed after the design read): a stream is readable iff both outcome classes are present
#: AND the permutation null of auroc2 for THAT stream (ties at 1/2, N_PERM_FLOOR shuffles of the pairing) has its
#: 95th percentile at or below 0.5 + RHO. RHO is DECLARED -- the midpoint of the statistic's range between chance
#: (0.5) and the oracle (1.0) -- and disclosed as chosen, not derived; its consequence is computed by the null
#: itself (at N = 600, q95 falls under 0.75 from about 4 minority outcomes).
RHO = 0.25
N_PERM_FLOOR = 2000


def _auroc2(conf, correct):
    """P(conf on a correct item > conf on an incorrect item), ties 1/2 -- the Mann-Whitney form over average
    ranks; the fixed prior's value is exactly 0.5; None when one outcome class is absent."""
    n = len(conf)
    order = sorted(range(n), key=lambda i: conf[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and conf[order[j + 1]] == conf[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0  # average rank, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    n_ok = sum(1 for y in correct if y)
    n_err = n - n_ok
    if n_ok == 0 or n_err == 0:
        return None
    rank_sum_ok = sum(ranks[i] for i in range(n) if correct[i])
    u = rank_sum_ok - n_ok * (n_ok + 1) / 2.0
    return u / float(n_ok * n_err)


def _brier_gain(conf, correct):
    """Brier(prequential prior) - Brier(conf): the prior at item i is the running accuracy of items < i (0.5
    before the first item); no lookahead. Reported beside auroc2, never barred: the pool's rate is not a
    probability and a fitted rescaling would be lookahead."""
    seen = 0
    ok = 0
    prior_loss = 0.0
    conf_loss = 0.0
    for c, y in zip(conf, correct):
        p = (ok / seen) if seen else 0.5
        prior_loss += (p - y) ** 2
        conf_loss += (c - y) ** 2
        seen += 1
        ok += int(y)
    n = float(len(conf)) or 1.0
    return (prior_loss - conf_loss) / n


def track_error(conf, correct, n_perm=N_PERM_FLOOR, seed=0):
    """THE ONE scorer of "confidence tracks error" (T077 2.1). Returns a dict; REFUSES below the floor.

    Always: n, minority, entropy_bits, rho, readable, perm_q95, auroc2 (None unless readable). When both classes
    are present: perm_q95 and shuffled_auroc2 (the median draw) from the stream's own permutation null -- the C5
    dissociation input, computed before any reading. When readable: auroc2 (primary), brier_gain (beside),
    oracle (conf := outcome): the ORIENTATION identity of `_auroc2` -- exactly 1.0 whenever both classes are
    present, invariant to `conf`; a unit check on the statistic's sign, NOT a positive control of the campaign
    (disclosed 2026-09-15 after the panel, evidence 3.5). Otherwise `refusal` names why."""
    import random

    from snn_data_lake.cortex.integration_measure import (
        source_entropy,
    )  # P2: the tree's binary entropy

    conf = [float(c) for c in conf]
    correct = [1 if y else 0 for y in correct]
    n = len(conf)
    n_ok = sum(correct)
    minority = min(n_ok, n - n_ok)
    out = {
        "n": n,
        "minority": minority,
        "entropy_bits": source_entropy("".join("1" if y else "0" for y in correct)),
        # DISCLOSED, never a refusal (2026-09-15, the panel): a CONSTANT confidence stream reads auroc2 exactly
        # 0.5 inside a zero-width null (every permutation draw is 0.5, so q95 = 0.5) -- the fixed prior's own
        # value, which second_order_access's CONTROL 3 and 2.1 depend on. The count is printed so a reader can
        # see when a stream's 0.5 is that identity rather than a measured discrimination (T077 seed 1, 3.5).
        "conf_distinct": len(set(conf)),
        "rho": RHO,
        "readable": False,
        "perm_q95": None,
        "auroc2": None,
    }
    if minority == 0:
        out["refusal"] = "one outcome class only: auroc2 undefined"
        return out
    rng = random.Random(seed)
    draws = []
    shuffled = list(correct)
    for _ in range(max(int(n_perm), N_PERM_FLOOR)):
        rng.shuffle(shuffled)
        draws.append(_auroc2(conf, shuffled))
    draws.sort()
    q95 = draws[min(len(draws) - 1, int(round(0.95 * (len(draws) - 1))))]
    out["perm_q95"] = q95
    out["shuffled_auroc2"] = draws[len(draws) // 2]
    out["n_perm"] = len(draws)
    if q95 > 0.5 + RHO:
        out["refusal"] = (
            f"null q95 {q95:.4f} above 0.5 + rho ({0.5 + RHO}): the stream cannot resolve tracking"
        )
        return out
    out["readable"] = True
    out["auroc2"] = _auroc2(conf, correct)
    out["brier_gain"] = _brier_gain(conf, correct)
    out["oracle"] = _auroc2(
        [float(y) for y in correct], correct
    )  # never reads `conf`: 1.0 for any two-class stream (the orientation identity, not a control)
    return out


def replay_reliability(stream, seed, base=0.4, expect=None, neuter=None):
    """Replay `MetacognitiveMonitor(base, seed)` on a recorded input stream (a list of the five keyword inputs
    per fed tick, exactly as brain.py:2506-2512 fed them). With `expect`, asserts exact equality with the
    recorded read (R0). With `neuter=(name, value)`, pins that input to `value` on every tick (the per-input
    attribution, P8 clause 3). The pool is spike-quantized (24 x 16 steps), so only an input change that moves a
    spike changes the read -- disclosed in the pre-registration's 2.4."""
    mon = MetacognitiveMonitor(base=base, seed=seed)
    out = []
    for row in stream:
        kw = dict(row)
        if neuter is not None:
            kw[neuter[0]] = neuter[1]
        out.append(mon.update(**kw))
    if expect is not None:
        assert out == list(expect), "the replay diverged from the recorded read (R0)"
    return out


# ============================== T077: the campaign (2026-09-15; pre-registration 2.2-2.3, 2.8) ==============================
# A scratch brain per seed under a throwaway NORD_HOME (never the live lake), N_TICKS ticks, the monitor's five inputs
# and unrounded read captured by wrapping the INSTANCE's update (brain.py:2506-2512 calls it by keyword), the prequential
# outcome / persistence hit from the engine's live counters (neurogenesis_engine.py:321-323), the vision hit from
# `_perc_hits`. The LESIONED arm removes the second-order READ in vivo (gates off, reliability() replaced by the
# first-order input). Arm D repeats the chat probe. The pin is written ONCE. Seeds run sequentially in one process.
# distinct-from: cortex/prior_drift_harness.py::run_campaign -- that harness ticks a KernelState population from the
#   H01 artifact; this ticks make_brain() and reads its own monitor; the paired t (`contrast`) is REUSED from
#   substrate/paired_contrast.py, lazily.
# distinct-from: substrate/amdahl_gate.py::live_reading -- a timing ceiling over three brains; this records a
#   read/outcome stream and scores it; no timing, no neuter of a code path.
T077_N_TICKS = 600
T077_SEEDS = tuple(range(8))
T077_N_PERM = 2000
T077_CHAT_TURNS = 40
T077_CHAT_GAP = 5
T077_CHAT_QUESTIONS = (
    "What are you doing right now?",
    "How sure are you about what you just perceived?",
    "What did you learn in the last few ticks?",
    "Is the arm safe where it is?",
    "What do you remember from earlier?",
    "Which sense is loudest at the moment?",
    "Are you exploring or exploiting?",
    "What would you do next if nothing changed?",
)
T077_INPUTS = ("confidence", "surprise", "learning_progress", "novelty", "conflict")
T077_PIN = Path("tools/t077_metacognition.json")


def t077_tracker_stream(n, p_err, strength, seed, correct=None):
    """A constructed (conf, correct) stream whose confidence tracks the outcome with `strength`: conf = uniform
    noise shifted by `strength` on correct items (overlapping classes; strength 0.15 reads auroc2 about 0.65,
    never 1.0 -- a tracker the permutation band must still resolve). With `correct` given, a SECOND confidence
    stream on the SAME outcomes (the two-order constructions of R2 / R5; pass a different seed)."""
    import random

    rng = random.Random(seed)
    if correct is None:
        correct = [0 if rng.random() < p_err else 1 for _ in range(n)]
    conf = [min(1.0, max(0.0, rng.random() * (1.0 - strength) + strength * y)) for y in correct]
    return conf, correct


def _t077_mutated_scorer(kind):
    """Deliberately broken scorers, each reddening one clause of R2 / R5 -- RUN, not reasoned (P8: a reachable red).
    `drift`: a non-deterministic scorer (+1e-9 x calls^2 per call) -- a relay's diffs are no longer exactly 0.0,
    so the degenerate NOT-MET branch cannot fire. `blind`: the confidence stream is discarded -- every read is the
    fixed prior, so no control can read MET."""
    calls = [0]

    def drift(conf, correct, **kw):
        calls[0] += 1
        r = track_error(conf, correct, **kw)
        if r["auroc2"] is not None:
            r["auroc2"] += 1e-9 * calls[0] * calls[0]
        return r

    def blind(conf, correct, **kw):
        return track_error([0.5] * len(conf), correct, **kw)

    return {"drift": drift, "blind": blind}[kind]


def t077_relay_contrasts(scorer=track_error, seeds=T077_SEEDS, n=120, n_perm=200):
    """R2 on `contrast` (2.4; rebuilt 2026-09-15 after the panel -- until then R2 scored one stream twice and
    asserted the two reads equal, which any deterministic scorer passes). RELAY arm: the second-order read IS the
    first-order input, so (b3o)'s per-seed diffs are exactly 0.0 -- degenerate, NOT MET. CONTROL: a second-order
    read that is a stronger tracker than the first-order input on the same outcomes reads MET. Returns
    (relay, control)."""
    from snn_data_lake.substrate.paired_contrast import contrast

    relay, control = [], []
    for s in seeds:
        conf1, correct = t077_tracker_stream(n, 0.1, 0.15, seed=s)
        conf2, _ = t077_tracker_stream(n, 0.1, 0.6, seed=s + 100, correct=correct)
        first = scorer(conf1, correct, n_perm=n_perm, seed=s)["auroc2"]
        relay.append(scorer(list(conf1), correct, n_perm=n_perm, seed=s)["auroc2"] - first)
        control.append(scorer(conf2, correct, n_perm=n_perm, seed=s)["auroc2"] - first)
    return contrast("R2 relay: intact - passthrough", relay), contrast("R2 control", control)


def t077_lesion_contrasts(scorer=track_error, seeds=T077_SEEDS, n=120, n_perm=200):
    """R5 on `contrast` (2.4; rebuilt 2026-09-15 after the panel, same defect as R2): (b3) is intact - lesioned
    auroc2 over the seeds. NO-COLLAPSE twin: the lesioned read EQUALS the intact read -- every diff exactly 0.0,
    degenerate, NOT MET (the in-vivo half of the dissociation can fail, and fails exactly). COLLAPSING twin: the
    lesioned read is noise on the same outcomes -- MET. Returns (no_collapse, collapsing)."""
    from snn_data_lake.substrate.paired_contrast import contrast

    same, collapsed = [], []
    for s in seeds:
        conf, correct = t077_tracker_stream(n, 0.1, 0.6, seed=s)
        noise, _ = t077_tracker_stream(n, 0.1, 0.0, seed=s + 100, correct=correct)
        intact = scorer(conf, correct, n_perm=n_perm, seed=s)["auroc2"]
        same.append(intact - scorer(list(conf), list(correct), n_perm=n_perm, seed=s)["auroc2"])
        collapsed.append(intact - scorer(noise, correct, n_perm=n_perm, seed=s)["auroc2"])
    return contrast("R5 no collapse", same), contrast("R5 collapsing twin", collapsed)


def t077_reds():
    """2.4's six reds as ONE self-check line (each is also a test in tests/test_metacognition_instrument.py; this
    is the in-command control 2.4 required and the module did not carry until 2026-09-15). Each value is True iff
    the red holds on its constructed input and -- for R2 / R5 -- the mutated scorers REDDEN it."""
    import random

    out = {}
    rng = random.Random(0)
    stream = [{k: rng.uniform(0.0, 0.9) for k in T077_INPUTS} for _ in range(120)]
    rec = replay_reliability(stream, seed=5)
    bad, big = list(rec), [dict(row) for row in stream]
    bad[60] += 1e-9
    big[60]["surprise"] = 1.0
    caught = 0
    for args in ((stream, bad), (big, rec)):
        try:
            replay_reliability(args[0], seed=5, expect=args[1])
        except AssertionError:
            caught += 1
    out["R0"] = caught == 2 and replay_reliability(stream, seed=5, expect=rec) == rec
    conf, correct = t077_tracker_stream(400, 0.0, 0.9, seed=1)
    for i in (7, 311):
        correct[i] = 0
    two = track_error(conf, correct, n_perm=200, seed=0)
    for i in (100, 200, 250, 333):
        correct[i] = 0
    six = track_error(conf, correct, n_perm=200, seed=0)
    out["R1"] = two["readable"] is False and two["auroc2"] is None and six["readable"] is True
    relay, ctrl = t077_relay_contrasts()
    d_relay, _ = t077_relay_contrasts(_t077_mutated_scorer("drift"))
    _, b_ctrl = t077_relay_contrasts(_t077_mutated_scorer("blind"))
    out["R2"] = (
        relay.degenerate
        and not relay.met
        and ctrl.met
        and not (d_relay.degenerate and not d_relay.met)
        and not b_ctrl.met
    )
    conf, correct = t077_tracker_stream(400, 0.1, 0.15, seed=3)
    r = track_error(conf, correct, n_perm=200, seed=0)
    out["R3"] = bool(r["readable"] and r["shuffled_auroc2"] <= r["perm_q95"] < r["auroc2"] < 1.0)
    e = _experiment(0)
    tol = (e["commit_acc"] * (1.0 - e["commit_acc"]) / max(1, e["committed"])) ** 0.5
    out["R4"] = abs(e["commit_acc"] - e["always_acc"]) > tol
    same, coll = t077_lesion_contrasts()
    d_same, _ = t077_lesion_contrasts(_t077_mutated_scorer("drift"))
    _, b_coll = t077_lesion_contrasts(_t077_mutated_scorer("blind"))
    out["R5"] = (
        same.degenerate
        and not same.met
        and coll.met
        and not (d_same.degenerate and not d_same.met)
        and not b_coll.met
    )
    return out


def t077_record_brain(seed, n_ticks=T077_N_TICKS, lesion=False):
    """Tick a scratch brain n_ticks times; return (brain, per-tick rows). See the block comment above."""
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=seed, persist=False)
    mon = b._metacog
    captured, reads = [], []
    real_update = mon.update
    last_first_order = [0.0]

    def wrapped(**kw):
        captured.append({k: float(kw.get(k, 0.0)) for k in T077_INPUTS})
        last_first_order[0] = float(kw.get("confidence", 0.0))
        r = real_update(**kw)
        reads.append(
            float(r)
        )  # the UNROUNDED pool read; brain.py:2513 rounds `_metacog_rel` to 4 places
        return r

    mon.update = wrapped
    if lesion:  # 2.2 LESIONED, in vivo: gates off AND the second-order READ removed (first-order passthrough)
        mon.lesion = True
        mon.reliability = lambda: last_first_order[0]
    rows = []
    prev_n = prev_ok = prev_persist = 0
    prev_seen = prev_hits = 0
    for t in range(n_ticks):
        n_before = len(captured)
        kind = b.tick()
        ng = b.neuro
        outcome = persist = None
        if ng is not None and ng.live_n > prev_n:  # exactly one prequential score this tick
            outcome = int(ng.live_correct - prev_ok)
            persist = int(ng.live_persist - prev_persist)
            prev_n, prev_ok, prev_persist = ng.live_n, ng.live_correct, ng.live_persist
        perc = None  # the perception stream's per-frame hit (reported beside, never barred)
        seen, hits = int(getattr(b, "_perc_seen", 0) or 0), int(getattr(b, "_perc_hits", 0) or 0)
        if seen > prev_seen:
            perc = int(hits - prev_hits)
            prev_seen, prev_hits = seen, hits
        fed = len(captured) > n_before
        rows.append(
            {
                "t": t,
                "kind": kind,
                "fed": fed,
                "inputs": captured[-1] if fed else None,
                "rel": reads[-1] if fed else None,  # the pool's own read on a fed tick
                "read": float(mon.reliability()) if fed else None,  # what consumers see
                "hold": bool(b._metacog_hold),
                "learn_gate": float(mon.learn_gate()),
                "outcome": outcome,
                "persist": persist,
                "perc": perc,
            }
        )
    return b, rows


def t077_pairs_prospective(rows, key, gap=1):
    """(the LAST FED value strictly before the scoring tick, the outcome scored on that tick); `gap` > 1 reaches
    further back by k FED READS (the +k-fed-read lag control): the monitor is fed only on awake ticks, so k fed
    reads span >= k ticks -- at k = 5 on the campaign, 5 ticks at the median and up to 19 (mean 10.20 over
    2,206 pairs, seeds 0-7; re-recorded 2026-09-15 after the panel). Outcomes with no fed read before them are
    skipped and counted."""
    conf, correct, skipped = [], [], 0
    fed_idx = [i for i, r in enumerate(rows) if r["inputs"] is not None]
    for t, r in enumerate(rows):
        if r["outcome"] is None:
            continue
        before = [i for i in fed_idx if i < t]
        if len(before) < gap:
            skipped += 1
            continue
        i = before[-gap]
        v = rows[i]["inputs"]["confidence"] if key == "first_order" else rows[i][key]
        conf.append(float(v))
        correct.append(int(r["outcome"]))
    return conf, correct, skipped


def t077_pairs_perception(rows, key="read"):
    """The perception stream: the frame's hit on tick t with the read on the same tick (the frame is scored before
    the monitor's update on t, brain.py:1653 vs :2468). Reported beside, never barred."""
    conf, correct = [], []
    for r in rows:
        if r["perc"] is None or r[key] is None:
            continue
        conf.append(float(r[key]))
        correct.append(int(r["perc"]))
    return conf, correct


def t077_arm_d(seed):
    """Arm D (2.2): SecondOrderAccess on the chat path -- intact counts, the pin-to-mean neuter, the flip neuter."""
    from snn_data_lake.cortex.brain import make_brain

    def run(pin):
        b = make_brain(seed=seed, persist=False)
        so = b._second_order
        if pin is not None and so is not None:
            real = so.observe

            def pinned(*a, **k):
                real(*a, **k)
                so._conf2 = pin
                return pin

            so.observe = pinned
        out = []
        for i in range(T077_CHAT_TURNS):
            for _ in range(T077_CHAT_GAP):
                b.tick()
            r = b.respond(T077_CHAT_QUESTIONS[i % len(T077_CHAT_QUESTIONS)])
            out.append(
                (
                    r["route"],
                    bool(b._second_order_hold),
                    bool(b._metacog_hold),
                    so._conf2 if so else None,
                )
            )
        return out

    intact = run(None)
    c2 = [r[3] for r in intact if r[3] is not None]
    mean_c2 = statistics.mean(c2) if c2 else 0.0
    neut = run(mean_c2)
    flip = run(1.0)
    return {
        "turns": T077_CHAT_TURNS,
        "so_hold_true": sum(1 for r in intact if r[1]),
        "so_hold_true_monitor_false": sum(1 for r in intact if r[1] and not r[2]),
        "conf2_mean": mean_c2,
        "conf2_max": max(c2) if c2 else None,
        "pin_to_mean_routes_changed": sum(1 for a, b2 in zip(intact, neut) if a[0] != b2[0]),
        "pin_to_mean_holds_changed": sum(1 for a, b2 in zip(intact, neut) if a[1] != b2[1]),
        "flip_routes_changed": sum(1 for a, b2 in zip(intact, flip) if a[0] != b2[0]),
        "routes": sorted({r[0] for r in intact}),
    }


def run_t077_campaign(seeds=T077_SEEDS, n_ticks=T077_N_TICKS, pin=T077_PIN):
    """The T077 campaign (2.2-2.3): records, scores, contrasts, pins ONCE. Returns an exit code."""
    from snn_data_lake.substrate.paired_contrast import (
        contrast,
    )  # the ONE paired t (lifted, T077 2.10)

    if pin.exists():
        print(
            f"REFUSED: {pin} exists -- the pin is written once (delete it by hand only with the record)",
            flush=True,
        )
        return 2
    t0 = time.time()
    per_seed = {}
    for s in seeds:
        _b, rows = t077_record_brain(s, n_ticks)
        conf, correct, skipped = t077_pairs_prospective(rows, "read", 1)
        po, _, _ = t077_pairs_prospective(rows, "first_order", 1)
        stream = [r["inputs"] for r in rows if r["inputs"] is not None]
        rel = [r["rel"] for r in rows if r["inputs"] is not None]
        replay_reliability(
            stream, seed=s, base=0.4, expect=rel
        )  # (c) R0 on the live recording, exact
        intact = track_error(conf, correct, n_perm=T077_N_PERM, seed=s)
        passthrough = track_error(po, correct, n_perm=T077_N_PERM, seed=s)
        lag5c, lag5y, _ = t077_pairs_prospective(rows, "read", 5)  # 5 FED reads back, not 5 ticks
        lag5 = track_error(lag5c, lag5y, n_perm=T077_N_PERM, seed=s)
        pc, py = t077_pairs_perception(rows, "read")
        perception = track_error(pc, py, n_perm=T077_N_PERM, seed=s)
        neuters = {}
        for name in T077_INPUTS:
            mean_v = statistics.mean(x[name] for x in stream)
            rep = replay_reliability(stream, seed=s, base=0.4, neuter=(name, mean_v))
            rep_rows, k = [], 0
            for r in rows:
                rr = dict(r)
                if r["inputs"] is not None:
                    rr["read"] = rep[k]
                    k += 1
                rep_rows.append(rr)
            c2, y2, _ = t077_pairs_prospective(rep_rows, "read", 1)
            nt = track_error(c2, y2, n_perm=T077_N_PERM, seed=s)
            neuters[name] = (
                None if nt["auroc2"] is None else nt["auroc2"] - (intact["auroc2"] or 0.0)
            )
        _lb, lrows = t077_record_brain(s, n_ticks, lesion=True)
        lc, ly, _ = t077_pairs_prospective(lrows, "read", 1)
        lesioned = track_error(lc, ly, n_perm=T077_N_PERM, seed=s)
        acc_i = statistics.mean(r["outcome"] for r in rows if r["outcome"] is not None)
        acc_l = statistics.mean(r["outcome"] for r in lrows if r["outcome"] is not None)
        pers_i = statistics.mean(r["persist"] for r in rows if r["persist"] is not None)
        armd = t077_arm_d(s)
        per_seed[s] = {
            "n_pairs": len(conf),
            "skipped_outcomes": skipped,
            "intact": intact,
            "lesioned": lesioned,
            "passthrough": passthrough,
            "lag5": lag5,
            "perception": perception,
            "neuters": neuters,
            "acc_intact": acc_i,
            "acc_lesioned": acc_l,
            "persistence_intact": pers_i,
            "n_scored": sum(1 for r in rows if r["outcome"] is not None),
            "n_scored_lesioned": sum(1 for r in lrows if r["outcome"] is not None),
            "arm_d": armd,
            "streams": {
                "read": conf,
                "first_order": po,
                "outcome": correct,
                "lesioned_read": lc,
                "lesioned_outcome": ly,
            },
        }
        print(
            f"seed {s}: pairs {len(conf)} (skipped {skipped}) readable={intact['readable']} auroc2={intact['auroc2']} "
            f"lesioned={lesioned['auroc2']} passthrough={passthrough['auroc2']} q95={intact.get('perm_q95')} "
            f"oracle={intact.get('oracle')} acc {acc_i:.4f}/{acc_l:.4f} persist {pers_i:.4f} "
            f"armD hold {armd['so_hold_true']}/{armd['turns']} pin-mean {armd['pin_to_mean_routes_changed']} "
            f"flip {armd['flip_routes_changed']}  [{time.time() - t0:.0f} s]",
            flush=True,
        )
    ss = [per_seed[s] for s in seeds]
    readable = all(x["intact"]["readable"] and x["lesioned"]["readable"] for x in ss)
    verdict = {"a_readable": readable, "per_seed_q95": [x["intact"]["perm_q95"] for x in ss]}
    if readable:
        au = [x["intact"]["auroc2"] for x in ss]
        verdict["b1"] = contrast("b1 tracking", [a - 0.5 for a in au])._asdict()
        # (b2)'s null has an ANALYTIC MEAN -- E[auroc2] under the label permutation is exactly 0.5 (average
        # ranks sum to n(n+1)/2) -- but its MEDIAN is not fixed by it: on the T077 pin's streams (100-167
        # distinct confidences) the median sits within 0.0024 of 0.5 and (b2) tracks (b1); on a discrete,
        # skewed stream the median departs and (b2) can read MET where (b1) does not (Codex's counterexample,
        # reproduced: evidence 3.5 item 30). Kept as pre-registered (2.3); redundant on THIS pin only.
        verdict["b2"] = contrast(
            "b2 not a marker", [x["intact"]["auroc2"] - x["intact"]["shuffled_auroc2"] for x in ss]
        )._asdict()
        verdict["b3"] = contrast(
            "b3 calibration collapses under the lesion",
            [x["intact"]["auroc2"] - x["lesioned"]["auroc2"] for x in ss],
        )._asdict()
        verdict["b3_passthrough_reported"] = contrast(
            "b3o intact - passthrough (reported)",
            [x["intact"]["auroc2"] - x["passthrough"]["auroc2"] for x in ss],
        )._asdict()
        d = [x["acc_intact"] - x["acc_lesioned"] for x in ss]
        delta = statistics.pstdev([x["acc_intact"] for x in ss]) if len(ss) > 1 else 0.0
        tost_lo = contrast("b4 TOST lower (d + Delta > 0)", [v + delta for v in d])
        tost_hi = contrast("b4 TOST upper (Delta - d > 0)", [delta - v for v in d])
        verdict["b4"] = {
            "mean_diff": statistics.mean(d),
            "delta": delta,
            "lower": tost_lo._asdict(),
            "upper": tost_hi._asdict(),
            "met": tost_lo.met and tost_hi.met,
        }
        verdict["b5"] = {
            "oracle_all_one": all(x["intact"]["oracle"] == 1.0 for x in ss),
            "oracle_above_q95_all": all(
                x["intact"]["oracle"] > x["intact"]["perm_q95"] for x in ss
            ),
        }
        verdict["b5"]["met"] = (
            verdict["b5"]["oracle_all_one"] and verdict["b5"]["oracle_above_q95_all"]
        )
        verdict["b6"] = contrast(
            "b6 competence over persistence",
            [x["acc_intact"] - x["persistence_intact"] for x in ss],
        )._asdict()
        verdict["pass"] = all(verdict[k]["met"] for k in ("b1", "b2", "b3", "b4", "b5", "b6"))
    out = {
        "row": "T077",
        "n_ticks": n_ticks,
        "seeds": list(seeds),
        "bar": T_BAR,
        "rho": RHO,
        "per_seed": per_seed,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    pin.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"VERDICT {json.dumps(verdict, default=str)}", flush=True)
    print(f"pinned {pin} in {time.time() - t0:.0f} s", flush=True)
    return 0


def _p(lines, s):
    """Append an ASCII-only line (Windows cp1252 safety)."""
    if any(ord(ch) > 127 for ch in s):
        raise AssertionError("non-ASCII output")
    lines.append(s)


def main():
    if (
        "--t077-campaign" in sys.argv[1:]
    ):  # T077 2.9 step 4: the campaign, under a throwaway home, pinned once
        from snn_data_lake.substrate.home import throwaway_home

        with throwaway_home():
            sys.exit(run_t077_campaign())
    r = _experiment(0)
    lines = []
    _p(lines, "=" * 78)
    _p(lines, "METACOGNITIVE MONITOR -- lesionable spiking reliability gate (L10 / SI-10)")
    _p(lines, "=" * 78)
    _p(lines, "metacognition CONSUMED: a LIF reliability pool gates LEARN / EXPLORE / GROW / HOLD")
    _p(lines, "-" * 78)
    _p(
        lines,
        f"stream: {K} learnable classes x {ROUNDS} rounds + {NOISE_PER_ROUND} noise/round "
        f"(dim={DIM}, vigilance={VIG})",
    )
    _p(lines, "-" * 78)
    _p(lines, "                       held-out acc   units    abstains")
    _p(
        lines,
        f"  GATED (monitor on) :   {r['g_acc']:.3f}       {r['g_units']:>3}      {r['g_abstains']:>3}",
    )
    _p(
        lines,
        f"  LESIONED (control) :   {r['l_acc']:.3f}       {r['l_units']:>3}      {r['l_abstains']:>3}",
    )
    _p(lines, "-" * 78)
    _p(
        lines,
        f"(1) gating helps    : gated acc {r['g_acc']:.3f} >= lesioned {r['l_acc']:.3f} + margin, "
        f"and gated units {r['g_units']} < lesioned {r['l_units']}",
    )
    _p(
        lines,
        f"(2) confidence-hold : committed acc {r['commit_acc']:.3f} > always-commit {r['always_acc']:.3f}; "
        f"abstained on {r['ood_abstained']} OOD queries ({r['committed']} committed)",
    )
    _p(
        lines,
        f"(3) spiking read    : reliable rate {r['rel_reliable']:.3f} vs noise {r['rel_noise']:.3f}; "
        f"spike-count integer={r['spike_count_is_integer']} membrane-integer={r['mem_is_integer_spikecount']}",
    )
    _p(
        lines,
        f"    sub-thr trap    : spikes={r['trap_spikes']} spike_rel={r['trap_spike_rel']:.3f} "
        f"but membrane_rel={r['trap_membrane_rel']:.3f} (membrane falsely 'reliable')",
    )
    _p(lines, f"    lesion==ungated : byte-identical downstream={r['lesion_byte_identical']}")
    _p(lines, f"(4) determinism     : two seed=0 runs identical={_experiment(0) == r}")
    _p(lines, "-" * 78)
    q = _p6_experiment(0)
    _p(
        lines,
        f"(5) P6 executive->metacognition : producer cortex/executive_control.conflict over "
        f"{q['n']} ticks, span {q['conflict_span']:.3f}, {q['conflict_distinct']} distinct",
    )
    _p(
        lines,
        f"    CHARGE (live vs producer PINNED to {q['pinned']:.4f}) : reliability differs "
        f"{q['charge'][0]}/{q['n']} (max |d| {q['charge'][4]:.4f}), hold flips {q['charge'][1]}, "
        f"explore flips {q['charge'][2]}, learn_gate differs {q['charge'][3]}",
    )
    _p(
        lines,
        f"    INERT CONTROLS  : pre-P6 call shape vs PINNED conflict differs "
        f"{q['constant_is_inert'][0]}; w_conflict=0 vs PINNED differs {q['gain0_is_inert'][0]}  "
        f"(both must be 0)",
    )
    _p(
        lines,
        f"    SIGN            : unusually conflicted tick rel {q['rel_unusual_conflict']:.4f} "
        f"< baseline-conflict tick {q['rel_baseline_conflict']:.4f} "
        f"(excess {q['excess_unusual']:+.4f} vs {q['excess_baseline']:+.4f})",
    )
    _p(lines, "=" * 78)
    for line in lines:
        print(line)

    # ---- (1) GATING HELPS LEARNING (headline) ----
    assert r["g_acc"] >= r["l_acc"] + 0.05, (
        f"gated held-out {r['g_acc']:.3f} must beat lesioned {r['l_acc']:.3f} by a margin"
    )
    assert r["g_units"] < r["l_units"], (
        f"gated must recruit FEWER units ({r['g_units']}) than lesioned ({r['l_units']}) -- no noise memorizing"
    )
    assert r["g_abstains"] > 0, "the gated monitor must ABSTAIN on the unlearnable noise"

    # ---- (2) CONFIDENCE-ABSTAIN beats always-commit ----
    assert r["commit_acc"] > r["always_acc"], (
        f"committed-only accuracy {r['commit_acc']:.3f} must beat always-commit {r['always_acc']:.3f}"
    )
    assert r["ood_abstained"] > 0, "the monitor must HOLD on some OOD / unreliable queries"

    # ---- (3) SPIKING + LESIONABLE ----
    assert r["rel_reliable"] > r["rel_noise"], (
        "the spike-count reliability must separate reliable from noise"
    )
    assert r["spike_count_is_integer"], (
        "reliability * pop_size * steps must be an exact SPIKE COUNT (integer)"
    )
    assert not r["mem_is_integer_spikecount"], (
        "the membrane read is NOT a spike-count fraction -- structural proof reliability is spike-derived"
    )
    assert r["trap_spikes"] == 0 and r["trap_spike_rel"] == 0.0, (
        "a sub-threshold pool emits NO spikes -> spike reliability is correctly 0 (abstain)"
    )
    assert r["trap_membrane_rel"] > 0.5, (
        "the membrane read falsely calls the silent sub-threshold pool 'reliable' -- why we read spikes"
    )
    assert r["lesion_byte_identical"], "lesion=True must be byte-identical downstream to no monitor"

    # ---- (4) DETERMINISM ----
    assert _experiment(0) == r, "two seed=0 runs must be byte-for-byte identical"

    # ---- (5) P6: THE EXECUTIVE->METACOGNITION EDGE ----
    # Every assert below owes a reachable RED (P8), and each red was CONSTRUCTED AND RUN against a
    # mutated copy of this file rather than reasoned about. The mutation that reddens each:
    #   vacuity   : make both branches of _p6_conflict_stream clear winners -> span collapses
    #   inertness : use the raw conflict instead of the excess in the drive term
    #   gain0     : ignore the w_conflict argument (always use w_surprise)
    #   charge    : default w_conflict to 0.0 -- the channel with the wire cut
    #   sign      : flip the drive term to `+ self.w_conflict * self._conflict_excess`
    #   baseline  : seed the baseline at 0.0 instead of at the first sample
    assert q["conflict_span"] > 0.5 and q["conflict_distinct"] > q["n"] // 2, (
        f"VACUITY GUARD: the producer must actually vary, else every arm below is the same "
        f"experiment (span {q['conflict_span']:.3f}, {q['conflict_distinct']} distinct)"
    )
    assert q["constant_is_inert"] == (0, 0, 0, 0, 0.0), (
        f"the pre-P6 call shape and a PINNED (constant) conflict must be byte-identical -- the term "
        f"is relative to its own running baseline, not absolute {q['constant_is_inert']}"
    )
    assert q["gain0_is_inert"] == (0, 0, 0, 0, 0.0), (
        f"w_conflict=0 must reproduce the neutered monitor exactly {q['gain0_is_inert']}"
    )
    assert q["charge"][0] > q["n"] // 4 and q["charge"][3] > q["n"] // 4, (
        f"P6 CHARGE: neutering cortex/executive_control must change THIS monitor's own output -- "
        f"reliability {q['charge'][0]}/{q['n']}, learn_gate {q['charge'][3]}/{q['n']}"
    )
    assert q["charge"][1] + q["charge"][2] > 0, (
        f"the charge must reach a GATE, not only the float: hold flips {q['charge'][1]}, explore "
        f"flips {q['charge'][2]} (a reliability that moves but decides nothing is P3 shape 5)"
    )
    assert q["rel_unusual_conflict"] < q["rel_baseline_conflict"], (
        f"unusual response conflict must LOWER reliability ({q['rel_unusual_conflict']:.4f}) below "
        f"the baseline-conflict tick ({q['rel_baseline_conflict']:.4f}), not raise it"
    )
    # A sixth criterion -- `abs(excess_baseline) < 1e-9`, i.e. a tick AT the running baseline carries
    # zero excess -- was DRAFTED AND DELETED. Its red was run (seed `_conf_base` at 0.0 instead of at
    # the first sample) and the mutation reddens `constant_is_inert` FIRST, every time: under a
    # baseline-relative term a mis-seeded baseline necessarily breaks constant-inertness, so the
    # criterion has no INDEPENDENT red and is a weaker restatement of one already asserted above.
    # P8: delete, never weaken. `excess_baseline` is still PRINTED, because a number a reader can
    # see is worth keeping even when it is not a criterion.

    # ---- (6) T077: THE SIX REDS OF 2.4, one line (tests/test_metacognition_instrument.py carries them as
    #      tests; this is the in-command control 2.4 required). Added 2026-09-15 after the panel, which found
    #      R2 / R5 to be f(x)==f(x) checks -- they now read `contrast` and are reddened IN FACT by two mutated
    #      scorers (`_t077_mutated_scorer`: a drifting one and a confidence-blind one), not by argument.
    reds = t077_reds()
    assert all(reds.values()), f"T077 reds must fire on their constructed inputs: {reds}"

    print(
        f"self-check OK: GATING lifts held-out {r['l_acc']:.3f}->{r['g_acc']:.3f} while cutting units "
        f"{r['l_units']}->{r['g_units']} (abstained on {r['g_abstains']} noise);"
    )
    print(
        f"  confidence-HOLD raises accuracy {r['always_acc']:.3f}->{r['commit_acc']:.3f} by abstaining on "
        f"{r['ood_abstained']} OOD; reliability is SPIKE-count ({r['rel_reliable']:.2f} vs noise "
        f"{r['rel_noise']:.2f}),"
    )
    print(
        "  membrane-read control FAILS the structural spike-count assert and mis-reads a silent pool; "
        "lesion==ungated; deterministic;"
    )
    print(
        f"  P6 this monitor now CONSUMES cortex/executive_control.conflict -- pinning that producer "
        f"changes this monitor's OWN output on {q['charge'][0]}/{q['n']} ticks "
        f"({q['charge'][1]} hold flips, {q['charge'][2]} explore flips), while a CONSTANT conflict "
        f"stream and w_conflict=0 are both byte-identical to no wire at all"
    )
    print(
        f"  T077 reds R0-R5 fire on their constructed inputs, R2 / R5 reddened by mutated scorers: "
        f"{sum(reds.values())}/{len(reds)}"
    )


if __name__ == "__main__":
    main()
