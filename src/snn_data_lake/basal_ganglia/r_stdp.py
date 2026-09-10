"""
Reward-modulated STDP (R-STDP) with a decaying eligibility trace -- the distal-reward rule.

THE three-factor rule for DISTAL reward in this brain -- one mechanism, eight importers, not an
extra beside another one. (Framing corrected 2026-08-04b: this opened "a new member of the
plasticity family (additive, opt-in; changes no existing behaviour)", which is the disposition P2
forbids in those exact words -- and it was also no longer true. `docs/POLICIES.md` names this module
as the REUSE ANCHOR for three-factor plasticity, and it is imported by `brain_native_selfbuild`,
`cortex/interaction_learning`, `cortex/task_set`, `neuromodulation/naca` and four others. What makes
that P2-clean is NOT that it is opt-in; it is the DISTINCTNESS argument three paragraphs below --
this rule and `valence_stdp` do different jobs on two independent axes, so neither supersedes the
other and there is nothing to retire.) This is the classic Izhikevich (2007) solution to the
DISTAL-REWARD problem: spike-timing
coincidences leave a slowly-decaying ELIGIBILITY TRACE at each synapse, so a reward arriving
many timesteps AFTER the causal pre-before-post pairing can still reinforce the CORRECT
synapse -- even though many other spikes happened in between. Mechanism (BrainCog learning
rule #5, R-STDP):

    STDP(dt) is added to the eligibility e, NOT to w:
        dt = t_post - t_pre;  LTP if dt > 0: e += A_plus  * exp(-dt / tau_plus)
                              LTD if dt < 0: e += -A_minus * exp( dt / tau_minus)
    e decays every step:      e *= exp(-dt_step / tau_e)   (tau_e is LONG: the reward bridge)
    delayed reward d gates:   w += lr * d * e              (per synapse; sign of d flips it)

HOW THIS DIFFERS from valence_stdp.py (do not confuse the two): ValenceLearner is a reward-
PREDICTION-ERROR rule over FEATURE PRESENCE with IMMEDIATE reward (w += lr*(reward-predicted)
*fired_now) -- it is NOT spike-timing-dependent and bridges NO temporal delay. R-STDP here
(a) keys on pre/post spike TIMING (the STDP window), and (b) bridges a DELAY between the
pairing and a LATER raw reward d via the decaying eligibility trace. Different on both axes.

Honesty: this improves temporal CREDIT ASSIGNMENT (distal reward) via a synaptic eligibility
trace. It is not cognition/AGI.
Single-layer, few-synapse rate-timing caricature on synthetic spike trains -- no
deep multi-step credit assignment, no hidden layers.
Refs: Zeng et al., BrainCog, Patterns 2023 (rule #5, R-STDP); Izhikevich, "Solving the distal
reward problem through STDP and dopamine", Cereb. Cortex 2007.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.basal_ganglia.r_stdp
"""

import math
import os
import random
import threading
import time
from typing import NamedTuple

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
#:
GRADIENT_CLAIM = {
    "calibrate": (
        "NOT_A_RULE",
        "TIMES this host -- the fork/join floor and the serial and parallel per-synapse rates on a "
        "throwaway population. A benchmark, not a learning rule. It was left UNDECLARED until "
        "NOT_A_RULE existed, because none of the original four dispositions described it and any "
        "choice would have been false",
    ),
    "RSTDPNeuron.reward": (
        "LOCAL",
        "a delayed dopamine signal gates the STORED eligibility into the weights: w += lr * d * e "
        "per synapse. Three-factor R-STDP -- the eligibility is formed locally from pre-before-post "
        "timing and nothing is propagated backward, so it owes no cosine",
    ),
}

LR = 0.1
A_PLUS = 1.0
A_MINUS = 1.0
TAU_PLUS = 20.0
TAU_MINUS = 20.0
TAU_E = 1000.0  # long eligibility time constant -- the distal-reward bridge


# ==========================================================================================
# THE RULE, ONCE (specs/022 T063, P2 MERGE 2026-09-09).
#
# Until this landed the three-factor rule existed only as list arithmetic INSIDE
# `RSTDPNeuron`'s four methods. T063 needs the same rule over a kernel bundle table
# (`substrate/kernel_plasticity.py`), and P2 forbids a second implementation: "a merge that
# keeps both paths is additive with extra steps". So the rule's operations are lifted HERE,
# to module scope, over any indexable container of eligibilities -- a Python list for the
# per-object neuron, a numpy array for the kernel's flat edge table -- and BOTH substrates
# call them. `tests/test_r_stdp_kernel_equivalence.py` is the behavioural verification P2
# demands: one spike/reward sequence through both forms, same weights to within the kernel's
# quantization step, plus a spy proving the per-object form GOES THROUGH these functions
# rather than merely agreeing with them, plus the incumbent's trajectory pinned byte-exact.
#
# APPENDED 2026-09-09, the paragraph above kept VERBATIM because it is the record of why
# these functions exist and deleting it would leave them looking arbitrary: THE SECOND
# SUBSTRATE IS GONE. `substrate/kernel_plasticity.py` was DELETED on its own retention
# criterion -- T063's campaign read t(d_f) = -13.0154, the frozen twin ahead of the learning
# arm on every one of the 8 seeds, and 2.9's clause is that a learner which moves weights
# without an attributed benefit is a null. The MERGE STANDS and is not reverted: P2 asks that
# exactly ONE mechanism survive, and one does. `RSTDPNeuron` still owns only the per-object
# bookkeeping and calls these functions for every arithmetic operation of the rule, which is
# what the surviving tests pin (the module-level form, the spy, and the incumbent trajectory).
# What is retired with the kernel form is the CONTAINER BRANCH those functions carried for it.
#
# WHY THE PAIR LIST RATHER THAN A VECTORIZED KERNEL. The formula is written exactly once in
# each function and applied per pair, so no container branch can make the two substrates
# diverge -- which is precisely the drift a "kernel form" of a rule usually introduces. The
# cost is a Python loop over the pairs a tick actually forms (spikes x incident edges), not
# over the whole table; only `decay_eligibility` touches every entry, and it is the one place
# a container branch is unavoidable, so the DECAY FACTOR is computed once above the branch and
# the branch decides only how to multiply.
#
# APPENDED 2026-09-09, the paragraph above kept VERBATIM: "unavoidable" was true of TWO
# substrates and stopped being true when the second was deleted. `decay_eligibility` and
# `reward_deltas` each carried a `hasattr(e, "dtype")` branch whose numpy arm existed for the
# kernel form alone; `accumulate_causal` and `accumulate_anticausal` never had one, because a
# per-pair `e[i] +=` is already container-agnostic. THE ORPHANED ARMS ARE DELETED, on a
# measurement rather than on a reading of the imports (P3; runtime evidence outranks static
# reading). Two instruments in ONE execution, each with its positive control taken at 963799e
# BEFORE the deletion -- a COUNTER of numpy-vs-list entries and a TRIPWIRE raising inside the
# numpy arm -- driven by r_stdp's bare self-check, the harness's bare self-check and the whole
# targeted pytest set:
#   at 963799e, kernel_plasticity's self-check alone: decay numpy 104, deltas numpy 28;
#     the harness's self-check alone: decay numpy 38,400, deltas numpy 1,920, list 0 for both
#   after the deletion, all three drivers together: decay list 297 numpy 0,
#     deltas list 4,476 numpy 0, tripwire entries 0
# So both functions are ONE CODE PATH again, and the per-index form each keeps is the one the
# per-object neuron always took -- which is why the incumbent's byte-exact trajectory pin at
# 851b026 is unmoved by this.
#
# distinct-from: cortex/neuron_zoo.py::_step_serial -- advances a MEMBRANE one timestep over a
#   list of neuron objects; no eligibility, no spike-time pairing, no reward. Read 2026-09-09.
# distinct-from: brainstem/arm_bridge.py::UdpArmBridge.recv_outcome -- reads a reward FLOAT off
#   a UDP socket; it transports the third factor, it never gates one. Read 2026-09-09.
# distinct-from: cortex/predictive_coding.py::_lif -- a vectorized LIF step over signed ON/OFF
#   pairs; "pairs" there are two neurons encoding one error component, not a (pre, post) spike
#   ordering. Read 2026-09-09.
# distinct-from: plasticity/competitive_learning.py::CompetitiveLearner.present -- drives a
#   pattern for `steps` ticks and returns spike counts; its learning is prototype-row movement
#   with no trace and no reward. Read 2026-09-09.
# distinct-from: reasoning/arc_agi/rule_inducer.py::step -- a symbolic world model returning
#   (state', reward, done); "reward" is an environment return, not a synaptic gate.
#   Read 2026-09-09.
# ==========================================================================================


def accumulate_causal(e, pairs, a_plus=A_PLUS, tau_plus=TAU_PLUS):
    """CAUSAL pairing (pre-before-post) -> LTP onto eligibility, for every (index, dt) in
    `pairs` with dt = t_post - t_pre. Only dt > 0 contributes: a pair at dt == 0 is not
    ordered and an anti-causal pair belongs to `accumulate_anticausal`. Mutates `e`."""
    for i, dt in pairs:
        if dt > 0:
            e[i] += a_plus * math.exp(-dt / tau_plus)
    return e


def accumulate_anticausal(e, pairs, a_minus=A_MINUS, tau_minus=TAU_MINUS):
    """ANTI-CAUSAL pairing (post-before-pre) -> LTD onto eligibility, for every (index, dt) in
    `pairs` with dt = t_post - t_pre. Only dt < 0 contributes. Mutates `e`."""
    for i, dt in pairs:
        if dt < 0:
            e[i] += -a_minus * math.exp(dt / tau_minus)
    return e


# distinct-from: cortex/lif_neuron.py::LIFNeuron.step and cortex/alif_neuron.py::ALIFNeuron.step --
#   advance a MEMBRANE one timestep under an input current and return a 0/1 spike, with a
#   refractory clamp. A membrane leak is not an eligibility decay: nothing here integrates an
#   input, nothing fires, and the quantity that decays is a per-synapse trace.
# distinct-from: cortex/neuron_zoo.py::NeuronZoo.make -- a factory returning objects that expose
#   step(current)->0/1; it builds the things above and owns no formula at all.
# distinct-from: cerebellum/spiking_jepa.py::_forward -- runs an event retina and a 2-layer
#   recurrent LIF backbone, recording per-timestep spikes and PRE-RESET MEMBRANES because its own
#   eligibility trace needs them. It is a forward pass that PRODUCES the inputs a trace is built
#   from; this is the one line that ages a trace already built.
# distinct-from: brainstem/arm_bridge.py::UdpArmBridge.recv_outcome -- reads the next reward float
#   off a UDP socket; it transports the third factor and never decays anything.
#   All five sites read 2026-09-09 before this note was written.
def decay_eligibility(e, dt_step=1.0, tau_e=TAU_E):
    """The distal-reward bridge decaying one timestep: e *= exp(-dt_step / tau_e), elementwise.

    The formula is the first line and is computed ONCE. It used to be applied through a
    `hasattr(e, "dtype")` branch -- one vectorized multiply for the kernel's flat edge table,
    because `list *= float` is a TypeError -- and that arm is DELETED with the kernel form it
    existed for (2026-09-09; the header carries the measurement). ONE code path, per index,
    mutating `e` in place."""
    decay = math.exp(-dt_step / tau_e)
    for i in range(len(e)):
        e[i] *= decay
    return e


def reward_deltas(e, d, lr=LR):
    """THE reward gate: the per-synapse weight delta lr * d * e for a (possibly long-delayed)
    reward d. Returns the deltas; it does NOT apply them, because the two substrates applied them
    differently and only the DELTA is the rule -- the per-object form adds them to a float
    weight list, the kernel form re-quantized them into an int64 bundle table under a
    sign-preserving clamp. Association order is (lr * d) * e[i], so the two forms were
    bit-comparable rather than approximately equal.

    THE KERNEL FORM IS GONE (2026-09-09) and so is the `hasattr(e, "dtype")` arm that returned
    the whole array in one multiply; the header carries the measurement. Returning the DELTA
    rather than applying it is KEPT even with one substrate: it is what let the equivalence be
    read at all, and re-fusing the gate into the update would delete that seam."""
    return [lr * d * e[i] for i in range(len(e))]


class RSTDPNeuron:
    """One output neuron fed by n_syn input synapses; each synapse has a weight w and an
    eligibility trace e. STDP (spike timing) writes to e; a delayed reward gates e into w.

    Every arithmetic operation of the rule is delegated to the module-level functions above
    (the P2 merge): this class owns the per-object BOOKKEEPING -- which synapse, which spike
    time -- and owns none of the rule."""

    def __init__(
        self,
        n_syn,
        lr=LR,
        a_plus=A_PLUS,
        a_minus=A_MINUS,
        tau_plus=TAU_PLUS,
        tau_minus=TAU_MINUS,
        tau_e=TAU_E,
        w0=0.5,
        seed=0,
    ):
        rng = random.Random(seed)
        self.n = n_syn
        self.w = [w0 + rng.uniform(-0.05, 0.05) for _ in range(n_syn)]
        self.e = [0.0] * n_syn
        self.t_pre = [None] * n_syn  # last pre-spike time per synapse
        self.t_post = None  # last post-spike time of the neuron
        self.lr, self.a_plus, self.a_minus = lr, a_plus, a_minus
        self.tau_plus, self.tau_minus, self.tau_e = tau_plus, tau_minus, tau_e

    # `step` below advances an ELIGIBILITY TRACE, never a membrane, which is why none of these
    # is the same job (all five read 2026-09-09 before this note was written):
    # distinct-from: cortex/lif_neuron.py::LIFNeuron.step and cortex/alif_neuron.py::ALIFNeuron.step
    #   -- integrate a membrane under an input current and return a 0/1 spike; no trace, no reward.
    # distinct-from: cortex/autapse_neuron.py::AutapseNeuron.step -- the same membrane step with a
    #   self-current added; it delegates to LIFNeuron.step.
    # distinct-from: plasticity/competitive_learning.py::CompetitiveLearner.step -- advances a
    #   competitive layer and applies prototype-row updates at post spikes; no eligibility trace
    #   and no third factor, so a delayed reward cannot reach it.
    # distinct-from: hippocampus/function_preserving_growth.py::step -- a forward pass over a
    #   hidden layer returning currents and spikes; it computes an output, not an update.
    def pre_spike(self, i, t):
        """Input synapse i fires at time t. Pair with the neuron's most recent POST spike:
        post-before-pre (dt < 0) is anti-causal -> LTD onto this synapse's eligibility."""
        self.t_pre[i] = t
        if self.t_post is not None:
            accumulate_anticausal(self.e, ((i, self.t_post - t),), self.a_minus, self.tau_minus)

    def post_spike(self, t):
        """The output neuron fires at time t. Pair with each synapse's most recent PRE spike:
        pre-before-post (dt > 0) is causal -> LTP onto that synapse's eligibility."""
        self.t_post = t
        accumulate_causal(
            self.e,
            ((i, t - self.t_pre[i]) for i in range(self.n) if self.t_pre[i] is not None),
            self.a_plus,
            self.tau_plus,
        )

    def step(self, dt_step=1.0):
        """Advance one timestep: the eligibility trace decays slowly (over tau_e). No reward
        yet -- this is the interval in which OTHER spikes may occur before the reward lands."""
        decay_eligibility(self.e, dt_step, self.tau_e)

    def reward(self, d):
        """A (possibly long-delayed) reward/dopamine signal d arrives: gate the stored
        eligibility into the weights. w += lr * d * e for every synapse; d>0 reinforces
        positive-eligibility (causally paired) synapses, d<0 flips the direction. Returns the
        per-synapse weight deltas."""
        deltas = reward_deltas(self.e, d, self.lr)
        for i in range(self.n):
            self.w[i] += deltas[i]
        return deltas


def declare_ports(manifest):
    """Manifest registrant (specs/022 T029, the consumer half): `dopamine_in` receives the
    midbrain's phasic DA rate as the THIRD FACTOR gating stored eligibility into weights
    (RSTDPNeuron.reward -- w += lr * d * e). The edge is this consumer's subscription; the
    route is the CBGTC dopamine broadcast the connectome declares. The port's meaningful
    delivery window is DERIVED, not picked: tau_e (the module's own distal-reward bridge,
    1000 steps) is how long a delayed burst still finds its eligibility."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "r_stdp",
        Port(
            name="dopamine_in",
            region="basal_ganglia",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="IN",
            declaring_module="r_stdp",
            join_policy="sum",
        ),
    )
    manifest.declare_edge("dopamine_nuclei", "da_out", "r_stdp", "dopamine_in")


#: the declared edge, exactly as the manifest carries it -- the charge evidence cites it
DA_EDGE = ("dopamine_nuclei", "da_out", "r_stdp", "dopamine_in")


def _da_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `dopamine_in` port, one per replay arm."""
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    port = m.ports[("r_stdp", "dopamine_in")]
    ring = SignalRing(slots=128)
    return ring, JoinConsumer(ring, port)


def _f4(x):
    """Round-trip through float32 -- the ring's framing dtype. The tonic reference MUST be
    quantized to the same grid the frames travel on: with a float64 reference, a frame
    carrying exactly-tonic came back as float32(tonic) and d was -2e-9 instead of 0, so a
    DEAD dopamine system 'moved' the weights and charged the port (review-constructed
    red). Sibling harness pins (0.0, 1.0, 0.2, 0.5) dodged this only by being exactly
    representable."""
    import struct

    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def consume_dopamine(consumer, n_synapses=4, seed=0):
    """The downstream computation under measurement: a fresh RSTDPNeuron forms one causal
    pairing (eligibility loaded), then each delivered DA frame gates that eligibility via
    THE reward mechanism, with the SIGNED third factor derived from the producer's own
    tonic rate (d = rate - tonic; tonic = f4-quantized phasic(0.0), the producer's stated
    rpe=0 point on the frame grid -- no constant picked here). Returns
    [(tick, total_weight), ...]. LOSS IS FATAL."""
    from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei

    tonic = _f4(DopamineNuclei(seed=seed).phasic(0.0))
    nrn = RSTDPNeuron(n_syn=n_synapses, seed=seed)
    for i in range(n_synapses):
        nrn.pre_spike(i, t=10.0)
    nrn.post_spike(t=15.0)  # causal pairing: LTP eligibility on every synapse
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        nrn.step()  # the eligibility bridge decays between deliveries (tau_e)
        nrn.reward(float(vec[0]) - tonic)
        out.append((tick, round(sum(nrn.w), 9)))
    return out


def _da_frames(n_ticks=60, acquire_until=40, seed=0):
    """Recorded frames from a REAL DopamineNuclei learning a cue->reward chain: early
    trials emit a big RPE at reward (unexpected), late trials a cue burst and ~tonic at
    reward (acquired) -- the classic TD transfer, one phasic rate per tick."""
    from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei

    nuc = DopamineNuclei(seed=seed)
    trial = [("ITI", 0.0), ("CUE", 0.0), ("REWARD", 1.0)]
    frames = []
    for t in range(1, n_ticks + 1):
        rpes = nuc.present(trial, learn=(t <= acquire_until))
        # the tick's broadcast: the reward-state RPE early, the cue RPE after transfer
        rpe = rpes["REWARD"] if t <= acquire_until else rpes["CUE"]
        frames.append((t, "dopamine_nuclei", [nuc.phasic(rpe)]))
    return frames, acquire_until


def _measure_da_charge():
    """The replay-twice A/B: live phasic DA stream vs a pinned f4-exact TONIC rate (the
    rpe=0 broadcast -- a dopamine system that never signals an error). -> (live,
    pinned_out, reads, responded): `responded` = the TD-extinction signature INSIDE the
    live arm (see the in-body comment; the criterion earned this shape by failing)."""
    from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei
    from snn_data_lake.substrate.port_charge_probe import replay_twice

    # WITHIN-LIVE-ARM structural predicate (review BLOCKER: the first form compared
    # against the pin's arithmetically-zero movement, collapsing to `moved > 0` -- a
    # constant 0.9, a reversed stream and even a dead-tonic stream all greened it): the
    # IMMEDIATE-GEOMETRIC-EXTINCTION signature of a real TD critic. Consecutive-trial
    # reward-RPE ratio is (1 - alpha) = 0.7, so the SECOND ten-trial window's mean weight
    # delta collapses to ~0.7^10 ~ 0.03x the first (measured 13-19x separation across 7
    # seeds); any constant stream holds ratio ~1.0 (eligibility decay alone,
    # exp(10/tau_e) ~ 1.01), and the reversed-stream attack -- a plateau first -- holds
    # ~1.01 too, so it REDs here where an early-vs-late form coincidentally greened it.
    # Margin 2.0 sits an order of magnitude from both populations. The pin is
    # f4-quantized so d is EXACTLY zero on the pinned arm and on any genuinely-tonic
    # live frame (the float32 leak was the second half of the review red: a dead DA
    # system 'moved' weights by 3.5e-8 and recorded charge).
    frames, acquire_until = _da_frames()
    tonic = _f4(DopamineNuclei(seed=0).phasic(0.0))
    live, pinned_out, reads = replay_twice(frames, _da_consumer_factory, consume_dopamine, [tonic])
    w = dict(live)
    deltas = [abs(w[t] - w[t - 1]) for t in range(2, acquire_until + 1)]
    first = sum(deltas[:10]) / 10.0
    second = sum(deltas[10:20]) / 10.0
    responded = first > 2.0 * second
    return live, pinned_out, reads, responded


def record_charge():
    """Measure the A/B and record the attribution via THE one writer. REFUSES a
    non-diverged or non-responding run -- evidence is a measurement, never a declaration."""
    from snn_data_lake.substrate.port_charge_probe import record_attribution

    live, pinned_out, reads, responded = _measure_da_charge()
    if live == pinned_out or not responded:
        print(f"record_charge REFUSED for {DA_EDGE}: no divergence or no response")
        return 1
    path = record_attribution(
        DA_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: a real DopamineNuclei acquiring a cue->reward chain (burst at "
        "unexpected reward, TD transfer to the cue) vs a pinned tonic rpe=0 rate; the "
        "three-factor gate moved the live arm's weights while the tonic pin moved "
        "~nothing -- the signed factor derives from the producer's own tonic point; "
        "self-check re-measures and requires `measured` to reproduce",
    )
    print(f"recorded {DA_EDGE} into {path}")
    return 0


# ======================================================================================
# POPULATION BROADCAST -- the parallel surface of the three-factor rule.
#
# A weight update here depends on its OWN eligibility trace and a BROADCAST SCALAR d, so
# updates are independent within a timestep. Under the free-threaded build (3.14,
# Py_GIL_DISABLED=1) that independence is finally expressible as parallelism. What follows
# is the measured account of WHERE it pays, because only one of the two obvious partitions
# actually does.
#
# ---- MEASURED, 2026-08-05, 28-core box, .venv-lake 3.14.6 free-threaded (sys._is_gil_enabled()
#      == False). All timings min-of-5 trials. ----
#
# PARTITION A -- across SYNAPSE INDEX inside one neuron. REFUTED, and it is structural.
#   `self.w` is a single Python list. Free-threaded CPython guards list mutation with a
#   per-object critical section, so every thread writing w[i] serializes on ONE mutex and then
#   pays cache-line contention on top. Measured `w[i] += k*e[i]` over a shared list, serial vs
#   threaded, at the fairest form (locals hoisted):
#         width       serial      T=2       T=4       T=8
#         10,000       514us     818us    1,848us   6,964us
#        100,000     5,247us   8,182us   20,075us  84,684us
#      1,000,000    53,347us  80,900us  221,122us 837,129us
#   Parallel LOSES at every width and every thread count, by 1.5x to 15.7x (and to 86x at
#   T=28). This is NOT a free-threading limit: the control that isolates it -- the identical
#   loop over PRIVATE per-thread lists -- scales 1.94x / 3.28x / 4.37x at T=2/4/8 on the same
#   1,000,000 elements. The blocker is the SHARED LIST OBJECT, nothing else. No gate can
#   rescue partition A, so none is offered.
#
# PARTITION B -- across whole NEURONS. This is what ships. Each thread owns entire neurons,
#   so no two threads ever touch the same weight list, and byte-identity is structural rather
#   than argued: every neuron's reward(d) executes the same code on the same object in the
#   same order it would serially, and nothing is reduced across threads. Measured, us/call:
#      synapses (shape)          serial     best-parallel   speedup
#           960 (  4n x 240s)      60.1u      62.8u (T=2)     1.00x   <- below crossover
#         1,536 ( 64n x  24s)     108.4u      78.5u (T=4)     1.38x
#         6,144 (256n x  24s)     437.0u     190.4u (T=4)     2.30x
#        15,360 ( 64n x 240s)     993.3u     400.8u (T=4)     2.48x
#        98,304 (4096n x 24s)   7,008.3u   1,504.1u (T=14)    4.66x
#       591,872 (256n x2312s)  42,206.1u   9,288.4u (T=14)    4.54x
#       983,040 (4096n x240s)  91,897.2u  16,690.1u (T=14)    5.51x
#     2,367,488 (1024n x2312s) 172,075.1u  37,360.5u (T=14)   4.61x
#
# EXECUTOR: a persistent barrier pool, NOT ThreadPoolExecutor. Measured head-to-head on the
#   identical workload (16,384 synapses), TPE.map is 0.69x-0.80x at T=2/4 -- SLOWER than
#   serial -- because it allocates a Future per chunk and pushes it through a lock-guarded
#   queue; the barrier pool is a stable 1.38x-2.70x on the same shapes. Picking the wrong
#   executor does not merely cost speed here, it inverts the sign of the result.
#
# THE GATE IS DERIVED, NEVER HARDCODED (P1). `calibrate()` times three things on the host that
#   will run the rule -- the fork/join floor F, the serial per-synapse rate r_ser, and the
#   parallel per-synapse rate r_par -- and solves S* = F / (r_ser - r_par). If r_ser <= r_par
#   there is no crossover and the plan is serial forever (honest null, reachable red below).
#   Stability of the derived S*, 7 independent calibrations: T=2 882..939, T=4 959..1017,
#   T=8 1842..2348, T=14 3399..3683 -- +-5%, zero non-finite. It agrees with the empirical
#   table above: S*(T=4) = 992 predicts the win starting just under 1,000 synapses, and the
#   960-synapse shape measured a wash while 1,536 won 1.38x.
#   An EARLIER draft of this calibration used TPE and returned "no crossover" in 4 of 7 runs;
#   that reading was TRUE OF TPE rather than noise, and it is recorded because a gate that
#   silently reports "always serial" is the failure mode this whole section is exposed to.
#
# HONEST STATUS -- read this before crediting anything. `reward_population` has ZERO live
#   callers today. Every RSTDPNeuron in this tree is a SINGLE neuron: interaction_learning
#   rewards one neuron of n_in=52 per deposit, task_set holds one of n_sets, benna_fusi one,
#   brain_native_selfbuild one. The live path is therefore ~52 synapses against a crossover of
#   ~1,000, three orders of magnitude below it, and the gate will correctly choose serial for
#   all of them. This is the channel, built and measured, for a population that does not exist
#   yet (P0 permits scaffolding on exactly these terms: label it, measure it, credit nothing
#   until a consumer exists). It buys the brain NOTHING at today's width and that is not a
#   claim being hedged -- it is the measurement.
#
#   AND IT IS INVISIBLE TO BOTH P3 GATES, which is the part worth saying out loud. `dormancy_gate`
#   is per-MODULE and r_stdp is reachable, so it passes on the module's account, not this
#   function's. `method_dormancy_gate` audits learning-verb METHODS inside classes, and
#   `reward_population` is a module-level function, so it is out of that gate's scope by
#   construction rather than by adjudication -- both gates run clean over this addition while
#   neither has an opinion about it. Under P4c that makes it a build the gates cannot see, and
#   the disposition is therefore carried here in prose: it is P3 shape 1 (present, importable,
#   never invoked) until a population consumer adopts it. The consumers that would are
#   `cortex/interaction_learning`, `cortex/task_set`, `plasticity/benna_fusi` and
#   `neuromodulation/naca`, none of which is in this lane's write scope, and none of which has a
#   broadcast-to-many shape today anyway -- adopting this is a change to what those modules DO,
#   not a call site to add.
# ======================================================================================

# Explicit documented cap, not a behaviour arbiter -- and the distinction is exact here, not a
# plea: because both paths are byte-identical, NO constant in this section can change a single
# weight. It can only change how long the update takes. P1 governs values that arbitrate
# BEHAVIOUR; nothing below can.
# 4 is the measured best value on the dev box: S*=992 at 2.68x, against S*=915 at 1.50x (T=2),
# S*=2094 at 3.16x (T=8) and S*=3501 at 2.63x (T=14) -- the lowest crossover with near-best
# speedup, which is the right trade for a population that has to GROW past the crossover before
# any of this matters.
# WHAT THE CAP COSTS, stated because it is a real forgone win rather than a free choice: at large
# populations more workers keep paying, and T=14 measured 4.66x at 98,304 synapses and 5.51x at
# 983,040 where T=4 was ~2.5-3x. A caller that knows it is at that scale should pass `workers=`
# explicitly; the crossover is re-derived per worker count, so overriding it stays measured
# rather than guessed.
_WORKER_CAP = 4
# Calibration workload, and its SIZE is the whole ballgame. S* = F/(r_ser - r_par) recovers r_par
# by subtracting the fork/join floor from the parallel timing, so the parallel region has to be
# comfortably larger than that floor or the remainder is mostly noise. A first draft used 4,096
# synapses: split 4 ways that is ~1,024 synapses (~66us) per thread against a ~40us floor, i.e.
# the measurement was only ~1.6x the thing being subtracted, and cold calibrations returned "never
# pays" often enough to matter. 16,384 synapses gives ~4,096 (~262us) per thread against the same
# floor -- ~6.5x -- which is what the lab sweep measured stable (7 runs, S* 893..1010, zero
# non-finite). Reps and trials are cut to keep the total near the smaller version's cost.
_CAL_NEURONS, _CAL_SYN = 64, 256
_CAL_REPS, _CAL_TRIALS, _FLOOR_REPS = 3, 2, 200
# MEDIAN OF THREE INDEPENDENT SOLVES. S* = F/(r_ser - r_par) divides by the difference of two
# measured numbers, so a solve taken while the machine is busy -- where r_par drifts up toward
# r_ser -- does not degrade gracefully, it explodes or goes non-finite. The first solve in a
# process is the one cached for the life of that process, which is exactly the wrong place for a
# fragile reading. Kept as a median rather than as a clamp on the measured efficiency, because a
# clamp would also manufacture a finite crossover on a host where fan-out genuinely never pays,
# destroying the honest null.
#
# HOW STABLE IT ACTUALLY IS, measured cold in fresh processes (this is the number to trust, not
# the mechanism's plausibility). 16 back-to-back launches at the shipped constants: S* = 898, 912,
# 914, 933, 934, 947, 949, 954, 959, 964, 980, 986, 1026, 1049, 1077 -- 15 of 16 inside 898..1077,
# every one consistent with the empirical crossover (the 960-synapse shape measured a wash, 1,536
# won 1.38x). 10 launches spaced 1.5s apart at the earlier, smaller constants were likewise all
# usable (853..1464). The 16th was process #1 of a batch started while the previous batch was
# still tearing down, and it came back "never pays"; its calibration took 270ms against the
# others' ~130ms, i.e. the retry below fired and the machine was still busy when it re-measured.
# That is a sustained-load artifact of the launcher, not a signal-to-noise problem in the solve --
# enlarging the calibration population 4x tightened the good runs and did not remove it, and
# neither the warm-up nor the retry is claimed to.
# DISCLOSED PLAINLY because it is the defect class this project hunts: when it happens, the gate
# reports "serial at every size" and the parallel path is off for that whole process. The
# direction is conservative -- an over-large or non-finite S* keeps the serial path, which is
# never slower than serial, so nothing is ever made worse -- and it is inspectable and
# recoverable rather than silent: `calibrate(workers).crossover_synapses` shows it and
# `calibrate(workers, force=True)` re-measures.
_CAL_SOLVES = 3

_POOL_LOCK = threading.Lock()
_POOLS = {}  # workers -> _FanOut          (lazy; nothing is constructed at module scope)
_CALIBRATIONS = {}  # workers -> Calibration
# Which pool (if any) the CURRENT thread is a worker of. Written once by each worker at the top
# of its loop, read only by that same thread, so it needs no lock and cannot race. It exists to
# turn nested fan-out from a silent permanent hang into an immediate error -- see _FanOut.run.
_LOCAL = threading.local()


class Calibration(NamedTuple):
    """What the host measured. `crossover_synapses` is math.inf when parallel never pays."""

    workers: int
    floor_us: float
    serial_ns_per_syn: float
    parallel_ns_per_syn: float
    crossover_synapses: float


class ParallelPlan(NamedTuple):
    """The decision for one call. `path` is 'serial' or 'parallel' -- exposed so a caller (and
    this module's own self-check) can assert which branch ran, rather than trusting that a pool
    silently did something."""

    path: str
    workers: int
    total_synapses: int
    crossover_synapses: float


class _FanOut:
    """Persistent workers joined by two barriers -- the fork/join handshake that measured
    1.38x-2.70x where ThreadPoolExecutor.map measured 0.69x-0.80x on the same workload.

    One call at a time per pool (`_run_lock`): concurrent population updates on the same worker
    count serialize rather than racing on the shared job slot. Threads are daemons and are never
    reclaimed; the pool is process-lifetime, created on first use and reused.

    A WORKER MUST REACH `_done` ON EVERY PATH, INCLUDING A RAISING ONE. This is the whole
    fragility of a barrier pool and it was a live defect here until 2026-08-05: `_loop` called
    `self._fn(k)` unguarded, so a job that raised killed its worker thread, the remaining parties
    never reached `Barrier(workers + 1)`, and the caller blocked FOREVER -- with the pool
    permanently destroyed for the life of the process. MEASURED before the fix, on plain
    `RSTDPNeuron`s with a plain bad broadcast value: `reward_population(pop, None, path="serial")`
    raised `TypeError` in microseconds while `path="parallel"` on identical input never returned.
    Two paths documented as byte-identical diverged into "raises" versus "hangs", and the
    byte-identity harness could not see it because it only ever compared SUCCESSFUL runs. The
    exception is now caught, parked per worker, and re-raised in the caller.

    NOT RE-ENTRANT: a worker that calls `run` on its OWN pool would block on `_run_lock` while the
    caller holds it and waits at `_done`. That used to be a documented deadlock; it now raises
    immediately (`_LOCAL.pool`). Nothing in this tree does it -- there is no population consumer at
    all yet -- but a hang is the one failure mode that gives a reader nothing to work from, and
    `run_selfchecks.py` runs part of the suite in ONE process, where a hung pool takes the whole
    gate with it rather than failing one module."""

    def __init__(self, workers):
        self.workers = workers
        self._fn = None
        self._n_jobs = 0
        self._err = [None] * workers
        self._start = threading.Barrier(workers + 1)
        self._done = threading.Barrier(workers + 1)
        self._run_lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._loop, args=(k,), daemon=True, name=f"rstdp-{k}")
            for k in range(workers)
        ]
        for t in self._threads:
            t.start()

    def _loop(self, k):
        _LOCAL.pool = self  # thread-local; only this thread ever reads it (see run())
        while True:
            self._start.wait()  # barrier gives happens-before on _fn / _n_jobs / _err
            try:
                if k < self._n_jobs:
                    self._fn(k)
            except BaseException as exc:  # noqa: BLE001 -- see the class docstring: a worker that
                self._err[k] = exc  # does not reach _done hangs the caller forever
            self._done.wait()

    def run(self, fn, n_jobs):
        """Call fn(k) for k in range(n_jobs) across the workers; returns when all have finished.

        If a job raises, every worker still reaches the join barrier (so the pool stays usable)
        and the LOWEST-k exception is re-raised here. Lowest-k is what keeps the error path
        deterministic AND serial-identical: `partition` chunks are contiguous and order-preserving,
        so the lowest failing chunk contains exactly the element the serial comprehension would
        have failed on -- same exception, same offender, either path."""
        if n_jobs > self.workers:
            raise ValueError(f"n_jobs {n_jobs} exceeds workers {self.workers}")
        if getattr(_LOCAL, "pool", None) is self:
            raise RuntimeError(
                f"nested fan-out on the same {self.workers}-worker pool: a worker called run() on "
                f"its own pool, which would deadlock. Use a different worker count, or run serially."
            )
        with self._run_lock:
            self._fn, self._n_jobs = fn, n_jobs
            self._err = [None] * self.workers
            self._start.wait()
            self._done.wait()
            err = next((e for e in self._err if e is not None), None)
        if err is not None:
            raise err


def default_workers():
    """Worker count for an uncapped call: the measured cap, never more than the machine has."""
    return max(1, min(_WORKER_CAP, os.cpu_count() or 1))


def partition(seq, k):
    """Split `seq` into at most k contiguous, DISJOINT, order-preserving chunks; empty chunks are
    dropped so k > len(seq) is not an error. Disjointness is what makes the parallel path
    byte-identical -- two threads must never hold the same neuron."""
    n = len(seq)
    if k < 1:
        raise ValueError(f"k must be >= 1 ({k})")
    step, rem = divmod(n, k)
    out, lo = [], 0
    for j in range(k):
        hi = lo + step + (1 if j < rem else 0)
        if hi > lo:
            out.append(seq[lo:hi])
        lo = hi
    return out


def _fanout(workers):
    with _POOL_LOCK:
        pool = _POOLS.get(workers)
        if pool is None:
            pool = _POOLS[workers] = _FanOut(workers)
        return pool


def _time_us(fn, reps, trials):
    for _ in range(3):
        fn()  # wake the pool threads BEFORE the timed region (cold-wake skews short runs)
    best = None
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = (time.perf_counter() - t0) / reps * 1e6
        best = dt if best is None else min(best, dt)
    return best


def calibrate(workers=None, force=False):
    """MEASURE this host; do not assume it. Times the fork/join floor, the serial per-synapse
    rate and the parallel per-synapse rate on a throwaway population, then solves the crossover
    S* = F / (r_ser - r_par). Repeated `_CAL_SOLVES` times and medianed (see that constant for
    the outlier that made it necessary). Cached per worker count for the life of the process.

    Costs ~130ms on first use for a given worker count (~270ms in the rare case the
    self-consistency retry fires). `reward_population` avoids paying it at all when the
    population cannot be partitioned (fewer than 2 neurons).

    Returns a Calibration. `crossover_synapses` is math.inf when the parallel form is no faster
    per synapse than the serial one -- on a single core, at workers=1, or on any host where the
    fan-out does not pay. That is the honest null and the plan is then 'serial' at every size."""
    w = default_workers() if workers is None else int(workers)
    if w < 1:
        raise ValueError(f"workers must be >= 1 ({w})")
    if not force:
        hit = _CALIBRATIONS.get(w)
        if hit is not None:
            return hit

    pop = [RSTDPNeuron(_CAL_SYN, seed=i) for i in range(_CAL_NEURONS)]
    for cell in pop:
        for i in range(_CAL_SYN):
            cell.e[i] = 1e-6  # tiny: calibration must not produce a meaningful weight change
    total = _CAL_NEURONS * _CAL_SYN
    parts = partition(pop, w)
    pool = _fanout(w)

    def _chunk(k):
        for cell in parts[k]:
            cell.reward(1e-9)

    def _solve():
        floor = _time_us(lambda: pool.run(lambda k: None, len(parts)), _FLOOR_REPS, _CAL_TRIALS)
        par = _time_us(lambda: pool.run(_chunk, len(parts)), _CAL_REPS, _CAL_TRIALS)
        ser = _time_us(lambda: [cell.reward(1e-9) for cell in pop], _CAL_REPS, _CAL_TRIALS)
        r_ser = ser / total
        r_par = max(0.0, par - floor) / total
        star = math.inf if r_ser <= r_par else floor / (r_ser - r_par)
        return floor, r_ser, r_par, star

    # WARM-UP, untimed. The threads were created moments ago and the process itself may still be
    # settling (imports, first-touch page faults, whatever launched it). Stated honestly: this did
    # NOT measurably change the spoiled-calibration rate under launch contention (see _CAL_SOLVES);
    # it is kept because timing a thread pool on its first fan-out is wrong regardless.
    for _ in range(5):
        pool.run(_chunk, len(parts))
        for cell in pop:
            cell.reward(1e-9)

    solves = [_solve() for _ in range(_CAL_SOLVES)]
    if not 0.0 < sorted(s[3] for s in solves)[_CAL_SOLVES // 2] < total:
        # SELF-CONSISTENCY, not a threshold on the host's speed, so it needs no constant. The
        # calibration population IS `total` synapses, so a solve returning S* >= total has just
        # claimed that the very population it timed would not benefit from fanning out -- while
        # timing it fan out. One retry at bounded cost. Measured honestly: this rescues a solve
        # spoiled by a brief disturbance, and does NOT rescue one spoiled by a sustained busy
        # machine (a contended launch defeated the retry too, ~1 in 20). If the retry agrees, the
        # answer is accepted as-is -- on a host where fan-out really does not pay at this size
        # that is the CORRECT answer, and accepting it only ever costs a foregone win: an
        # over-large S* keeps the serial path, which is never slower than serial.
        solves = [_solve() for _ in range(_CAL_SOLVES)]
    # median per term. `statistics.median` on an even count would average two solves; _CAL_SOLVES
    # is odd so every reported term is one solve's real measurement, never a blend of two.
    floor = sorted(s[0] for s in solves)[_CAL_SOLVES // 2]
    r_ser = sorted(s[1] for s in solves)[_CAL_SOLVES // 2]
    r_par = sorted(s[2] for s in solves)[_CAL_SOLVES // 2]
    star = sorted(s[3] for s in solves)[_CAL_SOLVES // 2]
    cal = Calibration(w, floor, r_ser * 1000.0, r_par * 1000.0, star)
    _CALIBRATIONS[w] = cal
    return cal


def parallel_plan(n_neurons, total_synapses, workers=None):
    """Which path a population of this shape takes, WITHOUT running it. Exposed so the decision
    is inspectable: a pool that silently falls back to serial is the defect class this project
    hunts, and this is how a caller proves which branch it got."""
    w = default_workers() if workers is None else int(workers)
    if n_neurons < 2 or w < 2 or total_synapses <= 0:
        # Cannot partition, or nothing to do. Serial WITHOUT paying for a calibration.
        return ParallelPlan("serial", w, int(total_synapses), math.inf)
    star = calibrate(w).crossover_synapses
    path = "parallel" if total_synapses > star else "serial"
    return ParallelPlan(path, w, int(total_synapses), star)


def reward_population(neurons, d, workers=None, path=None):
    """Apply ONE broadcast reward d across a population of RSTDPNeuron.

    BYTE-IDENTICAL to `[n.reward(d) for n in neurons]` -- same weights, same returned deltas, in
    the same order -- on both paths. That is structural, not a hope: the partition hands each
    thread whole neurons, so no two threads touch the same weight list and nothing is reduced
    across threads (float addition is not associative, so a parallel REDUCTION would not have
    this property and none is performed).

    Chooses serial or parallel from `parallel_plan`, whose threshold is measured on this host.
    `path` overrides that choice ('serial' / 'parallel') and exists because byte-identity makes
    the choice behaviour-neutral: the self-check forces BOTH branches and compares them exactly,
    so "the pool quietly stopped running in parallel" cannot pass as a green gate.

    ON FAILURE the two paths raise the SAME exception for the same offending neuron (see
    `_FanOut.run` for why lowest-k gives that), but they do NOT leave the population in the same
    state, and this is the one place the byte-identity claim above stops applying. Serial mutates
    neurons before the offender and none after; parallel additionally mutates whichever neurons
    the other threads had already reached. That is inherent to fanning out -- undoing it would
    need a transaction, and this rule has no rollback -- so the honest statement is that
    byte-identity is a claim about SUCCESSFUL updates only. `workers=1` with `path="parallel"`
    also runs serially: one worker cannot partition, and the result is identical either way.

    Returns the list of per-synapse delta lists, one per neuron, in input order."""
    if path not in (None, "serial", "parallel"):
        raise ValueError(f"path must be None, 'serial' or 'parallel' ({path!r})")
    pop = list(neurons)
    if len(pop) < 2:
        return [cell.reward(d) for cell in pop]

    if path is None:
        plan = parallel_plan(len(pop), sum(cell.n for cell in pop), workers=workers)
    else:
        w = default_workers() if workers is None else int(workers)
        plan = ParallelPlan(path, w, sum(cell.n for cell in pop), math.nan)
    if plan.path == "serial" or plan.workers < 2:
        return [cell.reward(d) for cell in pop]

    parts = partition(pop, plan.workers)
    out = [None] * len(parts)

    def _chunk(k):
        out[k] = [cell.reward(d) for cell in parts[k]]

    _fanout(plan.workers).run(_chunk, len(parts))
    return [deltas for sub in out for deltas in sub]


def _run(seed):
    """A fixed multi-synapse protocol used to check determinism (same seed -> same weights)."""
    n = RSTDPNeuron(3, seed=seed)
    n.pre_spike(0, 5.0)
    n.pre_spike(1, 6.0)
    n.post_spike(10.0)
    for _ in range(20):
        n.step()
    n.pre_spike(2, 30.0)
    n.reward(+1.0)
    return n.w


def main():
    print("=" * 70)
    print("R-STDP -- reward-modulated STDP with a decaying eligibility trace")
    print("(distal-reward rule: causal spike pairing reinforced by a LATER reward)")
    print("=" * 70)

    # 1. CAUSAL POTENTIATION: pre-before-post (dt>0) + positive delayed reward -> w up.
    n1 = RSTDPNeuron(1, seed=0)
    w0 = n1.w[0]
    n1.pre_spike(0, 5.0)
    n1.post_spike(10.0)
    for _ in range(5):
        n1.step()
    n1.reward(+1.0)
    causal = n1.w[0] - w0
    print(f"1 causal potentiation : pre-before-post, reward +1 -> dw = {causal:+.4f}")

    # 2. ANTI-CAUSAL DEPRESSION: post-before-pre (dt<0) + positive reward -> w down.
    n2 = RSTDPNeuron(1, seed=0)
    w0 = n2.w[0]
    n2.post_spike(5.0)
    n2.pre_spike(0, 10.0)
    for _ in range(5):
        n2.step()
    n2.reward(+1.0)
    anti = n2.w[0] - w0
    print(f"2 anti-causal depress : post-before-pre, reward +1 -> dw = {anti:+.4f}")

    # 3. DISTAL REWARD (headline): a causal pairing, then a LONG delay, then a distractor
    #    fires near reward time. The causally-paired synapse is still reinforced far more.
    nd = RSTDPNeuron(2, seed=0)
    before = list(nd.w)
    nd.pre_spike(0, 5.0)  # synapse 0: causal pre-before-post pairing
    nd.post_spike(10.0)
    for _ in range(100):  # many timesteps pass; other activity happens
        nd.step()
    nd.pre_spike(1, 112.0)  # synapse 1 (distractor): active near reward, NOT causally paired
    nd.reward(+1.0)
    paired_gain = nd.w[0] - before[0]
    distractor_gain = nd.w[1] - before[1]
    print(
        f"3 distal reward       : delay=100 steps  paired dw = {paired_gain:+.4f}  "
        f"distractor dw = {distractor_gain:+.4f}"
    )

    # 4. REWARD-SIGN FLIP: the same causal pairing but a NEGATIVE reward -> w down.
    n4 = RSTDPNeuron(1, seed=0)
    w0 = n4.w[0]
    n4.pre_spike(0, 5.0)
    n4.post_spike(10.0)
    for _ in range(5):
        n4.step()
    n4.reward(-1.0)
    neg = n4.w[0] - w0
    print(f"4 reward-sign flip    : causal pairing, reward -1 -> dw = {neg:+.4f}")

    # 5. DETERMINISM: same seed -> identical final weights across two fresh instances.
    w_a, w_b = _run(7), _run(7)
    print(f"5 determinism         : two seed=7 runs identical = {w_a == w_b}")

    # 6. POPULATION BROADCAST: serial vs parallel, compared EXACTLY (not approximately).
    #    Float addition is not associative, so "close enough" would hide exactly the defect
    #    this check exists to catch. Both branches are FORCED so neither can be vacuous.
    n_cpu = os.cpu_count() or 1
    workers = default_workers()
    syn, n_big = 64, 64

    def _fresh(n, k, seed0=11):
        pop = [RSTDPNeuron(k, seed=seed0 + i) for i in range(n)]
        for j, cell in enumerate(pop):
            for i in range(k):
                cell.e[i] = ((i * 7 + j * 13) % 21 - 10) / 100.0  # mixed signs, exact binaries
        return pop

    pop_s, pop_p = _fresh(n_big, syn), _fresh(n_big, syn)
    d_s = reward_population(pop_s, 0.37, workers=workers, path="serial")
    d_p = reward_population(pop_p, 0.37, workers=workers, path="parallel")
    w_par = [list(c.w) for c in pop_p]
    identical = [list(c.w) for c in pop_s] == w_par and d_s == d_p
    # and against the plain reference the docstring claims equivalence to
    pop_r = _fresh(n_big, syn)
    d_r = [c.reward(0.37) for c in pop_r]
    vs_reference = [list(c.w) for c in pop_r] == w_par and d_r == d_p

    # proof the forced-parallel branch really ran on more than one OS thread
    seen = set()
    if workers >= 2:
        _fanout(workers).run(lambda k: seen.add(threading.get_ident()), workers)

    # THE GATE'S DECISION IS TESTED AGAINST AN INJECTED CALIBRATION, NOT AGAINST THE HOST'S
    # CURRENT SPEED. An earlier version of this self-check asserted `math.isfinite(star)` on the
    # live measurement and went RED on a busy machine -- turning a load artifact into a gate
    # failure, which is a flaky test, not a real criterion. What must be correct here is the
    # DECISION LOGIC (does the comparison flip at the crossover, and does a "never pays"
    # calibration stay serial at any size); that is deterministic and is what is asserted. The
    # live measurement is REPORTED below instead, because it is evidence about this host, not a
    # property of the code.
    probe_w = 3 if workers != 3 else 2
    _saved = _CALIBRATIONS.get(probe_w)
    try:
        _CALIBRATIONS[probe_w] = Calibration(probe_w, 40.0, 64.0, 20.0, 1000.0)
        plan_below = parallel_plan(64, 1000, workers=probe_w)  # not strictly above -> serial
        plan_above = parallel_plan(64, 1001, workers=probe_w)
        _CALIBRATIONS[probe_w] = Calibration(probe_w, 40.0, 64.0, 64.0, math.inf)
        plan_null = parallel_plan(64, 10**9, workers=probe_w)  # "never pays" -> serial forever
    finally:
        if _saved is None:
            _CALIBRATIONS.pop(probe_w, None)  # leave no poisoned cache for the rest of the process
        else:
            _CALIBRATIONS[probe_w] = _saved

    plan_small = parallel_plan(2, 2, workers=workers)
    # the live host measurement: reported, and checked only for self-consistency with the plan
    cal = calibrate(workers)
    star = cal.crossover_synapses
    big_total = n_big * syn
    plan_live = parallel_plan(n_big, big_total, workers=workers)
    # whatever the gate picks, it must still equal forced-serial byte-for-byte
    pop_g = _fresh(n_big, syn)
    reward_population(pop_g, 0.37, workers=workers)
    gate_matches = [list(c.w) for c in pop_g] == w_par

    print("=" * 70)
    print(
        f"6 population broadcast: {n_big} neurons x {syn} syn = {big_total} synapses; "
        f"serial == parallel byte-exact = {identical}; gate-chosen == forced = {gate_matches}"
    )
    print(
        f"  gate LOGIC (injected crossover 1000): 1000 syn -> {plan_below.path}, "
        f"1001 syn -> {plan_above.path}, never-pays -> {plan_null.path}"
    )
    star_txt = f"{star:.0f}" if math.isfinite(star) else "none (host too busy to measure)"
    print(
        f"  gate MEASURED on this host (T={cal.workers}): floor {cal.floor_us:.1f}us, serial "
        f"{cal.serial_ns_per_syn:.1f}ns/syn, parallel {cal.parallel_ns_per_syn:.1f}ns/syn "
        f"-> crossover {star_txt}"
    )
    print(
        f"  plan: {big_total} syn -> {plan_live.path}, 2 syn -> {plan_small.path}; "
        f"pool ran on {len(seen)} distinct threads (cpus={n_cpu})"
    )
    print("=" * 70)

    # ---- self-check (deterministic asserts, ASCII) ----
    assert causal > 0.02, f"causal pre-before-post + reward must potentiate ({causal:+.4f})"
    assert anti < -0.02, f"anti-causal post-before-pre + reward must depress ({anti:+.4f})"
    assert paired_gain > distractor_gain + 0.03, (
        f"distal: causally-paired synapse must beat the distractor ({paired_gain:+.4f} vs {distractor_gain:+.4f})"
    )
    assert paired_gain > 0.02, (
        f"the paired synapse must still be reinforced after the delay ({paired_gain:+.4f})"
    )
    assert neg < 0, f"negative reward must flip the causal update downward ({neg:+.4f})"
    assert w_a == w_b, "same seed must give identical final weights (determinism)"

    # ---- population broadcast asserts ----
    assert identical, "parallel weight update must be BYTE-IDENTICAL to serial, not close to it"
    assert vs_reference, "both paths must equal the plain [n.reward(d) for n in pop] reference"
    assert gate_matches, "the gate-chosen path must match the forced paths byte-for-byte"
    assert plan_small.path == "serial", (
        f"a 2-synapse population must not fan out ({plan_small.path})"
    )
    # the gate's DECISION LOGIC, deterministic (injected calibration, no timing involved)
    assert plan_below.path == "serial", (
        f"total == crossover must NOT fan out ({plan_below.path}); the gate is strictly-above"
    )
    assert plan_above.path == "parallel", (
        f"one synapse above the crossover must fan out ({plan_above.path}) -- a gate stuck on "
        f"'serial' is the silent-fallback defect this project hunts"
    )
    assert plan_null.path == "serial", (
        f"a 'never pays' calibration must stay serial at any size ({plan_null.path})"
    )
    # zero / empty / one, per the new-baseline rule
    assert reward_population([], 0.5) == [], "empty population must return []"
    _one = RSTDPNeuron(3, seed=1)
    assert len(reward_population([_one], 0.5)) == 1, "single neuron must still be rewarded"
    assert parallel_plan(1, 52, workers=workers).path == "serial", "one neuron cannot partition"
    assert parallel_plan(8, 0, workers=workers).path == "serial", "zero synapses cannot pay"
    assert reward_population([RSTDPNeuron(0, seed=2), RSTDPNeuron(0, seed=3)], 0.5) == [[], []], (
        "zero-synapse neurons must round-trip as empty delta lists"
    )
    # partition: disjoint, total, order-preserving -- the property byte-identity rests on
    _seq = list(range(37))
    for _k in (1, 2, 4, 8, 40):
        _chunks = partition(_seq, _k)
        assert [x for c in _chunks for x in c] == _seq, f"partition must preserve order (k={_k})"
        assert sum(len(c) for c in _chunks) == len(_seq), f"partition must cover all (k={_k})"
        assert all(_chunks), f"partition must not emit empty chunks (k={_k})"
    if workers >= 2:
        assert len(seen) >= 2, f"forced-parallel pool must use >1 thread ({len(seen)})"
        # the live plan must AGREE with the live measurement -- consistency, not a speed target
        assert plan_live.path == ("parallel" if big_total > star else "serial"), (
            f"plan {plan_live.path} disagrees with the measured crossover {star} at {big_total}"
        )
    else:
        print(
            f"  NOTE: {n_cpu}-cpu host -- workers={workers}, so the parallel branch is "
            f"unreachable here and its thread assert did NOT run. Byte-identity and the gate "
            f"logic above still ran (the latter on an injected calibration)."
        )
    if not math.isfinite(star):
        print(
            "  NOTE: the crossover could not be MEASURED on this host right now (the parallel "
            "calibration timed no faster than the serial one, which happens under sustained "
            "load). The gate therefore selects serial at every size for this process -- "
            "conservative, never slower than serial, and re-measurable via calibrate(force=True). "
            "The gate LOGIC asserts above ran regardless, on an injected calibration."
        )

    # ---- REACHABLE RED: construct and RUN the input that makes each criterion FIRE ----
    reds = []
    # (a) byte-identity criterion must reject a ONE-ULP difference, not tolerate it
    a_pop, b_pop = _fresh(4, 8), _fresh(4, 8)
    reward_population(a_pop, 0.37, path="serial")
    reward_population(b_pop, 0.37, path="serial")
    b_pop[2].w[5] = math.nextafter(b_pop[2].w[5], math.inf)
    reds.append(("byte-identity", [list(c.w) for c in a_pop] != [list(c.w) for c in b_pop]))
    # (b) the deltas half of the same criterion, perturbed on a REAL returned delta list.
    #     (This compared two literals until 2026-08-05, which tested Python's `!=` rather than
    #     anything in this module -- a red that cannot fail for a reason the module controls is
    #     the tautological-criterion shape `tools/red_reachability.py` exists to catch.)
    d_a = reward_population(_fresh(4, 8), 0.37, path="serial")
    d_b = reward_population(_fresh(4, 8), 0.37, path="parallel")
    _ok_delta = d_a == d_b
    d_b[1][2] = math.nextafter(d_b[1][2], math.inf)
    reds.append(("delta-identity", _ok_delta and d_a != d_b))
    # (c) the gate must flip at the crossover -- BOTH sides constructed and run, no timing
    reds.append(("gate-flips", plan_below.path == "serial" and plan_above.path == "parallel"))
    # (d) partition's coverage criterion must reject a partition that drops an element -- run
    #     against REAL `partition` output, not a hand-written literal (same fix as (b)).
    _cov = partition(list(range(37)), 4)
    _drop = [c[1:] if j == 1 else c for j, c in enumerate(_cov)]
    reds.append(
        (
            "partition-covers",
            [x for c in _cov for x in c] == list(range(37))
            and [x for c in _drop for x in c] != list(range(37)),
        )
    )
    # (e) the honest null, reached two independent ways: a 'never pays' calibration, and
    #     workers=1 (cannot partition at all). Both must be serial at an absurd size.
    reds.append(
        (
            "no-crossover-null",
            plan_null.path == "serial" and parallel_plan(4096, 10**9, workers=1).path == "serial",
        )
    )
    # (f) `path=` must reject a typo rather than silently running serial -- a silent fallback
    #     on a bad argument is the same defect class as a silent fallback on a bad measurement
    try:
        reward_population(_fresh(2, 2), 0.5, path="paralel")
        reds.append(("path-validated", False))
    except ValueError:
        reds.append(("path-validated", True))

    # (g) A RAISING JOB MUST RAISE, NOT HANG -- the defect this pool actually had (see
    #     `_FanOut`). Bounded in a helper thread so that a REGRESSION fails this gate with a
    #     message instead of hanging `run_selfchecks.py` forever, which is what the unfixed pool
    #     did. Both paths must raise the SAME exception type on the SAME offending neuron.
    def _bounded(fn, timeout=30.0):
        box = {}

        def go():
            try:
                box["v"] = ("returned", fn())
            except BaseException as exc:  # noqa: BLE001 -- classifying, not swallowing
                box["v"] = ("raised", exc)

        t = threading.Thread(target=go, daemon=True)
        t.start()
        t.join(timeout)
        return box.get("v", ("hung", None))

    if workers >= 2:
        bad = [RSTDPNeuron(4, seed=i) for i in range(2 * workers)]
        ser_out = _bounded(
            lambda: reward_population(list(bad), None, workers=workers, path="serial")
        )
        par_out = _bounded(
            lambda: reward_population(list(bad), None, workers=workers, path="parallel")
        )
        reds.append(
            (
                "worker-exception-raises",
                ser_out[0] == "raised"
                and par_out[0] == "raised"
                and type(ser_out[1]) is type(par_out[1]),
            )
        )
        # ...and the pool must still be USABLE afterwards: a dead worker would leave the barrier
        # short a party forever, so this call is exactly what hung before the fix.
        after = _bounded(
            lambda: reward_population(_fresh(8, 4), 0.37, workers=workers, path="parallel")
        )
        reds.append(("pool-survives-exception", after[0] == "returned"))
        # (h) nested fan-out must RAISE rather than deadlock on its own _run_lock
        nested = _bounded(
            lambda: _fanout(workers).run(
                lambda k: reward_population(_fresh(4, 2), 0.5, workers=workers, path="parallel"), 1
            )
        )
        reds.append(
            ("nested-fanout-raises", nested[0] == "raised" and isinstance(nested[1], RuntimeError))
        )
        print(
            f"  failure paths: serial {ser_out[0]} {type(ser_out[1]).__name__}, parallel "
            f"{par_out[0]} {type(par_out[1]).__name__}; pool after failure: {after[0]}; "
            f"nested fan-out: {nested[0]} {type(nested[1]).__name__}"
        )
    assert all(ok for _, ok in reds), f"unreachable red: {[n for n, ok in reds if not ok]}"

    print(
        f"self-check OK: causal {causal:+.4f}, anti {anti:+.4f}, distal paired {paired_gain:+.4f} "
        f"beats distractor {distractor_gain:+.4f}, neg-reward {neg:+.4f}; eligibility trace bridges "
        f"the reward delay; additive; deterministic"
    )
    print(
        f"self-check OK: population broadcast byte-identical serial vs parallel over "
        f"{big_total} synapses (weights AND deltas, exact ==); gate DERIVED from a host "
        f"measurement (crossover {star_txt}, not a constant) and its decision logic proven "
        f"deterministically; {len(reds)}/{len(reds)} criteria proven falsifiable"
    )

    # ---- THE THIRD-FACTOR PORT IS CHARGED, measured (specs/022 T029) -------------------
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    live, pinned_out, reads, responded = _measure_da_charge()
    assert reads == len(live), f"the neuter must be READ every frame ({reads} vs {len(live)})"
    assert live != pinned_out, (
        "pinning the DA payload to its tonic rpe=0 rate must change the weight trajectory "
        "-- an undiverged third-factor port is a declared, not a charged, port"
    )
    assert responded, (
        "the live phasic stream must MOVE the weights more than the tonic pin -- "
        "divergence without a moved weight attributes nothing"
    )
    rec = read_attribution(DA_EDGE)
    assert rec is not None and rec.get("diverged") is True, (
        f"committed charge evidence must carry the diverged {DA_EDGE} attribution -- "
        "regenerate with --record-charge and commit it"
    )
    fresh = {"reads": reads, "n_frames": len(live)}
    assert rec.get("measured", {}) == fresh, (
        f"recorded charge evidence does not reproduce: {rec.get('measured')} vs {fresh} -- "
        "regenerate with --record-charge in the same commit"
    )
    print(
        f"THIRD-FACTOR PORT CHARGED (measured): live phasic DA vs tonic pin diverged over "
        f"{reads} read frames and the live arm's weights moved; committed attribution "
        f"reproduces."
    )


if __name__ == "__main__":
    import sys as _sys

    if "--record-charge" in _sys.argv[1:]:
        _sys.exit(record_charge())
    main()
