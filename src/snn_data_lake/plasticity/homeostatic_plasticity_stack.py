"""
Homeostatic plasticity STACK -- a fixed ordered sandwich of complementary rules that STABILIZES the
live R-STDP learning synapses, so the brain's one real learning path stays bounded instead of drifting.

WHY THIS EXISTS (a merge, from a study of the brain). The plasticity/ modules (short_term, homeostatic,
intrinsic, bcm, ...) were STANDALONE studies -- each self-checks in isolation, none touches the live
brain. The only learning the integrated brain actually runs online is reward-modulated spike-timing
plasticity (R-STDP: plasticity.r_stdp.RSTDPNeuron, driven by readout.interaction_learning). Pure
reward-gated potentiation, like pure Hebbian potentiation, is UNSTABLE on its own -- weights and firing
rate can run away. Real cortex fixes this with HOMEOSTASIS wrapped around the fast learning rule. This
module composes three of the dead standalone rules into that missing wrap: a FIXED ORDERED SANDWICH run
every learning step AROUND the live R-STDP update --

    (1) short-term dynamic gain     [PRE]   -- Tsodyks-Markram release modulates the deposit signal
        --> [ the existing R-STDP weight update happens: w[i] += lr * gain * d * e[i] ] <--
    (2) homeostatic synaptic scaling [POST]  -- Turrigiano multiplicative scaling bounds the synapses
    (3) intrinsic plasticity         [POST]  -- Triesch rule holds the neuron's output rate at a set-point

turning ~3 dead plasticity modules into a per-tick faculty on the REAL learning path. The synaptic
weights stay BOUNDED (a ceiling on runaway) and the output firing rate stays at its target, instead of
diverging as pure R-STDP potentiation would.

REUSE, not reinvention. Each phase is a THIN ADAPTER over the existing rule module -- it imports the
real class and delegates the actual math to it, behind a common `.stabilize(ctx) -> ctx` contract:
  * short_term_plasticity.DynamicSynapse     -- the fast onset/burst gain (release u * resources R)
  * homeostatic_scaling.HomeostaticNeuron    -- .stabilize(x): multiplicative synaptic scaling to a set-point
  * intrinsic_plasticity.IntrinsicPlasticity -- .adapt(x): tunes gain a / bias b toward an output-rate target
No rule is rewritten here. Homeostatic scaling is applied as a soft CEILING (only trims a drive that has
run ABOVE its set-point, never boosts a losing neuron) so it bounds runaway without flattening routing
discrimination. Intrinsic plasticity tunes the neuron's own EXCITABILITY (not the synapses) -- the
biologically correct division of labour: synaptic homeostasis bounds the weights, cellular homeostasis
bounds the firing rate.

HONESTY / SAFETY. Dependencies are governed by P7: numpy/torch/mujoco are unconditional core deps and any
tier may import them freely. This module happens to need none of them, which is a fact about the arithmetic
(three closed-form scalar rules), not a design property worth advertising.
Additive and OPT-IN: the learner takes the stack as an
injected duck-typed object; with no stack (the default) the R-STDP path is BYTE-IDENTICAL to today. It
stabilizes plasticity only and adds NO actuator route (the intent-recognition learning path only; the
motor path is untouched). Deterministic: every rule here is closed-form, no rng.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.homeostatic_plasticity_stack
"""

import math

from snn_data_lake.plasticity.homeostatic_scaling import HomeostaticNeuron
from snn_data_lake.plasticity.intrinsic_plasticity import IntrinsicPlasticity
from snn_data_lake.plasticity.short_term_plasticity import DynamicSynapse

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "HomeostaticPlasticityStack.learning_step": (
        "LOCAL",
        "runs the whole fixed-order sandwich of local rules once and logs the order it touched each. "
        "Orchestration over R-STDP, BCM and synaptic scaling -- every member is local",
    ),
    "_rstdp_update_fn": (
        "LOCAL",
        "a closure performing ONE real R-STDP deposit on the neuron and returning its delta. "
        "Three-factor and synapse-local",
    ),
    "_learner_off_on_selfcheck": (
        "NOT_A_RULE",
        "wire check asserting the learner with the stack OFF is byte-identical to today. A self-check "
        "fixture",
    ),
}


# --------------------------------------------------------------------------------------------------
# Thin adapters -- each wraps ONE real rule module behind a common `.stabilize(ctx) -> ctx` contract.
# ctx is a small dict carrying the per-step state: neuron_id, weights, activity (per-synapse drive),
# gain (the PRE dynamic gain), drive (membrane y = sum(w*activity)), rate (the intrinsic output rate),
# tick (a global event clock). Adapters mutate only their own field(s); the stack composes them in a
# fixed order. "or similar" of the requested `.stabilize(weights, activity) -> weights` contract.
# --------------------------------------------------------------------------------------------------
class _ShortTermGain:
    """PRE phase. REUSES short_term_plasticity.DynamicSynapse (Tsodyks-Markram): one depressing synapse
    per neuron turns the stream of deposits to that neuron into an onset detector. A deposit arriving
    after a gap (an onset) rings out near full strength; a rapid burst of deposits on the same neuron
    fades (release resources deplete) -- a fast dynamic gain on the effective learning signal. Normalized
    so a lone onset -> ~1.0 (does not slow normal, well-spaced teaching) and only genuine bursts damp."""

    name = "short_term_dynamic_gain"
    reused = "short_term_plasticity.DynamicSynapse"

    def __init__(self, n_neurons, U=0.6, tau_f=1.0, tau_d=2.0, floor=0.5):
        self._syn = [DynamicSynapse(U, tau_f, tau_d) for _ in range(n_neurons)]
        self._last = [0 for _ in range(n_neurons)]  # last global tick this neuron was deposited to
        self._onset = float(U)  # first-spike epsc == U -> normalization reference
        self.floor = float(floor)

    def stabilize(self, ctx):
        i = ctx["neuron_id"]
        now = ctx.get("tick", self._last[i] + 1)
        dt = max(1.0, float(now - self._last[i]))  # gap since this neuron's previous deposit
        self._last[i] = now
        epsc = self._syn[i].spike(dt)  # REUSE the real Tsodyks-Markram release (u * R)
        g = epsc / self._onset  # normalize: onset ~1.0, sustained burst < 1.0
        g = min(1.0, max(self.floor, g))
        ctx["gain"] = ctx["gain"] * g
        return ctx


class _HomeostaticScaling:
    """POST phase. REUSES homeostatic_scaling.HomeostaticNeuron.stabilize (Turrigiano multiplicative
    synaptic scaling toward a rate set-point). Applied as a soft CEILING: each neuron's set-point is
    calibrated once to a multiple of its baseline drive, and scaling runs ONLY when the drive has run
    ABOVE that ceiling -- so it trims runaway (keeping the synapses BOUNDED) but never scales a losing
    neuron up, preserving routing discrimination. Multiplicative -> the relative weight pattern is kept."""

    name = "homeostatic_scaling"
    reused = "homeostatic_scaling.HomeostaticNeuron"

    def __init__(self, n_neurons, ceiling_factor=3.0, scale_rate=0.5):
        self._cell = [
            HomeostaticNeuron([0.0], target=1.0, scale_rate=scale_rate) for _ in range(n_neurons)
        ]
        self._target = [
            None for _ in range(n_neurons)
        ]  # per-neuron drive ceiling, set on first sight
        self.ceiling_factor = float(ceiling_factor)

    def stabilize(self, ctx):
        i = ctx["neuron_id"]
        cell = self._cell[i]
        cell.w = list(ctx["weights"])
        x = list(ctx["activity"])
        y = cell.output(x)  # membrane drive y = sum(w*x) == the readout v
        ctx["drive"] = y
        if self._target[i] is None:  # calibrate the ceiling from the baseline drive
            self._target[i] = max(self.ceiling_factor * abs(y), 1e-3)
        cell.target = self._target[i]
        if y > cell.target:  # only trim genuine runaway (never boost a loser)
            cell.stabilize(x)  # REUSE the real multiplicative synaptic scaling
            ctx["weights"] = list(cell.w)
            ctx["drive"] = cell.output(x)  # bounded drive after scaling
        return ctx


class _IntrinsicPlasticity:
    """POST phase. REUSES intrinsic_plasticity.IntrinsicPlasticity.adapt (Triesch rule): tunes the
    neuron's OWN gain a and bias b so its output firing-rate distribution approaches a target mean mu.
    This is CELLULAR homeostasis -- it stabilizes the firing rate without touching the synapses (the
    weights pass through unchanged), the biologically correct complement to synaptic scaling."""

    name = "intrinsic_plasticity"
    reused = "intrinsic_plasticity.IntrinsicPlasticity"

    def __init__(self, n_neurons, mu=0.15, lr=0.02):
        self._ip = [IntrinsicPlasticity(mu=mu, lr=lr) for _ in range(n_neurons)]
        self.mu = float(mu)

    def stabilize(self, ctx):
        i = ctx["neuron_id"]
        y = ctx.get("drive")
        if y is None:  # defensive: compute the drive if scaling was off
            y = sum(w * a for w, a in zip(ctx["weights"], ctx["activity"]))
            ctx["drive"] = y
        ctx["rate"] = self._ip[i].adapt(y)  # REUSE the real Triesch excitability self-tuner
        return ctx  # weights unchanged: intrinsic plasticity tunes the cell


# --------------------------------------------------------------------------------------------------
# The stack -- composes the adapters into the FIXED ORDERED SANDWICH around the live R-STDP update.
# --------------------------------------------------------------------------------------------------
class HomeostaticPlasticityStack:
    """A fixed ordered sandwich -- (short-term dynamic gain) -> [live R-STDP update] -> (homeostatic
    scaling + intrinsic plasticity) -- run every learning step to STABILIZE the live R-STDP synapses.

    Wire into readout.interaction_learning.SpikingInteractionLearner by passing an instance as `stack=`
    (opt-in; None keeps the learner byte-identical). Each deposit the learner calls, in order:
        d *= stack.pre_gain(neuron_id)        # PRE  short-term dynamic gain
        ... the RSTDPNeuron.reward(d) update happens ...
        n.w = stack.stabilize(neuron_id, n.w, activity)   # POST homeostatic scaling + intrinsic plasticity
    Deterministic, no rng."""

    _UPDATE = "<r_stdp_weight_update>"

    def __init__(
        self,
        n_neurons,
        *,
        short_term=True,
        homeostatic=True,
        intrinsic=True,
        U=0.6,
        tau_f=1.0,
        tau_d=2.0,
        gain_floor=0.5,
        ceiling_factor=3.0,
        scale_rate=0.5,
        mu=0.15,
        ip_lr=0.02,
        seed=0,
    ):
        self.n_neurons = int(n_neurons)
        self.seed = int(seed)
        self._short = (
            _ShortTermGain(self.n_neurons, U=U, tau_f=tau_f, tau_d=tau_d, floor=gain_floor)
            if short_term
            else None
        )
        self._homeo = (
            _HomeostaticScaling(
                self.n_neurons, ceiling_factor=ceiling_factor, scale_rate=scale_rate
            )
            if homeostatic
            else None
        )
        self._intr = _IntrinsicPlasticity(self.n_neurons, mu=mu, lr=ip_lr) if intrinsic else None
        self._pre = [a for a in (self._short,) if a is not None]
        self._post = [a for a in (self._homeo, self._intr) if a is not None]
        # the fixed order the sandwich touches its rules each step (used for the ordering self-check)
        self.order = tuple(
            [a.name for a in self._pre] + [self._UPDATE] + [a.name for a in self._post]
        )
        self.reuses = tuple(
            a.reused for a in (self._short, self._homeo, self._intr) if a is not None
        )
        self._global_t = 0
        self._order_log = []  # one tuple per learning_step (fixed-order proof)

    # -- public API used by SpikingInteractionLearner --------------------------------------------
    def pre_gain(self, neuron_id):
        """PRE: advance the global clock and return the composed short-term dynamic gain for a deposit
        onto `neuron_id` (a scalar in [gain_floor, 1.0] the learner multiplies into the reward sign)."""
        self._global_t += 1
        ctx = {"neuron_id": int(neuron_id), "gain": 1.0, "tick": self._global_t}
        for a in self._pre:
            ctx = a.stabilize(ctx)
        return ctx["gain"]

    def stabilize(self, neuron_id, weights, activity):
        """POST: run the stabilizers (homeostatic scaling, then intrinsic plasticity) in fixed order on
        the just-updated synapses of `neuron_id`. `activity` is the per-synapse drive (PSP, >=0); the
        membrane drive y = sum(weights*activity) is the readout. Returns the stabilized weights."""
        ctx = {
            "neuron_id": int(neuron_id),
            "weights": list(weights),
            "activity": list(activity),
            "gain": 1.0,
            "drive": None,
            "rate": None,
        }
        for a in self._post:
            ctx = a.stabilize(ctx)
        return ctx["weights"]

    # -- full sandwich, for the self-check: threads a real R-STDP update between PRE and POST --------
    def learning_step(self, neuron_id, weights, activity, rstdp_update):
        """Run the WHOLE fixed-ordered sandwich once and log the exact order it touched each rule.
        `rstdp_update(gain, weights) -> weights` is the caller's LIVE R-STDP deposit (w += lr*gain*d*e).
        Returns (stabilized_weights, ctx). Records the order in self._order_log for the ordering proof."""
        self._global_t += 1
        trace = []
        ctx = {
            "neuron_id": int(neuron_id),
            "weights": list(weights),
            "activity": list(activity),
            "gain": 1.0,
            "drive": None,
            "rate": None,
            "tick": self._global_t,
        }
        for a in self._pre:  # PRE: short-term dynamic gain
            ctx = a.stabilize(ctx)
            trace.append(a.name)
        ctx["weights"] = list(
            rstdp_update(ctx["gain"], ctx["weights"])
        )  # LIVE R-STDP weight update
        trace.append(self._UPDATE)
        for a in self._post:  # POST: homeostatic scaling + intrinsic plasticity
            ctx = a.stabilize(ctx)
            trace.append(a.name)
        self._order_log.append(tuple(trace))
        return ctx["weights"], ctx


# --------------------------------------------------------------------------------------------------
# self-check helpers (deterministic, ASCII)
# --------------------------------------------------------------------------------------------------
def _drive_code(n_in):
    """A fixed spike code + its per-synapse PSP drive, matching interaction_learning's readout kernel
    (t_post = n_in, tau = 8): activity[i] = exp(-(t_post - t_i)/tau) >= 0."""
    tau, t_post = 8.0, float(n_in)
    code = list(range(n_in))
    activity = [math.exp(-(t_post - t) / tau) for t in code]
    return code, activity, t_post


def _rstdp_update_fn(neuron, code, t_post):
    """A closure that performs ONE real R-STDP deposit (reward = +1 * gain) on `neuron` and returns its
    weights -- the LIVE update the sandwich wraps. REUSES plasticity.r_stdp.RSTDPNeuron verbatim."""

    def update(gain, weights):
        neuron.w = list(weights)
        neuron.e = [0.0] * neuron.n
        neuron.t_pre = [None] * neuron.n
        neuron.t_post = None
        for i, t in enumerate(code):
            neuron.pre_spike(i, float(t))
        neuron.post_spike(t_post)
        neuron.reward(gain * 1.0)  # potentiating reward, scaled by the short-term gain
        return list(neuron.w)

    return update


def _run_stream(stabilized, steps, seed, n_in=10):
    """Drive `steps` potentiating R-STDP deposits onto one neuron. With stabilized=True the sandwich
    wraps each update; with False it is the raw unstabilized control. Returns per-step (max|w|, drive,
    rate) plus the final weights (for determinism checks)."""
    from snn_data_lake.plasticity.r_stdp import RSTDPNeuron

    code, activity, t_post = _drive_code(n_in)
    neuron = RSTDPNeuron(
        n_in, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=seed
    )
    update = _rstdp_update_fn(neuron, code, t_post)
    stack = HomeostaticPlasticityStack(1, seed=seed) if stabilized else None

    w = list(neuron.w)
    wmax_series, drive_series, rate_series = [], [], []
    for _ in range(steps):
        if stack is not None:
            w, ctx = stack.learning_step(0, w, activity, update)
            drive, rate = ctx["drive"], ctx["rate"]
        else:
            w = update(1.0, w)  # raw R-STDP, no stabilization
            drive = sum(wi * ai for wi, ai in zip(w, activity))
            rate = 1.0 / (
                1.0 + math.exp(-max(-60.0, min(60.0, drive)))
            )  # fixed a=1,b=0 sigmoid (no IP)
        wmax_series.append(max(abs(wi) for wi in w))
        drive_series.append(drive)
        rate_series.append(rate)
    return wmax_series, drive_series, rate_series, list(w), stack


def _stability_and_stack_selfcheck():
    """(a) bounded vs diverging + firing-rate homeostasis, (b) fixed order each step, (c) determinism,
    (d) reuses the real plasticity modules. Returns the headline numbers."""
    steps, seed, mu = 300, 1, 0.15
    s_wmax, s_drive, s_rate, s_w, stack = _run_stream(True, steps, seed)
    c_wmax, c_drive, c_rate, _c_w, _ = _run_stream(False, steps, seed)

    tail = 50
    s_drive_tail = max(s_drive[-tail:])  # stabilized: bounded
    c_drive_final = c_drive[-1]  # control: diverges
    s_rate_mean = sum(s_rate[-tail:]) / tail  # stabilized: -> mu (homeostasis)
    c_rate_mean = sum(c_rate[-tail:]) / tail  # control: saturates ~1.0 (no homeostasis)
    s_wmax_final, c_wmax_final = s_wmax[-1], c_wmax[-1]

    # (b) fixed order: every logged step is exactly (short-term) -> [update] -> (homeostatic, intrinsic)
    order_ok = bool(stack._order_log) and all(step == stack.order for step in stack._order_log)
    # (c) determinism: same seed -> identical stabilized weight trajectory
    s2_wmax, s2_drive, s2_rate, s2_w, _ = _run_stream(True, steps, seed)
    deterministic = s_w == s2_w and s_wmax == s2_wmax and s_drive == s2_drive and s_rate == s2_rate
    # (d) reuse: the adapters wrap the REAL rule classes (named)
    reuse_ok = (
        isinstance(stack._short._syn[0], DynamicSynapse)
        and isinstance(stack._homeo._cell[0], HomeostaticNeuron)
        and isinstance(stack._intr._ip[0], IntrinsicPlasticity)
    )
    # A `"numpy" not in sys.modules` purity report used to be computed and printed here. REMOVED
    # 2026-08-04 under P7, which names stdlib-purity residue as a defect in its own right. It was
    # also not a measurement of this module: `sys.modules` is process-global, so the line printed
    # True standalone and False whenever anything earlier in the same interpreter had imported
    # numpy -- which `run_selfchecks._run_module_inproc` (the frozen-build path) does. It reported
    # the process's import history and labelled it a property of this self-check.
    print("=" * 78)
    print("HOMEOSTATIC PLASTICITY STACK -- stabilizing the live R-STDP learning synapses")
    print("=" * 78)
    print(f"sandwich order each step: {' -> '.join(stack.order)}")
    print(f"rules reused (thin adapters, no rewrite): {', '.join(stack.reuses)}")
    print(f"stream: {steps} potentiating R-STDP deposits onto one neuron (reward +1 each)")
    print("-" * 78)
    print("STABILITY PROOF  (stabilized sandwich vs raw unstabilized R-STDP control):")
    print(f"  max|weight|  final : stabilized {s_wmax_final:8.3f}   control {c_wmax_final:12.3f}")
    print(
        f"  drive (v)          : stabilized {s_drive_tail:8.3f} (bounded)   control {c_drive_final:12.3f} (diverges)"
    )
    print(
        f"  firing rate -> mu  : stabilized {s_rate_mean:8.3f} (~mu={mu})   control {c_rate_mean:12.3f} (saturated)"
    )
    print(
        f"  fixed order every step: {order_ok}   deterministic: {deterministic}   "
        f"reuses real modules: {reuse_ok}"
    )
    print("=" * 78)

    # ---- asserts ----
    assert c_drive_final > 3.0 * s_drive_tail, (
        f"(a) unstabilized control drive must DIVERGE far past the stabilized ceiling ({c_drive_final:.2f} vs {s_drive_tail:.2f})"
    )
    assert s_drive_tail < 20.0 and s_wmax_final < 5.0, (
        f"(a) the stabilized synapses must stay BOUNDED (drive {s_drive_tail:.2f}, max|w| {s_wmax_final:.2f})"
    )
    assert abs(s_rate_mean - mu) < 0.08, (
        f"(a) intrinsic plasticity must hold the output firing rate at the set-point ({s_rate_mean:.3f} vs mu={mu})"
    )
    assert c_rate_mean > 0.9, (
        f"(a) the unstabilized control rate must saturate (no homeostasis) ({c_rate_mean:.3f})"
    )
    assert order_ok, "(b) every learning step must run the rules in the fixed sandwich order"
    assert deterministic, "(c) same seed must give a byte-identical trajectory"
    assert reuse_ok, (
        "(d) the stack must REUSE the real DynamicSynapse / HomeostaticNeuron / IntrinsicPlasticity"
    )
    return s_drive_tail, c_drive_final, s_rate_mean, c_rate_mean


def _learner_off_on_selfcheck():
    """Wire check: SpikingInteractionLearner with the stack OFF is byte-identical to today (the real
    _spiking_task_selfcheck still passes), and with the stack ON the held-out routing accuracy does NOT
    degrade. Runs the same task/schedule both ways and prints off-vs-on."""
    import random

    from snn_data_lake.readout.interaction_learning import (
        SpikingInteractionLearner,
        _make_task,
        _spiking_task_selfcheck,
    )

    # OFF byte-identical to today: the unmodified default-OFF self-check must still pass end-to-end.
    off_chance, off_learned, off_control = _spiking_task_selfcheck()

    # Same task + fixed schedule as _spiking_task_selfcheck, trained OFF vs ON on identical inputs.
    seed = 7
    responses = ["reach", "grasp", "retract", "hold"]
    n_in, epochs = 16, 40
    train, test = _make_task(seed, responses, n_in, n_train=8, n_test=8, noise=0.06)
    sched = random.Random(seed + 1)
    epoch_orders = []
    for _ in range(epochs):
        o = list(range(len(train)))
        sched.shuffle(o)
        epoch_orders.append(o)
    targets = [correct for _stim, correct in train]

    def _train_with(stack):
        lrn = SpikingInteractionLearner(responses, n_in, seed=seed, stack=stack)
        for order in epoch_orders:
            for idx in order:
                lrn.interact(train[idx][0], targets[idx])
        return lrn

    off = _train_with(None)
    on = _train_with(HomeostaticPlasticityStack(len(responses), seed=seed))
    off_acc, on_acc = off.accuracy(test), on.accuracy(test)
    off_bytes_identical = (
        off.weights() == _train_with(None).weights()
    )  # OFF is deterministic (no stack)

    print()
    print("=" * 78)
    print("WIRE CHECK -- SpikingInteractionLearner with the homeostatic plasticity stack OFF vs ON")
    print("=" * 78)
    print(
        f"default-OFF _spiking_task_selfcheck (today's behaviour): chance {off_chance:.2f} -> "
        f"learned {off_learned:.2f} (control {off_control:.2f}) -- still green, OFF byte-identical"
    )
    print(
        f"same task, held-out routing accuracy: stack OFF {off_acc:.2f}  ->  stack ON {on_acc:.2f}"
    )
    print(f"OFF run is byte-identical run-to-run: {off_bytes_identical}")
    print("=" * 78)

    assert off_bytes_identical, "OFF path must be deterministic / byte-identical to today"
    assert on_acc >= off_acc - 0.05, (
        f"stack ON must NOT degrade routing accuracy ({on_acc:.2f} vs OFF {off_acc:.2f})"
    )
    assert on_acc >= 0.80, f"stack ON must still learn the task well ({on_acc:.2f})"
    return off_acc, on_acc


def main():
    s_drive, c_drive, s_rate, c_rate = _stability_and_stack_selfcheck()
    off_acc, on_acc = _learner_off_on_selfcheck()
    print()
    print(
        f"self-check OK: a fixed ordered sandwich (short-term dynamic gain -> [live R-STDP update] -> "
        f"homeostatic scaling + intrinsic plasticity), REUSING DynamicSynapse / HomeostaticNeuron / "
        f"IntrinsicPlasticity verbatim, keeps the live learning synapses BOUNDED (drive {s_drive:.2f} vs "
        f"an unstabilized control's diverging {c_drive:.1f}) and holds the firing rate at its set-point "
        f"(rate {s_rate:.2f}~mu vs saturated {c_rate:.2f}); the rules run in the fixed order every step; "
        f"deterministic. Wired into SpikingInteractionLearner as an opt-in "
        f"stack: OFF is byte-identical to today, ON does not degrade routing ({off_acc:.2f} -> {on_acc:.2f})."
    )


if __name__ == "__main__":
    main()
