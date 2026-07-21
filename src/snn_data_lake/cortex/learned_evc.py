r"""
Learned expected-value-of-control (EVC) -- a PFC/ACC control-allocation unit whose
cost-of-control and value-of-control are LEARNED online, instead of the FIXED cost=0.1 and
fixed value gain baked into cortex/executive_control.py's evc().

Why this exists. ExecutiveControl.evc(conflict, value_gain, cost) arbitrates HOW MUCH top-down
control to exert as benefit(control) - cost(control), but both terms are DEVELOPER CONSTANTS: the
cost of engaging control is hard-wired (the autonomous loop passes cost=0.1 at every call) and the
value gain is a fixed function of the decision margin. A real prefrontal-cingulate controller does
not know its control cost a priori -- it DISCOVERS, from outcomes, whether engaging control tends to
pay off (raise the value-of-control) or tends to be wasted effort (raise the cost-of-control). This
module makes those two scalars LEARNED, driven by the same dopaminergic reward-prediction-error the
rest of the brain-stack already uses.

Mechanism (brain-native, spiking, three-factor local -- NO backprop):

  1. ALLOCATE (a spiking gain unit). allocate(conflict, stakes) forms the net expected value of
     control from the CURRENT learned weights,
         net = value * (conflict * stakes) - cost,
     rectifies it (control is WITHHELD when net <= 0), and reads it out as the normalized spike RATE
     of a small leaky integrate-and-fire (LIF) population whose drive is that net level. The
     population rate IS the allocated control gain in [0, 1] -- a genuinely spiking readout (the same
     LIF-pool convention as midbrain/dopamine_nuclei), not a bare arithmetic clamp. Higher
     conflict*stakes -> higher net -> higher rate -> more control.

  2. LEARN (dopamine-gated three-factor plasticity). learn(engaged, reward) computes the reward
     PREDICTION ERROR against a running baseline of the DEFAULT (control-withheld) outcome,
         dopamine = reward - expected                    (the RPE, same convention as
                                                          basal_ganglia/valence_stdp and
                                                          midbrain/dopamine_nuclei)
     and applies a THREE-FACTOR update gated by whether control was actually engaged (the eligibility
     of the control unit -- only an engaged, i.e. FIRING, control unit is eligible to change its
     weights):
         value += lr * dopamine       * engaged         (engaging control that BEAT the default
                                                          raises the value-of-control)
         cost  += lr * (-dopamine)    * engaged         (engaging control that UNDERperformed the
                                                          default raises the cost-of-control:
                                                          wasted control is penalized)
     The baseline is then nudged toward the outcome by TD(0) on every trial (engaged OR withheld), so
     it tracks the default expectation and the RPE SHRINKS as the value is learned -- Schultz's
     dopamine-fires-on-surprise. Because only engaged trials carry eligibility, the learned weights
     encode the CONTRAST between control-engaged and control-withheld outcomes: engaging that reliably
     beats withholding pushes allocation UP; engaging that reliably wastes effort pushes it DOWN.

Learning rule: local three-factor (dopamine * eligibility) only. No gradient, no loss.backward(),
no trained parameter tensor -- two scalar weights (value, cost) plus a scalar reward baseline,
updated by the RPE. This satisfies the brain-native constraint (fixed dynamics + local
neuromodulated plasticity over a gradient).

Honest framing / gaps. cost and value are two learned SCALARS, not a per-context table -- this learns
a GLOBAL control price, not a state-specific one; a context-conditioned cost would need a feature
vector and per-feature eligibility (as valence_stdp already does for valence). The RPE is against a
single running baseline, a reduced TD(0) critic, not a full state-value model. The spiking gain unit
is a reduced LIF rate readout, not a calibrated cortical microcircuit. This is control-ALLOCATION
machinery -- it emits a scalar gain and selects no action, drives no hardware; the basal-ganglia gate
+ reflex/E-STOP still own selection and actuation. It is not cognition, agency, or consciousness.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.learned_evc
"""

import random
import zlib


class LearnedEVC:
    """Learned expected-value-of-control: a spiking control-gain unit with dopamine-learned
    cost-of-control and value-of-control.

    allocate(conflict, stakes) -> control_gain in [0, 1] : the spiking-pool rate for the current net
        EVC (value*conflict*stakes - cost), rectified (0 when control is not worth it).
    learn(engaged, reward) : dopamine-gated three-factor update of the learned value/cost from the
        reward-prediction error of an engaged (control-active) vs withheld outcome. Returns the RPE.
    cost  (property) : the learned cost-of-control (rises when engaging control is wasted).
    value (property) : the learned value-of-control (rises when engaging control pays off).

    Representation/value-side only: emits a scalar gain, selects no action, drives no hardware.
    Deterministic: per-instance random.Random(seed); the LIF readout uses a fixed crc32-derived seed
    so the gain is a pure function of the net level (no global seeding anywhere)."""

    def __init__(
        self,
        seed=0,
        lr=0.12,
        alpha=0.10,
        value0=1.0,
        cost0=0.25,
        value_max=3.0,
        cost_max=2.0,
        n_cells=40,
        steps=60,
        leak=0.25,
        noise=0.006,
    ):
        self._seed = int(seed)
        self.rng = random.Random(seed)  # per-instance rng (thresholds); no global seeding
        self.lr = float(lr)  # three-factor plasticity rate of value / cost
        self.alpha = float(alpha)  # TD(0) rate of the reward baseline (the withheld/default critic)
        self._value = float(value0)  # learned value-of-control weight
        self._cost = float(cost0)  # learned cost-of-control weight
        self._expected = 0.0  # running baseline of the DEFAULT outcome (RPE reference)
        self._last_gain = 0.0  # most recent allocated control gain (for state()/telemetry)
        self.value_max = float(value_max)  # keep the learned weights in a sane bounded range
        self.cost_max = float(cost_max)
        # spiking gain unit: a small heterogeneous LIF population (dopamine_nuclei convention).
        self.n_cells = int(n_cells)
        self.steps = int(steps)
        self.leak = float(leak)
        self.noise = float(noise)
        # thresholds spread the per-neuron firing ONSET across the net-EVC range -> a graded,
        # monotone population rate (a pure winner would saturate a single-threshold unit).
        self.thresholds = [self.rng.uniform(0.2, 3.0) for _ in range(self.n_cells)]
        # fixed readout seed (crc32, never builtin hash) -> the rate is a reproducible fn of level.
        self.readout_seed = zlib.crc32(f"evc-readout-{self._seed}".encode())

    # ---- learned scalars (the whole point: NOT the fixed 0.1 / fixed gain) -----------------------
    @property
    def cost(self):
        """The learned cost-of-control (rises as engaging control proves wasteful)."""
        return self._cost

    @property
    def value(self):
        """The learned value-of-control (rises as engaging control proves to pay off)."""
        return self._value

    # ---- spiking readout -------------------------------------------------------------------------
    def _population_gain(self, level):
        """Normalized spike rate in [0, 1] of the LIF control pool for a constant net-EVC `level`.

        Deterministic readout (fixed crc32 seed) so the gain is a pure, reproducible, monotone
        function of the level: ~0 for a withheld (level 0) allocation, rising toward 1 as the net
        expected value of control grows."""
        rng = random.Random(self.readout_seed)
        spikes = 0
        for thresh in self.thresholds:
            v = 0.0
            for _ in range(self.steps):
                drive = level + rng.uniform(-self.noise, self.noise)
                v = v * (1.0 - self.leak) + drive
                if v >= thresh:
                    spikes += 1
                    v = 0.0
        return spikes / (self.n_cells * self.steps)

    def _net_evc(self, conflict, stakes):
        """Net expected value of control from the CURRENT learned weights: benefit - cost, where
        benefit = value * conflict * stakes (control only pays off with conflict to resolve AND value
        at stake). Not rectified here -- allocate() rectifies (withholds control when net <= 0)."""
        benefit = self._value * float(conflict) * float(stakes)
        return benefit - self._cost

    def allocate(self, conflict, stakes):
        """Allocated top-down control gain in [0, 1] for the current conflict and stakes, read out as
        the spiking-pool rate of the (rectified) net expected value of control. Control is WITHHELD
        (gain ~0) when the learned benefit does not clear the learned cost."""
        net = self._net_evc(conflict, stakes)
        level = net if net > 0.0 else 0.0
        if level > 1.0:
            level = 1.0  # LIF drive is a [0, 1] level (saturating control)
        gain = self._population_gain(level)
        self._last_gain = gain
        return gain

    # ---- learning --------------------------------------------------------------------------------
    def learn(self, engaged, reward):
        """Dopamine-gated three-factor update from the reward-prediction error.

        dopamine = reward - expected (RPE vs the running default/withheld baseline). Gated by
        `engaged` (the control unit's eligibility -- only a FIRING, i.e. engaged, control unit is
        eligible): engaging control that BEAT the default raises value and lowers cost; engaging that
        UNDERperformed raises cost and lowers value (wasted control penalized). The baseline is then
        nudged toward the outcome by TD(0) on every trial, so the RPE shrinks as the value is learned.
        Returns the dopamine (RPE) signal."""
        reward = float(reward)
        dopamine = reward - self._expected  # reward-prediction error (the phasic dopamine)
        elig = 1.0 if engaged else 0.0  # three-factor eligibility: control was actually engaged
        self._value += self.lr * dopamine * elig
        self._cost += self.lr * (-dopamine) * elig
        # keep the learned weights non-negative and bounded (a cost/value cannot be negative)
        self._value = min(self.value_max, max(0.0, self._value))
        self._cost = min(self.cost_max, max(0.0, self._cost))
        # TD(0) on the default-outcome baseline (every trial, engaged or withheld) -> RPE shrinks
        self._expected += self.alpha * dopamine
        return dopamine

    # ---- telemetry / determinism -----------------------------------------------------------------
    def state(self):
        """Learned-controller snapshot. The `cost` and `value` keys expose the LEARNED
        cost-of-control / value-of-control that a caller uses in place of the fixed 0.1 / fixed gain."""
        return {
            "cost": round(self._cost, 4),
            "value": round(self._value, 4),
            "expected": round(self._expected, 4),
            "last_gain": round(self._last_gain, 4),
        }

    def signature(self):
        """crc32 of the learned scalars (never builtin hash) -> a stable determinism fingerprint."""
        payload = f"{self._value:.6f}|{self._cost:.6f}|{self._expected:.6f}"
        return zlib.crc32(payload.encode())


# ==================================================================================================
# Un-riggable self-check. Deterministic (seeded), ASCII-only. Local three-factor plasticity only;
# no backprop. Every learned claim is paired with a control regime that moves the OPPOSITE way.
# ==================================================================================================

# a "high-conflict, real-stakes" probe state we track allocation for, pre vs post learning.
HI_CONFLICT, HI_STAKES = 0.8, 0.6


def _run_regime(evc, engaged_reward, withheld_reward, n):
    """Alternate a control-WITHHELD outcome (default, reward=withheld_reward) with a control-ENGAGED
    outcome (reward=engaged_reward). The learned value/cost move only on the engaged trials (three-
    factor eligibility); the baseline tracks the running default -- so the weights encode whether
    engaging BEATS or WASTES relative to withholding."""
    for _ in range(n):
        evc.learn(engaged=False, reward=withheld_reward)  # the default outcome sets the baseline
        evc.learn(engaged=True, reward=engaged_reward)  # engaging control: paid off or wasted


def main():
    n = 40

    # ---- (a) engaging control reliably PAYS OFF -> allocation for high-conflict states RISES -------
    pay = LearnedEVC(seed=0)
    pre_pay = pay.allocate(HI_CONFLICT, HI_STAKES)
    cost_pay0, value_pay0 = pay.cost, pay.value
    _run_regime(pay, engaged_reward=1.0, withheld_reward=0.0, n=n)  # engage beats withhold
    post_pay = pay.allocate(HI_CONFLICT, HI_STAKES)

    # ---- (b) engaging control is WASTEFUL (cost > benefit) -> allocation LOWERS --------------------
    waste = LearnedEVC(seed=0)
    pre_waste = waste.allocate(HI_CONFLICT, HI_STAKES)
    _run_regime(waste, engaged_reward=0.0, withheld_reward=1.0, n=n)  # engage wastes vs withhold
    post_waste = waste.allocate(HI_CONFLICT, HI_STAKES)

    # ---- (c) allocate() rises with conflict*stakes ------------------------------------------------
    ramp_evc = LearnedEVC(seed=0)
    ramp = [ramp_evc.allocate(c, 1.0) for c in (0.3, 0.5, 0.7, 0.9)]  # increasing conflict*stakes
    withheld = ramp_evc.allocate(0.05, 0.05)  # benefit << cost -> control withheld (~0)

    # ---- (d) determinism --------------------------------------------------------------------------
    a = LearnedEVC(seed=7)
    b = LearnedEVC(seed=7)
    _run_regime(a, 1.0, 0.0, n=12)
    _run_regime(b, 1.0, 0.0, n=12)
    det_alloc = a.allocate(HI_CONFLICT, HI_STAKES) == b.allocate(HI_CONFLICT, HI_STAKES)
    det_sig = a.signature() == b.signature()

    print("=" * 84)
    print("LEARNED EVC -- dopamine-gated three-factor learned cost/value of control (spiking gain)")
    print("=" * 84)
    print(
        f"probe state  : conflict {HI_CONFLICT}, stakes {HI_STAKES}  (allocation tracked pre vs post)"
    )
    print("-" * 84)
    print(
        f"(a) PAYS OFF : value {value_pay0:.2f}->{pay.value:.2f}  cost {cost_pay0:.2f}->{pay.cost:.2f}  "
        f"|  allocation {pre_pay:.3f} -> {post_pay:.3f}  (RISES)"
    )
    print(
        f"(b) WASTEFUL : value {value_pay0:.2f}->{waste.value:.2f}  cost {cost_pay0:.2f}->{waste.cost:.2f}  "
        f"|  allocation {pre_waste:.3f} -> {post_waste:.3f}  (LOWERS)"
    )
    print(
        f"(c) conflict ramp (stakes 1.0): {[round(x, 3) for x in ramp]}  withheld -> {withheld:.3f}"
    )
    print(f"(d) deterministic across seeds: allocate {det_alloc}, signature {det_sig}")
    print("=" * 84)

    # ---- un-riggable asserts ---------------------------------------------------------------------
    # (a) engaging control that pays off RAISES the learned value and the allocated control
    assert post_pay > pre_pay, (
        f"paying-off control must RAISE high-conflict allocation ({pre_pay:.3f} -> {post_pay:.3f})"
    )
    assert pay.value > value_pay0, "paying-off control must raise the learned value-of-control"
    assert pay.cost <= cost_pay0, "paying-off control must not raise the learned cost-of-control"

    # (b) wasteful control LOWERS the allocation (opposite regime, same probe)
    assert post_waste < pre_waste, (
        f"wasteful control must LOWER high-conflict allocation ({pre_waste:.3f} -> {post_waste:.3f})"
    )
    assert waste.cost > cost_pay0, "wasteful control must raise the learned cost-of-control"
    assert waste.value < value_pay0, "wasteful control must lower the learned value-of-control"
    assert post_pay > post_waste, "paying-off must allocate more control than wasteful (same probe)"

    # (c) allocation rises monotonically with conflict*stakes; withheld when benefit < cost
    assert ramp == sorted(ramp) and ramp[-1] > ramp[0], (
        f"allocation must rise with conflict*stakes ({[round(x, 3) for x in ramp]})"
    )
    assert withheld < ramp[-1] and withheld < 0.05, (
        "control must be WITHHELD (~0) when not worth it"
    )

    # (d) determinism
    assert det_alloc, "same seed must reproduce the allocated control gain"
    assert det_sig, "same seed must reproduce the learned-weight signature"

    print(
        f"self-check OK: pays-off RAISES allocation {pre_pay:.3f}->{post_pay:.3f} "
        f"(value {value_pay0:.2f}->{pay.value:.2f}, cost {cost_pay0:.2f}->{pay.cost:.2f}); "
        f"wasteful LOWERS it {pre_waste:.3f}->{post_waste:.3f} (cost ->{waste.cost:.2f}); "
        f"rises with conflict*stakes; withheld {withheld:.3f}; three-factor local, no backprop; deterministic"
    )


if __name__ == "__main__":
    main()
