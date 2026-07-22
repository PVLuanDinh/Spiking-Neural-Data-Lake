"""
Spiking task-set / set-shifting layer -- an executive TASK-SET (attentional-set) faculty.

The prefrontal executive (prefrontal_executive.py) holds a GOAL; this module holds a RULE. A
task set is a whole input->action MAPPING, not a single option: which set is active decides HOW
inputs are read out into actions, not just which action wins. The brain-atlas already has goal
working memory (PFC), a Go/NoGo action-selection circuit (basal_ganglia.action_gating), and
delayed-reward plasticity (basal_ganglia.r_stdp) -- but nothing that holds an attentional SET and
re-maps behaviour once the current set stops working (Wisconsin-Card-Sort-style set-shifting).
This module is that faculty.

Mechanism (spiking, deterministic, per-instance rng):
  1. ACTIVE SET = persistent-activity spiking working memory: a bistable self-exciting LIF pool,
     the SAME leaky-integrate-and-fire equation PrefrontalExecutive._run_wm_pool uses for its
     goal-holding attractor (leak -> integrate a constant recurrent current -> threshold -> spike
     -> reset). step() advances this pool one delay tick with NO external input; a low-pass trace
     of its firing rate is the set's settle CONFIDENCE. On a fresh commit (construction, or right
     after a shift) the pool's membranes are cold (V=0), so the confidence trace ramps up from 0
     over several step() ticks before reaching its held asymptote -- the same LIF mechanism the
     executive already uses for persistence, here also producing a natural settle transient.
  2. INPUT -> ACTION MAPPING. Each of the n_sets candidate task sets owns a fixed input->action
     weight matrix (its "rule"). bias(input) reads the ACTIVE set's matrix against `input` and
     scales the result by that set's current settle confidence -- so preferences are strong once
     the set has held for a while and are compressed toward zero right after a shift, before the
     LIF pool's confidence trace has ramped back up. That compression on the mapping readout IS
     the switch cost this module is built to show (the class has no external "correct action"
     ground truth to score against, so the compression of the mapping's own output is the honest,
     internal proxy for it).
  3. SHIFT TRIGGER = an ACC-style conflict/reward monitor gates a THREE-FACTOR R-STDP vote over
     which alternate set to shift into, reusing basal_ganglia.r_stdp.RSTDPNeuron directly (one
     synapse per candidate set, not reimplemented). The set is "failing" on a tick when reward is
     net-negative (reward <= reward_thresh -- the reward SIGN, a scale-natural zero, not a tuned
     knob) AND conflict is high RELATIVE TO a LEARNED baseline: the running mean of the conflict seen
     during GOOD (rewarded) ticks. That baseline is learned from the module's own experience, never a
     magic constant; during a sustained bad patch no good ticks arrive, so it FREEZES at the good-time
     level and conflict stays anomalous -- the trigger keeps firing (it does not habituate to a
     persistently-bad set, which is exactly what set-shifting needs). Before any good tick has been
     seen the reward-sign arm alone decides. While the trigger holds: every candidate (non-active)
     set's synapse gets a pre-synaptic probe and the ACC's own error crossing is the post-synaptic
     pairing (STDP deposits a pre-before-post eligibility on the candidates); an RPE-flavoured
     "dopamine" magnitude -- how far conflict/reward sit past their references (the learned good-time
     conflict baseline, and the reward sign), the same reward-prediction-error convention
     valence_stdp.py uses (`reward - predicted`) -- gates that eligibility into weight via
     RSTDPNeuron.reward(). Once this persists `patience` ticks in a row, the highest-weighted
     candidate WINS and shift() commits it: a genuinely reward-gated, spike-timing-driven re-mapping,
     not a scripted counter. With only one task set there is no candidate to shift to, so it is held.
  4. SWITCH COST. shift() snapshots the outgoing set's settled confidence just before the cold
     reset and reports it as last_switch_cost (how much settle confidence was just given up); the
     same drop is visible live in bias()'s compressed preferences on the very next call, and
     recovers over the following step() ticks as the new set's confidence trace ramps back up.

Fair control (un-riggable): with n_sets=1 there are no candidates, so shift() is never reachable
and the active set is held regardless of how bad conflict/reward get (the self-check drives it
with relentlessly bad inputs and confirms it never moves). With n_sets>1, sustained bad inputs
DO move it, and the bias() mapping's own output magnitude measurably drops the instant it moves
and climbs back as the new set's LIF pool re-settles.

Honest framing: fixed input->action rule matrices behind one persistent-activity LIF selector,
re-mapped by a reward/conflict-gated R-STDP vote. It is a genuinely spiking model of attentional
SET (which mapping is active) and SET-SHIFTING (switching mapping under sustained conflict/low
reward) with a measured switch cost -- not executive control, insight, or task understanding.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.task_set
"""

import random

from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron


class _RunningMean:
    """Running mean of a scalar (incremental count+mean). mean() returns None until the first sample
    so a caller can tell 'no baseline yet' from a real 0.0. Baseline-relative form, kept inline so
    this module depends only on r_stdp (no cross-region import for a six-line primitive)."""

    __slots__ = ("_n", "_mean")

    def __init__(self):
        self._n = 0
        self._mean = 0.0

    def observe(self, x):
        self._n += 1
        self._mean += (x - self._mean) / self._n

    def mean(self):
        return self._mean if self._n else None


class TaskSetLayer:
    """n_sets candidate input->action rule mappings; one persistent-activity LIF pool holds the
    ACTIVE set (working memory); bias() reads the active mapping scaled by its settle confidence;
    observe() runs an ACC-style conflict/reward monitor that, via three-factor R-STDP, gates a
    SHIFT to a different set once the current one stops working."""

    def __init__(
        self,
        n_sets=3,
        n_in=4,
        n_actions=4,
        seed=0,
        reward_thresh=0.0,
        patience=6,
        leak=0.2,
        wm_steps=8,
        trace_decay=0.85,
        rstdp_lr=0.15,
    ):
        self.n_sets = n_sets
        self.n_in = n_in
        self.n_actions = n_actions
        self.reward_thresh = (
            reward_thresh  # ACC monitor: reward at/below this is "bad" (reward SIGN)
        )
        self.patience = patience  # consecutive bad ticks needed to commit a shift
        # conflict "high" is judged RELATIVE to the running mean of conflict during GOOD ticks -- a
        # learned baseline, not a fixed cutoff (freezes during sustained-bad -> no habituation).
        self._good_conflict = _RunningMean()
        self.rng = random.Random(seed)

        # ---- fixed per-set input -> action rule matrices (the mapping each task set encodes) ---
        self._W = [
            [[self.rng.uniform(-1.0, 1.0) for _ in range(n_in)] for _ in range(n_actions)]
            for _ in range(n_sets)
        ]

        # ---- persistent-activity LIF pool holding the ACTIVE set (PrefrontalExecutive's
        # _run_wm_pool equation: leak -> integrate a constant recurrent current -> threshold ->
        # spike -> reset). Held at a constant recurrent drive for as long as it stays active;
        # only an explicit shift() cold-resets it (there is always exactly one active set here,
        # unlike the PFC's goal stack, so there is no separate decay-to-released state).
        self._n_wm = 6
        md = 1.0 - leak
        self._md = 0.5 if md < 0.5 else (0.95 if md > 0.95 else md)  # membrane decay (from leak)
        self._wm_gain = 0.55  # recurrent drive -> input current gain
        self._wm_steps = wm_steps  # LIF micro-steps advanced per step() tick
        self._trace_decay = trace_decay  # confidence low-pass smoothing (the settle time-constant)
        het = random.Random((seed * 2654435761 + 7) & 0xFFFFFFFF)
        self._wm_bias = [
            0.03 + 0.04 * het.random() for _ in range(self._n_wm)
        ]  # per-neuron heterogeneity

        self._active = 0
        self._Vs = [0.0] * self._n_wm  # active set's LIF membranes
        self._confidence = 0.0  # low-pass firing-rate trace (0..~1); the settle readout

        # ---- ACC conflict/reward monitor -> three-factor R-STDP shift vote (reused verbatim) ---
        self._shifter = RSTDPNeuron(n_sets, lr=rstdp_lr, seed=seed + 1)
        self._bad_streak = 0
        self._t = 0

        self.last_switch_cost = 0.0

    @property
    def active_set(self):
        """The currently held (active) task-set index."""
        return self._active

    def bias(self, input_vec):
        """Read the ACTIVE set's rule matrix against `input_vec`, scaled by that set's current
        settle confidence (near 0 right after a shift, ramping toward its held asymptote as the
        LIF pool re-settles). Returns a length-n_actions preference vector."""
        conf = self._confidence
        return [conf * sum(w * x for w, x in zip(row, input_vec)) for row in self._W[self._active]]

    def step(self):
        """Advance the active set's working-memory LIF pool one delay tick with NO external
        input -- only its recurrent self-excitation drive (PrefrontalExecutive._run_wm_pool's
        equation). Updates and returns the settle-confidence trace."""
        cur = self._wm_gain  # constant recurrent drive while a set is held
        spikes = 0
        for _ in range(self._wm_steps):
            for k in range(self._n_wm):
                v = self._md * self._Vs[k] + cur + self._wm_bias[k]  # LIF: leak + integrate
                if v >= 1.0:  # threshold -> spike -> reset
                    spikes += 1
                    v = 0.0
                self._Vs[k] = v
        rate = spikes / (self._n_wm * self._wm_steps)
        td = self._trace_decay
        self._confidence = td * self._confidence + (1.0 - td) * rate
        return self._confidence

    def observe(self, conflict, reward):
        """One ACC-monitor tick. The set is "failing" when reward is net-negative (reward <=
        reward_thresh, the reward sign) AND conflict is high relative to the LEARNED good-time
        conflict baseline (the running mean of conflict during rewarded ticks; before any good tick
        the reward-sign arm alone decides, and during a sustained bad patch the baseline freezes so
        the trigger keeps firing without habituating). While failing, every non-active (candidate)
        set is STDP-paired against the ACC's own error crossing and gated by an RPE-style dopamine
        magnitude into basal_ganglia.r_stdp.RSTDPNeuron's per-set weight. Once this persists
        `patience` ticks in a row, the highest-weighted candidate set commits via shift(). Returns
        active_set."""
        self._t += 1
        if reward > self.reward_thresh:  # a GOOD tick: learn the typical good-time conflict level
            self._good_conflict.observe(conflict)
        cref = self._good_conflict.mean()  # None until the first good tick has been seen
        conflict_high = cref is None or conflict >= cref
        bad = reward <= self.reward_thresh and conflict_high
        if bad:
            for s in range(self.n_sets):
                if s != self._active:
                    self._shifter.pre_spike(s, self._t)  # candidate pathway probed
            self._shifter.post_spike(self._t)  # the ACC's own error = the post pairing
            # RPE-style: how far conflict/reward sit past their references (the learned good-time
            # conflict baseline -- 0.0 while none exists yet -- and the reward sign).
            dopamine = (conflict - (cref or 0.0)) + (self.reward_thresh - reward)
            self._shifter.reward(dopamine if dopamine > 0.0 else 0.05)
            self._bad_streak += 1
        else:
            self._bad_streak = 0
        self._shifter.step()  # eligibility decays every tick regardless
        if self._bad_streak >= self.patience:
            candidates = [s for s in range(self.n_sets) if s != self._active]
            if candidates:  # n_sets == 1 -> nothing to shift to, held
                new_set = max(candidates, key=lambda s: self._shifter.w[s])
                self.shift(new_set)
            self._bad_streak = 0
        return self._active

    def shift(self, new_set):
        """Commit a SET SHIFT: snapshot the outgoing set's settled confidence as the switch cost,
        then cold-reset the LIF pool onto `new_set` (membranes back to 0, confidence back to 0 --
        it has to re-settle from scratch, which produces the transient dip visible in bias()'s
        next reading). Returns the new active_set."""
        if new_set == self._active:
            return self._active
        self.last_switch_cost = self._confidence
        self._active = new_set
        self._Vs = [0.0] * self._n_wm
        self._confidence = 0.0
        return self._active


def _mag(vec):
    """Sum of absolute values -- a simple readout-strength proxy for a preference vector."""
    return sum(abs(v) for v in vec)


def main():

    x = [0.6, -0.4, 0.3, 0.2]

    # ---- (a) one task set: nothing to shift to -> the mapping settles and stays held -----------
    single = TaskSetLayer(n_sets=1, n_in=4, n_actions=3, seed=0)
    conf_trace = []
    for _ in range(30):
        single.step()
        single.observe(conflict=0.95, reward=-1.0)  # relentlessly bad, but no alternative set
        conf_trace.append(single._confidence)
    settle_spread = max(conf_trace[-5:]) - min(conf_trace[-5:])
    settled_mag = _mag(single.bias(x))

    # ---- (b) 3 task sets: sustained high conflict + low reward triggers a SHIFT ------------------
    multi = TaskSetLayer(n_sets=3, n_in=4, n_actions=3, seed=1, patience=6)
    start_set = multi.active_set
    pre_shift_mag = 0.0
    shifted_at = None
    for t in range(40):
        multi.step()
        if multi.active_set == start_set:
            pre_shift_mag = _mag(multi.bias(x))  # last reading before the shift commits
        multi.observe(conflict=0.9, reward=-0.6)
        if multi.active_set != start_set:
            shifted_at = t
            break
    shifted_set = multi.active_set
    post_shift_mag = _mag(multi.bias(x))  # cold, right after the shift commits

    # ---- (c) switch cost: the transient drop, then recovery as the new set re-settles -----------
    switch_cost = multi.last_switch_cost
    recovery = []
    for _ in range(25):
        multi.step()
        recovery.append(_mag(multi.bias(x)))
    recovered_mag = recovery[-1]

    # ---- (d) the conflict trigger is BASELINE-RELATIVE, not a fixed cutoff -----------------------
    # SAME test conflict (0.7) + SAME bad reward, but different LEARNED good-time baselines: it must
    # shift when the good-time conflict was LOW (0.7 is anomalous) and HOLD when it was HIGH (0.7 is
    # normal). A fixed 0.6 cutoff would shift in BOTH -- this is the control a constant fails.
    def _relative(good_conflict):
        layer = TaskSetLayer(n_sets=3, n_in=4, n_actions=3, seed=5, patience=3)
        for _ in range(12):
            layer.step()
            layer.observe(conflict=good_conflict, reward=0.5)  # GOOD ticks establish the baseline
        s0 = layer.active_set
        shifted = False
        for _ in range(8):
            layer.step()
            layer.observe(conflict=0.7, reward=-0.4)  # SAME test conflict, now bad reward
            if layer.active_set != s0:
                shifted = True  # a shift fired at any point (it may later oscillate back)
        return shifted

    shift_when_baseline_low = _relative(0.3)  # 0.7 >> 0.3 good-time norm -> anomalous -> shift
    hold_when_baseline_high = not _relative(0.9)  # 0.7 < 0.9 good-time norm -> normal -> held

    print("=" * 72)
    print("SPIKING TASK-SET / SET-SHIFTING LAYER (LIF working memory + R-STDP-gated shift)")
    print("=" * 72)
    print(
        f"(a) single set : held={single.active_set} over 30 relentlessly-bad ticks; confidence "
        f"settle spread (last 5) {settle_spread:.5f}; settled mapping mag {settled_mag:.3f}"
    )
    print(
        f"(b) 3 sets     : start {start_set} -> sustained conflict=0.9/reward=-0.6 -> shifted to "
        f"{shifted_set} at tick {shifted_at} (patience={multi.patience})"
    )
    print(
        f"(c) switch cost: pre-shift mapping mag {pre_shift_mag:.3f} -> post-shift {post_shift_mag:.3f} "
        f"(cost {switch_cost:.3f}) -> recovered over 25 ticks to {recovered_mag:.3f}"
    )
    print(
        f"(d) relative   : conflict 0.7 + bad reward SHIFTS vs a 0.3 good-time baseline "
        f"({shift_when_baseline_low}) but HOLDS vs a 0.9 baseline ({hold_when_baseline_high}) "
        f"-- a fixed cutoff cannot do both"
    )
    print("=" * 72)

    # ---- un-riggable self-checks (deterministic asserts) ----------------------------------------

    # (a) one set: no candidate to shift to -> held; and the confidence trace genuinely settles.
    assert single.active_set == 0, "with a single task set there is nothing to shift to"
    assert settle_spread < 0.02, (
        f"a held set's settle confidence must stop moving ({settle_spread:.5f})"
    )
    assert settled_mag > 0.0, "a settled single task set must produce nonzero mapping preferences"

    # (b) sustained high conflict + low reward must trigger a SHIFT of the active set.
    assert shifted_at is not None, "sustained high conflict + low reward must trigger a shift"
    assert shifted_set != start_set, (
        f"active_set must change after the shift ({start_set} -> {shifted_set})"
    )

    # (c) a real switch cost: mapping strength DROPS right after the shift, then RECOVERS.
    assert switch_cost > 0.05, f"a real switch cost must be recorded ({switch_cost:.3f})"
    assert post_shift_mag < pre_shift_mag, "mapping strength must drop right after the shift"
    assert recovered_mag > post_shift_mag, "mapping strength must recover as the new set re-settles"

    # (d) the conflict trigger is genuinely baseline-relative: the SAME conflict (0.7) shifts against
    # a low learned good-time baseline and holds against a high one -- a fixed cutoff cannot do both.
    assert shift_when_baseline_low, (
        "conflict 0.7 is anomalous vs a 0.3 good-time baseline -> must shift"
    )
    assert hold_when_baseline_high, (
        "the same conflict 0.7 is normal vs a 0.9 good-time baseline -> must hold"
    )

    # determinism: same seed -> identical shift tick, active set and confidence trajectory.
    def _scenario(seed):
        layer = TaskSetLayer(n_sets=3, n_in=4, n_actions=3, seed=seed, patience=5)
        trail = []
        for _ in range(20):
            layer.step()
            layer.observe(conflict=0.9, reward=-0.6)
            trail.append((layer.active_set, round(layer._confidence, 12)))
        return trail

    same_a, same_b = _scenario(9), _scenario(9)
    assert same_a == same_b, "same seed must reproduce an identical shift trajectory"

    print(
        f"self-check OK: single set held (settle spread {settle_spread:.5f}); 3-set sustained "
        f"conflict shifted {start_set}->{shifted_set} at tick {shifted_at}; switch cost "
        f"{switch_cost:.3f} (mapping mag {pre_shift_mag:.3f} -> {post_shift_mag:.3f} -> "
        f"{recovered_mag:.3f}); deterministic"
    )


if __name__ == "__main__":
    main()
