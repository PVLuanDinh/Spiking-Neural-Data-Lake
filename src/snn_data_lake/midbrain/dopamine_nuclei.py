"""
Midbrain dopamine SOURCE nuclei (VTA / SNc) -- a spiking population that EMITS a
reward-prediction-error (RPE) teaching signal.

This is the SOURCE of dopamine, not a consumer of it. The nucleus keeps its own TD-style
prediction of upcoming reward (a scalar critic cache V over trial states) and broadcasts the
prediction ERROR as the firing of a small population of dopamine neurons. Downstream striatal
plasticity (basal_ganglia/valence_stdp.py) is the CONSUMER: it reads this dopamine level and
updates its synaptic weights. This module deliberately does NOT re-implement that striatal
plasticity -- it holds no per-feature synaptic weight vector and makes no APPROACH/AVOID
decision. It only computes an RPE and turns it into a dopamine spike rate.

Mechanism (Schultz 1997; Montague/Dayan/Sejnowski 1996 -- TD dopamine):
  1. A trial is an ordered chain of states (ITI -> CUE -> TRACE -> REWARD). The reward is
     delivered on entering the REWARD state.
  2. The nucleus caches a predicted value V(state) and, at each transition s -> s_next, computes
     the temporal-difference error   rpe = r + gamma * V(s_next) - V(s).
     This rpe IS the phasic dopamine signal at that moment. When learning is on, V(s) is nudged
     toward the outcome by rpe (TD(0)); the pre-cue baseline (ITI) is held un-learnable because
     trial onset is temporally unpredictable, which is what keeps a cue response alive after
     learning (see the transfer, below).
  3. The dopamine cells are a small leaky integrate-and-fire (LIF) population. Their drive is the
     dopamine LEVEL = clip(tonic_baseline + gain * rpe, 0, 1), a value in [0, 1] that the basal
     ganglia could consume directly. The population's normalized spike RATE tracks that level:
     tonic baseline rate at rpe ~= 0, a phasic BURST when rpe > 0, a phasic DIP toward silence
     when rpe < 0.

Fair control (un-riggable): the phasic burst tracks SURPRISE (the RPE), NOT reward magnitude.
The self-check delivers the SAME magnitude-1 reward in two conditions and shows the dopamine
rate differs only with prediction:
  - a FULLY PREDICTED reward evokes NO burst      (rpe ~= 0 -> rate stays at baseline),
  - an UNPREDICTED reward evokes a phasic BURST    (rate > baseline),
  - an OMITTED expected reward evokes a phasic DIP (rate < baseline),
  - classic TRANSFER: after learning a cue -> reward association, the burst moves FROM the reward
    TO the earliest predictor, the CUE (reward now predicted -> no reward burst; cue now carries
    the surprise -> cue burst). Magnitude is identical throughout; only prediction changes.

Honest framing: a reduced, rate-level model of the dopaminergic RPE broadcast on a synthetic
trial chain. It models the teaching-signal FUNCTION of VTA/SNc -- what the cells emit -- not
motivation, wanting, pleasure, or consciousness. A "burst" here is a raised population spike
rate in a toy LIF pool, nothing more.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.midbrain.dopamine_nuclei
"""

import random

# A trial is a chain of (state, reward-on-entry). Reward is delivered when entering REWARD.
CUED_REWARD = [("ITI", 0.0), ("CUE", 0.0), ("TRACE", 0.0), ("REWARD", 1.0)]
CUED_OMIT = [("ITI", 0.0), ("CUE", 0.0), ("TRACE", 0.0), ("REWARD", 0.0)]
BASELINE_STATES = frozenset(
    {"ITI"}
)  # pre-cue baseline: trial onset is unpredictable -> not learned


class DopamineNuclei:
    """VTA/SNc dopamine cells: a TD critic that EMITS an RPE as a spiking population rate.

    phasic(rpe) turns a reward-prediction error into a normalized [0, 1] dopamine population
    rate; present(trial) walks a trial chain, returns the per-event RPE, and (optionally) updates
    the internal value cache V. The nucleus never decides an action -- it only broadcasts dopamine.
    """

    def __init__(
        self,
        seed=0,
        gamma=0.9,
        alpha=0.3,
        tonic_baseline=0.2,
        gain=0.8,
        n_cells=24,
        steps=60,
        leak=0.1,
        noise=0.015,
    ):
        self.rng = random.Random(seed)
        self.gamma = gamma  # temporal discount of the critic
        self.alpha = alpha  # TD(0) learning rate of the value cache
        self.tonic_baseline = tonic_baseline  # dopamine level at rpe == 0 (tonic firing)
        self.gain = gain  # how strongly the phasic rate follows the rpe
        self.n_cells = n_cells  # dopamine neurons in the population
        self.steps = steps  # LIF integration steps per readout
        self.leak = leak  # LIF membrane leak
        self.noise = noise  # small per-step input noise (kept deterministic)
        # heterogeneous per-cell thresholds make it a population, not one unit:
        self.thresholds = [self.rng.uniform(0.9, 1.1) for _ in range(n_cells)]
        self.readout_seed = seed * 100003 + 7  # fixed readout noise -> rate is a pure fn of level
        self.v = {}  # critic cache: state -> predicted value (NOT synapses)

    def _value(self, state):
        return self.v.get(state, 0.0)

    def da_level(self, rpe):
        """Dopamine LEVEL in [0, 1] that downstream basal ganglia consume: tonic baseline shifted
        by the (gain-scaled) RPE, rectified and clipped."""
        return min(1.0, max(0.0, self.tonic_baseline + self.gain * rpe))

    def _population_rate(self, level):
        """Normalized spike rate of the LIF dopamine pool for a constant dopamine level.

        Deterministic readout (fixed seed) so the rate is a pure, reproducible function of the
        level: near 0 for a dip, tonic for baseline, saturating toward 1 for a strong burst."""
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

    def phasic(self, rpe):
        """The phasic dopamine population rate emitted for a reward-prediction error."""
        return self._population_rate(self.da_level(rpe))

    def present(self, trial, learn=True):
        """Walk a trial chain, computing the TD error emitted on ENTERING each state. Returns a
        dict state -> rpe. With learn=True the value cache is updated by TD(0) (the baseline
        pre-cue state is never learned, keeping the cue response alive after acquisition)."""
        rpe = {}
        for t in range(len(trial) - 1):
            s, _ = trial[t]
            s_next, r_next = trial[t + 1]
            delta = r_next + self.gamma * self._value(s_next) - self._value(s)
            rpe[s_next] = delta
            if learn and s not in BASELINE_STATES:
                self.v[s] = self._value(s) + self.alpha * delta
        return rpe


def _train(nuc, trial, n):
    """Run n learning trials of the same chain (acquisition)."""
    for _ in range(n):
        nuc.present(trial, learn=True)


def main():
    n_train = 80
    nuc = DopamineNuclei(seed=0)

    baseline_rate = nuc.phasic(0.0)  # tonic dopamine (no prediction error)

    naive = nuc.present(CUED_REWARD, learn=False)  # first exposure: nothing predicts reward
    naive_cue_rpe, naive_rew_rpe = naive["CUE"], naive["REWARD"]
    naive_cue_rate = nuc.phasic(naive_cue_rpe)
    naive_rew_rate = nuc.phasic(naive_rew_rpe)  # UNPREDICTED reward -> burst

    _train(nuc, CUED_REWARD, n_train)  # learn the cue -> reward association

    trained = nuc.present(CUED_REWARD, learn=False)  # reward now fully predicted by the cue
    tr_cue_rpe, tr_rew_rpe = trained["CUE"], trained["REWARD"]
    tr_cue_rate = nuc.phasic(tr_cue_rpe)  # burst has TRANSFERRED to the cue
    tr_rew_rate = nuc.phasic(tr_rew_rpe)  # PREDICTED reward -> no burst

    omit = nuc.present(CUED_OMIT, learn=False)  # expected reward withheld
    omit_rpe = omit["REWARD"]
    omit_rate = nuc.phasic(omit_rpe)  # -> phasic DIP below baseline

    print("=" * 70)
    print("MIDBRAIN DOPAMINE NUCLEI (VTA/SNc) -- spiking RPE teaching signal")
    print("=" * 70)
    print(
        f"cells: {nuc.n_cells} | gamma: {nuc.gamma} | alpha: {nuc.alpha} | "
        f"tonic baseline level: {nuc.tonic_baseline:.2f}"
    )
    print(
        f"learned values after {n_train} trials: V(CUE)={nuc._value('CUE'):.3f}  "
        f"V(TRACE)={nuc._value('TRACE'):.3f}  (V(REWARD)={nuc._value('REWARD'):.3f}, terminal)"
    )
    print(f"tonic baseline dopamine rate (rpe=0): {baseline_rate:.3f}")
    print("-" * 70)
    print("condition                     |   rpe  | da level | pop rate | vs baseline")
    print("-" * 70)
    rows = [
        ("naive  cue  (no prediction)", naive_cue_rpe, naive_cue_rate),
        ("naive  reward (UNPREDICTED)", naive_rew_rpe, naive_rew_rate),
        ("trained cue  (TRANSFER->cue)", tr_cue_rpe, tr_cue_rate),
        ("trained reward (PREDICTED)", tr_rew_rpe, tr_rew_rate),
        ("omitted reward (expected)", omit_rpe, omit_rate),
    ]
    for name, rpe, rate in rows:
        tag = (
            "BURST"
            if rate > baseline_rate + 0.05
            else ("DIP" if rate < baseline_rate - 0.03 else "baseline")
        )
        print(f"{name:<29} | {rpe:+.3f} |  {nuc.da_level(rpe):.3f}   |  {rate:.3f}   | {tag}")
    print("=" * 70)
    print("same magnitude-1 reward: UNPREDICTED bursts, PREDICTED does not -> the burst tracks")
    print("SURPRISE (rpe), not reward magnitude; after learning it TRANSFERS to the cue.")
    print("=" * 70)

    # ---- un-riggable self-checks (deterministic asserts) --------------------
    # (1) fully PREDICTED reward evokes no phasic burst: rate stays at baseline (rpe ~= 0).
    assert abs(tr_rew_rate - baseline_rate) < 0.05, (
        "predicted reward must not burst (stay baseline)"
    )
    assert abs(tr_rew_rpe) < 0.05, "a fully predicted reward must drive rpe to ~0"
    # (2) UNPREDICTED reward evokes a phasic burst: rate clearly above baseline.
    assert naive_rew_rate > baseline_rate + 0.3, "unpredicted reward must evoke a phasic burst"
    assert naive_rew_rpe > 0.5, "unpredicted reward must be a positive surprise (rpe > 0)"
    # SURPRISE not magnitude: identical reward, burst only when unpredicted.
    assert naive_rew_rate > tr_rew_rate + 0.3, (
        "same reward: unpredicted must burst, predicted must not"
    )
    # (3) OMITTED expected reward evokes a phasic dip: rate below baseline (rpe < 0).
    assert omit_rate < baseline_rate - 0.03, "omitted expected reward must dip below baseline"
    assert omit_rpe < -0.5, "omitted expected reward must be a negative surprise (rpe < 0)"
    # (4) TRANSFER: burst moves from reward (naive) to cue (trained).
    assert naive_cue_rate < baseline_rate + 0.05, "naive cue carries no surprise yet (no burst)"
    assert tr_cue_rate > baseline_rate + 0.3, "after learning the cue must carry the phasic burst"
    assert tr_cue_rate > tr_rew_rate + 0.3, "trained: burst is at the cue, not the reward"
    # Determinism: same seed -> identical learned values and identical rates.
    other = DopamineNuclei(seed=0)
    other.phasic(0.0)
    o_naive = other.present(CUED_REWARD, learn=False)
    _train(other, CUED_REWARD, n_train)
    o_trained = other.present(CUED_REWARD, learn=False)
    assert other._value("CUE") == nuc._value("CUE"), "same seed must reproduce learned values"
    assert other.phasic(o_naive["REWARD"]) == naive_rew_rate, "same seed must reproduce rates"
    assert other.phasic(o_trained["CUE"]) == tr_cue_rate, "same seed must reproduce transfer rate"

    print(
        f"self-check OK: baseline {baseline_rate:.3f}; UNPREDICTED reward bursts {naive_rew_rate:.3f} "
        f"(rpe {naive_rew_rpe:+.2f}) while PREDICTED stays {tr_rew_rate:.3f};"
    )
    print(
        f"  omission dips {omit_rate:.3f} (rpe {omit_rpe:+.2f}); burst TRANSFERS reward->cue "
        f"(cue {tr_cue_rate:.3f} vs reward {tr_rew_rate:.3f}); surprise not magnitude; deterministic"
    )


if __name__ == "__main__":
    main()
