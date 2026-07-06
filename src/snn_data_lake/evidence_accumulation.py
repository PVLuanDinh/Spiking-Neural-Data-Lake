"""
Evidence-accumulation / drift-diffusion decision model -- perceptual decision-making by
bounded integration of noisy evidence (pure stdlib, zero deps).

A noisy sensory stream is INTEGRATED over time until an accumulator crosses a decision BOUND;
the crossing yields BOTH a choice AND a reaction time (the number of integration steps to the
bound). This is the algorithmic-level drift-diffusion / bounded-accumulator model of two-choice
perceptual decision-making (Ratcliff, Psychol Rev 85:59, 1978; Gold & Shadlen, Annu Rev Neurosci
30:535, 2007; the spiking attractor realisation is Wang, Neuron 36:955, 2002). The N-alternative
form is a leaky competing accumulator race (Usher & McClelland, Psychol Rev 108:550, 2001).

The two signatures that define this model and that emerge from the self-check:
  - a PSYCHOMETRIC curve : accuracy rises with evidence strength (|drift|), from chance at zero
    drift to near-certain at strong drift;
  - a SPEED-ACCURACY tradeoff : raising the bound buys accuracy at the cost of a longer RT;
  - a CHRONOMETRIC curve : stronger evidence is decided FASTER (shorter RT).

DISTINCTNESS (this is perceptual DECISION-MAKING, not value learning or control):
  - valence_stdp.py / closed_loop.py are VALUE-based action selection: they LEARN a scalar value
    from reward prediction error (dopamine RPE) and pick the higher-valued action (APPROACH/AVOID
    or a matched command). There is no time-integration of noisy evidence, no decision bound, no
    reaction time and no speed-accuracy tradeoff. Here nothing is learned: noisy sensory evidence
    is integrated over time to a threshold, producing a choice AND a latency, with the classic
    speed-accuracy tradeoff.
  - spiking_pid.py is closed-loop CONTROL to a setpoint -- a continuous corrective signal, not a
    discrete commitment to one of several alternatives.
  - not predictive_coding (perceptual inference of latent causes) and not neural_sampling
    (posterior sampling of a joint): this is a DECISION -- an irreversible commit to one option at
    a threshold, timestamped by the step at which the bound was reached.

ponytail: the ceiling is a REDUCED drift-diffusion / bounded-accumulator decision model -- Gaussian
noise on synthetic scalar evidence integrated by a fixed Euler (Euler-Maruyama) step to a bound.
This is the algorithmic level of the decision only; NOT a spiking attractor network (Wang 2002),
not biophysics, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.evidence_accumulation
"""
import math
import random


class DriftDiffusion:
    """Two-choice drift-diffusion accumulator.

    A single decision variable x starts at 0 and each step integrates deterministic evidence plus
    Gaussian noise:  x += drift*dt + noise*gauss(0,1)*sqrt(dt)  (Euler-Maruyama on a Wiener process).
    It terminates when x >= +bound (choose +1) or x <= -bound (choose -1). sign(drift) is the correct
    option and |drift| is the evidence/stimulus strength ("coherence"). The RNG is seeded once at
    construction, so two fresh instances with the same params reproduce identical trials.
    """

    def __init__(self, drift, bound, noise=1.0, dt=0.01, seed=0, max_steps=100000):
        self.drift = drift
        self.bound = bound
        self.noise = noise
        self.dt = dt
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._sqrt_dt = math.sqrt(dt)

    def decide(self):
        """Run one trial to the bound. Returns (choice, rt) with choice in {+1, -1} and rt = steps."""
        x = 0.0
        drift_step = self.drift * self.dt
        for steps in range(1, self.max_steps + 1):
            x += drift_step + self.noise * self._rng.gauss(0.0, 1.0) * self._sqrt_dt
            if x >= self.bound:
                return 1, steps
            if x <= -self.bound:
                return -1, steps
        # Non-termination guard (unreached for sane params): commit to the leading side.
        return (1 if x >= 0.0 else -1), self.max_steps

    def run_trials(self, n):
        """Run n independent trials off the seeded stream. Returns a list of (choice, rt)."""
        return [self.decide() for _ in range(n)]


class RaceAccumulator:
    """N-alternative leaky competing accumulator race (Usher & McClelland 2001).

    N accumulators each integrate their own noisy evidence; the first to cross the bound wins.
    With leak > 0 and inhibition > 0 this is the leaky competing accumulator: each unit decays
    (leak) and is suppressed by the summed activity of the others (lateral inhibition), and
    activities are rectified to be non-negative (firing rates). leak=inhibition=0 is a pure race.
    Named RaceAccumulator (NOT "LCA") to avoid collision with the unrelated lca_sparse_coding.
    """

    def __init__(self, drifts, bound, noise=1.0, dt=0.01, leak=0.0, inhibition=0.0,
                 seed=0, max_steps=100000):
        self.drifts = list(drifts)
        self.bound = bound
        self.noise = noise
        self.dt = dt
        self.leak = leak
        self.inhibition = inhibition
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._sqrt_dt = math.sqrt(dt)

    def decide(self):
        """Run one race to the bound. Returns (winner_index, rt) with rt = steps."""
        n = len(self.drifts)
        x = [0.0] * n
        for steps in range(1, self.max_steps + 1):
            total = sum(x)
            for i in range(n):
                dx = (self.drifts[i] - self.leak * x[i] - self.inhibition * (total - x[i])) * self.dt
                xi = x[i] + dx + self.noise * self._rng.gauss(0.0, 1.0) * self._sqrt_dt
                x[i] = xi if xi > 0.0 else 0.0
            winner = max(range(n), key=lambda i: x[i])
            if x[winner] >= self.bound:
                return winner, steps
        return max(range(n), key=lambda i: x[i]), self.max_steps

    def run_trials(self, n):
        return [self.decide() for _ in range(n)]


def accuracy(trials, correct):
    """Fraction of trials whose choice equals the correct option."""
    if not trials:
        return 0.0
    return sum(1 for choice, _rt in trials if choice == correct) / len(trials)


def mean_rt(trials):
    """Mean reaction time (steps) over trials."""
    if not trials:
        return 0.0
    return sum(rt for _choice, rt in trials) / len(trials)


def psychometric(drifts, n_trials, bound=1.0, noise=1.0, dt=0.01, seed=0):
    """Accuracy per drift: correctness = choice matches sign(drift). Returns a list aligned to drifts.

    At drift 0 the correct option is defined as +1; by symmetry P(+1) is ~0.5 (true chance)."""
    out = []
    for i, d in enumerate(drifts):
        model = DriftDiffusion(d, bound, noise=noise, dt=dt, seed=seed + i)
        correct = 1 if d >= 0 else -1
        out.append(accuracy(model.run_trials(n_trials), correct))
    return out


def chronometric(drifts, n_trials, bound=1.0, noise=1.0, dt=0.01, seed=0):
    """Mean RT (steps) per drift. Returns a list aligned to drifts."""
    out = []
    for i, d in enumerate(drifts):
        model = DriftDiffusion(d, bound, noise=noise, dt=dt, seed=seed + i)
        out.append(mean_rt(model.run_trials(n_trials)))
    return out


def main():
    # Trial counts (modest but large enough for stable stats at the fixed seed).
    n_psy = 400        # trials per drift for the psychometric / chronometric sweep
    n_sa = 600         # trials per bound for the speed-accuracy comparison
    n_race = 300       # trials for the N-alternative race demo

    # ---- 1 + 3. psychometric (accuracy vs |drift|) and chronometric (RT vs |drift|) ----
    drifts = [0.0, 0.5, 1.0, 2.0, 3.5]
    acc = psychometric(drifts, n_psy)
    rts = chronometric(drifts, n_psy)

    print("=" * 66)
    print("DRIFT-DIFFUSION -- bounded integration of noisy evidence -> choice + RT")
    print("=" * 66)
    print("psychometric + chronometric (bound=1.0, noise=1.0, dt=0.01, "
          f"{n_psy} trials/drift):")
    print("   drift   accuracy   mean RT (steps)")
    for d, a, t in zip(drifts, acc, rts):
        print(f"   {d:>4.1f}     {a:>5.2f}       {t:>7.1f}")

    # ---- 2. speed-accuracy tradeoff: fixed moderate drift, LOW bound vs HIGH bound ----
    sa_drift = 1.0
    low_model = DriftDiffusion(sa_drift, bound=0.5, seed=0)
    high_model = DriftDiffusion(sa_drift, bound=1.5, seed=0)
    low = low_model.run_trials(n_sa)
    high = high_model.run_trials(n_sa)
    low_acc, low_rt = accuracy(low, 1), mean_rt(low)
    high_acc, high_rt = accuracy(high, 1), mean_rt(high)
    print(f"\nspeed-accuracy tradeoff (drift={sa_drift}, {n_sa} trials each):")
    print(f"   low  bound 0.5 -> accuracy {low_acc:.2f}, mean RT {low_rt:.1f} steps")
    print(f"   high bound 1.5 -> accuracy {high_acc:.2f}, mean RT {high_rt:.1f} steps")
    print("   -> raising the bound buys accuracy at the cost of a longer RT")

    # ---- N-alternative race demo (leaky competing accumulator with lateral inhibition) ----
    race_drifts = [0.4, 1.0, 2.0]
    race = RaceAccumulator(race_drifts, bound=1.0, leak=0.2, inhibition=0.1, seed=0)
    wins = [0, 0, 0]
    race_trials = race.run_trials(n_race)
    for winner, _rt in race_trials:
        wins[winner] += 1
    strongest = race_drifts.index(max(race_drifts))
    print(f"\nN-alternative race (drifts {race_drifts}, leak+inhibition, {n_race} trials):")
    print(f"   win counts {wins} -> strongest evidence (drift {race_drifts[strongest]}) "
          f"wins {wins[strongest]}/{n_race}")

    # ---- chronometric weak vs strong (a slice of the sweep above) ----
    weak_rt, strong_rt = rts[1], rts[-1]     # drift 0.5 vs drift 3.5
    print(f"\nchronometric: weak drift {drifts[1]} RT {weak_rt:.1f} steps "
          f"vs strong drift {drifts[-1]} RT {strong_rt:.1f} steps -> stronger is faster")
    print("=" * 66)

    # ---- self-checks (genuine, not rigged) --------------------------------------------
    # 1. psychometric: chance at zero drift, high at strong drift, weakly monotone in |drift|.
    assert 0.42 <= acc[0] <= 0.58, f"accuracy at zero drift must be ~chance ({acc[0]:.2f})"
    assert acc[-1] > 0.9, f"accuracy at strong drift must be high (>0.9) ({acc[-1]:.2f})"
    for i in range(len(acc) - 1):
        assert acc[i + 1] >= acc[i] - 0.02, (
            f"accuracy must rise (weakly) with |drift|: {acc[i]:.2f} -> {acc[i + 1]:.2f}")

    # 2. speed-accuracy tradeoff: high bound = higher accuracy AND longer RT.
    assert high_acc > low_acc, f"high bound must be more accurate ({high_acc:.2f} vs {low_acc:.2f})"
    assert high_rt > low_rt, f"high bound must be slower ({high_rt:.1f} vs {low_rt:.1f} steps)"

    # 3. chronometric: strong evidence decided clearly faster than weak evidence.
    assert strong_rt < weak_rt, f"strong drift must be faster ({strong_rt:.1f} vs {weak_rt:.1f})"

    # N-alternative race: the strongest accumulator wins the majority of races.
    assert wins[strongest] == max(wins) and wins[strongest] > n_race / 2, (
        f"strongest evidence must win the race majority ({wins})")

    # 4. determinism: same params/seed -> identical trials across two fresh instances.
    a1 = DriftDiffusion(1.0, 1.0, seed=7).run_trials(50)
    a2 = DriftDiffusion(1.0, 1.0, seed=7).run_trials(50)
    assert a1 == a2, "same seed must reproduce identical (choice, rt) trials"
    r1 = RaceAccumulator([0.5, 1.5], 1.0, seed=3).run_trials(30)
    r2 = RaceAccumulator([0.5, 1.5], 1.0, seed=3).run_trials(30)
    assert r1 == r2, "race must be deterministic under a fixed seed"

    print(f"self-check OK: psychometric chance {acc[0]:.2f} -> strong {acc[-1]:.2f}; "
          f"speed-accuracy low (acc {low_acc:.2f}, RT {low_rt:.1f}) vs high "
          f"(acc {high_acc:.2f}, RT {high_rt:.1f}); chronometric weak {weak_rt:.1f} -> "
          f"strong {strong_rt:.1f} steps; deterministic")


if __name__ == "__main__":
    main()
