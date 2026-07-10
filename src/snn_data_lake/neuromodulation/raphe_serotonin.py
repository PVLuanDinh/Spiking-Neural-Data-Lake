"""
Dorsal raphe serotonin (5-HT) -- PATIENCE / temporal-discount setpoint, the opponent to dopamine.

Serotonin from the dorsal raphe is modelled here as a patience / time-horizon signal and tonic
setpoint: the opponent process to phasic dopamine's impulsivity. HIGH 5-HT promotes WAITING for
a larger-later reward; LOW 5-HT is impulsive, taking the smaller-sooner reward (Miyazaki et al.
2011/2014; Doya 2002 discount-rate hypothesis; Cools, Nakamura & Roberts 2008 opponency). The
5-HT level in [0,1] sets a hyperbolic temporal-discount rate k, and an intertemporal-choice
method picks smaller-sooner vs larger-later given that level.

Mechanism (rate-level, deterministic):
  1. discount_rate() maps 5-HT -> the hyperbolic rate k, DECREASING monotonically with 5-HT
     (high 5-HT -> low k -> the future is barely discounted -> patient).
  2. value(reward, delay) = reward / (1 + k*delay); choose(sooner, later) takes whichever option
     has the higher discounted value.
  3. patience(battery) = the fraction of a fixed smaller-sooner-vs-larger-later battery on which
     the larger-later option is chosen.

Fair control (un-riggable -- 5-HT is load-bearing and monotonic; the SAME battery at every level):
  - HIGH 5-HT -> mostly larger-later choices (patient).
  - LOW  5-HT -> mostly smaller-sooner choices (impulsive).
  - a MID level interpolates, and the patience fraction increases MONOTONICALLY with 5-HT.
  The monotonicity is structural: for a fixed pair the discounted value ratio LL/SS equals
  (R_l/R_s) * (1 + k*d_s) / (1 + k*d_l), which rises monotonically as k FALLS (5-HT rises, since
  d_l > d_s), so a larger-later choice, once made, is never reversed by MORE 5-HT.

Honest framing: a reduced, rate-level scalar model of serotonergic patience via hyperbolic
discounting. Additive/opt-in, changes no existing behaviour, and models a time-horizon / discount
FUNCTION -- not mood affect, not clinical serotonin, not cognition/AGI. Credit: Doya 2002;
Schweighofer et al. 2008; Miyazaki, Miyazaki & Doya 2011/2014.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.raphe_serotonin
"""
import math
import random

K_HI = 2.0          # steep hyperbolic discount at 5-HT=0 (impulsive: the future is crushed)
K_LO = 0.02         # shallow discount at 5-HT=1 (patient: the future is barely discounted)
TAU = 50.0          # slow tonic 5-HT time constant (a mood/context setpoint, not a fast signal)

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 5-HT sweep: impulsive -> patient


class DorsalRaphe:
    """Tonic 5-HT level in [0,1] that sets the temporal-discount rate k (hyperbolic). HIGH 5-HT ->
    LOW k -> shallow discounting -> WAIT for the larger-later reward (patient). LOW 5-HT -> HIGH k
    -> steep discounting -> take the smaller-sooner reward (impulsive). 5-HT is the opponent to
    dopamine's impulsivity; here a patience / time-horizon knob, not reward and not mood affect."""

    def __init__(self, level=0.5, k_hi=K_HI, k_lo=K_LO, tau=TAU):
        self.level = max(0.0, min(1.0, level))
        self.k_hi = k_hi
        self.k_lo = k_lo
        self.decay = math.exp(-1.0 / tau)

    def step(self, drive):
        """Advance one tick. Tonic 5-HT tracks a slow mood/context `drive` in [0,1] (leaky EMA) --
        a setpoint on the minutes scale, not a fast phasic signal. Returns the new 5-HT level."""
        drive = max(0.0, min(1.0, drive))
        self.level = self.decay * self.level + (1.0 - self.decay) * drive
        return self.level

    def discount_rate(self):
        """Hyperbolic discount rate k -- DECREASES monotonically with 5-HT (more 5-HT -> more
        patient). Log-spaced between k_hi (at 5-HT=0) and k_lo (at 5-HT=1) so mid 5-HT gives a
        graded, not bunched, discount: k = k_hi * (k_lo/k_hi)**level."""
        return self.k_hi * (self.k_lo / self.k_hi) ** self.level

    def value(self, reward, delay):
        """Subjective value of `reward` available after `delay`, hyperbolically discounted:
        V = reward / (1 + k*delay)."""
        return reward / (1.0 + self.discount_rate() * delay)

    def choose(self, sooner, later):
        """Intertemporal choice between sooner=(reward, delay) and later=(reward, delay), the later
        option larger and further off. Returns 'LL' (patient) or 'SS' (impulsive)."""
        return "LL" if self.value(*later) >= self.value(*sooner) else "SS"

    def patience(self, battery):
        """Fraction of a fixed SS-vs-LL battery on which the larger-later option is chosen."""
        return sum(1 for sooner, later in battery if self.choose(sooner, later) == "LL") / len(battery)

    def tolerated_wait(self, ratio):
        """Max delay tolerated for a reward `ratio`x larger than an immediate one (indifference of a
        hyperbolic discounter): d = (ratio - 1) / k. Grows as 5-HT rises (k falls) -> more patient."""
        return (ratio - 1.0) / self.discount_rate()


def _battery(rng, n=60):
    """A battery of smaller-sooner vs larger-later choices. The later option is always larger and
    further off; the pairs span a range so which is 'worth it' depends on the discount rate (5-HT)."""
    pairs = []
    for _ in range(n):
        r_s = rng.uniform(1.0, 2.0)
        d_s = rng.uniform(0.0, 1.5)
        ratio = rng.uniform(1.4, 2.6)
        extra = rng.uniform(1.0, 18.0)
        pairs.append(((r_s, d_s), (r_s * ratio, d_s + extra)))
    return pairs


def demo_intertemporal(seed):
    """Sweep 5-HT from low (impulsive) to high (patient) over ONE fixed choice battery and record
    the patience fraction (fraction of larger-later choices) and tolerated 2x wait at each level."""
    rng = random.Random(seed)
    battery = _battery(rng)
    frac = [DorsalRaphe(level=lvl).patience(battery) for lvl in LEVELS]
    waits = [DorsalRaphe(level=lvl).tolerated_wait(2.0) for lvl in LEVELS]
    return {"levels": LEVELS, "patience": frac, "waits": waits, "battery_size": len(battery)}


def main():
    d = demo_intertemporal(0)
    lo, mid, hi = d["patience"][0], d["patience"][2], d["patience"][-1]

    print("=" * 66)
    print("DORSAL RAPHE 5-HT -- patience / temporal-discount setpoint (dopamine's opponent)")
    print("=" * 66)
    print(f"intertemporal battery: {d['battery_size']} smaller-sooner vs larger-later choices")
    print("-" * 66)
    print("  5-HT   discount k   patience (larger-later chosen)   tolerated 2x wait")
    for i, lvl in enumerate(d["levels"]):
        pf = d["patience"][i]
        w = d["waits"][i]
        k = DorsalRaphe(level=lvl).discount_rate()
        bar = "#" * int(round(pf * 20))
        print(f"  {lvl:.2f}   {k:8.3f}    {pf:.2f}  {bar:<20}   d <= {w:6.2f}")
    print("-" * 66)
    print(f"LOW 5-HT (impulsive) patience={lo:.2f}  ->  HIGH 5-HT (patient) patience={hi:.2f}")
    print("=" * 66)

    # ---- self-checks (deterministic, zero-dep, ASCII-only) ------------------
    frac = d["patience"]
    # 1. the switch: low 5-HT is impulsive (smaller-sooner), high 5-HT is patient (larger-later)
    assert lo < 0.5, f"low 5-HT must be impulsive (mostly smaller-sooner), got patience {lo:.2f}"
    assert hi > 0.5, f"high 5-HT must be patient (mostly larger-later), got patience {hi:.2f}"
    assert hi > lo, "raising 5-HT must increase the larger-later (patient) fraction"
    # 2. a mid level interpolates between the impulsive and patient extremes
    assert lo <= mid <= hi, f"mid 5-HT must interpolate ({lo:.2f} <= {mid:.2f} <= {hi:.2f})"
    # 3. monotonicity: patience is non-decreasing in 5-HT (5-HT is load-bearing, un-riggable)
    assert all(frac[i] <= frac[i + 1] + 1e-9 for i in range(len(frac) - 1)), \
        f"patience must increase monotonically with 5-HT: {[f'{f:.2f}' for f in frac]}"
    # 4. discount rate falls with 5-HT so the tolerated wait grows -> the mechanism is monotone
    assert d["waits"][0] < d["waits"][-1], "tolerated waiting time must grow as 5-HT rises"
    # 5. determinism: same seed -> identical results
    assert demo_intertemporal(0) == demo_intertemporal(0), "same seed must reproduce identical results"

    print(f"self-check OK: patience rises monotonically with 5-HT "
          f"{[f'{f:.2f}' for f in frac]} (low={lo:.2f} impulsive -> high={hi:.2f} patient); "
          f"tolerated 2x wait {d['waits'][0]:.2f} -> {d['waits'][-1]:.2f}")


if __name__ == "__main__":
    main()
