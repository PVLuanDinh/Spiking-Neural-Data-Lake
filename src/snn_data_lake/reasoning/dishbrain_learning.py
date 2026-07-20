"""
DishBrain free-energy embodied learning -- a spiking culture learns Pong from the
PREDICTABILITY of its sensory feedback (Kagan et al. 2022), not from a scalar reward.

Cortical Labs' "DishBrain" (Kagan, Kitchen, et al., "In vitro neurons learn and exhibit
sentience when embodied in a simulated game-world", Neuron 110(23):3952-3969, 2022,
DOI 10.1016/j.neuron.2022.09.001): a neural culture embodied in a Pong-like world learns to
hit the ball, driven ENTIRELY by the FREE ENERGY PRINCIPLE. The only teaching signal is the
ENTROPY STRUCTURE of the stimulation the network receives back after each rally:
  - HIT  -> PREDICTABLE stimulation (a consistent, low-entropy REPLAY of the rally's own sensory
            trace) -> Hebbian CONSOLIDATION of the pathway that was just active.
  - MISS -> UNPREDICTABLE stimulation (a RANDOM, high-entropy pattern) -> the active pathway is
            NOT consolidated; the surprise dissipates against a homeostatic prior.
A homeostatic synaptic DECAY continually relaxes every weight toward a neutral prior, so ONLY the
consistent (predictable-on-hit) stimulation accumulates -- random stimulation averages to nothing
and washes out. The culture therefore drifts toward whatever keeps its feedback predictable -- i.e.
toward KEEPING THE BALL HIT -- and the hit-rate RISES over training, an EMERGENT property of the
rule (nothing about the winning policy is coded in; it is discovered from the feedback structure).

THE DEFINING CONTROL (the honest, falsifiable result reproduced here, from the paper): a culture
whose feedback is CONTINGENT on performance (predictable-on-hit / random-on-miss) LEARNS; a control
given RANDOM feedback REGARDLESS of outcome (never any predictable stimulation) does NOT improve. It
is the CONTINGENCY between action-outcome and feedback-structure -- not stimulation per se -- that
drives learning. Same architecture, same seed; the ONLY difference is whether "predictable" feedback
is gated on the hit (structured) or is simply never delivered (control).

DISTINCT from the repo's other free-energy / reward modules (this is NOT a scalar-reward rule and IS
embodied in a task world -- do not duplicate):
  - research/free_energy_control.py : a single spike fires only if it lowers prediction error
      -- spike-level control, NO embodiment, NO task.
  - src/snn_data_lake/closed_loop.py: dopamine/cortisol learning from a graded SCALAR reward.
  - src/snn_data_lake/r_stdp.py, valence_stdp.py: reward-modulated STDP with an explicit scalar
      reward / reward-prediction-error.
Here the network NEVER sees a reward number. Its ONLY learning signal is whether the sensory
feedback it gets back is a CONSISTENT replay or a SCRAMBLED random pattern, and it self-organises
(free-energy principle) to keep that feedback consistent.

  python research/dishbrain_learning.py        # (also run by run_selfchecks.py)
"""
import random

# --- the 1-D Pong world -------------------------------------------------------------------------
H = 11                   # vertical positions 0..H-1 (the play-field)
START = H // 2           # paddle starts centred
W_HALF = 1               # paddle covers [p - W_HALF, p + W_HALF]  (HIT if |ball - paddle| <= W_HALF)
SUBSTEPS = H             # motor steps the agent gets to move the paddle before the ball arrives

# --- the spiking agent + DishBrain plasticity ---------------------------------------------------
N_OFF = 2 * H - 1        # place-coded RELATIVE position: offset d = ball - paddle, unit idx = d + (H-1)
ETA = 0.0004             # plasticity step (consolidation / random-stimulation magnitude)
NOISE = 0.10             # exploration: jitter on the motor drives (keeps behaviour stochastic while naive)
LAMBDA = 0.0003          # homeostatic decay per rally: relaxes every weight toward the neutral prior
CAP = 1.0                # soft weight clip (keeps dynamics bounded + deterministic)

# --- training / measurement ---------------------------------------------------------------------
N_TRIALS = 8000
WINDOW = 1000            # first-WINDOW (early / naive baseline) vs last-WINDOW (late / trained) hit-rate


class DishBrainAgent:
    """Place-coded sensory offset -> two plastic motor pathways (UP / DOWN). The motor is a minimal
    winner-take-all of two threshold units: whichever has the larger place-coded drive (weight +
    exploration noise) emits a spike and nudges the paddle one cell. Learning is the DishBrain
    feedback rule -- the entropy STRUCTURE of post-rally stimulation, never a reward number."""

    def __init__(self, seed, contingent=True):
        self.rng = random.Random(seed)          # ONE per-instance rng threaded through EVERYTHING
        self.contingent = contingent
        self.w_up = [0.0] * N_OFF               # neutral prior: no built-in policy
        self.w_down = [0.0] * N_OFF

    def _motor(self, idx):
        """One motor step for place-coded offset unit `idx`: two threshold units compete on their
        weighted drive + exploration noise. Returns +1 (UP spike) or -1 (DOWN spike)."""
        up = self.w_up[idx] + self.rng.uniform(-NOISE, NOISE)
        dn = self.w_down[idx] + self.rng.uniform(-NOISE, NOISE)
        return 1 if up > dn else -1

    def play_ball(self, y):
        """Run one rally against a ball at vertical position `y`. Sense the place-coded offset, let
        the motor move the paddle for SUBSTEPS, return (hit, eligibility). Eligibility = the
        (offset-unit, action) synapses that spiked this rally -- the active pathway."""
        p = START
        elig = []
        for _ in range(SUBSTEPS):
            idx = (y - p) + (H - 1)             # place code of the RELATIVE ball offset
            a = self._motor(idx)
            elig.append((idx, a))
            p = min(H - 1, max(0, p + a))
        return (abs(y - p) <= W_HALF), elig

    def feedback(self, hit, elig):
        """Deliver post-rally stimulation and Hebbian-update the pathway that was active. The
        STRUCTURE (predictable replay vs random scramble) -- not any reward magnitude -- is the
        signal:
          predictable (contingent agent, on a HIT): a consistent low-entropy replay reinforces every
              used synapse (+ETA) -> the pathway that scored CONSOLIDATES.
          random (a MISS, or ALL feedback for the control agent): a high-entropy scramble drives a
              random-signed nudge (from the rng) on each used synapse -> mean zero, no consolidation.
        Then a homeostatic decay relaxes ALL weights toward the neutral prior, so only the consistent
        predictable stimulation accumulates; random stimulation dissipates."""
        predictable = self.contingent and hit
        for idx, a in elig:
            w = self.w_up if a > 0 else self.w_down
            delta = ETA if predictable else ETA * self.rng.uniform(-1.0, 1.0)
            w[idx] = min(CAP, max(-CAP, w[idx] + delta))
        for i in range(N_OFF):                  # synaptic homeostasis (relax toward the neutral prior)
            self.w_up[i] *= (1.0 - LAMBDA)
            self.w_down[i] *= (1.0 - LAMBDA)


def train(seed, contingent=True, n_trials=N_TRIALS):
    """Play `n_trials` rallies; ball positions drawn from the agent's OWN seeded rng. Returns the
    hit/miss trajectory (list of 0/1) -- fully deterministic in `seed`."""
    agent = DishBrainAgent(seed, contingent=contingent)
    hits = []
    for _ in range(n_trials):
        y = agent.rng.randrange(H)
        hit, elig = agent.play_ball(y)
        hits.append(1 if hit else 0)
        agent.feedback(hit, elig)
    return hits


def _rate(hits, lo, hi):
    return sum(hits[lo:hi]) / (hi - lo)


def _deciles(hits):
    n = len(hits) // 10
    return [round(sum(hits[i * n:(i + 1) * n]) / n, 2) for i in range(10)]


def main():
    seed = 7
    struct = train(seed, contingent=True)
    control = train(seed, contingent=False)

    s_early, s_late = _rate(struct, 0, WINDOW), _rate(struct, N_TRIALS - WINDOW, N_TRIALS)
    c_early, c_late = _rate(control, 0, WINDOW), _rate(control, N_TRIALS - WINDOW, N_TRIALS)
    chance = (2 * W_HALF + 1) / H            # a static centred paddle covers this fraction of balls

    print("=" * 72)
    print("DISHBRAIN -- embodied spiking Pong learned from FEEDBACK PREDICTABILITY (free energy)")
    print("=" * 72)
    print(f"world: {H} positions, paddle half-width {W_HALF} (covers {2 * W_HALF + 1}/{H}), "
          f"{SUBSTEPS} motor steps/rally, {N_TRIALS} rallies")
    print("feedback: HIT -> predictable replay (consolidate) | MISS -> random scramble (dissipates)")
    print(f"geometric chance (static paddle): {chance:.2f}")
    print("-" * 72)
    print(f"structured learning curve (deciles): {_deciles(struct)}")
    print(f"control    learning curve (deciles): {_deciles(control)}")
    print("-" * 72)
    print("hit-rate      early   late    delta")
    print(f"structured    {s_early:.2f}    {s_late:.2f}   {s_late - s_early:+.2f}   <- LEARNS "
          f"(contingent feedback)")
    print(f"control       {c_early:.2f}    {c_late:.2f}   {c_late - c_early:+.2f}   <- flat "
          f"(random feedback, NO contingency)")
    print("=" * 72)

    # ---- self-check (deterministic asserts; the paper's control; EMERGENT, nothing hardcoded) ----
    # (a) the STRUCTURED agent LEARNS: late hit-rate clearly above its early baseline AND above chance.
    assert s_late > s_early + 0.15, \
        f"structured agent must LEARN: late {s_late:.2f} not clearly above early {s_early:.2f}"
    assert s_late > chance + 0.15, \
        f"structured late {s_late:.2f} must beat geometric chance {chance:.2f} comfortably"
    # (b) the CONTROL (random feedback regardless of outcome) does NOT learn -- proving CONTINGENCY,
    #     not mere stimulation, drives learning. Its late rate ~= its early rate, and structured wins.
    assert abs(c_late - c_early) < 0.12, \
        f"control must NOT learn: early {c_early:.2f} -> late {c_late:.2f} changed too much"
    assert s_late > c_late + 0.15, \
        f"structured {s_late:.2f} must beat the no-contingency control {c_late:.2f}"
    # (c) determinism: same seed -> byte-identical hit-rate trajectory.
    assert train(seed, contingent=True) == struct, "same seed must reproduce the exact trajectory"
    print(f"self-check OK: contingent feedback LEARNS ({s_early:.2f} -> {s_late:.2f}, > chance "
          f"{chance:.2f}); random-feedback control stays flat ({c_early:.2f} -> {c_late:.2f}); "
          f"contingency drives it; deterministic")


if __name__ == "__main__":
    main()
