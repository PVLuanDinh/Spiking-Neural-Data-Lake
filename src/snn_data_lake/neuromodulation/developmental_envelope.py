"""
v3.77 -- developmental envelope: a competence-gated neonatal->mature handoff.

The lake already MEASURES competence (brainstem/autonomous_loop.py `_competence()` -- the
fraction of a curriculum stage whose greedy learned action already matches the optimal one)
and STAGES a self-paced curriculum (neuromodulation/curriculum.py). What was missing is the
consumer: a single scalar "maturity" that reads competence and co-anneals the two knobs a
developing brain moves in lock-step --

  * endogenous EXPLORATION noise DOWN  -- neonatal motor/vestibular babbling is loud and
    narrows as the motor system calibrates. A puppy does not stop wobbling on a schedule;
    it stops wobbling because its cerebellum is calibrated. Maturation IS competence.
  * forward-model GAIN UP             -- the cerebellar internal (forward) model develops
    gradually. Neonates are FEEDBACK-dominated (react to what already happened); adults are
    PREDICTION-dominated (act on the forward model's estimate). Competence gates the handoff.

This is the human-brain-general developmental principle (Hadders-Algra 2018: motor development
is competence-driven, not clock-driven), not a species- or gait-specific schedule. It carries
no numpy/torch, no perturbation-direction physics, no arm geometry -- only the abstract envelope
math -- so it stays a pure-stdlib neuromodulator like its neighbours (cortisol.py, raphe_serotonin.py).

HOW IT WOULD WIRE (documented here; no other file is edited by this module):
  * autonomous_loop `_competence(tasks)` / develop()'s per-stage `competence` trace  -> update(competence).
      Feed the freshest stage competence each developmental batch so maturity tracks real mastery.
  * `explore_noise`        -> the arm's exploration/perturbation magnitude (the epsilon or babbling
      scale added on top of the gated step()). It anneals to ~0 as competence saturates, so
      exploration narrows with mastery -- and the brainstem reflex/E-STOP still preempts unchanged.
  * `forward_model_gain`   -> the weight on the JEPA rollout (predictive estimate) versus the
      reactive greedy selection. mix(reactive, predictive) IS the feedback->feedforward blend:
      at maturity 0 the action is the reactive one, at maturity 1 it is the JEPA-predicted one.

  python developmental_envelope.py
"""
import math
import random


class DevelopmentalEnvelope:
    """Competence -> maturity -> (exploration DOWN, forward-model gain UP).

    A leaky (EMA) integrator of a competence signal in [0,1]. `maturity` is that smoothed
    competence; the two properties are strictly monotone functions of it, so as competence
    rises exploration noise strictly falls and forward-model authority strictly rises."""

    def __init__(self, base_explore=0.3, ema=0.05, seed=0):
        # base_explore : the neonatal exploration/babbling scale (maturity 0).
        # ema          : smoothing factor for the competence integrator (small = slow, stable).
        self.base_explore = float(base_explore)
        self.ema = float(ema)
        self.maturity = 0.0                 # 0.0 = neonate (loud, reactive), 1.0 = mature (precise, predictive)
        self._rng = random.Random(seed)     # per-instance -> deterministic babbling stream

    def update(self, competence):
        """EMA the competence signal in [0,1] into `maturity` in [0,1]. Returns the new maturity.

        A convex combination of the old maturity and the (clamped) competence, so maturity stays
        in [0,1]. With a rising competence curve the EMA lags below it, so each step competence
        exceeds the current maturity and maturity strictly increases (the monotone guarantee)."""
        c = 0.0 if competence < 0.0 else 1.0 if competence > 1.0 else float(competence)
        self.maturity = (1.0 - self.ema) * self.maturity + self.ema * c
        return self.maturity

    @property
    def explore_noise(self):
        """Endogenous exploration/perturbation SCALE = base_explore * (1 - maturity).

        Neonatal at maturity 0 (== base_explore) and annealed to ~0 at maturity 1: infant motor
        and vestibular babbling that narrows as the motor system masters the task."""
        return self.base_explore * (1.0 - self.maturity)

    @property
    def forward_model_gain(self):
        """Forward-model (predictive) control authority = maturity.

        0 at neonate (feedback-dominated) ramping to 1 at maturity (prediction-dominated) as the
        cerebellar forward model calibrates -- the feedback->feedforward developmental handoff."""
        return self.maturity

    def mix(self, reactive, predictive):
        """The reactive->predictive handoff: (1 - gain) * reactive + gain * predictive.

        An action/estimate blend that starts as the pure reactive (feedback) value at neonate and
        shifts to the pure predictive (forward-model) value as competence -- and thus gain -- grows."""
        gain = self.forward_model_gain
        return (1.0 - gain) * reactive + gain * predictive

    def explore_sample(self):
        """One zero-mean endogenous exploration/babbling sample, scaled by explore_noise.

        The concrete perturbation the arm would add on top of its gated action. Drawn from the
        per-instance RNG (deterministic given seed); its scale anneals with competence, so the
        babbling amplitude narrows toward zero as the motor system matures."""
        return self._rng.gauss(0.0, 1.0) * self.explore_noise


def main():
    print("=" * 60)
    print("DEVELOPMENTAL ENVELOPE (competence-gated neonatal->mature)")
    print("=" * 60)

    # ---- (b) neonatal state: competence ~0 -> noisy + reactive ----------------
    env = DevelopmentalEnvelope(base_explore=0.3, ema=0.05, seed=0)
    neo_noise = env.explore_noise
    neo_gain = env.forward_model_gain
    neo_mix = env.mix(reactive=0.2, predictive=0.9)     # feedback vs forward-model action
    print(f"neonate  maturity={env.maturity:.3f}  explore_noise={neo_noise:.3f}  "
          f"forward_model_gain={neo_gain:.3f}  mix(0.2,0.9)={neo_mix:.3f}")

    # ---- (a) rising competence curve -> noise strictly DOWN, gain strictly UP --
    n = 60
    noises, gains, mixes = [], [], []
    for i in range(n):
        env.update((i + 1) / n)                          # strictly rising competence 1/n .. 1
        noises.append(env.explore_noise)
        gains.append(env.forward_model_gain)
        mixes.append(env.mix(reactive=0.0, predictive=1.0))   # == gain: reactive->predictive shift

    # ---- (b) mature state: drive competence=1 to saturation -> precise + predictive
    for _ in range(400):
        env.update(1.0)
    mat_noise = env.explore_noise
    mat_gain = env.forward_model_gain
    mat_mix = env.mix(reactive=0.2, predictive=0.9)
    mat_sample = env.explore_sample()
    print(f"mature   maturity={env.maturity:.3f}  explore_noise={mat_noise:.5f}  "
          f"forward_model_gain={mat_gain:.3f}  mix(0.2,0.9)={mat_mix:.3f}")
    print(f"babbling narrows: neonate |sample|~{env.base_explore:.2f} scale -> mature |sample|={abs(mat_sample):.5f}")

    # ---- (d) determinism: same seed -> identical babbling; different seed -> not
    def babble_stream(seed):
        e = DevelopmentalEnvelope(seed=seed)
        out = []
        for i in range(20):
            e.update((i + 1) / 20)
            out.append(e.explore_sample())
        return out
    stream_a = babble_stream(0)
    stream_b = babble_stream(0)
    stream_c = babble_stream(1)
    print(f"determinism: seed0==seed0 {stream_a == stream_b}, seed0!=seed1 {stream_a != stream_c}")
    print("=" * 60)

    # ---- self-checks ----------------------------------------------------------
    # (b) neonatal: explore_noise ~ base_explore, gain ~ 0
    assert math.isclose(neo_noise, env.base_explore, abs_tol=1e-9), "neonate explore_noise must equal base_explore"
    assert math.isclose(neo_gain, 0.0, abs_tol=1e-9), "neonate forward_model_gain must be ~0 (reactive)"

    # (a) monotone: as competence rises, explore_noise STRICTLY falls, gain STRICTLY rises
    for k in range(1, n):
        assert noises[k] < noises[k - 1], "explore_noise must strictly DECREASE as competence rises"
        assert gains[k] > gains[k - 1], "forward_model_gain must strictly INCREASE as competence rises"

    # (b) mature: explore_noise ~ 0, gain ~ 1
    assert mat_noise < 1e-3, "mature explore_noise must anneal toward 0 (precise)"
    assert mat_gain > 0.999, "mature forward_model_gain must saturate toward 1 (predictive)"
    assert abs(mat_sample) < 0.05, "mature babbling sample must be tiny (exploration narrowed)"

    # (c) mix shifts from reactive toward predictive as competence grows
    assert math.isclose(neo_mix, 0.2, abs_tol=1e-6), "neonate mix must be ~the reactive value"
    assert math.isclose(mat_mix, 0.9, abs_tol=1e-3), "mature mix must be ~the predictive value"
    for k in range(1, n):
        assert mixes[k] > mixes[k - 1], "mix(reactive,predictive) must shift toward predictive as competence rises"

    # (d) determinism
    assert stream_a == stream_b, "same seed must give identical babbling"
    assert stream_a != stream_c, "different seed must give different babbling (RNG is per-instance seeded)"

    print("self-check OK: competence anneals exploration DOWN and forward-model gain UP; "
          "neonate is noisy+reactive, mature is precise+predictive; mix hands feedback->feedforward; "
          "babbling is deterministic per seed")


if __name__ == "__main__":
    main()
