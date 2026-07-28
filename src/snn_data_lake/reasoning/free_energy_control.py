"""
Free-energy spiking control — a spike fires only if it reduces prediction error (scan #14).

The robot loop's control is a NEF spiking-PID + spiking-MBRL/MPC; neither has a predictive-coding
controller or a principled sparsity/robustness objective. This adds one: candidate control neurons each
propose a small action, but a neuron only SPIKES (applies its action) if doing so lowers the free energy
F = (x − target)² by more than a dead-band margin. Consequences: the controller goes SILENT near the
setpoint (event-driven sparsity — no spikes when none are needed), and the margin gate rejects
noise-driven twitching, so it degrades gracefully under sensor noise where an always-act (bang-bang)
controller jitters every step. Refs: arXiv:2603.09729 (2026, "Efficient and robust control with spikes
that constrain free energy").

  Run: python free_energy_control.py       # (also run by run_selfchecks.py)
"""

import random

random.seed(0)

TARGET = 1.0
STEPS = 60
ACTIONS = (0.2, -0.2, 0.05, -0.05)  # candidate control-neuron actions
DEADBAND = 0.03  # precision: prediction error below this = no spike (silence)
NOISE = 0.02  # sensor noise (< DEADBAND, so it is rejected near the setpoint)


def free_energy(x):
    return (x - TARGET) ** 2


def fe_step(x_measured, x_true):
    """Predictive-coding gate: if the measured prediction error is within precision (DEADBAND) the
    neuron stays SILENT; otherwise the action that most reduces free energy fires and is applied to
    the true state. Returns (new_x_true, fired)."""
    if abs(x_measured - TARGET) <= DEADBAND:
        return x_true, False  # error within precision -> no spike (event-driven silence)
    f0 = free_energy(x_measured)
    best_d, best_f = None, f0
    for d in ACTIONS:
        f = free_energy(x_measured + d)
        if f < best_f:
            best_f, best_d = f, d
    if best_d is None:
        return x_true, False
    return x_true + best_d, True


def run_fe(noise, seed):
    rng = random.Random(seed)
    x, spikes, tail = 0.0, 0, []
    for step in range(STEPS):
        x_meas = x + (rng.uniform(-noise, noise) if noise else 0.0)
        x, fired = fe_step(x_meas, x)
        spikes += fired
        if step >= STEPS - 12:
            tail.append(abs(x - TARGET))
    return spikes, sum(tail) / len(tail)  # (spike count, mean |error| over the last 12 steps)


def run_bangbang(noise, seed):
    """Baseline: always act toward the setpoint (fires every step) — never silent, jitters on noise."""
    rng = random.Random(seed)
    x, spikes, tail = 0.0, 0, []
    for step in range(STEPS):
        x_meas = x + (rng.uniform(-noise, noise) if noise else 0.0)
        x += 0.05 if x_meas < TARGET else -0.05
        spikes += 1
        if step >= STEPS - 12:
            tail.append(abs(x - TARGET))
    return spikes, sum(tail) / len(tail)


def main():
    fe_sp, fe_err = run_fe(0.0, 1)
    fe_sp_n, fe_err_n = run_fe(NOISE, 1)  # under sensor noise
    bb_sp_n, bb_err_n = run_bangbang(NOISE, 1)

    print("=" * 60)
    print("FREE-ENERGY SPIKING CONTROL (spike only if it lowers prediction error)")
    print("=" * 60)
    print(f"target={TARGET}  steps={STEPS}")
    print(f"  clean : final error {fe_err:.3f}  spikes {fe_sp}/{STEPS}  (silent near setpoint)")
    print(f"  noisy : free-energy  error {fe_err_n:.3f}  spikes {fe_sp_n}/{STEPS}")
    print(f"          bang-bang    error {bb_err_n:.3f}  spikes {bb_sp_n}/{STEPS}  (always acts)")
    print("free-energy gating -> sparse spikes + graceful degradation the PID/MPC path lacks.")
    print("=" * 60)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert fe_err < 0.05, f"free-energy controller should converge to the setpoint: {fe_err:.3f}"
    assert fe_sp < STEPS, f"should be sparse (silent near setpoint), not fire every step: {fe_sp}"
    assert fe_sp < bb_sp_n, (
        f"free-energy controller should spike less than always-act ({fe_sp} vs {bb_sp_n})"
    )
    assert fe_err_n < bb_err_n, (
        f"free-energy should track more steadily under noise ({fe_err_n:.3f} vs {bb_err_n:.3f})"
    )
    print(
        f"self-check OK: converges (err {fe_err:.3f}), sparse ({fe_sp}/{STEPS} spikes), and steadier "
        f"than bang-bang under noise ({fe_err_n:.3f} < {bb_err_n:.3f})"
    )


if __name__ == "__main__":
    main()
