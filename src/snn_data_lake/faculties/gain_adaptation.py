"""
Flocculus-style adaptive feedforward gain (and SIGN) recalibration -- online LMS / delta-rule
calibration of a single INVOLUNTARY compensatory gain from the residual sensory error (retinal slip
/ reach error) that REMAINS after the response, with the defining property that it can FLIP SIGN
when the world reverses (reversing prism, reversed VOR).

The vestibulo-ocular reflex (VOR) rotates the eyes opposite to head motion to hold gaze stable; its
gain g maps head velocity (the command u) to a compensatory eye velocity (the involuntary response
g*u). The cerebellar flocculus CALIBRATES g so the response cancels what the world requires -- when
gaze still slips across the retina after the reflex has fired, that residual slip is the teaching
signal carried by climbing fibers to Purkinje cells, and flocculus LTD/LTP nudges g until the slip
vanishes. The same signed calibration runs for saccadic gain (post-saccadic target error) and for
reach / prism adaptation (visual displacement of the seen hand). Refs: Ito, Annu Rev Neurosci 5:275
(1982); Miles & Lisberger, Annu Rev Neurosci 4:273 (1981) -- VOR PHASE INVERSION, the reflex can
reverse SIGN; Boyden, Katoh & Raymond, Annu Rev Neurosci 27:581 (2004); Raymond & Lisberger,
J Neurosci 18:9112 (1998).

Let g be the feedforward gain, u the command, W the world's TRUE required gain (unknown to the
adapter):

    response = g*u                       (the involuntary compensatory response the reflex emits)
    required = W*u                        (what the world actually needed on this step)
    error    = required - response        (= (W-g)*u : the residual retinal slip / reach error)
    g       += lr * error * u             (delta / climbing-fiber teaching, only if |error| >= dead)

Because dg = lr*(W-g)*u^2 and u^2 >= 0, g is ALWAYS driven toward W regardless of the sign of W: a
positive W is tracked up, a NEGATIVE W is tracked DOWN through zero to -|W|. That zero-crossing is
the flocculus / reversed-VOR signature -- a magnitude-only learner (one that adapts |g| or scales
multiplicatively) can weaken an over-strong reflex but can NEVER invert it. A dead-zone drops
sub-`dead` residuals so the gain does not chase sensor noise, and g is clipped to +/-clip.

DISTINCT from siblings in this repo (stated, having read them):
  * forward_model.py's ForwardModel is a READ-ONLY predictor / comparator: it learns the plant's own
    gains (a_hat, b_hat) to PREDICT the sensory consequence of a command and flag exafference. THIS
    emits an involuntary compensatory RESPONSE (g*u) and recalibrates the single gain g that
    produces it from the residual slip -- an adaptive reflex, not a predictor, and it can flip sign.
  * spiking_pid.py is closed-loop feedback to a setpoint (P+I+D on the running error, emitting a
    correction every step). THIS is a FEEDFORWARD gain: one multiply g*u with no feedback loop,
    whose scalar gain is slowly recalibrated by the residual left AFTER the response.
  * cortex/readout.py and the plasticity delta rules fit input->output WEIGHT VECTORS to a
    supervised target; THIS calibrates one scalar involuntary gain and, crucially, tracks a full
    SIGN REVERSAL of the world (not just magnitude).

A reduced scalar LMS gain-calibrator for an involuntary feedforward reflex (VOR / saccade / reach),
the flocculus gain-adaptation level -- NOT a spiking Purkinje network, not biophysics, not cognition.
Refs: Ito (1982); Miles & Lisberger (1981); Boyden, Katoh & Raymond (2004).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.gain_adaptation
"""

import random

from snn_data_lake.infra._mathutils import mean


class GainAdapter:
    """Adaptive feedforward gain (and sign) for one involuntary reflex channel.

    Holds a single scalar gain g mapping command -> compensatory response (g*command). Each step it
    measures the residual error left after the response (required - response = the sensory slip) and
    takes one delta / LMS step g += lr*error*command, so g converges to the world's true required
    gain and TRACKS it across a sign reversal, crossing zero. Errors smaller than `dead` in magnitude
    are ignored (dead-zone: do not chase sensor noise) and g is clipped to +/-clip. Deterministic;
    random.Random(seed) is used only for the optional sensor noise on the measured error."""

    def __init__(self, g0=0.0, lr=0.02, dead=0.05, clip=3.0, seed=0):
        self.g = float(g0)
        self.lr = float(lr)
        self.dead = float(dead)
        self.clip = float(clip)
        self._rng = random.Random(seed)

    def step(self, command, world_gain, noise=0.0):
        """One recalibration step for `command` in a world requiring gain `world_gain`.

        Emits the involuntary response g*command, measures the residual slip error = required -
        response = (world_gain - g)*command (plus optional zero-mean sensor noise of half-width
        `noise`), and -- unless |error| < dead -- nudges g by lr*error*command (clipped to +/-clip).
        Returns (response, error, g) with g the post-update gain."""
        response = self.g * command
        required = world_gain * command
        error = required - response
        if noise:
            error += self._rng.uniform(-noise, noise)
        if abs(error) >= self.dead:
            self.g += self.lr * error * command
            if self.g > self.clip:
                self.g = self.clip
            elif self.g < -self.clip:
                self.g = -self.clip
        return response, error, self.g


def _run(adapter, world_gain, commands, noise=0.0):
    """Drive `adapter` through `commands` in a world of true gain `world_gain`. Returns (g_trace,
    abs_error_trace) -- the gain after each step and the magnitude of the residual slip it saw."""
    gs, errs = [], []
    for u in commands:
        _, e, g = adapter.step(u, world_gain, noise=noise)
        gs.append(g)
        errs.append(abs(e))
    return gs, errs


def _noisy_reversal(seed, commands, w_pos, w_neg, noise):
    """A full converge-then-reverse run under sub-dead sensor noise from random.Random(seed).
    Returns (reversal_g_trace, final_gain). Used for the determinism + noise-robustness check."""
    ad = GainAdapter(g0=0.0, lr=0.02, dead=0.05, clip=3.0, seed=seed)
    _run(ad, w_pos, commands, noise=noise)  # converge to +W
    rev, _ = _run(ad, w_neg, commands, noise=noise)  # world reverses -> track to -W
    return rev, ad.g


def main():
    # Deterministic head-velocity-like command stream (fixed magnitude range, fixed seed).
    rng_cmd = random.Random(3)
    cmds = [rng_cmd.uniform(-1.5, 1.5) for _ in range(2000)]

    W_POS, W_NEG = 1.5, -1.5

    # ---- 1. MAGNITUDE convergence: fixed positive world gain, g learned from slip, g0 = 0 --------
    a1 = GainAdapter(g0=0.0, lr=0.02, dead=0.05, clip=3.0, seed=0)
    g1_trace, err1 = _run(a1, W_POS, cmds)
    g1_final = a1.g
    first_slip = mean(err1[:50])
    last_slip = mean(err1[-50:])

    # ---- 2. SIGN REVERSAL: converge to +W, flip the world to -W, must cross zero to -W -----------
    a2 = GainAdapter(g0=0.0, lr=0.02, dead=0.05, clip=3.0, seed=0)
    _run(a2, W_POS, cmds)  # first calibrate to +1.5
    g_before = a2.g
    rev_trace, _ = _run(a2, W_NEG, cmds)  # reversing prism / reversed VOR
    g_after = a2.g
    saw_pos = any(g > 0.0 for g in rev_trace)
    saw_neg = any(g < 0.0 for g in rev_trace)
    min_abs_g = min(abs(g) for g in rev_trace)  # closest approach to zero during the crossing

    # ---- 3. DEAD-ZONE: sub-`dead` residuals are ignored (frozen) vs supra-`dead` residuals move ---
    # frozen: g0 only 0.02 from W, so |error| = 0.02*|u| <= 0.03 < dead for every command -> no move
    a_dz = GainAdapter(g0=0.98, lr=0.02, dead=0.05, clip=3.0, seed=0)
    _run(a_dz, 1.0, cmds)
    g_frozen = a_dz.g
    # moves: g0 far enough that |error| exceeds dead -> g must climb toward W (then freeze in-band)
    a_mv = GainAdapter(g0=0.85, lr=0.02, dead=0.05, clip=3.0, seed=0)
    _run(a_mv, 1.0, cmds)
    g_moved = a_mv.g

    # ---- 4. DETERMINISM (+ sub-dead noise robustness): same seed -> identical trace, and it still
    #         converges to -W because the dead-zone rejects the sub-threshold sensor noise ----------
    NOISE = 0.03  # < dead = 0.05: pure noise alone can never trip an update
    rev_a, g_noise_a = _noisy_reversal(0, cmds, W_POS, W_NEG, NOISE)
    rev_b, g_noise_b = _noisy_reversal(0, cmds, W_POS, W_NEG, NOISE)
    rev_c, _ = _noisy_reversal(1, cmds, W_POS, W_NEG, NOISE)
    same_seed_same_trace = rev_a == rev_b
    diff_seed_diff_trace = rev_a != rev_c

    print("=" * 70)
    print("GAIN ADAPTATION -- flocculus feedforward gain+SIGN recalibration (VOR/reach)")
    print("=" * 70)
    print(
        f"magnitude  : g0 0.000 -> g {g1_final:+.4f} (world {W_POS:+.3f}); "
        f"slip |e| first-50 {first_slip:.4f} -> last-50 {last_slip:.4f}"
    )
    print(
        f"sign flip  : converged +{g_before:.3f}, world reversed to {W_NEG:+.3f} -> g {g_after:+.4f}; "
        f"crossed zero (min |g| {min_abs_g:.4f}, saw +={saw_pos} -={saw_neg})"
    )
    print(
        f"dead-zone  : sub-dead error frozen g {g_frozen:.4f} (=g0 0.98); "
        f"supra-dead error moved g0 0.85 -> {g_moved:.4f}"
    )
    print(
        f"determinism: same-seed identical trace {same_seed_same_trace}, diff-seed differs "
        f"{diff_seed_diff_trace}; noisy g {g_noise_a:+.4f} -> world {W_NEG:+.3f}"
    )
    print("=" * 70)

    # ---- self-check: deterministic asserts, ASCII-only prints -------------------
    assert abs(g1_final - W_POS) < 0.1, (
        f"magnitude: g must converge to the positive world gain from g0=0 (g {g1_final}, W {W_POS})"
    )
    assert last_slip < first_slip * 0.1, (
        f"magnitude: recalibration must drive the residual slip down (|e| {first_slip} -> {last_slip})"
    )
    assert g_before > 0.9 * W_POS, (
        f"sign: must first calibrate clearly to +W before the reversal (g_before {g_before})"
    )
    assert abs(g_after - W_NEG) < 0.1, (
        f"sign: after the world reverses, g must track to the NEGATIVE world gain (g {g_after}, W {W_NEG})"
    )
    assert saw_pos and saw_neg, (
        f"sign: g must pass from positive to negative (saw + {saw_pos}, saw - {saw_neg})"
    )
    assert min_abs_g < 0.1, (
        f"sign: g must CROSS zero, not jump -- min |g| during reversal {min_abs_g} (flocculus signature)"
    )
    assert g_frozen == 0.98, (
        f"dead-zone: a sub-`dead` residual must NOT move the gain at all (g {g_frozen}, expected 0.98)"
    )
    assert g_moved > 0.90, (
        f"dead-zone: a supra-`dead` residual MUST move the gain toward W (g {g_moved} from g0 0.85)"
    )
    assert same_seed_same_trace, "determinism: same seed must give a byte-identical g-trace"
    assert diff_seed_diff_trace, "determinism: a different seed must change the (noisy) g-trace"
    assert abs(g_noise_a - W_NEG) < 0.2, (
        f"robustness: sign reversal must still converge to -W under sub-dead noise (g {g_noise_a}, W {W_NEG})"
    )
    print(
        f"self-check OK: gain {g1_final:+.3f}->world {W_POS:+.3f} (slip {first_slip:.3f}->{last_slip:.4f}); "
        f"SIGN reversal +{g_before:.2f}->{g_after:+.3f} crossing zero (min |g| {min_abs_g:.3f}); "
        f"dead-zone freezes at {g_frozen:.2f}; deterministic; noisy->{g_noise_a:+.3f}"
    )


if __name__ == "__main__":
    main()
