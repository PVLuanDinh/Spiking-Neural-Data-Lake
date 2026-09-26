"""
Efference copy / forward model -- action-conditioned sensory prediction that cancels self-generated
sensation (reafference) and flags external perturbations (exafference).

The reafference principle (von Holst & Mittelstaedt 1950) + internal forward models (Wolpert,
Ghahramani & Jordan, Science 269:1880, 1995; Sperry's corollary discharge): when the system acts,
an EFFERENCE COPY of the motor command u_t feeds a forward model M that predicts the next sensory
state s_hat_{t+1} = M(s_t, u_t). The actual feedback s_{t+1} is then compared to the prediction:

    s_hat_{t+1} = s_t + dt*(a_hat*s_t + b_hat*u_t)     (the model's internal prediction)
    e           = s_{t+1} - s_hat_{t+1}                 (the residual = the exafference signal)

  * SELF-GENERATED motion (the command drives the plant as the model expects): e ~ 0 -- the
    reafferent sensation is PREDICTED and CANCELLED (attenuated to the sensor-noise floor).
  * EXTERNAL perturbation (an unmodeled force moves the plant): e is large -- the exafferent
    sensation is FLAGGED as externally caused.

The plant is a self-contained abstract 1-D linear system s_{t+1} = s_t + dt*(a*s_t + b*u_t +
external_t) + noise, with its own true gains (a, b) and an `external` input channel. The forward
model holds its OWN estimates (a_hat, b_hat) and can LEARN them online by a delta / LMS rule so a
mismatched-gain model's prediction error decreases over trials of self-generated motion. It is a
READ-ONLY predictor + comparator: it emits NO motor command and drives no hardware.

DISTINCT from siblings in this repo (stated, having read them):
  * predictive_coding.py infers the latent CAUSES of a STATIC sensory input by top-down/bottom-up
    error over a generative hierarchy -- there is NO action and NO efference copy. THIS predicts the
    sensory consequence of the system's OWN command (efference copy -> forward model) and uses the
    residual to separate self-caused (reafference, cancelled) from externally-caused (exafference,
    flagged) sensation.
  * spiking_pid.py is feedback CONTROL to a setpoint -- it emits a corrective command. THIS emits
    no command; it is a read-only predictor / comparator, not a controller.
  * ring_attractor.py's ProprioceptiveEstimator dead-reckons commanded joint deltas into a state
    ANGLE (it integrates state). THIS predicts the sensory FEEDBACK of an action and cancels
    reafference / flags exafference -- a prediction+comparison, not state integration.
  * drift_monitor.py is statistical distribution drift -- not action-conditioned. THIS is
    conditioned on the efference copy of the command.

A reduced LINEAR forward-model + comparator on a synthetic abstract plant (delta-rule
online learning) -- the efference-copy / reafference level, NOT a spiking motor-control network,
not biophysics; not cognition.
Refs: von Holst & Mittelstaedt (1950); Wolpert, Ghahramani & Jordan, Science 269:1880 (1995).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.forward_model
"""

import random

from snn_data_lake.substrate._mathutils import mean

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "ForwardModel.update": (
        "LOCAL",
        "one online delta / LMS step on the one-step plant error. A delta rule on its own inputs and "
        "targets: no gradient crosses a layer boundary, so it owes no cosine",
    ),
}


class Plant:
    """The TRUE 1-D linear dynamics (unknown to the model): a leaky state driven by the command,
    an external perturbation channel, and zero-mean sensor noise.

        s_next = s + dt*(a*s + b*command + external) + noise

    Deterministic given seed (noise drawn from random.Random(seed)). This is the world the forward
    model tries to predict; the model never sees a, b, external, or the noise."""

    def __init__(self, a, b, noise=0.0, dt=1.0, seed=0):
        self.a = float(a)
        self.b = float(b)
        self.noise = float(noise)
        self.dt = float(dt)
        self._rng = random.Random(seed)
        self.s = 0.0

    def reset(self, s0=0.0):
        """Set the plant state (does not touch the noise stream)."""
        self.s = float(s0)
        return self.s

    def step(self, command, external=0.0):
        """Advance one step under `command` plus an optional unmodeled `external` input. Returns the
        new true state s_next."""
        n = self._rng.uniform(-self.noise, self.noise) if self.noise else 0.0
        self.s = self.s + self.dt * (self.a * self.s + self.b * command + external) + n
        return self.s


class ForwardModel:
    """Internal forward model + comparator (read-only). Holds its OWN gain estimates a_hat, b_hat
    and predicts the next sensory state from the current state and the efference copy of the command.

    predict()  -> s_hat = s + dt*(a_hat*s + b_hat*u)          (action-conditioned sensory prediction)
    residual() -> e = s_next - s_hat                          (the exafference signal)
    detect()   -> |e| > thresh                                (external perturbation flagged?)
    update()   -> one delta / LMS step adapting a_hat, b_hat  (online learning of the plant)

    Deterministic and stateless w.r.t. randomness; `seed` is kept for API symmetry with Plant and
    reproducible extensions. Emits no command and drives no hardware."""

    def __init__(self, a, b, dt=1.0, seed=0):
        self.a_hat = float(a)
        self.b_hat = float(b)
        self.dt = float(dt)
        self.seed = seed  # deterministic predictor; retained for API symmetry, not sampled

    def predict(self, state, command):
        """Action-conditioned sensory prediction s_hat from the efference copy of `command`."""
        return state + self.dt * (self.a_hat * state + self.b_hat * command)

    def residual(self, state, command, next_state):
        """Prediction error e = s_next - s_hat -- the exafference signal (0 => reafference cancelled)."""
        return next_state - self.predict(state, command)

    def detect(self, state, command, next_state, thresh):
        """True if the residual exceeds `thresh` -- i.e. the sensation is flagged as externally caused."""
        return abs(self.residual(state, command, next_state)) > thresh

    def update(self, state, command, next_state, lr):
        """One online delta / LMS step on the one-step error, adapting a_hat, b_hat toward the plant.

        For E = 0.5*e^2 with e = s_next - (s + dt*(a_hat*s + b_hat*u)), the regressors are dt*s and
        dt*u, so gradient descent gives a_hat += lr*e*dt*s ; b_hat += lr*e*dt*u. Returns the pre-update
        residual e."""
        e = self.residual(state, command, next_state)
        self.a_hat += lr * e * self.dt * state
        self.b_hat += lr * e * self.dt * command
        return e


def _exafference_trace(a, b, dt, noise, seed, commands, perturb_step, external):
    """Run a matched forward model over a plant driven by `commands`, injecting `external` at
    `perturb_step`. Returns the per-step residual trace e (used for detection + determinism checks)."""
    plant = Plant(a, b, noise=noise, dt=dt, seed=seed)
    plant.reset(0.0)
    fm = ForwardModel(a, b, dt=dt, seed=0)
    trace = []
    for t, u in enumerate(commands):
        s = plant.s
        ext = external if t == perturb_step else 0.0
        nxt = plant.step(u, external=ext)
        trace.append(fm.residual(s, u, nxt))
    return trace


def main():
    a_true, b_true, dt = -0.3, 0.8, 0.1

    # ---- 1. REAFFERENCE cancellation: matched model + plant, KNOWN commands, no external ----
    # e_t reduces to the pure sensor noise of that step, so the mean cancels to ~0: the
    # self-generated (reafferent) sensation is predicted and cancelled up to zero-mean noise.
    plant1 = Plant(a_true, b_true, noise=0.01, dt=dt, seed=1)
    plant1.reset(0.0)
    fm1 = ForwardModel(a_true, b_true, dt=dt, seed=0)
    rng_u = random.Random(7)
    err1 = []
    for _ in range(200):
        s = plant1.s
        u = rng_u.uniform(-1.0, 1.0)
        nxt = plant1.step(u, external=0.0)
        err1.append(fm1.residual(s, u, nxt))
    mean_self = mean(err1)
    mean_abs_self = mean([abs(v) for v in err1])

    # ---- 2. EXAFFERENCE detection: inject an unmodeled external displacement at a KNOWN step ----
    thresh = 0.05
    n2 = 60
    perturb_step, external = 30, 2.0
    cmds2 = [random.Random(11).uniform(-1.0, 1.0) for _ in range(n2)]  # fixed command sequence
    plant2 = Plant(a_true, b_true, noise=0.005, dt=dt, seed=2)
    plant2.reset(0.0)
    fm2 = ForwardModel(a_true, b_true, dt=dt, seed=0)
    err2, flags2 = [], []
    for t, u in enumerate(cmds2):
        s = plant2.s
        ext = external if t == perturb_step else 0.0
        nxt = plant2.step(u, external=ext)
        err2.append(fm2.residual(s, u, nxt))
        flags2.append(fm2.detect(s, u, nxt, thresh))
    perturbed_err = abs(err2[perturb_step])
    baseline_err = max(abs(v) for t, v in enumerate(err2) if t != perturb_step)
    detected_steps = [t for t, f in enumerate(flags2) if f]

    # ---- 3. DISCRIMINATION: SAME sensory change, two ways -- command vs equal external ----
    # Same delta-s; only the command carries an efference copy, so only its sensation is cancelled.
    s0, u_d = 0.4, 0.7
    plant_self = Plant(a_true, b_true, noise=0.0, dt=dt, seed=0)
    plant_self.reset(s0)
    nxt_self = plant_self.step(command=u_d, external=0.0)  # self-generated
    delta_self = nxt_self - s0
    e_self = fm2.residual(s0, u_d, nxt_self)  # efference copy present -> ~0
    det_self = fm2.detect(s0, u_d, nxt_self, thresh)

    plant_ext = Plant(a_true, b_true, noise=0.0, dt=dt, seed=0)
    plant_ext.reset(s0)
    ext_equiv = b_true * u_d  # dt*external == dt*b*u => equal ds
    nxt_ext = plant_ext.step(command=0.0, external=ext_equiv)  # external, zero command
    delta_ext = nxt_ext - s0
    e_ext = fm2.residual(s0, 0.0, nxt_ext)  # no efference copy -> large
    det_ext = fm2.detect(s0, 0.0, nxt_ext, thresh)

    # ---- 4. ONLINE ADAPTATION: start with WRONG gains, learn the plant from self-generated motion ----
    fm_l = ForwardModel(a=a_true + 0.6, b=b_true - 0.5, dt=dt, seed=0)  # mismatched a_hat, b_hat
    plant_l = Plant(a_true, b_true, noise=0.0, dt=dt, seed=0)
    plant_l.reset(0.0)
    rng_l = random.Random(5)
    abs_err_l, lr = [], 40.0
    for _ in range(500):
        s = plant_l.s
        u = rng_l.uniform(-1.0, 1.0)
        nxt = plant_l.step(u, external=0.0)
        abs_err_l.append(abs(fm_l.update(s, u, nxt, lr)))
    first_err = mean(abs_err_l[:20])
    last_err = mean(abs_err_l[-20:])

    # ---- 5. DETERMINISM: same params/seed -> identical residual traces from two fresh instances ----
    trace_a = _exafference_trace(a_true, b_true, dt, 0.005, 2, cmds2, perturb_step, external)
    trace_b = _exafference_trace(a_true, b_true, dt, 0.005, 2, cmds2, perturb_step, external)
    same_trace = trace_a == trace_b

    print("=" * 70)
    print("FORWARD MODEL -- efference copy: cancel reafference, flag exafference")
    print("=" * 70)
    print(
        f"reafference : mean self-generated e {mean_self:+.2e} (mean |e| {mean_abs_self:.2e} = "
        f"sensor-noise floor) -> cancelled"
    )
    print(
        f"exafference : perturbed-step |e| {perturbed_err:.4f} vs baseline max |e| {baseline_err:.4f} "
        f"(thresh {thresh}); flagged steps {detected_steps}"
    )
    print(
        f"discriminate: same delta-s self {delta_self:+.4f} / external {delta_ext:+.4f}; "
        f"e_self {e_self:+.2e} (detect {det_self}) vs e_ext {e_ext:+.4f} (detect {det_ext})"
    )
    print(
        f"learning    : mean |e| first-20 {first_err:.4f} -> last-20 {last_err:.6f} "
        f"({first_err / last_err:.0f}x lower); a_hat {fm_l.a_hat:+.3f} (a {a_true}) "
        f"b_hat {fm_l.b_hat:+.3f} (b {b_true})"
    )
    print(f"determinism : identical residual traces {same_trace}")
    print("=" * 70)

    # ---- self-check: deterministic asserts, ASCII-only prints ----
    assert abs(mean_self) < 5e-3, (
        f"matched model must cancel reafference: mean self-generated e ~0 ({mean_self:.2e})"
    )
    assert mean_abs_self > 1e-4, (
        f"there must be real sensor noise being cancelled, not a trivial zero ({mean_abs_self:.2e})"
    )
    assert perturbed_err > thresh, (
        f"an external perturbation must spike the error above thresh ({perturbed_err:.4f})"
    )
    assert baseline_err < thresh, f"non-perturbed steps must stay below thresh ({baseline_err:.4f})"
    assert detected_steps == [perturb_step], (
        f"detect must flag exactly the perturbed step ({detected_steps})"
    )
    assert perturbed_err > 10 * baseline_err, (
        f"exafference must stand well clear of the reafference floor ({perturbed_err:.4f} vs {baseline_err:.4f})"
    )
    assert abs(delta_self - delta_ext) < 1e-9, (
        f"the two paths must produce the SAME sensory change ({delta_self:.4f} vs {delta_ext:.4f})"
    )
    assert abs(e_self) < thresh and not det_self, (
        f"the self-generated (commanded) change must be cancelled, not flagged (e {e_self:.2e})"
    )
    assert abs(e_ext) > thresh and det_ext, (
        f"the equal EXTERNAL change must be flagged as exafference (e {e_ext:.4f})"
    )
    assert last_err < first_err * 0.1, (
        f"online learning must cut the reafference error clearly ({first_err:.4f} -> {last_err:.6f})"
    )
    assert abs(fm_l.a_hat - a_true) < 0.05 and abs(fm_l.b_hat - b_true) < 0.05, (
        f"the model must converge to the true plant gains (a_hat {fm_l.a_hat}, b_hat {fm_l.b_hat})"
    )
    assert same_trace, "same params/seed must give identical residual traces"
    print(
        f"self-check OK: reafference cancelled (mean e {mean_self:.1e}); exafference flagged "
        f"(|e| {perturbed_err:.3f} vs floor {baseline_err:.3f}); same delta-s {delta_self:.3f} "
        f"read as self (e {e_self:.1e}) vs external (e {e_ext:.3f}); learning "
        f"{first_err:.3f} -> {last_err:.4f}; deterministic"
    )


if __name__ == "__main__":
    main()
