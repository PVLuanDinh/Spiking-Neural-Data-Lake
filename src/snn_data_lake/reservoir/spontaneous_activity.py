"""
Spontaneous-activity bootstrap -- a learnable OU process drives the percept path PRE-EXPERIENCE
through the SAME plasticity rule real percepts take (specs/022 T045).

SCOPE, STATED EXACTLY (T045 panel). "Drives sensors" in the task name is delivered at the
PERCEPT-SPACE CONSUMER, not at a sensor: the drive is synthesized directly in percept space
([0,1]^dim) and injected where the percept lands -- the actual sensor-side transduction
(event frames, the substrate stimulus port, the brainstem loop) is specs/014's unbuilt
boundary, and this module does not touch it. Likewise FR-033's second clause ("grown units
receive pre-experience structure from the spontaneous-activity bootstrap") is NOT discharged
here: the bootstrap reaches PerceptSurprise's FIXED set of Oja neurons; no grown unit receives
anything from this module today, and no task id names that wire yet -- both stated rather
than implied away.

WHAT THIS IS. Before a brain has seen anything, its perception circuitry is not blank in
biology: retinal waves -- structured, spatially correlated spontaneous activity -- drive the
same plasticity rules that experience will use later, so the first real stimulus arrives at an
organized circuit. This module is that bootstrap for the percept path: an Ornstein-Uhlenbeck
process on a small latent, projected through declared smooth spatial modes onto the percept
dimension, fed step by step into `cortex/percept_surprise.PerceptSurprise.observe` -- the SAME
method, the same Oja/Sanger rule, the same instance that live perception drives from
`Brain._perceive_real` (the wire sits at that method's lazy-init, BEFORE the first real percept
reaches the rule, and runs ONLY where the torch perception tier is armed: the eprop checkpoint
is gitignored, so a fresh clone and CI never fire the wire -- the tests pin the code path with
a stub encoder, not the deployment).

WHY THE SPATIAL MODES EXIST (the design decision that matters). A per-dimension independent OU
carries temporal structure but an ISOTROPIC spatial covariance -- Oja-PCA learns covariance
directions, so an independent drive would organize NOTHING measurable: the bootstrap would be
motion without effect. Retinal waves are spatially correlated; the analog here is an OU latent
(k dims) driven through low-frequency cosine modes over the percept index. THE MODES ARE A
DECLARED CHOICE, NOT A MEASUREMENT: the percept latent has no measured topology, so what the
self-check proves is that the bootstrap INSTALLS THE DECLARED STRUCTURE through the real
plasticity path (mode-span alignment vs a cold twin and a permuted-modes null) -- never that
this structure is the right one for real percepts. Whether it helps real perception is the
eventual A/B's question, and the mode basis is one of its variables.

WHAT "LEARNABLE" MEANS HERE, exactly. `fit_to_traces` fits theta -- the ONE parameter --
from REAL sensor traces by closed-form AR(1) moment matching on the mode-projected latents
(lag-1 autocorrelation, which is scale-invariant and so sees through the drive's own range
normalization). Analytical, no iteration, no gradient (GRADIENT_CLAIM: DIRECT_SOLVE). It
REFUSES a trace with no usable positive temporal correlation rather than fitting a made-up
theta to white noise -- the refusal, not a default, is what keeps "learnable" from meaning
"always returns numbers". There is deliberately NO amplitude parameter: the drive pins its
amplitude to the percept's own [0,1] range (the way the real percept is normalized), and the
first draft's own recovery red PROVED the consequence -- its sigma came back as gain * sigma
exactly, a knob the output could not feel, so it was deleted rather than decorated. The
initial theta is PICKED and says so; it stands only until the first real trace fit.

HONESTY NOTES CARRIED IN THE CHECKS. The percept is nonnegative ([0,1] normalized spike
counts), so the drive rides on a 0.5 offset and PerceptSurprise's L2 normalization makes the
DC direction genuinely dominant -- faithful to real percepts, and the alignment measurement
projects DC out rather than pretending it is signal. The A/B ships with a matched A/A control
(two same-seed bootstrapped twins, bit-identical) so a nonzero A/B cannot be harness noise
(the 007 C3 pattern). THE BASELINE IS LEFT COLD: observe() also feeds the surprise
standardizer, and the panel measured the un-reset bootstrap handing the first real percept a
saturated surprise (1.000 vs the documented cold 0.000) on an action-gate input --
bootstrap_percept_rule resets `_mu`/`_var` so the ONLY installed effect is subspace
structure; deliberately seeding the baseline was A/B-REFUTED (see bootstrap_percept_rule:
de-saturation bought by a 5.4x novelty-margin collapse -- the reset stays). THE
OU'S COST IS MEASURED AND KEPT OPENLY: Oja is order-blind, so on the alignment criterion an
iid latent does at least as well (criterion 3b ships the arm); the OU exists for the
temporally-sensitive rules the arc lands next, and deletes per P8 clause 5 if none
materializes. NULL DISPOSITION (P8 clause 5): if the eventual real-perception A/B shows the
bootstrap buys the first real percepts nothing, this module is DELETED, not parked.

  PYTHONPATH=src python -m snn_data_lake.reservoir.spontaneous_activity
"""

import math

import numpy as np

#: Initial OU rate -- PICKED, disclosed, standing only until the first fit_to_traces on a
#: real sensor trace replaces it (that replacement is the "learnable" in the task name).
#: THETA IS THE ONLY PARAMETER, and that is a MEASURED design fact, not a simplification: the
#: first draft carried a sigma, and its own recovery red returned sigma_hat = gain * sigma
#: exactly -- the range-normalizing drive gain cancels the latent amplitude, so sigma was a
#: dead knob whose every value produced the identical drive. A parameter the output cannot
#: feel is not learnable; it was deleted rather than kept as decoration.
THETA0 = 0.5


def _cosine_modes(dim, k):
    """(dim, k) low-frequency cosine basis over the percept index, columns unit-normalized --
    the DECLARED spatial-correlation choice (DCT-II shapes). A declared prior, not a measured
    topology; the eventual real-perception A/B treats it as a variable."""
    j = np.arange(dim, dtype=np.float64) + 0.5
    B = np.stack([np.cos(math.pi * (m + 1) * j / dim) for m in range(k)], axis=1)
    return B / np.linalg.norm(B, axis=0, keepdims=True)


# distinct-from: instruments/episode_driver.py -- its tick() is a faithful TRANSDUCER of caller
#   stimuli onto the kernel's fixed-point grid (episode replay); this GENERATES stimuli from
#   an OU process for a plasticity rule. Site read; the shared word is "stimulus".
# distinct-from: sleep/episode_consolidation.py -- kWTA activity budget; unrelated. Read.
# distinct-from: sleep/sleep_replay.py -- the nearest FUNCTIONAL neighbour (T045 panel):
#   noise-driven spontaneous REACTIVATION of STORED assemblies feeding consolidation, i.e.
#   post-experience replay of learned content; this is pre-experience bootstrap of an EMPTY
#   rule. Site read; different phase of life, no shared code.
class SpontaneousActivity:
    """The learnable OU drive: a unit-variance exact-transition OU latent (k dims, rate
    theta -- the one parameter), projected through the declared modes onto the percept
    dimension and pinned into the percept's own [0,1] range. Deterministic per instance
    (seeded generator)."""

    def __init__(self, dim, k=4, seed=0, theta=THETA0, dt=1.0):
        # k < dim strictly: the dim-th DCT-II column is identically zero in exact arithmetic,
        # and normalizing it promotes float noise to a full unit vector (the T045 panel
        # measured orthonormality error 0.61 at k == dim) -- refused, never normalized.
        if k <= 0 or dim <= 0 or k >= dim:
            raise ValueError(f"need 0 < k < dim, got k={k}, dim={dim}")
        # theta and dt must be finite and positive: theta=0 or dt=0 is a FROZEN constant
        # drive (1600 identical DC vectors through the plasticity rule -- motion without
        # effect, silently); negative or NaN values fail later and further from the cause.
        if not (math.isfinite(theta) and theta > 0.0):
            raise ValueError(f"need finite theta > 0, got {theta!r}")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"need finite dt > 0, got {dt!r}")
        self.dim, self.k, self.dt = int(dim), int(k), float(dt)
        self.theta = float(theta)
        self.modes = _cosine_modes(self.dim, self.k)
        self._rng = np.random.default_rng(seed)
        self._z = np.zeros(self.k, dtype=np.float64)

    def set_modes(self, B):
        """Replace the spatial basis -- the ONE validated write surface (the panel measured a
        raw `self.modes = Q` assignment failing deep inside matmul on a shape mismatch, and a
        matching-shape assignment silently bypassing orthonormality). Requires (dim, k) shape
        and orthonormal columns; DC-orthogonality is the caller's choice, disclosed."""
        B = np.asarray(B, dtype=np.float64)
        if B.shape != (self.dim, self.k):
            raise ValueError(f"modes shape {B.shape} does not match (dim={self.dim}, k={self.k})")
        if not np.allclose(B.T @ B, np.eye(self.k), atol=1e-8):
            raise ValueError("modes must have orthonormal columns")
        self.modes = B

    # distinct-from: faculties/homeostatic_drives.py -- its tick advances DEFICIT drives
    #   toward set points; this advances a stochastic OU latent. Site read; the shared word
    #   is "drive", and nothing overlaps.
    def step(self):
        """One OU step -> one percept-shaped drive vector in [0,1]^dim.

        THE EXACT OU TRANSITION, not Euler-Maruyama: z' = phi z + sqrt(1-phi^2) N with phi =
        exp(-theta dt) (unit stationary variance by construction). The first draft used Euler
        and its OWN stationarity red caught the theory disagreeing with the process by exactly
        the Euler correction (measured var 0.1197 vs the continuous 0.0900; the discrete
        Euler value is sigma^2/(2 theta - theta^2 dt) = 0.1200). The exact transition makes
        the stationary variance and rho1 = exp(-theta dt) hold at ANY theta, and keeps
        fit_to_traces' -ln(rho1) inversion self-consistent -- Euler at theta > 1/dt has a
        NEGATIVE lag-1 correlation and would refuse its own generated trace.
        """
        phi = math.exp(-self.theta * self.dt)
        self._z = phi * self._z + math.sqrt(1.0 - phi * phi) * self._rng.standard_normal(self.k)
        # UNIT-VARIANCE latent + a layout-derived scale times a DECLARED 3-sigma tail bound
        # (the "3.0" is picked and says so -- the T045 panel caught "never picked" as the
        # overclaim): sqrt(k/dim) is the MEAN per-dim drive std, and the cosine rows span a
        # measured 1.9x around it, so the tail is per-dim uneven -- measured clipping 0.57%
        # overall and ~3.5% on the widest dim (the first draft claimed ~0.3%). Strong
        # against the 0.5 offset (the unscaled draft sat at ~0.04 per dim and the same-path
        # A/B measured almost nothing). The amplitude is PINNED to the percept's own range
        # the way the real percept is (normalized spike counts), which is exactly why sigma
        # could not survive as a parameter: the drive cannot feel it.
        gain = 0.5 / (3.0 * math.sqrt(self.k / self.dim))
        return np.clip(0.5 + gain * (self.modes @ self._z), 0.0, 1.0)

    def bootstrap(self, observe_fn, n_steps=200):
        """Drive `observe_fn` (a REAL plasticity entry point -- in production,
        `PerceptSurprise.observe` of the live instance) with n_steps of spontaneous activity.
        Returns the step count ACTUALLY DELIVERED -- counted in the loop, not echoed from the
        argument (the panel measured the echo returning -5 for n_steps=-5 with zero calls
        delivered) -- so a caller can record that the bootstrap RAN rather than assuming it
        (a swallowed failure must not read as a completed one)."""
        n = 0
        for _ in range(int(n_steps)):
            observe_fn(self.step())
            n += 1
        return n

    # distinct-from: growth/arealization.py -- its L512 graded-arm RMSE is a profile-fit
    #   instrument; nothing here fits gene profiles. Site read.
    # distinct-from: cortex/learned_evc.py -- control-value readout; unrelated. Read.
    # distinct-from: plasticity/competitive_learning.py -- resets activity traces; the shared
    #   word is "traces", unrelated job. Read.
    # distinct-from: reasoning/arc_solver.py -- fits ARC primitive params; unrelated. Read.
    # distinct-from: reasoning/snn_motor_control.py -- learns forward kinematics; unrelated. Read.
    def fit_to_traces(self, traces):
        """Learn theta -- THE parameter -- from a REAL sensor trace by closed-form AR(1)
        moment matching on the mode-projected latents. `traces` is (T, dim). ANALYTICAL --
        lag-1 autocorrelation rho1 -> theta = -ln(rho1)/dt. rho1 is scale-invariant, so the
        fit sees through any amplitude normalization between the trace and the latent
        (including this module's own range-pinning gain -- which is also why there is no
        amplitude parameter to learn; the first draft's sigma recovery red measured
        sigma_hat = gain * sigma exactly, proving the drive cannot feel it). REFUSES rather
        than fabricates: a trace too short, flat, or without usable positive temporal
        correlation gets a named ValueError, never a made-up theta (white noise has rho1 ~ 0,
        and -ln(rho1) of a near-zero number is a huge confident parameter from nothing).
        Updates self.theta; the projected stationary variance rides along as a DIAGNOSTIC,
        never a parameter.
        """
        X = np.asarray(traces, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"trace shape {X.shape} does not match (T, dim={self.dim})")
        if X.shape[0] < 8:
            raise ValueError(f"trace too short to fit: T={X.shape[0]} < 8")
        if not np.isfinite(X).all():
            raise ValueError(
                "non-finite trace: NaN/inf present, nothing to fit (the panel measured NaN "
                "walking through every < comparison into theta and from there into the drive)"
            )
        Z = (X - X.mean(axis=0)) @ self.modes  # (T, k) mode-projected latents, centered
        v = float(Z.var())
        z0, z1 = Z[:-1].ravel(), Z[1:].ravel()
        # Guarded on the quantities actually divided by, not just the pooled variance (a
        # flat-then-jump trace has Z.var() > 0 with a constant z0, panel-constructed) -- and
        # with a RELATIVE tolerance, not <= 0: summation rounding leaves a ~1e-18 std on a
        # constant block, which slips a strict-zero check and hands rho1 pure noise.
        flat_tol = 1e-9 * max(1.0, float(np.abs(Z).max()))
        if v <= 0.0 or z0.std() <= flat_tol or z1.std() <= flat_tol:
            raise ValueError("flat trace: zero variance on the mode projection, nothing to fit")
        rho1 = float(np.dot(z0 - z0.mean(), z1 - z1.mean()) / (len(z0) * z0.std() * z1.std()))
        # The acceptance floor is SAMPLE-SIZE-RELATIVE (3 standard errors of rho1 under the
        # null, SE ~ 1/sqrt(n_pairs)), with 0.05 as the long-trace floor: the panel measured
        # the bare 0.05 bar accepting pure white noise 7-18% of the time at short T (the SE
        # at T=8 is ~0.19, so 0.05 sat INSIDE the null). The floor also implies a fit RANGE:
        # theta <= -ln(floor)/dt, ~3.0 at dt=1 for long traces -- the refusal message names
        # both readings rather than calling a fast OU "not OU-like".
        floor = max(0.05, 3.0 / math.sqrt(len(z0)))
        if rho1 < floor:
            raise ValueError(
                f"rho1={rho1:.3f} below the fit floor {floor:.3f}: either theta is too fast "
                f"for this trace (the floor caps the fittable range at theta <= "
                f"{-math.log(floor) / self.dt:.2f} at this length and dt={self.dt}), or the "
                "trace has no usable positive temporal correlation -- either way, fitting "
                "theta to it would be a made-up number wearing a measurement's shape"
            )
        if rho1 >= 1.0 - 1e-9:
            raise ValueError(
                f"rho1={rho1:.6f} at or above 1: a (near-)monotone trace is not "
                "mean-reverting, and -ln(rho1) would hand back theta ~ 0 -- a frozen drive "
                "wearing a fitted parameter's shape (panel-constructed: a linear ramp)"
            )
        theta = -math.log(rho1) / self.dt
        if not (math.isfinite(theta) and theta > 0.0):
            raise ValueError(f"fit produced a non-usable theta {theta!r} -- refusing to store it")
        self.theta = theta
        return {"theta": theta, "rho1": rho1, "projected_var_diagnostic": v}


# distinct-from: reasoning/nengo_gui_brain.py -- constructs the canonical Brain for the GUI;
#   unrelated job, site read; the matcher fired on the phrase "the one canonical".
# distinct-from: reasoning/rpm_solver.py -- NVSA rule induction; unrelated. Read.
# distinct-from: faculties/global_workspace.py -- the running-baseline rule; unrelated. Read.
# distinct-from: faculties/interoceptive_integration.py -- per-edge A/B writer; unrelated. Read.
# distinct-from: engine/live_brain.py -- integrative-stack presence check; unrelated. Read.
def bootstrap_percept_rule(ps, k=3, seed=0, n_steps=1600, theta=THETA0):
    """THE canonical pre-experience composition -- the one path both the Brain wire and this
    module's own harness take (two compositions would drift apart; this is the P2 posture).

    Drives `ps.observe` (the REAL rule) with spontaneous activity, then RESTORES the
    documented cold-start baseline contract (`_mu = None`, `_var = 0.0`): the T045 panel
    measured the un-reset version silently seeding the surprise STANDARDIZER with synthetic
    OU statistics, so the first real percept read a saturated surprise (1.000 vs the
    contract's 0.000) on a signal the action gate consumes as an eligibility source -- a
    live behavioural change the module never disclosed or measured. The bootstrap installs
    SUBSPACE structure only. Seeding the baseline DELIBERATELY was then MEASURED AND
    REFUTED (2026-08-26, the chip's A/B at this wire configuration: 20 real-ish percept
    sequences with planted novelties, criteria declared before measurement, A/A exactly
    zero, cold control 1.000 saturation / 0.000 discrimination): the seeded arm does
    de-saturate early surprise (0.98 -> 0.59 pinned) but buys it by collapsing the NOVELTY
    margin 5.4x (novel-minus-familiar 0.487 -> 0.091) -- a baseline calibrated on synthetic
    OU statistics reads every real early percept as comparably surprising, so the one thing
    the eligibility signal exists to flag no longer stands out. Both arms converge by tick
    ~48 (the seed is transient either way). Measured at the scalar the gate consumes
    (observe()'s return IS what brain.py reads), so the signal-level result transfers
    monotonically. THE RESET STAYS; re-open only with an A/B beating 0.487 on the same
    declared criteria.

    k adapts downward to the rule's width (k < in_dim is the modes' validity bound); a rule
    too narrow to drive (in_dim < 2) gets 0 steps, RECORDED as 0 rather than left unset.
    Returns the delivered step count.
    """
    k = min(int(k), int(ps.in_dim) - 1)
    if k < 1:
        return 0
    steps = SpontaneousActivity(dim=int(ps.in_dim), k=k, seed=seed, theta=theta).bootstrap(
        ps.observe, n_steps
    )
    ps._mu, ps._var = None, 0.0  # restore the cold-start contract (see docstring)
    return steps


# DIRECT_SOLVE (the NGSNN.fit family): closed-form moment matching, no
# iteration, no error propagation, no plasticity -- it sets the DRIVE's parameters, never a
# network weight. The e-prop invariant governs gradient transport through spiking dynamics,
# absent here.
GRADIENT_CLAIM = {
    "SpontaneousActivity.fit_to_traces": (
        "DIRECT_SOLVE",
        "closed-form AR(1) moment matching (lag-1 autocorrelation -> theta, the one "
        "parameter; the projected variance is a diagnostic, never fitted) on mode-projected "
        "sensor traces; refuses non-OU-like input rather than fabricating parameters; no "
        "iteration, no gradient, no weight touched",
    ),
}


def _principal_alignment(W, B):
    """Mean squared cosine of principal angles between span(W rows) and span(B cols), with
    the DC direction projected out of both (the percept's nonnegative offset makes DC
    dominant and uninformative about the modes). Degenerate directions are DROPPED before
    the comparison (the T045 panel measured plain QR promoting a zero row to a full unit of
    'subspace' -- a dead neuron scoring alignment -- and a rank-1 W overcounting 60%)."""
    dim = B.shape[0]
    dc = np.ones(dim) / math.sqrt(dim)
    P = np.eye(dim) - np.outer(dc, dc)

    def _basis(M):
        u, s, _vt = np.linalg.svd(M, full_matrices=False)
        keep = s > (s.max() * 1e-9 if s.size and s.max() > 0 else 1.0)
        return u[:, keep]

    Qw = _basis(P @ np.asarray(W).T)
    Qb = _basis(P @ B)
    s = np.linalg.svd(Qw.T @ Qb, compute_uv=False)
    return float((s**2).mean()) if s.size else 0.0


def main():
    print("=" * 96)
    print("SPONTANEOUS ACTIVITY -- learnable OU bootstrap through the SAME plasticity path")
    print("=" * 96)
    from snn_data_lake.cortex.percept_surprise import PerceptSurprise

    dim, k = 64, 4

    # 1. THE PROCESS IS WHAT IT CLAIMS: unit stationary variance BY CONSTRUCTION (exact
    # transition) and lag-1 autocorrelation exp(-theta) -- measured over a long run AND
    # asserted (both bounds below).
    sa = SpontaneousActivity(dim, k=k, seed=3)
    zs = []
    for _ in range(20000):
        sa.step()
        zs.append(sa._z.copy())
    Z = np.asarray(zs)[2000:]  # discard burn-in
    v_pred = 1.0  # by construction; the deleted sigma-draft formula agreed only at theta=0.5
    rho_pred = math.exp(-THETA0)  # rho1 = phi = exp(-theta dt), exact (see step())
    v_got = float(Z.var())
    z0, z1 = Z[:-1].ravel(), Z[1:].ravel()
    rho_got = float(np.corrcoef(z0, z1)[0, 1])
    assert abs(v_got - v_pred) < 0.25 * v_pred, f"stationary var {v_got:.4f} vs {v_pred:.4f}"
    assert abs(rho_got - rho_pred) < 0.05, f"autocorr {rho_got:.4f} vs exp(-theta) {rho_pred:.4f}"
    print(
        f"  green  OU stationarity: var {v_got:.4f} (theory {v_pred:.4f}), rho1 {rho_got:.4f} "
        f"(theory {rho_pred:.4f})"
    )

    # 2. SAME-PATH A/B with the REAL rule, plus the matched A/A -- run at BOTH measured
    # configurations: this harness's k=4 rule AND the wire's k=8 rule (the T045 panel caught
    # the first draft citing the k=4 number at the k=8 wire; the numbers are NOT comparable
    # across rule widths -- the chance floor is ~k/dim and doubles between them). k_modes=3
    # both times: at k=4 the DC direction eats the first Sanger slot and exactly 3 remain
    # (driving 4 modes into 3 slots was an earlier measured miss); at k=8, four slots stay
    # free for experience. 1600 steps: measured curves 400/800/1600/3200 -> k=4 rule
    # 0.49/0.58/0.81/0.997, k=8 rule 0.58/0.72/0.90/0.998 -- the wire's config clears the
    # bar at 800 and 1600 is kept for margin (~70 ms, once, at the wire). The bootstrap
    # goes through bootstrap_percept_rule, THE canonical composition the wire runs, so the
    # baseline-contract reset is exercised here too.
    k_modes = 3
    ps_boot = PerceptSurprise(in_dim=dim, k=k, seed=0)
    ps_cold = PerceptSurprise(in_dim=dim, k=k, seed=0)
    ps_twin = PerceptSurprise(in_dim=dim, k=k, seed=0)
    steps = bootstrap_percept_rule(ps_boot, k=k_modes, seed=5, n_steps=1600)
    bootstrap_percept_rule(ps_twin, k=k_modes, seed=5, n_steps=1600)
    assert steps == 1600, f"the bootstrap delivered {steps} of 1600 steps"
    # THE BASELINE-CONTRACT RED (panel blocker): the un-reset first draft silently seeded
    # the surprise STANDARDIZER with synthetic OU statistics, handing the first real percept
    # a saturated surprise (measured 1.000 vs the cold contract's 0.000) on a signal the
    # action gate consumes. The canonical composition must leave the baseline COLD.
    assert ps_boot._mu is None and ps_boot._var == 0.0, (
        "the bootstrap seeded the surprise baseline -- an undisclosed live charge on the "
        "action gate's eligibility input"
    )
    W_boot = np.asarray([n.w for n in ps_boot._neurons])
    W_cold = np.asarray([n.w for n in ps_cold._neurons])
    W_twin = np.asarray([n.w for n in ps_twin._neurons])
    assert (W_boot == W_twin).all(), "A/A is nonzero -- the harness has run nondeterminism"
    B = _cosine_modes(dim, k_modes)
    a_boot = _principal_alignment(W_boot, B)
    a_cold = _principal_alignment(W_cold, B)
    # a_cold + 0.3, not 2*a_cold: the panel measured the ratio clause non-binding (cold sits
    # near the ~k/dim chance floor, far under the absolute bar).
    assert a_boot > 0.6 and a_boot > a_cold + 0.3, (
        f"the bootstrap did not install the driven structure (alignment {a_boot:.3f} vs "
        f"cold {a_cold:.3f}) -- the same-path claim has no measured content"
    )
    ps_wire = PerceptSurprise(in_dim=dim, k=8, seed=0)
    ps_wire_cold = PerceptSurprise(in_dim=dim, k=8, seed=0)
    bootstrap_percept_rule(ps_wire, k=k_modes, seed=5, n_steps=1600)
    a_wire = _principal_alignment(np.asarray([n.w for n in ps_wire._neurons]), B)
    a_wire_cold = _principal_alignment(np.asarray([n.w for n in ps_wire_cold._neurons]), B)
    assert a_wire > 0.6 and a_wire > a_wire_cold + 0.3, (
        f"the WIRE configuration (k=8 rule) did not install structure ({a_wire:.3f} vs "
        f"cold {a_wire_cold:.3f}) -- the number cited at the wire must be measured there"
    )
    print(
        f"  green  same-path A/B ({steps} steps through PerceptSurprise.observe): harness "
        f"k=4 rule {a_boot:.3f} vs cold {a_cold:.3f}; WIRE k=8 rule {a_wire:.3f} vs cold "
        f"{a_wire_cold:.3f}; A/A bit-identical; baseline left cold"
    )

    # 3. PERMUTED-MODES NULL (the reachable red; the control is not the method): the same
    # OU, the same rule, the same step count -- but the drive projected through DC-MATCHED
    # seeded-random orthonormal directions (DC projected out before QR: the panel measured a
    # raw Q carrying 12-16% DC leakage the declared modes are exactly free of, making the
    # control differ from the method in TWO ways). Alignment to the DECLARED modes
    # collapses: the installed structure comes from the modes, not from the harness.
    ps_null = PerceptSurprise(in_dim=dim, k=k, seed=0)
    sa_null = SpontaneousActivity(dim, k=k_modes, seed=5)
    rng = np.random.default_rng(11)
    dc = np.ones(dim) / math.sqrt(dim)
    Q, _ = np.linalg.qr((np.eye(dim) - np.outer(dc, dc)) @ rng.standard_normal((dim, k_modes)))
    sa_null.set_modes(Q)
    sa_null.bootstrap(ps_null.observe, 1600)
    a_null = _principal_alignment(np.asarray([n.w for n in ps_null._neurons]), B)
    assert a_null < 0.5 * a_boot, (
        f"permuted modes still aligned to the declared basis ({a_null:.3f} vs {a_boot:.3f})"
    )
    print(f"  RED    permuted-modes null (DC-matched): {a_null:.3f} vs {a_boot:.3f} declared")

    # 3b. OU-ARM ATTRIBUTION (panel blocker: the OU had no per-item control, and the arm
    # REFUTES it on this criterion): a ~iid latent (theta=20, phi ~ 2e-9) through the same
    # modes, gain, rule and step count aligns AT LEAST as well -- Oja is ORDER-BLIND (it
    # learns spatial covariance, which temporal correlation does not change), so the OU
    # strictly COSTS effective samples here (measured: iid 0.845 vs OU 0.808 at the first
    # draft's harness). The OU is kept WITH that cost carried openly: its intended
    # consumers are the temporally-sensitive rules the arc lands next (the LPL/STDP family,
    # T049+), for which an iid drive carries nothing. If no temporally-sensitive consumer
    # materializes, the OU -- and theta, and fit_to_traces -- deletes per P8 clause 5; the
    # modes and gain survive it.
    ps_iid = PerceptSurprise(in_dim=dim, k=k, seed=0)
    bootstrap_percept_rule(ps_iid, k=k_modes, seed=5, n_steps=1600, theta=20.0)
    a_iid = _principal_alignment(np.asarray([n.w for n in ps_iid._neurons]), B)
    assert a_iid > 0.6, f"the iid attribution arm failed to install structure ({a_iid:.3f})"
    print(
        f"  green  OU-arm attribution: iid {a_iid:.3f} vs OU {a_boot:.3f} -- the OU COSTS "
        "on this criterion (order-blind Oja); kept for T049+'s temporal rules, deletes on null"
    )

    # 4. LEARNABLE, with its reds: a trace from a KNOWN theta (well away from the default,
    # and above 1/dt where the retired Euler draft could not even exist) is recovered
    # THROUGH the drive's own range normalization (rho1 is scale-invariant). Every refusal
    # is discriminated BY MESSAGE, including the panel-constructed traps: NaN walks through
    # < comparisons; a linear ramp drives rho1 to 1; a flat-then-jump trace passes a pooled
    # variance guard while z0 is constant; white noise sneaks under a fixed bar at short T
    # (the floor is SE-relative now).
    theta_true = 1.2
    gen = SpontaneousActivity(dim, k=k, seed=9, theta=theta_true)
    trace = np.stack([gen.step() for _ in range(4000)])
    fit = SpontaneousActivity(dim, k=k, seed=0)
    out = fit.fit_to_traces(trace)
    assert abs(out["theta"] - theta_true) < 0.3 * theta_true, f"theta {out['theta']:.3f}"
    assert fit.theta == out["theta"] != THETA0, "the fit did not move the live parameter"
    nan_trace = trace.copy()
    nan_trace[10, 0] = float("nan")
    ramp = 0.5 + np.outer(np.linspace(0.0, 1.0, 200), _cosine_modes(dim, 1)[:, 0] * 0.2)
    flat_jump = np.full((32, dim), 0.5)
    flat_jump[-1] += _cosine_modes(dim, 1)[:, 0] * 0.2
    # ONE generator instance -- the per-iteration-reconstruction trap the determinism check
    # documents bit this exact line in review (4000 copies of the same first step).
    fast_gen = SpontaneousActivity(dim, k=k, seed=9, theta=3.5)
    fast = np.stack([fast_gen.step() for _ in range(4000)])
    refusals = [
        (
            lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(
                np.random.default_rng(1).random((4000, dim))
            ),
            "below the fit floor",
        ),
        (
            lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(
                trace[np.random.default_rng(2).permutation(len(trace))]
            ),
            "below the fit floor",
        ),
        (lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(fast), "below the fit floor"),
        (lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(nan_trace), "non-finite trace"),
        (lambda: SpontaneousActivity(dim, k=1, seed=0).fit_to_traces(ramp), "at or above 1"),
        # flat-then-jump lands in DIFFERENT honest branches by k: at k=1 the constant z0
        # trips the flat guard (the panel's exact construction); at k>1 the pooled ravel
        # carries cross-COMPONENT variance, so the trace reads as correlation-free and the
        # floor refuses it -- both are refusals, each named by its own message.
        (lambda: SpontaneousActivity(dim, k=1, seed=0).fit_to_traces(flat_jump), "flat trace"),
        (
            lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(flat_jump),
            "below the fit floor",
        ),
        (lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(trace[:5]), "too short"),
        (
            lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(np.full((100, dim), 0.5)),
            "flat trace",
        ),
        (lambda: SpontaneousActivity(dim, seed=0).fit_to_traces(trace[:, :10]), "does not match"),
        (lambda: SpontaneousActivity(dim, k=0, seed=0), "0 < k < dim"),
        (lambda: SpontaneousActivity(dim, k=dim, seed=0), "0 < k < dim"),
        (lambda: SpontaneousActivity(dim, seed=0, theta=0.0), "finite theta > 0"),
        (lambda: SpontaneousActivity(dim, seed=0, theta=float("nan")), "finite theta > 0"),
        (lambda: SpontaneousActivity(dim, seed=0, dt=0.0), "finite dt > 0"),
        (lambda: SpontaneousActivity(dim, seed=0).set_modes(np.zeros((dim, 9))), "modes shape"),
        (
            lambda: SpontaneousActivity(dim, seed=0).set_modes(np.ones((dim, 4))),
            "orthonormal columns",
        ),
    ]
    for bad, needle in refusals:
        try:
            bad()
            raise AssertionError(f"a refusal path accepted bad input (wanted {needle!r})")
        except ValueError as e:
            assert needle in str(e), f"refused by the WRONG branch: {needle!r} not in {e}"
    print(
        f"  green  learnable: theta {out['theta']:.3f} (true {theta_true}) recovered from a "
        "4000-step trace through the range normalization"
    )
    print(
        f"  RED    {len(refusals)} refusals by message: white noise / shuffled / fast-OU "
        "(floor named) / NaN / ramp (rho1>=1) / flat-jump (both k branches) / short / flat "
        "/ shape / ctor bounds / modes"
    )

    # 5. Determinism: same seed -> bit-identical drive SEQUENCE (the persistence invariant).
    # ONE instance per arm, on purpose: the first draft re-constructed the instance inside
    # the comprehension, so it compared the same FIRST step three times -- the panel's
    # deterministic-step-1/unseeded-thereafter mutant passed it. The non-repetition assert
    # is what makes this a sequence claim.
    s1 = SpontaneousActivity(dim, seed=4)
    s2 = SpontaneousActivity(dim, seed=4)
    d1 = [s1.step() for _ in range(3)]
    d2 = [s2.step() for _ in range(3)]
    assert not (d1[0] == d1[1]).all(), "the sequence is frozen -- consecutive steps identical"
    assert all((a == b).all() for a, b in zip(d1, d2)), "the drive is not deterministic in seed"
    print("  green  determinism: same seed -> identical 3-step sequence (and steps DIFFER)")

    print()
    print(
        "DISCLOSURE: the modes are a DECLARED prior (the percept latent has no measured "
        "topology); what is proven is that the bootstrap INSTALLS the declared structure "
        "through the real plasticity rule, never that the structure is right for real "
        "percepts. The live wire sits at Brain._perceive_real's PerceptSurprise lazy-init "
        "(pre-first-percept, torch perception tier); whether the bootstrap helps REAL first "
        "percepts is the eventual A/B's question, with the mode basis one of its variables. "
        "NULL DISPOSITION: if that A/B shows no win, this module is DELETED, not parked."
    )
    print(
        "self-check OK: OU stationarity measured, same-path A/B (both rule widths) + A/A "
        "through the real rule + cold baseline, permuted-modes red, OU-arm attribution, "
        f"learnable fit + {len(refusals)} message-discriminated refusals, determinism"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
