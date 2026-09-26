"""
Continuous-attractor SPIKING grid network -- path integration by a moving LIF bump.

WHAT THE BRAIN DOES. Entorhinal grid cells path-integrate self-motion on a recurrent 2-D sheet: local
excitation + broader inhibition (a center-surround / "Mexican-hat" kernel) holds ONE self-sustaining bump
of activity on a torus, and an ASYMMETRIC, velocity-gated shift in that recurrent connectivity translates
the bump across the sheet at a rate proportional to the animal's velocity, so the bump POSITION integrates
velocity into position -- dead-reckoning with spikes (McNaughton et al., Nat Rev Neurosci 7:663, 2006;
Burak & Fiete, PLoS Comput Biol 5:e1000291, 2009; Zhang, J Neurosci 16:2112, 1996 for the derivative-kernel
shift). This is the 2-D generalization of the repo's 1-D ring_attractor CAN.

THE GAP THIS FILLS (verified against the repo's spatial substrate). grid_cells.py and grid_place_map.py build
the hexagonal grid code from an ANALYTIC plane-wave RATE formula r(p)=g(sum cos(b_k.p)) -- the position is
supplied to a closed-form field, no network integrates anything. ring_attractor.py is a genuine spiking/rate
continuous attractor but only on a 1-D RING (a heading angle, scalar angular velocity). This module is the
missing piece: an actual recurrent SPIKING network (LIF membranes, threshold spikes, spike-driven synapses)
whose moving bump PERFORMS the 2-D path integration, rather than a rate field being evaluated at a
handed-in coordinate.

MECHANISM (pure LIF + recurrent connectivity; no backprop).
  - N = n*n leaky-integrate-and-fire neurons tile a periodic n-by-n sheet (a torus of grid phases).
  - Center-surround recurrence is a DIFFERENCE OF VON MISES kernels: a sharp excitatory kernel VMe
    (kappa_e) minus a broad inhibitory kernel VMi (kappa_i), a_e*VMe - a_i*VMi -> Mexican-hat. The 2-D
    kernel is SEPARABLE (outer product of 1-D kernels), so the sheet convolution factorizes to
    VM @ syn @ VM (O(n^3), not O(n^4)); syn is the spike-driven synaptic-activity field.
  - Velocity path integration uses ASYMMETRIC shift connectivity = the ANGULAR DERIVATIVE of that kernel
    (dVMe, dVMi -- exactly ring_attractor's w_asym, one per sheet axis). A velocity input adds
    g*(vy*d/drow + vx*d/dcol) recurrent current, which translates the bump; the shift gain g is
    self-calibrated once at construction (like ring_attractor's ProprioceptiveEstimator) so the decoded
    displacement equals the integral of the commanded velocity.
  - Each tick: I = (center-surround + velocity-shift)/n ; v = leak*v + I ; spike where v >= vth ; reset to
    0 ; syn = syn_decay*syn + spikes. Position is read out as the population-vector (circular-mean) center
    of the spiking activity -- continuous WITHIN one grid period (grid cells are intrinsically periodic;
    a single module resolves position modulo its spacing, exactly like a real grid module).
  - reset(pos) CLAMPS the bump onto the phase that encodes pos -- the optional LANDMARK / place->grid phase
    reset that corrects accumulated drift when a known landmark is seen.

LANDMARK PHASE RESET (drift correction). Pure dead-reckoning DRIFTS: the shift gain is self-calibrated ONCE,
so residual gain error plus sheet discretization accumulate with path length and the decoded phase slides
away from ground truth. Real entorhinal grids do not drift without bound because sensory landmarks and
boundaries RESET the grid phase (Hardcastle, Ganguli & Giocomo, Neuron 86:827, 2015 -- error accumulates with
distance from a boundary and is corrected at it; Evans et al., J Neurosci 36:4266, 2016 -- place->grid
feedback anchors the phase). Implemented here as:
  - observe_landmark(id, true_pos=None) on the FIRST sighting imprints the co-active grid activity onto that
    landmark cell's synapses by a one-shot HEBBIAN outer product (pre = the landmark cell fires, post = the
    syn field), peak-normalized: w = syn / max(syn). If true_pos is supplied it imprints the idealized bump
    at that place instead. The anchor is imprinted ONCE, while the estimate is still trusted; re-sightings
    only READ it. Re-learning continually from an already-drifted bump would let the anchor itself drift --
    deliberately NOT done, and re-sightings never receive a fresh ground-truth oracle.
  - On RE-sighting the stored trace is injected as a graded CUE COMBINATION onto the synaptic field,
    syn <- (1-gain)*syn + gain*scale*w, where scale matches the trace to the live bump's amplitude. Two
    von Mises bumps at nearby phases combine into a bump whose circular mean lies BETWEEN them, weighted by
    gain: gain=0 is a no-op, gain=1 a full re-clamp onto the stored phase, anything between a partial nudge.
    This is NOT a teleport -- the membrane state v is left untouched and the CAN dynamics resolve the mixture.
  - MEASURED LIMIT (kept, not tuned away): the mixture only works while the anchor/bump mismatch is a modest
    fraction of a period. At gaps of 0.267..0.466 period the mixture is BIMODAL and the broad inhibition
    extinguishes BOTH lobes -- bump amplitude collapses to 0.000 (gaps <=0.213 recover to 2.000). Because the
    network knows this mismatch WITHOUT ground truth (stored anchor phase vs its own decode), a mismatch past
    capture_threshold (default 0.2 period) makes the landmark CAPTURE outright (gain -> 1, a unimodal
    re-clamp) rather than mix: large cue conflict resolves by the sensory cue winning, small conflict
    integrates. capture_threshold=None keeps the raw fragile mixture -- main() asserts that it still fails.
  - drift(true_pos) reports the current torus distance between decode() and a supplied ground truth.

IMPLEMENTATION NOTE (numpy). The sheet state (v, syn), the four von Mises kernels and every landmark trace
are numpy float64 arrays, and the per-tick hot path (the six 16x16 matrix products, the LIF threshold/reset,
the population-vector readout) is vectorized. This is a PERFORMANCE change only -- the mechanism, the
parameters and n=16 are untouched. MEASURED at n=16: 1.510 ms -> 0.0226 ms per step() (66.9x; best of 5 x
300 ticks), construction 58.5 ms -> 1.05 ms (55.5x). NOT BIT-IDENTICAL, and honestly so: the one-time
constants (theta, sin/cos tables, VMe/VMi/dVMe/dVMi, _bump) are still built elementwise with math.* and ARE
bit-identical, but per-tick BLAS/pairwise summation accumulates in a different ORDER than the old sequential
triple loop, so decoded trajectories agree to 1.110e-16 (1 ulp) rather than exactly -- 21 of 144 probed
trajectory coordinates differ in the last bit. Every drift, amplitude, shift-gain and landmark number in
this docstring and in main() is unchanged bit-for-bit, and every self-check assert and test threshold is
unchanged. One seam artifact worth naming: decode() of a bump sitting exactly at 0 is a knife-edge -- the
mathematically-zero sin marginal rounds to -2.2e-16 under sequential summation (atan2 -> just below 0 ->
wraps to x = spacing) and to exactly 0.0 under BLAS (x = 0.0). Both denote the SAME point (torus distance
0.0) and _circ_diff is indifferent, but the returned representative flips at that one measure-zero point.

Rationale for keeping n=16 here: spacing=1.0 and the n-by-n sheet tiles exactly ONE period, so a larger n
buys finer discretization WITHIN a period and does NOT extend range -- position stays resolved modulo one
spacing. What numpy actually unlocks is 8-16 MODULES at different spacings per tick (8 modules: 12.1 ms
pure-Python -> 0.18 ms here), and multi-scale phase disambiguation is what extends range. Separate work.

DISTINCTNESS. Not grid_cells.py / grid_place_map.py (analytic RATE fields, no recurrent dynamics). Not
ring_attractor.py (1-D heading ring, scalar angular velocity). ZERO coupling to the arm / safety envelope:
representation-only, imports only the stdlib + numpy, issues no commands, exposes no actuator route.

HONEST SCOPE. One grid module (single spacing) on a small toroidal sheet, so position is resolved only
WITHIN one period, not over an open world (a multi-scale population would disambiguate -- see grid_cells.py).
Velocity integration is accurate within a bounded velocity range (the self-calibrated linear regime, |v| up
to a few percent of the period per tick); larger per-tick velocities out-run the bump and integrate
sub-linearly. Deterministic (no stochastic units).

MEASURED DRIFT NUMBERS (this module's self-check; report them as they are).
  - Free dead-reckoning over 4 open circuits (50 ticks at v=0.02 each = one full grid period per circuit)
    drifts MONOTONICALLY: 0.1392 -> 0.2955 -> 0.4424 -> 0.5950 period, i.e. ~0.1487 per period travelled
    (a residual shift-gain calibration error), reaching 84% of the 0.7071 largest error the torus can even
    represent. The bump stays healthy throughout (amp 1.984) -- the estimate is LOST, the network is not dead.
  - The SAME trajectory with 'home' sighted once per circuit stays BOUNDED: 0.0617/0.0060 alternating,
    0.0060 at circuit 4 (99x lower than free) and still <= 0.0617 out to circuit 8, amp 1.941. The residual
    is not zero and does not shrink to zero -- error re-accumulates between sightings and is knocked back
    down at each one, which is exactly what "bounded, not corrected" means.
  - NEGATIVE RESULT worth stating: on a CLOSED square lap there is essentially NO accumulating drift to fix
    (0.067 after 5 laps and the same 0.067 after 30 laps) because equal +v and -v cancel the multiplicative
    gain error. Landmark reset buys nothing there. Drift is a property of the PATH, not just of path length.
  - The graded correction's failure mode is REAL and kept: see capture_threshold above (amp -> 0.000).

  python -m snn_data_lake.hippocampus.spiking_grid_can     # self-check; also runnable by run_selfchecks.py
"""

import math

import numpy as np

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "SpikingGridCAN._calibrate": (
        "NOT_A_RULE",
        "self-calibrates the shift gain by comparing decoded displacement against the commanded "
        "velocity integral. A one-off measurement of this network own transfer",
    ),
    "_graded_probe": (
        "NOT_A_RULE",
        "walks the attractor and corrects with a given gain so gains are directly comparable. A "
        "deterministic probe",
    ),
}


def _circ_diff(a, b, period):
    """Signed smallest difference a - b on a ring of circumference `period`, in (-period/2, period/2]."""
    return (a - b + period / 2.0) % period - period / 2.0


class SpikingGridCAN:
    """2-D continuous-attractor SPIKING grid network: a moving LIF bump path-integrates velocity.

    Usage:
        can = SpikingGridCAN(n=16, seed=0)
        can.reset((0.2, 0.1))                 # clamp the bump (landmark / place->grid phase reset)
        for _ in range(10):
            can.step(velocity=(0.03, 0.0))    # advance the bump by one integration tick
        can.decode()                          # -> (x, y) continuous position within one grid period
        can.bump_center()                     # -> (row, col) bump location on the n-by-n sheet
    """

    def __init__(
        self,
        n=16,
        seed=0,
        spacing=1.0,
        kappa_e=3.0,
        kappa_i=0.3,
        a_e=2.0,
        a_i=0.8,
        leak=0.5,
        syn_decay=0.5,
        vth=0.1,
        vel_gain=1.0,
        settle_steps=30,
        calibrate=True,
        landmark_gain=0.6,
        capture_threshold=0.2,
    ):
        self.n = int(n)
        self.seed = int(seed)  # network is deterministic; seed kept for interface parity
        self.spacing = float(spacing)
        self.kappa_e = float(kappa_e)
        self.a_e, self.a_i = float(a_e), float(a_i)
        self.leak = float(leak)
        self.syn_decay = float(syn_decay)
        self.vth = float(vth)
        self.vel_gain = float(vel_gain)
        self.landmark_gain = float(
            landmark_gain
        )  # default strength of a landmark phase reset in [0,1]
        self.capture_threshold = None if capture_threshold is None else float(capture_threshold)
        self.last_capture = (
            False  # did the most recent sighting CAPTURE (large cue conflict) rather than mix?
        )
        self._landmarks = {}  # landmark id -> Hebbian synaptic trace of the grid phase it was first seen at
        self.N = self.n * self.n

        n = self.n
        theta = [
            2.0 * math.pi * i / n for i in range(n)
        ]  # phase of each sheet row/col on the torus
        self.theta = theta
        # Readout tables and kernels are ONE-TIME constants: built elementwise with math.* (not the
        # numpy ufuncs) so every constant is bit-identical to the pure-Python version of this module.
        self._sin_theta = np.array([math.sin(t) for t in theta])  # population-vector readout tables
        self._cos_theta = np.array([math.cos(t) for t in theta])
        cos_d = np.array([[math.cos(theta[i] - theta[j]) for j in range(n)] for i in range(n)])
        sin_d = np.array([[math.sin(theta[i] - theta[j]) for j in range(n)] for i in range(n)])
        self.VMe = np.array(
            [[math.exp(kappa_e * (cos_d[i][j] - 1.0)) for j in range(n)] for i in range(n)]
        )  # sharp excitatory kernel (center)
        self.VMi = np.array(
            [[math.exp(kappa_i * (cos_d[i][j] - 1.0)) for j in range(n)] for i in range(n)]
        )  # broad inhibitory kernel (surround)
        self.dVMe = -kappa_e * sin_d * self.VMe  # angular derivative -> asymmetric shift
        self.dVMi = -kappa_i * sin_d * self.VMi  # (2-D generalization of ring w_asym)
        # dVMe.T / dVMi.T, precomputed AND made contiguous (constant across ticks, so the per-tick
        # matrix products never pay for a strided operand).
        self.dVMeT = np.ascontiguousarray(self.dVMe.T)
        self.dVMiT = np.ascontiguousarray(self.dVMi.T)

        self.v = np.zeros((n, n))  # LIF membrane potentials
        self.syn = np.zeros((n, n))  # spike-driven synaptic-activity field
        self._gain = 1.0
        if calibrate:
            self._gain = self._calibrate(settle_steps)
        self.reset((0.0, 0.0))

    # ---- recurrent connectivity: center-surround + velocity-gated asymmetric shift ----
    def _currents(self, vx, vy):
        """Recurrent synaptic current onto every neuron: Mexican-hat sustain + velocity-driven shift."""
        syn = self.syn
        VMe, VMi = self.VMe, self.VMi
        E = VMe @ syn  # VMe @ syn
        Im = VMi @ syn  # VMi @ syn
        dE = self.dVMe @ syn  # dVMe @ syn
        dIm = self.dVMi @ syn  # dVMi @ syn
        a_e, a_i = self.a_e, self.a_i
        rec = a_e * (E @ VMe) - a_i * (Im @ VMi)  # separable center-surround
        shift_row = a_e * (dE @ VMe) - a_i * (dIm @ VMi)  # d/drow  (y axis)
        shift_col = a_e * (E @ self.dVMeT) - a_i * (Im @ self.dVMiT)  # d/dcol  (x axis)
        g = self.vel_gain * self._gain
        return (rec + g * (vy * shift_row + vx * shift_col)) / self.n

    def step(self, velocity=(0.0, 0.0)):
        """Advance the bump one integration tick under `velocity`=(vx, vy); return decode()."""
        vx, vy = velocity
        nv = self.leak * self.v + self._currents(vx, vy)  # LIF membrane integration
        fired = nv >= self.vth  # threshold
        spikes = fired.astype(np.float64)
        self.v = np.where(fired, 0.0, nv)  # reset to 0
        self.syn = self.syn * self.syn_decay + spikes  # spike-driven recurrent synapses
        return self.decode()

    # ---- landmark reset: clamp the bump onto the phase encoding `pos` (place->grid phase reset) ----
    def _bump(self, pos):
        """Idealized separable von Mises bump at `pos`, as an n-by-n ndarray (outer product)."""
        x, y = pos
        n = self.n
        theta = self.theta
        ke = self.kappa_e
        py = 2.0 * math.pi * (y / self.spacing)
        px = 2.0 * math.pi * (x / self.spacing)
        row = np.array([math.exp(ke * (math.cos(theta[i] - py) - 1.0)) for i in range(n)])
        col = np.array([math.exp(ke * (math.cos(theta[j] - px) - 1.0)) for j in range(n)])
        return row[:, None] * col[None, :]

    def reset(self, pos=(0.0, 0.0)):
        """Clamp the bump onto the location `pos` (the optional landmark / place->grid phase reset).

        Learned landmark anchors are long-term memory and SURVIVE a reset -- only the bump is re-clamped.
        """
        self.syn = self._bump(pos)
        self.v = np.zeros((self.n, self.n))
        return self.decode()

    # ---- landmark phase reset: Hebbian place->grid anchor + graded correction of accumulated drift ----
    def observe_landmark(self, landmark_id, true_pos=None, gain=None):
        """Sight landmark `landmark_id`; return the magnitude of the phase correction applied (position units).

        FIRST sighting -> one-shot HEBBIAN imprint of the co-active grid phase (w = syn/max(syn), or the
        idealized bump at `true_pos` when supplied); no correction is applied, so 0.0 is returned.
        RE-sighting -> graded cue combination toward the STORED phase, syn <- (1-gain)*syn + gain*scale*w.
        `true_pos` is used ONLY on the first sighting: re-sightings are corrected by the landmark MEMORY,
        never by a fresh ground-truth oracle. `gain` defaults to self.landmark_gain and is clipped to [0,1].
        """
        if landmark_id not in self._landmarks:
            field = self._bump(true_pos) if true_pos is not None else self.syn.copy()
            peak = float(field.max())
            if peak <= 0.0:
                peak = 1.0
            self._landmarks[landmark_id] = field / peak
            return 0.0

        w = self._landmarks[landmark_id]
        g = self.landmark_gain if gain is None else float(gain)
        g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        before = self.decode()
        anchor = self._decode_field(w)
        # Internal mismatch between the anchor and the bump -- known WITHOUT ground truth (both are the
        # network's own state). MEASURED failure: a mixture of two bumps more than ~0.21 of a period apart
        # is bimodal and the broad inhibition extinguishes BOTH lobes (amplitude -> 0.000 at gaps
        # 0.267..0.466; see main()'s failing control). Past capture_threshold the landmark therefore CAPTURES
        # outright (g -> 1) instead of mixing -- large cue conflict resolves by the sensory cue winning,
        # small conflict integrates. Set capture_threshold=None to keep the raw (fragile) mixture.
        gap = math.hypot(
            _circ_diff(anchor[0], before[0], self.spacing),
            _circ_diff(anchor[1], before[1], self.spacing),
        )
        self.last_capture = self.capture_threshold is not None and gap > self.capture_threshold
        if self.last_capture:
            g = 1.0
        peak_w = float(w.max())
        scale = (self.bump_amplitude() / peak_w) if peak_w > 0.0 else 1.0
        # graded cue combination, NOT a teleport
        self.syn = (1.0 - g) * self.syn + g * scale * w
        after = self.decode()
        return math.hypot(
            _circ_diff(after[0], before[0], self.spacing),
            _circ_diff(after[1], before[1], self.spacing),
        )

    def stored_phase(self, landmark_id):
        """Decoded position (x, y) of the grid phase anchored to `landmark_id`, or None if never seen."""
        w = self._landmarks.get(landmark_id)
        return None if w is None else self._decode_field(w)

    def known_landmarks(self):
        """Ids of every landmark whose grid phase has been imprinted, in insertion order."""
        return list(self._landmarks)

    def drift(self, true_pos):
        """Current path-integration error: torus distance from decode() to the supplied ground truth."""
        x, y = self.decode()
        return math.hypot(
            _circ_diff(x, true_pos[0], self.spacing), _circ_diff(y, true_pos[1], self.spacing)
        )

    # ---- readout: population-vector (circular-mean) center of the spiking bump ----
    def _phase(self, marginal):
        """Circular mean of a length-n marginal. math.atan2 on Python floats -> a PYTHON float out."""
        s = float(np.dot(marginal, self._sin_theta))
        c = float(np.dot(marginal, self._cos_theta))
        return math.atan2(s, c)

    def bump_center(self):
        """Bump location on the n-by-n sheet as continuous (row, col) in [0, n)."""
        row = self._phase(self.syn.sum(axis=1)) % (2 * math.pi) / (2 * math.pi) * self.n
        col = self._phase(self.syn.sum(axis=0)) % (2 * math.pi) / (2 * math.pi) * self.n
        return (float(row), float(col))

    def _decode_field(self, field):
        """Population-vector position (x, y) of ANY n-by-n activity field (used for stored landmark traces)."""
        y = self._phase(field.sum(axis=1)) % (2 * math.pi) / (2 * math.pi) * self.spacing
        x = self._phase(field.sum(axis=0)) % (2 * math.pi) / (2 * math.pi) * self.spacing
        return (float(x), float(y))

    def decode(self):
        """Decoded continuous position (x, y) within one grid period [0, spacing)."""
        return self._decode_field(self.syn)

    def bump_amplitude(self):
        """Peak synaptic activity of the bump -- a liveness measure (single stable bump => > 0)."""
        return float(self.syn.max())

    def _calibrate(self, settle_steps):
        """Self-calibrate the shift gain: decoded displacement -> commanded velocity integral (like ring)."""
        self._gain = 1.0
        self.reset((0.5 * self.spacing, 0.5 * self.spacing))  # mid-period, away from the wrap seam
        for _ in range(settle_steps):
            self.step((0.0, 0.0))
        before = self.decode()
        cal_v, cal_steps = 0.02, 8
        for _ in range(cal_steps):
            self.step((cal_v, 0.0))
        moved = _circ_diff(self.decode()[0], before[0], self.spacing)
        return (cal_v * cal_steps) / moved if abs(moved) > 1e-9 else 1.0


CIRCUIT_V, CIRCUIT_STEPS = 0.02, 50  # 0.02 * 50 == 1.0 == exactly ONE grid period per circuit


def _run_circuits(can, circuits, use_landmarks, gain=0.6):
    """Walk `circuits` OPEN circuits around the periodic sheet; return (per-circuit drift, worst, min amp).

    Why an OPEN circuit and not a closed loop: a CLOSED loop commands equal +v and -v, so a multiplicative
    shift-gain error CANCELS and nothing accumulates (measured: a 24-tick square lap sits at 0.067 error after
    5 laps AND after 30 laps -- flat). Dead-reckoning drift only accumulates when velocity does not reverse.
    Because the sheet is a torus, translating by exactly one period returns to the SAME place, so a constant
    +v circuit both accumulates error monotonically and revisits the landmark once per circuit.

    Ground truth is the exact integral of the COMMANDED velocities. 'home' is imprinted BEFORE any motion
    (drift exactly 0, so the anchor is exact) and re-sighted at the end of every circuit.
    """
    tx, ty = can.decode()
    if use_landmarks:
        can.observe_landmark("home")  # one-shot Hebbian imprint while the estimate is still exact
    worst = 0.0
    min_amp = can.bump_amplitude()
    per_circuit = []
    for _ in range(circuits):
        for _k in range(CIRCUIT_STEPS):
            can.step((CIRCUIT_V, CIRCUIT_V))
            tx += CIRCUIT_V
            ty += CIRCUIT_V
            worst = max(worst, can.drift((tx, ty)))
            min_amp = min(min_amp, can.bump_amplitude())
        if use_landmarks:
            can.observe_landmark("home", gain=gain)
            min_amp = min(min_amp, can.bump_amplitude())
        per_circuit.append(can.drift((tx, ty)))
    return per_circuit, worst, min_amp


def _settled(seed=0):
    """A fresh network settled onto a stable bump at mid-period (away from the wrap seam)."""
    can = SpikingGridCAN(n=16, seed=seed)
    can.reset((0.5, 0.5))
    for _ in range(20):
        can.step((0.0, 0.0))
    return can


def _graded_probe(gain, walk=6, capture_threshold=0.2, settle=0):
    """Learn an anchor, walk away, correct with `gain`; deterministic, so gains are directly comparable.

    `settle` free-running ticks are taken AFTER the correction -- a bimodal mixture still looks alive on the
    tick it is written and only collapses once the recurrent inhibition resolves it, so the failing control
    needs settle > 0 to see the death.
    Returns (correction magnitude, gap-to-anchor before, gap-to-anchor after, bump amplitude after).
    """
    can = _settled()
    can.capture_threshold = capture_threshold
    can.observe_landmark("L")  # one-shot Hebbian imprint of the settled phase
    anchor = can.stored_phase("L")
    for _ in range(walk):
        can.step((0.02, 0.02))  # walk away -> the bump now sits elsewhere
    before = can.decode()
    corr = can.observe_landmark("L", gain=gain)
    for _ in range(settle):
        can.step((0.0, 0.0))
    after = can.decode()

    def gap(p):
        return math.hypot(
            _circ_diff(p[0], anchor[0], can.spacing), _circ_diff(p[1], anchor[1], can.spacing)
        )

    return corr, gap(before), gap(after), can.bump_amplitude()


def main():
    can = SpikingGridCAN(n=16, seed=0)

    # ---- 1. a single self-sustaining bump PERSISTS at a clamped location (analog spatial memory) ----
    can.reset((0.40, 0.30))
    for _ in range(50):
        can.step((0.0, 0.0))
    hold = can.decode()
    persist_err = math.hypot(
        _circ_diff(hold[0], 0.40, can.spacing), _circ_diff(hold[1], 0.30, can.spacing)
    )
    amp = can.bump_amplitude()

    # ---- 2. PATH INTEGRATION: a known velocity SEQUENCE moves the decoded position to its integral ----
    can.reset((0.15, 0.15))
    for _ in range(25):
        can.step((0.0, 0.0))  # let the clamped bump relax onto the sheet
    sx, sy = can.decode()
    seq = [(0.03, 0.00)] * 8 + [(0.00, 0.025)] * 8 + [(-0.015, -0.010)] * 6
    tx, ty = sx, sy
    min_amp = amp
    for vx, vy in seq:
        can.step((vx, vy))
        tx += vx
        ty += vy
        min_amp = min(min_amp, can.bump_amplitude())
    dx, dy = can.decode()
    pi_err = math.hypot(_circ_diff(dx, tx, can.spacing), _circ_diff(dy, ty, can.spacing))

    # ---- 3. a closed velocity LOOP returns the bump to where it started (drift-bounded integration) ----
    can.reset((0.30, 0.30))
    for _ in range(20):
        can.step((0.0, 0.0))
    start = can.decode()
    for vx, vy in [(0.03, 0)] * 6 + [(0, 0.03)] * 6 + [(-0.03, 0)] * 6 + [(0, -0.03)] * 6:
        can.step((vx, vy))
    end = can.decode()
    loop_err = math.hypot(
        _circ_diff(end[0], start[0], can.spacing), _circ_diff(end[1], start[1], can.spacing)
    )

    # ---- 4. LANDMARK reset re-clamps the bump exactly onto a supplied location ----
    can.reset((0.70, 0.20))
    r = can.decode()
    reset_err = math.hypot(_circ_diff(r[0], 0.70, can.spacing), _circ_diff(r[1], 0.20, can.spacing))

    # ---- 5. determinism: two fresh networks (same seed) integrate an identical trajectory ----
    a, b = SpikingGridCAN(n=16, seed=0), SpikingGridCAN(n=16, seed=0)
    a.reset((0.1, 0.1))
    b.reset((0.1, 0.1))
    det = all(a.step((0.02, 0.015)) == b.step((0.02, 0.015)) for _ in range(30))

    # ---- 6. DRIFT: pure dead-reckoning slides away from ground truth as the path lengthens ----
    n_circ = 4  # honest horizon: past ~4.8 circuits the error exceeds the half-period and the torus WRAPS it
    free_errs, free_worst, free_amp = _run_circuits(_settled(), n_circ, use_landmarks=False)
    free_final = free_errs[-1]
    per_circuit = free_final / n_circ
    torus_max = math.hypot(0.5, 0.5)  # largest error representable on the periodic sheet

    # ---- 7. LANDMARK phase reset: the SAME trajectory, error kept BOUNDED instead of accumulating ----
    lm_errs, lm_worst, lm_amp = _run_circuits(_settled(), 8, use_landmarks=True)
    lm_final = lm_errs[n_circ - 1]  # same horizon as the no-landmark run, for a fair head-to-head
    lm_long = max(lm_errs)  # ... and it stays bounded for twice as long

    # ---- 8. the correction is GRADED (monotone in gain) and leaves the bump alive ----
    grades = [(g,) + _graded_probe(g) for g in (0.0, 0.5, 1.0)]
    gaps_after = [
        row[3] for row in grades
    ]  # gap-to-anchor after correcting at gain 0.0 / 0.5 / 1.0
    corrs = [row[1] for row in grades]
    grade_amp = min(row[4] for row in grades)

    # ---- 9. FAILING CONTROL (kept, not tuned away): the raw mixture EXTINGUISHES the bump when the
    #         anchor/bump mismatch is large. capture_threshold=None disables the cue-capture rule. ----
    _, fail_gap0, _, fail_amp = _graded_probe(0.6, walk=12, capture_threshold=None, settle=20)
    _, _, ok_gap, ok_amp = _graded_probe(0.6, walk=12, capture_threshold=0.2, settle=20)

    # ---- 10. determinism of the landmark path itself ----
    det_lm = (
        _run_circuits(_settled(), 3, use_landmarks=True)[0]
        == _run_circuits(_settled(), 3, use_landmarks=True)[0]
    )

    print("=" * 72)
    print("SPIKING GRID CAN -- 2-D continuous-attractor path integration by a moving LIF bump")
    print("=" * 72)
    print(
        f"sheet {can.n}x{can.n} LIF ({can.N} neurons); center-surround VMe(k={can.kappa_e:.1f})-VMi; "
        f"shift gain {can._gain:.2f}"
    )
    print(
        f"persist: clamp (0.40,0.30) -> hold ({hold[0]:.3f},{hold[1]:.3f}) err {persist_err:.4f}, amp {amp:.3f}"
    )
    print(
        f"path integration: target ({tx:.4f},{ty:.4f}) -> decoded ({dx:.4f},{dy:.4f}) err {pi_err:.4f} "
        f"(min bump amp {min_amp:.3f})"
    )
    print(
        f"closed loop: start ({start[0]:.3f},{start[1]:.3f}) -> end ({end[0]:.3f},{end[1]:.3f}) "
        f"return err {loop_err:.4f}"
    )
    print(f"landmark reset -> ({r[0]:.3f},{r[1]:.3f}) err {reset_err:.4f}; determinism {det}")
    print("-" * 72)
    print(
        f"DRIFT (no landmarks), {n_circ} open circuits x {CIRCUIT_STEPS} ticks at v={CIRCUIT_V} "
        f"(= 1 grid period each):"
    )
    print(
        f"  per-circuit error {[round(e, 4) for e in free_errs]} -> DRIFTS to {free_final:.4f} "
        f"({per_circuit:.4f}/period, {100 * free_final / torus_max:.0f}% of the {torus_max:.4f} torus max)"
    )
    print(
        f"  the bump itself stays healthy while drifting (min amp {free_amp:.3f}) -- it is LOST, not dead"
    )
    print("LANDMARK PHASE RESET, same trajectory ('home' sighted once per circuit, gain 0.6):")
    print(
        f"  per-circuit error {[round(e, 4) for e in lm_errs]} -> BOUNDED at {lm_final:.4f} "
        f"@circuit {n_circ} (worst {lm_worst:.4f}, still <= {lm_long:.4f} at circuit 8), min amp {lm_amp:.3f}"
    )
    print(
        f"  head-to-head @circuit {n_circ}: {free_final:.4f} (free) vs {lm_final:.4f} (landmark) "
        f"= {free_final / max(lm_final, 1e-9):.0f}x lower; worst-case {free_worst:.4f} vs {lm_worst:.4f}"
    )
    print("GRADED correction (gap to anchor 0.1510 before; gain 0.0 / 0.5 / 1.0):")
    print(
        f"  correction {[round(c, 4) for c in corrs]} -> residual gap {[round(g, 4) for g in gaps_after]} "
        f"(monotone, gain 0 = no-op, gain 1 = full re-clamp); min amp {grade_amp:.3f}"
    )
    print("HONEST FAILURE (kept, not tuned away) -- raw mixture at a LARGE anchor/bump mismatch:")
    print(
        f"  gap {fail_gap0:.4f} + capture DISABLED -> bimodal mixture, bump EXTINGUISHED after settling "
        f"(amp {fail_amp:.3f})"
    )
    print(
        f"  same gap WITH the cue-capture rule -> unimodal re-clamp, residual {ok_gap:.4f}, amp {ok_amp:.3f}"
    )
    print(f"determinism (landmark path) {det_lm}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, no backprop ----
    assert amp > 0.1 and min_amp > 0.05, (
        f"a single spiking bump must survive (amp {amp:.3f}, min {min_amp:.3f})"
    )
    assert persist_err < 0.06, (
        f"the bump must persist at a clamped location (err {persist_err:.4f})"
    )
    assert pi_err < 0.06, f"decoded position must track the velocity integral (err {pi_err:.4f})"
    assert loop_err < 0.06, (
        f"a closed velocity loop must return the bump near its start (err {loop_err:.4f})"
    )
    assert reset_err < 1e-6, (
        f"landmark reset must re-clamp the bump exactly onto pos (err {reset_err:.4f})"
    )
    assert det, "same seed must integrate an identical trajectory (deterministic)"

    # (a) pure dead-reckoning DRIFTS, and the drift GROWS monotonically with path length
    assert free_errs == sorted(free_errs), (
        f"dead-reckoning error must grow monotonically with path length (got {free_errs})"
    )
    assert free_final > 0.3, (
        f"drift over {n_circ} circuits must be substantial, not negligible (got {free_final:.4f})"
    )
    assert free_amp > 0.5, (
        f"the drifting bump must still be ALIVE -- drift is a lost estimate, not a dead network "
        f"(min amp {free_amp:.3f})"
    )
    # (b) periodic landmark sightings keep the SAME trajectory's error BOUNDED and much lower
    assert lm_final < free_final / 5.0, (
        f"landmark reset must cut the error substantially ({lm_final:.4f} vs free {free_final:.4f})"
    )
    assert lm_worst < 0.25 and lm_long < 0.25, (
        f"landmark-corrected error must stay BOUNDED over 8 circuits (worst {lm_worst:.4f}, max {lm_long:.4f})"
    )
    # (c) the correction is GRADED (monotone in gain) and does not destroy the bump
    assert gaps_after[0] > gaps_after[1] > gaps_after[2], (
        f"correction must be GRADED: bigger gain -> smaller residual gap (got {gaps_after})"
    )
    assert corrs[0] == 0.0 and gaps_after[2] < 1e-9, (
        f"gain 0 must be a no-op and gain 1 a full re-clamp onto the stored phase (got {corrs[0]:.4f}, "
        f"{gaps_after[2]:.6f})"
    )
    assert grade_amp > 0.5 and lm_amp > 0.5, (
        f"a graded correction must NOT destroy the bump (min amp {grade_amp:.3f} / {lm_amp:.3f})"
    )
    # (d) the KEPT failure: the raw mixture really does extinguish the bump at a large mismatch
    assert fail_amp < 0.01, (
        f"HONEST FAILING CONTROL: the raw mixture at gap {fail_gap0:.4f} is expected to extinguish the bump "
        f"(amp {fail_amp:.3f}) -- if this now passes, the documented limit changed"
    )
    # (the residual here is NOT 0: the capture re-clamps exactly onto the anchor, then 20 free-running
    #  settle ticks let the bump relax onto the discrete sheet -- honest attractor pinning, ~0.03 period)
    assert ok_amp > 0.5 and ok_gap < 0.05, (
        f"the cue-capture rule must survive the same mismatch (amp {ok_amp:.3f}, residual {ok_gap:.6f})"
    )
    assert det_lm, "the landmark-corrected trajectory must be deterministic too"
    print(
        f"self-check OK: {can.N}-LIF spiking bump path-integrates velocity to target ({tx:.3f},{ty:.3f}) "
        f"decoded ({dx:.3f},{dy:.3f}) err {pi_err:.4f}; landmark reset re-clamps; loop err {loop_err:.4f}; "
        f"dead-reckoning DRIFTS {free_final:.4f} over {n_circ} periods vs {lm_final:.4f} with landmark "
        f"phase reset ({free_final / max(lm_final, 1e-9):.0f}x, bounded <= {lm_long:.4f} over 8); correction "
        f"graded (residual {gaps_after[0]:.4f}/{gaps_after[1]:.4f}/{gaps_after[2]:.4f} at gain 0/0.5/1, "
        f"amp {grade_amp:.3f}); raw mixture at gap {fail_gap0:.4f} KILLS the bump (amp {fail_amp:.3f}, kept); "
        f"deterministic"
    )


if __name__ == "__main__":
    main()
