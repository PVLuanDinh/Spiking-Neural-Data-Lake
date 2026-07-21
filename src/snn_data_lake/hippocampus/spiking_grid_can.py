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

DISTINCTNESS. Not grid_cells.py / grid_place_map.py (analytic RATE fields, no recurrent dynamics). Not
ring_attractor.py (1-D heading ring, scalar angular velocity). ZERO coupling to the arm / safety envelope:
representation-only, imports only the stdlib, issues no commands, exposes no actuator route.

HONEST SCOPE. One grid module (single spacing) on a small toroidal sheet, so position is resolved only
WITHIN one period, not over an open world (a multi-scale population would disambiguate -- see grid_cells.py).
Velocity integration is accurate within a bounded velocity range (the self-calibrated linear regime, |v| up
to a few percent of the period per tick); larger per-tick velocities out-run the bump and integrate
sub-linearly. Deterministic (no stochastic units).

  python -m snn_data_lake.hippocampus.spiking_grid_can     # self-check; also runnable by run_selfchecks.py
"""

import math


def _circ_diff(a, b, period):
    """Signed smallest difference a - b on a ring of circumference `period`, in (-period/2, period/2]."""
    return (a - b + period / 2.0) % period - period / 2.0


def _zeros(n):
    """n-by-n matrix of 0.0 (numpy np.zeros((n, n)) -> nested lists)."""
    return [[0.0] * n for _ in range(n)]


def _transpose(M, n):
    """Transpose of an n-by-n matrix (numpy M.T -> nested lists)."""
    return [[M[j][i] for j in range(n)] for i in range(n)]


def _matmul(A, B, n):
    """n-by-n matrix product A @ B (pure-Python triple loop; out[i][j] = sum_k A[i][k]*B[k][j])."""
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Oi = out[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Oi[j] += aik * Bk[j]
    return out


def _row_sums(M, n):
    """Sum along axis=1: one value per row (numpy M.sum(axis=1))."""
    out = []
    for i in range(n):
        s = 0.0
        Mi = M[i]
        for j in range(n):
            s += Mi[j]
        out.append(s)
    return out


def _col_sums(M, n):
    """Sum along axis=0: one value per column (numpy M.sum(axis=0))."""
    out = []
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += M[i][j]
        out.append(s)
    return out


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
        self.N = self.n * self.n

        n = self.n
        theta = [
            2.0 * math.pi * i / n for i in range(n)
        ]  # phase of each sheet row/col on the torus
        self.theta = theta
        cos_d = [[math.cos(theta[i] - theta[j]) for j in range(n)] for i in range(n)]
        sin_d = [[math.sin(theta[i] - theta[j]) for j in range(n)] for i in range(n)]
        self.VMe = [
            [math.exp(kappa_e * (cos_d[i][j] - 1.0)) for j in range(n)] for i in range(n)
        ]  # sharp excitatory kernel (center)
        self.VMi = [
            [math.exp(kappa_i * (cos_d[i][j] - 1.0)) for j in range(n)] for i in range(n)
        ]  # broad inhibitory kernel (surround)
        self.dVMe = [
            [-kappa_e * sin_d[i][j] * self.VMe[i][j] for j in range(n)] for i in range(n)
        ]  # angular derivative -> asymmetric shift
        self.dVMi = [
            [-kappa_i * sin_d[i][j] * self.VMi[i][j] for j in range(n)] for i in range(n)
        ]  # (2-D generalization of ring w_asym)
        self.dVMeT = _transpose(self.dVMe, n)  # dVMe.T, precomputed (constant across ticks)
        self.dVMiT = _transpose(self.dVMi, n)  # dVMi.T, precomputed

        self.v = _zeros(n)  # LIF membrane potentials
        self.syn = _zeros(n)  # spike-driven synaptic-activity field
        self._gain = 1.0
        if calibrate:
            self._gain = self._calibrate(settle_steps)
        self.reset((0.0, 0.0))

    # ---- recurrent connectivity: center-surround + velocity-gated asymmetric shift ----
    def _currents(self, vx, vy):
        """Recurrent synaptic current onto every neuron: Mexican-hat sustain + velocity-driven shift."""
        n = self.n
        syn = self.syn
        VMe, VMi, dVMe, dVMi = self.VMe, self.VMi, self.dVMe, self.dVMi
        dVMeT, dVMiT = self.dVMeT, self.dVMiT
        E = _matmul(VMe, syn, n)  # VMe @ syn
        Im = _matmul(VMi, syn, n)  # VMi @ syn
        dE = _matmul(dVMe, syn, n)  # dVMe @ syn
        dIm = _matmul(dVMi, syn, n)  # dVMi @ syn
        rec_e = _matmul(E, VMe, n)  # VMe @ syn @ VMe
        rec_i = _matmul(Im, VMi, n)  # VMi @ syn @ VMi
        sr_e = _matmul(dE, VMe, n)  # dVMe @ syn @ VMe   (d/drow, y axis)
        sr_i = _matmul(dIm, VMi, n)  # dVMi @ syn @ VMi
        sc_e = _matmul(E, dVMeT, n)  # VMe @ syn @ dVMe.T (d/dcol, x axis)
        sc_i = _matmul(Im, dVMiT, n)  # VMi @ syn @ dVMi.T
        a_e, a_i = self.a_e, self.a_i
        g = self.vel_gain * self._gain
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Oi = out[i]
            for j in range(n):
                rec = a_e * rec_e[i][j] - a_i * rec_i[i][j]  # separable center-surround
                shift_row = a_e * sr_e[i][j] - a_i * sr_i[i][j]  # d/drow  (y axis)
                shift_col = a_e * sc_e[i][j] - a_i * sc_i[i][j]  # d/dcol  (x axis)
                Oi[j] = (rec + g * (vy * shift_row + vx * shift_col)) / n
        return out

    def step(self, velocity=(0.0, 0.0)):
        """Advance the bump one integration tick under `velocity`=(vx, vy); return decode()."""
        vx, vy = velocity
        I = self._currents(vx, vy)
        n = self.n
        v, syn = self.v, self.syn
        leak, vth, syn_decay = self.leak, self.vth, self.syn_decay
        for i in range(n):
            vi, Ii, syni = v[i], I[i], syn[i]
            for j in range(n):
                nv = leak * vi[j] + Ii[j]  # LIF membrane integration
                if nv >= vth:  # threshold
                    spike = 1.0
                    nv = 0.0  # reset to 0
                else:
                    spike = 0.0
                vi[j] = nv
                syni[j] = syni[j] * syn_decay + spike  # spike-driven recurrent synapses
        return self.decode()

    # ---- landmark reset: clamp the bump onto the phase encoding `pos` (place->grid phase reset) ----
    def _bump(self, pos):
        x, y = pos
        n = self.n
        theta = self.theta
        ke = self.kappa_e
        py = 2.0 * math.pi * (y / self.spacing)
        px = 2.0 * math.pi * (x / self.spacing)
        row = [math.exp(ke * (math.cos(theta[i] - py) - 1.0)) for i in range(n)]
        col = [math.exp(ke * (math.cos(theta[j] - px) - 1.0)) for j in range(n)]
        return [[row[i] * col[j] for j in range(n)] for i in range(n)]

    def reset(self, pos=(0.0, 0.0)):
        """Clamp the bump onto the location `pos` (the optional landmark / place->grid phase reset)."""
        self.syn = self._bump(pos)
        self.v = _zeros(self.n)
        return self.decode()

    # ---- readout: population-vector (circular-mean) center of the spiking bump ----
    def _phase(self, marginal):
        theta = self.theta
        s = 0.0
        c = 0.0
        for i in range(self.n):
            s += marginal[i] * math.sin(theta[i])
            c += marginal[i] * math.cos(theta[i])
        return math.atan2(s, c)

    def bump_center(self):
        """Bump location on the n-by-n sheet as continuous (row, col) in [0, n)."""
        n = self.n
        row = self._phase(_row_sums(self.syn, n)) % (2 * math.pi) / (2 * math.pi) * self.n
        col = self._phase(_col_sums(self.syn, n)) % (2 * math.pi) / (2 * math.pi) * self.n
        return (row, col)

    def decode(self):
        """Decoded continuous position (x, y) within one grid period [0, spacing)."""
        n = self.n
        y = self._phase(_row_sums(self.syn, n)) % (2 * math.pi) / (2 * math.pi) * self.spacing
        x = self._phase(_col_sums(self.syn, n)) % (2 * math.pi) / (2 * math.pi) * self.spacing
        return (x, y)

    def bump_amplitude(self):
        """Peak synaptic activity of the bump -- a liveness measure (single stable bump => > 0)."""
        m = self.syn[0][0]
        for row in self.syn:
            for val in row:
                if val > m:
                    m = val
        return float(m)

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
    print(
        f"self-check OK: {can.N}-LIF spiking bump path-integrates velocity to target ({tx:.3f},{ty:.3f}) "
        f"decoded ({dx:.3f},{dy:.3f}) err {pi_err:.4f}; landmark reset re-clamps; loop err {loop_err:.4f}; "
        f"deterministic"
    )


if __name__ == "__main__":
    main()
