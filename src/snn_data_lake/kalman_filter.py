"""
Kalman filter -- recursive linear-Gaussian optimal state estimation via a predict/update
cycle with the Kalman gain and covariance propagation (pure stdlib, zero deps).

A linear-Gaussian state-space system
    x_{k+1} = F x_k + w_k        w_k ~ N(0, Q)   (process noise)
    z_k     = H x_k + v_k        v_k ~ N(0, R)   (measurement noise)
is optimally estimated (minimum mean-square error) by a two-step recursion that carries a
state estimate x and its covariance P forward in time:

    PREDICT:  x_pred = F x                  P_pred = F P F^T + Q
    UPDATE:   y = z - H x_pred              S = H P_pred H^T + R
              K = P_pred H^T S^{-1}         x = x_pred + K y      P = (I - K H) P_pred

K is the KALMAN GAIN: it optimally weights the model's prediction against the incoming
measurement by their relative uncertainties. A big R (noisy sensor) shrinks K (trust the
model); a small R (precise sensor) grows K (trust the measurement). P tracks how confident
the estimate is and converges to the Riccati fixed point for a stationary system.
Kalman, "A New Approach to Linear Filtering and Prediction Problems", J Basic Eng 82:35 (1960).

The demo model is a 2-D CONSTANT-VELOCITY tracker: state x = (position, velocity),
    F = [[1, dt], [0, 1]]   H = [[1, 0]]   (measure position only, so S is SCALAR)
with a discrete white-noise-acceleration process noise Q and a scalar measurement variance R.
The tiny linear algebra (2x2 matmul, transpose, matrix-vector, the (I - K H) update) is done
with plain nested lists -- no numpy.

DISTINCTNESS (stated, having read the siblings):
  - cue_integration.py is STATIC: it fuses several independent estimates of ONE quantity in
    closed form by inverse-variance weighting -- no time, no state, no recursion. The Kalman
    filter is the DYNAMIC recursive generalization: it propagates a STATE and its COVARIANCE
    through a motion model and optimally corrects each step with a measurement via the Kalman
    gain. (The single UPDATE step does reduce to inverse-variance fusion of prediction and
    measurement, but the filter adds the temporal recursion + covariance propagation the
    static fuser lacks.)
  - forward_model.py PREDICTS the next sensory state from an efference copy and FLAGS external
    perturbations, but it never optimally CORRECTS the estimate with the measurement -- no
    gain, no covariance. THIS filter tracks uncertainty (P) and optimally BLENDS prediction
    with measurement through K.
  - ring_attractor.py's ProprioceptiveEstimator DEAD-RECKONS commanded joint deltas into an
    angle -- integration only, no measurement update, no uncertainty. THIS filter corrects with
    measurements and reports calibrated uncertainty.

ponytail: the ceiling is a linear-Gaussian Kalman filter on a small constant-velocity model
with synthetic noise -- the optimal-linear-estimation level. NOT an extended/unscented KF, no
model-mismatch handling, no spiking implementation; not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.kalman_filter
"""
import math
import random


def _matvec(matrix, vector):
    """Matrix (m x n) times vector (n) -> m-vector."""
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]


def _matmul(a, b):
    """Matrix (m x n) times matrix (n x p) -> (m x p)."""
    inner = len(b)
    cols = len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(len(a))]


def _transpose(matrix):
    """Transpose a rectangular matrix."""
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def _mat_add(a, b):
    """Elementwise a + b for equally shaped matrices."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_sub(a, b):
    """Elementwise a - b for equally shaped matrices."""
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _identity(n):
    """n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


class KalmanFilter:
    """Linear-Gaussian Kalman filter with a SCALAR measurement (H is a single row).

    Carries a state estimate x (n-vector) and its covariance P (n x n). predict() advances both
    through the motion model F with process noise Q; update(z) corrects them with one scalar
    measurement z through the Kalman gain K; step(z) does predict-then-update and returns the
    corrected state. Deterministic -- holds no RNG."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = [list(row) for row in F]
        self.H = [list(row) for row in H]          # single-row measurement matrix (scalar z)
        self.Q = [list(row) for row in Q]
        self.R = float(R)
        self.x = [float(v) for v in x0]
        self.P = [list(row) for row in P0]
        self.n = len(self.x)
        self.K = [0.0] * self.n                    # last Kalman gain (set by update)

    def predict(self):
        """PREDICT: x_pred = F x ; P_pred = F P F^T + Q. Returns the predicted state."""
        self.x = _matvec(self.F, self.x)
        fp = _matmul(self.F, self.P)
        self.P = _mat_add(_matmul(fp, _transpose(self.F)), self.Q)
        return list(self.x)

    def update(self, z):
        """UPDATE: fold scalar measurement z in through the Kalman gain. Returns the state.

        y = z - H x_pred (innovation); S = H P H^T + R (scalar); K = P H^T / S;
        x = x_pred + K y; P = (I - K H) P_pred."""
        h = self.H[0]
        innovation = float(z) - sum(h[i] * self.x[i] for i in range(self.n))
        p_ht = _matvec(self.P, h)                                  # P H^T (n-vector)
        s = sum(h[i] * p_ht[i] for i in range(self.n)) + self.R    # scalar S = H P H^T + R
        self.K = [p_ht[i] / s for i in range(self.n)]              # Kalman gain K = P H^T / S
        self.x = [self.x[i] + self.K[i] * innovation for i in range(self.n)]
        kh = [[self.K[i] * h[j] for j in range(self.n)] for i in range(self.n)]
        self.P = _matmul(_mat_sub(_identity(self.n), kh), self.P)  # P = (I - K H) P_pred
        return list(self.x)

    def step(self, z):
        """One full recursion: predict, then update with z. Returns the estimated state."""
        self.predict()
        return self.update(z)

    @property
    def state(self):
        """Current state estimate (copy)."""
        return list(self.x)

    @property
    def covariance(self):
        """Current estimate covariance P (copy)."""
        return [list(row) for row in self.P]


class ConstantVelocityTracker:
    """2-D constant-velocity Kalman tracker (position, velocity) with a position-only sensor.

    Builds F, H, Q, R for a discrete white-noise-acceleration model (a random acceleration of
    variance `process_var` per step drives the state), simulate(n) generates a TRUE trajectory
    plus noisy position measurements from that SAME generative model (so the filter is
    calibrated: its Q, R match reality), and run() filters a measurement sequence. Deterministic
    given `seed`."""

    def __init__(self, dt, process_var, meas_var, seed=0, pos0=0.0, vel0=1.0):
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self.seed = seed
        self.pos0 = float(pos0)
        self.vel0 = float(vel0)
        dt = self.dt
        q = self.process_var
        self.F = [[1.0, dt], [0.0, 1.0]]
        self.H = [[1.0, 0.0]]
        # discrete white-noise-acceleration Q = G G^T q, with G = [dt^2/2, dt]^T
        self.Q = [[q * dt ** 4 / 4.0, q * dt ** 3 / 2.0],
                  [q * dt ** 3 / 2.0, q * dt ** 2]]
        self.R = self.meas_var
        # agnostic prior: zero state, loose covariance (the transient washes out)
        self.kf = KalmanFilter(self.F, self.H, self.Q, self.R, [0.0, 0.0],
                               [[10.0, 0.0], [0.0, 10.0]])

    def simulate(self, n):
        """Generate n true states s_k (s_{k+1}=F s_k + G a_k) and noisy position measurements."""
        rng = random.Random(self.seed)
        dt = self.dt
        g0, g1 = dt * dt / 2.0, dt
        sigma_a = math.sqrt(self.process_var)
        sigma_z = math.sqrt(self.meas_var)
        x = [self.pos0, self.vel0]
        true_states, measurements = [], []
        for _ in range(n):
            true_states.append(list(x))
            measurements.append(x[0] + rng.gauss(0.0, sigma_z))
            a = rng.gauss(0.0, sigma_a)                    # random acceleration this step
            x = [x[0] + dt * x[1] + g0 * a, x[1] + g1 * a]
        return true_states, measurements

    def run(self, measurements):
        """Filter a measurement sequence -> (estimated states, P[0][0] history)."""
        estimates, p_pos_history = [], []
        for z in measurements:
            estimates.append(self.kf.step(z))
            p_pos_history.append(self.kf.P[0][0])
        return estimates, p_pos_history


def main():
    dt = 1.0
    process_var = 0.01     # random-acceleration variance per step (drives the CV model)
    meas_var = 1.0         # position sensor variance R (sensor std 1.0)
    n_steps = 3000         # long fixed-seed run -> stable empirical error statistics
    burn = 500             # discard the filter/covariance transient before measuring stats

    tracker = ConstantVelocityTracker(dt, process_var, meas_var, seed=7)
    true_states, measurements = tracker.simulate(n_steps)
    estimates, p_pos_history = tracker.run(measurements)

    # ---- 1. optimal tracking: Kalman position RMSE beats sensor AND prediction-only ----
    rmse_meas = math.sqrt(sum((measurements[k] - true_states[k][0]) ** 2
                              for k in range(n_steps)) / n_steps)
    # prediction-only / open-loop: start from the TRUE initial state (fairest), dead-reckon with
    # F, NEVER correct -> the unmodeled random accelerations accumulate and it drifts off.
    ol = [true_states[0][0], true_states[0][1]]
    ol_pos = []
    for _ in range(n_steps):
        ol_pos.append(ol[0])
        ol = [ol[0] + dt * ol[1], ol[1]]
    rmse_pred_only = math.sqrt(sum((ol_pos[k] - true_states[k][0]) ** 2
                                   for k in range(n_steps)) / n_steps)
    rmse_kalman = math.sqrt(sum((estimates[k][0] - true_states[k][0]) ** 2
                                for k in range(n_steps)) / n_steps)

    # ---- 2. covariance convergence + consistency ----
    p_final = p_pos_history[-1]
    # convergence: successive P[0][0] changes vanish after burn-in (the Riccati fixed point)
    max_late_delta = max(abs(p_pos_history[k] - p_pos_history[k - 1])
                         for k in range(burn, n_steps))
    # consistency: empirical mean-square position error ~ reported P[0][0] (calibrated filter)
    errs = [estimates[k][0] - true_states[k][0] for k in range(burn, n_steps)]
    emp_err_var = sum(e * e for e in errs) / len(errs)

    # ---- 3. gain balances noise: large R -> small K[0]; small R -> large K[0] ----
    noisy = ConstantVelocityTracker(dt, process_var, meas_var=100.0, seed=1)
    _tn, m_noisy = noisy.simulate(400)
    noisy.run(m_noisy)
    gain_noisy = noisy.kf.K[0]
    precise = ConstantVelocityTracker(dt, process_var, meas_var=0.01, seed=1)
    _tp, m_precise = precise.simulate(400)
    precise.run(m_precise)
    gain_precise = precise.kf.K[0]

    # ---- 4. determinism: same params/seed -> identical filtered trajectory ----
    t_a = ConstantVelocityTracker(dt, process_var, meas_var, seed=7)
    _sa, ma = t_a.simulate(n_steps)
    est_a, _pa = t_a.run(ma)
    t_b = ConstantVelocityTracker(dt, process_var, meas_var, seed=7)
    _sb, mb = t_b.simulate(n_steps)
    est_b, _pb = t_b.run(mb)
    deterministic = est_a == est_b

    print("=" * 70)
    print("KALMAN FILTER -- recursive optimal state estimation (constant-velocity tracker)")
    print("=" * 70)
    print(f"RMSE to truth: measurement {rmse_meas:.4f}  prediction-only {rmse_pred_only:.4f}  "
          f"Kalman {rmse_kalman:.4f}  -> Kalman beats sensor AND model")
    print(f"covariance   : reported P[0][0] {p_final:.4f}  vs empirical error var {emp_err_var:.4f}"
          f"  (late max delta-P {max_late_delta:.2e} -> converged)")
    print(f"Kalman gain  : noisy R=100 -> K[0] {gain_noisy:.4f} (small, trust model); "
          f"precise R=0.01 -> K[0] {gain_precise:.4f} (large, trust measurement)")
    print(f"determinism  : identical filtered trajectory {deterministic}  "
          f"(n_steps {n_steps}, burn {burn})")
    print("=" * 70)

    # ---- self-checks (genuine, not rigged) ----
    # 1. the optimal combination beats both the sensor alone and the model alone.
    assert rmse_kalman < rmse_meas, (
        f"Kalman must beat the raw sensor ({rmse_kalman:.4f} < {rmse_meas:.4f})")
    assert rmse_kalman < rmse_pred_only, (
        f"Kalman must beat prediction-only dead-reckoning "
        f"({rmse_kalman:.4f} < {rmse_pred_only:.4f})")

    # 2. covariance converges to the Riccati fixed point AND is consistent with reality.
    assert max_late_delta < 1e-6, (
        f"P[0][0] must converge to the Riccati fixed point (late max delta {max_late_delta:.2e})")
    assert abs(emp_err_var - p_final) / p_final < 0.15, (
        f"reported P must match the empirical error variance -- neither over- nor under-confident "
        f"({p_final:.4f} vs {emp_err_var:.4f})")

    # 3. the gain balances noise: large R -> small gain, small R -> large gain.
    assert gain_noisy < 0.2, f"large R (noisy sensor) must give a small gain ({gain_noisy:.4f})"
    assert gain_precise > 0.5, (
        f"small R (precise sensor) must give a large gain ({gain_precise:.4f})")
    assert gain_precise > 3 * gain_noisy, (
        f"the precise sensor must earn a far larger gain than the noisy one "
        f"({gain_precise:.4f} vs {gain_noisy:.4f})")

    # 4. determinism: same params/seed -> identical filtered trajectory.
    assert deterministic, "same params/seed must reproduce an identical filtered trajectory"

    print(f"self-check OK: RMSE Kalman {rmse_kalman:.4f} < measurement {rmse_meas:.4f} and "
          f"prediction-only {rmse_pred_only:.4f}; reported P {p_final:.4f} ~ empirical error var "
          f"{emp_err_var:.4f} (converged, delta {max_late_delta:.1e}); gain noisy {gain_noisy:.4f} "
          f"<< precise {gain_precise:.4f}; deterministic")


if __name__ == "__main__":
    main()
