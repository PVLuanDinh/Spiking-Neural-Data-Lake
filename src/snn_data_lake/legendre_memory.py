"""
Legendre Memory Unit (LMU) delay network -- an optimal linear window memory (pure stdlib).

A Legendre Memory Unit maintains a rolling window of its scalar input, of length theta, by
projecting that recent history onto the first d shifted-Legendre polynomials. The state is a
d-vector m whose components are the Legendre COEFFICIENTS of the input over the last theta; the
continuous dynamics are the delay-network state space  theta * m_dot = A m + B u  (Voelker, Kajic
& Eliasmith, NeurIPS 2019, "Legendre Memory Units"), with A, B the shifted-Legendre state-space
realization of a pure delay:
    A[i][j] = (2i + 1) * ( -1            if i < j  else  (-1)**(i - j + 1) )
    B[i]    = (2i + 1) * (-1)**i
The state is advanced one input sample at a time by forward Euler,  m += (dt/theta)*(A m + B u),
with dt/theta kept small for stability. Because m encodes the whole window on a polynomial basis,
the input at ANY lag theta_prime in [0, theta] is decoded by one linear read-out,
    u_hat(theta_prime) = sum_i m_i * P_i(2r - 1),   r = theta_prime / theta in [0, 1],
where P_i is the shifted-Legendre polynomial (Legendre recurrence). r = 0 decodes "now" (u(t)),
r = 1 decodes the full delay u(t - theta).

The delay-line signature: once the window has filled, reconstruct(1.0) tracks u(t - theta) -- the
oldest edge of the stored window reproduces the delayed input, and the intermediate lags reproduce
the whole recent trajectory. The convention (which end of [0,1] is "now") is fixed EMPIRICALLY here:
only r = theta_prime/theta makes the full-delay decode match u(t - theta); the flipped orientation
r = 1 - theta_prime/theta corrupts the interior lags, since P_i(-x) = (-1)**i * P_i(x).

DISTINCT from existing modules (read them):
- research/nef_integrator.py holds a SINGLE scalar (a line-attractor working memory of one value).
  The LMU holds the ENTIRE recent WINDOW of the input -- a whole trajectory over the last theta,
  reconstructable at ANY lag -- by projecting it onto d Legendre coefficients. A window vs a point.
- htm_sequence_memory.py PREDICTS the next SDR from context (a discrete high-order sequence model).
  The LMU does not predict; it optimally COMPRESSES and reconstructs the continuous input history
  (a linear delay network).
- theta_gamma_code.py holds K DISCRETE items in oscillatory phase slots; the LMU holds a CONTINUOUS
  window on a polynomial basis. ring_attractor.py is a single SPATIAL bump / angle integrator --
  again one value, not a window.

Ceiling -- a discretized linear LMU delay network (forward-Euler-integrated shifted-
Legendre state space) reconstructing a synthetic input window at the continuous-memory level. NOT
the trained recurrent LMU cell and NOT spiking; small order / step so it is stable and reproducible;
not cognition. Deterministic (no randomness; seed kept only for API symmetry). Credit Voelker, Kajic
& Eliasmith 2019.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.legendre_memory
"""
import math


def _shifted_legendre(i, r):
    """i-th shifted-Legendre polynomial on [0, 1], evaluated at r via the Legendre recurrence.

    Standard Legendre P_i at x = 2r - 1 in [-1, 1]:  P_0 = 1, P_1 = x,
    (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1}."""
    x = 2.0 * r - 1.0
    if i == 0:
        return 1.0
    if i == 1:
        return x
    p_prev, p_cur = 1.0, x
    for n in range(1, i):
        p_next = ((2 * n + 1) * x * p_cur - n * p_prev) / (n + 1)
        p_prev, p_cur = p_cur, p_next
    return p_cur


def _decode_state(state, lag_fraction):
    """Read one lag out of a Legendre-coefficient state: sum_i state_i * P_i(lag_fraction)."""
    return sum(state[i] * _shifted_legendre(i, lag_fraction) for i in range(len(state)))


class LegendreMemory:
    """LMU delay network: a d-dim linear state m holding the last theta of input on a Legendre basis.

    theta * m_dot = A m + B u (shifted-Legendre delay realization), advanced by forward Euler per
    input sample. reconstruct(lag_fraction) decodes u at lag = lag_fraction*theta from the CURRENT
    state; the whole window is readable, not just one value."""

    def __init__(self, order=8, theta=1.0, dt=0.002, seed=0):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.theta = float(theta)
        self.dt = float(dt)
        self.seed = seed                       # kept for API symmetry; the system is deterministic
        self.A, self.B = self._delay_matrices(order)
        self.reset()

    @staticmethod
    def _delay_matrices(d):
        """Voelker's shifted-Legendre state-space realization (A, B) of a pure delay of length theta."""
        a = [[0.0] * d for _ in range(d)]
        b = [0.0] * d
        for i in range(d):
            for j in range(d):
                if i < j:
                    a[i][j] = (2 * i + 1) * -1.0
                else:
                    a[i][j] = (2 * i + 1) * ((-1.0) ** (i - j + 1))
            b[i] = (2 * i + 1) * ((-1.0) ** i)
        return a, b

    def reset(self):
        """Clear the stored window (state back to zero)."""
        self.m = [0.0] * self.order

    def step(self, u):
        """Advance the linear state one input sample (forward Euler); return a copy of the new state."""
        ratio = self.dt / self.theta
        am = [sum(self.A[i][j] * self.m[j] for j in range(self.order)) for i in range(self.order)]
        self.m = [self.m[i] + ratio * (am[i] + self.B[i] * u) for i in range(self.order)]
        return list(self.m)

    def run(self, signal):
        """Feed a sequence of input samples; return the state trace (one state copy per sample)."""
        return [self.step(u) for u in signal]

    def reconstruct(self, lag_fraction):
        """Decode the input at lag = lag_fraction*theta (lag_fraction in [0, 1]) from the CURRENT state.

        lag_fraction = 0 -> "now" u(t);  lag_fraction = 1 -> the full delay u(t - theta)."""
        if not 0.0 <= lag_fraction <= 1.0:
            raise ValueError("lag_fraction must be in [0, 1]")
        return _decode_state(self.m, lag_fraction)

    def window(self, n_points=9):
        """Reconstruct the whole stored window on a uniform lag grid; [(lag_fraction, u_hat), ...]."""
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        grid = [k / (n_points - 1) for k in range(n_points)]
        return [(lf, self.reconstruct(lf)) for lf in grid]


def _sine(t, f):
    return math.sin(2.0 * math.pi * f * t)


def _nrmse_corr(pred, true):
    """RMSE, normalized RMSE (by std of true), and Pearson correlation between two equal series."""
    n = len(pred)
    mp = sum(pred) / n
    mt = sum(true) / n
    rmse = math.sqrt(sum((pred[k] - true[k]) ** 2 for k in range(n)) / n)
    std_t = math.sqrt(sum((v - mt) ** 2 for v in true) / n)
    cov = sum((pred[k] - mp) * (true[k] - mt) for k in range(n))
    sp = math.sqrt(sum((v - mp) ** 2 for v in pred))
    st = math.sqrt(sum((v - mt) ** 2 for v in true))
    corr = cov / (sp * st) if sp * st > 0 else 0.0
    nrmse = rmse / std_t if std_t > 0 else 0.0
    return rmse, nrmse, corr


def _feed_sine(lmu, f, n_windows=8):
    """Drive an LMU with sin(2 pi f t) for n_windows worth of samples; return the state trace."""
    steps = int(n_windows * lmu.theta / lmu.dt)
    return [lmu.step(_sine(k * lmu.dt, f)) for k in range(steps)]


def main():
    theta, dt, f, order = 1.0, 0.002, 0.75, 8
    n_win = int(theta / dt)
    lmu = LegendreMemory(order=order, theta=theta, dt=dt)
    trace = _feed_sine(lmu, f)
    steady = range(4 * n_win, len(trace))     # after the window has filled several times over

    # ---- 1. DELAY LINE (the test): reconstruct(1.0) tracks u(t - theta) over the steady run ----
    pred_full = [_decode_state(trace[k], 1.0) for k in steady]
    true_full = [_sine(k * dt - theta, f) for k in steady]
    rmse_full, nrmse_full, corr_full = _nrmse_corr(pred_full, true_full)

    # ---- 2. WHOLE-WINDOW READOUT: several lags each match u(t - lag*theta) (whole trajectory) ----
    lags = [0.25, 0.5, 0.75, 1.0]
    per_lag = {}
    for lf in lags:
        pred = [_decode_state(trace[k], lf) for k in steady]
        true = [_sine(k * dt - lf * theta, f) for k in steady]
        per_lag[lf] = _nrmse_corr(pred, true)      # (rmse, nrmse, corr)

    # ---- 3. ORDER INCREASES CAPACITY: higher d reconstructs the window better than a small d ----
    def full_delay_rmse(d):
        tr = _feed_sine(LegendreMemory(order=d, theta=theta, dt=dt), f)
        p = [_decode_state(tr[k], 1.0) for k in range(4 * n_win, len(tr))]
        t = [_sine(k * dt - theta, f) for k in range(4 * n_win, len(tr))]
        return _nrmse_corr(p, t)[0]
    err_low = full_delay_rmse(3)
    err_high = full_delay_rmse(order)

    # ---- 4. DETERMINISM: two fresh instances, same signal -> identical state trace ----
    drive = [_sine(k * dt, f) for k in range(500)]
    trace_a = LegendreMemory(order=order, theta=theta, dt=dt).run(drive)
    trace_b = LegendreMemory(order=order, theta=theta, dt=dt).run(drive)
    deterministic = trace_a == trace_b

    print("=" * 72)
    print("LEGENDRE MEMORY UNIT -- optimal linear window memory (delay network)")
    print("=" * 72)
    print(f"order d={order}  theta={theta}  dt={dt}  (dt/theta={dt / theta:.4f})  input sin f={f}")
    print(f"delay line  reconstruct(1.0) vs u(t-theta) : nrmse {nrmse_full:.4f}  corr {corr_full:.4f}"
          f"  (rmse {rmse_full:.4f})")
    print("whole-window readout  reconstruct(lag) vs u(t - lag*theta):")
    for lf in lags:
        rmse_l, nrmse_l, corr_l = per_lag[lf]
        print(f"    lag {lf:.2f} -> nrmse {nrmse_l:.4f}  corr {corr_l:.4f}  (rmse {rmse_l:.4f})")
    print(f"order vs capacity  full-delay rmse : d=3 {err_low:.4f}  ->  d={order} {err_high:.4f}"
          f"  ({err_low / err_high:.1f}x finer)")
    print(f"determinism  two fresh instances identical trace : {deterministic}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, ASCII-only ----
    assert corr_full > 0.99 and nrmse_full < 0.08, (
        f"delay line: reconstruct(1.0) must track u(t-theta) (corr {corr_full:.4f}, nrmse {nrmse_full:.4f})")
    worst_nrmse = max(per_lag[lf][1] for lf in lags)
    assert worst_nrmse < 0.10, (
        f"whole-window: every lag must reconstruct u(t - lag*theta) well (worst nrmse {worst_nrmse:.4f})")
    assert err_high < 0.5 * err_low, (
        f"order must increase capacity: err(d={order})={err_high:.4f} not << err(d=3)={err_low:.4f}")
    assert deterministic, "same order/theta/dt/signal must give an identical state trace"
    print(f"self-check OK: delay line reconstruct(1.0)->u(t-theta) corr {corr_full:.4f} (nrmse "
          f"{nrmse_full:.4f}); whole window worst-lag nrmse {worst_nrmse:.4f}; order raises capacity "
          f"(d=3 {err_low:.4f} -> d={order} {err_high:.4f}); deterministic")


if __name__ == "__main__":
    main()
