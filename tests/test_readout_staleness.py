"""Brain-free reds R1-R5 and R8 of instruments/readout_staleness (L1b-STALE, spec 2.6 with A0b/A0e).

numpy fixture only: `FakeModel` / `FakeEngine` honour the NGSNN contract the instrument reads
(_rates, _solve_beta, decide, W, b, beta, prior_class_). Every relative identity (R2, R8) is
ASSERTED on every host (A0e M1). Tests that need torch (the real NGSNN, the sequential boundaries)
import it lazily and skip where absent; they run in the integrator's gate.
"""

import numpy as np
import pytest

from snn_data_lake.instruments import readout_staleness as rs


class FakeModel:
    def __init__(self, W, b, beta=None, prior=0):
        self.W, self.b, self.beta, self.prior_class_ = W, b, beta, prior
        self.last_abstentions_ = self.last_unanswerable_ = 0

    def _rates(self, X, W, b):
        return np.round(np.clip(np.asarray(X) @ W.T + b, 0, 1) * 16) / 16

    def _solve_beta(self, S, Y):
        return np.linalg.solve(S.T @ S + 0.1 * np.eye(S.shape[1]), S.T @ Y)

    def decide(self, S, beta=None):
        pred = (S @ (self.beta if beta is None else beta)).argmax(axis=1)
        silent = np.abs(S).sum(axis=1) == 0
        self.last_abstentions_ = int(silent.sum())
        self.last_unanswerable_ = int(silent.sum()) + 1
        return np.where(silent, self.prior_class_, pred)


def split(y, h):
    n = len(y)
    return list(range(n - h)), list(range(n - h, n))


def stream(n=120, d=6, m=3, seed=0):
    rng = np.random.default_rng(seed)
    protos = rng.uniform(0, 1, (m, d))
    y = rng.integers(0, m, n)
    return (protos[y] + rng.normal(0, 0.15, (n, d))).astype(np.float32), y


def model_for(X, y, L=8, seed=1, solve_rows=None):
    rng = np.random.default_rng(seed)
    W, b = rng.normal(0, 0.8, (L, X.shape[1])), rng.normal(0, 0.1, L)
    m = FakeModel(W, b)
    rows = slice(None) if solve_rows is None else solve_rows
    m.beta = m._solve_beta(m._rates(X[rows], W, b), rs._onehot(y[rows], int(y.max()) + 1))
    return m


def fixture():
    X, y = stream()
    tr, va = split(y, 40)
    return X, y, tr, va


# ---- R1: the shadow reads staleness --------------------------------------------------------------
def test_r1_green_unmoved_w_reads_zero_gap():
    X, y, tr, va = fixture()
    m = model_for(X, y, solve_rows=tr)
    r = rs.shadow_reading(m, m.beta, X[tr], y[tr], X[va], y[va])
    assert abs(r["gap_all"]) < 1e-6 and abs(r["gap_ans"]) < 1e-6


def test_r1_red_planted_rotation_reads_positive_gap():
    X, y, tr, va = fixture()
    m = model_for(X, y, solve_rows=tr)
    reader = m.beta.copy()
    m.W = rs.rotate_pairs(m.W, 30)
    r = rs.shadow_reading(m, reader, X[tr], y[tr], X[va], y[va])
    assert r["gap_all"] > 0


def test_r1_second_green_pruned_solve_calibrates_to_zero():
    X, y, tr, va = fixture()
    m = model_for(
        X, y, solve_rows=tr[::2]
    )  # reader solved on prune's own internal split, not the refit's
    sh = m._solve_beta(m._rates(X[tr], m.W, m.b), rs._onehot(y[tr], 3))
    assert np.abs(sh - m.beta).max() > 1e-3  # the pruned solve really differs from the shadow
    calib = rs.calibration_reading(m, X[tr], y[tr], X[va], y[va], pruned=True)
    raw = rs.shadow_reading(m, m.beta, X[tr], y[tr], X[va], y[va])["gap_all"]
    assert calib["pruned"] and abs(rs.calibrated(raw, calib["gap_all"])) < 1e-6


def test_unseen_by_reader_is_masked_not_hidden():
    X, y, tr, va = fixture()
    m = model_for(X, y, solve_rows=tr)
    narrow = m.beta[:, :2]  # a reader that cannot output class 2
    r = rs.shadow_reading(m, narrow, X[tr], y[tr], X[va], y[va])
    assert r["unseen_by_reader"] == pytest.approx(float(np.mean(y[va] >= 2)))
    assert "gap_all" in r and "gap_ans" in r


# ---- R2: the shadow never feeds back (through the real instance wrapper) -------------------------
class FakeEngine:
    """closed gate after the first model exists; open solve = new model on the train rows,
    beta solved there."""

    def __init__(self):
        self.ticks, self.trajectory, self._buf_x, self._buf_y = 0, [], [], []
        self._model = self._batch_beta = None
        self._pruned = 0
        self.rng = np.random.default_rng(5)
        self.W0 = self.rng.normal(0, 0.8, (8, 6))
        self.b0 = self.rng.normal(0, 0.1, 8)

    def push(self, X, y):
        self._buf_x.extend(X.tolist())
        self._buf_y.extend(int(v) for v in y)
        self.ticks += len(y)

    def _refit(self):
        n = len(self._buf_y)
        h = min(60, n // 3)
        if h < 3 or len(set(self._buf_y)) < 2:
            return
        X, y = np.asarray(self._buf_x, np.float32), np.asarray(self._buf_y)
        tr, va = split(y, h)
        m = self._model
        if m is not None:  # closed gate
            reader = rs.reader_for(m, self._batch_beta)
            acc = float(np.mean(m.decide(m._rates(X[va], m.W, m.b), reader) == y[va]))
            self.trajectory.append((self.ticks, 8, acc))
            return
        m = FakeModel(self.W0.copy(), self.b0.copy())
        m.beta = m._solve_beta(m._rates(X[tr], m.W, m.b), rs._onehot(y[tr], int(y[tr].max()) + 1))
        acc = float(np.mean(m.decide(m._rates(X[va], m.W, m.b)) == y[va]))
        self.trajectory.append((self.ticks, 8, acc))
        self._model, self._batch_beta = m, m.beta.copy()


def run_engine(shadow=True, leak=False, as_input=lambda a: np.asarray(a, np.float32)):
    X, y = stream(n=240)
    e = FakeEngine()
    log = rs.instrument_engine(e, as_input=as_input, split_fn=split, shadow=shadow)
    if leak:
        real = rs.shadow_reading

        def leaky(*a, **k):
            r = real(*a, **k)
            e._batch_beta = r["beta_shadow"]  # the shadow written into the gate's reader
            return r

        rs.shadow_reading, saved = leaky, real
    hashes = []
    try:
        for i in range(0, 240, 20):
            e.push(X[i : i + 20], y[i : i + 20])
            e._refit()
            hashes.append(
                rs.substrate_hash(e._model, e._batch_beta, e.trajectory) if e._model else None
            )
    finally:
        if leak:
            rs.shadow_reading = saved
    return e, log, hashes


def test_r2_green_shadow_on_equals_off_every_row():
    _, log_on, h_on = run_engine(shadow=True)
    _, _, h_off = run_engine(shadow=False)
    assert h_on == h_off and any(h_on)
    assert log_on["faults"] == 0 and any(r["gate"] == "closed" for r in log_on["rows"])
    assert any("gap_ans" in r for r in log_on["rows"]) and any(
        "calibration" in r for r in log_on["rows"]
    )


def test_r2_red_leaking_variant_changes_a_hash():
    _, _, h_leak = run_engine(shadow=True, leak=True)
    _, _, h_off = run_engine(shadow=False)
    assert h_leak != h_off


def test_instrument_fault_is_counted_and_brain_unharmed():
    def boom(a):
        raise RuntimeError("planted")

    _, log, h_bad = run_engine(shadow=True, as_input=boom)
    _, _, h_off = run_engine(shadow=False)
    assert log["faults"] > 0 and h_bad == h_off


# ---- R3: the operators move what they claim ------------------------------------------------------
def test_r3_operators_preserve_norm_and_move():
    rng = np.random.default_rng(0)
    W, s, xn = rng.normal(size=(5, 6)), np.full(5, 0.5), np.eye(6)[1]
    ok, moved, inert = rs.operator_check(W, rs.oja_step(W, s, xn))
    assert ok and moved > 0 and not inert
    ok, moved, inert = rs.operator_check(W, rs.RandStep(3)(W, s))
    assert ok and moved > 0 and not inert


def test_r3_red_mu_zero_inert_and_norm_break_fails():
    rng = np.random.default_rng(0)
    W, s, xn = rng.normal(size=(5, 6)), np.full(5, 0.5), np.eye(6)[1]
    ok, moved, inert = rs.operator_check(W, rs.oja_step(W, s, xn, mu=0.0))
    assert inert and moved < 1e-9
    assert not rs.operator_check(W, rs.oja_step(W, s, xn) * 1.01)[0]


def test_rand_is_seeded_by_crc32_not_global_state():
    W, s = np.random.default_rng(0).normal(size=(4, 6)), np.ones(4)
    np.random.seed(1)
    a = rs.RandStep(7)(W, s)
    np.random.seed(2)
    assert np.array_equal(a, rs.RandStep(7)(W, s))
    assert not np.array_equal(a, rs.RandStep(8)(W, s))


# ---- R4: the rate-drift instrument ---------------------------------------------------------------
def test_r4_frozen_reads_exactly_zero_rotation_monotone_permutation_positive():
    X, y, tr, va = fixture()
    m = model_for(X, y, L=12)
    P = X[:60]
    S0 = m._rates(P, m.W, m.b)
    idx = np.arange(12)
    assert rs.rate_drift(S0, m._rates(P, m.W, m.b), idx, idx) == 0.0
    d = [
        rs.rate_drift(S0, m._rates(P, rs.rotate_pairs(m.W, a), m.b), idx, idx) for a in (5, 15, 45)
    ]
    assert 0 < d[0] < d[1] < d[2]
    assert rs.rate_drift(S0, S0, idx, np.random.default_rng(0).permutation(12)) > 0


def test_survivor_pairs_follow_leading_rows_and_keep():
    p, c = rs.survivor_pairs(3, [0, 2, 5, 4], 4)
    assert list(p) == [0, 2] and list(c) == [0, 1]
    p, c = rs.survivor_pairs(4, None, 6)
    assert list(p) == list(c) == [0, 1, 2, 3]


# ---- R5: the null split --------------------------------------------------------------------------
def test_r5_null_split_fixture_and_swapped_projector():
    rng = np.random.default_rng(0)
    beta = rng.normal(size=(6, 3))  # L=6 > m=3
    P, null_dim = rs.readout_projector(beta)
    assert null_dim == 3
    Q = np.eye(6) - P
    dS = rng.normal(size=(40, 6))
    in_null, in_col = dS @ Q, dS @ P
    assert rs.visible_fraction(in_null, P) < 0.05 and rs.visible_fraction(in_col, P) > 0.95
    assert (
        rs.visible_fraction(in_null, Q) > 0.95 and rs.visible_fraction(in_col, Q) < 0.05
    )  # swapped: reversed
    assert rs.readout_projector(rng.normal(size=(3, 6)))[1] == 0  # narrow layer: split NOT READ


# ---- R8: the A/A ---------------------------------------------------------------------------------
def test_r8_frozen_vs_frozen_b_identical_and_one_ulp_caught():
    a, la, ha = run_engine()
    b, lb, hb = run_engine()
    assert ha == hb and a.trajectory == b.trajectory
    assert [r.get("gap_all") for r in la["rows"]] == [r.get("gap_all") for r in lb["rows"]]
    b._model.b[0] = np.nextafter(b._model.b[0], np.inf)
    assert rs.substrate_hash(b._model, b._batch_beta, b.trajectory) != ha[-1]


# ---- gate-run only (torch) -----------------------------------------------------------------------
def test_look_reading_uses_seq_boundaries_gate_only():
    pytest.importorskip("torch")
    ds = [0.2, 0.3, 0.25, 0.4, 0.35, 0.3, 0.28, 0.33]
    r = rs.look_reading(ds, look=0, looks=(8, 16, 24, 32))
    assert r["crossed"] and r["fpr"].startswith("exact")


def test_shadow_on_real_ngsnn_gap_zero_gate_only():
    torch = pytest.importorskip("torch")
    from snn_data_lake.growth.ng_snn import NGSNN

    X, y, tr, va = fixture()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    m = NGSNN(device="cpu", seed=0).fit(Xt[tr], yt[tr])
    r = rs.shadow_reading(m, m.beta, Xt[tr], y[tr], Xt[va], y[va])
    assert abs(r["gap_all"]) < 1e-6


# ---- round 3 (#5728 M1): every component of the substrate hash is pinned on its own -----------
def _hash_parts():
    X, y, tr, va = fixture()
    m = model_for(X, y, solve_rows=tr)
    m.last_abstentions_, m.last_unanswerable_ = 3, 4
    return m, m.beta.copy(), [(20, 8, 0.5), (40, 8, 0.75)]


def _ulp(a):
    a = np.array(a, dtype=float)
    a.flat[0] = np.nextafter(a.flat[0], np.inf)
    return a


@pytest.mark.parametrize("part", ["W", "b", "beta", "batch_beta", "trajectory", "abstain", "unans"])
def test_m1_substrate_hash_sees_each_component_alone(part):
    m, bb, traj = _hash_parts()
    base = rs.substrate_hash(m, bb, traj)
    assert base == rs.substrate_hash(m, bb, list(traj))
    if part == "batch_beta":
        bb = _ulp(bb)
    elif part == "trajectory":
        traj = [traj[0], (40, 8, float(np.nextafter(0.75, 1.0)))]
    elif part == "abstain":
        m.last_abstentions_ += 1
    elif part == "unans":
        m.last_unanswerable_ += 1
    else:
        setattr(m, part, _ulp(getattr(m, part)))
    assert rs.substrate_hash(m, bb, traj) != base


def test_m1_shadow_restores_both_abstention_counters():
    X, y, tr, va = fixture()
    m = model_for(X, y, solve_rows=tr)
    m.last_abstentions_, m.last_unanswerable_ = 7, 9  # decide() overwrites both
    h0 = rs.substrate_hash(m, m.beta)
    rs.shadow_reading(m, m.beta, X[tr], y[tr], X[va], y[va])
    assert (m.last_abstentions_, m.last_unanswerable_) == (7, 9)
    assert rs.substrate_hash(m, m.beta) == h0


def test_m1_r2_second_leak_that_cannot_change_any_accuracy_still_moves_the_hash():
    # the shadow written into `_batch_beta` as a one-ulp copy of the reader:
    # same argmax, same accuracy
    real = rs.shadow_reading

    def leaky(*a, **k):
        r = real(*a, **k)
        e._batch_beta = np.nextafter(e._batch_beta, np.inf)
        return r

    X, y = stream(n=240)
    e = FakeEngine()
    rs.instrument_engine(e, as_input=lambda a: np.asarray(a, np.float32), split_fn=split)
    rs.shadow_reading = leaky
    try:
        for i in range(0, 240, 20):
            e.push(X[i : i + 20], y[i : i + 20])
            e._refit()
    finally:
        rs.shadow_reading = real
    _, _, h_off = run_engine(shadow=False)
    assert rs.substrate_hash(e._model, e._batch_beta, e.trajectory) != h_off[-1]
    assert e.trajectory == run_engine(shadow=False)[0].trajectory  # accuracies unmoved: hash-only
