"""G1(b) pools: hand-computed adapter fixtures (R-b0x's pool part), R-b0, R-b3, R-b5, the manifest.

Every hand fixture below is small (N <= 8) and its expected values were worked by hand from the
section 2.3 rules; each rule also has a deliberately wrong twin that must disagree with them.
"""

import json
import random
import statistics
import zlib

import numpy as np
import pytest

from snn_data_lake.cortex import recurrent_pool as rp

# The four draw orders (section 2.4), pinned: crc32 of json.dumps(order, sort_keys, compact).
DRAW_ORDER_CRC32 = {"lsm": 1691104335, "esn": 176457545, "sorn": 489683113, "rbn": 959998414}

# One configuration per candidate for the real-N checks.
CONFIG = {
    "lsm": {"k": 8, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 2},
    "esn": {"rho": 0.9, "g": 2, "T_in": 2},
    "sorn": {"plasticity": (0.004, 0.01), "H_IP": 0.1, "g": 2, "T_in": 1},
    "rbn": {"K": 2, "p": 0.5, "f": 0.25, "mode": "xor", "tau": 0, "T_in": 2},
}


def _crc(value):
    return zlib.crc32(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii"))


# ---- constants, grids, manifest ----------------------------------------------------------------


def test_constants_and_grids_as_written():
    assert (rp.N, rp.N_IN, rp.WINDOW, rp.S_SETTLE) == (256, 64, 256, 64)
    assert rp.FORMS == ("carried", "reset", "settled", "front_end")
    assert rp.CANDIDATES == ("lsm", "esn", "sorn", "rbn")
    assert [rp.INDEX[c] for c in rp.CANDIDATES] == [1, 2, 3, 4]
    assert [len(rp.grid(c)) for c in rp.CANDIDATES] == [288, 48, 240, 432]
    lsm = rp.grid("lsm")
    assert lsm[0] == {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 1}
    assert lsm[1] == {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 2}  # last factor fastest
    assert lsm[-1] == {"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8}
    sorn = rp.grid("sorn")
    assert [sorn[i]["plasticity"] for i in (0, 48, 96, 144, 192)] == [
        "off",
        (0.001, 0.001),
        (0.001, 0.01),
        (0.004, 0.001),
        (0.004, 0.01),
    ]
    assert rp.grid("rbn")[0] == {"K": 1, "p": 0.5, "f": 0.1, "mode": "xor", "tau": -0.1, "T_in": 1}


def test_draw_order_is_the_seeded_sample_and_pinned():
    for candidate in rp.CANDIDATES:
        grid = rp.grid(candidate)
        order = rp.draw_order(candidate)
        assert order == random.Random(2026 + rp.INDEX[candidate]).sample(grid, len(grid))
        assert _crc(order) == DRAW_ORDER_CRC32[candidate]
    assert rp.draw_order_crc32() == DRAW_ORDER_CRC32


def test_screen_keeps_the_first_32_survivors_and_discloses_drops():
    order = rp.draw_order("esn")
    cost = {_crc(cfg): (30.0 if i % 3 == 0 else 5.0) for i, cfg in enumerate(order)}
    result = rp.screen("esn", lambda cfg: cost[_crc(cfg)])
    assert result["configs"] == [cfg for i, cfg in enumerate(order) if i % 3][:32]
    assert not result["short"]
    assert [d["cost_ms"] for d in result["dropped"]] == [30.0] * len(result["dropped"])
    assert [d["config"] for d in result["dropped"]] == [
        cfg for i, cfg in enumerate(order[: 32 + len(result["dropped"])]) if i % 3 == 0
    ]
    # the bound is "exceeds 25 ms": exactly 25 survives
    assert len(rp.screen("esn", lambda cfg: 25.0)["configs"]) == 32
    starved = rp.screen("esn", lambda cfg: 25.5)
    assert starved["configs"] == [] and starved["short"] and len(starved["dropped"]) == 48
    screens = {c: rp.screen(c, lambda cfg: 1.0) for c in rp.CANDIDATES}
    manifest = rp.manifest(screens)
    assert manifest["crc32"] == _crc({c: screens[c]["configs"] for c in rp.CANDIDATES})
    assert rp.manifest(screens) == manifest and manifest["short"] == []


# ---- encoder -----------------------------------------------------------------------------------


def test_encoder_recipe_and_sequential_drive():
    w = rp.w_in(200)
    assert w.shape == (256, 64)
    assert np.array_equal(w, np.random.default_rng(200 * 31 + 7).standard_normal((256, 64)) / 8)
    small = np.array([[1.0, 2.0], [0.5, -1.0], [0.25, 0.0]])
    assert rp.drive(small, [0.5, 0.25]).tolist() == [1.0, 0.0, 0.125]
    x = np.random.default_rng(1).random(64)
    loop = []
    for i in range(256):
        acc = 0.0
        for j in range(64):
            acc += w[i, j] * x[j]
        loop.append(acc)
    assert rp.drive(w, x).tolist() == loop
    # the declared order is left to right over the inputs: ((0 + 1) + 1e16) - 1e16 == 0.0, while
    # the reversed order (a deliberately wrong rule) recovers the 1.0
    assert rp.drive([[1.0, 1e16, -1e16]], [1.0, 1.0, 1.0]).tolist() == [0.0]
    assert rp.drive([[-1e16, 1e16, 1.0]], [1.0, 1.0, 1.0]).tolist() == [1.0]
    assert [rp.realization_seed(c, 200) for c in rp.CANDIDATES] == [6212, 6213, 6214, 6215]


# ---- g1 LSM, hand-computed (N 3, k 2) ------------------------------------------------------------
# unit 0 <- (1, 0.5), (2, 0.25); unit 1 <- (0, 1.0), (0, -0.5); unit 2 <- (1, 0.5), (0, 0.5)
LSM_PRE = [[1, 2], [0, 0], [1, 0]]
LSM_W = [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]]
LSM_CFG = {"lambda": 0.5, "g": 2, "T_in": 2}
U1 = [0.25, 0.5, 0.375]  # g u = [0.5, 1.0, 0.75]
U2 = [0.0, 0.25, 0.0]  # g u = [0.0, 0.5, 0.0]
U3 = [0.0, 0.3125, 0.0]  # g u = [0.0, 0.625, 0.0]


def _lsm(form="carried", window=4, cls=rp.LsmPool):
    return cls(LSM_PRE, LSM_W, LSM_CFG, form, window=window, s_settle=1)


def test_lsm_vector_drive_t_in_average_and_carry_across_percepts():
    pool = _lsm()
    # step 1: v = [0.5, 1, 0.75] -> s = [0,1,0]; step 2: v = [1.25, 1, 1.625] -> all spike
    assert pool.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    assert pool.v.tolist() == [0.0, 0.0, 0.0] and pool.s.tolist() == [True, True, True]
    # carried: step 1 rec = [0.75, 0.5, 1] -> v = [0.75, 1, 1] -> s = [0,1,1], v = [0.75, 0, 0];
    # step 2 rec = [0.75, 0, 0.5] -> v = [1.125, 0.5, 0.5] -> s = [1,0,0]
    assert pool.percept_drive(U2).tolist() == [0.5, 0.5, 0.5]
    assert pool.v.tolist() == [0.0, 0.5, 0.5] and pool.s.tolist() == [True, False, False]
    assert pool.count == 2 and pool.boundary is None


@pytest.mark.parametrize(
    ("form", "phi", "v", "s"),
    [
        ("carried", [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [True, False, False]),
        ("reset", [0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [False, False, False]),
        ("settled", [0.0, 0.0, 0.0], [0.0, 0.875, 0.0], [False, False, False]),
        ("front_end", [0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [False, False, False]),
    ],
)
def test_lsm_each_form_after_a_one_percept_window(form, phi, v, s):
    pool = _lsm(form, window=1)
    assert pool.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]  # the window is Carried in every form
    assert pool.percept_drive(U2).tolist() == phi
    assert pool.v.tolist() == v and pool.s.tolist() == s


def test_window_boundary_runs_carried_inside_and_the_form_after():
    pool = _lsm("reset", window=2)
    pool.percept_drive(U1)
    assert pool.boundary is None
    assert pool.percept_drive(U2).tolist() == [0.5, 0.5, 0.5]  # percept 2 is inside: Carried
    assert pool.count == 2 and pool.boundary == pool.params_fingerprint()
    assert pool.percept_drive(U2).tolist() == [0.0, 0.0, 0.0]  # percept 3: Reset
    assert pool.frozen()


def test_lsm_front_end_is_zero_synapse_but_stateful():
    front, reset = _lsm("front_end", window=1), _lsm("reset", window=1)
    for pool in (front, reset):
        pool.percept_drive(U1)
        pool.percept_drive(U2)
    assert front.boundary == reset.boundary  # one adapted state at the boundary
    assert front.w.tolist() == [[0.0, 0.0]] * 3 and front.lesion_record["changed"] == ["w"]
    assert front.lesion_record["params"] != front.boundary
    # the leak carries v1 = 0.75 in: 0.375 + 0.625 = 1.0 spikes; Reset starts from 0 and never does
    assert front.percept_drive(U3).tolist() == [0.0, 0.5, 0.0]
    assert reset.percept_drive(U3).tolist() == [0.0, 0.0, 0.0]
    assert front.frozen() and reset.frozen()


class _SoftResetLsm(rp.LsmPool):  # wrong rule: subtract the threshold instead of resetting to 0
    def step(self, u):
        v = (self.lam * self.v + self.g * np.asarray(u)) + self._rec()
        self.s = v >= 1.0
        self.v = np.where(self.s, v - 1.0, v)
        return self.s.astype(np.float64)


class _MeanOverTInPlusOneLsm(rp.LsmPool):  # wrong rule: T_in averaging over T_in + 1
    def percept_drive(self, u):
        return super().percept_drive(u) * self.t_in / (self.t_in + 1)


class _SettleWithoutResetLsm(rp.LsmPool):  # wrong rule: Settled that forgets the reset
    def reset_carriers(self):
        if self.count <= self.window:
            super().reset_carriers()


class _MemorylessFrontEndLsm(rp.LsmPool):  # wrong rule: the lesion also removes the leak
    def lesion(self):
        super().lesion()
        self.lam = 0.0


def test_lsm_fixtures_catch_each_deliberately_wrong_rule():
    soft = _lsm(cls=_SoftResetLsm)
    soft.percept_drive(U1)
    assert soft.v.tolist() != [0.0, 0.0, 0.0]
    assert _lsm(cls=_MeanOverTInPlusOneLsm).percept_drive(U1).tolist() != [0.5, 1.0, 0.5]
    settled = _lsm("settled", window=1, cls=_SettleWithoutResetLsm)
    settled.percept_drive(U1)
    assert settled.percept_drive(U2).tolist() != [0.0, 0.0, 0.0]
    memoryless = _lsm("front_end", window=1, cls=_MemorylessFrontEndLsm)
    memoryless.percept_drive(U1)
    memoryless.percept_drive(U2)
    assert memoryless.percept_drive(U3).tolist() != [0.0, 0.5, 0.0]


# ---- g2 ESN, hand-computed (N 3) -----------------------------------------------------------------
ESN_W = [[0.0, 0.5, 0.25], [0.25, 0.0, 0.5], [0.5, 0.125, 0.0]]  # [pre, post]


def test_esn_soft_reset_step_by_hand():
    pool = rp.EsnPool(ESN_W, {"g": 1, "T_in": 2})
    phi = pool.percept_drive([0.7, 0.5, 0.2])
    # step 1: v = u -> z = [1, 0, 0]; step 2: cur = W[0, :] + u, v = 0.9 v - 0.6 z + cur
    expected_v = [
        0.9 * 0.7 - 0.6 * 1.0 + (0.0 + 0.7),
        0.9 * 0.5 - 0.6 * 0.0 + (0.5 + 0.5),
        0.9 * 0.2 - 0.6 * 0.0 + (0.25 + 0.2),
    ]
    assert phi.tolist() == [1.0, 0.5, 0.5]
    assert pool.v.tolist() == expected_v and pool.z.tolist() == [1.0, 1.0, 1.0]
    hard = rp.EsnPool(ESN_W, {"g": 1, "T_in": 2})
    hard.v_th = 0.8  # a deliberately wrong threshold must disagree
    assert hard.percept_drive([0.7, 0.5, 0.2]).tolist() != [1.0, 0.5, 0.5]


def test_esn_front_end_keeps_leak_and_soft_reset():
    pool = rp.EsnPool(ESN_W, {"g": 1, "T_in": 1}, "front_end", window=1)
    pool.percept_drive([0.7, 0.5, 0.2])  # v = u, z = [1, 0, 0]
    assert pool.w_rec.tolist() == [[0.0] * 3] * 3 and pool.lesion_record["changed"] == ["w_rec"]
    pool.percept_drive([0.7, 0.5, 0.2])
    assert pool.v.tolist() == [0.9 * 0.7 - 0.6 * 1.0 + 0.7, 0.9 * 0.5 + 0.5, 0.9 * 0.2 + 0.2]


# ---- g3 SORN, hand-computed (3 E, 1 I) -----------------------------------------------------------
SORN_NET = {
    "ee_pre": [[1], [0, 2], [1]],
    "ee_w": [[1.0], [0.5, 0.5], [1.0]],
    "ei": [[(0, 1.0)], [], [(0, 1.0)]],
    "ie": [[(0, 0.5), (1, 0.5)]],
    "t_e": [0.25, 0.5, 0.75],
    "t_i": [0.75],
}
SORN_U = [0.5, 0.25, 0.5, 0.0]  # the I unit's slot is never read


def _sorn(config, form="carried", window=8, cls=rp.SornPool):
    return cls(rp._sorn_arrays(SORN_NET, 3, 1), ([0, 1, 0], [1]), config, form, window=window)


def test_sorn_step_by_hand_and_phi_order_e_then_i():
    pool = _sorn({"plasticity": "off", "H_IP": 0.1, "g": 1, "T_in": 4})
    # E: [1,0,0] -> [1,1,0] -> [1,1,1] -> [1,1,0];  I: 0 -> 0 -> 1 -> 1
    assert pool.percept_drive(SORN_U).tolist() == [1.0, 0.75, 0.25, 0.5]
    assert pool.x_e.tolist() == [True, True, False] and pool.x_i.tolist() == [True]


def test_sorn_plasticity_by_hand_then_frozen_after_the_window():
    pool = _sorn({"plasticity": (0.25, 0.5), "H_IP": 0.5, "g": 1, "T_in": 1}, window=2)
    pool.percept_drive(SORN_U)  # e [1,0,0]; no STDP from a silent past; IP: t_e += 0.5 (x - 0.5)
    assert pool.t_e.tolist() == [0.5, 0.25, 0.5]
    pool.percept_drive(
        SORN_U
    )  # e [1,1,1]; row 0: 1 - 0.25 -> 0.75 -> [1.0]; row 1: 0.75, 0.5 / 1.25
    assert [w for _, w in pool.ee_rows()] == [[1.0], [0.6, 0.4], [1.0]]
    assert pool.t_e.tolist() == [0.75, 0.5, 0.75] and pool.stdp_events == 2
    pool.percept_drive(SORN_U)  # after the window: no plasticity
    assert pool.t_e.tolist() == [0.75, 0.5, 0.75] and pool.frozen()


class _AntiStdpSorn(rp.SornPool):  # wrong rule: the STDP sign flipped
    def _stdp(self, old, new):
        super()._stdp(new, old)


def test_sorn_fixture_catches_a_flipped_stdp_sign():
    pool = _sorn({"plasticity": (0.25, 0.5), "H_IP": 0.5, "g": 1, "T_in": 1}, cls=_AntiStdpSorn)
    pool.percept_drive(SORN_U)
    pool.percept_drive(SORN_U)
    assert [w for _, w in pool.ee_rows()] != [[1.0], [0.6, 0.4], [1.0]]


def test_sorn_reset_state_and_front_end_lesion():
    pool = _sorn({"plasticity": "off", "H_IP": 0.1, "g": 1, "T_in": 1}, "front_end", window=1)
    pool.percept_drive(SORN_U)
    assert pool.lesion_record["changed"] == ["e_w", "i_w"]
    pool.percept_drive(SORN_U)  # x_E = [g u - T_E >= 0] = [1, 0, 0]; x_I = [-T_I >= 0] = [0]
    assert pool.x_e.tolist() == [True, False, False] and pool.x_i.tolist() == [False]
    reset = _sorn({"plasticity": "off", "H_IP": 0.1, "g": 1, "T_in": 1}, "reset", window=1)
    reset.percept_drive(SORN_U)
    reset.reset_carriers()
    assert reset.x_e.tolist() == [False, True, False] and reset.x_i.tolist() == [True]


def test_row_sums_reproduce_the_sources_builtin_sum():
    rng = np.random.default_rng(3)
    rows = [
        (rng.random(int(n)) * 10.0 ** rng.integers(-9, 2, int(n))).tolist()
        for n in rng.integers(1, 40, size=300)
    ]
    values = np.zeros((len(rows), max(map(len, rows))))
    for i, row in enumerate(rows):
        values[i, : len(row)] = row
    lengths = np.array([len(row) for row in rows])
    assert rp._builtin_sum_rows(values, lengths).tolist() == [sum(row) for row in rows]
    # red: a plain sequential cumsum is NOT what the source's sum() computes
    plain = [float(np.cumsum(values[i, : len(row)])[-1]) for i, row in enumerate(rows)]
    assert plain != [sum(row) for row in rows]


# ---- g4 RBN, hand-computed (N 4, K 1) ------------------------------------------------------------
# node 0 copies node 1, node 1 negates node 2, node 2 copies node 3, node 3 is constant 1
RBN_INPUTS = [[1], [2], [3], [0]]
RBN_TABLES = [[0, 1], [1, 0], [0, 1], [1, 1]]
RBN_U = [0.5, -1.0, -0.5, 1.0]  # input nodes 0 and 2: bits [0.5 > 0, -0.5 > 0] = [1, 0]


def _rbn(mode, form="carried", window=4, cls=rp.RbnPool):
    config = {"mode": mode, "tau": 0, "T_in": 2}
    return cls(RBN_INPUTS, RBN_TABLES, [0, 2], [0, 1, 0, 1], config, form, window=window)


def test_rbn_xor_and_clamp_apply_after_the_synchronous_update():
    xor = _rbn("xor")
    # step 1: update [0,1,0,1] -> [1,1,1,1], xor -> [0,1,1,1]; step 2: -> [1,0,1,1] -> [0,0,1,1]
    assert xor.percept_drive(RBN_U).tolist() == [0.0, 0.5, 1.0, 1.0]
    assert xor.state.tolist() == [0, 0, 1, 1]
    clamp = _rbn("clamp")
    # step 1: -> [1,1,1,1], clamp -> [1,1,0,1]; step 2: -> [1,1,1,1] -> [1,1,0,1]
    assert clamp.percept_drive(RBN_U).tolist() == [1.0, 1.0, 0.0, 1.0]


class _InputFirstRbn(rp.RbnPool):  # wrong timing: the bits land on the OLD state before the update
    def step(self, u):
        bits = (np.asarray(u)[self.in_nodes] > self.tau).astype(np.int64)
        state = self.state.copy()
        state[self.in_nodes] = state[self.in_nodes] ^ bits
        self.state = state
        self.state = self._next()
        return self.state.astype(np.float64)


def test_rbn_fixture_catches_wrong_timing_and_swapped_modes():
    assert _rbn("xor", cls=_InputFirstRbn).percept_drive(RBN_U).tolist() != [0.0, 0.5, 1.0, 1.0]
    assert _rbn("clamp").percept_drive(RBN_U).tolist() != [0.0, 0.5, 1.0, 1.0]


def test_rbn_front_end_holds_non_input_nodes():
    xor = _rbn("xor", "front_end", window=1)
    xor.percept_drive(RBN_U)  # -> [0,0,1,1]; then only the input bits act
    assert xor.percept_drive(RBN_U).tolist() == [0.5, 0.0, 1.0, 1.0]
    assert xor.lesion_record["changed"] == ["hold"]
    clamp = _rbn("clamp", "front_end", window=1)
    clamp.percept_drive(RBN_U)
    assert clamp.percept_drive(RBN_U).tolist() == [1.0, 1.0, 0.0, 1.0]


def test_rbn_realization_input_nodes_and_reset_state_follow_their_seeds():
    from snn_data_lake.reasoning.random_boolean_network import build_rbn

    seed = rp.realization_seed("rbn", 200)
    pool = rp.make_pool(
        "rbn", {"K": 2, "p": 0.7, "f": 0.1, "mode": "xor", "tau": 0, "T_in": 1}, 200
    )
    net = build_rbn(256, 2, seed, 0.7)
    assert pool.inputs.tolist() == net.inputs and pool.tables.tolist() == net.tables
    assert pool.in_nodes.tolist() == random.Random(seed + 1000).sample(range(256), 256)[:26]
    bits = random.Random(seed + 2000)
    assert pool.state.tolist() == [bits.randrange(2) for _ in range(256)]
    assert [round(f * 256) for f in (0.1, 0.25, 0.5)] == [26, 64, 128]


# ---- realizations at the real N ------------------------------------------------------------------


def test_lsm_realization_consumes_the_sources_w_in_draws_first():
    seed = rp.realization_seed("lsm", 200)
    pool = rp.make_pool("lsm", {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 1}, 200)
    rng = random.Random(seed)
    for _ in range(256):
        rng.uniform(-1, 1)
    rows = [[(rng.randrange(256), rng.uniform(-0.25, 0.25)) for _ in range(8)] for _ in range(256)]
    assert pool.pre.tolist() == [[j for j, _ in row] for row in rows]
    assert pool.w.tolist() == [[x for _, x in row] for row in rows]


def test_esn_realization_is_g1s_draw_densified_and_rescaled():
    pool = rp.make_pool("esn", {"rho": 1.2, "g": 1, "T_in": 1}, 200)
    _, pre, w = rp._lsm_draw(rp.realization_seed("esn", 200), 256, 8, 0.5)
    dense = np.zeros((256, 256))
    for i in range(256):
        for p in range(8):
            dense[pre[i, p], i] += w[i, p]
    np.fill_diagonal(dense, 0.0)
    assert np.array_equal(pool.w_rec != 0.0, dense != 0.0)
    assert abs(float(np.max(np.abs(np.linalg.eigvals(pool.w_rec)))) - 1.2) < 1e-9


def test_sorn_realization_sizes_reset_state_and_initial_zeros():
    seed = rp.realization_seed("sorn", 200)
    pool = rp.make_pool("sorn", {"plasticity": "off", "H_IP": 0.2, "g": 1, "T_in": 1}, 200)
    assert (pool.n_e, pool.n_i, pool.n) == (213, 43, 256)
    assert not pool.x_e.any() and not pool.x_i.any()  # all zeros before the window
    bits = random.Random(seed + 2000)
    drawn = [bits.random() < 0.2 for _ in range(256)]
    pool.reset_carriers()
    assert pool.x_e.tolist() == drawn[:213] and pool.x_i.tolist() == drawn[213:]


def test_make_pool_canonicalizes_json_configs_and_refuses_the_rest():
    pool = rp.make_pool("sorn", {"plasticity": [0.004, 0.01], "H_IP": 0.1, "g": 2, "T_in": 1}, 200)
    assert pool.config["plasticity"] == (0.004, 0.01)
    rbn = rp.make_pool(
        "rbn", {"K": 1, "p": 0.5, "f": 0.1, "mode": "xor", "tau": 0.0, "T_in": 1}, 200
    )
    assert rbn.config["tau"] == 0 and isinstance(rbn.config["tau"], int)
    with pytest.raises(ValueError):
        rp.make_pool("lsm", {"k": 9, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 1}, 200)
    with pytest.raises(ValueError):
        rp.make_pool("lsm", CONFIG["lsm"], 200, "sideways")


# ---- the startup diagnostic ----------------------------------------------------------------------


def test_fixture_percepts_follow_the_recipe():
    rng = np.random.default_rng(777)
    fixtures = rp.fixture_percepts()
    assert len(fixtures) == 16
    for x in fixtures:
        v = rng.random(64) ** 4
        norm = float(np.sqrt(sum(float(a) * float(a) for a in v)))
        assert np.allclose(x, np.clip(v * 1.73 / norm, 0.0, 1.0), rtol=0, atol=1e-15)


def test_startup_diagnostic_by_hand_on_a_period_two_node():
    # one node, its own negation: 1, 0, 1, 0, ... from reset 0. Steps 257-320 average 0.5.
    oscillator = {"mode": "xor", "tau": 0, "T_in": 1}
    pool = rp.RbnPool([[0]], [[1, 0]], [], [0], oscillator, w_in=np.zeros((1, 64)))
    result = rp.startup_diagnostic(pool)
    assert result["transient"] == {0: 0.5, 16: 0.5, 64: 0.5} and result["flagged"]
    oscillator["T_in"] = 2  # averaging two steps sees the long-run mean at once
    pool = rp.RbnPool([[0]], [[1, 0]], [], [0], oscillator, w_in=np.zeros((1, 64)))
    result = rp.startup_diagnostic(pool)
    assert result["transient"] == {0: 0.0, 16: 0.0, 64: 0.0} and not result["flagged"]


def test_startup_diagnostic_on_one_configuration_leaves_the_pool_untouched():
    pool = rp.make_pool("lsm", CONFIG["lsm"], 200)
    before = pool.fingerprint()
    result = rp.startup_diagnostic(pool)
    assert pool.fingerprint() == before
    assert set(result["transient"]) == {0, 16, 64}
    assert all(0.0 <= t <= 1.0 for t in result["transient"].values())
    assert result["flagged"] == (result["transient"][64] > 0.05)


# ---- cost ----------------------------------------------------------------------------------------


def test_cost_is_the_median_carried_time_after_the_window():
    percepts = rp.cycled_fixture(300)
    config = {"K": 1, "p": 0.5, "f": 0.1, "mode": "xor", "tau": 0, "T_in": 1}
    result = rp.cost("rbn", config, 200, percepts, settled=True)
    assert len(result["carried_ms"]) == 3 and result["cost_ms"] == statistics.median(
        result["carried_ms"]
    )
    assert len(result["init_s"]) == 3 and len(result["window_s"]) == 3 and "settled_ms" in result
    with pytest.raises(ValueError):
        rp.cost("rbn", config, 200, percepts[:256])


# ---- R-b0: parity with the sources, each with its red --------------------------------------------


@pytest.mark.parametrize(
    "check", [rp.check_r_b0_lsm, rp.check_r_b0_esn, rp.check_r_b0_sorn, rp.check_r_b0_rbn]
)
def test_r_b0_parity_with_its_red(check):
    result = check()
    assert result["verdict"], result["numbers"]
    assert result["red"]


# ---- R-b3: determinism by construction at the real N, and its reds -------------------------------


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_r_b3_thread_invariance_at_the_real_n(candidate):
    result = rp.check_r_b3(candidate, CONFIG[candidate])
    assert result["verdict"], result["numbers"]["equal"]
    assert {(s["blas"], s["torch"]) for s in result["numbers"]["settings"]} == {
        (1, 1),
        (1, 4),
        (24, 1),
        (24, 4),
    }
    t_in = CONFIG[candidate]["T_in"]  # every step is observed, Settled's uncounted 64 included
    assert result["numbers"]["runs"]["1/1/settled"]["steps"] == 256 * t_in + 256 * (64 + t_in)


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_r_b3_red_fires_on_its_declared_witness(candidate):
    result = rp.check_r_b3_red(candidate)
    assert result["verdict"], result["numbers"]
    assert result["red"]


# ---- R-b4's pool-noise red -----------------------------------------------------------------------


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_pool_noise_changes_the_continuation_state_and_phi(candidate):
    pool = rp.make_pool(candidate, CONFIG[candidate], 200)
    before = pool.fingerprint()
    pool.perturb()
    assert pool.fingerprint() != before
    assert rp.pool_noise_red(candidate, CONFIG[candidate], window=16)["verdict"]


# ---- R-b5: the forms do what they claim ----------------------------------------------------------


@pytest.mark.parametrize("form", ["reset", "settled"])
@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_r_b5_erasure_with_its_skip_a_carrier_red(candidate, form):
    result = rp.check_r_b5_erasure(candidate, CONFIG[candidate], form)
    assert result["verdict"] and result["numbers"]["prefixes_diverged"], result["numbers"]
    red = rp.check_r_b5_erasure_red(candidate, form)
    assert red["verdict"], red["numbers"]


def test_r_b5_persistence_on_its_witness_and_the_reset_every_percept_red():
    result = rp.check_r_b5_persistence()
    assert result["verdict"] and result["numbers"]["config"] == rp.R_B5_PERSISTENCE_WITNESS
    assert not rp.check_r_b5_persistence(carried_form="reset")["verdict"]


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_r_b5_freeze_across_all_four_forms(candidate):
    result = rp.check_r_b5_freeze(candidate, CONFIG[candidate])
    assert result["verdict"], result["numbers"]


def test_r_b5_freeze_red_sorn_plasticity_left_on():
    assert not rp.check_r_b5_freeze("sorn", CONFIG["sorn"], plastic_forever=True)["verdict"]
