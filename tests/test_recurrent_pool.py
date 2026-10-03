"""G1(b) pools: hand-computed adapter fixtures (R-b0x's pool part), R-b0, R-b3, R-b5, the manifest.

Every hand fixture below is small (N <= 8) and its expected values were worked by hand from the
section 2.3 rules; each rule also has a deliberately wrong twin that must disagree with them.
"""

import copy
import json
import random
import statistics
import zlib

import numpy as np
import pytest

from snn_data_lake.reservoir import recurrent_pool as rp

# g1's draw order (section 2.4), pinned: crc32 of json.dumps(order, sort_keys, compact). The deleted
# ports' crc32s are section 3.1's record.
DRAW_ORDER_CRC32 = {"lsm": 1691104335}

# One configuration per candidate for the real-N checks.
CONFIG = {
    "lsm": {"k": 8, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 2},
}


def _crc(value):
    return zlib.crc32(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii"))


# ---- constants, grids, manifest ----------------------------------------------------------------


def test_constants_and_grids_as_written():
    assert (rp.N, rp.N_IN, rp.WINDOW, rp.S_SETTLE) == (256, 64, 256, 64)
    assert rp.FORMS == ("carried", "reset", "settled", "front_end")
    # section 2.9: the winner alone, keeping section 2.3's index 1 (g2-g4 were 2-4)
    assert rp.CANDIDATES == ("lsm",)
    assert rp.INDEX == {"lsm": 1} and rp.LABEL == {"lsm": "g1"} and set(rp.POOLS) == {"lsm"}
    assert [len(rp.grid(c)) for c in rp.CANDIDATES] == [288]
    lsm = rp.grid("lsm")
    assert lsm[0] == {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 1}
    assert lsm[1] == {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 2}  # last factor fastest
    assert lsm[-1] == {"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8}


def test_draw_order_is_the_seeded_sample_and_pinned():
    for candidate in rp.CANDIDATES:
        grid = rp.grid(candidate)
        order = rp.draw_order(candidate)
        assert order == random.Random(2026 + rp.INDEX[candidate]).sample(grid, len(grid))
        assert _crc(order) == DRAW_ORDER_CRC32[candidate]
    assert rp.draw_order_crc32() == DRAW_ORDER_CRC32


def test_screen_keeps_the_first_32_survivors_and_discloses_drops():
    order = rp.draw_order("lsm")
    cost = {_crc(cfg): (30.0 if i % 3 == 0 else 5.0) for i, cfg in enumerate(order)}
    result = rp.screen("lsm", lambda cfg: cost[_crc(cfg)])
    assert result["configs"] == [cfg for i, cfg in enumerate(order) if i % 3][:32]
    assert not result["short"]
    assert [d["cost_ms"] for d in result["dropped"]] == [30.0] * len(result["dropped"])
    assert [d["config"] for d in result["dropped"]] == [
        cfg for i, cfg in enumerate(order[: 32 + len(result["dropped"])]) if i % 3 == 0
    ]
    # the bound is "exceeds 25 ms": exactly 25 survives
    assert len(rp.screen("lsm", lambda cfg: 25.0)["configs"]) == 32
    starved = rp.screen("lsm", lambda cfg: 25.5)
    assert starved["configs"] == [] and starved["short"] and len(starved["dropped"]) == 288
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
    assert [rp.realization_seed(c, 200) for c in rp.CANDIDATES] == [6212]


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


def test_make_pool_canonicalizes_json_configs_and_refuses_the_rest():
    # a JSON round trip can turn the grid's int into a float: the grid's own entry comes back
    pool = rp.make_pool("lsm", {"k": 8.0, "w": 0.5, "lambda": 0.9, "g": 2.0, "T_in": 1.0}, 200)
    assert pool.config == {"k": 8, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 1}
    assert all(isinstance(pool.config[key], int) for key in ("k", "g", "T_in"))
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


def _oscillator(t_in, lam=0.0):
    # one g1 unit, no leak, a self-inhibition of 1e6 and a drive g u = 8 sum(x) >= 54 on every
    # fixture: it spikes, is silenced, spikes, ... from reset 0 (1, 0, 1, 0, ...)
    config = {"lambda": lam, "g": 8, "T_in": t_in}
    return rp.LsmPool([[0]], [[-1e6]], config, w_in=np.ones((1, 64)))


def test_startup_diagnostic_by_hand_on_a_period_two_unit():
    # steps 257-320 average 0.5 and every settle S in {0, 16, 64} reads an odd step, a spike
    result = rp.startup_diagnostic(_oscillator(1))
    assert result["transient"] == {0: 0.5, 16: 0.5, 64: 0.5} and result["flagged"]
    # averaging two steps sees the long-run mean at once
    result = rp.startup_diagnostic(_oscillator(2))
    assert result["transient"] == {0: 0.0, 16: 0.0, 64: 0.0} and not result["flagged"]
    # the wrong twin: a leak carries the silenced membrane (8u - 1e6) forward, so the unit stays
    # silent for many steps before it recovers and the period is no longer two
    result = rp.startup_diagnostic(_oscillator(1, lam=0.5))
    assert result["transient"] != {0: 0.5, 16: 0.5, 64: 0.5}


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
    config = {"k": 8, "w": 0.25, "lambda": 0.5, "g": 1, "T_in": 1}
    result = rp.cost("lsm", config, 200, percepts, settled=True)
    assert len(result["carried_ms"]) == 3 and result["cost_ms"] == statistics.median(
        result["carried_ms"]
    )
    assert len(result["init_s"]) == 3 and len(result["window_s"]) == 3 and "settled_ms" in result
    with pytest.raises(ValueError):
        rp.cost("lsm", config, 200, percepts[:256])


# ---- R-b0: parity with the sources, each with its red --------------------------------------------


@pytest.mark.parametrize("check", [rp.check_r_b0_lsm])
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


def test_r_b5_freeze_red_a_weight_written_after_the_boundary():
    # g1 learns nothing, so (iii)'s red is planted: one weight of the Carried form's pool moves by
    # one ulp right after its boundary fingerprint. A freeze check blind to that write fails here.
    result = rp.check_r_b5_freeze_red("lsm", CONFIG["lsm"])
    assert result["verdict"], result["numbers"]
    frozen = {"carried": True, "reset": True, "settled": True, "front_end": True}
    assert result["numbers"]["twin_frozen"] == frozen
    assert result["numbers"]["faulted_frozen"] == dict(frozen, carried=False)
    assert result["numbers"]["boundary_agree"]  # the write came after the fingerprint


# ---- window 0: Settled from the first percept (G1(c) 2.6, 2.10) ----------------------------------

W0_CLAUSES = ("i_first_phi", "ii_boundary", "iii_settled_invariance", "iv_carried_identity")


def test_window_zero_is_a_non_negative_count_and_every_other_refusal_stands():
    pool = _lsm("settled", window=0)
    assert (pool.window, pool.count) == (0, 0)
    for window in (-1, -256):
        with pytest.raises(ValueError, match="non-negative count"):
            _lsm(window=window)
    with pytest.raises(ValueError, match="T_in must be positive"):
        rp.LsmPool(LSM_PRE, LSM_W, dict(LSM_CFG, T_in=0), window=0)
    with pytest.raises(ValueError, match="s_settle non-negative"):
        rp.LsmPool(LSM_PRE, LSM_W, LSM_CFG, window=0, s_settle=-1)


@pytest.mark.parametrize("form", rp.FORMS)
def test_window_zero_takes_the_boundary_at_construction_in_every_form(form):
    adapted = rp.LsmPool(LSM_PRE, LSM_W, LSM_CFG, window=1)  # a window pool has none yet
    assert adapted.boundary is None and adapted.frozen() is None
    pool = _lsm(form, window=0)
    assert pool.count == 0 and pool.boundary == adapted.params_fingerprint()
    assert pool.frozen()
    if form == "front_end":  # the lesion exists from construction, and the boundary precedes it
        assert pool.w.tolist() == [[0.0, 0.0]] * 3 and pool.lesion_record["changed"] == ["w"]
        assert pool.lesion_record["params"] != pool.boundary
    else:
        assert pool.lesion_record is None and pool.w.tolist() == LSM_W


def test_window_zero_settled_runs_its_form_from_the_first_percept_by_hand():
    # Settled resets, takes one uncounted step, then two counted ones. From zero U1 gives
    # v = [0.5, 1, 0.75] (settle: only unit 1 spikes), then two steps on which all three units spike
    pool = _lsm("settled", window=0)
    assert pool.percept_drive(U1).tolist() == [1.0, 1.0, 1.0]
    # percept 2 resets again: settle v = [0, 0.5, 0], then 0.75 and 0.875, nothing spiking
    assert pool.percept_drive(U2).tolist() == [0.0, 0.0, 0.0]
    assert pool.v.tolist() == [0.0, 0.875, 0.0] and pool.s.tolist() == [False] * 3
    # at window 1 percept 1 is Carried (no settle step): the first percept is where they part
    assert _lsm("settled", window=1).percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    # the wrong twin: a window silently clamped to 1 after construction gives that Carried phi
    clamped = _lsm("settled", window=0)
    clamped.window = 1
    assert clamped.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]


def test_window_zero_front_end_reset_and_carried_first_percepts_by_hand():
    front = _lsm("front_end", window=0)  # zero synapses from the start, the leak still carries
    assert front.percept_drive(U1).tolist() == [0.0, 1.0, 0.5]
    reset = _lsm("reset", window=0)  # the carriers are zero at construction: Carried's first phi
    assert reset.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    assert reset.percept_drive(U2).tolist() == [0.0, 0.0, 0.0]
    assert reset.v.tolist() == [0.0, 0.75, 0.0]
    carried = _lsm("carried", window=0)
    assert carried.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    assert carried.percept_drive(U2).tolist() == [0.5, 0.5, 0.5]
    assert carried.v.tolist() == [0.0, 0.5, 0.5] and carried.boundary is not None


def test_the_independent_first_phi_is_apart_from_the_pool_and_sees_the_settle_count():
    config, seed = rp.R_B5_PERSISTENCE_WITNESS, rp.DEV_SEED
    x = rp.random_percepts(seed, 1)[0]
    first = rp._settled_first_phi(config, seed, x)
    settled = rp.make_pool("lsm", config, seed, "settled", window=0)
    assert np.array_equal(settled.percept(x), first)
    for settle in (63, 65):  # a derivation one settle step off is a different phi
        wrong = rp.make_pool("lsm", config, seed, "settled", window=0, s_settle=settle)
        assert not np.array_equal(wrong.percept(x), first)
    carried = rp.make_pool("lsm", config, seed, "carried", window=0)
    assert not np.array_equal(carried.percept(x), first)  # Carried@0 does not settle


def test_settled_at_zero_meets_settled_at_the_window_only_after_the_window():
    config, seed, window = rp.R_B5_PERSISTENCE_WITNESS, rp.DEV_SEED, 8
    zero = rp.make_pool("lsm", config, seed, "settled", window=0)
    late = rp.make_pool("lsm", config, seed, "settled", window=window)
    sequence = rp.random_percepts(seed, window + 8)
    pairs = [(zero.percept(x), late.percept(x)) for x in sequence]
    # inside the window one is Carried and the other Settled; after it both reset and settle
    assert any(not np.array_equal(a, b) for a, b in pairs[:window])
    assert all(np.array_equal(a, b) for a, b in pairs[window:])


def test_carried_at_zero_is_carried_at_the_window_phi_state_and_chain():
    config, seed, window = rp.R_B5_PERSISTENCE_WITNESS, rp.DEV_SEED, 12
    zero = rp.make_pool("lsm", config, seed, "carried", window=0)
    late = rp.make_pool("lsm", config, seed, "carried", window=window)
    for x in rp.random_percepts(seed, window + 12):
        assert np.array_equal(zero.percept(x), late.percept(x))
        assert zero.fingerprint() == late.fingerprint()
    assert zero.boundary == late.boundary  # the same parameters, fingerprinted at two moments


def test_window_zero_check_holds_its_four_clauses_on_its_twin():
    result = rp.check_window_zero()
    assert result["verdict"], result["numbers"]
    numbers = result["numbers"]
    assert numbers["clauses"] == dict.fromkeys(W0_CLAUSES, True)
    assert numbers["config"] == rp.R_B5_PERSISTENCE_WITNESS and numbers["fault"] is None
    assert numbers["percepts"] == 320 and numbers["inside_window_differing"] >= 1
    assert numbers["after_boundary_equal"] and numbers["boundary_agree"]
    assert all(numbers["boundary_at_count_0"].values()) and all(numbers["frozen"].values())
    assert numbers["front_end_changed"] == ["w"] and numbers["carried_phi_identical"]
    assert numbers["carried_chain"][0] == numbers["carried_chain"][1]
    assert numbers["carried_final"][0] == numbers["carried_final"][1]


@pytest.mark.parametrize(("fault", "clause"), sorted(rp.W0_FAULTS.items()))
def test_each_window_zero_fault_fails_its_own_clause_and_no_other(fault, clause):
    result = rp.check_window_zero(fault=fault)
    clauses = result["numbers"]["clauses"]
    assert not result["verdict"] and clauses[clause] is False, clauses
    assert all(ok for key, ok in clauses.items() if key != clause), clauses


def test_the_window_zero_faults_runner_names_each_fault_and_can_say_no():
    twin = rp.check_window_zero()
    result = rp.check_window_zero_faults(twin=twin)
    assert result["verdict"], result["numbers"]
    assert result["numbers"]["failed"] == {f: [c] for f, c in rp.W0_FAULTS.items()}
    assert result["numbers"]["twin"] == dict.fromkeys(W0_CLAUSES, True)
    # a twin that is not clean makes the runner refuse, however the faults fire
    assert not rp.check_window_zero_faults(twin=dict(twin, verdict=False))["verdict"]


def test_the_freeze_red_of_the_default_window_cannot_see_a_write_at_window_zero():
    # R-b5 (iii)'s planted write waits for a boundary that, at window 0, was already taken inside
    # the constructor, so it never lands and the red does not fire: window 0's own red writes a
    # weight after construction instead (W0_FAULTS["write_after"])
    result = rp.check_r_b5_freeze_red("lsm", CONFIG["lsm"], window=0)
    assert not result["verdict"] and result["numbers"]["fired"] == []
    assert rp.W0_FAULTS["write_after"] == "ii_boundary"


def test_window_zero_check_refuses_a_run_inside_the_window_and_an_unknown_fault():
    with pytest.raises(ValueError, match="more than 256"):
        rp.check_window_zero(percepts=256)
    with pytest.raises(ValueError, match="unknown window-0 fault"):
        rp.check_window_zero(fault="sideways")
    assert set(rp.W0_FAULTS.values()) <= set(W0_CLAUSES)


def test_window_zero_adds_no_form_and_no_candidate():
    assert rp.FORMS == ("carried", "reset", "settled", "front_end")
    assert rp.CANDIDATES == ("lsm",) and set(rp.POOLS) == {"lsm"} and rp.WINDOW == 256


# ---- G1(c) 2.7: the transported encoder and the swap control's fresh draw -----------------------


def _block(rows, seed=5):
    """A float32 weight block as the grower holds one (rows x 64), as the tap lists it."""
    return np.random.default_rng(seed).standard_normal((rows, rp.N_IN)).astype(np.float32).tolist()


def test_row_norms_are_sequential_sums_and_the_mean_is_their_mean():
    w = rp.w_in(7)
    norms = rp.row_norms(w)
    assert norms.shape == (256,)
    assert np.array_equal(norms, np.sqrt(np.cumsum(w * w, axis=1)[:, -1]))
    assert abs(rp.mean_row_norm(w) - float(np.mean(norms))) < 1e-12
    assert abs(rp.mean_row_norm(w) - 1.0) < 0.1  # N(0, 1) / 8 over 64 inputs: a row norm near 1


def test_the_transported_encoder_replaces_the_leading_rows_and_matches_the_random_mean_norm():
    w_random = rp.w_in(7)
    block = _block(3)
    out = rp.transported_encoder(w_random, block)
    target = rp.mean_row_norm(w_random)
    assert out.shape == (256, 64) and out.dtype == np.float64
    assert np.array_equal(out[3:], w_random[3:])  # rows L..255 are kept, bit for bit
    for i in range(3):
        row = np.asarray(block[i], dtype=np.float64)  # float32 values cast up, exactly
        assert abs(np.linalg.norm(out[i]) - target) < 1e-12
        assert abs(rp.row_norms(out[i : i + 1])[0] - target) < 1e-12
        # the direction is the transported row's: the raw row times a positive scalar
        np.testing.assert_allclose(
            out[i] / np.linalg.norm(out[i]), row / np.linalg.norm(row), rtol=0, atol=1e-15
        )
        assert not np.array_equal(out[i], w_random[i])
    assert np.array_equal(w_random, rp.w_in(7))  # the random draw is never written


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (np.zeros((0, 64)), "no unit"),
        (np.ones((3, 63)), "width 63"),
        (np.ones((257, 64)), "more than the encoder"),
        (np.zeros((2, 64)), "cannot be rescaled"),
        (np.full((2, 64), np.nan), "cannot be rescaled"),
        (np.ones(64), "no unit|width"),  # a bare row is not a block
    ],
)
def test_the_transported_encoder_refuses_what_it_cannot_carry(block, match):
    with pytest.raises(ValueError, match=match):
        rp.transported_encoder(rp.w_in(7), block)


def test_the_transported_encoder_refuses_a_random_draw_that_is_not_the_encoder():
    with pytest.raises(ValueError, match="encoder shape"):
        rp.transported_encoder(np.ones((255, 64)), _block(2))
    # 256 units fill every row: the largest block it admits
    out = rp.transported_encoder(rp.w_in(7), _block(256))
    assert abs(rp.mean_row_norm(out) - rp.mean_row_norm(rp.w_in(7))) < 1e-12


def test_the_fresh_draw_has_its_own_tag_and_every_row_at_the_random_mean_norm():
    assert rp.SWAP_TAG == int.from_bytes(b"SWAP", "big") == 0x53574150
    w_random = rp.w_in(9)
    fresh = rp.fresh_encoder(9, w_random)
    raw = np.random.default_rng(np.random.SeedSequence([9, rp.SWAP_TAG])).standard_normal((256, 64))
    raw = raw / 8
    target = rp.mean_row_norm(w_random)
    scale = target / np.sqrt(np.cumsum(raw * raw, axis=1)[:, -1])
    np.testing.assert_allclose(fresh, raw * scale[:, None], rtol=1e-14, atol=0)
    assert np.max(np.abs(rp.row_norms(fresh) - target)) < 1e-12
    assert not np.array_equal(raw, w_random)  # not the construction draw of the same seed
    assert np.array_equal(fresh, rp.fresh_encoder(9, w_random))  # deterministic
    assert not np.array_equal(fresh, rp.fresh_encoder(10, rp.w_in(10)))  # one draw per seed
    with pytest.raises(ValueError, match="encoder shape"):
        rp.fresh_encoder(9, np.ones((255, 64)))


def test_replace_encoder_swaps_the_matrix_and_no_state():
    pool = rp.make_pool("lsm", CONFIG["lsm"], 3, "carried", window=4)
    percepts = rp.random_percepts(3, 6)
    for percept in percepts[:3]:
        pool.percept(percept)
    before = (pool.fingerprint(), pool.params_fingerprint(), pool.count)
    new = rp.fresh_encoder(3, pool.w_in)
    pool.replace_encoder(new)
    # the encoder is no part of the pool's state: its fingerprints, count and boundary stay
    assert (pool.fingerprint(), pool.params_fingerprint(), pool.count) == before
    assert pool.w_in is not new and np.array_equal(pool.w_in, new)
    twin = copy.deepcopy(pool)
    assert np.array_equal(pool.percept(percepts[3]), twin.percept_drive(rp.drive(new, percepts[3])))
    for bad in (np.ones((255, 64)), np.ones((256, 63)), np.full((256, 64), np.nan)):
        with pytest.raises(ValueError, match="encoder"):
            pool.replace_encoder(bad)


def test_the_transported_encoder_check_holds_and_each_red_fires():
    result = rp.check_transported_encoder()
    assert result["verdict"], result["numbers"]
    assert result["red"] == {
        "L == 0": True,
        "width 63": True,
        "L 257": True,
        "zero-norm row": True,
        "an unscaled copy": True,
    }
    assert result["numbers"]["norm_error"] < 1e-12
