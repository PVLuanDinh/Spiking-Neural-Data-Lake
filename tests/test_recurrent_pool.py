"""G1(b) pools: hand-computed adapter fixtures (R-b0x's pool part), R-b0, R-b3, R-b5, the manifest.

Every hand fixture below is small (N <= 8) and its expected values were worked by hand from the
section 2.3 rules; each rule also has a deliberately wrong twin that must disagree with them.
G1(c) 2.9 deleted the float64 port and the host switch: every fixture runs on the one pool
implementation, KernelLsmPool (the hand values are lattice values at the binding frac_bits), and
the constructed wrong hosts below are this file's own reds of that pool's arithmetic.
"""

import copy
import inspect
import json
import random
import statistics
import zlib

import numpy as np
import pytest

from snn_data_lake.gates import determinism_gate
from snn_data_lake.reservoir import kernel_tick
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


def _lsm(form="carried", window=4, cls=rp.KernelLsmPool):
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


@pytest.mark.parametrize("cls", [rp.KernelLsmPool])
@pytest.mark.parametrize(
    ("form", "phi", "v", "s"),
    [
        ("carried", [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [True, False, False]),
        ("reset", [0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [False, False, False]),
        ("settled", [0.0, 0.0, 0.0], [0.0, 0.875, 0.0], [False, False, False]),
        ("front_end", [0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [False, False, False]),
    ],
)
def test_lsm_each_form_after_a_one_percept_window(form, phi, v, s, cls):
    # the hand values are lattice values at lambda 1/2, which the KernelTick pool holds exactly
    pool = _lsm(form, window=1, cls=cls)
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


class _ResetBelowZeroLsm(rp.KernelLsmPool):  # wrong rule: a spike resets to -0.25, not to 0
    V_RESET = -0.25


class _MeanOverTInPlusOneLsm(rp.KernelLsmPool):  # wrong rule: T_in averaging over T_in + 1
    def percept_drive(self, u):
        return super().percept_drive(u) * self.t_in / (self.t_in + 1)


class _SettleWithoutResetLsm(rp.KernelLsmPool):  # wrong rule: Settled that forgets the reset
    def reset_carriers(self):
        if self.count <= self.window:
            super().reset_carriers()


class _MemorylessFrontEndLsm(rp.KernelLsmPool):  # wrong rule: the lesion also removes the leak
    def lesion(self):
        super().lesion()
        # KernelTick refuses a zero decay, so "no leak" is a decay far below one float32 ulp of v
        self._kt.decay = self.lam = 1e-30


def test_lsm_fixtures_catch_each_deliberately_wrong_rule():
    below = _lsm(cls=_ResetBelowZeroLsm)
    below.percept_drive(U1)
    assert below.v.tolist() != [0.0, 0.0, 0.0]
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


def _oscillator(t_in, lam=1e-30):
    # one g1 unit, no leak (KernelTick refuses a zero decay: 1e-30 carries less than 1e-23 of a
    # silenced membrane), a self-inhibition of 1e6 and a drive g u = 8 sum(x) >= 54 on every
    # fixture: it spikes, is silenced, spikes, ... from reset 0 (1, 0, 1, 0, ...)
    config = {"lambda": lam, "g": 8, "T_in": t_in}
    return rp.KernelLsmPool([[0]], [[-1e6]], config, w_in=np.ones((1, 64)))


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


# ---- R-b0: the draws against the source, with its red --------------------------------------------


def test_r_b0_the_draws_equal_the_sources_and_a_shifted_draw_does_not():
    # the parity clause (the float64 port's rates and SynOps) went with the port (G1(c) 2.9)
    result = rp.check_r_b0_lsm()
    assert result["verdict"] and result["numbers"]["structure"], result["numbers"]
    assert result["red"] and (result["numbers"]["units"], result["numbers"]["fan_in"]) == (100, 8)
    assert not rp.check_r_b0_lsm(red_shift=0)["verdict"]  # a red at no shift cannot fire


# ---- R-b3: determinism by construction at the real N, and its reds -------------------------------


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_r_b3_partition_and_thread_invariance_at_the_real_n(candidate):
    result = rp.check_r_b3(candidate, CONFIG[candidate])
    assert result["verdict"], result["numbers"]["equal"]
    assert result["numbers"]["settings"] == [{"P": p, "threads": t} for p, t in rp.R_B3_HOSTED]
    t_in = CONFIG[candidate]["T_in"]  # every step is observed, Settled's uncounted 64 included
    assert result["numbers"]["runs"]["P1/T1/settled"]["steps"] == 256 * t_in + 256 * (64 + t_in)


# ---- R-b4's pool-noise red -----------------------------------------------------------------------


@pytest.mark.parametrize("candidate", rp.CANDIDATES)
def test_pool_noise_changes_the_continuation_state_and_phi(candidate):
    pool = rp.make_pool(candidate, CONFIG[candidate], 200)
    before = pool.fingerprint()
    pool.perturb()
    assert pool.fingerprint() != before  # the red itself: test_the_hosted_pool_noise_red_...


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
        rp.KernelLsmPool(LSM_PRE, LSM_W, dict(LSM_CFG, T_in=0), window=0)
    with pytest.raises(ValueError, match="s_settle non-negative"):
        rp.KernelLsmPool(LSM_PRE, LSM_W, LSM_CFG, window=0, s_settle=-1)


@pytest.mark.parametrize("form", rp.FORMS)
def test_window_zero_takes_the_boundary_at_construction_in_every_form(form):
    adapted = rp.KernelLsmPool(LSM_PRE, LSM_W, LSM_CFG, window=1)  # a window pool has none yet
    assert adapted.boundary is None and adapted.frozen() is None
    pool = _lsm(form, window=0)
    assert pool.count == 0 and pool.boundary == adapted.params_fingerprint()
    assert pool.frozen()
    if form == "front_end":  # the lesion exists from construction, and the boundary precedes it
        assert pool.w.tolist() == [[0.0, 0.0]] * 3 and pool.lesion_record["changed"] == ["w"]
        assert pool.lesion_record["params"] != pool.boundary
    else:
        assert pool.lesion_record is None and pool.w.tolist() == LSM_W


@pytest.mark.parametrize("cls", [rp.KernelLsmPool])
def test_window_zero_settled_runs_its_form_from_the_first_percept_by_hand(cls):
    # Settled resets, takes one uncounted step, then two counted ones. From zero U1 gives
    # v = [0.5, 1, 0.75] (settle: only unit 1 spikes), then two steps on which all three units spike
    pool = _lsm("settled", window=0, cls=cls)
    assert pool.percept_drive(U1).tolist() == [1.0, 1.0, 1.0]
    # percept 2 resets again: settle v = [0, 0.5, 0], then 0.75 and 0.875, nothing spiking
    assert pool.percept_drive(U2).tolist() == [0.0, 0.0, 0.0]
    assert pool.v.tolist() == [0.0, 0.875, 0.0] and pool.s.tolist() == [False] * 3
    # at window 1 percept 1 is Carried (no settle step): the first percept is where they part
    assert _lsm("settled", window=1, cls=cls).percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    # the wrong twin: a window silently clamped to 1 after construction gives that Carried phi
    clamped = _lsm("settled", window=0, cls=cls)
    clamped.window = 1
    assert clamped.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]


@pytest.mark.parametrize("cls", [rp.KernelLsmPool])
def test_window_zero_front_end_reset_and_carried_first_percepts_by_hand(cls):
    front = _lsm("front_end", window=0, cls=cls)  # zero synapses from the start, the leak carries
    assert front.percept_drive(U1).tolist() == [0.0, 1.0, 0.5]
    reset = _lsm("reset", window=0, cls=cls)  # carriers are zero at construction: Carried's phi
    assert reset.percept_drive(U1).tolist() == [0.5, 1.0, 0.5]
    assert reset.percept_drive(U2).tolist() == [0.0, 0.0, 0.0]
    assert reset.v.tolist() == [0.0, 0.75, 0.0]
    carried = _lsm("carried", window=0, cls=cls)
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
        rp.fresh_encoder(9, np.ones((256, 63)))
    # the draw's row count, never a fixed 256, sets the fresh encoder's (RULING item 3)
    assert rp.fresh_encoder(9, np.ones((255, 64))).shape == (255, 64)


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


# ---- the KernelTick-hosted g1, the one pool implementation (G1(c) 2.9, P4a) ---------------------

TUNED = {"k": 8, "w": 0.25, "lambda": 0.9, "g": 4, "T_in": 1}  # the campaign's configuration
LSB = 2.0**-12
CHAIN_PRE = [[0], [0], [1]]  # unit 1 hears unit 0 and unit 2 hears unit 1, each at 1.0
CHAIN_W = [[0.0], [1.0], [1.0]]  # unit 0 hears itself at 0.0, so every unit has one entry
CHAIN_CFG = {"lambda": 0.5, "g": 1, "T_in": 1}


# The constructed wrong hosts: each differs from the pool in exactly one policy attribute or hook,
# so a test of the pool's arithmetic has a red that must fail it (they were G1(c) 2.9's parity
# fixtures' wrong hosts; the fixtures went with the port)
class _ResetsBelowZero(rp.KernelLsmPool):  # wrong host: the reset value is -0.25, not 0
    V_RESET = -0.25


class _Refracts(rp.KernelLsmPool):  # wrong host: one refractory tick after a spike
    REFRAC_TICKS = 1


class _DelaysTwo(rp.KernelLsmPool):  # wrong host: every edge takes two ticks
    DELAY = 2


class _StrictThreshold(
    rp.KernelLsmPool
):  # wrong host: `>` for `>=`, the threshold a float32 step up
    THR = float(np.nextafter(np.float32(rp.LSM_THRESH), np.float32(2.0)))


class _Truncates(rp.KernelLsmPool):  # wrong host: weights and drive truncated, not rounded
    def _edges(self):
        pre, post, weight, delay = super()._edges()
        scale = 2.0**self.frac_bits
        return pre, post, np.trunc(weight * scale) / scale, delay

    def _quantize_drive(self, drive_vector):
        return np.trunc(drive_vector * 2**self.frac_bits).astype(np.int64)


class _KeepsCharge(rp.KernelLsmPool):  # wrong host: a lesion leaves the charge already in flight
    def _zero_in_flight(self):
        pass


def _host(form="carried", window=4, **options):
    return rp.KernelLsmPool(LSM_PRE, LSM_W, LSM_CFG, form, window=window, s_settle=1, **options)


def _spikes(pool, drives):
    """Each step's activity on `drives`, as lists."""
    return [pool.step(np.asarray(u, dtype=np.float64)).tolist() for u in drives]


def test_the_host_policy_names_what_the_host_does_that_the_port_does_not():
    assert determinism_gate.binding_frac_bits() == 12
    assert _host().host_policy() == {
        "host": "kernel",
        "frac_bits": 12,
        "p_parts": 1,
        "decay_f32": 0.5,
        "v_reset": 0.0,
        "refrac_ticks": 0,
        "delay": 1,
        "thr": 1.0,
    }
    leaky = rp.KernelLsmPool(LSM_PRE, LSM_W, {**LSM_CFG, "lambda": 0.9}, frac_bits=16, p_parts=2)
    policy = leaky.host_policy()
    assert policy["decay_f32"] == float(np.float32(0.9)) and policy["decay_f32"] != 0.9
    assert (policy["frac_bits"], policy["p_parts"]) == (16, 2)
    # each constructed wrong host differs from the host in exactly the one policy key it names
    base = _host().host_policy()
    for wrong, key in (
        (_ResetsBelowZero, "v_reset"),
        (_Refracts, "refrac_ticks"),
        (_DelaysTwo, "delay"),
        (_StrictThreshold, "thr"),
    ):
        policy = wrong(LSM_PRE, LSM_W, LSM_CFG).host_policy()
        assert [k for k in base if base[k] != policy[k]] == [key]
    assert _StrictThreshold(LSM_PRE, LSM_W, LSM_CFG).host_policy()["thr"] > 1.0


def test_the_lineage_policy_is_the_policy_of_a_host_built_at_the_lineage_options(
    monkeypatch, tmp_path
):
    for config in (TUNED, CONFIG["lsm"], rp.R_B3_WITNESS["lsm"]):
        hosted = rp.make_pool("lsm", config, 100)
        assert rp.KernelLsmPool.lineage_policy(config) == hosted.host_policy()
    # the policy is the class's own: a wrong host's lineage policy is not the faithful host's
    wrong = _Refracts.lineage_policy(LSM_CFG)
    base = rp.KernelLsmPool.lineage_policy(LSM_CFG)
    assert [key for key in base if base[key] != wrong[key]] == ["refrac_ticks"]
    # and the construction is shared, not restated: the instance policy is policy_of at its options
    pool = rp.KernelLsmPool(LSM_PRE, LSM_W, LSM_CFG, frac_bits=16, p_parts=2)
    assert pool.host_policy() == rp.KernelLsmPool.policy_of(16, 2, LSM_CFG["lambda"])
    monkeypatch.setattr(determinism_gate, "BASELINE", tmp_path / "no_frac_bits.json")
    with pytest.raises(ValueError, match="no binding frac_bits"):
        rp.KernelLsmPool.lineage_policy(LSM_CFG)


def test_the_host_refuses_when_the_lineage_records_are_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(determinism_gate, "BASELINE", tmp_path / "no_frac_bits.json")
    with pytest.raises(ValueError, match="no binding frac_bits"):
        _host()
    # a given frac_bits needs no record, but the partition count still does
    monkeypatch.setattr(kernel_tick, "PARTITION_BASELINE", tmp_path / "no_partition.json")
    with pytest.raises(ValueError, match="partition baseline"):
        _host(frac_bits=12)
    assert _host(frac_bits=12, p_parts=1).host_policy()["frac_bits"] == 12  # both given: no read
    for bad in ({"frac_bits": 0, "p_parts": 1}, {"frac_bits": 12, "p_parts": 0}):
        with pytest.raises(ValueError, match="positive counts"):
            _host(**bad)
    # KernelTick refuses a zero leak, so lambda 0 cannot be hosted
    with pytest.raises(ValueError, match="decay"):
        rp.KernelLsmPool(LSM_PRE, LSM_W, {**LSM_CFG, "lambda": 0.0}, frac_bits=12, p_parts=1)


def _subclasses(cls):
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub} | _subclasses(sub)
    return found


def test_one_reachable_pool_implementation():
    """G1(c) 2.9 (P2: a merge must reduce): the migration gate's loser, the float64 port, and the
    host switch are deleted: make_pool reaches ONE pool implementation and nothing else steps one.
    The one other class that defines `step` is R-b3's float-accumulation twin (_FloatTwin, with its
    stored-order control _ForwardTwin inheriting it): a constructed red that only
    check_r_b3_hosted_red builds, never in POOLS and never returned by make_pool."""
    assert set(rp.POOLS.values()) == {rp.KernelLsmPool}
    assert not hasattr(rp, "HOSTS") and "host" not in inspect.signature(rp.make_pool).parameters
    assert "step" not in vars(rp.LsmPool) and "step" not in vars(rp._Pool)
    assert "_rec" not in vars(rp.LsmPool) and "reset_carriers" not in vars(rp._Pool)
    stepping = {cls for cls in _subclasses(rp._Pool) if "step" in vars(cls)}
    assert stepping == {rp.KernelLsmPool, rp._FloatTwin}, stepping
    assert rp._FloatTwin not in rp.POOLS.values() and rp._ForwardTwin not in rp.POOLS.values()
    for form in rp.FORMS:
        assert type(rp.make_pool("lsm", CONFIG["lsm"], 200, form)) is rp.KernelLsmPool
    # the host options reach the one pool; a host name is no option
    hosted = rp.make_pool("lsm", CONFIG["lsm"], 200, frac_bits=16, p_parts=2)
    assert hosted.host_policy()["frac_bits"] == 16
    assert rp.make_pool("lsm", CONFIG["lsm"], 200).host_policy()["frac_bits"] == 12
    for host in ("kernel", "port"):
        with pytest.raises(TypeError, match="host"):
            rp.make_pool("lsm", CONFIG["lsm"], 200, host=host)
    # LsmPool keeps the network (draws, leak, gain, lesion rule, parameters, kick) and no arithmetic
    with pytest.raises(AttributeError):
        rp.LsmPool(LSM_PRE, LSM_W, LSM_CFG)


@pytest.mark.parametrize("seed", [100, 101, 200])
def test_the_pool_is_the_draws_network_and_its_live_table_the_stored_lists(seed):
    for config in (TUNED, CONFIG["lsm"], rp.R_B3_WITNESS["lsm"]):
        drawn = rp.realization_seed("lsm", seed)
        _, pre, weights = rp._lsm_draw(drawn, rp.N, config["k"], config["w"])
        host = rp.make_pool("lsm", config, seed, "carried")
        assert pre.tobytes() == host.pre.tobytes() and weights.tobytes() == host.w.tobytes()
        assert (
            np.array_equal(rp.w_in(seed), host.w_in) and rp.canonical("lsm", config) == host.config
        )
        # the live table is the stored lists row by row: post is the unit, pre the entry's source,
        # q the weight at the lsb with half to even, and a repeated (pre, post) pair is kept
        posts = np.repeat(np.arange(host.n), host.pre.shape[1])
        pres = host.pre.ravel()
        qs = np.rint(host.w.ravel() * 4096).astype(np.int64)
        order = np.lexsort((qs, pres, posts))
        edges = host._live_edges()
        assert np.array_equal(edges["edge_post"], posts[order])
        assert np.array_equal(edges["edge_pre"], pres[order])
        assert np.array_equal(edges["edge_q"], qs[order])
    assert rp.realization_seed("lsm", 100) == 100 * 31 + 11 + 1  # never a second candidate


def test_one_kernel_tick_is_one_pool_step_in_every_form():
    for form, ticks in (("carried", 2), ("reset", 2), ("settled", 3), ("front_end", 2)):
        pool = _host(form, window=0)  # T_in 2, one settle step (Settled's only)
        pool.percept_drive(U1)
        assert pool._kt.tick_index == ticks, form
    pool = _host("carried", window=4)
    for _ in range(3):
        pool.percept_drive(U1)
    assert pool._kt.tick_index == 3 * LSM_CFG["T_in"]


def test_the_activity_is_the_fired_mask_as_float64_and_s_is_the_last_ticks_mask():
    pool = _host()
    activity = pool.step(U1)
    assert activity.dtype == np.float64 and activity.tolist() == [0.0, 1.0, 0.0]
    assert pool.s.dtype == bool and pool.s.tolist() == [False, True, False]
    assert pool.step(U1).tolist() == [1.0, 1.0, 1.0] and pool.s.tolist() == [True] * 3


def test_the_carriers_are_a_float32_view_a_fired_mask_and_summed_integer_accumulators():
    assert rp.KernelLsmPool.CARRIERS == ("v", "s", "acc")
    pool = _host(p_parts=8)  # 3 units over 8 partitions: five partitions own no unit
    pool.percept_drive(U1)
    carriers = pool.carriers()
    assert set(carriers) == {"v", "s", "acc"}
    assert (carriers["v"].dtype, carriers["s"].dtype, carriers["acc"].dtype) == (
        np.float32,
        np.dtype(bool),
        np.int64,
    )
    assert carriers["acc"].tolist() == [3072, 2048, 4096] and not np.shares_memory(
        carriers["v"], pool.v
    )
    assert pool.v is pool._kt.state.arrays["v"]  # the carrier is a view of the host's membranes
    # the setters write through, which the skip-a-carrier fault (setattr) relies on. The v setter
    # writes INTO the original block: a replacement array would break KernelState's 128-byte
    # alignment contract (kernel_state.py), which `identity` and the pointer modulus catch
    held = pool.v
    pool.v = np.array([0.25, 0.5, 0.75])
    assert pool.v is held and pool._kt.state.arrays["v"] is held
    assert held.ctypes.data % 128 == 0 and held.dtype == np.float32
    pool.s = [True, False, True]
    pool.acc = [5, -3, 7]
    assert pool.v.tolist() == [0.25, 0.5, 0.75] and pool.s.tolist() == [True, False, True]
    assert pool.acc.tolist() == [5, -3, 7]
    for part, members in enumerate(pool._kt._members):  # each unit's charge sits in its owner
        owned = np.zeros(3, dtype=bool)
        owned[members] = True
        assert not pool._kt._acc[part][~owned].any(), part


@pytest.mark.parametrize("form", ["carried", "settled"])
def test_the_hosted_state_and_fingerprints_are_independent_of_p_and_threads(form):
    sequence = rp.random_percepts(100, 10)
    runs = {}
    for p_parts, threads in ((1, 1), (2, 1), (8, 1), (8, 4)):
        pool = rp.make_pool("lsm", TUNED, 100, form, window=4, p_parts=p_parts, threads=threads)
        phis, chain = [], []
        for x in sequence:
            phis.append(pool.percept(x).tobytes())
            chain.append(pool.fingerprint())
        runs[(p_parts, threads)] = (phis, chain, pool.params_fingerprint(), pool.boundary)
    first = runs[(1, 1)]
    assert len(set(chain[-1] for _, chain, _, _ in runs.values())) == 1
    assert all(run == first for run in runs.values())


def test_the_fingerprint_refuses_a_bundle_in_flight_and_sees_the_tick_index_and_acc():
    wrong = _DelaysTwo(LSM_PRE, LSM_W, LSM_CFG, frac_bits=12, p_parts=1)
    wrong.step(U1)
    assert any(slot for ring in wrong._kt._rings for slot in ring)
    with pytest.raises(ValueError, match="in-flight bundle"):
        wrong.fingerprint()
    pool = _host(window=4)
    pool.step(U1)
    before = pool.fingerprint()
    pool.acc = pool.acc + 1  # the summed accumulators are in the fingerprint
    assert pool.fingerprint() != before
    pool.acc = pool.acc - 1
    assert pool.fingerprint() == before
    pool._kt.tick_index += 1  # so is the tick index
    assert pool.fingerprint() != before


def test_the_hosted_fingerprint_holds_every_state_array_and_tells_the_refracting_host_apart():
    # the G1(c) P4 panel's pair: the faithful host and the one-refractory-tick host, one unit, one
    # percept. Activity, v, s and acc are equal after it; refrac is not, and the next tick differs
    config = {"lambda": 0.5, "g": 1, "T_in": 1}
    percepts = [np.array([1.5])]
    faithful = rp._hosted(rp.KernelLsmPool, 12, [[0]], [[0.0]], config, w_in=[[1.0]])
    refracts = rp._hosted(_Refracts, 12, [[0]], [[0.0]], config, w_in=[[1.0]])
    runs = [rp._record(pool, percepts, chain=True) for pool in (faithful, refracts)]
    assert (
        runs[0]["activity_crc"] == runs[1]["activity_crc"]
        and runs[0]["phi_crc"] == runs[1]["phi_crc"]
    )
    assert [faithful._kt.state.arrays["refrac"][0], refracts._kt.state.arrays["refrac"][0]] == [
        0,
        1,
    ]
    assert not rp._runs_agree(runs)  # the state fingerprint, and so the chain, tell them apart
    assert faithful.step([1.5]).tolist() == [1.0] and refracts.step([1.5]).tolist() == [0.0]
    # each of the KernelState arrays moves the fingerprint on its own, and restoring it restores it
    pool = _host(window=4)
    pool.step(U1)
    before = pool.fingerprint()
    assert set(pool._kt.state.arrays) == {"v", "thr", "refrac", "last_spike", "sign"}
    for name, array in pool._kt.state.arrays.items():
        saved = array[0].copy()
        array[0] = saved + 1
        assert pool.fingerprint() != before, name
        array[0] = saved
        assert pool.fingerprint() == before, name


def test_params_fingerprint_chains_the_live_bundles_and_a_live_write_moves_it():
    pool = _host(window=1)
    base = pool.params_fingerprint()
    stored = rp._crc_arrays(pool.params())
    assert base != stored  # the chain adds the live (post, pre, q) triples
    key = sorted(pool._syn.bundles)[0]
    pool._syn.bundles[key][1][0] += 1  # one live weight, the stored lists untouched
    assert pool.params_fingerprint() != base and rp._crc_arrays(pool.params()) == stored
    assert sorted(pool.params()) == ["pre", "w"]  # the stored lists' parameter keys


def test_the_lesion_zeroes_the_weights_the_bundles_and_the_charge_in_flight():
    pool = _host("front_end", window=1)
    pool.percept_drive(U1)  # the window percept: the boundary is taken, then the lesion lands
    assert pool.lesion_record["changed"] == ["w"] and pool.w.tolist() == [[0.0, 0.0]] * 3
    assert not pool.acc.any() and not pool._live_edges()["edge_q"].any()
    assert pool._kt.syn is pool._syn and pool.v.dtype == np.float32
    # the red's premise: charge WAS in flight when the lesion landed
    wrong = _KeepsCharge(LSM_PRE, LSM_W, LSM_CFG, "front_end", window=1, s_settle=1)
    wrong.percept_drive(U1)
    assert wrong.acc.any() and not wrong._live_edges()["edge_q"].any()


def test_perturb_kicks_exactly_one_float32_membrane():
    pool = rp.make_pool("lsm", TUNED, 100, "carried", window=4)
    for x in rp.random_percepts(100, 3):
        pool.percept(x)
    before, fingerprint = pool.v.copy(), pool.fingerprint()
    kick = pool.perturb()
    changed = np.flatnonzero(before != pool.v)
    assert pool.v.dtype == np.float32 and changed.tolist() == [kick["unit"]]
    assert pool.v[kick["unit"]] == np.float32(before[kick["unit"]]) + np.float32(kick["noise"])
    assert 1.0 <= abs(kick["noise"]) < 2.0 and pool.fingerprint() != fingerprint


def test_the_membrane_step_is_a_float32_product_plus_the_float64_drive_rounded_once():
    rng = np.random.default_rng(5)
    leaky = {"lambda": 0.9, "g": 1, "T_in": 1}
    pool = rp.KernelLsmPool([[i] for i in range(64)], np.zeros((64, 1)), leaky)
    pool.v = rng.random(64) * 0.5
    v0, u = pool.v.copy(), rng.random(64) * 0.4  # nothing reaches the threshold
    assert not pool.step(u).any()
    ext = np.rint(u * 4096).astype(np.int64)
    expected = (v0 * np.float32(0.9)).astype(np.float64) + ext / 4096.0
    assert np.array_equal(pool.v, expected.astype(np.float32))
    # the leak the host applies is float32(0.9), and the product is rounded before the add
    not_this = (v0.astype(np.float64) * 0.9 + ext / 4096.0).astype(np.float32)
    assert not np.array_equal(pool.v, not_this)


def test_the_quantizer_rounds_half_to_even_and_keeps_the_sign():
    pre = [[0]] * 5
    weights = [[0.0], [2.5 * LSB], [3.5 * LSB], [-2.5 * LSB], [-3.5 * LSB]]
    pool = rp.KernelLsmPool(pre, weights, {"lambda": 1.0, "g": 1, "T_in": 1})
    assert pool._live_edges()["edge_q"].tolist() == [0, 2, 4, -2, -4]
    assert pool._quantize_drive(np.array([0.5, 1.5, 2.7, -0.5, -2.7]) * LSB).tolist() == [
        0,
        2,
        3,
        0,
        -3,
    ]
    assert _Truncates(pre, weights, {"lambda": 1.0, "g": 1, "T_in": 1})._live_edges()[
        "edge_q"
    ].tolist() == [0, 2, 3, -2, -3]


def test_kernel_state_and_tick_cannot_be_deep_copied_so_the_pool_rebuilds_its_copy():
    pool = rp.make_pool("lsm", TUNED, 100, "carried", window=4)
    with pytest.raises(TypeError, match="module"):
        copy.deepcopy(pool._kt)  # KernelTick holds the numpy module as an attribute
    with pytest.raises(TypeError, match="module"):
        copy.deepcopy(pool._kt.state)  # and so does KernelState
    sequence = rp.random_percepts(100, 8)
    for x in sequence[:6]:
        pool.percept(x)
    twin = copy.deepcopy(pool)
    assert type(twin) is type(pool) and twin.fingerprint() == pool.fingerprint()
    assert twin.boundary == pool.boundary and twin.count == pool.count == 6
    assert twin._kt is not pool._kt and twin._syn is not pool._syn and twin._kt.syn is twin._syn
    for name, array in pool._kt.state.arrays.items():
        assert not np.shares_memory(twin._kt.state.arrays[name], array), name
    for key, (posts, qs) in pool._syn.bundles.items():
        t_posts, t_qs = twin._syn.bundles[key]
        assert not np.shares_memory(posts, t_posts) and not np.shares_memory(qs, t_qs)
    # writes to the copy's live bundle, membranes and charge leave the original untouched
    before = pool.fingerprint()
    twin._syn.bundles[next(iter(twin._syn.bundles))][1][0] += 1
    twin.v = np.ones(twin.n)
    twin.acc = np.ones(twin.n, dtype=np.int64)
    assert pool.fingerprint() == before
    # a fresh copy continues exactly as the original does
    clean = copy.deepcopy(pool)
    assert np.array_equal(clean.percept(sequence[6]), pool.percept(sequence[6]))
    assert clean.fingerprint() == pool.fingerprint()


def test_the_copy_carries_the_ring_slots_the_tick_index_and_the_fault_class():
    wrong = _DelaysTwo(LSM_PRE, LSM_W, LSM_CFG, frac_bits=12, p_parts=2)
    wrong.step(U1)
    assert any(slot for ring in wrong._kt._rings for slot in ring)  # charge in a ring, not in acc
    twin = copy.deepcopy(wrong)
    assert type(twin) is _DelaysTwo and twin._kt.tick_index == wrong._kt.tick_index == 1
    assert twin._kt._rings == wrong._kt._rings and twin._kt._rings is not wrong._kt._rings
    for u in (U2, U3, U1):
        assert np.array_equal(wrong.step(u), twin.step(u))
        assert np.array_equal(wrong.v, twin.v) and np.array_equal(wrong.acc, twin.acc)


def test_the_startup_diagnostic_runs_on_a_hosted_pool_and_leaves_it_untouched():
    pool = rp.make_pool("lsm", CONFIG["lsm"], 200)
    before = pool.fingerprint()
    result = rp.startup_diagnostic(pool)
    assert pool.fingerprint() == before
    assert set(result["transient"]) == {0, 16, 64}
    assert all(0.0 <= t <= 1.0 for t in result["transient"].values())


@pytest.mark.parametrize("form", rp.FORMS)
def test_the_host_takes_its_window_zero_boundary_at_construction(form):
    pool = _host(form, window=0)
    assert pool.count == 0 and pool.boundary is not None and pool.frozen()
    assert pool._live_edges()["edge_q"].any() == (form != "front_end")
    assert (pool.lesion_record is not None) == (form == "front_end")


def test_the_pool_delivers_at_delay_one_fires_on_ge_and_never_refracts_by_hand():
    # the chain 0 -> 1 -> 2 with only unit 0 driven at its first step: spikes at ticks 0, 1, 2 at
    # delay one, at 0, 2, 4 with a delay of two
    drives = [[1.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 4
    for cls, spikes in ((rp.KernelLsmPool, [0, 1, 2]), (_DelaysTwo, [0, 2, 4])):
        pool = rp._hosted(cls, 12, CHAIN_PRE, CHAIN_W, CHAIN_CFG)
        assert [t for t, row in enumerate(_spikes(pool, drives)) if any(row)] == spikes
    # at lambda 1, 0.75 then 0.25 reaches exactly 1.0: it spikes on `>=` and not on `>`
    for cls, fires in ((rp.KernelLsmPool, True), (_StrictThreshold, False)):
        pool = rp._hosted(cls, 12, [[0]], [[0.0]], {"lambda": 1.0, "g": 1, "T_in": 1})
        assert (_spikes(pool, [[0.75], [0.25]])[1] == [1.0]) == fires
    # one lsb below it never does
    pool = rp._hosted(rp.KernelLsmPool, 12, [[0]], [[0.0]], {"lambda": 1.0, "g": 1, "T_in": 1})
    assert _spikes(pool, [[0.75], [0.25 - LSB]]) == [[0.0], [0.0]]
    # a supra-threshold drive spikes at every tick without refraction, every other tick with one
    for cls, spikes in ((rp.KernelLsmPool, [1.0] * 4), (_Refracts, [1.0, 0.0, 1.0, 0.0])):
        pool = rp._hosted(cls, 12, [[0]], [[0.0]], {"lambda": 0.5, "g": 1, "T_in": 1})
        assert [row[0] for row in _spikes(pool, [[1.5]] * 4)] == spikes


def test_a_weight_below_half_an_lsb_and_a_drive_below_it_round_to_the_lattice():
    weight = 0.25 - 2.0**-14
    pool = rp.KernelLsmPool([[0], [0]], [[0.0], [weight]], {"lambda": 1.0, "g": 1, "T_in": 1})
    assert weight != 0.25 and pool._live_edges()["edge_q"].tolist() == [0, 1024]
    pool = rp._hosted(rp.KernelLsmPool, 12, [[0]], [[0.0]], {"lambda": 1.0, "g": 1, "T_in": 1})
    assert pool._quantize_drive(np.array([0.25 - 2.0**-30])).tolist() == [1024]


# ---- G1(c) P4b: the hosted pool's R-b3, R-b4 and R-b5 --------------------------------------------

HOSTED_SMALL = {"percepts": 64, "window": 32}  # reduced sizes; the full sizes are the campaign's
TUNED = {"k": 8, "w": 0.25, "lambda": 0.9, "g": 4, "T_in": 1}  # G1(c) 2.2's tuned configuration
WITNESS = rp.R_B3_WITNESS["lsm"]


@pytest.mark.parametrize("config", [TUNED, WITNESS], ids=["tuned", "witness"])
def test_the_hosted_pool_is_equal_across_p_and_threads_activity_state_and_chain(config):
    result = rp.check_r_b3("lsm", config, **HOSTED_SMALL)
    numbers = result["numbers"]
    assert result["verdict"] and numbers["equal"] == {"carried": True, "settled": True}
    assert numbers["settings"] == [{"P": p, "threads": t} for p, t in ((1, 1), (8, 1), (8, 4))]
    assert "vacuous" in numbers["blas"] and "vacuous" in numbers["torch"]
    forms = ("carried", "settled")
    assert set(numbers["runs"]) == {f"P{p}/T{t}/{f}" for p, t in rp.R_B3_HOSTED for f in forms}
    # every run records its activity, phi, final state and the per-percept state chain
    assert all(
        set(run) == {"activity_crc", "steps", "phi_crc", "state", "state_chain"}
        for run in numbers["runs"].values()
    )
    assert numbers["runs"]["P8/T4/settled"]["steps"] == numbers["runs"]["P1/T1/settled"]["steps"]


def test_the_comparator_sees_each_field():
    result = rp.check_r_b3("lsm", WITNESS, percepts=16, window=8)
    assert result["verdict"]
    run = result["numbers"]["runs"]["P1/T1/carried"]
    assert rp._runs_agree([run, dict(run)]) and not rp._runs_agree([])  # no run, no agreement
    for key in ("activity_crc", "steps", "phi_crc", "state", "state_chain"):
        assert not rp._runs_agree([run, dict(run, **{key: run[key] + 1})]), key


def test_the_float_accumulation_twin_red_fires_and_its_fixture_exercises_the_spread():
    result = rp.check_r_b3_hosted_red("lsm", **HOSTED_SMALL)
    numbers = result["numbers"]
    assert result["verdict"] and result["red"], numbers
    # unseeded order changed the chain; the stored order twice agrees: the order is the cause
    assert numbers["differs"] and numbers["chain_changed"] and numbers["control_agrees"]
    assert numbers["config"] == WITNESS and numbers["exercise_count"] >= rp.R_B3_MULTI
    for row in numbers[
        "counts"
    ].values():  # both runs: >= 100 unit-steps with 3+ spikes, and with both scaled entries
        assert row["multi"] >= rp.R_B3_MULTI and row["spread"] >= rp.R_B3_MULTI
        assert row["spread"] <= row["multi"]
    assert numbers["unit_spread_min"] >= rp.R_B3_SPREAD == 2**12
    # the gate's own wiring control, beside it and deciding nothing here
    assert numbers["wiring"] == {"float_diverged": True, "int_agreed": True}
    assert tuple(determinism_gate.broken_control()) == (True, True)


def _twin(order, *, lattice=False, window=32):
    base = rp.make_pool("lsm", WITNESS, rp.DEV_SEED, "carried", window=window)
    weights = np.rint(base.w * 2**12) / 2**12 if lattice else base.w
    cls = {"unseeded": rp._FloatTwin, "forward": rp._ForwardTwin}[order]
    return cls(base.pre, weights, base.config, "carried", base.w_in, window=window, s_settle=64)


def test_the_twin_scales_one_entry_up_and_one_down_and_leaves_the_network_as_it_was():
    base = rp.make_pool("lsm", WITNESS, rp.DEV_SEED, "carried", window=32)
    twin = _twin("forward")
    assert np.array_equal(twin.pre, base.pre) and isinstance(twin, rp.KernelLsmPool)
    assert np.array_equal(twin.w[:, 2:], base.w[:, 2:])
    assert np.array_equal(twin.w[:, 0], base.w[:, 0] * 2**12)
    assert np.array_equal(twin.w[:, 1], base.w[:, 1] / 2**12)
    assert (
        rp._unit_spread(twin.w) >= 2**12 > rp._unit_spread(base.w)
    )  # the spread is the twin's own


def test_the_twin_is_deterministic_in_the_stored_order_and_blind_without_its_spread(monkeypatch):
    seq = rp.cycled_fixture(HOSTED_SMALL["percepts"])
    forward = [rp._record(_twin("forward"), seq, chain=True) for _ in range(2)]
    assert rp._runs_agree(forward)
    assert not rp._runs_agree([forward[0], rp._record(_twin("unseeded"), seq, chain=True)])
    # why 2.9 re-founded the red: lattice weights at 2**-12 and no spread sum exactly in float32 in
    # any order, so the same twin fed them cannot tell an unseeded order from the stored one
    monkeypatch.setattr(rp, "R_B3_SPREAD", 1)
    lattice = rp._record(_twin("forward", lattice=True), seq, chain=True)
    assert rp._runs_agree([lattice, rp._record(_twin("unseeded", lattice=True), seq, chain=True)])


def test_the_twin_counts_its_unit_steps_by_hand():
    # four units with four entries each; entries 0 and 1 are the scaled ones
    pre = [[0, 1, 2, 3], [0, 0, 0, 0], [1, 2, 3, 0], [2, 2, 2, 2]]
    twin = rp._FloatTwin(
        pre,
        [[0.5, 0.25, 0.125, 0.0625]] * 4,
        {"lambda": 1.0, "g": 1, "T_in": 1},
        window=4,
        s_settle=0,
    )
    twin.s = np.array(
        [True, True, True, True]
    )  # every unit hears four spiking entries, scaled ones too
    twin.step(np.zeros(4))
    assert (twin.multi, twin.spread) == (4, 4)
    # unit 1 silent: units 0 and 2 hear three entries but one scaled entry is silent (multi only),
    # units 1 and 3 hear four through unit 0 and unit 2 (both scaled entries spiking: multi, spread)
    twin.s = np.array([True, False, True, True])
    twin.step(np.zeros(4))
    assert (twin.multi, twin.spread) == (4 + 4, 4 + 2)
    twin.s = np.array(
        [False, False, True, False]
    )  # units 0 and 2 hear one entry, unit 1 none, unit 3 four
    twin.step(np.zeros(4))
    assert (twin.multi, twin.spread) == (8 + 1, 6 + 1)


# R-b4's kicks are unseeded. On the R-b3 witness (saturated drive) the hosted red left phi unmoved
# in 6 of 150 trials at 16 kicks (0 of 150 at 48), so it is asserted where it never failed in 150
# trials: the tuned configuration, which R-b4 is pre-registered on, and the persistence witness
@pytest.mark.parametrize(
    "config", [TUNED, rp.R_B5_PERSISTENCE_WITNESS], ids=["tuned", "persistence-witness"]
)
def test_the_hosted_pool_noise_red_changes_the_state_chain_and_phi(config):
    pool = rp.make_pool("lsm", config, rp.DEV_SEED)
    before = pool.fingerprint()
    pool.perturb()
    assert pool.fingerprint() != before and pool.v.dtype == np.float32
    result = rp.pool_noise_red("lsm", config, window=32)
    assert result["verdict"] and result["numbers"]["state"] and result["numbers"]["phi"], result


@pytest.mark.parametrize("form", ["reset", "settled"])
def test_hosted_erasure_holds_and_skipping_v_or_acc_fails_it_but_s_is_never_read(form):
    config, carrier = rp.R_B5_ERASURE_WITNESS["lsm"]
    green = rp.check_r_b5_erasure("lsm", config, form, window=32)
    assert green["verdict"] and green["numbers"]["prefixes_diverged"], green["numbers"]
    red = rp.check_r_b5_erasure_red("lsm", form, window=32)
    assert carrier == "v" and red["verdict"], red["numbers"]
    assert red["numbers"]["fired"] == {"v": True, "s": False, "acc": True}
    by = red["numbers"]["by"]
    assert "state" in by["v"] and "state" in by["acc"] and by["s"] == "none"


def test_the_hosted_s_is_a_recorded_mask_the_pool_never_reads():
    u = rp.drive(rp.w_in(rp.DEV_SEED), rp.fixture_percepts()[0])
    pool = rp.make_pool("lsm", WITNESS, rp.DEV_SEED, "carried", window=32)
    for _ in range(3):
        pool.step(u)
    twin = copy.deepcopy(pool)
    twin.s = ~twin.s  # a different last mask
    assert np.array_equal(pool.step(u), twin.step(u))
    # the red of this test's own reading: a different membrane DOES move the next step
    other = copy.deepcopy(pool)
    other.v = other.v + np.float32(0.5)
    assert not np.array_equal(pool.step(u), other.step(u))
    # so the hosted skip-s red cannot fire: the first step overwrites the mask it kept, and the
    # erasure it was built to break still holds
    skipped = rp.check_r_b5_erasure("lsm", WITNESS, "reset", window=32, skip="s")
    assert skipped["verdict"] and skipped["numbers"]["skip"] == "s"


def test_hosted_persistence_and_its_reset_every_percept_red():
    result = rp.check_r_b5_persistence(window=32)
    assert result["verdict"] and result["numbers"]["persists"]
    assert not rp.check_r_b5_persistence(window=32, carried_form="reset")["verdict"]


def test_the_hosted_freeze_is_re_based_on_the_live_bundles_and_fails_on_that_form_alone():
    green = rp.check_r_b5_freeze("lsm", WITNESS, window=32, after=4)
    assert green["verdict"], green["numbers"]
    red = rp.check_r_b5_freeze_red("lsm", WITNESS, window=32, after=4)
    assert red["verdict"], red["numbers"]
    assert red["numbers"]["fired"] == ["carried"] and red["numbers"]["boundary_agree"]
    # the write is q + 1 on one live bundle weight and leaves the stored w as it was
    pool = rp._write_live_after_boundary(
        rp.make_pool("lsm", WITNESS, rp.DEV_SEED, "carried", window=4)
    )
    before = {key: q.copy() for key, (_, q) in pool._syn.bundles.items()}
    stored = rp._crc_arrays(pool.params())  # the stored (pre, w) alone
    for x in rp.cycled_fixture(4):
        pool.percept(x)
    moved = [key for key, (_, q) in pool._syn.bundles.items() if not np.array_equal(q, before[key])]
    assert len(moved) == 1 and rp._crc_arrays(pool.params()) == stored  # w untouched ...
    assert pool.frozen() is False  # ... and only the live edges' chain in params_fingerprint saw it
    delta = pool._syn.bundles[moved[0]][1] - before[moved[0]]
    assert delta.tolist() == [1] + [0] * (len(delta) - 1)


# ---- G3 grow round 2: zero-born edges on the compiled host (launcher D-86, landing gate) --------


class _PlantedTrainer(rp.KernelLsmPool):
    """The positive control, not a method: after every step, each zero-born slot whose pre and post
    both fired takes +2**-8 and the table is recompiled, so a detector that cannot see a trained
    zero-born edge fails here first."""

    def step(self, u):
        activity = super().step(u)
        fired = self.s
        w = self.w.copy()
        for i, c in _zero_born_slots(self):
            if fired[i] and fired[self.pre[i, c]] and w[i, c] == 0.0:
                w[i, c] = 2.0**-8
        if not np.array_equal(w, self.w):
            self.w = w
            self._syn = self._compile()
            self._kt.syn = self._syn
        return activity


def _zero_born_slots(pool):
    return [(i, 0) for i in range(pool.n)]


def _zero_born(cls, form):
    """TUNED's draw at N, slot 0 of every row born zero, window 0 (parameters final at birth)."""
    _, pre, w = rp._lsm_draw(rp.realization_seed("lsm", rp.DEV_SEED), rp.N, TUNED["k"], TUNED["w"])
    w[:, 0] = 0.0
    return cls(pre, w, TUNED, form, rp.w_in(rp.DEV_SEED), window=0, s_settle=rp.S_SETTLE)


def _zero_born_outcome(pool, percepts):
    """'dormant' when, after the percepts, every zero-born slot's stored weight and live q are 0 and
    the parameters equal their construction fingerprint; 'trainable' otherwise. Also returns whether
    some zero-born slot saw its pre and its post fire on one tick (a quiet pool proves nothing)."""
    born = pool.params_fingerprint()
    coactive = []

    def observe(current, activity):
        fired = activity != 0
        if any(fired[i] and fired[current.pre[i, c]] for i, c in _zero_born_slots(current)):
            coactive.append(True)

    pool.observer = observe
    for x in percepts:
        pool.percept(x)
    pool.observer = None
    live = pool._live_edges()
    zero_q = all(
        np.all(live["edge_q"][(live["edge_post"] == i) & (live["edge_pre"] == pool.pre[i, c])] == 0)
        for i, c in _zero_born_slots(pool)
        if np.count_nonzero(pool.pre[i] == pool.pre[i, c]) == 1  # a duplicate pre shares the key
    )
    stored_zero = all(pool.w[i, c] == 0.0 for i, c in _zero_born_slots(pool))
    dormant = zero_q and stored_zero and pool.params_fingerprint() == born
    return ("dormant" if dormant else "trainable"), bool(coactive)


@pytest.mark.parametrize("form", ("carried", "reset", "settled"))
def test_zero_born_edge_trainability_on_the_compiled_host(form):
    """D-86 landing gate: runs after the campaign, before claude/g3-grow-1029 lands.

    Every live path that can write this pool is exercised: the wake percept loop (percept ->
    percept_drive -> step) under each form that keeps its table (front_end lesions it at birth),
    resets included. No sleep or plasticity module binds KernelLsmPool at base c4c10b28 (git grep
    over src/snn_data_lake/sleep and src/snn_data_lake/plasticity: no hit).
    PASS = 'dormant': zero-born edges stay zero; RULING.md's A3/B2 dormancy premise holds.
    FAIL = 'trainable': zero-init is trainable on this host and the A-choice reopens (D-86).
    """
    seq = rp.cycled_fixture(rp.W0_PERCEPTS)
    planted, planted_coactive = _zero_born_outcome(_zero_born(_PlantedTrainer, form), seq)
    assert planted_coactive and planted == "trainable"  # the detector sees a trained zero edge
    outcome, coactive = _zero_born_outcome(_zero_born(rp.KernelLsmPool, form), seq)
    assert coactive, "no zero-born slot saw pre and post fire together: the run proves nothing"
    assert outcome == "dormant"


# ---- G3 grow round 2: the encoder follows pool.n (RULING.md item 3) ------------------------------

# Measurement points, never caps: 257 is the smallest N above the born 256, 512 its first doubling.
JOIN_N = (257, 512)


@pytest.mark.parametrize("n", JOIN_N)
def test_encoder_extension_is_prefix_stable(n):
    grown, born = rp.w_in(rp.DEV_SEED, n), rp.w_in(rp.DEV_SEED)
    assert grown.shape == (n, rp.N_IN) and np.array_equal(grown[: rp.N], born)
    # the red: an extension drawn from any other stream changes the old rows
    reseeded = np.random.default_rng(rp.DEV_SEED * 31 + 8).standard_normal((n, rp.N_IN)) / 8
    assert not np.array_equal(reseeded[: rp.N], born)


def test_encoder_row_n_is_drawn_by_w_ins_law():
    row = rp.w_in(rp.DEV_SEED, rp.N + 1)[rp.N]
    law = np.random.default_rng(rp.DEV_SEED * 31 + 7).standard_normal((rp.N + 1, rp.N_IN)) / 8
    assert np.array_equal(row, law[rp.N])
    assert np.count_nonzero(row) == rp.N_IN  # the red, a zero row, cannot pass


@pytest.mark.parametrize("n", JOIN_N)
def test_encoder_shape_follows_pool_n(n):
    _, pre, w = rp._lsm_draw(rp.realization_seed("lsm", rp.DEV_SEED), n, TUNED["k"], TUNED["w"])
    pool = rp.KernelLsmPool(pre, w, TUNED, w_in=rp.w_in(rp.DEV_SEED, n), window=0)
    pool.replace_encoder(rp.w_in(rp.DEV_SEED + 1, n))
    assert pool.w_in.shape == (n, rp.N_IN)
    with pytest.raises(ValueError, match="REFUSED"):
        pool.replace_encoder(rp.w_in(rp.DEV_SEED))  # 256 rows no longer fit this pool
    drawn = rp.w_in(rp.DEV_SEED, n)
    assert rp.fresh_encoder(rp.DEV_SEED, drawn).shape == (n, rp.N_IN)
    moved = rp.transported_encoder(drawn, np.ones((n, rp.N_IN)))
    assert moved.shape == (n, rp.N_IN)
