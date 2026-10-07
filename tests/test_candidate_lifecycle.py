"""specs/022 T111 (G3) P2: ONE grower -- NGSNN.fit's candidate lifecycle factored for a second
consumer.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.5 and 2.9 C6. The percept
consumer must be BYTE-IDENTICAL across the refactor: `growth_fingerprint` and the trajectory are
pinned below on `SyntheticPerceptStream` seeds 9000-9003 (200 and 600 ticks) and on
`online_readout`'s self-check engines, each value MEASURED on the unrefactored tree (base a363ad7a /
e1b3ae3c's ng_snn.py) on 2026-10-04, cpu and cuda equal. The float32 1e12-bonus tie is preserved
byte-for-byte (its fix is its own later commit, owner question O5). C6's red: reversing the
admission tie-break must change the fingerprint. Every later stacked commit on
`neurogenesis_engine.py` (R2's hydration, R3's replay_rows, G7's persistence arrays) re-runs this
file. Each run builds a bare engine (no Brain, no lake writer).
"""

import json
import zlib

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.growth import candidate_lifecycle  # noqa: E402
from snn_data_lake.growth.neurogenesis_engine import (  # noqa: E402
    NeurogenesisSelfImprove,
    SyntheticPerceptStream,
    growth_fingerprint,
)

# (seed, ticks) -> (growth_fingerprint, trajectory crc, grown units); measured before the refactor
STREAM = {
    (9000, 200): (452866023, 11),
    (9001, 200): (3078134973, 7),
    (9002, 200): (97899718, 10),
    (9003, 200): (3362449400, 7),
    (9000, 600): (4153597545, 12),
    (9001, 600): (3409593730, 15),
    (9002, 600): (2073996705, 14),
    (9003, 600): (1976927636, 21),
}
TRAJ = {
    (9000, 200): 869405703,
    (9001, 200): 1746441905,
    (9002, 200): 3364797868,
    (9003, 200): 1141811203,
    (9000, 600): 1455820575,
    (9001, 600): 1991505230,
    (9002, 600): 2287058612,
    (9003, 600): 1128170985,
}
ONLINE = {
    (0, "batch"): (3259111491, 3695338077, 5),
    (0, "delta"): (3452081889, 3695338077, 5),
    (0, "delta_open"): (3249242654, 320847952, 13),
    (1, "batch"): (1808905792, 4245214917, 2),
    (1, "delta"): (459976505, 4245214917, 2),
    (1, "delta_open"): (2441742469, 934839829, 7),
}


def _traj(engine):
    return zlib.crc32(json.dumps(engine.trajectory, default=str).encode())


def _stream(seed, ticks):
    torch.set_num_threads(1)
    eng = NeurogenesisSelfImprove(seed=seed, device="cpu")
    SyntheticPerceptStream(seed=seed).drive(eng, ticks)
    return eng


@pytest.mark.parametrize("seed,ticks", sorted(STREAM))
def test_percept_consumer_is_byte_identical(seed, ticks):
    eng = _stream(seed, ticks)
    assert (growth_fingerprint(eng), int(eng.grown_units)) == STREAM[(seed, ticks)]
    assert _traj(eng) == TRAJ[(seed, ticks)]


@pytest.mark.parametrize("seed,name", sorted(ONLINE))
def test_online_readout_engines_are_byte_identical(seed, name):
    from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout, _run

    torch.set_num_threads(1)
    readout = None if name == "batch" else OnlineReadout("delta")
    gate = None if name == "delta_open" else ClosingGate(3)
    eng = _run(seed, readout, gate)
    assert (growth_fingerprint(eng), _traj(eng), int(eng.grown_units)) == ONLINE[(seed, name)]


def test_reversed_admission_tie_break_changes_the_fingerprint(monkeypatch):
    # C6's red: the equivalence check must be able to see a changed admission order
    original = candidate_lifecycle.select_block

    def reversed_order(*args, **kwargs):
        kwargs["reverse"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_lifecycle, "select_block", reversed_order)
    changed = [
        s
        for s in (9000, 9001, 9002, 9003)
        if growth_fingerprint(_stream(s, 200)) != STREAM[(s, 200)][0]
    ]
    assert changed, "reversing the tie-break changed no fingerprint: the equivalence check is blind"


def test_heldout_prune_is_consumer_agnostic():
    # the one-pass held-out discipline over any rate matrix: a duplicated column earns nothing
    g = torch.Generator().manual_seed(9900)
    S = torch.rand(40, 3, generator=g)
    S = torch.cat([S, S[:, :1]], 1)  # column 3 duplicates column 0
    y = torch.tensor([i % 2 for i in range(40)])
    Y = torch.zeros(40, 2)
    Y[torch.arange(40), y] = 1.0
    from snn_data_lake.growth.ng_snn import NGSNN

    solve = NGSNN(device="cpu")._solve_beta
    res = candidate_lifecycle.heldout_prune(S, Y, y.tolist(), solve)
    assert res.n_drop >= 1 and len(res.keep) >= 1
    assert set(res.keep) <= set(range(4)) and len(res.keep) + res.n_drop == 4


def test_contribution_scores_dead_and_constant_features_zero():
    resid = [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]]
    assert candidate_lifecycle.contribution([0.0, 0.0, 0.0], resid) == 0.0
    assert candidate_lifecycle.contribution([0.3, 0.3, 0.3], resid) == 0.0
    assert candidate_lifecycle.contribution([1.0, 0.0, 0.2], resid) > 0.0


def test_contribution_uses_the_incumbent_eq6_algebra_on_low_energy_residuals():
    # cp1 #16: Codex's case -- feature [1, 0], residual [[1e-4], [0]]: the incumbent regularizes
    # ||s||^2 * ||e_c||^2 + reg_eps, so its score is C * ||e||^2 with C = <e,s>^2 / (that)
    feature, residual = [1.0, 0.0], [[1e-4], [0.0]]
    proj2, sn, en = (1e-4) ** 2, 1.0, (1e-4) ** 2
    incumbent = proj2 / (sn * en + 1e-6) * en
    got = candidate_lifecycle.contribution(feature, residual)
    assert got == pytest.approx(incumbent, rel=1e-4)
    assert got != pytest.approx(
        proj2 / (sn + 1e-6), rel=0.5
    )  # the divergent form scores ~101x higher


# ---- cp1 #18 / C9: the shared silencing helper and the matched-source picker (2.5) ---------------


def test_silence_units_on_a_grown_segment():
    from snn_data_lake.growth.grown_code import GrownUnit

    units = tuple(GrownUnit(i, (1.0,), 0.5, 4, 1, (0.0,)) for i in (1, 2, 3))
    out = candidate_lifecycle.silence_units(units, [2])
    assert [u.active for u in out] == [True, False, True]
    assert all(u.active for u in units)  # the caller's segment is untouched
    with pytest.raises(KeyError):
        candidate_lifecycle.silence_units(units, [9])


def test_silence_units_on_an_ngsnn_readout():
    from snn_data_lake.growth.ng_snn import NGSNN

    m = NGSNN(device="cpu")
    m.W, m.b = torch.ones(3, 2), torch.zeros(3)
    m.beta = torch.ones(3, 2)
    out = candidate_lifecycle.silence_units(m, [0, 2])
    assert out.beta[:, 0].tolist() == [0.0, 1.0, 0.0] and m.beta.sum().item() == 6.0
    with pytest.raises(KeyError):
        candidate_lifecycle.silence_units(m, [3])


def test_match_sources_is_nearest_without_replacement_and_excludes_dead():
    targets = [("g1", 1.0, 0.5), ("g2", 1.0, 0.5), ("g3", 0.0, 0.5)]  # g3 is dead
    pool = [("d1", 1.1, 0.5), ("d2", 3.0, 0.5), ("d3", 0.9, 0.0), ("d4", 1.6, 0.6)]
    pairs = candidate_lifecycle.match_sources(targets, pool)
    assert pairs == [("g1", "d1"), ("g2", "d4")]  # d2: norm ratio 3 > 2; d3 dead; no reuse of d1


def test_match_feasibility_rule():
    ok = candidate_lifecycle.match_feasible
    S = candidate_lifecycle.DEV_SEEDS

    def d(*vals):
        return dict(zip(S, vals, strict=True))

    assert ok(d(*[5] * 8), d(*[9] * 8))
    assert not ok(d(*[5] * 7, 0), d(*[9] * 8))  # k >= 1 on EVERY seed
    assert not ok(d(*[4] * 3, *[5] * 5), d(*[9] * 8))  # half covered on only 5 of 8
    # Codex cp5 #8: an omitted dev seed is NOT feasible, and neither is a stranger seed in its place
    assert not ok({s: 5 for s in S[:6]}, {s: 9 for s in S[:6]})
    assert not ok({**d(*[5] * 8), 9099: 5}, d(*[9] * 8))
    assert not ok({s: 5 for s in (*S[:7], 9099)}, d(*[9] * 8))


# ---- the pure unearned-unit rule (P2: ONE rule for every pruner; G6 imports it) ----


def test_select_unearned_drops_every_unit_whose_removal_is_no_worse_than_base_plus_tol():
    base = 1.0
    assert candidate_lifecycle.select_unearned(base, [1.5, 1.0, 0.7, 2.0]) == ([1, 2], None)
    # <= : a cost exactly at base + tol is droppable; just above is not
    assert candidate_lifecycle.select_unearned(base, [1.25, 1.2500001], tol=0.25) == ([0], None)
    assert candidate_lifecycle.select_unearned(base, [3.0, 2.0]) == ([], None)  # none droppable


def test_select_unearned_keeps_exactly_the_costliest_removal_when_all_are_droppable():
    # costs 0.5 / 0.9 / 0.9 / 0.2 all <= base 1.0: survivor = largest cost, ties -> lowest index
    assert candidate_lifecycle.select_unearned(1.0, [0.5, 0.9, 0.9, 0.2]) == ([0, 2, 3], 1)
    assert candidate_lifecycle.select_unearned(1.0, [0.9, 0.9]) == ([1], 0)


def test_select_unearned_never_drops_or_keeps_an_infinite_cost_and_one_unit_is_never_pruned():
    inf = float("inf")
    # an infeasible removal (cost +inf) is not droppable and is never the all-droppable survivor
    assert candidate_lifecycle.select_unearned(1.0, [inf, 0.5, 0.9]) == ([1, 2], None)
    assert candidate_lifecycle.select_unearned(1.0, [inf, inf]) == ([], None)
    # n <= 1: nothing to prune
    assert candidate_lifecycle.select_unearned(1.0, [0.0]) == ([], None)
    assert candidate_lifecycle.select_unearned(1.0, []) == ([], None)


def test_select_unearned_is_the_rule_heldout_prune_runs():
    # control: planted duplicate column through the real held-out prune drops the same index set
    S = torch.tensor([[1.0, 1.0, 0.0]] * 12 + [[0.0, 0.0, 1.0]] * 12)
    Y = torch.zeros(24, 2)
    Y[:12, 0] = 1.0
    Y[12:, 1] = 1.0
    y = [0] * 12 + [1] * 12

    def solve(Sx, Yx):
        return torch.linalg.lstsq(Sx, Yx).solution

    res = candidate_lifecycle.heldout_prune(S, Y, y, solve)
    assert res.keep is not None and res.n_drop == 3 - len(res.keep)


def test_select_unearned_refuses_a_non_finite_base_or_tolerance():
    # Codex b2 MINOR: base=inf or tol=inf made a +inf (infeasible) removal droppable
    inf = float("inf")
    for base, tol in ((inf, 0.0), (1.0, inf), (float("nan"), 0.0), (1.0, float("nan"))):
        with pytest.raises(ValueError, match="finite"):
            candidate_lifecycle.select_unearned(base, [inf, 0.5], tol=tol)
