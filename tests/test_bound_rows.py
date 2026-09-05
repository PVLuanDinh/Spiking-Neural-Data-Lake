"""The predictor bound has ONE body (cerebellum/jepa_world_model.bound_rows; the live-floor arc's
commit A, 2026-09-05). Commit A landed on byte-identity of both module self-checks and an archived
planted cell -- and in all three the row-scaling branch fired 0 times (max row-abs-sum 1.735
against rho_max 4.0), so that evidence covered only the decay path (the record panel's INT-A1).
This is the scaling path's reachable red: bound_rows against verbatim copies of BOTH removed
bodies, on random matrices whose rows exceed the bound, bit for bit -- and a wrong body (the
inverted factor) is caught.
"""

import random

from snn_data_lake.cerebellum.jepa_world_model import bound_rows


def _removed_jwm(Wz, rho_max, wd):  # verbatim: JepaWorldModel.train_step at 4bae53a
    L = len(Wz)
    for i in range(L):
        for k in range(L):
            Wz[i][k] *= 1.0 - wd
    rowmax = max(sum(abs(Wz[i][k]) for k in range(L)) for i in range(L))
    if rowmax > rho_max:
        f = rho_max / rowmax
        for i in range(L):
            for k in range(L):
                Wz[i][k] *= f


def _removed_spk(Wz, rho_max, wd):  # verbatim: SpikingJEPA._spectral_bound at 4bae53a
    big_l = len(Wz)
    for i in range(big_l):
        wzi = Wz[i]
        for k in range(big_l):
            wzi[k] *= 1.0 - wd
    rowmax = max(sum(abs(Wz[i][k]) for k in range(big_l)) for i in range(big_l))
    if rowmax > rho_max:
        f = rho_max / rowmax
        for i in range(big_l):
            wzi = Wz[i]
            for k in range(big_l):
                wzi[k] *= f


def _inverted(Wz, rho_max, wd):  # a WRONG body: the factor upside down
    L = len(Wz)
    for i in range(L):
        for k in range(L):
            Wz[i][k] *= 1.0 - wd
    rowmax = max(sum(abs(Wz[i][k]) for k in range(L)) for i in range(L))
    if rowmax > rho_max:
        f = rowmax / rho_max
        for i in range(L):
            for k in range(L):
                Wz[i][k] *= f


def test_bound_rows_is_both_removed_bodies_where_the_scaling_fires():
    rng = random.Random(0)
    fired = caught = 0
    for _ in range(300):
        L = rng.choice((12, 16))
        scale = rng.choice((0.05, 0.5, 2.0))
        W = [[rng.uniform(-1.0, 1.0) * scale for _ in range(L)] for _ in range(L)]
        rho = rng.choice((0.9, L**0.5))
        a, b, c, d = ([r[:] for r in W] for _ in range(4))
        if max(sum(abs(v) for v in row) for row in W) * (1.0 - 1e-3) > rho:
            fired += 1
            _inverted(d, rho, 1e-3)
        bound_rows(a, rho, 1e-3)
        _removed_jwm(b, rho, 1e-3)
        _removed_spk(c, rho, 1e-3)
        assert a == b == c, "bound_rows must equal both removed bodies bit for bit"
        if d != W and d != a:
            caught += 1
    assert fired > 100, f"the scaling branch must fire on a real fraction of trials ({fired})"
    assert caught == fired, (
        f"the inverted body must differ wherever the branch fires ({caught}/{fired})"
    )
