"""specs/022 T111 (G3) P2: ONE grower -- the NG-SNN candidate lifecycle, factored so a second consumer can call it.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.5. `NGSNN.fit` drew a multi-scale
candidate pool, scored it on the residual (Eq. 6-7), and admitted the top block; `NGSNN.prune_unearned` dropped
units by the one-pass held-out discipline. Those three steps now live here and `NGSNN` calls them, so the taught
learner's problem loop (G3 P3) consumes the SAME lifecycle instead of a second grower (P2). The percept consumer
is byte-identical across the move (tests/test_candidate_lifecycle.py pins `growth_fingerprint` and the trajectory
measured before it), and the float32 1e12-bonus tie below is preserved byte-for-byte: its fix is its own later
commit with its own before/after fingerprint (owner question O5).

`select_block(..., reverse=True)` exists for 2.9 C6's red only (a reversed admission tie-break must change the
fingerprint); no live caller passes it.
"""

from dataclasses import dataclass

import torch


@dataclass
class Pool:
    W: torch.Tensor  # (P_total, d) candidate input weights, scales concatenated in draw order
    b: torch.Tensor  # (P_total,)
    score: torch.Tensor  # residual reduction (Eq. 6 weighted by per-class residual energy)
    passes: torch.Tensor  # meets the consistency criterion (Eq. 7)
    zero: torch.Tensor  # constant response across the scoring rows (and not dead)


def eq6(Sp, E, enorm, reg_eps):
    """THE Eq. 6 algebra, shared by every consumer of the one grower (cp1 #16): per candidate column p
    and class q, C_pq = <e_q, s_p>^2 / (||s_p||^2 ||e_q||^2 + reg_eps), and the residual reduction
    score_p = sum_q C_pq ||e_q||^2. Returns (C, score, ||s_p||^2). Sp: (N, P) responses, E: (N, m)
    residual, enorm: (m,) per-class residual energy."""
    proj = Sp.t() @ E  # (P, m) <e_q, s_p>
    spn = (Sp * Sp).sum(dim=0)  # (P,)   ||s_p||^2
    C = proj.pow(2) / (spn.unsqueeze(1) * enorm.unsqueeze(0) + reg_eps)  # (P,m) Eq.6
    score = (C * enorm.unsqueeze(0)).sum(dim=1)  # (P,) residual reduction
    return C, score, spn


def draw_and_score(model, d, Xsel, Esel, enorm, thresh, n_units):
    """Draw the whole multi-scale candidate POOL on the model's CPU generator, evaluate it in one vectorized LIF
    pass over the scoring rows, and score it against the residual Esel. RNG consumption is unchanged from the
    pre-factoring `NGSNN.fit` (one rand(P, d) and one rand(P) per scale, in scale order)."""
    poolW, poolb, poolscore, poolpass, poolzero = [], [], [], [], []
    for Lam in model.lambdas:  # multi-scale candidate search (Alg. 1)
        P = model.t_cand
        # drawn on the CPU stream (see NGSNN.__init__), then moved -- seed decides, hardware does not
        Wp = ((torch.rand(P, d, generator=model.g) * 2 - 1) * Lam).to(model.device)  # (P,d)
        bp = ((torch.rand(P, generator=model.g) * 2 - 1) * Lam).to(model.device)  # (P,)
        Sp = model._rates(Xsel, Wp, bp)  # (Nsel, P) candidate responses
        C, score, spn = eq6(Sp, Esel, enorm, model.reg_eps)
        passes = (C >= thresh).all(dim=1)  # (P,) meets consistency criterion
        # L3 (specs/022, 2026-09-05): a DEAD candidate (rate exactly 0 on every scoring row) is excluded always;
        # a CONSTANT-response candidate (exact equality across the scoring rows, not a variance epsilon -- P1)
        # carries nothing once any unit exists. The full measurement record stays in NGSNN.fit's comments.
        dead = spn == 0.0
        zero_var = (Sp == Sp[0]).all(dim=0) & ~dead
        passes = passes & ~dead
        score = score.masked_fill(dead, 0.0)
        if n_units > 0:
            passes = passes & ~zero_var
            score = score.masked_fill(zero_var, 0.0)
        poolW.append(Wp)
        poolb.append(bp)
        poolscore.append(score)
        poolpass.append(passes)
        poolzero.append(zero_var)
    return Pool(
        torch.cat(poolW, 0),
        torch.cat(poolb, 0),
        torch.cat(poolscore, 0),
        torch.cat(poolpass, 0),
        torch.cat(poolzero, 0),
    )


def select_block(pool, delta, room, n_units, reverse=False):
    """The admitted block: the top-`min(delta, room)` candidates, passing first then by score (the float32 1e12
    bonus ties passing candidates -- preserved, O5), keeping only genuinely contributing ones, and at most one
    zero-variance unit, only as the first unit (L3). `reverse=True` reverses the pool order before the top-k
    (C6's red only)."""
    rank = pool.score + pool.passes.float() * 1e12  # prefer passing, then by score
    k = min(delta, room)
    if reverse:
        n = int(rank.shape[0])
        top = [n - 1 - int(j) for j in torch.topk(rank.flip(0), k).indices.tolist()]
    else:
        top = torch.topk(rank, k).indices.tolist()
    keep = [int(j) for j in top if pool.score[j].item() > 0.0 or bool(pool.passes[j].item())]
    if n_units == 0 and keep:  # L3: at most one zero-variance unit, and only as the first
        zk = [j for j in keep if bool(pool.zero[j].item())]
        if zk:
            keep = [j for j in keep if not bool(pool.zero[j].item())] + zk[:1]
    return keep


# distinct-from: growth/function_preserving_growth.py -- its `_eligibility_seed` points a Net2Net/GradMax unit at
#   the MEAN input energy of a window (a disclosed unproven heuristic, no target); this trains a candidate on the
#   multiclass RESIDUAL through a fixed random feedback vector. Disclosed for the panel as a P2 question: if G3's
#   primary family deletes the trace-trained source (P8 clause 5), neither survives as a residual-directed seed.
# distinct-from: faculties/homeostatic_drives.py, engine/brain.py, faculties/wholistic_loop.py,
#   growth/growth_stop.py -- read 2026-10-04 (the duplicate guard's hits): a drive-preference softmax, a table
#   checksum, a metacognitive-signal source and a stop criterion; none trains a candidate unit. The softmax is
#   REUSED from infra/_mathutils.py, not rewritten.
def trace_train(w, threshold, window, tau, items, eta, epochs, seed):
    """THE TRACE-TRAINED CANDIDATE SOURCE (2.5): one capacity candidate's input row trained on the multiclass
    residual with an eligibility trace and a FIXED RANDOM feedback vector -- never a transposed forward weight
    (the e-prop invariant; 2.9 C7).

    `items` = [(ranks, y_onehot, p)] with p = softmax(v / T) of the learner's CURRENT label drives at the
    registered T = 1 drive unit (D4), fixed while the candidate trains (it has no output synapses before
    admission). Per item, e_i = PSP(t_ref - rank_i) at the candidate's membrane, t_ref = its first-spike step
    (the window end when it does not spike), and L = sum_c B_c (y_c - p_c) with B drawn once at the candidate's
    birth from `seed`. The update is the pre-registered dW_i = -eta * L * e_i; B is drawn from a symmetric law
    (B and -B equally likely), so the sign convention does not change the rule's distribution. Returns the
    trained row as a tuple."""
    import math
    import random

    from snn_data_lake.growth.grown_code import GrownUnit, latency

    rng = random.Random(seed)
    n_labels = len(items[0][1])
    feedback = [rng.gauss(0.0, 1.0) for _ in range(n_labels)]  # B: fixed at birth, never learned
    w = list(w)
    for _ in range(int(epochs)):
        for ranks, y, p in items:
            unit = GrownUnit(0, tuple(w), threshold, window, 1, ())
            t = latency(unit, ranks, tau)
            t_ref = window if t is None else t
            e = [math.exp(-(t_ref - r) / tau) if t_ref > r else 0.0 for r in ranks]
            L = sum(b * (yc - pc) for b, yc, pc in zip(feedback, y, p))
            for i in range(len(w)):
                w[i] += -eta * L * e[i]
    return tuple(w)


def label_probabilities(drives):
    """p = softmax(drives / T) at the registered T = 1 drive unit (D4), via the shared helper."""
    from snn_data_lake.infra._mathutils import softmax

    return softmax(list(drives))


def contribution(feature, residual, reg_eps=1e-6):
    """One candidate feature's Eq. 6 residual reduction over rows, through the SAME `eq6` algebra
    `draw_and_score` uses (cp1 #16), on the taught learner's residual (rows x labels lists); reg_eps
    defaults to NGSNN's. A dead or constant feature carries nothing (L3) and scores 0.0."""
    if all(x == 0.0 for x in feature) or all(x == feature[0] for x in feature):
        return 0.0
    Sp = torch.tensor([[float(x)] for x in feature], dtype=torch.float64)
    E = torch.tensor([[float(v) for v in row] for row in residual], dtype=torch.float64)
    _, score, _ = eq6(Sp, E, (E * E).sum(dim=0), reg_eps)
    return float(score[0])


@dataclass
class PruneResult:
    keep: list  # survivor columns, or None when nothing was pruned (identity unchanged)
    n_drop: int
    tr_idx: torch.Tensor = None  # the fitted rows the caller re-solves its readout on


def heldout_prune(S, Y, y_list, solve_beta, tol=0.0):
    """Consumer-agnostic one-pass held-out prune over a rate matrix S (N, L) and one-hot targets Y (N, m): a unit
    is dropped when removing it leaves held-out RMSE (on `marginal_holdout`'s window) no worse than `tol`; an
    all-redundant set keeps exactly the unit whose removal costs most (L3). Same discipline as the admission
    criterion; the single-pass ceiling is documented on `NGSNN.prune_unearned`."""
    from snn_data_lake.growth.ng_snn import marginal_holdout

    N = int(S.shape[0])
    n_val = max(8, int(0.2 * N))
    tr_list, va_list = marginal_holdout(y_list, n_val)  # L1a: the ONE window
    if len(tr_list) < 2 or len(va_list) < 2:
        return PruneResult(None, 0)  # too few rows to judge honestly -- keep everything
    tr_idx = torch.tensor(tr_list, device=S.device)
    va_idx = torch.tensor(va_list, device=S.device)
    L = int(S.shape[1])

    def _val_rmse(cols):
        beta = solve_beta(S[tr_idx][:, cols], Y[tr_idx])
        resid = Y[va_idx] - S[va_idx][:, cols] @ beta
        return resid.pow(2).mean().sqrt().item()

    allc = list(range(L))
    base = _val_rmse(allc)
    drop, cost = [], {}
    for j in range(L):
        keep = [c for c in allc if c != j]
        if not keep:
            continue
        cost[j] = _val_rmse(keep)
        if cost[j] <= base + tol:  # removing it costs nothing -> it earns nothing
            drop.append(j)
    if not drop:
        return PruneResult(None, 0, tr_idx)
    if len(drop) >= L:
        survivor = max(allc, key=lambda j: (cost.get(j, float("-inf")), -j))
        drop = [c for c in allc if c != survivor]
    keep = [c for c in allc if c not in set(drop)]
    return PruneResult(keep, len(drop), tr_idx)


def main():
    """Self-check: the percept consumer's byte identity on one pinned seed (C6's control), the reversed
    tie-break red (C6), and the consumer-agnostic prune on a planted duplicate column."""
    import importlib

    from snn_data_lake.growth.neurogenesis_engine import (
        NeurogenesisSelfImprove,
        SyntheticPerceptStream,
        growth_fingerprint,
    )

    torch.set_num_threads(1)

    def fp(reverse):
        # patch the PACKAGE module ng_snn calls through (under -m this file also runs as __main__)
        mod = importlib.import_module("snn_data_lake.growth.candidate_lifecycle")
        original = mod.select_block
        if reverse:
            mod.select_block = lambda *a, **k: original(*a, **{**k, "reverse": True})
        try:
            eng = NeurogenesisSelfImprove(seed=9000, device="cpu")
            SyntheticPerceptStream(seed=9000).drive(eng, 200)
            return growth_fingerprint(eng)
        finally:
            mod.select_block = original

    pinned = 452866023  # measured on the unrefactored ng_snn.py (tests/test_candidate_lifecycle.py)
    assert fp(False) == pinned, "the percept consumer is no longer byte-identical"
    assert fp(True) != pinned, "red: a reversed admission tie-break must change the fingerprint"
    g = torch.Generator().manual_seed(9900)
    S = torch.rand(40, 3, generator=g)
    S = torch.cat([S, S[:, :1]], 1)
    y = [i % 2 for i in range(40)]
    Y = torch.zeros(40, 2)
    Y[torch.arange(40), torch.tensor(y)] = 1.0
    from snn_data_lake.growth.ng_snn import NGSNN

    res = heldout_prune(S, Y, y, NGSNN(device="cpu")._solve_beta)
    assert res.n_drop >= 1, "red: a duplicated column must be pruned"
    print(
        f"candidate_lifecycle: seed 9000 fingerprint {pinned} byte-identical; reversed tie-break red fires; "
        f"duplicate column pruned ({res.n_drop} dropped)"
    )


if __name__ == "__main__":
    main()
