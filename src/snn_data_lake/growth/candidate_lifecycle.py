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

import math
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


def silence_units(consumer, unit_ids):
    """THE shared silencing operation (2.5; C9; R3 imports it), returning a silenced COPY -- the
    caller's object is never modified. A grown segment (a sequence of units with `unit_id` and
    `silenced()`, e.g. growth/grown_code.GrownUnit -- duck-typed, so this live module does not import
    it) gets those units' `active` flag cleared; an NG-SNN model (W and beta) gets those COLUMNS'
    readout rows zeroed. An unknown id raises KeyError: silencing nothing must never read as a
    silencing."""
    import copy

    ids = list(unit_ids)
    if hasattr(consumer, "beta") and hasattr(consumer, "W"):
        n = int(consumer.W.shape[0])
        missing = [i for i in ids if not 0 <= int(i) < n]
        if missing:
            raise KeyError(f"no unit column(s) {missing} in a model of {n} units")
        out = copy.deepcopy(consumer)
        for i in ids:
            out.beta[int(i)] = 0.0
        return out
    by_id = {u.unit_id: u for u in consumer}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"no grown unit(s) {missing} in the segment")
    silence = set(ids)
    return tuple(u.silenced() if u.unit_id in silence else u for u in consumer)


def match_sources(targets, pool, norm_ratio=(0.5, 2.0), rate_tol=0.25):
    """C9's matched control (2.9, Codex b1 G3 #1): pair each target source (id, output-synapse L2 norm,
    firing rate on the evaluated items) with its nearest pool source, WITHOUT replacement, within the
    tolerance fixed in the pre-registration (norm ratio in [0.5, 2], rate difference <= 0.25); dead
    sources (norm 0 or rate 0) are excluded from both sides. Distance = |log norm ratio| + |rate
    difference|; ties -> the earlier pool entry. Returns [(target_id, pool_id)] in target order."""
    import math

    lo, hi = norm_ratio
    free = [p for p in pool if p[1] > 0.0 and p[2] > 0.0]
    pairs = []
    for tid, tn, tr in targets:
        if tn <= 0.0 or tr <= 0.0:
            continue
        best, best_d = None, None
        for k, (_pid, pn, pr) in enumerate(free):
            if not lo <= pn / tn <= hi or abs(pr - tr) > rate_tol:
                continue
            d = abs(math.log(pn / tn)) + abs(pr - tr)
            if best_d is None or d < best_d:
                best, best_d = k, d
        if best is not None:
            pairs.append((tid, free.pop(best)[0]))
    return pairs


#: C9's eight development seeds (the G3 block's dev range, 2.12)
DEV_SEEDS = tuple(range(9000, 9008))


def match_feasible(pair_counts, grown_counts, min_seeds=6):
    """C9's feasibility rule, checked on dev seeds BEFORE confirmation: k >= 1 matched pair on EVERY
    seed, and pairs covering at least half of the grown units on at least `min_seeds` seeds (6 of 8).
    Both arguments are {seed: count} and must name exactly DEV_SEEDS: an omitted (failed) run is NOT
    feasible, never a silent skip (Codex cp5 #8)."""
    if set(pair_counts) != set(DEV_SEEDS) or set(grown_counts) != set(DEV_SEEDS):
        return False
    if any(k < 1 for k in pair_counts.values()):
        return False
    covered = sum(1 for s in DEV_SEEDS if 2 * pair_counts[s] >= grown_counts[s])
    return covered >= min_seeds


@dataclass
class PruneResult:
    keep: list  # survivor columns, or None when nothing was pruned (identity unchanged)
    n_drop: int
    tr_idx: torch.Tensor = None  # the fitted rows the caller re-solves its readout on


def select_unearned(base, costs, tol=0.0):
    """The ONE unearned-unit rule (P2), pure: `costs[j]` is the cost (LOWER is better) with unit j removed, `base`
    the cost with every unit. A unit is dropped when its removal is no worse than `base + tol` (<=); when ALL
    are droppable exactly the unit whose removal costs most survives (ties -> the lowest index; L3), never an
    empty set. An infeasible removal is passed as +inf: never droppable, never the survivor. Fewer than two
    units: nothing to prune. Returns (drop, survivor): drop excludes the survivor, survivor is None unless the
    all-droppable case fired."""
    # an inf threshold (even from finite operands: 1e308 + 1e308) would make a +inf removal droppable
    if not (math.isfinite(base) and math.isfinite(tol) and math.isfinite(base + tol)):
        raise ValueError(
            f"select_unearned: base, tol and base + tol must be finite, got {base!r}, {tol!r}"
        )
    n = len(costs)
    if n < 2:
        return [], None
    drop = [j for j, c in enumerate(costs) if c <= base + tol]
    if len(drop) < n:
        return drop, None
    survivor = max(range(n), key=lambda j: (costs[j], -j))
    return [j for j in drop if j != survivor], survivor


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
    costs = [_val_rmse([c for c in allc if c != j]) if L > 1 else float("inf") for j in allc]
    drop, survivor = select_unearned(base, costs, tol)
    if not drop and survivor is None:
        return PruneResult(None, 0, tr_idx)
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
