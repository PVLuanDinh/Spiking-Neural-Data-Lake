"""L1b-STALE (T121, red family LS): does a stale readout carry L1b's Oja cost? Instrument module.

Spec: specs/022-unified-brain-arc/evidence/d1-readout-staleness.md (section 2.1, 2.4, 2.6; A0b, A0e).
Measurement on scratch brains, never on a tick (dormancy_gate EXEMPT `permanent`, the prior_drift_harness shape).

This module is torch-free at import: every reading takes a model that offers `_rates`, `_solve_beta`, `decide`,
`W`, `b`, `beta`, `prior_class_`, so the brain-free reds run on a numpy fixture and the same code reads a real
NGSNN in a child process. The SHADOW is a scoring-only reading: nothing reads it back (R2).

GRADIENT_CLAIM (below): `oja_step`, `RandStep.__call__` = LOCAL (instrument-only, scratch homes);
`shadow_reading`, `calibration_reading`, drift and null-split readings = NOT_A_RULE; the re-solve
itself is the model's own DIRECT_SOLVE (`_solve_beta`).

distinct-from: plasticity/oja_pca.py -- one neuron's principal component, not a per-row norm-preserving step on an
  engine's grown input rows. growth/neurogenesis_engine.py::growth_fingerprint -- takes an engine's grown model and
  omits `_batch_beta`; `substrate_hash` covers W, b, beta and `_batch_beta`. faculties/generalizing_novelty.py::
  _spearman -- Spearman is NOT built here (secondary reading, owed to infra/_mathutils per A0c F3).
  graph-exists on the shadow / drift / angular-step / null-split vocabulary: NO LEXICAL MATCH (sites read).
reuses: instruments/paired_contrast (contrast, FPR exact and MC), instruments/rpe_charge.seq_boundaries (lazy).
"""

import zlib

import numpy as np

from snn_data_lake.instruments import paired_contrast as pc

TIE = 1e-6  # |d| at or below this counts as a tie (L1b's)
RANK_TOL = 1e-6  # numerical rank tolerance x sigma_max (section 2.4)
MU = 0.05  # L1b's Oja constant (CHOSEN there, unchanged)

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "oja_step": (
        "LOCAL",
        "L1b's per-row norm-preserving Oja step on a scratch copy of the retained model's input "
        "rows: pre-activation s_i times the percept, no error signal. Instrument-only, never live",
    ),
    "RandStep.__call__": (
        "LOCAL",
        "RAND control: the same local step with a random unit vector in place of the percept, on "
        "a scratch copy. Instrument-only, never live",
    ),
    "shadow_reading": (
        "NOT_A_RULE",
        "scores held-out rows with the model's own `_solve_beta` re-solve (its DIRECT_SOLVE) and "
        "the stale reader; reads W and b, writes nothing back",
    ),
    "calibration_reading": ("NOT_A_RULE", "a shadow reading at an open solve; scoring only"),
    "rate_drift": ("NOT_A_RULE", "descriptive norm ratio of two rate matrices; no update"),
    "readout_projector": ("NOT_A_RULE", "SVD projector onto col(beta); a reading, no update"),
    "visible_fraction": ("NOT_A_RULE", "share of a rate change inside col(beta); a reading"),
}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _like(ref, arr):
    """`arr` in `ref`'s array type (torch tensor if `ref` is one; torch imported only then)."""
    if hasattr(ref, "detach"):
        import torch

        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return np.asarray(arr, dtype=np.asarray(ref).dtype)


def _onehot(y, m):
    Y = np.zeros((len(y), m))
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def _acc(pred, y):
    return float(np.mean(_np(pred) == np.asarray(y)))


# ---- the shadow re-solve (section 2.1 item 1, A0b) -------------------------------------------------------------
def shadow_reading(model, reader_beta, x_tr, y_tr, x_te, y_te):
    """Score the held-out rows twice on the model's CURRENT W: with `reader_beta` (the gate's reader as it
    stands) and with the shadow `_solve_beta` on this split's training rows, encoded as `fit` encodes them
    (one-hot, width max label + 1). Returns a dict; reads W, b only; draws no RNG; restores the model's
    `last_abstentions_` / `last_unanswerable_` that `decide` writes. gap = acc_shadow - acc_reader.
    Rows whose label the stale reader cannot output (`unseen_by_reader`) are masked out of `gap_ans`;
    `gap_all` keeps them (A0b item 2)."""
    saved = (model.last_abstentions_, model.last_unanswerable_)
    y_tr, y_te = np.asarray(y_tr, dtype=int), np.asarray(y_te, dtype=int)
    S_tr = model._rates(x_tr, model.W, model.b)
    beta_sh = model._solve_beta(S_tr, _like(S_tr, _onehot(y_tr, int(y_tr.max()) + 1)))
    S_te = model._rates(x_te, model.W, model.b)
    pa = _np(model.decide(S_te, reader_beta))
    pb = _np(model.decide(S_te, beta_sh))
    model.last_abstentions_, model.last_unanswerable_ = saved
    unseen = y_te >= int(reader_beta.shape[1])
    ok = ~unseen
    acc_a, acc_b = float(np.mean(pa == y_te)), float(np.mean(pb == y_te))
    gap_ans = (
        float(np.mean(pb[ok] == y_te[ok]) - np.mean(pa[ok] == y_te[ok]))
        if ok.any()
        else float("nan")
    )
    return {
        "acc_reader": acc_a,
        "acc_shadow": acc_b,
        "gap_all": acc_b - acc_a,
        "gap_ans": gap_ans,
        "unseen_by_reader": float(unseen.mean()),
        "beta_shadow": beta_sh,
        "S_te": S_te,
        "flip": float(np.mean(pa != pb)),
    }


def calibration_reading(model, x_tr, y_tr, x_te, y_te, pruned):
    """At an open refit (A0b item 1a): the shadow against `_batch_beta` (the reader this solve just produced) on
    the refit's own split, same W. Nonzero when the solve pruned (prune re-solves beta on its internal split)."""
    r = shadow_reading(model, model.beta, x_tr, y_tr, x_te, y_te)
    return {"gap_all": r["gap_all"], "gap_ans": r["gap_ans"], "pruned": bool(pruned)}


def calibrated(raw_gap, calib_gap):
    """gap_pre minus the calibration of the open solve that produced the reader in use (A0b 1b, P1 reads this)."""
    return raw_gap - calib_gap


def reader_for(model, batch_beta):
    """The closed branch's reader fallback, mirrored exactly (neurogenesis_engine `_refit`)."""
    beta = batch_beta if batch_beta is not None else model.beta
    return model.beta if int(beta.shape[0]) != int(model.W.shape[0]) else beta


def substrate_hash(model, batch_beta, trajectory=()):
    """crc32 over W, b, beta, `_batch_beta` bytes, the two abstention counters (A0b) and the
    trajectory: the substrate the instrument must not move."""
    acc = 0
    for t in (model.W, model.b, model.beta, batch_beta):
        acc = zlib.crc32(b"-" if t is None else np.ascontiguousarray(_np(t)).tobytes(), acc)
    counters = [getattr(model, k, None) for k in ("last_abstentions_", "last_unanswerable_")]
    acc = zlib.crc32(repr(counters).encode(), acc)
    return zlib.crc32(repr(list(trajectory)).encode(), acc)


# ---- the engine wrapper: INSTANCE-level, with its own fault counter (A0b item 3) -------------------------------
def instrument_engine(engine, as_input=None, split_fn=None, shadow=True):
    """Wrap `engine._refit` on THIS instance (a class-level patch is refused: L1b's leaked into the gate).
    Before `_refit` it takes the PRE-REFIT reading on the split `_refit` is about to use; after it, if an open
    solve replaced the model, a CALIBRATION reading. `shadow=False` runs the identical wrapper with no reading
    (R2's off arm). Exceptions in the instrument are counted, never raised into the brain; the original
    `_refit` runs exactly as it would have; every row asserts the trajectory advanced as `_refit` alone makes it.
    Binds only _refit, _model, _batch_beta, trajectory, _buf_x, _buf_y (T111a-safe). Returns the log dict."""
    log = {"rows": [], "faults": 0, "fault_msgs": [], "refit_raised": 0}
    orig = engine._refit

    def _inputs():
        global_split = split_fn
        if global_split is None:
            from snn_data_lake.growth.ng_snn import marginal_holdout as global_split
        conv = as_input
        if conv is None:
            import torch

            def conv(a):
                return torch.tensor(a, dtype=torch.float32)

        n = len(engine._buf_x)
        holdout = min(60, n // 3)
        n_cat = len(set(engine._buf_y))
        if holdout < 3 or n_cat < 2 or (n - holdout) < n_cat:
            return None
        tr, va = global_split(engine._buf_y, holdout)
        if len(va) < 3 or len(tr) < n_cat:
            return None
        X, Y = np.asarray(engine._buf_x, dtype=np.float32), np.asarray(engine._buf_y, dtype=int)
        return conv(X[tr]), Y[tr], conv(X[va]), Y[va]

    def wrapped():
        n0, before = len(engine.trajectory), engine._model
        sp, rec = None, {"tick": engine.ticks}
        if shadow:
            try:
                sp = _inputs()
                if sp is not None and before is not None:
                    r = shadow_reading(before, reader_for(before, engine._batch_beta), *sp)
                    rec.update({k: v for k, v in r.items() if k not in ("beta_shadow", "S_te")})
            except Exception as e:  # noqa: BLE001 -- counted, then the brain runs on unharmed
                log["faults"] += 1
                log["fault_msgs"].append(repr(e))
        try:
            orig()
        except Exception as e:  # the brain's own raise: counted apart, then re-raised
            log["refit_raised"] += 1
            log["fault_msgs"].append(f"_refit raised {e!r}")
            raise
        advanced = len(engine.trajectory) - n0
        rec["gate"] = (
            "closed" if (advanced and engine._model is before and before is not None) else "open"
        )
        if advanced not in (0, 1) or (sp is not None and advanced != 1):
            log["faults"] += 1
            log["fault_msgs"].append(f"trajectory advanced {advanced}")
        if shadow and sp is not None and advanced and engine._model is not before:
            try:
                rec["calibration"] = calibration_reading(
                    engine._model, *sp, pruned=bool(getattr(engine, "_pruned", 0))
                )
            except Exception as e:  # noqa: BLE001
                log["faults"] += 1
                log["fault_msgs"].append(repr(e))
        if advanced:
            log["rows"].append(rec)

    engine._refit = wrapped
    return log


# ---- the operators (section 2.1 item 2, 2.2): instrument-local, LOCAL, never on a live path -------------------
def _unit_rows(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return n, np.divide(W, n, out=np.zeros_like(W), where=n > 0)


def oja_step(W, s, xn, mu=MU):
    """L1b's step on the retained model's input rows, ungated (g=1), post_i = s_i:
    u_i <- normalize(u_i + mu*post_i*xn), W_i <- ||W_i|| * u_i. Returns the new W (W is untouched).
    A row whose mu*post_i is exactly zero is returned bit-unchanged, so mu = 0 reads Sum|dW| = 0."""
    W = np.asarray(W, dtype=float)
    n, u = _unit_rows(W)
    step = mu * np.asarray(s, dtype=float)[:, None]
    v = u + step * np.asarray(xn, dtype=float)[None, :]
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(step == 0.0, W, n * np.divide(v, vn, out=np.zeros_like(v), where=vn > 0))


class RandStep:
    """RAND: the same step with xn replaced by a fresh random unit vector per percept. Its own generator, seeded
    by `zlib.crc32` of (seed, arm), never the global one (A0b determinism)."""

    def __init__(self, seed, arm="RAND", mu=MU):
        self.rng, self.mu = np.random.default_rng(zlib.crc32(f"{seed}|{arm}".encode())), mu

    def __call__(self, W, s, d=None):
        d = np.asarray(W).shape[1] if d is None else d
        r = self.rng.standard_normal(d)
        return oja_step(W, s, r / np.linalg.norm(r), self.mu)


def operator_check(W0, W1, tol=1e-6):
    """R3: (norms preserved per row to tol, Sum|dW|, inert). INERT when Sum|dW| is exactly 0 (mu = 0
    returns W unchanged, section 2.6), whatever the scale of W."""
    W0, W1 = np.asarray(W0, dtype=float), np.asarray(W1, dtype=float)
    ok = bool(np.all(np.abs(np.linalg.norm(W0, axis=1) - np.linalg.norm(W1, axis=1)) <= tol))
    moved = float(np.abs(W1 - W0).sum())
    return ok, moved, moved == 0.0


def rotate_pairs(W, degrees):
    """The planted rotation (R1, R4): rotate coordinate pairs (0,1), (2,3), ... of every row by `degrees`; norm kept."""
    W, a = np.array(W, dtype=float), np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    for i in range(0, W.shape[1] - 1, 2):
        x, y = W[:, i].copy(), W[:, i + 1].copy()
        W[:, i], W[:, i + 1] = c * x - s * y, s * x + c * y
    return W


# ---- drift readings (section 2.4; descriptive) ------------------------------------------------------------------
def survivor_pairs(n_prev, keep, n_cur):
    """(prev_idx, cur_idx) of units present at both rows. `fit(init_W=...)` keeps inherited units as the LEADING
    rows and `prune_unearned` keeps `keep` (indices into the fitted rows), so current unit j is fit row keep[j],
    inherited iff keep[j] < n_prev. `keep=None`: identity over the leading min(n_prev, n_cur)."""
    if keep is None:
        k = min(n_prev, n_cur)
        return np.arange(k), np.arange(k)
    cur = np.array([j for j, k in enumerate(keep) if k < n_prev], dtype=int)
    return np.array([keep[j] for j in cur], dtype=int), cur


def rate_drift(S_prev, S_cur, prev_idx, cur_idx):
    """||S_r - S_{r-1}||_F / ||S_{r-1}||_F over surviving units on the frozen probe set. nan on a zero reference."""
    a, b = _np(S_prev)[:, prev_idx], _np(S_cur)[:, cur_idx]
    den = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / den) if den > 0 else float("nan")


def readout_projector(beta, tol=RANK_TOL):
    """(P, null_dim): the projector onto col(beta) in R^L and L - numerical rank (tol x sigma_max)."""
    U, s, _ = np.linalg.svd(_np(beta), full_matrices=False)
    r = int((s > tol * s.max()).sum()) if s.size and s.max() > 0 else 0
    return U[:, :r] @ U[:, :r].T, int(beta.shape[0]) - r


def visible_fraction(dS, P):
    """||dS P||_F^2 / ||dS||_F^2: the share of a hidden-rate change the readout can see. Defined (meaningful) only
    where null_dim > 0; the caller prints NOT READ otherwise."""
    dS = _np(dS)
    den = float((dS**2).sum())
    return float(((dS @ P) ** 2).sum() / den) if den > 0 else float("nan")


# ---- statistic (section 2.5): reuse, no local copy -------------------------------------------------------------
def p1_diffs(gap_oja, gap_frozen):
    """Per-seed d1 = GAP_OJA - GAP_FROZEN (same seeds, same order)."""
    return [a - b for a, b in zip(gap_oja, gap_frozen, strict=True)]


def look_reading(diffs, look, looks, row="P1", alpha=0.05):
    """One look: paired t against `rpe_charge.seq_boundaries(looks)[look]['t']`, with the FPR printed beside it
    (exact to n=40, labelled MC past it). Ties at |d| <= TIE are zeros. Needs torch via pool_selection: runs in the gate."""
    from snn_data_lake.instruments.rpe_charge import seq_boundaries

    ds = [0.0 if abs(d) <= TIE else float(d) for d in diffs]
    bound = seq_boundaries(tuple(looks), alpha)[looks.index(len(ds))]["t"]
    t = pc.paired_t(ds)
    if len(ds) <= pc.MITM_MAX_N:
        fpr_txt = f"exact {pc.permutation_fpr(ds, bound):.4f}"
    else:
        fpr_txt = pc.mc_label(*pc.permutation_fpr_mc(ds, bound, row=row, look=look))
    return {"t": t, "bound": bound, "crossed": t > bound, "fpr": fpr_txt}


def excluded_report(arm, seed, excluded_fraction, calibrations_pruned):
    """A0e m1: per arm and seed, the excluded-row fraction and the count of calibrations taken at a pruned solve."""
    return f"{arm} seed {seed}: excluded={excluded_fraction:.3f} calibrations_pruned={calibrations_pruned}"


def main():
    """Self-check (no brain, numpy only): R1 planted rotation reads gap > 0; the operators preserve norms."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 6))
    ok, moved, inert = operator_check(W, oja_step(W, np.ones(5), np.eye(6)[0]))
    assert ok and moved > 0 and not inert
    ok2, moved2, inert2 = operator_check(W, oja_step(W, np.ones(5), np.eye(6)[0], mu=0.0))
    assert ok2 and inert2 and moved2 == 0.0
    assert np.allclose(np.linalg.norm(rotate_pairs(W, 30), axis=1), np.linalg.norm(W, axis=1))
    print("readout_staleness selfcheck OK")


if __name__ == "__main__":
    main()
