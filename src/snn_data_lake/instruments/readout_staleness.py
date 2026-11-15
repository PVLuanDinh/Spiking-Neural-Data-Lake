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

import json
import multiprocessing
import os
import sys
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
        from snn_data_lake.growth.ng_snn import refit_holdout

        global_split = split_fn
        if global_split is None:
            from snn_data_lake.growth.ng_snn import marginal_holdout as global_split
        conv = as_input
        if conv is None:
            import torch

            def conv(a):
                return torch.tensor(a, dtype=torch.float32)

        n = len(engine._buf_x)
        holdout = refit_holdout(n)
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


# ---- D1 driver (pre-registration 2.1 item 5; built 2026-11-15 by FCC/GLM 5.3, integrated by the orchestrator) ----
# ---- D1 driver (pre-registration items 2.6 and 5) -------------------------------------------------------------
# INTEGRATOR: move these four imports to the module top (they must not stay in this block): json, multiprocessing, os, sys

D1_ARMS = ("FROZEN", "FROZEN_B", "OJA", "RAND")
D1_TICKS = 800


def wire_arm(engine, arm, seed):
    """Item 5: install the arm's operator on THIS engine instance (never a class patch). FROZEN and
    FROZEN_B install nothing. Counts operator calls and Sum|dW|; a fault is counted, never raised."""
    stats = {
        "arm": arm,
        "seed": int(seed),
        "calls": 0,
        "sum_dw": 0.0,
        "skipped": 0,
        "faults": 0,
        "fault_msgs": [],
    }
    if arm not in ("OJA", "RAND"):
        return stats
    import torch

    rand = RandStep(seed, arm)  # one RandStep instance per arm, created once
    orig = engine.observe

    def observe_wrapped(features, category):
        ret = orig(features, category)
        try:
            m = getattr(engine, "_model", None)
            if m is None or getattr(m, "W", None) is None:
                stats["skipped"] += 1
                return ret
            x = np.asarray(features, dtype=float).ravel()
            nx = float(np.linalg.norm(x))
            xn = x / nx if nx > 0.0 else np.zeros_like(x)
            X = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
            s = _np(m._rates(X, m.W, m.b)).ravel()  # rates on THIS percept, BEFORE the update
            W0 = _np(m.W)
            W1 = oja_step(W0, s, xn) if arm == "OJA" else rand(W0, s)
            m.W = _like(m.W, W1)
            stats["calls"] += 1
            stats["sum_dw"] += float(np.abs(W1 - W0).sum())
        except Exception as exc:  # counted, never raised into the brain
            stats["faults"] += 1
            if len(stats["fault_msgs"]) < 20:
                stats["fault_msgs"].append("{}: {}".format(type(exc).__name__, exc))
        return ret

    engine.observe = observe_wrapped
    return stats


def _leak_beta(eng):
    """R2 red 'LEAK': after each refit the shadow beta is copied into the engine's batch beta."""
    beta = getattr(getattr(eng, "_model", None), "beta", None)
    if beta is None:
        return
    if hasattr(getattr(eng, "_batch_beta", None), "copy_"):
        eng._batch_beta.copy_(beta)
    else:
        eng._batch_beta = beta


def _ulp_once(eng, state):
    """R8 red 'ULP': one-ulp bump of the first bias entry, once, after the first refit that
    produced a model."""
    if state.get("done") or getattr(eng, "_model", None) is None:
        return
    b = getattr(eng._model, "b", None)
    if b is None:
        return
    import torch

    with torch.no_grad():
        flat = b.view(-1)
        flat[0] = torch.nextafter(flat[0].detach().clone(), float("inf"))
    state["done"] = True


def _plant_once(eng, log, state):
    """Positive control 'PLANTED': at the first refit whose gate row is 'closed', rotate the
    retained model's W by rotate_pairs(..., 30) once, so every later scoring sees the rotated
    substrate and the seed's mean gap_all must exceed FROZEN's."""
    if state.get("done"):
        return
    rows = log.get("rows") or []
    if not rows or rows[-1].get("gate") != "closed":
        return
    W = getattr(getattr(eng, "_model", None), "W", None)
    if W is None:
        return
    eng._model.W = _like(W, rotate_pairs(_np(W), 30))
    state["done"] = True


def _wire_red(eng, arm, log):
    """Red-arm dispatch (LEAK / ULP / PLANTED): wraps eng._refit on the instance AFTER
    instrument_engine, so each wrap sees the instrumented refit. Faults counted, not raised."""
    state = {"arm": arm, "faults": 0}
    if arm not in ("LEAK", "ULP", "PLANTED"):
        return state
    orig = eng._refit

    def refit_wrapped(*a, **k):
        ret = orig(*a, **k)
        try:
            if arm == "LEAK":
                _leak_beta(eng)
            elif arm == "ULP":
                _ulp_once(eng, state)
            elif arm == "PLANTED":
                _plant_once(eng, log, state)
        except Exception as exc:  # counted, never raised into the brain
            state["faults"] += 1
            state.setdefault("msgs", []).append("{}: {}".format(type(exc).__name__, exc))
        return ret

    eng._refit = refit_wrapped
    return state


def _traj_len(eng):
    """Length of the engine's trajectory, or None if it is not countable."""
    try:
        return len(getattr(eng, "trajectory", ()) or ())
    except TypeError:
        return None


def _wire_row_hashes(eng):
    """Fix 2: per-refit substrate hash of the ENGINE's model, batch beta and trajectory, wrapped
    on the instance AFTER instrument_engine (outermost, so red-wrap effects are hashed too);
    appended after each refit that advanced the trajectory."""
    row_hashes = []
    orig = eng._refit

    def refit_wrapped(*a, **k):
        n0 = _traj_len(eng)
        ret = orig(*a, **k)
        if _traj_len(eng) != n0 or n0 is None:  # advanced, or not countable: record anyway
            row_hashes.append(int(substrate_hash(eng._model, eng._batch_beta, eng.trajectory)))
        return ret

    eng._refit = refit_wrapped
    return row_hashes


def arm_cell(args):
    """One arm in a worker process (g5 pattern): fresh home, make_brain(persist=False),
    instrumented engine, arm dispatch (real operators + red-arm wraps), D1_TICKS ticks."""
    seed, arm = args
    import torch

    torch.set_num_threads(1)
    from snn_data_lake.instruments.g5_readout_cells import check_fresh_home
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home() as fresh:
        check_fresh_home(fresh)
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=seed + (1000 if arm == "OTHER_STREAM" else 0), persist=False)
        try:
            eng = b.neuro
            log = instrument_engine(eng, shadow=(arm != "SHADOW_OFF"))
            ops = wire_arm(eng, arm, seed)
            red = _wire_red(eng, arm, log)
            row_hashes = _wire_row_hashes(eng)
            for _ in range(D1_TICKS):
                b.tick()
            out = {
                "seed": int(seed),
                "arm": arm,
                "rows": log.get("rows", []),
                "faults": int(log.get("faults", 0)) + int(red["faults"]),
                "ops": ops,
                "row_hashes": row_hashes,
                "floors": [(float(m), float(p)) for _, m, p in getattr(eng, "baselines", [])],
            }
            if (
                getattr(eng, "_model", None) is not None
                and getattr(eng._model, "W", None) is not None
            ):
                out["hash"] = int(substrate_hash(eng._model, eng._batch_beta, eng.trajectory))
            return out
        finally:
            b.close()


def seed_summary(rows):
    """Last-third endpoints for one arm: GAP, ACC_SH, ACC (shadow_reading's keys acc_shadow /
    acc_reader / gap_all; missing keys are named, not guessed). Floors are NOT reported here."""
    rows = list(rows)
    k = max(1, len(rows) // 3) if rows else 0
    win = rows[-k:] if k else []
    out = {
        "n_rows": len(rows),
        "n_window": len(win),
        "GAP": None,
        "ACC_SH": None,
        "ACC": None,
        "missing_keys": [],
    }
    if not win:
        return out
    need = ("acc_shadow", "acc_reader")
    out["missing_keys"] = sorted({key for r in win for key in need if key not in r})
    gaps, ash, acc = [], [], []
    for r in win:
        a_sh, a_r = float(r.get("acc_shadow", np.nan)), float(r.get("acc_reader", np.nan))
        g = r.get("gap_all")
        gaps.append(float(g) if g is not None else a_sh - a_r)
        ash.append(a_sh)
        acc.append(a_r)
    out.update(GAP=float(np.mean(gaps)), ACC_SH=float(np.mean(ash)), ACC=float(np.mean(acc)))
    return out


def _num_leaves(x, out):
    """Every recorded number in x, in traversal order (dicts/lists flattened, arrays ravelled)."""
    if isinstance(x, dict):
        for v in x.values():
            _num_leaves(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _num_leaves(v, out)
    elif isinstance(x, np.ndarray):
        out.extend(float(v) for v in np.asarray(x).ravel())
    elif hasattr(x, "detach"):
        out.extend(float(v) for v in _np(x).ravel())
    elif isinstance(x, (bool, np.bool_)):
        out.append(float(bool(x)))
    elif isinstance(x, (int, float, np.integer, np.floating)):
        out.append(float(x))


def _rows_match(rows_a, rows_b, tol=1e-6):
    """Row-for-row numeric equality to tol (the rows half of the A/A check)."""
    if len(rows_a) != len(rows_b):
        return False
    for ra, rb in zip(rows_a, rows_b):
        la, lb = [], []
        _num_leaves(ra, la)
        _num_leaves(rb, lb)
        if len(la) != len(lb) or any(abs(p - q) > tol for p, q in zip(la, lb)):
            return False
    return True


def _aa_pass(a, b):
    """The A/A check (R8 green): rows match to 1e-6 AND per-refit row_hashes match EXACTLY
    (so a ulp-only change, which shows in the hash, fails the check)."""
    return _rows_match(a["rows"], b["rows"]) and a["row_hashes"] == b["row_hashes"]


def _floors_match(fa, fb, tol=1e-6):
    """Same-rows rule: per-refit (majority_floor, persistence_floor) lists identical to tol;
    lengths must match exactly."""
    if len(fa) != len(fb):
        return False
    return all(abs(m1 - m2) <= tol and abs(p1 - p2) <= tol for (m1, p1), (m2, p2) in zip(fa, fb))


def _mean_gap_all(rows):
    """PLANTED control endpoint: mean gap_all over ALL rows (acc_shadow - acc_reader fallback;
    rows missing both are skipped)."""
    vals = []
    for r in rows:
        g = r.get("gap_all")
        if g is None:
            a_sh, a_r = r.get("acc_shadow"), r.get("acc_reader")
            if a_sh is None or a_r is None:
                continue
            g = float(a_sh) - float(a_r)
        vals.append(float(g))
    return float(np.mean(vals)) if vals else None


def run_seed(seed, pool):
    """One seed: the four real arms, each its own child process; then the A/A check, the
    same-rows floors rule, and d1 = p1_diffs([GAP_OJA], [GAP_FROZEN])[0] for THIS seed."""
    by = {r["arm"]: r for r in pool.map(arm_cell, [(seed, a) for a in D1_ARMS])}
    summ = {a: seed_summary(by[a]["rows"]) for a in D1_ARMS}
    aa_fail = not _aa_pass(by["FROZEN"], by["FROZEN_B"])
    excluded = any(not _floors_match(by["FROZEN"]["floors"], by[a]["floors"]) for a in D1_ARMS[1:])
    d1 = None
    if summ["OJA"]["GAP"] is not None and summ["FROZEN"]["GAP"] is not None:
        d1 = p1_diffs([summ["OJA"]["GAP"]], [summ["FROZEN"]["GAP"]])[0]
    arms = {
        a: {
            "summary": summ[a],
            "ops": by[a]["ops"],
            "faults": by[a]["faults"],
            "hash": by[a].get("hash"),
            "floors": by[a]["floors"],
        }
        for a in D1_ARMS
    }
    return {
        "seed": int(seed),
        "arms": arms,
        "aa_fail": bool(aa_fail),
        "excluded": bool(excluded),
        "d1": d1,
    }


def stage(seeds, out_json):
    """Pilot: the four real arms over the seeds on one spawn pool, one JSON. Refuses to
    overwrite. Measures only; no confirmatory statistic or boundary."""
    if os.path.exists(out_json):
        raise FileExistsError("refusing to overwrite existing {}".format(out_json))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(D1_ARMS)) as pool:
        results = [run_seed(s, pool) for s in seeds]
    doc = {
        "seeds": [int(s) for s in seeds],
        "results": results,
        "d1": [r["d1"] for r in results],
        "aa_fail_seeds": [r["seed"] for r in results if r["aa_fail"]],
        "excluded_seeds": [r["seed"] for r in results if r["excluded"]],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=float)
    return doc


def stage0(out_json):
    """Pre-registration 2.6: seed 0. Four real arms plus the red arms (SHADOW_OFF, LEAK,
    OTHER_STREAM, ULP, PLANTED), each its own child process; reports each red MET/NOT MET
    and writes stage0_ok (all reds MET and the planted control MET)."""
    if os.path.exists(out_json):
        raise FileExistsError("refusing to overwrite existing {}".format(out_json))
    arms0 = D1_ARMS + ("SHADOW_OFF", "LEAK", "OTHER_STREAM", "ULP", "PLANTED")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(4, len(arms0))) as pool:  # RAM: at most 4 brains at once
        by = {r["arm"]: r for r in pool.map(arm_cell, [(0, a) for a in arms0])}
    fh = by["FROZEN"]["row_hashes"]
    greens = all(_floors_match(by["FROZEN"]["floors"], by[a]["floors"]) for a in D1_ARMS[1:])
    reds = {
        "R2_SHADOW_OFF": by["SHADOW_OFF"]["row_hashes"] == fh,
        "R2_LEAK": by["LEAK"]["row_hashes"] != fh,
        "R7_OTHER_STREAM": greens
        and not _floors_match(by["FROZEN"]["floors"], by["OTHER_STREAM"]["floors"]),
        "R8_ULP": (not _aa_pass(by["FROZEN"], by["ULP"]))
        and _aa_pass(by["FROZEN"], by["FROZEN_B"]),
    }
    pg, fg = _mean_gap_all(by["PLANTED"]["rows"]), _mean_gap_all(by["FROZEN"]["rows"])
    planted = pg is not None and fg is not None and pg > fg
    reds["PLANTED"] = bool(planted)
    for name, met in sorted(reds.items()):
        print("{}: {}".format(name, "MET" if met else "NOT MET"))
    if not planted:
        print("INSTRUMENT BLIND")
    ok = all(reds.values())
    doc = {
        "seed": 0,
        "arms": {
            a: {
                "summary": seed_summary(by[a]["rows"]),
                "ops": by[a]["ops"],
                "faults": by[a]["faults"],
                "hash": by[a].get("hash"),
                "floors": by[a]["floors"],
            }
            for a in D1_ARMS
        },
        "reds": {k: bool(v) for k, v in reds.items()},
        "planted_control": {"planted_gap": pg, "frozen_gap": fg, "met": bool(planted)},
        "stage0_ok": bool(ok),
        "verdict": "OK" if ok else ("BLIND" if not planted else "REDS_FAILED"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=float)
    return doc


def stage_cli(argv=None):
    """stage0 -> reds on seed 0 to tools/d1_stage0.json; pilot -> the four real arms on seeds
    0-7 to tools/d1_pilot.json. Returns None on a bare/unknown call: the integrator's single
    module guard calls this before main()."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "stage0":
        stage0(os.path.join("tools", "d1_stage0.json"))
        return 0
    if argv and argv[0] == "pilot":
        stage(list(range(8)), os.path.join("tools", "d1_pilot.json"))
        return 0
    return None


if __name__ == "__main__":
    if stage_cli() is None:
        main()
