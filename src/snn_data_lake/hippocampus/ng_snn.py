"""
src/snn_data_lake/hippocampus/ng_snn.py -- NG-SNN: a neurogenesis-inspired, contribution-grown
SPIKING classifier, promoted into the core (torch/GPU).

Anatomy. Adult NEUROGENESIS in mammals is concentrated in the HIPPOCAMPAL DENTATE GYRUS, where new
granule cells are continuously born and *integrated by activity / functional contribution* to
sharpen PATTERN SEPARATION. NG-SNN mirrors exactly that: it GROWS a single LIF hidden layer one (or
a block of) neuron(s) at a time, admitting each new "granule cell" only if it functionally
CONTRIBUTES to reducing the classification residual (contribution criterion, Eq. 6-7), then reads
the pool out in ONE analytical shot (activity-regularized ridge, Eq. 9-12) -- no backprop. It is the
spiking, contribution-driven, single-shot successor to the rate/novelty prototype learner in
`hippocampus/growing_learner.py`, which recruits on INPUT novelty and ignores correctness.

TORCH TIER. This core module imports torch and runs in the GPU/training tier (numpy/torch are allowed
project-wide). It promotes the already-validated research
NG-SNN (research/ng_snn/model.py -- UCI Iris ~95.6% / Wine ~97.2%, MNIST 99.14% / N-MNIST 98.84% at
scale) so the contribution-grown spiking classifier is a first-class brain-region module. The math
is UNCHANGED from that validated model.

PORTED MEASUREMENT (2026-08-06): on the 3-conv SoftHebb mean+max N-MNIST features, the LINEAR
readout ceiling is 99.42% against the 98.84% random-LIF head quoted above. That pair is what makes
either number readable -- 98.84% is only interesting as a distance from the ceiling a plain linear
readout on the same features already reaches. The ceiling was carried as `LINEAR_CEIL = 99.42` in
`reasoning/ng_snn/grow_head_nmnist.py`, a driver script RETIRED under P2 because this module carries
the mechanism it exercised (`NGSNN.fit(..., delta=1, score_subsample=None)`); the script held no
mechanism, but it did hold this number, and it was the ONLY site in the governed tree that did.
Recorded here as prose rather than as a constant deliberately: nothing consumes it, so a module-level
`LINEAR_CEIL` would be a dormant symbol (P3) where a measurement is what actually needed keeping.

Method (faithful to Tang, Li, Zhang, Zeng, "NG-SNN", Neural Networks 199 (2026) 108656):
  * LIF hidden neuron, discrete Euler (Eq. 2): V[t]=(1-lam)V[t-1]+lam(W x[t]+b), lam=dt/tau_m;
    spike o=1 at V>=v_th, hard reset V<-0 (Eq. 3-4); response s=mean_t o (Eq. 5). In feature mode a
    neuron is driven by the CONSTANT current A=W f over T steps (a random-feature expansion).
  * Contribution score C_q=<e_q,s>^2/(||s||^2 ||e_q||^2) (Eq. 6); consistency gate C_q>=1-r-eta_L
    for all output dims q (Eq. 7); grow the candidate that maximally reduces the residual (greedy
    matching pursuit over a multi-scale candidate pool, Alg. 1).
  * Analytical output weights (Eq. 9-12): mu_j=mean_n s_nj, gamma_j=c1 exp(-k mu_j)+c0,
    beta=(S^T S + diag(gamma))^-1 S^T Y  -- ONE torch.linalg.solve, zero iterations.
  * Grow until train RMSE < eps_tol or l_max (Alg. 3). Block neurogenesis (`delta`>1): admit the
    TOP-`delta` candidates per step with ONE ridge solve per block (acceleration). `score_subsample`:
    candidate SELECTION may score on a row subsample; the accepted neurons' S columns and the final
    solve always use FULL X/Y.

Deterministic via torch.Generator(seed). Runs in the torch tier (.venv-cu128); the
stdlib core never imports this.

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.hippocampus.ng_snn
"""

import torch

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "NGSNN.fit": (
        "DIRECT_SOLVE",
        "greedy matching pursuit with an analytical beta per block -- a closed-form batch solve over "
        "the whole buffer each sleep tick. docs/POLICIES.md adjudicates it: NOT a violation, because "
        "no gradient is propagated through a network and no weight is transported, so there is "
        "nothing for the cosine clause to measure. It IS NON-LOCAL, and that same adjudication says "
        "the policy should record this DISTINCTLY rather than lumping it in -- 'non-local' and "
        "'backprop' are different claims and only the second is forbidden. LOCAL would be the "
        "lumping-in it warns against; this value exists so the distinction survives in the census",
    ),
}


# ---------------------------------------------------------------------------------------------------
# INPUT ENCODERS -- ported 2026-08-06 from `reasoning/ng_snn/model.py`, which is DELETED (P2).
#
# That file was a second NGSNN of the same mechanism: same class name, same paper, same Eq. 2-5 /
# 6-7 / 9-12. A/B'd on this module's own `_make_task`, seeds 0/1/2, this module's own self-check
# config (l_max PINNED to 200 on both, T=15, t_cand=40, r=0.9, eps_tol=0.20, seed=7), .venv-lake +
# CUDA: grown size 30/30/32 on BOTH and held-out accuracy 0.9563/0.9250/0.9375 on BOTH -- IDENTICAL
# on 3 of 3. So "the math is UNCHANGED from that validated model" above is VERIFIED, not asserted,
# and the direction of the twin was checked rather than assumed (this project has already had a
# retirement verdict get a twin backwards).
#
# WHAT DOES NOT REPRODUCE, recorded because the tidy version of this claim is the tempting one. The
# adjudication that scoped this retirement reported the pinned A/B as 200/200/200 units at
# 0.9937/1.0000/0.9937, and reported the free comparison as "fewer units, higher accuracy" for this
# module. The CONCLUSION (identical mechanism) reproduces; those FIGURES do not, because they were
# taken at a different `eps_tol` -- growth stops on train RMSE, so the stopping config decides the
# unit count, and quoting a unit count without it says nothing. Re-measured at each module's OWN
# default (this one derives l_max, the twin pins 200): seed 0 28u@0.9438 vs 30u@0.9563, seed 1
# 26u@0.9187 vs 30u@0.9250, seed 2 33u@0.9500 vs 32u@0.9375. Honest reading: this module is
# consistently the SMALLER model and is ahead on accuracy in only 1 of 3 seeds, by 0.0125 -- a
# 2-4 unit / ~1 point trade on a 520-row synthetic task, which is not evidence that either is more
# accurate. What justifies the retirement is that the two are the same mechanism and one of them
# carries strictly more (`prune_unearned`, `init_W`/`init_b` growth-state seeding, `patience`,
# `return_state`, and the derived rather than fixed capacity bound `fit()` documents at length
# below) -- not a benchmark win that the benchmark does not support.
#
# These two functions were the ONE thing the twin had that this module did not, so they are ported
# before it is deleted. They are not decoration: without them nothing in the governed tree ever
# builds a (N, T, d) spike SEQUENCE, and `_rates`'s sequence branch (the `einsum` arm) is therefore
# never executed -- both live consumers (`cortex/neurogenesis_engine.py`, `tools/capacity_curve.py`)
# pass 2-D features. Claim 5 of the self-check below drives that branch through these encoders, so
# the port arrives with a consumer instead of becoming the next dormant symbol.
#
# NOT PORTED, and named rather than left to be discovered: the twin's `fit()` opened `X = X.float()`
# where this one opens `X = X.to(self.device).float()`, i.e. it left X on the caller's device and let
# `_rates` stream it batch-by-batch. That is a VRAM-footprint difference, not a capability gap --
# both accept X on any device and return the same result -- and no caller exercised it (the twin's
# own large-scale driver did `Xtr.to(DEV)` itself before calling `fit`). Adopting it would move a
# full-X host-to-device copy from once per `fit` to once per candidate block, which is a regression
# for every caller that exists. There is no benchmark in this repo measuring an out-of-core fit, so
# per P2 the honest disposition is to record the difference, not to swap on it.
# ---------------------------------------------------------------------------------------------------
def poisson_encode(feats, T, generator=None):
    """Rate/Poisson-encode normalized features [0,1]^(N,d) into a spike sequence {0,1}^(N,T,d): each
    feature value is the per-step Bernoulli firing probability (Sec 4.1, end-to-end mode). STOCHASTIC
    -- deterministic only given a `generator`, which is why every use here passes one.

    Honest note carried over from the retired module: at small T the sampling noise is large and this
    underperforms `current_encode` on the paper's UCI reproduction. It is kept because it is the
    genuinely SPIKING encoder -- the input really is a train of binary events -- and `current_encode`
    is a deterministic rate approximation of it."""
    N, d = feats.shape
    p = feats.clamp(0, 1).unsqueeze(1).expand(N, T, d)
    if generator is None:
        return torch.bernoulli(p)
    # The generator is CPU-bound by design (see NGSNN.__init__: torch's cpu and cuda streams differ
    # at the same seed, so binding one to the device makes "same seed -> same spikes" false across
    # hardware). Draw on the GENERATOR's device, then move the result to the features'.
    return torch.bernoulli(p.to(generator.device), generator=generator).to(feats.device)


def current_encode(feats, T):
    """Deterministic rate coding: hold the normalized feature as a CONSTANT input current over T
    steps (no Poisson sampling noise), so the LIF firing rate is a clean nonlinear function of the
    feature -- a random-feature expansion the analytical ridge reads out. This is the encoding that
    reproduced the paper's UCI numbers where Poisson sampling underperformed.

    Returns the same (N, T, d) SEQUENCE shape as `poisson_encode` on purpose: it is the noise-free
    CONTROL for it, and a control that differs in shape as well as in content controls nothing.
    (Passing raw 2-D features to `fit`/`_rates` reaches the same dynamics by the feature-mode branch
    without materialising the sequence -- that is the path both live consumers take.)"""
    N, d = feats.shape
    return feats.clamp(0, 1).unsqueeze(1).expand(N, T, d).contiguous()


def marginal_holdout(y, n_val):
    """THE admission window (specs/022 L1a, evidence/l-parallel-arms.md, 2026-09-05): the held-out rows of a
    category stream, chosen PER CATEGORY in proportion to the category marginal, taking the MOST RECENT rows
    of each category -- "recency-mixed": covers the marginal, prefers recent rows within it. Returns
    (train_idx, val_idx) as ascending Python lists over range(len(y)).

    Why: until this landed every window here was the most RECENT rows (the last n_val of what fit was handed,
    the last 20 % in prune_unearned, the last third in the engine's refit). On a bursty category stream the
    recent rows are usually one run of one category, so a unit that would predict a rarer category never
    earned its place and the readout answered the majority -- measured on the brain's own stream (s2 v2): the
    persistence ORACLE reached its own floor on 2 of 8 seeds. One function serves all three windows (P2).

    Quotas: round(n_val * n_c / N) per category (Python's round -- CHOSEN), at least one row for every
    category with >= 2 rows (CHOSEN: a category with one row has nothing to hold out and still learn
    from), the remainder added to / removed from the largest categories -- and, when the one-row floors
    alone exceed n_val, the floors yield for the SMALLEST categories first -- so exactly n_val rows are
    held out whenever n_val >= 1 and N > n_val. n_val itself is the caller's window size (the engine's
    min(60, n // 3), NGSNN's max(8, int(0.2 N)) -- both pre-existing, both CHOSEN, P1 debt carried).
    Deterministic; no rng."""
    ys = [int(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    n = len(ys)
    n_val = int(min(max(0, n_val), max(0, n - 1)))
    if n_val <= 0:
        return list(range(n)), []
    idx_by_cat = {}
    for i, c in enumerate(ys):
        idx_by_cat.setdefault(c, []).append(i)
    cats = sorted(idx_by_cat, key=lambda c: (-len(idx_by_cat[c]), c))
    quota = {}
    for c in cats:
        m = len(idx_by_cat[c])
        q = int(round(n_val * m / n))
        if m >= 2:
            q = max(1, min(q, m - 1))
        else:
            q = 0
        quota[c] = q
    total = sum(quota.values())
    # settle the rounding remainder on the largest categories, never emptying a category's training rows
    k = 0
    while total != n_val and k < 10 * len(cats) + 10:
        c = cats[k % len(cats)]
        m = len(idx_by_cat[c])
        if total < n_val and quota[c] < m - 1:
            quota[c] += 1
            total += 1
        elif total > n_val and quota[c] > (1 if m >= 2 else 0):
            quota[c] -= 1
            total -= 1
        k += 1
    if (
        total > n_val
    ):  # more categories than n_val: the one-row floors yield, smallest categories first
        for c in reversed(cats):
            if total <= n_val:
                break
            if quota[c] > 0:
                quota[c] -= 1
                total -= 1
    val = []
    for c in cats:
        if quota[c]:
            val.extend(idx_by_cat[c][-quota[c] :])  # the most recent rows of this category
    val_set = set(val)
    return [i for i in range(n) if i not in val_set], sorted(val)


class NGSNN:
    """NG-SNN spike classifier: contribution-grown LIF hidden layer + one-shot analytical readout.

    fit(X, y, delta, score_subsample) grows the pool and solves the output weights; predict(X)
    returns class indices. X is either a spike/feature SEQUENCE {0,1}^(N,T,d_in) or normalized
    FEATURES [0,1]^(N,d_in) (feature mode drives each LIF with a constant current over T steps).

    Params (paper defaults): tau_m, v_th, dt (LIF); lambdas (candidate scales {0.1,1,5}); r (0.9),
    t_cand (candidates per scale per step); c1,k,c0 (activity-dependent regularizer); l_max, eps_tol
    (stop); reg_eps numerical floor; T (feature-mode integration window). Deterministic via `seed`."""

    def __init__(
        self,
        tau_m=2.0,
        v_th=1.0,
        dt=1.0,
        lambdas=(0.1, 1.0, 5.0),
        r=0.9,
        t_cand=40,
        c1=1.0,
        k=0.1,
        c0=1e-4,
        l_max=None,  # None = DERIVED from the data (see below). An int still pins it explicitly.
        eps_tol=1e-2,
        patience=48,  # blocks without held-out gain before growth stops; MEASURED, see fit()
        reg_eps=1e-6,
        T=15,
        device="cuda",
        seed=0,
    ):
        self.tau_m, self.v_th, self.dt = float(tau_m), float(v_th), float(dt)
        self.T = int(T)  # LIF integration window (feature mode)
        self.lam = self.dt / self.tau_m  # LIF leak-integrate coefficient
        self.lambdas = tuple(float(x) for x in lambdas)
        self.r, self.t_cand = float(r), int(t_cand)
        self.c1, self.k, self.c0 = float(c1), float(k), float(c0)
        # l_max=None means DERIVE it per fit from the training set (see fit()). The old default was a
        # fixed 200, which is an ARTIFICIAL CEILING on cognitive growth -- and it binds: measured on
        # 500 synthetic rows it stopped growth at exactly 200/200 on every task tried, including a
        # trivially easy clean 3-class one. A capacity ceiling that fires even on an easy problem is
        # not a safety bound, it is the growth limit deciding the answer. Policy 0 says a missing
        # avenue of self-improvement is a gap to close, not a boundary to respect.
        self.l_max = None if l_max is None else int(l_max)
        self.eps_tol, self.reg_eps = float(eps_tol), float(reg_eps)
        self.patience = int(patience)
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        # THE RNG IS CPU-BOUND ON PURPOSE, WHATEVER `self.device` IS. torch's CPU and CUDA
        # generators produce DIFFERENT streams from the same seed -- measured, seed 7:
        # cpu [0.534923, 0.198803, 0.659212, ...] vs cuda [0.954597, 0.495040, 0.941953, ...].
        # Binding the generator to `self.device` therefore made "same seed -> same brain" false
        # across hardware: this box has a card and defaults to cuda, CI has none and gets cpu, so
        # the grown weights differed between them by construction. The determinism invariant is a
        # property of the SEED, not of the machine. Every draw below is taken on the CPU stream and
        # moved to `self.device`, which costs one small host-to-device copy per candidate block and
        # buys a brain that is reproducible off this machine.
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = None  # (L, d_in) grown hidden input weights
        self.b = None  # (L,)      grown hidden biases
        self.beta = None  # (L, m)    analytical output weights
        self.d_in = None
        self.m = None
        # `decide`'s no-evidence answer: the training-majority class, recorded at fit. None until a
        # fit has run, and `decide` then falls back to the bare argmax rather than inventing a prior
        # -- an untrained model has no majority to answer with, and absence is a fact (T046a).
        self.prior_class_ = None
        self.last_abstentions_ = 0  # rows `decide` ANSWERED FROM THE PRIOR on its last call
        self.last_unanswerable_ = 0  # silent rows it could NOT answer (no prior) -- the defect

    # ---- LIF hidden-layer firing rates (Eq. 2-5), vectorized + batched over samples ----
    def _rates(self, X, W, b, batch=4096, return_state=False):
        """Average firing rate s in [0,1]^(N,L) of the LIF pool (weights W (L,d), bias b (L,)).
        Dispatched on X.ndim: X (N,T,d) spike SEQUENCE -> current I[t]=W x[t]; X (N,d) FEATURES ->
        drive the LIF with the CONSTANT current A=W f over self.T steps. Batched over N to bound
        memory. If return_state: also return the INTEGER spike-count tensor and the final (float)
        membrane potential V -- the decision reads s = counts / T, so these expose that the pool is
        genuinely spike-count-based, not a continuous-voltage readout."""
        L = W.shape[0]
        N = X.shape[0]
        feat_mode = X.ndim == 2
        T = self.T if feat_mode else X.shape[1]
        out = torch.empty(N, L, device=self.device)
        cnt_all = torch.empty(N, L, device=self.device) if return_state else None
        v_all = torch.empty(N, L, device=self.device) if return_state else None
        for s0 in range(0, N, batch):
            xb = X[s0 : s0 + batch].to(self.device)
            cur = None
            if feat_mode:
                A = xb @ W.t() + b  # (nb, L) constant input current
            else:
                cur = torch.einsum("ntd,ld->ntl", xb, W) + b  # (nb, T, L) input current
            V = torch.zeros(xb.shape[0], L, device=self.device)
            cnt = torch.zeros(xb.shape[0], L, device=self.device)
            for t in range(T):
                it = A if feat_mode else cur[:, t, :]  # this step's input current
                V = (1.0 - self.lam) * V + self.lam * it  # Euler LIF (Eq. 2), V_rest=0
                o = (V >= self.v_th).float()  # spike (Eq. 3)
                cnt += o  # integer spike count
                V = V * (1.0 - o)  # hard reset (Eq. 4)
            out[s0 : s0 + batch] = cnt / T  # response s (Eq. 5)
            if return_state:
                cnt_all[s0 : s0 + batch] = cnt
                v_all[s0 : s0 + batch] = V
        if return_state:
            return out, cnt_all, v_all
        return out

    def _solve_beta(self, S, Y):
        """Activity-dependent regularized pseudoinverse (Eq. 9-12): gamma_j=c1 exp(-k mu_j)+c0,
        beta=(S^T S + diag(gamma))^-1 S^T Y."""
        mu = S.mean(dim=0)  # (L,) average activity (Eq. 9)
        gamma = self.c1 * torch.exp(-self.k * mu) + self.c0  # (L,) activity-dependent reg
        A = S.t() @ S + torch.diag(gamma)  # (L, L)
        return torch.linalg.solve(A, S.t() @ Y)  # (L, m) closed form (Eq. 12)

    def _eta(self, L):
        """Non-negative decreasing sequence eta_L <= 1-r, ->0 (paper leaves the form open); (1-r)/(L+1)."""
        return (1.0 - self.r) / (L + 1)

    def prune_unearned(self, X, y, tol=0.0):
        """Drop units that no longer earn their place, judged on rows the readout was NOT solved on.

        This is the counterpart grow-on-top makes necessary. While every refit started from scratch
        there were no old units to prune -- the question could not even be asked. Once a refit inherits
        units, some were earned against data that no longer arrives, and without a way out the
        substrate only ever grows.

        THE CRITERION IS THE SAME ONE THAT ADMITS A UNIT: does it reduce error on rows the solve did
        not fit? A unit is dropped when removing it leaves held-out RMSE no worse than `tol`. That
        deliberately reuses `fit`'s held-out discipline rather than inventing a second standard --
        magnitude of `beta`, or activation energy, would be proxies for contribution, and this repo's
        record is that a proxy tuned on convenient data gets the answer wrong (the capacity constant
        was wrong three times before it was measured on real rows).

        NO MAGIC THRESHOLD, per P1: `tol=0.0` means "keep only what pays for itself", derived from the
        measurement rather than picked. A caller wanting hysteresis passes it explicitly.

        KNOWN CEILING, stated rather than hidden: this is a SINGLE PASS. Each unit is scored against
        the full model, so two units that are individually redundant but jointly necessary can both be
        dropped, and a unit made redundant only by another's removal is not re-examined. Iterating to a
        fixed point costs O(L^2) solves; the upgrade path is to loop this until it removes nothing.
        The final re-solve below means the returned model is always self-consistent whatever is cut.
        """
        self.last_keep = None  # set to the survivor column list only when a prune actually happens
        if self.W is None or self.beta is None or int(self.W.shape[0]) <= 1:
            return 0
        X = X.to(self.device).float()
        N = X.shape[0]
        y = y.to(self.device)
        m = int(self.beta.shape[1])
        Y = torch.zeros(N, m, device=self.device)
        Y[torch.arange(N, device=self.device), y] = 1.0
        n_val = max(8, int(0.2 * N))
        tr_list, va_list = marginal_holdout(y.tolist(), n_val)  # L1a: the ONE window
        n_tr, n_val = len(tr_list), len(va_list)
        if n_tr < 2 or n_val < 2:
            return 0  # too few rows to judge honestly -- keep everything
        tr_idx = torch.tensor(tr_list, device=self.device)
        va_idx = torch.tensor(va_list, device=self.device)

        S = self._rates(X, self.W, self.b)  # (N, L)
        L = int(S.shape[1])

        def _val_rmse(cols):
            beta = self._solve_beta(S[tr_idx][:, cols], Y[tr_idx])
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
            return 0
        if len(drop) >= L:
            # L3 (specs/022, 2026-09-05): EVERY unit is droppable -- an all-redundant set. Until this
            # landed the answer was "never prune to nothing", which kept ALL of them (24 on a constant
            # percept, s2 v2). Keep exactly ONE: the unit whose removal costs the most held-out RMSE
            # (ties -> the lowest index), so a substrate is never empty and never padded.
            survivor = max(allc, key=lambda j: (cost.get(j, float("-inf")), -j))
            drop = [c for c in allc if c != survivor]
        keep = [c for c in allc if c not in set(drop)]
        # THE SURVIVOR MASK, published so a caller can follow a unit's IDENTITY across a prune.
        # `return len(drop)` alone says how many went, never WHICH -- and anything keyed per unit
        # (a placement, a layer, a Dale sign assigned at birth) silently re-associates to a
        # different neuron the first time this fires. Additive: the return value is unchanged and
        # no existing caller is affected. `None` elsewhere means "identity unchanged".
        self.last_keep = list(keep)
        self.W = self.W[keep]
        self.b = self.b[keep]
        self.beta = self._solve_beta(S[tr_idx][:, keep], Y[tr_idx])  # re-solve: beta must match W
        return len(drop)

    def fit(self, X, y, delta=1, score_subsample=None, init_W=None, init_b=None):
        """Grow the classifier on X (spike sequence {0,1}^(N,T,d_in) or features [0,1]^(N,d_in)) with
        integer labels y (N,). Greedy matching pursuit under the contribution criterion (Alg. 1),
        analytical beta after each block (Alg. 2), until train RMSE < eps_tol or l_max (Alg. 3).

        delta>1: block neurogenesis -- admit the top-`delta` candidates per step, ONE ridge solve per
        block. score_subsample: candidate SELECTION scores on a random row subsample of that size;
        the accepted neurons' S columns always span every row. When capacity is derived (l_max=None)
        the growth solve fits the training rows of a marginal-covering split (marginal_holdout; the
        withheld ~20 % cover the category marginal) and the withheld rows decide when to stop; the
        shipped readout is then refit on all of them."""
        X = X.to(self.device).float()
        N = X.shape[0]
        d = X.shape[2] if X.ndim == 3 else X.shape[1]  # sequence (N,T,d) or features (N,d)
        self.d_in, self.m = d, int(y.max().item()) + 1
        # `decide`'s no-evidence answer. `self.m` spans 0..max(y), so it can carry columns for
        # classes this fit never saw -- the prior is drawn from the TARGETS actually present, never
        # from the width of the one-hot.
        _yl = y.tolist()
        self.prior_class_ = max(set(_yl), key=_yl.count) if _yl else None
        Y = torch.zeros(N, self.m, device=self.device)  # one-hot targets Y (N, m)
        Y[torch.arange(N, device=self.device), y.to(self.device)] = 1.0
        delta = max(1, int(delta))

        # THE HELD-OUT SPLIT, and it has to be taken before anything else touches the rows. When
        # capacity is derived rather than preset, n_val marginal-covering rows are withheld from the growth
        # solve ENTIRELY -- not merely scored separately afterwards. Scoring rows the solve already
        # fit is in-sample under another name, and greedy pursuit drives in-sample residual toward
        # zero by construction, so such a criterion can never say "stop". Measured before this was
        # split properly: it grew 600 units to fit 600 RANDOM labels and reported that as earned.
        n_val = 0 if self.l_max is not None else max(8, int(0.2 * N))
        # L1a (2026-09-05): the withheld rows COVER THE CATEGORY MARGINAL (marginal_holdout), not the
        # most recent block -- see that function for the measurement that forced it.
        if n_val:
            tr_list, va_list = marginal_holdout(y.tolist(), n_val)
            n_tr, n_val = len(tr_list), len(va_list)
            tr_idx = torch.tensor(tr_list, device=self.device)
            val_idx = torch.tensor(va_list, device=self.device)
        else:
            n_tr = N
            tr_idx = torch.arange(N, device=self.device)
            val_idx = None

        # candidate SELECTION may run on a row subsample (acceleration); the accepted neurons' S
        # columns and the growth solve always use the full set of TRAINING rows.
        use_sub = score_subsample is not None and int(score_subsample) < n_tr
        sel = (
            torch.randperm(n_tr, generator=self.g)[: int(score_subsample)].to(self.device)
            if use_sub
            else None
        )
        Xtr = X[tr_idx]
        Xsel = Xtr[sel] if use_sub else Xtr

        W = torch.empty(0, d, device=self.device)
        b = torch.empty(0, device=self.device)
        S = torch.empty(
            N, 0, device=self.device
        )  # hidden activity over ALL rows: the withheld ones
        beta = torch.empty(0, self.m, device=self.device)  # still need columns to be scored against
        E = Y[tr_idx].clone()  # residual E_L = Y - S beta (L=0), over the FITTED rows only

        # GROW-ON-TOP (2026-07-30). `init_W`/`init_b` seed the growth from units this model already
        # earned, so a refit EXTENDS the substrate instead of replacing it. Until now every refit
        # started from `W = empty(0, d)` above, which is why persisting the grown model still did not
        # make growth accumulate: the restored substrate was discarded by the next refit.
        #
        # The seed re-establishes exactly the invariant the greedy loop below maintains -- W, b, the
        # full-row hidden activity S, the analytical beta, and the residual E over the fitted rows --
        # so the loop cannot tell a seeded start from a grown one, and the earn-your-place stop keeps
        # scoring on rows the solve never fit. beta is re-solved rather than restored on purpose: the
        # readout must describe THESE rows, and a carried-over beta would be a number measured on data
        # this fit has not seen.
        #
        # Refused rather than coerced when the input width disagrees: features of a different d_in are
        # not the same substrate, and silently reshaping them would restore weights that no longer mean
        # what they meant. A refusal falls back to a cold start, which is today's behaviour.
        # DEFAULT PATH UNCHANGED: init_W=None leaves every line below byte-identical.
        if init_W is not None and init_b is not None and int(init_W.shape[0]) > 0:
            if int(init_W.shape[1]) != d:
                init_W = None  # width mismatch -> cold start, and say so via n_seeded below
            else:
                W = init_W.to(self.device).float()
                b = init_b.to(self.device).float()
                S = self._rates(X, W, b)  # full rows, incl. the withheld ones
                beta = self._solve_beta(S[tr_idx], Y[tr_idx])
                E = Y[tr_idx] - S[tr_idx] @ beta
        self.n_seeded = int(W.shape[0])  # units inherited, not grown by this call

        # CAPACITY BOUND, EARNED RATHER THAN PRESET (v4.80), and the tuning story matters because a
        # synthetic benchmark gave the WRONG answer first.
        #
        # On a 6-class synthetic prototype task, growing to the hard bound looked harmful -- cap=200
        # beat unbounded at every noise level (0.865/0.608/0.417 vs 0.850/0.560/0.382) -- which
        # argued for stopping early, and patience=3 was chosen on that basis. Real data reversed it,
        # the synthetic clusters having too low an intrinsic dimensionality to saturate like real
        # perception does, and patience went to 16.
        #
        # BOTH of those numbers were tuned against a measurement that was not measuring anything.
        # The score came from rows the growth solve had already fit, so it fell monotonically and the
        # stop never fired at ANY scale -- pre-fix, growth ran to the hard bound every time (2000 of
        # 2000 rows, 3000 of 3000, 9984 of 10000). The table that used to sit here reported those cap
        # hits as earned capacity.
        #
        # Re-measured with the split made real, scoring ylecun/mnist's own `test` split (8x8, 2,000
        # unseen rows, majority-class floor 0.117, delta=8):
        #
        #   rows     grow to cap        patience=16       patience=48        fixed cap 200
        #   2,000    L=2000  0.901      L=392  0.868      L=640  0.883       L=200  0.826
        #   10,000   L=9984  0.951      L=1520 0.937      L=2096 0.939       L=200  0.871
        #
        # Read it honestly: the leak was NOT free accuracy that vanished. Growing to the cap really
        # is the most accurate setting here -- extra units kept paying right up to one-per-row, so
        # they were not merely memorising. What the corrected stop buys is a 3-5x smaller model for
        # 1-2 points, plus the thing the old code could not do at all: on RANDOM labels it now stops
        # at 16 units where it used to spend all 600, while still taking 256 on learnable data of
        # the same shape. That 16x gap between signal and noise is the whole point of the criterion,
        # and pre-fix it was 1.0x -- both ran to the bound. `tools/capacity_curve.py control` is
        # that check, kept as a regression test with the bug's signature written into its assert.
        #
        # patience=48 rather than 16 because 48 and 128 land on the IDENTICAL result at both scales
        # -- past 48 the contribution criterion saturates on its own and patience is no longer the
        # binding constraint. That makes 48 the point the data picks, not a number chosen to taste.
        #
        # The fixed 200 loses to earned capacity at BOTH scales, which is the durable finding and the
        # one thing that survived every correction: it is an artificial ceiling that fires even on
        # easy problems (measured: it bound at exactly 200/200 on every synthetic task tried) and it
        # truncates real learning.
        #
        # So the ceiling is neither preset nor absent: capacity grows while it still EARNS itself on
        # data the fit has not seen, and stops when it stops paying. `l_max=None` means "no preset
        # limit"; the hard bound is the number of rows the growth solve actually fits, because past
        # one unit per fitted row the readout is underdetermined regardless.
        l_max = self.l_max if self.l_max is not None else n_tr
        best_L, best_val, stale = 0, float("inf"), 0
        while W.shape[0] < l_max:
            if E.pow(2).mean().sqrt().item() < self.eps_tol:  # train RMSE target reached (Alg. 3)
                break
            Esel = E[sel] if use_sub else E  # residual on the scoring rows
            thresh = 1.0 - self.r - self._eta(W.shape[0])  # Eq. 7 acceptance bar (consistency)
            enorm = (Esel * Esel).sum(dim=0)  # (m,) per-class residual energy

            # Draw the whole multi-scale candidate POOL, evaluate it in ONE vectorized LIF pass over
            # the (possibly subsampled) rows, and keep the TOP-delta by residual reduction: the drop
            # from adding unit s is sum_q <e_q,s>^2/||s||^2 = sum_q C_q ||e_q||^2 (Eq. 6 weighted by
            # per-class residual norm). The consistency criterion (Eq. 7) is a preference tiebreak.
            poolW, poolb, poolscore, poolpass, poolzero = [], [], [], [], []
            for Lam in self.lambdas:  # multi-scale candidate search (Alg. 1)
                P = self.t_cand
                # drawn on the CPU stream (see __init__), then moved -- seed decides, hardware does not
                Wp = ((torch.rand(P, d, generator=self.g) * 2 - 1) * Lam).to(self.device)  # (P,d)
                bp = ((torch.rand(P, generator=self.g) * 2 - 1) * Lam).to(self.device)  # (P,)
                Sp = self._rates(Xsel, Wp, bp)  # (Nsel, P) candidate responses
                proj = Sp.t() @ Esel  # (P, m) <e_q, s_p>
                spn = (Sp * Sp).sum(dim=0)  # (P,)   ||s_p||^2
                C = proj.pow(2) / (
                    spn.unsqueeze(1) * enorm.unsqueeze(0) + self.reg_eps
                )  # (P,m) Eq.6
                score = (C * enorm.unsqueeze(0)).sum(dim=1)  # (P,) residual reduction
                passes = (C >= thresh).all(dim=1)  # (P,) meets consistency criterion
                # L3 (specs/022, 2026-09-05): a candidate whose response does not VARY across the
                # fitted rows cannot discriminate them -- identical rows give identical rates, so the
                # test is EXACT zero variance, not an epsilon (P1). At most ONE such unit is ever
                # admitted (the class prior, when nothing has been admitted yet); once any unit exists a
                # constant candidate carries nothing. Measured before this landed: a CONSTANT percept
                # admitted 24 units on the brain's own stream (s2 v2), each shaving ridge slack.
                # A DEAD candidate (zero norm: rate exactly 0 on every scoring row) is excluded always --
                # measured 2026-09-05: the first form of this rule admitted one such unit on a constant
                # percept, whose all-zero readout answered class 0 (accuracy 0.0 against a majority floor of
                # 0.3-0.7). Exact zero, not an epsilon (P1): a dead LIF unit's rate is exactly 0. The
                # constant-response test is an EQUALITY across the scoring rows (identical rows give
                # identical rates), not a variance rounding to zero -- the record panel showed float32
                # variance reads exactly 0.0 only for some (N, rate) pairs. Both tests read the SCORING rows
                # (the fitted rows unless score_subsample is given).
                dead = spn == 0.0
                zero_var = (Sp == Sp[0]).all(dim=0) & ~dead
                passes = passes & ~dead
                score = score.masked_fill(dead, 0.0)
                if W.shape[0] > 0:
                    passes = passes & ~zero_var
                    score = score.masked_fill(zero_var, 0.0)
                poolW.append(Wp)
                poolb.append(bp)
                poolscore.append(score)
                poolpass.append(passes)
                poolzero.append(zero_var)
            allW = torch.cat(poolW, 0)
            allb = torch.cat(poolb, 0)
            allscore = torch.cat(poolscore, 0)
            allpass = torch.cat(poolpass, 0)

            rank = allscore + allpass.float() * 1e12  # prefer passing, then by score
            k = min(delta, l_max - W.shape[0])  # l_max: the derived bound above, not self.l_max
            top = torch.topk(rank, k).indices
            keep = [
                int(j) for j in top.tolist() if allscore[j].item() > 0.0 or bool(allpass[j].item())
            ]  # genuinely contributing
            if (
                W.shape[0] == 0 and keep
            ):  # L3: at most one zero-variance unit, and only as the first
                allzero = torch.cat(poolzero, 0)
                zk = [j for j in keep if bool(allzero[j].item())]
                if zk:
                    keep = [j for j in keep if not bool(allzero[j].item())] + zk[:1]
            if not keep:  # nothing contributes -> saturate
                break
            newW = allW[keep]
            newb = allb[keep]
            Snew = self._rates(X, newW, newb)  # FULL S columns for the block
            W = torch.cat([W, newW], 0)
            b = torch.cat([b, newb], 0)
            S = torch.cat([S, Snew], 1)  # append accepted columns
            beta = self._solve_beta(S[tr_idx], Y[tr_idx])  # ONE analytical solve per block (Eq.12)
            E = Y[tr_idx] - S[tr_idx] @ beta  # update residual, over the fitted rows only

            # EARN-YOUR-PLACE STOP. Only when no preset l_max was given. Score the block on rows the
            # SOLVE did not fit to: train residual falls monotonically as units are added (that is
            # what greedy pursuit does), so it can never say "stop" -- which is exactly why the old
            # eps_tol never fired and the fixed cap ended up deciding capacity. Held-out error can
            # rise, and when it does the last units were memorisation.
            if n_val:  # beta above never saw these rows, so this can genuinely rise
                # RMSE, and NOT the argmax accuracy the classifier is actually scored on -- which is
                # counter-intuitive enough to be worth the lines. Scoring argmax directly was tried
                # and measured WORSE: L=272/0.859 at 2k and L=688/0.918 at 10k, against 392/0.868
                # and 1520/0.937 for RMSE. Held-out accuracy over a few hundred rows is a coarse
                # step function that sits on flat plateaus long enough to exhaust `patience`, while
                # the residual keeps registering the small continued gains underneath them. The
                # smooth proxy is the better STOPPING signal precisely because it is smooth.
                ve = (Y[val_idx] - S[val_idx] @ beta).pow(2).mean().sqrt().item()
                if ve < best_val - 1e-4:
                    best_val, best_L, stale = ve, W.shape[0], 0
                else:
                    stale += 1
                    if stale >= self.patience:  # no held-out gain for this long -> stop
                        break

        # roll back to the width that was actually best on held-out data, so the units that only
        # memorised do not ship. Growth is unbounded in principle and self-limiting on evidence.
        # Then fit the SHIPPED readout on every row: the withheld rows chose the width, and once it
        # is chosen there is no reason to discard their signal.
        if n_val:
            if 0 < best_L < W.shape[0]:
                W, b, S = W[:best_L], b[:best_L], S[:, :best_L]
            beta = self._solve_beta(S, Y)

        self.W, self.b, self.beta = W, b, beta
        return self

    # distinct-from: substrate/brain_bundle.py::predict -- a ROUTER that delegates to a named
    #   skill's trained brain; it implements no decision of its own. distinct-from:
    #   neuromodulation/locus_coeruleus.py::_population_spikes -- integrates an LC pool to a spike
    #   COUNT, no classes. distinct-from: reasoning/rpm_solver.py::_induce_attr -- induces a
    #   Raven's-matrix rule for one attribute. distinct-from: gates/gradient_claim_gate.py's
    #   ClassificationState -- a policy-census declaration. distinct-from:
    #   cerebellum/jepa_world_model.py's invariance loss -- a scalar loss, not a class decision.
    #   All five read 2026-09-07; none decides a class over an NG-SNN rate matrix. This method
    #   CONSOLIDATES the two sites that did: this one and the inline argmax in
    #   cortex/neurogenesis_engine.py::_refit's closed-gate branch, which now calls it.
    def decide(self, S, beta=None):
        """THE ONE decision rule over a rate matrix S (N, L): argmax of S @ beta (o_out, Eq. 13),
        EXCEPT that a row carrying no evidence answers the training-majority class.

        WHY THE EXCEPTION EXISTS (2026-09-07, opened by SC-012a's panel; the record is
        specs/022-unified-brain-arc/evidence/scoring-path-abstention.md). `_rates` returns cnt / T
        and is therefore NON-NEGATIVE, so a row on which no fitted unit fires has an exactly-zero
        rate vector and an exactly-zero score vector. A bare `argmax` resolves that tie to column 0
        -- and `fit` sizes beta as `int(y.max()) + 1`, so a column 0 exists even when the stream
        never emits class 0 as a target. The engine buffers percept_{t-1} -> category_t and its
        measured training classes are 1..5 / 1..4 / 1..3, so column 0 was a class that CANNOT be
        correct, chosen by tensor layout rather than by any decision rule, and recorded as an
        accuracy. Measured then: it decided 45 of 60 held-out rows on one seed and 60 of 60 on
        another (accuracies 0.233 and 0.000).

        The prior is ANSWERED rather than the row dropped from the denominator: excluding those
        rows would have moved that same seed 0.233 -> 0.933, an instrument change manufacturing a
        four-fold gain. Answering the prior leaves the denominator alone and makes an uninformative
        readout score AT its own majority floor, which is the comparison every accuracy here is
        already read against.

        Rows with any firing are UNTOUCHED -- pinned as an exact tensor equality against the bare
        argmax, because an instrument fix that moves a prediction it was not built to change is a
        second defect wearing a repair's clothes."""
        b = self.beta if beta is None else beta
        pred = (S @ b).argmax(dim=1)
        # "no evidence" is read on the RATE row, not on the score: a zero SCORE can also arise from
        # a row that fired into a zero beta column, and that IS an answer the readout made --
        # overriding it would be this fix reaching past its own defect.
        silent = S.abs().sum(dim=1) == 0
        n_silent = int(silent.sum().item())
        # COUNT WHAT WAS ANSWERED FROM THE PRIOR, not what was silent. The first form counted
        # silent rows BEFORE this guard, so on a model with no prior -- an unfitted one, or (until
        # 2026-09-07) a RESTORED one -- it reported N abstentions while `decide` had in fact
        # answered column 0 N times: the disclosure counter read healthiest exactly where the
        # defect was live. Found by this arc's own panel.
        if self.prior_class_ is not None and n_silent:
            pred = torch.where(silent, torch.full_like(pred, int(self.prior_class_)), pred)
            self.last_abstentions_ = n_silent
        else:
            self.last_abstentions_ = 0
        # a row with no evidence AND no prior to answer from is the retired defect still firing;
        # it is counted separately so it can never hide inside the abstention number.
        self.last_unanswerable_ = 0 if self.prior_class_ is not None else n_silent
        return pred

    def predict(self, X):
        """Class indices (N,) for feature/sequence rows X -- `decide` over their rates."""
        S = self._rates(X.to(self.device).float(), self.W, self.b)
        return self.decide(S)

    @property
    def n_units(self):
        return 0 if self.W is None else self.W.shape[0]


# ---------------------------------------------------------------------------------------------------
# __main__ un-riggable self-check: each claim + a CONTROL that MUST fail. Small synthetic task.
# ---------------------------------------------------------------------------------------------------
def _make_task(dev, seed=0):
    """Synthetic 4-class task: the label lives in 2 RELEVANT feature dims (4 well-separated blobs);
    8 further dims are label-INDEPENDENT noise. A contribution learner targets the 2 relevant dims
    (few neurons); a novelty learner that covers input variance wastes capacity on the 8 noise dims
    (many neurons). Deterministic via torch.Generator."""
    # CPU-bound generator and CPU draws, then ONE move at the end: this task's rows are the
    # self-check's evidence, and a dataset that differs between a box with a card and CI is a
    # benchmark whose numbers cannot be compared across machines. Same reason as NGSNN.__init__.
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, d_noise, m = 520, 8, 4
    centers = torch.tensor([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])  # 4 blobs, on cpu
    y = torch.randint(0, m, (N,), generator=g)
    rel = (centers[y] + 0.06 * torch.randn(N, 2, generator=g)).clamp(0, 1)  # (N,2)
    noise = torch.rand(N, d_noise, generator=g)  # (N,8) label-free
    X = torch.cat([rel, noise], dim=1).to(dev)  # (N,10) in [0,1]
    y = y.to(dev)
    ntr = 360
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:]


def _novelty_control(base, X, y):
    """CONTROL: grow the SAME LIF pool + analytical readout, but SELECT each neuron by INPUT novelty
    -- greedy matching pursuit on INPUT reconstruction residual, IGNORING the labels. Stops on the
    SAME train-RMSE-on-Y target as the contribution learner, capped at l_max. Because selection is
    label-blind it wastes picks on the 8 noise dims and OVER-GROWS to reach comparable accuracy."""
    dev = base.device
    X = X.to(dev).float()
    N, d = X.shape
    m = int(y.max().item()) + 1
    Y = torch.zeros(N, m, device=dev)
    Y[torch.arange(N, device=dev), y.to(dev)] = 1.0
    W = torch.empty(0, d, device=dev)
    b = torch.empty(0, device=dev)
    S = torch.empty(N, 0, device=dev)
    beta = torch.empty(0, m, device=dev)
    Rin = X - X.mean(dim=0, keepdim=True)  # input residual (novelty target)
    E = Y.clone()  # label residual: MEASURED, not selected on
    while W.shape[0] < base.l_max:
        if E.pow(2).mean().sqrt().item() < base.eps_tol:
            break
        best_s, best_w, best_b, best_sc = None, None, None, 0.0
        for Lam in base.lambdas:
            P = base.t_cand
            Wp = ((torch.rand(P, d, generator=base.g) * 2 - 1) * Lam).to(dev)
            bp = ((torch.rand(P, generator=base.g) * 2 - 1) * Lam).to(dev)
            Sp = base._rates(X, Wp, bp)  # (N, P)
            proj = Sp.t() @ Rin  # (P, d) response vs INPUT residual
            spn = (Sp * Sp).sum(dim=0)  # (P,)
            score = proj.pow(2).sum(dim=1) / (
                spn + base.reg_eps
            )  # input-residual reduction (ignores Y)
            j = int(score.argmax().item())
            if float(score[j].item()) > best_sc:
                best_sc, best_w, best_b, best_s = (
                    float(score[j].item()),
                    Wp[j],
                    bp[j : j + 1],
                    Sp[:, j],
                )
        if best_s is None or best_sc <= 0.0:
            break
        W = torch.cat([W, best_w.unsqueeze(0)], 0)
        b = torch.cat([b, best_b], 0)
        S = torch.cat([S, best_s.unsqueeze(1)], 1)
        beta = base._solve_beta(S, Y)  # readout still analytical on Y
        E = Y - S @ beta
        coef = torch.linalg.solve(
            S.t() @ S + base.reg_eps * torch.eye(S.shape[1], device=dev), S.t() @ X
        )
        Rin = X - S @ coef  # deflate input residual (novelty)
    out = NGSNN(device=dev)
    out.W, out.b, out.beta = W, b, beta
    return out


def _acc(clf, X, y):
    return (clf.predict(X) == y.to(clf.device)).float().mean().item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xte, yte = _make_task(dev, seed=0)

    # eps_tol=0.20: one-hot RMSE plateaus ~0.10 (contrib) / ~0.17 (novelty), so this target is met by
    # the contribution learner ~2.5x sooner (fewer neurons) than the label-blind novelty control while
    # both reach comparable accuracy -- and both saturate well under l_max.
    cfg = dict(l_max=120, T=15, t_cand=40, r=0.9, eps_tol=0.20, device=dev)

    # ---- claim 1: CONTRIBUTION-not-novelty PARSIMONY ------------------------------------------------
    contrib = NGSNN(seed=7, **cfg).fit(Xtr, ytr)
    novelty = _novelty_control(NGSNN(seed=7, **cfg), Xtr, ytr)
    c_units, c_acc = contrib.n_units, _acc(contrib, Xte, yte)
    n_units, n_acc = novelty.n_units, _acc(novelty, Xte, yte)
    c1_pass = (c_units < n_units) and (n_acc >= c_acc - 0.10)

    # ---- claim 2: SINGLE-SHOT analytical beats RANDOM output weights (zero iterations) --------------
    S_te = contrib._rates(Xte.to(dev).float(), contrib.W, contrib.b)
    ana_acc = (S_te @ contrib.beta).argmax(dim=1).eq(yte.to(dev)).float().mean().item()
    rg = torch.Generator(device="cpu").manual_seed(
        123
    )  # CPU stream: the control must not vary by box
    rand_beta = torch.randn(contrib.n_units, contrib.m, generator=rg).to(dev)
    rnd_acc = (S_te @ rand_beta).argmax(dim=1).eq(yte.to(dev)).float().mean().item()
    c2_pass = (ana_acc > 0.85) and (rnd_acc < 0.55) and (ana_acc - rnd_acc > 0.35)

    # ---- claim 3: GENUINELY SPIKING (decision is a function of INTEGER spike counts) ---------------
    _, counts, v_final = contrib._rates(
        Xte.to(dev).float(), contrib.W, contrib.b, return_state=True
    )
    counts_integer = bool(torch.equal(counts, counts.round()))  # spike counts are integers
    driven = bool((counts.sum(dim=1) > 0).all().item())  # every sample drives >=1 spike
    v_continuous = not bool(
        torch.equal(v_final, v_final.round())
    )  # CONTROL: membrane V is float, NOT integer
    c3_pass = counts_integer and driven and v_continuous

    # ---- claim 4: BOUNDED + DETERMINISTIC (+ block delta and score_subsample consistency) ----------
    bounded = contrib.n_units < cfg["l_max"]  # growth saturated under l_max
    a = NGSNN(seed=7, **cfg).fit(Xtr, ytr)
    c = NGSNN(seed=7, **cfg).fit(Xtr, ytr)
    deterministic = (
        a.n_units == c.n_units
        and torch.equal(a.W, c.W)
        and torch.equal(a.b, c.b)
        and torch.equal(a.predict(Xte), c.predict(Xte))
    )
    blk = NGSNN(seed=7, **cfg).fit(Xtr, ytr, delta=5)  # block neurogenesis
    blk_acc = _acc(blk, Xte, yte)
    delta_consistent = abs(blk_acc - c_acc) <= 0.10
    sub = NGSNN(seed=7, **cfg).fit(Xtr, ytr, score_subsample=180)  # subsampled candidate SELECTION
    sub_acc = _acc(sub, Xte, yte)
    sub_consistent = sub_acc >= c_acc - 0.10
    c4_pass = bounded and deterministic and delta_consistent and sub_consistent

    # ---- claim 5: SPIKE-SEQUENCE mode (the ported encoders drive `_rates`'s einsum branch) ----------
    # Claims 1-4 all run FEATURE mode, and so do both live consumers, so the (N,T,d) branch of
    # `_rates` had no exerciser anywhere in the governed tree until the encoders were ported in.
    # Poisson is the genuinely spiking encoder (binary events, seeded); current_encode is its
    # noise-free control. CONTROL that must FAIL: the same pipeline on a CONSTANT input carries no
    # label information, so it cannot beat the majority class.
    seq_g = torch.Generator(device="cpu").manual_seed(
        11
    )  # CPU stream; poisson_encode moves the draw
    Str_p = poisson_encode(Xtr, 15, generator=seq_g)  # (ntr, 15, d) binary spike trains
    Ste_p = poisson_encode(Xte, 15, generator=seq_g)
    Str_c = current_encode(Xtr, 15)  # (ntr, 15, d) constant-current control
    Ste_c = current_encode(Xte, 15)
    seq_is_3d = Str_p.ndim == 3 and Str_c.ndim == 3  # the einsum branch is the one being taken
    spikes_binary = bool(torch.equal(Str_p, (Str_p > 0).float()))  # Poisson output really is {0,1}
    poi = NGSNN(seed=7, **cfg).fit(Str_p, ytr)
    cur = NGSNN(seed=7, **cfg).fit(Str_c, ytr)
    poi_acc, cur_acc = _acc(poi, Ste_p, yte), _acc(cur, Ste_c, yte)
    flat = current_encode(torch.full_like(Xtr, 0.5), 15)  # CONTROL: label-free constant input
    flat_te = current_encode(torch.full_like(Xte, 0.5), 15)
    flat_acc = _acc(NGSNN(seed=7, **cfg).fit(flat, ytr), flat_te, yte)
    floor = float(torch.bincount(yte).max().item()) / float(yte.numel())  # majority-class rate
    c5_pass = (
        seq_is_3d
        and spikes_binary
        and poi_acc > floor + 0.10
        and cur_acc > floor + 0.10
        and flat_acc <= floor + 1e-6
    )

    # ---- ASCII report table ------------------------------------------------------------------------
    line = "+" + "-" * 4 + "+" + "-" * 46 + "+" + "-" * 34 + "+" + "-" * 6 + "+"
    print("=" * 94)
    print(
        f"NG-SNN (torch core) self-check  |  device={dev}  d_in={contrib.d_in}  classes={contrib.m}"
    )
    print(line)
    print(f"| {'#':>2} | {'claim':<44} | {'measured':<32} | {'ok':<4} |")
    print(line)
    print(
        f"| {1:>2} | {'contribution grows FEWER than novelty':<44} | "
        f"{f'contrib {c_units}u@{c_acc:.2f} < novelty {n_units}u@{n_acc:.2f}':<32} | {str(c1_pass):<4} |"
    )
    print(
        f"| {2:>2} | {'single-shot analytical >> random beta':<44} | "
        f"{f'analytical {ana_acc:.2f} vs random {rnd_acc:.2f}':<32} | {str(c2_pass):<4} |"
    )
    print(
        f"| {3:>2} | {'decision on INTEGER spike counts':<44} | "
        f"{f'counts int={counts_integer}; V float ctrl={v_continuous}':<32} | {str(c3_pass):<4} |"
    )
    print(
        f"| {4:>2} | {'bounded + deterministic + delta/sub':<44} | "
        f"{f'<lmax={bounded} det={deterministic} d5={blk_acc:.2f} sub={sub_acc:.2f}':<32} | {str(c4_pass):<4} |"
    )
    print(
        f"| {5:>2} | {'SPIKE-SEQUENCE mode via ported encoders':<44} | "
        f"{f'poisson {poi_acc:.2f} cur {cur_acc:.2f} flat {flat_acc:.2f}':<32} | {str(c5_pass):<4} |"
    )
    print(line)
    print(
        "CONTROLS (must FAIL): novelty over-grows; random beta ~chance; float membrane V is not "
        f"an integer spike count; a CONSTANT spike sequence cannot beat the majority class {floor:.2f}."
    )
    print("=" * 94)

    # ---- asserts -----------------------------------------------------------------------------------
    assert c1_pass, (
        f"claim1 parsimony: contrib {c_units}u@{c_acc:.2f} vs novelty {n_units}u@{n_acc:.2f}"
    )
    assert c_units < n_units, f"contrib must grow fewer neurons ({c_units} !< {n_units})"
    assert c2_pass, f"claim2 single-shot: analytical {ana_acc:.2f} vs random {rnd_acc:.2f}"
    assert ana_acc - rnd_acc > 0.35, "analytical readout must beat random output weights"
    assert c3_pass, (
        f"claim3 spiking: counts_int={counts_integer} driven={driven} V_float={v_continuous}"
    )
    assert counts_integer and driven, "decision must read INTEGER spike counts of a driven pool"
    assert v_continuous, (
        "CONTROL: continuous membrane V must NOT pass the integer-spike-count assert"
    )
    assert c4_pass, (
        f"claim4 bounded/det: <lmax={bounded} det={deterministic} d5={blk_acc:.2f} sub={sub_acc:.2f}"
    )
    assert bounded and deterministic, "growth must saturate under l_max and be seed-deterministic"
    assert seq_is_3d and spikes_binary, (
        f"claim5: the ported encoders must emit a BINARY (N,T,d) spike sequence "
        f"(3d={seq_is_3d} binary={spikes_binary}) -- that is what selects _rates's einsum branch"
    )
    assert c5_pass, (
        f"claim5 sequence mode: poisson {poi_acc:.2f} / current {cur_acc:.2f} must beat the "
        f"majority floor {floor:.2f}, constant-input control {flat_acc:.2f} must not"
    )
    assert flat_acc <= floor + 1e-6, (
        f"CONTROL: a label-free CONSTANT spike sequence must not beat the majority class "
        f"({flat_acc:.2f} vs {floor:.2f}) -- if it does, the sequence path is reading something "
        "other than the input"
    )

    print("self-check OK: torch core NG-SNN -- contribution-grown spiking classifier, single-shot")
    print("analytical weights (no backprop), block-neurogenesis, deterministic, and BOTH input")
    print("modes exercised (2-D features + Poisson/constant-current (N,T,d) spike sequences).")
    print("A MECHANISM demo; dataset SOTA (Iris/Wine/MNIST/N-MNIST) lives in research/ng_snn/.")


if __name__ == "__main__":
    main()
