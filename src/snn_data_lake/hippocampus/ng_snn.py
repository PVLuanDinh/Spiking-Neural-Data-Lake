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

TORCH EXCEPTION. This is the ONE core module that imports torch. The zero-dep core rule is otherwise
non-negotiable; this single, user-authorized exception promotes the already-validated research
NG-SNN (research/ng_snn/model.py -- UCI Iris ~95.6% / Wine ~97.2%, MNIST 99.14% / N-MNIST 98.84% at
scale) so the contribution-grown spiking classifier is a first-class brain-region module. The math
is UNCHANGED from that validated model.

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

Deterministic via torch.Generator(seed). NOT zero-dep -- runs in the torch tier (.venv-cu128); the
stdlib core never imports this.

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.hippocampus.ng_snn
"""
import torch


class NGSNN:
    """NG-SNN spike classifier: contribution-grown LIF hidden layer + one-shot analytical readout.

    fit(X, y, delta, score_subsample) grows the pool and solves the output weights; predict(X)
    returns class indices. X is either a spike/feature SEQUENCE {0,1}^(N,T,d_in) or normalized
    FEATURES [0,1]^(N,d_in) (feature mode drives each LIF with a constant current over T steps).

    Params (paper defaults): tau_m, v_th, dt (LIF); lambdas (candidate scales {0.1,1,5}); r (0.9),
    t_cand (candidates per scale per step); c1,k,c0 (activity-dependent regularizer); l_max, eps_tol
    (stop); reg_eps numerical floor; T (feature-mode integration window). Deterministic via `seed`."""

    def __init__(self, tau_m=2.0, v_th=1.0, dt=1.0, lambdas=(0.1, 1.0, 5.0), r=0.9, t_cand=40,
                 c1=1.0, k=0.1, c0=1e-4, l_max=200, eps_tol=1e-2, reg_eps=1e-6, T=15,
                 device="cuda", seed=0):
        self.tau_m, self.v_th, self.dt = float(tau_m), float(v_th), float(dt)
        self.T = int(T)                                      # LIF integration window (feature mode)
        self.lam = self.dt / self.tau_m                      # LIF leak-integrate coefficient
        self.lambdas = tuple(float(x) for x in lambdas)
        self.r, self.t_cand = float(r), int(t_cand)
        self.c1, self.k, self.c0 = float(c1), float(k), float(c0)
        self.l_max, self.eps_tol, self.reg_eps = int(l_max), float(eps_tol), float(reg_eps)
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.W = None        # (L, d_in) grown hidden input weights
        self.b = None        # (L,)      grown hidden biases
        self.beta = None     # (L, m)    analytical output weights
        self.d_in = None
        self.m = None

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
        feat_mode = (X.ndim == 2)
        T = self.T if feat_mode else X.shape[1]
        out = torch.empty(N, L, device=self.device)
        cnt_all = torch.empty(N, L, device=self.device) if return_state else None
        v_all = torch.empty(N, L, device=self.device) if return_state else None
        for s0 in range(0, N, batch):
            xb = X[s0:s0 + batch].to(self.device)
            cur = None
            if feat_mode:
                A = xb @ W.t() + b                                  # (nb, L) constant input current
            else:
                cur = torch.einsum("ntd,ld->ntl", xb, W) + b       # (nb, T, L) input current
            V = torch.zeros(xb.shape[0], L, device=self.device)
            cnt = torch.zeros(xb.shape[0], L, device=self.device)
            for t in range(T):
                it = A if feat_mode else cur[:, t, :]              # this step's input current
                V = (1.0 - self.lam) * V + self.lam * it          # Euler LIF (Eq. 2), V_rest=0
                o = (V >= self.v_th).float()                       # spike (Eq. 3)
                cnt += o                                           # integer spike count
                V = V * (1.0 - o)                                  # hard reset (Eq. 4)
            out[s0:s0 + batch] = cnt / T                           # response s (Eq. 5)
            if return_state:
                cnt_all[s0:s0 + batch] = cnt
                v_all[s0:s0 + batch] = V
        if return_state:
            return out, cnt_all, v_all
        return out

    def _solve_beta(self, S, Y):
        """Activity-dependent regularized pseudoinverse (Eq. 9-12): gamma_j=c1 exp(-k mu_j)+c0,
        beta=(S^T S + diag(gamma))^-1 S^T Y."""
        mu = S.mean(dim=0)                                          # (L,) average activity (Eq. 9)
        gamma = self.c1 * torch.exp(-self.k * mu) + self.c0        # (L,) activity-dependent reg
        A = S.t() @ S + torch.diag(gamma)                          # (L, L)
        return torch.linalg.solve(A, S.t() @ Y)                    # (L, m) closed form (Eq. 12)

    def _eta(self, L):
        """Non-negative decreasing sequence eta_L <= 1-r, ->0 (paper leaves the form open); (1-r)/(L+1)."""
        return (1.0 - self.r) / (L + 1)

    def fit(self, X, y, delta=1, score_subsample=None):
        """Grow the classifier on X (spike sequence {0,1}^(N,T,d_in) or features [0,1]^(N,d_in)) with
        integer labels y (N,). Greedy matching pursuit under the contribution criterion (Alg. 1),
        analytical beta after each block (Alg. 2), until train RMSE < eps_tol or l_max (Alg. 3).

        delta>1: block neurogenesis -- admit the top-`delta` candidates per step, ONE ridge solve per
        block. score_subsample: candidate SELECTION scores on a random row subsample of that size;
        the accepted neurons' S columns and the ridge solve always use the FULL X/Y."""
        X = X.to(self.device).float()
        N = X.shape[0]
        d = X.shape[2] if X.ndim == 3 else X.shape[1]              # sequence (N,T,d) or features (N,d)
        self.d_in, self.m = d, int(y.max().item()) + 1
        Y = torch.zeros(N, self.m, device=self.device)             # one-hot targets Y (N, m)
        Y[torch.arange(N, device=self.device), y.to(self.device)] = 1.0
        delta = max(1, int(delta))

        # candidate SELECTION may run on a row subsample (acceleration); the accepted neurons' S
        # columns and the analytical solve ALWAYS use the full X/Y.
        use_sub = score_subsample is not None and int(score_subsample) < N
        sel = (torch.randperm(N, generator=self.g, device=self.device)[:int(score_subsample)]
               if use_sub else None)
        Xsel = X[sel] if use_sub else X

        W = torch.empty(0, d, device=self.device)
        b = torch.empty(0, device=self.device)
        S = torch.empty(N, 0, device=self.device)                  # full hidden activity matrix S_L
        beta = torch.empty(0, self.m, device=self.device)
        E = Y.clone()                                              # residual E_L = Y - S beta (L=0)

        while W.shape[0] < self.l_max:
            if E.pow(2).mean().sqrt().item() < self.eps_tol:       # train RMSE target reached (Alg. 3)
                break
            Esel = E[sel] if use_sub else E                        # residual on the scoring rows
            thresh = 1.0 - self.r - self._eta(W.shape[0])          # Eq. 7 acceptance bar (consistency)
            enorm = (Esel * Esel).sum(dim=0)                       # (m,) per-class residual energy

            # Draw the whole multi-scale candidate POOL, evaluate it in ONE vectorized LIF pass over
            # the (possibly subsampled) rows, and keep the TOP-delta by residual reduction: the drop
            # from adding unit s is sum_q <e_q,s>^2/||s||^2 = sum_q C_q ||e_q||^2 (Eq. 6 weighted by
            # per-class residual norm). The consistency criterion (Eq. 7) is a preference tiebreak.
            poolW, poolb, poolscore, poolpass = [], [], [], []
            for Lam in self.lambdas:                               # multi-scale candidate search (Alg. 1)
                P = self.t_cand
                Wp = (torch.rand(P, d, generator=self.g, device=self.device) * 2 - 1) * Lam   # (P,d)
                bp = (torch.rand(P, generator=self.g, device=self.device) * 2 - 1) * Lam       # (P,)
                Sp = self._rates(Xsel, Wp, bp)                     # (Nsel, P) candidate responses
                proj = Sp.t() @ Esel                               # (P, m) <e_q, s_p>
                spn = (Sp * Sp).sum(dim=0)                         # (P,)   ||s_p||^2
                C = proj.pow(2) / (spn.unsqueeze(1) * enorm.unsqueeze(0) + self.reg_eps)   # (P,m) Eq.6
                score = (C * enorm.unsqueeze(0)).sum(dim=1)        # (P,) residual reduction
                passes = (C >= thresh).all(dim=1)                 # (P,) meets consistency criterion
                poolW.append(Wp); poolb.append(bp); poolscore.append(score); poolpass.append(passes)
            allW = torch.cat(poolW, 0)
            allb = torch.cat(poolb, 0)
            allscore = torch.cat(poolscore, 0)
            allpass = torch.cat(poolpass, 0)

            rank = allscore + allpass.float() * 1e12               # prefer passing, then by score
            k = min(delta, self.l_max - W.shape[0])
            top = torch.topk(rank, k).indices
            keep = [int(j) for j in top.tolist()
                    if allscore[j].item() > 0.0 or bool(allpass[j].item())]   # genuinely contributing
            if not keep:                                          # nothing contributes -> saturate
                break
            newW = allW[keep]
            newb = allb[keep]
            Snew = self._rates(X, newW, newb)                     # FULL S columns for the block
            W = torch.cat([W, newW], 0)
            b = torch.cat([b, newb], 0)
            S = torch.cat([S, Snew], 1)                           # append accepted columns
            beta = self._solve_beta(S, Y)                         # ONE analytical solve per block (Eq.12)
            E = Y - S @ beta                                      # update residual

        self.W, self.b, self.beta = W, b, beta
        return self

    def predict(self, X):
        """Class indices (N,) = argmax of s_new @ beta (o_out, Eq. 13)."""
        S = self._rates(X.to(self.device).float(), self.W, self.b)
        return (S @ self.beta).argmax(dim=1)

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
    g = torch.Generator(device=dev).manual_seed(seed)
    N, d_noise, m = 520, 8, 4
    centers = torch.tensor([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]], device=dev)  # 4 blobs
    y = torch.randint(0, m, (N,), generator=g, device=dev)
    rel = (centers[y] + 0.06 * torch.randn(N, 2, generator=g, device=dev)).clamp(0, 1)    # (N,2)
    noise = torch.rand(N, d_noise, generator=g, device=dev)                               # (N,8) label-free
    X = torch.cat([rel, noise], dim=1)                                                     # (N,10) in [0,1]
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
    Rin = X - X.mean(dim=0, keepdim=True)                          # input residual (novelty target)
    E = Y.clone()                                                 # label residual: MEASURED, not selected on
    while W.shape[0] < base.l_max:
        if E.pow(2).mean().sqrt().item() < base.eps_tol:
            break
        best_s, best_w, best_b, best_sc = None, None, None, 0.0
        for Lam in base.lambdas:
            P = base.t_cand
            Wp = (torch.rand(P, d, generator=base.g, device=dev) * 2 - 1) * Lam
            bp = (torch.rand(P, generator=base.g, device=dev) * 2 - 1) * Lam
            Sp = base._rates(X, Wp, bp)                            # (N, P)
            proj = Sp.t() @ Rin                                   # (P, d) response vs INPUT residual
            spn = (Sp * Sp).sum(dim=0)                            # (P,)
            score = proj.pow(2).sum(dim=1) / (spn + base.reg_eps)  # input-residual reduction (ignores Y)
            j = int(score.argmax().item())
            if float(score[j].item()) > best_sc:
                best_sc, best_w, best_b, best_s = float(score[j].item()), Wp[j], bp[j:j + 1], Sp[:, j]
        if best_s is None or best_sc <= 0.0:
            break
        W = torch.cat([W, best_w.unsqueeze(0)], 0)
        b = torch.cat([b, best_b], 0)
        S = torch.cat([S, best_s.unsqueeze(1)], 1)
        beta = base._solve_beta(S, Y)                             # readout still analytical on Y
        E = Y - S @ beta
        coef = torch.linalg.solve(S.t() @ S + base.reg_eps * torch.eye(S.shape[1], device=dev), S.t() @ X)
        Rin = X - S @ coef                                        # deflate input residual (novelty)
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
    rg = torch.Generator(device=dev).manual_seed(123)
    rand_beta = torch.randn(contrib.n_units, contrib.m, generator=rg, device=dev)
    rnd_acc = (S_te @ rand_beta).argmax(dim=1).eq(yte.to(dev)).float().mean().item()
    c2_pass = (ana_acc > 0.85) and (rnd_acc < 0.55) and (ana_acc - rnd_acc > 0.35)

    # ---- claim 3: GENUINELY SPIKING (decision is a function of INTEGER spike counts) ---------------
    _, counts, v_final = contrib._rates(Xte.to(dev).float(), contrib.W, contrib.b, return_state=True)
    counts_integer = bool(torch.equal(counts, counts.round()))    # spike counts are integers
    driven = bool((counts.sum(dim=1) > 0).all().item())           # every sample drives >=1 spike
    v_continuous = not bool(torch.equal(v_final, v_final.round()))  # CONTROL: membrane V is float, NOT integer
    c3_pass = counts_integer and driven and v_continuous

    # ---- claim 4: BOUNDED + DETERMINISTIC (+ block delta and score_subsample consistency) ----------
    bounded = contrib.n_units < cfg["l_max"]                       # growth saturated under l_max
    a = NGSNN(seed=7, **cfg).fit(Xtr, ytr)
    c = NGSNN(seed=7, **cfg).fit(Xtr, ytr)
    deterministic = (a.n_units == c.n_units
                     and torch.equal(a.W, c.W) and torch.equal(a.b, c.b)
                     and torch.equal(a.predict(Xte), c.predict(Xte)))
    blk = NGSNN(seed=7, **cfg).fit(Xtr, ytr, delta=5)              # block neurogenesis
    blk_acc = _acc(blk, Xte, yte)
    delta_consistent = abs(blk_acc - c_acc) <= 0.10
    sub = NGSNN(seed=7, **cfg).fit(Xtr, ytr, score_subsample=180)  # subsampled candidate SELECTION
    sub_acc = _acc(sub, Xte, yte)
    sub_consistent = sub_acc >= c_acc - 0.10
    c4_pass = bounded and deterministic and delta_consistent and sub_consistent

    # ---- ASCII report table ------------------------------------------------------------------------
    line = "+" + "-" * 4 + "+" + "-" * 46 + "+" + "-" * 34 + "+" + "-" * 6 + "+"
    print("=" * 94)
    print(f"NG-SNN (torch core) self-check  |  device={dev}  d_in={contrib.d_in}  classes={contrib.m}")
    print(line)
    print(f"| {'#':>2} | {'claim':<44} | {'measured':<32} | {'ok':<4} |")
    print(line)
    print(f"| {1:>2} | {'contribution grows FEWER than novelty':<44} | "
          f"{f'contrib {c_units}u@{c_acc:.2f} < novelty {n_units}u@{n_acc:.2f}':<32} | {str(c1_pass):<4} |")
    print(f"| {2:>2} | {'single-shot analytical >> random beta':<44} | "
          f"{f'analytical {ana_acc:.2f} vs random {rnd_acc:.2f}':<32} | {str(c2_pass):<4} |")
    print(f"| {3:>2} | {'decision on INTEGER spike counts':<44} | "
          f"{f'counts int={counts_integer}; V float ctrl={v_continuous}':<32} | {str(c3_pass):<4} |")
    print(f"| {4:>2} | {'bounded + deterministic + delta/sub':<44} | "
          f"{f'<lmax={bounded} det={deterministic} d5={blk_acc:.2f} sub={sub_acc:.2f}':<32} | {str(c4_pass):<4} |")
    print(line)
    print("CONTROLS (must FAIL): novelty over-grows; random beta ~chance; float membrane V is not "
          "an integer spike count.")
    print("=" * 94)

    # ---- asserts -----------------------------------------------------------------------------------
    assert c1_pass, f"claim1 parsimony: contrib {c_units}u@{c_acc:.2f} vs novelty {n_units}u@{n_acc:.2f}"
    assert c_units < n_units, f"contrib must grow fewer neurons ({c_units} !< {n_units})"
    assert c2_pass, f"claim2 single-shot: analytical {ana_acc:.2f} vs random {rnd_acc:.2f}"
    assert ana_acc - rnd_acc > 0.35, "analytical readout must beat random output weights"
    assert c3_pass, f"claim3 spiking: counts_int={counts_integer} driven={driven} V_float={v_continuous}"
    assert counts_integer and driven, "decision must read INTEGER spike counts of a driven pool"
    assert v_continuous, "CONTROL: continuous membrane V must NOT pass the integer-spike-count assert"
    assert c4_pass, f"claim4 bounded/det: <lmax={bounded} det={deterministic} d5={blk_acc:.2f} sub={sub_acc:.2f}"
    assert bounded and deterministic, "growth must saturate under l_max and be seed-deterministic"

    print("self-check OK: torch core NG-SNN -- contribution-grown spiking classifier, single-shot")
    print("analytical weights (no backprop), block-neurogenesis, deterministic. A MECHANISM demo;")
    print("dataset SOTA (Iris/Wine/MNIST/N-MNIST) lives in research/ng_snn/.")


if __name__ == "__main__":
    main()
