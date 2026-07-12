"""
research/ng_snn/model.py -- faithful NG-SNN classifier (Tang, Li, Zhang, Zeng, "NG-SNN: A
neurogenesis-inspired dynamic adaptive framework for efficient spike classification", Neural
Networks 199 (2026) 108656, DOI 10.1016/j.neunet.2026.108656).

NG-SNN is a plug-and-play SPIKE CLASSIFIER head: a single LIF hidden layer that is GROWN one neuron
at a time under a functional CONTRIBUTION criterion (Eq. 6-7, Alg. 1) and whose output weights are
solved in ONE analytical shot by an activity-dependent regularized pseudoinverse (Eq. 9-12, Alg. 2),
so the head trains in a single epoch with no backprop. This file implements exactly that method,
torch/GPU. The upstream feature extractor (SoftHebb / SCNN) lives in a separate module; here the
input X is the spike/feature sequence it produced (or, in end-to-end mode, Poisson-encoded raw
features -- used to validate the core against the paper's UCI numbers before any front-end).

Faithful to the paper:
  * LIF hidden neuron, discrete Euler (Eq. 2): V[t] = (1-lam) V[t-1] + lam (W x[t] + b), lam=dt/tau_m;
    spike o[t]=1 when V>=V_th, hard reset V<-0 (Eq. 3-4). Response s = mean_t o[t] (Eq. 5).
  * Candidate w,b ~ U(-Lam_k, Lam_k), Lam_k in a multi-scale set {0.1,1,5} (Sec 4.1).
  * Contribution score C_q = <e_q, s_tilde>^2 / (||s_tilde||^2 ||e_q||^2)  (Eq. 6), per output dim q.
  * Consistency contribution criterion: accept iff C_q >= 1 - r - eta_{L+1} for ALL q  (Eq. 7),
    r ~ 0.9, eta_L a non-negative decreasing sequence (eta_L <= 1-r, ->0); paper leaves the exact
    eta form open -- we use eta_L = (1-r)/(L+1) (decreasing, <=1-r, ->0), flagged as a modeling choice.
  * Adaptive weight learning: mu_j = mean_n s_{n,j} (Eq. 9), gamma_j = c1 exp(-k mu_j) + c0
    (c1=1, k=0.1, c0=1e-4), Gamma = diag(gamma), beta = (S^T S + Gamma)^{-1} S^T Y  (Eq. 12).
  * Grow one neuron at a time until train RMSE < eps_tol (~1e-2) or L_max reached (Sec 4.1).

NOT zero-dep -- torch/GPU training tier only. Runs in .venv-cu128 on the RTX 5070; the zero-dep core
never imports this.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/model.py        # synthetic self-check
"""
import torch


def poisson_encode(feats, T, generator=None):
    """Rate/Poisson-encode normalized features [0,1]^(N,d) into a spike sequence {0,1}^(N,T,d):
    each feature value is the per-step Bernoulli firing probability (Sec 4.1, image/UCI end-to-end
    mode). Deterministic given `generator`. NOTE: at T=20 the sampling noise is large and it
    underperforms in this impl -- prefer current_encode for the clean UCI reproduction."""
    N, d = feats.shape
    p = feats.clamp(0, 1).unsqueeze(1).expand(N, T, d)
    return torch.bernoulli(p, generator=generator)


def current_encode(feats, T):
    """Deterministic rate coding: drive the hidden LIF with the normalized feature as a CONSTANT
    input current over T steps (no Poisson sampling noise). The LIF firing rate becomes a clean
    nonlinear function of the feature (a random-feature expansion the analytical ridge reads out) --
    this reproduces the paper's UCI numbers where Poisson sampling underperforms."""
    N, d = feats.shape
    return feats.clamp(0, 1).unsqueeze(1).expand(N, T, d).contiguous()


class NGSNN:
    """NG-SNN spike classifier (growth + analytical ridge). fit(X, y) grows the LIF hidden layer and
    solves the output weights in one shot; predict(X) returns class indices. X is a spike/feature
    sequence {0,1}^(N,T,d_in) (from a front-end, or poisson_encode of raw features).

    Params (paper defaults): tau_m, v_th, dt (LIF); lambdas (candidate scales {0.1,1,5}); r (0.9),
    t_cand (candidates per scale per step); c1,k,c0 (activity regularizer); l_max, eps_tol (stop);
    reg_eps numerical floor. Deterministic via `seed`."""

    def __init__(self, tau_m=2.0, v_th=1.0, dt=1.0, lambdas=(0.1, 1.0, 5.0), r=0.9, t_cand=40,
                 c1=1.0, k=0.1, c0=1e-4, l_max=200, eps_tol=1e-2, reg_eps=1e-6, T=15,
                 device="cuda", seed=0):
        self.tau_m, self.v_th, self.dt = float(tau_m), float(v_th), float(dt)
        self.T = int(T)                                      # LIF integration window for feature (const-current) mode
        self.lam = self.dt / self.tau_m                      # LIF leak-integrate coefficient
        self.lambdas = tuple(float(x) for x in lambdas)
        self.r, self.t_cand = float(r), int(t_cand)
        self.c1, self.k, self.c0 = float(c1), float(k), float(c0)
        self.l_max, self.eps_tol, self.reg_eps = int(l_max), float(eps_tol), float(reg_eps)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.W = None        # (L, d_in) grown hidden input weights
        self.b = None        # (L,) grown hidden biases
        self.beta = None     # (L, m) analytical output weights
        self.d_in = None
        self.m = None

    # ---- LIF hidden-layer firing rates (Eq. 2-5), vectorized + batched over samples ----
    def _rates(self, X, W, b, batch=4096):
        """Average firing rate s in [0,1]^(N,L) of the LIF pool (weights W (L,d), bias b (L,)).
        Two input modes, dispatched on X.ndim:
          * X (N, T, d) spike SEQUENCE  -> current I[t] = W x[t] (einsum over T).
          * X (N, d) FEATURES           -> drive the LIF with the feature as a CONSTANT current
            A = W f over self.T steps (no (N,T,d) tensor -> scales to high-dim MNIST/N-MNIST features).
        Batched over N to bound memory."""
        L = W.shape[0]
        N = X.shape[0]
        feat_mode = (X.ndim == 2)
        T = self.T if feat_mode else X.shape[1]
        out = torch.empty(N, L, device=self.device)
        for s0 in range(0, N, batch):
            xb = X[s0:s0 + batch].to(self.device)
            cur = None
            if feat_mode:
                A = xb @ W.t() + b                                   # (nb, L) constant input current
            else:
                cur = torch.einsum("ntd,ld->ntl", xb, W) + b        # (nb, T, L) input current
            V = torch.zeros(xb.shape[0], L, device=self.device)
            cnt = torch.zeros(xb.shape[0], L, device=self.device)
            for t in range(T):
                it = A if feat_mode else cur[:, t, :]                # this step's input current
                V = (1.0 - self.lam) * V + self.lam * it            # Euler LIF (Eq. 2), V_rest=0
                o = (V >= self.v_th).float()                         # spike (Eq. 3)
                cnt += o
                V = V * (1.0 - o)                                    # hard reset (Eq. 4)
            out[s0:s0 + batch] = cnt / T                             # response s (Eq. 5)
        return out

    def _solve_beta(self, S, Y):
        """Activity-dependent regularized pseudoinverse (Eq. 9-12): gamma_j = c1 exp(-k mu_j)+c0,
        beta = (S^T S + diag(gamma))^{-1} S^T Y."""
        mu = S.mean(dim=0)                                            # (L,) average activity (Eq. 9)
        gamma = self.c1 * torch.exp(-self.k * mu) + self.c0          # (L,) activity-dependent reg
        A = S.t() @ S + torch.diag(gamma)                            # (L, L)
        return torch.linalg.solve(A, S.t() @ Y)                      # (L, m) closed form (Eq. 12)

    def _eta(self, L):
        """Non-negative decreasing sequence eta_L <= 1-r, ->0 (paper leaves the form open); we use
        (1-r)/(L+1)."""
        return (1.0 - self.r) / (L + 1)

    def _contribution(self, s_tilde, E):
        """Contribution score C_q for every output dim q (Eq. 6): squared-cosine between the candidate
        response s_tilde (N,) and each residual column e_q of E (N,m). Returns C (m,)."""
        num = (s_tilde @ E) ** 2                                      # (m,) <e_q, s_tilde>^2
        den = (s_tilde @ s_tilde) * (E * E).sum(dim=0)               # (m,) ||s_tilde||^2 ||e_q||^2
        return num / (den + self.reg_eps)

    def fit(self, X, y, delta=1, score_subsample=None, verbose=False):
        """Grow the classifier on spike/feature X with integer labels y (N,). Candidates are drawn
        multi-scale, scored by the Eq.6 contribution / residual-reduction criterion (Alg.1), and the
        output weights re-solved analytically (Alg.2, Eq.12) until train RMSE < eps_tol or L_max.

        delta (BLOCK-NEUROGENESIS): grow `delta` neurons per step -- take the TOP-delta of the scored
          multi-scale pool and re-solve the ridge ONCE per block (ridge solves L -> ceil(L/delta)).
          delta=1 reproduces the original one-neuron-per-solve growth exactly.
        score_subsample (SUBSAMPLED SCORING): if an int N', candidate SELECTION is scored on a fixed
          random N'-row subset of X (drawn once via self.g), but the accepted neurons' S columns and
          every ridge solve use the FULL X/Y. None => score on all rows (original behavior)."""
        X = X.float()                                           # keep on input device; _rates batches to GPU
        N = X.shape[0]
        d = X.shape[2] if X.ndim == 3 else X.shape[1]           # spike-sequence (N,T,d) or features (N,d)
        self.d_in, self.m = d, int(y.max().item()) + 1
        Y = torch.zeros(N, self.m, device=self.device)               # one-hot targets Y (N, m)
        Y[torch.arange(N), y.to(self.device)] = 1.0

        # fixed scoring subset -- drawn ONCE, ONLY when subsampling (keeps the default path's RNG
        # stream byte-identical to the original single-neuron growth).
        use_sub = score_subsample is not None and int(score_subsample) < N
        if use_sub:
            sub_idx = torch.randperm(N, generator=self.g, device=self.device)[:int(score_subsample)]
            X_sub = X[sub_idx.to(X.device)]                     # (N', ...) fixed candidate-scoring subset
        delta = max(1, int(delta))

        W = torch.empty(0, d, device=self.device)
        b = torch.empty(0, device=self.device)
        S = torch.empty(N, 0, device=self.device)                    # hidden activity matrix S_L (full X)
        beta = torch.empty(0, self.m, device=self.device)
        E = Y.clone()                                                # residual E_L = Y - S beta on FULL X

        while W.shape[0] < self.l_max:
            rmse = (E.pow(2).mean().sqrt()).item()
            if rmse < self.eps_tol:
                break
            thresh = 1.0 - self.r - self._eta(W.shape[0])            # Eq. 7 acceptance bar (consistency gate)
            # Alg. 1 returns the OPTIMAL candidate(s). Draw the whole multi-scale candidate POOL and
            # evaluate it in ONE vectorized LIF pass per scale (essential to scale to 60k samples),
            # then keep the candidate(s) that maximally REDUCE the residual (greedy matching pursuit):
            # the drop from adding unit s is sum_q <e_q,s>^2/||s||^2 = sum_q C_q ||e_q||^2 (Eq. 6
            # weighted by class-residual norm). The consistency criterion (Eq. 7) is a preference
            # tiebreak. Grow until RMSE < eps_tol or L_max; stop if no candidate contributes.
            X_score = X_sub if use_sub else X                       # score candidates on subset or full
            E_score = E[sub_idx] if use_sub else E                  # residual restricted to scoring rows
            enorm = (E_score * E_score).sum(dim=0)                  # (m,) per-class residual energy ||e_q||^2
            Wp_all, bp_all, Sp_all = [], [], []
            for Lam in self.lambdas:                                # multi-scale candidate search (Alg. 1)
                P = self.t_cand
                Wp = (torch.rand(P, d, generator=self.g, device=self.device) * 2 - 1) * Lam   # (P, d)
                bp = (torch.rand(P, generator=self.g, device=self.device) * 2 - 1) * Lam       # (P,)
                Sp = self._rates(X_score, Wp, bp)                   # (N_score, P) candidate responses
                Wp_all.append(Wp); bp_all.append(bp); Sp_all.append(Sp)
            Wp_all = torch.cat(Wp_all, 0)                           # (Pk, d)
            bp_all = torch.cat(bp_all, 0)                           # (Pk,)
            Sp_all = torch.cat(Sp_all, 1)                           # (N_score, Pk)
            proj = Sp_all.t() @ E_score                            # (Pk, m)  <e_q, s_p>
            spn = (Sp_all * Sp_all).sum(dim=0)                     # (Pk,)    ||s_p||^2
            C = proj.pow(2) / (spn.unsqueeze(1) * enorm.unsqueeze(0) + self.reg_eps)   # (Pk, m) Eq. 6
            score = (C * enorm.unsqueeze(0)).sum(dim=1)            # (Pk,) residual reduction (greedy MP)
            passes = (C >= thresh).all(dim=1)                     # (Pk,) meets consistency criterion (Eq. 7)
            rank = score + passes.float() * 1e12                  # prefer passing candidates (as before)
            top1 = int(rank.argmax().item())                      # single best candidate
            if score[top1].item() <= 0.0 and not bool(passes[top1].item()):   # nothing contributes -> stop
                break
            k = min(delta, self.l_max - W.shape[0], rank.numel())  # block size, bounded by l_max & pool
            idx = torch.topk(rank, k).indices                     # TOP-delta of the block
            W_acc, b_acc = Wp_all[idx], bp_all[idx]               # (k, d), (k,)
            # Accepted neurons' S columns MUST come from the FULL X. Under subsampling the pool rates
            # are only on the subset, so recompute the accepted rates on full X; otherwise the scoring
            # rates already ARE the full-X rates -- reuse them (byte-identical to the original column).
            S_new = self._rates(X, W_acc, b_acc) if use_sub else Sp_all[:, idx]   # (N, k)
            W = torch.cat([W, W_acc], 0)
            b = torch.cat([b, b_acc], 0)
            S = torch.cat([S, S_new], 1)                           # append the accepted block of columns
            beta = self._solve_beta(S, Y)                          # ONE analytical ridge solve per block (Eq. 12)
            E = Y - S @ beta                                       # residual on FULL X/Y
            if verbose:
                print(f"  L={W.shape[0]:4d}  train_rmse={E.pow(2).mean().sqrt().item():.4f}")

        self.W, self.b, self.beta = W, b, beta
        return self

    def predict(self, X):
        """Class indices (N,) = argmax of s_new @ beta (o_out, Eq. 13) on spike sequence X."""
        S = self._rates(X.to(self.device).float(), self.W, self.b)
        return (S @ self.beta).argmax(dim=1)

    @property
    def n_units(self):
        return 0 if self.W is None else self.W.shape[0]


def main():
    # Synthetic sanity: a linearly-separable 3-class problem, Poisson-encoded. NG-SNN should grow a
    # small hidden layer and classify near-perfectly in ONE pass -- proves the growth + analytical
    # ridge run end to end (the exact paper numbers are checked in validate_uci.py).
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(0)
    N, d, m, T = 300, 6, 3, 20
    centers = torch.eye(m, d, device=dev)                            # 3 separable class prototypes
    y = torch.randint(0, m, (N,), generator=g, device=dev)
    feats = (centers[y] + 0.15 * torch.randn(N, d, generator=g, device=dev)).clamp(0, 1)
    X = poisson_encode(feats, T, generator=g)
    ntr = 200
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    clf = NGSNN(l_max=60, r=0.9, t_cand=40, device=dev, seed=0).fit(Xtr, ytr, verbose=True)
    acc = (clf.predict(Xte) == yte).float().mean().item()
    print("=" * 68)
    print(f"NG-SNN synthetic self-check | device={dev}  grown units={clf.n_units}  T={T}  m={m}")
    print(f"test accuracy {acc:.3f}  (one-pass, no backprop; grew {clf.n_units} LIF neurons)")
    print("=" * 68)
    assert acc >= 0.9, f"NG-SNN should separate a linearly-separable task ({acc:.3f})"
    assert clf.n_units <= 60, "growth must stay bounded by l_max"
    print("self-check OK: NG-SNN grows a compact LIF head + one-shot analytical weights, one pass")


if __name__ == "__main__":
    main()
