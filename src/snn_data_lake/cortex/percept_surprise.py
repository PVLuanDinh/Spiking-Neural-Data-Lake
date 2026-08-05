"""
Percept-surprise -- an online, unsupervised SURPRISE signal over a percept-latent stream.

Surprise here is the normalized reconstruction residual of a latent against the top-k principal
subspace the cortex has learned SO FAR: how far the current percept sits OFF the manifold the
recent stream lives on. The subspace is learned by the brain-native local Hebbian rule -- Oja's
rule -- reusing reasoning/oja_pca.OjaNeuron unchanged. k Oja neurons stacked with Gram-Schmidt
deflation form Sanger's generalized-Hebbian rule (exactly as oja_pca's module docstring
describes), so the weights self-organize into the top-k variance directions with NO gradient,
NO backprop, only local plasticity.

The residual is made a bounded scalar by STANDARDIZING it against a running mean+variance (EMA) of
recent residuals: surprise = clip((residual - mean) / (n_sigma*std), 0, 1) -- how many sigma above
the recent mean this percept's residual sits. There is no fixed surprise threshold: the mean+scale
adapt to the recent regime, so "surprising" always means "large relative to what the stream has
recently looked like", not "larger than some magic constant". A plain ratio residual/mean is
degenerate (in-distribution the residual equals its own mean, so surprise pins at 1); standardizing
gives real dynamic range -- a typical on-manifold percept reads ~0 and a genuine off-manifold one ~1.

Inputs are L2-normalized per vector before the Oja update. That keeps the pure-Python Oja rule
numerically stable on real, large-magnitude latents (y = w.x stays in [-1, 1]) and makes surprise
a DIRECTIONAL off-manifold measure with no magic scale constant.

  python -m snn_data_lake.cortex.percept_surprise   # self-check on real N-MNIST latents

numpy-only, ASCII-only, deterministic (per-instance seeds; no global RNG).
"""

import numpy as np

from snn_data_lake.plasticity.oja_pca import OjaNeuron


class PerceptSurprise:
    """Online unsupervised surprise = normalized reconstruction residual off a locally-learned
    top-k principal subspace (stacked Oja neurons + deflation = Sanger's rule)."""

    def __init__(self, in_dim, k=4, lr=0.05, seed=0, baseline_decay=0.03, n_sigma=2.0, eps=1e-8):
        self.in_dim = int(in_dim)
        self.k = int(k)
        self.lr = float(lr)
        self.baseline_decay = float(baseline_decay)
        self.n_sigma = float(n_sigma)
        self.eps = float(eps)
        # k stacked Oja neurons with DISTINCT seeds so their weights start apart and the deflation
        # can order them into the top-k subspace instead of collapsing onto one direction.
        self._neurons = [OjaNeuron(self.in_dim, lr=self.lr, seed=seed + j) for j in range(self.k)]
        # running residual MEAN + VARIANCE (EMA); surprise is the residual STANDARDIZED against them
        # (how many sigma above the recent mean), so a typical on-manifold percept -> ~0 and a genuine
        # off-manifold percept -> ~1. A ratio res/mean is degenerate (in-distribution res == its own
        # mean baseline -> surprise pinned at 1); standardizing gives real dynamic range. Self-
        # calibrating: no fixed ABSOLUTE threshold -- the mean+scale adapt to the current regime, and
        # n_sigma is a RELATIVE shape parameter (cf. episodic_recall._analytic_thresh N_SIGMA).
        self._mu = None
        self._var = 0.0

    def _prep(self, x):
        """Latent -> L2-normalized numpy vector (keeps Oja stable; directional surprise)."""
        xv = np.asarray(x, dtype=float).ravel()
        n = float(np.linalg.norm(xv))
        if n > 0.0:
            xv = xv / n
        return xv

    def _surprise(self, xv):
        """(surprise in [0,1], raw residual) for an already-prepared unit latent, no learning."""
        w = np.asarray([nrn.w for nrn in self._neurons], dtype=float)  # (k, in_dim)
        coeffs = w @ xv  # project onto the current top-k components
        recon = coeffs @ w  # ... and back
        res = float(np.linalg.norm(xv - recon))
        if self._mu is None:
            return 0.0, res  # uncalibrated first sight -> no surprise drive yet
        sigma = max(0.0, self._var) ** 0.5
        s = (res - self._mu) / (
            self.n_sigma * sigma + self.eps
        )  # standardized: sigma above the mean
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        return s, res

    def observe(self, x):
        """One local Hebbian update of the subspace on x, then return surprise in [0,1].

        Learning is pure Oja plasticity (OjaNeuron.learn) applied down a Gram-Schmidt-deflated
        residual -- the Sanger stack. Surprise is read off the CURRENT (just-updated) components,
        and the residual baseline self-calibrates AFTER the read."""
        xv = self._prep(x)
        # NUMPY THROUGHOUT since 2026-08-04 (P7, completing the redesign at the boundary). This read
        # `residual = xv.tolist()` and deflated with a Python loop over `range(self.in_dim)` -- a
        # per-sample demotion of a numpy vector, at in_dim=2312, existing ONLY because OjaNeuron held
        # its weights as a Python list under the retired zero-dependency rule. OjaNeuron is now
        # numpy-backed, so the conversion has nothing left to serve.
        residual = xv
        for nrn in self._neurons:
            y = nrn.learn(residual)  # Oja rule lives entirely in OjaNeuron; we never reimplement it
            residual = residual - y * nrn.w  # deflate for the next PC
        s, res = self._surprise(xv)
        d = self.baseline_decay
        if self._mu is None:
            self._mu = res  # seed the running mean on first sight (variance grows from 0)
        else:
            delta = res - self._mu
            self._mu = self._mu + d * delta
            self._var = (1.0 - d) * (self._var + d * delta * delta)  # EMA variance (West's form)
        return s

    def surprise_only(self, x):
        """Surprise in [0,1] for a read WITHOUT learning or baseline update."""
        s, _ = self._surprise(self._prep(x))
        return s


class _NoveltyBaseline:
    """A DISTINCT familiarity/novelty signal used only by the self-check to test whether surprise
    is a separate faculty. novelty = 1 - max cosine-similarity to a small set of online prototypes
    (competitive nearest-prototype update). This is a density/nearest-neighbour notion, mechanically
    different from PCA reconstruction-residual surprise."""

    def __init__(self, in_dim, m=8, lr=0.1):
        self.in_dim = int(in_dim)
        self.m = int(m)
        self.lr = float(lr)
        self._protos = []

    def observe(self, x):
        xv = np.asarray(x, dtype=float).ravel()
        n = float(np.linalg.norm(xv))
        if n > 0.0:
            xv = xv / n
        if not self._protos:
            self._protos.append(xv.copy())
            return 1.0
        sims = [float(p @ xv) for p in self._protos]
        j = int(np.argmax(sims))
        maxsim = sims[j]
        if maxsim < 0.5 and len(self._protos) < self.m:
            self._protos.append(xv.copy())
        else:
            p = self._protos[j] + self.lr * (xv - self._protos[j])
            pn = float(np.linalg.norm(p))
            self._protos[j] = p / pn if pn > 0.0 else p
        return 1.0 - maxsim


def _load_latents():
    """Real N-MNIST latents: models/nmnist_pack.npz rate-coded (sum spikes over time) + flattened,
    mirroring reasoning/brain_adapters/laminar_local_readout._load_nmnist_split. Returns (X, y) or
    None if the pack is absent."""
    import pathlib

    npz_path = pathlib.Path(__file__).resolve().parents[3] / "models" / "nmnist_pack.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    frames = d["frames"]  # [N, T, 2, 34, 34] spike frames
    labels = d["labels"].astype(int)
    x = frames.sum(axis=1).reshape(frames.shape[0], -1).astype(float)  # rate code, flatten
    return x, labels


def _run_stream(x_all, seed, n=200):
    """One deterministic pass: learn surprise + a novelty baseline over n real latents.
    Returns (surprise_series, novelty_series) as numpy arrays."""
    in_dim = x_all.shape[1]
    ps = PerceptSurprise(in_dim, k=4, lr=0.05, seed=seed)
    nov = _NoveltyBaseline(in_dim, m=8)
    order = np.random.default_rng(seed).permutation(len(x_all))[:n]
    s_series, n_series = [], []
    for i in order:
        s_series.append(ps.observe(x_all[i]))
        n_series.append(nov.observe(x_all[i]))
    return np.asarray(s_series), np.asarray(n_series)


def main():
    loaded = _load_latents()
    if loaded is None:
        # FAIL, do not SKIP. This branch used to print a SKIP line and bare-return, which exits 0 --
        # and run_selfchecks.py scores a module PASS on exit status alone. So a CORE self-check whose
        # only real input is absent reported itself as passing, which made it structurally incapable
        # of failing: every claim below (surprise falls as the subspace learns, off-distribution
        # reads more surprising, surprise is distinct from novelty, determinism) was unverified and
        # counted as verified. An unverifiable check must be loud, not quiet.
        #
        # This will go RED on any checkout without models/nmnist_pack.npz, and that is the point:
        # a CORE self-check whose only real input is absent must be loud, not quiet.
        #
        # RE-VERIFIED 2026-08-04, and the census this comment used to carry was wrong on both of its
        # load-bearing claims -- recorded rather than quietly deleted, because both are the shape of
        # error this project keeps finding in its own notes:
        #   * "FIVE reader sites ... cortex/brain.py:583+599, reasoning/reservoir_eprop.py" -- the
        #     brain's reader is `brain.py`'s `SNN_PERCEPTION_PACK` default, not either cited line
        #     (:583 is `_load_enc`, :599 is the unrelated `SNN_PERCEPTION_CKPT`), and
        #     `reservoir_eprop.py` reads models/**shd_pack**.npz, a DIFFERENT artifact. Its ":43
        #     calls it 'the committed pack'" quote is about that other pack. Actual nmnist_pack
        #     readers: THIS module, `cortex/brain.py` (SNN_PERCEPTION_PACK) and
        #     `reasoning/brain_adapters/laminar_local_readout.py` (_PACK_REL) -- three, not five.
        #   * "NO producer anywhere in the tree" -- `training/make_packs.py` produces BOTH packs
        #     from tonic and is referenced by `run_selfchecks.py` for exactly this check. The claim
        #     was true when written and went stale when make_packs landed.
        #   * "models/ is untracked" -- `.gitignore` excludes `models/*` and then re-includes
        #     `!models/nmnist_pack.npz` and `!models/shd_pack.npz`. These two ARE tracked.
        # What survives is the only part that matters here: without the pack, nothing below is
        # tested, so this must fail rather than SKIP (a SKIP exits 0 and run_selfchecks scores exit
        # status alone, which made every claim in this file unverified-and-counted-as-verified).
        raise AssertionError(
            "percept_surprise CANNOT VERIFY: models/nmnist_pack.npz is absent, so none of this "
            "module's four claims were tested. Previously this printed SKIP and exited 0, which the "
            "self-check runner scores as PASS. Produce it with "
            "`python -m training.make_packs` (it writes models/nmnist_pack.npz and "
            "models/shd_pack.npz) before trusting any percept-surprise number."
        )
    x_all, y_all = loaded
    in_dim = x_all.shape[1]

    # 1) LEARNING: surprise must drop as the subspace captures the manifold.
    s_series, n_series = _run_stream(x_all, seed=0)
    first20 = float(np.mean(s_series[:20]))
    last20 = float(np.mean(s_series[-20:]))

    # 2) OFF-DISTRIBUTION: train on class-0, then a permuted (structure-destroyed) latent must
    #    read MORE surprising than the trained class average.
    c0 = x_all[y_all == 0]
    ps2 = PerceptSurprise(in_dim, k=4, lr=0.05, seed=1)
    for _ in range(3):  # a few epochs so the class-0 subspace converges
        for v in c0:
            ps2.observe(v)
    trained_mean = float(np.mean([ps2.surprise_only(v) for v in c0]))
    perm = np.random.default_rng(123).permutation(in_dim)
    offdist = float(np.mean([ps2.surprise_only(np.asarray(v, dtype=float)[perm]) for v in c0]))

    # 3) DISTINCTNESS FROM NOVELTY: honest Pearson correlation of the two per-step series.
    if float(np.std(s_series)) == 0.0 or float(np.std(n_series)) == 0.0:
        novelty_corr = float("nan")
    else:
        novelty_corr = float(np.corrcoef(s_series, n_series)[0, 1])

    # 4) DETERMINISM: identical re-run.
    s_series_b, _ = _run_stream(x_all, seed=0)
    deterministic = bool(np.array_equal(s_series, s_series_b))

    print(
        f"diag: in_dim={in_dim} N=200 k=4 | first20={first20:.4f} last20={last20:.4f} "
        f"trained_mean={trained_mean:.4f} offdist={offdist:.4f} "
        f"novelty_corr={novelty_corr:.4f} deterministic={deterministic}"
    )
    if abs(novelty_corr) > 0.9:
        print(
            f"WARNING: surprise ~= novelty (corr={novelty_corr:.4f}) -- redundant with the "
            f"nearest-prototype novelty baseline; may not be a distinct faculty"
        )

    assert last20 < first20, (
        f"surprise must drop with learning (first20={first20:.4f} last20={last20:.4f})"
    )
    assert offdist > trained_mean, (
        f"off-distribution must read more surprising (offdist={offdist:.4f} trained_mean={trained_mean:.4f})"
    )
    assert deterministic, "two identical runs must produce identical surprise series"

    print(
        f"percept_surprise OK: first20={first20:.4f} last20={last20:.4f} "
        f"offdist={offdist:.4f} novelty_corr={novelty_corr:.4f}"
    )


if __name__ == "__main__":
    main()
