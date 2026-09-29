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


def declare_ports(manifest):
    """Manifest registrant (specs/022 T028): THE predictor port. OUT `surprise_out` is the
    bounded off-manifold surprise scalar; its ONE S0-chargeable consumer is the brain's
    undirected-salience competition (`brain.salience_surprise_in`, loop C2's orienting
    drive -- the consumer declares the edge). The OTHER intended consumers are DATED
    FORWARD DISPOSITIONS in SYNTHESIS section 6, each discharged by the task that BUILDS
    that consumer: |learning-progress| gating -> T046a (learning_progress.py) --
    DISCHARGED at T046a: `learning_progress.declare_ports` declares the edge
    surprise_out -> surprise_in and `Brain._perceive_real` carries the stream (the T034/M10
    convention: the discharging task annotates the disposition where it was dated) --
    paging prefetch -> S6 (RE-DATED 2026-09-08 twice: by T066, a surprise scalar names no block until
    the locality reorder gives block adjacency meaning -- evidence/s3-neuron-pager.md 2.8; by T067,
    the adjacency now exists but a prefetch needs spare rounds and a live pager, both S6's --
    evidence/s4-lake-compactor.md 2.8), curriculum difficulty -> T076
    (curriculum.py) -- the first record of this mapping was a clean 3-cycle scramble,
    caught by review; declaring a not-yet-built consumer's edge today would be anatomy
    ahead of any consumer, the force-wiring P6 forbids.

    THE PAGING-PREFETCH DISPOSITION IS VOID 2026-09-09 (T068 LOSS), not re-dated a third time:
    there is no pager to prefetch for and no S6 paging wire to date it to --
    `substrate/neuron_pager.py` and `substrate/lake_compactor.py` are deleted, the layer having
    been measured at a 35.2x tick cost against a 3.0x bound fixed before the run (specs/022
    evidence/s5-paging-null.md 3.4/3.5). It is CLOSED. A successor paging row that wants a
    surprise-driven prefetch declares the edge itself, with its own consumer and its own red.
    The other dispositions above are untouched."""
    from snn_data_lake.engine.wiring_manifest import Port

    manifest.declare_port(
        "percept_surprise",
        Port(
            name="surprise_out",
            region="cortex",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="percept_surprise",
            delivery_policy="all-or-none",
        ),
    )


#: the declared edge, exactly as the manifest carries it -- the charge evidence cites it
SURPRISE_EDGE = ("percept_surprise", "surprise_out", "brain", "salience_surprise_in")


def _surprise_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `salience_surprise_in` port, per arm."""
    from snn_data_lake.engine.brain import declare_ports as brain_declare_ports
    from snn_data_lake.engine.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.engine.wiring_manifest import WiringManifest

    m = WiringManifest()
    brain_declare_ports(m)
    port = m.ports[("brain", "salience_surprise_in")]
    ring = SignalRing(slots=128)
    return ring, JoinConsumer(ring, port)


#: fixed co-source eligibilities for the charge harness -- a plausible quiet tick, held
#: CONSTANT across both replay arms so the only moving input is the surprise payload
_CO_SOURCES = {"novelty": 0.1, "ignition": 0.0, "timing": 0.2, "reroute": 0.0, "drive": 0.1}


def consume_surprise(consumer):
    """The downstream computation under measurement: each delivered surprise frame enters
    the brain's GENUINE undirected-salience competition (brain.undirected_salience,
    extracted at T028 so this harness never re-implements it) under equal starting weights
    -- the brain's own no-source-privileged initial condition. Returns
    [(tick, drive), ...]. LOSS IS FATAL (the T026 ring-overrun lesson)."""
    from snn_data_lake.engine.brain import undirected_salience

    weights = dict.fromkeys([*_CO_SOURCES, "surprise"], 1.0)
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        elig = dict(_CO_SOURCES)
        elig["surprise"] = max(0.0, min(1.0, float(vec[0])))
        out.append((tick, round(undirected_salience(weights, elig), 6)))
    return out


# distinct-from: neuromodulation/astrocyte_modulation.py -- its _gain_frames records the
#   glial gain schedule; this records the SURPRISE stream for the predictor port. Same
#   harness pattern, different wire. Read.
def _surprise_frames(n_ticks=120, offmanifold_at=100, in_dim=24, seed=0):
    """Recorded frames from a REAL PerceptSurprise over a stream that lives on a low-rank
    manifold until `offmanifold_at`, then jumps off it. Burn-in 100 because the review's
    24-seed sweep showed the subspace needs ~100 on-manifold ticks before the contrast is
    a property of the MECHANISM rather than of one seed. k=3 matches the synthetic
    manifold's rank DELIBERATELY (production runs k=8 over the percept latent): the
    harness proves the PORT moves the competition under a converged subspace -- the
    port's charge, not the production configuration, is what is being measured."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(3, in_dim))  # the stream's true 3-dim manifold
    ps = PerceptSurprise(in_dim=in_dim, k=3, seed=seed)
    frames = []
    for t in range(1, n_ticks + 1):
        if t < offmanifold_at:
            x = rng.normal(size=3) @ basis
        else:
            x = rng.normal(size=in_dim)  # full-rank: off the learned manifold
        frames.append((t, "percept_surprise", [float(ps.observe(x))]))
    return frames, offmanifold_at


def _measure_surprise_charge():
    """The replay-twice A/B: live surprise stream vs a pinned zero-surprise payload.
    -> (live, pinned_out, reads, responded): `responded` = the orienting drive rises at
    the off-manifold jump (the predictor port, not the schedule, moved the competition)."""
    from snn_data_lake.instruments.port_charge_probe import replay_twice

    frames, offmanifold_at = _surprise_frames()
    live, pinned_out, reads = replay_twice(
        frames, _surprise_consumer_factory, consume_surprise, [0.0]
    )
    # MEAN-based response criterion (review BLOCKER: max(pre) saturates at 1.0 as the
    # standardized+clipped surprise accumulates on-manifold evidence, so a max-vs-max test
    # passed on 6/24 seeds and INVERTED to 0/24 at long burn-in; the mean contrast is
    # 24/24 at burn-in >= 100 -- the statistic now measures the mechanism, not the seed)
    pre_vals = [d for t, d in live if t < offmanifold_at]
    post_vals = [d for t, d in live if t >= offmanifold_at]
    responded = (sum(post_vals) / len(post_vals)) > (sum(pre_vals) / len(pre_vals))
    return live, pinned_out, reads, responded


def record_charge():
    """Measure the A/B and record the attribution via THE one writer. REFUSES a
    non-diverged or non-responding run -- evidence is a measurement, never a declaration."""
    from snn_data_lake.instruments.port_charge_probe import record_attribution

    live, pinned_out, reads, responded = _measure_surprise_charge()
    if live == pinned_out or not responded:
        print(f"record_charge REFUSED for {SURPRISE_EDGE}: no divergence or no response")
        return 1
    path = record_attribution(
        SURPRISE_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: a real PerceptSurprise stream (on-manifold, then an off-manifold "
        "jump) vs a pinned zero-surprise payload, consumed by the brain's GENUINE "
        "undirected_salience competition under equal starting weights; the orienting "
        "drive diverged and rose at the jump -- the predictor port, not the schedule, "
        "moved the competition; self-check re-measures and requires `measured` to "
        "reproduce",
    )
    print(f"recorded {SURPRISE_EDGE} into {path}")
    return 0


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
    mirroring readout/laminar_local_readout._load_nmnist_split (promoted 2026-08-12). Returns (X, y) or
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
        #   * "FIVE reader sites ... engine/brain.py:583+599, reasoning/reservoir_eprop.py" -- the
        #     brain's reader is `brain.py`'s `SNN_PERCEPTION_PACK` default, not either cited line
        #     (:583 is `_load_enc`, :599 is the unrelated `SNN_PERCEPTION_CKPT`), and
        #     `reservoir_eprop.py` reads models/**shd_pack**.npz, a DIFFERENT artifact. Its ":43
        #     calls it 'the committed pack'" quote is about that other pack. Actual nmnist_pack
        #     readers: THIS module, `engine/brain.py` (SNN_PERCEPTION_PACK) and
        #     `readout/laminar_local_readout.py` (_PACK_REL) -- three, not five.
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

    # ---- THE PREDICTOR PORT IS CHARGED, measured (specs/022 T028) ----------------------
    from snn_data_lake.instruments.port_charge_probe import read_attribution

    live, pinned_out, reads, responded = _measure_surprise_charge()
    assert reads == len(live), f"the neuter must be READ every frame ({reads} vs {len(live)})"
    assert live != pinned_out, (
        "pinning the surprise payload to zero must change the orienting drive -- an "
        "undiverged predictor port is a declared, not a charged, port"
    )
    assert responded, (
        "the off-manifold jump must RAISE the orienting drive over the on-manifold "
        "baseline -- divergence without response attributes nothing"
    )
    rec = read_attribution(SURPRISE_EDGE)
    assert rec is not None and rec.get("diverged") is True, (
        f"committed charge evidence must carry the diverged {SURPRISE_EDGE} attribution -- "
        "regenerate with --record-charge and commit it"
    )
    fresh = {"reads": reads, "n_frames": len(live)}
    assert rec.get("measured", {}) == fresh, (
        f"recorded charge evidence does not reproduce: {rec.get('measured')} vs {fresh} -- "
        "regenerate with --record-charge in the same commit"
    )

    print(
        f"percept_surprise OK: first20={first20:.4f} last20={last20:.4f} "
        f"offdist={offdist:.4f} novelty_corr={novelty_corr:.4f}. PREDICTOR PORT CHARGED "
        f"(measured): live vs pinned diverged over {reads} read frames through the brain's "
        f"genuine undirected_salience competition; committed attribution reproduces."
    )


if __name__ == "__main__":
    import sys as _sys

    if "--record-charge" in _sys.argv[1:]:
        _sys.exit(record_charge())
    main()
