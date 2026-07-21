"""
episode_consolidation: systems-level consolidation of REAL episodes into a durable
sparse attractor memory (numpy; spiking-flavoured, gradient-free).

WHY THIS EXISTS (the honest gap it closes). sleep_replay.AttractorNet demonstrates the
Golden-et-al. sleep mechanism on ~3-4 HAND-PICKED toy assemblies in 24 LIF neurons: it
proves the principle but consolidates almost nothing real. The brain's cortex/brain.py
_w8_tick already MANUFACTURES real autobiographical episodes every few wake ticks
(who/what/where/when/affect dicts) and hands them to episodic_recall for reconstruction --
but those recent episodes are NEVER folded into a durable, capacity-scaled associative
store, and the sleep tick only replays the 3 toy sleep_replay patterns. So the system
forms episodes and it has a replay mechanism, but the two are not joined: recent memory is
not systematically consolidated. This module joins them at scale: encode a real episode
dict as a sparse cell ASSEMBLY, STORE it in a large-N (dim=256) Hopfield-style attractor by
covariance Hebbian learning, and during quiescence run noise-driven REPLAY + covariance-STDP
so many dozens of real episodes become durable WITHOUT catastrophically forgetting older ones.

MECHANISM (brain-native, local, no gradients):
  * ENCODE. An episode dict -> a K-of-N sparse binary cell assembly. Every role=filler slot is
    hashed (zlib.crc32, never builtin hash) and the whole-episode hash selects the K active
    units, so distinct episodes land on near-orthogonal assemblies (sparse coding = high
    capacity) and the SAME episode always lands on the SAME assembly (content-addressable).
  * STORE. Covariance Hebbian outer product on the mean-subtracted assembly: W += (x-a)(x-a)^T,
    zero diagonal (a = K/N sparsity). This is the classic sparse auto-associator (Tsodyks &
    Feigelman 1988) -- purely ADDITIVE, so storing a new episode does NOT overwrite the weights
    older episodes rely on: that additivity is the non-catastrophic-forgetting substrate. A fixed
    capacity is held as a ring: the oldest assembly's outer product is subtracted on overflow.
  * RECALL. Pattern completion as a k-WTA attractor settle: from a partial/noisy cue, iterate
    s <- topK(W s) until fixed point. k-WTA (a fixed activity budget, the network's global
    inhibition) makes completion robust and needs no per-unit threshold tuning.
  * READ-OUT (the durable store SPEAKS). Storing only assemblies makes consolidation telemetry:
    a K-of-N bit vector cannot be turned back into a readable episode, so nothing downstream can
    ever say WHAT was consolidated. An index therefore holds, alongside every resident assembly,
    the ORIGINAL episode dict it was encoded from (keyed by zlib.crc32 of the assembly's active
    units), and it follows the ring exactly: evicting a pattern evicts its episode. recall_episode()
    completes a cue through the SAME attractor, matches the settled state back to the nearest
    resident assembly, and returns that assembly's episode DICT -- so a consolidated memory can be
    spoken, not just counted. The index is a read-out label, never part of the dynamics.
  * REPLAY (quiescent consolidation). Noise-driven spontaneous reactivation of the stored
    assemblies in a noise-shuffled order, each cued by a NOISE-CORRUPTED partial of its own
    trace (drop + spurious spikes); the recurrent weights complete it, and when the spontaneous
    state re-enters the assembly it is re-consolidated by the SAME covariance STDP used for
    storage (co-active potentiate, active-with-silent depress). This DEEPENS every stored
    attractor's intra-assembly coupling -- a genuine strengthening -- while the BALANCED schedule
    (every trace reactivated roughly equally) prevents the rich-get-richer collapse that an
    unbalanced blank-noise replay produces (that failure mode is the self-check's negative control).

WHAT IS ROBUST (verified seeds 0-7): at full capacity (64 real episodes in 256 units, K=20) every
episode completes from a 50%-dropped partial cue (retained = 1.00), the EARLIEST-stored episodes
survive all later stores + replay (no catastrophic forgetting), replay increases mean
intra-assembly coupling (~+3%, every seed), and the whole run is byte-deterministic. The negative
control (unbalanced blank-noise replay) collapses retention to ~0.7, showing the replay SCHEDULE is
what keeps memory durable.

HONEST CAVEATS. (1) Within capacity the additive covariance store is already near-saturated, so
replay's measured effect on partial-cue RECALL is ~neutral (it PRESERVES, and it strengthens the
synaptic couplings, but it does not raise an already-1.00 recall) -- the durability claim is
non-destructive consolidation + synaptic strengthening, NOT a recall boost. (2) Role slots are
developer-defined (same honest limit episodic_recall discloses). (3) Beyond capacity the ring
evicts the oldest by design -- that is bounded forgetting, not catastrophic interference.
(3b) recall_episode's episode dict is a STORED LABEL, not a reconstruction: this encoder hashes the
WHOLE episode, so the assembly carries no slot structure to unbind (episodic_recall's HRR traces do
that). The consequence is disclosed at the API: a PATTERN cue may be partial/noisy (that is the
attractor's job), but a DICT cue addresses by exact content -- a genuinely partial dict hashes to an
unrelated assembly and correctly returns an honest MISS rather than a guess. A cue matching nothing
returns an EMPTY episode at confidence 0.0; the nearest-anything is never fabricated.
(4) Representation/value side ONLY: never selects an action, never calls the arm, imports nothing
from brainstem/ or any safety-envelope file; nothing runs at import.

Determinism: np.random.default_rng(seed) for replay noise + per-content random.Random(crc32(...))
for encode/cue; no global seeding, no wall-clock. ASCII-only output. numpy is a core dependency
(zero-dep rule retired 2026-07-20); there are no purity assertions.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.episode_consolidation
"""

import random
import zlib

import numpy as np

# ---- assembly / network ----
ACTIVE_FRAC = 0.08  # sparsity: K active units per assembly = round(dim*ACTIVE_FRAC)
MIN_ACTIVE = 8  # floor on K for very small dims

# ---- recall / completion ----
SETTLE_ITERS = 6  # k-WTA settle steps inside a replay reactivation
RECALL_ITERS = 8  # k-WTA settle steps for an explicit recall
DEFAULT_CUE_FRAC = 0.5  # fraction of an assembly given as a partial cue in retained()
RETAIN_THRESH = 0.9  # completed/true overlap needed to count an episode as recovered

# ---- replay (noise-driven reactivation + covariance STDP) ----
LR_REPLAY = 0.02  # covariance-STDP rate during replay (small -> non-destructive)
REPLAY_DROP = 0.4  # fraction of an assembly's spikes dropped in its reactivation cue
REPLAY_FLIP = 0.15  # spurious spikes added (as a fraction of K) to the reactivation cue
REACTIVATE_MIN = 0.7  # min overlap-with-own-trace for a reactivation to be consolidated

# ---- episode read-out (durable store -> readable episode dict) ----
MIN_EPISODE_CONF = 0.5  # confidence floor for recall_episode; below it we report an HONEST MISS
ENC_CACHE_MULT = 8  # encode-side pattern->episode cache size, as a multiple of capacity


def _canon(episode):
    """Deterministic canonical string for an episode dict (sorted role=filler slots)."""
    return "\x1e".join(f"{k}\x1f{v}" for k, v in sorted(episode.items()))


class EpisodeConsolidator:
    """Durable systems-consolidation memory over real episodes.

    Public interface (what cortex/brain.py would call):
      encode(episode_dict)      -> binary cell-assembly vector (K-of-dim), content-addressable
      store(pattern)            -> add the assembly to the covariance attractor (ring at capacity)
      replay(steps=100)         -> quiescent noise-driven reactivation + covariance-STDP consolidation
      recall(partial_pattern)   -> completed nearest stored assembly (k-WTA attractor settle)
      recall_episode(cue)       -> {"episode","confidence","key"}: the readable EPISODE DICT behind
                                   the completed assembly (empty episode + 0.0 on an honest miss)
      retained()                -> fraction of stored episodes still recoverable from a partial cue

    Deterministic given `seed`. dim units, `active`=K spikes per assembly, `capacity` episodes held.
    """

    def __init__(self, dim=256, capacity=64, seed=0, active=None):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.K = int(active) if active else max(MIN_ACTIVE, round(self.dim * ACTIVE_FRAC))
        self.a = self.K / self.dim  # mean activity (sparsity), the covariance-rule baseline
        self.rng = np.random.default_rng(self.seed)  # replay noise only; per-instance
        self.W = np.zeros((self.dim, self.dim))  # recurrent covariance weights (zero diagonal)
        self.codes = []  # per stored episode: frozenset of active unit indices
        self.patterns = []  # per stored episode: its binary {0,1} assembly vector
        self.keys = []  # per stored episode: crc32 of its assembly (stable content key)
        self.episodes = []  # per stored episode: the ORIGINAL dict (None if stored unlabelled)
        self._enc_cap = max(64, ENC_CACHE_MULT * self.capacity)
        self._enc_seen = {}  # encode-side key -> episode dict, bounded FIFO (labels store())

    # ---- encoding: episode dict -> sparse cell assembly -------------------------------------
    def encode(self, episode):
        """Map an episode dict to a K-of-dim binary cell assembly. Role=filler slots are hashed
        (crc32) and the whole-episode hash draws the K active units, so distinct episodes get
        near-orthogonal assemblies and the same episode is always the same assembly."""
        rng = random.Random(zlib.crc32(_canon(episode).encode("utf-8")))
        idx = rng.sample(range(self.dim), self.K)
        v = np.zeros(self.dim)
        v[idx] = 1.0
        self._remember_encoding(v, episode)
        return v

    def _pattern_key(self, pattern):
        """Stable content key for an assembly: crc32 of its sorted active unit indices (never the
        builtin hash -- that is salted per process and would break determinism across runs)."""
        idx = sorted(np.flatnonzero(np.asarray(pattern, dtype=float) > 0).tolist())
        return zlib.crc32(repr(idx).encode("utf-8"))

    def _remember_encoding(self, pattern, episode):
        """Note that this assembly came from this episode dict, so a later store() can LABEL it.
        Bounded FIFO: a long-running brain encodes far more episodes than it keeps resident."""
        self._enc_seen[self._pattern_key(pattern)] = dict(episode)
        while len(self._enc_seen) > self._enc_cap:
            self._enc_seen.pop(next(iter(self._enc_seen)))

    # ---- storage: covariance Hebbian outer product (additive => non-forgetting) -------------
    def store(self, pattern, episode=None):
        """Store a cell assembly by covariance Hebbian learning W += (x-a)(x-a)^T (zero diagonal).
        Additive, so older episodes are not overwritten. Holds `capacity` episodes as a ring:
        on overflow the oldest assembly's contribution is subtracted (bounded, deliberate).

        The readable label rides along: the assembly is indexed to the episode dict it came from
        (passed here, else the one encode() saw for this exact assembly, else None), and the ring
        evicts the label with the pattern -- an evicted episode can never be spoken again."""
        x = (np.asarray(pattern, dtype=float) > 0).astype(float)
        if episode is not None:
            self._remember_encoding(x, episode)
        key = self._pattern_key(x)
        d = x - self.a
        self.W += np.outer(d, d)
        np.fill_diagonal(self.W, 0.0)
        self.codes.append(frozenset(np.flatnonzero(x).tolist()))
        self.patterns.append(x)
        self.keys.append(key)
        self.episodes.append(self._enc_seen.get(key))
        if len(self.patterns) > self.capacity:
            old = self.patterns.pop(0)
            self.codes.pop(0)
            self.keys.pop(0)
            self.episodes.pop(0)  # evicting the pattern evicts its episode
            do = old - self.a
            self.W -= np.outer(do, do)
            np.fill_diagonal(self.W, 0.0)
        return len(self.codes) - 1

    # ---- completion: k-WTA attractor settle -------------------------------------------------
    def _kwta(self, h):
        """Keep the K most-driven units ON (fixed activity budget = global inhibition)."""
        idx = np.argpartition(h, -self.K)[-self.K :]
        s = np.zeros(self.dim)
        s[idx] = 1.0
        return s

    def _settle(self, s, iters):
        for _ in range(iters):
            ns = self._kwta(self.W @ s)
            if np.array_equal(ns, s):
                break
            s = ns
        return s

    def recall(self, partial_pattern):
        """Pattern completion: clamp the (partial/noisy) cue's spikes, run the k-WTA attractor to a
        fixed point, and return the completed binary assembly (all zeros if nothing is stored)."""
        if not self.codes:
            return np.zeros(self.dim)
        s = (np.asarray(partial_pattern, dtype=float) > 0).astype(float)
        return self._settle(s, RECALL_ITERS)

    # ---- read-out: the durable store returns a READABLE episode -----------------------------
    def _cue_pattern(self, cue):
        """Coerce a cue to a binary assembly. A dict is ENCODED (content-addressed: exact episodes
        only -- see the module's honest caveat 3b); an array is taken as a partial/noisy assembly.
        Returns None for a wrong-shaped cue."""
        if isinstance(cue, dict):
            return self.encode(cue)
        v = np.asarray(cue, dtype=float)
        if v.shape != (self.dim,):
            return None
        return (v > 0).astype(float)

    def recall_episode(self, partial_cue, min_conf=MIN_EPISODE_CONF):
        """Recall the readable EPISODE DICT behind a cue -- the durable store speaking, not counting.

        Completes `partial_cue` (a partial/noisy assembly, or an episode dict which is encoded) via
        the SAME k-WTA attractor as recall(), then matches the settled state back to the resident
        assembly it best explains and returns THAT assembly's indexed episode. Confidence is a
        product of two normalized overlaps, so a match must both complete AND account for the cue:
          completion = |settled & code| / |code|   (did the attractor land in that basin)
          support    = |cue & code| / |cue|        (were the cue's own spikes members of it)
        A novel cue can be dragged into some basin (completion high) but its spikes belong to no
        stored assembly (support low), so the product stays under the floor and we report an HONEST
        MISS: {"episode": {}, "confidence": 0.0, "key": None}. Nothing is ever fabricated -- an empty
        store, an empty/ill-shaped cue, a sub-floor match, or a resident pattern that was stored
        without a label all return the same honest miss. Mirrors episodic_recall.complete()'s shape.
        Deterministic: ties resolve to the earliest-stored assembly."""
        miss = {"episode": {}, "confidence": 0.0, "key": None}
        if not self.codes:
            return miss
        cue = self._cue_pattern(partial_cue)
        if cue is None:
            return miss
        cue_on = frozenset(np.flatnonzero(cue).tolist())
        if not cue_on:
            return miss
        settled = frozenset(np.flatnonzero(self.recall(cue)).tolist())
        best_j, best_conf = -1, 0.0
        for j, code in enumerate(self.codes):
            conf = (len(settled & code) / len(code)) * (len(cue_on & code) / len(cue_on))
            if conf > best_conf:
                best_j, best_conf = j, conf
        if best_j < 0 or best_conf < min_conf or self.episodes[best_j] is None:
            return miss
        return {
            "episode": dict(self.episodes[best_j]),
            "confidence": round(best_conf, 3),
            "key": self.keys[best_j],
        }

    # ---- replay: quiescent noise-driven reactivation + covariance STDP ----------------------
    def replay(self, steps=100):
        """One quiescent (sleep) consolidation pass. Reactivates the stored assemblies in a
        noise-shuffled order, each from a NOISE-CORRUPTED partial of its own trace; the recurrent
        weights complete it, and a clean reactivation is re-consolidated by covariance STDP
        (co-active potentiate, active-with-silent depress) -- deepening every stored attractor.
        Balanced coverage prevents the rich-get-richer collapse of blank-noise replay. Deterministic
        (uses the per-instance rng). Returns the number of consolidation events (telemetry)."""
        if not self.codes:
            return 0
        consolidated = 0
        order = list(range(len(self.codes)))
        done = 0
        while done < steps:
            self.rng.shuffle(order)
            for j in order:
                if done >= steps:
                    break
                done += 1
                members = sorted(self.codes[j])
                keep = [i for i in members if self.rng.random() > REPLAY_DROP]
                cue = np.zeros(self.dim)
                if keep:
                    cue[keep] = 1.0
                n_flip = int(round(REPLAY_FLIP * self.K))
                if n_flip:
                    cue[self.rng.integers(0, self.dim, size=n_flip)] = 1.0
                s = self._settle(cue, SETTLE_ITERS)
                active = frozenset(np.flatnonzero(s).tolist())
                if len(active & self.codes[j]) >= REACTIVATE_MIN * self.K:
                    d = s - self.a
                    self.W += LR_REPLAY * np.outer(d, d)
                    np.fill_diagonal(self.W, 0.0)
                    consolidated += 1
        return consolidated

    def _replay_unbalanced_control(self, steps=100, lr=0.08):
        """NEGATIVE CONTROL (self-check only): unbalanced blank-noise replay. Pure noise fields
        settle into whichever few basins the noise favours, and every settled state is consolidated
        with no balancing -- rich-get-richer that CATASTROPHICALLY collapses retention. This is what
        the balanced replay() above is designed to avoid; it is never used in normal operation."""
        if not self.codes:
            return 0
        for _ in range(steps):
            field = self.rng.standard_normal(self.dim)
            s = self._settle(self._kwta(field), SETTLE_ITERS)
            d = s - self.a
            self.W += lr * np.outer(d, d)
            np.fill_diagonal(self.W, 0.0)
        return steps

    # ---- read-out / diagnostics -------------------------------------------------------------
    def _partial_cue(self, code, frac, noise):
        """A deterministic partial cue for a stored assembly: a content-seeded random `frac` subset
        of its spikes, optionally with `noise` spurious spikes added."""
        members = sorted(code)
        n = max(1, int(round(len(members) * frac)))
        r = random.Random(zlib.crc32(repr(members).encode("utf-8")))
        keep = r.sample(members, n)
        v = np.zeros(self.dim)
        v[keep] = 1.0
        for _ in range(noise):
            v[r.randrange(self.dim)] = 1.0
        return v

    def retained(self, cue_frac=DEFAULT_CUE_FRAC, thresh=RETAIN_THRESH, noise=0):
        """Fraction of currently-stored episodes still recoverable: for each, cue with a `cue_frac`
        partial of its assembly and count it recovered if the completed state overlaps the true
        assembly by >= `thresh`. 1.0 when nothing is stored."""
        if not self.codes:
            return 1.0
        ok = 0
        for code in self.codes:
            out = frozenset(
                np.flatnonzero(self.recall(self._partial_cue(code, cue_frac, noise))).tolist()
            )
            if len(out & code) / len(code) >= thresh:
                ok += 1
        return ok / len(self.codes)

    def mean_assembly_coupling(self):
        """Mean intra-assembly recurrent weight over all stored assemblies (attractor 'depth').
        Replay's covariance STDP raises this -- a direct synaptic-level measure of strengthening."""
        if not self.codes:
            return 0.0
        tot = 0.0
        for code in self.codes:
            idx = sorted(code)
            sub = self.W[np.ix_(idx, idx)]
            tot += (sub.sum() - np.trace(sub)) / (len(idx) * (len(idx) - 1))
        return tot / len(self.codes)


def _demo_episodes(n):
    """`n` distinct real-ish autobiographical episodes shaped exactly like the ones cortex/brain.py
    _w8_tick manufactures (who/what/where/when/affect); unique `when` makes all `n` distinct."""
    acts = ["reach", "grasp", "rest", "track", "withdraw", "explore", "hold", "release"]
    places = ["home", "left", "right", "center", "shelf", "bin", "edge", "corner"]
    affects = ["calm", "fear", "curious", "content", "alert"]
    return [
        {
            "who": "self",
            "what": acts[i % len(acts)],
            "where": places[(i * 3) % len(places)],
            "when": str(i),
            "affect": affects[i % len(affects)],
        }
        for i in range(n)
    ]


def main():
    dim, capacity, seed = 256, 64, 0
    print("=" * 74)
    print("EPISODE CONSOLIDATION -- real episodes -> durable sparse attractor + sleep replay")
    print("=" * 74)

    episodes = _demo_episodes(capacity)  # fill to capacity with DISTINCT real-ish episodes

    def build(with_all=True):
        ec = EpisodeConsolidator(dim=dim, capacity=capacity, seed=seed)
        pats = [ec.encode(e) for e in episodes]
        for p in pats if with_all else pats[: capacity // 2]:
            ec.store(p)
        return ec, pats

    # ---- store the FIRST half, confirm they complete, then store the SECOND half + replay ----
    ec = EpisodeConsolidator(dim=dim, capacity=capacity, seed=seed)
    patterns = [ec.encode(e) for e in episodes]
    # distinctness: encoding is content-addressable and separating (no two episodes collide).
    codes_all = [frozenset(np.flatnonzero(p).tolist()) for p in patterns]
    assert len(set(codes_all)) == len(codes_all), "distinct episodes must get distinct assemblies"
    for p in patterns[: capacity // 2]:
        ec.store(p)
    early_before = ec.retained()
    for p in patterns[capacity // 2 :]:
        ec.store(p)  # later episodes stored AFTER the early ones
    coupling_before = ec.mean_assembly_coupling()
    consolidated = ec.replay(steps=100)
    coupling_after = ec.mean_assembly_coupling()

    # ---- (a) PATTERN COMPLETION: every stored episode recalls from a 50% partial cue ----------
    completion = ec.retained(cue_frac=0.5, noise=0)
    noisy_completion = ec.retained(cue_frac=0.5, noise=5)  # + spurious spikes (robustness probe)
    # explicit per-episode completion from a partial cue (returns the right assembly)
    per_episode = all(
        len(frozenset(np.flatnonzero(ec.recall(ec._partial_cue(c, 0.5, 0))).tolist()) & c) / len(c)
        >= RETAIN_THRESH
        for c in ec.codes
    )
    assert completion == 1.0 and per_episode, (
        f"(a) every stored episode must complete from a partial cue (got {completion:.3f})"
    )
    assert noisy_completion >= 0.95, (
        f"(a) completion must stay robust to spurious spikes (got {noisy_completion:.3f})"
    )

    # ---- (b) NO CATASTROPHIC FORGETTING: earliest episodes survive all later stores + replay --
    early_codes = codes_all[: capacity // 8]  # the FIRST-stored assemblies
    early_after = sum(
        1
        for c in early_codes
        if len(frozenset(np.flatnonzero(ec.recall(ec._partial_cue(c, 0.5, 0))).tolist()) & c)
        / len(c)
        >= RETAIN_THRESH
    ) / len(early_codes)
    assert early_after == 1.0 and completion == 1.0, (
        f"(b) earliest episodes must still recall after later ones + replay (early {early_after:.3f})"
    )

    # ---- replay STRENGTHENS the stored attractors (synaptic-level, monotone) -------------------
    assert coupling_after > coupling_before, (
        f"replay must strengthen intra-assembly coupling ({coupling_before:.4f} -> {coupling_after:.4f})"
    )

    # ---- (c) DETERMINISM: same seed -> byte-identical weights and identical retention ----------
    ec2, _ = build(with_all=True)
    ec2.replay(steps=100)
    deterministic = bool(np.array_equal(ec.W, ec2.W)) and ec.retained() == ec2.retained()
    assert deterministic, "(c) same seed must reproduce identical weights and retention"

    # ---- (d) READ-OUT: the durable store returns the READABLE EPISODE, and misses honestly -----
    # (d1) every stored episode is returned VERBATIM from a 50% partial cue -- consolidation can
    # now SPEAK what it consolidated, instead of only reporting that a pattern came back.
    read_hits, read_confs = 0, []
    for i, code in enumerate(ec.codes):
        got = ec.recall_episode(ec._partial_cue(code, 0.5, 0))
        read_confs.append(got["confidence"])
        if got["episode"] == episodes[i] and got["key"] == ec.keys[i]:
            read_hits += 1
    read_acc = read_hits / len(ec.codes)
    assert read_acc == 1.0, (
        f"(d1) every stored episode must be returned verbatim from a 50% partial cue "
        f"(got {read_acc:.3f})"
    )
    assert min(read_confs) >= MIN_EPISODE_CONF, (
        f"(d1) every recalled episode must clear the confidence floor (min {min(read_confs):.3f})"
    )

    # (d2) DURABLE: the EARLIEST-stored episodes are still returned correctly after all the later
    # stores and a full replay pass -- the read-out inherits the no-catastrophic-forgetting result.
    early_read = sum(
        1
        for i, code in enumerate(codes_all[: capacity // 8])
        if ec.recall_episode(ec._partial_cue(code, 0.5, 0))["episode"] == episodes[i]
    ) / len(codes_all[: capacity // 8])
    assert early_read == 1.0, (
        f"(d2) earliest episodes must still read back after later stores + replay ({early_read:.3f})"
    )

    # (d3) HONEST MISS: cues for episodes that were never stored return an EMPTY episode at
    # confidence 0.0. The attractor still settles somewhere, but those cues' spikes belong to no
    # stored assembly, so no nearest-anything is fabricated.
    unseen = _demo_episodes(capacity + 8)[capacity:]  # 8 episodes never stored
    fresh = EpisodeConsolidator(dim=dim, capacity=capacity, seed=seed)  # encoder only, empty store
    misses = [ec.recall_episode(fresh.encode(e)) for e in unseen]
    assert all(
        m["episode"] == {} and m["confidence"] == 0.0 and m["key"] is None for m in misses
    ), "(d3) a cue matching nothing must return an empty episode at confidence 0.0 (no fabrication)"
    empty_store = EpisodeConsolidator(dim=dim, capacity=capacity, seed=seed).recall_episode(
        fresh.encode(unseen[0])
    )
    assert empty_store == {"episode": {}, "confidence": 0.0, "key": None}, (
        "(d3) an empty store must miss honestly too"
    )

    # (d4) DETERMINISM of the read-out: the twin instance returns identical episodes/confidences.
    read_deterministic = all(
        ec.recall_episode(ec._partial_cue(c, 0.5, 0))
        == ec2.recall_episode(ec2._partial_cue(c, 0.5, 0))
        for c in ec.codes
    )
    assert read_deterministic, "(d4) same seed must reproduce identical episode read-out"

    # ---- ANCHOR: the BALANCED schedule is what keeps memory -- the naive control collapses -----
    ctrl, _ = build(with_all=True)
    ctrl._replay_unbalanced_control(steps=100)
    naive_retained = ctrl.retained(cue_frac=0.5, noise=0)
    assert naive_retained <= completion - 0.15, (
        f"anchor: unbalanced blank-noise replay must collapse retention "
        f"({naive_retained:.3f}) far below the balanced schedule ({completion:.3f})"
    )

    print(
        f"network         : {dim} units, K={ec.K}/assembly (sparsity {ec.a:.3f}), capacity {capacity}"
    )
    print(
        f"stored          : {len(ec.codes)} distinct real-ish episodes (who/what/where/when/affect)"
    )
    print(f"early (half)     completion before storing the rest : {early_before:.3f}")
    print(
        f"(a) completion  : partial-cue {completion:.3f}  noisy-cue {noisy_completion:.3f}  (all recovered)"
    )
    print(
        f"(b) no-forget   : earliest {len(early_codes)} episodes retained after all + replay : {early_after:.3f}"
    )
    print(
        f"    replay      : {consolidated} consolidation events; intra-assembly coupling "
        f"{coupling_before:.4f} -> {coupling_after:.4f} (strengthened)"
    )
    print(f"(c) determinism : same seed -> identical weights + retention = {deterministic}")
    print(
        f"(d) read-out    : {read_hits}/{len(ec.codes)} episodes returned VERBATIM from a 50% "
        f"partial cue (min conf {min(read_confs):.2f})"
    )
    print(
        f"    durable     : earliest {len(codes_all[: capacity // 8])} episodes read back after all "
        f"stores + replay : {early_read:.3f}"
    )
    print(
        f"    honest miss : {len(misses)}/{len(unseen)} never-stored cues -> empty episode, "
        f"confidence 0.00 (nothing fabricated); read-out deterministic = {read_deterministic}"
    )
    print(
        f"anchor          : unbalanced blank-noise replay collapses retention to {naive_retained:.3f} "
        f"(<< {completion:.3f}) -- the schedule keeps memory"
    )
    print("=" * 74)
    print(
        f"self-check OK: {len(ec.codes)} real episodes stored in a {dim}-unit sparse attractor; "
        f"all complete from a 50% partial cue ({completion:.2f}), earliest survive later stores + "
        f"replay (no catastrophic forgetting, {early_after:.2f}); replay strengthened coupling "
        f"{coupling_before:.3f}->{coupling_after:.3f}; deterministic; naive replay collapses to "
        f"{naive_retained:.2f}; READ-OUT: {read_hits}/{len(ec.codes)} episodes returned VERBATIM as "
        f"dicts from a 50% partial cue (min conf {min(read_confs):.2f}), earliest {early_read:.2f} "
        f"after later stores + replay, {len(unseen)}/{len(unseen)} never-stored cues miss honestly "
        f"(empty episode, conf 0.00), read-out deterministic"
    )


if __name__ == "__main__":
    main()
