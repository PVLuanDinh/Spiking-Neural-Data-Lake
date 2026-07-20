"""
Retrospective prefrontal-hippocampal CONTRAST -- structural credit assignment by
pooling episodes, not by updating one transition at a time.

Every existing credit-assignment path in this brain is PER-TRANSITION value updating:
experience_replay / r_stdp / valence_stdp / the arm OutcomeBridge all take a single
(state, action, reward) tuple and nudge one weight toward that reward. None of them ever
asks the retrospective question a prefrontal-hippocampal loop asks after the fact:

    "Across ALL the times I succeeded versus ALL the times I failed, which CONTEXT
     FEATURES actually separated the two?"

That is STRUCTURAL credit assignment: mining the discriminative feature, not crediting the
last action. This module fills that gap. It is the offline, pooled complement to the online
per-transition rules -- hippocampal replay of stored episode outcomes read out by a
prefrontal contrast, yielding a compact "I succeed when feature X is high" prior.

Mechanism (three moves, all pure statistics -- zero learned parameters, fully deterministic):

  RECORD   -- each finished episode drops its outcome (success / failure) and a context
      snapshot (a dict of named scalar features) into one of two pools: self.succ, self.fail.
      Nothing is fit online; the pools are just remembered.

  ANALYZE  -- for every feature that appears in the pools, compute the mean under success and
      the mean under failure, and a DISCRIMINABILITY score
          d = (mean_succ - mean_fail) / (pooled_std + eps)
      This is a Cohen's-d / t-like effect size: how many pooled standard deviations apart the
      two outcome distributions sit on that feature. Features are returned ranked by |d|. A
      feature that truly drives the outcome sits far apart (large |d|); a pure-noise feature
      sits on top of itself (|d| ~ 0) NO MATTER how much it varies -- variance alone cannot
      fake separation. The ranking is the retrospective insight: the CAUSAL context features
      float to the top, the noise sinks, and the analyzer never had to be told which was which.

  ADVICE   -- given a fresh context (a plan the agent is considering), predict its
      success-likelihood from the top discriminative features. Each feature votes by its
      SIGNED effect size times how far the new value sits from the success/failure midpoint
      (a linear-discriminant contribution); the votes sum and squash through a logistic. This
      is the prefrontal recommendation: "this plan looks like your past SUCCESSES (or FAILURES)
      because feature X is high (or low)."

Un-riggable by construction: the analyzer only ever sees anonymous ctx dicts and boolean
outcomes -- it is never told which feature is causal. The self-check GENERATES episodes from a
known rule (success iff one feature crosses a threshold, the rest pure noise) and then checks,
from the outside, that analyze() recovers exactly that feature with a large effect size while
every noise feature scores near zero. The causal feature's identity is even chosen at random in
one check, so the code cannot special-case a name. It finds the cause from the data or it fails.

Determinism: the pools and all statistics are exact functions of the recorded episodes. The only
randomness is a seeded random.Random(seed) used SOLELY to break exact score ties in a stable
order -- same episodes always yield the same ranking and the same advice.

Integration (documentation only -- not wired here): this analyzer would CONSUME the data lake's
stored episode outcomes (the same (outcome, context) records the loop already logs) and FEED
cortex/prefrontal_executive.py a learned prior -- a ranked list of which context features predict
success -- so the executive can bias planning toward high-success-likelihood contexts before it
ever acts, instead of relearning the same structure one transition at a time.

  python -m snn_data_lake.hippocampus.contrastive_episode_analysis
"""
import collections
import math
import random

# ---- tunables (hand-set; see self-check anchors) ----------------------------
EPS = 1e-9          # denominator floor: keeps the effect size finite when pooled_std -> 0
MIN_PER_GROUP = 2   # need >= this many values in BOTH pools to score a feature (variance needs 2)
ADVICE_GAIN = 0.5   # logistic slope on summed discriminant evidence
ADVICE_MIN_EFFECT = 0.5   # advice ignores features weaker than this |effect size| (noise gate)
ADVICE_TOP_K = 8    # advice consults at most this many of the strongest features
_LOGIT_CLAMP = 30.0  # clamp gain*evidence before exp() to avoid overflow

# A single feature's success-vs-failure contrast.
#   score      : signed Cohen's-d-like effect size (mean_succ - mean_fail) / (pooled_std + eps)
#   mean_succ  : feature mean over successful episodes
#   mean_fail  : feature mean over failed episodes
#   pooled_std : pooled (within-group) standard deviation
#   n_succ/n_fail : how many episodes contributed each mean
FeatureContrast = collections.namedtuple(
    "FeatureContrast",
    ["key", "score", "mean_succ", "mean_fail", "pooled_std", "n_succ", "n_fail"],
)

# A prefrontal recommendation for a candidate context.
#   prob            : predicted success-likelihood in (0, 1)
#   predict_success : prob > 0.5
#   evidence        : summed signed discriminant evidence (pre-logistic)
#   reasons         : [(key, signed_contribution)] for the features that voted, |vote| desc
Advice = collections.namedtuple(
    "Advice",
    ["prob", "predict_success", "evidence", "reasons"],
)


def _mean(xs):
    return sum(xs) / len(xs)


def _sample_var(xs, mean):
    """Unbiased (ddof=1) sample variance. Caller guarantees len(xs) >= 2."""
    return sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


class ContrastiveEpisodeAnalyzer:
    """Pools successful vs failed episode contexts and mines the features that separate them.

    The retrospective, offline complement to this brain's per-transition credit-assignment
    rules: instead of crediting the last action, it credits the CONTEXT FEATURE that
    statistically distinguishes success from failure across the whole pool.
    """

    def __init__(self, seed=0):
        self.seed = seed
        self.succ = []   # list[dict[str, float]]: context snapshots of successful episodes
        self.fail = []   # list[dict[str, float]]: context snapshots of failed episodes

    # ---- recording ----------------------------------------------------------
    def record(self, outcome, ctx):
        """Store one finished episode.

        Args:
            outcome: True == success, False == failure.
            ctx: dict of named scalar (numeric) context features observed for that episode.
                A shallow copy is stored so later mutation of the caller's dict is harmless.
        """
        pool = self.succ if outcome else self.fail
        pool.append(dict(ctx))

    def __len__(self):
        return len(self.succ) + len(self.fail)

    # ---- the contrast -------------------------------------------------------
    def analyze(self):
        """Rank every observed feature by how well it separates success from failure.

        For each feature present in BOTH pools with >= MIN_PER_GROUP samples on each side,
        compute the signed effect size d = (mean_succ - mean_fail) / (pooled_std + eps) and
        return the features as FeatureContrast tuples sorted by |d| descending.

        Exact ties in |d| are broken by a stable, seed-derived jitter (deterministic).
        Returns [] until at least one feature is scorable.
        """
        keys = sorted({k for ctx in self.succ for k in ctx}
                      | {k for ctx in self.fail for k in ctx})

        # seed-derived, per-key tie-break jitter -- drawn in a fixed (sorted) key order so it is
        # a pure function of (seed, key set); only ever consulted for exact |score| ties.
        rng = random.Random(self.seed)
        jitter = {k: rng.random() for k in keys}

        contrasts = []
        for key in keys:
            svals = [ctx[key] for ctx in self.succ if key in ctx]
            fvals = [ctx[key] for ctx in self.fail if key in ctx]
            if len(svals) < MIN_PER_GROUP or len(fvals) < MIN_PER_GROUP:
                continue
            m_s = _mean(svals)
            m_f = _mean(fvals)
            n_s, n_f = len(svals), len(fvals)
            var_s = _sample_var(svals, m_s)
            var_f = _sample_var(fvals, m_f)
            pooled_var = ((n_s - 1) * var_s + (n_f - 1) * var_f) / (n_s + n_f - 2)
            pooled_std = math.sqrt(pooled_var)
            score = (m_s - m_f) / (pooled_std + EPS)
            contrasts.append(FeatureContrast(key, score, m_s, m_f, pooled_std, n_s, n_f))

        contrasts.sort(key=lambda fc: (-abs(fc.score), jitter[fc.key]))
        return contrasts

    def top_features(self, k=ADVICE_TOP_K):
        """The k most discriminative features (convenience view over analyze())."""
        return self.analyze()[:k]

    # ---- the prefrontal recommendation --------------------------------------
    def advice(self, ctx, gain=ADVICE_GAIN, min_effect=ADVICE_MIN_EFFECT, top_k=ADVICE_TOP_K):
        """Predict success-likelihood for a candidate context from the top discriminative features.

        Each consulted feature f contributes a signed linear-discriminant vote
            vote_f = score_f * (ctx[f] - midpoint_f) / (pooled_std_f + eps)
        where midpoint_f = (mean_succ + mean_fail) / 2. A feature votes UP when the new value
        lies on its success side (sign of score_f) and DOWN on its failure side; a noise feature
        (|score| below min_effect) is never consulted, so it cannot vote. Votes sum to `evidence`
        and squash through a logistic into a probability.

        Returns an Advice tuple. With no scorable features (or none present in ctx), returns a
        neutral prob = 0.5.
        """
        ranked = [fc for fc in self.analyze() if abs(fc.score) >= min_effect][:top_k]
        reasons = []
        evidence = 0.0
        for fc in ranked:
            if fc.key not in ctx:
                continue
            midpoint = 0.5 * (fc.mean_succ + fc.mean_fail)
            z = (ctx[fc.key] - midpoint) / (fc.pooled_std + EPS)
            vote = fc.score * z
            evidence += vote
            reasons.append((fc.key, vote))

        reasons.sort(key=lambda kv: -abs(kv[1]))
        x = max(-_LOGIT_CLAMP, min(_LOGIT_CLAMP, gain * evidence))
        prob = 1.0 / (1.0 + math.exp(-x))
        return Advice(prob=prob, predict_success=prob > 0.5, evidence=evidence, reasons=reasons)


# ---- self-check helpers ------------------------------------------------------
def _make_episodes(n, causal_key, noise_keys, seed, threshold=0.5):
    """Generate n episodes under a KNOWN rule: outcome == (causal_feature > threshold).

    The causal feature is uniform(0,1); every noise feature is INDEPENDENT uniform(0,1) with no
    bearing on the outcome. Returns a list of (outcome, ctx) pairs. The analyzer never sees which
    key is causal -- it must recover it from the pooled statistics alone.
    """
    rng = random.Random(seed)
    episodes = []
    for _ in range(n):
        ctx = {causal_key: rng.random()}
        for k in noise_keys:
            ctx[k] = rng.random()
        outcome = ctx[causal_key] > threshold
        episodes.append((outcome, ctx))
    return episodes


def _fit(episodes, seed=0):
    az = ContrastiveEpisodeAnalyzer(seed=seed)
    for outcome, ctx in episodes:
        az.record(outcome, ctx)
    return az


def main():
    print("=" * 74)
    print("CONTRASTIVE EPISODE ANALYSIS -- retrospective prefrontal-hippocampal contrast")
    print("structural credit assignment: which CONTEXT FEATURE separates success from failure")
    print("=" * 74)

    causal = "featureA"
    noise = ["featureB", "featureC", "featureD"]
    episodes = _make_episodes(600, causal, noise, seed=1)
    az = _fit(episodes, seed=0)
    ranked = az.analyze()

    n_succ, n_fail = len(az.succ), len(az.fail)
    print(f"recorded {len(az)} episodes: {n_succ} success, {n_fail} failure")
    print(f"ground-truth rule (hidden from analyzer): success iff {causal} > 0.50")
    print("\nfeature contrast ranking (|effect size| descending):")
    for fc in ranked:
        tag = "  <- CAUSAL" if fc.key == causal else "  (noise)"
        print(f"  {fc.key:<10} d={fc.score:+.3f}  succ_mean={fc.mean_succ:.3f}  "
              f"fail_mean={fc.mean_fail:.3f}{tag}")

    top = ranked[0]
    noise_scores = [fc for fc in ranked if fc.key != causal]
    max_noise = max(abs(fc.score) for fc in noise_scores)

    # ---- (a) analyze() recovers the CAUSAL feature, ignores the noise -------
    assert top.key == causal, f"top feature must be the causal one, got {top.key!r}"
    assert abs(top.score) > 1.5, f"causal feature must show a LARGE effect size ({top.score:.3f})"
    assert max_noise < 0.6, f"every noise feature must score near zero (max |d|={max_noise:.3f})"
    assert abs(top.score) > 3.0 * max_noise, (
        f"causal effect must dominate noise by >3x: {abs(top.score):.3f} vs {max_noise:.3f}")

    # ---- (b) advice(): high-causal -> success, low-causal -> failure --------
    hi = az.advice({causal: 0.90, "featureB": 0.5, "featureC": 0.5, "featureD": 0.5})
    lo = az.advice({causal: 0.10, "featureB": 0.5, "featureC": 0.5, "featureD": 0.5})
    print(f"\nadvice(high {causal}=0.90): prob_success={hi.prob:.3f} -> predict "
          f"{'SUCCESS' if hi.predict_success else 'FAILURE'}")
    print(f"advice(low  {causal}=0.10): prob_success={lo.prob:.3f} -> predict "
          f"{'SUCCESS' if lo.predict_success else 'FAILURE'}")
    assert hi.predict_success and hi.prob > 0.7, f"high-{causal} ctx must predict success ({hi.prob:.3f})"
    assert (not lo.predict_success) and lo.prob < 0.3, f"low-{causal} ctx must predict failure ({lo.prob:.3f})"

    # noise cannot flip the verdict: swing every noise feature to its extremes, keep causal high
    flip_a = az.advice({causal: 0.90, "featureB": 0.0, "featureC": 0.0, "featureD": 0.0})
    flip_b = az.advice({causal: 0.90, "featureB": 1.0, "featureC": 1.0, "featureD": 1.0})
    assert flip_a.predict_success and flip_b.predict_success, (
        "noise features must not overturn a high-causal success prediction")
    print(f"noise robustness: high {causal} stays SUCCESS whether noise=0 "
          f"(prob={flip_a.prob:.3f}) or noise=1 (prob={flip_b.prob:.3f})")

    # ---- (c) determinism: same episodes -> same ranking and same advice -----
    az2 = _fit(episodes, seed=0)
    ranked2 = az2.analyze()
    det_rank = ranked == ranked2
    det_adv = az2.advice({causal: 0.90, "featureB": 0.5, "featureC": 0.5, "featureD": 0.5}) == hi
    print(f"\ndeterminism: identical episodes -> identical ranking={det_rank}, identical advice={det_adv}")
    assert det_rank, "same episodes must produce an identical ranking"
    assert det_adv, "same episodes must produce identical advice"

    # ---- un-riggable: the causal feature's IDENTITY is random; no name can be hardcoded ----
    all_keys = ["fx0", "fx1", "fx2", "fx3", "fx4"]
    pick = random.Random(4242)
    recovered = 0
    for trial in range(5):
        ck = pick.choice(all_keys)
        nz = [k for k in all_keys if k != ck]
        eps_trial = _make_episodes(500, ck, nz, seed=100 + trial)
        r = _fit(eps_trial, seed=0).analyze()
        recovered += int(r[0].key == ck)
    print(f"randomized-causal-key control: analyzer recovered the hidden cause in {recovered}/5 trials")
    assert recovered == 5, f"analyzer must recover a randomly-placed causal feature every time ({recovered}/5)"

    print("=" * 74)
    print(f"self-check OK: analyze() mined the causal feature {causal!r} (d={top.score:+.3f}) from the "
          f"pool while all noise scored < {max_noise:.3f}; advice predicted success for high-{causal} "
          f"({hi.prob:.3f}) and failure for low-{causal} ({lo.prob:.3f}); ranking is deterministic and "
          f"a randomly-placed cause was recovered 5/5")


if __name__ == "__main__":
    main()
