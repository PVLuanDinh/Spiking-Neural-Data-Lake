r"""
Intrinsic-motivation drive -- LEARNING-PROGRESS curiosity that turns a world-model
prediction error into an INTRINSIC reward for self-directed exploration.

Honest framing (read this first). This is the learning-progress curiosity MECHANISM
(Oudeyer & Kaplan 2007; Schmidhuber's formal theory of curiosity; Pathak et al. 2017,
ICM) -- a small deterministic step toward self-directed cumulative learning, NOT a claim
of general developmental intelligence. It computes an intrinsic reward from the RATE at
which a predictor's error is falling; it holds no world model of its own and makes no
action decision. It is the CONSUMER of a surprise signal, not its source.

Where the signal comes from. Curiosity is the complement of the midbrain dopamine nuclei
(midbrain/dopamine_nuclei.py), which EMIT the EXTRINSIC reward-prediction error (RPE) --
"was the world better than I expected?". This drive is the INTRINSIC reward -- "am I getting
BETTER at predicting the world?". The raw surprise it consumes is exactly the scalar
PREDICTION ERROR a world model reports: cerebellum/jepa_world_model.py train_step() returns
that error. The control loop ADDS intrinsic_reward() to any extrinsic reward.

The key idea -- why naive novelty/surprise-seeking is BROKEN. Rewarding raw surprise (high
prediction error) gets permanently stuck on unpredictable NOISE: a screen of static has
maximal prediction error forever (the "noisy-TV problem"). Real developmental curiosity
rewards LEARNING PROGRESS = the RATE at which prediction error DECREASES for a context.
Learning progress is:
  - HIGH where the model is still improving         (learnable novelty -- the frontier),
  - ~0  where the context is already mastered        (error low and flat -- boring),
  - ~0  where error is high but FLAT / unlearnable    (noise -- so it does NOT get stuck).
That third case is what makes curiosity cumulative: it automatically walks the agent to the
frontier of the LEARNABLE and off the cliff of the unlearnable, then on to the next frontier.

Mechanism (deterministic, per-instance rng):
  Per hashable context_key the drive keeps a visit count and TWO exponential moving averages
  of the reported prediction error, at different time constants:
     fast EMA  (rate alpha)            -- responds quickly to recent error,
     slow EMA  (rate alpha_slow<alpha) -- lags behind.
  learning_progress = max(0, slow_EMA - fast_EMA):
     while error is FALLING the fast EMA drops below the lagging slow EMA -> positive;
     once error is FLAT (mastered-low OR noisy-high) both EMAs meet the same mean -> ~0.
     Differencing two SMOOTHED estimates -- not the raw error -- is exactly what rejects the
     noisy TV: a constant-high-mean-plus-jitter stream yields ~0 progress.
  novelty = novelty_decay ** count, in [0,1]: 1.0 for a never-seen context, HABITUATES
     geometrically toward 0 with repeated observe().
  intrinsic_reward = learning_progress * (1 + novelty_gain * novelty): the PRIMARY term is
     learning progress (non-negative, bounded by the error scale), MULTIPLICATIVELY modulated
     by novelty. Because it multiplies learning progress, a mastered OR noisy context (lp ~0)
     earns ~0 reward no matter how novel -- the anti-noisy-TV guarantee holds at the reward
     level too. curiosity_value() adds a small ADDITIVE novelty bonus so an unseen context is
     still worth a first taste (novelty bootstraps; learning progress sustains) -- that drives
     the curiosity-greedy explorer without ever leaking sustained reward to noise.

Refs: Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational
approaches" (2007); Schmidhuber, "Formal Theory of Creativity, Fun, and Intrinsic
Motivation" (2010); Pathak, Agrawal, Efros & Darrell, "Curiosity-driven Exploration by
Self-supervised Prediction" (ICM, 2017).

Deps: random, substrate._mathutils.mean (numpy/torch allowed project-wide); ASCII-only prints, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.intrinsic_motivation
"""

import random

from snn_data_lake.substrate._mathutils import mean

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "CuriosityDrive.learning_progress": (
        "NOT_A_RULE",
        "returns rectified learning progress, max(0, slow_EMA - fast_EMA), in the units of prediction "
        "error. A MEASUREMENT of how fast learning is going, not learning",
    ),
    "CuriosityDrive.learning_progress_abs": (
        "NOT_A_RULE",
        "returns |slow_EMA - fast_EMA| from the SAME state (specs/022 T046a's third-factor "
        "readout -- a regression is progress-shaped news to a growth gate). A measurement, "
        "not learning",
    ),
}


class CuriosityDrive:
    """A learning-progress curiosity / intrinsic-motivation drive.

    observe(k, e) updates a per-context visit count and fast/slow prediction-error EMAs;
    novelty(k) is a habituating [0,1] recency signal; learning_progress(k) is the rectified rate
    at which the context's error is FALLING (the anti-noisy-TV signal); intrinsic_reward(k, e)
    observes THEN returns the curiosity reward the loop adds to any extrinsic reward; curiosity_value(k)
    is the expected-reward heuristic (learning progress + a bootstrapping novelty bonus) that drives a
    curiosity-greedy explorer. Deterministic; per-instance random.Random(seed)."""

    def __init__(
        self,
        alpha=0.2,
        novelty_decay=0.6,
        slow_ratio=0.2,
        novelty_gain=0.5,
        explore_bonus=0.05,
        seed=0,
    ):
        self.alpha = float(alpha)  # fast prediction-error EMA rate (the primary estimate)
        self.alpha_slow = float(alpha) * float(
            slow_ratio
        )  # slow EMA rate (lags -> the learning-progress reference)
        self.novelty_decay = float(
            novelty_decay
        )  # geometric habituation factor per observe, in (0,1)
        self.novelty_gain = float(
            novelty_gain
        )  # how strongly novelty multiplicatively boosts the reward
        self.explore_bonus = float(
            explore_bonus
        )  # additive novelty bonus in curiosity_value (bootstrap only)
        self.rng = random.Random(seed)  # per-instance rng (selector tie-break / determinism)
        self._ctx = {}  # context_key -> {"n": count, "fast": EMA, "slow": EMA}

    def observe(self, context_key, pred_error):
        """Fold one world-model prediction error for context_key into its running estimates: increment the
        visit count and update the fast (alpha) and slow (alpha_slow) error EMAs. The first sighting seeds
        BOTH EMAs to the error itself so there is no artificial startup transient (learning_progress ~0 at
        first observe -- no history yet). Returns None."""
        e = float(pred_error)
        st = self._ctx.get(context_key)
        if st is None:
            self._ctx[context_key] = {"n": 1, "fast": e, "slow": e}
            return
        st["n"] += 1
        st["fast"] += self.alpha * (e - st["fast"])
        st["slow"] += self.alpha_slow * (e - st["slow"])

    def novelty(self, context_key):
        """Novelty in [0,1]: 1.0 for a never-seen context, HABITUATES as novelty_decay**count with each
        observe() -- HIGH for rarely-seen contexts, decays toward 0 for familiar ones."""
        st = self._ctx.get(context_key)
        if st is None:
            return 1.0
        return self.novelty_decay ** st["n"]

    def learning_progress(self, context_key):
        """Rectified learning progress = max(0, slow_EMA - fast_EMA), in the units of pred_error.

        POSITIVE while the context's prediction error is FALLING (the model is still learning it); ~0 once
        the error is FLAT -- whether mastered-low OR unlearnable-high. That the high-but-flat (noisy) case
        gives ~0 -- not stuck-high -- is the anti-noisy-TV property: it is the DECREASE of a smoothed error,
        never the raw error. (MEASURED at T046a and worth stating: the EMA pair converges to a difference
        of EXACTLY 0.0 on an UNJITTERED flat stream at EVERY error level -- the structural property the
        spiking sibling's then-shipped threshold-pool pair lacked below its calibration reference, which
        is why T046a's |LP| facade adopts THIS drive. RE-DATED 2026-08-26: the spiking sibling was
        rebuilt to run this same EMA pair over LIF spike counts, closing that defect; what still
        separates the two is the spiking form's count-quantization residuals -- its own docstring
        carries the measurements -- and the facade's adoption of THIS drive stands unchanged.)"""
        st = self._ctx.get(context_key)
        if st is None:
            return 0.0
        return max(0.0, st["slow"] - st["fast"])

    # distinct-from: neuromodulation/spiking_curiosity.py -- the spiking sibling of THIS
    #   drive's rectified readout (a flag-selected coexisting pair, OPEN P2 debt per its own
    #   docstring; its flat-error bias below the calibration reference is measured in
    #   learning_progress.py's adoption record and re-derivable in one probe). This method
    #   adds the |.| readout to the EMA form ONLY. Site read.
    # distinct-from: cerebellum/jepa_world_model.py -- BRidge error transport; unrelated. Read.
    # distinct-from: cortex/learned_evc.py -- a dopamine-gated three-factor UPDATE; this is a
    #   read-only measurement. Site read.
    def learning_progress_abs(self, context_key):
        """|slow_EMA - fast_EMA| from the SAME state (specs/022 T046a): the UNRECTIFIED
        third-factor readout. A growth gate must see a REGRESSION (error rising: fast EMA
        above slow) as loudly as an improvement -- the reward-side rectification
        deliberately zeroes it there, which is right for reward and wrong for a gate.
        0.0 before any observe."""
        st = self._ctx.get(context_key)
        if st is None:
            return 0.0
        return abs(st["slow"] - st["fast"])

    def intrinsic_reward(self, context_key, pred_error):
        """The curiosity reward the loop ADDS to any extrinsic reward.

        Calls observe() internally (so the caller passes each fresh prediction error exactly ONCE -- do not
        also call observe() for the same error), then returns learning_progress * (1 + novelty_gain * novelty).
        Learning progress is the PRIMARY, non-negative, bounded term; novelty only MULTIPLICATIVELY modulates
        it, so a mastered or noisy context (lp ~0) earns ~0 no matter how novel. Deterministic."""
        self.observe(context_key, pred_error)
        return self.learning_progress(context_key) * (
            1.0 + self.novelty_gain * self.novelty(context_key)
        )

    def curiosity_value(self, context_key):
        """Expected intrinsic reward of visiting context_key NEXT, from current stats (no new error yet):
        learning_progress * (1 + novelty_gain * novelty) + explore_bonus * novelty. The additive novelty bonus
        makes an unseen context worth a first taste (bootstrap); once habituated only real learning progress
        sustains value -- so a curiosity-greedy selector chases the learnable frontier, never the noise."""
        lp = self.learning_progress(context_key)
        nov = self.novelty(context_key)
        return lp * (1.0 + self.novelty_gain * nov) + self.explore_bonus * nov

    def most_curious(self, candidates):
        """Return the context_key of maximal curiosity_value among `candidates` (a curiosity-greedy pick).
        Deterministic: the per-instance rng only shuffles first so exact ties are broken without positional
        bias -- same seed + same history -> same choice."""
        cand = list(candidates)
        self.rng.shuffle(cand)
        return max(cand, key=self.curiosity_value)


# =====================================================================================================
# Self-check: five un-riggable, deterministic controls. Every stream below is fixed (or drawn from a
# seeded rng), so the tail numbers reproduce byte-for-byte on every run.
# =====================================================================================================


def _run_explorer(drive, n_rounds, greedy):
    """Drive a K-candidate world with either a curiosity-greedy selector (most_curious) or a curiosity-blind
    random control, and count visits per candidate.

      learnable : hidden error starts at 1.0 and DROPS 15% each time it is VISITED (a learnable frontier that
                  masters with practice, then flattens at a 0.05 floor) -> sustained learning progress;
      mastered  : error already flat-low (0.05)                          -> learning progress ~0 (boring);
      noisy     : error flat-high with jitter (0.9 +/- 0.1)              -> learning progress ~0 (unlearnable).

    The greedy selector should CHASE the learnable candidate (real learning progress) and, once habituated,
    ignore the noisy one; the random control has no such preference."""
    err = {"learnable": 1.0, "mastered": 0.05, "noisy": 0.9}
    visits = {"learnable": 0, "mastered": 0, "noisy": 0}
    cands = ("learnable", "mastered", "noisy")
    for _ in range(n_rounds):
        k = drive.most_curious(cands) if greedy else drive.rng.choice(cands)
        if k == "learnable":
            e = err["learnable"]
            err["learnable"] = max(
                0.05, err["learnable"] * 0.85
            )  # visiting it teaches the model (error falls)
        elif k == "mastered":
            e = err["mastered"]  # already flat-low
        else:
            e = 0.9 + drive.rng.uniform(-0.1, 0.1)  # flat-high noise (no downward trend)
        drive.intrinsic_reward(k, e)  # observes + updates the per-context stats
        visits[k] += 1
    return visits


def main():
    # ---- (1) HABITUATION: novelty decays for a repeated context, stays high for an unseen one ----
    d1 = CuriosityDrive(seed=0)
    nov_curve = []
    for _ in range(30):
        d1.observe("A", 0.5)
        nov_curve.append(d1.novelty("A"))
    nov_unseen = d1.novelty("B")  # never observed

    # ---- (2) LEARNING PROGRESS tracks a LEARNING curve: positive during decay, ~0 once mastered ----
    d2 = CuriosityDrive(seed=0)
    lp_learning = []
    e = 1.0
    for _ in range(30):  # geometric decay 1.0, 0.9, 0.81, ... (mastering)
        d2.observe("L", e)
        lp_learning.append(d2.learning_progress("L"))
        e *= 0.9
    for _ in range(200):  # plateau: error now flat -> progress dies out
        d2.observe("L", e)
    lp_peak = max(lp_learning)
    lp_plateau = d2.learning_progress("L")

    # ---- (3) ANTI-NOISY-TV (the critical control): learnable frontier vs high-but-flat noise ----
    # Accumulate the learning_progress SIGNAL itself (the un-riggable proof it is not naive surprise): a
    # genuinely learnable context (decaying error) piles up progress; a high-but-flat noisy context ~does not.
    d3 = CuriosityDrive(seed=1)
    srng = random.Random(1234)
    learn_stream = [
        0.9**t for t in range(90)
    ]  # a genuinely learnable context (error decays, masters)
    noisy_stream = [
        0.9 + srng.uniform(-0.05, 0.05) for _ in range(90)
    ]  # constant-high mean + jitter, NO downward trend
    learn_lp = 0.0
    noisy_lp = 0.0
    learn_reward = 0.0
    noisy_reward = 0.0
    for le, ne in zip(learn_stream, noisy_stream):
        learn_reward += d3.intrinsic_reward(
            "learnable", le
        )  # observes, then accumulates the reward ...
        noisy_reward += d3.intrinsic_reward("noisy", ne)
        learn_lp += d3.learning_progress("learnable")  # ... and the raw learning-progress signal
        noisy_lp += d3.learning_progress("noisy")
    noisy_final_err = d3._ctx["noisy"]["fast"]  # noise stays HIGH ...
    noisy_final_lp = d3.learning_progress("noisy")  # ... yet its learning progress is ~0

    # ---- (4) EXPLORATION: a curiosity-greedy explorer prefers the learnable-novel context; random does not ----
    rounds = 48
    greedy_visits = _run_explorer(CuriosityDrive(seed=2), rounds, greedy=True)
    random_visits = _run_explorer(CuriosityDrive(seed=2), rounds, greedy=False)

    # ---- (5) DETERMINISM: two seed=s drives fed the same stream give identical signals ----
    stream = [
        ("x", 0.9),
        ("x", 0.7),
        ("x", 0.55),
        ("y", 0.30),
        ("x", 0.45),
        ("y", 0.30),
        ("x", 0.40),
    ]
    sig_a, sig_b = [], []
    da, db = CuriosityDrive(seed=7), CuriosityDrive(seed=7)
    for k, ev in stream:
        sig_a.append((da.novelty(k), da.learning_progress(k), da.intrinsic_reward(k, ev)))
    for k, ev in stream:
        sig_b.append((db.novelty(k), db.learning_progress(k), db.intrinsic_reward(k, ev)))
    explore_repro = _run_explorer(CuriosityDrive(seed=2), rounds, True) == _run_explorer(
        CuriosityDrive(seed=2), rounds, True
    )

    print("=" * 78)
    print("INTRINSIC MOTIVATION -- learning-progress curiosity (anti-noisy-TV, self-directed)")
    print("=" * 78)
    print(
        f"alpha(fast)={d3.alpha}  alpha_slow={d3.alpha_slow:.3f}  novelty_decay={d3.novelty_decay}  "
        f"novelty_gain={d3.novelty_gain}"
    )
    print("-" * 78)
    print(
        f"(1) habituation : novelty(A) {nov_curve[0]:.3f} -> {nov_curve[-1]:.2e} over 30 observes; "
        f"unseen(B)={nov_unseen:.2f}"
    )
    print(
        f"(2) learn curve : learning_progress peaks {lp_peak:.3f} during decay -> {lp_plateau:.2e} once mastered"
    )
    print(
        f"(3) anti-noisyTV: learnable learning-progress {learn_lp:.3f}  vs  noisy {noisy_lp:.3f}  "
        f"(ratio {learn_lp / max(noisy_lp, 1e-9):.1f}x); curiosity reward learnable {learn_reward:.2f} "
        f"vs noisy {noisy_reward:.2f}"
    )
    print(
        f"                  noisy stays HIGH (err~{noisy_final_err:.2f}) yet learning_progress~{noisy_final_lp:.2e}"
    )
    print(f"(4) exploration : greedy visits {greedy_visits}  |  random visits {random_visits}")
    print(
        f"(5) determinism : identical signals={sig_a == sig_b}  explorer reproducible={explore_repro}"
    )
    print("=" * 78)

    # ---- (1) HABITUATION asserts ----
    assert all(nov_curve[i + 1] < nov_curve[i] for i in range(len(nov_curve) - 1)), (
        "novelty must STRICTLY decrease with repeated observation (habituation)"
    )
    assert nov_curve[-1] < 0.01, "a heavily-repeated context's novelty must converge toward ~0"
    assert nov_unseen == 1.0 and nov_unseen > nov_curve[-1], (
        "a never-seen context must stay maximally novel (well above the habituated one)"
    )

    # ---- (2) LEARNING PROGRESS tracks a learning curve ----
    assert lp_peak > 0.05, (
        "learning progress must be clearly POSITIVE while error is decaying (learning)"
    )
    assert lp_plateau < 0.01, (
        "learning progress must fall to ~0 once the context is mastered (error flat)"
    )
    assert lp_plateau < 0.05 * lp_peak, "mastered progress must be far below the learning peak"

    # ---- (3) ANTI-NOISY-TV -- the HARD, un-riggable assert ----
    assert noisy_final_err > 0.5, (
        "the noisy context's error must stay HIGH (a real noisy TV, not low error)"
    )
    assert noisy_final_lp < 0.02, (
        "high-but-flat noise must give ~0 learning progress (NOT stuck-high)"
    )
    assert learn_lp > 12.0 * noisy_lp, (
        f"curiosity must CHASE the learnable frontier, NOT the noise (learnable learning-progress {learn_lp:.3f} "
        f"must be >> noisy {noisy_lp:.3f})"
    )
    assert learn_reward > 12.0 * noisy_reward, (
        f"the curiosity REWARD must inherit anti-noisy-TV (learnable {learn_reward:.3f} >> noisy {noisy_reward:.3f})"
    )

    # ---- (4) EXPLORATION: greedy prefers the learnable-novel context; random does not ----
    assert greedy_visits["learnable"] > greedy_visits["mastered"], (
        "curiosity-greedy exploration must visit the learnable-novel context more than the mastered one"
    )
    assert greedy_visits["learnable"] > greedy_visits["noisy"], (
        "curiosity-greedy exploration must prefer the learnable context over the noisy (unlearnable) one"
    )
    assert greedy_visits["learnable"] > random_visits["learnable"], (
        "the curiosity-blind random control must NOT prefer the learnable context the way greedy does"
    )

    # ---- (5) DETERMINISM ----
    assert sig_a == sig_b, (
        "two CuriosityDrive(seed=7) fed the same stream must give identical signals"
    )
    assert explore_repro, "the seeded explorer must reproduce its visit counts exactly"

    print(
        f"self-check OK: novelty habituates ({nov_curve[0]:.2f}->{nov_curve[-1]:.1e}) while unseen stays 1.0; "
        f"learning progress peaks {lp_peak:.3f} then dies to {lp_plateau:.1e} on mastery;"
    )
    print(
        f"  ANTI-NOISY-TV: learnable curiosity reward {learn_reward:.3f} >> noisy {noisy_reward:.3f} "
        f"({learn_reward / max(noisy_reward, 1e-9):.0f}x) though noise stays high (err {noisy_final_err:.2f}, "
        f"lp {noisy_final_lp:.1e});"
    )
    print(
        f"  greedy explorer visits learnable {greedy_visits['learnable']}x vs noisy {greedy_visits['noisy']}x / "
        f"mastered {greedy_visits['mastered']}x (random {random_visits['learnable']}x); deterministic "
        f"(mean lp during learning {mean(lp_learning):.3f})"
    )


if __name__ == "__main__":
    main()
