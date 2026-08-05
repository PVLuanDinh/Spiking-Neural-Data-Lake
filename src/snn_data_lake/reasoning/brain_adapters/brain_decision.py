r"""
brain_decision -- upgrade the developmental loop's OBJECTIVE-SELECTION from
curiosity+curriculum ALONE to a brain DECISION that also consults basal-ganglia
VALUE learning and a metacognitive controller.

Honest framing (read this first). Today the loop picks the next digit to study each
cycle with `SpikingCuriosityDrive.most_curious(candidates)` filtered by a
`SpikingCurriculum` -- a single signal (curiosity) plus a schedule. That answers
"what is novel / has learning-progress?" but NOT "what is it WORTH studying?" and NOT
"should I explore or exploit right now, and should I grow?". This adapter AUGMENTS
(never replaces) that with two REAL, shipped core modules:

  * basal_ganglia.latent_q.LatentQ   -- a shared linear action-value Q(z, a) that
    estimates the EXPECTED LEARNING VALUE of studying each digit a, given the current
    competence vector z. It is trained online from the REALISED learning
    (competence_after - competence_before) on the digit actually studied.
  * cortex.metacognitive_monitor.MetacognitiveMonitor -- a lesionable spiking
    reliability monitor whose gates decide explore-vs-exploit (explore()) and when to
    recruit capacity (grow_trigger()).

The loop KEEPS owning its curiosity drive + curriculum and passes their per-candidate
curiosity signal into choose(); this adapter only fuses value + metacognition on top.

P4c. This paragraph used to read "This is a RESEARCH-tier adapter: it lives outside src/ and
only imports/composes the shipped modules." The first clause was FALSE -- the file is at
`src/snn_data_lake/reasoning/brain_adapters/` -- and "research-tier" is not a disposition:
under P4c this adapter is part of the brain. What is true is the composition claim (it wraps
`basal_ganglia.latent_q` and `cortex.metacognitive_monitor`, re-implementing neither). Its open
debt: registered in neither `REGIONS` nor `run_selfchecks.py`, so no gate sees it, and its one
importer is `reasoning/developmental_loop_brain.py`.

Mappings (competence -> module inputs), stated once and used everywhere below:
  z   (LatentQ latent state)  = the length-`state_dim` competence vector, as-is.
  a   (LatentQ action)        = the digit index in 0..n_actions-1 (LatentQ action names
                                are the integers 0..n_actions-1, duck-typed by .value).
  target (LatentQ update)     = realised learning value on the studied digit =
                                competence_after[objective] - competence_before[objective]
                                (falls back to `learning_progress` if that delta is 0).
  monitor.confidence          = competence_after[objective]  (scalar competence AT the
                                studied digit; the monitor clamps it to [0,1]).
  monitor.surprise            = |learning_progress|          (clamped to [0,1]).
  monitor.learning_progress   = learning_progress            (>=0 inside the monitor).
  monitor.novelty             = novelty.

Deterministic: per-instance random.Random(seed); LatentQ and the
monitor are seeded from the same seed. ASCII-only prints.

  PYTHONPATH=src python -m snn_data_lake.reasoning.brain_adapters.brain_decision
"""

import random

from snn_data_lake.basal_ganglia.latent_q import LatentQ
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor


def _minmax(vals):
    """Min-max normalize a list to [0,1]; a flat list maps to all-0.5 (no preference)."""
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [0.5 for _ in vals]
    span = hi - lo
    return [(v - lo) / span for v in vals]


class BrainDecision:
    """Fuse basal-ganglia VALUE + metacognitive EXPLORE/EXPLOIT on top of the loop's
    own curiosity to pick the next study objective.

    The loop still runs its curiosity drive + curriculum; it passes the per-candidate
    curiosity score into choose(). This adapter adds (1) LatentQ value_scores -- the
    expected learning value of each digit given the current competence, learned online
    from realised learning in observe(); and (2) the metacognitive monitor's explore()
    gate to weight curiosity-vs-value, plus grow_trigger() for readout neurogenesis.
    Deterministic given seed."""

    def __init__(self, n_actions=10, state_dim=10, seed=0):
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance rng (tie-breaks only)
        # digits 0..n_actions-1 are the LatentQ action NAMES (integers; .value duck-types them).
        self.q = LatentQ(tuple(range(self.n_actions)), self.state_dim, lr=0.1, seed=seed)
        self.monitor = MetacognitiveMonitor(seed=seed)
        # weight placed on CURIOSITY (vs value) when the monitor says explore / exploit.
        self.alpha_explore = 0.8  # explore -> favor curiosity / novelty
        self.alpha_exploit = 0.2  # exploit -> favor learned value

    def _z(self, competence_vec):
        """Coerce the competence vector to a length-state_dim list of floats (the latent z)."""
        z = [float(v) for v in competence_vec]
        if len(z) != self.state_dim:
            raise ValueError(f"competence vector len {len(z)} != state_dim {self.state_dim}")
        return z

    def value_scores(self, competence_vec):
        """Expected learning value of studying each digit given competence_vec, as a
        length-n_actions list. Q(z, a) from LatentQ over z = the competence vector."""
        z = self._z(competence_vec)
        row = self.q.values(z)  # {digit: Q(z, digit)}
        return [row[a] for a in range(self.n_actions)]

    def choose(self, candidates, competence_vec, curiosity_scores):
        """Pick ONE study objective from `candidates`.

        Combination rule (documented exactly):
          1. val[c]  = LatentQ value Q(z, c) for each candidate c (z = competence_vec).
          2. cur[c]  = curiosity_scores[i], the loop's own per-candidate curiosity.
          3. Min-max normalize val and cur ACROSS the candidate set to [0,1] so a
             negative/large value and a raw curiosity become comparable.
          4. explore = monitor.explore(); alpha = 0.8 if explore else 0.2 (weight on
             curiosity). When the monitor says EXPLORE we favor curiosity/novelty; when
             EXPLOIT we favor the learned value.
          5. combined[c] = alpha*cur_norm[c] + (1 - alpha)*val_norm[c].
          6. argmax combined; exact ties broken by rng.choice (deterministic per seed).
        Both signals always contribute (alpha is 0.2..0.8, never 0 or 1), so neither the
        curiosity nor the value channel is ever ignored. Returns the chosen digit."""
        if not candidates:
            raise ValueError("choose() needs at least one candidate")
        cands = list(candidates)
        if len(curiosity_scores) != len(cands):
            raise ValueError("curiosity_scores must align 1:1 with candidates")
        z = self._z(competence_vec)
        val = [self.q.value(z, c) for c in cands]
        cur = [float(s) for s in curiosity_scores]
        val_n = _minmax(val)
        cur_n = _minmax(cur)
        explore = self.monitor.explore()
        alpha = self.alpha_explore if explore else self.alpha_exploit
        combined = [alpha * cur_n[i] + (1.0 - alpha) * val_n[i] for i in range(len(cands))]
        best = max(combined)
        tied = [cands[i] for i in range(len(cands)) if combined[i] >= best - 1e-12]
        return tied[0] if len(tied) == 1 else self.rng.choice(tied)

    def observe(self, objective, competence_before, competence_after, learning_progress, novelty):
        """Learn from the outcome of studying `objective` this cycle.

        (1) LatentQ.update(z=competence_before, a=objective, target=realised learning
            value) where target = competence_after[objective] - competence_before[objective]
            (the actually-realised gain; falls back to learning_progress if that delta is 0).
        (2) MetacognitiveMonitor.update with the mapped signals (see module docstring)."""
        obj = int(objective)
        zb = self._z(competence_before)
        za = self._z(competence_after)
        delta = za[obj] - zb[obj]
        target = delta if delta != 0.0 else float(learning_progress)
        self.q.update(zb, obj, target)
        lp = float(learning_progress)
        self.monitor.update(
            confidence=za[obj],  # scalar competence at the studied digit
            surprise=abs(lp),
            learning_progress=lp,
            novelty=float(novelty),
        )

    def grow_trigger(self):
        """True when the monitor says recruit capacity (novel AND learnable) -- the loop
        uses this to fire readout NEUROGENESIS."""
        return self.monitor.grow_trigger()

    def reliability(self):
        """Pass-through of the monitor's fused reliability in [0,1] (for logging)."""
        return self.monitor.reliability()

    def explore(self):
        """Pass-through of the monitor's explore-vs-exploit gate (for logging)."""
        return self.monitor.explore()


# =====================================================================================
# Self-check. Deterministic, ASCII-only. Simulate ~20 decision cycles on a synthetic
# competence trajectory (most digits improve when studied; one digit stays hard, lp~0),
# and ASSERT: (a) choose() returns a valid candidate every cycle; (b) LatentQ values MOVE
# with observe() -- a digit that yielded learning ends ABOVE an untouched digit; (c) the
# metacog gates return sane booleans / a reliability in [0,1]; (d) grow_trigger() fires at
# least once on a novel high-progress episode but NOT on every cycle.
# =====================================================================================

N_ACT = 10
STATE = 10
HARD = 7  # this digit never improves when studied (lp stays ~0)
CYCLES = 20


def _run_loop(seed):
    """Drive BrainDecision through CYCLES developmental cycles on a synthetic trajectory.
    Returns a dict of pure numbers/bools so two runs compare byte-for-byte (determinism)."""
    brain = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    competence = [0.05] * STATE  # everyone starts near-incompetent
    times_studied = [0] * N_ACT
    candidates = list(range(N_ACT))

    chose_valid = True
    grow_fires = []
    gate_types_ok = True
    rel_in_range = True
    chosen_seq = []

    for _ in range(CYCLES):
        # the loop's OWN curiosity signal: higher for less-studied digits (per candidate).
        curiosity = [1.0 / (1.0 + times_studied[c]) for c in candidates]
        obj = brain.choose(candidates, competence, curiosity)
        chosen_seq.append(obj)
        if obj not in candidates:
            chose_valid = False

        before = list(competence)
        # synthetic learning: learnable digits gain 0.15 (capped); the HARD digit gains 0.
        gain = 0.0 if obj == HARD else min(0.15, 1.0 - competence[obj])
        competence[obj] = competence[obj] + gain
        after = list(competence)
        lp = gain  # realised learning progress this cycle
        # novelty: high on a digit's first visits, decays with how often it has been studied.
        novelty = 0.5 ** times_studied[obj]
        times_studied[obj] += 1

        brain.observe(obj, before, after, lp, novelty)

        g = brain.grow_trigger()
        grow_fires.append(bool(g))
        if not (isinstance(brain.explore(), bool) and isinstance(g, bool)):
            gate_types_ok = False
        r = brain.reliability()
        if not (isinstance(r, float) and 0.0 <= r <= 1.0):
            rel_in_range = False

    # ---- (b) VALUE LEARNING: a fixed z, study a learnable digit repeatedly with a positive
    # realised gain, never touch another digit -> the studied digit's Q rises above it. This
    # isolates that LatentQ.update() wiring actually moves the value (independent of choose()).
    vb = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    z_fixed = [0.2] * STATE
    d_learn, d_untouched = 2, 9
    v_learn0 = vb.value_scores(z_fixed)[d_learn]
    for _ in range(12):
        after = list(z_fixed)
        after[d_learn] = z_fixed[d_learn] + 0.4  # realised gain -> positive target
        vb.observe(d_learn, z_fixed, after, 0.4, 0.5)
    scores = vb.value_scores(z_fixed)
    v_learn1 = scores[d_learn]
    v_untouched = scores[d_untouched]

    return {
        "chose_valid": chose_valid,
        "n_grow_fires": sum(grow_fires),
        "gate_types_ok": gate_types_ok,
        "rel_in_range": rel_in_range,
        "final_competence": competence,
        "hard_competence": competence[HARD],
        "chosen_seq": chosen_seq,
        "v_learn0": v_learn0,
        "v_learn1": v_learn1,
        "v_untouched": v_untouched,
    }


def main():
    r = _run_loop(0)
    r2 = _run_loop(0)  # determinism control

    print("=" * 78)
    print("BRAIN DECISION -- curiosity + basal-ganglia VALUE + metacognitive control")
    print("=" * 78)
    print(f"cycles={CYCLES} | n_actions={N_ACT} | state_dim={STATE} | hard_digit={HARD}")
    print("-" * 78)
    print(f"(a) choose() always returned a valid candidate : {r['chose_valid']}")
    print(
        f"(b) value learning: Q(d={2}) {r['v_learn0']:+.4f} -> {r['v_learn1']:+.4f} "
        f"vs untouched Q(d={9})={r['v_untouched']:+.4f}"
    )
    print(f"    => studied-digit value rose above untouched : {r['v_learn1'] > r['v_untouched']}")
    print(
        f"(c) metacog gates typed / reliability in [0,1]  : "
        f"{r['gate_types_ok']} / {r['rel_in_range']}"
    )
    print(
        f"(d) grow_trigger fired {r['n_grow_fires']}/{CYCLES} cycles "
        f"(>=1 and < {CYCLES}, i.e. not constant)"
    )
    print(f"    hard digit {HARD} stayed low (never learned)  : {r['hard_competence']:.3f}")
    print(f"(e) determinism: two seed=0 runs identical       : {r == r2}")
    print("=" * 78)

    # ---- (a) every choose() returned a candidate that was offered ----
    assert r["chose_valid"], "choose() must always return one of the offered candidates"

    # ---- (b) LatentQ value MOVED with observe(): learned digit > untouched digit ----
    assert r["v_learn1"] > r["v_learn0"], (
        f"studying a digit that yielded learning must raise its value "
        f"({r['v_learn0']:.4f} -> {r['v_learn1']:.4f})"
    )
    assert r["v_learn1"] > r["v_untouched"], (
        f"the learned digit's value {r['v_learn1']:.4f} must exceed the untouched "
        f"digit's value {r['v_untouched']:.4f} -- value learning is real, not shared"
    )

    # ---- (c) metacog gates are sane booleans and reliability is a valid probability ----
    assert r["gate_types_ok"], "explore()/grow_trigger() must return real booleans"
    assert r["rel_in_range"], "reliability() must be a float in [0,1] every cycle"

    # ---- (d) grow_trigger fired on SOME novel/learnable episode but not constantly ----
    assert r["n_grow_fires"] >= 1, "grow_trigger must fire at least once on novel high-progress"
    assert r["n_grow_fires"] < CYCLES, "grow_trigger must NOT fire on every cycle (not constant)"

    # ---- (e) determinism ----
    assert r == r2, "two seed=0 runs must be byte-for-byte identical"

    print(
        f"self-check OK: choose() fuses curiosity + LatentQ value under the metacog "
        f"explore/exploit gate;"
    )
    print(
        f"  value learning moved Q(studied) {r['v_learn0']:+.4f}->{r['v_learn1']:+.4f} "
        f"above untouched {r['v_untouched']:+.4f}; grow_trigger fired "
        f"{r['n_grow_fires']}/{CYCLES} (novel+learnable only); deterministic"
    )


if __name__ == "__main__":
    main()
