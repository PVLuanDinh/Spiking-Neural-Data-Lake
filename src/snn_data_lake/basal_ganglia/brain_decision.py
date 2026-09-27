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

PROMOTED 2026-08-12 (#10 lane 4d): reasoning/brain_adapters/ -> basal_ganglia/, the
region whose charter (action selection, reward learning) is this module's identity --
`latent_q.py` is the VALUE table, this file is the SELECTION stage over it. Composition
verified at promotion: LatentQ and MetacognitiveMonitor are composed, never
re-implemented (no delta rule, no Q table, no reliability computation is redefined
here). Registered in REGIONS + run_selfchecks CORE; dormancy EXEMPT-permanent -- the consumer
(`instruments/developmental_loop_brain.py`, promoted the same arc) is itself
EXEMPT-permanent train-tier, and the old "converts to reachable when the consumer
promotes" clause was MEASURED FALSE at that promotion (reachable() seeds only from live
roots; see the dormancy_gate entry).

MEASURED AT PROMOTION, 2026-08-12 (the fusion A/B the ledger row owed; every figure
below is from the four-arm neuter A/B in this file's self-check, 10-20 seeds, and was
independently reproduced with a second workload construction before landing):

  * THE FUSION HEADLINE IS A RECORDED NULL AT THE SHIPPED RULE. Under the min-max
    weighted-sum rule, on a budgeted heterogeneous-learnability workload with measured
    headroom (oracle > round-robin), the VALUE channel captures 0.0% of available
    headroom at every alpha in [0.6, 1.0] -- the real arm and a value-DEAD arm (LatentQ
    reads pinned to 0.0 while update() keeps running) score BIT-IDENTICAL on all seeds.
    A knife-edge at alpha ~ 0.5 captures ~19-27% of headroom; alpha <= 0.45 is
    catastrophic (the value channel actively destroys performance). So the shipped
    alpha = 0.8 sits in the dead zone, and any "P1 fix" that derives alpha into
    [0.6, 1.0] reproduces today's behaviour exactly, while pinning alpha at the
    knife-edge would be tuning on the benchmark the module is scored on.
  * WHAT REMAINS REAL: LatentQ value learning charges at the component level (the
    self-check's neuter-reachable assert (b)), the harness is sensitive (neutering the
    CURIOSITY channel collapses the outcome by ~ -0.35), and the workload non-vacuity
    precondition has a reachable red (it fires on the old flat-gain workload).
  * THE EXPLOIT BRANCH WAS UNREACHABLE: monitor.explore() returned True on every
    measured cycle (20/20 x 5 seeds; the LIF pool's windowed rate never clears
    r_explore=0.30 on this workload), so the former alpha_exploit=0.2 was a value with
    no reachable state -- DELETED per P8 rather than kept as documentation of a state
    that cannot occur. Behaviour is unchanged: alpha was 0.8 on every measured cycle.
  * THE SOLE CONSUMER'S BYPASS WAS ADJUDICATED 2026-08-12 (the P2 this bullet used to
    record as open): developmental_loop_brain DELETED its NORD_EXEC flag and the fused
    choose() branch -- the planner is the one selection mechanism, decided on a
    measured tie (planner vs fused 0.5917 vs 0.5894 over 30 seeds at the shipped
    config, with a positive control separating a worse policy 29/30 in the same runs)
    plus this module's own recorded fusion null, re-confirmed by a permutation null on
    that workload. choose() keeps exactly ONE external call site: the consumer's
    mechanics-check split criterion (its ASSERT 4), executed by gate command #1.
    observe() still trains on the consumer's default path while nothing reads the
    table there -- recorded as presence-not-participation in the consumer's docstring
    and in this module's dormancy EXEMPT entry, not laundered.

  Disposition of the fusion rule itself, settled at that adjudication: the rule stays
  as the mechanics-check CONTROL (the split criterion needs a fused pick to disagree
  with), the null stays recorded, and per the criteria-measure-the-pathology lesson NO
  assert pins the dead zone (a fix would red it looking like a regression).

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

  PYTHONPATH=src python -m snn_data_lake.basal_ganglia.brain_decision
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
    gate (logged; see the module docstring for why it no longer switches alpha), plus
    grow_trigger() for readout neurogenesis. Deterministic given seed."""

    def __init__(self, n_actions=10, state_dim=10, seed=0):
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance rng (tie-breaks only)
        # digits 0..n_actions-1 are the LatentQ action NAMES (integers; .value duck-types them).
        # lr is inherited from LatentQ's own default -- restating it here was a second
        # site for one constant (P1); the constant lives and is swept in latent_q.py.
        self.q = LatentQ(tuple(range(self.n_actions)), self.state_dim, seed=seed)
        self.monitor = MetacognitiveMonitor(seed=seed)
        # Weight placed on CURIOSITY (vs value) in choose(). MEASURED 2026-08-12: under
        # the min-max rule the outcome is invariant across alpha in [0.6, 1.0] (real ==
        # value-DEAD, bit-identical per seed), i.e. this constant is not load-bearing at
        # its shipped value; values < 0.6 are measurably harmful except a knife-edge at
        # ~0.5 that would be tuning. The fusion-rule adjudication (docstring) owns it.
        self.alpha = 0.8

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
          4. combined[c] = alpha*cur_norm[c] + (1 - alpha)*val_norm[c], alpha = 0.8.
          5. argmax combined; exact ties broken by rng.choice (deterministic per seed).

        MEASURED honesty note (2026-08-12, the self-check's four-arm A/B): at alpha in
        [0.6, 1.0] this rule's output is decided by the curiosity channel alone -- the
        value channel's contribution to the argmax is exactly zero (real == value-DEAD,
        bit-identical). The former claim here that "both signals always contribute" was
        falsified by that measurement. Returns the chosen digit."""
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
        alpha = self.alpha
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
# Self-check. Deterministic, ASCII-only. Three parts:
#   PART 1 (contracts, shipped synthetic trajectory): (a) choose() returns a valid
#     candidate; (b) LatentQ values MOVE with observe() -- the component charge proof;
#     (c) metacog gates typed / reliability in [0,1]; (e) determinism.
#   PART 2 (grow_trigger DISCRIMINATION): the gate fires on novel+learnable episodes
#     and does NOT fire on decayed-novelty zero-progress repeats -- its documented
#     contract, with a reachable red on each side (swap the feeds and each assert
#     fires), replacing the old near-vacuous ">=1 and < CYCLES" count.
#   PART 3 (the owed fusion A/B, four-arm neuter on a budgeted heterogeneous-
#     learnability workload): asserts the WORKLOAD NON-VACUITY precondition (oracle >
#     round-robin; red reachable: fires on the old flat-gain workload) and the
#     HARNESS-SENSITIVITY positive control (curiosity-neuter collapses the outcome;
#     red reachable: a dead harness returns ~0). The value-channel charge itself is
#     PRINTED with the honest verdict -- a NULL at the shipped rule -- and deliberately
#     NOT asserted in either direction: asserting charge>0 is red today, and asserting
#     charge==0 would pin the pathology (a later fix would red it as a "regression").
# =====================================================================================

N_ACT = 10
STATE = 10
HARD = 7  # this digit never improves when studied (lp stays ~0)
CYCLES = 20
AB_SEEDS = 10
AB_CYCLES = 40
AB_RATES = [0.02 + i * (0.40 - 0.02) / 9.0 for i in range(10)]  # hidden learn rates


def _run_loop(seed):
    """Drive BrainDecision through CYCLES developmental cycles on a synthetic trajectory.
    Returns a dict of pure numbers/bools so two runs compare byte-for-byte (determinism)."""
    brain = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    competence = [0.05] * STATE  # everyone starts near-incompetent
    times_studied = [0] * N_ACT
    candidates = list(range(N_ACT))

    chose_valid = True
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
        "gate_types_ok": gate_types_ok,
        "rel_in_range": rel_in_range,
        "final_competence": competence,
        "hard_competence": competence[HARD],
        "chosen_seq": chosen_seq,
        "v_learn0": v_learn0,
        "v_learn1": v_learn1,
        "v_untouched": v_untouched,
    }


def _grow_discrimination(seed):
    """grow_trigger()'s documented contract (novel AND learnable), tested on isolated
    instances so the attribution is clean. Returns (fires_learnable, fires_stale)."""
    z = [0.1] * STATE
    # arm 1: novel, high-progress episodes -> the gate must fire at least once.
    b1 = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    fires_learnable = 0
    for _ in range(8):
        after = list(z)
        after[3] = min(1.0, z[3] + 0.3)
        b1.observe(3, z, after, 0.3, 1.0)  # lp high, novelty max
        if b1.grow_trigger():
            fires_learnable += 1
    # arm 2: decayed novelty, zero progress (the HARD-digit repeat shape) -> no fire.
    b2 = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    fires_stale = 0
    for _ in range(8):
        b2.observe(HARD, z, list(z), 0.0, 0.0)  # lp 0, novelty 0
        if b2.grow_trigger():
            fires_stale += 1
    return fires_learnable, fires_stale


def _ab_world(seed):
    """Per-seed hidden learn-rate assignment: the same 10 rates, shuffled."""
    rng = random.Random(1000 + seed)
    g = list(AB_RATES)
    rng.shuffle(g)
    return g


def _ab_arm(seed, value_dead=False, flat_curiosity=False, flat_gain=False):
    """One arm of the fusion A/B on the budgeted heterogeneous-learnability workload.
    value_dead pins the LatentQ READ surface to 0.0 while update() keeps running (the
    movement-is-not-activation neuter: the weights still move, nothing can read them)."""
    g = _ab_world(seed)
    brain = BrainDecision(n_actions=N_ACT, state_dim=STATE, seed=seed)
    if value_dead:
        brain.q.value = lambda z, a: 0.0
        brain.q.values = lambda z: {a: 0.0 for a in range(N_ACT)}
    comp = [0.05] * N_ACT
    times = [0] * N_ACT
    cands = list(range(N_ACT))
    for _ in range(AB_CYCLES):
        cur = [1.0] * N_ACT if flat_curiosity else [1.0 / (1.0 + times[c]) for c in cands]
        obj = brain.choose(cands, comp, cur)
        before = list(comp)
        if flat_gain:  # the OLD degenerate workload shape (identical gain everywhere)
            gain = min(0.15, 1.0 - comp[obj])
        else:
            gain = g[obj] * (1.0 - comp[obj])
        comp[obj] = min(1.0, comp[obj] + gain)
        novelty = 0.5 ** times[obj]
        times[obj] += 1
        brain.observe(obj, before, comp, gain, novelty)
    return sum(comp) / N_ACT


def _ab_oracle(seed, flat_gain=False):
    g = _ab_world(seed)
    comp = [0.05] * N_ACT
    for _ in range(AB_CYCLES):
        gains = [
            (min(0.15, 1.0 - comp[d]) if flat_gain else g[d] * (1.0 - comp[d]))
            for d in range(N_ACT)
        ]
        obj = max(range(N_ACT), key=lambda d: gains[d])
        comp[obj] = min(1.0, comp[obj] + gains[obj])
    return sum(comp) / N_ACT


def _ab_roundrobin(seed, flat_gain=False):
    g = _ab_world(seed)
    comp = [0.05] * N_ACT
    for t in range(AB_CYCLES):
        obj = t % N_ACT
        gain = min(0.15, 1.0 - comp[obj]) if flat_gain else g[obj] * (1.0 - comp[obj])
        comp[obj] = min(1.0, comp[obj] + gain)
    return sum(comp) / N_ACT


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
    print(f"    hard digit {HARD} stayed low (never learned)  : {r['hard_competence']:.3f}")
    print(f"(e) determinism: two seed=0 runs identical       : {r == r2}")

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

    # ---- (e) determinism ----
    assert r == r2, "two seed=0 runs must be byte-for-byte identical"

    # ---- (d) grow_trigger DISCRIMINATION (novel+learnable fires, stale repeat does not) ----
    fl, fs = _grow_discrimination(0)
    print(
        f"(d) grow_trigger discrimination: novel+learnable fired {fl}/8, "
        f"stale zero-progress fired {fs}/8"
    )
    assert fl >= 1, "grow_trigger must fire on novel high-progress episodes (contract half 1)"
    assert fs == 0, (
        "grow_trigger must NOT fire on decayed-novelty zero-progress repeats "
        f"(contract half 2; fired {fs}/8)"
    )

    # ---- PART 3: the owed fusion A/B (four arms, budgeted heterogeneous workload) ----
    print("-" * 78)
    oracle = [_ab_oracle(s) for s in range(AB_SEEDS)]
    rr = [_ab_roundrobin(s) for s in range(AB_SEEDS)]
    real = [_ab_arm(s) for s in range(AB_SEEDS)]
    dead = [_ab_arm(s, value_dead=True) for s in range(AB_SEEDS)]
    flat = [_ab_arm(s, flat_curiosity=True) for s in range(AB_SEEDS)]
    mo = sum(oracle) / AB_SEEDS
    mr = sum(rr) / AB_SEEDS
    mreal = sum(real) / AB_SEEDS
    mdead = sum(dead) / AB_SEEDS
    mflat = sum(flat) / AB_SEEDS
    charge = mreal - mdead
    identical = all(abs(a - b) < 1e-12 for a, b in zip(real, dead))
    print(
        f"fusion A/B (alpha={BrainDecision().alpha}, {AB_SEEDS} seeds x {AB_CYCLES} cycles): "
        f"oracle {mo:.4f} | round-robin {mr:.4f} | real {mreal:.4f} | value-DEAD {mdead:.4f}"
    )
    print(
        f"  value-channel charge {charge:+.4f} (real == dead bit-identical: {identical}) "
        f"-> VERDICT: recorded NULL at the shipped rule (see module docstring)"
    )
    print(
        f"  positive control: curiosity-neutered arm {mflat:.4f} "
        f"(delta {mflat - mreal:+.4f} -- the harness is sensitive)"
    )

    # workload NON-VACUITY precondition: the benchmark itself must have headroom.
    # Reachable red: the same assert run against the OLD flat-gain workload fires.
    assert mo > mr + 0.01, (
        f"A/B workload must have headroom (oracle {mo:.4f} vs round-robin {mr:.4f}) -- "
        f"a null on a degenerate workload would be meaningless"
    )
    mo_flat = sum(_ab_oracle(s, flat_gain=True) for s in range(AB_SEEDS)) / AB_SEEDS
    mr_flat = sum(_ab_roundrobin(s, flat_gain=True) for s in range(AB_SEEDS)) / AB_SEEDS
    assert mo_flat <= mr_flat + 0.01, (
        "the OLD flat-gain workload must show ~no headroom (the degeneracy proof that "
        f"licenses the new workload); got oracle {mo_flat:.4f} vs rr {mr_flat:.4f}"
    )

    # HARNESS-SENSITIVITY positive control: the same probe that returns ~0 for the value
    # channel must return a large negative for the curiosity channel. Reachable red: a
    # dead harness (or a workload where selection cannot matter) returns ~0 here too.
    assert mflat < mreal - 0.10, (
        f"curiosity-neuter must collapse the outcome (got {mreal:.4f} -> {mflat:.4f}); "
        f"if this fails the A/B harness itself cannot detect a dead channel"
    )

    print("=" * 78)
    print(
        "self-check OK: composition contracts + value-learning component charge + "
        "grow_trigger discrimination hold;"
    )
    print(
        "  the fusion A/B ran with its controls (workload non-vacuous, harness "
        "sensitive) and its value-channel verdict is a recorded NULL at the shipped "
        "rule -- see the module docstring for the adjudication that owns it"
    )


if __name__ == "__main__":
    main()
