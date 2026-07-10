"""
Prefrontal central executive -- working-memory maintenance + gated updating + evidence-based
deliberation producing a top-down bias over goals (the "center of thoughts and reasoning").

The prefrontal cortex holds a GOAL in working memory as persistent delay activity, UPDATES it by
gated deliberation over incoming evidence, and broadcasts a TOP-DOWN BIAS that downstream action
selection consumes. The deliberation is not reinvented here: it reuses the leaky competing
accumulator race (basal_ganglia/evidence_accumulation.RaceAccumulator), so a commitment is a
drift-diffusion bounded integration with both a winner AND a reaction time.

Mechanism (rate-level, deterministic, per-instance rng):
  1. Working memory as persistent activity. A held goal index carries a scalar activation a
     (a = 1.0 at commit) renewed on every delay tick by recurrent self-excitation: a <- recurrence
     * a. While `recurrence` is near 1 the activity self-sustains and the goal is HELD across a
     delay with NO input; it slowly decays, and once a falls below `hold_threshold` the goal fades
     to None. With recurrence ~ 0 (no recurrence) the SAME goal collapses to None within a tick --
     the recurrent gain, not the input, does the maintenance.
  2. Deliberation. update()/deliberate() map the length-n_options `evidence` vector DIRECTLY onto
     the drifts of a RaceAccumulator: each option integrates its own noisy evidence and the first
     to cross the bound is the commitment; larger drift (stronger evidence) crosses SOONER (shorter
     rt) -- the chronometric signature inherited from the race.
  3. Gated updating (stability vs flexibility). update() deliberates to a winner, then GATES the
     write into working memory: it overwrites the held goal ONLY if the new commitment's evidence
     margin over the best alternative exceeds `gate_threshold`; otherwise the held goal PERSISTS.
     Weak new evidence cannot dislodge a maintained goal (stability); strong new evidence can
     (flexibility).
  4. Top-down bias. top_down_bias(strength) emits a length-n_options preference vector, maximal on
     the held goal and zero elsewhere (all zeros when nothing is held) -- the executive's top-down
     vote over options that action selection can read.

Fair control (un-riggable):
  - persistence vs decay: after a commit, several maintain() delay ticks with NO input still return
    the goal; an otherwise-identical executive with recurrence=0 loses the SAME goal within a tick.
    The recurrence sustains the memory, not the stimulus.
  - stability vs flexibility: holding goal A, a WEAK evidence for B (margin below the gate) leaves
    the goal at A; a STRONG evidence for B (margin above the gate) switches it to B.
  - deliberation reuses evidence accumulation: over many trials the committed option is the
    highest-evidence option (the race winner), and stronger evidence commits FASTER (shorter rt) --
    both signatures come from the drift-diffusion race, not from a bespoke argmax.
  - determinism: two executives with the same seed reproduce identical goals, biases and reaction
    times; a different seed differs.

Honest framing: a reduced model of the prefrontal central executive -- working-memory maintenance
(persistent activity), gated updating (stability/flexibility), and evidence-based deliberation
(reusing the drift-diffusion race) producing a top-down bias over goals. It HOLDS and UPDATES a
goal and biases behaviour; it is NOT cognition, planning-in-depth, or consciousness. "Reasoning"
here means bounded evidence integration to a commitment, nothing more.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.prefrontal_executive
"""
import random

from snn_data_lake.basal_ganglia.evidence_accumulation import RaceAccumulator


class PrefrontalExecutive:
    """Central executive over n_options goals: persistent-activity working memory, gated updating,
    and evidence-based deliberation (reusing RaceAccumulator) emitting a top-down bias.

    Hold a goal across a delay with maintain(); update it by gated deliberation with update();
    read the raw drift-diffusion commitment with deliberate(); read the held goal via .goal; and
    read the top-down preference over options with top_down_bias()."""

    def __init__(self, n_options, gate_threshold=0.5, recurrence=0.98, hold_threshold=0.2,
                 bound=1.0, noise=1.0, dt=0.01, leak=0.2, inhibition=0.1, commit_races=25, seed=0):
        self.n_options = n_options
        self.gate_threshold = gate_threshold    # evidence margin needed to overwrite the held goal
        self.recurrence = recurrence            # recurrent self-excitation gain (~1 -> persistent)
        self.hold_threshold = hold_threshold    # persistent activity fades to None below this
        self.bound = bound                      # deliberation decision bound (race threshold)
        self.noise = noise                      # per-step evidence noise in the race
        self.dt = dt                            # integration step of the race
        self.leak = leak                        # accumulator leak (leaky competing accumulator)
        self.inhibition = inhibition            # lateral inhibition between accumulators
        self.commit_races = commit_races        # races averaged into a stable commit for a WM write
        self.rng = random.Random(seed)
        self._goal = None                       # currently held goal index (or None)
        self._activity = 0.0                    # persistent activity of the held goal

    @property
    def goal(self):
        """The currently held goal index, or None if nothing is maintained."""
        return self._goal

    def deliberate(self, evidence):
        """Raw drift-diffusion commitment over `evidence` (one drift per option), NO WM write.

        Reuses the leaky competing accumulator race: evidence maps directly onto the accumulator
        drifts. Returns (goal, rt) -- the winning option and the number of deliberation steps."""
        drifts = [float(e) for e in evidence]
        race = RaceAccumulator(drifts, bound=self.bound, noise=self.noise, dt=self.dt,
                               leak=self.leak, inhibition=self.inhibition,
                               seed=self.rng.randrange(1 << 30))
        return race.decide()

    def _stable_commit(self, evidence):
        """Deliberate to a STABLE commitment for a working-memory write: run commit_races races and
        take the majority winner (a WM write integrates more evidence than a reflexive deliberate).
        Returns the majority-winning option index."""
        votes = [0] * len(evidence)
        for _ in range(self.commit_races):
            winner, _rt = self.deliberate(evidence)
            votes[winner] += 1
        return _argmax(votes)

    def update(self, evidence):
        """Deliberate over `evidence`, then GATE the write into working memory.

        Overwrites the held goal only if the new commitment's evidence margin over the best
        alternative exceeds gate_threshold (or nothing is held yet); otherwise the held goal
        persists. Returns the current goal index."""
        winner = self._stable_commit(evidence)
        alternatives = [evidence[j] for j in range(len(evidence)) if j != winner]
        margin = evidence[winner] - (max(alternatives) if alternatives else float("-inf"))
        if self._goal is None or margin >= self.gate_threshold:
            self._goal = winner
            self._activity = 1.0
        return self._goal

    def maintain(self):
        """Advance one delay tick with NO input: recurrent self-excitation renews the persistent
        activity of the held goal. Returns the held goal, or None once the activity decays below
        hold_threshold (the goal fades out)."""
        if self._goal is None:
            return None
        self._activity *= self.recurrence
        if self._activity < self.hold_threshold:
            self._goal = None
            self._activity = 0.0
        return self._goal

    def top_down_bias(self, strength=1.0):
        """A length-n_options top-down preference vector: `strength` on the held goal and 0.0
        elsewhere. All zeros when no goal is held."""
        bias = [0.0] * self.n_options
        if self._goal is not None:
            bias[self._goal] = strength
        return bias


def _argmax(values):
    """Index of the largest value (tie-break by lowest index)."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _mean_rt(pfc, evidence, n):
    """Mean deliberation rt (steps) over n raw deliberations of `evidence`."""
    return sum(pfc.deliberate(evidence)[1] for _ in range(n)) / n


def main():
    n_opt = 4

    # ---- 1. working-memory persistence (persistent activity) vs crippled maintenance ----------
    pfc = PrefrontalExecutive(n_options=n_opt, seed=0)
    pfc.update([2.0, 0.1, 0.1, 0.1])                 # strong evidence for option 0 -> commit goal
    committed = pfc.goal
    held_trace = [pfc.maintain() for _ in range(6)]  # delay ticks with NO input
    persisted = pfc.goal

    crippled = PrefrontalExecutive(n_options=n_opt, recurrence=0.0, seed=0)  # no recurrence
    crippled.update([2.0, 0.1, 0.1, 0.1])
    crippled_trace = [crippled.maintain() for _ in range(3)]
    decayed = crippled.goal

    # ---- 2. gated updating: stability (weak B) vs flexibility (strong B) -----------------------
    gate = PrefrontalExecutive(n_options=3, seed=1)
    gate.update([2.0, 0.2, 0.2])                     # commit goal A = 0
    goal_a = gate.goal
    gate.update([0.5, 0.7, 0.4])                     # WEAK evidence for B (margin < gate) -> keep A
    after_weak = gate.goal
    gate.update([0.2, 2.0, 0.2])                     # STRONG evidence for B (margin > gate) -> B
    after_strong = gate.goal

    # ---- 3. deliberation reuses evidence accumulation: winner + chronometric rt ordering -------
    delib = PrefrontalExecutive(n_options=3, seed=2)
    competing = [0.3, 0.8, 1.8]                      # option 2 has the strongest evidence
    n_win = 60
    wins = [0, 0, 0]
    for _ in range(n_win):
        g, _rt = delib.deliberate(competing)
        wins[g] += 1
    strongest = _argmax(competing)
    weak_rt = _mean_rt(delib, [0.6, 0.1, 0.1], 30)   # weak dominant evidence -> slow commitment
    strong_rt = _mean_rt(delib, [2.2, 0.1, 0.1], 30)  # strong dominant evidence -> fast commitment

    # ---- 4. top-down bias reflects the goal ----------------------------------------------------
    biaser = PrefrontalExecutive(n_options=n_opt, seed=3)
    bias_none = biaser.top_down_bias()
    biaser.update([2.0, 0.1, 0.1, 0.1])              # goal 0
    bias_0 = biaser.top_down_bias()
    biaser.update([0.1, 0.1, 0.1, 2.0])              # goal 3
    bias_3 = biaser.top_down_bias()

    print("=" * 72)
    print("PREFRONTAL EXECUTIVE -- working memory + gated updating + deliberation -> top-down bias")
    print("=" * 72)
    print(f"options: {n_opt} | gate: {pfc.gate_threshold} | recurrence: {pfc.recurrence} | "
          f"hold threshold: {pfc.hold_threshold}")
    print(f"commit goal {committed}; after 6 delay ticks (no input) -> still held: {persisted} "
          f"(trace {held_trace})")
    print(f"crippled (recurrence=0): same goal after 3 ticks -> {decayed} (trace {crippled_trace})")
    print("-" * 72)
    print(f"gated updating: hold A={goal_a}; weak evidence for B -> {after_weak} (stays A); "
          f"strong evidence for B -> {after_strong} (switches to B)")
    print("-" * 72)
    print(f"deliberation over evidence {competing} ({n_win} trials): win counts {wins} -> "
          f"strongest option {strongest} wins {wins[strongest]}/{n_win}")
    print(f"chronometric: weak-evidence rt {weak_rt:.1f} steps vs strong-evidence rt "
          f"{strong_rt:.1f} steps -> stronger commits faster")
    print("-" * 72)
    print(f"top-down bias: no goal -> {bias_none}; goal 0 -> {bias_0}; goal 3 -> {bias_3}")
    print("=" * 72)

    # ---- un-riggable self-checks (deterministic asserts) --------------------------------------
    # (1) persistence: recurrence sustains the goal across a delay; crippled maintenance loses it.
    assert committed == 0, "a strong evidence commitment must set the goal"
    assert all(g == 0 for g in held_trace), "the goal must persist across delay ticks (no input)"
    assert persisted == committed, "the recurrence must sustain the held goal across the delay"
    assert decayed is None, "with no recurrence the same goal must decay to None within a tick"

    # (2) stability vs flexibility: weak new evidence cannot overwrite; strong new evidence can.
    assert goal_a == 0, "the executive must first commit goal A"
    assert after_weak == goal_a, "weak new evidence (margin below gate) must NOT overwrite goal A"
    assert after_strong == 1 and after_strong != goal_a, \
        "strong new evidence (margin above gate) must switch the goal to B"

    # (3) deliberation reuses the race: strongest evidence wins the majority; stronger is faster.
    assert wins[strongest] == max(wins) and wins[strongest] > n_win / 2, \
        f"the strongest-evidence option must win the deliberation majority ({wins})"
    assert strong_rt < weak_rt, \
        f"stronger evidence must commit faster ({strong_rt:.1f} vs {weak_rt:.1f} steps)"

    # (4) top-down bias reflects the held goal, switches with it, and is all-zeros when none held.
    assert bias_none == [0.0] * n_opt, "no held goal must yield an all-zero top-down bias"
    assert _argmax(bias_0) == 0, "the top-down bias must peak on the held goal (0)"
    assert _argmax(bias_3) == 3, "the top-down bias must peak on the held goal (3)"
    assert bias_0 != bias_3, "the top-down bias must switch when the goal switches"
    assert biaser.top_down_bias(strength=2.0)[3] == 2.0, "bias strength must scale the held option"

    # (5) determinism: same seed -> identical goals, biases, rts; a different seed differs.
    def scenario(seed):
        p = PrefrontalExecutive(n_options=3, seed=seed)
        out = [p.update([2.0, 0.2, 0.2]), p.maintain()]
        out.append(p.deliberate([0.5, 1.5, 0.3]))
        out.append(tuple(p.top_down_bias()))
        out.append(p.update([0.1, 0.1, 2.0]))
        return out

    same_a = scenario(5)
    same_b = scenario(5)
    diff_c = scenario(6)
    assert same_a == same_b, "same seed must reproduce identical goals, biases and rts"
    assert same_a != diff_c, "a different seed must differ"

    print(f"self-check OK: goal {committed} persists across delay (crippled -> {decayed}); "
          f"stability keeps A={goal_a} under weak B, flexibility switches to {after_strong};")
    print(f"  strongest evidence wins {wins[strongest]}/{n_win} and commits faster "
          f"({strong_rt:.1f} < {weak_rt:.1f} steps); top-down bias tracks the goal; deterministic")


if __name__ == "__main__":
    main()
