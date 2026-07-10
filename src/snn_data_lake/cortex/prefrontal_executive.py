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
  4. Top-down bias. top_down_bias(strength) emits a length-n_options preference vector: `strength`
     summed on every held goal (primary + sub-goals) and zero elsewhere (all zeros when nothing is
     held) -- the executive's top-down vote over options that action selection can read.
  5. Model-based deliberation (lookahead). plan(state, options, predictor, goal_value, horizon)
     scores each candidate option by rolling a forward model `predictor.predict(s, command)` forward
     `horizon` steps and evaluating `goal_value` on the terminal predicted state, then commits via
     the SAME drift-diffusion race (over the rollout values, not the immediate ones). A short horizon
     is reactive (the immediate one-step consequence); a long horizon is planned (the terminal
     consequence), so lookahead can reject a high-immediate-value TRAP that rolls into a bad state.
     The predictor is injected (any object exposing predict(state, command)) -- the class does not
     couple to a specific forward model.
  6. Richer working memory (goal hierarchy). Working memory is a bounded goal STACK of (index,
     activity) frames, not a single scalar. update() sets/gates the primary (base) goal exactly as
     before; push_subgoal(evidence) deliberates and pushes a sub-goal above it. maintain() decays the
     TOP frame only (lower goals are HELD while a sub-goal is pursued) and pops a faded frame,
     popping a sub-goal back to its parent. The single-goal API is unchanged: with no sub-goal
     pushed, .goal / update / maintain / top_down_bias behave exactly as the scalar version did.

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
                 bound=1.0, noise=1.0, dt=0.01, leak=0.2, inhibition=0.1, commit_races=25,
                 max_goals=4, seed=0):
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
        self.max_goals = max_goals              # bounded goal-stack depth (primary + sub-goals)
        self.rng = random.Random(seed)
        self._stack = []                        # goal stack: [[index, activity], ...] base .. top

    @property
    def goal(self):
        """The currently held goal index (top of the goal stack), or None if nothing is maintained."""
        return self._stack[-1][0] if self._stack else None

    @property
    def goals(self):
        """The full held goal hierarchy, base -> top, as a list of indices (empty if none held)."""
        return [frame[0] for frame in self._stack]

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
        """Deliberate over `evidence`, then GATE the write into working memory (the PRIMARY goal).

        Overwrites the primary (base) goal only if the new commitment's evidence margin over the best
        alternative exceeds gate_threshold (or nothing is held yet); otherwise the held goal
        persists. Returns the current goal index."""
        winner = self._stable_commit(evidence)
        alternatives = [evidence[j] for j in range(len(evidence)) if j != winner]
        margin = evidence[winner] - (max(alternatives) if alternatives else float("-inf"))
        if not self._stack or margin >= self.gate_threshold:
            if self._stack:
                self._stack[0] = [winner, 1.0]          # overwrite the primary goal (keep sub-goals)
            else:
                self._stack.append([winner, 1.0])       # first goal
        return self.goal

    def _rollout_value(self, state, command, predictor, goal_value, horizon):
        """Roll `predictor.predict(s, command)` forward `horizon` steps from `state` (at least one --
        the immediate one-step consequence) and score the TERMINAL predicted state with goal_value."""
        s = state
        for _ in range(max(1, horizon)):
            s = predictor.predict(s, command)
        return goal_value(s)

    def plan(self, state, options, predictor, goal_value, horizon=1, write=True):
        """Model-based deliberation: score each candidate in `options` by a `horizon`-step rollout of
        `predictor` from `state`, evaluate goal_value on the terminal predicted state, then commit via
        the SAME drift-diffusion race (over the rollout values). horizon=1 is reactive (the immediate
        one-step consequence); horizon>=2 is planned (the terminal consequence), so lookahead can
        reject a high-immediate-value trap. If `write`, the commitment gate-updates the primary goal
        exactly like update(). Returns (goal_index, mean_rt)."""
        values = [self._rollout_value(state, o, predictor, goal_value, horizon) for o in options]
        lo = min(values)
        drifts = [v - lo + 0.1 for v in values]         # shift rollout scores to positive race drifts
        votes = [0] * len(options)
        rts = []
        for _ in range(self.commit_races):
            w, rt = self.deliberate(drifts)
            votes[w] += 1
            rts.append(rt)
        goal = _argmax(votes)
        if write:
            alternatives = [drifts[j] for j in range(len(drifts)) if j != goal]
            margin = drifts[goal] - (max(alternatives) if alternatives else float("-inf"))
            if not self._stack or margin >= self.gate_threshold:
                if self._stack:
                    self._stack[0] = [goal, 1.0]
                else:
                    self._stack.append([goal, 1.0])
        return goal, sum(rts) / len(rts)

    def push_subgoal(self, evidence):
        """Deliberate a stable commitment over `evidence` and push it as a SUB-GOAL above the primary.

        The sub-goal becomes the active (top) goal; the parent is HELD (frozen) beneath it until the
        sub-goal fades and pops back to it. Bounded by max_goals (drops the oldest frame when full).
        Returns the current (sub-)goal index."""
        winner = self._stable_commit(evidence)
        if len(self._stack) >= self.max_goals:
            self._stack.pop(0)                          # bounded WM: drop the oldest (base) frame
        self._stack.append([winner, 1.0])
        return self.goal

    def maintain(self):
        """Advance one delay tick with NO input. Recurrent self-excitation decays the TOP goal's
        persistent activity (lower goals are HELD frozen while a sub-goal is pursued); a frame that
        falls below hold_threshold is popped -- a sub-goal pops back to its parent, and the last goal
        fades to None. Returns the currently held goal, or None once the stack empties."""
        if not self._stack:
            return None
        top = self._stack[-1]
        top[1] *= self.recurrence
        if top[1] < self.hold_threshold:
            self._stack.pop()                           # fade the top frame (pop back to the parent)
        return self.goal

    def top_down_bias(self, strength=1.0):
        """A length-n_options top-down preference vector: `strength` SUMMED over every held goal
        (primary + sub-goals) and 0.0 elsewhere -- the whole active goal hierarchy votes. All zeros
        when nothing is held."""
        bias = [0.0] * self.n_options
        for idx, _activity in self._stack:
            bias[idx] += strength
        return bias


def _argmax(values):
    """Index of the largest value (tie-break by lowest index)."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _mean_rt(pfc, evidence, n):
    """Mean deliberation rt (steps) over n raw deliberations of `evidence`."""
    return sum(pfc.deliberate(evidence)[1] for _ in range(n)) / n


def main():
    from snn_data_lake.cerebellum.forward_model import ForwardModel  # the injected lookahead model

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

    # ---- 5. model-based lookahead: reactive picks the immediate TRAP; planning rejects it --------
    # abstract plant s' = s + 0.2*(-0.5*s + u): a constant command u converges to steady state 2u.
    fm_trap = ForwardModel(a=-0.5, b=1.0, dt=0.2)
    target = 2.0
    def goal_value(s):                               # closer to the target state -> higher value
        return -abs(s - target)
    options = [1.0, 5.0, 0.2]                         # cmd 0 -> s*=2.0 (goal); cmd 1 -> s*=10 (trap)
    reactive = PrefrontalExecutive(n_options=len(options), seed=10)
    reactive_goal, reactive_rt = reactive.plan(0.0, options, fm_trap, goal_value, horizon=1)
    planned = PrefrontalExecutive(n_options=len(options), seed=10)
    planned_goal, planned_rt = planned.plan(0.0, options, fm_trap, goal_value, horizon=40)
    # one-step vs terminal values, to SHOW the trap is real (option 1 best now, worst later)
    immediate_vals = [reactive._rollout_value(0.0, o, fm_trap, goal_value, 1) for o in options]
    terminal_vals = [planned._rollout_value(0.0, o, fm_trap, goal_value, 40) for o in options]

    # ---- 5b. the lookahead composes with the REAL forward model + is deterministic --------------
    fm_chk = ForwardModel(a=-0.5, b=1.0, dt=0.2)
    manual = 0.0
    for _ in range(3):                               # the model's own forward chain, by hand
        manual = fm_chk.predict(manual, 1.0)
    via_rollout = PrefrontalExecutive(n_options=2, seed=0)._rollout_value(0.0, 1.0, fm_chk,
                                                                          lambda s: s, 3)
    def plan_scene(seed):
        p = PrefrontalExecutive(n_options=len(options), seed=seed)
        return p.plan(0.0, options, ForwardModel(a=-0.5, b=1.0, dt=0.2), goal_value, horizon=40)
    plan_det = plan_scene(9) == plan_scene(9)

    # ---- 5c. planning keeps the chronometric signature: a wider terminal gap commits faster -----
    wide = PrefrontalExecutive(n_options=2, seed=11)
    _wg, wide_rt = wide.plan(0.0, [1.0, 5.0], fm_trap, goal_value, horizon=40)     # terminals 2 vs 10
    narrow = PrefrontalExecutive(n_options=2, seed=11)
    _ng, narrow_rt = narrow.plan(0.0, [1.0, 1.05], fm_trap, goal_value, horizon=40)  # terminals 2 vs 2.1

    # ---- 6. richer working memory: a sub-goal over a primary, pops back on decay ----------------
    hier = PrefrontalExecutive(n_options=4, seed=8)
    hier.update([2.0, 0.1, 0.1, 0.1])                # primary goal 0
    primary_goal = hier.goal
    hier.push_subgoal([0.1, 0.1, 2.0, 0.1])          # sub-goal 2, pushed above the primary
    sub_goal = hier.goal
    hier_goals = list(hier.goals)                    # [0, 2] -- the held hierarchy
    hier_bias = hier.top_down_bias()                 # the whole hierarchy votes (nonzero on 0 and 2)
    popped_back = None                               # the sub-goal decays and pops back to the parent
    for _ in range(400):
        g = hier.maintain()
        if len(hier.goals) == 1:
            popped_back = g
            break
    after_pop = list(hier.goals)

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
    print("-" * 72)
    print(f"lookahead   : options {options} immediate values {[round(v, 2) for v in immediate_vals]} "
          f"-> terminal values {[round(v, 2) for v in terminal_vals]}")
    print(f"deliberation: reactive (h=1) commits option {reactive_goal} (the immediate-best TRAP); "
          f"planned (h=40) commits option {planned_goal} (the terminal-best)")
    print(f"chronometric: wide terminal gap rt {wide_rt:.1f} < narrow gap rt {narrow_rt:.1f} steps")
    print(f"hierarchy   : primary goal {primary_goal}; push sub-goal -> held {hier_goals}, bias "
          f"{hier_bias}; sub-goal decays -> pops back to primary {after_pop}")
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

    # (6) model-based lookahead changes behaviour: reactive takes the immediate trap, planning avoids it.
    assert _argmax(immediate_vals) == 1, \
        f"the trap must be the immediate-best option (values {immediate_vals})"
    assert _argmax(terminal_vals) == 0, \
        f"a different option must be the terminal-best (values {terminal_vals})"
    assert reactive_goal == 1, f"reactive deliberation (h=1) must take the immediate trap ({reactive_goal})"
    assert planned_goal == 0 and planned_goal != reactive_goal, \
        f"planning (h=40) must reject the trap and commit the terminal-best ({planned_goal})"

    # (7) the lookahead composes with the REAL forward model and is deterministic.
    assert via_rollout == manual, \
        f"the rollout must equal the forward model's own prediction chain ({via_rollout} vs {manual})"
    assert plan_det, "same seed must reproduce an identical plan (goal + rt)"

    # (8) planning keeps the chronometric signature: a wider terminal value gap commits faster.
    assert wide_rt < narrow_rt, \
        f"a wider terminal-value gap must commit faster ({wide_rt:.1f} vs {narrow_rt:.1f} steps)"

    # (9) richer working memory: a sub-goal sits above the primary, the hierarchy votes, it pops back.
    assert primary_goal == 0 and sub_goal == 2 and sub_goal != primary_goal, \
        f"push_subgoal must place a sub-goal above the primary ({hier_goals})"
    assert hier_goals == [0, 2], f"the held hierarchy must be primary then sub-goal ({hier_goals})"
    assert hier_bias[0] > 0 and hier_bias[2] > 0 and sum(1 for b in hier_bias if b > 0) == 2, \
        f"the top-down bias must vote for the whole held hierarchy ({hier_bias})"
    assert popped_back == primary_goal and after_pop == [primary_goal], \
        f"a decayed sub-goal must pop back to the held primary goal (got {popped_back}, stack {after_pop})"

    print(f"self-check OK: goal {committed} persists across delay (crippled -> {decayed}); "
          f"stability keeps A={goal_a} under weak B, flexibility switches to {after_strong};")
    print(f"  strongest evidence wins {wins[strongest]}/{n_win} and commits faster "
          f"({strong_rt:.1f} < {weak_rt:.1f} steps); top-down bias tracks the goal; deterministic")
    print(f"  lookahead: reactive takes trap option {reactive_goal}, planning commits option "
          f"{planned_goal} (terminal-best); sub-goal {sub_goal} over primary {primary_goal} pops back")


if __name__ == "__main__":
    main()
