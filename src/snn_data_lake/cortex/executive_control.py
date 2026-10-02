r"""
Executive control -- ACC CONFLICT MONITORING + EVC (expected value of control) + a model-based
SEQUENCE/TREE planner. The representation/value side of the prefrontal-cingulate executive:
it MONITORS competition, DECIDES how hard to think, and SEARCHES multi-step plans. It selects no
action and drives no hardware -- it emits scalars and a plan; the basal-ganglia gate + reflex/
E-STOP still own selection and actuation.

Why this exists. cortex/prefrontal_executive.py already HOLDS a goal and PLANS, but its plan()
caps at a short latent rollout that scores only the TERMINAL state of ONE constant action over a
tiny option set -- no anterior-cingulate conflict signal, no cost/benefit arbitration of control,
and no search over action SEQUENCES. This module adds those three missing executive faculties as
pure, deterministic functions over developer-supplied drives / value / transition estimates.

The three faculties (Botvinick, Braver, Cohen 2001; Shenhav, Botvinick, Cohen 2013):

  1. ACC CONFLICT (conflict). The anterior cingulate reads response conflict as the co-activation
     of competing action drives -- the Hopfield ENERGY of a response layer whose units mutually
     inhibit (E = sum_{i<j} p_i p_j on the normalized drives). It is HIGH when two drives are
     near-tied (both strongly active at once) and ~0 for a clear single winner (one-hot -> exactly
     0). Normalized to [0, 1] by the uniform-distribution maximum, so it is scale- and
     permutation-invariant: only the SHAPE of the competition matters, not the magnitude or order.
     (Normalized Shannon entropy of the same distribution tracks it monotonically -- cross-checked
     in the self-check.)

  2. EVC (evc). The expected value of exerting cognitive control = benefit(control) - cost(control).
     Control only pays off when there is conflict to resolve AND value at stake, so
     benefit = conflict * value_gain; the net EVC = benefit - cost gates HOW MUCH top-down bias to
     apply. When the net is <= 0 (low conflict, or the payoff does not clear the cost) control is
     WITHHELD (gain 0.0) -- you do not think hard about a decision that is either easy or not worth
     it. When positive, control is allocated by a bounded, saturating map in [0, max_gain)
     (diminishing returns of control).

  3. SEQUENCE / TREE PLANNER (plan_sequence). Given a tiny action set with injected value_fn(state)
     and transition_fn(state, action) estimates, search action SEQUENCES of length `depth` (breadth
     `breadth`, or exhaustive) and return the best-rollout sequence -- scoring the WHOLE trajectory
     (cumulative discounted value) or its terminal, not one terminal of one constant action. On a
     delayed-reward landscape this finds a multi-step plan that a 1-step greedy MISSES: it will
     accept an immediate LOSS to reach a later jackpot.

P6 -- WHAT THIS FACULTY CONSUMES AND WHAT CONSUMES IT, re-derived at runtime 2026-08-05 rather than
assumed, because the brief this pass started from said "read by nobody" and that is false:

  CONSUMES  faculties/jepa_world_model -- `plan_sequence` is called live with `transition_fn =
            jepa.predict` and `value_fn = jepa.make_goal_value(z_goal)`, so the search runs over
            another faculty's LEARNED latent world model, not over developer estimates.
            cortex/learned_evc -- the live `evc()` call takes its `cost` and `value_gain` from the
            cortex's learned three-factor EVC when that has been pushed, and only falls back to
            constants before it has.
  PRODUCES  `conflict(drives)` -> the ACTION path (the STN hyperdirect brake), cortex/learned_evc,
            and -- ADDED THIS PASS -- cortex/metacognitive_monitor, which now consumes it as
            DECISION uncertainty. That third reader is the one that satisfies P6 condition (b):
            the first is the single action gate P6 explicitly names as the DEFECT, not the remedy.

The measured charge of the new reader is in `cortex/metacognitive_monitor.py`'s docstring and is
re-proven by that module's self-check on every run: pin this faculty's `conflict` to a constant and
the monitor's own reliability changes on 295/300 live ticks with 38 HOLD flips. Both figures were
re-derived 2026-08-05 by a pass that did not write the wire, and both came back exact -- see that
docstring for the two harness conditions (natural circadian clock; neuter applied at the class
before tick 1) without which they do not reproduce. That pass also carried the charge one step
further than this block claimed: cutting the metacognition edge ALONE changes which route
`Brain.respond()` takes on 3 of 30 sampled turns and the reply text on 23 of 30, so this faculty's
output reaches spoken behaviour through its new reader and not only through the action gate.

Note what is NOT claimed: none of these routes goes through the shared GlobalWorkspace, so
`tools/p6_gate.py` -- which counts workspace verbs only -- cannot see any of them and reports this
module as a non-participant either way. That is a limit of the instrument, not of the wiring, and it
is stated here so the next reader does not "fix" a number by pushing these signals through a bus
that would only raise the WRITE count the gate itself says is not progress.

Learning rule: NONE by design -- no-gradient, no trained weights, no backprop. All three faculties
are FIXED dynamics (Hopfield-energy conflict, EVC arithmetic, deterministic search) over injected
value/transition estimates. This satisfies the brain-native constraint by preferring fixed dynamics
/ local computation over any gradient; there is no loss.backward() anywhere.

Honest framing / gaps. The conflict signal is the canonical Botvinick ENERGY read over the drive
distribution (a rate-level co-activation of spiking-derived drives), NOT itself a spiking pool.
Planning is over a TINY action set with DEVELOPER value/transition estimates -- it is bounded
model-based search, not general planning, and a too-narrow `breadth` can prune a delayed-reward
trajectory (an honest, demonstrated limitation). EVC's benefit/cost are scalar estimates the caller
supplies, not learned. This is executive CONTROL machinery, not cognition, agency, or consciousness.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.executive_control
"""

import math
import random


class ExecutiveControl:
    """ACC conflict monitoring + EVC control arbitration + a model-based sequence/tree planner.

    conflict(drives) -> [0,1] response conflict (Hopfield co-activation energy over competing drives).
    evc(conflict, value_gain, cost) -> [0, max_gain) top-down control gain (0 when control is not
        worth it: low conflict, or benefit does not clear cost).
    plan_sequence(state, actions, value_fn, transition_fn, depth) -> best action sequence over a tiny
        action set (multi-step rollout search, not one terminal of one constant action).

    Representation/value-side only: emits scalars + a plan, selects no action, drives no hardware.
    Deterministic; per-instance random.Random(seed) is retained for API symmetry / stochastic
    extensions (the core methods are deterministic and do not sample)."""

    def __init__(self, control_max=1.0, control_half_sat=0.5, seed=0):
        self.control_max = float(control_max)  # ceiling of the allocated top-down control gain
        self.control_half_sat = float(
            control_half_sat
        )  # net-EVC at which the gain reaches half its ceiling
        self.rng = random.Random(seed)  # per-instance rng (deterministic core; kept for symmetry)

    # ---- (1) ACC conflict monitoring --------------------------------------------------------------
    def conflict(self, drives):
        """Response conflict in [0, 1] -- the Botvinick Hopfield ENERGY of the competing action
        drives (co-activation of mutually-inhibiting response units), read over the normalized drive
        distribution. HIGH for near-tied drives, ~0 for a clear winner (one-hot -> exactly 0.0).
        Scale- and permutation-invariant; negative drives are clamped to 0 (drives are non-negative).
        Fewer than two options, or all-zero drive, -> 0.0 (no competition)."""
        vals = [d if d > 0.0 else 0.0 for d in drives]
        n = len(vals)
        total = sum(vals)
        if n < 2 or total <= 0.0:
            return 0.0
        p = [v / total for v in vals]  # normalize -> distribution over responses
        energy = 0.0
        for i in range(n):  # Hopfield energy of an all-to-all inhibitory layer
            pi = p[i]
            for j in range(i + 1, n):
                energy += pi * p[j]  # co-activation of each competing pair
        max_energy = (n - 1) / (2.0 * n)  # uniform distribution = maximum possible conflict
        return energy / max_energy  # in [0, 1]

    @staticmethod
    def drive_entropy(drives):
        """Normalized Shannon entropy in [0, 1] of the drive distribution -- an ALTERNATIVE conflict
        read (1 at a flat tie, 0 at a one-hot winner). Provided so the self-check can cross-validate
        that the Hopfield-energy conflict tracks the entropy of the same competition."""
        vals = [d if d > 0.0 else 0.0 for d in drives]
        n = len(vals)
        total = sum(vals)
        if n < 2 or total <= 0.0:
            return 0.0
        h = 0.0
        for v in vals:
            if v > 0.0:
                p = v / total
                h -= p * math.log(p)
        return h / math.log(n)  # divide by max entropy log(n) -> [0, 1]

    # ---- (2) expected value of control ------------------------------------------------------------
    def evc(self, conflict, value_gain, cost, max_gain=None, half_sat=None):
        """Expected value of control -> allocated top-down control gain in [0, max_gain).

        benefit = conflict * value_gain (control only pays off when there IS conflict to resolve AND
        value at stake); net EVC = benefit - cost. If net <= 0 the control is NOT worth it and is
        WITHHELD (returns 0.0): an easy decision (low conflict) or one whose payoff does not clear the
        cost gets no extra control. If net > 0, gain rises with the net through a bounded, saturating
        map (diminishing returns), reaching half of `max_gain` at net == `half_sat`."""
        mg = self.control_max if max_gain is None else float(max_gain)
        hs = self.control_half_sat if half_sat is None else float(half_sat)
        benefit = conflict * value_gain
        net = benefit - cost
        if net <= 0.0:  # control not worth exerting -> withhold it
            return 0.0
        return mg * (net / (net + hs))  # bounded saturating allocation in [0, mg)

    # ---- (3) model-based sequence / tree planner --------------------------------------------------
    @staticmethod
    def _rollout_value(state, sequence, value_fn, transition_fn, discount=1.0, terminal_only=False):
        """Score one action SEQUENCE by rolling transition_fn forward from `state`: cumulative
        discounted value_fn over the visited states, or (terminal_only) the value of the final state."""
        s = state
        total = 0.0
        last = value_fn(state)
        for d, a in enumerate(sequence):
            s = transition_fn(s, a)
            v = value_fn(s)
            total += (discount**d) * v
            last = v
        return last if terminal_only else total

    def plan_sequence(
        self,
        state,
        actions,
        value_fn,
        transition_fn,
        depth,
        breadth=None,
        discount=1.0,
        terminal_only=False,
    ):
        """Search action SEQUENCES of length `depth` for the best rollout and return that sequence.

        Expands a beam of partial rollouts one step at a time, keeping the top `breadth` by score
        (or ALL of them when breadth is None -> exhaustive tree search). Score is the cumulative
        discounted value_fn over the trajectory, or value_fn of the terminal state when terminal_only.
        Unlike a single constant-action terminal rollout, this evaluates whole action sequences, so on
        a delayed-reward landscape it can accept an immediate LOSS to reach a later jackpot -- a plan a
        1-step greedy misses. Deterministic (fixed action order breaks ties toward earlier actions).

        Honest limit: a too-narrow `breadth` prunes by intermediate score and can DISCARD a
        delayed-reward path whose partial sums are low; use breadth=None (exhaustive) or a wide beam
        on adversarial delayed-reward landscapes."""
        acts = list(actions)
        if depth < 1 or not acts:
            return []
        beam = [(0.0, state, [])]  # (score_so_far, state, sequence)
        for d in range(depth):
            cands = []
            for score, s, seq in beam:
                for a in acts:
                    s2 = transition_fn(s, a)
                    v = value_fn(s2)
                    if terminal_only:
                        rank = v  # prune toward high-value states
                        new_score = v  # (terminal score is the last state's value)
                    else:
                        new_score = score + (discount**d) * v
                        rank = new_score
                    cands.append((rank, new_score, s2, seq + [a]))
            cands.sort(key=lambda c: -c[0])  # stable sort -> deterministic tie-break
            kept = cands if breadth is None else cands[:breadth]
            beam = [(nc, s2, seq) for _rank, nc, s2, seq in kept]
        best = max(
            range(len(beam)), key=lambda i: beam[i][0]
        )  # highest final score (first on ties)
        return beam[best][2]


# ==================================================================================================
# Un-riggable self-check. Deterministic (seeded), ASCII-only. Every claim is paired with
# a CONTROL that FAILS (a clear winner ~0 conflict, an over-cost EVC withheld, a 1-step greedy and a
# random-sequence baseline that both MISS the delayed reward).
# ==================================================================================================

# --- a tiny delayed-reward "valley" world: you must accept two -1 steps to reach a +10 jackpot ---
GOAL = 3
LANDSCAPE = {0: 0.0, 1: -1.0, 2: -1.0, 3: 10.0}  # immediate value of LANDING on each position
ACTIONS = ["stay", "advance", "retreat"]  # fixed order: greedy tie -> "stay" (never advances)


def _value_fn(pos):
    """Immediate value of being at a position (the injected developer value estimate)."""
    return LANDSCAPE[pos]


def _transition_fn(pos, action):
    """Deterministic 1-D valley dynamics (the injected developer transition estimate)."""
    if action == "advance":
        return min(pos + 1, GOAL)
    if action == "retreat":
        return max(pos - 1, 0)
    return pos  # "stay"


def _greedy_sequence(state, actions, value_fn, transition_fn, depth):
    """1-step myopic control: at each step take the action with the best IMMEDIATE next-state value."""
    s = state
    seq = []
    for _ in range(depth):
        a = max(actions, key=lambda a: value_fn(transition_fn(s, a)))  # ties -> first (="stay")
        seq.append(a)
        s = transition_fn(s, a)
    return seq


def _brute_force_best(state, actions, value_fn, transition_fn, depth, discount=1.0):
    """Enumerate EVERY length-`depth` action sequence -> the true optimum (independent reference)."""
    best_seq, best_val = None, float("-inf")

    def rec(s, seq):
        nonlocal best_seq, best_val
        if len(seq) == depth:
            val = ExecutiveControl._rollout_value(state, seq, value_fn, transition_fn, discount)
            if val > best_val:
                best_val, best_seq = val, list(seq)
            return
        for a in actions:
            rec(transition_fn(s, a), seq + [a])

    rec(state, [])
    return best_seq, best_val


def main():
    ec = ExecutiveControl(seed=0)

    # ---- (1) ACC conflict: near-tied drives -> HIGH, clear winner -> ~0 (the failing control) ------
    near_tied = [1.0, 0.98, 0.05]  # two drives near-tied -> strong response conflict
    clear_winner = [1.0, 0.01, 0.01]  # one dominant drive -> essentially no conflict
    one_hot = [1.0, 0.0, 0.0]  # a pure winner -> exactly 0 conflict
    c_tied = ec.conflict(near_tied)
    c_clear = ec.conflict(clear_winner)
    c_one_hot = ec.conflict(one_hot)
    # permutation invariance: shuffling the SAME drive values must not change the conflict
    shuffled = list(near_tied)
    random.Random(123).shuffle(shuffled)
    c_shuffled = ec.conflict(shuffled)
    # monotonic: as the runner-up approaches the leader, conflict rises
    ramp = [ec.conflict([1.0, r, 0.02]) for r in (0.1, 0.4, 0.7, 0.99)]
    # degenerate control: a batch of clear-winner drives (one dominant + small others) stays ~0
    db = random.Random(7)
    degen_conflicts = [
        ec.conflict([1.0, db.uniform(0.0, 0.05), db.uniform(0.0, 0.05)]) for _ in range(400)
    ]
    degen_max = max(degen_conflicts)  # even the worst clear-winner stays well below the tie
    # entropy cross-check: energy-conflict and entropy agree on ordering (tied > clear)
    e_tied, e_clear = ec.drive_entropy(near_tied), ec.drive_entropy(clear_winner)

    # ---- (2) EVC: apply control only when conflict is high AND worth it; else WITHHOLD -------------
    g_worth = ec.evc(conflict=0.80, value_gain=1.0, cost=0.10)  # high conflict, cheap -> engage
    g_costly = ec.evc(
        conflict=0.80, value_gain=1.0, cost=1.00
    )  # SAME conflict, benefit < cost -> withhold
    g_loweasy = ec.evc(
        conflict=0.05, value_gain=1.0, cost=0.20
    )  # easy decision (low conflict) -> withhold
    g_more = ec.evc(conflict=0.90, value_gain=1.0, cost=0.10)  # more conflict -> more control
    g_less = ec.evc(conflict=0.50, value_gain=1.0, cost=0.10)  # less conflict -> less control
    # THE ENGAGE BOUNDARY, exercised through evc() itself. This line used to read
    #     boundary = 0.80 * 1.0 > 0.79
    # which is a literal arithmetic tautology evaluated before evc() is ever called: it asserted
    # that 0.8 > 0.79, not that the CONTROLLER engages there. It could not go red if evc() inverted
    # its comparison, dropped the rectifier, or ignored `cost` entirely. Straddle the boundary
    # instead -- benefit == conflict*value_gain == 0.80, so cost just below it must ENGAGE and cost
    # just above it must WITHHOLD.
    g_just_under = ec.evc(conflict=0.80, value_gain=1.0, cost=0.79)
    g_just_over = ec.evc(conflict=0.80, value_gain=1.0, cost=0.81)
    g_exact = ec.evc(conflict=0.80, value_gain=1.0, cost=0.80)  # net == 0 -> withheld
    boundary = g_just_under > 0.0 and g_just_over == 0.0 and g_exact == 0.0

    # ---- (3) sequence planner finds the delayed reward a 1-step greedy MISSES ----------------------
    depth = 3
    planned = ec.plan_sequence(0, ACTIONS, _value_fn, _transition_fn, depth)  # exhaustive
    planned_val = ExecutiveControl._rollout_value(0, planned, _value_fn, _transition_fn)
    greedy = _greedy_sequence(0, ACTIONS, _value_fn, _transition_fn, depth)
    greedy_val = ExecutiveControl._rollout_value(0, greedy, _value_fn, _transition_fn)
    brute_seq, brute_val = _brute_force_best(0, ACTIONS, _value_fn, _transition_fn, depth)
    # random-sequence control at chance
    rs = random.Random(42)
    rand_vals = [
        ExecutiveControl._rollout_value(
            0, [rs.choice(ACTIONS) for _ in range(depth)], _value_fn, _transition_fn
        )
        for _ in range(500)
    ]
    rand_mean = sum(rand_vals) / len(rand_vals)
    # terminal-only search reaches the same jackpot; determinism
    planned_terminal = ec.plan_sequence(
        0, ACTIONS, _value_fn, _transition_fn, depth, terminal_only=True
    )
    plan_det = ec.plan_sequence(0, ACTIONS, _value_fn, _transition_fn, depth) == ExecutiveControl(
        seed=9
    ).plan_sequence(0, ACTIONS, _value_fn, _transition_fn, depth)

    print("=" * 84)
    print("EXECUTIVE CONTROL -- ACC conflict monitoring + EVC + model-based sequence/tree planner")
    print("=" * 84)
    print(
        f"(1) ACC conflict : near-tied {near_tied} -> {c_tied:.3f} (HIGH); "
        f"clear winner {clear_winner} -> {c_clear:.3f}; one-hot -> {c_one_hot:.3f}"
    )
    print(
        f"    invariance   : shuffled drives -> {c_shuffled:.3f} (== near-tied); "
        f"monotonic runner-up ramp {[round(x, 3) for x in ramp]}"
    )
    print(
        f"    control      : worst of 400 clear-winner drives -> {degen_max:.3f} (<< tie); "
        f"entropy agrees (tied {e_tied:.3f} > clear {e_clear:.3f})"
    )
    print("-" * 84)
    print(
        f"(2) EVC          : conflict 0.80 cost 0.10 -> gain {g_worth:.3f} (ENGAGE); "
        f"cost 1.00 -> gain {g_costly:.3f} (WITHHELD, not worth it)"
    )
    print(
        f"    gating       : low conflict 0.05 -> gain {g_loweasy:.3f} (WITHHELD); "
        f"more conflict 0.90 -> {g_more:.3f} > 0.50 -> {g_less:.3f}"
    )
    print(
        f"    boundary     : benefit 0.80 vs cost 0.79 -> {g_just_under:.3f} (ENGAGE), "
        f"0.80 -> {g_exact:.3f}, 0.81 -> {g_just_over:.3f} (WITHHELD)"
    )
    print("-" * 84)
    print(f"(3) planner      : delayed-reward valley {LANDSCAPE}, depth {depth}")
    print(
        f"    planned seq  : {planned} -> value {planned_val:.1f} (== brute-force optimum {brute_val:.1f})"
    )
    print(
        f"    greedy MISSES: {greedy} -> value {greedy_val:.1f}; random-seq mean {rand_mean:.2f} (chance)"
    )
    print(f"    terminal-only: {planned_terminal}; deterministic across seeds: {plan_det}")
    print("=" * 84)

    # ---- un-riggable asserts --------------------------------------------------------------------

    # (1) conflict is high for near-tied, ~0 for a clear winner, exactly 0 for one-hot
    assert c_tied > 0.7, f"near-tied drives must give HIGH conflict ({c_tied:.3f})"
    assert c_clear < 0.15, f"a clear winner must give ~0 conflict ({c_clear:.3f})"
    assert c_one_hot == 0.0, f"a one-hot winner must give exactly 0 conflict ({c_one_hot:.3f})"
    assert c_tied > 4.0 * c_clear, "near-tied conflict must clearly exceed the clear-winner control"
    assert abs(c_shuffled - c_tied) < 1e-12, (
        "conflict must be permutation-invariant (shuffle control)"
    )
    assert ramp == sorted(ramp) and ramp[-1] > ramp[0], (
        f"conflict must rise monotonically as the runner-up approaches the leader ({ramp})"
    )
    assert degen_max < 0.5 * c_tied, (
        f"every clear-winner (degenerate) drive must stay well below the near-tie ({degen_max:.3f})"
    )
    assert e_tied > e_clear, (
        "entropy cross-check must agree: near-tied is higher-entropy than a clear winner"
    )

    # (2) EVC engages only when conflict is high AND the benefit clears the cost
    assert g_worth > 0.0, "high conflict with benefit > cost must ENGAGE control"
    assert g_costly == 0.0, "control whose benefit does not clear its cost must be WITHHELD (0.0)"
    assert g_loweasy == 0.0, "an easy (low-conflict) decision must get no extra control"
    assert g_more > g_less > 0.0, "control must scale with conflict in the worth-it region"
    assert g_worth < ec.control_max, "the control gain must stay bounded below its ceiling"
    assert boundary, (
        f"control must engage exactly when conflict*value_gain exceeds cost: cost 0.79 -> "
        f"{g_just_under:.3f} (>0), cost 0.80 -> {g_exact:.3f} (==0), cost 0.81 -> {g_just_over:.3f} (==0)"
    )

    # (3) the sequence planner finds the delayed reward that a 1-step greedy misses
    assert planned == ["advance", "advance", "advance"], (
        f"the planner must find the 3-step path to the jackpot ({planned})"
    )
    assert planned_val == brute_val == 8.0, (
        f"the planned value must equal the true optimum 8.0 ({planned_val} vs {brute_val})"
    )
    assert greedy_val == 0.0 and greedy != planned, (
        f"the 1-step greedy must MISS the delayed reward ({greedy} -> {greedy_val})"
    )
    assert planned_val > greedy_val, "the planner must beat the greedy baseline"
    assert planned_val > rand_mean + 4.0, (
        f"the planner must beat the random-sequence control at chance ({planned_val} vs {rand_mean:.2f})"
    )
    assert planned_terminal == ["advance", "advance", "advance"], (
        "terminal-only search must reach the same jackpot"
    )
    assert plan_det, "the planner must be deterministic"

    print(
        f"self-check OK: ACC conflict {c_tied:.2f} (near-tied) vs {c_clear:.2f} (clear winner) / "
        f"0.00 (one-hot), worst degenerate {degen_max:.2f};"
    )
    print(
        f"  EVC engages at cost 0.10 (gain {g_worth:.2f}) but WITHHOLDS at cost 1.00 and low conflict "
        f"(0.00); scales {g_less:.2f}->{g_more:.2f};"
    )
    print(
        f"  planner finds {planned} = {planned_val:.0f} (true optimum), greedy MISSES ({greedy_val:.0f}) "
        f"and random-seq is chance ({rand_mean:.2f}); no-gradient fixed dynamics; deterministic"
    )


if __name__ == "__main__":
    main()
