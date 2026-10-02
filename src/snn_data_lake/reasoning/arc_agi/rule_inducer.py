"""RELATIONAL GRID RULE-INDUCER -- an ARC-AGI-style reasoner that INDUCES the causal change-rules of
a grid game from interaction, then PLANS with them, so it GENERALIZES where a tabular-state agent
explodes.

WHAT THIS IS
------------
A deterministic agent that plays the ARC-AGI-3-STYLE toy games in `env.py`. Instead of
memorizing (exact-grid-state -> value) like a tabular RL agent, it:

  1. PARSES each colored integer grid into OBJECTS = connected components of same color (color, cell
     set, bbox, centroid). All reasoning happens over OBJECTS, not pixels -- this symbolic
     abstraction is what makes the learned rules grid-size- and layout-INDEPENDENT.
  2. INDUCES relational change-rules from a stream of (grid, action, grid') transitions, each with
     SUPPORT + CONFIDENCE:
       - ACTION SEMANTICS : the controllable object (the "agent") TRANSLATES by a per-action (dr,dc).
       - COLLECT/DISAPPEAR: an object of color C VANISHES when the agent reaches it -> a boolean
         EVENT `collected(C)`.
       - PUSH (physics)   : an object of color P TRANSLATES by the agent's delta when the agent moves
         into it (precondition: the far cell is in-bounds and clear).
       - DEPENDENCY       : an object of color X BLOCKS the agent UNLESS event Y has occurred
         (e.g. the door is solid until the key is collected) -- an "effect only after event"
         precondition, DISCOVERED from the diffs, never hardcoded per game.
       - GOAL/REWARD      : which colored-cell entry (by the agent, or by a pushed object) yields the
         terminal reward, and under which event precondition.
     The rules are stated in terms of COLORS + RELATIVE DISPLACEMENTS + EVENTS, so they transfer to a
     bigger grid or a never-seen layout. The HIGHEST-CONFIDENCE rule set is kept. NOTHING is
     hardcoded per game -- shuffle the action->effect pairing and no correct rule is induced.
  3. PREDICTS grid' from (grid, action) via the induced rules, PLANS toward the goal by value
     iteration over the LEARNED (relational, imagined) model, and SELECTS each action with a genuine
     spiking `faculties.evidence_accumulation.RaceAccumulator` race over the planned values.

Reused repo modules (genuine, load-bearing):
  - faculties.evidence_accumulation.RaceAccumulator : spiking winner-take-all action selection.
  - cortex.vsa_reasoning (bind/bundle/cleanup/hv)        : the induced ACTION SEMANTICS are stored and
    recalled as spike-polarity role-filler hypervectors (bind(action, delta) bundled; queried by
    spike XOR + coincidence cleanup). The planner reads deltas THROUGH this spike-code store.
  - env.py (KeyDoor/PushBox/ColorLock/ToggleToGoal)      : the verification suite.
  The held-out / memoriser / shuffle CONTROL methodology mirrors cortex.reasoning_curriculum's
  un-riggable protocol (train on observed, score on never-seen, each claim paired with a failing
  control); its dependency-ordering is the length-1 degenerate case of that module's transitive chain.

THE UN-RIGGABLE PROOF (generalization, not coverage)
----------------------------------------------------
For >=2 of the toy games (key_door, push_box) we LEARN the rules on SMALL instances, then TEST on
HELD-OUT layouts (same size, new seeds) and SCALED-UP instances (bigger grids), with ZERO test-time
learning:
  (1) the RULE-INDUCER SOLVES the held-out/scaled instances -- the relational rule TRANSFERS (parse
      the new grid, plan with the transferred model).
  (2) CONTROL -- a TABULAR-STATE agent (the repo's own BrainPlayer: keys value on exact grid states,
      SAME small-instance interaction budget) FAILS the held-out/scaled instances: the test states
      were never visited, so its value table has no entry and it degrades to chance. Asserted:
      rule-inducer solve-rate >> tabular solve-rate.
  (3) CONTROL -- SHUFFLED transitions (action labels permuted, destroying the action->effect
      pairing): no coherent rule is induced -> the planner's model is wrong -> solve-rate collapses
      to chance.

HONEST FRAMING (this is the whole point, stated plainly)
--------------------------------------------------------
These TOY games have learnable relational rules BY CONSTRUCTION -- that is exactly what lets
induction win, and it is why a rule that is learned ONCE transfers to every scaled instance while a
tabular table must re-pay the full exploration cost per instance (and explodes as the grid grows).
This does NOT mean the real frontier ARC-AGI-3 games are solved: those need richer rule classes and
long horizons this inducer does not cover (compositional multi-step causation, novel object types,
non-Markov structure). The honest claim is ONLY: "relational rule induction generalizes where
tabular coverage explodes, on learnable-rule games." No claim is made about the live ARC-AGI-3
arcade. A caveat also holds for the tabular control: given a FRESH, LARGE budget on each test
instance it would of course solve these tiny MDPs -- the point is that it does not GENERALIZE across
instances, so under a fixed transfer budget it fails while the relational rule does not.

Constraints: per-instance random.Random(seed); zlib.crc32 (never hash()); no wall-clock
/ no global RNG; ASCII-only prints; deterministic (same seed -> byte-identical run).

  PYTHONUTF8=1 PYTHONPATH=src:. python research/arc_agi/rule_inducer.py
  (Windows bash: if the ':' path separator fails, use  PYTHONPATH="src;."  ; the file also
   bootstraps sys.path itself, so a bare `python research/arc_agi/rule_inducer.py` works.)
"""

import os
import random
import sys
import zlib
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root
for _p in (os.path.join(_ROOT, "src"), _ROOT, _HERE):
    if _p not in sys.path:
        pass  # migrated: sys.path bootstrap removed (now an importable package)

from snn_data_lake.faculties.evidence_accumulation import RaceAccumulator  # noqa: E402
from snn_data_lake.cortex.vsa_reasoning import bind, bundle, cleanup, hv  # noqa: E402

import env as arc_env  # noqa: E402
from brain_player import BrainPlayer, random_player  # noqa: E402

EMPTY = arc_env.EMPTY


# =====================================================================================================
# 1. PARSE -- colored integer grid -> OBJECTS (connected components of same color).
# =====================================================================================================

_NEI = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 4-connectivity


def connected_components(grid):
    """Every maximal 4-connected same-color region as (color, frozenset(cells))."""
    R, C = len(grid), len(grid[0])
    seen = [[False] * C for _ in range(R)]
    comps = []
    for r in range(R):
        for c in range(C):
            if seen[r][c]:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in _NEI:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < R and 0 <= nx < C and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            comps.append((color, frozenset(cells)))
    return comps


def parse_objects(grid, background=EMPTY):
    """Symbolic scene: a list of OBJECTS (color, cells, bbox, centroid, size), background dropped."""
    objs = []
    for color, cells in connected_components(grid):
        if color == background:
            continue
        rs = [y for y, x in cells]
        cs = [x for y, x in cells]
        objs.append(
            {
                "color": color,
                "cells": cells,
                "size": len(cells),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "centroid": (sum(rs) / len(cells), sum(cs) / len(cells)),
            }
        )
    return objs


def cells_by_color(grid):
    """color -> set of cells (includes the background color; callers skip EMPTY where needed)."""
    d = defaultdict(set)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            d[v].add((r, c))
    return d


def _color_counts(grid):
    cnt = Counter()
    for row in grid:
        for v in row:
            cnt[v] += 1
    return cnt


def sig_of(grid):
    """Exact hashable grid signature (what a TABULAR agent keys value on)."""
    return tuple(tuple(row) for row in grid)


# =====================================================================================================
# 2. COLLECT transitions -- Go-Explore full coverage of the reachable (state, action) graph.
# =====================================================================================================


def collect_transitions(game, budget=6000):
    """Systematically take every action from every reachable state (via snapshot/restore) and record
    (grid, action, grid', reward, done). Deterministic; terminates when the reachable graph is fully
    expanded or the budget is hit -- so the inducer sees the blocked-door, key-collection,
    door-with-key, all four push directions, and blocked-push transitions it needs."""
    trans = []
    snap = {}
    tried = {}
    states = set()
    frontier = []

    def register(sg, snp, terminal):
        if sg not in states:
            states.add(sg)
            tried[sg] = set()
            if not terminal:
                snap[sg] = snp
                frontier.append(sg)

    g0 = game.reset()
    register(sig_of(g0), game.snapshot(), False)
    steps = 0
    while frontier and steps < budget:
        s = frontier.pop(0)
        while len(tried[s]) < game.n_actions and steps < budget:
            a = min(x for x in range(game.n_actions) if x not in tried[s])
            before = game.restore(snap[s])
            g2, r, done = game.step(a)
            trans.append((before, a, g2, r, done))
            tried[s].add(a)
            steps += 1
            register(sig_of(g2), game.snapshot(), done)
    return trans


def shuffle_transitions(trans, seed):
    """CONTROL: permute the ACTION labels across the stream, destroying the action->effect pairing.
    The (grid, grid') diffs are untouched, but which action caused which effect is now scrambled."""
    rng = random.Random(seed)
    actions = [a for _g, a, _g2, _r, _d in trans]
    rng.shuffle(actions)
    return [(g, actions[i], g2, r, d) for i, (g, _a, g2, r, d) in enumerate(trans)]


# =====================================================================================================
# 3. INDUCE -- relational change-rules with support / confidence (the rules, NEVER hardcoded).
# =====================================================================================================


class _ActionSemantics:
    """The induced action->delta map, stored + recalled in the SPIKING VSA substrate (vsa_reasoning).
    Each `bind(hv(ACT:a), hv(DELTA:dr:dc))` role-filler is bundled into one spike code; recall binds
    the query action back out and snaps to the nearest delta by spike coincidence (cleanup). At this
    tiny load recall is lossless, so the planner reading deltas through the spike store is faithful --
    yet it is genuinely the spike-code algebra, and a shuffled (incoherent) delta map recalls wrongly."""

    def __init__(self, deltas):
        self.deltas = dict(deltas)  # {action: (dr,dc)}
        uniq = set(self.deltas.values())
        self._codebook = {"%d:%d" % (dr, dc): hv("DELTA:%d:%d" % (dr, dc)) for dr, dc in uniq}
        if self.deltas:
            self._bundle = bundle(
                [
                    bind(hv("ACT:%d" % a), self._codebook["%d:%d" % d])
                    for a, d in self.deltas.items()
                ]
            )
        else:
            self._bundle = None

    def recall(self, a):
        """Spike-XOR unbind + coincidence cleanup -> the delta the induced rule assigned to action a."""
        if a not in self.deltas or self._bundle is None:
            return None
        lab, _ = cleanup(bind(self._bundle, hv("ACT:%d" % a)), self._codebook)
        dr, dc = (int(x) for x in lab.split(":"))
        return (dr, dc)


class RuleSet:
    """The kept (highest-confidence) relational change-rules for one game, plus their support/conf."""

    def __init__(self):
        self.agent_color = None
        self.deltas = {}  # action -> (dr,dc)   (induced action semantics)
        self.delta_conf = {}  # action -> confidence
        self.collectibles = set()  # colors that vanish on agent contact -> event `collected(C)`
        self.pushables = set()  # colors that translate by the agent delta when entered
        self.push_goal = {}  # pushable color -> target color that yields the reward
        self.dependency = {}  # obstacle color -> required event color (block unless present)
        self.goal_agent = {}  # color -> required event color (agent entry -> reward/done)
        self.support = {}  # rule tag -> support count (evidence)
        self.semantics = None  # _ActionSemantics (VSA store)

    def delta(self, a):
        """Read the action's delta THROUGH the spiking VSA store (falls back to the raw map)."""
        d = self.semantics.recall(a) if self.semantics else None
        return d if d is not None else self.deltas.get(a)


def induce(trans, min_conf=0.5):
    """Induce the relational change-rule set from a transition stream. Every rule comes from the
    object-level DIFF between grid and grid'; support/confidence are counted, not assumed."""
    rs = RuleSet()

    # --- (a) agent color: the object whose single cell displaces under actions most often ----------
    disp = Counter()
    for g, _a, g2, _r, _d in trans:
        cb, cb2 = cells_by_color(g), cells_by_color(g2)
        for color, cells in cb.items():
            if color == EMPTY or len(cells) != 1:
                continue
            c2 = cb2.get(color)
            if c2 and len(c2) == 1 and next(iter(cells)) != next(iter(c2)):
                disp[color] += 1
    if not disp:
        return rs
    rs.agent_color = max(disp, key=lambda c: (disp[c], -c))
    ac = rs.agent_color

    # --- (b) action semantics: modal nonzero agent displacement per action -------------------------
    per_action = defaultdict(Counter)
    for g, a, g2, _r, _d in trans:
        cb, cb2 = cells_by_color(g), cells_by_color(g2)
        if len(cb.get(ac, ())) == 1 and len(cb2.get(ac, ())) == 1:
            p1, p2 = next(iter(cb[ac])), next(iter(cb2[ac]))
            per_action[a][(p2[0] - p1[0], p2[1] - p1[1])] += 1
    for a, cnt in per_action.items():
        nz = {d: n for d, n in cnt.items() if d != (0, 0)}
        if nz:
            d = max(nz, key=lambda k: (nz[k], k))
            rs.deltas[a] = d
            rs.delta_conf[a] = nz[d] / sum(nz.values())
        else:
            rs.deltas[a] = (0, 0)
            rs.delta_conf[a] = 0.0
    rs.semantics = _ActionSemantics(rs.deltas)

    # --- (c) collect / push / push-goal from agent-entry diffs --------------------------------------
    collect_sup, push_sup, pushgoal_sup = Counter(), Counter(), Counter()
    for g, a, g2, r, _d in trans:
        d = rs.deltas.get(a)
        cb, cb2 = cells_by_color(g), cells_by_color(g2)
        if not d or len(cb.get(ac, ())) != 1 or len(cb2.get(ac, ())) != 1:
            continue
        p1, p2 = next(iter(cb[ac])), next(iter(cb2[ac]))
        if (p2[0] - p1[0], p2[1] - p1[1]) != d:  # agent did not translate by this action -> skip
            continue
        tgt_color = g[p2[0]][p2[1]]  # what stood on the cell the agent entered
        if tgt_color in (EMPTY, ac):
            continue
        cg, cg2 = _color_counts(g), _color_counts(g2)
        far = (p2[0] + d[0], p2[1] + d[1])
        pushed = 0 <= far[0] < len(g) and 0 <= far[1] < len(g[0]) and far in cb2.get(tgt_color, ())
        if pushed:  # the object moved on by the same delta
            push_sup[tgt_color] += 1
            if r > 0.0:
                pushgoal_sup[(tgt_color, g[far[0]][far[1]])] += 1
        elif cg2[tgt_color] < cg[tgt_color]:  # the object vanished on contact
            collect_sup[tgt_color] += 1
    rs.collectibles = {c for c, n in collect_sup.items() if n >= 1}
    rs.pushables = {c for c, n in push_sup.items() if n >= 1}
    for (p, t), n in pushgoal_sup.items():
        if p in rs.pushables and t not in (EMPTY, ac):
            rs.push_goal[p] = t
            rs.support["pushgoal:%d->%d" % (p, t)] = n
    for c in rs.collectibles:
        rs.support["collect:%d" % c] = collect_sup[c]
    for c in rs.pushables:
        rs.support["push:%d" % c] = push_sup[c]

    # --- (d) dependency + agent-goal: entry into color X blocked unless event Y (then maybe reward) --
    def before_events(g):
        cg = _color_counts(g)
        return frozenset(
            c for c in rs.collectibles if cg[c] == 0
        )  # collectible absent => collected

    entry = defaultdict(list)  # X -> [(moved, events, reward, done)]
    for g, a, g2, r, done in trans:
        d = rs.deltas.get(a)
        cb, cb2 = cells_by_color(g), cells_by_color(g2)
        if not d or len(cb.get(ac, ())) != 1:
            continue
        p1 = next(iter(cb[ac]))
        tr = (p1[0] + d[0], p1[1] + d[1])
        if not (0 <= tr[0] < len(g) and 0 <= tr[1] < len(g[0])):
            continue  # boundary block is not an object dependency
        X = g[tr[0]][tr[1]]
        if X in (EMPTY, ac):
            continue
        moved = len(cb2.get(ac, ())) == 1 and next(iter(cb2[ac])) == tr
        entry[X].append((moved, before_events(g), r, done))
    for X, lst in entry.items():
        blocked = [e for m, e, _r, _dn in lst if not m]
        entered = [(e, r, dn) for m, e, r, dn in lst if m]
        if blocked and entered:
            for Y in rs.collectibles:
                if all(Y in e for e, _r, _dn in entered) and all(Y not in e for e in blocked):
                    rs.dependency[X] = Y
                    rs.support["dep:%d<-%d" % (X, Y)] = len(blocked)
                    if any(r > 0.0 and dn for _e, r, dn in entered):
                        rs.goal_agent[X] = Y
                    break
        elif entered and not blocked:  # unconditional agent-goal (no precondition)
            wins = [(e, r, dn) for e, r, dn in entered if r > 0.0 and dn]
            if wins:
                common = set.intersection(*[set(e) for e, _r, _dn in wins]) if wins else set()
                rs.goal_agent[X] = next(iter(common)) if common else None

    # a color better explained as a DEPENDENCY obstacle (door) or a PUSH-GOAL target (target) is NOT
    # a collectible -- it only "vanished" because the agent/box occluded it. Drop those spurious flags.
    rs.collectibles -= set(rs.dependency) | set(rs.push_goal.values())

    # keep only confident action semantics (a shuffled stream fails this -> unreliable model)
    rs.support["agent_color"] = disp[ac]
    rs.mean_delta_conf = (
        (sum(rs.delta_conf.values()) / len(rs.delta_conf)) if rs.delta_conf else 0.0
    )
    rs.confident = rs.mean_delta_conf >= min_conf
    return rs


# =====================================================================================================
# 4. PREDICT + PLAN -- an imagined relational world model, value iteration, spiking action select.
# =====================================================================================================


class InducedModel:
    """The transferred relational model instantiated on ONE test grid. Instance CONSTANTS (obstacle,
    collectible, target positions; bounds) are PARSED from the test grid; the DYNAMICS are the induced
    rules. A symbolic state is (agent_pos, frozenset(events), frozenset((color,pos) for movables)) --
    tiny and generated fresh per instance, which is why the model transfers where a table cannot."""

    def __init__(self, rules, grid, n_actions):
        self.rules = rules
        self.nA = n_actions
        self.R, self.C = len(grid), len(grid[0])
        cb = cells_by_color(grid)
        self.agent0 = next(iter(cb[rules.agent_color])) if cb.get(rules.agent_color) else None
        self.cellcolor = {(r, c): grid[r][c] for r in range(self.R) for c in range(self.C)}
        self.obstacles = {
            p: grid[p[0]][p[1]] for p in {q for X in rules.dependency for q in cb.get(X, ())}
        }
        self.collectible_pos = {
            p: grid[p[0]][p[1]] for p in {q for C in rules.collectibles for q in cb.get(C, ())}
        }
        self.movable0 = frozenset(
            (color, p) for color in rules.pushables for p in cb.get(color, ())
        )

    def initial_state(self, grid):
        return self.state_from_grid(grid)

    def state_from_grid(self, grid):
        cb = cells_by_color(grid)
        if len(cb.get(self.rules.agent_color, ())) != 1:
            return None
        apos = next(iter(cb[self.rules.agent_color]))
        cg = _color_counts(grid)
        events = frozenset(c for c in self.rules.collectibles if cg[c] == 0)
        movs = frozenset((color, p) for color in self.rules.pushables for p in cb.get(color, ()))
        return (apos, events, movs)

    def _in(self, p):
        return 0 <= p[0] < self.R and 0 <= p[1] < self.C

    def step(self, state, a):
        """Predict (state', reward, done) for taking action a from a symbolic state via the rules."""
        apos, events, movs = state
        d = self.rules.delta(a)
        if not d or d == (0, 0):
            return state, 0.0, False
        tr = (apos[0] + d[0], apos[1] + d[1])
        if not self._in(tr):
            return state, 0.0, False
        movpos = {p: color for color, p in movs}

        if tr in movpos:  # push a movable object
            P = movpos[tr]
            far = (tr[0] + d[0], tr[1] + d[1])
            if (
                not self._in(far)
                or far in movpos
                or far in self.obstacles
                or far in self.collectible_pos
            ):
                return state, 0.0, False  # blocked push
            new_movs = set(movs)
            new_movs.discard((P, tr))
            new_movs.add((P, far))
            reward, done = 0.0, False
            if P in self.rules.push_goal and self.cellcolor.get(far) == self.rules.push_goal[P]:
                reward, done = 1.0, True
            return (tr, events, frozenset(new_movs)), reward, done

        if tr in self.obstacles:  # dependency obstacle (e.g. door)
            need = self.rules.dependency.get(self.obstacles[tr])
            if need is not None and need not in events:
                return state, 0.0, False  # solid until the event fires

        new_events = events
        if tr in self.collectible_pos and self.collectible_pos[tr] not in events:
            new_events = frozenset(set(events) | {self.collectible_pos[tr]})

        destcolor = self.cellcolor.get(tr)
        reward, done = 0.0, False
        if destcolor in self.rules.goal_agent:
            need = self.rules.goal_agent[destcolor]
            if need is None or need in new_events:
                reward, done = 1.0, True
        return (tr, new_events, movs), reward, done

    def value_iterate(self, init, gamma=0.95, iters=1500, cap=80000):
        """Model-based value iteration over the reachable symbolic set imagined from `init`."""
        states = {init}
        frontier = [init]
        edges = {}
        while frontier and len(states) <= cap:
            s = frontier.pop()
            outs = []
            for a in range(self.nA):
                ns, r, done = self.step(s, a)
                outs.append((a, ns, r, done))
                if not done and ns not in states:
                    states.add(ns)
                    frontier.append(ns)
            edges[s] = outs
        V = dict.fromkeys(states, 0.0)
        for _ in range(iters):
            delta = 0.0
            for s in states:
                best = 0.0
                for _a, ns, r, done in edges.get(s, ()):
                    val = r if done else r + gamma * V.get(ns, 0.0)
                    if val > best:
                        best = val
                if abs(best - V[s]) > 1e-12:
                    delta = max(delta, abs(best - V[s]))
                    V[s] = best
            if delta < 1e-9:
                break
        Q = {}
        for s in states:
            qs = [0.0] * self.nA
            for a, ns, r, done in edges.get(s, ()):
                qs[a] = r if done else r + gamma * V.get(ns, 0.0)
            Q[s] = qs
        return Q


def _race_action(qvals, seed, race_scale, nA):
    """Pick an action by a genuine spiking evidence-accumulation race over the planned action-values
    (same normalization brain_player uses: values -> [0, race_scale] drifts, then a spiking race)."""
    mn, mx = min(qvals), max(qvals)
    span = mx - mn
    drifts = [race_scale * (v - mn) / span for v in qvals] if span > 1e-9 else [0.0] * len(qvals)
    acc = RaceAccumulator(drifts, bound=1.0, noise=0.2, dt=0.01, seed=seed, max_steps=150)
    winner, _rt = acc.decide()
    return winner


def solve_with_rules(rules, game, seed, race_scale=6.0):
    """PLAY one instance with the transferred rules: parse -> value-iterate the imagined model ->
    execute, choosing each action with the spiking RaceAccumulator. Solved iff a rewarding terminal
    is reached within game.max_steps. No learning happens here (pure transfer + planning)."""
    if rules.agent_color is None or not rules.deltas:
        return False
    grid = game.reset()
    model = InducedModel(rules, grid, game.n_actions)
    s = model.state_from_grid(grid)
    if s is None:
        return False
    Q = model.value_iterate(s)
    rng = random.Random(seed)
    for step in range(game.max_steps):
        qv = Q.get(s)
        if qv is None or max(qv) <= 0.0:
            a = rng.randrange(game.n_actions)  # off the planned frontier -> fall back
        else:
            rseed = (zlib.crc32(repr(s).encode()) ^ (step * 2654435761)) & 0xFFFFFFFF
            a = _race_action(qv, rseed, race_scale, game.n_actions)
        grid, r, done = game.step(a)
        if done:
            return r > 0.0
        s = model.state_from_grid(grid)
        if s is None:
            return False
    return False


# =====================================================================================================
# 5. TABULAR CONTROL -- the repo's own BrainPlayer, trained on small, applied to the test (no transfer).
# =====================================================================================================


def train_tabular(game_cls, small_size, small_seeds, seed=0, budget=8000):
    """Train BrainPlayer (memorized transitions + value iteration, keyed on EXACT grid states) on the
    same small instances the inducer learned from. Its table holds only small-instance states."""
    tab = BrainPlayer(
        game_cls(seed=small_seeds[0], size=small_size).n_actions
        if _accepts_size(game_cls)
        else game_cls(seed=small_seeds[0]).n_actions,
        seed=seed,
        explore_budget=budget,
    )
    for s in small_seeds:
        tab._explore(_mk(game_cls, s, small_size))
    tab._plan()
    return tab


def tabular_solve(tab, game, seed):
    """Apply the trained tabular table to a test instance. Unseen states -> random fallback."""
    tab.rng = random.Random(seed)
    return tab._exploit(game)


# =====================================================================================================
# helpers for size-parameterized construction
# =====================================================================================================


def _accepts_size(game_cls):
    return game_cls in (arc_env.KeyDoor, arc_env.PushBox, arc_env.ColorLock)


def _mk(game_cls, seed, size):
    return game_cls(seed=seed, size=size) if _accepts_size(game_cls) else game_cls(seed=seed)


def _mkc(game_cls, seed, size, cap_factor):
    """A test instance whose step budget is LINEAR in the grid size (cap_factor * size), applied
    uniformly to EVERY agent. A planned solution has length O(size); solving by wandering / state
    coverage needs O(size^2). This linear budget is exactly what the plan clears and coverage cannot
    -- so it is the honest instrument that turns 'coverage explodes' into a measurable solve gap."""
    g = _mk(game_cls, seed, size)
    g.max_steps = cap_factor * size
    return g


# =====================================================================================================
# 6. PROOF HARNESS -- learn small, test held-out + scaled; rule-inducer vs tabular vs shuffle.
# =====================================================================================================


def _rate(fn, instances):
    return sum(1 for inst in instances if fn(inst)) / len(instances)


def _random_rate(instances_factory, seeds, reps=8):
    hits = tot = 0
    for i, mk in enumerate(instances_factory):
        for r in range(reps):
            hits += 1 if random_player(mk(), 7919 * i + r) else 0
            tot += 1
    return hits / tot


def run_game_proof(
    game_cls, small_size, small_seeds, held_seeds, scaled_sizes, scaled_seeds, cap_factor
):
    name = game_cls.name

    # ---- LEARN the rules on the SMALL instances (full budget; no cap during learning) ----
    trans = []
    for s in small_seeds:
        trans += collect_transitions(_mk(game_cls, s, small_size))
    rules = induce(trans)
    rules_shuf = induce(shuffle_transitions(trans, seed=12345))
    tab = train_tabular(game_cls, small_size, small_seeds)

    # ---- build the test sets: held-out (same size) + scaled (bigger); LINEAR step budget for all ----
    held = [(s, small_size) for s in held_seeds]
    scaled = [(s, sz) for sz in scaled_sizes for s in scaled_seeds]

    def inducer(seed, size, ruleset):
        return solve_with_rules(
            ruleset,
            _mkc(game_cls, seed, size, cap_factor),
            seed=zlib.crc32(("%s:%d:%d" % (name, seed, size)).encode()) & 0x7FFFFFFF,
        )

    def solve_ind(cfg):
        return inducer(cfg[0], cfg[1], rules)

    def solve_shuf(cfg):
        return inducer(cfg[0], cfg[1], rules_shuf)

    def solve_tab(cfg):
        return tabular_solve(
            tab,
            _mkc(game_cls, cfg[0], cfg[1], cap_factor),
            seed=zlib.crc32(("T:%d:%d" % (cfg[0], cfg[1])).encode()) & 0x7FFFFFFF,
        )

    res = {
        "ind_held": _rate(solve_ind, held),
        "ind_scaled": _rate(solve_ind, scaled),
        "tab_held": _rate(solve_tab, held),
        "tab_scaled": _rate(solve_tab, scaled),
        "shuf_held": _rate(solve_shuf, held),
        "shuf_scaled": _rate(solve_shuf, scaled),
        "rnd_held": _random_rate(
            [(lambda s=s, sz=sz: _mkc(game_cls, s, sz, cap_factor)) for s, sz in held], held
        ),
        "rnd_scaled": _random_rate(
            [(lambda s=s, sz=sz: _mkc(game_cls, s, sz, cap_factor)) for s, sz in scaled], scaled
        ),
        "rules": rules,
    }
    return res


def _fmt_rules(rules):
    ds = " ".join("%d:(%d,%d)" % (a, d[0], d[1]) for a, d in sorted(rules.deltas.items()))
    dep = " ".join("%d<-collected(%d)" % (x, y) for x, y in sorted(rules.dependency.items()))
    pg = " ".join("%d->target%d" % (p, t) for p, t in sorted(rules.push_goal.items()))
    return "agent=%s deltas[%s] collect=%s push=%s dep=[%s] goal=%s pushgoal=[%s] conf=%.2f" % (
        rules.agent_color,
        ds,
        sorted(rules.collectibles),
        sorted(rules.pushables),
        dep,
        sorted(rules.goal_agent),
        pg,
        rules.mean_delta_conf,
    )


def main():
    print("=" * 90)
    print("RELATIONAL GRID RULE-INDUCER -- induce causal change-rules, plan, and GENERALIZE")
    print("REPRESENTATIVE ARC-AGI-3-style toy games (env.py), NOT the official ARC-AGI-3 eval.")
    print("Claim: relational rule induction generalizes where TABULAR coverage explodes, on")
    print("learnable-rule games. No claim about the live ARC-AGI-3 arcade.")
    print("=" * 90)

    # symbolic abstraction: a grid is parsed into OBJECTS (connected components), reasoned over as
    # (color, centroid, size) -- not pixels. This is what makes the induced rules grid-size agnostic.
    demo = arc_env.KeyDoor(seed=1, size=5).reset()
    objs = parse_objects(demo)
    print(
        "object parse (example key_door 5x5): %d objects -> %s"
        % (
            len(objs),
            ", ".join(
                "color%d@(%.0f,%.0f)size%d"
                % (o["color"], o["centroid"][0], o["centroid"][1], o["size"])
                for o in objs
            ),
        )
    )

    # (game, small_size, small_seeds, held_seeds, scaled_sizes, scaled_seeds, linear-step-budget factor)
    configs = [
        (
            arc_env.KeyDoor,
            5,
            [0, 1, 2, 3],
            list(range(100, 112)),
            [8, 10],
            list(range(200, 208)),
            6,
        ),
        (
            arc_env.PushBox,
            5,
            [0, 1, 2, 3],
            list(range(100, 112)),
            [7, 9],
            list(range(200, 208)),
            10,
        ),
    ]

    results = {}
    for game_cls, ssz, sseeds, hseeds, scsz, scseeds, capf in configs:
        res = run_game_proof(game_cls, ssz, sseeds, hseeds, scsz, scseeds, capf)
        results[game_cls.name] = res
        r = res["rules"]
        print("\n[%s]  learned on %d small seeds (size %d)" % (game_cls.name, len(sseeds), ssz))
        print("  induced rules: %s" % _fmt_rules(r))
        print(
            "  test sets    : held-out=%d layouts (size %d) | scaled=%d layouts (sizes %s)"
            % (len(hseeds), ssz, len(scsz) * len(scseeds), scsz)
        )
        print(
            "  %-22s %10s %10s %10s %10s"
            % ("solve-rate", "inducer", "tabular", "shuffle", "random")
        )
        print(
            "  %-22s %10.2f %10.2f %10.2f %10.2f"
            % (
                "held-out (same size)",
                res["ind_held"],
                res["tab_held"],
                res["shuf_held"],
                res["rnd_held"],
            )
        )
        print(
            "  %-22s %10.2f %10.2f %10.2f %10.2f"
            % (
                "SCALED-UP (bigger)",
                res["ind_scaled"],
                res["tab_scaled"],
                res["shuf_scaled"],
                res["rnd_scaled"],
            )
        )

    # ---- determinism: a second full run reproduces byte-identical solve rates ----
    det_ok = True
    for game_cls, ssz, sseeds, hseeds, scsz, scseeds, capf in configs:
        res2 = run_game_proof(game_cls, ssz, sseeds, hseeds, scsz, scseeds, capf)
        a = results[game_cls.name]
        for k in ("ind_held", "ind_scaled", "tab_scaled", "shuf_scaled"):
            det_ok = det_ok and abs(a[k] - res2[k]) < 1e-12

    print("\n" + "=" * 90)
    print("VERDICT (un-riggable proof)")
    print("=" * 90)

    # ---- assertions: each headline paired with a control that FAILS ----
    for gname, res in results.items():
        ind_s, tab_s, shuf_s = res["ind_scaled"], res["tab_scaled"], res["shuf_scaled"]
        ind_h, tab_h = res["ind_held"], res["tab_held"]
        print(
            "[%s] scaled: inducer %.2f  >>  tabular %.2f (transfer fails: unseen states)  "
            "shuffle %.2f (no rule induced)" % (gname, ind_s, tab_s, shuf_s)
        )
        print("           held-out: inducer %.2f  vs  tabular %.2f" % (ind_h, tab_h))
        # (1) the rule-inducer solves the held-out and scaled instances.
        assert ind_h >= 0.85, "%s: held-out inducer solve-rate must be high (got %.2f)" % (
            gname,
            ind_h,
        )
        assert ind_s >= 0.85, "%s: SCALED inducer solve-rate must be high (got %.2f)" % (
            gname,
            ind_s,
        )
        # (2) the tabular-state control fails (state explosion, never-visited test states).
        assert ind_s - tab_s >= 0.6, (
            "%s: inducer must beat tabular on scaled by >=0.6 (%.2f vs %.2f)"
            % (gname, ind_s, tab_s)
        )
        assert ind_h - tab_h >= 0.5, (
            "%s: inducer must beat tabular on held-out by >=0.5 (%.2f vs %.2f)"
            % (gname, ind_h, tab_h)
        )
        # (3) shuffled transitions collapse to chance (no coherent action->effect rule).
        assert ind_s - shuf_s >= 0.5, (
            "%s: shuffle control must collapse on scaled (%.2f vs %.2f)" % (gname, ind_s, shuf_s)
        )

    assert det_ok, "two full runs must reproduce byte-identical solve rates (deterministic)"

    kd, pb = results["key_door"]["ind_scaled"], results["push_box"]["ind_scaled"]
    kt, pt = results["key_door"]["tab_scaled"], results["push_box"]["tab_scaled"]
    ks, ps = results["key_door"]["shuf_scaled"], results["push_box"]["shuf_scaled"]
    print(
        "\nself-check OK: rule-inducer SCALED solve key_door %.2f / push_box %.2f  >>  "
        "tabular %.2f / %.2f  >>  shuffle %.2f / %.2f;" % (kd, pb, kt, pt, ks, ps)
    )
    print(
        "  relational rules TRANSFER to bigger grids & new layouts; the tabular table does not "
        "(unseen states); shuffling action->effect destroys induction; deterministic; spiking "
        "RaceAccumulator selects."
    )
    print(
        "  HONEST: toy games have learnable relational rules by construction -- this does NOT solve "
        "frontier ARC-AGI-3 (richer rule classes + long horizons uncovered); the claim is only that"
    )
    print("  relational induction generalizes where tabular coverage explodes.")
    print(
        "  Scope of THIS demo: 2 of 4 games (dependency=key_door, push=push_box), both grid-scalable "
        "with pure zero-shot transfer. color_lock's safe color is a per-INSTANCE latent (needs a few"
    )
    print(
        "  grounding steps, not zero-shot) and toggle_to_goal is not grid-scalable -- honestly out of "
        "the scaled-transfer proof, though the same object/rule machinery parses them."
    )


if __name__ == "__main__":
    main()
