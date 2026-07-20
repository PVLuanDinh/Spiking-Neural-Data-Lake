"""Wire the SNDL brain's FULL cognitive decision-stack to the LIVE ARC-AGI-3 arcade for REAL benchmarks.

HONEST SCOPE (read first). "All tiers" for PLAYING means the full COGNITIVE stack -- every faculty
that maps to choosing a game action. The brain's torch LANGUAGE tier (SpikeGPT) is conversation, not
a game policy; its torch PERCEPTION tier is trained on 34x34 N-MNIST event data and would not transfer
to 64x64 arcade grids -- and neither co-installs with arc-agi's numpy>=2.2. So they are deliberately
NOT used here; using them would be theatre, not function. What DOES play, composed below, is the brain's
real zero-dep cognition:
  * exploration  -- neuromodulation/intrinsic_motivation.CuriosityDrive (learning-progress + novelty)
                    + neuromodulation/generalizing_novelty.GeneralizingNovelty (SimHash pseudo-count);
  * rule discovery -- an online world model T (transitions) + model-based value iteration (planning);
  * decision     -- basal_ganglia/evidence_accumulation.RaceAccumulator (a genuine spiking N-alternative
                    race over the candidate actions' values);
  * control      -- cortex/metacognitive_monitor.MetacognitiveMonitor gates explore-vs-exploit.

ACTION SPACE (the thing that actually matters): ARC-AGI-3 mixes SIMPLE actions (ACTION1-5, ACTION7,
RESET) with a COMPLEX action ACTION6 = CLICK at (x, y). A bare ACTION6 is a no-op, so the brain must
choose a click POSITION. We tile the frame into a KxK grid of click targets and let curiosity explore
them, so the agent's real action set per step is {simple legal actions} + {click each of KxK cells}.

Value/representation-ONLY: imports NOTHING from brainstem/; emits only LEGAL arcade actions; no actuator.
Real arcade, real scorecard, NO overclaim -- ARC-AGI-3 is very hard; we report whatever the brain scores.

Run (in the arc venv, with src on the path so the zero-dep brain modules import):
    ARC_API_KEY=<key> PYTHONPATH=src .venv-arc/Scripts/python research/arc_agi/play_brain.py ls20 400
"""
import os
import random
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arcengine import GameAction, GameState  # noqa: E402
from play import _BY_VALUE, play  # noqa: E402  (the arcade bridge)

from rule_inducer import InducedModel, _race_action, induce  # noqa: E402  (relational reasoning layer)
from snn_data_lake.basal_ganglia.evidence_accumulation import RaceAccumulator  # noqa: E402
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor  # noqa: E402
from snn_data_lake.neuromodulation.generalizing_novelty import GeneralizingNovelty  # noqa: E402
from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive  # noqa: E402

_CLICK = GameAction.ACTION6.value                  # the one complex (x,y) action
K = 8                                              # click grid: K*K targets tiled over the frame


def _sig(grids):
    """Compact deterministic state id: crc32 of the raw frame stack (never builtin hash())."""
    return zlib.crc32(repr(grids).encode()) & 0xFFFFFFFF


def _to_list(grid):
    """Coerce a grid (numpy array from the arcade, or a list from the toy env) to a nested int list,
    so the pure-stdlib rule-inducer + change detection never touch numpy truth-value semantics."""
    return [[int(v) for v in row] for row in grid]


def _raster(grids):
    """Per-color occupancy fractions over the frame stack (length 16) -- the SimHash pseudo-count key,
    so SIMILAR frames share generalizing visitation credit."""
    counts = [0] * 16
    total = 0
    for grid in grids:
        for row in grid:
            for c in row:
                if 0 <= c < 16:
                    counts[c] += 1
                    total += 1
    total = total or 1
    return [c / total for c in counts]


class FullBrainArcadeAgent:
    """The brain's full cognitive decision-stack as an online ARC-AGI-3 arcade policy with clicks."""

    def __init__(self, seed=0, gamma=0.95, race_scale=6.0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.gamma = gamma
        self.race_scale = race_scale
        self.cur = CuriosityDrive(seed=seed)
        self.gen = GeneralizingNovelty(n_bits=16, dim=16, seed=seed)
        self.meta = MetacognitiveMonitor(base=0.4, seed=seed)
        self.T = {}                    # (sig, key) -> (sig2, reward, done)
        self.Q = {}                    # sig -> {key: value}
        self.tried = {}                # sig -> set(keys)
        self.last = None
        self.prev_levels = 0
        self.steps = 0
        # ---- RELATIONAL REASONING LAYER (opportunistic): induce move-agent rules from simple-action
        # transitions; if a game FITS the relational schema (one move-agent + goal/collect/push), plan
        # with InducedModel; else fall back. Also an object-change exploration bonus that attacks the
        # no-op problem (prefer actions that actually move objects). Honest: only games matching the
        # schema get the planner -- opaque frontier games (no single move-agent) fall back to curiosity.
        self._trans = []               # [(grid, simple_action_id, grid2)] for induction
        self._prev_grid = None
        self._prev_sig = None
        self._model = None             # InducedModel when the schema fits, else None
        self._model_Q = {}             # symbolic state -> [value per action] from the induced model
        self.plan_hits = 0             # how many steps the reasoning layer chose the action
        self.induce_ok = 0             # how many times a usable relational model was built

    def _try_induce(self, grid):
        """Opportunistic relational induction: if the game FITS the move-agent schema (one agent that
        moves by deltas + a goal/collect/push), build a planning model + value-iterate it; else leave
        self._model None -- opaque frontier games (no single move-agent) fall back to curiosity."""
        try:
            rules = induce(self._trans, min_conf=0.5)
            if rules.agent_color is None or not rules.deltas:
                self._model = None
                return
            model = InducedModel(rules, grid, 8)
            s = model.state_from_grid(grid)
            if s is None:                      # schema does not fit this game -> no plan
                self._model = None
                return
            self._model = model
            self._model_Q = model.value_iterate(s)
            self.induce_ok += 1
        except Exception:
            self._model = None

    def _planned_action(self, grid):
        """The planned SIMPLE action for the current grid from the induced model (spiking race over the
        planned values), or None when no model fits / no valued action exists."""
        if self._model is None:
            return None
        try:
            s = self._model.state_from_grid(grid)
            qv = self._model_Q.get(s) if s is not None else None
            if not qv or max(qv) <= 0.0:
                return None
            seed = (self.seed ^ (self.steps * 40503)) & 0xFFFFFFFF
            return _race_action(qv, seed, self.race_scale, len(qv))
        except Exception:
            return None

    def _keys(self, frame):
        """The action set this step: simple legal ids (int) + click targets (('c', x, y)) if ACTION6
        is legal. Click targets are the centers of a KxK tiling of the frame."""
        legal = list(frame.available_actions or [GameAction.ACTION1.value])
        keys = [a for a in legal if a != _CLICK]
        if _CLICK in legal and frame.frame:
            h = len(frame.frame[0])
            w = len(frame.frame[0][0]) if frame.frame[0] else h
            for cy in range(K):
                for cx in range(K):
                    x = min(w - 1, int((cx + 0.5) * w / K))
                    y = min(h - 1, int((cy + 0.5) * h / K))
                    keys.append(("c", x, y))
        return keys

    def _emit(self, key):
        """Turn an action key into an arcade action object."""
        if isinstance(key, tuple):                 # ('c', x, y) -> the complex click action
            return GameAction.ACTION6.set_data({"x": key[1], "y": key[2]})
        return _BY_VALUE[key]                      # a simple action id

    def __call__(self, frame):
        self.steps += 1
        grids = frame.frame
        sig = _sig(grids)
        levels = frame.levels_completed or 0
        reward = float(levels - self.prev_levels)
        self.prev_levels = levels
        done = frame.state in (GameState.WIN, GameState.GAME_OVER)
        grid0 = _to_list(grids[0]) if len(grids) else []

        if self.last is not None:                  # fold the last transition into world model + drives
            s0, k0 = self.last
            predicted = self.T.get((s0, k0), (None,))[0]
            pred_err = 0.0 if predicted == sig else 1.0
            self.cur.intrinsic_reward((s0, k0), pred_err)
            changed = self._prev_sig is not None and sig != self._prev_sig   # crc32-based (numpy-safe)
            eff = reward + (0.01 if changed else 0.0)   # frame-change bonus -> prefer non-no-op actions
            self.T[(s0, k0)] = (sig, eff, done)
            self.tried.setdefault(s0, set()).add(k0)
            self.meta.update(confidence=1.0 - pred_err, surprise=pred_err,
                             learning_progress=self.cur.learning_progress((s0, k0)),
                             novelty=self.cur.novelty((s0, k0)))
            if isinstance(k0, int) and self._prev_grid is not None:   # record a simple-action transition
                self._trans.append((self._prev_grid, k0, grid0))
                if len(self._trans) > 4000:
                    self._trans = self._trans[-4000:]
        self.gen.bonus(_raster(grids))
        if self.steps % 24 == 0:
            self._plan()
        if self.steps % 60 == 0 and len(self._trans) >= 40:
            self._try_induce(grid0)                # opportunistic relational induction (fits some games)

        keys = self._keys(frame)
        planned = self._planned_action(grid0)      # REASONING: follow the induced plan when a game fits
        if planned is not None and planned in keys:
            key = planned
            self.plan_hits += 1
        else:
            qv = self.Q.get(sig) or {}
            untried = [k for k in keys if k not in self.tried.get(sig, set())]
            if untried and (self.meta.explore() or not qv or max(qv.values(), default=0.0) <= 0.0):
                key = self.rng.choice(untried)     # explore an untried (simple or click) action
            else:
                vals = [qv.get(k, 0.0) for k in keys]
                key = keys[self._race(vals)] if max(vals) > 0.0 else self.rng.choice(keys)
        self.last = (sig, key)
        self._prev_grid = grid0
        self._prev_sig = sig
        return self._emit(key)

    def _plan(self):
        self.Q = {}
        for _ in range(150):
            delta = 0.0
            for (s, k), (s2, r, dn) in self.T.items():
                nxt = 0.0 if dn else max((self.Q.get(s2) or {0: 0.0}).values(), default=0.0)
                new = r + (0.0 if dn else self.gamma * nxt)
                self.Q.setdefault(s, {})
                prev = self.Q[s].get(k, 0.0)
                if new != prev:
                    self.Q[s][k] = new
                    delta = max(delta, abs(new - prev))
            if delta < 1e-9:
                break

    def _race(self, vals):
        mn, mx = min(vals), max(vals)
        span = mx - mn
        drifts = ([self.race_scale * (v - mn) / span for v in vals] if span > 1e-9
                  else [0.0] * len(vals))
        seed = (self.seed ^ (self.steps * 2654435761)) & 0xFFFFFFFF
        return RaceAccumulator(drifts, bound=1.0, noise=0.3, dt=0.01, seed=seed, max_steps=400).decide()[0]


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    agent = FullBrainArcadeAgent(seed=0)
    frame, scorecard = play(agent_fn=agent, game=game, max_steps=steps)
    print(f"BRAIN  game={frame.game_id}  state={frame.state.value}  "
          f"levels={frame.levels_completed}/{frame.win_levels}  steps={agent.steps}  "
          f"states_seen={len(agent.tried)}  model_transitions={len(agent.T)}  "
          f"reasoning: schema_fit={agent.induce_ok} plan_actions={agent.plan_hits}")
    print("scorecard:", scorecard)


if __name__ == "__main__":
    main()
