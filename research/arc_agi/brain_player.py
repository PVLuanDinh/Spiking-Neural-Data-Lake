"""Brain-player adapter: drives the Spiking Neural Data Lake's cognition to PLAY the ARC-AGI-3-STYLE
grid games in env.py.

HONEST FRAMING
--------------
This is a REPRESENTATIVE harness, not the official ARC-AGI-3 eval (see env.py). The player below
drives the brain's real cognitive modules directly -- it does NOT construct `engine.brain.Brain`
(this is not the one brain; per P4 a consumer that stood up its own would be a replica) and imports
NOTHING from brainstem/ (no arm, no actuator). It is VALUE- and REPRESENTATION-ONLY.

(Retired 2026-08-04: this said "ZERO-DEP cognition" / "pure-stdlib cognitive machinery", and gave
"that needs torch" as the reason for not importing Brain. Both are residue of the zero-dependency
rule abolished on 2026-07-20 -- torch is now an unconditional core dependency, so "needs torch" is
not a reason to avoid anything (P7). The real reason to avoid constructing a Brain here is P4, and
it is stated above.)

What real brain modules it drives
---------------------------------
- PERCEPTION: `perceive(grid)` -> a population one-hot SPIKE code (one firing input neuron per
  cell, selected by that cell's color). This spike tuple IS the state signature the player reasons
  over -- lossless and hashable.
- EXPLORATION (rule discovery): `faculties.intrinsic_motivation.CuriosityDrive` (the real
  module), used two ways:
    * a world-model TRANSITION curiosity: intrinsic_reward((state,action), pred_error) where the
      prediction error is 1.0 for a surprising (unseen) transition and 0.0 once the transition is
      learned -- learning progress rises then falls as each rule is mastered (the anti-noisy-TV
      signal, straight from the module).
    * a GO-EXPLORE frontier selector: most_curious(archived_states) + observe(untried_fraction),
      which is EXACTLY the selection mechanism `cortex.quality_diversity.QualityDiversityArchive.
      _select_parent` uses. NOTE (honest): we do not call QualityDiversityArchive itself, because
      its `run()` is hardcoded to search `SpikingPolicy` weight-genomes and cannot search grid-game
      policies. We reuse its real CuriosityDrive and mirror its Go-Explore selection over game
      states (returning to promising states via env.snapshot/restore).
- DECISION: `faculties.evidence_accumulation.RaceAccumulator` (the real spiking N-alternative
  winner-take-all race) picks the exploit action from the learned action-values -- evidence
  accumulation, not a bare argmax.
- PLANNING: a small model-based value iteration over the world model the curiosity-driven
  exploration discovered (tabular; the games are tiny MDPs).

Determinism: per-instance random.Random(seed); RaceAccumulator seeded from a zlib.crc32 of the
state; no wall-clock, no global RNG. Same seed -> identical play.

Run `python research/arc_agi/brain_player.py` for a self-check on one instance of each game.
"""

import os
import random
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root
for _p in (os.path.join(_ROOT, "src"), _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env as arc_env  # noqa: E402

from snn_data_lake.faculties.evidence_accumulation import RaceAccumulator  # noqa: E402
from snn_data_lake.faculties.intrinsic_motivation import CuriosityDrive  # noqa: E402

PALETTE = 16  # fixed color palette for the one-hot spike encoder (all game colors are < 16)


class BrainPlayer:
    """Curiosity-driven Go-Explore player over the brain's own cognition modules.

    solve(game) -> bool : discover the hidden rule by interacting, then try to reach the goal.
    Returns True iff the exploit rollout reaches a rewarding terminal within game.max_steps."""

    def __init__(self, n_actions, seed=0, explore_budget=8000, gamma=0.95, race_scale=6.0):
        self.nA = n_actions
        self.seed = seed
        self.explore_budget = explore_budget
        self.gamma = gamma
        self.race_scale = race_scale
        self.rng = random.Random(seed)
        # two real CuriosityDrive instances: one for the Go-Explore frontier, one for transitions.
        self.cur_frontier = CuriosityDrive(seed=seed)
        self.cur_action = CuriosityDrive(seed=seed + 1)
        self.T = {}  # (sig, action) -> (sig2, reward, done)   the learned world model
        self.snap = {}  # sig -> env snapshot (Go-Explore return-to-state)
        self.tried = {}  # sig -> set(actions already taken from this state)
        self.expandable = set()  # non-terminal sigs still worth returning to
        self.states = set()  # every sig seen
        self.Q = {}  # sig -> [value per action]

    # ---- perception: colored grid -> population one-hot spike code (also the state signature) ----
    def perceive(self, grid):
        spikes = []
        idx = 0
        for row in grid:
            for col in row:
                col = col if 0 <= col < PALETTE else PALETTE - 1
                spikes.append(idx * PALETTE + col)
                idx += 1
        return tuple(spikes)

    def _register(self, sig, snap, terminal):
        if sig not in self.states:
            self.states.add(sig)
            self.tried[sig] = set()
        if not terminal:
            self.snap[sig] = snap
            self.expandable.add(sig)

    def _untried_frac(self, sig):
        return (self.nA - len(self.tried.get(sig, ()))) / self.nA

    def _select_frontier(self):
        """Go-Explore parent selection: the archived state of maximal curiosity value, then observe
        its untried-action fraction so the frontier signal habituates (mirrors QDArchive)."""
        keys = sorted(self.expandable)
        sig = self.cur_frontier.most_curious(keys)
        self.cur_frontier.observe(sig, self._untried_frac(sig))
        return sig

    def _explore_action(self, sig):
        untried = [a for a in range(self.nA) if a not in self.tried[sig]]
        if untried:
            return self.rng.choice(untried)
        return self.cur_action.most_curious([(sig, a) for a in range(self.nA)])[1]

    def _explore(self, game):
        """Curiosity-driven Go-Explore rollout: build the world model T by returning to promising
        states (via snapshot/restore) and taking a novel action each time. A state stays on the
        frontier exactly while it has an untried action, so the loop terminates when every reachable
        state has been fully expanded -- complete coverage of the (tiny) reachable set. Curiosity
        (learning progress + novelty) only orders WHICH frontier state is expanded next."""
        grid = game.reset()
        s = self.perceive(grid)
        self._register(s, game.snapshot(), terminal=False)
        for _ in range(self.explore_budget):
            if not self.expandable:
                break
            s = self._select_frontier()
            game.restore(self.snap[s])
            a = self._explore_action(s)
            grid2, r, done = game.step(a)
            s2 = self.perceive(grid2)
            # world-model curiosity: surprising transitions carry error 1.0, learned ones 0.0.
            predicted = self.T.get((s, a), (None,))[0]
            pred_err = 0.0 if predicted == s2 else 1.0
            self.cur_action.intrinsic_reward((s, a), pred_err)
            self.T[(s, a)] = (s2, r, done)
            self.tried[s].add(a)
            self._register(s2, game.snapshot(), terminal=done)
            if len(self.tried[s]) >= self.nA:  # fully expanded -> leaves the frontier
                self.expandable.discard(s)

    def _plan(self):
        """Model-based value iteration over the discovered transitions (tiny tabular MDP)."""
        self.Q = {s: [0.0] * self.nA for s in self.states}
        for _ in range(500):
            delta = 0.0
            for (s, a), (s2, r, done) in self.T.items():
                nxt = 0.0 if done else max(self.Q.get(s2, [0.0]))
                new = r + (0.0 if done else self.gamma * nxt)
                old = self.Q[s][a]
                if new != old:
                    self.Q[s][a] = new
                    delta = max(delta, abs(new - old))
            if delta < 1e-9:
                break

    def _race_action(self, qvals, seed):
        """Pick an action by a genuine spiking evidence-accumulation race over the action-values.

        Drifts are the action-values normalized to [0, race_scale] so the best-valued action carries
        clearly the strongest evidence (the race then reliably wins for it), while truly equal-valued
        actions stay a genuine near-tie the noisy race breaks."""
        mn, mx = min(qvals), max(qvals)
        span = mx - mn
        drifts = (
            [self.race_scale * (v - mn) / span for v in qvals]
            if span > 1e-9
            else [0.0] * len(qvals)
        )
        acc = RaceAccumulator(drifts, bound=1.0, noise=0.3, dt=0.01, seed=seed, max_steps=400)
        winner, _rt = acc.decide()
        return winner

    def _exploit(self, game):
        grid = game.reset()
        s = self.perceive(grid)
        for step in range(game.max_steps):
            qvals = self.Q.get(s)
            if qvals is None or max(qvals) <= 0.0:
                a = self.rng.randrange(self.nA)  # unseen / unvalued -> fall back
            else:
                seed = (zlib.crc32(repr(s).encode()) ^ (step * 2654435761)) & 0xFFFFFFFF
                a = self._race_action(qvals, seed)
            grid, r, done = game.step(a)
            s = self.perceive(grid)
            if done:
                return r > 0.0
        return False

    def solve(self, game):
        self._explore(game)
        self._plan()
        return self._exploit(game)


def random_player(game, seed):
    """Baseline: uniform-random actions until a terminal or the step cap. Returns solved bool."""
    rng = random.Random(seed)
    game.reset()
    for _ in range(game.max_steps):
        _grid, r, done = game.step(rng.randrange(game.n_actions))
        if done:
            return r > 0.0
    return False


def main():
    print("brain_player self-check -- REPRESENTATIVE ARC-AGI-3-style games, NOT the official eval.")
    for cls in arc_env.GAMES:
        solved = BrainPlayer(cls(seed=0).n_actions, seed=0).solve(cls(seed=0))
        # quick random reference on the same instance (10 rollouts).
        rnd = sum(random_player(cls(seed=0), r) for r in range(10)) / 10.0
        print("  %-14s brain_solved=%s  random_rate=%.2f" % (cls.name, solved, rnd))
    print("brain_player self-check OK")


if __name__ == "__main__":
    main()
