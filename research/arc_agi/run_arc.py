"""Run the brain-player on the ARC-AGI-3-STYLE suite and report per-game solve rate vs a
random-action baseline. Deterministic (two runs are byte-identical).

HONEST FRAMING (repeated on purpose)
------------------------------------
This is a REPRESENTATIVE, hand-authored harness -- NOT the official ARC-AGI-3 evaluation, which
runs on a private game server we do not have. The numbers below are ONLY the brain-player's solve
rate against a random-action baseline on THESE four toy games. They make NO claim about the real
ARC-AGI-3 leaderboard or about generalization to the official tasks. Where the brain-player only
ties the random baseline, that is reported as-is.

Method
------
For each game and each of N_SEEDS layout seeds:
  - brain: BrainPlayer(seed).solve(game(seed)) -> solved bool (deterministic per seed).
  - random: mean over R_RANDOM uniform-random rollouts (distinct seeds) -> per-seed solve rate.
Per-game brain rate = fraction of seeds solved; per-game random rate = mean per-seed random rate.

Run:  PYTHONUTF8=1 PYTHONPATH=src:. python research/arc_agi/run_arc.py
  (on Windows bash, if the ':' path separator fails, use  PYTHONPATH="src;."  instead;
   the file also bootstraps sys.path itself, so a bare `python research/arc_agi/run_arc.py` works.)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env as arc_env  # noqa: E402
from brain_player import BrainPlayer, random_player  # noqa: E402

N_SEEDS = 12          # layout seeds per game
R_RANDOM = 20         # random rollouts per seed for the baseline


def eval_game(game_cls):
    brain_solved = 0
    rnd_total = 0.0
    for seed in range(N_SEEDS):
        # brain: deterministic solve on this layout.
        game = game_cls(seed=seed)
        if BrainPlayer(game.n_actions, seed=seed).solve(game):
            brain_solved += 1
        # random baseline on the SAME layout, averaged over R_RANDOM rollouts.
        hits = sum(random_player(game_cls(seed=seed), 1000 * seed + r) for r in range(R_RANDOM))
        rnd_total += hits / R_RANDOM
    return brain_solved / N_SEEDS, rnd_total / N_SEEDS


def main():
    print("=" * 78)
    print("ARC-AGI-3-STYLE interactive-reasoning harness  (REPRESENTATIVE analogs)")
    print("NOT the official ARC-AGI-3 eval. Solve rates are brain-player vs RANDOM baseline")
    print("on these hand-authored toy games only -- no claim about the real ARC-AGI-3 benchmark.")
    print("player = the brain's own cognition: population spike-code perception + CuriosityDrive")
    print("Go-Explore + model-based value iteration + spiking RaceAccumulator action selection.")
    print("=" * 78)
    print("%-16s %8s %8s %8s   %s" % ("game", "seeds", "brain", "random", "verdict"))
    print("-" * 78)
    brain_rates = []
    for cls in arc_env.GAMES:
        brate, rrate = eval_game(cls)
        brain_rates.append(brate)
        margin = brate - rrate
        verdict = "beats random (+%.2f)" % margin if margin > 1e-9 else \
                  ("ties random" if abs(margin) <= 1e-9 else "below random (%.2f)" % margin)
        print("%-16s %8d %8.2f %8.2f   %s" % (cls.name, N_SEEDS, brate, rrate, verdict))
    print("-" * 78)
    print("mean brain solve rate across games: %.2f" % (sum(brain_rates) / len(brain_rates)))
    print("harness OK")


if __name__ == "__main__":
    main()
