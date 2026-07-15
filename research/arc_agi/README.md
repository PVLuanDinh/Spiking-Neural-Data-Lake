# ARC-AGI-3 (research/arc_agi)

Two ways an SNDL brain touches the [ARC Prize](https://docs.arcprize.org) ARC-AGI-3
interactive-reasoning benchmark. Quarantined in `research/` (not part of the zero-dep core gate).

## 1. Self-contained representative harness (no network, no key)

- `env.py` — 4 hand-authored ARC-AGI-3-STYLE interactive grid games (key_door, push_box,
  color_lock, toggle_to_goal), each a deterministic env with a hidden rule to discover.
- `brain_player.py` — a Go-Explore player over the brain's zero-dep cognition (CuriosityDrive +
  snapshot/restore Go-Explore + model-based value iteration + spiking RaceAccumulator).
- `run_arc.py` — `python research/arc_agi/run_arc.py` reports per-game brain-vs-random solve rate.

Honest: these are hand-authored ANALOGS, NOT the official eval. On them the brain beats random on all
four (1.00 vs 0.50 / 0.13 / 0.70 / 0.28); makes no claim about the real leaderboard.

## 2. LIVE arcade — the full brain plays for real (network + key)

- `play.py` — the bridge to the real arcade via the `arc-agi` pip package (`play(agent_fn, game, steps)`;
  `agent_fn(frame) -> action`). Needs its own venv (`.venv-arc`, py3.12+, numpy>=2.2) because `arc-agi`'s
  numpy conflicts with `tonic`/`snntorch` in `.venv-cu128`.
- `play_brain.py` — the brain's FULL cognitive decision-stack as a live arcade policy: CuriosityDrive +
  GeneralizingNovelty (SimHash) exploration, an online world model + value iteration, a spiking
  RaceAccumulator decision, and a MetacognitiveMonitor explore/exploit gate. Handles both SIMPLE actions
  and the COMPLEX ACTION6 click(x,y) (tiles the frame into KxK click targets). Value/representation-only:
  imports nothing from `brainstem/`, emits only legal actions.

Honest scope: the brain's torch LANGUAGE (SpikeGPT = conversation) and PERCEPTION (N-MNIST 34x34 encoder)
tiers do NOT apply to arcade gameplay and don't co-install with `arc-agi`'s numpy>=2.2 — so "all tiers"
for PLAYING is the full zero-dep COGNITIVE stack above.

### Setup + run

```bash
python -m venv .venv-arc && .venv-arc/Scripts/python -m pip install arc-agi   # get a key: three.arcprize.org
ARC_API_KEY=<key> PYTHONPATH=src .venv-arc/Scripts/python research/arc_agi/play_brain.py ls20 400
ARC_API_KEY=<key> PYTHONPATH=src .venv-arc/Scripts/python research/arc_agi/play.py       ls20 400  # random baseline
```

### Honest benchmark result (real arcade, registered key)

The full-brain agent connects and plays the real games. On `ls20` and `ft09` it completes **0 levels** --
TIED with the random baseline (also 0). This is honest: ARC-AGI-3 is a frontier, largely-unsolved
benchmark. On `ls20` only ACTION1-4 are offered and single actions do not change the frame from the start
(near-static reachability, ~3 distinct states), so a tabular-curiosity + value-iteration agent has almost
nothing to climb -- these games need relational/causal RULE INDUCTION over the grid and long credit
horizons, which this policy does not do. No overclaim: the brain plays for real; it does not solve these.
