# ARC-AGI-3 (research/arc_agi)

Two ways an SNDL brain touches the [ARC Prize](https://docs.arcprize.org) ARC-AGI-3
interactive-reasoning benchmark.

> Corrected 2026-08-04: this line read *"Quarantined in `research/` (not part of the zero-dep core
> gate)"*. There is no zero-dep core gate — the zero-dependency rule was abolished project-wide on
> 2026-07-20 and replaced by P7 — and under P4c *"it is only a benchmark"* is not a disposition.
> These files drive real brain faculties on the live construction path's own classes; that they are
> not yet in `run_selfchecks.py` is debt. `rule_inducer.py` self-checks and passes today (run below).

## 1. Self-contained representative harness (no network, no key)

- `env.py` — 4 hand-authored ARC-AGI-3-STYLE interactive grid games (key_door, push_box,
  color_lock, toggle_to_goal), each a deterministic env with a hidden rule to discover.
- `brain_player.py` — a Go-Explore player over the brain's cognition (CuriosityDrive +
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

Honest scope: the brain's LANGUAGE and PERCEPTION (N-MNIST 34x34 encoder) tiers do NOT apply to
arcade gameplay and don't co-install with `arc-agi`'s numpy>=2.2 — so "all tiers" for PLAYING is the
full COGNITIVE stack above.

> Corrected 2026-08-04: this said *"the brain's torch LANGUAGE (SpikeGPT = conversation)"*. The
> SpikeGPT-216M organ was **scrapped in v3.77** — `cortex/brain.py` says so at the language-organ
> block, and conversation now comes from the acquired organ (`cortex/language_acquisition`, built
> lazily in `_lm_serve`). Naming a deleted component as the reason for a scope limit makes the limit
> unverifiable. `play_brain.py`'s own docstring carried the same stale name and is corrected too.

### Setup + run

```bash
python -m venv .venv-arc && .venv-arc/Scripts/python -m pip install arc-agi   # get a key: three.arcprize.org
ARC_API_KEY=<key> PYTHONPATH=src .venv-arc/Scripts/python research/arc_agi/play_brain.py ls20 400
ARC_API_KEY=<key> PYTHONPATH=src .venv-arc/Scripts/python research/arc_agi/play.py       ls20 400  # random baseline
```

## 3. Relational rule-inducer + reasoning layer

- `rule_inducer.py` — a relational grid rule-inducer: parse the grid into OBJECTS (connected components),
  INDUCE causal change-rules from (grid, action, grid') transitions (object translate / click-transform /
  appear / dependency / goal), then PLAN with a model-based value iteration; the executed action is chosen
  by a spiking RaceAccumulator. Reuses `cortex/vsa_reasoning` (rule store), `basal_ganglia/evidence_accumulation`.
  PROVEN (self-check): rules learned on 4 small size-5 instances TRANSFER zero-shot to held-out + SCALED-UP
  grids -- key_door **1.00** and push_box **1.00** solve, vs a tabular-state agent (same budget) at **0.06 / 0.00**
  (state explosion) and a shuffled-action control at 0.12 / 0.00. That is the honest win: relational induction
  GENERALIZES where tabular coverage explodes, on learnable-rule games.
- `play_brain.py` wires it as an OPPORTUNISTIC reasoning layer: it induces move-agent rules from the live
  transition stream and plans with them WHEN a game fits the schema; else it falls back to curiosity + a
  frame-change exploration bonus. The summary prints `reasoning: schema_fit=N plan_actions=M`.

### Honest benchmark result (real arcade, registered key)

The full-brain agent connects and plays the real games. On `ls20` and `ft09` it completes **0 levels** --
TIED with the random baseline (also 0). The reasoning layer reports **`schema_fit=0, plan_actions=0`** on
these games: they do not expose the relational schema the inducer needs (no single move-agent that translates
by deltas toward a goal), so the planner never engages -- exactly the inducer's own stated honest-negative,
now confirmed on the live arcade. On `ls20`, single actions do not even change the frame from the start
(~3 reachable states), so curiosity has nothing to climb. No overclaim: the rule-inducer is a genuine
relational reasoner (proven to generalize on learnable-schema toy games); the frontier ARC-AGI-3 games need
richer rule classes + long horizons it does not cover, so the brain plays for real but does not solve these.
