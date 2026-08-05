"""ARC-AGI-3 bridge for SNDL.

Runs the interactive ARC-AGI-3 arcade (https://docs.arcprize.org) so an SNDL
agent can play. Needs the `arc-agi` pip package, which requires Python >=3.12
and numpy>=2.2 -- a hard conflict with tonic in .venv-cu128 -- so it runs in
its own venv (.venv-arc). That is a RESOLVER conflict, measured, which is the
kind of reason P7 accepts; the word "quarantined" was here until 2026-08-04 and
is retired with the zero-dependency rule it belonged to.

Run:
    .venv-arc/Scripts/python research/arc_agi/play.py            # 10-step smoke
    .venv-arc/Scripts/python research/arc_agi/play.py ft09 100   # other game/steps

Set ARC_API_KEY for a registered key; anonymous key is used otherwise.

Agent contract: agent_fn(frame) -> GameAction, where frame is FrameDataRaw
(frame.frame = list of 64x64 grids of color ints 0-15, frame.available_actions
= legal action ids, frame.state / frame.levels_completed = progress). An SNDL
brain plugs in by spike-encoding frame.frame and returning one of the legal
actions.
"""

import os
import pathlib
import random
import sys

import arc_agi
from arcengine import GameAction, GameState


# GameAction(int) is broken in this arcengine build, so map value -> member.
_BY_VALUE = {a.value: a for a in GameAction}


def load_env_file():
    """Load the gitignored .env (ARC_API_KEY) if present -- no python-dotenv needed. Never
    overrides an exported value.

    CALLED FROM play(), not at module scope -- moved 2026-08-04. It used to run as a bare top-level
    `if`/`for` that wrote os.environ the moment anything imported this module, and play_brain.py
    imports it (`from play import _BY_VALUE, play`). Two problems with that, and the second is the
    one that matters: (1) it is the module-scope side effect the P4c promotion of the daemon spent
    three bullet points removing (POLICIES.md P4c -- `os.chdir`, `sys.path.insert`,
    `os.environ.setdefault`), and an in-process runner applies it to every module after this one;
    (2) it injects a CREDENTIAL into the process environment as a side effect of an import, so a
    reader of another module cannot see where the key came from. Now it happens once, in the
    function that is about to hand the key to the arcade, which is also the only place it is needed.
    """
    envf = pathlib.Path(__file__).with_name(".env")
    if not envf.exists():
        return False
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return True


def random_agent(frame, rng=random):
    """Uniform-random legal action. `rng` defaults to the `random` module so existing one-argument
    callers are unchanged; play() passes a per-instance random.Random(seed) so a run is
    reproducible without touching the process-global RNG (the project's determinism rule)."""
    ids = frame.available_actions or [GameAction.ACTION1.value]
    return _BY_VALUE[rng.choice(ids)]


def play(agent_fn=None, game="ls20", max_steps=10, render=False, seed=0):
    """Play one episode; returns (final_frame, scorecard).

    `agent_fn=None` uses the random baseline driven by a PER-INSTANCE random.Random(seed) instead of
    the module-global RNG, so two runs at the same seed match and a caller's own `random` state is
    left alone.
    """
    if agent_fn is None:
        rng = random.Random(seed)

        def agent_fn(frame):
            return random_agent(frame, rng)

    load_env_file()  # must precede Arcade(), which reads ARC_API_KEY
    arc = arc_agi.Arcade()
    env = arc.make(game, render_mode="terminal" if render else None)
    frame = env.reset()
    for _ in range(max_steps):
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            break
        frame = env.step(agent_fn(frame))
    return frame, arc.get_scorecard()


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # `random.seed(0)` was here. Removed: play() now seeds a per-instance random.Random(seed), which
    # is the project's determinism rule ("random.Random(seed) per instance, NEVER module-scope
    # seeding"). The global seed was also load-bearing for reproducibility while random_agent read
    # the module RNG, so deleting one without the other would have quietly made runs unrepeatable.
    frame, scorecard = play(game=game, max_steps=steps, seed=0)
    print(f"game={frame.game_id} state={frame.state.value} "
          f"levels={frame.levels_completed}/{frame.win_levels}")
    print(scorecard)
    assert frame.frame and len(frame.frame[0]) == 64, "expected 64x64 frame"
    print("self-check OK")
