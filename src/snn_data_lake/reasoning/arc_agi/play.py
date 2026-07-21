"""ARC-AGI-3 bridge for SNDL.

Runs the interactive ARC-AGI-3 arcade (https://docs.arcprize.org) so an SNDL
agent can play. Quarantined in research/: needs the `arc-agi` pip package,
which requires Python >=3.12 and numpy>=2.2 (conflicts with tonic in
.venv-cu128), so it lives in its own venv.

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

# Load the gitignored .env (ARC_API_KEY) if present -- no python-dotenv needed. Does NOT
# override an already-exported ARC_API_KEY. Must run before arc_agi.Arcade() reads the key.
_envf = pathlib.Path(__file__).with_name(".env")
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import snn_data_lake.reasoning.arc_agi as arc_agi
from arcengine import GameAction, GameState


# GameAction(int) is broken in this arcengine build, so map value -> member.
_BY_VALUE = {a.value: a for a in GameAction}


def random_agent(frame):
    ids = frame.available_actions or [GameAction.ACTION1.value]
    return _BY_VALUE[random.choice(ids)]


def play(agent_fn=random_agent, game="ls20", max_steps=10, render=False):
    """Play one episode; returns (final_frame, scorecard)."""
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
    random.seed(0)
    frame, scorecard = play(game=game, max_steps=steps)
    print(
        f"game={frame.game_id} state={frame.state.value} "
        f"levels={frame.levels_completed}/{frame.win_levels}"
    )
    print(scorecard)
    assert frame.frame and len(frame.frame[0]) == 64, "expected 64x64 frame"
    print("self-check OK")
