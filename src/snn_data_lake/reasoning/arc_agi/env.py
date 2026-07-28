"""ARC-AGI-3-STYLE interactive grid games -- a REPRESENTATIVE, hand-authored harness.

HONEST FRAMING (read this first)
--------------------------------
ARC-AGI-3 (2025) is the "Interactive Reasoning Benchmark": 2D grid GAMES where an agent
perceives a colored grid, takes discrete actions, and must DISCOVER the hidden rules through
interaction to reach a goal (unlike ARC-AGI-1/2's static input->output puzzles).

This module is NOT the official ARC-AGI-3 evaluation. The official eval runs on a private
game server / API that we do not have. These are small, hand-authored ANALOGS in the same
*spirit* -- interactive rule-discovery on colored integer grids. Any solve rate reported here
is ONLY against a random-action baseline on THESE toy games. It says nothing about the real
ARC-AGI-3 leaderboard, nor about generalization to the official tasks.

Design constraints (all games)
------------------------------
- Pure standard library (random, collections); per-instance random.Random(seed); no wall-clock,
  no global RNG; deterministic (same seed -> byte-identical episode).
- Colored INTEGER grid, small (<= 10x10), discrete action set.
- A HIDDEN rule that can only be found by INTERACTING (bumping a closed door does nothing until
  you carry the key; a wrong-colored tile just won't let you step; etc.).
- Uniform interface so one generic player can drive all of them:

    game.name                      -> str
    game.n_actions                 -> int          (discrete action set size)
    game.action_names              -> tuple[str]
    game.max_steps                 -> int           (episode step cap for rollouts; enforced by
                                                     the RUNNER, never inside step())
    game.reset()                   -> grid          (list[list[int]])
    game.step(action)              -> (grid, reward, done)   reward in {0.0, 1.0}; done True on
                                                            a terminal (goal reached or dead end)
    game.snapshot()                -> hashable      (pure game state; NO step counter)
    game.restore(snapshot)         -> grid          (Go-Explore return-to-state; deterministic)

`step` NEVER ends an episode on a timeout -- only on a genuine terminal. The step cap lives in
the runner so Go-Explore can freely restore and expand states without a clock interfering.

Four games (the rule-classes named in the ARC-AGI-3 write-ups):
  key_door        -- a DEPENDENCY rule   : the door is solid until you have stepped on the key.
  push_box        -- a PUSH/physics rule : walking into the box shoves it one cell if the far
                                           side is clear (Sokoban-lite).
  color_lock      -- a COLOR-nav rule    : only cells of one hidden "safe" color are walkable.
  toggle_to_goal  -- a TOGGLE rule       : cycle each tile's color until every tile matches the
                                           hidden target shown on the beacon row.

Run `python research/arc_agi/env.py` for a short self-check.
"""

import random

# Small integer "colors". Kept < 16 so the player's one-hot spike encoder has a fixed palette.
EMPTY = 0
WALL = 1
AGENT = 2
KEY = 3
DOOR = 4
BOX = 5
TARGET = 6
GOAL = 7
MARK = 9  # toggle-game agent marker on its own row
COL0 = 10  # color_lock / toggle palette base (COL0, COL0+1, COL0+2)

_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right
_MOVE_NAMES = ("up", "down", "left", "right")


def _clone(grid):
    return [row[:] for row in grid]


# =====================================================================================================
# Game 1 -- key_door : DEPENDENCY rule (key must be collected before the door opens).
# =====================================================================================================


class KeyDoor:
    """5x5 open room. Reach the DOOR, but the door is SOLID until you have walked over the KEY.

    Hidden rule: stepping into the door does nothing while has_key is False (it reads as a wall);
    once the key has been collected the same move ends the episode with reward 1. Nothing in the
    grid announces the dependency -- the agent must bump the door, fail, find the key, and retry."""

    name = "key_door"
    n_actions = 4
    action_names = _MOVE_NAMES

    def __init__(self, seed=0, size=5):
        self.size = size
        self.max_steps = 4 * size * size
        rng = random.Random(seed)
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        self._start, self._key, self._door = cells[0], cells[1], cells[2]
        self.reset()

    def reset(self):
        self.ar, self.ac = self._start
        self.has_key = False
        return self.render()

    def render(self):
        g = [[EMPTY] * self.size for _ in range(self.size)]
        if not self.has_key:
            kr, kc = self._key
            g[kr][kc] = KEY
        dr, dc = self._door
        g[dr][dc] = DOOR
        g[self.ar][self.ac] = AGENT
        return g

    def step(self, action):
        dr, dc = _DELTAS[action]
        nr, nc = self.ar + dr, self.ac + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if (nr, nc) == self._door:
                if self.has_key:
                    self.ar, self.ac = nr, nc
                    return self.render(), 1.0, True
                # solid door -- no move
            else:
                self.ar, self.ac = nr, nc
                if (self.ar, self.ac) == self._key:
                    self.has_key = True
        return self.render(), 0.0, False

    def snapshot(self):
        return (self.ar, self.ac, self.has_key)

    def restore(self, snap):
        self.ar, self.ac, self.has_key = snap
        return self.render()


# =====================================================================================================
# Game 2 -- push_box : PUSH/physics rule (Sokoban-lite).
# =====================================================================================================


def _pushbox_solvable(size, agent, box, target):
    """BFS over (agent, box) states to confirm the box can reach the target. Used only during
    layout generation -- the player never sees this oracle."""
    from collections import deque

    seen = {(agent, box)}
    dq = deque([(agent, box)])
    while dq:
        (ar, ac), (br, bc) = dq.popleft()
        if (br, bc) == target:
            return True
        for dr, dc in _DELTAS:
            nr, nc = ar + dr, ac + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            nbox = (br, bc)
            if (nr, nc) == (br, bc):
                pr, pc = br + dr, bc + dc
                if not (0 <= pr < size and 0 <= pc < size):
                    continue
                nbox = (pr, pc)
            st = ((nr, nc), nbox)
            if st not in seen:
                seen.add(st)
                dq.append(st)
    return False


class PushBox:
    """5x5 room. Shove the BOX onto the TARGET. Walking into the box pushes it one cell in the
    same direction IF the cell beyond is inside the room; otherwise the move is blocked.

    Hidden rule: the PUSH mechanic itself (that touching the box moves it, and that a box against
    a wall cannot be pushed further). Layout is generated to be solvable."""

    name = "push_box"
    n_actions = 4
    action_names = _MOVE_NAMES

    def __init__(self, seed=0, size=5):
        self.size = size
        self.max_steps = 6 * size * size
        rng = random.Random(seed)
        for _ in range(200):
            cells = [(r, c) for r in range(size) for c in range(size)]
            rng.shuffle(cells)
            agent, box, target = cells[0], cells[1], cells[2]
            if box != target and _pushbox_solvable(size, agent, box, target):
                self._start, self._box0, self._target = agent, box, target
                break
        else:  # deterministic fallback (always solvable): a clear line agent->box->target
            self._start, self._box0, self._target = (2, 0), (2, 1), (2, 3)
        self.reset()

    def reset(self):
        self.ar, self.ac = self._start
        self.br, self.bc = self._box0
        return self.render()

    def render(self):
        g = [[EMPTY] * self.size for _ in range(self.size)]
        tr, tc = self._target
        g[tr][tc] = TARGET
        g[self.br][self.bc] = BOX
        g[self.ar][self.ac] = AGENT
        return g

    def step(self, action):
        dr, dc = _DELTAS[action]
        nr, nc = self.ar + dr, self.ac + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self.render(), 0.0, False
        if (nr, nc) == (self.br, self.bc):
            pr, pc = self.br + dr, self.bc + dc
            if not (0 <= pr < self.size and 0 <= pc < self.size):
                return self.render(), 0.0, False  # box against the wall -- blocked
            self.br, self.bc = pr, pc
            self.ar, self.ac = nr, nc
            if (self.br, self.bc) == self._target:
                return self.render(), 1.0, True
        else:
            self.ar, self.ac = nr, nc
        return self.render(), 0.0, False

    def snapshot(self):
        return (self.ar, self.ac, self.br, self.bc)

    def restore(self, snap):
        self.ar, self.ac, self.br, self.bc = snap
        return self.render()


# =====================================================================================================
# Game 3 -- color_lock : COLOR-navigation rule (one hidden safe color is walkable).
# =====================================================================================================


def _colorlock_solvable(size, colors, start, goal, safe):
    from collections import deque

    seen = {start}
    dq = deque([start])
    while dq:
        r, c = dq.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in _DELTAS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            walkable = (nr, nc) == goal or colors[nr][nc] == safe
            if walkable and (nr, nc) not in seen:
                seen.add((nr, nc))
                dq.append((nr, nc))
    return False


class ColorLock:
    """5x5 room where every cell has a color. Exactly ONE hidden "safe" color is walkable. Reach the
    GOAL cell -- but stepping onto any non-safe color SENDS THE AGENT BACK to the entrance.

    Hidden rule: WHICH of the palette colors is safe. The agent discovers it by trying moves onto
    different colors and noticing which one lets it advance instead of resetting it. Because a wrong
    step is punished (reset), flailing does not drift to the goal -- the safe-color rule is load-
    bearing. A safe-colored path from start to goal is guaranteed by construction."""

    name = "color_lock"
    n_actions = 4
    action_names = _MOVE_NAMES
    n_colors = 3

    def __init__(self, seed=0, size=5):
        self.size = size
        self.max_steps = 6 * size * size
        rng = random.Random(seed)
        palette = [COL0 + i for i in range(self.n_colors)]
        for _ in range(200):
            safe = rng.choice(palette)
            start = (rng.randrange(size), rng.randrange(size))
            goal = (rng.randrange(size), rng.randrange(size))
            if goal == start:
                continue
            # carve a safe-colored random walk from start toward goal, then fill the rest.
            colors = [[rng.choice(palette) for _ in range(size)] for _ in range(size)]
            r, c = start
            colors[r][c] = safe
            for _ in range(4 * size):
                colors[r][c] = safe
                if (r, c) == goal:
                    break
                r = min(size - 1, max(0, r + (1 if goal[0] > r else -1 if goal[0] < r else 0)))
                colors[r][c] = safe
                c = min(size - 1, max(0, c + (1 if goal[1] > c else -1 if goal[1] < c else 0)))
            if _colorlock_solvable(size, colors, start, goal, safe):
                self._start, self._goal, self._safe, self._colors = start, goal, safe, colors
                break
        else:  # trivial fallback: a straight safe row
            colors = [[COL0 + 1] * size for _ in range(size)]
            for c in range(size):
                colors[0][c] = COL0
            self._start, self._goal, self._safe, self._colors = (0, 0), (0, size - 1), COL0, colors
        self.reset()

    def reset(self):
        self.ar, self.ac = self._start
        return self.render()

    def render(self):
        g = [row[:] for row in self._colors]
        gr, gc = self._goal
        g[gr][gc] = GOAL
        g[self.ar][self.ac] = AGENT
        return g

    def step(self, action):
        dr, dc = _DELTAS[action]
        nr, nc = self.ar + dr, self.ac + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self.render(), 0.0, False
        if (nr, nc) == self._goal:
            self.ar, self.ac = nr, nc
            return self.render(), 1.0, True
        if self._colors[nr][nc] == self._safe:
            self.ar, self.ac = nr, nc
        else:
            self.ar, self.ac = self._start  # wrong color -> sent back to the entrance
        return self.render(), 0.0, False

    def snapshot(self):
        return (self.ar, self.ac)

    def restore(self, snap):
        self.ar, self.ac = snap
        return self.render()


# =====================================================================================================
# Game 4 -- toggle_to_goal : TOGGLE rule (cycle every tile to the hidden target color).
# =====================================================================================================


class ToggleToGoal:
    """A row of T tiles, each showing a color. Cycle each tile until EVERY tile matches the hidden
    target color (the target is shown on the top "beacon" row). Actions: move left, move right,
    or toggle the tile under the agent (color -> (color+1) mod C).

    Grid is 3 rows x T cols: row0 = beacon (target repeated), row1 = the tiles, row2 = agent marker.
    Hidden rule: that TOGGLE cycles the color, and that matching the beacon is the goal. The agent
    finds this by toggling and watching the reward."""

    name = "toggle_to_goal"
    n_actions = 3
    action_names = ("left", "right", "toggle")
    n_tiles = 3
    n_colors = 3

    def __init__(self, seed=0):
        self.max_steps = 40
        rng = random.Random(seed)
        self._target = rng.randrange(self.n_colors)
        while True:
            tiles = [rng.randrange(self.n_colors) for _ in range(self.n_tiles)]
            if any(t != self._target for t in tiles):  # not already solved
                self._tiles0 = tiles
                break
        self._astart = rng.randrange(self.n_tiles)
        self.reset()

    def reset(self):
        self.tiles = self._tiles0[:]
        self.apos = self._astart
        return self.render()

    def render(self):
        beacon = [COL0 + self._target] * self.n_tiles
        tile_row = [COL0 + t for t in self.tiles]
        agent_row = [MARK if c == self.apos else EMPTY for c in range(self.n_tiles)]
        return [beacon, tile_row, agent_row]

    def step(self, action):
        if action == 0:
            self.apos = max(0, self.apos - 1)
        elif action == 1:
            self.apos = min(self.n_tiles - 1, self.apos + 1)
        else:
            self.tiles[self.apos] = (self.tiles[self.apos] + 1) % self.n_colors
        if all(t == self._target for t in self.tiles):
            return self.render(), 1.0, True
        return self.render(), 0.0, False

    def snapshot(self):
        return (self.apos, tuple(self.tiles))

    def restore(self, snap):
        self.apos, tiles = snap
        self.tiles = list(tiles)
        return self.render()


# The suite. Each entry is a factory game_cls(seed) -> game.
GAMES = (KeyDoor, PushBox, ColorLock, ToggleToGoal)


def _fmt(grid):
    return "\n".join(" ".join("%2d" % v for v in row) for row in grid)


def main():
    print("ARC-AGI-3-STYLE harness -- REPRESENTATIVE analogs, NOT the official ARC-AGI-3 eval.")
    print("games:", ", ".join(g.name for g in GAMES))
    ok = True
    for cls in GAMES:
        g = cls(seed=1)
        grid = g.reset()
        # determinism: same seed reproduces the exact opening grid.
        assert cls(seed=1).reset() == grid, "%s not deterministic" % g.name
        # snapshot/restore round-trips.
        snap = g.snapshot()
        g.step(0)
        assert g.restore(snap) == grid, "%s restore mismatch" % g.name
        # there exists at least one action; stepping returns a well-formed transition.
        grid2, r, done = g.step(0)
        assert isinstance(r, float) and isinstance(done, bool) and len(grid2) >= 1
        rows, cols = len(grid), max(len(row) for row in grid)
        assert rows <= 10 and cols <= 10, "%s grid too big" % g.name
        print("  [ok] %-14s actions=%d grid=%dx%d" % (g.name, g.n_actions, rows, cols))
    print("opening grid of %s:" % GAMES[0].name)
    print(_fmt(KeyDoor(seed=1).reset()))
    print("env self-check OK" if ok else "env self-check FAILED")


if __name__ == "__main__":
    main()
