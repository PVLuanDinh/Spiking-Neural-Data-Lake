"""
Reaction-diffusion wave computer -- shortest paths from an excitable medium.

Steinbock, Toth & Showalter (Science 267:868-871, 1995) showed that a chemical
excitable medium (the Belousov-Zhabotinsky reaction) SOLVES a maze: ignite a wave at
the start and, because the wavefront travels at a CONSTANT speed, the time it FIRST
arrives at any point equals the GEODESIC distance from the start through the free
space. Obstacles bend the wave; the trailing REFRACTORY zone stops a front from ever
running backwards or passing through another front (head-on fronts ANNIHILATE).
Back-tracking down the arrival-time gradient from the target then traces the shortest
path. This is "reaction-diffusion computing": geometry solved by physics, not by an
explicit graph search.

Honest discrete realization -- an EXCITABLE cellular automaton. Each cell is one of
REST, EXCITED, REFRACTORY(k), or WALL. Synchronous 4-neighbour (von Neumann) update:
  * REST non-wall  -> EXCITED at t+1  iff >=1 neighbour is EXCITED at t
  * EXCITED        -> REFRACTORY(r) at t+1
  * REFRACTORY(k)  -> REFRACTORY(k-1) -> ... -> REST     (r >= 1)
The refractory tail is the whole point: a cell that just fired cannot be re-excited,
so the wave is ONE-DIRECTIONAL (it never re-enters its own wake) and two fronts
ANNIHILATE on contact rather than crossing. arrival_time[cell] = the step a cell FIRST
became EXCITED from a single ignition source.

Why this is a genuine reaction-diffusion computation and not flood-fill relabeled: an
INDEPENDENT breadth-first search (collections.deque, a plain FIFO queue) computes the
geodesic distance over the same 4-connected free grid by a completely different code
path, and the PRIMARY assert is arrival_time == bfs_dist for every reachable cell. Two
unrelated algorithms agreeing that the first-arrival front == the geodesic is a real
correctness result, not a tautology. The excitable-medium SIGNATURE that plain diffusion
lacks -- every cell is EXCITED on exactly one step (the refractory wake blocks
re-excitation; head-on fronts annihilate) -- is asserted directly, single- and
two-source.

On thesis for this repo: a future arm-loop path planner can ignite a wave from the goal
over an occupancy grid and steer the end effector down the arrival-time gradient.

Pure standard library (collections.deque), no randomness -- a FIXED hand-designed maze
and a deterministic synchronous update. ASCII-only output.

  python -m snn_data_lake.reaction_diffusion_compute     # (also run by run_selfchecks.py)

Papers:
  Steinbock, O., Toth, A. & Showalter, K. (1995), "Navigating complex labyrinths:
    optimal paths from chemical waves", Science 267(5199):868-871.
  Adamatzky, A. (2005), "Reaction-Diffusion Computers", Elsevier.
"""
from collections import deque

# ---- the fixed maze (hand-designed so the shortest path forces a long detour) ----
# '#' wall, '.' free, 'S' source/start, 'T' target. The medium is 4-connected. The ONLY
# two gaps (row 3 has a single opening at col 7, row 5 a single opening at col 1) force a
# serpentine: down the right edge, across, down the left edge, across to the target. The
# geodesic S(1,1) -> T(7,7) is 24 steps while the Manhattan distance is only 12 -- so the
# recovered path is provably a real detour, not a straight line. Hand-checkable: 8 (down
# to the row-3 gap) + 8 (across row 4 to the row-5 gap) + 8 (across to T) = 24.
MAZE = [
    "#########",
    "#S......#",
    "#.......#",
    "#######.#",
    "#.......#",
    "#.#######",
    "#.......#",
    "#......T#",
    "#########",
]
KNOWN_GEODESIC = 24    # hand-checked shortest S->T path length on the fixed maze
REFRACTORY = 2         # r >= 1; the refractory tail enforces one-directional propagation

REST, EXCITED = "REST", "EXCITED"    # a refractory cell is stored as an int k >= 1


def parse(maze):
    """Return (free-cell set, source cell, target cell). Walls are simply absent."""
    free, src, tgt = set(), None, None
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            if ch != "#":
                free.add((r, c))
                if ch == "S":
                    src = (r, c)
                elif ch == "T":
                    tgt = (r, c)
    return free, src, tgt


def neighbours(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))   # 4-connected von Neumann


def bfs_dist(free, sources):
    """INDEPENDENT geodesic oracle: plain breadth-first search (a FIFO deque) over the
    4-connected free cells from one or more sources. A different code path from the
    excitable CA below -- that is what makes their agreement a real result."""
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        cell = q.popleft()
        for nb in neighbours(cell):
            if nb in free and nb not in dist:
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return dist


def excite_wave(free, sources, refractory=REFRACTORY):
    """Run the excitable cellular automaton from one or more ignition cells until the
    medium is quiescent. Returns (arrival, events): arrival[cell] is the step the cell
    FIRST fired; events is the list of (step, cell) firings -- one entry per excitation,
    so len(events) == number of cells iff every cell fires exactly once (the refractory
    wake blocks re-excitation -- the annihilation signature)."""
    state = {cell: REST for cell in free}
    for s in sources:
        state[s] = EXCITED
    arrival = {s: 0 for s in sources}
    events = [(0, s) for s in sources]
    excited = set(sources)
    t = 0
    while excited:
        nxt, newly = {}, set()
        for cell in free:
            st = state[cell]
            if st == REST:
                if any(state.get(nb) == EXCITED for nb in neighbours(cell)):
                    nxt[cell] = EXCITED
                    newly.add(cell)
                else:
                    nxt[cell] = REST
            elif st == EXCITED:
                nxt[cell] = refractory                      # EXCITED -> REFRACTORY(r)
            else:                                            # REFRACTORY(k)
                nxt[cell] = REST if st == 1 else st - 1
        state = nxt
        t += 1
        for cell in newly:
            arrival.setdefault(cell, t)
            events.append((t, cell))
        excited = newly
    return arrival, events


def backtrace(arrival, target, source):
    """Steepest descent on the arrival-time field: from the target, repeatedly step to a
    4-neighbour whose arrival time is exactly one less, until the source. Because
    arrival == geodesic distance, this recovers a shortest path."""
    path, cell = [target], target
    while cell != source:
        t = arrival[cell]
        nxt = next((nb for nb in neighbours(cell) if arrival.get(nb) == t - 1), None)
        assert nxt is not None, f"steepest descent stuck at {cell} (arrival {t})"
        cell = nxt
        path.append(cell)
    return path


def render(maze, path=()):
    """ASCII render; overlay '*' on interior path cells, keep the 'S'/'T' glyphs."""
    pathset = set(path)
    rows = []
    for r, row in enumerate(maze):
        rows.append("".join(
            ch if ch in "ST" else ("*" if (r, c) in pathset else ch)
            for c, ch in enumerate(row)))
    return "\n".join(rows)


def main():
    print("=" * 64)
    print("REACTION-DIFFUSION WAVE COMPUTER -- shortest paths from excitable fronts")
    print("=" * 64)
    free, src, tgt = parse(MAZE)
    print("maze ('#' wall, '.' free, 'S' start, 'T' target):")
    print(render(MAZE))
    print()

    # ---- single ignition at S: the excitable wave fills the maze ----
    arrival, events = excite_wave(free, [src])
    bdist = bfs_dist(free, [src])
    reachable = set(bdist)
    agree = all(arrival[cell] == bdist[cell] for cell in reachable)
    print(f"reachable free cells: {len(reachable)} (of {len(free)} total)")
    print(f"first-arrival wave == BFS geodesic for every reachable cell: {agree}")

    # ---- shortest path by steepest descent on the arrival-time field ----
    path = backtrace(arrival, tgt, src)
    steps = len(path) - 1
    manhattan = abs(src[0] - tgt[0]) + abs(src[1] - tgt[1])
    print(f"shortest path S->T: {steps} steps "
          f"(Manhattan {manhattan}, known geodesic {KNOWN_GEODESIC}) -- recovered path '*':")
    print(render(MAZE, path))
    print()

    # ---- excitable-medium SIGNATURE: two head-on fronts annihilate on collision ----
    arr2, ev2 = excite_wave(free, [src, tgt])
    bdist2 = bfs_dist(free, [src, tgt])
    once = len(ev2) == len(free)
    minmatch = all(arr2[cell] == bdist2[cell] for cell in free)
    print(f"two sources (S and T): excitation events {len(ev2)} == free cells {len(free)}"
          f" -> each cell fires once: {once}")
    print(f"each cell claimed by the NEARER front (arrival == min-distance BFS): {minmatch}")
    print("=" * 64)

    # ---------------- self-check (deterministic, emergent -- nothing hardcoded) --------
    # 1. PRIMARY -- the excitable wave computes the geodesic: first-arrival == BFS. Two
    #    independent algorithms (synchronous CA relaxation vs a deque BFS) agree.
    assert reachable, "no reachable cells"
    assert all(arrival[cell] == bdist[cell] for cell in reachable), "wave != geodesic"
    # 2. ONE-DIRECTIONAL propagation -- no cell is ever excited on two different steps
    #    (the refractory tail blocks re-excitation): one firing event per reachable cell.
    assert len(events) == len(reachable), "a cell fired more than once (single source)"
    assert len({cell for _, cell in events}) == len(events), "duplicate excitation event"
    # 3. SHORTEST PATH recovered by steepest descent equals the known geodesic, is a real
    #    detour (longer than Manhattan), 4-adjacent, non-wall, and runs T -> S.
    assert steps == bdist[tgt] == KNOWN_GEODESIC, f"path {steps} != geodesic {bdist[tgt]}"
    assert path[0] == tgt and path[-1] == src, "path must start at target, end at source"
    for a, b in zip(path, path[1:]):
        assert b in free and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"bad step {a}->{b}"
    assert steps > manhattan, "maze must force a real detour (geodesic > Manhattan)"
    # 4. ANNIHILATION -- two head-on fronts do NOT cross: every free cell fires exactly
    #    once even with two sources, and each is claimed by its nearer front.
    assert len(ev2) == len(free), "two-source: a cell fired twice (fronts crossed)"
    assert len({cell for _, cell in ev2}) == len(ev2), "two-source duplicate excitation"
    assert all(arr2[cell] == bdist2[cell] for cell in free), "two-source arrival != min-BFS"
    # 5. deterministic: no RNG -> byte-identical arrival field on a repeat run.
    assert excite_wave(free, [src])[0] == arrival, "nondeterministic wave"
    print(f"self-check OK: excitable wave == BFS geodesic on {len(reachable)} cells; "
          f"shortest path {steps} steps (Manhattan {manhattan}) recovered by steepest "
          f"descent; single- & two-source annihilation (each cell fires exactly once); "
          f"deterministic (zero RNG)")


if __name__ == "__main__":
    main()
