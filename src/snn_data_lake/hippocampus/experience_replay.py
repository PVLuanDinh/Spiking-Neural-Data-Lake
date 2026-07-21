"""
v2.10 - experience_replay: sharp-wave-ripple (SWR) style offline replay for
sample-efficient value learning.

Grounded in Nasrin et al. 2021, "Spiking Neural Networks and Hippocampal Function"
(Cognitive Systems Research; DOI 10.1016/j.cogsys.2021.07.008): during quiescence the
hippocampus replays previously stored experience in FORWARD and REVERSE order at a
compressed timescale (sharp-wave ripples), consolidating memory so an animal learns
more from each real interaction. This is the same idea as reinforcement-learning
experience replay (Lin 1992; Mnih et al. 2015, DQN).

`ReplayBuffer` is a fixed-capacity ring of (unit, action, reward) transitions harvested
from live ticks. `replay(q, ...)` re-applies the SAME reward-prediction-error update the
live loop uses to a sampled batch, sweeping it forward then reverse -> the value table
converges toward the rewarded action in FEWER real interactions. It pairs directly with
autonomous_loop.AutonomousArm (the arm's grown-unit contextual action-value table).

SAFETY: replay ONLY mutates the value table `q`. It NEVER selects an action, NEVER calls
the arm, NEVER emits a motor command -- it is value consolidation, not control. In the arm
loop it runs on quiescent ticks only, strictly after the safety-gated action has already
been applied; the reflex / E-STOP / collision gate is entirely upstream and untouched.

This improves SAMPLE EFFICIENCY of the value table -- it is NOT cognition /
planning / a world model. Ceiling: uniform-sampled tabular replay over discrete grown
units (no prioritization, no model, no generative rollouts); the streams are synthetic.
Additive / opt-in: nothing replays unless you build a buffer and call replay().

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.experience_replay
"""

import random


class ReplayBuffer:
    """Fixed-capacity ring buffer of (unit, action, reward) transitions + SWR-style replay."""

    def __init__(self, capacity=256, seed=0):
        self.capacity = capacity
        self.buf = []  # (unit, action, reward) transitions
        self._i = 0  # ring write pointer
        self.rng = random.Random(seed)  # seeded -> deterministic sampling

    def __len__(self):
        return len(self.buf)

    def add(self, unit, action, reward):
        """Record one live transition (overwrites oldest once full)."""
        t = (unit, action, reward)
        if len(self.buf) < self.capacity:
            self.buf.append(t)
        else:
            self.buf[self._i] = t
            self._i = (self._i + 1) % self.capacity

    def sample(self, n):
        """n transitions sampled without replacement (fewer if the buffer is small)."""
        n = min(n, len(self.buf))
        return self.rng.sample(self.buf, n) if n else []

    def replay(self, q, lr, batch=16, mod=1.0):
        """Consolidate a sampled batch into value table `q` with the live loop's RPE rule,
        swept FORWARD then REVERSE (SWR). q[unit][action] += lr*mod*(reward - q[unit][action]).
        Only mutates existing q entries; returns the number of value updates applied.
        SAFETY: no action is chosen or emitted here -- value learning only."""
        picks = self.sample(batch)
        n = 0
        for unit, action, reward in picks + picks[::-1]:  # forward sweep then reverse sweep
            row = q.get(unit)
            if row is not None and action in row:
                row[action] += lr * mod * (reward - row[action])
                n += 1
        return n


def main():
    ACTIONS = ("A", "B", "C")
    REWARD = {"A": 1.0, "B": -1.0, "C": -1.0}  # A is the rewarded action

    def live_learn(steps, lr=0.3, replay_batch=0, seed=0):
        """Drive one unit with `steps` live rewarded ticks; optionally replay each tick.
        Returns (q_row, live_ticks_used). Replay reuses ONLY past stored transitions."""
        q = {0: dict.fromkeys(ACTIONS, 0.0)}
        buf = ReplayBuffer(seed=seed)
        rng = random.Random(seed)
        for _ in range(steps):
            a = rng.choice(ACTIONS)  # explore uniformly (same stream both modes)
            r = REWARD[a]
            q[0][a] += lr * (r - q[0][a])  # the live RPE update
            buf.add(0, a, r)
            if replay_batch:
                buf.replay(q, lr, batch=replay_batch)  # SWR consolidation (value table only)
        return q[0]

    def greedy(row):
        return max(ACTIONS, key=lambda a: row[a])

    print("=" * 60)
    print("EXPERIENCE REPLAY (SWR forward+reverse -> sample-efficient value)")
    print("=" * 60)

    # 1 SAMPLE EFFICIENCY: with the SAME live stream, replay reaches a more decided value
    # for the rewarded action A than no-replay (it consolidates each tick's transitions).
    no_rep = live_learn(steps=12, replay_batch=0, seed=1)
    rep = live_learn(steps=12, replay_batch=16, seed=1)
    assert greedy(no_rep) == "A" == greedy(rep), "both must eventually prefer the rewarded action"
    margin_no = no_rep["A"] - max(no_rep["B"], no_rep["C"])
    margin_rep = rep["A"] - max(rep["B"], rep["C"])
    assert margin_rep > margin_no, (
        f"replay must sharpen the value gap ({margin_rep:.3f} <= {margin_no:.3f})"
    )

    # 2 FEWER LIVE TICKS: replay hits a decisive A-margin in fewer real interactions.
    def ticks_to_margin(target, replay_batch, seed=2):
        q = {0: dict.fromkeys(ACTIONS, 0.0)}
        buf = ReplayBuffer(seed=seed)
        rng = random.Random(seed)
        for t in range(1, 200):
            a = rng.choice(ACTIONS)
            r = REWARD[a]
            q[0][a] += 0.3 * (r - q[0][a])
            buf.add(0, a, r)
            if replay_batch:
                buf.replay(q, 0.3, batch=replay_batch)
            if q[0]["A"] - max(q[0]["B"], q[0]["C"]) >= target:
                return t
        return 200

    live_only = ticks_to_margin(1.2, replay_batch=0)
    with_replay = ticks_to_margin(1.2, replay_batch=16)
    assert with_replay < live_only, (
        f"replay must need fewer live ticks ({with_replay} vs {live_only})"
    )

    # 3 VALUE-ONLY: replay returns an int update count and mutates only q (no action, no emit).
    q = {0: dict.fromkeys(ACTIONS, 0.0)}
    b = ReplayBuffer(seed=3)
    for _ in range(20):
        b.add(0, "A", 1.0)
    updates = b.replay(q, 0.3, batch=8)
    assert isinstance(updates, int) and updates == 16, "forward+reverse over batch=8 -> 16 updates"
    assert q[0]["A"] > 0.0 and q[0]["B"] == 0.0, "replay only touches the transitions' own entries"

    # 4 DETERMINISM: same seed -> identical replayed value table.
    assert live_learn(12, 16, seed=5) == live_learn(12, 16, seed=5), "same seed -> identical q"

    print(f"1 SAMPLE EFF : A-margin replay={margin_rep:.3f} > no-replay={margin_no:.3f}")
    print(
        f"2 FEWER TICKS: live ticks to a decisive margin replay={with_replay} < live-only={live_only}"
    )
    print(f"3 VALUE-ONLY : {updates} value updates (fwd+rev), no action chosen/emitted")
    print("4 DETERMINISM: same seed -> identical replayed value table")
    print("=" * 60)
    print(
        f"self-check OK: SWR replay sharpens value (margin {margin_rep:.3f}>{margin_no:.3f}) in fewer "
        f"live ticks ({with_replay}<{live_only}); value-table-only; deterministic"
    )


if __name__ == "__main__":
    main()
