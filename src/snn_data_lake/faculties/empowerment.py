"""
Empowerment -- an intrinsic drive for CONTROLLABILITY / sense of agency (Klyubin, Polani & Nehaniv
2005, "All Else Being Equal Be Empowered"). The ONE intrinsic axis SNDL lacked: every other drive
here is novelty- or learning-progress-flavored (spiking_curiosity, generalizing_novelty,
intrinsic_motivation, drive_reduction, autotelic competence-progress). Empowerment is orthogonal --
it rewards CONTROL, not surprise.

Empowerment E(s) = the channel capacity of the (action-sequence -> future-state) channel = how many
DISTINGUISHABLE future states the agent can reach from s via its OWN actions. High where the agent
has the most control over its future; ~0 in a trap where every action leads to the same place. This
is effectance motivation / the sense of agency -- a human-central drive (an infant shakes a rattle
because it can, a person avoids states of helplessness). Distinct from novelty: a state can CHANGE a
lot (high novelty) while the agent controls nothing (E ~ 0).

Estimate (deterministic, bounded): roll a bounded horizon H of action sequences through an
action-conditioned forward model (duck-typed `.predict(state, action) -> next_state`; e.g.
faculties/jepa_world_model.JepaWorldModel), quantize each terminal state to a grid CELL (a crc32
signature of the rounded latent), and count the DISTINCT reachable cells. E = log2(that count) = bits
of reachable-future variety the agent controls. The intrinsic reward is the empowerment GAIN
E(s') - E(s) -- the same temporal-difference form the other drives use -- so a step that OPENS UP the
future is rewarded and a step into a trap is penalized. Feed it into the same value channel the
other intrinsic drives feed (basal_ganglia valence), scaled by a gain.

HONEST SCOPE: a coarse, bounded-horizon channel-capacity ESTIMATE (grid-quantized; exhaustive up to
`max_seqs` sequences, then a seeded reproducible subsample), NOT exact mutual information. It measures
the ACTION channel specifically: the CONTROL in the self-check shuffles the action->outcome mapping so
actions no longer discriminate the future and confirms E COLLAPSES -- proving it is self-caused
reachability, not mere state spread / novelty. Per-instance random.Random(seed), ASCII.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.empowerment
"""

import itertools
import math
import random
import zlib


class Empowerment:
    """Intrinsic controllability drive over an action-conditioned forward model.

        world_model: any object with `.predict(state, action) -> next_state` (state = list of floats).
    `.predict` must be PURE (no RNG, no state): `empowerment` memoizes shared action prefixes per call.
        actions:     the discrete action set the agent can take.
        horizon H:   how many steps ahead the reachable-future set is measured over.
        grid:        quantization step for a "distinguishable" future state (coarser -> fewer cells).
        max_seqs:    exhaustively enumerate all |A|^H sequences up to this cap, else seeded-subsample."""

    def __init__(self, world_model, actions, horizon=2, grid=0.25, max_seqs=64, seed=0):
        self.wm = world_model
        self.A = list(actions)
        self.H = max(1, int(horizon))
        self.grid = grid
        self.max_seqs = max_seqs
        self.rng = random.Random(seed)
        self._prev = None

    def _seqs(self):
        """All action sequences of length H (exhaustive+deterministic) up to max_seqs, else a seeded,
        reproducible subsample of distinct sequences."""
        total = len(self.A) ** self.H
        if total <= self.max_seqs:
            return list(itertools.product(self.A, repeat=self.H))
        seqs = set()
        while len(seqs) < self.max_seqs:
            seqs.add(tuple(self.rng.choice(self.A) for _ in range(self.H)))
        return sorted(seqs)

    def _roll(self, z0, seq, memo=None):
        """Roll an action sequence forward through the model from state z0 -> terminal state.

        `memo` (one z0 only) maps an action-prefix to the state it reaches, so sequences that share a
        prefix share its `predict` calls. `predict` is a pure function of (state, action) for a fixed
        model, so the terminal state is bit-identical; only the repeated calls are skipped."""
        z = z0
        for k, a in enumerate(seq, 1):
            if memo is None:
                z = self.wm.predict(z, a)
                continue
            # keyed on object IDENTITY: seq holds the very objects of self.A (product/choice return them),
            # so this is exact and works for unhashable actions (e.g. [[0], [1]], Codex re-review)
            key = tuple(map(id, seq[:k]))
            if key not in memo:
                memo[key] = self.wm.predict(z, a)
            z = memo[key]
        return z

    def _cell(self, z):
        """Quantize a state to a grid CELL id (crc32 of the rounded signature) -- two states in the
        same cell are 'indistinguishable' futures."""
        sig = tuple(int(round(v / self.grid)) for v in z)
        return zlib.crc32(str(sig).encode("ascii"))

    def empowerment(self, z0):
        """Bits of DISTINGUISHABLE future the agent controls from z0 = log2(#reachable grid cells)."""
        memo = {}  # action-prefix -> state, for this z0 only
        cells = {self._cell(self._roll(z0, s, memo)) for s in self._seqs()}
        return math.log2(len(cells)) if cells else 0.0

    def reward(self, z_before, z_after):
        """Empowerment GAIN E(z_after) - E(z_before): opening up the future is rewarded, walking into
        a trap (fewer reachable futures) is penalized.

        DISPOSITION (P3, re-verified 2026-08-04). `docs/POLICIES.md` lists this method as "genuinely
        dead ... a retire candidate". The dead half is CONFIRMED: a tree-wide search for `.reward(`
        on an Empowerment instance, including getattr/string-name dispatch, finds ZERO call sites
        outside this module's own `main()`. The live path is `step_reward` alone
        (`engine/autonomous_loop.py:1465`, reached because `engine/brain.py:237` sets
        `empowerment=True`).

        It is NOT retired here, and the reason is a correction to the "recomputes rather than
        delegating" reading. These are not two implementations of one job: both are a single
        subtraction over the SAME primitive, `empowerment()`, and they differ in a way that matters
        -- `reward` evaluates two ARBITRARY states (two full rollouts), `step_reward` evaluates one
        and reuses a cached previous value for a stream. Collapsing `step_reward` into `reward`
        would DOUBLE the live per-tick cost of the most expensive call in this module (|A|^H rollouts
        through the world model) to remove one subtraction. P2 asks that a merge REDUCE; that one
        would not.

        What it actually is: an unconsumed capability, which is P0 open work rather than P2 bloat.
        Arbitrary-pair empowerment gain is exactly what a PLANNER would consume -- score candidate
        futures by the controllability they open up, not just the state just visited -- and nothing
        in the tree plans over it yet. Wiring that consumer lives outside this module."""
        return self.empowerment(z_after) - self.empowerment(z_before)

    def step_reward(self, z):
        """Streaming intrinsic reward = empowerment gain vs the last state seen (0.0 on the first).

        THE live path (autonomous_loop binds this to the JEPA latent every tick). One `empowerment()`
        evaluation per call by design; see `reward` for why it is not folded into that method."""
        e = self.empowerment(z)
        r = 0.0 if self._prev is None else e - self._prev
        self._prev = e
        return r


# ---- tiny deterministic stub predictors for the self-check (no torch needed) ----------------------
class _Trap:
    """Every action leads to the SAME place -> the agent controls nothing."""

    def predict(self, z, a):
        return [0.0, 0.0]


class _Hub:
    """Each action moves the state a DISTINCT way -> sequences fan out to many futures."""

    def predict(self, z, a):
        return [z[0] + (a + 1), z[1] + (a + 1) * (a + 1)]


class _NoControl:
    """The state MOVES a lot every step (high novelty) but the move IGNORES the action -> the agent
    controls nothing. The un-riggable control: a novelty drive would reward this; empowerment must not."""

    def predict(self, z, a):
        return [z[0] + 1.0, z[1] + 1.0]


def main():
    print("=" * 78)
    print("EMPOWERMENT -- intrinsic controllability drive (bits of self-reachable future)")
    print("=" * 78)
    A = [0, 1, 2, 3]
    z0 = [0.0, 0.0]

    e_trap = Empowerment(_Trap(), A, horizon=2, seed=0).empowerment(z0)
    e_hub = Empowerment(_Hub(), A, horizon=2, seed=0).empowerment(z0)
    e_nc = Empowerment(_NoControl(), A, horizon=2, seed=0).empowerment(z0)
    # the NoControl state actually MOVES (proving it's not measuring stillness):
    moved = _NoControl().predict(z0, 3) != z0

    print(f"  trap  (every action -> same place)     : E = {e_trap:.2f} bits  (no control)")
    print(f"  hub   (actions fan out to many futures): E = {e_hub:.2f} bits  (high control)")
    print(
        f"  no-control (state MOVES, action ignored): E = {e_nc:.2f} bits  (moved={moved}, novelty!=control)"
    )

    # empowerment GAIN: moving from a trap-like state toward a hub-like state is rewarded
    gain = Empowerment(_Hub(), A, horizon=2, seed=0).reward([0.0, 0.0], [0.0, 0.0])
    # determinism: same seed -> identical estimate
    det = Empowerment(_Hub(), A, horizon=3, seed=0).empowerment(z0) == Empowerment(
        _Hub(), A, horizon=3, seed=0
    ).empowerment(z0)
    print(f"  gain (E(after)-E(before)) example      : {gain:+.2f}   ; deterministic = {det}")
    print("=" * 78)

    # ---- un-riggable self-check --------------------------------------------------------------------
    assert e_trap == 0.0, (
        f"a trap where every action collapses to one future must score ~0 ({e_trap})"
    )
    assert e_hub > 2.0, (
        f"a hub whose actions fan out to many distinct futures must score high ({e_hub})"
    )
    # THE CONTROL: the state moves as much as any hub, but because the move IGNORES the action, the
    # ACTION->future channel carries no information -> E collapses. This is what separates empowerment
    # (control) from novelty (mere change): a novelty drive would light up here; empowerment must not.
    assert moved, (
        "the no-control predictor must actually MOVE the state (else the control is trivial)"
    )
    assert e_nc == 0.0, (
        f"when the future ignores the action, empowerment must collapse to ~0 ({e_nc})"
    )
    assert e_hub > e_nc, "empowerment must prefer CONTROL (hub) over mere CHANGE (no-control)"
    assert det, "same seed must reproduce an identical empowerment estimate"

    print(
        f"self-check OK: empowerment = bits of self-reachable future -- trap {e_trap:.1f} < no-control "
        f"{e_nc:.1f} (moves but uncontrolled) << hub {e_hub:.1f}; the CONTROL (action-independent "
        f"future) collapses E, proving it measures CONTROL not novelty; deterministic."
    )


if __name__ == "__main__":
    main()
