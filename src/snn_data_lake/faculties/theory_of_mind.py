"""
Theory of Mind -- a functional FALSE-BELIEF tracker (the canonical Sally-Anne task).

BrainCog groups social cognition as its own capability; this repo had none. This module adds the
minimal primitive: an agent that tracks ANOTHER agent's BELIEF about where an object is, where that
belief can DIVERGE from the true world state. The whole point is a second belief population that is
NOT the world -- it updates only from what the other agent actually OBSERVED, so when the world
changes behind the other's back, its belief goes STALE (a false belief).

Reuse, not reinvention. The design mirrors faculties/self_model.py -- a belief/state population that
changes ONLY through observe() and otherwise persists. The believed location is carried as a genuine
spiking POPULATION CODE (reusing cortex/population_encoding.PopulationEncoder: N overlapping RBF
neurons tile the locations) and read back out by a spiking WINNER-TAKE-ALL over cortex/lif_neuron.LIFNeuron
(one leaky integrate-and-fire neuron per location, the most-firing location wins). Nothing here is a
re-implementation of those primitives.

The Sally-Anne task (Baron-Cohen/Leslie/Frith 1985), staged concretely:
  1. Sally puts the marble in basket A.  Sally AND Anne watch  -> both believe A; world = A.
  2. Sally LEAVES the room (absent).
  3. Anne moves the marble to basket B.  Only Anne watches     -> Anne believes B; world = B;
     Sally is absent, so her belief population PERSISTS at A (a FALSE belief).
  4. Sally RETURNS. Where will Sally LOOK?

The correct answer is A -- Sally looks where she BELIEVES the marble is, not where it really is. A
model that has a theory of Sally's mind reads her (false) belief and predicts A. A model that only
knows reality predicts B and is wrong. Those two predictors are exactly the claim and its control below.

Honest framing. This is a FUNCTIONAL false-belief tracker: a belief population that can lag the world.
It is NOT machine consciousness or genuine mentalizing, and it does not know "what it is like" to hold
a belief -- it models only THAT a belief can diverge from reality. The scenarios are toy and the readout
is near-tabular. Deterministic, ASCII.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.theory_of_mind
"""

import random

from snn_data_lake.growth.lif_neuron import LIFNeuron
from snn_data_lake.sensorimotor.population_encoding import PopulationEncoder

# spiking winner-take-all readout over the belief population
_WINDOW = 20  # steps each location's LIF neuron is driven
_TAU_M = 10.0  # LIF membrane time constant
_DRIVE = 0.3  # input current for the peak-count location (others scaled by count/peak)
_MAX_SPIKES = 12  # population-code peak spike count


def _loc_name(i):
    """Map a location index to a basket letter (0->A, 1->B, ...) for ASCII reporting; None -> '?'."""
    return "?" if i is None else chr(ord("A") + int(i))


class BeliefPopulation:
    """One agent's belief about WHERE an object is, held as a spiking population code.

    The believed location is encoded across N overlapping RBF neurons (PopulationEncoder) and read
    back out by a spiking winner-take-all -- one LIFNeuron per location driven by that location's
    spike count, the most-firing neuron wins. observe() is the ONLY way the belief changes: an agent
    not present to witness a change leaves this population untouched, so it can go STALE relative to
    the world -- a FALSE belief. This mirrors the self_model.py pattern of a state population that
    only moves on observation."""

    def __init__(self, n_locations, seed=0, max_spikes=_MAX_SPIKES):
        if n_locations < 2:
            raise ValueError("need n_locations >= 2")
        self.n_locations = n_locations
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance RNG (noise-robustness path)
        self.encoder = PopulationEncoder(
            n=n_locations, lo=0.0, hi=n_locations - 1, max_spikes=max_spikes
        )
        self.counts = None  # no belief formed yet

    def observe(self, location):
        """Fold a DIRECTLY observed location into the belief population (a fresh percept overwrites)."""
        if not 0 <= location < self.n_locations:
            raise ValueError(f"location {location} out of range [0,{self.n_locations})")
        self.counts = self.encoder.encode(float(location))

    def believed_location(self, window=_WINDOW, jitter=0):
        """Read the belief with a SPIKING winner-take-all: one LIFNeuron per location, each driven by
        a current proportional to its spike count over `window` steps; the most-firing location wins
        (ties -> lowest index). `jitter` (>0) perturbs the counts via the per-instance RNG to show the
        readout is noise-robust. Returns None if no belief has formed yet."""
        if self.counts is None:
            return None
        counts = self.counts
        if jitter:
            counts = [max(0, k + self.rng.randint(-jitter, jitter)) for k in counts]
        peak = max(counts)
        if peak <= 0:
            return None
        fired = []
        for k in counts:
            neuron = LIFNeuron(tau_m=_TAU_M)
            current = _DRIVE * k / peak
            fired.append(sum(neuron.run([current] * window)))
        return max(range(len(fired)), key=lambda i: (fired[i], -i))


class TheoryOfMind:
    """Tracks the TRUE world state PLUS a separate belief population per agent.

    place()/move() change the world AND update the belief of every agent PRESENT to observe -- absent
    agents keep their prior belief. predict_from_belief(agent) reads that agent's tracked belief (what
    it THINKS -> passes Sally-Anne); predict_from_world(agent) ignores the mind and returns reality
    (the CONTROL that FAILS the unobserved-change cases)."""

    def __init__(self, n_locations=4, seed=0, window=_WINDOW):
        self.n_locations = n_locations
        self.seed = seed
        self.window = window
        self.rng = random.Random(seed)  # per-instance RNG
        self.world = None  # true object location
        self.beliefs = {}  # agent -> BeliefPopulation

    def _belief(self, agent):
        if agent not in self.beliefs:
            self.beliefs[agent] = BeliefPopulation(self.n_locations, seed=self.seed)
        return self.beliefs[agent]

    def place(self, location, observers):
        """Place the object at `location`; update the world and every PRESENT observer's belief."""
        self.world = location
        for agent in observers:
            self._belief(agent).observe(location)

    def move(self, location, observers):
        """Move the object (identical mechanic; named for the Anne-moves-the-marble step)."""
        self.place(location, observers)

    def predict_from_belief(self, agent, jitter=0):
        """Predict where `agent` will LOOK from ITS TRACKED BELIEF (spiking WTA readout)."""
        return self._belief(agent).believed_location(self.window, jitter=jitter)

    def predict_from_world(self, agent):
        """CONTROL predictor: ignore the agent's mind, return reality. Fails false-belief cases.
        `agent` is accepted for signature parity with predict_from_belief but deliberately unused."""
        del agent
        return self.world


def _run_sally_anne(start, moved, sally_present_for_move, seed=0):
    """Stage the Sally-Anne task. Sally and Anne both watch the marble placed in `start`; then Anne
    moves it to `moved`, with Sally present only if sally_present_for_move. Returns the TheoryOfMind
    at the 'Sally returns' read-out moment."""
    tom = TheoryOfMind(n_locations=4, seed=seed)
    tom.place(start, observers=("Sally", "Anne"))  # both witness the placement
    present = ("Sally", "Anne") if sally_present_for_move else ("Anne",)
    tom.move(moved, observers=present)  # Anne moves it; Sally may be absent
    return tom


def main():
    lines = []

    def emit(s):
        lines.append(s)

    emit("=" * 78)
    emit("THEORY OF MIND -- a functional FALSE-BELIEF tracker (Sally-Anne)")
    emit("=" * 78)

    # false-belief scenarios: marble placed in `start`, then moved to `moved` while Sally is AWAY.
    # varied (start, moved) pairs across all four baskets prove nothing is hardcoded to one location.
    scenarios = [(0, 2), (1, 3), (2, 0), (3, 1)]

    # (1) BELIEF-TRACKING: predict Sally's look FROM HER BELIEF -> the false-belief location (start) ---
    emit("(1) BELIEF-TRACKING (predict from Sally's tracked belief)")
    belief_hits = 0
    world_hits = 0
    for start, moved in scenarios:
        tom = _run_sally_anne(start, moved, sally_present_for_move=False)
        look_belief = tom.predict_from_belief("Sally")
        look_world = tom.predict_from_world("Sally")
        correct = start  # she looks where SHE believes it is
        if look_belief == correct:
            belief_hits += 1
        if look_world == correct:
            world_hits += 1
        emit(
            f"    marble {_loc_name(start)}->{_loc_name(moved)} (Sally away): "
            f"belief-predict={_loc_name(look_belief)}  world-predict={_loc_name(look_world)}  "
            f"true-look={_loc_name(correct)}"
        )
        assert look_belief == start, (
            f"belief predictor missed the false-belief basket {_loc_name(start)}"
        )
    belief_acc = belief_hits / len(scenarios)
    world_acc = world_hits / len(scenarios)
    emit(f"    belief-predictor accuracy = {belief_acc:.2f}  (predicts the false-belief basket A)")
    assert belief_acc == 1.0, "belief predictor must pass every Sally-Anne false-belief case"

    # (2) CONTROL that FAILS: the reality predictor looks where the marble REALLY is (moved) -> wrong --
    emit("-" * 78)
    emit("(2) CONTROL -- predict from the TRUE world state (reality, not belief)")
    for start, moved in scenarios:
        tom = _run_sally_anne(start, moved, sally_present_for_move=False)
        look_world = tom.predict_from_world("Sally")
        assert look_world == moved, "world predictor should report reality (the moved-to basket)"
        assert look_world != start, (
            "world predictor must FAIL the false-belief case (predicts B not A)"
        )
    emit(
        f"    world-predictor accuracy = {world_acc:.2f}  (predicts B, where the marble really is)"
    )
    emit("    -> the tracker represents a BELIEF distinct from reality, not merely the world state")
    assert world_acc == 0.0, "the reality control must FAIL every false-belief case"

    # (3) TRUE-BELIEF control: Sally SAW the move -> belief == world, both predictors AGREE ------------
    emit("-" * 78)
    emit("(3) TRUE-BELIEF control (Sally present for the move)")
    for start, moved in scenarios:
        tom = _run_sally_anne(start, moved, sally_present_for_move=True)
        lb = tom.predict_from_belief("Sally")
        lw = tom.predict_from_world("Sally")
        emit(
            f"    marble {_loc_name(start)}->{_loc_name(moved)} (Sally present): "
            f"belief={_loc_name(lb)}  world={_loc_name(lw)}  agree={lb == lw}"
        )
        assert lb == lw == moved, "with a TRUE belief both predictors must agree on reality"
    emit("    -> divergence is SPECIFIC to the unobserved-change case, not a general disagreement")

    # (4) NOISE-ROBUST spiking readout + DETERMINISM ------------------------------------------------
    emit("-" * 78)
    emit("(4) ROBUSTNESS + DETERMINISM")
    tom_a = _run_sally_anne(0, 2, sally_present_for_move=False, seed=0)
    tom_b = _run_sally_anne(0, 2, sally_present_for_move=False, seed=0)
    det_ok = tom_a.predict_from_belief("Sally") == tom_b.predict_from_belief("Sally")
    noisy = _run_sally_anne(1, 3, sally_present_for_move=False, seed=0)
    noisy_look = noisy.predict_from_belief("Sally", jitter=2)  # perturb the population code
    emit(f"    identical seeds -> identical prediction : {det_ok}")
    emit(
        f"    spike-count jitter=2 -> WTA still recovers {_loc_name(noisy_look)} "
        f"(the belief basket {_loc_name(1)})"
    )
    assert det_ok, "identical seeds produced different predictions -- not deterministic"
    assert noisy_look == 1, "spiking WTA readout must survive population-code jitter"

    emit("=" * 78)

    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "HONEST NEGATIVE: this is a FUNCTIONAL false-belief TRACKER -- a belief population that can "
        "lag world state -- NOT machine consciousness or genuine mentalizing. It models THAT a belief "
        "can diverge from reality, not WHAT it is like to hold one; the scenarios are toy and the "
        "readout is near-tabular. Deterministic, ASCII."
    )
    print(
        "self-check OK: the belief predictor passes Sally-Anne (accuracy 1.00, predicts the "
        "false-belief basket A); the reality CONTROL fails (0.00, predicts B); with a true belief "
        "both predictors agree; the spiking WTA readout is noise-robust and deterministic"
    )


if __name__ == "__main__":
    main()
