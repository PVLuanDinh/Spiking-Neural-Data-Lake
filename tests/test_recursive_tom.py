"""
Edge tests for recursive_tom: N-agent, depth-k recursive Theory of Mind over VSA role-filler
propositions read out by a spiking LIF winner-take-all. Covers the behaviors the __main__ self-check
exercises -- false-belief transfer (an absent agent's belief stays stale), a divergent second-order
belief, non-location attitudes (wants/knows) with nesting, depth-3 nesting, and determinism.

(This list ended "and zero-dependency purity" until 2026-08-04. No test in this file has ever
asserted that, and P7 abolished the rule it named -- so the docstring was advertising coverage that
does not exist, for a property the project no longer holds. Both halves are wrong independently,
which is why it goes rather than gets a purity test written for it.)

Run: pytest tests/test_recursive_tom.py
"""

from snn_data_lake.cortex.recursive_tom import RecursiveToM


def test_false_belief_transfer_and_world_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("B", "believes", "Y", witnesses={"B"})  # A absent for the move
    assert t.predict_first_order("A") == "X"  # false belief persists
    assert t.world_content("believes") == "Y"  # world moved on
    assert t.predict_first_order("A") != t.world_content("believes")


def test_absent_agent_belief_does_not_update():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A"})
    t.observe_event("B", "believes", "Y", witnesses={"B"})  # A not present -> untouched
    assert t.predict_first_order("A") == "X"
    assert t.predict_first_order("B") == "Y"


def test_second_order_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("A", "believes", "Y", witnesses={"A"})  # B absent for A's update
    assert t.predict_first_order("A") == "Y"
    assert t.predict_nested("B", "A") == "X"
    assert t.divergence("B", "A") is True


def test_true_belief_has_no_divergence():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})
    t.observe_event("A", "believes", "Y", witnesses={"A", "B"})  # B present -> tracks the update
    assert t.predict_nested("B", "A") == "Y"
    assert t.divergence("B", "A") is False


def test_non_location_attitudes_store_and_nest():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("C", "wants", "cookie", witnesses={"C", "A"})
    t.observe_event("C", "knows", "secret", witnesses={"C"})
    assert t.predict_first_order("C", "wants") == "cookie"
    assert t.predict_first_order("C", "knows") == "secret"
    assert t.predict_nested("A", "C", "wants") == "cookie"
    assert t.predict_nested("A", "C", "knows") is None  # A didn't witness C learn it


def test_depth_three_nesting():
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "P", witnesses={"A", "B", "C"})
    assert t.predict_chain(("C", "B", "A")) == "P"
    assert t.predict_chain(("B", "A")) == "P"
    assert t.predict_first_order("A") == "P"


def test_unheld_paths_return_none():
    t = RecursiveToM(n_agents=3, seed=0)
    assert t.predict_first_order("A") is None  # nothing observed yet
    t.observe_event("A", "believes", "X", witnesses={"A"})
    assert t.predict_nested("B", "A") is None  # B never modeled A


def test_determinism_and_noise_robustness():
    def build():
        t = RecursiveToM(n_agents=3, seed=0)
        t.observe_event("A", "believes", "X", witnesses={"A", "B"})
        t.observe_event("B", "believes", "Y", witnesses={"B"})
        return t

    a, b = build(), build()
    assert a.predict_first_order("A") == b.predict_first_order("A")
    assert a.predict_nested("B", "A") == b.predict_nested("B", "A")
    assert a.predict_first_order("A", jitter=2) == "X"  # population readout survives jitter
