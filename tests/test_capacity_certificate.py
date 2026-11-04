"""specs/022 T111 (G3) P1: the capacity certificate -- the trigger's three-way diagnosis.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.4 (the diagnosis) and 2.9
C2 (the certificate fires on XOR with an EXACT lift witness, not on the separable twin, whose
separating witness reads 4/4 through the learner's real `route_text`; one doubled multiplier fails
the exact check). Fixtures are the learner's own `text_features` on hand-built texts (never a copy
of G2's module) at fixture seed 9900; nothing here builds a Brain or touches a lake.
"""

import copy
import threading
from dataclasses import replace
from fractions import Fraction

import pytest

pytest.importorskip("scipy")

from snn_data_lake.growth.capacity_certificate import (  # noqa: E402
    check_farkas,
    contradiction_count,
    current_keys,
    diagnose,
    farkas_witness,
    item_ranks,
    lift_keys,
    separating_witness,
)
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner  # noqa: E402

INTENTS = ["GREET", "GOAL", "ARM", "CONSCIOUS", "STATUS", "MEMORY", "SLEEP", "HELP", "OTHER"]
SEED = 9900
TEXTS = ["cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun"]
XOR = ["GREET", "GOAL", "GOAL", "GREET"]  # label = (first pair order) XOR (second pair order)
TWIN = ["GREET", "GREET", "GOAL", "GOAL"]  # label = first pair order only: separable


def _learner():
    return SpikingInteractionLearner(INTENTS, n_in=52, seed=SEED, t_post=6)


def _keys(learner, texts):
    return [lift_keys(r) for r in item_ranks(learner, texts)]


def _idx(labels):
    return [INTENTS.index(lab) for lab in labels]


def test_xor_has_no_contradiction():
    learner = _learner()
    assert contradiction_count(_keys(learner, TEXTS), _idx(XOR)) == 0


def test_planted_contradiction_is_counted():
    learner = _learner()
    texts = ["cat dog sun sky", "cat dog sun sky", "dog cat sky sun"]
    assert contradiction_count(_keys(learner, texts), _idx(["GREET", "GOAL", "GREET"])) == 2


def test_certificate_fires_on_xor_with_an_exact_witness():
    learner = _learner()
    keys = _keys(learner, TEXTS)
    w = farkas_witness(keys, _idx(XOR), len(INTENTS))
    assert w is not None
    assert all(isinstance(v, Fraction) and v >= 0 for v in w.lam.values())
    assert check_farkas(keys, _idx(XOR), w.lam)
    assert w.path in ("rational", "support")


def test_certificate_does_not_fire_on_the_separable_twin():
    learner = _learner()
    assert farkas_witness(_keys(learner, TEXTS), _idx(TWIN), len(INTENTS)) is None


def test_doubled_multiplier_fails_the_exact_check():
    learner = _learner()
    keys = _keys(learner, TEXTS)
    w = farkas_witness(keys, _idx(XOR), len(INTENTS))
    first = next(iter(w.lam))
    bad = dict(w.lam)
    bad[first] = bad[first] * 2
    assert not check_farkas(keys, _idx(XOR), bad)


def test_twin_labels_pushed_through_the_xor_witness_fail():
    learner = _learner()
    keys = _keys(learner, TEXTS)
    w = farkas_witness(keys, _idx(XOR), len(INTENTS))
    assert not check_farkas(keys, _idx(TWIN), w.lam)


def test_separating_witness_reads_the_twin_right_through_route_text():
    learner = _learner()
    sep = separating_witness(learner, TEXTS, TWIN)
    assert sep is not None
    assert sep.reads == TWIN  # the REAL float reader, ties to the lowest index
    assert learner.weights() == _learner().weights()  # the caller's learner is untouched


def test_separating_witness_is_absent_on_xor():
    assert separating_witness(_learner(), TEXTS, XOR) is None


def test_diagnose_xor_is_capacity():
    d = diagnose(_learner(), TEXTS, XOR)
    assert d.kind == "CAPACITY"
    assert d.k == 0 and d.witness is not None


def test_diagnose_twin_is_separable():
    d = diagnose(_learner(), TEXTS, TWIN)
    assert d.kind == "SEPARABLE"
    assert d.separator is not None


def test_diagnose_history_resolves_state():
    texts = ["sun sky", "sun sky"]
    history = ["cat sun sky", "dog sun sky"]
    d = diagnose(_learner(), texts, ["GREET", "GOAL"], history_texts=history)
    assert d.kind == "STATE"
    assert d.k == 2 and d.k_history == 0


def test_diagnose_contradiction_with_history_is_data():
    texts = ["sun sky", "sun sky"]
    history = ["cat sun sky", "cat sun sky"]
    d = diagnose(_learner(), texts, ["GREET", "GOAL"], history_texts=history)
    assert d.kind == "DATA"
    assert d.k_history == 2


def test_diagnose_state_and_capacity_in_one_pass():
    # current texts identical in pairs; the history form is XOR over (cue, probe order)
    texts = ["sun sky", "sun sky", "sky sun", "sky sun"]
    history = ["cat dog sun sky", "dog cat sun sky", "cat dog sky sun", "dog cat sky sun"]
    labels = ["GREET", "GOAL", "GOAL", "GREET"]
    d = diagnose(_learner(), texts, labels, history_texts=history)
    assert d.kind == "STATE+CAPACITY"


def test_support_path_recovers_an_exact_witness_without_rounding():
    # the fallback the certificate takes when the rounded LP point fails the exact check: solve
    # exactly on the LP's support (here the support of the rounded witness itself)
    from snn_data_lake.growth.capacity_certificate import _exact_support_solve, _farkas_system

    learner = _learner()
    keys = _keys(learner, TEXTS)
    ys = _idx(XOR)
    w = farkas_witness(keys, ys, len(INTENTS))
    cons, row_of, entries = _farkas_system(keys, ys, len(INTENTS))
    support = [cons.index(c) for c in w.lam]
    exact = _exact_support_solve(cons, entries, len(row_of), support)
    assert exact is not None and check_farkas(keys, ys, exact)


def _augmented(learner, texts, units):
    from snn_data_lake.growth.grown_code import latencies

    return [lift_keys(r, latencies(units, r, learner.tau)) for r in item_ranks(learner, texts)]


def test_a_capable_grown_unit_stops_the_certificate_on_the_augmented_code():
    # a two-synapse unit (the synapses holding ranks 0 and 2 of item 0) crosses threshold on item 1
    # only (latencies [None, 3, None, None], measured 2026-10-04): that one-item indicator
    # separates XOR in the lift
    from snn_data_lake.growth.grown_code import GrownUnit

    learner = _learner()
    ranks = item_ranks(learner, TEXTS)
    a, b = ranks[0].index(0), ranks[0].index(2)
    w = [0.0] * 52
    w[a], w[b] = 1.0, 1.0
    unit = GrownUnit(1, tuple(w), threshold=1.6, window=12, sign=1, out=(0.0,) * len(INTENTS))
    keys = _augmented(learner, TEXTS, (unit,))
    assert len({k[-1] for k in keys}) > 1  # the unit's latency is not constant across the items
    assert farkas_witness(_keys(learner, TEXTS), _idx(XOR), len(INTENTS)) is not None
    assert farkas_witness(keys, _idx(XOR), len(INTENTS)) is None


def test_an_incapable_grown_unit_keeps_the_certificate_firing():
    # a unit reading ONE word slot is a function of a single synapse's rank: the lift witness
    # survives (C16)
    from snn_data_lake.growth.grown_code import GrownUnit

    learner = _learner()
    ranks = item_ranks(learner, TEXTS)
    w = [0.0] * 52
    w[ranks[0].index(0)] = 2.0  # one synapse only: its latency is a function of that synapse's rank
    unit = GrownUnit(1, tuple(w), threshold=1.0, window=12, sign=1, out=(0.0,) * len(INTENTS))
    keys = _augmented(learner, TEXTS, (unit,))
    aug = farkas_witness(keys, _idx(XOR), len(INTENTS))
    assert aug is not None and check_farkas(keys, _idx(XOR), aug.lam)


def test_lift_covers_every_rank_not_only_the_contributing_ones():
    learner = _learner()
    ranks = item_ranks(learner, TEXTS)[0]
    assert (
        len(lift_keys(ranks)) == 52
    )  # the full lift: the soundness claim holds for any PSP kernel


# ---- cp1 #3: the diagnosis reads the FULL current code (frozen + grown segment) ---------------


def _two_synapse_unit(learner):
    from snn_data_lake.growth.grown_code import GrownUnit

    ranks = item_ranks(learner, TEXTS)
    w = [0.0] * 52
    w[ranks[0].index(0)], w[ranks[0].index(2)] = 1.0, 1.0
    return GrownUnit(1, tuple(w), threshold=1.6, window=12, sign=1, out=(0.0,) * len(INTENTS))


def test_diagnose_reads_the_grown_segment():
    learner = _learner()
    unit = _two_synapse_unit(learner)
    assert diagnose(learner, TEXTS, XOR).kind == "CAPACITY"
    d = diagnose(learner, TEXTS, XOR, units=(unit,))
    assert d.kind != "CAPACITY"  # the capable unit removes the lift witness on the current code
    assert d.kind == "UNRESOLVED" and "reader" in d.log.get("separator", "")


def test_diagnose_keeps_capacity_under_an_incapable_unit():
    from snn_data_lake.growth.grown_code import GrownUnit

    learner = _learner()
    w = [0.0] * 52
    w[item_ranks(learner, TEXTS)[0].index(0)] = 2.0
    unit = GrownUnit(1, tuple(w), threshold=1.0, window=12, sign=1, out=(0.0,) * len(INTENTS))
    assert diagnose(learner, TEXTS, XOR, units=(unit,)).kind == "CAPACITY"


def test_diagnose_counts_contradictions_on_the_full_code():
    learner = _learner()
    unit = _two_synapse_unit(learner)
    d = diagnose(learner, ["sun sky", "sun sky"], ["GREET", "GOAL"], units=(unit,))
    assert d.kind == "DATA" and d.k == 2


def _actual_learner():
    from snn_data_lake.readout import _IXN_INTENTS

    return SpikingInteractionLearner(list(_IXN_INTENTS), n_in=52, seed=SEED, t_post=6)


def _actual_context(learner, units=(), *, afferents=None, eligible=True):
    """Constructed real-port fixture, not an admission or learned Engine result."""
    from snn_data_lake.readout.posed_problem import unit_rows

    learner._g9_context = {
        "problem_id": "ii.xor",
        "units": unit_rows(units),
        "afferents": [[u.unit_id, list((afferents or {}).get(u.unit_id, ()))] for u in units],
        "t_post_g": 15.0,
        "out_eligible": [[eligible] * len(learner.responses) for _ in units],
    }
    return learner


def test_actual_context_starts_with_the_same_empty_port_capacity_certificate():
    learner = _actual_context(_actual_learner())
    assert diagnose(learner, TEXTS, XOR, actual_context=True).kind == "CAPACITY"
    assert diagnose(learner, TEXTS, TWIN, actual_context=True).kind == "SEPARABLE"


@pytest.mark.parametrize("sign", [1, -1])
def test_actual_grown_separator_is_verified_by_real_reader_and_preserves_parent(sign):
    learner = _actual_learner()
    unit = replace(_two_synapse_unit(learner), sign=sign)
    _actual_context(learner, (unit,))
    before = learner._snapshot()
    result = diagnose(learner, TEXTS, XOR, units=(unit,), actual_context=True)
    assert result.kind == "SEPARABLE"
    assert result.separator.reads == XOR
    assert learner._snapshot() == before
    assert len(result.separator.grown_outputs) == 1
    row = result.separator.grown_outputs[0]
    assert row["unit_id"] == unit.unit_id and all(sign * w >= 0 for w in row["out"])
    probe = copy.deepcopy(learner)
    for neuron, weights in zip(probe.neurons, result.separator.weights, strict=True):
        neuron.w = weights
    probe._g9_context["units"][0]["out"] = list(row["out"])
    assert [probe.route_text(t) for t in TEXTS] == XOR


def test_actual_context_refuses_absent_or_mismatched_port():
    learner = _actual_learner()
    with pytest.raises(ValueError, match="bound grown/return reader"):
        diagnose(learner, TEXTS, XOR, actual_context=True)
    unit = _two_synapse_unit(learner)
    _actual_context(learner, (unit,))
    with pytest.raises(ValueError, match="differ from the actual"):
        diagnose(learner, TEXTS, XOR, units=(replace(unit, threshold=1.7),), actual_context=True)


def test_actual_context_uses_return_arrivals_in_keys_and_witness():
    learner = _actual_learner()
    unit = _two_synapse_unit(learner)
    _actual_context(learner, (unit,))
    bare = current_keys(learner, TEXTS, actual_context=True)
    assert diagnose(learner, TEXTS, XOR, actual_context=True).kind == "SEPARABLE"
    # A strong uniform actual arrival makes the unit cross at t=1 for every item;
    # the nonlinear discriminating feature is gone, so the exact witness fires again.
    _actual_context(learner, (unit,), afferents={unit.unit_id: [[0.0, 10.0]]})
    fed = current_keys(learner, TEXTS, actual_context=True)
    assert fed != bare
    assert {keys[-1][-1] for keys in fed} == {1}
    assert diagnose(learner, TEXTS, XOR, actual_context=True).kind == "CAPACITY"


def test_actual_separator_cannot_change_ineligible_grown_output_weights():
    learner = _actual_learner()
    unit = _two_synapse_unit(learner)
    _actual_context(learner, (unit,), eligible=False)
    before = learner._snapshot()
    assert separating_witness(learner, TEXTS, XOR, actual_context=True) is None
    result = diagnose(learner, TEXTS, XOR, actual_context=True)
    assert result.kind == "UNRESOLVED"
    assert learner._snapshot() == before


def test_actual_inactive_or_nonspiking_unit_does_not_fake_separability():
    learner = _actual_learner()
    unit = _two_synapse_unit(learner)
    for absent in (replace(unit, active=False), replace(unit, threshold=1e6)):
        _actual_context(learner, (absent,))
        assert diagnose(learner, TEXTS, XOR, actual_context=True).kind == "CAPACITY"


def test_actual_witness_does_not_clone_or_invoke_bound_writer_callbacks():
    class Owner:
        def __init__(self):
            self.lock = threading.RLock()
            self.calls = 0

        def write(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("a diagnostic called its external writer")

    owner = Owner()
    learner = _actual_learner()
    _actual_context(learner, (_two_synapse_unit(learner),))
    learner.event_sink = owner.write
    learner.teach_sink = owner.write
    before = learner._snapshot()
    assert diagnose(learner, TEXTS, XOR, actual_context=True).kind == "SEPARABLE"
    assert learner.event_sink.__self__ is owner and learner.teach_sink.__self__ is owner
    assert owner.calls == 0 and learner._snapshot() == before


def test_actual_separator_cannot_rewrite_a_protected_shared_core():
    learner = _actual_learner()
    _actual_context(learner, (_two_synapse_unit(learner),), eligible=False)
    assert [learner.route_text(t) for t in TEXTS] != TWIN
    # Positive control: the same frozen input code admits a real separator when
    # its base weights may move. This is a diagnostic, not a learned solution.
    assert separating_witness(learner, TEXTS, TWIN, actual_context=True) is not None
    learner._g9_context["core_eligible"] = [False] * len(learner.responses)
    before = learner._snapshot()
    assert separating_witness(learner, TEXTS, TWIN, actual_context=True) is None
    assert diagnose(learner, TEXTS, TWIN, actual_context=True).kind == "UNRESOLVED"
    assert learner._snapshot() == before
