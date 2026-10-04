"""specs/022 T111 (G3) P1: the capacity certificate -- the trigger's three-way diagnosis.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.4 (the diagnosis) and 2.9
C2 (the certificate fires on XOR with an EXACT lift witness, not on the separable twin, whose
separating witness reads 4/4 through the learner's real `route_text`; one doubled multiplier fails
the exact check). Fixtures are the learner's own `text_features` on hand-built texts (never a copy
of G2's module) at fixture seed 9900; nothing here builds a Brain or touches a lake.
"""

from fractions import Fraction

import pytest

pytest.importorskip("scipy")

from snn_data_lake.growth.capacity_certificate import (  # noqa: E402
    check_farkas,
    contradiction_count,
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
