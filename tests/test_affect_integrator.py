"""
Edge tests for neuromodulation/affect_integrator.py: the four learned neuromodulators fuse into a
coupled (valence, arousal) core with the right sign structure, distinct affect regions self-organize
into distinct emergent emotion prototypes (same region -> same id), the serotonergic mood tone
enters slowly, and the same seed + protocol is deterministic. The full un-riggable protocol (with
printed numbers) lives in the module's own __main__ self-check; this file is the narrower
pytest-style coverage the rest of tests/ uses.

Run: pytest tests/test_affect_integrator.py
"""

import random

from snn_data_lake.neuromodulation.affect_integrator import AffectIntegrator


def test_threat_and_cortisol_give_negative_aroused_reward_gives_positive():
    v_neg, a_neg = AffectIntegrator(seed=0).integrate(
        threat=0.9, reward=0.0, mood=0.0, cortisol=0.9
    )
    v_pos, a_pos = AffectIntegrator(seed=0).integrate(
        threat=0.0, reward=0.9, mood=0.0, cortisol=0.0
    )

    assert v_neg < 0.0 < v_pos  # sign-separated valence
    assert v_pos - v_neg > 0.5  # by a clear margin
    assert a_neg > 0.4  # threat + load -> high LC-NE arousal
    assert a_neg > a_pos  # threat is more salient/arousing than a bare reward


def test_valence_and_arousal_stay_in_range():
    ai = AffectIntegrator(seed=3)
    rng = random.Random(0)
    for _ in range(20):
        v, a = ai.integrate(
            threat=rng.random(), reward=rng.random(), mood=rng.uniform(-1, 1), cortisol=rng.random()
        )
        assert -1.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0


def test_distinct_regions_grow_distinct_emotions_same_region_shares_id():
    regions = [(-0.75, 0.80), (0.75, 0.80), (0.70, 0.15), (-0.70, 0.20)]
    ai = AffectIntegrator(seed=0)
    assert ai.n_emotions == 0  # starts with no emergent categories
    assert ai.label(0.0, 0.5) == -1  # labelling before any growth -> no prototype

    rng = random.Random(1)
    for _ in range(8):
        for cv, ca in regions:
            ai.observe(cv + rng.uniform(-0.05, 0.05), ca + rng.uniform(-0.05, 0.05))

    assert ai.n_emotions == len(regions)  # each well-separated region grew its own prototype

    probe = random.Random(42)
    ids = []
    for cv, ca in regions:
        got = {
            ai.label(cv + probe.uniform(-0.05, 0.05), ca + probe.uniform(-0.05, 0.05))
            for _ in range(6)
        }
        assert len(got) == 1  # same region -> one stable emergent id
        ids.append(next(iter(got)))
    assert len(set(ids)) == len(regions)  # different regions -> different ids


def test_first_observation_recruits_then_familiar_refines():
    ai = AffectIntegrator(seed=0)
    _, grew_first = ai.observe(-0.7, 0.8)
    assert grew_first and ai.n_emotions == 1  # neurogenesis on the very first point
    _, grew_again = ai.observe(-0.69, 0.79)  # essentially the same point
    assert not grew_again and ai.n_emotions == 1  # familiar -> refine, no new prototype


def test_label_with_no_args_uses_last_integrate_point():
    ai = AffectIntegrator(seed=0)
    ai.observe(-0.75, 0.80)
    ai.observe(0.75, 0.80)
    v, a = ai.integrate(threat=0.9, reward=0.0, mood=0.0, cortisol=0.9)  # a negative/aroused point
    assert ai.label() == ai.label(v, a)  # no-arg labels the most recent core-affect point


def test_slow_serotonergic_mood_lifts_valence_over_repeats():
    ai = AffectIntegrator(seed=0)
    v_first, _ = ai.integrate(threat=0.0, reward=0.0, mood=1.0, cortisol=0.0)
    v_last = v_first
    for _ in range(12):
        v_last, _ = ai.integrate(threat=0.0, reward=0.0, mood=1.0, cortisol=0.0)
    assert v_last > v_first  # sustained good mood accumulates a positive tone
    assert v_last > 0.0


def test_same_seed_and_protocol_is_deterministic():
    regions = [(-0.75, 0.80), (0.75, 0.80), (0.70, 0.15), (-0.70, 0.20)]

    def build(seed):
        ai = AffectIntegrator(seed=seed)
        rng = random.Random(1)
        for _ in range(8):
            for cv, ca in regions:
                ai.observe(cv + rng.uniform(-0.05, 0.05), ca + rng.uniform(-0.05, 0.05))
        core = ai.integrate(threat=0.9, reward=0.0, mood=0.0, cortisol=0.9)
        return ai.signature(), core

    assert build(0) == build(0)  # identical emotion map + identical core affect
