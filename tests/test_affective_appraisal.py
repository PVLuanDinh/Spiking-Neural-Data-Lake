"""
Edge tests for faculties/affective_appraisal.py's learned (valence, arousal) readout: training on
one context moves its appraisal toward the target (vs the pre-training/naive baseline), a different
(disjoint-feature) context is left exactly untouched, and the same seed + protocol is deterministic.
The full un-riggable protocol (with printed pre/post numbers) lives in the module's own __main__
self-check; this file is narrower edge coverage in the pytest style the rest of tests/ uses.

Run: pytest tests/test_affective_appraisal.py
"""

import random

from snn_data_lake.faculties.affective_appraisal import AffectiveAppraisal, _context, _jitter


def test_learning_moves_trained_context_closer_to_target_than_baseline():
    dim = 48
    ctx = _context(1, 0, dim // 2, dim)
    target_v, target_a = -0.70, 0.80

    net = AffectiveAppraisal(dim=dim, seed=0)
    pre_v, pre_a = net.appraise(ctx)
    pre_err = abs(target_v - pre_v) + abs(target_a - pre_a)

    trng = random.Random(7)
    for _ in range(25):
        net.learn(_jitter(ctx, trng), target_v, target_a, neuromod=1.0)
    post_v, post_a = net.appraise(ctx)
    post_err = abs(target_v - post_v) + abs(target_a - post_a)

    assert post_err < 0.5 * pre_err  # substantially closer to target than the naive baseline


def test_disjoint_context_stays_neutral_after_training_a_different_one():
    dim = 48
    ctx_trained = _context(1, 0, dim // 2, dim)
    ctx_other = _context(2, dim // 2, dim, dim)  # disjoint band -> shares no active features

    net = AffectiveAppraisal(dim=dim, seed=0)
    trng = random.Random(7)
    for _ in range(25):
        net.learn(_jitter(ctx_trained, trng), -0.70, 0.80, neuromod=1.0)

    assert net.appraise(ctx_other) == (0.0, 0.0)  # never-trained, non-overlapping context untouched


def test_same_seed_and_protocol_is_deterministic():
    dim = 48
    ctx = _context(1, 0, dim // 2, dim)

    def train(seed):
        net = AffectiveAppraisal(dim=dim, seed=0)
        trng = random.Random(seed)
        for _ in range(25):
            net.learn(_jitter(ctx, trng), -0.70, 0.80, neuromod=1.0)
        return net.appraise(ctx)

    assert train(7) == train(7)
