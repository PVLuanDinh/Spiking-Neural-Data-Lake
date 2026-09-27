"""specs/022 -- the JEPA-collapse arc (2026-09-05): the arm's world model must read succession.

The arm builds `cerebellum/jepa_world_model.JepaWorldModel(ACTIONS, in_dim=64, seed)` at the
module defaults and trains it online on every wake tick; `neuromodulation/spiking_curiosity`
is fed its prediction error. Measured in-brain (evidence/w2-context-latents.md sections
4.2-4.3b) and brain-free (evidence/jepa-collapse.md sections 3-11): at the defaults this tree
had before the arc's fix commit, the latent COLLAPSED (its spread over distinct patterns fell to
x0.005-0.12 of a fresh model's) and the prediction error was the same under a deterministic
succession and under a per-cycle permutation of it -- the model could not tell them apart, so no
curiosity built on that error could see structure.

Four tests, in the order the arc's protocol runs them (seeds 0-2 as pre-registered; 10 cycles,
stricter than the 12 the record measured at):
  * THE WORLD IS LEARNABLE (the positive control at the mechanism's input contract, P8 clause
    2): on the planted 3-bit successor world a tabular oracle reaches ~0 error from cycle 2 and
    a plain linear predictor on the raw 64-d pattern -- fit on cycles 1-5 with a regularised
    least squares, scored on the id bands of cycles 6-10 -- separates ordered from shuffled by
    an absolute AND a relative bar (the arc's panel found a first draft passing by numerical
    accident: rank 24 of 65 with rcond=None, both errors ~1e17).
  * THE MODEL AT THE ARM'S DEFAULTS READS IT -- ordered error under half the shuffled twin's by
    cycle 10 on all three seeds. RED on the tree before the fix commit: this file landed one
    commit BEFORE the defaults changed and failed there (the arc's gate-before-rule; the counted
    red is evidence/jepa-collapse.md section 3).
  * THE ABLATION DOES NOT (the control that is not the method): `anti_collapse=False` at the
    shipped configuration must NOT clear the order bar and must collapse -- if it cleared, the
    attribution to the anti-collapse term would be void (record section 2.4).
  * THE LATENT AT THE ARM'S DEFAULTS DOES NOT COLLAPSE -- the trained model's spread over the
    world's distinct patterns stays above a quarter of a fresh model's and latent_std() above
    the module's own 0.05 floor. Likewise red before the fix.
  * THE BATCH MUST CARRY DIVERSITY (the live-floor arc, evidence/jepa-live-floor.md section 2.2
    Amendment 1, 2026-09-05): on the same world with every (vision, audio) pair shown 32 consecutive
    steps (the proprioception walk still moving, so consecutive patterns are near-duplicates as the
    live stream's are), the model at the arm's defaults must still read the succession AND its
    latent_std() must clear the module's own 0.05 floor. RED on the tree, and a STANDING red (a
    strict xfail; the reservoir that fixed it did not land -- sections 3.4b and 5 there). In the
    archived harness (stage05/) the consecutive ring buffer drove the encoder to the isotropic
    ceiling, the predictor read NO order (ordered/shuffled 4.1-4.6) and the consecutive gauge read
    0.001; this test's own count at commit C was seed 0's first conjunct (x4.45, -x halted).
Slow (pure-Python JEPA steps; ~18 minutes in total); marked accordingly.
"""

import random

import numpy as np
import pytest

from snn_data_lake.brainstem.autonomous_loop import ACTIONS
from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.sensorimotor.multimodal_encoder import encode_multimodal

CYCLE = 256
CYCLES = 10
SEEDS = (0, 1, 2)
ID_CELLS = list(range(0, 16)) + list(range(32, 48))  # the vision and audio bands


def _vdc(i):
    i = int(i)
    x, d = 0.0, 0.5
    while i:
        x += d * (i & 1)
        i >>= 1
        d *= 0.5
    return x


def _world(seed, shuffled, k=1, cycles=CYCLES):
    """The planted world -- vision id i%8, audio (3i)%8, a deterministic 3-bit successor map --
    as the 64-d context the arm's own encoder builds; ordered, or re-permuted under one fresh
    permutation per cycle (pairing kept). The proprioception walk and the action draws are the
    same for both arms of a seed. k > 1 shows every pair k consecutive steps (the repeated
    world); `cycles` sets the walk's length so the action stream matches the archived harness
    (the record panel's I-1: a sliced 10-cycle world draws its actions after a longer walk)."""
    vis = [i % 8 for i in range(CYCLE)]
    aud = [(3 * i) % 8 for i in range(CYCLE)]
    n = cycles * CYCLE * k
    rng = random.Random(seed * 7919 + 1)
    prop, p = [], 0.5
    for _ in range(n + 1):
        p = min(1.0, max(0.0, p + rng.uniform(-0.05, 0.05)))
        prop.append(p)
    acts = [rng.choice(ACTIONS) for _ in range(n + 1)]
    seq = []
    for c in range(cycles + 1):
        if shuffled:
            perm = random.Random(seed * 100003 + c).sample(range(CYCLE), CYCLE)
        else:
            perm = list(range(CYCLE))
        for g in perm:
            seq.extend([(vis[g], aud[g])] * k)  # k-fold repeats: near-duplicate consecutives
    pats = [
        encode_multimodal([_vdc(v)] * 4, [prop[t]], [_vdc(a)] * 4, 0.12)
        for t, (v, a) in enumerate(seq[: n + 1])
    ]
    return pats, acts, seq


def _run(model, pats, acts):
    """The arm's order: read prediction_error, then train_step. Returns the last-block mean."""
    n = len(pats) - 1
    err = []
    for t in range(n):
        err.append(float(model.prediction_error(pats[t], acts[t], pats[t + 1])))
        model.train_step(pats[t], acts[t], pats[t + 1])
    last = err[-CYCLE:]
    return sum(last) / len(last)


def _run_block(model, pats, acts, block):
    """As _run, with the last-block length given (the repeated world's block is CYCLE * k)."""
    n = len(pats) - 1
    err = []
    for t in range(n):
        err.append(float(model.prediction_error(pats[t], acts[t], pats[t + 1])))
        model.train_step(pats[t], acts[t], pats[t + 1])
    last = err[-block:]
    return sum(last) / len(last)


def _spread(model, pats):
    zs = [model.target_encode(p) for p in pats]
    tot = cnt = 0
    for i in range(len(zs)):
        for k in range(i + 1, len(zs)):
            tot += sum((a - b) ** 2 for a, b in zip(zs[i], zs[k], strict=True))
            cnt += 1
    return tot / cnt


def _distinct(pats, cap=64):
    seen = {}
    for p in pats:
        seen.setdefault(tuple(round(v, 4) for v in p), p)
    return list(seen.values())[:cap]


def _oracle_last_block(pats, acts, seq):
    """A tabular (vis, aud, action) -> next-pattern lookup; its last-block error on the id bands."""
    table, errs = {}, []
    for t in range(len(acts) - 1):
        key = (seq[t][0], seq[t][1], acts[t])
        if key in table:
            yp, y = table[key], pats[t + 1]
            errs.append(sum((yp[i] - y[i]) ** 2 for i in ID_CELLS))
        table[key] = pats[t + 1]
    last = errs[-CYCLE:]
    return sum(last) / len(last)


def _linear_held_out(x):
    """Regularised least squares x_{t+1} ~ A x_t + b on cycles 1-5, scored on the id bands of
    cycles 6-10. rcond regularises the rank-deficient design (the RBF bands span few directions)."""
    n = len(x) - 1
    fit, score = slice(0, 5 * CYCLE), slice(5 * CYCLE, n)
    xa = np.hstack([x, np.ones((len(x), 1))])
    a, *_ = np.linalg.lstsq(xa[fit], x[1 : 5 * CYCLE + 1], rcond=1e-6)
    pred = xa[score] @ a
    truth = x[5 * CYCLE + 1 : n + 1]
    return float(((pred[:, ID_CELLS] - truth[:, ID_CELLS]) ** 2).sum(axis=1).mean())


@pytest.mark.slow
def test_the_planted_world_is_learnable_at_the_arms_input():
    """Positive control: the world carries exactly the succession the red claims -- and the
    per-cycle permutation destroys it."""
    for seed in SEEDS:
        ord_p, acts, ord_seq = _world(seed, shuffled=False)
        shf_p, _, shf_seq = _world(seed, shuffled=True)
        o_ord = _oracle_last_block(ord_p, acts, ord_seq)
        o_shf = _oracle_last_block(shf_p, acts, shf_seq)
        assert o_ord < 1e-9, (
            f"seed {seed}: the oracle must read the ordered world exactly ({o_ord})"
        )
        assert o_shf > 1.0, f"seed {seed}: the oracle must NOT read the shuffled world ({o_shf})"
        e_o = _linear_held_out(np.array(ord_p))
        e_s = _linear_held_out(np.array(shf_p))
        assert e_o < 0.1 and e_o < 0.25 * e_s, (
            f"seed {seed}: a linear predictor must read the ordered id bands (held-out "
            f"{e_o:.4f}, absolute bar 0.1) and separate the worlds (shuffled {e_s:.4f})"
        )


@pytest.mark.slow
def test_the_arms_world_model_reads_the_planted_succession():
    for seed in SEEDS:
        ord_p, acts, _ = _world(seed, shuffled=False)
        shf_p, _, _ = _world(seed, shuffled=True)
        e_o = _run(JepaWorldModel(ACTIONS, in_dim=64, seed=seed), ord_p, acts)
        e_s = _run(JepaWorldModel(ACTIONS, in_dim=64, seed=seed), shf_p, acts)
        assert e_o < 0.5 * e_s, f"seed {seed}: ordered {e_o:.5f} vs shuffled {e_s:.5f}"


@pytest.mark.slow
def test_the_ablation_does_not_read_it_and_collapses():
    """The control that is not the method: anti_collapse=False at the shipped configuration."""
    seed = 0
    ord_p, acts, _ = _world(seed, shuffled=False)
    shf_p, _, _ = _world(seed, shuffled=True)
    abl = JepaWorldModel(ACTIONS, in_dim=64, seed=seed, anti_collapse=False)
    e_o = _run(abl, ord_p, acts)
    e_s = _run(JepaWorldModel(ACTIONS, in_dim=64, seed=seed, anti_collapse=False), shf_p, acts)
    assert e_o >= 0.5 * e_s, f"the ablation must NOT clear the order bar ({e_o:.5f} vs {e_s:.5f})"
    pats = _distinct(ord_p)
    ratio = _spread(abl, pats) / _spread(JepaWorldModel(ACTIONS, in_dim=64, seed=seed), pats)
    assert ratio < 0.05, f"the ablation must collapse (trained/fresh spread ratio {ratio:.4f})"


@pytest.mark.slow
def test_the_latent_at_the_arms_defaults_does_not_collapse():
    for seed in (0, 2):
        ord_p, acts, _ = _world(seed, shuffled=False)
        model = JepaWorldModel(ACTIONS, in_dim=64, seed=seed)
        _run(model, ord_p, acts)
        pats = _distinct(ord_p)
        fresh = JepaWorldModel(ACTIONS, in_dim=64, seed=seed)
        ratio = _spread(model, pats) / _spread(fresh, pats)
        assert ratio > 0.25, f"seed {seed}: trained/fresh spread ratio {ratio:.4f}"
        std = model.latent_std()
        assert std > 0.05, f"seed {seed}: latent_std {std:.4f} below the module's own floor"


REPEAT_K = 32
REPEAT_CYCLES = 5


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the live-floor arc's STANDING RED (evidence/jepa-live-floor.md sections 3.4b and 5): in "
        "the archived harness a VICReg reservoir batch turned this green (k=32 order ratio "
        "0.42-0.49, std 0.123) but lost the planted k=1 order read (seed 0 ratio 1.6 at 10 "
        "cycles) and did not land; strict so a fix must update the record"
    ),
)
def test_the_batch_carries_diversity_on_a_repeated_world():
    """The live-floor arc's brain-free red (evidence/jepa-live-floor.md section 2.2 Amendment 1):
    every pair shown 32 consecutive near-duplicate steps. The defaults must still read the
    succession (ordered under half the shuffled twin) and latent_std() must clear the module's own
    floor."""
    for seed in SEEDS:
        ord_p, acts, _ = _world(seed, shuffled=False, k=REPEAT_K, cycles=REPEAT_CYCLES)
        shf_p, _, _ = _world(seed, shuffled=True, k=REPEAT_K, cycles=REPEAT_CYCLES)
        model = JepaWorldModel(ACTIONS, in_dim=64, seed=seed)
        e_o = _run_block(model, ord_p, acts, CYCLE * REPEAT_K)
        twin = JepaWorldModel(ACTIONS, in_dim=64, seed=seed)
        e_s = _run_block(twin, shf_p, acts, CYCLE * REPEAT_K)
        assert e_o < 0.5 * e_s, (
            f"seed {seed}: on the repeated world the predictor must read order (ordered "
            f"{e_o:.5f} vs shuffled {e_s:.5f}); the archived harness read none (4.1-4.6)"
        )
        std = model.latent_std()
        assert std > 0.05, (
            f"seed {seed}: latent_std {std:.4f} below the module's floor on the repeated world "
            "(the archived harness read 0.001 here on a maximally spread encoder)"
        )
