"""G5 / T113 Part D D1's read-out adaptors (evidence/g5-readout-variants.md D.2, D.4 PD-R3).

The adaptors are ADOPTED, never re-implemented (P2): DELTA is `OnlineReadout("delta")`'s own rule
driven through the module's own `learn`; PC is `readout/predictive_coding.py`; SOM is
`reasoning/spiking_som.py`. The reds that must be able to fire before any D1 cell is read:

  * the read-out-only contract -- no pool import or pool construction in the adaptors module (D.2);
  * DELTA is the incumbent bit for bit on the same events, with every column allocated and no widen;
  * DELTA's error surface: a wrong-width stimulus and an unknown label are REFUSED, not broadcast;
  * PC is deterministic per seed, and its neutered twin shares the arm's seeded W1/W2 while the
    learned coupling differs (a control that learned nothing is inert, P8);
  * SOM is deterministic, and SOM0 (the neighbourhood width at zero) moves ONLY the BMU;
  * the printed code hash is this module's own (D.2).

NOTHING here reads a registered seed block. Every number is a fixture diagnostic, disclosed as such.
"""

import pytest

from snn_data_lake.instruments import g5_part_d_adaptors as ad

FIX = 20261003  # the fixture RNG seed (SEEDS.md), never a recording seed


def test_the_adaptors_are_readout_only_and_carry_their_code_hash():
    """D.2's claim has a reachable red: `self_check` parses the adaptors module's own AST and
    refuses any pool import or pool construction, so the claim cannot silently stop being true."""
    ad.self_check()
    assert len(ad.code_hash()) == 64
    for adaptor in (
        ad.delta(("c0",), 8),
        ad.pc(("c0",), 8, seed=FIX),
        ad.som(("c0",), 8, seed=FIX),
    ):
        assert adaptor.state()["code"] == ad.code_hash()


def test_delta_is_the_incumbent_bit_for_bit_on_the_same_events():
    """PD-R3's DELTA arm is the incumbent's rule ADOPTED: beta after the same events is equal to
    `OnlineReadout("delta")`'s, so the arm differs from the incumbent by nothing but its readout
    job. The common vocabulary init (record 2.3, carried into Part D) allocates every column up
    front, so `learn`'s own widen never fires."""
    import torch

    from snn_data_lake.readout.online_readout import OnlineReadout

    labels = ("c0", "c1")  # _chain's own labels; an unknown label is refused below
    teach = ad._chain(FIX)[:180]
    ours = ad.delta(labels, len(teach[0][0]))
    theirs = OnlineReadout("delta")

    class _M:
        pass

    m = _M()
    m.beta = torch.zeros((len(teach[0][0]), len(labels)), dtype=torch.float32)
    m.m = len(labels)
    for x, c in teach:
        ours.learn(x, c)
        theirs.learn(m, x, labels.index(c))
    assert torch.equal(ours.beta, m.beta)
    assert ours.state()["widened"] == 0 and theirs.widened == 0
    assert ours.rule.mu == theirs.mu and ours.rule.sign == theirs.sign


def test_a_wrong_width_stimulus_and_an_unknown_label_are_refused():
    d = ad.delta(("c0", "c1"), 8)
    with pytest.raises(ad.AdaptorError):
        d.respond([0.0] * 9)
    with pytest.raises(ad.AdaptorError):
        d.learn([0.0] * 8, "c9")


def test_pcs_neutered_twin_shares_the_arm_seed_and_differs_only_in_the_learned_coupling():
    """D.4 PD-R3's PC control is the neutered-coupling twin: the same net at the same seed with
    W1/W2 NOT learned. It must start from the arm's own seeded weights (one seed difference only),
    and the trained arm must differ from it exactly in that coupling -- an inert control is not a
    control."""
    teach = ad._chain(FIX)[:120]
    arm = ad.pc(("c0", "c1"), len(teach[0][0]), seed=FIX)
    neutered = ad.pc(("c0", "c1"), len(teach[0][0]), seed=FIX, neutered=True)
    fresh = ad.pc(("c0", "c1"), len(teach[0][0]), seed=FIX)
    for x, c in teach:
        arm.learn(x, c)
    arm.respond(teach[0][0])  # settle the read-out: trains W1/W2 on the buffered teach split
    assert neutered.state()["w1_sha"] == fresh.state()["w1_sha"]
    assert arm.state()["w1_sha"] != neutered.state()["w1_sha"]


def test_som0_moves_only_the_bmu_and_the_default_map_is_its_own_neighbourhood():
    """D.2/D.4: SOM's control is the neighbourhood width AT ZERO. No non-BMU unit's weight may move
    under it, or the control is not the zero-width map the registration names."""
    labels, n_in = ("c0", "c1"), 8
    s0 = ad.som(labels, n_in, seed=FIX, control=True)
    x = [1.0] * n_in
    s0.learn(x, "c0")
    before = [list(w) for w in s0.som.w]
    bmu = s0.som.bmu(x)
    s0.som.train([x], epochs=1, lr0=ad.SOM_LR0, sigma0=ad.SOM0_WIDTH)
    moved = [i for i in range(len(before)) if before[i] != list(s0.som.w[i])]
    assert moved == [bmu], f"the width is not at zero: units {moved} moved, not only the BMU {bmu}"
    assert s0.state()["width"] == ad.SOM0_WIDTH
    assert ad.som(labels, n_in).state()["width"] is None
