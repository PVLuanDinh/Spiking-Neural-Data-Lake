"""G5 / T113 Part B's module reds (specs/022 evidence/g5-readout-variants.md section 3.5):
frozen counts are the bare argmax in both modes (PB3); a count before scoring raises (PB4); the
full coupling recovers a systematic relabelling while the marginal control stays diagonal (the
property D2's control rests on)."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.readout.categorical_coupling import (  # noqa: E402
    MODES,
    CategoricalCoupling,
    main,
)


def test_self_check():
    main()


@pytest.mark.parametrize("mode", MODES)
def test_empty_counts_are_the_identity(mode):
    cc = CategoricalCoupling(5, mode=mode)
    assert torch.equal(cc.coupling(), torch.eye(5, dtype=torch.float64))


@pytest.mark.parametrize("mode", MODES)
def test_frozen_counts_never_move_the_decision(mode):
    g = torch.Generator().manual_seed(20261003)
    cc = CategoricalCoupling(4, mode=mode, frozen=True)
    for t in range(100):
        q = torch.softmax(torch.randn(4, generator=g), dim=0)
        assert cc.decide(q) == int(q.argmax())
        cc.scored()
        cc.update(int(q.argmax()), t % 4)
    assert cc.moved == 0 and int(cc.C.sum()) == 0


def test_a_count_before_scoring_raises():
    cc = CategoricalCoupling(3)
    with pytest.raises(RuntimeError):
        cc.update(0, 0)  # never decided
    cc.decide(torch.tensor([1.0, 0.0, 0.0]))
    with pytest.raises(RuntimeError):
        cc.update(0, 0)  # decided but NOT scored: Codex cp4 #4's leak
    cc.scored()
    cc.update(0, 0)
    with pytest.raises(RuntimeError):  # one decision closes one event, never two
        cc.update(0, 0)
    with pytest.raises(RuntimeError):
        cc.scored()  # nothing decided to score


def test_full_recovers_a_relabelling_marg_cannot_cross_classes():
    full, marg = CategoricalCoupling(3, "full"), CategoricalCoupling(3, "marg")
    for _ in range(30):
        for cc in (full, marg):
            cc.decide(torch.tensor([0.0, 1.0, 0.0]))
            cc.scored()
            cc.update(1, 0)
    q = torch.tensor([0.2, 0.7, 0.1])
    assert full.decide(q) == 0
    K = marg.coupling()
    assert torch.equal(K, torch.diag(torch.diagonal(K)))


def test_marg_is_a_label_shift_correction():
    # truth is always 0, the readout leans to 2: marg up-weights 0 and down-weights 2
    marg = CategoricalCoupling(3, "marg")
    for _ in range(40):
        marg.decide(torch.tensor([0.3, 0.2, 0.5]))
        marg.scored()
        marg.update(2, 0)
    assert marg.decide(torch.tensor([0.3, 0.2, 0.5])) == 0


def test_refuses_bad_construction_and_width():
    with pytest.raises(ValueError):
        CategoricalCoupling(3, mode="perm")
    with pytest.raises(ValueError):
        CategoricalCoupling(0)
    with pytest.raises(ValueError):
        CategoricalCoupling(3).decide(torch.ones(4))
