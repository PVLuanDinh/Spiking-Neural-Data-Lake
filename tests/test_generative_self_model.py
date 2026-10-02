"""
Edge tests for generative_self_model: the LEARNED spiking predictive-coding / e-prop forward model
that predicts the brain's own next self-signal (replaces the hand-coded proxy aggregate in
self_model.py). Covers the three API behaviors the __main__ self-check exercises at length: learning
lowers self_surprise on a predictable trajectory, novelty raises it, and reset() clears transient
state without touching the learned weights.

Run: pytest tests/test_generative_self_model.py
"""

from snn_data_lake.faculties.generative_self_model import GenerativeSelfModel

DIM = 4


def _rotation(n_cycles, dim=DIM):
    base = [[0.85 if i == k else 0.05 for i in range(dim)] for k in range(dim)]
    seq = []
    for _ in range(n_cycles):
        for k in range(dim):
            seq.append((base[k % dim], base[(k + 1) % dim]))
    return seq, base


def test_self_surprise_falls_on_repeating_trajectory():
    model = GenerativeSelfModel(DIM, seed=0)
    traj, _ = _rotation(30)
    surprises = []
    for cur, nxt in traj:
        model.predict(cur)
        surprises.append(model.observe(nxt))
    early = sum(surprises[:12]) / 12
    late = sum(surprises[-12:]) / 12
    assert late < early, f"self_surprise did not fall as expected ({early:.4f} -> {late:.4f})"


def test_self_surprise_rises_on_novel_transition():
    model = GenerativeSelfModel(DIM, seed=0)
    traj, base = _rotation(30)
    for cur, nxt in traj:
        model.predict(cur)
        model.observe(nxt)
    model.predict(base[0])
    s_learned = model.observe(base[1])  # the actually-learned successor
    model.predict(base[0])
    s_novel = model.observe([0.05, 0.05, 0.85, 0.85])  # a transition that breaks the rotation
    assert s_novel > s_learned, f"novel must surprise more ({s_novel:.4f} !> {s_learned:.4f})"


def test_reset_clears_trace_but_keeps_learned_weights():
    model = GenerativeSelfModel(DIM, seed=0)
    traj, _ = _rotation(10)
    for cur, nxt in traj:
        model.predict(cur)
        model.observe(nxt)
    w_before = [row[:] for row in model.w]
    model.reset()
    assert model.w == w_before, "reset() must not change the learned weights"
    probe = [0.4, 0.1, 0.1, 0.1]
    hat = model.predict(probe)
    expected = [sum(model.w[j][k] * probe[k] for k in range(DIM)) for j in range(DIM)]
    assert all(abs(a - b) < 1e-9 for a, b in zip(hat, expected, strict=True)), (
        "predict() right after reset() must equal W @ probe (eligibility trace cleared to zero)"
    )
