"""Same seed must grow the same brain on any hardware. It did not.

`growth/ng_snn.py` built its generator as `torch.Generator(device=self.device)`, and `NGSNN`
defaults to `device="cuda"` wherever a card exists. torch's CPU and CUDA generators produce
DIFFERENT streams from the same seed -- measured directly, seed 7:

    cpu  generator -> [0.534923, 0.198803, 0.659212, 0.656890, 0.232762]
    cuda generator -> [0.954597, 0.495040, 0.941953, 0.592580, 0.699600]

So the grown hidden weights `W`/`b` came off a different stream on a machine with a card than on
one without. This box has an RTX 5070 and CI has none, which means the two grew DIFFERENT BRAINS
from the same seed by construction. That is a breach of the determinism invariant -- which is a
property of the SEED, not of the machine -- and it was MASKED rather than absent: nothing in the
suite compared a grown weight across devices, so every gate stayed green.

THE FIX is to draw on the CPU stream always and move the result to `self.device`. Note what that
does and does not change: on a CPU-only machine the stream is byte-identical to before (a CPU
generator is what it already built), so CI sees no movement at all; on a CUDA machine the numbers
DO change, to the CPU stream's -- which is the entire point, because that is the stream CI and every
cardless clone will reproduce.

WHAT IS MEASURED HERE. The observable is the GROWN WEIGHTS, not accuracy. Accuracy is a lossy
readout: two different-but-equally-good brains score the same, so an accuracy assert would pass
straight through this defect. The weights are what the seed is supposed to determine.

Run: pytest tests/test_growth_rng_is_hardware_independent.py
"""

import torch

from snn_data_lake.growth.ng_snn import NGSNN

_SEED = 7
_N_DRAW = 6


def _draws(device, seed=_SEED, n=_N_DRAW):
    """The first n numbers a fresh NGSNN's own generator yields, as CPU floats for comparison."""
    clf = NGSNN(seed=seed, device=device)
    return torch.rand(n, generator=clf.g).tolist()


def test_the_generator_is_cpu_bound_whatever_the_device():
    """The stream must not be a function of where the tensors live."""
    assert NGSNN(seed=_SEED, device="cpu").g.device.type == "cpu"
    # `device="cuda"` degrades to cpu when no card is present, so this is meaningful on both boxes
    assert NGSNN(seed=_SEED, device="cuda").g.device.type == "cpu"


def test_same_seed_same_stream_across_requested_devices():
    """THE RED this file exists for: before the fix these two lists differed on any machine with a
    card, because one generator was CUDA-bound and the other was not."""
    assert _draws("cpu") == _draws("cuda")


def test_a_different_seed_still_moves_the_stream():
    """POSITIVE CONTROL.

    Without it, a generator stuck at a constant would satisfy the test above."""
    assert _draws("cpu", seed=_SEED) != _draws("cpu", seed=_SEED + 1)


def test_cpu_and_cuda_torch_generators_really_do_diverge():
    """The premise, asserted rather than assumed -- if torch ever unified the two streams this file
    would still pass while pinning nothing, and that would be worth knowing. SKIPPED without a card,
    because the divergence cannot be observed on a machine that has only one of the two."""
    if not torch.cuda.is_available():
        return  # no card: the claim is untestable here, and a false green is worse than no test
    a = torch.rand(_N_DRAW, generator=torch.Generator(device="cpu").manual_seed(_SEED))
    b = torch.rand(
        _N_DRAW, generator=torch.Generator(device="cuda").manual_seed(_SEED), device="cuda"
    ).cpu()
    assert not torch.equal(a, b), (
        "torch's cpu and cuda generators now agree at the same seed -- the hazard this module "
        "guards against has changed shape, so re-derive the guard rather than trusting it"
    )
