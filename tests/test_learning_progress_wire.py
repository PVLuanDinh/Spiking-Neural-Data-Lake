"""specs/022 T046a: the |LP| feed fires at the REAL wire, and it CHARGES.

The wire sits beside the surprise consumer in `Brain._perceive_real` (the same stub-encoder
production-path pattern tests/test_spontaneous_bootstrap.py pins): every real percept's
surprise scalar feeds the cortex competence channel of the |LP| facade -- the T028 dated
disposition's discharged consumer. These tests pin (1) the feed runs per percept on the
production path, and (2) the signal is the DRIVE's, not a stub: a varying surprise stream
moves |LP| while the facade's absence-is-a-fact contract holds for unfed regions."""

import numpy as np

from snn_data_lake.cortex.brain import make_brain


class _StubEnc:
    def __init__(self):
        self._i = 0

    def encode(self, frame):
        # varying latents -> varying surprise (a constant latent would starve the signal)
        self._i += 1
        rng = np.random.default_rng(self._i)
        return [float(v) for v in rng.random(64) * 0.8 + 0.1]

    def classify(self, frame):
        return 0


def _perceiving_brain(n_frames=6):
    b = make_brain(seed=0)
    b._perceiving_real = True
    b._perception_enc = _StubEnc()
    b._real_frames = [object()] * n_frames
    b._real_labels = [0] * n_frames
    return b


def test_lp_feed_fires_per_percept_and_charges():
    b = _perceiving_brain()
    for _ in range(6):
        assert b._perceive_real() is not None
    assert b._lp is not None, "the |LP| facade never constructed at the wire"
    assert b._lp.observations("cortex") == 6, (
        f"fed {b._lp.observations('cortex')} of 6 percepts -- the feed is not per-percept"
    )
    lp = b._lp.lp("cortex")
    # STRICTLY positive (panel A-F2): a dead channel delivering nothing would satisfy
    # any 0.0-inclusive bound; the varying stream must MOVE |LP|.
    assert 0.0 < lp < 1.0 and np.isfinite(lp)
    # absence is a fact: no other region has a fed channel from this path
    assert set(b._lp.lp_all()) == {"cortex"}


def test_constant_stream_reads_exactly_zero(monkeypatch):
    """The dissociation that makes the positive assert honest (panel A-F2): the SAME wire
    carrying a FLAT surprise stream reads |LP| exactly 0.0 (the module's measured flat
    regime), while the observation count proves the feed ran. Together with the varying
    arm's lp > 0.0 this discriminates a live moving signal from a dead channel -- either
    alone cannot."""
    from snn_data_lake.cortex.percept_surprise import PerceptSurprise

    monkeypatch.setattr(PerceptSurprise, "observe", lambda self, x: 0.4)
    b = _perceiving_brain()
    for _ in range(6):
        b._perceive_real()
    assert b._lp is not None and b._lp.observations("cortex") == 6
    assert b._lp.lp("cortex") == 0.0


def test_neutered_feed_leaves_no_channel(monkeypatch):
    """Charge, not reachability: without the feed the facade holds NO cortex channel."""
    from snn_data_lake.basal_ganglia import learning_progress as lpmod

    monkeypatch.setattr(lpmod.LearningProgress, "consume_surprise", lambda self, s: 0.0)
    b = _perceiving_brain()
    for _ in range(3):
        b._perceive_real()
    assert b._lp is not None and b._lp.observations("cortex") == 0, (
        "the neutered feed still registered observations -- the charge claim is elsewhere"
    )
