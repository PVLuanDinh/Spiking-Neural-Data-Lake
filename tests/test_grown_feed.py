"""specs/022 T111 (G3) P5: the feed -- admitted grown units project into the G1 reservoir.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.9 C15 (the norm-matched
random-feed arm must differ from the fed arm on the pool's carriers; the instrument's positive
control is that a "random" arm equal to the fed matrix does NOT differ) and C17's reservoir-segment
half (Dale's sign on every grown -> reservoir synapse: a planted deposit that would flip a sign is
clipped, a planted stored synapse that violates its unit's sign is refused at restore, the refusal
names the synapse). The port is OFF by default: percept_fed with a zero feed is byte-identical to
percept. Constructed fixtures on seeds 9900-9999 only; nothing here builds a Brain or a lake.
"""

import numpy as np
import pytest

from snn_data_lake.growth.grown_code import GrownUnit
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features
from snn_data_lake.reservoir import recurrent_pool as rp

TAU = 8.0
T_POST = 14
LSM_PRE = [[1, 2], [0, 0], [1, 0]]
LSM_W = [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]]
LSM_CFG = {"lambda": 0.5, "g": 2, "T_in": 3}
N_POOL = 3


def _pool(window=2):
    return rp.KernelLsmPool(
        LSM_PRE, LSM_W, LSM_CFG, "carried", window=window, w_in=np.full((N_POOL, 64), 0.01)
    )


def _ranks(text="cat dog sun sky"):
    learner = SpikingInteractionLearner(["A", "B"], n_in=52, seed=9900, t_post=6)
    return tuple(learner.encode(text_features(text, 52)))


def _unit(uid, slot, sign=1):
    w = [0.0] * 52
    w[slot] = 2.0
    return GrownUnit(
        unit_id=uid, w_in=tuple(w), threshold=1.0, window=12, sign=sign, out=(0.0, 0.0)
    )


UNITS = (_unit(3, 0, +1), _unit(7, 1, -1), _unit(5, 2, +1))  # unit_id order: 3, 5, 7
# columns in unit_id order (3, 5, 7); E columns >= 0, the I column (id 7) <= 0
W_FEED = np.array([[2.0, 1.0, -1.5], [0.0, 3.0, -2.0], [1.0, 0.5, -3.0]])
X = np.linspace(0.0, 1.0, 64)


def test_port_off_is_todays_bytes():
    base, fed = _pool(), _pool()
    a = base.percept(X)
    b = fed.percept_fed(X, np.zeros(N_POOL))
    assert a.tolist() == b.tolist() and base.fingerprint() == fed.fingerprint()


def test_port_refuses_a_bad_feed():
    for bad in (np.zeros(N_POOL + 1), np.zeros((N_POOL, 1)), np.array([0.0, np.nan, 0.0])):
        with pytest.raises(ValueError, match="REFUSED"):
            _pool().percept_fed(X, bad)
