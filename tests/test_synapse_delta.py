"""SynapseDelta (_T_SYNDELTA, 0x09) -- spike-derived dirty set, quantize-compare-drop.

specs/022 T010a. The constructed red here reproduces the shape of the measured pathology (97.64%
of the live lake was repeated snapshots of an unchanged substrate): a naive writer that snapshots
every dirty synapse each tick pays every tick; SynapseDelta pays only when a quantized value the
lake has not seen appears. The other half of the contract is that cheap never means forgetful:
zero-base replay reconstructs the exact quantized substrate.
"""

import struct

import numpy as np
import pytest

from snn_data_lake.substrate.lake_engine import RETENTION, SynapseDelta
from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS

Q = 0.001


def test_q_step_is_required_and_positive():
    """P1: the weight quantum is the caller's to derive -- no default, no zero, no negative."""
    with pytest.raises(TypeError):
        SynapseDelta()  # no default exists to fall back to
    for bad in (0, -0.5, None):
        with pytest.raises((ValueError, TypeError)):
            SynapseDelta(bad)


def test_dirty_set_is_spike_derived_and_capture_is_deferred():
    sd = SynapseDelta(Q)
    w = np.zeros((5, 5))
    w[2, 2] = 0.7  # changes while BOTH endpoints are silent
    rec = sd.collect(w, pre_fired=[0], post_fired=[1])
    got = set() if rec is None else {(i, j) for i, j, _q in SynapseDelta.entries(rec)}
    assert (2, 2) not in got, "a synapse with silent endpoints must not be captured this tick"
    # ...and the change is NOT lost: the first tick an endpoint fires, it lands, exact.
    rec2 = sd.collect(w, pre_fired=[2], post_fired=[])
    assert rec2 is not None
    assert (2, 2, round(0.7 / Q)) in set(SynapseDelta.entries(rec2)), (
        "deferred capture must deliver the exact quantized value on the first firing tick"
    )


def test_compare_drop_and_sub_quantum_drift():
    sd = SynapseDelta(Q)
    w = np.zeros((3, 3))
    w[0, 0] = 0.5
    assert sd.collect(w, [0], []) is not None
    assert sd.collect(w, [0], []) is None, "an unchanged substrate must write NOTHING"
    w[0, 0] += Q / 4  # drift below the quantum: same quantized value -> still nothing
    assert sd.collect(w, [0], []) is None, (
        "sub-quantum drift must be dropped by the quantize-compare, not written as churn"
    )
    w[0, 0] += Q  # a full quantum: now it is a change the lake should see
    assert sd.collect(w, [0], []) is not None


def test_zero_base_reconstruction_round_trip():
    rng = np.random.default_rng(7)
    sd = SynapseDelta(Q)
    w = np.zeros((12, 12))
    records = []
    for _ in range(30):
        # sparse random plasticity on random spikes, the shape a live tick produces
        pre = rng.choice(12, size=3, replace=False)
        post = rng.choice(12, size=3, replace=False)
        for i in pre:
            w[i, rng.integers(0, 12)] += rng.normal(0, 0.05)
        rec = sd.collect(w, pre, post)
        if rec:
            records.append(rec)
    # one final all-fired tick synchronises anything whose endpoints stayed silent above
    rec = sd.collect(w, range(12), range(12))
    if rec:
        records.append(rec)
    twin, replayer = np.zeros_like(w), SynapseDelta(Q)
    n = sum(replayer.apply(r, twin) for r in records)
    assert n > 0, "the round trip proved nothing: no entries were ever written"
    assert np.array_equal(twin, np.rint(w / Q) * Q), (
        "zero-base replay must reconstruct the exact quantized substrate"
    )


def test_pathology_red_frozen_substrate_costs_nothing():
    """The 97.64% shape, constructed: substrate frozen after tick 1, endpoints firing every tick.
    A naive per-tick snapshot writer pays 16 B per dirty synapse per tick, forever. SynapseDelta
    pays once."""
    sd = SynapseDelta(Q)
    w = np.zeros((8, 8))
    w[:2, :] = 0.3  # rows 0-1 carry weight
    delta_bytes = 0
    naive_bytes = 0
    for _ in range(200):
        rec = sd.collect(w, pre_fired=[0, 1], post_fired=[])
        delta_bytes += len(rec) if rec else 0
        naive_bytes += 5 + 16 * 16  # header + 2 rows x 8 posts, what a snapshot writer would pay
    assert delta_bytes <= 5 + 16 * 16, "SynapseDelta paid more than the single genuine change"
    assert delta_bytes < 0.01 * naive_bytes, (
        f"the pathology red failed: {delta_bytes:,} B vs a naive writer's {naive_bytes:,} B"
    )


def test_malformed_payloads_refuse_whole():
    sd = SynapseDelta(Q)
    w = np.zeros((2, 2))
    w[0, 0] = 0.1
    rec = sd.collect(w, [0], [])
    with pytest.raises(ValueError):
        SynapseDelta.entries(rec[:-1])  # truncated
    with pytest.raises(ValueError):
        SynapseDelta.entries(rec + b"\x00")  # trailing bytes
    with pytest.raises(ValueError):
        SynapseDelta.entries(struct.pack("<BI", 99, 0))  # wrong version
    with pytest.raises(ValueError):
        SynapseDelta.entries(b"")  # too short for a header
    with pytest.raises(ValueError):
        sd.collect(w, [5], [])  # fired index out of range


def test_registered_in_both_census_views():
    """The standing clause: a new record class carries its disposition + retention rows in the
    same commit that introduces it. Asserted, not assumed."""
    assert SynapseDelta.TAG in RETENTION, "0x09 missing a retention label (P4b)"
    assert SynapseDelta.TAG in TAG_DISPOSITIONS, "0x09 missing a migration disposition"
    retention, reclaimable, _reason = RETENTION[SynapseDelta.TAG]
    assert (retention, reclaimable) == ("IMMORTAL", False), (
        "0x09 reclaim needs its own landed proof first; there is no FULL kind to compact against"
    )
