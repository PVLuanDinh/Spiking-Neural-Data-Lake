"""specs/022 T057 -- the lake recall contract and its wire into the brain's durable recall.

The heavy separation red (48 degraded vs 40 scrambled cues) runs in lake_recall's
registered self-check; these arms pin the contract's fast paths and the CHARGE of the
brain.py wire: a partial cue for a NON-RESIDENT episode now answers from the lake, where
the superseded exact-content scan missed by construction (its own docstring said so).
"""

import os

from snn_data_lake.hippocampus.lake_recall import LakeRecall, encode_episode
from snn_data_lake.substrate.home import save_episode

EPS = [
    {"who": "arm", "what": "moved", "where": "bench", "when": "t0", "affect": "calm"},
    {"who": "daemon", "what": "slept", "where": "home", "when": "t1", "affect": "calm"},
    {"who": "owner", "what": "spoke", "where": "lake", "when": "t2", "affect": "joy"},
    {"who": "sensor", "what": "failed", "where": "field", "when": "t3", "affect": "fear"},
    {"who": "arm", "what": "grew", "where": "cortex", "when": "t4", "affect": "surprise"},
]


def _lake(tmp_path, episodes=EPS):
    os.makedirs(str(tmp_path), exist_ok=True)
    lake = os.path.join(str(tmp_path), "lake.bin")
    for e in episodes:
        assert save_episode(e, path=lake)
    return lake


def test_encoding_is_deterministic_and_slotwise(tmp_path):
    a = encode_episode(EPS[0])
    assert a == encode_episode(dict(reversed(list(EPS[0].items()))))  # order-free
    assert len(a) == 5  # one event per slot token
    # sharing a slot shares an event; changing one changes exactly that event
    b = encode_episode({**EPS[0], "affect": "fear"})
    assert len(set(a) & set(b)) == 4


def test_exact_and_degraded_recall_with_margins(tmp_path):
    lr = LakeRecall(path=_lake(tmp_path), seed=0)
    exact = lr.recall(EPS[2], k=1)[0]
    assert exact["episode"] == EPS[2] and exact["distance"] == 0
    assert exact["confidence"] == 1.0 and exact["calibrated"]
    deg = lr.recall({"when": "t3", "affect": "fear"}, k=1)[0]  # degraded: 2 of 5 slots
    assert deg["episode"] == EPS[3] and deg["confidence"] == 1.0 and deg["contradicted"] == 0
    # a 5-episode lake may not calibrate every arity; a margin, when present, is positive
    assert deg["margin"] is None or deg["margin"] > 0
    ks = lr.recall({"who": "arm"}, k=3)
    assert len(ks) == 3 and ks[0]["distance"] <= ks[1]["distance"] <= ks[2]["distance"]


def test_refusal_shapes(tmp_path):
    empty = LakeRecall(path=os.path.join(str(tmp_path), "none.bin"))
    assert empty.recall({"who": "x"}) == []
    tiny = LakeRecall(path=_lake(tmp_path / "t2", EPS[:2]), seed=0)
    hit = tiny.recall(EPS[0], k=1)[0]
    assert hit["episode"] == EPS[0] and hit["margin"] is None and not hit["calibrated"]
    assert hit["confidence"] == 1.0  # content verification needs no calibration
    assert hit["retrieval"] == 0.0  # retrieval strength DOES, and refuses without it
    lr = LakeRecall(path=_lake(tmp_path / "t3"), seed=0)
    assert lr.recall({}) == [] and lr.recall("nope") == []


def test_staleness_and_fresh_reader_agree(tmp_path):
    lake = _lake(tmp_path)
    lr = LakeRecall(path=lake, seed=0)
    assert lr.refresh() == 5
    extra = {"who": "owner", "what": "returned", "where": "home", "when": "t9", "affect": "joy"}
    assert save_episode(extra, path=lake)
    assert lr.recall(extra, k=1)[0]["distance"] == 0  # same instance sees the append
    assert LakeRecall(path=lake, seed=0).recall(extra, k=1)[0]["distance"] == 0  # and a fresh one


def test_brain_wire_charge(tmp_path, monkeypatch):
    # THE CHARGE: a partial cue for a lake-restored episode the attractor cannot address
    # (its recall_episode hashes the WHOLE episode, so a partial dict misses in working
    # memory by its own disclosed limit) now answers from the lake window with VERIFIED
    # confidence -- the superseded exact-content scan missed any partial dict by
    # construction, so source=="lake" on a partial cue IS the outcome change.
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    from snn_data_lake.cortex.brain import make_brain

    for e in EPS + [
        {"who": "owner", "what": "tested", "where": "bench", "when": f"t{10 + i}", "affect": "calm"}
        for i in range(3)
    ]:
        save_episode(e)  # default path -> the temp NORD_HOME lake, BEFORE construction
    brain = make_brain(seed=0, persist=True)  # restore fills the recall window FROM the lake
    assert brain._episode_consolidator is not None, "the wire needs the episodic tier live"
    assert brain._episode_window, "the restore must fill the lake-derived recall window"
    got = brain.recall_episode_durable({"when": "t3", "affect": "fear"}, min_conf=1.0)
    assert got["source"] == "lake", f"a partial cue must now reach the lake window ({got})"
    assert got["episode"]["who"] == "sensor" and got["confidence"] == 1.0
    assert got.get("contradicted") == 0  # every cue slot VERIFIED, none contradicted
    miss = brain.recall_episode_durable({"when": "t777", "affect": "dread"}, min_conf=1.0)
    assert miss["source"] == "miss", f"a cue matching nothing must miss, never guess ({miss})"
