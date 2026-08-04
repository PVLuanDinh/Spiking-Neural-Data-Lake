"""P4b -- the durable lake is part of the brain: episodes go IN it and are recalled FROM it.

This suite exists because the failure it guards against was invisible from inside the writer. The
episodic store reported `retained() == 1.000` for months while persisting NOTHING: measured
2026-08-04, 59 episodes over 600 ticks with persist=True added 0 records to the lake, and 149
episodes over 1,500 ticks were recalled 0 times. A store perfectly recalling episodes it was about
to lose entirely looks identical, from the object's own API, to one that works.

So every assertion here reads the lake back through a path the writer does not own.

Run: pytest tests/test_episode_durability.py
"""

import numpy as np
import pytest

from snn_data_lake.substrate.binlog import pack_episode, unpack_episode

EPISODE = {"who": "arm", "what": "JOINT_SHOULDER_LIFT", "where": "0@2", "when": "290"}


def test_episode_round_trips_through_raw_binary():
    """The encoding is raw binary, not JSON, like every other record in the lake."""
    got = unpack_episode(pack_episode(EPISODE))
    assert got == EPISODE, got


def test_packing_is_slot_order_independent():
    """BinLog dedups on the payload bytes, so the same episode must pack identically however its
    dict was built -- otherwise one memory is stored repeatedly under different byte strings."""
    a = pack_episode({"who": "x", "what": "y", "when": "1"})
    b = pack_episode({"when": "1", "what": "y", "who": "x"})
    assert a == b, "slot order changed the bytes; dedup would store this memory twice"


def test_episodes_reach_the_lake_and_survive_the_writer(tmp_path, monkeypatch):
    """P4b: what the brain learns goes IN the lake. Asserted from a RE-READ, not from the brain."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    import importlib

    from snn_data_lake.substrate import home as H

    importlib.reload(H)
    from snn_data_lake.cortex.brain import make_brain

    assert H.load_episodes() == [], "a fresh home must start with no episodes"
    b = make_brain(seed=0, persist=True)
    for _ in range(400):
        b.tick()
    formed = len(b._episode_consolidator.keys)
    b.close()

    on_disk = H.load_episodes()
    assert formed > 0, "the brain formed no episodes; this test would pass vacuously"
    assert len(on_disk) >= formed, (
        f"the brain formed {formed} episodes but the lake holds {len(on_disk)}. This is the exact "
        "defect P4b exists for: a second store that nothing persists."
    )
    assert all(isinstance(e, dict) and e for e in on_disk), "episodes must read back as dicts"


def test_a_restarted_brain_is_not_amnesic(tmp_path, monkeypatch):
    """P4b: a restart is not amnesia. A new brain over the same lake starts with what it holds."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    import importlib

    from snn_data_lake.substrate import home as H

    importlib.reload(H)
    from snn_data_lake.cortex.brain import make_brain

    first = make_brain(seed=0, persist=True)
    for _ in range(400):
        first.tick()
    first.close()
    stored = len(H.load_episodes())
    assert stored > 0

    second = make_brain(seed=0, persist=True)
    restored = getattr(second, "_episodes_restored", 0)
    resident = len(second._episode_consolidator.keys)
    second.close()
    assert restored > 0, "a persistent brain booted empty beside a full lake -- P4b failure"
    assert resident > 0, f"restored {restored} but {resident} are resident"


def test_recall_reaches_past_working_memory(tmp_path, monkeypatch):
    """P4b: a working-memory miss means 'not resident', not 'never happened'."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    import importlib

    from snn_data_lake.substrate import home as H

    importlib.reload(H)
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=0, persist=True)
    for _ in range(1600):  # comfortably past the attractor's capacity
        b.tick()
    cap = b._episode_consolidator.capacity
    b.close()

    eps = H.load_episodes()
    if len(eps) <= cap:
        pytest.skip(f"only {len(eps)} episodes vs capacity {cap}: nothing was evicted to test")

    b2 = make_brain(seed=0, persist=True)
    oldest = eps[0]
    working = b2._episode_consolidator.recall_episode(oldest)
    durable = b2.recall_episode_durable(oldest)
    miss = b2.recall_episode_durable({"who": "nobody", "what": "never", "when": "-1"})
    b2.close()

    assert not working.get("episode"), (
        "the oldest episode was still resident; this test cannot show the lake being reached"
    )
    assert durable.get("source") == "lake", f"expected a lake hit, got {durable.get('source')}"
    assert durable.get("episode") == oldest
    assert miss.get("source") == "miss" and not miss.get("episode"), (
        "a never-stored cue must return an honest miss, not the nearest anything"
    )


def test_foreign_records_are_preserved(tmp_path, monkeypatch):
    """P4b: one lake, several tenants. Adding episodes must not destroy facts."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    import importlib

    from snn_data_lake.substrate import home as H

    importlib.reload(H)

    H.KnowledgeLog().append({"cue": ["capital of france"], "answer": "paris"})
    H.save_episode(EPISODE)
    H.save_growth(np.zeros(4, dtype=np.float16).tobytes())

    assert H.KnowledgeLog().count() == 1, "the episode or growth record ate the fact"
    assert len(H.load_episodes()) == 1, "the fact or growth record ate the episode"
    assert H.load_growth() is not None, "the fact or episode ate the growth blob"
