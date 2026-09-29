"""A closed brain releases its reasoning bridge: no atexit hook keeps it alive.

Found by the amdahl gate's census control in the 2026-09-24 battery: a brain whose chat fell
through to LiveBrain._reasoning_reply registered an atexit closure over its LakeChat (lc.brain is
the LiveBrain), so every such brain stayed reachable after close() until the process exited.
"""

from snn_data_lake.engine.brain import make_brain
from snn_data_lake.gates.amdahl_gate import _count_lif_process
from snn_data_lake.substrate.home import throwaway_home


def test_closed_brain_after_a_fallthrough_turn_is_released():
    before = _count_lif_process()
    with throwaway_home():
        brain = make_brain(seed=100, persist=False)
        brain.respond("tell me a joke about cats", human=True)  # reaches the reasoning bridge
        assert brain.brain._lakechat is not None
        assert _count_lif_process() > before
        brain.close()
        assert brain.brain._lakechat is None
        assert brain.brain._lakechat_close is None
    del brain
    assert _count_lif_process() == before
