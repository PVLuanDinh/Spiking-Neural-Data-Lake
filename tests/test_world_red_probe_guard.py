"""specs/022 ruling 16: world_red_probe's scratch-home check is the shared live-home guard.

The private check it replaced compared home() with the live home for EQUALITY only, so a home
INSIDE the live home (or aliasing it) ran the probe against the live lake. Reds use a decoy.
"""

import pytest

from snn_data_lake.instruments import world_red_probe as W
from snn_data_lake.substrate import home as H


@pytest.fixture
def decoy(tmp_path, monkeypatch):
    live = tmp_path / "live"
    (live / "inside").mkdir(parents=True)
    monkeypatch.setattr(H, "_default_live_home", lambda: str(live))
    return live


def test_a_home_inside_the_live_home_is_refused_through_the_probe(decoy, monkeypatch):
    monkeypatch.setenv(
        "NORD_HOME", str(decoy / "inside")
    )  # contained, not equal: the old check passed it
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        W._assert_scratch_home()


def test_the_live_home_and_an_unset_home_are_refused_through_the_probe(decoy, monkeypatch):
    monkeypatch.setenv("NORD_HOME", str(decoy))
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        W._assert_scratch_home()
    monkeypatch.delenv("NORD_HOME")
    with pytest.raises(H.LiveHomeRefused):
        W._assert_scratch_home()


def test_a_throwaway_home_passes_the_probe_check(decoy):
    with H.throwaway_home(live_home=str(decoy)):
        W._assert_scratch_home()
