"""specs/022 T102 (R1) / lead ruling 16 -- the ONE shared live-home guard in substrate/home.py.

Every red runs against a DECOY live home passed as `live_home`; no test here touches the real
~/.snn_data_lake. A prefix check on `selfcheck_home_` is not isolation (Codex #14 on G5): a
prefixed junction or symlink can alias the live lake, so the guard resolves before it compares.
"""

import os
import subprocess
import sys

import pytest

from snn_data_lake.substrate import home as H


def _alias(link, target):
    """A directory alias the way an attacker-by-accident would make one: a junction on Windows
    (no admin needed), a symlink elsewhere."""
    if sys.platform == "win32":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        assert r.returncode == 0, r.stderr
    else:
        os.symlink(target, link, target_is_directory=True)


@pytest.fixture
def decoy(tmp_path):
    live = tmp_path / "decoy" / ".snn_data_lake"
    live.mkdir(parents=True)
    return live


def test_a_prefixed_junction_to_the_live_home_is_refused(tmp_path, decoy):
    link = tmp_path / "selfcheck_home_alias"
    _alias(link, decoy)
    assert link.name.startswith("selfcheck_home_")  # the prefix check alone would accept it
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        H.assert_not_live_home(link, live_home=decoy)


def test_containment_is_refused_in_both_directions(decoy):
    inside = decoy / "sub"
    inside.mkdir()
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        H.assert_not_live_home(inside, live_home=decoy)
    with pytest.raises(H.LiveHomeRefused, match="contains"):
        H.assert_not_live_home(decoy.parent, live_home=decoy)


def test_empty_and_unset_homes_are_refused(monkeypatch, decoy):
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.assert_not_live_home("", live_home=decoy)
    monkeypatch.delenv("NORD_HOME", raising=False)
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.require_isolated_home(live_home=decoy)  # the default home IS the live lake
    monkeypatch.setenv("NORD_HOME", "")
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.require_isolated_home(live_home=decoy)


def test_the_refusal_type_is_catchable_as_value_error(decoy):
    with pytest.raises(ValueError):
        H.assert_not_live_home(decoy, live_home=decoy)


def test_a_fresh_throwaway_home_is_accepted_and_resolved(decoy):
    # the positive: the guard does not refuse a real isolated home, and NORD_HOME holds the
    # RESOLVED path inside the context
    with H.throwaway_home(live_home=decoy) as d:
        assert os.environ["NORD_HOME"] == d
        assert os.path.normcase(d) == os.path.normcase(os.path.realpath(d))
        assert H.require_isolated_home(live_home=decoy) == H.assert_not_live_home(
            d, live_home=decoy
        )


def test_throwaway_home_honours_its_parent(tmp_path, decoy):
    # Codex b3 #11: the multi-GB homes must land under the named parent, not the system TEMP
    parent = tmp_path / "scratch_root"
    parent.mkdir()
    with H.throwaway_home(live_home=decoy, parent=str(parent)) as d:
        assert os.path.normcase(os.path.dirname(d)) == os.path.normcase(os.path.realpath(parent))


def test_a_home_on_another_drive_is_not_contained(tmp_path):
    # commonpath raises on different drives; the guard must read that as "not contained", not crash
    if sys.platform != "win32":
        pytest.skip("drive letters are a Windows concept")
    here = os.path.splitdrive(str(tmp_path))[0].upper()
    other = "D:\\" if here != "D:" else "C:\\"
    if not os.path.isdir(other):
        pytest.skip(f"no {other} on this host")
    assert H.assert_not_live_home(
        tmp_path, live_home=os.path.join(other, "nowhere", ".snn_data_lake")
    )
