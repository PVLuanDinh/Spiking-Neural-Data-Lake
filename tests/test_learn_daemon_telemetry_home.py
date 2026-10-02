"""The daemon's telemetry follows the active home (NORD_HOME), never the checkout.

Measured 2026-10-01 by the local verify skill: a daemon started on a throwaway NORD_HOME still
appended to <checkout>/data/telemetry.jsonl, the same file the real daemon writes, so every probe
brain mixed its snapshots into the real record (1,163 lines, 32.6 MB that day).
"""

import os

from snn_data_lake.engine import learn_daemon
from snn_data_lake.infra.self_mod_harness import _repo_root
from snn_data_lake.substrate.home import home, throwaway_home


def test_telemetry_path_is_inside_the_active_home():
    with throwaway_home():
        path = learn_daemon._telemetry_path()
        assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(home())
        assert os.path.basename(path) == "telemetry.jsonl"


def test_telemetry_path_never_points_into_the_checkout():
    with throwaway_home():
        path = os.path.abspath(learn_daemon._telemetry_path())
        assert not path.startswith(os.path.abspath(_repo_root()) + os.sep)


def test_telemetry_path_moves_with_the_home():
    with throwaway_home():
        first = learn_daemon._telemetry_path()
    with throwaway_home():
        second = learn_daemon._telemetry_path()
    assert first != second
