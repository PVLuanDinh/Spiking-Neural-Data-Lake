"""The daemon's telemetry follows the active home (NORD_HOME), not the checkout.

Measured 2026-10-01 by the local verify skill: a daemon started on a throwaway NORD_HOME still
appended to <checkout>/data/telemetry.jsonl, the same file the real daemon writes, so every probe
brain mixed its snapshots into the real record (1,163 lines, 32.6 MB that day).
"""

import ast
import inspect
import os

from snn_data_lake.engine import learn_daemon
from snn_data_lake.infra.self_mod_harness import _repo_root
from snn_data_lake.substrate.home import home, throwaway_home


def _norm(path):
    # realpath resolves junctions/symlinks, normcase folds Windows drive-letter and path case
    return os.path.normcase(os.path.realpath(path))


def test_telemetry_path_is_inside_the_active_home():
    with throwaway_home():
        path = learn_daemon._telemetry_path()
        assert _norm(os.path.dirname(path)) == _norm(home())
        assert os.path.basename(path) == "telemetry.jsonl"


def test_telemetry_path_never_points_into_the_checkout_from_a_throwaway_home():
    with throwaway_home():
        path = _norm(learn_daemon._telemetry_path())
        assert not path.startswith(_norm(_repo_root()) + os.sep)


def test_telemetry_path_moves_with_the_home():
    with throwaway_home():
        first = learn_daemon._telemetry_path()
    with throwaway_home():
        second = learn_daemon._telemetry_path()
    assert _norm(first) != _norm(second)


def test_the_daemons_recorder_writes_where_the_path_says():
    with throwaway_home():
        rec = learn_daemon._make_recorder()
        try:
            assert _norm(rec.path) == _norm(learn_daemon._telemetry_path())
        finally:
            rec.close()


def test_the_drive_loop_builds_its_recorder_only_through_make_recorder():
    """Pins the wiring the helper tests cannot see: main() must not construct a Recorder itself
    (reverting it to Recorder(<checkout path>) would leave the helper tests green)."""
    tree = ast.parse(inspect.getsource(learn_daemon))
    direct = [
        node.lineno
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name != "_make_recorder"
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Recorder"
    ]
    assert direct == [], f"Recorder constructed outside _make_recorder at lines {direct}"
    # the assignment must sit directly in main()'s body, so a guarded or dead call cannot satisfy it
    main = next(
        fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "main"
    )
    assigned = [
        stmt
        for stmt in main.body
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and getattr(stmt.value.func, "id", None) == "_make_recorder"
        and [getattr(t, "id", None) for t in stmt.targets] == ["rec"]
    ]
    assert len(assigned) == 1, "main() must assign rec = _make_recorder() unconditionally"
