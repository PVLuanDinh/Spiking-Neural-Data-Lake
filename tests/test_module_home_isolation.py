"""Module-scoped fixtures get the module's own NORD_HOME (conftest._hermetic_module_home)."""

import os

import pytest


@pytest.fixture(scope="module")
def seen_at_module_setup():
    return os.environ.get("NORD_HOME")


def test_a_module_fixture_sees_its_own_isolated_home(seen_at_module_setup):
    # red without the module fixture: unset under pytest -n, or a LEAKED home serially
    assert seen_at_module_setup and "module_home" in seen_at_module_setup, seen_at_module_setup


class _FD:
    def __init__(self, scope):
        self.scope = scope


class _Item:
    def __init__(self, fixtures):
        self._fixtureinfo = type("FI", (), {"name2fixturedefs": fixtures})()
        self.module = type("M", (), {"__name__": "tests.test_fake"})()
        self.marks = []

    def get_closest_marker(self, name):
        return None

    def add_marker(self, mark):
        self.marks.append(mark)


def test_the_hermetic_module_home_alone_does_not_group_a_module_under_xdist():
    # red when the hook counts _hermetic_module_home: every module then lands on one worker
    from conftest import pytest_collection_modifyitems

    only_home = _Item({"_hermetic_module_home": [_FD("module")], "tmp_path": [_FD("function")]})
    real = _Item({"_hermetic_module_home": [_FD("module")], "live": [_FD("module")]})
    pytest_collection_modifyitems([only_home, real])
    assert only_home.marks == []
    assert len(real.marks) == 1 and real.marks[0].name == "xdist_group"
