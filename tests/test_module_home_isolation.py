"""Module-scoped fixtures get the module's own NORD_HOME (conftest._hermetic_module_home)."""

import os
import types

import pytest


class PostTestHomes:
    """NORD_HOME as each test of `module` leaves it, read after the whole function teardown.

    The module's last test is skipped: its teardown also unwinds the module home.

    The module fixtures using it group the whole module onto one xdist worker (conftest), which is
    what lets the recorder see every test of the module.
    """

    def __init__(self, module):
        self.module = module
        self.module_home = os.environ.get("NORD_HOME")
        self.homes = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        yield
        if item.module is self.module and nextitem is not None and nextitem.module is self.module:
            self.homes.append(os.environ.get("NORD_HOME"))


@pytest.fixture(scope="module", autouse=True)
def post_test_homes(request):
    recorder = PostTestHomes(request.module)
    request.config.pluginmanager.register(recorder)
    yield recorder
    request.config.pluginmanager.unregister(recorder)


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


_FUNCTION_HOMES = []


def test_separation_first_test_records_its_function_home():
    _FUNCTION_HOMES.append(os.environ["NORD_HOME"])


def test_separation_two_tests_in_one_module_see_different_function_homes(seen_at_module_setup):
    # a leaked home or a dropped function fixture makes both tests see one value
    here = os.environ["NORD_HOME"]
    if not _FUNCTION_HOMES:
        pytest.skip("needs test_separation_first_test_records_its_function_home in the same run")
    assert here != _FUNCTION_HOMES[0]
    assert seen_at_module_setup not in (here, _FUNCTION_HOMES[0])


def _teardown(recorder, item, nextitem):
    hook = recorder.pytest_runtest_teardown(item, nextitem)
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)


def test_the_recorder_reads_a_leaked_home_after_teardown(monkeypatch, request):
    # sensitivity control: proves an empty recorder result means no leak
    recorder = PostTestHomes(request.module)
    monkeypatch.setenv("NORD_HOME", "leaked")
    item = types.SimpleNamespace(module=request.module)
    _teardown(recorder, item, item)
    assert recorder.homes == ["leaked"]


def test_the_recorder_skips_the_last_test_and_other_modules(monkeypatch, request):
    recorder = PostTestHomes(request.module)
    monkeypatch.setenv("NORD_HOME", "leaked")
    mine = types.SimpleNamespace(module=request.module)
    other = types.SimpleNamespace(module=object())
    _teardown(recorder, mine, None)
    _teardown(recorder, mine, other)
    _teardown(recorder, other, mine)
    assert recorder.homes == []


def test_nord_home_after_each_earlier_test_is_the_module_home(post_test_homes):
    # a leak shows up as an earlier test's function home or the launch value
    assert set(post_test_homes.homes) <= {post_test_homes.module_home}, post_test_homes.homes
    assert post_test_homes.homes or not _FUNCTION_HOMES, "recorder saw none of the earlier tests"
