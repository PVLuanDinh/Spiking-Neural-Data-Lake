"""Module-scoped fixtures get the module's own NORD_HOME (conftest._hermetic_module_home)."""

import os

import pytest


@pytest.fixture(scope="module")
def seen_at_module_setup():
    return os.environ.get("NORD_HOME")


def test_a_module_fixture_sees_its_own_isolated_home(seen_at_module_setup):
    # red without the module fixture: unset under pytest -n, or a LEAKED home serially
    assert seen_at_module_setup and "module_home" in seen_at_module_setup, seen_at_module_setup
