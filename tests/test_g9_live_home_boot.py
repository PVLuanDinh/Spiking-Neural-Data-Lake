"""N1 (closing report 2026-10-14): a persisted Brain with no posed problems boots R2 open."""

import gc
import os

import pytest


def _boot_and_close(monkeypatch, *, decoy):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    live = os.environ["NORD_HOME"]
    if decoy:
        # home() has its own pytest tripwire that refuses the decoy before R2 boots; drop only that
        # signal so the R2 path under test (require_isolated_home) is what decides.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(home, "_default_live_home", lambda: live)
    monkeypatch.setenv("NORD_PERSIST", "1")
    brain = make_brain(seed=0, persist=True)
    try:
        return brain._r2_fault
    finally:
        brain.close()
        del brain
        gc.collect()


def test_persisted_brain_without_posed_problems_boots_r2_open_on_the_live_home(monkeypatch):
    assert _boot_and_close(monkeypatch, decoy=True) is None


def test_control_persisted_brain_boots_r2_open_on_an_isolated_home(monkeypatch):
    assert _boot_and_close(monkeypatch, decoy=False) is None


def test_birth_records_still_refused_on_the_live_home(monkeypatch):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    live = os.environ["NORD_HOME"]
    monkeypatch.setenv("NORD_PERSIST", "1")
    brain = make_brain(seed=0, persist=True)
    try:
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(home, "_default_live_home", lambda: live)
        with pytest.raises(home.LiveHomeRefused):
            brain._g9_birth_records()
    finally:
        brain.close()
        del brain
        gc.collect()
