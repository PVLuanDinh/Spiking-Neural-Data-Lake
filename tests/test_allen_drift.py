"""allen_drift (final project, P4c move 2026-11-15): the self-check passes, and its two new guards
are reachable reds -- a tampered frozen record fails check_records(), and a frozen function that
no longer matches its record is named by frozen_identical()."""

import shutil

import pytest

from snn_data_lake.instruments import allen_drift as A


def test_allen_selfcheck_passes():
    assert A._selfcheck() == 0


def test_tampered_record_is_refused(tmp_path, monkeypatch):
    shutil.copytree(A.RECORDS, tmp_path / "rec")
    rec = tmp_path / "rec" / "extract.py.txt"
    rec.write_bytes(rec.read_bytes() + b"# tampered\n")
    monkeypatch.setattr(A, "RECORDS", tmp_path / "rec")
    with pytest.raises(SystemExit, match="extract.py"):
        A.check_records()


def test_drifted_frozen_copy_is_named(tmp_path, monkeypatch):
    shutil.copytree(A.RECORDS, tmp_path / "rec")
    rec = tmp_path / "rec" / "analyze_dev.py.txt"
    rec.write_text(rec.read_text(encoding="utf-8").replace("1e-12", "1e-9"), encoding="utf-8")
    monkeypatch.setattr(A, "RECORDS", tmp_path / "rec")
    assert A.frozen_identical() == ["residualize"]


def test_sign_test_and_critical_count():
    assert A.crit(44) == 30
    assert A.sign_test([0.1, -0.2, 0.0, float("nan")], "pos") == (1, 2, 0.75, 2)
