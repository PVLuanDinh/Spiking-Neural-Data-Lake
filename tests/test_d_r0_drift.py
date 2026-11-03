"""D-R0 round 1: the drift instrument, its reader and the Part A driver, on synthetic data only."""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from snn_data_lake.instruments import representational_drift as rd
from snn_data_lake.sensorimotor import ophys_ingest

_spec = importlib.util.spec_from_file_location(
    "d_r0_partA", Path(__file__).resolve().parents[1] / "tools" / "d_r0_partA.py"
)
partA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(partA)


def test_d_r0_drift_selfcheck():
    assert rd._selfcheck() == 0


def test_d_r0_ingest_selfcheck():
    assert ophys_ingest._selfcheck() == 0


def test_d_r0_pinned_reference_is_the_built_estimator():
    ref = rd.load_reference()
    now = rd.reference(n=200, seed=rd.SEED_REF)
    assert ref["n"] == 2000 and ref["seed"] == 20261104
    assert abs(now["doses"]["0.7"]["mean"] - ref["doses"]["0.7"]["mean"]) < 0.05


def test_d_r0_stability_index_reads_no_drift_and_drift():
    s1, s2 = rd.synth_pair(30, 0.0, 1)
    d1, d2 = rd.synth_pair(30, 0.85, 1)
    flat = rd.population_rdi(s1, s2, rd.stability_index, 1).mean()
    moved = rd.population_rdi(d1, d2, rd.stability_index, 1).mean()
    assert abs(flat) < 0.05 < moved


def test_d_r0_driver_split_is_the_sealed_draw():
    assert sorted(random.Random(20261103).sample(range(1, 13), 4)) == list(partA.DEV)
    assert sorted(partA.DEV + partA.HELD_OUT) == list(range(1, 13))


def test_d_r0_driver_refuses_held_out_without_the_amendment(tmp_path):
    ev = tmp_path / "ev.md"
    ev.write_text("# pre-reg\nno amendment here\n", encoding="utf-8")
    with pytest.raises(partA.HeldOutRefused, match="no FROZEN-SHA"):
        partA.refuse_held_out_unless_frozen(ev, head="a" * 40)
    ev.write_text("x\nFROZEN-SHA 2026-11-10 " + "b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(partA.HeldOutRefused, match="not the frozen sha"):
        partA.refuse_held_out_unless_frozen(ev, head="a" * 40)
    assert partA.refuse_held_out_unless_frozen(ev, head="b" * 40) == "b" * 40


def test_d_r0_driver_run_refuses_held_out_on_the_registered_file():
    with pytest.raises(partA.HeldOutRefused):
        partA.run("heldout")


def _field(dose_mov, dose_pdg, n=6, weeks=(2, 3), seed=5):
    mov1, mov_w = rd.synth_pair(n, dose_mov, seed, trials=30)
    pdg1, pdg_w = rd.synth_pair(n, dose_pdg, seed + 1)
    return partA.FieldData(
        mov1,
        pdg1,
        {w: mov_w for w in weeks},
        {w: pdg_w for w in weeks},
        {w: np.ones(n, bool) for w in weeks},
    )


def test_d_r0_driver_e1_sign_follows_which_stimulus_drifts():
    more_mov = _field(0.85, 0.0)
    more_pdg = _field(0.0, 0.85)
    assert partA.field_d(more_mov) > 0.05
    assert partA.field_d(more_pdg) < -0.05


def test_d_r0_driver_field_d_is_none_with_no_present_neuron():
    f = _field(0.5, 0.5)
    empty = f._replace(present={w: np.zeros(6, bool) for w in f.present})
    assert partA.field_d(empty) is None


def test_d_r0_driver_read_field_is_not_guessed():
    with pytest.raises(NotImplementedError):
        partA.read_field(1)


def _repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
    ):
        ophys_ingest._git(tmp_path, *cmd)
    return tmp_path


def _mat(tmp_path):
    from scipy.io import savemat

    from snn_data_lake.sensorimotor.organoid_ingest import stream_sha256

    mat = tmp_path / "unpinned.mat"
    savemat(mat, {"RespMat": np.arange(6.0)})
    sha, size = stream_sha256(mat)
    return mat, {"path": "unpinned.mat", "size": size, "sha256": sha}


def test_d_r0_ingest_refuses_a_file_absent_from_the_pin(tmp_path):
    repo = _repo(tmp_path)
    mat, entry = _mat(tmp_path)
    pin = {"license": "spdx:CC0-1.0", "assets": []}
    with pytest.raises(ophys_ingest.IngestRefused):
        ophys_ingest.read_response(mat, pin, tmp_path, repo, repo / "tools" / "d_r0_dev_log.jsonl")
