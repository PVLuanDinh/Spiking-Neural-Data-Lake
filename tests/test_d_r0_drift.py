"""D-R0 round 1: the drift instrument, its reader and the Part A driver, on synthetic data only."""

import random
import sys

import numpy as np
import pytest

from snn_data_lake.instruments import d_r0_part_a as partA
from snn_data_lake.instruments import representational_drift as rd
from snn_data_lake.sensorimotor import ophys_ingest, ophys_layout


@pytest.fixture(autouse=True)
def _a_hashable_calling_tool(monkeypatch):
    """Under xdist the worker's __main__ is execnet's bootstrap, which has no __file__, so
    read_response refuses every call as interactive (5 reds under -n, green without it; D-190).
    The calling tool here is this test module, a real file source_hashes() can read. The
    interactive-main red below still replaces __main__ outright, so the refusal stays tested."""
    main = sys.modules.get("__main__")
    if main is not None and not getattr(main, "__file__", None):
        monkeypatch.setattr(main, "__file__", __file__, raising=False)


def test_d_r0_part_a_selfcheck():
    assert partA._selfcheck() == 0


def test_d_r0_layout_selfcheck():
    assert ophys_layout._selfcheck() == 0


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
    return mat, {"path": "unpinned.mat", "size": size, "sha256": sha, "split": "dev"}


def test_d_r0_ingest_refuses_a_file_absent_from_the_pin(tmp_path):
    repo = _repo(tmp_path)
    mat, entry = _mat(tmp_path)
    pin = {"license": "spdx:CC0-1.0", "assets": []}
    with pytest.raises(ophys_ingest.IngestRefused):
        ophys_ingest.read_response(
            mat, pin, "dev", tmp_path, repo, repo / "tools" / "d_r0_dev_log.jsonl"
        )


def test_d_r0_ingest_refuses_a_read_from_an_interactive_main(tmp_path, monkeypatch):
    import sys
    import types

    repo = _repo(tmp_path)
    mat, entry = _mat(tmp_path)
    pin = {"license": "spdx:CC0-1.0", "assets": [entry]}
    log = repo / "tools" / "d_r0_dev_log.jsonl"
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    with pytest.raises(ophys_ingest.IngestRefused, match="interactive"):
        ophys_ingest.read_response(mat, pin, "dev", tmp_path, repo, log)
    assert not log.exists()


_DEV_LINES = "DEV-AGREEMENT MET 2026-11-08\nDEV-CONTROL MET 2026-11-07\n"


def _gate(tmp_path, text, **kw):
    ev = tmp_path / "ev.md"
    ev.write_text(text + "FROZEN-SHA 2026-11-10 " + "b" * 40 + "\n", encoding="utf-8")
    return partA.refuse_held_out_unless_frozen(ev, head="b" * 40, **kw)


def test_d_r0_driver_held_out_needs_the_dev_agreement_line(tmp_path):
    with pytest.raises(partA.HeldOutRefused, match="DEV-AGREEMENT"):
        _gate(
            tmp_path,
            "DEV-CONTROL MET 2026-11-07\n",
            selfchecks=lambda: [],
            log_committed=lambda: True,
        )


def test_d_r0_driver_held_out_needs_the_dev_control_line(tmp_path):
    with pytest.raises(partA.HeldOutRefused, match="DEV-CONTROL"):
        _gate(
            tmp_path,
            "DEV-AGREEMENT MET 2026-11-08\n",
            selfchecks=lambda: [],
            log_committed=lambda: True,
        )


def test_d_r0_driver_held_out_needs_both_selfchecks_green(tmp_path):
    with pytest.raises(partA.HeldOutRefused, match="self-check"):
        _gate(tmp_path, _DEV_LINES, selfchecks=lambda: ["ophys_ingest"], log_committed=lambda: True)


def test_d_r0_driver_held_out_needs_the_dev_log_committed(tmp_path):
    with pytest.raises(partA.HeldOutRefused, match="dev log"):
        _gate(tmp_path, _DEV_LINES, selfchecks=lambda: [], log_committed=lambda: False)


def test_d_r0_driver_held_out_green_when_every_condition_holds(tmp_path):
    assert (
        _gate(tmp_path, _DEV_LINES, selfchecks=lambda: [], log_committed=lambda: True) == "b" * 40
    )


def test_d_r0_driver_dev_log_committed_reads_git(tmp_path):
    repo = _repo(tmp_path)
    log = repo / "tools" / "d_r0_dev_log.jsonl"
    assert not partA.dev_log_committed(repo, log)
    log.write_text("{}\n", encoding="utf-8")
    assert not partA.dev_log_committed(repo, log)
    ophys_ingest._git(repo, "add", ".")
    ophys_ingest._git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "log")
    assert partA.dev_log_committed(repo, log)
    log.write_text("{}\n{}\n", encoding="utf-8")
    assert not partA.dev_log_committed(repo, log)


def test_d_r0_driver_refuses_to_start_while_its_result_file_is_dirty(tmp_path):
    repo = _repo(tmp_path)
    out = repo / "tools" / "d_r0_partA.json"
    assert not partA.result_file_dirty(repo, out)
    out.write_text("{}", encoding="utf-8")
    assert partA.result_file_dirty(repo, out)


def _pin(*rows):
    return {
        "assets": [{"path": p, "sha256": sha, "split": sp, "mouse": None} for p, sha, sp in rows]
    }


def _dev_log(tmp_path, *shas):
    import json

    log = tmp_path / "dev_log.jsonl"
    log.write_text(
        "".join(
            json.dumps({"commit": "a" * 40, "files": [{"path": "x.mat", "sha256": sha}]}) + "\n"
            for sha in shas
        ),
        encoding="utf-8",
    )
    return log


_PIN = _pin(
    ("Main experiments/Mouse 1/r.mat", "h" * 64, "held-out"),
    ("Prism experiments/Mouse 2/r.mat", "h" * 64, "held-out"),
    ("Main experiments/Mouse 2/r.mat", "d" * 64, "dev"),
    ("Inhibitory experiments/Mouse 1/r.mat", "n" * 64, "none"),
)


def test_d_r0_driver_held_out_read_once_refuses_a_logged_held_out_file(tmp_path):
    log = _dev_log(tmp_path, "d" * 64, "h" * 64)
    assert partA.held_out_read_in_log(log, _PIN)
    with pytest.raises(partA.HeldOutRefused, match="read ONCE"):
        _gate(
            tmp_path,
            _DEV_LINES,
            selfchecks=lambda: [],
            log_committed=lambda: True,
            held_out_logged=lambda: partA.held_out_read_in_log(log, _PIN),
        )


def test_d_r0_driver_held_out_read_once_sees_a_prism_copy_under_a_dev_numbered_path(tmp_path):
    # the log row's path says nothing; only the sha (pin: held-out) decides
    log = _dev_log(tmp_path, "h" * 64)
    assert partA.held_out_read_in_log(log, _PIN)


def test_d_r0_driver_held_out_read_once_fails_closed_on_a_sha_absent_from_the_pin(tmp_path):
    assert partA.held_out_read_in_log(_dev_log(tmp_path, "z" * 64), _PIN)


def test_d_r0_driver_held_out_read_once_passes_a_dev_only_log(tmp_path):
    assert not partA.held_out_read_in_log(_dev_log(tmp_path, "d" * 64, "n" * 64), _PIN)
    assert not partA.held_out_read_in_log(tmp_path / "absent.jsonl", _PIN)


def test_d_r0_driver_the_tracked_pin_split_is_the_sealed_draw(tmp_path):
    import json

    pin = json.loads(partA.PIN_PATH.read_text(encoding="utf-8"))
    assets = pin["assets"]
    mice = lambda sp: sorted({a["mouse"] for a in assets if a["split"] == sp})  # noqa: E731
    assert mice("held-out") == sorted(partA.HELD_OUT)
    assert mice("dev") == sorted(partA.DEV)
    for a in assets:
        if not a["path"].startswith("Main experiments/") and a["split"] != "none":
            assert a.get("copy_of"), a["path"]
    prism_held = next(
        a for a in assets if a["split"] == "held-out" and a["path"].startswith("Prism")
    )
    assert partA.held_out_read_in_log(_dev_log(tmp_path, prism_held["sha256"]), pin)


def test_d_r0_driver_held_out_read_once_is_checked_before_the_selfchecks(tmp_path):
    def boom():
        raise AssertionError("self-checks ran before the read-once check")

    with pytest.raises(partA.HeldOutRefused, match="read ONCE"):
        _gate(
            tmp_path,
            _DEV_LINES,
            selfchecks=boom,
            log_committed=lambda: True,
            held_out_logged=lambda: True,
        )


def _read_with_split(tmp_path, pin_split, call_split):
    repo = _repo(tmp_path)
    mat, entry = _mat(tmp_path)
    entry["split"] = pin_split
    pin = {"license": "spdx:CC0-1.0", "assets": [entry]}
    log = repo / "tools" / "d_r0_dev_log.jsonl"
    return ophys_ingest.read_response(
        mat, pin, split=call_split, data_dir=tmp_path, root=repo, log_path=log
    ), log


def test_d_r0_ingest_refuses_a_held_out_entry_read_as_dev(tmp_path):
    with pytest.raises(ophys_ingest.IngestRefused, match="split"):
        _read_with_split(tmp_path, "held-out", "dev")


def test_d_r0_ingest_refuses_a_dev_entry_read_as_held_out(tmp_path):
    with pytest.raises(ophys_ingest.IngestRefused, match="split"):
        _read_with_split(tmp_path, "dev", "heldout")


def test_d_r0_ingest_refuses_a_split_none_entry_in_either_split(tmp_path):
    for call in ("dev", "heldout"):
        sub = tmp_path / call
        sub.mkdir()
        with pytest.raises(ophys_ingest.IngestRefused, match="split"):
            _read_with_split(sub, "none", call)


def test_d_r0_ingest_reads_an_entry_whose_split_matches(tmp_path):
    for pin_split, call in (("dev", "dev"), ("held-out", "heldout")):
        sub = tmp_path / call
        sub.mkdir()
        data, _ = _read_with_split(sub, pin_split, call)
        assert "RespMat" in data
