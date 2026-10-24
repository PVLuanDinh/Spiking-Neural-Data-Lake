"""NightlySkills.save writes uncompressed npz blobs into the lake.

Sidecar files written by the old savez_compressed call still load (imported once).
"""

import io
import os
import zipfile

import numpy as np

from snn_data_lake.sleep import nightly_skills as ns
from snn_data_lake.substrate import home as h

_SMALL = {
    "NORD_NIGHT_LANG": "2500",
    "NORD_NIGHT_MATHN": "400",
    "NORD_NIGHT_TRAJ": "40",
    "NORD_NIGHT_KBDIM": "24",
    "NORD_NIGHT_KBEVAL": "60",
    "NORD_NIGHT_KBTRIPLES": "1500",
    "NORD_NIGHT_RFF": "300",
}


def _built(monkeypatch, home):
    for k, v in _SMALL.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("NORD_HOME", str(home))
    n = ns.NightlySkills(seed=0)
    n._ensure_domains()
    rs = np.random.RandomState(7)
    for i, d in enumerate(n._domains.values()):
        d.competence, d.visits, d.baseline, d.registered = 0.25 + i / 10, i + 1, 0.1, True
        if isinstance(d, ns._NightKB):
            d.model.rng.rand(5)
            d.rng.rand(3)
        else:
            s = d.slda
            s.W = rs.randn(*s.W.shape)
            s.b = rs.randn(*s.b.shape)
            s.sum_c = rs.rand(*s.sum_c.shape)
            s.cnt_c = rs.rand(*s.cnt_c.shape)
            s.ZZt = rs.randn(*s.ZZt.shape)
            d.pos = 11 + i
    return n


def _names(n):
    return [ns._skill_state(d) for d in n._domains] + [ns._skill_state("open_world")]


def _raw(names):
    return {k: v for k in names if (v := ns._load_arrays(k)) is not None}


def _methods(names):
    blobs = [b for k in names if (b := h.load_named_state(k)) is not None]
    return {zi.compress_type for b in blobs for zi in zipfile.ZipFile(io.BytesIO(b)).infolist()}


def test_save_is_uncompressed_and_round_trips(monkeypatch, tmp_path):
    a = _built(monkeypatch, tmp_path / "new")
    a.save()
    assert _methods(_names(a)) == {zipfile.ZIP_STORED}
    b = ns.NightlySkills(seed=0)
    b._ensure_domains()
    for name, d in a._domains.items():
        e = b._domains[name]
        assert (e.competence, e.visits, e.baseline, e.registered) == (
            d.competence,
            d.visits,
            d.baseline,
            d.registered,
        )
        if isinstance(d, ns._NightKB):
            assert np.array_equal(e.model.E, d.model.E) and np.array_equal(e.model.R, d.model.R)
            assert e.model.rng.get_state()[2:] == d.model.rng.get_state()[2:]
            assert np.array_equal(e.model.rng.get_state()[1], d.model.rng.get_state()[1])
            assert np.array_equal(e.rng.get_state()[1], d.rng.get_state()[1])
            assert e.tr_pos == d.tr_pos
        else:
            for k in ("W", "b", "sum_c", "cnt_c", "ZZt"):
                assert np.array_equal(getattr(e.slda, k), getattr(d.slda, k))
            assert e.pos == d.pos


def test_old_compressed_files_load_and_match_new(monkeypatch, tmp_path):
    a = _built(monkeypatch, tmp_path / "old")
    os.makedirs(tmp_path / "old" / "skills")
    real = ns._save_arrays
    monkeypatch.setattr(  # the pre-port writer: a compressed sidecar file per blob
        ns,
        "_save_arrays",
        lambda name, **arrays: np.savez_compressed(h.state_path(name), **arrays),
    )
    a.save()
    monkeypatch.setattr(ns, "_save_arrays", real)
    old = {
        n: dict(np.load(h.state_path(n), allow_pickle=False))
        for n in _names(a)
        if os.path.exists(h.state_path(n))
    }
    assert {
        zi.compress_type for n in old for zi in zipfile.ZipFile(h.state_path(n)).infolist()
    } == {zipfile.ZIP_DEFLATED}
    b = ns.NightlySkills(seed=0)
    b._ensure_domains()
    for name, d in a._domains.items():
        assert b._domains[name].competence == d.competence
        assert b._domains[name].visits == d.visits
    # the same state saved by the new code carries identical arrays
    b.save()
    new = _raw(_names(b))
    for fn, arrays in old.items():
        assert arrays.keys() == new[fn].keys()
        for k in arrays:
            assert np.array_equal(arrays[k], new[fn][k]), (fn, k)
    assert _methods(_names(b)) == {zipfile.ZIP_STORED}
