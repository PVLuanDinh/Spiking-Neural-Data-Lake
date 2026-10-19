"""NightlySkills.save writes uncompressed npz.

Files written by the old savez_compressed call still load.
"""

import types
import zipfile

import numpy as np

from snn_data_lake.sleep import nightly_skills as ns

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


def _raw(skills_dir):
    out = {}
    for p in sorted(skills_dir.glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            out[p.name] = {k: z[k] for k in z.files}
    return out


def _methods(skills_dir):
    return {
        zi.compress_type for p in skills_dir.glob("*.npz") for zi in zipfile.ZipFile(p).infolist()
    }


def test_save_is_uncompressed_and_round_trips(monkeypatch, tmp_path):
    a = _built(monkeypatch, tmp_path / "new")
    a.save()
    sk = tmp_path / "new" / "skills"
    assert _methods(sk) == {zipfile.ZIP_STORED}
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
    real = ns.np
    monkeypatch.setattr(
        ns, "np", types.SimpleNamespace(**{**vars(real), "savez": real.savez_compressed})
    )
    a.save()
    monkeypatch.setattr(ns, "np", real)
    old = tmp_path / "old" / "skills"
    assert _methods(old) == {zipfile.ZIP_DEFLATED}
    old_raw = _raw(old)
    b = ns.NightlySkills(seed=0)
    b._ensure_domains()
    for name, d in a._domains.items():
        assert b._domains[name].competence == d.competence
        assert b._domains[name].visits == d.visits
    # the same state saved by the new code carries identical arrays
    b.save()
    for fn, arrays in old_raw.items():
        new = _raw(old)[fn]
        assert arrays.keys() == new.keys()
        for k in arrays:
            assert np.array_equal(arrays[k], new[k]), (fn, k)
    assert _methods(old) == {zipfile.ZIP_STORED}
