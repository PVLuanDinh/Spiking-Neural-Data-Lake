"""P4b: the acquired language organ and the nightly skills keep their learned arrays IN the lake.

Every test runs on conftest's hermetic NORD_HOME and asserts it is not the live home.
"""

import os
import time

import numpy as np

from snn_data_lake.faculties.language_acquisition import STATE_NAME, LanguageAcquisition
from snn_data_lake.gates.lake_rebuild_gate import assert_rebuild
from snn_data_lake.sleep import nightly_skills as ns
from snn_data_lake.substrate import home as h
from tests.test_nightly_skills_savez import _built

_LIVE = os.path.realpath(os.path.expanduser("~/.snn_data_lake"))


def _home():
    root = h.home()
    assert os.path.realpath(root) != _LIVE
    return root


def _restart():
    """A fresh process: drop every cached BinLog (and with it the named-state index)."""
    h._LOG_CACHE.clear()


def _sidecars(root):
    out = [STATE_NAME] if os.path.exists(os.path.join(root, STATE_NAME)) else []
    sk = os.path.join(root, "skills")
    if os.path.isdir(sk):
        out += ["skills/" + f for f in os.listdir(sk) if f.endswith(".npz")]
    return out


def _lm(seed=0):
    lm = LanguageAcquisition(seed=seed)
    lm.W = np.random.RandomState(seed + 3).randn(*lm.W.shape)
    lm._ctx_ids = [1, 2, 3]
    lm.n_seen = 41
    return lm


def _same_lm(a, b):
    assert np.array_equal(a.W, b.W) and list(a._ctx_ids) == list(b._ctx_ids)
    assert a.n_seen == b.n_seen


def _same_skills(a, b):
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
        else:
            for k in ("W", "b", "sum_c", "cnt_c", "ZZt"):
                assert np.array_equal(getattr(e.slda, k), getattr(d.slda, k))


def _fresh_skills():
    s = ns.NightlySkills(seed=0)
    s._ensure_domains()
    return s


def test_save_writes_no_sidecar(monkeypatch):
    root = _home()
    _lm().save()
    _built(monkeypatch, root).save()
    assert _sidecars(root) == []


def test_restart_reads_weights_from_the_lake(monkeypatch):
    root = _home()
    a = _lm()
    a.save()
    s = _built(monkeypatch, root)
    s.save()
    for p in _sidecars(root):
        os.remove(os.path.join(root, p))
    _restart()
    _same_lm(a, LanguageAcquisition(seed=9).load())
    _same_skills(s, _fresh_skills())


def test_legacy_language_sidecar_migrates_once():
    root = _home()
    old = _lm(seed=5)
    path = os.path.join(root, STATE_NAME)
    np.savez(path, W=old.W, ctx=np.asarray(old._ctx_ids), n=old.n_seen)  # the c9d0c097 writer
    assert h.load_named_state(STATE_NAME, path=h.knowledge_path()) is None
    _same_lm(old, LanguageAcquisition(seed=9).load())
    np.savez(path, W=old.W * 0, ctx=np.asarray([7]), n=0)
    _restart()
    _same_lm(old, LanguageAcquisition(seed=9).load())


def test_legacy_skills_sidecars_migrate_once(monkeypatch):
    root = _home()
    s = _built(monkeypatch, root)
    real = h.save_named_state

    def sidecar_writer(name, blob, path=None):  # what <home>/skills/*.npz held before the port
        if not name.startswith("skills/"):
            return real(name, blob, path)
        with open(h.state_path(name), "wb") as f:
            f.write(blob)
        return True

    os.makedirs(os.path.join(root, "skills"), exist_ok=True)
    monkeypatch.setattr(ns, "save_named_state", sidecar_writer)
    s.save()
    monkeypatch.setattr(ns, "save_named_state", real)
    legacy = _sidecars(root)
    assert legacy and all(p.startswith("skills/") for p in legacy)
    _restart()
    _same_skills(s, _fresh_skills())
    for p in legacy:
        with open(os.path.join(root, p), "wb") as f:
            f.write(b"not an npz")
    _restart()
    _same_skills(s, _fresh_skills())


def test_open_world_and_discovered_skills_round_trip(monkeypatch):
    root = _home()
    s = _built(monkeypatch, root)
    slda = next(n for n, d in s._domains.items() if not isinstance(d, ns._NightKB))
    s._domains["discovered-3"] = s._domains[slda]
    ow = s._ensure_ow()
    ow.cents[3] = np.arange(4.0)
    ow.counts[3] = 5
    s._ow_pos = 7
    s.save()
    summary = h.load_state(ns._STATE_JSON)
    del summary["domains"]["discovered-3"]  # a crash between the arrays and the summary
    h.save_state(ns._STATE_JSON, summary)
    _restart()
    t = ns.NightlySkills(seed=0)
    got, want = t._pending["discovered-3"], s._domains[slda].slda
    assert np.array_equal(got["W"], want.W) and np.array_equal(got["ZZt"], want.ZZt)
    tw = t._ensure_ow()
    assert np.array_equal(tw.cents[3], ow.cents[3]) and tw.counts[3] == 5 and t._ow_pos == 7


def test_lake_rebuild_gate_green_with_new_records(monkeypatch, tmp_path):
    root = _home()
    _lm().save()
    s = _built(monkeypatch, root)
    s.save()
    for name in [STATE_NAME] + [ns._skill_state(n) for n in s._domains]:
        assert h.load_named_state(name, path=h.knowledge_path()) is not None, name
    ok, msg = assert_rebuild(root, tmp_path / "idx.db")
    assert ok, msg


def test_unchanged_blob_is_not_rewritten(monkeypatch):
    _home()
    # the skip relies on np.savez writing a FIXED zip timestamp: it opens each member by name, so
    # CPython zipfile stamps ZipInfo's 1980 default. A wall-clock stamp would re-append every
    # save, and this clock puts each save an hour apart.
    clock = iter(range(10**9, 10**9 + 10**6, 3600))
    monkeypatch.setattr(time, "time", lambda: next(clock))
    lm = _lm()
    assert lm.save() is True and lm.save() is False


def test_persisting_brain_serves_the_lake_language_state():
    from snn_data_lake.engine.brain import make_brain

    _home()
    a = _lm()
    a.save()
    _restart()
    b = make_brain(seed=0, persist=True)
    try:
        _same_lm(a, b._lm_serve())
    finally:
        b.close()


def test_brain_close_saves_the_language_state_into_the_lake():
    from snn_data_lake.engine.brain import make_brain

    _home()
    b = make_brain(seed=0, persist=True)
    lm = b._lm_serve()
    lm.W = lm.W + 1.0
    lm.n_seen += 3
    b.close()
    _restart()
    _same_lm(lm, LanguageAcquisition(seed=9).load())


def test_unchanged_skills_are_not_rewritten(monkeypatch):
    root = _home()
    s = _built(monkeypatch, root)
    s.save()
    size = os.path.getsize(h.knowledge_path())
    s.save()
    assert os.path.getsize(h.knowledge_path()) == size
