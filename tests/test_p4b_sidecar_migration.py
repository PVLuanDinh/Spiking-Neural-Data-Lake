"""P4b: the acquired language organ and the nightly skills keep their learned arrays IN the lake.

Every test runs on conftest's hermetic NORD_HOME and asserts it is not the live home.
"""

import hashlib
import io
import json
import os
import shutil
import time

import numpy as np
import pytest

from snn_data_lake.faculties import language_acquisition as la
from snn_data_lake.faculties.language_acquisition import STATE_NAME, LanguageAcquisition
from snn_data_lake.gates.lake_rebuild_gate import assert_rebuild
from snn_data_lake.infra.brain_native_selfbuild import _LEGACY_STATE as _SB
from snn_data_lake.sleep import nightly_skills as ns
from snn_data_lake.substrate import home as h
from tests.test_nightly_skills_savez import _built

_LIVE = os.path.realpath(os.path.expanduser("~/.snn_data_lake"))


def _home():
    root = h.home()
    assert os.path.realpath(root) != _LIVE
    return root


def _restart():
    """A fresh process: drop every cached BinLog (and with it the named-state index) and the
    per-process record of retired sidecars."""
    h._LOG_CACHE.clear()
    h._RETIRED.clear()


def _probe_sidecar(root, text='{"v": 1}', name="probe.json"):
    path = os.path.join(root, name)
    with open(path, "w", encoding="ascii") as f:
        f.write(text)
    return path


def _retired(path, suffix, text):
    with open(path + suffix, "w", encoding="ascii") as f:
        f.write(text)


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
    shutil.copy2(path + ".imported", path)  # the retired bytes again: not learning, not imported
    _restart()
    _same_lm(old, LanguageAcquisition(seed=9).load())
    newer = _lm(seed=6)  # a rollback window rewrote the file: imported before the rename (R3-M3)
    np.savez(path, W=newer.W, ctx=np.asarray(newer._ctx_ids), n=newer.n_seen)
    _restart()
    _same_lm(newer, LanguageAcquisition(seed=9).load())


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
    monkeypatch.setattr(la, "save_named_state", sidecar_writer)  # under save_npz_state
    s.save()
    monkeypatch.setattr(la, "save_named_state", real)
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


def test_an_imported_sidecar_is_not_imported_again_after_a_lake_reset():
    """A-m1: the import renames the file to `<name>.imported` -- bytes kept, never re-read."""
    root = _home()
    old = _lm(seed=5)
    path = os.path.join(root, STATE_NAME)
    np.savez(path, W=old.W, ctx=np.asarray(old._ctx_ids), n=old.n_seen)
    with open(path, "rb") as f:
        raw = f.read()
    _same_lm(old, LanguageAcquisition(seed=9).load())
    with open(path + ".imported", "rb") as f:
        assert f.read() == raw
    _restart()
    os.remove(h.knowledge_path())  # a lake reset
    _restart()
    assert h.load_named_state(STATE_NAME) is None


def test_a_legacy_skills_npz_missing_from_the_summary_is_imported():
    """A-m2: a domain whose first save crashed between its arrays and the summary."""
    root = _home()
    h.save_state(ns._STATE_JSON, {"domains": {}})
    os.makedirs(os.path.join(root, "skills"), exist_ok=True)
    np.savez(h.state_path(ns._skill_state("orphan")), W=np.arange(3.0))
    np.savez(os.path.join(root, "skills", "scratch.tmp.npz"), W=np.arange(2.0))
    np.savez(h.state_path(ns._skill_state("open_world")), dids=np.arange(0))
    for _boot in range(2):  # the second boot finds the orphan by its renamed `.imported` file
        _restart()
        t = ns.NightlySkills(seed=0)
        assert np.array_equal(t._pending["orphan"]["W"], np.arange(3.0))
        assert "scratch.tmp" not in t._pending and "open_world" not in t._pending
    assert os.path.exists(h.state_path(ns._skill_state("orphan")) + ".imported")


def test_a_corrupt_legacy_skills_npz_skips_only_itself():
    """A stray unreadable file under skills/ must not cost the domains the summary names."""
    root = _home()
    ns._save_arrays(ns._skill_state("kept"), W=np.arange(2.0))
    h.save_state(ns._STATE_JSON, {"domains": {"kept": {}}})
    os.makedirs(os.path.join(root, "skills"), exist_ok=True)
    with open(h.state_path(ns._skill_state("broken")), "wb") as f:
        f.write(b"not an npz")
    _restart()
    t = ns.NightlySkills(seed=0)
    assert np.array_equal(t._pending["kept"]["W"], np.arange(2.0)) and "broken" not in t._pending


def test_a_corrupt_record_the_summary_names_still_fails_the_whole_load():
    """Only stray legacy files are skipped one by one; a domain the summary names keeps the old
    contract (the load raises and the constructor starts fresh), so a learned domain is never
    silently dropped while the rest restore around it."""
    _home()
    h.save_named_state(ns._skill_state("named"), b"not an npz")
    ns._save_arrays(ns._skill_state("other"), W=np.arange(2.0))
    h.save_state(ns._STATE_JSON, {"domains": {"named": {}, "other": {}}})
    _restart()
    with pytest.raises(ValueError):  # np.load refuses non-zip bytes (allow_pickle=False)
        ns.NightlySkills(seed=0).load()


def test_a_record_missing_a_field_leaves_the_language_organ_untouched():
    """A-m5: never half-restored -- a blob with W but no n raises before anything is assigned."""
    _home()
    buf = io.BytesIO()
    np.savez(buf, W=np.zeros((2, 2)), ctx=np.asarray([1]))
    lm = _lm()
    before = lm.W.copy()
    with pytest.raises(KeyError):
        lm.load(buf.getvalue())
    assert np.array_equal(lm.W, before) and lm.n_seen == 41


def test_one_npz_blob_codec_serves_both_organs():
    """A-m8 (P2): the np.savez-to-bytes codec lives once (beside the language organ, so home.py
    stays numpy-free) and the nightly skills use it."""
    _home()
    assert la.save_npz_state("codec", a=np.arange(4.0)) is True
    assert np.array_equal(la.load_npz_state("codec")["a"], np.arange(4.0))
    assert la.load_npz_state("absent") is None
    assert (ns._save_arrays, ns._load_arrays) == (la.save_npz_state, la.load_npz_state)


def test_a_sidecar_imported_before_the_rename_existed_is_retired_on_the_next_read():
    """A home whose lake already holds the record but whose sidecar still sits under its own name
    (the pre-2026-10-24 import): the next read renames it, so a lake reset cannot re-import it."""
    root = _home()
    h.save_state("probe.json", {"v": 1})  # the record an earlier import put in the lake
    path = _probe_sidecar(root)
    _restart()
    assert h.load_state("probe.json") == {"v": 1}
    assert os.path.exists(path + ".imported") and not os.path.exists(path)


def test_retiring_never_overwrites_an_earlier_imported_file():
    """C2-M2: a newer `<name>` beside an existing `.imported` is kept as `<name>.imported.<n>` (one
    past the highest number), so no file is overwritten and a lake reset imports none."""
    root = _home()
    h.save_state("probe.json", {"v": 1})
    path = _probe_sidecar(root, '{"v": 2}')
    _retired(path, ".imported", '{"v": 1}')
    _restart()
    h.load_state("probe.json")
    _probe_sidecar(root, '{"v": 3}')
    _restart()
    h.load_state("probe.json")
    for suffix, text in (
        (".imported", '{"v": 1}'),
        (".imported.1", '{"v": 2}'),
        (".imported.2", '{"v": 3}'),
    ):
        with open(path + suffix, encoding="ascii") as f:
            assert f.read() == text, suffix
    assert not os.path.exists(path)
    _restart()
    os.remove(h.knowledge_path())  # a lake reset
    _restart()
    assert h.load_state("probe.json") is None


def test_a_sidecar_newer_than_its_retired_copy_is_imported_before_the_rename():
    """R3-M3, the rollback window: the lake holds A and the file was retired; an older checkout
    then writes B as the file. One read imports B BEFORE renaming it, so the round trip back to
    this build is lossless: the lake holds B and the file is `.imported.1`."""
    root = _home()
    path = _probe_sidecar(root, '{"v": 1}')
    assert h.load_state("probe.json") == {"v": 1} and os.path.exists(path + ".imported")
    _probe_sidecar(root, '{"v": 2}')
    _restart()
    assert h.load_state("probe.json") == {"v": 2}
    assert os.path.exists(path + ".imported.1") and not os.path.exists(path)
    _restart()
    assert h.load_named_state("probe.json", path=h.knowledge_path()) == b'{"v": 2}'


def test_a_copied_back_retired_copy_is_renamed_unread():
    """UNDO's copy: bytes that match a retired copy are not learning the lake lacks."""
    root = _home()
    path = _probe_sidecar(root, '{"v": 1}')
    h.load_state("probe.json")
    h.save_state("probe.json", {"v": 2})  # the lake learns on
    shutil.copy2(path + ".imported", path)
    _restart()
    assert h.load_state("probe.json") == {"v": 2} and os.path.exists(path + ".imported.1")


def test_a_torn_newer_file_never_replaces_a_readable_record():
    """`_same_kind`: a newer file the record's reader would refuse is renamed unread."""
    root = _home()
    good = io.BytesIO()
    np.savez(good, a=np.arange(3.0))
    path = os.path.join(root, "probe.npz")
    with open(path, "wb") as f:
        f.write(good.getvalue())
    h.load_named_state("probe.npz")
    with open(path, "wb") as f:
        f.write(good.getvalue()[:-7])  # truncated: no end-of-central-directory
    _restart()
    assert h.load_named_state("probe.npz") == good.getvalue()
    assert os.path.exists(path + ".imported.1") and not os.path.exists(path)


def test_a_newer_file_whose_import_did_not_land_keeps_its_name(monkeypatch):
    root = _home()
    path = _probe_sidecar(root, '{"v": 1}')
    h.load_state("probe.json")
    _probe_sidecar(root, '{"v": 2}')
    _restart()
    with monkeypatch.context() as m:
        m.setattr(h, "save_named_state", lambda *a, **k: False)
        assert h.load_state("probe.json") == {"v": 1}
    assert os.path.exists(path) and not os.path.exists(path + ".imported.1")
    _restart()
    assert h.load_state("probe.json") == {"v": 2} and os.path.exists(path + ".imported.1")


def test_named_states_lists_a_prefix_newest_first_across_a_consolidation():
    _home()
    for n in ("a", "b", "c"):
        h.save_state(f"pfx.{n}.json", {"n": n})
    h.save_state("pfy.z.json", {"n": "z"})
    want = ["pfx.c.json", "pfx.b.json", "pfx.a.json"]
    assert [n for n, _ in h.named_states("pfx.")] == want
    kl = h.KnowledgeLog(h.knowledge_path())
    kl.consolidate(list(kl.replay()))
    _restart()
    assert [n for n, _ in h.named_states("pfx.")] == want


def test_a_copy_of_a_retired_copy_past_a_gap_is_renamed_unread():
    """The scan sees every retired copy, gaps included, and a new one lands past the highest."""
    root = _home()
    h.save_state("probe.json", {"v": 3})  # the lake learned on
    path = os.path.join(root, "probe.json")
    _retired(path, ".imported", '{"v": 1}')
    _retired(path, ".imported.2", '{"v": 2}')  # `.imported.1` is gone
    shutil.copy2(path + ".imported.2", path)  # ROLLBACK's copy, never rewritten
    _restart()
    assert h.load_state("probe.json") == {"v": 3}
    assert os.path.exists(path + ".imported.3") and not os.path.exists(path + ".imported.1")


def test_a_name_that_is_not_a_retired_copy_is_ignored():
    """Only `.imported` and `.imported.<n>` are retired copies: a stray `.imported.bak` neither
    blocks the unread rename of a pre-2026-10-24 file nor takes part in the numbering."""
    root = _home()
    h.save_state("probe.json", {"v": 2})
    path = _probe_sidecar(root, '{"v": 1}')
    _retired(path, ".imported.bak", '{"v": 0}')
    _restart()
    assert h.load_state("probe.json") == {"v": 2}
    assert os.path.exists(path + ".imported") and os.path.exists(path + ".imported.bak")


def test_a_newer_file_is_imported_when_only_a_numbered_retired_copy_exists():
    root = _home()
    h.save_state("probe.json", {"v": 1})
    path = _probe_sidecar(root, '{"v": 2}')
    _retired(path, ".imported.1", '{"v": 1}')  # `.imported` was renamed away
    _restart()
    assert h.load_state("probe.json") == {"v": 2} and os.path.exists(path + ".imported.2")


def test_a_stale_sidecar_never_overwrites_the_lake_record_it_was_imported_into():
    """The other side of R3-M3: a file left under its own name by a pre-2026-10-24 import (no
    retired copy beside it) holds OLDER bytes than the lake; it is renamed unread."""
    root = _home()
    h.save_state("probe.json", {"v": 2})  # the lake learned on after importing the file
    path = _probe_sidecar(root, '{"v": 1}')
    _restart()
    assert h.load_state("probe.json") == {"v": 2}
    assert os.path.exists(path + ".imported") and not os.path.exists(path)


def test_the_selfbuild_sidecar_in_the_home_is_imported_once_by_the_one_migration():
    """T120: `<home>/selfbuild_state.json` restores the self-build learner through
    `home._migrate_sidecar` (its `_T_STATE` record names the file), is retired `.imported`, and a
    lake reset afterwards cold-starts instead of importing the Aug-2026 weights again."""
    from snn_data_lake.infra.brain_native_selfbuild import _ALL_IDS
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    tid = _ALL_IDS[0]
    path = _probe_sidecar(_home(), json.dumps({"futile": 3, "weights": {tid: 0.77}}), _SB)
    p = P(sources=[])
    assert p._futile == 3 and p.weights()[tid] == 0.77
    assert h.load_named_state(_SB, path=h.knowledge_path()) is not None
    assert h.load_selfbuild() is not None
    assert os.path.exists(path + ".imported") and not os.path.exists(path)
    _restart()
    assert P(sources=[])._futile == 3  # the next boot reads the lake
    _restart()
    os.remove(h.knowledge_path())  # a lake reset
    _restart()
    assert P(sources=[])._futile == 0


def test_a_selfbuild_sidecar_moved_beside_a_learned_lake_is_imported_but_not_applied():
    """The operator's case: the lake already holds a _T_SELFBUILD record (the pre-T120 migration
    copied the repo file into it), so the moved file is older than the lake. The first boot imports
    it as a frozen _T_STATE record and renames it `.imported`. The _T_SELFBUILD record wins, and the
    rename means a lake reset cannot import the file again."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    _home()
    h.save_selfbuild(b'{"futile": 5}')
    path = _probe_sidecar(h.home(), '{"futile": 3}', _SB)
    assert P(sources=[])._futile == 5
    assert os.path.exists(path + ".imported") and not os.path.exists(path)
    assert h.load_named_state(_SB, path=h.knowledge_path()) == b'{"futile": 3}'


def test_a_learned_lake_with_no_sidecar_never_rereads_the_frozen_copy(monkeypatch):
    """rv-arc3 A-m8 / B-m1: once the lake holds a _T_SELFBUILD record and no file sits in the home,
    a boot does not read the frozen _T_STATE copy at all."""
    from snn_data_lake.infra import brain_native_selfbuild as m

    _home()
    h.save_selfbuild(b'{"futile": 5}')
    calls = []
    monkeypatch.setattr(m, "load_state", lambda *a, **k: calls.append(a))
    assert m.BrainNativeSelfBuildProposer(sources=[])._futile == 5
    assert calls == []


def test_an_isolated_selfbuild_lake_leaves_the_home_sidecar_alone(tmp_path):
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    path = _probe_sidecar(_home(), '{"futile": 3}', _SB)
    assert P(sources=[], lake_path=str(tmp_path / "iso.bin"))._futile == 0
    assert os.path.exists(path) and not os.path.exists(path + ".imported")
    assert h.load_selfbuild() is None


@pytest.mark.parametrize("sidecar", [False, True])
def test_a_corrupt_selfbuild_record_cold_starts_without_superseding_it(sidecar):
    """rv-arc3 A-M1: a lake whose only _T_SELFBUILD record does not decode cold-starts, but the cold
    learner never appends a record over it, and a sidecar beside it keeps its name for a later boot
    (importing it now would retire it without applying it)."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    _home()
    h.save_selfbuild(b"\xff")
    path = _probe_sidecar(h.home(), '{"futile": 3}', _SB) if sidecar else None
    p = P(sources=[])
    assert p._futile == 0
    assert ("*", "cold_start", "no self-build record could be read; persistence off") in p.log
    p._save()
    assert h.load_selfbuild() == b"\xff"
    assert path is None or (os.path.exists(path) and not os.path.exists(path + ".imported"))


def test_a_corrupt_newest_selfbuild_record_restores_the_newest_record_that_decodes():
    """rv-arc3 A-M1 / B-m7: bad newest records (a non-object, then undecodable bytes) fall back to
    the learned history before them."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    _home()
    h.save_selfbuild(b'{"futile": 4}')
    h.save_selfbuild(b'{"futile": 5}')
    h.save_selfbuild(b"[1]")
    h.save_selfbuild(b"\xff")
    p = P(sources=[])
    assert p._futile == 5
    p._save()
    assert json.loads(h.load_selfbuild())["futile"] == 5


@pytest.mark.parametrize(
    "records, futile",
    [([b"\xff"], 0), ([b'{"futile": 4}', b'{"futile": 5}', b"\xff"], 5)],
)
def test_an_isolated_selfbuild_lake_with_a_corrupt_newest_record(tmp_path, records, futile):
    """On an isolated lake (`lake_path`): restore the newest record that decodes;
    with none, cold-start and write nothing over the record."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    _home()
    iso = str(tmp_path / "iso.bin")
    for rec in records:
        h.save_selfbuild(rec, iso)
    p = P(sources=[], lake_path=iso)
    assert p._futile == futile
    p._save()
    blob = h.load_selfbuild(iso)
    if futile:
        assert json.loads(blob)["futile"] == futile
    else:
        assert blob == records[-1]


def test_load_selfbuild_skips_only_the_records_its_decoder_rejects():
    """`decode` returning None or raising ValueError skips a record; any other error propagates."""
    _home()
    for rec in (b"good", b"none", b"value"):
        h.save_selfbuild(rec)

    def decode(blob):
        if blob == b"value":
            raise ValueError(blob)
        return None if blob == b"none" else blob.decode("ascii")

    assert h.load_selfbuild(decode=decode) == "good"
    assert h.load_selfbuild() == b"value"

    def broken(blob):
        raise TypeError(blob)

    with pytest.raises(TypeError):
        h.load_selfbuild(decode=broken)


def test_an_unreadable_selfbuild_record_never_restores_the_legacy_state(monkeypatch):
    """rv-arc3 A-M1: a replay that raises leaves both the lake and the sidecar as they were; the
    next boot that can read the lake imports the sidecar."""
    from snn_data_lake.infra import brain_native_selfbuild as m

    path = _probe_sidecar(_home(), '{"futile": 3}', _SB)

    def unreadable(*_a, **_k):
        raise OSError("replay failed")

    monkeypatch.setattr(m, "load_selfbuild", unreadable)
    p = m.BrainNativeSelfBuildProposer(sources=[])
    assert p._futile == 0
    p._save()
    assert h.load_selfbuild() is None
    assert os.path.exists(path) and not os.path.exists(path + ".imported")
    monkeypatch.setattr(m, "load_selfbuild", h.load_selfbuild)
    _restart()
    assert m.BrainNativeSelfBuildProposer(sources=[])._futile == 3
    assert os.path.exists(path + ".imported")


@pytest.mark.parametrize("text", ["[1]", "null", "{bad"])
def test_a_selfbuild_sidecar_that_is_not_a_state_object_cold_starts(text):
    """Imported and retired like any sidecar (`_migrate_sidecar` validates nothing), never parsed
    into the learner and never copied into _T_SELFBUILD."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    path = _probe_sidecar(_home(), text, _SB)
    assert P(sources=[])._futile == 0
    assert h.load_selfbuild() is None and os.path.exists(path + ".imported")


def test_a_repo_selfbuild_sidecar_is_never_read(tmp_path):
    """T120 (rv-arc2-r3 m8): `<repo>/data/selfbuild_state.json` is not a home's file; a temp-home
    learner never imports it, and it stays untouched for the operator to move."""
    from snn_data_lake.infra.brain_native_selfbuild import BrainNativeSelfBuildProposer as P

    _home()
    legacy = tmp_path / "data" / "selfbuild_state.json"
    legacy.parent.mkdir()
    legacy.write_text('{"futile": 3}', encoding="ascii")
    assert P(repo=str(tmp_path), sources=[])._futile == 0
    assert h.load_selfbuild() is None and legacy.read_text(encoding="ascii") == '{"futile": 3}'
    assert not list(legacy.parent.glob("*.imported*"))


def test_one_function_migrates_legacy_sidecars():
    """P2 census: exactly one legacy-sidecar migration survives in src/, `home._migrate_sidecar`."""
    import re

    src = os.path.join(os.path.dirname(h.__file__), os.pardir)
    defs = []
    for d, _, files in os.walk(src):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    text = fh.read()
                defs += [
                    (f, m) for m in re.findall(r"^\s*def (_?migrate\w*sidecar\w*)\(", text, re.M)
                ]
    assert defs == [("home.py", "_migrate_sidecar")]


def test_a_held_sidecar_keeps_its_name_and_is_retired_on_a_later_read(monkeypatch):
    """Windows: a file another process holds open cannot be renamed; the import still lands, and
    the rename is retried once the file is free."""
    root = _home()
    path = _probe_sidecar(root)

    def held(*_a):
        raise PermissionError("held")

    with monkeypatch.context() as m:  # scoped: conftest's NORD_HOME lives on the same fixture
        m.setattr(h.os, "replace", held)
        assert h.load_state("probe.json") == {"v": 1} and os.path.exists(path)
        assert h.load_named_state("probe.json", path=h.knowledge_path()) == b'{"v": 1}'
    assert h.load_state("probe.json") == {"v": 1}
    assert os.path.exists(path + ".imported") and not os.path.exists(path)


def test_a_sidecar_whose_import_did_not_land_keeps_its_name(monkeypatch):
    """The rename follows the save: a failed import leaves the file for the next read."""
    root = _home()
    path = _probe_sidecar(root)
    real = h.save_named_state
    monkeypatch.setattr(h, "save_named_state", lambda *a, **k: False)
    h.load_state("probe.json")
    assert os.path.exists(path) and not os.path.exists(path + ".imported")
    monkeypatch.setattr(h, "save_named_state", real)
    _restart()
    assert h.load_state("probe.json") == {"v": 1} and os.path.exists(path + ".imported")


# A writer or a reader in a process of its own (B MINOR-1: P4b's "it persists" is a claim about a
# NEW process). argv: home, mode.
_PROC = r"""
import hashlib, json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
import numpy as np
from snn_data_lake.faculties.language_acquisition import LanguageAcquisition
from snn_data_lake.sleep import nightly_skills as ns
from snn_data_lake.substrate.home import save_state
if sys.argv[2] == "write":
    lm = LanguageAcquisition(seed=0)
    lm.W = np.random.RandomState(3).randn(*lm.W.shape); lm._ctx_ids = [1, 2, 3]; lm.n_seen = 41
    lm.save()
    ns._save_arrays(ns._skill_state("math"), W=np.arange(6.0).reshape(2, 3))
    save_state(ns._STATE_JSON, {"domains": {"math": {}}})
    print("PROC true")
else:
    lm = LanguageAcquisition(seed=9).load()
    t = ns.NightlySkills(seed=0)
    lm_digest = hashlib.sha256(lm.W.tobytes()).hexdigest()
    print("PROC " + json.dumps([lm_digest, list(map(int, lm._ctx_ids)), lm.n_seen,
                                t._pending["math"]["W"].tolist()]))
"""


@pytest.mark.slow
@pytest.mark.pool
def test_a_new_process_reads_the_language_and_skills_records_back():
    """The writer process is gone before the reader starts; both see only the lake on disk."""
    from tests.test_p4b_restart_restore import _run

    root = _home()
    assert _run(_PROC, root, "PROC", "write") is True
    assert _sidecars(root) == []
    lm = LanguageAcquisition(seed=0)
    w = np.random.RandomState(3).randn(*lm.W.shape)
    assert _run(_PROC, root, "PROC", "read") == [
        hashlib.sha256(w.tobytes()).hexdigest(),
        [1, 2, 3],
        41,
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
    ]


def test_unchanged_skills_are_not_rewritten(monkeypatch):
    root = _home()
    s = _built(monkeypatch, root)
    s.save()
    size = os.path.getsize(h.knowledge_path())
    s.save()
    assert os.path.getsize(h.knowledge_path()) == size


def _rollback_export_command():
    """The export command docs/RUNNING.md gives the operator before a rollback, read from the
    doc so the two cannot drift."""
    import re

    doc = os.path.join(os.path.dirname(__file__), "..", "docs", "RUNNING.md")
    with open(doc, encoding="utf-8") as f:
        pat = r'`python -c "(import os; from snn_data_lake\.substrate import home as h;[^"]*)"`'
        m = re.search(pat, f.read())
    assert m, "docs/RUNNING.md lost its rollback export command"
    return m.group(1)


@pytest.mark.parametrize("route, expected", [("copy_back", 10), ("export", 11)])
def test_the_documented_rollback_export_keeps_what_arc2_learned(route, expected):
    """rv-arc2-r3 R3D-M1: after the first rename arc 2 saves the npz organs only to the lake.
    A rollback that starts from the highest `.imported` copy (v=0) and learns in its window
    (+10) is imported back OVER the lake's arc-2 record (v=1): 10, the arc-2 learning lost.
    The doc's export starts the window from the lake's record instead: 11, nothing lost."""
    from tests.test_p4b_restart_restore import _run

    root = _home()
    names = (STATE_NAME, "skills/demo.npz")

    def blob(v):
        b = io.BytesIO()
        np.savez(b, v=np.array([v]))
        return b.getvalue()

    def v_of(raw):
        return int(np.load(io.BytesIO(raw))["v"][0])

    for n in names:  # a pre-arc-2 sidecar, imported and renamed by the first arc-2 read
        os.makedirs(os.path.dirname(h.state_path(n)), exist_ok=True)
        with open(h.state_path(n), "wb") as f:
            f.write(blob(0))
    _restart()
    for n in names:
        assert v_of(h.load_named_state(n)) == 0 and not os.path.exists(h.state_path(n))
        assert h.save_named_state(n, blob(1))  # arc 2 learns, lake only
    _restart()
    if route == "export":
        assert _run(_rollback_export_command() + "; print('EXPORT true')", root, "EXPORT") is True
    else:
        for n in names:
            shutil.copy2(h.state_path(n) + ".imported", h.state_path(n))
    for n in names:  # the older checkout reads the FILE and learns in the rollback window
        with open(h.state_path(n), "rb") as f:
            base = v_of(f.read())
        with open(h.state_path(n), "wb") as f:
            f.write(blob(base + 10))
    _restart()  # back on arc 2
    assert [v_of(h.load_named_state(n)) for n in names] == [expected, expected]


def test_the_rollback_export_never_overwrites_an_unread_window_file():
    """rv-arc2-r3 d2-m1: a skills domain no arc-2 night read since the last rollback still holds
    that window's learning in its file; a second rollback's export must not write the older lake
    record over it."""
    from tests.test_p4b_restart_restore import _run

    root = _home()
    n = "skills/demo.npz"
    os.makedirs(os.path.dirname(h.state_path(n)), exist_ok=True)
    assert h.save_named_state(n, b"PK-lake-record")  # the lake's record (bytes only compared here)
    with open(h.state_path(n), "wb") as f:
        f.write(b"PK-window-1")  # the first window's learning, never read back by arc 2
    _restart()
    assert _run(_rollback_export_command() + "; print('EXPORT true')", root, "EXPORT") is True
    with open(h.state_path(n), "rb") as f:
        assert f.read() == b"PK-window-1"
