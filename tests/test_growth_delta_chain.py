"""The growth delta chain (_T_GROWTH_DELTA) -- the reconstruction proof that flips `reclaimable`.

specs/022 T010-ii. The measured pathology: 97.64% of the live lake was repeated whole-substrate
_T_GROWTH snapshots (83,405 records) whose bytes differed only in their `ticks`/`acc` members, so
BinLog's byte dedup never fired and nothing could ever reclaim them. The chain kills that at the
door: an npz save decomposes into .npy members and lands as a FULL record or a DELTA carrying only
the changed members plus a hash manifest of the rest.

WHAT THIS FILE PROVES (each is cited by `lake_engine.RETENTION`'s reclaimable=True rows and by
POLICIES.md's retention rule -- weakening an assert here weakens the licence those flags claim):
  * round-trip: chain replay reconstructs the newest substrate exactly, through real np.savez blobs
    and a real np.load on the way back out;
  * the pathology red: a run of saves differing only in `ticks` costs a small fraction of what
    whole-blob snapshots cost;
  * the tamper red: one flipped byte in a delta's member data makes the reader REFUSE that whole
    link and hold the last verified state -- never the tampered bytes, never a half-applied record;
  * the FULL cadence bounds replay; latest-wins still holds across classes.

The restart half of the proof (a real interpreter boundary) lives in
tests/test_brain_growth_restart.py, whose counters now assert the cutover too.
"""

import io

import numpy as np

from snn_data_lake.substrate import home as H


def _use_home(monkeypatch, tmp_path):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    # Fresh instruments per test: the BinLog cache and the chain-tail cache are keyed by resolved
    # path, but clearing them makes each test's first load a genuine cold replay from disk.
    H._LOG_CACHE.clear()
    H._GROWTH_CHAIN.clear()


def _blob(w, ticks, acc=0.5):
    """A real np.savez blob shaped like the engine's (subset of keys; the chain is key-agnostic)."""
    buf = io.BytesIO()
    np.savez(
        buf,
        fmt=np.array("ngsnn.growth.v1"),
        W=np.asarray(w, dtype=np.float32),
        ticks=np.array(int(ticks)),
        acc=np.array(float(acc)),
    )
    return buf.getvalue()


def _arrays(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _chain_payloads():
    return [p[1:] for p in H.BinLog(H.knowledge_path()).replay() if p[:1] == H._T_GROWTH_DELTA]


def test_full_then_delta_round_trip(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)
    v1 = _blob(np.zeros(64), ticks=1)
    v2 = _blob(np.zeros(64), ticks=2)  # only `ticks` moved -- the 97.64% case
    assert H.save_growth(v1), "first npz save must write (FULL)"
    assert H.save_growth(v2), "changed member must write (DELTA)"

    got = _arrays(H.load_growth())
    want = _arrays(v2)
    assert set(got) == set(want)
    for k in want:
        assert np.array_equal(got[k], want[k]), f"member {k} did not round-trip"

    recs = _chain_payloads()
    assert len(recs) == 2
    k0, _ = H._growth_unpack(recs[0])
    k1, e1 = H._growth_unpack(recs[1])
    assert (k0, k1) == (H._GC_FULL, H._GC_DELTA)
    carried = [n for n, _h, d in e1 if d is not None]
    assert carried == ["ticks.npy"], f"the DELTA must carry ONLY the changed member, got {carried}"


def test_byte_identical_substrate_writes_nothing(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)
    v1 = _blob(np.ones(8), ticks=1)
    assert H.save_growth(v1)
    assert H.save_growth(v1) == 0, "an unchanged substrate must write NOTHING"
    assert len(_chain_payloads()) == 1


def test_tamper_holds_the_last_verified_state(monkeypatch, tmp_path):
    """One flipped byte in a delta's member data -> that whole link REFUSES; the reader returns the
    last verified state, never the tampered bytes. This is the red `reclaimable=True` cites."""
    _use_home(monkeypatch, tmp_path)
    w1 = np.zeros(64, dtype=np.float32)
    w2 = np.arange(64, dtype=np.float32) + 1000.0  # distinctive bytes to find on disk
    v1, v2 = _blob(w1, ticks=1), _blob(w2, ticks=2)
    assert H.save_growth(v1) and H.save_growth(v2)

    # POSITIVE CONTROL before the tamper: the probe sees v2, so a v1 answer after the flip is the
    # refusal working and not a reader that was blind all along.
    assert np.array_equal(_arrays(H.load_growth())["W"], w2)

    lake = H.knowledge_path()
    raw = open(lake, "rb").read()
    needle = w2.tobytes()[16:48]  # a slice of v2's W payload; appears only in the DELTA record
    at = raw.find(needle)
    assert at != -1, "test instrument broke: v2's W bytes not found in the lake file"
    assert raw.find(needle, at + 1) == -1, "needle not unique; the flip below would be ambiguous"
    tampered = bytearray(raw)
    tampered[at] ^= 0xFF
    open(lake, "wb").write(bytes(tampered))

    H._LOG_CACHE.clear()  # same file size after an in-place flip; force a cold re-read
    H._GROWTH_CHAIN.clear()
    got = _arrays(H.load_growth())
    assert np.array_equal(got["W"], w1), "a tampered link must fall back to the last VERIFIED state"
    assert int(got["ticks"]) == 1, (
        "the tampered DELTA was half-applied: its untouched `ticks` member landed while its W was "
        "refused -- a chain record must apply whole or not at all"
    )


def test_full_cadence_bounds_replay(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)
    monkeypatch.setattr(H, "_GROWTH_CHAIN_EVERY", 3)
    for t in range(1, 7):
        assert H.save_growth(_blob(np.zeros(8), ticks=t))
    kinds = [H._growth_unpack(p)[0] for p in _chain_payloads()]
    assert kinds == [H._GC_FULL, H._GC_DELTA, H._GC_DELTA, H._GC_DELTA, H._GC_FULL, H._GC_DELTA], (
        f"FULL cadence at every 3 deltas not honoured: {kinds}"
    )
    assert int(_arrays(H.load_growth())["ticks"]) == 6


def test_pathology_red_repeated_snapshots_shrink(monkeypatch, tmp_path):
    """The constructed 97.64% case: 50 saves where only `ticks` moves. Whole-blob snapshots would
    cost 50x the blob; the chain must cost a small fraction of that."""
    _use_home(monkeypatch, tmp_path)
    w = np.zeros(4096, dtype=np.float32)  # a substrate big enough that the ratio means something
    blob_len = len(_blob(w, ticks=1))
    for t in range(1, 51):
        H.save_growth(_blob(w, ticks=t))
    chain_bytes = sum(len(p) for p in _chain_payloads()) + len(_chain_payloads())  # + tag bytes
    whole_blob_bytes = 50 * blob_len
    assert chain_bytes < 0.15 * whole_blob_bytes, (
        f"the chain did not kill the snapshot pathology: {chain_bytes:,}B for 50 saves vs "
        f"{whole_blob_bytes:,}B as whole blobs"
    )
    assert int(_arrays(H.load_growth())["ticks"]) == 50  # cheap is worthless if it forgets


def test_latest_wins_across_classes(monkeypatch, tmp_path):
    """A raw (non-npz) record still takes the legacy tag and still shadows by position -- the
    contract the legacy tag always had -- and the chain reclaims the lead on the next npz save."""
    _use_home(monkeypatch, tmp_path)
    assert H.save_growth(_blob(np.ones(8), ticks=1))
    assert H.save_growth(b"raw-legacy-after-chain")
    assert H.load_growth() == b"raw-legacy-after-chain", "a later legacy record must win"
    assert H.save_growth(_blob(np.ones(8), ticks=2))
    assert int(_arrays(H.load_growth())["ticks"]) == 2, "a later chain record must win back"
