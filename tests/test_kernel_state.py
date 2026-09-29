"""KernelState (specs/022 T013, FR-035) -- the SoA layout contracts, each with its red.

128-byte alignment (the paging contract: one activation hop = one aligned block read),
contiguity (a faculty is one O(1) slice per field, never a gather), lake persistence through
the sanctioned named-state route with latest-wins, and whole-frame refusal on anything the
parser cannot fully account for.
"""

import numpy as np
import pytest

from snn_data_lake.reservoir.kernel_state import _ALIGN, _FRAME, FIELDS, KernelState


def _ks():
    return KernelState([("cortex", 7), ("amygdala", 4), ("thalamus", 5)])


def test_every_field_block_is_128_byte_aligned():
    ks = _ks()
    for field, _dt in FIELDS:
        assert ks.arrays[field].ctypes.data % _ALIGN == 0, f"{field} block misaligned"


def test_faculty_slices_are_contiguous_views():
    ks = _ks()
    assert ks.ranges == {"cortex": (0, 7), "amygdala": (7, 11), "thalamus": (11, 16)}
    view = ks.slice("amygdala", "v")
    view[:] = 3.5
    assert np.all(ks.arrays["v"][7:11] == 3.5), "the slice must be a VIEW, not a copy"
    assert np.all(ks.arrays["v"][:7] == 0.0) and np.all(ks.arrays["v"][11:] == 0.0), (
        "a faculty write leaked outside its contiguous range"
    )


def test_layout_registry_refusals():
    with pytest.raises(ValueError):
        KernelState([])
    with pytest.raises(ValueError):
        KernelState([("cortex", 0)])
    with pytest.raises(ValueError):
        KernelState([("cortex", 3), ("cortex", 2)])


def test_lake_round_trip_latest_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    ks = _ks()
    ks.slice("cortex", "v")[:] = np.arange(7, dtype=np.float32)
    ks.slice("thalamus", "sign")[:] = -1
    assert ks.save()
    ks.slice("cortex", "v")[:] += 100.0  # save again: latest-wins per name must return THIS one
    assert ks.save()
    back = KernelState.load()
    assert back is not None and back.ranges == ks.ranges
    for field, _dt in FIELDS:
        assert np.array_equal(back.arrays[field], ks.arrays[field]), f"{field} diverged"


def test_cold_start_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    assert KernelState.load() is None, "an empty lake must cold-start, never invent state"


def test_frames_refuse_whole():
    """The foreign frame is CONSTRUCTED, not byte-patched: the first draft used bytes.replace
    on the UTF-8 layout string, which occurs nowhere in an npy str array (UCS-4), so the
    control never mutated -- the module's own self-check caught it on the first run."""
    import io
    import json

    ks = _ks()
    blob = ks.to_blob()
    assert KernelState.from_blob(blob[: len(blob) - 40]) is None, "truncated frame restored"
    fbuf = io.BytesIO()
    np.savez(
        fbuf,
        layout=np.array("kernel_state.v9"),
        faculties=np.array(json.dumps([[k, *v] for k, v in ks.ranges.items()])),
        **ks.arrays,
    )
    foreign = fbuf.getvalue()
    assert foreign != blob, "the foreign-frame control did not differ from the real frame"
    assert _FRAME == "kernel_state.v1", "test premise: the live layout name moved; update me"
    assert KernelState.from_blob(foreign) is None, (
        "a foreign layout name restored -- the version lives in the name so latest-wins can "
        "never silently mix layouts, and this is the assert that makes that a mechanism"
    )
    assert KernelState.from_blob(b"") is None and KernelState.from_blob(b"junk") is None
