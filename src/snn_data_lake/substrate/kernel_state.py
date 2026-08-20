"""KernelState -- substrate state as struct-of-arrays, outside Python objects (specs/022 T013,
FR-035): contiguous within-faculty indices, 128-byte-aligned field blocks, lake-backed.

WHY SoA, stated against the incumbents it will supersede: today every per-unit quantity on a
hot path lives as a Python attribute on a model object (`cortex/lif_neuron.py`,
`cortex/alif_neuron.py`, `cortex/adex_neuron.py`, `cortex/izhikevich_neuron.py`,
`cortex/resonate_and_fire.py`) or inside `cortex/neuron_zoo.py`'s heterogeneous population --
per-object storage that costs an attribute lookup per unit per tick and can never page, vectorize
or land on a GPU as a block. The scale ceiling is ARCHITECTURAL (~40k units, measured); a
lake-resident brain needs state that is bytes first.

P2 DISPOSITION, DECLARED AT INTRODUCTION (executed at the kernel cutover, bound by S0 exit):
this class becomes the ONE owner of per-unit hot-path STATE. The neuron-model modules above
SURVIVE as equation owners -- their dynamics functions run over SoA slices -- and their
per-instance attribute state (v/w/u/thr and friends) RETIRES at the T014+ cutover, faculty by
faculty, each with its own parity A/B. Nothing is retired in THIS commit: the kernel is not yet
the live path, and deleting a mechanism before its successor carries traffic is how capability
vanishes (P2's port-first guard).

THE S0 FIELD SET IS THE EQUATION OWNERS', NOT INVENTED HERE (P1): v (membrane), thr (adaptive
threshold), refrac (refractory ticks), last_spike (tick index), sign (Dale, immutable at birth --
`cortex/placement.py` owns the assignment rule) -- exactly what the LIF/ALIF pair needs, cited as
the S0 incumbents. A richer model (AdEx's w, Izhikevich's u) adds its field WHEN it adopts the
kernel, and every layout change is a NEW named frame with its own migration disposition (§3.5;
the `.v1` in the frame name is that mechanism).

LAYOUT CONTRACTS, each asserted in main():
  * 128-byte alignment: every field block starts on a 128-byte boundary (one activation hop =
    one aligned block read -- the paging contract from data-model.md).
  * Contiguity within faculty: a faculty's units are one [start, stop) slice per field; slicing
    a faculty is an O(1) view, never a gather.
  * Persistence is THE named-state route (REUSE, P4b): one `_T_STATE` frame named
    `kernel_state.v1` via home.save_named_state -- latest-wins per name, pruned by the lake's
    own consolidation, cold start when the name is absent. crc32-for-pinned-values stays the
    project convention (`substrate/home.py` owns it); this module pins nothing itself.

distinct-from: cortex/neuron_zoo.py -- a POPULATION of Python model objects behind one
  interface; this is the byte layout that will hold such populations' state. distinct-from:
  cortex/neurogenesis_engine.py -- grows/retains the NGSNN readout substrate (torch tensors,
  its own _T_GROWTH_DELTA chain); different substrate, different persistence class.
  distinct-from: substrate/growable.py -- growth bookkeeping, not state layout. All sites read.
"""

from __future__ import annotations

import io
import json
import sys

_ALIGN = 128
_FRAME = "kernel_state.v1"  # the _T_STATE name; a layout change ships kernel_state.v2 + its
# own migration disposition (§3.5) -- the version lives in the NAME so latest-wins per name
# can never silently mix layouts.

#: (field, dtype-string) -- the S0 equation owners' state set (LIF/ALIF; see module docstring).
FIELDS = (
    ("v", "<f4"),
    ("thr", "<f4"),
    ("refrac", "<i4"),
    ("last_spike", "<i8"),
    ("sign", "|i1"),
)


def _aligned_zeros(n, dtype):
    """A zeroed 1-D array of `n` items whose data pointer is 128-byte aligned. numpy's own
    allocator gives 16/64 depending on build; over-allocate and slice to the boundary (the
    base buffer rides along as the view's .base, so the alignment survives GC)."""
    import numpy as np

    itemsize = np.dtype(dtype).itemsize
    raw = np.zeros(n * itemsize + _ALIGN, dtype=np.uint8)
    off = (-raw.ctypes.data) % _ALIGN
    return raw[off : off + n * itemsize].view(dtype)


class KernelState:
    """SoA state for `n_units` units across contiguously-indexed faculties."""

    def __init__(self, faculties):
        """faculties: ordered [(name, n_units)]. Index space is their concatenation, so every
        faculty is one contiguous [start, stop) slice -- the layout IS the registry."""
        import numpy as np

        if not faculties:
            raise ValueError("a kernel with zero faculties holds nothing to lay out")
        self.ranges: dict[str, tuple[int, int]] = {}
        at = 0
        for name, n in faculties:
            name, n = str(name), int(n)
            if n <= 0:
                raise ValueError(f"faculty {name!r} declares {n} units; layout refuses <= 0")
            if name in self.ranges:
                raise ValueError(f"faculty {name!r} declared twice; one slice per faculty")
            self.ranges[name] = (at, at + n)
            at += n
        self.n_units = at
        self.arrays = {f: _aligned_zeros(at, dt) for f, dt in FIELDS}
        assert all(a.ctypes.data % _ALIGN == 0 for a in self.arrays.values())
        self._np = np

    def slice(self, faculty, field):
        """The faculty's contiguous view of one field -- O(1), never a copy."""
        lo, hi = self.ranges[faculty]
        return self.arrays[field][lo:hi]

    # ---- persistence: ONE _T_STATE frame, the sanctioned named-state route (P4b) ----
    def to_blob(self):
        import numpy as np

        buf = io.BytesIO()
        np.savez(
            buf,
            layout=np.array(_FRAME),
            faculties=np.array(json.dumps([[k, *v] for k, v in self.ranges.items()])),
            **self.arrays,
        )
        return buf.getvalue()

    def save(self, path=None):
        from snn_data_lake.substrate.home import save_named_state

        return save_named_state(_FRAME, self.to_blob(), path)

    @classmethod
    def from_blob(cls, blob):
        """Rebuild from a frame, or None on ANY refusal -- wrong layout name, missing field,
        range/size disagreement. Refuses whole rather than half-restores (the same rule every
        lake parser here follows): a kernel with a stranger's `v` beside its own `thr` would
        integrate a substrate nobody ever held."""
        import numpy as np

        try:
            with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                if str(z["layout"]) != _FRAME:
                    return None
                ranges = json.loads(str(z["faculties"]))
                arrays = {f: z[f].copy() for f, _dt in FIELDS}
        except Exception:
            return None
        state = cls([(name, hi - lo) for name, lo, hi in ranges])
        if any(state.ranges[name] != (lo, hi) for name, lo, hi in ranges):
            return None
        for f, _dt in FIELDS:
            if arrays[f].shape != (state.n_units,):
                return None
            state.arrays[f][:] = arrays[f]
        return state

    @classmethod
    def load(cls, path=None):
        from snn_data_lake.substrate.home import load_named_state

        blob = load_named_state(_FRAME, path)
        return cls.from_blob(blob) if blob else None


def main() -> int:
    import os
    import tempfile

    import numpy as np

    prior = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = tempfile.mkdtemp(prefix="kernel_state_")
    try:
        ks = KernelState([("cortex", 5), ("amygdala", 3), ("thalamus", 2)])
        # alignment: every field block on a 128-byte boundary
        misaligned = [f for f, a in ks.arrays.items() if a.ctypes.data % _ALIGN]
        if misaligned:
            print(f"kernel_state self-check FAILED: misaligned blocks {misaligned}")
            return 1
        # contiguity: slices are O(1) VIEWS over adjacent ranges covering the whole space
        if ks.ranges != {"cortex": (0, 5), "amygdala": (5, 8), "thalamus": (8, 10)}:
            print(f"kernel_state self-check FAILED: ranges not contiguous ({ks.ranges})")
            return 1
        if ks.slice("amygdala", "v").base is not ks.arrays["v"].base:
            print("kernel_state self-check FAILED: a faculty slice copied instead of viewing")
            return 1
        # round-trip through a REAL scratch lake, with a mutation control in the same run
        ks.slice("cortex", "v")[:] = np.arange(5, dtype=np.float32)
        ks.slice("amygdala", "sign")[:] = -1
        ks.arrays["last_spike"][:] = 42
        assert ks.save()
        back = KernelState.load()
        if back is None or any(
            not np.array_equal(back.arrays[f], ks.arrays[f]) for f, _dt in FIELDS
        ):
            print("kernel_state self-check FAILED: lake round-trip diverged")
            return 1
        # refusal reds: wrong layout name; truncated frame. The first draft mutated the blob
        # with bytes.replace on the UTF-8 layout string -- which occurs NOWHERE in the frame
        # (npy stores str arrays as UCS-4), so the control never mutated and the unchanged
        # blob "restored". Caught by this very check on its first run; the foreign frame is
        # now CONSTRUCTED, and its difference from the real one is asserted, not assumed.
        blob = ks.to_blob()
        if KernelState.from_blob(blob[: len(blob) // 2]) is not None:
            print("kernel_state self-check FAILED: a truncated frame restored")
            return 1
        fbuf = io.BytesIO()
        np.savez(
            fbuf,
            layout=np.array("kernel_state.v9"),
            faculties=np.array(json.dumps([[k, *v] for k, v in ks.ranges.items()])),
            **ks.arrays,
        )
        wrong = fbuf.getvalue()
        if wrong == blob:
            print("kernel_state self-check FAILED: the foreign-frame control did not differ")
            return 1
        if KernelState.from_blob(wrong) is not None:
            print("kernel_state self-check FAILED: a foreign layout name restored")
            return 1
        print(
            f"kernel_state self-check OK: {len(FIELDS)} fields x {ks.n_units} units, every "
            f"block {_ALIGN}-byte aligned; faculty slices are contiguous views; _T_STATE "
            f"'{_FRAME}' round-trip byte-equal through a scratch lake; truncated and "
            "foreign-layout frames both REFUSED whole"
        )
        return 0
    finally:
        if prior is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prior


if __name__ == "__main__":
    sys.exit(main())
