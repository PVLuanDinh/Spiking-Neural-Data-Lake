"""Deep unrounded object-graph fingerprint of a live Brain.

WHY THIS EXISTS. `Brain.state()` rounds to 3-6 decimals and returns ~114 scalar leaves. It is
blind to anything smaller than its rounding, and it does not reach most of the object graph. Every
NULL result measured through it is suspect, and two load-bearing ones in this repo were wrong:

  specs/007 SC-001 "growth changes 0 non-tautological fields"
      state()          : 5 of 114 differ, all tautological  -> reported as a null
      deep fingerprint : 83 of 127,818 differ at seed 0, including brain.brain.arm (13 leaves),
                         brain._drive_w (10) and brain._task_set._shifter (2)
      A/A control      : 0 of 127,833 on every seed measured
      -> the null was FALSE. Growth reaches the arm and the drive weights.

  P6 "winner identity carries zero information downstream"
      state()          : 0 of 114
      deep fingerprint : 0 of 127,833, max abs divergence 0.000000e+00
      -> the null HOLDS. Confirmed, not overturned.

That contrast is the instrument's own validation: it finds 83 where there is an effect and exactly
0 where there is not, so it is discriminating rather than merely noise-sensitive.

USE. Always with a matched A/A control. A nonzero A/A means the harness has run nondeterminism and
the A/B through it is unreadable -- that check is not optional, it is the reason this file exists.

    python tools/deep_fingerprint.py --demo      # A/A control on a default brain

Run: python tools/deep_fingerprint.py
"""

import math
import os
import sys

#: Leaves whose path contains any of these belong to a growth mechanism's own bookkeeping and
#: cannot serve as evidence that growth CHANGED anything (specs/007 contracts C3).
TAUTOLOGICAL = (
    "time_cells_grown",
    "time_cells_refused",
    "time_cells_width",
    "elapsed_time",
    "timing_sd",
    "_time_clock",
)

_MAX_SEQ = 4000


def walk(obj, path="brain", out=None, seen=None, depth=0, max_depth=14):
    """Every numeric leaf reachable from `obj`, unrounded, keyed by its path."""
    if out is None:
        out = {}
    if seen is None:
        seen = set()
    if depth > max_depth:
        return out
    if isinstance(obj, bool):
        out[path] = int(obj)
        return out
    if isinstance(obj, int):
        out[path] = obj
        return out
    if isinstance(obj, float):
        out[path] = obj if not math.isnan(obj) else "nan"
        return out
    if obj is None or isinstance(obj, (str, bytes)):
        return out
    oid = id(obj)
    if oid in seen:
        return out
    seen.add(oid)
    try:
        if isinstance(obj, dict):
            for k in list(obj)[:_MAX_SEQ]:
                if isinstance(k, (str, int, float, bool)):
                    walk(obj[k], f"{path}[{k!r}]", out, seen, depth + 1, max_depth)
            return out
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:_MAX_SEQ]):
                walk(v, f"{path}[{i}]", out, seen, depth + 1, max_depth)
            return out
        if hasattr(obj, "shape") and hasattr(obj, "ravel"):
            for i, v in enumerate(obj.ravel()[:_MAX_SEQ]):
                out[f"{path}#{i}"] = float(v)
            return out
        d = getattr(obj, "__dict__", None)
        if d:
            for k in sorted(d):
                if k.startswith("__"):
                    continue
                v = d[k]
                if callable(v) and not hasattr(v, "__dict__"):
                    continue
                walk(v, f"{path}.{k}", out, seen, depth + 1, max_depth)
    except Exception:
        pass  # a walker that raises on one odd attribute must not lose the other 127,000 leaves
    return out


def diff(a, b):
    """(n_differ, n_common, differing_keys, max_abs_divergence, n_integer_leaves_moved)."""
    keys = set(a) & set(b)
    dif, maxabs, nint = [], 0.0, 0
    for k in keys:
        x, y = a[k], b[k]
        if x != y:
            dif.append(k)
            if isinstance(x, int) and isinstance(y, int):
                nint += 1
            else:
                try:
                    maxabs = max(maxabs, abs(float(x) - float(y)))
                except (TypeError, ValueError):
                    pass
    return len(dif), len(keys), dif, maxabs, nint


def non_tautological(keys):
    """The differing leaves that are NOT the manipulated mechanism's own bookkeeping."""
    return [k for k in keys if not any(t in k for t in TAUTOLOGICAL)]


def _selfcheck():
    print("DEEP FINGERPRINT self-check")
    obj = {"a": 1, "b": [1.0, 2.0], "c": {"d": True}}
    fp = walk(obj, "root")
    assert len(fp) == 4, fp
    n, c, _, _, _ = diff(fp, fp)
    assert (n, c) == (0, 4), (n, c)
    fp2 = walk({"a": 1, "b": [1.0, 2.5], "c": {"d": True}}, "root")
    n, c, k, mx, ni = diff(fp, fp2)
    assert n == 1 and abs(mx - 0.5) < 1e-12, (n, mx)

    # a cycle must terminate rather than recurse forever
    cyc = {}
    cyc["self"] = cyc
    cyc["v"] = 3
    assert walk(cyc, "c") == {"c['v']": 3}

    # the tautology filter must actually remove growth bookkeeping
    assert non_tautological(["brain._time_clock.n_cells", "brain.brain.arm.x"]) == [
        "brain.brain.arm.x"
    ]
    print("  leaves walked, diff, cycle guard, tautology filter: OK")
    print("self-check OK: 4 of 4 properties hold")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        # Derived from __file__, not the literal "src": a relative sys.path entry resolves against
        # the cwd, so --demo worked only from the repo root and imported nothing anywhere else.
        _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from snn_data_lake.cortex.brain import make_brain

        def run():
            b = make_brain(seed=0, persist=False)
            for _ in range(50):
                b.tick()
            fp = walk(b)
            b.close()
            return fp

        a, b_ = run(), run()
        n, c, _, mx, _ = diff(a, b_)
        print(f"A/A control: {n} of {c:,} leaves differ, max abs {mx:.3e}")
        print(
            "A nonzero A/A means run nondeterminism -- every A/B through this harness is unreadable."
        )
    else:
        _selfcheck()
