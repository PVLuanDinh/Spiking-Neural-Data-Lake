# Restart probe: four live learners with no lake path (2026-10-28)

**Status.** A finding only (owner decision 5, approved in the owner's reply of 2026-10-28). No code was changed and
nothing is proposed here: any fix is its own pre-registered arc, by the integrator.

**Tree and host.**
- Tree: `c1df3887` (origin/main), branch `cloud/restart-probe-1028`.
- Host: a cloud Linux container; CPython 3.14.6 free-threaded; torch 2.14.1+cu130.
- Perception tier `stdlib`: the e-prop encoders are gitignored and absent here.
- Every brain is `make_brain(seed, persist=True)` on a scratch `NORD_HOME` (a fresh temp dir), one interpreter per
  role. The live lake is not on this host.

**Question.** P4b says "a restart is not amnesia". A static read (cloud audit 2026-10-27, finding N2) found no lake
path for four live learners:
- the JEPA world model (`AutonomousArm.jepa`);
- `ValenceLearner` (`AutonomousArm._valence`);
- `_plan_w` (`AutonomousArm._plan_w`);
- `LearnedAttention` (`Brain._learned_attn`).

This probe asks the P4b question as P4b requires: does a NEW process on the same home get their learned state back?

## Result

**None of the four is restored by a new process. The positive control is restored.**

| learner | learns on this tier in 300 ticks? | natural run (seeds 7 and 11): child on the learned home vs fresh brain | planted run (seed 7) | verdict |
|---|---|---|---|---|
| JEPA world model | yes (3,119 numeric state values) | child bit-identical to fresh after the 1 tick that builds it lazily, both seeds | -- | **NOT RESTORED** |
| `ValenceLearner` | yes (66 values) | child bit-identical to fresh after 1 tick, both seeds | parent planted `w = 0.123...`; child has no learner until its first rewarded tick, then a fresh one | **NOT RESTORED** |
| `_plan_w` | yes | child = fresh = 0.1 (initial) at 0 ticks; distance to the parent's learned value 0.2107 (s7), 0.2199 (s11) | parent planted 0.4321; child reads 0.1 | **NOT RESTORED** |
| `LearnedAttention` | **no**: `w` unchanged after 300 ticks on the stdlib tier (parent = child = fresh) | not readable from a natural run | parent planted `w = [0.25, 5.5, 0.5, 4.0]`; child reads the construction values, bit-identical to fresh | **NOT RESTORED** |
| *positive control:* arm `GrowingLearner` prototypes (`growing_learners.json`) | yes (7 units s7, 4 units s11) | child at 0 ticks equals parent; fresh differs | parent 6 units; child 6; fresh 0 | **RESTORED** (the persistence path works on this host) |

**How to read it.**
- "Child bit-identical to fresh" means the learned home contributed nothing to that learner: its state is exactly
  what a never-trained brain with the same seed builds.
- After the parent's `close()`, the scratch home held only `brain_lake.bin` and `skills.spkl`. No side file exists for
  these learners either, so this is not a P4b side-file case. They are simply not persisted.
- `LearnedAttention` did not learn on this tier in 300 ticks; on the operator's e-prop tier it may. The planted run
  shows that even when its weights differ, a restart discards them.

**Consequences, stated only.**
- Every daemon relaunch resets these four learners.
- The owner's reply notes that `ValenceLearner` and `_plan_w` are E1b/E1c subjects. E1c runs fresh per-seed brains, so
  this does not affect E1c's reading. It does matter for daemon continuity.
- The JEPA world model (curiosity and surprise) restarts from zero at every relaunch.

**Limits.**
- Stdlib tier, 300 ticks, two seeds for the natural run and one for the planted run.
- `close()` is the clean shutdown. A killed process can only do worse, so it was not run.
- The JEPA and Valence learners are built lazily, so their natural comparison is made after the 1 tick that builds
  them, in both child and fresh.
- The digest flattens each learner's numeric attributes, skipping RNG objects. The planted run does not depend on that
  digest: it writes the weights the learner actually uses and reads them back.

## Probe code (verbatim, as run)

The driver runs three roles in separate interpreters:
1. parent: N ticks, then `close()`;
2. child: the same home, at 0 and then 1 ticks;
3. fresh: an empty home, at 0 and then 1 ticks.

The planted variant writes known values into the learners before `close()`.

Probe:

```python
"""Decision 5: cross-process restart probe for four live learners (P4b "a restart is not amnesia").

Three roles, each its own interpreter:
  parent HOME N  -- make_brain(seed, persist=True) on HOME, N ticks, digest, close()
  child  HOME K  -- make_brain(seed, persist=True) on the SAME HOME, K ticks (K=1 builds lazy learners), digest
  fresh  HOME K  -- the same on an EMPTY home (the never-learned reference)
Positive control: the arm's GrowingLearner prototypes (persisted via `growing_learners.json` since s2b):
child at K=0 must equal parent, fresh must differ.
Verdict per learner (from the vectors): RESTORED if |child-parent| < |fresh-parent| and child != fresh;
NOT RESTORED if child == fresh bit for bit (or child is the never-learned value).
"""

import json
import os
import sys
import zlib


def _flat(obj, depth=0):
    import numpy as np

    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    if obj is None:
        return []
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, np.ndarray):
        return [float(v) for v in obj.ravel().tolist()] if obj.dtype.kind in "fiu" else []
    if torch is not None and isinstance(obj, torch.Tensor):
        return [float(v) for v in obj.detach().cpu().double().ravel().tolist()]
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out += _flat(v, depth + 1)
        return out
    if isinstance(obj, dict) and depth < 3:
        out = []
        for k in sorted(obj, key=str):
            out += _flat(obj[k], depth + 1)
        return out
    if hasattr(obj, "__dict__") and depth < 2:
        out = []
        for k in sorted(vars(obj)):
            if k.startswith("_rng") or k in ("rng", "_rng"):
                continue
            out += _flat(vars(obj)[k], depth + 1)
        return out
    return []


def _digest(vec):
    import struct

    return zlib.crc32(struct.pack(f"{len(vec)}d", *vec)) if vec else None


def snapshot(b):
    arm = b.brain.arm
    learners = {
        "jepa": getattr(arm, "jepa", None),
        "valence": getattr(arm, "_valence", None),
        "plan_w": getattr(arm, "_plan_w", None),
        "learned_attention": getattr(b, "_learned_attn", None),
    }
    out = {}
    for name, obj in learners.items():
        v = _flat(obj)
        out[name] = {"built": obj is not None, "n": len(v), "crc": _digest(v), "vec": v}
    pc = arm.learner.state_dict()["prototypes"]
    out["POSITIVE_CONTROL_arm_prototypes"] = {
        "built": True,
        "n": len(pc),
        "crc": zlib.crc32(json.dumps(pc).encode()),
        "vec": [],
    }
    return out


def main():
    role, home, ticks, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.environ["NORD_HOME"] = home
    import torch

    torch.set_num_threads(1)
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=seed, persist=True)
    tier = b.state()["perception_tier"]
    for _ in range(ticks):
        b.tick()
    snap = snapshot(b)
    if role == "parent":
        b.close()
    print("SNAP " + json.dumps({"role": role, "tier": tier, "ticks": ticks, "snap": snap}))


if __name__ == "__main__":
    main()
```

Driver:

```python
"""Driver: runs parent/child/fresh for one seed and prints per-learner verdicts."""

import json
import math
import os
import subprocess
import sys
import tempfile

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restart_probe.py")
PY = sys.executable


def run(role, home, ticks, seed):
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1")
    p = subprocess.run(
        [PY, PROBE, role, home, str(ticks), str(seed)],
        capture_output=True,
        text=True,
        env=env,
        timeout=3000,
        cwd=os.path.expanduser("~"),
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("SNAP ")]
    if p.returncode != 0 or not line:
        raise SystemExit(f"{role} failed rc={p.returncode}\n{p.stderr[-3000:]}")
    return json.loads(line[0][5:])


def dist(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    seed, n_ticks = int(sys.argv[1]), int(sys.argv[2])
    home = tempfile.mkdtemp(prefix="rp_home_")
    empty0 = tempfile.mkdtemp(prefix="rp_empty0_")
    empty1 = tempfile.mkdtemp(prefix="rp_empty1_")
    P = run("parent", home, n_ticks, seed)
    lake = os.listdir(home)
    C0 = run("child", home, 0, seed)
    C1 = run("child", home, 1, seed)
    F0 = run("fresh", empty0, 0, seed)
    F1 = run("fresh", empty1, 1, seed)
    rep = {"seed": seed, "parent_ticks": n_ticks, "tier": P["tier"], "home_files": lake, "learners": {}}
    for name in P["snap"]:
        p = P["snap"][name]
        row = {"parent_built": p["built"], "parent_n": p["n"]}
        for tag, S in (("child0", C0), ("child1", C1), ("fresh0", F0), ("fresh1", F1)):
            s = S["snap"][name]
            row[tag] = {
                "built": s["built"],
                "crc": s["crc"],
                "eq_parent": s["crc"] == p["crc"],
                "d_parent": dist(s["vec"], p["vec"]),
            }
        row["child1_eq_fresh1"] = C1["snap"][name]["crc"] == F1["snap"][name]["crc"]
        row["child0_eq_fresh0"] = C0["snap"][name]["crc"] == F0["snap"][name]["crc"]
        rep["learners"][name] = row
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
```

Planted variant:

```python
"""Planted variant: the parent WRITES a known value into each learner (the change learning would make), closes; a new
process on the same home reads it back. A learner whose state survives reads the planted value; one that does not reads
its construction value. Positive control: arm prototype count planted by ticking (persisted path)."""
import json, os, sys
role, home, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.environ["NORD_HOME"] = home
import numpy as np, torch
torch.set_num_threads(1)
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=seed, persist=True)
arm = b.brain.arm
if role == "parent":
    for _ in range(60):
        b.tick()
    b._learned_attn.w = np.array([0.25, 5.5, 0.5, 4.0])  # inside its [0.2, 6.0] bounds
    arm._plan_w = 0.4321
    if arm._valence is not None:
        arm._valence.w = [0.123] * len(arm._valence.w)
out = {"role": role, "attn_w": [float(v) for v in b._learned_attn.w], "plan_w": float(arm._plan_w),
       "valence_w0": (arm._valence.w[0] if arm._valence is not None else None),
       "arm_units": arm.learner.n_units}
if role == "parent":
    b.close()
print("SNAP " + json.dumps(out))
```
