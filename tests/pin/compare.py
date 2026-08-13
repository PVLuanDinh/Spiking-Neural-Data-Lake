"""FR-017 bit-exact numeric pin -- COMPARE side.

Uses `torch.equal` (EXACT), never `allclose`. A tolerance wide enough to survive a torch-build
difference is wide enough to hide the `flatten_frames` reshape error this exists to catch, and a
control that cannot fail on the defect it exists to catch is deleted rather than weakened
(P8 guard 1).

Lives in tests/ for the reason given in capture.py: it is an instrument that measures a build, not
a build -- the category p4c already excludes tests/** under.

Usage:
    .venv-lake/Scripts/python.exe tests/pin/compare.py
    .venv-lake/Scripts/python.exe tests/pin/compare.py --b reshape

Exit code 0 = every tensor bit-identical AND at least one was compared. Exit code 1 = a difference
(blocks the lane, SC-003) OR nothing was compared -- a run that checks zero tensors is a failure,
not a pass.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

KEYS = [
    "forward.logits_sum",
    "forward.latent_sum",
    "eprop_grads.grads",
    "eprop_grads.loss",
    "eprop_grads.spikes",
    "bptt_grads.grads",
]


def _eq(a, b, path=""):
    """Exact equality over a tensor, a list of tensors, or a dict of them.

    `bptt_grads` returns a dict keyed W_in/W_rec/W_out/b_out; `eprop_grads` returns its own
    carrier. Recursing rather than assuming one shape is what lets the SAME comparator run on both
    sides of the merge.

    NaN != NaN under torch.equal, and that is the honest answer: a NaN where none existed before is
    a difference, not a wash.
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False, f"{path}: dict vs non-dict"
        if set(a) != set(b):
            return False, f"{path}: keys differ {sorted(set(a) ^ set(b))}"
        for k in sorted(a):
            ok, why = _eq(a[k], b[k], f"{path}.{k}" if path else k)
            if not ok:
                return False, why
        return True, ""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return False, f"{path}: sequence length differs"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            ok, why = _eq(x, y, f"{path}[{i}]")
            if not ok:
                return False, why
        return True, ""
    if a.shape != b.shape:
        return False, f"{path}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    if not torch.equal(a, b):
        return False, f"{path}: max|delta|={(a - b).abs().max().item():.6g}"
    return True, ""


def compare(tag_a, tag_b):
    failures = 0
    checked = 0
    for name in ("nmnist", "shd"):
        pa = os.path.join(_HERE, f"{name}-{tag_a}.pt")
        pb = os.path.join(_HERE, f"{name}-{tag_b}.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"SKIP {name}: missing {os.path.basename(pa if not os.path.exists(pa) else pb)}")
            continue
        A = torch.load(pa, map_location="cpu", weights_only=False)
        B = torch.load(pb, map_location="cpu", weights_only=False)

        # Disclosure, not a silent pass: a cross-build comparison is not a contract this project
        # holds (research.md D-3). Say so loudly rather than reporting a difference as a merge bug.
        if A["meta"]["torch"] != B["meta"]["torch"] or A["meta"]["device"] != B["meta"]["device"]:
            am, bm = A["meta"], B["meta"]
            print(
                f"WARNING {name}: build/device differs "
                f"({am['torch']}/{am['device']} vs {bm['torch']}/{bm['device']}). "
                "Bit-exactness is NOT expected across builds; this comparison is not evidence."
            )
        if A["meta"].get("shd_module") != B["meta"].get("shd_module"):
            print(
                f"NOTE {name}: shd_module moved ({A['meta'].get('shd_module')} -> "
                f"{B['meta'].get('shd_module')}); expected across C3."
            )

        print(f"--- {name}: {tag_a} vs {tag_b} ---")
        for k in KEYS:
            ok, why = _eq(A["data"][k], B["data"][k], k)
            checked += 1
            if ok:
                print(f"  OK        {k}")
            else:
                failures += 1
                print(f"  DIFFERS   {k}  {why}")

    print(f"\n{checked} tensors checked, {failures} differ")
    # A run that compared NOTHING must not exit 0. Both capture files can be missing -- a typo in
    # --a/--b, a tag never captured, a cleaned directory -- and the loop above `continue`s past each
    # one, so without this the harness reports success for having done no work. That is this
    # project's own harness-false-negative shape, and it would land in the ONE instrument the merge
    # was accepted on. Found by the pre-push review, not by the harness itself.
    return 1 if (failures or checked == 0) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FR-017 bit-exact pin -- compare side")
    ap.add_argument("--a", default="before")
    ap.add_argument("--b", default="after")
    a = ap.parse_args()
    sys.exit(compare(a.a, a.b))
