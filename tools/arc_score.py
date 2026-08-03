r"""arc_score -- score reasoning/arc_solver on an ARC-AGI split, program library ON or OFF.

This is the instrument behind the ARC rates in docs/DEPTH_CEILINGS.md ("Ceiling 1 -- general fluid
intelligence"): the post-arc_solver rate on the evaluation split, and the library-on / library-off A/B
that says what the learned composite programs are actually worth.

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88). A document may
not present an untracked script as its reproduction path -- see tools/citation_guard.py for the three
incidents that established the rule and now enforce it. Promotion means three concrete things: the input
path is EXPLICIT (--data), an absent input SKIPs with a message instead of printing 0/0, and the module
bootstraps its own sys.path so `python tools/arc_score.py ...` works on a fresh clone with no env setup.

Usage:
  python tools/arc_score.py <training|evaluation> <on|off> [limit] [--data DIR]

  --data DIR   root holding DIR/training and DIR/evaluation (default: <repo>/data/arc).
               Download: https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.reasoning import arc_bench  # noqa: E402
from snn_data_lake.reasoning.arc_solver import make_solver  # noqa: E402

USAGE = "usage: python tools/arc_score.py <training|evaluation> [limit] [--data DIR]"


def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default.

    Deliberately not argparse: these tools are read as often as they are run, and the positional
    contract below is the one docs/DEPTH_CEILINGS.md quotes verbatim.
    """
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    data = pop_opt(argv, "--data", os.path.join(_ROOT, "data", "arc"))
    if not argv:
        print(USAGE)
        return 2
    split = argv[0]
    # The library A/B is RETIRED (P2, 2026-08-03): ArcLibrary was measured earning 0 solves --
    # library on and off returned the byte-identical solved-ID set on BOTH splits -- and is deleted
    # from arc_solver. A legacy `on|off` token is still ACCEPTED so the reproduction path quoted in
    # docs/DEPTH_CEILINGS.md does not silently mean something new, but it selects nothing: there is
    # exactly one code path now, and this says so out loud rather than pretending to switch.
    if len(argv) > 1 and argv[1] in ("on", "off"):
        print(
            f"NOTE: '{argv[1]}' ignored -- the composite library is RETIRED (P2); one path remains"
        )
        del argv[1]
    limit = int(argv[1]) if len(argv) > 1 else None
    if split not in arc_bench.SPLITS:
        print(USAGE)
        return 2

    # SKIP rather than score an empty split. arc_bench.score() returns rate 0.0 on zero tasks, so
    # without this guard a missing download prints "0/0 = 0.00%" -- a number, and a false one. The
    # recurring defect this repo tracks is a stage that fails, swallows it and reports a placeholder.
    if not os.path.isdir(os.path.join(data, split)):
        print(f"SKIP arc_score: no ARC '{split}' split under {data}")
        print(
            "  download: https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip -> data/arc/"
        )
        return 0

    solver = make_solver(node_budget=4000, time_budget=5.0)
    t0 = time.time()
    r = arc_bench.score(solver, split, limit=limit, root=data)
    dt = time.time() - t0
    print(
        f"split={split} limit={limit} : "
        f"{r['solved']}/{r['total']} = {100 * r['rate']:.2f}%  ({dt:.0f} s)"
    )
    print(f"solved_ids: {','.join(r['solved_ids'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
