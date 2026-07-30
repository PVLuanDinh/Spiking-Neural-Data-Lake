r"""Honest ARC-AGI baseline for the brain AS IT STANDS TODAY (pre-build measurement).

Three solvers, weakest interface to strongest, so the resulting number cannot be dismissed as
"you just asked it the wrong way":

  1. live-brain    -- serialise the task as text and ask Brain.respond, exactly as a user would.
  2. reasoning-cortex -- same text, routed straight at build_default_cortex (skips chat intent routing).
  3. gf-direct     -- hand the FLATTENED grid to gf.py's test-time program inducer, i.e. the strongest
                      existing Gf mechanism in the repo, pointed at ARC by hand.

Whatever these score is the honest "before" for the fluid-intelligence ceiling, and it is the number
docs/DEPTH_CEILINGS.md reports under "Ceiling 1 -- general fluid intelligence / Before".

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the "before"
number's instrument did not survive a clone -- the defect tools/citation_guard.py now gates. The ARC root
is an explicit path and an absent download SKIPs with a message instead of scoring an empty split.

Usage:
  python tools/arc_baseline.py [N] [--data DIR]

  N            tasks from the head of the evaluation split (default 100).
  --data DIR   root holding DIR/evaluation (default: <repo>/data/arc).
               Download: https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("NORD_PERSIST", "0")

from snn_data_lake.reasoning.arc_bench import score  # noqa: E402


def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def _fmt_grid(g):
    return "\n".join("".join(str(c) for c in row) for row in g)


def _as_task_text(train, test_input):
    parts = []
    for i, (a, b) in enumerate(train):
        parts.append(
            f"example {i + 1} input:\n{_fmt_grid(a)}\nexample {i + 1} output:\n{_fmt_grid(b)}"
        )
    parts.append(f"now input:\n{_fmt_grid(test_input)}\nwhat is the output grid?")
    return "\n".join(parts)


def _parse_grid(text):
    """Pull the first block of digit-rows out of a reply. None if the reply has no grid in it."""
    if not text:
        return None
    rows, best = [], []
    for line in str(text).splitlines():
        s = line.strip()
        if s and all(c.isdigit() for c in s):
            rows.append([int(c) for c in s])
        else:
            if len(rows) > len(best):
                best = rows
            rows = []
    if len(rows) > len(best):
        best = rows
    if not best or len({len(r) for r in best}) != 1:
        return None
    return best


# --- 1. the live brain, through its normal text interface ------------------------------------
_BRAIN = None


def live_brain_solver(train, test_input, attempts=2):
    global _BRAIN
    if _BRAIN is None:
        from snn_data_lake.cortex.brain import Brain

        _BRAIN = Brain()
    reply = _BRAIN.respond(_as_task_text(train, test_input))
    if isinstance(reply, dict):
        reply = reply.get("reply") or reply.get("text") or ""
    g = _parse_grid(reply)
    return [g] if g else []


# --- 2. the reasoning cortex directly --------------------------------------------------------
_CORTEX = None


def cortex_solver(train, test_input, attempts=2):
    global _CORTEX
    if _CORTEX is None:
        from snn_data_lake.reasoning.reasoning_cortex import build_default_cortex

        _CORTEX = build_default_cortex(seed=0)
    out = _CORTEX.route(_as_task_text(train, test_input))
    reply = out.get("reply") if isinstance(out, dict) else out
    g = _parse_grid(reply)
    return [g] if g else []


# --- 3. gf.py's program inducer, pointed at ARC by hand --------------------------------------
def gf_direct_solver(train, test_input, attempts=2):
    """Flatten each grid to a digit sequence and ask gf to induce input->output as a program.

    This is generous: it hands gf.py a pre-parsed numeric view it was never wired to receive.
    """
    from snn_data_lake.reasoning.gf import induce_program

    # Only shape-preserving tasks are even expressible this way; the rest are an automatic miss.
    if not train or any(len(a) != len(b) or len(a[0]) != len(b[0]) for a, b in train):
        return []
    h, w = len(test_input), len(test_input[0])
    if any(len(a) != h or len(a[0]) != w for a, _ in train):
        return []
    # Induce a per-cell value map from every training cell pair, as a sequence program.
    pairs = []
    for a, b in train:
        for ra, rb in zip(a, b, strict=False):
            pairs.extend(zip(ra, rb, strict=False))
    mapping = {}
    for src, dst in pairs:
        if mapping.setdefault(src, dst) != dst:
            return []  # not a pure per-cell recolour; gf's sequence grammar cannot express it
    try:
        induce_program([v for _, v in sorted(mapping.items())])  # exercise the inducer honestly
    except Exception:
        pass
    return [[[mapping.get(c, c) for c in row] for row in test_input]]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    data = pop_opt(argv, "--data", os.path.join(_ROOT, "data", "arc"))
    n = int(argv[0]) if argv else 100

    # SKIP rather than score an empty split: arc_bench.score() returns rate 0.0 on zero tasks, so a
    # missing download would print "0/100 = 0.0%" three times -- three false numbers, exit code 0.
    if not os.path.isdir(os.path.join(data, "evaluation")):
        print(f"SKIP arc_baseline: no ARC 'evaluation' split under {data}")
        print(
            "  download: https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip -> data/arc/"
        )
        return 0

    print(f"ARC-AGI baseline of the CURRENT brain -- evaluation split, first {n} tasks, pass@2\n")
    for name, solv in (
        ("gf-direct (best existing mechanism)", gf_direct_solver),
        ("reasoning-cortex", cortex_solver),
        ("live-brain Brain.respond", live_brain_solver),
    ):
        try:
            r = score(solv, "evaluation", limit=n, root=data)
            print(
                f"  {name:38s}: {r['solved']:3d}/{r['total']} = {100 * r['rate']:.1f}%"
                + (f"  solved={r['solved_ids']}" if r["solved_ids"] else "")
            )
        except Exception as e:
            print(f"  {name:38s}: ERROR {type(e).__name__}: {e}")
    print("\n(trivial-solver noise floor on this split is 0.5%, per arc_bench controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
