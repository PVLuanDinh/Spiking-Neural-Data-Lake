r"""ARC-AGI scoring harness -- the honest Gf yardstick.

ARC-AGI (Chollet 2019, github.com/fchollet/ARC-AGI) is the benchmark this project uses to decide
whether a fluid-intelligence claim is real. Each task gives a few input/output GRID pairs and asks for
the output of a held-out input. The solution program is different for every task and is NOT in any
taught grammar, so a solver must SYNTHESISE it at test time -- exactly the structural bar
("developer-aware generalisation") that a lookup table, a template bank, or a taught-triple traversal
cannot clear.

Scoring is the official one: EXACT grid match, pass@k with k=2 attempts, no partial credit.

    PYTHONPATH=src python -m snn_data_lake.reasoning.arc_bench

Data lives in data/arc/{training,evaluation}/*.json (800 tasks). Download once with:
    curl -sL -o arc.zip https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip
and extract the repo's data/ subtree into data/arc/.

HONEST ANCHORS for reading any score this harness prints:
  - a solver that echoes its input scores ~0% (the self-check asserts this -- it is the control that
    must genuinely fail, so a 0 here means the harness is measuring, not that it is broken)
  - human performance on the public evaluation set is ~85% (Chollet 2019 / ARC Prize 2024 reporting)
  - a score is only meaningful on tasks the solver has never been tuned against: report TRAINING and
    EVALUATION splits separately and never tune on evaluation

Pure stdlib. No numpy, no torch. Representation-only: imports no brainstem, routes no actuator.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA = os.path.join(_ROOT, "data", "arc")

SPLITS = ("training", "evaluation")


def load_tasks(split="evaluation", root=None):
    """Return [(task_id, task_dict), ...] sorted by id. Empty list if the split is not downloaded."""
    d = os.path.join(root or DATA, split)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            out.append((fn[:-5], json.load(f)))
    return out


def _grids_equal(a, b):
    """Exact match, shape included. Tolerates tuples vs lists; rejects anything malformed."""
    try:
        if len(a) != len(b):
            return False
        for ra, rb in zip(a, b):
            if len(ra) != len(rb) or any(int(x) != int(y) for x, y in zip(ra, rb)):
                return False
        return True
    except (TypeError, ValueError):
        return False


def score_task(solver, task, attempts=2):
    """Solve every test input of one task. Returns True only if ALL test outputs are exact-matched.

    solver(train_pairs, test_input, attempts) -> list of candidate grids (at most `attempts` used).
    A solver that raises, returns None, or returns junk simply scores False -- never crashes the run.
    """
    train = [(p["input"], p["output"]) for p in task.get("train", [])]
    for pair in task.get("test", []):
        want = pair["output"]
        try:
            cands = solver(train, pair["input"], attempts)
        except Exception:
            return False
        if not cands:
            return False
        if not any(_grids_equal(c, want) for c in list(cands)[:attempts]):
            return False
    return True


def score(solver, split="evaluation", limit=None, attempts=2, root=None, verbose=False):
    """Score a solver over a split. Returns {'split','solved','total','rate','solved_ids'}."""
    tasks = load_tasks(split, root=root)
    if limit is not None:
        tasks = tasks[:limit]
    solved = []
    for tid, task in tasks:
        ok = score_task(solver, task, attempts=attempts)
        if ok:
            solved.append(tid)
        if verbose:
            print(f"  {tid} {'PASS' if ok else 'fail'}")
    n = len(tasks)
    return {
        "split": split,
        "solved": len(solved),
        "total": n,
        "rate": (len(solved) / n) if n else 0.0,
        "solved_ids": solved,
    }


def identity_solver(train, test_input, attempts=2):
    """Control: echo the input. Must score ~0 -- if this scores well the harness is broken."""
    return [test_input]


def constant_solver(train, test_input, attempts=2):
    """Control: copy the first training output. Must also score ~0."""
    return [train[0][1]] if train else []


def main():
    ok = True
    tasks = load_tasks("evaluation")
    if not tasks:
        print(f"SKIP arc_bench: no ARC data at {DATA} (see module docstring for the download)")
        return 0

    tr = load_tasks("training")
    print(f"arc_bench: {len(tr)} training tasks, {len(tasks)} evaluation tasks at {DATA}")

    # The controls MUST genuinely fail. This is what makes a real score believable.
    for name, solv in (("identity", identity_solver), ("constant", constant_solver)):
        r = score(solv, "evaluation", limit=200)
        print(f"  control {name:9s}: {r['solved']}/{r['total']} = {100 * r['rate']:.1f}%")
        if r["rate"] > 0.02:
            print(f"  FAIL: control '{name}' scored {r['rate']:.3f}; a trivial solver must not pass ARC")
            ok = False

    # Exact-match plumbing must be strict in both directions.
    if not _grids_equal([[1, 2]], [[1, 2]]):
        print("  FAIL: _grids_equal rejected identical grids")
        ok = False
    for a, b in (([[1, 2]], [[1, 3]]), ([[1, 2]], [[1, 2], [3, 4]]), ([[1, 2]], [[1]]), ([[1]], None)):
        if _grids_equal(a, b):
            print(f"  FAIL: _grids_equal accepted {a!r} == {b!r}")
            ok = False

    # An oracle that reads the answer must score 100% -- proves the harness can register a success.
    def oracle(train, test_input, attempts=2, _tasks=dict(tasks)):
        for _tid, t in tasks:
            for p in t.get("test", []):
                if _grids_equal(p["input"], test_input):
                    return [p["output"]]
        return []

    r = score(oracle, "evaluation", limit=50)
    print(f"  oracle           : {r['solved']}/{r['total']} = {100 * r['rate']:.1f}%")
    if r["solved"] != r["total"]:
        print("  FAIL: oracle did not score 100% -- the harness cannot register a correct answer")
        ok = False

    print("arc_bench self-check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
