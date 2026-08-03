r"""arc_probe_ids -- per-ID winning-program accounting for specs/006-fixed-point-gf DESIGN B.

A solve count is not a result until each solve says HOW it was earned. For every task id given,
this re-runs the ONE searcher at the FROZEN bench budgets (node_budget 4000, time_budget 5.0,
attempts 2 -- tools/arc_score.py semantics) and reports, for the winning program:

  * the program's step names and closer;
  * whether a minted record contributed, and if so its declarative fields;
  * FALSIFIER CLAUSE (a) FUNCTIONAL: the record's rule form is parametric (not rank/lookup), so the
    answer is COMPUTED, not looked up;
  * FALSIFIER CLAUSE (b) SELECTOR-BEARING: the record carries >=1 induced literal AND a degenerate
    always-true conjunction does NOT reproduce it on the train pairs. Both halves are checked --
    carrying a literal is not the same as the literal doing work.

Clause (c) OUTSIDE the contaminated dev subset is a set membership the caller checks against
scratchpad/oracle_dev.json; the ids in that file are printed here when it is present.

Usage:
  python tools/arc_probe_ids.py <training|evaluation> <id>[,<id>...] [--data DIR]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.reasoning import arc_bench, arc_mint  # noqa: E402
from snn_data_lake.reasoning.arc_solver import synthesize  # noqa: E402

USAGE = "usage: python tools/arc_probe_ids.py <training|evaluation> <id>[,<id>...] [--data DIR]"
_BUDGETS = dict(attempts=2, max_depth=3, node_budget=4000, time_budget=5.0)


def pop_opt(argv, name, default):
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _dev_ids():
    path = os.path.join(_ROOT, "scratchpad", "oracle_dev.json")
    if not os.path.isfile(path):
        return None
    import json

    with open(path, encoding="ascii", errors="replace") as fh:
        blob = json.load(fh)
    if isinstance(blob, dict):
        return sorted(blob)
    if isinstance(blob, list):
        return sorted(str(x) if not isinstance(x, dict) else str(x.get("id", x)) for x in blob)
    return None


def probe(task, want_ids=None):
    """Return (solved, report lines) for one loaded task."""
    train = [(p["input"], p["output"]) for p in task["train"]]
    lines = []
    solved = False
    for ti, test in enumerate(task["test"]):
        cands, progs = synthesize(train, test["input"], **_BUDGETS)
        hit = None
        for cand, prog in zip(cands, progs, strict=True):
            if arc_bench._grids_equal(cand, test["output"]):
                hit = prog
                break
        if hit is None:
            lines.append(f"    test {ti}: NOT solved")
            continue
        solved = True
        names, closer = hit
        lines.append(
            f"    test {ti}: SOLVED via {' > '.join(names) or '(empty)'} [closer={closer}]"
        )
        for name in names:
            if not name.startswith("mint:"):
                continue
            recs = [r for r in arc_mint.mint_records(train) if "mint:" + r.name == name]
            if not recs:
                lines.append(f"      {name}: record not re-derivable (report this)")
                continue
            rec = recs[0]
            functional = not rec.memo_flag
            bearing = bool(rec.selector)
            works = None
            if bearing:
                degenerate = arc_mint.compile_record(
                    rec.unit, rec.attribute, rec.rule_form, rec.rule_params, rec.consumer, ()
                )
                pairs = [(arc_mint._grid(i), arc_mint._grid(o)) for i, o in train]
                works = not all(degenerate(gi) == go for gi, go in pairs)
            lines.append(
                f"      record unit={rec.unit} attribute={rec.attribute} "
                f"rule_form={rec.rule_form} params={rec.rule_params}"
            )
            lines.append(
                f"      (a) FUNCTIONAL={functional}  (b) SELECTOR-BEARING={bearing}"
                f" selector={rec.selector} always_true_fails={works}"
            )
    return solved, lines


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    data = pop_opt(argv, "--data", os.path.join(_ROOT, "data", "arc"))
    if len(argv) < 2:
        _emit(USAGE)
        return 2
    split, ids = argv[0], [x for x in argv[1].split(",") if x]
    if split not in arc_bench.SPLITS:
        _emit(USAGE)
        return 2
    if not os.path.isdir(os.path.join(data, split)):
        _emit(f"SKIP arc_probe_ids: no ARC '{split}' split under {data}")
        return 0

    dev = _dev_ids()
    _emit("=" * 96)
    _emit(f"PER-ID WINNING-PROGRAM ACCOUNTING | split={split} | frozen budgets {_BUDGETS}")
    if dev is None:
        _emit("dev subset: scratchpad/oracle_dev.json ABSENT -- clause (c) cannot be checked here")
    else:
        _emit(f"dev subset (contaminated): {len(dev)} ids -> {','.join(dev)}")
    _emit("=" * 96)
    tasks = dict(arc_bench.load_tasks(split, root=data))
    for tid in ids:
        task = tasks.get(tid)
        if task is None:
            _emit(f"  {tid}: NOT FOUND in split {split}")
            continue
        contaminated = "UNKNOWN" if dev is None else ("YES" if tid in dev else "NO")
        _emit(f"  {tid}  (in dev subset: {contaminated})")
        _solved, lines = probe(task)
        for line in lines:
            _emit(line)
    _emit("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
