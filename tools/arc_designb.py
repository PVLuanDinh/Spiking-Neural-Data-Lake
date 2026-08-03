r"""arc_designb -- telemetry and control arms for specs/006-fixed-point-gf DESIGN B.

docs/INVENTION.md section 3.6 pre-registers the control arms. This tool runs the ones that are
OFFLINE (they need the mint, not the searcher) and prints them as counts:

  * TELEMETRY (never a success claim -- INVENTION.md 3.6 "the trap"): how many records the mint
    emits per task, how many carry an induced literal, the selector-length distribution, and which
    descriptors actually win literals. Mint counts rising is NOT a result.
  * CONTROL ARM 3 -- LABEL-SHUFFLED SELECTOR. Induce selectors from residuals whose written /
    untouched labels have been permuted within each train pair. If junk labels admit at the same
    rate as real ones, the admission bar is not a bar. Reported as "N of M" admitted, both arms.
  * CONTROL ARM 2 (partial) -- POOL SIZE. The node count at max_depth 3 scales with the CUBE of the
    pool size, so a mechanism that adds vocabulary while tripling the branching factor can lose
    solves at a fixed budget. Pool size per task is reported so a solve delta can be read against
    the search cost that bought it.

Determinism: random.Random(seed) per instance, sorted iteration, ASCII-only output.

Usage:
  python tools/arc_designb.py <training|evaluation> [limit] [--data DIR] [--seed N]
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.reasoning import arc_bench, arc_mint, arc_residual, arc_solver  # noqa: E402

USAGE = "usage: python tools/arc_designb.py <training|evaluation> [limit] [--data DIR] [--seed N]"


def pop_opt(argv, name, default):
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _shuffled_admits(splits, rows_per_pair, rng):
    """Control arm 3: permute the written / untouched labels WITHIN each train pair, keeping the
    counts identical, and ask whether the alphabet still admits a conjunction. Same units, same
    descriptors, same greedy covering -- only the labels are junk."""
    pos_rows, neg_rows = [], []
    for (pos, neg), rows in zip(splits, rows_per_pair, strict=True):
        idx = [k for k, _c in pos] + [k for k, _c in neg]
        rng.shuffle(idx)
        npos = len(pos)
        pos_rows.extend(rows[k] for k in idx[:npos])
        neg_rows.extend(rows[k] for k in idx[npos:])
    if not pos_rows:
        return None
    return arc_residual.induce_selector(pos_rows, neg_rows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    data = pop_opt(argv, "--data", os.path.join(_ROOT, "data", "arc"))
    seed = int(pop_opt(argv, "--seed", "0"))
    if not argv:
        _emit(USAGE)
        return 2
    split = argv[0]
    limit = int(argv[1]) if len(argv) > 1 else None
    if split not in arc_bench.SPLITS:
        _emit(USAGE)
        return 2
    if not os.path.isdir(os.path.join(data, split)):
        _emit(f"SKIP arc_designb: no ARC '{split}' split under {data}")
        return 0

    tasks = arc_bench.load_tasks(split, root=data)
    if limit:
        tasks = tasks[:limit]
    rng = random.Random(seed)

    n_tasks = 0
    n_with_records = 0
    n_records = 0
    n_bearing = 0
    sel_len = {}
    lit_use = {}
    real_admit = 0
    shuf_admit = 0
    separable = 0
    pool_sizes = []

    for _tid, task in tasks:
        n_tasks += 1
        train = [(p["input"], p["output"]) for p in task["train"]]
        pairs = [{"in": arc_mint._grid(i), "out": arc_mint._grid(o)} for i, o in train]

        arc_residual.clear_cache()
        recs = arc_mint.mint_records(train)
        if recs:
            n_with_records += 1
        n_records += len(recs)
        for rec in recs:
            if rec.selector:
                n_bearing += 1
            sel_len[len(rec.selector)] = sel_len.get(len(rec.selector), 0) + 1
            for pid, _v in rec.selector:
                name = arc_residual.DESCRIPTOR_NAMES[pid]
                lit_use[name] = lit_use.get(name, 0) + 1

        tin = tuple(arc_solver._tup(i) for i, _o in train)
        tout = tuple(arc_solver._tup(o) for _i, o in train)
        pool_sizes.append(len(arc_solver._pool(tin, tout)))

        # the residual split this task admits at all -- the two arms only differ in the labels
        splits, rows_per_pair, ok = [], [], True
        for pair in pairs:
            got = arc_mint._split_changed(pair["in"], pair["out"])
            if got is None:
                ok = False
                break
            splits.append(got)
            rows_per_pair.append(arc_residual.descriptor_rows(pair["in"]))
        if not ok or not any(p for p, _n in splits):
            continue
        separable += 1
        pos_rows, neg_rows = [], []
        for (pos, neg), rows in zip(splits, rows_per_pair, strict=True):
            pos_rows.extend(rows[k] for k, _c in pos)
            neg_rows.extend(rows[k] for k, _c in neg)
        if arc_residual.induce_selector(pos_rows, neg_rows) is not None:
            real_admit += 1
        if _shuffled_admits(splits, rows_per_pair, rng) is not None:
            shuf_admit += 1

    width = 96
    _emit("=" * width)
    _emit(f"DESIGN B -- telemetry and offline control arms | split={split} tasks={n_tasks}")
    _emit(
        f"descriptor alphabet version {arc_residual.DESCRIPTOR_VERSION}, "
        f"{arc_residual.N_DESCRIPTORS} predicates, literal cap {arc_residual.SELECTOR_LITERAL_CAP}"
    )
    _emit("=" * width)
    _emit("TELEMETRY (NOT an outcome -- INVENTION.md 3.6 forbids these in any success claim)")
    _emit(f"  tasks minting at least one record : {n_with_records} of {n_tasks}")
    _emit(f"  records minted (total)            : {n_records}")
    _emit(f"  records carrying >=1 literal      : {n_bearing} of {n_records}")
    _emit(f"  selector-length distribution      : {sorted(sel_len.items())}")
    top = sorted(lit_use.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    _emit(f"  descriptors winning literals (top): {top}")
    if pool_sizes:
        _emit(
            f"  pool size per task min/med/max    : {min(pool_sizes)} / "
            f"{sorted(pool_sizes)[len(pool_sizes) // 2]} / {max(pool_sizes)}"
        )
    _emit("-" * width)
    _emit("CONTROL ARM 3 -- LABEL-SHUFFLED SELECTOR (the admission bar must be a bar)")
    _emit(f"  tasks admitting a residual split  : {separable} of {n_tasks}")
    _emit(f"  REAL labels    -> selector admitted: {real_admit} of {separable}")
    _emit(f"  SHUFFLED labels-> selector admitted: {shuf_admit} of {separable}  (seed {seed})")
    if separable:
        _emit("  reading: a shuffled rate at or near the real rate means the alphabet can separate")
        _emit(
            "  ANY split of these units, so admission is not evidence the selector found structure."
        )
    _emit("=" * width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
