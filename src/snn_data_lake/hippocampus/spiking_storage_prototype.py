"""
Spiking associative-memory storage prototype.

Project aim: reduce computational power and storage space with the experimental
SNN data-storing method described in the research docs. This prototype both
DEMONSTRATES the method and MEASURES those two metrics against a dense baseline.

Storage mechanism (straight from the docs):
  - long-term storage  == the stored patterns themselves. The Hopfield weight
    matrix W = sum_p (xi_p - a)(xi_p - a)^T is rank-P, so we never materialise the
    N x N matrix: we keep the P sparse patterns as index lists and reconstruct the
    needed correlations on the fly (v0.7 -- factored storage). The data IS the
    network fabric: storage and compute are co-located, no Von Neumann shuttle.
  - volatile recall    == attractor dynamics. Inject a NOISY / PARTIAL cue, then
    iterate sparse k-winners-take-all spiking until the network settles onto the
    stored pattern. Recall is content-addressable: a corrupted cue recovers the
    clean memory, AND denoises it -- that is the value over a plain pattern list.

Where the savings come from:
  - storage : O(P*k) index bytes instead of O(N^2) weights. For P,k << N this is a
    real, large reduction -- the factored memory is hundreds of times smaller than
    the dense W it represents (reported below). Fixed in v0.7.
  - compute : recall is event-driven and factored -- O(P*k) work per step instead
    of the dense O(N^2) matvec a materialised Hopfield net would do each step.

Recall math (so the factored form is exact, not an approximation):
  With W_ij = sum_p (xi_p,i - a)*(xi_p,j - a), zero diagonal, and cue active-set s,
  let d_p = |xi_p & s| - a*|s|. Then
      h_i = sum_p (xi_p,i - a)*d_p  -  diag_i  =  S_i - a*D - diag_i,
  where S_i = sum of d_p over the patterns that contain i, D = sum_p d_p (constant
  in i, so it drops out of arg-top-k), and diag_i = cnt_i*(1-a)^2 + (P-cnt_i)*a^2
  for i in s (else 0). Identical ranking to the dense matrix -- same recall, far
  less memory.

DEPENDENCY NOTE (P7; the zero-dependency rule was retired 2026-07-20). This module
is stdlib ON THE MERITS, measured 2026-08-04 in .venv-lake -- not by habit and not
by a purity argument. A numpy port of `recall` (dense P x N pattern matrix, two
matmuls, argpartition; settled output verified IDENTICAL on every probe of all
three configurations) is SLOWER at both widths this module is actually run at:
    consumer width   n=64,   k=12, P=6   : stdlib   0.306 ms   numpy   0.456 ms  (0.67x)
    self-check width n=256,  k=20, P=15  : stdlib   2.325 ms   numpy   3.660 ms  (0.64x)
    scale-up         n=4096, k=64, P=200 : stdlib 862.685 ms   numpy 711.002 ms  (1.21x)
numpy only starts winning somewhere between n=256 and n=4096 -- far above every
live width -- and the vectorised form gets there by MATERIALISING the P x N pattern
matrix, which is the one thing this module exists to avoid. That is a measurement
plus a structural argument, which is what P7 requires of a rejection rather than a
reflex. If a caller ever runs this at n >= 4096, port `recall` to numpy in FACTORED
form (np.bincount scatter over pattern indices), never the dense form benchmarked
above.

E-PROP: this module does not learn. `train` is a frozenset conversion; there is no
weight update, no `.backward()`, no BPTT, no gradient of any kind. It is therefore
not on any learning path, live or otherwise.

Determinism: no function here touches the process-global RNG unless the caller
lets it. `make_patterns` / `corrupt` take an optional `rng`; `measure` always
builds its own `random.Random(seed)`. The self-check runs in-process under
`run_selfchecks.run_all_in_process` (and under pytest after promotion), where a
module-level or main()-level `random.seed()` would silently reseed every module
that runs after it.

Run: python -m <this module's dotted path>   -- the exact path is printed by the
self-check and DERIVED from `__spec__` (see `_dotted`), so relocating this file
cannot leave a stale path behind that still passes green.
"""

import random

# random.seed(0) REMOVED 2026-08-04 (promotion redesign). It was moved out of module
# scope into main() earlier the same day (specs/010, C33), which fixed the import-time
# clobber but NOT the in-process one: run_selfchecks.run_all_in_process() and pytest
# both call main() inside the shared interpreter, where random.seed(0) reseeds the
# global stream for every module that runs afterwards. The seed now lives in a
# per-instance random.Random(seed) inside measure(), which is the tree's documented
# idiom. Printed numbers shifted because the RNG stream changed (recorded in the
# promotion report); the recall/compute/storage conclusions did not.

# ---- self-check fixture dimensions ------------------------------------------
# P1: these size the DEMONSTRATION only. They arbitrate no live behaviour -- the one
# live consumer (cortex/recognizer_bank.py::StorageRecognizer) passes its own n, k and
# steps into recall() and never reads these names. They are the shape of the figure
# this file prints, not a threshold anything is judged against; every value that IS
# judged against something is measured from the same run (see `criteria`).
N = 256  # neurons (== data dimensionality, bits per pattern)
K = 20  # active neurons per pattern (sparse code, ~8% activity)
NUM_PATTERNS = 15
RECALL_STEPS = 8  # attractor settle steps cap
# P1 exception (a): a dimensionless fraction of k on a normalized quantity. Both the
# drop count and the add count derive from it, so the "corrupted cue" figure printed
# below cannot drift away from the corruption actually applied.
CUE_CORRUPTION = 0.4


def _dotted():
    """This module's dotted import path, DERIVED rather than written down.

    A promotion moves this file between regions. Any hardcoded 'snn_data_lake.X.
    spiking_storage_prototype' string would go stale on that move and keep passing
    green, so the one place the path is printed reads it off the import machinery.
    """
    spec = globals().get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return spec.name
    return __name__


# ---- 1. write data into the (factored) memory -------------------------------
def make_patterns(num, n, k, rng=None):
    """num random k-hot patterns; each is a sorted tuple of active indices.

    `rng` defaults to the process-global `random` module for backward compatibility
    with existing callers; pass a `random.Random(seed)` for isolated determinism.
    """
    r = random if rng is None else rng
    return [tuple(sorted(r.sample(range(n), k))) for _ in range(num)]


def train(patterns, n, k):
    """Store patterns in FACTORED form: the rank-P Hopfield memory is just the
    patterns themselves (as frozensets) -- no N x N matrix is ever built. Recall
    reconstructs the correlations on the fly. Signature kept for compatibility."""
    return [frozenset(p) for p in patterns]


# ---- 2. recall: noisy cue -> sparse attractor dynamics (kWTA) ----------------
def topk(values, k):
    """indices of the k largest values; deterministic tie-break by index."""
    return set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:k])


def recall(memory, cue_active, n, k, steps):
    """Iterate event-driven kWTA to a fixed point, computing the field directly
    from the stored patterns (factored, never the N x N matrix). Returns
    (final_active_set, steps_used, synops). SynOps counts the spike-driven work:
    overlap probes (P*|active|) + scatter to pattern members (P*k) per step."""
    a = k / n
    P = len(memory)
    hi2, lo2 = (1.0 - a) * (1.0 - a), a * a
    cnt = [0] * n  # cnt_i = #patterns containing neuron i
    for p in memory:
        for i in p:
            cnt[i] += 1
    active = set(cue_active)
    synops = 0
    used = 0
    for _ in range(steps):
        used += 1
        m = len(active)
        S = [0.0] * n
        for p in memory:
            d_p = sum(1 for j in active if j in p) - a * m  # |xi_p & s| - a|s|
            synops += m  # overlap probes
            if d_p == 0.0:
                continue
            for i in p:  # scatter to members
                S[i] += d_p
            synops += k
        for i in active:  # diagonal correction
            S[i] -= cnt[i] * hi2 + (P - cnt[i]) * lo2
        new_active = topk(S, k)
        if new_active == active:  # settled onto an attractor
            break
        active = new_active
    return active, used, synops


def corrupt(pattern, n, k, drop, add, rng=None):
    """Make a partial/noisy cue: remove `drop` true bits, add `add` false bits.

    `rng` defaults to the process-global `random` module for backward compatibility
    with existing callers; pass a `random.Random(seed)` for isolated determinism.
    """
    r = random if rng is None else rng
    active = list(pattern)
    r.shuffle(active)
    kept = set(active[drop:])  # drop some real bits
    inactive = [i for i in range(n) if i not in pattern]
    kept.update(r.sample(inactive, add))  # add spurious bits
    return tuple(sorted(kept))


def overlap(a, b, k):
    """fraction of the stored pattern recovered, in [0,1]."""
    return len(set(a) & set(b)) / k


# ---- 3. measure -------------------------------------------------------------
def measure(
    n=N,
    k=K,
    num_patterns=NUM_PATTERNS,
    steps=RECALL_STEPS,
    corruption=CUE_CORRUPTION,
    seed=0,
    probes=None,
):
    """Run the prototype once and return every number the criteria are judged on.

    Every argument is a real parameter, not a decoration: the RED configurations in
    RED_CONFIGS are ordinary calls to this function, which is what makes the success
    criteria falsifiable at all (P8: construct the input that makes each one fail).

    Two CONTROLS are measured alongside the recall, because "recall was high" on its
    own is a number with nothing to beat (P8: a control that is not the method):
      - `cue`  : the do-nothing control. The overlap the CORRUPTED CUE already has
                 with its target. Recall must improve on the cue, or the attractor
                 dynamics contributed nothing and the memory is a pattern list.
      - `null` : the permutation control. Recall driven by an unrelated random k-hot
                 cue, scored against the same target. This is what "recovered the
                 pattern" looks like when the cue carries no information about it.
    `probes` limits how many stored patterns are probed (default: all of them); the
    compute and storage ratios are invariant to it, since both sides scale with the
    number of settle steps actually taken.
    """
    rng = random.Random(seed)
    drop = add = round(corruption * k)
    patterns = make_patterns(num_patterns, n, k, rng)
    memory = train(patterns, n, k)
    probed = patterns if probes is None else patterns[:probes]

    recalled, cued, nulled = [], [], []
    total_synops, total_steps = 0, 0
    for p in probed:
        cue = corrupt(p, n, k, drop, add, rng)
        out, steps_used, synops = recall(memory, set(cue), n, k, steps)
        recalled.append(overlap(out, p, k))
        cued.append(overlap(cue, p, k))
        total_synops += synops
        total_steps += steps_used
        # permutation control: an unrelated cue, scored against the same target
        null_cue = tuple(sorted(rng.sample(range(n), k)))
        null_out, _, _ = recall(memory, set(null_cue), n, k, steps)
        nulled.append(overlap(null_out, p, k))

    return {
        "n": n,
        "k": k,
        "num_patterns": num_patterns,
        "probes": len(probed),
        "drop": drop,
        "add": add,
        # fraction of the stored pattern the cue is missing, DERIVED from the drop
        # actually applied -- so the printed "N% corrupted cue" cannot drift away
        # from the corruption the run used.
        "corruption": drop / k,
        "recall": sum(recalled) / len(recalled),
        "cue": sum(cued) / len(cued),
        "null": sum(nulled) / len(nulled),
        "perfect": sum(1 for o in recalled if o == 1.0),
        "steps": total_steps,
        "synops": total_synops,
        # dense Hopfield recomputes the full N x N field every settle step
        "dense_macs": total_steps * n * n,
        # the factored memory: P patterns x k indices x uint16
        "factored_bytes": num_patterns * k * 2,
        # the N x N float64 matrix the factored memory stands in for
        "dense_w_bytes": n * n * 8,
        # third honest reference point: the same patterns as dense 1-bit rows
        "bitmap_bytes": num_patterns * (-(-n // 8)),
    }


def criteria(m):
    """The success criteria, as booleans over a `measure` result.

    Each one is judged against a quantity measured in the SAME run, never against a
    fixed threshold (P1), and each one has a demonstrated red in RED_CONFIGS (P8).
    """
    return {
        # the attractor DENOISED: the settled state is closer to the stored pattern
        # than the cue it started from. This is the whole claim of the module.
        "recall_beats_cue": m["recall"] > m["cue"],
        # ... and it did so because the cue addressed that memory, not because any
        # cue lands near any pattern at this sparsity.
        "recall_beats_null": m["recall"] > m["null"],
        "compute_beats_dense": m["synops"] < m["dense_macs"],
        "storage_beats_dense": m["factored_bytes"] < m["dense_w_bytes"],
    }


# P8: a success criterion with no reachable red is worthless. Each entry is a real
# configuration where the named criteria MEASURABLY FAIL, and `check_reds` runs them.
# Both configurations overload the memory far past capacity (P=300 patterns over
# n=32 neurons); they differ in how much of the cue survives corruption, which is
# what separates the denoising criterion from the permutation-null criterion.
RED_CONFIGS = (
    (
        {"n": 32, "k": 16, "num_patterns": 300, "corruption": 0.4, "probes": 8},
        ("recall_beats_cue", "compute_beats_dense", "storage_beats_dense"),
    ),
    (
        {"n": 32, "k": 16, "num_patterns": 300, "corruption": 0.875, "probes": 16},
        ("recall_beats_null", "compute_beats_dense", "storage_beats_dense"),
    ),
)

# DELETED 2026-08-04, promotion redesign (P8: no reachable red -> delete, not weaken).
# The retired fifth assert was `aer_bytes < dense_state_bytes`, i.e. k*2 < n*4 for one
# recalled state. Patterns are drawn by random.sample(range(n), k), which RAISES for
# k > n, so k <= n holds by construction and 2k <= 2n < 4n is true for every input the
# module can be given. No configuration reds it, so it certified nothing; the AER-vs-
# dense-state point it was standing in for is arithmetic, not a measurement.


def check_reds():
    """Execute every RED_CONFIGS entry and assert the named criteria really do fail.

    This is the criteria's own sensitivity control. If a later edit weakened a
    criterion into something that cannot go red, this assert -- not the green run --
    is what catches it. Returns the list of (config, red-criteria-observed).
    """
    observed = []
    for cfg, expected_red in RED_CONFIGS:
        crit = criteria(measure(**cfg))
        went_red = tuple(sorted(name for name, ok in crit.items() if not ok))
        for name in expected_red:
            assert not crit[name], f"criterion {name} did not go red on {cfg}"
        observed.append((cfg, went_red))
    covered = {name for _cfg, exp in RED_CONFIGS for name in exp}
    missing = sorted(set(criteria(measure(probes=1))) - covered)
    assert not missing, f"criteria with no demonstrated red: {missing}"
    return observed


# ---- 4. run + report --------------------------------------------------------
def main():
    m = measure()
    crit = criteria(m)
    compute_ratio = m["dense_macs"] / m["synops"]
    storage_ratio = m["dense_w_bytes"] / m["factored_bytes"]

    print("=" * 60)
    print("SPIKING ASSOCIATIVE-MEMORY STORAGE PROTOTYPE  (v0.8 factored)")
    print("=" * 60)
    print(f"module   : {_dotted()}")
    print(
        f"neurons N={m['n']}  active K={m['k']} ({m['k'] / m['n']:.1%})  "
        f"patterns stored={m['num_patterns']}"
    )
    print()
    print(f"RECALL (content-addressable, from {m['corruption']:.0%} corrupted cue):")
    print(f"  mean overlap with stored pattern : {m['recall']:.1%}")
    print(f"  perfectly recalled               : {m['perfect']}/{m['probes']}")
    print(f"  avg settle steps                 : {m['steps'] / m['probes']:.1f}")
    print("  CONTROLS (what the recall has to beat, measured in the same run):")
    print(f"    do-nothing  : cue overlap before recall  : {m['cue']:.1%}")
    print(f"    permutation : recall from unrelated cue  : {m['null']:.1%}")
    print()
    print("COMPUTE  (lower = less power):")
    print(f"  factored SynOps  : {m['synops']:,}")
    print(f"  dense Hopfield   : {m['dense_macs']:,}  (N x N matvec each step)")
    print(f"  reduction        : {compute_ratio:.1f}x")
    print()
    print("STORAGE of the memory (lower = less space):")
    print(f"  factored (P x k indices) : {m['factored_bytes']:,} B")
    print(f"  patterns as bitmaps      : {m['bitmap_bytes']:,} B")
    print(f"  dense N x N matrix       : {m['dense_w_bytes']:,} B")
    print(f"  reduction vs dense W     : {storage_ratio:.0f}x")
    print("  note: storage is O(P*k), NOT O(N^2). The memory is just the sparse")
    print("        patterns; recall reconstructs the correlations on the fly and")
    print("        denoises corrupted cues (content-addressable). That denoising")
    print("        recall is the value over a plain pattern list.")
    print()

    # ---- self-check: assert-based, every criterion relative to a measured control ----
    assert crit["recall_beats_cue"], f"no denoising: recall {m['recall']:.4f} <= cue {m['cue']:.4f}"
    assert crit["recall_beats_null"], (
        f"content-addressing absent: recall {m['recall']:.4f} <= permutation null {m['null']:.4f}"
    )
    assert crit["compute_beats_dense"], "factored compute did not beat dense matvec"
    assert crit["storage_beats_dense"], "factored storage not smaller than dense W"

    print("REACHABLE RED (P8: each criterion above, failed on purpose):")
    for cfg, went_red in check_reds():
        shape = f"n={cfg['n']} k={cfg['k']} P={cfg['num_patterns']} corrupt={cfg['corruption']:.0%}"
        print(f"  {shape:<44} RED: {', '.join(went_red)}")
    print()
    print(
        "self-check OK: recall beat BOTH controls (do-nothing cue and permutation null), "
        "factored < dense on compute AND storage, and every criterion was shown to go red."
    )


if __name__ == "__main__":
    main()
