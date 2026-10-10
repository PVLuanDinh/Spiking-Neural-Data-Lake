"""
Unit organization -- the T1 and T4 instrument of decision C (G7 / T115, P4).

Pre-registration: specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md, 2.6-2.7 and 2.6a-2.6b
(frozen 854de0e7, amended da4bcfb7). This module DECIDES; it runs nothing confirmatory: G3's runner executes
T1's arms on G3's seeds and G4 runs T4's wiring half on this row's seeds, each handing their numbers here.

VERDICTS (judge): Part F's paired one-sided t > 1.895 through instruments/paired_contrast (one statistic,
P2), with this row's degenerate rule -- a zero-variance SUPERIORITY sample reads NOT MET (ruling 9), a
zero-variance NON-INFERIORITY sample reads by its common value (ruling 9a). Beside every verdict: the EXACT
sign-flip rate over all 2^S flips judged by THAT rule (Codex b3 #21), and paired_contrast's own rate,
labelled as the reference.

T1 (2.6, 2.6a): the size and retention differences, the strictly positive retention margin delta_NI (in
items), the per-problem granularity guard (censored values excluded; one failing problem REFUSES the whole
U contrast), and the two decision tables. A T1-struct deletion waits on the owner's O9.

T4 (2.7): the layer term read from reservoir/cortical_microcircuit.CONN_PROB (Potjans-Diesmann's 8
populations, C[target][source]; Layer 2 and Layer 3 are its L2/3, Layer 1 has no population and is
excluded, D22), symmetrized c(a, b) = 1 - (1 - p_ab)(1 - p_ba); an exact-M sequential sampler; Louvain
modularity (max over ten seeded restarts); the planted-partition and random references for IR1-IR3.
THE SAMPLER DRAWS EACH PARTNER ONE AT A TIME (a build finding against the frozen "Efraimidis-Spirakis keys"
wording, reported to the lead): a one-shot keyed draw holds every shared-partner count at 0 -- an arriving
unit has no neighbours yet -- so beta would be inert and its attribution contrast could never be credited.
With fixed weights the sequential draw has the same distribution as the keyed one. The arrival process
alone is not uniform (older units collect edges), so RULE is credited only if it beats BOTH gnm and
RANDOM-SEQ, the same sampler with every term flat (ruling 23).
networkx is imported lazily (modularity, random_graph only): it is not yet a declared dependency.
"""

from __future__ import annotations

import math
import statistics
from collections import namedtuple

import numpy as np

from snn_data_lake.gates.growth_coverage_gate import nmi  # the ONE partition-agreement home (P2)
from snn_data_lake.growth.wiring_prior import layer_term_sym
from snn_data_lake.instruments.paired_contrast import T_BAR, paired_t, permutation_fpr

#: Louvain restart seeds (estimator seeds, not recordings; pre-registration 2.10).
LOUVAIN_SEEDS = tuple(range(13900, 13910))

#: S: the paired seeds every decision is read on. T_BAR (1.895) is the one-sided t bar at S - 1 = 7 d.f.
PAIRS = 8

ContrastVerdict = namedtuple(
    "ContrastVerdict", "name kind diffs t met degenerate exact_flip_rate reference_fpr"
)


class Infeasible(ValueError):
    """An arm whose generator cannot place exactly M edges: REFUSED, never repaired (2.7)."""


def _met(diffs, kind):
    if len(set(diffs)) <= 1:  # degenerate: ruling 9 (superiority) / 9a (non-inferiority)
        return kind == "noninferiority" and bool(diffs) and diffs[0] > 0.0
    return paired_t(diffs) > T_BAR


def judge(name, diffs, kind):
    """ContrastVerdict for one paired contrast; `kind` is "superiority" or "noninferiority"."""
    if kind not in ("superiority", "noninferiority"):
        raise ValueError(f"kind must be superiority or noninferiority, got {kind!r}")
    diffs = tuple(float(d) for d in diffs)
    if len(diffs) != PAIRS:  # Codex D+E: judge("x", [1], "noninferiority") was MET
        raise ValueError(f"{name}: a decision needs exactly {PAIRS} paired seeds, got {len(diffs)}")
    if not all(
        math.isfinite(d) for d in diffs
    ):  # Codex D+E: [inf] * 8 read as a MET non-inferiority
        raise ValueError(f"{name}: every paired difference must be finite, got {diffs!r}")
    degenerate = len(set(diffs)) <= 1
    hits = 0
    for mask in range(1 << len(diffs)):
        flipped = tuple(-d if (mask >> i) & 1 else d for i, d in enumerate(diffs))
        hits += _met(flipped, kind)
    return ContrastVerdict(
        name,
        kind,
        diffs,
        0.0 if degenerate else paired_t(diffs),
        _met(diffs, kind),
        degenerate,
        hits / float(1 << len(diffs)),
        permutation_fpr(diffs),
    )


# ---- T1 --------------------------------------------------------------------------------------------------


def delta_ni(f_bound):
    """The retention margin in ITEMS of task A's future set: max(F_bound, 1 item), strictly positive."""
    if not (isinstance(f_bound, (int, float)) and math.isfinite(f_bound) and f_bound >= 0):
        raise ValueError(f"F_bound must be a finite count >= 0, got {f_bound!r}")
    return max(f_bound, 1)


def size_diffs(u_col, u_ref):
    """d_s = log 1.2 - log(U(COL) / U(ref)) per seed (the 20 % non-inferiority bar)."""
    return [math.log(1.2) - math.log(c / r) for c, r in zip(u_col, u_ref, strict=True)]


def retention_diffs(f_col, f_ref, delta):
    """e_s = delta_NI - (F(COL) - F(ref)) per seed, F in items lost; delta must be > 0."""
    if not delta > 0:
        raise ValueError(f"the retention margin must be strictly positive, got {delta!r}")
    return [delta - (c - r) for c, r in zip(f_col, f_ref, strict=True)]


def granularity_guard(pilot, k):
    """(passed, {problem: detail}). `pilot`: {problem: [U(SINGLE) per pilot seed, None if censored]}.
    U* = 10 k (worst overshoot k units <= 10 % of U). Per problem: median over SOLVED seeds >= U*, and
    no more than half censored. ANY failing problem fails the whole guard (2.6a, Codex b3 #22)."""
    ustar, detail = 10 * k, {}
    for problem, values in pilot.items():
        solved = [u for u in values if u is not None]
        censored = len(values) - len(solved)
        med = statistics.median(solved) if solved else None
        ok = bool(solved) and censored * 2 <= len(values) and med >= ustar
        detail[problem] = {"median": med, "solved": len(solved), "censored": censored, "ok": ok}
    return bool(detail) and all(d["ok"] for d in detail.values()), detail


# distinct-from: engine/autonomous_loop.py -- persists a coverage ledger; unrelated. Site read.
# distinct-from: faculties/agency_monitor.py -- per-arm reported quantities of its own rows; unrelated.
# distinct-from: faculties/evidence_accumulation.py -- a replay consumer factory; unrelated. Site read.
# distinct-from: faculties/generalizing_novelty.py -- a Spearman correlation; unrelated. Site read.
# distinct-from: brainstem/so101_arm.py -- the arm's pose-breach predicate; read only, never touched.
def censoring_refused(censored):
    """True if more than half the seeds are censored in EITHER arm (2.6); `censored` = (arm flags,
    reference flags), one bool per seed. Checked BEFORE any scoring (Codex cp3 #8)."""
    return any(sum(bool(f) for f in flags) * 2 > len(flags) for flags in censored)


def t1_struct_decision(u_verdict, e_verdict, *, owner_o9, censored):
    """2.6a's table. `u_verdict` None = the U half was REFUSED by the guard; `censored` per censoring_refused."""
    if censoring_refused(censored):
        return {
            "verdict": "REFUSED (censoring)",
            "deletes_structure": False,
            "disposition": "more than half the seeds censored in an arm: refused, reported, nothing deleted",
        }
    if u_verdict is None:
        verdict = "PASS (retention only; U untested)" if e_verdict.met else "RED"
    else:
        verdict = "PASS" if (u_verdict.met and e_verdict.met) else "RED"
    red = verdict == "RED"
    if not red:
        disposition = "column structure kept"
    elif owner_o9:
        disposition = (
            "column structure not shown non-inferior within delta_NI / 20 %: the shared "
            "inhibitory unit with its E->I and I->E wiring deleted (the neuron count stays), owner told"
        )
    else:
        disposition = "RED reported to the owner; nothing deleted until O9 is answered"
    return {"verdict": verdict, "deletes_structure": red and owner_o9, "disposition": disposition}


def t1_size_decision(u_verdict, e_verdict, *, censored):
    """T1-size: censoring refusal first; its U half REFUSED leaves the whole test UNTESTED (the retention
    half alone is count-confounded and never decides it); otherwise the record's disposition (2.6)."""
    if censoring_refused(censored):
        return {"verdict": "REFUSED (censoring)", "reverts_to_single": False}
    if u_verdict is None:
        return {"verdict": "UNTESTED", "reverts_to_single": False}
    passed = u_verdict.met and e_verdict.met
    return {"verdict": "PASS" if passed else "RED", "reverts_to_single": not passed}


# ---- T4 --------------------------------------------------------------------------------------------------


def _allotment(n, m, capacity):
    """m_t per arrival position: min(capacity_t, q), q = m // (n - 1), then one more edge each to the
    LAST arrivals with a spare candidate until the sum is exactly m."""
    q = m // max(n - 1, 1)
    alloc = [min(c, q) for c in capacity]
    rem = m - sum(alloc)
    while rem > 0:
        added = 0
        for t in range(n - 1, -1, -1):
            if rem and alloc[t] < capacity[t]:
                alloc[t] += 1
                rem -= 1
                added += 1
        if not added:
            raise Infeasible(f"cannot place {m} edges: every arrival is at capacity")
    return alloc


def _check_pathway(n, pathway):
    """(classes, n_class, kernels): per-unit pathway class, the class count, and {cell: kernel} with
    cell = class(arrival) * n_class + class(earlier unit) -- G4's `pathway_cell(pre = arrival, post = earlier)`.
    A cell absent from `kernels` reads the global kernel. Malformed -> ValueError, never repaired."""
    classes, n_class, kernels = pathway
    classes = np.asarray(classes)
    if classes.shape != (n,) or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(f"pathway classes must be {n} integers")
    if not isinstance(n_class, int) or n_class < 1 or classes.min() < 0 or classes.max() >= n_class:
        raise ValueError(f"pathway classes must lie in [0, {n_class})")
    if not isinstance(kernels, dict) or any(
        not isinstance(c, int) or not 0 <= c < n_class * n_class for c in kernels
    ):
        raise ValueError("pathway kernels must be a dict of cells in [0, n_class ** 2)")
    return classes, n_class, kernels


def sample_exact_m(n, m, *, xy, kernel, seed, beta=0.0, pops=None, columns=None, pathway=None):
    """Sorted (i, j) edges, i < j, exactly m of them, simple and undirected (2.7, frozen construction):
    units arrive in a seeded order; arrival t draws m_t partners among earlier units ONE AT A TIME,
    without replacement, with probability proportional to
        c(pop_i, pop_j) x kernel(distance) x (1 + beta x shared_partners_so_far).
    `pops` None drops the layer term (c = 1); `kernel` takes a numpy distance array; `columns` restricts
    candidates to the unit's own column (IMPOSED); `pathway` (see `_check_pathway`) replaces `kernel` cell by
    cell for a pooled Part B record. Infeasible -> REFUSED."""
    if m > n * (n - 1) // 2:
        raise Infeasible(f"M = {m} exceeds the {n * (n - 1) // 2} pairs of N = {n}")
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    xy = np.asarray(xy, dtype=float)[order]
    pop = None if pops is None else np.asarray(pops)[order]
    col = None if columns is None else np.asarray(columns)[order]
    pw = None if pathway is None else _check_pathway(n, pathway)
    cls = None if pw is None else pw[0][order]
    # ONE common allotment, computed BEFORE any column restriction: an IMPOSED arrival short of earlier
    # column mates is REFUSED below, never given fewer edges (Codex cp3 #9)
    alloc = _allotment(n, m, list(range(n)))
    nbrs = [[] for _ in range(n)]  # by arrival position
    edges = []
    for t in range(1, n):
        if not alloc[t]:
            continue
        d = np.hypot(xy[:t, 0] - xy[t, 0], xy[:t, 1] - xy[t, 1])
        w0 = np.broadcast_to(np.asarray(kernel(d), dtype=float), (t,)).copy()
        if pw is not None:
            cells = cls[t] * pw[1] + cls[:t]
            for cell in np.unique(cells):
                if int(cell) in pw[2]:
                    hit = cells == cell
                    w0[hit] = np.broadcast_to(
                        np.asarray(pw[2][int(cell)](d[hit]), dtype=float), (int(hit.sum()),)
                    )
        if pop is not None:
            w0 *= np.array([layer_term_sym(int(pop[t]), int(p)) for p in pop[:t]])
        if col is not None:
            w0[col[:t] != col[t]] = 0.0
        if not (
            np.isfinite(w0).all() and (w0 >= 0.0).all()
        ):  # Codex D+E: NaN weights drew duplicates
            raise ValueError(f"arrival {t}: kernel x layer weights must be finite and non-negative")
        if int(np.count_nonzero(w0 > 0)) < alloc[t]:
            raise Infeasible(
                f"arrival {t} has fewer positive-weight candidates than its {alloc[t]} edges"
            )
        shared = np.zeros(t)
        for _ in range(alloc[t]):
            w = w0 * (1.0 + beta * shared)
            if not (np.isfinite(w).all() and (w >= 0.0).all()):
                raise ValueError(
                    f"arrival {t}: shared-partner weights must be finite and non-negative"
                )
            peak = float(np.max(w))
            if peak > 0.0:  # Codex REQUEST 1 finding 3: a common positive scale on the weights
                # must not overflow the cumulative mass. 3 x 1e308 -> inf without this, and
                # searchsorted's last-index fallback then silently picks a different partner for
                # the same probability. Dividing by the peak holds every cumulative entry at or
                # below the arrival index, so the mass stays finite and the relative draw is
                # unchanged (a peak of 1.0 divides out exactly). peak == 0 is the exhausted-mass
                # case, which the remaining-mass check below refuses.
                w = w / peak
            cum = np.cumsum(w)
            if not (
                np.isfinite(cum[-1]) and cum[-1] > 0.0
            ):  # Codex REQUEST 1 finding 4: the loop's positivity check runs once, per arrival,
                # but the shared-partner term can zero the surviving candidates during it. Without
                # a per-draw check the step-back below walks off the front of the array
                # (index -3, size 2) instead of refusing the arm.
                raise Infeasible(
                    f"arrival {t} exhausted its positive-weight candidates before its "
                    f"{alloc[t]} edges were drawn"
                )
            j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
            j = min(j, t - 1)
            while (
                w[j] <= 0.0
            ):  # float edge case at the top of the cumsum: step back to a live candidate
                j -= 1
            w0[j] = 0.0  # without replacement
            for q in nbrs[j]:
                if q < t:
                    shared[q] += 1.0
            nbrs[j].append(t)
            nbrs[t].append(j)
            a, b = int(order[j]), int(order[t])
            edges.append((min(a, b), max(a, b)))
    assert len(set(edges)) == m, "sample_exact_m drew a duplicate edge"
    return sorted(edges)


# distinct-from: faculties/access_consciousness.py -- a workspace slot recorder; unrelated. Site read.
# distinct-from: gates/amdahl_gate.py -- interleaved harness arms for a timing ceiling; unrelated. Read.
# distinct-from: gates/determinism_gate.py -- an FP64 reference run; unrelated. Site read.
# distinct-from: gates/docs_only_battery.py -- folds instrument records into a test map; unrelated. Read.
# distinct-from: instruments/energy.py -- spiking energy terms; the matcher fired on "terms". Site read.
def random_seq(n, m, *, seed):
    """RANDOM-SEQ (ruling 23): the SAME sequential sampler with every term flat (kernel 1, no layer term,
    beta 0) -- random attachment under RULE's own arrival process, so RULE vs RANDOM-SEQ differs only in
    the rule. Positions are irrelevant under a flat kernel."""
    return sample_exact_m(n, m, xy=np.zeros((n, 2)), kernel=lambda d: 1.0, seed=seed)


def t4_rule_credited(vs_gnm, vs_seq):
    """RULE is credited (T4's wiring half PASSES) only if it beats BOTH random arms: gnm (the record's
    literal red) AND RANDOM-SEQ (the arrival-process control) -- ruling 23."""
    return bool(vs_gnm.met and vs_seq.met)


def planted_blocks(n, m, *, blocks, between, seed):
    """IR1/IR3's reference: exactly m simple edges, round(m x between) of them across blocks of a
    contiguous `blocks`-way partition, the rest within; uniform over each pair class."""
    rng = np.random.default_rng(seed)
    label = np.arange(n) * blocks // n
    m_between = round(m * between)
    within_pairs = int(sum(c * (c - 1) // 2 for c in np.bincount(label, minlength=blocks)))
    between_pairs = n * (n - 1) // 2 - within_pairs
    if m_between > between_pairs or m - m_between > within_pairs:  # Codex D+E: the loop never ended
        raise Infeasible(
            f"planted_blocks: {m - m_between} within / {m_between} between edges exceed the "
            f"{within_pairs} / {between_pairs} pairs of N = {n} in {blocks} blocks"
        )
    edges = set()
    while len(edges) < m:
        want_between = len(edges) < m_between
        i, j = (int(v) for v in rng.integers(0, n, size=2))
        if i == j or (label[i] != label[j]) != want_between:
            continue
        edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def random_graph(n, m, *, seed):
    """RANDOM: networkx.gnm_random_graph(N, M, seed) -- simple, undirected, exactly M."""
    import networkx as nx

    if m > n * (n - 1) // 2:  # gnm would return the complete graph, silently short of M
        raise Infeasible(f"M = {m} exceeds the {n * (n - 1) // 2} pairs of N = {n}")
    return sorted((min(a, b), max(a, b)) for a, b in nx.gnm_random_graph(n, m, seed=seed).edges())


def _triangles(edges, n):
    nb = [set() for _ in range(n)]
    for i, j in edges:
        nb[i].add(j)
        nb[j].add(i)
    return sum(len(nb[i] & nb[j]) for i, j in edges) // 3


# distinct-from: instruments/attention_schema.py -- its t078_reds line; that module's own reds. Read.
# distinct-from: instruments/binding_instrument.py -- its t080_reds line; that module's own reds. Read.
# distinct-from: instruments/report_order_probe.py -- its t081_reds line; that module's own reds. Read.
# distinct-from: faculties/agency_monitor.py -- its t079_reds line; that module's own reds. Read.
# distinct-from: faculties/metacognitive_monitor.py -- its t077_reds line; that module's own reds. Read.
def main():
    """Bare self-check: six instrument reds, each of which must fire (networkx-free; no Brain, no lake)."""

    def raises(fn):
        try:
            fn()
        except (ValueError, Infeasible):
            return True
        return False

    xy = np.random.default_rng(13237).random((300, 2)) * 1000.0
    tri = [
        sum(
            _triangles(sample_exact_m(300, 1200, xy=xy, kernel=lambda d: 1.0, seed=s, beta=b), 300)
            for s in (13233, 13234, 13235)
        )
        for b in (0.0, 5.0)
    ]
    reds = {
        "degenerate superiority NOT MET (ruling 9)": not judge("r", [0.3] * 8, "superiority").met,
        "exact flip rate 8/256 on a constant-positive sample": judge(
            "r", [0.5] * 8, "superiority"
        ).exact_flip_rate
        == 8 / 256,
        "guard refuses a small judging problem": not granularity_guard({"A": [2, 3, 4]}, k=8)[0],
        "a zero retention margin is refused": raises(lambda: retention_diffs([0], [0], delta=0)),
        "an infeasible density is refused": raises(
            lambda: sample_exact_m(10, 46, xy=np.zeros((10, 2)), kernel=lambda d: 1.0, seed=1)
        ),
        "the shared-partner term is live": tri[1] > 2 * tri[0],
    }
    silent = [k for k, fired in reds.items() if not fired]
    print(
        f"instrument reds fired: {len(reds) - len(silent)} of {len(reds)}  (triangles beta 0 vs 5: {tri})"
    )
    assert not silent, f"reds that did not fire: {silent}"
    return 0


# distinct-from: gates/growth_coverage_gate.py -- partition agreement is NOT computed here: IR1 reads
#   growth_coverage_gate.nmi, which lives beside adjusted_rand (the one home, P2; lead ruling on cp3).
#   This module only partitions (Louvain); the bar stays the FROZEN NMI >= 0.9 (2.7).
# distinct-from: instruments/paired_contrast.py, faculties/self_model.py, reservoir/kernel_tick.py,
#   substrate/lake_bridge.py -- a self-check, a self-model, a partition drain, a digest round trip;
#   the matcher fired on "partition" / "own". All sites read.
def best_partition(edges, n, seeds=LOUVAIN_SEEDS):
    """(Q, labels): the Louvain partition (resolution 1.0) with the MAXIMUM Newman Q over seeded restarts,
    as one community label per node; unweighted."""
    import networkx as nx

    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("Louvain needs at least one restart seed")
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    if (
        g.number_of_edges() == 0
    ):  # Newman Q divides by the degree sum: undefined, refused not crashed
        raise Infeasible("modularity is undefined on a graph with no edges")
    best = None
    for s in seeds:
        comms = nx.community.louvain_communities(g, resolution=1.0, seed=s)
        q = nx.community.modularity(g, comms)
        if best is None or q > best[0]:
            labels = [0] * n
            for c, members in enumerate(comms):
                for v in members:
                    labels[v] = c
            best = (q, labels)
    return best


def modularity(edges, n, seeds=LOUVAIN_SEEDS):
    """Newman Q of the Louvain partition, the maximum over seeded restarts (best_partition's Q)."""
    return best_partition(edges, n, seeds)[0]


def t4_wiring_half(
    n, m, *, xy, pops, kernel, beta, columns, seeds, louvain_seeds=LOUVAIN_SEEDS, pathway=None
):
    """T4's wiring half (2.7) on caller-supplied substrate: per seed, Q (Louvain, max over restarts) of every
    arm at the same N and EXACTLY M edges. `kernel` is G4's tangential kernel (a callable on a distance array; `pathway` its per-cell form for a pooled record),
    `beta` G4's Part B2 value (0.0 = the clause reported untested). Returns {"refused": [...], "q": {arm:
    [Q per seed]}, "verdicts": {...}, "rule_credited": bool|None}. A refused RULE / RANDOM / RANDOM-SEQ arm refuses
    the half (rule_credited None); a refused attribution variant leaves its term NOT CLAIMED; IMPOSED is
    reported and decides nothing. The caller runs ir1/ir2 first (t4_instruments_refuse)."""
    # IMPOSED is a restricted arm: the caller declares its column mapping, and an undeclared one is
    # refused rather than degrading into a second copy of RULE (a G7 lens MINOR). `columns` is None
    # elsewhere as "no restriction"; here it is a missing declaration.
    if columns is None:
        raise ValueError("IMPOSED needs columns")
    flat = lambda d: 1.0  # noqa: E731 -- a removed kernel term
    seeds, louvain_seeds = (
        tuple(seeds),
        tuple(louvain_seeds),
    )  # read once: an iterator is spent by arm 1
    if not louvain_seeds:
        raise ValueError("Louvain needs at least one restart seed")
    arms = {
        "RULE": dict(kernel=kernel, pops=pops, beta=beta, pathway=pathway),
        "DIST-ONLY": dict(kernel=kernel, pops=pops, beta=0.0, pathway=pathway),
        "FLAT-KERNEL": dict(kernel=flat, pops=pops, beta=beta),
        "NO-LAYER": dict(kernel=kernel, pops=None, beta=beta, pathway=pathway),
        "IMPOSED": dict(kernel=kernel, pops=pops, beta=beta, columns=columns, pathway=pathway),
    }
    q, refused = {}, []
    for name, kw in arms.items():
        try:
            q[name] = [
                modularity(sample_exact_m(n, m, xy=xy, seed=s, **kw), n, louvain_seeds)
                for s in seeds
            ]
        except Infeasible:
            refused.append(name)
    try:
        q["RANDOM"] = [modularity(random_graph(n, m, seed=s), n, louvain_seeds) for s in seeds]
        q["RANDOM-SEQ"] = [modularity(random_seq(n, m, seed=s), n, louvain_seeds) for s in seeds]
    except Infeasible:
        refused += [a for a in ("RANDOM", "RANDOM-SEQ") if a not in q]

    def diff(a, b):
        return [x - y for x, y in zip(q[a], q[b], strict=True)]

    out = {"refused": refused, "q": q, "verdicts": {}, "rule_credited": None}
    if {"RULE", "RANDOM", "RANDOM-SEQ"} & set(refused):
        return out  # the half is REFUSED, nothing is deleted
    out["verdicts"] = {
        "RULE>RANDOM": judge("RULE vs RANDOM", diff("RULE", "RANDOM"), "superiority"),
        "RULE>RANDOM-SEQ": judge("RULE vs RANDOM-SEQ", diff("RULE", "RANDOM-SEQ"), "superiority"),
    }
    for term, ref in (
        ("shared-partner", "DIST-ONLY"),
        ("distance", "FLAT-KERNEL"),
        ("layer", "NO-LAYER"),
    ):
        if ref in q:  # a refused variant leaves its term NOT CLAIMED (absent here)
            out["verdicts"][term] = judge(f"RULE vs {ref}", diff("RULE", ref), "superiority")
    out["rule_credited"] = t4_rule_credited(
        out["verdicts"]["RULE>RANDOM"], out["verdicts"]["RULE>RANDOM-SEQ"]
    )
    return out


#: IR seeds (dev block, pre-registration 2.10): the planted / RANDOM pairs; RANDOM's TWIN draws from the
#: second range (13240-13247, the build's addition, disclosed in the dated amendment).
IR_SEEDS = tuple(range(13230, 13238))
IR_TWIN_SEEDS = tuple(range(13240, 13248))
NMI_BAR = 0.9  # CHOSEN, disclosed (2.7)


def ir1(n, m, *, blocks, between=0.2, seeds=IR_SEEDS, louvain_seeds=LOUVAIN_SEEDS):
    """IR1: a planted block model at the same N and M must beat RANDOM (gnm) by Part F's contrast AND its
    Louvain partition must recover the plant (NMI >= NMI_BAR on EVERY seed). Fails -> T4 REFUSED."""
    seeds, louvain_seeds = tuple(seeds), tuple(louvain_seeds)
    truth = list(np.arange(n) * blocks // n)
    diffs, nmis = [], []
    for s in seeds:
        q_p, labels = best_partition(
            planted_blocks(n, m, blocks=blocks, between=between, seed=s), n, louvain_seeds
        )
        diffs.append(q_p - modularity(random_graph(n, m, seed=s), n, louvain_seeds))
        nmis.append(nmi(truth, labels))
    verdict = judge("IR1 planted vs RANDOM", diffs, "superiority")
    return {"verdict": verdict, "nmi": nmis, "ok": verdict.met and min(nmis) >= NMI_BAR}


def ir2(n, m, *, seeds=IR_SEEDS, twin_seeds=IR_TWIN_SEEDS, louvain_seeds=LOUVAIN_SEEDS):
    """IR2, the false-positive control: RANDOM against a RANDOM twin must NOT pass. Passes -> T4 REFUSED."""
    seeds, twin_seeds, louvain_seeds = tuple(seeds), tuple(twin_seeds), tuple(louvain_seeds)
    diffs = [
        modularity(random_graph(n, m, seed=s), n, louvain_seeds)
        - modularity(random_graph(n, m, seed=t), n, louvain_seeds)
        for s, t in zip(seeds, twin_seeds, strict=True)
    ]
    verdict = judge("IR2 RANDOM vs RANDOM twin", diffs, "superiority")
    return {"verdict": verdict, "ok": not verdict.met}


def ir3(n, m, *, blocks, seeds=IR_SEEDS, louvain_seeds=LOUVAIN_SEEDS):
    """IR3: the planted model at 50 % between-block edges vs RANDOM -- REPORTED only, to locate the floor."""
    seeds, louvain_seeds = tuple(seeds), tuple(louvain_seeds)
    diffs = [
        modularity(planted_blocks(n, m, blocks=blocks, between=0.5, seed=s), n, louvain_seeds)
        - modularity(random_graph(n, m, seed=s), n, louvain_seeds)
        for s in seeds
    ]
    return {"verdict": judge("IR3 planted (50% between) vs RANDOM", diffs, "superiority")}


def t4_instruments_refuse(ir1_result, ir2_result):
    """T4 is REFUSED (before any arm is scored) if IR1 or IR2 fails (2.7)."""
    return not (ir1_result["ok"] and ir2_result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
