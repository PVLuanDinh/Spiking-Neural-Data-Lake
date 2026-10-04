"""specs/022 T111 (G3) P1: the capacity certificate -- the growth trigger's three-way diagnosis.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.4. Over a problem's taught items,
on the taught learner's CURRENT code:

- k = the number of items whose full code equals another item's code under a different label;
- k > 0 and k = 0 on the history-augmented code -> STATE (memory kinds); with the certificate also firing on
  the history code in the same pass -> STATE+CAPACITY;
- k > 0 with history too -> DATA (contradictory: no growth, the problem stays OPEN);
- k = 0 and an EXACT Farkas witness on the (synapse, rank) one-hot lift -> CAPACITY (growth licensed);
- k = 0 and a separating weight setting that reads every item right through the REAL `route_text` -> SEPARABLE
  (not a capacity deficit);
- k = 0, no lift witness and no verified separator -> UNRESOLVED (no growth licensed; Codex b1 G3 #8).

Soundness: the learner's drive is v = sum_i w_i * PSP(t_post - rank_i), a linear map of the one-hot lift over
EVERY (synapse, rank) pair, so a lift witness proves inseparability for ANY PSP kernel and any t_post. The lift
is therefore never truncated at t_post. When the lift admits no witness the certificate does not fire
(conservative). The LP only FINDS the witness; it is accepted only after an independent check in exact
`Fraction` arithmetic. Pure functions: no RNG, no lake, no Brain.
"""

import copy
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from snn_data_lake.readout.interaction_learning import text_features

_TOL = 1e-9  # LP support threshold: a multiplier below it is treated as zero before the exact check


def item_ranks(learner, texts):
    """The learner's frozen rank code (one integer latency per input synapse) for each text."""
    return [tuple(learner.encode(text_features(t, learner.n_in))) for t in texts]


def lift_keys(ranks, grown_latencies=()):
    """The one-hot lift of one item: every (synapse, rank) pair of the frozen code, then one
    (grown unit, latency) pair per grown unit in unit_id order (latency None = no spike)."""
    keys = [("f", i, int(r)) for i, r in enumerate(ranks)]
    keys += [("g", j, lat) for j, lat in enumerate(grown_latencies)]
    return tuple(keys)


def contradiction_count(keys_list, labels):
    """k: the number of items whose code equals another item's code under a different label."""
    by_code = {}
    for keys, y in zip(keys_list, labels):
        by_code.setdefault(keys, set()).add(y)
    return sum(1 for keys in keys_list if len(by_code[keys]) > 1)


def _constraints(labels, n_labels):
    """The multiclass constraints (item i, competitor c != y_i), in a fixed order."""
    return [(i, c) for i, y in enumerate(labels) for c in range(n_labels) if c != y]


def check_farkas(keys_list, labels, lam):
    """Exact check: lam >= 0, sum > 0, and sum lam_ic (e_{y_i} - e_c) (x) lift_i == 0 coordinate-wise."""
    if not lam or any(v < 0 for v in lam.values()) or sum(lam.values()) <= 0:
        return False
    acc = {}
    for (i, c), v in lam.items():
        if v == 0:
            continue
        y = labels[i]
        if c == y:
            return False
        for key in keys_list[i]:
            acc[(y, key)] = acc.get((y, key), Fraction(0)) + v
            acc[(c, key)] = acc.get((c, key), Fraction(0)) - v
    return all(v == 0 for v in acc.values())


@dataclass
class FarkasWitness:
    lam: dict
    path: str  # "rational" (rounded LP solution verified) or "support" (exact solve on the LP's support)


def _farkas_system(keys_list, labels, n_labels):
    cons = _constraints(labels, n_labels)
    row_of = {}
    entries = []  # (row, col, value)
    for col, (i, c) in enumerate(cons):
        y = labels[i]
        for key in keys_list[i]:
            for lab, sgn in ((y, 1.0), (c, -1.0)):
                r = row_of.setdefault((lab, key), len(row_of))
                entries.append((r, col, sgn))
    return cons, row_of, entries


def _exact_support_solve(cons, entries, n_rows, support):
    """Solve A_s lam_s = 0, sum lam_s = 1 exactly on the support columns (Gaussian elimination in Fractions).
    Returns the unique solution, or None when it is not unique or not consistent."""
    cols = {c: k for k, c in enumerate(support)}
    rows = {}
    for r, c, v in entries:
        if c in cols:
            rows.setdefault(r, [Fraction(0)] * (len(support) + 1))[cols[c]] += Fraction(int(v))
    mat = [row for row in rows.values() if any(row[:-1])]
    mat.append([Fraction(1)] * len(support) + [Fraction(1)])
    n = len(support)
    piv_row = 0
    pivots = []
    for col in range(n):
        p = next((r for r in range(piv_row, len(mat)) if mat[r][col] != 0), None)
        if p is None:
            continue
        mat[piv_row], mat[p] = mat[p], mat[piv_row]
        inv = 1 / mat[piv_row][col]
        mat[piv_row] = [x * inv for x in mat[piv_row]]
        for r in range(len(mat)):
            if r != piv_row and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[piv_row])]
        pivots.append(col)
        piv_row += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in mat):
        return None  # inconsistent
    if len(pivots) < n:
        return None  # not unique: the LP point was not a vertex of the support
    sol = [Fraction(0)] * n
    for r, col in enumerate(pivots):
        sol[col] = mat[r][-1]
    return {cons[c]: sol[k] for c, k in cols.items()}


def farkas_witness(keys_list, labels, n_labels):
    """Find multipliers proving the multiclass constraints infeasible on the lift; None if the LP finds none
    or no candidate passes the exact check (the certificate then does not fire)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    cons, row_of, entries = _farkas_system(keys_list, labels, n_labels)
    if not cons:
        return None
    n_rows = len(row_of)
    r_idx = [e[0] for e in entries] + [n_rows] * len(cons)
    c_idx = [e[1] for e in entries] + list(range(len(cons)))
    vals = [e[2] for e in entries] + [1.0] * len(cons)
    a_eq = coo_matrix((vals, (r_idx, c_idx)), shape=(n_rows + 1, len(cons))).tocsr()
    b_eq = np.zeros(n_rows + 1)
    b_eq[n_rows] = 1.0
    res = linprog(np.zeros(len(cons)), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    x = res.x
    lam = {
        cons[k]: Fraction(float(v)).limit_denominator(10**6) for k, v in enumerate(x) if v > _TOL
    }
    if check_farkas(keys_list, labels, lam):
        return FarkasWitness(lam, "rational")
    support = [k for k, v in enumerate(x) if v > _TOL]
    exact = _exact_support_solve(cons, entries, n_rows, support)
    if exact is not None and check_farkas(keys_list, labels, exact):
        return FarkasWitness(exact, "support")
    return None


@dataclass
class Separator:
    weights: list  # one row per label neuron, n_in floats
    reads: list  # route_text of each item through a learner carrying `weights`


def _psp_rows(learner, texts):
    return [
        [learner._psp(learner.t_post - r) for r in ranks] for ranks in item_ranks(learner, texts)
    ]


def separating_witness(learner, texts, labels, margin=1.0):
    """A weight setting that reads every item right through the REAL float `route_text` (G2's witness
    pattern), or None. Found by the Kesler LP on the PSP code; accepted only on the real reader's reads.
    The caller's learner is never modified (a deep copy carries the weights)."""
    from scipy.optimize import linprog

    phis = np.array(_psp_rows(learner, texts))
    n_lab, d = len(learner.responses), phis.shape[1]
    ys = [learner.responses.index(lab) for lab in labels]
    rows = []
    for phi, y in zip(phis, ys):
        for c in range(n_lab):
            if c != y:
                row = np.zeros(n_lab * d)
                row[y * d : (y + 1) * d] -= phi
                row[c * d : (c + 1) * d] += phi
                rows.append(row)
    res = linprog(
        np.zeros(n_lab * d),
        A_ub=np.array(rows),
        b_ub=np.full(len(rows), -margin),
        bounds=(None, None),
        method="highs",
    )
    if res.status != 0:
        return None
    w = res.x.reshape(n_lab, d)
    probe = copy.deepcopy(learner)
    for li, neuron in enumerate(probe.neurons):
        neuron.w = [float(v) for v in w[li]]
    reads = [probe.route_text(t) for t in texts]
    if reads != list(labels):
        return None
    return Separator([list(map(float, r)) for r in w], reads)


@dataclass
class Diagnosis:
    kind: str  # CAPACITY | SEPARABLE | UNRESOLVED | STATE | STATE+CAPACITY | DATA
    k: int
    k_history: int = None
    witness: FarkasWitness = None
    separator: Separator = None
    log: dict = field(default_factory=dict)


def diagnose(learner, texts, labels, history_texts=None):
    """The three-way diagnosis on the learner's frozen code (2.4). Labels are response names."""
    ys = [learner.responses.index(lab) for lab in labels]
    n_lab = len(learner.responses)
    keys = [lift_keys(r) for r in item_ranks(learner, texts)]
    k = contradiction_count(keys, ys)
    if k > 0:
        if history_texts is None:
            return Diagnosis("DATA", k, log={"history": "absent"})
        hkeys = [lift_keys(r) for r in item_ranks(learner, history_texts)]
        kh = contradiction_count(hkeys, ys)
        if kh > 0:
            return Diagnosis("DATA", k, kh)
        w = farkas_witness(hkeys, ys, n_lab)  # same pass: state AND capacity (D-G2-5)
        return Diagnosis("STATE+CAPACITY" if w else "STATE", k, kh, witness=w)
    w = farkas_witness(keys, ys, n_lab)
    if w is not None:
        return Diagnosis("CAPACITY", 0, witness=w, log={"path": w.path})
    sep = separating_witness(learner, texts, labels)
    if sep is not None:
        return Diagnosis("SEPARABLE", 0, separator=sep)
    return Diagnosis("UNRESOLVED", 0)


def main():
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    intents = ["GREET", "GOAL", "ARM", "CONSCIOUS", "STATUS", "MEMORY", "SLEEP", "HELP", "OTHER"]
    texts = ["cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun"]
    xor = ["GREET", "GOAL", "GOAL", "GREET"]
    twin = ["GREET", "GREET", "GOAL", "GOAL"]
    learner = SpikingInteractionLearner(intents, n_in=52, seed=9900, t_post=6)
    d = diagnose(learner, texts, xor)
    assert d.kind == "CAPACITY", d.kind
    keys = [lift_keys(r) for r in item_ranks(learner, texts)]
    ys = [intents.index(x) for x in xor]
    bad = dict(d.witness.lam)
    first = next(iter(bad))
    bad[first] *= 2
    assert not check_farkas(keys, ys, bad), "red: a doubled multiplier must fail the exact check"
    assert diagnose(learner, texts, twin).kind == "SEPARABLE"
    print(
        f"capacity_certificate: XOR CAPACITY (path {d.witness.path}), twin SEPARABLE, doubled-multiplier red fires"
    )


if __name__ == "__main__":
    main()
