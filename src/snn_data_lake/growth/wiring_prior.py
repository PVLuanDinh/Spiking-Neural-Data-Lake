"""The anatomical attachment rule (G4 / T112, g4-wiring-prior.md 3.2): the interface G3 and G7 consume.

p(attach u -> v) = layer_target(u, v) x kernel(tangential distance) x (1 + beta x shared partners), u
presynaptic, v postsynaptic. layer_target reads `cortical_microcircuit.CONN_PROB[target][source]` (its
first reader, P3); the kernel is a PART B record's (tangential frame) mixture shape, normalised to 1 at
d = 0 so the layer target carries the amplitude; beta comes from the record (Part B2) and defaults to 0.

No per-unit field is stored: units are `placement.place_labeled` label dicts, partners a caller-owned
{unit_id: set of ids}. Distance is within ONE map cell (equal (ap, dv)); across cells p = 0. Depth never
enters the kernel (D1). Refused, never defaulted: T3's 3-D soma-distance record (wrong frame), a record
not from Part B, and layers CONN_PROB has no population for (L1, white matter), whose disposition is
owed to Part C's amendment. No hand width exists here: a record is required.
"""

import math

import numpy as np

from snn_data_lake.growth.placement import LAYER_NAMES
from snn_data_lake.reservoir.cortical_microcircuit import CONN_PROB

# distinct-from: reservoir/cortical_microcircuit.py -- owns the population table this reads; no kernel there.
# distinct-from: growth/arealization.py::OrganizerField -- a 1-D positional-information field, not a pair kernel.


class RecordRefused(ValueError):
    """The record is not a Part B tangential record."""


class LayerNotModelled(ValueError):
    """CONN_PROB has no population for this layer (L1, white matter)."""


_L23, _L4, _L5, _L6 = 0, 2, 4, 6  # CONN_PROB population base index (excitatory; +1 = inhibitory)
_BASE = {"Layer 2": _L23, "Layer 3": _L23, "Layer 4": _L4, "Layer 5": _L5, "Layer 6": _L6}


def population(layer, sign):
    """CONN_PROB population index of (layer, sign); layer is a LAYER_NAMES name or its index."""
    name = LAYER_NAMES[layer] if isinstance(layer, (int, np.integer)) else layer
    if name not in _BASE:
        raise LayerNotModelled(f"{name!r}: no CONN_PROB population (Part C amendment owes L1)")
    return _BASE[name] + (0 if sign > 0 else 1)


def _check(record):
    if record.get("source") != "part_b" or record.get("frame") != "tangential":
        raise RecordRefused("a Part B tangential record is required (T3's 3-D record is not one)")
    return record


N_CLASS = 14  # Part B's pathway classes: 7 bands x 2 signs


def pathway_cell(pre, post):
    """Part B's cell id for a (source, target) pair of label records: pre class * 14 + post class, class =
    band index * 2 + (sign > 0); band index = LAYER_NAMES position (the artifact's layer code)."""

    def cls(u):
        li = (
            u["layer"]
            if isinstance(u["layer"], (int, np.integer))
            else LAYER_NAMES.index(u["layer"])
        )
        return int(li) * 2 + (1 if u["sign"] > 0 else 0)

    return cls(pre) * N_CLASS + cls(post)


def _widths(rec, cell):
    """(sigma_um, lambda_um) of a pathway when the record kept the pooled widths, else the global ones. A pooled
    record MUST carry its `pathway_widths` mapping (a pathway absent from it kept the global width: zero offset);
    a pooled record without the field is refused, never served the global kernel."""
    if cell is not None and rec.get("keep") == "pooled":
        if not isinstance(rec.get("pathway_widths"), dict):
            raise RecordRefused("a pooled Part B record carries no pathway_widths mapping")
        w = rec["pathway_widths"]
        hit = w.get(cell, w.get(str(cell)))  # JSON round trips turn int keys into strings
        if hit is not None and not hit:
            # distinct from `not hit`: an EMPTY entry names a pathway the record carries and the
            # global shape must not be substituted for it (G7 lens MINOR; docstring forbids it).
            raise RecordRefused(f"pathway_widths[{cell!r}] is empty: a named pathway is required")
        if hit:
            return hit[0], hit[1]
    return rec["sigma_um"], rec["lambda_um"]


def kernel_shape(record, cell=None):
    """f(d) on a numpy distance array (um): the record's kernel, 1 at d = 0; only relative values matter. With
    `cell` and a pooled record, the pathway's own widths (B2: the delivered kernel is the one that was read)."""
    rec = _check(record)
    a, c = rec["a"], rec["c"]
    sg, lam = _widths(rec, cell)
    return lambda d: (
        (a * np.exp(-(np.asarray(d) ** 2) / (2.0 * sg**2)) + c * np.exp(-np.asarray(d) / lam))
        / (a + c)
    )


def layer_term(source_pop, target_pop):
    """Directed layer target: CONN_PROB[target][source] (population indices)."""
    return CONN_PROB[target_pop][source_pop]


def layer_term_sym(a_pop, b_pop):
    """Undirected layer target for symmetrised graphs: an edge if either direction exists, 1-(1-C[a][b])(1-C[b][a])."""
    return 1.0 - (1.0 - CONN_PROB[a_pop][b_pop]) * (1.0 - CONN_PROB[b_pop][a_pop])


def xy(units):
    """(N, 2) tangential positions (x_um, y_um) of label records."""
    return np.array([[u["position"][2], u["position"][3]] for u in units], dtype=float)


def attach_probabilities(new_units, existing_units, record, partners=None):
    """Array [len(new_units), len(existing_units)] of p(new -> existing), each in [0, 1]."""
    rec = _check(record)
    beta = rec.get("beta", 0.0)
    shapes = {}
    partners = partners or {}
    out = np.zeros((len(new_units), len(existing_units)))
    for i, u in enumerate(new_units):
        pu = population(u["layer"], u["sign"])
        ua, ud, ux, uy, _ = u["position"]
        for j, v in enumerate(existing_units):
            va, vd, vx, vy, _ = v["position"]
            if (ua, ud) != (va, vd):
                continue
            target = layer_term(pu, population(v["layer"], v["sign"]))
            m = len(set(partners.get(u["unit_id"], ())) & set(partners.get(v["unit_id"], ())))
            cell = pathway_cell(u, v)
            if cell not in shapes:
                shapes[cell] = kernel_shape(rec, cell)
            out[i, j] = target * shapes[cell](np.hypot(ux - vx, uy - vy)) * (1.0 + beta * m)
    return np.clip(out, 0.0, 1.0)


def main():
    """No-argument self-check (run_selfchecks CORE): SYNTHETIC label records, no H01 file, no Brain,
    no lake, no seed, no deciding number. Every expected number below is written out from the CONN_PROB
    row values (pops 2/3/4) and plain arithmetic, NOT recomputed through this module, so a break in
    population / layer_term / pathway_cell / kernel_shape / the clip or the beta term reds here.

    ASCII-only, deterministic, and it RAISES (does not print success quietly) on any broken state.
    """
    rec = {
        "source": "part_b",
        "frame": "tangential",
        "a": 1.0,
        "sigma_um": 50.0,
        "c": 0.0,
        "lambda_um": 100.0,
    }

    def u(uid, x, y, layer="Layer 4", sign=1, cell=(0, 0)):
        return {
            "unit_id": uid,
            "position": (cell[0], cell[1], x, y, 0.5),
            "layer": layer,
            "sign": sign,
        }

    # -- population: LAYER_NAMES index -> CONN_PROB base, +1 for the inhibitory partner.
    assert population("Layer 4", 1) == 2 and population("Layer 4", -1) == 3
    assert population("Layer 5", 1) == 4 and population("Layer 6", -1) == 7
    assert population(3, 1) == 2, "a layer INDEX must resolve like its name"
    assert population("Layer 2", 1) == population("Layer 3", 1) == 0, (
        "L2/L3 aggregate to one population"
    )
    for unmodelled in ("Layer 1", "White matter"):
        try:
            population(unmodelled, 1)
            raise AssertionError(f"{unmodelled} was served a population instead of being refused")
        except LayerNotModelled:
            pass

    # -- layer_term is DIRECTED: CONN_PROB[target][source]. These two cells are 0.0505 and 0.0067,
    #    so a swapped read is caught by the pair, not by one of them matching by luck.
    assert layer_term(2, 4) == CONN_PROB[4][2] == 0.0505
    assert layer_term(4, 2) == CONN_PROB[2][4] == 0.0067
    assert layer_term(2, 4) != layer_term(4, 2), "the layer target is directed"
    assert math.isclose(layer_term_sym(2, 4), 1.0 - (1.0 - 0.0067) * (1.0 - 0.0505))
    assert layer_term_sym(2, 4) == layer_term_sym(4, 2)

    # -- pathway_cell: pre class * 14 + post class, class = LAYER_NAMES index * 2 + (sign > 0).
    assert N_CLASS == 14
    assert pathway_cell(u("a", 0, 0, "Layer 4", 1), u("b", 0, 0, "Layer 4", 1)) == 105  # 7 * 14 + 7
    assert pathway_cell(u("a", 0, 0, "Layer 2", -1), u("b", 0, 0, "Layer 6", 1)) == 2 * 14 + 11
    assert (
        pathway_cell(u("a", 0, 0, "Layer 6", 1), u("b", 0, 0, "Layer 6", 1)) == 11 * 14 + 11
    )  # LAYER_NAMES index 5 -> class 11

    # -- kernel_shape: normalised to 1 at d = 0 (the layer target carries the amplitude), and
    #    monotonically falling. exp(-0.32) at d = 40 um is the value the matrix below is built on.
    f = kernel_shape(rec)
    d = np.array([0.0, 40.0, 400.0])
    got = f(d)
    assert got[0] == 1.0, f"the kernel must carry the amplitude, got {got[0]!r} at d = 0"
    assert np.all(np.diff(got) < 0), "the kernel must fall with tangential distance"
    assert math.isclose(float(got[1]), math.exp(-(40.0**2) / (2 * 50.0**2)))
    assert float(got[2]) < 1e-2, "a 400 um separation must be far out on the tail"

    # The normalisation is only observable when the mixture weights do NOT already sum to 1. With
    # a = 1, c = 0 above, a + c == 1 and dropping "/ (a + c)" changes no number at all -- so the
    # pure-Gaussian record cannot see that term, and this mixture (a = 2, c = 0.5, a + c = 2.5) can.
    mix = kernel_shape(dict(rec, a=2.0, c=0.5, lambda_um=80.0))
    assert float(mix(np.array([0.0]))[0]) == 1.0, (
        f"a {2.0 + 0.5}-weight mixture must still carry the amplitude at d = 0, got "
        f"{float(mix(np.array([0.0]))[0])!r} -- the /(a + c) normalisation is what makes it 1"
    )
    assert math.isclose(
        float(mix(np.array([40.0]))[0]),
        (2.0 * math.exp(-(40.0**2) / (2 * 50.0**2)) + 0.5 * math.exp(-40.0 / 80.0)) / 2.5,
        rel_tol=1e-12,
    )
    assert float(mix(np.array([40.0]))[0]) != float(kernel_shape(rec)(np.array([40.0]))[0]), (
        "the exponential tail must actually change the shape when c != 0"
    )

    # -- the attachment computation, on a 2 x 3 population of known units, cell (0, 0) throughout.
    #    Row pops: both presynaptic L4e (2). Column pops: L4e (2), L5e (4), L4i (3).
    #    Each entry is CONN_PROB[col][2] * exp(-(d^2) / 5000); expected values written out by hand:
    #      [0, 0] -> 0.0497 * 1                      = 0.0497
    #      [0, 40] -> 0.0505 * exp(-0.32)             = 0.036670526372221394
    #      [0,  0] -> 0.0794 * 1                     = 0.0794   (the INHIBITORY target row)
    #      [40, 0] -> 0.0497 * exp(-0.32)             = 0.036089607142562444
    #      [40, 40] -> 0.0505 * 1                    = 0.0505
    #      [40, 0] -> 0.0794 * exp(-0.32)             = 0.05765623354365106
    new_units = [u("n0", 0, 0), u("n1", 40, 0)]
    old_units = [u("o0", 0, 0), u("o1", 40, 0, "Layer 5", 1), u("o2", 0, 0, "Layer 4", -1)]
    expected = [
        [0.0497, 0.036670526372221394, 0.0794],
        [0.036089607142562444, 0.0505, 0.05765623354365106],
    ]
    snap = (repr(new_units), repr(old_units), repr(rec))
    p = attach_probabilities(new_units, old_units, rec)
    for i in range(2):
        for j in range(3):
            want = expected[i][j]
            assert math.isclose(float(p[i, j]), want, rel_tol=1e-12), (
                f"attach[{i}][{j}] = {p[i, j]!r}, expected {want!r}"
            )
    assert (repr(new_units), repr(old_units), repr(rec)) == snap, "the rule mutated its inputs"
    assert np.array_equal(p, attach_probabilities(new_units, old_units, rec)), "not deterministic"

    # The same matrix on the MIXTURE record, whose weights do not sum to 1: entry [0][1] is
    # CONN_PROB[4][2] * (2*exp(-0.32) + 0.5*exp(-0.5)) / 2.5 = 0.0505 * 0.6945376618616513.
    pm = attach_probabilities(new_units, old_units, dict(rec, a=2.0, c=0.5, lambda_um=80.0))
    mix_want = (2.0 * math.exp(-(40.0**2) / (2 * 50.0**2)) + 0.5 * math.exp(-40.0 / 80.0)) / 2.5
    assert math.isclose(float(pm[0, 1]), 0.0505 * mix_want, rel_tol=1e-12), (
        f"the mixture attachment is wrong: {pm[0, 1]!r} is not {0.0505 * mix_want!r}"
    )
    assert float(pm[0, 0]) == 0.0497, (
        "a unit-normalised mixture must still equal the layer target at d = 0"
    )

    # -- across ONE map cell the probability is 0 whatever else matches (equal (ap, dv) is the gate).
    assert (
        attach_probabilities([u("n", 0, 0, cell=(0, 0))], [u("o", 0, 0, cell=(1, 0))], rec)[0, 0]
        == 0.0
    )
    # ... and depth is not part of the distance: the 5th coordinate must not move the kernel.
    deep = u("o", 40, 0, "Layer 5", 1)
    deep["position"] = (0, 0, 40.0, 0.0, 0.9)
    assert attach_probabilities([u("n", 0, 0)], [deep], rec)[0, 0] == p[0, 1]

    # -- the beta shared-partner multiplier: with m = 2 shared partners the entry is
    #    base * (1 + beta * m) = base * 2 at beta = 0.5, and base unchanged at the beta = 0 default.
    shared = {"n0": {"q1", "q2"}, "n1": {"r"}, "o0": {"q1", "q2"}, "o1": {"r"}, "o2": set()}
    base = attach_probabilities(new_units, old_units, rec)[0, 0]
    assert attach_probabilities(new_units, old_units, rec, partners=shared)[0, 0] == base, (
        "the default beta of 0 must ignore partners entirely"
    )
    doubled = attach_probabilities(new_units, old_units, dict(rec, beta=0.5), partners=shared)[0, 0]
    assert math.isclose(float(doubled), base * 2.0, rel_tol=1e-12), (
        f"the shared-partner term is wrong: {doubled!r} is not 2 x {base!r} at beta = 0.5, m = 2"
    )
    # A partner set with nothing in common must not be treated as shared.
    disjoint = {"n0": {"q1", "q2"}, "n1": {"q9"}, "o0": {"q9"}, "o1": {"q9"}, "o2": set()}
    assert (
        attach_probabilities(new_units, old_units, dict(rec, beta=0.5), partners=disjoint)[0, 0]
        == base
    )

    # -- the clip fires on BOTH sides, and each side has its own reachable input:
    #    0.0497 * (1 + 100 * 3) = 14.9597 > 1 and 0.0497 * (1 - 1 * 3) = -0.0994 < 0.
    s, t = u("s", 0, 0), u("t", 0, 0)
    trio = {"s": {"q1", "q2", "q3"}, "t": {"q1", "q2", "q3"}}
    assert 0.0497 * (1.0 + 100.0 * 3) > 1.0 and 0.0497 * (1.0 - 1.0 * 3) < 0.0, (
        "the clip fixture drifted"
    )
    hi = attach_probabilities([s], [t], dict(rec, beta=100.0), partners=trio)[0, 0]
    lo = attach_probabilities([s], [t], dict(rec, beta=-1.0), partners=trio)[0, 0]
    assert hi == 1.0, f"the upper clip did not fire: {hi!r} (raw 14.9597)"
    assert lo == 0.0, f"the lower clip did not fire: {lo!r} (raw -0.0994)"
    # Every emitted entry is a probability whatever the record asks for.
    wide = attach_probabilities(new_units, old_units, dict(rec, beta=1e6), partners=shared)
    assert wide.shape == p.shape and np.all((wide >= 0.0) & (wide <= 1.0))
    assert wide[0, 0] == 1.0, "a huge beta on the shared pair must saturate, not run off the end"
    assert wide[0, 2] < 1.0, "an entry with no shared partner must stay unsaturated"

    # -- kernel_shape(rec, cell): a pooled record delivers the pathway's own widths (B2), and the
    #    pathway ABSENT from the mapping keeps the global width at zero offset.
    pooled = dict(rec, keep="pooled", pathway_widths={105: [160.0, 160.0]})
    at100 = attach_probabilities([s], [u("t", 100, 0)], pooled)[0, 0]
    assert math.isclose(
        float(at100), 0.0497 * math.exp(-(100.0**2) / (2 * 160.0**2)), rel_tol=1e-12
    )
    glob = attach_probabilities([s], [u("t", 100, 0)], rec)[0, 0]
    assert math.isclose(float(glob), 0.0497 * math.exp(-(100.0**2) / (2 * 50.0**2)), rel_tol=1e-12)
    assert at100 != glob, "the pooled width and the global width must not coincide at d = 100 um"
    # A JSON round trip turns the int cell key into a string; both must deliver the same width.
    strkey = dict(rec, keep="pooled", pathway_widths={"105": [160.0, 160.0]})
    assert attach_probabilities([s], [u("t", 100, 0)], strkey)[0, 0] == at100
    # An absent pathway keeps the global width; a pooled record with NO mapping is refused, not served.
    other = attach_probabilities([s], [u("t", 100, 0, "Layer 5", 1)], pooled)[0, 0]
    assert math.isclose(
        float(other), CONN_PROB[4][2] * math.exp(-(100.0**2) / (2 * 50.0**2)), rel_tol=1e-12
    )
    try:
        attach_probabilities([s], [t], dict(rec, keep="pooled"))
        raise AssertionError("a pooled record with no pathway_widths was served the global kernel")
    except RecordRefused:
        pass

    # -- the record refusals: T3's 3-D soma-distance record is not a Part B tangential one.
    t3 = {
        "artifact_sha256": "a1d0491a",
        "model": "mix",
        "width_um": {"sigma": 42.757, "lambda": 104.266},
    }
    for bad, why in ((t3, "T3's 3-D record"), (dict(rec, frame="3d"), "a non-tangential frame")):
        try:
            attach_probabilities([s], [t], bad)
            raise AssertionError(f"{why} was served an attachment probability")
        except RecordRefused:
            pass

    print(
        "wiring_prior self-check OK: population (L2/L3 aggregated, L1 + white matter refused); "
        "layer_term directed 0.0505 vs 0.0067; pathway_cell 105 / 39 / 165; kernel_shape 1 at d = 0 and "
        "falling and unit-normalised for a 2.5-weight mixture (which is the only record that can see the "
        "/(a + c) term); the 2 x 3 attachment matrix matches its six written-out entries on both the "
        "pure-Gaussian and the mixture record; cross-cell 0 and depth "
        "excluded; beta 0 default ignores partners, beta 0.5 with m = 2 doubles it; the clip fires on both "
        "sides (raw 14.9597 -> 1.0, raw -0.0994 -> 0.0); a pooled record delivers the pathway width "
        "(160 um) and refuses to serve the global one without its mapping; T3's 3-D record refused. "
        "Synthetic units, no H01 file, no Brain, no lake, no seed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
