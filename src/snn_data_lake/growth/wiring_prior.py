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
