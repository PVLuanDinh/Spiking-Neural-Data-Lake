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


def _shape(d, rec):
    a, c = rec["a"], rec["c"]
    v = a * np.exp(-(d**2) / (2.0 * rec["sigma_um"] ** 2)) + c * np.exp(-d / rec["lambda_um"])
    return v / (a + c)


def attach_probabilities(new_units, existing_units, record, partners=None):
    """Array [len(new_units), len(existing_units)] of p(new -> existing), each in [0, 1]."""
    rec = _check(record)
    beta = rec.get("beta", 0.0)
    partners = partners or {}
    out = np.zeros((len(new_units), len(existing_units)))
    for i, u in enumerate(new_units):
        pu = population(u["layer"], u["sign"])
        ua, ud, ux, uy, _ = u["position"]
        for j, v in enumerate(existing_units):
            va, vd, vx, vy, _ = v["position"]
            if (ua, ud) != (va, vd):
                continue
            target = CONN_PROB[population(v["layer"], v["sign"])][pu]
            m = len(set(partners.get(u["unit_id"], ())) & set(partners.get(v["unit_id"], ())))
            out[i, j] = target * _shape(np.hypot(ux - vx, uy - vy), rec) * (1.0 + beta * m)
    return np.clip(out, 0.0, 1.0)
