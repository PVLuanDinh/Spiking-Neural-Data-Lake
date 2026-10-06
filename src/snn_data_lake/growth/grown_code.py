"""specs/022 T111 (G3) P1: the encoding contract's GROWN SEGMENT.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.3. The taught learner's frozen code
(`text_features(text, 52)` -> `rank_code`, 52 integer spike ranks) is byte-identical forever; widening it would
re-bucket every word (2.9 C1's red). Grown units add a SEPARATE segment instead: each grown unit j integrates the
frozen code's PSPs over integer steps of its own window and fires at its first-spike latency t_j (no spike = no
contribution); each label neuron reads it as PSP(t_post_g - t_j). The 52 ranks are never re-ranked with it.

Contract, all pure functions over immutable units (no RNG, no state carried between stimuli):
- `latency(unit, ranks, tau)` -- a capacity unit's latency is a function of (the frozen code, its parameters);
- `column_latencies(units, ranks, tau, gamma)` -- COL's causal E/I coupling (G7's amendment (h)); gamma = 0 is BATCH9;
- `segment_psp(units, ranks, tau, t_post_g)` -- the segment in unit_id order, never insertion order;
- `grown_version(units)` -- crc32 over every unit's parameters in unit_id order, active flag included, so it
  changes on every admission, silencing or output-weight change;
- `apply_unit_row(units, row)` -- moves the segment one journal step (admit / out / silence) and REFUSES a
  reused unit_id or a stored output synapse that violates the unit's Dale sign (2.9 C17's restore half);
- `admit_rows(segment, units, label_rows, ticket)` -- THE single append: a ticketed, label-matched batch only;
- `clip_to_sign(w, sign)` -- the deposit half of C17: E outputs >= 0, I outputs <= 0 (D9).

The port that feeds this segment into `SpikingInteractionLearner` lands after R2's P1 (OWNERS.md landing order);
this module is the segment it will call.

distinct-from: growth/srm_neuron.py and growth/lif_neuron.py -- those are STATEFUL step neurons (membrane carried
  across steps, reset/refractory kernels, their own eps kernel); a grown capacity unit here is the STATELESS
  first-crossing of the learner's own PSP kernel over one frozen rank code, so its latency is a pure function the
  redo and the certificate's lift can recompute exactly. Disclosed for the panel as a P2 question: if the column
  arc (P4) needs LIF dynamics, the unit kind adopts lif_neuron there rather than growing a third neuron here.
distinct-from: faculties/astrocyte_modulation.py, gates/perf_baseline.py, growth/placement.py,
  instruments/prior_drift_harness.py, sensorimotor/rank_order_coding.py -- read 2026-10-04 (the duplicate guard's
  hits): a glial gain relay, a bytes-per-unit measurement, GRN placement, a planted wiring control and a Van Rossum
  matcher; none computes a grown unit's latency segment.
"""

import math
import struct
import zlib
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GrownUnit:
    unit_id: int  # minted at admission, never reused
    w_in: tuple  # one input weight per frozen-code synapse; frozen after admission
    threshold: float
    window: int  # integer steps the unit integrates over; no crossing inside it = no spike
    sign: int  # Dale sign from the unit's label (+1 E, -1 I); write-once per unit_id
    out: tuple  # one output synapse per label neuron (the only weights that learn after admission)
    active: bool = True

    def silenced(self):
        return replace(self, active=False)


def _psp(dt, tau):
    return math.exp(-dt / tau) if dt > 0 else 0.0


def _drive(unit, ranks, tau, t):
    """The unit's PSP-weighted drive from the frozen code at step t (one expression for latency and the column)."""
    return sum(w * _psp(t - r, tau) for w, r in zip(unit.w_in, ranks) if w != 0.0)


def latency(unit, ranks, tau):
    """The unit's first-spike step: the first t in 1..window where the PSP-weighted frozen input reaches threshold."""
    for t in range(1, unit.window + 1):
        if _drive(unit, ranks, tau, t) >= unit.threshold:
            return t
    return None


def column_latencies(units, ranks, tau, gamma):
    """COL's wiring (G7's dated amendment (h)): one batch of k E units (sign +1) and exactly one I unit (sign -1),
    evaluated causally step by step, every unit spiking at most once, EARLIER steps' spikes only. An E unit's
    drive is its frozen-code drive minus gamma*theta_e*psp(t - t_I) once the I unit has spiked (I->E); the I
    unit's gains gamma*theta_I*psp(t - t_e) per spiked E unit (E->I). gamma is ONE fixed constant (never learned);
    gamma = 0 is BATCH9 and reproduces `latency` byte-identically. A silenced unit never spikes. All units share
    one window. Returns {unit_id: latency or None}."""
    inhibitory = [u for u in units if u.sign < 0]
    if len(inhibitory) != 1:
        raise ValueError(f"a column has exactly one inhibitory unit, got {len(inhibitory)}")
    if len({u.window for u in units}) != 1:
        raise ValueError("a column shares one window")
    inh, spiked = inhibitory[0], {}
    for t in range(1, inh.window + 1):
        new = {}
        for u in units:
            if u.unit_id in spiked or not u.active:
                continue
            v = _drive(u, ranks, tau, t)
            if u is inh:
                v += (
                    gamma
                    * u.threshold
                    * sum(_psp(t - te, tau) for e, te in spiked.items() if e != u.unit_id)
                )
            elif inh.unit_id in spiked:
                v -= gamma * u.threshold * _psp(t - spiked[inh.unit_id], tau)
            if v >= u.threshold:
                new[u.unit_id] = t
        spiked.update(
            new
        )  # spikes of step t act from step t + 1 (psp(0) = 0): the evaluation is causal
    return {u.unit_id: spiked.get(u.unit_id) for u in units}


def segment_psp(units, ranks, tau, t_post_g):
    """Each unit's PSP onto the label neurons, in unit_id order; silenced or non-spiking units give 0.0."""
    out = []
    for u in sorted(units, key=lambda u: u.unit_id):
        t = latency(u, ranks, tau) if u.active else None
        out.append(_psp(t_post_g - t, tau) if t is not None else 0.0)
    return out


def latencies(units, ranks, tau):
    """Each unit's latency (None = no spike), in unit_id order: the grown part of the certificate's lift."""
    return [
        latency(u, ranks, tau) if u.active else None for u in sorted(units, key=lambda u: u.unit_id)
    ]


def grown_version(units):
    """crc32 over every unit's parameters in unit_id order (little-endian doubles, the active flag included)."""
    crc = 0
    for u in sorted(units, key=lambda u: u.unit_id):
        blob = struct.pack("<qdqb?", u.unit_id, u.threshold, u.window, u.sign, u.active)
        blob += struct.pack(f"<{len(u.w_in)}d", *u.w_in) + struct.pack(f"<{len(u.out)}d", *u.out)
        crc = zlib.crc32(blob, crc)
    return crc


def _require_sign(uid, sign):
    if sign not in (1, -1) or isinstance(sign, bool):
        raise ValueError(f"unit {uid}: Dale sign must be exactly +1 or -1, got {sign!r}")


def _require_finite(uid, what, values):
    bad = [i for i, x in enumerate(values) if not math.isfinite(x)]
    if bad:
        raise ValueError(f"unit {uid}: {what} synapse {bad[0]} is not finite ({values[bad[0]]!r})")


def clip_to_sign(w, sign, uid=None):
    """Dale's sign at a deposit: an E unit's output synapses stay >= 0, an I unit's <= 0. A sign other
    than exactly +-1 or a non-finite synapse is REFUSED, never clipped (cp1 #17: NaN passes both
    comparisons and would survive a clip)."""
    _require_sign(uid, sign)
    _require_finite(uid, "output", w)
    return [max(x, 0.0) if sign > 0 else min(x, 0.0) for x in w]


def _check_unit(u, out):
    """Admission and restore: sign exactly +-1, every input and output synapse and the threshold finite,
    every output synapse on the unit's Dale side; each refusal names the unit and the synapse."""
    _require_sign(u.unit_id, u.sign)
    _require_finite(u.unit_id, "input", u.w_in)
    _require_finite(u.unit_id, "threshold", (u.threshold,))
    _require_finite(u.unit_id, "output", out)
    bad = [i for i, x in enumerate(out) if (u.sign > 0 and x < 0) or (u.sign < 0 and x > 0)]
    if bad:
        raise ValueError(f"unit {u.unit_id}: output synapse(s) {bad} violate Dale sign {u.sign:+d}")


def apply_unit_row(units, row):
    """One journal step on an immutable segment; returns the new tuple sorted by unit_id."""
    by_id = {u.unit_id: u for u in units}
    op = row["op"]
    if op == "admit":
        u = row["unit"]
        if u.unit_id in by_id:
            raise ValueError(f"unit {u.unit_id}: unit_id already admitted (ids are never reused)")
        _check_unit(u, u.out)
        by_id[u.unit_id] = u
    elif op == "out":
        u = by_id[row["unit_id"]]
        out = tuple(float(x) for x in row["out"])
        _check_unit(u, out)
        by_id[u.unit_id] = replace(u, out=out)
    elif op == "silence":
        by_id[row["unit_id"]] = by_id[row["unit_id"]].silenced()
    else:
        raise ValueError(f"unknown unit-row op {op!r}")
    return tuple(by_id[k] for k in sorted(by_id))


def admit_rows(segment, units, label_rows, ticket):
    """THE single append point of the grown segment (2026-10-04 amendment): returns the segment with `units`
    admitted, and REFUSES unless `ticket` (G7's AdmissionTicket from `require_admissible`) covers exactly these
    label rows, unchanged, and has not been spent. Each unit's Dale sign must be its label's sign. The new
    segment is built whole first and the ticket is consumed last, so any refusal appends nothing and spends
    nothing; the label gate precedes every append by construction."""
    label_rows = list(label_rows)
    sign_of = {r["unit_id"]: r["sign"] for r in label_rows}
    if sorted(sign_of) != sorted(u.unit_id for u in units):
        raise ValueError("the units and their label rows name different unit ids")
    for u in units:
        if u.sign != sign_of[u.unit_id]:
            raise ValueError(
                f"unit {u.unit_id}: Dale sign {u.sign:+d} differs from its label's {sign_of[u.unit_id]:+d}"
            )
    for u in units:
        segment = apply_unit_row(segment, {"op": "admit", "unit": u})
    ticket.consume(label_rows)
    return segment


def main():
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features

    texts = [
        "cat dog sun sky",
        "cat dog sky sun",
        "dog cat sun sky",
        "dog cat sky sun",
        "hello there",
        "sky",
    ]

    def ranks(n_in, t):
        return tuple(
            SpikingInteractionLearner(["A"], n_in=n_in, seed=9900).encode(text_features(t, n_in))
        )

    assert [ranks(52, t) for t in texts] == [ranks(52, t) for t in texts]
    changed = sum(ranks(52, t) != ranks(62, t)[:52] for t in texts)
    assert changed == len(texts), f"red: widening 52 -> 62 must change every probe code ({changed})"
    u = GrownUnit(1, tuple([2.0] + [0.0] * 51), 1.0, 12, -1, (-0.2,))
    try:
        apply_unit_row((u,), {"op": "out", "unit_id": 1, "out": (0.3,)})
        raise AssertionError("red: a sign-violating restore must be refused")
    except ValueError:
        pass
    assert grown_version((u,)) != grown_version((u.silenced(),))
    print(
        f"grown_code: C1 red {changed}/{len(texts)} codes change on widening; C17 restore red fires"
    )


if __name__ == "__main__":
    main()
