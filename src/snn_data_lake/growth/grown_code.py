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

import copy
import math
import struct
import zlib
from dataclasses import dataclass, replace

from snn_data_lake.plasticity.r_stdp import (
    accumulate_anticausal,
    accumulate_causal,
    decay_eligibility,
    reward_deltas,
)

GRADIENT_CLAIM = {
    "GrownReturnPort.apply": (
        "LOCAL",
        "Calls the existing canonical R-STDP operations over local pre/post eligibility and one scalar reward; no error or forward weight is transported.",
    ),
}


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


def _drive(unit, ranks, tau, t, *, afferents=None):
    """The unit's PSP-weighted drive from the frozen code at step t (one expression for latency and the column)."""
    base = sum(w * _psp(t - r, tau) for w, r in zip(unit.w_in, ranks) if w != 0.0)
    if afferents is None:
        return base
    return base + sum(w * _psp(t - arrival, tau) for arrival, w in afferents.get(unit.unit_id, ()))


def latency(unit, ranks, tau, *, afferents=None):
    """The unit's first-spike step: the first t in 1..window where the PSP-weighted frozen input reaches threshold."""
    for t in range(1, unit.window + 1):
        if _drive(unit, ranks, tau, t, afferents=afferents) >= unit.threshold:
            return t
    return None


def column_latencies(units, ranks, tau, gamma, *, afferents=None):
    """COL's wiring (G7's dated amendment (h)): one batch of k E units (sign +1) and exactly one I unit (sign -1),
    evaluated causally step by step, every unit spiking at most once, EARLIER steps' spikes only. An E unit's
    drive is its frozen-code drive minus gamma*theta_e*psp(t - t_I) once the I unit has spiked (I->E); the I
    unit's gains gamma*theta_I*psp(t - t_e) per spiked E unit (E->I). gamma is ONE fixed constant (never learned);
    gamma = 0 is BATCH9 and reproduces `latency` byte-identically. A silenced unit never spikes. All units share
    one window. Returns {unit_id: latency or None} in unit_id order."""
    units = sorted(
        units, key=lambda u: u.unit_id
    )  # unit_id order, never the caller's (bytes, not just ==)
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
            v = _drive(u, ranks, tau, t, afferents=afferents)
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


def segment_psp(units, ranks, tau, t_post_g, *, afferents=None):
    """Each unit's PSP onto the label neurons, in unit_id order; silenced or non-spiking units give 0.0."""
    out = []
    for u in sorted(units, key=lambda u: u.unit_id):
        t = latency(u, ranks, tau, afferents=afferents) if u.active else None
        out.append(_psp(t_post_g - t, tau) if t is not None else 0.0)
    return out


def latencies(units, ranks, tau, *, afferents=None):
    """Each unit's latency (None = no spike), in unit_id order: the grown part of the certificate's lift."""
    return [
        latency(u, ranks, tau, afferents=afferents) if u.active else None
        for u in sorted(units, key=lambda u: u.unit_id)
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
    if isinstance(u.window, bool) or not isinstance(u.window, int) or u.window < 1:
        raise ValueError(f"unit {u.unit_id}: window must be a positive int, got {u.window!r}")
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


def _return_need(value, reason):
    if not value:
        raise ValueError("G9 return refused: " + reason)


class GrownReturnPort:
    """Plastic afferents on ACTUAL born targets; the frozen input/out tables remain distinct."""

    def __init__(self, units, birth_rows, source_ids, source_signs, attachments, rule):
        units = tuple(sorted(units, key=lambda u: u.unit_id))
        ids = tuple(u.unit_id for u in units)
        _return_need(units and len(ids) == len(set(ids)), "actual unique targets required")
        _return_need(
            type(source_ids) is tuple
            and source_ids
            and all(type(x) is str and x for x in source_ids)
            and len(source_ids) == len(set(source_ids)),
            "ordered real source IDs",
        )
        _return_need(
            type(source_signs) is dict
            and set(source_signs) == set(source_ids)
            and all(type(x) is int and x in (-1, 1) for x in source_signs.values()),
            "explicit source Dale signs required",
        )
        _return_need(
            type(rule) is dict
            and set(rule) == {"lr", "a_plus", "a_minus", "tau_plus", "tau_minus", "tau_e"}
            and all(type(v) is float and math.isfinite(v) and v > 0 for v in rule.values()),
            "complete existing R-STDP parameters",
        )
        _return_need(
            type(birth_rows) is dict and set(birth_rows) == set(ids),
            "one actual immutable birth row per target",
        )
        for u in units:
            _check_unit(u, u.out)
            row = birth_rows[u.unit_id]
            _return_need(
                type(u.active) is bool
                and row["origin"] in {"born", "constructed"}
                and row["unit_id"] == u.unit_id
                and row["sign"] == u.sign
                and tuple(row["w_in"]) == u.w_in
                and row["threshold"] == u.threshold
                and row["window"] == u.window,
                "target/birth identity or frozen input changed",
            )
        _return_need(
            type(attachments) in (list, tuple) and attachments,
            "no default or zero-dimensional return projection",
        )
        rows, seen = [], set()
        for row in attachments:
            _return_need(
                type(row) is dict
                and set(row) == {"source", "target", "sign", "weight", "delay", "receipt"},
                "typed attachment fields required",
            )
            key = (row["source"], row["target"])
            _return_need(
                key not in seen
                and row["source"] in source_ids
                and type(row["target"]) is int
                and row["target"] in ids,
                "duplicate/unknown attachment endpoint",
            )
            _return_need(
                type(row["sign"]) is int
                and row["sign"] == source_signs[row["source"]]
                and type(row["weight"]) is float
                and math.isfinite(row["weight"])
                and row["weight"] * row["sign"] >= 0,
                "nonfinite or wrong-source-sign attachment",
            )
            _return_need(
                type(row["delay"]) is int
                and row["delay"] >= 0
                and type(row["receipt"]) is str
                and len(row["receipt"]) == 64
                and all(c in "0123456789abcdef" for c in row["receipt"]),
                "actual delay/attachment-source receipt required",
            )
            rows.append(dict(row))
            seen.add(key)
        _return_need(set(r["target"] for r in rows) == set(ids), "every target must be attached")
        self.factory = {
            "sources": list(source_ids),
            "signs": dict(source_signs),
            "targets": list(ids),
            "birth_rows": [[uid, copy.deepcopy(birth_rows[uid])] for uid in ids],
            "attachments": copy.deepcopy(rows),
            "rule": dict(rule),
        }
        self.rows = rows
        self.weights = [r["weight"] for r in rows]
        self.eligibility = [0.0] * len(rows)
        self.pre = [None] * len(rows)
        self.post = [None] * len(rows)
        self.pending = []
        self.clock = 0.0
        self.last_event_id = None
        self.reset_count = 0

    def validate_population(self, units):
        by_id = {u.unit_id: u for u in units}
        _return_need(
            set(by_id) == set(self.factory["targets"]),
            "target membership changed without journaled port rebind",
        )
        for uid, u in by_id.items():
            _check_unit(u, u.out)
            born = dict(self.factory["birth_rows"])[uid]
            _return_need(
                tuple(born["w_in"]) == u.w_in
                and born["sign"] == u.sign
                and born["threshold"] == u.threshold
                and born["window"] == u.window,
                "frozen target semantics changed",
            )
        return by_id

    def context(self, units, start, *, neutered=False):
        """Relative arrivals feed the SAME grown-code PSP/latency functions."""
        by_id = self.validate_population(units)
        _return_need(
            type(start) is float and math.isfinite(start) and start >= self.clock,
            "actual monotone attempt clock required",
        )
        result = {uid: [] for uid in sorted(by_id)}
        if not neutered:
            for row_index, arrival in self.pending:
                row = self.rows[row_index]
                if by_id[row["target"]].active:
                    result[row["target"]].append((arrival - start, self.weights[row_index]))
        return result

    def apply(
        self,
        *,
        event_id,
        units,
        packet,
        target_latencies,
        start,
        end,
        reward,
        learn,
        eligible,
        neutered=False,
    ):
        """Called ONLY by the actual owner after one common R2 row admission.

        packet comes from the real readout's selected neuron post_spike. Target
        post times come from its real context-aware GrownUnit first crossings.
        No expected-answer label is accepted by this input.
        """
        by_id = self.validate_population(units)
        _return_need(
            type(event_id) is str and event_id and event_id != self.last_event_id,
            "missing/duplicate admitted event",
        )
        _return_need(
            type(start) is float
            and type(end) is float
            and math.isfinite(start)
            and math.isfinite(end)
            and self.clock <= start < end,
            "actual bounded interval",
        )
        _return_need(
            type(learn) is bool
            and type(neutered) is bool
            and type(reward) is float
            and math.isfinite(reward),
            "typed local third factor",
        )
        _return_need(
            type(eligible) is tuple
            and len(eligible) == len(self.rows)
            and all(type(x) is bool for x in eligible),
            "full G6 projection mask, including every plastic afferent",
        )
        _return_need(
            type(target_latencies) is dict and set(target_latencies) == set(by_id),
            "complete actual target latency reading",
        )
        for uid, t in target_latencies.items():
            _return_need(
                t is None or type(t) is int and 1 <= t <= by_id[uid].window,
                "invalid target first crossing",
            )
        _return_need(
            type(packet) is tuple
            and len(packet) == 2
            and packet[0] in self.factory["sources"]
            and type(packet[1]) is float
            and math.isfinite(packet[1])
            and start <= packet[1] < end,
            "actual selected readout post spike required",
        )
        local = copy.deepcopy(
            (
                self.weights,
                self.eligibility,
                self.pre,
                self.post,
                self.pending,
                self.clock,
                self.last_event_id,
            )
        )
        try:
            rule = self.factory["rule"]
            decay_eligibility(self.eligibility, start - self.clock, rule["tau_e"])
            source, spike = packet
            for k, row in enumerate(self.rows):
                t = target_latencies[row["target"]]
                if t is not None and by_id[row["target"]].active:
                    self.post[k] = start + t
                    if self.pre[k] is not None and learn and eligible[k]:
                        accumulate_causal(
                            self.eligibility,
                            ((k, self.post[k] - self.pre[k]),),
                            rule["a_plus"],
                            rule["tau_plus"],
                        )
                if source == row["source"] and not neutered:
                    arrival = spike + row["delay"]
                    self.pending.append((k, arrival))
                    self.pre[k] = arrival
                    if self.post[k] is not None and learn and eligible[k]:
                        accumulate_anticausal(
                            self.eligibility,
                            ((k, self.post[k] - arrival),),
                            rule["a_minus"],
                            rule["tau_minus"],
                        )
            if learn and not neutered:
                deltas = reward_deltas(self.eligibility, reward, rule["lr"])
                for k, row in enumerate(self.rows):
                    if eligible[k] and by_id[row["target"]].active:
                        self.weights[k] = clip_to_sign(
                            [self.weights[k] + deltas[k]], row["sign"], row["target"]
                        )[0]
            self.clock, self.last_event_id = end, event_id
            self._validate_state()
        except BaseException:
            (
                self.weights,
                self.eligibility,
                self.pre,
                self.post,
                self.pending,
                self.clock,
                self.last_event_id,
            ) = local
            raise

    def _validate_state(self):
        n = len(self.rows)
        _return_need(
            all(
                type(v) is list and len(v) == n
                for v in (self.weights, self.eligibility, self.pre, self.post)
            ),
            "return state shape changed",
        )
        _return_need(
            all(type(v) is float and math.isfinite(v) for v in self.weights + self.eligibility),
            "nonfinite return state",
        )
        _return_need(
            all(t is None or type(t) is float and math.isfinite(t) for t in self.pre + self.post),
            "invalid spike history",
        )
        _return_need(
            all(w * r["sign"] >= 0 for w, r in zip(self.weights, self.rows, strict=True)),
            "restored attachment sign mismatch",
        )
        _return_need(
            type(self.pending) is list
            and all(
                type(p) is tuple
                and len(p) == 2
                and type(p[0]) is int
                and 0 <= p[0] < n
                and type(p[1]) is float
                and math.isfinite(p[1])
                for p in self.pending
            ),
            "invalid full pending return queue",
        )
        _return_need(
            type(self.clock) is float
            and math.isfinite(self.clock)
            and self.clock >= 0
            and type(self.reset_count) is int
            and self.reset_count >= 0,
            "invalid clock/reset count",
        )
        _return_need(
            self.last_event_id is None or type(self.last_event_id) is str and self.last_event_id,
            "invalid return event identity",
        )

    def state_dict(self):
        from snn_data_lake.readout.interaction_learning import _encode_state

        self._validate_state()
        return {"schema": "g9.return.v1", "tree": _encode_state(self)}

    def load_state_dict(self, carrier):
        from snn_data_lake.readout.interaction_learning import _merge_state

        _return_need(
            type(carrier) is dict
            and set(carrier) == {"schema", "tree"}
            and carrier["schema"] == "g9.return.v1",
            "return carrier schema",
        )
        staged = copy.deepcopy(self)
        _merge_state(staged, carrier["tree"])
        _return_need(
            staged.factory == self.factory and staged.rows == self.rows,
            "carrier factory/source/target/attachment provenance changed",
        )
        staged._validate_state()
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)

    def eligible_circuits(self):
        """Explicit G6-facing source-owned identities, not a whole-Brain census."""
        return tuple(
            {
                "circuit": "g9.return",
                "source": r["source"],
                "target": r["target"],
                "index": k,
                "plastic": True,
            }
            for k, r in enumerate(self.rows)
        )

    def reset_context(self):
        """Journaled explicit context reset; learned weights remain, counted honestly."""
        self.eligibility[:] = [0.0] * len(self.rows)
        self.pre[:] = [None] * len(self.rows)
        self.post[:] = [None] * len(self.rows)
        self.pending.clear()
        self.reset_count += 1


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
