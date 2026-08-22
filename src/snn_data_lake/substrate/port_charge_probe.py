"""Port charge instruments (specs/022 T021, invention #2's measurement half, contracts/
manifest.md 'Charge measurement'): pinned-payload neuters, cleanup-margin surveillance, and
the replay-twice A/B that ATTRIBUTES a consumer's divergence to one port.

WHY THIS EXISTS: registration is never charge (P3's honesty guard). A port can be declared,
edged and polled while nothing downstream depends on its payload -- presence without
participation. These instruments make the dependence measurable:

  * PINNED-PAYLOAD NEUTER -- wrap a JoinConsumer so every joined payload is replaced by a
    pinned constant while READS ARE RECORDED. The pinned-dict lesson applies at port
    granularity: a MOVING payload can still never matter, and a pinned one that changes
    nothing downstream proves it never did.
  * CLEANUP-MARGIN SURVEILLANCE -- arbitrate joins carry {winner, margin_to_runner_up} per
    decision (signal_ring.JoinConsumer.last_margin); a consumer whose margin NEVER MOVES
    across a varied run is not consuming, and margin_frozen() is that tell as a criterion.
  * REPLAY-TWICE A/B -- the same recorded frame stream driven through the consumer twice,
    once live and once with ONE port pinned; output divergence attributes the charge to that
    port, per item (P8 clause 3).

THE POSITIVE CONTROL IS EXTERNAL AND EXISTS AT LANDING (invention #2's clause 2): a
constructed consumer whose output IS a function of the payload MUST flip under the neuter
(sensitivity), and a constant consumer MUST NOT (specificity) -- both run in --self-check
and in the bare wiring check, so the instrument is never trusted on reds nobody has seen it
fire. Re-proven against the FIRST REAL REGISTRANT when T025 lands (recorded obligation).

PLACEMENT: the task text says tools/port_charge_probe.py; it lands in substrate/ for the
same reason T001's perf_baseline did -- the p4c ratchet refuses +1 ungoverned file in
tools/, and an instrument is part of the brain (P4c). Deviation recorded, not smoothed.
"""

# distinct-from: tools/charge_probe.py -- the MODULE-level neuter A/B (numeric state fields
#   of whole faculties, 47/60 external-control flips); this operates at PORT granularity on
#   manifest-declared payloads. Same philosophy, different unit of attribution -- charge_probe
#   remains the module-level incumbent, cited not merged (different denominators would make a
#   merged census lie in both directions).
# distinct-from: substrate/signal_ring.py -- the delivery mechanism these instruments wrap.
# distinct-from: substrate/manifest_gate.py -- static contract checks; this is the LIVE half
#   its own docstring defers to. All sites read.

from __future__ import annotations

import sys


class PinnedJoin:
    """A JoinConsumer wrapper: same interface, every joined payload replaced by `pinned`,
    every read counted. The reads counter is the anti-rigging half (P8 clause 6): a neuter
    that nobody read proves nothing about the reader it claims to neuter."""

    def __init__(self, consumer, pinned):
        self._consumer = consumer
        self._pinned = pinned
        self.reads = 0

    @property
    def last_margin(self):
        return self._consumer.last_margin

    def poll_joined(self):
        joined, lost = self._consumer.poll_joined()
        out = []
        for tick, _combined in joined:
            self.reads += 1
            out.append((tick, self._pinned))
        return out, lost


def margin_frozen(margins, tol=0.0) -> bool:
    """True when the arbitrate margin never moved across the run -- the not-consuming tell.
    `tol` is the caller's noise floor (P1: derived from its own margin distribution, never
    picked here; 0.0 = exact freeze, the strictest honest default)."""
    vals = [m["margin_to_runner_up"] for m in margins if m]
    return len(vals) >= 2 and max(vals) - min(vals) <= tol


# consolidates: basal_ganglia/evidence_accumulation.py (merge)
# Its T025 record_charge wrote the whole evidence file as a single row (the next registrant
# would have deleted the first's attribution). This read-modify-write per edge supersedes
# that inline writer; evidence_accumulation's record_charge becomes a caller of THIS
# function in the same change (merge disposition -- the module stays, its writer delegates).
def record_attribution(edge, measured, provenance):
    """THE one writer for tools/region_charge_evidence.json: read-modify-write that
    replaces this edge's row or appends it, so one registrant's regeneration can never
    delete another's attribution. Callers must have MEASURED divergence before calling --
    this records, it never certifies."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools" / "region_charge_evidence.json"
    rows = json.loads(path.read_text()).get("attributions", []) if path.exists() else []
    rows = [r for r in rows if tuple(r.get("edge", ())) != tuple(edge)]
    rows.append(
        {"edge": list(edge), "diverged": True, "measured": measured, "provenance": provenance}
    )
    rows.sort(key=lambda r: r["edge"])
    path.write_text(json.dumps({"attributions": rows}, indent=2) + "\n")
    return path


def read_attribution(edge):
    """The committed row for `edge`, or None. The read half of record_attribution."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools" / "region_charge_evidence.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text()).get("attributions", [])
    mine = [r for r in rows if tuple(r.get("edge", ())) == tuple(edge)]
    return mine[0] if mine else None


# distinct-from: basal_ganglia/evidence_accumulation.py -- its _measure_patience_charge
#   records SEROTONIN frames from the raphe (a route the connectome declares, so the
#   consumer may import its producer directly); this builds the T026 INTEROCEPTIVE frames,
#   which touch two regions whose direct import the connectome does NOT declare -- hence
#   the instrument-layer home. Read.
def interoceptive_challenge_frames(vary, n_ticks=40, spike_at=30):
    """The T026 recorded challenge schedules (moved here from insula at review: a consumer
    importing its producers DIRECTLY put two region pairs on the import graph that the
    connectome declares no route for -- the declared tract relays through thalamus -- and
    the p6 undeclared-pairs ratchet went red exactly as designed. The INSTRUMENT layer is
    where measurement code that touches multiple regions belongs, the T021 precedent).

    Exactly ONE producer follows its challenge schedule; the other is FROZEN at its quiet
    vector -- so any live-arm response attributes to the varying producer's edge (per-edge
    attribution; a whole-join neuter cannot separate two producers). vary in
    {'arousal', 'drives'}. -> (frames, spike_at). Deterministic."""
    from snn_data_lake.hypothalamus.homeostatic_drives import SET_POINT, HomeostaticDrives
    from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal
    from snn_data_lake.thalamus.interoceptive_relay import InteroceptiveRelay

    arousal = InteroceptiveArousal(seed=0)
    drives = HomeostaticDrives()
    # T039j: the ascending half now passes through the THALAMIC RELAY, because that is the hop the
    # connectome declares and the manifest edge is `relay_out -> body_in`. Recording frames from
    # the medulla and labelling them as the relay would be measuring a path the tree does not run.
    relay = InteroceptiveRelay()
    quiet_a = arousal.sense(0.0)
    quiet_d = drives.deficit()

    def deficit_vec(deficits):
        return [min(1.0, deficits[k] / SET_POINT) for k in ("energy", "curiosity", "social")]

    service = ("rest", "explore", "socialize")  # round-robin keeps every deficit near zero
    frames = []
    for t in range(1, n_ticks + 1):
        challenged = t >= spike_at
        if vary == "arousal":
            a = arousal.sense(3.0 if challenged else 0.0)
            frames.append((t, "interoceptive_relay", relay.relay([a["stress"], a["caution"]])))
            frames.append((t, "homeostatic_drives", deficit_vec(quiet_d)))
        else:
            # the challenge is TOTAL neglect: servicing stops, every deficit climbs at its
            # drift rate -- a real discontinuity against the near-zero serviced baseline
            drives.tick(None if challenged else service[t % 3])
            frames.append(
                (t, "interoceptive_relay", relay.relay([quiet_a["stress"], quiet_a["caution"]]))
            )
            frames.append((t, "homeostatic_drives", deficit_vec(drives.deficit())))
    return frames, spike_at


def visceral_challenge_frames(n_ticks=40, spike_at=30):
    """The T039j ascending schedule: the medullary pair as the THALAMIC RELAY receives it.

    Deliberately NOT relayed -- `interoceptive_challenge_frames` above records the tract as the
    INSULA receives it (post-relay), and this records it as the relay receives it (pre-relay).
    The two schedules measure two different edges of the same tract and neither can stand in for
    the other: pinning a relayed frame tells you nothing about whether the relay's own input
    matters. Deterministic. -> (frames, spike_at)."""
    from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal

    arousal = InteroceptiveArousal(seed=0)
    frames = []
    for t in range(1, n_ticks + 1):
        a = arousal.sense(3.0 if t >= spike_at else 0.0)
        frames.append((t, "interoceptive_arousal", [a["stress"], a["caution"]]))
    return frames, spike_at


def interoception_challenge_frames(n_ticks=40, spike_at=30):
    """The T039j descending schedule: `interoception_out` as the ignition law receives it.

    A REAL insula terminus is settled on a quiet body and then challenged, so the frames carry the
    salience the tract actually produces rather than a synthetic ramp -- the point of the A/B is
    whether THIS payload moves the threshold, and a made-up payload could not answer that.
    Deterministic. -> (frames, spike_at)."""
    from snn_data_lake.insula.interoceptive_integration import (
        BODY_CHANNELS,
        InteroceptiveIntegration,
    )
    from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal
    from snn_data_lake.thalamus.interoceptive_relay import InteroceptiveRelay

    organ, relay = InteroceptiveArousal(seed=0), InteroceptiveRelay()
    ii = InteroceptiveIntegration(channels=BODY_CHANNELS)
    frames = []
    for t in range(1, n_ticks + 1):
        a = organ.sense(3.0 if t >= spike_at else 0.0)
        s, c = relay.relay([a["stress"], a["caution"]])
        sal = ii.observe(
            energy_deficit=0.0,
            curiosity_deficit=0.0,
            social_deficit=0.0,
            stress=s,
            caution=c,
        )
        frames.append((t, "interoceptive_integration", [sal]))
    return frames, spike_at


def replay_twice(frames, consumer_factory, consume, pinned):
    """The A/B: drive the SAME recorded frames through a fresh consumer twice -- live, then
    with the payload pinned -- and return (live_out, pinned_out, reads). Divergence between
    the two outputs attributes the consumer's behaviour to THIS port's payload, per item.

    `frames` are (tick, producer, vector) triples; `consumer_factory()` builds a fresh ring +
    JoinConsumer pair per arm (state never leaks between arms); `consume(consumer)` runs the
    downstream computation under measurement and returns its output."""
    from snn_data_lake.substrate.signal_ring import JoinConsumer

    ring_a, cons_a = consumer_factory()
    for tick, producer, vec in frames:
        ring_a.publish(JoinConsumer.frame(tick, producer, vec))
    live_out = consume(cons_a)

    ring_b, cons_b = consumer_factory()
    for tick, producer, vec in frames:
        ring_b.publish(JoinConsumer.frame(tick, producer, vec))
    neutered = PinnedJoin(cons_b, pinned)
    pinned_out = consume(neutered)
    return live_out, pinned_out, neutered.reads


def main() -> int:
    import numpy as np

    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import Port

    def factory(policy="sum"):
        def build():
            ring = SignalRing(slots=64)
            port = Port(
                name="probe_in",
                region="thalamus",
                shape=(4,),
                dtype="f4",
                unit="rate_hz",
                encoding="VEC_DENSE",
                kind="IN",
                declaring_module="global_workspace",
                join_policy=policy,
            )
            return ring, JoinConsumer(ring, port)

        return build

    frames = [(t, "alpha", [float(t), 2.0, 3.0, 4.0]) for t in range(1, 6)]
    pinned = np.zeros(4, dtype=np.float32)

    # POSITIVE CONTROL (sensitivity): a consumer whose output depends on the payload MUST flip.
    def dependent(consumer):
        joined, _ = consumer.poll_joined()
        return round(sum(float(np.sum(v)) for _t, v in joined), 6)

    live, neut, reads = replay_twice(frames, factory(), dependent, pinned)
    if live == neut or reads != 5:
        print(
            f"port_charge_probe self-check FAILED: sensitivity control did not flip "
            f"(live={live}, pinned={neut}, reads={reads})"
        )
        return 1

    # GREEN CONTROL (specificity): a constant consumer MUST NOT flip -- an instrument that
    # fires on everything attributes nothing.
    def constant(consumer):
        consumer.poll_joined()
        return 42.0

    live_c, neut_c, _ = replay_twice(frames, factory(), constant, pinned)
    if live_c != neut_c:
        print("port_charge_probe self-check FAILED: specificity control flipped on a constant")
        return 1

    # MARGIN SURVEILLANCE: varied arbitrate input moves the margin; identical input freezes it.
    ring, cons = factory("arbitrate")()
    margins = []
    for t in range(1, 5):
        ring.publish(JoinConsumer.frame(t, "alpha", [float(t) * 2, 0.0, 0.0, 0.0]))
        ring.publish(JoinConsumer.frame(t, "beta", [1.0, 1.0, 1.0, 1.0]))
        cons.poll_joined()
        margins.append(dict(cons.last_margin))
    if margin_frozen(margins):
        print(f"port_charge_probe self-check FAILED: varied input read as frozen ({margins})")
        return 1
    ring2, cons2 = factory("arbitrate")()
    frozen = []
    for t in range(1, 5):
        ring2.publish(JoinConsumer.frame(t, "alpha", [3.0, 0.0, 0.0, 0.0]))
        ring2.publish(JoinConsumer.frame(t, "beta", [1.0, 1.0, 1.0, 1.0]))
        cons2.poll_joined()
        frozen.append(dict(cons2.last_margin))
    if not margin_frozen(frozen):
        print(f"port_charge_probe self-check FAILED: identical input not read as frozen ({frozen})")
        return 1

    print(
        "port_charge_probe self-check OK: sensitivity control flipped under the pinned "
        f"payload ({live} -> {neut}, 5 reads counted), specificity control held, margin "
        "surveillance separated varied from frozen. RE-PROOF DISCHARGED at T025: "
        "evidence_accumulation's self-check runs both controls against the first real "
        "registrant (the serotonin patience wire) every battery."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
