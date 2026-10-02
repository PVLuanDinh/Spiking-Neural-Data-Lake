"""Thalamic interoceptive relay (`thal_relay`) -- the hop the connectome declared and no module
implemented (specs/022 T039j).

THE GAP THIS CLOSES, stated as the tree itself stated it. `connectome.py` declares the ascending
visceral tract as `viscera -> nts_rvlm -> thal_relay -> insula -> {gwt, ctx_pfc}` -- four hops, and
until now the middle one existed only as a NODE ID. The consequence was recorded in
`instruments/port_charge_probe.interoceptive_challenge_frames`, verbatim: *a consumer importing its
producers DIRECTLY put two region pairs on the import graph that the connectome declares no route
for -- the declared tract relays through thalamus -- and the p6 undeclared-pairs ratchet went red
exactly as designed*. The workaround was to move the multi-region measurement into the instrument
layer. That was the right call THEN, and it left `(medulla, thalamus)` and `(thalamus, insula)`
sitting in `p6_gate`'s `unimplemented_declared_pairs` -- declared anatomy with no code path.

THE ASYMMETRY THAT MAKES THIS BUILD LEGAL WHERE THE EARLIER ONE WAS NOT: the insula importing the
MEDULLA created `(medulla, insula)`, a pair the connectome declares NO route for, so the ratchet
was right to redden. `(medulla, thalamus)` and `(thalamus, insula)` are BOTH declared. Implementing
them removes debt instead of adding it.

WHAT A RELAY ACTUALLY DOES, because a module that only forwards is P3 dormancy and would deserve
the redness a force-wire earns. A thalamic relay neuron is not a wire; it has two transmission
modes (Sherman & Guillery). Held hyperpolarized by a quiet input, its T-type calcium conductance
DE-INACTIVATES; the next rise then triggers a BURST -- a nonlinear amplification that reports
CHANGE. Held depolarized by sustained input, the conductance stays inactivated and the cell relays
TONICALLY, roughly faithfully. So the same visceral value arrives at the insula LARGER when it is
new and FAITHFUL when it has been there a while.

That is exactly the right thing for this tract to do. The insula computes deviation from its own
running baseline; a relay that emphasises change hands it a signal whose novelty is already
foregrounded, and a body event that has persisted arrives unamplified -- which is the same
habituation the terminus applies one hop later, applied at the hop where physiology puts it.

P1 -- NO NEW CONSTANT. Transmission is `m + (1 + h*novelty) * rise`, where `m` is the channel's
own running mean, `rise` its excess over that mean, `novelty` the excess of `rise` over the
channel's own running mean OF RISES, and `h` in [0, 1] the available de-inactivation. Every term is
one of the brain's own running statistics, and the ONE rate is DERIVED from a stated horizon
exactly as `faculties/interoceptive_integration` derives its alpha -- `alpha = 2 / (horizon + 1)`,
same default horizon, so the two hops of this tract share one time base instead of two tuned ones.
The amplification factor is bounded by 2 STRUCTURALLY (both `h` and `novelty` are in [0, 1]), not
by a chosen ceiling.

TWO MEASURED LIMITS, STATED RATHER THAN DISCOVERED LATER.
  * SATURATION. Output is clipped to the unit interval the ports declare, so above a step of about
    0.45 from a quiet baseline the relay hits 1.0 and stops being graded (measured from a 0.05
    baseline: 0.10 -> 0.136, 0.30 -> 0.520, 0.40 -> 0.712, 0.60 and up -> 1.0). The LIVE channels
    do not reach that: at the measured live peaks (stress 0.192, caution 0.382) the same step
    relays 0.247 and 0.480, well clear of the ceiling.
  * TIMESCALE. This is NOT millisecond burst kinetics and the name should not be read as claiming
    them. Sharing the terminus's 200-tick horizon, amplification decays to within 5% of faithful
    in about 70 ticks and 1% in about 90. What is faithful to the physiology is the DISTINCTION --
    transmission mode set by recent history -- not the time constant.

Representation-only: no arm import, nothing under `brainstem/`. Deterministic: no RNG at all --
the relay is a state machine over its input, so replay is byte-identical by construction and it
adds nothing to the G5 snapshot. ASCII.

  PYTHONPATH=src python -m snn_data_lake.faculties.interoceptive_relay   # self-check, exit 0
"""

from __future__ import annotations

import sys

#: this module's anatomical id in connectome.NODES -- the atlas-node rule (T024): a node id must be
#: referenced by >= 1 module in its region, and `thal_relay` had no claimant until this module.
CONNECTOME_NODES = ("thal_relay",)

#: the tract hop this module IS, named so the two neighbours can be read off in one place
TRACT = ("nts_rvlm", "thal_relay", "insula")

#: the ascending channel layout, matching `interoceptive_arousal.arousal_out`'s (2,) VEC_DENSE
RELAY_CHANNELS = ("stress", "caution")

#: the declared edge this module CONSUMES (its own IN port; the producer declares only the port)
VISCERAL_EDGE = ("interoceptive_arousal", "arousal_out", "interoceptive_relay", "visceral_in")


def declare_ports(manifest):
    """Manifest registrant (specs/022 T039j): the relay's IN port takes the medullary ascending
    pair and its OUT port carries the transmitted pair onward to the insula terminus.

    The `relay_out -> body_in` edge is declared by the INSULA, keeping this tree's convention that
    the CONSUMER declares the edge (its subscription, its join) -- the same split
    `interoceptive_arousal` documents on its own OUT port."""
    from snn_data_lake.engine.wiring_manifest import Port

    manifest.declare_port(
        "interoceptive_relay",
        Port(
            name="visceral_in",
            region="thalamus",
            shape=(len(RELAY_CHANNELS),),
            dtype="f4",
            unit="level01",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="interoceptive_relay",
            join_policy="concat",
        ),
    )
    manifest.declare_port(
        "interoceptive_relay",
        Port(
            name="relay_out",
            region="thalamus",
            shape=(len(RELAY_CHANNELS),),
            dtype="f4",
            unit="level01",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="interoceptive_relay",
            delivery_policy="all-or-none",
        ),
    )
    manifest.declare_edge(*VISCERAL_EDGE)


def _visceral_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `visceral_in` port, one pair per replay arm.
    slots=256 for the same reason the insula uses it: the schedule publishes well under that and
    a silent ring overrun would make a partial record look like a smaller measurement."""
    from snn_data_lake.engine.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.engine.wiring_manifest import WiringManifest

    m = WiringManifest()
    from snn_data_lake.faculties.interoceptive_arousal import declare_ports as _producer_ports

    _producer_ports(m)
    declare_ports(m)
    port = m.ports[("interoceptive_relay", "visceral_in")]
    ring = SignalRing(slots=256)
    return ring, JoinConsumer(ring, port)


def consume_visceral(consumer):
    """The downstream computation under measurement: each joined ascending frame is TRANSMITTED
    and the relayed pair recorded. LOSS IS FATAL -- a measurement over a partial record is a
    different experiment, not a smaller one."""
    relay = InteroceptiveRelay()
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        out.append((tick, [round(v, 6) for v in relay.relay([float(x) for x in vec])]))
    return out


def _measure_edge_charge():
    """The per-edge replay-twice A/B for `arousal_out -> visceral_in` -> (live, pinned, reads,
    responded). `responded` = the relayed pair rose at the challenge, so the medullary payload
    did not merely diverge, it MOVED this hop in the direction the tract claims."""
    from snn_data_lake.instruments.port_charge_probe import replay_twice, visceral_challenge_frames

    frames, spike_at = visceral_challenge_frames()
    live, pinned_out, reads = replay_twice(
        frames, _visceral_consumer_factory, consume_visceral, [0.2] * len(RELAY_CHANNELS)
    )
    pre = max(max(v) for t, v in live if spike_at - 5 <= t < spike_at)
    post = max(max(v) for t, v in live if t >= spike_at)
    return live, pinned_out, reads, post > max(pre * 2, 0.05)


# distinct-from: faculties/interoceptive_integration.record_charge -- that measures the two edges
#   landing ON the terminus; this measures the one edge landing on the HOP BEFORE it. Both
#   delegate to port_charge_probe.record_attribution, THE one writer. Read.
def record_charge():
    """Measure the ascending A/B and record its attribution via THE one writer. REFUSES a
    non-diverged or non-responding run."""
    from snn_data_lake.instruments.port_charge_probe import record_attribution

    live, pinned_out, reads, responded = _measure_edge_charge()
    if live == pinned_out or not responded:
        print(f"record_charge REFUSED for {VISCERAL_EDGE}: no divergence or no response")
        return 1
    path = record_attribution(
        VISCERAL_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: the real medullary ascending pair vs a pinned quiet vector; the relayed "
        "output diverged and rose at the challenge -- the transmitted signal carries THIS "
        "producer's payload, not the schedule. Took over the medulla's charge when T039j moved "
        "the ascending edge off the insula and onto this hop; self-check re-measures and "
        "requires `measured` to reproduce",
    )
    print(f"recorded {VISCERAL_EDGE} into {path}")
    return 0


# distinct-from: faculties/reticular_gate.py -- the TRN, the inhibitory GATE ON relay (attentional
#   gain redistribution across channels); this is the RELAY ITSELF and its transmission mode. Its
#   own docstring draws the line: "the thalamus is a RELAY, but the TRN is what makes it a GATE".
# distinct-from: faculties/connectome_router.py -- routes by declared topology, transmits nothing.
# distinct-from: faculties/interoceptive_integration.py -- the cortical TERMINUS, computing deviation
#   from baseline; this is the hop BEFORE it. distinct-from: faculties/interoceptive_arousal.py --
#   the SOURCE that senses the body. All sites read.
class InteroceptiveRelay:
    """Tonic/burst thalamic transmission of the ascending visceral pair."""

    def __init__(self, channels=RELAY_CHANNELS, horizon=200):
        # P1: the ONE rate is DERIVED from a stated horizon, never picked -- the same derivation
        # (and the same default horizon, the daemon's day-period order) the insula terminus uses,
        # so the two hops of this tract share one time base instead of two tuned ones.
        self.channels = tuple(channels)
        self.horizon = int(horizon)
        self.alpha = 2.0 / (self.horizon + 1.0)
        self._mean = dict.fromkeys(self.channels, 0.0)
        self._rise = dict.fromkeys(self.channels, 0.0)  # running mean of the channel's own rises
        self._deinact = dict.fromkeys(self.channels, 0.0)  # h in [0,1]: T-type availability
        self._seen = 0
        self._last_mode = {}  # per channel: "burst" | "tonic" -- reported, never an input

    def relay(self, vec) -> list:
        """Transmit one ascending frame. Returns the relayed vector, same layout, clipped to [0,1].

        BURST: a value arriving ABOVE the channel's running mean is amplified by the
        de-inactivation currently available, which the channel accumulated while sitting BELOW its
        mean. TONIC: sustained input keeps de-inactivation near zero, so the relay is faithful.
        A channel that has never been quiet cannot burst, which is the physiology and is also what
        stops this from being a gain applied to everything.
        """
        vals = [float(v) for v in vec]
        out = []
        for ch, x in zip(self.channels, vals, strict=False):
            m = self._mean[ch]
            h = self._deinact[ch]
            rise = max(0.0, x - m)
            # NOVELTY IS THE RISE AGAINST THE CHANNEL'S OWN RISES, not against a threshold. An EMA
            # mean converges to a constant input FROM BELOW and never reaches it, so a naive
            # `x <= mean` test for "quiet" is false forever and de-inactivation never recovers --
            # measured on the first draft, where the burst could not fire at all. Both moves are
            # therefore CONTINUOUS in novelty, which has no comparison and no constant in it.
            r_bar = self._rise[ch]
            # NOVELTY IS THE EXCESS OVER THE CHANNEL'S TYPICAL RISE, not the rise itself. A plain
            # ratio rise/(rise + r_bar) sits at ~0.5 in steady state, because a settled channel's
            # rise IS its own mean rise -- measured: the never-quiet control burst anyway, so the
            # relay amplified everything a little and had no honest tonic mode at all.
            excess = max(0.0, rise - r_bar)
            novelty = excess / (rise + r_bar) if (rise + r_bar) > 0.0 else 0.0
            # THE BURST AT MOST DOUBLES THE DEVIATION, and the bound is structural rather than
            # tuned: the factor is (1 + h*novelty) with both h and novelty in [0, 1]. Amplifying
            # toward the ceiling instead -- x + h*novelty*(1-x) -- was rejected because a full
            # burst then lands on exactly 1.0 for EVERY step, which destroys the gradedness the
            # terminus one hop later needs.
            transmitted = min(1.0, max(0.0, m + (1.0 + h * novelty) * rise))
            self._last_mode[ch] = "burst" if (h * novelty * rise) > 0.0 else "tonic"
            # de-inactivation relaxes toward (1 - novelty): a channel with nothing novel re-arms,
            # a channel that just burst has spent it. One derived rate for both directions.
            h += self.alpha * ((1.0 - novelty) - h)
            self._deinact[ch] = min(1.0, max(0.0, h))
            self._rise[ch] = r_bar + self.alpha * (rise - r_bar)
            self._mean[ch] = m + self.alpha * (x - m)
            out.append(transmitted)
        self._seen += 1
        return out

    def mode(self, channel) -> str:
        """The transmission mode this channel was last in. REPORTED, never fed back in."""
        return self._last_mode.get(channel, "tonic")

    def state(self) -> dict:
        return {
            "channels": {
                ch: {
                    "mean": self._mean[ch],
                    "deinactivation": self._deinact[ch],
                    "rise_mean": self._rise[ch],
                }
                for ch in self.channels
            },
            "seen": self._seen,
            "horizon_ticks": self.horizon,
        }


def main() -> int:
    quiet = [0.05, 0.05]
    step = [0.60, 0.60]

    # (1) THE BURST: quiet long enough to de-inactivate, then a step -> transmitted ABOVE the input.
    r = InteroceptiveRelay()
    for _ in range(300):
        r.relay(quiet)
    burst = r.relay(step)
    if not (burst[0] > step[0] and r.mode("stress") == "burst"):
        print(
            f"interoceptive_relay self-check FAILED: a step after quiet did not burst "
            f"(in={step[0]:.4f} out={burst[0]:.4f} mode={r.mode('stress')})"
        )
        return 1

    # (2) THE TONIC MODE: hold the step; transmission must fall back toward faithful.
    for _ in range(300):
        held = r.relay(step)
    if not (held[0] < burst[0] and abs(held[0] - step[0]) < abs(burst[0] - step[0])):
        print(
            f"interoceptive_relay self-check FAILED: sustained input did not relax to tonic "
            f"(burst={burst[0]:.4f} held={held[0]:.4f} in={step[0]:.4f})"
        )
        return 1

    # (3) THE RED (constructed): a PASS-THROUGH relay -- de-inactivation pinned at 0 -- must be
    # unable to produce (1). If it can, this module's whole claim is a rename of the identity.
    flat = InteroceptiveRelay()
    for _ in range(300):
        flat.relay(quiet)
    flat._deinact = dict.fromkeys(flat.channels, 0.0)
    flat_out = flat.relay(step)
    if flat_out[0] > step[0]:
        print(
            "interoceptive_relay self-check FAILED: the pass-through red did not fire -- a relay "
            f"with no de-inactivation still amplified ({flat_out[0]:.4f} > {step[0]:.4f})"
        )
        return 1

    # (4) A CHANNEL THAT WAS NEVER QUIET CANNOT BURST -- the guard that stops this being a gain
    # applied to everything.
    hot = InteroceptiveRelay()
    for _ in range(300):
        hot.relay(step)
    if hot.relay(step)[0] > step[0]:
        print("interoceptive_relay self-check FAILED: a never-quiet channel burst anyway")
        return 1

    # (5) THE BOUND: no input in [0,1] may leave it.
    edge = InteroceptiveRelay()
    for _ in range(300):
        edge.relay([0.0, 0.0])
    for v in ([1.0, 1.0], [0.0, 0.0], [0.5, 1.0]):
        got = edge.relay(v)
        if not all(0.0 <= g <= 1.0 for g in got):
            print(f"interoceptive_relay self-check FAILED: {v} relayed to {got}, outside [0,1]")
            return 1

    # (6) DETERMINISM: no RNG anywhere, so two relays fed the same frames agree byte for byte.
    a, b = InteroceptiveRelay(), InteroceptiveRelay()
    frames = [quiet] * 40 + [step] * 10 + [quiet] * 10
    if [a.relay(f) for f in frames] != [b.relay(f) for f in frames]:
        print("interoceptive_relay self-check FAILED: two relays diverged on identical input")
        return 1

    # (7) THE REAL COMPOSITION, END TO END, and this is the check that closes the two declared
    # connectome pairs rather than asserting them. The frames come from a REAL medullary
    # InteroceptiveArousal and the relayed pair is fed to a REAL insula terminus, so
    # (medulla, thalamus) and (thalamus, insula) become import paths the p6 connectome diff can
    # see. Both pairs are DECLARED, which is what makes these imports legal here when the insula
    # importing the medulla directly was not: that created (medulla, insula), which the connectome
    # declares no route for, and the undeclared-pairs ratchet reddened exactly as designed.
    from snn_data_lake.faculties.interoceptive_arousal import InteroceptiveArousal
    from snn_data_lake.faculties.interoceptive_integration import (
        BODY_CHANNELS,
        InteroceptiveIntegration,
    )

    organ = InteroceptiveArousal(seed=0)
    quiet_v = organ.sense(0.0)
    loaded = organ.sense(3.0)
    direct = InteroceptiveIntegration(channels=BODY_CHANNELS)
    relayed = InteroceptiveIntegration(channels=BODY_CHANNELS)
    hop = InteroceptiveRelay()

    def _frame(a, vec):
        return dict(
            energy_deficit=0.0,
            curiosity_deficit=0.0,
            social_deficit=0.0,
            stress=vec[0],
            caution=vec[1],
        )

    q = [quiet_v["stress"], quiet_v["caution"]]
    for _ in range(300):
        direct.observe(**_frame(direct, q))
        relayed.observe(**_frame(relayed, hop.relay(q)))
    hit = [loaded["stress"], loaded["caution"]]
    s_direct = direct.observe(**_frame(direct, hit))
    s_relayed = relayed.observe(**_frame(relayed, hop.relay(hit)))
    if not (s_relayed > s_direct):
        print(
            "interoceptive_relay self-check FAILED: routing the real medullary signal through the "
            f"relay did not change the real terminus's salience (direct={s_direct:.4f}, "
            f"relayed={s_relayed:.4f}) -- a relay that changes nothing downstream is a wire, and "
            "P3 says presence is not participation"
        )
        return 1

    # (8) THE EDGE IS CHARGED, measured, and the committed attribution must reproduce.
    from snn_data_lake.instruments.port_charge_probe import read_attribution

    live, pinned_out, reads, responded = _measure_edge_charge()
    if reads != len(live):
        print(f"interoceptive_relay self-check FAILED: {reads} reads != {len(live)}")
        return 1
    if live == pinned_out:
        print(
            "interoceptive_relay self-check FAILED: pinning the ascending payload did not change "
            "what this hop transmits -- a declared, not a charged, edge"
        )
        return 1
    if not responded:
        print(
            "interoceptive_relay self-check FAILED: the ascending challenge did not raise the "
            "relayed pair -- divergence without response attributes nothing"
        )
        return 1
    rec = read_attribution(VISCERAL_EDGE)
    if rec is None or rec.get("diverged") is not True:
        print(
            f"interoceptive_relay self-check FAILED: no committed attribution for "
            f"{VISCERAL_EDGE} -- regenerate with --record-charge and commit it"
        )
        return 1
    if rec.get("measured") != {"reads": reads, "n_frames": len(live)}:
        print(
            f"interoceptive_relay self-check FAILED: committed attribution does not reproduce "
            f"({rec.get('measured')} vs {{'reads': {reads}, 'n_frames': {len(live)}}})"
        )
        return 1

    print(
        f"interoceptive_relay self-check OK: burst {step[0]:.2f} -> {burst[0]:.4f} after quiet; "
        f"tonic relaxation -> {held[0]:.4f} under sustained input; pass-through red fired; a "
        f"never-quiet channel cannot burst; range held; deterministic. REAL COMPOSITION: the "
        f"medullary organ's loaded reading through this hop raised the real insula terminus's "
        f"salience {s_direct:.4f} -> {s_relayed:.4f}. ASCENDING EDGE CHARGED: the medullary "
        f"payload moved this hop over {len(live)} frames and the committed attribution "
        f"reproduces. Tract hop {TRACT[0]} -> {TRACT[1]} -> {TRACT[2]}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(record_charge() if "--record-charge" in sys.argv else main())
