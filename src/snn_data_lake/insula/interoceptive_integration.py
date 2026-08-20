"""Interoceptive integration -- the insula's first faculty (specs/022 T021a): body-state
signals arriving over the declared tract become a bounded, salience-ready feeling state.

THE TRACT THIS MODULE TERMINATES (each hop a declared `connectome` edge, T021a):
`viscera -> nts_rvlm -> thal_relay -> insula -> {gwt, ctx_pfc}` -- baroreceptor/visceral
afferents through the medullary NTS/RVLM complex, relayed by thalamus into THIS node
(`insula` in connectome.NODES is this module's anatomical id), projecting onward to the
global workspace (interoceptive ignition afferent) and prefrontal cortex. The upstream
modules exist and are charged (`medulla/interoceptive_arousal`, `medulla/vital_nuclei`);
this module is the cortical terminus that was missing.

WHAT IT COMPUTES, minimally and honestly: a running interoceptive STATE -- per-channel
exponential means with deviation tracking -- and a scalar `feeling_salience` in [0, 1]:
how far the body currently sits from its own running baseline (P1: the baseline is the
brain's own statistics, never a fixed set-point). Deviation-from-self is the standard
interoceptive-salience account (the insula responds to PREDICTION MISMATCH in body state,
not to raw intensity), and it is the quantity the US1 coalition channels consume
(`arousal`/`valence` producers, T026): a body far from baseline should find it easier to
ignite the workspace.

WIRED AT T026 (specs/022): this module is now a manifest registrant. IN `body_in` joins the
two declared interoceptive producers by CONCAT in producer-sorted order --
homeostatic_drives' three normalized deficits, then interoceptive_arousal's [stress,
caution] -- so BODY_CHANNELS below IS the concat layout, stated once and asserted in the
self-check. OUT `interoception_out` (feeling_salience) is the interoception channel the
dormancy exemption promised; its coalition consumer lands with the workspace resolution
(T030/T031), so the OUT port is declared edge-less today and this module remains off the
static live-import graph until that consumer imports it (the exemption row says exactly
that). Charge is MEASURED per edge: two replay-twice A/Bs, each varying ONE producer's
payload while the other stays constant, so divergence attributes to THAT edge -- a single
whole-join neuter could not separate the two producers.
"""

# distinct-from: medulla/interoceptive_arousal.py -- the BRAINSTEM half (arousal drive from
#   visceral afferents, already charged); this is the CORTICAL terminus computing
#   deviation-from-baseline salience. distinct-from: cortex/affect_core.py -- valence/arousal
#   AFFECT dynamics downstream of appraisal; this is upstream body-state integration.
# distinct-from: hypothalamus/set_points.py -- homeostatic TARGETS; this tracks the brain's
#   own running statistics, never fixed set-points. All sites read.

from __future__ import annotations

import sys

#: this module's anatomical ids in connectome.NODES (the atlas-node rule, T024: a node id
#: must be referenced by >=1 module in its region -- these are those references for insula)
CONNECTOME_NODES = ("insula",)
TRACT = ("viscera", "nts_rvlm", "thal_relay", "insula", "gwt")

#: the concat layout of `body_in`, producer-sorted (signal_ring's concat order):
#: homeostatic_drives (3 normalized deficits) then interoceptive_arousal ([stress, caution])
BODY_CHANNELS = ("energy_deficit", "curiosity_deficit", "social_deficit", "stress", "caution")

#: the declared edges, exactly as the manifest carries them -- the charge evidence cites them
AROUSAL_EDGE = ("interoceptive_arousal", "arousal_out", "interoceptive_integration", "body_in")
DRIVES_EDGE = ("homeostatic_drives", "drive_deficit_out", "interoceptive_integration", "body_in")


def declare_ports(manifest):
    """Manifest registrant (specs/022 T026, the consumer half): `body_in` (concat over the
    two interoceptive producers) + both edges, and `interoception_out` -- the interoception
    channel producer the T021a exemption promised (its coalition consumer lands T030/T031,
    so it carries no edge yet; a declared-ahead OUT is the anatomy, not force-wiring)."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "interoceptive_integration",
        Port(
            name="body_in",
            region="insula",
            shape=(len(BODY_CHANNELS),),
            dtype="f4",
            unit="level01",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="interoceptive_integration",
            join_policy="concat",
        ),
    )
    manifest.declare_port(
        "interoceptive_integration",
        Port(
            name="interoception_out",
            region="insula",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="interoceptive_integration",
            delivery_policy="all-or-none",
        ),
    )
    manifest.declare_edge(*DRIVES_EDGE)
    manifest.declare_edge(*AROUSAL_EDGE)


def _body_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `body_in` port, one pair per replay arm.
    slots=256: the 40-tick schedule publishes 80 frames, and the review's ring-overrun
    finding proved 64 slots silently dropped ticks 1-8 while every counter still agreed --
    capacity is now derived-with-headroom, and consume_body treats ANY loss as fatal."""
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    port = m.ports[("interoceptive_integration", "body_in")]
    ring = SignalRing(slots=256)
    return ring, JoinConsumer(ring, port)


def consume_body(consumer):
    """The downstream computation under measurement: each joined body frame feeds observe()
    and yields that tick's feeling_salience. Returns [(tick, salience), ...]. LOSS IS
    FATAL here: a measurement over a partial record is not a smaller measurement, it is a
    different experiment (the review caught 16 of 80 frames silently dropped with every
    downstream counter still self-consistent)."""
    ii = InteroceptiveIntegration(channels=BODY_CHANNELS)
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        sal = ii.observe(**dict(zip(BODY_CHANNELS, (float(x) for x in vec))))
        out.append((tick, round(sal, 6)))
    return out


# distinct-from: basal_ganglia/evidence_accumulation.py -- serotonin's single-edge A/B;
#   this is the body wire's per-edge pair over the instrument-layer schedules. Read.
def _measure_edge_charge(vary):
    """One per-edge replay-twice A/B: -> (live, pinned_out, reads, responded) where
    `responded` = the live salience at the challenge exceeds the settled pre-challenge
    salience (the varying producer moved the insula; the frozen one could not have).
    The frame schedules live in the INSTRUMENT layer
    (port_charge_probe.interoceptive_challenge_frames) -- importing the producers from
    here put two undeclared region pairs on the import graph and the p6 ratchet went red;
    the declared tract relays through thalamus, and code was not allowed to shortcut it."""
    from snn_data_lake.substrate.port_charge_probe import (
        interoceptive_challenge_frames,
        replay_twice,
    )

    frames, spike_at = interoceptive_challenge_frames(vary)
    live, pinned_out, reads = replay_twice(
        frames, _body_consumer_factory, consume_body, [0.2] * len(BODY_CHANNELS)
    )
    pre = max(s for t, s in live if spike_at - 5 <= t < spike_at)
    post = max(s for t, s in live if t >= spike_at)
    return live, pinned_out, reads, post > max(pre * 2, 0.05)


class InteroceptiveIntegration:
    """Per-channel running baselines + deviation -> one bounded feeling_salience."""

    def __init__(self, channels=("cardio", "respiratory", "gut"), alpha=None):
        # P1: alpha is DERIVED from the horizon the baseline should span, not picked -- the
        # caller states a horizon in ticks and alpha follows; the default horizon is the
        # daemon's own day period order (~200 ticks), stated rather than hidden.
        self.horizon = 200
        self.alpha = (2.0 / (self.horizon + 1.0)) if alpha is None else float(alpha)
        self.channels = tuple(channels)
        self._mean = dict.fromkeys(self.channels, 0.0)
        self._var = dict.fromkeys(self.channels, 1.0)
        self._seen = 0

    def observe(self, **readings) -> float:
        """Feed one tick of body-state readings; returns feeling_salience in [0, 1] -- the
        normalized deviation of the body from its OWN running baseline."""
        devs = []
        for ch in self.channels:
            if ch not in readings:
                continue
            x = float(readings[ch])
            m, v = self._mean[ch], self._var[ch]
            d = x - m
            self._mean[ch] = m + self.alpha * d
            self._var[ch] = (1.0 - self.alpha) * (v + self.alpha * d * d)
            devs.append(abs(d) / (self._var[ch] ** 0.5 + 1e-9))
        self._seen += 1
        if not devs or self._seen <= 2:
            return 0.0  # a baseline two samples old is not a baseline; silence over noise
        z = sum(devs) / len(devs)
        return z / (1.0 + z)  # bounded, monotone in deviation

    def state(self) -> dict:
        return {
            "channels": {
                ch: {"mean": self._mean[ch], "var": self._var[ch]} for ch in self.channels
            },
            "seen": self._seen,
            "horizon_ticks": self.horizon,
        }


def main() -> int:
    ii = InteroceptiveIntegration()
    # settle a stable body: salience must fall toward quiet
    for _ in range(300):
        late_quiet = ii.observe(cardio=1.0, respiratory=0.5, gut=0.2)
    # a real perturbation: salience must RISE (the sensitivity half)
    spike = ii.observe(cardio=3.0, respiratory=0.5, gut=0.2)
    if not (spike > late_quiet * 2 and spike > 0.2):
        print(
            f"interoceptive_integration self-check FAILED: perturbation did not raise "
            f"salience (quiet={late_quiet:.4f}, spike={spike:.4f})"
        )
        return 1
    # the RED (constructed): a dead-baseline variant that never updates its mean would call
    # the SAME steady input salient forever -- deviation-from-self is the criterion, and
    # freezing the self must break it.
    frozen = InteroceptiveIntegration(alpha=0.0)
    for _ in range(50):
        frozen_sal = frozen.observe(cardio=1.0)
    if frozen_sal <= late_quiet:
        print(
            "interoceptive_integration self-check FAILED: the frozen-baseline red did not "
            f"fire (frozen={frozen_sal:.4f} vs live-quiet={late_quiet:.4f}) -- the criterion "
            "cannot distinguish a dead baseline from a settled one"
        )
        return 1
    # re-baseline: after the spike regime persists, salience must fall again (adaptation --
    # the insula habituates to the new normal; a fixed set-point could not)
    for _ in range(600):
        adapted = ii.observe(cardio=3.0, respiratory=0.5, gut=0.2)
    if not (adapted < spike / 2):
        print(
            f"interoceptive_integration self-check FAILED: no habituation to the new normal "
            f"(spike={spike:.4f} -> adapted={adapted:.4f})"
        )
        return 1
    # ---- THE BODY WIRE IS CHARGED, per edge, measured (specs/022 T026) -----------------
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    for edge, vary in ((AROUSAL_EDGE, "arousal"), (DRIVES_EDGE, "drives")):
        live, pinned_out, reads, responded = _measure_edge_charge(vary)
        if reads != len(live):
            print(f"interoceptive_integration self-check FAILED: {reads} reads != {len(live)}")
            return 1
        if live == pinned_out:
            print(
                f"interoceptive_integration self-check FAILED: pinning the joined body "
                f"payload did not change salience (vary={vary}) -- a declared, not a "
                f"charged, edge"
            )
            return 1
        if not responded:
            print(
                f"interoceptive_integration self-check FAILED: the {vary} challenge did "
                f"not raise live salience -- divergence without response attributes nothing"
            )
            return 1
        rec = read_attribution(edge)
        if rec is None or rec.get("diverged") is not True:
            print(
                f"interoceptive_integration self-check FAILED: no committed attribution "
                f"for {edge} -- regenerate with --record-charge and commit it"
            )
            return 1
        fresh = {"reads": reads, "n_frames": len(live)}
        if rec.get("measured", {}) != fresh:
            print(
                f"interoceptive_integration self-check FAILED: recorded {rec.get('measured')} "
                f"does not reproduce fresh {fresh} -- regenerate with --record-charge"
            )
            return 1
    print(
        "interoceptive_integration self-check OK: quiet body -> low salience "
        f"({late_quiet:.4f}); perturbation -> {spike:.4f}; frozen-baseline red fired; "
        f"habituation to the new normal -> {adapted:.4f}. Tract terminus for "
        f"{' -> '.join(TRACT)}. BODY WIRE CHARGED per edge: arousal and drives challenges "
        "each diverged live-vs-pinned with every frame read and raised salience; committed "
        "attributions reproduce."
    )
    return 0


# distinct-from: basal_ganglia/evidence_accumulation.py -- its record_charge measures the
#   SEROTONIN wire's single edge; this measures the BODY wire's TWO edges with per-edge
#   vary-one-freeze-one schedules. Both delegate to port_charge_probe.record_attribution,
#   THE one writer. Read.
def record_charge():
    """Measure both per-edge A/Bs and record their attributions via THE one writer
    (port_charge_probe.record_attribution). REFUSES a non-diverged or non-responding run."""
    from snn_data_lake.substrate.port_charge_probe import record_attribution

    for edge, vary in ((AROUSAL_EDGE, "arousal"), (DRIVES_EDGE, "drives")):
        live, pinned_out, reads, responded = _measure_edge_charge(vary)
        if live == pinned_out or not responded:
            print(f"record_charge REFUSED for {edge}: no divergence or no response")
            return 1
        path = record_attribution(
            edge,
            {"reads": reads, "n_frames": len(live)},
            f"per-edge replay_twice: only the {vary} producer varied (the other frozen at "
            "its quiet vector), live diverged from the pinned join and salience rose at "
            "the challenge -- divergence attributes to THIS edge; self-check re-measures "
            "and requires `measured` to reproduce",
        )
        print(f"recorded {edge} into {path}")
    return 0


if __name__ == "__main__":
    if "--record-charge" in sys.argv[1:]:
        sys.exit(record_charge())
    sys.exit(main())
