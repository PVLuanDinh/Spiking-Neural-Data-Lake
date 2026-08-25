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
dormancy exemption promised.

WIRED AND CHARGED AT T039f, AND THE PROMISE THIS PARAGRAPH USED TO CARRY WAS NOT KEPT AS
WRITTEN. It said the coalition consumer would land with the workspace resolution (T030/T031)
and that this module would stay off the static live-import graph until it did. T030 and T031
both closed and it did not land -- because T039b MEASURED that sink out of range: the live bus
admits on a band whose floor is 0.5505 and `feeling_salience` peaks at 0.2798, so an
interoceptive bid never competes, and pinning the signal to its own live mean left the brain
byte-identical on 120/120 ticks. That route is REFUSED, not pending. What landed instead is the
sink this module's own opening paragraph names -- a body far from baseline finding it EASIER TO
IGNITE, i.e. THRESHOLD MODULATION: `cortex/brain.py` feeds this terminus the declared
`body_in` frame every tick and passes the salience into the ONE ignition law
(`region_coherence._adjust`, joined by noisy-OR so it adds no constant). The module is now on
the live-import graph and the region is CHARGED.

CLOSED AT T039j, and this paragraph is corrected rather than deleted because it was WRITTEN
here at T039f and falsified two commits later by the same session -- the comment-outliving-its-
body shape, authored in the very module it describes. It said `interoception_out` was STILL
DECLARED EDGE-LESS and that the manifest did not yet say what the anatomy did. It now does:
`substrate/region_coherence` declares the typed IN port (FR-012/FR-013) the edge needed, and
`interoception_out -> region_coherence.interoception_in` is declared and charge-measured. The
ascending half likewise relays through `thalamus/interoceptive_relay` rather than arriving
straight from the medulla. Charge is MEASURED per edge: two replay-twice A/Bs, each varying ONE producer's
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

#: the name this faculty SUBSCRIBES to the re-entrant workspace under (specs/022 T039h). One
#: statement, read by the brain when it subscribes and by `_top_down` when it drains, so the two
#: cannot drift apart into a subscription nobody reads or a read nobody registered.
SPECIALIST = "interoception"

#: this module's anatomical ids in connectome.NODES (the atlas-node rule, T024: a node id
#: must be referenced by >=1 module in its region -- these are those references for insula)
CONNECTOME_NODES = ("insula",)
TRACT = ("viscera", "nts_rvlm", "thal_relay", "insula", "gwt")

#: the concat layout of `body_in`, producer-sorted (signal_ring's concat order):
#: homeostatic_drives (3 normalized deficits) then interoceptive_arousal ([stress, caution])
BODY_CHANNELS = ("energy_deficit", "curiosity_deficit", "social_deficit", "stress", "caution")

#: the declared edges, exactly as the manifest carries them -- the charge evidence cites them.
#: T039j: the arousal half now arrives RELAYED. The connectome always declared
#: `nts_rvlm -> thal_relay -> insula`; until the relay existed the manifest short-cut the middle
#: hop and this constant named the medulla directly. The producer-sorted concat layout is
#: UNCHANGED -- `homeostatic_drives` still sorts before `interoceptive_relay` exactly as it sorted
#: before `interoceptive_arousal` -- so BODY_CHANNELS below is the same layout it always was.
RELAY_EDGE = ("interoceptive_relay", "relay_out", "interoceptive_integration", "body_in")
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
    manifest.declare_edge(*RELAY_EDGE)


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
        # P1: alpha is DERIVED from the horizon this baseline spans -- fixed here at the
        # daemon's own day-period order (~200 ticks), with alpha following from it; a caller
        # that needs a different timescale overrides alpha directly. (US1 audit M3: this
        # comment used to describe a `horizon` PARAMETER the constructor never accepted.)
        self.horizon = 200
        self.alpha = (2.0 / (self.horizon + 1.0)) if alpha is None else float(alpha)
        self.channels = tuple(channels)
        self._mean = dict.fromkeys(self.channels, 0.0)
        # P1: the variance prior is DERIVED from the declared port contract, not picked. The
        # production feed is body_in, unit level01, and the largest variance any [0,1] signal
        # can have is 0.25 (the 50/50 two-point mass at the ends) -- so the prior is the
        # declared unit's own upper bound and the EMA can only move it toward the body's real
        # spread. The retired seed was 1.0: 4x anything the port can produce, ~30x the settled
        # live variance, and still suppressing salience ~31 percent at a tick-500 challenge
        # (US1 audit M0; the 0.0 seed was measured WORSE -- a near-maximal ignition subsidy at
        # tick 3, because the _seen guard covers only two ticks). Self-check carries the red.
        self._var = dict.fromkeys(self.channels, 0.25)
        self._seen = 0
        self._last_salience = 0.0  # T039i: this faculty's own last output, for it to report
        self._last_top_down = 0.0  # US1 audit B4: the term observe() last APPLIED, for reporters

    def attach(self, brain):
        """Hold the brain whose re-entrant workspace this faculty is subscribed to (T039h).

        The handle is the SUBSCRIBER side of a GWT specialist, not ownership: this module
        constructs no workspace and submits nothing to one. Detached (the default) the top-down
        term is 0.0 and every salience is what T039f computed, so the module's own self-check,
        the per-edge charge probes and every offline caller are untouched."""
        self._brain = brain

    def _top_down(self) -> float:
        """The bias the workspace is currently broadcasting to THIS specialist, or 0.0.

        `broadcast_to(name)` returns {} when nothing is ignited or the name is not subscribed, so
        a detached insula, an unsubscribed one, and a quiet workspace all read 0.0 -- one value for
        three situations that are genuinely the same situation here: no top-down signal.

        The read is deliberately written through `Brain.reentrant`, the PUBLIC accessor, rather
        than through `brain._reentrant`. Until T039g the p6 walk could only see the private form,
        which meant its own success criterion rewarded reaching past a class into its state; the
        accessor is the honest route and is now the visible one."""
        brain = getattr(self, "_brain", None)
        if brain is None:
            return 0.0
        try:
            bias = brain.reentrant.broadcast_to(SPECIALIST)
        except Exception:
            return 0.0
        if not bias:
            return 0.0
        content, mag = max(bias.items(), key=lambda kv: kv[1])
        # POST-CONDITION: tell the bus this specialist USED what it was given. T039c's listener
        # term has a writer only where a faculty reports, and a specialist that consumes without
        # reporting is exactly the declaration-instead-of-measurement P3 refuses. `language` stays
        # the never-reporting control.
        try:
            brain.reentrant.report_use(SPECIALIST, content)
        except Exception:
            pass
        return max(0.0, float(mag))

    def observe(self, **readings) -> float:
        """Feed one tick of body-state readings; returns feeling_salience in [0, 1] -- the
        normalized deviation of the body from its OWN running baseline, scaled by whatever the
        global workspace is broadcasting to this specialist (T039h; 0.0 when detached)."""
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
            self._last_salience = 0.0
            self._last_top_down = 0.0  # no term was applied on a tick that computed nothing
            return 0.0  # a baseline two samples old is not a baseline; silence over noise
        z = sum(devs) / len(devs)
        # TOP-DOWN ATTENTION ENTERS BEFORE THE BOUND, NOT AFTER IT (T039h). Scaling z and then
        # squashing keeps the output in [0, 1) by construction -- the same guarantee the
        # unattended form had -- whereas scaling the squashed value would need a clamp, and a
        # clamp here would silently hide an unbounded upstream bias instead of exposing it.
        # attention 0.0 => z unchanged => byte-identical to the pre-T039h salience.
        td = self._top_down()
        self._last_top_down = td
        z = z * (1.0 + td)
        self._last_salience = z / (1.0 + z)  # bounded, monotone in deviation
        return self._last_salience

    def last_salience(self) -> float:
        """The salience this faculty last computed, for a caller that must REPORT it rather than
        recompute it. Recomputing would advance the running baseline a second time and quietly
        make the reported number a different measurement from the one the ignition law used."""
        return float(self._last_salience)

    # distinct-from: brainstem/autonomous_loop.py -- _pfc_bias COMPUTES a prefrontal action
    #   preference; this returns a cached term and computes nothing. Read.
    # distinct-from: cortex/access_consciousness.py -- workspace exposes an identity handle;
    #   this reports a scalar the last observe() applied. Read.
    def last_top_down(self) -> float:
        """The top-down term observe() last APPLIED, for a caller that must REPORT it.

        `_top_down()` is the CONSUMING read: its post-condition writes the bus's listener-use
        term, which is correct exactly once per observed tick and nowhere else. A reporter
        calling it re-reads the bus and charges the listener term from the instrument -- the
        cockpit poll inflating the very consumption record US1 certified, and reporting "use"
        for sleep ticks the faculty never observed (US1 policy audit, finding B4; the same
        discipline last_salience() states for the baseline)."""
        return float(self._last_top_down)

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
    # ---- THE PRIOR MUST WASH OUT (US1 audit M0's criterion + its constructed red) ------
    # After settling on a noisy level01 body, the settled variance must be within an order
    # of magnitude of the input's own sample variance. The retired 1.0 seed FAILS this at
    # tick 100 (reconstructed below as the red); the derived 0.25 seed passes at tick 400.
    import random as _random

    _rng = _random.Random(11)
    _frames = [max(0.0, min(1.0, 0.3 + _rng.gauss(0.0, 0.05))) for _ in range(400)]
    _m = sum(_frames) / len(_frames)
    _sample_var = sum((x - _m) ** 2 for x in _frames) / len(_frames)
    _calib = InteroceptiveIntegration(channels=("stress",))
    for _x in _frames:
        _calib.observe(stress=_x)
    _settled = _calib.state()["channels"]["stress"]["var"]
    if not (_sample_var / 10.0 <= _settled <= _sample_var * 10.0):
        print(
            f"interoceptive_integration self-check FAILED: settled variance {_settled:.6f} is "
            f"not within an order of magnitude of the input's sample variance "
            f"{_sample_var:.6f} -- the prior is arbitrating, not washing out"
        )
        return 1
    _red = InteroceptiveIntegration(channels=("stress",))
    _red._var["stress"] = 1.0  # the retired prior, reconstructed as the reachable red
    for _x in _frames[:100]:
        _red.observe(stress=_x)
    _red_var = _red.state()["channels"]["stress"]["var"]
    if _red_var <= _sample_var * 10.0:
        print(
            f"interoceptive_integration self-check FAILED: the reconstructed 1.0 seed reads "
            f"{_red_var:.6f} at tick 100 (sample {_sample_var:.6f}) -- inside the criterion, "
            f"so the red is unreachable and the criterion measures nothing"
        )
        return 1
    # ---- THE BODY WIRE IS CHARGED, per edge, measured (specs/022 T026) -----------------
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    for edge, vary in ((RELAY_EDGE, "arousal"), (DRIVES_EDGE, "drives")):
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

    for edge, vary in ((RELAY_EDGE, "arousal"), (DRIVES_EDGE, "drives")):
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
