"""The REPORT CHANNEL (specs/022 T034): the declared route from the workspace composite to
every reporter -- what the brain can SAY about what it is broadcasting, as one typed
snapshot {content, availability, strength} instead of ad-hoc private reads.

WHY A CHANNEL AND NOT ANOTHER ACCESSOR. The chat's reply path already proved the consumer
half (spike_chat_lake._attended: adopts the arm's bus, constructs nothing, submits
nothing) -- but the read was PRIVATE: no declared port, so the manifest's anatomy said
the broadcast reached nobody while a consumer read it daily (T030 left `broadcast_out`
edge-less 'until T031/T034'; this is the T034 half).

THE CONSUMERS, each a real runtime read of snapshot() (the T034 review measured the
first draft's consumer claims to be prose, and the honest census is now code):
  * `Brain.state()['workspace_report']` -- the /api/state field the cockpit renders
    (P4c), matching the declared edge `report_channel.report_out -> brain.report_in`.
  * `LiveBrain.thought` -- the cockpit's thought row, routed through the channel (it was
    the review's named second reporter bypassing the route; now ONE reporter mechanism).
  * LiveBrain's attending-turn reply -- composed from the full record on the live chat
    path (content + hold, or the honest DARK sentence).
  * `spike_chat_lake._attended` -- the chat-side adapter. DISCLOSED SHAPE, measured by
    the review: its clause is real at LakeChat.converse() (426-byte vs 156-byte gap
    replies) and LiveBrain's bridge DISCARDS gap-turn replies, so this consumer's charge
    is chat-level, not production-level; the attachment (`lc.brain = self`, landed with
    this channel) is what makes even that chat-level read live.

THE RECORD. `snapshot(workspace)` returns {content, availability, strength}:
  content      -- what the workspace is broadcasting (the winning token; None when dark).
  availability -- 1.0 when something is globally available, 0.0 when the workspace is
                  dark. Deliberately binary at this port: access is all-or-none at the
                  bus (ignition), and a graded availability would re-invent the strength
                  field beside it.
  strength     -- broadcast_hold: how strongly the incumbent is still held, in [0,1] --
                  the workspace's own accessor, read from the same single snapshot.

`consciousness_probe` (RETIRED at T081, 2026-09-16) is NOT retired here: its retirement is DEFERRED to T081 (P2 -- both
replacement halves, this channel AND the report-order probe, must exist when the incumbent
dies; only the first half lands now).

DISCLOSURE (P8 sentience clause): this channel reports what the workspace MEASURES --
content, availability, hold. It is a report wire, not a marker of experience; no
phenomenal claim is made or implied.

# distinct-from: reasoning/spike_chat_lake.py -- the chat CONSUMER; its _attended now
#   delegates here (one mechanism, P2). distinct-from: cortex/access_consciousness.py --
#   CLAIMS/criteria over the bus; this is the typed transport of the report itself.
# distinct-from: cortex/live_brain.py -- owns the composite and the `thought` accessor;
#   this snapshots through the same public accessors, constructs nothing.
# distinct-from: reasoning/consciousness_probe.py (RETIRED at T081, 2026-09-16) -- the INCUMBENT this channel half-
#   replaces at T081; untouched here. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.report_channel
"""


def declare_ports(manifest):
    """Manifest registrant (specs/022 T034): the report channel's IN port receives the
    workspace broadcast (discharging T030's edge-less disposition on `broadcast_out`),
    and `report_out` carries the typed {content, availability, strength} snapshot to its
    DECLARED consumer -- `brain.report_in`, whose runtime read is
    `Brain.state()['workspace_report']` on the state the cockpit renders. The type row's
    (3,) f4 describes `as_vector`'s numeric projection (availability, strength,
    content-present -- built by code, asserted in the self-check); the content token
    itself is symbolic and rides the EVENT encoding, same discipline as the coalition
    ports' TYPE HONESTY note."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "report_channel",
        Port(
            name="report_in",
            region="substrate",
            shape=(1,),
            dtype="f4",
            unit="workspace_content",
            encoding="EVENT",
            kind="IN",
            declaring_module="report_channel",
            join_policy="arbitrate",
        ),
    )
    manifest.declare_port(
        "report_channel",
        Port(
            name="report_out",
            region="substrate",
            shape=(3,),
            dtype="f4",
            unit="report_snapshot",
            encoding="EVENT",
            kind="OUT",
            declaring_module="report_channel",
            delivery_policy="all-or-none",
        ),
    )
    # the channel declares its own subscription to the workspace broadcast
    manifest.declare_edge("global_workspace", "broadcast_out", "report_channel", "report_in")


def snapshot(workspace):
    """The report record over ONE workspace snapshot. Adopts the given (composite) bus,
    constructs nothing, submits nothing -- the same read discipline `_attended` proved.
    None workspace -> the dark record: three honest situations (no brain, workspace=False,
    nothing ignited) all report the same 'nothing globally available'."""
    if workspace is None:
        return {"content": None, "availability": 0.0, "strength": 0.0}
    content = workspace.broadcast()
    hold = float(workspace.broadcast_hold)
    return {
        "content": content,
        "availability": 1.0 if content is not None else 0.0,
        "strength": hold if content is not None else 0.0,
    }


# distinct-from: substrate/signal_ring.py -- ring transport records; this projects ONE
#   report dict to the manifest row's numeric vector. distinct-from: thalamus/
#   reentrant_workspace.py -- a registrant, no projection. Read.
def as_vector(rec):
    """The (3,) numeric projection the manifest row declares: [availability, strength,
    content_present]. The content TOKEN itself is symbolic and rides the EVENT encoding
    (the coalition ports' TYPE HONESTY discipline); this vector is what the f4 row
    DESCRIBES, built by code rather than asserted by prose -- the review measured the
    first draft declaring a payload nothing constructed."""
    return [
        float(rec["availability"]),
        float(rec["strength"]),
        1.0 if rec["content"] is not None else 0.0,
    ]


# distinct-from: thalamus/global_workspace.py -- the REAL bus; this stub only answers the
#   two accessor names so the record contract is testable without constructing a bus.
# distinct-from: substrate/episodic_recall.py -- phrase overlap only. Both read.
class _StubBus:
    """A minimal broadcast()/broadcast_hold stand-in for the arms below. A STUB, not a
    GlobalWorkspace, deliberately: the first draft's self-check constructed real buses
    here and registered this module as a FACULTY holding private buses in the p6 census
    (two ratchets red, measured). The LIVE arms -- real kernel, decaying hold, release,
    determinism -- live in global_workspace's own self-check (the T030 precedent: the
    bus module owns the arms that drive buses; its role carve-out is the honest home).
    What THIS module's arms own is the record contract itself."""

    def __init__(self, content, hold):
        self._c = content
        self._h = hold

    def broadcast(self):
        return self._c

    @property
    def broadcast_hold(self):
        return self._h


def main():
    print("=" * 74)
    print("REPORT CHANNEL -- typed workspace-report snapshot {content, availability, strength}")
    print("=" * 74)

    # (1) DARK record: the no-workspace and dark-bus situations collapse to one honest
    # shape, and a dark bus can NEVER leak a stale strength (hold gated on content).
    dark = snapshot(None)
    assert dark == {"content": None, "availability": 0.0, "strength": 0.0}
    assert snapshot(_StubBus(None, 0.73)) == dark, (
        "a dark bus with a residual hold must STILL report the dark record -- strength "
        "without content is a stale-report leak"
    )

    # (2) the record IS the bus's own pair -- and the RED is any reporter that transforms
    # it. A 1-minus-hold impostor, a clamped impostor, and a pinned impostor all differ
    # from the identity on some stub, and this arm sweeps enough holds to catch each.
    for hold in (0.05, 0.25, 0.5, 0.55, 0.9):
        rec = snapshot(_StubBus("scene", hold))
        assert rec == {"content": "scene", "availability": 1.0, "strength": hold}, (
            f"the record must be the bus's own (content, hold) IDENTITY at hold={hold}, "
            f"got {rec} -- any transform here is a reporter editing the measurement"
        )

    # (3) the (3,) projection the manifest row declares, built and checked, both branches.
    assert as_vector(snapshot(_StubBus("x", 0.4))) == [1.0, 0.4, 1.0]
    assert as_vector(dark) == [0.0, 0.0, 0.0]

    print(
        "\nself-check OK: dark-record honesty (incl. the stale-strength leak red); the "
        "record is the bus's own pair at every swept hold (transform impostors refused); "
        "the declared (3,) projection is built by code. LIVE-kernel arms (decay, release, "
        "determinism) run in global_workspace's own self-check -- the bus owns its "
        "drivers. Report wire only -- no phenomenal claim (P8)."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
