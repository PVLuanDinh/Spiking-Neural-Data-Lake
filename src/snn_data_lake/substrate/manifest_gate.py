"""Manifest gate (specs/022 T019, FR-012, gates.md 'Manifest type-check' row): the five
contract checks over the COMMITTED wiring-manifest artifact, plus the divergence red --
the committed copy must equal what the registrants would rebuild.

WHAT RUNS AT CI (--assert): load tools/wiring_manifest.nir with strict=False -- a defective
artifact must REACH the checks, not die in a constructor where its findings would be a
stack trace instead of a verdict -- then validate_structure / check_types / region_check /
join_check (infer_types feeds check_types), then rebuild the manifest from the REGISTRANTS
census and require dict-equality with the committed copy. Any finding = exit 1. An EMPTY
manifest prints exactly that -- "EMPTY: 0 ports, 0 edges" -- because an empty anatomy
rendered as a plain green is coverage-shaped silence (the day-one state; first registrant
lands at T025).

WHY REGISTRANTS IS A CURATED LIST AND NOT AN IMPORT SCAN: collecting declarations by
importing every module is how the GIL got silently re-enabled mid-suite (the pygenn
incident, T016) -- imports have side effects, so the gate imports EXACTLY the modules that
declare ports, listed in wiring_manifest.REGISTRANTS, and nothing else. A registrant that
fails to import is a RED, never a skip.

THE REDS (contract 'Reds this contract must be able to fire', proven in --self-check and in
tests/test_manifest_gate.py): (1) undeclared-join -- two producers into a policy-stripped IN
port; (2) region-mismatch -- a port claiming anatomy its module does not hold; (4)
manifest/gate divergence -- the committed artifact differs from the registrant rebuild.
Red (3), neuter-insensitivity, is charge measurement and belongs to the charge probe
(T022) -- a static gate asserting liveness would be presence-as-participation.
"""

# distinct-from: substrate/wiring_manifest.py -- the MECHANISM this gate runs; the gate owns
#   the CI verdict and the divergence criterion, the manifest owns the checks.
# distinct-from: substrate/policy_manifest.py -- POLICIES.md rows vs ci.yml census, a
#   different manifest entirely. distinct-from: substrate/import_manifest.py -- lazy-import
#   reachability of ruff-excluded tiers. All sites read.

from __future__ import annotations

import sys

from snn_data_lake.substrate.wiring_manifest import (
    ARTIFACT,
    WiringManifest,
)

#: (module, attr) pairs that declare ports -- the ONLY imports the gate performs.
#: First entries: specs/022 T025, the serotonin patience wire (producer, then consumer --
#: the consumer's declare_ports also declares THE edge, its subscription).
REGISTRANTS: tuple = (
    ("snn_data_lake.neuromodulation.raphe_serotonin", "declare_ports"),
    ("snn_data_lake.basal_ganglia.evidence_accumulation", "declare_ports"),
    # T026: the interoceptive channel -- two producer halves, then the insula terminus
    # (whose declare_ports carries both edges and the interoception_out channel)
    ("snn_data_lake.medulla.interoceptive_arousal", "declare_ports"),
    ("snn_data_lake.hypothalamus.homeostatic_drives", "declare_ports"),
    # T039j: the THALAMIC RELAY hop. Ordered here because declare_edge validates endpoints:
    # the relay's edge needs `arousal_out` to exist, and the insula's edge needs `relay_out`.
    ("snn_data_lake.thalamus.interoceptive_relay", "declare_ports"),
    ("snn_data_lake.insula.interoceptive_integration", "declare_ports"),
    # T039j: the ignition law is the DECLARED consumer of interoception_out -- the port that had
    # carried an edge-less disposition since T026. Ordered after the insula, whose OUT port its
    # edge needs.
    ("snn_data_lake.substrate.region_coherence", "declare_ports"),
    # T027: the glia wire -- the astrocyte gain field as a diffuse channel (self-edge)
    ("snn_data_lake.neuromodulation.astrocyte_modulation", "declare_ports"),
    # T028: THE predictor port -- percept surprise into the brain's undirected-salience
    # competition (the one S0-chargeable consumer; the other three are dated forward
    # dispositions in SYNTHESIS section 6, discharged at T046a/T066/T076)
    ("snn_data_lake.cortex.percept_surprise", "declare_ports"),
    ("snn_data_lake.cortex.brain", "declare_ports"),
    # T046a: the |LP| third factor -- the SECOND of percept_surprise's dated forward
    # dispositions discharged (the consumer declares the edge surprise_out -> surprise_in);
    # lp_out was declared EDGE-LESS with T047's growth stopping criterion as the dated
    # consumer (the T030 broadcast_out precedent) -- DISCHARGED at T047, the row below.
    # Ordered after percept_surprise, whose OUT port the edge needs.
    ("snn_data_lake.basal_ganglia.learning_progress", "declare_ports"),
    # T047: the growth stopping criterion -- lp_out's dated consumer lands, discharging
    # T046a's edge-less disposition (the consumer declares the edge lp_out -> lp_in).
    # Ordered after learning_progress, whose OUT port the edge needs.
    ("snn_data_lake.cortex.growth_stop", "declare_ports"),
    # T029: the RPE third-factor port -- phasic DA into the R-STDP eligibility gate
    ("snn_data_lake.midbrain.dopamine_nuclei", "declare_ports"),
    ("snn_data_lake.basal_ganglia.r_stdp", "declare_ports"),
    # T030: the workspace composite -- the arm's bus producer, the shared bus, and the
    # re-entry stage (brain's reroute_in consumer edge rides in brain.declare_ports)
    ("snn_data_lake.brainstem.autonomous_loop", "declare_ports"),
    ("snn_data_lake.thalamus.global_workspace", "declare_ports"),
    ("snn_data_lake.thalamus.reentrant_workspace", "declare_ports"),
    # T034: the report channel -- the workspace broadcast's declared consumer (discharges
    # T030's edge-less disposition on broadcast_out) + the typed report_out snapshot
    # consumed by brain.report_in (Brain.state()'s workspace_report, the cockpit field)
    ("snn_data_lake.substrate.report_channel", "declare_ports"),
    # T036: the afferent boundary -- the S0 proprioceptive lane's producer (brain's
    # proprio_in consumer edge rides in brain.declare_ports); interoception's row is the
    # T026 tract, vision/audition are S1 dispositions, chemosensation out-of-scope
    ("snn_data_lake.substrate.afferent_boundary", "declare_ports"),
    # T037: the efferent lane -- the arm's action token observed by the efference-copy
    # boundary (autonomous_loop declares action_out; the boundary declares its
    # subscription and feeds the cerebellar forward model)
    ("snn_data_lake.substrate.efferent_boundary", "declare_ports"),
)


def rebuild_from_registrants() -> WiringManifest:
    """The manifest the tree WOULD produce: import exactly the registrant modules, let each
    declare into a fresh registry, return it. A failing registrant raises -- loudly."""
    fresh = WiringManifest()
    for mod_name, attr in REGISTRANTS:
        import importlib

        mod = importlib.import_module(mod_name)
        getattr(mod, attr)(fresh)
    return fresh


def run(artifact=ARTIFACT) -> tuple[bool, str]:
    """-> (ok, report). The CI body: five checks + divergence, census stated."""
    try:
        committed = WiringManifest.load_artifact(artifact, strict=False)
    except Exception as exc:
        return False, f"manifest artifact unreadable: {exc}"
    findings = committed.all_findings()
    rebuilt = rebuild_from_registrants()
    if rebuilt.to_dict() != committed.to_dict():
        findings.append(
            f"DIVERGENCE: committed artifact ({len(committed.ports)} ports, "
            f"{len(committed.edges)} edges) != registrant rebuild ({len(rebuilt.ports)} ports, "
            f"{len(rebuilt.edges)} edges) -- regenerate the artifact in this commit"
        )
    n_p, n_e = len(committed.ports), len(committed.edges)
    census = (
        "EMPTY: 0 ports, 0 edges (nothing declares yet; first registrant = T025)"
        if n_p == 0 and n_e == 0
        else f"{n_p} ports, {n_e} edges, {len(committed.refs())} REF escape hatches"
    )
    warns = committed.warnings()
    report = f"manifest {census}; findings={len(findings)}; warnings={len(warns)}"
    if findings:
        report += "\n  " + "\n  ".join(findings)
    # warning TEXTS, not just a count (T046a panel): a warning is a declared honest state
    # (an edge-less OUT port is the T030 precedent) -- a count that never names its items
    # is a report nobody can act on.
    if warns:
        report += "\n  " + "\n  ".join(f"warn: {w}" for w in warns)
    return not findings, report


def self_check() -> int:
    """Constructed reds 1, 2 and 4 fired through the GATE's own verdict path, with the clean
    committed artifact as the in-command positive control."""
    ok_clean, clean_report = run()
    if not ok_clean:
        print(f"manifest_gate self-check FAILED its own positive control:\n{clean_report}")
        return 1
    # red 4: divergence -- a committed artifact the registrants would not rebuild
    import os
    import tempfile
    from pathlib import Path

    from snn_data_lake.substrate.wiring_manifest import Port

    diverged = WiringManifest()
    diverged.declare_port(
        "ghost",
        Port(
            name="phantom_out",
            region="thalamus",
            shape=(1,),
            dtype="f4",
            unit="rate_hz",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="global_workspace",
            delivery_policy="all-or-none",
        ),
    )
    fd, tmp_name = tempfile.mkstemp(suffix=".nir")
    os.close(fd)  # Windows: an open descriptor locks the file against its own unlink
    tmp = Path(tmp_name)
    try:
        diverged.commit_artifact(tmp)
        ok_div, div_report = run(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if ok_div or "DIVERGENCE" not in div_report:
        print("manifest_gate self-check FAILED: the divergence red did not fire")
        return 1
    # reds 1 + 2 through the artifact path: strip a join, fake a region, RE-LOAD, run checks
    m = WiringManifest()
    m.declare_port(
        "salience",
        Port(
            name="s_out",
            region="neuromodulation",
            shape=(4,),
            dtype="f4",
            unit="rate_hz",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="raphe_serotonin",
            delivery_policy="all-or-none",
        ),
    )
    m.declare_port(
        "ws",
        Port(
            name="d_in",
            region="thalamus",
            shape=(4,),
            dtype="f4",
            unit="rate_hz",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="global_workspace",
            join_policy="sum",
        ),
    )
    m.declare_edge("salience", "s_out", "ws", "d_in")
    m.declare_edge("salience", "s_out", "ws", "d_in")  # second producer edge for the join red
    d = m.to_dict()
    d["ports"][1][1]["join_policy"] = None
    d["ports"][0][1]["region"] = "cerebellum"
    defective = WiringManifest.from_dict(d, strict=False)
    join_red = bool(defective.join_check())
    region_red = bool(defective.region_check())
    if not join_red or not region_red:
        print(
            f"manifest_gate self-check FAILED: undeclared-join fired={join_red}, "
            f"region-mismatch fired={region_red}"
        )
        return 1
    print(
        "manifest_gate self-check OK: positive control green on the committed artifact; "
        "divergence, undeclared-join and region-mismatch reds all fired "
        "(neuter-insensitivity is T022's charge probe, on purpose)"
    )
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    if mode == "--self-check":
        return self_check()
    if mode == "--assert":
        ok, report = run()
        print(f"manifest_gate [assert] {report}")
        return 0 if ok else 1
    # bare (run_selfchecks): the assert body plus the divergence red -- cheap, never red-free
    ok, report = run()
    if not ok:
        print(f"manifest_gate wiring-check FAILED:\n{report}")
        return 1
    rc = self_check()
    if rc != 0:
        return rc
    print(f"manifest_gate wiring-check OK: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
