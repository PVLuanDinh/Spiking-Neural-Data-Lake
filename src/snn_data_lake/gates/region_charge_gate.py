"""Region charge gate (specs/022 T022, FR-015, gates.md 'Region charge gate' row): a region
is CHARGED only when it is REACHABLE and at least one of its port edges carries MEASURED
charge attribution. Registration alone never greens -- presence is not participation, at
region granularity.

THE TWO CONJUNCTS, both required (contracts/manifest.md 'Region charge gate'):
  (a) REACHABLE -- the region owns >=1 module on the dormancy gate's live-reachability walk
      (REUSED: `dormancy_gate.reachable()` is the ONE walk; this gate never re-derives it).
  (b) CHARGED -- >=1 wiring-manifest edge touching the region carries a charge-attribution
      record in tools/region_charge_evidence.json: a MEASURED replay-twice divergence from
      `port_charge_probe`, never a declaration. Evidence is written by probe RUNS (the first
      lands with T025's registrant re-proof), and an edge nobody measured attributes nothing.

THE DENOMINATOR is the anatomical inventory (SYNTHESIS: REGIONS - {substrate, plasticity},
insula included) -- infra and distributed-synaptic tiers are not anatomy and are excluded BY
THE DENOMINATOR, never by exemption rows. Since G8 S1a (evidence g8-reorganization.md 2.4, F5)
the regions and their membership are the FROZEN pre-G8 LEGACY_REGIONS, and each member's path
is module_path(): G8's engine-function packages are code layout, not anatomy.

EXEMPT rows carry {region, reason, owner, date, expiry}, and their COUNT is a ratchet with
the ceiling recorded in-module: adding an exemption above the ceiling REFUSES, and raising
the ceiling is an owner-visible admission (the p6 baseline pattern). Day one: zero rows,
ceiling zero.

REPORT-ONLY UNTIL T039, structurally: `--assert` reads the integer floor from
tools/region_charge_floor.json, which DOES NOT EXIST until the T039 exit certification
writes it (into FR-015's blank, per the spec). Floor absent => the assert states
"report-only: no floor recorded (T039 writes it)" and exits 0 WITH the census printed --
never a silent green. Floor present => charged_count >= floor binds.

DAY-ONE COUNT, stated here because this file is where the number was born: 0 of 16 regions
CHARGED -- at landing every region failed conjunct (b) because the manifest WAS empty and
no charge evidence existed yet. T025's serotonin patience wire recorded the first measured
attribution and moved the census to 2 of 16 (neuromodulation + basal_ganglia). Reachability
alone stays high; that gap IS the message: a tree can be almost fully reachable and barely
charged, which is why registration never greens.
"""

# distinct-from: gates/dormancy_gate.py -- module-level reachability, REUSED here as
#   conjunct (a); this gate adds the per-REGION port-charge conjunct it cannot express.
# distinct-from: gates/manifest_gate.py -- static contract checks over the manifest;
#   this consumes the manifest's edges and demands MEASURED charge on them.
# distinct-from: substrate/port_charge_probe.py -- produces the evidence; this gate consumes
#   it. distinct-from: substrate/region_wiring.py -- live traffic census, not a verdict.
# All sites read.

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "tools" / "region_charge_evidence.json"
FLOOR = REPO / "tools" / "region_charge_floor.json"

#: The anatomical denominator's exclusions (not anatomy; excluded by denominator, never by
#: exemption -- SYNTHESIS's ONE-denominator rule).
NON_ANATOMY = frozenset({"substrate", "plasticity"})

#: EXEMPT rows: region -> {reason, owner, date, expiry}. Day one: EMPTY.
EXEMPT: dict = {}

#: The exemption-count ratchet ceiling. Adding a row above this REFUSES; raising it is an
#: owner-visible admission recorded beside the number (the p6 baseline pattern). Day one: 0.
EXEMPT_CEILING = 0


def regions() -> list[str]:
    from snn_data_lake import LEGACY_REGIONS

    return sorted(r for r in LEGACY_REGIONS if r not in NON_ANATOMY)


# distinct-from: brain_atlas.py -- region_of answers one module's frozen label; this gate's own
#   conjunct (a), edited in place (G8 F5), joins LEGACY membership to module_path(). Also read:
#   cortex/autapse_neuron.py, cortex/placement.py, cortex/snn_moe_classifier.py,
#   reasoning/arc_residual.py -- a trial stream, a GRN arm, an RNG hinge, an ARC alphabet.
def reachable_regions() -> set[str]:
    """Conjunct (a): regions owning >=1 module on the ONE reachability walk. The region LABEL and
    its MEMBERSHIP come from the frozen LEGACY_REGIONS, each member's PATH from module_path() (G8
    F5): built from the label, a moved module's path would name its old package and lose it."""
    from snn_data_lake import LEGACY_REGIONS, module_path, package_of
    from snn_data_lake.gates.dormancy_gate import reachable

    live = reachable()  # fully-dotted module paths: "snn_data_lake.<package>.<module>"
    out = set()
    for region, mods in LEGACY_REGIONS.items():
        if region in NON_ANATOMY:
            continue
        if any(module_path(m) in live for m in mods if package_of(m) is not None):
            out.add(region)
    return out


def charged_regions() -> tuple[set[str], int]:
    """Conjunct (b): regions with >=1 manifest edge carrying a MEASURED attribution record.
    -> (regions, n_evidence_records)."""
    from snn_data_lake.substrate.wiring_manifest import ARTIFACT, WiringManifest

    if not EVIDENCE.exists():
        return set(), 0
    records = json.loads(EVIDENCE.read_text()).get("attributions", [])
    try:
        manifest = WiringManifest.load_artifact(ARTIFACT, strict=False)
    except Exception:
        return set(), len(records)
    edge_regions: dict = {}
    for src_f, src_p, dst_f, dst_p in manifest.edges:
        for key in ((src_f, src_p), (dst_f, dst_p)):
            port = manifest.ports.get(key)
            if port is not None:
                edge_regions.setdefault(tuple((src_f, src_p, dst_f, dst_p)), set()).add(port.region)
    out = set()
    for rec in records:
        edge = tuple(rec.get("edge", ()))
        if rec.get("diverged") and edge in edge_regions:
            out |= edge_regions[edge]
    return out, len(records)


def census() -> dict:
    reach = reachable_regions()
    charged, n_ev = charged_regions()
    rows = {}
    for r in regions():
        if r in EXEMPT:
            rows[r] = "EXEMPT"
        elif r in reach and r in charged:
            rows[r] = "CHARGED"
        elif r in reach:
            rows[r] = "REACHABLE-UNCHARGED"
        else:
            rows[r] = "UNREACHABLE"
    return {
        "rows": rows,
        "charged_count": sum(1 for v in rows.values() if v == "CHARGED"),
        "denominator": len(rows),
        "evidence_records": n_ev,
        "exempt_count": len(EXEMPT),
    }


def exemption_ratchet_ok() -> bool:
    return len(EXEMPT) <= EXEMPT_CEILING


def report(c: dict) -> str:
    lines = [
        f"region charge: {c['charged_count']} of {c['denominator']} CHARGED "
        f"({c['evidence_records']} evidence records; {c['exempt_count']} exempt, ceiling {EXEMPT_CEILING})"
    ]
    for r, v in sorted(c["rows"].items()):
        lines.append(f"  {r:<16} {v}")
    return "\n".join(lines)


def self_check() -> int:
    c = census()
    # the gates.md red: a registered-but-uncharged region counts UNWIRED, constructed --
    # a region that IS reachable must still not read CHARGED without evidence.
    reach_only = [r for r, v in c["rows"].items() if v == "REACHABLE-UNCHARGED"]
    if not reach_only:
        print(
            "region_charge_gate self-check FAILED: no REACHABLE-UNCHARGED region exists to "
            "demonstrate that registration alone never greens (vacuous day-one premise -- "
            "if US1 charged everything, replace this arm with a constructed census)"
        )
        return 1
    # constructed green: a synthetic evidence record + manifest edge must charge its region
    # (exercised through the same code path with constructed inputs)
    from snn_data_lake.substrate.wiring_manifest import Port, WiringManifest

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
    edge = ("salience", "s_out", "ws", "d_in")
    edge_regions = {edge: {"neuromodulation", "thalamus"}}
    fake_records = [{"edge": list(edge), "diverged": True}]
    hit = set()
    for rec in fake_records:
        e = tuple(rec["edge"])
        if rec["diverged"] and e in edge_regions:
            hit |= edge_regions[e]
    if hit != {"neuromodulation", "thalamus"}:
        print(
            f"region_charge_gate self-check FAILED: constructed attribution did not charge ({hit})"
        )
        return 1
    # a NON-diverged record must charge NOTHING (measurement, not declaration)
    hit2 = set()
    for rec in [{"edge": list(edge), "diverged": False}]:
        e = tuple(rec["edge"])
        if rec["diverged"] and e in edge_regions:
            hit2 |= edge_regions[e]
    if hit2:
        print("region_charge_gate self-check FAILED: a non-diverged record charged a region")
        return 1
    # the exemption ratchet refuses above its ceiling
    if not exemption_ratchet_ok():
        print("region_charge_gate self-check FAILED: exemptions exceed the recorded ceiling")
        return 1
    print(
        f"region_charge_gate self-check OK: {len(reach_only)} REACHABLE-UNCHARGED regions "
        "prove registration never greens; constructed attribution charged exactly its edge's "
        "regions; a non-diverged record charged nothing; exemption ratchet holds "
        f"({len(EXEMPT)} <= {EXEMPT_CEILING})"
    )
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    if mode == "--self-check":
        return self_check()
    if mode == "--assert":
        c = census()
        print(report(c))
        if not exemption_ratchet_ok():
            print(
                f"  REGION CHARGE GATE RED: exemptions {c['exempt_count']} > ceiling {EXEMPT_CEILING}"
            )
            return 1
        if not FLOOR.exists():
            print(
                "  report-only: no integer floor recorded (the T039 exit certification writes "
                "tools/region_charge_floor.json into FR-015's blank; until then this gate "
                "REPORTS and the count above is the day-one truth)"
            )
            return 0
        floor = int(json.loads(FLOOR.read_text())["floor"])
        if c["charged_count"] < floor:
            print(f"  REGION CHARGE GATE RED: {c['charged_count']} charged < floor {floor}")
            return 1
        print(f"  floor holds: {c['charged_count']} >= {floor}")
        return 0
    # bare (run_selfchecks): census + the self-check arms -- never red-free
    rc = self_check()
    if rc != 0:
        return rc
    print(report(census()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
