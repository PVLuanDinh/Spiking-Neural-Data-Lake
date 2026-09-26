r"""Atlas-node gate (specs/022 T024, FR-032, gates.md 'Atlas-node gate' row): every
`connectome.NODES` entry must resolve to >=1 module IN ITS REGION whose source references
the node id -- or carry a dated EXEMPT row. Day-one phantoms are RECORDED, never hidden.

WHY: a connectome node nobody's code names is anatomy on paper only -- the schematic can
grow structures the tree never implements, and the diff between "declared" and "named by a
module that computes near it" is exactly the drift this gate freezes. The reference test is
deliberately modest: the node id appears as a standalone token, CASE-INSENSITIVE, in >=1
source file of the node's own region. Case-insensitive because the id is the lowercase
canonical form of an anatomical name ("ca3" IS "CA3"); a case-sensitive first draft read
25 phantoms where regions genuinely name the structure in prose case (BLA, CeA). The token
pattern is `(?<![\w.])id(?!\w)` -- the dot-guard exists because a plain `\b` boundary
matches `md` inside "README.md" (proven by this gate's own self-check control before the
guard existed).

SHORT IDS (<= 4 chars, no underscore) get a stricter rule, and every clause of it is
pinned to a FALSE RESOLUTION the landing census actually produced: `sub` "resolved" on
"sub-pixel" / "sub-eps" / a local `sub = self.W[...]`, and `ec` on the local variable
`ec = EpisodeConsolidator(...)` -- zero anatomy in either. A short abbreviation therefore
counts only when the match is (a) the QUOTED exact id ('md' / "md" -- connectome
vocabulary, how connectome_router genuinely participates), or (b) prose carrying at least
one UPPERCASE letter (CeA, BLA, SC, LHb -- how anatomy is actually written), never a bare
all-lowercase token (that is English or a variable). Long ids and underscore compounds
keep the plain token match: `striatum` in lowercase prose is unambiguous anatomy, and
compounds (`gpi_snr`, `vta_snc`) are the connectome's OWN vocabulary that resolve only
when a module says the id itself -- prose "GPi/SNr" deliberately does not count, so a
compound node retires its row by participating, not by being mentioned.

Modest is honest here -- a stronger "uses it in computation" claim is the region CHARGE
gate's job (T022), not this one's; this gate polices NAMING, the necessary condition.

Effector nodes (region None: `actuator`, `viscera`) are excluded by the denominator -- they
are deliberately outside every region so that reaching them requires the gated brainstem
edge (the safety property connectome's own self-check asserts).

DAY-ONE PHANTOMS, counted at landing and recorded in EXEMPT with dates rather than hidden:
the census at introduction found the phantoms the contract predicted (cb_ctx, pn) plus the
rest listed in the EXEMPT table below -- each a node declared before any module in its
region said its name. Every row carries the task that retires it or the reason it stays.
"""

# distinct-from: gates/region_charge_gate.py -- CHARGE at region granularity (measured
#   attribution); this polices node-id NAMING, the necessary condition below it.
# distinct-from: snn_data_lake/connectome.py -- declares the nodes; this asks whether the
#   tree knows them. distinct-from: gates/manifest_gate.py -- port contract checks.
# All sites read.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "snn_data_lake"

#: The day-one phantom census, FROZEN at landing (2026-08-20). Regenerating this tuple is
#: FORBIDDEN -- it is the recorded baseline; a phantom retired gets DELETED from it (the
#: ratchet: the set may shrink, never grow -- a NEW node must land WITH its referencing
#: module, the way T021a's insula/pulvinar/md did).
_PHANTOMS_AT_LANDING: tuple = (
    "bf_ach",
    "bs_motor",
    "ca1",
    "ca3",
    # "cb_ctx" + "cb_deep" RETIRED 2026-08-31 (specs/022 T054-i): cerebellum/microcircuit.py
    # computes near both and says their ids -- the retirement condition, met (the ratchet:
    # rows only ever leave this tuple)
    "ctx_motor",
    "dg",
    "ec",
    "gpe",
    "gpi_snr",
    "hypothal",
    "io",
    "nacc",
    "nts_rvlm",
    "ob",
    "pbc",
    "pir",
    # "pn" RETIRED 2026-08-31 (T054-i): pons/mossy_relay.py is the region's first
    # computing module and says its id
    "prf",
    "sc_deep",
    "sub",
    "vta_snc",
)

#: node_id -> {reason, date, retires} -- the recorded phantoms. A row here is a DEBT with a
#: named retirement, never coverage.
EXEMPT: dict = {
    node: {
        "reason": "declared before any module in its region referenced it (day-one phantom, "
        "recorded not hidden; gates.md predicted cb_ctx/pn and the census found the full set)",
        "date": "2026-08-20",
        "retires": "the module that first computes near this structure says its id",
    }
    for node in _PHANTOMS_AT_LANDING
}


def _token(node: str) -> re.Pattern:
    """The reference pattern: standalone token, case-insensitive, dot-guarded (see docstring)."""
    return re.compile(rf"(?<![\w.]){re.escape(node)}(?!\w)", re.IGNORECASE)


def references(node: str, text: str) -> bool:
    """Does `text` name this node? The docstring's short-id rule, in code."""
    if len(node) > 4 or "_" in node:
        return bool(_token(node).search(text))
    if re.search(rf"['\"]{re.escape(node)}['\"]", text):
        return True  # quoted exact id: connectome vocabulary (how 'md' participates)
    return any(
        not m.group(0).islower() for m in _token(node).finditer(text)
    )  # prose with an uppercase letter (CeA); bare lowercase is English or a variable


# distinct-from: brain_atlas.py -- _region_files lists a package's files for the atlas's
#   filesystem parity; this gate's own census, edited in place (G8 F5), reads each member's TEXT for
#   node ids. Also read: cortex/interaction_learning.py (g0_campaign), cortex/arealization.py
#   (emit_regions), cortex/h01_ingest.py (build_arrays), cortex/autapse_neuron.py -- none a census.
def node_references() -> dict:
    """node_id -> [module files in its region referencing it]. The census. The region's LABEL and
    MEMBERSHIP come from the frozen LEGACY_REGIONS, each member's FILE from module_path() (G8 S1a,
    F5): `SRC / region` stops holding a module once G8 moves it into an engine-function package."""
    from snn_data_lake import LEGACY_REGIONS, module_path, package_of
    from snn_data_lake.connectome import NODES

    out = {}
    for node, (region, _label) in NODES.items():
        if region is None:
            continue  # effectors: outside every region by design (the safety property)
        hits = []
        for mod in LEGACY_REGIONS.get(region, ()):
            if package_of(mod) is None:
                continue
            f = SRC.joinpath(*module_path(mod).split(".")[1:]).with_suffix(".py")
            if f.exists() and references(node, f.read_text(encoding="utf-8", errors="replace")):
                hits.append(f"{region}/{mod}")
        out[node] = hits
    return out


def census() -> dict:
    refs = node_references()
    phantoms = sorted(n for n, hits in refs.items() if not hits and n not in EXEMPT)
    exempt_live = sorted(n for n in EXEMPT if n in refs and not refs[n])
    exempt_stale = sorted(n for n in EXEMPT if n not in refs or refs[n])
    return {
        "n_nodes": len(refs),
        "resolved": sorted(n for n, hits in refs.items() if hits),
        "phantoms": phantoms,
        "exempt_live": exempt_live,
        "exempt_stale": exempt_stale,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    c = census()
    header = (
        f"atlas-node gate: {len(c['resolved'])} of {c['n_nodes']} nodes resolved; "
        f"{len(c['phantoms'])} NEW phantoms; {len(c['exempt_live'])} recorded (dated) phantoms; "
        f"{len(c['exempt_stale'])} stale exemptions"
    )
    if mode == "--self-check":
        # red 1 (constructed phantom): a fake node with no possible reference must be caught
        # by the same token machinery the census uses
        pattern = _token("zz_constructed_phantom_zz")
        anywhere = any(
            pattern.search(p.read_text(encoding="utf-8", errors="replace"))
            for p in (SRC / "thalamus").glob("*.py")
        )
        if anywhere:
            print("atlas_node_gate self-check FAILED: the constructed phantom was 'referenced'")
            return 1
        # control (sensitivity): a node known-referenced must be found -- insula names itself
        refs = node_references()
        if not refs.get("insula"):
            print("atlas_node_gate self-check FAILED: the reference detector is blind (insula)")
            return 1
        # dot-guard control: 'md' must NOT match the .md file extension (a plain \b does --
        # that is why the guard exists), and MUST match the prose abbreviation and quoted id
        probes = [
            ("md", "see README.md and mdata", False),  # the dot-guard's reason to exist
            ("md", "the mediodorsal (MD) relay", True),  # uppercase prose resolves
            ("md", "NUCLEI = ('pulvinar', 'md')", True),  # quoted id resolves (participation)
            ("sub", "sub-pixel refined by local centroid", False),  # landing false-positive 1
            ("ec", "ec = EpisodeConsolidator(dim=dim)", False),  # landing false-positive 2
            ("cea", "SUPPRESSES the CeA readout", True),  # genuine short-id prose resolves
            ("striatum", "dorsal striatum spiny units", True),  # long id: lowercase prose OK
        ]
        for node, text, want in probes:
            if references(node, text) != want:
                print(
                    f"atlas_node_gate self-check FAILED: reference rule wrong on "
                    f"({node!r}, {text!r}) -- expected {want}"
                )
                return 1
        # disclosed hazard pin: a literal 'import io' would false-resolve node io -- assert
        # no cerebellum module carries one, so the hazard stays theoretical on this tree
        from snn_data_lake import REGIONS

        io_hazard = [
            m
            for m in REGIONS.get("cerebellum", ())
            if (SRC / "cerebellum" / f"{m}.py").exists()
            and re.search(
                r"^\s*import io\b",
                (SRC / "cerebellum" / f"{m}.py").read_text(encoding="utf-8", errors="replace"),
                re.MULTILINE,
            )
        ]
        if io_hazard:
            print(f"atlas_node_gate self-check FAILED: import-io hazard is live ({io_hazard})")
            return 1
        print(
            f"atlas_node_gate self-check OK: constructed phantom caught; insula reference "
            f"detected ({refs['insula']}); all {len(probes)} reference-rule probes hold "
            f"(dot-guard, quoted-id, uppercase-prose, both landing false-positives); "
            f"import-io hazard absent. {header}"
        )
        return 0
    print(header)
    for n in c["phantoms"]:
        print(f"  NEW PHANTOM: {n} -- a node must land WITH its referencing module (FR-032)")
    for n in c["exempt_stale"]:
        print(f"  STALE EXEMPT: {n} -- now referenced (or gone); DELETE its row (the ratchet)")
    if mode == "--assert":
        return 1 if (c["phantoms"] or c["exempt_stale"]) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
