"""TypedPort + WiringManifest -- the interface every faculty exposes to every other
(specs/022 T018, FR-012, contracts/manifest.md is the binding text).

WHAT THIS IS: the declared anatomy. A faculty exposes typed ports; edges are 4-tuples
(src_faculty, src_port, dst_faculty, dst_port), port-to-port; and the five checks
(validate_structure / infer_types / check_types / region_check / join_check) run at CI over
the committed artifact via `substrate/manifest_gate.py` (T019, same commit). The ported
`p6_gate` (T023, same commit) and the region charge gate (T022) read the SAME manifest --
one anatomy, several instruments.

THE POLICIES ARE NOT DEFAULTS, deliberately: an IN port MUST declare its join_policy
({sum, max, concat, typed-merge, arbitrate}) and an OUT port MUST declare its
delivery_policy ("all-or-none" or ("graded", association_source)) AT CONSTRUCTION. A default
join is how two producers silently sum into a consumer that never chose summation -- the
"silent-sum" defect T020's enforcement red is named after. REF ports are the flagged escape
hatch: legal, enumerated by `refs()`, audited, never the quiet path.

TWO HOMES, TWO MEANINGS (disclosed, not smoothed): the TRACKED REPO ARTIFACT
(tools/wiring_manifest.nir) is the source of truth the gates read -- it must change in the
same commit as any declaration change, which is what makes the CI checks meaningful. The
LAKE copy is the brain's own runtime provenance record ("this is the anatomy I actually
loaded"), written by the daemon at the kernel cutover (T033+), NEVER by a gate run -- a gate
that writes the live lake is the writer-vs-probe hazard. Until cutover the lake copy does
not exist, and saying so here is the honest state.

SERIALIZATION: NIR-HDF5 (the [nir] extra; nir 1.0.8 + h5py in the gate env). The NIR graph
is the INTEROPERABLE PROJECTION (nodes with shapes, edges); it cannot carry join/delivery
policies or units, so the full-fidelity manifest rides in the same HDF5 file as a JSON
attribute, and the round-trip criterion is on THAT (byte-exact), with the NIR projection
checked for node/edge count agreement. Where nir is absent the serialization arm SKIPS
LOUDLY; the five structure checks never depend on it.

DAY-ONE STATE, counted (T018): the committed manifest WAS EMPTY -- zero ports, zero edges,
stated in exactly those words by every reader rather than rendered as a green that looks
like coverage. The first real registrant pair landed at T025 (the serotonin PATIENCE wire:
raphe_serotonin.five_ht_out -> evidence_accumulation.patience_in); the committed census now
lives with the artifact and manifest_gate.REGISTRANTS, and main() reports it from there.
"""

# distinct-from: snn_data_lake/connectome.py -- region->region ANATOMICAL wiring (typed,
#   signed loops between brain regions); this is faculty PORT-level signal wiring with
#   policies, the machine-checked layer the connectome does not carry.
# distinct-from: substrate/region_wiring.py -- measures LIVE cross-module traffic; this
#   DECLARES intended wiring (p6 compares the two at T023+).
# distinct-from: substrate/nir_export.py -- exports the NEURON substrate to NIR for
#   neuromorphic backends; this serializes the PORT manifest (same container format on
#   purpose, different graph). All sites read.

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "tools" / "wiring_manifest.nir"
MANIFEST_VERSION = "wiring_manifest.v1"

ENCODINGS = frozenset({"VEC_DENSE", "VEC_SPARSE", "SCALAR_DENSE", "SCALAR_SPARSE", "EVENT"})
KINDS = frozenset({"IN", "OUT", "REF"})
JOIN_POLICIES = frozenset({"sum", "max", "concat", "typed-merge", "arbitrate"})


@dataclass(frozen=True)
class Port:
    """One typed signal surface. Construction REFUSES an under-specified port -- the checks
    below exist for what construction cannot see (cross-port facts), never as a second
    chance for what it can."""

    name: str
    region: str
    shape: tuple
    dtype: str
    unit: str
    encoding: str
    kind: str
    declaring_module: str
    precision_bits: int | None = None
    join_policy: str | None = None
    delivery_policy: tuple | str | None = None

    def __post_init__(self):
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"port {self.name!r}: encoding {self.encoding!r} not in {sorted(ENCODINGS)}"
            )
        if self.kind not in KINDS:
            raise ValueError(f"port {self.name!r}: kind {self.kind!r} not in {sorted(KINDS)}")
        if self.kind == "IN":
            if self.join_policy not in JOIN_POLICIES:
                raise ValueError(
                    f"IN port {self.name!r} MUST declare join_policy in {sorted(JOIN_POLICIES)} "
                    "(no default -- a defaulted join is the silent-sum defect)"
                )
            if self.delivery_policy is not None:
                raise ValueError(f"IN port {self.name!r} cannot carry a delivery_policy")
        elif self.kind == "OUT":
            ok = self.delivery_policy == "all-or-none" or (
                isinstance(self.delivery_policy, tuple)
                and len(self.delivery_policy) == 2
                and self.delivery_policy[0] == "graded"
                and isinstance(self.delivery_policy[1], str)
                and self.delivery_policy[1]
            )
            if not ok:
                raise ValueError(
                    f"OUT port {self.name!r} MUST declare delivery_policy 'all-or-none' or "
                    "('graded', association_source)"
                )
            if self.join_policy is not None:
                raise ValueError(f"OUT port {self.name!r} cannot carry a join_policy")
        if not isinstance(self.shape, tuple) or not all(
            isinstance(d, int) and d > 0 for d in self.shape
        ):
            raise ValueError(f"port {self.name!r}: shape must be a tuple of positive ints")

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "region": self.region,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "unit": self.unit,
            "encoding": self.encoding,
            "kind": self.kind,
            "declaring_module": self.declaring_module,
            "precision_bits": self.precision_bits,
            "join_policy": self.join_policy,
            "delivery_policy": list(self.delivery_policy)
            if isinstance(self.delivery_policy, tuple)
            else self.delivery_policy,
        }
        return d


@dataclass
class WiringManifest:
    """The declared anatomy: ports keyed (faculty, port_name) + 4-tuple edges."""

    ports: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    # ---- declaration ---------------------------------------------------------------------
    def declare_port(self, faculty: str, port: Port):
        key = (str(faculty), port.name)
        if key in self.ports:
            raise ValueError(f"duplicate port declaration {key}")
        self.ports[key] = port

    def declare_edge(self, src_f: str, src_p: str, dst_f: str, dst_p: str):
        self.edges.append((str(src_f), str(src_p), str(dst_f), str(dst_p)))

    def refs(self) -> list:
        """The audited escape-hatch census: every REF port, enumerated."""
        return sorted(k for k, p in self.ports.items() if p.kind == "REF")

    # ---- the five checks (contracts/manifest.md, CI via manifest_gate) --------------------
    def validate_structure(self) -> list[str]:
        """Endpoints exist, no duplicate edges, directions sane. RED findings as strings."""
        out = []
        seen = set()
        for e in self.edges:
            if e in seen:
                out.append(f"duplicate edge {e}")
            seen.add(e)
            src, dst = (e[0], e[1]), (e[2], e[3])
            for end, which in ((src, "src"), (dst, "dst")):
                if end not in self.ports:
                    out.append(f"edge {e}: {which} endpoint {end} is not a declared port")
            sp, dp = self.ports.get(src), self.ports.get(dst)
            if sp is not None and sp.kind == "IN":
                out.append(f"edge {e}: source {src} is an IN port (edges flow OUT/REF -> IN)")
            if dp is not None and dp.kind == "OUT":
                out.append(f"edge {e}: destination {dst} is an OUT port")
        return out

    def infer_types(self) -> dict:
        """Per-IN-port inferred incoming shape, propagated from producers under the declared
        join. concat concatenates axis 0; every other policy requires identical shapes (a
        disagreement is left for check_types to report -- inference never repairs)."""
        incoming: dict = {}
        for src_f, src_p, dst_f, dst_p in self.edges:
            sp = self.ports.get((src_f, src_p))
            if sp is not None:
                incoming.setdefault((dst_f, dst_p), []).append(sp)
        inferred = {}
        for key, producers in incoming.items():
            dst = self.ports.get(key)
            if dst is None or dst.kind != "IN":
                continue
            shapes = [p.shape for p in producers]
            if dst.join_policy == "concat" and all(s[1:] == shapes[0][1:] for s in shapes):
                inferred[key] = (sum(s[0] for s in shapes),) + shapes[0][1:]
            elif all(s == shapes[0] for s in shapes):
                inferred[key] = shapes[0]
            else:
                inferred[key] = None  # producers disagree; check_types reds it
        return inferred

    def check_types(self) -> list[str]:
        """Producer/consumer mismatch = RED: shape (under the join), dtype, unit."""
        out = []
        inferred = self.infer_types()
        for src_f, src_p, dst_f, dst_p in self.edges:
            sp, dp = self.ports.get((src_f, src_p)), self.ports.get((dst_f, dst_p))
            if sp is None or dp is None or dp.kind != "IN":
                continue  # structural problems are validate_structure's findings
            if sp.dtype != dp.dtype:
                out.append(
                    f"edge {(src_f, src_p, dst_f, dst_p)}: dtype {sp.dtype!r} -> {dp.dtype!r}"
                )
            if sp.unit != dp.unit:
                out.append(f"edge {(src_f, src_p, dst_f, dst_p)}: unit {sp.unit!r} -> {dp.unit!r}")
        for key, shape in inferred.items():
            dst = self.ports[key]
            if shape is None:
                out.append(
                    f"IN port {key}: producers disagree on shape under join {dst.join_policy!r}"
                )
            elif shape != dst.shape:
                out.append(f"IN port {key}: inferred incoming {shape} != declared {dst.shape}")
        return out

    def region_check(self) -> list[str]:
        """port.region MUST equal brain_atlas.region_of(declaring_module) -- the manifest
        cannot invent anatomy the atlas does not hold."""
        from snn_data_lake.brain_atlas import region_of

        out = []
        for key, p in self.ports.items():
            actual = region_of(p.declaring_module)
            if actual != p.region:
                out.append(
                    f"port {key}: declares region {p.region!r} but "
                    f"brain_atlas.region_of({p.declaring_module!r}) = {actual!r}"
                )
        return out

    def join_check(self) -> list[str]:
        """>1 edge into an IN port whose join_policy is missing/invalid = RED. Construction
        forbids that state, so the reachable path is DESERIALIZATION -- from_dict re-checks
        because a loaded artifact is exactly where an undeclared join can enter."""
        out = []
        fan_in: dict = {}
        for e in self.edges:
            fan_in[(e[2], e[3])] = fan_in.get((e[2], e[3]), 0) + 1
        for key, n in fan_in.items():
            p = self.ports.get(key)
            if p is not None and p.kind == "IN" and n > 1 and p.join_policy not in JOIN_POLICIES:
                out.append(f"IN port {key}: {n} producers and NO declared join_policy (silent sum)")
        return out

    def warnings(self) -> list[str]:
        """Non-red findings: declared IN ports no edge references (the contract's warning)."""
        referenced = {(e[2], e[3]) for e in self.edges}
        return [
            f"IN port {k} is declared but unreferenced by any edge"
            for k, p in sorted(self.ports.items())
            if p.kind == "IN" and k not in referenced
        ]

    def all_findings(self) -> list[str]:
        return (
            self.validate_structure() + self.check_types() + self.region_check() + self.join_check()
        )

    # ---- serialization -------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "ports": [[list(k), p.to_dict()] for k, p in sorted(self.ports.items())],
            "edges": [list(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = True) -> WiringManifest:
        """Rebuild from a dict. strict=True re-validates every port through the Port
        constructor (refuses an under-specified artifact); strict=False admits raw entries
        so the GATE can hold a defective manifest up to the checks -- that is how the
        constructed reds are built, and it is the only sanctioned use."""
        if d.get("version") != MANIFEST_VERSION:
            raise ValueError(f"manifest version {d.get('version')!r} is not {MANIFEST_VERSION!r}")
        m = cls()
        for key, pd in d.get("ports", []):
            dp = dict(pd)
            if isinstance(dp.get("delivery_policy"), list):
                dp["delivery_policy"] = tuple(dp["delivery_policy"])
            dp["shape"] = tuple(dp.get("shape", ()))
            if strict:
                port = Port(**dp)
            else:
                port = object.__new__(Port)
                for f_name, val in dp.items():
                    object.__setattr__(port, f_name, val)
            m.ports[(key[0], key[1])] = port
        for e in d.get("edges", []):
            m.edges.append(tuple(e))
        return m

    def to_hdf5_bytes(self) -> bytes:
        """The NIR-HDF5 artifact: the interoperable NIR projection + the full-fidelity JSON
        as a file attribute (the round-trip criterion is on the JSON, byte-exact)."""
        import h5py
        import nir
        import numpy as np

        nodes, nir_edges = {}, []
        for (fac, name), p in sorted(self.ports.items()):
            nid = f"{fac}.{name}"
            arr = np.array(p.shape, dtype=np.int64)
            nodes[nid] = (
                nir.Output(output_type={"output": arr})
                if p.kind == "OUT"
                else nir.Input(input_type={"input": arr})
            )
        for src_f, src_p, dst_f, dst_p in self.edges:
            nir_edges.append((f"{src_f}.{src_p}", f"{dst_f}.{dst_p}"))
        graph = nir.NIRGraph(nodes=nodes, edges=nir_edges)
        # nir.write() takes a PATH and opens the file itself (measured: handing it an h5py
        # File raises TypeError in filename_encode) -- so write to a temp file, append the
        # full-fidelity JSON attr, and return the bytes.
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".nir")
        os.close(fd)
        try:
            nir.write(tmp, graph)
            with h5py.File(tmp, "a") as f:
                f.attrs["sndl_manifest_json"] = json.dumps(self.to_dict(), sort_keys=True)
            return Path(tmp).read_bytes()
        finally:
            os.unlink(tmp)

    @classmethod
    def from_hdf5_bytes(cls, blob: bytes, *, strict: bool = True) -> WiringManifest:
        import h5py

        with h5py.File(io.BytesIO(blob), "r") as f:
            payload = json.loads(f.attrs["sndl_manifest_json"])
        return cls.from_dict(payload, strict=strict)

    def commit_artifact(self, path: Path = ARTIFACT) -> Path:
        path.write_bytes(self.to_hdf5_bytes())
        return path

    @classmethod
    def load_artifact(cls, path: Path = ARTIFACT, *, strict: bool = True) -> WiringManifest:
        return cls.from_hdf5_bytes(Path(path).read_bytes(), strict=strict)


# The module-global "declare at faculty import" registry that once sat here is RETIRED (P2,
# T025 review): it had zero callers, and collecting declarations by import side effect is
# the exact mechanism manifest_gate's curated REGISTRANTS list exists to forbid (the pygenn
# incident, T016). Exactly one declaration path survives: manifest_gate.REGISTRANTS.


def _nir_available() -> bool:
    try:
        import nir  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    # Constructed manifest: every mechanism exercised, every red constructed, on scratch.
    m = WiringManifest()
    m.declare_port(
        "salience",
        Port(
            name="salience_out",
            region="neuromodulation",
            shape=(8,),
            dtype="f4",
            unit="rate_hz",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="raphe_serotonin",
            delivery_policy="all-or-none",
        ),
    )
    m.declare_port(
        "workspace",
        Port(
            name="drive_in",
            region="thalamus",
            shape=(8,),
            dtype="f4",
            unit="rate_hz",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="global_workspace",
            join_policy="sum",
        ),
    )
    m.declare_edge("salience", "salience_out", "workspace", "drive_in")
    findings = m.validate_structure() + m.check_types() + m.join_check()
    if findings:
        print(f"wiring_manifest self-check FAILED: clean manifest produced findings {findings}")
        return 1
    if m.region_check() != []:
        # the two modules above ARE in those regions in brain_atlas; a finding means the
        # atlas moved or the check broke -- either way loud
        print(
            f"wiring_manifest self-check FAILED: region check on true regions: {m.region_check()}"
        )
        return 1
    # constructed reds, one per check
    bad = WiringManifest()
    bad.ports = dict(m.ports)
    bad.edges = list(m.edges) + [("ghost", "nope", "workspace", "drive_in"), m.edges[0]]
    if len(bad.validate_structure()) < 2:
        print(
            "wiring_manifest self-check FAILED: missing-endpoint/duplicate-edge reds did not fire"
        )
        return 1
    d = m.to_dict()
    d["ports"][1][1]["join_policy"] = None  # strip the join -- the silent-sum construction
    d["edges"].append(["salience", "salience_out", "workspace", "drive_in"])
    loaded = WiringManifest.from_dict(d, strict=False)
    if not loaded.join_check():
        print("wiring_manifest self-check FAILED: the undeclared-join red did not fire")
        return 1
    d2 = m.to_dict()
    d2["ports"][0][1]["region"] = "cerebellum"  # a region the module does not belong to
    if not WiringManifest.from_dict(d2, strict=False).region_check():
        print("wiring_manifest self-check FAILED: the region-mismatch red did not fire")
        return 1
    d3 = m.to_dict()
    d3["ports"][0][1]["unit"] = "volts"
    if not WiringManifest.from_dict(d3, strict=False).check_types():
        print("wiring_manifest self-check FAILED: the type-mismatch red did not fire")
        return 1
    # serialization round-trip (skip-loudly without nir; the checks above never needed it)
    if _nir_available():
        blob = m.to_hdf5_bytes()
        back = WiringManifest.from_hdf5_bytes(blob)
        if back.to_dict() != m.to_dict():
            print("wiring_manifest self-check FAILED: HDF5 round-trip diverged")
            return 1
        nir_note = "NIR-HDF5 round-trip byte-exact on the JSON payload"
    else:
        nir_note = "NIR SKIPPED LOUDLY (nir not importable here; structure checks unaffected)"
    # The declared-anatomy census, reported from the ONE declaration path (REGISTRANTS).
    from snn_data_lake.substrate.manifest_gate import rebuild_from_registrants

    live = rebuild_from_registrants()
    n_ports, n_edges = len(live.ports), len(live.edges)
    state = (
        "EMPTY (nothing declares)"
        if n_ports == 0
        else f"{n_ports} ports, {n_edges} edges (first registrant pair: T025 serotonin patience)"
    )
    print(
        f"wiring_manifest self-check OK: five checks green on the constructed manifest; "
        f"undeclared-join, region-mismatch, type-mismatch, missing-endpoint and duplicate-edge "
        f"reds all fired; {nir_note}. DECLARED ANATOMY: {state}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
