"""
NIR export — emit the product's trained spiking models as a Neuromorphic Intermediate Representation
graph (roadmap C1). JSON core; optional bridge to canonical NIR.

PROMOTED from research/ into the product (v0.81). NIR is the cross-platform IR that Loihi 2,
SpiNNaker 2, Speck and Xylo all import, so emitting ONE small NIR graph exports a model to every
neuromorphic backend at once. This wires it to the ACTUAL product recognizer: `export_classifier`
maps a trained `snn_classifier` weight matrix W[class][input] to a NIR graph
Input(N) -> Linear(W) -> LIF(C) -> Output(C) — the shape a chip needs to run the gesture/shape
classifier on-silicon. Node types + parameter names follow the NIR spec (Affine=(weight,bias),
Linear=(weight), LIF=(tau,r,v_leak,v_threshold), Delay=(delay)); `to_nir_graph()` lifts the JSON into
real `nir` dataclasses (needs nir+numpy) for `nir.write` -> canonical HDF5. Refs: NIR, Nat. Commun. 15
(2024) s41467-024-52259-9 / arXiv:2311.14641; SpiNNaker2 DVS-NIR arXiv:2504.06748.

  python -m snn_data_lake.substrate.nir_export        # (also run by run_selfchecks.py)
"""

import json

_PRIMS = {
    "Input": ("shape",),
    "Output": ("shape",),
    "Affine": ("weight", "bias"),
    "Linear": ("weight",),
    "LIF": ("tau", "r", "v_leak", "v_threshold"),
    "Delay": ("delay",),
    "Threshold": ("threshold",),
}


class NIRGraph:
    """A NIR computational graph: named primitive nodes + directed edges. Serializes to JSON."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add(self, name, prim, **params):
        if prim not in _PRIMS:
            raise ValueError(f"unknown NIR primitive {prim!r}; known: {sorted(_PRIMS)}")
        self.nodes[name] = {"type": prim, **params}
        return name

    def connect(self, src, dst):
        self.edges.append([src, dst])
        return self

    def to_json(self, indent=None):
        return json.dumps({"nodes": self.nodes, "edges": self.edges}, sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        g = cls()
        g.nodes = d["nodes"]
        g.edges = [list(e) for e in d["edges"]]
        return g


def validate(g):
    """Return a list of human-readable problems; empty list == a well-formed NIR graph."""
    issues = []
    for name, node in g.nodes.items():
        prim = node.get("type")
        if prim not in _PRIMS:
            issues.append(f"node {name!r}: unknown primitive {prim!r}")
            continue
        for key in _PRIMS[prim]:
            if key not in node:
                issues.append(f"node {name!r} ({prim}): missing parameter {key!r}")
    for src, dst in g.edges:
        if src not in g.nodes:
            issues.append(f"edge {src!r}->{dst!r}: source node missing")
        if dst not in g.nodes:
            issues.append(f"edge {src!r}->{dst!r}: target node missing")
    prims = [n.get("type") for n in g.nodes.values()]
    if "Input" not in prims:
        issues.append("graph has no Input node")
    if "Output" not in prims:
        issues.append("graph has no Output node")
    for name, node in g.nodes.items():
        if node.get("type") == "Affine" and "weight" in node and "bias" in node:
            if len(node["bias"]) != len(node["weight"]):
                issues.append(f"node {name!r} (Affine): bias len != weight rows")
        if node.get("type") == "LIF":
            lens = {k: len(node[k]) for k in _PRIMS["LIF"] if k in node}
            if len(set(lens.values())) > 1:
                issues.append(f"node {name!r} (LIF): parameter arrays differ in length {lens}")
    return issues


def export_classifier(weights, tau=0.02, v_threshold=1.0):
    """Map a trained snn_classifier weight matrix W[class][input] (C x N) to a NIR graph:
    Input(N) -> Linear(W) -> LIF(C) -> Output(C). This is the deploy shape for the gesture/shape
    recognizer — the spiking readout's weights are the Linear layer, one LIF neuron per class."""
    if not weights or not weights[0]:
        raise ValueError("export_classifier needs a non-empty C x N weight matrix")
    n_class = len(weights)
    n_in = len(weights[0])
    g = NIRGraph()
    g.add("input", "Input", shape=[n_in])
    g.add("readout", "Linear", weight=[list(row) for row in weights])
    g.add(
        "lif",
        "LIF",
        tau=[tau] * n_class,
        r=[1.0] * n_class,
        v_leak=[0.0] * n_class,
        v_threshold=[v_threshold] * n_class,
    )
    g.add("output", "Output", shape=[n_class])
    g.connect("input", "readout").connect("readout", "lif").connect("lif", "output")
    return g


def to_nir_graph(g):
    """Lift this JSON graph into real `nir` dataclasses (needs `nir` + numpy) so
    `nir.write(path, to_nir_graph(g))` -> canonical NIR HDF5 for Loihi 2 / SpiNNaker 2 / Speck / Xylo.
    Field names track the NIR spec; raises ImportError with an install hint if the stack is absent."""
    try:
        import nir
        import numpy as np
    except ImportError as e:
        raise ImportError("to_nir_graph needs the neuromorphic stack: pip install nir numpy") from e
    arr = lambda x: np.array(x, dtype=float)
    build = {
        "Input": lambda p: nir.Input(input_type={"input": arr(p["shape"])}),
        "Output": lambda p: nir.Output(output_type={"output": arr(p["shape"])}),
        "Affine": lambda p: nir.Affine(weight=arr(p["weight"]), bias=arr(p["bias"])),
        "Linear": lambda p: nir.Linear(weight=arr(p["weight"])),
        "LIF": lambda p: nir.LIF(
            tau=arr(p["tau"]),
            r=arr(p["r"]),
            v_leak=arr(p["v_leak"]),
            v_threshold=arr(p["v_threshold"]),
        ),
        "Delay": lambda p: nir.Delay(delay=arr(p["delay"])),
        "Threshold": lambda p: nir.Threshold(threshold=arr(p["threshold"])),
    }
    nodes = {
        name: build[node["type"]]({k: v for k, v in node.items() if k != "type"})
        for name, node in g.nodes.items()
    }
    return nir.NIRGraph(nodes=nodes, edges=[tuple(e) for e in g.edges])


def _demo_graph():
    """A small delay-SNN for the round-trip / validation checks."""
    g = NIRGraph()
    g.add("input", "Input", shape=[3])
    g.add("affine", "Affine", weight=[[0.5, -0.2, 0.1], [0.0, 0.3, 0.4]], bias=[0.1, -0.1])
    g.add("lif", "LIF", tau=[0.02, 0.02], r=[1.0, 1.0], v_leak=[0.0, 0.0], v_threshold=[1.0, 1.0])
    g.add("output", "Output", shape=[2])
    for a, b in [("input", "affine"), ("affine", "lif"), ("lif", "output")]:
        g.connect(a, b)
    return g


def main():
    import snn_data_lake.cortex.snn_classifier as C

    # export the ACTUAL trained gesture/shape recognizer to a NIR graph
    W, _ = C.train(C.dataset(15, C.templates()), 4, quiet=True)
    g = export_classifier(W)
    js = g.to_json(indent=2)

    print("=" * 62)
    print("NIR EXPORT -- trained spiking classifier -> Neuromorphic IR graph")
    print("=" * 62)
    print(
        f"classifier: {len(W)} classes x {len(W[0])} inputs -> NIR "
        f"Input({len(W[0])}) -> Linear -> LIF({len(W)}) -> Output({len(W)})"
    )
    print(
        f"NIR JSON: {len(js)} bytes; validate(): {len(validate(g))} issues; "
        f"portable to Loihi 2 / SpiNNaker 2 / Speck / Xylo (via nir.write)"
    )
    print("=" * 62)

    # ---- self-check: deterministic asserts ----
    assert validate(g) == [], f"exported classifier must be a valid NIR graph: {validate(g)}"
    assert g.nodes["readout"]["type"] == "Linear" and len(g.nodes["readout"]["weight"]) == 4, (
        "readout Linear must carry the C x N weight matrix"
    )
    assert len(g.nodes["readout"]["weight"][0]) == C.N, "Linear weight must have N inputs per class"
    assert g.nodes["lif"]["type"] == "LIF" and len(g.nodes["lif"]["tau"]) == 4, (
        "one LIF neuron per class"
    )
    rt = NIRGraph.from_json(g.to_json())
    assert rt.nodes == g.nodes and rt.edges == g.edges, "exported NIR must round-trip losslessly"

    # the emitter's own guarantees (validate catches corruption)
    assert validate(_demo_graph()) == [], "demo graph must validate"
    bad = _demo_graph()
    bad.edges.append(["lif", "ghost"])
    assert any("target node missing" in i for i in validate(bad)), (
        "validate must catch a dangling edge"
    )
    bad = _demo_graph()
    del bad.nodes["input"]
    assert any("no Input" in i for i in validate(bad)), "validate must catch a missing Input"
    print(
        f"self-check OK: trained classifier exported to a valid NIR graph "
        f"({len(W)}x{len(W[0])} -> Linear+LIF), round-trips; validate() catches corruption"
    )


if __name__ == "__main__":
    main()
