"""
NIR export — emit the repo's spiking models as a Neuromorphic Intermediate Representation
graph (research scan #10, roadmap C1). Pure-stdlib JSON core; optional bridge to canonical NIR.

The Paradigm B/C models here are hand-built spiking graphs: LIF neurons, learned synaptic
delays, and linear/affine weight maps. NIR (Neuromorphic Intermediate Representation) is the
cross-platform IR that Loihi 2, SpiNNaker 2, Speck and Xylo all import — so emitting ONE small
NIR graph exports a model to every neuromorphic backend at once, instead of writing a bespoke
per-chip translator. Refs: NIR, Nat. Commun. 15 (2024) s41467-024-52259-9 / arXiv:2311.14641;
github/neuromorphs/NIR. (Same move as energy.py: adopt the standard's structure, stay zero-dep.)

This module is the ZERO-DEP half — build / emit / parse / validate a NIR graph as JSON. Node
types and parameter names follow the NIR primitives so the JSON is a faithful rendering of the
IR, not an invented format:

    Input/Output -> shape       Affine -> weight, bias      Linear -> weight
    LIF -> tau, r, v_leak, v_threshold      Delay -> delay      Threshold -> threshold

`to_nir_graph()` is the optional bridge: if the `nir` package (+numpy) is installed it lifts
this JSON into real `nir` dataclasses, so `nir.write(path, to_nir_graph(g))` produces the
canonical HDF5 the chips load. The bridge needs numpy/nir; the JSON core needs nothing.

  Run: python nir_export.py        # (also run by run_selfchecks.py)
"""
import json

# NIR primitive -> the parameter keys it carries (subset covering feed-forward + delay SNNs).
# Keys follow the NIR spec so to_nir_graph() maps 1:1 onto nir.<Type>(**params).
_PRIMS = {
    "Input":     ("shape",),
    "Output":    ("shape",),
    "Affine":    ("weight", "bias"),                       # y = W x + b
    "Linear":    ("weight",),                              # y = W x
    "LIF":       ("tau", "r", "v_leak", "v_threshold"),   # leaky integrate-and-fire
    "Delay":     ("delay",),                              # per-unit synaptic delay
    "Threshold": ("threshold",),
}


class NIRGraph:
    """A NIR computational graph: named primitive nodes + directed edges. Serializes to JSON."""

    def __init__(self):
        self.nodes = {}          # name -> {"type": prim, **params}
        self.edges = []          # [[src_name, dst_name], ...]

    def add(self, name, prim, **params):
        if prim not in _PRIMS:
            raise ValueError(f"unknown NIR primitive {prim!r}; known: {sorted(_PRIMS)}")
        self.nodes[name] = {"type": prim, **params}
        return name

    def connect(self, src, dst):
        self.edges.append([src, dst])
        return self

    def to_json(self, indent=None):
        # sort_keys -> deterministic bytes (the repo's serialize-then-parse-back invariant)
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
            rows = len(node["weight"])
            if len(node["bias"]) != rows:
                issues.append(f"node {name!r} (Affine): bias len {len(node['bias'])} != weight rows {rows}")
        if node.get("type") == "LIF":
            lens = {k: len(node[k]) for k in _PRIMS["LIF"] if k in node}
            if len(set(lens.values())) > 1:
                issues.append(f"node {name!r} (LIF): parameter arrays differ in length {lens}")
    return issues


def feedforward(weights, biases, lif):
    """Build Input -> (Affine -> LIF)* -> Output from per-layer weights/biases + shared LIF params.

    weights[k] is an (out x in) matrix, biases[k] a length-out vector; `lif` is a dict of the four
    LIF parameter scalars, broadcast to each layer's width. This is how a trained spiking classifier
    (snn_classifier's weight rows + neuron constants) exports to NIR. Returns a NIRGraph.
    """
    g = NIRGraph()
    g.add("input", "Input", shape=[len(weights[0][0])])
    prev = "input"
    for k, (w, b) in enumerate(zip(weights, biases)):
        width = len(w)
        a = g.add(f"affine{k}", "Affine", weight=w, bias=b)
        n = g.add(f"lif{k}", "LIF", **{p: [lif[p]] * width for p in _PRIMS["LIF"]})
        g.connect(prev, a).connect(a, n)
        prev = n
    g.add("output", "Output", shape=[len(weights[-1])])
    g.connect(prev, "output")
    return g


def to_nir_graph(g):
    """Lift this JSON graph into real `nir` dataclasses (needs `nir` + numpy). Enables
    `nir.write(path, to_nir_graph(g))` -> canonical NIR HDF5 loadable on Loihi 2 / SpiNNaker 2 /
    Speck / Xylo. Field names track the NIR spec; adjust the builders if your nir version differs.
    Raises ImportError with an install hint when the optional stack is absent (JSON core stays zero-dep).
    """
    try:
        import numpy as np
        import nir
    except ImportError as e:
        raise ImportError("to_nir_graph needs the neuromorphic stack: pip install nir numpy") from e
    arr = lambda x: np.array(x, dtype=float)
    build = {
        "Input":     lambda p: nir.Input(input_type={"input": arr(p["shape"])}),
        "Output":    lambda p: nir.Output(output_type={"output": arr(p["shape"])}),
        "Affine":    lambda p: nir.Affine(weight=arr(p["weight"]), bias=arr(p["bias"])),
        "Linear":    lambda p: nir.Linear(weight=arr(p["weight"])),
        "LIF":       lambda p: nir.LIF(tau=arr(p["tau"]), r=arr(p["r"]),
                                       v_leak=arr(p["v_leak"]), v_threshold=arr(p["v_threshold"])),
        "Delay":     lambda p: nir.Delay(delay=arr(p["delay"])),
        "Threshold": lambda p: nir.Threshold(threshold=arr(p["threshold"])),
    }
    nodes = {name: build[node["type"]]({k: v for k, v in node.items() if k != "type"})
             for name, node in g.nodes.items()}
    return nir.NIRGraph(nodes=nodes, edges=[tuple(e) for e in g.edges])


def _demo_graph():
    """A small delay-SNN — Input -> Affine -> LIF -> Delay -> LIF -> Output — the Paradigm-B shape:
    weighted projection, spike, a learned synaptic delay, spike again, readout."""
    g = NIRGraph()
    g.add("input", "Input", shape=[3])
    g.add("affine", "Affine", weight=[[0.5, -0.2, 0.1], [0.0, 0.3, 0.4]], bias=[0.1, -0.1])
    g.add("lif1", "LIF", tau=[0.02, 0.02], r=[1.0, 1.0], v_leak=[0.0, 0.0], v_threshold=[1.0, 1.0])
    g.add("delay", "Delay", delay=[0.003, 0.001])
    g.add("lif2", "LIF", tau=[0.02, 0.02], r=[1.0, 1.0], v_leak=[0.0, 0.0], v_threshold=[1.0, 1.0])
    g.add("output", "Output", shape=[2])
    for a, b in [("input", "affine"), ("affine", "lif1"), ("lif1", "delay"),
                 ("delay", "lif2"), ("lif2", "output")]:
        g.connect(a, b)
    return g


def main():
    g = _demo_graph()
    js = g.to_json(indent=2)
    types = ", ".join(sorted({n["type"] for n in g.nodes.values()}))
    print("=" * 62)
    print("NIR EXPORT — spiking model -> Neuromorphic Intermediate Representation")
    print("=" * 62)
    print(f"graph: {len(g.nodes)} nodes, {len(g.edges)} edges; primitives: {types}")
    print(f"JSON:  {len(js)} bytes -> Loihi 2 / SpiNNaker 2 / Speck / Xylo (via nir.write)")
    print(f"validate(): {len(validate(g))} issues")
    print("=" * 62)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert validate(g) == [], f"demo graph must be a valid NIR graph: {validate(g)}"
    rt = NIRGraph.from_json(g.to_json())
    assert rt.nodes == g.nodes and rt.edges == g.edges, "JSON round-trip must be lossless"

    ff = feedforward(weights=[[[0.1, 0.2], [0.3, 0.4]]], biases=[[0.0, 0.0]],
                     lif={"tau": 0.02, "r": 1.0, "v_leak": 0.0, "v_threshold": 1.0})
    assert validate(ff) == [], f"feedforward graph must validate: {validate(ff)}"

    # validate() must CATCH corruption, else it is theatre
    bad = _demo_graph(); bad.edges.append(["lif2", "ghost"])
    assert any("target node missing" in i for i in validate(bad)), "must catch a dangling edge"
    bad = _demo_graph(); bad.nodes["affine"]["bias"] = [0.1]
    assert any("bias len" in i for i in validate(bad)), "must catch an Affine bias/weight mismatch"
    bad = _demo_graph(); del bad.nodes["input"]
    assert any("no Input" in i for i in validate(bad)), "must catch a missing Input node"

    print("self-check OK: demo + feedforward validate; JSON round-trip lossless; "
          "validate() catches dangling edges / bad dims / missing I/O")


if __name__ == "__main__":
    main()
