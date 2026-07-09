"""Mesoscale schematic connectome -- the white-matter WIRING layer of the brain architecture.

`brain_atlas.py` is the region -> module label map (the NODES). This module is the EDGES: a typed,
directed, signed, loop-aware inter-region projection graph, so the project is a queryable
ARCHITECTURE, not just a bag of region modules whose only connections are Python imports (which
carry no direction, sign, weight, or loop-closure).

Nodes are canonical brain sub-structures, each tagged with the project region it belongs to. Edges
are the major long-range pathways, each typed:
  * tract  : the anatomical pathway name (e.g. "corticospinal", "perforant path")
  * sign   : exc | inh | da | ne | 5ht | ach   (excitatory / inhibitory / neuromodulatory class)
  * cls    : relay | feedback | broadcast | descending
  * loop   : which canonical loop the edge participates in (or None)
The graph encodes the canonical loops -- thalamocortical, cortico-basal-ganglia-thalamo-cortical
(CBGTC), the hippocampal trisynaptic loop, cortico-cerebellar, Papez (limbic), the amygdala threat
pathway, ascending neuromodulatory broadcast, and the descending corticospinal path to the arm.

HONEST label: this is a MESOSCALE SCHEMATIC of canonical pathways (like the Allen mesoscale atlas),
NOT a synapse-level or measured connectome. It asserts graph-topological facts (loop closure, the
single gated actuator path), not biological connection strengths.

Safety tie-in: the ONLY path from any cortical node to the arm actuator runs through the brainstem
motor node's gated edge -- a deterministic self-check that reinforces the arm-safety envelope (no
region reaches the actuator except through the reflex/E-STOP-gated final common path).

  python -m snn_data_lake.connectome        # (also run by run_selfchecks.py)
"""
import json
import os

# node_id -> (project region or None for the effector, human-readable structure name)
NODES = {
    "ctx_sensory": ("cortex", "sensory cortex"),
    "ctx_motor": ("cortex", "motor cortex"),
    "ctx_pfc": ("cortex", "prefrontal / association cortex"),
    "thal_relay": ("thalamus", "thalamic relay nucleus"),
    "trn": ("thalamus", "thalamic reticular nucleus (TRN, inhibitory gate)"),
    "striatum": ("basal_ganglia", "striatum (D1 direct / D2 indirect)"),
    "gpe": ("basal_ganglia", "external pallidum (GPe)"),
    "gpi_snr": ("basal_ganglia", "internal pallidum / SNr (BG output)"),
    "stn": ("basal_ganglia", "subthalamic nucleus (STN)"),
    "vta_snc": ("basal_ganglia", "VTA / SNc (dopamine source)"),
    "ec": ("hippocampus", "entorhinal cortex"),
    "dg": ("hippocampus", "dentate gyrus"),
    "ca3": ("hippocampus", "CA3"),
    "ca1": ("hippocampus", "CA1"),
    "sub": ("hippocampus", "subiculum"),
    "bla": ("amygdala", "basolateral amygdala (BLA)"),
    "cea": ("amygdala", "central amygdala (CeA)"),
    "cb_ctx": ("cerebellum", "cerebellar cortex (Purkinje)"),
    "cb_deep": ("cerebellum", "deep cerebellar nuclei"),
    "hypothal": ("neuromodulation", "hypothalamus (mammillary body / HPA)"),
    "bf_ach": ("neuromodulation", "basal forebrain (acetylcholine)"),
    "bs_motor": ("brainstem", "brainstem / spinal motor + reflex (the gated final path)"),
    "actuator": (None, "robot arm actuator"),
}


def _e(source, target, tract, sign, cls, loop=None, weight=1.0):
    return {"source": source, "target": target, "tract": tract, "sign": sign,
            "cls": cls, "loop": loop, "weight": weight}


# the directed, typed edge list (canonical long-range projections)
EDGES = [
    # thalamocortical loop (+ the TRN inhibitory gate)
    _e("thal_relay", "ctx_sensory", "thalamocortical radiation", "exc", "relay", "thalamocortical"),
    _e("ctx_sensory", "thal_relay", "corticothalamic feedback", "exc", "feedback", "thalamocortical"),
    _e("ctx_sensory", "trn", "corticoreticular", "exc", "feedback", "thalamocortical"),
    _e("trn", "thal_relay", "reticulothalamic (gate)", "inh", "feedback", "thalamocortical"),
    # cortico-basal-ganglia-thalamo-cortical (CBGTC) selection loop
    _e("ctx_motor", "striatum", "corticostriatal", "exc", "relay", "cbgtc"),
    _e("striatum", "gpi_snr", "striatonigral (direct / Go)", "inh", "relay", "cbgtc"),
    _e("striatum", "gpe", "striatopallidal (indirect / NoGo)", "inh", "relay", "cbgtc"),
    _e("gpe", "stn", "pallidosubthalamic", "inh", "relay", "cbgtc"),
    _e("stn", "gpi_snr", "subthalamopallidal", "exc", "relay", "cbgtc"),
    _e("gpe", "gpi_snr", "intrapallidal", "inh", "relay", "cbgtc"),
    _e("gpi_snr", "thal_relay", "pallidothalamic (tonic inhibition)", "inh", "relay", "cbgtc"),
    _e("thal_relay", "ctx_motor", "thalamocortical (motor)", "exc", "relay", "cbgtc"),
    _e("vta_snc", "striatum", "nigrostriatal (dopamine)", "da", "broadcast", "cbgtc"),
    # hippocampal trisynaptic loop
    _e("ec", "dg", "perforant path", "exc", "relay", "hippocampal"),
    _e("dg", "ca3", "mossy fibers", "exc", "relay", "hippocampal"),
    _e("ca3", "ca1", "Schaffer collaterals", "exc", "relay", "hippocampal"),
    _e("ca1", "sub", "CA1-subiculum", "exc", "relay", "hippocampal"),
    _e("sub", "ec", "subiculum-entorhinal (return)", "exc", "feedback", "hippocampal"),
    # cortico-cerebellar loop
    _e("ctx_motor", "cb_ctx", "cortico-ponto-cerebellar", "exc", "relay", "cortico_cerebellar"),
    _e("cb_ctx", "cb_deep", "Purkinje projection", "inh", "relay", "cortico_cerebellar"),
    _e("cb_deep", "thal_relay", "dentatothalamic", "exc", "relay", "cortico_cerebellar"),
    # Papez / limbic circuit
    _e("sub", "hypothal", "fornix -> mammillary body", "exc", "relay", "papez"),
    _e("hypothal", "thal_relay", "mammillothalamic tract", "exc", "relay", "papez"),
    _e("thal_relay", "ctx_pfc", "thalamo-cingulate", "exc", "relay", "papez"),
    _e("ctx_pfc", "ec", "cingulum -> entorhinal", "exc", "relay", "papez"),
    # amygdala threat pathway
    _e("ctx_sensory", "bla", "sensory -> amygdala (CS)", "exc", "relay", "amygdala"),
    _e("bla", "cea", "intra-amygdala", "exc", "relay", "amygdala"),
    _e("cea", "hypothal", "amygdala -> HPA (stress)", "exc", "descending", "amygdala"),
    _e("cea", "bs_motor", "amygdala -> PAG/brainstem (freeze/defense)", "exc", "descending", "amygdala"),
    # ascending neuromodulatory broadcast
    _e("vta_snc", "ctx_pfc", "mesocortical dopamine", "da", "broadcast", None),
    _e("bf_ach", "ctx_sensory", "cholinergic broadcast", "ach", "broadcast", None),
    _e("bf_ach", "ec", "septohippocampal cholinergic", "ach", "broadcast", None),
    # descending corticospinal path to the effector -- THE gated final common path
    _e("ctx_motor", "bs_motor", "corticospinal tract", "exc", "descending", "corticospinal"),
    _e("bs_motor", "actuator", "final common motor path (reflex/E-STOP gated)", "exc", "descending", "corticospinal"),
]

# canonical loops as CLOSED node cycles (first repeated at the end)
LOOPS = {
    "thalamocortical": ["thal_relay", "ctx_sensory", "thal_relay"],
    "cbgtc": ["ctx_motor", "striatum", "gpi_snr", "thal_relay", "ctx_motor"],
    "hippocampal": ["ec", "dg", "ca3", "ca1", "sub", "ec"],
    "cortico_cerebellar": ["ctx_motor", "cb_ctx", "cb_deep", "thal_relay", "ctx_motor"],
    "papez": ["sub", "hypothal", "thal_relay", "ctx_pfc", "ec", "dg", "ca3", "ca1", "sub"],
}

ACTUATOR = "actuator"
GATE_NODE = "bs_motor"   # the brainstem/spinal gated node -- the sole gateway to the actuator


def _adjacency(block_node=None, block_edge=None):
    adj = {n: [] for n in NODES}
    for e in EDGES:
        if e["source"] == block_node or e["target"] == block_node:
            continue
        if block_edge is not None and (e["source"], e["target"]) == block_edge:
            continue
        adj[e["source"]].append(e["target"])
    return adj


def has_edge(source, target):
    return any(e["source"] == source and e["target"] == target for e in EDGES)


def reachable(source, target, block_node=None, block_edge=None):
    """BFS: can `target` be reached from `source` (optionally with a node/edge removed)?"""
    adj = _adjacency(block_node=block_node, block_edge=block_edge)
    seen, stack = set(), [source]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return False


def route(source, target):
    """Return a shortest node path source -> target over the connectome, or None."""
    adj = _adjacency()
    prev, queue, seen = {source: None}, [source], {source}
    while queue:
        n = queue.pop(0)
        if n == target:
            path = []
            while n is not None:
                path.append(n)
                n = prev[n]
            return list(reversed(path))
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                prev[m] = n
                queue.append(m)
    return None


def is_closed_loop(nodes):
    """True if every consecutive pair in `nodes` is a real directed edge (the loop closes)."""
    return all(has_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)) and nodes[0] == nodes[-1]


def cortical_nodes():
    return [n for n, (region, _) in NODES.items() if region == "cortex"]


def _import_graph_is_untyped():
    """FAIR CONTROL: the graphify EXTRACTED import graph carries no sign/loop/tract typing, so the
    connectome is not redundant with Python imports. Returns (checked, untyped). checked=False if the
    graph file is absent/unreadable (the anchor is then skipped, not failed)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "graphify-out", "graph.json")
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
    except (OSError, ValueError):
        return False, None
    edges = g.get("edges") or g.get("links") or []
    if not isinstance(edges, list) or not edges:
        return False, None
    typed_keys = {"sign", "loop", "tract"}
    untyped = not any(isinstance(e, dict) and (set(e) & typed_keys) for e in edges)
    return True, untyped


def main():
    n_edges, n_loops = len(EDGES), len(LOOPS)
    print("=" * 68)
    print("CONNECTOME -- mesoscale schematic wiring layer (canonical pathways)")
    print("=" * 68)
    print(f"nodes={len(NODES)} (across {len(set(r for r, _ in NODES.values() if r))} regions) "
          f"edges={n_edges} loops={n_loops}")
    for name, nodes in LOOPS.items():
        print(f"  loop {name:<18} closed={is_closed_loop(nodes)}  ({' -> '.join(nodes)})")
    cs = route("ctx_motor", ACTUATOR)
    print(f"corticospinal route ctx_motor -> actuator: {' -> '.join(cs) if cs else None}")
    print("=" * 68)

    # ---- self-check: graph-topological facts, zero-dep ----
    # every edge is well-typed and connects real nodes
    for e in EDGES:
        assert e["source"] in NODES and e["target"] in NODES, f"edge on unknown node: {e}"
        assert e["sign"] in ("exc", "inh", "da", "ne", "5ht", "ach"), f"bad sign: {e}"
        assert e["cls"] in ("relay", "feedback", "broadcast", "descending"), f"bad class: {e}"

    # (a) every canonical loop is a CLOSED directed cycle (un-riggable topology)
    for name, nodes in LOOPS.items():
        assert is_closed_loop(nodes), f"canonical loop {name} must be a closed cycle"

    # (b) SAFETY: the actuator is reachable ONLY through the gated brainstem node/edge.
    assert reachable("ctx_motor", ACTUATOR), "motor cortex must reach the actuator (via corticospinal)"
    for c in cortical_nodes():
        # removing the brainstem gate node disconnects EVERY cortical node from the actuator
        assert not reachable(c, ACTUATOR, block_node=GATE_NODE), \
            f"{c} must not reach the actuator except through the gated brainstem node"
    # the actuator's only in-edge is the gated final common path
    in_edges = [(e["source"], e["target"]) for e in EDGES if e["target"] == ACTUATOR]
    assert in_edges == [(GATE_NODE, ACTUATOR)], f"actuator must have exactly one (gated) in-edge, got {in_edges}"
    assert not reachable("ctx_motor", ACTUATOR, block_edge=(GATE_NODE, ACTUATOR)), \
        "removing the gated final motor edge must cut cortex off from the actuator"

    # (c) FAIR CONTROL: imports lack the connectome's typing (not redundant with Python imports)
    checked, untyped = _import_graph_is_untyped()
    typed_here = all(isinstance(e.get("sign"), str) and "cls" in e for e in EDGES)
    assert typed_here, "connectome edges must carry sign + class typing"
    if checked:
        assert untyped, "import graph unexpectedly carries sign/loop typing (control invalid)"

    print(f"self-check OK: {n_loops} canonical loops all closed cycles; the actuator is reachable "
          f"ONLY through the gated brainstem edge (safety); {n_edges} edges typed sign+class+loop"
          + (" > untyped import graph (fair control)" if checked else " (import-graph control skipped)"))


if __name__ == "__main__":
    main()
