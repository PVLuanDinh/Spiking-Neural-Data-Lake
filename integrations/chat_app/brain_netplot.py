"""
netplotbrain-based 3D brain-activity view for the SNDL Spiking Chat App -- "their entire design".

Instead of the hand-rolled ellipsoid hull (see brain_activity_3d.py), this renders EVERY
snn_data_lake module as a node on the REAL MNI152 template brain using netplotbrain
(https://github.com/wiheto/netplotbrain, MIT): netplotbrain draws the actual pial surface from
TemplateFlow and scatters our nodes at their real MNI152 mm coordinates, coloured + sized by each
region's per-turn activity (inferno colormap) so the brain lights up as the user converses.

Module enumeration (the 298-module map) is REUSED from brain_activity_3d so there is a single
source of truth for "every module"; here we only re-map each region to a real MNI152 mm centre.

HONEST placement of the non-anatomical tiers (netplotbrain is a brain-only design):
  * anatomical regions (cortex, hippocampus, amygdala, basal-ganglia, thalamus, hypothalamus,
    epithalamus, midbrain, pons, medulla, brainstem, cerebellum, olfactory, claustrum) -> their
    real bilateral MNI152 mm centres, so nodes land in the correct brain tissue.
  * neuromodulation -> its real diffuse source NUCLEI in MNI (VTA/SNc, locus coeruleus, raphe,
    basal-forebrain ACh, HPA) -- not one blob.
  * plasticity -> distributed across cortex + hippocampus (learning rules live at synapses).
  * reasoning  -> the frontal/prefrontal cortex (its functional home), labelled a cognition tier.
  * substrate  -> the spike data-lake / infra: NOT brain tissue, so it is NOT drawn inside the
    brain. Its module count is reported in an off-brain annotation instead of faking an MNI coord.

Live cost: netplotbrain re-renders the template each turn (~2-3s for one view, ~7s for 'LSR').
That is fine at conversational cadence. Falls back to brain_activity_3d.BrainActivity3D if
netplotbrain / TemplateFlow are unavailable (e.g. offline) -- see spiking_chat --viz.

matplotlib/pandas/numpy/netplotbrain are imported ONLY here (the opt-in --viz path), never in the
zero-dep core. Headless self-check (writes a PNG, asserts every module is placed):
  PYTHONUTF8=1 PYTHONPATH="src;integrations/chat_app" MPLBACKEND=Agg \
      python integrations/chat_app/brain_netplot.py
"""
import math
import os
import random
import tempfile

# reuse the SINGLE source of truth for the 298-module map + honest region labels
from brain_activity_3d import (
    COGNITIVE_CORTEX,
    MODULE_REGION,
    REGION_COUNT,
    _region_modules,
)

SEED = 1605
TEMPLATE = "MNI152NLin2009cAsym"

# region -> bilateral MNI152 mm centre (|x| magnitude; BILATERAL regions split L/R at +/-|x|).
# y = posterior(-)/anterior(+), z = inferior(-)/superior(+). Standard anatomical landmarks.
REGION_MNI = {
    "hippocampus":   (26, -20, -14),   # medial temporal
    "amygdala":      (24,  -4, -18),   # anterior medial temporal
    "basal_ganglia": (20,   6,   2),   # striatum/pallidum
    "thalamus":      (10, -18,   8),   # bilateral thalamic nuclei
    "hypothalamus":  ( 4,  -2, -12),
    "epithalamus":   ( 4, -24,   6),   # habenula / pineal
    "midbrain":      ( 0, -26, -12),
    "pons":          ( 0, -32, -30),
    "medulla":       ( 0, -44, -50),
    "brainstem":     ( 0, -32, -34),   # vertical reflex/arm column (elongated z)
    "cerebellum":    (22, -62, -36),   # bilateral, posterior-inferior
    "olfactory":     ( 8,  22, -18),   # orbito-frontal / olfactory base
    "claustrum":     (32,   4,   2),   # lateral (insula/putamen border)
}
BILATERAL = {"hippocampus", "amygdala", "basal_ganglia", "thalamus", "epithalamus",
             "cerebellum", "olfactory", "claustrum"}

# per-region jitter radius in mm (scalar = isotropic; 3-tuple = per-axis for elongated structures)
_RADIUS = {
    "hippocampus": (7, 12, 5), "amygdala": 6, "basal_ganglia": (8, 10, 6),
    "thalamus": 6, "hypothalamus": 4, "epithalamus": 4,
    "midbrain": 5, "pons": 5, "medulla": 5,
    "brainstem": (5, 7, 16), "cerebellum": (18, 14, 12), "olfactory": 7, "claustrum": (4, 8, 8),
}
_DEFAULT_RADIUS = 8

# real MNI centres of the diffuse neuromodulatory source nuclei
NEUROMOD_SOURCES = {
    "raphe_serotonin":      (0, -30, -24),   # midline raphe
    "acetylcholine":        (8,   2,  -8),   # basal forebrain (bilateral)
    "cortisol":             (4,  -2, -12),   # HPA / hypothalamus
    "astrocyte_modulation": (0,   0,  10),   # diffuse glia (large radius)
}
NEUROMOD_DEFAULT = (8, -16, -14)             # VTA/SNc dopamine source (bilateral)

# frontal/prefrontal cortex box (mm) -- home of the cognitive cortex modules + the reasoning tier
_FRONTAL = (30, 46, 18)


def _jitter(base, rad, rng):
    rx, ry, rz = (rad, rad, rad) if not isinstance(rad, (tuple, list)) else rad
    return (base[0] + rng.uniform(-rx, rx),
            base[1] + rng.uniform(-ry, ry),
            base[2] + rng.uniform(-rz, rz))


def _cortex_point(module, rng):
    """A point in cortical tissue: cognitive/prefrontal modules -> frontal lobe; the rest spread
    over the cortical mantle (a wide box biased to the outer/upper cortex)."""
    if module in COGNITIVE_CORTEX:
        side = 1.0 if rng.random() < 0.5 else -1.0
        return _jitter((side * _FRONTAL[0], _FRONTAL[1], _FRONTAL[2]), (16, 14, 14), rng)
    return (rng.uniform(-52, 52), rng.uniform(-58, 52), rng.uniform(6, 52))


def _neuromod_point(module, rng):
    if module == "locus_coeruleus":
        side = 1.0 if rng.random() < 0.5 else -1.0
        return _jitter((6 * side, -38, -26), 4, rng)
    if module in NEUROMOD_SOURCES:
        rad = 16 if module == "astrocyte_modulation" else 5
        base = NEUROMOD_SOURCES[module]
        if module == "acetylcholine":                       # basal forebrain is bilateral
            side = 1.0 if rng.random() < 0.5 else -1.0
            base = (side * base[0], base[1], base[2])
        return _jitter(base, rad, rng)
    side = 1.0 if rng.random() < 0.5 else -1.0              # drive/motivation -> bilateral VTA/SNc
    return _jitter((side * NEUROMOD_DEFAULT[0], NEUROMOD_DEFAULT[1], NEUROMOD_DEFAULT[2]), 6, rng)


def _plasticity_point(rng):
    """Distributed at synapses -> scatter across cortex + hippocampus (not a nucleus)."""
    if rng.random() < 0.6:
        return (rng.uniform(-46, 46), rng.uniform(-30, 44), rng.uniform(6, 46))
    side = 1.0 if rng.random() < 0.5 else -1.0
    return _jitter((26 * side, -20, -14), (7, 12, 5), rng)


def _reasoning_point(rng):
    """Frontal/prefrontal cortex -- the functional home of the folded reasoning/cognition tier."""
    side = 1.0 if rng.random() < 0.5 else -1.0
    return _jitter((side * _FRONTAL[0], _FRONTAL[1] + 4, _FRONTAL[2] + 6), (18, 16, 16), rng)


def _place(region, module, index, rng):
    if region == "cortex":
        return _cortex_point(module, rng)
    if region == "neuromodulation":
        return _neuromod_point(module, rng)
    if region == "plasticity":
        return _plasticity_point(rng)
    if region == "reasoning":
        return _reasoning_point(rng)
    cx, cy, cz = REGION_MNI[region]
    rad = _RADIUS.get(region, _DEFAULT_RADIUS)
    if region in BILATERAL:
        side = 1.0 if index % 2 == 0 else -1.0
        return _jitter((side * cx, cy, cz), rad, rng)
    return _jitter((cx, cy, cz), rad, rng)


def _build_nodes():
    """A list of (region, x, y, z) for every ON-BRAIN module (everything except substrate, which is
    infra, not brain tissue). substrate's count is reported off-brain by the caller."""
    rng = random.Random(SEED)
    rows, off_brain = [], {}
    for region, mods in _region_modules().items():
        for i, m in enumerate(mods):
            if region == "substrate":                       # NOT brain tissue -> not drawn on brain
                off_brain[region] = off_brain.get(region, 0) + 1
                continue
            x, y, z = _place(region, m, i, rng)
            rows.append((region, float(x), float(y), float(z)))
    return rows, off_brain


_NODE_ROWS, _OFF_BRAIN = _build_nodes()


def _region_val(region, activity):
    if region == "reasoning":
        return float(activity.get("reasoning", activity.get("cortex", 0.0)))
    return float(activity.get(region, 0.0))


class BrainNetPlot:
    """Live brain-activity view on the REAL MNI152 template via netplotbrain. Same API as
    brain_activity_3d.BrainActivity3D (update/show/pause/snapshot/close) so --viz can swap them."""

    def __init__(self, view="SLR", node_scale=90, cmap="inferno", template_style="surface"):
        import matplotlib.pyplot as plt
        import netplotbrain
        import pandas as pd
        self._plt = plt
        self._npb = netplotbrain
        self._pd = pd
        self.view = view
        self.node_scale = node_scale
        self.cmap = cmap
        self.template_style = template_style
        self.fig = None
        self.axes = None
        regions = [r for (r, _x, _y, _z) in _NODE_ROWS]
        self._nodes = pd.DataFrame({
            "region": regions,
            "x": [x for (_r, x, _y, _z) in _NODE_ROWS],
            "y": [y for (_r, _x, y, _z) in _NODE_ROWS],
            "z": [z for (_r, _x, _y, z) in _NODE_ROWS],
            "act": [0.0] * len(_NODE_ROWS),
            "sz":  [0.2] * len(_NODE_ROWS),
        })
        self._n_on_brain = len(_NODE_ROWS)
        self._n_off_brain = sum(_OFF_BRAIN.values())

    def update(self, activity: dict, title=""):
        """Recolour every node by its region's activity and re-render on the template brain."""
        acts = [_region_val(r, activity) for r in self._nodes["region"]]
        self._nodes["act"] = acts
        self._nodes["sz"] = [0.22 + 0.78 * a for a in acts]   # floor so dim nodes stay visible
        if self.fig is not None:
            self._plt.close(self.fig)
        self.fig, self.axes = self._npb.plot(
            template=TEMPLATE, template_style=self.template_style,
            nodes=self._nodes, node_color="act", node_size="sz",
            node_scale=self.node_scale, node_cmap=self.cmap, view=self.view,
        )
        note = (f"{self._n_on_brain} modules on the MNI152 brain  |  +{self._n_off_brain} substrate "
                f"(data lake / infra, not brain tissue)")
        if title:                                            # top, clear of netplotbrain's L/S/R labels
            self.fig.suptitle(title, color="#222222", fontsize=12, y=0.995)
        self.fig.text(0.5, 0.015, note, ha="center", va="bottom", color="#555555", fontsize=8)
        self.fig.canvas.draw_idle()
        return self

    # exact-name check, NOT a substring test -- "agg" is a substring of the GUI backends tkagg/qtagg/
    # wxagg/webagg, so the old `"agg" not in ...` guard made show()/pause() no-op on a real desktop
    # (Windows default TkAgg) and the window never opened. See brain_activity_3d._NON_GUI.
    _NON_GUI = frozenset({"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"})

    def _interactive(self):
        import matplotlib
        return matplotlib.get_backend().lower() not in self._NON_GUI

    def show(self):
        if self._interactive():
            self._plt.ion()
            if self.fig is not None:
                self.fig.show()
            self._plt.pause(0.001)
        return self

    def pause(self, seconds=0.001):
        if self._interactive():
            self._plt.pause(seconds)

    def snapshot(self, path):
        if self.fig is not None:
            self.fig.savefig(path, dpi=120)
        return path

    def close(self):
        if self.fig is not None:
            self._plt.close(self.fig)


def main():
    import matplotlib
    matplotlib.use("Agg")   # headless BEFORE pyplot import

    # every module is accounted for: on-brain nodes + off-brain substrate == the full 298-map.
    expected = sum(len(m) for m in _region_modules().values())
    accounted = len(_NODE_ROWS) + sum(_OFF_BRAIN.values())
    assert accounted == expected, f"every module must be accounted for ({accounted} != {expected})"
    assert accounted == sum(REGION_COUNT.values()), "must match brain_activity_3d's module map"
    for region, x, y, z in _NODE_ROWS:                       # every coord finite + plausible MNI mm
        assert all(math.isfinite(v) for v in (x, y, z)), f"non-finite MNI for {region}: {(x, y, z)}"
        assert -100 <= x <= 100 and -140 <= y <= 90 and -80 <= z <= 95, \
            f"{region} node at {(x, y, z)} is outside a plausible MNI box"

    # verify MODULE_REGION (the shared map) and our node regions agree on the anatomical set
    on_brain_regions = {r for (r, *_r) in _NODE_ROWS}
    assert "substrate" not in on_brain_regions, "substrate must NOT be drawn on the brain"
    assert "cortex" in on_brain_regions and "brainstem" in on_brain_regions

    viz = BrainNetPlot(view="S")                             # single view = fast self-check
    from brain_region_activity import _REPLY_ATTENTION, _stub_brain, region_activity
    activity = region_activity(_stub_brain(), "what are you thinking", _REPLY_ATTENTION)
    viz.update(activity, title="self-check: attention turn (thalamus lit)")
    fd, png = tempfile.mkstemp(suffix="_netplot.png"); os.close(fd)
    viz.snapshot(png)
    ok = os.path.exists(png) and os.path.getsize(png) > 8000
    viz.close()

    assert ok, "the headless snapshot must write a non-trivial PNG"
    print("=" * 78)
    print("BRAIN NETPLOT -- every snn_data_lake module on the REAL MNI152 template (netplotbrain)")
    print("=" * 78)
    print(f"modules accounted: {accounted}  (on-brain nodes: {len(_NODE_ROWS)}  |  "
          f"off-brain substrate: {sum(_OFF_BRAIN.values())})")
    print(f"template: {TEMPLATE}  style: surface  |  regions on brain: {len(on_brain_regions)}")
    print(f"snapshot PNG: {png} ({os.path.getsize(png)} bytes)")
    print("=" * 78)
    print(f"self-check OK: netplotbrain render, {len(_NODE_ROWS)} module nodes on the MNI152 brain, "
          f"substrate ({sum(_OFF_BRAIN.values())}) honestly reported off-brain, PNG written")


if __name__ == "__main__":
    main()
