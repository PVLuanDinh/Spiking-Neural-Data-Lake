"""
Interactive 3D brain-activity visualization (matplotlib) for the SNDL Spiking Chat App.

Renders EVERY module of the `snn_data_lake` package as one small node, placed at its
region's ANATOMICALLY-plausible location inside a translucent, brain-SHAPED hull, and
lights each node by its REGION's per-turn activity (a perceptual colormap + node size)
so the brain "lights up" as the user converses. The real region->region tracts of
`snn_data_lake.connectome` are drawn as faint lines between the region CENTROIDS, and
only the centroids are labelled (with a per-region module count) -- the ~300 module
nodes themselves stay unlabelled so the figure reads as a brain, not a wall of text.

The nodes come from three sources, imported/enumerated at load:
  * `snn_data_lake.REGIONS`     -- the authoritative region -> modules map (the anatomy),
  * `snn_data_lake.TORCH_TIER`  -- the torch-gated organs (part of the anatomical map),
  * `snn_data_lake/reasoning/*` -- the folded research / cognition tier (~120 modules).

Coordinate frame: x = left(-)/right(+) lateral, y = posterior(-)/anterior(+),
z = inferior(-)/superior(+); the brain fits ~[-1, 1]^3.

HONEST framing of the non-anatomical tiers:
  * `substrate`  is NOT a brain structure (the spike data-lake / telemetry / infra). Its
    nodes are drawn as a clearly-separated cloud BELOW the brain and labelled as such.
  * `plasticity` is NOT a discrete nucleus (learning rules live distributed AT synapses).
    Its nodes are scattered faintly across cortex + hippocampus and labelled as distributed.
  * `reasoning`  is the folded research/cognition tier, not a wired connectome region. Its
    nodes form an anterior cloud near the prefrontal cortex and are labelled honestly.
The per-region activity is a region-LEVEL estimate (from `brain_region_activity`); the viz
applies a region's value to all of its modules, with a tiny bounded per-module jitter added
for VISUAL texture only (it never changes the underlying region estimate).

matplotlib (and numpy) are imported ONLY here (and on the opt-in --viz chat path) -- never
in the zero-dep-core `snn_data_lake` package. pyplot is imported lazily inside the class so
a headless caller can force the Agg backend BEFORE the first pyplot import (see main()).

  # headless self-check (no display, writes a PNG, prints the module count + region breakdown):
  PYTHONUTF8=1 PYTHONPATH="src;integrations/chat_app" MPLBACKEND=Agg \
      python integrations/chat_app/brain_activity_3d.py
"""
import glob
import math
import os
import random
import tempfile

import matplotlib  # NOTE: backend NOT chosen here -- pyplot is imported lazily
import numpy as np
from brain_region_activity import REGIONS as REGION_NAMES  # the 17 anatomical region names

# The authoritative anatomical map lives in the zero-dep core (no matplotlib there).
from snn_data_lake import REGIONS as _SNN_REGIONS, TORCH_TIER as _TORCH_TIER
import snn_data_lake as _snn_pkg

SEED = 1605  # fixed -> deterministic jitter (a per-instance random.Random, never the global RNG)

# ----------------------------------------------------------------------------------------
# region -> ANATOMICAL centre (x=L-/R+, y=post-/ant+, z=inf-/sup+). Regions in BILATERAL are
# paired: their modules are split L/R at x = +/- |x| (the sign of x here is the magnitude).
# ----------------------------------------------------------------------------------------
REGION_CENTER = {
    "cortex":          (0.00,  0.10,  0.55),   # whole cortical mantle -- placed over the outer upper hull
    "hippocampus":     (0.35, -0.05, -0.30),   # medial temporal, bilateral
    "amygdala":        (0.30,  0.20, -0.30),   # anterior to hippocampus, bilateral
    "basal_ganglia":   (0.25,  0.10,  0.10),   # deep, lateral to thalamus, bilateral
    "thalamus":        (0.00, -0.05,  0.05),   # central deep relay + GWT hub
    "hypothalamus":    (0.00,  0.05, -0.18),   # below thalamus, anterior-inferior
    "epithalamus":     (0.00, -0.22,  0.12),   # posterior-superior to thalamus (habenula/pineal)
    "midbrain":        (0.00, -0.10, -0.20),   # superior brainstem
    "pons":            (0.00, -0.15, -0.45),   # mid brainstem
    "medulla":         (0.00, -0.15, -0.72),   # inferior brainstem (lowest)
    "brainstem":       (0.00, -0.12, -0.45),   # the reflex/arm-control column
    "cerebellum":      (0.00, -0.70, -0.45),   # posterior + inferior (behind + below)
    "olfactory":       (0.00,  0.60, -0.42),   # anterior-inferior frontal base
    "claustrum":       (0.45,  0.10,  0.05),   # lateral (insula/putamen border), bilateral
    "neuromodulation": (0.00, -0.10, -0.25),   # diffuse nuclei -- real sub-sources below
    "plasticity":      (0.00,  0.00,  0.25),   # distributed (scattered across cortex + hippocampus)
    "substrate":       (0.00, -0.35, -1.05),   # NON-anatomical infra cloud, BELOW the brain
    "reasoning":       (0.00,  0.70,  0.40),    # folded research/cognition cloud, anterior (prefrontal)
}

BILATERAL = {"hippocampus", "amygdala", "basal_ganglia", "claustrum"}

# per-region jitter radius (scalar = isotropic; a 3-tuple = per-axis, for elongated structures)
_RADIUS = {
    "thalamus": 0.07, "hypothalamus": 0.06, "epithalamus": 0.05,
    "midbrain": 0.07, "pons": 0.06, "medulla": 0.06,
    "brainstem": (0.06, 0.08, 0.20),    # a vertical reflex/arm-control column
    "cerebellum": (0.30, 0.22, 0.16),   # a wide posterior-inferior lobe
    "olfactory": 0.07,
    "hippocampus": 0.11, "amygdala": 0.08, "basal_ganglia": 0.10, "claustrum": 0.06,
    "substrate": 0.22, "reasoning": (0.26, 0.20, 0.20),
}
_DEFAULT_RADIUS = 0.08

# the cortical modules whose function is prefrontal / executive / language / reasoning-cognition
# -- these bias ANTERIOR (+y) into the frontal lobe rather than spreading over the whole mantle.
COGNITIVE_CORTEX = {
    "language_brain", "nl_interface", "brain_reasoning", "spiking_lm_serve", "vsa_reasoning",
    "prefrontal_executive", "live_brain", "interaction_learning",
    "access_consciousness", "self_model", "metacognitive_monitor", "theory_of_mind",
    "reasoning_curriculum", "autotelic_goals", "quality_diversity", "self_improvement_suite",
    "wholistic_loop", "integration_measure", "agency_monitor",
}

# neuromodulation: place modules at their REAL diffuse source nuclei (locus_coeruleus handled
# separately because it is a paired/bilateral pontine nucleus). Anything not listed is a
# drive / motivation module -> clustered near the VTA/SNc dopamine source.
NEUROMOD_SOURCES = {
    "raphe_serotonin":      (0.00, -0.15, -0.40),   # midline raphe (serotonin)
    "acetylcholine":        (0.00,  0.30, -0.25),   # basal forebrain (ACh)
    "cortisol":             (0.00,  0.08, -0.22),   # HPA / hypothalamic border (stress hormone)
    "astrocyte_modulation": (0.00,  0.00,  0.10),   # diffuse glia (large radius below)
}
NEUROMOD_DEFAULT = (0.00, -0.10, -0.20)   # VTA/SNc midbrain dopamine source

# honest labels for the non-anatomical / distributed tiers (appended to their centroid label)
_LABEL_NOTE = {
    "substrate":  "data lake / infra (non-anatomical)",
    "plasticity": "distributed learning rules (not a nucleus)",
    "reasoning":  "folded research / cognition tier",
}

# plotted bounds -- every module node MUST fall inside this box (asserted in the self-check).
XLIM = (-1.05, 1.05)
YLIM = (-1.25, 1.15)
ZLIM = (-1.40, 0.90)


def _reasoning_modules():
    """The folded research/cognition tier: every module under snn_data_lake/reasoning/ (except
    the package __init__). Enumerated from the installed package location, sorted for determinism."""
    rdir = os.path.join(os.path.dirname(_snn_pkg.__file__), "reasoning")
    names = []
    for p in glob.glob(os.path.join(rdir, "*.py")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem != "__init__":
            names.append(stem)
    return sorted(names)


def _region_modules():
    """region -> [modules], covering the FULL package: REGIONS + the torch tier + the reasoning tier."""
    rm = {r: list(mods) for r, mods in _SNN_REGIONS.items()}
    for r, mods in _TORCH_TIER.items():
        rm.setdefault(r, []).extend(mods)
    rm["reasoning"] = _reasoning_modules()
    return rm


def _jitter(base, rad, rng):
    """A bounded uniform offset around `base`; `rad` is a scalar (isotropic) or a per-axis 3-tuple."""
    rx, ry, rz = (rad, rad, rad) if not isinstance(rad, (tuple, list)) else rad
    return (base[0] + rng.uniform(-rx, rx),
            base[1] + rng.uniform(-ry, ry),
            base[2] + rng.uniform(-rz, rz))


def _cortex_point(module, rng):
    """A point on the OUTER UPPER hull (the cortical mantle). Cognitive/prefrontal modules bias
    anterior (+y) into the frontal lobe; the rest spread over the whole upper mantle."""
    ax, by, cz = 0.66, 0.86, 0.56       # inset half-axes -> nodes sit just inside the hull
    if module in COGNITIVE_CORTEX:
        theta = rng.uniform(math.pi * 0.30, math.pi * 0.70)   # anterior band (+y)
        phi = rng.uniform(0.35, 1.05)                         # upper-front
        by = 0.90                                             # push a touch more anterior
    else:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(0.15, 1.50)                         # whole upper mantle down to the sides
    st = math.sin(phi)
    x = ax * st * math.cos(theta) + rng.uniform(-0.03, 0.03)
    y = by * st * math.sin(theta) + rng.uniform(-0.03, 0.03)
    z = cz * math.cos(phi) + 0.12 + rng.uniform(-0.03, 0.03)  # +0.12 -> sit on the upper hull
    return (x, y, z)


def _neuromod_point(module, rng):
    """Place a neuromodulatory module at its real diffuse source nucleus."""
    if module == "locus_coeruleus":                            # paired pontine LC (bilateral)
        side = 1.0 if rng.random() < 0.5 else -1.0
        return _jitter((0.10 * side, -0.15, -0.45), 0.05, rng)
    if module in NEUROMOD_SOURCES:
        rad = 0.18 if module == "astrocyte_modulation" else 0.05
        return _jitter(NEUROMOD_SOURCES[module], rad, rng)
    return _jitter(NEUROMOD_DEFAULT, 0.12, rng)                # drive/motivation -> near VTA/SNc


def _plasticity_point(rng):
    """Scatter faintly across cortex + hippocampus -- learning rules are distributed at synapses,
    NOT a discrete locus (see the docstring / legend note)."""
    if rng.random() < 0.6:                                     # cortical synapses (upper mantle)
        return (rng.uniform(-0.55, 0.55), rng.uniform(-0.20, 0.50), rng.uniform(0.15, 0.55))
    side = 1.0 if rng.random() < 0.5 else -1.0                 # hippocampal synapses (medial temporal)
    return _jitter((0.35 * side, -0.05, -0.30), 0.10, rng)


def _place(region, module, index, rng):
    """The anatomical position of one module: a per-region cluster with deterministic jitter."""
    if region == "cortex":
        return _cortex_point(module, rng)
    if region == "neuromodulation":
        return _neuromod_point(module, rng)
    if region == "plasticity":
        return _plasticity_point(rng)
    cx, cy, cz = REGION_CENTER[region]
    rad = _RADIUS.get(region, _DEFAULT_RADIUS)
    if region in BILATERAL:                                    # alternate L/R hemispheres
        side = 1.0 if index % 2 == 0 else -1.0
        return _jitter((side * cx, cy, cz), rad, rng)
    return _jitter((cx, cy, cz), rad, rng)


def _build_positions():
    """MODULE_POS: 'region/module' -> (x,y,z) for EVERY module; MODULE_REGION: 'region/module' -> region.
    Deterministic -- a single per-instance random.Random(SEED) draws every jitter (no global RNG)."""
    rng = random.Random(SEED)
    pos, region_of = {}, {}
    for region, mods in _region_modules().items():
        for i, m in enumerate(mods):
            key = f"{region}/{m}"
            pos[key] = _place(region, m, i, rng)
            region_of[key] = region
    return pos, region_of


MODULE_POS, MODULE_REGION = _build_positions()

# per-region module count + centroid (the label anchor + the tract endpoints)
REGION_COUNT, _sums = {}, {}
for _k, _r in MODULE_REGION.items():
    REGION_COUNT[_r] = REGION_COUNT.get(_r, 0) + 1
    sx, sy, sz = _sums.get(_r, (0.0, 0.0, 0.0))
    px, py, pz = MODULE_POS[_k]
    _sums[_r] = (sx + px, sy + py, sz + pz)
REGION_CENTROID = {r: (sx / REGION_COUNT[r], sy / REGION_COUNT[r], sz / REGION_COUNT[r])
                   for r, (sx, sy, sz) in _sums.items()}


def _connectome_edges():
    """The major region->region tracts from the REAL `snn_data_lake.connectome` (map each typed
    edge's endpoint nodes to their region; drop the None-region effectors and self-edges). Falls
    back to a small curated tract set if the connectome is not importable. Two CURATED infra edges
    are always added so `substrate` and `plasticity` (which have no connectome node) are wired in:
    substrate<->hippocampus (the lake working memory) and cortex<->plasticity (learning rules)."""
    regset = set(REGION_NAMES)
    pairs = set()
    try:
        from snn_data_lake import connectome as C
        for e in C.EDGES:
            rs = C.NODES[e["source"]][0]
            rt = C.NODES[e["target"]][0]
            if rs in regset and rt in regset and rs != rt:
                pairs.add(tuple(sorted((rs, rt))))
    except Exception:
        pairs = {
            ("cortex", "thalamus"), ("thalamus", "basal_ganglia"), ("cortex", "basal_ganglia"),
            ("cortex", "hippocampus"), ("amygdala", "brainstem"), ("amygdala", "hypothalamus"),
            ("cerebellum", "pons"), ("brainstem", "midbrain"), ("cortex", "claustrum"),
        }
    pairs.add(tuple(sorted(("hippocampus", "substrate"))))   # curated: the lake working memory
    pairs.add(tuple(sorted(("cortex", "plasticity"))))       # curated: learning rules on cortex
    return sorted(pairs)


EDGES = _connectome_edges()


class BrainActivity3D:
    """A live 3D brain-activity figure: every snn_data_lake module is a small node at its
    region's anatomical position inside a translucent brain-shaped hull, wired by the connectome
    tracts between region centroids, recoloured + resized each turn by a per-region activity dict."""

    def __init__(self, cmap="inferno", figsize=(9.5, 8.5), elev=18.0, azim=-60.0):
        import matplotlib.pyplot as plt  # lazy -> respects any prior matplotlib.use()
        from matplotlib.colors import Normalize
        self._plt = plt
        self._cmap = matplotlib.colormaps[cmap]
        self._norm = Normalize(vmin=0.0, vmax=1.0)
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._scat = None
        self._cbar = None
        self._labels = []                               # per-turn centroid labels (redrawn each update)
        # stable node order + coordinate arrays
        self._keys = list(MODULE_POS)
        self._xs = [MODULE_POS[k][0] for k in self._keys]
        self._ys = [MODULE_POS[k][1] for k in self._keys]
        self._zs = [MODULE_POS[k][2] for k in self._keys]
        # tiny bounded per-module VISUAL jitter (never changes the region estimate); own fixed-seed RNG
        _vj = random.Random(SEED + 7)
        self._vjit = {k: _vj.uniform(-0.04, 0.04) for k in self._keys}
        self.ax.view_init(elev=elev, azim=azim)
        self._draw_static()

    def _draw_hull(self):
        """A translucent brain-SHAPED hull: an ellipsoid (wider than tall, elongated front<->back,
        flattened base) plus a small posterior-inferior cerebellar bulge. Low alpha so nodes show."""
        ax = self.ax
        u = np.linspace(0.0, 2.0 * np.pi, 64)
        v = np.linspace(0.0, np.pi, 44)
        U, V = np.meshgrid(u, v)
        sinV = np.sin(V)
        a, b, c = 0.74, 0.96, 0.66                      # x lateral, y ant-post, z vertical
        X = a * sinV * np.cos(U)
        Y = b * sinV * np.sin(U)
        Z = c * np.cos(V)
        Y = np.where(Y > 0, Y * 1.06, Y * 1.02)         # slight frontal(+y)/occipital(-y) elongation
        Z = np.where(Z < 0, Z * 0.80, Z) + 0.06         # flatten + lift the cerebrum base
        hull_col = (0.55, 0.60, 0.78)
        ax.plot_surface(X, Y, Z, color=hull_col, alpha=0.10, linewidth=0,
                        antialiased=True, shade=False, zorder=0)
        # small cerebellar bulge (posterior + inferior)
        Xc = 0.42 * sinV * np.cos(U)
        Yc = 0.34 * sinV * np.sin(U) - 0.66
        Zc = 0.26 * np.cos(V) - 0.42
        ax.plot_surface(Xc, Yc, Zc, color=hull_col, alpha=0.10, linewidth=0,
                        antialiased=True, shade=False, zorder=0)

    def _draw_static(self):
        """Draw the hull, the faint region->region tracts, and the region-centroid labels once."""
        ax = self.ax
        self._draw_hull()
        for a, b in EDGES:                              # tracts between region CENTROIDS
            if a in REGION_CENTROID and b in REGION_CENTROID:
                xa, ya, za = REGION_CENTROID[a]
                xb, yb, zb = REGION_CENTROID[b]
                ax.plot([xa, xb], [ya, yb], [za, zb], color=(0.5, 0.5, 0.58),
                        alpha=0.22, linewidth=0.7, zorder=1)
        # region-centroid labels are drawn PER-TURN in update() (only for the regions active that
        # turn + the always-on honesty tags) so the dense deep-brain nuclei do not pile up.
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_zlim(*ZLIM)
        ax.set_box_aspect((XLIM[1] - XLIM[0], YLIM[1] - YLIM[0], ZLIM[1] - ZLIM[0]), zoom=1.55)
        ax.set_axis_off()                               # hull is the frame -- no gray bounding cube
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96)  # kill the empty margins
        self.fig.patch.set_facecolor("#0b0b10")
        ax.set_facecolor("#0b0b10")

    def _region_val(self, r, activity):
        """A region's per-turn activity in [0,1] (reasoning falls back to the cortex value)."""
        if r == "reasoning":
            return float(activity.get("reasoning", activity.get("cortex", 0.0)))
        return float(activity.get(r, 0.0))

    def _draw_labels(self, activity, thr=0.20):
        """Label region centroids for regions ACTIVE this turn (val >= thr) plus the always-on
        honesty tags (substrate/plasticity/reasoning) -- so the dim deep nuclei don't pile up."""
        for t in self._labels:
            t.remove()
        self._labels = []
        for r, (x, y, z) in REGION_CENTROID.items():
            if self._region_val(r, activity) < thr and r not in _LABEL_NOTE:
                continue
            label = f"{r} ({REGION_COUNT[r]})"
            note = _LABEL_NOTE.get(r)
            if note:
                label += f"\n{note}"
            self._labels.append(self.ax.text(x, y, z + 0.10, label, fontsize=6.5,
                                 color=(0.85, 0.85, 0.90), ha="center", va="bottom", zorder=6))

    def _values(self, activity):
        """Per-node value in [0,1]: its REGION's activity (reasoning falls back to cortex), plus the
        tiny bounded per-module visual jitter. HONEST: this is a region-LEVEL estimate applied to all
        of a region's modules; the jitter is cosmetic and never alters the region value itself."""
        out = []
        for k in self._keys:
            r = MODULE_REGION[k]
            v = activity.get("reasoning", activity.get("cortex", 0.0)) if r == "reasoning" \
                else float(activity.get(r, 0.0))
            v += self._vjit[k]
            out.append(0.0 if v < 0.0 else (1.0 if v > 1.0 else v))
        return out

    def update(self, activity: dict, title=""):
        """Recolour + resize every module node by its REGION's `activity` (region -> [0,1]) and set
        the title. Nodes brighten (perceptual colormap) and grow with activity, so the brain lights up."""
        vals = self._values(activity)
        sizes = [7.0 + 55.0 * v for v in vals]
        if self._scat is not None:
            self._scat.remove()
        self._scat = self.ax.scatter(
            self._xs, self._ys, self._zs, c=vals, cmap=self._cmap, norm=self._norm,
            s=sizes, edgecolors="white", linewidths=0.2, depthshade=True, zorder=3)
        if self._cbar is None:                          # one colorbar (fixed 0..1 norm)
            import matplotlib.cm as cm
            mappable = cm.ScalarMappable(norm=self._norm, cmap=self._cmap)
            self._cbar = self.fig.colorbar(mappable, ax=self.ax, shrink=0.6, pad=0.02)
            self._cbar.set_label("region activity (0..1)", color="#cccccc")
            self._cbar.ax.yaxis.set_tick_params(color="#cccccc")
            for t in self._cbar.ax.get_yticklabels():
                t.set_color("#cccccc")
        self._draw_labels(activity)                     # label only the regions lit this turn
        self.ax.set_title(title, color="#e8e8ee", fontsize=11)
        self.fig.canvas.draw_idle()
        return self

    # Backends that render to a FILE and have no event loop. Everything else has a window.
    # NOTE: this MUST be an exact-name check, not a substring test. The old guard was
    # `"agg" not in get_backend().lower()`, which is True for headless "agg" -- but ALSO matches the
    # GUI backends "tkagg" / "qtagg" / "wxagg" / "webagg", because their names CONTAIN "agg". On a
    # normal desktop matplotlib (Windows default: TkAgg) that made show() and pause() silently no-op,
    # so the 3D window never opened and nothing errored. Headless tests could not catch it either:
    # under MPLBACKEND=Agg the no-op IS the correct behaviour, so the bug only ever showed on a real
    # desktop run.
    _NON_GUI = frozenset({"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"})

    def _interactive(self):
        """True only on a GUI backend -- lets show()/pause() no-op cleanly under headless Agg."""
        return matplotlib.get_backend().lower() not in self._NON_GUI

    def show(self):
        """Open the figure interactively (non-blocking, so a live chat loop can keep updating).
        No-op on a non-interactive (headless Agg) backend."""
        if self._interactive():
            self._plt.ion()
            self.fig.show()
            self._plt.pause(0.001)
        return self

    def pause(self, seconds=0.001):
        """Let the GUI event loop process a live update (used between turns). No-op under Agg."""
        if self._interactive():
            self._plt.pause(seconds)

    def snapshot(self, path):
        """Save the current figure to `path` (PNG). Returns the path."""
        self.fig.savefig(path, dpi=120, facecolor=self.fig.get_facecolor())
        return path

    def close(self):
        self._plt.close(self.fig)


def _sample_activity():
    """A deterministic sample activity dict over the 17 regions (an 'attention' turn profile)."""
    from brain_region_activity import _REPLY_ATTENTION, _stub_brain, region_activity
    return region_activity(_stub_brain(), "what are you thinking", _REPLY_ATTENTION)


def main():
    # HEADLESS-SAFE: force the non-interactive Agg backend BEFORE pyplot is ever imported,
    # so this self-check runs in a gate with no display.
    matplotlib.use("Agg")

    # (b) MODULE_POS must cover EVERY module in REGIONS + TORCH_TIER + reasoning.
    expected = sum(len(m) for m in _SNN_REGIONS.values()) \
        + sum(len(m) for m in _TORCH_TIER.values()) + len(_reasoning_modules())
    n_modules = len(MODULE_POS)
    assert n_modules == expected, f"MODULE_POS must cover every module ({n_modules} != {expected})"
    assert n_modules >= 200, f"the package has >= 200 modules; MODULE_POS mapped only {n_modules}"

    # (c) every module position is finite and inside the plotted bounds.
    for k, (x, y, z) in MODULE_POS.items():
        assert all(math.isfinite(v) for v in (x, y, z)), f"non-finite position for {k}: {(x, y, z)}"
        assert XLIM[0] <= x <= XLIM[1] and YLIM[0] <= y <= YLIM[1] and ZLIM[0] <= z <= ZLIM[1], \
            f"module {k} at {(x, y, z)} is outside the plotted bounds"

    viz = BrainActivity3D()
    activity = _sample_activity()
    viz.update(activity, title="self-check: attention turn (thalamus lit) -- every module mapped")

    fd, png = tempfile.mkstemp(suffix="_brain3d.png")
    os.close(fd)
    viz.snapshot(png)
    exists = os.path.exists(png) and os.path.getsize(png) > 5000
    viz.close()

    # (a) the headless snapshot must write a non-trivial PNG.
    assert exists, "the headless snapshot must write a non-trivial PNG"
    assert len(EDGES) >= 10, f"the connectome must supply the major tracts (got {len(EDGES)} edges)"

    print("=" * 78)
    print("BRAIN ACTIVITY 3D -- every snn_data_lake module mapped inside a brain-shaped hull")
    print("=" * 78)
    print(f"modules mapped: {n_modules} (REGIONS+TORCH_TIER+reasoning) | tract edges: {len(EDGES)} "
          f"| backend: {matplotlib.get_backend()}")
    print("region breakdown (module count @ anatomical centroid):")
    for r in sorted(REGION_COUNT, key=lambda x: -REGION_COUNT[x]):
        cx, cy, cz = REGION_CENTROID[r]
        note = f"  [{_LABEL_NOTE[r]}]" if r in _LABEL_NOTE else ""
        print(f"  {r:15} {REGION_COUNT[r]:3d}  centroid=({cx:+.2f},{cy:+.2f},{cz:+.2f}){note}")
    print(f"snapshot PNG: {png} ({os.path.getsize(png)} bytes)")
    print("=" * 78)
    print(f"self-check OK: headless Agg render, {n_modules} module nodes placed in-bounds inside the "
          f"brain hull, {len(EDGES)} tracts wired between region centroids, PNG written to {png}")


if __name__ == "__main__":
    main()
