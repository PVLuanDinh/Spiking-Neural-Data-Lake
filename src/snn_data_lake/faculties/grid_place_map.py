"""
Grid -> place map: bind a percept to a location via a spiking grid-cell population.

WHAT THE BRAIN DOES. Entorhinal GRID cells tile space with a multi-scale hexagonal code; a hippocampal
PLACE cell fires in one small region of an environment (O'Keefe & Dostrovsky, Brain Res 34:171, 1971) and
is built, in the standard model, from a weighted SUM of grid-cell inputs across several scales (Solstad,
Moser & Einevoll, Hippocampus 16:1026, 2006; Fiete, Burak & Brookings, J Neurosci 28:6858, 2008): the
grid PHASES across scales combine to a population code that is UNIQUE over an area far larger than any
single grid period, so a thresholded read-out of that population fires at exactly one place. A place cell
therefore BINDS "what is here" (a percept / landmark) to "where here is" (a location), and the map answers
two questions: place_at(location) -> the landmark bound there, and recall_place(landmark) -> where it was
seen.

THE GAP THIS FILLS (verified against the repo's spatial substrate). grid_cells.py has the hexagonal grid
code but forms NO place fields and drives no behaviour; allocentric_map.py tracks a 2-D position estimate
but binds names to raw (x, y) FLOATS, not to a neural spatial code. This module turns the grid population
into PLACE cells (one-shot Hebbian prototypes over the grid code) and uses them to content-address a
percept FROM a location and to decode a location FROM a bound percept -- the grid->place read-out that was
missing.

LEARNING RULE. One-shot HEBBIAN prototype imprint: a place cell stores the grid-population rate vector at
the moment of binding (weight_i = presynaptic grid rate_i -- "cells that fire together wire together"),
consolidated by a running average on re-observation. This is LOCAL plasticity: NO gradient, NO backprop,
no loss anywhere -- so there is no BPTT/e-prop cosine to report (that check applies only to gradient rules).

DISTINCTNESS. REUSES GridCell from grid_cells.py for every hexagonal firing field (does NOT reimplement the
grid math). Not ring_attractor.py (1-D heading integrator), not episodic_recall.py / sleep_replay.py
(DISCRETE-memory separation/completion; this named `hippocampus.py`, RETIRED in the Policy-3 wave --
see `hippocampus/__init__.py` -- corrected 2026-08-04). ZERO coupling to the arm / safety envelope: it
imports only grid_cells (which is pure math/random) and receives a plain (x, y) from the caller (e.g.
AllocentricMap.position).

HONEST SCOPE. 2-D TOY positions from the arm loop, NOT real navigation. Grid and place cells are analytic
RATE units -- rate/prototype coding, which is the spiking-native mode chosen here; a deterministic
rate->spike readout (grid_spikes / place_spikes) emits literal integer spike counts, but the place DECISION
is taken on the stable population-vector CORRELATION for reproducibility. No path integration, no drift, no
theta phase; the code is unique only within the bounded toy extent, not over an open world.

  python -m snn_data_lake.faculties.grid_place_map     # (self-check; also runnable by run_selfchecks.py)
"""

import math
import random
import zlib

from snn_data_lake.faculties.grid_cells import GridCell


def _pearson(a, b):
    """Population-vector correlation of two grid codes, in [-1, 1] (0 if either is flat)."""
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) * (x - ma) for x in a)
    vb = sum((y - mb) * (y - mb) for y in b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


class GridPlaceMap:
    """Multi-scale spiking grid population -> Hebbian place cells that bind a percept to a location.

    Usage:
        gpm = GridPlaceMap(seed=0)
        gpm.observe((0.3, -0.4), "door")      # form a place field binding 'door' at this location
        gpm.place_at((0.3, -0.4))             # -> 'door'   (the percept whose place cell fires here)
        gpm.place_at((1.2,  1.2))             # -> None      (silent: no place cell bound out here)
        gpm.recall_place("door")              # -> (~0.30, ~-0.40)  decoded from the grid-code template
    """

    def __init__(
        self,
        spacings=(0.4, 0.57, 0.8, 1.1, 1.6, 2.3),
        orientations_deg=None,
        phases_per_side=3,
        extent=1.5,
        gain=3.0,
        spike_window=20,
        max_spike_rate=1.0,
        seed=0,
    ):
        """
        Args:
            spacings: grid-module periods (multi-scale, ratio ~1.4 like entorhinal modules).
            orientations_deg: per-module lattice orientation; default staggers 12 deg per module.
            phases_per_side: phase offsets per axis within each module's unit cell (tiles the module).
            extent: half-width of the square toy environment [-extent, extent]^2 (decode range).
            gain: grid-cell firing sharpness (passed to GridCell).
            (place_threshold was removed: the firing criterion is now derived per query in
            place_at() from the population's own off-field baseline, not configured.)
            spike_window: timesteps for the deterministic rate->spike readout.
            max_spike_rate: per-step spike probability at a grid cell's peak rate.
            seed: RNG seed -> reproducible grid phases and spike readout.
        """
        self.extent = float(extent)
        self.spike_window = int(spike_window)
        self.max_spike_rate = float(max_spike_rate)
        self.seed = int(seed)
        if orientations_deg is None:
            orientations_deg = [12.0 * k for k in range(len(spacings))]
        # build the grid population by REUSING GridCell (one module per spacing, phases tile its unit cell)
        self._cells = []
        for mi, s in enumerate(spacings):
            o = math.radians(orientations_deg[mi % len(orientations_deg)])
            offs = [s * (p + 0.5) / phases_per_side for p in range(phases_per_side)]
            for px in offs:
                for py in offs:
                    self._cells.append(
                        GridCell(
                            spacing=s,
                            orientation=o,
                            phase=(px, py),
                            gain=gain,
                            seed=seed + 1000 * mi,
                        )
                    )
        self.n_grid = len(self._cells)
        self._places = {}  # percept KEY (always str, see _key) -> {"template", "center", "n"}
        self._lattice = None  # cached decode lattice: (points, grid codes)

    # ---- grid population (rate + spiking readout) ----
    def grid_code(self, position):
        """Grid-population RATE vector at a 2-D position (one entry per grid cell)."""
        x, y = position
        return [c.rate(x, y) for c in self._cells]

    def grid_spikes(self, position, window=None):
        """Deterministic rate->spike readout of the grid population (per-position seeded; reproducible)."""
        w = self.spike_window if window is None else int(window)
        code = self.grid_code(position)
        key = zlib.crc32(f"{self.seed}|{position[0]:.4f}|{position[1]:.4f}".encode("ascii"))
        rng = random.Random(key)
        spikes = []
        for r in code:
            p = min(1.0, self.max_spike_rate * r)
            spikes.append(sum(1 for _ in range(w) if rng.random() < p))
        return spikes

    # ---- place cells: bind / query / recall ----
    @staticmethod
    def _key(percept):
        """Normalize a percept to its place-cell KEY. Place keys are ALWAYS str.

        The map already commits to str keys the moment a percept is seen at a second location --
        variants are named `f"{percept}@{k}"` -- but the FIRST binding used to store the caller's
        object verbatim, so a non-str percept left `_places` heterogeneously typed. The variant scan
        in observe() then raised `AttributeError: 'int' object has no attribute 'startswith'` for
        every later percept that was not that int. Measured on the live wake path: 28 of 31 observe
        calls raised into brain.py's guard, place_at ran 3 times in 60 ticks, and `_place_here` froze
        on one stale label for the remaining 51 -- a constant sitting where a computed value belongs.
        Normalizing here makes the existing key contract TOTAL rather than partial, for every caller
        rather than only the one that was caught."""
        return percept if isinstance(percept, str) else str(percept)

    def _match_bar(self, acts, key):
        """The derived firing bar for `key`: the midpoint between a perfect self-match (1.0) and what
        the REST of the population reads here. See place_at() for the derivation."""
        others = [v for p, v in acts.items() if p != key]
        return (1.0 + (sum(others) / len(others) if others else 0.0)) / 2.0

    def observe(self, position, percept):
        """Bind `percept` to `position`: one-shot Hebbian imprint of the grid code as a place cell.

        New place    -> a place cell whose synaptic template = the grid rate vector here.
        Known place  -> running-average consolidation of template and field center (Hebbian, no gradient).

        A percept seen at a SPATIALLY DIFFERENT location opens a NEW place cell rather than being
        averaged into the old one. Without this, percept identity silently doubles as PLACE identity:
        the live caller binds whatever it currently perceives every tick, so one label perceived at
        many positions had its template averaged over unrelated locations (grid codes averaged toward
        the population mean). Measured consequence upstream: the live winner correlation collapsed to
        a ~0.226 median and place_at answered None on roughly 79% of ticks -- the place code was being
        destroyed by its own consolidation rule, which no firing threshold could repair.

        "Same place" uses the SAME derived criterion as place_at, so binding and recall cannot
        disagree: consolidate into the best-matching existing variant of this percept if it fires
        here, else open a new one. Variants are keyed `percept@k`; the first keeps the bare label, so
        a map with one location per percept is unchanged."""
        percept = self._key(percept)
        code = self.grid_code(position)
        acts = self.place_activation(position)
        variants = [p for p in self._places if p == percept or p.startswith(f"{percept}@")]
        best_v = max(variants, key=lambda p: acts[p]) if variants else None
        if best_v is not None and acts[best_v] >= self._match_bar(acts, best_v):
            rec = self._places[best_v]  # the same place again -> Hebbian consolidation
            n = rec["n"]
            rec["template"] = [
                (rec["template"][i] * n + code[i]) / (n + 1) for i in range(self.n_grid)
            ]
            rec["center"][0] = (rec["center"][0] * n + float(position[0])) / (n + 1)
            rec["center"][1] = (rec["center"][1] * n + float(position[1])) / (n + 1)
            rec["n"] = n + 1
            return best_v
        key = percept if percept not in self._places else f"{percept}@{len(variants)}"
        self._places[key] = {
            "template": code,
            "center": [float(position[0]), float(position[1])],
            "n": 1,
        }
        return key

    def place_activation(self, position):
        """{percept: place-cell activation in [-1, 1]} = grid population-vector correlation with each template."""
        code = self.grid_code(position)
        return {p: _pearson(code, rec["template"]) for p, rec in self._places.items()}

    def place_at(self, position):
        """Content-addressed spatial query: the percept whose place cell fires here, or None if all silent.

        The winner is decided by WITHIN-TICK POPULATION COMPETITION (lateral inhibition), not by a
        fixed correlation cut. The old `place_threshold=0.5` was an absolute cut on a Pearson
        correlation, so place recognition could not tell how discriminable the CURRENT environment
        is: as more percepts are bound the correlation distribution shifts and a frozen 0.5 silently
        becomes either permissive or deaf. That matters because a non-None answer here keys the
        R-STDP place-VALUE table and triggers the grid-bump landmark reset, so a wrong "I am here"
        corrupts the very code the value learns on.

        The criterion is DERIVED, not chosen. A Pearson correlation against a place's own template
        is exactly 1.0 when standing on it, by construction, and settles at the population's
        off-field baseline when far away. So the honest decision boundary is the MIDPOINT between
        those two references: fire when the winner is closer to a perfect self-match than to what
        the rest of this population is currently reading,

            best >= (1.0 + mean(others)) / 2

        which recovers the old hand-set 0.5 as its value in a sparse environment (the measured
        off-field mean is ~-0.006) while tracking the population instead of being frozen to it.

        Measured honestly: the bar sits at 0.436 / 0.532 / 0.502 / 0.506 for 3 / 8 / 20 / 40 bound
        places, with 100% identification at every density. So it MOVES with the population but does
        NOT climb steadily with crowding -- in this environment the grid codes stay well
        decorrelated, so the off-field baseline stays near zero however many places are bound. The
        adaptivity is real but only bites where off-field correlation is genuinely elevated (a small
        or spatially aliased environment); do not claim a crowding response this has not shown.

        It also fixes the failure the constant hid: mere population competition is NOT enough,
        because at a far corner one landmark does genuinely stand out from the pack (measured 0.398
        vs a -0.006 mean) without the agent being anywhere near it. Beating the alternatives and
        matching the template are different claims, and place identity needs the second one.

        A noise floor of 1/sqrt(n_grid) -- the analytic scale of a Pearson correlation over n_grid
        samples, the same idiom used for the episodic-recall thresholds -- guards the degenerate
        case where a single bound place leaves no population to compare against.

        Deliberately STATELESS -- no EMA, no adapted threshold. An adaptive threshold here would
        close a feedback loop: firing triggers observe_landmark, which mutates the CAN bump, which
        re-derives the position fed back into this query. A rate-regulating homeostat in that loop
        would also pin the output rate, destroying the ability to answer "I am nowhere" at all."""
        acts = self.place_activation(position)
        if not acts:
            return None
        best_p = max(acts, key=acts.get)
        best = acts[best_p]
        floor = 1.0 / math.sqrt(self.n_grid)  # correlation noise floor over n_grid samples
        if best < floor:
            return None  # indistinguishable from chance -> nowhere
        return best_p if best >= self._match_bar(acts, best_p) else None

    def place_spikes(self, position, percept, window=None):
        """Literal spike count of one place cell here (rate->spikes on its correlation; on-field high, off ~0)."""
        rec = self._places.get(self._key(percept))
        if rec is None:
            return 0
        w = self.spike_window if window is None else int(window)
        r = max(0.0, _pearson(self.grid_code(position), rec["template"]))
        key = zlib.crc32(f"{self.seed}|place|{position[0]:.4f}|{position[1]:.4f}".encode("ascii"))
        rng = random.Random(key)
        return sum(1 for _ in range(w) if rng.random() < r)

    def _ensure_lattice(self, n_side=31):
        if self._lattice is not None:
            return
        e = self.extent
        axis = [-e + 2.0 * e * i / (n_side - 1) for i in range(n_side)]
        pts = [(x, y) for y in axis for x in axis]
        codes = [self.grid_code(p) for p in pts]
        self._lattice = (pts, codes)

    def recall_place(self, percept):
        """Decode WHERE `percept` was seen from its place-cell template alone (grid-code argmax + centroid).

        Content-addressed: this does NOT return a stored (x, y) float -- it searches for the location whose
        grid code best matches the imprinted template, so it round-trips only because the grid code is
        spatially unique. Returns (x, y), or None if the percept was never bound.
        """
        rec = self._places.get(self._key(percept))
        if rec is None:
            return None
        self._ensure_lattice()
        pts, codes = self._lattice
        tmpl = rec["template"]
        corrs = [_pearson(codes[i], tmpl) for i in range(len(pts))]
        bi = max(range(len(pts)), key=lambda i: corrs[i])
        bx, by = pts[bi]
        thr = (
            0.6 * corrs[bi] if corrs[bi] > 0.0 else 0.0
        )  # centroid over the correlated node cluster
        sx = sy = sw = 0.0
        for i, (px, py) in enumerate(pts):
            w = corrs[i]
            if w > thr and math.hypot(px - bx, py - by) < 0.5:
                sx += w * px
                sy += w * py
                sw += w
        return (sx / sw, sy / sw) if sw > 0.0 else (bx, by)


def main():
    rng = random.Random(7)
    K = 6
    # ---- K landmarks at well-separated random locations inside the toy environment ----
    centers, percepts = [], []
    while len(centers) < K:
        p = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        if all(math.hypot(p[0] - c[0], p[1] - c[1]) > 0.6 for c in centers):
            centers.append(p)
            percepts.append(f"lm{len(centers)}")

    gpm = GridPlaceMap(seed=0)

    # ---- 1. DISTINCT grid codes: population-vector correlation is 1 at self, low between far places ----
    codes = [gpm.grid_code(c) for c in centers]
    off = [_pearson(codes[i], codes[j]) for i in range(K) for j in range(K) if i != j]
    mean_off = sum(off) / len(off)
    max_off = max(off)
    c0 = centers[0]
    near_corr = _pearson(gpm.grid_code(c0), gpm.grid_code((c0[0] + 0.03, c0[1] + 0.03)))

    # ---- 2. form place fields; place_at content-query is separable (on-field fires, off-field silent) ----
    for c, p in zip(centers, percepts):
        gpm.observe(c, p)
    id_correct = sum(1 for c, p in zip(centers, percepts) if gpm.place_at(c) == p)
    id_acc = id_correct / K
    # on-field vs off-field activation margin
    on_vals, off_vals = [], []
    for c, p in zip(centers, percepts):
        acts = gpm.place_activation(c)
        on_vals.append(acts[p])
        off_vals.append(sum(acts[q] for q in percepts if q != p) / (K - 1))
    on_mean = sum(on_vals) / K
    off_mean = sum(off_vals) / K
    silent = gpm.place_at((1.45, 1.45))  # a corner far from every landmark -> should be None

    # FAILING CONTROL: shuffle the percept<->place bindings; identify accuracy collapses to chance ~1/K
    perm = list(range(K))
    rng.shuffle(perm)
    shuf = GridPlaceMap(seed=0)
    for i in range(K):
        shuf.observe(centers[perm[i]], percepts[i])  # percept i bound at the WRONG center
    ctrl_correct = sum(1 for i in range(K) if shuf.place_at(centers[i]) == percepts[i])
    ctrl_acc = ctrl_correct / K

    # ---- 3. recall_place round-trips a bound percept to ~its location (decoded from the grid template) ----
    errs = [
        math.hypot(*(a - b for a, b in zip(gpm.recall_place(p), c)))
        for c, p in zip(centers, percepts)
    ]
    mean_err = sum(errs) / K
    # FAILING CONTROL: decode using a template that belongs to a DIFFERENT percept -> lands at chance
    ctrl_errs = []
    for i in range(K):
        wrong = gpm._places[percepts[(i + 1) % K]]["template"]
        gpm._ensure_lattice()
        pts, lat = gpm._lattice
        corrs = [_pearson(lat[k], wrong) for k in range(len(pts))]
        bx, by = pts[max(range(len(pts)), key=lambda k: corrs[k])]
        ctrl_errs.append(math.hypot(bx - centers[i][0], by - centers[i][1]))
    ctrl_mean_err = sum(ctrl_errs) / K

    # ---- 4. literal spiking: on-field place cell spikes vs off-field (silence) ----
    on_spk = gpm.place_spikes(centers[0], percepts[0])
    off_spk = gpm.place_spikes(centers[3], percepts[0])
    grid_on = sum(gpm.grid_spikes(centers[0]))

    # ---- 5. determinism: same seed + same observations -> identical code and decisions ----
    g2 = GridPlaceMap(seed=0)
    for c, p in zip(centers, percepts):
        g2.observe(c, p)
    det = (
        g2.grid_code(c0) == gpm.grid_code(c0)
        and [g2.place_at(c) for c in centers] == [gpm.place_at(c) for c in centers]
        and g2.recall_place(percepts[0]) == gpm.recall_place(percepts[0])
    )

    n_scales = 6
    print("=" * 72)
    print("GRID -> PLACE MAP -- spiking grid population binds a percept to a location")
    print("=" * 72)
    print(
        f"grid population: {gpm.n_grid} cells over {n_scales} scales; "
        f"env [-{gpm.extent:.1f},{gpm.extent:.1f}]^2"
    )
    print(
        f"distinct grid codes: mean far-corr {mean_off:.3f} (max {max_off:.3f}) "
        f"vs near-corr {near_corr:.3f} (self 1.000)"
    )
    print(
        f"place fields: on-field act {on_mean:.3f} vs off-field {off_mean:.3f} "
        f"(margin {on_mean - off_mean:.3f})"
    )
    print(
        f"place_at identify: TRUE binding {id_correct}/{K} ({100 * id_acc:.0f}%)  "
        f"vs  SHUFFLED control {ctrl_correct}/{K} ({100 * ctrl_acc:.0f}%)"
    )
    print(f"place_at at a far corner -> {silent!r} (silent)")
    print(
        f"recall_place round-trip: mean err {mean_err:.3f}  vs  wrong-template control {ctrl_mean_err:.3f}"
    )
    print(
        f"spikes: on-field place cell {on_spk}, off-field {off_spk} (window {gpm.spike_window}); "
        f"grid pop on-field {grid_on} spikes"
    )
    print(f"determinism (seed 0): identical code + decisions {det}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, un-riggable (controls must fail at chance) ----
    assert mean_off < 0.5 and near_corr > mean_off + 0.2, (
        f"grid codes must be distinct across far places and vary with position (mean_off {mean_off:.3f}, near {near_corr:.3f})"
    )
    assert id_acc == 1.0, (
        f"place_at must identify every bound landmark at its location ({id_correct}/{K})"
    )
    assert on_mean > off_mean + 0.4, (
        f"place fields must be separable: on-field >> off-field ({on_mean:.3f} vs {off_mean:.3f})"
    )
    assert silent is None, f"no place cell should fire far from every landmark (got {silent!r})"
    assert ctrl_acc <= 0.34, f"shuffled-binding control must be near chance ~1/K ({ctrl_acc:.2f})"
    assert mean_err < 0.4, (
        f"recall_place must round-trip to ~the bound location (mean err {mean_err:.3f})"
    )
    assert ctrl_mean_err > 3.0 * mean_err, (
        f"wrong-template decode must land far off (chance): ctrl {ctrl_mean_err:.3f} vs true {mean_err:.3f}"
    )
    assert on_spk > 3 * (off_spk + 1), (
        f"on-field place cell must spike, off-field ~silent ({on_spk} vs {off_spk})"
    )
    assert det, (
        "same seed + same observations must give identical code and decisions (deterministic)"
    )
    print(
        f"self-check OK: {gpm.n_grid} distinct grid codes bind {K} place fields; "
        f"place_at {100 * id_acc:.0f}% (control {100 * ctrl_acc:.0f}%); "
        f"recall err {mean_err:.3f} (control {ctrl_mean_err:.3f}); "
        f"on/off spikes {on_spk}/{off_spk}; deterministic"
    )


if __name__ == "__main__":
    main()
