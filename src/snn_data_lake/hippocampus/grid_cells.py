"""
Entorhinal grid cells -- oscillatory-interference 2D path integration.

A grid cell fires on a triangular/HEXAGONAL lattice of locations as an animal crosses a plane
(Hafting, Fyhn, Molden, Moser & Moser, Nature 436:801, 2005). The oscillatory-interference model
(Burgess, Barry & O'Keefe, Hippocampus 17:801, 2007) builds that lattice from three velocity-controlled
oscillators whose interference is, in the reduced analytic form, a sum of three plane waves whose
wave-vectors point 60deg apart:

    r(p) = g( (1/3) * sum_k cos( b_k . (p - phase) ) ),   b_k at theta0 + {0, 60, 120} deg,
    |b_k| = 4*pi / (sqrt(3) * lambda),   g monotone: (0,1] rate from the [-1/2, 1] cosine sum.

The three 60deg-separated waves interfere to a hexagonal grid whose nearest-peak SPACING is exactly the
grid period lambda, oriented by theta0 and translated by the spatial phase (this module VERIFIES that from
the rate map, it does not assume it). Advancing the agent by a displacement v*dt accumulates an internal
2D position, so integrating a known velocity trajectory recovers the true displacement (dead-reckoning);
a small population at >=2 spacings mirrors the multi-scale entorhinal grid modules.

DISTINCTNESS (this is a different neural code from the repo's other attractor / hippocampal modules):
  - ring_attractor.py is a 1D ring continuous-attractor: ONE activity bump on a circle encodes a HEADING
    ANGLE and integrates a scalar ANGULAR velocity into a 1D angle. This module is 2D SPATIAL path
    integration whose firing is a HEXAGONAL lattice tiling a plane (position, not heading), built by
    oscillatory interference -- a 2D periodic code, not a single 1D bump.
  - NOT hippocampus.py (DG/CA3 pattern separation / completion of discrete memories).
  - NOT theta_gamma_code.py (phase-multiplexed working-memory buffer of ordered items).

A reduced/idealized interference model on synthetic trajectories -- analytic plane-wave rate
fields with literal velocity accumulation, NO real VCO spiking dynamics, no neural noise, no phase drift
or drift correction. A mathematical grid-cell twin (geometry + dead-reckoning), not biophysics, not cognition.

  python -m snn_data_lake.hippocampus.grid_cells        # (also run by run_selfchecks.py)
"""

import math
import random


class GridCell:
    """One grid module: spacing lambda, orientation theta0, spatial phase -> hexagonal analytic rate field."""

    def __init__(self, spacing=1.0, orientation=0.0, phase=None, gain=3.0, seed=0):
        self.spacing = float(spacing)
        self.orientation = float(orientation)
        self.gain = float(gain)
        if phase is None:  # seed -> reproducible random spatial phase
            rng = random.Random(seed)
            phase = (rng.uniform(0.0, self.spacing), rng.uniform(0.0, self.spacing))
        self.phase = (float(phase[0]), float(phase[1]))
        kmag = (
            4.0 * math.pi / (math.sqrt(3.0) * self.spacing)
        )  # |b_k| that makes peak spacing == lambda
        self.k = [
            (kmag * math.cos(self.orientation + a), kmag * math.sin(self.orientation + a))
            for a in (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)
        ]  # three 60deg-separated wave-vectors
        self._pos = [0.0, 0.0]  # accumulated agent position (path integration)

    def rate(self, x, y):
        """Firing rate at absolute position (x, y): interference of three plane waves through g (in (0,1])."""
        dx, dy = x - self.phase[0], y - self.phase[1]
        s = sum(math.cos(kx * dx + ky * dy) for kx, ky in self.k) / 3.0  # cosine sum in [-1/2, 1]
        return math.exp(self.gain * (s - 1.0))  # monotone: peaks (s=1) -> 1, troughs -> small

    def rate_at(self, pos):
        return self.rate(pos[0], pos[1])

    def reset(self, pos=(0.0, 0.0)):
        """Reset the accumulated agent position (start a fresh trajectory)."""
        self._pos = [float(pos[0]), float(pos[1])]

    def step(self, velocity, dt=1.0):
        """Advance the agent by velocity*dt (path integration); return the firing rate at the new position."""
        self._pos[0] += velocity[0] * dt
        self._pos[1] += velocity[1] * dt
        return self.rate(self._pos[0], self._pos[1])

    @property
    def position(self):
        return (self._pos[0], self._pos[1])

    def rate_map(self, x0, x1, y0, y1, n):
        """n-by-n rates over [x0,x1]x[y0,y1]; returns (xs, ys, grid) with grid[j][i] = rate(xs[i], ys[j])."""
        xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
        ys = [y0 + (y1 - y0) * j / (n - 1) for j in range(n)]
        grid = [[self.rate(x, y) for x in xs] for y in ys]
        return xs, ys, grid


class GridModulePopulation:
    """A small multi-scale population: grid modules at >=2 spacings (the real entorhinal organization)."""

    def __init__(self, spacings=(0.6, 1.0), orientation=0.0, phase=(0.0, 0.0), gain=3.0, seed=0):
        self.modules = [
            GridCell(spacing=s, orientation=orientation, phase=phase, gain=gain, seed=seed + i)
            for i, s in enumerate(spacings)
        ]

    def rates(self, x, y):
        """Population rate vector at (x, y) -- one entry per module."""
        return [m.rate(x, y) for m in self.modules]

    def step(self, velocity, dt=1.0):
        """Advance every module's agent by the same velocity; return the population rate at the new position."""
        for m in self.modules:
            m.step(velocity, dt)
        return self.rates(*self.modules[0].position)

    @property
    def position(self):
        return self.modules[0].position


def _find_peaks(xs, ys, grid, thresh, merge):
    """Local maxima of a rate map, sub-pixel refined by local centroid and clustered within `merge`."""
    n = len(grid)
    cands = []
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            v = grid[j][i]
            if v < thresh:
                continue
            if all(v >= grid[j + dj][i + di] for dj in (-1, 0, 1) for di in (-1, 0, 1)):
                w = 3  # centroid over a small window -> sub-pixel peak
                sx = sy = sw = 0.0
                for jj in range(max(0, j - w), min(n, j + w + 1)):
                    for ii in range(max(0, i - w), min(n, i + w + 1)):
                        wt = grid[jj][ii] - thresh
                        if wt > 0.0:
                            sx += wt * xs[ii]
                            sy += wt * ys[jj]
                            sw += wt
                cands.append((sx / sw, sy / sw))
    peaks = []  # cluster adjacent maxima into one peak each
    for px, py in cands:
        if not any(math.hypot(px - qx, py - qy) < merge for qx, qy in peaks):
            peaks.append((px, py))
    return peaks


def _hex_metrics(cell):
    """Measure the lattice around the cell's central peak: (spacing, angle_step_deg, base_angle_deg, center)."""
    half = 1.6 * cell.spacing
    px, py = cell.phase
    xs, ys, grid = cell.rate_map(px - half, px + half, py - half, py + half, 161)
    peaks = _find_peaks(xs, ys, grid, 0.5, 0.4 * cell.spacing)
    center = min(peaks, key=lambda p: math.hypot(p[0] - px, p[1] - py))
    others = sorted(
        (p for p in peaks if p != center),
        key=lambda p: math.hypot(p[0] - center[0], p[1] - center[1]),
    )[:6]
    dists = [math.hypot(p[0] - center[0], p[1] - center[1]) for p in others]
    angs = sorted(
        math.degrees(math.atan2(p[1] - center[1], p[0] - center[0])) % 360.0 for p in others
    )
    steps = [(angs[(i + 1) % 6] - angs[i]) % 360.0 for i in range(6)]
    return sum(dists) / 6.0, sum(steps) / 6.0, min(a % 60.0 for a in angs), center


def main():
    # ---- 1. hexagonal periodicity: measure the lattice straight off the rate map ----
    cell = GridCell(spacing=1.0, orientation=0.0, phase=(0.0, 0.0))
    spacing_m, angle_step, base0, center0 = _hex_metrics(cell)

    # ---- 2. grid geometry from parameters: spacing / orientation / phase ----
    small, big = GridCell(spacing=0.6, phase=(0.0, 0.0)), GridCell(spacing=1.0, phase=(0.0, 0.0))
    small_m, _, _, _ = _hex_metrics(small)
    big_m, _, _, _ = _hex_metrics(big)
    rot = GridCell(spacing=1.0, orientation=math.radians(15.0), phase=(0.0, 0.0))
    _, _, base_rot, _ = _hex_metrics(rot)
    rot_deg = (base_rot - base0) % 60.0
    shifted = GridCell(spacing=1.0, orientation=0.0, phase=(0.2, 0.0))
    shift_m, _, _, center_s = _hex_metrics(shifted)
    peak_shift = math.hypot(center_s[0] - center0[0], center_s[1] - center0[1])

    # ---- 3. path integration: straight leg lands at v*t; a closed loop returns to the start ----
    leg = GridCell(spacing=1.0, phase=(0.0, 0.0))
    leg.reset((0.0, 0.0))
    for _ in range(10):
        leg.step((0.3, 0.1), dt=0.5)  # -> displacement (1.5, 0.5)
    straight_err = math.hypot(leg.position[0] - 1.5, leg.position[1] - 0.5)
    loop = GridCell(spacing=1.0, phase=(0.0, 0.0))
    loop.reset((0.0, 0.0))
    for vel in (
        (0.4, 0.0),
        (0.0, 0.4),
        (-0.4, 0.0),
        (0.0, -0.4),
    ):  # square loop, net displacement 0
        for _ in range(5):
            loop.step(vel, dt=0.5)
    loop_err = math.hypot(*loop.position)

    # ---- 4. determinism: two fresh cells (same params/seed) -> identical map and trajectory ----
    a, b = GridCell(spacing=0.8, seed=7), GridCell(spacing=0.8, seed=7)
    same_map = a.rate_map(-1.0, 1.0, -1.0, 1.0, 40) == b.rate_map(-1.0, 1.0, -1.0, 1.0, 40)
    for _ in range(8):
        a.step((0.11, -0.07), dt=0.3)
        b.step((0.11, -0.07), dt=0.3)
    same_traj = a.position == b.position

    pop = GridModulePopulation(spacings=(0.6, 1.0))  # multi-scale readout at the origin
    pop_rates = pop.rates(0.0, 0.0)

    print("=" * 68)
    print("GRID CELLS -- oscillatory-interference 2D path integration")
    print("=" * 68)
    print(
        f"hexagonal lattice (lambda=1.00): 6 nearest peaks mean dist {spacing_m:.3f} "
        f"(set 1.00), angle step {angle_step:.1f} deg"
    )
    print(
        f"geometry: spacing 0.60 -> measured {small_m:.3f} ; spacing 1.00 -> measured {big_m:.3f}"
    )
    print(
        f"orientation +15deg rotates lattice by {rot_deg:.1f} deg ; "
        f"phase (0.2,0) translates peak by {peak_shift:.3f} (period {shift_m:.3f})"
    )
    print(
        f"path integration: straight leg -> pos ({leg.position[0]:.3f},{leg.position[1]:.3f}) "
        f"err {straight_err:.1e} ; closed loop return err {loop_err:.1e}"
    )
    print(f"determinism (seed 7): identical rate map {same_map} and trajectory {same_traj}")
    print(
        f"population rates at origin (spacings 0.6, 1.0): [{pop_rates[0]:.3f}, {pop_rates[1]:.3f}]"
    )
    print("=" * 68)

    # ---- self-check: deterministic asserts ----
    assert abs(spacing_m - 1.0) < 0.05, f"measured grid spacing must match lambda ({spacing_m:.3f})"
    assert abs(angle_step - 60.0) < 3.0, (
        f"6 nearest peaks must sit 60deg apart -> hexagonal ({angle_step:.2f})"
    )
    assert abs(small_m - 0.6) < 0.05 and abs(big_m - 1.0) < 0.05 and big_m > small_m, (
        f"different spacing -> different measured period ({small_m:.3f}, {big_m:.3f})"
    )
    assert abs(rot_deg - 15.0) < 3.0, (
        f"orientation must rotate the lattice by ~15deg ({rot_deg:.2f})"
    )
    assert abs(peak_shift - 0.2) < 0.03 and abs(shift_m - 1.0) < 0.05, (
        f"phase must translate peaks (~0.2) without changing period ({peak_shift:.3f}, {shift_m:.3f})"
    )
    assert straight_err < 1e-9, f"straight leg must land at v*t ({straight_err:.1e})"
    assert loop_err < 1e-9, f"closed velocity loop must return to the start ({loop_err:.1e})"
    assert same_map and same_traj, (
        "same params/seed must give identical rate map and trajectory (deterministic)"
    )
    print(
        f"self-check OK: hexagonal lattice (spacing {spacing_m:.3f}, step {angle_step:.1f} deg); "
        f"spacing/orientation/phase set the geometry; path integration exact "
        f"(loop err {loop_err:.1e}); deterministic"
    )


if __name__ == "__main__":
    main()
