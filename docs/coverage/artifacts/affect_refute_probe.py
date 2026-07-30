"""affect_refute_probe -- the adversarial pass on Affect.md's one band-61 row.

WHAT IS UNDER ATTACK. `tonic_stress_caution_bias` 61, on `affect_cortisol_caution_60.txt`:
`Cortisol.caution_bias()` pinned to 0.0 changed the selected action on 4/31 selections and the
`map_position` trajectory on 21/60 ticks, self-control 0/31 and 0/60, exit 0. That artifact
`--rerun`s clean at this HEAD, so it is not a hand-written transcript. This probe attacks the three
things a reproducing artifact still does not establish.

  1. ONSET STRUCTURE. `Executive.md:23-45` and `WholeBrain.md:88-94` both measured that
     `map_position` integrates its own motion, so the trajectory count is `N - onset + 1` and
     carries no information beyond WHEN the run first split. `wb_refute_onset_60.txt` records
     `drive` splitting at tick 39 and printing 22/60. The Affect artifact prints 21/60, i.e. onset
     40 -- one tick from a figure four other category files already share. So: how many independent
     onsets are in the 4/31 action flips, and where is the first one?

  2. MAGNITUDE vs SIGNAL. Pinning a term of a sum to 0.0 both removes the signal AND shifts the
     threshold by its mean. Two controls separate those: HALF (the signal's dynamics kept, magnitude
     halved) and CONST-MEAN (magnitude kept at the run-A mean, dynamics destroyed). If CONST-MEAN
     flips nothing, the variation is what charges and the row is safe. If CONST-MEAN flips the same
     actions as the full neuter, what was measured is a threshold offset of ~0.6*level, and the
     "affect signal" contributes nothing the same constant would not.

  3. ONE SEED. Every figure in the Affect file is seed 0. `Executive.md:241-243` names this as the
     single biggest caveat on its own 61s and did not run it. A charge that exists at one seed of a
     stochastic gate and nowhere else is not a faculty.

It also reports what actually drives the signal live: `Cortisol.step`'s call sites and its
`aversive` argument, since `caution_bias` is `0.6 * level` and `level` is a leaky integrator of
whatever that argument turns out to be.

METHOD is the one under attack, deliberately: `make_brain(seed=..., persist=False)`, one bound
method replaced for the whole run, behavioural observables (the action `AutonomousArm._select`
returned, the `map_position` trajectory), plus the `da` scalar `BasalGangliaGate.select` receives so
the perturbation's size is visible rather than assumed. Every run is paired with a second, un-
neutered A.

EXIT CODES describe whether the ATTACKED CLAIM SURVIVED, not whether the brain is charged:
  0  the claim survives this attack   (an action flip at a majority of seeds)
  1  REFUTED                          (the action flip is seed-0-specific)
  2  harness degenerate               (a self-control diverged, so no verdict)

  PYTHONPATH=src python docs/coverage/artifacts/affect_refute_probe.py onset 60
  PYTHONPATH=src python docs/coverage/artifacts/affect_refute_probe.py seeds 60 4
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.basal_ganglia.action_gating import BasalGangliaGate  # noqa: E402
from snn_data_lake.brainstem.autonomous_loop import AutonomousArm  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402
from snn_data_lake.neuromodulation.cortisol import Cortisol  # noqa: E402

_CB = Cortisol.caution_bias
_STEP = Cortisol.step
_GATE = BasalGangliaGate.select


def _restore():
    Cortisol.caution_bias = _CB
    Cortisol.step = _STEP
    BasalGangliaGate.select = _GATE


def run(seed, n, arm="wired", const=0.0, factor=1.0):
    """One run. arm: wired | zero (the auditor's neuter) | half | const | scale.

    Returns (actions, positions, trace) where trace carries the live signal census."""
    _restore()
    if arm == "zero":
        Cortisol.caution_bias = lambda self: 0.0
    elif arm == "half":
        Cortisol.caution_bias = lambda self: 0.5 * _CB(self)
    elif arm == "const":
        Cortisol.caution_bias = lambda self: const
    elif arm == "scale":
        Cortisol.caution_bias = lambda self: factor * _CB(self)

    tr = {"level": [], "cb": [], "steps": [], "da": []}

    def _step_spy(self, aversive=0.0):
        tr["steps"].append((sys._getframe(1).f_lineno, round(float(aversive), 4)))
        return _STEP(self, aversive)

    def _gate_spy(self, drive, da, stn=0.0):
        tr["da"].append(round(float(da), 6))
        return _GATE(self, drive, da, stn=stn)

    Cortisol.step = _step_spy
    BasalGangliaGate.select = _gate_spy
    chosen = []
    _sel = AutonomousArm._select

    def _spy(self, *a, **k):
        r = _sel(self, *a, **k)
        chosen.append(r)
        return r

    AutonomousArm._select = _spy
    try:
        b = make_brain(seed=seed, persist=False)
        pos = []
        for _ in range(n):
            b.tick()
            st = b.state()
            pos.append(tuple(st.get("map_position") or ()))
            cort = getattr(getattr(b.brain, "arm", None), "cort", None)
            tr["level"].append(round(float(cort.level), 6) if cort is not None else None)
            tr["cb"].append(round(float(_CB(cort)), 6) if cort is not None else None)
        b.close()
    finally:
        AutonomousArm._select = _sel
        _restore()
    return chosen, pos, tr


def _struct(a, b):
    """(flips, n, first, onsets, reconvergences) -- the shape of a divergence, not just its size."""
    m = min(len(a), len(b))
    d = [a[i] != b[i] for i in range(m)]
    first = next((i for i, x in enumerate(d) if x), None)
    onsets = sum(1 for i in range(m) if d[i] and not (i and d[i - 1]))
    recon = sum(1 for i in range(1, m) if d[i - 1] and not d[i])
    return sum(d), m, first, onsets, recon


def _line(tag, a, b, n):
    f, m, first, on, rc = _struct(a, b)
    print(
        f"{tag:<44}: {f}/{m} ticks  first {first}  onsets {on}  reconverge {rc}"
        f"{'' if m == n else f'  (n={n})'}"
    )
    return f, first, on


def onset(n):
    aA, pA, trA = run(0, n, "wired")
    aC, pC, _ = run(0, n, "wired")
    aZ, pZ, trZ = run(0, n, "zero")

    lv = [x for x in trA["level"] if x is not None]
    cb = [x for x in trA["cb"] if x is not None]
    sites = {}
    for ln, _av in trA["steps"]:
        sites[ln] = sites.get(ln, 0) + 1
    avs = sorted({av for _, av in trA["steps"]})
    firstnz = next((i for i, x in enumerate(lv) if x > 0.0), None)

    print(f"attacked claim                              : Affect.md:9 band 61, {n} ticks, seed 0")
    print(f"Cortisol.step call sites (autonomous_loop ln): {sites}")
    print(f"distinct `aversive` args it was given        : {avs[:8]}")
    print(f"cort.level first nonzero at tick             : {firstnz}   max {max(lv):.4f}")
    print(f"caution_bias live range                     : {min(cb):.4f}..{max(cb):.4f}")
    print(f"caution_bias distinct values / mean          : {len(set(cb))}  {sum(cb) / len(cb):.4f}")
    da_a, da_b = trA["da"], trZ["da"]
    print(f"gate `da` A range                           : {min(da_a):.4f}..{max(da_a):.4f}")
    dz, _, _, _, _ = _struct(da_a, da_b)
    print(f"gate `da` changed by the neuter              : {dz}/{min(len(da_a), len(da_b))} calls")
    print(f"selections made A / control / neutered       : {len(aA)} / {len(aC)} / {len(aZ)}")

    ca, _, _ = _line("SELF-CONTROL A vs A action", aA, aC, n)
    cp, _, _ = _line("SELF-CONTROL A vs A trajectory", pA, pC, n)
    fa, first_a, on_a = _line("NEUTER zero: action", aA, aZ, n)
    _line("NEUTER zero: trajectory", pA, pZ, n)

    mean = sum(cb) / len(cb)
    aH, pH, _ = run(0, n, "half")
    _line("CONTROL half (dynamics kept, magnitude/2)", aA, aH, n)
    _line("CONTROL half: trajectory", pA, pH, n)
    aK, pK, _ = run(0, n, "const", const=mean)
    _line(f"CONTROL const-mean {mean:.4f} (dynamics gone)", aA, aK, n)
    _line("CONTROL const-mean: trajectory", pA, pK, n)

    if ca or cp:
        print("NOT-ELIGIBLE: the self-control diverged -- no verdict")
        return 2
    if first_a is None:
        print("no action flip at all")
        return 1
    print(f"the flip at the first onset (A -> B)         : {aA[first_a]} -> {aZ[first_a]}")
    return 0


def seeds(n, k):
    """Per seed: the self-control, the signal census (so an inert seed can be told apart from an
    INELIGIBLE one -- neutering a signal that was 0.0 all run proves nothing either way), and the
    divergence. A seed is only evidence against the claim if the signal actually fired there."""
    rows = []
    for s in range(k):
        aA, pA, trA = run(s, n, "wired")
        aC, _, _ = run(s, n, "wired")
        aZ, pZ, trZ = run(s, n, "zero")
        cb = [x for x in trA["cb"] if x is not None]
        ctl, _, _, _, _ = _struct(aA, aC)
        fa, m, first, on, _ = _struct(aA, aZ)
        fp, np_, firstp, _, _ = _struct(pA, pZ)
        # The INTERNAL read, per seed: the `da` the BG gate is actually handed. A seed where no
        # action moves can still be a seed where the gate's input moved, which is band 41-60 rather
        # than band 21-40 -- so it is measured here instead of being assumed from seed 0.
        dd, dn, _, _, _ = _struct(trA["da"], trZ["da"])
        rows.append((s, ctl, fa, m, first, on, fp, np_, firstp, max(cb), len(set(cb)), dd, dn))
    print(f"multi-seed attack on Affect.md:9 (band 61)  : {k} seeds x {n} ticks")
    for s, ctl, fa, m, first, on, fp, np_, firstp, mx, dv, dd, dn in rows:
        print(
            f"seed {s}: caution_bias max {mx:.3f} ({dv} distinct) | self-control {ctl}/{m} ticks "
            f"| gate da {dd}/{dn} calls | action {fa}/{m} ticks first {first} onsets {on} "
            f"| trajectory {fp}/{np_} ticks first {firstp}"
        )
    elig = [r for r in rows if r[9] > 1e-9 and r[10] > 1]
    hot = [r for r in elig if r[2]]
    print(f"seeds where the gate input (da) moved       : {sum(1 for r in rows if r[11])}/{k}")
    print(f"seeds where the signal was eligible at all  : {len(elig)}/{k}")
    print(f"eligible seeds where the action changed     : {len(hot)}/{len(elig)}")
    if any(r[1] for r in rows):
        print("NOT-ELIGIBLE: a self-control diverged -- no verdict")
        return 2
    if elig and len(hot) * 2 > len(elig):
        print("CLAIM SURVIVES: the action change holds at a majority of ELIGIBLE seeds")
        return 0
    print("REFUTED: the action change does not hold at a majority of eligible seeds")
    return 1


def dose(n):
    """Dose-response at seed 0. A GRADED influence gives a curve; a knife-edge gives one step --
    any reduction at all produces the same trace as deleting the signal, which would mean the graded
    tonic stress LEVEL does no work and only its presence/absence does."""
    aA, pA, trA = run(0, n, "wired")
    aC, _, _ = run(0, n, "wired")
    ctl, _, _, _, _ = _struct(aA, aC)
    cb = [x for x in trA["cb"] if x is not None]
    print(f"dose-response on Affect.md:9, seed 0, {n} ticks")
    print(f"caution_bias in A: {min(cb):.4f}..{max(cb):.4f}, mean {sum(cb) / len(cb):.4f}")
    print(f"SELF-CONTROL A vs A action                  : {ctl}/{len(aA)} ticks")
    for f in (0.99, 0.9, 0.75, 0.5, 0.25, 0.0):
        aS, pS, _ = run(0, n, "scale", factor=f)
        fa, m, first, on, rc = _struct(aA, aS)
        fp, np_, firstp, _, _ = _struct(pA, pS)
        print(
            f"caution_bias x {f:<5}                        : action {fa}/{m} ticks first {first} "
            f"onsets {on} reconverge {rc} | trajectory {fp}/{np_} first {firstp}"
        )
    if ctl:
        print("NOT-ELIGIBLE: the self-control diverged -- no verdict")
        return 2
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "onset"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if mode == "onset":
        return onset(n)
    if mode == "seeds":
        return seeds(n, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    if mode == "dose":
        return dose(n)
    print("modes: onset <ticks> | seeds <ticks> <nseeds> | dose <ticks>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
