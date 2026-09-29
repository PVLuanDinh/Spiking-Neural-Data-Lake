"""Ignition scorer + baseline (specs/022 T033) -- P8 invention #6's ignition instrument,
anchored to docs/INVENTION_REGISTER.md row 6 ("dual-derived gate, actuator-whole,
published-band checks -- 100% ignition = saturated detector").

WHAT IT MEASURES. Whether the declared subject -- the re-entry stage of the T030 workspace
composite (thalamus/reentrant_workspace.ReentrantWorkspace), theta commanded through the
T032 actuator -- exhibits IGNITION: an all-or-none transition in sustained firing as drive
crosses the rheobase. The scorer never asserts a marker; it derives the transition TWICE by
independent routes and refuses unless they agree:

  * ANTIMODE route: the temporal antimode (the last-spike histogram's trough, re-derived
    below) classifies each trial as sustained-past-the-trough or not; the transition is
    the mean 0.5-crossing of the per-drive classification fractions.
  * LOGISTIC route: a logistic fit to the SAME classification against drive; the
    transition is the inflection d0, with d0 candidates on the sweep's own drive lattice
    so the two routes share their quantization.

  The dual gate: |antimode-route - logistic-route| <= one drive-grid step (the
  instrument's own resolution -- a derived tolerance, not a picked one). The kernel's own
  winner verdicts are checked SEPARATELY against this classification at every unanimously
  settled drive -- a reporter whose verdicts flip at a foreign drive refuses as DISAGREE.

THE WINDOW IS RE-DERIVED, NEVER TRUSTED. The kernel splits 'late' at _T // 2; this scorer
rebuilds the LAST-SPIKE histogram from BOTH pools' rasters across a seeded trial ensemble
(the silenced loser's early death vs the winner's end-of-settle tail -- this kernel has no
stimulus-locked onset volley, so the two-peak structure is temporal-TERMINAL), requires
the early mode to carry mass from at least two DISTINCT trials, finds the trough, and
opens its late window past it. The kernel's half must land inside the derived trough's
basin or the scorer reports the disagreement -- the published-window check, run every time.

FIVE FAILURE MODES, EACH A DISTINCT VERDICT, EACH WITH A CONSTRUCTED RED IN main():
  SATURATED  -- every trial ignites (a detector that cannot say no; register row 6's own
                warning). SILENT -- none does. UNIMODAL -- no two-mode temporal structure
                (no trough, or its early mass is a single-trial coincidence): nothing
                all-or-none to score.
  DISAGREE   -- the two routes name different transitions: the scorer refuses rather than
                averaging. CONTROL_FAIL -- an external control came back wrong: the
                shuffled-label refit kept its separation (the fit is reading structure that
                survives shuffling = rigged), or the recurrence ablation changed nothing
                (whatever was measured does not depend on the loop it claims to measure).

CONTROLS ARE EXTERNAL TO THE METHOD (P8): a TRUE permutation test -- twenty seeded label
shuffles, each refit from scratch on a fit family that can express flatness, with the real
fit required to beat EVERY permutation on per-point logloss (one-sided exceedance,
p < 1/21) -- and recurrence-ablation: the two-tick sustain protocol with reentry ON vs OFF
on the REAL subject, requiring the sustain signature to vanish OFF. Ignition-as-attractor
must depend on the re-entrant loop or the scorer is measuring something else.

BASELINE: `--record` writes tools/ignition_baseline.json (the lake_append_baseline
pattern); the self-check, when the file exists, checks the current measurement against the
recorded band -- drift detection for T039's exit certification.

DISCLOSURE (P8 sentience clause): a green verdict here is evidence about a MECHANISM -- an
all-or-none dynamical transition in a spiking competition -- never a verdict about
experience. No phenomenal claim is made or implied.

# distinct-from: thalamus/reentrant_workspace.py -- the SUBJECT under measurement; the
#   scorer drives it through its public API and re-reads its rasters, it re-implements
#   nothing of it. distinct-from: thalamus/gwt_deep.py -- an EXEMPT phase-binding
#   benchmark, no scoring. distinct-from: instruments/port_charge_probe.py -- charge A/B
#   over recorded frames; this scores a dynamical transition. distinct-from:
#   cortex/access_consciousness.py -- claims over the LIVE bus; this is an offline
#   instrument on scratch subjects. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.instruments.ignition_instrument
"""

import json
import math
import random
from pathlib import Path

from snn_data_lake.thalamus.reentrant_workspace import ReentrantWorkspace

_BASELINE = Path(__file__).resolve().parents[3] / "tools" / "ignition_baseline.json"

# FR-002c gradient-claim declaration (gates/gradient_claim_gate reads this without
# importing): the one fit in this module is an ANALYSIS fit, not a learning rule.
GRADIENT_CLAIM = {
    "_logistic_fit": (
        "NOT_A_RULE",
        "offline grid-search MLE inside a measurement instrument: it fits a logistic to a "
        "completed sweep's classification labels to LOCALIZE a transition, updates no "
        "weights, touches no substrate, and its output is a report field. Nothing is "
        "propagated anywhere, so it owes no cosine",
    ),
}

# Verdicts -- the five failure modes plus the one success. An unknown state has no name
# here on purpose: anything the scorer cannot classify lands on the nearest refusal.
OK = "OK"
SATURATED = "SATURATED"
SILENT = "SILENT"
UNIMODAL = "UNIMODAL"
DISAGREE = "DISAGREE"
CONTROL_FAIL = "CONTROL_FAIL"


def _drive_grid(subject_cls):
    """The sweep grid, derived from the subject's own constants: spans the admissible drive
    domain (0, _COAL_CAP] at a resolution of half the coalition bonus -- the smallest drive
    difference the subject's own mechanics treat as meaningful. No picked density."""
    cap = float(subject_cls._COAL_CAP)
    step = float(subject_cls._COAL_BONUS) / 2.0
    n = int(cap / step)
    return [round((i + 1) * step, 6) for i in range(n)], step


# distinct-from: reasoning/test_prototype.py -- codebook recall trials, no workspace. Read.
def _trial(theta, drive, seed, reentry=True, subject_cls=ReentrantWorkspace):
    """ONE trial: a fresh subject (independence across trials -- sustain and hysteresis are
    cross-tick state a shared instance would leak between drives), theta commanded through
    THE ACTUATOR with the sweep hold (this call is the T032 channel's commanding client --
    the wiring disposition in set_base_rheobase's docstring discharges here).

    THE PROBE COMPETES AGAINST A SUPRA-THETA DISTRACTOR, deliberately, and the reference
    drive is derived: `ref = theta + 2 * grid_step` (two resolution steps clear of the
    rheobase and its jitter). Two measured facts forced this shape. (1) A lone pool under
    constant drive has NO temporal structure to score -- sub-theta trials fire ZERO spikes
    (the rheobase is a hard subthreshold gate; there is no fire-then-fade), supra trials
    fire wall-to-wall, so the transition is a bare count gap and any window is vacuously
    correct. (2) A distractor AT theta sits on the rheobase boundary and never reliably
    fires, so no losing raster exists either. Only a SUPRA-theta loser produces the
    ignition literature's first peak: it fires early, the winner's inhibition silences it,
    and its early-then-dead raster against the winner's wall-to-wall tail is the two-peak
    structure whose trough IS the derived window. `ignited` is 'the PROBE took the
    workspace'; the transition the routes localize is therefore the COMPETITION POINT
    (ref), reported as such -- theta itself gates whether any pool fires at all."""
    ws = subject_cls(seed=seed, reentry=reentry)
    ws.set_base_rheobase(theta, provenance="ignition_instrument:sweep", hold_bias=True)
    grid, gstep = _drive_grid(subject_cls)
    ref = min(theta + 2 * gstep, float(subject_cls._COAL_CAP))
    ws.submit("probe_src", "probe", drive)
    ws.submit("distractor_src", "distractor", ref)
    out = ws.step()
    assert ws.ignition_threshold == theta, (
        "READ-BACK FAILED inside the scorer's own sweep -- the actuator hold did not hold"
    )
    # BOTH pools' rasters, recorded explicitly and exactly once each. The first draft
    # selected 'the loser' with a conditional that, whenever the probe LOST, returned the
    # probe's raster twice and dropped the winning distractor entirely -- the review
    # measured the resulting 'early mode' to be one trial double-counted. No selection
    # here: the window derivation reads both terminal times and dedups nothing.
    return {
        "drive": drive,
        "ref": ref,
        "ignited": bool(out["winner"] == "probe"),
        "raster": list(ws._last_raster.get("probe", ())),
        "other_raster": list(ws._last_raster.get("distractor", ())),
        "n_steps": ws._T,
    }


_ENSEMBLE = 5  # seeds per drive point -- see sweep(); the early mode needs independent mass


# distinct-from: engine/brain.py -- its 'ensemble' is the wake tick's model committee;
#   this is a seeded TRIAL ensemble for a measurement sweep. Read.
def sweep(theta, seed=0, subject_cls=ReentrantWorkspace, reentry=True):
    """The drive sweep at commanded theta: an ENSEMBLE of `_ENSEMBLE` seeded trials per
    grid point, not one. The kernel is deterministic per (seed, drive), so a single-seed
    sweep produces the die-early regime -- the transition's actual signature -- on at most
    ONE hairline boundary trial, and the review measured the whole two-mode structure
    resting on that lone trial: 7 of 8 seeds and most thetas read UNIMODAL, an OK that was
    a seed-0 accident. The per-instance jitter moves the boundary a little per seed, so an
    ensemble crosses it several independent times and the early mode carries real mass
    from DISTINCT trials. Trial seeds derive from the instrument seed deterministically
    (no wall clock, no global RNG). Returns (trials, grid_step)."""
    grid, step = _drive_grid(subject_cls)
    # BOUNDARY-CONCENTRATED SAMPLING (staircase psychometrics, and the fix the territory
    # forced): the die-early regime -- the transition's actual temporal signature -- lives
    # only within the symmetry-break jitter of the competition point, and the coarse grid
    # (step 0.03 > _JITTER 0.02) contains AT MOST one point in that band, so a coarse-only
    # sweep sees zero or one death per run (measured: 1 event at theta 0.55, 0 at 0.70,
    # across 200 trials each). A fine local ensemble around ref at _JITTER/2 spacing
    # (spanning +-2 jitter -- the whole stochastic band, derived from the subject's own
    # jitter constant) crosses the boundary several independent times at ANY theta.
    ref = min(theta + 2 * step, float(subject_cls._COAL_CAP))
    fine = float(subject_cls._JITTER) / 2.0
    drives = list(grid) + [round(ref + k * fine, 6) for k in range(-4, 5)]
    trials = []
    for i, d in enumerate(drives):
        if not (0.0 < d <= float(subject_cls._COAL_CAP)):
            continue
        for e in range(_ENSEMBLE):
            trials.append(
                _trial(
                    theta,
                    d,
                    seed=seed * 10007 + i * 101 + e,
                    reentry=reentry,
                    subject_cls=subject_cls,
                )
            )
    return trials, step


# distinct-from: sensorimotor/signal_loop.py -- its windows are sensor segments, no PSTH.
# distinct-from: engine/autonomous_loop.py -- salience deviance, no spike raster. Read.
def derive_window(trials):
    """Re-derive the late window from the DATA, via the distribution the ignition
    signature actually lives in on THIS kernel: LAST-SPIKE TIMES. Under constant drive the
    subject has no stimulus-locked early volley -- every sufficiently driven trial spikes
    from the first micro-steps -- so the two-peak structure is temporal-TERMINAL, not
    temporal-onset: sub-theta trials fire briefly and DIE (leak + inhibition win), supra
    trials fire to the last micro-step. The last-spike histogram is therefore bimodal
    (die-early mode vs end-of-settle mode), and the derived late window opens at the END
    of the trough between them: the time by which every dying trial has died. (The first
    draft aggregated a plain PSTH over ignited trials and found its minimum on the CHARGING
    RAMP at t~1 -- the measured failure that forced this derivation.)

    Returns (window_start, last_spike_hist, trough_span), or (None, hist, None) when no
    two-mode structure exists -- the UNIMODAL refusal."""
    if not trials:
        return None, [], None
    n_steps = trials[0]["n_steps"]
    hist = [0] * n_steps
    terminal = []  # (bin, trial_idx) pairs, for the post-hoc distinct-mass guard
    for idx, t in enumerate(trials):
        # BOTH pools contribute their terminal time, each exactly once: a silenced
        # early death populates the first mode, a wall-to-wall survivor the second.
        for key in ("raster", "other_raster"):
            r = t.get(key)
            if r:
                hist[r[-1]] += 1
                terminal.append((r[-1], idx))
    occ = [i for i, c in enumerate(hist) if c > 0]
    if not occ:
        return None, hist, None
    first, last = occ[0], occ[-1]
    if last - first < 4:  # all pools terminate together: one mode, nothing to separate
        return None, hist, None
    interior = hist[first + 1 : last]
    lo = min(interior)
    if lo >= min(hist[first], hist[last]):  # flanks do not FALL into the interior
        return None, hist, None
    best_len, best_start, cur_len, cur_start = 0, None, 0, None
    for i in range(first + 1, last):
        if hist[i] == lo:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_start is None:
        return None, hist, None
    trough = (best_start, best_start + best_len - 1)
    # TWO-PEAK MEANS SEPARATED: the trough must span the MAJORITY of the occupied axis
    # between the modes. A graded responder's terminal times spread continuously with
    # local wiggles, and a width-1 dip is a wiggle, not a separation (measured: the
    # graded impostor's spread histogram offered a 1-bin 'trough' the local-minimum
    # finder accepted). Derived, not picked: separation > half the occupied span.
    if (trough[1] - trough[0] + 1) * 2 <= (last - first):
        return None, hist, None
    # THE SINGLE-COINCIDENCE GUARD, non-circular: AFTER the trough is located from shape
    # alone, the mass strictly BEFORE it must come from at least TWO DISTINCT trials. The
    # first draft's 'two-peak structure' was one boundary trial counted twice, and a
    # looser draft of this guard counted winner-edge terminals (t = 57/58) as 'early' --
    # both die here: only pre-trough deaths count, and one trial is a coincidence, not a
    # mode.
    early_trials = {idx for b, idx in terminal if b < trough[0]}
    if len(early_trials) < 2:
        return None, hist, None
    return trough[1] + 1, hist, trough


def _logistic_fit(points):
    """Grid-search MLE for P(ignite|d) = 1/(1+exp(-k(d-d0))). Returns (d0, k, logloss).
    Grid search rather than Newton on purpose: deterministic, derivative-free, and the
    surface's flatness under shuffled labels is exactly what the control reads."""
    drives = [d for d, _y in points]
    lattice = sorted(set(drives))
    best = None
    # d0 candidates live on the SWEEP'S OWN drive lattice plus midpoints -- the same
    # lattice the crossing estimator quantizes on, so the two routes' quantization is
    # SHARED and cancels in the dual gate instead of masquerading as disagreement (an
    # earlier uniform-97-point grid put d0 on an unrelated lattice and contributed a
    # systematic ~half-step bias).
    d0_grid = sorted(set(lattice) | {(a + b) / 2.0 for a, b in zip(lattice, lattice[1:])})
    # k from NEAR-FLAT upward. The first draft floored k at 2.0, which FORBADE flat fits:
    # every permutation of the labels was forced onto the same pinned sigmoid, so the
    # permutation null had zero variance and the shuffle criterion compared constants
    # (review, measured: 20 shuffles -> 1 distinct fit). A null that cannot express
    # 'no structure' is not a null.
    k_grid = [0.25 * (1.5**i) for i in range(18)]  # 0.25 .. ~370, geometric
    for d0 in d0_grid:
        for k in k_grid:
            ll = 0.0
            for d, y in points:
                p = 1.0 / (1.0 + math.exp(-k * (d - d0)))
                p = min(1.0 - 1e-9, max(1e-9, p))
                ll -= y * math.log(p) + (1 - y) * math.log(1.0 - p)
            if best is None or ll < best[2]:
                best = (d0, k, ll)
    return best


def _permutation_null(points, real_fit, seed, n_perm=20):
    """A TRUE permutation test on the logistic fit's quality. The statistic is per-point
    logloss -- a quantity that DEPENDS ON THE LABELS (the first draft's mean|p-0.5|
    ignored them entirely, reducing the 'shuffle control' to a base-rate readout that
    false-failed honest subjects at skewed base rates and had zero variance at balanced
    ones -- both measured by the review). Each seeded permutation refits from scratch;
    with flat fits now expressible (see k_grid), a shuffled label set settles near the
    base-rate entropy while genuine structure fits far below it. Returns (real_ll,
    null_lls); the criterion -- real strictly below EVERY permutation -- is a one-sided
    p < 1/(n_perm+1) exceedance test, derived, not picked."""
    real_ll = real_fit[2] / len(points)
    rng = random.Random(seed)
    ys = [y for _d, y in points]
    null_lls = []
    for _ in range(n_perm):
        rng.shuffle(ys)
        shuffled = [(d, y) for (d, _), y in zip(points, ys)]
        null_lls.append(_logistic_fit(shuffled)[2] / len(points))
    return real_ll, null_lls


def _sustain_pair(theta, seed, subject_cls=ReentrantWorkspace):
    """The recurrence-ablation protocol: ignite at supra drive on tick 1, probe tick 2 at a
    SUB-theta drive. With re-entry ON the incumbent's recharge sustains ignition; OFF the
    workspace goes dark. Returns (on, off) ignition booleans for tick 2."""
    out = []
    for reentry in (True, False):
        ws = subject_cls(seed=seed, reentry=reentry)
        ws.set_base_rheobase(theta, provenance="ignition_instrument:ablation", hold_bias=True)
        ws.submit("probe_src", "probe", min(theta + 0.3, float(subject_cls._COAL_CAP)))
        ws.step()
        ws.submit("probe_src", "probe", max(theta - 0.25, 0.05))
        r2 = ws.step()
        out.append(bool(r2["ignited"]))
    return out[0], out[1]


def score(theta=0.55, seed=0, subject_cls=ReentrantWorkspace):
    """Run the whole instrument at commanded theta. Returns the verdict + every derived
    quantity, controls included -- one call, one report, nothing asserted that a failure
    mode could not redden."""
    trials, grid_step = sweep(theta, seed=seed, subject_cls=subject_cls)
    frac = sum(1 for t in trials if t["ignited"]) / len(trials)
    report = {
        "theta": theta,
        "ref": trials[0]["ref"],  # the derived competition point both routes localize
        "fraction_ignited": round(frac, 4),
        "grid_step": grid_step,
        "n_trials": len(trials),
    }
    if frac >= 1.0:
        report["verdict"] = SATURATED
        return report
    if frac <= 0.0:
        report["verdict"] = SILENT
        return report

    window_start, psth, trough = derive_window(trials)
    report["window_start"] = window_start
    report["trough"] = trough
    report["kernel_half"] = trials[0]["n_steps"] // 2
    if window_start is None:
        report["verdict"] = UNIMODAL
        return report
    # the published-window check: the kernel's own half must sit at or before the derived
    # late-window opening and past the trough's start -- else the kernel's 'late' mislabels
    report["kernel_half_in_basin"] = bool(trough[0] <= report["kernel_half"] <= window_start)

    # ONE label set, TWO estimators. The physics label is 'the probe SUSTAINED PAST THE
    # TROUGH' -- the temporal antimode's own classification, nothing more. An earlier
    # layer put a SECOND antimode over in-window spike counts and classified on that; it
    # invented a third ignition definition which unanimously excluded the partial-victory
    # band the kernel unanimously includes (measured at drives 0.60-0.61: kernel frac 1.0,
    # count-classification 0.0) -- P2 killed the extra mechanism: the trough already
    # separates dying from sustaining, and 'fired past it' IS the separation. The dual
    # gate below then tests ESTIMATOR consistency on this one dataset (temporal-antimode
    # crossing vs logistic inflection), which is what 'must agree at instrument
    # resolution' can honestly mean. The route aggregates PER DRIVE over the ensemble
    # (per-trial flip detection lands inside boundary noise -- measured a grid step of
    # bias).
    phys = [1 if (t["raster"] and t["raster"][-1] >= window_start) else 0 for t in trials]
    by_drive = {}
    for t, y in zip(trials, phys):
        by_drive.setdefault(t["drive"], []).append(y)
    drives_sorted = sorted(by_drive)
    fracs = [sum(by_drive[d]) / len(by_drive[d]) for d in drives_sorted]
    cross = [i for i in range(1, len(fracs)) if (fracs[i - 1] < 0.5) != (fracs[i] < 0.5)]
    # MEAN of the crossing midpoints, not the last: boundary-ensemble noise produces
    # spurious re-crossings scattered symmetrically about the true transition; the last
    # one alone biased the estimate a fine-step or two high.
    theta_anti = (
        sum((drives_sorted[i] + drives_sorted[i - 1]) / 2.0 for i in cross) / len(cross)
        if cross
        else None
    )
    report["theta_hat_antimode"] = round(theta_anti, 6) if theta_anti is not None else None

    points = [(t["drive"], y) for t, y in zip(trials, phys)]
    fit = _logistic_fit(points)
    report["theta_hat_logistic"] = round(fit[0], 6)
    report["logistic_k"] = round(fit[1], 3)

    # THE DUAL GATE at ONE grid step -- the instrument's resolution, as documented.
    if theta_anti is None or abs(theta_anti - fit[0]) > grid_step:
        report["verdict"] = DISAGREE
        return report

    # KERNEL-vs-PHYSICS: at every drive where BOTH label sets are SETTLED (the ensemble
    # unanimously 0 or unanimously 1 under each), the kernel's winner verdict and the
    # sustain classification must agree. Fray zones -- where either label set is mixed
    # across the ensemble -- drop out by construction: near the boundary the kernel
    # declares winners on partial volleys the sustain window has not certified (measured:
    # a 12-spike probe 'wins' two grid steps below ref while sustaining nothing), and
    # that is boundary physics, not lying. A reporter whose verdict flips at a foreign
    # drive (the lying-label rig) is unanimous-and-wrong there and refuses. Two earlier
    # region definitions -- a fixed 2x-jitter band, then the physics fracs' own support --
    # each false-DISAGREEd a genuine theta; both are recorded here as the derivation.
    kern_by_drive = {}
    for t in trials:
        kern_by_drive.setdefault(t["drive"], []).append(1 if t["ignited"] else 0)
    mismatch = 0
    for d in drives_sorted:
        kf = sum(kern_by_drive[d]) / len(kern_by_drive[d])
        pf = sum(by_drive[d]) / len(by_drive[d])
        if kf in (0.0, 1.0) and pf in (0.0, 1.0) and kf != pf:
            mismatch += 1
    report["kernel_phys_mismatch"] = mismatch
    if mismatch > 0:
        report["verdict"] = DISAGREE
        return report

    # -- external controls, run every score() ---------------------------------------------
    real_ll, null_lls = _permutation_null(points, fit, seed=seed + 1)
    report["logloss_real"] = round(real_ll, 4)
    report["logloss_null_min"] = round(min(null_lls), 4)
    on, off = _sustain_pair(theta, seed, subject_cls=subject_cls)
    report["sustain_reentry_on"] = on
    report["sustain_reentry_off"] = off
    # the real fit must beat EVERY permutation (one-sided exceedance, p < 1/21), and the
    # ablation must remove the sustain signature. Either surviving means the instrument,
    # not the subject, carries the structure.
    if real_ll >= min(null_lls) or not (on and not off):
        report["verdict"] = CONTROL_FAIL
        return report

    report["verdict"] = OK
    return report


def record_baseline(path=_BASELINE):
    """Write the ignition baseline (the lake_append_baseline pattern): the T039 exit
    certification and later self-checks compare against this band."""
    rep = score()
    payload = {
        "method": "ignition_instrument.score(theta=0.55, seed=0): seeded 5-trial ensemble "
        "per drive point on fresh ReentrantWorkspace subjects, theta via the T032 actuator "
        "under sweep hold; window re-derived from BOTH pools' last-spike times (>=2 "
        "distinct early trials required); antimode + logistic dual gate at ONE grid step; "
        "permutation-null (real logloss < all 20 shuffles) + recurrence-ablation controls "
        "in the same call",
        "theta": rep["theta"],
        "theta_hat_antimode": rep["theta_hat_antimode"],
        "theta_hat_logistic": rep["theta_hat_logistic"],
        "window_start": rep["window_start"],
        "kernel_half_in_basin": rep["kernel_half_in_basin"],
        "fraction_ignited": rep["fraction_ignited"],
        "grid_step": rep["grid_step"],
        "verdict": rep["verdict"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class _LinearImpostor(ReentrantWorkspace):
    """UNIMODAL red: a graded responder -- spikes spread uniformly, count proportional to
    drive, no threshold, no attractor. A scorer that finds 'ignition' here is reading its
    own machinery. Constructed INPUT to the scorer, never part of the method (P8)."""

    def _simulate(self, drives):
        # Fires from t=0 and STOPS when the drive budget is spent: terminal time is
        # PROPORTIONAL to drive, so last-spike times spread continuously across the
        # settle -- no trough anywhere. (A first construction spread n spikes uniformly
        # over the whole settle, which pinned every n>=2 terminal at T-1 and every n=1 at
        # 0 -- an accidental two-mode artifact of the impostor, not gradedness.)
        total, late = {}, {}
        raster = {}
        for k, d in drives.items():
            n = max(1, min(self._T, int(d * self._T / 1.2)))
            times = list(range(n))
            raster[k] = times
            total[k] = n
            late[k] = sum(1 for t in times if t >= self._T // 2)
        self._last_raster = raster
        return total, late


def main():
    print("=" * 74)
    print("IGNITION INSTRUMENT -- dual-derived transition scorer over the re-entry stage")
    print("=" * 74)

    rep = score()
    print(
        f"genuine subject @ theta 0.55: verdict={rep['verdict']} "
        f"antimode-route={rep.get('theta_hat_antimode')} logistic={rep.get('theta_hat_logistic')} "
        f"window={rep.get('window_start')} (kernel half {rep.get('kernel_half')} in basin: "
        f"{rep.get('kernel_half_in_basin')}) ignited {rep['fraction_ignited']:.2f} "
        f"logloss {rep.get('logloss_real')} vs null-min {rep.get('logloss_null_min')} "
        f"sustain on/off {rep.get('sustain_reentry_on')}/{rep.get('sustain_reentry_off')}"
    )
    assert rep["verdict"] == OK, f"the genuine subject must score OK, got {rep['verdict']}"
    # LOCALIZATION at ONE grid step, with the discrimination made STRUCTURAL: theta itself
    # must sit OUTSIDE the acceptance band, so 'localizes ref' can never be satisfied by a
    # route that actually localizes theta. (The review measured the first draft's band
    # reaching theta exactly -- zero margin, a float artifact away from covering it.)
    assert rep["ref"] - rep["theta"] > rep["grid_step"], (
        "the ref-vs-theta discrimination must be structural: theta inside the acceptance "
        "band would make localization unfalsifiable against the trivial estimator"
    )
    assert abs(rep["theta_hat_antimode"] - rep["ref"]) <= rep["grid_step"] and (
        abs(rep["theta_hat_logistic"] - rep["ref"]) <= rep["grid_step"]
    ), (
        f"both routes must localize the competition point {rep['ref']} within ONE grid "
        f"step, got {rep['theta_hat_antimode']} / {rep['theta_hat_logistic']}"
    )
    assert rep["kernel_half_in_basin"], (
        "the kernel's late split must sit in the derived trough basin -- if this fails the "
        "kernel's own 'late' mislabels sustained firing and reentrant_workspace owes a fix"
    )

    # NOT A SEED-0 ISLAND (review BLOCKER): the verdict must be a property of the SUBJECT,
    # reproducible across instrument seeds and across a theta band -- the first draft's OK
    # existed at exactly one (theta, seed) and rested on one boundary trial.
    for s in (1, 2):
        rs = score(seed=s)
        assert rs["verdict"] == OK, f"seed {s}: genuine subject must score OK, got {rs['verdict']}"
    for th in (0.40, 0.70):
        rt = score(theta=th)
        assert rt["verdict"] == OK, (
            f"theta {th}: genuine subject must score OK, got {rt['verdict']}"
        )

    # determinism: same seed, same bytes
    assert score() == rep, "the instrument must be deterministic at fixed seed"

    # RED 1 -- SATURATED: a detector that cannot say no (register row 6: '100% ignition =
    # saturated detector'). Constructed as a subject whose report ALWAYS names the probe;
    # under the competitive protocol no honest subject can win every grid point (drives
    # below the reference lose), so a full sweep of wins is the detector lying.
    # distinct-from: reasoning/free_energy_control.py -- always-act CONTROLLER baseline.
    # distinct-from: reservoir/criticality_regulator.py -- pool-rate saturation metric.
    # These are constructed LYING REPORTERS for this scorer's reds; both sites read.
    class _YesReporter(ReentrantWorkspace):
        def step(self):
            out = super().step()
            out["winner"] = "probe"
            return out

    r1 = score(subject_cls=_YesReporter)
    assert r1["verdict"] == SATURATED, (
        f"a cannot-say-no reporter must read SATURATED, got {r1['verdict']}"
    )

    # RED 2 -- SILENT: the mirror -- a report that never names the probe.
    class _NoReporter(ReentrantWorkspace):
        def step(self):
            out = super().step()
            out["winner"] = None
            return out

    r2 = score(subject_cls=_NoReporter)
    assert r2["verdict"] == SILENT, f"a never-yes reporter must read SILENT, got {r2['verdict']}"
    # RED 3 -- UNIMODAL: the graded impostor has no two-peak structure anywhere.
    r3 = score(subject_cls=_LinearImpostor)
    assert r3["verdict"] == UNIMODAL, f"the graded impostor must read UNIMODAL, got {r3['verdict']}"

    # RED 4 -- DISAGREE, driven through score() ITSELF (the review found the first draft
    # re-computed the inequality inline and never executed score()'s own DISAGREE branch):
    # a reporter whose winner verdict flips at 0.30 while the spiking physics keep the
    # real competition point -- the logistic route follows the lying labels, the antimode
    # route follows the rasters, and the dual gate must refuse.
    # distinct-from: sleep/nightly_skills.py -- its label pools are dataset curation;
    #   this is a constructed lying reporter for the dual gate's red. Read.
    class _SkewedReporter(ReentrantWorkspace):
        def step(self):
            probe_drive = max(
                (sal for _src, content, sal in self._buffer if content == "probe"),
                default=0.0,
            )
            out = super().step()
            out["winner"] = "probe" if probe_drive >= 0.30 else "distractor"
            return out

    r4 = score(subject_cls=_SkewedReporter)
    assert r4["verdict"] == DISAGREE, (
        f"a label-skewing reporter must drive score() to DISAGREE, got {r4['verdict']}"
    )

    # RED 5 -- CONTROL_FAIL: an ablation-blind impostor (reentry flag ignored -> sustain
    # identical ON and OFF) must fail the recurrence control even though its sweep looks
    # genuine.
    class _AblationBlind(ReentrantWorkspace):
        def __init__(self, seed=0, reentry=True):
            super().__init__(seed=seed, reentry=True)  # ignores the requested ablation

    r5 = score(subject_cls=_AblationBlind)
    assert r5["verdict"] == CONTROL_FAIL, (
        f"an ablation-blind subject must read CONTROL_FAIL, got {r5['verdict']}"
    )

    # baseline drift check -- and the baseline's ABSENCE is loud, never a silent skip
    # (review: a guard that no-ops when the file is missing is the empty-result-equals-
    # dead-detector shape; the file is tracked, so absence means deletion or a broken
    # path resolution, both of which this red now catches).
    assert _BASELINE.exists(), (
        f"tools/ignition_baseline.json missing at {_BASELINE} -- the drift detector "
        "cannot run; re-record with --record or fix the path resolution"
    )
    base = json.loads(_BASELINE.read_text())
    drift = abs(rep["theta_hat_logistic"] - base["theta_hat_logistic"])
    assert drift <= rep["grid_step"], (
        f"ignition baseline drift {drift} exceeds instrument resolution -- re-derive "
        "or explain before re-recording (tools/ignition_baseline.json)"
    )
    print(
        f"baseline      : recorded {base['theta_hat_logistic']} vs live "
        f"{rep['theta_hat_logistic']} (drift {round(drift, 6)}) -- within band"
    )

    print(
        "\nself-check OK: dual routes agree on the genuine subject at instrument "
        "resolution; window re-derived with the kernel half in its basin; all five "
        "failure modes reached on constructed inputs (SATURATED/SILENT/UNIMODAL/"
        "DISAGREE/CONTROL_FAIL); shuffled-label collapses; ablation removes sustain; "
        "deterministic. MECHANISM evidence only -- no phenomenal claim (P8)."
    )
    return 0


if __name__ == "__main__":
    import sys

    if "--record" in sys.argv:
        print(json.dumps(record_baseline(), indent=2))
        sys.exit(0)
    sys.exit(main())
