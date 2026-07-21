"""
Locus coeruleus (LC) -- a SPIKING noradrenergic population whose FIRING RATE is the adaptive gain.

The LC is the brain's main noradrenaline (NA) source and runs in two regimes (Aston-Jones &
Cohen 2005, adaptive-gain theory): a PHASIC mode -- brief NA bursts locked to task-relevant /
salient (target) events that transiently RAISE neural gain and sharpen the downstream decision
(exploitation) -- and a high-TONIC mode -- an elevated baseline with few phasic bursts that is
distractible (exploration). Here the LC is a POOL of leaky integrate-and-fire (LIF) neurons that
emit DISCRETE spikes: a tonic input drives a baseline population firing rate, and a salient event
injects a phasic current that drives a synchronized BURST of extra spikes. The NA "level" is NOT
an EMA scalar -- it is the LC population's windowed FIRING RATE (spikes counted over the pool x a
readout window), and that spike rate sets the downstream readout gain.

Mechanism (genuinely spiking, deterministic):
  1. step(salience) relaxes a phasic input trace toward zero (back to the tonic baseline drive),
     then -- ONLY for a supra-threshold (salient/target) event -- adds a phasic current kick on
     top. A sub-threshold (non-target) event adds nothing: the input stays at the tonic baseline.
  2. The current input drive is fed to a pool of N_CELLS LIF neurons (heterogeneous thresholds).
     Each neuron leaks + integrates the drive over SUBSTEPS micro-steps and emits a discrete spike
     (with reset) whenever its membrane crosses threshold. NA = spikes / (N_CELLS * SUBSTEPS) --
     the windowed population firing rate. Tonic drive -> a low baseline rate; a phasic burst ->
     a high rate; a sustained higher tonic -> an elevated baseline rate.
  3. gain() maps that spike RATE -> the slope of a sigmoid readout: a high LC rate (phasic burst)
     = a steep, decisive slope; the tonic baseline rate = a shallow, mushy slope.
  4. decisiveness = the mean confidence margin |2*sigmoid(gain*e) - 1| over a fixed battery of
     near-threshold 2AFC evidence values e. Steeper gain -> larger margin -> more decisive.

Fair control (un-riggable -- SAME evidence battery and SAME LC in all three conditions, only the
event presented differs):
  (a) baseline  : LC fires at its tonic rate (no event) -- the reference gain/decisiveness.
  (b) TARGET    : a salient target -> a phasic spike BURST -> rate up -> gain up -> readout SHARPENS.
  (c) NON-target: a non-salient event -> NO phasic burst -> rate stays tonic -> gain and readout
      UNCHANGED from baseline.
  The sharpening must track the phasic burst on the target, and the non-target (an event that
  simply 'happened' but was not salient) must NOT sharpen -- proving the effect is the
  target-locked spike burst, not merely that an event occurred.

Honest framing: a reduced, genuinely-spiking model of LC-NE adaptive gain on synthetic
near-threshold decisions. The pool is re-integrated per readout (a windowed rate code, same choice
as the sibling dopamine source midbrain/dopamine_nuclei.py); the cross-tick burst dynamics live in
the decaying phasic input trace. Additive/opt-in, changes no existing behaviour, and models a
gain-control FUNCTION -- not arousal, not cognition/AGI. Credit: Aston-Jones & Cohen 2005
(adaptive-gain theory); Servan-Schreiber, Printz & Cohen 1990 (gain as sigmoid slope).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.locus_coeruleus
"""

import math
import random

TONIC = 0.20  # tonic (baseline) input drive in [0,1] -> a low baseline firing rate
TAU = 4.0  # phasic relaxation time constant (steps) back toward the tonic drive
PHASIC_KICK = 0.60  # phasic input current added by a salient/target event (burst drive)
SALIENCE_TH = 0.50  # drive >= this is a salient/target event -> phasic burst; below -> none
PHASIC_TH = 0.45  # LC spike rate >= this = phasic/exploit regime, else tonic/explore
GAIN_MIN, GAIN_MAX = 0.5, 15.0  # downstream readout gain (sigmoid slope) as the LC rate runs 0 -> 1

# --- LIF population parameters (the LC pool) --------------------------------------------------
N_CELLS = 64  # LIF neurons in the LC pool (a population, not one unit)
SUBSTEPS = 20  # LIF micro-integration steps per readout = the spike-count window
LEAK = 0.10  # LIF membrane leak per micro-step
THRESH_LO, THRESH_HI = 0.90, 1.10  # heterogeneous per-cell spike thresholds (makes it a population)
DRIVE_GAIN = 1.30  # maps the [0,1] input drive -> LIF input current (phasic pushes > threshold)
NOISE = 0.02  # small deterministic per-micro-step membrane input noise

TARGET_SALIENCE = 1.0  # a task-relevant/salient target -> supra-threshold -> phasic burst
NONTARGET_SALIENCE = 0.2  # a non-salient distractor -> sub-threshold -> NO phasic burst


class LocusCoeruleus:
    """A SPIKING LC: a pool of LIF neurons whose windowed population FIRING RATE is the NA level
    (Aston-Jones & Cohen 2005 adaptive-gain theory). A tonic input drives a low baseline spike
    rate; a salient event injects a phasic current -> a synchronized spike BURST -> a high rate ->
    a raised downstream gain (sigmoid slope) -> a steeper, more decisive readout (exploit). With no
    salient event the phasic drive relaxes and the rate returns to its tonic baseline (explore)."""

    def __init__(
        self,
        tonic=TONIC,
        tau=TAU,
        kick=PHASIC_KICK,
        salience_th=SALIENCE_TH,
        gain_min=GAIN_MIN,
        gain_max=GAIN_MAX,
        seed=0,
    ):
        self.tonic = tonic
        self.decay = math.exp(-1.0 / tau)
        self.kick = kick
        self.salience_th = salience_th
        self.gain_min = gain_min
        self.gain_max = gain_max
        # LIF pool: per-instance deterministic random.Random(seed) for the heterogeneous thresholds;
        # a fixed readout seed keeps the population rate a pure, reproducible function of the drive.
        self.rng = random.Random(seed)
        self.thresholds = [self.rng.uniform(THRESH_LO, THRESH_HI) for _ in range(N_CELLS)]
        self.readout_seed = seed * 100003 + 7
        self.window = N_CELLS * SUBSTEPS  # total spike slots in one readout window
        self.phasic = 0.0  # decaying phasic input trace (cross-tick burst memory)
        self.last_spikes = self._population_spikes(self.tonic)  # discrete spike count this readout
        self.level = self.last_spikes / self.window  # NA = windowed population firing rate

    def _population_spikes(self, drive):
        """Integrate the LC pool for one readout window under a constant input `drive` in [0,1] and
        return the TOTAL number of discrete spikes emitted. Each of the N_CELLS LIF neurons leaks,
        integrates the (gain-scaled) drive over SUBSTEPS micro-steps, and fires a spike + resets its
        membrane whenever it crosses its threshold. A fixed readout rng makes the count a pure,
        deterministic function of `drive`, so the population firing rate is exactly reproducible."""
        rng = random.Random(self.readout_seed)
        current = DRIVE_GAIN * max(0.0, min(1.0, drive))
        spikes = 0
        for thresh in self.thresholds:
            v = 0.0
            for _ in range(SUBSTEPS):
                v = v * (1.0 - LEAK) + current + rng.uniform(-NOISE, NOISE)
                if v >= thresh:
                    spikes += 1  # a discrete spike event
                    v = 0.0  # reset
        return spikes

    def step(self, salience=0.0):
        """Advance one tick. `salience` in [0,1] = task-relevance/salience of this event. The phasic
        input trace first relaxes toward the tonic baseline; a supra-threshold (salient/target) event
        then adds a PHASIC current kick on top -> a synchronized spike burst. A sub-threshold
        (non-target) event adds nothing. The LC pool is read out and NA = the population's windowed
        spike RATE. Returns the NA level (that spike rate)."""
        drive = max(0.0, min(1.0, salience))
        self.phasic *= self.decay  # phasic drive relaxes back toward tonic
        if drive >= self.salience_th:
            self.phasic += self.kick * drive  # phasic burst -- only on a salient/target event
        input_drive = self.tonic + self.phasic
        self.last_spikes = self._population_spikes(input_drive)  # discrete LC spikes this window
        self.level = self.last_spikes / self.window  # NA = windowed population firing rate
        return self.level

    def gain(self):
        """Downstream readout gain (sigmoid slope) -- INCREASES with the LC spike rate. A high rate
        (phasic burst) -> steep slope -> decisive; the tonic baseline rate -> shallow slope -> mushy."""
        return self.gain_min + (self.gain_max - self.gain_min) * self.level

    def mode(self):
        """Adaptive-gain regime label: PHASIC (exploit) when the LC spike rate is burst-elevated,
        else TONIC (explore)."""
        return "PHASIC" if self.level >= PHASIC_TH else "TONIC"


def _sigmoid(x):
    if x <= -60.0:
        return 0.0
    if x >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _decisiveness(gain, battery):
    """Mean confidence margin |2*sigmoid(gain*e) - 1| over a battery of near-threshold evidence
    values e. A steeper gain pushes the readout toward 0/1 -> a larger margin -> more decisive."""
    return sum(abs(2.0 * _sigmoid(gain * e) - 1.0) for e in battery) / len(battery)


def _battery(rng, n=32, spread=0.6):
    """A battery of near-threshold 2AFC evidence values e (net evidence for A over B), drawn around
    the decision boundary so the readout gain is what decides how decisive the answer is."""
    return [rng.uniform(-spread, spread) for _ in range(n)]


def demo_phasic_gain(seed):
    """Three conditions on ONE fixed evidence battery, identical LC except the event presented:
      baseline  : LC fires at its tonic rate (no event) -- reference gain/decisiveness.
      TARGET    : a salient target -> phasic spike burst -> rate up -> gain up -> sharper readout.
      NON-target: a non-salient event -> NO burst -> rate stays tonic -> gain and readout unchanged.
    A fourth, HOT LC (a sustained higher tonic) shows an elevated BASELINE firing rate. The
    sharpening must come from the phasic burst on the target, not from 'an event happened'."""
    rng = random.Random(seed)
    battery = _battery(rng)

    base = LocusCoeruleus()
    for _ in range(12):
        base.step(0.0)  # settle at the tonic baseline firing rate
    tonic_level = base.level
    tonic_spikes = base.last_spikes
    gain_tonic = base.gain()
    decis_base = _decisiveness(gain_tonic, battery)

    tgt = LocusCoeruleus()
    for _ in range(12):
        tgt.step(0.0)
    tonic_before = tgt.level
    na_target = tgt.step(TARGET_SALIENCE)  # supra-threshold -> phasic spike burst
    target_spikes = tgt.last_spikes
    gain_phasic = tgt.gain()
    decis_target = _decisiveness(gain_phasic, battery)
    mode_target = tgt.mode()

    ntg = LocusCoeruleus()
    for _ in range(12):
        ntg.step(0.0)
    na_nontarget = ntg.step(NONTARGET_SALIENCE)  # sub-threshold -> NO burst
    gain_nontarget = ntg.gain()
    decis_nontarget = _decisiveness(gain_nontarget, battery)
    mode_nontarget = ntg.mode()

    hot = LocusCoeruleus(tonic=0.50)  # a sustained higher tonic drive
    for _ in range(12):
        hot.step(0.0)
    hot_tonic_level = hot.level  # -> an elevated baseline firing rate

    return {
        "tonic_level": tonic_level,
        "tonic_before": tonic_before,
        "na_target": na_target,
        "na_nontarget": na_nontarget,
        "tonic_spikes": tonic_spikes,
        "target_spikes": target_spikes,
        "window": base.window,
        "hot_tonic_level": hot_tonic_level,
        "gain_tonic": gain_tonic,
        "gain_phasic": gain_phasic,
        "gain_nontarget": gain_nontarget,
        "decis_base": decis_base,
        "decis_target": decis_target,
        "decis_nontarget": decis_nontarget,
        "mode_target": mode_target,
        "mode_nontarget": mode_nontarget,
    }


def main():
    d = demo_phasic_gain(0)

    print("=" * 66)
    print("LOCUS COERULEUS -- spiking noradrenergic pool (phasic burst rate sets the gain)")
    print("=" * 66)
    print(
        f"LC pool: {N_CELLS} LIF neurons x {SUBSTEPS} micro-steps = {d['window']} spike slots / readout"
    )
    print(
        f"tonic baseline: {d['tonic_spikes']} spikes -> rate {d['tonic_level']:.3f}  (readout gain {d['gain_tonic']:.2f})"
    )
    print("-" * 66)
    print(
        f"(a) baseline / no event : rate {d['tonic_level']:.3f}  gain {d['gain_tonic']:5.2f}  decisiveness {d['decis_base']:.3f}"
    )
    print(
        f"(b) TARGET  (salient)   : rate {d['na_target']:.3f}  gain {d['gain_phasic']:5.2f}  decisiveness {d['decis_target']:.3f}  [{d['mode_target']}]"
    )
    print(
        f"(c) NON-target (no sal.): rate {d['na_nontarget']:.3f}  gain {d['gain_nontarget']:5.2f}  decisiveness {d['decis_nontarget']:.3f}  [{d['mode_nontarget']}]"
    )
    print("-" * 66)
    print(
        f"phasic spike burst      : {d['tonic_spikes']} -> {d['target_spikes']} spikes  "
        f"(rate {d['tonic_before']:.3f} -> {d['na_target']:.3f}, gain x{d['gain_phasic'] / d['gain_tonic']:.2f})"
    )
    print(
        f"sustained higher tonic  : baseline rate {d['tonic_level']:.3f} -> {d['hot_tonic_level']:.3f}  (elevated baseline firing)"
    )
    print(
        f"readout sharpening      : decisiveness {d['decis_base']:.3f} -> {d['decis_target']:.3f} on the SAME battery"
    )
    print("=" * 66)

    # ---- self-checks (deterministic, ASCII-only) ------------------
    # (a) FUNCTION: a salient target RAISES the LC spike rate (into the phasic regime) and the gain
    assert d["na_target"] > d["tonic_before"] + 0.2, (
        "a salient target must evoke a phasic burst (rate up)"
    )
    assert d["mode_target"] == "PHASIC", "a target burst must enter the phasic/exploit regime"
    assert d["target_spikes"] > d["tonic_spikes"], (
        "the phasic burst must emit MORE population spikes"
    )
    # NO burst on a non-target: the spike rate stays EXACTLY at the tonic baseline (explore regime)
    assert abs(d["na_nontarget"] - d["tonic_level"]) < 1e-9, (
        "a non-salient event must evoke NO phasic burst"
    )
    assert d["mode_nontarget"] == "TONIC", "a non-target must stay in the tonic/explore regime"
    # sustained HIGHER tonic gives an elevated BASELINE firing rate (adaptive-gain: tonic sets baseline)
    assert d["hot_tonic_level"] > d["tonic_level"], (
        "a sustained higher tonic must raise the baseline firing rate"
    )
    # the burst RAISES downstream gain and SHARPENS the readout (more decisive)
    assert d["gain_phasic"] > d["gain_tonic"], "the phasic burst must raise downstream gain"
    assert d["decis_target"] > d["decis_base"], (
        "higher gain must sharpen the readout (more decisive)"
    )
    # fair control: the non-target does NOT sharpen -- the benefit is the burst, not 'an event'
    assert abs(d["decis_nontarget"] - d["decis_base"]) < 1e-9, (
        "a non-target must not sharpen the readout"
    )
    assert d["decis_target"] > d["decis_nontarget"], "only the target burst sharpens the readout"

    # (b) GENUINELY SPIKING: NA is a windowed count of DISCRETE spikes, gain scales with that rate
    lc = LocusCoeruleus()
    assert isinstance(lc.last_spikes, int) and lc.last_spikes > 0, (
        "LC neurons must emit discrete integer spikes"
    )
    assert abs(lc.level - lc.last_spikes / lc.window) < 1e-12, (
        "NA level must be the population spike COUNT / window"
    )
    quiet_rate = lc.step(0.0)
    burst_rate = lc.step(TARGET_SALIENCE)
    assert lc.last_spikes / lc.window == burst_rate, (
        "NA must be recomputed from the spike count each readout"
    )
    assert lc.gain() == lc.gain_min + (lc.gain_max - lc.gain_min) * burst_rate, (
        "gain must scale with the spike rate"
    )
    assert burst_rate > quiet_rate, "the spiking burst must raise the population firing rate"

    # (c) DETERMINISM: same seed -> identical results
    assert demo_phasic_gain(0) == demo_phasic_gain(0), "same seed must reproduce identical results"

    print(
        f"self-check OK: target evokes a spike burst {d['tonic_spikes']} -> {d['target_spikes']} "
        f"(rate {d['tonic_before']:.3f} -> {d['na_target']:.3f}, gain x{d['gain_phasic'] / d['gain_tonic']:.1f}) "
        f"sharpening decisiveness {d['decis_base']:.3f} -> {d['decis_target']:.3f}; "
        f"non-target gives no burst; NA is a windowed population spike rate; deterministic"
    )


if __name__ == "__main__":
    main()
