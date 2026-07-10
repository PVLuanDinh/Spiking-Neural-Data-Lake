"""
Locus coeruleus (LC) -- noradrenergic ADAPTIVE GAIN: a phasic burst sharpens the readout.

The LC is the brain's main noradrenaline (NA) source and runs in two regimes (Aston-Jones &
Cohen 2005, adaptive-gain theory): a PHASIC mode -- brief NA bursts locked to task-relevant /
salient (target) events that transiently RAISE neural gain and sharpen the downstream decision
(exploitation) -- and a high-TONIC mode -- an elevated baseline with few phasic bursts that is
distractible (exploration). This models NA output as a single scalar level in [0,1]; a phasic
burst on a salient/target event raises a downstream readout gain (the slope of a sigmoid).

Mechanism (rate-level, deterministic):
  1. step(salience) relaxes NA toward the tonic baseline, then -- ONLY for a supra-threshold
     (salient/target) event -- adds a phasic burst on top. A sub-threshold (non-target) event
     adds nothing: the NA level simply stays at the tonic baseline.
  2. gain() maps NA -> the slope of a sigmoid readout: high NA (phasic) = a steep, decisive
     slope; tonic NA = a shallow, mushy slope.
  3. decisiveness = the mean confidence margin |2*sigmoid(gain*e) - 1| over a fixed battery of
     near-threshold 2AFC evidence values e. Steeper gain -> larger margin -> more decisive.

Fair control (un-riggable -- SAME evidence battery and SAME LC in all three conditions, only the
event presented differs):
  (a) baseline  : LC idles at tonic (no event) -- the reference gain/decisiveness.
  (b) TARGET    : a salient target -> a phasic NA burst -> gain up -> the readout SHARPENS.
  (c) NON-target: a non-salient event -> NO phasic burst -> NA stays tonic -> gain and readout
      UNCHANGED from baseline.
  The sharpening must track the phasic burst on the target, and the non-target (an event that
  simply 'happened' but was not salient) must NOT sharpen -- proving the effect is the
  target-locked burst, not merely that an event occurred.

Honest framing: a reduced, rate-level scalar model of LC-NE adaptive gain on synthetic
near-threshold decisions. Additive/opt-in, changes no existing behaviour, and models a
gain-control FUNCTION -- not arousal, not cognition/AGI. Credit: Aston-Jones & Cohen 2005
(adaptive-gain theory); Servan-Schreiber, Printz & Cohen 1990 (gain as sigmoid slope).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.locus_coeruleus
"""
import math
import random

TONIC = 0.20               # tonic (baseline) noradrenaline level in [0,1]
TAU = 4.0                  # phasic relaxation time constant (steps) back toward tonic
PHASIC_KICK = 0.60         # NA added by a phasic burst to a salient/target event
SALIENCE_TH = 0.50         # drive >= this is a salient/target event -> phasic burst; below -> none
PHASIC_TH = 0.45           # NA >= this = phasic/exploit regime, else tonic/explore
GAIN_MIN, GAIN_MAX = 0.5, 15.0   # downstream readout gain (sigmoid slope) as NA runs 0 -> 1

TARGET_SALIENCE = 1.0      # a task-relevant/salient target -> supra-threshold -> phasic burst
NONTARGET_SALIENCE = 0.2   # a non-salient distractor -> sub-threshold -> NO phasic burst


class LocusCoeruleus:
    """Tonic NA level in [0,1] with a phasic burst on salient/target events (Aston-Jones & Cohen
    2005 adaptive-gain theory). A phasic burst transiently RAISES NA -> raises the downstream gain
    (sigmoid slope) -> a steeper, more decisive readout (exploit). With no salient event NA relaxes
    to the tonic baseline -- a low, undifferentiated gain (distractible/explore)."""

    def __init__(self, tonic=TONIC, tau=TAU, kick=PHASIC_KICK, salience_th=SALIENCE_TH,
                 gain_min=GAIN_MIN, gain_max=GAIN_MAX):
        self.tonic = tonic
        self.decay = math.exp(-1.0 / tau)
        self.kick = kick
        self.salience_th = salience_th
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.level = tonic

    def step(self, salience=0.0):
        """Advance one tick. `salience` in [0,1] = task-relevance/salience of this event. NA first
        relaxes toward the tonic baseline; a supra-threshold (salient/target) event then adds a
        PHASIC burst on top. A sub-threshold (non-target) event adds nothing. Returns the NA level."""
        drive = max(0.0, min(1.0, salience))
        self.level = self.tonic + (self.level - self.tonic) * self.decay
        if drive >= self.salience_th:
            self.level += self.kick * drive        # phasic burst -- only on a salient/target event
        self.level = max(0.0, min(1.0, self.level))
        return self.level

    def gain(self):
        """Downstream readout gain (sigmoid slope) -- INCREASES with NA. High NA (phasic) -> steep
        slope -> decisive; tonic NA -> shallow slope -> mushy/undifferentiated."""
        return self.gain_min + (self.gain_max - self.gain_min) * self.level

    def mode(self):
        """Adaptive-gain regime label: PHASIC (exploit) when NA is burst-elevated, else TONIC (explore)."""
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
      baseline  : LC idles at tonic (no event) -- reference gain/decisiveness.
      TARGET    : a salient target -> phasic burst -> NA up -> gain up -> sharper readout.
      NON-target: a non-salient event -> NO burst -> NA stays tonic -> gain and readout unchanged.
    The sharpening must come from the phasic burst on the target, not from 'an event happened'."""
    rng = random.Random(seed)
    battery = _battery(rng)

    base = LocusCoeruleus()
    for _ in range(12):
        base.step(0.0)                            # settle at the tonic baseline
    tonic_level = base.level
    gain_tonic = base.gain()
    decis_base = _decisiveness(gain_tonic, battery)

    tgt = LocusCoeruleus()
    for _ in range(12):
        tgt.step(0.0)
    tonic_before = tgt.level
    na_target = tgt.step(TARGET_SALIENCE)         # supra-threshold -> phasic burst
    gain_phasic = tgt.gain()
    decis_target = _decisiveness(gain_phasic, battery)
    mode_target = tgt.mode()

    ntg = LocusCoeruleus()
    for _ in range(12):
        ntg.step(0.0)
    na_nontarget = ntg.step(NONTARGET_SALIENCE)   # sub-threshold -> NO burst
    gain_nontarget = ntg.gain()
    decis_nontarget = _decisiveness(gain_nontarget, battery)
    mode_nontarget = ntg.mode()

    return {
        "tonic_level": tonic_level, "tonic_before": tonic_before,
        "na_target": na_target, "na_nontarget": na_nontarget,
        "gain_tonic": gain_tonic, "gain_phasic": gain_phasic, "gain_nontarget": gain_nontarget,
        "decis_base": decis_base, "decis_target": decis_target, "decis_nontarget": decis_nontarget,
        "mode_target": mode_target, "mode_nontarget": mode_nontarget,
    }


def main():
    d = demo_phasic_gain(0)

    print("=" * 66)
    print("LOCUS COERULEUS -- noradrenergic adaptive gain (phasic burst sharpens readout)")
    print("=" * 66)
    print(f"tonic NA baseline = {d['tonic_level']:.2f}  (readout gain {d['gain_tonic']:.2f})")
    print("-" * 66)
    print(f"(a) baseline / no event : NA {d['tonic_level']:.2f}  gain {d['gain_tonic']:5.2f}  decisiveness {d['decis_base']:.3f}")
    print(f"(b) TARGET  (salient)   : NA {d['na_target']:.2f}  gain {d['gain_phasic']:5.2f}  decisiveness {d['decis_target']:.3f}  [{d['mode_target']}]")
    print(f"(c) NON-target (no sal.): NA {d['na_nontarget']:.2f}  gain {d['gain_nontarget']:5.2f}  decisiveness {d['decis_nontarget']:.3f}  [{d['mode_nontarget']}]")
    print("-" * 66)
    print(f"phasic burst on target : NA {d['tonic_before']:.2f} -> {d['na_target']:.2f}  (gain x{d['gain_phasic'] / d['gain_tonic']:.2f})")
    print(f"readout sharpening     : decisiveness {d['decis_base']:.3f} -> {d['decis_target']:.3f} on the SAME battery")
    print("=" * 66)

    # ---- self-checks (deterministic, zero-dep, ASCII-only) ------------------
    # 1. phasic burst ON a target: NA rises above tonic, into the phasic/exploit regime
    assert d["na_target"] > d["tonic_before"] + 0.2, "a salient target must evoke a phasic NA burst"
    assert d["mode_target"] == "PHASIC", "a target burst must enter the phasic/exploit regime"
    # 2. NO burst on a non-target: NA stays exactly at the tonic baseline (explore regime)
    assert abs(d["na_nontarget"] - d["tonic_level"]) < 1e-9, "a non-salient event must evoke NO phasic burst"
    assert d["mode_nontarget"] == "TONIC", "a non-target must stay in the tonic/explore regime"
    # 3. the burst RAISES downstream gain and SHARPENS the readout (more decisive)
    assert d["gain_phasic"] > d["gain_tonic"], "the phasic burst must raise downstream gain"
    assert d["decis_target"] > d["decis_base"], "higher gain must sharpen the readout (more decisive)"
    # 4. fair control: the non-target does NOT sharpen -- the benefit is the burst, not 'an event'
    assert abs(d["decis_nontarget"] - d["decis_base"]) < 1e-9, "a non-target must not sharpen the readout"
    assert d["decis_target"] > d["decis_nontarget"], "only the target burst sharpens the readout"
    # 5. determinism: same seed -> identical results
    assert demo_phasic_gain(0) == demo_phasic_gain(0), "same seed must reproduce identical results"

    print(f"self-check OK: target evokes phasic burst NA {d['tonic_before']:.2f} -> {d['na_target']:.2f} "
          f"(gain x{d['gain_phasic'] / d['gain_tonic']:.1f}) sharpening decisiveness "
          f"{d['decis_base']:.3f} -> {d['decis_target']:.3f}; non-target gives no burst and no sharpening")


if __name__ == "__main__":
    main()
