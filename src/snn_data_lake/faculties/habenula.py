"""
Lateral habenula (LHb) -- the anti-reward node, dopamine's negative-RPE opponent.

The VTA dopamine cell fires to BETTER-than-expected outcomes (a positive reward-prediction
error). The LHb is its MIRROR: it fires to WORSE-than-expected outcomes -- reward omission,
punishment, disappointment -- a NEGATIVE reward-prediction error -- and, through the RMTg (a
GABAergic relay just caudal to the VTA), it SUPPRESSES the midbrain dopamine burst. Two
opponent channels, one for each sign of surprise.

Mechanism (spiking readout + a self-contained downstream dopamine cell, deterministic):
  1. Reward-prediction error is the signed surprise, RPE = received - predicted.
  2. The LHb is a readout of the RECTIFIED NEGATIVE part only, neg = max(0, predicted - received)
     -- the amount by which the outcome fell SHORT of the prediction. Better-than-expected
     (received > predicted) gives neg = 0: the LHb has nothing to say (that is the VTA's job).
  3. That negative-RPE current drives a leaky integrate-and-fire LHb neuron over a short window;
     its spike COUNT is the LHb burst. No shortfall -> no drive -> no burst (baseline silence).
  4. A self-contained "dopamine cell" scalar sits downstream: da_drive = da_base + gain * pos,
     where pos = max(0, received - predicted) is the VTA's own positive-RPE excitation. The RMTg
     relay converts the LHb firing RATE into inhibition and SUBTRACTS it from the DA cell:
     da = max(0, da_drive - w_rmtg * lhb_rate). An LHb burst therefore pushes DA BELOW baseline.
  (The DA cell is a standalone scalar defined right here -- this module imports no other region.)

Fair control (un-riggable -- the LHb responds to the NEGATIVE RPE only, it is dopamine's opponent):
  Present outcomes at a FIXED surprise magnitude (|RPE| = 1) so only the SIGN differs.
  (a) EXPECTED  (received == predicted, RPE 0)      -> NO LHb burst; DA rests at baseline.
  (b) WORSE     (omission / punishment, RPE -1)     -> LHb BURST; DA driven BELOW baseline.
  (c) BETTER    (windfall, RPE +1, same magnitude)  -> NO LHb burst; DA rises above baseline.
  The signature is the ASYMMETRY: equal-magnitude errors of opposite sign, and only the negative
  sign drives the LHb. A causal control cuts the RMTg relay (w_rmtg = 0): the omission DA dip then
  vanishes, proving the dip is caused by LHb -> RMTg inhibition and not by the outcome itself.

Honest framing: a reduced, rate/spike-level model of the LHb -> RMTg -> DA anti-reward pathway on
scalar outcomes. It models a negative-RPE / opponent-signalling FUNCTION, not aversion, mood, or
cognition; it is additive/opt-in and changes no existing behaviour.
Credit: Matsumoto & Hikosaka 2007 (LHb encodes negative reward signals); Jhou et al. 2009 (RMTg);
Schultz, Dayan & Montague 1997 (reward-prediction-error dopamine).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.habenula
"""

import random

GAIN = 0.5  # LHb drive gain applied to the rectified negative RPE
LEAK = 0.1  # LIF membrane leak per step
VTH = 1.0  # LIF spike threshold
NOISE = 0.03  # LIF membrane noise sigma (deterministic per-call reseed)
STEPS = 40  # readout window length
DA_BASE = 1.0  # tonic dopamine baseline
DA_GAIN = 1.0  # positive-RPE excitation of the DA cell (the VTA's own job)
W_RMTG = 1.4  # RMTg GABAergic relay weight: LHb rate -> DA inhibition
SEED = 7


class LateralHabenula:
    """LHb negative-RPE readout + RMTg relay onto a self-contained downstream dopamine cell."""

    def __init__(
        self,
        gain=GAIN,
        leak=LEAK,
        vth=VTH,
        noise=NOISE,
        da_base=DA_BASE,
        da_gain=DA_GAIN,
        w_rmtg=W_RMTG,
        seed=SEED,
    ):
        self.gain = gain
        self.leak = leak
        self.vth = vth
        self.noise = noise
        self.da_base = da_base
        self.da_gain = da_gain
        self.w_rmtg = w_rmtg
        self.seed = seed

    @staticmethod
    def rpe(predicted, received):
        """Signed reward-prediction error (the standard convention): received - predicted."""
        return received - predicted

    @staticmethod
    def _neg_rpe(predicted, received):
        """Rectified NEGATIVE RPE: how far the outcome fell SHORT of the prediction (>= 0)."""
        return max(0.0, predicted - received)

    def lhb_spikes(self, predicted, received, steps=STEPS):
        """LIF spike count over the window driven by the negative-RPE current. Per-call reseed ->
        the burst is a PURE, reproducible function of the outcome (worse-than-expected -> spikes;
        expected or better-than-expected -> zero)."""
        drive = self.gain * self._neg_rpe(predicted, received)
        rng = random.Random(self.seed)
        v = 0.0
        spikes = 0
        for _ in range(steps):
            v = (1.0 - self.leak) * v + drive + rng.gauss(0.0, self.noise)
            if v >= self.vth:
                spikes += 1
                v = 0.0
        return spikes

    def burst(self, predicted, received):
        """True when the LHb produces a phasic burst (a worse-than-expected outcome)."""
        return self.lhb_spikes(predicted, received) > 0

    def dopamine(self, predicted, received, steps=STEPS, w_rmtg=None):
        """Downstream DA level after RMTg inhibition. da_drive excites on the POSITIVE RPE (VTA);
        the RMTg relay subtracts the LHb firing rate, so an LHb burst drops DA below baseline.
        Pass w_rmtg=0.0 to lesion the relay (the causal control)."""
        w = self.w_rmtg if w_rmtg is None else w_rmtg
        pos = max(0.0, received - predicted)  # better-than-expected excites DA
        da_drive = self.da_base + self.da_gain * pos
        lhb_rate = self.lhb_spikes(predicted, received, steps) / steps
        rmtg = w * lhb_rate  # RMTg: LHb firing -> DA inhibition
        return max(0.0, da_drive - rmtg)


def main():
    lhb = LateralHabenula()

    # Four outcomes at a FIXED surprise magnitude (|RPE| = 1) so only the SIGN differs -- un-riggable.
    scenarios = [
        ("expected reward", 1.0, 1.0),  # received == predicted   : no error
        ("omission (worse)", 1.0, 0.0),  # expected +1, received 0 : negative RPE
        ("punishment(worse)", 0.0, -1.0),  # expected 0, received -1 : negative RPE
        ("windfall (better)", 0.0, 1.0),  # expected 0, received +1 : POSITIVE RPE (the VTA's job)
    ]

    print("=" * 70)
    print("LATERAL HABENULA -- the anti-reward node (negative-RPE opponent of dopamine)")
    print("=" * 70)
    print(
        f"DA baseline = {lhb.da_base:.2f}   RMTg weight = {lhb.w_rmtg:.2f}   window = {STEPS} steps"
    )
    print("-" * 70)
    print(f"{'outcome':<18}{'pred':>6}{'recv':>6}{'RPE':>7}{'LHb spk':>9}{'DA':>7}   note")
    print("-" * 70)
    rows = {}
    for name, pred, recv in scenarios:
        rpe = lhb.rpe(pred, recv)
        spk = lhb.lhb_spikes(pred, recv)
        da = lhb.dopamine(pred, recv)
        rows[name] = (spk, da)
        if spk > 0:
            note = "LHb burst -> DA suppressed"
        elif da > lhb.da_base:
            note = "DA above baseline (VTA)"
        else:
            note = "baseline"
        print(f"{name:<18}{pred:>6.2f}{recv:>6.2f}{rpe:>+7.2f}{spk:>9d}{da:>7.2f}   {note}")
    print("-" * 70)

    # Causal control: cut the RMTg relay and the omission DA dip must vanish.
    da_omit_intact = lhb.dopamine(1.0, 0.0)
    da_omit_lesion = lhb.dopamine(1.0, 0.0, w_rmtg=0.0)
    print(
        f"RMTg lesion control (omission): DA {da_omit_lesion:.2f} (LHb->DA relay cut) "
        f"vs {da_omit_intact:.2f} (relay intact)"
    )
    print("=" * 70)

    spk_expect, da_expect = rows["expected reward"]
    spk_omit, da_omit = rows["omission (worse)"]
    spk_punish, da_punish = rows["punishment(worse)"]
    spk_better, da_better = rows["windfall (better)"]

    # ---- self-check: deterministic asserts, ASCII-only ----
    # 1. LHb BURSTS on a negative RPE (worse-than-expected: omission and punishment alike).
    assert spk_omit > 0 and spk_punish > 0, "LHb must burst on a negative RPE (worse-than-expected)"
    # 2. NO burst when the outcome matches the prediction (no error).
    assert spk_expect == 0, "an expected reward must evoke NO LHb burst (baseline silence)"
    # 3. NO burst on a POSITIVE RPE -- the asymmetry that makes LHb dopamine's OPPONENT, not its echo.
    assert spk_better == 0, (
        "a better-than-expected outcome must evoke NO LHb burst (that is the VTA's job)"
    )
    # 4. Same |RPE| in both directions, opposite LHb response -- un-riggable sign selectivity.
    assert lhb.rpe(0.0, 1.0) == -lhb.rpe(1.0, 0.0), (
        "windfall and omission are equal-magnitude, opposite-sign RPE"
    )
    # 5. LHb activity REDUCES DA below baseline (the opponent effect, via the RMTg relay).
    assert da_omit < lhb.da_base - 0.1 and da_punish < lhb.da_base - 0.1, (
        "LHb burst must suppress DA below baseline"
    )
    assert da_expect == lhb.da_base, "no error -> DA rests exactly at baseline"
    assert da_better > lhb.da_base, (
        "a positive RPE lifts DA above baseline while the LHb stays silent"
    )
    # 6. The suppression is CAUSED by LHb -> RMTg: cutting the relay abolishes the DA dip.
    assert da_omit_lesion == lhb.da_base and da_omit_intact < lhb.da_base, (
        "cutting RMTg must abolish the DA dip"
    )
    # 7. Determinism: same seed and repeated calls -> identical spikes and DA.
    a, b = LateralHabenula(), LateralHabenula()
    assert a.lhb_spikes(1.0, 0.0) == b.lhb_spikes(1.0, 0.0), (
        "same seed must give identical LHb spike counts"
    )
    assert lhb.lhb_spikes(1.0, 0.0) == spk_omit, (
        "LHb spike count must be a pure, reproducible function of the input"
    )
    assert a.dopamine(1.0, 0.0) == b.dopamine(1.0, 0.0), "same seed must give identical DA"

    print(
        f"self-check OK: LHb bursts on negative RPE ({spk_omit} spk), silent on expected/positive "
        f"(0 spk); DA {lhb.da_base:.2f} -> {da_omit:.2f} on omission (opponent via RMTg); deterministic"
    )


if __name__ == "__main__":
    main()
