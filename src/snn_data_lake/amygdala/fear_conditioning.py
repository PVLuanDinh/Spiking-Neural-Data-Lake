"""
Spiking CS-US fear conditioning -- the amygdala as a real learned threat detector.

Models associative aversive conditioning as a small spiking mechanism, in three parts that
mirror the amygdala micro-circuit:

  BLA (basolateral amygdala) -- an associative LIF readout. Conditioned-stimulus (CS) features
      (a "tone" pattern, deterministically latency-encoded) drive a leaky integrate-and-fire
      neuron through plastic weights w_exc. A three-factor Hebbian rule BINDS the CS to the
      unconditioned aversive stimulus (US, "shock"): w_exc[i] += lr * US * cs_active[i] * bound.
      The US is the third factor (the aversive teaching signal); potentiation happens ONLY when
      a CS feature is active AND the US co-occurs. Same style as basal_ganglia/valence_stdp.py
      (three-factor plasticity) but keyed on CS-US COINCIDENCE.

  CeA (central nucleus) -- a threshold readout. It emits a THREAT signal (scalar in [0, 1];
      >= THETA_CEA == "conditioned response") from the BLA spike response, GATED DOWN by a
      separate inhibitory pathway. Quiet for an un-conditioned cue, active for a conditioned one.

  EXTINCTION -- a SEPARATE inhibitory path (w_inh, the infralimbic / intercalated-cell gate).
      Presenting the conditioned CS with the US OMITTED grows w_inh, which SUPPRESSES the CeA
      response. It does NOT erase w_exc: the CS-US memory persists under new inhibitory learning
      (biologically honest). Because the inhibitory trace is time-labile (it decays with rest)
      while the excitatory CS-US trace is not, the conditioned response SPONTANEOUSLY RECOVERS
      after a delay -- a signature only a suppressed-not-erased associative memory shows.

Fair control (un-riggable): an explicitly UNPAIRED CS is presented the same number of times but
NEVER co-occurs with the US. Because the learning rule only binds CS features when the US is
present, the unpaired cue CANNOT acquire threat -- by construction of the rule, not by tuning.
The self-check reports paired-CS threat vs unpaired-CS threat side by side.

Honest framing: this is a threat-DETECTOR / associative-memory FUNCTION. It models the learned
input-output mapping of aversive conditioning (acquisition, discrimination, extinction,
spontaneous recovery). It does NOT model fear phenomenology, distress, or suffering, and nothing
here experiences anything.

Deterministic: per-instance random.Random(seed); two runs with the same seed are identical.

  python -m snn_data_lake.amygdala.fear_conditioning
"""
import math
import random

from snn_data_lake.cortex.spike_preprocessing import N, T, encode_latency

# ---- LIF / plasticity constants (hand-tuned; see self-check anchors) ---------
LEAK = 0.90          # BLA membrane leak per step
TH_BLA = 1.0         # BLA firing threshold (subtractive reset)
US_CURRENT = 1.4     # unconditioned drive: the US makes the BLA fire on its own
SAT_EXC = 14.0       # BLA spike count that saturates the excitatory response to 1.0
LR_EXC = 0.10        # three-factor CS-US potentiation rate
LR_INH = 0.22        # extinction (inhibitory) learning rate
W_MAX = 1.0          # soft weight bound
ERASE = 0.30         # per-trial excitatory decay in the ERASURE control model
TAU_REC = 22.0       # inhibitory-trace recovery time constant (rest units)
THETA_CEA = 0.30     # CeA threat threshold: threat >= THETA_CEA == conditioned response


def _by_step(cs):
    """Latency-encode a CS pattern and bucket its spikes by timestep.

    Returns (events_by_step, active_features): events_by_step[t] lists the feature indices that
    spike at step t; active_features is the set of features that spike at all (each <= 1 spike)."""
    ev = encode_latency(cs, T)
    by = [[] for _ in range(T)]
    for t, i in ev:
        by[t].append(i)
    return by, sorted({i for _, i in ev})


class Amygdala:
    """A small spiking amygdala: BLA associative net + CeA threshold readout + extinction gate."""

    def __init__(self, seed=0, n=N):
        self.n = n
        self.w_exc = [0.0] * n     # BLA: CS-feature -> threat association (built by pairing)
        self.w_inh = [0.0] * n     # CeA gate: inhibitory suppression (built by extinction)
        self.elapsed = 0.0         # rest time since extinction (drives inhibitory recovery)
        self.rng = random.Random(seed)

    # ---- spiking readout ----------------------------------------------------
    def _bla_spikes(self, by_step, us_current=0.0):
        """LIF integrate-and-fire over T steps. Per-step input = US drive + sum of w_exc for the
        CS features spiking this step. Returns the BLA output spike count (the raw threat drive)."""
        v = 0.0
        spikes = 0
        for t in range(T):
            cur = us_current
            for i in by_step[t]:
                cur += self.w_exc[i]
            v = v * LEAK + cur
            if v >= TH_BLA:
                spikes += 1
                v -= TH_BLA
        return spikes

    def _recovery_factor(self):
        """Inhibitory trace is time-labile: its expressed strength decays with rest. -> 1 right
        after extinction, -> 0 as rest -> infinity (unmasking the intact CS-US memory)."""
        return math.exp(-self.elapsed / TAU_REC)

    def threat(self, cs):
        """CeA readout: excitatory BLA response minus the (recovery-scaled) inhibitory gate,
        clamped to [0, 1]. This is the conditioned threat signal for cue `cs`."""
        by, active = _by_step(cs)
        exc = min(1.0, self._bla_spikes(by, 0.0) / SAT_EXC)
        if active:
            mean_inh = sum(self.w_inh[i] for i in active) / len(active)
            inh = min(1.0, mean_inh * self._recovery_factor())
        else:
            inh = 0.0
        return max(0.0, min(1.0, exc - inh))

    def is_threat(self, cs):
        """Threshold decision: True when the CeA threat crosses THETA_CEA (conditioned response)."""
        return self.threat(cs) >= THETA_CEA

    # ---- learning -----------------------------------------------------------
    def pair(self, cs):
        """Paired trial: CS presented WITH the US. Three-factor Hebbian potentiation binds the
        active CS features to the US-driven threat pathway (US is the gating third factor).
        Returns the unconditioned BLA response (US drives the BLA regardless of w_exc)."""
        by, active = _by_step(cs)
        ucr = self._bla_spikes(by, US_CURRENT)     # unconditioned response (US drive)
        for i in active:
            self.w_exc[i] += LR_EXC * (1.0 - self.w_exc[i] / W_MAX)
        return ucr

    def expose(self, cs):
        """Present a CS with NO US and NO plasticity -- pure exposure. Used for the explicitly
        unpaired control cue: it is experienced equally often, but because the US never co-occurs
        the three-factor rule binds nothing to it. Returns the current BLA response."""
        by, _ = _by_step(cs)
        return self._bla_spikes(by, 0.0)

    def extinguish(self, cs, erasure=False):
        """CS presented with the US OMITTED.

        Default (suppression, the real amygdala): grow the SEPARATE inhibitory gate w_inh, driven
        by the excitatory prediction the CS still evokes (US expected but absent). w_exc is left
        intact -- the CS-US memory is masked, not erased.

        erasure=True (control model): directly decay w_exc and learn NO inhibition -- the memory
        is destroyed. Used to prove spontaneous recovery needs suppression, not erasure."""
        by, active = _by_step(cs)
        if erasure:
            for i in active:
                self.w_exc[i] *= (1.0 - ERASE)
            return
        exc = min(1.0, self._bla_spikes(by, 0.0) / SAT_EXC)   # prediction error = expected threat
        for i in active:
            self.w_inh[i] += LR_INH * exc * (1.0 - self.w_inh[i] / W_MAX)

    def rest(self, delay):
        """Let time pass. The inhibitory extinction trace recovers (decays); the CS-US trace does
        not -- so a suppressed conditioned response spontaneously returns."""
        self.elapsed += delay


# ---- CS "tone" patterns ------------------------------------------------------
def _tone(seed, lo, hi):
    """A sparse CS 'tone': ~50% of the features in the band [lo, hi) active, random intensity.
    The paired cue and the unpaired control cue live in DISJOINT bands (distinct tones on
    separate channels), so the control shares no features with the paired cue and cannot inherit
    threat through generalization -- the discrimination is genuine CS-US learning, not overlap."""
    r = random.Random(seed)
    v = [0.0] * N
    for i in range(lo, hi):
        if r.random() < 0.50:
            v[i] = r.random()
    return v


def _jitter(vec, rng, noise=0.06):
    """Small trial-to-trial variability so learning sees a noisy CS, not one frozen vector."""
    return [min(1.0, max(0.0, v + rng.uniform(-noise, noise))) if v > 0 else 0.0 for v in vec]


def _condition(seed, erasure=False, rest_delay=30.0):
    """Full protocol on one amygdala; returns the key threat numbers.

    pre     : conditioned cue threat BEFORE any training
    paired  : conditioned (paired) cue threat AFTER paired training
    unpaired: control cue threat AFTER training (presented equally, never with the US)
    ext     : conditioned cue threat AFTER extinction
    rec     : conditioned cue threat AFTER a rest delay (spontaneous recovery)"""
    amyg = Amygdala(seed=seed)
    half = N // 2
    paired_cs = _tone(1, 0, half)        # CS+ tone: lower feature band
    unpaired_cs = _tone(2, half, N)      # CS- tone: upper feature band (disjoint)
    trng = random.Random(seed + 100)

    pre = amyg.threat(paired_cs)

    # --- acquisition: paired cue gets CS+US; control cue gets CS alone (US never contingent) ---
    for _ in range(18):
        amyg.pair(_jitter(paired_cs, trng))
        amyg.expose(_jitter(unpaired_cs, trng))   # presented equally, US never co-occurs
    paired = amyg.threat(paired_cs)
    unpaired = amyg.threat(unpaired_cs)

    # --- extinction: conditioned CS with the US omitted (suppress, or erase in the control) ---
    for _ in range(22):
        amyg.extinguish(_jitter(paired_cs, trng), erasure=erasure)
    ext = amyg.threat(paired_cs)

    # --- spontaneous recovery: let time pass, then re-test the SAME cue ---
    amyg.rest(rest_delay)
    rec = amyg.threat(paired_cs)
    return pre, paired, unpaired, ext, rec


def main():
    print("=" * 70)
    print("AMYGDALA -- spiking CS-US fear conditioning (a learned THREAT DETECTOR)")
    print("threat detector / associative memory only -- NOT fear, distress, or suffering")
    print("=" * 70)

    pre, paired, unpaired, ext, rec = _condition(seed=0)
    print("suppression model (the real amygdala: extinction masks, does not erase):")
    print(f"  pre-training   conditioned CS threat : {pre:.3f}  (< {THETA_CEA} -> no threat)")
    print(f"  paired-CS      threat AFTER training  : {paired:.3f}  (>= {THETA_CEA} -> THREAT)")
    print(f"  unpaired-CS    threat AFTER training  : {unpaired:.3f}  (control: never with US)")
    print(f"  post-extinction conditioned CS threat : {ext:.3f}  (suppressed)")
    print(f"  spontaneous recovery after rest       : {rec:.3f}  (memory returns -> not erased)")

    # erasure control: extinction destroys the CS-US weights instead of masking them
    e_pre, e_paired, e_unpaired, e_ext, e_rec = _condition(seed=0, erasure=True)
    print("\nerasure control (extinction ERASES the CS-US weights):")
    print(f"  paired-CS threat AFTER training       : {e_paired:.3f}")
    print(f"  post-extinction threat                : {e_ext:.3f}  (destroyed)")
    print(f"  after the SAME rest delay             : {e_rec:.3f}  (NO recovery -> nothing left)")

    # determinism: same seed -> identical numbers
    again = _condition(seed=0)
    det = again == (pre, paired, unpaired, ext, rec)
    print(f"\ndeterminism: two seed=0 runs identical = {det}")
    print("=" * 70)

    # ---- un-riggable anchors (deterministic asserts) ------------------------
    # (1) BEFORE training the CS evokes no threat.
    assert pre < THETA_CEA, f"conditioned CS must evoke NO threat before training ({pre:.3f})"
    # (2) AFTER paired training the paired CS is a threat, the unpaired control CS is not.
    assert paired >= THETA_CEA, f"paired CS must become a threat ({paired:.3f})"
    assert unpaired < THETA_CEA, f"unpaired control CS must NOT become a threat ({unpaired:.3f})"
    assert paired > unpaired + 0.30, \
        f"pairing (not mere exposure) must drive learning: {paired:.3f} vs {unpaired:.3f}"
    # (3) AFTER extinction the response drops, then spontaneously recovers on rest...
    assert ext < paired - 0.20, f"extinction must suppress the conditioned response ({ext:.3f})"
    assert ext < THETA_CEA, f"extinction should push the response below threshold ({ext:.3f})"
    assert rec > ext + 0.15, \
        f"conditioned response must spontaneously recover after rest ({rec:.3f})"
    assert rec < paired, \
        f"spontaneous recovery is PARTIAL for a finite rest ({rec:.3f} vs {paired:.3f})"
    # ...but ONLY because the memory was suppressed, not erased: the erasure control cannot recover.
    assert e_ext < THETA_CEA, f"erasure control must also suppress the response ({e_ext:.3f})"
    assert e_rec <= e_ext + 0.05, \
        f"erasure control must NOT recover (nothing left to unmask): {e_rec:.3f} vs {e_ext:.3f}"
    assert rec > e_rec + 0.15, \
        f"recovery is the signature of suppression-not-erasure: {rec:.3f} vs erasure {e_rec:.3f}"
    # determinism
    assert det, "same seed must produce identical numbers"

    print(f"self-check OK: pairing learned threat ({paired:.3f}) while the equally-exposed "
          f"unpaired control did not ({unpaired:.3f}); extinction suppressed it ({ext:.3f}) and it "
          f"spontaneously recovered ({rec:.3f}) -- the erasure control did not ({e_rec:.3f})")


if __name__ == "__main__":
    main()
