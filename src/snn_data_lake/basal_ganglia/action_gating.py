"""
Spiking basal-ganglia Go/NoGo action SELECTION -- the CBGTC selection stage (#4).

The region already has the LEARNING signal (dopamine RPE in valence_stdp / r_stdp) and a
DECISION rule (evidence_accumulation) but not the selection CIRCUIT that turns several
candidate actions into exactly one released action. This is that circuit: the
cortico-basal-ganglia-thalamo-cortical (CBGTC) loop's selection bottleneck.

Mechanism (Go/NoGo disinhibition -- Mink 1996; Frank 2005; Redgrave 1999):
  - Cortex drives the striatum with one candidate channel per action.
  - DIRECT pathway  (D1, "Go")   : excited by cortex, INHIBITS GPi/SNr on its own channel.
  - INDIRECT pathway (D2, "NoGo"): excited by cortex, EXCITES GPi/SNr on its own channel.
  - GPi/SNr hold the thalamus under strong TONIC INHIBITION. A channel is RELEASED only by
    DISINHIBITION: Go lowers its GPi output (thalamic relay is freed to spike), NoGo raises it.
  - Winner-take-all: lateral inhibition in the striatum (k-WTA, k=1, the same primitive used
    in research/fly_hash.py and hippocampus/hippocampus.py) lets only the strongest candidate
    open its Go gate, so at most one channel is ever disinhibited -- the disinhibition bottleneck.
  - Dopamine sets the D1/D2 balance: high DA raises the Go gain and suppresses NoGo (the RPE
    from valence_stdp would supply this level); DA=0 is an SNc lesion (Parkinsonian).

The released action is read out as a spike from a leaky integrate-and-fire (LIF) thalamic relay
neuron: it integrates a constant cortical drive minus its GPi inhibition, and fires when the net
crosses threshold. A channel that stays under tonic GPi inhibition never reaches threshold.

Fair control (un-riggable): dopamine is load-bearing, not decorative. With DA removed the D2/NoGo
pathway dominates, GPi over-inhibits, and selection SLOWS then FAILS on the SAME inputs:
  intact DA   -> fast release of the correct channel,
  partial DA  -> bradykinesia (every release is later),
  DA lesion   -> akinesia (nothing is released at all).
The self-check reports the intact / partial / lesion numbers so the effect is verifiable.

Honest framing: this models the action-SELECTION function -- which candidate the circuit
disinhibits and releases -- NOT volition, choice, agency, or consciousness. A "selection" here
is a disinhibition event in a spiking circuit, nothing more.

  python -m snn_data_lake.basal_ganglia.action_gating
"""
import random


class BasalGangliaGate:
    """Cortex -> striatum(D1/D2) -> GPi/SNr -> LIF thalamic relay Go/NoGo selection circuit.

    select(drive, da) takes a per-channel cortical drive vector and a dopamine level in [0, 1]
    and returns which channel (if any) is disinhibited and released, plus its latency in steps.
    """

    def __init__(self, seed=0, d1_base=0.25, d2_base=0.50, da_gain=0.90,
                 tonic=1.0, base_exc=1.0, leak=0.15, v_thresh=1.0, k=1, steps=40):
        self.rng = random.Random(seed)
        self.d1_base = d1_base      # baseline D1 (Go) gain, before dopamine
        self.d2_base = d2_base      # baseline D2 (NoGo) gain, before dopamine
        self.da_gain = da_gain      # how strongly dopamine tilts D1 up / D2 down
        self.tonic = tonic          # tonic GPi/SNr inhibition on every thalamic channel
        self.base_exc = base_exc    # constant cortical excitation onto the thalamic relay
        self.leak = leak            # LIF membrane leak
        self.v_thresh = v_thresh    # release (spike) threshold of the thalamic relay
        self.k = k                  # k-WTA winners allowed to open a Go gate (bottleneck = 1)
        self.steps = steps          # max integration steps per selection

    @staticmethod
    def _kwta(values, k):
        """Indices of the k largest values (lateral-inhibition winner-take-all).

        Deterministic tie-break by lowest index, matching fly_hash.py / hippocampus.py.
        """
        order = sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)
        return set(order[:k])

    def _striatum(self, drive, da):
        """Cortex -> D1/D2 -> GPi outputs. Returns (go, nogo, gpi) per channel.

        High dopamine raises the Go gain and suppresses NoGo. Only the k-WTA Go winner opens
        its gate (lateral inhibition), so at most one channel's GPi drops below tonic.
        """
        go = [d * (self.d1_base + self.da_gain * da) for d in drive]
        nogo = [d * max(0.0, self.d2_base - self.da_gain * da) for d in drive]
        winners = self._kwta(go, self.k)
        gpi = []
        for j in range(len(drive)):
            go_eff = go[j] if j in winners else 0.0        # losers: lateral inhibition shuts Go
            gpi.append(max(0.0, self.tonic - go_eff + nogo[j]))   # Go disinhibits, NoGo inhibits
        return go, nogo, gpi

    def select(self, drive, da):
        """Run the circuit. Returns a dict: winner channel + latency, or winner=None if the
        tonic GPi inhibition is never overcome (no action released)."""
        go, nogo, gpi = self._striatum(drive, da)
        v = [0.0] * len(drive)
        fired = [False] * len(drive)
        released = []
        for t in range(1, self.steps + 1):
            for j in range(len(drive)):
                if fired[j]:
                    continue
                net = self.base_exc - gpi[j]               # cortical drive minus GPi inhibition
                v[j] = max(0.0, v[j] + (-self.leak * v[j] + net))
                if v[j] >= self.v_thresh:                  # thalamic relay releases (spikes)
                    fired[j] = True
                    released.append((j, t))
            if released:                                   # bottleneck: first release wins
                break
        winner = released[0] if released else None
        disinhibited = min(range(len(gpi)), key=lambda j: gpi[j])
        return {
            "winner": winner[0] if winner else None,
            "latency": winner[1] if winner else None,
            "released": released,
            "disinhibited": disinhibited,
            "gpi": gpi,
        }

    def make_trials(self, strongest_vals, n_channels):
        """Build one drive vector per value in strongest_vals: that value is the unique strongest
        channel (placed at a random channel via self.rng), the rest are weaker competitors."""
        trials = []
        for s in strongest_vals:
            others = [self.rng.uniform(0.15, s - 0.12) for _ in range(n_channels - 1)]
            pos = self.rng.randrange(n_channels)
            trials.append(others[:pos] + [s] + others[pos:])
        return trials


def _sweep(gate, trials, da):
    """Return (successes, total, mean_latency) over trials at a fixed dopamine level."""
    lat = [r["latency"] for r in (gate.select(tr, da) for tr in trials) if r["winner"] is not None]
    total = len(trials)
    mean = sum(lat) / len(lat) if lat else float("nan")
    return len(lat), total, mean


def main():
    n_ch = 4
    da_intact, da_partial, da_lesion = 0.80, 0.50, 0.00
    strongest = [0.87, 0.72, 0.60, 0.55, 0.90, 0.68]

    gate = BasalGangliaGate(seed=0)
    trials = gate.make_trials(strongest, n_ch)

    rep = [0.40, 0.55, 0.87, 0.30]           # a clear representative trial: channel 2 is strongest
    rep_res = gate.select(rep, da_intact)

    intact = _sweep(gate, trials, da_intact)
    partial = _sweep(gate, trials, da_partial)
    lesion = _sweep(gate, trials, da_lesion)

    null_res = gate.select([0.0] * n_ch, da_intact)   # tonic GPi only, NO cortical drive

    print("=" * 68)
    print("BASAL GANGLIA Go/NoGo ACTION SELECTION (CBGTC disinhibition bottleneck)")
    print("=" * 68)
    print(f"channels/trial: {n_ch} | trials: {len(trials)} | max steps: {gate.steps}")
    print(f"representative drive: {[round(x, 2) for x in rep]}")
    print(f"  -> strongest channel     : {rep.index(max(rep))} (drive {max(rep):.2f})")
    print(f"  -> GPi output per channel : {[round(g, 2) for g in rep_res['gpi']]}  "
          "(winner's GPi is disinhibited)")
    print(f"  -> released channel       : {rep_res['winner']} at step {rep_res['latency']}\n")

    print("dopamine is load-bearing (same 6 inputs, only DA differs):")
    print(f"  intact  DA={da_intact:.2f} : {intact[0]}/{intact[1]} released, "
          f"mean latency {intact[2]:.2f} steps")
    print(f"  partial DA={da_partial:.2f} : {partial[0]}/{partial[1]} released, "
          f"mean latency {partial[2]:.2f} steps   (bradykinesia -- later)")
    lesion_lat = "n/a" if lesion[0] == 0 else f"{lesion[2]:.2f}"
    print(f"  lesion  DA={da_lesion:.2f} : {lesion[0]}/{lesion[1]} released, "
          f"mean latency {lesion_lat} steps   (akinesia -- nothing released)\n")

    print(f"null case (tonic GPi only, NO cortical drive): released = {null_res['winner']} "
          "(no spurious action)")
    print("=" * 68)

    # ---- un-riggable anchors (deterministic asserts) ------------------------
    # (1) correct ordering: the strongest channel is the one disinhibited AND released.
    assert rep_res["winner"] == rep.index(max(rep)), "strongest channel must be the one released"
    assert rep_res["disinhibited"] == rep.index(max(rep)), "strongest channel must be disinhibited"
    # (3) exactly one channel wins -- the disinhibition bottleneck.
    assert len(rep_res["released"]) == 1, "exactly one channel may be released"
    # (2) SAFETY: no cortical drive -> tonic GPi inhibits everything -> nothing is released.
    assert null_res["winner"] is None and null_res["released"] == [], \
        "no-drive tonic-GPi state must release NO action"
    # Fair control: DA is load-bearing. Removing it FAILS selection; reducing it SLOWS it.
    assert intact[0] == len(trials), "intact dopamine should release every trial"
    assert lesion[0] == 0, "DA lesion (SNc) must fail selection -- akinesia"
    assert lesion[0] < intact[0], "DA lesion must release fewer actions than intact"
    assert partial[2] > intact[2], "reduced dopamine must slow selection -- bradykinesia"
    # Determinism: same seed -> identical trials and identical selection.
    gate2 = BasalGangliaGate(seed=0)
    trials2 = gate2.make_trials(strongest, n_ch)
    rep2 = gate2.select(rep, da_intact)
    assert trials2 == trials, "same seed must reproduce identical trials"
    assert (rep2["winner"], rep2["latency"], rep2["released"]) == \
        (rep_res["winner"], rep_res["latency"], rep_res["released"]), "same seed must reproduce run"

    print(f"self-check OK: strongest channel {rep_res['winner']} disinhibited+released (1 winner); "
          "no-drive releases nothing;")
    print(f"  intact {intact[0]}/{intact[1]} @ {intact[2]:.2f} steps -> partial "
          f"{partial[0]}/{partial[1]} @ {partial[2]:.2f} (slower) -> lesion {lesion[0]}/{lesion[1]} "
          "(akinesia); deterministic")


if __name__ == "__main__":
    main()
