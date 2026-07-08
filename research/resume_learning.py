"""
v2.67 -- ReSuMe: supervised learning of a PRECISE target output spike train (research tier).

The tempotron (src/snn_data_lake/tempotron.py, Gutig & Sompolinsky 2006) learns a BINARY
decision: adjust the weights so the neuron's peak sub-threshold voltage crosses (or does not
cross) threshold -- fire vs. stay silent for a whole pattern. It says nothing about WHEN the
spike lands.

ReSuMe (Remote Supervised Method; Ponulak & Kasinski, Neural Computation 22(2):467-510, 2010)
is a different problem: make a readout neuron fire at a PRECISE set of target times S_d -- the
exact output spike TRAIN, not a yes/no label. The weight update is a Widrow-Hoff ("delta") rule
lifted onto spikes:

    dw_i(t) = eta * ( S_d(t) - S_out(t) ) * ( a + INT_0^inf K(s) * x_i(t-s) ds )

where S_d is the desired output train, S_out the neuron's actual output, x_i the presynaptic
train through synapse i, K an STDP-like exponential learning kernel, and `a` a small non-Hebbian
offset. Read the sign factor (S_d - S_out):

  * at a DESIRED time (S_d=1, S_out=0)  -> +1: POTENTIATE synapses that were recently active
    (STDP driven by the teacher spike);
  * at a WRONG output time (S_d=0, S_out=1) -> -1: DEPRESS synapses that just fired
    (anti-STDP driven by the actual output);
  * where the trains agree -> 0: no change.

The only fixed point is S_out == S_d: the neuron reproduces the exact target spike train.

Setup here: a leaky integrate-and-fire readout with N=200 plastic synapses, each driven by a
FIXED seeded Poisson input train over a T=100-step window, and a FIXED seeded target train of a
handful of times. Training runs the neuron forward each epoch, then applies the ReSuMe delta
summed over all steps. The HONEST metric is the van Rossum spike-train distance -- REUSED from
snn_data_lake.spike_preprocessing (the module's standard, not a hand-rolled score) -- between the
output train and the target train. It drops from a large "untrained" value to ~0 as the output
converges onto the target times.

Everything is emergent and measured: inputs/target/weights are seeded but never placed near a
solution, spikes are not hand-planted, and the match tolerance and thresholds are fixed a priori.
Deterministic (per-instance random.Random), pure stdlib, ASCII-only.

Ref: Ponulak, F. & Kasinski, A. (2010), "Supervised Learning in Spiking Neural Networks with
ReSuMe: Sequence Learning, Classification, and Spike Shifting", Neural Computation 22(2):467-510,
DOI 10.1162/neco.2009.11-08-901.

  python research/resume_learning.py        # (also run by run_selfchecks.py)
"""
import math
import random

from snn_data_lake.spike_preprocessing import van_rossum_distance

# ---- fixed hyperparameters (chosen a priori, NOT fitted after seeing results) ----
N_SYN = 200        # plastic input synapses
T_WIN = 100        # steps in the learning window
TAU_M = 10.0       # LIF membrane time constant (steps)
TAU_S = 5.0        # synaptic PSC time constant (steps)
TAU_K = 5.0        # ReSuMe learning-kernel time constant (steps)
TAU_VR = 3.0       # van Rossum metric time constant (steps)
THR = 1.0          # firing threshold
REFRAC = 4         # refractory steps after a spike
A_OFF = 0.05       # ReSuMe non-Hebbian offset `a`
ETA = 0.005        # learning rate
EPOCHS = 80        # training epochs
INPUT_RATE = 0.05  # per-step Poisson spike probability per synapse
N_TARGET = 5       # number of desired output spikes
TARGET_LO = 10     # target times sampled from [TARGET_LO, T_WIN - TARGET_LO)
TARGET_SPACING = 8 # minimum spacing between target spikes
TOL = 3            # match tolerance (steps) for the timing-precision metric


def _poisson_inputs(rng):
    """N_SYN fixed seeded Poisson spike trains, one per synapse (list of spike-time lists)."""
    return [[t for t in range(T_WIN) if rng.random() < INPUT_RATE] for _ in range(N_SYN)]


def _target_train(rng):
    """A fixed seeded desired output train: N_TARGET well-separated times in the window."""
    times = set()
    while len(times) < N_TARGET:
        t = rng.randint(TARGET_LO, T_WIN - TARGET_LO - 1)
        if all(abs(t - u) >= TARGET_SPACING for u in times):
            times.add(t)
    return sorted(times)


def _traces(trains, tau):
    """Exponentially filtered spike trains: trace[i][t] = sum_{s<=t} exp(-(t-s)/tau).
    Recurrence trace[t] = trace[t-1]*decay + spike(t). Used both for the synaptic PSC
    (tau=TAU_S, drives the membrane) and the ReSuMe learning kernel (tau=TAU_K)."""
    decay = math.exp(-1.0 / tau)
    out = [[0.0] * T_WIN for _ in range(len(trains))]
    for i, spikes in enumerate(trains):
        sset = set(spikes)
        acc = 0.0
        for t in range(T_WIN):
            acc = acc * decay + (1.0 if t in sset else 0.0)
            out[i][t] = acc
    return out


def _forward(w, psc):
    """Run the LIF readout for one window; return its output spike times.
    Membrane V[t] = V[t-1]*exp(-1/TAU_M) + sum_i w_i*psc_i[t]; fire+reset when V>=THR."""
    decay_m = math.exp(-1.0 / TAU_M)
    v = 0.0
    refr = 0
    out = []
    for t in range(T_WIN):
        cur = 0.0
        for i in range(N_SYN):
            wi = w[i]
            if wi:
                cur += wi * psc[i][t]
        v = v * decay_m + cur
        if refr > 0:
            refr -= 1
            v = 0.0
            continue
        if v >= THR:
            out.append(t)
            v = 0.0
            refr = REFRAC
    return out


def _to_events(times):
    """Spike-time list -> single-channel event list for van_rossum_distance."""
    return [(t, 0) for t in times]


def _distance(out_times, target):
    """Honest metric: REUSED van Rossum distance between output and target trains (channel 0)."""
    return van_rossum_distance(_to_events(out_times), _to_events(target),
                               n_features=1, t_steps=T_WIN, tau=TAU_VR)


def train(seed):
    """One ReSuMe run. Returns (target, initial_out, final_out, trajectory, psc)."""
    rng = random.Random(seed)
    trains = _poisson_inputs(rng)
    target = _target_train(rng)
    w = [rng.uniform(0.0, 0.05) for _ in range(N_SYN)]   # small random -- NOT near a solution
    psc = _traces(trains, TAU_S)          # drives the membrane
    kern = _traces(trains, TAU_K)         # ReSuMe eligibility (a + kern) at each step
    tgtset = set(target)

    out0 = _forward(w, psc)
    traj = [_distance(out0, target)]
    for _ in range(EPOCHS):
        out = _forward(w, psc)
        outset = set(out)
        for i in range(N_SYN):
            ki = kern[i]
            acc = 0.0
            for t in range(T_WIN):
                sd = 1.0 if t in tgtset else 0.0
                so = 1.0 if t in outset else 0.0
                if sd or so:                       # (S_d - S_out) is 0 elsewhere
                    acc += (sd - so) * (A_OFF + ki[t])
            w[i] += ETA * acc
        traj.append(_distance(_forward(w, psc), target))
    return target, out0, _forward(w, psc), traj, psc


def _match_stats(out_times, target):
    """Timing precision: targets hit within +/- TOL, and spurious (false-positive) outputs."""
    matched = sum(any(abs(o - t) <= TOL for o in out_times) for t in target)
    false_pos = sum(all(abs(o - t) > TOL for t in target) for o in out_times)
    return matched, false_pos


def main():
    seed = 0
    target, out0, outf, traj, psc = train(seed)
    d0, df = traj[0], traj[-1]
    matched, false_pos = _match_stats(outf, target)

    # CONTROL: an independent random-weight (untrained) neuron on the same inputs.
    crng = random.Random(9999)
    wc = [crng.uniform(0.0, 0.05) for _ in range(N_SYN)]
    control = _distance(_forward(wc, psc), target)

    print("=" * 64)
    print("ReSuMe -- supervised learning of a precise target output spike train")
    print("=" * 64)
    print(f"{N_SYN} plastic synapses, T={T_WIN} steps, target = {target}")
    print(f"initial output   : {out0}")
    print(f"final output     : {outf}")
    print(f"van Rossum dist  : initial {d0:.3f} -> final {df:.3f}  (ratio {df / d0:.3f})")
    print(f"target match     : {matched}/{len(target)} within +/-{TOL} steps, {false_pos} false positives")
    print(f"control (untrained neuron) van Rossum distance : {control:.3f}")
    print("=" * 64)

    # ---- self-checks: deterministic asserts, fixed a priori, zero-dep ----
    # (a) LEARNS: the reused van Rossum distance drops substantially, in both relative and
    #     absolute terms. Fixed thresholds chosen before running.
    assert df < 0.35 * d0, f"van Rossum distance must drop a lot: {df:.3f} vs 0.35*{d0:.3f}"
    assert df < 0.30, f"final van Rossum distance must be small in absolute terms: {df:.3f}"
    # (b) PRECISE TIMING: every target spike is matched within tolerance, with few spurious spikes.
    assert matched == len(target), f"every target time must be matched: {matched}/{len(target)}"
    assert false_pos <= 1, f"too many spurious output spikes: {false_pos}"
    # (c) CONTROL: an untrained neuron is genuinely FAR from the target (the drop is learning,
    #     not a trivially-close init/target).
    assert control > 2.0, f"control neuron must be far from target: {control:.3f}"
    assert df < 0.2 * control, f"learned distance must be far below control: {df:.3f} vs {control:.3f}"
    # (d) DETERMINISM: same seed -> identical distance trajectory on a repeat run.
    _, _, _, traj2, _ = train(seed)
    assert traj == traj2, "ReSuMe must be deterministic for a fixed seed"

    print(f"self-check OK: ReSuMe learned the exact target train (van Rossum {d0:.2f} -> {df:.2f}), "
          f"{matched}/{len(target)} target spikes matched, {false_pos} false positives, "
          f"control {control:.2f}; deterministic")


if __name__ == "__main__":
    main()
