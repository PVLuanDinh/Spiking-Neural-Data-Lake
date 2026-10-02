"""
Evidence-accumulation decision model -- perceptual decision-making by a SPIKING attractor network
(stdlib + numpy/torch allowed project-wide).

Each choice is represented by a POOL of leaky integrate-and-fire (LIF) neurons with recurrent
self-excitation; all pools drive, and are suppressed by, a SHARED pool of inhibitory LIF neurons.
Evidence for a choice is injected as an external input CURRENT into that pool. Positive feedback
(self-excitation) plus the shared inhibition makes the pools compete: the pool with the stronger
input ramps its population firing rate up while the shared inhibition silences the others. The
decision is WHICH POOL first reaches a SUSTAINED high population firing rate; the step at which it
crosses that rate BOUND is the reaction time. This is the spiking-attractor realisation of a
two/N-alternative perceptual decision (Wang, Neuron 36:955, 2002), a biophysical account of the
bounded-accumulator / drift-diffusion algorithm (Ratcliff, Psychol Rev 85:59, 1978; Gold & Shadlen,
Annu Rev Neurosci 30:535, 2007). Winner-take-all competition among the selective pools is the
N-alternative race (cf. Usher & McClelland, Psychol Rev 108:550, 2001), here realised in spikes.

The three signatures that define this model and that emerge from the self-check:
  - a PSYCHOMETRIC curve : accuracy rises with evidence strength (|drift|), from chance at zero
    drift to near-certain at strong drift;
  - a SPEED-ACCURACY tradeoff : raising the firing-rate bound buys accuracy at the cost of a
    longer RT (the attractor is given longer to separate before committing);
  - a CHRONOMETRIC curve : a wider evidence gap is decided FASTER (shorter RT) -- the attractor
    separates sooner.

DISTINCTNESS (this is perceptual DECISION-MAKING, not value learning or control):
  - valence_stdp.py / closed_loop.py are VALUE-based action selection: they LEARN a scalar value
    from reward prediction error (dopamine RPE) and pick the higher-valued action. There is no
    time-integration of evidence, no rate bound, no reaction time. Here nothing is learned: input
    evidence drives a spiking attractor to a threshold, producing a choice AND a latency, with the
    classic speed-accuracy tradeoff.
  - spiking_pid.py is closed-loop CONTROL to a setpoint -- a continuous corrective signal, not a
    discrete commitment to one of several alternatives.
  - not predictive_coding (perceptual inference of latent causes) and not neural_sampling
    (posterior sampling of a joint): this is a DECISION -- an irreversible commit to one option
    when a pool reaches sustained high firing, timestamped by the step at which the bound was hit.

GENUINELY SPIKING (not a rate reduction): the state is the membrane potential of every neuron in
every pool. Each step every neuron leaks, integrates its synaptic + external + noise current, and
emits a DISCRETE spike (reset to rest) when it crosses threshold. The "evidence" is an input
current; the "accumulator" is the population's collective spiking; the "bound" is a sustained
population firing rate, not a scalar integral. numpy / torch are NOT imported.

HONEST scope: this is a REDUCED, minimal spiking attractor -- point LIF neurons, current-based
(not conductance-based) synapses, a single shared inhibitory pool, discrete unit-time steps.
It reproduces the decision-making signatures (psychometric / chronometric / speed-accuracy /
winner-take-all) in spikes; it is NOT the full conductance-based NMDA/GABA network of Wang 2002,
not biophysics, not cognition.

SEROTONERGIC PATIENCE BIAS (specs/022 T025 -- the manifest's FIRST real wire): the dorsal
raphe's tonic 5-HT level arrives on the declared port `patience_in` and sets the decision
BOUND -- the one knob this attractor already owns whose behavioural meaning IS patience
(the speed-accuracy tradeoff above: a higher bound waits longer before committing and buys
accuracy). `patience_bound(level)` maps the level linearly onto this module's own
demonstrated speed-accuracy operating range [_SA_BOUND_LO, _SA_BOUND_HI] -- the two bounds
main() has always used to demonstrate the tradeoff, reused not invented; and because the
raphe's discount rate k is log-interpolated in its level, a bound linear in level is a
bound linear in log-patience. Charge is MEASURED, never declared: the self-check replays
recorded 5-HT frames through `port_charge_probe.replay_twice` (live vs pinned payload) and
the module refuses to call the wire live unless the arms diverge with every frame read --
`--record-charge` writes that measured attribution to tools/region_charge_evidence.json
(the region charge gate's evidence; regenerating it is that mode's ONLY caller -- the
self-check only reads and re-measures, so a battery run can never republish the artifact).

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.faculties.evidence_accumulation
"""

import random

# ---- spiking-attractor constants (shared by both decision classes) --------------------------
# One excitatory POOL per choice + one SHARED inhibitory pool of LIF neurons.
_N_EXC = 12  # neurons per choice pool
_N_INH = 8  # neurons in the shared inhibitory pool
_V_THRESH = 1.0  # spike threshold (membrane units)
_V_RESET = 0.0  # reset potential after a spike
_LEAK_E = 0.82  # excitatory membrane retention per step (leak = 1 - 0.82)
_LEAK_I = 0.82  # inhibitory membrane retention per step
_W_REC = 0.95  # recurrent self-excitation within a pool (the attractor gain)
_W_EI = 0.90  # inhibition from the shared pool onto every choice pool (competition)
_W_IE = 0.80  # excitation from all choice pools onto the shared inhibitory pool
_I_INH_BASE = 0.15  # tonic offset subtracted from the inhibitory drive
_NOISE_SCALE = 0.35  # membrane-noise amplitude per unit of the `noise` parameter
_RATE_DECAY = 0.80  # EMA smoothing of population rate (sustained-firing readout)
_BOUND_SCALE = 0.42  # maps the public `bound` onto a sustained population-rate threshold
_BOUND_CAP = 0.52  # ceiling on the internal rate bound (the attractor's clamped max rate)
_DT_REF = 0.01  # reference dt; the public `dt` scales the per-step integration


_SA_BOUND_LO = 0.5  # the speed-accuracy demo's LOW bound: fast, less accurate (impulsive)
_SA_BOUND_HI = 1.5  # the speed-accuracy demo's HIGH bound: slow, more accurate (patient)


def _rate_bound(bound):
    """Map the public scalar `bound` onto an internal sustained population-rate threshold in (0,1)."""
    return min(_BOUND_CAP, max(0.02, bound * _BOUND_SCALE))


def patience_bound(level):
    """5-HT level in [0,1] -> decision bound, linear across this module's own demonstrated
    speed-accuracy operating range (see the docstring: linear in level = linear in the
    raphe's log-patience). level 0 = impulsive commit-fast bound; level 1 = patient
    wait-for-separation bound."""
    level = max(0.0, min(1.0, float(level)))
    return _SA_BOUND_LO + (_SA_BOUND_HI - _SA_BOUND_LO) * level


def declare_ports(manifest):
    """Manifest registrant (specs/022 T025, the consumer half): the `patience_in` port and
    THE edge from the raphe's `five_ht_out` -- the subscription and its join live with the
    consumer that chose them. join `sum` over the single declared producer is identity."""
    from snn_data_lake.engine.wiring_manifest import Port

    manifest.declare_port(
        "evidence_accumulation",
        Port(
            name="patience_in",
            region="basal_ganglia",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="IN",
            declaring_module="evidence_accumulation",
            join_policy="sum",
        ),
    )
    manifest.declare_edge("raphe_serotonin", "five_ht_out", "evidence_accumulation", "patience_in")


#: The declared edge, exactly as the manifest carries it -- the charge evidence cites it.
PATIENCE_EDGE = ("raphe_serotonin", "five_ht_out", "evidence_accumulation", "patience_in")


def _patience_consumer_factory():
    """Fresh ring + JoinConsumer over the declared IN port, one pair per replay arm."""
    from snn_data_lake.engine.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.engine.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    port = m.ports[("evidence_accumulation", "patience_in")]
    ring = SignalRing(slots=64)
    return ring, JoinConsumer(ring, port)


def consume_patience(consumer, drift=2.0, trials=8, seed=0):
    """The downstream computation under measurement: each joined 5-HT frame sets the bound
    for a block of decisions at fixed evidence. Returns [(tick, accuracy, mean_rt), ...] --
    the behaviour the payload is supposed to move. Deterministic per (tick, seed) so the
    replay arms differ ONLY through the payload."""
    out = []
    joined, _lost = consumer.poll_joined()
    for tick, vec in joined:
        level = float(vec[0])
        model = DriftDiffusion(drift, patience_bound(level), seed=seed + tick)
        block = model.run_trials(trials)
        out.append((tick, accuracy(block, 1), round(mean_rt(block), 6)))
    return out


def _simulate(currents, bound, noise, dt, leak_e, w_ei, max_steps, rng):
    """Run one spiking-attractor decision.

    currents : external input current per choice pool (the injected evidence).
    Returns (winner_index, rt_steps, trace) where trace records genuine spiking:
      trace = {"spikes": total discrete spike events, "rates": final EMA population rates,
               "winner_rate": winner's rate at commit, "runnerup_rate": best loser's rate}.
    The winner is the first pool whose SMOOTHED population firing rate crosses the bound; if no
    pool crosses within max_steps the current leader is returned (still a spiking decision).
    """
    n = len(currents)
    time_scale = dt / _DT_REF
    noise_amp = noise * _NOISE_SCALE
    rate_bound = _rate_bound(bound)

    # membrane potentials: one list per choice pool + the shared inhibitory pool
    v_exc = [[0.0] * _N_EXC for _ in range(n)]
    v_inh = [0.0] * _N_INH
    s_exc_prev = [0.0] * n  # per-pool spiking fraction on the previous step
    s_inh_prev = 0.0  # inhibitory-pool spiking fraction on the previous step
    rate = [0.0] * n  # EMA of each pool's population firing rate
    total_spikes = 0

    for step in range(1, max_steps + 1):
        s_exc = [0.0] * n
        exc_sum_prev = 0.0
        for f in s_exc_prev:
            exc_sum_prev += f

        # --- excitatory choice pools: leak + integrate(evidence + recurrent - inhibition + noise) ---
        for i in range(n):
            base_drive = currents[i] + _W_REC * s_exc_prev[i] - w_ei * s_inh_prev
            row = v_exc[i]
            fired = 0
            for j in range(_N_EXC):
                v = row[j] * leak_e + time_scale * (base_drive + noise_amp * rng.gauss(0.0, 1.0))
                if v >= _V_THRESH:
                    fired += 1
                    v = _V_RESET
                row[j] = v
            s_exc[i] = fired / _N_EXC
            total_spikes += fired

        # --- shared inhibitory pool: driven by total excitatory activity, feeds back to all pools ---
        inh_drive = _W_IE * exc_sum_prev - _I_INH_BASE
        inh_fired = 0
        for k in range(_N_INH):
            v = v_inh[k] * _LEAK_I + time_scale * (inh_drive + noise_amp * rng.gauss(0.0, 1.0))
            if v >= _V_THRESH:
                inh_fired += 1
                v = _V_RESET
            v_inh[k] = v
        s_inh = inh_fired / _N_INH
        total_spikes += inh_fired

        # --- sustained-firing readout: EMA of each pool's population rate, then the bound test ---
        for i in range(n):
            rate[i] = rate[i] * _RATE_DECAY + s_exc[i] * (1.0 - _RATE_DECAY)
        lead = 0
        for i in range(1, n):
            if rate[i] > rate[lead]:
                lead = i
        if rate[lead] >= rate_bound:
            runnerup = max((rate[i] for i in range(n) if i != lead), default=0.0)
            return (
                lead,
                step,
                {
                    "spikes": total_spikes,
                    "rates": rate,
                    "winner_rate": rate[lead],
                    "runnerup_rate": runnerup,
                },
            )

        s_exc_prev = s_exc
        s_inh_prev = s_inh

    lead = 0
    for i in range(1, n):
        if rate[i] > rate[lead]:
            lead = i
    runnerup = max((rate[i] for i in range(n) if i != lead), default=0.0)
    return (
        lead,
        max_steps,
        {
            "spikes": total_spikes,
            "rates": rate,
            "winner_rate": rate[lead],
            "runnerup_rate": runnerup,
        },
    )


class DriftDiffusion:
    """Two-choice spiking-attractor decision (Wang 2002), drop-in for the drift-diffusion accumulator.

    Two excitatory LIF pools (one per option, -1 and +1) compete through a shared inhibitory pool.
    The evidence `drift` is injected as a DIFFERENTIAL input current: pool +1 gets a current above
    baseline and pool -1 gets one below, by |drift| (so sign(drift) is the correct option and
    |drift| is the stimulus strength / "coherence"). The pool that first reaches a sustained high
    population firing rate wins; the crossing step is the reaction time. The RNG is seeded once at
    construction, so two fresh instances with the same params reproduce identical trials.

    Public API is unchanged: __init__(drift, bound, noise, dt, seed, max_steps), decide() -> (choice,
    rt) with choice in {+1, -1}, and run_trials(n) -> [(choice, rt), ...]. After each decide() the
    genuine-spiking trace of the last trial is available as `.last_trace`.
    """

    def __init__(self, drift, bound, noise=1.0, dt=0.01, seed=0, max_steps=2000):
        self.drift = drift
        self.bound = bound
        self.noise = noise
        self.dt = dt
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.last_trace = None
        # baseline drive keeps both pools firing; evidence tilts the balance between them.
        self._base = 0.62
        self._gain = 0.060

    def decide(self):
        """Run one trial to the sustained-firing bound. Returns (choice, rt), choice in {+1, -1}."""
        half = self._gain * self.drift * 0.5
        currents = [self._base - half, self._base + half]  # pool 0 -> -1, pool 1 -> +1
        winner, rt, trace = _simulate(
            currents, self.bound, self.noise, self.dt, _LEAK_E, _W_EI, self.max_steps, self._rng
        )
        self.last_trace = trace
        return (1 if winner == 1 else -1), rt

    def run_trials(self, n):
        """Run n independent trials off the seeded stream. Returns a list of (choice, rt)."""
        return [self.decide() for _ in range(n)]


class RaceAccumulator:
    """N-alternative spiking winner-take-all race (Wang-2002 attractor with N selective pools).

    N excitatory LIF pools (one per alternative) each receive an input current set by their own
    evidence `drifts[i]`; all pools compete through the SHARED inhibitory pool. The first pool to
    reach a sustained high population firing rate wins the race, and the crossing step is the RT.
    `leak` adds membrane leak to the choice pools and `inhibition` adds lateral inhibition on top of
    the network's built-in competition (leak=inhibition=0 keeps the default attractor). Named
    RaceAccumulator (NOT "LCA") to avoid collision with the unrelated lca_sparse_coding.

    Public API is unchanged: __init__(drifts, bound, noise, dt, leak, inhibition, seed, max_steps),
    decide() -> (winner_index, rt), run_trials(n) -> [(winner_index, rt), ...]. After each decide()
    the last trial's genuine-spiking trace is available as `.last_trace`.
    """

    def __init__(
        self, drifts, bound, noise=1.0, dt=0.01, leak=0.0, inhibition=0.0, seed=0, max_steps=2000
    ):
        self.drifts = list(drifts)
        self.bound = bound
        self.noise = noise
        self.dt = dt
        self.leak = leak
        self.inhibition = inhibition
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.last_trace = None
        self._base = 0.46
        self._gain = 0.040

    def decide(self):
        """Run one race to the sustained-firing bound. Returns (winner_index, rt)."""
        currents = [self._base + self._gain * d for d in self.drifts]
        leak_e = max(0.60, _LEAK_E - 0.20 * self.leak)  # more leak -> lower retention
        w_ei = _W_EI + 0.50 * self.inhibition  # more lateral inhibition
        winner, rt, trace = _simulate(
            currents, self.bound, self.noise, self.dt, leak_e, w_ei, self.max_steps, self._rng
        )
        self.last_trace = trace
        return winner, rt

    def run_trials(self, n):
        return [self.decide() for _ in range(n)]


def accuracy(trials, correct):
    """Fraction of trials whose choice equals the correct option."""
    if not trials:
        return 0.0
    return sum(1 for choice, _rt in trials if choice == correct) / len(trials)


def mean_rt(trials):
    """Mean reaction time (steps) over trials."""
    if not trials:
        return 0.0
    return sum(rt for _choice, rt in trials) / len(trials)


def psychometric(drifts, n_trials, bound=1.0, noise=1.0, dt=0.01, seed=0):
    """Accuracy per drift: correctness = choice matches sign(drift). Returns a list aligned to drifts.

    At drift 0 the correct option is defined as +1; by symmetry P(+1) is ~0.5 (true chance)."""
    out = []
    for i, d in enumerate(drifts):
        model = DriftDiffusion(d, bound, noise=noise, dt=dt, seed=seed + i)
        correct = 1 if d >= 0 else -1
        out.append(accuracy(model.run_trials(n_trials), correct))
    return out


def chronometric(drifts, n_trials, bound=1.0, noise=1.0, dt=0.01, seed=0):
    """Mean RT (steps) per drift. Returns a list aligned to drifts."""
    out = []
    for i, d in enumerate(drifts):
        model = DriftDiffusion(d, bound, noise=noise, dt=dt, seed=seed + i)
        out.append(mean_rt(model.run_trials(n_trials)))
    return out


def main():
    # Trial counts: modest (a spiking population per trial is far heavier than a scalar accumulator)
    # but large enough for stable stats at the fixed seed.
    n_psy = 100  # trials per drift for the psychometric / chronometric sweep
    n_sa = 60  # trials per bound for the speed-accuracy comparison
    n_race = 40  # trials for the N-alternative race demo

    # ---- 1 + 3. psychometric (accuracy vs |drift|) and chronometric (RT vs |drift|) ----
    drifts = [0.0, 1.0, 2.0, 4.0, 7.0]
    acc, rts = [], []  # one spiking sweep per drift; acc + RT are two reads of the SAME trials
    for i, d in enumerate(drifts):
        trials = DriftDiffusion(d, 1.0, noise=1.0, dt=0.01, seed=i).run_trials(n_psy)
        acc.append(accuracy(trials, 1 if d >= 0 else -1))
        rts.append(mean_rt(trials))

    print("=" * 68)
    print("SPIKING DECISION ATTRACTOR (Wang 2002) -- LIF pools + shared inhibition -> choice + RT")
    print("=" * 68)
    print(f"psychometric + chronometric (bound=1.0, noise=1.0, dt=0.01, {n_psy} trials/drift):")
    print("   drift   accuracy   mean RT (steps)")
    for d, a, t in zip(drifts, acc, rts):
        print(f"   {d:>4.1f}     {a:>5.2f}       {t:>7.1f}")

    # ---- 2. speed-accuracy tradeoff: fixed moderate drift, LOW bound vs HIGH bound ----
    sa_drift = 2.0
    low_model = DriftDiffusion(sa_drift, bound=0.5, seed=0)
    high_model = DriftDiffusion(sa_drift, bound=1.5, seed=0)
    low = low_model.run_trials(n_sa)
    high = high_model.run_trials(n_sa)
    low_acc, low_rt = accuracy(low, 1), mean_rt(low)
    high_acc, high_rt = accuracy(high, 1), mean_rt(high)
    print(f"\nspeed-accuracy tradeoff (drift={sa_drift}, {n_sa} trials each):")
    print(f"   low  bound 0.5 -> accuracy {low_acc:.2f}, mean RT {low_rt:.1f} steps")
    print(f"   high bound 1.5 -> accuracy {high_acc:.2f}, mean RT {high_rt:.1f} steps")
    print("   -> raising the bound gives the attractor longer to separate: accuracy up, RT up")

    # ---- N-alternative spiking race (winner-take-all among selective pools) ----
    race_drifts = [2.0, 5.0, 10.0]
    race = RaceAccumulator(race_drifts, bound=1.0, leak=0.2, inhibition=0.1, seed=0)
    wins = [0, 0, 0]
    race_trials = race.run_trials(n_race)
    for winner, _rt in race_trials:
        wins[winner] += 1
    strongest = race_drifts.index(max(race_drifts))
    print(f"\nN-alternative spiking race (drifts {race_drifts}, leak+inhibition, {n_race} trials):")
    print(
        f"   win counts {wins} -> strongest evidence (drift {race_drifts[strongest]}) "
        f"wins {wins[strongest]}/{n_race}"
    )

    # ---- chronometric weak vs strong (a slice of the sweep above) ----
    weak_rt, strong_rt = rts[1], rts[-1]  # drift 1.0 vs drift 7.0
    print(
        f"\nchronometric: weak drift {drifts[1]} RT {weak_rt:.1f} steps "
        f"vs strong drift {drifts[-1]} RT {strong_rt:.1f} steps -> wider gap commits faster"
    )

    # ---- evidence this is GENUINELY SPIKING (from the last trial's trace) ----
    hi = DriftDiffusion(4.0, bound=1.0, seed=1)
    _choice, _rt = hi.decide()
    tr = hi.last_trace
    print(
        f"\nspiking evidence (drift 4.0 trial): {tr['spikes']} discrete spike events; "
        f"winner pool rate {tr['winner_rate']:.2f} vs loser {tr['runnerup_rate']:.2f} at commit"
    )
    print("=" * 68)

    # ---- self-checks (genuine, not rigged) --------------------------------------------
    # (a) FUNCTION -----------------------------------------------------------------------
    # 1. psychometric: chance at zero drift, high at strong drift, weakly monotone in |drift|.
    assert 0.40 <= acc[0] <= 0.60, f"accuracy at zero drift must be ~chance ({acc[0]:.2f})"
    assert acc[-1] > 0.9, f"accuracy at strong drift must be high (>0.9) ({acc[-1]:.2f})"
    for i in range(len(acc) - 1):
        assert acc[i + 1] >= acc[i] - 0.05, (
            f"accuracy must rise (weakly) with |drift|: {acc[i]:.2f} -> {acc[i + 1]:.2f}"
        )

    # 2. speed-accuracy tradeoff: high bound = higher accuracy AND longer RT.
    assert high_acc > low_acc, f"high bound must be more accurate ({high_acc:.2f} vs {low_acc:.2f})"
    assert high_rt > low_rt, f"high bound must be slower ({high_rt:.1f} vs {low_rt:.1f} steps)"

    # 3. chronometric: a wider evidence gap is decided clearly faster.
    assert strong_rt < weak_rt, f"strong drift must be faster ({strong_rt:.1f} vs {weak_rt:.1f})"

    # N-alternative race: the strongest pool wins the majority of races.
    assert wins[strongest] == max(wins) and wins[strongest] > n_race / 2, (
        f"strongest evidence must win the race majority ({wins})"
    )

    # (b) GENUINELY SPIKING --------------------------------------------------------------
    # discrete spike events actually occurred, and the decision is a sustained-firing winner
    # that beats the loser's rate (winner-take-all), not a scalar integral.
    assert tr["spikes"] > 0, "a spiking decision must emit discrete spike events"
    assert tr["winner_rate"] > tr["runnerup_rate"], (
        "winner-take-all: the winning pool must out-fire the loser at commit"
    )

    # (c) DETERMINISM: same params/seed -> identical trials across two fresh instances.
    a1 = DriftDiffusion(2.0, 1.0, seed=7).run_trials(20)
    a2 = DriftDiffusion(2.0, 1.0, seed=7).run_trials(20)
    assert a1 == a2, "same seed must reproduce identical (choice, rt) trials"
    r1 = RaceAccumulator([2.0, 6.0], 1.0, seed=3).run_trials(15)
    r2 = RaceAccumulator([2.0, 6.0], 1.0, seed=3).run_trials(15)
    assert r1 == r2, "race must be deterministic under a fixed seed"

    print(
        f"self-check OK: psychometric chance {acc[0]:.2f} -> strong {acc[-1]:.2f}; "
        f"speed-accuracy low (acc {low_acc:.2f}, RT {low_rt:.1f}) vs high "
        f"(acc {high_acc:.2f}, RT {high_rt:.1f}); chronometric weak {weak_rt:.1f} -> "
        f"strong {strong_rt:.1f} steps; {tr['spikes']} spikes/trial; deterministic"
    )

    # ---- (d) THE SEROTONIN WIRE IS CHARGED, measured (specs/022 T025) -------------------
    live, pinned_out, reads, spec_holds = _measure_patience_charge()
    lo_rt_live, hi_rt_live = live[0][2], live[-1][2]
    assert reads == len(live), f"the neuter must be READ every frame ({reads} vs {len(live)})"
    assert live != pinned_out, (
        "pinning the 5-HT payload must change the decision behaviour -- an undiverged wire "
        "is a declared, not a charged, port"
    )
    assert hi_rt_live > lo_rt_live, (
        f"patience must slow commitment: RT at 5-HT=1 ({hi_rt_live:.1f}) must exceed RT at "
        f"5-HT=0 ({lo_rt_live:.1f})"
    )
    assert spec_holds, (
        "specificity control broke: a payload-ignoring consumer diverged under the pin -- "
        "an instrument that fires on everything attributes nothing (probe re-proof, T021)"
    )
    _assert_recorded_evidence_live(live, reads)
    print(
        f"serotonin wire CHARGED (measured): live vs pinned diverged over {reads} read "
        f"frames; RT {lo_rt_live:.1f} (impulsive) -> {hi_rt_live:.1f} (patient) steps; "
        f"specificity control held (probe re-proof discharged against the first real "
        f"registrant)"
    )


def _measure_patience_charge():
    """The replay-twice A/B over recorded 5-HT frames: (live_out, pinned_out, reads,
    specificity_holds). Frames sweep the raphe's own LEVELS so the wire's two ends are the
    real modules, not stand-ins."""
    from snn_data_lake.faculties.raphe_serotonin import LEVELS
    from snn_data_lake.instruments.port_charge_probe import replay_twice

    frames = [(t + 1, "raphe_serotonin", [lvl]) for t, lvl in enumerate(LEVELS)]
    pinned = [0.5]  # the mid setpoint: a plausible constant, not a degenerate zero
    live, pinned_out, reads = replay_twice(
        frames, _patience_consumer_factory, consume_patience, pinned
    )

    def constant_consumer(consumer):
        joined, _lost = consumer.poll_joined()
        return [(tick, 0) for tick, _vec in joined]

    spec_live, spec_pinned, _ = replay_twice(
        frames, _patience_consumer_factory, constant_consumer, pinned
    )
    return live, pinned_out, reads, spec_live == spec_pinned


# distinct-from: faculties/metacognitive_monitor.py -- that is a live rate-quantile HOLD bar;
#   this checks a COMMITTED evidence artifact against a fresh in-command measurement. Read.
def _assert_recorded_evidence_live(live, reads):
    """The committed attribution must correspond to the measurement THIS run just took --
    a recorded charge nobody can re-measure is a declaration with a timestamp. As of T025
    the evidence file is TRACKED, so an absent row is a RED, not a note (review finding:
    a commit that drops the untracked JSON would otherwise leave every gate green while
    the census claim silently evaporates). The recorded numbers must REPRODUCE: same read
    count, same RT endpoints, exactly (the measurement is deterministic). (Read-only: the
    writer is --record-charge, never the self-check -- a battery run must not republish a
    tracked artifact.)"""
    from snn_data_lake.instruments.port_charge_probe import read_attribution

    rec = read_attribution(PATIENCE_EDGE)
    assert rec is not None and rec.get("diverged") is True, (
        f"committed charge evidence must carry the diverged {PATIENCE_EDGE} attribution -- "
        "regenerate with --record-charge and commit it"
    )
    fresh = {"reads": reads, "lo_rt": live[0][2], "hi_rt": live[-1][2]}
    assert rec.get("measured", {}) == fresh, (
        f"recorded charge evidence does not reproduce: recorded {rec.get('measured')} vs "
        f"fresh {fresh} -- the wire changed; regenerate with --record-charge in the same commit"
    )


def record_charge():
    """Measure the A/B and record the attribution via THE one writer
    (port_charge_probe.record_attribution -- T026 consolidated the T025 inline writer into
    it). REFUSES to record a non-diverged run -- evidence is a measurement, never a
    declaration."""
    from snn_data_lake.instruments.port_charge_probe import record_attribution

    live, pinned_out, reads, spec_holds = _measure_patience_charge()
    if live == pinned_out or not spec_holds:
        print("record_charge REFUSED: the A/B did not diverge (or specificity broke)")
        return 1
    path = record_attribution(
        PATIENCE_EDGE,
        {"reads": reads, "lo_rt": live[0][2], "hi_rt": live[-1][2]},
        "port_charge_probe.replay_twice over the raphe LEVELS sweep; the self-check "
        "re-measures and requires `measured` to reproduce exactly, so this record cannot "
        "outlive the wire it describes; specificity control held (payload-ignoring "
        "consumer did not flip)",
    )
    print(f"recorded {PATIENCE_EDGE} into {path}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    if "--record-charge" in _sys.argv[1:]:
        _sys.exit(record_charge())
    main()
