"""Rational-thinking readout over the trained SNN brain (+ the spike-data lake) -- the DECISION is spiking.

Combines the two halves of the stack: the SNN brain does PERCEPTION (spikes -> class scores, zero-dep
``predict_scores``), and this module makes the reasoned DECISION on top. The decision is NOT rate
arithmetic (no ``argmax``/score-subtraction): the delegated class scores are injected as input CURRENTS
into a spiking WINNER-TAKE-ALL layer -- one leaky integrate-and-fire (LIF) neuron per class, with lateral
inhibition -- and the competition is run in spikes. The reasoned class is the neuron that spikes to the
highest SUSTAINED rate; the CONFIDENCE is the spike-count MARGIN between that winner and the runner-up
(a decisive input drives a large spike-rate gap -> high confidence; a close call -> near-equal spike
counts -> low confidence). An optional LARGE local LLM can then turn that spiking decision into prose.

The LLM backend is INJECTED and OPTIONAL. Any object with ``generate(prompt) -> str`` works -- in
particular ``local_llm.airllm_backend.AirLLMBackend`` (the [llm] extra), which disk-streams a large
model's layers so a 70B-class LLM runs on limited VRAM *alongside* the SNN on one GPU. With NO backend
the reasoner falls back to a deterministic, zero-dependency templated summary, so a clone-and-run
install still explains itself with no torch/LLM.

  SNN brain --> class scores --> [ LIF winner-take-all: spikes -> class + confidence ] --> [ LLM | template ]

  python -m snn_data_lake.cortex.brain_reasoning        # (also run by run_selfchecks.py)
"""

import random

from snn_data_lake.cortex.trained_brain import predict_scores

# --- spiking winner-take-all readout: a LIF layer, one neuron per class, with lateral inhibition ---
WTA_STEPS = 80  # timesteps the spiking competition runs
V_THRESH = 1.0  # LIF firing threshold (membrane crosses this -> a discrete spike)
V_RESET = 0.0  # post-spike membrane reset
MEM_LEAK = 0.9  # membrane retention per step (leaky integrator)
I_BASE = 0.15  # baseline input current: even the weakest class competes (nonzero runner rate)
I_GAIN = 0.60  # score-proportional current gain (the max-score neuron gets I_BASE + I_GAIN)
LATERAL_INHIB = 0.25  # inhibition each spiking neuron subtracts from EVERY other neuron's membrane
_WTA_SEED = 0x5A17  # fixed seed for the per-instance symmetry-breaking RNG (keeps it deterministic)

# confidence bands on the spiking winner-vs-runner spike-count margin (a calibrated [0, 1] value)
CONF_HI = 0.60
CONF_LO = 0.20


class BrainReasoner:
    """Wraps a trained brain with a spiking winner-take-all decision + optional LLM explanation.

    The reasoned class and its confidence emerge from a LIF WTA competition over the brain's class
    scores (not argmax arithmetic). backend: any object exposing ``generate(prompt) -> str`` (e.g.
    AirLLMBackend). If None, a deterministic zero-dep templated summary is produced instead.
    """

    def __init__(self, brain, backend=None):
        self.brain = brain
        self.backend = backend
        self.rng = random.Random(
            _WTA_SEED
        )  # per-instance RNG, reseeded each readout -> deterministic

    def _wta(self, scores):
        """Spiking winner-take-all readout: the reasoned decision, made in spikes.

        Class scores become input CURRENTS to a layer of LIF neurons (one per class) with lateral
        inhibition; the competition runs for ``WTA_STEPS`` steps and the decision is read off the SPIKES.

          * current   : I_i = I_BASE + I_GAIN * (score_i - min) / span  -- a decisive score spread ->
                        a big current gap; a near-tie -> near-equal currents (min gets I_BASE, max gets
                        I_BASE + I_GAIN). span preserves the relative margins.
          * dynamics  : V_i <- MEM_LEAK * V_i + I_i ; a DISCRETE spike (+reset) when V_i >= V_THRESH ;
                        every spike then subtracts LATERAL_INHIB from every OTHER neuron's membrane.
          * winner    : the neuron that spikes to the highest sustained rate (max spike count).
          * margin    : (winner_spikes - runner_spikes) / winner_spikes, in [0, 1] -- the spike-count
                        gap winner-vs-runner. Decisive -> large gap -> ~1 ; a close call -> ~0.

        Returns (winner, runner, margin, counts) where counts is the per-class spike count list.
        """
        n = len(scores)
        lo = min(scores)
        span = (max(scores) - lo) or 1.0
        cur = [I_BASE + I_GAIN * (s - lo) / span for s in scores]  # scores -> input currents
        self.rng.seed(_WTA_SEED)  # reproducible per call
        bias = [1e-6 * self.rng.random() for _ in range(n)]  # tiny deterministic tie-breaker
        V = [0.0] * n  # membrane potential per neuron
        counts = [0] * n  # spike count per neuron
        for _ in range(WTA_STEPS):
            for i in range(n):
                V[i] = MEM_LEAK * V[i] + cur[i] + bias[i]  # leaky integrate the current
            fired = [1 if V[i] >= V_THRESH else 0 for i in range(n)]  # discrete spike events (0/1)
            for i in range(n):
                if fired[i]:
                    counts[i] += 1
                    V[i] = V_RESET  # spike -> reset
            total = sum(fired)
            for i in range(n):
                V[i] -= LATERAL_INHIB * (total - fired[i])  # lateral inhibition from the OTHERS
        order = sorted(
            range(n), key=lambda c: (counts[c], cur[c])
        )  # rank by spike count (tie: current)
        winner = order[-1]
        runner = order[-2] if n > 1 else order[-1]
        margin = (counts[winner] - counts[runner]) / (counts[winner] or 1)
        return winner, runner, margin, counts

    def analyze(self, x):
        """Perceive + reason: delegate to the (already-spiking) classifier for class scores, then let
        a LIF winner-take-all layer make the DECISION -- the reasoned class is the neuron that spikes to
        the highest sustained rate; confidence (margin) is the spike-count gap to the runner-up."""
        scores = predict_scores(self.brain, x)
        pred, runner, margin, counts = self._wta(scores)
        return {
            "pred": pred,
            "runner": runner,
            "margin": margin,
            "scores": [round(s, 4) for s in scores],
            "spikes": counts,
        }

    def _confidence(self, margin):
        return (
            "confident" if margin >= CONF_HI else "tentative" if margin >= CONF_LO else "uncertain"
        )

    def _summary(self, a):
        """Zero-dep deterministic natural-language summary from the spiking decision (no-LLM fallback)."""
        return (
            f"The brain is {self._confidence(a['margin'])} that this is class {a['pred']} "
            f"(spike-count margin {a['margin']:.2f} over the runner-up class {a['runner']})."
        )

    def prompt(self, a):
        """The factual prompt handed to the LLM backend (the model turns it into prose)."""
        return (
            f"A spiking neural network classified an input. Class scores: {a['scores']}. "
            f"A spiking winner-take-all layer elected class {a['pred']} (top score "
            f"{a['scores'][a['pred']]:.2f}), with class {a['runner']} the runner-up (spike-count "
            f"margin {a['margin']:.2f}). Explain the classification and its confidence in one or two sentences."
        )

    def explain(self, x):
        """Perceive + reason: return the spiking decision plus a natural-language explanation. Uses the
        LLM backend if one was injected, else the deterministic zero-dep summary."""
        a = self.analyze(x)
        a["text"] = (
            self.backend.generate(self.prompt(a)) if self.backend is not None else self._summary(a)
        )
        return a


def summarize_store(hub):
    """Zero-dep natural-language summary of a spike-data-lake store (a SpikeTelemetryHub / .spk).
    The lake side of the combine: describe what spike data is on disk, in words."""
    total = hub.n_events()
    counts = [len(hub.ch[c]) for c in range(hub.n)]
    busiest = max(range(hub.n), key=lambda c: counts[c]) if hub.n else -1
    active = sum(1 for c in counts if c)
    return (
        f"The store holds {total} spike events across {hub.n} channels over {hub.dur} time units; "
        f"{active} channels are active, the busiest is channel {busiest} with {counts[busiest] if busiest >= 0 else 0} "
        f"spikes."
    )


def reason_over_store(reasoner, hub):
    """Combine the lake + the brain: turn a .spk store into a feature vector (per-channel event
    counts normalized to [0,1]), let the brain perceive it, and explain -- IF the store's channel
    count matches the brain's input size; otherwise just summarize the store. Returns text."""
    store_text = summarize_store(hub)
    if hub.n != reasoner.brain["in"]:
        return (
            store_text + " (channel count does not match the brain's input size; not classified.)"
        )
    counts = [len(hub.ch[c]) for c in range(hub.n)]
    hi = max(counts) or 1
    x = [c / hi for c in counts]  # per-channel activity as a [0,1] feature
    a = reasoner.explain(x)
    return store_text + " " + a["text"]


class _EchoBackend:
    """A stand-in LLM backend for the self-check: proves any generate(prompt)->str object plugs in
    (AirLLMBackend is the real one). Deterministic: echoes a fixed transform of the prompt."""

    def generate(self, prompt):
        return "LLM> " + prompt.split(". ")[0] + "."


def main():

    from snn_data_lake import snn_classifier as scl
    from snn_data_lake.cortex.trained_brain import make_brain
    from snn_data_lake.substrate.spike_telemetry_hub import SpikeTelemetryHub

    # a real trained brain (single-layer, from the core delta-rule classifier) to reason over
    random.seed(0)
    temps = scl.templates()
    C = len(temps)
    W, _ = scl.train(scl.dataset(scl.TRAIN_PER_CLASS, temps), C, quiet=True)
    brain = make_brain([W], steps=scl.STEPS)
    test = scl.dataset(scl.TEST_PER_CLASS, temps)

    zero_dep = BrainReasoner(brain)  # no LLM -> deterministic template
    with_llm = BrainReasoner(
        brain, backend=_EchoBackend()
    )  # any generate()-backend (AirLLM in prod)

    x, y = test[0]
    a0 = zero_dep.explain(x)
    a1 = with_llm.explain(x)

    # a spike store + the lake-side summary / combine
    hub = SpikeTelemetryHub(8, 100)
    hub.ingest(0, [3, 17, 42])
    hub.ingest(3, [5, 5, 90])
    hub.ingest(7, [10, 11, 12, 13])
    store_text = summarize_store(hub)

    print("=" * 66)
    print("BRAIN REASONING -- spiking winner-take-all decision + optional LLM language layer")
    print("=" * 66)
    print(f"input true class : {y}")
    print(
        f"spiking decision : class {a0['pred']} | spikes {a0['spikes']} | conf {a0['margin']:.2f} ({zero_dep._confidence(a0['margin'])})"
    )
    print(f"zero-dep explain : {a0['text']}")
    print("LLM-backed explain (echo stand-in; AirLLMBackend in prod):")
    print(f"                   {a1['text']}")
    print(f"store summary    : {store_text}")
    print("=" * 66)

    # ---- self-check: deterministic asserts ----
    # (a) FUNCTION -- the spiking WTA elects the RIGHT class: the winner is the argmax of the scores
    #     (the reasoned decision agrees with the brain's own classification on this decisive input).
    argmax = max(range(C), key=lambda c: scl.forward(W, x, C)[0][c])
    assert a0["pred"] == argmax, "spiking WTA winner must be the argmax class on a non-tie input"
    # (a) FUNCTION -- confidence is MONOTONE in the score margin: raise only the winner's score and the
    #     spike-count margin must strictly increase (decisive -> high; near-tie -> low). Uses _wta direct.
    r = BrainReasoner(brain)
    confs = [r._wta([a, 0.5, 0.2])[2] for a in (0.55, 0.70, 1.20, 1.60, 3.00)]
    assert all(0.0 <= m <= 1.0 for m in confs), (
        "spike-margin confidence must be a calibrated [0,1] value"
    )
    assert all(b > a for a, b in zip(confs, confs[1:])), (
        f"confidence must be strictly monotone in the score margin ({[round(m, 3) for m in confs]})"
    )
    assert confs[0] < CONF_LO and confs[-1] > CONF_HI, (
        "a near-tie must read low-confidence and a decisive input high-confidence"
    )
    # and _wta elects the raised (index-0) neuron as winner for every one of those margins
    assert all(r._wta([a, 0.5, 0.2])[0] == 0 for a in (0.55, 0.70, 1.20, 1.60, 3.00)), (
        "the max-score neuron must win the spiking competition"
    )

    # (b) GENUINELY SPIKING -- the readout is a LIF WTA with lateral inhibition; the decision and the
    #     confidence come from DISCRETE spike counts.
    winner, runner, margin, counts = r._wta([3.0, 0.5, 0.2])
    assert all(isinstance(c, int) and 0 <= c <= WTA_STEPS for c in counts), (
        "spikes are discrete events: each neuron fires at most once per step (0/1) -> integer counts <= steps"
    )
    assert sum(counts) > 0, "the WTA layer must actually spike"
    assert counts[winner] > counts[runner] and counts[winner] > counts[2], (
        "lateral inhibition must suppress the losing neurons below the sustained-firing winner"
    )
    assert margin == (counts[winner] - counts[runner]) / (counts[winner] or 1), (
        "confidence must be the winner-vs-runner spike-count margin"
    )

    # (c) DETERMINISM -- same input -> identical spiking decision, across calls and across instances.
    assert BrainReasoner(brain).analyze(x) == BrainReasoner(brain).analyze(x), (
        "the spiking readout must be deterministic across instances"
    )
    assert a0["text"] == zero_dep.explain(x)["text"] and str(a0["pred"]) in a0["text"], (
        "zero-dep summary must be deterministic and name the class"
    )

    # the backend protocol works: any generate(prompt)->str object is used verbatim (AirLLM plugs in).
    assert a1["text"].startswith("LLM> ") and a1["text"] != a0["text"], (
        "an injected LLM backend must drive the explanation text"
    )
    # the lake summary reflects the store contents, and the combine path runs.
    assert "10 spike events" in store_text and "channel 7" in store_text, (
        f"store summary must describe the .spk contents ({store_text})"
    )
    assert reason_over_store(zero_dep, hub).startswith("The store holds"), (
        "reason_over_store must return text"
    )

    print(
        "self-check OK: reasoned decision + confidence emerge from a LIF winner-take-all (spikes, "
        "lateral inhibition, spike-count margin), winner==argmax on non-ties, monotone/calibrated, "
        "any generate()-backend (AirLLM) plugs in, deterministic, zero-dep"
    )


if __name__ == "__main__":
    main()
