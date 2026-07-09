"""Natural-language reasoning over the trained SNN brain (+ the spike-data lake).

Combines the two halves of the stack: the SNN brain does PERCEPTION (spikes -> a class, zero-dep
``trained_brain``), and an optional LARGE local LLM does the LANGUAGE / REASONING layer on top --
explaining a classification, or summarizing what is in a ``.spk`` store, in natural language.

The LLM backend is INJECTED and OPTIONAL. Any object with ``generate(prompt) -> str`` works -- in
particular ``local_llm.airllm_backend.AirLLMBackend`` (the [llm] extra), which disk-streams a large
model's layers so a 70B-class LLM runs on limited VRAM *alongside* the SNN on one GPU. That is where
AirLLM's disk-streaming genuinely fits: not accelerating the tiny SNN (it fits VRAM many times over),
but running a big language model beside it. With NO backend the reasoner falls back to a deterministic,
zero-dependency templated summary, so a clone-and-run install still explains itself with no torch/LLM.

  SNN brain  -->  class + scores  -->  [ AirLLM big local LLM | zero-dep template ]  -->  natural language

  python -m snn_data_lake.cortex.brain_reasoning        # (also run by run_selfchecks.py)
"""
from snn_data_lake.cortex.trained_brain import predict_scores

# confidence bands on the top-1 vs top-2 margin (fixed a priori; used by the zero-dep summary)
CONF_HI = 0.60
CONF_LO = 0.20


class BrainReasoner:
    """Wraps a trained brain with an optional LLM backend to explain its predictions in words.

    backend: any object exposing ``generate(prompt) -> str`` (e.g. AirLLMBackend). If None, a
    deterministic zero-dep templated summary is produced instead -- no LLM, no torch.
    """

    def __init__(self, brain, backend=None):
        self.brain = brain
        self.backend = backend

    def analyze(self, x):
        """Perception: run the brain, return the structured decision (no language)."""
        scores = predict_scores(self.brain, x)
        order = sorted(range(len(scores)), key=lambda c: -scores[c])
        pred = order[0]
        runner = order[1] if len(order) > 1 else order[0]
        return {"pred": pred, "runner": runner, "margin": scores[pred] - scores[runner],
                "scores": [round(s, 4) for s in scores]}

    def _confidence(self, margin):
        return "confident" if margin >= CONF_HI else "tentative" if margin >= CONF_LO else "uncertain"

    def _summary(self, a):
        """Zero-dep deterministic natural-language summary from the decision (the no-LLM fallback)."""
        return (f"The brain is {self._confidence(a['margin'])} that this is class {a['pred']} "
                f"(margin {a['margin']:.2f} over the runner-up class {a['runner']}).")

    def prompt(self, a):
        """The factual prompt handed to the LLM backend (the model turns it into prose)."""
        return (f"A spiking neural network classified an input. Class scores: {a['scores']}. "
                f"It predicts class {a['pred']} (top score {a['scores'][a['pred']]:.2f}), with class "
                f"{a['runner']} the runner-up (margin {a['margin']:.2f}). Explain the classification "
                f"and its confidence in one or two sentences.")

    def explain(self, x):
        """Perceive + reason: return the decision plus a natural-language explanation. Uses the LLM
        backend if one was injected, else the deterministic zero-dep summary."""
        a = self.analyze(x)
        a["text"] = self.backend.generate(self.prompt(a)) if self.backend is not None else self._summary(a)
        return a


def summarize_store(hub):
    """Zero-dep natural-language summary of a spike-data-lake store (a SpikeTelemetryHub / .spk).
    The lake side of the combine: describe what spike data is on disk, in words."""
    total = hub.n_events()
    counts = [len(hub.ch[c]) for c in range(hub.n)]
    busiest = max(range(hub.n), key=lambda c: counts[c]) if hub.n else -1
    active = sum(1 for c in counts if c)
    return (f"The store holds {total} spike events across {hub.n} channels over {hub.dur} time units; "
            f"{active} channels are active, the busiest is channel {busiest} with {counts[busiest] if busiest >= 0 else 0} "
            f"spikes.")


def reason_over_store(reasoner, hub):
    """Combine the lake + the brain: turn a .spk store into a feature vector (per-channel event
    counts normalized to [0,1]), let the brain perceive it, and explain -- IF the store's channel
    count matches the brain's input size; otherwise just summarize the store. Returns text."""
    store_text = summarize_store(hub)
    if hub.n != reasoner.brain["in"]:
        return store_text + " (channel count does not match the brain's input size; not classified.)"
    counts = [len(hub.ch[c]) for c in range(hub.n)]
    hi = max(counts) or 1
    x = [c / hi for c in counts]                         # per-channel activity as a [0,1] feature
    a = reasoner.explain(x)
    return store_text + " " + a["text"]


class _EchoBackend:
    """A stand-in LLM backend for the self-check: proves any generate(prompt)->str object plugs in
    (AirLLMBackend is the real one). Deterministic: echoes a fixed transform of the prompt."""

    def generate(self, prompt):
        return "LLM> " + prompt.split(". ")[0] + "."


def main():
    import random

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

    zero_dep = BrainReasoner(brain)                       # no LLM -> deterministic template
    with_llm = BrainReasoner(brain, backend=_EchoBackend())   # any generate()-backend (AirLLM in prod)

    x, y = test[0]
    a0 = zero_dep.explain(x)
    a1 = with_llm.explain(x)

    # a spike store + the lake-side summary / combine
    hub = SpikeTelemetryHub(8, 100)
    hub.ingest(0, [3, 17, 42]); hub.ingest(3, [5, 5, 90]); hub.ingest(7, [10, 11, 12, 13])
    store_text = summarize_store(hub)

    print("=" * 66)
    print("BRAIN REASONING -- SNN perception + optional LLM language layer")
    print("=" * 66)
    print(f"input true class : {y}")
    print(f"zero-dep explain : pred {a0['pred']} | {a0['text']}")
    print("LLM-backed explain (echo stand-in; AirLLMBackend in prod):")
    print(f"                   {a1['text']}")
    print(f"store summary    : {store_text}")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep ----
    # perception is real: the reasoner's predicted class matches the brain / classifier readout.
    assert a0["pred"] == max(range(C), key=lambda c: scl.forward(W, x, C)[0][c]), \
        "reasoner perception must match the brain's own classification"
    # the zero-dep path produces a deterministic, informative summary (no LLM needed).
    assert a0["text"] == zero_dep.explain(x)["text"] and str(a0["pred"]) in a0["text"], \
        "zero-dep summary must be deterministic and name the class"
    # the backend protocol works: any generate(prompt)->str object is used verbatim (AirLLM plugs in).
    assert a1["text"].startswith("LLM> ") and a1["text"] != a0["text"], \
        "an injected LLM backend must drive the explanation text"
    # the lake summary reflects the store contents.
    assert "10 spike events" in store_text and "channel 7" in store_text, \
        f"store summary must describe the .spk contents ({store_text})"
    # the combine path runs (channel count 8 != brain in=N, so it summarizes without classifying).
    assert reason_over_store(zero_dep, hub).startswith("The store holds"), "reason_over_store must return text"
    print("self-check OK: SNN perception matches the brain; zero-dep summary deterministic; any "
          "generate()-backend (AirLLM) plugs in; .spk store summarized; deterministic")


if __name__ == "__main__":
    main()
