"""One object merging the spiking brain with a language model: SpikingLanguageBrain.

This is the single entry point that ties the two halves together -- SPIKING PERCEPTION (a
`brain_bundle` of trained spiking skills, served zero-dep by `trained_brain`) and LANGUAGE (an
optional LLM backend, e.g. `local_llm.AirLLMBackend`, a big model disk-streamed to run beside the
SNN). One object: `perceive` (spiking) + `explain` (perception -> words) + `converse` (language).

HONEST NAMING -- read this. It is a spiking brain WORKING WITH a language model, NOT a spiking
language model. The LLM backend (AirLLM) is a standard DENSE transformer; only the perception side
is spiking. You cannot fuse a small spiking net and a dense LLM into "one large spiking LLM" by
wrapping them -- a true all-spiking LM (SpikeGPT / SpikingBERT) is a transformer built from spiking
neurons and TRAINED FROM SCRATCH; a research-scale scaffold of that lives in `research/spiking_lm.py`
(torch, not this module). This class is the honest, usable integration; that file is the (unverified)
real-spiking-LM starting point.

  spike input --perceive--> class --explain--> words        (SNN + optional LLM)
  text prompt --converse--> words                            (LLM backend only)

  python -m snn_data_lake.cortex.language_brain        # (also run by run_selfchecks.py)
"""

from snn_data_lake.cortex.brain_reasoning import BrainReasoner
from snn_data_lake.substrate.brain_bundle import BrainBundle, load_bundle


class SpikingLanguageBrain:
    """Unified brain-LM. `perception` is a BrainBundle, a single trained_brain dict, or a bundle
    manifest path; `backend` is an optional LLM with `generate(prompt) -> str` (AirLLM in prod)."""

    def __init__(self, perception, backend=None):
        if isinstance(perception, str):
            self.bundle = load_bundle(perception)
        elif isinstance(perception, BrainBundle):
            self.bundle = perception
        elif isinstance(perception, dict) and "layers" in perception:  # a single trained_brain
            self.bundle = BrainBundle({"default": perception})
        else:
            raise TypeError(
                "perception must be a BrainBundle, a trained_brain dict, or a bundle path"
            )
        self.backend = backend

    def skills(self):
        return self.bundle.skill_names()

    def _default_skill(self):
        names = self.bundle.skill_names()
        return "vision" if "vision" in names else names[0]

    def perceive(self, x, skill=None):
        """Spiking perception: route x to a skill's trained brain -> predicted class."""
        return self.bundle.predict(skill or self._default_skill(), x)

    def explain(self, x, skill=None):
        """Perceive + put it in words (the LLM backend if attached, else a deterministic summary)."""
        brain = self.bundle.skills[skill or self._default_skill()]
        return BrainReasoner(brain, self.backend).explain(x)

    def converse(self, prompt):
        """Pure language: hand the prompt to the LLM backend. Without a backend this returns an
        honest note rather than pretending to speak (the SNN alone does not do open language)."""
        if self.backend is None:
            return (
                "(no language backend attached -- perception is spiking and zero-dep, but open "
                "language needs an LLM: install the [llm] extra and pass AirLLMBackend() to "
                "SpikingLanguageBrain.)"
            )
        return self.backend.generate(prompt)

    def describe(self):
        shapes = ", ".join(
            f"{n} ({self.bundle.skills[n]['in']}->{self.bundle.skills[n]['out']})"
            for n in self.bundle.skill_names()
        )
        lang = "an LLM backend" if self.backend is not None else "NO language backend"
        return (
            f"SpikingLanguageBrain: {len(self.skills())} spiking perception skill(s) [{shapes}] "
            f"+ {lang}. This is a spiking brain working WITH a language model -- NOT a spiking "
            f"LLM (the backend is a dense transformer; only perception is spiking)."
        )


class _EchoBackend:
    """Stand-in LLM backend for the self-check (AirLLMBackend is the real one)."""

    def generate(self, prompt):
        return "LLM> " + prompt.split(". ")[0] + "."


def main():
    import random

    from snn_data_lake import snn_classifier as scl
    from snn_data_lake.cortex.trained_brain import make_brain
    from snn_data_lake.cortex.trained_brain import predict as tb_predict

    random.seed(0)
    temps = scl.templates()
    C = len(temps)
    W, _ = scl.train(scl.dataset(scl.TRAIN_PER_CLASS, temps), C, quiet=True)
    brain = make_brain([W], steps=scl.STEPS)
    test = scl.dataset(scl.TEST_PER_CLASS, temps)
    x, y = test[0]

    # a real bundle (one spiking skill), unified with an (echo) language backend
    bundle = BrainBundle({"vision": brain}, {"reflex": True}, {"pid": True})
    slb = SpikingLanguageBrain(bundle, backend=_EchoBackend())
    slb_nolang = SpikingLanguageBrain(brain)  # single brain, no backend

    pred = slb.perceive(x)
    exp = slb.explain(x)
    said = slb.converse("Describe the input. It is a shape.")
    no_lang = slb_nolang.converse("hello")

    print("=" * 68)
    print("SPIKING LANGUAGE BRAIN -- spiking perception + a language model (one object)")
    print("=" * 68)
    print(f"input true class : {y}")
    print(f"perceive (SNN)   : class {pred}")
    print(f"explain          : {exp['text']}")
    print(f"converse (LLM)   : {said}")
    print(f"no-backend say   : {no_lang[:60]}...")
    print(f"describe         : {slb.describe()}")
    print("=" * 68)

    # ---- self-check: deterministic asserts ----
    assert pred == tb_predict(brain, x), "perceive must route to the spiking skill's brain"
    assert exp["pred"] == pred and exp["text"].startswith("LLM> "), (
        "explain must perceive + use the LLM backend"
    )
    exp_nolang = slb_nolang.explain(x)  # no backend -> deterministic summary names the class
    assert exp_nolang["pred"] == pred and str(pred) in exp_nolang["text"], (
        "no-backend explain must name the class in a zero-dep summary"
    )
    assert said.startswith("LLM> "), "converse must use the attached LLM backend"
    assert "backend" in no_lang.lower() and "[llm]" in no_lang.lower(), (
        "without a backend, converse must return an honest note, not fake speech"
    )
    assert "NOT a spiking" in slb.describe(), "describe must state it is not a spiking LLM (honest)"
    assert slb.skills() == ["vision"] and SpikingLanguageBrain(brain).skills() == ["default"], (
        "a single trained_brain wraps as a one-skill bundle"
    )
    assert slb.perceive(x) == pred, "perception is deterministic"
    print(
        f"self-check OK: unified perceive(class {pred}) + explain + converse(LLM) in one object; "
        f"single-brain wraps; honest describe (spiking brain + LLM, not a spiking LLM); deterministic"
    )


if __name__ == "__main__":
    main()
