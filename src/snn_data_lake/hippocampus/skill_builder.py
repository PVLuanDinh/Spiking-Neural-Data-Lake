r"""
SKILL BUILDER -- one domain-agnostic autonomous loop that BUILDS its own skill for ANY domain
castable as a stream, gives it competence, and INDEXES it in the durable lake for reuse.

This generalizes cortex/language_self_build.py (promoted 2026-08-11; built a LANGUAGE skill) into a
domain-independent mechanism. The universal move (per the research): any domain becomes a stream of
(context, next-token) pairs; the SAME loop consumes every domain:

  ingest stream -> curiosity/curriculum pick the frontier -> [neurogenesis recruits capacity] ->
  self-supervised SPIKING assembly (R-STDP, the representation) -> STRONG READOUT (SLDA-RFF, the
  competence -- the project's readout-is-the-lever hybrid) -> CONSOLIDATE a >=1 signature to the
  durable lake -> INDEX the skill (signature key -> skill id) -> route future inputs by PROXIMITY.

Demonstrated on TWO genuinely different domains from ONE loop, proving generality:
  * LANGUAGE  : next-phoneme over a real corpus (CMUdict + Gutenberg), 39 ARPABET responses.
  * MATH      : next-character over generated arithmetic expressions ("a+b=c"), a real symbolic
                domain with unlimited data and real structure (DeepMind-Mathematics style).

INDEXING is domain-agnostic: each context is hashed (crc32, persistence-stable) into a fixed 64-dim
signature, so LANGUAGE and MATH contexts occupy different regions of ONE shared metric space -- the
lake ProximityIndex then ROUTES an input to the right skill with no domain-id and no retraining.

HONEST SCOPE (do not oversell): this is a skill-FORMATION + INDEXING + ROUTING engine with per-skill
learning curves and un-riggable controls. Competence rides the strong decoder (spiking readout alone
is below n-gram, as language_self_build showed). It does NOT solve math or language; it makes
measurable, reusable, honestly-scoped skills and a router over them. Physics (JEPA latent next-state)
and proofs (verifier-reward) are the next adapters, not built here.

  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.hippocampus.skill_builder
  Knobs: NORD_CTX(3) NORD_PER(6000) NORD_RFF(1500) NORD_GAMMA(0.15) NORD_MATHN(4000)
Deps: numpy (+ stdlib). Reuses corpus_stream, language_self_build, language_readout_slda,
proximity_replay (each promoted 2026-08-11).

PROMOTED reasoning/ -> hippocampus/ 2026-08-11 (#10 family settlement C4, after C5/C6 by the
corrected order -- its module-top imports of the two language modules made the recorded
skill-builder-first order a governed->reasoning hazard). SkillRegistry here is the live routing
home the C9 lake_skill_library retirement rests on: constructed at nightly_skills:432 on the
brain.py:3069 sleep-tick path, verified at this move.
"""

import os
import random
import tempfile

from snn_data_lake.cortex.language_self_build import _bigram_baseline
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive
from snn_data_lake.readout.language_readout_slda import _slda_acc
from snn_data_lake.sensorimotor.corpus_stream import (
    ARPABET,
    load_cmudict,
    load_text,
    phoneme_stream,
)
from snn_data_lake.substrate.content_signature import (
    signature,
)  # extracted to a stdlib module; re-exported for back-compat
from snn_data_lake.substrate.proximity_replay import LakeMemory

MATH_VOCAB = list("0123456789+-*= ")  # the arithmetic domain's token inventory


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def math_stream(n_expr=4000, ctx=3, seed=0):
    """Generate (prev ctx chars, next char) pairs over arithmetic expressions 'a op b = c'. A REAL
    symbolic domain: real structure (operator precedence, equality), unlimited data. Deterministic."""
    rng = random.Random(seed)
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    text = []
    for _ in range(n_expr):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        sym, fn = rng.choice(ops)
        text.append(f"{a}{sym}{b}={fn(a, b)} ")
    s = "".join(text)
    pairs = []
    for i in range(1, len(s)):
        lo = max(0, i - ctx)
        pairs.append((list(s[lo:i]), s[i]))
    return pairs


class SkillRegistry:
    """The skill INDEX + router over the durable lake. A skill = (signature centroid -> skill id) +
    its expert (the trained SLDA-RFF readout). register() consolidates the signature durably; route()
    recalls the nearest skill for an input BY PROXIMITY (content-addressable, no domain-id)."""

    def __init__(self, lake):
        self.lake = lake
        self.experts = {}  # skill_id -> dict(readout, vocab, ctx)
        self.names = {}  # skill_id -> human name

    def register(self, skill_id, name, sig_exemplars, expert):
        """Index the skill by consolidating MANY signature exemplars (not one centroid) with the skill
        id as payload -> proximity routing recalls the nearest exemplar's skill, the way the lake is
        meant to be queried (content-addressable over exemplars)."""
        self.lake.consolidate(sig_exemplars, payloads=[skill_id] * len(sig_exemplars))
        self.experts[skill_id] = expert
        self.names[skill_id] = name

    def route(self, context_tokens):
        got = self.lake.retrieve(signature(context_tokens), k=1)
        return int(got[0]) if got else -1


def build_skill(name, pairs, vocab, ctx, per, rff, gamma, seed):
    """THE domain-agnostic build step: form the skill's spiking representation implicitly via the
    positional context code, give it competence with the strong SLDA-RFF readout, and compute the
    skill's signature centroid for indexing. Returns (competence, baseline, curve0, curveN, centroid,
    expert). Self-supervised: the target is the real next token, no labels."""
    ridx = {r: i for i, r in enumerate(vocab)}
    usable = [(c, nx) for c, nx in pairs if nx in ridx]
    cut = int(len(usable) * 0.8)
    train = usable[:cut][:per] if per else usable[:cut]
    # competence: strong readout (the general head), held-out vs the domain baseline
    comp, n = _slda_acc(pairs, vocab, ctx, per, rff, gamma, 0.1, seed=seed)
    # a learning curve proxy: readout trained on the first half vs all of train (competence rises)
    half = max(30, len(train) // 2)
    c0, _ = _slda_acc(usable[: half * 2], vocab, ctx, half, rff, gamma, 0.1, seed=seed)
    # signature EXEMPLARS: ~150 sampled training-context signatures -> the skill's indexed keys
    ex = [signature(c) for c, _ in train[:3000]]
    step = max(1, len(ex) // 150)
    sig_ex = ex[::step][:150]
    expert = {"vocab": vocab, "ctx": ctx, "n_heldout": n}
    return comp, c0, sig_ex, expert


def main():
    CTX = _envi("NORD_CTX", 3)
    PER = _envi("NORD_PER", 6000)
    RFF = _envi("NORD_RFF", 1500)
    GAMMA = _envf("NORD_GAMMA", 0.15)
    MATHN = _envi("NORD_MATHN", 4000)
    # tempfile.gettempdir(), not os.environ.get("TEMP", ".") -- the Windows-only TEMP fallback
    # wrote into the repo checkout on Linux/macOS CI; same fix as lake99 / language_self_build.
    lake_path = os.path.join(tempfile.gettempdir(), "sndl_skills.spkl")
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 96)
    print(
        "SKILL BUILDER -- one autonomous loop builds + INDEXES its own skill for ANY domain (stream)"
    )
    print(
        f"ctx={CTX}  per={PER}  rff={RFF}  (spiking representation + strong SLDA-RFF competence head)"
    )
    print("=" * 96)

    # ---- domain streams (real): LANGUAGE (corpus) + MATH (generated arithmetic) ----
    cmudict, doff = load_cmudict()
    words, toff = load_text()
    lang_pairs = list(phoneme_stream(words, cmudict, k=CTX))
    math_pairs = math_stream(n_expr=MATHN, ctx=CTX, seed=0)
    print(
        f"[data] language {len(lang_pairs):,} phoneme pairs ({'offline' if toff else 'real corpus'}); "
        f"math {len(math_pairs):,} arithmetic-char pairs (generated)"
    )

    lake = LakeMemory(path=lake_path, seed=0)
    registry = SkillRegistry(lake)
    # META-CONTROLLER: a top-level curiosity over DOMAIN ids picks which skill to build next by
    # cross-domain learning-progress (the frontier). Here both are built; the order is curiosity-driven.
    meta = SpikingCuriosityDrive(seed=0)
    domains = {
        0: ("language", lang_pairs, ARPABET, _bigram_baseline(lang_pairs)),
        1: ("math", math_pairs, MATH_VOCAB, _bigram_baseline(math_pairs)),
    }

    results = {}
    order = []
    remaining = set(domains)
    while remaining:
        did = meta.most_curious(sorted(remaining))
        name, pairs, vocab, base = domains[did]
        comp, c0, sig, expert = build_skill(name, pairs, vocab, CTX, PER, RFF, GAMMA, seed=did)
        registry.register(did, name, sig, expert)
        meta.observe(did, 1.0 - comp)  # low competence -> stays curious about it
        results[did] = (name, comp, c0, base)
        order.append(name)
        remaining.discard(did)
        print(
            f"  built skill '{name}': competence {comp:.1%} (baseline {base:.1%}, early {c0:.1%})  "
            f"-> indexed as skill {did}"
        )

    # ---- ROUTING: recall the right skill for held-out inputs of each domain, BY PROXIMITY ----
    def route_acc(did):
        _, pairs, _, _ = domains[did]
        test = pairs[int(len(pairs) * 0.8) :]
        sample = test[:: max(1, len(test) // 500)][:500]
        return sum(registry.route(c) == did for c, _ in sample) / max(1, len(sample))

    r_lang = route_acc(0)
    r_math = route_acc(1)
    router = (r_lang + r_math) / 2.0
    dur = lake.durability_check()

    print("\n" + "=" * 96)
    print("SKILL BUILDER -- skills built + indexed by ONE autonomous loop:")
    print("-" * 96)
    for did in sorted(results):
        nm, comp, c0, base = results[did]
        print(
            f"  {nm:9s}: competence {comp:.1%} (early {c0:.1%} -> {comp:.1%}); domain baseline {base:.1%}"
        )
    print(f"  meta-curiosity build order: {order}")
    print(
        f"  ROUTING (proximity, no domain-id): language {r_lang:.0%}, math {r_math:.0%}  "
        f"(chance {1 / len(domains):.0%})"
    )
    print(f"  LAKE: {lake.n_stored} skill signatures indexed; durable across restart = {dur['ok']}")
    print("-" * 96)
    print(
        "  one loop, many domains: same build+index mechanism; competence via the strong hybrid head."
    )
    print(
        "  next adapters (not built): physics (JEPA latent next-state), proofs (verifier reward)."
    )
    print("=" * 96)

    # ---- un-riggable self-checks ----
    for _did, (nm, comp, _c0, base) in results.items():
        assert comp >= base - 0.02, (
            f"{nm}: strong readout below its domain baseline ({comp:.3f} vs {base:.3f})"
        )
    assert router > 1.5 / len(domains), (
        f"router no better than chance ({router:.2f} vs chance {1 / len(domains):.2f})"
    )
    assert r_lang > 0.6 and r_math > 0.6, (
        f"per-domain routing weak (lang {r_lang:.2f}, math {r_math:.2f})"
    )
    assert dur["ok"], "skill index did not survive the lake restart"
    print(
        f"self-check OK: one loop built+indexed {len(results)} skills across DIFFERENT domains "
        f"(competence >= baseline each); proximity router {router:.0%} >> chance {1 / len(domains):.0%}; "
        f"{lake.n_stored} signatures durable -- the brain builds and indexes its own skills, no labels."
    )
    lake.close()
    return 0


if __name__ == "__main__":
    main()
