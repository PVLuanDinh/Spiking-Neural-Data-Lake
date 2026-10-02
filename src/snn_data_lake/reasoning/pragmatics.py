"""
Pragmatics / speech-acts -- the layer between LITERAL form and SPEAKER meaning.

The lake's language/social stack took every utterance LITERALLY: spike_chat_lake.converse() routed on
surface intent, rose_language parsed constituency, theory_of_mind tracked WHERE-beliefs -- but nothing
recovered what a speaker MEANT beyond what they SAID. "Can you pass the salt?" was a yes/no question
about ability; "it's cold in here" was a weather report. This module adds the missing primitive: a
functional pragmatics head that (a) classifies the SPEECH ACT of an utterance, (b) takes a Gricean
IMPLICATURE step to the intended (non-literal) meaning, and (c) detects INDIRECT requests.

WHY THEORY OF MIND. Pragmatics is exactly the SPEAKER-meaning != SENTENCE-meaning gap, and that gap is
the SAME shape as ToM's false-belief gap: a believed state that DIVERGES from the surface/world state.
So the implicature step reuses faculties/theory_of_mind.TheoryOfMind as its divergence engine -- two
readings tile two "locations" (0 = LITERAL, 1 = INTENDED); a literal hearer's reading stays at the
world/surface (location 0), while the SPEAKER's communicative goal, when a maxim is flouted, MOVES to
the intended reading (location 1). implicature() returns the non-literal reading ONLY when the ToM
readout confirms belief != world -- the divergence is the switch, not decoration (graceful stdlib
fallback if the ToM organ is unavailable).

LEARNING (brain-native, NO gradient). Speech-act classification is ALSO learnable from labelled
examples by a VSA / rank-order HEBBIAN rule (reusing faculties/vsa_reasoning spike-polarity codes): each
utterance's surface cues become atomic spike codes, rank-order permuted and BUNDLED (superposition)
into one utterance code; a class PROTOTYPE is the Hebbian bundle of its members' codes; classify by
spike COINCIDENCE (argmax). No loss.backward(), no gradient anywhere -- a local superposition rule.
The self-check trains it and shows it GENERALISES to unseen utterances, while the same rule trained on
SHUFFLED labels collapses to chance (the learning is in the label->cue association, not the substrate).

The heuristic surface rules in act() are the deterministic backbone the brain calls; the VSA learner is
the evidence the same map is acquirable brain-natively.

Honest framing. This is a HEURISTIC maxim/pattern model over a SMALL cue set, NOT general pragmatic
competence -- no discourse model, no common-ground update, no open-ended inference. It nails the classic
textbook cases (indirect ability-request, environmental-relevance request, scalar 'some'->'not all') and
is honest (returns None) when nothing non-literal is licensed. Representation/value-side only: it selects
NO action and never touches the arm. ASCII, deterministic, per-instance seeded.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.pragmatics
"""

import random
import re

# spike-polarity VSA substrate (crc32-seeded spike codes) -- the ONLY hard import.
from snn_data_lake.faculties.vsa_reasoning import bundle, coincidence, hv, permute

# ------------------------------------------------------------------ speech-act taxonomy (Searle) ----
ASSERT = "assert"  # representative: states a proposition            ("the cat is on the mat")
ASK = "ask"  # directive-info: seeks information               ("what time is it?")
REQUEST = "request"  # directive-polite/indirect: asks the hearer to act ("can you pass the salt?")
COMMAND = "command"  # directive-imperative: orders the hearer to act  ("close the door")
PROMISE = "promise"  # commissive: commits the speaker to a future act ("i will call you")
EXPRESS = "express"  # expressive: vents a psychological state         ("thank you", "i'm sorry")
_ACTS = (ASSERT, ASK, REQUEST, COMMAND, PROMISE, EXPRESS)

# ------------------------------------------------------------------ small cue lexicons --------------
# request-action verbs: the transitive here-and-now acts that make "can you V" a REQUEST, not a
# literal ability question ("can you pass..." is a request; "can you swim?" is not).
_ACTION_VERBS = {
    "pass",
    "hand",
    "give",
    "close",
    "open",
    "turn",
    "help",
    "send",
    "move",
    "bring",
    "take",
    "tell",
    "show",
    "get",
    "make",
    "fetch",
    "grab",
    "hold",
    "lower",
    "raise",
    "share",
    "lend",
    "put",
    "set",
    "remove",
    "fix",
    "repeat",
    "explain",
    "call",
    "email",
    "forward",
    "lift",
    "carry",
    "return",
    "check",
    "find",
    "write",
    "read",
    "play",
    "stop",
    "wait",
}
# intransitive / short commands that can head an imperative
_IMPER_EXTRA = {
    "go",
    "come",
    "sit",
    "stand",
    "listen",
    "look",
    "hurry",
    "relax",
    "calm",
    "run",
    "jump",
    "leave",
    "be",
    "let",
    "keep",
    "stay",
    "follow",
    "watch",
    "drop",
    "begin",
    "start",
    "finish",
}
_IMPER_VERBS = _ACTION_VERBS | _IMPER_EXTRA
_WH = {"what", "who", "when", "where", "why", "how", "which", "whose", "whom"}
_AUX = {
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "will",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "am",
    "may",
    "might",
    "shall",
}
_PROMISSIVE = {"promise", "swear", "guarantee", "vow", "pledge", "assure"}
_EXPRESSIVE = {
    "thank",
    "thanks",
    "thankful",
    "grateful",
    "appreciate",
    "sorry",
    "apologies",
    "apologize",
    "apologise",
    "congrats",
    "congratulations",
    "welcome",
    "wow",
    "ugh",
    "yay",
    "hooray",
    "bravo",
    "condolences",
}
_ENV_STATES = {
    "cold",
    "hot",
    "warm",
    "chilly",
    "freezing",
    "stuffy",
    "dark",
    "bright",
    "loud",
    "noisy",
    "quiet",
    "stale",
}
# situational cues that make a bare state-report RELEVANT (and thus an implied request)
_RELEVANCE_CUES = {
    "window",
    "door",
    "ac",
    "air",
    "conditioner",
    "heater",
    "heating",
    "vent",
    "thermostat",
    "light",
    "lamp",
    "fan",
    "radiator",
    "blind",
    "curtain",
}
# gerund -> base, so "would you mind CLOSING the door" paraphrases as "close the door"
_DEGERUND = {
    "closing": "close",
    "opening": "open",
    "passing": "pass",
    "handing": "hand",
    "helping": "help",
    "turning": "turn",
    "moving": "move",
    "sending": "send",
    "bringing": "bring",
    "giving": "give",
    "taking": "take",
    "telling": "tell",
    "showing": "show",
    "lowering": "lower",
    "raising": "raise",
    "waiting": "wait",
    "holding": "hold",
    "getting": "get",
    "making": "make",
    "sharing": "share",
}


def _tok(s):
    return re.findall(r"[a-z']+", s.lower())


def _action_bases(token):
    """Candidate lemmas for a possibly-inflected verb token (pass/passes/passed/passing -> pass)."""
    forms = {token}
    if token.endswith("s") and len(token) > 3:
        forms.add(token[:-1])
    if token.endswith("es") and len(token) > 4:
        forms.add(token[:-2])
    if token.endswith("ed") and len(token) > 4:
        forms.add(token[:-2])
        forms.add(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        forms.add(token[:-3])
        forms.add(token[:-3] + "e")
    return forms


def _has_action_verb(tokens):
    for t in tokens:
        if _action_bases(t) & _ACTION_VERBS:
            return True
    return False


class Pragmatics:
    """Heuristic speech-act / implicature head over a small surface-cue set.

    Public interface the brain calls:
      act(utterance)                 -> one of ASSERT/ASK/REQUEST/COMMAND/PROMISE/EXPRESS
      implicature(utterance, context)-> intended non-literal meaning (str) | None if literal
      is_indirect_request(utterance) -> bool

    Deterministic, per-instance seeded, representation-only (selects no action, touches no arm)."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)  # per-instance RNG (controls only)

    # =============================================================== (a) SPEECH-ACT CLASSIFICATION ==
    def act(self, utterance):
        """Classify the illocutionary force of `utterance` from surface cues. Deterministic rule
        backbone; order encodes precedence (an indirect request outranks the bare question it wears)."""
        raw = str(utterance).strip()
        low = raw.lower()
        body = low.rstrip("?.! ")
        toks = _tok(body)
        if not toks:
            return ASSERT
        first = toks[0]

        # 1) REQUEST -- politeness / indirect directive (checked first: it wears a question's clothes)
        if self._looks_request(body, toks):
            return REQUEST
        # 2) PROMISE -- first-person commissive ("i promise", "i will help", "we swear to ...")
        if first in ("i", "we"):
            if (set(toks) & _PROMISSIVE) or ("will" in toks) or ("'ll" in low) or ("ll" in toks):
                # a first-person future commitment, not a second-person question
                if "you" not in toks[:2]:
                    return PROMISE
        # 3) EXPRESS -- an expressive marker leads and no directive/interrogative outranks it
        if (set(toks) & _EXPRESSIVE) or low.startswith(("thank", "sorry", "congrat")):
            return EXPRESS
        # 4) COMMAND -- imperative: leads with a base verb, not a question
        if ("?" not in raw) and (first in _IMPER_VERBS):
            return COMMAND
        # 5) ASK -- interrogative: question mark, wh-word, or aux-inversion
        if ("?" in raw) or (first in _WH) or (first in _AUX):
            return ASK
        # 6) ASSERT -- declarative default
        return ASSERT

    def _looks_request(self, body, toks):
        """True when the surface functions as a REQUEST: explicit politeness, 'would you mind ...',
        a modal-you frame over a request-action verb, or a first-person 'i want/need you to ...'."""
        if "please" in toks:
            return True
        if body.startswith("would you mind"):
            return True
        if self._is_wondering(body):
            return True
        if self._modal_you_request(body):
            return True
        if re.match(r"^(i|we)\s+(want|need|would like|'d like)\b.*\byou\b", body):
            return True
        return False

    def _modal_you_request(self, body):
        """'can/could/would/will you [please] <request-action verb> ...' -> an indirect request.
        The action-verb gate is what separates 'can you pass the salt' (request) from 'can you swim'."""
        m = re.match(r"^(can|could|would|will)\s+you\b(.*)$", body)
        if not m:
            return False
        rest = _tok(m.group(2))
        if rest and rest[0] == "please":
            rest = rest[1:]
        return _has_action_verb(rest)

    @staticmethod
    def _is_wondering(body):
        return (
            body.startswith("i was wondering if")
            or body.startswith("i wonder if")
            or body.startswith("do you think you could")
            or body.startswith("any chance you could")
        )

    # =============================================================== (c) INDIRECT-REQUEST DETECTOR ==
    def is_indirect_request(self, utterance):
        """True iff the utterance's SURFACE form is not a plain imperative request yet functions as
        one: a modal-you ability/willingness question over an action verb, 'would you mind ...', an
        'i was wondering if ...' hedge, or a conventional environmental state-report ('it's cold in
        here'). A literal yes/no question ('is it raining?', 'can you swim?') is NOT indirect."""
        body = str(utterance).strip().lower().rstrip("?.! ")
        if not body:
            return False
        if self._modal_you_request(body):
            return True
        if body.startswith("would you mind"):
            return True
        if self._is_wondering(body):
            return True
        if self._env_state(body) is not None:
            return True
        return False

    # =============================================================== (b) GRICEAN IMPLICATURE ========
    def implicature(self, utterance, context=None):
        """Infer the intended meaning BEYOND the literal, or None if the utterance is literal.

        Maxim heuristics:
          * MANNER  -- an ability/willingness question over an action verb flouts 'be brief' (its literal
                       answer is trivially yes) -> it is a REQUEST to act, not a yes/no question.
          * RELATION-- a bare state-report ('it's cold in here') is irrelevant unless it targets the
                       situation; WITH a relevance cue in `context` (a window/heater/...) it implicates
                       a request to fix it. WITHOUT such context it is just an assertion -> None.
          * QUANTITY-- 'some X' scalar-implicates 'not all X' (the speaker would have said 'all' if true).

        The ToM engine gates the return: literal reading = 0, intended = 1; the speaker's goal MOVES to
        1 only when a maxim is flouted, so a non-None reading is returned iff belief(speaker) != world."""
        raw = str(utterance).strip()
        body = raw.lower().rstrip("?.! ")
        ctx = "" if context is None else str(context).lower()
        intended = None

        # MANNER: indirect ability/willingness request -> the act itself is the meaning
        if body.startswith("would you mind"):
            action = self._degerund_phrase(body[len("would you mind") :].strip())
            intended = f"request to {action} -- 'would you mind' softens a directive into a request"
        elif self._modal_you_request(body):
            action = self._strip_modal(body)
            intended = (
                f"request to {action} -- an indirect request to ACT, "
                "not a yes/no question about ability"
            )
        elif self._is_wondering(body):
            action = self._strip_wondering(body)
            intended = f"request to {action} -- a hedged ('i was wondering') indirect request"
        else:
            # RELATION: environmental state-report needs a relevance cue in context to become a request
            state = self._env_state(body)
            if state is not None and self._relevance_cue(ctx):
                intended = (
                    f"implied request: {self._env_action(state, ctx)} "
                    "-- a bare temperature/state report would be irrelevant here (Relation)"
                )
            elif re.search(r"\bsome\b", body) and "not all" not in body:
                # QUANTITY: scalar implicature on 'some'
                intended = "implicates 'not all' -- scalar implicature on 'some' (Quantity maxim)"

        nonliteral = intended is not None
        speaker_reading, hearer_reading = self._tom_reading(nonliteral)
        # divergence is the switch: a non-literal meaning is returned only when the SPEAKER's reading
        # departs from a naive literal HEARER's reading (the false-belief gap, speaker-meaning side).
        if speaker_reading == hearer_reading:
            return None
        return intended

    # ------------------------------------------------------------- implicature helpers --------------
    @staticmethod
    def _strip_modal(body):
        rest = re.sub(r"^(can|could|would|will)\s+you\s+(please\s+)?", "", body).strip()
        return rest or "act"

    @staticmethod
    def _strip_wondering(body):
        rest = re.sub(r"^(i was wondering if|i wonder if|do you think|any chance)\s+", "", body)
        rest = re.sub(r"^(you\s+)?(could|can|would|might|please)\s+", "", rest).strip()
        return rest or "act"

    @staticmethod
    def _degerund_phrase(phrase):
        toks = _tok(phrase)
        if toks and toks[0] in _DEGERUND:
            toks[0] = _DEGERUND[toks[0]]
        return " ".join(toks) or "act"

    @staticmethod
    def _env_state(body):
        """Return the state adjective of an 'it's <state> in here' report, else None."""
        m = re.search(r"\bit'?s\s+(\w+)\s+in here\b", body) or re.search(
            r"\bit is\s+(\w+)\s+in here\b", body
        )
        if m and m.group(1) in _ENV_STATES:
            return m.group(1)
        # also 'i'm cold' style
        m2 = re.match(r"^(i'?m|i am)\s+(\w+)$", body)
        if m2 and m2.group(2) in _ENV_STATES:
            return m2.group(2)
        return None

    @staticmethod
    def _relevance_cue(ctx):
        return bool(set(_tok(ctx)) & _RELEVANCE_CUES) or ("open" in ctx) or ("on" in _tok(ctx))

    @staticmethod
    def _env_action(state, ctx):
        toks = set(_tok(ctx))
        if "window" in toks:
            return (
                "close the window" if state in ("cold", "chilly", "freezing") else "open the window"
            )
        if state in ("cold", "chilly", "freezing"):
            return "raise the temperature (close the window / turn up the heat)"
        if state in ("hot", "warm", "stuffy"):
            return "lower the temperature (open a window / turn on the fan)"
        if state == "dark":
            return "turn on the light"
        if state in ("loud", "noisy"):
            return "reduce the noise"
        return "address the condition"

    def _tom_reading(self, nonliteral):
        """Run the ToM divergence via the false-belief mechanic: both parties start on the LITERAL
        reading (location 0); when a maxim is flouted the SPEAKER's communicative goal MOVES to the
        intended reading (location 1) with only the speaker 'present', so a naive literal HEARER's
        belief PERSISTS at 0 (a false belief about what was meant). Returns (speaker_reading,
        hearer_reading); they diverge exactly when a non-literal meaning is intended. Falls back to a
        direct model if the ToM organ is unavailable."""
        try:
            from snn_data_lake.faculties.theory_of_mind import TheoryOfMind

            tom = TheoryOfMind(n_locations=2, seed=self.seed)
            tom.place(0, observers=("speaker", "hearer"))  # both start on the literal reading
            if nonliteral:
                tom.move(
                    1, observers=("speaker",)
                )  # speaker-meaning moves; literal hearer stays at 0
            return tom.predict_from_belief("speaker"), tom.predict_from_belief("hearer")
        except Exception:
            return (1 if nonliteral else 0), 0


# ==================================================================== brain-native VSA learner ======
# fixed canonical cue-feature order (rank-order coding permutes each present feature by its rank)
_FEATURES = (
    "MOOD_DECL",
    "MOOD_INTERROG",
    "MOOD_DIRECTIVE",
    "HAS_Q",
    "WH_INIT",
    "AUX_INIT",
    "MODAL_YOU",
    "PLEASE",
    "MIND",
    "WONDERING",
    "PROMISSIVE",
    "EXPRESSIVE",
    "IMPER_INIT",
    "ACTION_VERB",
    "FIRST_PERSON",
    "ENV_STATE",
)


def _cue_features(utterance):
    """Extract the present surface cues (coarse mood tags + fine markers) for the VSA encoder."""
    raw = str(utterance).strip()
    low = raw.lower()
    body = low.rstrip("?.! ")
    toks = _tok(body)
    if not toks:
        return []
    first = toks[0]
    p = Pragmatics()
    feats = []
    is_q = ("?" in raw) or (first in _WH) or (first in _AUX)
    modal_you = p._modal_you_request(body)
    imper_init = ("?" not in raw) and (first in _IMPER_VERBS)
    directive = modal_you or imper_init or ("please" in toks) or body.startswith("would you mind")
    if is_q:
        feats.append("MOOD_INTERROG")
    elif directive:
        feats.append("MOOD_DIRECTIVE")
    else:
        feats.append("MOOD_DECL")
    if "?" in raw:
        feats.append("HAS_Q")
    if first in _WH:
        feats.append("WH_INIT")
    if first in _AUX:
        feats.append("AUX_INIT")
    if modal_you:
        feats.append("MODAL_YOU")
    if "please" in toks:
        feats.append("PLEASE")
    if body.startswith("would you mind"):
        feats.append("MIND")
    if p._is_wondering(body):
        feats.append("WONDERING")
    if set(toks) & _PROMISSIVE or "will" in toks:
        feats.append("PROMISSIVE")
    if set(toks) & _EXPRESSIVE:
        feats.append("EXPRESSIVE")
    if imper_init:
        feats.append("IMPER_INIT")
    if _has_action_verb(toks):
        feats.append("ACTION_VERB")
    if first in ("i", "we"):
        feats.append("FIRST_PERSON")
    if p._env_state(body) is not None:
        feats.append("ENV_STATE")
    return feats


def _encode(utterance):
    """Rank-order spike code for an utterance: each present cue -> its atomic spike code, PERMUTED by
    the cue's canonical rank (axonal-delay order code), then BUNDLED (spike superposition) into one."""
    feats = _cue_features(utterance)
    if not feats:
        return hv("__EMPTY_UTTERANCE__")
    codes = [permute(hv(f), _FEATURES.index(f)) for f in feats]
    return bundle(codes)


class PragmaticsLearner:
    """VSA / rank-order HEBBIAN speech-act classifier -- learning WITHOUT a gradient.

    fit(examples): for each (utterance, label), add the utterance's rank-order spike code to the
    label's PROTOTYPE bundle (Hebbian superposition -- a local association, no error signal, no
    backprop). classify(utterance): the prototype with the highest spike COINCIDENCE wins. This is the
    evidence that the act map is acquirable brain-natively; shuffling the labels destroys the cue->class
    association and drops accuracy to chance (the substrate carries no prior)."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self._members = {a: [] for a in _ACTS}  # class -> list of member spike codes
        self._proto = {}  # class -> bundled prototype spike code

    def fit(self, examples):
        """examples: iterable of (utterance, label). Bundles each class's member codes (Hebbian)."""
        for utt, label in examples:
            if label not in self._members:
                self._members[label] = []
            self._members[label].append(_encode(utt))
        self._proto = {}
        for label, codes in self._members.items():
            if codes:
                self._proto[label] = bundle(codes) if len(codes) > 1 else codes[0]
        return self

    def classify(self, utterance):
        """Nearest class PROTOTYPE by spike coincidence; None until fit() has seen a class."""
        if not self._proto:
            return None
        code = _encode(utterance)
        return max(self._proto, key=lambda a: coincidence(code, self._proto[a]))


# ==================================================================== self-check ====================
def _labeled_trainset():
    return [
        ("the cat is on the mat", ASSERT),
        ("it is raining outside", ASSERT),
        ("the meeting starts at noon", ASSERT),
        ("what time is it", ASK),
        ("is it raining", ASK),
        ("do you like coffee", ASK),
        ("can you pass the salt", REQUEST),
        ("would you mind closing the door", REQUEST),
        ("please hand me the book", REQUEST),
        ("close the door", COMMAND),
        ("stop right now", COMMAND),
        ("give me the report", COMMAND),
        ("i promise i will call you", PROMISE),
        ("i will finish it tonight", PROMISE),
        ("i swear to help you", PROMISE),
        ("thank you so much", EXPRESS),
        ("i am so sorry", EXPRESS),
        ("congratulations on the win", EXPRESS),
    ]


def _labeled_testset():
    """Unseen utterances (different surface strings, same classes) -> a real generalisation test."""
    return [
        ("the sky is blue", ASSERT),
        ("dinner is ready", ASSERT),
        ("where is the station", ASK),
        ("are you coming", ASK),
        ("could you open the window", REQUEST),
        ("can you help me lift this", REQUEST),
        ("sit down now", COMMAND),
        ("hand over the keys", COMMAND),
        ("i will be there soon", PROMISE),
        ("we promise to return it", PROMISE),
        ("thanks a lot", EXPRESS),
        ("i am really grateful", EXPRESS),
    ]


def _accuracy(pred_label_pairs):
    hits = sum(1 for pred, gold in pred_label_pairs if pred == gold)
    return hits / len(pred_label_pairs) if pred_label_pairs else 0.0


def main():
    lines = []

    def emit(s):
        lines.append(s)

    prag = Pragmatics(seed=0)
    train = _labeled_trainset()
    test = _labeled_testset()
    chance = 1.0 / len(_ACTS)

    emit("=" * 78)
    emit("PRAGMATICS / SPEECH-ACTS -- literal form -> speaker meaning (heuristic + VSA-Hebbian)")
    emit("=" * 78)

    # (1) HEURISTIC SPEECH-ACT CLASSIFICATION + random-label control ---------------------------------
    emit("(1) SPEECH-ACT CLASSIFICATION (heuristic act())")
    heur = [(prag.act(u), g) for u, g in train]
    heur_acc = _accuracy(heur)
    for (u, g), (pred, _) in zip(train, heur):
        flag = "" if pred == g else "   <-- MISS"
        emit(f"    {u:<34} -> {pred:<8} (gold {g}){flag}")
    # CONTROL: assign each item a RANDOM label -> chance
    ctrl_rng = random.Random(0)
    rand_acc = _accuracy([(ctrl_rng.choice(_ACTS), g) for _u, g in train])
    emit(
        f"    heuristic accuracy = {heur_acc:.2f}   random-label control = {rand_acc:.2f} "
        f"(chance ~ {chance:.2f})"
    )
    assert heur_acc >= 0.9, "heuristic speech-act classifier must be well above chance"
    assert rand_acc <= heur_acc - 0.4, "random-label control must collapse to ~chance"

    # (2) BRAIN-NATIVE LEARNING (VSA rank-order Hebbian) + shuffled-label control --------------------
    emit("-" * 78)
    emit("(2) BRAIN-NATIVE LEARNING -- VSA rank-order Hebbian prototypes, NO gradient")
    learner = PragmaticsLearner(seed=0).fit(train)
    learned_acc = _accuracy([(learner.classify(u), g) for u, g in test])
    # CONTROL: same rule, SHUFFLED labels -> the cue->class association is destroyed (mean over seeds)
    shuf_accs = []
    for s in range(5):
        rng = random.Random(s)
        labels = [g for _u, g in train]
        rng.shuffle(labels)
        shuf = PragmaticsLearner(seed=0).fit([(u, labels[i]) for i, (u, _g) in enumerate(train)])
        shuf_accs.append(_accuracy([(shuf.classify(u), g) for u, g in test]))
    shuf_acc = sum(shuf_accs) / len(shuf_accs)
    emit(f"    learned test accuracy (unseen utterances) = {learned_acc:.2f}")
    emit(
        f"    shuffled-label control (mean/5 seeds)      = {shuf_acc:.2f}  (chance ~ {chance:.2f})"
    )
    emit("    -> learning lives in the label->cue Hebbian bundle, not the spike substrate")
    assert learned_acc >= 0.6, "VSA-Hebbian learner must generalise well above chance"
    assert shuf_acc <= learned_acc - 0.25, "shuffled-label control must collapse toward chance"

    # (3) INDIRECT-REQUEST DETECTION + literal-question control --------------------------------------
    emit("-" * 78)
    emit("(3) INDIRECT-REQUEST DETECTION")
    indirect = [
        "can you pass the salt",
        "would you mind closing the door",
        "i was wondering if you could open the window",
        "it's cold in here",
    ]
    literal_q = ["are you hungry", "is it raining", "do you like coffee", "can you swim"]
    for u in indirect:
        got = prag.is_indirect_request(u)
        emit(f"    indirect : {u:<44} -> {got}")
        assert got is True, f"indirect request not detected: {u}"
    for u in literal_q:
        got = prag.is_indirect_request(u)
        emit(f"    literal Q: {u:<44} -> {got}  (control: must be False)")
        assert got is False, f"literal question wrongly flagged as indirect request: {u}"

    # (4) GRICEAN IMPLICATURE + literal control + ToM divergence -------------------------------------
    emit("-" * 78)
    emit("(4) GRICEAN IMPLICATURE (non-literal meaning; None when literal)")
    cases = [
        ("can you pass the salt?", None, "MANNER: ability-Q used as a request"),
        ("would you mind closing the door?", None, "indirect directive"),
        ("it's cold in here", "the window is open", "RELATION: relevant only near the window"),
        ("some of the students passed the exam", None, "QUANTITY: scalar 'some'->'not all'"),
    ]
    for utt, ctx, note in cases:
        got = prag.implicature(utt, ctx)
        emit(f"    {utt!r:<42} [{note}]")
        emit(f"        -> {got}")
        assert got is not None, f"implicature should fire on: {utt}"
    # context is LOAD-BEARING: same utterance, no relevance context -> literal -> None
    no_ctx = prag.implicature("it's cold in here", context=None)
    emit(f"    'it's cold in here' with NO context -> {no_ctx}  (context is load-bearing)")
    assert no_ctx is None, "environmental request must need a relevance cue in context"
    # literal controls -> None
    for utt in ("the cat is on the mat", "is it raining?"):
        got = prag.implicature(utt, context=None)
        emit(f"    literal  : {utt!r:<40} -> {got}  (control: must be None)")
        assert got is None, f"literal utterance must yield no implicature: {utt}"
    # ToM divergence is the switch: speaker vs literal-hearer diverges on an implicature, agrees on a literal
    s1, h1 = prag._tom_reading(nonliteral=True)
    s0, h0 = prag._tom_reading(nonliteral=False)
    emit(
        f"    ToM readout: implicature speaker={s1} hearer={h1} (diverge)  "
        f"literal speaker={s0} hearer={h0} (agree)"
    )
    assert s1 != h1 and s0 == h0, "ToM must diverge on speaker-meaning and agree on the literal"

    emit("=" * 78)
    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "HONEST NEGATIVE: this is a HEURISTIC maxim/pattern model over a SMALL cue set, NOT general "
        "pragmatic competence -- no discourse model, no common-ground update, no open-ended inference. "
        "It nails the classic textbook cases and honestly returns None when nothing non-literal is "
        "licensed. Representation-only: selects no action, touches no arm. Deterministic, ASCII."
    )
    print(
        f"self-check OK: heuristic speech-acts {heur_acc:.2f} vs random-label {rand_acc:.2f}; "
        f"VSA-Hebbian learning generalises {learned_acc:.2f} vs shuffled-label {shuf_acc:.2f} (chance "
        f"{chance:.2f}); indirect requests detected while literal yes/no questions are NOT; implicature "
        "returns the non-literal meaning on the classic cases (ToM-gated) and None when literal"
    )


if __name__ == "__main__":
    main()
