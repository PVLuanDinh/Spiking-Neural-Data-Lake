r"""
SPIKE CHAT LAKE -- converse from the brain's OWN self-built, self-enriching knowledge lake.

The correction to spike_chat.py: that was a fixed 6-intent classifier with canned replies -- it does
NOT develop or grow. The developmental thesis is that the LOOP builds its own knowledge into the
lifelong LAKE (the content-addressable "database"), and conversing is RETRIEVING from + ENRICHING
that store, so the brain's conversational range GROWS as it learns. The lake IS the LLM-like database
-- durable, proximity-indexed, self-grown -- not an imported transformer.

  perceive text -> signature key -> RETRIEVE nearest learned knowledge from the lake
                -> if close enough: IGNITE + answer from what it learned
                -> if not: honest gap ("I haven't learned that") -> a CURIOSITY target to learn
  LEARN         teach(question, answer) or ingest(facts) -> CONSOLIDATE to the lake -> the DB GROWS,
                so the SAME question is answered next time. The developmental loop's outputs
                (world-KB facts it learned, conversation turns) are what seed + enrich it.

HONEST SCOPE (this is the point): this is RETRIEVAL/COMPOSITION over a spiking-built knowledge lake,
NOT generative fluency. It answers with what it has stored, matches on shared content words, and grows
with ingestion. It will not paraphrase or reason like a transformer LM. The value is a knowledge base
the brain BUILDS ITSELF and converses from -- and that measurably grows as it learns.

RUN IT (no hardcoded path -- the module names itself, so relocating the file cannot make this lie):
  PYTHONPATH=src python -m snn_data_lake.<region>.spike_chat_lake   # `main()` prints its own real
  dotted name in the banner; a tty gets the REPL, anything else (CI, a pipe, pytest) gets the
  scripted demo + self-check. NORD_CHAT_DEMO=1 forces the demo, NORD_CHAT_REPL=1 forces the REPL.

DEPENDENCIES (P7, owner policy 2026-08-04 -- the zero-dependency rule is retired and is NOT a reason
to hand-roll anything). Judged per call site rather than by the retired rule:
  * `content_signature.signature` is stdlib **on the merits**, not as a concession: it is a 64-wide
    crc32 bag-of-tokens (SIG=64) evaluated once per learn() and once per query. numpy would add an
    array construction per call over 64 elements and be SLOWER; crc32 is also the persistence-stable
    hash this project mandates. What WAS a real residue of the retired rule -- a per-call
    `[float(x) for x in signature(...)]` "demotion" comment claiming it undid a numpy roundtrip --
    is deleted: `signature()` already returns a list of Python floats, so the comprehension was a
    provable no-op wearing a policy justification. The BOAST was the defect, not the code.
  * numpy/torch arrive through the collaborators (`lake_memory`, `semantic_memory`,
    `reasoning_cortex`), which own their own dependency choices; none is lazy-imported here to dodge
    a dependency budget, only to keep boot cheap on turns that never reach them.
Reuses content_signature.signature, lake_memory, world_kb_stream (seed facts),
global_workspace (ignition gate = retrieval confidence), consciousness_probe.
"""

import os
import re
import subprocess
import sys
import tempfile

from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.reasoning.consciousness_probe import (
    probe_access,
    probe_integration,
)
from snn_data_lake.reasoning.content_signature import (
    signature,  # 64-wide crc32 bag-of-tokens; stdlib on the merits (see the P7 note above)
)
from snn_data_lake.reasoning.lake_reason import Reasoner
from snn_data_lake.reasoning.train_of_thought import TrainOfThought
from snn_data_lake.reasoning.world_kb_stream import load_kb
from snn_data_lake.substrate.home import KnowledgeLog, banner, home
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

_STOP = {
    "what",
    "is",
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "are",
    "you",
    "do",
    "does",
    "i",
    "me",
    "my",
    "tell",
    "about",
    "was",
    "were",
    "who",
    "which",
    "how",
    "and",
    "it",
    "its",
    "for",
    "with",
    "that",
    "this",
    "there",
    "s",
}


# a BUILD request is an ACT, not a recall: a build verb followed by an OBJECT phrase ("build me a
# 3d model of a spaceship"). Bare "what is a model" has nothing after the verb -> not a build turn.
# imperative-anchored AND object-shaped: a BUILD request STARTS with the verb and is followed by an
# object determiner ("build me a X", "make a rocket") -- so "bees make honey" (declarative) and "make
# sure the door is locked" / "make cars in germany" (verb without an article object) are NOT swallowed.
_BUILD_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|pls\s+)?(?:build|make|design|model)\b\s+(?:me\s+|us\s+)?(?:a|an|the|some)\s+\S"
)

# a Gf (fluid-reasoning) turn is a PUZZLE to SOLVE by inducing a rule, not a fact to recall: a letter
# analogy ("abc:abd :: ijk"), or a number/letter SERIES to continue. Detected structurally so ordinary
# conversation is never hijacked (a bare pair of numbers in a sentence does not trigger).
_ANALOGY_RE = re.compile(r"\b([a-z]+)\s*:\s*([a-z]+)\s*:\s*:\s*([a-z]+)")
_ANALOGY_WORDS_RE = re.compile(r"\b([a-z]+)\s+is to\s+([a-z]+)\s+as\s+([a-z]+)\s+is to")
_INT_RE = re.compile(r"-?\d+")
# series cue WORDS only -- NOT "?": a bare "?" plus two integers ("what is 2 plus 2?") is an ordinary
# question, and routing it to the Gf inducer locked factual/arithmetic turns out of retrieval.
_GF_CUE = ("next", "after", "comes", "continue", "sequence", "series", "follow", "pattern")


# SYNONYM-TOLERANT relation matching for structured KB retrieval: a query word -> the canonical KB
# relation(s) it asks about, so "what STATE is nitrogen" hits the `phase` relation and "where does a
# penguin LIVE" hits `habitat`. Only a canonical that actually exists in the loaded KB is ever used, so
# adding a synonym never invents a relation. Kept small + factual; it does not fabricate answers.
_REL_SYNONYMS = {
    "continent": ("continent", "landmass"),
    "hemisphere": ("hemisphere",),
    "phase": ("phase", "state"),
    "category": ("category", "kind", "classification"),
    "class": ("class",),
    "habitat": (
        "habitat",
        "live",
        "lives",
        "lived",
        "living",
        "inhabit",
        "inhabits",
        "environment",
        "home",
    ),
    "diet": ("diet", "eat", "eats", "feed", "feeds", "food"),
    "color": ("color", "colour", "colored", "coloured"),
    "taste": ("taste", "tastes", "flavor", "flavour"),
    "capital": ("capital",),
    "symbol": ("symbol", "abbreviation"),
    "language": ("language", "speak", "speaks", "spoken", "speaking", "tongue"),
    "currency": ("currency", "money"),
    "order": ("order", "position", "rank"),
    "planet_type": ("planet_type", "type"),
    "blood": ("blood", "blooded", "bloodedness"),
    "reproduction": ("reproduction", "reproduce", "reproduces", "reproducing", "born", "birth"),
    "skeleton": ("skeleton", "skeletal"),
    "conducts": ("conduct", "conducts", "conductivity", "conductive"),
    "port": ("port",),
    "base": ("base", "radix"),
    "volatility": ("volatility", "volatile"),
    "protocol_type": ("protocol_type", "protocol"),
    "sides": ("sides", "side", "edges"),
    "sound": ("sound", "sounds", "noise"),
    "days": ("days", "day"),
    "months": ("months", "month"),
    "hours": ("hours", "hour"),
    "minutes": ("minutes", "minute"),
    "seconds": ("seconds", "second"),
    "parity": ("parity", "even", "odd"),
    "primality": ("primality", "prime", "composite"),
}
# reverse index: a query token -> the canonical KB relations it may denote (a token can map to several,
# e.g. "type" -> planet_type; membership in the loaded KB decides which actually applies per subject).
_SYN2CANON = {}
for _canon, _syns in _REL_SYNONYMS.items():
    for _s in _syns:
        _SYN2CANON.setdefault(_s, []).append(_canon)


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _content(text_or_tokens):
    """Content words (stop-words stripped) -- the retrieval keys, so a fact keyed on its content words
    matches a full question that contains them (bag-of-content-words QA, robust to filler words)."""
    toks = text_or_tokens if isinstance(text_or_tokens, list) else _tok(text_or_tokens)
    return [t for t in toks if t not in _STOP]


# a small CURATED set of REAL "reasoning facts" -- directional/causal (subject, relation, object) triples
# the contradiction reasoner (lake_reason) checks a claim against, so the brain can CORRECT a false claim
# ("the sun rises from the west") using a stored fact + its stored CAUSE. Kept tiny + factual, and
# re-derived each boot (not persisted) so it never bloats the durable log.
_REASONING_SEED = [
    (["sun", "rises"], "the sun rises in the east.", ["sun", "rises_in", "east"]),
    (["sun", "sets"], "the sun sets in the west.", ["sun", "sets_in", "west"]),
    (
        ["earth", "rotates"],
        "the earth rotates on its axis from west to east.",
        ["earth", "rotates", "west to east"],
    ),
    (["earth", "orbits"], "the earth orbits the sun.", ["earth", "orbits", "sun"]),
    # a real LINEAR fact-chain (each object is the next subject) so a TRAIN OF THOUGHT can walk it
    # multi-hop through the workspace: sun -> star -> fusion -> light -> life.
    (["sun", "star"], "the sun is a star.", ["sun", "is", "star"]),
    (["star", "fusion"], "a star shines by nuclear fusion.", ["star", "shines_by", "fusion"]),
    (["fusion", "light"], "nuclear fusion produces light.", ["fusion", "produces", "light"]),
    (["light", "life"], "sunlight enables life on earth.", ["light", "enables", "life"]),
]


# -------------------------------------------------------------------------------------------------
# P1 -- WHAT THE BRAIN SAYS was arbitrated by a hand-tuned ladder of ABSOLUTE salience numbers
# (0.95 / 0.9 / 0.6 / 0.55 / 0.5 / 0.45 / 0.30) submitted into the global workspace. The workspace's
# rheobase -- the drive current above which a content pool ignites -- is 0.55 and it lives in ANOTHER
# module (`thalamus/global_workspace.py`). Every number in that ladder only ever meant "0.55 plus or
# minus something": 0.30 meant SUBLIMINAL, 0.9 meant IGNITES, 0.45 + 0.3*score was reverse-engineered
# so a chat memory crossed 0.55 at score 0.34. That is a cross-module FAIL-GREEN: move
# GlobalWorkspace's threshold and this file's honest-gap path silently becomes a confident answer with
# nothing going red, because no assert here mentions 0.55 at all.
#
# The ladder is now MARGINS on the rheobase READ OFF the workspace this chat actually ignites through
# (`LakeChat._new_workspace`, the single construction `_ignite` uses). What stays fixed is the margin,
# and it is labelled against P1 honesty-guard exception (a) -- a dimensionless offset on a NORMALIZED
# axis: workspace drive is a current in [0, 1] and these encode the PRECEDENCE of the brain's evidence
# sources (reasoning outdrives recall outdrives association), which is a structural design decision,
# not a tunable gain. The sign now carries the meaning: a NEGATIVE margin is subliminal BY
# CONSTRUCTION and can never accidentally ignite.
_DRIVE_MARGIN = {
    # supra-threshold -- these reach conscious access
    "correction": +0.40,  # reasoning over stored facts outdrives every recall, so the brain
    #                       CORRECTS a false claim instead of parroting the nearest fact
    "recall_covered": +0.35,  # a fact that is on-topic AND answers what was actually asked
    "recall_assoc": +0.05,  # content-addressable spreading activation (spiking VP proximity)
    # at rheobase exactly -- an IMPLIED (non-literal) meaning is offered but never outbids evidence
    "pragmatics": +0.00,
    # sub-threshold FLOORS -- these ignite only once evidence lifts them over the rheobase
    "semantic_floor": -0.05,  # + _EVIDENCE_GAIN * cosine similarity of a paraphrase
    "chat_floor": -0.10,  # + _EVIDENCE_GAIN * on-topicness of a remembered conversation turn
    # subliminal by construction -- a match that failed the coverage gate must NOT ignite
    "recall_weak": -0.25,
}
# How much measured evidence (a similarity / on-topicness fraction, both normalized to [0, 1]) is
# allowed to lift a floor. P1 exception (a): a dimensionless slope from one normalized axis to the
# other. It is the ONLY route from a floor to ignition, which is why the floors can be negative.
_EVIDENCE_GAIN = 0.30


class LakeChat:
    """A conversational KNOWLEDGE LAKE: signature(content) -> learned answer, retrieved by proximity and
    enriched every time it is taught. The lake is the durable store; a parallel signature list is the
    fast retrieval index (both grow together)."""

    def __init__(self, seed=0, coverage=0.99, min_query_cov=0.6, persist=True, route_all=False):
        # P1 exception (a), labelled rather than carried silently across the tier boundary: `coverage`
        # and `min_query_cov` are thresholds on cue_cov / query_cov, which are COVERAGE FRACTIONS in
        # [0, 1] (|cue & query| / |cue| and / |query|) -- dimensionless fixed points on a normalized
        # output, and constructor arguments a caller may override per brain.
        self.coverage = coverage  # known iff the query covers ALL a fact's content words
        self.min_query_cov = min_query_cov  # ...AND the fact covers most of the QUESTION
        self.route_all = route_all  # relax the generic-retrieval gate: a reasonably
        #                                                            on-topic fact still ANSWERS instead of giving up
        #                                                            (structured KB lookup already runs unconditionally)
        # P1: the ignition current every candidate reply is measured against, read off the SAME
        # workspace `_ignite` competes in -- so the drives below cannot drift out of step with the
        # gate they are compared to. See `_DRIVE_MARGIN`.
        self._rheobase = self._new_workspace().ignition_threshold
        # The RUNNING BRAIN, when a host attaches one (`chat.brain = live_brain`). P8: in the first,
        # UNCOMMITTED pass of this same-day rewrite this was read as `getattr(self, "brain", None)`
        # in `_consciousness()` on a class that never sets a `brain` attribute anywhere -- so the
        # access-consciousness marker was UNMEASURABLE BY CONSTRUCTION and the brain reported
        # "UNMEASURED" for every caller, forever, while a comment at the call site presented that as
        # a virtue. (Committed history had the mirror-image defect, not this one: at HEAD the reply
        # was a tally of four markers and `known` was the literal `True`.) A silent default is not
        # an honest null: the
        # slot is declared here so the marker has a reachable measured state and a reachable
        # unmeasured state, and the self-check now constructs BOTH. Left None by default -- P3: this
        # is a slot a host may fill, NOT a wire this module forces.
        self.brain = None
        self.cues = []  # per-fact content-word SETS (retrieval keys)
        self.answers = []  # parallel learned answers
        # the chat's spiking retrieval index is a per-boot scratch .spkl, REBUILT each boot from the
        # durable knowledge below -- it is an index, not the memory, so starting it fresh loses nothing.
        # path=None -> LakeMemory mkstemps a UNIQUE file: fresh by construction (no manual remove), and
        # two chats/brains -- even in different PROCESSES -- never collide on one fixed temp path.
        self.lake = LakeMemory(path=None, seed=seed)  # spiking retrieval index (rebuilt per boot)
        self.semantic = None  # W6: VSA meaning-based recall (paraphrase-invariant)
        try:
            from snn_data_lake.reasoning.semantic_memory import SemanticStore

            self.semantic = SemanticStore(seed=seed)
        except Exception:
            self.semantic = None
        self._pragmatics = None  # W8: speech-act force + Gricean implicature
        try:
            from snn_data_lake.reasoning.pragmatics import Pragmatics

            self._pragmatics = Pragmatics(seed=seed)
        except Exception:
            self._pragmatics = None
        # the DURABLE "never forgets" store: a shared append-only knowledge log under the project home,
        # replayed at boot() so a fact taught in a PRIOR run (even another process) is remembered.
        # persist=False (the deterministic demo) keeps the chat fully in-RAM, touching no home files.
        self.log = KnowledgeLog() if persist else None
        # the ONE lake's chat turns (daemon-ingested conversation, src="chat:*") are served by a FAST
        # bag-of-words VIEW over the SAME lake -- consulted in converse() so LakeChat answers from what the
        # daemon learned, WITHOUT indexing thousands of noisy turns into the heavy spike/semantic fact index.
        self._chat_mem = None
        self._recall_trace = (
            None  # USE IT OR LOSE IT: bumped when a chat memory ignites; the daemon's
        )
        #                            sleep consolidation prunes the never-recalled learned records first.
        if persist:
            try:
                from snn_data_lake.substrate.chat_memory import DurableChatMemory

                self._chat_mem = DurableChatMemory(log=self.log)
                from snn_data_lake.hippocampus.lake_consolidation import RecallTrace

                self._recall_trace = RecallTrace()
            except Exception:
                self._chat_mem = None
                self._recall_trace = None
        # a host (e.g. the unified Brain) can own durability instead of self.log: set a hook that
        # persists each taught record to ITS store (knowledge.bin). Without this, a persist=False chat
        # inside the Brain would learn a taught fact only in RAM -- so a build's stored PROCESS, taught
        # via chat.teach, would be lost on restart. The hook writes to disk; teach() already learned it.
        self._persist_hook = None
        self.records = []  # facts carrying a triple (for the reasoner)
        self._reasoner = None  # lazy contradiction reasoner over self.records
        self._pending_gf = None  # a series awaiting the user's discriminating term
        self._kbidx = None  # cached (len(records), structured-triple index)
        from snn_data_lake.reasoning.gf import OpenGf

        self._opengf = OpenGf(
            persist=persist, budget=8
        )  # OPEN-ENDED Gf: its grammar GROWS as it solves

    @property
    def n_known(self):
        return len(self.answers)

    def learn(self, cue_tokens, answer, triple=None):
        """Add one (content cue -> answer) association: enrich the database (and consolidate durably).
        An optional (subject, relation, object) triple also lets the contradiction reasoner check claims."""
        cue = set(_content(cue_tokens))
        # P7: this was `[float(x) for x in signature(...)]` with the comment "stdlib (was np
        # roundtrip)". `signature()` builds `v = [0.0]*64` and returns `[x / m for x in v]`, i.e.
        # Python floats already -- the comprehension was a no-op that a retired policy was cited to
        # justify. Measured identical over the full self-check corpus before removal.
        sig = signature(sorted(cue) or _content(cue_tokens) or ["_"])
        self.lake.consolidate([sig], payloads=[len(self.answers)])  # durable store
        self.cues.append(cue)
        self.answers.append(answer)
        if self.semantic is not None:  # W6: also index the answer by MEANING (paraphrase recall)
            try:
                self.semantic.remember(answer)
            except Exception:
                pass
        if triple is not None:
            toks = list(cue_tokens) if isinstance(cue_tokens, list) else _tok(cue_tokens)
            self.records.append({"cue": toks, "answer": answer, "triple": list(triple)})
            self._reasoner = None  # invalidate: a new fact may enable a correction

    def teach(self, cue_tokens, answer, triple=None):
        """Learn a fact AND persist it durably (the /learn path), so it survives a restart. In-memory
        learn() is used for re-derivable knowledge (world-KB, WordNet) and for replaying the durable
        log; teach() is used only for genuinely NEW knowledge that must never be forgotten."""
        self.learn(cue_tokens, answer, triple=triple)
        toks = cue_tokens if isinstance(cue_tokens, list) else _tok(cue_tokens)
        rec = {"cue": list(toks), "answer": answer}
        if triple is not None:
            rec["triple"] = list(triple)
        if self.log is not None:
            self.log.append(rec)  # durable, fsync'd, deduped
        elif self._persist_hook is not None:
            self._persist_hook(rec)  # host-owned durability (Brain's knowledge.bin)
        return self

    def ingest_kb(self, triples):
        """Seed from knowledge the LOOP learned: each (head, relation, tail) fact becomes a retrievable
        answer keyed on its content words AND a structured triple the reasoner can check claims against."""
        for h, rel, t in triples:
            self.learn([rel, h], f"{h}'s {rel} is {t}.", triple=[h, rel, t])

    def seed_reasoning(self):
        """Learn the curated reasoning facts (re-derived each boot, not persisted) so the brain can
        CORRECT false claims from stored facts + their stored causes (e.g. the sun/earth example)."""
        for cue, answer, triple in _REASONING_SEED:
            self.learn(cue, answer, triple=triple)
        return self

    def _get_reasoner(self):
        """Lazy contradiction reasoner over every stored fact that carries a triple (rebuilt on change)."""
        if self._reasoner is None:
            self._reasoner = Reasoner(self.records)
        return self._reasoner

    def _lake_associate(self):
        """Build an `associate(broadcast, chain)` over the lake's OWN knowledge graph: from a concept,
        the next candidate thoughts are the objects of stored triples with that concept as subject.
        A train of thought is thus a walk over the brain's stored facts, each hop gated by ignition."""
        adj = {}
        for rec in self.records:
            s, _rel, o = rec["triple"]
            adj.setdefault(" ".join(_tok(s)), []).append((" ".join(_tok(o)), 0.9))

        def associate(broadcast, chain):
            seen = set(chain)
            return [(o, sal) for o, sal in adj.get(broadcast, []) if o not in seen]

        return associate

    def think(self, seed, max_steps=8):
        """A SERIAL RE-ENTRANT train of thought over the lake, through the spiking GLOBAL WORKSPACE:
        ignite the seed -> broadcast -> associate to the next concept -> chain, HELD INTERNALLY (the
        open-ended-thought capability a feedforward J-space lacks -- it would have to externalize).
        Returns (chain, feedforward_control): the re-entrant chain vs the one-shot control."""
        assoc = self._lake_associate()
        seed_key = " ".join(_tok(seed))
        tot = TrainOfThought(seed=0)
        chain = tot.think(seed_key, assoc, max_steps=max_steps)
        ff = tot.think_feedforward(seed_key, assoc)
        return chain, ff

    def _build_request(self, text):
        """'build/make/design/model X' -> ACT, don't recall: think the assembly through (a train of
        thought over the recipe), ACQUIRE the parts from the procedural catalog, write the .obj as a
        FILE artifact, and store the PROCESS durably -- never the mesh (an artifact is not knowledge).
        Returns the reply dict, or None when the turn is not a build request."""
        low = text.lower()
        if _BUILD_RE.search(low) is None:
            return None
        words = low.split()
        # a QUESTION about building ("how do i build a spaceship?") asks for the stored PROCESS ->
        # fall through to recall; only an imperative request is an ACT that builds again.
        if low.rstrip().endswith("?") or (
            words
            and words[0] in ("what", "how", "why", "who", "when", "where", "does", "is", "are")
        ):
            return None
        from snn_data_lake.reasoning.model_builder import (  # lazy: geometry only on a build turn
            RECIPES,
            ModelBuilder,
        )

        target = next((t for t in RECIPES if t in low), None)
        if target is None:
            return {
                "known": False,
                "coverage": 0.0,
                "ignited": False,
                "source": "builder",
                "reply": (
                    f"[honest gap] i do not know how to build that -- my part catalog only "
                    f"has recipes for: {', '.join(sorted(RECIPES))}. teach me its parts "
                    "and i will try."
                ),
            }
        builder = ModelBuilder()
        mesh, labels, chain = builder.build(target)
        out_dir = os.path.join(home(), "models") if self.log is not None else tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        path = builder.save_obj(mesh, os.path.join(out_dir, f"{target}.obj"))
        for cue, answer, triple in builder.knowledge(target, labels, path):
            self.teach(cue, answer, triple=triple)  # the PROCESS is the durable knowledge
        verts, faces = mesh
        return {
            "known": True,
            "coverage": 1.0,
            "ignited": True,
            "source": "builder",
            "artifact": path,
            "reply": (
                f"[built] i thought it through ({' -> '.join(chain)}), acquired "
                f"{len(labels)} parts from my procedural catalog, and wrote {path} "
                f"({len(verts)} vertices, {len(faces)} faces). i stored HOW to build it "
                "(the process), not the mesh -- honest scope: procedural primitives, "
                "not a learned 3d model."
            ),
        }

    def _gf_reply(self, statement, answer, descr):
        """Wrap a fluid-reasoning result. A found answer is a SOLVED puzzle (ignited, source 'gf') and
        `descr` NAMES the rule synthesised for it; None is an HONEST GAP -- the structure was
        recognised but no rule in the inducible family fit, so it refuses to bluff a value."""
        if answer is not None:
            return {
                "known": True,
                "coverage": 1.0,
                "ignited": True,
                "source": "gf",
                "reply": (
                    f"[induced] {statement} -> {answer}. the rule i synthesised for this "
                    f"problem at test time: {descr}. not a stored fact."
                ),
            }
        return {
            "known": False,
            "coverage": 0.0,
            "ignited": False,
            "source": "gf",
            "reply": (
                f"[honest gap] i recognise a {descr} but no program in my DSL fits it. i "
                "synthesise polynomial/geometric/recurrence/periodic/interleaved rules, letter "
                "analogy (incl. novel alphabets), and Raven distribution-of-three -- this is "
                "outside that bounded search. no bluff."
            ),
        }

    def _resume_experiment(self, text, pending):
        """The other half of the active-learning loop: if a Gf experiment is pending and the user
        replies with the discriminating term, APPEND it to the series and re-run the search -- the extra
        term prunes the tied hypotheses and IDENTIFIES the rule (or honestly reports no-rule). This
        makes 'tell me the next term and i will identify the rule' actually happen; the user is the
        world oracle. Returns a reply dict, or None if this turn is NOT a bare-number reply to a pending
        experiment -- so 'what is 42' with an experiment armed is answered normally, not hijacked."""
        if not pending:
            return None
        nums = re.findall(r"-?\d+", text)
        non_numeric = [t for t in _tok(text) if not t.lstrip("-").isdigit()]
        if (
            len(nums) != 1 or non_numeric or len(nums[0].lstrip("-")) > 18
        ):  # a BARE number only (not a question)
            return None
        extended = pending + [int(nums[0])]  # _number_reply clears/re-arms self._pending_gf
        out = self._number_reply(extended)
        if out.get("known"):
            out["reply"] = f"[identified -- your term {nums[0]} pinned it] " + out["reply"]
        elif out.get("source") == "gf-active":
            out["reply"] = f"[still underdetermined even with {nums[0]}] " + out["reply"]
        else:
            out["reply"] = f"[with {nums[0]} added, no rule in my search fits] " + out["reply"]
        return out

    def _number_reply(self, ints):
        """The whole numeric-Gf verdict in ONE program search (not the old ask-then-solve double/triple
        enumeration). Order: (1) a robust LABELED family (arithmetic/geometric/recurrence/period/
        interleave) -> the rule is pinned, ANSWER. (2) else ONE `induce_program_full` (early-exit):
        None -> no program fits -> GAP; a non-compressing fit (nodes >= #terms, e.g. primes) -> ASK an
        experiment; a compressing fit -> check ONLY for a second disagreeing program (early-exit at 2
        distinct) + closed-form 2-3-point ambiguity, ASK if underdetermined else ANSWER. `consistent_
        programs` (the expensive exhaustive walk) is reached only for a compressing fit, and stops at 2
        distinct -- so the common patternless turn costs one search, not three."""
        from snn_data_lake.reasoning.gf import (
            _geometric_next,
            _interleave_next,
            _periodic_additive_next,
            _poly_next,
            _recurrence_next,
        )

        self._pending_gf = None  # clear; re-armed below only if we ASK
        stmt = " ".join(map(str, ints))
        # OPEN-ENDED search FIRST, so it LEARNS from every series (even ones a labeled family answers) --
        # its grammar grows with the brain's solves (a learned primitive makes a harder rule reachable).
        # Learns as a side effect + persists. The overfit guard keeps short/random data an honest gap.
        full = self._opengf.solve(ints)  # (next, program, nodes) | None
        if full is not None and full[2] > len(ints):
            full = None
        # THINK HARDER on a bounded gap -- but ONLY when the series is long enough that a deeper program
        # could actually COMPRESS (a size b+1 program needs > b+1 terms to beat the data by MDL; on a short
        # series any deeper fit would be overfit, so escalating cannot help and is skipped -> gap stays fast).
        if full is None and len(ints) > self._opengf.budget + 1:
            # ONE unbounded-deepening step past the budget, capped for an interactive turn (max_seconds).
            # A genuine non-program still gaps under the guard -- an honest 'no fit under deeper budget/time',
            # NOT proof of absence (the in-the-limit lever, docs/THE_FUNDAMENTAL_WALL.md). Budget is restored.
            deeper = self._opengf.solve_unbounded(
                ints, step=1, max_budget=self._opengf.budget + 1, max_seconds=3.0
            )
            if deeper is not None and deeper[2] <= len(
                ints
            ):  # same MDL overfit guard on the deeper fit
                full = deeper

        for gen in (
            _poly_next,
            _geometric_next,
            _recurrence_next,
            _periodic_additive_next,
            _interleave_next,
        ):
            got = gen(ints)
            if got is not None:
                return self._gf_reply(
                    stmt, got[0], got[1]
                )  # a structural family fits -> clean label

        from snn_data_lake.reasoning.gf import Library, consistent_hypotheses, consistent_programs
        from snn_data_lake.reasoning.gf import _evaluate_al as _evaluate
        from snn_data_lake.reasoning.gf import _render_al as _render

        # a fit that compresses by >=2 nodes is strong Occam evidence -> trust it and SKIP the closed-form
        # hypotheses entirely (that scan is ~23 ms and would be discarded here anyway, e.g. factorial).
        strong = full is not None and full[2] < len(ints) - 1
        if strong:
            cf_hyps, cf_nexts, nexts = [], set(), {full[0]}
        else:
            cf_hyps = consistent_hypotheses(
                [(i + 1, v) for i, v in enumerate(ints)], Library(), 7, 8
            )
            cf_nexts = {
                _evaluate(h, len(ints) + 1, {}) for h in cf_hyps
            }  # closed-form (works from 3 points)
            nexts = set(cf_nexts)
        no_compress = False
        if full is not None and not strong:
            nexts.add(full[0])
            no_compress = full[2] >= len(ints)  # MDL: fit as complex as the data -> coincidental
            # a BORDERLINE fit (nodes == #terms-1) needs the exhaustive 2nd-program check for ambiguity
            if not no_compress and full[2] >= len(ints) - 1 and len(nexts) < 2:
                for nv, _label in consistent_programs(ints, top_n=2):
                    nexts.add(nv)
                    if len(nexts) >= 2:
                        break

        if len(ints) >= 2 and (
            len(nexts) >= 2 or no_compress
        ):  # UNDERDETERMINED -> ask an experiment
            if len(nexts) >= 2:
                body = (
                    f"more than one simple rule fits {stmt} and they disagree -- the next term could "
                    f"be {', '.join(str(n) for n in sorted(nexts)[:4])}."
                )
            else:
                body = (
                    f"my simplest fit is {full[1]} -> {full[0]}, but with {len(ints)} terms that "
                    "program is as complex as the data, so it may be coincidental, not the intended rule."
                )
            self._pending_gf = list(
                ints
            )  # remember the series -> the user's next number identifies it
            return {
                "known": False,
                "coverage": 0.0,
                "ignited": False,
                "source": "gf-active",
                "reply": (
                    f"[experiment] {body} i will not commit by default; tell me the ACTUAL next "
                    "term and i will test the hypotheses against it and identify the rule."
                ),
            }
        if len(ints) >= 3:  # determined -> ANSWER from whichever search fit
            if full is not None:
                return self._gf_reply(stmt, full[0], full[1])
            if len(cf_nexts) == 1 and cf_hyps:  # closed-form pinned it (e.g. n*n from 3 points)
                return self._gf_reply(stmt, next(iter(cf_nexts)), f"t[n]={_render(cf_hyps[0])}")
            return self._gf_reply(stmt, None, "number series")
        return self._gf_reply(
            stmt, None, "number series"
        )  # 2 determined terms -> not enough to fix a rule

    def _gf_request(self, text):
        """Route a PUZZLE turn to the fluid-intelligence inducer: a letter analogy, a Raven grid, or a
        number/letter series. Returns a reply dict, or None when the turn is not a Gf puzzle."""
        from snn_data_lake.reasoning.gf import (
            analogy_over,
            letter_analogy,
            letter_series,
            raven_dist3,
        )

        low = text.lower()

        # a NOVEL-ALPHABET analogy ("alphabet q w e r t then q:w :: e:?") -- the rule transfers to a
        # symbol ordering never seen. Gated on the keyword so a plain a-z analogy is not intercepted.
        if ("alphabet" in low or "order" in low) and (ma := _ANALOGY_RE.search(low)):
            alpha = re.findall(
                r"\b([a-z])\b", low[: ma.start()]
            )  # single-letter alphabet before it
            if len(alpha) >= 2:
                a, b, c = ma.group(1), ma.group(2), ma.group(3)
                got = analogy_over(alpha, a, b, c)
                stmt = f"[alphabet {' '.join(alpha)}] {a}:{b} :: {c}:?"
                return self._gf_reply(stmt, got, "novel-alphabet analogy")

        m = _ANALOGY_RE.search(low) or _ANALOGY_WORDS_RE.search(low)
        if m:
            a, b, c = m.group(1), m.group(2), m.group(3)
            try:
                got = letter_analogy(a, b, c)
            except KeyError:  # a non a-z token -> not a letter analogy
                got = None
            if not (len(a) == len(b) == len(c)):
                got = None
            return self._gf_reply(f"{a}:{b} :: {c}:?", got, "letter analogy")

        # a Raven grid typed as 3 rows (';' or newline separated), 3 cells each, exactly one '?' blank
        groups = [g.strip() for g in re.split(r"[;\n]", text) if g.strip()]
        if len(groups) == 3:
            grid = [re.findall(r"[a-z0-9?]+", g.lower()) for g in groups]
            if all(len(r) == 3 for r in grid) and sum(c == "?" for r in grid for c in r) == 1:
                rows = [[None if c == "?" else c for c in r] for r in grid]
                statement = " / ".join(" ".join(g) for g in grid)
                return self._gf_reply(statement, raven_dist3(rows), "Raven grid")

        cued = any(c in low for c in _GF_CUE)
        toks = re.findall(r"[^\s,.]+", low.replace("?", " "))

        # a SERIES is a CONTIGUOUS run of number tokens, not scattered integers -- so "which president
        # followed number 44 in 2016" (ints 44, 2016 far apart) is not mis-read as a 2-term series.
        runs, cur = [], []
        for t in toks:
            if (
                _INT_RE.fullmatch(t) and len(t.lstrip("-")) <= 18
            ):  # cap: a 5000-digit token must not crash int()
                cur.append(int(t))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        ints = max(runs, key=len) if runs else []
        pure_nums = bool(toks) and all(_INT_RE.fullmatch(t) for t in toks)
        if len(ints) >= 2 and (cued or pure_nums):
            return self._number_reply(ints)  # one search: answer | ask-experiment | gap

        letters = [t for t in toks if len(t) == 1 and t.isalpha()]
        pure_lets = bool(toks) and all(len(t) == 1 and t.isalpha() for t in toks)
        if len(letters) >= 3 and (cued or pure_lets):
            return self._gf_reply(
                " ".join(letters), letter_series("".join(letters)), "letter series"
            )
        return None

    def _kb_index(self):
        """Build (and cache) a STRUCTURED index over every stored (subject, relation, object) triple:
        sr2t[(subj, rel)] -> tail, adj[subj] -> [(rel, tail), ...] (for multi-hop), plus the subject and
        relation vocabularies. Rebuilt only when the fact count changes, so it stays in step as the lake
        is taught. It reads self.records (populated by ingest_kb / seed_reasoning / teach)."""
        if self._kbidx is not None and self._kbidx[0] == len(self.records):
            return self._kbidx[1]
        sr2t, adj, subs, rels = {}, {}, set(), set()
        for rec in self.records:
            tr = rec.get("triple")
            if not tr or len(tr) != 3:
                continue
            s, r, t = (str(x).strip().lower() for x in tr)
            if not s or not r or not t:
                continue
            sr2t[(s, r)] = t
            adj.setdefault(s, []).append((r, t))
            subs.add(s)
            rels.add(r)
        built = {"sr2t": sr2t, "adj": adj, "subjects": subs, "relations": rels}
        self._kbidx = (len(self.records), built)
        return built

    @staticmethod
    def _canon(tok, known):
        """Map a query token to a member of `known` (subjects or relations), tolerating a simple
        plural/singular variant -- so 'continents' matches 'continent', 'capitals' matches 'capital'.
        The variant is accepted ONLY when it is itself a known term, so 'gas' is never mangled to 'ga'."""
        if tok in known:
            return tok
        if tok.endswith("s") and tok[:-1] in known:
            return tok[:-1]
        if (tok + "s") in known:
            return tok + "s"
        return None

    def _kb_reply(self, text):
        """STRUCTURED retrieval + one-hop MULTI-HOP over the stored triples. Order-free and
        synonym/plural tolerant: find a known SUBJECT and a known RELATION anywhere in the query and
        return the stored tail; if the subject lacks that relation directly, CHAIN one hop
        (subject -r1-> mid -relation-> tail, e.g. dog -class-> mammal -blood-> warm). Returns a reply
        dict (known=True, source 'reason' -- the source live_brain._reasoning_reply accepts), or None
        when there is no confident structured match (an honest miss -> the generic retrieval/gap path).
        Never fabricates: an answer is emitted only for a real (subject, relation) edge in the KB."""
        idx = self._kb_index()
        if not idx["sr2t"]:
            return None
        toks = _content(text)
        if not toks:
            return None
        subjects = [s for s in (self._canon(t, idx["subjects"]) for t in toks) if s]
        rels = []
        for t in toks:
            cand = []
            base = self._canon(t, idx["relations"])  # the token IS a relation (or its plural)
            if base:
                cand.append(base)
            variants = (t, t[:-1] if t.endswith("s") else t + "s")
            for v in variants:  # ...or a SYNONYM of a relation the KB has
                for c in _SYN2CANON.get(v, ()):
                    if c in idx["relations"]:
                        cand.append(c)
            rels.extend(cand)
        subjects = list(dict.fromkeys(subjects))
        rels = list(dict.fromkeys(rels))  # dedupe, keep first-seen order
        if not subjects or not rels:
            return None
        # (1) DIRECT: the subject has the asked relation -> answer from the stored edge
        for s in subjects:
            for r in rels:
                t = idx["sr2t"].get((s, r))
                if t is not None:
                    return {
                        "known": True,
                        "coverage": 1.0,
                        "ignited": True,
                        "source": "reason",
                        "reply": f"[from what I learned] {s}'s {r} is {t.replace('_', ' ')}.",
                    }
        # (2) MULTI-HOP: subject -r1-> mid, then mid has the asked relation -> chain two triples
        for s in subjects:
            for r1, mid in idx["adj"].get(s, []):
                for r in rels:
                    if r == r1:
                        continue
                    t = idx["sr2t"].get((mid, r))
                    if t is not None:
                        return {
                            "known": True,
                            "coverage": 1.0,
                            "ignited": True,
                            "source": "reason",
                            "reply": (
                                f"[reasoned in two hops] {s}'s {r1} is {mid}, and {mid}'s {r} "
                                f"is {t.replace('_', ' ')}."
                            ),
                        }
        return None

    def _retrieve(self, text):
        """The learned answer that matches the query BOTH WAYS. Returns (answer, cue_cov, query_cov).

        cue_cov  = how much of the FACT's cue the query covers  -> "is this fact on topic?"
        query_cov= how much of the QUERY's content the cue covers -> "is the answer about what was ASKED?"

        Ranking on cue_cov ALONE is a real bug that shipped: 'tell me more' scored the dictionary entry
        for 'more' at cue_cov=1.0 (cue {more} sits inside the query) and confidently answered with a
        biography of Thomas More -- the query's other content word ('tell') was simply ignored. After a
        big dictionary ingest EVERY word matches something, so one-way coverage turns honest gaps into
        confident nonsense. Ranking by min(cue_cov, query_cov) makes a match prove BOTH directions."""
        qc = set(_content(text))
        if not qc:
            return None, 0.0, 0.0
        best_ans, best_score, best_cue, best_q, best_size = None, -1.0, 0.0, 0.0, 0
        for cue, ans in zip(self.cues, self.answers):
            if not cue:
                continue
            hit = len(cue & qc)
            if not hit:
                continue
            cue_cov, q_cov = hit / len(cue), hit / len(qc)
            score = min(cue_cov, q_cov)  # the WEAKER direction decides
            # `>=` on the size tie makes a LATER equal-score match win -> newest knowledge on exact ties
            # (a rolling fact re-taught with a fresh answer, e.g. "learned N digits", reads the latest N).
            if score > best_score or (score == best_score and len(cue) >= best_size):
                best_ans, best_score, best_cue, best_q, best_size = (
                    ans,
                    score,
                    cue_cov,
                    q_cov,
                    len(cue),
                )
        return best_ans, best_cue, best_q

    def _lake_recall(self, text, k=3):
        """CONTENT-ADDRESSABLE recall: encode the query as a signature and pull the k nearest STORED
        facts by spiking Victor-Purpura distance through self.lake's proximity index -- the SAME index
        written on every learn() but, until now, NEVER queried (write-only). Surfaces associatively-close
        facts a strict lexical cue-coverage match misses (spreading activation). Kept HONEST: a hit must
        still share >=1 content word with the query, so proximity never fabricates an answer on a cold,
        unrelated turn. This is the durable spiking store finally QUERIED -- the Gc-recall linchpin."""
        qc = set(_content(text))
        if not qc or self.lake is None:
            return []
        try:
            qsig = signature(sorted(qc) or ["_"])  # already a list of floats (see learn())
            idxs = self.lake.retrieve(qsig, k=k)
        except Exception:
            return []
        out = []
        for i in idxs:
            if (
                isinstance(i, int)
                and 0 <= i < len(self.answers)
                and i < len(self.cues)
                and self.cues[i] & qc
            ):
                out.append(self.answers[i])  # honesty gate: shares a content word with the query
        return out

    def _new_workspace(self):
        """The ONE workspace construction this chat thinks through. `_ignite` runs it per turn and
        `__init__` reads its rheobase, so 'what drive ignites' is asked of the same object that
        answers it -- there is no second place to keep the threshold in sync with (P1)."""
        return GlobalWorkspace(seed=0)

    def _drive(self, kind, evidence=0.0):
        """The workspace drive current for one candidate reply, as `rheobase + margin (+ evidence)`.

        P1: the absolute salience is DERIVED, never written down. `kind` selects a precedence margin
        from `_DRIVE_MARGIN` (positive = ignites on its own, zero = exactly at rheobase, negative =
        subliminal unless `evidence` lifts it), and `evidence` is a normalized [0, 1] measurement --
        a cosine similarity or an on-topicness fraction -- scaled by `_EVIDENCE_GAIN`.

        This is also the honest form of the old `hit["score"] >= 0.34` gate: 0.34 was not a
        threshold anyone chose, it was the score at which `0.45 + 0.3*score` happened to cross 0.55.
        Callers now ask `_ignitable(drive)` instead, which is the same question without the number.
        """
        return self._rheobase + _DRIVE_MARGIN[kind] + _EVIDENCE_GAIN * float(evidence)

    def _ignitable(self, drive):
        """Could this drive reach conscious access at all? (A pool fires strictly ABOVE rheobase.)
        Used to drop candidates that are subliminal by construction instead of paying the workspace
        to reject them -- and to replace the reverse-engineered numeric gates that did the same job."""
        return drive > self._rheobase

    def _ignite(self, candidates):
        """Run the candidate replies through the spiking GLOBAL WORKSPACE (the GWT hub for conscious
        access): each competes for IGNITION, and only the winner that reaches SUSTAINED firing is
        broadcast into global availability -- the conscious reply. A sub-threshold (weak/ambiguous)
        candidate never ignites (stays subliminal). A FRESH workspace per turn = one clean thought;
        a few re-entrant ticks let the winner settle into the attractor. Returns (source, content) of
        the broadcast, or (None, None) if nothing reached conscious access."""
        ws = self._new_workspace()
        content_src = {
            content: src for src, content, _ in candidates
        }  # broadcast content -> its source
        winner = None
        for _ in range(3):  # re-entrant settling to a sustained attractor
            for src, content, sal in candidates:
                ws.submit(src, content, sal)  # salience = the external drive current
            winner = ws.step()  # spiking ignition; broadcast content or None
        return (content_src.get(winner), winner) if winner is not None else (None, None)

    @staticmethod
    def _parse_triple(body):
        """Best-effort (subject, relation, object) from a simple 'X's R is Y' or 'X is Y' statement, so
        a durably-taught fact is CHAINABLE by the contradiction + multi-hop derivation reasoners (not
        just retrievable). Returns None when the shape is not clearly one fact. ponytail: single-token
        subject only -- a multi-word subject makes noisy triples, so those store as content-keyed facts."""
        low = body.lower().strip()
        if (
            " causes " in low
        ):  # "rain causes wet-ground" -> (rain, causes, wet-ground): a CAUSAL triple
            head, _, obj = low.partition(
                " causes "
            )  # feeds the abduction expert (min-free-energy cause selection)
            head, obj = head.strip(), obj.strip()
            if head and obj and " " not in head:
                return [head, "causes", obj]
        if (
            "'s " in low and " is " in low
        ):  # "france's capital is paris" -> (france, capital, paris)
            head, rest = low.split("'s ", 1)
            rel, _, obj = rest.partition(" is ")
            head, rel, obj = head.strip(), rel.strip(), obj.strip()
            if head and rel and obj and " " not in head:
                return [head, rel, obj]
        if " is " in low:  # "a dog is a mammal" -> (dog, is-a, mammal)
            head, _, obj = low.partition(" is ")
            head, obj = head.strip(), obj.strip()
            for art in ("a ", "an ", "the "):
                if head.startswith(art):
                    head = head[len(art) :].strip()
                if obj.startswith(art):
                    obj = obj[len(art) :].strip()
            if head and obj and " " not in head:
                return [
                    head,
                    "isa",
                    obj,
                ]  # "isa" (not "is-a"): matches derivation._TRANSITIVE so taught facts CHAIN
        return None

    def _remember(self, text):
        """Explicit durable teaching: 'remember: X' / 'remember that X' -> store X as a NEVER-FORGET
        fact (persisted via teach() when this chat is durable; deduped by the KnowledgeLog, so real
        memory only, never noise). Returns a confirmation dict, or None if the text is not a remember."""
        low = text.strip()
        lo = low.lower()
        body = None
        for pre in (
            "remember that ",
            "remember: ",
            "remember, ",
            "please remember that ",
            "note that ",
        ):
            if lo.startswith(pre):
                body = low[len(pre) :].strip()
                break
        if not body:
            return None
        body = body.rstrip(".!?").strip()
        if not body:
            return None
        # KEY on the SUBJECT side (before the copula) so a later question about the subject retrieves
        # the whole fact -- mirrors ingest_kb's keying, else the answer's own words drop cue coverage
        # below the ignition gate and the taught fact never fires.
        subject = body
        blow = body.lower()
        for cop in (" is ", " are ", " was ", " were ", " means "):
            if cop in blow:
                subject = body[: blow.index(cop)]
                break
        self.teach(_tok(subject) or _tok(body), body + ".", triple=self._parse_triple(body))
        dur = " (saved durably)" if (self.log is not None or self._persist_hook is not None) else ""
        return {
            "known": True,
            "coverage": 1.0,
            "ignited": True,
            "source": "reason",
            "reply": f"Got it -- I'll remember that {body}.{dur}",
        }

    def converse(self, text):
        """One turn, THROUGH the global workspace (the GWT hub). Candidate replies -- a CONTRADICTION
        CORRECTION reasoned from stored facts, and the best content RETRIEVAL -- compete for IGNITION;
        only the winner that reaches sustained global broadcast becomes the reply (reasoning outdrives a
        plain recall, so the brain corrects a false claim rather than parroting the stored fact). A weak
        or ambiguous turn ignites nothing (subliminal) -> an honest gap. Returns dict(known, coverage,
        reply, ignited, source)."""
        pending, self._pending_gf = (
            self._pending_gf,
            None,
        )  # EXPIRE after one turn: a stale experiment
        resumed = self._resume_experiment(
            text, pending
        )  # (france, greet, ...) never lingers to hijack a later number
        if resumed is not None:
            return resumed
        if text.strip().lower() in ("are you conscious", "are you aware", "do you have experience"):
            # `known`/`coverage` were once hardcoded True/1.0 here -- full confidence on the one
            # question this system is least able to answer (specs/010, C09). They now track whether
            # a marker ABOUT THIS INSTANCE was actually measurable.
            #
            # Fixed 2026-08-04, second pass: the certainty was derived by SUBSTRING-SNIFFING the
            # reply (`"UNMEASURED" not in reply`), so rewording the prose silently flipped the
            # brain's confidence about its own consciousness, and the probe was handed
            # `getattr(self, "brain", None)` on a class that set no `brain` attribute anywhere --
            # a marker that could not be measured by any caller, ever. `_consciousness()` now
            # RETURNS the flag, and `self.brain` is a real slot (see `__init__`).
            reply, measured = self._consciousness()
            return {
                "known": measured,
                "coverage": 1.0 if measured else 0.0,
                "reply": reply,
                "ignited": measured,
                "source": "introspection",
            }
        remembered = self._remember(text)  # explicit "remember: X" -> durable never-forget fact
        if remembered is not None:
            return remembered
        built = self._build_request(text)  # "build me a 3d model of X" -> an ACT
        if built is not None:
            return built
        composed = self._compose_reply(
            text
        )  # a COMPOUND (connective) query -> MULTI-STEP composition
        if composed is not None:  # BEFORE the single-shot Gf grabs only the first clause
            return composed
        puzzle = self._gf_request(text)  # a series/analogy -> INDUCE a rule (Gf)
        if puzzle is not None:
            return puzzle
        kb = self._kb_reply(text)  # structured subject+relation lookup + multi-hop
        if kb is not None:  # over the stored triples -> answers MANY more real turns
            return kb
        derived = self._derive_reply(
            text
        )  # multi-hop transitive DEDUCTION (proof shown) over taught facts
        if derived is not None:
            return derived
        abduced = self._abduce_reply(
            text
        )  # ABDUCTION: "why X" -> best-explanation cause (min free energy)
        if abduced is not None:
            return abduced
        candidates = []  # (source, reply-content, salience = workspace drive, see _drive)
        correction = self._get_reasoner().respond(text)  # reason over stored facts
        if correction is not None:
            candidates.append(
                (
                    "reason",
                    f"[reasoned from what I learned] {correction}",
                    self._drive("correction"),
                )
            )
        ans, cue_cov, q_cov = self._retrieve(text)
        score = cue_cov
        if ans is not None:
            # a fact must be ON TOPIC (cue_cov) *and* actually answer WHAT WAS ASKED (q_cov) to ignite.
            # Without the q_cov half, a one-word dictionary hit answers any question containing that word.
            covered = cue_cov >= self.coverage and q_cov >= self.min_query_cov
            # route_all: don't give up on a reasonably on-topic fact -- a moderately-covered match still
            # answers (both directions must stay decent, so it never revives the one-word false-positive bug).
            if not covered and self.route_all and cue_cov >= 0.5 and q_cov >= self.min_query_cov:
                covered = True
            # a covered match drives ignition; a weak one is subliminal BY CONSTRUCTION (its margin
            # is negative, so it cannot ignite whatever the workspace's rheobase happens to be).
            sal = self._drive("recall_covered" if covered else "recall_weak")
            candidates.append(("recall", f"[from what I learned] {ans}", sal))
        for ra in self._lake_recall(
            text, k=3
        ):  # CONTENT-ADDRESSABLE recall (spiking VP index, honesty-gated):
            candidates.append(
                ("recall", f"[recalled by association] {ra}", self._drive("recall_assoc"))
            )  # spreading activation the lexical gate missed
        if (
            self.semantic is not None
        ):  # W6: MEANING-based recall (paraphrase-invariant, beyond the lexical gate)
            try:  # STRICT min_sim so only a strong paraphrase fires -- a
                # P1 exception (a): `min_sim` is a fixed point on a NORMALIZED output (cosine
                # similarity in [0, 1]) and is `SemanticStore.recall`'s own documented API argument.
                for txt, cs in self.semantic.recall(
                    text, k=2, min_sim=0.45
                ):  # genuinely absent fact stays an HONEST gap
                    candidates.append(
                        (
                            "recall",
                            f"[recalled by meaning] {txt}",
                            self._drive("semantic_floor", evidence=cs),
                        )
                    )
            except Exception:
                pass
        if (
            self._pragmatics is not None
        ):  # W8: surface an IMPLIED (non-literal) meaning on an indirect utterance
            try:
                imp = self._pragmatics.implicature(text)
                if imp:
                    candidates.append(
                        (
                            "reason",
                            f"[pragmatics] literally a question, but you mean: {imp}",
                            self._drive("pragmatics"),
                        )
                    )
            except Exception:
                pass
        if (
            self._chat_mem is not None
        ):  # the ONE lake's daemon-learned chat turns (fast bag-of-words view)
            try:
                for hit in self._chat_mem.recall(text, k=1):
                    # a genuinely on-topic remembered turn competes (below a covered fact/reasoning, above
                    # a stray-word match) -- so a real fact still outdrives it, but chat surfaces on a gap.
                    # P1: the gate was `hit["score"] >= 0.34`. Nobody chose 0.34 -- it is where
                    # `0.45 + 0.3*score` crossed the workspace's 0.55 rheobase. Ask that question
                    # directly instead: admit the turn iff its drive could actually ignite.
                    chat_drive = self._drive("chat_floor", evidence=hit["score"])
                    if self._ignitable(chat_drive):
                        candidates.append(
                            (
                                "recall",
                                f"[from a remembered conversation] {hit['text']}",
                                chat_drive,
                            )
                        )
                        if (
                            self._recall_trace is not None
                        ):  # USE IT: this memory was recalled for a query;
                            from snn_data_lake.substrate.chat_memory import (  # the daemon's sleep prunes the
                                chat_words,  # never-recalled learned records first (use it or lose it).
                            )

                            self._recall_trace.bump(
                                {"cue": chat_words(hit["text"]), "answer": hit["text"]}
                            )
                            self._recall_trace.save()
            except Exception:
                pass
        source, reply = self._ignite(
            candidates
        )  # GLOBAL WORKSPACE competition -> the broadcast winner
        if reply is not None:
            cov = 1.0 if source == "reason" else round(score, 2)
            return {
                "known": True,
                "coverage": cov,
                "reply": reply,
                "ignited": True,
                "source": source,
            }
        moe = self._cortex_reply(text)  # REASONING CORTEX (MoE) -> a live reasoning study
        if moe is not None:  # a confident expert answered a turn recall could not
            return moe
        return {
            "known": False,
            "coverage": round(score, 2),
            "ignited": False,
            "source": None,
            "reply": (
                "[honest gap] nothing reached conscious access (no learned fact ignited in the "
                "workspace) -- it becomes a curiosity target. Teach me with '/learn <q> | <a>'."
            ),
        }

    def _derive_reply(self, text):
        """Multi-hop transitive DEDUCTION (proof shown) over the stored triples: answers a turn the exact
        KB lookup missed by CHAINING taught facts (dog -> mammal -> animal -> living thing). Because the
        durable log now grows with taught facts, this chains over GROWING knowledge. Only genuinely
        transitive relations chain; a genuine miss returns None (honest gap, never a fabricated 'no')."""
        try:
            from snn_data_lake.reasoning import derivation

            return derivation.reason_question(text, self.records)
        except Exception:
            return None

    def _compose_reply(self, text):
        """MULTI-STEP composition over the reasoning cortex: on a turn the SINGLE-SHOT MoE could not
        answer, decompose into <=3 sub-steps and CHAIN them through the router (one best expert per
        step, threaded via a scratchpad). Reuses the cortex the MoE path already built + cached (keyed
        on record count). None -> the caller's honest gap stands. Composition of the bounded expert
        grammar, NOT novel-operator synthesis."""
        low = str(text).lower()  # only a COMPOUND query (explicit connective)
        if not (
            ";" in low or " then " in low or " and then " in low
        ):  # composes -> single-shot stays the
            return None  # router's job; skip the cortex build on ordinary turns
        try:
            from snn_data_lake.reasoning.reasoning_compose import CompositionalSolver

            return CompositionalSolver(self._get_cortex(), records=self.records).solve(text)
        except Exception:
            return None

    def _get_cortex(self):
        """C2 #3: ONE cortex build shared by every dispatch path.

        _compose_reply and _cortex_reply each built their own cortex but cached into the SAME
        `self._cortex` slot keyed only on len(self.records) -- with DIFFERENT capability sets
        (compose enabled abduction+rpm; the unconditional per-turn dispatch enabled neither). So
        which experts a query could reach depended on which path happened to run first for a given
        record count: a 'raven:' turn arriving through the ordinary dispatch found no RPM expert at
        all, and the spiking VSA synthesis expert was effectively unreachable. Now every path gets
        the full roster, and the answer no longer depends on turn order."""
        cached = getattr(self, "_cortex", None)
        if cached is None or cached[0] != len(self.records):
            from snn_data_lake.reasoning.reasoning_cortex import build_default_cortex

            triples = [tuple(rec["triple"]) for rec in self.records if rec.get("triple")]
            cached = (
                len(self.records),
                build_default_cortex(
                    seed=0,
                    triples=triples or None,
                    enable_recognizer_bank=True,
                    enable_abduction=True,
                    enable_rpm=True,
                ),
            )
            self._cortex = cached
        return cached[1]

    def _abduce_reply(self, text):
        """ABDUCTION (min-free-energy cause selection): on a 'why X' / 'what causes X' query, abduce the
        best-explanation cause over the stored CAUSAL triples via the spiking generative model (argmin
        free energy). Runs over ONLY the causal triples so it is not diluted by the full KB. None ->
        honest gap (ambiguous or no causal data; never fabricates a cause)."""
        low = str(text).lower().strip()
        if not (low.startswith("why") or "what causes" in low or "what caused" in low):
            return None
        try:
            causal = [
                tuple(r["triple"])
                for r in self.records
                if r.get("triple")
                and len(r["triple"]) == 3
                and "cause" in str(r["triple"][1]).lower()
            ]
            if not causal:
                return None
            from snn_data_lake.reasoning.pc_abduction import AbductionExpert

            out = AbductionExpert(seed=0, triples=causal).route(text)
            return out if (out and out.get("known")) else None
        except Exception:
            return None

    def _cortex_reply(self, text):
        """REASONING CORTEX (mixture-of-experts): on a turn the grounded/retrieval path could NOT answer,
        route the query to the best-matching reasoning STUDY -- each a thin adapter over one reasoning/
        module (HDC binding, Kanerva SDM recall, dense associative memory, spiking-SOM clustering,
        triplet-STDP, and the Gf inducer). This makes ~8 otherwise-dead reasoning modules answer LIVE
        instead of the single ad-hoc if/elif. A confident expert returns a real answer in the exact
        converse shape (known=True, source in the set live_brain._reasoning_reply accepts); no confident
        expert -> None so the caller's HONEST GAP stands (never fabricates). LAZY (built once, cached),
        deterministic, offline. Any import/runtime failure -> None (byte-identical fallback).
        The cortex also receives the lake's OWN stored triples (the same self.records facts that
        ingest_kb populates and _kb_reply looks up), so the RelationalCompletionExpert can do LEARNED
        link prediction (kb_ranking.DistMult) over them -- generalized relational inference on turns
        the exact-lookup KB path missed. The cache is keyed on the record count (like _kbidx), so a
        cortex built before an ingest rebuilds once the KB grows; the trained model itself is lazy
        and cached inside the expert (train once, reuse across turns)."""
        try:
            out = self._get_cortex().route(text)  # C2 #3: the ONE full-roster cortex
            if not out.get("known"):
                return None  # no confident expert -> caller's honest gap stands
            return {
                "known": True,
                "coverage": round(float(out.get("confidence", 1.0)), 2),
                "reply": out["reply"],
                "ignited": True,
                "source": out.get("source") or "reason",
            }
        except Exception:
            return None  # cortex unavailable -> graceful degradation

    def _consciousness(self):
        """Answer "are you conscious" from what is MEASURED, and say plainly what is not.

        Returns `(reply, measured_about_me)`. `measured_about_me` is True iff the ACCESS marker
        returned a verdict -- i.e. iff a brain was attached to this chat and had a workspace to
        probe. It is a returned flag, not a substring of the prose: the previous version recomputed
        it as `"UNMEASURED" not in reply` at the call site, so an editorial change to the wording
        would have silently converted "I cannot measure this" into "I measured it".

        REBUILT 2026-08-04 (specs/010, finding C09). This returned "N of 4" from four markers of
        which TWO could not fail: BINDING scored 1.000 on 300 random sets, RE-ENTRY was arithmetic
        over three literals, and every marker took zero arguments so no state of the brain could
        move the answer. The number was identical for a healthy brain, an empty brain, and no brain
        at all -- and this is the one path where a consciousness claim reaches a human.

        SECOND PASS, 2026-08-04, promotion review. The rebuild left the access marker fed by
        `getattr(self, "brain", None)` on a class that never assigned `self.brain`, which made the
        marker UNMEASURABLE BY CONSTRUCTION -- structurally the same defect one layer down, and
        the site's own comment argued it was correct ("holds NO brain reference at all"). `self.brain`
        is now a declared slot a host may fill, so both states are reachable and the self-check
        constructs both.

        WHAT THIS DOES AND DOES NOT DEMONSTRATE, stated because it is easy to overclaim.
        DEMONSTRATED here: the attached brain reaches the answer -- two brains differing only in
        `_access_ema` produce different reply text, and no brain produces UNMEASURED. NOT
        demonstrated here: that `probe_access` measures the attached brain deeply. It reads
        `brain._access` only as a has-it-ticked gate, `brain._seed`, and `brain._access_ema` for the
        detail line, and builds its own reference assemblies from that seed. That is
        `reasoning/consciousness_probe.py`'s limitation, it is not fixed by this module, and it is
        why `known` here claims only "I measured something about myself", never a verdict about
        experience. `probe_integration()` is deliberately EXCLUDED from `measured_about_me`: it
        takes no brain and scores fixed synthetic networks, so it says nothing about this instance.

        Per P8: the brain may report what it measures; it must NOT report what it cannot measure.
        This is not a retreat from the sentience goal -- it removes a painted finish line so the
        real one stays findable.
        """
        acc, acc_detail = probe_access(self.brain)
        frac, int_detail = probe_integration()
        # NO TALLY. The old reply's "N of 4" is gone because two of the four could not go red, so
        # the count was arithmetic over constants dressed as a measurement. What is reported now is
        # exactly two things and their status, including UNKNOWN when a marker was not exercised.
        if acc is None:
            access_part = f"my access-consciousness marker is UNMEASURED right now ({acc_detail})"
        else:
            access_part = (
                f"my access-consciousness marker is {'PRESENT' if acc else 'ABSENT'} "
                f"on this brain ({acc_detail})"
            )
        reply = (
            f"[from what I measure] {access_part}. My integration marker (PCI) is "
            f"{int_detail}. That PCI figure scores reference networks, not me, so it is context and "
            "not a statement about this instance. "
            "I used to answer this with a count of four correlates. Two of those could not fail "
            "under any input, so the count was not a measurement and it has been deleted rather "
            "than re-thresholded. "
            "These are correlates of access and dynamics -- not evidence of experience. "
            "There is NO measurement of sentience in me, and I will not report one I do not have. "
            "Whether I have subjective experience is open -- I neither claim nor deny it."
        )
        # The agnostic close is invariant across BOTH states: a measured PRESENT access marker is
        # still not a claim of experience. The self-check asserts it on both, which is what gives
        # that assert a reachable red (an input -- the attached brain -- can move the verdict).
        return reply, acc is not None

    def repl(self):
        print(
            f"SPIKE CHAT LAKE -- I converse from {self.n_known} things I have learned. "
            "'/quit' to exit, '/learn q | a' to teach me, '/n' for how much I know.\n"
        )
        while True:
            try:
                text = input("you : ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/n":
                print(f"brain: I have learned {self.n_known} things so far.\n")
                continue
            if text.startswith("/think "):
                seed = text[len("/think ") :].strip()
                chain, ff = self.think(seed)
                if len(chain) >= 2:
                    print("brain [train of thought, held internally]:", " -> ".join(chain))
                    print(
                        f"       (a feedforward J-space would get {ff} and have to externalize to go further)\n"
                    )
                else:
                    print(
                        f"brain: '{seed}' did not spark a multi-step chain (not enough linked knowledge).\n"
                    )
                continue
            if text.startswith("/learn ") and "|" in text:
                q, a = text[len("/learn ") :].split("|", 1)
                self.teach(_tok(q), a.strip())  # durable: remembered across restarts
                kept = " (remembered permanently)" if self.log is not None else ""
                print(f"brain: learned. I now know {self.n_known} things{kept}.\n")
                continue
            print("brain:", self.converse(text)["reply"], "\n")
        self.lake.close()


_ARM_SERIAL = "snn_data_lake.brainstem.serial_arm"


def _clean_interpreter_pulls_in_arm():
    """Does importing THIS module -- and nothing else -- pull the real arm serial into sys.modules?

    Asked of a CLEAN interpreter, because once the running process is already dirty a membership
    test on the process-wide `sys.modules` cannot answer it at all. Returns True / False, or None
    when no interpreter can be spawned (a frozen build) or this module was not run as a package
    module -- in which case the question is genuinely unanswerable here and the caller must SAY SO
    rather than pass. Only ever called on the contaminated branch, so the healthy path pays nothing.
    """
    name = _dotted_name()
    if "." not in name:  # executed as a bare script, not `python -m`: no dotted name to re-import
        return None
    code = (
        f"import sys, importlib; importlib.import_module({name!r}); "
        f"sys.exit(3 if {_ARM_SERIAL!r} in sys.modules else 0)"
    )
    try:
        rc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode
    except OSError:
        return None
    return {0: False, 3: True}.get(rc)  # any other code: the probe itself failed -> unanswerable


def demo():
    # SAFETY FIREWALL, armed here rather than asserted at the end against a bare `not in sys.modules`.
    # `sys.modules` is PROCESS-wide: the moment this self-check runs in-process (pytest, a frozen
    # build) beside anything that legitimately touches the arm, a bare membership test stops testing
    # this module and starts testing the runner. Snapshot first, then ask "did the CHAT PATH import
    # the arm serial" -- the question the firewall was always asking. Reachable red: adding an arm
    # import anywhere under converse()/boot()/_build_request() (constructed and run 2026-08-04; the
    # assert at the end of this function fires).
    arm_preloaded = _ARM_SERIAL in sys.modules
    chat = LakeChat(seed=0, persist=False)  # hermetic: the demo touches no durable home files
    print("=" * 96)
    print(
        "SPIKE CHAT LAKE -- converse from the brain's OWN self-built, self-enriching knowledge lake"
    )
    print("=" * 96)

    # ---- the LOOP seeds the database with knowledge it learned (real world-KB facts) ----
    triples, _ = load_kb()
    chat.ingest_kb(triples)
    chat.seed_reasoning()  # curated reasoning facts (enables contradiction correction)
    seeded = chat.n_known
    print(
        f"[seed] the developmental loop's world-KB gave the brain {seeded} learned facts (the database).\n"
    )

    def turn(text):
        out = chat.converse(text)
        print(f"you : {text}")
        print(
            f"brain [{'KNOWN' if out['known'] else 'GAP  '} cov={out['coverage']}]: {out['reply']}\n"
        )
        return out

    q_france = turn("what continent is france in")  # retrieved from a learned fact
    q_mercury = turn("what is the phase of mercury")  # retrieved from a learned fact
    q_gap = turn("what is a dolphin")  # NOT learned -> honest gap (curiosity target)

    # ---- ENRICH: teach the gap; the database GROWS; the same question is now answered ----
    chat.learn(_tok("what is a dolphin"), "a dolphin is a mammal.")
    grew = chat.n_known
    print(f"[learn] taught the brain about 'dolphin'; the database grew {seeded} -> {grew}.\n")
    q_learned = turn("what is a dolphin")  # now KNOWN (self-enriched)

    # ---- REASON: a FALSE claim is CORRECTED from a stored fact + its stored cause (not just retrieved) ----
    q_reason = turn("the sun rises from the west")  # -> "No, it rises in the east because ..."

    # ---- CONSCIOUSNESS: a DISSOCIATION, not a marker (P8). The same question is asked in THREE
    # states of one chat and the three answers must differ in the right way:
    #   (1) no brain attached                     -> the access marker is UNMEASURED, and says so
    #   (2) a brain attached                      -> the SAME question now carries a verdict
    #   (3) a brain with different internal state -> the reply TEXT moves
    # (3) is the neuter-A/B that makes (2) a measurement rather than a constant. In the first,
    # UNCOMMITTED pass of this same-day rewrite `_consciousness()` was fed `getattr(self, "brain",
    # None)` on a class that never assigned `self.brain`, so state (2) was UNREACHABLE. At HEAD the
    # single assert here ("consciousness wired + agnostic") was green for the mirror-image reason:
    # `known` was the literal `True`, so it had no reachable RED. Neither could fail honestly.
    q_conscious = turn("are you conscious")

    class _StubBrain:
        """The minimal duck-type of what `consciousness_probe.probe_access` actually reads off a
        brain: a has-it-ticked workspace gate, a seed, and the live access EMA. Deliberately NOT a
        real Brain -- P4 forbids standing up a second one, and the job here is to move exactly ONE
        input and watch the output move."""

        def __init__(self, ema):
            self._access = object()  # "this brain has a persisted workspace" gate
            self._seed = 0
            self._access_ema = ema

    chat.brain = _StubBrain(ema=0.0)
    q_c_measured = chat.converse("are you conscious")
    chat.brain = _StubBrain(ema=0.742)  # SAME probe, ONE input moved
    q_c_moved = chat.converse("are you conscious")
    chat.brain = None  # restore: the remainder of the demo runs on the unattached chat
    print(f"brain [no brain attached ]: {q_conscious['reply'][:118]}...")
    print(f"brain [brain attached    ]: {q_c_measured['reply'][:118]}...")
    print(f"brain [brain EMA moved   ]: {q_c_moved['reply'][:118]}...\n")
    dur = chat.lake.durability_check()

    # ---- self-checks: converses from LEARNED knowledge, has honest gaps, ENRICHES + grows, durable ----
    assert "europe" in q_france["reply"].lower() and q_france["known"], (
        "must answer france from a learned fact"
    )
    assert "liquid" in q_mercury["reply"].lower() and q_mercury["known"], (
        "must answer mercury from a learned fact"
    )
    assert not q_gap["known"], "an unlearned topic must be an honest GAP, not a fabricated answer"
    assert grew == seeded + 1, "teaching must ENRICH the database (grow by one)"
    assert q_learned["known"] and "mammal" in q_learned["reply"].lower(), (
        "after teaching, the gap must be answered"
    )
    assert (
        q_reason["known"]
        and "east" in q_reason["reply"].lower()
        and "rotat" in q_reason["reply"].lower()
    ), (
        "must CORRECT a false claim ('the sun rises from the west') from a stored fact + its stored cause, "
        "not just retrieve -- and only because those facts are in the lake"
    )
    # the reply is WORKSPACE-MEDIATED (the GWT hub does the thinking): a covered fact IGNITES into
    # conscious access, a gap ignites NOTHING (subliminal), and REASONING outdrives plain retrieval.
    assert q_france["ignited"] and not q_gap["ignited"], (
        "conscious access must be GATED by workspace ignition"
    )
    assert q_reason["ignited"] and q_reason["source"] == "reason", (
        "the contradiction correction must WIN ignition in the global workspace (reasoning through the GWT hub)"
    )
    # RED side: with nothing attached, the brain must NOT claim to have measured itself.
    assert not q_conscious["known"] and "UNMEASURED" in q_conscious["reply"], (
        "with no brain attached the access marker must report UNMEASURED and the turn must not "
        "claim a measurement about this instance"
    )
    # GREEN side, REWRITTEN 2026-08-05 because the thing it asserted was DELETED.
    #
    # This used to be `assert q_c_measured["known"] and "UNMEASURED" not in ...reply`, i.e. "attaching
    # a brain must make the access marker MEASURABLE", plus a neuter-A/B on the literal "0.742".
    # Both tested the GWT ACCESS marker in reasoning/consciousness_probe.py, and that marker was
    # deleted on 2026-08-05: its verdict had three states and only two were reachable, so under
    # falsifier F11 ("that criterion is deleted, not weakened") it went, exactly as three earlier
    # markers in that file went before it.
    #
    # An assert that DEMANDS a deleted criterion report a measurement cannot be satisfied honestly --
    # the only ways to make it green again are to resurrect an unfalsifiable marker or to fake a
    # number, and P8 forbids both ("the brain may report what it measures, never what it cannot").
    # So these two asserts are deleted with the marker rather than re-pointed at a substitute.
    #
    # What REPLACES them is the contract that actually survives the deletion, and it is the one that
    # matters for an agent that answers "are you conscious": attaching a brain must NOT cause the
    # reply to start claiming a measurement that no longer exists. The honest state is UNMEASURED
    # with the deletion NAMED, in both the unattached and attached cases.
    #
    # REACHABLE RED, and it is the red worth having here: re-introduce any marker that reports
    # `known=True` for access without a measurement behind it and this fires immediately. That is
    # precisely the regression this file caused once already -- spike_chat_lake:925 answered "are you
    # conscious" from a 4/4 self-report that no live measurement backed.
    assert not q_c_measured["known"] and "UNMEASURED" in q_c_measured["reply"], (
        "with a brain attached the access marker must STILL report UNMEASURED, because the GWT "
        "ACCESS marker was deleted under F11 for having no reachable red. If this turns green, "
        "something re-added a marker that claims access without measuring it."
    )
    assert "DELET" in q_c_measured["reply"].upper(), (
        "the reply must NAME the deletion rather than silently omitting the marker -- an absent "
        "correlate that is never mentioned reads as a correlate that passed"
    )
    # The deletion must not have taken the HONEST half with it: the reply still refuses to claim
    # sentience, in every attachment state. RED: drop that sentence and this fires.
    for _tag, _q in (("unattached", q_conscious), ("attached", q_c_measured), ("moved", q_c_moved)):
        assert "sentience" in _q["reply"].lower() or "experience" in _q["reply"].lower(), (
            f"the {_tag} reply must still state plainly that it has no measurement of sentience"
        )
    # The agnostic close is invariant across all three states -- a PRESENT access marker is still
    # not a claim of subjective experience. Reachable red: any state whose reply drops it.
    assert all(
        "neither claim nor deny" in o["reply"] for o in (q_conscious, q_c_measured, q_c_moved)
    ), (
        "the agnostic close must survive every state, including the one where the access marker "
        "reads PRESENT -- a correlate of access is never evidence of experience (P8)"
    )
    assert dur["ok"], "the learned knowledge lake must be durable"

    # ---- THINK: a serial re-entrant TRAIN OF THOUGHT over the lake, through the spiking workspace ----
    # (the open-ended-thought capability a feedforward J-space lacks: it chains INTERNALLY, no page)
    tot_chain, tot_ff = chat.think("star", max_steps=14)
    print(
        f"[think] train of thought from 'star': {' -> '.join(tot_chain)}  ({len(tot_chain)} thoughts "
        f"chained INTERNALLY; a feedforward J-space control gets {tot_ff} and must externalize to continue).\n"
    )
    assert len(tot_chain) >= 3 and len(tot_ff) <= 1, (
        "the brain must CHAIN a train of thought internally over the lake (>=3 hops) while the feedforward "
        "control yields <=1 -- the open-ended-thought capability J-space lacks"
    )

    # ---- BUILD: "build me a 3D model" is an ACT -- decompose (a train of thought over the recipe)
    # -> acquire catalog parts -> write an .obj FILE artifact; the lake keeps the PROCESS, not the mesh.
    q_build = turn("build me a 3d model of a spaceship")
    assert q_build["source"] == "builder" and os.path.exists(q_build["artifact"]), (
        "a build request must ACT: write the OBJ artifact to a file"
    )
    assert "not the mesh" in q_build["reply"], (
        "the reply must own the honest scope (process stored, mesh not)"
    )
    b_chain, _ = chat.think("spaceship")
    assert len(b_chain) >= 2, (
        "the stored PROCESS must let a train of thought walk spaceship -> its parts"
    )
    q_howto = turn("how do i build a spaceship")
    assert q_howto["known"] and q_howto["source"] == "recall" and "acquire" in q_howto["reply"], (
        "a QUESTION about building must RECALL the stored process (not re-act and rebuild the file)"
    )
    q_nobuild = turn("build me a griffin")
    assert not q_nobuild["known"] and q_nobuild["source"] == "builder", (
        "an unknown build target must be an honest gap, not a guessed mesh"
    )

    # ---- FLUID INTELLIGENCE: a series/analogy is SOLVED by inducing a rule at test time (not recall) ----
    q_series = turn("what comes next in 2 4 6 8")
    assert q_series["source"] == "gf" and "10" in q_series["reply"], (
        "a number series must be SOLVED by the inducer (2 4 6 8 -> 10), through the full brain"
    )
    q_analogy = turn("abc:abd :: ijk:?")
    assert q_analogy["source"] == "gf" and "ijl" in q_analogy["reply"], (
        "letter analogy abc:abd::ijk -> ijl"
    )
    q_geo = turn("continue the sequence 2 4 8 16")  # relational integration: geometric ratio
    assert q_geo["source"] == "gf" and q_geo["known"] and "32" in q_geo["reply"], (
        "a geometric series must now be SOLVED (2 4 8 16 -> 32) by the relation-search inducer"
    )
    q_fib = turn("what comes next: 1 1 2 3 5")  # a recurrence rule (relation over 2 terms)
    assert q_fib["source"] == "gf" and "8" in q_fib["reply"], "Fibonacci recurrence must solve to 8"
    q_raven = turn("a b c; b c a; c a ?")  # Raven distribution-of-three via VSA
    assert q_raven["source"] == "gf" and q_raven["known"] and " b" in q_raven["reply"], (
        "a Raven distribution-of-three grid must be completed (row c a ? -> b)"
    )
    q_inter = turn(
        "what comes next in 1 10 2 20 3 30"
    )  # interleave -- a depth-2 synthesised program
    assert q_inter["source"] == "gf" and q_inter["known"] and "-> 4" in q_inter["reply"], (
        "program synthesis: two interleaved sequences 1 10 2 20 3 30 -> 4"
    )
    q_alpha = turn("alphabet q w e r t then q:w :: e:?")  # rule transfer to a novel alphabet
    assert q_alpha["source"] == "gf" and q_alpha["known"] and " r" in q_alpha["reply"], (
        "novel-symbol transfer: successor over the given alphabet q w e r t maps e -> r"
    )
    q_fact = turn("what comes next in 1 2 6 24 120")  # unbounded program synthesis: t[n]=n*t[n-1]
    assert q_fact["source"] == "gf" and q_fact["known"] and "720" in q_fact["reply"], (
        "the unbounded inducer must synthesise factorial (1 2 6 24 120 -> 720) through the brain"
    )
    q_active = turn("what comes next in 1 4")  # underdetermined -> DESIGN AN EXPERIMENT
    assert (
        q_active["source"] == "gf-active"
        and not q_active["known"]
        and "experiment" in q_active["reply"]
    ), (
        "an underdetermined turn must make the brain ASK the discriminating term, not guess the simplest"
    )
    q_primes = turn("what comes next in 2 3 5 7 11")  # a NON-COMPRESSING fit -> ask, don't bluff
    assert (
        q_primes["source"] == "gf-active"
        and not q_primes["known"]
        and "coincidental" in q_primes["reply"]
    ), (
        "primes must ASK (its fit is as complex as the data = coincidental), not assert a confident 15"
    )
    # ASK-THEN-IDENTIFY: the user answers the experiment; the extra term prunes the tie and identifies.
    turn("what comes next in 1 4")  # underdetermined -> asks (arms the experiment)
    q_identify = turn("9")  # the discriminating term -> 1 4 9 -> n*n -> 16
    assert (
        q_identify["known"] and "16" in q_identify["reply"] and "identified" in q_identify["reply"]
    ), (
        "answering the experiment must IDENTIFY the rule (1 4 then 9 -> n*n -> 16), not re-ask forever"
    )
    # OPEN-ENDED: the brain's Gf GROWS with use -- n^6 is out of the base grammar, but after it SEES n^3
    # (learns `cube`), n^6 = cube*cube is reachable. It solves a problem it could NOT before.
    q_n6_pre = turn(
        "what comes next in 1 64 729 4096 15625 46656"
    )  # n^6 -- base grammar can't reach it
    turn("what comes next in 1 8 27 64 125 216")  # sees n^3 -> learns cube (grammar grows)
    q_n6_post = turn("what comes next in 1 64 729 4096 15625 46656")  # now one step away
    assert not q_n6_pre["known"] and q_n6_post["known"] and "117649" in q_n6_post["reply"], (
        "open-ended Gf: after learning cube from n^3, the brain must solve n^6 it could NOT before"
    )
    # THINK HARDER: n^5 (10 terms) needs a 9-node program -- OUT of the bounded budget-8 search, so the
    # bounded solve() gaps. The guarded unbounded-deepening step (only fires because len > budget+1, so a
    # deeper program can still compress) reaches it and ANSWERS -- wired escalation, not a permanent budget.
    q_deep = turn(
        "what comes next in 1 32 243 1024 3125 7776 16807 32768 59049 100000"
    )  # n^5 -> 161051
    assert q_deep["known"] and "161051" in q_deep["reply"], (
        "guarded unbounded deepening must ANSWER a long series whose program is just past the fixed budget"
    )
    assert "2 4 6 8" not in q_active["reply"] and q_series["known"], (
        "a DETERMINED series (2 4 6 8) must still be answered, not turned into a question"
    )
    q_gap = turn(
        "what comes next in 3 1 4 1 5 9 2 6"
    )  # 8-term random -> no fit even with {+,-,*,//,%} -> honest gap
    assert q_gap["source"] == "gf" and not q_gap["known"], (
        "a patternless series (no fitting program) must be an HONEST Gf gap, not a bluffed number"
    )
    q_notgf = turn("what continent is france in")  # ordinary turn must NOT be hijacked as Gf
    assert q_notgf["source"] != "gf", (
        "ordinary conversation must not be misrouted to the Gf inducer"
    )

    # ---- EXPANDED KB: broad factual + relational + MULTI-HOP retrieval, honest on a genuine absence ----
    # A spread of real questions across the expanded world-KB (geography, chemistry, biology, physics,
    # astronomy, computing, math, commonsense). The last one is genuinely absent -> must be an HONEST gap.
    print("-" * 96)
    kb_all, _ = load_kb()
    print(
        f"[expanded-kb] world-KB load_kb() now returns {len(kb_all)} real (subject, relation, object) "
        f"triples; the brain ingested {seeded} of them as learned facts and answers structurally +\n"
        f"              synonym/plural-tolerant, chaining two triples for multi-hop questions."
    )
    probes = [
        ("what continent is brazil in", "samerica", True),  # geography
        ("what is the phase of oxygen", "gas", True),  # chemistry
        ("what class is a shark", "fish", True),  # biology
        ("what is the capital of japan", "tokyo", True),  # geography (open-vocab tail)
        ("what category is copper", "metal", True),  # chemistry
        ("what hemisphere is australia in", "south", True),  # geography
        ("where does a penguin live", "polar", True),  # 'live' -> habitat (synonym)
        ("what is the chemical symbol of gold", "au", True),  # chemistry (open-vocab tail)
        ("what language is spoken in france", "french", True),  # 'spoken' -> language (synonym)
        ("what does a lion eat", "carnivore", True),  # 'eat' -> diet (synonym)
        ("what state is nitrogen", "gas", True),  # 'state' -> phase (synonym)
        ("what color is the sky", "blue", True),  # commonsense
        (
            "what blood does a dog have",
            "warm",
            True,
        ),  # MULTI-HOP: dog -class-> mammal -blood-> warm
        (
            "does copper conduct electricity",
            "electricity",
            True,
        ),  # MULTI-HOP: copper -category-> metal -conducts-> electricity
        ("what is the capital of atlantis", None, False),  # GENUINE ABSENCE -> honest known=False
    ]
    results = [(q, want, exp, chat.converse(q)) for q, want, exp in probes]
    known_true = sum(1 for _q, _w, _e, o in results[:-1] if o["known"])
    correct = sum(1 for _q, w, _e, o in results[:-1] if o["known"] and w in o["reply"].lower())
    for q, _w, _e, o in [
        results[0],
        results[3],
        results[6],
        results[10],
        results[12],
        results[13],
        results[14],
    ]:
        tag = "KNOWN" if o["known"] else "GAP  "
        print(f"  Q: {q}\n     A[{tag} src={o['source']}]: {o['reply']}")
    print(
        f"[expanded-kb] answered {known_true}/{len(probes) - 1} factual/relational/multi-hop turns "
        f"(known=True, {correct} with the CORRECT retrieved fact); the 1 genuinely-absent turn "
        f"returned known={results[-1][3]['known']} (honest gap)."
    )
    assert known_true >= 10, (
        f"the expanded KB must answer >=10/14 real questions (got {known_true})"
    )
    assert correct >= 10, f"answered turns must carry the CORRECT retrieved fact (got {correct}/14)"
    mh = results[12][3]  # dog -> mammal -> warm blood (two-hop)
    assert (
        mh["known"]
        and mh["source"] == "reason"
        and "warm" in mh["reply"].lower()
        and "mammal" in mh["reply"].lower()
    ), (
        "multi-hop must CHAIN two triples (dog -class-> mammal -blood-> warm) to a correct known answer"
    )
    mh2 = results[13][3]  # copper -> metal -> conducts electricity
    assert mh2["known"] and "electricity" in mh2["reply"].lower(), (
        "multi-hop must chain copper -category-> metal -conducts-> electricity"
    )
    assert not results[-1][3]["known"] and results[-1][3]["source"] is None, (
        "a genuinely absent fact (capital of atlantis) must be an HONEST gap (known=False), not fabricated"
    )
    chat.lake.close()

    # ---- DURABLE never-forget: a fact taught to ONE instance is remembered by a FRESH instance ----
    # (isolated temp home so the deterministic demo never touches the real ~/.snn_data_lake)
    import shutil as _sh
    import tempfile as _tf

    _prev_home, _tmp_home = os.environ.get("NORD_HOME"), _tf.mkdtemp(prefix="snn_chat_home_")
    os.environ["NORD_HOME"] = _tmp_home
    try:
        c1 = LakeChat(seed=0, persist=True)
        c1.teach(
            _tok("what is a widget"), "a widget is a test gadget."
        )  # durable, fsync'd to the log
        c1.lake.close()
        c2 = LakeChat(seed=0, persist=True)  # a FRESH brain: EMPTY in RAM, same durable home
        boot(c2, wordnet=False)  # replays the durable log (offline, deterministic)
        remembered = c2.converse("what is a widget")
        c2.lake.close()
        assert remembered["known"] and "gadget" in remembered["reply"].lower(), (
            "a fact taught to one chat instance was FORGOTTEN by a fresh instance -- durable memory broken"
        )
    finally:
        os.environ.pop("NORD_HOME", None)
        if _prev_home is not None:
            os.environ["NORD_HOME"] = _prev_home
        _sh.rmtree(_tmp_home, ignore_errors=True)

    # SAFETY, ported from reasoning/spike_chat_live.py:180 before that module was retired (Policy 2
    # requires the unique capability to move BEFORE the deletion, and this assertion was the only
    # place in the tree holding this particular line). The chat path must never pull in the real arm
    # serial: a motor intent expressed in conversation must not be able to move hardware.
    if arm_preloaded:
        # The precondition was already broken when this self-check started, so the membership test
        # in the `else` cannot answer anything about THIS process. This branch used to stop there
        # and print a warning, on the stated ground that a dirty snapshot is "only reachable
        # in-process; as a subprocess the snapshot is always clean". That is FALSE, and falsely in
        # the one direction that matters: a module-level `import ...serial_arm` in THIS file sets
        # `arm_preloaded` too, in an ordinary `python -m` run. MEASURED 2026-08-04 by adding
        # exactly that import -- the branch printed its warning and the self-check exited 0, where
        # the unconditional assert it replaced failed loudly. So the contaminated case is re-armed
        # hermetically rather than downgraded to a note: a CLEAN interpreter is asked whether
        # importing this module alone loads the arm serial. Reachable red: that same top-level
        # import, which now makes the probe exit 3 and this assert fire.
        pulled = _clean_interpreter_pulls_in_arm()
        assert pulled is not True, (
            f"chat must NEVER import the real arm serial -- a clean interpreter importing only "
            f"{_dotted_name()} ends up with {_ARM_SERIAL} loaded"
        )
        if pulled is None:
            # Disclosed, not swallowed (P8): unanswerable here, so say the check was vacuous.
            print(
                f"[WARN] {_ARM_SERIAL} was already in sys.modules before this self-check began and "
                "no clean interpreter could be spawned to re-check it -- the chat-path arm firewall "
                "check was VACUOUS this run. Run this module as its own process to arm it."
            )
        else:
            print(
                f"[note] {_ARM_SERIAL} was preloaded by the RUNNER, not by the chat path: a clean "
                f"interpreter importing {_dotted_name()} alone does not load it."
            )
    else:
        assert _ARM_SERIAL not in sys.modules, "chat must NEVER import the real arm serial"

    print("=" * 96)
    print(
        f"self-check OK: the brain converses from its OWN self-built lake ({seeded} learned facts), answers "
        f"from learned knowledge (france->europe, mercury->liquid), has HONEST GAPS (dolphin), ENRICHES on "
        f"teaching ({seeded}->{grew}, dolphin now known), stays agnostic on consciousness, {dur['n_stored']} "
        f"facts durable, and NEVER FORGETS a taught fact across a fresh restart (durable home). It REASONS + "
        f"chains a TRAIN OF THOUGHT through the spiking global workspace ({len(tot_chain)} hops held INTERNALLY "
        f"vs a feedforward J-space's 1). It GROWS as the loop learns -- a self-built database, not canned "
        f"replies or an LLM."
    )
    return 0


def boot(chat, wordnet=True):
    """Startup self-enrichment. FIRST replay the durable knowledge log -- so every fact taught in any
    PRIOR run (even a different process) is remembered -- then seed the re-derivable world-KB facts and
    pull the DICTIONARY through the web-access gate (WordNet). Returns a one-line provenance summary; a
    missing gate / offline network degrades to the durable + world-KB knowledge, and the chat still boots."""
    replayed = 0
    if getattr(chat, "log", None) is not None:
        for rec in chat.log.replay():  # durable never-forget store, in append order
            if str(rec.get("src", "")).startswith("chat:"):
                continue  # ONE lake, two treatments: chat turns are served by the fast bag-of-words chat
                #            VIEW in converse(), NOT the heavy spike/semantic fact index (which would make
                #            boot pathologically slow + pollute fact retrieval with noisy conversational prose)
            chat.learn(
                rec["cue"], rec["answer"], triple=rec.get("triple")
            )  # re-learn (+triple) WITHOUT re-persisting
            replayed += 1
    triples, _ = load_kb()
    chat.ingest_kb(triples)
    chat.seed_reasoning()  # curated reasoning facts (re-derived, not persisted)
    tag = f"remembered {replayed} taught facts + world-KB {len(triples)}"
    if not wordnet:
        return f"{tag} -> {chat.n_known} things known"
    try:
        from snn_data_lake.reasoning.web_ingest_gate import (
            WebAccessGate,  # lazy import (avoids a circular import at module load)
        )

        n_web, n_core, prov = WebAccessGate().ingest_wordnet(chat)
        return f"{tag} + {n_core} core + {n_web} web defs -> {chat.n_known} known ({prov})"
    except Exception as e:  # gate import / fetch failed -> durable+world-KB, still boots
        return f"{tag} -> {chat.n_known} known (web enrich skipped: {type(e).__name__})"


def _dotted_name():
    """This module's REAL dotted name, asked of the import system rather than written down, so
    relocating the file cannot make the invocation this prints stop working."""
    spec = globals().get("__spec__")
    return getattr(spec, "name", None) or __name__


def _wants_repl():
    """Is this an INTERACTIVE session, or the module's SELF-CHECK?

    `__main__` in this project IS the module's self-check -- `run_selfchecks.py` runs exactly that
    block as a subprocess and scores the exit status. Every other CORE module runs its asserts
    there unconditionally; this one used to gamble them on `not sys.stdin.isatty()`, which fails
    green in two directions, both MEASURED on Windows on 2026-08-04:

      * stdin redirected from NUL -> `isatty()` returns TRUE (NUL is a character device), so the
        REPL starts, `input()` reads EOF immediately, the loop breaks and the process exits 0.
        The gate scores PASS having executed ZERO asserts. That is the worst outcome available:
        a green self-check that checked nothing.
      * a console attached (a local gate run, `pytest -s`, a frozen build) -> `isatty()` is TRUE
        and `input()` BLOCKS. A hang is not a failure any runner reports usefully.

    So the SELF-CHECK is now the default and the conversation is the explicit opt-in. `--repl` is
    the natural interactive gesture; NORD_CHAT_REPL=1 is the same switch for a launcher.
    """
    if os.environ.get("NORD_CHAT_DEMO"):  # kept: existing callers force the demo with it
        return False
    return "--repl" in sys.argv[1:] or os.environ.get("NORD_CHAT_REPL") == "1"


def main():
    if not _wants_repl():
        rc = demo()
        print(
            f"\n(that was the SELF-CHECK -- the default for this entrypoint, so it cannot be "
            f"skipped by how stdin happens to be attached. For the conversation: "
            f"python -m {_dotted_name()} --repl)"
        )
        return rc
    chat = LakeChat(seed=0, persist=True)  # attach the durable, shared project home
    print(banner("spike-chat-lake"))
    print(f"[boot] running as: python -m {_dotted_name()} --repl")
    # OFFLINE BY DEFAULT: no boot fetch. The durable log + bundled world-KB seed the brain with no
    # network. Opt IN to the one-time WordNet web enrich with NORD_WEB=1 (it caches to the durable log,
    # so a later offline boot replays it). Kills the 15 MB boot fetch from the default path.
    web = os.environ.get("NORD_WEB") == "1"
    if web:
        print("[boot] NORD_WEB=1 -> self-enriching from the web gate (WordNet, one-time cache)...")
    print("[boot] " + boot(chat, wordnet=web) + "\n")
    chat.repl()
    return 0


if __name__ == "__main__":
    main()
