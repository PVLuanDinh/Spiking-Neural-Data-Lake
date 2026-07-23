r"""
BRAIN LOOP -- the UNIFIED, ALWAYS-ON Brain: Spike Chat and the developmental loop merged into ONE
program over ONE durable brain on the physical SSD, with every learned thing written as RAW BINARY.

Before this, the chat and the learning loops were separate programs: the chat conversed but never ran
the loop, and the loops learned into TEMP .spkl files that were WIPED on startup -- nothing reached the
SSD, and the chat could not converse from what the loops learned. This merges them:

  ALWAYS-ON: a background daemon thread LEARNS continuously (knowledge -> skills -> reflection, cycled)
             while the foreground REPL CONVERSES instantly. Both share one lock + one SSD brain.
  ONE BRAIN ON THE SSD (under substrate.home.home(), i.e. ~/.snn_data_lake):
      knowledge.bin    RAW BINARY (binlog SNDLBIN1: [u32 len][packed fact]) -- the authoritative
                       never-forget store. Replayed into the chat's index at boot. Deduped.
      prototypes.bin   learned prototypes/weights as FLOAT16 raw binary (half the float32 bytes).
      lake.spkl        learned exemplars as 4 B packed AER spike binary (LakeMemory/StreamingHub).
      develop_state.json  volatile bookkeeping only (rff width, last accuracies) -- never conversable.
  NEVER FORGETS: boot replays knowledge.bin -> the brain wakes knowing everything any prior run (or any
             other process, or the GPU venv) ever learned. Proven cross-process in the self-check.

FACULTIES (composed, not reimplemented): LakeChat (retrieval + the spiking GLOBAL WORKSPACE ignition +
the contradiction Reasoner + the re-entrant TrainOfThought + the consciousness probes), ConceptReflector
(reflect on each new concept, wire what it thought together), LakeMemory (durable AER exemplars),
WebAccessGate (scholarly Wikipedia through the SSRF-firewalled gate), OpenWorldController (discover new
skills from unlabeled data), and -- on GPU -- developmental_core.develop_cycles (the full conv-STDP digit
curriculum: perceive -> hold-in-WM -> learn -> grow neurons -> consolidate -> reflect -> sleep-refit).

TWO VENVS, ONE BRAIN: NOTHING here imports torch at module level -- the perceptual source imports
developmental_core LAZILY. So this file boots, converses, learns knowledge/skills/reflection and
self-checks under the torch-free .venv-lake, while `--develop` under .venv-cu128 runs the GPU
curriculum into the SAME knowledge.bin. A fresh .venv-lake chat then converses the GPU-learned concepts.

HONEST SCOPE: (1) the always-on background sources are the numpy ones (knowledge/skills/reflection);
PERCEPTION is torch/CUDA and runs on `/develop` or `--develop`, not on every background tick -- CUDA in a
daemon thread is fragile, so this is stated rather than risked. (2) The chat answers from NARRATED facts
about what the loop learned ("digit 3 resembles digit 5"), not from raw pixels. (3) float16 prototypes
are lossy (fine for stored weights); the precision-sensitive SLDA covariance is never float16-persisted --
it re-accumulates from the exemplars. (4) Nothing here is claimed to be conscious; the workspace/
consciousness framing stays agnostic.

  PYTHONPATH=src .venv-lake/Scripts/python research/brain_loop.py            # boot + always-on + REPL
  NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/brain_loop.py --develop   # + GPU perception
  (piped / NORD_BRAIN_SELFCHECK=1 -> the deterministic offline self-check)
Deps: numpy + stdlib (torch ONLY for the opt-in perceptual develop).
"""
import os
import random
import re
import sys
import threading
import time
from collections import deque

from snn_data_lake.substrate.home import KnowledgeLog, SpikeLake, banner, home, lake_path, save_state
from snn_data_lake.substrate.binlog import BinLog, pack_array, pack_fact, unpack_array, unpack_fact
from snn_data_lake.plasticity.competitive_learning import CompetitiveLearner
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace
from snn_data_lake.reasoning.spike_chat_lake import LakeChat, _content, _tok
from snn_data_lake.reasoning.world_kb_stream import load_kb
from snn_data_lake.reasoning.concept_reflection import ConceptReflector
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory

# scholarly frontier for the always-on knowledge source (real encyclopedic topics)
_TOPICS = ["Calculus", "Cell (biology)", "Photosynthesis", "DNA", "Mitochondrion", "Derivative",
           "Integral", "Evolution", "Entropy", "Gravity", "Neuron", "Molecule", "Atom",
           "Thermodynamics", "Prime number", "Protein", "Enzyme", "Galaxy"]


def _concept_facts(concept, chain, n_learned, kind="digit", src="perception:mnist"):
    """NARRATE a learned concept into SYMBOLIC facts the chat can converse from: what it is, what the
    reflection train-of-thought decided it RESEMBLES (as triples the reasoner + think() can walk), and a
    rolling count. Phrasing is STABLE (no volatile accuracy in the answer text) so re-developing does not
    bloat the deduped binary log. HONEST: the chat answers from these narrations, not from raw pixels."""
    c = str(concept)
    facts = [{"cue": [kind, c], "answer": f"{kind} {c} is a handwritten numeral i learned to recognize.",
              "src": src}]
    for e in chain[1:]:                                   # the reflection chain: c -> e -> ...
        facts.append({"cue": [c, str(e)], "answer": f"{kind} {c} resembles {kind} {e}.",
                      "triple": [c, "resembles", str(e)], "src": "reflection"})
    facts.append({"cue": ["how", "many", kind + "s"],
                  "answer": f"i have learned {n_learned} {kind}s so far.", "src": src})
    return facts


def _sym_proto(c, dim=16, per_cluster=2):
    """A synthetic CLUSTERED prototype in [0,1] (concepts sharing a band resemble each other). Used by
    the torch-free symbolic develop stand-in; the real prototypes come from conv-STDP perception."""
    v = [0.0] * dim
    base = ((c // per_cluster) * 4) % dim
    for j in range(4):
        v[(base + j) % dim] = 0.9
    v[(base + (c % per_cluster)) % dim] = 1.0             # within-cluster variation; stays <= 1.0
    return v


class Brain:
    """ONE always-on brain: it LEARNS in a background thread and CONVERSES in the foreground, over one
    durable raw-binary brain on the SSD. Composes the chat faculties + the developmental faculties."""

    def __init__(self, seed=0, home_dir=None):
        if home_dir:
            os.environ["NORD_HOME"] = home_dir            # isolate a test/self-check brain
        self.seed = seed
        self._rng = random.Random(seed)
        self.kpath = os.path.join(home(), "knowledge.bin")
        self.ppath = os.path.join(home(), "prototypes.bin")
        self._lock = threading.RLock()                    # guards the RAM index + the binary appends
        self._running = False
        self._thread = None
        self.n_learned = 0
        self.learned_log = []                             # what the background learner has learned
        self._skill_ctrl = None
        self._skill_stream = None
        self._skill_pos = 0
        self._skills_seen = set()
        self._topics_done = set()

    # ---- boot: wake up knowing everything ever learned (raw binary -> RAM) ------------------------
    def boot(self):
        self.klog = BinLog(self.kpath)                    # authoritative durable knowledge (raw binary)
        self.protolog = BinLog(self.ppath)                # learned prototypes as float16 raw binary
        self.chat = LakeChat(seed=self.seed, persist=False)   # persist=False: the Brain owns the binlog
        # durability for anything taught THROUGH the chat (e.g. a build's stored PROCESS via
        # _build_request -> chat.teach): persist the record to the Brain's binlog. Disk-only -- teach()
        # already learned it into the RAM index, so this must NOT re-learn (that would double the cue).
        self.chat._persist_hook = lambda rec: self.klog.append(pack_fact(rec))
        replayed = 0
        for payload in self.klog.replay():                # NEVER-FORGET: the SSD binary rehydrates the mind
            rec = unpack_fact(payload)
            self.chat.learn(rec["cue"], rec["answer"], triple=rec.get("triple"))
            replayed += 1
        # also replay the shared JSONL KnowledgeLog: the always-on autonomous enricher (a separate
        # process / scheduled task) writes what it learns THERE, so without this those facts would be
        # written-but-never-conversable. Boot knows everything ANY prior run or process learned.
        for rec in KnowledgeLog().replay():
            self.chat.learn(rec["cue"], rec["answer"], triple=rec.get("triple"))
            replayed += 1
        triples, _ = load_kb()                            # re-derivable seeds (not persisted -- stays lean)
        self.chat.ingest_kb(triples)
        self.chat.seed_reasoning()
        self.protos = [unpack_array(p) for p in self.protolog.replay()]
        # ONE durable lake: AER exemplars now persist (vector + payload) to the shared brain_lake.bin via
        # SpikeLake. The .spkl at lake_path() is a per-boot SCRATCH index (deleted fresh so its stale spikes
        # never DOUBLE the rehydrate), and rebuild_from_durable() re-populates the RAM proximity index from
        # the ONE durable lake at boot. Fixes the AER durability gap: pre-fold the payload map lived only in
        # RAM and reset on restart (durable bytes on the .spkl, but not durable MEMORY).
        _scratch = lake_path()
        try:
            os.remove(_scratch)  # stale scratch from a prior boot; the durable store is now brain_lake.bin
        except FileNotFoundError:
            pass
        self.lake = LakeMemory(path=_scratch, seed=self.seed, packed=True, durable=SpikeLake())
        self.lake.rebuild_from_durable()  # rehydrate the proximity index from the ONE durable lake
        self.reflector = ConceptReflector(seed=self.seed)
        # ---- the CONVERSATION (what a stateless lookup lacks) ----
        # ONE workspace for the whole conversation: its cross-tick attractor + hysteresis is the thought
        # HELD IN MIND between turns. (A fresh workspace per turn -- the first version -- discards it.)
        self.ws = GlobalWorkspace(seed=self.seed)
        self.topic = None                                 # what it is currently holding in mind
        self.history = deque(maxlen=24)                   # (you, brain) turns -- so it can recall the conversation
        self.thoughts = deque(maxlen=12)                  # (chain, why) -- trains of thought it actually had
        self.questions = deque(maxlen=40)                 # what it WANTS to ask (found by testing itself)
        self.selftests = [0, 0]                           # [posed, failed] -- its own record of testing itself
        for i, p in enumerate(self.protos):               # rehydrate the reflector from the float16 protos
            self.reflector.register(str(i), p)
        self.n_learned = len(self.protos)
        kb = os.path.getsize(self.kpath) if os.path.exists(self.kpath) else 0
        return (f"{banner('brain', knowledge=False)} | knowledge.bin={replayed} facts ({kb:,}B raw binary) "
                f"| prototypes={len(self.protos)} (float16) | knows {self.chat.n_known} things")

    # ---- durable remember: one fact -> SSD raw binary + the RAM index ------------------------------
    def _remember(self, rec):
        with self._lock:
            wrote = self.klog.append(pack_fact(rec))      # raw binary, fsync'd, deduped
            self.chat.learn(rec["cue"], rec["answer"], triple=rec.get("triple"))
            return wrote

    def teach(self, cue_tokens, answer, triple=None):
        """Teach the brain a fact -- durable on the SSD immediately, remembered forever."""
        rec = {"cue": list(cue_tokens), "answer": answer, "src": "taught"}
        if triple:
            rec["triple"] = list(triple)
        self._remember(rec)
        return self

    def teach_many(self, facts):
        """BULK teach: many facts under ONE lock acquisition with ONE fsync -- the mass-ingest path (a
        dictionary, a corpus). Much faster per fact than teach() in a loop: the per-record fsync
        dominates (binlog's self-check measures ~32x on the write alone), and the single lock also stops
        it contending with the background learner on every fact -- which profiling showed was the real
        cost of teaching over HTTP (~97ms/fact under contention vs ~1.5ms of actual work).
        `facts` = [{cue:[...], answer:str, triple?:[...], src?:str}, ...].
        Returns the number actually written (duplicates are skipped)."""
        recs = []
        for f in facts:
            rec = {"cue": [str(c) for c in f["cue"]], "answer": str(f["answer"]),
                   "src": str(f.get("src") or "taught")}
            if f.get("triple"):
                rec["triple"] = [str(t) for t in f["triple"]]
            recs.append(rec)
        with self._lock:
            wrote = self.klog.extend([pack_fact(r) for r in recs])       # ONE fsync for the batch
            for r in recs:                                               # index them in RAM
                self.chat.learn(r["cue"], r["answer"], triple=r.get("triple"))
        return wrote

    def emit_concept(self, concept, prototype, chain, kind="digit", src="perception:mnist"):
        """A newly-learned concept from the developmental loop: narrate it into facts (durable binary),
        and store its learned prototype as float16 raw binary."""
        with self._lock:
            self.n_learned += 1
            for rec in _concept_facts(concept, chain, self.n_learned, kind=kind, src=src):
                self._remember(rec)
            self.protolog.append(pack_array(prototype))   # float16 weight on the SSD
            self.protos.append(list(prototype))

    # ---- converse: an actual CONVERSATION, not stateless lookup ------------------------------------
    def converse(self, text):
        """One conversational turn. Unlike a bare lookup this HOLDS the conversation: the topic persists
        in the workspace attractor across turns, referring expressions ('it', 'tell me more', 'what about
        X') resolve against it, and questions ABOUT the conversation or about its own thinking are
        answered from conversational state -- never by looking the word up in the dictionary."""
        with self._lock:
            t = (text or "").strip()
            if not t:
                return {"known": False, "coverage": 0.0, "reply": "", "ignited": False, "source": None}
            meta = self._meta(t)                              # conversational/introspective turns
            if meta is not None:
                self.history.append((t, meta["reply"]))
                return meta
            resolved, note = self._resolve(t)                 # 'it'/'more'/'what about X' -> the held topic
            # THINK, in the three forms -- each says which kind of inference it used, and how sure it is
            for reason in (self._reason_question,             # DEDUCTIVE: guaranteed, with proof
                           self._reason_induce,               # INDUCTIVE: probable, with its evidence
                           self._reason_abduce):              # ABDUCTIVE: best explanation, a guess
                got = reason(resolved)
                if got is not None:
                    self.history.append((t, got["reply"]))
                    return got
            out = self.chat.converse(resolved)
            if not out.get("known"):                          # a gap is something to be CURIOUS about
                topic = self._topic_of(resolved)
                if topic and len(topic) > 2:
                    q = "what is %s? you asked me and i did not know." % topic
                    if q not in self.questions:
                        self.questions.append(q)
            if note and out.get("known"):
                out["reply"] = out["reply"] + note
            if out.get("known"):                              # HOLD the new topic in mind (workspace attractor)
                self._hold(self._topic_of(resolved))
            self.history.append((t, out["reply"]))
            return out

    # relations that CHAIN (a IsA b, b IsA c => a IsA c). "has"/"resembles"/"causes" do NOT chain, and
    # pretending they do would manufacture false conclusions -- so they are deliberately excluded.
    # NOTE: bare "is" is deliberately EXCLUDED -- it marks PREDICATION ("sky is blue"), not class
    # membership, and chaining it fabricates ("sky is blue" + "blue isa color" => "sky is a color").
    # Genuine class facts use isa/kindof/typeof/subclassof; predication does not license deduction.
    _TRANSITIVE = {"isa", "partof", "part_of", "subclassof", "kindof", "typeof"}

    def _relation_graph(self):
        """subject -> [(relation, object)] over every stored triple: the brain's own knowledge graph."""
        adj = {}
        for rec in self.chat.records:
            t = rec.get("triple")
            if not t or len(t) < 3:
                continue
            adj.setdefault(" ".join(_tok(t[0])), []).append((str(t[1]), " ".join(_tok(t[2]))))
        return adj

    def _derive(self, subject, target, max_depth=6):
        """DEDUCE by chaining transitive relations over its OWN stored triples: breadth-first from
        `subject`, following only relations that legitimately chain, looking for `target`. Returns the
        PROOF as [(a, rel, b), ...] or None. This is where thinking becomes ANSWERING: the train of
        thought already produced 'dog -> mammal -> animal'; this reads the same graph and uses it to
        settle a question, showing its work instead of asserting."""
        adj = self._relation_graph()
        start, goal = " ".join(_tok(subject)), " ".join(_tok(target))
        if not start or not goal or start not in adj:
            return None
        seen, queue = {start}, [(start, [])]
        while queue:
            node, path = queue.pop(0)
            if len(path) >= max_depth:
                continue
            for rel, obj in adj.get(node, []):
                if rel.lower().replace(" ", "") not in self._TRANSITIVE:
                    continue                                  # only chain what actually chains
                step = path + [(node, rel, obj)]
                if obj == goal:
                    return step                               # proof found
                if obj not in seen:
                    seen.add(obj)
                    queue.append((obj, step))
        return None

    def _reason_question(self, t):
        """Answer a yes/no question by DERIVING it (with the proof), or say honestly that it cannot."""
        # the article group MUST require a following space: `(?:a|an|the)?` happily eats the "a" of
        # "animal" and leaves "n animal" as the target -- which is why 2-hop failed while 3-hop passed.
        m = re.match(r"^(?:is|are)\s+(?:(?:an|a|the)\s+)?(.+?)\s+(?:(?:an|a|the)\s+)?([\w' -]+?)\s*\??$",
                     t.lower())
        if not m:
            return None
        subj, targ = m.group(1).strip(), m.group(2).strip()
        proof = self._derive(subj, targ)
        if proof:
            art = lambda w: "an" if w[:1] in "aeiou" else "a"          # noqa: E731
            steps = ", and ".join("%s %s is %s %s" % (art(a), a, "part of" if "part" in r.lower() else art(b), b)
                                  for a, r, b in proof)
            self._hold(_tok(subj)[-1] if _tok(subj) else subj)
            return {"known": True, "coverage": 1.0, "ignited": True, "source": "derive",
                    "reply": "Yes -- %s. (I worked that out: %s)"
                             % (steps, " -> ".join([proof[0][0]] + [p[2] for p in proof]))}
        # No proof. Absence of a proof is NOT falsity -- say what it actually knows, and ask.
        known_chain, _ff = self.chat.think(subj, max_steps=5)
        if len(known_chain) >= 2:
            self.questions.append("is %s %s? i could not derive it -- can you tell me?" % (subj, targ))
            return {"known": False, "coverage": 0.0, "ignited": False, "source": "derive-fail",
                    "reply": "I cannot derive that from what I know -- and not being able to prove it is "
                             "not the same as it being false. What I do know: %s. Is it?"
                             % " -> ".join(known_chain)}
        return None

    def _members(self, cls):
        """Everything it has OBSERVED to be a `cls` (X IsA cls) -- the evidence any generalisation rests on."""
        adj = self._relation_graph()
        return sorted({s for s, edges in adj.items() for r, o in edges
                       if r.lower().replace(" ", "") in self._TRANSITIVE and o == cls})

    def _induce(self, cls, prop=None, min_members=3, min_support=0.6):
        """INDUCTION: from the specific cases it has OBSERVED, form a general rule -- and carry the
        evidence with it. Returns (relation, object, supporters, support, n_members) or None.

        Honest by construction: this is NOT proof. 4 of 4 observed birds flying does not make 'birds fly'
        true -- one penguin refutes it. So induction here refuses to generalise from fewer than
        `min_members` cases, reports the support fraction it rests on, and the reply must say it is a
        generalisation. IsA links are excluded: they are the class itself, not a property of it."""
        adj = self._relation_graph()
        mem = self._members(cls)
        if len(mem) < min_members:
            return None                                   # too little evidence to generalise honestly
        counts = {}
        for m in mem:
            for r, o in adj.get(m, []):
                if r.lower().replace(" ", "") in self._TRANSITIVE:
                    continue                              # 'is a bird' is the class, not a property
                counts.setdefault((r, o), set()).add(m)
        best = None
        for (r, o), who in counts.items():
            if prop and prop not in _content(o) and prop not in _content(r):
                continue
            support = len(who) / len(mem)
            if support >= min_support and len(who) >= min_members:
                if best is None or support > best[3]:
                    best = (r, o, sorted(who), support, len(mem))
        return best

    def _abduce(self, text):
        """ABDUCTION: infer the BEST EXPLANATION of what was observed. Ranks candidate causes by
        EXPLANATORY SCOPE -- the cause accounting for the MOST of the observation wins (rain explains
        wet grass AND dark clouds; a sprinkler only the grass). Returns [(cause, explained), ...] best
        first, or None. Honest: this is a guess, never a proof, and rivals are named rather than hidden."""
        adj = self._relation_graph()
        q = set(_content(text))
        if not q:
            return None
        explained = {}
        for s, edges in adj.items():
            for r, o in edges:
                if "cause" not in r.lower():
                    continue
                ow = set(_content(o))
                if ow and ow <= q:                        # this effect was actually observed
                    explained.setdefault(s, set()).add(o)
        if not explained:
            return None
        return sorted(explained.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    def _reason_induce(self, t):
        """'can birds fly' / 'can a wren fly' -> generalise from observations, or project onto an unseen case."""
        # the article group MUST require a trailing space, else "(?:a|an|the)?" eats the "a" of the
        # subject ("can animals run" -> subj "nimals"). Same fix already applied in _reason_question.
        m = re.match(r"^(?:can|do|does|are)\s+(?:(?:an|a|the)\s+)?([\w' -]+?)\s+([\w' -]+?)\s*\??$", t.lower())
        if not m:
            return None
        subj, prop = m.group(1).strip(), m.group(2).strip()
        cls = subj[:-1] if subj.endswith("s") and len(subj) > 3 else subj      # birds -> bird
        got = self._induce(cls, prop=prop)
        if got:
            r, o, who, support, n = got
            self._hold(cls)
            hedge = "all %d" % n if support >= 0.999 else "%d of the %d" % (len(who), n)
            return {"known": True, "coverage": round(support, 2), "ignited": True, "source": "induce",
                    "reply": "Probably -- %s %ss I have seen can %s (%s). So I generalise that %ss %s. "
                             "That is INDUCTION from what I observed, not a proof: one counterexample "
                             "would break it." % (hedge, cls, o, ", ".join(who[:4]), cls, o)}
        # unseen case: project a class rule onto it if we can place it in the class -- else ASK
        adj = self._relation_graph()
        node = " ".join(_tok(subj))
        parent = next((o for r, o in adj.get(node, []) if r.lower().replace(" ", "") in self._TRANSITIVE), None)
        if parent:
            got = self._induce(parent, prop=prop)
            if got:
                r, o, who, support, n = got
                return {"known": True, "coverage": round(support, 2), "ignited": True, "source": "induce",
                        "reply": "I have never seen a %s do that. But a %s is a %s, and %d of the %d %ss "
                                 "I have seen can %s -- so probably yes. That is a generalisation, not a "
                                 "proof." % (subj, subj, parent, len(who), n, parent, o)}
        if self._members(cls):
            return None
        q = "is a %s a kind of something i know? i have never seen one." % subj
        if q not in self.questions:
            self.questions.append(q)
        return None

    def _reason_abduce(self, t):
        """'why is the grass wet' / 'what happened' -> the best explanation, honestly hedged."""
        if not re.search(r"^\s*why\b|\bwhat happened\b|\bwhat explains\b|\bhow come\b", t.lower()):
            return None
        ranked = self._abduce(t)
        if not ranked:
            return None
        best, explained = ranked[0]
        rivals = [c for c, e in ranked[1:] if len(e) < len(explained)]
        tied = [c for c, e in ranked[1:] if len(e) == len(explained)]
        self._hold(best)
        if tied:
            self.questions.append("was it %s or %s? both explain what you told me." % (best, tied[0]))
            return {"known": True, "coverage": 0.5, "ignited": True, "source": "abduce",
                    "reply": "Either %s or %s -- both explain %s equally well, so I cannot choose "
                             "honestly. What else did you notice?"
                             % (best, tied[0], " and ".join(sorted(explained)))}
        why = "it explains %s" % " and ".join(sorted(explained))
        alt = ("  (%s would explain only %s.)"
               % (rivals[0], " and ".join(sorted(dict(ranked)[rivals[0]])))) if rivals else ""
        return {"known": True, "coverage": 0.75, "ignited": True, "source": "abduce",
                "reply": "Best guess: %s -- %s.%s That is ABDUCTION: the best explanation I have, "
                         "not a proof." % (best, why, alt)}

    def _topic_of(self, text):
        """The topic a turn is about: its most specific content word (what the brain now holds in mind)."""
        c = [w for w in _content(text) if len(w) > 2]
        return c[-1] if c else None

    def _hold(self, topic):
        """Keep the topic ALIVE in the PERSISTENT workspace. The workspace's cross-tick attractor +
        hysteresis IS the 'held thought' -- rebuilding a fresh workspace every turn (which the first
        version did) throws that away, which is exactly why the brain could not follow a conversation.

        The incumbent CANNOT be displaced in one tick (its hysteresis defends the slot -- train_of_thought
        found the same property): a switch needs the old attractor to decay while the new content is
        re-posted. So post the new topic across a few ticks. That is not a workaround, it is the
        behaviour we want: a sustained new topic takes over, a passing mention does not."""
        if not topic:
            return
        held = None
        for _ in range(5):                                # let the old topic decay while the new one drives
            self.ws.submit("topic", topic, 0.95)
            held = self.ws.step()
        self.topic = held or topic

    def _resolve(self, t):
        """Resolve referring expressions against the topic HELD in the workspace. Returns (text, note)."""
        low = t.lower()
        toks = _tok(low)
        if not self.topic:
            return t, ""
        m = re.match(r"^(?:and\s+)?(?:what|how)\s+about\s+(?:a|an|the)?\s*(.+?)\??$", low)
        if m:                                                 # "what about a cat" -> reuse the last frame
            return "what is %s" % m.group(1).strip(), f"  (you asked about {m.group(1).strip()}; I was holding '{self.topic}')"
        if any(p in toks for p in ("it", "that", "they", "them", "this", "those")):
            # NOT "that": it is usually a complementizer ("i heard that dogs bark"), and substituting the
            # held topic for it corrupts the claim. Only true pronouns/demonstratives resolve to the topic.
            sub = re.sub(r"\b(it|they|them|this|those)\b", self.topic, low)
            return sub, f"  (taking 'it' to mean '{self.topic}')"
        return t, ""

    def _meta(self, t):
        """Turns ABOUT the conversation or about the brain's own thinking. These MUST NOT hit the
        dictionary: 'tell me more' once returned a biography of Thomas More, and 'what were you thinking
        about' returned the definition of the word 'thinking'. They are answered from real state, and
        honestly say so when there is none."""
        low = t.lower().strip(" ?.!")
        def out(reply, src):
            return {"known": True, "coverage": 1.0, "reply": reply, "ignited": True, "source": src}

        # a GREETING-ONLY turn ("hi", "hello there!") is a greeting; "hi what continent is france in"
        # carries a real question -> do NOT swallow it as a greeting (fall through to the question).
        if re.fullmatch(r"(hi|hello|hey|yo|greetings)[\s!,.]*", low):
            return out("Hello. I am a spiking brain. I answer from what I have learned, and I keep "
                       "learning while we talk -- ask me something, or correct me.", "greet")
        if re.search(r"\bwhat (were|are) you thinking\b|\bwhat's on your mind\b|\bany thoughts\b", low):
            if not self.thoughts:
                return out("Nothing yet -- I have not had a train of thought since I woke up. "
                           "Give me a topic and I will think about it.", "introspect")
            chain, why = self.thoughts[-1]
            return out("I was thinking: %s  (it started from %s)" % (" -> ".join(chain), why), "introspect")
        if re.search(r"\bwhat did i (just )?(ask|say)\b|\bwhat was my (last )?question\b", low):
            prev = [h for h in self.history if h[0].lower().strip(" ?.!") != low]
            if not prev:
                return out("Nothing yet -- that is the first thing you have asked me.", "recall-turn")
            return out("You asked: %r" % prev[-1][0], "recall-turn")
        if re.search(r"\bwhat are we (talking|discussing)\b|\bwhat is the topic\b", low):
            if not self.topic:
                return out("Nothing yet -- we have not settled on a topic.", "topic")
            return out("We are on %r -- that is what I am holding in mind." % self.topic, "topic")
        if re.search(r"\bask me (a|another)? ?question\b|\bwhat (would you like|do you want) to (learn|know)\b"
                     r"|\banything you want to ask\b|\bwhat are you curious about\b", low):
            if not self.questions:
                return out("Nothing right now -- I have not found a hole in what I know yet. Give me a "
                           "moment to test myself, or ask me something I cannot answer.", "ask")
            q = self.questions.popleft()                   # a REAL question, found by testing itself
            return out("Yes -- %s" % q, "ask")
        if re.search(r"\bwhat do you not know\b|\bwhat don't you know\b|\bwhat are your gaps\b", low):
            if not self.questions:
                return out("I have not caught myself out yet this session. I find gaps by testing my own "
                           "derivations -- ask me something hard and I will find one.", "gaps")
            top = list(self.questions)[:3]
            return out("Honestly: %d things I have caught myself on. For instance -- %s"
                       % (len(self.questions), "  |  ".join(top)), "gaps")
        if re.search(r"\bhave you tested yourself\b|\bdid you test yourself\b|\bself.?test\b", low):
            posed, failed = self.selftests
            if not posed:
                return out("Not yet -- I test myself in the background by inventing derivations from my "
                           "own knowledge and trying to prove them.", "selftest")
            return out("Yes: I posed myself %d problems from my own knowledge and failed %d of them. "
                       "The failures became questions I want to ask you." % (posed, failed), "selftest")
        if re.fullmatch(r"(tell me more|more|go on|continue|elaborate|and\?)", low):
            if not self.topic:
                return out("More about what? I am not holding a topic yet -- ask me something first.", "elaborate")
            chain, _ff = self.chat.think(self.topic, max_steps=6)
            if len(chain) >= 2:
                self.thoughts.append((chain, "you asked me to go on about %r" % self.topic))
                return out("More on %r -- following it: %s" % (self.topic, " -> ".join(chain)), "elaborate")
            return out("I know %r but I have not linked it to anything else yet, so I cannot go deeper "
                       "honestly." % self.topic, "elaborate")
        return None

    def think(self, seed, max_steps=8):
        with self._lock:
            return self.chat.think(seed, max_steps)

    # ---- the developmental loop -------------------------------------------------------------------
    def develop(self, mode="auto", **cfg):
        """Run the developmental curriculum. mode='perceptual' -> the FULL conv-STDP digit curriculum
        (torch/GPU, .venv-cu128), consolidating exemplars to the SSD lake and emitting each learned
        concept to the durable binary knowledge. mode='symbolic' -> the torch-free stand-in.
        mode='auto' -> perceptual if torch is importable, else symbolic."""
        if mode in ("auto", "perceptual"):
            try:
                from snn_data_lake.reasoning.developmental_core import cfg_from_env, develop_cycles   # LAZY: torch tier
                c = {**cfg_from_env(), **cfg}
                res = develop_cycles(c, lake=self.lake, reflector=self.reflector, emit=self.emit_concept)
                save_state("develop_state.json", {"rff": res["rff"], "cycles": res["cycles"],
                                                  "online": res["online"], "rbf": res["rbf"],
                                                  "links": len(res["links"])})
                return res
            except Exception as e:
                if mode == "perceptual":
                    raise
                print(f"[develop] perceptual needs torch/GPU (.venv-cu128) -- {type(e).__name__}; "
                      f"using the symbolic stand-in.")
        return self._develop_symbolic(**cfg)

    def _develop_symbolic(self, n_concepts=6, dim=16, **_):
        """Torch-FREE developmental learning: an UNSUPERVISED competitive-learning (FEELING) layer sees
        clustered samples LABEL-FREE and its winner-take-all units SELF-ORGANIZE into class-family
        prototypes -- real fire-together Hebbian competition + learned lateral inhibition, not the old
        synthetic _sym_proto stand-in. Each self-organized prototype is reflected on via the spiking
        train of thought, consolidated to the durable AER lake, and emitted exactly like the perceptual
        loop. HONEST: still synthetic INPUT (not a camera), but the prototypes are now genuinely learned
        by unsupervised STDP-style competition. Deterministic (seeded)."""
        rng = random.Random(self.seed + 909)
        learner = CompetitiveLearner(n_inputs=dim, n_units=n_concepts, seed=self.seed + 909)
        samples = [_sym_proto(c, dim) for c in range(n_concepts)]
        for _ in range(24 * n_concepts):                          # UNLABELED presentations -> WTA self-organization
            learner.present(samples[rng.randrange(n_concepts)], rng, steps=8, learn=True)
        got = []
        for j in range(n_concepts):
            wj = learner.w[j]
            m = max(wj) or 1.0
            proto = [x / m for x in wj]                            # the SELF-ORGANIZED prototype, scaled to [0,1]
            with self._lock:
                self.reflector.register(str(j), proto)
                chain = self.reflector.reflect(str(j))
                self.lake.consolidate([proto], payloads=[j])      # durable AER exemplar
            self.emit_concept(j, proto, chain)
            got.append((j, chain))
        return {"mode": "symbolic-unsup", "concepts": [j for j, _ in got],
                "links": self.reflector.association_graph()}

    # ---- the always-on background learning sources (each -> durable binary facts) ------------------
    def _source_knowledge(self):
        """Learn ONE scholarly concept from the internet through the guarded (allowlist + SSRF) gate."""
        todo = [t for t in _TOPICS if t not in self._topics_done]
        if not todo:
            return None
        topic = todo[0]
        try:
            from snn_data_lake.reasoning.web_ingest_gate import WebAccessGate
            title, extract = WebAccessGate().fetch_wikipedia(topic)
        except Exception:
            return None                                        # transient (403/timeout): retry later, don't consume budget
        self._topics_done.add(topic)                          # mark done only after a SUCCESSFUL fetch
        if not extract:
            return None
        sentence = extract.split(". ")[0].strip()
        self._remember({"cue": _tok(title), "answer": f"{title}: {sentence}.", "src": f"wikipedia:{title}"})
        return f"knowledge: learned {title}"

    def _source_skills(self):
        """Discover a NEW skill from unlabeled data (open-world DP-means over frozen signatures)."""
        from snn_data_lake.reasoning.open_world_loop import OpenWorldController, dna_stream
        from snn_data_lake.reasoning.skill_builder import math_stream
        if self._skill_ctrl is None:
            self._skill_ctrl = OpenWorldController(1.7, 240, 25, 10)
            srcs = {"dna": dna_stream(n=600, ctx=3, seed=self.seed),
                    "math": math_stream(n_expr=200, ctx=3, seed=self.seed)[:600]}
            mixed, n = [], max(len(v) for v in srcs.values())
            for i in range(n):                                  # interleave -> a genuinely mixed stream
                for lbl, pairs in srcs.items():
                    if i < len(pairs):
                        mixed.append((pairs[i][0], pairs[i][1], lbl))
            self._skill_stream, self._skill_pos = mixed, 0
        fed = 0
        while self._skill_pos < len(self._skill_stream) and fed < 300:
            c, nx, lbl = self._skill_stream[self._skill_pos]
            self._skill_ctrl.observe(c, nx, lbl, self._skill_pos)
            self._skill_pos += 1
            fed += 1
        if fed == 0:
            return None
        new = [d for d in sorted(self._skill_ctrl.cents)
               if d not in self._skills_seen and len(self._skill_ctrl.items.get(d, [])) >= 120]
        for d in new:
            self._skills_seen.add(d)
            self._remember({"cue": ["skill", str(d)],
                            "answer": f"skill {d} is a pattern i discovered by myself in unlabeled data.",
                            "src": "skill-discovery"})
        return f"skills: discovered {len(new)} new" if new else None

    def _source_thinking(self):
        """THINK: run a train of thought through the spiking workspace over what is on its mind (the held
        topic) or something it recently learned, and REMEMBER having thought it. This is what makes
        'what were you thinking about' answerable with something true instead of a dictionary lookup --
        the brain has an actual (small) mental life between turns, not just ingestion."""
        seed = self.topic
        if not seed:
            with self._lock:
                cands = [r["triple"][0] for r in self.chat.records[-400:] if r.get("triple")]
            if not cands:
                return None
            seed = cands[self._rng.randrange(len(cands))]
        with self._lock:
            chain, _ff = self.chat.think(seed, max_steps=6)
        if len(chain) < 2:
            return None
        why = "it was on my mind" if seed == self.topic else "it came up in what i was learning"
        self.thoughts.append((chain, why))
        return "thought: %s" % " -> ".join(chain)

    def _source_selftest(self):
        """CURIOSITY, pointed at itself: invent a problem from its OWN knowledge, try to solve it, and
        turn every failure into a question it WANTS to ask. Two kinds of problem it can pose honestly:
          (1) a DERIVATION it should be able to make (a IsA b, b IsA c => 'is a a c?') -- then actually
              try to derive it. Success = it verified itself; failure = a real hole in its reasoning.
          (2) a DANGLING concept -- a word it stores with NO relation to anything. It cannot think about
              such a word at all, only recite it, so it asks what it IS.
        This is where the brain stops waiting to be taught and starts interrogating itself."""
        with self._lock:
            adj = self._relation_graph()
        if not adj:
            return None
        # (1) SEEK a 2-hop derivation it can genuinely pose. Picking a node blindly almost always lands
        # on one whose relations do not chain ('resembles', 'continent'), wasting the test -- a mind
        # testing itself should look for a question it can actually put to itself.
        chainable = []
        for node, edges in adj.items():
            for rel, mid in edges:
                if rel.lower().replace(" ", "") not in self._TRANSITIVE:
                    continue
                for rel2, end in adj.get(mid, []):
                    if rel2.lower().replace(" ", "") in self._TRANSITIVE and end != node:
                        chainable.append((node, mid, end))
        if chainable:
            node, mid, end = chainable[self._rng.randrange(len(chainable))]
            with self._lock:
                proof = self._derive(node, end)
                self.selftests[0] += 1
                if proof:
                    self.thoughts.append(([node, mid, end], "i tested myself: is %s a %s?" % (node, end)))
                    return "self-test PASSED: derived %s -> %s -> %s" % (node, mid, end)
                self.selftests[1] += 1
                q = "is %s a %s? i can see the steps but could not derive it." % (node, end)
                if q not in self.questions:
                    self.questions.append(q)
            return "self-test FAILED: could not derive %s -> %s (now a question)" % (node, end)
        # (2) a concept it stores but has no relations for -> it can only recite it, so ASK
        with self._lock:
            cues = [next(iter(c)) for c in self.chat.cues[-500:] if len(c) == 1]
        dangling = [w for w in cues if w not in adj][:1]
        if dangling:
            q = "what is %s? i know the word but nothing about what it IS." % dangling[0]
            with self._lock:
                if q not in self.questions:
                    self.questions.append(q)
                    return "curious: found a word i can only recite -- %r" % dangling[0]
        return None

    def _source_reflection(self):
        """THINK about a concept it already knows -> new association facts (fire together, wire together)."""
        with self._lock:
            concepts = sorted(self.reflector.protos)
        if len(concepts) < 2:
            return None
        c = concepts[self._rng.randrange(len(concepts))]
        with self._lock:
            chain = self.reflector.reflect(c)
        if len(chain) < 2:
            return None
        for e in chain[1:]:
            self._remember({"cue": [c, str(e)], "answer": f"digit {c} resembles digit {e}.",
                            "triple": [c, "resembles", str(e)], "src": "reflection"})
        return f"reflection: {' -> '.join(chain)}"

    # ---- the ALWAYS-ON background learner ---------------------------------------------------------
    def start_learning(self, sources=("knowledge", "thinking", "selftest", "skills", "reflection"),
                       interval=2.0, budget=None):
        """Start learning CONTINUOUSLY in a background daemon thread while the foreground converses.
        `budget` (self-check only) stops after N successful learning steps, so the thread is bounded
        and deterministic. PERCEPTION is deliberately NOT a background source (CUDA in a daemon thread
        is fragile) -- run it with develop(mode='perceptual') / `--develop` on the GPU venv."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._learn_loop, args=(tuple(sources), interval, budget),
                                        daemon=True)
        self._thread.start()
        return self

    def join_learning(self, timeout=120.0):
        """Wait for a BOUNDED learner (started with a `budget`) to finish its budget naturally. Used by
        the self-check for determinism -- unlike stop_learning(), it does not cut the loop short."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self

    def stop_learning(self, timeout=60.0):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self

    @property
    def learning(self):
        return self._running

    def _learn_loop(self, sources, interval, budget):
        fns = {"knowledge": self._source_knowledge, "skills": self._source_skills,
               "reflection": self._source_reflection, "thinking": self._source_thinking,
               "selftest": self._source_selftest}
        i, done = 0, 0
        while self._running and (budget is None or done < budget):
            name = sources[i % len(sources)]
            i += 1
            try:
                got = fns[name]()
            except Exception as e:                        # a source must never kill the always-on loop
                self.learned_log.append(f"{name}: error ({type(e).__name__})")
                got = None                                # an error is NOT learning -> don't consume budget
            if got:
                done += 1
                self.learned_log.append(got)
            if self._running and interval:
                time.sleep(interval)
        self._running = False

    def close(self):
        self.stop_learning()
        for closeable in (self.lake, getattr(self.chat, "lake", None)):   # also the chat's per-boot temp .spkl
            try:
                closeable.close()
            except Exception:
                pass

    # ---- SELF-REWRITE: the brain does what a Godel machine does ------------------------------------
    def self_rewrite(self, name, rewrite, repo=None, verify=None, utility=None):
        """Attempt ONE proof-gated rewrite of the brain's OWN codebase through the Godel-machine kernel:
        `rewrite(worktree)` GENERATES code into an ISOLATED git worktree (the native self-rewrite path),
        the kernel VERIFIES it FROM OUTSIDE, and COMMITS only on strict utility gain. Real-mode default:
        repo = this repo; gate = run_selfchecks + ruff; utility = the brain's own competence on a pinned
        probe battery. A rewrite may NOT touch the fixed SAFETY KERNEL (the 5 brainstem arm-safety files
        + run_selfchecks + godel_machine), so it cannot weaken the E-STOP / arm limits nor co-rewrite
        the gate to hide it. One immutable meta-rule (commit only on external proof of strict gain);
        everything else rewritable. Not auto-pushed; invoked deliberately. Returns (accepted, reason)."""
        from snn_data_lake.reasoning.godel_machine import GodelMachine, attempt_on_repo
        repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if verify is None and utility is None:
            return attempt_on_repo(repo, name, rewrite, self._competence_probe)
        return GodelMachine(repo, verify or (lambda _w: True),
                            utility or self._competence_probe).attempt(name, rewrite)

    @staticmethod
    def _competence_probe(worktree):
        """The brain's own competence, MEASURED FROM OUTSIDE the candidate: boot a Brain inside the
        worktree and score a behavioural battery. ANTI-GOODHART: the battery is generated PER ATTEMPT
        from RANDOM parameters (random arithmetic/geometric series, random analogies) with a system-
        entropy seed the rewrite-author cannot know at write time, and scored by EXACT last-token match
        -- so a rewrite that special-cases a few fixed query strings cannot game it; only a genuine
        capability gain raises the score. Returns accuracy in [0,1], or None if it could not boot."""
        import random
        import subprocess
        import tempfile
        rng = random.Random()                                  # system entropy: unknown to the rewrite author
        battery = []
        for _ in range(8):
            a, d = rng.randint(1, 9), rng.randint(2, 6)
            if rng.random() < 0.5:                             # arithmetic series
                seq = [a + d * i for i in range(4)]
                battery.append((f"what comes next in {' '.join(map(str, seq))}", str(a + d * 4)))
            else:                                              # geometric series
                r = rng.randint(2, 4)
                seq = [a * r**i for i in range(4)]
                battery.append((f"what comes next in {' '.join(map(str, seq))}", str(a * r**4)))
        src = ("import os, re, sys; sys.path[:0] = ['src']\n"
               "from snn_data_lake.reasoning.brain_loop import Brain\n"
               "b = Brain(); b.boot()\n"
               f"bat = {battery!r}\n"
               "ok = 0\n"
               "for q, a in bat:\n"
               "    rep = b.converse(q).get('reply') or ''\n"
               "    nums = re.findall(r'-?\\d+', rep)\n"        # EXACT answer = the induced value, not a substring
               "    ok += 1 if nums and nums[-1] == a else 0\n"
               "try:\n    b.stop_learning(); b.join_learning()\nexcept Exception:\n    pass\n"
               "b.close(); print('UTIL', ok / len(bat))\n")
        probe = os.path.join(tempfile.mkdtemp(prefix="probe_"), "brain_competence_probe.py")
        with open(probe, "w") as f:
            f.write(src)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "NORD_HOME": tempfile.mkdtemp(prefix="probe_home_")}
        r = subprocess.run([sys.executable, probe], cwd=worktree, env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("UTIL "):
                return float(line.split()[1])
        return None

    # ---- the interactive foreground ---------------------------------------------------------------
    def repl(self):
        print("\nUNIFIED BRAIN -- i am learning in the background while we talk.")
        print("  /learn on|off   start/stop the always-on background learner")
        print("  /develop        run the developmental curriculum (GPU perception if available)")
        print("  /think X        a train of thought from X    /teach q | a   teach me a fact")
        print("  /n              what i know                  /quit\n")
        while True:
            try:
                text = input("you  : ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/n":
                print(f"brain: i know {self.chat.n_known} things; {self.n_learned} concepts; "
                      f"background learner {'ON' if self._running else 'off'}; "
                      f"learned this session: {len(self.learned_log)}\n")
                continue
            if text.startswith("/learn "):
                if text.endswith("on"):
                    self.start_learning()
                    print("brain: background learning ON -- i keep learning while we talk.\n")
                else:
                    self.stop_learning()
                    print("brain: background learning off.\n")
                continue
            if text == "/develop":
                res = self.develop()
                print(f"brain: developed -> {res.get('mode', 'perceptual')}; i now know "
                      f"{self.chat.n_known} things.\n")
                continue
            if text.startswith("/think "):
                chain, ff = self.think(text[len("/think "):].strip())
                print(f"brain [train of thought]: {' -> '.join(chain) if chain else '(nothing ignited)'}\n")
                continue
            if text.startswith("/teach ") and "|" in text:
                q, a = text[len("/teach "):].split("|", 1)
                self.teach(_tok(q), a.strip())
                print(f"brain: learned (durable on the SSD). i know {self.chat.n_known} things.\n")
                continue
            print("brain:", self.converse(text)["reply"], "\n")
        self.close()


def _selfcheck():
    """Deterministic, OFFLINE, torch-free: the merge works and the brain never forgets (cross-process)."""
    import shutil
    import subprocess
    import tempfile

    prev = os.environ.get("NORD_HOME")
    work = tempfile.mkdtemp(prefix="snn_brain_")
    os.environ["NORD_HOME"] = work
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the src/ dir
    try:
        print("=" * 98)
        print("UNIFIED BRAIN -- chat + developmental loop merged; one raw-binary brain on the SSD")
        print("=" * 98)
        b = Brain(seed=0)
        print(b.boot())

        # ---- DEVELOP: the loop learns concepts + reflects; every concept lands on the SSD as binary --
        dev = b._develop_symbolic(n_concepts=6)
        print(f"[develop] learned concepts {dev['concepts']}; reflection wired {len(dev['links'])} links")

        # ---- ALWAYS-ON: the background learner runs while we converse (bounded here -> deterministic) -
        b.start_learning(sources=("skills", "reflection"), interval=0, budget=3)   # offline sources only
        got = b.converse("what is digit 3")                                        # converse WHILE learning
        b.join_learning()                                  # let the BOUNDED learner finish its budget
        b.stop_learning()
        print(f"[always-on] background learner ran while conversing -> {b.learned_log}")

        # ---- CONVERSATION: it must HOLD the thread, not do stateless lookup -------------------------
        # The trap: a dictionary makes every word match something, so one-way coverage answered
        # "tell me more" with a biography of Thomas More, and "what were you thinking about" with the
        # definition of "thinking". Both must now come from conversational state.
        b.teach(["dog"], "dog: a domesticated carnivorous mammal kept as a pet.")
        b.teach(["cat"], "cat: a small domesticated carnivorous mammal with soft fur.")
        b.teach(["more"], "more: English statesman who opposed Henry VIII.")     # the trap
        c_dog = b.converse("what is a dog")
        c_cat = b.converse("what about a cat")            # elliptical follow-up -> new topic
        c_more = b.converse("tell me more")               # must elaborate on the HELD topic
        c_topic = b.converse("what are we talking about")
        c_ask = b.converse("what did i just ask you")
        c_think = b.converse("what were you thinking about")
        conv_topic = b.topic                              # snapshot NOW: later turns move the topic on
        print(f"[conversation] dog -> 'what about a cat' -> topic={conv_topic!r}; "
              f"'tell me more' -> {c_more['reply'][:46]!r}")
        print(f"[conversation] recalls the turn: {c_ask['reply'][:44]!r}; thinks: {c_think['reply'][:40]!r}")

        # ---- THINKING: derive an answer over its OWN knowledge, and be curious enough to test itself --
        b.teach(["dog"], "a dog is a mammal.", triple=["dog", "IsA", "mammal"])
        b.teach(["mammal"], "a mammal is an animal.", triple=["mammal", "IsA", "animal"])
        b.teach(["animal"], "an animal is a living thing.", triple=["animal", "IsA", "living thing"])
        d_yes = b.converse("is a dog a living thing")      # 3 transitive hops, never stated directly
        d_no = b.converse("is a dog a car")                # unprovable -> must NOT bluff
        st = b._source_selftest()                          # invent a problem from its own knowledge
        b.converse("what is a zqxwv")                      # a gap -> becomes something it wants to ask
        d_ask = b.converse("ask me a question")
        d_gaps = b.converse("what do you not know")

        # ---- the THREE forms of reasoning, each honest about what it is worth --------------------
        for bird in ("sparrow", "eagle", "hawk", "robin"):     # specific observations
            b.teach([bird], "a %s is a bird." % bird, triple=[bird, "IsA", "bird"])
            b.teach([bird, "fly"], "a %s can fly." % bird, triple=[bird, "CapableOf", "fly"])
        r_ind = b.converse("can birds fly")                    # INDUCTIVE: observations -> general rule
        b.teach(["rain"], "rain causes wet grass.", triple=["rain", "Causes", "wet grass"])
        b.teach(["rain", "clouds"], "rain causes dark clouds.", triple=["rain", "Causes", "dark clouds"])
        b.teach(["sprinkler"], "a sprinkler causes wet grass.", triple=["sprinkler", "Causes", "wet grass"])
        r_abd = b.converse("the grass is wet and the clouds are dark what happened")   # ABDUCTIVE
        print(f"[induce] {r_ind['reply'][:74]!r}")
        print(f"[abduce] {r_abd['reply'][:74]!r}")

        # ---- FLUID INTELLIGENCE: solve a NOVEL puzzle by inducing a rule at test time (through the
        # full brain), and honestly gap when no constant rule exists -- not recall, real synthesis ----
        g_series = b.converse("what comes next in 2 4 6 8")   # constant difference (spiking VSA)
        g_analogy = b.converse("abc:abd :: ijk:?")            # letter analogy
        g_geo = b.converse("continue the sequence 2 4 8 16")  # geometric -- relational integration
        g_fib = b.converse("what comes next: 1 1 2 3 5")      # order-2 recurrence
        g_prog = b.converse("what comes next in 1 10 2 20 3 30")   # interleave -- depth-2 program synthesis
        g_alpha = b.converse("alphabet q w e r t then q:w :: e:?")  # novel-symbol transfer
        g_fact = b.converse("what comes next in 1 2 6 24 120")     # unbounded program synthesis
        g_active = b.converse("what comes next in 1 4")       # underdetermined -> ask an experiment
        g_gap = b.converse("what comes next in 3 1 4 1 5 9 2 6")  # 8-term random -> honest gap (no fit even w/ //,%)
        print(f"[fluid] geometric -> {g_geo['reply'][:44]!r}; interleave -> {g_prog['reply'][:44]!r}")
        print(f"[think] 'is a dog a living thing' -> {d_yes['reply'][:78]!r}")
        print(f"[think] unprovable -> {d_no['reply'][:56]!r}")
        print(f"[curious] self-test: {st!r}; asks: {d_ask['reply'][:52]!r}")

        q3 = b.converse("what is digit 3")
        chain, ff = b.think("3")
        n_facts = b.klog.count()
        with open(b.kpath, "rb") as f:
            raw = f.read()
        proto_sz = os.path.getsize(b.ppath)
        dur = b.lake.durability_check()

        print(f"you  : what is digit 3\nbrain: {q3['reply']}")
        print(f"[think] 3 -> {' -> '.join(chain)}")
        print(f"[ssd] knowledge.bin magic={raw[:8]!r} {len(raw):,}B ({n_facts} facts, RAW BINARY) | "
              f"prototypes.bin {proto_sz:,}B (float16) | lake.spkl AER durable={dur['ok']}")

        # ---- CROSS-PROCESS never-forget: a BRAND-NEW process boots from the SSD binary and knows it ---
        child = (
            f"import sys, os\n"
            f"sys.path.insert(0, {src_dir!r})\n"
            f"os.environ['NORD_HOME'] = {work!r}\n"
            "from snn_data_lake.reasoning.brain_loop import Brain\n"
            "b = Brain(); b.boot()\n"
            "r = b.converse('what is digit 3')\n"
            "b.close()\n"
            "raise SystemExit(0 if r['known'] and 'numeral' in r['reply'] else 3)\n"
        )
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
        rc = proc.returncode
        if rc != 0:                                        # surface the child's real error, never hide it
            print(f"[cross-process] child FAILED rc={rc}:\n{(proc.stderr or '')[-800:]}")
        print(f"[cross-process] a FRESH process booted from knowledge.bin and knew digit 3 (rc={rc})")
        print("=" * 98)

        # ---- self-checks ----
        # ---- the conversation must be REAL (these all shipped broken once) ----
        assert c_dog["known"] and "domesticated" in c_dog["reply"], "a plain question must still answer"
        assert conv_topic == "cat", f"a follow-up must MOVE the held topic (got {conv_topic!r})"
        assert "cat" in c_more["reply"] and "more:" not in c_more["reply"].lower(), (
            f"'tell me more' must elaborate the HELD TOPIC, not look up the word 'more' "
            f"(Thomas More regression): {c_more['reply']!r}")
        assert "cat" in c_topic["reply"], "it must be able to say what it is holding in mind"
        assert "what about a cat" in c_ask["reply"] or "talking" in c_ask["reply"], (
            f"it must recall the actual previous turn, got {c_ask['reply']!r}")
        assert "->" in c_think["reply"] and "process of using your mind" not in c_think["reply"], (
            f"'what were you thinking' must report a REAL train of thought, not define 'thinking': "
            f"{c_think['reply']!r}")
        assert len(b.history) >= 6 and len(b.thoughts) >= 1, "it must remember the turns + the thoughts it had"
        # ---- it must THINK: derive a conclusion never stated, and SHOW the proof ----
        assert d_yes["known"] and d_yes["source"] == "derive" and "living thing" in d_yes["reply"], (
            f"must DERIVE 'a dog is a living thing' over 3 transitive hops, got {d_yes['reply']!r}")
        assert "mammal" in d_yes["reply"] and "animal" in d_yes["reply"], (
            "the derivation must SHOW ITS WORK (the intermediate steps), not just assert the conclusion")
        # ---- and it must NOT bluff: no proof is not the same as false ----
        assert not d_no["known"] and "cannot derive" in d_no["reply"], (
            f"an unprovable claim must be refused honestly, not guessed: {d_no['reply']!r}")
        assert "not the same as it being false" in d_no["reply"], (
            "it must distinguish 'i cannot prove it' from 'it is false'")
        # ---- CURIOSITY: it invents problems from its own knowledge and wants to ask ----
        assert b.selftests[0] >= 1, "it must POSE ITSELF problems from its own knowledge"
        assert len(b.questions) >= 1, "a gap it hit must become a question it WANTS to ask"
        assert d_ask["source"] == "ask" and "?" in d_ask["reply"], (
            f"'ask me a question' must produce a REAL question it found by testing itself: {d_ask['reply']!r}")
        assert d_gaps["known"] and "caught myself" in d_gaps["reply"], "it must own its gaps honestly"
        # ---- INDUCTIVE: generalise from observations, carrying the evidence, claiming only probability ----
        assert r_ind["known"] and r_ind["source"] == "induce", f"must INDUCE a rule: {r_ind['reply']!r}"
        assert "not a proof" in r_ind["reply"] and "4" in r_ind["reply"], (
            "induction must state the EVIDENCE it rests on and that it is not proof (it is defeasible)")
        # ---- ABDUCTIVE: best explanation by scope, hedged, naming the rival it rejected ----
        assert r_abd["known"] and r_abd["source"] == "abduce", f"must ABDUCE: {r_abd['reply']!r}"
        assert "rain" in r_abd["reply"] and "sprinkler" in r_abd["reply"], (
            "abduction must pick the wider-scope cause AND name the rival it rejected")
        assert "not a proof" in r_abd["reply"], "abduction must admit it is a guess, not a proof"
        # ---- FLUID: the full brain must SOLVE a novel series/analogy by test-time rule induction, and
        # honestly GAP on a non-constant relation -- routed to the spiking VSA inducer, not recall ----
        assert g_series["source"] == "gf" and "10" in g_series["reply"], (
            f"the full brain must SOLVE '2 4 6 8' -> 10 by inducing a rule, got {g_series['reply']!r}")
        assert g_analogy["source"] == "gf" and "ijl" in g_analogy["reply"], (
            f"the full brain must solve the letter analogy abc:abd::ijk -> ijl, got {g_analogy['reply']!r}")
        assert g_geo["source"] == "gf" and g_geo["known"] and "32" in g_geo["reply"], (
            f"relational integration: a geometric series 2 4 8 16 -> 32, got {g_geo['reply']!r}")
        assert g_fib["source"] == "gf" and "8" in g_fib["reply"], (
            f"a Fibonacci recurrence must solve to 8, got {g_fib['reply']!r}")
        assert g_prog["source"] == "gf" and g_prog["known"] and "-> 4" in g_prog["reply"], (
            f"program synthesis: interleaved 1 10 2 20 3 30 -> 4, got {g_prog['reply']!r}")
        assert g_alpha["source"] == "gf" and g_alpha["known"] and " r" in g_alpha["reply"], (
            f"novel-symbol transfer over a given alphabet: e -> r, got {g_alpha['reply']!r}")
        assert g_fact["source"] == "gf" and g_fact["known"] and "720" in g_fact["reply"], (
            f"unbounded program synthesis: factorial 1 2 6 24 120 -> 720, got {g_fact['reply']!r}")
        assert g_active["source"] == "gf-active" and "experiment" in g_active["reply"], (
            f"underdetermined -> the brain must ASK the discriminating term, got {g_active['reply']!r}")
        assert g_gap["source"] == "gf" and not g_gap["known"], (
            f"a patternless series (3 1 4 1 5 9 2 6) must be an HONEST Gf gap, not a bluff: {g_gap['reply']!r}")
        assert q3["known"] and "numeral" in q3["reply"], "the chat must converse from what the LOOP learned"
        assert got["known"] == q3["known"], "conversing while the background learner runs must work"
        assert len(chain) >= 2, f"a train of thought must chain the loop's reflection triples, got {chain}"
        assert len(ff) <= 1, "the feedforward control must not chain"
        assert raw[:8] == b"SNDLBIN1" and raw.lstrip()[:1] != b"{", "knowledge must be RAW BINARY, not JSON"
        assert n_facts > 0 and proto_sz > 0, "concepts + float16 prototypes must be on the SSD"
        assert len(b.learned_log) >= 1, "the ALWAYS-ON background learner must have learned something"
        assert rc == 0, f"cross-process never-forget FAILED (rc={rc}): a fresh process did not know digit 3"
        assert dur["ok"], "the durable AER exemplar lake must survive a restart"

        # ---- SELF-REWRITE: the brain drives a proof-gated Godel self-rewrite in a REAL git worktree ----
        # (hermetic throwaway repo; skipped where git is unavailable so the CI stays green)
        from snn_data_lake.reasoning.godel_machine import _have_git
        if _have_git():
            import subprocess as _sp
            import tempfile as _tf
            grepo = _tf.mkdtemp(prefix="brain_godel_")
            with open(os.path.join(grepo, "core.py"), "w") as f:
                f.write("VALUE = 1\n")
            _sp.run(["git", "-C", grepo, "init", "-q"], check=True)
            _sp.run(["git", "-C", grepo, "add", "-A"], check=True)
            _sp.run(["git", "-C", grepo, "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-q", "-m", "seed"], check=True)

            def _u(wt):
                g = {}
                exec(open(os.path.join(wt, "core.py")).read(), g)   # measured from OUTSIDE the candidate
                return g.get("VALUE")
            def _gen_better(wt):                                    # the brain GENERATES code into the worktree
                with open(os.path.join(wt, "core.py"), "w") as f:
                    f.write("VALUE = 2   # brain-generated improvement\n")
            def _gen_nogain(wt):
                with open(os.path.join(wt, "core.py"), "w") as f:
                    f.write("VALUE = 1   # rewritten, but no gain\n")

            r_ok = b.self_rewrite("raise-value", _gen_better, repo=grepo, verify=lambda _w: True, utility=_u)
            r_no = b.self_rewrite("no-gain", _gen_nogain, repo=grepo, verify=lambda _w: True, utility=_u)
            head = _sp.run(["git", "-C", grepo, "show", "HEAD:core.py"], capture_output=True, text=True).stdout
            _sp.run(["git", "-C", grepo, "worktree", "prune"], check=False)
            import shutil as _sh
            _sh.rmtree(grepo, ignore_errors=True)
            assert r_ok[0] is True and "VALUE = 2" in head, (
                "the brain must drive a proof-gated self-rewrite that COMMITS the generated code")
            assert r_no[0] is False, "a no-gain self-rewrite must be rejected (only enhance, monotonic)"
            print(f"[self-rewrite] brain drove a Godel self-rewrite in a real git worktree: "
                  f"committed VALUE 1->2 ({r_ok[1]}), rejected the no-gain rewrite")

        b.close()

        print(f"self-check OK: chat + developmental loop are ONE always-on brain -- the loop learned "
              f"{len(dev['concepts'])} concepts + reflected ({len(dev['links'])} links), the background "
              f"learner learned while the foreground conversed, and the chat answers from what the loop "
              f"learned ('{q3['reply'][:52]}...') + chains it ({' -> '.join(chain)}). All of it is RAW "
              f"BINARY on the SSD ({n_facts} facts in knowledge.bin, float16 prototypes, AER lake durable) "
              f"and a BRAND-NEW process boots from it knowing everything -- it never forgets. Torch-free "
              f"path; GPU perception via --develop.")
        return 0
    finally:
        os.environ.pop("NORD_HOME", None)
        if prev is not None:
            os.environ["NORD_HOME"] = prev
        shutil.rmtree(work, ignore_errors=True)


def main():
    # Default (no flags) = the deterministic self-check, so `python research/brain_loop.py` is SAFE for
    # run_selfchecks even from a real terminal (an isatty() check would launch the REPL and HANG the gate).
    # `--repl` wakes the real always-on brain; `--develop` runs the curriculum (GPU perception if available).
    args = sys.argv[1:]
    if not ("--repl" in args or "--brain" in args or "--develop" in args):
        return _selfcheck()
    b = Brain(seed=0)
    print(b.boot())
    if "--develop" in args:
        res = b.develop(mode="perceptual" if "--gpu" in args else "auto")
        print(f"[develop] done -> {res.get('mode', 'perceptual')}; the brain knows {b.chat.n_known} things.")
    if "--repl" in args or "--brain" in args:
        b.start_learning()                                # ALWAYS-ON: keep learning while we talk
        b.repl()
    else:
        b.close()
    return 0


if __name__ == "__main__":
    main()
