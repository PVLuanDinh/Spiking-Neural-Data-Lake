r"""
AUTONOMOUS ENRICH -- the brain's self-improvement loop that GROWS its durable lake from scholarly
knowledge on the internet, so the chat learns new concepts BETWEEN sessions instead of starting blank.

WHAT IT DOES: each round it CURIOSITY-picks a topic from a frontier (preferring topics it has not
learned yet), fetches that topic's short encyclopedic summary from Wikipedia through the ONE guarded
doorway (reasoning/web_ingest_gate.py: host allowlist + SSRF firewall + hard size cap + provenance log),
and stores it as ONE durable fact in the shared KnowledgeLog (substrate/home.py). Dedup keeps the lake
lean -- re-seeing a known topic never writes it twice. Because the log is append-only + fsynced, a fact
learned in one run is REPLAYED by every later process: the brain never forgets what it learned.

"LEARNING AT ALL TIMES" -- HONEST MEANING: this loop learns only while ITS PROCESS is alive. There is
no hidden always-on daemon. For genuine round-the-clock learning, run the `--forever` mode in the
background, or register it as a Windows Scheduled Task so the OS keeps it running / restarts it. One line:

  schtasks /Create /SC ONLOGON /TN snn-enrich /TR ^
    "cmd /c cd /d D:\AI-project\spiking-neural-data-lake && set PYTHONUTF8=1 && set PYTHONPATH=src && ^
     .venv-lake\Scripts\python -m snn_data_lake.reasoning.autonomous_enrich --forever"

  (or just: `PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.autonomous_enrich --forever &`)

SOURCES & LICENCE -- HONEST: it fetches SHORT encyclopedic summaries from Wikipedia (CC BY-SA) for
PERSONAL LOCAL use only -- not republishing, not the full article. Every fetch is scholarly and
allowlisted, and passes the SSRF firewall + size cap before a byte is read; the provenance log records
host + bytes + source. Sources: en.wikipedia.org / simple.wikipedia.org REST summary API.

  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.autonomous_enrich             # offline self-check demo
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.autonomous_enrich --once      # learn ONE new topic (live)
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.autonomous_enrich --rounds 20 # learn N topics (live)
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.autonomous_enrich --forever   # learn at all times (Ctrl-C to stop)
Deps: stdlib only. Reuses web_ingest_gate.WebAccessGate (the doorway) + substrate.home.KnowledgeLog (durable memory).
"""

import argparse
import os
import random
import re
import sys
import time

from snn_data_lake.reasoning.web_ingest_gate import WebAccessGate
from snn_data_lake.substrate.home import KnowledgeLog

# words that carry no retrieval signal -- stripped from cues (incl. the literal "what"/"is" of a query)
_STOP = {
    "what",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "these",
    "those",
    "as",
    "at",
    "by",
    "from",
    "it",
    "its",
    "into",
    "than",
    "then",
    "so",
    "such",
    "used",
    "also",
    "which",
    "who",
    "whom",
}

# REAL scholarly frontier -- foundational math / biology / physics / chemistry concepts.
SEED_FRONTIER = [
    "Calculus",
    "Cell (biology)",
    "Photosynthesis",
    "DNA",
    "Mitochondrion",
    "Derivative",
    "Integral",
    "Evolution",
    "Entropy",
    "Gravity",
    "Neuron",
    "Molecule",
    "Atom",
    "Thermodynamics",
    "Prime number",
]

# bundled REAL short encyclopedic summaries -- the OFFLINE path so the self-check needs NO network.
# topic -> (canonical_title, plain-text summary). Kept short + factual (personal local use).
_BUNDLED_SUMMARIES = {
    "Calculus": (
        "Calculus",
        "Calculus is the mathematical study of continuous change, in the same way that geometry is "
        "the study of shape and algebra is the study of generalizations of arithmetic operations. It "
        "has two major branches, differential calculus and integral calculus.",
    ),
    "Cell (biology)": (
        "Cell (biology)",
        "The cell is the basic structural and functional unit of all known living organisms, and is "
        "the smallest unit of life that can replicate independently. Cells are often called the "
        "building blocks of life.",
    ),
    "Photosynthesis": (
        "Photosynthesis",
        "Photosynthesis is the process used by plants, algae, and some bacteria to convert light "
        "energy into chemical energy stored in sugars, releasing oxygen as a byproduct.",
    ),
    "DNA": (
        "DNA",
        "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions used in "
        "the growth, development, functioning, and reproduction of all known living organisms.",
    ),
}


def _content_words(text):
    """Lowercase content-word tokens of `text`, stopwords (incl. 'what'/'is') and 1-char tokens dropped.
    'Cell (biology)' -> ['cell', 'biology']; 'What is DNA' -> ['dna']. These are the retrieval cues."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) > 1 and t not in _STOP]


def _first_sentence(text):
    """First sentence of `text` (up to and including the first sentence-ending period), else the whole
    stripped string. Deterministic + honest: no NLP model, just a period split of an encyclopedic summary."""
    text = " ".join(text.split())  # normalise whitespace
    m = re.search(r"\.(?:\s|$)", text)
    return text[: m.start() + 1] if m else text


def _trivial_triple(title, sentence):
    """Return a structured [subject, 'is_a', object] triple ONLY when the first sentence is a clear
    copular 'X is a/an <noun>' -- otherwise None. Conservative on purpose: we never guess a fact."""
    m = re.match(rf"^{re.escape(title)}\s+(?:is|are|was)\s+(?:a|an)\s+([a-z]+)\b", sentence)
    return [title.lower(), "is_a", m.group(1)] if m else None


class AutonomousEnricher:
    """The brain's self-improvement loop: curiosity-pick a frontier topic it has not learned, fetch its
    scholarly summary through the guarded gate, and store it as ONE durable, dedup'd fact in the lake.

    Durability + never-forget come from KnowledgeLog (append-only + fsync). Leanness comes from dedup:
    a topic already learned is never written twice, so `--forever` can re-see the frontier without bloat."""

    def __init__(self, gate=None, log=None, frontier=None, expand=True, max_frontier=2000, seed=0):
        self.gate = gate if gate is not None else WebAccessGate()
        self.log = log if log is not None else KnowledgeLog()  # durable shared home
        self.frontier = list(frontier if frontier is not None else SEED_FRONTIER)
        self.expand = bool(expand)
        self.max_frontier = int(max_frontier)
        self._rng = random.Random(seed)  # per-instance RNG, never the module-global stream
        self._attempted = set()  # topics tried this session
        # SEED curiosity from what the lake ALREADY knows: any topic whose src is on disk is "learned".
        self._learned_src = {r.get("src") for r in self.log.replay() if r.get("src")}

    def _is_learned(self, topic):
        return f"wikipedia:{topic}" in self._learned_src or topic in self._attempted

    def pick_topic(self):
        """CURIOSITY: prefer a frontier topic not yet learned/attempted; if all are, re-pick at random
        (so `--forever` keeps cycling). Returns None only if the frontier is empty."""
        if not self.frontier:
            return None
        unlearned = [t for t in self.frontier if not self._is_learned(t)]
        pool = unlearned if unlearned else self.frontier
        return self._rng.choice(pool)

    def _expand_frontier(self, extract):
        """Bounded curiosity growth: add capitalized noun-like terms from the summary as NEW topics
        (dedup'd, capped at max_frontier). Off in the deterministic self-check (expand=False)."""
        if not self.expand or len(self.frontier) >= self.max_frontier:
            return 0
        seen = set(self.frontier)
        added = 0
        for term in re.findall(r"\b[A-Z][a-z]{3,}\b", extract):
            if term not in seen and term not in _STOP:
                self.frontier.append(term)
                seen.add(term)
                added += 1
                if added >= 3 or len(self.frontier) >= self.max_frontier:  # bounded per round
                    break
        return added

    def learn_topic(self, topic, simple=False):
        """Fetch `topic`'s summary through the gate and store it as a durable fact. Returns a result
        dict {topic,title,learned,written,count}. learned=False (nothing stored) for a missing page."""
        self._attempted.add(topic)
        title, extract = self.gate.fetch_wikipedia(topic, simple=simple)
        if not extract:  # missing / redirect / disambiguation
            return {
                "topic": topic,
                "title": title,
                "learned": False,
                "written": False,
                "count": self.log.count(),
            }
        sentence = _first_sentence(extract)
        record = {
            "cue": _content_words(title),  # retrieval keys (title content words)
            "answer": f"{title}: {sentence}",  # the reply the chat speaks back
            "src": f"wikipedia:{title}",  # provenance of the learned fact
        }
        triple = _trivial_triple(title, sentence)
        if triple is not None:
            record["triple"] = triple
        written = self.log.append(record)  # durable + dedup'd
        self._learned_src.add(f"wikipedia:{title}")
        self._expand_frontier(extract)
        return {
            "topic": topic,
            "title": title,
            "learned": True,
            "written": written,
            "count": self.log.count(),
        }

    def run_once(self, simple=False):
        """Learn ONE curiosity-picked topic. Returns the learn_topic result (or a no-op if frontier empty)."""
        topic = self.pick_topic()
        if topic is None:
            return {
                "topic": None,
                "title": None,
                "learned": False,
                "written": False,
                "count": self.log.count(),
            }
        return self.learn_topic(topic, simple=simple)

    def run_rounds(self, n, simple=False):
        """Learn up to n topics; return the list of per-round results."""
        return [self.run_once(simple=simple) for _ in range(int(n))]


# --------------------------------------------------------------------------------------------------
# OFFLINE self-check: a fake gate serves bundled REAL summaries with NO network, into a temp NORD_HOME.
# --------------------------------------------------------------------------------------------------
class _BundledGate:
    """Offline stand-in for WebAccessGate.fetch_wikipedia -- serves _BUNDLED_SUMMARIES with NO network,
    so the self-check is deterministic. Same (title, extract) contract; logs provenance like the real gate."""

    def __init__(self, summaries):
        self.summaries = summaries
        self.log = []

    def fetch_wikipedia(self, topic, simple=False):
        if topic in self.summaries:
            title, extract = self.summaries[topic]
            self.log.append(
                (
                    f"{'simple' if simple else 'en'}.wikipedia.org",
                    len(extract),
                    f"wikipedia:{topic}",
                )
            )
            return (title, extract)
        return (topic, "")  # missing page -> graceful empty


def _selfcheck():
    import shutil
    import tempfile

    from snn_data_lake.substrate.home import KnowledgeLog as _KLog

    work = tempfile.mkdtemp(prefix="snn_enrich_")
    # Save-and-RESTORE, not set-and-pop. The teardown below used to `os.environ.pop("NORD_HOME")`
    # unconditionally, which DELETES a pre-existing value instead of putting it back -- and
    # NORD_HOME is what selects the one durable lake, so a caller that had it set (run_selfchecks
    # sets it for every module; the daemon sets it for the live brain) would silently fall back to
    # the DEFAULT home for everything it did afterwards. Harmless under run_all()'s subprocess
    # isolation, not harmless under run_one_in_process() (the frozen-app path, one process).
    _prev_home = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = work  # isolate: never touch the real home
    try:
        summaries = _BUNDLED_SUMMARIES
        topics = list(summaries.keys())

        print("=" * 92)
        print(
            "AUTONOMOUS ENRICH -- the brain self-improves by learning scholarly concepts into its lake"
        )
        print("=" * 92)
        print(
            f"offline demo: {len(topics)} bundled encyclopedic summaries (no network), temp home {work}"
        )

        gate = _BundledGate(summaries)
        log = _KLog()  # durable log in the temp home
        enr = AutonomousEnricher(gate=gate, log=log, frontier=list(topics), expand=False, seed=0)

        # ---- PROPERTY 4: the lake GROWS as new topics are learned (count climbs 0 -> N) ----
        counts = [log.count()]  # 0 to start
        rows = []
        for _ in range(len(topics)):
            res = enr.run_once()
            counts.append(res["count"])
            rows.append(res)

        # ---- PROPERTY 1: each learned topic is a STORED, retrievable fact (cue overlaps the topic) ----
        records = list(_KLog().replay())  # reload purely from disk
        for topic in topics:
            title = summaries[topic][0]
            recs = [r for r in records if r.get("src") == f"wikipedia:{title}"]
            assert recs, f"topic {topic!r} was not stored as a durable fact"
            rec = recs[0]
            topic_words = _content_words(title)
            assert any(w in rec["cue"] for w in topic_words), (
                f"cue {rec['cue']} does not overlap topic {title!r} ({topic_words}) -- not retrievable"
            )
            assert rec["answer"].startswith(f"{title}:"), (
                f"answer for {title!r} malformed: {rec['answer']!r}"
            )

        # ---- PROPERTY 2: re-running does NOT grow the lake (dedup / lean) ----
        before = _KLog().count()
        enr2 = AutonomousEnricher(
            gate=_BundledGate(summaries), log=_KLog(), frontier=list(topics), expand=False, seed=1
        )
        for _ in range(len(topics) * 2):  # hammer the frontier twice
            enr2.run_once()
        after = _KLog().count()
        assert after == before, f"re-running grew the lake {before} -> {after} (dedup failed)"

        # ---- PROPERTY 3: a fact learned by one enricher is present for a FRESH KnowledgeLog (durable) ----
        fresh = _KLog()  # brand-new reader, no shared RAM
        answers = [r["answer"] for r in fresh.replay()]
        assert any(a.startswith("Calculus:") for a in answers), (
            "durable fact 'Calculus' missing on fresh reload"
        )
        assert any(a.startswith("DNA:") for a in answers), (
            "durable fact 'DNA' missing on fresh reload"
        )

        # ---- the ONE lake is RAW BINARY (BinLog): its file carries the binary magic, not JSON text ----
        from snn_data_lake.substrate.home import knowledge_path

        with open(knowledge_path(), "rb") as f:
            assert f.read(8) == b"SNDLBIN1", "the one lake must be the raw-binary BinLog store"

        # ---- report table ----
        print("-" * 92)
        print(f"{'round':>5} | {'topic':<16} | {'title':<16} | {'stored?':<8} | {'lake count':>10}")
        print("-" * 92)
        for i, r in enumerate(rows, 1):
            print(
                f"{i:>5} | {str(r['topic']):<16} | {str(r['title']):<16} | "
                f"{('yes' if r['written'] else 'dup'):<8} | {r['count']:>10}"
            )
        print("-" * 92)
        print(
            f"lake count grew: {counts}   (0 -> {counts[-1]} as {len(topics)} scholarly topics were learned)"
        )
        print(f"re-run twice   : count stayed {before} -> {after} (dedup keeps the lake lean)")
        print(
            f"fresh reload   : {len(answers)} facts replayed purely from disk (never forgotten across a restart)"
        )
        print(f"provenance log : {gate.log}")
        print("=" * 92)

        # ---- hard asserts ----
        assert counts == list(range(len(topics) + 1)), (
            f"the lake must grow by exactly 1 per new topic; counts={counts}"
        )
        assert before == len(topics), f"expected {len(topics)} durable facts, got {before}"
        assert len(records) == len(topics), "one durable fact per learned topic"

        print(
            f"self-check OK: the brain autonomously learned {len(topics)} scholarly concepts through the "
            f"guarded gate into its durable lake (count 0->{before}), each is a retrievable fact (cue overlaps "
            f"the topic), re-running does NOT grow the lake (dedup/lean), and every fact is REPLAYED by a fresh "
            f"KnowledgeLog (durable, never forgotten). Run --forever (or a Scheduled Task) to learn at all times."
        )
        return 0
    finally:
        if _prev_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = _prev_home
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="autonomous_enrich",
        description="The brain's autonomous self-improvement loop: learn scholarly concepts from "
        "Wikipedia into the durable lake so the chat learns new concepts between sessions.",
    )
    ap.add_argument("--once", action="store_true", help="learn ONE new topic (live network fetch)")
    ap.add_argument("--rounds", type=int, metavar="N", help="learn N topics (live network fetch)")
    ap.add_argument(
        "--forever",
        action="store_true",
        help="learn at all times: loop with a short sleep between rounds (Ctrl-C to stop)",
    )
    ap.add_argument(
        "--simple", action="store_true", help="use simple.wikipedia.org (plainer summaries)"
    )
    ap.add_argument(
        "--sleep", type=float, default=5.0, help="seconds between rounds in --forever (default 5)"
    )
    args = ap.parse_args(argv)

    # Default (no action flag / piped / not a tty) -> deterministic OFFLINE self-check demo.
    if not (args.once or args.rounds or args.forever):
        return _selfcheck()

    # ---- LIVE modes: real gate + real durable home ----
    from snn_data_lake.substrate.home import banner

    enr = AutonomousEnricher()  # WebAccessGate + durable KnowledgeLog
    print(banner("autonomous_enrich"))

    def _do(res):
        if res["topic"] is None:
            print("[enrich] frontier empty -- nothing to learn")
        elif res["learned"]:
            tag = "learned" if res["written"] else "known (dedup)"
            print(f"[enrich] {tag}: {res['title']}  (lake={res['count']} facts)")
        else:
            print(f"[enrich] no summary for {res['topic']!r} (missing/redirect) -- skipped")

    def _guarded_round():
        """One live round, resilient to a transient network error (Wikipedia can 403 / rate-limit) so
        the always-on loop keeps running instead of dying on a single blip. Returns True on success."""
        try:
            _do(enr.run_once(simple=args.simple))
            return True
        except Exception as e:  # transient HTTP/network error
            print(f"[enrich] fetch failed ({type(e).__name__}: {e}) -- will retry next round")
            return False

    if args.once:
        _guarded_round()
    elif args.rounds:
        for _ in range(int(args.rounds)):
            _guarded_round()
    elif args.forever:
        print(f"[enrich] learning at all times -- one topic every {args.sleep}s (Ctrl-C to stop)")
        try:
            while True:
                _guarded_round()
                time.sleep(max(0.0, args.sleep))
        except KeyboardInterrupt:
            print(
                f"\n[enrich] stopped -- lake holds {enr.log.count()} durable facts (nothing forgotten)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
