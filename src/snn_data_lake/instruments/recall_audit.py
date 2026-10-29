"""specs/022 T102 (R1) -- the bounded-recall audit's generator-side oracle and scoring (A4 core).

Pre-registration: specs/022-unified-brain-arc/evidence/r1-bounded-recall.md (2.3, 2.7, A2, A4).
Ported from claude/r1's instruments/recall_audit.py as a SOURCE (integrator ruling, B4 step 1); this
file is the minimal core the A4 text names, rebuilt on main. Brain-free and lake-free: it imports no
retrieval or storage reader (`binlog`, `lake_recall` at module level, `brain`, home's readers), so
the oracle cannot share a bug with the thing it scores. The brain-touching arms, cost driver, child
processes, I2 and I7 are OWED to the campaign build arc (REPORT.md of lane r1-static).

Outcomes (2.14): VERIFIED / FALSE / MISS / REFUSED / ERROR.
"""

from dataclasses import dataclass

ABSENT_CLASSES = frozenset({"C6", "C8"})
SOURCES = frozenset({"working", "lake", "miss", "none", "refused", "error"})
TRAP_FREE = ("C1", "C2", "C3", "C4", "C7")  # classes whose every present cue the oracle can answer


class ScopeMismatch(AssertionError):
    """The generator's recallability label disagrees with home's predicate (2.3)."""


def _base_class(cls):
    return cls.split("_")[0]


def episode_key(episode):
    """The record identity pack_episode gives an episode: its sorted, stringified slots."""
    return tuple(sorted((str(k), str(v)) for k, v in dict(episode).items()))


@dataclass(frozen=True)
class Record:
    """One record the generator writes: an episode (with the generator's OWN recallability label:
    'auto' = autobiographical, 'teach' / 'learner' = not) or a foreign filler blob."""

    episode: dict | None
    label: str | None
    blob: bytes | None = None

    @staticmethod
    def auto(episode):
        return Record(dict(episode), "auto")


@dataclass(frozen=True)
class Cue:
    cls: str
    cue: dict


class Oracle:
    """The generator-side truth table (2.3). Distinct episodes in first-occurrence order -- the
    order a deduping append-only lake keeps them in -- each with the generator's label."""

    def __init__(self, records):
        seen, self.distinct = set(), []
        for r in records:
            if r.episode is None:
                continue
            k = episode_key(r.episode)
            if k not in seen:
                seen.add(k)
                self.distinct.append(
                    Record({str(a): str(b) for a, b in r.episode.items()}, r.label)
                )
        self.written = sum(1 for r in records if r.episode is not None)

    def recallable(self):
        return [r for r in self.distinct if r.label == "auto"]

    def match(self, cue):
        """Indices (into `distinct`) of RECALLABLE episodes matching `cue`, presence-checked."""
        return [
            i
            for i, r in enumerate(self.distinct)
            if r.label == "auto"
            and all(k in r.episode and r.episode[k] == str(v) for k, v in cue.items())
        ]

    def expected_window(self, capacity):
        """What the boot restore should hold: the newest `capacity` recallable lake episodes."""
        return [r.episode for r in self.recallable()][-capacity:]

    def assert_scope(self, predicate):
        """2.3: the generator's label must equal home's predicate on every written episode."""
        bad = [
            r.episode for r in self.distinct if bool(predicate(r.episode)) != (r.label == "auto")
        ]
        if bad:
            raise ScopeMismatch(f"{len(bad)} records disagree with the predicate, e.g. {bad[0]}")

    def score(self, cue, result):
        """Every result shape has an outcome; an unknown source or a malformed episode is ERROR
        (cp4 #6), never VERIFIED and never an exception."""
        if not isinstance(result, dict) or result.get("source") not in SOURCES:
            return "ERROR"
        src = result["source"]
        if src == "error":
            return "ERROR"
        if src == "refused":
            return "REFUSED"
        ep = result.get("episode")
        if ep is None:
            ep = {}
        if not isinstance(ep, dict):
            return "ERROR"
        if src in ("miss", "none") or not ep:
            return "MISS"
        try:
            got = episode_key(ep)
        except Exception:  # noqa: BLE001 - an unreadable episode is a malformed answer
            return "ERROR"
        hits = {episode_key(self.distinct[i].episode) for i in self.match(cue)}
        return "VERIFIED" if got in hits else "FALSE"


@dataclass(frozen=True)
class Row:
    cue: Cue
    outcome: str
    tier: str | None
    seconds: float = 0.0
    episode: tuple | None = None  # the returned episode's canonical identity (2.4 B5)


def identity(result):
    """The canonical identity of the episode a result returned (None when it returned none)."""
    ep = result.get("episode") if isinstance(result, dict) else None
    return episode_key(ep) if isinstance(ep, dict) and ep else None


def score_rows(oracle, cues, responses):
    """One Row per CUE (cp4 #6): a missing or non-dict response is an ERROR row, so a truncated child
    report raises L instead of silently dropping the cues it did not answer."""
    rows = []
    for i, c in enumerate(cues):
        r = responses[i] if i < len(responses) and isinstance(responses[i], dict) else {}
        res = r.get("result")
        rows.append(
            Row(c, oracle.score(c.cue, res), r.get("tier"), r.get("seconds") or 0.0, identity(res))
        )
    return rows


def tally(oracle, rows, faulted=False):
    """Per base class: n, L (present cues lost), F (trusted false), A (absent cues abstained / absent),
    and the tier counts. `faulted` admits REFUSED as the permitted outcome (B4's arms only)."""
    lost = {"MISS", "ERROR"} if faulted else {"MISS", "REFUSED", "ERROR"}
    out = {}
    for r in rows:
        cls = _base_class(r.cue.cls)
        t = out.setdefault(
            cls, {"n": 0, "L": 0, "F": 0, "absent": 0, "abstained": 0, "tiers": {}, "by_tier": {}}
        )
        t["n"] += 1
        present = cls not in ABSENT_CLASSES
        if present and r.outcome in lost:
            t["L"] += 1
        if r.outcome == "FALSE":
            t["F"] += 1
        if not present:
            t["absent"] += 1
            t["abstained"] += r.outcome == "MISS"
        t["tiers"][str(r.tier)] = t["tiers"].get(str(r.tier), 0) + 1
        bt = t["by_tier"].setdefault(str(r.tier), {})  # cp4 #10: outcome x tier, for D-a
        bt[r.outcome] = bt.get(r.outcome, 0) + 1
    for t in out.values():
        t["A"] = (t["abstained"] / t["absent"]) if t["absent"] else None
    return out


def i1_check(oracle_rows):
    """I1: on the trap-free classes the scan-only chain and the generator-side oracle AGREE -- every
    present cue is VERIFIED. (The planted-wrong-FALSE red lives in the tests.)"""
    rows = [r for r in oracle_rows if _base_class(r.cue.cls) in TRAP_FREE]
    return bool(rows) and all(r.outcome == "VERIFIED" for r in rows)
