"""specs/022 T102 (R1) -- the bounded-recall audit's generator-side oracle and scoring (A4 core).

Pre-registration: specs/022-unified-brain-arc/evidence/r1-bounded-recall.md (2.3, 2.7, A2, A4).
Ported from claude/r1's instruments/recall_audit.py as a SOURCE (integrator ruling, B4 step 1); this
file is the minimal core the A4 text names, rebuilt on main. Brain-free and lake-free: it imports no
retrieval or storage reader (`binlog`, `lake_recall` at module level, `brain`, home's readers), so
the oracle cannot share a bug with the thing it scores. I2 (tier attribution) and I7 (the home guard
and live-home stamp) are here, their imports lazy. The brain-touching arms, cost driver and child
processes are OWED to the campaign build arc (REPORT.md of lane r1-static).

Outcomes (2.14): VERIFIED / FALSE / MISS / REFUSED / ERROR.
"""

import contextlib
import hashlib
import os
import random
import time
import tracemalloc
from dataclasses import dataclass, field

# ---- the frozen workload shape (2.2) ------------------------------------------------------------
P_SELF = 0.9  # 2.2: who = self with p .9, else owner / arm uniformly (cp4 #14: 9-of-11 was p .818)
OTHER_WHO = ["owner", "arm"]
ACTIONS = ["rest", "reach", "grasp", "look", "wave", "push", "pull", "speak"]
PLACES = [f"cell{i}" for i in range(10)]
AFFECT = ["calm", "joy", "fear", "surprise", "anger"]
SLOTS = ("who", "what", "where", "affect")
FILLER_TAG = b"\x7e"  # a tag no tenant reads (1.4 (e)'s foreign filler)

#: cue counts per class at W_c (2.2); C9 (faulted index) belongs to the successor arms, C10 is the
#: restart arm (every cue re-asked in a new process), so neither is a generated class here.
DEFAULT_COUNTS = {
    "C1": 100,
    "C2": 100,
    "C3": 200,
    "C4": 200,
    "C5": 200,
    "C6_novel": 100,
    "C6_none": 50,
    "C6_ghost": 25,
    "C6_extra": 25,
    "C7": 50,
    "C8": 40,
}
TINY_COUNTS = {k: max(2, v // 25) for k, v in DEFAULT_COUNTS.items()}
LOAD_TOLERANCE = 0.5  # B7: a witness drift beyond 50 % VOIDS the seed's cost reading
N_NONE_TRAP_EPISODES = 20
N_CROSS_BOOT_DUPS = 50
COLLISION_SCAN_LIMIT = 200000  # lake_recall._find_cell_collision's default `limit`
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
        if capacity <= 0:
            return []  # [-0:] would be the whole list
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


# ---- the workload and its generator (2.2) ---------------------------------------------------------
@dataclass
class Workload:
    seed: int
    capacity: int
    records: list
    cues: list
    learner_kinds: tuple = ()
    meta: dict = field(default_factory=dict)


def _episode(rng, when, session=None):
    """One autobiographical episode. `session` set = the F3 session-bearing schema (engine/brain.py
    stamps it on durable episodes); None = the legacy shape with no session slot."""
    ep = {
        "who": "self" if rng.random() < P_SELF else rng.choice(OTHER_WHO),
        "what": rng.choice(ACTIONS),
        "where": rng.choice(PLACES),
        "when": str(when),
        "affect": rng.choice(AFFECT),
    }
    if session is not None:
        ep["session"] = str(session)
    return ep


def _collision_pairs(n):
    """`n` pairs of distinct `when` values whose tokens share a LakeRecall code cell (the
    collision-ghost construction of lake_recall._find_cell_collision, extended to n pairs). The one
    import of a retrieval module, lazy and for the cell function only: the ORACLE never reads it."""
    from snn_data_lake.substrate.lake_recall import _token_cell

    seen, pairs, i = {}, [], 0
    while len(pairs) < n:
        if i >= COLLISION_SCAN_LIMIT:  # the scan budget lake_recall._find_cell_collision has
            raise AssertionError("no collision within the scan budget -- the code space grew?")
        v = f"t{i}"
        cell = _token_cell("when", v)
        if cell in seen:
            pairs.append((seen.pop(cell), v))
        else:
            seen[cell] = v
        i += 1
    return pairs


def make_workload(seed, *, n_auto, boots, n_filler, filler_bytes, capacity, counts, teach, learner):
    """The frozen generator (2.2: `make_workload(seed, watermark)` is this with the W_c shape args --
    n_auto / boots / n_filler / filler_bytes -- spelled keyword-only, and `capacity` the watermark).
    Pure: returns the record stream and the cue list. Deterministic in `seed` (I6). `teach` /
    `learner` are the A2 fixture's episode dicts (tests/r1_c8_fixture.py): the legacy teach rows and
    the current-schema learner-kind distractors, written but not recallable. A shape that cannot
    hold the registered classes raises ValueError naming the argument."""
    if boots < 2:
        raise ValueError(f"boots={boots}: cross-boot duplicates need at least 2 boots")
    if n_auto % boots:
        raise ValueError(f"n_auto={n_auto} is not a multiple of boots={boots}: records would drop")
    if n_auto // boots < N_CROSS_BOOT_DUPS:
        raise ValueError(f"per_boot={n_auto // boots} is below the {N_CROSS_BOOT_DUPS} duplicates")
    if not 1 <= capacity < n_auto - N_CROSS_BOOT_DUPS:
        raise ValueError(f"capacity={capacity} leaves no resident or no non-resident episode")
    rng = random.Random(seed)
    per_boot = n_auto // boots
    stream = []
    boot0 = []
    for b in range(boots):
        for i in range(per_boot):
            # `when` restarts at each simulated boot (1.4 (d)); boot 0 is the legacy schema (no session
            # slot), later boots carry one, so W_c mixes both schemas (2.2, A2)
            ep = _episode(rng, 5 * (i + 1), session=b if b else None)
            if b == 0:
                boot0.append(ep)
            stream.append(Record.auto(ep))
    # cross-boot byte-identical re-formations (the 1.4 (d) collapse), spread over boots 1..B-1
    dup_idx = rng.sample(range(per_boot), N_CROSS_BOOT_DUPS)
    dups = []
    for j, i in enumerate(dup_idx):
        b = 1 + j % max(1, boots - 1)
        stream[b * per_boot + i] = Record.auto(boot0[i])
        dups.append(boot0[i])
    # planted classes, inserted at seeded positions
    lacking = [
        {"who": "self", "what": rng.choice(ACTIONS), "where": rng.choice(PLACES), "when": f"u{j}"}
        for j in range(N_NONE_TRAP_EPISODES)
    ]
    pairs = _collision_pairs(2)
    collided = [_episode(rng, a) for a, _b in pairs]
    planted = (
        [Record.auto(e) for e in lacking + collided]
        + [Record(dict(e), "teach") for e in teach]
        + [Record(dict(e), "learner") for e in learner]
    )
    for r in planted:
        stream.insert(rng.randrange(len(stream) + 1), r)
    filler = [
        Record(None, None, FILLER_TAG + rng.randbytes(filler_bytes - 1)) for _ in range(n_filler)
    ]
    for r in filler:
        stream.insert(rng.randrange(len(stream) + 1), r)
    oracle = Oracle(stream)
    cues = _make_cues(
        rng, oracle, capacity, counts, lacking, pairs, collided, dups, list(teach) + list(learner)
    )
    return Workload(
        seed,
        capacity,
        stream,
        cues,
        tuple(sorted({e["kind"] for e in learner})),
        meta={"pairs": pairs, "per_boot": per_boot, "boots": boots},
    )


def _subcue(rng, ep, with_when, k):
    keys = rng.sample([s for s in SLOTS if s in ep], k)
    cue = {s: ep[s] for s in keys}
    if with_when:
        cue["when"] = ep["when"]
    return cue


def _draw(make, accept, what, tries=1000):
    """Resample `make()` until `accept` holds; refuse visibly (never spin) when the shape cannot."""
    for _ in range(tries):
        c = make()
        if accept(c):
            return c
    raise ValueError(
        f"no {what} cue satisfies its class in {tries} draws: the workload shape is too small"
    )


def _make_cues(rng, oracle, capacity, counts, lacking, pairs, collided, dups, non_auto):
    rec = [r.episode for r in oracle.recallable()]
    resident, older = rec[-capacity:], rec[:-capacity]
    win = {episode_key(e) for e in resident}
    cues = []
    add = lambda cls, cue: cues.append(Cue(cls, dict(cue)))  # noqa: E731

    def hits(c):
        return {episode_key(oracle.distinct[i].episode) for i in oracle.match(c)}

    # `when` restarts every boot, so a partial cue can also match a same-index episode of another
    # tier: C1/C2 need a match INSIDE the boot-restore window, C3/C4 matches ALL OUTSIDE it (#6673)
    inside = lambda c: bool(hits(c) & win)  # noqa: E731
    outside = lambda c: bool(hits(c)) and not hits(c) & win  # noqa: E731
    sub2 = lambda pool: _subcue(rng, rng.choice(pool), True, rng.randint(1, 2))  # noqa: E731

    for _ in range(counts["C1"]):
        add("C1", rng.choice(resident))
    for _ in range(counts["C2"]):
        add("C2", _draw(lambda: sub2(resident), inside, "C2"))
    for _ in range(counts["C3"]):
        add("C3", _draw(lambda: rng.choice(older), outside, "C3"))
    for _ in range(counts["C4"]):
        add("C4", _draw(lambda: sub2(older), outside, "C4"))
    for _ in range(counts["C5"]):
        draw = lambda: _subcue(rng, rng.choice(rec), False, rng.randint(2, 3))  # noqa: E731
        add("C5", _draw(draw, lambda c: len(oracle.match(c)) >= 2, "C5"))
    for i in range(counts["C6_novel"]):
        add("C6_novel", {"when": rng.choice(rec)["when"], "what": f"novel{i}"})
    for i in range(counts["C6_none"]):
        add("C6_none", {"when": lacking[i % len(lacking)]["when"], "affect": "None"})
    for i in range(counts["C6_ghost"]):
        (_a, ghost), ep = pairs[i % len(pairs)], collided[i % len(collided)]
        add("C6_ghost", dict(ep, when=ghost))
    for _ in range(counts["C6_extra"]):
        add("C6_extra", dict(rng.choice(rec), mood="x"))
    for i in range(counts["C7"]):
        add("C7", dups[i % len(dups)])
    for i in range(counts["C8"]):
        add("C8", non_auto[i % len(non_auto)])
    for c in cues:  # the generator proves its own classes before anything is run
        n = len(oracle.match(c.cue))
        if (_base_class(c.cls) in ABSENT_CLASSES) != (n == 0):
            raise AssertionError(f"cue class {c.cls} has {n} matches: {c.cue}")
    return cues


def workload_digest(wl):
    """I6: a hash over the record stream and the cue list; equal iff the workload is equal."""
    h = hashlib.sha256()
    for r in wl.records:
        h.update(repr((sorted(r.episode.items()) if r.episode else None, r.label, r.blob)).encode())
    for c in wl.cues:
        h.update(repr((c.cls, sorted(c.cue.items()))).encode())
    return h.hexdigest()


# ---- instrument probes I3 / I4 / I5 (2.7) ---------------------------------------------------------
def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def latency_probe_ok(planted, measured):
    """I3's floor: a planted delay must be visible in the measurement (a span timed outside the
    call reads ~0 and fails here)."""
    return measured >= planted


def traced_peak(fn):
    """I4: the peak traced allocation of `fn()`, in bytes."""
    outer = (
        tracemalloc.is_tracing()
    )  # a caller's trace (the cost driver nests this) is not ours to end
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not outer:
            tracemalloc.stop()
    return out, peak


def spin_witness(n=2_000_000):
    """A fixed CPU spin, timed: the load witness read before and after each cost arm."""
    t0 = time.perf_counter()
    x = 0
    for i in range(n):
        x += i & 7
    return time.perf_counter() - t0


def load_verdict(before, after, tol=LOAD_TOLERANCE):
    return "VOID" if abs(after - before) > tol * before else "OK"


# ---- I7: the throwaway-home guard (2.7 I7, 2.14 D-e) ----------------------------------------------
# `home` is imported lazily below: the oracle half of this module must not share a reader with the
# thing it scores, and the guard is the only home function the instrument itself needs.
class GuardMissing(RuntimeError):
    """The shared live-home guard (lead ruling 16) is not in `home`: refuse to run brains."""


class LiveHomeWritten(RuntimeError):
    """The live home's stamp (mtime, size, content digest per file) changed during a run (I7, cp4 #8)."""


def require_isolation():
    """Call the ONE shared guard (`home.require_isolated_home`); never a second guard here (P2).
    Fails CLOSED when it is absent, and lets its refusal through unchanged."""
    from snn_data_lake.substrate import home as H

    guard = getattr(H, "require_isolated_home", None)
    if guard is None:
        raise GuardMissing("home.require_isolated_home is absent: the shared guard must land first")
    return guard()


def confine(root):
    """The output root passes the shared guard AND sits inside the active throwaway home, so nothing
    is created outside it (NORD_HOME alone says nothing about where `root` writes)."""
    from snn_data_lake.substrate import home as H

    home = require_isolation()
    resolved = H.assert_not_live_home(root)
    a, b = os.path.normcase(resolved), os.path.normcase(home)
    try:
        inside = os.path.commonpath([a, b]) == b
    except ValueError:  # different drives
        inside = False
    if not inside:
        raise H.LiveHomeRefused(
            f"output root refused (outside the active home): {resolved} not in {home}"
        )
    return resolved


# distinct-from: gates/move_gate.py::_live_home_stamp -- the gate's PRIVATE census helper (top-level files
#   of the default home, size and mtime); the repo's ONE live-lake stamp is substrate/home.py::lake_snapshot
#   / live_lake_untouched (nested files, content digest), which this wrapper calls and does not copy (P2).
def stamped(fn):
    """Run `fn()` under I7's live-home stamp (home.lake_snapshot), compared before and after with
    home.live_lake_untouched, on the exception path too."""
    from snn_data_lake.substrate import home as H

    live = H._default_live_home()
    before = H.lake_snapshot(live)
    try:
        out = fn()
    except BaseException as exc:
        ok, changed = H.live_lake_untouched(live, before)
        if not ok:
            raise LiveHomeWritten(
                f"the live home changed during a run that then raised: {changed[:3]}"
            ) from exc
        raise
    ok, changed = H.live_lake_untouched(live, before)
    if not ok:
        raise LiveHomeWritten(f"the live home changed during the run: {changed[:3]}")
    return out


# ---- I2: tier attribution (2.7 I2; wraps only, no brain.py edit) -----------------------------------
def tier_of(result, deep_calls):
    """I2: which tier answered. A 'lake' answer that read the whole lake is the deep tier."""
    if not isinstance(result, dict) or not result.get("episode"):
        return None
    if result.get("source") == "working":
        return "working"
    if result.get("source") == "lake":
        return "deep" if deep_calls else "window"
    return None


# distinct-from: substrate/lake_recall.py::LakeRecall and home.load_episodes -- the WRAPPED subjects, not
#   rivals; instruments/memory_diagnostic.py::_blas, substrate/episodic_recall.py -- prose keyword hits.
@contextlib.contextmanager
def _instrumented(brain, deep_attr="load_episodes"):
    """Install counting wrappers AFTER construction (the boot restore reads `load_episodes` too and
    must not be counted): `home.<deep_attr>` counts deep-tier reads, `LakeRecall.recall` counts window
    reads. Both are restored on exit. `brain` is kept for the arms that neuter its tiers (owed)."""
    from snn_data_lake.substrate import home as H
    from snn_data_lake.substrate import lake_recall as LR

    calls = {"deep": 0, "window": 0}
    orig_load, orig_recall = getattr(H, deep_attr), LR.LakeRecall.recall

    def load(*a, **k):
        calls["deep"] += 1
        return orig_load(*a, **k)

    def recall(self, cue, k=3):
        calls["window"] += 1
        return orig_recall(self, cue, k=k)

    setattr(H, deep_attr, load)
    LR.LakeRecall.recall = recall
    try:
        yield calls
    finally:
        setattr(H, deep_attr, orig_load)
        LR.LakeRecall.recall = orig_recall


def i2_check(home_dir, deep_attr="load_episodes"):
    """I2 through the REAL wrappers and the real chain (`Brain.recall_episode_durable` on a stub that
    carries only the attributes it reads): a resident exact cue attributes `working`, a window-partial
    one `window`, a nonresident exact cue `deep`. Pointing the wrapper at the wrong attribute
    (`deep_attr`) makes the deep cue read `window` -- the red."""
    from types import SimpleNamespace

    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.substrate import home as H

    eps = [
        {"who": "self", "what": a, "where": f"cell{i}", "when": str(5 * (i + 1)), "affect": "calm"}
        for i, a in enumerate(ACTIONS[:4])
    ]
    # the guard resolves STRICTLY (a path must exist), so the PARENT is confined first (inside the active
    # throwaway home, not the live one); only then is the home created, and confined itself.
    confine(os.path.dirname(os.path.abspath(home_dir)))
    os.makedirs(home_dir, exist_ok=True)
    resolved = confine(home_dir)
    prev = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = resolved
    try:
        H.save_episode(eps[3])  # on the lake only
        got = {}
        for want, window, held in (
            ("working", [], eps[0]),
            ("window", eps[:3], None),
            ("deep", [], None),
        ):
            cons = SimpleNamespace(
                recall_episode=lambda cue, min_conf=0.5, h=held: (
                    {"episode": h, "confidence": 1.0} if h else {}
                )
            )
            stub = SimpleNamespace(
                _episode_consolidator=cons, _episode_window=list(window), _lake_recall=None, _seed=0
            )
            cue = dict(eps[0] if want != "deep" else eps[3])
            with _instrumented(stub, deep_attr) as calls:
                res = Brain.recall_episode_durable(stub, cue, min_conf=0.5)
            got[want] = tier_of(res, calls["deep"])
    finally:
        if prev is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prev
    return got == {"working": "working", "window": "window", "deep": "deep"}
