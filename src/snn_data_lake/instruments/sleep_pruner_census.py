r"""G6 / T114 part (a): the live sleep pruners -- a census that is generated, never hand-kept, and the
per-pruner neuter-A/B that decides whether each one earns its place.

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md, frozen at P0 (df19f61e),
sections 2.3 (part (a)), 2.8 (null dispositions), 2.11 (checks C1-C6), and 2.16's governing amendments A1
(the keep rule), A2 (one fixed margin per pruner), A9 (degenerate contrasts), A11 (E, executed work).

TWO HALVES
  1. CENSUS. `census_static` walks the Brain's two sleep hosts (`_sleep_tick`, every asleep tick;
     `_enter_sleep`, once per night) through attribute calls, transitively, and collects every callee
     whose name is in the prune vocabulary. Receivers are typed by three static rules (`self.X =
     Cls(...)` anywhere in the class, function-local imports included, or `self.X = self.f(...)` where
     f returns a constructed `Cls`; a local `v = Cls(...)`; a direct `Cls(...).m()`), plus, when a
     scratch brain is handed in, the RUNTIME type of each `self.a.b` chain the hosts use. A receiver
     neither rule types is COUNTED as unresolved, never chased. `census_runtime` wraps every found
     site on a scratch brain and counts calls and removals per tick.
  2. A/B. `run_arm` drives one throwaway brain per (seed, arm) for 900 ticks, the four
     tests/test_sleep_capabilities.py TEACH pairs injected at every wake onset, under one surgical
     neuter / sabotage / control; `decide_pruner` reads 2.3's R0-R4 and A11's E in order.

Learning is never touched by a read here: every outcome is read off counters the live brain already
computes (`_consolidation_acc`, the grower's held-out refit score, the lifelong learner's test
accuracy). Every Brain this module builds is under a throwaway NORD_HOME that is checked not to
resolve to the live home; it never constructs, and is never constructed by, a live brain.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.sleep_pruner_census        (self-check)
  ... --census [--ticks N] [--seed S]       runtime census on one scratch brain
  ... --arm SEED ARM [--schedule F] [--out F] [--ticks N]   one A/B arm, JSON out
"""

import ast
import contextlib
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/snn_data_lake

HOSTS = ("_sleep_tick", "_enter_sleep")
BRAIN = ("engine.brain", "Brain")

# 2.3's prune vocabulary {prune*, *_prune*, downscale*, rewire, evict*, consolidate}.
PRUNE_VOCAB = re.compile(r"^(_?prune\w*|\w+_prune\w*|downscale\w*|rewire|evict\w*|consolidate)$")

# The census CLASSIFIES what it finds; it never supplies a site. (class, method) -> pruner id.
PRUNERS = {
    ("StructuralRewiring", "rewire"): "P-SR",
    ("AttractorNet", "downscale_prune"): "P-DP",
    ("NGSNN", "prune_unearned"): "P-EY",
    ("LifelongConsolidator", "_prune_to_budget"): "P-LL",
    ("LakeConsolidation", "consolidate"): "P-LAKE",
    ("KnowledgeLog", "consolidate"): "P-LAKE",
}
# Vocabulary hits that are not prune operations, each with the reason (section 1 item 1's kept-out list).
NOT_PRUNERS = {
    (
        "RecallTrace",
        "prune_to",
    ): "P-LAKE's own bookkeeping: drops trace entries of records P-LAKE already "
    "removed; removes nothing of its own",
    (
        "SchemaInducer",
        "consolidate",
    ): "schema induction DERIVES prototype triples from the durable ones; its "
    "docstring: 'Does NOT mutate any KnowledgeLog' -- an insert, never a removal",
    (
        "LakeMemory",
        "consolidate",
    ): "substrate/proximity_replay.LakeMemory.consolidate persists a batch of exemplars and marks the "
    "proximity index stale -- an insert, never a removal (reached from SkillRegistry.register)",
    (
        "_BennaFusiConsolidation",
        "consolidate",
    ): "the plasticity stack's per-step Benna-Fusi deposit + one diffusion step (T058) -- moves weight "
    "into deep beakers, removes no synapse or unit",
}
# Vocabulary hits whose receiver no rule types, classified by (module, method) after reading the site.
NOT_PRUNERS_AT = {
    (
        "faculties.skill_builder",
        "consolidate",
    ): "SkillRegistry.register -> `self.lake.consolidate(sig_exemplars, payloads=...)`, the lake a "
    "substrate/proximity_replay.LakeMemory (sleep/nightly_skills.py:536) -- an insert, never a removal",
}
EXPECTED = ("P-SR", "P-DP", "P-EY", "P-LL", "P-LAKE")  # section 1 item 1's table
AB_SCOPE = (
    "P-DP",
    "P-EY",
    "P-LL",
)  # 2.3 + A1: G6 reads three; P-SR is R3's; P-LAKE is out of (a)'s scope

TICKS = 900  # 2.3: the 2026-08-05 length
# A2 + the dated pre-data amendment 2026-10-04 (ruling 24): night k = the k-th `_enter_wake` reading, from 1
NIGHTS = range(
    2, 37
)  # the P-DP window k = 2..36; a run needs >= 36 readings and stored patterns at k = 2
DP_TRIALS = 36  # 3 patterns x TEST_TRIALS = 12 per `_consolidation_acc` reading
DELTA = {
    "P-DP": 1.0 / (DP_TRIALS * len(NIGHTS)),  # 1/1260
    "P-LL": 1.0 / 120.0,  # 3 tasks x 40 test items
    "P-EY": 1.0
    / 1111.0,  # 1/T_ref; T_ref = 1111, the largest dev pooled total (12006), amendment 2026-10-04
}
FLOOR_DP = 1.0 / 3.0  # the harness-alive control's pin (C4)

ARMS = (
    "live",
    "dp_neuter",
    "dp_sabotage",
    "dp_floor",
    "ey_neuter",
    "ey_sabotage",
    "ll_neuter",
    "ll_random",
    "ll_sabotage",
)
TEACH = (  # tests/test_sleep_capabilities.py's four pairs, verbatim
    ("what is your goal today", "GOAL"),
    ("do you remember the bench", "MEMORY"),
    ("hello there brain", "GREET"),
    ("are you aware of yourself", "CONSCIOUS"),
)


# ================================================================================================
# 1. CENSUS -- static half
# ================================================================================================
@dataclass(frozen=True)
class Site:
    pruner: str  # registered id, "not-a-pruner", or "UNCLASSIFIED"
    host: str
    chain: tuple
    cls: str
    method: str
    module: str
    line: int


@dataclass
class Census:
    sites: list
    unresolved: int  # attribute calls whose receiver no rule typed (counted, never chased)
    unresolved_vocab: list  # the subset whose METHOD is in the prune vocabulary (each one listed)

    def found(self):
        return sorted({s.pruner for s in self.sites if s.pruner in EXPECTED})

    def missing(self):
        return [p for p in EXPECTED if p not in self.found()]

    def unclassified(self):
        return [s for s in self.sites if s.pruner == "UNCLASSIFIED"] + list(self.unresolved_vocab)

    def table(self):
        rows = {}
        for s in self.sites:
            rows.setdefault(s.pruner, []).append(
                f"{s.cls}.{s.method} ({s.module}:{s.line}) via {s.host}"
            )
        return {k: sorted(set(v)) for k, v in sorted(rows.items())}


class _Index:
    """Lazily parsed package modules; `sources` overrides a module's text (the C1 reds plant into it)."""

    def __init__(self, sources=None):
        self._src = dict(sources or {})
        self._trees = {}

    def tree(self, mod):
        if mod not in self._trees:
            text = self._src.get(mod)
            if text is None:
                path = os.path.join(_PKG_DIR, *mod.split(".")) + ".py"
                if not os.path.isfile(path):
                    self._trees[mod] = None
                    return None
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            self._trees[mod] = ast.parse(text)
        return self._trees[mod]

    @staticmethod
    def _imports(nodes):
        out = {}
        for n in nodes:
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("snn_data_lake."):
                for a in n.names:
                    out[a.asname or a.name] = (n.module[len("snn_data_lake.") :], a.name)
        return out

    def module_imports(self, mod):
        t = self.tree(mod)
        return self._imports(t.body) if t else {}

    def local_imports(self, fn):
        return self._imports([n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)])

    def classdef(self, mod, cls):
        t = self.tree(mod)
        if t is None:
            return None
        for n in t.body:
            if isinstance(n, ast.ClassDef) and n.name == cls:
                return n
        return None

    def method(self, mod, cls, name):
        cd = self.classdef(mod, cls)
        if cd is None:
            return None
        for n in cd.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return n
        return None

    def resolve_class(self, mod, fn, name):
        """A bare Name -> (module, class) when it names a package class (same module or imported)."""
        if self.classdef(mod, name) is not None:
            return (mod, name)
        imp = dict(self.module_imports(mod))
        if fn is not None:
            imp.update(self.local_imports(fn))
        hit = imp.get(name)
        if hit and self.classdef(hit[0], hit[1]) is not None:
            return hit
        return None

    def _ctor(self, mod, fn, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self.resolve_class(mod, fn, node.func.id)
        return None

    def returned_class(self, mod, cls, fname):
        """`self.f(...)` -> the class f constructs and returns (`return Cls(...)` or `v = Cls(...); return v`)."""
        fn = self.method(mod, cls, fname)
        if fn is None:
            return None
        local = self.local_types(mod, fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and n.value is not None:
                got = self._ctor(mod, fn, n.value)
                if got is None and isinstance(n.value, ast.Name):
                    got = local.get(n.value.id)
                if got:
                    return got
        return None

    def local_types(self, mod, fn):
        out = {}
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                got = self._ctor(mod, fn, n.value)
                if got:
                    out[n.targets[0].id] = got
        return out

    def attr_types(self, mod, cls):
        """self.X -> class, from every `self.X = Cls(...)` / `self.X = self.f(...)` in the class."""
        cd = self.classdef(mod, cls)
        out = {}
        if cd is None:
            return out
        for fn in cd.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if not (isinstance(n, ast.Assign) and len(n.targets) == 1):
                    continue
                t = n.targets[0]
                if not (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    continue
                got = self._ctor(mod, fn, n.value)
                v = n.value
                if (
                    got is None
                    and isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and isinstance(v.func.value, ast.Name)
                    and v.func.value.id == "self"
                ):
                    got = self.returned_class(mod, cls, v.func.attr)
                if got and t.attr not in out:
                    out[t.attr] = got
        return out


def _self_chain(node):
    """`self.a.b` -> "a.b"; anything not rooted at `self` -> None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "self" and parts:
        return ".".join(reversed(parts))
    return None


def _classify(cls, method):
    if (cls, method) in PRUNERS:
        return PRUNERS[(cls, method)]
    if (cls, method) in NOT_PRUNERS:
        return "not-a-pruner"
    return "UNCLASSIFIED"


def census_static(runtime_types=None, sources=None, max_depth=6):
    """The census from the two hosts. `runtime_types` maps a host's `self.a.b` chain -> (module, class)."""
    idx = _Index(sources)
    runtime_types = dict(runtime_types or {})
    sites, unresolved_vocab = [], []
    unresolved = [0]
    seen = set()

    def receiver(mod, cls, fn, local, recv, at_brain):
        if isinstance(recv, ast.Name):
            if recv.id == "self":
                return (mod, cls)
            return local.get(recv.id)
        if isinstance(recv, ast.Call):
            return idx._ctor(mod, fn, recv)
        chain = _self_chain(recv)
        if chain is None:
            return None
        if at_brain and chain in runtime_types:
            return runtime_types[chain]
        if "." not in chain:
            return idx.attr_types(mod, cls).get(chain)
        return None

    def walk(mod, cls, fname, host, chain, depth):
        key = (mod, cls, fname)
        if key in seen or depth > max_depth:
            return
        seen.add(key)
        fn = idx.method(mod, cls, fname)
        if fn is None:
            return
        local = idx.local_types(mod, fn)
        at_brain = (mod, cls) == BRAIN
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            m = node.func.attr
            tgt = receiver(mod, cls, fn, local, node.func.value, at_brain)
            if tgt is None:
                unresolved[0] += 1
                if PRUNE_VOCAB.match(m):
                    site = Site(
                        "UNCLASSIFIED", host, tuple(chain) + (f"?.{m}",), "?", m, mod, node.lineno
                    )
                    if (mod, m) in NOT_PRUNERS_AT:
                        sites.append(site.__class__(**{**asdict(site), "pruner": "not-a-pruner"}))
                    else:
                        unresolved_vocab.append(site)
                continue
            step = f"{tgt[1]}.{m}"
            if PRUNE_VOCAB.match(m):
                sites.append(
                    Site(
                        _classify(tgt[1], m),
                        host,
                        tuple(chain) + (step,),
                        tgt[1],
                        m,
                        mod,
                        node.lineno,
                    )
                )
            walk(tgt[0], tgt[1], m, host, list(chain) + [step], depth + 1)

    for host in HOSTS:
        walk(BRAIN[0], BRAIN[1], host, host, [f"Brain.{host}"], 0)
    return Census(sites=sites, unresolved=unresolved[0], unresolved_vocab=unresolved_vocab)


def runtime_types_of(brain):
    """The runtime class of every `self.a.b` receiver chain the two hosts use, read off a scratch brain."""
    idx = _Index()
    chains = set()
    for host in HOSTS:
        fn = idx.method(BRAIN[0], BRAIN[1], host)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                c = _self_chain(n.func.value)
                if c:
                    chains.add(c)
    out = {}
    for c in sorted(chains):
        obj = brain
        try:
            for part in c.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        t = type(obj)
        if t.__module__.startswith("snn_data_lake."):
            out[c] = (t.__module__[len("snn_data_lake.") :], t.__qualname__)
    return out


# ================================================================================================
# 1b. CENSUS -- runtime half (and the removal extractors C2 checks)
# ================================================================================================
def _removed_dict(ret, _before, _after):
    return int(ret["pruned"]) if isinstance(ret, dict) else 0


def _removed_int(ret, _before, _after):
    return int(ret or 0)


def _removed_units(_ret, before, after):
    return max(0, int(before) - int(after))


def _removed_lake(ret, before, _after):
    if isinstance(ret, tuple) and len(ret) == 2:  # KnowledgeLog.consolidate -> (kept, pruned)
        return int(ret[1])
    return max(0, int(before) - len(ret)) if ret is not None else 0


# (module, class, method) -> (pruner id, removal extractor, size-before reader)
SITES = {
    ("sleep.sleep_replay", "AttractorNet", "downscale_prune"): ("P-DP", _removed_dict, None),
    ("sleep.structural_rewiring", "StructuralRewiring", "rewire"): ("P-SR", _removed_dict, None),
    ("growth.ng_snn", "NGSNN", "prune_unearned"): ("P-EY", _removed_int, None),
    ("sleep.lifelong_consolidation", "LifelongConsolidator", "_prune_to_budget"): (
        "P-LL",
        _removed_units,
        lambda self, a, k: self.n_units,
    ),
    ("sleep.lake_consolidation", "LakeConsolidation", "consolidate"): (
        "P-LAKE",
        _removed_lake,
        lambda self, a, k: len(a[0]) if a else len(k.get("records", [])),
    ),
    ("substrate.home", "KnowledgeLog", "consolidate"): ("P-LAKE", _removed_lake, None),
}


class Recorder:
    """Per-pruner (tick, removed) rows; `tick` is advanced by the driver."""

    def __init__(self):
        self.tick = 0
        self.rows = {}

    def add(self, pid, n):
        self.rows.setdefault(pid, []).append((self.tick, int(n)))

    def calls(self, pid):
        return len(self.rows.get(pid, []))

    def removals(self, pid):
        return sum(n for _, n in self.rows.get(pid, []))


def _import(mod):
    import importlib

    return importlib.import_module("snn_data_lake." + mod)


@contextlib.contextmanager
def _patched(patches):
    """[(owner, name, new)] applied, then restored exactly (class or instance attributes)."""
    saved = []
    try:
        for owner, name, new in patches:
            had = name in vars(owner)
            saved.append((owner, name, had, vars(owner).get(name)))
            setattr(owner, name, new)
        yield
    finally:
        for owner, name, had, old in reversed(saved):
            if had:
                setattr(owner, name, old)
            else:
                delattr(owner, name)


def counting_patches(rec, sites=SITES):
    """Class-level wrappers that COUNT each site's removals; behaviour unchanged."""
    out = []
    for (mod, cls, meth), (pid, extract, size) in sites.items():
        owner = getattr(_import(mod), cls)
        orig = getattr(owner, meth)

        def wrap(self, *a, _orig=orig, _pid=pid, _ex=extract, _size=size, **k):
            before = _size(self, a, k) if _size else None
            ret = _orig(self, *a, **k)
            after = _size(self, (), {}) if (_size and _pid == "P-LL") else None
            rec.add(_pid, _ex(ret, before, after))
            return ret

        out.append((owner, meth, wrap))
    return out


# distinct-from: engine/brain.py::_force -- it FORCES the clock phase before each tick for its self-check;
#   this drives the natural clock and only counts the registered sites. distinct-from:
#   gates/perf_baseline.py::_measure_ticks -- it TIMES ticks and classifies phase; nothing here is timed.
#   distinct-from: gates/gradient_claim_gate.py, instruments/everyday_battery.py, engine/autonomous_loop.py --
#   a def enumerator, a scrambled-key control and a perception classifier; phrase overlap only. Sites read.
def census_runtime(brain, ticks, rec=None, teach=True):
    """Count every registered site's removals while `brain` runs `ticks` ticks; returns the Recorder."""
    rec = rec or Recorder()
    with _patched(counting_patches(rec)):
        _drive(brain, ticks, rec, teach)
    return rec


# distinct-from: faculties/access_consciousness.py -- `_first_denied` searches a peer load; no call census.
# distinct-from: gates/move_gate.py -- `legacy_pkgs` lists legacy package keys; no call census.
# distinct-from: gates/gradient_claim_gate.py -- a census of learning RULES owing a gradient cosine.
@contextlib.contextmanager
def discovering(seen):
    """The static walk's independent net: a profile hook on this thread records every package function in
    the prune vocabulary that runs while a sleep host is on the stack -- including classes imported lazily
    DURING the run, which a wrap-what-is-imported net misses (measured: the first version did)."""
    depth = [0]
    host_mod = "snn_data_lake." + BRAIN[0]

    def prof(frame, event, _arg):
        code = frame.f_code
        mod = frame.f_globals.get("__name__", "")
        if code.co_name in HOSTS and mod == host_mod:
            if event == "call":
                depth[0] += 1
            elif event == "return":
                depth[0] -= 1
            return
        if event == "call" and depth[0] > 0 and mod.startswith("snn_data_lake."):
            if PRUNE_VOCAB.match(code.co_name):
                qual = getattr(code, "co_qualname", code.co_name)
                seen.add((qual.rsplit(".", 1)[0] if "." in qual else "?", code.co_name))

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield seen
    finally:
        sys.setprofile(old)


def census_misses(static, seen):
    """Methods sleep called (runtime discovery) that the static census neither reached nor classified as
    not-a-pruner; a registered pruner the static walk missed is a miss too."""
    known = {(s.cls, s.method) for s in static.sites} | set(NOT_PRUNERS)
    return sorted(k for k in seen if k not in known)


# ================================================================================================
# 2. A/B -- arms, outcomes, executed-work counters
# ================================================================================================
def _assert_scratch_home(live_home=None):
    """Refuse to build a Brain unless NORD_HOME is set and is not, contains, or sits inside the live home
    (the one shared guard, lead ruling 16; `live_home` only for a planted decoy)."""
    from snn_data_lake.substrate.home import require_isolated_home

    return require_isolated_home(live_home=live_home)


def _teach(brain):
    from snn_data_lake.readout.interaction_learning import text_features

    ixn = brain.brain._ixn
    for text, intent in TEACH:
        ixn.interact(text_features(text, ixn.n_in), intent)


def _drive(brain, ticks, rec, teach=True):
    """The frozen workload: TEACH once before the first tick and at every wake onset."""
    if teach:
        _teach(brain)
    prev = "asleep"
    for t in range(ticks):
        rec.tick = t
        mode = brain.tick()
        if teach and mode == "awake" and prev == "asleep":
            _teach(brain)
        prev = mode


@dataclass
class ArmRun:
    seed: int
    arm: str
    ticks: int
    nights: list = field(default_factory=list)  # [k (1-based reading), stored, correct of 36]
    refits: list = field(default_factory=list)  # [grower tick, n_heldout, correct]
    expected_refits: int = 0
    ll: list = field(
        default_factory=list
    )  # [task, correct, total] at the run's end, learned tasks only
    ll_learned: int = 0
    removals: dict = field(default_factory=dict)  # pid -> [[tick, n], ...]
    work: dict = field(default_factory=dict)  # pid -> executed operations (A11)
    sabotage_shortfall: int = (
        0  # a sabotage call that could not remove its scheduled count (disclosed)
    )
    digest: str = ""

    def finish(self):
        body = {k: v for k, v in asdict(self).items() if k != "digest"}
        self.digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return self


# ---- the executed-work counters (A11): each the loop bound of the component's own forward step ----
def _work_patches(run):
    from snn_data_lake.growth.ng_snn import NGSNN
    from snn_data_lake.sleep import sleep_replay as sr
    from snn_data_lake.sleep.lifelong_consolidation import LifelongConsolidator

    run.work = {"P-DP": 0, "P-EY": 0, "P-LL": 0}
    n2 = sr.N * sr.N
    steps = sr.T_CUE + sr.T_FREE
    o_step, o_batch = sr.AttractorNet._recurrent_step, sr.AttractorNet._batch_states
    o_rates, o_cls = NGSNN._rates, LifelongConsolidator.classify

    def step(self, *a, **k):  # the scalar loop visits every (i, k) pair: N*N, zeros included
        run.work["P-DP"] += n2
        return o_step(self, *a, **k)

    def batch(self, trials, *a, **k):  # the fused loop: (T_CUE+T_FREE) steps x B trials x N*N
        run.work["P-DP"] += steps * len(trials) * n2
        return o_batch(self, trials, *a, **k)

    def rates(self, X, W, b, *a, **k):  # current W f (N*L*d) + T LIF updates per unit (N*L*T)
        n, (L, d) = int(X.shape[0]), (int(W.shape[0]), int(W.shape[1]))
        t = self.T if X.ndim == 2 else int(X.shape[1])
        run.work["P-EY"] += n * L * d + n * L * t
        return o_rates(self, X, W, b, *a, **k)

    def classify(self, x, *a, **k):  # nearest prototype: n_units x dim
        run.work["P-LL"] += self.n_units * self.dim
        return o_cls(self, x, *a, **k)

    return [
        (sr.AttractorNet, "_recurrent_step", step),
        (sr.AttractorNet, "_batch_states", batch),
        (NGSNN, "_rates", rates),
        (LifelongConsolidator, "classify", classify),
    ]


# ---- neuters, sabotage twins and the floor control (2.3; each surgical, each disclosed) ----
def _dp_sabotage(run, schedule):
    """Remove the SAME count per call as the live arm, chosen adversarially: the STRONGEST synapses."""
    from snn_data_lake.sleep import sleep_replay as sr

    calls = [0]

    def sab(self, eps=1e-3):
        i = calls[0]
        calls[0] += 1
        k = schedule[i] if i < len(schedule) else 0
        cand = sorted(
            (
                (self.W[r][c], r, c)
                for r in range(sr.N)
                for c in range(sr.N)
                if r != c and self.W[r][c] > 0.0
            ),
            reverse=True,
        )
        if k > len(cand):
            run.sabotage_shortfall += k - len(cand)
        removed = 0.0
        for w, r, c in cand[:k]:
            self.W[r][c] = 0.0
            removed += w
        self._normalize()
        total = sr.N * (sr.N - 1)
        n = min(k, len(cand))
        return {"pruned": n, "total": total, "weight_removed": removed, "frac_pruned": n / total}

    return sab


def _ey_neuter(self, X, y, tol=0.0):
    self.last_keep = None
    return 0


def _ey_sabotage(run, schedule):
    """Remove the SAME count per call as the live arm: the units whose removal costs the most
    TRAINING-row RMSE (A8 -- never the held-out rows that are the outcome). Keeps >= 1 unit."""
    import torch

    from snn_data_lake.growth.ng_snn import marginal_holdout

    calls = [0]

    def sab(self, X, y, tol=0.0):
        i = calls[0]
        calls[0] += 1
        k = schedule[i] if i < len(schedule) else 0
        self.last_keep = None
        if k <= 0 or self.W is None or self.beta is None or int(self.W.shape[0]) <= 1:
            run.sabotage_shortfall += max(0, k)
            return 0
        X = X.to(self.device).float()
        y = y.to(self.device)
        N, m = X.shape[0], int(self.beta.shape[1])
        Y = torch.zeros(N, m, device=self.device)
        Y[torch.arange(N, device=self.device), y] = 1.0
        tr_list, _ = marginal_holdout(y.tolist(), max(8, int(0.2 * N)))
        tr = torch.tensor(tr_list, device=self.device)
        S = self._rates(X, self.W, self.b)
        L = int(S.shape[1])

        def tr_rmse(cols):
            beta = self._solve_beta(S[tr][:, cols], Y[tr])
            return (Y[tr] - S[tr][:, cols] @ beta).pow(2).mean().sqrt().item()

        cost = {j: tr_rmse([c for c in range(L) if c != j]) for j in range(L)}
        n = min(k, L - 1)
        run.sabotage_shortfall += k - n
        drop = set(sorted(range(L), key=lambda j: (-cost[j], j))[:n])
        keep = [c for c in range(L) if c not in drop]
        self.last_keep = list(keep)
        self.W, self.b = self.W[keep], self.b[keep]
        self.beta = self._solve_beta(S[tr][:, keep], Y[tr])
        return n

    return sab


def _ll_top_utility(self):
    """Sabotage: drop units above the budget by the HIGHEST utility (the module's ranking, reversed)."""
    n = self.n_units
    if not self.prune_on or n <= self.budget:
        return
    ranked = sorted(range(n), key=lambda i: (-self.utility[i], i))
    victims = set(ranked[: n - self.budget])
    keep = [i for i in range(n) if i not in victims]
    self.learner.prototypes = [self.learner.prototypes[i] for i in keep]
    self.labels = [self.labels[i] for i in keep]
    self.utility = [self.utility[i] for i in keep]


def _arm_patches(brain, arm, run, schedule):
    from snn_data_lake.growth.ng_snn import NGSNN

    if arm in ("live", "dp_floor"):
        if (
            arm == "dp_floor"
        ):  # C4: no offline replay learning at all -- the instrument must read the floor
            return [(brain.attractor, "sleep", lambda *a, **k: None)]
        return []
    if arm == "dp_neuter":
        return [
            (
                brain.attractor,
                "downscale_prune",
                lambda eps=1e-3: {
                    "pruned": 0,
                    "total": 0,
                    "weight_removed": 0.0,
                    "frac_pruned": 0.0,
                },
            )
        ]
    if arm == "dp_sabotage":
        return [
            (
                brain.attractor,
                "downscale_prune",
                _dp_sabotage(run, schedule).__get__(brain.attractor),
            )
        ]
    if (
        arm == "ey_neuter"
    ):  # CLASS level: the model is built inside `_refit`, no instance exists to patch
        return [(NGSNN, "prune_unearned", _ey_neuter)]
    if arm == "ey_sabotage":
        return [(NGSNN, "prune_unearned", _ey_sabotage(run, schedule))]
    if arm == "ll_neuter":
        brain._ll.prune_on = False
        return []
    if arm == "ll_random":  # the module's own control
        brain._ll.prune_mode = "random"
        return []
    if arm == "ll_sabotage":
        return [(brain._ll, "_prune_to_budget", _ll_top_utility.__get__(brain._ll))]
    raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")


def _outcome_patches(brain, run):
    """Read the three outcomes off the counters the brain already computes (no extra read of learning)."""
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    o_wake = brain._enter_wake
    o_refit = NeurogenesisSelfImprove._refit

    def wake(*a, **k):
        acc = float(brain._consolidation_acc)
        run.nights.append(
            [len(run.nights) + 1, bool(brain._stored_any), int(round(acc * DP_TRIALS))]
        )
        return o_wake(*a, **k)

    def refit(self, *a, **k):
        n0 = len(self.trajectory)
        out = o_refit(self, *a, **k)
        if len(self.trajectory) > n0:
            n = int(self._heldout_n)
            run.refits.append([int(self.ticks), n, int(round(float(self._acc) * n))])
        return out

    return [(brain, "_enter_wake", wake), (NeurogenesisSelfImprove, "_refit", refit)]


def run_arm(seed, arm, *, ticks=TICKS, schedule=None, teach=True):
    """One (seed, arm) on a fresh throwaway brain. `schedule` = the live arm's per-call removals for the
    sabotage twins. The caller owns NORD_HOME (a throwaway); this refuses the live home."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")
    _assert_scratch_home()
    from snn_data_lake.engine.brain import make_brain

    brain = make_brain(seed=seed)
    run = ArmRun(seed=seed, arm=arm, ticks=ticks)
    rec = Recorder()
    patches = (
        counting_patches(rec)
        + _work_patches(run)
        + _outcome_patches(brain, run)
        + _arm_patches(brain, arm, run, list(schedule or []))
    )
    with _patched(patches):
        _drive(brain, ticks, rec, teach)
    neuro = brain.neuro
    run.expected_refits = (neuro.ticks // neuro.refit_every) if neuro is not None else 0
    run.ll_learned = int(brain._ll_i)
    for t in range(run.ll_learned):
        test = brain._ll_tasks[t]["test"]
        run.ll.append([t, sum(1 for x, y in test if brain._ll.classify(x) == y), len(test)])
    run.removals = {pid: [list(r) for r in rows] for pid, rows in sorted(rec.rows.items())}
    return run.finish()


def schedule_of(live, pid):
    """The live arm's per-call removal counts for one pruner (what its sabotage twin must match)."""
    return [n for _, n in live.removals.get(pid, [])]


# ---- outcomes as pooled correct/total over A2's fixed windows; a short window REFUSES ----
def outcome(run, pid):
    """-> (correct, total, None) or (None, None, refusal reason)."""
    if pid == "P-DP":
        by_k = {k: (stored, c) for k, stored, c in run.nights}
        if any(k not in by_k for k in NIGHTS):
            return None, None, f"P-DP: the run ends before night {NIGHTS[-1]}"
        if not by_k[NIGHTS[0]][0]:
            return (
                None,
                None,
                f"P-DP: no stored pattern at night {NIGHTS[0]}, the first window night",
            )
        return sum(by_k[k][1] for k in NIGHTS), DP_TRIALS * len(NIGHTS), None
    if pid == "P-EY":
        if len(run.refits) != run.expected_refits or not run.refits:
            return (
                None,
                None,
                f"P-EY: {len(run.refits)} refits measured, {run.expected_refits} scheduled",
            )
        return sum(r[2] for r in run.refits), sum(r[1] for r in run.refits), None
    if pid == "P-LL":
        if run.ll_learned < 3:
            return None, None, f"P-LL: {run.ll_learned} of 3 tasks learned"
        return sum(r[1] for r in run.ll), sum(r[2] for r in run.ll), None
    raise ValueError(pid)


def _acc(run, pid):
    c, t, why = outcome(run, pid)
    return (c / t if t else None), why


# ================================================================================================
# 3. DECISION (2.3 R0-R4 + A1 + A9 + A11), first failing reading decides
# ================================================================================================
@dataclass
class Verdict:
    pruner: str
    disposition: str  # KEEP | RETIRE-OR-EXEMPT | INERT | REFUSED
    words: str
    readings: dict


def superiority(name, diffs):
    """A9 / ruling 9: a degenerate superiority contrast reads NOT MET (stricter than `contrast`)."""
    c = contrast(name, diffs)
    return c._replace(met=False) if c.degenerate else c


def non_inferiority(name, diffs, delta):
    """A9 / ruling 9a: R4 is read by `contrast` on (live - neutered + delta); degenerate -> its common value."""
    return contrast(name, [d + delta for d in diffs])


def decide_pruner(pid, seeds, *, delta="registered", repeat_ok=True):
    """`seeds`: per-seed dicts {"live", "neuter", "sabotage", "floor"(P-DP), "random"(P-LL)} of
    accuracy (or None with "refused": [reasons]), {"removals_live": int}, {"work_live", "work_neuter"}.
    `repeat_ok` is R0 (the live arm byte-identical twice on the dev seeds)."""
    delta = DELTA[pid] if delta == "registered" else delta
    r = {"seeds": len(seeds)}
    refusals = [why for s in seeds for why in s.get("refused", [])]
    if refusals:
        return Verdict(pid, "REFUSED", "per-seed precondition failed", {**r, "refused": refusals})
    if delta is None:
        return Verdict(pid, "REFUSED", "no registered margin (T_ref not frozen)", r)
    if not repeat_ok:
        return Verdict(pid, "REFUSED", "R0: nondeterministic instrument", r)
    zero = [s["seed"] for s in seeds if s["removals_live"] <= 0]
    if zero:
        return Verdict(
            pid, "INERT", f"R1: no removal over the run on seeds {zero}", {**r, "inert_seeds": zero}
        )
    r2 = superiority(f"{pid} R2 live > sabotage", [s["live"] - s["sabotage"] for s in seeds])
    r["R2"] = r2
    if not r2.met:
        return Verdict(pid, "REFUSED", "R2: blind outcome (the sabotage twin did not lose)", r)
    if pid == "P-DP":
        pinned = all(abs(s["floor"] - FLOOR_DP) < 1e-12 for s in seeds)
        off = all(abs(s["live"] - FLOOR_DP) >= 1e-12 for s in seeds)
        r["C4_floor_pinned"], r["C4_live_off_floor"] = pinned, off
        if not (pinned and off):
            return Verdict(pid, "REFUSED", "C4: harness-alive control failed", r)
    if pid == "P-LL" and all("random" in s for s in seeds):
        r["random_control"] = contrast(
            f"{pid} live > random (reported)", [s["live"] - s["random"] for s in seeds]
        )
    d = [s["live"] - s["neuter"] for s in seeds]
    r3 = superiority(f"{pid} R3 benefit", d)
    r4 = non_inferiority(f"{pid} R4 non-inferiority (delta {delta:.6g})", d, delta)
    e = superiority(
        f"{pid} E executed work saved", [s["work_neuter"] - s["work_live"] for s in seeds]
    )
    r.update(R3=r3, R4=r4, E=e, union_bound=2 * 0.05)
    if r4.met and (r3.met or e.met):
        words = " + ".join(w for w, ok in (("beneficial", r3.met), ("efficient", e.met)) if ok)
        return Verdict(pid, "KEEP", f"KEEP ({words})", r)
    why = (
        "R4 not met"
        if not r4.met
        else "R4 met but neither R3 nor E (no benefit, no measured saving)"
    )
    return Verdict(
        pid,
        "RETIRE-OR-EXEMPT",
        f"{why}: retire (P3 disposition 2) or an owner-visible dated exemption",
        r,
    )


def seed_readings(pid, runs):
    """Fold one seed's arms ({arm: ArmRun}) into `decide_pruner`'s per-seed dict."""
    arm = {"P-DP": "dp", "P-EY": "ey", "P-LL": "ll"}[pid]
    s = {"seed": runs["live"].seed, "refused": []}
    for key, name in (
        ("live", "live"),
        ("neuter", f"{arm}_neuter"),
        ("sabotage", f"{arm}_sabotage"),
    ):
        a, why = _acc(runs[name], pid)
        s[key] = a
        if why:
            s["refused"].append(f"{name}: {why}")
    if pid == "P-DP":
        s["floor"] = _acc(runs["dp_floor"], pid)[0]
    if pid == "P-LL" and "ll_random" in runs:
        s["random"] = _acc(runs["ll_random"], pid)[0]
    s["removals_live"] = sum(n for _, n in runs["live"].removals.get(pid, []))
    s["work_live"] = runs["live"].work[pid]
    s["work_neuter"] = runs[f"{arm}_neuter"].work[pid]
    if runs[f"{arm}_sabotage"].sabotage_shortfall:
        s["refused"].append(
            f"{arm}_sabotage could not match the live count ({runs[f'{arm}_sabotage'].sabotage_shortfall} short)"
        )
    return s


def surgical_ok(live, neutered, pid):
    """C5: neutering `pid` leaves every OTHER pruner's activity series byte-identical up to the first tick
    `pid` removed something in the live arm (or over the whole run if it never did)."""
    first = next((t for t, n in live.removals.get(pid, []) if n > 0), None)
    for other in set(live.removals) | set(neutered.removals):
        if other == pid:
            continue
        a = [r for r in live.removals.get(other, []) if first is None or r[0] < first]
        b = [r for r in neutered.removals.get(other, []) if first is None or r[0] < first]
        if a != b:
            return False, other
    return True, None


# ================================================================================================
# self-check (brain-free): the census finds the registered table and its two C1 reds fire; the removal
# extractors read planted counts (C2); the decision reads every branch on constructed readings; the
# executed-work counter discriminates (C25).
# ================================================================================================
def _brain_source():
    with open(os.path.join(_PKG_DIR, "engine", "brain.py"), encoding="utf-8") as fh:
        return fh.read()


_DP_LINE = "self._shy = self.attractor.downscale_prune()"


def planted_sources(kind):
    """C1's two reds: 'add' plants a vocabulary call into _sleep_tick; 'delete' removes downscale_prune."""
    text = _brain_source()
    assert text.count(_DP_LINE) == 1, "the P-DP call site moved; C1's anchor must be re-read"
    if kind == "add":
        text = text.replace(_DP_LINE, _DP_LINE + "\n            self.attractor.evict_weakest()")
    elif kind == "delete":
        text = text.replace(_DP_LINE, "self._shy = None")
    else:
        raise ValueError(kind)
    return {"engine.brain": text}


def _constructed(pid, live, neuter, sabotage, *, floor=None, removals=5, w_live=10, w_neuter=10):
    out = []
    for i in range(8):
        jitter = (i % 3) * 1e-3
        s = {
            "seed": i,
            "live": live + jitter,
            "neuter": neuter + jitter * (1 if i % 2 else -1),
            "sabotage": sabotage,
            "removals_live": removals,
            "work_live": w_live,
            "work_neuter": w_neuter
            + ((i % 2) - 0.5),  # symmetric jitter: no saving unless w_neuter > w_live
        }
        if floor is not None:
            s["floor"] = floor
        out.append(s)
    return out


def main():
    cen = census_static()
    assert cen.missing() == [], f"static census misses registered pruners: {cen.missing()}"
    assert not cen.unclassified(), f"unclassified prune-vocabulary sites: {cen.unclassified()}"
    add = census_static(sources=planted_sources("add"))
    assert any(s.method == "evict_weakest" and s.pruner == "UNCLASSIFIED" for s in add.sites), (
        "C1 add: unseen"
    )
    gone = census_static(sources=planted_sources("delete"))
    assert gone.missing() == ["P-DP"], (
        f"C1 delete: the census must red on a removed site ({gone.missing()})"
    )

    # C2: each extractor reads a planted count exactly
    assert _removed_dict({"pruned": 7}, None, None) == 7 and _removed_int(4, None, None) == 4
    assert _removed_units(None, 9, 6) == 3 and _removed_lake((10, 2), None, None) == 2
    assert _removed_lake([1, 2], 5, None) == 3

    # the decision's branches (A1, A9, A11), constructed
    keep = decide_pruner("P-LL", _constructed("P-LL", 0.9, 0.8, 0.5))
    assert keep.disposition == "KEEP" and "beneficial" in keep.words, keep
    eff = decide_pruner("P-LL", _constructed("P-LL", 0.9, 0.9, 0.5, w_live=5, w_neuter=10))
    assert eff.disposition == "KEEP" and eff.words == "KEEP (efficient)", eff
    inert = decide_pruner("P-LL", _constructed("P-LL", 0.9, 0.9, 0.5, removals=0))
    assert inert.disposition == "INERT", inert
    blind = decide_pruner("P-LL", _constructed("P-LL", 0.9, 0.9, 0.95))
    assert blind.disposition == "REFUSED" and "R2" in blind.words, blind
    idle = decide_pruner(
        "P-LL", _constructed("P-LL", 0.9, 0.9, 0.5)
    )  # harmless, no benefit, no saving
    assert idle.disposition == "RETIRE-OR-EXEMPT", idle
    worse = decide_pruner("P-LL", _constructed("P-LL", 0.7, 0.9, 0.5, w_live=5, w_neuter=10))
    assert worse.disposition == "RETIRE-OR-EXEMPT" and not worse.readings["R4"].met, worse
    dead = decide_pruner("P-DP", _constructed("P-DP", 0.5, 0.4, 0.3, floor=0.4))
    assert dead.disposition == "REFUSED" and "C4" in dead.words, dead
    nomargin = decide_pruner("P-EY", _constructed("P-EY", 0.9, 0.8, 0.5), delta=None)
    assert nomargin.disposition == "REFUSED" and "T_ref" in nomargin.words, nomargin
    # A9: a degenerate superiority contrast is NOT MET; a degenerate R4 reads by its common value
    assert not superiority("deg", [0.1] * 8).met and non_inferiority("deg", [0.0] * 8, 0.01).met
    assert not non_inferiority("deg", [-0.02] * 8, 0.01).met

    # C25: the executed-work counter discriminates a dense recurrence from one that skips pruned units
    class Dense:
        def __init__(self, n):
            self.W = [[1.0] * n for _ in range(n)]

        def step(self, z):
            ops = 0
            for _i in range(len(self.W)):
                for _k in range(len(self.W)):
                    ops += 1  # visited whether the weight is zero or not
            return ops

    class Sparse(Dense):
        def step(self, z):
            return sum(1 for row in self.W for w in row if w != 0.0)

    for cls_, expect_saving in ((Dense, False), (Sparse, True)):
        live, neut = cls_(6), cls_(6)
        for r in range(6):
            live.W[r][(r + 1) % 6] = 0.0  # the live arm pruned six synapses
        w_live, w_neut = live.step(None), neut.step(None)
        assert (w_neut - w_live == 6) is expect_saving and (w_neut == w_live) is (not expect_saving)

    print(
        f"sleep_pruner_census self-check OK: census {cen.found()} via {len(cen.sites)} sites "
        f"({cen.unresolved} unresolved receivers counted, 0 in the prune vocabulary); C1 add+delete red; "
        f"C2 extractors exact; decision KEEP(beneficial)/KEEP(efficient)/INERT/REFUSED(R2)/REFUSED(C4)/"
        f"REFUSED(no T_ref)/RETIRE x2; A9 degenerate rules; C25 dense vs sparse; bar {T_BAR}"
    )


def _cli(argv):
    import argparse

    from snn_data_lake.substrate.home import throwaway_home

    p = argparse.ArgumentParser(prog="sleep_pruner_census")
    p.add_argument("--census", action="store_true")
    p.add_argument("--arm", nargs=2, metavar=("SEED", "ARM"))
    p.add_argument("--schedule")
    p.add_argument("--out")
    p.add_argument("--ticks", type=int, default=TICKS)
    p.add_argument("--seed", type=int, default=12001)
    a = p.parse_args(argv)
    if a.census:
        with throwaway_home():
            _assert_scratch_home()
            from snn_data_lake.engine.brain import make_brain

            b = make_brain(seed=a.seed)
            cen = census_static(runtime_types=runtime_types_of(b))
            seen = set()
            with discovering(seen):
                rec = census_runtime(b, a.ticks)
        print(
            json.dumps(
                {
                    "table": cen.table(),
                    "missing": cen.missing(),
                    "unresolved": cen.unresolved,
                    "misses": census_misses(cen, seen),
                    "discovered": sorted(seen),
                    "runtime": {p_: [rec.calls(p_), rec.removals(p_)] for p_ in sorted(rec.rows)},
                },
                indent=1,
            )
        )
        return 0
    if a.arm:
        sched = json.load(open(a.schedule)) if a.schedule else None
        with throwaway_home():
            run = run_arm(int(a.arm[0]), a.arm[1], ticks=a.ticks, schedule=sched)
        text = json.dumps(asdict(run), sort_keys=True)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            print(text)
        return 0
    main()
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
