"""p6_gate -- the workspace write:read topology. P6's gate, specified in docs/POLICIES.md and, until
this file, NOT BUILT, which is why nothing has been forcing the issue.

  set PYTHONUTF8=1 & .venv-lake\\Scripts\\python.exe tools/p6_gate.py

WHAT IT MEASURES, AND WHY THIS PROXY. P6 asks whether a faculty changes ANOTHER FACULTY. P3 asks
whether a faculty changes BEHAVIOUR -- the arm, a value, a weight, a reply. The two come apart, and
the gap is the whole point: a faculty wired straight into `_route_percept_to_action` charges the arm
decision and talks to nothing, which is P3-clean and P6-dead. The global workspace is the one
structure in this brain shaped like a mutual-consumption bus (many writers; later computation READS
it -- the role a transformer's residual stream plays), so its write:read topology is the checkable
proxy POLICIES.md names.

HOW TO READ THE OUTPUT. A rising READ count against a flat WRITE count is what P6 progress looks
like. A rising WRITE count on its own is NOT progress -- it is more shouting into the same sink, and
this gate is deliberately built so that adding a `submit()` moves the number that does not matter.

WHAT WOULD FALSIFY A P6 CLAIM MADE WITH THIS TOOL, stated up front because POLICIES.md requires it.
This walk is STATIC. It proves a read EXISTS; it cannot prove the read CHANGES the reader's output.
The failure this repo has already built, measured and reverted -- see the worked example at
`brainstem/autonomous_loop.py:969` -- is a wire that turns a constant telemetry field into a varying
one and changes nothing downstream (`_wm_spikes` 1 distinct -> 22 distinct, selected action changed
0 of 200 ticks). So a green line here is a NECESSARY, not sufficient, condition. The sufficient
evidence is the neuter-A/B `tools/charge_probe.py` performs, aimed one level over: disable the
WRITER faculty and show the READER faculty's own output changes. If it does not, the wire is
reverted and the faculty is exempted with a reason, exactly as under P3.

--------------------------------------------------------------------------------------------------
MEASURED 2026-08-02 on HEAD 95c3f806 + working tree. Every number below is reproduced by running
this file; none of it is quoted from the prior reading.

  42 workspace writes across 10 modules, 9 reads across 5 modules, 10 subscribe registrations.
  25 of the 42 writes are inside the two workspace modules' own self-checks (global_workspace 14,
  reentrant_workspace 11), so real broadcast is 17 writes from 8 modules.
  CROSS-MODULE reads: 1 of 9 -- `cortex/live_brain.py:287`, and live_brain is the tick driver.
  FACULTY-to-FACULTY reads: 0 of 9.

THE VERDICT IS UNCHANGED AND THE NUMBERS ARE NOT. P6's 2026-07-31 reading recorded "36 writes across
8 modules, 2 reads across 2". Four differences, each a finding about the INSTRUMENT rather than about
the tree, because git blame dates every affected line to 2026-07-21 -- ten days BEFORE that reading:

  1. `cortex/brain.py` writes the bus THREE times (`:1733 :1738 :1742`, `self._reentrant.submit`) and
     was absent from the prior module list entirely. The receiver is named `_reentrant`, not
     `workspace`/`ws`/`_ws`; a walk keyed on those names cannot see it. This is the single largest
     correction: 8 writer modules were really 9 on that same tree.
  2. `cortex/brain.py:1610` `self._reentrant.broadcast_to("action")` is a REAL read, and a consuming
     one -- it feeds `elig["reroute"]` into the three-factor drive-weight update at `:1615-1620`. The
     prior reading did not have it, so "2 reads" was really 3 on that same tree.
  3. `reasoning/developmental_loop_brain.py:295` was counted as one of the two READS. It is a WRITE.
     The receiver is `WorkspaceBroadcast` (`reasoning/brain_adapters/executive_control.py:90`), a
     wrapper whose `broadcast(source, content, salience)` calls `_ws.submit()` then `_ws.step()`. The
     giveaway is arity: `GlobalWorkspace.broadcast()` takes ZERO arguments, so a 3-argument
     `.broadcast(` is a submit wearing a reader's name. This gate splits them on arity, which is why
     it is the one distinction hard-coded into the API table below.
  4. `executive_control.py:231` and `:234` are two more writes of the same wrapper shape, uncounted
     before for the same reason.

  So the prior headline was low on both sides -- it missed a writer module and a read, and it counted
  a write as a read. Netting to a small number by two errors in opposite directions is exactly the
  silent-degradation shape this repo keeps catching, and it is why the self-check at the bottom
  asserts the walk still SEES what it saw, rather than asserting a total.

  What did NOT change is the conclusion, and that matters more than the counts. Every read but one is
  a module reading a workspace instance IT ITSELF CONSTRUCTED (`train_of_thought` reads `self.ws`;
  `brain` reads the `self._reentrant` it submits to; the two bus modules read themselves in their own
  self-checks). The one cross-module read is `live_brain.thought`, and its consumers are a telemetry
  vector (`:240`, `thought_on` -> a region-activation dict) and a chat string (`:402`). So no faculty
  reads another faculty's contribution: zero-way at the faculty level, precisely what P6 forbids.

THE SHARPER FINDING THE RATIO HIDES: THERE IS NO BUS. There are 26 workspace INSTANTIATION sites
across 10 modules -- 11 of them outside the two bus modules' self-checks, across 8 modules -- and
each faculty constructs its OWN private workspace object: `train_of_thought` `self.ws`,
`access_consciousness` `self._ws`, `rpm_solver` a function-local `ws`, `spike_chat_lake` a
function-local `ws`, `WorkspaceBroadcast` `self._ws`, `brain` `self._reentrant`, `autonomous_loop`
`self.workspace`. Faculty A's `submit()` is not merely unread by faculty B; it is INVISIBLE to it,
because they are different objects. `cortex/brain.py:2299-2307` already records half of this
discovery from the runtime side (25 GlobalWorkspace constructions over 40 ticks against ONE
ReentrantWorkspace). The remedy POLICIES.md proposes -- "faculties should READ the workspace, not
only submit() to it" -- therefore cannot be done by adding a reader alone. There is no shared `it` to
read. Sharing one instance is the prerequisite, and it is the cheaper half of the work.

TWO SIGHTINGS OF THE FAILURE THE HONESTY GUARD PREDICTS, both surfaced by running this file:

  * `cortex/brain.py:714` subscribes TWO specialists, "action" and "language". Only "action" is ever
    drained (`:1610`). "language" is a registered subscriber whose inbox nothing reads -- a route that
    carries nothing, which POLICIES.md says is reverted and exempted, not counted.
  * `thalamus/reentrant_workspace.py:399-409` ships `_Specialist.emit()`, a WORKING example of one
    faculty consuming another's broadcast (`bias = ws.broadcast_to(self.name)`, added to its next
    output). It exists only inside the bus's own self-check. The pattern P6 asks for is already
    written, already passing, and used by zero faculties.

WHAT THIS GATE DOES NOT DO. It does not diff against `connectome.py`'s 81 declared region->region
edges (verified 81 today, 40 nodes, 15 regions). That diff is the other half of the instrument
POLICIES.md specifies and is deliberately out of scope here: the workspace topology is the half that
returns a non-rubber-stamp-sized answer, and per this repo's own bar a gate whose output is a
rubber-stamp-sized list kills the gate, not the tree.
--------------------------------------------------------------------------------------------------
"""

import ast
import os
from collections import defaultdict

# --------------------------------------------------------------------------------------------------
# What counts as "the workspace". Two bus classes plus the one wrapper that owns a bus instance and
# re-exports it under a different verb -- omitting the wrapper is how the prior reading turned three
# writes into one read.
# --------------------------------------------------------------------------------------------------
WORKSPACE_CLASSES = {"GlobalWorkspace", "ReentrantWorkspace", "WorkspaceBroadcast"}

# (method, total argument count) -> kind. Arity is load-bearing, not decoration: `broadcast` with
# zero arguments is GlobalWorkspace's reader and `broadcast` with three is WorkspaceBroadcast's
# writer. Any other arity is some other class's method that merely shares a name (Claustrum.broadcast
# takes two; the cockpit's chat queue has a one-argument submit) and is correctly not matched.
API = {
    ("submit", 3): "WRITE",
    ("broadcast", 3): "WRITE",
    ("broadcast", 0): "READ",
    ("broadcast_to", 1): "READ",
    ("inbox", 1): "READ",
    ("subscribe", 1): "REGISTER",
}
READ_ATTR = "current"  # a property on both bus classes; an attribute load, not a call

# REGISTER is reported but never folded into the read count. Registering a callback is not
# consumption -- POLICIES.md's honesty guard says so in as many words ("adding a reader that stores a
# value is not consumption"), and a `subscribe()` that nothing later drains is exactly the telemetry
# bus P6 predicts it will attract.

# --------------------------------------------------------------------------------------------------
# Module roles. Declared, not inferred, so the classification is auditable instead of hidden -- and
# every entry carries the reason it claims, printed in the report. `verify_roles()` re-derives each
# claim from the AST so a stale label goes RED rather than quietly mislabelling a module.
# --------------------------------------------------------------------------------------------------
ROLES = {
    "thalamus/global_workspace.py": (
        "BUS",
        "defines GlobalWorkspace; its writes are its own __main__ self-check, not broadcast",
    ),
    "thalamus/reentrant_workspace.py": (
        "BUS",
        "defines ReentrantWorkspace; likewise self-check writes",
    ),
    "cortex/live_brain.py": (
        "ACTION",
        "the tick driver: owns tick() and constructs the AutonomousArm it ticks",
    ),
    "brainstem/autonomous_loop.py": (
        "ACTION",
        "the arm's wake loop -- step() is the single action gate P6 says faculties all report to",
    ),
}
DEFAULT_ROLE = ("FACULTY", "not the bus and not the action path")

# The self-check's floor. Recorded 2026-08-02 from this walk's own output, and asserted as a MINIMUM
# so that adding a write is fine and losing sight of one is red. It is a liveness assertion about the
# INSTRUMENT: a broken walk resolves to zero findings, which reads exactly like a clean tree.
KNOWN_WRITERS = {
    "thalamus/global_workspace.py": 14,
    "thalamus/reentrant_workspace.py": 11,
    "reasoning/train_of_thought.py": 4,
    "brainstem/autonomous_loop.py": 3,
    "cortex/brain.py": 3,
    "reasoning/brain_adapters/executive_control.py": 3,
    "cortex/access_consciousness.py": 1,
    "reasoning/rpm_solver.py": 1,
    "reasoning/spike_chat_lake.py": 1,
    "reasoning/developmental_loop_brain.py": 1,
}
KNOWN_READERS = {
    "thalamus/global_workspace.py": 3,
    "reasoning/train_of_thought.py": 2,
    "thalamus/reentrant_workspace.py": 2,
    "cortex/brain.py": 1,
    "cortex/live_brain.py": 1,
}


# --------------------------------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------------------------------
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_dir():
    return os.path.join(repo_root(), "src", "snn_data_lake")


def rel(path):
    """Repo-relative POSIX path under src/snn_data_lake/, e.g. 'cortex/brain.py'."""
    return os.path.relpath(path, source_dir()).replace(os.sep, "/")


def chain(node):
    """The dotted segments of a receiver expression, outermost last.

    `self.arm.workspace` -> ['self', 'arm', 'workspace'].  A non-name root yields None, which is how
    a receiver this walk cannot attribute stays visibly unattributed instead of being guessed at."""
    segs = []
    cur = node
    while isinstance(cur, ast.Attribute):
        segs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    segs.append(cur.id)
    segs.reverse()
    return segs


def ctor_class(value):
    """The workspace class this expression constructs, or None.

    Walks the whole value subtree so `GlobalWorkspace(seed=s) if on else None` is seen -- the exact
    shape of `autonomous_loop.py:230`, the module that owns the only instance anything else reads."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else None)
            )
            if name in WORKSPACE_CLASSES:
                return name
    return None


# --------------------------------------------------------------------------------------------------
# Pass 1 -- vocabulary: which receiver names denote a workspace, and who constructs one
# --------------------------------------------------------------------------------------------------
class _BindVisitor(ast.NodeVisitor):
    """Collect workspace constructions and `self.<attr> = SomeClass(...)` bindings.

    The second kind is what lets `self.arm.workspace` resolve to the module that owns that instance
    rather than to whichever module happens to use the token `workspace`."""

    def __init__(self, relpath):
        self.rel = relpath
        self.instances = []  # (token, class_name, lineno)
        self.attr_class = {}  # 'arm' -> 'AutonomousArm'

    def _handle(self, targets, value, lineno):
        cls = ctor_class(value)
        for tgt in targets:
            segs = chain(tgt)
            if segs is None:
                continue
            token = segs[-1]
            if cls is not None:
                self.instances.append((token, cls, lineno))
            elif isinstance(value, ast.Call) and segs[0] == "self" and len(segs) == 2:
                fn = value.func
                cname = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else (fn.attr if isinstance(fn, ast.Attribute) else None)
                )
                if cname and cname[:1].isupper():
                    self.attr_class[token] = cname

    def visit_Assign(self, node):
        self._handle(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._handle([node.target], node.value, node.lineno)
        self.generic_visit(node)


# --------------------------------------------------------------------------------------------------
# Pass 2 -- sites: every workspace read, write and registration
# --------------------------------------------------------------------------------------------------
class _SiteVisitor(ast.NodeVisitor):
    def __init__(self, relpath, vocab):
        self.rel = relpath
        self.vocab = vocab  # receiver tokens known to denote a workspace, tree-wide
        self.sites = []  # (kind, lineno, method, receiver_text, specialist_names)
        self._cls = []
        self._as_func = set()
        self._loop = {}  # loop var -> the constant strings it takes, for `for s in ("a","b")`

    def visit_ClassDef(self, node):
        self._cls.append(node.name)
        self.generic_visit(node)
        self._cls.pop()

    def visit_For(self, node):
        """Bind `for _spec in ("action", "language")` so the subscriber NAMES stay resolvable.

        Without this, `brain.py:714` registers two specialists under an unreadable loop variable and
        the drained-vs-registered diff below can only shrug -- which is how a subscriber nothing ever
        drains stays invisible."""
        bound = None
        if isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)):
            vals = [
                e.value
                for e in node.iter.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if vals and len(vals) == len(node.iter.elts):
                bound = node.target.id
                self._loop[bound] = tuple(vals)
        self.generic_visit(node)
        if bound is not None:
            self._loop.pop(bound, None)

    def _names(self, args):
        """The specialist name(s) an API call refers to: a string literal, or a bound loop variable."""
        if not args:
            return ()
        arg = args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return (arg.value,)
        if isinstance(arg, ast.Name) and arg.id in self._loop:
            return self._loop[arg.id]
        return ()

    def _receiver(self, node):
        """(token, text) if `node` is a workspace receiver, else None.

        Inside a workspace class body `self` IS the workspace -- without this the bus modules' own
        reads of `self.current` are invisible, and a walk that cannot see the bus read itself has no
        standing to report that nobody else does."""
        segs = chain(node)
        if segs is None:
            return None
        token = segs[-1]
        if token in self.vocab:
            return token, ".".join(segs)
        if token == "self" and self._cls and self._cls[-1] in WORKSPACE_CLASSES:
            return token, "self"
        return None

    def visit_Call(self, node):
        self._as_func.add(id(node.func))
        if isinstance(node.func, ast.Attribute):
            recv = self._receiver(node.func.value)
            if recv is not None:
                arity = len(node.args) + len(node.keywords)
                kind = API.get((node.func.attr, arity))
                if kind is not None:
                    self.sites.append(
                        (kind, node.lineno, node.func.attr, recv[1], self._names(node.args))
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if (
            id(node) not in self._as_func
            and node.attr == READ_ATTR
            and isinstance(node.ctx, ast.Load)
        ):
            recv = self._receiver(node.value)
            if recv is not None:
                self.sites.append(("READ", node.lineno, READ_ATTR, recv[1], ()))
        self.generic_visit(node)


# --------------------------------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------------------------------
def walk():
    """Return (sites, instances, attr_class, class_module, n_found, n_parsed, failures)."""
    found, trees = [], {}
    for dirpath, dirnames, filenames in os.walk(source_dir()):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                found.append(os.path.join(dirpath, fn))

    failures = []
    for path in found:
        try:
            with open(path, encoding="utf-8") as fh:
                trees[rel(path)] = ast.parse(fh.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append((rel(path), type(exc).__name__))

    class_module = {}
    instances = defaultdict(list)  # relpath -> [(token, class, lineno)]
    attr_class = {}  # relpath -> {attr: ClassName}
    vocab = set()
    for relpath, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_module.setdefault(node.name, relpath)
        bv = _BindVisitor(relpath)
        bv.visit(tree)
        if bv.instances:
            instances[relpath] = bv.instances
            vocab.update(tok for tok, _c, _l in bv.instances)
        attr_class[relpath] = bv.attr_class

    sites = defaultdict(list)
    for relpath, tree in trees.items():
        sv = _SiteVisitor(relpath, vocab)
        sv.visit(tree)
        if sv.sites:
            sites[relpath] = sorted(sv.sites, key=lambda s: s[1])

    return sites, instances, attr_class, class_module, len(found), len(trees), failures


def role_of(relpath):
    return ROLES.get(relpath, DEFAULT_ROLE)


def resolve_owner(relpath, receiver_text, instances, attr_class, class_module):
    """Which module owns the workspace instance this receiver reaches, or None if unattributable.

    Resolves one hop through `self.<attr> = SomeClass(...)`: `self.arm.workspace` read in live_brain
    walks `arm` -> AutonomousArm -> brainstem/autonomous_loop.py, and only then matches `workspace`
    against that module's own constructions. Token matching alone is ambiguous here -- two modules
    bind the token `workspace` -- and an ambiguous answer to "who is read by nobody" is no answer."""
    segs = receiver_text.split(".")
    token = segs[-1]
    here = relpath
    if len(segs) >= 3 and segs[0] == "self":
        cls = attr_class.get(relpath, {}).get(segs[1])
        if cls is not None and cls in class_module:
            here = class_module[cls]
    if any(tok == token for tok, _c, _l in instances.get(here, [])):
        return here
    owners = [m for m, inst in instances.items() if any(t == token for t, _c, _l in inst)]
    return owners[0] if len(owners) == 1 else None


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def report():
    sites, instances, attr_class, class_module, n_found, n_parsed, failures = walk()

    per = {}
    for relpath, ss in sites.items():
        per[relpath] = {
            "WRITE": [s for s in ss if s[0] == "WRITE"],
            "READ": [s for s in ss if s[0] == "READ"],
            "REGISTER": [s for s in ss if s[0] == "REGISTER"],
        }

    n_writes = sum(len(v["WRITE"]) for v in per.values())
    n_reads = sum(len(v["READ"]) for v in per.values())
    n_regs = sum(len(v["REGISTER"]) for v in per.values())
    w_mods = sorted(m for m, v in per.items() if v["WRITE"])
    r_mods = sorted(m for m, v in per.items() if v["READ"])
    n_inst = sum(len(v) for v in instances.values())

    print("=" * 98)
    print("P6 GATE -- workspace write:read topology over src/snn_data_lake/")
    print("=" * 98)
    print(f"scanned {n_parsed} of {n_found} .py files parsed")
    if failures:
        for relpath, why in failures:
            print(f"  PARSE FAILURE  {relpath}  ({why})")
    print(
        f"workspace classes tracked: {len(WORKSPACE_CLASSES)} "
        f"({', '.join(sorted(WORKSPACE_CLASSES))})"
    )

    # ---- instance census: the finding the ratio hides -------------------------------------------
    print(
        "\n--- WORKSPACE INSTANCES (each construction is a SEPARATE object; a faculty's submit is"
    )
    print("    invisible to any faculty holding a different instance) ---")
    for relpath in sorted(instances):
        role, _why = role_of(relpath)
        toks = ", ".join(f"{t}={c}@:{ln}" for t, c, ln in instances[relpath])
        print(f"  {relpath:<52} [{role:<7}] {toks}")
    bus_inst_mods = [m for m in instances if role_of(m)[0] == "BUS"]
    bus_inst = sum(len(instances[m]) for m in bus_inst_mods)
    print(f"\n  {n_inst} instantiation sites across {len(instances)} modules")
    print(
        f"  {n_inst - bus_inst} of {n_inst} are outside the bus modules' self-checks, across "
        f"{len(instances) - len(bus_inst_mods)} modules -- that many PRIVATE, mutually invisible buses"
    )

    # ---- per-module topology ---------------------------------------------------------------------
    print("\n--- PER-MODULE TOPOLOGY ---")
    print(f"  {'module':<52} {'role':<8} {'W':>3} {'R':>3} {'sub':>4}  owns")
    for relpath in sorted(per):
        role, _why = role_of(relpath)
        v = per[relpath]
        owns = "yes" if relpath in instances else "NO (reaches into another module's instance)"
        print(
            f"  {relpath:<52} {role:<8} {len(v['WRITE']):>3} {len(v['READ']):>3} "
            f"{len(v['REGISTER']):>4}  {owns}"
        )

    print("\n--- EVERY SITE, so no count here is unauditable ---")
    for relpath in sorted(per):
        v = per[relpath]
        print(f"\n  {relpath}")
        for kind in ("WRITE", "READ", "REGISTER"):
            for _k, ln, meth, recv, names in v[kind]:
                tag = f"  -> {', '.join(names)}" if names else ""
                print(f"    {kind:<8} :{ln:<5} {recv}.{meth}{tag}")

    # ---- the P6 numbers --------------------------------------------------------------------------
    bus_mods = [m for m in w_mods if role_of(m)[0] == "BUS"]
    bus_writes = sum(len(per[m]["WRITE"]) for m in bus_mods)
    print("\n" + "=" * 98)
    print("P6 TOPOLOGY")
    print("=" * 98)
    print(f"  writes  : {n_writes} across {len(w_mods)} modules")
    print(f"  reads   : {n_reads} across {len(r_mods)} modules")
    print(
        f"  subscribe registrations: {n_regs}  (NOT counted as reads -- registering is not consuming)"
    )
    print(
        f"  {bus_writes} of {n_writes} writes are inside the {len(bus_mods)} bus modules' own "
        f"self-checks, so real broadcast is {n_writes - bus_writes} of {n_writes} writes"
    )

    # A read is CROSS-MODULE when the instance it reaches is owned by a different module. That is the
    # only shape that can carry one faculty's contribution to another; everything else is a module
    # talking to itself through an object it built.
    cross, selfread = [], []
    for relpath in sorted(per):
        for _k, ln, meth, recv, _n in per[relpath]["READ"]:
            owner = resolve_owner(relpath, recv, instances, attr_class, class_module)
            (cross if owner is not None and owner != relpath else selfread).append(
                (relpath, ln, meth, recv, owner)
            )
    print(
        f"\n  self-reads    : {len(selfread)} of {n_reads}  (a module reading an instance it built)"
    )
    print(f"  cross-module  : {len(cross)} of {n_reads}")
    for reader, ln, meth, recv, owner in cross:
        role, why = role_of(reader)
        print(f"      {reader}:{ln}  {recv}.{meth}  <- instance owned by {owner}")
        print(f"        reader role {role}: {why}")

    # ---- registered vs drained: the telemetry-bus shape the honesty guard predicts ---------------
    # A subscribed specialist whose inbox nothing ever reads is a route that carries nothing. It
    # inflates every "modules connected" count while changing no consumer's output -- the same shape
    # as the reverted wire at brainstem/autonomous_loop.py:969.
    registered, drained = {}, set()
    for relpath in sorted(per):
        for _k, ln, _m, _r, names in per[relpath]["REGISTER"]:
            for nm in names:
                registered.setdefault(nm, []).append(f"{relpath}:{ln}")
        for _k, _ln, meth, _r, names in per[relpath]["READ"]:
            if meth in ("broadcast_to", "inbox"):
                drained.update(names)
    undrained = sorted(n for n in registered if n not in drained)
    print(
        f"\n  named subscribers registered: {len(registered)}; drained by a broadcast_to/inbox read: "
        f"{len(registered) - len(undrained)} of {len(registered)}"
    )
    for nm in undrained:
        # Role-tagged, because a BUS registration can be a deliberate negative control -- the
        # reentrant self-check subscribes a "loser" precisely to show a sub-threshold post never wins.
        # An undrained registration OUTSIDE the bus has no such excuse.
        where = ", ".join(registered[nm])
        roles = {role_of(loc.rsplit(":", 1)[0])[0] for loc in registered[nm]}
        note = " (bus self-check control)" if roles == {"BUS"} else ""
        print(f"      NEVER DRAINED  {nm!r}  registered at {where}{note}")

    fac_cross = [c for c in cross if role_of(c[0])[0] == "FACULTY"]
    print(f"\n  FACULTY-to-FACULTY reads: {len(fac_cross)} of {n_reads}")
    if not fac_cross:
        print(
            "      No faculty reads another faculty's contribution. Zero-way at the faculty level,"
        )
        print(
            "      which is the condition P6 forbids. Condition (b) -- another faculty consumes its"
        )
        print("      output -- is unmet for EVERY writer below.")

    # ---- the answer P6 asks for ------------------------------------------------------------------
    read_owners = {c[4] for c in cross}
    unread = [m for m in w_mods if role_of(m)[0] != "BUS" and m not in read_owners]
    print("\n" + "-" * 98)
    print(f"  WRITERS READ BY NOBODY: {len(unread)} of {len(w_mods)} writer modules")
    print("-" * 98)
    for m in unread:
        role, _why = role_of(m)
        lines = ", ".join(f":{ln}" for _k, ln, _meth, _r, _n in per[m]["WRITE"])
        print(f"    {m:<52} [{role:<7}] {len(per[m]['WRITE'])} write(s) at {lines}")
    print(
        "\n  Each posts into a workspace nothing else ever reads. Adding a reader is NOT sufficient"
    )
    print(
        "  while each writer holds a PRIVATE instance -- see the instance census above; there is no"
    )
    print("  shared object for a reader to read.")
    return {
        "per": per,
        "instances": instances,
        "n_found": n_found,
        "n_parsed": n_parsed,
        "failures": failures,
        "cross": cross,
        "fac_cross": fac_cross,
        "unread": unread,
    }


# --------------------------------------------------------------------------------------------------
# Self-check. A gate that cannot go red is worthless, and the failure mode this one guards is not a
# wrong number -- it is a walk that stops matching and reports a clean tree. Every assertion below is
# about the INSTRUMENT being alive, in the shape tools/citation_guard.py already uses.
# --------------------------------------------------------------------------------------------------
def verify_roles(instances):
    """Re-derive each declared ROLE from the source so a stale label goes red."""
    problems = []
    for relpath, (role, _why) in sorted(ROLES.items()):
        full = os.path.join(source_dir(), relpath.replace("/", os.sep))
        if not os.path.exists(full):
            problems.append(f"{relpath}: declared {role} but the file does not exist")
            continue
        with open(full, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=full)
        names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        if role == "BUS" and not (names & WORKSPACE_CLASSES):
            problems.append(f"{relpath}: declared BUS but defines no workspace class")
        if role == "ACTION" and not ({"tick", "step"} & names):
            problems.append(f"{relpath}: declared ACTION but defines neither tick() nor step()")
    return problems


def main():
    res = report()
    per, instances = res["per"], res["instances"]

    print("\n" + "=" * 98)
    print("SELF-CHECK -- is this walk still alive?")
    print("=" * 98)

    assert not res["failures"], (
        "a source file failed to parse, so its sites are silently missing from every count above: "
        + ", ".join(f"{r} ({w})" for r, w in res["failures"])
    )
    assert res["n_parsed"] == res["n_found"], (
        f"parsed {res['n_parsed']} of {res['n_found']} files -- the walk is dropping files"
    )
    assert res["n_found"] > len(KNOWN_WRITERS), (
        f"found only {res['n_found']} .py files under src/snn_data_lake -- the walk lost the tree, "
        "not the tree the modules"
    )

    # The load-bearing one. Recorded 2026-08-02; asserted as a MINIMUM so a new write is fine and a
    # write this walk can no longer SEE is red. Losing brain.py here is exactly the defect that made
    # the 2026-07-31 reading say 8 writer modules when the tree had 9.
    missing, shrunk = [], []
    for relpath, floor in sorted(KNOWN_WRITERS.items()):
        got = len(per.get(relpath, {}).get("WRITE", []))
        if got == 0:
            missing.append(relpath)
        elif got < floor:
            shrunk.append(f"{relpath}: {got} of {floor} expected writes")
    assert not missing, (
        "known writer module(s) produced ZERO writes -- the receiver vocabulary or the API table has "
        "stopped matching, and a broken walk reports a clean tree: " + ", ".join(missing)
    )
    assert not shrunk, (
        "known writer module(s) lost writes. If the writes were genuinely removed, lower the floor in "
        "KNOWN_WRITERS and say so; do not leave a walk that half-matches: " + "; ".join(shrunk)
    )
    print(
        f"  writers seen: {len(KNOWN_WRITERS)} of {len(KNOWN_WRITERS)} known writer modules found"
    )

    missing_r = [
        relpath
        for relpath, floor in sorted(KNOWN_READERS.items())
        if len(per.get(relpath, {}).get("READ", [])) < floor
    ]
    assert not missing_r, (
        "known reader module(s) lost reads -- READ detection (arity-0 broadcast, broadcast_to, inbox, "
        "the `current` property) has stopped matching: " + ", ".join(missing_r)
    )
    print(
        f"  readers seen: {len(KNOWN_READERS)} of {len(KNOWN_READERS)} known reader modules found"
    )

    # Both bus classes must still be constructed somewhere, and the wrapper too -- dropping any one of
    # the three is how a whole category of sites disappears at once.
    built = {c for inst in instances.values() for _t, c, _l in inst}
    assert built == WORKSPACE_CLASSES, (
        f"constructed workspace classes {sorted(built)} != tracked {sorted(WORKSPACE_CLASSES)} -- a "
        "class was renamed, retired, or is no longer being detected"
    )
    print(f"  classes built: {len(built)} of {len(WORKSPACE_CLASSES)} tracked classes constructed")

    # The arity split is the correction that separates a write from a read. If it ever stops firing,
    # writes get reported as reads and P6 looks solved.
    wb = per.get("reasoning/brain_adapters/executive_control.py", {}).get("WRITE", [])
    assert any(m == "broadcast" for _k, _l, m, _r, _n in wb), (
        "the 3-argument WorkspaceBroadcast.broadcast() is no longer classified as a WRITE -- the "
        "arity split has broken, and a submit wearing a reader's name is what made the 2026-07-31 "
        "reading report 2 reads where there was 1"
    )
    print("  arity split  : 3-arg broadcast still classified WRITE, 0-arg still READ")

    problems = verify_roles(instances)
    assert not problems, "declared ROLES no longer match the source: " + "; ".join(problems)
    print(f"  roles verified: {len(ROLES)} of {len(ROLES)} declared roles re-derived from the AST")

    # The P6 claim itself. NOT asserted -- this is a reporting gate, and asserting 0 faculty reads
    # would go green forever by measuring nothing and red the moment someone does the right thing.
    print(
        f"\n  P6 STATUS: {len(res['fac_cross'])} faculty-to-faculty read(s). "
        f"{len(res['unread'])} writer module(s) read by nobody. Re-run after any wiring change: a "
        "rising READ count against a flat WRITE count is progress; a rising write count is not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
