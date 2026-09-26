#!/usr/bin/env python3
"""eprop_transport_gate -- a TREE-LEVEL check for weight transport, not a `.backward()` tripwire.

WHY THIS EXISTS, and why the obvious instrument is forbidden. The e-prop invariant is
review-enforced with NO GATE, and `docs/POLICIES.md` says exactly why the easy one was never built:
*"a `.backward()` tripwire would be theatre: it passed green while `cerebellum/jepa_world_model.py`
performed explicit weight transport on every wake tick ... Absence of `.backward()` is not evidence
of locality."* `tools/promotion_sweep.py`'s `BACKPROP_MARKERS` is that substring grep, it has no
`--assert`, and it scans only OUTSIDE the governed tier.

What the invariant forbids is WEIGHT TRANSPORT: a forward weight, read transposed, carrying error
into a weight update. Today exactly two modules police that, each scoped to itself --
`cerebellum/jepa_world_model.py` inside `_feedback_gz`, and `cerebellum/spiking_jepa.py`. NEITHER
CAN SEE A THIRD MODULE, which is how `basal_ganglia/mlp_q.py` sat in `run_selfchecks.py` passing
while performing transport, self-declared in its own docstring and absent from the policy record for
the nine days it took an audit to find it.

THE RULE IS THE ADJUDICATION, not a proxy for it. `POLICIES.md` settled four sites on one test,
stated in its own words -- **"THE DISTINGUISHING TEST IS WHAT THE TRANSPOSE FEEDS"**:

    jepa               `Wz^T gp`  -> the predictor's GRADIENT      -> VIOLATION (closed 2026-08-04)
    mlp_q              `W2^T`     -> `dh` -> updates `W1`          -> VIOLATION (retired 2026-08-14)
    predictive_coding  `w1.T`     -> `self.r1`, an ACTIVATION      -> NOT a violation
    ng_snn::fit        direct solve, no transpose at all           -> NOT a violation

So: a transposed WEIGHT whose value reaches a WEIGHT UPDATE, within one function. Nothing else.

COUNTED FIRST, four widths, per the standing invariant. Three are unadoptable and the numbers are
kept here because that is the whole point of counting before adopting:

    W1  any transpose anywhere in the governed tier             82   allowlist
    W2  transpose of a weight-ish name                          36   allowlist
    W3  `.backward()`/`bptt` substring (promotion_sweep's)       -   POLICIES calls it theatre
    W4  transposed weight REACHING a weight update               0   ADOPTED

W5 WAS BUILT, MEASURED, AND REJECTED, and it is recorded because it looked better than W4 on paper.
W5 asked a wider question -- does this function update weight A while reading a DIFFERENT weight B?
-- which catches credit assignment regardless of spelling. It scored **6 of 6** on the
reconstructions below, including the one W4 misses. On this tree it scored **59**, and among the 59
was `basal_ganglia/mlp_q.py::update` ITSELF -- the module whose transport was retired. That is not a
tuning problem, it is structural: every local rule reads its own forward weights to compute its own
error, and the shipped Kolen-Pollack rule legitimately reads a SECOND matrix `B`. Whether reading a
second matrix is transport depends on how that matrix is MAINTAINED (`B` receives the same increment
as the read-out and never reads it), which is a semantic property no static pass recovers. A gate
that reddens the fix is the shape this repo has killed proposals at twice.

THE BLIND SPOT, DISCLOSED HERE AND PRINTED IN THE GATE'S OWN OUTPUT rather than left to be
discovered. W4 keys on an explicit transpose, so it CANNOT see the spelling `mlp_q` actually used:

    w2i = self.W2[a]                                  # a row, bound to a local
    dh  = [err * w2i[j] for j in range(hidden)]        # transposed by INDEXING, not by syntax
    W1[j][k] += lr * dh[j] * z[k]

There is no `.T` anywhere in it. So of the two violations this tree has had, W4 would have caught
ONE. It is regression protection against the common spelling; it is not a proof of compliance, and
it says so on every run.

Reproduce:
    python -m snn_data_lake.gates.eprop_transport_gate            # report, exit 0
    python -m snn_data_lake.gates.eprop_transport_gate --assert   # exit 1 on a governed finding
"""

# distinct-from: cortex/eprop_kernel.py -- that is a RUNTIME NUMERIC check on ONE network
#   (`bptt_grads` + `_cosine` against `EPROP_BPTT_COSINE_FLOOR = 0.5`): it builds a net, runs it,
#   and compares an e-prop gradient against an independent BPTT one. It proves a rule it is HANDED
#   is local; it cannot see a module nobody handed it. This is a STATIC scan over every module in
#   the tree. The two are complements, not duplicates -- eprop_kernel answers "is this rule local?",
#   this answers "is there a rule nobody checked?" -- and neither substitutes for the other.
# distinct-from: gates/method_dormancy_gate.py -- same TECHNIQUE (an AST pass over src/),
#   different QUESTION: it asks whether a learning-verb METHOD has a live call site, which is
#   dormancy. It says nothing about what the method computes, so a transported weight inside a
#   method that IS called passes it clean -- exactly how mlp_q sat in run_selfchecks passing.
# distinct-from: tools/promotion_sweep.py -- its `BACKPROP_MARKERS` is the substring grep
#   POLICIES.md calls theatre, it has no `--assert`, it is in no CI job, and it scans only OUTSIDE
#   the governed tier. This gate exists because that one cannot be made to answer the question.
# (The remaining hook matches -- learn_daemon.py, self_build_loop.py, access_consciousness.py,
#  eprop_kernel's _TinyNet -- matched on self-check boilerplate phrasing, not on this job. Read.)

import ast
import os
import re
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = "snn_data_lake"

#: A name that plausibly holds a weight matrix. WIDE on purpose -- it only decides what is worth
#: following, and the update-target test below is what makes a finding.
#:
#: WIDENED 2026-08-15, after this predicate was measured MISSING the names it most needed. It read
#: `^_?[wW][a-z]?$` (at most ONE trailing lowercase letter) and excluded anything `.isupper()`, so:
#:   * `Wrec` / `Win` / `Wout` -- reasoning/eprop.py's ACTUAL weights -- were not weight-ish at all;
#:   * `W2` was rejected as a CONSTANT, because "W2".isupper() is True. `W2` is the name in
#:     `mlp_q`'s retired transport, i.e. the exact case this module's own docstring cites as its
#:     miss. The gate was blind to it twice over.
#: The constant exclusion still has to exist -- dropping it is most of what made W5 score 59 -- so
#: it is NARROWED rather than removed: a real constant is caps-and-underscores with two or more
#: trailing caps (`W_MAX`, `WEIGHT_MAX`). `W2`, `Wz`, `Win` are not. RE-COUNTED before adopting:
#: the shipped rule scores GOVERNED 0 / UNGOVERNED 1 under both predicates -- widening cost zero
#: findings and bought the names above.
_WEIGHTISH = re.compile(r"(^_?[wW]($|[\d_])|weight|^_?[wW][a-z]{0,4}$|^W_)", re.I)
_CONSTANTISH = re.compile(r"^[A-Z][A-Z_]{2,}$")  # W_MAX / WEIGHT_MAX -- not W2, Wz, Win


def _is_weight(name):
    return bool(name) and not _CONSTANTISH.match(name) and bool(_WEIGHTISH.search(name))


# distinct-from: cortex/second_order_access.py -- that module's "key" is a TELEMETRY key in a
# per-tick state() dict (its "NO phenomenal key, ever" is a P8 honesty constraint about what the
# brain may report). The "key" below is a lookup string into this gate's static EXEMPT allowlist.
# Same English word, unrelated jobs; nothing to reuse or consolidate in either direction.

#: `module.function` -> why this transport is not a violation of "e-prop, not backprop".
#:
#: OWNER-GRANTED ONLY. Narrowing what a mechanized gate asserts is an owner decision, never something
#: an adjudication may do on its own authority (`docs/POLICIES.md`, e-prop invariant block), so an
#: entry here records a decision that was TAKEN rather than a finding somebody found inconvenient.
#: The bar is the one the invariant sets for itself: it governs how the BRAIN learns, and its own
#: verification clause REQUIRES "an independent reverse-mode BPTT gradient" -- so backprop is obliged
#: to exist in this tree as an INSTRUMENT. A control is that instrument. A faculty is not, however
#: far off the live tick it sits: `mlp_q` was exactly one constructor argument from the tick and its
#: transport was RETIRED, not exempted.
#:
#: Follows `dormancy_gate.EXEMPT`'s pattern, including the half that makes it honest: every entry is
#: asserted to still match a REAL finding, so an exemption that blesses nothing fails loudly instead
#: of reading as coverage.
# EXEMPT IS RETIRED (FR-019 / specs/020 T070). Its one owner-granted entry now lives in
# `reasoning/gf.py::GRADIENT_CLAIM["MLP.train"]`, beside the code it describes, and is read via
# `resolve_disposition`. P2: exactly ONE hand-maintained record of a symbol's gradient
# disposition survives. The honesty property came WITH it -- see `stale_exemptions`, which keeps
# a reachable red rather than becoming vacuous once exemption is derived per-finding.


def _exempt_key(rel, fn):
    """The EXEMPT lookup string for a finding: module path plus the function it was found in."""
    return f"{rel}.{fn}"


class AmbiguousDisposition(Exception):
    """A finding's key matches several declared symbols that DISAGREE about the disposition.

    Refusing is the only honest answer. A finding carries the ENCLOSING FUNCTION name (`update`),
    while the registry keys the QUALIFIED symbol (`LatentQ.update`), so the two cannot always be put
    in correspondence. MEASURED on this tree: 4 modules carry one bare method name on two or more
    classes, and 2 of those disagree -- `basal_ganglia/latent_q` declares `LatentQ.update` LOCAL and
    `_TabularQ.update` CONTROL; `cortex/reasoning_curriculum` declares `MemorizationLearner`'s
    CONTROL and `ReasoningLearner`'s LOCAL. A resolver that picked one would be wrong in both, and
    silently -- which is worse than the gate refusing to answer.
    """


#: A declaration claims a TRANSPORT exemption when its reason names this gate. Mechanical, greppable,
#: and already true of the one live grant. MEASURED: exactly 1 of 157 declarations matches
#: (`reasoning/gf.py::MLP.train`), so the marker discriminates rather than selecting everything.
#:
#: This marker is what PORTS `stale_exempt`'s honesty property (FR-019a) rather than losing it in the
#: merge. Derived per-finding, "is this exempt?" would have no standalone record to go stale, and the
#: property would be VACUOUS -- green because nothing could ever be checked. An explicit claim can
#: still be shown to bless nothing.
TRANSPORT_EXEMPT_MARKER = "eprop_transport_gate"


def _module_declarations(rel):
    """`{qualified_symbol: (disposition, reason)}` for one module path like `reasoning.gf`.

    REUSES `gradient_claim_gate.read_declaration` -- the registry has exactly one reader, and this
    gate does not get a second one (P2). Imported lazily: the transport walk runs in the stdlib tier
    and must not pay for the census on every scan.
    """
    from snn_data_lake.gates import gradient_claim_gate as gcg

    root = gcg._repo_root()
    if root is None:
        return {}
    path = os.path.join(root, "src", "snn_data_lake", *rel.split("."))
    path += ".py"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    return gcg.read_declaration(tree)


def resolve_disposition(rel, fn):
    """The declared disposition for a finding, read from the ONE registry. None when undeclared.

    FR-019/T067: this replaces the local `EXEMPT` dict. The finding's `fn` is a BARE name and the
    registry key is qualified, so the match is on the final dotted component -- and it REFUSES
    (raises `AmbiguousDisposition`) when several matching symbols disagree, rather than guessing.
    Several matching symbols that AGREE are not ambiguous: the answer is the same either way, and
    refusing there would make the resolver useless on the common case.
    """
    decls = _module_declarations(rel)
    hits = {sym: v for sym, v in decls.items() if sym.split(".")[-1] == fn}
    if not hits:
        return None
    dispositions = {disp for disp, _ in hits.values()}
    if len(dispositions) > 1:
        raise AmbiguousDisposition(
            f"{rel}.{fn} matches {sorted(hits)} with disagreeing dispositions "
            f"{sorted(dispositions)} -- the finding's key cannot distinguish them"
        )
    return next(iter(hits.values()))


def claims_transport_exemption(rel, fn):
    """True when the declaration for this symbol explicitly names THIS gate as its exemption."""
    try:
        found = resolve_disposition(rel, fn)
    except AmbiguousDisposition:
        return False
    return bool(found) and TRANSPORT_EXEMPT_MARKER in found[1]


def stale_exemptions(violations, extra=None):
    """Declared transport exemptions matching NO finding in the tree (FR-019a, ported from
    `stale_exempt`).

    An exemption that blesses nothing is an allowlist row nobody can see fail. `extra` plants such a
    claim so the property has a REACHABLE RED -- without it, an implementation returning `[]`
    unconditionally would satisfy every "nothing is stale" assertion ever written against it.
    """
    from snn_data_lake.gates import gradient_claim_gate as gcg

    seen = {_exempt_key(v[0], v[1]) for v in violations}
    claimed = {}
    for v in violations:
        if claims_transport_exemption(v[0], v[1]):
            claimed[_exempt_key(v[0], v[1])] = True
    # Every module carrying the marker must still produce a finding somewhere. Walk the declared
    # registry rather than the findings, or a claim whose finding VANISHED could never be seen.
    root = gcg._repo_root()
    if root is not None:
        for relpath, tree in (gcg.walk(root)[0] or {}).items():
            decls = gcg.read_declaration(tree)
            mod = relpath[:-3].replace("/", ".")
            for sym, (_disp, reason) in decls.items():
                if TRANSPORT_EXEMPT_MARKER not in reason:
                    continue
                key = _exempt_key(mod, sym.split(".")[-1])
                claimed.setdefault(key, key in seen)
                if key not in seen:
                    claimed[key] = False
    out = sorted(k for k, ok in claimed.items() if not ok)
    if extra:
        out.extend(sorted(k for k in extra if k not in seen))
    return sorted(set(out))


def _base(node):
    """`self.w1` -> 'w1'; `W2[a]` -> 'W2'; `w` -> 'w'; anything else -> None."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _transposed(node):
    """If `node` transposes something, the name being transposed -- else None.

    Covers `.T` / `.mT`, `.t()`, `.transpose(...)`, `.permute(...)`, `.swapaxes(...)` as methods,
    and `np.transpose(x)` / `torch.transpose(x, ...)` as functions taking the array first.
    """
    if isinstance(node, ast.Attribute) and node.attr in ("T", "mT"):
        return _base(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("t", "transpose", "permute", "swapaxes"):
            owner = _base(node.func.value)
            # `np.transpose(w)` -- the owner is the module, the array is the first argument.
            if owner in ("np", "numpy", "torch") and node.args:
                return _base(node.args[0])
            return owner
    return None


def _findings_in(fn):
    """(lineno, transposed weight, updated weight) for one function body.

    A local bound to an expression containing a transposed weight is TAINTED, and taint follows
    further assignments. A finding is a tainted value -- or a transpose used inline -- reaching an
    assignment whose target is a weight ATTRIBUTE or SUBSCRIPT (`self.W1 +=`, `W1[j][k] = ...`).
    Assigning to a bare local is not an update; it is how the value travels.
    """
    tainted, out = {}, []
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        direct = [t for t in (_transposed(n) for n in ast.walk(node.value)) if _is_weight(t)]
        via = [n.id for n in ast.walk(node.value) if isinstance(n, ast.Name) and n.id in tainted]
        if not (direct or via):
            continue
        source = direct[0] if direct else tainted[via[0]]
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            name = _base(t)
            if not name:
                continue
            if isinstance(t, (ast.Attribute, ast.Subscript)) and _is_weight(name):
                out.append((node.lineno, source, name))
            elif isinstance(t, ast.Name):
                tainted[name] = source
    return out


# distinct-from: cerebellum/spiking_jepa.py -- serializes weight VALUES to JSON; this reads a
#   SOURCE TREE and never touches a live weight. distinct-from: cortex/snn_moe_classifier.py -- its
#   "gate weights" are softmax scores over experts, not a matrix. distinct-from:
#   plasticity/developmental_pruning.py -- its (class, index) pairs enumerate LIVE synapses; these
#   are AST index EXPRESSIONS as source text. distinct-from: substrate/signal_calibrator.py --
#   reads persisted calibration records. All four were read; none extracts subscript index order or
#   detects a transpose, and there is nothing to reuse in either direction.
def _index_pair(node):
    """`X[a][b]` -> ('X', 'a', 'b'), else None. `X` may be `self.X` or a bare name."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Subscript):
        return None
    inner = node.value
    base = inner.value
    name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
    if not name:
        return None
    try:
        return name, ast.unparse(inner.slice), ast.unparse(node.slice)
    except Exception:
        return None


def index_swaps_in(tree):
    """(lineno, weight, detail) for weights READ with their index order REVERSED.

    THE SECOND TRANSPOSE SPELLING. `_transposed` above keys on an array-library transpose -- `.T`,
    `.t()`, `transpose(...)`. A pure-python loop has none of those and writes the transpose as an
    index swap instead: `reasoning/eprop.py::Net.grad_bptt` reads `self.Wrec[k][j]` to carry error
    backward while the module stores and updates `self.Wrec[j][k]`. No `.T` exists anywhere in that
    file, so the shipped rule could not see it AT ALL -- not as a finding, not as a census entry.

    THE FORWARD ORDER IS THE ONE THE MODULE ASSIGNS TO, which is what makes this narrow enough to
    adopt. A weight's storage layout is whatever `self.W[j][k] -= ...` uses; a READ in the reversed
    order is then a transpose. A module that only ever COMPARES the two orders establishes no
    forward order and cannot fire -- so `cortex/neural_sampling.py`, which raises unless
    `weights[i][j] == weights[j][i]` to ENFORCE symmetric coupling, drops out by construction
    rather than by an allowlist entry. An earlier draft keyed on "both orders appear anywhere" and
    scored 2, with that validator as the false positive.

    Deliberately NOT gated on `_findings_in`'s terminal test (the transport must reach a weight
    update in the SAME function): `grad_bptt` RETURNS gradients and `apply()` performs the update,
    so that test would miss the one case this rule exists for.

    Counted before adopting: 1 finding on this tree.
    """
    writes, targets, out = {}, set(), []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                got = _index_pair(t)
                if got and _is_weight(got[0]):
                    writes.setdefault(got[0], set()).add((got[1], got[2]))
                    targets.add(id(t))
    for node in ast.walk(tree):
        got = _index_pair(node)
        if not got or id(node) in targets or not _is_weight(got[0]):
            continue
        name, i, j = got
        if i != j and (j, i) in writes.get(name, ()):
            out.append((node.lineno, name, f"writes [{j}][{i}], reads [{i}][{j}]"))
    return out


def scan(brain_modules=None):
    """(violations, census, swaps).

    A violation is (module, function, lineno, source, target, governed) -- the array-library
    transpose reaching a weight update. A swap is (module, lineno, weight, detail, governed) -- the
    SAME offence spelled by index order, from `index_swaps_in`. Both are collected in ONE walk of
    the package rather than by a second traversal.
    """
    if brain_modules is None:
        from snn_data_lake.gates.dormancy_gate import BRAIN_MODULES

        brain_modules = BRAIN_MODULES
    violations, census, swaps = [], 0, []
    for dirpath, _dirs, files in os.walk(_PKG_DIR):
        if "__pycache__" in dirpath:
            continue
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, _PKG_DIR).replace(os.sep, ".")[:-3]
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except (OSError, SyntaxError):
                continue
            census += 1
            governed = f"{_PKG}.{rel}" in brain_modules
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for lineno, src, tgt in _findings_in(node):
                    violations.append((rel, node.name, lineno, src, tgt, governed))
            for lineno, name, detail in index_swaps_in(tree):
                swaps.append((rel, lineno, name, detail, governed))
    return violations, census, swaps


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv
    violations, census, swaps = scan()
    gov_swaps = [s for s in swaps if s[4]]
    ung_swaps = [s for s in swaps if not s[4]]
    exempt = [v for v in violations if claims_transport_exemption(v[0], v[1])]
    gov = [v for v in violations if v[5] and not claims_transport_exemption(v[0], v[1])]
    ungov = [v for v in violations if not v[5] and not claims_transport_exemption(v[0], v[1])]
    stale = stale_exemptions(violations)

    print("=" * 78)
    print("E-PROP TRANSPORT GATE -- a transposed weight may not reach a weight update")
    print("=" * 78)
    print(f"scanned {census} modules under src/{_PKG}/")
    print()
    print("--- GOVERNED TIER (asserted) ---")
    for rel, fn, lineno, src, tgt, _g in gov:
        print(f"  {rel}.{fn}:{lineno}   {src} transposed -> updates {tgt}")
    if not gov:
        print("  (none)")
    print()
    print("--- OUTSIDE THE GOVERNED TIER (reported, not asserted) ---")
    for rel, fn, lineno, src, tgt, _g in ungov:
        print(f"  {rel}.{fn}:{lineno}   {src} transposed -> updates {tgt}")
    if not ungov:
        print("  (none)")
    print()
    print(
        "--- EXEMPT (owner-granted, read from GRADIENT_CLAIM; each must match a real finding) ---"
    )
    for rel, fn, lineno, src, tgt, g in exempt:
        tier = "governed" if g else "ungoverned"
        print(f"  {rel}.{fn}:{lineno}   {src} transposed -> updates {tgt}   [{tier}]")
        print(f"      {resolve_disposition(rel, fn)[1]}")
    if not exempt:
        print("  (none)")
    print()
    if stale:
        print("--- STALE EXEMPT (asserted) ---")
        for k in stale:
            print(f"  {k}   blesses NOTHING -- no finding in the tree matches this key")
        print()
    print("--- INDEX-SWAP TRANSPOSE (the loop spelling; governed half asserted) ---")
    for rel, lineno, name, detail, g in swaps:
        print(f"  {'GOVERNED  ' if g else 'ungoverned'} {rel}:{lineno}   {name}: {detail}")
    if not swaps:
        print("  (none)")
    print()
    print(
        f"GOVERNED: {len(gov)}   UNGOVERNED: {len(ungov)}   EXEMPT: {len(exempt)}   "
        f"SWAP gov/ung: {len(gov_swaps)}/{len(ung_swaps)}"
    )
    print()
    # Printed on EVERY run, not buried in the docstring: a gate that reports 0 and does not say what
    # it cannot see gets read as a proof of compliance, which this is not.
    print("  BLIND SPOT, stated so a green is not over-read. The FIRST rule keys on an EXPLICIT")
    print("  transpose, and the transport retired from basal_ganglia/mlp_q.py was spelled by")
    print("  SINGLE-ROW INDEXING -- 'w2i = W2[a]' then 'dh = err * w2i[j]' -- with no '.T' and no")
    print("  reversed index PAIR, so neither rule here catches that spelling. The wider rule that")
    print("  would was built, scored 59 findings including mlp_q's own SHIPPED fix, and was")
    print("  rejected -- see this module's docstring.")
    print()
    print("  WHAT THE SECOND RULE CLOSED, 2026-08-15: an explicit transpose is an ARRAY-LIBRARY")
    print("  spelling, and reasoning/eprop.py hand-rolls reverse-mode BPTT in pure math/random")
    print("  loops with NO numpy and NO '.T' anywhere. That file was invisible to this walk")
    print("  entirely -- not a finding, not a census entry. INDEX-SWAP now sees it. It is not a")
    print("  violation (Net.grad_bptt is the independent gradient the invariant's own cosine")
    print("  clause REQUIRES) and it is ungoverned, so it REPORTS; but a module hand-rolling")
    print("  backprop in loops no longer passes this gate silently on the day it is promoted.")
    print()
    failed = bool(gov) or bool(stale) or bool(gov_swaps)
    if gov:
        print("gate FAIL: weight transport on a governed module")
    elif gov_swaps:
        print("gate FAIL: index-swap weight transport on a governed module")
    elif stale:
        # A stale row is a failure in its own right, not a nit. An allowlist entry that matches
        # nothing reads as coverage of a risk nobody is carrying, and it is how an exemption
        # outlives the decision that justified it.
        print("gate FAIL: a declared exemption blesses nothing -- re-derive it or delete it")
    else:
        print("gate OK")
    return 1 if (do_assert and failed) else 0


# ---------------------------------------------------------------------------------------------
# Self-check. The gate lands GREEN, which is normally the shape that makes a gate untrustworthy --
# `POLICIES.md`'s rule for exactly this case is *"if a gate lands green, ship the reconstruction of
# the defect it was built for"*, as `tools/red_reachability.py` does with the pre-fix
# consciousness_probe (RETIRED at T081, 2026-09-16). So the reds below are the tree's OWN adjudicated history, reconstructed.
# ---------------------------------------------------------------------------------------------
_CASES = (
    (
        "jepa PRE-FIX -- Wz^T gp into the predictor's own gradient (closed 2026-08-04)",
        "class J:\n"
        "    def _feedback_gz(self, gp):\n"
        "        gz = gp + self.Wz.T @ gp\n"
        "        self.Wp = self.Wp - self.lr * gz\n",
        True,
    ),
    (
        "the same, spelled .t() -- an import style may not evade a rule",
        "class J:\n"
        "    def step(self, gp):\n"
        "        self.Wp = self.Wp - self.lr * (self.Wz.t() @ gp)\n",
        True,
    ),
    (
        "predictive_coding LIVE -- w1.T feeds an ACTIVATION, adjudicated NOT a violation",
        "class P:\n"
        "    def infer_step(self, s_e0, s_e1, lr):\n"
        "        g1 = self.w1.T @ s_e0\n"
        "        self.r1 = self.r1 + lr * (g1 - s_e1)\n",
        False,
    ),
    (
        "predictive_coding's weight rule -- local Hebbian outer product, no transpose in it",
        "class P:\n"
        "    def learn_step(self, lr_w):\n"
        "        self.w1 += lr_w * outer(self.cum_e0 / self.n_steps, self.r1)\n",
        False,
    ),
    (
        "a forward matmul through a transposed weight -- reads, never updates",
        "class F:\n    def forward(self, x):\n        return x @ self.W_in.T\n",
        False,
    ),
    (
        "transposing an INPUT, not a weight",
        "class F:\n    def fit(self, x):\n        self.W = self.W + self.lr * (x.T @ x)\n",
        False,
    ),
)

#: The disclosed miss, kept as a CASE rather than as prose so it is re-run rather than remembered.
#: Deliberately NOT asserted: pinning it would forbid the improvement that fixes it. It is printed.
_BLIND = (
    "mlp_q PRE-FIX -- transposed by INDEXING, no '.T' in the source",
    "class M:\n"
    "    def update(self, err, a, h):\n"
    "        w2i = self.W2[a]\n"
    "        dh = [err * w2i[j] for j in range(len(h))]\n"
    "        self.W1[0][0] += self.lr * dh[0]\n",
)


def _fires(source):
    return any(
        _findings_in(fn)
        for fn in ast.walk(ast.parse(source))
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _selfcheck():
    print("E-PROP TRANSPORT GATE self-check")
    for label, source, should_fire in _CASES:
        got = _fires(source)
        assert got == should_fire, (
            f"{label}: expected {'a finding' if should_fire else 'no finding'}, got the opposite. "
            "These cases are this tree's own adjudicated history -- if one flips, either the rule "
            "changed or an adjudication did, and both need POLICIES.md updated in the same commit"
        )
        print(f"  {'RED  ' if should_fire else 'green'}  {label}")

    blind_label, blind_source = _BLIND
    still_blind = not _fires(blind_source)
    print(f"  {'BLIND' if still_blind else 'FIXED'}  {blind_label}")
    if not still_blind:
        print("    ^ this now FIRES. The docstring and main()'s printed disclosure both say it")
        print("      does NOT -- update them in this commit, then delete this branch.")

    # INDEX-SWAP, proven both ways on constructed sources -- the rule ships with the reconstruction
    # of the defect it was built for, and with the false positive an earlier draft produced.
    _swap_red = (
        "class N:\n"
        "    def apply(self, d, lr):\n"
        "        self.Wrec[j][k] -= lr * d[j][k]\n"
        "    def back(self, dv):\n"
        "        s = self.Wrec[k][j] * dv[k]\n"
        "        return s\n"
    )
    _swap_green = (  # cortex/neural_sampling.py's symmetry VALIDATOR, reconstructed
        "def check(weights, n):\n"
        "    for i in range(n):\n"
        "        for j in range(i + 1, n):\n"
        "            if weights[i][j] != weights[j][i]:\n"
        "                raise ValueError('weights must be symmetric')\n"
    )
    assert index_swaps_in(ast.parse(_swap_red)), (
        "the INDEX-SWAP red did not fire: a weight written [j][k] and read [k][j] is the loop "
        "spelling of a transpose, and it is the whole reason this rule exists"
    )
    print("  RED    a weight written [j][k] and read [k][j] -- the loop-spelled transpose")
    assert not index_swaps_in(ast.parse(_swap_green)), (
        "the INDEX-SWAP rule fired on a SYMMETRY VALIDATOR. Comparing both orders to ENFORCE "
        "W[i][j] == W[j][i] is a modelling requirement, not transport; an earlier draft keyed on "
        "'both orders appear' and scored this as a false positive"
    )
    print("  green  a symmetry validator comparing both orders -- enforces, never transports")

    violations, census, _swaps = scan()

    # The EXEMPT allowlist, proven BOTH ways. An exemption nobody can see fail is the shape this
    # gate exists to refuse, so its red is constructed here rather than asserted into existence.
    assert not stale_exemptions(violations), (
        f"declared exemptions blessing nothing: {stale_exemptions(violations)} -- each must match a "
        "real finding, or it is coverage of a risk nobody is carrying"
    )
    n_claimed = sum(1 for v in violations if claims_transport_exemption(v[0], v[1]))
    print(
        f"  green  registry exemptions: {n_claimed} entry(ies), every one still matching a "
        "real finding"
    )

    try:
        planted = stale_exemptions(
            violations, extra={"nonexistent.module.function": "a row that blesses nothing"}
        )
        assert planted == ["nonexistent.module.function"], (
            f"the STALE-EXEMPT red did not fire: expected the planted key, got {planted}. An "
            "allowlist whose staleness check cannot fire is an allowlist that never expires"
        )
    finally:
        pass
    print("  RED    a planted exemption matching no finding -- staleness check fires")

    gov = [v for v in violations if v[5] and not claims_transport_exemption(v[0], v[1])]
    assert not gov, f"governed weight-transport findings: {gov}"
    print(f"self-check OK: {len(_CASES)} constructed cases, {census} modules scanned, 0 governed")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    rc = main()
    _selfcheck()
    raise SystemExit(rc)
