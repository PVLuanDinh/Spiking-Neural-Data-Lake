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
    python -m snn_data_lake.substrate.eprop_transport_gate            # report, exit 0
    python -m snn_data_lake.substrate.eprop_transport_gate --assert   # exit 1 on a governed finding
"""

# distinct-from: cortex/eprop_kernel.py -- that is a RUNTIME NUMERIC check on ONE network
#   (`bptt_grads` + `_cosine` against `EPROP_BPTT_COSINE_FLOOR = 0.5`): it builds a net, runs it,
#   and compares an e-prop gradient against an independent BPTT one. It proves a rule it is HANDED
#   is local; it cannot see a module nobody handed it. This is a STATIC scan over every module in
#   the tree. The two are complements, not duplicates -- eprop_kernel answers "is this rule local?",
#   this answers "is there a rule nobody checked?" -- and neither substitutes for the other.
# distinct-from: substrate/method_dormancy_gate.py -- same TECHNIQUE (an AST pass over src/),
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
#: following, and the update-target test below is what makes a finding. Constants (ALL-CAPS) are
#: excluded: `W_MAX` is a bound, not a weight, and treating it as one is most of what made W5 score
#: 59 instead of 0.
_WEIGHTISH = re.compile(r"(^_?[wW]($|[\d_])|weight|^_?[wW][a-z]?$|^W_)")


def _is_weight(name):
    return bool(name) and not name.isupper() and bool(_WEIGHTISH.search(name))


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


def scan(brain_modules=None):
    """(violations, census). A violation is (module, function, lineno, source, target, governed)."""
    if brain_modules is None:
        from snn_data_lake.substrate.dormancy_gate import BRAIN_MODULES

        brain_modules = BRAIN_MODULES
    violations, census = [], 0
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
    return violations, census


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv
    violations, census = scan()
    gov = [v for v in violations if v[5]]
    ungov = [v for v in violations if not v[5]]

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
    print(f"GOVERNED: {len(gov)}   UNGOVERNED: {len(ungov)}")
    print()
    # Printed on EVERY run, not buried in the docstring: a gate that reports 0 and does not say what
    # it cannot see gets read as a proof of compliance, which this is not.
    print("  BLIND SPOT, stated so a green is not over-read: this keys on an EXPLICIT transpose.")
    print("  The transport retired from basal_ganglia/mlp_q.py was spelled by INDEXING --")
    print("  'w2i = W2[a]' then 'dh = err * w2i[j]' -- with no '.T' anywhere, and this gate would")
    print("  NOT have caught it. Of the two violations this tree has had, it catches one. The")
    print("  wider rule that catches both was built, scored 59 findings including mlp_q's own")
    print("  SHIPPED fix, and was rejected -- see this module's docstring.")
    print()
    failed = bool(gov)
    print("gate FAIL: weight transport on a governed module" if failed else "gate OK")
    return 1 if (do_assert and failed) else 0


# ---------------------------------------------------------------------------------------------
# Self-check. The gate lands GREEN, which is normally the shape that makes a gate untrustworthy --
# `POLICIES.md`'s rule for exactly this case is *"if a gate lands green, ship the reconstruction of
# the defect it was built for"*, as `tools/red_reachability.py` does with the pre-fix
# consciousness_probe. So the reds below are the tree's OWN adjudicated history, reconstructed.
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

    violations, census = scan()
    gov = [v for v in violations if v[5]]
    assert not gov, f"governed weight-transport findings: {gov}"
    print(f"self-check OK: {len(_CASES)} constructed cases, {census} modules scanned, 0 governed")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    rc = main()
    _selfcheck()
    raise SystemExit(rc)
