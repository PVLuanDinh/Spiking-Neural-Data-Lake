"""
method_dormancy_gate -- Policy 3, per-METHOD. The amendment's mechanization, finally built.

WHY THE MODULE GATE IS NOT ENOUGH. substrate/dormancy_gate.py keys on the module basename, so a module
passes P3 clean with half its capability dead. The worked case: cortex/predictive_coding.py settled its
3-level Rao-Ballard net on every wake tick with the error-neuron architecture implemented faithfully,
while PredictiveCodingNetwork.learn() had exactly ONE call site in the tree -- inside the module's own
__main__ self-check. Inference wired, learning never called, weights frozen at their random seed for the
life of the feature, and the module-level gate saw a live, imported, fully-compliant module. Fixed in
v4.71; the blind spot it proves is not.

THE RULE. If a module on a live path computes a prediction error, a free energy, or any other learning
signal, the update rule that consumes it must also be called from a live path.

THREE DETAILS ARE LOAD-BEARING, and each was found by writing the naive version first and watching it
fail. This is the "implement the gate and count its output before adopting the rule" discipline applied
to the gate itself:

  1. IMPORT-SCOPED RESOLUTION. Matching bare attribute-call names across all reachable modules scores 0
     failures / 47 UNCERTAIN -- the name `learn` resolves in 20 unrelated modules, so the naive pass
     cannot even catch its own motivating case. A call site counts only if its module imports the
     defining module.
  2. SAME-MODULE CALLERS COUNT, EXCEPT UNDER `if __name__ == "__main__"`. Excluding the defining module
     produces a real false positive: plasticity/meta_plasticity.py drives its own private _Unit.learn
     from MetaPlasticLearner.run(). The __main__ guard is precisely what isolates the genuine case.
  3. A FUNCTION USED AS A VALUE IS USED. reasoning/arc_mint.py registers its fitters in a dispatch table
     -- `("affine", 3, _fit_affine)` -- so a pass that inspects only ast.Call nodes reported all five as
     dead. That is 5 false positives out of 12 on the first run. Bare Name/Attribute references count.

HONESTY GUARDS, inherited from P3 and not negotiable:
  (a) A call site proves a CALLER, not a CHARGE. Adding a dead consumer to silence this gate is the
      failure P3 already forbids; charge is proven separately by the runtime neuter-A/B.
  (b) An unresolved name is UNCERTAIN and hand-adjudicated, NEVER auto-dormant. This gate therefore
      reports three buckets and only ASSERTS on the first.
  (c) EXEMPT_METHOD entries carry a disposition and a reason, and `wire` entries that outlive their
      wave are reported as a standing TODO -- so exempting everything stays visible rather than silent.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.method_dormancy_gate
"""

import ast
import os

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Method-name fragments that mark an UPDATE RULE -- the thing that consumes a learning signal. Kept
# deliberately broad: a false positive lands in a bucket a human reads, a false negative is invisible.
LEARN_VERBS = (
    "learn",
    "update",
    "train",
    "adapt",
    "consolidate",
    "reinforce",
    "fit",
    "plast",
    "potentiate",
    "depress",
    "calibrate",
)

# name -> (disposition, reason). disposition in {"permanent", "wire", "retire"}.
# "permanent" = legitimately not on the live tick. "wire"/"retire" are open work and are REPORTED.
EXEMPT_METHOD = {
    "train_image": ("permanent", "deep_stdp_2stage: an offline [train]-tier trainer entrypoint"),
    "train_model": ("permanent", "training/: offline GPU trainer entrypoints (snn-train CLI)"),
    "train_with_offload": ("permanent", "training/offload: offline GPU trainer entrypoint"),
    "learned_map": (
        "permanent",
        "drive_reduction: a READOUT of the learned state->action map, not an update rule -- the verb "
        "match fires on 'learn' inside an accessor name. Kept as a named false positive rather than "
        "narrowed away, because narrowing the heuristic to dodge it would also hide real accessors "
        "that SHOULD be consumed",
    ),
    "consolidate": (
        "wire",
        "AffectiveEpisodicStore.consolidate (hippocampus/affective_memory.py:173) -- amygdala-weighted "
        "replay sampling whose only callers are inside its own main(). The module IS live, but its one "
        "cross-module importer (brainstem/autonomous_loop.py:41) calls .store() and never this. Open "
        "P3 work: wire it into the sleep replay and prove the charge with a neuter-A/B, or retire it. "
        "Held as an OPEN exemption so this gate reports it as a standing TODO instead of hiding it. "
        "ADJUDICATED 2026-07-31 (runtime, adversarially reviewed): still dead, and the disposition "
        "should become RETIRE rather than wire -- the promised wiring is not a call site but a "
        "three-file redesign of Episode + ReplayBuffer, and the prioritised-replay capability it was "
        "built for has no consumer today. Left as 'wire' pending the P2 port-before-retire step",
    ),
    # QUALIFIED `module.method` key, added 2026-07-31 once the lookup learned to prefer them.
    "cortex.predictive_coding.learn": (
        "permanent",
        "PredictiveCodingNetwork.learn (cortex/predictive_coding.py:268) -- the OFFLINE BATCH WRAPPER "
        "over learn_step(), which IS called every wake tick at cortex/live_brain.py:185. Verified by "
        "runtime instrumentation to be never called on a live path, and that is CORRECT: wiring the "
        "batch form would run the same rule twice. P2 already holds -- one rule, one definition. It is "
        "kept rather than retired because predictive_coding.py:353's self-check needs it for the "
        "learning-curve assert, i.e. the module's own proof that the rule learns. Safe to qualify: "
        "`learn` is the only def by that name in this module (learn_step is a distinct name)",
    ),
}

# STILL NOT EXEMPTIBLE: latent_q._TabularQ.update, and the reason is a REMAINING granularity limit.
# It is the self-check's tabular control, constructed only inside main(); a control class that ran on a
# live path would be the bug, so it is dormant AND correct to be dormant. But `basal_ganglia/latent_q.py`
# defines `update` TWICE -- LatentQ.update at :102 and _TabularQ.update at :165 -- so even the qualified
# `basal_ganglia.latent_q.update` key would exempt BOTH, and LatentQ.update is GENUINELY dormant open P3
# work (module imported at autonomous_loop.py:30, every method cold). Exempting it to excuse the control
# would hide the real finding, which is the failure this gate exists to prevent.
#
# Module qualification was enough for predictive_coding (one `learn` def); it is not enough here. The
# key needs to reach module.Class.method, which means `_parse_all` tracking the enclosing ClassDef.
# Until then _TabularQ stays UNCERTAIN with its disposition recorded in docs/coverage/README.md.
#
# Related defect, same root: `defs` is keyed (module, name), so two same-named defs in one module
# collapse to ONE entry and the surviving lineno names whichever came last. That is why this finding was
# reported at latent_q.py:165 (_TabularQ) when the class it is about is LatentQ at :102.


def _modname(path):
    return os.path.relpath(path, _PKG_DIR)[:-3].replace(os.sep, ".")


def _parse_all():
    trees = {}
    for dp, _dn, fn in os.walk(_PKG_DIR):
        for f in fn:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            p = os.path.join(dp, f)
            try:
                trees[_modname(p)] = ast.parse(open(p, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
    return trees


def _imports(tree):
    """The in-package modules this file imports, as dotted paths relative to snn_data_lake."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "snn_data_lake" in n.module:
            out.add(n.module.split("snn_data_lake.")[-1])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if "snn_data_lake" in a.name:
                    out.add(a.name.split("snn_data_lake.")[-1])
    return out


def _main_guard_lines(tree):
    """Line numbers that belong to the module's own self-check. Detail 2: these do NOT count as callers.

    DETAIL 2b, and the first version of this gate got it wrong: the `if __name__ == "__main__":` block
    is almost always one line -- `main()`. The self-check's real body lives in a top-level `def main()`
    that the guard calls. Marking only the literal `if` block therefore counts every assertion inside
    main() as a live caller, which is exactly how the first run reported 0 dormant while
    hippocampus/affective_memory.py:173 consolidate() -- whose only callers are at :242/:252/:253 inside
    its own main() -- sat there dead. So: find the functions the guard calls, and mark THEIR bodies too.
    """
    bad, called = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            t = ast.dump(n.test)
            if "__name__" in t and "__main__" in t:
                for c in ast.walk(n):
                    if hasattr(c, "lineno"):
                        bad.add(c.lineno)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                        called.add(c.func.id)
    # one level of indirection is enough in this tree: the guard calls main(), main() holds the asserts
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in called:
            for c in ast.walk(n):
                if hasattr(c, "lineno"):
                    bad.add(c.lineno)
    return bad


def audit():
    """-> {"dormant": [...], "uncertain": [...], "open_exempt": [...], "n_defs": int}.

    dormant   = zero uses of the name anywhere in an importing (or the defining) module, outside __main__.
    uncertain = uses exist, but only as a bare same-named attribute in an importing module, so the
                RECEIVER is unproven. Hand-adjudicated per honesty guard (b) -- never auto-dormant.
    """
    trees = _parse_all()
    imports = {m: _imports(t) for m, t in trees.items()}

    # SCOPE, and getting this wrong is how the gate swings from useless to unadoptable. Run 1 marked
    # only the literal `if __name__` block and found 0. Run 2 marked main()'s body too and found 59 --
    # the 58-entry rubber-stamp shape POLICIES.md:24-26 records as fatal to an earlier proposal. The
    # 59 were mostly module-level helpers inside research/train-tier files, which the MODULE gate
    # already scopes out. So this gate takes the amendment at its word -- "a learning-verb METHOD on a
    # REACHABLE module" -- and applies both halves:
    #   (1) the defining module must be reachable from a live root (same set dormancy_gate asserts on)
    #   (2) the def must be a METHOD (inside a class), not a module-level self-check helper
    from snn_data_lake.substrate import dormancy_gate

    live = {m.split("snn_data_lake.")[-1] for m in dormancy_gate.reachable()}

    defs = {}
    for m, t in trees.items():
        if m not in live:
            continue
        for cls in ast.walk(t):
            if not isinstance(cls, ast.ClassDef):
                continue
            for n in cls.body:
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if n.name.startswith("__"):
                    continue
                if any(v in n.name.lower() for v in LEARN_VERBS):
                    defs[(m, n.name)] = n.lineno

    # every use of a name, call or value-reference (detail 3), tagged with whether it is under __main__
    uses = {}
    for m, t in trees.items():
        guard = _main_guard_lines(t)
        defsites = {
            n.lineno for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for n in ast.walk(t):
            nm = (
                n.attr
                if isinstance(n, ast.Attribute)
                else (n.id if isinstance(n, ast.Name) else None)
            )
            if nm is None or getattr(n, "lineno", None) in defsites:
                continue
            uses.setdefault((m, nm), []).append(n.lineno in guard)

    dormant, uncertain = [], []
    for (dm, name), ln in sorted(defs.items()):
        same_module = any(not g for g in uses.get((dm, name), []))
        importers = [
            cm
            for (cm, cn), flags in uses.items()
            if cn == name and cm != dm and dm in imports.get(cm, ()) and any(not f for f in flags)
        ]
        entry = f"{dm}.py:{ln} {name}()"
        if same_module or importers:
            # A cross-module hit is a NAME match, not a proven receiver -- honesty guard (b).
            if not same_module and importers:
                uncertain.append(f"{entry}  [name-matched in: {', '.join(sorted(set(importers)))}]")
            continue
        # Qualified `module.method` FIRST, bare `method` second. The bare form exempts a name across all
        # 194 modules, which is far too blunt for a real disposition: exempting `learn` to excuse ONE
        # offline wrapper would also excuse the next PredictiveCodingNetwork.learn -- the exact case this
        # gate was built for, and itself a method named `learn`. Bare keys stay honoured so the existing
        # entries keep working; new dispositions should be qualified.
        if f"{dm}.{name}" in EXEMPT_METHOD or name in EXEMPT_METHOD:
            continue
        dormant.append(entry)

    open_exempt = [
        f"{k} ({v[0]}: {v[1]})" for k, v in sorted(EXEMPT_METHOD.items()) if v[0] != "permanent"
    ]
    return {
        "dormant": dormant,
        "uncertain": uncertain,
        "open_exempt": open_exempt,
        "n_defs": len(defs),
    }


def main():
    a = audit()
    print("=" * 78)
    print("METHOD DORMANCY GATE -- Policy 3, per-METHOD amendment")
    print("=" * 78)
    print(f"update-rule definitions scanned : {a['n_defs']}")
    print(f"EXEMPT_METHOD entries           : {len(EXEMPT_METHOD)}")
    print(f"dormant (asserted)              : {len(a['dormant'])}")
    print(f"uncertain (hand-adjudicated)    : {len(a['uncertain'])}")
    for e in a["dormant"]:
        print(f"  DORMANT   {e}")
    for e in a["uncertain"]:
        print(f"  UNCERTAIN {e}")
    if a["open_exempt"]:
        print("  STANDING TODO -- non-permanent exemptions still open:")
        for e in a["open_exempt"]:
            print(f"    {e}")

    assert not a["dormant"], (
        "update rules with no live-path call site: "
        + "; ".join(a["dormant"])
        + " -- wire them and prove the charge, retire them, or add an EXEMPT_METHOD entry with a reason"
    )
    print(
        f"self-check OK: {a['n_defs']} update rules, none dormant, "
        f"{len(a['uncertain'])} UNCERTAIN reported not asserted"
    )


if __name__ == "__main__":
    main()
