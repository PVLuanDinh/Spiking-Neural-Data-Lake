"""Policy-4 replication guard -- there is only ONE brain; no build may re-implement the brain stack.

Policy 4 (user, 2026-07-24): "there is only 1 brain; a build must never replicate brain functionality
-- utilize the original, not re-implement it, because it causes bloat." The canonical brain is a clean
3-layer COMPOSITION (verified): cortex.brain.Brain (cognition + circadian clock) composes
cortex.live_brain.LiveBrain (perception + workspace) composes brainstem.autonomous_loop.AutonomousArm
(action + reward + arm + SAFETY). Composition is fine; RE-DEFINING one of those stack classes a second
time (a parallel "brain") is the bloat.

This gate MECHANIZES "one brain" the same way dormancy_gate mechanizes "no dormancy": it AST-scans
snn_data_lake for definitions of the canonical stack classes and ASSERTS each is defined exactly once,
in its canonical module -- unless the duplicate is on the PENDING_RETIREMENT allowlist (a known replica
scheduled for a Policy-4 consolidation wave, with a reason). A NEW second `class Brain` / `class
LiveBrain` / `class AutonomousArm` that is not allowlisted FAILS run_selfchecks -- so a replica cannot
be introduced silently. The allowlist shrinks to empty as the consolidation waves retire the replicas.

The companion to this guard is cortex.brain.make_brain(): the single sanctioned CONSTRUCTION path.
Consumers obtain the brain through make_brain() rather than calling Brain(...) directly.

  python -m snn_data_lake.substrate.replication_guard   # (also run by run_selfchecks.py)
"""

import ast
import os

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake

# the canonical single-definition stack classes -> the ONE module each MUST be defined in
CANON = {
    "Brain": f"{_PKG}.cortex.brain",
    "LiveBrain": f"{_PKG}.cortex.live_brain",
    "AutonomousArm": f"{_PKG}.brainstem.autonomous_loop",
}

# known DUPLICATE definitions still on disk, pending retirement in a Policy-4 consolidation wave.
# key = (class_name, module_path); value = reason. Empty this out as the replicas are retired.
PENDING_RETIREMENT = {
    ("Brain", f"{_PKG}.reasoning.brain_loop"): (
        "2nd Brain class -- a parallel stack around LakeChat + its own GlobalWorkspace + learn-loop. "
        "Now INERT: its only front-end (reasoning/brain_web.py) was retired in v4.53, so it has ZERO "
        "importers anywhere. Its reasoning layer IS already covered canonically -- deduction by "
        "reasoning/derivation.py (a verbatim port of _relation_graph/_derive), abduction by "
        "pc_abduction.AbductionExpert via LakeChat._abduce_reply, and induction by schema_induction, "
        "which keeps the same refuse-to-generalise-on-thin-evidence guard (min_support). "
        "BLOCKER on deletion: self_rewrite() -- user-authorized GATED self-modification -- exists ONLY "
        "here; grep finds no canonical equivalent (Brain.propose_improvements is proposer-only). Port "
        "self_rewrite into the canonical Brain and verify it, THEN delete. Do not drop it silently."
    ),
}


def _mpath(pyfile):
    rel = os.path.relpath(pyfile, os.path.dirname(_PKG_DIR)).replace("\\", "/")[:-3]
    return rel.replace("/", ".")


def _class_sites():
    """All definition sites of the canonical stack class NAMES: {name: [module_path, ...]}."""
    sites = {name: [] for name in CANON}
    for root, _d, files in os.walk(_PKG_DIR):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            mod = _mpath(p)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in CANON:
                    sites[node.name].append(mod)
    return sites


def audit():
    sites = _class_sites()
    replicas, missing = [], []
    for name, canon_mod in CANON.items():
        found = sites[name]
        if canon_mod not in found:
            missing.append((name, canon_mod))
        for mod in found:
            if mod == canon_mod:
                continue
            if (name, mod) not in PENDING_RETIREMENT:
                replicas.append((name, mod))  # an UNSANCTIONED second definition
    return {"sites": sites, "replicas": replicas, "missing": missing}


def main():
    a = audit()
    print("=" * 66)
    print("REPLICATION GUARD -- Policy 4: there is only ONE brain (no re-implementation)")
    print("=" * 66)
    for name, canon in CANON.items():
        dups = [m for m in a["sites"][name] if m != canon]
        tag = f"  DUPLICATES: {dups}" if dups else ""
        print(f"  {name:<14} defined in {len(a['sites'][name])} site(s){tag}")
    if PENDING_RETIREMENT:
        print("--- allowlisted replicas pending retirement ---")
        for (name, mod), reason in PENDING_RETIREMENT.items():
            print(f"  {name} @ {mod.replace(_PKG + '.', '')}: {reason[:60]}...")
    print("=" * 66)
    assert not a["missing"], f"canonical stack class not found where expected: {a['missing']}"
    assert not a["replicas"], (
        f"Policy 4: unsanctioned brain-stack replica(s) -- a second definition of a canonical class "
        f"that is not on PENDING_RETIREMENT: {a['replicas']}. Utilize the one brain (cortex.brain."
        f"make_brain / the canonical nesting), do not re-implement it."
    )
    print(
        "replication guard OK: one canonical Brain/LiveBrain/AutonomousArm (+ allowlisted replicas)"
    )


if __name__ == "__main__":
    main()
