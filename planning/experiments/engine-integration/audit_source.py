"""Final static declarations and exact numerical AST comparison with acceptance pins."""
import ast
import hashlib
import json
from pathlib import Path

from snn_data_lake.gates import gradient_claim_gate as gate

root = Path(__file__).resolve().parent
repo = root.parents[2] / "repo"
pin = json.loads((root / "cycle-04-grow-controller.json").read_text())


def numerical(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GRADIENT_CLAIM" for t in n.targets
        )
    )]
    return ast.dump(tree, include_attributes=False)


rows = []
for rel, oldsha in pin["source_sha256"].items():
    before = (root / "cycle-04-grow-source" / rel).read_text()
    after = (repo / rel).read_text()
    assert numerical(before) == numerical(after), rel
    tree = ast.parse(after)
    declared = gate.read_declaration(tree)
    missing = sorted(set(gate.enumerate_rules(tree, declared)) - set(declared))
    phantom = sorted(set(declared) - gate._symbol_surface(tree))
    assert not missing and not phantom, (rel, missing, phantom)
    rows.append({"path": rel, "acceptance_sha256": oldsha,
                 "final_sha256": hashlib.sha256(after.encode()).hexdigest(),
                 "numerical_ast_identical_excluding_claim_declaration": True,
                 "missing_claims": missing, "phantom_claims": phantom})
report = {"scope": "targeted 8-module audit, not full operator battery",
          "passed": True, "modules": rows}
(root / "final-source-equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
print("8/8 numerical ASTs identical; no missing or phantom declarations")
