"""Disclosed teaching-policy diagnosis, not a canonical learning repair.

Correct responses receive no reward/deposit in corrective mode. They are still
presented, read and counted in this external diagnostic journal. Wrong responses
use unchanged canonical Reader.teach/learn_turn and the full enabled rule stack.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from snn_data_lake.instruments import taught_problems as tp, inseparable_mapping as im
from snn_data_lake.instruments.taught_campaign import Reader


ROOT = Path(__file__).resolve().parent
SEED = 9900


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("always_positive", "corrective_only")); args = ap.parse_args()
    started = time.monotonic(); out = ROOT / "teaching-policy-probes"; out.mkdir(exist_ok=True)
    p = tp.PROBLEMS["aii"]; learner = tp.canonical_incumbent(SEED); reader = Reader(learner, history=False)
    validation = list(p.episodes(SEED, "validation"))
    texts = [reader.text(e) for e in validation]; targets = [e[-1].label for e in validation]
    recipe = {"scope": "exposed engineering teaching-policy diagnostic, not canonical repair/promotion or efficacy claim",
              "mode": args.mode, "seed": SEED, "problem": "aii", "max_epochs": 1200, "stable_epochs": 10,
              "initialization": "canonical working-source PRE baseline fix, all faculties enabled",
              "source_modules": {}, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "corrective_action_if_correct": "count/log external presentation, no learn_turn or learned/cache/replay mutation",
              "corrective_action_if_wrong": "unchanged actual Reader.teach / canonical learn_turn(label)",
              "validation": "pure route_text, no target input or update"}
    for module in (tp,):
        f = Path(module.__file__); recipe["source_modules"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    for name in ("readout/interaction_learning.py", "plasticity/plasticity_rule_stack.py", "plasticity/homeostatic_plasticity_stack.py"):
        f = Path(tp.__file__).parents[1] / name; recipe["source_modules"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / f"{args.mode}-recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    history = []; presentations = []; checkpoints = []; streak = 0; teaching_calls = 0
    for epoch, episodes in enumerate(p.teach_epochs(SEED, 1200), 1):
        for episode in episodes:
            text = reader.text(episode); target = episode[-1].label
            before = json.dumps(learner._snapshot(), sort_keys=True) if not presentations else None
            prediction = learner.route_text(text)
            if before is not None: assert json.dumps(learner._snapshot(), sort_keys=True) == before
            teach = args.mode == "always_positive" or prediction != target
            if teach:
                reader.teach(episode, target); teaching_calls += 1
            presentations.append({"epoch": epoch, "text": text, "label": target, "pre_prediction": prediction, "teaching_call": teach})
        before = json.dumps(learner._snapshot(), sort_keys=True) if epoch in (1, 100, 400, 800, 1200) else None
        predictions = [learner.route_text(t) for t in texts]
        if before is not None: assert json.dumps(learner._snapshot(), sort_keys=True) == before
        correct = sum(x == y for x, y in zip(predictions, targets)); streak = streak + 1 if correct == len(targets) else 0
        history.append({"epoch": epoch, "correct": correct, "n": len(targets), "streak": streak, "teaching_calls": teaching_calls})
        if epoch in (1, 100, 400, 800, 1200) or streak == 10:
            checkpoints.append({"epoch": epoch, "predictions": predictions, "drives": {t: [learner._drive(n, im.ranks(t)) for n in learner.neurons] for t in set(texts)},
                                "homeo_targets": list(learner.stack._homeo._target), "snapshot": learner._snapshot()})
        if streak == 10: break
    before_clear = [learner.route_text(t) for t in texts]
    snapshot = learner._snapshot(); learner._authority.clear()
    assert [learner.route_text(t) for t in texts] == before_clear
    learner._restore_snapshot(snapshot)
    assert [learner.route_text(t) for t in texts] == before_clear
    result = {"recipe": recipe, "stable_solved": streak == 10, "epochs": epoch, "presentations": len(presentations),
              "teaching_calls": teaching_calls, "best_correct": max(r["correct"] for r in history),
              "max_streak": max(r["streak"] for r in history), "final_correct": correct, "validation_n": len(targets),
              "authority_clear_preserves_predictions": True, "snapshot_restore_preserves_predictions": True,
              "wall_seconds": time.monotonic() - started, "history": history, "checkpoints": checkpoints,
              "presentation_log": presentations}
    (out / f"{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "checkpoints", "presentation_log")}))


if __name__ == "__main__": main()
