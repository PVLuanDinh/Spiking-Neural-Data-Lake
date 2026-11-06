"""Single exposed-seed engineering controls; not Engine validation or benefit data.

Delegates every teaching/read update to canonical_incumbent + Reader. Explicit
initialization controls change no learned weight and use no labels. Disabled-rule
controls are diagnosis only and cannot satisfy the full-stack success contract.
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
MODES = ("baseline", "bf_pre", "homeo_pre", "both_pre", "common_pre",
         "bf_off", "homeo_off", "ei_off")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=MODES); args = ap.parse_args()
    started = time.monotonic(); out = ROOT / "initialization-probes"; out.mkdir(exist_ok=True)
    p = tp.PROBLEMS["aii"]; learner = tp.canonical_incumbent(SEED); reader = Reader(learner, history=False)
    construction = list(next(iter(p.teach_epochs(SEED, 1))))
    texts = [reader.text(e) for e in construction]
    codes = [im.ranks(t) for t in texts]
    validation = list(p.episodes(SEED, "validation"))
    valtexts = [reader.text(e) for e in validation]; targets = [e[-1].label for e in validation]
    initial_weights = learner.weights(); initial_drives = [[learner._drive(n, c) for n in learner.neurons] for c in codes]
    stack = learner.stack; homeo = stack._homeo; bf = stack._cons
    config = {"seed": SEED, "problem": "aii", "max_epochs": 1200, "stable_epochs": 10,
              "faculties_disabled": [], "source": "actual canonical_incumbent and Reader",
              "init_source": "registered first teach epoch codes; no labels consumed by initialization"}
    if args.mode in ("bf_pre", "both_pre"):
        for i, weights in enumerate(initial_weights): bf._ensure(i, list(weights))
        config["bf_preinitialization"] = "all rows at initial weights before any teaching"
    if args.mode in ("homeo_pre", "both_pre"):
        homeo._target = [max(homeo.ceiling_factor * abs(y), 1e-3) for y in initial_drives[0]]
        config["homeo_preinitialization"] = "each row factor times abs(initial drive on first teach code)"
    if args.mode == "common_pre":
        ceiling = max(homeo.ceiling_factor * max(abs(y) for row in initial_drives for y in row), 1e-3)
        homeo._target = [ceiling] * len(learner.responses)
        config["common_preinitialization"] = "factor times max abs(initial drive), all rows/all first-epoch codes"
    if args.mode == "bf_off": stack._cons = None; config["faculties_disabled"] = ["Benna-Fusi"]
    if args.mode == "ei_off": stack._ei = None; config["faculties_disabled"] = ["E/I"]
    if args.mode == "homeo_off":
        stack._post = [x for x in stack._post if x is not homeo]; stack._homeo = None
        config["faculties_disabled"] = ["homeostatic scaling"]
    assert learner.weights() == initial_weights
    config["initial_homeo_targets"] = None if stack._homeo is None else list(stack._homeo._target)

    def measure(epoch):
        predictions = [learner.route_text(t) for t in valtexts]
        drives = [[learner._drive(n, c) for n in learner.neurons] for c in codes]
        margins = []
        for d, episode in zip(drives, construction):
            k = learner.responses.index(episode[-1].label)
            margins.append(d[k] - max(y for j, y in enumerate(d) if j != k))
        state = {"epoch": epoch, "correct": sum(x == y for x, y in zip(predictions, targets)),
                 "n": len(targets), "drives": drives, "margins": margins,
                 "homeo_targets": None if stack._homeo is None else list(stack._homeo._target),
                 "predictions": predictions, "responses": learner.responses}
        if stack._cons is not None:
            state["bf_first_synapse_levels"] = [None if row is None else list(row[0].u) for row in stack._cons._cas]
        return state

    initial = measure(0); history = []; checkpoints = []; streak = 0
    for epoch, episodes in enumerate(p.teach_epochs(SEED, 1200), 1):
        for episode in episodes: reader.teach(episode, episode[-1].label)
        row = measure(epoch); streak = streak + 1 if row["correct"] == row["n"] else 0
        history.append({"epoch": epoch, "correct": row["correct"], "streak": streak, "margins": row["margins"]})
        if epoch in (1, 100, 400, 800, 1200) or streak == 10: checkpoints.append(row)
        if streak == 10: break
    final = measure(epoch)
    result = {"scope": "single exposed-seed diagnostic, no Engine/retention/statistical claim", "mode": args.mode,
              "config": config, "stable_solved": streak == 10, "epochs": epoch, "teach_presentations": epoch * 4,
              "best_correct": max(r["correct"] for r in history), "max_streak": max(r["streak"] for r in history),
              "initial": initial, "checkpoints": checkpoints, "final": final, "history": history,
              "wall_seconds": time.monotonic() - started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / f"{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("mode", "stable_solved", "epochs", "best_correct", "max_streak", "wall_seconds")}))


if __name__ == "__main__": main()
