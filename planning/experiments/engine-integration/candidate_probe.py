"""Engineering source-feasibility probe, no Brain/lake and no scientific superiority claim.

Recipes were frozen before this probe's outcomes. Uses incumbent input encoding,
candidate eligibility update, residual contribution score and actual chat R-STDP port.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from snn_data_lake.instruments import taught_problems as tp
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments.taught_campaign import Reader
from snn_data_lake.growth.candidate_lifecycle import contribution
from snn_data_lake.growth.grown_code import GrownUnit, segment_psp, grown_version
from snn_data_lake.growth.growth_loop import trace_train, label_probabilities
from snn_data_lake.readout.posed_problem import unit_rows, chat_grown_drives, units_from_rows

SEED = 9900
RECIPES = {
    "bare": (0, 0),
    "random9": (9, 0),
    "trace9": (9, 20),
    "random27": (27, 0),
    "trace27": (27, 20),
}
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=RECIPES)
    args = parser.parse_args()
    width, trace_epochs = RECIPES[args.recipe]
    start = time.monotonic()
    out = ROOT / "candidate-probes"
    out.mkdir(exist_ok=True)
    result = {"scope": "exposed single-seed engineering feasibility, not G9 or benefit claim",
              "recipe": args.recipe, "seed": SEED, "source_rng": "random.Random(9900), Gaussian(0,1) row then Uniform(0.5,2) threshold per candidate",
              "candidate_pool": 128 if width else 0, "target_admitted_count": width,
              "window": 12, "t_post_g": 15.0, "output_init": "all zero; actual R-STDP learns outputs",
              "role_rule": "8 excitatory then 1 inhibitory per selected 9-unit block; no column coupling claimed",
              "trace_epochs": trace_epochs, "trace_eta": 0.05, "trace_feedback_seed": SEED,
              "max_teach_epochs": 400, "stable_epoch_ends": 10,
              "source_selection": "incumbent contribution(feature,residual) on validation, deterministic score/source-index order",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    problem = tp.PROBLEMS["ii.xor"]
    unit = tp.canonical_incumbent(SEED)
    reader = Reader(unit, history=False)
    result["tau"] = unit.tau
    teach = list(next(iter(problem.teach_epochs(SEED, 1))))
    validation = list(problem.episodes(SEED, "validation"))
    texts = [reader.text(ep) for ep in validation]
    ranks = [im.ranks(t) for t in texts]
    targets = [ep[-1].label for ep in validation]

    def probabilities(code):
        drives = [unit._drive(n, code) for n in unit.neurons]
        if unit._g9_context is not None:
            drives = [a + b for a, b in zip(drives, chat_grown_drives(unit, code), strict=True)]
        return label_probabilities(drives)

    trace_items = []
    for ep in teach:
        code = im.ranks(reader.text(ep))
        y = [float(label == ep[-1].label) for label in unit.responses]
        trace_items.append((code, y, probabilities(code)))
    residual = [[float(label == target) - p for label, p in zip(unit.responses, probabilities(code))]
                for code, target in zip(ranks, targets)]
    selected = []
    pool = []
    source_rng = random.Random(SEED)
    for i in range(result["candidate_pool"]):
        w = tuple(source_rng.gauss(0, 1) for _ in range(im.N_IN))
        threshold = source_rng.uniform(0.5, 2.0)
        if trace_epochs:
            w = trace_train(w, threshold, 12, unit.tau, trace_items, 0.05, trace_epochs, seed=SEED)
        candidate = GrownUnit(i, w, threshold, 12, 1, (0.0,) * len(unit.responses))
        features = [segment_psp((candidate,), code, unit.tau, 15.0)[0] for code in ranks]
        score = contribution(features, residual)
        pool.append((score, i, candidate, features))
    ranked = sorted((p for p in pool if p[0] > 0), key=lambda p: (-p[0], p[1]))
    for j, (score, i, candidate, features) in enumerate(ranked[:width]):
        sign = -1 if j % 9 == 8 else 1
        selected.append(GrownUnit(j, candidate.w_in, candidate.threshold, 12, sign, candidate.out))
    result["contributing_candidates"] = len(ranked)
    result["admitted_count"] = len(selected)
    result["selected_source_indices"] = [p[1] for p in ranked[:width]]
    result["selected_features"] = [p[3] for p in ranked[:width]]
    result["selected_units_initial"] = unit_rows(selected)
    result["candidate_search_seconds"] = time.monotonic() - start
    if selected:
        unit._g9_context = {"problem_id": problem.id, "units": unit_rows(selected),
                            "afferents": [[u.unit_id, []] for u in selected],
                            "t_post_g": 15.0, "out_eligible": [[True] * len(unit.responses) for _ in selected]}
    history = []
    stable = 0
    first_all = None
    for epoch, episodes in enumerate(problem.teach_epochs(SEED, 400), 1):
        for episode in episodes:
            reader.teach(episode, episode[-1].label)
        predictions = [unit.route_text(text) for text in texts]
        correct = sum(a == b for a, b in zip(predictions, targets))
        total = len(targets)
        if correct == total:
            if first_all is None:
                first_all = epoch
            stable += 1
        else:
            stable = 0
        history.append({"epoch": epoch, "correct": correct, "n": total, "stable_run": stable})
        if stable == 10:
            break
    predictions = [unit.route_text(text) for text in texts]
    final_context = unit._g9_context
    unit._g9_context = None
    masked = [unit.route_text(text) for text in texts]
    unit._g9_context = final_context
    snapshot = unit._snapshot()
    unit._authority.clear()
    cache_cleared = [unit.route_text(text) for text in texts]
    unit._restore_snapshot(snapshot)
    assert cache_cleared == predictions
    result.update(history=history, first_all_correct_epoch=first_all, stable_solved=stable == 10,
                  epochs=len(history), train_exposures=len(history)*len(teach),
                  validation_n=len(targets), correct=sum(a == b for a,b in zip(predictions,targets)),
                  targets=targets, predictions=predictions, masked_predictions=masked,
                  masked_correct=sum(a == b for a,b in zip(masked,targets)),
                  cache_clearing_preserves_predictions=True, wall_seconds=time.monotonic()-start)
    if final_context is not None:
        final_units = units_from_rows(final_context["units"], len(unit.responses))
        result["grown_version"] = grown_version(final_units)
        result["selected_units_final"] = unit_rows(final_units)
    (out / f"{args.recipe}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("history","selected_units_initial","selected_units_final","selected_features","targets","predictions","masked_predictions")}), flush=True)


if __name__ == "__main__":
    main()
