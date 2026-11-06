"""Conditional known-sigma planning only. This does not authorize a confirmation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ND = statistics.NormalDist()
ALPHA, POWER, DELTA, LOOKS, REPS = 0.05, 0.8, 0.10, 4, 20_000


def target(sd, delta, alpha):
    if sd <= 0:
        raise ValueError("ZERO_SD: more pilot information required; no powered N")
    return max(1, math.ceil(((ND.inv_cdf(1 - alpha) + ND.inv_cdf(POWER)) * sd / delta) ** 2))


def design(sd, delta=DELTA):
    fixed = target(sd, delta, ALPHA)
    required_cap = target(sd, delta, ALPHA / LOOKS)
    block = max(1, math.ceil(required_cap / LOOKS))
    looks = [block * k for k in range(1, LOOKS + 1)]
    z = ND.inv_cdf(1 - ALPHA / LOOKS)
    return {"sigma_assumed_known": sd, "delta": delta, "alpha": ALPHA,
            "fixed_n": fixed, "cap": looks[-1], "inflation": looks[-1] / fixed,
            "looks": [{"n": n, "alpha_increment": ALPHA / LOOKS, "z": z,
                       "mde": (z + ND.inv_cdf(POWER)) * sd / math.sqrt(n)} for n in looks],
            "futility": "nonbinding; MDE <= delta and conditional final-crossing power under delta < .10"}


def simulate(plan, residuals, model, effect, rng_seed):
    sd, cap = plan["sigma_assumed_known"], plan["cap"]
    schedule = {r["n"]: r for r in plan["looks"]}
    rng = random.Random(rng_seed)
    counts = {"efficacy": 0, "futility": 0, "cap_no_crossing_at_stated_mde": 0, "inconclusive": 0}
    stopped_n = 0
    for _ in range(REPS):
        total = 0.0
        for n in range(1, cap + 1):
            noise = rng.gauss(0, sd) if model == "normal" else rng.choice(residuals)
            total += noise + effect
            if n not in schedule:
                continue
            row = schedule[n]
            if total / (sd * math.sqrt(n)) > row["z"]:
                outcome = "efficacy"
                break
            if n < cap and row["mde"] <= plan["delta"]:
                threshold = row["z"] * sd * math.sqrt(cap)
                cp = ND.cdf((total + (cap - n) * plan["delta"] - threshold) / (sd * math.sqrt(cap - n)))
                if cp < 0.10:
                    outcome = "futility"
                    break
            if n == cap:
                outcome = "cap_no_crossing_at_stated_mde" if row["mde"] <= plan["delta"] else "inconclusive"
        counts[outcome] += 1
        stopped_n += n
    rates = {k: v / REPS for k, v in counts.items()}
    p = rates["efficacy"]
    # Wilson interval reports numerical simulation uncertainty, not pilot uncertainty.
    z, n = 1.959963984540054, REPS
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"model": model, "planted_effect": effect, "rng_seed": rng_seed, "replicates": REPS,
            "rates": rates, "efficacy_mc_95_wilson": [centre - half, centre + half],
            "mean_seeds_used": stopped_n / REPS,
            "meets_point_check": p <= ALPHA if effect == 0 else p >= POWER}


def plan(summary):
    xs = [r["trained_minus_shuffled"] for r in summary["paired"]]
    sd = statistics.stdev(xs)
    base = {"status": "DRAFT / NOT LAUNCH READY", "pilot_seeds": [r["seed"] for r in summary["paired"]],
            "pilot_differences": xs, "measured_paired_sd": sd,
            "primary": "final accuracy trained minus shuffled; learned dev positive and frozen control required",
            "limitations": ["four exposed dev seeds cannot establish population variance or tails",
                            "known-sigma normal model only; do not run a Z-test using this small pilot SD as known sigma",
                            "bootstrap simulation is a sensitivity check, not validation for every possible distribution",
                            "new confirmatory task seeds unassigned; no old frozen protocol inherited",
                            "10 percentage points is a design assumption, not an empirically established useful effect"],
            "formula_source": "https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm"}
    if sd == 0:
        return dict(base, status="ZERO_SD / POWER_UNRESOLVED / NOT LAUNCH READY", simulations=[])
    main = design(sd)
    # Residual empirical draws have population SD == measured sample SD; disclose rescaling.
    centre = statistics.mean(xs)
    pop = statistics.pstdev(xs)
    residuals = [(x - centre) * sd / pop for x in xs]
    simulations = [simulate(main, residuals, model, effect, seed)
                   for model, effect, seed in (("normal", 0, 301), ("normal", DELTA, 302),
                                                ("centred_rescaled_pilot", 0, 303),
                                                ("centred_rescaled_pilot", DELTA, 304))]
    base.update(design=main, sensitivity=[design(sd * multiplier, delta)
                                        for multiplier, delta in ((1, .05), (1, .20), (2, .10))],
                bootstrap_residuals=residuals, simulations=simulations,
                all_simulation_point_checks=all(r["meets_point_check"] for r in simulations))
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output file")
    # Genuine reds for scale/SD sensitivity, before using pilot values.
    try:
        target(0, .10, .05)
    except ValueError:
        pass
    else:
        raise RuntimeError("zero SD red failed")
    if not (target(.4, .1, .05) > target(.2, .1, .05) and target(.2, .05, .05) > target(.2, .1, .05)):
        raise RuntimeError("power sensitivity red failed")
    raw = args.summary.read_bytes()
    result = plan(json.loads(raw))
    result["summary_sha256"] = hashlib.sha256(raw).hexdigest()
    result["planner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(result["status"], "; paired SD:", result["measured_paired_sd"])
    if "design" in result:
        print("Fixed N / cap:", result["design"]["fixed_n"], result["design"]["cap"])
        for row in result["simulations"]:
            print(row["model"], row["planted_effect"], row["rates"], row["efficacy_mc_95_wilson"])


if __name__ == "__main__":
    main()
