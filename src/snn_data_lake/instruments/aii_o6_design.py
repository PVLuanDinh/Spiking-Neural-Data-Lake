#!/usr/bin/env python3
# ruff: noqa: E501
"""O-6 of the A_ii stack-repair pre-registration: the committed sample-size arithmetic (P1), stdlib only, no brain.

Section 4 + Fold 2 of `specs/022-unified-brain-arc/evidence/aii-stack-repair-prereg-DRAFT.md`, as a script:

  * the inputs: p0 = Clopper-Pearson upper bound on the original stack's 0/12, p1 = lower bound on the unstacked 4/4;
  * the fixed-sample target (smallest n whose exact one-sided test has alpha <= ALPHA and power >= 0.80);
  * the sequential design on the binomial lattice: looks, Lan-DeMets O'Brien-Fleming spending
    a(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt t), t = n_k / cap, efficacy counts solved EXACTLY under p0, non-binding
    futility (exact upper bound below p1), cap and inflation;
  * RD-4 / RD-5 validity tolerances (Bonferroni VOID_ERR/K per look) and their own error over the looks;
  * the exact JOINT lattice over the candidate, original and bare arms (paired guard x > x_orig, P(VOID), E[n]);
  * a Monte Carlo of the candidate arm alone, RNG seed 20261029.

ALPHA and the look schedule are PARAMETERS: `derive(alpha=..., looks=...)` and `--alpha/--looks` re-derive section 4. The
DEFAULT alpha is ALPHA_1 = .05 x 6/pi^2 = 0.0303964, candidate 1 of the D-82 alpha-spending series, i.e. the REGISTERED
design (n 27, cap 30, looks 10/20/30). The former alpha .05 run (looks 8/15/23) is Fold 2's reproduction. The printed
bound lines are derived from the design (looks, efficacy counts, confidence 1 - alpha); only claim (b)'s constants
(G2's S = 8, rate .917) are fixed by that claim itself.

Convention mirrored: `src/snn_data_lake/instruments/rpe_charge.py:440-486` (`seq_boundaries`, Lan-DeMets O'Brien-Fleming
spending, boundaries per look, non-binding futility) and `:434-438` (the `distinct-from` note). It is NOT imported: that
module's boundaries are a normal-model t-scale integral; this design is a COUNT lattice, a different object.
Moved from tools/ into the governed tree (P4c, 2026-11-01): the former "importing `snn_data_lake` pulls torch" caveat (campaign rule 1) is history, the campaign rule ended. The script layout (module docstring with a Reproduce line, `main`)
follows `tools/red_reachability.py:1-24`.
# distinct-from: instruments/rpe_charge.py seq_boundaries -- continuous t-scale boundaries; this solves integer counts
#   exactly on the binomial lattice and adds the joint three-arm VOID/guard lattice (graph-exists "exact binomial
#   lattice sequential design power": NO LEXICAL MATCH; "sequential stopping boundary alpha spending": rpe_charge only).

Reproduce (WSL system python3, stdlib, well under 200 MB):
    python3 src/snn_data_lake/instruments/aii_o6_design.py                    # the REGISTERED design: alpha_1 = 0.0303964, looks 10/20/30
    python3 src/snn_data_lake/instruments/aii_o6_design.py --alpha 0.05       # Fold 2 reproduction (superseded): looks 8/15/23
    python3 src/snn_data_lake/instruments/aii_o6_design.py --looks 10,20,23   # another look schedule
Reds: tests/test_aii_o6_design.py.
"""

import argparse
import math
import random
import statistics
import sys

SEED = 20261029
ALPHA_1 = 0.05 * 6 / math.pi**2  # D-82 series, candidate 1: alpha_i = .05 x 6 / (pi^2 i^2)
REPS = 200_000
_ND = statistics.NormalDist()


def pmf(n, p):
    return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def tail_ge(n, p, t):
    """P(X >= t), X ~ Bin(n, p)."""
    return math.fsum(pmf(n, p)[max(t, 0) :])


def _bisect(f, lo=0.0, hi=1.0):
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return (lo + hi) / 2


def cp_upper(x, n, conf=0.95):
    """Exact one-sided upper bound on p after x successes in n: P(X <= x | p) = 1 - conf."""
    if x >= n:
        return 1.0
    return _bisect(lambda m: 1 - tail_ge(n, m, x + 1) > 1 - conf)


def cp_lower(x, n, conf=0.95):
    """Exact one-sided lower bound on p after x successes in n: P(X >= x | p) = 1 - conf."""
    if x <= 0:
        return 0.0
    return _bisect(lambda m: tail_ge(n, m, x) < 1 - conf)


def spend_at(t, alpha):
    """Lan-DeMets O'Brien-Fleming spending a(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt t), in tail form (no underflow)."""
    return 2 * 0.5 * math.erfc(_ND.inv_cdf(1 - alpha / 2) / math.sqrt(t) / math.sqrt(2))


def _convolve(dist, axis, step):
    new = {}
    for s, q in dist.items():
        for k, r in enumerate(step):
            if r == 0.0:
                continue
            t = list(s)
            t[axis] += k
            t = tuple(t)
            new[t] = new.get(t, 0.0) + q * r
    return new


def cum_alpha(looks, eff, p):
    """Cumulative P(candidate crosses its efficacy count by look k | rate p); no futility (non-binding), guard idle."""
    dist, prev, out, cross = {0: 1.0}, 0, [], 0.0
    for n, c in zip(looks, eff, strict=True):
        step, new = pmf(n - prev, p), {}
        for x, q in dist.items():
            for k, r in enumerate(step):
                new[x + k] = new.get(x + k, 0.0) + q * r
        cross += math.fsum(q for x, q in new.items() if x >= c)
        out.append(cross)
        dist = {x: q for x, q in new.items() if x < c}
        prev = n
    return out


def run_candidate(looks, eff, fut, p):
    """Candidate arm alone: (P(MET), P(futility stop), E[seeds read]). `fut` None = futility ignored."""
    dist, prev, cross, futp, en, last = {0: 1.0}, 0, 0.0, 0.0, 0.0, len(looks) - 1
    for k, (n, c) in enumerate(zip(looks, eff, strict=True)):
        step, new = pmf(n - prev, p), {}
        for x, q in dist.items():
            for j, r in enumerate(step):
                new[x + j] = new.get(x + j, 0.0) + q * r
        dist = {}
        for x, q in new.items():
            if x >= c:
                cross += q
                en += q * n
            elif fut is not None and k < last and x <= fut[k]:
                futp += q
                en += q * n
            else:
                dist[x] = q
        prev = n
    en += math.fsum(dist.values()) * looks[-1]
    return cross, futp, en


def alpha_ok(d, eff):
    """The efficacy counts spend no more than the Lan-DeMets budget at any look (and so <= alpha overall)."""
    got = cum_alpha(d["looks"], eff, d["p0"])
    return all(g <= s + 1e-12 for g, s in zip(got, d["spend"], strict=True))


def power_ok(d, eff):
    """The efficacy counts keep power >= the target at p1 (futility ignored)."""
    return cum_alpha(d["looks"], eff, d["p1"])[-1] >= d["power"] - 1e-12


def _eff_counts(looks, spend, p0):
    eff = []
    for k, n in enumerate(looks):
        for c in range(n + 2):
            trial = [*eff, c] + [n + 1000] * (len(looks) - k - 1)
            if cum_alpha(looks, trial, p0)[k] <= spend[k] + 1e-15:
                eff.append(c)
                break
    return eff


def _fut_counts(looks, p1, conf):
    return [max([x for x in range(n + 1) if cp_upper(x, n, conf) < p1], default=-1) for n in looks]


def _rd_counts(looks, p0, p1, per_look):
    rd4 = [min(t for t in range(n + 2) if tail_ge(n, p0, t) <= per_look) for n in looks]
    rd5 = [max(lo for lo in range(n + 1) if 1 - tail_ge(n, p1, lo) <= per_look) for n in looks]
    return rd4, rd5


def _flag_prob(looks, p, flagged):
    """P(an arm is ever flagged at a look), unstopped paths: flagged(k, x) per look."""
    dist, prev, ever = {0: 1.0}, 0, 0.0
    for k, n in enumerate(looks):
        step, new = pmf(n - prev, p), {}
        for x, q in dist.items():
            for j, r in enumerate(step):
                new[x + j] = new.get(x + j, 0.0) + q * r
        ever += math.fsum(q for x, q in new.items() if flagged(k, x))
        dist = {x: q for x, q in new.items() if not flagged(k, x)}
        prev = n
    return ever


def _fixed_sample(p0, p1, alpha, power, n_max=400):
    for n in range(1, n_max):
        c = next((c for c in range(n + 2) if tail_ge(n, p0, c) <= alpha), n + 1)
        if c <= n and tail_ge(n, p1, c) >= power:
            return n, c
    raise ValueError(
        f"no n < {n_max} meets alpha {alpha} and power {power} for p0 {p0:.4f}, p1 {p1:.4f}"
    )


def _design(looks, alpha, p0, p1, conf, void_err):
    cap = looks[-1]
    spend = [spend_at(n / cap, alpha) for n in looks]
    eff = _eff_counts(looks, spend, p0)
    fut = _fut_counts(looks, p1, conf)
    rd4, rd5 = _rd_counts(looks, p0, p1, void_err / len(looks))
    d = {
        "alpha": alpha,
        "conf": conf,
        "p0": p0,
        "p1": p1,
        "looks": list(looks),
        "cap": cap,
        "spend": spend,
        "eff": eff,
        "fut": fut[:-1],
        "fut_cap": fut[-1],
        "rd4": rd4,
        "rd5": rd5,
        "void_err": void_err,
        "alpha_real": cum_alpha(looks, eff, p0),
    }
    d["power_cap_ignored"] = run_candidate(looks, eff, None, p1)[0]
    d["power_cap_obeyed"] = run_candidate(looks, eff, d["fut"], p1)[0]
    d["void4_err"] = _flag_prob(looks, p0, lambda k, x: x >= rd4[k])
    d["void5_err"] = _flag_prob(looks, p1, lambda k, x: x < rd5[k])
    return d


def derive(
    alpha=ALPHA_1,
    looks=None,
    conf=0.95,
    k0=(0, 12),
    k1=(4, 4),
    power=0.8,
    void_err=0.02,
    cap_max=None,
):
    """Section 4 under (alpha, looks). `looks` None: target n, looks (ceil(n/3), round(2n/3), n), cap raised from n until
    the worst-case P(MET and no VOID) at p1 (original p0, bare p1, paired guard) is >= `power` (the inflation factor);
    no cap below `cap_max` (default 3n) reaching it RAISES ValueError. `conf` (the confidence of the p0 / p1 input
    bounds, a property of the data, not of the test level) defaults to 0.95 whatever alpha is."""
    p0 = cp_upper(k0[0], k0[1], conf)
    p1 = cp_lower(k1[0], k1[1], conf)
    n, c = _fixed_sample(p0, p1, alpha, power)
    if looks is None:
        for cap in range(n, cap_max or 3 * n):
            lk = [math.ceil(cap / 3), round(2 * cap / 3), cap]
            d = _design(lk, alpha, p0, p1, conf, void_err)
            if (
                joint(d, p1, p0, p1)["met"] >= power
            ):  # VOID counted, worst case the measured counts allow
                break
        else:
            raise ValueError(
                f"no cap in [{n}, {cap_max or 3 * n}) reaches joint power {power} at alpha {alpha}"
            )
    else:
        if not looks or looks[0] < 1 or any(b <= a for a, b in zip(looks, looks[1:], strict=False)):
            raise ValueError(
                f"looks must be strictly increasing positive counts, got {list(looks)}"
            )
        d = _design(list(looks), alpha, p0, p1, conf, void_err)
    d.update(
        power=power,
        n_fixed=n,
        c_fixed=c,
        alpha_fixed=tail_ge(n, p0, c),
        power_fixed=tail_ge(n, p1, c),
    )
    d["inflation"] = round(d["cap"] / n, 2)
    d["power_joint_worst"] = joint(d, p1, p0, p1)["met"]
    return d


def joint(d, pc, po, pb, guard=True, futility=True):
    """Exact joint lattice over candidate (rate pc), original (po) and bare (pb), seeds read in look order.

    At each look, in this order: VOID iff original >= RD-4 count or bare < RD-5 count (validity first); else MET iff
    candidate >= its efficacy count AND (if `guard`) candidate > original; else (not the cap, `futility`) stop for
    futility iff candidate <= its futility count; at the cap anything else ends INCONCLUSIVE or NOT MET powered
    (candidate <= fut_cap and no crossing). A crossing whose guard fails continues to the next look.
    Returns met, void, void_l1, notmet, inconclusive, en (expected seeds read), rounded to 12 dp (float noise)."""
    looks, last = d["looks"], len(d["looks"]) - 1
    dist, prev = {(0, 0, 0): 1.0}, 0
    out = dict.fromkeys(("met", "void", "void_l1", "notmet", "inconclusive", "en"), 0.0)
    for k, n in enumerate(looks):
        for axis, p in enumerate((pc, po, pb)):
            dist = _convolve(dist, axis, pmf(n - prev, p))
        nxt = {}
        for (x, xo, xb), q in dist.items():
            if xo >= d["rd4"][k] or xb < d["rd5"][k]:
                out["void"] += q
                out["void_l1"] += q if k == 0 else 0.0
            elif x >= d["eff"][k] and (not guard or x > xo):
                out["met"] += q
            elif futility and k < last and x <= d["fut"][k]:
                out["notmet"] += q
            elif k == last:
                out["notmet" if (x < d["eff"][k] and x <= d["fut_cap"]) else "inconclusive"] += q
            else:
                nxt[(x, xo, xb)] = q
                continue
            out["en"] += q * n
        dist, prev = nxt, n
    return {key: round(v, 12) for key, v in out.items()}


def mc(d, ps, reps=REPS, seed=SEED):
    """Monte Carlo P(MET) of the candidate arm alone (futility obeyed, guard idle), one RNG seeded once, `ps` in order."""
    rnd, res = random.Random(seed), {}
    for p in ps:
        met = 0
        for _ in range(reps):
            s, prev = 0, 0
            for k, n in enumerate(d["looks"]):
                s += sum(rnd.random() < p for _ in range(n - prev))
                prev = n
                if s >= d["eff"][k]:
                    met += 1
                    break
                if k < len(d["looks"]) - 1 and s <= d["fut"][k]:
                    break
        res[p] = met / reps
    return res


def mc_joint(d, ps, reps=REPS, seed=SEED):
    """Monte Carlo of the JOINT rule (same order as `joint`: VOID, then MET with the guard, then futility) for arm rates
    `ps` = (candidate, original, bare): {"met", "void"} fractions, one RNG seeded once."""
    rnd, last, out = random.Random(seed), len(d["looks"]) - 1, {"met": 0, "void": 0}
    for _ in range(reps):
        xs, prev = [0, 0, 0], 0
        for k, n in enumerate(d["looks"]):
            for a, p in enumerate(ps):
                xs[a] += sum(rnd.random() < p for _ in range(n - prev))
            prev = n
            if xs[1] >= d["rd4"][k] or xs[2] < d["rd5"][k]:
                out["void"] += 1
                break
            if xs[0] >= d["eff"][k] and xs[0] > xs[1]:
                out["met"] += 1
                break
            if k < last and xs[0] <= d["fut"][k]:
                break
    return {key: v / reps for key, v in out.items()}


def report(d, reps=REPS, seed=SEED):
    f = lambda xs, nd=5: "/".join(f"{x:.{nd}f}" for x in xs)  # noqa: E731
    print(
        f"alpha {d['alpha']}  conf {d['conf']}  looks {d['looks']}  cap {d['cap']}  inflation {d['inflation']:.2f}"
    )
    print(f"p0 {d['p0']:.4f}  p1 {d['p1']:.4f}")
    print(
        f"fixed-sample target n {d['n_fixed']} c {d['c_fixed']} alpha {d['alpha_fixed']:.4f} power {d['power_fixed']:.4f}"
    )
    print(f"spend {f(d['spend'])}  realised cumulative alpha {f(d['alpha_real'])}")
    print(
        f"efficacy counts {d['eff']}  futility counts (looks 1..K-1) {d['fut']}  futility at cap {d['fut_cap']}"
    )
    print(
        "futility upper bounds "
        + ", ".join(
            f"{cp_upper(x, n, d['conf']):.3f}" for x, n in zip(d["fut"], d["looks"], strict=False)
        )
    )
    print(
        f"power at cap: futility ignored {d['power_cap_ignored']:.4f}, obeyed {d['power_cap_obeyed']:.4f}"
    )
    print(f"RD-4 original VOID counts >= {d['rd4']}  own error {d['void4_err']:.4f}")
    print(f"RD-5 bare VOID counts < {d['rd5']}  own error {d['void5_err']:.4f}")
    p0, p1 = d["p0"], d["p1"]
    print("operating characteristics, measured points (original 0, bare 1; guard idle, no VOID):")
    for lab, pc in (
        ("0", 0.0),
        ("p0", p0),
        ("0.35", 0.35),
        ("p1", p1),
        ("0.6", 0.6),
        ("0.8", 0.8),
        ("0.9", 0.9),
        ("1.0", 1.0),
    ):
        j = joint(d, pc, 0.0, 1.0)
        print(
            f"  cand {lab:>4}: P(MET) {j['met']:.4f}  P(VOID) {j['void']:.4f}  E[n] {j['en']:.1f}"
        )
    print("joint worst case (original p0, bare p1; paired guard on):")
    for lab, pc in (("p0", p0), ("0.35", 0.35), ("p1", p1), ("0.6", 0.6), ("0.9", 0.9)):
        j = joint(d, pc, p0, p1)
        print(
            f"  cand {lab:>4}: P(MET) {j['met']:.4f}  P(VOID) {j['void']:.4f}  INCONCLUSIVE {j['inconclusive']:.4f}"
        )
    print("drift row (original's true rate drifted to q, candidate = original, bare 1; guard on):")
    for lab, q in (("p0", p0), ("0.25", 0.25), ("0.30", 0.30), ("0.35", 0.35)):
        j = joint(d, q, q, 1.0)
        print(f"  q {lab:>4}: P(MET) {j['met']:.4f}  P(VOID) {j['void']:.4f}")
    print(
        f"always-solving original, cand p1: P(VOID) {joint(d, p1, 1.0, 1.0)['void']:.4f} at L1 {joint(d, p1, 1.0, 1.0)['void_l1']:.4f}"
    )
    cap, k = d["cap"], d["conf"]
    print(
        f"MET lower bounds at confidence 1-alpha_1 = {1 - d['alpha']:.4f} (ignore the sequential stop): "
        + ", ".join(
            f"x={x}/{n} {cp_lower(x, n, 1 - d['alpha']):.3f}"
            for x, n in zip(d["eff"], d["looks"], strict=True)
        )
    )
    print(
        f"MET lower bounds at confidence {k} (same counts): "
        + ", ".join(
            f"x={x}/{n} {cp_lower(x, n, k):.3f}" for x, n in zip(d["eff"], d["looks"], strict=True)
        )
    )
    print(
        f"INCONCLUSIVE at the cap (futility count {d['fut_cap']} < x < efficacy count {d['eff'][-1]}), upper bounds at {k}: "
        + ", ".join(
            f"x={x}/{cap} {cp_upper(x, cap, k):.3f}" for x in range(d["fut_cap"] + 1, d["eff"][-1])
        )
    )
    lo, hi = p0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (
            (lo, mid)
            if run_candidate(d["looks"], d["eff"], d["fut"], mid)[0] >= d["power"]
            else (mid, hi)
        )
    print(f"mde (power {d['power']} at the cap, futility obeyed) p = {hi:.3f}")
    q = 0.8 ** (1 / 8)
    n_b = math.ceil(math.log(0.05) / math.log(0.917))
    n_s = math.ceil(math.log(0.05) / math.log(q))
    print(
        f"claim (b): 0.8^(1/8) = {q:.4f}; all-pass n to reject p <= 0.917 = {n_b}, power {q**n_b:.2f}; n to show {q:.4f}: {n_s}"
    )
    r = mc(d, [p0, p1], reps, seed)
    print(
        f"Monte Carlo, candidate arm alone ({reps} reps per p, RNG seed {seed}): P(MET) at p0 {r[p0]:.4f}, at p1 {r[p1]:.4f}"
    )
    rj = mc_joint(d, (p1, p0, p1), reps, seed)
    ej = joint(d, p1, p0, p1)
    print(
        f"Monte Carlo, joint lattice (candidate p1, original p0, bare p1; guard on; {reps} reps, seed {seed}): "
        f"P(MET) {rj['met']:.4f} vs exact {ej['met']:.4f}, P(VOID) {rj['void']:.4f} vs exact {ej['void']:.4f}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--alpha", type=float, default=ALPHA_1, help="default: alpha_1 = .05 x 6/pi^2 (D-82 series)"
    )
    ap.add_argument(
        "--looks", default=None, help="cumulative seeds per look, e.g. 8,15,23 (default: derived)"
    )
    ap.add_argument("--conf", type=float, default=0.95, help="confidence of the p0/p1 input bounds")
    ap.add_argument("--void-err", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    looks = [int(v) for v in a.looks.split(",")] if a.looks else None
    try:
        d = derive(a.alpha, looks, a.conf, void_err=a.void_err)
    except ValueError as e:
        ap.error(str(e))
    report(d, a.reps, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
