r"""G6 / T114 part (b): neuron-mask search by joint-mask validation -- one search contract, the arms behind
it at an EQUAL evaluation budget, the ordered decision `decide_b`, and the one `mask_digest`.

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md, frozen at P0 (df19f61e) and
amended 16d7e3b8: section 2.4 (the search), 2.11 (C7-C10), 2.16's A3 (`decide_b`, C22), A13 (budget
accounting, SA initialization), A14 (two retention sets, C26) and the DATED PRE-DATA AMENDMENT 2026-10-03
(`mask_digest`, shipped HERE so R2's P2 imports it without waiting on G6's P4 -- ruling 17a).

THE CONTRACT. A mask problem exposes `n` mask elements and four reads:
  cost(mask)          -> C = active units + active synapses of the masked network (the objective)
  evaluate(mask)      -> (feasible, score): refit on `teach`, the solved predicate and `retention_select`
                         on validation. ONE call = ONE evaluation charged against B, feasible or not.
  judge(mask)         -> (future_ok, retention_ok): the untouched `future` and `retention_final` reads,
                         allowed ONCE per arm and ONLY after the winner is frozen (C26 refuses earlier).
  work_per_tick(mask) -> per-tick compute of the masked network (net savings, 2.4)
The arms never see more than this, so a planted fixture and G3's grown populations run through the same
code. This module builds NO brain and touches no lake; the deciding run supplies a problem built from
G3's solved `ProblemRecord` (waits: 2.10).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.joint_mask_search   (self-check)
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast

# distinct-from: faculties/theory_of_mind.py::believed_location -- a spiking WTA readout of a belief.
#   distinct-from: growth/basic_components.py::Node, growth/neuron_zoo.py::_AdExCell -- the neuron step
#   contract. distinct-from: brainstem/arm_config.py::_joint_limits -- joint ranges. distinct-from:
#   engine/brain.py::Brain.tick -- the wake/sleep iteration. Phrase overlap only; none searches masks
#   ("No simulated annealing exists in the tree", PLAN; `grep anneal` hits only exploration schedules in
#   engine/autonomous_loop.py and faculties/developmental_envelope.py). Sites read.

COMPETITORS = ("incumbent", "greedy", "random", "SA")


# ================================================================================================
# mask_digest -- the ONE digest of a prepared mask (dated pre-data amendment 2026-10-03, item 3)
# ================================================================================================
def mask_digest(prepared_row):
    """sha256 hex of the canonical JSON (sorted keys, separators (",", ":")) of {sleep_id, tick, sorted kept
    ids, sorted pruned ids, sorted [kind, pre, post, kept_bit] synapses}. A digest of the MASK, never of the
    network state. R2's redo imports this; it is never reimplemented."""
    body = {
        "sleep_id": str(prepared_row["sleep_id"]),
        "tick": int(prepared_row["tick"]),
        "kept": sorted(str(u) for u in prepared_row.get("kept", ())),
        "pruned": sorted(str(u) for u in prepared_row.get("pruned", ())),
        "synapses": sorted(
            [str(kind), str(pre), str(post), int(bool(bit))]
            for kind, pre, post, bit in prepared_row.get("synapses", ())
        ),
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ================================================================================================
# budget accounting (A13) and the selection-leak guard (A14 / C26)
# ================================================================================================
class BudgetExhausted(Exception):
    pass


class SelectionLeak(Exception):
    """A judging set (`future`, `future_d`, `retention_final`) was read before the winner was frozen."""


class Meter:
    """Wraps a problem for ONE arm: every `evaluate` is charged (calibration and infeasible included);
    `judge` refuses until `freeze()`, and refuses a second read."""

    def __init__(self, problem, budget=None):
        self.p = problem
        self.budget = budget
        self.spent = 0
        self.frozen = False
        self.judged = 0

    def evaluate(self, mask, *, feasibility=True):
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExhausted(self.spent)
        self.spent += 1
        feasible, score = self.p.evaluate(tuple(mask))
        return (feasible if feasibility else True), score

    def cost(self, mask):
        return self.p.cost(tuple(mask))

    def freeze(self):
        self.frozen = True

    def judge(self, mask):
        if not self.frozen:
            raise SelectionLeak("judge() read before the winner was frozen")
        if self.judged:
            raise SelectionLeak("judge() read twice: the untouched sets are read once")
        self.judged += 1
        return self.p.judge(tuple(mask))


# ================================================================================================
# the arms
# ================================================================================================
def _full(n):
    return tuple([1] * n)


def _one_pass(meter, mask, feasibility=True):
    """The incumbent's discipline (`prune_unearned`): score every single removal against the CURRENT mask,
    drop ALL that are individually droppable at once (feasible and score no worse), keep >= 1 element.
    Returns (new_mask, removed_count). Its known ceiling (jointly necessary pairs both dropped) is kept."""
    _, base = meter.evaluate(mask, feasibility=feasibility)
    active = [i for i, b in enumerate(mask) if b]
    drop, cost = [], {}
    for i in active:
        trial = list(mask)
        trial[i] = 0
        f, s = meter.evaluate(trial, feasibility=feasibility)
        cost[i] = s
        if f and s >= base:
            drop.append(i)
    if not drop:
        return tuple(mask), 0
    if len(drop) >= len(active):  # all droppable: keep exactly the one whose removal costs the most
        survivor = min(active, key=lambda i: (cost.get(i, math.inf), i))
        drop = [i for i in active if i != survivor]
    out = list(mask)
    for i in drop:
        out[i] = 0
    return tuple(out), len(drop)


def arm_incumbent(meter, n, feasibility=True):
    mask, _ = _one_pass(meter, _full(n), feasibility)
    return mask


def arm_greedy(meter, n, feasibility=True):
    """`prune_unearned` iterated until it removes nothing; each pass re-checks feasibility; stops at the
    first infeasible pass (keeping the previous mask) or the fixed point. Its evaluation count DEFINES B."""
    mask = _full(n)
    while True:
        nxt, removed = _one_pass(meter, mask, feasibility)
        if removed == 0:
            return mask
        f, _ = meter.evaluate(nxt, feasibility=feasibility)
        if not f:
            return mask
        mask = nxt


def arm_random(meter, n, q, rng, feasibility=True):
    """Exactly B draws (every draw charged), each element kept with probability q; the best feasible draw on
    validation (smallest C, then higher score, then first) wins; none feasible -> the all-active mask."""
    best = None
    while True:
        mask = tuple(1 if rng.random() < q else 0 for _ in range(n))
        try:
            f, s = meter.evaluate(mask, feasibility=feasibility)
        except BudgetExhausted:
            break
        if f:
            key = (meter.cost(mask), -s)
            if best is None or key < best[0]:
                best = (key, mask)
    return best[1] if best else _full(n)


def _flip(mask, i):
    out = list(mask)
    out[i] = 1 - out[i]
    return tuple(out)


def arm_sa(meter, n, rng, feasibility=True):
    """A13: start all-active; a proposal flips one uniformly chosen element; the first min(n, floor(B/2))
    proposals are the calibration sweep (accept every feasible downhill move, part of B); T0 = median
    |delta C| / ln 2; geometric cooling to T0 / B over the rest; infeasible proposals rejected and counted."""
    B = meter.budget
    if B is None or B < 4:
        raise ValueError("SA needs a budget B >= 4 (A13: below it the instance is REFUSED)")
    cur = _full(n)
    cur_c = meter.cost(cur)
    best, best_c = cur, cur_c
    deltas = []
    try:
        for _ in range(min(n, B // 2)):
            prop = _flip(cur, rng.randrange(n))
            pc = meter.cost(prop)
            deltas.append(abs(pc - cur_c))
            f, _ = meter.evaluate(prop, feasibility=feasibility)
            if f and pc <= cur_c:
                cur, cur_c = prop, pc
                if pc < best_c:
                    best, best_c = prop, pc
        deltas.sort()
        temp = (deltas[len(deltas) // 2] if deltas else 1.0) / math.log(2.0)
        alpha = (1.0 / B) ** (
            1.0 / max(1, B - meter.spent)
        )  # T0 -> T0 / B over the remaining proposals
        while True:
            prop = _flip(cur, rng.randrange(n))
            pc = meter.cost(prop)
            f, _ = meter.evaluate(prop, feasibility=feasibility)
            if f and (pc <= cur_c or rng.random() < math.exp(-(pc - cur_c) / temp)):
                cur, cur_c = prop, pc
                if pc < best_c:
                    best, best_c = prop, pc
            temp *= alpha
    except BudgetExhausted:
        pass
    return best


@dataclass
class ArmResult:
    arm: str
    mask: tuple
    cost: int
    evaluations: int
    future_ok: bool = False
    retention_ok: bool = False
    work_per_tick: float = 0.0


# distinct-from: gates/amdahl_gate.py -- it proves a patch installs and counts its calls over an untimed run;
#   distinct-from: infra/self_mod_harness.py -- it runs run_selfchecks in a sandbox. Neither runs search arms
#   on a mask problem. Sites read.
def run_instance(problem, seed, *, feasibility=True, budget=None, q=None):
    """All arms on ONE instance at equal B (= greedy's measured evaluation count; `budget` and `q` override
    it ONLY for the planted C8 fixture, disclosed). Judging happens once per arm, after its winner is
    frozen. Returns {arm: ArmResult} plus 'no_prune' and '_B'."""
    n = problem.n
    g = Meter(problem)
    gm = arm_greedy(g, n, feasibility)
    B = g.spent if budget is None else int(budget)
    q = sum(gm) / float(n) if q is None else float(q)
    masks = {"greedy": (gm, g)}
    inc = Meter(problem)
    masks["incumbent"] = (arm_incumbent(inc, n, feasibility), inc)
    rm = Meter(problem, budget=B)
    masks["random"] = (arm_random(rm, n, q, random.Random(f"g6.random:{seed}"), feasibility), rm)
    if B >= 4:
        sm = Meter(problem, budget=B)
        masks["SA"] = (arm_sa(sm, n, random.Random(f"g6.sa:{seed}"), feasibility), sm)
    masks["no_prune"] = (_full(n), Meter(problem, budget=0))
    out = {}
    for arm, (mask, meter) in masks.items():
        meter.freeze()
        fut, ret = meter.judge(mask)
        out[arm] = ArmResult(
            arm, mask, problem.cost(mask), meter.spent, fut, ret, problem.work_per_tick(mask)
        )
    out["_B"] = B
    return out


def net_savings(res, no_prune, *, horizon, eval_cost):
    """2.4: (per-tick compute saved) x H - (search compute). H is the measured median wake-period length
    in ticks (a disclosed session call); eval_cost is one evaluation's compute in the same unit."""
    return (no_prune.work_per_tick - res.work_per_tick) * horizon - res.evaluations * eval_cost


# ================================================================================================
# decide_b -- A3's ordered decision function
# ================================================================================================
@dataclass
class Disposition:
    outcome: str  # SA | GREEDY | RANDOM-WIN | INCUMBENT | UNBUILT | REFUSED
    words: str
    eligible: list = field(default_factory=list)
    contrasts: dict = field(default_factory=dict)


def _sup(name, diffs):
    c = contrast(name, diffs)
    return c._replace(met=False) if c.degenerate else c  # A9: degenerate superiority -> NOT MET


def decide_b(seeds, *, instrument_gates_fired):
    """`seeds`: per-seed {arm: {"cost", "future_ok", "retention_ok", "net"}} for every arm in COMPETITORS
    (an arm absent on a refused small-budget instance REFUSES the read). The first applicable step decides."""
    if not instrument_gates_fired:
        return Disposition("REFUSED", "C7-C10/C22 did not all fire on their constructed inputs")
    if any(a not in s for s in seeds for a in COMPETITORS):
        return Disposition("REFUSED", "an arm is missing on some seed (A13 small-budget refusal)")
    con = {}
    eligible = []
    for a in COMPETITORS:
        gates = all(s[a]["future_ok"] and s[a]["retention_ok"] for s in seeds)
        net = _sup(f"{a} net savings > 0", [s[a]["net"] for s in seeds])
        con[f"{a} net"] = net
        if gates and net.met:
            eligible.append(a)

    def beats(x, y):
        c = _sup(f"{x} smaller than {y}", [s[y]["cost"] - s[x]["cost"] for s in seeds])
        con[f"{x}<{y}"] = c
        return c.met

    others = [a for a in eligible if a != "SA"]
    if "SA" in eligible and all(beats("SA", o) for o in others):
        return Disposition(
            "SA",
            "SA adopted as THE unit pruner; the single pass retired after its port",
            eligible,
            con,
        )
    if (
        "greedy" in eligible
        and beats("greedy", "incumbent")  # A3 step 3: the incumbent need not be eligible
        and not ("random" in eligible and beats("random", "greedy"))
    ):
        return Disposition(
            "GREEDY", "the incumbent adopts greedy's fixed-point loop; SA deleted", eligible, con
        )
    structured = [a for a in eligible if a != "random"]
    if "random" in eligible and structured and all(beats("random", s_) for s_ in structured):
        return Disposition(
            "RANDOM-WIN",
            "SA deleted (lost to its own control); greedy not adopted; incumbent unchanged; opening recorded: "
            "a smaller feasible mask exists than the structured searches find",
            eligible,
            con,
        )
    if "incumbent" in eligible:
        return Disposition("INCUMBENT", "the incumbent stays as is; SA deleted", eligible, con)
    return Disposition(
        "UNBUILT",
        "nothing eligible: SA deleted; the prune step for grown units stands UNBUILT",
        eligible,
        con,
    )


# ================================================================================================
# the planted fixture (C8): k needed blocks, `dup` redundant copies each, an old-task unit, a known minimum
# ================================================================================================
class CoverFixture:
    """Elements = units. Unit u covers block `blk[u]`; blocks 0..k-1 are the solved problem's, block k is the
    OLD task's (retention). A mask is feasible on validation iff every problem block is covered AND the old
    block is covered (`retention_select`). Judging re-reads the same predicate on separate untouched sets,
    broken on demand by `future_breaks` (so the future judgment can red). Each unit carries `syn` synapses,
    so C = sum over active units of (1 + syn)."""

    def __init__(self, k=4, dup=3, syn=5, future_breaks=False):
        self.k, self.dup, self.syn = k, dup, syn
        self.blk = [b for b in range(k) for _ in range(dup)] + [
            k
        ]  # the last unit is the old task's
        self.n = len(self.blk)
        self.future_breaks = future_breaks

    def _covered(self, mask):
        return {self.blk[i] for i, b in enumerate(mask) if b}

    def cost(self, mask):
        return sum(1 + self.syn for b in mask if b)

    def evaluate(self, mask):
        cov = self._covered(mask)
        return (all(b in cov for b in range(self.k)) and self.k in cov), float(len(cov))

    def judge(self, mask):
        cov = self._covered(mask)
        return (all(b in cov for b in range(self.k)) and not self.future_breaks), self.k in cov

    def work_per_tick(self, mask):
        return float(self.cost(mask))

    def minimum(self):
        return (self.k + 1) * (1 + self.syn)


def constructed_seeds(over=None):
    """C22's constructed per-seed readings: every arm eligible and tied, then `over` {(arm, key): value}."""
    out = []
    for i in range(8):
        s = {
            a: {"cost": 30.0, "future_ok": True, "retention_ok": True, "net": 5.0 + (i % 3) * 0.1}
            for a in COMPETITORS
        }
        s["incumbent"]["cost"] += (i % 2) * 0.5
        for (a, key), val in (over or {}).items():
            s[a][key] = val(i) if callable(val) else val
        out.append(s)
    return out


def _smaller(i):
    return 20.0 + (i % 2) * 0.3


def _decide(seeds):
    return decide_b(seeds, instrument_gates_fired=True)


def main():
    row = {
        "sleep_id": "s1-n3",
        "tick": 7,
        "kept": ["u2", "u1"],
        "pruned": ["u3"],
        "synapses": [("feed", "x", "u1", 1), ("readout", "u3", "r0", 0)],
    }
    perm = {**row, "kept": ["u1", "u2"], "synapses": list(reversed(row["synapses"]))}
    flip = {**row, "synapses": [("feed", "x", "u1", 0), ("readout", "u3", "r0", 0)]}
    assert mask_digest(row) == mask_digest(perm) and mask_digest(row) != mask_digest(flip)

    # C8, paired over 8 fixture seeds at a PLANTED budget B = 200 and the most favourable random q (the
    # minimum's kept fraction) -- the deciding run's B is greedy's count and q greedy's kept fraction (A13)
    fx = CoverFixture(k=8, dup=4)
    q_min = (fx.k + 1) / float(fx.n)
    runs = [run_instance(fx, s, budget=200, q=q_min) for s in range(8)]
    B = runs[0]["_B"]
    assert all(r["SA"].evaluations == B == r["random"].evaluations for r in runs), "equal budgets"
    assert all(r["SA"].cost == fx.minimum() for r in runs), ("C8", [r["SA"].cost for r in runs])
    c8 = _sup("C8 SA smaller than random", [r["random"].cost - r["SA"].cost for r in runs])
    assert c8.met, ("C8: SA must beat random on the planted fixture", c8)
    assert all(a == "_B" or r[a].retention_ok for r in runs for a in r), (
        "feasibility keeps the old unit"
    )
    res = runs[0]

    bad = run_instance(fx, 0, feasibility=False)
    assert not all(r.future_ok for a, r in bad.items() if a not in ("_B", "no_prune")), (
        "C7: judge must red"
    )

    assert fx.evaluate(tuple([1] * (fx.n - 1) + [0]))[0] is False, (
        "C9: dropping the old unit must red"
    )

    full = _full(fx.n)
    zero = ArmResult("x", full, fx.cost(full), evaluations=5, work_per_tick=fx.work_per_tick(full))
    assert net_savings(zero, zero, horizon=100, eval_cost=1.0) < 0, "C10"

    m = Meter(fx)
    try:
        m.judge(full)
        raise AssertionError("C26: a pre-freeze judge read must red")
    except SelectionLeak:
        pass

    assert (
        _decide(constructed_seeds({("SA", "cost"): _smaller, ("SA", "future_ok"): False})).outcome
        != "SA"
    )
    assert (
        _decide(constructed_seeds({("SA", "cost"): _smaller, ("SA", "net"): -3.0})).outcome != "SA"
    )
    assert _decide(constructed_seeds({("random", "cost"): _smaller})).outcome == "RANDOM-WIN"
    assert (
        _decide(constructed_seeds({(a, "future_ok"): False for a in COMPETITORS})).outcome
        == "UNBUILT"
    )
    assert _decide(constructed_seeds({("SA", "cost"): _smaller})).outcome == "SA"
    assert decide_b(constructed_seeds(), instrument_gates_fired=False).outcome == "REFUSED"

    print(
        f"joint_mask_search self-check OK: mask_digest order-invariant + bit-sensitive; C8 SA == minimum "
        f"{fx.minimum()} on 8/8 seeds, < random paired t {c8.t:.3f} (exact FPR {c8.fpr}) at planted B={B} "
        f"(seed 0 random {res['random'].cost}); C7 judge red; C9 retention red; "
        f"C10 net<0; C26 pre-freeze judge refused; C22 SA / RANDOM-WIN / UNBUILT / REFUSED + two non-adoptions; "
        f"bar {T_BAR}"
    )


if __name__ == "__main__":
    main()
