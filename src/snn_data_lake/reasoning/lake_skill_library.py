"""
Lake-backed task-expert SKILL LIBRARY -- a Mixture-of-Task-Experts / skill-composition system
(Continual Prune-and-Select + Skill-Compositional Experts paradigm), research tier.

WHAT THIS IS. Train a small SPECIALIST expert per task, FREEZE it, CONSOLIDATE each frozen expert
into the substrate DATA LAKE as a named, content-addressable SKILL, and at inference ROUTE each
input to the right expert BY PROXIMITY (content-addressable, no task-id, no retraining). It is the
inverse of a shared model: instead of one network that must fit every task at once, we keep a BANK
of frozen experts and a proximity ROUTER that recalls which one to apply. The router substrate + the
durable skill store are the project's stdlib spike lake; only the experts (and the single
shared-model baseline) use torch/GPU.

THE REGIME MATTERS (stated honestly). Expert libraries WIN when tasks are DISSIMILAR: each task lives
in a DISTINCT, proximity-separable region of feature space with its OWN class geometry, so no single
linear/MLP boundary can serve all of them, but a per-task specialist can. On FEATURE-SHARING tasks
(e.g. MNIST digits, where one representation transfers) a single SHARED model wins instead -- the
project has already measured that. So this file deliberately builds the HETEROGENEOUS regime where
the paradigm pays off, and the key comparison (routed library vs one shared model) reports both.

PIPELINE (all deterministic; ASCII output; GPU-safe -- torch never touches the lake):
  1. TASKS. T tasks x K classes in a shared D-dim space (D<=64). Each task t has a random CENTER
     (its distinct region) and its K class-means sit along a CYCLICALLY-SHIFTED set of shared
     direction vectors (label k -> dir[(k+t)%K]). The cyclic shift makes the SAME label occupy
     DIFFERENT directions across tasks -> a single shared classifier cannot separate all labels,
     while a within-task expert (consistent labels) can. Tight gaussian blobs, seeded, train/test split.
  2. EXPERTS. Per task, train a torch nn.Linear(D,K) on that task's TRAIN split, PRUNE low-magnitude
     weights (compact expert -> report sparsity), then FREEZE (requires_grad off). Kept in a dict
     `bank[task_id]` (the expert bank). torch only here; outputs converted to python before the lake.
  3. CONSOLIDATE as a SKILL. Per task: the task CENTROID (mean train input, <=64-dim) is the proximity
     KEY -> LakeMemory.consolidate([centroid], payloads=[task_id]) stores it durably in the substrate
     spike lake + ProximityIndex; SleepConsolidator.wake(skill_id="expert_task_t", ...) registers the
     durable NAMED skill.
  4. ROUTER. At test time (no label, no task-id): LakeMemory.retrieve(input, k=1) -> nearest task's
     centroid -> its payload task_id -> apply bank[task_id]. Content-addressable expert SELECTION, no
     retraining.

MEASURED (main() prints a report + asserts): (a) specialist expert acc on own test set; (b) proximity
ROUTER selection acc over a mixed, task-id-hidden stream; (c) ROUTED end-to-end acc (route->apply->
predict, no retraining); (d) THE KEY COMPARISON: routed library vs ONE shared model of the same
architecture, on the heterogeneous stream -- the routed library BEATS the shared model; (e) skill
reuse/durability: the mixed stream solved by RECALL ONLY, then LakeMemory.durability_check() proves
the routing skills survive a lake close+reopen (durable in the data lake, not RAM); (f) a 2-expert
COMPOSITION (averaged logits) on an input between two task regions.

Run (GPU on the RTX 5070):
  NORD_GPU=1 PYTHONPATH=src python research/lake_skill_library.py
CPU (byte-identical logic, slower, needs the [train] extra for torch):
  PYTHONPATH=src python research/lake_skill_library.py
"""

import os
import random
import sys

# ---- import bootstrap: runnable as `python research/lake_skill_library.py` from the repo root ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    pass  # migrated: sys.path bootstrap removed (now an importable package)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    pass  # migrated: sys.path bootstrap removed (now an importable package)

from snn_data_lake.reasoning.brain_adapters.lake_memory import (
    LakeMemory,
)  # durable proximity router + skill store
from snn_data_lake.hippocampus.sleep_consolidation import (
    SleepConsolidator,
)  # named-skill registry API
from snn_data_lake.reasoning.torch_device import (
    resolve_device,
)  # opt-in GPU (NORD_GPU=1), CPU default

# ---- config (deterministic) -------------------------------------------------
SEED = 0
T_TASKS = 5  # number of tasks (chance router selection = 1/T = 0.20)
K_CLASSES = 3  # classes per task
D_DIM = 32  # feature width (MUST be <= 64: the lake/index channel width N)
N_TRAIN = 40  # train points per class per task
N_TEST = 20  # test points per class per task
DIR_SCALE = 0.16  # class-mean offset from the task center along a direction (blob separation)
NOISE_SIGMA = 0.03  # per-feature gaussian blob spread (tight -> classes separable)
PRUNE_FRAC = 0.30  # fraction of expert weights pruned to zero (compact expert)
TRAIN_STEPS = 300
LR = 0.05


# ---- 1. heterogeneous synthetic tasks (pure python; torch never sees this) --
def make_tasks(seed=SEED, T=T_TASKS, K=K_CLASSES, D=D_DIM, n_train=N_TRAIN, n_test=N_TEST):
    """T tasks, each K-class, in DISTINCT proximity-separable regions of [0,1]^D.

    Task t: a random CENTER (its region) + K shared direction vectors assigned by a per-task CYCLIC
    SHIFT (label k -> dir[(k+t)%K]). Cyclic shift => the same label occupies different directions
    across tasks -> a single shared classifier cannot separate all labels (the heterogeneous regime);
    an expert seeing only its own task's consistent labels separates its K tight blobs easily.
    Returns a list of task dicts with train/test splits (python lists) and the centroid router key."""
    rng = random.Random(seed)
    # K shared unit-ish direction vectors, reused across tasks (only the label<->dir mapping shifts)
    dirs = []
    for _ in range(K):
        v = [rng.uniform(-1.0, 1.0) for _ in range(D)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        dirs.append([x / norm for x in v])

    def clip01(x):
        return min(1.0, max(0.0, x))

    tasks = []
    for t in range(T):
        center = [rng.uniform(0.25, 0.85) for _ in range(D)]  # this task's distinct region
        class_means = []
        for k in range(K):
            d = dirs[(k + t) % K]  # cyclic shift -> cross-task conflict
            class_means.append([clip01(center[i] + DIR_SCALE * d[i]) for i in range(D)])

        def sample(cm):
            return [clip01(cm[i] + rng.gauss(0.0, NOISE_SIGMA)) for i in range(D)]

        train_X, train_y, test_X, test_y = [], [], [], []
        for k in range(K):
            for _ in range(n_train):
                train_X.append(sample(class_means[k]))
                train_y.append(k)
            for _ in range(n_test):
                test_X.append(sample(class_means[k]))
                test_y.append(k)
        # centroid = mean TRAIN input = the proximity KEY (<= D <= 64 dims); sits in the task's region
        centroid = [sum(x[i] for x in train_X) / len(train_X) for i in range(D)]
        tasks.append(
            {
                "center": center,
                "train_X": train_X,
                "train_y": train_y,
                "test_X": test_X,
                "test_y": test_y,
                "centroid": centroid,
            }
        )
    return tasks


# ---- 2. experts: train a specialist, prune, freeze (torch/GPU tier) ---------
def train_model(X, y, D, K, device, seed, steps=TRAIN_STEPS, lr=LR):
    """Train a torch nn.Linear(D,K) classifier on (X,y). Tensors placed EXPLICITLY on `device`
    (no set_default_device -> avoids the CPU-tensor-lands-on-GPU gotcha; the lake stays pure python)."""
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(D, K).to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(Xt), yt).backward()
        opt.step()
    return model


def prune_and_freeze(model, frac=PRUNE_FRAC):
    """Zero the smallest-|w| `frac` of the Linear weight (compact expert), then freeze all params.
    Returns the achieved sparsity (fraction of weights set to exactly zero)."""
    import torch

    with torch.no_grad():
        w = model.weight.data
        sparsity = 0.0
        if frac > 0.0 and w.numel() > 0:
            k = max(1, int(frac * w.numel()))
            thresh = w.abs().flatten().kthvalue(k).values.item()  # k-th smallest magnitude
            mask = (w.abs() > thresh).to(w.dtype)
            w.mul_(mask)
            sparsity = 1.0 - float(mask.mean().item())
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return sparsity


def predict_one(model, x, device):
    """Frozen-expert prediction for a single input (python list) -> python int class."""
    import torch

    with torch.no_grad():
        t = torch.tensor([x], dtype=torch.float32, device=device)
        return int(model(t).argmax(dim=1).item())


def logits_one(model, x, device):
    """Frozen-expert logits for a single input -> python list (for composition; torch->python here)."""
    import torch

    with torch.no_grad():
        t = torch.tensor([x], dtype=torch.float32, device=device)
        return model(t)[0].detach().cpu().tolist()


def acc_batch(model, X, y, device):
    """Accuracy of a frozen model over a batch (python lists)."""
    import torch

    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(t).argmax(dim=1).cpu().tolist()
    return sum(int(p == c) for p, c in zip(pred, y)) / len(y)


# ---- 4. proximity router over the durable lake ------------------------------
def route(mem, x):
    """Content-addressable expert SELECTION: nearest consolidated centroid's payload = its task_id."""
    got = mem.retrieve(x, k=1)
    return got[0] if got else -1


def _mean(bools):
    bools = list(bools)
    return sum(1 for b in bools if b) / len(bools) if bools else 0.0


def main():
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "torch not installed; the expert/shared-model tier needs the [gpu]/[train] extra. "
            "Skipping (the lake router/store are exercised by lake_memory.py)."
        )
        return

    assert D_DIM <= 64, f"router key width D={D_DIM} must be <= 64 (the lake/index channel width N)"
    device = resolve_device()  # 'cuda' if NORD_GPU=1 and available, else 'cpu'

    # ---- build the heterogeneous tasks ----
    tasks = make_tasks()
    chance = 1.0 / T_TASKS

    # ---- 2. EXPERT BANK: train + prune + freeze one specialist per task ----
    bank = {}
    sparsities = []
    for t in range(T_TASKS):
        m = train_model(
            tasks[t]["train_X"], tasks[t]["train_y"], D_DIM, K_CLASSES, device, seed=1000 + t
        )
        sparsities.append(prune_and_freeze(m))  # compact + frozen
        bank[t] = m

    # ---- 3. CONSOLIDATE each expert as a durable, content-addressable SKILL ----
    # proximity KEY = task centroid; payload = task_id; durable in the substrate spike lake + index.
    mem = LakeMemory(seed=SEED, probe=max(4, T_TASKS))  # probe >= n_medoids -> exact routing
    mem.consolidate([tasks[t]["centroid"] for t in range(T_TASKS)], payloads=list(range(T_TASKS)))
    # register the durable NAMED skills (the SleepConsolidator skill API); quiescent() is never called.
    consolidator = SleepConsolidator(seed=SEED)
    for t in range(T_TASKS):
        assembly = frozenset((t * 5 + j) % 24 for j in range(6))  # valid 6-unit subset of range(24)
        consolidator.wake(skill_id=f"expert_task_{t}", assemblies=[assembly], reps=2)

    # ---- the SHARED baseline: ONE model of the SAME architecture on ALL tasks pooled ----
    all_X = [x for t in range(T_TASKS) for x in tasks[t]["train_X"]]
    all_y = [y for t in range(T_TASKS) for y in tasks[t]["train_y"]]
    shared = train_model(all_X, all_y, D_DIM, K_CLASSES, device, seed=2000)
    shared.eval()

    # ---- the mixed, task-id-HIDDEN test stream (all tasks pooled, shuffled) ----
    stream = [
        (x, t, c) for t in range(T_TASKS) for x, c in zip(tasks[t]["test_X"], tasks[t]["test_y"])
    ]
    random.Random(7).shuffle(stream)

    # ---- (a) specialist experts on their OWN task test set ----
    spec_accs = [
        acc_batch(bank[t], tasks[t]["test_X"], tasks[t]["test_y"], device) for t in range(T_TASKS)
    ]

    # ---- (b) ROUTER selection acc + (c) ROUTED end-to-end acc (RECALL ONLY, no retraining) ----
    route_preds = [route(mem, x) for x, _, _ in stream]
    router_acc = _mean(rp == tt for rp, (_, tt, _) in zip(route_preds, stream))
    routed_correct = 0
    for (x, _tt, cc), rp in zip(stream, route_preds):
        expert = bank.get(rp, bank[0])  # selected purely by proximity
        routed_correct += int(predict_one(expert, x, device) == cc)
    routed_acc = routed_correct / len(stream)

    # ---- (d) THE KEY COMPARISON: routed library vs the ONE shared model ----
    shared_acc = _mean(predict_one(shared, x, device) == cc for x, _t, cc in stream)
    delta = routed_acc - shared_acc

    # ---- (e) SKILL REUSE / durability: the routing skills survive a lake close+reopen ----
    dur = mem.durability_check()  # close + reopen the .spkl
    route_preds_after = [
        route(mem, x) for x, _, _ in stream
    ]  # recall-only routing after the reopen
    recall_only_stable = route_preds_after == route_preds

    # ---- (f) COMPOSITION: average two experts' logits on an input between two task regions ----
    mid = [(tasks[0]["center"][i] + tasks[1]["center"][i]) / 2.0 for i in range(D_DIM)]
    comp = [
        (a + b) / 2.0
        for a, b in zip(logits_one(bank[0], mid, device), logits_one(bank[1], mid, device))
    ]
    comp_class = comp.index(max(comp))

    # ---- report ----
    print("=" * 78)
    print("LAKE SKILL LIBRARY -- Mixture-of-Task-Experts, proximity-routed over the data lake")
    print("=" * 78)
    print(
        f"regime        : {T_TASKS} HETEROGENEOUS tasks x {K_CLASSES} classes in R^{D_DIM} "
        f"(distinct regions, cyclically-shifted class geometry)"
    )
    print(
        f"device        : {device}   experts: nn.Linear({D_DIM},{K_CLASSES}) each, pruned "
        f"{100 * sum(sparsities) / len(sparsities):.0f}% then frozen"
    )
    print(
        f"durable lake  : {os.path.basename(mem.path)}  {mem.n_stored} skill keys "
        f"({mem.hub.n_events()} spike events); named skills registered = {len(consolidator.skills)}"
    )
    print("-" * 78)
    print(
        f"(a) specialist experts (own test) : {[f'{a:.2f}' for a in spec_accs]}  "
        f"mean {sum(spec_accs) / len(spec_accs):.2f}"
    )
    print(
        f"(b) proximity ROUTER selection    : {router_acc:.2f}   (chance 1/{T_TASKS} = {chance:.2f}) "
        f"<- picks the correct task's expert with NO task-id"
    )
    print(
        f"(c) ROUTED end-to-end (recall)    : {routed_acc:.2f}   (route -> apply expert -> predict, "
        f"no retraining)"
    )
    print("-" * 78)
    print(f"(d) KEY COMPARISON on the heterogeneous mixed stream:")
    print(f"      routed expert library       : {routed_acc:.2f}")
    print(f"      ONE shared model (same arch) : {shared_acc:.2f}")
    print(
        f"      delta (library - shared)     : {delta:+.2f}   <- experts WIN when tasks are dissimilar"
    )
    print("-" * 78)
    print(
        f"(e) durability : close+reopen .spkl -> events {dur['events_before']} -> "
        f"{dur['events_after']}, patterns match = {dur['patterns_match']}, ok = {dur['ok']}"
    )
    print(
        f"    recall-only routing stable across the reopen = {recall_only_stable} "
        f"(the routing skills are durable in the lake, not RAM)"
    )
    print(
        f"(f) composition: mean of expert0+expert1 logits at the task0/task1 midpoint -> "
        f"class {comp_class} (illustrative, not asserted)"
    )
    print("-" * 78)
    print(
        "HONEST NOTE: this is the HETEROGENEOUS-task regime, where a per-task expert library beats a"
    )
    print(
        "single shared model. On FEATURE-SHARING tasks (e.g. MNIST digits) a shared model wins instead"
    )
    print(
        "-- one representation transfers across classes -- which the project has already measured. The"
    )
    print("expert weights live as torch objects in the bank; the LAKE durably stores the content-")
    print(
        "addressable ROUTING skill (centroid key -> task_id) + the named SleepConsolidator skill record."
    )
    print("=" * 78)

    # ---- self-checks (deterministic asserts) ----
    # (i) proximity router selects the right expert far above chance (content-addressable, not guessing)
    assert router_acc > chance + 0.4, (
        f"router selection {router_acc:.2f} not clearly above chance {chance:.2f} (need > {chance + 0.4:.2f})"
    )
    # specialists actually specialize (sanity that the experts are real)
    assert sum(spec_accs) / len(spec_accs) > 0.8, (
        f"specialist experts too weak (mean {sum(spec_accs) / len(spec_accs):.2f})"
    )
    # (ii) the routed library BEATS the one shared model -- the paradigm's win on heterogeneous tasks
    assert routed_acc > shared_acc, (
        f"routed library ({routed_acc:.2f}) must beat the shared model ({shared_acc:.2f}) on heterogeneous tasks"
    )
    # (iii) the routing skills survive the lake close+reopen (durable in the data lake, not RAM)
    assert dur["ok"], f"durability check failed: {dur}"
    assert dur["events_after"] == dur["events_before"] > 0, (
        "skill keys did not survive the lake reopen"
    )
    assert recall_only_stable, "recall-only routing changed after reopening the durable lake"
    # every expert was registered as a durable named skill
    assert len(consolidator.skills) == T_TASKS, "not every expert was registered as a named skill"

    mem.close(remove=True)
    print(
        f"self-check OK: proximity router selects experts {router_acc:.0%} >> chance {chance:.0%}; "
        f"routed library {routed_acc:.2f} > shared model {shared_acc:.2f} (delta {delta:+.2f}, "
        f"experts win on heterogeneous tasks); {mem.n_stored} routing skills + "
        f"{dur['events_after']} events survive the lake close+reopen (durable, recall-only)"
    )


if __name__ == "__main__":
    main()
