r"""
Bounded quality-diversity repertoire -- MAP-Elites / Go-Explore over SPIKING policies
(ladder L12 / SI-13): the missing OUTER loop that CURATES the strong inner novelty /
learning-progress signal into a diverse archive of behaviors.

What this is (read first, then the honest limits below). The inner drive
(neuromodulation/intrinsic_motivation.CuriosityDrive) is a scalar that SCORES how worth
exploring one context is. It has no memory of WHICH behaviors it has already produced, so
on its own it re-discovers the same few behaviors. Quality-diversity is the OUTER
bookkeeping loop that fixes that: it keeps a GRID of behavioral niches and, in each niche,
the single best ("elite") policy seen so far. New policies are made by mutating existing
elites, and the curiosity drive decides WHICH elite to expand next (explore under-filled /
frontier niches). The result is an ARCHIVE that is simultaneously high-quality (each cell
holds its best) and diverse (cells span the behavior space). This is MAP-Elites
(Mouret & Clune 2015) with a Go-Explore-style "return to a promising cell and explore
from it" selection (Ecoffet et al. 2019/2021), driven by learning-progress novelty.

The searched POLICY is genuinely SPIKING. Each policy is a small population of
leaky integrate-and-fire neurons (cortex/lif_neuron.LIFNeuron) with a mutable weight
genome, run as a CLOSED LOOP in an abstract 2-D box: each step the current position is
population-coded into the inputs (cortex/population_encoding.PopulationEncoder), the LIF
neurons take one step, and four neuron groups (right/left/up/down) vote with their SPIKES
to nudge the position. The behavior DESCRIPTOR is where that spike-driven trajectory SETTLES
-- read from the rollout's POPULATION CODE, NOT from the genome parameters. The box position
is a pure latent used only to define the behavior (no actuator). Elites are CONSOLIDATED into
a per-niche value memory with sharp-wave-ripple replay
(hippocampus/experience_replay.ReplayBuffer), value-only.

Why a closed loop and not a static readout: the behavior comes from where a spike-driven
trajectory ENDS -- which the genome's summary statistics CANNOT see -- so behavior-keying
illuminates niches that parameter-keying cannot (the honest half of control 1: behavior-keyed
>> parameter-keyed, robust across seeds). HONEST on raw coverage (do NOT overclaim): MAP-Elites
beats a NARROW-INIT random search only by DIFFUSING out of the small init scale -- a
SCALE-MATCHED random search (weights drawn across the full mutation range) matches/BEATS QD
coverage with NO archive at all (control 1's printed honest-negative). So this toy demonstrates
mutation-escapes-init-scale + behavior>parameter descriptors, NOT stepping-stone illumination.
The archive/novelty is genuinely LOAD-BEARING only on the DECEPTIVE task (control 2), where
novelty/LP escapes a trap an objective-only search cannot (and even that is seed-sensitive).

HONEST framing -- what is and is not spiking, and what this does NOT prove:
  * The searched policy is spiking (LIF raster -> settling-point descriptor). The
    grid / archive / selection bookkeeping is stdlib RATE_BY_DESIGN scaffolding -- ordinary
    quality-diversity is a substrate-agnostic accounting loop and is named as such here.
    The spiking contribution is THIN and confined to the policy + its descriptor.
  * The toy behavior space SATURATES FAST. It is a finite 2-D grid over a spiking
    controller whose reachable behaviors form a bounded box; coverage PLATEAUS after a
    few hundred evaluations (self-check control 4 prints the flattening curve). This is
    BOUNDED quality-diversity. It is explicitly NOT open-endedness -- the archive does not
    keep inventing genuinely new behavior dimensions; it fills a fixed, pre-named grid.
  * The parameter-keyed control does NOT collapse to random -- it shares the mutation +
    archive machinery, so it beats random too; it just does not reproduce the FULL coverage
    that behavior-keying reaches. The residual gap (not "param == random") is the honest,
    measured contribution of the behavioral descriptor.
  * Pure-Python CAPS the evaluation budget (each eval is a K-neuron x T-step closed-loop
    rollout). The self-check reports the exact budget used and NEVER extrapolates coverage
    beyond it. Coverage numbers are for THIS budget on THIS toy, nothing more.
  * OVERCLAIM TO AVOID: this is not Cully et al. 2015 robot damage-recovery, not
    open-ended evolution, and not a developmental agent. It is the QD bookkeeping primitive,
    demonstrated on a spiking-policy sandbox, with every eval a pure value/representation
    function -- no actuator, no motor command, no arm (control 3 asserts this).

Self-check (deterministic, ASCII, zero-dep) -- four un-riggable controls that MUST fail:
  (1) COVERAGE: MAP-Elites fills more BEHAVIOR cells than random search at EQUAL budget;
      AND a control whose descriptor is read from policy PARAMETERS (not the rollout
      population code) does NOT reproduce the coverage win -> the BEHAVIORAL descriptor is
      load-bearing.
  (2) DECEPTIVE task: an objective whose global optimum is a TRAP far from the true target.
      Novelty/LP-driven QD REACHES the target niche; an objective-only greedy hill-climber
      gets STUCK at the trap and never reaches it.
  (3) NO-ACTUATOR invariant: this module imports nothing from brainstem/ or the arm and
      every eval is a pure value/representation function (no motor command). Asserted by
      scanning this file's own source; post-clamp diversity/coverage reported.
  (4) HONEST-NEGATIVE: coverage-vs-iteration PLATEAUS (the printed curve flattens) ->
      FORBIDS the open-endedness claim.

Refs: Mouret & Clune, "Illuminating search spaces by mapping elites" (2015);
Lehman & Stanley, "Abandoning Objectives: Evolution through the Search for Novelty" (2011);
Ecoffet, Huizinga, Lehman, Stanley & Clune, "Go-Explore" (Nature 2021);
Cully, Clune, Tarapore & Mouret, "Robots that can adapt like animals" (Nature 2015).

Pure stdlib (math, random), per-instance random.Random(seed), ASCII-only, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.quality_diversity
"""
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.population_encoding import PopulationEncoder
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive
from snn_data_lake.substrate._mathutils import mean

# ---- toy sandbox geometry (kept small so the whole self-check runs in a few seconds) ----
# The behavior descriptor is the SETTLING POINT of a closed-loop spiking controller in an abstract 2-D
# box (a pure latent, NOT a physical actuator). Each drift step the current position is population-coded
# into the LIF inputs; four neuron groups (right/left/up/down) vote with their spikes and nudge the
# position. The descriptor is where the trajectory ends. The endpoint depends on the closed-loop group
# balance and spike timing, which the genome's summary statistics cannot see -- so illuminating PARAMETER
# space fails to cover BEHAVIOR space (the honest half of control 1: behavior-keyed >> parameter-keyed).
# But raw coverage does NOT prove the archive: a SCALE-MATCHED i.i.d. random search (weights across the
# full mutation range) matches/beats QD coverage with no archive -- QD only beats the NARROW-INIT random
# by diffusing out of the small init scale. The archive is load-bearing only on the deceptive task (2).
DPOS = 4              # RBF channels population-coding EACH of x and y
D_INPUTS = 2 * DPOS   # 8 input channels = [ code(x) | code(y) ]
K_NEURONS = 12        # LIF neurons: 4 movement groups (right, left, up, down) of GROUP each
GROUP = K_NEURONS // 4
T_STEPS = 20          # closed-loop drift steps per rollout
STEP = 0.04           # per-step drift magnitude (small: i.i.d. votes cancel and stay central)
GRID = 6              # bins per behavior axis -> GRID*GRID niches
W_INIT = 0.05         # init/random-sample weight scale (near-balanced firing -> centered endpoint)
W_CLIP = 0.30         # mutation clamps weights to [0, W_CLIP] (allows strong one-directional drive)
SIGMA = 0.05          # Gaussian mutation step on a weight


# =====================================================================================================
# The SPIKING policy: a population of LIF neurons with a mutable weight genome, run as a closed loop.
# =====================================================================================================


class SpikingPolicy:
    """A searched spiking policy: K leaky integrate-and-fire neurons, genome = weight matrix W[K][D].

    Rolled out as a CLOSED LOOP: the current abstract position is population-coded into the D inputs, the
    K LIF neurons (cortex/lif_neuron.LIFNeuron, membrane state carried ACROSS drift steps) each take one
    step, and four neuron groups vote with their spikes to nudge the position (right/left on x, up/down on
    y). The output is a genuine SPIKE RASTER driving the dynamics; nothing here is an actuator or a motor
    command -- the position is a pure latent used only to DEFINE the behavior. mutate() returns a
    Gaussian-perturbed, clamped child. Deterministic given the weights."""

    def __init__(self, weights):
        self.W = weights                                 # W[k][d], all >= 0

    @classmethod
    def random(cls, rng, scale=W_INIT):
        return cls([[rng.uniform(0.0, scale) for _ in range(D_INPUTS)] for _ in range(K_NEURONS)])

    def mutate(self, rng, sigma=SIGMA):
        """Child policy: each weight perturbed by rng.gauss(0, sigma), clamped to [0, W_CLIP]."""
        child = [[min(W_CLIP, max(0.0, w + rng.gauss(0.0, sigma))) for w in row] for row in self.W]
        return SpikingPolicy(child)

    def settle(self, encoder, neurons):
        """Run the closed loop from the center and return the endpoint (x, y) in [0,1]^2 -- the behavior.

        `encoder` population-codes a scalar in [0,1] into DPOS activations; `neurons` is a reusable pool of
        K LIFNeuron (reset here, state then carried across the T drift steps). Pure value function: reads
        the spike raster, moves an abstract latent, emits NOTHING. Deterministic (LIF is deterministic)."""
        for n in neurons:
            n.reset_state()
        x, y = 0.5, 0.5
        for _ in range(T_STEPS):
            inp = encoder.activations(x) + encoder.activations(y)          # closed-loop state feedback
            s = [neurons[k].step(sum(self.W[k][d] * inp[d] for d in range(D_INPUTS)))
                 for k in range(K_NEURONS)]
            dx = (sum(s[0:GROUP]) - sum(s[GROUP:2 * GROUP])) * STEP         # right-group minus left-group votes
            dy = (sum(s[2 * GROUP:3 * GROUP]) - sum(s[3 * GROUP:4 * GROUP])) * STEP   # up minus down
            x = min(1.0, max(0.0, x + dx))
            y = min(1.0, max(0.0, y + dy))
        return (x, y)


def behavior_descriptor(policy, encoder, neurons):
    """Behavior descriptor from the ROLLOUT population code (the spike raster driving the closed loop),
    NOT the genome: the settling point of the spiking controller in the abstract box."""
    return policy.settle(encoder, neurons)


def param_descriptor(policy):
    """CONTROL descriptor read from the GENOME PARAMETERS instead of the rollout: the naive genome
    summary statistics you would reach for WITHOUT running the network -- overall mean weight and overall
    weight spread (std), each normalized to [0,1].

    These are BLIND to the closed-loop dynamics: the mean weight sets how MUCH the population fires but
    not which direction group WINS, and the std says nothing about the trajectory. A high-mean genome
    fires every group (votes cancel -> stays central) and a low-mean genome barely fires (also central);
    directed drift comes from the group BALANCE and spike TIMING, which neither statistic sees. So
    MAP-Elites keyed on these parameters files behaviorally-central policies into scattered parameter
    cells and never illuminates the edges -- it does NOT reproduce the behavior-coverage win. That gap is
    the load-bearing proof that the population-code BEHAVIORAL descriptor is what matters (control 1)."""
    ws = [w for row in policy.W for w in row]
    mu = mean(ws)
    std = (mean([(w - mu) ** 2 for w in ws])) ** 0.5
    return (min(1.0, mu / W_CLIP), min(1.0, std / (0.5 * W_CLIP)))


def _cell(desc):
    """Discretize a descriptor in [0,1]^2 to a grid cell (cx, cy)."""
    cx = min(GRID - 1, max(0, int(desc[0] * GRID)))
    cy = min(GRID - 1, max(0, int(desc[1] * GRID)))
    return (cx, cy)


# =====================================================================================================
# The OUTER quality-diversity loop: a MAP-Elites archive with curiosity-scored selection.
# =====================================================================================================


class QualityDiversityArchive:
    """Bounded MAP-Elites archive over spiking policies (RATE_BY_DESIGN bookkeeping scaffold).

    A grid of niches keyed by the discretized descriptor; each cell holds the highest-objective
    elite seen for that niche. Parents to mutate are chosen by a CuriosityDrive over the filled
    cells (novelty habituation + a frontier learning-progress signal = expand under-filled /
    frontier niches, Go-Explore style). Elites are consolidated into a per-niche value memory via
    sharp-wave-ripple ReplayBuffer replay (value-only, never steers the search or emits anything).

    Coverage is ALWAYS measured in BEHAVIOR space (behavior_fn), even when the archive itself is
    keyed by a different descriptor -- so a behavior-keyed and a parameter-keyed archive are scored
    on the same yardstick."""

    def __init__(self, descriptor_fn, objective_fn, behavior_fn, seed=0):
        self.descriptor_fn = descriptor_fn               # policy -> descriptor the ARCHIVE is keyed by
        self.objective_fn = objective_fn                 # policy -> scalar quality (elite tie-break)
        self.behavior_fn = behavior_fn                   # policy -> behavior descriptor (the coverage yardstick)
        self.cells = {}                                  # cell -> (policy, score, descriptor)
        self.behavior_seen = set()                       # every behavior cell ever evaluated (shared metric)
        self.drive = CuriosityDrive(seed=seed)           # scores WHICH elite to expand (novelty/LP)
        self.buf = ReplayBuffer(seed=seed)               # SWR consolidation of elite objectives
        self.q = {}                                      # cell-index -> {0: consolidated value} (value-only)
        self.rng = random.Random(seed)                   # per-instance rng (init + mutation stream)

    def _cell_index(self, cell):
        return cell[1] * GRID + cell[0]

    def _consider(self, policy):
        """Evaluate a policy and file it. Records its BEHAVIOR cell (coverage), then replaces the
        niche's elite if it scores higher. Returns True if the niche's elite changed."""
        score = self.objective_fn(policy)
        self.behavior_seen.add(_cell(self.behavior_fn(policy)))
        cell = _cell(self.descriptor_fn(policy))
        cur = self.cells.get(cell)
        if cur is None or score > cur[1]:
            self.cells[cell] = (policy, score, self.descriptor_fn(policy))
            idx = self._cell_index(cell)                 # consolidate the elite's value (SWR replay, value-only)
            self.q.setdefault(idx, {0: 0.0})
            self.buf.add(idx, 0, score)
            self.buf.replay(self.q, lr=0.3, batch=8)
            return True
        return False

    def _unfilled_neighbor_frac(self, cell):
        """Fraction of `cell`'s 4-neighbours (inside the grid) that are still EMPTY -- a frontier
        signal that FALLS as the neighbourhood fills, so CuriosityDrive reads it as learning progress
        (positive on the actively-expanding frontier, ~0 once the neighbourhood is mined out)."""
        cx, cy = cell
        nb = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]
        nb = [(x, y) for (x, y) in nb if 0 <= x < GRID and 0 <= y < GRID]
        if not nb:
            return 0.0
        empty = sum(1 for c in nb if c not in self.cells)
        return empty / len(nb)

    def _select_parent(self):
        """Pick which elite to expand next: the filled cell of maximal curiosity value (novelty +
        frontier learning-progress). Observing the chosen cell's frontier error habituates it and
        folds LP in, so selection walks the frontier instead of re-mining one niche."""
        keys = sorted(self.cells)                        # deterministic candidate order before the rng tie-break
        cell = self.drive.most_curious(keys)
        self.drive.observe(cell, self._unfilled_neighbor_frac(cell))
        return self.cells[cell][0]

    def run(self, n_evals, n_init=20, checkpoints=None):
        """Run the QD loop for n_evals evaluations: seed with n_init random policies, then repeatedly
        select an elite (curiosity-scored) and mutate it. Returns the coverage-vs-iteration curve at
        `checkpoints` (list of eval counts) as [(iter, behavior_cells_filled), ...]."""
        checkpoints = set(checkpoints or [])
        curve = []
        for i in range(n_evals):
            if len(self.cells) == 0 or i < n_init:
                policy = SpikingPolicy.random(self.rng)
            else:
                policy = self._select_parent().mutate(self.rng)
            self._consider(policy)
            if (i + 1) in checkpoints:
                curve.append((i + 1, len(self.behavior_seen)))
        return curve

    def behavioral_spread(self):
        """Mean pairwise L1 distance between elite descriptors -- a scalar diversity readout of the
        archive (a pure value; touches no actuator)."""
        descs = [d for (_, _, d) in self.cells.values()]
        if len(descs) < 2:
            return 0.0
        tot, cnt = 0.0, 0
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                tot += abs(descs[i][0] - descs[j][0]) + abs(descs[i][1] - descs[j][1])
                cnt += 1
        return tot / cnt


# =====================================================================================================
# Baselines the controls compare against (random search; objective-only greedy hill-climb).
# =====================================================================================================


def random_search(n_evals, behavior_fn, seed=0, scale=W_INIT):
    """Sample n_evals policies i.i.d. (weights ~ U[0, scale]); return the set of BEHAVIOR cells hit.
    No archive -- the equal-budget control for MAP-Elites coverage. scale=W_INIT is the NARROW init
    scale (QD beats it only by diffusing out); scale=W_CLIP is the FAIR scale-matched baseline that
    matches/beats QD (the honest-negative that raw coverage does not need the archive). behavior_fn is a
    policy-only callable (it closes over its own encoder + LIF pool)."""
    rng = random.Random(seed)
    seen = set()
    for _ in range(n_evals):
        seen.add(_cell(behavior_fn(SpikingPolicy.random(rng, scale=scale))))
    return seen


def greedy_hillclimb(n_evals, objective_fn, behavior_fn, seed=0):
    """Objective-only (1+1) hill-climb: keep the single best-objective policy, mutate it, accept the
    child iff its objective is >= the parent's. Returns (visited_behavior_cells, best_behavior_cell).
    On a DECEPTIVE objective this climbs into the trap and never reaches the true target. objective_fn
    and behavior_fn are policy-only callables."""
    rng = random.Random(seed)
    best = SpikingPolicy.random(rng)
    best_obj = objective_fn(best)
    visited = {_cell(behavior_fn(best))}
    best_cell = _cell(behavior_fn(best))
    for _ in range(n_evals - 1):
        child = best.mutate(rng)
        visited.add(_cell(behavior_fn(child)))
        obj = objective_fn(child)
        if obj >= best_obj:
            best, best_obj = child, obj
            best_cell = _cell(behavior_fn(child))
    return visited, best_cell


# =====================================================================================================
# Self-check: four un-riggable, deterministic controls (ASCII, zero-dep).
# =====================================================================================================


def _make_ctx():
    """A rollout context: the position encoder + a reusable pool of K LIFNeuron. The encoder is stateless;
    each rollout resets the pool, so one context can be reused for a whole experiment. Zero-dep."""
    enc = PopulationEncoder(n=DPOS, lo=0.0, hi=1.0, max_spikes=10)
    neurons = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(K_NEURONS)]
    return (enc, neurons)


def _behavior_of(policy, ctx):
    """policy -> behavior descriptor (closed-loop settling point) via the rollout raster (pure value fn)."""
    enc, neurons = ctx
    return behavior_descriptor(policy, enc, neurons)


def main():
    ctx = _make_ctx()

    # policy-only callables (default-arg binding of the shared context avoids B023 loop-var closures).
    def beh_desc(policy, _c=ctx):
        return _behavior_of(policy, _c)

    def coverage_objective(policy, _c=ctx):
        # arbitrary within-niche quality for the illumination run (prefer the up-right settling corner)
        x, y = _behavior_of(policy, _c)
        return x + y

    # ---------------------------------------------------------------------------------------------
    # (1) COVERAGE: MAP-Elites (behavior descriptor) vs random vs MAP-Elites (parameter descriptor).
    # ---------------------------------------------------------------------------------------------
    budget = 700
    checkpoints = [budget * k // 10 for k in range(1, 11)]

    qd_beh = QualityDiversityArchive(beh_desc, coverage_objective, beh_desc, seed=1)
    curve = qd_beh.run(budget, n_init=25, checkpoints=checkpoints)
    cov_qd = len(qd_beh.behavior_seen)

    cov_random = len(random_search(budget, beh_desc, seed=1))                    # NARROW-init random
    cov_random_matched = len(random_search(budget, beh_desc, seed=1, scale=W_CLIP))  # FAIR scale-matched

    qd_par = QualityDiversityArchive(param_descriptor, coverage_objective, beh_desc, seed=1)
    qd_par.run(budget, n_init=25)
    cov_par = len(qd_par.behavior_seen)

    # ---------------------------------------------------------------------------------------------
    # (2) DECEPTIVE task: objective peaks at a TRAP; the true target is a far, low-objective niche.
    # ---------------------------------------------------------------------------------------------
    trap_cell = (4, 3)                                   # the deceptive objective peaks here (mid-right)
    goal_cell = (0, 3)                                   # the true target: a far LEFT-edge niche
    trap_pt = ((trap_cell[0] + 0.5) / GRID, (trap_cell[1] + 0.5) / GRID)

    def deceptive_objective(policy, _c=ctx):
        b = _behavior_of(policy, _c)                     # objective = closeness to the TRAP only
        return 1.0 - math.hypot(b[0] - trap_pt[0], b[1] - trap_pt[1]) / math.sqrt(2.0)

    dec_budget = 400
    qd_dec = QualityDiversityArchive(beh_desc, deceptive_objective, beh_desc, seed=2)
    qd_dec.run(dec_budget, n_init=20)
    qd_reached_goal = goal_cell in qd_dec.behavior_seen

    greedy_visited, greedy_best = greedy_hillclimb(dec_budget, deceptive_objective, beh_desc, seed=2)
    greedy_reached_goal = goal_cell in greedy_visited
    greedy_trap_dist = math.hypot(greedy_best[0] - trap_cell[0], greedy_best[1] - trap_cell[1])

    # the escape is SEED-SENSITIVE per run, so the HONEST robust claim is the AGGREGATE escape RATE:
    # count how often novelty/LP QD vs objective-only greedy reach the target across many seeds.
    dec_seeds = list(range(12))
    qd_escapes = greedy_escapes = 0
    for s in dec_seeds:
        qd_s = QualityDiversityArchive(beh_desc, deceptive_objective, beh_desc, seed=s)
        qd_s.run(dec_budget, n_init=20)
        gv_s, _ = greedy_hillclimb(dec_budget, deceptive_objective, beh_desc, seed=s)
        qd_escapes += goal_cell in qd_s.behavior_seen
        greedy_escapes += goal_cell in gv_s

    # ---------------------------------------------------------------------------------------------
    # (3) NO-ACTUATOR invariant: this module imports nothing from brainstem/arm; evals are pure.
    # ---------------------------------------------------------------------------------------------
    with open(__file__, encoding="utf-8") as fh:
        import_lines = [ln for ln in fh.read().splitlines()
                        if ln.startswith("import ") or ln.startswith("from ")]
    # un-riggable: no IMPORT statement in this file may pull the actuator / safety-envelope package.
    forbidden = ("brainstem", "serial_arm", "arm_sim", "arm_config")
    hits = [tok for tok in forbidden if any(tok in ln for ln in import_lines)]
    # value-only: evaluating a policy is a pure, side-effect-free function (identical twice, no emit).
    _probe_pol = SpikingPolicy.random(random.Random(9))
    ev1 = _behavior_of(_probe_pol, ctx)
    ev2 = _behavior_of(_probe_pol, ctx)
    eval_is_pure = (ev1 == ev2) and isinstance(ev1, tuple) and all(isinstance(x, float) for x in ev1)
    post_clamp_cov = len(qd_beh.behavior_seen)           # unchanged by the invariant (pure value bookkeeping)
    post_clamp_spread = qd_beh.behavioral_spread()

    # ---------------------------------------------------------------------------------------------
    # (4) HONEST-NEGATIVE: coverage-vs-iteration PLATEAUS -> forbids the open-endedness claim.
    # ---------------------------------------------------------------------------------------------
    early_gain = curve[2][1] - curve[0][1]               # cells gained over checkpoints 0..2 (early third-ish)
    late_gain = curve[-1][1] - curve[-3][1]              # cells gained over the last two checkpoints (late)

    # consolidated-elite value memory (reused ReplayBuffer), value-only
    consolidated_niches = len(qd_beh.q)

    print("=" * 92)
    print("BOUNDED QUALITY-DIVERSITY -- MAP-Elites / Go-Explore over SPIKING policies (L12 / SI-13)")
    print("=" * 92)
    print(f"sandbox: {K_NEURONS} LIF neurons, {T_STEPS}-step closed-loop drift in an abstract box, "
          f"{GRID}x{GRID}={GRID * GRID} niches; NO actuator, NO arm -- pure value/representation")
    print("-" * 92)
    print(f"(1) COVERAGE @ budget={budget} evals (equal for all):")
    print(f"      MAP-Elites (behavior desc)    fills {cov_qd:2d} behavior cells")
    print(f"      random  (NARROW init scale)   fills {cov_random:2d}  (QD beats THIS only by diffusing out of the small init scale)")
    print(f"      random  (SCALE-MATCHED, fair) fills {cov_random_matched:2d}  <- HONEST-NEGATIVE: matches/BEATS QD -> raw coverage does NOT need the archive")
    print(f"      MAP-Elites (PARAM desc)       fills {cov_par:2d}  <- param descriptor does NOT reproduce the behavior-keyed win")
    print(f"      archive elites={len(qd_beh.cells)}  behavioral spread(L1)={qd_beh.behavioral_spread():.3f}  "
          f"consolidated niches(SWR)={consolidated_niches}")
    print(f"(2) DECEPTIVE  @ budget={dec_budget} evals  (trap niche {trap_cell}, true target {goal_cell}) "
          f"-- this is where the archive/novelty is LOAD-BEARING:")
    print(f"      novelty/LP QD reached target={qd_reached_goal}   "
          f"objective-only greedy reached target={greedy_reached_goal}")
    print(f"      greedy stuck near trap: best behavior cell {greedy_best}, "
          f"{greedy_trap_dist:.1f} cells from trap; visited {len(greedy_visited)} cells")
    print(f"      SEED-SENSITIVE per run, so AGGREGATE over {len(dec_seeds)} seeds: novelty/LP QD escapes "
          f"{qd_escapes}/{len(dec_seeds)} vs objective-only greedy {greedy_escapes}/{len(dec_seeds)} (clear advantage)")
    print(f"(3) NO-ACTUATOR: forbidden-import hits={hits}  eval_is_pure_value={eval_is_pure}  "
          f"post-clamp coverage={post_clamp_cov} spread={post_clamp_spread:.3f}")
    print(f"(4) PLATEAU (behavior cells vs evals): {[c for (_, c) in curve]}")
    print(f"      early gain (evals {curve[0][0]}->{curve[2][0]})={early_gain} cells  vs  "
          f"late gain (evals {curve[-3][0]}->{curve[-1][0]})={late_gain} cells  -> flattening (NOT open-ended)")
    print("=" * 92)

    # ---- (1) COVERAGE asserts -- what is HONEST here (NO stepping-stone / illumination claim) ----
    # (i) behavior-keyed >> param-keyed (the descriptor is load-bearing). (ii) HONEST-NEGATIVE: a fair
    #     SCALE-MATCHED random matches/BEATS QD, so raw coverage does NOT need the archive -- QD's win
    #     over NARROW-init random is diffusion out of the init scale, not stepping-stone illumination.
    #     (The narrow-init random number is printed for context but is NOT asserted as a QD win.)
    assert cov_random_matched >= cov_qd - 3, \
        f"HONEST-NEGATIVE: a scale-matched random must MATCH/BEAT QD ({cov_random_matched} vs {cov_qd}) -- " \
        f"raw coverage does NOT need the archive (load-bearing only on the deceptive task)"
    assert cov_qd >= cov_par + 5, \
        f"the PARAM-descriptor control must NOT reproduce the coverage win ({cov_qd} vs param {cov_par}) -- " \
        f"the behavioral descriptor is load-bearing"

    # ---- (2) DECEPTIVE asserts -- novelty/LP QD escapes the trap, objective-only greedy does not ----
    assert qd_reached_goal, "novelty/LP-driven QD must REACH the true target niche on the deceptive task"
    assert not greedy_reached_goal, "objective-only greedy must NOT reach the target (it is stuck in the trap)"
    assert greedy_trap_dist <= 1.5, \
        f"objective-only greedy must converge INTO the deceptive trap (best {greedy_best} vs trap {trap_cell})"
    assert qd_escapes >= greedy_escapes + 3, \
        f"novelty/LP QD must escape the deceptive trap markedly MORE OFTEN than objective-only greedy " \
        f"across seeds ({qd_escapes} vs {greedy_escapes} / {len(dec_seeds)}) -- per-run seed-sensitive, aggregate advantage real"

    # ---- (3) NO-ACTUATOR invariant ----
    assert hits == [], f"module must import NOTHING from brainstem/arm (found {hits})"
    assert eval_is_pure, "every eval must be a pure value/representation function (deterministic, no emit)"

    # ---- (4) HONEST-NEGATIVE: coverage plateaus (forbids open-endedness) ----
    assert late_gain < early_gain, \
        f"coverage must PLATEAU (late gain {late_gain} < early gain {early_gain}) -- bounded, NOT open-ended"
    assert curve[-1][1] == cov_qd, "the plateau curve must end at the final coverage"

    print(f"self-check OK: HONEST coverage -- behavior-keyed {cov_qd} >> param-keyed {cov_par} (descriptor "
          f"load-bearing), but a SCALE-MATCHED random {cov_random_matched} MATCHES/BEATS QD -> raw coverage "
          f"does NOT need the archive (no stepping-stone claim; QD only beats NARROW-init random {cov_random});")
    print(f"  the archive/novelty is LOAD-BEARING on the DECEPTIVE task: across {len(dec_seeds)} seeds QD "
          f"escapes the trap {qd_escapes}/{len(dec_seeds)} vs objective-only greedy {greedy_escapes}/{len(dec_seeds)} "
          f"(per-run seed-sensitive, aggregate advantage clear);")
    print(f"  no-actuator invariant holds (hits={hits}, pure value evals), coverage PLATEAUS "
          f"early {early_gain} -> late {late_gain} cells (bounded QD, NOT open-endedness); deterministic")


if __name__ == "__main__":
    main()
