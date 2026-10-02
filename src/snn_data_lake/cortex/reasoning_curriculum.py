r"""
Reasoning curriculum -- a TRANSITIVE-INFERENCE environment plus a neurogenesis-grown spiking
learner that the brain's sleep-loop trains on, so HELD-OUT reasoning accuracy RISES across nights.

Honest framing (read this first). This is NARROW, LEARNED reasoning -- transitive inference (TI),
the classic "A>B, B>C => A>C" task from comparative/developmental psychology (Bryant & Trabasso,
Nature 1971; the associative-value account of Frank, Rudy & O'Reilly, 2003). It is a REAL cognitive
skill that a simple reinforcement rule provably grows into, and it is exactly the kind of skill an
offline replay/curriculum sleep-loop should improve. It is NOT general intelligence, NOT symbolic
logic, NOT a proof system: the learner fits a scalar RANK per item and generalizes by the geometry
of those ranks. The value is that the generalization is GENUINE -- the learner is trained ONLY on
ADJACENT premises and is scored ONLY on NON-adjacent pairs it has never seen, so a rising held-out
curve is real inference, not memorization.

Why adjacent-only training yields transitive generalization. Each premise "a is before b" nudges a
learned scalar rank[a] up and rank[b] down by an error-scaled (Bradley-Terry / Elo) step. The stable
fixed point of those adjacent constraints is a MONOTONE rank vector consistent with the hidden total
order; once the ranks are monotone, ANY pair -- including the never-seen non-adjacent ones -- is
ordered correctly by its rank gap. That is the whole mechanism of the associative account of TI, and
it is why the symbolic-distance effect (bigger gaps are easier) falls out for free.

Genuinely SPIKING decision. The answer to a query is NOT a bare float compare of two ranks. The rank
gap is injected as the DRIFT of a spiking evidence-accumulation attractor (basal_ganglia/
evidence_accumulation.DriftDiffusion -- LIF pools + shared inhibition, Wang 2002); the returned bool
is a function of WHICH POOL wins the spiking race. Small gaps (an inconsistent / cyclic premise set)
leave the ranks flat -> near-zero drift -> chance, exactly the psychometric behaviour of the
accumulator; large gaps (a well-ordered chain) drive a confident spiking commitment.

Genuinely SPIKING identity + NEUROGENESIS. Every item is a spike-polarity hypervector
(cortex/vsa_reasoning.hv -- a balanced two-line spike code). The learner routes an item to its rank
unit by spike-COINCIDENCE cleanup over a grown codebook; a code that no unit recognises (novelty)
RECRUITS a new rank unit -- neurogenesis in the spike-code substrate (mirrors
growth/growing_learner). So n_units grows as the world grows.

The sleep-loop that grows the skill (develop()):
  * WAKE  : sample ADJACENT premises from the currently-active item set and train on them; a
            learning-progress CURIOSITY drive (faculties/intrinsic_motivation) can pick the
            still-improving FRONTIER premise instead of a uniform one.
  * SLEEP : offline REPLAY-consolidation (sleep/experience_replay.ReplayBuffer) re-applies the
            SAME rank update to sampled stored premises -- more learning per real interaction.
  * RAMP  : a self-paced CURRICULUM (faculties/spiking_curiosity.SpikingCurriculum) grows the active item set
            small -> large as the learner's own learning-progress on the current chunk dries up.
  * SCORE : held_out_accuracy() over ALL non-adjacent pairs, evaluated in BOTH directions (so the
            held-out labels are BALANCED and a constant answer scores ~0.5), recorded per night.

Un-riggable self-check (deterministic seed=0, ASCII), each claim paired with a control that fails:
  (1) HEADLINE: adjacent-only training -> non-adjacent held-out RISES to >= 0.9. CONTROL-MEM (a
      store-the-pairs memoriser) stays ~0.5 on the never-seen pairs; CONTROL-SHUFFLE (an
      inconsistent, cyclic premise set with no total order) stays ~chance.
  (2) NO LEAKAGE: train pairs and held-out are DISJOINT, every held-out pair has order-distance >= 2,
      the learner is never trained on a held-out pair, and the held-out label set is balanced.
  (3) SLEEP/REPLAY HELPS: at an equal LIVE-premise budget, offline replay-consolidation reaches
      higher held-out than no replay.
  (4) NEUROGENESIS: n_units grows with #items, one rank unit per item. The curriculum comparison is an
      HONEST NEGATIVE -- measured over 10 seed shifts a small->large curriculum does NOT beat
      jump-to-largest at an equal budget (0.753 vs 0.790 mean held-out); both schedules learn the
      chain well above the memoriser control, and that is what is asserted.
  (5) SPIKING + DETERMINISM + ASCII: decide() routes through the spiking accumulator (discrete spike
      events), two seed=0 runs give an identical arc, all output is ASCII. Curiosity is reported
      HONESTLY (assert it helps, or print an honest no-help).

Per-instance random.Random(seed), zlib.crc32 for
persisted decision seeds (never hash()), no wall-clock, ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.reasoning_curriculum
"""

import random
import sys
import zlib

from snn_data_lake.cortex.vsa_reasoning import cleanup, hv
from snn_data_lake.faculties.evidence_accumulation import DriftDiffusion
from snn_data_lake.faculties.intrinsic_motivation import CuriosityDrive
from snn_data_lake.faculties.spiking_curiosity import SpikingCurriculum
from snn_data_lake.infra._mathutils import sigmoid
from snn_data_lake.sleep.experience_replay import ReplayBuffer

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "ReasoningLearner._train_units": (
        "LOCAL",
        "one error-scaled Bradley-Terry nudge for 'unit ua is before unit ub', with weight decay. A "
        "pairwise rank update applied directly to the two units involved -- nothing crosses a layer",
    ),
    "ReasoningLearner.train_premise": (
        "LOCAL",
        "routes/recruits both items and applies the same rank nudge for one premise; inherits the "
        "claim of the rule it applies",
    ),
    "ReasoningLearner.replay_consolidate": (
        "LOCAL",
        "SLEEP: re-applies the rank nudge to a sampled batch of STORED premises. Offline replay of a "
        "local rule is still that local rule",
    ),
    "MemorizationLearner.train_premise": (
        "CONTROL",
        "the rote memoriser whole training step is a single dict assignment marking the pair seen. It "
        "exists to LOSE: storing exact adjacent pairs scores about 0.5 on held-out non-adjacent ones, "
        "which is what makes the real learner generalisation meaningful",
    ),
    "ReasoningEnv.train_pairs": (
        "NOT_A_RULE",
        "returns the unordered SET of pairs training may touch. A property describing the split",
    ),
}

# ---- learner / decision constants ----------------------------------------------------------------
_VIGILANCE = 0.30  # spike-coincidence sim below this -> a novel item code -> recruit (neurogenesis)
_LR = 0.35  # Bradley-Terry / Elo rank step (fraction of the prediction error applied)
_DECAY = 0.03  # rank weight decay: a consistent chain still separates (data gradient dominates),
# but a contradictory CYCLE has ~0 net gradient so decay keeps its ranks flat -> chance
_DRIFT_SCALE = 2.2  # rank-gap -> accumulator drift (raw, NOT normalised: a flat chain -> ~0 drift)
_BOUND = 1.0  # accumulator sustained-firing bound
_NOISE = 0.6  # accumulator membrane-noise amplitude
_MAX_STEPS = 400  # accumulator step cap (bounds runtime; a big drift commits far sooner)


def _pair_seed(ua, ub):
    """Deterministic per-decision accumulator seed from the two unit ids (crc32, never hash())."""
    return zlib.crc32(f"{ua}:{ub}".encode()) & 0x7FFFFFFF


class ReasoningEnv:
    """A transitive-inference world with a HIDDEN total order over n_items.

    The order is a permutation drawn from a per-instance rng. TRAIN premises are ADJACENT pairs only
    (order[i] is before order[i+1]); HELD-OUT queries are all NON-adjacent pairs (order-distance
    >= 2), which the learner never sees -- so answering them is genuine transitive inference.
    encode(item) is a spike-polarity hypervector (vsa_reasoning.hv); label(a, b) is True iff a is
    before b in the hidden order."""

    def __init__(self, n_items, seed=0, tag="TI"):
        self.n_items = int(n_items)
        self.seed = int(seed)
        self.tag = tag
        rng = random.Random(seed)
        self.order = list(range(self.n_items))
        rng.shuffle(self.order)  # the hidden total order (a permutation)
        self.pos = {
            item: i for i, item in enumerate(self.order)
        }  # item -> rank position (0 = earliest)

    def encode(self, item):
        """Spike-polarity hypervector identity for an item (balanced two-line spike code)."""
        return hv(f"{self.tag}:{self.seed}:{item}")

    def label(self, a, b):
        """Ground truth: True iff item a is BEFORE item b in the hidden total order."""
        return self.pos[a] < self.pos[b]

    def adjacent_premises(self, k=None):
        """Directed (winner, loser) premises among the first k order positions (winner is before
        loser). k=None uses the full order. These ADJACENT (distance-1) pairs are the ONLY training
        signal."""
        k = self.n_items if k is None else min(k, self.n_items)
        return [(self.order[i], self.order[i + 1]) for i in range(k - 1)]

    def train_pairs(self):
        """The unordered set of pairs that training is allowed to touch (adjacent pairs)."""
        return {frozenset(p) for p in self.adjacent_premises()}

    def heldout_pairs(self):
        """Unordered NON-adjacent pairs (order-distance >= 2) minus any training pair -> the
        never-trained held-out query set. Genuinely transitive: only inferable from the chain."""
        train = self.train_pairs()
        out = []
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                a, b = self.order[i], self.order[j]
                if abs(self.pos[a] - self.pos[b]) >= 2 and frozenset((a, b)) not in train:
                    out.append((a, b))
        return out


class InconsistentEnv(ReasoningEnv):
    """CONTROL: an inconsistent premise set with NO total order. Every adjacent pair is asserted in
    BOTH directions ('a is before b' AND 'b is before a') -- a length-2 cycle on each edge, so the
    premises are mutually CONTRADICTORY and no ranking can satisfy them. The two opposing nudges
    cancel and (with decay) the ranks self-correct to FLAT, so the spiking drift stays ~0 and
    held-out accuracy stays at chance. Held-out is scored against the linear labels and excludes
    every adjacent training pair (same unordered train set as the consistent env)."""

    def adjacent_premises(self, k=None):
        directed = super().adjacent_premises(k)  # the linear chain (respects k) ...
        return [
            p for a, b in directed for p in ((a, b), (b, a))
        ]  # ... each edge in BOTH directions


class ReasoningLearner:
    """A neurogenesis-grown spiking transitive-inference learner.

    Items are identified by spike-COINCIDENCE cleanup over a grown codebook; a novel code RECRUITS a
    new rank unit (neurogenesis). train_premise(code_a, code_b) applies an error-scaled Bradley-Terry
    nudge (rank[a] up, rank[b] down) whose fixed point is a MONOTONE rank vector -> transitive
    generalization. decide(code_a, code_b) injects the rank gap as the DRIFT of a spiking
    evidence-accumulation attractor and returns which pool wins the spiking race (NOT a float
    compare). Deterministic; per-instance random.Random(seed)."""

    def __init__(
        self,
        seed=0,
        vigilance=_VIGILANCE,
        lr=_LR,
        decay=_DECAY,
        drift_scale=_DRIFT_SCALE,
        bound=_BOUND,
        noise=_NOISE,
        max_steps=_MAX_STEPS,
    ):
        self.rng = random.Random(seed)
        self.vigilance = float(vigilance)
        self.lr = float(lr)
        self.decay = float(decay)
        self.drift_scale = float(drift_scale)
        self.bound = float(bound)
        self.noise = float(noise)
        self.max_steps = int(max_steps)
        self.codebook = {}  # unit_id -> spike code (grown)
        self.rank = {}  # unit_id -> learned scalar rank
        self._next_uid = 0
        self.buffer = ReplayBuffer(capacity=1024, seed=seed)  # premise store for sleep replay
        self.last_trace = None  # last spiking-decision trace

    @property
    def n_units(self):
        return len(self.codebook)

    def state_dict(self):
        """JSON-able learner state: the grown codebook + learned ranks -- enough to decide()/score
        identically after a reload. The replay buffer is NOT saved (it re-fills from live premises;
        the ranks ARE the skill), keeping the durable blob small (real memory only, no noise)."""
        return {
            "codebook": {str(u): list(c) for u, c in self.codebook.items()},
            "rank": {str(u): float(r) for u, r in self.rank.items()},
            "next_uid": self._next_uid,
        }

    def load_state_dict(self, d):
        """Restore a state_dict() blob (grown codebook + ranks). None/empty -> a fresh learner (no-op)."""
        if not d:
            return self
        self.codebook = {int(u): list(c) for u, c in d.get("codebook", {}).items()}
        self.rank = {int(u): float(r) for u, r in d.get("rank", {}).items()}
        self._next_uid = int(d.get("next_uid", len(self.codebook)))
        return self

    def _route(self, code, grow):
        """Return the unit id for a spike code, recruiting a new rank unit on novelty (grow=True).
        Routing is a spike-coincidence cleanup over the codebook; a best sim below vigilance is a
        never-seen item -> neurogenesis. Returns -1 for an unknown code when grow=False."""
        if self.codebook:
            uid, s = cleanup(code, self.codebook)  # nearest stored unit by spike coincidence
            if s >= self.vigilance:
                return uid
        if not grow:
            return -1
        uid = self._next_uid
        self._next_uid += 1
        self.codebook[uid] = list(code)
        self.rank[uid] = 0.0
        return uid

    def _predict(self, ua, ub):
        """Predicted P(a before b) from the current ranks (logistic of the rank gap)."""
        return sigmoid(self.rank[ua] - self.rank[ub])

    def _train_units(self, ua, ub):
        """Apply one error-scaled Bradley-Terry nudge for 'unit ua is before unit ub', with weight
        decay. A consistent chain's data gradient dominates decay and separates; a symmetric cycle's
        net gradient is ~0 so decay pulls its ranks flat (-> chance). Returns the pre-update
        prediction error (1 - P), the curiosity/learning-progress signal."""
        p = self._predict(ua, ub)
        err = 1.0 - p
        self.rank[ua] += self.lr * err - self.decay * self.rank[ua]
        self.rank[ub] += -self.lr * err - self.decay * self.rank[ub]
        return err

    def train_premise(self, code_a, code_b):
        """Learn one premise 'a is before b' from spike codes (routes/recruits both items, records
        the premise for replay, applies the rank nudge). Returns the pre-update prediction error."""
        ua = self._route(code_a, grow=True)
        ub = self._route(code_b, grow=True)
        self.buffer.add(ua, ub, 1.0)  # store for offline sleep replay
        return self._train_units(ua, ub)

    def replay_consolidate(self, batch):
        """SLEEP: re-apply the rank nudge to a sampled batch of STORED premises (no new live
        premise). Offline consolidation -> more learning per real interaction. Returns #updates."""
        if not batch:
            return 0
        picks = self.buffer.sample(batch)
        for ua, ub, _r in picks:
            self._train_units(ua, ub)
        return len(picks)

    def decide(self, code_a, code_b):
        """Answer 'is a before b?' via the SPIKING accumulator: drift = drift_scale * (rank gap);
        the returned bool is which LIF pool wins the spiking race. Unknown codes -> a balanced guess.
        Records the spiking trace on self.last_trace."""
        ua = self._route(code_a, grow=False)
        ub = self._route(code_b, grow=False)
        if ua < 0 or ub < 0:
            self.last_trace = None
            return bool(zlib.crc32(f"{ua}:{ub}".encode()) & 1)  # unseen item -> balanced coin
        drift = self.drift_scale * (self.rank[ua] - self.rank[ub])
        model = DriftDiffusion(
            drift,
            bound=self.bound,
            noise=self.noise,
            seed=_pair_seed(ua, ub),
            max_steps=self.max_steps,
        )
        choice, _rt = model.decide()  # spiking race: choice in {+1, -1}
        self.last_trace = model.last_trace
        return choice == 1  # +1 pool wins -> a is before b

    def held_out_accuracy(self, env):
        """Fraction of NON-adjacent held-out queries decided correctly. Every unordered pair is
        evaluated in BOTH directions, so the label set is balanced (a constant answer -> ~0.5)."""
        correct = total = 0
        for a, b in env.heldout_pairs():
            for x, y in ((a, b), (b, a)):
                if self.decide(env.encode(x), env.encode(y)) == env.label(x, y):
                    correct += 1
                total += 1
        return correct / total if total else 0.0


def _mean_active_error(learner, env, premises):
    """Mean pre-update prediction error over the active premises (no update) -- the learning-progress
    source that paces the curriculum. Unrouted items (not yet recruited) count as max error 1.0."""
    if not premises:
        return 0.0
    tot = 0.0
    for a, b in premises:
        ua = learner._route(env.encode(a), grow=False)
        ub = learner._route(env.encode(b), grow=False)
        tot += (1.0 - learner._predict(ua, ub)) if (ua >= 0 and ub >= 0) else 1.0
    return tot / len(premises)


def develop(
    env,
    nights,
    live_budget,
    seed=0,
    replay_batch=0,
    curiosity=False,
    stage_sizes=None,
    jump=False,
    decay=_DECAY,
    eval_last_only=False,
):
    """Run the wake/sleep sleep-loop for `nights`, recording held-out accuracy each night.

    WAKE samples `live_budget` ADJACENT premises from the active item set (curriculum-controlled) and
    trains on them; a learning-progress CURIOSITY drive optionally picks the frontier premise. SLEEP
    replays `replay_batch` stored premises offline. A self-paced CURRICULUM (stage_sizes) grows the
    active item set small->large; jump=True disables it (always the full set). eval_last_only=True
    scores held-out on the final night only (0.0 placeholders earlier) to save accumulator cost.
    Returns (arc, units, learner, trained_pairs): arc is the per-night held-out accuracy, units the
    per-night n_units, trained_pairs the unordered item pairs training touched (leakage assertion)."""
    learner = ReasoningLearner(seed=seed, decay=decay)
    drive = CuriosityDrive(seed=seed) if curiosity else None
    if stage_sizes and not jump:
        cur = SpikingCurriculum(len(stage_sizes), mastery_threshold=0.05, patience=1, seed=seed)
    else:
        cur = None
    trained_pairs = set()
    arc = []
    units = []
    prev_err = None
    for night in range(nights):
        if cur is not None:
            k = stage_sizes[cur.stage]
        else:
            k = env.n_items
        premises = env.adjacent_premises(k)
        keys = [f"{a}:{b}" for a, b in premises]
        by_key = dict(zip(keys, premises))
        # ---- WAKE: live premises (curiosity frontier pick or uniform) ----
        for _ in range(live_budget):
            if drive is not None:
                key = drive.most_curious(keys)
                a, b = by_key[key]
            else:
                a, b = premises[learner.rng.randrange(len(premises))]
                key = f"{a}:{b}"
            err = learner.train_premise(env.encode(a), env.encode(b))
            trained_pairs.add(frozenset((a, b)))
            if drive is not None:
                drive.observe(key, err)
        # ---- SLEEP: offline replay-consolidation ----
        learner.replay_consolidate(replay_batch)
        # ---- RAMP: pace the curriculum on the active chunk's learning-progress dry-up ----
        if cur is not None:
            cur_err = _mean_active_error(learner, env, premises)
            lp = 0.0 if prev_err is None else max(0.0, prev_err - cur_err)
            cur.observe(lp)
            prev_err = cur_err
        scored = (not eval_last_only) or night == nights - 1
        arc.append(learner.held_out_accuracy(env) if scored else 0.0)
        units.append(learner.n_units)
    return arc, units, learner, trained_pairs


class MemorizationLearner:
    """CONTROL: a rote memoriser -- stores the exact ADJACENT premises and, for any UNSEEN pair,
    guesses. Because held-out pairs are all non-adjacent (never stored) and the guess is a function
    of the UNORDERED pair, evaluating both directions scores it at exactly ~0.5. No generalization."""

    def __init__(self):
        self.seen = {}

    def train_premise(self, a, b):
        self.seen[(a, b)] = True  # 'a is before b'

    def decide(self, a, b):
        if (a, b) in self.seen:
            return True
        if (b, a) in self.seen:
            return False
        lo, hi = sorted((a, b))
        return bool(
            zlib.crc32(f"{lo}:{hi}".encode()) & 1
        )  # unseen -> balanced, direction-invariant

    def held_out_accuracy(self, env):
        correct = total = 0
        for a, b in env.heldout_pairs():
            for x, y in ((a, b), (b, a)):
                if self.decide(x, y) == env.label(x, y):
                    correct += 1
                total += 1
        return correct / total if total else 0.0


def _fmt_arc(arc):
    return " ".join(f"{a:.2f}" for a in arc)


def main():
    n_items = 6
    nights = 8
    stage_sizes = [3, 4, 5, 6]

    # ---- (1) HEADLINE: adjacent-only training -> non-adjacent held-out rises to >= 0.9 ----
    env = ReasoningEnv(n_items, seed=0)
    arc, units_arc, learner, trained = develop(
        env,
        nights,
        live_budget=24,
        seed=0,
        replay_batch=24,
        curiosity=False,
        stage_sizes=stage_sizes,
    )
    reasoning_final = arc[-1]

    # CONTROL-MEM: rote memoriser trained on the SAME adjacent premises.
    mem = MemorizationLearner()
    for a, b in env.adjacent_premises():
        mem.train_premise(a, b)
    mem_acc = mem.held_out_accuracy(env)

    # CONTROL-SHUFFLE: an inconsistent (contradictory-cycle) premise set has NO total order. A single
    # instance yields an arbitrary rank order (uncorrelated with the true labels -> high per-seed
    # variance), so the un-riggable statistic is the ENSEMBLE MEAN over independent instances, which
    # centers at chance. Same training budget as the reasoning learner (a fair mirror).
    shuffle_seeds = list(range(10))
    shuffle_each = []
    for s in shuffle_seeds:
        sh_env = InconsistentEnv(n_items, seed=s, tag="CYC")
        sh_arc, _u, _l, _t = develop(
            sh_env,
            nights,
            live_budget=24,
            seed=s,
            replay_batch=24,
            curiosity=False,
            stage_sizes=None,
            jump=True,
            eval_last_only=True,
        )
        shuffle_each.append(sh_arc[-1])
    shuffle_acc = sum(shuffle_each) / len(shuffle_each)

    # ---- (2) NO LEAKAGE: disjoint, distance >= 2, balanced held-out labels ----
    train_set = env.train_pairs()
    held = env.heldout_pairs()
    held_set = {frozenset(p) for p in held}
    disjoint = train_set.isdisjoint(held_set)
    all_far = all(abs(env.pos[a] - env.pos[b]) >= 2 for a, b in held)
    never_trained = trained.isdisjoint(held_set)
    labels = [env.label(x, y) for a, b in held for x, y in ((a, b), (b, a))]
    balanced = labels.count(True) == labels.count(False)

    # ---- (3) SLEEP/REPLAY and (4) CURRICULUM, both at an equal LIVE budget ---------------------------
    # Both comparisons are a difference of two held-out accuracies measured on ONE draw of a ~6-12 item
    # held-out set, which is far too coarse to carry a strict `>`. Each is now averaged over REPS seed
    # shifts. 4 x REPS develop() runs cost ~1 s in total, so this is nearly free.
    cur_n, cur_stages = 7, [3, 4, 5, 6, 7]
    REPS = 8
    no_reps, with_reps, curs, jumps = [], [], [], []
    for k in range(REPS):
        no_reps.append(
            develop(
                ReasoningEnv(n_items, seed=1 + k),
                nights=6,
                live_budget=6,
                seed=3 + k,
                replay_batch=0,
                stage_sizes=None,
                jump=True,
                eval_last_only=True,
            )[0][-1]
        )
        with_reps.append(
            develop(
                ReasoningEnv(n_items, seed=1 + k),
                nights=6,
                live_budget=6,
                seed=3 + k,
                replay_batch=24,
                stage_sizes=None,
                jump=True,
                eval_last_only=True,
            )[0][-1]
        )
        curs.append(
            develop(
                ReasoningEnv(cur_n, seed=2 + k),
                nights=10,
                live_budget=4,
                seed=5 + k,
                replay_batch=0,
                stage_sizes=cur_stages,
                eval_last_only=True,
            )[0][-1]
        )
        jumps.append(
            develop(
                ReasoningEnv(cur_n, seed=2 + k),
                nights=10,
                live_budget=4,
                seed=5 + k,
                replay_batch=0,
                stage_sizes=None,
                jump=True,
                eval_last_only=True,
            )[0][-1]
        )
    no_rep, with_rep = no_reps[0], with_reps[0]  # draw 0 stays the printed worked example
    cur_final, jump_final = curs[0], jumps[0]
    mean_no_rep, mean_with_rep = sum(no_reps) / REPS, sum(with_reps) / REPS
    mean_cur, mean_jump = sum(curs) / REPS, sum(jumps) / REPS

    # neurogenesis: n_units grows one unit per newly-seen item as the active set grows.
    g_env = ReasoningEnv(n_items, seed=0)
    g_learner = ReasoningLearner(seed=0)
    grow_curve = []
    for k in range(2, n_items + 1):
        for a, b in g_env.adjacent_premises(k):
            g_learner.train_premise(g_env.encode(a), g_env.encode(b))
        grow_curve.append((k, g_learner.n_units))

    # ---- (5) SPIKING structure + DETERMINISM + CURIOSITY (honest) ----
    _pred = learner.decide(env.encode(env.order[0]), env.encode(env.order[-1]))
    trace = learner.last_trace
    spiking = trace is not None and trace["spikes"] > 0
    arc2, _u2, _l2, _t2 = develop(
        ReasoningEnv(n_items, seed=0),
        nights,
        live_budget=24,
        seed=0,
        replay_batch=24,
        curiosity=False,
        stage_sizes=stage_sizes,
    )
    deterministic = arc == arc2
    no_tensor = "numpy" not in sys.modules and "torch" not in sys.modules

    # CURIOSITY (honest): frontier-premise selection vs uniform, at an equal small budget.
    cur_on, _x, _xl, _y = develop(
        ReasoningEnv(n_items, seed=4),
        nights=6,
        live_budget=6,
        seed=6,
        replay_batch=0,
        curiosity=True,
        stage_sizes=None,
        jump=True,
        eval_last_only=True,
    )
    cur_off, _p, _pl, _q = develop(
        ReasoningEnv(n_items, seed=4),
        nights=6,
        live_budget=6,
        seed=6,
        replay_batch=0,
        curiosity=False,
        stage_sizes=None,
        jump=True,
        eval_last_only=True,
    )
    curiosity_on, curiosity_off = cur_on[-1], cur_off[-1]
    curiosity_helps = curiosity_on > curiosity_off + 0.02

    print("=" * 72)
    print("REASONING CURRICULUM -- transitive inference grown by the sleep-loop (spiking)")
    print("=" * 72)
    print(f"n_items={n_items}  nights={nights}  curriculum stages(item-counts)={stage_sizes}")
    print(
        "train = ADJACENT premises only | held-out = NON-adjacent pairs (distance >= 2), both dirs"
    )
    print("-" * 72)
    print("night | held-out acc | n_units")
    for night, (acc, uni) in enumerate(zip(arc, units_arc), start=1):
        bar = "#" * int(round(acc * 24))
        print(f"  {night:>2}  |     {acc:.2f}     |   {uni:>2}   {bar}")
    print("-" * 72)
    print(f"(1) HEADLINE  : reasoning held-out arc [{_fmt_arc(arc)}] -> {reasoning_final:.2f}")
    print(
        f"    controls  : memoriser {mem_acc:.2f}  |  inconsistent/cyclic MEAN {shuffle_acc:.2f} "
        f"over {len(shuffle_seeds)} seeds (both ~chance)"
    )
    print(
        f"(2) no-leak   : disjoint={disjoint}  all-dist>=2={all_far}  never-trained-heldout={never_trained}  "
        f"balanced-labels={balanced} ({labels.count(True)}T/{labels.count(False)}F)"
    )
    print(
        f"(3) replay    : held-out no-replay {no_rep:.2f} -> with-replay {with_rep:.2f} (equal live "
        f"budget); over {REPS} draws {mean_no_rep:.3f} -> {mean_with_rep:.3f}"
    )
    print(
        f"(4) curric.   : small->large {cur_final:.2f} vs jump-to-largest {jump_final:.2f} "
        f"(n={cur_n}, equal budget); neurogenesis {grow_curve}"
    )
    print(
        f"    HONEST NEGATIVE over {REPS} draws: small->large {mean_cur:.3f} vs jump-to-largest "
        f"{mean_jump:.3f} -- the curriculum does NOT beat jumping to the largest set; both learn it"
    )
    print(
        f"(5) spiking   : accumulator trace spikes={trace['spikes'] if spiking else 0}  "
        f"deterministic={deterministic}  no-numpy/torch={no_tensor}"
    )
    print(
        f"    curiosity : frontier {curiosity_on:.2f} vs uniform {curiosity_off:.2f} -> "
        f"{'HELPS' if curiosity_helps else 'no clear help (honest)'}"
    )
    print("=" * 72)

    # ---- (1) HEADLINE asserts: genuine transitive inference, beating both controls ----
    assert reasoning_final >= 0.9, (
        f"held-out transitive inference must reach >= 0.9 (got {reasoning_final:.2f})"
    )
    assert arc[-1] > arc[0] + 0.1, (
        f"held-out accuracy must RISE across nights ({arc[0]:.2f} -> {arc[-1]:.2f})"
    )
    assert 0.4 <= mem_acc <= 0.6, (
        f"memoriser must score ~0.5 on never-seen pairs (got {mem_acc:.2f})"
    )
    assert 0.4 <= shuffle_acc <= 0.6, (
        f"inconsistent/cyclic ensemble mean must be ~chance (got {shuffle_acc:.2f})"
    )
    assert reasoning_final >= mem_acc + 0.25, "reasoning must clearly beat the memoriser control"
    assert reasoning_final >= shuffle_acc + 0.25, (
        "reasoning must clearly beat the inconsistent-order control"
    )

    # ---- (2) NO-LEAKAGE asserts ----
    assert disjoint, "train pairs and held-out pairs must be DISJOINT"
    assert all_far, "every held-out pair must have order-distance >= 2"
    assert never_trained, "the learner must NEVER have trained on a held-out pair"
    assert balanced, (
        "held-out labels must be balanced (both directions) so a constant answer scores ~0.5"
    )

    # ---- (3) SLEEP/REPLAY asserts ----
    # `assert with_rep > no_rep` on draw 0 alone held 3/7 under a seed sweep of this self-check. Over 10
    # seed shifts replay wins EVERY draw (no-replay 0.700-0.950, with-replay 0.750-1.000), so the claim
    # is real and only its estimator was too coarse; the mean over REPS draws is what is asserted now.
    assert mean_with_rep > mean_no_rep, (
        f"offline replay must beat no-replay at equal live budget over {REPS} draws "
        f"({mean_with_rep:.3f} vs {mean_no_rep:.3f}; draw 0 {with_rep:.2f} vs {no_rep:.2f})"
    )

    # ---- (4) NEUROGENESIS + CURRICULUM asserts ----
    # `assert cur_final > jump_final` ("small->large curriculum must beat jump-to-largest") stood here
    # on ONE draw and held 3/7 under a seed sweep. Re-measured over 10 seed shifts, both arms retrained
    # per shift at an equal budget:
    #     small->large     0.800 0.767 0.667 0.700 0.767 0.833 0.767 0.767 0.833 0.633   mean 0.753
    #     jump-to-largest  0.700 0.833 0.667 0.800 0.800 0.900 0.700 0.800 0.900 0.800   mean 0.790
    # The curriculum wins 2 of 10 draws and LOSES on the mean, and no window of 2, 3 or 4 draws rescues
    # it. This claim is REFUTED, not noisy -- averaging it harder would only make the failure sharper,
    # so it is not asserted at all any more. What is true on every draw is that BOTH schedules learn the
    # 7-item chain far above the 0.5 memoriser control, and that is asserted instead; the ordering is
    # printed as an honest negative. Curriculum ordering is open work, not a demonstrated property.
    assert min(mean_cur, mean_jump) >= 0.65, (
        f"both schedules must learn the {cur_n}-item chain well above the ~0.5 memoriser control "
        f"(small->large {mean_cur:.3f}, jump-to-largest {mean_jump:.3f} over {REPS} draws)"
    )
    assert grow_curve[0][1] < grow_curve[-1][1], (
        "n_units must GROW as the number of items grows (neurogenesis)"
    )
    assert grow_curve[-1][1] == n_items, (
        f"one rank unit per item must be recruited (got {grow_curve[-1][1]}/{n_items})"
    )

    # ---- (5) SPIKING + DETERMINISM asserts ----
    assert spiking, "decide() must route through the spiking accumulator (discrete spike events)"
    assert deterministic, "two seed=0 runs must give an identical held-out arc"

    # ---- (6) DURABILITY: a state_dict() round-trip reloads the SAME skill (never-forget across a boot) ----
    blob = learner.state_dict()
    reloaded = ReasoningLearner(seed=0).load_state_dict(blob)
    same = all(
        learner.decide(env.encode(a), env.encode(b))
        == reloaded.decide(env.encode(a), env.encode(b))
        for a, b in env.heldout_pairs()
    )
    reloaded_acc = reloaded.held_out_accuracy(env)
    print(
        f"(6) durable   : state_dict round-trip -> reloaded held-out {reloaded_acc:.2f} "
        f"(units {reloaded.n_units}), decides identically={same}"
    )
    assert same, "a reloaded learner (state_dict round-trip) must decide held-out pairs identically"
    assert reloaded.n_units == learner.n_units, (
        "state_dict round-trip must preserve the grown codebook"
    )

    print(
        f"self-check OK: adjacent-only training -> transitive held-out {arc[0]:.2f} -> {reasoning_final:.2f} "
        f"(>= 0.9), beating memoriser {mem_acc:.2f} and cyclic {shuffle_acc:.2f};"
    )
    print(
        f"  no leakage (disjoint, dist>=2, balanced); replay {no_rep:.2f}->{with_rep:.2f}; "
        f"curriculum {mean_cur:.2f} vs jump {mean_jump:.2f} over {REPS} draws (NO curriculum advantage -- "
        f"honest negative); neurogenesis 1 unit/item; spiking + deterministic; "
        f"curiosity {'helps' if curiosity_helps else 'no clear help (reported honestly)'}"
    )


if __name__ == "__main__":
    main()
