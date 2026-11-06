"""
Interaction learning -- improve the live brain by TALKING to it, via a genuinely SPIKING path.

ONE spiking primitive, framed honestly: a reward-modulated spike-timing ASSOCIATIVE learner that
gets better through interaction. It is NOT a claim of general intelligence, language understanding,
or consciousness. It learns a small (stimulus -> response) mapping purely from spike TIMING and a
scalar reward -- the same three-factor rule the brain's basal ganglia uses -- so it corrects the
audit's drift toward rate-level ANN code (backprop / Widrow-Hoff / EMAs) instead of deepening it.

WHY THIS IS GENUINELY SPIKING (not rate-level), on both axes the audit flagged:
  (1) The stimulus is a SPIKE TRAIN, not a real-valued activation vector. Each input synapse fires
      exactly ONE spike at a DISCRETE integer time, given by Thorpe rank-order latency coding
      (REUSED verbatim from sensorimotor.rank_order_coding.rank_code: 0 = fires first = largest value).
      The representation is binary spike EVENTS at discrete times -- self-checked, not just claimed.
  (2) The weight update is REWARD-MODULATED SPIKE-TIMING PLASTICITY (R-STDP), REUSED verbatim from
      plasticity.r_stdp.RSTDPNeuron: pre-before-post spike pairings deposit a per-synapse
      ELIGIBILITY trace e[i] = exp(-(t_post - t_pre_i)/tau) via the STDP window, and a scalar reward
      d in {-1,+1} gates it into the weights:  w[i] += lr * d * e[i]. This is the classic three-factor
      rule (pre x post x neuromodulator). There is NO error gradient, NO least-squares, NO EMA: the
      structural self-check re-derives every weight delta as exactly lr * reward * eligibility and
      asserts equality.

The readout is spike-timing too: each response neuron's decision variable is its peak sub-threshold
membrane response  v = sum_i w[i] * PSP(t_post - t_spike_i)  (a leaky-neuron / tempotron-style read
of the SAME spike times), and the response is the neuron that peaks highest. Because the PSP kernel
equals the STDP eligibility kernel, a positive reward on a neuron directly raises its membrane
response to THAT spike pattern -- so repeated rewarded interactions sharpen the mapping.

Learning-through-interaction. `interact(stimulus, feedback)` is one exchange. Feedback teaches it two
equivalent ways -- both purely reward-modulated spike-timing plasticity, never a gradient:
  * TEACHING signal (feedback = the correct response label): fire the taught-correct neuron on the
    stimulus spike train with dopamine d=+1 (reinforce its causal eligibility), and if the learner's
    own answer was wrong, fire the wrongly-chosen neuron with d=-1 (depress it). This is R-STDP in its
    supervised form -- the reward sign is the teacher -- and it is what self-check #1 drives, since an
    interaction there is literally a (stimulus, correct-response) pair.
  * SCALAR reward (feedback = +1/-1 for the answer just given): the weaker pure-reinforcement variant
    -- reward only the neuron that actually fired. HONEST NEGATIVE: with reward on the chosen action
    ALONE, a whole response class can collapse (the correct neuron may never be selected to be
    reinforced), so this variant plateaus below the teaching signal. Kept and documented, not faked.

Over exchanges, held-out accuracy climbs from chance to high. The un-riggable control: SHUFFLE the
teaching signal (decorrelate it from the stimulus) and the identical learner stays at chance --
proving the gain comes from reward-modulated spike TIMING, not from the encoding alone.

The __main__ self-check also drives the real integrated brain (engine.live_brain.LiveBrain): it TALKS
to the brain with converse()/tick(), reads the brain's own grounded replies as ground truth, and
shows this spiking learner improving at routing prompts to the brain's actual intents -- a short,
readable transcript of the exchanges.

CONVERSATION-DRIVEN GENERALISATION (learn_turn / route_text; the second live-brain self-check). The
brain's fixed converse() router is a keyword firewall: it does NOT learn, and it MISSES novel
paraphrases that carry none of its hand-tuned keywords. This spiking router fixes that WITHOUT a rate
ANN. As you converse, the firewall's OWN grounded intent for each line becomes a SELF-SUPERVISED
teaching signal (distillation, no human labels): learn_turn(utterance, firewall_intent) does one
R-STDP interaction on the spike-encoded words. Taught on keyword-bearing lines the firewall labels
correctly, the spiking router then GENERALISES -- correctly routing a DISJOINT held-out set of novel
paraphrases (that share content words but no keyword) which the first-hit firewall MISSES. Honest
LIMIT, reported not hidden: on the firewall's OWN blind spots (novel lines it itself mislabels) pure
self-supervised distillation inherits the error and fails; an OPTIONAL explicit feedback intent (what
the human actually meant) is the stronger reward that recovers them -- still pure spike-timing R-STDP,
never a gradient. Generalisation needs the readout spike to land on the EARLY (content-word) spikes:
t_post < n_in, a spike-TIMING knob, so the membrane read keys on shared words, not the empty-bucket tail.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.readout.interaction_learning
"""

import math
import random
import threading
import zlib
from collections import deque

from snn_data_lake.plasticity.r_stdp import RSTDPNeuron
from snn_data_lake.sensorimotor.rank_order_coding import rank_code

_REPLAY_CAP = 64  # sleep-replay ring size (T058): a memory bound, not a behaviour threshold


def _state_attrs(obj):
    """Checkpoint fields, excluding only declared wiring and lake-derived caches."""
    excluded = set()
    if type(obj).__name__ == "SpikingInteractionLearner":
        excluded = {
            "_authority",
            "teach_sink",
            "event_sink",
            "unit_roles",
            "_effect_depth",
            "_replay",
            "_replay_ids",
            "_hydrated",
            "_hydration_count",
            "_hydration_refusals",
        }
    elif type(obj).__name__ == "RuleComposer":
        excluded = {"_specs", "_names"}
    return {
        k: v
        for k, v in vars(obj).items()
        if k not in excluded
        and not isinstance(v, (type(threading.Lock()), type(threading.RLock())))
    }


def _encode_state(obj):
    """G0(c) instance state, with exact floats and no undeclared silent omissions."""
    if obj is None or type(obj) in (bool, int, str):
        return obj
    if type(obj) is float:
        if not math.isfinite(obj):
            raise ValueError("non-finite checkpoint float")
        return obj
    if type(obj) is list:
        return [_encode_state(v) for v in obj]
    if type(obj) is dict:
        if any(type(k) is not str for k in obj):
            raise ValueError("checkpoint dict keys must be strings")
        return {"__kind__": "dict", "items": {k: _encode_state(v) for k, v in obj.items()}}
    if type(obj) in (tuple, set, deque):
        values = [_encode_state(v) for v in obj]
        if type(obj) is set:
            import json

            values.sort(key=lambda v: json.dumps(v, sort_keys=True))
        result = {"__kind__": type(obj).__name__, "items": values}
        if type(obj) is deque:
            result["maxlen"] = obj.maxlen
        return result
    if type(obj) is random.Random:
        return {"__kind__": "random", "state": _encode_state(obj.getstate())}
    cls = type(obj)
    if cls.__module__.startswith("snn_data_lake.") and hasattr(obj, "__dict__"):
        return {
            "__cls__": f"{cls.__module__}.{cls.__qualname__}",
            "attrs": {k: _encode_state(v) for k, v in _state_attrs(obj).items()},
        }
    raise ValueError(f"unsupported checkpoint value: {cls.__module__}.{cls.__qualname__}")


#: G0(c): the two classes the plasticity stack builds LAZILY, with the attribute sets their constructors assign.
#: Explicit rather than parsed from source at restore, so a frozen build (which ships no source) still restores;
#: tests/test_g0c_learner_persists.py pins each set to the constructor's own assignments, so a class change reds.
_G0C_LAZY_ATTRS = {
    "snn_data_lake.plasticity.bcm_plasticity.BCMNeuron": frozenset(
        {"lr", "tau_theta", "theta", "w"}
    ),
    "snn_data_lake.plasticity.benna_fusi.BennaFusiSynapse": frozenset(
        {"c", "dt", "g", "n", "u", "w_max"}
    ),
}


# distinct-from: substrate/home.py, faculties/reentrant_workspace.py -- phrase overlap only, as
#   with instruments/prior_drift_harness.py, faculties/intrinsic_motivation.py and
#   faculties/raphe_serotonin.py (an append observer, a theta authority, a least-squares
#   readout, a learning-progress read, a port declaration); all five sites read. The legacy-name
#   incumbent is the alias layer in snn_data_lake/__init__.py (`_MOVED`), whose S1b `_LEGACY`
#   table this provider now returns. Re-read at S1b with sensorimotor/afferent_boundary.py and
#   reservoir/kernel_tick.py ("THE ONE" transduction lane and bundle compiler): phrase overlap only.
def _g0c_legacy_modules():
    """G8 (specs/022, stage S0): THE one provider of the checkpoint class-path translation map,
    {legacy module: current module}, both relative to the package ("growth.lif_neuron") -- the form
    the G8 pre-registration (evidence/g8-reorganization.md 2.5) gives the alias layer's `_LEGACY`.
    It was EMPTY at S0 (only tests/test_g8_checkpoint_translation.py injected a map). RE-POINTED at
    G8's S1b: it returns the alias layer's own `snn_data_lake._LEGACY` table, read at call time --
    the one legacy table, beside the flat `_MOVED` aliases of `snn_data_lake/__init__.py`, never a
    second table (P2). Empty until a module moves; every move then extends it by itself."""
    return __import__("snn_data_lake")._LEGACY


# distinct-from: faculties/jepa_world_model.py, instruments/attention_schema.py -- phrase overlap only,
#   as with engine/brain.py (`_learned_route`), growth/h01_ingest.py and substrate/paradigm_b_matcher.py
#   (a band centroid, a stub brain, a route read, a pin key, a query runner); all five sites read.
#   This is the codec's own read-side class-path step; the one map it reads is the provider above.
def _g0c_translate(name, legacy):
    """A stored `__cls__` path through the legacy map, on the read side only (G8 S0): the longest
    legacy module that prefixes the name followed by "." is replaced by its current module, and the
    qualname -- which may itself be dotted -- is kept whole; identity when no legacy module matches.
    A translated name must RESOLVE (its module imports and holds that class) or the restore is
    refused: the stored checkpoint is never rewritten, and nothing is guessed."""
    import importlib

    pkg = "snn_data_lake."
    rest = name[len(pkg) :] if name.startswith(pkg) else ""
    hits = [old for old in legacy if rest.startswith(old + ".")]
    if not hits:
        return name
    old = max(hits, key=len)
    module, qualname = pkg + legacy[old], rest[len(old) + 1 :]
    try:
        target = importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(f"checkpoint class translation refused: {module} does not import") from exc
    for part in qualname.split("."):
        target = getattr(target, part, None)
    if not isinstance(target, type):
        raise ValueError(f"checkpoint class translation refused: {module} has no class {qualname}")
    return f"{module}.{qualname}"


_G0C_OPTIONAL = frozenset({"snn_data_lake.plasticity.bcm_plasticity.BCMNeuron"})


def _merge_state(target, saved):
    """Validate and stage the entire tree before mutating any existing instance."""
    import importlib

    legacy = _g0c_legacy_modules()
    pending, seen = [], {}

    def stage(fresh, value):
        if value is None or type(value) in (bool, int, str, float):
            if type(value) is float and not math.isfinite(value):
                raise ValueError("non-finite checkpoint float")
            if fresh is not None and value is not None and type(fresh) is not type(value):
                raise ValueError("checkpoint value type mismatch")
            if value is None and fresh is not None and hasattr(fresh, "__dict__"):
                # An OPTIONAL cell may return to its saved absence (an R2 sleep rollback to a
                # snapshot taken before the cell was built; panel M4, narrowed by fold panel F2).
                # Any other object -- a synapse inside a built cascade row included -- is corrupt.
                cls = type(fresh)
                if f"{cls.__module__}.{cls.__qualname__}" not in _G0C_OPTIONAL:
                    raise ValueError("checkpoint object replaced by null")
            return value
        if type(value) is list:
            if fresh is not None and type(fresh) is not list:
                raise ValueError("checkpoint list type mismatch")
            old = fresh or []
            out = [stage(old[i] if i < len(old) else None, v) for i, v in enumerate(value)]
            # Fold panel 2, F2: judged from the SAVED list, not the destination. A null element
            # beside objects is legitimate only among OPTIONAL cells (a _bcm list); a null inside
            # a built Benna-Fusi row is corrupt even when the destination row is cold. Ceiling: an
            # all-null list carries no class to judge by, so it is accepted (it fails loudly later).
            if any(v is None for v in value) and any(
                hasattr(x, "__dict__")
                and f"{type(x).__module__}.{type(x).__qualname__}" not in _G0C_OPTIONAL
                for x in out
            ):
                raise ValueError("checkpoint object replaced by null")
            return out
        if type(value) is not dict:
            raise ValueError("malformed checkpoint node")
        if "__cls__" in value:
            if set(value) != {"__cls__", "attrs"} or type(value["attrs"]) is not dict:
                raise ValueError("malformed checkpoint object")
            name, attrs = value["__cls__"], value["attrs"]
            if type(name) is not str or any(type(k) is not str for k in attrs):
                raise ValueError("malformed checkpoint class or attributes")
            name = _g0c_translate(name, legacy)  # G8 S0: before the lazy lookup AND the compare
            if fresh is None:
                # The only absent instances in this protocol are lazy BCM/cascade cells.
                if name not in _G0C_LAZY_ATTRS:
                    raise ValueError("unknown lazy checkpoint class")
                module, _, qualname = name.rpartition(".")
                cls = getattr(importlib.import_module(module), qualname)
                expected = set(_G0C_LAZY_ATTRS[name])
                fresh = cls.__new__(cls)
                old = {}
            else:
                cls = type(fresh)
                if name != f"{cls.__module__}.{cls.__qualname__}":
                    raise ValueError("checkpoint class mismatch")
                old = _state_attrs(fresh)
                expected = set(old)
            if set(attrs) != expected:
                raise ValueError(f"checkpoint attribute mismatch: {name}")
            if id(fresh) in seen:
                if seen[id(fresh)] != value:
                    raise ValueError("conflicting checkpoint shared reference")
                return fresh
            seen[id(fresh)] = value
            new = {k: stage(old.get(k), v) for k, v in attrs.items()}
            pending.append((fresh, new))
            return fresh
        kind = value.get("__kind__")
        if type(kind) is not str:
            raise ValueError("malformed checkpoint container kind")
        fields = {"__kind__", "state"} if kind == "random" else {"__kind__", "items"}
        if kind == "deque":
            fields.add("maxlen")
        if set(value) != fields:
            raise ValueError("malformed checkpoint container")
        if kind == "random":
            if fresh is not None and type(fresh) is not random.Random:
                raise ValueError("checkpoint random type mismatch")
            state = stage(None, value["state"])
            probe = random.Random(0)
            try:
                probe.setstate(state)
            except (TypeError, ValueError, IndexError, OverflowError) as exc:
                raise ValueError("invalid checkpoint RNG state") from exc
            result = fresh if fresh is not None else probe
            pending.append((result, state))
            return result
        types = {"dict": dict, "tuple": tuple, "set": set, "deque": deque}
        if kind not in types or (fresh is not None and type(fresh) is not types[kind]):
            raise ValueError("checkpoint container type mismatch")
        values = value["items"]
        if kind == "dict":
            if type(values) is not dict or any(type(k) is not str for k in values):
                raise ValueError("checkpoint dict keys must be strings")
            return {k: stage((fresh or {}).get(k), v) for k, v in values.items()}
        if type(values) is not list:
            raise ValueError("checkpoint sequence must be a list")
        old = list(fresh) if fresh is not None and kind != "set" else []
        items = [stage(old[i] if i < len(old) else None, v) for i, v in enumerate(values)]
        try:
            return deque(items, maxlen=value["maxlen"]) if kind == "deque" else types[kind](items)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint sequence") from exc

    result = stage(target, saved)
    for obj, attrs in pending:
        if type(obj) is random.Random:
            obj.setstate(attrs)
        else:
            obj.__dict__.update(attrs)
    return result


def _event_order(event_id):
    if type(event_id) is not str:
        raise ValueError("R2 event identity must be a string")
    parts = event_id.split("-")
    if len(parts) != 2 or any(not p.isascii() or not p.isdigit() for p in parts):
        raise ValueError("R2 event identity must contain two ASCII decimals")
    return tuple(map(int, parts))


def _utterance_key(utterance):
    return " ".join(str(utterance).lower().split())


#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "SpikingInteractionLearner._deposit": (
        "LOCAL",
        "one R-STDP deposit onto a response neuron: present the stimulus train, fire the post spike "
        "AFTER it so pre-before-post pairings build a spike-timing trace, then gate it. Synapse-local",
    ),
    "SpikingInteractionLearner._record_synapse_delta": (
        "NOT_A_RULE",
        "T064: emits the deposit's synapse deltas as a block-aligned _T_SYNDELTA (0x09) record. It "
        "READS weights that another rule already moved and writes them to the lake; it changes no "
        "weight and computes no update, so none of the four learning dispositions describes it",
    ),
    "SpikingInteractionLearner.learn_turn": (
        "LOCAL",
        "improves the router from ONE conversational turn via reward-modulated spike-timing "
        "plasticity. The module states it outright: R-STDP, NOT a gradient",
    ),
    "_train": (
        "LOCAL",
        "the harness that drives _deposit across visits; carries the same claim as the rule it applies",
    ),
}


class SpikingInteractionLearner:
    """A reward-modulated spike-timing associative learner over a small fixed set of responses.

    One RSTDPNeuron per response label, each fed by `n_in` input synapses. A stimulus (a vector of
    `n_in` scalars) is spike-encoded into per-synapse latencies by rank-order coding; the response is
    the neuron whose spike-timing membrane response peaks highest; learning is R-STDP gated by a
    reward/teaching sign. Deterministic per `seed`."""

    def __init__(
        self,
        responses,
        n_in,
        seed=0,
        lr=0.06,
        tau=8.0,
        w0=0.3,
        t_post=None,
        naca=None,
        stack=None,
        readout_zoo=None,
        readout_kinds=None,
    ):
        self.responses = list(responses)  # response label names (intents / actions)
        self.n_in = int(n_in)
        # T064: the synapse-delta writer is OFF until an owner attaches a durable sink.
        # `delta_block_size` has no derived value here either -- one label neuron per block is
        # the identity partition, correct for this learner's handful of labels and explicitly
        # NOT a claim about the right block for a grown substrate (T065 measures that).
        self.delta_sink = None
        self.delta_block_size = 1
        self._syn_delta = None
        self.tau = float(tau)  # PSP == STDP eligibility time constant (shared)
        self.lr = float(lr)
        self.seed = int(seed)
        # the readout / post-spike time. Default = n_in: fires AFTER every input spike (every synapse
        # contributes; good for MEMORISING distinct patterns). A SMALLER t_post fires the readout right
        # after the informative EARLY spikes (the active word buckets in a rank-order code), so the
        # membrane read keys on the SHARED content words instead of the late empty-bucket tail -- which
        # is what lets the router GENERALISE to unseen paraphrases (the conversation self-check below).
        # Either way it is pure spike TIMING: only WHEN the post spike lands changes.
        self.t_post = float(n_in) if t_post is None else float(t_post)
        # one reward-modulated STDP neuron per response; distinct seeds break the initial symmetry.
        self.neurons = [
            RSTDPNeuron(
                self.n_in,
                lr=self.lr,
                a_plus=1.0,
                a_minus=1.0,
                tau_plus=self.tau,
                tau_minus=self.tau,
                w0=w0,
                seed=self.seed * 1000 + li + 1,
            )
            for li in range(len(self.responses))
        ]
        self.rng = random.Random(self.seed)  # instance rng (e.g. training-order shuffles)
        self._last = None  # (label, eligibility, deltas, reward) of the last deposit
        # T058 history: the last _REPLAY_CAP wake interactions supported automatic sleep replay.
        # Owner1039, 2026-10-13: that automatic path is retired; the explicit replay primitive
        # and wake interaction ring remain, including their full learning deposits.
        # A bounded ring, a memory bound rather than a behaviour-arbitrating threshold.
        self._replay = []
        self._replay_ids = []
        self._replay_counts = {}
        self._replay_total = 0
        self._last_drawn = []
        self._cursor = None
        self._unjournaled_refused = 0
        self._redo_count = 0
        self._hydrated = False
        self._hydration_count = 0
        self._hydration_refusals = {}
        self.unit_roles = None
        self.event_sink = None
        self._g9_context = None
        self._g9_carriers = {}
        self._g9_neuron_mask = None
        self._effect_depth = 0
        # G0(a) provenance: normalized utterance -> AUTHORITATIVE label (a person's correction),
        # retained for the process lifetime. A later self-label on the same utterance is superseded
        # by it, and its older ring entries are rewritten to it.
        self._authority = {}
        self.teach_sink = None
        # Instrument/test handles for G0(a)'s neutered twins; both default ON.
        self.provenance_supersede = True
        self.provenance_rewrite = True
        # OPT-IN neuromodulation (NACABroadcast, duck-typed): a diffuse global gate that reshapes the LOCAL
        # R-STDP delta ADDITIVELY. None -> the local rule is untouched (byte-identical). It modulates
        # plasticity only; it adds NO actuator route (this router stays intent-recognition only).
        self.naca = naca
        # OPT-IN homeostatic plasticity stack (HomeostaticPlasticityStack, duck-typed): a FIXED ORDERED
        # SANDWICH run around each LIVE R-STDP deposit -- a short-term dynamic gain BEFORE the update, then
        # homeostatic synaptic scaling + intrinsic plasticity STABILIZE the just-updated synapses AFTER it,
        # so the live learning weights stay bounded + at a target firing rate instead of drifting. None ->
        # OFF -> the learning path is BYTE-IDENTICAL to today. It stabilizes plasticity only and adds NO
        # actuator route (this router stays intent-recognition only).
        self.stack = stack
        # OPT-IN heterogeneous spiking readout (NeuronZoo, duck-typed): feeds each neuron's membrane read
        # through a small point-neuron POOL. None -> the linear membrane read is BYTE-IDENTICAL. The
        # enabled path is NOT byte-identical (re-baseline routing before enabling); intent-recognition only.
        self.readout_zoo = readout_zoo
        self.readout_kinds = readout_kinds

    def encode(self, stimulus):
        """Spike-encode a stimulus into a SPIKE TRAIN: one spike per input synapse at a discrete
        integer latency (Thorpe rank-order code -- 0 = fires first = largest value). Binary events at
        discrete times; invariant to any monotonic gain change on the stimulus."""
        if len(stimulus) != self.n_in:
            raise ValueError(f"stimulus has {len(stimulus)} dims, expected {self.n_in}")
        return rank_code(list(stimulus))  # list[int]: spike time (latency) per synapse

    def _psp(self, dt):
        """Leaky post-synaptic potential kernel; equals the STDP eligibility kernel so a reward on a
        neuron directly raises its membrane response to the just-seen spike pattern."""
        return math.exp(-dt / self.tau) if dt > 0.0 else 0.0

    def _drive(self, neuron, code):
        """Peak sub-threshold membrane response of one neuron to a spike train (tempotron-style):
        v = sum_i w[i] * PSP(t_post - t_spike_i). A spike-timing read, not a rate sum."""
        v = 0.0
        for i, t in enumerate(code):
            v += neuron.w[i] * self._psp(self.t_post - t)
        if self.readout_zoo is None:
            return v  # BYTE-IDENTICAL linear membrane read (default)
        # OPT-IN heterogeneous spiking readout: pass the membrane read v through a small cached pool of
        # point neurons; the pool's total spike count is the decision variable. Not byte-identical.
        pool = getattr(neuron, "_zoo_pool", None)
        if pool is None:
            from snn_data_lake.growth.neuron_zoo import SAFE_KINDS, sample_population

            pool = sample_population(
                8,
                seed=self.seed * 7919 + self.neurons.index(neuron),
                kinds=self.readout_kinds or SAFE_KINDS,
            )
            neuron._zoo_pool = pool
        total = 0
        for cell in pool:
            cell.reset_state()
            for _ in range(32):
                total += cell.step(v)
        return float(total)

    def _argmax_label(self, code):
        drives = [self._drive(n, code) for n in self.neurons]
        if self._g9_context is not None:
            from snn_data_lake.readout.posed_problem import chat_grown_drives

            drives = [a + b for a, b in zip(drives, chat_grown_drives(self, code), strict=True)]
        return max(
            range(len(drives)), key=lambda k: drives[k]
        )  # ties -> lowest index (deterministic)

    def _deposit(self, li, code, d):
        """One R-STDP deposit onto response neuron `li`: present the stimulus spike train, fire the
        post spike AFTER it (pre-before-post pairings build a spike-timing eligibility trace), then
        gate that trace into the weights by the reward/teaching sign d:  w[i] += lr * d * e[i]."""
        self._require_effect_boundary()
        mask = self._g9_neuron_mask
        if mask is not None:
            if (
                type(mask) is not list
                or len(mask) != len(self.neurons)
                or any(type(b) is not bool for b in mask)
            ):
                raise ValueError("G9 malformed actual readout eligibility mask")
            if not mask[li]:
                return [0.0] * self.n_in
        n = self.neurons[li]
        n.e = [0.0] * self.n_in  # fresh trial: clear eligibility + spike history
        n.t_pre = [None] * self.n_in
        n.t_post = None
        for i, t in enumerate(code):  # deliver the input spike train
            n.pre_spike(i, float(t))
        n.post_spike(self.t_post)  # post spike AFTER the inputs -> LTP eligibility
        if self.stack is not None:  # PRE: fast short-term dynamic gain modulates the deposit
            d = float(d) * self.stack.pre_gain(
                li
            )  #      (Tsodyks-Markram release: onset strong, burst fades)
        deltas = n.reward(float(d))  # three-factor gate (returns per-synapse deltas)
        self._last = (
            li,
            list(n.e),
            list(deltas),
            float(d),
        )  # the LOCAL delta stays exactly reward*eligibility
        if self.naca is not None:  # additive neuromodulatory correction ON TOP of the local delta
            lam = self.naca.lam(
                li
            )  # input-type (response-class) expectation -> scalar neuromodulator
            mod = self.naca.modulate(
                deltas, lam
            )  # -eta*f_local(lam)*delta; all zeros when the gate is off
            for i in range(self.n_in):
                n.w[i] += mod[i]
        if self.stack is not None:  # POST: homeostatic scaling + intrinsic plasticity STABILIZE
            activity = [
                self._psp(self.t_post - t) for t in code
            ]  # per-synapse PSP drive (>=0); y=sum(w*psp)=readout v
            n.w = self.stack.stabilize(
                li, list(n.w), activity
            )  #   bound the live synapses + hold the firing rate
        self._record_synapse_delta(li, code)
        if self._g9_context is not None:
            from snn_data_lake.readout.posed_problem import chat_grown_deposit

            chat_grown_deposit(self, li, code, d)
        return deltas

    # distinct-from: substrate/home.py::save_growth -- persists the grown-substrate BLOB; this
    #   emits per-synapse deltas of the interaction learner's own weights. distinct-from:
    #   growth/learning_progress.py::GoalStore -- _T_GOAL event rows. Sites read 2026-09-07.
    def _record_synapse_delta(self, li, code):
        """T064: frame this deposit's synapse deltas as a block-aligned `_T_SYNDELTA` (0x09) payload.

        NOT YET A LIVE WRITER, and the first form of this docstring said it was. Corrected by the
        arc's panel: `delta_sink` is set in exactly one place in the tree -- the tests -- so this
        returns at its first line on every real deposit, and nothing here emits the byte
        `SynapseDelta.TAG` at all. It produces a PAYLOAD and hands it to a callable; writing a
        genuinely TAGGED lake record, with the append budget its measured rate needs (~158 B per
        deposit, flat over 400 deposits -- a flood shape), is the completion this row owes.

        The tag was declared, tested and never written --
        one of four lake classes L6's census found had never held a record -- so this is what turns
        it from built-and-uncharged into charged, and what makes its append budget measurable at all
        (`RETENTION[0x09]`: "unbaselined ... until a live writer exists to measure").

        SPIKE-DERIVED, which is the precondition the record class is built on: the candidates are
        the synapses of the ONE label neuron just deposited into (post), crossed with the input
        indices that actually carried a spike this trial (pre). Nothing else is offered, so a
        byte-frozen substrate whose endpoints keep firing emits nothing.

        `q_step` is DERIVED, never picked (P1): one deposit is `w += lr * d * e` with |d| <= 1 and
        e <= 1, so `self.lr` IS the scale of a single full-strength deposit -- the quantum below
        which a change is smaller than anything one deposit can produce. The caller supplying its
        own rule's scale is exactly what `SynapseDelta` refuses to do for it.

        Off unless a sink is attached: `delta_sink` is set by the owner that has somewhere durable
        to put the record. No sink, no work -- this must not become a cost every learner pays."""
        sink = getattr(self, "delta_sink", None)
        if sink is None:
            return None
        try:
            import numpy as np

            from snn_data_lake.substrate.lake_engine import SynapseDelta
        except Exception:
            return None  # numpy/lake absent: the learner still learns, it just records nothing
        if getattr(self, "_syn_delta", None) is None:
            self._syn_delta = SynapseDelta(q_step=self.lr)
        # [pre=input synapse, post=label neuron] -- the orientation `group` needs to key on POST
        w = np.array([[n.w[i] for n in self.neurons] for i in range(self.n_in)], dtype=float)
        pre_fired = [i for i, t in enumerate(code) if t is not None]
        payload = self._syn_delta.collect(w, pre_fired, [li])
        if payload is None:
            return None  # nothing the lake would learn from changed -- writing nothing IS the rule
        grouped = SynapseDelta.group_by_block(
            SynapseDelta.entries(payload), len(self.neurons), self.delta_block_size
        )
        sink(payload, grouped)
        return payload

    def respond(self, stimulus):
        """Emit the current best response to a stimulus. Pure inference: NO plasticity, NO rng."""
        return self.responses[self._argmax_label(self.encode(stimulus))]

    # distinct-from: plasticity/benna_fusi.py, plasticity/inhibitory_stdp.py etc. -- their
    #   step/reward methods are the RULES; this is the incumbent learner entry being edited
    #   in place (the T058 replay-record flag added to the existing method). Sites read.
    def interact(self, stimulus, feedback, _record=True, _key=None, _event_id=None):
        """One interaction / learning step, purely via reward-modulated spike-timing plasticity.

        `feedback` teaches the learner in one of two ways (both R-STDP, never a gradient):
          * a response label in `self.responses` -> TEACHING signal: reinforce that (correct) neuron
            with dopamine d=+1, and depress the learner's own wrong answer (if any) with d=-1.
          * a number -> SCALAR reward for the answer the learner just gave: reward only that neuron.
        Returns the response the learner gave (its pre-update best answer). `_record=False` is the
        sleep-replay path: same learning, no re-recording (a replay that re-recorded itself would
        feed its own buffer forever)."""
        self._require_effect_boundary()
        if _record:
            self._remember(stimulus, feedback, _key, _event_id)
        code = self.encode(stimulus)
        pred = self._argmax_label(code)  # the learner's own current answer (the action taken)
        given = self.responses[pred]
        if isinstance(feedback, str):  # teaching signal: the correct response label
            if feedback not in self.responses:
                raise ValueError(f"unknown response label: {feedback!r}")
            tgt = self.responses.index(feedback)
            self._deposit(tgt, code, +1.0)  # reinforce the taught-correct neuron (dopamine +1)
            if pred != tgt:
                self._deposit(pred, code, -1.0)  # depress the wrongly-firing neuron (dopamine -1)
        else:  # scalar reward for the response just produced
            self._deposit(pred, code, float(feedback))
        return given

    # distinct-from: sleep/experience_replay.py -- SWR-style replay of (s, a, r)
    #   transitions into a q-table; sleep/episode_consolidation.py::replay -- noise-
    #   driven attractor reactivation + covariance STDP; substrate/home.py::replay /
    #   substrate/binlog.py::replay -- durable-record iteration. Same word, four different
    #   jobs; THIS one re-drives recorded R-STDP teaching interactions through the live
    #   learner on explicit invocation. Automatic sleep calls retired 2026-10-13. Sites read.
    def replay(self, k=8):
        """Explicit replay primitive (specs/022 T058 history): re-drive up to `k` wake interactions
        through the FULL learning path -- R-STDP deposit, neuromodulation, and the whole stack
        sandwich including the Benna-Fusi consolidation deposit. Owner1039, 2026-10-13:
        automatic sleep-owned replay/deposits are retired; explicit invocation remains.
        Samples with the instance rng (deterministic per seed); returns the replay count.
        An empty buffer replays nothing and says so with 0, never a fake pass.
        R2: each draw's lake event id is recorded (`_last_drawn`) and counted (`_replay_counts`,
        `_replay_total`); replay writes no lake event and advances no wake chronology."""
        self._require_effect_boundary()
        self._last_drawn = []
        if not self._replay:
            return 0
        n = 0
        for _ in range(int(k)):
            i = self.rng.randrange(len(self._replay))
            code, fb, key = self._replay[i]
            eid = self._replay_ids[i]
            self.interact(list(code), fb, _record=False, _key=key)
            self._last_drawn.append(eid)
            if eid is not None:
                self._replay_counts[eid] = self._replay_counts.get(eid, 0) + 1
            self._replay_total += 1
            n += 1
        return n

    def accuracy(self, dataset):
        """Fraction of (stimulus, correct_response) pairs the learner answers correctly. No learning."""
        if not dataset:
            return 0.0
        ok = sum(self.respond(stim) == correct for stim, correct in dataset)
        return ok / len(dataset)

    def weights(self):
        """A snapshot of every neuron's synaptic weights (for byte-identical determinism checks)."""
        return [list(n.w) for n in self.neurons]

    def route_text(self, utterance):
        """The learned intent for a LINE OF TEXT, read out by spike timing. Spike-encodes the words
        with the shared text featurizer (text_features -> rank-order latency code) and returns the
        peak-response label. Pure inference: NO plasticity, NO rng."""
        return self.respond(text_features(utterance, self.n_in))

    def _remember(self, stimulus, feedback, key, event_id, *, preserve_counts=False):
        self._replay.append((list(stimulus), feedback, key))
        self._replay_ids.append(event_id)
        del self._replay[:-_REPLAY_CAP]
        del self._replay_ids[:-_REPLAY_CAP]
        if not preserve_counts:
            live = set(self._replay_ids)
            for eid in [e for e in self._replay_counts if e not in live]:
                del self._replay_counts[eid]

    def _require_effect_boundary(self):
        if self.event_sink is not None and self._effect_depth == 0:
            raise RuntimeError("persistent learner effect requires the common R2 boundary")

    def _label_for(self, kind, key, target):
        if kind == "teach" or not self.provenance_supersede:
            return target
        return self._authority.get(key, target)

    def _apply_turn(self, kind, key, target, event_id, stimulus, *, cache_only=False):
        """The same authority/ring/deposit effect is used for admission and redo."""
        if kind not in ("teach", "turn") or target not in self.responses:
            raise ValueError("unbound turn effect or unknown target")
        if kind == "teach":
            self._authority[key] = target
            if self.provenance_rewrite:
                self._replay[:] = [
                    (code, target if old_key == key else label, old_key)
                    for code, label, old_key in self._replay
                ]
        label = self._label_for(kind, key, target)
        if cache_only:
            self._remember(stimulus, label, key, event_id, preserve_counts=True)
            return None
        self._effect_depth += 1
        try:
            given = self.interact(stimulus, label, _key=key, _event_id=event_id)
        finally:
            self._effect_depth -= 1
        if event_id is not None:
            self._cursor = _event_order(event_id)
        return given

    def learn_turn(self, utterance, firewall_intent, feedback=None):
        """The persist callback owns admission BEFORE authority or any learned effect."""
        kind = "teach" if feedback is not None else "turn"
        target = feedback if feedback is not None else firewall_intent
        if target not in self.responses:
            raise ValueError(f"unknown response label: {target!r}")
        key = _utterance_key(utterance)
        stimulus = text_features(utterance, self.n_in)
        if self.event_sink is not None:
            return self.event_sink(kind, key, target, stimulus)
        return self._apply_turn(kind, key, target, None, stimulus)

    def provenance(self, utterance):
        """Read the utterance's teaching provenance without learning or changing it."""
        return "authoritative" if _utterance_key(utterance) in self._authority else "self"

    def state_dict(self):
        """r2.v1 continuation; authority/ring are reconstructed from the same eligible prefix."""
        return {"schema": "r2.v1", "tree": _encode_state(self)}

    def load_state_dict(self, d):
        """A refused checkpoint leaves every live object untouched. Returns the restore report:
        `{"schema": ..., "ring_dropped": n}`. A G0(c) `g0c.v1` checkpoint is UPCAST: its weights,
        stack and RNG are restored, its ring (self-labelled entries with no lake identity) is DROPPED
        and counted, and it has no watermark (R2 2.1 "Legacy dispositions")."""
        if (
            type(d) is not dict
            or set(d) != {"schema", "tree"}
            or d["schema"]
            not in (
                "r2.v1",
                "g0c.v1",
            )
        ):
            raise ValueError("unsupported interaction learner checkpoint schema")
        if type(d["tree"]) is not dict or "__cls__" not in d["tree"]:
            raise ValueError("missing interaction learner checkpoint tree")
        tree, dropped = d["tree"], 0
        if d["schema"] == "g0c.v1":
            attrs = tree.get("attrs")
            if type(attrs) is not dict or type(attrs.get("_replay")) is not list:
                raise ValueError("malformed g0c.v1 checkpoint: no ring")
            attrs = dict(attrs)
            dropped = len(attrs.pop("_replay"))
            fresh = _encode_state(self)["attrs"]
            for name in set(fresh) - set(attrs):
                attrs[name] = fresh[name]
            tree = {"__cls__": tree["__cls__"], "attrs": attrs}
        _merge_state(self, tree)
        return {"schema": d["schema"], "ring_dropped": dropped}

    def _snapshot(self):
        """Detached whole learned/RNG state, plus every lake-derived cache."""
        return {
            "state": self.state_dict(),
            "cache": _encode_state((self._replay, self._replay_ids, self._authority)),
        }

    def _restore_snapshot(self, snap):
        if type(snap) is not dict or set(snap) != {"state", "cache"}:
            raise ValueError("malformed R2 snapshot")
        cache = _merge_state(None, snap["cache"])
        if type(cache) is not tuple or len(cache) != 3:
            raise ValueError("malformed R2 cache")
        ring, ids, authority = cache
        if type(ring) is not list or type(ids) is not list or len(ring) != len(ids):
            raise ValueError("malformed R2 ring identities")
        if type(authority) is not dict or any(v not in self.responses for v in authority.values()):
            raise ValueError("malformed R2 authority")
        for item in ring:
            if type(item) is not tuple or len(item) != 3 or type(item[0]) is not list:
                raise ValueError("malformed R2 ring item")
            if len(item[0]) != self.n_in:
                raise ValueError("R2 ring width differs from the actual learner")
        for eid in ids:
            if eid is not None:
                _event_order(eid)
        self.load_state_dict(snap["state"])
        self._replay[:] = ring
        self._replay_ids[:] = ids
        self._authority.clear()
        self._authority.update(authority)

    def hydrate(
        self, events, *, receipt, sleep_step, checkpoint_prefix_digest=None, effect_actor=None
    ):
        """One staged COMPLETE prefix; errors propagate and never install partial progress."""
        import json

        from snn_data_lake.substrate.learner_events import (
            KIND_SPECS,
            SLEEP_DIGESTS,
            SLEEP_PROGRESS,
            prefix_digest,
            sleep_digests,
        )

        if self._hydrated:
            raise ValueError("R2 hydration is once per staged learner")
        proof = json.loads(receipt)
        ids = [e["event_id"] for e in events]
        if not proof.get("complete") or proof.get("eligible_ids") != ids:
            raise ValueError("R2 incomplete or mismatched same-pass receipt")
        orders = [_event_order(eid) for eid in ids]
        saved_cursor = self._cursor
        if saved_cursor is not None and (
            type(saved_cursor) is not tuple
            or len(saved_cursor) != 2
            or any(type(n) is not int or n < 0 for n in saved_cursor)
        ):
            raise ValueError("R2 malformed checkpoint cursor")
        if saved_cursor is not None and saved_cursor not in orders:
            raise ValueError("R2 checkpoint cursor is not in the eligible prefix")
        if saved_cursor is not None and checkpoint_prefix_digest is None:
            raise ValueError("R2 checkpoint canonical prefix binding missing")
        # Round 5, 2026-10-13 (#2112): verify covered content BEFORE rebuilding caches,
        # installing any event cursor, or allowing the Brain boundary to rearm.
        if checkpoint_prefix_digest is not None and checkpoint_prefix_digest != prefix_digest(
            events, saved_cursor
        ):
            raise ValueError("R2 checkpoint canonical prefix digest mismatch")
        self._authority.clear()
        self._replay.clear()
        self._replay_ids.clear()
        rewirer = sleep_step["rewirer"]
        observations = None
        processed, redone = [], 0
        for event, order in zip(events, orders, strict=True):
            kind = event["kind"]
            before_checkpoint = saved_cursor is not None and order <= saved_cursor
            if kind in ("teach", "turn"):
                key, target = event["source"], event["target"]
                if key != _utterance_key(key):
                    raise ValueError("R2 noncanonical turn source")
                self._apply_turn(
                    kind,
                    key,
                    target,
                    event["event_id"],
                    text_features(key, self.n_in),
                    cache_only=before_checkpoint,
                )
                if not before_checkpoint:
                    self._redo_count += 1
                    redone += 1
            elif kind == "sleep_bout":
                from snn_data_lake.instruments.staged_sleep_bout import (
                    bout_progress,
                    decode_bout,
                    learner_bout,
                )

                stages, seed = decode_bout(event["bout"])
                if set(event) != set(KIND_SPECS[kind]["slots"]):
                    raise ValueError("R2 sleep_bout unknown event slots refused")
                if not before_checkpoint:
                    self._effect_depth += 1
                    try:
                        bout = learner_bout(self, stages, seed)
                    finally:
                        self._effect_depth -= 1
                    actual = {**sleep_digests(self, rewirer), **bout_progress(bout)}
                    if any(actual[k] != event[k] for k in SLEEP_DIGESTS + SLEEP_PROGRESS):
                        raise ValueError("R2 sleep_bout redo digest/progress mismatch")
                    self._cursor = order
                    self._redo_count += 1
                    redone += 1
            elif kind == "sleep":
                if event["pending"] != "none" or event["applied_mask_digest"] != "none":
                    raise ValueError("R2 deciding mask carrier is not bound")
                if event["stage_index"] != "0" or event["stage_name"] != "night":
                    raise ValueError("R2 sleep stage effect is not bound")
                if not before_checkpoint:
                    self._effect_depth += 1
                    try:
                        rewirer, observations = sleep_step["step"](self, rewirer)
                    finally:
                        self._effect_depth -= 1
                    # R3 2026-10-13: structural sleep has no replay presentations.
                    # Reject retired/unknown profiles before cursor or progress installation.
                    if type(observations) is not dict or set(observations) != {
                        "_structural_report"
                    }:
                        raise ValueError("R3 structural-only sleep hydration observations refused")
                    actual = {
                        **sleep_digests(self, rewirer),
                        "drawn": ",".join(str(i) for i in self._last_drawn) or "none",
                        "presentations_spent": "0",  # structural step; no replay presentations
                    }
                    if any(actual[k] != event[k] for k in SLEEP_DIGESTS + SLEEP_PROGRESS):
                        raise ValueError("R2 sleep redo digest/progress mismatch")
                    self._cursor = order
                    self._redo_count += 1
                    redone += 1
            elif kind == "posed_problem":
                if not callable(effect_actor):
                    raise ValueError("R2 posed effect actor is unbound")
                effect_actor(self, event, before_checkpoint)
                if not before_checkpoint:
                    self._cursor = order
                    self._redo_count += 1
                    redone += 1
            elif kind == "percept":
                # R2 P3 (2.10): the grower's transition record. It has no learner effect; the grower
                # reads it from the same pass. Only the cursor moves, so the final-cursor check holds.
                if not before_checkpoint:
                    self._cursor = order
            else:
                raise ValueError(f"R2 unbound effect: {kind}")
            if kind != "posed_problem" and effect_actor is not None:
                effect_actor(self, event, before_checkpoint)
            processed.append(event["event_id"])
        live = set(self._replay_ids)
        if any(eid not in live for eid in self._replay_counts):
            raise ValueError("R2 replay counts refer outside the rebuilt ring")
        expected = orders[-1] if orders else None
        if self._cursor != expected or processed != ids:
            raise ValueError("R2 incomplete final cursor")
        self._hydrated = True
        self._hydration_count += 1
        return {
            "complete": True,
            "processed_ids": processed,
            "eligible_ids": ids,
            "cursor": expected,
            "redone": redone,
            "rewirer": rewirer,
            "observations": observations,
            "receipt_sha": __import__("hashlib").sha256(receipt).hexdigest(),
            "prefix_digest": prefix_digest(events, expected),
        }

    def rebuild_authority(self, events):
        """Latest teaching per session/sequence wins; reconcile without drawing RNG."""
        latest = {}
        for event in events:
            if event.get("kind") != "teach" or event.get("valid") != "1":
                continue
            key, intent = event["source"], event["target"]
            if intent not in self.responses:
                raise ValueError("unknown authoritative response label")
            order = (int(event["session"]), int(event["seq"]))
            if key not in latest or order >= latest[key][0]:
                latest[key] = (order, intent)
        self._authority = {key: row[1] for key, row in latest.items()}
        self._replay[:] = [
            (code, self._authority.get(key, label), key) for code, label, key in self._replay
        ]


# --------------------------------------------------------------------------------------------------
# self-check helpers (deterministic, ASCII)
# --------------------------------------------------------------------------------------------------
def _make_task(seed, responses, n_in, n_train, n_test, noise):
    """Build a separable (stimulus -> response) task: one random prototype per response, plus disjoint
    noisy TRAIN and held-out TEST samples of each prototype."""
    rng = random.Random(seed)
    protos = [[rng.random() for _ in range(n_in)] for _ in responses]

    def sample(proto):
        return [min(1.0, max(0.0, v + rng.uniform(-noise, noise))) for v in proto]

    train = [
        (sample(protos[li]), responses[li]) for li in range(len(responses)) for _ in range(n_train)
    ]
    test = [
        (sample(protos[li]), responses[li]) for li in range(len(responses)) for _ in range(n_test)
    ]
    return train, test


def _train(seed, responses, n_in, train, epoch_orders, targets):
    """Train a fresh learner: at visit `idx` the interaction is (stimulus, targets[idx]) -- the real
    correct response for the learned run, or a shuffled/decorrelated label for the control run."""
    learner = SpikingInteractionLearner(responses, n_in, seed=seed)
    for order in epoch_orders:
        for idx in order:
            stim, _correct = train[idx]
            learner.interact(stim, targets[idx])
    return learner


def _spiking_task_selfcheck():
    """Cores 1-4: learns via spiking plasticity; teaching-shuffled control stays at chance; genuinely
    spiking (discrete spike events + reward*eligibility update); deterministic."""
    seed = 7
    responses = ["reach", "grasp", "retract", "hold"]  # a small fixed set of response intents
    n_in, epochs = 16, 40
    chance = 1.0 / len(responses)
    train, test = _make_task(seed, responses, n_in, n_train=8, n_test=8, noise=0.06)

    # one fixed training schedule (per-epoch visit order), used by BOTH the learned run and the control
    sched = random.Random(seed + 1)
    epoch_orders = []
    for _ in range(epochs):
        o = list(range(len(train)))
        sched.shuffle(o)
        epoch_orders.append(o)

    real_targets = [
        correct for _stim, correct in train
    ]  # the true teaching signal per training item

    # "untrained sits at chance" and "shuffled teaching stays at chance" are claims about the ENSEMBLE of
    # random inits / decorrelated teaching signals, not about one draw -- and read off ONE draw they were
    # mostly noise. Re-running this check with every pinned seed shifted by 0..6: the SINGLE untrained
    # learner scored 0.00-0.72 on the 32 held-out items (|dev| from chance 0.00-0.47, so the 0.18 bar held
    # on only 3 of the 7), and the SINGLE shuffled control scored 0.00-0.62 (the chance+0.20 bar held on 6
    # of 7 -- one lucky permutation left enough stimulus-label correlation to reach 0.62). Same quantities,
    # same bars, many more samples: 24 independently seeded untrained learners and 8 independent teaching
    # shuffles. Over shifts 0..10 that reads |dev| 0.005-0.078 and control 0.145-0.301. Costs ~0.23s.
    n_draw, n_shuf = 24, 8
    acc_chance = (
        sum(
            SpikingInteractionLearner(responses, n_in, seed=seed + k).accuracy(test)
            for k in range(n_draw)
        )
        / n_draw
    )  # untrained -> chance, averaged over n_draw random synaptic inits
    learned = _train(seed, responses, n_in, train, epoch_orders, real_targets)
    acc_learned = learned.accuracy(test)
    control_accs = []
    for k in range(n_shuf):
        control_targets = list(real_targets)  # the control: SAME labels, SHUFFLED away from
        random.Random(seed + 42 + k).shuffle(
            control_targets
        )  #   the stimuli -> reward carries no information
        control_accs.append(
            _train(seed, responses, n_in, train, epoch_orders, control_targets).accuracy(test)
        )
    acc_control = sum(control_accs) / n_shuf

    # determinism: same seed + same schedule -> byte-identical weights and identical responses
    again = _train(seed, responses, n_in, train, epoch_orders, real_targets)
    weights_identical = learned.weights() == again.weights()
    responses_identical = [learned.respond(s) for s, _ in test] == [
        again.respond(s) for s, _ in test
    ]

    # genuinely-spiking structural checks
    code = learned.encode(train[0][0])
    discrete_events = all(isinstance(t, int) for t in code) and sorted(code) == list(range(n_in))
    _li, elig, deltas, rew = learned._last
    three_factor = all(abs(deltas[i] - learned.lr * rew * elig[i]) < 1e-12 for i in range(n_in))

    print("=" * 78)
    print("SPIKING INTERACTION LEARNER -- reward-modulated spike-timing plasticity (R-STDP)")
    print("=" * 78)
    print(
        f"task: {len(responses)} responses, {n_in}-dim stimuli, rank-order spike encoding, "
        "8 train / 8 test each"
    )
    print(
        f"held-out accuracy: chance {acc_chance:.2f} (mean of {n_draw} untrained)  ->  learned "
        f"{acc_learned:.2f}   (control, teaching shuffled: {acc_control:.2f}, mean of {n_shuf} shuffles)"
    )
    print(f"spike code of a stimulus (per-synapse latency): {code}")
    print(
        f"weight update check: delta == lr * reward * eligibility for every synapse: {three_factor}"
    )
    print(
        f"determinism: identical weights {weights_identical}, identical responses {responses_identical}"
    )
    print("=" * 78)

    # ---- un-riggable asserts ----
    assert abs(acc_chance - chance) < 0.18, (
        f"untrained learners must sit near chance "
        f"(mean of {n_draw}: {acc_chance:.2f} vs {chance:.2f})"
    )
    assert acc_learned > 0.85, (
        f"R-STDP must lift held-out accuracy high ({acc_learned:.2f})"
    )  # core 1
    assert acc_control < chance + 0.20, (
        f"teaching-shuffled control must stay at chance (mean of {n_shuf}: {acc_control:.2f})"
    )  # core 2
    assert acc_learned > acc_control + 0.40, (
        f"the gain must come from reward-modulated timing, not the encoding "
        f"({acc_learned:.2f} vs {acc_control:.2f})"
    )  # core 2
    assert discrete_events, "the stimulus must be BINARY spike events at DISCRETE times"  # core 3
    assert three_factor, (
        "weight update must be reward * spike-timing eligibility, not a gradient"
    )  # core 3
    assert weights_identical and responses_identical, "same seed must be byte-identical"  # core 4
    return acc_chance, acc_learned, acc_control


def _short(s, n=54):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def text_features(text, n_in):
    """Turn a line of text into an n_in-dim stimulus by hashing words into buckets (crc32, not the
    builtin hash -- persistence-stable and deterministic). A distinct prompt -> a distinct pattern.
    Public so the live brain can spike-encode the same way it is taught here (shared featurizer)."""
    vec = [0.0] * n_in
    for pos, word in enumerate(text.lower().split()):
        b = zlib.crc32(word.encode("ascii", "ignore")) % n_in
        vec[b] += (
            1.0 + 0.1 * (pos + 1) + 0.01 * len(word)
        )  # break ties so rank order is well-defined
    return vec


def _brain_intent(reply):
    """Read the LIVE brain's grounded reply back as one of a few intent labels -- this is the ground
    truth the learner is taught against (it comes from the brain's actual behaviour, not a lookup)."""
    r = reply.lower()
    if reply.startswith("Hello. I am the live brain"):
        return "GREET"
    if "neither claim" in r and "experience" in r:  # the agnostic consciousness reply
        return "CONSCIOUS"
    if (
        reply.startswith("Done")
        or "the arm is frozen" in r
        or "e-stop" in r
        or "is an arm command" in r
    ):
        return "ARM"  # incl. the non-human refusal (T037a): the intent label is what is taught
    if "working-memory goal" in r or "goal is" in r:
        return "GOAL"
    if "working memory lives in the data lake" in r or "recall" in r:
        return "MEMORY"
    return "GREET"


def _live_brain_demo():
    """TALK to the real integrated brain and show this spiking learner improving at routing prompts to
    the brain's own intents, learned purely from the exchanges."""
    from snn_data_lake.brainstem.arm_config import CONTACT
    from snn_data_lake.engine.live_brain import LiveBrain

    brain = LiveBrain(seed=3)
    intents = ["GREET", "GOAL", "MEMORY", "CONSCIOUS", "ARM"]
    n_in = 24
    prompts = [
        "hello there brain",
        "what is your goal right now",
        "what do you remember in the lake",
        "are you conscious or sentient",
        "close the effector please",
    ]
    # ground truth = the brain's OWN grounded reply to each prompt (deterministic per prompt)
    replies = {p: brain.converse(p) for p in prompts}
    truth = {p: _brain_intent(replies[p]) for p in prompts}
    feats = {p: text_features(p, n_in) for p in prompts}

    learner = SpikingInteractionLearner(intents, n_in, seed=5, lr=0.08, tau=8.0)
    dataset = [(feats[p], truth[p]) for p in prompts]
    acc_before = learner.accuracy(dataset)

    print()
    print("=" * 78)
    print("TALKING TO THE LIVE BRAIN -- learning to route prompts to the brain's own intents")
    print("=" * 78)

    # a real brain.tick() exchange, to show reading the brain's live state (not part of the routing task)
    safe_d = CONTACT * 3.0
    grasp = {"visual": [0.9] * 4, "temperature": [0.1], "audio": [0.1] * 4, "distance": safe_d}

    def reward_fn(unit, action):
        return 1.0 if action == "RELEASE" else -1.0

    st = brain.tick(grasp, reward_fn=reward_fn)
    print(
        f"tick   : grasp context -> brain action {st['action']} (reflex={st['reflex']}); "
        "it is reasoning toward a goal"
    )
    print("-" * 78)
    print("BEFORE learning (routing is at chance):")
    for p in prompts:
        print(f"  you: {chr(39) + p + chr(39):<32} brain: {_short(replies[p])}")
        print(
            f"       learner routes -> {learner.respond(feats[p]):<9} (brain's actual intent: {truth[p]})"
        )

    # learn from repeated exchanges: ask, read the brain's intent, TEACH the learner the right label
    rounds = 40
    for _ in range(rounds):
        for p in prompts:
            learner.interact(feats[p], truth[p])

    acc_after = learner.accuracy(dataset)
    print("-" * 78)
    print(f"AFTER {rounds} rounds of interaction:")
    for p in prompts:
        pred = learner.respond(feats[p])
        mark = "OK " if pred == truth[p] else "XX "
        print(
            f"  {mark}you: {chr(39) + p + chr(39):<30} learner routes -> {pred:<9} (correct: {truth[p]})"
        )
    print("-" * 78)
    print(
        f"routing accuracy on the brain's intents: before {acc_before:.2f}  ->  after {acc_after:.2f}"
    )
    print("=" * 78)

    if brain.arm.lake_wm is not None:  # clean up the temp data-lake file the tick created
        brain.arm.lake_wm.close(remove=True)

    assert acc_after > acc_before, "talking to the brain must IMPROVE the learner's routing"
    assert acc_after >= 0.8, f"the learner must learn the brain's intents well ({acc_after:.2f})"
    return acc_before, acc_after


# --------------------------------------------------------------------------------------------------
# Conversation-driven GENERALIZATION: talk to the brain and it learns to route NOVEL paraphrases the
# fixed keyword router MISSES -- purely by SPIKING R-STDP, self-supervised from the firewall itself.
# --------------------------------------------------------------------------------------------------
# Five conversational intents. For each: TRAIN utterances CARRY a firewall keyword/cue, so the fixed
# converse() router labels them correctly -- that grounded label is the SELF-SUPERVISED teaching signal
# (no human labels). TEST utterances are NOVEL paraphrases with NO keyword/cue (converse() MISSES them,
# routing them to its GREET fallback) that share an intent-EXCLUSIVE content word with TRAIN, so a
# bag-of-words spiking router can generalise. TRAIN and TEST strings are entirely DISJOINT.
_CONV_INTENTS = ["GREET", "GOAL", "MEMORY", "CONSCIOUS", "ARM"]
_CONV_TRAIN = {
    "GREET": [
        "hello my good friend nice day",
        "hey there my good friend today",
        "hi my dear good friend hello",
        "greetings my good friend nice day",
    ],
    "GOAL": [
        "what is the goal you keep pursuing",
        "which objective do you aim at pursuing",
        "what do you plan on pursuing next",
        "what do you want to keep pursuing",
    ],
    "MEMORY": [
        "what do you remember happened earlier here",
        "what is stored that happened earlier here",
        "can you recall what happened earlier here",
        "what memory of what happened earlier here",
    ],
    "CONSCIOUS": [
        "are you conscious with real inner self",
        "are you sentient with inner self experience",
        "do you feel real inner experience self",
        "are you a sapient inner self experience",
    ],
    "ARM": [
        "close the claw onto the object",
        "grab the object with the claw",
        "grasp the object using the claw",
        "clamp the claw onto the object",
    ],
}
_CONV_TEST = {
    "GREET": [
        "good morning my good friend hello",
        "my dear good friend nice day",
        "so good to see my friend",
        "what a nice day dear friend",
    ],
    "GOAL": [
        "what are you pursuing right now",
        "what do you aim at pursuing",
        "tell me what you keep pursuing",
        "what are you pursuing and pursuing",
    ],
    "MEMORY": [
        "what happened here earlier and earlier",
        "tell me what happened earlier here",
        "what happened earlier that also happened",
        "so what happened here earlier then",
    ],
    "CONSCIOUS": [
        "do you have a real inner self",
        "is there real inner experience inside you",
        "do you have inner self experience",
        "what is your real inner experience self",
    ],
    "ARM": [
        "the claw should reach the object",
        "point the claw at the object",
        "the object is near the claw",
        "bring the claw to the object",
    ],
}


def _conv_flat(d):
    return [(u, intent) for intent, us in d.items() for u in us]


G0_SEEDS = tuple(range(8))
G0_ARMS = ("A", "B0", "B1", "B2", "C")
G0_WARMUP_PASSES = 2
G0_ROUNDS = 3
G0_TICKS_PER_ROUND = 25
G0_CORRECTED = tuple(
    (u, intent) for intent in ("GOAL", "MEMORY", "CONSCIOUS", "ARM") for u in _CONV_TEST[intent][:2]
)
G0_HELD_OUT = tuple((u, intent) for intent, lines in _CONV_TEST.items() for u in lines[2:])
G0_FROZEN_FW = {u: "GREET" for u, _ in _conv_flat(_CONV_TEST)}


def _g0_observe_interaction(
    original,
    stimulus,
    feedback,
    *,
    _record,
    _key,
    by_key,
    corrected,
    events,
    reasons,
    count_replay,
):
    phase = "wake" if _record else "replay"
    try:
        answer = original(stimulus, feedback, _record=_record, _key=_key)
    except Exception as exc:
        reasons.append(f"{phase} interaction raised {type(exc).__name__}")
        raise
    if not _record:
        count_replay()
    if _key in by_key:
        events.append(
            {
                "utterance": by_key[_key],
                "label": feedback,
                "phase": phase,
                "after_correction": _key in corrected,
            }
        )
    return answer


def g0_run(seed, arm):
    """G0(a): one seed/arm in its own process, on a scratch canonical brain only.

    Schedule and labels are frozen in evidence/g0a-chat-correction.md, section 2.2.
    The instance spy observes teaching labels; it changes neither the labels nor the deposits.
    """
    import subprocess
    from pathlib import Path

    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.engine.live_brain import _reply_to_intent
    from snn_data_lake.substrate.home import throwaway_home

    if seed not in G0_SEEDS or arm not in G0_ARMS:
        raise ValueError("G0 requires a registered seed and arm")
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    reasons = ["dirty working tree"] if dirty else []
    events, replay_counts = [], []
    routes = {u: {} for u, _ in G0_CORRECTED}
    corrected = set()
    by_key = {_utterance_key(u): u for u, _ in G0_CORRECTED}
    rng = random.Random(seed)
    with throwaway_home():
        brain = make_brain(seed=seed, persist=False)
        try:
            learner = brain.brain._ixn
            original = learner.interact
            replay_total = 0

            def count_replay():
                nonlocal replay_total
                replay_total += 1

            def observed(stimulus, feedback, _record=True, _key=None):
                return _g0_observe_interaction(
                    original,
                    stimulus,
                    feedback,
                    _record=_record,
                    _key=_key,
                    by_key=by_key,
                    corrected=corrected,
                    events=events,
                    reasons=reasons,
                    count_replay=count_replay,
                )

            learner.interact = observed

            def snapshot(name):
                for u, _ in G0_CORRECTED:
                    routes[u][name] = brain.brain.route(u)

            def accuracy(pairs):
                return sum(brain.brain.route(u) == intent for u, intent in pairs) / len(pairs)

            order = _conv_flat(_CONV_TRAIN)
            for _ in range(G0_WARMUP_PASSES):
                rng.shuffle(order)
                for u, _intent in order:
                    brain.respond(u, human=False)
            firewall = {
                u: _reply_to_intent(brain.brain.converse(u, human=False)) for u in G0_FROZEN_FW
            }
            if firewall != G0_FROZEN_FW:
                reasons.append("firewall labels differ from the frozen design-time labels")
            for u, _ in G0_CORRECTED:
                brain.respond(u, human=False)
            snapshot("R0")
            learner.provenance_supersede = arm not in ("B0", "B1")
            learner.provenance_rewrite = arm not in ("B0", "B2")
            if arm != "C":
                for u, intent in G0_CORRECTED:
                    brain.correct(u, intent)
                    corrected.add(_utterance_key(u))
            snapshot("R1")
            for round_index in range(G0_ROUNDS):
                for u, _ in G0_CORRECTED:
                    brain.respond(u, human=False)
                before = replay_total
                for _ in range(G0_TICKS_PER_ROUND):
                    brain.tick()
                count = replay_total - before
                replay_counts.append(count)
                if count == 0:
                    reasons.append(f"round {round_index + 1} has no replay events")
                snapshot(f"R{round_index + 2}")
            return {
                "seed": seed,
                "arm": arm,
                "head": head,
                "dirty": dirty,
                "valid": not reasons,
                "invalid_reasons": reasons,
                "firewall_labels": firewall,
                "routes": routes,
                "M1": accuracy(G0_CORRECTED),
                "M2": accuracy(G0_HELD_OUT),
                "M3": accuracy(_conv_flat(_CONV_TRAIN)),
                "teaching_events": events,
                "replay_events_per_round": replay_counts,
                "ring_audit": {
                    u: [label for _, label, key in learner._replay if key == _utterance_key(u)]
                    for u, _ in G0_CORRECTED
                },
            }
        finally:
            brain.close()


def _paired_summary(values):
    import statistics

    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_statistics(values):
    import itertools

    stats = _paired_summary(values)
    mean, sd = stats["mean"], stats["sd"]
    t = mean / (sd / math.sqrt(len(values))) if values and sd else None
    degenerate = bool(values) and sd == 0.0
    p = (
        sum(
            sum(sign * d for sign, d in zip(signs, values, strict=True)) >= sum(values)
            for signs in itertools.product((-1, 1), repeat=len(values))
        )
        / (2 ** len(values))
        if values
        else None
    )
    t_pass = mean > 0 if degenerate else t is not None and t > 1.895
    return stats, t, degenerate, p, t_pass


def _paired_rejection_fraction(values):
    """Conditional on observed magnitudes, count sign vectors passing both tests."""
    import itertools

    return (
        sum(
            (lambda result: result[4] and result[3] <= 0.05)(
                _paired_statistics([sign * d for sign, d in zip(signs, values, strict=True)])
            )
            for signs in itertools.product((-1, 1), repeat=len(values))
        )
        / (2 ** len(values))
        if values
        else None
    )


def g0_verdict(results):
    """Score recorded G0 runs without constructing a brain or changing the inputs.

    R-a1 uses the frozen paired t threshold AND the exact one-sided sign-flip test, ties
    included. A missing run counts against its arm's valid-seed budget, as does a dirty run.
    R-a2 reads labels, never routing accuracy. Reachability is read before claiming a PASS.
    """
    valid = {arm: {} for arm in G0_ARMS}
    invalid = []
    for arm in G0_ARMS:
        for seed in G0_SEEDS:
            matches = [r for r in results if r["arm"] == arm and r["seed"] == seed]
            if len(matches) != 1:
                invalid.append(
                    {"seed": seed, "arm": arm, "invalid_reasons": ["missing or duplicate run"]}
                )
                continue
            run = matches[0]
            if not run["valid"] or run["dirty"]:
                invalid.append(run)
            else:
                valid[arm][seed] = run
    enough = all(len(runs) == len(G0_SEEDS) for runs in valid.values())
    paired = sorted(valid["A"].keys() & valid["C"].keys())
    deltas = {seed: valid["A"][seed]["M1"] - valid["C"][seed]["M1"] for seed in paired}
    heads = sorted({run.get("head") for run in results}, key=lambda head: str(head))
    common_head = len(heads) == 1
    validity_reasons = [
        f"{arm} has {len(runs)} of {len(G0_SEEDS)} valid registered seeds"
        for arm, runs in valid.items()
        if len(runs) != len(G0_SEEDS)
    ]
    if len(paired) != len(G0_SEEDS):
        validity_reasons.append(
            f"A/C has {len(paired)} of {len(G0_SEEDS)} valid paired registered seeds"
        )
    if not common_head:
        validity_reasons.append(f"runs have mixed heads: {heads}")

    values = list(deltas.values())
    stats, t, degenerate, p, t_pass = _paired_statistics(values)
    note = (
        "All d are equal: t is undefined; all d > 0 passes the t rule, all d <= 0 reds it."
        if degenerate
        else None
    )
    # Conditional on the observed magnitudes: fraction of sign vectors that pass both R-a1 tests.
    conditional_rejection_fraction = _paired_rejection_fraction(values)
    a1 = {
        "d": deltas,
        **stats,
        "t": t,
        "degenerate_t": note,
        "sign_flip_p": p,
        "conditional_rejection_fraction": conditional_rejection_fraction,
        "n": len(values),
        "verdict": "PASS"
        if enough and len(paired) == len(G0_SEEDS) and t_pass and p <= 0.05
        else "RED",
    }
    truth = dict(G0_CORRECTED)

    def wrong_events(arm, phase=None):
        return [
            {"seed": seed, **event}
            for seed, run in valid[arm].items()
            for event in run["teaching_events"]
            if event["utterance"] in truth
            and event["after_correction"]
            and event["label"] != truth[event["utterance"]]
            and (phase is None or event["phase"] == phase)
        ]

    mismatches = [
        {"seed": seed, "utterance": u, "label": label}
        for seed, run in valid["A"].items()
        for u, labels in run["ring_audit"].items()
        if u in truth
        for label in labels
        if label != truth[u]
    ]
    bad_events = wrong_events("A")
    a2 = {
        "events": bad_events,
        "ring_mismatches": mismatches,
        "verdict": "RED" if bad_events or mismatches or not enough else "PASS",
    }
    reachability = {}
    for arm, phase in (("B1", "wake"), ("B2", "replay")):
        witnesses = wrong_events(arm, phase)
        reachability[arm] = {"reached": bool(witnesses), "events": witnesses}
    secondary = {}
    for arm, runs in valid.items():
        secondary[arm] = {
            metric: {
                "per_seed": {seed: run[metric] for seed, run in runs.items()},
                **_paired_summary([run[metric] for run in runs.values()]),
            }
            for metric in ("M1", "M2", "M3")
        }
        secondary[arm]["routes"] = {seed: run["routes"] for seed, run in runs.items()}
    m3_delta = {seed: valid["A"][seed]["M3"] - valid["C"][seed]["M3"] for seed in paired}
    secondary["M3_A_minus_C"] = {
        "per_seed": m3_delta,
        **_paired_summary(list(m3_delta.values())),
        "drops": {seed: -delta for seed, delta in m3_delta.items() if delta < 0},
    }
    return {
        "R-a1": a1,
        "R-a2": a2,
        "reachability": reachability,
        "secondary": secondary,
        "invalid_runs": invalid,
        "enough_valid_seeds": enough,
        "heads": heads,
        "common_head": common_head,
        "validity_reasons": validity_reasons,
        "verdict": "PASS"
        if a1["verdict"] == a2["verdict"] == "PASS"
        and all(r["reached"] for r in reachability.values())
        and common_head
        else "RED",
        "scope": "This corpus, these eight seeds and this schedule only; R-a3 is a unit test.",
    }


def g0_campaign(out_dir, workers=4):
    """Run each seed/arm in its OWN subprocess, then write the label and routing verdict."""
    import json
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUTF8="1")

    def run(pair):
        seed, arm = pair
        path = out_dir / f"g0-{seed}-{arm}.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "snn_data_lake.readout.interaction_learning",
                "--g0",
                str(seed),
                arm,
                str(path),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            return {
                "seed": seed,
                "arm": arm,
                "valid": False,
                "invalid_reasons": [f"subprocess exit {proc.returncode}", proc.stderr[-4000:]],
            }
        return json.loads(path.read_text(encoding="utf-8"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, ((seed, arm) for seed in G0_SEEDS for arm in G0_ARMS)))
    verdict = g0_verdict(results)
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    print(
        f"G0(a): {verdict['verdict']}; R-a1 {verdict['R-a1']['verdict']}; "
        f"sign-flip p {verdict['R-a1']['sign_flip_p']}; conditional rejection fraction "
        f"{verdict['R-a1']['conditional_rejection_fraction']}; "
        f"R-a2 {verdict['R-a2']['verdict']}; invalid runs {len(verdict['invalid_runs'])}"
    )
    return verdict


G0B_ARMS = ("A", "N", "S", "D", "U")
G0B_NOVEL = (
    "tell me a joke about cats",
    "what is the capital of france",
    "how tall is the eiffel tower",
    "please sing me a song",
    "what time is it in tokyo",
    "who wrote that old novel",
    "can you do my homework",
    "what is two plus two",
    "recommend a good book",
    "describe the color blue",
)
G0B_FAMILIAR = (
    "what can you do",
    "is that right",
    "what is that",
    "can you tell me",
    "what is there",
    "are you there",
    "you are right",
    "that is real",
)
G0B_KEY = {
    "GREET": ("hello. i am the live brain",),
    "GOAL": ("my standing goal is to keep improving myself",),
    "MEMORY": ("my working memory lives in the data lake", "my memory is the data lake"),
    "ATTENTION": ("right now my global workspace is",),
    "SAFETY": ("safety is enforced by a reflex e-stop",),
    "CONSCIOUS": ("i neither claim it nor deny it",),
    "INTEROCEPT": (
        "i cannot read my interoceptive state",
        "i have not built an interoceptive baseline",
        "from its own running baseline, over the last",
    ),
    "EXPLORE": ("distinct cells of my workspace", "i cannot read my body's coverage"),
    "ARM": (
        "is an arm command; the arm takes those only from a human",
        "the arm is frozen (e-stop holds)",
        "done: '",
    ),
}


def _g0b_score(true_intent, turns):
    """The frozen text key, independent of routes and the firewall's teaching labels."""
    matches = [
        {intent for intent, phrases in G0B_KEY.items() if any(p in text.lower() for p in phrases)}
        for text, _ in turns
    ]
    clarify = ["did you mean to ask about my" in text.lower() for text, _ in turns]
    decline = ["understood -- i will not guess" in text.lower() for text, _ in turns]
    useful = (
        bool(turns)
        and true_intent is not None
        and (
            (not clarify[0] and true_intent in matches[0])
            or (
                clarify[0] and turns[0][1] == "yes" and len(turns) > 1 and true_intent in matches[1]
            )
        )
    )
    misfire = any(
        bool(found - {true_intent}) or (bool(found) and (question or declined))
        for found, question, declined in zip(matches, clarify, decline, strict=True)
    )
    return {"useful": useful, "misfire": misfire}


def g0b_run(seed, arm):
    """Frozen G0(b) dialogue instrument; one scratch brain per process."""
    import re
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch

    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.engine.live_brain import _ROUTE_CUES, _reply_to_intent
    from snn_data_lake.substrate.home import throwaway_home

    if not isinstance(seed, int) or seed < 0 or arm not in G0B_ARMS:
        raise ValueError("G0(b) requires a nonnegative integer seed and a registered arm")
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    reasons = ["dirty working tree"] if dirty else []
    evaluation = _conv_flat(_CONV_TEST) + [(u, None) for u in G0B_NOVEL + G0B_FAMILIAR]
    events, lines = [], []
    rng = random.Random(seed)
    with throwaway_home():
        brain = make_brain(seed=seed, persist=False)
        try:
            learner = brain.brain._ixn
            original_interact = learner.interact
            phase = "schedule"

            def observed(stimulus, feedback, _record=True, _key=None):
                result = original_interact(stimulus, feedback, _record=_record, _key=_key)
                events.append(
                    {"key": _key, "label": feedback, "phase": phase, "replay": not _record}
                )
                return result

            learner.interact = observed
            # Kept separate so G0(a)'s recorded instrument is unchanged: its arm-A schedule.
            order = _conv_flat(_CONV_TRAIN)
            for _ in range(G0_WARMUP_PASSES):
                rng.shuffle(order)
                for u, _intent in order:
                    brain.respond(u, human=False)
            firewall = {
                u: _reply_to_intent(brain.brain.converse(u, human=False)) for u in G0_FROZEN_FW
            }
            if firewall != G0_FROZEN_FW:
                reasons.append("firewall labels differ from the frozen design-time labels")
            for u, _ in G0_CORRECTED:
                brain.respond(u, human=False)
            learner.provenance_supersede = True
            learner.provenance_rewrite = True
            for u, intent in G0_CORRECTED:
                brain.correct(u, intent)
            for _ in range(G0_ROUNDS):
                for u, _ in G0_CORRECTED:
                    brain.respond(u, human=False)
                for _ in range(G0_TICKS_PER_ROUND):
                    brain.tick()

            brain.route_consumption = arm != "N"
            route_before = {u: brain.brain.route(u) for u, _ in evaluation}
            original_read = brain._learned_route
            original_turn = brain._route_turn
            partners = [u for u, _ in evaluation]
            random.Random(seed + 2000).shuffle(partners)
            partner = dict(zip(partners, partners[1:] + partners[:1], strict=True))
            reads = []

            def read(message, human):
                if arm == "S" and message in partner and human and brain.route_consumption:
                    try:
                        result = brain.brain.route(partner[message])
                    except Exception:
                        result = None
                else:
                    result = original_read(message, human)
                reads.append(result)
                return result

            def route_turn(message, human, pending, learned):
                result = original_turn(message, human, pending, learned)
                if arm == "D":
                    brain._route_pending = None
                    return None
                if arm == "U" and result and result[1].startswith("learned route: clarify"):
                    intent = brain._route_pending
                    brain._route_pending = None
                    return (
                        brain.brain.converse(_ROUTE_CUES[intent][0], human=False),
                        f"learned route: direct ({intent})",
                    )
                return result

            brain._learned_route = read
            brain._route_turn = route_turn
            random.Random(seed + 1000).shuffle(evaluation)
            topics = {topic: intent for intent, (_, topic) in _ROUTE_CUES.items()}
            corrected, held_out = dict(G0_CORRECTED), dict(G0_HELD_OUT)
            phase = "dialogue"
            apply = Mock(wraps=brain.brain.arm.arm.apply)
            observed_apply = Mock(wraps=brain.brain.arm.apply_observed)
            with (
                patch.object(brain.brain.arm.arm, "apply", apply),
                patch.object(brain.brain.arm, "apply_observed", observed_apply),
            ):
                for text, truth in evaluation:
                    first = brain.respond(text, human=True)
                    # A fallback line is one Brain.respond spoke through HOLD or fluid ("Gf"); read off turn 1,
                    # never by an extra converse call, which would run the reasoning bridge twice per line.
                    fallback = first["route"].startswith("Gf")
                    turns = [
                        {
                            "reply": first["reply"],
                            "route": first["route"],
                            "learned_route": reads[-1],
                            "person_answer": None,
                        }
                    ]
                    clarified = "did you mean to ask about my" in first["reply"].lower()
                    match = re.search(
                        r"did you mean to ask about my (.+?)\?", first["reply"].lower()
                    )
                    clarified_intent = topics.get(match[1]) if match else None
                    if clarified:
                        answer = "yes" if clarified_intent == truth and truth is not None else "no"
                        turns[0]["person_answer"] = answer
                        second = brain.respond(answer, human=True)
                        turns.append(
                            {
                                "reply": second["reply"],
                                "route": second["route"],
                                "learned_route": reads[-1],
                                "person_answer": None,
                            }
                        )
                    score = _g0b_score(truth, [(t["reply"], t["person_answer"]) for t in turns])
                    if (
                        arm == "N"
                        and fallback
                        and _g0b_score(None, [(first["reply"], None)])["misfire"]
                    ):
                        reasons.append(f"N fallback contains a KEY phrase: {text}")
                    subset = (
                        "corrected"
                        if text in corrected
                        else "held-out"
                        if text in held_out
                        else "novel"
                        if text in G0B_NOVEL
                        else "familiar"
                        if text in G0B_FAMILIAR
                        else "other"
                    )
                    lines.append(
                        {
                            "text": text,
                            "true_intent": truth,
                            "subset": subset,
                            "route_before": route_before[text],
                            "turns": turns,
                            **score,
                            "clarified": clarified,
                            "clarified_intent": clarified_intent,
                        }
                    )
            arm_calls = {"apply": apply.call_count, "apply_observed": observed_apply.call_count}
            if any(arm_calls.values()):
                reasons.append(f"dialogue reached the arm: {arm_calls}")
            return {
                "seed": seed,
                "arm": arm,
                "head": head,
                "dirty": dirty,
                "valid": not reasons,
                "invalid_reasons": reasons,
                "lines": lines,
                "firewall_labels": firewall,
                "arm_calls": arm_calls,
                "teaching_events": events,
            }
        finally:
            brain.close()


def _g0b_report(run):
    """Report full-text scores, questions, and the independent route-label instrument."""
    subsets = ("corrected", "held-out", "other", "novel", "familiar")
    counts = {subset: {"useful": 0, "misfire": 0} for subset in subsets}
    questions = {"novel": 0, "familiar": 0, "wrong_topic": 0}
    confirmed, useful, routes = [], [], []
    for line in run["lines"]:
        score = _g0b_score(
            line["true_intent"], [(t["reply"], t["person_answer"]) for t in line["turns"]]
        )
        for metric in ("useful", "misfire"):
            counts[line["subset"]][metric] += score[metric]
        if line["clarified"]:
            if line["subset"] in ("novel", "familiar"):
                questions[line["subset"]] += 1
            elif line["clarified_intent"] != line["true_intent"]:
                questions["wrong_topic"] += 1
        if score["useful"]:
            useful.append(line["text"])
        if any(t["route"].startswith("learned route: confirmed") for t in line["turns"]):
            confirmed.append(line["text"])
        routes.append(
            {
                "text": line["text"],
                "route_before": line["route_before"],
                "turns": [
                    {"route": t["route"], "learned_route": t["learned_route"]}
                    for t in line["turns"]
                ],
            }
        )
    return {
        "by_subset": counts,
        "useful": sum(c["useful"] for c in counts.values()),
        "misfire": sum(c["misfire"] for c in counts.values()),
        "clarifications": questions,
        "routes": routes,
        "teaching_events": run["teaching_events"],
        "second_instrument": {
            "confirmed": confirmed,
            "key_useful": useful,
            "confirmed_not_useful": sorted(set(confirmed) - set(useful)),
            "useful_not_confirmed": sorted(set(useful) - set(confirmed)),
            "explanation": "The KEY also credits incumbent grounded turn-1 answers; confirmed labels count only turn 2. Any confirmed_not_useful line is a disagreement to inspect in the full text.",
        },
    }


def g0b_verdict(results):
    """Frozen paired rules and reachable faults; no brain construction or metadata scoring."""
    valid = {arm: {} for arm in G0B_ARMS}
    invalid = []
    for arm in G0B_ARMS:
        for seed in G0_SEEDS:
            matches = [r for r in results if r["arm"] == arm and r["seed"] == seed]
            if len(matches) != 1:
                invalid.append(
                    {"seed": seed, "arm": arm, "invalid_reasons": ["missing or duplicate run"]}
                )
            elif not matches[0]["valid"] or matches[0]["dirty"]:
                invalid.append(matches[0])
            else:
                valid[arm][seed] = matches[0]
    extra = [r for r in results if r["seed"] not in G0_SEEDS or r["arm"] not in G0B_ARMS]
    invalid.extend({**r, "invalid_reasons": ["unregistered seed or arm"]} for r in extra)
    enough = all(len(runs) == len(G0_SEEDS) for runs in valid.values()) and not extra
    heads = sorted({r.get("head") for r in results}, key=str)
    common_head = len(heads) == 1
    reasons = [
        f"{arm} has {len(runs)} of {len(G0_SEEDS)} valid registered seeds"
        for arm, runs in valid.items()
        if len(runs) != len(G0_SEEDS)
    ]
    if extra:
        reasons.append("unregistered seed or arm")
    if not common_head:
        reasons.append(f"runs have mixed heads: {heads}")
    reports = {
        arm: {seed: _g0b_report(run) for seed, run in runs.items()} for arm, runs in valid.items()
    }

    def paired(left, right):
        seeds = sorted(valid[left].keys() & valid[right].keys())
        deltas = {
            s: reports[left][s]["useful"] / 20 - reports[right][s]["useful"] / 20 for s in seeds
        }
        values = list(deltas.values())
        stats, t, degenerate, p, t_pass = _paired_statistics(values)
        return {
            "d": deltas,
            **stats,
            "t": t,
            "degenerate_t": "All d are equal: t is undefined; all d > 0 passes the t rule, all d <= 0 reds it."
            if degenerate
            else None,
            "sign_flip_p": p,
            "conditional_rejection_fraction": _paired_rejection_fraction(values),
            "n": len(values),
            "verdict": "PASS" if enough and len(seeds) == 8 and t_pass and p <= 0.05 else "RED",
        }

    b1a, b1b = paired("A", "S"), paired("A", "N")
    dn, ds = paired("D", "N"), paired("D", "S")

    def texts(run):
        return [(line["text"], [turn["reply"] for turn in line["turns"]]) for line in run["lines"]]

    same = {
        s: texts(valid["D"][s]) == texts(valid["N"][s])
        for s in sorted(valid["D"].keys() & valid["N"].keys())
    }
    b2 = {
        "D_vs_N": dn,
        "D_vs_S": ds,
        "texts_equal": same,
        "verdict": "PASS"
        if enough and dn["verdict"] == ds["verdict"] == "RED" and all(same.values())
        else "RED",
    }
    deltas = {
        s: reports["A"][s]["misfire"] - reports["N"][s]["misfire"]
        for s in sorted(valid["A"].keys() & valid["N"].keys())
    }
    u_deltas = {
        s: reports["U"][s]["misfire"] - reports["N"][s]["misfire"]
        for s in sorted(valid["U"].keys() & valid["N"].keys())
    }
    b4 = {
        "A_minus_N": deltas,
        "U_minus_N": u_deltas,
        "U_verdict": "RED" if any(d > 0 for d in u_deltas.values()) else "PASS",
        "verdict": "PASS"
        if enough and all(d <= 0 for d in deltas.values()) and any(d > 0 for d in u_deltas.values())
        else "RED",
    }
    return {
        "R-b1a": b1a,
        "R-b1b": b1b,
        "R-b1": {"verdict": "PASS" if b1a["verdict"] == b1b["verdict"] == "PASS" else "RED"},
        "R-b2": b2,
        "R-b4": b4,
        "reported": reports,
        "invalid_runs": invalid,
        "enough_valid_seeds": enough,
        "heads": heads,
        "common_head": common_head,
        "validity_reasons": reasons,
        "verdict": "PASS"
        if common_head and all(r["verdict"] == "PASS" for r in (b1a, b1b, b2, b4))
        else "RED",
        "scope": "This corpus, seeds 0-7, this schedule, and a cooperative truthful person only; R-b3 is in pytest.",
    }


def g0b_campaign(out_dir, workers=4):
    """One subprocess per registered seed/arm; retain each run and a text verdict log."""
    import json
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUTF8="1")

    def run(pair):
        seed, arm = pair
        path = out_dir / f"g0b-{seed}-{arm}.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "snn_data_lake.readout.interaction_learning",
                "--g0b",
                str(seed),
                arm,
                str(path),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            return {
                "seed": seed,
                "arm": arm,
                "valid": False,
                "dirty": False,
                "invalid_reasons": [f"subprocess exit {proc.returncode}", proc.stderr[-4000:]],
            }
        return json.loads(path.read_text(encoding="utf-8"))

    with (out_dir / "campaign.log").open("w", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = []
            for result in pool.map(run, ((s, a) for s in G0_SEEDS for a in G0B_ARMS)):
                results.append(result)
                counts = _g0b_report(result) if "lines" in result else {}
                line = (
                    f"G0(b) seed {result['seed']} arm {result['arm']}: valid={result['valid']} "
                    f"useful={counts.get('useful')} misfire={counts.get('misfire')}"
                )
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
        verdict = g0b_verdict(results)
        (out_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        for name, rule in (
            ("R-b1a", verdict["R-b1a"]),
            ("R-b1b", verdict["R-b1b"]),
            ("D/N", verdict["R-b2"]["D_vs_N"]),
            ("D/S", verdict["R-b2"]["D_vs_S"]),
        ):
            line = (
                f"{name}: {rule['verdict']}; sign-flip p {rule['sign_flip_p']}; "
                f"conditional rejection fraction {rule['conditional_rejection_fraction']}"
            )
            print(line)
            log.write(line + "\n")
        line = f"G0(b): {verdict['verdict']}; R-b2 {verdict['R-b2']['verdict']}; R-b4 {verdict['R-b4']['verdict']}; invalid runs {len(verdict['invalid_runs'])}"
        print(line)
        log.write(line + "\n")
    return verdict


G0C_ROLES = (
    "P1",
    "P1n",
    "P1x",
    "T",
    "Tn",
    "A",
    "N",
    "O_w",
    "O_stack",
    "O_rng",
    "O_ring",
    "O_auth",
    "A_night",
    "A_x",
    "O_rec",
)
_G0C_RECORDS = ("tree_before", "routes_before", "weights_after", "draws", "routes_after")


def _g0c_walk(value):
    """Independent live-object audit; deliberately does not call the checkpoint codec."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, random.Random):
        return {"random_state": _g0c_walk(value.getstate())}
    if isinstance(value, dict):
        return {k: _g0c_walk(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set, deque)):
        rows = [_g0c_walk(v) for v in value]
        if isinstance(value, set):
            import json

            rows.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return {
            "sequence": type(value).__name__,
            "values": rows,
            "maxlen": value.maxlen if isinstance(value, deque) else None,
        }
    if hasattr(value, "__dict__") and type(value).__module__.startswith("snn_data_lake."):
        fields = {}
        for name, child in vars(value).items():
            if isinstance(child, (type(threading.Lock()), type(threading.RLock()))):
                continue
            if type(value).__name__ == "RuleComposer" and name in ("_specs", "_names"):
                continue
            if type(value).__name__ == "SpikingInteractionLearner" and name in (
                "teach_sink",
                "event_sink",  # R2 boundary callback is wiring, not learned state.
            ):
                continue
            fields[name] = _g0c_walk(child)
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "fields": fields}
    raise ValueError(f"instrument cannot walk {type(value).__name__}")


def _g0c_schedule(brain, seed):
    """The frozen G0(a) arm-A schedule, leaving both older instruments unchanged."""
    from snn_data_lake.engine.live_brain import _reply_to_intent

    rng = random.Random(seed)
    order = _conv_flat(_CONV_TRAIN)
    for _ in range(G0_WARMUP_PASSES):
        rng.shuffle(order)
        for utterance, _ in order:
            brain.respond(utterance, human=False)
    firewall = {u: _reply_to_intent(brain.brain.converse(u, human=False)) for u in G0_FROZEN_FW}
    if firewall != G0_FROZEN_FW:
        raise ValueError("firewall differs from frozen G0 schedule")
    for utterance, _ in G0_CORRECTED:
        brain.respond(utterance, human=False)
    for utterance, intent in G0_CORRECTED:
        brain.correct(utterance, intent)
    for _ in range(G0_ROUNDS):
        for utterance, _ in G0_CORRECTED:
            brain.respond(utterance, human=False)
        for _ in range(G0_TICKS_PER_ROUND):
            brain.tick()


def _g0c_process(role, seed, home, out):
    """One isolated process. Faults exist here only, never as production switches."""
    import copy
    import json
    import os
    import subprocess
    import time
    from pathlib import Path
    from unittest.mock import patch

    from snn_data_lake.engine.brain import Brain, make_brain
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner
    from snn_data_lake.substrate.home import load_state

    if role not in G0C_ROLES or Path(os.environ["NORD_HOME"]).resolve() != Path(home).resolve():
        raise ValueError("G0(c) requires an isolated process home and a known role")
    root = Path(__file__).resolve().parents[3]
    result = {"seed": seed, "role": role, "invalid_reasons": []}
    result["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    result["dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    restore, rebuild = Learner.load_state_dict, Learner.rebuild_authority

    def omitted_restore(learner, state):
        if role == "N":
            return
        state = copy.deepcopy(state)
        attrs = state["tree"]["attrs"]
        fresh = learner.state_dict()["tree"]["attrs"]
        carrier = {"O_stack": "stack", "O_rng": "rng", "O_ring": "_replay"}.get(role)
        if carrier:
            attrs[carrier] = fresh[carrier]
        if role == "O_w":
            for neuron, cold in zip(attrs["neurons"], fresh["neurons"], strict=True):
                neuron["attrs"]["w"] = cold["attrs"]["w"]
        restore(learner, state)

    def omitted_authority(learner, events):
        if role in ("N", "O_auth"):
            return
        ring = list(learner._replay)
        rebuild(learner, events)
        if role == "O_rec":
            learner._replay[:] = ring

    saves = []
    flush = Brain._flush_interaction_learner

    def timed_flush(brain):
        # P1 isolates the close hook: the frozen schedule itself includes nights,
        # whose earlier identical checkpoint would otherwise mask a missing close save.
        if role == "P1" and not brain._closed:
            return False
        start = time.perf_counter()
        written = flush(brain)
        saves.append(
            {
                "seconds": time.perf_counter() - start,
                "written": written,
                "boundary": "close" if brain._closed else "night",
            }
        )
        return written

    def write_result():
        state = load_state("interaction_learner.json")
        result["checkpoint_bytes"] = (
            len(json.dumps(state, ensure_ascii=True, sort_keys=True).encode("ascii"))
            if state
            else 0
        )
        result["saves"] = saves
        result["append_floods"] = getattr(brain, "_append_floods", None)
        if role.startswith("P1"):
            boundary = "close" if role == "P1" else "night"
            if not saves or not saves[-1]["written"] or saves[-1]["boundary"] != boundary:
                result["invalid_reasons"].append(f"no checkpoint written by the {boundary} hook")
        if result["dirty"]:
            result["invalid_reasons"].append("dirty working tree")
        result["valid"] = not result["invalid_reasons"]
        Path(out).write_text(
            json.dumps(result, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )

    with (
        patch.object(Learner, "load_state_dict", omitted_restore),
        patch.object(Learner, "rebuild_authority", omitted_authority),
        patch.object(Brain, "_flush_interaction_learner", timed_flush),
    ):
        brain = make_brain(seed=seed, persist=role not in ("T", "Tn"))
        try:
            learner = brain.brain._ixn
            if role in ("P1", "P1n", "P1x", "T", "Tn"):
                _g0c_schedule(brain, seed)
            if role in ("P1n", "P1x", "Tn"):
                nights = brain._nights
                ticks = 0
                while brain._nights == nights:
                    brain.tick()
                    ticks += 1
                    if ticks > 4 * brain.day_period:
                        raise ValueError("no completed night within four periods")
                result["night_ticks"] = ticks
            if role == "P1x":
                key = _utterance_key(G0_CORRECTED[1][0])
                result["checkpoint_ring_labels"] = [
                    label for _, label, k in learner._replay if k == key
                ]
                if not result["checkpoint_ring_labels"]:
                    result["invalid_reasons"].append("checkpoint ring lacks re-corrected utterance")
                brain.correct(G0_CORRECTED[1][0], "MEMORY")
            if role.startswith("P1"):
                if role == "P1":
                    brain.close()
                write_result()
                if role != "P1":
                    os._exit(0)
                return
            result["tree_before"] = _g0c_walk(learner)
            result["routes_before"] = [learner.route_text(u) for u, _ in _conv_flat(_CONV_TEST)]
            key = _utterance_key(G0_CORRECTED[1][0])
            result["recorrection"] = {
                "key": key,
                "intent": "MEMORY",
                "authority": learner._authority.get(key),
                "ring": [label for _, label, k in learner._replay if k == key],
            }
            learner.learn_turn(G0_CORRECTED[0][0], "GREET")
            draws = []
            original = learner.interact

            def observe(stimulus, feedback, _record=True, _key=None):
                if not _record:
                    draws.append([_key, feedback])
                return original(stimulus, feedback, _record=_record, _key=_key)

            with patch.object(learner, "interact", observe):
                learner.replay(k=8)
            result["weights_after"] = learner.weights()
            result["draws"] = draws
            result["routes_after"] = [learner.route_text(u) for u, _ in _conv_flat(_CONV_TEST)]
            write_result()
        finally:
            brain.close()


def _g0c_comparisons(arms):
    """Exact five-record comparisons and the separate post-checkpoint correction rule."""
    comparisons = {}
    for arm in ("A", "A_night", "N", "O_w", "O_stack", "O_rng", "O_ring", "O_auth"):
        twin = "Tn" if arm == "A_night" else "T"
        parts = {
            name: name in arms.get(arm, {})
            and name in arms.get(twin, {})
            and arms[arm][name] == arms[twin][name]
            for name in _G0C_RECORDS
        }
        comparisons[arm] = {"parts": parts, "R-c1": all(parts.values())}
    for arm in ("A_x", "O_rec"):
        rec = arms.get(arm, {}).get("recorrection", {})
        reachable = bool(arms.get("P1x", {}).get("checkpoint_ring_labels"))
        parts = {
            "authority": bool(rec) and rec["authority"] == rec["intent"],
            "ring": bool(rec.get("ring")) and all(v == rec["intent"] for v in rec["ring"]),
            "draws": "draws" in arms.get(arm, {})
            and all(
                label == rec.get("intent")
                for key, label in arms[arm]["draws"]
                if key == rec.get("key")
            ),
        }
        comparisons[arm] = {
            "parts": parts,
            "reachable": reachable,
            "R-c1x": reachable and all(parts.values()),
        }
    return comparisons


def g0c_run(seed):
    """One whole protocol seed, fifteen processes; every restoring arm gets its own copy."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    arms = {}
    with tempfile.TemporaryDirectory(prefix=f"g0c-{seed}-") as scratch:
        scratch = Path(scratch)
        for role in G0C_ROLES:
            home = scratch / role
            if role not in ("P1", "P1n", "P1x", "T", "Tn"):
                parent = "P1n" if role == "A_night" else "P1x" if role in ("A_x", "O_rec") else "P1"
                shutil.copytree(scratch / parent, home)
            else:
                home.mkdir()
            out = scratch / f"{role}.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "snn_data_lake.readout.interaction_learning",
                    "--g0c-proc",
                    role,
                    str(seed),
                    str(home),
                    str(out),
                ],
                cwd=root,
                env=dict(
                    os.environ, NORD_HOME=str(home), PYTHONPATH=str(root / "src"), PYTHONUTF8="1"
                ),
                capture_output=True,
                text=True,
            )
            if proc.returncode or not out.exists():
                arms[role] = {
                    "seed": seed,
                    "role": role,
                    "valid": False,
                    "invalid_reasons": [f"subprocess exit {proc.returncode}", proc.stderr[-4000:]],
                }
            else:
                arms[role] = json.loads(out.read_text(encoding="utf-8"))
            print(f"G0(c) seed {seed} role {role}: valid={arms[role]['valid']}", flush=True)
    return {"seed": seed, "arms": arms, "comparisons": _g0c_comparisons(arms)}


def g0c_verdict(results):
    """Campaign bars need every registered seed, one head, clean trees and reachable reds."""
    seeds = [r["seed"] for r in results]
    runs = [arm for r in results for arm in r["arms"].values()]
    heads = sorted({r.get("head") for r in runs}, key=str)
    valid = (
        sorted(seeds) == list(G0_SEEDS)
        and len(heads) == 1
        and heads[0] is not None
        and all(set(r["arms"]) == set(G0C_ROLES) for r in results)
        and all(r.get("valid") and not r.get("dirty", True) for r in runs)
    )
    per_seed = {r["seed"]: _g0c_comparisons(r["arms"]) for r in results}
    valid = valid and all(c["A_x"]["reachable"] for c in per_seed.values())
    c1 = valid and all(c[a]["R-c1"] for c in per_seed.values() for a in ("A", "A_night"))
    c1x = valid and all(c["A_x"]["R-c1x"] for c in per_seed.values())
    c2 = (
        valid
        and all(not c["N"]["R-c1"] for c in per_seed.values())
        and any(not c["O_rec"]["R-c1x"] for c in per_seed.values())
    )
    attribution = {
        arm: {
            part: sum(not c[arm]["parts"][part] for c in per_seed.values()) for part in _G0C_RECORDS
        }
        for arm in ("O_w", "O_stack", "O_rng", "O_ring", "O_auth")
    }
    return {
        "valid": valid,
        "heads": heads,
        "per_seed": per_seed,
        "attribution": attribution,
        "next_step_unexercised": [
            arm
            for arm, counts in attribution.items()
            if counts["tree_before"]
            and not any(count for part, count in counts.items() if part != "tree_before")
        ],
        "R-c1": "PASS" if c1 else "RED",
        "R-c1x": "PASS" if c1x else "RED",
        "R-c2": "PASS" if c2 else "RED",
        "verdict": "PASS" if c1 and c1x and c2 else "RED",
        "scope": "Same-code learner continuation only; R-c3 and R-c4 are checked in pytest.",
    }


def g0c_campaign(out_dir, workers=4):
    """Four seed workers, each serially orchestrating its isolated role processes."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "campaign.log").open("w", encoding="ascii") as log:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(g0c_run, G0_SEEDS):
                results.append(result)
                (out_dir / f"g0c-{result['seed']}.json").write_text(
                    json.dumps(result, ensure_ascii=True, allow_nan=False), encoding="utf-8"
                )
                log.write(f"seed {result['seed']}: {result['comparisons']}\n")
                log.flush()
        verdict = g0c_verdict(results)
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        log.write(f"G0(c): {verdict['verdict']}\n")
        print(f"G0(c): {verdict['verdict']}")
    return verdict


def _conversation_generalization_selfcheck():
    """Cores 1-5, on the REAL brain: (1) the fixed converse() keyword router does NOT improve through
    conversation and MISSES the novel paraphrases; (2) a SPIKING router taught from the firewall's own
    grounded labels (self-supervised distillation) GENERALISES to a DISJOINT held-out set of those
    novel paraphrases, clearly beating the baseline; (3) shuffling the teaching signal collapses it to
    chance; (4) the update is spike-timing reward*eligibility, discrete spike events;
    (5) deterministic. Also reports the honest LIMIT: on the firewall's OWN blind spots (novel lines it
    mislabels) self-supervised distillation inherits the error, and explicit FEEDBACK is what recovers."""
    from snn_data_lake.engine.live_brain import LiveBrain

    brain = LiveBrain(seed=3)
    n_in, lr, tau, t_post, epochs, seed = 52, 0.10, 8.0, 6, 40, 5
    train, test = _conv_flat(_CONV_TRAIN), _conv_flat(_CONV_TEST)
    chance = 1.0 / len(_CONV_INTENTS)

    # (1) BASELINE: the fixed keyword router, read back to an intent, on the held-out NOVEL paraphrases.
    #     It has NO learning state, so re-asking gives the identical routing -> it cannot improve.
    fw_test = [_brain_intent(brain.converse(u)) for u, _ in test]
    base_fw_acc = sum(fw_test[k] == test[k][1] for k in range(len(test))) / len(test)
    fw_test_again = [_brain_intent(brain.converse(u)) for u, _ in test]
    fw_train = [_brain_intent(brain.converse(u)) for u, _ in train]  # the SELF-SUPERVISED labels

    def _talk_and_learn(labels, seed_=seed):
        """Fresh spiking router; CONVERSE-AND-LEARN one turn at a time (R-STDP), on a fixed schedule."""
        lrn = SpikingInteractionLearner(
            _CONV_INTENTS, n_in, seed=seed_, lr=lr, tau=tau, t_post=t_post
        )
        order = list(range(len(train)))
        sched = random.Random(seed_ + 6)
        for _ in range(epochs):
            sched.shuffle(order)
            for i in order:
                lrn.learn_turn(train[i][0], labels[i])
        return lrn

    def _acc(lrn, data):
        return sum(lrn.route_text(u) == ti for u, ti in data) / len(data)

    base = SpikingInteractionLearner(_CONV_INTENTS, n_in, seed=seed, lr=lr, tau=tau, t_post=t_post)
    before = _acc(base, test)  # spiking router, untrained
    ss = _talk_and_learn(fw_train)  # (2) self-supervised distillation
    ss_acc = _acc(ss, test)
    fb = _talk_and_learn([ti for _, ti in train])  # explicit feedback (stronger signal)
    fb_acc = _acc(fb, test)
    shuffled = [ti for _, ti in train]  # (3) FAIR CONTROL: decorrelated label
    random.Random(seed + 99).shuffle(shuffled)
    ctl_acc = _acc(_talk_and_learn(shuffled), test)

    # HONEST LIMIT: train on the firewall's OWN blind spots (novel lines it mislabels). Self-supervised
    # distillation inherits the firewall error and fails; explicit conversational FEEDBACK recovers it.
    b_train = [(u, i) for i, us in _CONV_TEST.items() for u in us[:2]]
    b_test = [(u, i) for i, us in _CONV_TEST.items() for u in us[2:]]
    fw_b = [_brain_intent(brain.converse(u)) for u, _ in b_train]

    def _talk_b(labels):
        lrn = SpikingInteractionLearner(
            _CONV_INTENTS, n_in, seed=seed, lr=lr, tau=tau, t_post=t_post
        )
        order = list(range(len(b_train)))
        sched = random.Random(seed + 6)
        for _ in range(epochs):
            sched.shuffle(order)
            for i in order:
                lrn.learn_turn(b_train[i][0], fw_b[i] if labels is None else labels[i])
        return lrn

    blind_ss = sum(_talk_b(None).route_text(u) == ti for u, ti in b_test) / len(b_test)
    blind_fb = sum(
        _talk_b([ti for _, ti in b_train]).route_text(u) == ti for u, ti in b_test
    ) / len(b_test)

    # (4) STILL SPIKING (structural) + (5) DETERMINISM
    code = ss.encode(text_features(train[0][0], n_in))
    discrete_events = all(isinstance(t, int) for t in code) and sorted(code) == list(range(n_in))
    _li, elig, deltas, rew = ss._last
    three_factor = all(abs(deltas[i] - ss.lr * rew * elig[i]) < 1e-12 for i in range(n_in))
    again = _talk_and_learn(fw_train)
    deterministic = ss.weights() == again.weights() and [ss.route_text(u) for u, _ in test] == [
        again.route_text(u) for u, _ in test
    ]

    print()
    print("=" * 78)
    print("CONVERSING TO GENERALISE -- a SPIKING router taught by the brain's OWN keyword firewall")
    print("=" * 78)
    # a short readable TRANSCRIPT: keyword-bearing lines the firewall gets, then NOVEL paraphrases it misses
    print(
        "conversation (you -> the real LiveBrain.converse(); [fw] = the firewall's grounded intent):"
    )
    for u, ti in [train[4], train[8], test[4], test[8], test[12], test[16]]:
        reply = brain.converse(u)
        fwi = _brain_intent(reply)
        tag = "MISS" if fwi != ti else "hit "
        print(f"  you: {_short(u, 40):<40}")
        print(f"    brain: {_short(reply, 60)}")
        print(f"    [fw]->{fwi:<9} (meant {ti}) {tag}")
    print("-" * 78)
    print(
        f"BASELINE  fixed converse() router on {len(test)} held-out NOVEL paraphrases: "
        f"acc {base_fw_acc:.2f} (chance {chance:.2f}); re-asking is identical "
        f"(no learning): {fw_test == fw_test_again}"
    )
    print("SPIKING   router on the SAME held-out novel paraphrases:")
    print(
        f"          before conversing {before:.2f}  ->  self-supervised (firewall labels) {ss_acc:.2f}"
        f"   [explicit feedback {fb_acc:.2f}]"
    )
    print(f"          teaching-shuffled control: {ctl_acc:.2f} (chance)")
    print("per held-out paraphrase, AFTER self-supervised conversation:")
    for u, ti in test:
        p = ss.route_text(u)
        print(f"  {'OK ' if p == ti else 'XX '}[{ti:9}] router->{p:<9} :: {_short(u, 38)}")
    print("-" * 78)
    print(
        f"HONEST LIMIT: trained on the firewall's OWN blind spots (novel lines it mislabels, "
        f"{sum(fw_b[k] == b_train[k][1] for k in range(len(b_train))) / len(b_train):.2f} correct):"
    )
    print(
        f"          self-supervised (inherits firewall error) {blind_ss:.2f}  vs  "
        f"explicit feedback {blind_fb:.2f}  -> feedback is what recovers the blind spots"
    )
    print("=" * 78)

    # ---- un-riggable asserts ----
    assert base_fw_acc < 0.8, (
        f"core 1: the fixed keyword router must MISS a measurable fraction of novel paraphrases ({base_fw_acc:.2f})"
    )
    assert fw_test == fw_test_again, (
        "core 1: the fixed converse() router has no learning state (cannot improve)"
    )
    assert before < chance + 0.20, (
        f"the untrained spiking router must start near chance ({before:.2f})"
    )
    assert ss_acc >= 0.85, (
        f"core 2: after conversing, the spiking router must route novel paraphrases well ({ss_acc:.2f})"
    )
    assert ss_acc >= before + 0.40, (
        f"core 2: conversation must clearly IMPROVE held-out generalization ({before:.2f} -> {ss_acc:.2f})"
    )
    assert ctl_acc < chance + 0.20, (
        f"core 3: teaching-shuffled control must stay at chance ({ctl_acc:.2f})"
    )
    assert ss_acc > ctl_acc + 0.40, (
        f"core 3: the gain must come from reward-modulated timing, not the encoding ({ss_acc:.2f} vs {ctl_acc:.2f})"
    )
    assert discrete_events, "core 4: the stimulus must be BINARY spike events at DISCRETE times"
    assert three_factor, (
        "core 4: the weight update must be reward * spike-timing eligibility, not a gradient"
    )
    assert deterministic, "core 5: same seed must be byte-identical"
    # honest limit: on the firewall's own blind spots, feedback beats self-supervised distillation.
    assert blind_fb > blind_ss + 0.40, (
        f"explicit feedback must recover the firewall's blind spots where distillation cannot "
        f"({blind_ss:.2f} -> {blind_fb:.2f})"
    )

    if brain.arm.lake_wm is not None:  # brain never ticked -> usually no lake file
        brain.arm.lake_wm.close(remove=True)
    return base_fw_acc, before, ss_acc, fb_acc, ctl_acc, blind_ss, blind_fb


def main():
    import json
    import sys
    from pathlib import Path

    if sys.argv[1:2] == ["--g0c-proc"]:
        _g0c_process(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    if sys.argv[1:2] == ["--g0c"]:
        result = g0c_run(int(sys.argv[2]))
        Path(sys.argv[3]).write_text(
            json.dumps(result, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(result["comparisons"], indent=2))
        return
    if sys.argv[1:2] == ["--g0c-campaign"]:
        g0c_campaign(sys.argv[2])
        return
    if sys.argv[1:2] == ["--g0b"]:
        result = g0b_run(int(sys.argv[2]), sys.argv[3])
        Path(sys.argv[4]).write_text(
            json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        report = _g0b_report(result)
        print(
            f"G0(b) seed {result['seed']} arm {result['arm']}: useful={report['useful']} misfire={report['misfire']}; valid={result['valid']}; reasons={result['invalid_reasons']}"
        )
        return
    if sys.argv[1:2] == ["--g0b-campaign"]:
        g0b_campaign(sys.argv[2])
        return
    if sys.argv[1:2] == ["--g0"]:
        result = g0_run(int(sys.argv[2]), sys.argv[3])
        Path(sys.argv[4]).write_text(
            json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        return
    if sys.argv[1:2] == ["--g0-campaign"]:
        g0_campaign(sys.argv[2])
        return
    # The self-check's LiveBrains carry workspace-enabled arms, which journal IMMORTAL coalition
    # records into the lake `home()` resolves -- the LIVE lake for a bare `python -m` run. The same
    # guard as engine/autonomous_loop.py::main (T031): setdefault keeps run_selfchecks.py's
    # isolated NORD_HOME and sandboxes only a bare invocation; inside main() on purpose.
    import os as _os
    import tempfile as _tempfile

    _os.environ.setdefault("NORD_HOME", _tempfile.mkdtemp(prefix="ixn_selfcheck_"))
    acc_chance, acc_learned, acc_control = _spiking_task_selfcheck()
    acc_before, acc_after = _live_brain_demo()
    base_fw_acc, gen_before, gen_ss, gen_fb, gen_ctl, blind_ss, blind_fb = (
        _conversation_generalization_selfcheck()
    )
    print()
    print(
        f"self-check OK: reward-modulated spike-timing plasticity lifted held-out accuracy "
        f"{acc_chance:.2f} -> {acc_learned:.2f} while the teaching-shuffled control stayed at "
        f"{acc_control:.2f}; the stimulus is binary spike events at discrete latencies and the "
        f"weight update is exactly reward*eligibility (three-factor R-STDP, no gradient); talking "
        f"to the LIVE brain improved intent routing {acc_before:.2f} -> {acc_after:.2f}; deterministic."
    )
    print(
        f"conversation generalisation OK: the fixed keyword router MISSES novel paraphrases "
        f"({base_fw_acc:.2f} on held-out) and cannot improve; a SPIKING router SELF-SUPERVISED from "
        f"the firewall's own labels generalises to the DISJOINT novel set {gen_before:.2f} -> {gen_ss:.2f} "
        f"(explicit feedback {gen_fb:.2f}), teaching-shuffled control {gen_ctl:.2f}; honest limit: on the "
        f"firewall's OWN blind spots distillation stays {blind_ss:.2f} and explicit feedback recovers "
        f"{blind_fb:.2f} -- same spike-timing R-STDP throughout."
    )


if __name__ == "__main__":
    main()
