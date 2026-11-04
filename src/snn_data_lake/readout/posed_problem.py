"""G9 Brain-owned Part D reader and typed continuation.

Composes Part D's real driver/reader/producer. Existing pd_cell/read_d1 remain
the deciding rule/control path; this owner wrapper cannot choose a winning rule.
"""

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

GRADIENT_CLAIM = {
    "PosedActor._effect": (
        "NOT_A_RULE",
        "Typed source orchestration calls the canonical learner and return port under masks.",
    ),
    "chat_grown_deposit": (
        "LOCAL",
        "Calls canonical accumulate_causal and reward_deltas on each grown output with its own pre/post trace and scalar reward, preserving Dale sign.",
    ),
}


class PosedRefused(ValueError):
    pass


def _need(ok, reason):
    if not ok:
        raise PosedRefused(reason)


ENGINE_SELECTION_SCOPE = "g9.grow-retain-restore.v1"


def pack_engine_binding(config):
    """Lossless frozen source in a compact carrier, avoiding repeated mutable schedule trees."""
    import base64
    import zlib

    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema": "g9.engine.binding.zlib.v1",
        "cycle": copy.deepcopy(config["cycle"]),
        "raw_size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }


def unpack_engine_binding(carrier):
    """Decode exact canonical source; no schedule is regenerated from a seed on restore."""
    import base64
    import zlib

    _need(type(carrier) is dict, "complete engine binding carrier")
    if "schema" not in carrier:
        return copy.deepcopy(carrier)
    _need(
        set(carrier) == {"schema", "cycle", "raw_size", "sha256", "payload"}
        and carrier["schema"] == "g9.engine.binding.zlib.v1"
        and type(carrier["raw_size"]) is int
        and 0 < carrier["raw_size"] <= 16 * 1024 * 1024
        and type(carrier["sha256"]) is str
        and type(carrier["payload"]) is str,
        "G9_ENGINE_BINDING_CODEC: complete bounded source carrier",
    )
    try:
        compressed = base64.b64decode(carrier["payload"], validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, carrier["raw_size"] + 1)
        _need(
            decoder.eof
            and not decoder.unused_data
            and not decoder.unconsumed_tail
            and len(raw) == carrier["raw_size"]
            and hashlib.sha256(raw).hexdigest() == carrier["sha256"],
            "G9_ENGINE_BINDING_CODEC: corrupt frozen source",
        )
        config = json.loads(raw)
        _need(
            type(config) is dict and set(config) == {"schedule", "projections", "cycle"},
            "G9_ENGINE_BINDING_CODEC: complete scoped source",
        )
        canonical = json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        _need(
            canonical == raw and config["cycle"] == carrier["cycle"],
            "G9_ENGINE_BINDING_CODEC: source/cycle drift",
        )
        return config
    except (ValueError, TypeError, UnicodeError, zlib.error) as exc:
        raise PosedRefused("G9_ENGINE_BINDING_CODEC: invalid frozen source") from exc


def engine_model_digest(ixn, actors):
    """Bind actual shared neural/cache state and every posed circuit, not journal bookkeeping."""
    shared = copy.deepcopy(ixn._snapshot())
    attrs = shared["state"]["tree"]["attrs"]
    for key in (
        "_cursor",
        "_g9_carriers",
        "_redo_count",
        "_unjournaled_refused",
        "_hydrated",
        "_hydration_count",
        "_hydration_refusals",
    ):
        attrs.pop(key, None)
    circuits = {}
    for pid, actor in sorted(actors.items()):
        state = actor.state_dict()
        for key in ("event", "selection", "retention_claims"):
            state.pop(key, None)
        circuits[pid] = state
    return hashlib.sha256(
        json.dumps(
            {"shared": shared, "actors": circuits},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _source_receipt():
    names = (
        "snn_data_lake.instruments.g5_part_d_cells",
        "snn_data_lake.instruments.g5_readout_cells",
        "snn_data_lake.readout.interaction_learning",
        "snn_data_lake.reservoir.recurrent_pool",
        "snn_data_lake.gates.replay_gate",
        "snn_data_lake.growth.grown_code",
        "snn_data_lake.growth.grown_feed",
        "snn_data_lake.plasticity.r_stdp",
    )
    result = {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in names
    }
    result[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


class OwnedPartDReader:
    """An actual Brain-owned Part D label reader; no second neural rule or encoder.

    The caller supplies the already registered label order, recording seed and
    Problem. No episodes or scientific schedule are drawn here. Mutable PC/SOM/
    DELTA comparators stay in pd_cell until their OWN full carriers are provided.
    """

    def __init__(self, problem, labels, seed, arm):
        from snn_data_lake.instruments import g5_part_d_cells as cells
        from snn_data_lake.instruments import taught_problems as tp
        from snn_data_lake.instruments.g5_readout_cells import PD_POOL_MODES, BrainProducer
        from snn_data_lake.reservoir.recurrent_pool import KernelLsmPool, LsmPool

        _need(tp.PROBLEMS.get(problem.id) is problem, "unknown/custom Problem factory")
        _need(
            type(seed) is int and type(arm) is str and arm in cells.LABEL_ARMS,
            "unknown reader seed/arm",
        )
        mode = cells.PD_MODES[arm]
        _need(
            mode in PD_POOL_MODES,
            "expansion front-end carrier unbound; use unchanged pd_cell control path",
        )
        _need(
            type(labels) in (list, tuple)
            and bool(labels)
            and all(type(x) is str for x in labels)
            and len(set(labels)) == len(labels),
            "label order must be explicit and unique",
        )
        self.problem_id, self.labels, self.seed, self.arm = problem.id, tuple(labels), seed, arm
        self.drive = cells._Drive(seed, mode, problem)
        _need(
            isinstance(self.drive._pool, (KernelLsmPool, LsmPool))
            and isinstance(self.drive._pool, KernelLsmPool),
            "noncanonical pool host",
        )
        self.learner = cells._learner(self.labels, cells.PD_H, self.seed)
        self.producer = None
        if arm in ("RSTDP_BRAIN", "RSTDP_BRAIN_SHUF", "RSTDP_BRAIN_PRODNEUTER"):
            self.producer = BrainProducer(seed, neutered=arm == "RSTDP_BRAIN_PRODNEUTER")

    def state_dict(self):
        """Canonical learner codec + the existing replay gate's kernel carrier.

        Parameters/encoder are fixed factory inputs in THIS slice. Transferred/
        grown pool tables require their own actual transport carrier extension.
        Never encode a CRC alone as though it contains the weights.
        """
        from snn_data_lake.gates.replay_gate import capture
        from snn_data_lake.readout.interaction_learning import _encode_state
        from snn_data_lake.reservoir.recurrent_pool import _crc_arrays

        pool = self.drive._pool
        _need(pool.observer is None, "external pool observer effect has no carrier")
        _need(
            self.learner.naca is None
            and self.learner.readout_zoo is None
            and self.learner.delta_sink is None,
            "optional external reader effect unbound",
        )
        return {
            "schema": "g9.part_d.owner.v1",
            "source_receipt": _source_receipt(),
            "problem_id": self.problem_id,
            "labels": list(self.labels),
            "seed": self.seed,
            "arm": self.arm,
            "learner": self.learner._snapshot(),
            "producer": _encode_state(self.producer),
            "pool": {
                "params_crc": pool.params_fingerprint(),
                "input_crc": _crc_arrays({"w_in": pool.w_in}),
                "host": [pool.frac_bits, pool.p_parts, pool.threads],
                "count": pool.count,
                "boundary": pool.boundary,
                "lesion_record": _encode_state(pool.lesion_record),
                "fired": pool._s.tolist(),
                "kernel_hex": capture(pool._kt, {}).hex(),
                "fingerprint": pool.fingerprint(),
            },
        }

    def load_state_dict(self, saved):
        """Stage whole real instances before installing any of them."""
        from snn_data_lake.gates.replay_gate import restore
        from snn_data_lake.instruments import taught_problems as tp
        from snn_data_lake.readout.interaction_learning import _merge_state
        from snn_data_lake.reservoir.recurrent_pool import _crc_arrays

        _need(
            type(saved) is dict
            and set(saved)
            == {
                "schema",
                "source_receipt",
                "problem_id",
                "labels",
                "seed",
                "arm",
                "learner",
                "producer",
                "pool",
            },
            "malformed posed carrier",
        )
        _need(
            saved["schema"] == "g9.part_d.owner.v1"
            and saved["source_receipt"] == _source_receipt()
            and (saved["problem_id"], saved["labels"], saved["seed"], saved["arm"])
            == (self.problem_id, list(self.labels), self.seed, self.arm),
            "posed factory/label identity changed",
        )
        staged = type(self)(tp.PROBLEMS[self.problem_id], self.labels, self.seed, self.arm)
        staged.learner._restore_snapshot(saved["learner"])
        _merge_state(staged.producer, saved["producer"])
        p = saved["pool"]
        _need(
            type(p) is dict
            and set(p)
            == {
                "params_crc",
                "input_crc",
                "host",
                "count",
                "boundary",
                "lesion_record",
                "fired",
                "kernel_hex",
                "fingerprint",
            },
            "malformed pool carrier",
        )
        pool = staged.drive._pool
        _need(
            p["params_crc"] == pool.params_fingerprint()
            and p["input_crc"] == _crc_arrays({"w_in": pool.w_in})
            and p["host"] == [pool.frac_bits, pool.p_parts, pool.threads],
            "pool parameters/transport/encoder are not reconstructible by this frozen factory",
        )
        _need(
            type(p["count"]) is int
            and p["count"] >= 0
            and (p["boundary"] is None or type(p["boundary"]) is int)
            and type(p["fired"]) is list
            and len(p["fired"]) == pool.n
            and all(type(x) is bool for x in p["fired"])
            and type(p["kernel_hex"]) is str,
            "invalid count/fired/runtime carrier",
        )
        restore(bytes.fromhex(p["kernel_hex"]), pool._kt)
        pool._s[:] = p["fired"]
        pool.count, pool.boundary = p["count"], p["boundary"]
        pool.lesion_record = _merge_state(None, p["lesion_record"])
        _need(pool.fingerprint() == p["fingerprint"], "whole pool continuation differs")
        self.drive, self.learner, self.producer = staged.drive, staged.learner, staged.producer

    def snapshot(self):
        """Rollback uses independent real pool copies, preserving every parameter."""
        from snn_data_lake.readout.interaction_learning import _encode_state

        _need(
            callable(getattr(self.learner, "_snapshot", None)),
            "canonical R2 full learner snapshot is not installed",
        )
        return {
            "pool": copy.deepcopy(self.drive._pool),
            "learner": self.learner._snapshot(),
            "producer": _encode_state(self.producer),
        }

    def rollback(self, saved):
        from snn_data_lake.readout.interaction_learning import _merge_state

        _need(
            callable(getattr(self.learner, "_restore_snapshot", None)),
            "canonical R2 full learner rollback is not installed",
        )
        self.learner._restore_snapshot(saved["learner"])
        _merge_state(self.producer, saved["producer"])
        # A14 restores this SAME snapshot before and after the set. Installing
        # its pool object directly would mutate the saved pre-read reference.
        self.drive._pool = copy.deepcopy(saved["pool"])


def posed_source(value):
    """Strict canonical source JSON; no construction, feature extraction or draw."""

    def need(v, m):
        if not v:
            raise PosedRefused(m)

    need(
        type(value) is dict
        and value.get("op")
        in {
            "bind",
            "teach",
            "read",
            "reset",
            "admit",
            "engine_teach",
            "engine_read",
            "freeze_selection",
            "claim_retention",
        },
        "unknown typed posed operation",
    )
    op = value["op"]
    common = {"op", "problem_id"}
    need(type(value.get("problem_id")) is str and value["problem_id"], "problem identity")
    if op == "bind":
        need(set(value) == common | {"factory"}, "bind fields")
        f = value["factory"]
        need(
            type(f) is dict
            and set(f) - {"engine_binding"}
            == {
                "problem_id",
                "labels",
                "seed",
                "arm",
                "units",
                "birth_rows",
                "w_feed",
                "t_post_g",
                "source_signs",
                "attachments",
                "out_eligible",
                "population_origin",
            },
            "factory fields",
        )
        need(f["population_origin"] in {"born", "constructed"}, "named real population origin")
        if "engine_binding" in f:
            config = unpack_engine_binding(f["engine_binding"])
            need(
                type(config) is dict
                and set(config)
                in ({"schedule", "projections"}, {"schedule", "projections", "cycle"}),
                "complete engine binding carrier",
            )
            if "cycle" in config:
                need(
                    type(config["cycle"]) is dict
                    and config["cycle"].get("schema") == "g9.engine.cycle.v1",
                    "typed scoped engine cycle",
                )

                # json.dumps below rejects non-finite values; no opaque runtime object is a carrier.
                def primitive(item):
                    return (
                        item is None
                        or type(item) in (str, bool, int, float)
                        or type(item) is list
                        and all(primitive(x) for x in item)
                        or type(item) is dict
                        and all(type(k) is str and primitive(v) for k, v in item.items())
                    )

                need(primitive(config["cycle"]), "primitive scoped cycle source")
                # One validator owns supported options, including restore: an
                # unknown core/feedback flag cannot silently select a default.
                from snn_data_lake.growth.growth_loop import LedgerError, _cycle_recipe

                try:
                    _cycle_recipe(config["cycle"])
                except LedgerError as exc:
                    raise PosedRefused(str(exc)) from exc
            need(
                type(config["schedule"]) is list
                and config["schedule"]
                and all(
                    type(epoch) is list
                    and epoch
                    and all(
                        type(episode) is list
                        and episode
                        and all(
                            type(t) is list
                            and len(t) == 3
                            and type(t[0]) is str
                            and t[1] in {"context", "probe"}
                            for t in episode
                        )
                        for episode in epoch
                    )
                    for epoch in config["schedule"]
                ),
                "typed frozen engine schedule",
            )
            need(type(config["projections"]) is dict, "typed frozen projection carrier")
        need(f["problem_id"] == value["problem_id"] and type(f["seed"]) is int, "factory identity")
        from snn_data_lake.instruments.g5_part_d_cells import LABEL_ARMS, PD_H
        from snn_data_lake.readout import _IXN_INTENTS

        need(
            type(f["labels"]) is list
            and tuple(f["labels"]) == _IXN_INTENTS
            and f["arm"] in LABEL_ARMS,
            "actual canonical labels/registered arm",
        )
        units_from_rows(f["units"], len(_IXN_INTENTS))
        need(
            type(f["birth_rows"]) is list
            and all(type(r) is dict and type(r.get("unit_id")) is int for r in f["birth_rows"]),
            "birth carrier shape",
        )
        need(
            type(f["w_feed"]) is list
            and len(f["w_feed"]) == PD_H
            and all(
                type(r) is list
                and len(r) == len(f["units"])
                and all(type(w) is float and math.isfinite(w) for w in r)
                for r in f["w_feed"]
            ),
            "G9_POOL256_WIDTH: explicit float64 pool256-by-grown feed; no padding/transposition",
        )
        need(
            type(f["t_post_g"]) is float
            and math.isfinite(f["t_post_g"])
            and f["t_post_g"] > max((u["window"] for u in f["units"]), default=0),
            "source-bound grown read time",
        )
        need(
            type(f["source_signs"]) is dict
            and set(f["source_signs"]) == set(_IXN_INTENTS)
            and all(type(v) is int and v in (-1, 1) for v in f["source_signs"].values()),
            "explicit readout source-sign projection contract",
        )
        need(
            type(f["attachments"]) is list and type(f["out_eligible"]) is list,
            "typed attachment and output-mask lists",
        )
    elif op in {"teach", "read", "engine_teach", "engine_read"}:
        read = op in {"read", "engine_read"}
        need(
            set(value)
            == common | {"turns", "shifted_label", "neuron_mask", "return_mask", "return_neutered"},
            "pose fields",
        )
        turns = value["turns"]
        need(
            type(turns) is list
            and turns
            and all(
                type(t) is list
                and len(t) == 3
                and type(t[0]) is str
                and t[1] in {"context", "probe"}
                for t in turns
            ),
            "typed actual turns",
        )
        need(
            (turns[-1][1] == "probe" or read and turns[-1][1] == "context")
            and all(t[1] == "context" for t in turns[:-1]),
            "actual teach episode or label-free read_split prefix boundary",
        )
        need(
            all(t[2] is None for t in turns)
            if read
            else all(t[2] is None for t in turns[:-1]) and type(turns[-1][2]) is str,
            "OFF reads must be label-free; teaching label is probe-only",
        )
        need(
            value["shifted_label"] is None or op == "teach" and type(value["shifted_label"]) is str,
            "shuffle identity",
        )
        need(
            type(value["return_neutered"]) is bool
            and all(
                type(m) is list and all(type(b) is bool for b in m)
                for m in (value["neuron_mask"], value["return_mask"])
            ),
            "actual circuit masks",
        )
    elif op == "freeze_selection":
        need(
            set(value) == common | {"scope", "winner", "model_digest", "retention"},
            "complete frozen selection source",
        )
        need(value["scope"] == ENGINE_SELECTION_SCOPE, "registered no-prune selection scope")
        need(
            type(value["winner"]) is list and all(type(x) is bool for x in value["winner"]),
            "typed actual winner mask",
        )
        need(
            type(value["model_digest"]) is str
            and len(value["model_digest"]) == 64
            and all(c in "0123456789abcdef" for c in value["model_digest"]),
            "model digest",
        )
        need(type(value["retention"]) is dict, "frozen retention schedule manifest")
        for pid, entry in value["retention"].items():
            need(
                type(pid) is str
                and pid
                and type(entry) is dict
                and set(entry) == {"seed", "select_digest", "final_digest"}
                and type(entry["seed"]) is int,
                "retention identity",
            )
            need(
                all(
                    type(entry[k]) is str
                    and len(entry[k]) == 64
                    and all(c in "0123456789abcdef" for c in entry[k])
                    for k in ("select_digest", "final_digest")
                ),
                "retention schedule digest",
            )
    elif op == "claim_retention":
        need(set(value) == common | {"old_problem_id", "selection_id"}, "retention claim source")
        need(
            all(type(value[k]) is str and value[k] for k in ("old_problem_id", "selection_id")),
            "retention claim identity",
        )
    elif op == "admit":
        need(
            set(value) == common | {"units", "birth_rows", "w_feed", "attachments"},
            "admission fields",
        )
    else:
        need(set(value) == common, "reset fields")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unit_rows(units):
    return [
        {
            "unit_id": u.unit_id,
            "w_in": list(u.w_in),
            "threshold": u.threshold,
            "window": u.window,
            "sign": u.sign,
            "out": list(u.out),
            "active": u.active,
        }
        for u in sorted(units, key=lambda u: u.unit_id)
    ]


def units_from_rows(rows, label_count):
    from snn_data_lake.growth.grown_code import GrownUnit, _check_unit

    _need(type(rows) is list, "actual unit rows required")
    units = []
    for r in rows:
        _need(
            type(r) is dict
            and set(r) == {"unit_id", "w_in", "threshold", "window", "sign", "out", "active"}
            and type(r["unit_id"]) is int
            and r["unit_id"] >= 0
            and type(r["active"]) is bool,
            "unit carrier fields",
        )
        u = GrownUnit(
            r["unit_id"],
            tuple(r["w_in"]),
            r["threshold"],
            r["window"],
            r["sign"],
            tuple(r["out"]),
            r["active"],
        )
        _check_unit(u, u.out)
        _need(
            len(u.w_in) == 52 and len(u.out) == label_count,
            "G9_CHAT52_WIDTH: 52-code/canonical-label unit",
        )
        units.append(u)
    _need(
        [u.unit_id for u in units] == sorted({u.unit_id for u in units}),
        "unit IDs must be distinct and ordered",
    )
    return tuple(units)


class PosedActor:
    """Real pool, Part D reader, actual grown segment/feed and plastic return state."""

    def __init__(self, factory, birth_records):
        import numpy as np

        from snn_data_lake.growth.grown_code import GrownReturnPort
        from snn_data_lake.growth.grown_feed import check_feed
        from snn_data_lake.instruments import inseparable_mapping as im
        from snn_data_lake.instruments import taught_problems as tp

        posed_source({"op": "bind", "problem_id": factory["problem_id"], "factory": factory})
        _need(factory["problem_id"] in tp.PROBLEMS, "registered problem")
        self.factory = copy.deepcopy(factory)
        self.reader = OwnedPartDReader(
            tp.PROBLEMS[factory["problem_id"]],
            tuple(factory["labels"]),
            factory["seed"],
            factory["arm"],
        )
        _need(self.reader.labels == tuple(im._IXN_INTENTS), "canonical ordered output vocabulary")
        self.units = units_from_rows(factory["units"], len(im._IXN_INTENTS))
        _need(
            type(factory["out_eligible"]) is list
            and len(factory["out_eligible"]) == len(self.units)
            and all(
                type(m) is list
                and len(m) == len(im._IXN_INTENTS)
                and all(type(b) is bool for b in m)
                for m in factory["out_eligible"]
            ),
            "complete actual grown-output eligibility mask",
        )
        born = {r["unit_id"]: r for r in factory["birth_rows"]}
        if factory["population_origin"] == "born":
            _need(
                len(born) == len(factory["birth_rows"])
                and set(born) == {u.unit_id for u in self.units}
                and all(birth_records.get(k) == r for k, r in born.items()),
                "actual same-lake IMMORTAL birth rows, not caller assertions",
            )
            _need(
                all(r["problem_id"] == factory["problem_id"] for r in born.values()),
                "wrong posing population",
            )
            from snn_data_lake.gates.unit_label_gate import require_admissible

            require_admissible(
                born.values(),
                resolve_problem=lambda p: p in tp.PROBLEMS,
                known_unit=lambda uid: False,
            )
        else:
            _need(not born, "constructed arms are not admissions and carry no birth rows")
            born = {
                u.unit_id: {**row, "origin": "constructed"}
                for u, row in zip(self.units, unit_rows(self.units), strict=True)
            }
        feed = np.asarray(factory["w_feed"])
        _need(
            feed.dtype.kind in "fiu" and feed.shape == (self.reader.drive._pool.n, len(self.units)),
            "real pool x unit feed matrix; no shape coercion",
        )
        check_feed(feed, self.units)
        self.feed = feed.astype(np.float64).tolist()
        _need(
            type(factory["t_post_g"]) is float
            and math.isfinite(factory["t_post_g"])
            and (not self.units or factory["t_post_g"] > max(u.window for u in self.units)),
            "explicit source-bound grown-segment read time",
        )
        self.port = None
        if self.units:
            n = self.reader.learner.neurons[0]
            rule = {
                k: float(getattr(n, k))
                for k in ("lr", "a_plus", "a_minus", "tau_plus", "tau_minus", "tau_e")
            }
            _need(
                all(
                    {k: float(getattr(other, k)) for k in rule} == rule
                    for other in self.reader.learner.neurons
                ),
                "one actual Part D local rule profile",
            )
            self.port = GrownReturnPort(
                self.units,
                born,
                self.reader.labels,
                factory["source_signs"],
                factory["attachments"],
                rule,
            )
        else:
            _need(not factory["attachments"], "empty pre-growth segment cannot have return targets")
        self.clock = 0.0
        self.last_event_id = None
        self.return_neutered = False
        self.reads = self.teaches = 0
        self.selection = None
        self.retention_claims = {}

    def snapshot(self):
        # Selection/claim facts are append-only protocol state, never reversible neural trials.
        return {
            "factory": copy.deepcopy(self.factory),
            "neutered": self.return_neutered,
            "reader": self.reader.snapshot(),
            "port": None if self.port is None else self.port.state_dict(),
            "units": self.units,
            "feed": copy.deepcopy(self.feed),
            "clock": self.clock,
            "event": self.last_event_id,
            "reads": self.reads,
            "teaches": self.teaches,
        }

    def rollback(self, snap):
        self.factory, self.return_neutered = snap["factory"], snap["neutered"]
        self.reader.rollback(snap["reader"])
        if self.port is not None:
            self.port.load_state_dict(snap["port"])
        self.units, self.feed = snap["units"], snap["feed"]
        self.clock, self.last_event_id = snap["clock"], snap["event"]
        self.reads, self.teaches = snap["reads"], snap["teaches"]

    def apply(self, value, *, owner, event_id):
        """The owner has already admitted the exact canonical source; no second emit."""

        source = posed_source(value)
        _need(
            owner._r2_fault is None
            and owner._g9_actors[self.factory["problem_id"]] is self
            and owner._g9_apply_receipt
            == {
                "event_id": event_id,
                "problem_id": self.factory["problem_id"],
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
            "no owner/common-writer admitted-effect binding",
        )
        return self._effect(value, event_id=event_id)

    def validate_effect(self, value):
        # distinct-from: gates/selfcheck_cache.py -- that reads source-file inputs; this is the
        # pure width/vocabulary refusal of a posed teach/read source against this actor.
        # The owner runs it BEFORE the common writer appends (a deterministic input error must
        # not become an immortal unreplayable event); replay runs it again through `_effect`.
        posed_source(value)
        op = value["op"]
        if self.selection is not None:
            _need(op not in {"teach", "engine_teach", "reset"}, "G9_SELECTION_FROZEN")
        if op not in {"teach", "read", "engine_teach", "engine_read"}:
            return
        if op in {"engine_teach", "engine_read"}:
            _need(
                self.factory["arm"] == "RSTDP_BRAIN"
                and len(value["turns"]) == 1
                and value["turns"][0][1] == "probe"
                and value["shifted_label"] is None,
                "G9_ENGINE_SCOPE_UNREADY",
            )
        _need(
            len(value["neuron_mask"]) == len(self.reader.learner.neurons)
            and len(value["return_mask"]) == (0 if self.port is None else len(self.port.rows)),
            "complete actual readout and return circuit masks",
        )
        if op in {"teach", "engine_teach"}:
            label = (
                value["shifted_label"]
                if self.reader.arm == "RSTDP_BRAIN_SHUF"
                else value["turns"][-1][2]
            )
            _need(
                label in self.reader.labels
                and (self.reader.arm == "RSTDP_BRAIN_SHUF" or value["shifted_label"] is None),
                "registered target/shuffle vocabulary",
            )

    def _effect(self, value, *, event_id):
        """One effect implementation; live caller admits first, boot caller validates full receipt."""
        from snn_data_lake.growth.grown_code import latencies, segment_psp
        from snn_data_lake.instruments import g5_readout_cells as cells
        from snn_data_lake.instruments import inseparable_mapping as im

        posed_source(value)
        _need(event_id != self.last_event_id, "duplicate actor event")
        if value["op"] == "freeze_selection":
            _need(self.selection is None, "G9_SELECTION_ALREADY_FROZEN")
            self.selection = {
                k: copy.deepcopy(v) for k, v in value.items() if k not in {"op", "problem_id"}
            }
            self.selection["selection_id"] = event_id
            self.last_event_id = event_id
            return copy.deepcopy(self.selection)
        if value["op"] == "claim_retention":
            old = value["old_problem_id"]
            _need(
                self.selection is not None
                and value["selection_id"] == self.selection["selection_id"]
                and old in self.selection["retention"],
                "G9_RETENTION_SELECTION_MISMATCH",
            )
            _need(old not in self.retention_claims, "G9_RETENTION_ALREADY_CONSUMED")
            self.retention_claims[old] = event_id
            self.last_event_id = event_id
            return {
                "selection_id": value["selection_id"],
                "old_problem_id": old,
                "claim_id": event_id,
            }
        if value["op"] in {"engine_teach", "engine_read"}:
            self.validate_effect(value)
            value = {**value, "op": "teach" if value["op"] == "engine_teach" else "read"}
        if value["op"] == "reset":
            self.reader.drive._pool.reset_carriers()
            if self.port is not None:
                self.port.reset_context()
            self.last_event_id = event_id
            return None
        _need(
            value["op"] in {"teach", "read"}, "bind/admit belongs to owner atomic registry staging"
        )
        self.return_neutered = value["return_neutered"]
        self.validate_effect(value)
        pool = self.reader.drive._pool
        pool.reset_carriers()  # actual G5 episode reset, return continuation is explicitly separate.
        tau = self.reader.learner.tau
        afferents = (
            {}
            if self.port is None
            else self.port.context(self.units, self.clock, neutered=value["return_neutered"])
        )
        features, ranks, crossing = None, None, None
        for text, _role, _target in value["turns"]:
            ranks = im.ranks(text)
            crossing = dict(
                zip(
                    (u.unit_id for u in self.units),
                    latencies(self.units, ranks, tau, afferents=afferents),
                    strict=True,
                )
            )
            phi = segment_psp(self.units, ranks, tau, self.factory["t_post_g"], afferents=afferents)
            feed = [sum(w * p for w, p in zip(row, phi, strict=True)) for row in self.feed]
            features = cells.pool_features(pool, text, feed=feed).tolist()
        learner = self.reader.learner
        answer = learner.respond(features)
        reward = 0.0
        if value["op"] == "teach":
            label = (
                value["shifted_label"]
                if self.reader.arm == "RSTDP_BRAIN_SHUF"
                else value["turns"][-1][2]
            )
            if self.reader.arm != "RSTDP_NOTUNE":
                if self.reader.producer is not None:
                    reward = float(
                        self.reader.producer.factor(
                            value["turns"][-1][0],
                            cells.PD_OUTCOME[0] if answer == label else cells.PD_OUTCOME[1],
                        )
                    )
                elif self.reader.arm == "RSTDP_BYPASS":
                    reward = float(cells.PD_OUTCOME[0] if answer == label else cells.PD_OUTCOME[1])
                old_mask = learner._g9_neuron_mask
                learner._g9_neuron_mask = list(value["neuron_mask"])
                try:
                    learner.interact(
                        features,
                        reward
                        if self.reader.producer is not None or self.reader.arm == "RSTDP_BYPASS"
                        else label,
                        _key=value["turns"][-1][0],
                        _event_id=event_id,
                    )
                finally:
                    learner._g9_neuron_mask = old_mask
            self.teaches += 1
        else:
            self.reads += 1
        answer = learner.respond(features)
        # Existing Part D posts are scheduled at t_post. Emit the selected REAL
        # neuron's post through the existing pre_spike/post_spike operations;
        # no new firing threshold or expected-answer source is introduced.
        li = learner.responses.index(answer)
        # A scheduled readout emission uses the actual R-STDP neuron methods.
        # Learning-OFF may carry return state but never changes reader traces.
        neuron = copy.deepcopy(learner.neurons[li])
        code = learner.encode(features)
        for k, t in enumerate(code):
            neuron.pre_spike(k, float(t))
        neuron.post_spike(learner.t_post)
        packet = (answer, self.clock + float(neuron.t_post))
        delay = max((r["delay"] for r in self.port.rows), default=0) if self.port else 0
        end = (
            self.clock
            + max(float(learner.t_post) + delay, max((u.window for u in self.units), default=0))
            + 1.0
        )
        if self.port is not None:
            self.port.apply(
                event_id=event_id,
                units=self.units,
                packet=packet,
                target_latencies=crossing,
                start=self.clock,
                end=end,
                reward=reward,
                learn=value["op"] == "teach" and self.reader.arm != "RSTDP_NOTUNE",
                eligible=tuple(value["return_mask"]),
                neutered=value["return_neutered"],
            )
        self.clock, self.last_event_id = end, event_id
        return answer

    def state_dict(self):
        return {
            "schema": "g9.posed.actor.v2",
            "factory": copy.deepcopy(self.factory),
            "units": unit_rows(self.units),
            "feed": copy.deepcopy(self.feed),
            "reader": self.reader.state_dict(),
            "port": None if self.port is None else self.port.state_dict(),
            "clock": self.clock,
            "event": self.last_event_id,
            "neutered": self.return_neutered,
            "reads": self.reads,
            "teaches": self.teaches,
            "selection": copy.deepcopy(self.selection),
            "retention_claims": copy.deepcopy(self.retention_claims),
        }

    def expanded(self, value, birth_records):
        """Stage an admission, preserving every old learned weight/trace/pending arrival."""
        from snn_data_lake.instruments import inseparable_mapping as im

        _need(self.selection is None, "G9_SELECTION_FROZEN")
        _need(
            self.factory["population_origin"] == "born", "constructed control cannot admit growth"
        )
        factory = copy.deepcopy(self.factory)
        for key in ("units", "birth_rows", "w_feed", "attachments"):
            factory[key] = copy.deepcopy(value[key])
        old_ids = [u.unit_id for u in self.units]
        new_ids = [r["unit_id"] for r in value["units"]]
        _need(
            new_ids[: len(old_ids)] == old_ids and len(new_ids) > len(old_ids),
            "admission extends the actual ordered segment; no old target replacement",
        )
        # New output circuits are explicitly eligible, per owner1038. Existing masks persist.
        factory["out_eligible"] += [
            [True] * len(u.out)
            for u in units_from_rows(value["units"], len(im._IXN_INTENTS))
            if u.unit_id not in old_ids
        ]
        staged = type(self)(factory, birth_records)
        _need(
            unit_rows(staged.units[: len(old_ids)]) == unit_rows(self.units),
            "admission cannot erase old learned outputs, active flags or immutable fields",
        )
        _need(
            all(
                new[: len(old_ids)] == old for new, old in zip(staged.feed, self.feed, strict=True)
            ),
            "admission cannot resample old feed",
        )
        staged.reader.load_state_dict(self.reader.state_dict())
        if self.port is not None:
            old_keys = [(r["source"], r["target"]) for r in self.port.rows]
            new_index = {(r["source"], r["target"]): k for k, r in enumerate(staged.port.rows)}
            _need(
                all(
                    key in new_index and staged.port.rows[new_index[key]] == old
                    for key, old in zip(old_keys, self.port.rows, strict=True)
                ),
                "admission cannot replace an old attachment or receipt",
            )
            for old_k, key in enumerate(old_keys):
                new_k = new_index[key]
                for attr in ("weights", "eligibility", "pre", "post"):
                    getattr(staged.port, attr)[new_k] = copy.deepcopy(
                        getattr(self.port, attr)[old_k]
                    )
            staged.port.pending = [(new_index[old_keys[k]], t) for k, t in self.port.pending]
            for attr in ("clock", "last_event_id", "reset_count"):
                setattr(staged.port, attr, getattr(self.port, attr))
            staged.port._validate_state()
        staged.clock, staged.last_event_id = self.clock, self.last_event_id
        staged.reads, staged.teaches = self.reads, self.teaches
        staged.return_neutered = self.return_neutered
        staged.selection = copy.deepcopy(self.selection)
        staged.retention_claims = copy.deepcopy(self.retention_claims)
        return staged

    @classmethod
    def restore(cls, saved, birth_records):
        legacy_fields = {
            "schema",
            "factory",
            "units",
            "feed",
            "reader",
            "port",
            "clock",
            "event",
            "neutered",
            "reads",
            "teaches",
        }
        if type(saved) is dict and saved.get("schema") == "g9.posed.actor.v1":
            _need(set(saved) == legacy_fields, "complete legacy actor carrier")
            # Genuine v1 could never execute a final retention read.
            saved = {
                **saved,
                "schema": "g9.posed.actor.v2",
                "selection": None,
                "retention_claims": {},
            }
        _need(
            type(saved) is dict
            and set(saved) == legacy_fields | {"selection", "retention_claims"}
            and saved["schema"] == "g9.posed.actor.v2",
            "complete actor carrier",
        )
        staged = cls(saved["factory"], birth_records)
        _need(
            unit_rows(staged.units) == saved["units"] and staged.feed == saved["feed"],
            "admitted population/feed differs from the saved factory",
        )
        staged.reader.load_state_dict(saved["reader"])
        _need((staged.port is None) == (saved["port"] is None), "missing actual return carrier")
        if staged.port is not None:
            staged.port.load_state_dict(saved["port"])
        _need(
            type(saved["neutered"]) is bool
            and type(saved["clock"]) is float
            and math.isfinite(saved["clock"])
            and saved["clock"] >= 0
            and all(type(saved[k]) is int and saved[k] >= 0 for k in ("reads", "teaches")),
            "invalid actor clock/counts",
        )
        _need(
            saved["event"] is None or type(saved["event"]) is str and saved["event"],
            "invalid actor continuation identity",
        )
        staged.clock, staged.last_event_id = saved["clock"], saved["event"]
        _need(
            staged.port is None or staged.port.clock == staged.clock,
            "G9_RETURN_CLOCK_JOIN: actor and return carrier differ",
        )
        staged.return_neutered = saved["neutered"]
        staged.reads, staged.teaches = saved["reads"], saved["teaches"]
        selection, claims = saved["selection"], saved["retention_claims"]
        _need(
            type(claims) is dict
            and all(type(k) is str and type(v) is str and v for k, v in claims.items()),
            "retention claim carrier",
        )
        if selection is None:
            _need(not claims, "claims without a frozen selection")
        else:
            _need(
                type(selection) is dict
                and set(selection)
                == {"selection_id", "scope", "winner", "model_digest", "retention"}
                and type(selection["selection_id"]) is str
                and selection["selection_id"],
                "complete frozen selection carrier",
            )
            posed_source(
                {
                    "op": "freeze_selection",
                    "problem_id": staged.factory["problem_id"],
                    **{k: v for k, v in selection.items() if k != "selection_id"},
                }
            )
            _need(set(claims) <= set(selection["retention"]), "claim for unregistered old task")
            _need(selection["winner"] == [True] * len(staged.units), "no-prune winner carrier")
        staged.selection, staged.retention_claims = copy.deepcopy(selection), copy.deepcopy(claims)
        return staged


def actor_context(actor):
    """Primitive learned carrier shared with the ACTUAL 52-input Brain chat learner."""
    aff = (
        {}
        if actor.port is None
        else actor.port.context(actor.units, actor.clock, neutered=actor.return_neutered)
    )
    context = {
        "problem_id": actor.factory["problem_id"],
        "units": unit_rows(actor.units),
        "afferents": [[uid, [[t, w] for t, w in aff[uid]]] for uid in sorted(aff)],
        "t_post_g": actor.factory["t_post_g"],
        "out_eligible": copy.deepcopy(actor.factory["out_eligible"]),
    }
    if actor.factory.get("engine_binding", {}).get("cycle", {}).get("core_plasticity") == "frozen":
        context["core_eligible"] = [False] * len(actor.reader.learner.responses)
    return context


def sync_actor_context(ixn, actors):
    """Install actual locally updated out weights; immutable born fields cannot drift."""

    context = ixn._g9_context
    if context is None:
        return
    pid = context["problem_id"]
    _need(pid in actors, "live context lacks its real owner actor")
    actor = actors[pid]
    updated = units_from_rows(context["units"], len(ixn.responses))
    _need(
        len(updated) == len(actor.units)
        and all(
            (a.unit_id, a.w_in, a.threshold, a.window, a.sign, a.active)
            == (b.unit_id, b.w_in, b.threshold, b.window, b.sign, b.active)
            for a, b in zip(actor.units, updated, strict=True)
        ),
        "chat changed immutable population",
    )
    actor.units = updated
    actor.factory["units"] = unit_rows(updated)
    if actor.port is not None:
        actor.port.validate_population(updated)


def _chat_context(ixn):
    from snn_data_lake.readout import _IXN_INTENTS

    c = ixn._g9_context
    fields = {"problem_id", "units", "afferents", "t_post_g", "out_eligible"}
    _need(
        type(c) is dict
        and set(c) in (fields, fields | {"core_eligible"})
        and ixn.n_in == 52
        and tuple(ixn.responses) == _IXN_INTENTS,
        "typed Brain chat52/grown return context",
    )
    if "core_eligible" in c:
        _need(
            type(c["core_eligible"]) is list
            and len(c["core_eligible"]) == len(ixn.responses)
            and all(type(v) is bool for v in c["core_eligible"]),
            "all actual core circuits",
        )
    units = units_from_rows(c["units"], len(ixn.responses))
    ids = [u.unit_id for u in units]
    pairs = c["afferents"]
    _need(
        type(pairs) is list
        and [r[0] for r in pairs] == ids
        and all(type(r) is list and len(r) == 2 and type(r[1]) is list for r in pairs),
        "complete ordered return queue",
    )
    aff = {}
    for uid, values in pairs:
        _need(
            all(
                type(r) is list
                and len(r) == 2
                and all(type(v) is float and math.isfinite(v) for v in r)
                for r in values
            ),
            "finite actual return time/weight pair",
        )
        aff[uid] = [tuple(r) for r in values]
    _need(
        type(c["t_post_g"]) is float
        and math.isfinite(c["t_post_g"])
        and c["t_post_g"] > max((u.window for u in units), default=0),
        "actual grown read time",
    )
    mask = c["out_eligible"]
    _need(
        type(mask) is list
        and len(mask) == len(units)
        and all(
            type(r) is list and len(r) == len(ixn.responses) and all(type(v) is bool for v in r)
            for r in mask
        ),
        "all actual output circuits",
    )
    return c, units, aff


def chat_grown_drives(ixn, code):
    from snn_data_lake.growth.grown_code import segment_psp

    if ixn._g9_context is None:
        return [0.0] * len(ixn.responses)
    c, units, aff = _chat_context(ixn)
    phi = segment_psp(units, code, ixn.tau, c["t_post_g"], afferents=aff)
    return [
        sum(u.out[k] * p for u, p in zip(units, phi, strict=True))
        for k in range(len(ixn.responses))
    ]


def chat_grown_deposit(ixn, li, code, reward):
    """The existing trial-local R-STDP deposit extended to actual G3 out synapses."""
    from dataclasses import replace

    from snn_data_lake.growth.grown_code import clip_to_sign, latency
    from snn_data_lake.plasticity.r_stdp import accumulate_causal, reward_deltas

    if ixn._g9_context is None:
        return
    c, units, aff = _chat_context(ixn)
    n = ixn.neurons[li]
    updated = []
    for k, unit in enumerate(units):
        t = latency(unit, code, ixn.tau, afferents=aff) if unit.active else None
        if t is None or not c["out_eligible"][k][li]:
            updated.append(unit)
            continue
        e = [0.0]
        accumulate_causal(e, ((0, c["t_post_g"] - t),), n.a_plus, n.tau_plus)
        delta = reward_deltas(e, float(reward), n.lr)[0]
        out = list(unit.out)
        out[li] = clip_to_sign([out[li] + delta], unit.sign, unit.unit_id)[0]
        updated.append(replace(unit, out=tuple(out)))
    c["units"] = unit_rows(updated)


def resource_receipt(actors, engine_arm, ledger, *, silenced=None):
    """RQ1/RQ2/RQ3 of the dated Round 3 brief: actual populations, never profiles.

    G3 2.7 counts every E/I unit; plan G9 compares the final size. SC-012b
    compares equal-count silence using the complete actual inactive population;
    caller-listed subsets cannot discount inactive units from the intervention.
    Evaluations, stored synapses, attachments and
    actual pool ticks/actor intervals are reported, with no equality assertion.
    Logical compute is reported here; this receipt makes no wall-time claim.
    """
    from snn_data_lake.growth.growth_loop import Ledger

    _need(
        type(actors) is dict and engine_arm in actors and actors,
        "G9 resource receipt needs named real arms",
    )
    _need(type(ledger) is Ledger, "G9 resource receipt needs the real ledger")
    _need(
        all(type(a) is PosedActor for a in actors.values()),
        "G9 resource receipt refuses constructed profiles",
    )
    _need(
        actors[engine_arm].factory["population_origin"] == "born",
        "G9 engine arm needs admitted units",
    )
    from snn_data_lake.growth.batch_reconciler import reconcile_batches

    for actor in actors.values():
        if actor.factory["population_origin"] == "born":
            # Accounting is never certified over an unsettled batch; the named refusal stands.
            reconcile_batches(ledger, actor.factory["problem_id"])
    births = ledger.birth_records()
    result = {}
    for name, actor in actors.items():
        ids = tuple(u.unit_id for u in actor.units)
        _need(len(ids) == len(set(ids)), "G9 resource duplicate actual unit")
        if actor.factory["population_origin"] == "born":
            saved = {r["unit_id"]: r for r in actor.factory["birth_rows"]}
            _need(
                set(saved) == set(ids)
                and all(births.get(uid) == row for uid, row in saved.items()),
                "G9 resource actual admission provenance mismatch",
            )
            _need(
                set(ids)
                == {
                    uid
                    for uid, row in births.items()
                    if row["problem_id"] == actor.factory["problem_id"]
                },
                "G9 resource omitted admitted unit",
            )
            evaluations = ledger.ledger(actor.factory["problem_id"]).spend_evals
        else:
            _need(not actor.factory["birth_rows"], "G9 constructed arms are not admissions")
            evaluations = 0  # no candidate search in this actual constructor
        pool = actor.reader.drive._pool
        result[name] = {
            "origin": actor.factory["population_origin"],
            "U": len(ids),
            "E": sum(u.sign == 1 for u in actor.units),
            "I": sum(u.sign == -1 for u in actor.units),
            "unit_ids": list(ids),
            "candidate_evaluations": evaluations,
            "synapses": {
                "grown_input": sum(len(u.w_in) for u in actor.units),
                "grown_output": sum(len(u.out) for u in actor.units),
                "feed": sum(len(r) for r in actor.feed),
                "reader": sum(len(n.w) for n in actor.reader.learner.neurons),
                "pool_recurrent": int(pool.w.size),
                "pool_input": int(pool.w_in.size),
            },
            "attachments": 0 if actor.port is None else len(actor.port.rows),
            "compute": {
                "pool_ticks_since_reset": pool.count,
                "actor_clock": actor.clock,
                "teach_events": actor.teaches,
                "read_events": actor.reads,
            },
        }
    final_u = result[engine_arm]["U"]
    for name, row in result.items():
        _need(
            row["U"] == final_u,
            f"G9_RQ1_UNIT_COUNT_MISMATCH: {name} U={row['U']} engine U={final_u}",
        )
    if silenced is not None:
        _need(
            type(silenced) is dict and set(silenced) <= set(actors) and len(silenced) >= 2,
            "G9 silencing needs named compared arms",
        )
        counts = {}
        for name, ids in silenced.items():
            _need(
                type(ids) is tuple
                and len(ids) == len(set(ids))
                and all(type(uid) is int for uid in ids),
                "G9 actual unique silenced IDs",
            )
            units = {u.unit_id: u for u in actors[name].units}
            _need(
                set(ids) <= set(units) and all(not units[uid].active for uid in ids),
                "G9 silencing receipt lacks actual silenced units",
            )
            inactive = {uid for uid, unit in units.items() if not unit.active}
            _need(
                set(ids) == inactive,
                f"G9_RQ2_UNLISTED_INACTIVE_UNITS: {name} unlisted={sorted(inactive - set(ids))}",
            )
            counts[name] = len(inactive)
        _need(len(set(counts.values())) == 1, "G9_RQ2_SILENCING_COUNT_MISMATCH")
        _need(
            {actors[name].factory["population_origin"] for name in counts}
            == {"born", "constructed"},
            "G9 SC-012b requires grown versus pre-sized silencing",
        )
        for name, count in counts.items():
            result[name]["silenced_N"] = count
    return {
        "schema": "g9.resources.v1",
        "engine_arm": engine_arm,
        "equalized": ["U", "silenced_N"] if silenced is not None else ["U"],
        "reported_only": ["candidate_evaluations", "synapses", "attachments", "compute"],
        "arms": result,
    }


def main():
    """Cheap typed-source/width RED, no Brain, campaign or lake writes."""
    from snn_data_lake.growth.grown_code import GrownUnit

    unit = GrownUnit(1, (0.0,) * 52, 1.0, 12, 1, (0.0,) * 6)
    assert units_from_rows(unit_rows((unit,)), 6) == (unit,)
    broken = unit_rows((unit,))
    broken[0]["w_in"] = [0.0] * 64
    try:
        units_from_rows(broken, 6)
    except PosedRefused as exc:
        assert "52-code" in str(exc)
    else:
        raise AssertionError("G9 width RED failed")
    print("posed_problem self-check OK: typed unit roundtrip and named width RED")


if __name__ == "__main__":
    main()
