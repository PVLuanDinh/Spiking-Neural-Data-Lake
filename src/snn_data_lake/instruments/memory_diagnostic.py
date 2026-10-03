"""G1(a) causal stream recorder and replay harness.

Recordings are experiment artifacts; no brain or replay state is persisted to the lake.

G1(b) extends it (specs/022-unified-brain-arc/evidence/g1b-pool-selection.md section 2, frozen at
23c7e97) without altering G1(a)'s arms: the reference readout (section 2.2) runs beside the grower in
every replay, the pool arm feeds x alone into reservoir/recurrent_pool.py's pools, every replay runs at
one BLAS thread, and R-b6's synthetic stream family (section 2.7) lives here. The campaign that runs
them is instruments/pool_selection.py.

G1(c) (specs/022-unified-brain-arc/evidence/g1c-carried-state.md 2.7, 2.10 "Transport") adds, after
the last arm G1(b) had: a read-only tap of the incumbent grower, taken from an arm-(a) replay
(`replay(..., "a", tap=b)`), the transport arms "pool-copy" (the grower's input weights at b, copied
into the pool's encoder), "pool-act" (the pool driven by the grower's hidden rates) and "pool-swap"
(the swap control: a fresh random encoder at the same b), each fed that tap as `side=`, and the checks
of the channels, the tap's timing, the transport's causality and the tap's read-only contract. The
default replay paths, ARMS[:8] and SCORED_ARMS are as G1(b) left them: a replay without `tap`/`side`
carries no extra key and a plain pool arm's result carries neither "encoder" nor "transport".

This module is past the 800-line soft ceiling, as it was before G1(c) (1,746 lines at P3b). It stays
one module because the tap is taken inside `_replay`'s event loop and the transport arms are arms of
`replay`: splitting the arms from their loop would put a second copy of the loop's event order,
refit bookkeeping and view construction in another file, which is exactly the duplicate-mechanism
defect POLICIES P2 refuses. The transport checks sit beside the R-a checks they mirror for the same
reason; moving the whole instrument family out is an opening, not part of this change.
"""

import argparse
import contextlib
import copy
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info

from snn_data_lake.growth.growth_stop import GrowthStop, feed_and_decide
from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove
from snn_data_lake.reservoir import recurrent_pool as rp

ASSETS = {
    "nmnist_pack.npz": "9863a50ead6190ae",
    "shd_pack.npz": "295eef1755e2993a",
    "eprop_nmnist.pt": "9b471619e39acfa0",
    "eprop_shd.pt": "6b32f9eb38beeca3",
}
ASSET_ENV = (
    "SNN_EPROP_CKPT",
    "SNN_PERCEPTION_PACK",
    "SNN_EPROP_AUDIO_CKPT",
    "SNN_SHD_PACK",
)
_EXECUTIONS = itertools.count()

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it. `_refit_positions`
#: (G1(a), 829fb40) carries the learn verb "fit" and left the gate's governed UNDETERMINABLE ratchet
#: at 1 against a baseline of 0; it is bookkeeping, and says so here. `_corrupt_w` (G1(c)) is the
#: one constructed write of the writing-tap red, declared for the same reason.
GRADIENT_CLAIM = {
    "_refit_positions": (
        "NOT_A_RULE",
        "bookkeeping for R-a1: lists the (tick, event) positions at which the grower refit, from a "
        "recording's rows or a replay's, so the two can be compared. It reads fingerprints already "
        "taken; no weight is touched and no learning signal is consumed",
    ),
    "_corrupt_w": (
        "NOT_A_RULE",
        "the writing-tap red (G1(c) 2.10): adds a constant 1e-3 to the incumbent's W[0, 0], once, and "
        "is reached only inside `_writing_tap`, to prove check_tap_read_only can fail. It consumes "
        "no learning signal and moves no weight in any real replay",
    ),
}


def _crc(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return zlib.crc32(np.asarray(value).tobytes())


def _fingerprint(engine):
    model = engine._model
    return {
        "W": _crc(getattr(model, "W", None)),
        "b": _crc(getattr(model, "b", None)),
        "beta": _crc(getattr(model, "beta", None)),
        "prior": _crc(None if model is None else model.prior_class_),
        "batch_beta": _crc(engine._batch_beta),
    }


def _read_lp(brain):
    facade = getattr(brain, "_lp", None)
    if facade is None:
        return None
    try:
        value = float(facade.lp("cortex"))
    except Exception:
        return None
    return value if math.isfinite(value) else None


# distinct-from: reservoir/recurrent_pool.py -- `_blas_threads` LIMITS the pools (replay() reuses it for
#   its pin) and check_r_b3 reads the backends inline per thread setting; this is the policy's record
#   of them. Folding check_r_b3's inline read onto this one is an opening (recurrent_pool.py is
#   outside this build's files). distinct-from: growth/neuron_zoo.py -- `_worker_ladder` derives
#   thread counts to race; distinct-from: substrate/lake_working_memory.py -- `reopen` rebuilds a
#   cursor from the lake. Neither records a BLAS policy (sites read).
def _blas():
    """The loaded BLAS and OpenMP pools with their effective thread counts (threadpoolctl)."""
    return [
        {
            key: info.get(key)
            for key in ("user_api", "internal_api", "prefix", "version", "num_threads")
        }
        for info in threadpool_info()
    ]


def _policy():
    return {
        "NORD_DEVICE": os.environ.get("NORD_DEVICE"),
        "NORD_GPU": os.environ.get("NORD_GPU"),
        **{name: os.environ.get(name) for name in ASSET_ENV},
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "device": "cpu",
        # G1(b) section 2.2: the BLAS record. Never a replay refusal key (replay() compares the keys
        # above only); a replay's own policy is read inside its one-thread pin.
        "blas": _blas(),
    }


def _revision():
    # git runs at the repository this module lives in (the root `_assets` also anchors on), never
    # the caller's cwd: a self-check run from a scratch cwd (tests/test_selfchecks.py chdirs into
    # tmp_path) or a campaign launched from outside the repo would otherwise fail or bind the wrong
    # head. From inside the repo the reading is the same: porcelain paths are repo-rooted.
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    dirty = bool(status.stdout.strip())
    return head.stdout.strip(), dirty


def _assets():
    root = Path(__file__).resolve().parents[3] / "models"
    found = {}
    for name, expected in ASSETS.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"REFUSED: missing asset {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not digest.startswith(expected):
            raise ValueError(f"REFUSED: asset hash {name}: {digest}")
        found[name] = digest
    return found


def onehot_32(category):
    category = int(category)
    if not 0 <= category < 32:
        raise ValueError(f"category outside onehot_32: {category}")
    row = [0.0] * 32
    row[category] = 1.0
    return row


class _View:
    def __init__(self, channels, values):
        self.channels = frozenset(channels)
        self.values = {key: values[key] for key in self.channels if key in values}

    def get(self, channel):
        if channel not in self.channels:
            raise ValueError(f"forbidden feature channel: {channel}")
        return self.values[channel]


def _feature(arm, view, pool=None):
    if arm == "pool":
        # section 2.2: the pool's phi_t, from x alone through its view (R-b2)
        if pool is None:
            raise ValueError("REFUSED: the pool arm's features come from its pool")
        return pool.percept(view.get("x")).tolist()
    if arm == "a":
        return list(view.get("x"))
    if arm == "leak":
        return list(view.get("x"))
    if arm == "f64":
        return onehot_32(view.get("c")) + [0.0] * 32
    if arm == "d":
        pred = view.get("own_predictions")
        return list(view.get("x")) + (onehot_32(pred) if pred is not None else [0.0] * 32)
    if arm == "d0":
        return list(view.get("x")) + [0.0] * 32
    if arm == "e1":
        return list(view.get("x")) + sum(view.get("x_history"), [])
    if arm == "e1-0":
        return list(view.get("x")) + [0.0] * 192
    if arm == "e2":
        return list(view.get("x")) + sum(
            (onehot_32(c) if c is not None else [0.0] * 32 for c in view.get("c_history")),
            [],
        )
    if arm == "e2-0":
        return list(view.get("x")) + onehot_32(view.get("c")) + [0.0] * 96
    if arm == "forbidden":
        return list(view.get("x")) + onehot_32(view.get("c"))
    if arm == "noise":
        row = list(view.get("x"))
        return [v + int.from_bytes(os.urandom(4), "little") / 2**32 for v in row]
    if arm in ("pool-copy", "pool-swap"):
        # G1(c) 2.7: x alone through the pool's encoder; what the copy copies (w@b) and where the swap
        # swaps act on the encoder, in _PoolArm, never on this row
        if pool is None:
            raise ValueError("REFUSED: the pool arm's features come from its pool")
        return pool.percept(view.get("x")).tolist()
    if arm == "pool-act":
        if pool is None:
            raise ValueError("REFUSED: the pool arm's features come from its pool")
        return pool.percept(_padded_activity(view.get("grower_activity"))).tolist()
    raise ValueError(f"unknown arm {arm}")


# Sites the duplicate-capability hook flagged, read; none pads a grower's activity into the pool's inputs:
# distinct-from: plasticity/eprop_kernel.py::EpropLIFBase -- the recurrent-LIF forward of the two trained perception encoders
# distinct-from: reservoir/recurrent_pool.py::cost -- times a pool's per-percept wall clock
# distinct-from: growth/neurogenesis_engine.py::SyntheticPerceptStream.next -- draws a synthetic percept and its mode
# distinct-from: growth/neuron_zoo.py::_DendriticCell -- one dendritic neuron behind the zoo's contract
# distinct-from: reservoir/spontaneous_activity.py::fit_to_traces -- an AR(1) moment match of a sensor trace
def _padded_activity(activity):
    """The grower's hidden rates for the current percept, zero-padded to the 64 inputs of the
    pool's W_in (2.7's activity form). None (the grower holds no model yet) is zeros; more than
    64 units is refused."""
    padded = np.zeros(rp.N_IN)
    if activity is None:
        return padded
    if len(activity) > rp.N_IN:
        raise ValueError(
            f"REFUSED: the grower holds {len(activity)} units, more than 64: its rates do not fit W_in"
        )
    padded[: len(activity)] = activity
    return padded


ARMS = {
    "a": (64, ("x",)),
    "f64": (64, ("c",)),
    "d": (96, ("x", "own_predictions")),
    "d0": (96, ("x",)),
    "e1": (256, ("x", "x_history")),
    "e1-0": (256, ("x",)),
    "e2": (192, ("x", "c_history")),
    "e2-0": (192, ("x", "c")),
    "forbidden": (96, ("x",)),
    "noise": (64, ("x",)),
    "leak": (64, ("x",)),
    "pool": (rp.N, ("x",)),  # G1(b): phi_t of a recurrent pool; its view declares ONLY x (R-b2)
    # G1(c) 2.7, appended after the last G1(b) arm (SCORED_ARMS is the first eight and never moves):
    # the transport arms. Their declarations are written out here AND frozen in TRANSPORT_CHANNELS,
    # so check_transport_channels' comparison of the two can fail
    "pool-copy": (rp.N, ("x", "w@b")),
    "pool-act": (rp.N, ("grower_activity",)),
    "pool-swap": (rp.N, ("x",)),
}
SCORED_ARMS = tuple(ARMS)[:8]
SCORED_SEGMENTS = ("repeated-canonical", "unseen-order", "repeated-permuted")

# G1(c) 2.7: b is the adaptation-window boundary; the copy is taken at the START of delivery b + 1
TRANSPORT_B = 256
LATE_TAP_EVENTS = 40  # the late-tap red copies at b + 40
TRANSPORT_CHANNELS = {
    "pool-copy": ("x", "w@b"),
    "pool-act": ("grower_activity",),
    "pool-swap": ("x",),
}
TRANSPORT_ARMS = tuple(TRANSPORT_CHANNELS)
POOL_ARMS = ("pool", *TRANSPORT_ARMS)
# the transport record's form, per arm
TRANSPORT_FORM = {"pool-copy": "copy", "pool-act": "activity", "pool-swap": "swap"}
# the tap's fault switch (None: the honest tap); set only by the context managers below
_FAULT = {"tap": None, "late": LATE_TAP_EVENTS}


def _segments(events, acquisitions, n_frames):
    """Section 2.3's segments from the ACQUISITION records: the pass of x_t's vision frame through its order.
    A reflex tick acquires without delivering, so counting deliveries would drift from the frame passes."""
    first_permuted = next(
        (a["vision"]["pointer"] for a in acquisitions if a["vision"]["order"] == "permuted"), None
    )
    for event in events:
        source = event.get("source")
        if source is None:
            event["segment"] = None
        elif source["vision"]["order"] == "permuted":
            passed = source["vision"]["pointer"] - first_permuted
            event["segment"] = "unseen-order" if passed < n_frames else "repeated-permuted"
        else:
            passed = source["vision"]["pointer"]
            event["segment"] = "warm-up" if passed < n_frames else "repeated-canonical"


def record_stream(seed, ticks=2400, *, arm_feed=None):
    """Record delivery, acquisition, sleep, and refit events from an isolated brain."""
    if arm_feed not in (None, "f64"):
        raise ValueError("REFUSED: unsupported in-brain feed")
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    torch.set_num_threads(1)
    if any(os.environ.get(name) is not None for name in ASSET_ENV):
        raise ValueError("REFUSED: asset environment override")
    assets = _assets()
    with throwaway_home():
        brain = make_brain(seed=seed, persist=False)
        if brain.neuro is None or not brain._perceiving_real or not brain._perceiving_audio:
            raise ValueError("REFUSED: live packs or grower unavailable")
        brain.neuro.device = "cpu"
        engine = brain.neuro
        stream = {
            "seed": seed,
            "config": {
                "seed": engine.seed,
                "refit_every": engine.refit_every,
                "buffer_cap": engine.buffer_cap,
                "l_max": engine._l_max,
                "eps_tol": engine._eps_tol,
                "T": engine._T,
            },
            "policy": _policy(),
            "assets": assets,
            "events": [],
            "sleep": [],
            "acquisitions": [],
            "gate_verdicts": [],
            "refits": [],
        }
        inverse = {
            "vision": list(range(len(brain._real_frames))),
            "audio": list(range(len(brain._audio_frames))),
        }
        order = "canonical"
        pending = None
        attended = {}
        original_attend = brain._attend_real_stream

        def attend(latent, gain):
            modality = "vision" if latent is brain._real_percept else "audio"
            result = original_attend(latent, gain)
            attended[modality] = result is not None
            return result

        brain._attend_real_stream = attend

        def acquisition(modality, original, pointer_name, frames_name):
            def wrapped():
                pointer = getattr(brain, pointer_name)
                frames = getattr(brain, frames_name)
                slot = pointer % len(frames)
                result = original()
                row = {
                    "modality": modality,
                    "pointer": pointer,
                    "success": result is not None,
                    "index": inverse[modality][slot],
                    "order": order,
                }
                if pending is not None:
                    pending[modality] = row
                return result

            return wrapped

        brain._perceive_real = acquisition(
            "vision", brain._perceive_real, "_real_ptr", "_real_frames"
        )
        brain._perceive_audio = acquisition(
            "audio", brain._perceive_audio, "_audio_ptr", "_audio_frames"
        )
        original_observe = engine.observe

        def observe(features, category):
            index = len(stream["events"]) + 1
            source = stream["acquisitions"][-1] if stream["acquisitions"] else None
            if index > 1 and (
                source is None or not all(source[m]["success"] for m in ("vision", "audio"))
            ):
                raise ValueError(f"REFUSED: fallback context at delivery {index}")
            if index > 1 and not all(attended.get(m, False) for m in ("vision", "audio")):
                raise ValueError(f"REFUSED: fallback context at delivery {index}")
            prediction = engine.predict_next()
            fallback = bool(engine._pending_prediction[4]) if engine._pending_prediction else False
            row = {
                "t": index,
                "tick": brain._step,
                "x": list(map(float, features)),
                "c": int(category),
                "source": copy.deepcopy(source),
                "attended": {m: attended.get(m, False) for m in ("vision", "audio")},
                "causal_lp": _read_lp(brain),
                "prediction": prediction,
                "fallback": fallback,
            }
            stream["events"].append(row)
            if arm_feed is None:
                result = original_observe(features, category)
            else:
                view = _View(
                    ARMS[arm_feed][1],
                    {
                        "x": row["x"],
                        "c": row["c"],
                        "own_predictions": prediction,
                        "x_history": [[0.0] * 64] * 3,
                        "c_history": [row["c"], None, None, None],
                    },
                )
                result = original_observe(_feature(arm_feed, view), category)
            if index == 720:
                nonlocal order
                rng = np.random.default_rng(seed * 7919 + 1)
                for modality, frame_attr, label_attr in (
                    ("vision", "_real_frames", "_real_labels"),
                    ("audio", "_audio_frames", "_audio_labels"),
                ):
                    permutation = rng.permutation(len(getattr(brain, frame_attr)))
                    setattr(brain, frame_attr, getattr(brain, frame_attr)[permutation])
                    setattr(brain, label_attr, getattr(brain, label_attr)[permutation])
                    inverse[modality] = permutation.tolist()
                order = "permuted"
            return result

        engine.observe = observe
        original_tick = engine.tick

        def sleep_tick():
            row = {
                "tick": engine.ticks + 1,
                "brain_tick": brain._step,
                "event": len(stream["events"]),
                "live_lp": _read_lp(brain),
            }
            before = len(engine.trajectory)
            result = original_tick()
            row["refit"] = _fingerprint(engine) if len(engine.trajectory) > before else None
            if (engine.ticks % engine.refit_every) == 0:
                stream["refits"].append(row)
            stream["sleep"].append(row)
            return result

        engine.tick = sleep_tick
        original_gate = engine._growth_gate

        def gate(trajectory):
            verdict = original_gate(trajectory)
            stream["gate_verdicts"].append(
                {"tick": engine.ticks, "event": len(stream["events"]), "verdict": verdict}
            )
            return verdict

        engine._growth_gate = gate
        for _ in range(ticks):
            pending = {}
            attended = {}
            phase = brain.tick()
            if phase == "awake":
                if set(pending) != {"vision", "audio"} or not all(
                    v["success"] for v in pending.values()
                ):
                    raise ValueError("REFUSED: unsuccessful acquisition")
                stream["acquisitions"].append(pending)
        if brain._neuro_feed_refusals or brain._growth_feed_refusals:
            raise ValueError("REFUSED: grower or growth-gate feed refusal")
        stream["pack_lengths"] = {modality: len(order) for modality, order in inverse.items()}
        _segments(stream["events"], stream["acquisitions"], stream["pack_lengths"]["vision"])
        head, dirty = _revision()
        stream["final"] = {
            "live_n": engine.live_n,
            "live_correct": engine.live_correct,
            "live_persist": engine.live_persist,
            "ticks": engine.ticks,
            "observed": engine.observed,
            "trajectory": engine.trajectory,
            "baselines": engine.baselines,
            "gate_verdicts": stream["gate_verdicts"],
            "fingerprint": _fingerprint(engine),
            "head": head,
            "dirty": dirty,
        }
        return stream


def _source_digest(stream):
    payload = {key: stream[key] for key in ("events", "sleep")}
    return zlib.crc32(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _config_digest(stream):
    return zlib.crc32(
        json.dumps(stream["config"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


# ---- G1(b): the reference readout (g1b-pool-selection.md section 2.2) ---------------------------

# Section 2.2's fixed estimator settings of a measurement (P1 does not bind them: 2.2's policy
# disposition): a solve after every 20th pair, and P = 1.0 * I with a zero at the bias.
READOUT_EVERY = 20
READOUT_LAMBDA = 1.0
# the grower's fields R-b4's twins and R-b8's two halves compare
GROWER_KEYS = ("predictions", "fallbacks", "phi", "fingerprint", "refits")


# distinct-from: readout/online_readout.py -- OnlineReadout moves the GROWER's beta by a delta rule
#   on every wake pair; this reads phi beside the grower and never touches it. distinct-from:
#   reasoning/sorn.py::train_readout, reasoning/pc_cortex_torch.py::train_ridge_readout,
#   reasoning/hjepa_torch.py and hjepa_arm_torch.py's ridge readouts -- one-shot batch solves with the
#   bias penalized like every weight; this is section 2.2's frozen streaming estimator (probe_ridge_
#   frozen.py): a solve every 20th pair, an unpenalized bias, W's columns fixed at the last solve.
class ReferenceReadout:
    """Section 2.2's online ridge from x = [phi_{t-1}, 1] to c_t, run beside the grower.

    `predict()` answers the argmax of x @ W over W's columns -- the categories present at the last
    solve, ascending, the first maximum winning (an exact tie goes to the lowest id) -- or None before
    the first solve and at event 1 (there is no phi_0). `observe(phi_t, c_t)`, called after predicting,
    adds the pair: XtX += multiply.outer(x, x) and Xty[c_t] += x in event order, float64; after every
    20th pair W = solve(XtX + P, [Xty[k] for k in the sorted categories]), P = lambda I except a zero
    at the bias. Its state is fingerprinted apart from the grower's and it never touches the grower."""

    def __init__(self, every=READOUT_EVERY, lam=READOUT_LAMBDA):
        self.every = int(every)
        self.lam = float(lam)
        self.x = None  # [phi_{t-1}, 1]: the row the next prediction reads
        self.xtx = None
        self.xty = {}
        self.w = None
        self.classes = ()
        self.pairs = 0
        self.solves = 0

    def predict(self):
        if self.x is None or self.w is None:
            return None
        return self.classes[int(np.argmax(self.x @ self.w))]

    def observe(self, phi, category):
        self._pair(category)
        self._hold(phi)

    def _pair(self, category):
        if self.x is None:  # event 1: no phi_0, so no pair
            return
        if self.xtx is None:
            self.xtx = np.zeros((self.x.size, self.x.size))
        self.xtx += np.multiply.outer(self.x, self.x)
        c = int(category)
        if c not in self.xty:
            self.xty[c] = np.zeros(self.x.size)
        self.xty[c] += self.x
        self.pairs += 1
        if self.pairs % self.every == 0:
            self._solve()

    def _hold(self, phi):
        self.x = np.append(np.asarray(phi, dtype=np.float64), 1.0)

    def _penalty(self):
        penalty = self.lam * np.eye(self.x.size)
        penalty[-1, -1] = 0.0  # the unpenalized bias
        return penalty

    def _solve(self):
        self.classes = tuple(sorted(self.xty))
        targets = np.stack([self.xty[k] for k in self.classes], axis=1)
        # LAPACK, at replay's one BLAS thread
        self.w = np.linalg.solve(self.xtx + self._penalty(), targets)
        self.solves += 1

    def fingerprint(self):
        """crc32 of XtX, of Xty stacked in sorted-category order, and of W (section 2.2)."""
        order = sorted(self.xty)
        return {
            "XtX": _crc(self.xtx),
            "Xty": _crc(np.stack([self.xty[k] for k in order])) if order else None,
            "W": _crc(self.w),
            "categories": order,
            "classes": list(self.classes),
            "pairs": self.pairs,
            "solves": self.solves,
        }


# ---- R-b0x's readout fixtures: each rule beside its deliberately wrong twin ----------------------


class _PenalizedBias(ReferenceReadout):
    """Red: the bias penalized like every other weight."""

    def _penalty(self):
        return self.lam * np.eye(self.x.size)


class _GuessBeforeSolve(ReferenceReadout):
    """Red: a default answer where section 2.2 answers None."""

    def predict(self):
        guess = super().predict()
        return 0 if guess is None else guess


class _LastMaximum(ReferenceReadout):
    """Red: an exact tie goes to the highest id."""

    def predict(self):
        if self.x is None or self.w is None:
            return None
        scores = self.x @ self.w
        return self.classes[len(scores) - 1 - int(np.argmax(scores[::-1]))]


class _SeenSoFar(ReferenceReadout):
    """Red: W re-solved over every category seen so far, not only those present at the last solve."""

    def predict(self):
        if self.w is not None and tuple(sorted(self.xty)) != self.classes:
            self._solve()
        return super().predict()


def _drive_readout(readout, rows, *, pair_first=False):
    """(phi_t, c_t) rows in section 2.2's order: predict c_t, then add the pair. `pair_first` is the
    constructed leak: the pair (and any solve it triggers) before the prediction."""
    predictions = []
    for phi, category in rows:
        if pair_first:
            readout._pair(category)
            predictions.append(readout.predict())
            readout._hold(phi)
        else:
            predictions.append(readout.predict())
            readout.observe(phi, category)
    return predictions


def check_r_b0x_readout():
    """R-b0x's readout part (section 2.7): fixtures worked by hand at every = 2, each with its red."""
    numbers, red = {}, {}
    # (1) The hand fit. Pairs ([1,1] -> 0), ([0,1] -> 1): XtX + diag(1, 0) = [[2,1],[1,2]], whose
    # inverse is [[2,-1],[-1,2]] / 3, times [[1,0],[1,1]] gives W = [[1,-1],[1,2]] / 3; then x = [2, 1]
    # scores (1, 0) -> 0. Event 1 and the two events before the first solve answer None.
    rows = [([1.0], 0), ([0.0], 0), ([2.0], 1), ([0.0], 0)]
    fit = ReferenceReadout(every=2)
    predictions = _drive_readout(fit, rows)
    hand = np.array([[1.0, -1.0], [1.0, 2.0]]) / 3.0
    numbers["fit"] = bool(
        predictions == [None, None, None, 0]
        and fit.classes == (0, 1)
        and np.allclose(fit.w, hand, rtol=0.0, atol=1e-12)
    )
    penalized = _PenalizedBias(every=2)  # its W is [[2,-1],[1,2]] / 5
    _drive_readout(penalized, rows)
    red["bias"] = not np.allclose(penalized.w, hand, rtol=0.0, atol=1e-12)
    red["none_before_solve"] = _drive_readout(_GuessBeforeSolve(every=2), rows) != predictions
    # (2) The exact tie: phi = [0] throughout, pairs ([0,1] -> 0), ([0,1] -> 1): XtX + P = diag(1, 2)
    # and W = [[0, 0], [0.5, 0.5]] exactly, so both scores are 0.5 and the lowest id wins.
    rows = [([0.0], 0), ([0.0], 0), ([0.0], 1), ([0.0], 1)]
    numbers["tie"] = _drive_readout(ReferenceReadout(every=2), rows)[-1] == 0
    red["tie"] = _drive_readout(_LastMaximum(every=2), rows)[-1] == 1
    # (3) The last-solve category rule. a = [1,0], b = [0,1], c = [1,1]. The solve at pair 2 fixes W's
    # columns at (0, 1); pair 3 adds (c -> 2), so event 5 still answers from (0, 1). The seen-so-far
    # twin re-solves over (0, 1, 2) and scores c at (1/4, 1/4, 1/2), answering 2.
    a, b, c = [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    rows = [(a, 0), (b, 0), (c, 1), (c, 2), (c, 0)]
    predictions = _drive_readout(ReferenceReadout(every=2), rows)
    numbers["last_solve"] = predictions[:3] == [None] * 3 and 2 not in predictions
    red["last_solve"] = _drive_readout(_SeenSoFar(every=2), rows)[4] == 2
    # (4) A future-only category: 2 first appears at event 5, whose pair is the 4th (a solve). Before
    # its appearance it is never answered; the pair-first leak re-solves with (c -> 2) and answers 2
    # AT event 5 (scores 4/13, 3/13, 6/13 by hand); after it, event 6 answers 2 legitimately.
    rows = [(a, 0), (b, 0), (a, 1), (c, 0), (c, 2), (c, 0)]
    predictions = _drive_readout(ReferenceReadout(every=2), rows)
    numbers["future_only"] = 2 not in predictions[:5] and predictions[5] == 2
    red["future_only"] = _drive_readout(ReferenceReadout(every=2), rows, pair_first=True)[4] == 2
    return {
        "verdict": bool(all(numbers.values()) and all(red.values())),
        "numbers": numbers,
        "red": red,
    }


def check_r_b2():
    """R-b2 (section 2.7): the pool's view declares only x; a (forbidden) pool asking for c raises."""
    values = {
        "x": [0.0] * 64,
        "c": 3,
        "own_predictions": 1,
        "x_history": [[0.0] * 64] * 3,
        "c_history": [3, None, None, None],
    }
    view = _View(ARMS["pool"][1], values)
    declared = ARMS["pool"][1] == ("x",) and set(view.values) == {"x"}
    red = False
    try:
        view.get("c")
    except ValueError:
        red = True
    return {
        "verdict": bool(declared and red),
        "numbers": {"channels": sorted(view.channels)},
        "red": red,
    }


# ---- R-b8's constructed faults (section 2.7) ----------------------------------------------------


class _WritingReadout(ReferenceReadout):
    """R-b8's red: a reference readout that writes into the grower's input."""

    def observe(self, phi, category):
        super().observe(phi, category)
        phi[0] += 1.0  # the write section 2.2 forbids


@contextlib.contextmanager
def _writing_readout():
    """Every replay inside builds R-b8's writing readout in place of ReferenceReadout."""
    names = globals()
    original = names["ReferenceReadout"]
    names["ReferenceReadout"] = _WritingReadout
    try:
        yield
    finally:
        names["ReferenceReadout"] = original


# distinct-from: reservoir/recurrent_pool.py::_skip_carrier -- that rewires ONE pool's reset (R-b5 (i)); this is its replay-level carrier, the way _writing_readout is R-b8's: every pool a replay builds inside has the same skip (G1(c) 2.10's skip-a-carrier red of Settled invariance), by calling it on what make_pool returns
@contextlib.contextmanager
def _skipped_carrier(carrier):
    """Every pool a replay builds inside has a reset that leaves `carrier` as it was."""
    original = rp.make_pool

    def skipping(*args, **kwargs):
        return rp._skip_carrier(original(*args, **kwargs), carrier)

    rp.make_pool = skipping
    try:
        yield
    finally:
        rp.make_pool = original


@contextlib.contextmanager
def _e1_mutant():
    """R-b8's pre-build red: `_feature("e1")` with one element changed."""
    names = globals()
    original = names["_feature"]

    def mutant(arm, view, pool=None):
        row = original(arm, view, pool)
        if arm == "e1":
            row[-1] += 1.0
        return row

    names["_feature"] = mutant
    try:
        yield
    finally:
        names["_feature"] = original


# ---- G1(c) 2.10 "Transport": the constructed faults ---------------------------------------------
# Each is a context manager; pool_selection.FAULTS names them so a child process can run one.


@contextlib.contextmanager
def _tap_fault(name, **settings):
    saved = dict(_FAULT)
    _FAULT.update(tap=name, **settings)
    try:
        yield
    finally:
        _FAULT.clear()
        _FAULT.update(saved)


def _late_tap(by=LATE_TAP_EVENTS):
    # the tap taken `by` events after b (the late-tap red copies at b + 40)
    return _tap_fault("late_tap", late=by)


def _leaky_tap():
    # a tap that lists the grower's FINAL weights as its weights at b
    return _tap_fault("leaky_tap")


def _writing_tap():
    # a tap that adds 1e-3 to the grower's W[0, 0] as it reads (read-only is its contract)
    return _tap_fault("writing_tap")


@contextlib.contextmanager
def _w_live_mutant():
    # the frozen request red: pool-copy keeps its permitted declaration and REQUESTS w_live
    names = globals()
    original = names["_feature"]

    def mutant(arm, view, pool=None):
        if arm == "pool-copy":
            view.get("w_live")
        return original(arm, view, pool)

    names["_feature"] = mutant
    try:
        yield
    finally:
        names["_feature"] = original


# ---- G1(b): the pool arm's bookkeeping ----------------------------------------------------------


def _movement(before, after):
    """Parameter movement across the window: L1 and max |change| of each float parameter that moved."""
    moved = {}
    for key in sorted(before):
        old, new = before[key], after[key]
        if old.dtype.kind == "f" and not np.array_equal(old, new):
            change = np.abs(new - old).ravel()
            moved[key] = {"l1": float(np.cumsum(change)[-1]), "max": float(change.max())}
    return moved


def _degeneracy(rows):
    """Section 2.5's degeneracy statistics over the scored rows: the fraction of units constant (one
    value on every row) and the count of distinct phi rows."""
    if not rows:
        return {"rows": 0, "constant_fraction": None, "distinct_rows": 0}
    table = np.asarray(rows, dtype=np.float64)
    constant = np.all(table == table[0], axis=0)
    return {
        "rows": len(rows),
        "constant_fraction": float(np.count_nonzero(constant) / table.shape[1]),
        "distinct_rows": len(np.unique(table, axis=0)),
    }


class _PoolArm:
    """The pool arm's pool, recurrent_pool.make_pool(candidate, config, stream seed, form), and what
    its result carries: R-b7's binding, section 2.5's degeneracy rows, the parameter movement across
    the window, the Carried form's startup diagnostic, and R-b4's pool-noise kicks when asked."""

    KEYS = ("candidate", "config", "form", "noise", "window")

    def __init__(self, spec, seed, arm="pool", b=TRANSPORT_B):
        self.arm, self.b, self.seed = arm, b, seed
        unknown = sorted(set(spec) - set(self.KEYS))
        if unknown:
            raise ValueError(f"REFUSED: unknown pool spec keys {unknown}")
        self.candidate, self.form = spec["candidate"], spec["form"]
        self.config = rp.canonical(self.candidate, spec["config"])
        self.noise = bool(spec.get("noise", False))
        # the window is shortened ONLY by the self-checks' short streams (disclosed) and set to 0 by
        # G1(c)'s Settled@0 arm, which files it as window 0; a campaign table refuses any binding
        # whose window is not the one it is filed under (R-b7)
        window = int(spec.get("window", rp.WINDOW))
        self.pool = rp.make_pool(self.candidate, self.config, seed, self.form, window=window)
        self.start = {key: value.copy() for key, value in self.pool.params().items()}
        # window 0 has no window to move across: nothing can move, so the field is the empty
        # movement a window leaves g1, which keeps it comparable with window 256's
        self.movement = {} if window == 0 else None
        self.rows = []
        self.kicks = 0
        self.chain = 0  # R-b4's per-percept state chain (amended 2026-09-25, section 2.7)
        self.seconds = 0.0
        self.binding = {
            "candidate": self.candidate,
            "form": self.form,
            "config": self.config,
            "config_crc32": rp.crc_json(self.config),
            "realization_seed": rp.realization_seed(self.candidate, seed),
            "window": self.pool.window,
            "s_settle": self.pool.s_settle,
            "noise": self.noise,
            "w_in": _crc(self.pool.w_in),
            "topology": self.pool.params_fingerprint(),
            "initial": self.pool.fingerprint(),
        }
        if arm != "pool":
            # G1(c) 2.7: binding["w_in"] stays the construction draw's crc; "encoder" is the crc of
            # the matrix after the single swap (None before it, or without one). A plain pool arm
            # carries neither this key nor "transport", so its result is G1(b)'s.
            self.binding["encoder"] = None
            self.random = self.pool.w_in.copy()
            self.silent = self.events = self.widest = 0
            self.record = {
                "pool-copy": {
                    "copied_at_event": None,
                    "L": None,
                    "padding_fraction": None,
                    "raw_row_norms": None,
                    "mean_row_norm": None,
                    "encoder_crc": None,
                },
                "pool-swap": {"swapped_at_event": None, "mean_row_norm": None, "encoder_crc": None},
                "pool-act": {},
            }[arm]

    # Sites the duplicate-capability hook flagged, read; neither changes a pool's input matrix:
    # distinct-from: gates/amdahl_gate.py::_sham_propagate -- a sham stand-in installed on Connection.propagate
    # distinct-from: sensorimotor/wavefront_planner.py::collides -- reports whether the arm is in contact
    def _install(self, matrix, **record):
        self.pool.replace_encoder(matrix)
        self.binding["encoder"] = _crc(self.pool.w_in)
        self.record.update(
            record, mean_row_norm=rp.mean_row_norm(self.random), encoder_crc=self.binding["encoder"]
        )

    def _arm_moves(self, view):
        # copy: once, when the view first holds w@b; swap: once, at b; activity: count the silent events
        pool = self.pool
        if self.arm == "pool-copy":
            block = view.get("w@b")
            if block is not None and self.binding["encoder"] is None:
                block = np.asarray(block, dtype=np.float64)
                units = block.shape[0]
                self._install(
                    rp.transported_encoder(self.random, block),
                    copied_at_event=pool.count,
                    L=units,
                    padding_fraction=(rp.N - units) / rp.N,
                    raw_row_norms=rp.row_norms(block).tolist(),
                )
        elif self.arm == "pool-swap":
            if self.binding["encoder"] is None and pool.count >= self.b:
                self._install(rp.fresh_encoder(self.seed, self.random), swapped_at_event=pool.count)
        else:
            activity = view.get("grower_activity")
            self.events += 1
            if activity is None:
                self.silent += 1
            else:
                self.widest = max(self.widest, len(activity))

    def features(self, view):
        pool = self.pool
        if self.noise and pool.count >= pool.window:
            # R-b4's red (pool-noise): a kick before every percept after the window
            pool.perturb()
            self.kicks += 1
        if self.arm != "pool":
            self._arm_moves(view)
        started = time.perf_counter()
        features = _feature(self.arm, view, pool)
        if pool.count > pool.window:
            self.seconds += time.perf_counter() - started
        self.chain = zlib.crc32(pool.fingerprint().to_bytes(4, "little"), self.chain)
        if pool.count == pool.window and self.form != "front_end":
            self.movement = _movement(self.start, pool.params())
        return features

    def finish(self):
        pool = self.pool
        after = pool.count - pool.window
        startup = None
        if self.form == "carried":
            diagnostic = rp.startup_diagnostic(pool)
            startup = {
                "transient": {str(s): v for s, v in diagnostic["transient"].items()},
                "flagged": diagnostic["flagged"],
            }
        result = {
            **self.binding,
            "boundary": pool.boundary,
            "frozen": pool.frozen(),
            "state": pool.fingerprint(),
            "state_chain": self.chain,
            "lesion": pool.lesion_record,
            "percepts": pool.count,
            "kicks": self.kicks,
            "movement": self.movement,
            "degeneracy": _degeneracy(self.rows),
            "startup": startup,
            "pool_ms": 1000.0 * self.seconds / after if after > 0 else None,
        }
        if self.arm != "pool":
            record = {"form": TRANSPORT_FORM[self.arm], **self.record}
            if self.arm == "pool-act":
                record.update(silent_events=self.silent, events=self.events, widest=self.widest)
            result["transport"] = record
        return result


# ---- G1(c) 2.7: the read-only tap and what a transport arm may be handed from it ----------------


def _corrupt_w(model):
    # the writing-tap red's write (GRADIENT_CLAIM above declares it): a constant added to W[0, 0]
    with torch.no_grad():
        model.W[0, 0] += 1e-3


def _grower_w(engine):
    # the incumbent's W (L x 64) as a list; None while it holds no model
    model = engine._model
    if model is None or model.W is None:
        return None
    return model.W.detach().cpu().tolist()


# Reads the incumbent beside an arm-(a) replay and writes nothing (the writing-tap fault is the red
# of that). Three readings: `at`, the weights at the START of delivery b + 1 (before predict_next);
# one activity row per delivery, the hidden rates for the CURRENT percept (after predict_next,
# before observe); and the weights at every completed refit.
class _Tap:
    def __init__(self, b):
        self.b = b
        self.at = None
        self.activity = []
        self.weights = []

    @property
    def due(self):
        return self.b + (_FAULT["late"] if _FAULT["tap"] == "late_tap" else 0)

    def at_delivery(self, engine, event):
        if self.at is not None or event != self.due:
            return
        model = engine._model
        if model is None or model.W is None or int(model.W.shape[0]) == 0:
            raise ValueError(
                "REFUSED: the grower holds no model at the tap (never padded from nothing)"
            )
        self.at = {
            "event": event,
            "observed": engine.observed,
            "W": _grower_w(engine),
            "crc_W": _crc(model.W),
        }
        if _FAULT["tap"] == "writing_tap":
            _corrupt_w(model)

    def rates(self, engine, x):
        model = engine._model
        if model is None or model.W is None:
            self.activity.append(None)
            return
        row = torch.tensor([x], dtype=torch.float32)
        self.activity.append(model._rates(row, model.W, model.b)[0].tolist())

    def note(self, engine, event):
        self.weights.append({"event": event, "W": _grower_w(engine)})

    def close(self, engine, identity):
        if self.at is None:
            raise ValueError(
                f"REFUSED: the stream has no delivery {self.due + 1} to take the tap at"
            )
        if _FAULT["tap"] == "leaky_tap":
            # the leak: the weights the grower ends the whole replay with, not the weights at b
            self.at = dict(self.at, W=_grower_w(engine), crc_W=_crc(engine._model.W))
        return {
            "b": self.b,
            **identity,
            "w_at_b": self.at,
            "activity": self.activity,
            "refit_weights": self.weights,
        }


# The three values a transport arm's loop supplies from a tap: w@b (the tap's W once the delivery
# count reaches its event, else None), this delivery's activity, and w_live (the latest refit's W),
# which no legitimate arm declares.
class _TapSupply:
    def __init__(self, tap):
        self.tap = tap
        self.cursor = 0
        self.live = None

    def at_event(self, event):
        weights = self.tap["refit_weights"]
        while self.cursor < len(weights) and weights[self.cursor]["event"] <= event:
            self.live = weights[self.cursor]["W"]
            self.cursor += 1
        at = self.tap["w_at_b"]
        return {
            "w@b": at["W"] if event >= at["event"] else None,
            "grower_activity": self.tap["activity"][event],
            "w_live": self.live,
        }


_TAP_FIELDS = (
    "b",
    "source_digest",
    "config_digest",
    "seed",
    "revision",
    "mode",
    "w_at_b",
    "activity",
    "refit_weights",
)


def _belongs(side, identity, count):
    # a side tap belongs to the replay it feeds: that replay's digests, seed, revision and MODE (a live
    # tap read the recorded live |LP| and a causal one the causal |LP|: the two taps differ in their
    # weights, rates and refits, and either must feed only a replay of its own mode), and one
    # activity slot per delivery
    if not isinstance(side, dict) or any(key not in side for key in _TAP_FIELDS):
        raise ValueError("REFUSED: side tap lacks a tap's fields")
    for key, expected in identity.items():
        if side[key] != expected:
            raise ValueError(f"REFUSED: side tap {key} does not match this replay")
    if len(side["activity"]) != count:
        raise ValueError("REFUSED: side tap activity does not hold one slot per delivery")


# distinct-from: faculties/evidence_accumulation.py -- `_measure_patience_charge` replays 5-HT frames
#   twice through instruments/port_charge_probe.replay_twice for a port-charge A/B; this is G1's recorded
#   event schedule replayed into the grower (site read). Extended in place for G1(b), not duplicated.
def replay(
    stream, arm, mode="causal", *, mutate=None, readout=True, pool=None, tap=None, side=None
):
    """Replay the recorded event schedule, rejecting missing gate evidence.

    G1(b) (g1b-pool-selection.md section 2): the reference readout runs beside the grower for every
    arm and predicts c_t from phi_{t-1} where the grower predicts (`readout=False` switches it off,
    for R-b8's second half); `arm="pool"` takes a spec {"candidate", "config", "form"} (plus "noise",
    R-b4's red, and "window": the self-checks' short streams, and 0 for G1(c)'s Settled@0) and feeds x alone through its view
    into recurrent_pool.make_pool(candidate, config, stream["seed"], form); the whole replay runs at
    one BLAS thread, recorded in the result's policy.

    G1(c) 2.7: `tap=b` (arm "a" only) records, read-only, the incumbent grower's weights at the START
    of delivery b + 1, its hidden rates at every delivery and its weights at every refit, returned as
    result["tap"]. `side=` is such a tap, handed to a transport arm ("pool-copy", "pool-act",
    "pool-swap"; the copy and the activity arms require it); it must belong to this replay: its source
    and config digests, seed, revision and MODE (a live tap never feeds a causal replay)."""
    if arm not in ARMS or mode not in ("causal", "live"):
        raise ValueError("unknown arm or mode")
    if (arm in POOL_ARMS) != (pool is not None):
        raise ValueError("REFUSED: the pool arms take a pool spec, and only they do")
    if tap is not None:
        if arm != "a":
            raise ValueError("REFUSED: the tap reads the incumbent grower: arm a only")
        if isinstance(tap, bool) or not isinstance(tap, int) or tap < 1:
            raise ValueError(f"REFUSED: the tap's boundary is a positive event count, not {tap!r}")
    if side is not None and arm not in TRANSPORT_ARMS:
        raise ValueError("REFUSED: a side tap feeds only the transport arms")
    if side is None and arm in ("pool-copy", "pool-act"):
        raise ValueError(f"REFUSED: arm {arm} needs a side tap and never runs on nothing")
    torch.set_num_threads(1)
    if stream.get("policy", {}).get("threads") != 1:
        raise ValueError("REFUSED: recording policy")
    if stream["policy"]["NORD_DEVICE"] != os.environ.get("NORD_DEVICE") or stream["policy"][
        "NORD_GPU"
    ] != os.environ.get("NORD_GPU"):
        raise ValueError("REFUSED: device environment changed")
    if any(
        stream["policy"][key] != _policy()[key]
        for key in ("torch", "numpy", "python", "platform", "threads", "device")
    ):
        raise ValueError("REFUSED: library policy changed")
    with rp._blas_threads(1):  # section 2.2: the readout's products and solves at one BLAS thread
        return _replay(stream, arm, mode, mutate, readout, pool, tap, side)


def _replay(stream, arm, mode, mutate, readout, pool, tap=None, side=None):
    started = time.perf_counter()
    rows = mutate if mutate is not None else stream
    source_digest = _source_digest(rows)
    recorder = _Tap(tap) if tap is not None else None
    supply = None
    if side is not None:
        identity = {
            "source_digest": source_digest,
            "config_digest": _config_digest(stream),
            "seed": stream.get("seed"),
            "revision": dict(zip(("head", "dirty"), _revision(), strict=True)),
            "mode": mode,
        }
        _belongs(side, identity, len(rows["events"]))
        supply = _TapSupply(side)
    ledger = []
    state = {"tick": 0, "event": 0, "value": None, "exists": False}

    class Reader:
        def lp(self, region):
            ledger.append(
                {
                    "tick": state["tick"],
                    "event": state["event"],
                    "mode": mode,
                    "value": state["value"],
                    "exists": state["exists"],
                }
            )
            if not state["exists"]:
                raise KeyError(region)
            if state["value"] is None:
                raise KeyError(region)
            return state["value"]

    stop = GrowthStop(lp_reader=None if rows.get("_no_lp_reader") else Reader())
    verdicts = []

    def gate(trajectory):
        verdict, refused = feed_and_decide(stop, trajectory)
        if refused:
            raise ValueError("REFUSED: poisoned trajectory")
        verdicts.append({"tick": engine.ticks, "event": state["event"], "verdict": verdict})
        return verdict

    engine = NeurogenesisSelfImprove(
        **stream["config"], device="cpu", growth_gate=gate, online_readout="delta"
    )
    guide = ReferenceReadout() if readout else None
    unit = (
        _PoolArm(pool, stream["seed"], arm, side["b"] if side is not None else TRANSPORT_B)
        if pool is not None
        else None
    )
    predictions, fallbacks, phi, refits = [], [], [], []
    guesses = []
    history_x, history_c = [], []
    events = rows["events"]
    sleep = rows["sleep"]
    event_i = sleep_i = 0
    if any("brain_tick" not in s for s in sleep):
        raise ValueError("REFUSED: sleep timeline absent")
    timeline = sorted(
        [(e["tick"], 0, e) for e in events] + [(s["brain_tick"], 1, s) for s in sleep]
    )
    for _global_tick, kind, item in timeline:
        if kind == 0:
            if recorder is not None:
                recorder.at_delivery(engine, state["event"])  # the START of this delivery
            prediction = engine.predict_next()
            fallback = bool(engine._pending_prediction[4]) if engine._pending_prediction else False
            # the readout predicts c_t from phi_{t-1} at the point the grower does
            guess = guide.predict() if guide is not None else None
            if arm == "leak":
                prediction = item["c"]
                fallback = False
                guess = item["c"] if guide is not None else None  # R-b1's (leak), for both learners
            predictions.append(prediction)
            fallbacks.append(fallback)
            guesses.append(guess)
            x, c = item["x"], item["c"]
            if recorder is not None:
                recorder.rates(engine, x)  # after predict_next, before observe
            values = {
                "x": x,
                "c": c,
                "own_predictions": prediction,
                "x_history": [
                    history_x[-i] if len(history_x) >= i else [0.0] * 64 for i in range(1, 4)
                ],
                "c_history": [c]
                + [history_c[-i] if len(history_c) >= i else None for i in range(1, 4)],
            }
            if supply is not None:
                # G1(c): always supplied (None when not yet available), the forbidden w_live among them
                values.update(supply.at_event(state["event"]))
            view = _View(ARMS[arm][1], values)
            features = unit.features(view) if unit is not None else _feature(arm, view)
            if len(features) != ARMS[arm][0]:
                raise ValueError("REFUSED: feature width")
            if guide is not None:
                # the pair (phi_{t-1}, c_t), then phi_t held; before the grower, so a readout that
                # wrote into its input would reach it (R-b8's red), and the correct one copies
                guide.observe(features, c)
            phi.append(_crc(np.asarray(features, dtype=np.float32)))
            if unit is not None and item.get("segment") in SCORED_SEGMENTS:
                unit.rows.append(features)
            engine.observe(features, c)
            history_x.append(x)
            history_c.append(c)
            state["event"] += 1
            event_i += 1
        else:
            state["tick"] = engine.ticks + 1
            field = "live_lp" if mode == "live" else "causal_lp"
            source = (
                item if mode == "live" else events[state["event"] - 1] if state["event"] else None
            )
            if mode == "causal" and source is None:
                # before the first delivery nothing has been acquired, so the facade is unfed: a recorded
                # None (section 2.1's CAUSAL value at "the delivery of x_0"), not a missing record
                state["exists"], state["value"] = True, None
            else:
                state["exists"] = source is not None and field in source
                state["value"] = source.get(field) if source is not None else None
            before = len(engine.trajectory)
            engine.tick()
            if len(engine.trajectory) > before:
                refits.append(
                    {
                        "tick": engine.ticks,
                        "event": state["event"],
                        "fingerprint": _fingerprint(engine),
                    }
                )
                if recorder is not None:
                    recorder.note(engine, state["event"])
            sleep_i += 1
    if event_i != len(events) or sleep_i != len(sleep):
        raise ValueError("REFUSED: event count")
    if any(not entry["exists"] for entry in ledger) or len(ledger) != stop.observed("cortex"):
        raise ValueError("REFUSED: missing or disconnected LP ledger")
    result = {
        "arm": arm,
        "mode": mode,
        "source_digest": source_digest,
        "config_digest": _config_digest(stream),
        "revision": dict(zip(("head", "dirty"), _revision(), strict=True)),
        "execution_id": f"{os.getpid()}-{time.monotonic_ns()}-{next(_EXECUTIONS)}",
        "predictions": predictions,
        "fallbacks": fallbacks,
        "phi": phi,
        "refits": refits,
        "fingerprint": _fingerprint(engine),
        "live_n": engine.live_n,
        "live_correct": engine.live_correct,
        "live_persist": engine.live_persist,
        "ticks": engine.ticks,
        "observed": engine.observed,
        "trajectory": engine.trajectory,
        "baselines": engine.baselines,
        "gate_verdicts": verdicts,
        "ledger": ledger,
        # G1(b), R-b7's binding: the recording's seed, the numeric policy read inside the one-thread
        # pin, the readout beside the grower, and the pool's binding (None for G1(a)'s arms)
        "seed": stream.get("seed"),
        "policy": _policy(),
        "readout": None,
        "pool": None if unit is None else unit.finish(),
    }
    if guide is not None:
        result["readout"] = {"predictions": guesses, "fingerprint": guide.fingerprint()}
    if recorder is not None:
        # the identity a side tap is held to when it feeds a transport arm
        result["tap"] = recorder.close(
            engine,
            {
                "source_digest": source_digest,
                "config_digest": result["config_digest"],
                "seed": result["seed"],
                "revision": result["revision"],
                "mode": mode,
            },
        )
    result["seconds"] = time.perf_counter() - started  # end-to-end, reported only
    return result


def _entropy(values):
    if not values:
        return None
    counts = Counter(values)
    return -sum((n / len(values)) * math.log2(n / len(values)) for n in counts.values())


# G1(a)'s scorer, extended in place with `floors`; none of the sites below scores a replay's
# predictions against a recorded category stream (all read):
# distinct-from: sensorimotor/ingest_guard.py -- license_ok refuses a non-permissive licence id
# distinct-from: growth/cortical_prior.py -- validate lists reference data missing an attribute
# distinct-from: faculties/counterfactual_sim.py -- counterfactual scores imagined rollouts by value_fn
# distinct-from: instruments/prior_drift_harness.py -- arm_wrong builds a band-permuted target table
# distinct-from: faculties/wholistic_loop.py -- _perceive degrades an input by substrate fidelity
def score(stream, predictions, fallbacks, *, floors=True):
    """Per segment: accuracy (None is a miss), coverage, entropies, the prior-fallback rate and the
    stream's own floors. `floors=False` (G1(b)'s campaign scores thousands of replays of one
    recording) omits the recording-level floors -- truth entropy, persistence, causal majority and
    transitions -- whose causal-majority loop is quadratic in the stream; every other value is the
    same computation."""
    if len(predictions) != len(stream["events"]) or len(fallbacks) != len(predictions):
        raise ValueError("prediction count differs from stream")
    output = {}
    for segment in ("warm-up", "repeated-canonical", "unseen-order", "repeated-permuted"):
        indices = [i for i, e in enumerate(stream["events"]) if e.get("segment") == segment]
        if segment == "warm-up":
            output[segment] = {"n": len(indices), "available": False}
            continue
        if not indices:
            output[segment] = {"n": 0, "available": False}
            continue
        truth = [stream["events"][i]["c"] for i in indices]
        pred = [predictions[i] for i in indices]
        if not floors:
            n = len(indices)
            output[segment] = {
                "n": n,
                "available": True,
                "accuracy": sum(p == c for p, c in zip(pred, truth, strict=True)) / n,
                "coverage": sum(p is not None for p in pred) / n,
                "prediction_entropy": _entropy([p for p in pred if p is not None]),
                "prior_fallback": sum(fallbacks[i] for i in indices) / n,
            }
            continue
        floor = [stream["events"][i - 1]["c"] if i else None for i in indices]
        majority = []
        for i in indices:
            seen = [e["c"] for e in stream["events"][:i]]
            majority.append(min(seen, key=lambda c: (-seen.count(c), c)) if seen else None)
        n = len(indices)
        output[segment] = {
            "n": n,
            "available": True,
            "accuracy": sum(p == c for p, c in zip(pred, truth, strict=True)) / n,
            "coverage": sum(p is not None for p in pred) / n,
            "prediction_entropy": _entropy([p for p in pred if p is not None]),
            "truth_entropy": _entropy(truth),
            "prior_fallback": sum(fallbacks[i] for i in indices) / n,
            "persistence": sum(p == c for p, c in zip(floor, truth, strict=True)) / n,
            "causal_majority": sum(p == c for p, c in zip(majority, truth, strict=True)) / n,
            "transitions": sum(p != c for p, c in zip(floor, truth, strict=True)),
        }
    return output


def mutate_suffix(stream, t_star, seed):
    changed = copy.deepcopy(stream)
    rng = np.random.default_rng(seed)
    categories = sorted({e["c"] for e in stream["events"]})
    if len(categories) < 2:
        raise ValueError("suffix mutation requires two categories")
    for index, event in enumerate(changed["events"], 1):
        if index < t_star:
            continue
        event["x"] = rng.random(len(event["x"])).tolist()
        event["c"] = (
            next(c for c in categories if c != event["c"])
            if index == t_star
            else int(rng.choice(categories))
        )
        event["causal_lp"] = float(rng.random())
    return changed


def _lp_sensitive_red(arm="a", mode="live"):
    """A single recorded LP change must alter replay's gate and learner."""
    if mode not in ("live", "causal"):
        raise ValueError("unknown replay mode")
    stream = _synthetic_stream()
    stream["events"] = [
        dict(event, t=i + 1, tick=i * 2 + 1) for i, event in enumerate(stream["events"] * 3)
    ]
    stream["sleep"] = [
        dict(row, tick=i + 1, brain_tick=i * 2 + 2, event=i + 1)
        for i, row in enumerate(stream["sleep"] * 3)
    ]
    target = 92 if arm == "f64" else 80
    rows = stream["sleep"] if mode == "live" else stream["events"]
    field = "live_lp" if mode == "live" else "causal_lp"
    for event_index in (target - 8, target - 4):
        rows[event_index - 1][field] = 10.0
    base = replay(stream, arm, mode)
    changed = copy.deepcopy(stream)
    changed_rows = changed["sleep"] if mode == "live" else changed["events"]
    changed_rows[target - 1][field] = 10.0
    altered = replay(changed, arm, mode)
    return any(
        a["verdict"] != b["verdict"]
        for a, b in zip(base["gate_verdicts"], altered["gate_verdicts"], strict=True)
    ) and (
        base["fingerprint"] != altered["fingerprint"]
        or base["predictions"] != altered["predictions"]
    )


def _validate_run(run, stream, source, arm, mode):
    if (
        run.get("arm") != arm
        or run.get("mode") != mode
        or run.get("source_digest") != _source_digest(source)
        or any(entry.get("mode") != mode for entry in run.get("ledger", ()))
    ):
        raise ValueError(f"REFUSED: replay identity {arm} {mode}")
    final = stream.get("final", {})
    if (
        run.get("config_digest") != _config_digest(stream)
        or run.get("revision") != {"head": final.get("head"), "dirty": final.get("dirty")}
        or not isinstance(run.get("execution_id"), str)
        or not run["execution_id"]
    ):
        raise ValueError(f"REFUSED: replay provenance {arm} {mode}")
    count = len(source["events"])
    if any(len(run.get(key, ())) != count for key in ("predictions", "fallbacks", "phi")):
        raise ValueError(f"REFUSED: incomplete replay events {arm} {mode}")
    ledger_count = len(run.get("ledger", ()))
    trajectory_count = len(run.get("trajectory", ()))
    if (
        ledger_count != len(run.get("gate_verdicts", ()))
        or ledger_count != max(0, trajectory_count - 1)
        or (run.get("refits") and not ledger_count)
    ):
        raise ValueError(f"REFUSED: incomplete LP ledger {arm} {mode}")


def _refit_positions(rows, *, recorded):
    if recorded:
        completed = [(r["tick"], r["event"], r["refit"]) for r in rows if r["refit"]]
        skipped = [(r["tick"], r["event"]) for r in rows if not r["refit"]]
    else:
        completed = [(r["tick"], r["event"], r["fingerprint"]) for r in rows]
        skipped = []
    return completed, skipped


def _replay_skipped(stream, run):
    completed = {(r["tick"], r["event"]) for r in run["refits"]}
    candidates = [
        (r["tick"], r["event"])
        for r in stream["sleep"]
        if r["tick"] % stream["config"]["refit_every"] == 0
    ]
    return [position for position in candidates if position not in completed]


def check_r_a1(stream, live=None, arm="a"):
    live = live if live is not None else replay(stream, arm, "live")
    _validate_run(live, stream, stream, arm, "live")
    expected = stream["final"]
    eligible = bool(
        any(
            e.get("segment") in ("repeated-canonical", "unseen-order", "repeated-permuted")
            for e in stream["events"]
        )
        and live["refits"]
    )
    ok = live["predictions"] == [e["prediction"] for e in stream["events"]]
    ok &= live["gate_verdicts"] == expected["gate_verdicts"]
    ok &= json.dumps(live["trajectory"]) == json.dumps(expected["trajectory"])
    ok &= all(
        live[k] == expected[k]
        for k in ("live_n", "live_correct", "live_persist", "ticks", "observed")
    )
    ok &= live["fingerprint"] == expected["fingerprint"]
    expected_refits, expected_skips = _refit_positions(stream["refits"], recorded=True)
    actual_refits, _ = _refit_positions(live["refits"], recorded=False)
    ok &= actual_refits == expected_refits
    ok &= _replay_skipped(stream, live) == expected_skips
    changed = copy.deepcopy(stream)
    first = live["refits"][0]["event"] if live["refits"] else 0
    cats = sorted({e["c"] for e in changed["events"]})
    count_red = False
    if first > 1 and len(cats) > 1:
        row = changed["events"][1]
        row["c"] = next(c for c in cats if c != row["c"])
        altered = replay(changed, arm, "live")
        count_red = (
            altered["fingerprint"] != live["fingerprint"]
            or altered["predictions"] != live["predictions"]
        )
    lp_red = _lp_sensitive_red(arm)
    red = bool(count_red and lp_red)
    numbers = {
        "events": len(live["predictions"]),
        "refits": len(live["refits"]),
        "equivalent": bool(ok),
        "eligible": eligible,
    }
    return {
        "verdict": bool(ok and red and eligible),
        "numbers": numbers,
        "red": {"count_preserving": bool(count_red), "lp_sensitive": bool(lp_red)},
    }


def check_r_a1b(stream, oracle, replayed=None):
    replayed = replayed if replayed is not None else replay(oracle, "f64", "live")
    _validate_run(replayed, oracle, oracle, "f64", "live")
    fields = ("x", "c", "tick", "causal_lp", "source", "attended")
    same = len(stream["events"]) == len(oracle["events"]) and all(
        all(a[key] == b[key] for key in fields)
        for a, b in zip(stream["events"], oracle["events"], strict=True)
    )
    same &= stream["acquisitions"] == oracle["acquisitions"]
    same &= [(r["tick"], r["brain_tick"], r["live_lp"]) for r in stream["sleep"]] == [
        (r["tick"], r["brain_tick"], r["live_lp"]) for r in oracle["sleep"]
    ]
    same &= replayed["predictions"] == [e["prediction"] for e in oracle["events"]]
    expected_refits, expected_skips = _refit_positions(oracle["refits"], recorded=True)
    actual_refits, _ = _refit_positions(replayed["refits"], recorded=False)
    same &= actual_refits == expected_refits
    same &= _replay_skipped(oracle, replayed) == expected_skips
    red = all(check_r_a1(oracle, arm="f64")["red"].values())
    return {"verdict": bool(same and red), "numbers": {"events": len(oracle["events"])}, "red": red}


def check_r_a2(stream, runs=None):
    results = {}
    for arm in SCORED_ARMS:
        base = _load(runs[arm]["base"]) if runs else replay(stream, arm)
        _validate_run(base, stream, stream, arm, "causal")
        cuts = []
        for cut in (
            len(stream["events"]) // 4,
            len(stream["events"]) // 2,
            3 * len(stream["events"]) // 4,
        ):
            changed = mutate_suffix(stream, max(1, cut), stream["seed"] + cut)
            other = _load(runs[arm][cut]) if runs else replay(changed, arm, mutate=changed)
            _validate_run(other, stream, changed, arm, "causal")
            cuts.append(base["predictions"][:cut] == other["predictions"][:cut])
        results[arm] = cuts
    leak = replay(stream, "leak")
    red = True
    for cut in (
        len(stream["events"]) // 4,
        len(stream["events"]) // 2,
        3 * len(stream["events"]) // 4,
    ):
        changed = mutate_suffix(stream, max(1, cut), stream["seed"] + cut)
        altered = replay(changed, "leak", mutate=changed)
        red &= leak["predictions"][:cut] != altered["predictions"][:cut]
    return {
        "verdict": all(all(v) for v in results.values()) and red,
        "numbers": results,
        "red": red,
    }


def check_r_a2b(stream):
    red = False
    try:
        _feature("forbidden", _View(ARMS["forbidden"][1], {"x": [0.0] * 64, "c": 0}))
    except ValueError:
        red = True
    return {"verdict": red, "numbers": {"arms": len(SCORED_ARMS)}, "red": red}


def check_r_a3(stream):
    cats = [e["c"] for e in stream["events"]]
    persistence = [None, *cats[:-1]]
    known = score(stream, persistence, [False] * len(cats))
    oracle = score(stream, cats, [False] * len(cats))
    constant = [cats[0]] * len(cats)
    constant_scores = score(stream, constant, [False] * len(cats))
    valid = True
    discrimination = {}
    for segment, row in known.items():
        if segment == "warm-up":
            continue
        if not row["available"]:
            discrimination[segment] = {"transitions": 0, "discriminating": False}
            continue
        indices = [i for i, e in enumerate(stream["events"]) if e.get("segment") == segment]
        numerator = sum(i > 0 and cats[i] == cats[i - 1] for i in indices)
        expected = numerator / len(indices)
        transitions = len(indices) - numerator
        discrimination[segment] = {
            "transitions": transitions,
            "discriminating": transitions > 0,
        }
        valid &= (
            row["accuracy"] == expected
            and row["persistence"] == expected
            and oracle[segment]["accuracy"] == 1
        )
        valid &= constant_scores[segment]["accuracy"] == sum(
            cats[i] == cats[0] for i in indices
        ) / len(indices)
    fixture = {"events": [{"c": c, "segment": "unseen-order"} for c in (0, 0, 1, 1, 0, 1)]}
    truth = [event["c"] for event in fixture["events"]]
    correct = score(fixture, truth, [False] * len(truth))
    shifted = score(fixture, [None, *truth[:-1]], [False] * len(truth))
    red = correct["unseen-order"]["accuracy"] == 1 and shifted["unseen-order"]["accuracy"] < 1
    return {
        "verdict": bool(valid and red),
        "numbers": known,
        "discrimination": discrimination,
        "red": red,
    }


def check_r_a4(stream, runs=None):
    equal = {}
    for arm in SCORED_ARMS:
        a, b = (
            tuple(_load(path) for path in runs[arm])
            if runs
            else (replay(stream, arm), replay(stream, arm))
        )
        _validate_run(a, stream, stream, arm, "causal")
        _validate_run(b, stream, stream, arm, "causal")
        if a["execution_id"] == b["execution_id"]:
            raise ValueError(f"REFUSED: duplicate replay execution {arm}")
        equal[arm] = all(
            a[key] == b[key] for key in ("phi", "refits", "predictions", "fingerprint")
        )
    red = replay(stream, "noise")["phi"] != replay(stream, "noise")["phi"]
    return {"verdict": all(equal.values()) and red, "numbers": equal, "red": red}


# ---- G1(c) 2.10 "Transport": the checks, each beside its red ------------------------------------


def _declaration_problems(arm, declared):
    # 2.7's declaration-validation check: what a transport arm declares beyond, or lacks of, its frozen set
    expected = TRANSPORT_CHANNELS.get(arm)
    if expected is None:
        return ["not a transport arm"]
    declared = tuple(declared)
    return [f"missing {c}" for c in expected if c not in declared] + [
        c for c in declared if c not in expected
    ]


def _raises_forbidden(call):
    try:
        call()
    except ValueError as error:
        return "forbidden feature channel" in str(error)
    return False


# 2.7 and 2.10: the three transport arms declare exactly TRANSPORT_CHANNELS; a view over values that
# also hold w_live, c and pairs holds only the declared keys and raises on any other.
# RED 1 (the frozen request red): pool-copy keeps its permitted declaration and REQUESTS w_live while
# it is present, which must raise (through a real replay too when a stream and a pool spec are
# given). RED 2: an arm that DECLARES w_live fails the declaration check; pool-act's view cannot
# read x.
def check_transport_channels(stream=None, spec=None, *, b=TRANSPORT_B):
    declared = {arm: ARMS[arm][1] for arm in TRANSPORT_ARMS}
    values = {
        "x": [0.0] * 64,
        "c": 1,
        "pairs": 3,
        "w@b": [[1.0] * 64],
        "grower_activity": [0.1],
        "w_live": [[2.0] * 64],
    }
    views = {arm: _View(channels, values) for arm, channels in declared.items()}
    problems = {arm: _declaration_problems(arm, channels) for arm, channels in declared.items()}
    only = all(set(view.values) == set(declared[arm]) for arm, view in views.items())
    shut = all(
        _raises_forbidden(lambda view=view, key=key: view.get(key))
        for view in views.values()
        for key in ("w_live", "c", "pairs")
    )
    pool = rp.make_pool("lsm", rp.R_B5_PERSISTENCE_WITNESS, rp.DEV_SEED)
    request = _View(ARMS["pool-copy"][1], values)
    control = len(_feature("pool-copy", request, pool)) == rp.N  # the permitted call runs
    with _w_live_mutant():
        asked = _raises_forbidden(lambda: _feature("pool-copy", request, pool))
    red = {
        "request_w_live": asked,
        "declares_w_live": _declaration_problems("pool-copy", (*declared["pool-copy"], "w_live"))
        != [],
        "pool_act_reads_x": _raises_forbidden(lambda: views["pool-act"].get("x")),
    }
    if stream is not None:
        side = replay(stream, "a", tap=b)["tap"]
        replay(stream, "pool-copy", pool=spec, side=side)  # as declared, it runs
        with _w_live_mutant():
            red["request_w_live_in_replay"] = _raises_forbidden(
                lambda: replay(stream, "pool-copy", pool=spec, side=side)
            )
    return {
        "verdict": bool(
            not any(problems.values()) and only and shut and control and all(red.values())
        ),
        "numbers": {
            "declared": {arm: list(channels) for arm, channels in declared.items()},
            "views": {arm: sorted(view.values) for arm, view in views.items()},
        },
        "red": red,
    }


def _truncated(stream, b):
    # the recording cut to its first b deliveries and the sleep ticks before delivery b + 1
    events = stream["events"]
    if len(events) <= b:
        raise ValueError(f"REFUSED: the stream has no delivery {b + 1}")
    horizon = events[b]["tick"]
    return {
        **stream,
        "events": events[:b],
        "sleep": [s for s in stream["sleep"] if s["brain_tick"] < horizon],
    }


def _last_w_crc(run, event):
    return [r for r in run["refits"] if r["event"] <= event][-1]["fingerprint"]["W"]


# 2.10: the tap's timing by two instruments that must agree. (i) Its crc_W is the W crc of the last
# completed refit at or before b in the same replay. (ii) An arm-(a) replay of the stream cut at b
# (its first b events and the sleep before delivery b + 1) ends at the same position (b events
# observed) with the same W crc. RED: a tap taken `late` events after b fails (ii) always, by its
# position, and (i) exactly when the refits between b and the late tap moved W's crc.
def check_transport_timing(stream, *, b=TRANSPORT_B, late=LATE_TAP_EVENTS):
    base = replay(stream, "a", tap=b)
    at = base["tap"]["w_at_b"]
    last = _last_w_crc(base, b)
    cut = _truncated(stream, b)
    ended = replay(cut, "a")
    end = (len(cut["events"]), ended["observed"], ended["fingerprint"]["W"])
    agree_i = at["crc_W"] == last
    agree_ii = (at["event"], at["observed"], at["crc_W"]) == end
    with _late_tap(by=late):
        slip = replay(stream, "a", tap=b)["tap"]["w_at_b"]
    late_i = slip["crc_W"] == last
    late_ii = (slip["event"], slip["observed"], slip["crc_W"]) == end
    moved = _last_w_crc(base, b + late) != last
    return {
        "verdict": bool(agree_i and agree_ii and not late_ii and (not late_i) == moved),
        "numbers": {
            "b": b,
            "tap": {key: at[key] for key in ("event", "observed", "crc_W")},
            "last_refit_crc": last,
            "truncated": dict(zip(("events", "observed", "crc_W"), end, strict=True)),
            "agree_i": agree_i,
            "agree_ii": agree_ii,
            "late": {"by": late, "tap": {key: slip[key] for key in ("event", "observed", "crc_W")}},
        },
        "red": {"late_fails_ii": not late_ii, "late_fails_i": not late_i, "moved_between": moved},
    }


# 2.10: R-a2's three cuts (all after b) on pool-copy and pool-act. Each stream, the recording and its
# suffix mutation, is tapped afresh (a side tap belongs to its own stream); the grower's predictions
# at t <= cut must be unchanged, and the re-taken weights at b equal the original's.
# RED: a tap that lists the FINAL model as its weights at b changes pool-copy's predictions before
# every cut. The cuts are exactly three distinct integers, each strictly after b and before the end of
# the stream: no cut at all (or a repeated one) leaves nothing to compare and would pass, and a cut at
# or past the end mutates nothing a later prediction could read.
# distinct-from: gates/eprop_transport_gate.py::_selfcheck -- that gate polices WEIGHT TRANSPORT in the source tree (a transposed forward weight feeding an update); this is G1(c)'s transport arms' R-a2 causality, replayed at three cuts
def check_transport_causality(stream, spec, *, b=TRANSPORT_B, cuts=None):
    count = len(stream["events"])
    cuts = (count // 4, count // 2, 3 * count // 4) if cuts is None else tuple(cuts)
    if any(isinstance(cut, bool) or not isinstance(cut, int) for cut in cuts):
        raise ValueError(f"REFUSED: every cut is an integer event count: {cuts}")
    if len(cuts) != 3 or len(set(cuts)) != 3:
        raise ValueError(f"REFUSED: the causality check takes exactly three distinct cuts: {cuts}")
    if any(cut <= b for cut in cuts):
        raise ValueError(f"REFUSED: every cut must lie after b ({b}): {cuts}")
    if any(cut >= count for cut in cuts):
        raise ValueError(
            f"REFUSED: every cut must lie before the end of the stream ({count} events): {cuts}"
        )
    taps = {"honest": replay(stream, "a", tap=b)["tap"]}
    with _leaky_tap():
        taps["leaky"] = replay(stream, "a", tap=b)["tap"]
    base = {
        arm: replay(stream, arm, pool=spec, side=taps["honest"])
        for arm in ("pool-copy", "pool-act")
    }
    leaked = replay(stream, "pool-copy", pool=spec, side=taps["leaky"])
    kept = {arm: [] for arm in base}
    same_weights, changed_by_leak = [], []
    for cut in cuts:
        changed = mutate_suffix(stream, cut, stream["seed"] + cut)
        tap = replay(changed, "a", tap=b)["tap"]
        same_weights.append(tap["w_at_b"]["crc_W"] == taps["honest"]["w_at_b"]["crc_W"])
        for arm in kept:
            other = replay(changed, arm, pool=spec, side=tap)
            kept[arm].append(base[arm]["predictions"][:cut] == other["predictions"][:cut])
        with _leaky_tap():
            leaky = replay(changed, "a", tap=b)["tap"]
        other = replay(changed, "pool-copy", pool=spec, side=leaky)
        changed_by_leak.append(leaked["predictions"][:cut] != other["predictions"][:cut])
    return {
        "verdict": bool(
            all(all(v) for v in kept.values()) and all(same_weights) and all(changed_by_leak)
        ),
        "numbers": {"cuts": list(cuts), **kept, "tap_crc_unchanged": same_weights},
        "red": {"leaky_tap": changed_by_leak},
    }


# 2.10: arm (a) with the tap on and off is equal on GROWER_KEYS and on the reference readout's
# predictions and fingerprint. RED: a writing tap (1e-3 added to W[0, 0]) must change them.
def check_tap_read_only(stream, *, b=TRANSPORT_B):
    off = replay(stream, "a")
    on = replay(stream, "a", tap=b)
    grower = {key: on[key] == off[key] for key in GROWER_KEYS}
    readout = on["readout"] == off["readout"]
    with _writing_tap():
        written = replay(stream, "a", tap=b)
    red = any(written[key] != off[key] for key in GROWER_KEYS)
    return {
        "verdict": bool(all(grower.values()) and readout and red),
        "numbers": {"grower": grower, "readout": readout},
        "red": red,
    }


def _load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def _save(path, value):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as file:
        json.dump(value, file, allow_nan=False)


def _synthetic_stream():
    """A hand-built event schedule; the self-check never constructs a brain."""
    torch.set_num_threads(1)
    events, sleep = [], []
    for i in range(48):
        category = (i // 3) % 3
        features = [0.0] * 64
        features[(i % 3) * 8] = 1.0
        events.append(
            {
                "t": i + 1,
                "tick": i * 2 + 1,
                "x": features,
                "c": category,
                "causal_lp": 0.0,
                "segment": "warm-up" if i < 24 else "unseen-order",
            }
        )
        sleep.append({"tick": i + 1, "brain_tick": i * 2 + 2, "event": i + 1, "live_lp": 0.0})
    return {
        "seed": 17,
        "config": {
            "seed": 17,
            "refit_every": 4,
            "buffer_cap": 1000,
            "l_max": None,
            "eps_tol": 0.28,
            "T": 15,
        },
        "policy": _policy(),
        "events": events,
        "sleep": sleep,
    }


# ---- G1(b): R-b6's synthetic stream family (g1b-pool-selection.md section 2.7) -------------------

RB6_WORDS = ((0, 2, 3, 4), (1, 2, 3, 5))  # the source SORN task at SHARED_LEN 2
RB6_WORDS_HISTORY_FREE = ((0, 2, 3, 4), (1, 2, 3, 4))  # the red twin: after 3 always 4
RB6_SYMBOLS = 6  # symbols 0-5, each an 8-dim block of the 64
RB6_EVENTS = 1200
RB6_WARM_UP = 256
RB6_NORM = 1.73  # the live percepts' L2 norm
RB6_BAR = 0.05  # the readout's (e1) - (e1-0) must reach it on 4242; its red twin must stay below


# distinct-from: growth/neurogenesis_engine.py -- SyntheticPerceptStream drives K Markov context modes
#   straight into the engine; this is a recorded stream for replay(), from the SORN words
# distinct-from: reasoning/sorn.py -- make_stream yields (symbol, noise units) for its own network,
#   WORD1 when random() < 0.5; this family is section 2.7's, WORDS[random() < 0.5], as a recording
# distinct-from: reasoning/jepa_scaffold_torch.py -- SyntheticStream renders an action-conditioned
#   latent; no category stream
# distinct-from: sensorimotor/corpus_stream.py -- yields (context words, next word) from a text corpus
def rb6_stream(seed, *, history_free=False):
    """R-b6's synthetic stream (section 2.7; the design generator is archive/g1b-2026-09-25/
    probe_rb6_positive_control.py::synthetic_stream). Whole words WORDS[rng.random() < 0.5], rng =
    random.Random(seed), are appended until there are at least 1,200 symbols, then truncated to
    1,200. Symbol k is the disjoint block dims[8k : 8k + 8] of dims = default_rng(seed)
    .permutation(64), valued 1.73 / sqrt(8), and its category is k. Event i has tick 2i + 1 and one
    sleep tick at 2i + 2; every |LP| is a recorded None; the first 256 events are warm-up and the rest
    "unseen-order". The grower's configuration is the recorded one with seed = the stream seed, and
    the seed also stands for the recording seed in section 2.2's W_in and realization seeds.
    `history_free` builds the red twin family, in which the past never predicts beyond the current
    symbol. The `final` block carries the current revision, so replays validate."""
    torch.set_num_threads(1)
    words = RB6_WORDS_HISTORY_FREE if history_free else RB6_WORDS
    rng = random.Random(seed)
    symbols = []
    while len(symbols) < RB6_EVENTS:
        symbols.extend(words[rng.random() < 0.5])
    symbols = symbols[:RB6_EVENTS]
    dims = np.random.default_rng(seed).permutation(64)
    patterns = []
    for k in range(RB6_SYMBOLS):
        pattern = np.zeros(64)
        pattern[dims[8 * k : 8 * k + 8]] = RB6_NORM / np.sqrt(8)
        patterns.append([float(v) for v in pattern])
    events, sleep = [], []
    for i, symbol in enumerate(symbols):
        events.append(
            {
                "t": i + 1,
                "tick": 2 * i + 1,
                "x": list(patterns[symbol]),
                "c": symbol,
                "causal_lp": None,
                "segment": "warm-up" if i < RB6_WARM_UP else "unseen-order",
            }
        )
        sleep.append({"tick": i + 1, "brain_tick": 2 * i + 2, "event": i + 1, "live_lp": None})
    head, dirty = _revision()
    return {
        "seed": seed,
        "synthetic": {"family": "rb6", "history_free": bool(history_free)},
        "config": {
            "seed": seed,
            "refit_every": 20,
            "buffer_cap": 1000,
            "l_max": None,
            "eps_tol": 0.28,
            "T": 15,
        },
        "policy": _policy(),
        "events": events,
        "sleep": sleep,
        "final": {"head": head, "dirty": dirty},
    }


def _child(args, out):
    try:
        result = subprocess.run(
            # the module's own name, never __name__: under `python -m` that is "__main__", and a child started
            # as `-m __main__` dies before running (measured: the first pilot's 8 recordings, 2 s)
            [sys.executable, "-m", "snn_data_lake.instruments.memory_diagnostic", *args],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode:
            status = {"status": "FAIL", "path": str(out), "stderr": result.stderr[-1000:]}
        else:
            status = {"status": "OK", "path": str(out)}
    except subprocess.TimeoutExpired:
        status = {"status": "TIMEOUT", "path": str(out)}
    _save(str(out) + ".status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", nargs=2, metavar=("SEED", "OUT"))
    group.add_argument("--inbrain-oracle", nargs=2, metavar=("SEED", "OUT"))
    group.add_argument("--replay", nargs=4, metavar=("STREAM", "ARM", "MODE", "OUT"))
    group.add_argument("--pilot")
    group.add_argument("--check")
    args = parser.parse_args()
    if args.record:
        _save(args.record[1], record_stream(int(args.record[0])))
    elif args.inbrain_oracle:
        _save(args.inbrain_oracle[1], record_stream(int(args.inbrain_oracle[0]), arm_feed="f64"))
    elif args.replay:
        source, arm, mode, out = args.replay
        _save(out, replay(_load(source), arm, mode))
    elif args.pilot:
        from concurrent.futures import ThreadPoolExecutor

        directory = Path(args.pilot)
        directory.mkdir(parents=True, exist_ok=True)
        jobs = []
        for seed in range(100, 104):
            for flag, suffix in (("--record", "stream"), ("--inbrain-oracle", "oracle")):
                out = directory / f"{seed}-{suffix}.json.gz"
                jobs.append((flag, [str(seed), str(out)], out))
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(lambda job: _child([job[0], *job[1]], job[2]), jobs))
        if any(row["status"] != "OK" for row in statuses):
            _save(directory / "summary.json", statuses)
            raise SystemExit(1)
        replay_jobs = []
        for seed in range(100, 104):
            source = directory / f"{seed}-stream.json.gz"
            for arm in SCORED_ARMS:
                replay_jobs.append(
                    (
                        [
                            "--replay",
                            str(source),
                            arm,
                            "causal",
                            str(directory / f"{seed}-{arm}-causal.json"),
                        ],
                        directory / f"{seed}-{arm}-causal.json",
                    )
                )
            for arm in ("a", "f64"):
                live_source = directory / f"{seed}-oracle.json.gz" if arm == "f64" else source
                replay_jobs.append(
                    (
                        [
                            "--replay",
                            str(live_source),
                            arm,
                            "live",
                            str(directory / f"{seed}-{arm}-live.json"),
                        ],
                        directory / f"{seed}-{arm}-live.json",
                    )
                )
            stream = _load(source)
            cuts = (
                len(stream["events"]) // 4,
                len(stream["events"]) // 2,
                3 * len(stream["events"]) // 4,
            )
            for cut in cuts:
                changed = directory / f"{seed}-cut-{cut}.json.gz"
                _save(changed, mutate_suffix(stream, cut, seed + cut))
                for arm in SCORED_ARMS:
                    out = directory / f"{seed}-{arm}-cut-{cut}.json"
                    replay_jobs.append((["--replay", str(changed), arm, "causal", str(out)], out))
            if seed == 100:
                for arm in SCORED_ARMS:
                    for twin in (1, 2):
                        out = directory / f"100-{arm}-twin-{twin}.json"
                        replay_jobs.append(
                            (["--replay", str(source), arm, "causal", str(out)], out)
                        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses += list(pool.map(lambda job: _child(*job), replay_jobs))
        _save(directory / "summary.json", statuses)
        if any(row["status"] != "OK" for row in statuses):
            raise SystemExit(1)
    elif args.check:
        directory = Path(args.check)
        report = {}
        for seed in range(100, 104):
            stream = _load(directory / f"{seed}-stream.json.gz")
            oracle = _load(directory / f"{seed}-oracle.json.gz")
            cuts = (
                len(stream["events"]) // 4,
                len(stream["events"]) // 2,
                3 * len(stream["events"]) // 4,
            )
            a2_runs = {
                arm: {
                    "base": directory / f"{seed}-{arm}-causal.json",
                    **{cut: directory / f"{seed}-{arm}-cut-{cut}.json" for cut in cuts},
                }
                for arm in SCORED_ARMS
            }
            a4_runs = (
                {
                    arm: (
                        directory / f"100-{arm}-twin-1.json",
                        directory / f"100-{arm}-twin-2.json",
                    )
                    for arm in SCORED_ARMS
                }
                if seed == 100
                else None
            )
            report[str(seed)] = {
                "R-a1": check_r_a1(stream, _load(directory / f"{seed}-a-live.json")),
                "R-a1b": check_r_a1b(stream, oracle, _load(directory / f"{seed}-f64-live.json")),
                "R-a2": check_r_a2(stream, a2_runs),
                "R-a2b": check_r_a2b(stream),
                "R-a3": check_r_a3(stream),
                "R-a4": check_r_a4(stream, a4_runs),
            }
        if any(not result["verdict"] for checks in report.values() for result in checks.values()):
            _save(directory / "checks.json", report)
            raise SystemExit(1)
        for seed in range(100, 104):
            stream = _load(directory / f"{seed}-stream.json.gz")
            report[str(seed)]["contrast_f64_a"] = {
                arm: score(stream, result["predictions"], result["fallbacks"])
                for arm, result in (
                    ("a", _load(directory / f"{seed}-a-causal.json")),
                    ("f64", _load(directory / f"{seed}-f64-causal.json")),
                )
            }
        _save(directory / "checks.json", report)
    else:
        fixture = _synthetic_stream()
        live = replay(fixture, "a", "live")
        fixture["final"] = {
            key: live[key]
            for key in (
                "live_n",
                "live_correct",
                "live_persist",
                "ticks",
                "observed",
                "trajectory",
                "fingerprint",
            )
        }
        fixture["final"]["gate_verdicts"] = live["gate_verdicts"]
        fixture["final"].update(live["revision"])
        completed = {(r["tick"], r["event"]): r["fingerprint"] for r in live["refits"]}
        fixture["refits"] = [
            dict(row, refit=completed.get((row["tick"], row["event"])))
            for row in fixture["sleep"]
            if row["tick"] % fixture["config"]["refit_every"] == 0
        ]
        for event, prediction in zip(fixture["events"], live["predictions"], strict=True):
            event["prediction"] = prediction
        assert check_r_a1(fixture)["verdict"]
        assert check_r_a2b(fixture)["red"]
        assert check_r_a3(fixture)["red"]
        assert check_r_a2(fixture)["verdict"]
        assert check_r_a4(fixture)["verdict"]
        assert _lp_sensitive_red("a", "causal")
        categories = [event["c"] for event in fixture["events"]]
        assert (
            score(fixture, categories, [False] * len(categories))["unseen-order"]["accuracy"] == 1
        )
        assert (
            score(fixture, [None, *categories[:-1]], [False] * len(categories))["unseen-order"][
                "accuracy"
            ]
            < 1
        )
        missing = copy.deepcopy(fixture)
        del missing["events"][15]["causal_lp"]
        try:
            replay(fixture, "a", mutate=missing)
        except ValueError:
            pass
        else:
            raise AssertionError("deleted LP record accepted")
        disconnected = copy.deepcopy(fixture)
        disconnected["_no_lp_reader"] = True
        try:
            replay(fixture, "a", mutate=disconnected)
        except ValueError:
            pass
        else:
            raise AssertionError("disconnected LP reader accepted")
        trajectory = [(1, 1, 0.5), (2, 2, float("nan")), (3, 3, 0.6)]
        old = GrowthStop()
        refusals = 0
        for _, units, accuracy in trajectory[old.observed("cortex") :]:
            try:
                old.observe_refit("cortex", int(units), float(accuracy))
            except (ValueError, TypeError):
                refusals += 1
        expected = (old.should_grow("cortex"), refusals, old.observed("cortex"))
        shared = GrowthStop()
        verdict, actual_refusals = feed_and_decide(shared, trajectory)
        assert (verdict, actual_refusals, shared.observed("cortex")) == expected
        dropped = GrowthStop()
        feed_and_decide(dropped, trajectory[:-1])
        assert dropped.observed("cortex") != shared.observed("cortex")
        assert onehot_32(31)[31] == 1
        try:
            onehot_32(32)
        except ValueError:
            pass
        else:
            raise AssertionError("onehot_32 accepted 32")
        # G1(b), g1b-pool-selection.md section 2: the readout's R-b0x fixtures, R-b2, R-b8's two
        # comparisons with their reds on this fixture, one pool-arm replay (its window shortened to 16
        # for this 48-event fixture only), and R-b6's generator.
        assert check_r_b0x_readout()["verdict"]
        assert check_r_b2()["verdict"]
        on, off = replay(fixture, "a"), replay(fixture, "a", readout=False)
        assert off["readout"] is None and all(on[key] == off[key] for key in GROWER_KEYS)
        with _writing_readout():
            written = replay(fixture, "a")
        assert any(written[key] != on[key] for key in GROWER_KEYS)
        e1 = replay(fixture, "e1")
        with _e1_mutant():
            mutant = replay(fixture, "e1")
        assert any(mutant[key] != e1[key] for key in GROWER_KEYS)
        spec = {
            "candidate": "lsm",
            "config": rp.R_B5_PERSISTENCE_WITNESS,
            "form": "carried",
            "window": 16,
        }
        pooled = replay(fixture, "pool", pool=spec)
        _validate_run(pooled, fixture, fixture, "pool", "causal")
        assert pooled["pool"]["frozen"] and pooled["pool"]["degeneracy"]["rows"] == 24
        assert len(pooled["readout"]["predictions"]) == len(fixture["events"])
        # G1(c) 2.6: window 0 is admitted and Carried@0 is Carried@256. The default window never
        # reaches its boundary on 48 events, so the boundary, frozen and movement fields are not
        # compared (they differ before a boundary exists); everything the learners and the pool's
        # continuation state carry is.
        zero = replay(fixture, "pool", pool=dict(spec, window=0))
        default = replay(fixture, "pool", pool=dict(spec, window=rp.WINDOW))
        _validate_run(zero, fixture, fixture, "pool", "causal")
        assert all(zero[key] == default[key] for key in GROWER_KEYS)
        assert zero["readout"] == default["readout"]
        assert all(zero["pool"][key] == default["pool"][key] for key in ("state", "state_chain"))
        assert zero["pool"]["window"] == 0 and zero["pool"]["boundary"] is not None
        assert zero["pool"]["movement"] == {}
        for history_free, followers in ((False, {4, 5}), (True, {4})):
            family = rb6_stream(4242, history_free=history_free)
            cats = [event["c"] for event in family["events"]]
            assert {b for a, b in itertools.pairwise(cats) if a == 3} == followers
            assert sum(e["segment"] == "unseen-order" for e in family["events"]) == 944
        # G1(c) 2.7 and 2.10 "Transport": the channel, timing, causality and read-only checks and the
        # encoder's contract, each with its red, and G1(b)'s arms where they were. b is 256 for the
        # recordings; on this 48-event fixture ONLY it is 16 (256 does not fit, and the first refit
        # lands at event 12), the late tap's offset 8 (not 40) and R-a2's three cuts 24, 32, 40.
        assert SCORED_ARMS == tuple(ARMS)[:8] and TRANSPORT_B == 256
        assert tuple(ARMS)[8:12] == ("forbidden", "noise", "leak", "pool")
        fixture_b = 16
        channels = check_transport_channels(fixture, spec, b=fixture_b)
        assert channels["verdict"], channels
        timing = check_transport_timing(fixture, b=fixture_b, late=8)
        assert timing["verdict"], timing
        causality = check_transport_causality(fixture, spec, b=fixture_b, cuts=(24, 32, 40))
        assert causality["verdict"], causality
        read_only = check_tap_read_only(fixture, b=fixture_b)
        assert read_only["verdict"], read_only
        encoder = rp.check_transported_encoder()
        assert encoder["verdict"], encoder
        print(
            "G1(b): readout R-b0x fixtures, R-b2, R-b8 comparisons and reds, pool arm, R-b6 family OK"
        )
        print("G1(c): Carried@0 = Carried@256 on the 48-event fixture OK")
        print(
            f"G1(c) transport on the 48-event fixture at b {fixture_b}: channels, timing (late tap "
            f"fails ii: {timing['red']['late_fails_ii']}, i: {timing['red']['late_fails_i']}), "
            f"causality at cuts 24/32/40 (leaky tap changed {causality['red']['leaky_tap']}), "
            "tap read-only (writing tap changed it), encoder norm and refusals OK"
        )
        print("memory_diagnostic self-check OK")


if __name__ == "__main__":
    main()
