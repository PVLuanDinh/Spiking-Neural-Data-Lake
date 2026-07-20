r"""
NENGO-GUI DEVELOPMENTAL LOOP -- watch the loop's REAL spikes drive a live Nengo network.

WHAT THIS IS
------------
research/developmental_loop_deep.py runs the autonomous developmental cycle -- spiking curiosity
picks the next objective digit, a locus-coeruleus noradrenaline burst boosts the learning gain,
a CONTINUAL linear readout learns it on FROZEN unsupervised conv1 latency-STDP features, replay
consolidates old digits, competence feeds the next decision. This file wires that loop THROUGH
Nengo so you can VISUALIZE it in NengoGUI: a `nengo.Node` advances the loop and, every sim step,
emits (a) the genuine conv1 PERCEPTION SPIKES for the digit currently streaming and (b) the loop's
neuromodulation / competence / accuracy signals. Each output drives its own LABELLED spiking NEF
`nengo.Ensemble`, so NengoGUI draws a live spike raster per signal -- the headline being the
"perception (conv1 feature spikes)" population.

This follows research/nengo_gui_brain.py exactly (module-level `model`, one Node driving labelled
ensembles, spikes + decoded probes) and reuses the substrate/nengo_bridge idiom of turning a numeric
signal vector into its own spiking NEF Ensemble.

HOW THE LOOP DRIVES THE ENSEMBLES (the Node output layout)
----------------------------------------------------------
The Node advances ONE developmental cycle every NORD_CYCLE_EVERY sim-steps (a short cheap Adam burst
on cached features) and, EVERY step, streams the next of a handful of cached test digits. Per step it
returns one concatenated vector, sliced onto these ensembles:
  * "perception (conv1 feature spikes)"  dim = C1 -- per-map FIRST-SPIKE activity of the streaming
        digit: (t_lat - fired).amax-over-space / t_lat per conv map, in [0,1]. THE headline signal;
        it changes as different digits stream, so the raster visibly tracks perception.
  * "competence (per-digit)"             dim = 10 -- held-out test accuracy per digit (grows as the
        loop learns more digits).
  * "objective (digit/9)"                dim = 1  -- the digit the loop is currently learning.
  * "neuromod (LC-NA, curiosity)"        dim = 2  -- locus-coeruleus noradrenaline level + the spiking
        curiosity value of the current objective (the loop's dopamine-like drive).
  * "progress (test-acc, curric/10)"     dim = 2  -- overall test accuracy + curriculum mastered / 10.
Radii are set from the actual signal magnitudes so the populations spike visibly (not saturated).

HONEST framing: Nengo does NOT run the loop's computation (the loop is torch: conv1 STDP + a linear
readout + the zero-dep neuromodulators). It OBSERVES and re-encodes the loop's spiking/telemetry
signals into spiking NEF populations -- the same observability seam as nengo_gui_brain.py. The conv1
PERCEPTION spikes are genuine first-spike-latency activity of the unsupervised extractor; the loop
telemetry (competence/objective/neuromod/accuracy) are re-encoded scalars. Observation + classification
ONLY -- no actuator / motor route. Deployable to NengoLoihi as a monitoring layer.

LAUNCH THE GUI (from the repo root; needs the TORCH venv, which also has nengo)
------------------------------------------------------------------------------
    .venv-cu128\Scripts\nengo.exe -P 8765 research\nengo_gui_devloop.py
Notes: the port flag is CAPITAL -P (lowercase -p is password); default port 8080 is blocked on this
Windows box, so pass -P 8765. The browser opens at http://localhost:8765 . Press play, then right-click
the "perception (conv1 feature spikes)" ensemble for its spike raster. The module-level conv1 pre-train
runs once at import (small by default: NORD_C1=16, NORD_L1_EPOCHS=2, NORD_TRAIN=3000) to keep the import
responsive; shrink those envs if your machine imports slowly.

HEADLESS VERIFY (no GUI socket needed -- how the main agent checks it)
---------------------------------------------------------------------
    NORD_GPU=1 PYTHONPATH=src .venv-cu128\Scripts\python research\nengo_gui_devloop.py
main() builds `model`, runs a headless nengo.Simulator(model) for sim.run(0.2), and ASSERTS the spike
probes recorded > 0 spikes across the ensembles (proving the loop drives real Nengo spikes), printing
n signals / total spikes / per-ensemble spike counts.

Scale knobs (env): NORD_C1 (16) NORD_T (15) NORD_L1_EPOCHS (2) NORD_TRAIN (3000) NORD_TEST (800)
                   NORD_SGD (60) NORD_CYCLE_EVERY (40) NORD_PER (200) NORD_REPLAY (100) NORD_CAP (1)
                   NORD_GPU=1
Deps: numpy torch torchvision snntorch nengo   (research/GPU+nengo tier; the zero-dep core is untouched)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
pass  # migrated: sys.path bootstrap removed (now an importable package)
pass  # migrated: sys.path bootstrap removed (now an importable package)

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import nengo  # noqa: E402

from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist  # noqa: E402
from snn_data_lake.reasoning.torch_device import enable, resolve_device  # noqa: E402
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp  # noqa: E402

from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus  # noqa: E402
from snn_data_lake.neuromodulation.spiking_curiosity import (  # noqa: E402
    SpikingCuriosityDrive, SpikingCurriculum)

BASE_LR = 0.01          # continual linear-readout Adam lr (matches developmental_loop_deep)
WEIGHT_DECAY = 1e-4


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


# ---------------------------------------------------------------------------------------------------
# CONFIG -- small by default so the module-level pre-train keeps the GUI import responsive
# ---------------------------------------------------------------------------------------------------
C1 = _envi("NORD_C1", 16)              # conv1 feature maps (== perception ensemble dimensions)
T_LAT = _envi("NORD_T", 15)            # latency steps
L1_EPOCHS = _envi("NORD_L1_EPOCHS", 2)  # unsupervised conv1 pre-training epochs
TRAIN_N = _envi("NORD_TRAIN", 3000)    # train images (conv pre-train + cached readout features)
TEST_N = _envi("NORD_TEST", 800)       # held-out test images (cached for competence)
SGD = _envi("NORD_SGD", 60)            # Adam steps per developmental cycle
CYCLE_EVERY = _envi("NORD_CYCLE_EVERY", 40)  # advance ONE loop cycle every this many sim-steps
PER = _envi("NORD_PER", 200)           # objective-digit train features per cycle
REPLAY = _envi("NORD_REPLAY", 100)     # replayed feature rows stored + rehearsed per learned digit
CAP = _envi("NORD_CAP", 1)             # max consecutive revisits of the same digit (anti-fixation)

torch.manual_seed(0)                   # determinism
_DEV = resolve_device()
enable(_DEV)                           # route torch factories to GPU when NORD_GPU=1 (else CPU)


def _accuracy(clf, feats, labels, idx):
    """Held-out linear-readout accuracy over the given cached-feature rows (test cache only)."""
    if not idx:
        return 0.0
    with torch.no_grad():
        i = torch.tensor(idx, dtype=torch.long, device=_DEV)
        pred = clf(feats[i]).argmax(dim=1)
        return float((pred == labels[i]).float().mean().item())


def _perception_spikes(conv, image):
    """The genuine conv1 PERCEPTION SPIKES for one image, as a length-C1 vector in [0,1]:
    per conv map, (t_lat - first_spike_time).amax-over-space / t_lat  (earlier spike = stronger;
    never-fired map = 0). This is the frozen extractor's real first-spike-latency activity."""
    st = conv._dog_latency(image)
    fired, _ = conv._forward(st)                          # [C, hout, hout] first-spike times (t_lat == never)
    val = torch.where(fired < conv.t_lat,
                      (conv.t_lat - fired).float() / conv.t_lat,
                      torch.zeros_like(fired, dtype=torch.float32))
    per_map = val.reshape(conv.C, -1).amax(dim=1)         # [C] in [0,1]
    return per_map


# ---------------------------------------------------------------------------------------------------
# MODULE-LEVEL PRE-TRAIN (runs once at import) -- unsupervised conv1, then cache features + demo digits
# ---------------------------------------------------------------------------------------------------
print("[nengo_gui_devloop] loading MNIST + pre-training conv1 (unsupervised, once at import)...")
_Xtr, _Ytr, _Xte, _Yte = load_mnist(TRAIN_N, TEST_N)

torch.manual_seed(0)
_conv = ConvLatencyStdp(C1, T_LAT)                        # frozen unsupervised perception extractor
for _ep in range(L1_EPOCHS):
    for _img in _Xtr:
        _conv.train_image(_img)
_DIVERSITY = _conv.kernel_diversity()
print(f"[nengo_gui_devloop] conv1 frozen: C1={C1} feat_dim={_conv.feat_dim()} "
      f"kernel-diversity(cos-sim)={_DIVERSITY:.3f} ({'no collapse' if _DIVERSITY < 0.6 else 'WARN collapse'})")

# cache the frozen readout features ONCE for the whole train + test pool (features never change)
_Ftr = torch.stack([_conv.features(img).reshape(-1) for img in _Xtr], dim=0).to(_DEV)
_Fte = torch.stack([_conv.features(img).reshape(-1) for img in _Xte], dim=0).to(_DEV)
_ytr = _Ytr.long().to(_DEV)
_yte = _Yte.long().to(_DEV)
_D = _Ftr.shape[1]
_tr_by = {d: (_Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
_te_by = {d: (_Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

# a handful of test digits (one per class) to STREAM through the perception ensemble, with their
# genuine conv1 first-spike activity precomputed (frozen conv -> cheap per-step indexing in the Node)
_demo_idx = [(_te_by[d][0] if _te_by[d] else 0) for d in range(10)]
_demo_perc = [_perception_spikes(_conv, _Xte[i]).detach().to("cpu").tolist() for i in _demo_idx]
# perception NEF radius from the actual demo-vector magnitudes so the population spikes visibly
_perc_maxnorm = max(float(np.linalg.norm(v)) for v in _demo_perc) or 1.0
_PERC_RADIUS = max(1.0, 1.2 * _perc_maxnorm)
print(f"[nengo_gui_devloop] cached features train{tuple(_Ftr.shape)} test{tuple(_Fte.shape)}; "
      f"{len(_demo_perc)} demo digits; perception radius={_PERC_RADIUS:.2f}")


# ---------------------------------------------------------------------------------------------------
# THE DRIVER -- advances the developmental loop and emits (perception spikes + loop telemetry) per step
# ---------------------------------------------------------------------------------------------------
class DevLoopDriver:
    """Runs the autonomous developmental loop (verbatim control flow from developmental_loop_deep.py:
    spiking curiosity picks the objective, LC noradrenaline boosts the readout gain, a CONTINUAL linear
    head learns it on frozen conv1 features with balanced replay, competence feeds the next decision).
    Called once per Nengo sim step by the model Node; advances one cycle every CYCLE_EVERY steps and
    streams one cached test digit's perception spikes every step."""

    def __init__(self):
        self.clf = nn.Linear(_D, 10).to(_DEV)
        self.opt = torch.optim.Adam(self.clf.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        self.curiosity = SpikingCuriosityDrive(seed=0)                     # spiking learning-progress + novelty
        self.curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
        self.lc = LocusCoeruleus()                                        # noradrenaline: salient/novel -> higher gain
        self.learned = []                                                # digits chosen/learned so far
        self.competence = {d: 0.0 for d in range(10)}
        self.replay_pool = {}
        self.consec = {d: 0 for d in range(10)}
        self.step_i = 0
        self.cyc = 0
        # held telemetry snapshot (updated each cycle; represented as spikes between cycles)
        self.comp = [0.0] * 10
        self.obj_scalar = 0.0
        self.na = 0.0
        self.curio = 0.0
        self.acc = 0.0
        self.mastered_frac = 0.0

    def _advance_cycle(self):
        # DECISION: spiking curiosity picks the objective (phase 1 explores unseen; phase 2 revisits the
        # weakest-still-improving digit, with the CAP anti-fixation guard) -- no hand-coded heuristic.
        unseen = [d for d in range(10) if d not in self.learned]
        candidates = unseen if unseen else list(range(10))
        if not unseen:
            allowed = [d for d in candidates if self.consec[d] < CAP]
            candidates = allowed or candidates
        objective = int(self.curiosity.most_curious(candidates))
        self.consec = {d: (self.consec[d] + 1 if d == objective else 0) for d in range(10)}
        is_new = objective not in self.learned

        # NEUROMODULATION: a novel / hard objective is SALIENT -> LC phasic NA burst -> bounded lr boost.
        salience = 1.0 if is_new else (1.0 - self.competence[objective])
        na = float(self.lc.step(salience))
        gain = 1.0 + 0.5 * na
        for g in self.opt.param_groups:
            g["lr"] = BASE_LR * gain

        # LEARN: continual SGD on the objective digit's cached features + a BALANCED REPLAY of every
        # previously-learned digit's stored cached features (rehearsal = anti-forgetting).
        n_train = PER if is_new else max(REPLAY, PER // 4)
        obj_rows = _tr_by[objective][:n_train]
        others = [d for d in self.learned if d != objective]
        replay_rows = [r for d in others for r in self.replay_pool.get(d, [])]
        batch_rows = obj_rows + replay_rows
        idx = torch.tensor(batch_rows, dtype=torch.long, device=_DEV)
        Xb, yb = _Ftr[idx], _ytr[idx]
        self.clf.train()
        loss = torch.tensor(0.0)
        for _ in range(SGD):
            self.opt.zero_grad()
            loss = F.cross_entropy(self.clf(Xb), yb)
            loss.backward()
            self.opt.step()
        self.clf.eval()
        if is_new:
            self.replay_pool[objective] = _tr_by[objective][n_train:n_train + REPLAY]
            self.learned.append(objective)

        # MEASURE competence on the held-out TEST cache (train/test never mix), overall + per-digit.
        for d in self.learned:
            self.competence[d] = _accuracy(self.clf, _Fte, _yte, _te_by[d])
        pooled = [r for d in self.learned for r in _te_by[d]]
        overall = _accuracy(self.clf, _Fte, _yte, pooled)

        # DECIDE: feed the REAL curiosity + curriculum modules the outcome (pred error = 1 - competence).
        self.curiosity.observe(objective, 1.0 - self.competence[objective])
        lp = self.curiosity.learning_progress(objective)
        self.curriculum.observe(lp)

        # update the held telemetry snapshot the Node re-encodes as spikes
        self.comp = [float(self.competence[d]) for d in range(10)]
        self.obj_scalar = objective / 9.0
        self.na = na
        self.curio = min(2.0, float(self.curiosity.curiosity_value(objective)))
        self.acc = float(overall)
        self.mastered_frac = self.curriculum.mastered / 10.0
        self.cyc += 1

    def tick(self, t):
        """One Nengo sim step: advance a loop cycle on the CYCLE_EVERY cadence, then return the
        concatenated [perception(C1) | competence(10) | objective(1) | neuromod(2) | progress(2)]."""
        if self.step_i % CYCLE_EVERY == 0:
            self._advance_cycle()
        perc = _demo_perc[self.step_i % len(_demo_perc)]
        self.step_i += 1
        return list(perc) + list(self.comp) + [self.obj_scalar, self.na, self.curio,
                                               self.acc, self.mastered_frac]


# ---------------------------------------------------------------------------------------------------
# THE NENGO MODEL -- NengoGUI imports this file and renders `model`
# ---------------------------------------------------------------------------------------------------
# ensemble layout: (label, dimensions, radius, n_neurons). Radii cover each signal's [0,1] magnitude
# so the populations spike visibly rather than saturating.
_SPEC = [
    ("perception (conv1 feature spikes)", C1, _PERC_RADIUS, max(100, 6 * C1)),
    ("competence (per-digit)",            10, math.sqrt(10),                80),
    ("objective (digit/9)",               1,  1.0,                          40),
    ("neuromod (LC-NA, curiosity)",       2,  2.0,                          50),
    ("progress (test-acc, curric/10)",    2,  1.5,                          50),
]

_driver = DevLoopDriver()
_SIZE_OUT = sum(dim for _, dim, _, _ in _SPEC)

model = nengo.Network(label="SNDL developmental loop (live spikes under Nengo)")
spike_probes = {}
decoded_probes = {}
with model:
    loop_node = nengo.Node(_driver.tick, size_out=_SIZE_OUT, label="dev-loop.tick() -> signals")
    _off = 0
    for _label, _dim, _radius, _nn in _SPEC:
        ens = nengo.Ensemble(n_neurons=_nn, dimensions=_dim, radius=_radius, label=_label, seed=0)
        nengo.Connection(loop_node[_off:_off + _dim], ens, synapse=0.01)
        spike_probes[_label] = nengo.Probe(ens.neurons)         # raw spikes -> NengoGUI raster / verify
        decoded_probes[_label] = nengo.Probe(ens, synapse=0.02)  # decoded signal
        _off += _dim


def main():
    """Headless verify (no GUI socket): build + run nengo.Simulator(model) for a short pass and
    ASSERT the spike probes recorded > 0 spikes across the ensembles."""
    print("=" * 88)
    print("NENGO-GUI DEVELOPMENTAL LOOP -- headless verify (nengo.Simulator, no GUI socket)")
    print("=" * 88)
    print(f"device={_DEV}  C1={C1}  ensembles={len(_SPEC)}  node size_out={_SIZE_OUT}  "
          f"cycle_every={CYCLE_EVERY}  sgd={SGD}")
    with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
        sim.run(0.2)

    per_ens = {label: int(np.count_nonzero(sim.data[spike_probes[label]])) for label, _, _, _ in _SPEC}
    total = sum(per_ens.values())
    print("-" * 88)
    print(f"advanced {_driver.cyc} developmental cycles; learned digits {sorted(_driver.learned)} "
          f"(overall test acc {_driver.acc:.1%}, curriculum mastered {_driver.curriculum.mastered}/10)")
    print(f"signals re-encoded as spiking NEF ensembles: {len(_SPEC)}   total spikes: {total}")
    for label, _, _, _ in _SPEC:
        print(f"    {label:36s} : {per_ens[label]} spikes")
    print("-" * 88)

    # asserts: the loop must drive REAL nengo spikes. Perception (always-positive first-spike activity)
    # guarantees the headline population fires; total across all ensembles must be > 0.
    assert total > 0, "the developmental loop drove NO nengo spikes (expected > 0)"
    assert per_ens["perception (conv1 feature spikes)"] > 0, \
        "the conv1 PERCEPTION population must spike (it is driven by real first-spike activity)"
    print(f"self-check OK: developmental loop drove {total} nengo spikes across {len(_SPEC)} "
          f"labelled spiking ensembles; perception population "
          f"{per_ens['perception (conv1 feature spikes)']} spikes. Observation-only, no actuator route; "
          f"deployable to NengoLoihi.")
    print("Launch the GUI:  .venv-cu128\\Scripts\\nengo.exe -P 8765 research\\nengo_gui_devloop.py")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
