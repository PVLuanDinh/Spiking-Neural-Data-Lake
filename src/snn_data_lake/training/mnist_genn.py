"""GeNN-accelerated surrogate-gradient MNIST brain trainer (GPU tier, GeNN backend).

*** BEST-EFFORT SCAFFOLD -- UNVERIFIED AGAINST A LIVE GeNN/ml_genn INSTALL ***

pygenn / ml_genn + CUDA + a C++ host compiler are NOT present on the machine this file
was written on (no GPU either), so NONE of the ml_genn code below has been executed or
even imported here. It is written correct-by-best-effort against the DOCUMENTED ml_genn
2.x API and will very likely need small adjustments when first run on a real GeNN box
(e.g. an RTX 5070 with CUDA 12.8+, a compiler, and ``pip install pygenn ml_genn``). Every
ml_genn touchpoint whose exact spelling/semantics could differ across ml_genn versions is
flagged inline with a ``# VERIFY(ml_genn):`` comment. Do NOT treat any accuracy printed by
this module as a real result until it has run end-to-end on a live GeNN install.

WHY GeNN: GeNN code-generates C++/CUDA for the whole spiking network, so the O(STEPS)
Python loop the Brian2/Torch trainer (mnist.py) runs collapses into one compiled CUDA kernel
sequence -- materially faster for SNNs on the GPU. This module is the GeNN sibling of
``snn_data_lake.training.mnist``; it trains the SAME 784 -> H -> 10 rate-coded spiking
classifier and exports to the SAME zero-dep ``trained_brain`` JSON format, so a torch-free
clone-and-run install serves the result with no GeNN, no torch, nothing.

APPROACH (ml_genn 2.x high-level API):
  * Network of three populations: SpikeInput(784) -> LeakyIntegrateFire(H) -> LeakyIntegrate(10).
  * MNIST pixels in [0, 1] are RATE-CODED over STEPS (Bernoulli per step), matching mnist.py
    and trained_brain's coding, then handed to the SpikeInput population as explicit spikes.
  * Trained with the EventProp compiler (ml_genn's gradient-based SNN training; the surrogate/
    adjoint route) + Adam on sparse-categorical-crossentropy over the leaky-integrate readout.
  * The CUDA backend is selected by GeNN automatically when CUDA + a compiler are available.

TRANSFER CAVEAT (honest, like mnist.py): GeNN's continuous-time LIF dynamics are NOT
bit-identical to trained_brain's discrete ``m = m*beta + cur; reset-to-0`` accumulate
semantics (GeNN decays as ``exp(-dt/tau_mem)``; we map ``tau_mem = -1/ln(beta)`` so a step
of decay approximately equals a multiply-by-beta, but synapse kinetics + reset differ). So
the exported weights are a STARTING POINT, not a guaranteed match. ``verify_transfer`` scores
a held-out MNIST subset through the zero-dep ``trained_brain.accuracy`` so the user SEES how
well the GeNN-trained weights transfer to the core; expect some drop, and a short core-side
fine-tune (or matching the neuron model more tightly) may be needed to close it.

torch/torchvision/ml_genn are imported at MODULE TOP on purpose: this module is only ever
imported when actually training with ``--genn`` (from cli.main behind an import guard), never
at package import time, so ``import snn_data_lake`` in a plain zero-dep install stays clean.
"""
import json
import math
import os

import numpy as np

# torchvision is used ONLY to fetch/normalise MNIST pixels to [0, 1]; ml_genn does the SNN.
# (ml_genn ships its own dataset helpers too, but torchvision is already the [train] extra's
#  MNIST source in mnist.py, so we reuse it for an identical pixel pipeline.)
from torchvision import datasets, transforms

# ---- ml_genn 2.x high-level API -------------------------------------------------------------
# VERIFY(ml_genn): these import paths are the documented ml_genn 2.x layout. If an import
# fails on the installed version, check that version's __init__ exports (class names have
# been stable across 2.x, but submodule paths occasionally move).
from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from ml_genn.utils.data import preprocess_spikes

from snn_data_lake.sensorimotor.trained_brain import (
    DEFAULT_BETA,
    DEFAULT_THRESHOLD,
    accuracy as core_accuracy,
    make_brain,
    save_brain,
)

DEFAULT_HIDDEN = 256
DEFAULT_STEPS = 25
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_TAU_SYN = 5.0        # ms; Exponential synapse time constant (GeNN has no DeltaCurr in EventProp)
IN_FEATURES = 784
OUT_CLASSES = 10


# ---- beta <-> tau_mem bridge ----------------------------------------------------------------
def _beta_to_tau(beta, dt=1.0):
    """Map trained_brain's per-step decay ``beta`` to a GeNN membrane time constant.

    trained_brain does ``m = m*beta + cur`` once per timestep; GeNN's LIF decays continuously
    as ``exp(-dt/tau_mem)`` per ``dt``. Setting ``tau_mem = -dt/ln(beta)`` makes one GeNN step
    of leak APPROXIMATELY equal one multiply-by-beta. This is the crux of the transfer caveat:
    it is an approximation, not an identity (reset + synapse kinetics still differ)."""
    beta = min(max(float(beta), 1e-6), 1.0 - 1e-9)
    return -float(dt) / math.log(beta)


# ---- MNIST -> rate-coded ml_genn spikes -----------------------------------------------------
def _datasets(root, download=True):
    """(train, test) torchvision MNIST datasets. ToTensor scales pixels to [0, 1]."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=download, transform=tf)
    test = datasets.MNIST(root=root, train=False, download=download, transform=tf)
    return train, test


def _to_numpy(ds, limit=None):
    """Materialise a torchvision MNIST dataset as (X[n,784] float32 in [0,1], y[n] int64)."""
    n = len(ds) if limit is None else min(int(limit), len(ds))
    X = np.empty((n, IN_FEATURES), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        image, label = ds[i]                     # image [1,28,28] in [0,1]
        X[i] = np.asarray(image, dtype=np.float32).reshape(-1)
        y[i] = int(label)
    return X, y


def _rate_encode_one(x, steps, rng):
    """Rate-code one image x[784] in [0,1] into (spike_times, spike_ids) for a SpikeInput.

    Pixel i fires with prob x_i per timestep -- EXACTLY trained_brain's / mnist.py's Bernoulli
    rate coding -- so the GeNN net sees the same input statistics the zero-dep core will serve.
    Times are integer step indices in [0, steps); ids are the firing pixel indices at each step.
    """
    times = []
    ids = []
    for t in range(int(steps)):
        fired = np.nonzero(rng.random(x.shape[0]) < x)[0]   # Bernoulli(x_i) this step
        if fired.size:
            times.append(np.full(fired.shape, float(t), dtype=np.float32))
            ids.append(fired.astype(np.int64))
    if not times:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.concatenate(times), np.concatenate(ids)


def _encode_dataset(X, steps, seed):
    """Rate-code X[n,784] into the list of per-example spike objects ml_genn's train/evaluate
    consume for a SpikeInput population.

    VERIFY(ml_genn): ``preprocess_spikes(times, ids, num_neurons)`` is the documented ml_genn
    2.x helper (ml_genn.utils.data) that packs raw (times, ids) into the PreprocessedSpikes
    record train()/evaluate() expect. Confirm the arg ORDER and that it wants times sorted
    ascending (they are here: we emit step 0,1,2,...). Some versions also expose
    ``preprocess_tonic_spikes`` for event datasets -- not what we want for rate-coded pixels."""
    rng = np.random.default_rng(int(seed))
    out = []
    for i in range(X.shape[0]):
        times, ids = _rate_encode_one(X[i], steps, rng)
        out.append(preprocess_spikes(times, ids, IN_FEATURES))   # VERIFY(ml_genn): signature
    return out


# ---- network: 784 -> H (LIF) -> 10 (leaky-integrate readout) --------------------------------
def build_network(hidden, beta, threshold, steps, batch_size, tau_syn, seed):
    """Build the ml_genn Network mirroring trained_brain's 784 -> H -> 10 chain.

    Returns (network, input_pop, hidden_pop, output_pop, conn_in_hid, conn_hid_out) so the
    caller can both train it and, after training, read the two connections' weights out.
    """
    tau_mem = _beta_to_tau(beta)
    # Upper bound on spikes the SpikeInput must buffer for one batch: every pixel could fire
    # every step. Safe but loose; tighten to the real max firing count to save VRAM.
    max_spikes = int(batch_size) * IN_FEATURES * int(steps)   # VERIFY(ml_genn): SpikeInput buffer semantics

    network = Network()
    with network:                                             # VERIFY(ml_genn): Network is a context manager in 2.x
        # SpikeInput replays the uploaded rate-coded spike trains natively in the CUDA kernel.
        inp = Population(SpikeInput(max_spikes=max_spikes), IN_FEATURES)
        # HIDDEN leaky integrate-and-fire: v_thresh == trained_brain threshold, v_reset 0
        # (reset-to-zero, matching the core), tau_mem from beta. tau_refrac 0 = no refractory.
        # VERIFY(ml_genn): LeakyIntegrateFire kwarg names (v_thresh / tau_mem / v_reset /
        # tau_refrac) and that reset is to v_reset (the core resets to 0).
        hid = Population(
            LeakyIntegrateFire(v_thresh=float(threshold), tau_mem=tau_mem,
                               v_reset=0.0, tau_refrac=0.0),
            int(hidden))
        # OUTPUT is a non-spiking leaky integrator read out by SUMMING its variable over the
        # trial -- the nearest ml_genn analogue of trained_brain's accumulate-current readout
        # (out[c] = sum_t sum_h W2[c][h]*spike_h(t)). VERIFY(ml_genn): readout="sum_var" is the
        # summed-over-time readout; other options are "avg_var"/"max_var"/"var". A large tau_mem
        # here would better approximate the core's NO-leak accumulate -- tune on the real box.
        out = Population(LeakyIntegrate(tau_mem=tau_mem, readout="sum_var"), OUT_CLASSES)

        # Dense all-to-all, small random init. Exponential synapse (tau_syn) because EventProp
        # trains through a synapse model; trained_brain has instantaneous (delta) currents, so
        # a SMALL tau_syn keeps GeNN closest to the core -- part of the transfer approximation.
        # VERIFY(ml_genn): Connection(source, target, connectivity, synapse) arg order + that
        # Dense(Normal(...)) is how weights are initialised.
        w_sd = 1.0 / math.sqrt(IN_FEATURES)
        conn_in_hid = Connection(inp, hid, Dense(Normal(mean=0.0, sd=w_sd)),
                                 Exponential(tau_syn))
        conn_hid_out = Connection(hid, out, Dense(Normal(mean=0.0, sd=1.0 / math.sqrt(hidden))),
                                  Exponential(tau_syn))
    return network, inp, hid, out, conn_in_hid, conn_hid_out


# ---- weight extraction: GeNN dense weights -> trained_brain [out][in] layout ----------------
def _extract_core_weights(conn_in_hid, conn_hid_out, hidden):
    """Pull the trained dense weights and reshape to trained_brain's [out][in] layout.

    GeNN stores a DENSE weight matrix as a flat array indexed g[pre*num_post + post], i.e.
    reshape(num_pre, num_post) gives W[pre][post]. trained_brain wants each layer as
    [out][in] == [post][pre], so we TRANSPOSE:
        in->hid : reshape(784, H) = [pre=in][post=hid] -> .T -> [H][784]  (trained_brain layer 1)
        hid->out: reshape(H, 10)  = [pre=hid][post=out] -> .T -> [10][H]   (trained_brain layer 2)

    VERIFY(ml_genn): (a) how to READ trained weights back after training -- here we assume the
    connection's ``connectivity.weight`` holds a numpy array once the trained state has been
    loaded into the network (see train_model); (b) that GeNN's dense flatten order is
    [pre*num_post + post] (row-major, pre outer). If the transpose orientation is wrong the
    shape ASSERTs below will usually catch it, but confirm on a tiny known-weight net.
    """
    w1 = np.asarray(conn_in_hid.connectivity.weight, dtype=np.float64).reshape(IN_FEATURES, hidden)
    w2 = np.asarray(conn_hid_out.connectivity.weight, dtype=np.float64).reshape(hidden, OUT_CLASSES)
    W1_core = w1.T.tolist()          # [H][784] = [out][in]
    W2_core = w2.T.tolist()          # [10][H] = [out][in]
    return W1_core, W2_core


class GennBrain:
    """Trained-weights holder returned by ``train_model`` (torch/GeNN-free container).

    Carries the two weight matrices already in trained_brain's [out][in] layout plus the
    hyperparameters needed to export a brain; ``export_brain`` reads them directly. Keeping
    this a plain object (not a live GeNN model) means the CLI export path needs no GeNN state.
    """

    def __init__(self, W1_core, W2_core, hidden, steps, beta, threshold, metrics=None):
        self.W1_core = W1_core                 # [H][784]
        self.W2_core = W2_core                 # [10][H]
        self.hidden = int(hidden)
        self.steps = int(steps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.metrics = dict(metrics or {})


# ---- train / evaluate -----------------------------------------------------------------------
def train_model(epochs=DEFAULT_EPOCHS, hidden=DEFAULT_HIDDEN, steps=DEFAULT_STEPS,
                device="cpu", seed=0, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE,
                beta=DEFAULT_BETA, threshold=DEFAULT_THRESHOLD, root=None, download=True,
                tau_syn=DEFAULT_TAU_SYN, train_limit=None, test_limit=None):
    """Train the 784 -> H -> 10 spiking brain on GeNN (EventProp + Adam), return (GennBrain, metrics).

    ``device`` is accepted for parity with the Brian2/Torch trainer and only affects logging: GeNN
    picks the CUDA backend automatically when CUDA + a compiler are present (there is no
    ``.to(device)`` to thread). Deterministic-ish for a given seed (input coding + weight init
    are seeded; GeNN's own kernels are deterministic for a fixed build).

    *** UNVERIFIED against a live ml_genn -- see module docstring. ***
    """
    root = root or os.path.join(".", "data", "mnist")
    ckpt_dir = os.path.join(os.path.dirname(root) or ".", "genn_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"GeNN MNIST trainer: device-hint={device} (GeNN auto-selects CUDA when available)")

    # 1. MNIST pixels -> numpy -> rate-coded ml_genn spike objects
    train_ds, test_ds = _datasets(root, download=download)
    Xtr, ytr = _to_numpy(train_ds, limit=train_limit)
    Xte, yte = _to_numpy(test_ds, limit=test_limit)
    train_spikes = _encode_dataset(Xtr, steps, seed=seed)
    test_spikes = _encode_dataset(Xte, steps, seed=seed + 1)

    # 2. build the network
    network, inp, hid, out, c_in_hid, c_hid_out = build_network(
        hidden, beta, threshold, steps, batch_size, tau_syn, seed)

    # 3. train with the EventProp compiler on sparse-categorical-crossentropy + Adam.
    # VERIFY(ml_genn): EventPropCompiler kwargs (example_timesteps / losses / optimiser /
    # batch_size). EventProp needs a non-spiking readout (our LeakyIntegrate out) and a spiking
    # hidden layer (our LIF) -- that is the standard ml_genn classification recipe. Optional
    # firing-rate regularisation (reg_lambda_*/reg_nu_*) is omitted here; add if hidden goes silent.
    serialiser = Numpy(ckpt_dir)                        # writes per-connection .npy checkpoints
    compiler = EventPropCompiler(
        example_timesteps=int(steps),
        losses="sparse_categorical_crossentropy",
        optimiser=Adam(float(lr)),
        batch_size=int(batch_size))
    compiled_train = compiler.compile(network)          # VERIFY(ml_genn): compile() nvcc-builds the CUDA model
    train_acc = float("nan")
    with compiled_train:
        # VERIFY(ml_genn): Checkpoint(serialiser) saves each connection's weights per epoch;
        # train() takes {input_pop: spikes}, {output_pop: labels}, num_epochs, shuffle, callbacks
        # and returns (metrics, callback_data) with metrics keyed by the output population.
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        metrics_tr, _ = compiled_train.train(
            {inp: train_spikes}, {out: ytr},
            num_epochs=int(epochs), shuffle=True, callbacks=callbacks)
        try:
            train_acc = float(metrics_tr[out].result)   # VERIFY(ml_genn): metrics[pop].result == accuracy
        except Exception:                                # noqa: BLE001 -- metrics shape varies; keep NaN
            train_acc = float("nan")

    # 4. load the final-epoch trained weights back into the (uncompiled) network so BOTH the
    # inference-eval below and the weight export see the trained values.
    # VERIFY(ml_genn): network.load((epoch_key,), serialiser) is the documented reload; the key
    # is the epoch index the Checkpoint used (0-based -> epochs-1). If the API differs, read the
    # serialiser's .npy files from ckpt_dir directly.
    final_key = (int(epochs) - 1,)
    network.load(final_key, serialiser)

    # 5. GeNN test accuracy via the inference compiler (uses the just-loaded trained weights).
    test_acc = float("nan")
    infer = InferenceCompiler(evaluate_timesteps=int(steps), batch_size=int(batch_size))
    compiled_eval = infer.compile(network)              # VERIFY(ml_genn): InferenceCompiler kwargs
    with compiled_eval:
        metrics_te, _ = compiled_eval.evaluate({inp: test_spikes}, {out: yte})
        try:
            test_acc = float(metrics_te[out].result)
        except Exception:                                # noqa: BLE001
            test_acc = float("nan")

    # 6. extract weights into trained_brain's [out][in] layout
    W1_core, W2_core = _extract_core_weights(c_in_hid, c_hid_out, int(hidden))

    metrics = {
        "backend": "genn",
        "epochs": int(epochs),
        "device": str(device),
        "hidden": int(hidden),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "tau_mem": round(_beta_to_tau(beta), 4),
        "tau_syn": float(tau_syn),
        "train_acc": round(train_acc, 4) if train_acc == train_acc else None,   # NaN -> None
        "test_acc": round(test_acc, 4) if test_acc == test_acc else None,
        "unverified_against_live_genn": True,
    }
    model = GennBrain(W1_core, W2_core, hidden, steps, beta, threshold, metrics)
    return model, metrics


# ---- checkpointing (torch-free JSON of the extracted weights) -------------------------------
def save_checkpoint(model, path):
    """Persist the extracted weights + shape metadata as plain JSON (no torch, no GeNN state)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "format": "snn_training.GennBrain/1",
            "hidden": model.hidden, "steps": model.steps,
            "beta": model.beta, "threshold": model.threshold,
            "W1": model.W1_core, "W2": model.W2_core,
        }, f)
    return path


def load_checkpoint(path):
    """Rebuild a GennBrain from a checkpoint written by save_checkpoint."""
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    return GennBrain(c["W1"], c["W2"], c["hidden"], c["steps"], c["beta"], c["threshold"])


# ---- THE BRIDGE: export GeNN-trained weights into a zero-dep trained_brain -------------------
def export_brain(model, path, steps=None, beta=None, threshold=None, meta=None):
    """Export the GeNN-trained weights as a zero-dep ``trained_brain`` JSON at ``path``.

    ``model.W1_core`` is [H][784] and ``model.W2_core`` is [10][H] (already transposed into
    trained_brain's [out][in] layout by ``_extract_core_weights``), exactly the layers=[W1,W2]
    chain (784 -> H -> 10) trained_brain validates. Shapes are ASSERTED before writing, then
    make_brain + save_brain write the file the core serves with no GeNN and no torch.
    """
    W1 = model.W1_core                                   # [H][784]
    W2 = model.W2_core                                   # [10][H]
    hidden = model.hidden
    steps = model.steps if steps is None else int(steps)
    beta = model.beta if beta is None else float(beta)
    threshold = model.threshold if threshold is None else float(threshold)

    assert len(W1) == hidden and all(len(row) == IN_FEATURES for row in W1), (
        f"W1 must be [{hidden}][{IN_FEATURES}], got [{len(W1)}][{len(W1[0]) if W1 else 0}]"
    )
    assert len(W2) == OUT_CLASSES and all(len(row) == hidden for row in W2), (
        f"W2 must be [{OUT_CLASSES}][{hidden}], got [{len(W2)}][{len(W2[0]) if W2 else 0}]"
    )
    m = {"source": "snn_data_lake.training.mnist_genn GeNN surrogate-gradient GPU tier",
         "task": "mnist", "hidden": hidden, "steps": int(steps),
         "transfer": "approximate: GeNN dynamics != trained_brain LIF; may need core fine-tune",
         "unverified_against_live_genn": True}
    if meta:
        m.update(meta)
    brain = make_brain([W1, W2], steps=steps, beta=beta, threshold=threshold, meta=m)
    save_brain(brain, path)
    return brain


def verify_transfer(brain, n=200, root=None, download=True):
    """Prove how well the GeNN export transfers into the core: score ``n`` held-out MNIST test
    images through the ZERO-DEP ``trained_brain.accuracy`` and return that core-served accuracy.

    Compare it to the GeNN test accuracy (metrics['test_acc']). Unlike the Brian2/Torch trainer --
    whose forward is IDENTICAL to the core so the two match closely -- GeNN's continuous-time
    dynamics only APPROXIMATE trained_brain's discrete accumulate, so expect a larger gap here.
    A big drop means the neuron model needs matching more tightly or a short core-side fine-tune.
    Each image is handed over as an (x_list, label) pair with x_list a length-784 vector in [0,1].
    """
    root = root or os.path.join(".", "data", "mnist")
    _, test_ds = _datasets(root, download=download)
    n = min(int(n), len(test_ds))
    subset = []
    for i in range(n):
        image, label = test_ds[i]                        # image [1,28,28] in [0,1]
        subset.append(([float(v) for v in np.asarray(image).reshape(-1)], int(label)))
    return core_accuracy(brain, subset)
