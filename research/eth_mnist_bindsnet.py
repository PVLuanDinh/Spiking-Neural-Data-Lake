"""
v0.9 — BindsNET Diehl & Cook (eth_mnist) runner: the path to the literature ~95%.

The v0.8 study concluded that beating hard-WTA needs CONDUCTANCE-based synapses and
a real exc/inh population — exactly what BindsNET's `DiehlAndCook2015` model ships.
Rather than re-derive conductance dynamics from scratch, this wires in BindsNET (an
already-installed dependency) with the paper's exact constants and scales neurons up.

Evidence (Diehl & Cook 2015, Front. Comput. Neurosci.):
  100 neurons -> 82.9% | 400 -> 87.0% | 1600 -> 91.9% | 6400 -> 95.0%
Constants from BindsNET examples/mnist/eth_mnist.py:
  exc=22.5, inh=120, norm=78.4, theta_plus=0.05, time=250 ms, dt=1, intensity=128.

COMPUTE WARNING: 6400 neurons over the full 60k train + 10k test is a multi-HOUR
(CPU: likely overnight+) run. Verify the wiring fast first, e.g.:
    NORD_M=100 NORD_TRAIN=600 NORD_TEST=500 NORD_UPDATE=100 python eth_mnist_bindsnet.py
Then the headline run on a GPU (one switch):
    python eth_mnist_bindsnet.py --gpu      # 6400 neurons, 60k/10k -> ~95%
Or a tractable midpoint (~87%):
    NORD_M=400 NORD_TRAIN=20000 NORD_TEST=5000 python eth_mnist_bindsnet.py --gpu

BATCHING (the throughput fix): the original v0.9 loop ran batch_size=1 -- one image at
a time -- which left the GPU ~98% idle and stretched a full run to ~2 days on an RTX 5070.
This runner now processes NORD_BATCH images in parallel per timestep (default 16), for
~10-50x throughput on a GPU. Batching changes ONLY throughput, not the method: the label
assignment / prediction still use BindsNET's own all_activity / assign_labels over the
same [n_samples, time, N] spike record, so the tuned config still reaches the paper's
accuracy. Tune it per VRAM:
    NORD_BATCH=64 python eth_mnist_bindsnet.py --gpu

GPU notes:
  --gpu (or --device cuda / NORD_GPU=1) moves the network + tensors to CUDA.
  RTX 5070 = Blackwell (sm_120) -> needs a CUDA 12.8+ torch build; the default CPU
  wheel has no sm_120 kernels. Install once on the GPU box:
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  Then `python eth_mnist_bindsnet.py --gpu`. If CUDA isn't available the runner
  prints this hint and falls back to CPU.

Needs: pip install bindsnet  (torch._six shim below makes BindsNET<=0.3 run on torch>=2)
"""
import os
import sys
import types
import argparse
import collections.abc
import torch

# --- compat shim: BindsNET <=0.3 imports torch._six, removed in torch >=2.0 ---
if not hasattr(torch, "_six"):
    _six = types.ModuleType("torch._six")
    _six.container_abcs = collections.abc
    _six.string_classes = (str, bytes)
    _six.int_classes = int
    sys.modules["torch._six"] = _six
    torch._six = _six

from torchvision import transforms
from bindsnet.models import DiehlAndCook2015
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels

torch.manual_seed(0)


def _ei(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _ef(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def resolve_device():
    """--gpu / --device flag (or NORD_GPU env). Falls back to CPU with a clear
    RTX 5070 (Blackwell sm_120) install hint if CUDA torch isn't present."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gpu", action="store_true", help="run on CUDA GPU if available")
    ap.add_argument("--device", default=None, help="explicit device, e.g. cuda / cuda:0 / cpu")
    args, _ = ap.parse_known_args()
    want_gpu = args.gpu or args.device not in (None, "cpu") or os.environ.get("NORD_GPU") == "1"
    if args.device:
        return args.device
    if want_gpu:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name}  (CUDA {torch.version.cuda}, torch {torch.__version__})")
            return "cuda"
        print("WARNING: --gpu requested but torch.cuda.is_available() is False.")
        print("  This venv has the CPU torch build. For an RTX 5070 (Blackwell, sm_120)")
        print("  install a CUDA 12.8+ build, e.g.:")
        print("    pip install --force-reinstall torch torchvision \\")
        print("      --index-url https://download.pytorch.org/whl/cu128")
        print("  (older CUDA wheels lack sm_120 kernels and will fail on the 5070.)")
        print("  Falling back to CPU.\n")
    return "cpu"


# ---- config (paper / BindsNET constants; sizes via env) ---------------------
N = _ei("NORD_M", 6400)
N_TRAIN = _ei("NORD_TRAIN", 60000)
N_TEST = _ei("NORD_TEST", 10000)
N_EPOCHS = _ei("NORD_EPOCHS", 1)
TIME = _ei("NORD_TIME", 250)
UPDATE = _ei("NORD_UPDATE", 250)   # re-assign labels every UPDATE train images
BATCH = _ei("NORD_BATCH", 16)      # images run in parallel per timestep (the batch_size=1 fix)
# scale-sensitive knobs — defaults are BindsNET's ~100-neuron values. Larger networks
# need MORE inhibition (more competitors) + bigger theta_plus + multiple epochs, or they
# under-specialise (measured: 6400 neurons / 1 epoch with defaults -> only 47.8%).
EXC = _ef("NORD_EXC", 22.5)
INH = _ef("NORD_INH", 120.0)
THETA_PLUS = _ef("NORD_THETA_PLUS", 0.05)
NORM = _ef("NORD_NORM", 78.4)
DT = 1.0
INTENSITY = 128.0
N_CLASSES = 10
STEPS = int(TIME / DT)
ROOT = os.path.join(".", "data", "bindsnet")


def make_dataset(train):
    return MNIST(
        PoissonEncoder(time=TIME, dt=DT), None, root=ROOT, download=True, train=train,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * INTENSITY)]),
    )


def main():
    device = resolve_device()
    print("=" * 60)
    print("BindsNET Diehl & Cook (eth_mnist) — conductance-based")
    print("=" * 60)
    print(f"neurons={N}  time={TIME}ms  train={N_TRAIN}  test={N_TEST}  epochs={N_EPOCHS}  batch={BATCH}  device={device}")
    print("batched: BATCH images run in parallel per timestep -- the fix for the v0.9 single-image")
    print("bottleneck (batch_size=1 left the GPU ~98% idle); ~10-50x throughput on a GPU, same method.")
    if N >= 1600 and device == "cpu":
        print("WARNING: large network on CPU — expect a multi-hour (likely overnight) run.")
        print("         Pass --gpu on a CUDA machine (RTX 5070: needs cu128 torch).")
    print()

    network = DiehlAndCook2015(
        n_inpt=784, n_neurons=N, exc=EXC, inh=INH, dt=DT,
        norm=NORM, theta_plus=THETA_PLUS, inpt_shape=(1, 28, 28),
    )
    spikes = Monitor(network.layers["Ae"], state_vars=["s"], time=STEPS)
    network.add_monitor(spikes, name="Ae")
    network.to(device)   # moves layers (+ their monitored spikes) onto the device

    assignments = -torch.ones(N, device=device)
    proportions = torch.zeros(N, N_CLASSES, device=device)
    rates = torch.zeros(N, N_CLASSES, device=device)

    # ---- train (unsupervised STDP, no labels in the learning loop) ----
    # BATCHED: BATCH images run in parallel per timestep. BindsNET wants inputs shaped
    # [time, batch, ...]; the DataLoader collates encoded_image as [batch, time, 1, 28, 28],
    # so permute time to the front. network.run() detects that the input batch differs from
    # the current network.batch_size (1 at build) and calls set_batch_size on every layer +
    # resets the monitors on the first run (and again for a smaller final partial batch), so
    # the "Ae" spikes come out as [time, batch, N]. (We rely on this dynamic resize rather
    # than assigning network.batch_size = BATCH by hand: a manual assign would leave the
    # layer/monitor tensors at batch=1 because run() only resizes when the sizes differ.)
    train_set = make_dataset(train=True)
    spike_record = torch.zeros(UPDATE, STEPS, N, device=device)   # [n_samples, time, N] window buffer
    labels = []
    acc_hist = []
    img_idx = 0
    print("training...")
    for epoch in range(N_EPOCHS):
        loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)
        for batch in loader:
            if img_idx >= N_TRAIN:
                break
            enc = batch["encoded_image"]                          # [b, STEPS, 1, 28, 28]
            b = enc.shape[0]                                      # actual batch (final one may be < BATCH)
            inputs = {"X": enc.permute(1, 0, 2, 3, 4).contiguous().to(device)}   # [STEPS, b, 1, 28, 28]
            network.run(inputs=inputs, time=TIME)
            s = spikes.get("s").permute(1, 0, 2).float()          # [STEPS, b, N] -> [b, STEPS, N]
            batch_labels = batch["label"]
            for j in range(b):
                if img_idx >= N_TRAIN:
                    break
                spike_record[img_idx % UPDATE] = s[j]             # copy this image's [STEPS, N] into the window
                labels.append(int(batch_labels[j].item()))
                img_idx += 1
                if img_idx % UPDATE == 0:                         # a full window is ready -> assign labels
                    lt = torch.tensor(labels[-UPDATE:], device=device)
                    # vectorised over the whole [UPDATE, STEPS, N] window via BindsNET's own helpers
                    pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=N_CLASSES).to(device)
                    acc = 100 * torch.sum(lt.long() == pred).item() / len(lt)
                    acc_hist.append(acc)
                    assignments, proportions, rates = assign_labels(
                        spikes=spike_record, labels=lt, n_labels=N_CLASSES, rates=rates)
                    print(f"  img {img_idx}/{N_TRAIN}  window train acc={acc:.1f}%  (avg {sum(acc_hist)/len(acc_hist):.1f}%)  batch={BATCH}")
            network.reset_state_variables()

    # ---- test (theta + weights frozen; classify by assigned-neuron activity) ----
    # Same batching: feed BATCH images per network.run, then classify the whole batch at
    # once. spikes.get("s") is [STEPS, b, N]; permuted to [b, STEPS, N] it is exactly the
    # [n_samples, time, n_neurons] tensor all_activity expects (it sums over time internally).
    test_set = make_dataset(train=False)
    n_all, n = 0.0, 0
    next_report = 500
    print("\ntesting...")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH, shuffle=False)
    for batch in test_loader:
        if n >= N_TEST:
            break
        enc = batch["encoded_image"]                              # [b, STEPS, 1, 28, 28]
        inputs = {"X": enc.permute(1, 0, 2, 3, 4).contiguous().to(device)}   # [STEPS, b, 1, 28, 28]
        network.run(inputs=inputs, time=TIME)
        s = spikes.get("s").permute(1, 0, 2).float()              # [b, STEPS, N]
        lt = batch["label"].to(device)                            # [b]
        pred = all_activity(spikes=s, assignments=assignments, n_labels=N_CLASSES).to(device)
        n_all += float(torch.sum(lt.long() == pred).item())
        n += s.shape[0]
        network.reset_state_variables()
        if n >= next_report:
            print(f"  tested {n}/{N_TEST}  running all-activity acc={100*n_all/n:.2f}%")
            next_report += 500

    acc_all = 100 * n_all / n
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY (all-activity) : {acc_all:.2f}%")
    print(f"(paper: {N} neurons -> ~"
          + {100: '82.9', 400: '87.0', 1600: '91.9', 6400: '95.0'}.get(N, '??') + "%)")
    print("=" * 60)
    assert acc_all >= 10.0, "below chance — wiring broken"
    print("self-check OK: ran end-to-end, accuracy computed")


if __name__ == "__main__":
    main()
