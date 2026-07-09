"""Shared GPU/device selection for the research torch training scripts.

Opt-in GPU acceleration, matching the pattern in eth_mnist_bindsnet.py: set
`NORD_GPU=1` (or `NORD_DEVICE=cuda` / `cuda:0`) to move training onto CUDA when it
is available. The DEFAULT is CPU, so a plain `python research/<script>.py` run is
byte-identical to before (same numbers of record). If CUDA is requested but not
available, it falls back to CPU with a clear message rather than crashing.

Usage (at the top of a script's main(), before any tensor/model/data is created):

    from torch_device import resolve_device, enable
    device = enable(resolve_device())   # routes torch factories to `device`

`enable()` calls torch.set_default_device(device) for a CUDA device, so every
tensor and module built afterwards lands on the GPU without threading `.to(device)`
through every call site; for CPU it is a no-op (default is already CPU).

torch is imported lazily inside the functions, so importing this module never
requires torch (keeps it harmless in the zero-dep environment).
"""
import os


def resolve_device(explicit=None):
    """Return 'cuda[:N]' if GPU is requested (arg / NORD_DEVICE / NORD_GPU=1) AND
    torch.cuda.is_available(); otherwise 'cpu'. Prints the resolved GPU or a clear
    fallback warning."""
    import torch

    want = explicit or os.environ.get("NORD_DEVICE")
    if not want and os.environ.get("NORD_GPU") == "1":
        want = "cuda"
    if not want or want == "cpu":
        return "cpu"
    if want.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} "
                  f"(CUDA {torch.version.cuda}, torch {torch.__version__}) -> device={want}")
            return want
        print(f"WARNING: device '{want}' requested but torch.cuda.is_available() is False; "
              "falling back to CPU.")
        return "cpu"
    return want


def enable(device):
    """Route torch tensor/module factories to `device`. No-op for 'cpu' (already the
    default), so the CPU path is unchanged. Returns `device` for convenience."""
    import torch

    if device != "cpu" and hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    return device
