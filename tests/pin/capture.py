"""FR-017 bit-exact numeric pin -- CAPTURE side.

WHY THIS EXISTS. The e-prop kernel MERGE (spec 013) rests on byte-identity everywhere except one
place: the `flatten_frames` collapse (FR-007). The existing cosine assert CANNOT verify that
collapse, because `bptt_grads` computes its reference through the SAME `forward`, therefore the
same `flatten_frames` -- a defect there moves BOTH sides together and the cosine stays high. That
is Standing Invariant 5 ("a green gate that structurally cannot see the thing it checks is worse
than no gate"). This pin is the verifier that CAN see it.

LOCAL ONLY, DELIBERATELY. This is not a CI assert. The local gate runs torch 2.11.0+cu128 on a
free-threaded 3.14 interpreter; CI has no GPU and installs CPU torch, so bit-exact float equality
across the two is not a contract this project can hold (research.md D-3). The pin's job is to catch
a mistake made by THIS lane on THIS machine between two commits minutes apart -- same interpreter,
same build, same seeds -- which is exactly where bit-exactness is meaningful and cheap.

WHY IT LIVES IN tests/. `tools/p4c_ratchet.py` counts a .py under specs/ as an ungoverned BUILD and
reddens on it -- correctly, per P4c. This is not a build: it is "the instrument that measures a
build", which is the exact category p4c already excludes `tests/**` under. Moved here rather than
adding an exclusion, because widening a gate to admit one's own artifact is how a check gets
silenced (Standing Invariant 2). It is not named test_* so pytest does not collect it; it is a
manual instrument, deliberately not a CI assert (see D-3 above).

Usage:
    .venv-lake/Scripts/python.exe tests/pin/capture.py --tag before
    .venv-lake/Scripts/python.exe tests/pin/capture.py --tag after
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# The vision consumer is in-package. The audio consumer is at training/ until the C3 move, and in
# cortex/ after it -- try the governed path FIRST so this script keeps working across that commit
# without editing. Recorded in the payload as `shd_module` so a comparison can never silently
# straddle the move.
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

SEED = 20260812
DEVICE = "cpu"  # pin on CPU: deterministic across the cuda/no-cuda split, and the merge is not a
# compute-placement change. Recorded in the payload regardless.


def _load_shd():
    """Return (module, dotted_name). Governed path first, root path second."""
    try:
        import snn_data_lake.sensorimotor.eprop_shd as m

        return m, "snn_data_lake.sensorimotor.eprop_shd"
    except ImportError:
        import training.eprop_shd as m

        return m, "training.eprop_shd"


def _snap(v):
    """Detach/clone whatever shape a gradient carrier arrives in.

    `eprop_grads` returns (grads, loss, spikes) and `bptt_grads` returns a DICT keyed W_in/W_rec/
    W_out/b_out -- see `eprop_grads` in each consumer and `bptt_grads` in
    `snn_data_lake.cortex.eprop_kernel`. Handling both rather than assuming a list is what keeps
    this harness working across the merge, which is the point of pinning.

    Cited by SYMBOL, not by line: this docstring named `eprop_nmnist.py:343` and `:387`, exact when
    written at C0 and both rotted by C4 -- which moved `bptt_grads` out of that file entirely. The
    rot went unnoticed through three ticked FR-020 re-derivation passes because those swept the
    commits' cites, not the harness's own. Caught by the specs/013 pre-push review.
    """
    if isinstance(v, dict):
        return {k: _snap(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_snap(x) for x in v]
    if torch.is_tensor(v):
        return v.detach().clone()
    return torch.as_tensor(float(v))


def _capture_one(mod, net, x, y, label):
    """Everything the merge could silently move. All six come from the SAME net and input."""
    torch.manual_seed(SEED)
    logits_sum, latent_sum = net.forward(x)
    e_grads, e_loss, e_spikes = mod.eprop_grads(net, x, y)
    b_grads = mod.bptt_grads(net, x, y)
    return {
        "label": label,
        "forward.logits_sum": _snap(logits_sum),
        "forward.latent_sum": _snap(latent_sum),
        "eprop_grads.grads": _snap(e_grads),
        "eprop_grads.loss": _snap(e_loss),
        "eprop_grads.spikes": _snap(e_spikes),
        "bptt_grads.grads": _snap(b_grads),
    }


def _reshape_defect(net):
    """T009b: simulate the ACTUAL defect class this pin exists to catch.

    A wrong `flatten_frames` does not nudge a value -- it changes WHICH input element lands on
    which input unit. So the honest sensitivity test is a permutation of the flattened feature
    axis, not an epsilon on one element. (A one-element nudge leaves nmnist's forward outputs
    unchanged, because they are spike COUNTS and the nudge is subthreshold -- which would leave
    2 of 6 tensors per module unexercised and the control overclaimed.)

    Monkeypatches the instance only; the caller discards the net afterwards. No file is edited.
    """
    orig = net.flatten_frames

    def permuted(x):
        flat = orig(x)
        # reverse the feature axis
        idx = torch.arange(flat.shape[-1] - 1, -1, -1, device=flat.device)
        return flat.index_select(-1, idx)

    net.flatten_frames = permuted
    return net


def capture(tag, perturb=False, reshape=False):
    import snn_data_lake.sensorimotor.eprop_nmnist as nm

    shd, shd_name = _load_shd()

    out_dir = _HERE
    written = []

    # ---- vision -------------------------------------------------------------
    torch.manual_seed(SEED)
    net = nm.EpropNet(seed=SEED)
    if reshape:
        net = _reshape_defect(net)
    xs, ys = nm._make_synthetic(
        n_per_class=1,
        n_classes=nm.NUM_CLASSES,
        t_steps=nm.DEFAULT_T,
        in_ch=nm.IN_CH,
        h=nm.SENSOR_H,
        w=nm.SENSOR_W,
        seed=SEED,
    )
    x, y = xs.to(DEVICE), ys.to(DEVICE)
    if perturb:
        # T009 sensitivity check: move ONE element. A comparator that has never reported a
        # difference is not a control.
        x = x.clone()
        x.view(-1)[0] = x.view(-1)[0] + 1.0
    payload = {
        "meta": {
            "torch": torch.__version__,
            "device": DEVICE,
            "seed": SEED,
            "cuda_available": torch.cuda.is_available(),
            "shd_module": shd_name,
            "perturbed": bool(perturb),
            "reshape_defect": bool(reshape),
        },
        "data": _capture_one(nm, net, x, y, "nmnist"),
    }
    p = os.path.join(out_dir, f"nmnist-{tag}.pt")
    torch.save(payload, p)
    written.append(p)

    # ---- audio --------------------------------------------------------------
    torch.manual_seed(SEED)
    anet = shd.EpropAudioNet(seed=SEED)
    if reshape:
        anet = _reshape_defect(anet)
    axs, ays = shd._make_synthetic(
        n_per_class=1,
        n_classes=shd.NUM_CLASSES,
        t_steps=shd.DEFAULT_T,
        channels=shd.CHANNELS,
        seed=SEED,
    )
    ax, ay = axs.to(DEVICE), ays.to(DEVICE)
    if perturb:
        ax = ax.clone()
        ax.view(-1)[0] = ax.view(-1)[0] + 1.0
    payload = {
        "meta": {
            "torch": torch.__version__,
            "device": DEVICE,
            "seed": SEED,
            "cuda_available": torch.cuda.is_available(),
            "shd_module": shd_name,
            "perturbed": bool(perturb),
            "reshape_defect": bool(reshape),
        },
        "data": _capture_one(shd, anet, ax, ay, "shd"),
    }
    p = os.path.join(out_dir, f"shd-{tag}.pt")
    torch.save(payload, p)
    written.append(p)

    for w in written:
        print(f"wrote {os.path.relpath(w, _ROOT)}")
    print(
        f"torch={torch.__version__} device={DEVICE} seed={SEED} "
        f"shd_module={shd_name} perturbed={perturb} reshape_defect={reshape}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FR-017 bit-exact pin -- capture side")
    ap.add_argument("--tag", required=True, help="before | after | probe")
    ap.add_argument(
        "--perturb",
        action="store_true",
        help="T009a: shift ONE input element, to prove the comparator can report a difference",
    )
    ap.add_argument(
        "--reshape-defect",
        action="store_true",
        dest="reshape",
        help="T009b: permute flatten_frames' feature axis -- the ACTUAL defect class this pin "
        "exists to catch. Every pinned tensor must move.",
    )
    a = ap.parse_args()
    capture(a.tag, perturb=a.perturb, reshape=a.reshape)
