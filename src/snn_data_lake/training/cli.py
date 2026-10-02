"""``snn-train`` console script -- entrypoint for the GPU training tier.

torch/trainer are imported LAZILY inside main(), wrapped so that if the training
extra is not installed the user gets a single clean line instead of an ImportError
traceback:

    snn-train needs the training extra: pip install ".[train]"

so running ``snn-train`` in a plain zero-dep install never crashes with a stack trace.
"""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="snn-train",
        description="Train the MNIST brain on the GeNN backend (--mnist --genn). The BPTT trainers "
                    "(default readout, --offload, --mnist without --genn, --lm) were retired 2026-10-02.",
    )
    parser.add_argument("--gpu", action="store_true",
                        help="request CUDA (falls back to CPU if unavailable)")
    parser.add_argument("--mnist", action="store_true",
                        help="with --genn: train the 784->H->10 MNIST brain and export a zero-dep "
                             "trained_brain (use with --export-brain); the BPTT --mnist path was "
                             "retired 2026-10-02")
    parser.add_argument("--genn", action="store_true",
                        help="with --mnist, train on the GeNN CUDA backend (code-generated, faster "
                             'for SNNs) instead of the Brian2/Torch bridge; needs the GeNN extra '
                             '(pip install ".[genn]"). '
                             "GeNN dynamics only approximate the core, so transfer is approximate.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="training epochs (default 50; a few suffice for --mnist)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="MNIST brain hidden-layer width (default 256; --mnist only)")
    parser.add_argument("--steps", type=int, default=25,
                        help="MNIST rate-coding timesteps (default 25; --mnist only)")
    parser.add_argument("--lm", action="store_true",
                        help="the char-level spiking LM trainer (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--corpus", metavar="PATH",
                        help="the LM's training text (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--export-lm", metavar="PATH", dest="export_lm",
                        help="the LM's snn_lm export; its serve core was retired too (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--d-model", type=int, default=96, dest="d_model",
                        help="LM width (--lm only; retired)")
    parser.add_argument("--blocks", type=int, default=2,
                        help="LM depth (--lm only; retired)")
    parser.add_argument("--seq-len", type=int, default=64, dest="seq_len",
                        help="LM unroll length (--lm only; retired)")
    parser.add_argument("--resume", metavar="PATH",
                        help="resume the LM from a checkpoint (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="save a torch checkpoint (state_dict) to PATH")
    parser.add_argument("--export-core", metavar="PATH", dest="export_core",
                        help="the default readout trainer's snn_classifier weight export (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--export-brain", metavar="PATH", dest="export_brain",
                        help="export the trained MNIST brain as a zero-dep trained_brain JSON at "
                             "PATH (loadable + servable with no torch present; --mnist only)")
    parser.add_argument("--offload", action="store_true",
                        help="CPU/disk-offload BPTT training (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args(argv)

    # Every flag whose only path was a retired BPTT trainer refuses BEFORE any dispatch: the GeNN path
    # silently ignored them and exited 0, which reads as honoured (Codex panel, 2026-10-02).
    retired = [flag for flag, on in (("--lm", args.lm), ("--offload", args.offload),
                                     ("--export-core", args.export_core), ("--export-lm", args.export_lm),
                                     ("--corpus", args.corpus), ("--resume", args.resume)) if on]
    if retired:
        print(f"snn-train {' '.join(retired)}: RETIRED 2026-10-02 with the BPTT trainers (owner: "
              "port-then-retire) -- see docs/TRAINING.md and docs/SPIKING_LM.md")
        sys.exit(2)

    # Lazy, guarded import: Torch/Brian2 (and the Torch-based training modules) are
    # only touched here. A missing extra -> one clean line, not an ImportError trace.
    try:
        import torch  # noqa: F401
        from snn_data_lake.infra import device as devmod
    except ImportError:
        print('snn-train needs the training extra: pip install ".[train]"')
        sys.exit(1)

    # resolve the device but do NOT set_default_device: every trainer places its tensors
    # explicitly with .to(device), and a global default of 'cuda' breaks the DataLoader's
    # CPU index sampler (Generator device mismatch). Explicit placement is the robust path.
    dev = devmod.resolve_device("cuda" if args.gpu else None)

    # ---- MNIST brain path: 784->H->10, forward == trained_brain, exports a zero-dep brain ----
    if args.mnist:
        # GeNN CUDA backend: nested guarded import so a missing ml_genn -> one clean line
        # (its own [genn] message), NOT an ImportError traceback. mnist_genn imports ml_genn at
        # its module top, so it is imported ONLY here, only when --genn is requested.
        if args.genn:
            try:
                from snn_data_lake.training import mnist_genn as gennmod
            except ImportError:
                print('snn-train --genn needs the GeNN extra: pip install ".[genn]"')
                sys.exit(1)
            model, metrics = gennmod.train_model(
                epochs=args.epochs, hidden=args.hidden, steps=args.steps,
                device=dev, seed=args.seed)
            print("training metrics:")
            for key, val in metrics.items():
                print(f"  {key}: {val}")
            if args.checkpoint:
                gennmod.save_checkpoint(model, args.checkpoint)
                print(f"checkpoint saved: {args.checkpoint}")
            if args.export_brain:
                brain = gennmod.export_brain(
                    model, args.export_brain, steps=args.steps,
                    meta={"epochs": args.epochs, "device": str(dev), "backend": "genn"})
                print(f"exported brain: {args.export_brain}")
                # honesty win: score the export through the ZERO-DEP core. GeNN dynamics only
                # approximate trained_brain, so expect a wider gap than the Brian2/Torch path.
                core_acc = gennmod.verify_transfer(brain, n=200)
                gt = metrics.get("test_acc")
                gt_str = f"{gt:.4f}" if isinstance(gt, float) else str(gt)
                print(f"transfer check: GeNN test_acc={gt_str}  "
                      f"core-served acc (200 imgs)={core_acc:.4f}")
            return 0

        print("snn-train --mnist (without --genn) was RETIRED 2026-10-02 (owner: port-then-retire): "
              "see docs/TRAINING.md")
        sys.exit(2)

    print("snn-train's default 8x8 readout trainer was RETIRED 2026-10-02 (owner: port-then-retire): "
          "readout/snn_classifier's LOCAL rule trains the same task -- see docs/TRAINING.md")
    sys.exit(2)


if __name__ == "__main__":
    main()
