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
        description="Train a surrogate-gradient spiking classifier on GPU and export "
                    "the weights back into the zero-dep snn_classifier core.",
    )
    parser.add_argument("--gpu", action="store_true",
                        help="request CUDA (falls back to CPU if unavailable)")
    parser.add_argument("--mnist", action="store_true",
                        help="train the 784->H->10 MNIST brain (torchvision MNIST) whose forward "
                             "matches trained_brain, then export a zero-dep brain the core serves "
                             "with no torch (use with --export-brain)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="training epochs (default 50; a few suffice for --mnist)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="MNIST brain hidden-layer width (default 256; --mnist only)")
    parser.add_argument("--steps", type=int, default=25,
                        help="MNIST rate-coding timesteps (default 25; --mnist only)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="save a torch checkpoint (state_dict) to PATH")
    parser.add_argument("--export-core", metavar="PATH", dest="export_core",
                        help="export learned weights as a zero-dep snn_classifier "
                             "weight file at PATH (loadable with no torch present)")
    parser.add_argument("--export-brain", metavar="PATH", dest="export_brain",
                        help="export the trained MNIST brain as a zero-dep trained_brain JSON at "
                             "PATH (loadable + servable with no torch present; --mnist only)")
    parser.add_argument("--offload", action="store_true",
                        help="use CPU/disk-offload training (gradient checkpointing "
                             "+ CPU optimizer state) for models larger than VRAM")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args(argv)

    # Lazy, guarded import: torch/snntorch (and the torch-based training modules) are
    # only touched here. A missing extra -> one clean line, not an ImportError trace.
    try:
        import torch  # noqa: F401
        from snn_data_lake.training import device as devmod
        from snn_data_lake.training import trainer
        from snn_data_lake.training import offload as offloadmod
        from snn_data_lake.training import mnist as mnistmod
    except ImportError:
        print('snn-train needs the training extra: pip install ".[train]"')
        sys.exit(1)

    dev = devmod.enable(devmod.resolve_device("cuda" if args.gpu else None))

    # ---- MNIST brain path: 784->H->10, forward == trained_brain, exports a zero-dep brain ----
    if args.mnist:
        model, metrics = mnistmod.train_model(
            epochs=args.epochs, hidden=args.hidden, steps=args.steps,
            device=dev, seed=args.seed)
        print("training metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")
        if args.checkpoint:
            mnistmod.save_checkpoint(model, args.checkpoint)
            print(f"checkpoint saved: {args.checkpoint}")
        if args.export_brain:
            brain = mnistmod.export_brain(
                model, args.export_brain, steps=args.steps,
                meta={"epochs": args.epochs, "device": str(dev)})
            print(f"exported brain: {args.export_brain}")
            # honesty win: prove the export transfers by re-scoring MNIST through the ZERO-DEP core
            core_acc = mnistmod.verify_transfer(brain, n=200)
            print(f"transfer check: torch test_acc={metrics['test_acc']:.4f}  "
                  f"core-served acc (200 imgs)={core_acc:.4f}")
        return 0

    if args.offload:
        model, metrics = offloadmod.train_with_offload(
            epochs=args.epochs, compute_device=dev, seed=args.seed)
    else:
        model, metrics = trainer.train_model(
            epochs=args.epochs, device=dev, seed=args.seed)

    print("training metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    if args.checkpoint:
        trainer.save_checkpoint(model, args.checkpoint)
        print(f"checkpoint saved: {args.checkpoint}")

    if args.export_core:
        trainer.export_to_core(
            model, args.export_core,
            meta={"epochs": args.epochs, "device": str(dev), "offload": bool(args.offload)})
        print(f"exported core weights: {args.export_core}")

    return 0


if __name__ == "__main__":
    main()
