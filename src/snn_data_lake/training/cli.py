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
    parser.add_argument("--epochs", type=int, default=50,
                        help="training epochs (default 50)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="save a torch checkpoint (state_dict) to PATH")
    parser.add_argument("--export-core", metavar="PATH", dest="export_core",
                        help="export learned weights as a zero-dep snn_classifier "
                             "weight file at PATH (loadable with no torch present)")
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
    except ImportError:
        print('snn-train needs the training extra: pip install ".[train]"')
        sys.exit(1)

    dev = devmod.enable(devmod.resolve_device("cuda" if args.gpu else None))

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
