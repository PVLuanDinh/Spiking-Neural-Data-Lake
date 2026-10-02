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
                        help="MNIST rate-coding timesteps (default 25; --mnist only), or "
                             "LM training steps (--lm)")
    parser.add_argument("--lm", action="store_true",
                        help="train the char-level SPIKING language model (training-tier "
                             "spiking_lm, SpikeGPT lineage) on --corpus (default: a built-in "
                             "toy paragraph), then export a zero-dep snn_lm the core serves "
                             "with no torch (use with --export-lm)")
    parser.add_argument("--corpus", metavar="PATH",
                        help="UTF-8 text file to train the LM on (--lm only; default: the "
                             "built-in toy paragraph)")
    parser.add_argument("--export-lm", metavar="PATH", dest="export_lm",
                        help="export the trained spiking LM as a zero-dep snn_lm JSON at PATH "
                             "(served by snn_data_lake.sensorimotor.spiking_lm_serve with no torch present; "
                             "--lm only)")
    parser.add_argument("--d-model", type=int, default=96, dest="d_model",
                        help="LM width (default 96; --lm only)")
    parser.add_argument("--blocks", type=int, default=2,
                        help="LM spiking-block depth (default 2; --lm only)")
    parser.add_argument("--seq-len", type=int, default=64, dest="seq_len",
                        help="LM context/BPTT unroll length (default 64; --lm only)")
    parser.add_argument("--resume", metavar="PATH",
                        help="resume the LM from a checkpoint written by --checkpoint "
                             "(--lm only; model config comes from the checkpoint)")
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

    # ---- spiking-LM path: char-level spiking LM -> exports a zero-dep snn_lm ----
    # Nested guard FIRST (before the shared [train] guard) so --lm gets its own clean
    # line. spiking_lm itself imports torch-FREE (its module guard just sets a flag),
    # so the real "is torch here" check is _TORCH_IMPORT_ERROR, not the import.
    if args.lm:
        try:
            from snn_data_lake.training import spiking_lm as lmmod
        except ImportError:
            print('snn-train --lm needs Torch + Brian2: pip install ".[gpu]"')
            sys.exit(1)
        if lmmod._TORCH_IMPORT_ERROR is not None:
            print('snn-train --lm needs Torch + Brian2: pip install ".[gpu]"')
            sys.exit(1)
        from snn_data_lake.infra import device as lm_devmod
        dev = lm_devmod.resolve_device("cuda" if args.gpu else None)

        corpus = lmmod.load_corpus(args.corpus)          # real UTF-8 file, or built-in toy
        tok = lmmod.CharTokenizer(corpus)
        if args.resume:
            model, cfg = lmmod.load_checkpoint(args.resume)
            # honesty guard: the char->id map comes from the corpus tokenizer, so a resumed
            # model is only meaningful on the SAME corpus; vocab size is the cheap cross-check.
            if cfg.vocab_size != tok.vocab_size:
                print(f"--resume vocab mismatch: checkpoint vocab={cfg.vocab_size}, "
                      f"corpus vocab={tok.vocab_size} (resume with the SAME corpus)")
                sys.exit(1)
            print(f"resumed checkpoint: {args.resume} "
                  f"(d_model={cfg.d_model} blocks={cfg.n_blocks} seq_len={cfg.seq_len})")
        else:
            cfg = lmmod.SpikingLMConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                                        n_blocks=args.blocks, seq_len=args.seq_len)
            model = lmmod.SpikingLM(cfg)
        loss = lmmod.train(model, tok, corpus, steps=args.steps, seq_len=cfg.seq_len,
                           device=dev, seed=args.seed)
        ppl = lmmod.perplexity(model, tok, corpus, seq_len=cfg.seq_len, device=dev)
        n_params = sum(p.numel() for p in model.parameters())
        print("training metrics:")
        print(f"  steps: {int(args.steps)}")
        print(f"  device: {dev}")
        print(f"  corpus_chars: {len(corpus)}")
        print(f"  vocab: {tok.vocab_size}")
        print(f"  d_model: {cfg.d_model}")
        print(f"  blocks: {cfg.n_blocks}")
        print(f"  seq_len: {cfg.seq_len}")
        print(f"  params: {int(n_params)}")
        print(f"  final_loss: {loss:.6f}")
        print(f"  perplexity: {ppl:.4f}")
        if args.checkpoint:
            lmmod.save_checkpoint(model, cfg, args.checkpoint)
            print(f"checkpoint saved: {args.checkpoint}")
        if args.export_lm:
            lmmod.export_lm(model, tok, args.export_lm, cfg,
                            meta={"steps": int(args.steps), "device": str(dev),
                                  "corpus_chars": len(corpus),
                                  "final_loss": round(loss, 6),
                                  "perplexity": round(ppl, 4)})
            print(f"exported lm: {args.export_lm}")
            # honesty win: generate THROUGH the zero-dep core serve, so the user SEES the
            # trained LM run with no torch (greedy; may differ slightly from torch sampling).
            sample = lmmod.verify_transfer(args.export_lm, prompt="the ", n=60)
            print("transfer check (greedy, served by zero-dep spiking_lm_serve):")
            print("  " + repr(sample))
        return 0

    # Lazy, guarded import: Torch/Brian2 (and the Torch-based training modules) are
    # only touched here. A missing extra -> one clean line, not an ImportError trace.
    try:
        import torch  # noqa: F401
        from snn_data_lake.infra import device as devmod
        from snn_data_lake.training import trainer
        from snn_data_lake.training import mnist as mnistmod
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
        print("snn-train --offload was RETIRED 2026-10-02 (owner: port-then-retire): a BPTT memory "
              "technique with no e-prop analogue -- see docs/TRAINING.md")
        sys.exit(2)
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
