"""``snn-train`` console script -- every trainer behind it is retired; each path refuses with a message."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="snn-train",
        description="RETIRED: every trainer behind snn-train (BPTT 2026-10-02, GeNN/EventProp 2026-10-06) "
                    "refuses; see docs/TRAINING.md.",
    )
    parser.add_argument("--gpu", action="store_true",
                        help="request CUDA (falls back to CPU if unavailable)")
    parser.add_argument("--mnist", action="store_true",
                        help="the MNIST brain trainer (RETIRED 2026-10-02; refuses)")
    parser.add_argument("--genn", action="store_true",
                        help="the GeNN/EventProp MNIST trainer (RETIRED 2026-10-06; refuses)")
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
    parser.add_argument("--d-model", type=int, default=None, dest="d_model",
                        help="LM width (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="LM depth (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
    parser.add_argument("--seq-len", type=int, default=None, dest="seq_len",
                        help="LM unroll length (RETIRED 2026-10-02 with the BPTT trainers; refuses)")
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

    # Every path is retired: each refuses BEFORE any dispatch, so no flag is silently ignored and
    # exits 0, which reads as honoured (Codex panel, 2026-10-02). The GeNN path went last (2026-10-06).
    retired = [flag for flag, on in (("--lm", args.lm), ("--offload", args.offload),
                                     ("--export-core", args.export_core), ("--export-lm", args.export_lm),
                                     ("--corpus", args.corpus), ("--resume", args.resume),
                                     ("--d-model", args.d_model is not None),
                                     ("--blocks", args.blocks is not None),
                                     ("--seq-len", args.seq_len is not None),
                                     ("--mnist", args.mnist), ("--genn", args.genn)) if on]
    if retired:
        print(f"snn-train {' '.join(retired)}: RETIRED (BPTT trainers 2026-10-02, GeNN/EventProp trainer "
              "2026-10-06; owner: port-then-retire) -- see docs/TRAINING.md and docs/SPIKING_LM.md")
        sys.exit(2)

    print("snn-train's default 8x8 readout trainer was RETIRED 2026-10-02 (owner: port-then-retire): "
          "readout/snn_classifier's LOCAL rule trains the same task -- see docs/TRAINING.md")
    sys.exit(2)


if __name__ == "__main__":
    main()
