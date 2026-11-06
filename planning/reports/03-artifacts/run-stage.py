"""One command: bounded exploratory run, offline report, then conditional power plan."""

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id")
    mode.add_argument("--records", type=Path, help="Recompute existing records without any training")
    p.add_argument("--output", type=Path, help="New analysis directory for --records")
    args = p.parse_args()
    commands = []
    if args.records:
        if args.output is None:
            p.error("--records requires a new --output directory")
        folder, output = args.records.resolve(), args.output.resolve()
    else:
        if not args.run_id.replace("-", "").isalnum():
            p.error("Use a new alphanumeric run-id")
        folder = HERE / args.run_id
        output = folder / "audit"
        commands.append(["evaluate.py", "--run-id", args.run_id])
    commands += [
        ["recompute.py", str(folder), "--output", str(output)],
        ["power.py", str(output / "summary.json"), "--output", str(output / "power.json")],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-B", str(HERE / command[0]), *command[1:]], cwd=ROOT, check=True, timeout=1200)
    print("Diagnostic report:", output / "table.md")


if __name__ == "__main__":
    main()
