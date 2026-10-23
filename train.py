"""Launch the archived BioSense pipeline with separate input and output folders."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("kanser", "cftr", "pah", "master"))
    parser.add_argument("--objective", choices=("f2", "f1"), default="f2")
    parser.add_argument("--mode", choices=("fast", "default", "ultra"), default="fast")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without training")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    data_file = data_dir / f"YARISMA_TRAIN_{args.dataset.upper()}.csv"
    if not data_file.is_file():
        parser.error(f"Missing local input: {data_file}")
    run_dir = (args.output_dir / f"{args.dataset}_{args.objective}").resolve()
    if run_dir == data_dir or data_dir.is_relative_to(run_dir) or run_dir.is_relative_to(data_dir):
        parser.error("Input and output directories must be separate and must not contain each other")
    pipeline = Path(__file__).resolve().parent / "pipelines" / f"train_{args.objective}.py"
    command = [sys.executable, str(pipeline), args.dataset, args.mode, args.device]
    print(f"Input: {data_file}\nOutputs: {run_dir}\nObjective: {args.objective.upper()}", flush=True)
    if args.dry_run:
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BIOSENSE_DATA_DIR=str(data_dir), PYTHONUTF8="1")
    return subprocess.run(command, cwd=run_dir, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
