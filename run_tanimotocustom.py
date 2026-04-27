#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run REINVENT with tanimoto scoring.")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps.")
    parser.add_argument("--num-processes", type=int, default=0, help="Number of scoring processes (use 0 on Windows).")
    parser.add_argument("--model-path", default="random_forest_model_amp.pkl", help="Path to AMP RF model file.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python_exe = (script_dir / ".." / ".venv" / "Scripts" / "python.exe").resolve()
    main_py = (script_dir / "main.py").resolve()

    if not python_exe.exists():
        print(f"ERROR: Python not found at {python_exe}")
        print("Create the venv first: python -m venv ..\\.venv")
        return 1

    if not main_py.exists():
        print(f"ERROR: main.py not found at {main_py}")
        return 1

    cmd = [
        str(python_exe),
        str(main_py),
        "--scoring-function",
        "tanimoto",
        "--num-steps",
        str(args.num_steps),
        "--num-processes",
        str(args.num_processes),
        "--scoring-function-kwargs",
        "clf_path",
        args.model_path,
    ]

    print(f"Using Python: {python_exe}")
    print("Running tanimoto scoring...")

    completed = subprocess.run(cmd, cwd=str(script_dir))
    if completed.returncode != 0:
        print(f"Run failed with exit code {completed.returncode}")
        return completed.returncode

    print("Run completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
