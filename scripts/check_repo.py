"""Run static checks; opt into the full mocked test suite at integration boundaries."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="Also run the full unittest suite")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "scripts/export_openapi.py", "--check"],
    ]
    if args.tests:
        commands.append([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])
    for command in commands:
        result = subprocess.run(command, cwd=root, env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
