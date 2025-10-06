#!/usr/bin/env python3
"""Code quality check entry points for uv run commands."""

import subprocess
import sys


def run_quality_checks() -> None:
    """Run all code quality checks: ruff linting, formatting, and pyright type checking.

    Usage:
        uv run quality-check
    """
    checks = [
        ("Ruff lint", ["ruff", "check", "src", "tests"]),
        ("Ruff format", ["ruff", "format", "--check", "src", "tests"]),
        ("Pyright", ["pyright"]),
    ]

    failed = []

    for name, cmd in checks:
        print(f"▶ {name}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            print()
            print(result.stdout.decode())
            print(result.stderr.decode())
            failed.append(name)
        else:
            # Print success message inline
            output = result.stdout.decode().strip()
            if output:
                print(output.split("\n")[0])
            else:
                print("All checks passed!")

    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ All checks passed")
        sys.exit(0)
