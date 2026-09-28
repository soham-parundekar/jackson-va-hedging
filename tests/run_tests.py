"""Run the test suite without any test framework installed.

Discovers every ``test_*`` function in every ``tests/test_*.py`` module, runs them, and
reports failures with a traceback. Exits non-zero if anything failed, so ``make test``
behaves the way a build step should.

Usage:  python -m tests.run_tests [name fragment ...]
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def discover(filters: list[str]) -> list[tuple[str, str, object]]:
    found = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            label = f"{path.stem}::{name}"
            if filters and not any(f in label for f in filters):
                continue
            found.append((path.stem, name, function))
    return found


def main(argv: list[str]) -> int:
    cases = discover(argv)
    if not cases:
        print("no tests matched")
        return 1

    failures = []
    current_module = None
    started = time.time()
    for module_name, name, function in cases:
        if module_name != current_module:
            print(f"\n{module_name}")
            current_module = module_name
        label = f"  {name}"
        try:
            elapsed = time.time()
            function()
            print(f"{label:<62} ok   {time.time() - elapsed:6.2f}s")
        except Exception:
            print(f"{label:<62} FAIL")
            failures.append((module_name, name, traceback.format_exc()))

    print(f"\n{len(cases) - len(failures)} passed, {len(failures)} failed "
          f"in {time.time() - started:.1f}s")
    for module_name, name, trace in failures:
        print(f"\n{'=' * 78}\n{module_name}::{name}\n{'=' * 78}\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
