#!/usr/bin/env python3
"""Zero-dependency test runner.

The suite is written in plain pytest style, so ``pytest`` runs it directly.
This script exists so the tests can also be run on a machine that has nothing
installed but NumPy - which is the same promise the rest of the project makes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")


def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, ROOT)
    selected = sys.argv[1:]
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py"))
    if selected:
        files = [f for f in files if any(s in f for s in selected)]

    passed = failed = 0
    failures = []
    started = time.time()
    for filename in files:
        module = load(os.path.join(TESTS, filename))
        names = [n for n in dir(module) if n.startswith("test_")]
        print(f"\n{filename}")
        for name in sorted(names):
            fn = getattr(module, name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
            except Exception:
                failed += 1
                failures.append((filename, name, traceback.format_exc()))
                print(f"  FAIL  {name}  ({time.time() - t0:.2f}s)")
            else:
                passed += 1
                print(f"  ok    {name}  ({time.time() - t0:.2f}s)")

    print("\n" + "=" * 70)
    for filename, name, tb in failures:
        print(f"\n--- {filename}::{name} ---\n{tb}")
    print(f"{passed} passed, {failed} failed in {time.time() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
