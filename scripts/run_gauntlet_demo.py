#!/usr/bin/env python3
"""Convenience entry point: run the self-contained gauntlet demo.

Equivalent to ``python examples/03_gauntlet.py`` but importable as a script from
the repo root. Requires only numpy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from importlib import import_module   # noqa: E402


def main():
    demo = import_module("03_gauntlet")
    demo.main()


if __name__ == "__main__":
    main()
