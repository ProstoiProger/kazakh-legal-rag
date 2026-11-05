#!/usr/bin/env python3
"""Entry point of the simulation, run from the repository root:

    python sim/simulate_results.py --seed 2026

It runs src/simulate.py. No retrieval or generation model is run: every reported
mean is an author-specified target reached by calibration, and the pattern of
significant and non-significant tests is pre-specified (DESIGNED in
src/simulate.py); see TARGETS_SOURCES.md. Outputs are written to outputs/
(data/, figures/, tables/) unless --out is given.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import simulate  # noqa: E402

if __name__ == "__main__":
    simulate.main()
