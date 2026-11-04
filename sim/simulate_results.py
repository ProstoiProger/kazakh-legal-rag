#!/usr/bin/env python3
"""Entry point of the simulation, run from the repository root:

    python sim/simulate_results.py --seed 2026

It runs src/simulate.py, which generates EXPECTED values for the Results section
of the article; it does not run retrieval or generation models. Effect sizes are
assumptions calibrated on the literature cited in the article. Outputs are
written to outputs/ (data/, figures/, tables/) unless --out is given.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import simulate  # noqa: E402

if __name__ == "__main__":
    simulate.main()
