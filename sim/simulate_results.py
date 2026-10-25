#!/usr/bin/env python3
"""Entry point named in the results specification:

    python sim/simulate_results.py --seed 2026

It runs ../simulate.py, which generates EXPECTED values for the Results section
of the article; it does not run retrieval or generation models. Effect sizes are
assumptions calibrated on the literature cited in the article.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate  # noqa: E402

if __name__ == "__main__":
    simulate.main()
