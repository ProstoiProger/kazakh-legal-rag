#!/usr/bin/env python3
"""Check that the outputs in this repository (or a fresh run) match the published
checksums in outputs/SHA256SUMS.

Usage (from the repository root):
    python src/verify_outputs.py            # check the files as they are now
    python src/verify_outputs.py --write    # (maintainers) rewrite outputs/SHA256SUMS

runtime.txt and versions.txt are not checked: they describe the machine and the
wall-clock time of a run and are expected to differ between computers.
"""
import argparse
import hashlib
import sys
from pathlib import Path

SUMS = Path("outputs/SHA256SUMS")
PATTERNS = ["outputs/data/*.csv", "outputs/data/results_summary.json", "outputs/data/sim_config.json",
            "outputs/tables/*.tex", "figures/*.pdf"]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        files = sorted({f for pat in PATTERNS for f in Path(".").glob(pat)})
        SUMS.write_text("".join(f"{sha256(f)}  {f.as_posix()}\n" for f in files), encoding="utf-8")
        print(f"wrote {SUMS} ({len(files)} files)")
        return 0
    bad = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        p = Path(name)
        status = "MISSING" if not p.exists() else ("OK" if sha256(p) == digest else "DIFFERENT")
        bad += status != "OK"
        print(f"{status:9s} {name}")
    print("all files identical" if bad == 0 else f"{bad} file(s) differ or are missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
