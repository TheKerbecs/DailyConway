#!/usr/bin/env python3
"""List hit JSON files sorted by iteration count (highest first by default)."""

import argparse
import json
import os
import sys


def load_hits(hits_dir: str):
    hits = []
    for fname in os.listdir(hits_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(hits_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            hits.append({
                "file": fname,
                "path": path,
                "iterations": data.get("iterations", 0),
                "peak": data.get("peak", 0),
                "owner": data.get("owner", ""),
                "suite": data.get("suite", ""),
                "bin": data.get("bin", ""),
            })
        except (json.JSONDecodeError, OSError):
            print(f"Warning: could not read {fname}", file=sys.stderr)
    return hits


def main():
    parser = argparse.ArgumentParser(description="Sort hit files by iteration count.")
    parser.add_argument("--hits-dir", default="./hits", help="Directory containing hit JSON files.")
    parser.add_argument("--asc", action="store_true", help="Sort ascending (lowest first).")
    parser.add_argument("--top", type=int, default=None, help="Only show top N results.")
    args = parser.parse_args()

    hits = load_hits(args.hits_dir)
    hits.sort(key=lambda h: h["iterations"], reverse=not args.asc)

    if args.top is not None:
        hits = hits[:args.top]

    print(f"{'Rank':<6} {'Iterations':<12} {'Peak':<6} {'Suite':<6} {'File'}")
    print("-" * 80)
    for rank, h in enumerate(hits, 1):
        print(f"{rank:<6} {h['iterations']:<12} {h['peak']:<6} {h['suite']:<6} {h['file']} {h['bin']} ")


if __name__ == "__main__":
    main()
