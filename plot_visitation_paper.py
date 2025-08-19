#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_region_visits(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Handle either {"episodes": [...], "region_hits": [...]} or just a raw list
    if isinstance(data, dict):
        hits = data.get("region_hits", [])
        episodes = data.get("episodes", list(range(len(hits))))
    else:
        hits = list(data)
        episodes = list(range(len(hits)))
    return np.asarray(episodes), np.asarray(hits, dtype=float)

def rolling_mean(x, w):
    if w <= 1:
        return None
    if w > len(x):
        return None
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="experiments/safety_region_visits/region_visits.json",
                    help="Path to region_visits.json saved by RegionVisitObserver")
    ap.add_argument("--out", type=str, default=None,
                    help="Output figure path (default: same dir, 'region_visits_plot.pdf')")
    ap.add_argument("--rolling", type=int, default=0,
                    help="Rolling window size for smoothing (0 = off)")
    ap.add_argument("--show", action="store_true", help="Also display the plot")
    args = ap.parse_args()

    episodes, hits = load_region_visits(args.file)
    if args.out is None:
        out_dir = os.path.dirname(args.file) or "."
        args.out = os.path.join(out_dir, "region_visits_plot.pdf")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, hits, linewidth=1.5, label="Region visits per episode")

    if args.rolling and args.rolling > 1:
        rm = rolling_mean(hits, args.rolling)
        if rm is not None:
            ep_mid = episodes[args.rolling - 1:]  # align centers
            ax.plot(ep_mid, rm, linestyle="--", linewidth=2,
                    label=f"Rolling mean (w={args.rolling})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("# Visits in Region")
    ax.set_title("Region Visits per Episode")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
