#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def find_block_files(root_dir):
    """
    Return a list of region_visits_XXXXXXXX-YYYYYY.json files in ascending
    order of start episode.
    """
    patt = os.path.join(root_dir, "region_visits_*.json")
    files = glob(patt)
    def start_ep(path):
        m = re.search(r"region_visits_(\d+)-(\d+)\.json$", os.path.basename(path))
        return int(m.group(1)) if m else 10**12
    return sorted(files, key=start_ep)

def load_block(path, regions=None):
    """
    Load one block file. Returns:
      episodes: np.array shape (T,)
      visits_per_ep: np.array shape (T,) after summing (or selecting) regions
    'regions' can be:
      - None: sum across all regions
      - list of ints: select these region indices and sum across them
    """
    with open(path, "r") as f:
        data = json.load(f)

    episodes = np.asarray(data["episodes"], dtype=int)
    cpr = data.get("counts_per_region", {})

    # Convert counts_per_region dict -> list per region index
    # keys look like 'region0', 'region1', ...
    n_regions = len(cpr)
    per_region = []
    for i in range(n_regions):
        key = f"region{i}"
        if key not in cpr:
            raise ValueError(f"Missing {key} in {path}")
        per_region.append(np.asarray(cpr[key], dtype=float))
    per_region = np.asarray(per_region)  # (R, T)

    if regions is None:
        visits = per_region.sum(axis=0)
    else:
        visits = per_region[np.asarray(regions, dtype=int)].sum(axis=0)

    if visits.shape[0] != episodes.shape[0]:
        raise ValueError(f"Length mismatch in {path}: episodes {episodes.shape[0]} vs visits {visits.shape[0]}")

    return episodes, visits

def load_blocks_concat(paths, regions=None, rolling=0):
    """
    Load a list of block files, concatenate episodes and visits in order.
    Returns:
      episodes_all: (T,) int
      visits_all:   (T,) float  (sum across chosen regions)
    """
    eps_all = []
    vis_all = []
    last_end = -1

    for p in paths:
        eps, visits = load_block(p, regions=regions)
        if len(eps_all) > 0 and (eps[0] <= last_end):
            print(f"[warn] Overlap or out-of-order episodes in {p}: "
                  f"prev_end={last_end}, this_start={eps[0]}")
        last_end = eps[-1]
        eps_all.append(eps)
        vis_all.append(visits)

    if not eps_all:
        return None, None

    episodes_all = np.concatenate(eps_all, axis=0)
    visits_all = np.concatenate(vis_all, axis=0)

    if rolling and rolling > 1:
        visits_all = rolling_mean(visits_all, rolling)

    return episodes_all, visits_all

def rolling_mean(x, w):
    if w <= 1:
        return x
    if w > len(x):
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    rm = (c[w:] - c[:-w]) / w
    # pad to original length by repeating the first valid value for the first (w-1) positions
    pad = np.full(w-1, rm[0], dtype=float)
    return np.concatenate([pad, rm])

def collect_env_seed_curves(base_dir, env, seeds, regions=None, rolling=0, require_same_len=True):
    """
    For an env and a list of seeds, return:
      episodes_common, cum_mean, cum_std, num_used
    Only uses the FIRST block file per seed.
    If require_same_len=True, trims all curves to the minimum length across seeds.
    """
    curves = []
    lengths = []
    for seed in seeds:
        root = os.path.join(base_dir, f"{env}_{seed}")
        files = find_block_files(root)
        if not files:
            print(f"[warn] No block files for {env=} {seed=} under {root}")
            continue

        eps, visits = load_blocks_concat(files, regions=regions, rolling=rolling)
        if eps is None:
            print(f"[warn] Failed to load {env=} {seed=}")
            continue

        cum = np.cumsum(visits)
        curves.append((eps, cum))
        lengths.append(len(eps))

    if not curves:
        return None, None, None, 0

    if require_same_len:
        L = min(lengths)
        trimmed = [(e[:L], c[:L]) for (e, c) in curves]
        episodes_common = trimmed[0][0]
        stack = np.stack([c for (_, c) in trimmed], axis=0)
    else:
        # Align by episode index (assumes all start at 0 and are contiguous)
        # Find max last common episode
        L = min([len(e) for (e, _) in curves])
        episodes_common = curves[0][0][:L]
        stack = np.stack([c[:L] for (_, c) in curves], axis=0)

    cum_mean = stack.mean(axis=0)
    cum_std = stack.std(axis=0, ddof=0)
    return episodes_common, cum_mean, cum_std, stack.shape[0]

def parse_comma_separated(arg):
    if arg is None or arg == "":
        return []
    return [a.strip() for a in arg.split(",") if a.strip()]
#"2220,2221,2222,2223,2224,2225,2226,2227"
#"5550,5551,5552,5553,5554,5555,5556,5557"
#="5550,5551,5552,5553,5554,5556,5557",
def main():
    ap = argparse.ArgumentParser(description="Plot average cumulative region visits over episodes across seeds (first block only).")
    ap.add_argument("--envs", type=str, default="point_FourRooms",
                    help="Comma-separated env names, e.g. 'Spiral11x11,Spiral15x15'")
    ap.add_argument("--seeds", type=str, default="4450,4451,4452,4453,4454,4455,4456,4457",
                    help="Comma-separated seeds, e.g. '0,1,2,3,4'")
    ap.add_argument("--base_dir", type=str, default="experiments/safety_region_visits",
                    help="Base dir containing <env>_<seed>/ subfolders")
    ap.add_argument("--regions", type=str, default="",
                    help="Comma-separated region indices to include (default: sum all regions). Example: '0' or '0,1'")
    ap.add_argument("--rolling", type=int, default=0,
                    help="Rolling window on per-episode visits BEFORE accumulation (0 = off)")
    ap.add_argument("--std_band", action="store_true",
                    help="Show ±1 std shading across seeds")
    ap.add_argument("--title", type=str, default="Average Cumulative Region Visits (First Block Only)")
    args = ap.parse_args()

    envs = parse_comma_separated(args.envs)
    seeds = [int(s) for s in parse_comma_separated(args.seeds)]
    regions = [int(r) for r in parse_comma_separated(args.regions)] if args.regions else None

    
    seed_str = "_".join(str(s) for s in seeds)
    args.out = f"avg_cum_region_visits_seeds{seed_str}_{regions}.pdf"

    fig, ax = plt.subplots(figsize=(8.5, 5))

    any_plotted = False
    for env in envs:
        episodes, cum_mean, cum_std, used = collect_env_seed_curves(
            base_dir=args.base_dir,
            env=env,
            seeds=seeds,
            regions=regions,
            rolling=args.rolling,
            require_same_len=True
        )
        if episodes is None:
            print(f"[warn] Skipping {env}: no data found.")
            continue

        label = f"{env} (n={used})"
        ax.plot(episodes, cum_mean, linewidth=2, label=label)
        if args.std_band and used > 1:
            ax.fill_between(episodes, cum_mean - cum_std, cum_mean + cum_std, alpha=0.2)

        any_plotted = True

    if not any_plotted:
        print("[error] No data to plot.")
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative # Region Visits")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")

if __name__ == "__main__":
    main()
