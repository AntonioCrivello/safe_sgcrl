#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import matplotlib as mpl  # add this import

mpl.rcParams.update({
    "figure.dpi": 200,      # canvas dpi while viewing
    "savefig.dpi": 600,     # export dpi for raster bits (PDF stays vector)
    "pdf.fonttype": 42,     # editable text in Illustrator/InkScape
    "ps.fonttype": 42,
    "font.family": "serif", # looks academic; switch to "DejaVu Serif"/"Times New Roman" if you prefer
    "font.size": 18,        # base font size (bump to 16–18 if you like)
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.2,  # thicker spines
    "lines.linewidth": 2.2, # thicker lines by default
    "grid.alpha": 0.25,
    # "text.usetex": True,  # uncomment if you want LaTeX rendering and have it installed
    "mathtext.fontset": "stix",
})


def find_block_files(root_dir):
    """
    Return a list of region_visits_XXXXXXXX-YYYYYY.json files in ascending
    order of start episode.
    """
    patt = os.path.join(root_dir, "region_visits_*.json")
    files = glob(patt)
    def start_ep(path):
        m = re.search(r"region_visits_(\d+)-(\d+)\.json$", os.path.basename(path))
        print(f"Found file: {path}, start episode: {m.group(1) if m else 'unknown'}")
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
        (indices outside the available range are ignored; if none remain and
         there is exactly one region, we fall back to that single region)
    """
    with open(path, "r") as f:
        data = json.load(f)

    episodes = np.asarray(data["episodes"], dtype=int)
    cpr = data.get("counts_per_region", {})

    # Convert counts_per_region dict -> per-region array
    n_regions = len(cpr)
    per_region = []
    for i in range(n_regions):
        key = f"region{i}"
        if key not in cpr:
            raise ValueError(f"Missing {key} in {path}")
        per_region.append(np.asarray(cpr[key], dtype=float))
    per_region = np.asarray(per_region)  # (R, T)

    # Pick regions robustly
    if regions is None:
        visits = per_region.sum(axis=0)
    else:
        # keep only valid indices
        valid = [int(r) for r in regions if 0 <= int(r) < n_regions]
        if len(valid) == 0:
            # if file has exactly one region, fall back to it
            if n_regions == 1:
                visits = per_region[0]
            else:
                raise IndexError(
                    f"Requested regions {regions} are out of range for file with {n_regions} region(s): {path}"
                )
        else:
            visits = per_region[np.asarray(valid, dtype=int)].sum(axis=0)

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

        # cum = np.cumsum(visits)
        # curves.append((eps, cum))
        rate = visits / 50.0          # per-episode visitation rate
        curves.append((eps, rate))

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
#"3330,3331,3332,3333,3334,3335,3336,3337"
# "4440,4441,4442,4443,4444,4445,4446,4447"
def main():
    ap = argparse.ArgumentParser(description="Plot average cumulative region visits over episodes across seeds.")
    ap.add_argument("--envs", type=str, default="point_FourRooms",
                    help="Comma-separated env names, e.g. 'Spiral11x11,Spiral15x15'")
    # ap.add_argument("--seeds", type=str, default="2220,2221,2222,2223,2224,2225,2226,2227",
    #                 help="(Ignored for the custom two-plot comparison) Comma-separated seeds.")
    ap.add_argument("--base_dir", type=str, default="experiments/safety_region_visits",
                    help="Base dir containing <env>_<seed>/ subfolders")
    ap.add_argument("--regions", type=str, default="",
                    help="(Ignored for the custom two-plot comparison) Comma-separated region indices.")
    ap.add_argument("--rolling", type=int, default=1000,
                    help="Rolling window on per-episode visits BEFORE accumulation (0 = off)")
    ap.add_argument("--std_band", action="store_true", default=True,
                    help="Show ±1 std shading across seeds")
    args = ap.parse_args()
    name = "safety"
    name = "red"

    envs = parse_comma_separated(args.envs)

    # --- Seed groups as requested ---
    ## for the safety experiment
    if name == "safety":
        seeds_safety_bottomleft = list(range(2220, 2228))   # "safety"
        # seeds_safety_topright   = list(range(3330, 3338))   # "safety"
        seeds_nosafety_common   = list(range(4440, 4448))   # "no safety"

        # --- Regions per plot (index-based) ---
        region = [1]  # "second region"
        labels = ["Safety", "No safety"]
        title="Bottom-left room visitation"
        clip = -1000
        dim = 5

    ## for the red hole experiment
    if name == "red":
        dim = 6
   
        seeds_safety_bottomleft = [s for s in range(5550, 5558) if s != 5555]

        seeds_nosafety_common   = list(range(4450, 4458))   # normal
        labels = ["Intervention", "Control"]
        region=[]
        title="Top-right room visitation"
        clip = -1


    for env in envs:
        # ===== Plot 1: bottom-left room =====
        #fig1, ax1 = plt.subplots(figsize=(8.5, 5))
        # replace: fig, ax = plt.subplots(figsize=(8.5, 5))

        fig1, ax1 = plt.subplots(figsize=(dim,dim))
        ax1.set_ylim(0, 0.37)
        if name == "red":
            ax1.set_ylim(-0.01, 0.4)
                
        # Safety group on region 1
        ep_s1, mean_s1, std_s1, n_s1 = collect_env_seed_curves(
            base_dir=args.base_dir, env=env, seeds=seeds_safety_bottomleft,
            regions=region, rolling=args.rolling, require_same_len=True
        )
        if ep_s1 is not None:
            # denom = (ep_s1 - ep_s1[0] + 1) * 50.0  # num_episodes * 50
            mean_s1 = mean_s1 
            std_s1  = std_s1  
            ax1.plot(ep_s1[:clip], mean_s1[:clip], linewidth=3, label=labels[0])
            if args.std_band and n_s1 > 1:
                ax1.fill_between(ep_s1[:clip], mean_s1[:clip] - std_s1[:clip], mean_s1[:clip] + std_s1[:clip], alpha=0.2)


        # No-safety group on region 1
        ep_ns1, mean_ns1, std_ns1, n_ns1 = collect_env_seed_curves(
            base_dir=args.base_dir, env=env, seeds=seeds_nosafety_common,
            regions=region, rolling=args.rolling, require_same_len=True
        )
        if ep_ns1 is not None:
            mean_ns1 = mean_ns1
            std_ns1  = std_ns1  
            ax1.plot(ep_ns1[:clip], mean_ns1[:clip], linewidth=3, label=labels[1])
            if args.std_band and n_ns1 > 1:
                ax1.fill_between(ep_ns1[:clip], mean_ns1[:clip] - std_ns1[:clip], mean_ns1[:clip] + std_ns1[:clip], alpha=0.2)

        ax1.tick_params(direction="out", length=6, width=1.2, top=False, right=False)
        ax1.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
        ax1.set_axisbelow(True)  # grid behind data
        

        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Visitation Rate")
        

        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        if name == "red":
            ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False
            )
            ax1.set_xlim(0, 20000)
        else:
            ax1.legend(loc="best", frameon=False, handlelength=2)
            ax1.set_xlim(0, 24000)


        plt.tight_layout()
        out1 = f"experiments/{env}_visiting_bottom_left_room_{name}.pdf"
        plt.savefig(out1, dpi=600, bbox_inches="tight", pad_inches=0.02)
        out1 = f"experiments/{env}_visiting_bottom_left_room_{name}.png"
        # plt.savefig(out1, dpi=600, bbox_inches="tight", pad_inches=0.02)
        # print(f"Saved plot to: {out1}")

    

        


if __name__ == "__main__":
    main()
