"""Per-layer width allocation: depth-tapered budgets at equal total bytes.

Uniform truncation is not optimal. Measured on both a dense 64-layer model and a
48-layer MoE (three-direction controlled comparison at exactly equal bytes):
sensitivity to width cuts INCREASES monotonically with depth — late-layer error
has no remaining layers to damp it, while early-layer error is partially
absorbed downstream. Allocating more width to deep layers (a gentle linear
taper, ratio ~0.85 first/last) dominated the uniform allocation on the
generative signals measured against it (GSM8K +10, both perplexities; HumanEval
86%, matching the fine flagship's level), at
a small cost in broad-knowledge ranking (MMLU −5). The aggressive ratio 0.714
is strictly worse than 0.85; protecting shallow layers instead is the worst
possible allocation.

Output: k_alloc_*.json with {"k_per_layer": [n_layers ints]} consumed by
convert_fine.py / convert_block.py (--k-json) and per_layer.py at load time.

Usage:
    python src/allocate.py --n-layers 48 --width 768 --k-mean 640 --ratio 0.85
"""

from __future__ import annotations

import argparse
import json

import numpy as np

GROUP = 64


def alloc_taper(n_layers: int, n_blocks: int, budget_blocks: int,
                ratio: float, lo_frac: float, hi_frac: float) -> np.ndarray:
    """Linear taper: ratio = k_first/k_last (<1 gives deep layers more width)."""
    lo_total, hi_total = n_layers * int(lo_frac * n_blocks), n_layers * int(hi_frac * n_blocks)
    assert lo_total <= budget_blocks <= hi_total, (
        f"presupuesto inviable con los clamps: {budget_blocks} bloques no está en "
        f"[{lo_total}, {hi_total}] (suelo/techo x {n_layers} capas)")
    mean = budget_blocks / n_layers
    s = 2 * mean * (ratio - 1) / ((ratio + 1) * (n_layers - 1))
    a = mean + s * (n_layers - 1) / 2
    n = np.clip(np.round(a - s * np.arange(n_layers)).astype(int),
                int(lo_frac * n_blocks), int(hi_frac * n_blocks))
    i = 0
    while n.sum() > budget_blocks:
        li = n_layers - 1 - (i % n_layers)
        if n[li] > int(lo_frac * n_blocks):
            n[li] -= 1
        i += 1
    while n.sum() < budget_blocks:
        li = i % n_layers
        if n[li] < int(hi_frac * n_blocks):
            n[li] += 1
        i += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary per-layer width allocation")
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--width", type=int, required=True, help="original intermediate width")
    parser.add_argument("--k-mean", type=int, required=True,
                        help="average k (total budget = n_layers * k_mean)")
    parser.add_argument("--ratio", type=float, default=0.85,
                        help="k_first/k_last; <1 protects deep layers (recommended)")
    parser.add_argument("--floor-frac", type=float, default=0.75,
                        help="per-layer floor as a fraction of the original width")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    assert args.k_mean % GROUP == 0 and args.width % GROUP == 0
    n_blocks = args.width // GROUP
    budget = args.n_layers * (args.k_mean // GROUP)
    n = alloc_taper(args.n_layers, n_blocks, budget, args.ratio, args.floor_frac, 1.0)
    ks = (n * GROUP).tolist()
    assert sum(ks) == budget * GROUP

    out = args.out or f"runs/k_alloc_taper{args.ratio}.json"
    json.dump({"k_per_layer": ks, "budget_total": sum(ks)}, open(out, "w"))
    arr = np.array(ks)
    print(f"[alloc] k per layer: {arr.min()}–{arr.max()} (mean {arr.mean():.0f}), "
          f"{int((arr == args.width).sum())} layers uncut")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
