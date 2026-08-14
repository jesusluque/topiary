"""Block build (post-hoc): sculpt an ALREADY-QUANTIZED community checkpoint.

Permutes whole quantization groups ("blocks" of `group_size` neurons) on the
packed tensors — entire packed words move together with their scales/biases, so
not a single weight bit is touched and there is zero requantization error. The
ordering is coarser than the fine build, but no bf16 original is needed.

The output is a standard, self-contained mlx-lm checkpoint with
`moe_intermediate_size = k`: it loads and serves with stock mlx-lm, no custom
code. Streams shard by shard.

Usage:
    python src/convert_block.py --src mlx-community/Qwen3-30B-A3B-4bit \
        --orders runs/orders.npz --k 576 --out models/qwen3-30b-4bit-w576
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import block_orders

EXPERT_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def permute_and_truncate(
    key: str, arr: mx.array, border: np.ndarray, k: int, group_size: int, bits: int
) -> mx.array:
    """Apply the block permutation and prefix truncation to one expert tensor."""
    proj = key.split(".")[-2]
    n_experts = arr.shape[0]
    n_keep = k // group_size

    if proj in ("gate_proj", "up_proj"):
        # Blocks are output rows: gather the rows of the n_keep best blocks.
        row_idx = np.stack(
            [np.concatenate([np.arange(group_size) + group_size * b for b in border[e][:n_keep]])
             for e in range(n_experts)]
        )
        sel = mx.array(row_idx)[..., None]
        out = mx.take_along_axis(arr, mx.broadcast_to(sel, (n_experts, k, arr.shape[2])), axis=1)
    else:  # down_proj: blocks are segments of the (packed) input axis
        if key.endswith(".weight"):
            wpb = group_size * bits // 32  # packed words per block
            idx = np.stack(
                [np.concatenate([np.arange(wpb) + wpb * b for b in border[e][:n_keep]])
                 for e in range(n_experts)]
            )
        else:  # scales / biases: one column per block
            idx = np.stack([border[e][:n_keep] for e in range(n_experts)])
        sel = mx.array(idx)[:, None, :]
        sel = mx.broadcast_to(sel, (n_experts, arr.shape[1], idx.shape[1]))
        out = mx.take_along_axis(arr, sel, axis=2)

    mx.eval(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary block build on a quantized checkpoint")
    parser.add_argument("--src", required=True)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--k-json", help="per-layer k allocation (see allocate.py)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if Path(args.src).exists():
        src = Path(args.src)
    else:
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.src, local_files_only=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(src / "config.json"))
    qmap = cfg["quantization"]
    group_size = int(qmap["group_size"])
    bits = int(qmap["bits"])

    def bits_of(key: str) -> int:
        # Per-MODULE bits, resolved the way the loader does: config overrides win
        # (e.g. routers kept at 8-bit). Using global bits would slice the packed
        # words of an overridden down_proj at the wrong stride.
        val = qmap.get(key.rsplit(".", 1)[0])
        return int(val["bits"]) if isinstance(val, dict) else bits

    inter_key = "moe_intermediate_size" if "moe_intermediate_size" in cfg else "intermediate_size"
    inter = int(cfg[inter_key])
    n_layers = int(cfg["num_hidden_layers"])
    if args.k_json:
        ks = json.load(open(args.k_json))["k_per_layer"]
        assert len(ks) == n_layers and all(kk % group_size == 0 and kk <= inter for kk in ks)
        print(f"[convert] {args.src}: {bits}-bit, per-layer k {min(ks)}-{max(ks)} "
              f"(mean {sum(ks) / len(ks):.0f}), blocks of {group_size}")
    else:
        assert args.k % group_size == 0 and args.k < inter
        ks = [args.k] * n_layers
        print(f"[convert] {args.src}: {bits}-bit, inter {inter} -> {args.k} "
              f"({args.k / inter:.0%}), blocks of {group_size}")

    orders_cache: dict[int, np.ndarray] = {}

    def orders_for(layer: int) -> np.ndarray:
        if layer not in orders_cache:
            orders_cache[layer] = block_orders(args.orders, layer, group_size)
        return orders_cache[layer]

    index = json.load(open(src / "model.safetensors.index.json"))
    new_map: dict[str, str] = {}
    total = 0
    for shard_path in sorted(glob.glob(str(src / "model-*.safetensors"))):
        shard = Path(shard_path).name
        tensors = mx.load(shard_path)
        out_tensors = {}
        n_conv = 0
        for key, arr in tensors.items():
            if ".switch_mlp." in key and key.split(".")[-2] in EXPERT_SUFFIXES:
                layer = int(key.split(".layers.")[1].split(".")[0])
                arr = permute_and_truncate(key, arr, orders_for(layer), ks[layer],
                                           group_size, bits_of(key))
                n_conv += 1
            out_tensors[key] = arr
            new_map[key] = shard
            total += arr.nbytes
        mx.save_safetensors(str(out / shard), out_tensors)
        del tensors, out_tensors
        mx.clear_cache()
        print(f"  {shard}: {n_conv} expert tensors converted")

    index["weight_map"] = new_map
    index["metadata"]["total_size"] = total
    json.dump(index, open(out / "model.safetensors.index.json", "w"))

    if args.k_json:
        cfg[inter_key] = max(ks)
        cfg[inter_key + "_per_layer"] = ks  # consumed by per_layer.maybe_patch
    else:
        cfg[inter_key] = args.k
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for aux in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                "chat_template.jinja", "special_tokens_map.json", "vocab.json",
                "merges.txt", "added_tokens.json"):
        if (src / aux).exists():
            shutil.copy(src / aux, out / aux)

    print(f"[out] {out}  ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
