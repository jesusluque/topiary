"""Fine build (preferred): permute the ORIGINAL bf16 weights at neuron
granularity, truncate to the prefix, then quantize.

Quantization groups form *after* the reorder, so the ordering is exact and no
requantization noise is introduced. This is the variant that produced the best
checkpoints in our evaluation.

Streams shard by shard — the full model is never resident.

Inputs:
    --src     original (bf16) checkpoint, local dir or HF repo (cached)
    --ref     a quantized reference checkpoint: its config supplies group_size,
              bits and per-module quantization overrides, which are respected
              both when packing and in the output config
    --orders  .npz with per-layer salience (from runtime.py or profile_stream.py)
    --k       target intermediate width (multiple of group_size)

Usage:
    python src/convert_fine.py --src Qwen/Qwen3-30B-A3B --ref mlx-community/Qwen3-30B-A3B-4bit \
        --orders runs/orders.npz --k 640 --out models/qwen3-30b-4bit-w640
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERT_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def resolve(path_or_repo: str) -> Path:
    if Path(path_or_repo).exists():
        return Path(path_or_repo)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, local_files_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary fine build: permute -> truncate -> quantize")
    parser.add_argument("--src", required=True, help="original bf16 checkpoint")
    parser.add_argument("--ref", required=True, help="quantized reference (supplies quantization config)")
    parser.add_argument("--orders", required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--k-json", help="per-layer k allocation (see allocate.py)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src, ref = resolve(args.src), resolve(args.ref)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref_cfg = json.load(open(ref / "config.json"))
    qmap = ref_cfg["quantization"]
    group_size, bits = int(qmap["group_size"]), int(qmap["bits"])
    if qmap.get("mode", "affine") != "affine":
        raise SystemExit(f"[abort] non-affine reference quantization ({qmap['mode']!r}): "
                         "this tool packs affine only and the copied config would lie")
    orders_data = np.load(args.orders)
    src_cfg = json.load(open(src / "config.json"))
    n_layers_ref = int(src_cfg["num_hidden_layers"])
    moe_layers = sorted(int(kk.split("_")[1]) for kk in orders_data.files
                        if kk.startswith("salience_"))
    assert moe_layers == list(range(n_layers_ref)), (
        "orders npz does not cover layers 0..N-1: per-layer k would misalign "
        f"(missing: {sorted(set(range(n_layers_ref)) - set(moe_layers))[:5]}...)")
    assert (args.k is None) != (args.k_json is None), "pass exactly one of --k / --k-json"
    if args.k_json:
        ks = json.load(open(args.k_json))["k_per_layer"]
        assert len(ks) == n_layers_ref and all(kk % group_size == 0 for kk in ks)
        print(f"[fine] neuron-level permutation + {bits}-bit g{group_size} quantization, "
              f"per-layer k {min(ks)}-{max(ks)} (mean {sum(ks) / len(ks):.0f})")
    else:
        assert args.k % group_size == 0, f"k={args.k} must be a multiple of group_size={group_size}"
        ks = [args.k] * n_layers_ref
        print(f"[fine] neuron-level permutation + {bits}-bit g{group_size} quantization, k={args.k}")

    def neuron_order(layer: int) -> np.ndarray:
        # Fine per-neuron order by routed salience: [E, inter], no block grouping.
        return np.argsort(-orders_data[f"salience_{layer}"].astype(np.float64), axis=1)

    def is_quantized_module(key: str) -> bool:
        # The reference map lists modules as bool or per-module dict entries.
        mod = key.rsplit(".", 1)[0]
        val = qmap.get(mod)
        if isinstance(val, bool):
            return val
        if isinstance(val, dict):
            return True
        # default: quantize what the reference quantizes globally (2-D weights)
        return True

    weight_map: dict[str, str] = {}
    total = 0
    shard_idx, acc, acc_bytes = 1, {}, 0

    def flush():
        nonlocal shard_idx, acc, acc_bytes
        if not acc:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        mx.save_safetensors(str(out / name), acc)
        for kk in acc:
            weight_map[kk] = name
        print(f"  {name}: {len(acc)} tensors, {acc_bytes / 1e9:.2f} GB "
              f"(MLX active {mx.get_active_memory() / 1e9:.1f} GB)")
        shard_idx += 1
        acc, acc_bytes = {}, 0
        mx.clear_cache()

    def emit(key: str, arrs: dict[str, mx.array]):
        nonlocal acc_bytes, total
        for suffix, a in arrs.items():
            mx.eval(a)
            acc[key + suffix] = a
            acc_bytes += a.nbytes
            total += a.nbytes

    def qparams_of(key_base: str) -> tuple[int, int]:
        # The output config copies the reference qmap verbatim, so packing MUST
        # honor its per-module overrides (e.g. routers kept at 8-bit) — packing
        # everything at global bits would make the metadata lie.
        val = qmap.get(key_base)
        if isinstance(val, dict):
            return int(val.get("group_size", group_size)), int(val.get("bits", bits))
        return group_size, bits

    def quantize_full(key_base: str, w: mx.array):
        gs, b = qparams_of(key_base)
        q_w, q_s, *q_b = mx.quantize(w, group_size=gs, bits=b)
        emit(key_base, {".weight": q_w, ".scales": q_s,
                        **({".biases": q_b[0]} if q_b else {})})

    for shard_path in sorted(glob.glob(str(src / "model-*.safetensors"))):
        tensors = dict(mx.load(shard_path))
        for key in list(tensors.keys()):
            arr = tensors.pop(key)
            base = key[:-7] if key.endswith(".weight") else key
            proj = base.split(".")[-1]

            if ".switch_mlp." in key and proj in EXPERT_SUFFIXES and key.endswith(".weight"):
                layer = int(key.split(".layers.")[1].split(".")[0])
                k_l = ks[layer]
                order = neuron_order(layer)[:, :k_l]
                sel = mx.array(np.ascontiguousarray(order))
                if proj == "down_proj":  # [E, out, inter] -> permute columns
                    idx = mx.broadcast_to(sel[:, None, :], (arr.shape[0], arr.shape[1], k_l))
                    arr = mx.take_along_axis(arr, idx, axis=2)
                else:  # [E, inter, in] -> permute rows
                    idx = mx.broadcast_to(sel[..., None], (arr.shape[0], k_l, arr.shape[2]))
                    arr = mx.take_along_axis(arr, idx, axis=1)
                quantize_full(base, arr)
            elif key.endswith(".weight") and arr.ndim >= 2 and is_quantized_module(key):
                quantize_full(base, arr)
            else:
                emit(key, {"": arr})

            if acc_bytes >= 4_500_000_000:
                flush()
        del tensors
    flush()

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    json.dump(index, open(out / "model.safetensors.index.json", "w"))

    cfg = json.load(open(src / "config.json"))
    inter_key = "moe_intermediate_size" if "moe_intermediate_size" in cfg else "intermediate_size"
    if args.k_json:
        cfg[inter_key] = max(ks)
        cfg[inter_key + "_per_layer"] = ks  # consumed by per_layer.maybe_patch
    else:
        cfg[inter_key] = args.k
    cfg["quantization"] = qmap
    cfg["quantization_config"] = qmap
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for aux in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                "chat_template.jinja", "special_tokens_map.json", "vocab.json",
                "merges.txt", "added_tokens.json"):
        for origin in (src, ref):
            if (origin / aux).exists():
                shutil.copy(origin / aux, out / aux)
                break

    print(f"[out] {out}  ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
