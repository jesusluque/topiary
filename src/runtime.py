"""Online profiling and in-place self-compression: the model profiles itself
while serving normal traffic, then permutes and truncates itself IN RAM.

The routed-salience statistic lives inside the forward pass: for the experts a
token is routed to, `h = act(gate(x)) * up(x)` is already computed — accumulating
`sum(h^2)` per neuron is one extra reduction. After enough traffic the model can
bake the salience permutation into its own quantized tensors and drop every
suffix, freeing the memory immediately (2.9 GB freed in ~1 s on a 30B model in
our runs) with no service interruption and no offline calibration phase.

Statistically this differs from classic calibration: offline, every expert sees
every token; online, each expert sees only the tokens the router actually sends
it. Routed-only statistics produced strictly better orderings in our evaluation.

Modes:
    --profile-only   serve N tokens, save the salience orders (.npz) and exit —
                     use this to feed convert_fine.py / convert_block.py
    (default)        profile, then bake + truncate the LIVE model to k and
                     report freed RAM and held-out perplexity

Usage:
    python src/runtime.py --model mlx-community/Qwen3-30B-A3B-4bit \
        --data data/calib.jsonl --profile-only --orders-out runs/orders.npz
    python src/runtime.py --model mlx-community/Qwen3-30B-A3B-4bit \
        --data data/calib.jsonl --k 576 --held-out data/held_out.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import mlx.core as mx
import numpy as np

from common import (
    REPO_ROOT,
    block_orders,
    dequant_stack,
    experts_of,
    find_moe_blocks,
    load_corpus,
    set_seeds,
    token_nll,
)


# ---------------------------------------------------------- online salience


def install_online_salience(model) -> tuple[Any, dict[int, dict[str, np.ndarray]]]:
    """Patch SwitchGLU to accumulate sum(h^2) and counts per (layer, expert, neuron).

    Replicates the original forward (including its sorted fast path) plus one
    host-side accumulation per call. A production deployment would keep the
    accumulator on GPU with ~zero overhead; clarity wins here.
    """
    from mlx_lm.models import switch_layers as sl

    blocks = find_moe_blocks(model)
    acc: dict[int, dict[str, np.ndarray]] = {}
    id_to_layer: dict[int, int] = {}
    for layer_idx, block in blocks:
        experts = experts_of(block)
        n_e = experts["gate_proj"]["weight"].shape[0]
        inter = experts["gate_proj"]["weight"].shape[1]
        acc[layer_idx] = {
            "h2": np.zeros((n_e, inter), dtype=np.float64),
            "count": np.zeros(n_e, dtype=np.int64),
        }
        id_to_layer[id(experts)] = layer_idx

    original = sl.SwitchGLU.__call__

    def patched(self, x, indices):
        layer = id_to_layer.get(id(self))
        if layer is None:
            return original(self, x, indices)

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        h = self.activation(x_up, x_gate)

        # h is already computed by the normal forward; this only sums it.
        h2 = (h.astype(mx.float32) ** 2).reshape(-1, h.shape[-1])
        flat_idx = idx.reshape(-1)
        mx.eval(h2)
        np.add.at(acc[layer]["h2"], np.array(flat_idx), np.array(h2))
        np.add.at(acc[layer]["count"], np.array(flat_idx), 1)

        out = self.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            out = sl._scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    sl.SwitchGLU.__call__ = patched
    return original, acc


def salience_from_online(model, acc: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """salience^2 = E[h^2] * ||down column||^2, expectation over routed tokens only."""
    out: dict[str, np.ndarray] = {}
    undersampled = 0
    for layer_idx, block in find_moe_blocks(model):
        experts = experts_of(block)
        wd = dequant_stack(experts["down_proj"])  # [E, out, inter]
        col2 = np.array((wd.astype(mx.float32) ** 2).sum(axis=1))  # [E, inter]
        a = acc[layer_idx]
        count = np.maximum(a["count"], 1)[:, None]
        mean_h2 = a["h2"] / count
        out[f"salience_{layer_idx}"] = (mean_h2 * col2).astype(np.float32)
        out[f"layer_{layer_idx}"] = np.argsort(
            -out[f"salience_{layer_idx}"], axis=1
        ).astype(np.int32)
        undersampled += int((a["count"] < 100).sum())
        del wd
    print(f"[online] experts with <100 routed tokens: {undersampled}")
    return out


# --------------------------------------------------- bake + truncate in place


def bake_permutation(model, orders_path, group_size: int) -> None:
    """Permute the quantized expert tensors IN PLACE by block salience.

    gate/up: blocks are output rows -> permute weight/scales/biases on axis 1.
    down: blocks are segments of the packed input axis -> permute the packed
    words (group_size*bits/32 per block) and the scales/biases columns.
    """
    for layer_idx, block in find_moe_blocks(model):
        border = block_orders(orders_path, layer_idx, group_size)
        experts = experts_of(block)
        n_experts, n_blocks = border.shape

        row_idx = np.stack(
            [np.concatenate([np.arange(group_size) + group_size * b for b in border[e]])
             for e in range(n_experts)]
        )  # [E, inter]

        for name in ("gate_proj", "up_proj"):
            proj = experts[name]
            sel = mx.array(row_idx)
            for key in ("weight", "scales", "biases"):
                arr = proj[key]
                proj[key] = mx.take_along_axis(arr, sel[..., None], axis=1)
                mx.eval(proj[key])

        proj = experts["down_proj"]
        assert proj.group_size == group_size, (
            f"module group_size ({proj.group_size}) != requested ({group_size})"
        )
        wpb = group_size * proj.bits // 32  # packed words per block
        word_idx = np.stack(
            [np.concatenate([np.arange(wpb) + wpb * b for b in border[e]])
             for e in range(n_experts)]
        )
        for key, idx in (("weight", word_idx), ("scales", border), ("biases", border)):
            arr = proj[key]
            sel = mx.array(idx)[:, None, :]  # [E, 1, n] -> broadcast over out
            sel = mx.broadcast_to(sel, (arr.shape[0], arr.shape[1], idx.shape[1]))
            proj[key] = mx.take_along_axis(arr, sel, axis=2)
            mx.eval(proj[key])


def truncate_in_place(model, k: int, group_size: int) -> float:
    """After baking the permutation, cut the LIVE model to prefix k and free the
    suffixes. Returns GB freed as seen by MLX."""
    before = mx.get_active_memory()
    for _, block in find_moe_blocks(model):
        experts = experts_of(block)
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = experts[name]
            if name == "down_proj":
                words = k * proj.bits // 32
                new = {
                    "weight": proj["weight"][:, :, :words],
                    "scales": proj["scales"][:, :, : k // group_size],
                    "biases": proj["biases"][:, :, : k // group_size],
                }
            else:
                new = {key: proj[key][:, :k] for key in ("weight", "scales", "biases")}
            for key, arr in new.items():
                proj[key] = mx.contiguous(arr)
            mx.eval(proj["weight"], proj["scales"], proj["biases"])
    # The original tensors are unreferenced now: MLX can release them.
    return (before - mx.get_active_memory()) / 1024**3


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary runtime: self-profile + self-compress")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="calibration jsonl (see build_corpus.py)")
    parser.add_argument("--k", type=int, default=576)
    parser.add_argument("--profile-tokens", type=int, default=40000)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--profile-only", action="store_true",
                        help="capture routed salience, save orders, exit")
    parser.add_argument("--orders-out", default=str(REPO_ROOT / "runs" / "orders_online.npz"))
    parser.add_argument("--held-out", help="held-out jsonl for the post-truncation PPL check")
    args = parser.parse_args()

    set_seeds(1234)
    from mlx_lm import load

    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    blocks = find_moe_blocks(model)
    if not blocks:
        raise SystemExit("[abort] no MoE blocks found — this tool targets MoE checkpoints")
    if experts_of(blocks[0][1])["down_proj"].get("biases") is None:
        raise SystemExit("[abort] biasless quantization mode: bake/truncate assume "
                         "affine weight/scales/biases triples")
    group_size = int(experts_of(blocks[0][1])["down_proj"].group_size)

    print(f"[profile] serving ~{args.profile_tokens} tokens with online salience...")
    original, acc = install_online_salience(model)
    rows = load_corpus(args.data, args.profile_tokens)
    t0 = time.perf_counter()
    served = 0
    for row in rows:
        ids = mx.array(tokenizer.encode(row["text"])[:1024])[None]
        mx.eval(model(ids))
        served += ids.shape[1]
    from mlx_lm.models import switch_layers as sl

    sl.SwitchGLU.__call__ = original
    print(f"[profile] {served} tokens in {time.perf_counter() - t0:.0f}s")

    orders = salience_from_online(model, acc)
    np.savez_compressed(args.orders_out, **orders)
    print(f"[out] {args.orders_out}")
    if args.profile_only:
        return

    print(f"[bake] permuting and truncating to k={args.k} IN RAM...")
    t0 = time.perf_counter()
    bake_permutation(model, args.orders_out, group_size)
    freed = truncate_in_place(model, args.k, group_size)
    print(f"[bake] done in {time.perf_counter() - t0:.0f}s — RAM freed: {freed:.2f} GB "
          f"(MLX active now: {mx.get_active_memory() / 1024**3:.2f} GB)")

    if args.held_out:
        rows = load_corpus(args.held_out, 10**9)[: args.chunks]
        nlls = []
        for row in rows:
            ids = mx.array(tokenizer.encode(row["text"])[:1024])[None]
            out = model(ids)
            mx.eval(out)
            nlls.append(np.array(token_nll(out, ids)))
        ppl = float(np.exp(np.concatenate(nlls).mean()))
        print(f"[ppl] {ppl:.4f} (k={args.k}, {sum(n.size for n in nlls)} tokens)")
        out_path = REPO_ROOT / "runs" / f"runtime_k{args.k}.json"
        out_path.write_text(json.dumps({"k": args.k, "ppl": ppl, "freed_gb": freed}, indent=2))
        print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
