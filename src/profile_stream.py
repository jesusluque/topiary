"""Layer-streaming routed-salience profiler: calibrate models LARGER than RAM.

Lazy (mmap) loading, layer-by-layer forward keeping only the chunk activations
between layers, each layer released after use. Peak memory ~ one layer: an 80B
model was profiled with a 1.1 GB peak in our runs.

Correct masks per layer type (causal for full attention, ssm for the linear
layers of hybrids): with mask=None the prefill would be non-causal and the
measured salience silently garbage.

Validation: --validate compares the resulting orders against a reference .npz
(e.g. from runtime.py on a model that does fit). Run the validation on a
verifiable model BEFORE spending the tool on one you cannot check.

Usage:
    python src/profile_stream.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
        --data data/calib.jsonl --tokens 24000 --out runs/orders_80b.npz
"""

from __future__ import annotations

import argparse
import gc

import mlx.core as mx
import numpy as np

from common import dequant_stack, experts_of, load_corpus, moe_block_of, set_seeds


class _Passthrough:
    def __call__(self, x, *a, **k):
        return x


class _CallShim:
    """Wraps down_proj to capture its INPUT (which is exactly h) with its expert
    indices, then delegates to the real module."""

    def __init__(self, mod, acc):
        self._mod, self._acc = mod, acc

    def __call__(self, x, indices, sorted_indices=False):
        h2 = (x.astype(mx.float32) ** 2).reshape(-1, x.shape[-1])
        idx = indices.reshape(-1)
        mx.eval(h2)
        np.add.at(self._acc["h2"], np.array(idx), np.array(h2))
        np.add.at(self._acc["count"], np.array(idx), 1)
        return self._mod(x, indices, sorted_indices=sorted_indices)

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def __getitem__(self, k):
        return self._mod[k]

    def get(self, k, default=None):
        return self._mod.get(k, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary streaming profiler (models > RAM)")
    parser.add_argument("--src", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokens", type=int, default=24000)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validate", help="reference .npz to measure order overlap against")
    args = parser.parse_args()

    set_seeds(1234)
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.utils import load

    model, tokenizer = load(args.src, lazy=True)
    inner = model.model
    n_layers = len(inner.layers)

    try:  # hybrid architectures only (e.g. Qwen3-Next)
        from mlx_lm.models.qwen3_next import create_ssm_mask
    except ImportError:
        create_ssm_mask = None

    rows = load_corpus(args.data, args.tokens)
    chunks = []
    for r in rows:
        ids = tokenizer.encode(r["text"])[: args.chunk_len]
        if len(ids) >= 64:
            chunks.append(mx.array(ids)[None])
    print(f"[stream] {len(chunks)} chunks (~{sum(c.shape[1] for c in chunks)} tokens), "
          f"{n_layers} layers")

    xs, fa_masks, ssm_masks = [], [], []
    for c in chunks:
        h = inner.embed_tokens(c)
        mx.eval(h)
        xs.append(h)
        fa_masks.append(create_attention_mask(h, None))
        ssm_masks.append(create_ssm_mask(h, None) if create_ssm_mask else None)

    out: dict[str, np.ndarray] = {}
    for li in range(n_layers):
        layer = inner.layers[li]
        block = moe_block_of(layer)
        acc = None
        if block is not None:
            experts = experts_of(block)
            n_e = experts["gate_proj"]["weight"].shape[0]
            inter = experts["gate_proj"].scales.shape[1]
            acc = {"h2": np.zeros((n_e, inter), np.float64), "count": np.zeros(n_e, np.int64)}
            real_down = experts["down_proj"]
            experts["down_proj"] = _CallShim(real_down, acc)

        is_linear = bool(getattr(layer, "is_linear", False))
        new_xs = []
        for h, fam, ssm in zip(xs, fa_masks, ssm_masks):
            y = layer(h, mask=(ssm if is_linear else fam), cache=None)
            mx.eval(y)
            new_xs.append(y)
        xs = new_xs

        if acc is not None:
            experts["down_proj"] = real_down
            wd = dequant_stack(real_down)
            col2 = np.array((wd.astype(mx.float32) ** 2).sum(axis=1))
            del wd
            sal = (acc["h2"] / np.maximum(acc["count"], 1)[:, None] * col2).astype(np.float32)
            out[f"salience_{li}"] = sal
            out[f"layer_{li}"] = np.argsort(-sal, axis=1).astype(np.int32)

        inner.layers[li] = _Passthrough()
        del layer, block
        gc.collect()
        mx.clear_cache()
        if (li + 1) % 8 == 0:
            print(f"  layer {li + 1}/{n_layers} (MLX active {mx.get_active_memory() / 1e9:.1f} GB)")

    np.savez_compressed(args.out, **out)
    curves = []
    for k in out:
        if k.startswith("salience_"):
            s = np.sort(out[k].astype(np.float64), axis=1)[:, ::-1]
            curves.append((np.cumsum(s, 1) / np.maximum(s.sum(1, keepdims=True), 1e-30)).mean(0))
    c = np.mean(curves, axis=0)
    inter = c.size
    for f in (0.25, 0.375, 0.5):
        print(f"[salience] {f:.0%} prefix captures {100 * c[int(f * inter) - 1]:.1f}%")
    print(f"[out] {args.out}")

    if args.validate:
        ref = np.load(args.validate)
        overlaps = []
        for k in out:
            if not k.startswith("salience_"):
                continue
            li = k.split("_")[1]
            if f"salience_{li}" not in ref:
                continue
            n_keep = max(1, inter * 3 // 4 // 64)  # top-75% of 64-neuron blocks
            a = np.argsort(-out[k].reshape(out[k].shape[0], -1, 64).sum(2), axis=1)[:, :n_keep]
            b = np.argsort(-ref[f"salience_{li}"].reshape(out[k].shape[0], -1, 64).sum(2),
                           axis=1)[:, :n_keep]
            for e in range(a.shape[0]):
                overlaps.append(len(set(a[e]) & set(b[e])) / n_keep)
        print(f"[validate] top-75% block overlap vs reference: "
              f"{100 * float(np.mean(overlaps)):.1f}% (min {100 * float(np.min(overlaps)):.0f}%)")


if __name__ == "__main__":
    main()
