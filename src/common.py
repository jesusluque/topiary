"""Shared helpers: seeds, corpus I/O, MoE block discovery, salience block orders.

Everything here is model-agnostic: MoE blocks are found by *content* (a module
holding a gate/router plus stacked 3-D expert tensors), never by hardcoded names,
so the tools work across Qwen3-MoE, OLMoE, Mixtral-style and hybrid architectures
as served by mlx-lm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

GATE_NAMES = ("gate", "router", "gate_proj", "wg")
EXPERTS_NAMES = ("switch_mlp", "experts", "mlp", "moe")


def set_seeds(seed: int) -> None:
    mx.random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------- corpus I/O


def load_corpus(path: Path, limit_tokens: int) -> list[dict[str, Any]]:
    """Read calibration rows ({"text", "n_tokens"} per line) up to a token budget."""
    rows: list[dict[str, Any]] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            total += row["n_tokens"]
            if total >= limit_tokens:
                break
    return rows


def token_nll(logits: mx.array, ids: mx.array) -> mx.array:
    """Teacher-forced per-token NLL. logits [1,L,V], ids [1,L] -> [L-1]."""
    lp = mx.log(mx.softmax(logits[0, :-1].astype(mx.float32), axis=-1) + 1e-12)
    return -mx.take_along_axis(lp, ids[0, 1:][:, None], axis=-1).squeeze(-1)


# ---------------------------------------------------------- MoE introspection


def _find_child(module: nn.Module, candidates: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in candidates:
        child = module.get(name) if hasattr(module, "get") else None
        if isinstance(child, nn.Module):
            return name, child
    return None, None


def _has_stacked_experts(module: nn.Module) -> bool:
    """A stacked-experts container holds at least one 3-D tensor [n_experts, out, in]."""
    for _, value in module.items():
        if isinstance(value, mx.array) and value.ndim == 3:
            return True
        if isinstance(value, nn.Module) and _has_stacked_experts(value):
            return True
    return False


def moe_block_of(layer: nn.Module) -> nn.Module | None:
    """Return the MoE block inside a decoder layer, or None for dense layers."""
    for _, cand in layer.items():
        if not hasattr(cand, "items"):
            continue
        gate_name, _ = _find_child(cand, GATE_NAMES)
        _, experts = _find_child(cand, EXPERTS_NAMES)
        if gate_name and experts is not None and _has_stacked_experts(experts):
            return cand
    return None


def find_moe_blocks(model) -> list[tuple[int, nn.Module]]:
    """All (layer_index, moe_block) pairs of a loaded model."""
    blocks = []
    for i, layer in enumerate(model.model.layers):
        block = moe_block_of(layer)
        if block is not None:
            blocks.append((i, block))
    return blocks


def experts_of(block: nn.Module) -> nn.Module:
    _, experts = _find_child(block, EXPERTS_NAMES)
    return experts


def dequant_stack(proj) -> mx.array:
    """Dequantize a stacked expert projection -> [n_experts, out, in]."""
    w = mx.dequantize(
        proj["weight"],
        proj["scales"],
        proj.get("biases"),
        group_size=proj.group_size,
        bits=proj.bits,
        mode=getattr(proj, "mode", "affine"),
    )
    mx.eval(w)
    return w


# -------------------------------------------------------------- block orders


def block_orders(orders_path, layer_idx: int, group_size: int) -> np.ndarray:
    """Block order by aggregated salience: [n_experts, n_blocks], best block first.

    A "block" is one quantization group of `group_size` consecutive neurons; the
    block variant permutes whole groups so packed words and scales move together.
    """
    data = np.load(orders_path)
    sal = data[f"salience_{layer_idx}"]  # [E, inter]
    n_experts, inter = sal.shape
    n_blocks = inter // group_size
    per_block = sal.reshape(n_experts, n_blocks, group_size).sum(axis=2)
    return np.argsort(-per_block, axis=1)
