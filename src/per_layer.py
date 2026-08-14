"""Loader shim for checkpoints with PER-LAYER intermediate widths.

A checkpoint whose layers have different widths is not expressible in the
standard config (a single `intermediate_size` / `moe_intermediate_size`), so
stock mlx-lm would build the wrong shapes. The fix is small: decoder blocks are
constructed in layer order, so patching the block constructor to consume a
`*_per_layer` list from the config is sufficient — quantization and weight
loading then see the correct shapes with no further changes.

Call before EVERY mlx_lm.load(), including for uniform models (the call is what
re-binds or clears the active width list, so a per-layer model loaded earlier
in the process cannot leak its widths into the next load):

    from per_layer import maybe_patch
    maybe_patch(model_path_or_repo)
    model, tokenizer = load(model_path_or_repo)

Covers the qwen3 (dense) and qwen3_moe families. Models that interleave dense
and sparse layers (`mlp_only_layers` / `decoder_sparse_step`) are rejected
explicitly: there the constructor-call order does not match the decoder-layer
index and the widths would silently shift.
"""

from __future__ import annotations

import json
from pathlib import Path

# Active per-layer lists (None = uniform model, patched ctors fall through to
# the stock behavior). Rebound on every maybe_patch() call.
_STATE: dict = {"dense": None, "moe": None, "i_dense": 0, "i_moe": 0,
                "installed_dense": False, "installed_moe": False}


def _read_config(model: str) -> dict | None:
    p = Path(model)
    if p.exists():
        cfg_path = p / "config.json"
        return json.load(open(cfg_path)) if cfg_path.exists() else None
    # HF repo: fetch just the config (works on a cold cache too — downloading
    # only config.json, not the shards; a silent no-op here would make the
    # subsequent full load crash with an opaque shape mismatch).
    try:
        from huggingface_hub import hf_hub_download

        return json.load(open(hf_hub_download(model, "config.json")))
    except Exception:
        return None


def maybe_patch(model: str) -> bool:
    """Bind (or clear) the per-layer width lists for the next load.
    Idempotent per call; safe across multiple loads in one process.
    Returns True if the next load will use per-layer widths."""
    cfg = _read_config(model)
    ks_dense = (cfg or {}).get("intermediate_size_per_layer")
    ks_moe = (cfg or {}).get("moe_intermediate_size_per_layer")

    # Rebind state FIRST: this also clears any previous model's lists, so a
    # uniform model loaded after a per-layer one gets stock construction.
    _STATE.update({"dense": ks_dense, "moe": ks_moe, "i_dense": 0, "i_moe": 0})
    if not ks_dense and not ks_moe:
        return False

    if ks_moe and cfg is not None:
        sparse_step = int(cfg.get("decoder_sparse_step", 1) or 1)
        if cfg.get("mlp_only_layers") or sparse_step > 1:
            raise SystemExit(
                "[per-layer] this loader only supports fully-sparse MoE models: "
                "with interleaved dense layers the constructor order does not "
                "match layer indices and widths would silently shift")

    if ks_dense and not _STATE["installed_dense"]:
        import mlx_lm.models.qwen3 as q3

        original = q3.MLP.__init__

        def patched_dense(self, dim, hidden_dim):
            ks = _STATE["dense"]
            if ks is None:
                return original(self, dim, hidden_dim)
            i = _STATE["i_dense"]
            if i >= len(ks):
                raise RuntimeError(f"[per-layer] MLP #{i} built but only "
                                   f"{len(ks)} widths declared")
            _STATE["i_dense"] = i + 1
            original(self, dim, ks[i])

        q3.MLP.__init__ = patched_dense
        _STATE["installed_dense"] = True

    if ks_moe and not _STATE["installed_moe"]:
        import copy

        import mlx_lm.models.qwen3_moe as qm

        original_moe = qm.Qwen3MoeSparseMoeBlock.__init__

        def patched_moe(self, args):
            ks = _STATE["moe"]
            if ks is None:
                return original_moe(self, args)
            i = _STATE["i_moe"]
            if i >= len(ks):
                raise RuntimeError(f"[per-layer] MoE block #{i} built but only "
                                   f"{len(ks)} widths declared")
            _STATE["i_moe"] = i + 1
            a = copy.copy(args)
            a.moe_intermediate_size = ks[i]
            original_moe(self, a)

        qm.Qwen3MoeSparseMoeBlock.__init__ = patched_moe
        _STATE["installed_moe"] = True

    spans = [f"{kind} {min(ks)}-{max(ks)}x{len(ks)}"
             for kind, ks in (("dense", ks_dense), ("moe", ks_moe)) if ks]
    print(f"[per-layer] active: {', '.join(spans)}")
    return True
