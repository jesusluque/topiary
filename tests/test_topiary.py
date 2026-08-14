"""Self-contained unit tests: no model downloads, no GPU work.

Covers the pure-logic core: the exact McNemar test, the per-layer width
allocator (budget conservation, infeasibility errors), and the per-layer
loader's state machine across successive loads.
"""

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from allocate import alloc_taper  # noqa: E402
from eval_paired import mcnemar_exact  # noqa: E402


# ------------------------------------------------------------- exact McNemar


def test_mcnemar_known_values():
    # 9/1 discordants -> p = 2*(C(10,9)+C(10,10))/2^10 = 0.0215
    a = {f"i{k}": 1 for k in range(9)} | {"x": 0} | {f"c{k}": 1 for k in range(37)}
    b = {f"i{k}": 0 for k in range(9)} | {"x": 1} | {f"c{k}": 1 for k in range(37)}
    r = mcnemar_exact(a, b)
    assert (r["discordant_a_wins"], r["discordant_b_wins"]) == (9, 1)
    assert abs(r["p_exact"] - 0.021484375) < 1e-12

    # 18/5 -> the value quoted in the paper (0.0106)
    a = {f"i{k}": 1 for k in range(18)} | {f"j{k}": 0 for k in range(5)}
    b = {f"i{k}": 0 for k in range(18)} | {f"j{k}": 1 for k in range(5)}
    expected = 2 * sum(comb(23, k) for k in range(18, 24)) / 2**23
    assert abs(mcnemar_exact(a, b)["p_exact"] - expected) < 1e-12
    assert round(expected, 4) == 0.0106


def test_mcnemar_edge_cases():
    same = {"a": 1, "b": 0}
    assert mcnemar_exact(same, same)["p_exact"] == 1.0
    # the p_min = 2^(1-d) bound at d=4
    a = {f"i{k}": 1 for k in range(4)}
    b = {f"i{k}": 0 for k in range(4)}
    assert abs(mcnemar_exact(a, b)["p_exact"] - 0.125) < 1e-12
    assert mcnemar_exact({"a": 1, "only_a": 1}, {"a": 0, "only_b": 0})["n"] == 1


# ------------------------------------------------------------------ allocator


def test_taper_conserves_budget_and_direction():
    n = alloc_taper(48, 12, 480, ratio=0.85, lo_frac=0.75, hi_frac=1.0)
    assert n.sum() == 480
    assert n[0] < n[-1], "ratio < 1 must give deep layers more width"
    assert n.min() >= 9 and n.max() <= 12


def test_taper_infeasible_budget_raises():
    with pytest.raises(AssertionError):
        alloc_taper(48, 12, 384, ratio=0.85, lo_frac=0.75, hi_frac=1.0)
    with pytest.raises(AssertionError):
        alloc_taper(48, 12, 999, ratio=0.85, lo_frac=0.75, hi_frac=1.0)


# ------------------------------------------------------- per-layer loader


def _mkcfg(tmp_path, name, d):
    p = tmp_path / name
    p.mkdir()
    (p / "config.json").write_text(json.dumps(d))
    return str(p)


def test_loader_state_transitions(tmp_path):
    import per_layer as pl

    A = _mkcfg(tmp_path, "A", {"moe_intermediate_size_per_layer": [576] * 48})
    B = _mkcfg(tmp_path, "B", {"moe_intermediate_size": 640})
    C = _mkcfg(tmp_path, "C", {"moe_intermediate_size_per_layer": [512] * 48})

    assert pl.maybe_patch(A) is True and pl._STATE["moe"] == [576] * 48
    # a uniform model after a per-layer one CLEARS the state
    assert pl.maybe_patch(B) is False and pl._STATE["moe"] is None
    # a different list re-binds and resets the counter
    assert pl.maybe_patch(C) is True
    assert pl._STATE["moe"] == [512] * 48 and pl._STATE["i_moe"] == 0


def test_loader_rejects_interleaved_models(tmp_path):
    import per_layer as pl

    D = _mkcfg(tmp_path, "D", {"moe_intermediate_size_per_layer": [576] * 48,
                               "mlp_only_layers": [0]})
    with pytest.raises(SystemExit):
        pl.maybe_patch(D)
