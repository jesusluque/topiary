---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: mlx
tags:
  - mlx
  - moe
  - topiary
  - quantized
---

# {MODEL_NAME} — Topiary {K}-width

A **Topiary** checkpoint: the routed experts of [{BASE_MODEL}]({BASE_URL}) were
sculpted by *routed activation salience* — each expert's intermediate neurons
ranked by measured contribution on real traffic, permuted (a free,
function-preserving transformation) and truncated to their top-{K} prefix —
then quantized. No training, no distillation.

**This is a standard mlx-lm checkpoint.** It loads and serves with stock
`mlx-lm` — no custom code:

```bash
pip install mlx-lm
mlx_lm.generate --model {HF_REPO} --prompt "..."
```

## Why

At equal memory, sculpting width beats lowering bits: quantization blurs *every*
weight equally, while salience truncation concentrates the damage on what the
model barely used. See the [Topiary repository]({GITHUB_URL}) for the method,
the full evaluation and the tools to build your own.

## Results ({SIZE_GB} GB, Apple Silicon)

| Signal | This model | Community 3-bit | Mixed 3–4-bit |
|---|---|---|---|
| Code PPL ↓ | **{PPL_CODE}** | 3.26 | 3.07 |
| WikiText PPL ↓ | **{PPL_WIKI}** | 15.7 | 13.4 |
| GSM8K | **{GSM8K}** | 88% | 82% |
| MMLU | **{MMLU}** | 57% | 59% |
| HumanEval | **{HUMANEVAL}** | 76% | — |
| IFEval | **{IFEVAL}** | 68% | — |

**Honest limits** (where the 3-bit keeps an edge): distribution tails —
MATH-500 (46% vs {MATH500}) and HellaSwag (66% vs {HELLASWAG}). Bit reduction
keeps every capability blurry-but-alive; width truncation sacrifices the tails
it never saw in calibration. Pick per use case.

## Calibration

Mixed corpus: ~40% code, ~30% GSM8K-train, ~30% WikiText, {CALIB_TOKENS} tokens,
routed-only statistics. The calibration corpus is the method's main dial —
recalibrating on your own domain data is cheap (minutes) and supported by the
tools in the repository.

## Provenance

- Base: {BASE_MODEL} (Apache-2.0)
- Build: `convert_fine.py` (neuron-granularity permutation of the original bf16,
  truncation to k={K}, then 4-bit g64 quantization), orders from routed salience.
- All numbers reproducible from the frozen configs in the repository.
