# Topiary 🌳

> *First we stood on the shoulders of giants; now we ride dragons.*

**Salience-shaped, prefix-servable MoE checkpoints.**
Better quality per gigabyte than lowering bits — no training, no distillation,
minutes on consumer hardware, operating on community quantized checkpoints, and
the output is a **standard mlx-lm model** that loads with zero custom code.

```
Qwen3-30B-A3B at 14.46 GB (Apple Silicon, MLX) — equal-bytes suite:

                        PPL code/gen   GSM8K   MMLU   HumanEval   HSwag
  Topiary (per-layer)   2.64/10.3      94%     70%    92%         70%
  Topiary w640 uniform  2.70/10.4      94%     74%    84%         65%
  community 3-bit       3.26/15.7      88%     57%    76%         66%

  Flagship = neuron-granularity salience + a depth-tapered per-layer budget
  (576 shallow -> 704 deep; measured law: cut sensitivity grows with depth).
  The 3-bit keeps MATH-500 (distribution tails); the uniform sibling keeps
  MMLU-style ranking and zero-shim loading — see "Honest limits".
```

📄 **[Paper draft](paper/topiary.md)** · 🌐 **[Project page](https://jesusluque.github.io/topiary/)** · 🤗 Models: [flagship](https://huggingface.co/jesusluque/qwen3-30b-topiary) · [uniform w640](https://huggingface.co/jesusluque/qwen3-30b-topiary-w640) · [code w576](https://huggingface.co/jesusluque/qwen3-30b-topiary-w576-code) (private until release)

## The idea, in three steps

1. **Measure.** The importance of every neuron of every expert is estimated from
   real traffic: `salience²ᵢ = E[h²ᵢ] · ‖W_down[:,i]‖²`, accumulated **only over
   the tokens the router actually sends to that expert**. The routed statistic is
   nearly free to collect inside the forward pass — and beats classic whole-corpus
   calibration.
2. **Reorder.** Permuting an expert's intermediate neurons is *free*: the model's
   function is unchanged if the rows of `gate/up` and the columns of `down` are
   permuted identically. The permutation is baked into the file, most-important
   first.
3. **Truncate.** Keep the prefix. Because the file is sorted by importance,
   cutting bytes = cutting what contributed least. `k` is a dial in steps of one
   quantization group (64 neurons).

The damage concentrates on what the model barely used, whereas removing a bit
degrades *every* weight equally. That asymmetry is the entire advantage.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# 1. calibration corpus with the target model's tokenizer
python src/build_corpus.py --tokenizer mlx-community/Qwen3-30B-A3B-4bit \
    --input your_code.jsonl --input general_text.jsonl --out data/

# 2. routed salience — pick one:
python src/runtime.py --model mlx-community/Qwen3-30B-A3B-4bit \
    --data data/calib.jsonl --profile-only --orders-out runs/orders.npz
python src/profile_stream.py --src <repo> --data data/calib.jsonl \
    --out runs/orders.npz              # for models larger than RAM

# 3. per-layer width budget (recommended: gentle depth taper, ratio 0.85)
python src/allocate.py --n-layers 48 --width 768 --k-mean 640 --ratio 0.85 \
    --out runs/k_alloc.json

# 4. convert (block variant: works directly on the quantized checkpoint)
python src/convert_block.py --src mlx-community/Qwen3-30B-A3B-4bit \
    --orders runs/orders.npz --k-json runs/k_alloc.json --out models/my-model

# 5. always measure
python src/eval_ppl.py --model models/my-model --data data/held_out.jsonl
```

Per-layer checkpoints need a ~30-line loader shim (`src/per_layer.py`, called
automatically by the eval tools; add one `maybe_patch()` call in your own
serving code). Uniform-k builds (`--k`) load with stock mlx-lm, zero shim.

For the best results use the **fine variant** (needs the original bf16):
`convert_fine.py` permutes at neuron granularity *before* quantizing, so the
groups form after the reorder and the ordering is exact.

## Tools

All streaming (the full model is never resident), all under the same controls
(fixed seeds, greedy generation, swap watchdogs).

| Tool | What it does |
|---|---|
| `src/runtime.py` | **Self-profiling + self-compression**: salience accumulated during normal serving, then the live model permutes and truncates itself in RAM (2.9 GB freed in ~1 s on a 30B). `--profile-only` exports orders for the converters. |
| `src/profile_stream.py` | Layer-streaming profiler for models **larger than RAM** (an 80B profiled with a 1.1 GB peak). |
| `src/convert_fine.py` | **The full build**: permute the original bf16 per neuron, truncate, quantize after. Best quality. |
| `src/convert_block.py` | Convert already-quantized checkpoints (block permutation, not a single bit touched). |
| `src/allocate.py` | Per-layer width budgets (depth taper — measured: protect deep layers; the reverse is the worst allocation). |
| `src/per_layer.py` | Loader shim for per-layer-width checkpoints (qwen3 / qwen3_moe families). |
| `src/build_corpus.py` | Mixed calibration corpus builder (the corpus is the method's main dial). |
| `src/eval_ppl.py` | Teacher-forced PPL + sanity generation on any checkpoint, as served. |
| `src/eval_paired.py` | Per-item HumanEval/MMLU + exact McNemar (flagship vs 3-bit: p = 0.0215 / 0.0106). |

## Results

### Where Topiary wins

Qwen3-30B-A3B (128 fine-grained experts of width 768), mild compression (~25%),
equal-bytes head-to-head:

| Checkpoint | GB | Code PPL | WikiText PPL |
|---|---|---|---|
| w512 | 11.7 | 3.44 | 15.5 |
| **w576-fine** | **13.1** | **2.87** | **13.0** |
| full 3-bit | 13.4 | 3.26 | 15.7 |
| mixed 3–4-bit | 14.0 | 3.07 | 13.4 |
| w640 (uniform) | 14.5 | 2.83 | 11.1 |
| **w640 per-layer taper** | **14.5** | **2.64** | **10.3** |

It generalizes across domains (calibrated on code, still −17% PPL on WikiText).
The improvement chain, measured step by step: block permutation + classic
calibration 3.11 → + routed statistics 3.02 → **neuron granularity 2.87**.

### Honest limits (all measured, not speculated)

- **Coarse-expert architectures** (Mixtral, 8 experts of width 14336): bits win.
  Over-parameterized experts absorb diffuse quantization noise better than they
  survive amputation. Failure modes differ visibly: bits → diffuse blur;
  excessive width cuts → confident-but-wrong logic.
- **Aggressive cuts**: quality falls off a cliff between 40–55% width reduction.
- **~3× compression** (80B → 14 GB): composing bits × width improves 5× over
  either axis alone, but a smaller native model still wins. That regime belongs
  to distillation.
- **Distribution tails**: benchmarks probing rare capabilities (MATH-500,
  HellaSwag) favor bit-reduction, which keeps everything blurry-but-alive.
  Width truncation sacrifices what calibration never exercised.
- **Speed is not the lever**: at equal bytes, ±3% tokens/s vs quantization.
  Batch-1 decode on Apple Silicon is not memory-bound even at 30B.

### Cross-cutting findings

- **Hierarchy lives in ACTIVATIONS, not weights** — measured three ways: weight
  spectra are near-Gaussian, weight-only salience is nearly flat, routed
  statistics beat whole-corpus calibration.
- **Routed-only calibration** can be collected during normal serving at <1%
  overhead → models that self-profile and self-compress in place.
- **No clusters**: neurons and experts are near-orthogonal by design; what works
  does so by ORDER, not by merging.
- **Cut sensitivity grows with depth** — controlled three-direction comparison at
  equal bytes, on dense (64 layers) and MoE (48 layers): protecting deep layers
  wins everywhere; protecting shallow layers is the worst possible allocation.
  A gentle taper (ratio ~0.85) beats an aggressive one. Independently
  corroborates TENP's trapezoidal budget with a control it does not report.

## Repository map

```
paper/topiary.md   the paper draft (method, evaluation, limits, related work)
src/               the tools above, self-contained
tests/             unit tests (McNemar, allocator, loader state) — run in CI
huggingface/       model card template + private upload script
docs/              project page (GitHub Pages)
```

## Engineering notes (learned the hard way)

- `gather_qmm` with `sorted_indices=True` requires **contiguous** tensors; lazy
  slices read garbage without erroring.
- Quantized configs carry **per-module overrides** (e.g. routers at 8-bit):
  always preserve them and resolve bits per module, the way the loader does.
- Layer-streaming forwards need the **per-layer-type masks** (causal / ssm in
  hybrids); `mask=None` silently makes prefill non-causal.
- Speed benchmarks only with interleaved rounds: consecutive per-config repeats
  confound policy with memory residency and thermal state.

## Status

Private preview. The three checkpoints above are uploaded (private); method and
models go public together.

## Citing Topiary

```bibtex
@misc{luque2026topiary,
  author       = {luque, jesus},
  title        = {Topiary: Salience-Shaped, Prefix-Servable MoE Checkpoints},
  year         = {2026},
  howpublished = {\url{https://github.com/jesusluque/topiary}},
  note         = {Training-free width axis for MoE compression}
}
```

## License

MIT — see [LICENSE](LICENSE).
