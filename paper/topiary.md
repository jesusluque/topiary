# Topiary: Salience-Shaped, Prefix-Servable MoE Checkpoints

**A training-free width axis for compressing Mixture-of-Experts models that
outperforms lower-bit quantization at equal memory — on consumer hardware, with
standard serving.**

**jesus luque**

*Draft v0.1 — August 2026*

> *First we stood on the shoulders of giants; now we ride dragons.*

## Abstract

When a quantized Mixture-of-Experts (MoE) model does not fit in memory, the
standard remedy is to reduce bits per weight. We present **Topiary**, an
alternative axis: rank each expert's intermediate neurons by *routed activation
salience*, bake the resulting permutation into the checkpoint (a free, function-
preserving transformation), and truncate every expert to a prefix of its most
important units. The output is a smaller, standard checkpoint — no custom kernels,
no inference-time code, no training or distillation. On Qwen3-30B-A3B under a
~13–14.5 GB budget on Apple Silicon, Topiary checkpoints dominate both the
community 3-bit and mixed 3–4-bit quantizations across code and general-text
perplexity, an executable-answer battery, and MMLU, while matching decode speed
within 3%; on GSM8K the outcome depends on the calibration corpus, which we
analyze as the method's main sensitivity. We map the technique's limits
empirically: it loses to bit-reduction on coarse-expert architectures (Mixtral),
degrades past a cliff at ~40–55% width cuts, and cannot bridge order-of-magnitude
compression. We further show that (i) salience must be measured on *activations* —
weight-norm proxies carry almost no signal; (ii) *routed-only* statistics beat
whole-corpus calibration and can be collected during normal serving at negligible
cost, enabling models that self-profile and self-compress in place (2.9 GB freed
in one second on a 30B model); (iii) a layer-streaming profiler makes calibration
possible for models larger than RAM (an 80B profiled with a 1.1 GB peak); and
(iv) width composes sub-additively with bit reduction. All results are
reproducible from frozen configs; all tools operate shard-by-shard in streaming.
None of the ingredients is new in isolation — function-preserving permutations,
intra-expert pruning, and routed statistics each have prior art (§5). The
contribution is a *checkpoint format* (prefix-servable, standard-loading) plus an
*empirical frontier study*: the first systematic width-versus-bits comparison at
equal bytes, including the regimes where width loses.

## 1. Method

### 1.1 Setting

A MoE feed-forward block computes, per token,
`y = Σ_{i∈topk} g_i · E_i(x)`, where each expert `E_i` is a gated MLP
(`down( act(gate(x)) ⊙ up(x) )`) with intermediate width `d_ff`. In modern MoE
families (Qwen3, OLMoE, DeepSeek-style), routed experts account for 93–95% of
checkpoint bytes, making them the only compression target that matters.

### 1.2 Routed salience

For expert `e`, neuron `i`, we define

```
salience²(e, i) = E[ h_i² | token routed to e ] · ‖W_down[:, i]‖²
```

with `h = act(gate(x)) ⊙ up(x)`. The expectation is taken **only over tokens the
router actually sends to that expert**. Two facts make this the right statistic:

- `h` is already computed in every forward pass; accumulating `Σh²` per neuron is
  one extra reduction (an 18 MB accumulator for a 30B model, <1% overhead).
- Whole-corpus calibration (evaluating every expert on every token) dilutes the
  signal with counterfactual inputs the expert never sees. Empirically, routed
  statistics produce strictly better orderings (§3.4).

### 1.3 Permute, then truncate

Permuting the intermediate neurons of an expert — the same permutation applied to
the rows of `gate/up` and the columns of `down` — leaves the network function
exactly unchanged. Channel permutation is a known tool in the quantization
literature (GPTQ's act-order, PTQ-SL, GPTQModel's GAR, PermuQuant), where it is
oriented at *reducing quantization error* while keeping every channel. Topiary
orients it at **truncability**: we sort each expert's neurons by descending
salience and bake the permutation into the checkpoint so that *dropping the file's
suffix* is the compression operation. Truncating to width `k` is then
a contiguous prefix read of every tensor, and the truncated model is a standard
checkpoint with `moe_intermediate_size = k`.

Two build variants:

- **Fine (preferred)**: permute the original bf16 weights at neuron granularity,
  truncate, then quantize. Quantization groups form *after* the reorder, so no
  extra noise is introduced and the ordering is exact.
- **Block (post-hoc)**: operate directly on already-quantized community
  checkpoints by permuting whole quantization groups (blocks of `group_size`
  neurons, i.e., entire packed words and their scales), touching no weight bits.
  Coarser ordering, zero requantization error, works for any bit width whose
  block spans an integer number of packed words (4/3/2-bit with 32-bit packing).

The measured value of each refinement on Qwen3-30B (code PPL at k=576):
block + whole-corpus calibration 3.107 → block + routed 3.024 → fine + routed
**2.869**.

### 1.4 Runtime self-compression

Because routed salience is nearly free to collect, a served model can profile
itself on live traffic and then compress *in place*: bake the permutation
(seconds of index operations) and free every expert's suffix. On Qwen3-30B this
released 2.88 GB in one second with the model serving. In our ablation the
online orders were not merely as good as offline calibration — they were better
(§3.4), so the offline pipeline is an optimization, not a requirement.

### 1.5 Profiling models larger than RAM

Calibration requires forwarding data through the model, which is impossible when
the model does not fit. We stream it **layer by layer**: load one layer (lazy
mmap), forward all calibration chunks through it (with the correct per-layer-type
attention masks — full-attention layers need causal masks; hybrid linear layers
need their SSM masks), collect that layer's expert salience, free the layer, and
continue. Peak memory is one layer plus activations: an 80B hybrid model was
profiled end-to-end with a **1.1 GB** peak. Validated against ground truth on a
model that does fit: 89% top-block agreement with full-model profiling.

### 1.6 Nested bit-planes (orthogonal, composable)

For affine group quantization, the 2-bit truncation of a 4-bit tensor is itself a
valid affine quantization: `w = s·q₄ + β` with `q₄ = 4·q_hi + q_lo` gives
`w = [(4s)·q_hi + (β + 1.5s)] + [s·q_lo − 1.5s]` (the 1.5s term corrects the
floor bias). A checkpoint can therefore be stored as `[LO plane | Δ plane]` —
one copy, two precisions, prefix-readable — with quality on par with an
independently re-fitted 2-bit model (measured: ΔPPL within noise across three
policies). Width and bit-planes compose sub-additively (§3.5).

## 2. Experimental setup

- **Hardware**: Apple M5 Pro, 24 GB unified memory, MLX / mlx-lm.
- **Models**: Qwen3-30B-A3B (48 layers, 128 experts × 768, top-8),
  OLMoE-1B-7B (64×1024, top-8), Mixtral-8x7B (8×14336, top-2),
  Qwen3-Next-80B-A3B (512×512, top-10, hybrid attention).
- **Metrics**: teacher-forced perplexity (code held-out and WikiText-2, 20×1k-token
  chunks, leak-free document-level splits); a 12-item battery with *verifiable*
  answers (generated code executed against asserts; exact numerics; required
  keywords); GSM8K (n=50) and MMLU (4 subjects, n=100) with deterministic
  subsamples and automatic scoring; decode tok/s under an interleaved-rounds
  protocol (sequential per-model timing is confounded by thermal drift and memory
  residency — we measured 2–3× artifacts before adopting interleaving).
- **Controls throughout**: greedy decoding, fixed seeds, byte-identical golden
  generations across instrumentation, Gaussian nulls for spectral claims,
  equal-byte comparisons for every head-to-head.
- **Statistical power**: at these n, a paired accuracy difference of d items has
  a minimum achievable exact-McNemar p of 2^(1−d); differences under 6 items
  (12 pp at n=50, 6 pp at n=100) cannot reach p < 0.05 under any discordance
  pattern. We therefore treat perplexities (18k/16k tokens) and the largest
  benchmark gaps as the established results, and read smaller gaps only as
  direction, supported by sign-consistency across independent builds. Where a
  gap can be significant we report exact paired McNemar from per-item re-runs
  (`eval_paired.py`): flagship vs community 3-bit — HumanEval 9/1 discordants,
  p = 0.0215; generative MMLU 18/5, p = 0.0106.

## 3. Results

### 3.1 Main result: width beats bits at equal memory (fine-expert MoE)

Qwen3-30B-A3B, all arms 13.1–14.5 GB:

| Checkpoint | GB | PPL code | PPL general | Battery | GSM8K | MMLU |
|---|---|---|---|---|---|---|
| Topiary w512 | 11.74 | 3.440 | 15.51 | — | — | — |
| **Topiary w576-fine** | **13.10** | 2.869 | 12.98 | **11/12** | 78% | **64%** |
| 3-bit (community) | 13.36 | 3.259 | 15.70 | 8/12 | 88% | 57% |
| mixed 3–4-bit | 14.00 | 3.065 | 13.38 | — | 82% | 59% |
| **Topiary w640** | **14.46** | **2.833** | **11.11** | — | **90%** | 61% |

**w640 dominates the strong mixed-precision baseline on every measured metric
at a cost of 0.46 GB more** — and combining it with mixed-corpus calibration (below) yields
the series champion, w640-mixed: PPL 2.698/10.36, GSM8K 94%, MMLU 74%, HumanEval
84%, ARC 49%, IFEval 76% — winning or tying 10 of 12 signals against all baselines,
losing only MATH-500 (38% vs 46%), which together with HellaSwag delineates the
method's empirical law: **width sacrifices distribution tails** (salience marks the
infrequent as expendable; diffuse bit noise keeps tails blurry but alive). The k-curve is monotone (w512→w576→w640 improves in both
domains), i.e., width is a continuous quality/memory dial in steps of 64 neurons.
Calibrated on code only, the orderings still transfer: w576 beats 3-bit on
WikiText by −17% PPL. Decode speed at equal bytes: 87.0 vs 89.9 tok/s (−3.2%),
under the interleaved-rounds protocol of §2.

**The GSM8K exception is the method's fingerprint.** w576 (code-calibrated)
loses 10 pp to 3-bit on grade-school math, while w640 — 64 more neurons per
expert — *leads* at 90%. The cut depth interacts with calibration coverage:
neurons under-valued by a code corpus but used by arithmetic chains fall between
k=640 and k=576. **Mixed-corpus calibration closes this completely**: re-profiling
with a code+GSM8K-train+WikiText mixture (~15% of per-expert selections change)
and rebuilding at the same k=576 yields GSM8K 92% (from 78%; above every baseline
including the larger w640) and MMLU 70% (from 64%; MMLU appears in no form in the
calibration mix, making this the clean generalization signal), with code PPL
unchanged (2.872) and general PPL improved 11% (12.98→11.50). The calibration
corpus is a first-class hyperparameter — and a well-mixed one appears to have no
downside at this cut depth.

**Composing with per-layer budgets yields the series flagship.** Applying the
depth-taper of §3.6 (ratio 0.85: per-expert width 576 in shallow layers rising
to 704 in deep ones, mean 640) to the fine build produces, at the same 14.46 GB:
code PPL 2.639, general 10.27 (both series records), HumanEval 92% (+8 over the
uniform champion, +16 over 3-bit), HellaSwag 70%, GSM8K 94%, at a small ranking
toll (MMLU 70 vs 74, ARC −1); intra-Topiary benchmark differences at these n are
individually below statistical significance (§2) — the perplexities carry the
established part of the claim, and the flagship's gaps over the 3-bit are
significant under exact paired McNemar (HumanEval p = 0.0215, MMLU p = 0.0106). The block variant of the same allocation — no bf16
needed, minutes of streaming — reaches GSM8K 98% and HumanEval 86% on its own.
Per-layer widths require a ~30-line loader shim (block construction patched in
layer order); the uniform sibling remains the zero-shim option.

Qualitatively, the failure modes of the two axes differ: over-truncation produces
*confidently wrong logic* (structural capacity loss), while aggressive
quantization produces *diffuse muddiness* (circular or vague but rarely
syntactically broken output). Executable scoring catches failures that eyeballing
misses — one baseline generated a bracket-matching function that reads perfectly
and fails every assert.

### 3.2 Where bits win: coarse experts

Mixtral-8x7B (47B, 26.3 GB at 4-bit — does not fit in 24 GB). Both arms derived
from the same 4-bit parent, non-expert weights identical, ~15 GB each:

| Arm | PPL code | PPL general |
|---|---|---|
| experts → 2-bit | **2.838** | **7.245** |
| experts → width 55% | 3.438 | 12.406 |

Not a salience failure — Mixtral's per-neuron salience is *more* concentrated
than Qwen3's (83% vs 78% energy in the 55% prefix). The cause is per-parameter
redundancy: 47B parameters serving ~13B active per token absorb diffuse
quantization noise, whereas removing 45% of units removes structural capacity no
redundancy can restore. **Rule: width for mild cuts on fine-expert MoEs; bits for
aggressive compression or monolithic experts.** (On the verifiable battery the
two arms tie 6/12 — pass-rate metrics forgive diffuse noise more than perplexity
does; report both.)

### 3.3 The compression frontier

Qwen3-Next-80B (44.8 GB at 4-bit) squeezed toward 24 GB hardware:

| 80B arm (~12–15 GB) | Battery | PPL code |
|---|---|---|
| width 25%, weight-norm salience | 1/12 | 8.52 |
| width 25%, routed salience (layer-streamed) | 2/12 | 8.39 |
| **3-bit × width 37.5% (composed)** | **5/12** | **4.70** |
| *native Qwen3-30B Topiary (13.1 GB)* | *11/12* | *2.87* |

Three boundary lessons: (i) weight-norm salience is nearly flat (25% prefix
captures 32% of norm vs 25% uniform; routed captures 56%) — hierarchy lives in
activations, not weights; (ii) at 4× per-expert cuts the cliff dominates — better
salience bought almost nothing; (iii) composing two moderate axes beats one
extreme axis by a wide margin (sub-additivity holds even here), yet at ~3×
compression the smaller native model wins decisively. That regime belongs to
distillation or vector-quantization codebooks.

### 3.4 Routed > whole-corpus calibration

Routed-only statistics formalize the same intuition as REAP's gate-weighted
activation norms at expert granularity; here we quantify it at *neuron*
granularity. On Qwen3-30B (3-bit, truncated to k=576 with each ordering, same held-out):
whole-corpus orders PPL 5.667 vs **routed orders 4.568**. Rebuilding the 4-bit
flagship with routed orders improved both domains (code 3.107→3.024, general
13.06→12.87 at block granularity). Online/offline order agreement: 82% of top
blocks; the disagreement favors online.

### 3.5 Composition and speed

Width × nested-2-bit on cold experts composes sub-additively (measured
k=256×Q2: +1.51% PPL where the naive sum of parts is +1.98%). Speed is *not*
the win: batch-1 decode on this hardware is kernel-latency-bound, not
bandwidth-bound (a 30B moves 74 GB/s of 307 available; even per-token dynamic
prefix serving — which works, with zero-copy slices at decode — costs ~10% from
dual dispatch). Topiary's currency is memory and quality, at parity speed.

### 3.6 Dense transformers: valid with a caveat

The method requires no MoE: a dense FFN is one expert per layer. On dense
Qwen3-32B (64 layers, d_ff 25600 — whose salience is *more* concentrated than the
MoE's: 90% of energy in the 71% prefix), the equal-byte duel against 3-bit splits:
width wins likelihood and ranking massively (code PPL −17%; general PPL 14.9 vs a
collapsed 37.7; MMLU-loglik +8; ARC +6; HellaSwag +5) and GSM8K, but loses
HumanEval by 34 points (52% vs 86%). PPL and generation diverge in *both*
directions across the two arms. Proposed mechanism: in a dense model every token
traverses the cut in every layer, so truncation error compounds with depth and
punishes long-form precise generation; single-step metrics do not compound. MoE
sparsity avoids this by construction (8 experts per token, milder cuts).

**Per-layer width budgets fix most of it — and the direction is the finding.** We
compared three allocations at *exactly* equal total bytes (14.36 GB each):
uniform; an equal-capture waterfill (each layer keeps enough blocks to capture the
same fraction of its own salience mass); and linear depth tapers in both
directions. Protecting shallow layers is the *worst* arm (+13.5% code PPL);
equal-capture also loses (+3.8%); **protecting deep layers wins on both domains**
(code PPL 2.978, −7.5%; general 13.09, −12%) and recovers most of the generation
collapse: HumanEval 52%→70%, GSM8K 82%→90%, HellaSwag +2, decode +10%, leaving
only long-form code (70 vs 86) and loglik-ranking (ARC/MMLU-ll) to the
alternatives. Sensitivity to truncation thus *increases* monotonically with depth
— late-layer error has no remaining layers to damp it, while early-layer error is
partially absorbed downstream. This independently corroborates, with a controlled
three-direction comparison that the original lacks, the trapezoidal allocation
that TENP (§5) adopts for MoE experts. Note the layer's own salience
concentration does *not* predict its sensitivity (equal-capture loses): ordering
quality within a layer and capacity needed by a layer are different variables.
Practical note: at equal memory, the sculpted MoE still dominates the sculpted
dense model (94%/74% at 80 tok/s vs 82%/67% at 14).

### 3.7 Update (August 2026): the knowledge cost, isolated

A four-benchmark suite run after this draft (MATH-500 n=100, MBPP n=100,
MMLU n=500 subject-stratified, LAMBADA n=500; greedy, fixed seed, identical
samples — protocol and run records in the topiary-stream companion repo)
compared the taper flagship against its own unpruned base, same-day and
same-machine:

| | Qwen3-30B-A3B original (16.4 GB) | Topiary taper (14.5 GB) |
|---|---|---|
| MATH-500 | 70% | **72%** |
| MBPP | **83%** | 81% |
| MMLU | **78.2%** | 68.2% |
| LAMBADA | **64.6%** | 60.2% |

Salience pruning preserved reasoning entirely (MATH/MBPP differences are not
significant) but **cost 10 MMLU points and 4.4 LAMBADA points** — both
strongly significant at n=500. The taper removes *knowledge*, not
*reasoning*: the earlier battery (§3.1), being reasoning-heavy and n=100 on
MMLU, could not see it. This does not change the equal-bytes comparisons
against quantization baselines (§3.1 stands), but it does change what a
sculpted checkpoint *is*: a reasoning-preserving, knowledge-lossy compression.
Same-day interleaved decode-only rounds also correct the speed claim: the
taper serves at 108.1–108.3 tok/s vs the original's 101.4–103.0 (+6%, with
−2.7 GB); the earlier "87.0 vs 89.9" compared different sessions.

## 4. Limitations

- **Knowledge loss under pruning (see §3.7):** −10 MMLU / −4.4 LAMBADA vs the
  unpruned base at n=500, invisible to reasoning-heavy batteries. If broad
  world knowledge is the workload, prefer the unpruned checkpoint or a
  Stream-served larger model.
- **Calibration-domain sensitivity** is real and measurable (GSM8K, §3.1), and
  correctable: mixed-corpus calibration recovered the full gap at zero cost in
  other metrics. Narrow corpora bias the artifact — which also cuts the other
  way: calibrating on *your* workload specializes the model to it. Note that
  distribution affinity inflates in-mix benchmarks (GSM8K-train was in the mix);
  out-of-mix signals (MMLU +6pp) are the honest evidence.
- Coarse-expert / heavily over-parameterized MoEs favor bit reduction (§3.2).
- Cuts beyond ~40% cross a quality cliff regardless of ordering quality.
- Benchmarks here are small (n=50/100) and single-hardware; the strongest
  baselines lack activation-aware (imatrix/DWQ-style) calibration, which would
  narrow — though, being orthogonal, likely not erase — the gap.
- Models with a dominant shared expert (e.g., Qwen1.5-MoE, where it absorbs 61%
  of block output) leave little addressable mass.

## 5. Related work

**Intra-expert neuron pruning.** TENP (He et al., ACL Findings 2026) is the
closest work: structured pruning of neurons inside MoE experts, preserving the
routing topology, with a *trapezoidal* per-layer budget (aggressive in shallow
layers, capacity retained in deep layers) and a projected-contribution neuron
score. Our §3.6 independently corroborates the trapezoid's direction with a
controlled equal-bytes three-direction comparison (which TENP does not report),
on a dense model. Topiary differs in deliverable and regime: TENP prunes only the
less-important experts (heterogeneous widths, full-precision models); Topiary
keeps k homogeneous so the output is a *standard servable checkpoint*, operates
on already-quantized community checkpoints, and accounts bytes. MoE-Pruner and
the one-shot scoring family surveyed in 2606.15716 — SEER-MoE (routing
frequency), Expert Activation Norm (Jaiswal et al., 2025), REAP (gate-weighted
activation norms), MoNE (frequency-weighted activation variance) — prune at
expert granularity; our routed-only statistic is REAP's conditioning applied at
neuron granularity, and §3.4 quantifies its advantage over whole-corpus
calibration. Expert-wise mixed-precision quantization (EAC-MoE and others)
allocates bits, not width.

**Weight permutation.** Function-preserving channel permutation for quantization
is established: GPTQ's act-order quantizes columns by decreasing activation
magnitude; PTQ-SL adjusts channel order for fine-grained quantization; GAR
(GPTQModel) restricts permutations to within-group or whole-group reorderings —
mechanically our block variant — keeping scales addressable with no inference
overhead; PermuQuant (2605.09503) orders channels by a second-moment criterion
with permutations absorbed offline. All aim at *precision*; none orients the
permutation at making the file *truncatable*. That orientation, and the resulting
prefix-servable format, is Topiary's delta.

**Trained elasticity.** Nested-width transformers exist as trained architectures
(MatFormer, Flextron), and Star Elastic (2605.07182) demonstrates the full
program on our model class — importance-ranked nesting across embedding, attention,
SSM, MoE-expert and FFN axes on Nemotron Nano v3 30B/3.6A, with a learned router,
curriculum distillation and quantization-aware distillation, at a cost of ~160B
training tokens. These set the quality ceiling for elasticity; Topiary is the
training-free, post-hoc floor that the existing checkpoint population — which
will never be retrained — can have today, at roughly six orders of magnitude less
compute. Any-Precision LLM builds nested bit-plane models with custom formats;
our §1.6 shows affine group quantization already contains this structure up to a
bias correction.

## 6. Future work

The trained and post-hoc routes to prefix-servability are convergent rather than
competing. A MatFormer-style model makes Topiary's ordering trivial (it is born
sorted) and flattens the truncation cliff; conversely, Topiary is what the
existing population of checkpoints — which will never be retrained as nested
models — can have today. The bridge is cheap: a light training-time
regularizer that penalizes unsorted per-expert salience (encouraging importance
to decay monotonically along the intermediate dimension) would make every
released checkpoint "Topiary-ready" by construction, with deep cuts viable at
near-MatFormer quality and zero inference-time machinery. We believe this is the
highest-leverage follow-up, alongside: carrying the per-layer allocation of §3.6
back to MoE (TENP shows per-layer budgets matter there; our directional result
predicts the profile), a head-to-head of neuron scores (TENP's projected
contribution vs our routed energy) at equal k, and an mmap serving mode in which
baked suffixes page from disk on demand — making RAM scale with the working set
rather than the file.

## 7. Reproducibility

All tools are streaming (no full-model materialization), MIT-licensable Python
over MLX: `convert_fine.py` (permute-then-quantize build), `convert_block.py`
(post-hoc on quantized checkpoints), `allocate.py` (per-layer width budgets),
`runtime.py` (online profiling and in-place compression), `profile_stream.py`
(larger-than-RAM profiling), `per_layer.py` (loader shim for per-layer widths),
and the evaluation suite (`eval_ppl.py`, `eval_paired.py` with exact McNemar).
The lab repository additionally holds the full instrumentation (router
telemetry, dynamic prefix serving, legacy checkpoint modernization) behind the
ablations reported here.
Every experiment writes a frozen config and metrics JSON. Engineering notes that
cost us real debugging time — kernel contiguity requirements of sorted
`gather_qmm`, per-module quantization overrides in configs, per-layer-type masks
for layer-streaming, interleaved timing protocols — are documented in the
repository.

## Acknowledgments

Developed and measured end-to-end on a single MacBook Pro (M5 Pro, 24 GB) in one
extended session; every negative result retained its run and its config.
