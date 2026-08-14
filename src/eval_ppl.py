"""Standard perplexity evaluator: the equal-bytes head-to-head.

Loads any model (HF repo or local path) and measures teacher-forced PPL over the
same held-out chunks, plus a short greedy generation as a sanity check. No
patches, no baking: what is measured is exactly the checkpoint as it would be
served.

Usage:
    python src/eval_ppl.py --model mlx-community/Qwen3-30B-A3B-3bit --data data/held_out.jsonl
    python src/eval_ppl.py --model models/qwen3-30b-4bit-w640 --data data/held_out.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from common import REPO_ROOT, load_corpus, set_seeds, token_nll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="held-out jsonl (see build_corpus.py)")
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    set_seeds(1234)
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from per_layer import maybe_patch

    maybe_patch(args.model)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    n_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    print(f"[load] {args.model}: {n_bytes / 1e9:.2f} GB of weights in {time.perf_counter() - t0:.0f}s")

    rows = load_corpus(args.data, 10**9)[: args.chunks]
    nlls = []
    for row in rows:
        ids = mx.array(tokenizer.encode(row["text"])[: args.max_tokens])[None]
        out = model(ids)
        mx.eval(out)
        nlls.append(np.array(token_nll(out, ids)))
        del out
    nll = np.concatenate(nlls)
    ppl = float(np.exp(nll.mean()))
    print(f"[ppl] {ppl:.4f} over {nll.size} tokens ({len(rows)} chunks)")

    # Qualitative sanity: 40 greedy tokens.
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    text = generate(
        model, tokenizer,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a Python function that reverses a list."}],
            add_generation_prompt=True, tokenize=False,
        ),
        max_tokens=40, sampler=make_sampler(temp=0.0),
    )
    print(f"[gen] {text[:200]!r}")

    tag = args.tag or args.model.replace("/", "_")
    out_path = REPO_ROOT / "runs" / f"ppl_{tag}.json"
    out_path.write_text(json.dumps(
        {"model": args.model, "ppl": ppl, "n_tokens": int(nll.size),
         "weight_gb": n_bytes / 1e9, "chunks": len(rows)}, indent=2))
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
