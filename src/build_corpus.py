"""Build a calibration corpus with the target model's own tokenizer.

Chunks input text into ~fixed-token rows and splits calib/held-out. The corpus
choice is the method's main sensitivity — it acts as an amplitude-vs-
specialization dial (see paper §3.5): a mixed corpus (code + math + general
text) gave the best all-round checkpoints in our evaluation; a single-domain
corpus sharpens that domain at the cost of others.

Input: one or more --input paths, each either a .jsonl with {"text": ...} rows
or a plain-text/source file. Rows are interleaved round-robin across inputs so
every domain is represented at any token budget.

Usage:
    python src/build_corpus.py --tokenizer mlx-community/Qwen3-30B-A3B-4bit \
        --input my_code.jsonl --input wikitext.jsonl \
        --tokens 400000 --chunk-tokens 1024 --out data/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_texts(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
    return [path.read_text(encoding="utf-8")]


def chunk(texts: list[str], tokenizer, chunk_tokens: int) -> list[dict]:
    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        for i in range(0, len(ids), chunk_tokens):
            piece = ids[i : i + chunk_tokens]
            if len(piece) < 64:
                continue
            rows.append({"text": tokenizer.decode(piece), "n_tokens": len(piece)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Topiary calibration corpus builder")
    parser.add_argument("--tokenizer", required=True, help="model repo/path whose tokenizer to use")
    parser.add_argument("--input", action="append", required=True,
                        help="jsonl or text file; repeat for a mixed corpus")
    parser.add_argument("--tokens", type=int, default=400_000)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--held-out-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(args.tokenizer) if Path(args.tokenizer).exists()
                               else _snapshot(args.tokenizer))

    rng = np.random.default_rng(args.seed)
    streams = []
    for inp in args.input:
        rows = chunk(read_texts(Path(inp)), tokenizer, args.chunk_tokens)
        rng.shuffle(rows)
        streams.append(rows)
        print(f"[in] {inp}: {len(rows)} chunks")

    # Round-robin interleave so every domain appears at any budget.
    mixed, total = [], 0
    for i in range(max(len(s) for s in streams)):
        for s in streams:
            if i < len(s):
                mixed.append(s[i])
                total += s[i]["n_tokens"]
        if total >= args.tokens * (1 + args.held_out_frac):
            break

    n_held = max(1, int(len(mixed) * args.held_out_frac))
    held, calib = mixed[:n_held], mixed[n_held:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("calib.jsonl", calib), ("held_out.jsonl", held)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] {out / name}: {len(rows)} chunks, {sum(r['n_tokens'] for r in rows)} tokens")


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=["tokenizer*", "*.json", "merges.txt",
                                                        "vocab.json", "chat_template.jinja"]))


if __name__ == "__main__":
    main()
