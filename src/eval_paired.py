"""Paired evaluation with exact McNemar tests.

Aggregate benchmark scores cannot support paired significance tests; this tool
re-runs HumanEval (executable, n=50) and generative MMLU (last-letter scoring,
n=100) storing the PER-ITEM outcome vector, with fixed seeds so items are
identical across models by construction. Mode `test` computes the exact McNemar
p-value (two-sided binomial over discordant pairs).

A useful analytical bound before running anything: a difference of d items has a
minimum achievable p of 2^(1-d) — comparisons closer than 6 items (12 pp at
n=50, 6 pp at n=100) cannot reach p < 0.05 under ANY discordance pattern.

Modes:
    run:  python src/eval_paired.py run --model <path> --tag <tag>
    test: python src/eval_paired.py test runs/paired_A.json runs/paired_B.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from hashlib import sha1
from math import comb

import mlx.core as mx
import numpy as np

from common import REPO_ROOT, set_seeds

MMLU_SUBJECTS = ["college_computer_science", "high_school_mathematics",
                 "logical_fallacies", "world_religions"]


def load_mmlu(n: int):
    from datasets import load_dataset

    per = n // len(MMLU_SUBJECTS)
    rng = np.random.default_rng(1234)
    out = []
    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test")
        for i in rng.permutation(len(ds))[:per]:
            r = ds[int(i)]
            out.append((r["question"], r["choices"], "ABCD"[r["answer"]]))
    return out


def run_humaneval_items(model, tokenizer, n: int) -> dict[str, int]:
    from datasets import load_dataset
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    ds = load_dataset("openai/openai_humaneval", split="test")
    idx = np.random.default_rng(1234).permutation(len(ds))[:n]
    items: dict[str, int] = {}
    t0 = time.perf_counter()
    for j, i in enumerate(idx):
        r = ds[int(i)]
        prompt = ("Complete this Python function. Reply with ONLY the complete "
                  "function inside a ```python code block.\n\n```python\n"
                  + r["prompt"] + "\n```")
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                enable_thinking=False, add_generation_prompt=True, tokenize=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        out = generate(model, tokenizer, text, max_tokens=512,
                       sampler=make_sampler(temp=0.0))
        m = re.findall(r"```(?:python)?\n(.*?)```", out, re.DOTALL)
        code = m[0] if m else out
        if r["entry_point"] not in code:
            code = r["prompt"] + code
        program = code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\nprint('PASS')\n"
        ok = 0
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
        try:
            res = subprocess.run(["python3", f.name], capture_output=True,
                                 text=True, timeout=15)
            ok = int("PASS" in res.stdout)
        except Exception:
            pass
        items[r["task_id"]] = ok
        if (j + 1) % 10 == 0:
            print(f"  humaneval {j + 1}/{n}: {sum(items.values())} ✓")
    print(f"[humaneval] {sum(items.values())}/{n} ({time.perf_counter() - t0:.0f}s)")
    return items


def run_mmlu_items(model, tokenizer, n: int) -> dict[str, int]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    def ask(prompt: str, max_tokens: int) -> str:
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, **kwargs)
        return generate(model, tokenizer, text, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0))

    items: dict[str, int] = {}
    t0 = time.perf_counter()
    for i, (q, choices, ref) in enumerate(load_mmlu(n)):
        lines = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", choices))
        out = ask(f"{q}\n\n{lines}\n\nPiensa brevemente si hace falta y termina tu "
                  f"respuesta SOLO con la letra correcta (A, B, C o D).", 256)
        ms = re.findall(r"\b([ABCD])\b", out)
        # Clave por CONTENIDO (no posicional): con n distintos entre runs, la
        # posición i sería otra pregunta y McNemar emparejaría pares corruptos.
        items["mmlu_" + sha1(q.encode()).hexdigest()[:16]] = int(bool(ms) and ms[-1] == ref)
        if (i + 1) % 25 == 0:
            print(f"  mmlu {i + 1}/{n}: {sum(items.values())} ✓")
    print(f"[mmlu] {sum(items.values())}/{n} ({time.perf_counter() - t0:.0f}s)")
    return items


def mcnemar_exact(a: dict[str, int], b: dict[str, int]) -> dict:
    """Exact McNemar (two-sided binomial over discordant pairs)."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)  # A wins
    b10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)  # B wins
    n_d = b01 + b10
    if n_d == 0:
        p = 1.0
    else:
        big = max(b01, b10)
        p = min(1.0, 2 * sum(comb(n_d, k) for k in range(big, n_d + 1)) / 2**n_d)
    return {"n": len(keys), "acc_a": sum(a[k] for k in keys) / len(keys),
            "acc_b": sum(b[k] for k in keys) / len(keys),
            "discordant_a_wins": b01, "discordant_b_wins": b10, "p_exact": p}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired eval with exact McNemar")
    sub = parser.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--model", required=True)
    pr.add_argument("--tag", required=True)
    pr.add_argument("--humaneval", type=int, default=50)
    pr.add_argument("--mmlu", type=int, default=100)
    pt = sub.add_parser("test")
    pt.add_argument("file_a")
    pt.add_argument("file_b")
    args = parser.parse_args()

    if args.mode == "test":
        A, B = json.load(open(args.file_a)), json.load(open(args.file_b))
        for bench in sorted(set(A) & set(B)):
            if not isinstance(A[bench], dict):
                continue
            r = mcnemar_exact(A[bench], B[bench])
            print(f"[{bench}] n={r['n']}  A {r['acc_a']:.1%} vs B {r['acc_b']:.1%}  "
                  f"discordantes {r['discordant_a_wins']}/{r['discordant_b_wins']}  "
                  f"p={r['p_exact']:.4f}")
        return

    set_seeds(1234)
    from per_layer import maybe_patch
    from mlx_lm import load

    maybe_patch(args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    res = {
        "tag": args.tag, "model": args.model,
        "n_humaneval": args.humaneval, "n_mmlu": args.mmlu,
        "humaneval": run_humaneval_items(model, tokenizer, args.humaneval),
        "mmlu": run_mmlu_items(model, tokenizer, args.mmlu),
    }
    out = REPO_ROOT / "runs" / f"paired_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
