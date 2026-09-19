"""Local mirror of the public workloads: throughput plus exactness vs baseline.

    python bench/bench.py --model /path/to/qwen3-4b [--check] [--baseline]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

WORKLOADS = [
    ("public-0", 1, 512, 32),
    ("public-1", 4, 2048, 32),
    ("public-2", 16, 512, 128),
]


def make_inputs(batch, in_len, vocab=151000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, in_len), generator=g).tolist()


def run(engine, prompts, n_new):
    t0 = time.perf_counter()
    steps, ttft = [], None
    for tok in engine.generate(prompts, n_new):
        if ttft is None:
            ttft = time.perf_counter() - t0
        steps.append(tok)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return steps, ttft, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="compare tokens against the baseline")
    ap.add_argument("--baseline", action="store_true", help="benchmark the baseline too")
    args = ap.parse_args()

    from engine import Engine

    t0 = time.perf_counter()
    eng = Engine(args.model)
    print(f"engine load + capture: {time.perf_counter() - t0:.1f}s")

    ref = None
    if args.check or args.baseline:
        from baseline import BaselineEngine
        ref = BaselineEngine(args.model)

    rates = []
    for name, b, in_len, out_len in WORKLOADS:
        prompts = make_inputs(b, in_len)
        run(eng, prompts, min(4, out_len))  # touch the shape once

        times, ttfts = [], []
        for _ in range(args.samples):
            steps, ttft, dt = run(eng, prompts, out_len)
            times.append(dt)
            ttfts.append(ttft)
        med = statistics.median(times)
        tps = b * out_len / med
        spread = (max(times) - min(times)) / med
        rates.append(tps)
        line = (f"{name}: {tps:8.1f} tok/s  median {med * 1000:7.1f}ms  "
                f"ttft {statistics.median(ttfts) * 1000:6.1f}ms  spread {spread * 100:4.1f}%")

        if ref is not None:
            bsteps, bttft, bdt = run(ref, prompts, out_len)
            if args.check:
                mism = [i for i, (a, c) in enumerate(zip(steps, bsteps)) if list(a) != list(c)]
                line += "  MATCH" if not mism else f"  MISMATCH@{mism[:5]}"
            if args.baseline:
                line += f"  baseline {b * out_len / bdt:7.1f} tok/s ({bdt / med:.2f}x faster)"
        print(line)

    geo = 1.0
    for r in rates:
        geo *= r
    print(f"geometric mean: {geo ** (1 / len(rates)):.1f} tok/s")
    if torch.cuda.is_available():
        print(f"peak memory: {torch.cuda.max_memory_allocated() / (1 << 30):.1f} GiB")


if __name__ == "__main__":
    main()
