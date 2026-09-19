"""The platform's acceptance rule, reproduced locally.

Rank is decided by throughput, but eligibility is decided by this: replay our
generated tokens through the baseline and require each one to be the baseline's
greedy choice at that position, or within 2 logits of it. Naive token equality
is stricter than the real rule and fails on near-ties.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

TEXT = (
    "The history of computing hardware spans the development of machines able to "
    "perform arithmetic automatically. Early devices were mechanical; later ones "
    "used vacuum tubes, then transistors, then integrated circuits. Each step "
    "reduced cost and increased speed, which changed what problems were worth "
    "solving. Today the limiting factor for large language model inference is "
    "usually memory bandwidth rather than arithmetic throughput, because every "
    "generated token requires reading the entire set of model weights. "
)


def make_prompts(model_path, batch, in_len, mode):
    if mode == "random":
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, 151000, (batch, in_len), generator=g).tolist()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok(TEXT * 200, return_tensors=None)["input_ids"]
    out = []
    for i in range(batch):
        s = (i * 37) % max(1, len(ids) - in_len - 1)
        out.append(ids[s:s + in_len])
    return out


@torch.inference_mode()
def verify(model, prompts, gen, device, chunk=2):
    """Teacher-force prompt+generated through the baseline; return per-step gaps."""
    steps = len(gen)
    b = len(prompts)
    results = []
    for lo in range(0, b, chunk):
        hi = min(lo + chunk, b)
        rows = []
        for i in range(lo, hi):
            rows.append(list(prompts[i]) + [gen[t][i] for t in range(steps)])
        n = len(rows[0])
        ids = torch.as_tensor(rows, dtype=torch.long, device=device)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits.float()
        plen = len(prompts[lo])
        for i in range(hi - lo):
            for t in range(steps):
                lg = logits[i, plen + t - 1]
                ours = gen[t][lo + i]
                best = int(lg.argmax())
                results.append((lo + i, t, ours, best, float(lg.max() - lg[ours])))
        del logits, out
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--in-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=32)
    ap.add_argument("--mode", choices=["random", "text"], default="text")
    args = ap.parse_args()

    dev = "cuda"
    prompts = make_prompts(args.model, args.batch, args.in_len, args.mode)

    from engine import Engine
    eng = Engine(args.model)
    gen = list(eng.generate(prompts, args.out_len))
    del eng
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    res = verify(model, prompts, gen, dev)
    diff = [r for r in res if r[2] != r[3]]
    over = [r for r in diff if r[4] > 2.0]
    print(f"mode={args.mode} batch={args.batch} in={args.in_len} out={args.out_len}")
    print(f"  positions checked : {len(res)}")
    print(f"  argmax differs    : {len(diff)}  ({100 * len(diff) / len(res):.1f}%)")
    print(f"  gap > 2 logits    : {len(over)}   <-- these are the only real failures")
    if diff:
        gaps = sorted(r[4] for r in diff)
        print(f"  gap distribution  : min={gaps[0]:.4f} median={gaps[len(gaps) // 2]:.4f} max={gaps[-1]:.4f}")
        print("  first few differing positions (row, step, ours, baseline, gap):")
        for r in diff[:8]:
            print(f"    {r}")
    print("VERDICT:", "PASS" if not over else "FAIL")


if __name__ == "__main__":
    main()
