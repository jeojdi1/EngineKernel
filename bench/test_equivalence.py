"""Token-for-token equivalence against transformers, on a tiny random Qwen3.

Runs on CPU in seconds, so every kernel/layout change can be checked before it
costs a submission. Exercises padded batches, which the hidden workloads may
not, but a bug there would be silent otherwise.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))


def build_tiny(path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=1024, rope_theta=1000000.0,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(path, safe_serialization=True)
    return cfg


def hf_greedy(path, prompts, n):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    lens = [len(p) for p in prompts]
    s = max(lens)
    b = len(prompts)
    ids = torch.zeros((b, s), dtype=torch.long)
    mask = torch.zeros((b, s), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, s - lens[i]:] = torch.as_tensor(p)
        mask[i, s - lens[i]:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    past, cur, cur_pos, cur_mask = None, ids, pos, mask
    out_toks = []
    with torch.inference_mode():
        for _ in range(n):
            o = m(input_ids=cur, attention_mask=cur_mask, position_ids=cur_pos,
                  past_key_values=past, use_cache=True)
            past = o.past_key_values
            t = o.logits[:, -1, :].argmax(-1)
            out_toks.append(t.tolist())
            cur = t.unsqueeze(1)
            cur_pos = cur_pos[:, -1:] + 1
            cur_mask = torch.cat([cur_mask, torch.ones((b, 1), dtype=torch.long)], dim=1)
    return out_toks


def main():
    tmp = tempfile.mkdtemp()
    build_tiny(tmp)
    torch.manual_seed(1)

    cases = [
        ("uniform batch 1", [torch.randint(0, 512, (37,)).tolist()], 12),
        ("uniform batch 4", [torch.randint(0, 512, (29,)).tolist() for _ in range(4)], 10),
        ("ragged batch 3", [torch.randint(0, 512, (n,)).tolist() for n in (31, 17, 24)], 8),
    ]

    from engine import Engine

    eng = Engine(tmp)
    failures = 0
    for name, prompts, n in cases:
        ref = hf_greedy(tmp, prompts, n)
        got = list(eng.generate(prompts, n))
        if len(got) != n:
            print(f"FAIL {name}: yielded {len(got)} steps, expected {n}")
            failures += 1
            continue
        bad = [i for i in range(n) if list(got[i]) != list(ref[i])]
        if bad:
            print(f"FAIL {name}: mismatch at steps {bad}")
            print(f"  ref {ref}")
            print(f"  got {got}")
            failures += 1
        else:
            print(f"ok   {name}: {n} steps x {len(prompts)} seqs match")
    # The engine degrades silently by design, so a broken fast path would still
    # "pass" here on the reference loop. Make that a test failure.
    tier = getattr(eng, "tier", "fast")
    if tier != "fast":
        print(f"FAIL engine fell back to the {tier!r} tier: the fast path raised")
        failures += 1
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
