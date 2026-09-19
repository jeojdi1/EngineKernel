"""Worst-case acceptance margin over many positions, not just a pass/fail.

For every generated position: native_max_logit - native_logit[our token], from
a teacher-forced replay of our own output through transformers. 0 means we
picked native's argmax; the judge fails anything above 2.0. Native replayed
against itself reaches ~0.75, so that is the noise floor to compare with.
"""
import argparse, glob, os, random, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

def corpus(tok, n_files=300):
    files = [f for pat in ("/usr/share/doc/**/*.txt", "/usr/share/doc/**/README*", "/usr/share/common-licenses/*",
                           "/usr/lib/python3*/[a-z]*.py", "/usr/share/doc/**/changelog*")
             for f in glob.glob(pat, recursive=True) if os.path.isfile(f) and os.path.getsize(f) > 8000]
    random.Random(5).shuffle(files); ids = []
    for f in files[:n_files]:
        raw = open(f, "rb").read()[:80000]
        if raw[:2] != b"\x1f\x8b": ids.extend(tok(raw.decode("utf-8", "ignore"))["input_ids"])
    return ids

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from engine import Engine
    tok = AutoTokenizer.from_pretrained(a.model); ids = corpus(tok); rng = random.Random(a.seed)
    shapes = [(1, 512, 64), (4, 2048, 64), (16, 512, 192), (32, 768, 256), (8, 1024, 256)]
    eng = Engine(a.model); runs = []
    for b, s, n in shapes:
        for rep in range(3):
            p = [ids[o:o + s] for o in (rng.randrange(0, len(ids) - s - 1) for _ in range(b))]
            out = list(eng.generate(p, n)); runs.append((b, s, n, p, out))
    del eng; torch.cuda.empty_cache()
    ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    worst = {}; allm = []
    for b, s, n, p, out in runs:
        for r0 in range(0, b, 4):
            rows = [p[r] + [out[t][r] for t in range(n)] for r in range(r0, min(b, r0 + 4))]
            lg = ref(input_ids=torch.tensor(rows, device="cuda"), use_cache=False).logits[:, s - 1:s - 1 + n].float()
            ours = torch.tensor([[out[t][r] for t in range(n)] for r in range(r0, min(b, r0 + 4))], device="cuda")
            m = (lg.max(-1).values - lg.gather(-1, ours[..., None])[..., 0]).flatten()
            allm.append(m.cpu()); worst[(b, s, n)] = max(worst.get((b, s, n), 0.0), float(m.max()))
            del lg
    m = torch.cat(allm)
    print(f"positions {m.numel()} | argmax differs {(m > 0).sum().item()} | >0.5: {(m > 0.5).sum().item()} | >1.0: {(m > 1.0).sum().item()} | >2.0 (FAIL): {(m > 2.0).sum().item()} | MAX {m.max().item():.3f}")
    for k, v in worst.items(): print(f"   b={k[0]:2d} {k[1]}->{k[2]}: worst margin {v:.3f}")

if __name__ == "__main__":
    main()
