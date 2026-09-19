"""Marginal cost of each stage, measured inside a realistic dependent chain.

Isolated GEMM benchmarks mislead: independent matmuls pipeline, a 36-layer
decode does not. Each variant below reproduces the real dependency structure
and drops one stage, so the delta is that stage's true marginal cost.
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))


@torch.inference_mode()
def gtime(fn, iters=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=544)
    a = ap.parse_args()
    from engine import Engine
    import ek_kernels as K
    dev = torch.device("cuda")
    eng = Engine(a.model); m = eng.model; c = m.cfg; b = a.batch; L = m.layers; NL = c.num_layers
    nm = torch.cuda.get_device_name(0)
    pk = 3.35 if "H100" in nm else (4.0 if "GH200" in nm else 1.555)

    x0 = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    res0 = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    kc = [torch.randn(b, c.num_kv_heads, a.seq, c.head_dim, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    vc = [torch.randn_like(t) for t in kc]
    len_t = torch.tensor([a.seq], dtype=torch.int64, device=dev)
    st = torch.zeros(b, dtype=torch.int32, device=dev)
    slot = torch.zeros(1, dtype=torch.int64, device=dev)
    cs = torch.randn(b, c.head_dim, device=dev, dtype=torch.bfloat16)
    ws = eng._make_ws(b, a.seq)
    qs, kvs = c.q_size, c.kv_size

    def chain(norms=True, rope=True, attn=True, silu=True, kvwrite=True, head=True, gemv=False):
        MM = (lambda t, l, key: K.gemv(t, l[key])) if gemv else (lambda t, l, key: torch.matmul(t, l[key + "_t"]))
        x, res = x0, res0.clone()
        for i in range(NL):
            l = L[i]
            x = K.add_rms_norm(x, res, l["ln1"], 1e-6)[0] if norms else x
            qkv = MM(x, l, "qkv")
            if rope or kvwrite:
                K.qk_norm_rope_kv(qkv, l["qn"], l["kn"], cs, cs, kc[i], vc[i], slot,
                                  c.num_heads, c.num_kv_heads, 1e-6, 1)
            if attn:
                o = K.flash_decode(qkv[:, :qs].view(b, c.num_heads, c.head_dim),
                                   kc[i], vc[i], len_t, st, ws, c.head_dim ** -0.5).view(b, qs)
            else:
                o = qkv[:, :qs]
            x = MM(o, l, "o")
            x = K.add_rms_norm(x, res, l["ln2"], 1e-6)[0] if norms else x
            gu = MM(x, l, "gu")
            act = K.silu_mul(gu) if silu else gu[:, :c.intermediate_size]
            x = MM(act, l, "down")
        if head:
            hx = K.gemv(x, m.lm_head) if gemv else torch.matmul(x, m.lm_head_t)
            x = torch.argmax(hx, -1)
        return x

    wb = sum(t.numel() * 2 for l in L for t in (l["qkv"], l["o"], l["gu"], l["down"])) + m.lm_head.numel() * 2
    variants = [
        ("full chain",            dict()),
        ("  -norms",              dict(norms=False)),
        ("  -attention",          dict(attn=False)),
        ("  -silu",               dict(silu=False)),
        ("  -rope+kvwrite",       dict(rope=False, kvwrite=False)),
        ("  -lm_head",            dict(head=False)),
        ("GEMMs only",            dict(norms=False, rope=False, attn=False, silu=False, kvwrite=False)),
        ("full chain, triton gemv", dict(gemv=True)),
        ("GEMMs only, triton gemv", dict(gemv=True, norms=False, rope=False, attn=False, silu=False, kvwrite=False)),
    ]
    print(f"{nm}  batch={b} seq={a.seq} splits={ws[4]}  weights={wb/1e9:.2f} GB  peak {pk} TB/s\n")
    base = None
    for label, kw in variants:
        dt = gtime(lambda: chain(**kw))
        if base is None:
            base = dt
            print(f"  {label:20s} {dt*1e6:8.1f} us   {wb/dt/1e12:5.2f} TB/s ({100*wb/dt/1e12/pk:3.0f}% of peak)")
        else:
            print(f"  {label:20s} {dt*1e6:8.1f} us   saves {(base-dt)*1e6:7.1f} us "
                  f"({100*(base-dt)/base:4.1f}%)")

    prompts = [list(range(1000, 1000 + a.seq - 32))] * b
    it = eng.generate(prompts, 24); next(it)
    ts = [time.perf_counter()]
    for _ in it:
        ts.append(time.perf_counter())
    d = sorted((ts[i]-ts[i-1])*1e6 for i in range(1, len(ts)))
    print(f"\n  {'real engine step':20s} {d[len(d)//2]:8.1f} us")


if __name__ == "__main__":
    main()
