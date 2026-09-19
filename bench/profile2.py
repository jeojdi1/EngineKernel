"""True per-component decode cost, measured under CUDA graph replay.

Eager timing is launch-bound and useless here. Each component is captured in a
graph that walks all 36 layers' real weights, so DRAM traffic matches a step.
"""
from __future__ import annotations
import argparse, os, sys, time
import torch, torch.nn.functional as F
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
    import kernels as K
    dev = torch.device("cuda")
    eng = Engine(a.model); m = eng.model; c = m.cfg; b = a.batch; L = m.layers
    peak = {"A100": 1.555, "H100": 3.35, "GH200": 4.0}
    name = torch.cuda.get_device_name(0)
    pk = next((v for k, v in peak.items() if k in name), 1.555)

    x = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    oi = torch.randn(b, c.q_size, device=dev, dtype=torch.bfloat16)
    ai = torch.randn(b, c.intermediate_size, device=dev, dtype=torch.bfloat16)
    res = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    gu = torch.randn(b, 2 * c.intermediate_size, device=dev, dtype=torch.bfloat16)
    qkv = torch.randn(b, (c.num_heads + 2 * c.num_kv_heads) * c.head_dim, device=dev, dtype=torch.bfloat16)
    cs = torch.randn(b, c.head_dim, device=dev, dtype=torch.bfloat16)
    kc = torch.randn(b, c.num_kv_heads, a.seq, c.head_dim, device=dev, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(b, c.num_heads, c.head_dim, device=dev, dtype=torch.bfloat16)
    len_t = torch.tensor([a.seq], dtype=torch.int64, device=dev)
    st = torch.zeros(b, dtype=torch.int32, device=dev)
    slot = torch.zeros(1, dtype=torch.int64, device=dev)
    ws = eng._make_ws(b, a.seq)
    NL = c.num_layers

    def all_layers(f):
        return lambda: [f(L[i]) for i in range(NL)]

    comps = [
        ("qkv GEMM",      all_layers(lambda l: F.linear(x, l["qkv"])),  L[0]["qkv"].numel() * 2 * NL),
        ("o GEMM",        all_layers(lambda l: F.linear(oi, l["o"])),   L[0]["o"].numel() * 2 * NL),
        ("gate_up GEMM",  all_layers(lambda l: F.linear(x, l["gu"])),   L[0]["gu"].numel() * 2 * NL),
        ("down GEMM",     all_layers(lambda l: F.linear(ai, l["down"])), L[0]["down"].numel() * 2 * NL),
        ("lm_head",       lambda: F.linear(x, m.lm_head),               m.lm_head.numel() * 2),
        ("add_rms_norm x2", lambda: [K.add_rms_norm(x, res, L[i % NL]["ln1"], 1e-6) for i in range(2 * NL)], 0),
        ("silu_mul",      lambda: [K.silu_mul(gu) for _ in range(NL)],  0),
        ("qk_norm_rope",  all_layers(lambda l: K.qk_norm_rope_(qkv, l["qn"], l["kn"], cs, cs, c.num_heads, c.num_kv_heads, 1e-6)), 0),
        ("kv index_copy x2", lambda: [t.index_copy_(2, slot, q[:, :c.num_kv_heads].unsqueeze(2)) for _ in range(NL) for t in (kc, vc)], 0),
        ("flash_decode",  lambda: [K.flash_decode(q, kc, vc, len_t, st, ws, c.head_dim ** -0.5) for _ in range(NL)], 0),
        ("embed+argmax",  lambda: (F.embedding(slot.expand(b), m.embed), torch.argmax(F.linear(x, m.lm_head), -1)), 0),
    ]
    print(f"{name}  peak {pk} TB/s   batch={b} seq={a.seq} splits={ws[4]}\n")
    tot = 0.0
    for nm, fn, byts in comps:
        dt = gtime(fn)
        if nm == "embed+argmax":
            dt -= comps[4][1] and 0  # argmax includes an lm_head; noted below
        tot += dt
        bw = f"{byts / dt / 1e12:5.2f} TB/s ({100 * byts / dt / 1e12 / pk:3.0f}%)" if byts else ""
        print(f"  {nm:20s} {dt * 1e6:8.1f} us   {bw}")
    print(f"\n  {'SUM of components':20s} {tot * 1e6:8.1f} us  (embed+argmax double-counts one lm_head)")

    prompts = [list(range(1000, 1000 + a.seq - 32))] * b
    it = eng.generate(prompts, 24); next(it)
    ts = [time.perf_counter()]
    for _ in it:
        ts.append(time.perf_counter())
    d = sorted((ts[i] - ts[i - 1]) * 1e6 for i in range(1, len(ts)))
    med = d[len(d) // 2]
    wb = sum(t.numel() * 2 for l in L for t in (l["qkv"], l["o"], l["gu"], l["down"])) + m.lm_head.numel() * 2
    print(f"  {'MEASURED step':20s} {med:8.1f} us   {wb / (med * 1e-6) / 1e12:5.2f} TB/s "
          f"({100 * wb / (med * 1e-6) / 1e12 / pk:3.0f}% of peak)")


if __name__ == "__main__":
    main()
