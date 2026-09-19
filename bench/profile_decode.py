"""Where does a decode step actually go?

Compares the full captured step against a GEMM-only step built from the same
weights. The difference is what fusing norms/RoPE/SwiGLU into GEMM epilogues
could recover. Also reports achieved bandwidth per projection.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def graph_time(fn, iters=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=544)
    args = ap.parse_args()

    from engine import Engine
    import ek_kernels as K

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    eng = Engine(args.model)
    m = eng.model
    c = m.cfg
    b = args.batch

    wbytes = sum(t.numel() * t.element_size()
                 for L in m.layers for t in (L["qkv"], L["o"], L["gu"], L["down"]))
    wbytes += m.lm_head.numel() * m.lm_head.element_size()
    print(f"gpu: {props.name}  weights streamed per step: {wbytes / 1e9:.2f} GB")

    # --- per-projection achieved bandwidth -------------------------------
    print(f"\nprojection GEMMs at batch={b}:")
    x = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    o_in = torch.randn(b, c.q_size, device=dev, dtype=torch.bfloat16)
    act = torch.randn(b, c.intermediate_size, device=dev, dtype=torch.bfloat16)
    L0 = m.layers[0]
    total_gemm = 0.0
    for name, inp, w in (("qkv", x, L0["qkv"]), ("o", o_in, L0["o"]),
                         ("gate_up", x, L0["gu"]), ("down", act, L0["down"])):
        dt = timeit(lambda: F.linear(inp, w), iters=200)
        by = w.numel() * w.element_size()
        total_gemm += dt * c.num_layers
        print(f"  {name:8s} {str(tuple(w.shape)):18s} {dt * 1e6:7.1f} us  "
              f"{by / dt / 1e12:5.2f} TB/s")
    dt = timeit(lambda: F.linear(x, m.lm_head), iters=200)
    total_gemm += dt
    print(f"  {'lm_head':8s} {str(tuple(m.lm_head.shape)):18s} {dt * 1e6:7.1f} us  "
          f"{m.lm_head.numel() * 2 / dt / 1e12:5.2f} TB/s")

    # --- elementwise / attention kernels ---------------------------------
    print("\nnon-GEMM kernels (per call, batch={}):".format(b))
    w2560 = torch.randn(c.hidden_size, device=dev, dtype=torch.bfloat16)
    res = torch.randn(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
    gu = torch.randn(b, 2 * c.intermediate_size, device=dev, dtype=torch.bfloat16)
    qkv = torch.randn(b, (c.num_heads + 2 * c.num_kv_heads) * c.head_dim,
                      device=dev, dtype=torch.bfloat16)
    cs = torch.randn(b, c.head_dim, device=dev, dtype=torch.bfloat16)
    parts = {
        "add_rms_norm": lambda: K.add_rms_norm(x, res, w2560, 1e-6),
        "silu_mul": lambda: K.silu_mul(gu),
        "qk_norm_rope": lambda: K.qk_norm_rope_(qkv, L0["qn"], L0["kn"], cs, cs,
                                                c.num_heads, c.num_kv_heads, 1e-6),
    }
    per_layer = 0.0
    for name, fn in parts.items():
        dt = timeit(fn, iters=300)
        n = 2 if name == "add_rms_norm" else 1
        per_layer += dt * n
        print(f"  {name:14s} {dt * 1e6:6.1f} us  x{n}/layer")

    kc = torch.randn(b, c.num_kv_heads, args.seq, c.head_dim, device=dev, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(b, c.num_heads, c.head_dim, device=dev, dtype=torch.bfloat16)
    len_t = torch.tensor([args.seq], dtype=torch.int64, device=dev)
    st = torch.zeros(b, dtype=torch.int32, device=dev)
    ws = eng._make_ws(b, args.seq)
    dt = timeit(lambda: K.flash_decode(q, kc, vc, len_t, st, ws, c.head_dim ** -0.5), iters=300)
    per_layer += dt
    print(f"  {'flash_decode':14s} {dt * 1e6:6.1f} us  x1/layer  (seq={args.seq}, splits={ws[4]})")

    # --- whole step -------------------------------------------------------
    prompts = [list(range(args.seq - 32))] * b
    it = eng.generate(prompts, 8)
    next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in it:
        n += 1
    step = (time.perf_counter() - t0) / n

    gemm_floor = total_gemm
    elem = per_layer * c.num_layers
    print(f"\n{'measured decode step':28s} {step * 1e6:8.1f} us")
    print(f"{'  GEMM floor (36 layers)':28s} {gemm_floor * 1e6:8.1f} us  "
          f"({100 * gemm_floor / step:.0f}%)")
    print(f"{'  non-GEMM kernels':28s} {elem * 1e6:8.1f} us  ({100 * elem / step:.0f}%)")
    print(f"{'  unaccounted':28s} {(step - gemm_floor - elem) * 1e6:8.1f} us  "
          f"({100 * (step - gemm_floor - elem) / step:.0f}%)")
    bw = wbytes / step / 1e12
    print(f"\nachieved {bw:.2f} TB/s of {props.name} peak; "
          f"weight-roofline step = {wbytes / (props.memory_clock_rate * 0):.0f}" if False else
          f"\nachieved {bw:.2f} TB/s streaming weights")


if __name__ == "__main__":
    main()
