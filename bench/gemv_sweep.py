"""Is cuBLAS leaving bandwidth on the table at decode batch sizes?

At M=1 every projection is a pure reduction over K for each of N outputs, so
parallelism is N/BLOCK_N. o_proj (N=2560) cannot fill the GPU, which is why it
runs at 58% of peak while lm_head (N=151936) reaches 91%. This sweeps cuBLAS
against a Triton GEMV over block sizes and split-K.
"""
from __future__ import annotations
import argparse, time
import torch, torch.nn.functional as F
import triton, triton.language as tl


@triton.jit
def _gemv(X, W, Y, K, M, stride_xm, stride_wn, stride_ym,
          N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          MP: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, MP)
    nmask = offs_n < N
    acc = tl.zeros([MP, BLOCK_N], tl.float32)
    k_per = K // SPLIT_K
    for k0 in range(pid_k * k_per, (pid_k + 1) * k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :])
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :], mask=nmask[:, None], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    o = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    smask = (offs_m[:, None] < M) & nmask[None, :]
    if SPLIT_K == 1:
        tl.store(o, acc.to(tl.bfloat16), mask=smask)
    else:
        tl.atomic_add(o, acc, mask=smask)


def gemv(x, w, block_n, block_k, split_k, out=None):
    m, k = x.shape
    n = w.shape[0]
    mp = max(16, triton.next_power_of_2(m))
    dt = torch.bfloat16 if split_k == 1 else torch.float32
    y = torch.zeros(m, n, device=x.device, dtype=dt) if out is None else out
    if split_k > 1:
        y.zero_()
    xp = x if m == mp else F.pad(x, (0, 0, 0, mp - m))
    _gemv[(triton.cdiv(n, block_n), split_k)](
        xp, w, y, k, m, xp.stride(0), w.stride(0), y.stride(0),
        N=n, BLOCK_N=block_n, BLOCK_K=block_k, MP=mp, SPLIT_K=split_k,
        num_warps=4, num_stages=3)
    return y


L2_BYTES = 40 << 20


def gbench(fn, iters=20):
    """Time under graph replay: no launch overhead, and the caller cycles
    distinct weights so DRAM traffic is real rather than L2 hits."""
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
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
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    pk = 3.35 if "H100" in name else (4.0 if "GH200" in name else 1.555)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{name}  {sms} SMs  peak {pk} TB/s  M={a.batch}")
    print("(distinct weight copies per measurement to defeat L2)\n")

    shapes = [("qkv", 6144, 2560), ("o", 2560, 4096),
              ("gate_up", 19456, 2560), ("down", 2560, 9728),
              ("lm_head", 151936, 2560)]
    for nm, n, k in shapes:
        wb = n * k * 2
        copies = max(2, min(36, -(-(8 * L2_BYTES) // wb)))
        x = torch.randn(a.batch, k, device=dev, dtype=torch.bfloat16)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        by = wb * copies

        cands = {"F.linear": lambda: [F.linear(x, w) for w in ws],
                 "matmul(x,w.t)": lambda: [torch.matmul(x, w.t()) for w in ws]}
        if a.batch == 1:
            x1 = x[0]
            cands["mv"] = lambda: [torch.mv(w, x1) for w in ws]
        best = (float("inf"), "")
        for cn, cf in cands.items():
            ct = gbench(cf)
            print(f"{nm:8s} N={n:6d} K={k:5d} x{copies:2d}  {cn:14s} {ct/copies*1e6:7.1f}us "
                  f"{by/ct/1e12:5.2f} TB/s ({100*by/ct/1e12/pk:3.0f}%)")
            if ct < best[0]:
                best = (ct, cn)
        t, cublas_tag = best
        ref = F.linear(x, ws[0]).float()
        errs = []
        for bn in (16, 32, 64, 128, 256):
            for bk in (64, 128, 256):
                for sk in (1, 2, 4, 8):
                    if (k // sk) % bk or n % bn or k % (bk * sk):
                        continue
                    try:
                        got = gemv(x, ws[0], bn, bk, sk)
                        rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
                        if rel > 3e-2:
                            errs.append(f"BN{bn}/BK{bk}/SK{sk} rel={rel:.3f}")
                            continue
                        outs = [torch.zeros(a.batch, n, device=dev,
                                            dtype=torch.bfloat16 if sk == 1 else torch.float32)
                                for _ in ws]
                        t2 = gbench(lambda: [gemv(x, w, bn, bk, sk, out=o) for w, o in zip(ws, outs)])
                    except Exception as e:
                        errs.append(f"BN{bn}/BK{bk}/SK{sk} {type(e).__name__}: {str(e)[:60]}")
                        continue
                    if t2 < best[0]:
                        best = (t2, f"triton BN={bn} BK={bk} SK={sk}")
        t2, tag = best
        gain = (t / t2 - 1) * 100
        flag = f"  <-- {gain:+.0f}% vs {cublas_tag}" if tag.startswith("triton") else "  (cublas wins)"
        print(f"{'':8s} {'':17s}  best   {t2/copies*1e6:7.1f}us {by/t2/1e12:5.2f} TB/s "
              f"({100*by/t2/1e12/pk:3.0f}%)  {tag}{flag}")
        if a.verbose and errs:
            print(f"         rejected: {errs[:4]}")


if __name__ == "__main__":
    main()
