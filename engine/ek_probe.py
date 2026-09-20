"""Child-process checks: can Triton compile and launch our kernels here?

Kept out of the engine process on purpose. A Triton failure is not always an
exception: on Hopper, Triton 3.1.0 aborts the compiler for a 64-row tl.dot, and
a sandbox without a C compiler or a loadable libcuda fails before that. A
child's exit code turns all of those into a plain "no".

    ek_probe.py basic
    ek_probe.py shape <batch> <nq> <bucket> <n_kv> <n_heads> <head_dim>
"""
import sys

import torch
import triton
import triton.language as tl


# module level on purpose: a jitted kernel resolves `tl` as a module global
@triton.jit
def _bump(X, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(X + offs, tl.load(X + offs) + 1.0)


def basic() -> int:
    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    _bump[(1,)](x, N=64)
    torch.cuda.synchronize()
    return 0 if float(x.sum().item()) == 64.0 else 3


def shape(b, nq, bucket, n_kv, n_heads, d) -> int:
    """Compile every kernel the verify step uses, with this shape's constexprs."""
    import ek_kernels as K

    dev, dt = "cuda", torch.bfloat16
    hidden, inter = 2560, 9728
    m = b * max(nq, 1)
    x = torch.randn(m, hidden, device=dev, dtype=dt)
    w = torch.randn(hidden, device=dev, dtype=dt)
    K.rms_norm(x, w, 1e-6)
    K.add_rms_norm(x, x.clone(), w, 1e-6)
    K.silu_mul(torch.randn(m, 2 * inter, device=dev, dtype=dt))
    K.heads_to_rows(torch.randn(b, n_heads, 130, d, device=dev, dtype=dt))   # prefill attention output copy

    if m <= 32:  # the decode projections may run on the Triton GEMV at this row count
        for n_out, k_in in ((n_heads * d + 2 * n_kv * d, hidden), (hidden, n_heads * d),
                            (2 * inter, hidden), (hidden, inter)):
            K.gemv(torch.randn(m, k_in, device=dev, dtype=dt), torch.randn(n_out, k_in, device=dev, dtype=dt))
        if m <= 32:  # split-K projections and the add+norm that sums their partials
            for n_out, k_in in ((hidden, n_heads * d), (hidden, inter)):
                parts = K.gemv_parts(torch.randn(m, k_in, device=dev, dtype=dt),
                                     torch.randn(n_out, k_in, device=dev, dtype=dt))
                K.add_rms_norm_parts(parts, torch.randn(m, hidden, device=dev, dtype=dt), w, 1e-6)

    kc = torch.zeros(b, n_kv, bucket, d, device=dev, dtype=dt)
    vc = torch.zeros_like(kc)
    qkv = torch.randn(m, (n_heads + 2 * n_kv) * d, device=dev, dtype=dt)
    hn = torch.ones(d, device=dev, dtype=dt)
    cs = torch.randn(m, d, device=dev, dtype=dt)
    len_b = torch.full((b,), max(1, bucket // 2), dtype=torch.int64, device=dev)
    for rows, mpb in ((m, max(nq, 1)), (b * 64, 64)):  # decode-sized and prefill-sized tilings
        q2 = qkv if rows == m else torch.randn(rows, qkv.shape[1], device=dev, dtype=dt)
        c2 = cs if rows == m else torch.randn(rows, d, device=dev, dtype=dt)
        K.qk_norm_rope_kv(q2, hn, hn, c2, c2, kc, vc,
                          len_b if rows == m else torch.zeros(b, dtype=torch.int64, device=dev),
                          n_heads, n_kv, 1e-6, mpb)

    splits, chunk, bn = K.plan_splits(b, n_kv, bucket)
    sp = K._next_pow2(splits)
    if nq == 0:  # plain decode: one query per sequence, shared length
        gp = K.group_pad(n_heads, n_kv)
        ws = (torch.zeros(b, n_kv, sp, gp, d, dtype=torch.float32, device=dev),
              torch.zeros(b, n_kv, sp, gp, dtype=torch.float32, device=dev),
              torch.full((b, n_kv, sp, gp), -1e30, dtype=torch.float32, device=dev),
              torch.empty(b, n_heads, d, device=dev, dtype=dt), splits, chunk, bn)
        out = K.flash_decode(torch.randn(b, n_heads, d, device=dev, dtype=dt), kc, vc,
                             torch.tensor([max(1, bucket // 2)], dtype=torch.int64, device=dev),
                             torch.zeros(b, dtype=torch.int32, device=dev), ws, d ** -0.5)
        # the fused norm+rope+cache-write+attention kernel the plain decode path uses
        out2 = K.rope_attn_decode(qkv[:b].contiguous(), hn, hn, cs[:b].contiguous(), cs[:b].contiguous(),
                                  kc, vc, torch.tensor([max(2, bucket // 2)], dtype=torch.int64, device=dev),
                                  torch.zeros(b, dtype=torch.int32, device=dev), ws, d ** -0.5, n_heads, 1e-6)
        torch.cuda.synchronize()
        return 0 if torch.isfinite(out.float()).all() and torch.isfinite(out2.float()).all() else 4
    # integer kernels of the speculative step: n-gram drafting and accept bookkeeping
    hist = torch.randint(0, 1000, (b, bucket), device=dev, dtype=torch.int64)
    hlen = torch.full((b,), max(4, bucket // 2), dtype=torch.int64, device=dev)
    toks = torch.zeros((b, nq), dtype=torch.int64, device=dev)
    K.ngram_draft(hist, hlen, toks)
    K.spec_accept(toks, toks.clone(), hist, hlen, hlen.clone(), hlen.clone(),
                  torch.full((b,), 8, dtype=torch.int64, device=dev),
                  torch.zeros((4, b, nq), dtype=torch.int32, device=dev),
                  torch.zeros((4, b), dtype=torch.int32, device=dev),
                  torch.zeros(1, dtype=torch.int64, device=dev))
    gp = max(16, K._next_pow2(nq * (n_heads // n_kv)))
    ws = (torch.zeros(b, n_kv, sp, gp, d, dtype=torch.float32, device=dev),
          torch.zeros(b, n_kv, sp, gp, dtype=torch.float32, device=dev),
          torch.full((b, n_kv, sp, gp), -1e30, dtype=torch.float32, device=dev),
          torch.empty(m, n_heads, d, device=dev, dtype=dt), splits, chunk, bn)
    q = torch.randn(m, n_heads, d, device=dev, dtype=dt)
    out = K.flash_verify(q, kc, vc, len_b, torch.zeros(b, dtype=torch.int32, device=dev),
                         ws, d ** -0.5, nq)
    torch.cuda.synchronize()
    return 0 if torch.isfinite(out.float()).all() else 4


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    sys.exit(basic() if mode == "basic" else shape(*[int(a) for a in sys.argv[2:8]]))
