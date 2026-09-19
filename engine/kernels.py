"""Triton kernels for the Qwen3 engine, each paired with a pure-torch fallback.

The fallbacks exist so the whole model can be exercised on CPU against
`transformers` without a GPU; they are never used on the benchmark path.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only dev boxes
    triton = None
    tl = None
    _HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _rms_norm_kernel(
        X, W, Y,
        stride_xm, stride_ym,
        N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
        # Match HF: normalise in fp32, round to bf16, *then* apply the weight.
        xn = (x * rstd).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(Y + row * stride_ym + cols, (xn * w).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        X, R, W, Y,
        stride_xm, stride_rm, stride_ym,
        N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
    ):
        """R += X (kept in bf16, as the reference does); Y = rmsnorm(R) * W."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
        r = tl.load(R + row * stride_rm + cols, mask=mask, other=0.0)
        r = (r + x).to(tl.bfloat16)
        tl.store(R + row * stride_rm + cols, r, mask=mask)
        rf = r.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(rf * rf, axis=0) / N + EPS)
        xn = (rf * rstd).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(Y + row * stride_ym + cols, (xn * w).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _silu_mul_kernel(
        X, Y,
        stride_xm, stride_ym,
        N: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Y[m, :N] = silu(X[m, :N]) * X[m, N:2N] for a fused gate/up GEMM."""
        row = tl.program_id(0)
        blk = tl.program_id(1)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        g = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(X + row * stride_xm + N + cols, mask=mask, other=0.0).to(tl.float32)
        y = (g / (1.0 + tl.exp(-g))) * u
        tl.store(Y + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _qk_norm_rope_kv_kernel(
        QKV, QN, KN, COS, SIN, KC, VC, SlotBase,
        stride_qkv_m, stride_cos_m,
        stride_cb, stride_ch, stride_cs,
        N_Q: tl.constexpr, N_KV: tl.constexpr, D: tl.constexpr,
        HALF: tl.constexpr, EPS: tl.constexpr, M_PER_BATCH: tl.constexpr,
    ):
        """Per-head QK-RMSNorm + rotary, writing k/v straight into the cache.

        Rows of QKV are [ q: N_Q*D | k: N_KV*D | v: N_KV*D ]. One program owns
        one (row, head), so the in-place rotate-half read/write stays local.
        Folding the cache write in here removes two index_copy_ launches per
        layer, which at decode batch sizes cost far more than the 4 KB they move.
        """
        r = tl.program_id(0)
        h = tl.program_id(1)
        is_q = h < N_Q
        is_v = h >= N_Q + N_KV

        cols = tl.arange(0, D)
        base = QKV + r * stride_qkv_m + h * D
        x = tl.load(base + cols).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
        w = tl.where(is_q, tl.load(QN + cols), tl.load(KN + cols))
        idx = tl.where(cols < HALF, cols + HALF, cols - HALF)
        xp = tl.load(base + idx).to(tl.float32)
        wp = tl.where(is_q, tl.load(QN + idx), tl.load(KN + idx))

        xn = (x * rstd).to(tl.bfloat16) * w
        xpn = (xp * rstd).to(tl.bfloat16) * wp
        rot = tl.where(cols < HALF, -xpn.to(tl.float32), xpn.to(tl.float32))
        cos = tl.load(COS + r * stride_cos_m + cols).to(tl.float32)
        sin = tl.load(SIN + r * stride_cos_m + cols).to(tl.float32)
        out = (xn.to(tl.float32) * cos + rot * sin).to(tl.bfloat16)

        if is_q:
            tl.store(base + cols, out)
        else:
            bidx = r // M_PER_BATCH
            slot = r % M_PER_BATCH + tl.load(SlotBase)
            kv_h = tl.where(is_v, h - N_Q - N_KV, h - N_Q)
            dst = bidx * stride_cb + kv_h * stride_ch + slot * stride_cs + cols
            if is_v:
                tl.store(VC + dst, x.to(tl.bfloat16))
            else:
                tl.store(KC + dst, out)

    @triton.jit
    def _gemv_kernel(
        X, W, Y, M, K,
        stride_xm, stride_wn, stride_ym,
        N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
    ):
        """y = x @ w.T for decode-shaped x. W is [N, K] row-major.

        At decode batch sizes this is a pure reduction over K per output, so
        parallelism is N/BLOCK_N; the M dimension is padded to 16 to satisfy
        tl.dot and the padded lanes cost nothing (the op is bandwidth-bound).
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, MP)
        nmask = offs_n < N
        mmask = offs_m < M
        acc = tl.zeros([MP, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < K
            x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :],
                        mask=mmask[:, None] & kmask[None, :], other=0.0)
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
        tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _gemv1_kernel(
        X, W, Y, K,
        stride_wn,
        N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """y[0, :] = x[0, :] @ w.T for a single row. Pure FMA reduction."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        acc = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < K
            x = tl.load(X + offs_k, mask=kmask, other=0.0).to(tl.float32)
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(Y + offs_n, acc.to(tl.bfloat16), mask=nmask)

    @triton.jit
    def _flash_decode_split_kernel(
        Q, K, V, SeqLen, Start,
        Acc, Lsum, Mmax, Out,
        sm_scale,
        stride_qb, stride_qh,
        stride_ob, stride_oh,
        stride_kb, stride_kh, stride_ks,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        N_KV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
        BLOCK_N: tl.constexpr, CHUNK: tl.constexpr, SPLITS_ONE: tl.constexpr,
    ):
        """One program per (batch, kv-head, sequence-split).

        GQA is handled by loading the whole group of G query heads as the M
        dimension of the dot, padded up to GP=16 so `tl.dot` is legal. Decode
        attention is bandwidth-bound, so the padded lanes cost nothing real.
        """
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // N_KV
        h = pid_bh % N_KV

        seq_len = tl.load(SeqLen)
        start = tl.load(Start + b)

        lo = pid_s * CHUNK
        hi = tl.minimum(lo + CHUNK, seq_len)
        lo = tl.maximum(lo, start)

        offs_d = tl.arange(0, D)
        offs_g = tl.arange(0, GP)
        gmask = offs_g < G

        q = tl.load(
            Q + b * stride_qb + (h * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
            mask=gmask[:, None], other=0.0,
        )

        m_i = tl.full([GP], -1e30, tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)

        kbase = K + b * stride_kb + h * stride_kh
        vbase = V + b * stride_kb + h * stride_kh

        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < hi
            k = tl.load(
                kbase + offs_n[:, None] * stride_ks + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(nmask[None, :], qk, -1e30)

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            v = tl.load(
                vbase + offs_n[:, None] * stride_ks + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        if SPLITS_ONE:
            # one split means no partials to merge: write the answer here and
            # skip the combine launch. Two launches per layer is most of the
            # cost of decode attention at small batch.
            tl.store(Out + b * stride_ob + (h * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
                     (acc / l_i[:, None]).to(tl.bfloat16), mask=gmask[:, None])
        else:
            aptr = Acc + b * stride_ab + h * stride_ah + pid_s * stride_as
            tl.store(aptr + offs_g[:, None] * stride_ag + offs_d[None, :], acc, mask=gmask[:, None])
            lptr = Lsum + b * stride_lb + h * stride_lh + pid_s * stride_ls
            mptr = Mmax + b * stride_lb + h * stride_lh + pid_s * stride_ls
            tl.store(lptr + offs_g, l_i, mask=gmask)
            tl.store(mptr + offs_g, m_i, mask=gmask)

    @triton.jit
    def _flash_decode_combine_kernel(
        Acc, Lsum, Mmax, Out,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        stride_ob, stride_oh,
        N_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        """Log-sum-exp merge of the per-split partials into one output head."""
        pid = tl.program_id(0)
        g = pid % G
        h = (pid // G) % N_KV
        b = pid // (G * N_KV)

        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, SPLITS)

        m_s = tl.load(Mmax + b * stride_lb + h * stride_lh + offs_s * stride_ls + g)
        l_s = tl.load(Lsum + b * stride_lb + h * stride_lh + offs_s * stride_ls + g)
        m = tl.max(m_s, axis=0)
        scale = tl.exp(m_s - m)
        # splits that covered no tokens contribute l == 0 and drop out here
        denom = tl.sum(l_s * scale, axis=0)

        a = tl.load(
            Acc + b * stride_ab + h * stride_ah + offs_s[:, None] * stride_as
            + g * stride_ag + offs_d[None, :]
        )
        num = tl.sum(a * scale[:, None], axis=0)
        out = num / denom
        tl.store(Out + b * stride_ob + (h * G + g) * stride_oh + offs_d, out.to(tl.bfloat16))


# --------------------------------------------------------------------------
# Dispatch wrappers
# --------------------------------------------------------------------------
def _torch_rms_norm(x, w, eps):
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xn.to(x.dtype)


def rms_norm(x, w, eps):
    if not (_HAS_TRITON and x.is_cuda):
        return _torch_rms_norm(x, w, eps)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    y = torch.empty_like(x2)
    n = shape[-1]
    _rms_norm_kernel[(x2.shape[0],)](
        x2, w, y, x2.stride(0), y.stride(0),
        N=n, BLOCK=_next_pow2(n), EPS=eps, num_warps=8,
    )
    return y.view(shape)


def add_rms_norm(x, residual, w, eps):
    """residual += x; returns (rmsnorm(residual) * w, residual)."""
    if not (_HAS_TRITON and x.is_cuda):
        residual = residual + x
        return _torch_rms_norm(residual, w, eps), residual
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    r2 = residual.reshape(-1, shape[-1])
    y = torch.empty_like(x2)
    n = shape[-1]
    _add_rms_norm_kernel[(x2.shape[0],)](
        x2, r2, w, y, x2.stride(0), r2.stride(0), y.stride(0),
        N=n, BLOCK=_next_pow2(n), EPS=eps, num_warps=8,
    )
    return y.view(shape), residual


def silu_mul(gu):
    """gu: [..., 2I] from a fused gate/up projection -> [..., I]."""
    if not (_HAS_TRITON and gu.is_cuda):
        g, u = gu.chunk(2, dim=-1)
        return torch.nn.functional.silu(g) * u
    shape = list(gu.shape)
    n = shape[-1] // 2
    x2 = gu.reshape(-1, shape[-1])
    shape[-1] = n
    y = torch.empty(x2.shape[0], n, dtype=gu.dtype, device=gu.device)
    BLOCK = 1024
    _silu_mul_kernel[(x2.shape[0], _cdiv(n, BLOCK))](
        x2, y, x2.stride(0), y.stride(0), N=n, BLOCK=BLOCK, num_warps=4,
    )
    return y.view(shape)


def qk_norm_rope_kv(qkv, qn, kn, cos, sin, k_cache, v_cache, slot_base,
                    n_q, n_kv, eps, m_per_batch):
    """QK-norm + rotary on the fused QKV, with k/v written into the caches.

    qkv: [M, (n_q + 2*n_kv)*D]; caches: [B, n_kv, S, D]; slot_base: device int64
    scalar added to each row's within-sequence position (0 for prefill).
    """
    d = qn.shape[0]
    m = qkv.shape[0]
    if not (_HAS_TRITON and qkv.is_cuda):
        b = m // m_per_batch
        q = qkv[:, : n_q * d].view(m, n_q, d)
        k = qkv[:, n_q * d: (n_q + n_kv) * d].view(m, n_kv, d)
        v = qkv[:, (n_q + n_kv) * d:].view(m, n_kv, d)
        half = d // 2
        for t, w in ((q, qn), (k, kn)):
            t.copy_(_torch_rms_norm(t, w, eps))
            rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            t.copy_(t * cos.unsqueeze(1) + rot * sin.unsqueeze(1))
        base = int(slot_base.item())
        kk = k.view(b, m_per_batch, n_kv, d).transpose(1, 2)
        vv = v.view(b, m_per_batch, n_kv, d).transpose(1, 2)
        k_cache[:, :, base:base + m_per_batch].copy_(kk)
        v_cache[:, :, base:base + m_per_batch].copy_(vv)
        return qkv
    _qk_norm_rope_kv_kernel[(m, n_q + 2 * n_kv)](
        qkv, qn, kn, cos, sin, k_cache, v_cache, slot_base,
        qkv.stride(0), cos.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        N_Q=n_q, N_KV=n_kv, D=d, HALF=d // 2, EPS=eps,
        M_PER_BATCH=m_per_batch, num_warps=4,
    )
    return qkv


# A tile pair (x and w) is staged num_stages deep in shared memory, so
# BLOCK_N * BLOCK_K must stay small enough to fit: 16384 elements of bf16 is
# 32 KiB per tile, 96 KiB at 3 stages, inside the 164 KiB SM budget.
_TILE_BUDGET = 16384


_SMS = None


def _sm_count() -> int:
    global _SMS
    if _SMS is None:
        _SMS = (torch.cuda.get_device_properties(0).multi_processor_count
                if torch.cuda.is_available() else 108)
    return _SMS


def gemv_config(n: int, k: int, sms: int = 0):
    """Pick BLOCK_N so the grid covers the SMs without starving each CTA."""
    sms = sms or _sm_count()
    bn = 16
    for cand in (16, 32, 64, 128, 256):
        bn = cand
        if _cdiv(n, cand) <= sms * 2:
            break
    bk = 64
    for cand in (256, 128, 64):
        if k % cand == 0 and bn * cand <= _TILE_BUDGET:
            bk = cand
            break
    return bn, bk


def gemv1(x, w, out=None):
    """Single-row projection: x [1, K], w [N, K] -> [1, N]."""
    k = x.shape[1]
    n = w.shape[0]
    y = torch.empty(1, n, device=x.device, dtype=x.dtype) if out is None else out
    bn = int(os.environ.get("ENGINE_GEMV1_BN", "0")) or 64
    bk = int(os.environ.get("ENGINE_GEMV1_BK", "0")) or 128
    _gemv1_kernel[(_cdiv(n, bn),)](
        x, w, y, k, w.stride(0), N=n, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=int(os.environ.get("ENGINE_GEMV1_W", "8")),
        num_stages=int(os.environ.get("ENGINE_GEMV1_S", "3")),
    )
    return y


def gemv(x, w, out=None, cfg=None):
    """x: [M, K]; w: [N, K] (untransposed) -> [M, N]."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype) if out is None else out
    bn, bk = cfg if cfg else gemv_config(n, k)
    _gemv_kernel[(_cdiv(n, bn),)](
        x, w, y, m, k, x.stride(0), w.stride(0), y.stride(0),
        N=n, BLOCK_N=bn, BLOCK_K=bk, MP=max(16, _next_pow2(m)),
        num_warps=4, num_stages=3,
    )
    return y


def group_pad(n_heads: int, n_kv: int) -> int:
    """tl.dot needs M >= 16, so a GQA group of G queries is padded up to this."""
    return max(16, _next_pow2(n_heads // n_kv))


def plan_splits(batch: int, n_kv: int, bucket: int, block_n: int = 0, target_cta: int = 0):
    """Pick a sequence-split count that keeps the SMs busy.

    More splits means more parallelism but a second (combine) launch; at small
    batch the launches dominate the tiny amount of KV actually read.
    """
    sms = _sm_count()
    block_n = block_n or int(os.environ.get("ENGINE_ATTN_BLOCK", "0")) or 128
    target_cta = target_cta or int(os.environ.get("ENGINE_ATTN_CTA", "0")) or 2 * sms
    base = batch * n_kv
    if base >= sms * float(os.environ.get("ENGINE_ATTN_FILL", "0.8")):
        # (batch x kv-heads) already fills the machine; splitting only buys a
        # second launch. Measured: b=16 is ~1% faster at one split.
        splits = 1
    else:
        splits = max(1, min(32, _cdiv(target_cta, base)))
    splits = max(1, min(splits, _cdiv(bucket, block_n)))
    chunk = _cdiv(_cdiv(bucket, splits), block_n) * block_n
    return splits, chunk, block_n


def flash_decode(q, k_cache, v_cache, seq_len_t, start_t, workspace, sm_scale):
    """q: [B, HQ, D]; caches: [B, HKV, S, D]; seq_len_t/start_t are device ints."""
    b, hq, d = q.shape
    hkv = k_cache.shape[1]
    g = hq // hkv
    acc, lsum, mmax, out, splits, chunk, block_n = workspace
    _flash_decode_split_kernel[(b * hkv, splits)](
        q, k_cache, v_cache, seq_len_t, start_t, acc, lsum, mmax, out, sm_scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        N_KV=hkv, G=g, GP=group_pad(hq, hkv), D=d, BLOCK_N=block_n, CHUNK=chunk,
        SPLITS_ONE=(splits == 1),
        num_warps=int(os.environ.get("ENGINE_ATTN_WARPS", "8")),
        num_stages=int(os.environ.get("ENGINE_ATTN_STAGES", "3")),
    )
    if splits == 1:
        return out
    _flash_decode_combine_kernel[(b * hkv * g,)](
        acc, lsum, mmax, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        out.stride(0), out.stride(1),
        N_KV=hkv, G=g, D=d, SPLITS=_next_pow2(splits), num_warps=4,
    )
    return out
