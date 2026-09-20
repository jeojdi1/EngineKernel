"""Triton kernels for the Qwen3 engine, each paired with a pure-torch fallback.

The fallbacks exist so the whole model can be exercised on CPU against
`transformers` without a GPU; they are never used on the benchmark path.
"""

from __future__ import annotations

import os
import tempfile

import torch


def _ensure_triton_cache():
    """Triton needs a writable cache dir; a sandbox's $HOME may not be one."""
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    home = os.path.join(os.path.expanduser("~"), ".triton")
    try:
        os.makedirs(home, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=home):
            pass
    except Exception:
        d = os.path.join(tempfile.gettempdir(), "ek-triton-cache")
        os.makedirs(d, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = d


try:
    if os.environ.get("ENGINE_NO_TRITON") == "1":
        raise ImportError("disabled")
    _ensure_triton_cache()
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only dev boxes / Triton-less sandboxes
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON


def disable_triton() -> None:
    global _HAS_TRITON
    _HAS_TRITON = False


def probe_triton() -> bool:
    """Actually compile and run one kernel. Importing Triton proves nothing: it
    builds its launcher with a C compiler at first use, and a slim sandbox may
    not have one. On any failure every wrapper below drops to torch ops."""
    global _HAS_TRITON
    if not (_HAS_TRITON and torch.cuda.is_available()):
        _HAS_TRITON = False
        return False
    try:
        x = torch.ones(2, 64, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        y = rms_norm(x, w, 1e-6)
        torch.cuda.synchronize()
        if not torch.isfinite(y.float()).all():
            raise RuntimeError("bad output")
    except Exception:
        _HAS_TRITON = False
    return _HAS_TRITON


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
        u = tl.load(X + row * stride_xm + N + cols, mask=mask, other=0.0)
        # Rounding points follow the reference exactly: torch's bf16 silu computes
        # in fp32 and rounds to bf16, and the product with `up` is then a bf16
        # multiply. One fp32 multiply rounded once is more accurate and is a
        # different function -- enough, on some prompt, to cross the tie margin.
        act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
        tl.store(Y + row * stride_ym + cols, act * u, mask=mask)

    @triton.jit
    def _heads_to_rows_kernel(
        X, Y, S,
        stride_xb, stride_xh, stride_xs, stride_yr,
        H: tl.constexpr, D: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        """Y[b*S + s, h*D + d] = X[b, h, s, d]: attention output, head-major, to
        the row layout the output projection consumes. Both sides are
        contiguous along d, so this is a permutation of 256-byte rows."""
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, D)
        mask = offs_s < S
        x = tl.load(X + b * stride_xb + h * stride_xh + offs_s[:, None] * stride_xs + offs_d[None, :],
                    mask=mask[:, None], other=0.0)
        tl.store(Y + (b * S + offs_s)[:, None] * stride_yr + h * D + offs_d[None, :], x,
                 mask=mask[:, None])

    @triton.jit
    def _qk_norm_rope_kv_kernel(
        QKV, QN, KN, COS, SIN, KC, VC, SlotBase,
        M, stride_qkv_m, stride_cos_m,
        stride_cb, stride_ch, stride_cs,
        N_Q: tl.constexpr, N_KV: tl.constexpr, D: tl.constexpr,
        HALF: tl.constexpr, EPS: tl.constexpr, M_PER_BATCH,
        BLOCK_M: tl.constexpr,
    ):
        """Per-head QK-RMSNorm + rotary, writing k/v straight into the cache.

        Rows of QKV are [ q: N_Q*D | k: N_KV*D | v: N_KV*D ]. Each program owns
        BLOCK_M rows of one head: at prefill M is batch*seq, and one program per
        (row, head) means hundreds of thousands of 128-element programs, which
        ran ~5x off roofline. The rotate-half partner stays inside the head, so
        the in-place read/write is still local to the program.
        """
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        is_q = h < N_Q
        is_v = h >= N_Q + N_KV

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        cols = tl.arange(0, D)
        base = QKV + rows[:, None] * stride_qkv_m + h * D
        x = tl.load(base + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
        w = tl.where(is_q, tl.load(QN + cols), tl.load(KN + cols))
        idx = tl.where(cols < HALF, cols + HALF, cols - HALF)
        xp = tl.load(base + idx[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        wp = tl.where(is_q, tl.load(QN + idx), tl.load(KN + idx))

        xn = (x * rstd[:, None]).to(tl.bfloat16) * w[None, :]
        xpn = (xp * rstd[:, None]).to(tl.bfloat16) * wp[None, :]
        rot = tl.where(cols[None, :] < HALF, -xpn, xpn)
        cos = tl.load(COS + rows[:, None] * stride_cos_m + cols[None, :],
                      mask=rmask[:, None], other=0.0)
        sin = tl.load(SIN + rows[:, None] * stride_cos_m + cols[None, :],
                      mask=rmask[:, None], other=0.0)
        # reference: (q * cos) + (rotate_half(q) * sin), every op in bf16 -- two
        # rounded products and a rounded sum, not one fp32 multiply-add
        out = (xn * cos) + (rot * sin)

        if is_q:
            tl.store(base + cols[None, :], out, mask=rmask[:, None])
        else:
            bidx = rows // M_PER_BATCH
            slot = rows % M_PER_BATCH + tl.load(SlotBase + bidx, mask=rmask, other=0)
            kv_h = tl.where(is_v, h - N_Q - N_KV, h - N_Q)
            dst = (bidx[:, None] * stride_cb + kv_h * stride_ch
                   + slot[:, None] * stride_cs + cols[None, :])
            if is_v:
                tl.store(VC + dst, x.to(tl.bfloat16), mask=rmask[:, None])
            else:
                tl.store(KC + dst, out, mask=rmask[:, None])

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
    def _norm_gemv_kernel(
        X, R, RO, NW, W, Y, M, K,
        stride_xm, stride_rm, stride_wn, stride_ym,
        N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
        EPS: tl.constexpr, HAS_ADD: tl.constexpr, SILU: tl.constexpr,
    ):
        """[residual add] + RMSNorm + projection [+ SwiGLU], one launch.

        A decode step is ~340 strictly dependent kernels and each boundary costs
        a few microseconds of dead time, which adds up to more than a third of
        the step. The norm cannot fuse into cuBLAS, so the projection is done
        here and the norm becomes its prologue (every program recomputes the
        row variance from the 2560-wide input, which is tiny next to its weight
        tile) and, for the MLP, SwiGLU becomes its epilogue. Rounding points
        are the reference's: bf16 residual add, fp32 norm rounded to bf16, bf16
        gain, fp32-accumulated products rounded to bf16, fp32 silu rounded to
        bf16, bf16 product with `up`.
        """
        pid = tl.program_id(0)
        offs_m = tl.arange(0, MP)
        mmask = offs_m < M

        ss = tl.zeros([MP], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            km = mmask[:, None] & (offs_k[None, :] < K)
            r = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=km, other=0.0)
            if HAS_ADD:
                r = r + tl.load(R + offs_m[:, None] * stride_rm + offs_k[None, :], mask=km, other=0.0)
                if pid == 0:
                    tl.store(RO + offs_m[:, None] * stride_rm + offs_k[None, :], r, mask=km)
            rf = r.to(tl.float32)
            ss += tl.sum(rf * rf, axis=1)
        rstd = tl.rsqrt(ss / K + EPS)

        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        acc = tl.zeros([MP, BLOCK_N], tl.float32)
        acc_u = tl.zeros([MP, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < K
            km = mmask[:, None] & kmask[None, :]
            r = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=km, other=0.0)
            if HAS_ADD:
                r = r + tl.load(R + offs_m[:, None] * stride_rm + offs_k[None, :], mask=km, other=0.0)
            nw = tl.load(NW + offs_k, mask=kmask, other=0.0)
            xn = (r.to(tl.float32) * rstd[:, None]).to(tl.bfloat16) * nw[None, :]
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0.0)
            acc += tl.dot(xn, tl.trans(w))
            if SILU:
                wu = tl.load(W + (offs_n + N)[:, None] * stride_wn + offs_k[None, :],
                             mask=nmask[:, None] & kmask[None, :], other=0.0)
                acc_u += tl.dot(xn, tl.trans(wu))
        om = mmask[:, None] & nmask[None, :]
        optr = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
        if SILU:
            g = acc.to(tl.bfloat16).to(tl.float32)
            act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
            tl.store(optr, act * acc_u.to(tl.bfloat16), mask=om)
        else:
            tl.store(optr, acc.to(tl.bfloat16), mask=om)

    @triton.jit
    def _gemv_swiglu_kernel(
        X, W, Y, M, K,
        stride_xm, stride_wn, stride_ym,
        I: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
    ):
        """Fused gate/up projection with SwiGLU as its epilogue.

        W is the [2I, K] gate-over-up matrix. Each program owns a block of
        activation outputs and computes both the gate rows and the matching up
        rows, so silu(gate) * up is written directly and the separate SwiGLU
        launch disappears. Same bytes streamed, half as many programs. Rounding
        follows the reference: both projections round to bf16, silu is fp32
        rounded to bf16, and the product is a bf16 multiply.
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < I
        offs_m = tl.arange(0, MP)
        mmask = offs_m < M
        acc_g = tl.zeros([MP, BLOCK_N], tl.float32)
        acc_u = tl.zeros([MP, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < K
            x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :],
                        mask=mmask[:, None] & kmask[None, :], other=0.0)
            wg = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                         mask=nmask[:, None] & kmask[None, :], other=0.0)
            wu = tl.load(W + (offs_n + I)[:, None] * stride_wn + offs_k[None, :],
                         mask=nmask[:, None] & kmask[None, :], other=0.0)
            acc_g += tl.dot(x, tl.trans(wg))
            acc_u += tl.dot(x, tl.trans(wu))
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
        tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], act * acc_u.to(tl.bfloat16),
                 mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _gemv_parts_kernel(
        X, W, P, M, K, KPER,
        stride_xm, stride_wn,
        N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
    ):
        """Split-K projection: program (n-block, k-split) writes its partial dot
        products, fp32, to P[split, row, n].

        o_proj and down_proj have only 2560 outputs, so tiling over outputs alone
        gives ~80 programs for 132 SMs and they run furthest from the bandwidth
        ceiling. Both have long inputs (4096 / 9728), so split those across
        programs too. The partials are summed by the add+norm kernel that
        consumes both projections anyway, so the split costs no extra launch.
        """
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        offs_m = tl.arange(0, MP)
        mmask = offs_m < M
        acc = tl.zeros([MP, BLOCK_N], tl.float32)
        hi = tl.minimum((ps + 1) * KPER, K)
        for k0 in range(ps * KPER, hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < hi
            x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :],
                        mask=mmask[:, None] & kmask[None, :], other=0.0)
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
        tl.store(P + ps * M * N + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _add_rms_norm_parts_kernel(
        P, R, W, Y,
        stride_ps, stride_rm, stride_ym,
        N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr, SK: tl.constexpr,
    ):
        """add+RMSNorm whose addend arrives as SK fp32 partial sums. The sum is
        rounded to bf16 first -- that is the projection's output in the
        reference -- and everything after matches _add_rms_norm_kernel."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        offs_s = tl.arange(0, SK)
        parts = tl.load(P + offs_s[:, None] * stride_ps + row * N + cols[None, :],
                        mask=mask[None, :], other=0.0)
        x = tl.sum(parts, axis=0).to(tl.bfloat16)
        r = tl.load(R + row * stride_rm + cols, mask=mask, other=0.0)
        r = (r + x).to(tl.bfloat16)
        tl.store(R + row * stride_rm + cols, r, mask=mask)
        rf = r.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(rf * rf, axis=0) / N + EPS)
        xn = (rf * rstd).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(Y + row * stride_ym + cols, (xn * w).to(tl.bfloat16), mask=mask)

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
    def _gemv1_sk_kernel(
        X, W, P, K,
        stride_wn, stride_ps,
        N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Split-K partial for a single-row projection.

        o_proj is N=2560: at BLOCK_N=64 that is 40 CTAs, which cannot fill 132
        SMs, and cuBLAS reaches only ~47% of peak for the same reason. Splitting
        K multiplies the grid by SPLIT_K. Partials are written per split and
        summed by a separate (tiny) kernel rather than atomically, so the result
        is deterministic run to run.
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        acc = tl.zeros([BLOCK_N], tl.float32)
        k_per = K // SPLIT_K
        for k0 in range(pid_k * k_per, (pid_k + 1) * k_per, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_k).to(tl.float32)
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=nmask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(P + pid_k * stride_ps + offs_n, acc, mask=nmask)

    @triton.jit
    def _gemv1_sk_combine(
        P, Y, stride_ps,
        N: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        offs_s = tl.arange(0, SPLIT_K)
        v = tl.load(P + offs_s[:, None] * stride_ps + offs_n[None, :],
                    mask=nmask[None, :], other=0.0)
        tl.store(Y + offs_n, tl.sum(v, axis=0).to(tl.bfloat16), mask=nmask)

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
        BLOCK_N: tl.constexpr, CHUNK, SPLITS_ONE: tl.constexpr,
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
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
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
    def _ngram_draft_kernel(
        HIST, HLEN, TOK,
        H, Q: tl.constexpr, HB: tl.constexpr, QP: tl.constexpr,
    ):
        """Per row: find the most recent earlier occurrence of the trailing
        3-/2-/1-gram in the row's own history (longer match wins, then the later
        position) and propose what followed it, wrapping around short cycles.
        TOK[row] = [last emitted token, draft_1 .. draft_{Q-1}]. Integer-exact
        replacement for ~25 torch launches."""
        row = tl.program_id(0)
        hl = tl.load(HLEN + row)
        base = HIST + row * H
        idx = tl.arange(0, HB)
        inb = idx < hl
        h = tl.load(base + idx, mask=inb, other=-1)
        k_last = tl.load(base + hl - 1)
        k_prev = tl.load(base + tl.maximum(hl - 2, 0))
        k_pp = tl.load(base + tl.maximum(hl - 3, 0))
        h1 = tl.load(base + idx - 1, mask=inb & (idx >= 1), other=-1)
        h2 = tl.load(base + idx - 2, mask=inb & (idx >= 2), other=-1)
        m1 = (h == k_last) & (idx <= hl - 2)
        m2 = m1 & (h1 == k_prev) & (idx >= 1)
        m3 = m2 & (h2 == k_pp) & (idx >= 2)
        big = 1048576
        score = (tl.where(m1, idx + 1, 0) + tl.where(m2, big, 0) + tl.where(m3, 2 * big, 0)).to(tl.int64)
        p = tl.max(score, axis=0) % big
        span = tl.maximum(hl - p, 1)
        j = tl.arange(0, QP)
        src = tl.minimum(p + (j - 1) % span, H - 1)
        d = tl.load(base + src, mask=(j >= 1) & (j < Q), other=0)
        out = tl.where(j == 0, k_last, d)
        tl.store(TOK + row * Q + j, out, mask=j < Q)

    @triton.jit
    def _spec_accept_kernel(
        TOK, AM, HIST, HLEN, LENB, POSB, REM, OUT_TOK, OUT_ADV, STEP,
        H, B, Q: tl.constexpr, QP: tl.constexpr,
    ):
        """Per row: accept the longest draft prefix the model agrees with, then
        do all the bookkeeping -- outputs, history, lengths, positions, budget.
        Integer-exact replacement for ~25 torch launches."""
        row = tl.program_id(0)
        j = tl.arange(0, QP)
        inq = j < Q
        am = tl.load(AM + row * Q + j, mask=inq, other=0)
        nxt = tl.load(TOK + row * Q + j + 1, mask=j < Q - 1, other=-1)
        match = (nxt == am) & (j < Q - 1)
        nacc = tl.minimum(tl.min(tl.where(match, QP, j), axis=0), Q - 1)
        rem = tl.load(REM + row)
        adv = tl.minimum(nacc + 1, rem)
        step = tl.load(STEP)
        tl.store(OUT_TOK + (step * B + row) * Q + j, am.to(tl.int32), mask=inq)
        tl.store(OUT_ADV + step * B + row, adv.to(tl.int32))
        hl = tl.load(HLEN + row)
        tl.store(HIST + row * H + hl + j, am, mask=inq & (hl + j < H))
        tl.store(HLEN + row, hl + adv)
        tl.store(LENB + row, tl.load(LENB + row) + adv)
        tl.store(POSB + row, tl.load(POSB + row) + adv)
        tl.store(REM + row, rem - adv)

    @triton.jit
    def _rope_attn_decode_kernel(
        QKV, QN, KN, COS, SIN, K, V, SeqLen, Start,
        Acc, Lsum, Mmax, Out,
        sm_scale,
        stride_qkv_b, stride_cos_b,
        stride_ob, stride_oh,
        stride_kb, stride_kh, stride_ks,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        N_Q: tl.constexpr, N_KV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        D: tl.constexpr, HALF: tl.constexpr, EPS: tl.constexpr,
        BLOCK_N: tl.constexpr, CHUNK, SPLITS_ONE: tl.constexpr,
    ):
        """Decode attention that does its own QK-norm, rotary and cache write.

        One program per (sequence, kv-head, split), as in _flash_decode_split_kernel,
        but it starts from the raw fused QKV row: it norms and rotates its group
        of G query heads and the new key, and the split that owns the new slot
        stores the key/value before attending. That retires the separate
        norm+rope+cache kernel -- a launch per layer whose work is a few hundred
        elements. Arithmetic and rounding points are identical to
        _qk_norm_rope_kv_kernel.
        """
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // N_KV
        h = pid_bh % N_KV

        seq_len = tl.load(SeqLen)          # valid length including the new token
        start = tl.load(Start + b)
        slot = seq_len - 1
        lo = pid_s * CHUNK
        hi = tl.minimum(lo + CHUNK, seq_len)
        owns_new = (slot >= lo) & (slot < hi)
        lo = tl.maximum(lo, start)

        cols = tl.arange(0, D)
        idx = tl.where(cols < HALF, cols + HALF, cols - HALF)
        cos = tl.load(COS + b * stride_cos_b + cols)
        sin = tl.load(SIN + b * stride_cos_b + cols)
        row = QKV + b * stride_qkv_b

        # ---- the G query heads of this kv-head: norm + rope, padded to GP rows ----
        offs_g = tl.arange(0, GP)
        gmask = offs_g < G
        qbase = row + ((h * G + offs_g) * D)[:, None]
        xq = tl.load(qbase + cols[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
        xqp = tl.load(qbase + idx[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
        rq = tl.rsqrt(tl.sum(xq * xq, axis=1) / D + EPS)
        wq = tl.load(QN + cols)
        wqp = tl.load(QN + idx)
        qn_ = (xq * rq[:, None]).to(tl.bfloat16) * wq[None, :]
        qpn = (xqp * rq[:, None]).to(tl.bfloat16) * wqp[None, :]
        qrot = tl.where(cols[None, :] < HALF, -qpn, qpn)
        q = (qn_ * cos[None, :]) + (qrot * sin[None, :])
        q = tl.where(gmask[:, None], q, 0.0).to(tl.bfloat16)   # the 0.0 literal would promote q to fp32

        kbase = K + b * stride_kb + h * stride_kh
        vbase = V + b * stride_kb + h * stride_kh
        if owns_new:
            krow = row + (N_Q + h) * D
            xk = tl.load(krow + cols).to(tl.float32)
            xkp = tl.load(krow + idx).to(tl.float32)
            rk = tl.rsqrt(tl.sum(xk * xk, axis=0) / D + EPS)
            kn_ = (xk * rk).to(tl.bfloat16) * tl.load(KN + cols)
            kpn = (xkp * rk).to(tl.bfloat16) * tl.load(KN + idx)
            krot = tl.where(cols < HALF, -kpn, kpn)
            tl.store(kbase + slot * stride_ks + cols, (kn_ * cos) + (krot * sin))
            tl.store(vbase + slot * stride_ks + cols, tl.load(row + (N_Q + N_KV + h) * D + cols))

        m_i = tl.full([GP], -1e30, tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < hi
            k = tl.load(kbase + offs_n[:, None] * stride_ks + cols[None, :],
                        mask=nmask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(nmask[None, :], qk, -1e30)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(vbase + offs_n[:, None] * stride_ks + cols[None, :],
                        mask=nmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        if SPLITS_ONE:
            tl.store(Out + b * stride_ob + (h * G + offs_g)[:, None] * stride_oh + cols[None, :],
                     (acc / l_i[:, None]).to(tl.bfloat16), mask=gmask[:, None])
        else:
            aptr = Acc + b * stride_ab + h * stride_ah + pid_s * stride_as
            tl.store(aptr + offs_g[:, None] * stride_ag + cols[None, :], acc, mask=gmask[:, None])
            lptr = Lsum + b * stride_lb + h * stride_lh + pid_s * stride_ls
            mptr = Mmax + b * stride_lb + h * stride_lh + pid_s * stride_ls
            tl.store(lptr + offs_g, l_i, mask=gmask)
            tl.store(mptr + offs_g, m_i, mask=gmask)

    @triton.jit
    def _rope_attn_verify_kernel(
        QKV, QN, KN, COS, SIN, K, V, LenB, Start,
        Acc, Lsum, Mmax, Out,
        sm_scale,
        stride_qkv_r, stride_cos_r,
        stride_ob, stride_oh,
        stride_kb, stride_kh, stride_ks,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        N_Q: tl.constexpr, N_KV: tl.constexpr, G: tl.constexpr, NQ: tl.constexpr,
        GP: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, EPS: tl.constexpr,
        BLOCK_N: tl.constexpr, CHUNK, SPLITS_ONE: tl.constexpr,
    ):
        """Speculative-verify attention that does its own QK-norm, rotary and
        cache writes: _flash_verify_split_kernel starting from the raw fused QKV
        rows, as _rope_attn_decode_kernel does for plain decode. Row b*NQ+j of
        QKV is query j of sequence b; its key/value go to slot len_b+j, stored by
        whichever split owns that slot before it attends. Arithmetic and
        rounding points match _qk_norm_rope_kv_kernel."""
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // N_KV
        h = pid_bh % N_KV
        len_b = tl.load(LenB + b)
        start = tl.load(Start + b)
        lo0 = pid_s * CHUNK
        hi = tl.minimum(lo0 + CHUNK, len_b + NQ)
        lo = tl.maximum(lo0, start)

        cols = tl.arange(0, D)
        idx = tl.where(cols < HALF, cols + HALF, cols - HALF)
        kbase = K + b * stride_kb + h * stride_kh
        vbase = V + b * stride_kb + h * stride_kh
        wk = tl.load(KN + cols)
        wkp = tl.load(KN + idx)
        for jj in tl.static_range(NQ):
            slot = len_b + jj
            if (slot >= lo0) & (slot < lo0 + CHUNK):
                r = QKV + (b * NQ + jj) * stride_qkv_r
                c_ = tl.load(COS + (b * NQ + jj) * stride_cos_r + cols)
                s_ = tl.load(SIN + (b * NQ + jj) * stride_cos_r + cols)
                xk = tl.load(r + (N_Q + h) * D + cols).to(tl.float32)
                xkp = tl.load(r + (N_Q + h) * D + idx).to(tl.float32)
                rk = tl.rsqrt(tl.sum(xk * xk, axis=0) / D + EPS)
                kn_ = (xk * rk).to(tl.bfloat16) * wk
                kpn = (xkp * rk).to(tl.bfloat16) * wkp
                krot = tl.where(cols < HALF, -kpn, kpn)
                tl.store(kbase + slot * stride_ks + cols, (kn_ * c_) + (krot * s_))
                tl.store(vbase + slot * stride_ks + cols, tl.load(r + (N_Q + N_KV + h) * D + cols))

        offs_qi = tl.arange(0, GP)
        qmask = offs_qi < NQ * G
        j = offs_qi // G
        g = offs_qi % G
        qrow = b * NQ + j
        qlen = len_b + j + 1
        qb = QKV + qrow[:, None] * stride_qkv_r + ((h * G + g) * D)[:, None]
        xq = tl.load(qb + cols[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
        xqp = tl.load(qb + idx[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
        rq = tl.rsqrt(tl.sum(xq * xq, axis=1) / D + EPS)
        cq = tl.load(COS + qrow[:, None] * stride_cos_r + cols[None, :], mask=qmask[:, None], other=0.0)
        sq = tl.load(SIN + qrow[:, None] * stride_cos_r + cols[None, :], mask=qmask[:, None], other=0.0)
        qn_ = (xq * rq[:, None]).to(tl.bfloat16) * tl.load(QN + cols)[None, :]
        qpn = (xqp * rq[:, None]).to(tl.bfloat16) * tl.load(QN + idx)[None, :]
        qrot = tl.where(cols[None, :] < HALF, -qpn, qpn)
        q = tl.where(qmask[:, None], (qn_ * cq) + (qrot * sq), 0.0).to(tl.bfloat16)

        m_i = tl.full([GP], -1e30, tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < hi
            k = tl.load(kbase + offs_n[:, None] * stride_ks + cols[None, :], mask=nmask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            valid = nmask[None, :] & (offs_n[None, :] < qlen[:, None])
            qk = tl.where(valid, qk, -1e30)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(vbase + offs_n[:, None] * stride_ks + cols[None, :], mask=nmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        if SPLITS_ONE:
            tl.store(Out + qrow[:, None] * stride_ob + (h * G + g)[:, None] * stride_oh + cols[None, :],
                     (acc / l_i[:, None]).to(tl.bfloat16), mask=qmask[:, None])
        else:
            aptr = Acc + b * stride_ab + h * stride_ah + pid_s * stride_as
            tl.store(aptr + offs_qi[:, None] * stride_ag + cols[None, :], acc, mask=qmask[:, None])
            lptr = Lsum + b * stride_lb + h * stride_lh + pid_s * stride_ls
            mptr = Mmax + b * stride_lb + h * stride_lh + pid_s * stride_ls
            tl.store(lptr + offs_qi, l_i, mask=qmask)
            tl.store(mptr + offs_qi, m_i, mask=qmask)

    @triton.jit
    def _flash_verify_split_kernel(
        Q, K, V, LenB, Start,
        Acc, Lsum, Mmax, Out,
        sm_scale,
        stride_qb, stride_qh,
        stride_ob, stride_oh,
        stride_kb, stride_kh, stride_ks,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        N_KV: tl.constexpr, G: tl.constexpr, NQ, GP: tl.constexpr,
        D: tl.constexpr, BLOCK_N: tl.constexpr, CHUNK,
        SPLITS_ONE: tl.constexpr,
    ):
        """Speculative verify: NQ query tokens per sequence share one pass
        over that sequence's KV. Query j sees slots [start, len_b + j], so the
        drafts are causal among themselves. All NQ*G query rows of a kv-head
        ride the M dimension of one dot, so K/V is read once, not NQ times.
        """
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // N_KV
        h = pid_bh % N_KV

        len_b = tl.load(LenB + b)
        start = tl.load(Start + b)
        lo = pid_s * CHUNK
        hi = tl.minimum(lo + CHUNK, len_b + NQ)
        lo = tl.maximum(lo, start)

        offs_d = tl.arange(0, D)
        offs_qi = tl.arange(0, GP)
        qmask = offs_qi < NQ * G
        j = offs_qi // G
        g = offs_qi % G
        qrow = b * NQ + j
        qlen = len_b + j + 1

        q = tl.load(
            Q + qrow[:, None] * stride_qb + (h * G + g)[:, None] * stride_qh + offs_d[None, :],
            mask=qmask[:, None], other=0.0,
        )
        m_i = tl.full([GP], -1e30, tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        kbase = K + b * stride_kb + h * stride_kh
        vbase = V + b * stride_kb + h * stride_kh

        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < hi
            k = tl.load(kbase + offs_n[:, None] * stride_ks + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            valid = nmask[None, :] & (offs_n[None, :] < qlen[:, None])
            qk = tl.where(valid, qk, -1e30)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(vbase + offs_n[:, None] * stride_ks + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        if SPLITS_ONE:
            tl.store(Out + qrow[:, None] * stride_ob + (h * G + g)[:, None] * stride_oh
                     + offs_d[None, :],
                     (acc / l_i[:, None]).to(tl.bfloat16), mask=qmask[:, None])
        else:
            aptr = Acc + b * stride_ab + h * stride_ah + pid_s * stride_as
            tl.store(aptr + offs_qi[:, None] * stride_ag + offs_d[None, :], acc,
                     mask=qmask[:, None])
            lptr = Lsum + b * stride_lb + h * stride_lh + pid_s * stride_ls
            mptr = Mmax + b * stride_lb + h * stride_lh + pid_s * stride_ls
            tl.store(lptr + offs_qi, l_i, mask=qmask)
            tl.store(mptr + offs_qi, m_i, mask=qmask)

    @triton.jit
    def _flash_verify_combine_kernel(
        Acc, Lsum, Mmax, Out,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_lb, stride_lh, stride_ls,
        stride_ob, stride_oh,
        N_KV: tl.constexpr, G: tl.constexpr, NQ, D: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        qi = pid % (NQ * G)
        h = (pid // (NQ * G)) % N_KV
        b = pid // (NQ * G * N_KV)
        j = qi // G
        g = qi % G
        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, SPLITS)
        m_s = tl.load(Mmax + b * stride_lb + h * stride_lh + offs_s * stride_ls + qi)
        l_s = tl.load(Lsum + b * stride_lb + h * stride_lh + offs_s * stride_ls + qi)
        m = tl.max(m_s, axis=0)
        scale = tl.exp(m_s - m)
        denom = tl.sum(l_s * scale, axis=0)
        a = tl.load(Acc + b * stride_ab + h * stride_ah + offs_s[:, None] * stride_as
                    + qi * stride_ag + offs_d[None, :])
        out = tl.sum(a * scale[:, None], axis=0) / denom
        tl.store(Out + (b * NQ + j) * stride_ob + (h * G + g) * stride_oh + offs_d,
                 out.to(tl.bfloat16))

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


def heads_to_rows(o):
    """[B, H, S, D] attention output -> [B*S, H*D]. Free when the transposed
    view is already contiguous (the flash backend's layout); cuDNN's output is
    head-major, and torch's generic copy of it ran at ~1.3 TB/s."""
    b, h, s, d = o.shape
    t = o.transpose(1, 2)
    if t.is_contiguous() or not (_HAS_TRITON and o.is_cuda):
        return t.reshape(b * s, h * d)
    y = torch.empty(b * s, h * d, dtype=o.dtype, device=o.device)
    _heads_to_rows_kernel[(b * h, _cdiv(s, 64))](
        o, y, s, o.stride(0), o.stride(1), o.stride(2), y.stride(0),
        H=h, D=d, BLOCK_S=64, num_warps=4,
    )
    return y


def qk_norm_rope_kv(qkv, qn, kn, cos, sin, k_cache, v_cache, slot_base,
                    n_q, n_kv, eps, m_per_batch):
    """QK-norm + rotary on the fused QKV, with k/v written into the caches.

    qkv: [M, (n_q + 2*n_kv)*D]; caches: [B, n_kv, S, D]; slot_base: device int64
    [B], the first slot each sequence writes (zeros for prefill).
    """
    d = qn.shape[0]
    m = qkv.shape[0]
    if not (_HAS_TRITON and qkv.is_cuda):
        # device-side only (no .item()), so this path can sit inside a CUDA graph
        b = m // m_per_batch
        half = d // 2
        q = qkv[:, : n_q * d].view(m, n_q, d)
        k = qkv[:, n_q * d: (n_q + n_kv) * d].view(m, n_kv, d)
        v = qkv[:, (n_q + n_kv) * d:].view(m, n_kv, d)
        c, sn = cos.unsqueeze(1), sin.unsqueeze(1)
        for t, w in ((q, qn), (k, kn)):
            tn = _torch_rms_norm(t, w, eps)
            rot = torch.cat((-tn[..., half:], tn[..., :half]), dim=-1)
            t.copy_(tn * c + rot * sn)
        ar = torch.arange(m_per_batch, device=qkv.device)
        slots = slot_base[:b, None] + ar[None, :]
        bi = torch.arange(b, device=qkv.device)[:, None].expand(b, m_per_batch)
        k_cache[bi, :, slots] = k.view(b, m_per_batch, n_kv, d)
        v_cache[bi, :, slots] = v.view(b, m_per_batch, n_kv, d)
        return qkv
    block_m = int(os.environ.get("ENGINE_ROPE_BM", "0")) or (1 if m <= 64 else 16)  # swept on sm_90
    _qk_norm_rope_kv_kernel[(_cdiv(m, block_m), n_q + 2 * n_kv)](
        qkv, qn, kn, cos, sin, k_cache, v_cache, slot_base,
        m, qkv.stride(0), cos.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        N_Q=n_q, N_KV=n_kv, D=d, HALF=d // 2, EPS=eps,
        M_PER_BATCH=m_per_batch, BLOCK_M=block_m,
        num_warps=int(os.environ.get("ENGINE_ROPE_W", "4")),
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


_SK_PARTIALS: dict = {}


def gemv1_sk(x, w, out=None, bn=64, bk=128, sk=4, warps=8, stages=3):
    """Single-row projection with split-K. x: [1, K], w: [N, K] -> [1, N]."""
    k = x.shape[1]
    n = w.shape[0]
    y = torch.empty(1, n, device=x.device, dtype=x.dtype) if out is None else out
    key = (n, sk, x.device.index)
    p = _SK_PARTIALS.get(key)
    if p is None:
        p = torch.empty(sk, n, device=x.device, dtype=torch.float32)
        _SK_PARTIALS[key] = p
    _gemv1_sk_kernel[(_cdiv(n, bn), sk)](
        x, w, p, k, w.stride(0), p.stride(0),
        N=n, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=warps, num_stages=stages,
    )
    _gemv1_sk_combine[(_cdiv(n, 256),)](
        p, y, p.stride(0), N=n, BLOCK_N=256, SPLIT_K=sk, num_warps=4,
    )
    return y


def norm_gemv(x, residual, norm_w, w, eps, silu=False):
    """y = proj(rmsnorm(x + residual)); returns (y, x + residual).

    residual=None means no add (first layer): the returned residual is x itself.
    With silu=True, w is the fused [2I, K] gate/up matrix and y is silu(gate)*up.
    Decode-sized inputs only: the row dimension is padded to a power of two and
    must stay below 64 on Hopper with Triton 3.1.0.
    """
    m, k = x.shape
    n = w.shape[0] // 2 if silu else w.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    has_add = residual is not None
    r_out = torch.empty_like(x) if has_add else x
    t_bn, t_bk, t_w, t_s = gemv_tuned(w.shape[0], m, k)
    bn = int(os.environ.get("ENGINE_NG_BN", "0")) or t_bn
    bk = int(os.environ.get("ENGINE_NG_BK", "0")) or t_bk
    _norm_gemv_kernel[(_cdiv(n, bn),)](
        x, residual if has_add else x, r_out, norm_w, w, y, m, k,
        x.stride(0), (residual if has_add else x).stride(0), w.stride(0), y.stride(0),
        N=n, BLOCK_N=bn, BLOCK_K=bk, MP=max(16, _next_pow2(m)), EPS=eps,
        HAS_ADD=has_add, SILU=silu,
        num_warps=int(os.environ.get("ENGINE_NG_W", "0")) or t_w,
        num_stages=int(os.environ.get("ENGINE_NG_S", "0")) or t_s,
    )
    return y, r_out


def gemv_swiglu(x, w):
    """x: [M, K]; w: fused [2I, K] gate/up -> silu(x @ gate.T) * (x @ up.T), [M, I]."""
    m, k = x.shape
    i = w.shape[0] // 2
    y = torch.empty(m, i, device=x.device, dtype=x.dtype)
    t_bn, t_bk, t_w, t_s = gemv_tuned(w.shape[0], m, k)
    bn = int(os.environ.get("ENGINE_SG_BN", "0")) or t_bn
    bk = int(os.environ.get("ENGINE_SG_BK", "0")) or t_bk
    _gemv_swiglu_kernel[(_cdiv(i, bn),)](
        x, w, y, m, k, x.stride(0), w.stride(0), y.stride(0),
        I=i, BLOCK_N=bn, BLOCK_K=bk, MP=max(16, _next_pow2(m)),
        num_warps=int(os.environ.get("ENGINE_SG_W", "0")) or t_w,
        num_stages=int(os.environ.get("ENGINE_SG_S", "0")) or t_s,
    )
    return y


def parts_tuned(k: int, m: int):
    """(splits, BLOCK_N, BLOCK_K, warps, stages) for the split-K projections.
    k <= 4096 is o_proj, larger is down_proj.

    Chosen by alternating A/B of the whole engine on an H100, not by the
    standalone sweep: four splits won every standalone measurement but are ~1%
    slower than two in the real chain up to 16 rows, and 1.6% faster at 32.
    """
    if m <= 16:
        return (2, 64, 128, 4, 5) if k <= 4096 else (2, 64, 256, 4, 5)
    return (4, 16, 64, 2, 5) if k <= 4096 else (4, 32, 128, 4, 3)


def gemv_parts(x, w, sk=None):
    """Split-K projection for the narrow-output matrices. Returns fp32 partial
    sums [sk, M, N]; feed them to add_rms_norm_parts."""
    m, k = x.shape
    n = w.shape[0]
    t_sk, t_bn, t_bk, t_w, t_s = parts_tuned(k, m)
    sk = sk or int(os.environ.get("ENGINE_SPLIT_K", "0")) or t_sk
    parts = torch.empty(sk, m, n, device=x.device, dtype=torch.float32)
    bn = int(os.environ.get("ENGINE_SK_BN", "0")) or t_bn
    bk = int(os.environ.get("ENGINE_SK_BK", "0")) or t_bk
    _gemv_parts_kernel[(_cdiv(n, bn), sk)](
        x, w, parts, m, k, _cdiv(k, sk), x.stride(0), w.stride(0),
        N=n, BLOCK_N=bn, BLOCK_K=bk, MP=max(16, _next_pow2(m)),
        num_warps=int(os.environ.get("ENGINE_SK_W", "0")) or t_w,
        num_stages=int(os.environ.get("ENGINE_SK_S", "0")) or t_s,
    )
    return parts


def add_rms_norm_parts(parts, residual, w, eps):
    """residual += sum(parts) (rounded to bf16 first); returns (norm * w, residual)."""
    sk, m, n = parts.shape
    y = torch.empty(m, n, device=parts.device, dtype=residual.dtype)
    _add_rms_norm_parts_kernel[(m,)](
        parts, residual, w, y, parts.stride(0), residual.stride(0), y.stride(0),
        N=n, BLOCK=_next_pow2(n), EPS=eps, SK=sk, num_warps=8,
    )
    return y, residual


def gemv_tuned(n: int, m: int, k: int = 0):
    """(BLOCK_N, BLOCK_K, warps, stages) by projection and row count.

    From per-shape coordinate descent on an H100 with Triton 3.1.0, timed in a
    36-layer dependent chain, then a finer second pass (warps 1-4, stages 2-8).
    """
    if n < 4096:
        if k > 8192:                  # down_proj
            return (32, 512, 4, 4) if m == 1 else (32, 512, 4, 5) if m <= 16 else (32, 256, 4, 5)
        return 32, 256, 4, 5          # o_proj
    if n < 16384:                     # fused qkv
        return (16, 128, 2, 5) if m == 1 else (32, 64, 2, 8) if m <= 16 else (32, 128, 2, 5)
    return (16, 128, 1, 5) if m == 1 else (64, 128, 4, 3) if m <= 16 else (32, 128, 2, 3)   # fused gate/up


def gemv(x, w, out=None, cfg=None):
    """x: [M, K]; w: [N, K] (untransposed) -> [M, N]."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype) if out is None else out
    bn, bk = cfg if cfg else gemv_config(n, k)
    # Tuned on an H100 with Triton 3.1.0 inside a 36-layer dependent chain (a
    # 254-config sweep, then per-shape coordinate descent). Pipeline depth was
    # the knob earlier sweeps never pushed past 3; with it the chain runs 12%
    # faster than cuBLAS. Wide projections want narrower blocks and fewer warps.
    t_bn, t_bk, t_w, t_s = gemv_tuned(n, m, k)
    if k % t_bk:
        t_bk = bk
    bn = int(os.environ.get("ENGINE_GEMV_BN", "0")) or t_bn
    bk = int(os.environ.get("ENGINE_GEMV_BK", "0")) or t_bk
    t_w = int(os.environ.get("ENGINE_GEMV_W", "0")) or t_w
    t_s = int(os.environ.get("ENGINE_GEMV_S", "0")) or t_s
    # Hopper's wgmma needs M>=64; padding only to 16 drops tl.dot onto a much
    # slower path, which is why this kernel lost badly on sm_90.
    mp = max(int(os.environ.get("ENGINE_GEMV_MP", "0")) or 16, _next_pow2(m))
    _gemv_kernel[(_cdiv(n, bn),)](
        x, w, y, m, k, x.stride(0), w.stride(0), y.stride(0),
        N=n, BLOCK_N=bn, BLOCK_K=bk, MP=mp,
        num_warps=t_w, num_stages=t_s,
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
    target_cta = target_cta or int(os.environ.get("ENGINE_ATTN_CTA", "0")) or 2 * sms
    base = batch * n_kv
    if base >= sms * float(os.environ.get("ENGINE_ATTN_FILL", "0.8")):
        # (batch x kv-heads) already fills the machine; splitting only buys a
        # second launch. Measured: b=16 is ~1% faster at one split.
        splits = 1
    else:
        splits = max(1, min(32, _cdiv(target_cta, base)))
    if not block_n:
        block_n = int(os.environ.get("ENGINE_ATTN_BLOCK", "0"))
    if not block_n:
        # 64-key tiles with 4 warps (see attn_warps) ran 20-40% faster than
        # 128/8 for batch <= 8 at every context swept (b=4 x 2048: 23.4 -> 14.5
        # us per layer; verify at b=1 x 2048: 12.3 -> 8.8) and 11% faster at
        # batch 32; only a single wave of unsplit programs (batch 16) prefers
        # the wider tile.
        block_n = 128 if (splits == 1 and base <= sms) else 64
    splits = max(1, min(splits, _cdiv(bucket, block_n)))
    chunk = _cdiv(_cdiv(bucket, splits), block_n) * block_n
    return splits, chunk, block_n


def attn_warps() -> int:
    """Warps per attention program; see plan_splits."""
    return int(os.environ.get("ENGINE_ATTN_WARPS", "0")) or 4


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
        num_warps=attn_warps(),
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


def flash_verify(q, k_cache, v_cache, len_b, start_t, workspace, sm_scale, nq):
    """q: [B*nq, HQ, D]; caches [B, HKV, S, D]; len_b/start_t: device [B]."""
    bq, hq, d = q.shape
    b = bq // nq
    hkv = k_cache.shape[1]
    g = hq // hkv
    acc, lsum, mmax, out, splits, chunk, block_n = workspace
    gp = acc.shape[3]
    _flash_verify_split_kernel[(b * hkv, splits)](
        q, k_cache, v_cache, len_b, start_t, acc, lsum, mmax, out, sm_scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        N_KV=hkv, G=g, NQ=nq, GP=gp, D=d, BLOCK_N=block_n, CHUNK=chunk,
        SPLITS_ONE=(splits == 1),
        num_warps=attn_warps(),
        num_stages=int(os.environ.get("ENGINE_ATTN_STAGES", "3")),
    )
    if splits == 1:
        return out
    _flash_verify_combine_kernel[(b * hkv * nq * g,)](
        acc, lsum, mmax, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        out.stride(0), out.stride(1),
        N_KV=hkv, G=g, NQ=nq, D=d, SPLITS=_next_pow2(splits), num_warps=4,
    )
    return out


def attn_torch(q, k_cache, v_cache, len_b, start_t, sm_scale, nq, bucket):
    """Triton-free verify/decode attention, graph-safe.

    GQA is done by folding the query group into the row dimension of a bmm, so
    K/V are never repeat-interleaved (that copy would double the step's memory
    traffic). Reads the whole bucket and masks; the Triton path early-exits.
    """
    bq, hq, d = q.shape
    b = bq // nq
    hkv = k_cache.shape[1]
    g = hq // hkv
    qg = q.view(b, nq, hkv, g, d).permute(0, 2, 1, 3, 4).reshape(b, hkv, nq * g, d)
    k = k_cache[:, :, :bucket]
    v = v_cache[:, :, :bucket]
    scores = torch.matmul(qg, k.transpose(-1, -2)).float() * sm_scale
    ar = torch.arange(bucket, device=q.device)
    qlen = len_b[:, None] + torch.arange(nq, device=q.device)[None, :] + 1
    qlen = qlen.repeat_interleave(g, dim=1)
    ok = (ar[None, None, :] < qlen[:, :, None]) & (ar[None, None, :] >= start_t[:, None, None])
    scores = scores.masked_fill(~ok[:, None, :, :], float("-inf"))
    p = torch.softmax(scores, dim=-1).to(q.dtype)
    out = torch.matmul(p, v)
    return out.view(b, hkv, nq, g, d).permute(0, 2, 1, 3, 4).reshape(bq, hq, d)


def rope_attn_decode(qkv, qn, kn, cos, sin, k_cache, v_cache, seq_len_t, start_t, workspace,
                     sm_scale, n_q, eps):
    """QK-norm + rotary + cache write + decode attention, from the raw fused QKV
    rows [B, (n_q + 2*n_kv)*D]. seq_len_t already counts the new token."""
    b = qkv.shape[0]
    hkv = k_cache.shape[1]
    d = qn.shape[0]
    g = n_q // hkv
    acc, lsum, mmax, out, splits, chunk, block_n = workspace
    _rope_attn_decode_kernel[(b * hkv, splits)](
        qkv, qn, kn, cos, sin, k_cache, v_cache, seq_len_t, start_t,
        acc, lsum, mmax, out, sm_scale,
        qkv.stride(0), cos.stride(0),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        N_Q=n_q, N_KV=hkv, G=g, GP=group_pad(n_q, hkv), D=d, HALF=d // 2, EPS=eps,
        BLOCK_N=block_n, CHUNK=chunk, SPLITS_ONE=(splits == 1),
        num_warps=attn_warps(),
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


def ngram_draft(hist, hist_len, tokens):
    """tokens[B, Q] <- [last token, Q-1 n-gram drafts] per row. All int64."""
    b, h = hist.shape
    q = tokens.shape[1]
    _ngram_draft_kernel[(b,)](hist, hist_len, tokens, h, Q=q, HB=_next_pow2(h),
                              QP=_next_pow2(max(q, 2)), num_warps=8)
    return tokens


def spec_accept(tokens, am, hist, hist_len, len_b, pos_b, remaining, out_tok, out_adv, step_idx):
    b, q = tokens.shape
    _spec_accept_kernel[(b,)](tokens, am, hist, hist_len, len_b, pos_b, remaining,
                              out_tok, out_adv, step_idx, hist.shape[1], b,
                              Q=q, QP=_next_pow2(max(q, 2)), num_warps=1)


def rope_attn_verify(qkv, qn, kn, cos, sin, k_cache, v_cache, len_b, start_t, workspace,
                     sm_scale, n_q, nq, eps):
    """QK-norm + rotary + cache writes + verify attention from raw QKV rows
    [B*nq, (n_q + 2*n_kv)*D]; cos/sin are per row. len_b is the length BEFORE
    these nq tokens."""
    rows = qkv.shape[0]
    b = rows // nq
    hkv = k_cache.shape[1]
    d = qn.shape[0]
    g = n_q // hkv
    acc, lsum, mmax, out, splits, chunk, block_n = workspace
    _rope_attn_verify_kernel[(b * hkv, splits)](
        qkv, qn, kn, cos, sin, k_cache, v_cache, len_b, start_t,
        acc, lsum, mmax, out, sm_scale,
        qkv.stride(0), cos.stride(0),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        N_Q=n_q, N_KV=hkv, G=g, NQ=nq, GP=acc.shape[3], D=d, HALF=d // 2, EPS=eps,
        BLOCK_N=block_n, CHUNK=chunk, SPLITS_ONE=(splits == 1),
        num_warps=attn_warps(),
        num_stages=int(os.environ.get("ENGINE_ATTN_STAGES", "3")),
    )
    if splits == 1:
        return out
    _flash_verify_combine_kernel[(b * hkv * nq * g,)](
        acc, lsum, mmax, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lsum.stride(0), lsum.stride(1), lsum.stride(2),
        out.stride(0), out.stride(1),
        N_KV=hkv, G=g, NQ=nq, D=d, SPLITS=_next_pow2(splits), num_warps=4,
    )
    return out
