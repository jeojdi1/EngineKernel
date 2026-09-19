"""Per-kernel checks against torch references. Isolates a failure to one kernel."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

import kernels as K  # noqa: E402

DEV = "cuda"
FAIL = 0


def check(name, got, want, tol=2e-2):
    """Max error as a fraction of the reference tensor's dynamic range.

    Per-element relative error is meaningless for kernels whose output
    cancels toward zero; bf16 carries 8 mantissa bits, so one ULP at full
    scale is ~0.4% and this tolerance allows a handful of them.
    """
    global FAIL
    got, want = got.float(), want.float()
    err = (got - want).abs().max().item()
    scale = max(want.abs().max().item(), 1e-6)
    ok = err / scale <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:38s} max_abs={err:.4g} "
          f"rel_to_scale={err / scale:.2e}")
    if not ok:
        FAIL += 1


def t_rms_norm():
    print("rms_norm / add_rms_norm")
    for m, n in ((1, 2560), (16, 2560), (8192, 2560)):
        x = torch.randn(m, n, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(n, device=DEV, dtype=torch.bfloat16)
        check(f"rms_norm m={m}", K.rms_norm(x, w, 1e-6), K._torch_rms_norm(x, w, 1e-6))

        r = torch.randn(m, n, device=DEV, dtype=torch.bfloat16)
        r_ref = r.clone()
        y = K.add_rms_norm(x, r, w, 1e-6)[0]
        r_want = (r_ref + x).to(torch.bfloat16)
        check(f"add_rms_norm out m={m}", y, K._torch_rms_norm(r_want, w, 1e-6))
        check(f"add_rms_norm residual m={m}", r, r_want)


def t_silu_mul():
    print("silu_mul")
    for m, n in ((1, 9728), (16, 9728), (8192, 9728)):
        gu = torch.randn(m, 2 * n, device=DEV, dtype=torch.bfloat16)
        g, u = gu.chunk(2, dim=-1)
        check(f"silu_mul m={m}", K.silu_mul(gu), torch.nn.functional.silu(g) * u)


def t_qk_norm_rope():
    print("qk_norm_rope_kv (vs fp32 gold, incl. cache writes)")
    nq, nkv, d = 32, 8, 128
    for b, mpb in ((1, 1), (4, 1), (2, 16)):
        m = b * mpb
        smax = 64
        qkv = torch.randn(m, (nq + 2 * nkv) * d, device=DEV, dtype=torch.bfloat16)
        qn = torch.randn(d, device=DEV, dtype=torch.bfloat16)
        kn = torch.randn(d, device=DEV, dtype=torch.bfloat16)
        cos = torch.randn(m, d, device=DEV, dtype=torch.bfloat16)
        sin = torch.randn(m, d, device=DEV, dtype=torch.bfloat16)
        kc = torch.zeros(b, nkv, smax, d, device=DEV, dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        base = 7 if mpb == 1 else 0
        slot = torch.tensor([base], dtype=torch.int64, device=DEV)

        x = qkv.float()
        gold = x.clone()
        for off, n, w in ((0, nq, qn), (nq * d, nkv, kn)):
            t = x[:, off:off + n * d].reshape(m, n, d)
            rst = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
            tn = (t * rst).to(torch.bfloat16).float() * w.float()
            rot = torch.cat((-tn[..., d // 2:], tn[..., :d // 2]), -1)
            gold[:, off:off + n * d] = (tn * cos.float().unsqueeze(1)
                                        + rot * sin.float().unsqueeze(1)).reshape(m, n * d)

        got = qkv.clone()
        K.qk_norm_rope_kv(got, qn, kn, cos, sin, kc, vc, slot, nq, nkv, 1e-6, mpb)
        tag = f"b={b} mpb={mpb}"
        check(f"q in place {tag}", got[:, :nq * d], gold[:, :nq * d])

        kexp = gold[:, nq * d:(nq + nkv) * d].reshape(b, mpb, nkv, d).transpose(1, 2)
        vexp = qkv[:, (nq + nkv) * d:].reshape(b, mpb, nkv, d).transpose(1, 2)
        check(f"k -> cache {tag}", kc[:, :, base:base + mpb], kexp)
        check(f"v -> cache {tag}", vc[:, :, base:base + mpb], vexp.float())
        untouched = torch.cat([kc[:, :, :base], kc[:, :, base + mpb:]], dim=2)
        check(f"cache elsewhere untouched {tag}", untouched, torch.zeros_like(untouched))


def ref_attn(q, kc, vc, seq_len, start):
    b, hq, d = q.shape
    hkv = kc.shape[1]
    rep = hq // hkv
    k = kc[:, :, :seq_len].repeat_interleave(rep, 1).float()
    v = vc[:, :, :seq_len].repeat_interleave(rep, 1).float()
    s = torch.einsum("bhd,bhsd->bhs", q.float(), k) * (d ** -0.5)
    idx = torch.arange(seq_len, device=q.device)
    s = s.masked_fill(idx[None, None, :] < start[:, None, None], float("-inf"))
    return torch.einsum("bhs,bhsd->bhd", torch.softmax(s, -1), v)


def t_flash_decode():
    print("flash_decode (split-K)")
    hq, hkv, d = 32, 8, 128
    for b, smax, slen, pad in ((1, 1024, 600, 0), (4, 2048, 2048, 0),
                               (16, 1024, 640, 0), (3, 1024, 97, 0), (2, 1024, 500, 37)):
        q = torch.randn(b, hq, d, device=DEV, dtype=torch.bfloat16)
        kc = torch.randn(b, hkv, smax, d, device=DEV, dtype=torch.bfloat16)
        vc = torch.randn(b, hkv, smax, d, device=DEV, dtype=torch.bfloat16)
        len_t = torch.tensor([slen], dtype=torch.int64, device=DEV)
        start_t = torch.full((b,), pad, dtype=torch.int32, device=DEV)

        splits, chunk, bn = K.plan_splits(b, hkv, smax)
        sp = K._next_pow2(splits)
        gp = K.group_pad(hq, hkv)
        ws = (torch.zeros(b, hkv, sp, gp, d, dtype=torch.float32, device=DEV),
              torch.zeros(b, hkv, sp, gp, dtype=torch.float32, device=DEV),
              torch.full((b, hkv, sp, gp), -1e30, dtype=torch.float32, device=DEV),
              torch.empty(b, hq, d, dtype=torch.bfloat16, device=DEV),
              splits, chunk, bn)
        got = K.flash_decode(q, kc, vc, len_t, start_t, ws, d ** -0.5)
        want = ref_attn(q, kc, vc, slen, start_t)
        check(f"flash_decode b={b} len={slen} pad={pad} splits={splits}", got, want, 3e-2)


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    t_rms_norm()
    t_silu_mul()
    t_qk_norm_rope()
    t_flash_decode()
    print(f"\nFAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)
