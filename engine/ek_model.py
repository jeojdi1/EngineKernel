"""Qwen3 forward pass, written for one purpose: fast greedy decode on an H100.

Differences from the reference implementation, all output-preserving:
  * q/k/v and gate/up are fused into single GEMMs at load time
  * the KV cache is one preallocated tensor per layer, [B, H_kv, S_max, D]
  * decode never touches the host: sequence length, positions and the emitted
    token all live in device tensors so the whole step can be captured in a
    CUDA graph
"""

from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F

import ek_kernels
from ek_kernels import (add_rms_norm, add_rms_norm_parts, attn_torch, gemv_parts, gemv_swiglu, heads_to_rows,
                        rope_attn_decode, rope_attn_verify, flash_decode, flash_verify, gemv,
                        norm_gemv, qk_norm_rope_kv, rms_norm, silu_mul)


class Qwen3Config:
    def __init__(self, path: str):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        self.hidden_size = cfg["hidden_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = cfg["intermediate_size"]
        self.vocab_size = cfg["vocab_size"]
        self.rms_eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 10000.0)
        self.tie_embeddings = cfg.get("tie_word_embeddings", False)
        self.max_position = cfg.get("max_position_embeddings", 32768)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim


def _shard_map(path: str):
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    from safetensors import safe_open

    out = {}
    for fn in files:
        with safe_open(fn, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = os.path.basename(fn)
    return out


class Qwen3(torch.nn.Module):
    def __init__(self, model_path: str, device: torch.device, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = Qwen3Config(model_path)
        self.device = device
        self.dtype = dtype
        self._load(model_path)
        self._build_rope(self.cfg.max_position if self.cfg.max_position <= 16384 else 16384)
        self.sm_scale = self.cfg.head_dim ** -0.5
        self.zero_slot = torch.zeros(1024, dtype=torch.int64, device=device)
        self.arange_q = torch.arange(64, dtype=torch.int64, device=device)
        self._gemv_choice = {}
        self.split_k = os.environ.get("ENGINE_SPLIT", "1") == "1"
        self.fuse_swiglu = os.environ.get("ENGINE_SWIGLU", "0") == "1"
        self.fuse_rope_verify = (os.environ.get("ENGINE_ROPE_VERIFY", "0") == "1"
                                 and ek_kernels.has_triton())
        self.fuse_rope_attn = (os.environ.get("ENGINE_ROPE_ATTN", "1") == "1"
                               and ek_kernels.has_triton())
        self.fused = os.environ.get("ENGINE_FUSED", "0") == "1" and ek_kernels.has_triton()
        self.use_gemv = (os.environ.get("ENGINE_GEMV") == "1" if "ENGINE_GEMV" in os.environ
                         else self._pick_projection_path())
        if not self.use_gemv and device.type == "cuda" and os.environ.get("ENGINE_CONTIG", "0") == "1":
            self._contiguous_weights()
        import inspect
        try:
            self._sdpa_gqa = "enable_gqa" in inspect.signature(
                F.scaled_dot_product_attention).parameters
        except (TypeError, ValueError):
            self._sdpa_gqa = torch.__version__ >= "2.5"
        # cuDNN's fused attention runs the prefill at ~2x the FlashAttention-2
        # kernel torch picks by default on an H100 (4 x 2048: 0.50 -> 0.26 ms per
        # layer, ~530 TFLOPS). torch 2.5 ranks it below flash, so it has to be
        # selected on its own; padded batches (an explicit mask) and any build
        # that cannot run it fall back to the default choice.
        self._sdpa_cudnn = None
        if (self._sdpa_gqa and device.type == "cuda"
                and os.environ.get("ENGINE_CUDNN_SDPA", "1") == "1"):
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                self._sdpa_cudnn = (sdpa_kernel, SDPBackend.CUDNN_ATTENTION)
            except Exception:
                self._sdpa_cudnn = None

    # ------------------------------------------------------------------
    def _load(self, model_path: str):
        from safetensors import safe_open

        wmap = _shard_map(model_path)
        dev = "cuda" if self.device.type == "cuda" else "cpu"
        handles, opened = {}, {}

        def get(name):
            fn = wmap[name]
            if fn not in opened:
                opened[fn] = safe_open(os.path.join(model_path, fn), framework="pt", device=dev)
                handles[fn] = opened[fn].__enter__()
            return handles[fn].get_tensor(name).to(self.dtype)

        c = self.cfg
        self.embed = get("model.embed_tokens.weight")
        self.final_norm = get("model.norm.weight")
        self.lm_head = self.embed if c.tie_embeddings else get("lm_head.weight")
        self.lm_head_t = self.lm_head.t()

        self.layers = []
        for i in range(c.num_layers):
            p = f"model.layers.{i}."
            qkv = torch.cat(
                [get(p + "self_attn.q_proj.weight"),
                 get(p + "self_attn.k_proj.weight"),
                 get(p + "self_attn.v_proj.weight")], dim=0,
            ).contiguous()
            gu = torch.cat(
                [get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")], dim=0
            ).contiguous()
            o_w = get(p + "self_attn.o_proj.weight")
            down_w = get(p + "mlp.down_proj.weight")
            self.layers.append({
                "qkv": qkv, "qkv_t": qkv.t(),
                "o_t": o_w.t(), "gu_t": gu.t(), "down_t": down_w.t(),
                "o": o_w,
                "gu": gu,
                "down": down_w,
                "ln1": get(p + "input_layernorm.weight"),
                "ln2": get(p + "post_attention_layernorm.weight"),
                "qn": get(p + "self_attn.q_norm.weight"),
                "kn": get(p + "self_attn.k_norm.weight"),
            })
        for fn, h in opened.items():
            h.__exit__(None, None, None)

    def _build_rope(self, max_len: int):
        c = self.cfg
        inv = 1.0 / (c.rope_theta ** (
            torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=self.device) / c.head_dim
        ))
        t = torch.arange(max_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(self.dtype).contiguous()
        self.sin = emb.sin().to(self.dtype).contiguous()
        self.rope_len = max_len

    def ensure_rope(self, max_len: int):
        if max_len > self.rope_len:
            self._build_rope(max(max_len, self.rope_len * 2))

    # ------------------------------------------------------------------
    def _contiguous_weights(self):
        """Store projections as contiguous [in, out].

        matmul(x, W.t()) hands cuBLAS a transposed operand; a contiguous [in, out]
        matrix takes its non-transposed kernel, ~2% faster over the decode chain
        on an H100. The [out, in] originals are only needed by the Triton GEMV,
        so they are dropped here and memory use is unchanged (the tied lm_head
        keeps its own copy, since the embedding lookup needs [vocab, hidden]).
        """
        for layer in self.layers:
            for key in ("qkv", "o", "gu", "down"):
                layer[key + "_t"] = layer[key].t().contiguous()
                layer[key] = None
        self.lm_head_t = self.lm_head.t().contiguous()
        torch.cuda.empty_cache()

    def _proj(self, x, layer, key):
        """A decode-shaped projection, via whichever path won on this device."""
        # gemv pads its row dimension to a power of two >= 16; keep it below 64,
        # where Triton 3.1.0 aborts the compiler on Hopper
        if self.use_gemv_for(x.shape[0]):
            return gemv(x, layer[key])
        return torch.matmul(x, layer[key + "_t"])

    def _mlp(self, x, layer, small: bool = False):
        """small=True is the decode path; prefill's large M always wants cuBLAS."""
        if small:
            return self._proj(silu_mul(self._proj(x, layer, "gu")), layer, "down")
        return torch.matmul(silu_mul(torch.matmul(x, layer["gu_t"])), layer["down_t"])

    def _pick_projection_path(self) -> bool:
        return False  # decided per batch size at first use, see use_gemv_for()

    def use_gemv_for(self, rows: int) -> bool:
        """Triton GEMV or cuBLAS for decode projections with this many rows?

        Timed back to back on this GPU at this row count, once, during the
        untimed warmup: the winner depends on architecture, Triton version and
        batch (on an H100 the Triton kernel wins by ~10% at 1 row and loses at
        32), and a box that is heat-throttling shifts absolute numbers but not
        a back-to-back comparison. Ties go to cuBLAS.
        """
        if rows in self._gemv_choice:
            return self._gemv_choice[rows]
        choice = False
        forced = os.environ.get("ENGINE_GEMV")
        if forced is not None:
            choice = forced == "1" and rows <= 32
        elif ek_kernels.has_triton() and self.device.type == "cuda" and rows <= 32 \
                and self.layers[0]["qkv"] is not None:
            import time
            c, L = self.cfg, self.layers
            x = torch.zeros(rows, c.hidden_size, device=self.device, dtype=self.dtype)
            o = torch.zeros(rows, c.q_size, device=self.device, dtype=self.dtype)
            a = torch.zeros(rows, c.intermediate_size, device=self.device, dtype=self.dtype)

            def run(tri):
                for l in L:
                    if tri:
                        gemv(x, l["qkv"]); gemv(o, l["o"]); gemv(x, l["gu"]); gemv(a, l["down"])
                    else:
                        torch.matmul(x, l["qkv_t"]); torch.matmul(o, l["o_t"])
                        torch.matmul(x, l["gu_t"]); torch.matmul(a, l["down_t"])

            def timed(tri):
                g = torch.cuda.CUDAGraph()
                run(tri); torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    run(tri)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(8):
                    g.replay()
                torch.cuda.synchronize()
                return time.perf_counter() - t0

            try:
                t = {False: 0.0, True: 0.0}
                for _ in range(2):          # interleaved so drift hits both alike
                    for tri in (False, True):
                        t[tri] += timed(tri)
                choice = t[True] < t[False] * 0.98
            except Exception:
                choice = False
        self._gemv_choice[rows] = choice
        return choice

    def _sdpa(self, q, k, v, attn_bias):
        kw = dict(attn_mask=attn_bias, is_causal=attn_bias is None, scale=self.sm_scale,
                  enable_gqa=True)
        if self._sdpa_cudnn is not None and attn_bias is None:
            sdpa_kernel, backend = self._sdpa_cudnn
            try:
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q, k, v, **kw)
            except Exception:
                self._sdpa_cudnn = None   # this build cannot run it: stay on the default
        return F.scaled_dot_product_attention(q, k, v, **kw)

    def prefill(self, input_ids, positions, k_cache, v_cache, attn_bias=None):
        """input_ids/positions: [B, S]. Writes slots [0, S) of the caches.

        Returns the hidden state of the final position of each row, [B, H].
        """
        c = self.cfg
        b, s = input_ids.shape
        cos = self.cos.index_select(0, positions.reshape(-1))
        sin = self.sin.index_select(0, positions.reshape(-1))

        h = F.embedding(input_ids, self.embed).reshape(b * s, c.hidden_size)
        residual = h
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = rms_norm(residual, layer["ln1"], c.rms_eps)
            else:
                x, residual = add_rms_norm(x, residual, layer["ln1"], c.rms_eps)

            qkv = torch.matmul(x, layer["qkv_t"])
            qk_norm_rope_kv(qkv, layer["qn"], layer["kn"], cos, sin,
                            k_cache[i], v_cache[i], self.zero_slot,
                            c.num_heads, c.num_kv_heads, c.rms_eps, s)

            q = qkv[:, : c.q_size].view(b, s, c.num_heads, c.head_dim).transpose(1, 2)
            k = k_cache[i][:, :, :s]
            v = v_cache[i][:, :, :s]

            if self._sdpa_gqa:
                o = self._sdpa(q, k, v, attn_bias)
            else:
                rep = c.num_heads // c.num_kv_heads
                o = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                    attn_mask=attn_bias, is_causal=attn_bias is None, scale=self.sm_scale,
                )
            o = heads_to_rows(o)

            x = torch.matmul(o, layer["o_t"])
            x, residual = add_rms_norm(x, residual, layer["ln2"], c.rms_eps)
            x = self._mlp(x, layer)

        residual = residual + x
        # final norm belongs here: decode() applies it, so prefill must too or
        # the first token's logits come from an unnormalised hidden state.
        # Slice first — only the last position feeds the lm_head.
        last = residual.view(b, s, c.hidden_size)[:, -1, :]
        return rms_norm(last, self.final_norm, c.rms_eps)

    def decode_fused(self, tokens, positions, k_cache, v_cache, slot_t, len_t, start_t, ws):
        """decode() with the norms and SwiGLU folded into Triton projections:
        6-7 dependent kernels per layer instead of 9-10."""
        c = self.cfg
        b = tokens.shape[0]
        cos = self.cos.index_select(0, positions)
        sin = self.sin.index_select(0, positions)
        slot_t.copy_(len_t)
        len_t.add_(1)
        x = F.embedding(tokens, self.embed)
        residual = None
        for i, layer in enumerate(self.layers):
            qkv, residual = norm_gemv(x, residual, layer["ln1"], layer["qkv"], c.rms_eps)
            qk_norm_rope_kv(qkv, layer["qn"], layer["kn"], cos, sin,
                            k_cache[i], v_cache[i], slot_t,
                            c.num_heads, c.num_kv_heads, c.rms_eps, 1)
            q = qkv[:, : c.q_size].view(b, c.num_heads, c.head_dim)
            o = flash_decode(q, k_cache[i], v_cache[i], len_t, start_t, ws, self.sm_scale)
            x = torch.matmul(o.view(b, c.q_size), layer["o_t"])
            act, residual = norm_gemv(x, residual, layer["ln2"], layer["gu"], c.rms_eps, silu=True)
            x = torch.matmul(act, layer["down_t"])
        x, residual = add_rms_norm(x, residual, self.final_norm, c.rms_eps)
        return x

    def decode(self, tokens, positions, k_cache, v_cache, slot_t, len_t, start_t, ws):
        """One decode step. tokens/positions: [B]. Everything stays on device."""
        c = self.cfg
        b = tokens.shape[0]
        if self.fused and ws is not None and b <= 32:
            return self.decode_fused(tokens, positions, k_cache, v_cache, slot_t, len_t, start_t, ws)
        cos = self.cos.index_select(0, positions)
        sin = self.sin.index_select(0, positions)
        # split-K for the two narrow projections: their partial sums are folded
        # into the add+norm that consumes them, so it needs the Triton GEMV path
        parts = (self.split_k and ws is not None and self.use_gemv_for(b)
                 and b <= int(os.environ.get("ENGINE_SPLIT_MAXB", "32")))

        # slot_t = index this token occupies; len_t = valid length including it.
        # Both are advanced once per step so all 36 layers agree on the slot.
        slot_t.copy_(len_t)
        len_t.add_(1)

        residual = F.embedding(tokens, self.embed)
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = rms_norm(residual, layer["ln1"], c.rms_eps)
            elif parts:
                x, residual = add_rms_norm_parts(x, residual, layer["ln1"], c.rms_eps)
            else:
                x, residual = add_rms_norm(x, residual, layer["ln1"], c.rms_eps)

            qkv = self._proj(x, layer, "qkv")
            if self.fuse_rope_attn and ws is not None:
                o = rope_attn_decode(qkv, layer["qn"], layer["kn"], cos, sin,
                                     k_cache[i], v_cache[i], len_t, start_t, ws,
                                     self.sm_scale, c.num_heads, c.rms_eps)
            else:
                qk_norm_rope_kv(qkv, layer["qn"], layer["kn"], cos, sin,
                                k_cache[i], v_cache[i], slot_t,
                                c.num_heads, c.num_kv_heads, c.rms_eps, 1)
                q = qkv[:, : c.q_size].view(b, c.num_heads, c.head_dim)
                if ws is None:
                    o = self._decode_attn_ref(q, k_cache[i], v_cache[i], len_t, start_t)
                else:
                    o = flash_decode(q, k_cache[i], v_cache[i], len_t, start_t, ws, self.sm_scale)
            o = o.view(b, c.q_size)
            if parts:
                x, residual = add_rms_norm_parts(gemv_parts(o, layer["o"]), residual,
                                                 layer["ln2"], c.rms_eps)
                act = gemv_swiglu(x, layer["gu"]) if self.fuse_swiglu else silu_mul(self._proj(x, layer, "gu"))
                x = gemv_parts(act, layer["down"])
            else:
                x = self._proj(o, layer, "o")
                x, residual = add_rms_norm(x, residual, layer["ln2"], c.rms_eps)
                x = self._mlp(x, layer, small=True)

        if parts:
            return add_rms_norm_parts(x, residual, self.final_norm, c.rms_eps)[0]
        residual = residual + x
        return rms_norm(residual, self.final_norm, c.rms_eps)

    def verify(self, tokens, pos_b, k_cache, v_cache, len_b, start_t, ws):
        """Score NQ tokens per sequence in one pass (speculative verify).

        tokens: [B, NQ] = the last emitted token followed by NQ-1 drafts.
        Returns the greedy token after each position, [B, NQ]. K/V for all NQ
        inputs is written at slots len_b + j; whatever follows a rejected draft
        is simply overwritten by the next step.
        """
        c = self.cfg
        b, nq = tokens.shape
        positions = (pos_b[:, None] + self.arange_q[None, :nq]).reshape(-1)
        positions = positions.clamp(max=self.rope_len - 1)
        cos = self.cos.index_select(0, positions)
        sin = self.sin.index_select(0, positions)

        rows = b * nq
        tri = ws[0] != "torch" and self.use_gemv_for(rows)      # tuned Triton projections (<= 32 rows)
        parts = tri and self.split_k
        residual = F.embedding(tokens.reshape(-1), self.embed)
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = rms_norm(residual, layer["ln1"], c.rms_eps)
            elif parts:
                x, residual = add_rms_norm_parts(x, residual, layer["ln1"], c.rms_eps)
            else:
                x, residual = add_rms_norm(x, residual, layer["ln1"], c.rms_eps)
            qkv = self._proj(x, layer, "qkv")
            if self.fuse_rope_verify and ws[0] != "torch":
                o = rope_attn_verify(qkv, layer["qn"], layer["kn"], cos, sin,
                                     k_cache[i], v_cache[i], len_b, start_t, ws,
                                     self.sm_scale, c.num_heads, nq, c.rms_eps)
            else:
                qk_norm_rope_kv(qkv, layer["qn"], layer["kn"], cos, sin,
                                k_cache[i], v_cache[i], len_b,
                                c.num_heads, c.num_kv_heads, c.rms_eps, nq)
                q = qkv[:, : c.q_size].view(rows, c.num_heads, c.head_dim)
                if ws[0] == "torch":
                    o = attn_torch(q, k_cache[i], v_cache[i], len_b, start_t, self.sm_scale, nq, ws[1])
                else:
                    o = flash_verify(q, k_cache[i], v_cache[i], len_b, start_t, ws, self.sm_scale, nq)
            o = o.view(rows, c.q_size)
            if parts:
                x, residual = add_rms_norm_parts(gemv_parts(o, layer["o"]), residual,
                                                 layer["ln2"], c.rms_eps)
                x = gemv_parts(silu_mul(self._proj(x, layer, "gu")), layer["down"])
            else:
                x = self._proj(o, layer, "o")
                x, residual = add_rms_norm(x, residual, layer["ln2"], c.rms_eps)
                x = self._mlp(x, layer, small=True)
        if parts:
            hidden = add_rms_norm_parts(x, residual, self.final_norm, c.rms_eps)[0]
        else:
            residual = residual + x
            hidden = rms_norm(residual, self.final_norm, c.rms_eps)
        return torch.argmax(torch.matmul(hidden, self.lm_head_t), dim=-1).view(b, nq)

    def _decode_attn_ref(self, q, kc, vc, len_t, start_t):
        """Torch reference for decode attention, used on CPU and in tests."""
        c = self.cfg
        n = int(len_t.item())
        k = kc[:, :, :n]
        v = vc[:, :, :n]
        rep = c.num_heads // c.num_kv_heads
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        scores = torch.einsum("bhd,bhsd->bhs", q.float(), k.float()) * self.sm_scale
        idx = torch.arange(n, device=q.device)
        scores = scores.masked_fill(idx[None, None, :] < start_t[:, None, None], float("-inf"))
        p = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.einsum("bhs,bhsd->bhd", p.float(), v.float()).to(q.dtype)

    def argmax_token(self, hidden):
        # hidden is [B, H] in both prefill and decode, so always the small path
        logits = torch.matmul(hidden, self.lm_head_t)  # cuBLAS already streams this at the ceiling
        return torch.argmax(logits, dim=-1)
