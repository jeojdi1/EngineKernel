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

from kernels import (add_rms_norm, flash_decode, gemv, qk_norm_rope_kv, rms_norm,
                     silu_mul)


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
        self.zero_slot = torch.zeros(1, dtype=torch.int64, device=device)
        import inspect
        try:
            self._sdpa_gqa = "enable_gqa" in inspect.signature(
                F.scaled_dot_product_attention).parameters
        except (TypeError, ValueError):
            self._sdpa_gqa = torch.__version__ >= "2.5"

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
    def _mlp(self, x, layer, small: bool = False):
        """small=True uses the Triton GEMV, which beats cuBLAS at decode batch
        sizes; prefill's large M belongs on cuBLAS."""
        if small:
            return gemv(silu_mul(gemv(x, layer["gu"])), layer["down"])
        return torch.matmul(silu_mul(torch.matmul(x, layer["gu_t"])), layer["down_t"])

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
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, is_causal=attn_bias is None,
                    scale=self.sm_scale, enable_gqa=True,
                )
            else:
                rep = c.num_heads // c.num_kv_heads
                o = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                    attn_mask=attn_bias, is_causal=attn_bias is None, scale=self.sm_scale,
                )
            o = o.transpose(1, 2).reshape(b * s, c.q_size)

            x = torch.matmul(o, layer["o_t"])
            x, residual = add_rms_norm(x, residual, layer["ln2"], c.rms_eps)
            x = self._mlp(x, layer)

        residual = residual + x
        # final norm belongs here: decode() applies it, so prefill must too or
        # the first token's logits come from an unnormalised hidden state.
        # Slice first — only the last position feeds the lm_head.
        last = residual.view(b, s, c.hidden_size)[:, -1, :]
        return rms_norm(last, self.final_norm, c.rms_eps)

    def decode(self, tokens, positions, k_cache, v_cache, slot_t, len_t, start_t, ws):
        """One decode step. tokens/positions: [B]. Everything stays on device."""
        c = self.cfg
        b = tokens.shape[0]
        cos = self.cos.index_select(0, positions)
        sin = self.sin.index_select(0, positions)

        # slot_t = index this token occupies; len_t = valid length including it.
        # Both are advanced once per step so all 36 layers agree on the slot.
        slot_t.copy_(len_t)
        len_t.add_(1)

        h = F.embedding(tokens, self.embed)
        residual = h
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = rms_norm(residual, layer["ln1"], c.rms_eps)
            else:
                x, residual = add_rms_norm(x, residual, layer["ln1"], c.rms_eps)

            qkv = gemv(x, layer["qkv"])
            qk_norm_rope_kv(qkv, layer["qn"], layer["kn"], cos, sin,
                            k_cache[i], v_cache[i], slot_t,
                            c.num_heads, c.num_kv_heads, c.rms_eps, 1)

            q = qkv[:, : c.q_size].view(b, c.num_heads, c.head_dim)

            if ws is None:
                o = self._decode_attn_ref(q, k_cache[i], v_cache[i], len_t, start_t)
            else:
                o = flash_decode(q, k_cache[i], v_cache[i], len_t, start_t, ws, self.sm_scale)
            x = gemv(o.view(b, c.q_size), layer["o"])
            x, residual = add_rms_norm(x, residual, layer["ln2"], c.rms_eps)
            x = self._mlp(x, layer, small=True)

        residual = residual + x
        return rms_norm(residual, self.final_norm, c.rms_eps)

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
        return torch.argmax(gemv(hidden, self.lm_head), dim=-1)
