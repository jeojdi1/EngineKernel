"""Reference engine: plain transformers greedy decode. The thing we must match."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


class BaselineEngine:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens: int):
        lens = [len(x) for x in input_ids]
        s = max(lens)
        b = len(input_ids)
        ids = torch.zeros((b, s), dtype=torch.long, device=self.device)
        mask = torch.zeros((b, s), dtype=torch.long, device=self.device)
        for i, seq in enumerate(input_ids):
            ids[i, s - lens[i]:] = torch.as_tensor(seq, device=self.device)
            mask[i, s - lens[i]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)

        past = None
        cur, cur_pos, cur_mask = ids, pos, mask
        for _ in range(max_new_tokens):
            out = self.model(input_ids=cur, attention_mask=cur_mask,
                             position_ids=cur_pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1, :].argmax(-1)
            yield tok.tolist()
            cur = tok.unsqueeze(1)
            cur_pos = cur_pos[:, -1:] + 1
            cur_mask = torch.cat([cur_mask, torch.ones((b, 1), dtype=torch.long,
                                                       device=self.device)], dim=1)
