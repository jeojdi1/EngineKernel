"""Minimal diagnostic engine: plain transformers greedy decode, nothing else."""
import torch
from transformers import AutoModelForCausalLM


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens: int):
        cur = torch.as_tensor(input_ids, dtype=torch.long, device="cuda")
        past = None
        for _ in range(max_new_tokens):
            out = self.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1)
            yield tok.tolist()
            cur = tok.unsqueeze(1)
