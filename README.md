# EngineKernel

Fast exact greedy decode for Qwen3-4B-Instruct-2507 on a single H100.

    engine/     the submission (this folder, and only this folder, is uploaded)
      engine.py   Engine: cache sizing, CUDA-graph capture, generate loop
      model.py    hand-written Qwen3 forward, fused weights, static KV cache
      kernels.py  Triton kernels, each with a torch fallback for CPU testing
    bench/      local benchmark + correctness harness (never submitted)
    infra/      Lambda Cloud provisioning helpers

## Why it is faster

Batch-1 decode on this model is bound by two things: streaming ~8 GB of weights
per token (~2.4 ms at H100 bandwidth), and the ~1000 kernel launches per token
a stock eager loop issues. The second one dominates, so the design targets it:

* **The whole decode step is one CUDA graph.** Sequence length, position and the
  emitted token live in device tensors that the graph advances itself, so N
  tokens are N graph replays with no host round-trip between them.
* **Fused projections.** q/k/v become one GEMM, gate/up another, done once at
  load time.
* **Split-K flash decoding.** At batch 1 there are only 8 KV heads, so attention
  without a sequence split leaves 124 of 132 SMs idle.
* **Graphs captured in `__init__`**, which the harness does not time, over a
  grid of plausible (batch, length) shapes. Capturing inside `generate()` would
  land on sample 1 only and show up as timing spread.

Output is bit-comparable to the reference by construction: same BF16 weights,
same op order, fp32 reductions where the reference uses them. No quantization,
no approximation.

## Local loop

    python bench/test_equivalence.py                  # CPU, seconds, no GPU needed
    python bench/bench.py --model ~/qwen3-4b --check  # H100: speed + exactness

`test_equivalence.py` builds a tiny random Qwen3 and checks our tokens against
`transformers` greedy decode, including ragged/left-padded batches. Run it
before every push; a mismatch on the real benchmark costs a submission.
