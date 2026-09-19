"""Run as a child process: can Triton compile and launch a kernel here?

Kept out of the engine process on purpose. Triton builds a C launcher and
loads libcuda on first use; if that segfaults or hangs in a sandbox, a
try/except in the engine cannot save it, but a child's exit code can.
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _bump(X, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(X + offs, tl.load(X + offs) + 1.0)


def main() -> int:
    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    _bump[(1,)](x, N=64)
    torch.cuda.synchronize()
    return 0 if float(x.sum().item()) == 64.0 else 3


if __name__ == "__main__":
    sys.exit(main())
