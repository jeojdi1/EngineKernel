#!/usr/bin/env bash
# Run ON the GPU box. Targets the benchmark runtime; falls back where the
# host architecture (GH200 is aarch64) has no wheel for the exact pin.
set -euo pipefail
ARCH="$(uname -m)"
echo "host arch: $ARCH"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

python3 -m pip install -q --upgrade pip
if [ "$ARCH" = "x86_64" ]; then
  python3 -m pip install -q torch==2.5.1 triton==3.1.0 \
    --index-url https://download.pytorch.org/whl/cu124
else
  # aarch64: take whatever torch+cu124 build exists; triton ships with it
  python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 || \
    python3 -m pip install -q torch
fi
python3 -m pip install -q transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 \
  'huggingface_hub[cli]'

python3 - <<'PY'
import torch
print("torch", torch.__version__)
try:
    import triton; print("triton", triton.__version__)
except Exception as e:
    print("triton MISSING:", e)
import transformers; print("transformers", transformers.__version__)
p = torch.cuda.get_device_properties(0)
print(f"gpu {p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs  "
      f"{p.total_memory/(1<<30):.0f} GiB")
PY
