#!/usr/bin/env bash
set -euo pipefail

echo "ENVIRONMENT_BEGIN"
date -u '+utc_start=%Y-%m-%dT%H:%M:%SZ'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -V
echo "ENVIRONMENT_END"

# The NGC image bundles torchao 0.11+git. PEFT detects it but requires >=0.16;
# LoRA does not need torchao, so remove the optional incompatible integration.
python -m pip uninstall --yes torchao >/dev/null 2>&1 || true
python -m pip uninstall --yes torchao >/dev/null 2>&1 || true
python -m pip install --quiet --no-cache-dir -r requirements.txt

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m mirror_repro.run \
  --config configs/experiment.json
