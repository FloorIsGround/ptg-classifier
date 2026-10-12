#!/usr/bin/env bash
# Public model only. Use inside an approved Vast.ai vLLM container.
# Bind to loopback; the workstation connects through SSH forwarding.
set -euo pipefail
exec vllm serve ibm-granite/granite-4.2-8b \
  --revision f8de16cdcdbc6c779ca517604e050d82cc119e44 \
  --served-model-name granite-4.2-8b \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --no-enable-log-requests \
  --host 127.0.0.1 \
  --port 8000
