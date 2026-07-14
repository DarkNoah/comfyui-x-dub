#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PLUGIN_DIR"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
[[ -n "$UV_BIN" ]] || { echo "uv is required" >&2; exit 1; }
"$UV_BIN" venv --python 3.10 .venv
"$UV_BIN" pip install --python .venv/bin/python -r runtime-requirements.txt
PYTHONPATH="$PLUGIN_DIR/xdub_runtime/source" .venv/bin/python -c 'import torch, onnxruntime, diffsynth; print("X-Dub runtime ready:", torch.__version__, "CUDA", torch.cuda.is_available(), "ONNX providers", onnxruntime.get_available_providers())'
