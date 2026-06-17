#!/bin/bash
# Exit on error
set -e

echo "=== Creating Python Virtual Environment ==="
python3 -m venv venv

echo "=== Upgrading Pip ==="
venv/bin/pip install --upgrade pip

echo "=== Installing PyTorch, Transformers, NumPy, and ONNX ==="
# Standard install will auto-fetch CUDA-enabled PyTorch on Linux
venv/bin/pip install torch transformers numpy onnx onnxscript

echo "=== Setup Complete! ==="
