#!/bin/bash
# entrypoint for dvlt:jetson — handles weight setup and inference

set -euo pipefail

WEIGHTS="model/weights.dvlt"
SAFETENSORS="model/model.safetensors"
HF_URL="https://huggingface.co/nvidia/dvlt/resolve/main/model.safetensors"

# --setup: download and convert weights into the mounted model/ volume
if [ "${1:-}" = "--setup" ]; then
    if [ -f "$WEIGHTS" ]; then
        echo "weights already present at $WEIGHTS, nothing to do."
        exit 0
    fi
    mkdir -p model
    if [ ! -f "$SAFETENSORS" ]; then
        echo "==> downloading model.safetensors (~468 MB) from Hugging Face..."
        wget -c -O "$SAFETENSORS.part" "$HF_URL" && mv "$SAFETENSORS.part" "$SAFETENSORS" || {
            rm -f "$SAFETENSORS.part"
            echo ""
            echo "ERROR: download failed. The nvidia/dvlt repo may require a Hugging Face login."
            echo "Download model.safetensors manually from:"
            echo "    https://huggingface.co/nvidia/dvlt"
            echo "Place it at  model/model.safetensors  (inside your mounted volume)"
            echo "then re-run: docker run ... dvlt:jetson --setup"
            exit 1
        }
    fi
    echo "==> converting safetensors -> $WEIGHTS (bf16)..."
    ./build/convert
    echo "==> done. You can now run inference."
    exit 0
fi

# normal inference — check weights exist first
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: weights not found at $WEIGHTS"
    echo ""
    echo "Mount a host directory to /dvlt/model and run setup first:"
    echo "  docker run --rm --runtime nvidia -v \$(pwd)/model:/dvlt/model dvlt:jetson --setup"
    exit 1
fi

exec ./build/dvlt "$@"
