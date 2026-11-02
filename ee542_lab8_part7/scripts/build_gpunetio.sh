#!/usr/bin/env bash
set -euo pipefail
cd "$HOME"
if [ ! -d gpunetio/.git ]; then git clone https://github.com/NVIDIA-DOCA/gpunetio; fi
cd gpunetio
export PATH=/usr/local/cuda-13.0/bin:$PATH
make install install_examples PREFIX="$HOME/gpunetio/install" CUDA_ARCH=80
ls -lh "$HOME/gpunetio/install/examples"
