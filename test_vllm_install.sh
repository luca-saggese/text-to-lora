#!/bin/bash
# Script to test vllm installation on ARM64/CUDA

set -e

echo "=========================================="
echo "Testing vLLM installation on $(uname -m)"
echo "=========================================="

# Check architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Detect if uv is available, otherwise use pip
if command -v uv &> /dev/null; then
    PIP="uv pip"
else
    PIP="pip"
fi
echo "Using: $PIP"

if [ "$ARCH" = "aarch64" ]; then
    echo ""
    echo "Installing build dependencies..."
    $PIP install setuptools_scm wheel packaging ninja cmake
    
    echo ""
    echo "=========================================="
    echo "Option 1: Try vllm 0.6.4.post1 with git install"
    echo "=========================================="
    # git clone --depth 1 --branch v0.6.4.post1 https://github.com/vllm-project/vllm.git /tmp/vllm
    # cd /tmp/vllm
    # VLLM_INSTALL_PUNICA_KERNELS=1 MAX_JOBS=4 uv pip install -e . --no-build-isolation
    
    echo ""
    echo "=========================================="
    echo "Option 2: Try latest vllm from main branch"
    echo "=========================================="
    git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-main
    cd /tmp/vllm-main
    
    echo "Installing dependencies..."
    $PIP install -r requirements-build.txt || true
    $PIP install -r requirements-common.txt || true
    $PIP install -r requirements-cuda.txt || true
    
    echo ""
    echo "Building vllm from source..."
    VLLM_INSTALL_PUNICA_KERNELS=1 MAX_JOBS=2 $PIP install -e . --no-build-isolation
    
    echo ""
    echo "=========================================="
    echo "Testing vllm import..."
    echo "=========================================="
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    
else
    echo "x86_64 architecture - installing prebuilt vllm 0.5.4"
    $PIP install vllm==0.5.4
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

echo ""
echo "=========================================="
echo "Installation successful!"
echo "=========================================="
