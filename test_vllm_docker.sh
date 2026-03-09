#!/bin/bash
# Script to test vllm installation inside Docker container

set -e

echo "=========================================="
echo "Testing vLLM installation in Docker on $(uname -m)"
echo "=========================================="

ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Activate the venv created by uv sync
source /app/.venv/bin/activate

if [ "$ARCH" = "aarch64" ]; then
    echo ""
    echo "Installing build dependencies..."
    pip install setuptools_scm wheel packaging ninja cmake
    
    echo ""
    echo "Cloning vllm from main branch..."
    rm -rf /tmp/vllm-main
    git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-main
    cd /tmp/vllm-main
    
    echo ""
    echo "Installing vllm dependencies..."
    if [ -f requirements-build.txt ]; then
        pip install -r requirements-build.txt
    fi
    if [ -f requirements-common.txt ]; then
        pip install -r requirements-common.txt
    fi
    if [ -f requirements-cuda.txt ]; then
        pip install -r requirements-cuda.txt
    fi
    
    echo ""
    echo "Building vllm from source with CUDA support..."
    echo "This may take 10-15 minutes..."
    VLLM_INSTALL_PUNICA_KERNELS=1 MAX_JOBS=2 pip install -e . --no-build-isolation
    
    echo ""
    echo "=========================================="
    echo "Testing vllm import..."
    echo "=========================================="
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    
else
    echo "x86_64 architecture - installing prebuilt vllm 0.5.4"
    pip install vllm==0.5.4
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

echo ""
echo "=========================================="
echo "Installation successful!"
echo "=========================================="
