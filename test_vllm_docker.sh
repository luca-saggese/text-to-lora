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
    echo "Installing flash-attention from prebuilt wheels for ARM64..."
    # Using CUDA 13.0 compatible version with Python 3.10 from the image you showed
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp314-cp314-linux_aarch64.whl || \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu130torch2.10-cp310-cp310-linux_aarch64.whl || \
    echo "Flash-attention install skipped (will use without flash-attn)"
    
    echo ""
    echo "=========================================="
    echo "Option 1: Try vLLM nightly with CUDA support"
    echo "=========================================="
    # Try nightly builds which may have CUDA wheels for ARM64
    pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index || echo "Nightly build not available"
    
    # Check if vllm was installed
    if python -c "import vllm" 2>/dev/null; then
        echo "vLLM installed successfully!"
        python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    else
        echo ""
        echo "=========================================="
        echo "Option 2: Building vLLM from source with CUDA support"
        echo "=========================================="
        
        # Install TCMalloc (recommended for CPU performance)
        echo "Installing system dependencies..."
        apt-get update && apt-get install -y --no-install-recommends \
            libtcmalloc-minimal4 \
            libnuma-dev \
            gcc-12 g++-12 || echo "Some packages may already be installed"
        
        pip install setuptools_scm wheel packaging ninja cmake
        
        echo "Cloning vllm from main branch..."
        rm -rf /tmp/vllm-source
        git clone https://github.com/vllm-project/vllm.git /tmp/vllm-source
        cd /tmp/vllm-source
        
        echo "Installing vllm build dependencies..."
        pip install -r requirements-cpu-build.txt --torch-backend cpu || pip install -r requirements-build.txt || true
        
        echo ""
        echo "Building vLLM with CUDA support..."
        echo "This may take 15-20 minutes..."
        # Build for CUDA on ARM64
        VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 pip install . --no-build-isolation
        
        echo ""
        echo "=========================================="
        echo "Testing vllm import..."
        echo "=========================================="
        python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    fi
else
    echo "x86_64 architecture - installing prebuilt vllm 0.5.4"
    pip install vllm==0.5.4
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

echo ""
echo "=========================================="
echo "Installation successful!"
echo "=========================================="
