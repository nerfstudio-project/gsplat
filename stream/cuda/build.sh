#!/bin/bash

# Build script for stream CUDA extensions
# Usage: ./build.sh [--clean]

set -e

# Check if --clean is passed
if [ "$1" = "--clean" ]; then
    echo "Cleaning previous builds..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -name "*.so" -delete
    find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "Clean completed."
    if [ "$#" -eq 1 ]; then
        exit 0
    fi
fi

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed and in PATH."
    exit 1
fi

echo "Found CUDA version:"
nvcc --version

# Check PyTorch CUDA support
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Build extension
echo "Building stream CUDA extension..."
python3 setup.py build_ext --inplace

echo "Build completed successfully!"
echo ""
echo "To test the extension, run:"
echo "cd .."
echo "python3 -c \"from cuda._wrapper import is_cuda_available; print('CUDA extension available:', is_cuda_available())\""
