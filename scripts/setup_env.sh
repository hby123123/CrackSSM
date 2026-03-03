#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== CrackSSM env (PyTorch 2.4.1 + CUDA 12.1) ==="
if ! command -v nvcc &>/dev/null; then
    echo "Warning: nvcc not found. Install CUDA 12.1 and add to PATH."
fi

echo "Installing dependencies (PyTorch 2.4.1+cu121 pinned in requirements.txt)..."
pip install -r "$PROJECT_ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121

SELECTIVE_SCAN_DIR="$PROJECT_ROOT/models/encoders/selective_scan"
if [ ! -d "$SELECTIVE_SCAN_DIR" ] || [ ! -f "$SELECTIVE_SCAN_DIR/setup.py" ]; then
    echo "Error: selective_scan not found: $SELECTIVE_SCAN_DIR"
    exit 1
fi
echo "Installing selective_scan CUDA extension..."
cd "$SELECTIVE_SCAN_DIR"
pip install . --no-build-isolation
cd "$PROJECT_ROOT"

echo "Verifying..."
python -c "
import torch
assert torch.__version__.startswith('2.4'), f'PyTorch version: {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA not available'
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| Device:', torch.cuda.get_device_name(0))
"
echo "=== Done ==="
