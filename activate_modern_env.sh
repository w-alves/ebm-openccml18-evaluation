#!/bin/bash
echo "🔌 Activating Modern ML environment (Python 3.11)..."
eval "$(conda shell.bash hook)"
conda activate ml_modern_py311
echo "✅ Modern ML environment activated!"
echo "💡 To deactivate, run: conda deactivate"
