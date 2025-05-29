#!/bin/bash
echo "🔌 Activating Auto-sklearn environment (Python 3.8)..."
eval "$(conda shell.bash hook)"
conda activate ml_autosklearn_py38
echo "✅ Auto-sklearn environment activated!"
echo "💡 To deactivate, run: conda deactivate"
