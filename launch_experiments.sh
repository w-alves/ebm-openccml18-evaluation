#!/bin/bash

# ML Comparison Study - Multi-Environment Experiment Launcher
# This script launches the multi-environment experiment coordinator

set -e

echo "🚀 ML Comparison Study - Multi-Environment Experiment Launcher"
echo "=============================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment config exists
if [ ! -f "envs/environment_config.json" ]; then
    echo "❌ Environment configuration not found!"
    echo "Please run ./setup_multi_environment.sh first to create the environments."
    exit 1
fi

# Check if required environments exist
ENV_38="ml_autosklearn_py38"
ENV_311="ml_modern_py311"

if ! conda info --envs | grep -q "$ENV_38"; then
    echo "❌ Environment $ENV_38 not found!"
    echo "Please run ./setup_multi_environment.sh first to create the environments."
    exit 1
fi

if ! conda info --envs | grep -q "$ENV_311"; then
    echo "❌ Environment $ENV_311 not found!"
    echo "Please run ./setup_multi_environment.sh first to create the environments."
    exit 1
fi

# Check if datasets are available
if [ ! -d "data/split" ] || [ ! -f "data/split/datasets_summary.csv" ]; then
    echo "❌ Datasets not found!"
    echo "Please run the data preparation pipeline first:"
    echo "  1. python fetch_datasets.py"
    echo "  2. python preprocess.py"
    echo "  3. python split_data.py"
    exit 1
fi

echo "✅ All prerequisites checked"
echo ""

# Activate modern environment to run the coordinator
echo "🔌 Activating modern environment to run experiment coordinator..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_311"

echo "📊 Starting multi-environment experiment execution..."
echo "This will coordinate training across multiple conda environments:"
echo "  🐍 $ENV_38 (Python 3.8) → Auto-sklearn"
echo "  🐍 $ENV_311 (Python 3.11) → AutoGluon, XGBoost, LightGBM, CatBoost, EBM"
echo ""

# Run the multi-environment experiment coordinator
python run_experiments_multienv.py

# Deactivate environment
conda deactivate

echo ""
echo "🎉 Multi-environment experiment execution completed!"
echo "Check the logs and results directories for detailed output." 