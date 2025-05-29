#!/bin/bash

# ML Comparison Study Multi-Environment Setup Script
# This script sets up multiple Conda environments for different model groups due to version conflicts

set -e  # Exit on any error

echo "🚀 Setting up ML Comparison Study Multi-Environment with Conda"
echo "=============================================================="
echo "📋 Environment Plan:"
echo "   🐍 Python 3.8 → Auto-sklearn"
echo "   🐍 Python 3.11 → AutoGluon, XGBoost, LightGBM, CatBoost, EBM"
echo ""

# Check if we're in WSL/Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux/WSL2. Please run in WSL2 environment."
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Environment names
ENV_38="ml_autosklearn_py38"
ENV_311="ml_modern_py311"

# Function to setup environment
setup_environment() {
    local env_name=$1
    local python_version=$2
    local requirements_file=$3
    local description=$4
    
    echo "🔧 Setting up $env_name ($description)..."
    
    # Install system dependencies for Auto-sklearn
    if [[ "$python_version" == "3.8" ]]; then
        echo "🛠️  Installing system dependencies for Auto-sklearn..."
        sudo apt-get update
        sudo apt-get install -y build-essential swig python3-dev
    fi
    
    # Remove existing env if user agrees
    if conda info --envs | grep -q "$env_name"; then
        echo "📁 Conda environment '$env_name' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Removing existing Conda environment..."
            conda remove --name "$env_name" --all -y
        else
            echo "🔄 Using existing environment..."
            return 0
        fi
    fi
    
    # Create conda env
    echo "🐍 Creating Conda environment '$env_name' with Python $python_version..."
    conda create -n "$env_name" python="$python_version" -y
    
    # Activate conda environment
    echo "🔌 Activating Conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    
    # Set up conda-forge channel with strict priority (for all environments)
    echo "🌐 Configuring conda-forge channel..."
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    
    # Install compilers and swig for Python 3.8 (needed for Auto-sklearn)
    if [[ "$python_version" == "3.8" ]]; then
        echo "🔧 Installing compiler tools and swig for Auto-sklearn..."
        conda install -y gxx_linux-64 gcc_linux-64 swig
    fi
    
    # Install requirements
    if [ -f "$requirements_file" ]; then
        echo "📦 Installing packages from $requirements_file..."
        
        # For Python 3.8, install packages individually to avoid conflicts
        if [[ "$python_version" == "3.8" ]]; then
            echo "🚧 Installing packages individually for Python 3.8..."
            pip install --upgrade pip
            pip install -r "$requirements_file"
        else
            # For Python 3.11, try conda first, then pip
            echo "📦 Installing packages via conda/pip..."
            pip install --upgrade pip
            pip install -r "$requirements_file"
        fi
    else
        echo "❌ $requirements_file not found!"
        exit 1
    fi
    
    echo "✅ Environment $env_name setup completed!"
    conda deactivate
}

# Create project directories
echo "📁 Creating project directory structure..."
mkdir -p data/{raw,processed,split}
mkdir -p results/{raw,plots}
mkdir -p models
mkdir -p logs
mkdir -p envs

# Setup Python 3.7 environment for Auto-sklearn
setup_environment "$ENV_38" "3.8" "requirements_python38.txt" "Auto-sklearn"

# Setup Python 3.11 environment for modern models
setup_environment "$ENV_311" "3.11" "requirements_python311.txt" "Modern ML models"

# Create environment activation scripts
echo "📝 Creating environment activation scripts..."

# Script for Python 3.7 environment
cat > activate_autosklearn_env.sh << EOF
#!/bin/bash
echo "🔌 Activating Auto-sklearn environment (Python 3.8)..."
eval "\$(conda shell.bash hook)"
conda activate $ENV_38
echo "✅ Auto-sklearn environment activated!"
echo "💡 To deactivate, run: conda deactivate"
EOF
chmod +x activate_autosklearn_env.sh

# Script for Python 3.11 environment
cat > activate_modern_env.sh << EOF
#!/bin/bash
echo "🔌 Activating Modern ML environment (Python 3.11)..."
eval "\$(conda shell.bash hook)"
conda activate $ENV_311
echo "✅ Modern ML environment activated!"
echo "💡 To deactivate, run: conda deactivate"
EOF
chmod +x activate_modern_env.sh

# Create environment configuration file
echo "📋 Creating environment configuration..."
cat > envs/environment_config.json << EOF
{
  "environments": {
    "autosklearn": {
      "name": "$ENV_38",
      "python_version": "3.8",
      "models": ["AutoSklearn"],
      "activation_script": "./activate_autosklearn_env.sh"
    },
    "modern": {
      "name": "$ENV_311",
      "python_version": "3.11", 
      "models": ["AutoGluon", "XGBoost", "LightGBM", "CatBoost", "EBM"],
      "activation_script": "./activate_modern_env.sh"
    }
  }
}
EOF

# Set up logging
echo "📋 Setting up logging configuration..."
cat > logging.conf << 'EOF'
[loggers]
keys=root

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('logs/ml_comparison.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
EOF

# Verify installations
echo "🔍 Verifying installations..."

echo "🧪 Testing Python 3.8 environment (Auto-sklearn)..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_38"
python -c "
try:
    import autosklearn
    print(f'✅ Auto-sklearn: {autosklearn}')
except ImportError as e:
    print(f'❌ Auto-sklearn failed: {e}')
"
conda deactivate

echo "🧪 Testing Python 3.11 environment (Modern models)..."
conda activate "$ENV_311"
python -c "import sklearn; print(f'✅ scikit-learn')" || echo "❌ scikit-learn failed"
python -c "import lightgbm; print(f'✅ LightGBM')" || echo "❌ LightGBM failed"
python -c "import catboost; print(f'✅ CatBoost')" || echo "❌ CatBoost failed"
python -c "import xgboost; print(f'✅ XGBoost')" || echo "❌ XGBoost failed"
python -c "import interpret; print(f'✅ Interpret (EBM)')" || echo "❌ Interpret failed"
python -c "
try:
    import autogluon
    print(f'✅ AutoGluon: {autogluon}')
except ImportError as e:
    print(f'❌ AutoGluon failed: {e}')
"
conda deactivate

echo ""
echo "🎉 Multi-Environment Setup Completed Successfully!"
echo "================================================="
echo ""
echo "📋 Environment Summary:"
echo "  🐍 $ENV_38 (Python 3.8) → Auto-sklearn"
echo "  🐍 $ENV_311 (Python 3.11) → AutoGluon, XGBoost, LightGBM, CatBoost, EBM"
echo ""
echo "📋 Next steps:"
echo "1. Test environments:"
echo "   source activate_autosklearn_env.sh"
echo "   source activate_modern_env.sh"
echo ""
echo "2. Run experiments (will automatically use correct environments):"
echo "   python run_experiments.py"
echo ""
echo "📁 Project structure:"
echo "   data/          - Dataset storage"
echo "   models/        - Model implementations"
echo "   results/       - Experiment results"
echo "   logs/          - Log files"
echo "   envs/          - Environment configurations"
echo ""
echo "🚀 Ready for multi-environment ML experiments! 🔬✨" 