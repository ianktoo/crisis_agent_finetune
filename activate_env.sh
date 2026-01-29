#!/bin/bash
# Activation script for crisis_agent conda environment
# Usage: source activate_env.sh
#
# This script will:
# - Create the conda environment if it doesn't exist
# - Install dependencies from requirements.txt
# - Activate the environment
# - Navigate to the project directory

PROJECT_DIR="/home/jovyan/work/projects/crisis_agent_finetune"
ENV_NAME="crisis_agent"
PYTHON_VERSION="3.10"

# Initialize conda
eval "$(conda shell.bash hook)"

# Check if the environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "🔧 Conda environment '${ENV_NAME}' not found. Creating it..."
    
    # Create the environment
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create conda environment"
        return 1
    fi
    
    echo "✅ Conda environment created successfully"
    
    # Activate and install dependencies
    conda activate ${ENV_NAME}
    
    echo "📦 Installing dependencies from requirements.txt..."
    cd ${PROJECT_DIR}
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Some dependencies may have failed to install"
    else
        echo "✅ Dependencies installed successfully"
    fi
else
    echo "✅ Conda environment '${ENV_NAME}' found"
    conda activate ${ENV_NAME}
fi

# Navigate to project directory
cd ${PROJECT_DIR}

# Verify activation
echo ""
echo "=========================================="
echo "✅ Environment Ready!"
echo "=========================================="
echo "📍 Project directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Conda env: $CONDA_DEFAULT_ENV"
echo "=========================================="
