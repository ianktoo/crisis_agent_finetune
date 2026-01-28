#!/bin/bash
# Activation script for crisis_agent conda environment
# Usage: source activate_env.sh

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate crisis_agent

# Navigate to project directory
cd /home/jovyan/work/projects/crisis_agent_finetune

# Verify activation
echo "✅ Conda environment activated: $CONDA_DEFAULT_ENV"
echo "📍 Project directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
