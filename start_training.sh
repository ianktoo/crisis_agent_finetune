#!/bin/bash
# Quick start training script
# Usage: ./start_training.sh [--model-name MODEL_NAME]

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate crisis_agent

# Navigate to project directory
cd /home/jovyan/work/projects/crisis_agent_finetune

# Start training with any provided arguments
python scripts/train.py "$@"
