#!/bin/bash
# Commands to set up the crisis-agent fine-tuning environment

# 1. Create a new conda environment for this project
conda create -n crisis_agent python=3.10 -y

# 2. Activate the environment
conda activate crisis_agent

# 3. Install dependencies
cd /home/jovyan/work/projects/crisis_agent_finetune
pip install -r requirements.txt

# 4. Verify setup
python scripts/verify_setup.py

# 5. Run tests
pytest tests/ -v

# Or run tests with coverage
# pytest tests/ --cov=src --cov-report=html --cov-report=term
