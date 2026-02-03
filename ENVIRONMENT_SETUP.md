# Environment Setup Guide

## Quick Activation

### Method 1: Use the activation script (Recommended)

```bash
# Source the activation script
source activate_env.sh
```

This will:
- Initialize conda
- Activate the `crisis_agent` environment
- Navigate to the project directory
- Show confirmation

### Method 2: Manual activation

```bash
# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate crisis_agent

# Navigate to project (from repo root)
cd /path/to/crisis_agent_finetune
```

## Quick Start Training

Use the training script:

```bash
# Basic training
./start_training.sh

# With custom model name
./start_training.sh --model-name "crisis_agent_v2"

# With other options
./start_training.sh --output-dir "outputs/my_experiment"
```

## Make Activation Persistent

To avoid having to activate every time, add to your `~/.bashrc`:

```bash
# Add conda initialization
echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
echo 'conda activate crisis_agent' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

Or create an alias:

```bash
# Add to ~/.bashrc
echo 'alias crisis-env="source /path/to/crisis_agent_finetune/activate_env.sh"' >> ~/.bashrc
source ~/.bashrc

# Then just use:
crisis-env
```

## Running Tests

Tests require the project environment (so `datasets`, `pytest`, etc. are available). Activate first, then run:

```bash
source activate_env.sh
pytest tests/ -v
# Or: make test
```

## Verify Environment

```bash
# Check if environment is active
echo $CONDA_DEFAULT_ENV

# Should show: crisis_agent

# Check Python version
python --version
# Should show: Python 3.10.x
```

## Install Dependencies

After activating the environment:

```bash
source activate_env.sh
pip install -r requirements.txt
```

## Troubleshooting

### Environment not found

```bash
# List all environments
eval "$(conda shell.bash hook)"
conda env list

# If crisis_agent is missing, recreate it:
conda create -n crisis_agent python=3.10 -y
```

### Conda command not found

```bash
# Initialize conda
eval "$(conda shell.bash hook)"

# Or add to ~/.bashrc permanently
echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
```

### Environment disappears between sessions

This can happen if:
1. Conda isn't initialized in your shell
2. The environment was removed

**Solution:** Always use `source activate_env.sh` or add conda initialization to your `~/.bashrc`

## Environment Details

- **Name:** `crisis_agent`
- **Python:** 3.10.x
- **Location:** typically `$CONDA_PREFIX` or `~/miniconda3/envs/crisis_agent`
