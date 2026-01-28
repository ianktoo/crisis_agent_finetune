# Conda Environment Setup Guide

## Creating a Conda Environment

### Basic Commands

#### 1. Create a new conda environment
```bash
# Create environment with Python 3.10
conda create -n crisis_agent python=3.10

# Or specify a different Python version
conda create -n crisis_agent python=3.9
conda create -n crisis_agent python=3.11
```

#### 2. Create environment with packages
```bash
# Create environment and install packages in one command
conda create -n crisis_agent python=3.10 numpy pandas

# Or install from a requirements file after creation
conda create -n crisis_agent python=3.10 -y
conda activate crisis_agent
pip install -r requirements.txt
```

#### 3. Create environment from environment.yml file
```bash
# If you have an environment.yml file
conda env create -f environment.yml

# Or create from a specific file
conda env create -f environment.yml -n crisis_agent
```

## Activating and Deactivating

### Activate Environment
```bash
# Standard activation
conda activate crisis_agent

# If conda is not initialized in your shell
eval "$(conda shell.bash hook)"  # For bash
conda activate crisis_agent
```

### Deactivate Environment
```bash
conda deactivate
```

## Managing Environments

### List all environments
```bash
conda env list
# or
conda info --envs
```

### Remove an environment
```bash
conda env remove -n crisis_agent
# or
conda remove -n crisis_agent --all
```

### Clone an environment
```bash
conda create --name crisis_agent_backup --clone crisis_agent
```

### Export environment
```bash
# Export to environment.yml
conda env export > environment.yml

# Export only manually installed packages (no pip packages)
conda env export --from-history > environment.yml

# Export with no build strings (more portable)
conda env export --no-builds > environment.yml
```

## Installing Packages

### Using conda
```bash
conda activate crisis_agent
conda install numpy pandas scikit-learn
```

### Using pip (in conda environment)
```bash
conda activate crisis_agent
pip install -r requirements.txt
```

### Mixing conda and pip
```bash
# Install conda packages first
conda install numpy pandas

# Then install pip-only packages
pip install some-package
```

## For This Project

### Step-by-Step Setup

1. **Create the environment:**
```bash
conda create -n crisis_agent python=3.10 -y
```

2. **Activate the environment:**
```bash
# Initialize conda if needed
eval "$(conda shell.bash hook)"

# Activate
conda activate crisis_agent
```

3. **Install dependencies:**
```bash
# Make sure you're in the project directory
cd /home/jovyan/work/projects/crisis_agent_finetune

# Install from requirements.txt
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python scripts/verify_setup.py
```

5. **Start training:**
```bash
python scripts/train.py
```

## Creating environment.yml (Optional)

You can create an `environment.yml` file for easy environment recreation:

```yaml
name: crisis_agent
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - -r requirements.txt
```

Then create the environment with:
```bash
conda env create -f environment.yml
```

## Troubleshooting

### Conda not found
```bash
# Initialize conda
eval "$(conda shell.bash hook)"

# Or add to your ~/.bashrc
echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
source ~/.bashrc
```

### Environment not activating
```bash
# Check if environment exists
conda env list

# Try full path activation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crisis_agent
```

### Package conflicts
```bash
# Update conda
conda update conda

# Update all packages in environment
conda update --all -n crisis_agent
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `conda create -n name python=3.10` | Create new environment |
| `conda activate name` | Activate environment |
| `conda deactivate` | Deactivate current environment |
| `conda env list` | List all environments |
| `conda env remove -n name` | Remove environment |
| `conda env export > env.yml` | Export environment |
| `conda env create -f env.yml` | Create from file |
