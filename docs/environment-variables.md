# Environment Variables

This document describes all environment variables used by the crisis-agent fine-tuning pipeline.

## Quick Setup

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values

3. Load variables (if using a tool like `python-dotenv`):
   ```bash
   # Variables are automatically loaded by the pipeline
   ```

## Required Variables

### Hugging Face Token

**Variable**: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`

**Description**: Hugging Face authentication token for accessing private datasets or models.

**Required**: Only if using private datasets or models

**Example**:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**How to get**:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access for datasets, write access if uploading)
3. Copy the token

**Usage in code**: The pipeline automatically checks for this token when loading datasets.

## Optional Variables

### AI Evaluation (Claude, OpenAI, Gemini)

Used by `scripts/evaluate.py --ai` for AI-based quality evaluation.

| Variable | Provider | Required when |
|----------|----------|----------------|
| `ANTHROPIC_API_KEY` | Claude | `--ai-provider anthropic` (default) |
| `OPENAI_API_KEY` | OpenAI | `--ai-provider openai` |
| `GEMINI_API_KEY` | Gemini | `--ai-provider gemini` |

**Example**:
```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
```

**How to get**:
- **Anthropic**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys
- **Gemini**: https://aistudio.google.com/app/apikey

**Usage**: Set in `.env` or export before running `python scripts/evaluate.py --ai`. See [ai-evaluation.md](ai-evaluation.md).

### Weights & Biases (W&B) Integration

**Variable**: `WANDB_API_KEY`

**Description**: API key for Weights & Biases experiment tracking.

**Required**: Only if using W&B for experiment tracking

**Example**:
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

**Usage**: Set `report_to: ["wandb"]` in `configs/training_config.yaml`

### Hugging Face Cache Directory

**Variable**: `HF_HOME` or `TRANSFORMERS_CACHE`

**Description**: Custom directory for Hugging Face model and dataset cache.

**Default**: `~/.cache/huggingface`

**Example**:
```bash
export HF_HOME="/path/to/custom/cache"
```

### CUDA Device

**Variable**: `CUDA_VISIBLE_DEVICES`

**Description**: Specify which GPU device(s) to use.

**Default**: Uses all available GPUs

**Example**:
```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

### Python Path

**Variable**: `PYTHONPATH`

**Description**: Additional paths to add to Python's module search path.

**Example**:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/additional/modules"
```

## Setting Environment Variables

### Linux/macOS

**Temporary (current session only)**:
```bash
export HF_TOKEN="your_token_here"
```

**Permanent (add to `~/.bashrc` or `~/.zshrc`)**:
```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**Using `.env` file** (recommended):
```bash
# Install python-dotenv if needed
pip install python-dotenv

# Create .env file
cat > .env << EOF
HF_TOKEN=your_token_here
WANDB_API_KEY=your_wandb_key_here
EOF

# Load in Python (if using python-dotenv)
from dotenv import load_dotenv
load_dotenv()
```

### Windows (PowerShell)

**Temporary (current session only)**:
```powershell
$env:HF_TOKEN="your_token_here"
```

**Permanent (User Environment Variables)**:
1. Open System Properties → Environment Variables
2. Add new User variable: `HF_TOKEN` = `your_token_here`

**Using `.env` file**:
```powershell
# Create .env file
@"
HF_TOKEN=your_token_here
WANDB_API_KEY=your_wandb_key_here
"@ | Out-File -FilePath .env -Encoding utf8
```

### Jupyter Notebooks

```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

## Environment Variable Priority

The pipeline checks for environment variables in this order:

1. **System environment variables** (highest priority)
2. **`.env` file** (if using `python-dotenv`)
3. **Config file values** (lowest priority, not recommended for secrets)

## Security Best Practices

1. **Never commit `.env` files**: Already in `.gitignore`
2. **Never commit tokens in config files**: Use environment variables
3. **Use read-only tokens when possible**: For dataset access only
4. **Rotate tokens regularly**: Especially if exposed
5. **Use different tokens for different environments**: Dev, staging, production

## Verification

Check if your environment variables are set:

**Linux/macOS**:
```bash
echo $HF_TOKEN
```

**Windows (PowerShell)**:
```powershell
$env:HF_TOKEN
```

**Python**:
```python
import os
print(os.getenv('HF_TOKEN'))
```

## Troubleshooting

### Token Not Working

1. **Verify token is set**:
   ```bash
   echo $HF_TOKEN  # Linux/macOS
   # or
   $env:HF_TOKEN   # Windows PowerShell
   ```

2. **Check token format**: Should start with `hf_`

3. **Verify token permissions**: Ensure it has access to the dataset/model

4. **Test token manually**:
   ```python
   from huggingface_hub import whoami
   print(whoami())  # Should print your username
   ```

### Environment Variable Not Loading

1. **Check if variable is exported**: Use `env | grep HF_TOKEN`
2. **Restart terminal/session**: Environment variables are session-specific
3. **Check `.env` file location**: Should be in project root
4. **Verify `.env` file format**: No spaces around `=`, no quotes needed

### Permission Denied

1. **Check token has correct permissions**: Read for datasets, write for uploads
2. **Verify dataset/model access**: You may need to accept terms or request access
3. **Check organization permissions**: If dataset belongs to an org

## Example `.env` File

Create a `.env` file in the project root:

```bash
# Hugging Face
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# AI evaluation (optional; for evaluate.py --ai)
# ANTHROPIC_API_KEY=your_anthropic_key_here
# OPENAI_API_KEY=your_openai_key_here
# GEMINI_API_KEY=your_gemini_key_here

# Weights & Biases (optional)
# WANDB_API_KEY=your_wandb_api_key_here

# Custom cache directory (optional)
# HF_HOME=/path/to/cache

# CUDA device selection (optional)
# CUDA_VISIBLE_DEVICES=0
```

**Note**: The `.env` file is already in `.gitignore` and will not be committed.
