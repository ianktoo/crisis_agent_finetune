# Getting Started

This guide walks you through setting up and running your first training.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (tested with 3.10.12)
- **CUDA-capable GPU** with 16GB+ VRAM
- **Conda** installed (Miniconda or Anaconda)
- **HuggingFace account** (for dataset access)

---

## Step 1: Environment Setup

### Option A: Automatic Setup (Recommended)

The `activate_env.sh` script automatically creates the conda environment if it doesn't exist:

```bash
# Navigate to project directory
cd /home/jovyan/work/projects/crisis_agent_finetune

# Activate (creates environment if needed)
source activate_env.sh
```

This script will:
1. Check if `crisis_agent` conda environment exists
2. Create it with Python 3.10 if not
3. Install all dependencies from `requirements.txt`
4. Activate the environment
5. Navigate to the project directory

### Option B: Manual Setup

```bash
# Create conda environment
conda create -n crisis_agent python=3.10 -y

# Activate environment
conda activate crisis_agent

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Configure Environment Variables

### Create .env File

```bash
# Copy example file
cp .env.example .env

# Edit and add your tokens
nano .env
```

### Required Variables

```bash
# HuggingFace token (required for dataset access)
HF_TOKEN=hf_your_token_here
```

### Optional Variables (for AI Evaluation)

```bash
# For Claude-based evaluation
ANTHROPIC_API_KEY=sk-ant-your_key_here

# For OpenAI-based evaluation
OPENAI_API_KEY=sk-your_key_here

# For Gemini-based evaluation
GEMINI_API_KEY=your_key_here
```

### Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create new token with **read** access
3. Copy and add to `.env`

---

## Step 3: Verify Setup

Run the verification script to check everything is ready:

```bash
python scripts/verify_setup.py

# Or use Makefile
make verify
```

This checks:
- All dependencies installed
- CUDA available
- Configuration files valid
- Dataset accessible
- Required directories exist

**Expected output:**
```
✅ All dependencies installed
✅ CUDA is available (GPU: NVIDIA ...)
✅ Configuration files valid
✅ Dataset accessible
✅ Directories created
```

---

## Step 4: Review Configuration

The default configuration is optimized for 16GB VRAM. Review these files if needed:

### Dataset Configuration
`configs/dataset_config.yaml`:
```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training-v2"
  instruction_column: "Input"
  response_column: "Output"
```

### Model Configuration
`configs/model_config.yaml`:
```yaml
model:
  model_name: "unsloth/Mistral-7B-Instruct-v0.2"
  load_in_4bit: true
  max_seq_length: 2048
  lora:
    r: 16
    lora_alpha: 32
```

### Training Configuration
`configs/training_config.yaml`:
```yaml
training:
  num_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4
```

---

## Step 5: Start Training

```bash
# Start training
make train

# Or with custom model name
python scripts/train.py --model-name "my_crisis_agent_v1"
```

**Training takes approximately:**
- 30-60 minutes per epoch
- 1.5-3 hours total (3 epochs)

**Monitor progress:**
- Watch console output for loss values
- Check `outputs/logs/` for detailed logs
- Checkpoints saved to `outputs/checkpoints/`

---

## Step 6: Evaluate Model

After training completes:

```bash
# Standard evaluation
make evaluate

# With AI-based quality assessment (optional)
make evaluate-ai          # Uses Claude
make evaluate-ai-gemini   # Uses Gemini
```

**Evaluation report:** `outputs/evaluation_report.json`

---

## Step 7: Test Interactively

```bash
# Interactive mode
make infer

# Or single prompt
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "A building is on fire with people trapped"
```

---

## Next Steps

After successful training and evaluation:

1. **[Merge LoRA weights](Deployment.md#merge-lora)** - Prepare for deployment
2. **[Export to GGUF](Deployment.md#export-to-gguf)** - For LM Studio/Ollama
3. **[Upload to HuggingFace](Deployment.md#huggingface)** - Share your model

---

## Quick Reference

| Step | Command | Time |
|------|---------|------|
| Setup | `source activate_env.sh` | 5-15 min |
| Verify | `make verify` | < 1 min |
| Train | `make train` | 1.5-3 hours |
| Evaluate | `make evaluate` | 5-15 min |
| Test | `make infer` | Interactive |

---

## Troubleshooting

### Environment activation fails

```bash
# Initialize conda first
eval "$(conda shell.bash hook)"

# Then activate
source activate_env.sh
```

### CUDA not available

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset access denied

1. Ensure `HF_TOKEN` is set in `.env`
2. Verify token has read access
3. Accept dataset terms on HuggingFace

See [Troubleshooting](Troubleshooting.md) for more solutions.

---

[← Home](Home.md) | [Configuration →](Configuration.md)
