# Configuration Reference

All configuration files are located in the `configs/` directory.

---

## Configuration Files Overview

| File | Purpose |
|------|---------|
| `dataset_config.yaml` | Dataset source, columns, splits |
| `model_config.yaml` | Model, quantization, LoRA settings |
| `training_config.yaml` | Training hyperparameters |

---

## Dataset Configuration

**File:** `configs/dataset_config.yaml`

```yaml
dataset:
  # HuggingFace dataset name
  hf_dataset_name: "ianktoo/crisis-response-training-v2"
  
  # Dataset splits
  train_split: "train"
  eval_split: "validation"
  
  # Column mappings
  instruction_column: "Input"    # Input/prompt column
  response_column: "Output"      # Expected response column
  
  # Sample limits (-1 for all)
  max_samples: -1
  
  # Local dataset path (alternative to HF)
  local_dataset_path: null  # e.g., "data/local_dataset.jsonl"
```

### Column Mappings

| Dataset Source | Instruction Column | Response Column |
|----------------|-------------------|-----------------|
| HuggingFace (default) | `Input` | `Output` |
| Local JSONL | `instruction` | `response` |

### Using Local Dataset

```yaml
dataset:
  hf_dataset_name: null
  local_dataset_path: "data/my_dataset.jsonl"
  instruction_column: "instruction"
  response_column: "response"
```

---

## Model Configuration

**File:** `configs/model_config.yaml`

```yaml
model:
  # Base model (Unsloth version for faster training)
  model_name: "unsloth/Mistral-7B-Instruct-v0.2"
  
  # Quantization settings (4-bit for memory efficiency)
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  
  # LoRA configuration
  lora:
    r: 16                    # LoRA rank
    lora_alpha: 32           # Alpha (typically 2x r)
    target_modules:          # Modules to apply LoRA
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"
  
  # Attention
  use_flash_attention_2: true
  
  # Sequence length
  max_seq_length: 2048
  
  # Other
  trust_remote_code: false
```

### LoRA Rank Guidelines

| VRAM | Recommended `r` | Quality |
|------|-----------------|---------|
| 12GB | 8 | Good |
| 16GB | 16 | Better |
| 24GB | 32 | Best |

### Sequence Length vs Memory

| `max_seq_length` | Memory Usage | Use Case |
|-----------------|--------------|----------|
| 1024 | ~10GB | Memory constrained |
| 2048 | ~12-14GB | Default |
| 4096 | ~16-20GB | Long contexts |

---

## Training Configuration

**File:** `configs/training_config.yaml`

```yaml
training:
  # Output directories
  output_dir: "outputs/checkpoints"
  logging_dir: "outputs/logs"
  final_model_name: "final"    # Name for final checkpoint
  
  # Training parameters
  num_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  
  # Optimizer
  learning_rate: 2.0e-4
  optim: "adamw_torch"
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Learning rate schedule
  lr_scheduler_type: "cosine"
  warmup_steps: 100
  
  # Checkpointing
  save_steps: 1000
  save_total_limit: 2
  save_strategy: "steps"
  
  # Evaluation
  eval_steps: 1000
  eval_strategy: "steps"
  
  # Logging
  logging_steps: 25
  report_to: []              # ["wandb"] for W&B
  
  # Precision
  fp16: true
  bf16: false
  
  # Data loading
  dataloader_num_workers: 4  # Match CPU cores
  remove_unused_columns: true
  
  # Reproducibility
  seed: 42
```

### Effective Batch Size

```
Effective batch size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
                     = 2 × 4 × 1 = 8
```

### Memory-Constrained Settings

For 12GB VRAM:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  
model:
  max_seq_length: 1024
  lora:
    r: 8
```

### Quality-Focused Settings

For 24GB+ VRAM:

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  num_epochs: 5
  
model:
  max_seq_length: 4096
  lora:
    r: 32
    lora_alpha: 64
```

---

## Environment Variables

**File:** `.env`

```bash
# Required
HF_TOKEN=hf_your_token_here

# Optional - AI Evaluation
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Optional - HuggingFace Upload
HF_WRITE_TOKEN=hf_...  # If different from HF_TOKEN
```

### Getting API Keys

| Service | URL |
|---------|-----|
| HuggingFace | https://huggingface.co/settings/tokens |
| Anthropic (Claude) | https://console.anthropic.com/ |
| OpenAI | https://platform.openai.com/api-keys |
| Google (Gemini) | https://aistudio.google.com/apikey |

---

## Configuration Profiles

### Development (Fast Iteration)

```yaml
# training_config.yaml
training:
  num_epochs: 1
  save_steps: 100
  eval_steps: 100
  max_steps: 500  # Limit steps

# dataset_config.yaml
dataset:
  max_samples: 500  # Subset of data
```

### Production (Full Training)

```yaml
# training_config.yaml
training:
  num_epochs: 3
  save_steps: 1000
  eval_steps: 1000
  max_steps: -1

# dataset_config.yaml
dataset:
  max_samples: -1  # All data
```

---

## Configuration Validation

The pipeline validates configurations at startup. Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `Dataset not found` | Invalid `hf_dataset_name` | Check dataset name/access |
| `Column not found` | Wrong column names | Match column names to dataset |
| `CUDA OOM` | Settings too aggressive | Reduce batch size/seq length |

---

## Overriding Configuration

### Command Line

```bash
# Custom output directory
python scripts/train.py --output-dir outputs/experiment1

# Custom model name
python scripts/train.py --model-name crisis_agent_v2
```

### Environment Variables

Some settings can be overridden via environment:

```bash
export HF_TOKEN="your_token"
export CUDA_VISIBLE_DEVICES="0"
```

---

[← Getting Started](Getting-Started.md) | [Home](Home.md) | [Training →](Training.md)
