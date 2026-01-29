# Training Guide

Complete guide to training your AI Emergency Kit model.

---

## Training Overview

The training pipeline uses:
- **Unsloth** for optimized training
- **LoRA** for parameter-efficient fine-tuning
- **4-bit quantization** for memory efficiency
- **Transformers Trainer** for training loop

---

## Quick Start

```bash
# Ensure environment is activated
source activate_env.sh

# Start training
make train
```

---

## Training Commands

### Basic Training

```bash
# Default training (uses configs/)
make train

# Or directly
python scripts/train.py
```

### Custom Training

```bash
# Custom model name
python scripts/train.py --model-name "crisis_agent_v2"

# Custom output directory
python scripts/train.py --output-dir "outputs/experiment1"

# Combined
python scripts/train.py \
  --model-name "experiment_lr3e4" \
  --output-dir "outputs/experiments/lr_test"
```

### Resume Training

```bash
# Resume from checkpoint
python scripts/train.py --resume-from outputs/checkpoints/checkpoint-500
```

---

## Training Process

### What Happens During Training

1. **Model Loading**
   - Loads Mistral-7B with 4-bit quantization
   - Applies LoRA adapters to target modules
   - ~4-5GB VRAM for base model

2. **Dataset Preparation**
   - Loads from HuggingFace Hub
   - Formats with chat template
   - Tokenizes and caches

3. **Training Loop**
   - Forward pass through model
   - Compute loss
   - Backward pass (gradients)
   - Update LoRA weights
   - Save checkpoints at intervals

4. **Final Save**
   - Saves final checkpoint
   - Logs training summary

### Training Outputs

```
outputs/
├── checkpoints/
│   ├── checkpoint-500/     # Mid-training checkpoint
│   ├── checkpoint-1000/    # Another checkpoint
│   └── final/              # Final trained model
└── logs/
    └── training.log        # Detailed logs
```

---

## Monitoring Training

### Console Output

```
[2026-01-28 10:00:00] Starting training...
[2026-01-28 10:00:05] Epoch 1/3
[2026-01-28 10:05:00] Step 100/750, Loss: 0.8234
[2026-01-28 10:10:00] Step 200/750, Loss: 0.5123
...
[2026-01-28 11:30:00] Training completed!
[2026-01-28 11:30:00] Final checkpoint saved to: outputs/checkpoints/final
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Log Files

```bash
# View training logs
tail -f outputs/logs/training.log

# Search for errors
grep -i error outputs/logs/*.log
```

---

## Training Parameters

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 3 | Training epochs |
| `learning_rate` | 2e-4 | Initial learning rate |
| `batch_size` | 2 | Per-device batch size |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `warmup_steps` | 100 | LR warmup steps |

### Effective Batch Size

```
Effective = batch_size × gradient_accumulation × num_gpus
         = 2 × 4 × 1 = 8
```

### Learning Rate Schedule

Default: **Cosine decay** with warmup

```
LR
↑
│    /‾‾‾‾\
│   /      \
│  /        \
│ /          \___
└──────────────────→ Steps
  ↑          ↑
  warmup     decay
```

---

## Training Tips

### For Better Quality

1. **More epochs** (3-5 typically optimal)
2. **Larger LoRA rank** (r=32 if VRAM allows)
3. **Lower learning rate** (1e-4 to 2e-4)
4. **More data** (if available)

### For Faster Training

1. **Fewer epochs** (1-2 for quick tests)
2. **Limit samples** (`max_samples: 500` in config)
3. **Reduce checkpointing** (increase `save_steps`)
4. **Flash Attention** (enabled by default)

### For Memory Efficiency

1. **Reduce batch size** to 1
2. **Increase gradient accumulation** to 8
3. **Reduce sequence length** to 1024
4. **Reduce LoRA rank** to 8

---

## Expected Training Metrics

### Loss Curve

| Epoch | Typical Loss Range |
|-------|-------------------|
| Start | 1.5 - 2.5 |
| Mid | 0.5 - 1.0 |
| End | 0.15 - 0.4 |

### Training Time

| Dataset Size | Time (16GB GPU) |
|--------------|-----------------|
| 500 samples | 15-30 min |
| 2000 samples | 45-90 min |
| 5000 samples | 2-4 hours |

---

## Checkpoints

### Checkpoint Structure

```
checkpoint-500/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── config.json              # Model config
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── training_state.json      # Training state
```

### Managing Checkpoints

```bash
# List checkpoints
ls -la outputs/checkpoints/

# Check checkpoint size
du -sh outputs/checkpoints/*

# Remove old checkpoints (keep last 2 by default)
# Handled automatically by save_total_limit
```

---

## Training with Weights & Biases

Enable W&B logging:

```yaml
# training_config.yaml
training:
  report_to: ["wandb"]
```

```bash
# Set W&B API key
export WANDB_API_KEY="your_key"

# Start training
make train
```

---

## Full Training Pipeline

```bash
# 1. Setup
source activate_env.sh
make verify

# 2. Train
make train

# 3. Evaluate
make evaluate

# 4. Test
make infer

# 5. Deploy
make merge
make export-gguf
```

Or use the combined pipeline:

```bash
make pipeline  # train -> evaluate -> merge
```

---

## Troubleshooting Training

### CUDA Out of Memory

```bash
# Reduce batch size
# In training_config.yaml:
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### Training Loss Not Decreasing

1. Check learning rate (try 1e-4 or 5e-5)
2. Check data quality
3. Increase warmup steps
4. Check for data issues in logs

### Training Interrupted

```bash
# Resume from last checkpoint
python scripts/train.py --resume-from outputs/checkpoints/checkpoint-500
```

See [Troubleshooting](Troubleshooting.md) for more solutions.

---

[← Configuration](Configuration.md) | [Home](Home.md) | [Evaluation →](Evaluation.md)
