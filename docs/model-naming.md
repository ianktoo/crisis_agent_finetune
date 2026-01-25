# Model Naming Guide

This guide explains how to control the output model names for checkpoints and merged models.

## Overview

The pipeline saves models in several places:

1. **Training checkpoints**: Saved during training (e.g., `checkpoint-500`, `checkpoint-1000`)
2. **Final checkpoint**: Saved after training completes (default: `final`)
3. **Merged model**: Created when merging LoRA weights (default: `final_model`)

## Controlling Final Checkpoint Name

### Method 1: Configuration File (Recommended)

Edit `configs/training_config.yaml`:

```yaml
training:
  output_dir: "outputs/checkpoints"
  final_model_name: "crisis_agent_v1"  # Custom name for final checkpoint
```

The final model will be saved to: `outputs/checkpoints/crisis_agent_v1/`

### Method 2: Command-Line Argument

Use the `--model-name` argument when running training:

```bash
python scripts/train.py --model-name "crisis_agent_v1"
```

This overrides the config file setting.

### Method 3: Using Makefile

You can modify the Makefile or pass arguments:

```bash
python scripts/train.py --model-name "my_custom_model"
```

## Examples

### Example 1: Versioned Model Names

```bash
# Train with version number
python scripts/train.py --model-name "crisis_agent_v1.0"

# Output: outputs/checkpoints/crisis_agent_v1.0/
```

### Example 2: Date-Based Naming

```bash
# Train with date
python scripts/train.py --model-name "crisis_agent_2026-01-25"

# Output: outputs/checkpoints/crisis_agent_2026-01-25/
```

### Example 3: Experiment-Based Naming

```bash
# Train with experiment identifier
python scripts/train.py --model-name "exp1_lr2e4_epoch3"

# Output: outputs/checkpoints/exp1_lr2e4_epoch3/
```

## Controlling Merged Model Name

The merge script uses the `--output` argument:

```bash
# Default merge location
python scripts/merge_lora.py --checkpoint outputs/checkpoints/final

# Custom merge location
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/crisis_agent_v1 \
  --output outputs/models/crisis_agent_v1_merged
```

## Complete Workflow Example

```bash
# 1. Train with custom name
python scripts/train.py --model-name "crisis_agent_v1.0"

# 2. Evaluate the model
python scripts/evaluate.py --checkpoint outputs/checkpoints/crisis_agent_v1.0

# 3. Merge with custom output name
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/crisis_agent_v1.0 \
  --output outputs/models/crisis_agent_v1.0_merged

# 4. Use merged model for inference
python scripts/infer.py \
  --checkpoint outputs/models/crisis_agent_v1.0_merged \
  --prompt "Your crisis scenario"
```

## Model Directory Structure

After training and merging, your directory structure might look like:

```
outputs/
├── checkpoints/
│   ├── checkpoint-500/          # Intermediate checkpoint
│   ├── checkpoint-1000/         # Intermediate checkpoint
│   └── crisis_agent_v1.0/       # Final checkpoint (custom name)
├── logs/
│   └── crisis_agent_20260125.log
└── models/                      # Custom directory for merged models
    └── crisis_agent_v1.0_merged/ # Merged model
```

## Best Practices

### 1. Use Descriptive Names

```yaml
# Good
final_model_name: "crisis_agent_v1.0_lr2e4"
final_model_name: "experiment_2026-01-25"
final_model_name: "crisis_agent_baseline"

# Avoid
final_model_name: "model"
final_model_name: "test"
final_model_name: "final"
```

### 2. Include Version Information

```yaml
final_model_name: "crisis_agent_v1.0"
final_model_name: "crisis_agent_v1.1"
final_model_name: "crisis_agent_v2.0"
```

### 3. Include Hyperparameters

```yaml
final_model_name: "crisis_agent_lr2e4_epoch3"
final_model_name: "crisis_agent_lr1e4_epoch5_bs4"
```

### 4. Use Dates for Experiments

```yaml
final_model_name: "crisis_agent_2026-01-25"
final_model_name: "crisis_agent_20260125_exp1"
```

## Configuration Examples

### Example 1: Simple Versioning

```yaml
# configs/training_config.yaml
training:
  output_dir: "outputs/checkpoints"
  final_model_name: "crisis_agent_v1"
```

### Example 2: Experiment Tracking

```yaml
# configs/training_config.yaml
training:
  output_dir: "outputs/experiments/exp1"
  final_model_name: "final"
```

Then use command-line for specific runs:

```bash
python scripts/train.py --model-name "exp1_run1"
python scripts/train.py --model-name "exp1_run2"
```

### Example 3: Date-Based Organization

```yaml
# configs/training_config.yaml
training:
  output_dir: "outputs/checkpoints"
  final_model_name: "crisis_agent_2026-01-25"
```

## Accessing Models

After training, you can reference your model by its name:

```bash
# Evaluate
python scripts/evaluate.py --checkpoint outputs/checkpoints/crisis_agent_v1.0

# Merge
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/crisis_agent_v1.0 \
  --output outputs/models/crisis_agent_v1.0_merged

# Inference
python scripts/infer.py --checkpoint outputs/checkpoints/crisis_agent_v1.0
```

## Troubleshooting

### Model Name Not Found

If you get an error about the model not being found:

1. Check the exact path: `ls outputs/checkpoints/`
2. Verify the name matches exactly (case-sensitive)
3. Check logs for the actual saved path

### Special Characters

Avoid special characters in model names that might cause issues:

```yaml
# Good
final_model_name: "crisis_agent_v1"
final_model_name: "crisis-agent-v1"
final_model_name: "crisis_agent_v1.0"

# Avoid
final_model_name: "crisis agent v1"  # Spaces
final_model_name: "crisis/agent"     # Slashes
final_model_name: "crisis:agent"      # Colons
```

### Default Behavior

If no custom name is specified:

- Config file: Uses `final_model_name` from config (default: `"final"`)
- Command-line: Uses config value or `"final"` as fallback
- Final checkpoint: `outputs/checkpoints/final/`

## Summary

- **Config file**: Set `final_model_name` in `training_config.yaml`
- **Command-line**: Use `--model-name` argument
- **Merge output**: Use `--output` argument in merge script
- **Best practice**: Use descriptive, versioned names

For more information, see the main [README.md](../README.md).
