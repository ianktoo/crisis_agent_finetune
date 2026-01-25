# Dataset Setup Guide

This guide explains how to configure and use Hugging Face datasets with the crisis-agent fine-tuning pipeline.

## Quick Start

1. **Find your dataset** on [Hugging Face Datasets](https://huggingface.co/datasets)
2. **Update `configs/dataset_config.yaml`** with your dataset name
3. **Set column names** to match your dataset structure
4. **Run training** - the dataset will be automatically downloaded and cached

## Configuration

### Basic Configuration

Edit `configs/dataset_config.yaml`:

```yaml
dataset:
  # Your Hugging Face dataset name
  hf_dataset_name: "username/dataset_name"
  
  # Column names in your dataset
  instruction_column: "instruction"  # Column with prompts/instructions
  response_column: "response"        # Column with responses
```

### Dataset Name Formats

The `hf_dataset_name` can be specified in several ways:

1. **Public dataset**: `"username/dataset_name"` or `"dataset_name"`
2. **Private dataset**: `"username/private_dataset"` (requires authentication)
3. **With revision**: `"username/dataset_name:main"` or `"username/dataset_name:v1.0"`

### Private Datasets

For private datasets, you need to authenticate:

**Option 1: Environment Variable (Recommended)**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Option 2: Hugging Face CLI**
```bash
huggingface-cli login
```

**Option 3: In config file (Not recommended for security)**
```yaml
dataset:
  hf_token: "your_token_here"  # Only use if necessary
```

Get your token from: https://huggingface.co/settings/tokens

### Dataset Structure

Your dataset should have columns for instructions and responses. Common formats:

**Format 1: Separate columns**
```python
{
  "instruction": "What should I do in a fire emergency?",
  "response": '{"action": "evacuate", "priority": "high", ...}'
}
```

**Format 2: Combined format**
```python
{
  "prompt": "What should I do in a fire emergency?",
  "completion": '{"action": "evacuate", "priority": "high", ...}'
}
```

Update `instruction_column` and `response_column` in the config to match your dataset.

### Dataset Splits

The pipeline automatically handles different split configurations:

- **If dataset has train/validation splits**: Uses them directly
- **If dataset has only one split**: Automatically splits 90/10 (train/validation)
- **If dataset has train/test**: Uses train for training, test for validation

You can specify which splits to use:
```yaml
dataset:
  train_split: "train"
  eval_split: "validation"  # or "test"
```

## Examples

### Example 1: Public Dataset

```yaml
dataset:
  hf_dataset_name: "microsoft/DialoGPT-medium"
  instruction_column: "instruction"
  response_column: "response"
```

### Example 2: Private Dataset with Authentication

```bash
# Set token
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

```yaml
dataset:
  hf_dataset_name: "my_org/private_crisis_dataset"
  instruction_column: "prompt"
  response_column: "answer"
```

### Example 3: Dataset with Multiple Configs

```yaml
dataset:
  hf_dataset_name: "username/multi_config_dataset"
  dataset_config_name: "crisis_scenarios"  # Specific config name
  instruction_column: "instruction"
  response_column: "response"
```

### Example 4: Specific Dataset Version

```yaml
dataset:
  hf_dataset_name: "username/crisis_dataset"
  revision: "v1.2"  # Specific version/tag
  instruction_column: "instruction"
  response_column: "response"
```

## Data Format Requirements

### Instruction Column

The instruction column should contain:
- Crisis scenarios
- Questions about emergency situations
- Prompts that require structured responses

Example:
```
"What should I do if I smell gas in my home?"
"A building is on fire with people trapped inside. What's the priority?"
```

### Response Column

The response column should contain:
- Structured JSON responses (recommended)
- Or plain text responses that can be validated

Example JSON:
```json
{
  "action": "evacuate immediately",
  "priority": "critical",
  "reasoning": "Gas leaks are extremely dangerous",
  "resources": ["fire_department", "gas_company"]
}
```

### Prompt Template

The pipeline formats your data using a template. Default:

```yaml
prompt_template: |
  <s>[INST] {instruction} [/INST]
  {response}</s>
```

You can customize this in `configs/dataset_config.yaml` to match your model's expected format.

## Data Processing Options

### Limiting Samples

For testing or quick iterations:

```yaml
dataset:
  max_samples: 1000  # Limit to first 1000 samples
```

### Shuffling

Shuffle is enabled by default:

```yaml
dataset:
  shuffle: true
  shuffle_seed: 42
```

### JSON Validation

Validate that responses are valid JSON:

```yaml
dataset:
  validate_json: true
  strict_json: false  # false = warn but continue, true = reject invalid
```

## Caching

Datasets are automatically cached to speed up subsequent loads:

```yaml
dataset:
  cache_dir: "data/local_cache"
  use_cache: true
```

The cache persists between runs, so you only download once.

## Troubleshooting

### Dataset Not Found

**Error**: `Dataset not found: username/dataset_name`

**Solutions**:
1. Check the dataset name is correct on Hugging Face
2. Verify the dataset is public or you have access
3. For private datasets, set `HF_TOKEN` environment variable

### Authentication Failed

**Error**: `401 Unauthorized` or `Authentication failed`

**Solutions**:
1. Get your token from https://huggingface.co/settings/tokens
2. Set environment variable: `export HF_TOKEN="your_token"`
3. Or login via CLI: `huggingface-cli login`

### Column Not Found

**Error**: `KeyError: 'instruction'` or similar

**Solutions**:
1. Check your dataset's column names
2. Update `instruction_column` and `response_column` in config
3. Inspect dataset: `python -c "from datasets import load_dataset; print(load_dataset('your_dataset')['train'][0])"`

### Wrong Dataset Format

**Error**: Dataset doesn't have expected structure

**Solutions**:
1. Check if dataset has multiple configs - use `dataset_config_name`
2. Verify dataset has train/validation splits or can be auto-split
3. Check dataset documentation on Hugging Face

### Out of Memory During Loading

**Error**: Memory issues when loading large dataset

**Solutions**:
1. Use `max_samples` to limit dataset size
2. Enable streaming for very large datasets (requires code modification)
3. Increase system RAM or use a machine with more memory

## Inspecting Your Dataset

Before training, you can inspect your dataset:

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("username/your_dataset")

# Check structure
print(dataset)
print(dataset["train"][0])  # First sample
print(dataset["train"].features)  # Column names and types
```

## Best Practices

1. **Test with small sample first**: Set `max_samples: 100` to test your config
2. **Validate data format**: Ensure instruction/response columns exist
3. **Check JSON structure**: If using JSON responses, validate format
4. **Monitor cache size**: Large datasets can use significant disk space
5. **Use versioning**: Specify `revision` for reproducible experiments

## Next Steps

Once your dataset is configured:

1. **Test loading**: The training script will load and validate your dataset
2. **Check logs**: Review `outputs/logs/` for dataset loading information
3. **Start training**: Run `python scripts/train.py`

For more information, see the main [README.md](../README.md).
