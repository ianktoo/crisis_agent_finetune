# Dataset Information

## Primary Dataset

This pipeline is configured to use the **Crisis Response Training Dataset (v2)**:

- **Hugging Face Repository**: [ianktoo/crisis-response-training-v2](https://huggingface.co/datasets/ianktoo/crisis-response-training-v2)
- **Purpose**: Training data for crisis companion application
- **Use Case**: Fine-tuning Mistral-7B to provide structured, actionable responses to crisis scenarios

## Dataset Configuration

The dataset is pre-configured in `configs/dataset_config.yaml`:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training-v2"
  train_split: "train"
  eval_split: "validation"
  instruction_column: "Input"   # Scenario / instruction
  response_column: "Output"     # Structured text response
```

## Dataset Structure

The dataset contains crisis scenario training data with:

- **Input**: Crisis scenarios (Category, Scenario, Role). Served as the instruction.
- **Output**: Structured text with sections **FACTS**, **UNCERTAINTIES**, **ANALYSIS**, **GUIDANCE** (bullet points).
- **Columns**: `Instruction`, `Input`, `Output`, `category`, `role`.

### Expected Format

- **Input**: e.g. `"Category: substance abuse crisis\n\nScenario:\n..."` (full scenario + role).
- **Output**: Structured text, e.g.:

  ```
  FACTS:
    • Person is unconscious in a park
    • Multiple syringes are near the person
  UNCERTAINTIES:
    • Person's medical condition
    • Cause of unconsciousness
  ANALYSIS:
    ...
  GUIDANCE:
    ...
  ```

For local JSONL, use `instruction` / `response` and point `hf_dataset_name` to the file path.

## Verifying Dataset Structure

Before training, verify your dataset structure:

```python
from datasets import load_dataset

dataset = load_dataset("ianktoo/crisis-response-training-v2")
print(dataset)
print(dataset["train"][0])  # Check first sample
print(dataset["train"].column_names)  # Expect: Instruction, Input, Output, category, role
```

## Customizing Dataset Configuration

If your dataset has different column names, update `configs/dataset_config.yaml`:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training-v2"
  instruction_column: "Input"       # Or "prompt", "instruction", etc.
  response_column: "Output"         # Or "answer", "response", etc.
```

## Dataset Access

### Public Dataset

If the dataset is public, no authentication is needed. The pipeline will automatically download it.

### Private Dataset

If the dataset is private, set your Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or use the `.env` file:

```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

## Dataset Splits

The pipeline expects:

- **Training split**: `train` (required)
- **Validation split**: `validation` or `test` (optional, will auto-split if missing)

If your dataset only has one split, the pipeline will automatically create a 90/10 train/validation split.

## Dataset Size Considerations

For optimal training:

- **Minimum**: ~100-500 samples (for testing/validation)
- **Recommended**: ~1,000-5,000 samples (for good performance)
- **Large**: 10,000+ samples (may require longer training time)

You can limit the dataset size for testing:

```yaml
dataset:
  max_samples: 1000  # Limit to first 1000 samples
```

## Dataset Updates

If the dataset is updated on Hugging Face:

1. The pipeline will use the cached version by default
2. To force re-download, clear the cache: `rm -rf data/local_cache/*`
3. Or specify a revision in the config:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training-v2"
  revision: "main"  # Or specific tag/commit
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{crisis_response_training_2026,
  title = {Crisis Response Training Dataset},
  author = {Ian K. T.},
  year = {2026},
  url = {https://huggingface.co/datasets/ianktoo/crisis-response-training-v2},
  note = {Synthetic dataset for training crisis response language models}
}
```

## Related Resources

- **Dataset Repository**: https://huggingface.co/datasets/ianktoo/crisis-response-training-v2
- **Dataset Setup Guide**: [dataset-setup.md](dataset-setup.md)
- **Main README**: [../README.md](../README.md)
