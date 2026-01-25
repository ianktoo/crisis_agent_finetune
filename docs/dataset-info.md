# Dataset Information

## Primary Dataset

This pipeline is configured to use the **Crisis Response Training Dataset**:

- **Hugging Face Repository**: [ianktoo/crisis-response-training](https://huggingface.co/datasets/ianktoo/crisis-response-training)
- **Purpose**: Training data for crisis companion application
- **Use Case**: Fine-tuning Mistral-7B to provide structured, actionable responses to crisis scenarios

## Dataset Configuration

The dataset is pre-configured in `configs/dataset_config.yaml`:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training"
  train_split: "train"
  eval_split: "validation"
  instruction_column: "instruction"
  response_column: "response"
```

## Dataset Structure

The dataset should contain crisis scenario training data with:

- **Instructions**: Crisis scenarios, emergency situations, or questions requiring structured responses
- **Responses**: Structured JSON responses with actions, priorities, reasoning, and resources

### Expected Format

```json
{
  "instruction": "A building is on fire with people trapped inside. What should be done?",
  "response": "{\"action\": \"evacuate immediately\", \"priority\": \"critical\", \"reasoning\": \"Fire poses immediate danger\", \"resources\": [\"fire_department\", \"ambulance\"]}"
}
```

## Verifying Dataset Structure

Before training, verify your dataset structure:

```python
from datasets import load_dataset

dataset = load_dataset("ianktoo/crisis-response-training")
print(dataset)
print(dataset["train"][0])  # Check first sample
print(dataset["train"].features)  # Check column names
```

## Customizing Dataset Configuration

If your dataset has different column names, update `configs/dataset_config.yaml`:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training"
  instruction_column: "prompt"      # If your dataset uses "prompt" instead of "instruction"
  response_column: "answer"         # If your dataset uses "answer" instead of "response"
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
  hf_dataset_name: "ianktoo/crisis-response-training"
  revision: "v1.2"  # Specific version
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{crisis_response_training_2026,
  title = {Crisis Response Training Dataset},
  author = {Ian K. T.},
  year = {2026},
  url = {https://huggingface.co/datasets/ianktoo/crisis-response-training},
  note = {Synthetic dataset for training crisis response language models}
}
```

## Related Resources

- **Dataset Repository**: https://huggingface.co/datasets/ianktoo/crisis-response-training
- **Dataset Setup Guide**: [dataset-setup.md](dataset-setup.md)
- **Main README**: [../README.md](../README.md)
