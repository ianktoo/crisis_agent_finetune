# Crisis-Agent Fine-Tuning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](CHANGELOG.md)
[![GitHub](https://img.shields.io/badge/GitHub-ianktoo%2Fcrisis__agent__finetune-blue)](https://github.com/ianktoo/crisis_agent_finetune)

A complete end-to-end pipeline for fine-tuning Mistral-7B as a **crisis companion application** using crisis scenario datasets, Unsloth, LoRA, and Hugging Face.

> **Purpose**: This pipeline fine-tunes Mistral-7B to serve as a crisis companion that can provide structured, actionable responses to crisis scenarios. The model is trained on crisis-response datasets to assist in emergency situations.

> **Note**: Please see [LICENSE](LICENSE) for attribution requirements if you use this software.

## 🌟 Features

- **Production-grade structure** with clean separation of concerns
- **4-bit quantization** for efficient training on 16GB VRAM
- **LoRA adapters** for parameter-efficient fine-tuning
- **Comprehensive logging** with rotating file handlers
- **Error handling** with graceful recovery
- **JSON validation** for structured responses
- **Evaluation tools** for model quality assessment
- **Easy deployment** with merged model support

## 📋 Prerequisites

- Python 3.10+ (tested with Python 3.10.12)
- CUDA-capable GPU (16GB+ VRAM recommended)
- Hugging Face account (for dataset access)

> **Note**: Python 3.10.12 is fully supported. The code uses features available in Python 3.9+ (type hints, pathlib, etc.) and is compatible with Python 3.10.12.

> **Note**: This pipeline is optimized for **16GB VRAM servers** (e.g., Jupyter 16G Pytorch). See [docs/hardware-setup.md](docs/hardware-setup.md) for detailed hardware configuration and optimization tips.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use Makefile
make setup
```

### 2. Environment Setup

Set up environment variables (required for private datasets):

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Hugging Face token
# HF_TOKEN=your_token_here
```

Or set directly:
```bash
export HF_TOKEN="your_huggingface_token_here"  # Linux/macOS
# or
$env:HF_TOKEN="your_huggingface_token_here"     # Windows PowerShell
```

> **See [docs/environment-variables.md](docs/environment-variables.md) for complete environment variable documentation.**

### 3. Configuration

The dataset is pre-configured to use `ianktoo/crisis-response-training`. Verify the configuration:

- **`dataset_config.yaml`**: Dataset is already set. **Verify column names** match your dataset structure
- **`model_config.yaml`**: Adjust model and LoRA parameters if needed
- **`training_config.yaml`**: Configure training hyperparameters if needed

> **See [docs/dataset-setup.md](docs/dataset-setup.md) for detailed instructions on configuring Hugging Face datasets.**

### 3.5. Verify Setup (Recommended)

Run the verification script to check everything is ready:

```bash
python scripts/verify_setup.py
```

This will check:
- All dependencies are installed
- CUDA is available
- Configuration files are valid
- Dataset is configured correctly
- Required directories exist

### 4. Training

```bash
# Run training
python scripts/train.py

# Train with custom model name
python scripts/train.py --model-name "crisis_agent_v1.0"

# Or use Makefile
make train
```

> **See [docs/model-naming.md](docs/model-naming.md) for detailed guide on controlling output model names.**

### 5. Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# Or use Makefile
make evaluate
```

### 6. Merge LoRA (Optional)

```bash
# Merge LoRA weights into base model
python scripts/merge_lora.py --checkpoint outputs/checkpoints/final --output outputs/final_model

# Or use Makefile
make merge
```

### 7. Inference

```bash
# Interactive inference
python scripts/infer.py --checkpoint outputs/checkpoints/final

# Single prompt
python scripts/infer.py --checkpoint outputs/checkpoints/final --prompt "Your crisis scenario here"

# Or use Makefile
make infer
```

## 📁 Project Structure

```
crisis_agent_finetune/
│
├── configs/              # Configuration files
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── dataset_config.yaml
│
├── data/                 # Dataset cache
│   └── local_cache/
│
├── src/                  # Source code
│   ├── data/            # Dataset loading and formatting
│   ├── model/           # Model loading and LoRA
│   ├── training/        # Training and evaluation
│   └── utils/           # Utilities (logging, error handling)
│
├── scripts/              # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── merge_lora.py
│   └── infer.py
│
├── outputs/              # Outputs
│   ├── checkpoints/     # Training checkpoints
│   ├── logs/            # Log files
│   └── final_model/     # Merged model
│
├── docs/                 # Documentation
│   ├── hardware-setup.md
│   ├── dataset-setup.md
│   ├── environment-variables.md
│   └── testing.md
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── README.md        # Testing documentation
├── Makefile              # Automation
├── requirements.txt     # Dependencies
├── pytest.ini           # Pytest configuration
├── .env.example          # Environment variables template
├── LICENSE               # MIT License with attribution
├── CHANGELOG.md          # Version history and changes
└── README.md            # This file
```

## ⚙️ Configuration

### Dataset Configuration

Edit `configs/dataset_config.yaml`:

```yaml
dataset:
  hf_dataset_name: "ianktoo/crisis-response-training"  # Crisis scenario training dataset
  train_split: "train"
  eval_split: "validation"
  instruction_column: "instruction"
  response_column: "response"
  max_samples: -1  # -1 for all samples
```

> **Note**: The default dataset is configured to use `ianktoo/crisis-response-training`. Update the column names (`instruction_column`, `response_column`) to match your dataset structure.

### Model Configuration

Edit `configs/model_config.yaml`:

```yaml
model:
  model_name: "unsloth/Mistral-7B-Instruct-v0.2"
  load_in_4bit: true
  lora:
    r: 16  # LoRA rank
    lora_alpha: 32
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", ...]
```

### Training Configuration

Edit `configs/training_config.yaml`:

```yaml
training:
  output_dir: "outputs/checkpoints"
  final_model_name: "final"  # Custom name for final checkpoint
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  save_steps: 500
```

> **See [docs/model-naming.md](docs/model-naming.md) for detailed guide on controlling output model names.**

## 🔧 Usage Examples

### Full Pipeline

```bash
# Train, evaluate, and merge in one go
make pipeline
```

### Custom Training Run

```bash
# Basic training
python scripts/train.py

# With custom model name
python scripts/train.py --model-name "crisis_agent_v1.0"

# With custom output directory
python scripts/train.py --output-dir "outputs/my_experiments"
```

### Evaluate with Custom Settings

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 200 \
  --output outputs/custom_eval_report.json
```

### Inference with JSON Validation

```bash
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "A building is on fire with people trapped inside" \
  --validate-json
```

## 📊 Evaluation Metrics

The evaluation script reports on the crisis companion's performance:

- **Valid JSON**: Percentage of responses with valid JSON structure
- **Valid Structure**: Percentage with correct crisis-response structure
- **Crisis Response Quality**: Validation of action, priority, reasoning, and resources
- **Safety Alignment**: Basic safety checks on responses
- **Error Logging**: Detailed logs of invalid responses

## 🐛 Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Reduce `per_device_train_batch_size` in `training_config.yaml`
2. Reduce `max_seq_length` in `model_config.yaml`
3. Increase `gradient_accumulation_steps` in `training_config.yaml`

> **See [docs/hardware-setup.md](docs/hardware-setup.md) for detailed memory optimization and troubleshooting guide.**

### Dataset Loading Issues

- Verify your Hugging Face dataset name is correct
- Check that you have access to the dataset
- Ensure the column names match your configuration

### Model Loading Issues

- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check that you have enough VRAM (16GB+ recommended)
- Try reducing `max_seq_length` if memory is tight

## 📝 Logging

Logs are automatically saved to `outputs/logs/` with:

- Rotating file handlers (10MB max, 5 backups)
- Console output for real-time monitoring
- Detailed error traces

## 🔒 Error Handling

The pipeline includes comprehensive error handling:

- **Dataset errors**: Graceful handling of malformed records
- **CUDA errors**: OOM detection and helpful suggestions
- **Training errors**: Checkpoint recovery on interruption
- **Validation errors**: JSON validation with detailed reports

## 🧪 Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only

# Or use Makefile
make test                   # Run all tests
make test-cov              # Run with coverage
make test-unit             # Unit tests only
make test-integration      # Integration tests only
```

See [docs/testing.md](docs/testing.md) for comprehensive testing documentation, or [tests/README.md](tests/README.md) for a quick reference.

## 📚 Additional Resources

- [LICENSE](LICENSE) - MIT License with attribution requirements
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
- [docs/hardware-setup.md](docs/hardware-setup.md) - Detailed hardware configuration and optimization
- [docs/dataset-setup.md](docs/dataset-setup.md) - Complete guide for configuring Hugging Face datasets
- [docs/dataset-info.md](docs/dataset-info.md) - Information about the crisis-response-training dataset
- [docs/environment-variables.md](docs/environment-variables.md) - Environment variables configuration
- [docs/model-naming.md](docs/model-naming.md) - Guide for controlling output model names
- [docs/testing.md](docs/testing.md) - Comprehensive testing documentation
- [tests/README.md](tests/README.md) - Test suite structure and quick reference
- [GitHub Repository](https://github.com/ianktoo/crisis_agent_finetune) - Source code and issues
- [Dataset Repository](https://huggingface.co/datasets/ianktoo/crisis-response-training) - Training dataset
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

This is a complete pipeline ready for use. Feel free to customize:

- Add custom evaluation metrics for crisis response quality
- Implement additional safety checks for emergency scenarios
- Add experiment tracking (Weights & Biases, MLflow)
- Extend to other base models
- Improve crisis scenario handling and response structure

## 🔗 Links

- **GitHub Repository**: [ianktoo/crisis_agent_finetune](https://github.com/ianktoo/crisis_agent_finetune)
- **Training Dataset**: [ianktoo/crisis-response-training](https://huggingface.co/datasets/ianktoo/crisis-response-training)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Attribution

If you use this software in your research, publications, or projects, please include attribution as specified in the [LICENSE](LICENSE) file.

For academic publications, please cite:

**Pipeline:**
```bibtex
@software{crisis_agent_finetune,
  title = {Crisis-Agent Fine-Tuning Pipeline},
  author = {ianktoo},
  year = {2026},
  url = {https://github.com/ianktoo/crisis_agent_finetune},
  note = {A complete end-to-end pipeline for fine-tuning Mistral-7B as a crisis companion application using crisis scenario datasets}
}
```

**Dataset:**
```bibtex
@dataset{crisis_response_training_2026,
  title = {Crisis Response Training Dataset},
  author = {Ian K. T.},
  year = {2026},
  url = {https://huggingface.co/datasets/ianktoo/crisis-response-training},
  note = {Synthetic dataset for training crisis response language models}
}
```

## 🎯 Next Steps

1. **Verify dataset configuration** in `configs/dataset_config.yaml` (default: `ianktoo/crisis-response-training`)
2. **Check column names** match your dataset structure (`instruction_column`, `response_column`)
3. **Adjust hyperparameters** based on your GPU memory
4. **Run training** with `python scripts/train.py`
5. **Evaluate** your crisis companion model
6. **Deploy** using the merged model or LoRA checkpoint

> **🚀 Ready to deploy?** See [DEPLOYMENT.md](DEPLOYMENT.md) for a complete deployment checklist and quick start guide.

## 📦 Dataset

This pipeline is configured to use the **Crisis Response Training Dataset**:

- **Hugging Face**: [ianktoo/crisis-response-training](https://huggingface.co/datasets/ianktoo/crisis-response-training)
- **Purpose**: Training data for crisis companion application
- **Format**: Instruction-response pairs for crisis scenarios
- **Type**: Synthetic dataset for training crisis response language models

### Dataset Citation

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

Update `configs/dataset_config.yaml` if you need to use a different dataset or adjust column mappings.

---

**Happy fine-tuning! 🚀**
