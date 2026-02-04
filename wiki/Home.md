# Crisis-Agent Fine-Tuning Pipeline Wiki

Welcome to the **AI Emergency Kit** fine-tuning pipeline documentation wiki!

This wiki provides comprehensive documentation for training, evaluating, and deploying Mistral-7B as an intelligent crisis response assistant.

---

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](Getting-Started.md) | Installation, setup, and first training run |
| [Configuration](Configuration.md) | All configuration files explained |
| [Training Guide](Training.md) | Complete training workflow |
| [Evaluation](Evaluation.md) | Model evaluation including AI-based assessment |
| [Deployment](Deployment.md) | Deploy to HuggingFace, LM Studio, Ollama |
| [Makefile Reference](Makefile-Reference.md) | All make commands explained |
| [Scripts Reference](Scripts-Reference.md) | Detailed script documentation |
| [Troubleshooting](Troubleshooting.md) | Common issues and solutions |

---

## What is AI Emergency Kit?

**AI Emergency Kit** is a fine-tuned Mistral-7B model specialized for crisis response and emergency management. It provides:

- **Structured responses** with FACTS, UNCERTAINTIES, ANALYSIS, and GUIDANCE sections
- **Actionable advice** for emergency situations
- **JSON-formatted output** for integration with other systems

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Emergency Kit Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SETUP           2. TRAIN           3. EVALUATE                 │
│  ┌─────────┐       ┌─────────┐        ┌─────────┐                  │
│  │ make    │  -->  │ make    │  -->   │ make    │                  │
│  │ setup   │       │ train   │        │ evaluate│                  │
│  └─────────┘       └─────────┘        └─────────┘                  │
│                                              │                      │
│                                              v                      │
│  6. DEPLOY          5. EXPORT          4. TEST                     │
│  ┌─────────┐       ┌─────────┐        ┌─────────┐                  │
│  │ upload  │  <--  │ export  │  <--   │ make    │                  │
│  │ to HF   │       │ gguf    │        │ infer   │                  │
│  └─────────┘       └─────────┘        └─────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (5 Minutes)

```bash
# 1. Activate environment
source activate_env.sh

# 2. Verify setup
make verify

# 3. Start training
make train

# 4. Evaluate results
make evaluate

# 5. Test interactively
make infer

# 6. Export for local use
make export-ollama
```

---

## Key Commands

### Essential Commands

| Command | Description |
|---------|-------------|
| `source activate_env.sh` | Activate conda environment (auto-creates if needed) |
| `make train` | Start model training |
| `make evaluate` | Run standard evaluation |
| `make infer` | Interactive testing |
| `make merge` | Merge LoRA weights for deployment |

### Export Commands

| Command | Description |
|---------|-------------|
| `make export-gguf` | Export to GGUF (q4_k_m) |
| `make export-lmstudio` | Export for LM Studio (q8_0) |
| `make export-ollama` | Export and setup Ollama |

### Deployment Commands

| Command | Description |
|---------|-------------|
| `make upload-hf CHECKPOINT=... REPO=...` | Upload to HuggingFace |
| `make upload-gguf CHECKPOINT=... REPO=...` | Upload GGUF to HuggingFace |

---

## Project Structure

```
crisis_agent_finetune/
├── configs/                 # Configuration files
│   ├── dataset_config.yaml  # Dataset settings
│   ├── model_config.yaml    # Model and LoRA settings
│   └── training_config.yaml # Training hyperparameters
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── infer.py            # Inference script
│   ├── merge_lora.py       # LoRA merge script
│   ├── export_gguf.py      # GGUF export script
│   └── upload_to_hf.py     # HuggingFace upload
├── src/                    # Source code modules
│   ├── data/               # Data loading and formatting
│   ├── model/              # Model loading and LoRA
│   ├── training/           # Training and evaluation
│   └── utils/              # Utilities
├── outputs/                # Training outputs
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/               # Training logs
│   ├── final_model/        # Merged model
│   └── gguf/               # GGUF exports
├── wiki/                   # This documentation
├── docs/                   # Additional documentation
├── tests/                  # Test suite
├── Makefile                # Automation commands
├── activate_env.sh         # Environment activation
└── requirements.txt        # Dependencies
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12GB | 16GB+ |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |
| CPU | 2 cores | 4+ cores |

---

## Support & Resources

- **Main README**: [README.md](../README.md)
- **Documentation map**: [DOCUMENTATION.md](../DOCUMENTATION.md) — where to find everything
- **Documentation index**: [docs/README.md](../docs/README.md) — full index by topic (setup, training, deployment, evaluation)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)
- **License**: [LICENSE](../LICENSE) (MIT)
- **GitHub**: [ianktoo/crisis_agent_finetune](https://github.com/ianktoo/crisis_agent_finetune)
- **Dataset**: [ianktoo/crisis-response-training-v2](https://huggingface.co/datasets/ianktoo/crisis-response-training-v2)

---

## Wiki Pages

1. **[Getting Started](Getting-Started.md)** - Installation and initial setup
2. **[Configuration](Configuration.md)** - Configuration files reference
3. **[Training Guide](Training.md)** - Training workflow and options
4. **[Evaluation](Evaluation.md)** - Evaluation methods and AI assessment
5. **[Deployment](Deployment.md)** - HuggingFace, LM Studio, Ollama
6. **[Makefile Reference](Makefile-Reference.md)** - All make commands
7. **[Scripts Reference](Scripts-Reference.md)** - Script documentation
8. **[Troubleshooting](Troubleshooting.md)** - Common issues and fixes

---

*Last updated: January 2026*
