# Documentation

Single entry point for all crisis-agent fine-tuning pipeline documentation.

---

## Where to start

| If you want to… | Start here |
|-----------------|------------|
| **Follow a full workflow** (install → train → evaluate → deploy) | [Wiki → Getting Started](../wiki/Getting-Started.md) |
| **Quick reference** (commands, scripts, config) | [Wiki Home](../wiki/Home.md) |
| **Set up environment and data** | [Setup](#setup) below |
| **Train and name models** | [Training & model](#training--model) below |
| **Export and deploy** (Ollama, LM Studio, Hugging Face) | [Deployment & export](#deployment--export) below |
| **Evaluate models** | [Evaluation](#evaluation) below |
| **Fix issues** | [Wiki → Troubleshooting](../wiki/Troubleshooting.md) |

---

## Wiki (workflow and reference)

The wiki is the main narrative documentation:

| Page | Description |
|------|-------------|
| [Home](../wiki/Home.md) | Overview and quick navigation |
| [Getting Started](../wiki/Getting-Started.md) | Installation and first run |
| [Configuration](../wiki/Configuration.md) | Config files (dataset, model, training) |
| [Training](../wiki/Training.md) | Training workflow and options |
| [Evaluation](../wiki/Evaluation.md) | Standard and AI-based evaluation |
| [Deployment](../wiki/Deployment.md) | Hugging Face, LM Studio, Ollama |
| [Makefile Reference](../wiki/Makefile-Reference.md) | All `make` targets |
| [Scripts Reference](../wiki/Scripts-Reference.md) | Script usage and options |
| [Troubleshooting](../wiki/Troubleshooting.md) | Common issues and fixes |

---

## Setup

Environment, hardware, and data.

| Doc | Description |
|-----|-------------|
| [hardware-setup.md](hardware-setup.md) | Server specs, 16GB VRAM tuning, OOM tips |
| [environment-variables.md](environment-variables.md) | `.env`, HF token, AI eval keys |
| [dataset-setup.md](dataset-setup.md) | Hugging Face dataset and column config |
| [dataset-info.md](dataset-info.md) | Crisis-response dataset overview |
| [DATASET_OPTIONS.md](DATASET_OPTIONS.md) | Dataset format and source options |
| [llama-cpp-setup.md](llama-cpp-setup.md) | Build llama.cpp for GGUF export |
| [python-compatibility.md](python-compatibility.md) | Supported Python versions |

---

## Training & model

Training, checkpoints, and naming.

| Doc | Description |
|-----|-------------|
| [model-naming.md](model-naming.md) | Checkpoint and model naming, versioning |
| [model-optimization.md](model-optimization.md) | Shrinking models for Ollama/LM Studio |
| [export-naming.md](export-naming.md) | GGUF filenames and file-size display |

---

## Deployment & export

Exporting and deploying (GGUF, Ollama, LM Studio, Hugging Face).

| Doc | Description |
|-----|-------------|
| [model-optimization.md](model-optimization.md) | Quantization and size for local run |
| [export-naming.md](export-naming.md) | Export filenames and listing exports |
| [llama-cpp-setup.md](llama-cpp-setup.md) | Prerequisite for GGUF export |
| [Wiki → Deployment](../wiki/Deployment.md) | Full deployment flow |

---

## Evaluation

| Doc | Description |
|-----|-------------|
| [ai-evaluation.md](ai-evaluation.md) | AI-based evaluation (Claude, OpenAI, Gemini) |
| [evaluation-timing.md](evaluation-timing.md) | When and how evaluation runs |
| [Wiki → Evaluation](../wiki/Evaluation.md) | Evaluation workflow |

---

## Reference

| Doc | Description |
|-----|-------------|
| [testing.md](testing.md) | Test suite, running tests, coverage |

---

## Root-level docs

These live in the project root; use them for checklists and short guides.

| File | Description |
|------|-------------|
| [README.md](../README.md) | Project overview and quick start |
| [QUICK_GUIDE.md](../QUICK_GUIDE.md) | Short post-training guide |
| [PIPELINE.md](../PIPELINE.md) | Pipeline checklist and progress |
| [POST_TRAINING.md](../POST_TRAINING.md) | Post-training steps and HF deployment |
| [DEPLOYMENT.md](../DEPLOYMENT.md) | Pre-training deployment checklist |
| [DEPLOYMENT_NOTES.md](../DEPLOYMENT_NOTES.md) | Deployment notes and target repo |
| [OPTIMIZATION_QUICKSTART.md](../OPTIMIZATION_QUICKSTART.md) | Quick model optimization for Ollama/LM Studio |
| [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md) | Training command reference |
| [MODEL_BRANDING.md](../MODEL_BRANDING.md) | AI Emergency Kit branding |
| [folder-structure.md](../folder-structure.md) | Project layout |
| [CONDA_SETUP.md](../CONDA_SETUP.md) | Conda environment setup |
| [ENVIRONMENT_SETUP.md](../ENVIRONMENT_SETUP.md) | Environment setup |
| [SETUP_COMPLETE.md](../SETUP_COMPLETE.md) | Setup completion checklist |
| [CUDA_INFO.md](../CUDA_INFO.md) | CUDA / GPU info |
| [CHANGELOG.md](../CHANGELOG.md) | Version history |

---

## Documentation map

```
project root
├── README.md                 # Start here (overview + quick start)
├── DOCUMENTATION.md          # This map (where to find docs)
├── QUICK_GUIDE.md            # Short post-training guide
├── PIPELINE.md               # Pipeline checklist
├── POST_TRAINING.md          # Post-training details
├── DEPLOYMENT.md             # Deployment checklist
├── OPTIMIZATION_QUICKSTART.md # Model size / Ollama & LM Studio
├── TRAINING_COMMANDS.md      # Training commands
├── MODEL_BRANDING.md         # Branding
├── folder-structure.md       # Project layout
├── CONDA_SETUP.md            # Conda
├── ENVIRONMENT_SETUP.md      # Env setup
├── SETUP_COMPLETE.md         # Setup checklist
├── CUDA_INFO.md              # CUDA
├── CHANGELOG.md              # History
│
├── docs/                     # Topic docs (you are here; all files in this dir)
│   ├── README.md             # Doc index (this file)
│   ├── hardware-setup.md, environment-variables.md, dataset-setup.md
│   ├── dataset-info.md, DATASET_OPTIONS.md, llama-cpp-setup.md
│   ├── python-compatibility.md, model-naming.md, model-optimization.md
│   ├── export-naming.md, ai-evaluation.md, evaluation-timing.md
│   └── testing.md
│
└── wiki/                     # Workflow & reference
    ├── Home.md
    ├── Getting-Started.md
    ├── Configuration.md
    ├── Training.md
    ├── Evaluation.md
    ├── Deployment.md
    ├── Makefile-Reference.md
    ├── Scripts-Reference.md
    └── Troubleshooting.md
```

*Sections above group these files by topic; all `docs/` files live in the same directory.*

---

## Quick links

- **Main README**: [../README.md](../README.md)
- **Wiki home**: [../wiki/Home.md](../wiki/Home.md)
- **Troubleshooting**: [../wiki/Troubleshooting.md](../wiki/Troubleshooting.md)
- **Changelog**: [../CHANGELOG.md](../CHANGELOG.md)
- **License**: [../LICENSE](../LICENSE)
