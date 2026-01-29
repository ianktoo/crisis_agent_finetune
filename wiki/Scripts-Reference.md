# Scripts Reference

Detailed documentation for all executable scripts.

---

## Scripts Overview

| Script | Purpose | Location |
|--------|---------|----------|
| `train.py` | Model training | `scripts/train.py` |
| `evaluate.py` | Model evaluation | `scripts/evaluate.py` |
| `infer.py` | Interactive inference | `scripts/infer.py` |
| `merge_lora.py` | Merge LoRA weights | `scripts/merge_lora.py` |
| `export_gguf.py` | Export to GGUF format | `scripts/export_gguf.py` |
| `upload_to_hf.py` | Upload to HuggingFace | `scripts/upload_to_hf.py` |
| `verify_setup.py` | Verify installation | `scripts/verify_setup.py` |

---

## train.py

Train the crisis response model with LoRA fine-tuning.

### Usage

```bash
python scripts/train.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-name` | string | `"final"` | Name for final checkpoint |
| `--output-dir` | string | config value | Output directory |
| `--resume-from` | string | None | Checkpoint to resume from |
| `--config` | string | `"configs/training_config.yaml"` | Config file path |

### Examples

```bash
# Basic training
python scripts/train.py

# Custom model name
python scripts/train.py --model-name "crisis_agent_v2"

# Custom output directory
python scripts/train.py --output-dir "outputs/experiments/exp1"

# Resume from checkpoint
python scripts/train.py --resume-from outputs/checkpoints/checkpoint-500
```

### Output

- `outputs/checkpoints/` - Training checkpoints
- `outputs/checkpoints/final/` - Final model
- `outputs/logs/training.log` - Training logs

---

## evaluate.py

Evaluate trained model quality.

### Usage

```bash
python scripts/evaluate.py --checkpoint PATH [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | string | Required | Path to model checkpoint |
| `--max-samples` | int | 100 | Max samples to evaluate |
| `--output` | string | `outputs/evaluation_report.json` | Report output path |
| `--ai` | flag | False | Enable AI-based evaluation |
| `--ai-provider` | string | `"anthropic"` | AI provider (anthropic/openai/gemini) |
| `--ai-model` | string | Provider default | Specific model to use |
| `--ai-max-samples` | int | All | Limit AI evaluation samples |

### Examples

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# With more samples
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 200

# AI evaluation (Claude)
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai

# AI evaluation (OpenAI)
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai

# AI evaluation (Gemini)
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini

# Limit AI evaluation costs
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-max-samples 50

# Custom AI model
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-model claude-3-opus-20240229
```

### Output

`outputs/evaluation_report.json`:
```json
{
  "total_samples": 100,
  "valid_json": 85,
  "valid_json_percentage": 85.0,
  "valid_structured_text": 95,
  "ai_evaluation": {
    "enabled": true,
    "average_score": 85.3,
    "criterion_averages": {...}
  }
}
```

---

## infer.py

Interactive inference and testing.

### Usage

```bash
python scripts/infer.py --checkpoint PATH [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | string | Required | Path to model checkpoint |
| `--prompt` | string | None | Single prompt (non-interactive) |
| `--validate-json` | flag | False | Validate JSON output |
| `--max-new-tokens` | int | 512 | Max tokens to generate |
| `--temperature` | float | 0.7 | Generation temperature |

### Examples

```bash
# Interactive mode
python scripts/infer.py --checkpoint outputs/checkpoints/final

# Single prompt
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "A building is on fire with people trapped"

# With JSON validation
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "Medical emergency at workplace" \
  --validate-json

# Adjust generation
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --temperature 0.5 \
  --max-new-tokens 1024
```

### Interactive Mode

```
Crisis Agent Inference
======================
Type 'quit' to exit

Enter prompt: A building is on fire
[Response generated...]

Enter prompt: quit
Goodbye!
```

---

## merge_lora.py

Merge LoRA adapters into base model.

### Usage

```bash
python scripts/merge_lora.py --checkpoint PATH [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | string | Required | Path to LoRA checkpoint |
| `--output` | string | `outputs/final_model` | Output path |
| `--max-seq-length` | int | 2048 | Max sequence length |

### Examples

```bash
# Default merge
python scripts/merge_lora.py --checkpoint outputs/checkpoints/final

# Custom output
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/merged_v1
```

### Output

`outputs/final_model/`:
- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

---

## export_gguf.py

Export model to GGUF format for LM Studio and Ollama.

### Usage

```bash
python scripts/export_gguf.py --checkpoint PATH [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | string | `outputs/checkpoints/final` | Model checkpoint path |
| `--output`, `-o` | string | `outputs/gguf` | Output directory |
| `--quantization`, `-q` | string | `q4_k_m` | Quantization method |
| `--max-seq-length` | int | 2048 | Max sequence length |
| `--ollama` | flag | False | Create Ollama Modelfile |
| `--ollama-name` | string | `crisis-agent` | Ollama model name |
| `--push-to-hub` | string | None | Push to HuggingFace (repo name) |
| `--private` | flag | False | Make HF repo private |
| `--list-quantizations` | flag | False | List quantization methods |

### Quantization Methods

| Method | Size | Quality | Description |
|--------|------|---------|-------------|
| `q4_k_m` | ~4GB | Good | **Recommended** - best balance |
| `q5_k_m` | ~5GB | Better | Quality/size balance |
| `q6_k` | ~6GB | Good+ | 6-bit quantization |
| `q8_0` | ~8GB | High | Higher quality |
| `q3_k_m` | ~3GB | Lower | Smallest practical |
| `q2_k` | ~2GB | Low | Experimental |
| `f16` | ~14GB | Full | Float16 precision |
| `f32` | ~28GB | Full | Float32 precision |

### Examples

```bash
# Default export (q4_k_m)
python scripts/export_gguf.py --checkpoint outputs/checkpoints/final

# Higher quality
python scripts/export_gguf.py --checkpoint outputs/checkpoints/final -q q8_0

# For Ollama
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --ollama \
  --ollama-name my-crisis-agent

# Push to HuggingFace
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --push-to-hub username/crisis-agent-gguf \
  -q q4_k_m

# List available quantizations
python scripts/export_gguf.py --list-quantizations
```

### Output

`outputs/gguf/`:
- `model-{quantization}.gguf` - GGUF model file
- `Modelfile` - Ollama configuration (if `--ollama`)

---

## upload_to_hf.py

Upload model to HuggingFace Hub.

### Usage

```bash
python scripts/upload_to_hf.py --checkpoint PATH --repo-name REPO [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | string | Required | Path to model |
| `--repo-name` | string | Required | HF repo (username/repo) |
| `--private` | flag | False | Make repo private |
| `--merged` | flag | False | Mark as merged model |
| `--commit-message` | string | Auto | Custom commit message |

### Examples

```bash
# Upload LoRA checkpoint
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name username/crisis-agent-v1

# Upload merged model
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name username/crisis-agent-v1 \
  --merged

# Private repository
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name username/crisis-agent-v1 \
  --private

# Custom commit message
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name username/crisis-agent-v1 \
  --commit-message "v1.0: Initial release"
```

### Requirements

- `HF_TOKEN` environment variable with write access
- HuggingFace account

---

## verify_setup.py

Verify the installation and setup is complete.

### Usage

```bash
python scripts/verify_setup.py
```

### Checks Performed

1. **Dependencies** - All packages installed
2. **CUDA** - GPU available
3. **Configuration** - Config files valid
4. **Dataset** - Dataset accessible
5. **Directories** - Required directories exist

### Output

```
Verifying Crisis-Agent Fine-Tuning Setup
========================================

[1/5] Checking dependencies...
  ✅ torch: 2.10.0+cu128
  ✅ transformers: 4.47.0
  ✅ unsloth: 2026.1.4
  ✅ All dependencies installed

[2/5] Checking CUDA...
  ✅ CUDA available
  ✅ GPU: NVIDIA A100 (16GB)

[3/5] Checking configuration files...
  ✅ configs/dataset_config.yaml
  ✅ configs/model_config.yaml
  ✅ configs/training_config.yaml

[4/5] Checking dataset access...
  ✅ Dataset accessible: ianktoo/crisis-response-training-v2

[5/5] Checking directories...
  ✅ outputs/checkpoints
  ✅ outputs/logs

========================================
✅ All checks passed! Ready to train.
```

---

## Source Modules

### src/data/

| Module | Purpose |
|--------|---------|
| `load_dataset.py` | Dataset loading from HF or local |
| `format_records.py` | Data formatting and tokenization |

### src/model/

| Module | Purpose |
|--------|---------|
| `load_model.py` | Model loading with quantization |
| `apply_lora.py` | LoRA adapter application |

### src/training/

| Module | Purpose |
|--------|---------|
| `trainer.py` | Training loop and checkpointing |
| `evaluation.py` | Standard evaluation metrics |
| `ai_evaluation.py` | AI-based evaluation |

### src/utils/

| Module | Purpose |
|--------|---------|
| `logging.py` | Logging configuration |
| `error_handling.py` | Error handling utilities |
| `json_validator.py` | JSON validation |

---

[← Makefile Reference](Makefile-Reference.md) | [Home](Home.md) | [Troubleshooting →](Troubleshooting.md)
