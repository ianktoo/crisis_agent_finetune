# Makefile Reference

Complete reference for all `make` commands in the pipeline.

---

## Quick Reference

| Category | Command | Description |
|----------|---------|-------------|
| **Setup** | `make setup` | Install dependencies |
| **Setup** | `make verify` | Verify installation |
| **Training** | `make train` | Start training |
| **Training** | `make pipeline` | Train → Evaluate → Merge |
| **Evaluation** | `make evaluate` | Standard evaluation |
| **Evaluation** | `make evaluate-ai` | AI evaluation (Claude) |
| **Evaluation** | `make evaluate-ai-openai` | AI evaluation (OpenAI) |
| **Evaluation** | `make evaluate-ai-gemini` | AI evaluation (Gemini) |
| **Testing** | `make infer` | Interactive testing |
| **Testing** | `make test` | Run test suite |
| **Testing** | `make test-cov` | Tests with coverage |
| **Deployment** | `make merge` | Merge LoRA weights |
| **Export** | `make export-gguf` | Export to GGUF (q4_k_m) |
| **Export** | `make export-lmstudio` | Export for LM Studio (q8_0) |
| **Export** | `make export-ollama` | Export and setup Ollama |
| **Upload** | `make upload-hf` | Upload to HuggingFace |
| **Upload** | `make upload-gguf` | Upload GGUF to HuggingFace |
| **Utility** | `make clean` | Clean output directories |
| **Help** | `make help` | Show all commands |

---

## Setup Commands

### `make setup`

Install all Python dependencies.

```bash
make setup
```

**Equivalent to:**
```bash
pip install -r requirements.txt
```

---

### `make verify`

Verify the setup is complete and working.

```bash
make verify
```

**Equivalent to:**
```bash
python scripts/verify_setup.py
```

**Checks:**
- Dependencies installed
- CUDA available
- Configuration files valid
- Dataset accessible
- Directories exist

---

## Training Commands

### `make train`

Start model training with default settings.

```bash
make train
```

**Equivalent to:**
```bash
python scripts/train.py
```

**Options (via scripts):**
```bash
python scripts/train.py --model-name "custom_name"
python scripts/train.py --output-dir "custom/path"
python scripts/train.py --resume-from "checkpoint/path"
```

---

### `make pipeline`

Run the complete pipeline: Train → Evaluate → Merge.

```bash
make pipeline
```

**Equivalent to:**
```bash
make train && make evaluate && make merge
```

---

## Evaluation Commands

### `make evaluate`

Run standard evaluation (structure and JSON validation).

```bash
make evaluate
```

**Equivalent to:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final
```

**Options (via scripts):**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 200
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --output custom_report.json
```

---

### `make evaluate-ai`

Run AI-based evaluation using Claude (default).

```bash
make evaluate-ai
```

**Equivalent to:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai
```

**Requires:** `ANTHROPIC_API_KEY` in `.env`

---

### `make evaluate-ai-openai`

Run AI-based evaluation using OpenAI.

```bash
make evaluate-ai-openai
```

**Equivalent to:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai
```

**Requires:** `OPENAI_API_KEY` in `.env`

---

### `make evaluate-ai-gemini`

Run AI-based evaluation using Google Gemini.

```bash
make evaluate-ai-gemini
```

**Equivalent to:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
```

**Requires:** `GEMINI_API_KEY` in `.env`

---

## Testing Commands

### `make infer`

Start interactive inference mode.

```bash
make infer
```

**Equivalent to:**
```bash
python scripts/infer.py --checkpoint outputs/checkpoints/final
```

**Options (via scripts):**
```bash
python scripts/infer.py --checkpoint outputs/checkpoints/final --prompt "Your prompt"
python scripts/infer.py --checkpoint outputs/checkpoints/final --validate-json
```

---

### `make test`

Run the full test suite.

```bash
make test
```

**Equivalent to:**
```bash
pytest tests/ -v
```

---

### `make test-cov`

Run tests with coverage report.

```bash
make test-cov
```

**Equivalent to:**
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

**Output:** `htmlcov/index.html`

---

### `make test-unit`

Run only unit tests.

```bash
make test-unit
```

**Equivalent to:**
```bash
pytest tests/unit/ -v
```

---

### `make test-integration`

Run only integration tests.

```bash
make test-integration
```

**Equivalent to:**
```bash
pytest tests/integration/ -v
```

---

## Deployment Commands

### `make merge`

Merge LoRA weights into base model for deployment.

```bash
make merge
```

**Equivalent to:**
```bash
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/final_model
```

**Output:** `outputs/final_model/` (~14GB)

---

## Export Commands

### `make export-gguf`

Export to GGUF format with q4_k_m quantization.

```bash
make export-gguf
```

**Equivalent to:**
```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf
```

**Output:** `outputs/gguf/*.gguf` (~4GB)

---

### `make export-lmstudio`

Export to GGUF for LM Studio with q8_0 (higher quality).

```bash
make export-lmstudio
```

**Equivalent to:**
```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q8_0
```

**Output:** `outputs/gguf/*.gguf` (~8GB)

---

### `make export-ollama`

Export to GGUF and create Ollama Modelfile.

```bash
make export-ollama
```

**Equivalent to:**
```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  --ollama
```

**Output:**
- `outputs/gguf/*.gguf`
- `outputs/gguf/Modelfile`

---

### `make export-gguf-custom`

Export with custom settings.

```bash
make export-gguf-custom CHECKPOINT=outputs/checkpoints/final QUANT=q8_0 OUTPUT=outputs/my_gguf
```

**Parameters:**
- `CHECKPOINT` (required): Path to checkpoint
- `QUANT` (optional): Quantization method (default: q4_k_m)
- `OUTPUT` (optional): Output directory (default: outputs/gguf)
- `OLLAMA` (optional): Include Ollama Modelfile

**Example:**
```bash
make export-gguf-custom CHECKPOINT=outputs/checkpoints/final QUANT=q5_k_m OLLAMA=1
```

---

## Upload Commands

### `make upload-hf`

Upload model to HuggingFace Hub.

```bash
make upload-hf CHECKPOINT=outputs/checkpoints/final REPO=username/repo-name
```

**Parameters:**
- `CHECKPOINT` (required): Path to checkpoint or merged model
- `REPO` (required): HuggingFace repo (format: username/repo-name)

**Equivalent to:**
```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name username/repo-name
```

**Requires:** `HF_TOKEN` in `.env`

---

### `make upload-gguf`

Push GGUF model directly to HuggingFace Hub.

```bash
make upload-gguf CHECKPOINT=outputs/checkpoints/final REPO=username/model-gguf
```

**Parameters:**
- `CHECKPOINT` (required): Path to checkpoint
- `REPO` (required): HuggingFace repo for GGUF
- `QUANT` (optional): Quantization method (default: q4_k_m)

**Equivalent to:**
```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --push-to-hub username/model-gguf \
  -q q4_k_m
```

**Requires:** `HF_TOKEN` in `.env`

---

## Utility Commands

### `make clean`

Clean all output directories.

```bash
make clean
```

**Removes:**
- `outputs/checkpoints/*`
- `outputs/logs/*`
- `outputs/final_model/*`

**Warning:** This deletes trained models! Backup first if needed.

---

### `make help`

Show all available commands.

```bash
make help
```

**Output:**
```
Crisis-Agent Fine-Tuning Pipeline
==================================

Available targets:
  make setup            - Install dependencies
  make train            - Run training
  make evaluate         - Evaluate model (standard)
  make evaluate-ai      - Evaluate with AI (Claude)
  ...
```

---

## Common Workflows

### First Training Run

```bash
source activate_env.sh
make verify
make train
make evaluate
make infer
```

### Full Pipeline to HuggingFace

```bash
make train
make evaluate
make merge
make upload-hf CHECKPOINT=outputs/final_model REPO=username/crisis-agent-v1
```

### Export for Local Use

```bash
make train
make evaluate
make export-ollama
ollama run crisis-agent
```

### Development Iteration

```bash
# Quick test with limited data
python scripts/train.py --model-name test_run
make evaluate
make infer
make clean  # Clean up
```

---

## Environment Variables

These affect Makefile command behavior:

| Variable | Purpose | Used By |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace authentication | upload commands |
| `ANTHROPIC_API_KEY` | Claude API | `evaluate-ai` |
| `OPENAI_API_KEY` | OpenAI API | `evaluate-ai-openai` |
| `GEMINI_API_KEY` | Gemini API | `evaluate-ai-gemini` |

---

## Tips

### Run Multiple Commands

```bash
# Sequential
make train && make evaluate && make merge

# Or use pipeline
make pipeline
```

### Dry Run

To see what a command does without running:

```bash
make -n train  # Shows commands without executing
```

### Verbose Output

```bash
make train VERBOSE=1  # More detailed output
```

---

[← Deployment](Deployment.md) | [Home](Home.md) | [Scripts Reference →](Scripts-Reference.md)
