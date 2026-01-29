# Deployment Guide

Deploy your trained model to HuggingFace Hub, LM Studio, or Ollama.

---

## Deployment Options

| Option | Use Case | Size | Requirements |
|--------|----------|------|--------------|
| **HuggingFace (LoRA)** | Cloud deployment, sharing | ~100MB | HF account |
| **HuggingFace (Merged)** | Standalone deployment | ~14GB | HF account |
| **LM Studio** | Local desktop app | ~4-8GB | LM Studio app |
| **Ollama** | Local CLI/API server | ~4-8GB | Ollama installed |

---

## Merge LoRA Weights

Before deployment, optionally merge LoRA weights into base model:

```bash
# Using Makefile
make merge

# Or directly
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/final_model
```

**Output:** `outputs/final_model/` (~14GB)

### When to Merge

| Scenario | Merge? | Reason |
|----------|--------|--------|
| Sharing on HF | Optional | LoRA is smaller |
| LM Studio | Yes (via GGUF) | Needs standalone |
| Ollama | Yes (via GGUF) | Needs standalone |
| API deployment | Optional | Depends on framework |

---

## HuggingFace Deployment

### Setup

1. **Create HuggingFace account**: https://huggingface.co/join
2. **Get write token**: https://huggingface.co/settings/tokens
3. **Add to `.env`**:
   ```bash
   HF_TOKEN=hf_your_write_token_here
   ```

### Upload LoRA Checkpoint (Recommended)

Smaller (~100MB), requires base model for inference:

```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name YOUR_USERNAME/crisis-agent-v1 \
  --private
```

### Upload Merged Model

Standalone (~14GB), no base model needed:

```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name YOUR_USERNAME/crisis-agent-v1 \
  --merged \
  --private
```

### Using Makefile

```bash
make upload-hf CHECKPOINT=outputs/checkpoints/final REPO=username/crisis-agent-v1
```

### Load from HuggingFace

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "YOUR_USERNAME/crisis-agent-v1",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

---

## Export to GGUF

GGUF format is required for LM Studio and Ollama.

### Note: “noexec” mounts (important)

Some training environments mount the project/work volume with **`noexec`**, which prevents running compiled binaries from that filesystem. GGUF export needs `llama.cpp/llama-quantize`, so you may see errors like:
- `No working quantizer found in llama.cpp`
- `Permission denied`

This repo’s `scripts/export_gguf.py` supports running llama.cpp tools from an executable directory. Recommended:

```bash
CRISIS_GGUF_EXEC_DIR=/home/jovyan make export-gguf
```

### Quick Export

```bash
# Default (q4_k_m quantization)
make export-gguf

# Higher quality for LM Studio
make export-lmstudio

# With Ollama setup
make export-ollama
```

### Custom Export

```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  --quantization q8_0
```

### Quantization Options

| Method | Size (7B) | Quality | Use Case |
|--------|-----------|---------|----------|
| `q4_k_m` | ~4GB | Good | **Recommended** |
| `q5_k_m` | ~5GB | Better | Balanced |
| `q8_0` | ~8GB | High | Quality priority |
| `q3_k_m` | ~3GB | Lower | Size priority |
| `f16` | ~14GB | Full | Maximum quality |

### List Quantizations

```bash
python scripts/export_gguf.py --list-quantizations
```

---

## LM Studio Deployment

### Step 1: Export to GGUF

```bash
make export-lmstudio
```

Creates: `outputs/gguf/*.gguf`

### Step 2: Import into LM Studio

**Option A: CLI Import**
```bash
lms import outputs/gguf/model-q8_0.gguf
```

**Option B: Manual Import**
```bash
# Copy to LM Studio models folder
cp outputs/gguf/*.gguf ~/.lmstudio/models/crisis-agent/
```

### Step 3: Use in LM Studio

1. Open LM Studio
2. Go to **Chat** tab
3. Click **model selector** at top
4. Select your imported model
5. Start chatting!

### Step 4: Serve as API (Optional)

1. Load model in LM Studio
2. Go to **Developer** tab
3. Click **Start Server**
4. Use API at `http://localhost:1234/v1`

**Python client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # Can be anything
)

response = client.chat.completions.create(
    model="crisis-agent",
    messages=[
        {"role": "user", "content": "A building is on fire"}
    ]
)
print(response.choices[0].message.content)
```

---

## Ollama Deployment

### Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.ai
```

### Step 2: Export with Ollama Setup

```bash
make export-ollama
```

This creates:
- `outputs/gguf/*.gguf` - Model file
- `outputs/gguf/Modelfile` - Ollama configuration

### Step 3: Register Model

If Ollama is installed, the export script auto-registers. Otherwise:

```bash
cd outputs/gguf
ollama create crisis-agent -f Modelfile
```

### Step 4: Run Model

```bash
# Interactive chat
ollama run crisis-agent

# With specific prompt
ollama run crisis-agent "A building is on fire with people trapped"
```

### Step 5: Use as API

```bash
# Start Ollama server (if not running)
ollama serve

# API endpoint
curl http://localhost:11434/api/generate -d '{
  "model": "crisis-agent",
  "prompt": "A building is on fire"
}'
```

### Custom Modelfile

Edit `outputs/gguf/Modelfile` to customize:

```dockerfile
FROM model-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """You are AI Emergency Kit, a crisis response assistant..."""
```

---

## Push GGUF to HuggingFace

Share your GGUF model:

```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --push-to-hub YOUR_USERNAME/crisis-agent-gguf \
  --quantization q4_k_m

# Or with Makefile
make upload-gguf CHECKPOINT=outputs/checkpoints/final REPO=username/crisis-agent-gguf
```

---

## Deployment Checklist

### Before Deployment

- [ ] Training completed successfully
- [ ] Evaluation results are acceptable (>80% valid)
- [ ] Manual testing shows good responses
- [ ] Model name/version decided

### HuggingFace

- [ ] HF account created
- [ ] Write token generated
- [ ] Token added to `.env`
- [ ] Upload completed
- [ ] Model card updated
- [ ] Tested loading from HF

### Local (LM Studio/Ollama)

- [ ] GGUF exported
- [ ] Quantization chosen
- [ ] Model imported/registered
- [ ] Tested locally
- [ ] API working (if needed)

---

## Troubleshooting

### Upload Fails: Authentication

```bash
# Check token
echo $HF_TOKEN

# Login via CLI
huggingface-cli login
```

### GGUF Export Fails

```bash
# Check CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Try with smaller quantization
python scripts/export_gguf.py --checkpoint outputs/checkpoints/final -q q4_k_m
```

### Ollama: Model Not Found

```bash
# List models
ollama list

# Re-create model
cd outputs/gguf
ollama create crisis-agent -f Modelfile
```

### LM Studio: Gibberish Output

This is usually a prompt template mismatch:
1. Go to model settings in LM Studio
2. Set correct prompt template (Mistral format)
3. Reload and try again

See [Troubleshooting](Troubleshooting.md) for more solutions.

---

## Quick Reference

| Task | Command |
|------|---------|
| Merge LoRA | `make merge` |
| Export GGUF (default) | `make export-gguf` |
| Export for LM Studio | `make export-lmstudio` |
| Export for Ollama | `make export-ollama` |
| Upload to HF | `make upload-hf CHECKPOINT=... REPO=...` |
| Upload GGUF to HF | `make upload-gguf CHECKPOINT=... REPO=...` |

---

[← Evaluation](Evaluation.md) | [Home](Home.md) | [Makefile Reference →](Makefile-Reference.md)
