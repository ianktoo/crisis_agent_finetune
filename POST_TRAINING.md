# Post-Training Steps & Deployment Guide

This guide covers the steps to take **after training is complete** to evaluate, test, and deploy your **AI Emergency Kit** model to Hugging Face Hub.

> **Note**: These steps are designed as a separate sub-pipeline that won't interfere with any running training processes.

---

## 🚀 Quick Start Guide

**Just finished training? Follow these 4 simple steps:**

### Step 1: Check Your Model Works
```bash
make evaluate
```
This tests your model and shows how well it performs.

### Step 2: Test It Yourself
```bash
make infer
```
Type in a crisis scenario and see what the model responds.

### Step 3: Prepare for Sharing
```bash
make merge
```
This combines everything into one file (takes a few minutes).

### Step 4: Upload to Hugging Face
```bash
# First, get your token from: https://huggingface.co/settings/tokens
# Add it to .env file: HF_TOKEN=your_token_here

python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name your-username/crisis-agent-v1 \
  --private
```
Replace `your-username` with your Hugging Face username. The model will be automatically branded as **AI Emergency Kit** in the model card, regardless of repository name.

**That's it!** Your model is now on Hugging Face. 🎉

---

**Want more details?** Continue reading below for step-by-step instructions.

---

## 📋 Post-Training Checklist

After your training completes, follow these steps in order:

### Phase 1: Evaluation & Testing
- [ ] **Step 1**: Evaluate the trained model
  - [ ] Run evaluation script (`make evaluate`)
  - [ ] Review evaluation report (`outputs/evaluation_report.json`)
  - [ ] Check valid JSON percentage
  - [ ] Check valid structure percentage
  - [ ] Review error logs if any
- [ ] **Step 2**: Run test suite
  - [ ] Run unit tests (`make test-unit`)
  - [ ] Run integration tests (`make test-integration`)
  - [ ] Run full test suite (`make test`)
  - [ ] Verify all tests pass
- [ ] **Step 3**: Test inference manually
  - [ ] Test interactive mode (`make infer`)
  - [ ] Test single prompts with validation
  - [ ] Test multiple crisis scenarios
  - [ ] Verify JSON output format
  - [ ] Assess response quality
- [ ] **Step 4**: Compare with baseline (optional)
  - [ ] Test base model responses
  - [ ] Test fine-tuned model responses
  - [ ] Compare and document improvements

### Phase 2: Preparation & Deployment
- [ ] **Step 5**: Merge LoRA weights (for deployment)
  - [ ] Backup checkpoints (optional)
  - [ ] Run merge script (`make merge`)
  - [ ] Verify merged model in `outputs/final_model/`
  - [ ] Test merged model inference
- [ ] **Step 6**: Upload to Hugging Face Hub
  - [ ] Set up HF account and token
  - [ ] Add token to `.env` file
  - [ ] Choose repository name
  - [ ] Upload model (LoRA or merged)
  - [ ] Verify upload on Hugging Face
  - [ ] Create/update model card
  - [ ] Test loading model from HF
- [ ] **Step 7**: Export for Local Deployment (Optional)
  - [ ] Export to GGUF (`make export-gguf` or `make export-lmstudio`)
  - [ ] Test in LM Studio
  - [ ] Setup for Ollama (`make export-ollama`)
  - [ ] Test with `ollama run crisis-agent`

---

## Step 1: Evaluate the Model

Run the evaluation script to check model quality metrics:

```bash
# Evaluate your trained model
python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# Or use Makefile
make evaluate

# Evaluate with more samples
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 200
```

**What it checks:**
- Valid JSON output percentage
- Correct response structure
- Crisis response quality
- Error logging

**Expected output:**
- Evaluation report saved to `outputs/evaluation_report.json`
- Console summary with metrics

---

## Step 2: Run Test Suite

Verify all code still works correctly:

```bash
# Run all tests
pytest

# Or use Makefile
make test

# Run with coverage
make test-cov

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only
```

**What it checks:**
- Unit tests for all components
- Integration tests for full pipeline
- Code coverage

---

## Step 3: Test Inference Manually

Test the model with real crisis scenarios:

```bash
# Interactive mode
python scripts/infer.py --checkpoint outputs/checkpoints/final

# Single prompt test
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "A building is on fire with people trapped inside" \
  --validate-json

# Or use Makefile
make infer
```

**What to test:**
- Different crisis scenarios
- JSON output format
- Response quality and relevance
- Edge cases

---

## Step 4: Compare with Baseline (Optional)

Compare your fine-tuned model with the base model:

```bash
# Test base model
python scripts/infer.py \
  --checkpoint unsloth/Mistral-7B-Instruct-v0.2 \
  --prompt "Your test prompt here"

# Test fine-tuned model
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "Your test prompt here"
```

Compare outputs to verify improvement.

---

## Step 5: Merge LoRA Weights

Before deploying, merge LoRA weights into the base model:

```bash
# Merge LoRA weights
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/final_model

# Or use Makefile
make merge
```

**What this does:**
- Merges LoRA adapters into base model weights
- Creates a standalone model (no LoRA files needed)
- Saves to `outputs/final_model/`

**Note**: Merged models are larger but easier to deploy.

---

## Step 6: Upload to Hugging Face Hub

Upload your model to Hugging Face for sharing and deployment:

### Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co/join
2. **Access Token**: Get a write token from https://huggingface.co/settings/tokens
3. **Set Token**: Add to `.env` file:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```

### Upload Options

#### Option A: Upload LoRA Checkpoint (Smaller, Faster)

```bash
# Upload LoRA checkpoint (recommended for sharing)
# Repository name can be anything - model will be branded as "AI Emergency Kit"
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name your-username/crisis-agent-v1 \
  --private
```

#### Option B: Upload Merged Model (Standalone)

```bash
# Upload merged model (standalone, no base model needed)
# Repository name can be anything - model will be branded as "AI Emergency Kit"
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name your-username/crisis-agent-v1 \
  --private \
  --merged
```

### Upload Script Usage

```bash
python scripts/upload_to_hf.py \
  --checkpoint PATH_TO_MODEL \
  --repo-name USERNAME/REPO_NAME \
  [--private] \
  [--merged] \
  [--commit-message "Your commit message"]
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint or merged model
- `--repo-name`: Hugging Face repo name (format: `username/repo-name`)
- `--private`: Make repository private (optional)
- `--merged`: Indicate this is a merged model (optional)
- `--commit-message`: Custom commit message (optional)

**Example:**
```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name ianktoo/crisis-agent-mistral-7b \
  --private \
  --commit-message "Initial release: Crisis agent fine-tuned on crisis-response-training dataset"
```

### After Upload

1. **Verify Upload**: Check your model at `https://huggingface.co/USERNAME/REPO_NAME`
2. **Add Model Card**: Edit `README.md` on Hugging Face to add:
   - Model description
   - Training details
   - Usage examples
   - Evaluation metrics
3. **Test Loading**: Verify others can load it:
   ```python
   from unsloth import FastLanguageModel
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       "your-username/ai-emergency-kit",
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=True,
   )
   ```

---

## 📊 Full Pipeline Command

Run all post-training steps in sequence:

```bash
# 1. Evaluate
make evaluate

# 2. Run tests
make test

# 3. Merge LoRA
make merge

# 4. Upload to Hugging Face
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name your-username/ai-emergency-kit \
  --private
```

---

## 🎯 Quick Reference

| Step | Command | Output |
|------|---------|--------|
| Evaluate | `make evaluate` | `outputs/evaluation_report.json` |
| Test | `make test` | Test results |
| Merge | `make merge` | `outputs/final_model/` |
| Upload | `python scripts/upload_to_hf.py ...` | Hugging Face repo |
| Export GGUF | `make export-gguf` | `outputs/gguf/*.gguf` |
| Export Ollama | `make export-ollama` | `outputs/gguf/` + Modelfile |
| Export LM Studio | `make export-lmstudio` | `outputs/gguf/*.gguf` (q8_0) |

---

## ⚠️ Important Notes

1. **Don't interfere with training**: All post-training scripts are read-only on training checkpoints
2. **Backup checkpoints**: Consider backing up `outputs/checkpoints/` before merging
3. **Model size**: Merged models are ~14GB, LoRA checkpoints are ~100MB
4. **Token permissions**: Ensure your HF token has **write** access
5. **Repository naming**: Use lowercase, hyphens, no spaces (e.g., `crisis-agent-v1`). The model will be automatically branded as "AI Emergency Kit" in the model card regardless of repository name.

---

## 🐛 Troubleshooting

### Upload Fails: Authentication Error

```bash
# Verify token is set
echo $HF_TOKEN

# Or check .env file
cat .env | grep HF_TOKEN

# Login via CLI (alternative)
huggingface-cli login
```

### Upload Fails: Repository Already Exists

- Delete the repo on Hugging Face, or
- Use a different `--repo-name`

### Model Too Large for Upload

- Use LoRA checkpoint instead of merged model
- Or use Git LFS: `git lfs install` (handled automatically by upload script)

### Evaluation Shows Low Scores

- Review `outputs/evaluation_report.json` for detailed errors
- Check training logs for issues
- Consider retraining with adjusted hyperparameters

---

---

## Step 7: Export for Local Deployment (LM Studio / Ollama)

Export your model to GGUF format for running locally in LM Studio or Ollama.

### Export to GGUF (for LM Studio)

```bash
# Export with default settings (q4_k_m quantization)
make export-gguf

# Or export with higher quality (q8_0)
make export-lmstudio

# Custom export
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q8_0
```

**Available quantization methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `q4_k_m` | 4-bit (default) | Best balance of quality/size |
| `q8_0` | 8-bit | Higher quality, larger size |
| `q5_k_m` | 5-bit | Good quality/size balance |
| `q3_k_m` | 3-bit | Smallest, lower quality |
| `f16` | Float16 | Full precision |

### Using with LM Studio

1. Export your model:
   ```bash
   make export-lmstudio
   ```

2. Import into LM Studio:
   ```bash
   # Using LM Studio CLI
   lms import outputs/gguf/model-q8_0.gguf
   
   # Or manually copy to LM Studio models folder
   cp outputs/gguf/*.gguf ~/.lmstudio/models/crisis-agent/
   ```

3. Open LM Studio → Chat → Select your model → Start chatting!

4. (Optional) Serve as local API:
   ```bash
   # Start LM Studio server (in LM Studio Developer tab)
   # Or via CLI
   lms load crisis-agent --gpu=auto
   lms server start --port 1234
   ```

### Export and Setup for Ollama

```bash
# Export and create Ollama Modelfile
make export-ollama

# Or with custom name
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  --ollama \
  --ollama-name my-crisis-agent
```

**Using with Ollama:**

1. Export the model:
   ```bash
   make export-ollama
   ```

2. If Ollama is installed, the model is automatically registered. Run:
   ```bash
   ollama run crisis-agent
   ```

3. If Ollama wasn't installed during export, manually register:
   ```bash
   cd outputs/gguf
   ollama create crisis-agent -f Modelfile
   ollama run crisis-agent
   ```

### Push GGUF to Hugging Face Hub

```bash
# Push GGUF directly to Hub
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --push-to-hub username/crisis-agent-gguf \
  -q q4_k_m

# Or use Makefile
make upload-gguf CHECKPOINT=outputs/checkpoints/final REPO=username/crisis-agent-gguf
```

### Troubleshooting GGUF Export

**Model output is gibberish/repeating:**
- This is usually a chat template mismatch
- Ensure you use the same template in LM Studio/Ollama as during training
- The Modelfile we generate includes the correct template

**File too large:**
- Use a smaller quantization (q4_k_m or q3_k_m)
- Consider using `q4_k_m` which is ~4GB for a 7B model

**Ollama not found:**
- Install Ollama from: https://ollama.ai
- After installation, run: `make export-ollama`

---

## 📚 Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/docs/hub/models-uploading)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Unsloth GGUF Export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Unsloth Ollama Export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama)
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)

---

**Ready to deploy?** Follow the steps above, and your model will be available on Hugging Face, LM Studio, or Ollama! 🚀
