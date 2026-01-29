# Troubleshooting Guide

Solutions to common issues in the crisis-agent fine-tuning pipeline.

---

## Quick Diagnostics

Run these commands to diagnose issues:

```bash
# Check Python version
python --version

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi

# Verify setup
python scripts/verify_setup.py

# Check environment
conda info --envs
```

---

## Environment Issues

### Conda Environment Not Found

**Error:**
```
EnvironmentNameNotFound: Could not find conda environment: crisis_agent
```

**Solution:**
```bash
# Create the environment
conda create -n crisis_agent python=3.10 -y

# Or use automatic script
source activate_env.sh
```

---

### Conda Not Initialized

**Error:**
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
```

**Solution:**
```bash
# Initialize conda
eval "$(conda shell.bash hook)"

# Then activate
conda activate crisis_agent
```

---

### Package Version Conflicts

**Error:**
```
ERROR: Could not find a version that satisfies the requirement ...
```

**Solution:**
```bash
# Check requirements.txt for conflicting versions
# Some packages may need flexible versioning

# Update pip
pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

---

## CUDA/GPU Issues

### CUDA Not Available

**Error:**
```python
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Check PyTorch CUDA version:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

### CUDA Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   # training_config.yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8
   ```

2. **Reduce sequence length:**
   ```yaml
   # model_config.yaml
   model:
     max_seq_length: 1024
   ```

3. **Reduce LoRA rank:**
   ```yaml
   # model_config.yaml
   model:
     lora:
       r: 8
       lora_alpha: 16
   ```

4. **Clear GPU memory:**
   ```bash
   # Restart Python process
   # Or clear cache in code:
   python -c "import torch; torch.cuda.empty_cache()"
   ```

---

### GPU Not Detected

**Error:**
```
No GPU detected
```

**Solutions:**

1. **Check GPU visibility:**
   ```bash
   nvidia-smi
   echo $CUDA_VISIBLE_DEVICES
   ```

2. **Set CUDA device:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Check driver compatibility:**
   ```bash
   nvidia-smi
   # Check CUDA version matches PyTorch
   ```

---

## Dataset Issues

### Dataset Not Found

**Error:**
```
FileNotFoundError: Dataset 'xxx' not found
```

**Solutions:**

1. **Check dataset name:**
   ```yaml
   # dataset_config.yaml
   dataset:
     hf_dataset_name: "ianktoo/crisis-response-training-v2"
   ```

2. **Set HuggingFace token:**
   ```bash
   export HF_TOKEN="your_token"
   # Or add to .env file
   ```

3. **Accept dataset terms** on HuggingFace website

---

### Column Not Found

**Error:**
```
KeyError: 'instruction' not found in dataset
```

**Solution:**
```yaml
# Check column names in dataset_config.yaml
dataset:
  instruction_column: "Input"   # Match your dataset
  response_column: "Output"     # Match your dataset
```

---

### Dataset Access Denied

**Error:**
```
401 Client Error: Unauthorized
```

**Solutions:**

1. **Check HF_TOKEN:**
   ```bash
   echo $HF_TOKEN
   ```

2. **Verify token has read access:**
   - Go to https://huggingface.co/settings/tokens
   - Ensure token has appropriate permissions

3. **Login via CLI:**
   ```bash
   huggingface-cli login
   ```

---

## Training Issues

### Training Loss Not Decreasing

**Symptoms:**
- Loss stays flat or increases
- No improvement over epochs

**Solutions:**

1. **Check learning rate:**
   ```yaml
   training:
     learning_rate: 2.0e-4  # Try 1e-4 or 5e-5
   ```

2. **Increase warmup:**
   ```yaml
   training:
     warmup_steps: 200  # More warmup
   ```

3. **Check data quality:**
   ```bash
   # Inspect dataset
   python scripts/inspect_dataset.py
   ```

4. **Reduce batch size** (more frequent updates):
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 4
   ```

---

### Training Interrupted

**Error:**
```
KeyboardInterrupt
# or
Connection lost
```

**Solution:**
```bash
# Resume from last checkpoint
python scripts/train.py --resume-from outputs/checkpoints/checkpoint-500
```

---

### NaN Loss

**Error:**
```
Loss: nan
```

**Solutions:**

1. **Reduce learning rate:**
   ```yaml
   training:
     learning_rate: 1.0e-5  # Much lower
   ```

2. **Enable gradient clipping:**
   ```yaml
   training:
     max_grad_norm: 0.5  # Stricter clipping
   ```

3. **Check for data issues:**
   - Empty samples
   - Extremely long sequences
   - Invalid characters

---

## Evaluation Issues

### AI Evaluation API Key Error

**Error:**
```
AuthenticationError: Invalid API key
```

**Solution:**
```bash
# Set the appropriate key
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude
export OPENAI_API_KEY="sk-..."          # For OpenAI
export GEMINI_API_KEY="..."             # For Gemini

# Or add to .env file
```

---

### Rate Limit Exceeded

**Error:**
```
RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Reduce samples:**
   ```bash
   python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-max-samples 20
   ```

2. **Wait and retry**

3. **Use different provider:**
   ```bash
   python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
   ```

---

### Low Evaluation Scores

**Symptoms:**
- Valid JSON < 80%
- AI scores < 70

**Solutions:**

1. **Check training completed:**
   - Look for loss decrease in logs
   - Ensure all epochs ran

2. **Check data quality:**
   - Verify training data format
   - Check for corrupted samples

3. **Consider retraining:**
   - More epochs
   - Different hyperparameters
   - More/better data

---

## Deployment Issues

### GGUF export fails with “No working quantizer found” / “Permission denied”

**Symptoms:**
- `outputs/gguf/` stays empty after running `make export-gguf`
- Error mentions:
  - `Unsloth: No working quantizer found in llama.cpp`
  - or `Permission denied` when running `llama-quantize`

**Cause:**
Some environments mount the project/work drive with **`noexec`**, which prevents running compiled binaries from that filesystem. Unsloth’s GGUF export requires executing `llama.cpp/llama-quantize`.

**Fix (recommended):**
Use an executable directory for llama.cpp tooling by setting `CRISIS_GGUF_EXEC_DIR`:

```bash
CRISIS_GGUF_EXEC_DIR=/home/jovyan make export-gguf
```

Or run directly:

```bash
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/final_model \
  --output outputs/gguf \
  -q q4_k_m
```

**If you still get “no quantizer found”:**
Build llama.cpp once via CMake:

```bash
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j"$(nproc)"
```

Then retry the export.

### HuggingFace Upload Failed

**Error:**
```
HTTPError: 401 Unauthorized
```

**Solutions:**

1. **Check token permissions:**
   - Token needs **write** access
   - Go to https://huggingface.co/settings/tokens

2. **Verify token:**
   ```bash
   echo $HF_TOKEN
   huggingface-cli whoami
   ```

3. **Login via CLI:**
   ```bash
   huggingface-cli login
   ```

---

### GGUF Export Failed

**Error:**
```
Error during GGUF conversion
```

**Solutions:**

1. **Check llama.cpp installed:**
   ```bash
   # Unsloth handles this, but verify
   pip install unsloth --upgrade
   ```

2. **Try smaller quantization:**
   ```bash
   python scripts/export_gguf.py --checkpoint outputs/checkpoints/final -q q4_k_m
   ```

3. **Check disk space:**
   ```bash
   df -h
   ```

---

### Ollama Model Not Working

**Symptoms:**
- Gibberish output
- Repeated text
- Wrong responses

**Solutions:**

1. **Check Modelfile template:**
   - Ensure chat template matches training
   - Edit `outputs/gguf/Modelfile`

2. **Re-register model:**
   ```bash
   cd outputs/gguf
   ollama rm crisis-agent
   ollama create crisis-agent -f Modelfile
   ```

3. **Check context length:**
   ```
   # In Modelfile
   PARAMETER num_ctx 2048
   ```

---

### LM Studio Output Issues

**Symptoms:**
- Wrong format
- Missing responses

**Solutions:**

1. **Set correct prompt template:**
   - Go to model settings in LM Studio
   - Select "Mistral Instruct" or custom template

2. **Check stop tokens:**
   - Add `</s>` and `<|im_end|>` as stop tokens

3. **Reload model** in LM Studio

---

## Disk Space Issues

### Disk Full

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Check usage:**
   ```bash
   df -h
   du -sh outputs/*
   ```

2. **Clean old checkpoints:**
   ```bash
   rm -rf outputs/checkpoints/checkpoint-*
   # Keep only 'final'
   ```

3. **Clear dataset cache:**
   ```bash
   rm -rf ~/.cache/huggingface/datasets/*
   ```

4. **Compress logs:**
   ```bash
   gzip outputs/logs/*.log
   ```

---

## Getting Help

### Collect Diagnostic Info

```bash
# System info
python --version
nvidia-smi
conda info
pip list | grep -E "torch|transformers|unsloth"

# Error logs
tail -100 outputs/logs/*.log
```

### Resources

- **GitHub Issues**: Report bugs or feature requests
- **Documentation**: Check docs/ folder
- **Unsloth Docs**: https://github.com/unslothai/unsloth
- **HuggingFace Docs**: https://huggingface.co/docs

---

## Common Error Quick Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| `CUDA OOM` | GPU memory full | Reduce batch size |
| `Dataset not found` | Wrong name/no access | Check config/token |
| `API key invalid` | Missing/wrong key | Set in .env |
| `Module not found` | Missing dependency | `pip install -r requirements.txt` |
| `NaN loss` | Learning rate too high | Reduce LR |
| `Permission denied` | HF token wrong | Get write token |
| `Gibberish output` | Template mismatch | Fix chat template |

---

[← Scripts Reference](Scripts-Reference.md) | [Home](Home.md)
