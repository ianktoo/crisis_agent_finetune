# Model Optimization Guide for Ollama & LM Studio

This guide explains how to optimize your fine-tuned model to make it small enough to run efficiently on Ollama or LM Studio.

## Quick Start

**For smallest size (recommended for most users):**
```bash
# Export with 4-bit quantization (~4GB for 7B model)
make export-gguf

# Or with 3-bit for even smaller (~3GB)
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q3_k_m
```

**For best quality/size balance:**
```bash
# Export with 4-bit quantization (default)
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m
```

## Understanding Model Size

### Your Current Setup

- **Base Model**: Mistral-7B-Instruct (~13-14GB in FP16)
- **Training**: Using LoRA adapters (~100-300MB)
- **Current Export**: Depends on quantization method

### Size Comparison (Mistral-7B)

| Quantization | Size | Quality | Use Case |
|--------------|------|---------|----------|
| `f32` | ~28GB | Full precision | Not recommended |
| `f16` | ~14GB | Full precision | Maximum quality, large |
| `q8_0` | ~8GB | Very high | High quality priority |
| `q5_k_m` | ~5GB | High | Balanced quality/size |
| **`q4_k_m`** | **~4GB** | **Good** | **Recommended (best balance)** |
| `q3_k_m` | ~3GB | Lower | Size priority |
| `q2_k` | ~2GB | Lowest | Experimental, minimal quality |

## Optimization Strategies

### 1. Use Aggressive Quantization

**Recommended: `q4_k_m` (4-bit)**
- Best balance of quality and size
- ~4GB for a 7B model
- Quality loss is minimal for most use cases

**For smallest size: `q3_k_m` (3-bit)**
- ~3GB for a 7B model
- Some quality loss, but still usable
- Good for resource-constrained systems

**Avoid: `f16` or `q8_0`**
- These are too large for most local deployments
- Only use if you have plenty of RAM/VRAM

### 2. Export Directly from LoRA (Don't Merge)

**✅ DO THIS:**
```bash
# Export directly from LoRA checkpoint (smaller, faster)
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m
```

**❌ DON'T DO THIS:**
```bash
# Don't merge first - it creates a larger intermediate model
make merge  # Creates ~14GB merged model
python scripts/export_gguf.py --checkpoint outputs/final_model ...
```

**Why?** Unsloth's GGUF export automatically merges LoRA during export, so you don't need to merge separately. Merging first wastes disk space and time.

### 3. Reduce Sequence Length (Optional)

If you need even smaller models, you can reduce the maximum sequence length:

```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --max-seq-length 1024  # Instead of default 2048
```

**Trade-offs:**
- Smaller model size (slightly)
- Faster inference
- Can't handle as long contexts

### 4. Use Multiple Quantization Levels

Export multiple versions and test which works best for your hardware:

```bash
# Small version
python scripts/export_gguf.py -q q3_k_m --output outputs/gguf-small

# Balanced version (recommended)
python scripts/export_gguf.py -q q4_k_m --output outputs/gguf-balanced

# Quality version (if you have RAM)
python scripts/export_gguf.py -q q5_k_m --output outputs/gguf-quality
```

## Step-by-Step Optimization

### Step 1: Check Your Current Model Size

```bash
# Check checkpoint size
du -sh outputs/checkpoints/final

# Check if you have any existing GGUF exports
ls -lh outputs/gguf/*.gguf 2>/dev/null || echo "No GGUF files found"
```

### Step 2: Export with Optimal Quantization

```bash
# Set executable directory (important for some environments)
export CRISIS_GGUF_EXEC_DIR=/home/jovyan

# Export with 4-bit quantization (recommended)
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m
```

### Step 3: Verify Export Size

```bash
# Check the exported GGUF file size
ls -lh outputs/gguf/*.gguf

# Should show ~4GB for q4_k_m quantization
```

### Step 4: Test with Ollama or LM Studio

**For Ollama:**
```bash
# Export with Ollama setup
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --ollama

# Test
ollama run crisis-agent
```

**For LM Studio:**
```bash
# Export
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m

# Import into LM Studio
lms import outputs/gguf/*.gguf
# Or manually copy to ~/.lmstudio/models/
```

## Troubleshooting

### Model Still Too Large

1. **Use more aggressive quantization:**
   ```bash
   # Try 3-bit instead of 4-bit
   python scripts/export_gguf.py -q q3_k_m ...
   ```

2. **Check if you're exporting from merged model:**
   - Use `outputs/checkpoints/final` (LoRA checkpoint)
   - Don't use `outputs/final_model` (merged model)

3. **Reduce sequence length:**
   ```bash
   python scripts/export_gguf.py --max-seq-length 1024 ...
   ```

### Export Fails with "Permission denied" or "noexec"

Set the executable directory:
```bash
export CRISIS_GGUF_EXEC_DIR=/home/jovyan
python scripts/export_gguf.py ...
```

Or use the Makefile (which sets this automatically):
```bash
make export-gguf
```

### Quality Too Low After Quantization

1. **Try higher quantization:**
   ```bash
   # Upgrade from q3_k_m to q4_k_m
   python scripts/export_gguf.py -q q4_k_m ...
   
   # Or try q5_k_m for better quality
   python scripts/export_gguf.py -q q5_k_m ...
   ```

2. **Check your base model:**
   - Ensure you're using a good base model
   - Consider retraining with better data

## Advanced: Custom Optimization Script

Create a script to export multiple quantization levels at once:

```bash
#!/bin/bash
# optimize_model.sh - Export multiple quantization levels

CHECKPOINT=${1:-outputs/checkpoints/final}
OUTPUT_DIR=${2:-outputs/gguf}

export CRISIS_GGUF_EXEC_DIR=/home/jovyan

echo "Exporting with different quantization levels..."

for quant in q3_k_m q4_k_m q5_k_m; do
  echo "Exporting with $quant..."
  python scripts/export_gguf.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR-$quant" \
    -q "$quant"
done

echo "Done! Check sizes:"
ls -lh "$OUTPUT_DIR"*/*.gguf
```

## Recommended Settings by Use Case

### Desktop/Laptop (8-16GB RAM)
```bash
python scripts/export_gguf.py -q q4_k_m
```
**Result:** ~4GB model, good quality

### Resource-Constrained (4-8GB RAM)
```bash
python scripts/export_gguf.py -q q3_k_m
```
**Result:** ~3GB model, acceptable quality

### High-End System (16GB+ RAM)
```bash
python scripts/export_gguf.py -q q5_k_m
```
**Result:** ~5GB model, high quality

### Maximum Quality (if size doesn't matter)
```bash
python scripts/export_gguf.py -q q8_0
```
**Result:** ~8GB model, very high quality

## Summary

**For most users, the optimal approach is:**

1. ✅ Export directly from LoRA checkpoint (don't merge first)
2. ✅ Use `q4_k_m` quantization (~4GB)
3. ✅ Export with `--ollama` flag if using Ollama
4. ✅ Test the exported model before deploying

**Quick command:**
```bash
export CRISIS_GGUF_EXEC_DIR=/home/jovyan
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --ollama
```

This will create a ~4GB model optimized for Ollama and LM Studio!
