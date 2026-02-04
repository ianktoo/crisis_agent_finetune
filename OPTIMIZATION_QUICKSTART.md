# Model Optimization Quick Start

**Problem:** Your model is too large for Ollama/LM Studio.

**Solution:** Use quantization to reduce model size.

## 🚀 Quick Commands

### Recommended (Best Balance)
```bash
# ~4GB model, good quality
make optimize-balanced
```

### Smallest Size
```bash
# ~3GB model, acceptable quality
make optimize-small
```

### Higher Quality
```bash
# ~5GB model, better quality
make optimize-quality
```

## 📊 Size Comparison

| Command | Quantization | Size (7B) | Quality |
|---------|--------------|-----------|---------|
| `make optimize-small` | q3_k_m | ~3GB | Lower |
| `make optimize-balanced` | q4_k_m | ~4GB | **Good** ⭐ |
| `make optimize-quality` | q5_k_m | ~5GB | Better |
| `make export-lmstudio` | q8_0 | ~8GB | High |

## ✅ Best Practices

1. **Export directly from LoRA checkpoint** (don't merge first)
   - ✅ Use: `outputs/checkpoints/final`
   - ❌ Don't use: `outputs/final_model` (merged model)

2. **Use q4_k_m quantization** for best balance
   - Good quality, reasonable size

3. **Test multiple quantization levels** if unsure
   - Start with q4_k_m, try q3_k_m if too large, q5_k_m if quality is low

## 🔧 Manual Export

```bash
# Basic export (creates informative filename automatically)
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m

# With custom model name
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --model-name crisis-agent-v1

# With Ollama setup
CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --ollama

# List all exports with sizes
python scripts/export_gguf.py --list-exports --output outputs/gguf
# Or use Makefile
make list-exports
```

## 📋 File Naming

Exports are automatically named with versionable, informative names:
- Format: `{model-name}-{quantization}-{date}.gguf`
- Example: `crisis-agent-v1-q4_k_m-20260204.gguf`

This makes it easy to:
- Track different versions
- Identify quantization level
- See when exports were created
- Manage multiple exports

## 📚 More Information

- **Full guide:** `docs/model-optimization.md`
- **Helper script:** `scripts/optimize_model.sh --help`
- **List exports:** `make list-exports` or `python scripts/export_gguf.py --list-exports`

## 🎯 Recommended Workflow

1. **Export optimized model:**
   ```bash
   make optimize-balanced
   ```

2. **Check file size:**
   ```bash
   # View all exports with detailed info
   make list-exports
   
   # Or manually
   ls -lh outputs/gguf/*.gguf
   ```

3. **Test with Ollama:**
   ```bash
   ollama run crisis-agent
   ```

4. **Or import to LM Studio:**
   ```bash
   lms import outputs/gguf/*.gguf
   ```

## ⚠️ Common Issues

**Model still too large?**
- Use `make optimize-small` (q3_k_m)
- Check you're exporting from LoRA checkpoint, not merged model

**Quality too low?**
- Use `make optimize-quality` (q5_k_m)
- Or try `make export-lmstudio` (q8_0) if you have RAM

**Export fails?**
- Set `CRISIS_GGUF_EXEC_DIR=/home/jovyan`
- Or use Makefile targets (they set this automatically)
