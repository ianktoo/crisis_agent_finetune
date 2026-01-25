# Evaluation Timing Guide

## ⏱️ How Long Will Evaluation Take?

### Default Settings

- **Default samples:** 100 (configurable with `--max-samples`)
- **Max tokens per sample:** 512 (hardcoded in evaluation script)
- **Progress logging:** Every 10 samples

### Time Estimation

**Per Sample Time:**
- Model generation: ~2-5 seconds per sample (depends on GPU)
- Tokenization + validation: ~0.1-0.5 seconds per sample
- **Total per sample: ~2.5-5.5 seconds**

**Total Time Estimates:**

| Samples | Estimated Time | Notes |
|---------|---------------|-------|
| 10 | ~25-55 seconds | Quick test |
| 50 | ~2-5 minutes | Fast evaluation |
| 100 (default) | ~4-9 minutes | Standard evaluation |
| 200 | ~8-18 minutes | Extended evaluation |
| 500 | ~20-45 minutes | Comprehensive evaluation |

**Note:** Times vary based on:
- GPU performance (faster GPU = faster generation)
- Model size and quantization
- Sequence length of prompts
- Actual tokens generated (may be less than 512)

---

## 📊 Monitoring Progress

### During Evaluation

The evaluation script logs progress every 10 samples:

```
Evaluated 10/100 samples...
Evaluated 20/100 samples...
Evaluated 30/100 samples...
...
```

### Check Current Progress

If evaluation is running, you can:

1. **Check the log file:**
   ```bash
   tail -f outputs/logs/crisis_agent_*.log | grep "Evaluated"
   ```

2. **Check process status:**
   ```bash
   ps aux | grep evaluate.py
   ```

3. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Estimate Remaining Time

Based on progress logs:

1. **Note the time when evaluation started**
2. **Check current sample number** (from logs)
3. **Calculate:**
   ```
   Time per sample = Elapsed time / Samples completed
   Remaining samples = Total samples - Samples completed
   Estimated remaining = Time per sample × Remaining samples
   ```

**Example:**
- Started: 12:00:00
- Current time: 12:02:30 (2.5 minutes = 150 seconds)
- Samples completed: 30
- Time per sample: 150 / 30 = 5 seconds
- Remaining: 100 - 30 = 70 samples
- Estimated remaining: 70 × 5 = 350 seconds = ~6 minutes

---

## 🎛️ Adjusting Evaluation Time

### Reduce Samples (Faster)

```bash
# Quick evaluation (10 samples, ~30 seconds)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 10

# Fast evaluation (50 samples, ~2-5 minutes)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 50
```

### Increase Samples (More Comprehensive)

```bash
# Extended evaluation (200 samples, ~8-18 minutes)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 200

# Comprehensive evaluation (500 samples, ~20-45 minutes)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 500
```

---

## 🔍 Factors Affecting Evaluation Time

### 1. Number of Samples (`--max-samples`)
- **Most important factor**
- Linear relationship: 2x samples = 2x time

### 2. GPU Performance
- Faster GPU = faster generation
- 16GB VRAM GPU: ~2-5 seconds per sample
- More powerful GPU: ~1-3 seconds per sample

### 3. Model Generation Settings
- `max_new_tokens`: 512 (default, hardcoded)
- `temperature`: 0.7 (default)
- `do_sample`: True (default)

### 4. Prompt Length
- Longer prompts = slightly longer generation time
- Usually minimal impact

### 5. Actual Tokens Generated
- Model may generate fewer than 512 tokens
- Early stopping can reduce time

---

## 📈 Real-World Examples

### Example 1: Quick Test (10 samples)
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 10
```
**Expected time:** ~30 seconds to 1 minute

### Example 2: Standard Evaluation (100 samples - default)
```bash
make evaluate
# or
python scripts/evaluate.py --checkpoint outputs/checkpoints/final
```
**Expected time:** ~4-9 minutes

### Example 3: Comprehensive Evaluation (200 samples)
```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --max-samples 200
```
**Expected time:** ~8-18 minutes

---

## ⚡ Tips for Faster Evaluation

1. **Start with fewer samples** to test:
   ```bash
   python scripts/evaluate.py --max-samples 10
   ```

2. **Monitor first few samples** to estimate time:
   - Check logs after 10 samples
   - Calculate time per sample
   - Estimate total time

3. **Run in background** if needed:
   ```bash
   nohup make evaluate > eval_output.log 2>&1 &
   ```

4. **Check progress periodically:**
   ```bash
   tail -f eval_output.log
   ```

---

## 🛑 Stopping Evaluation

If you need to stop evaluation:

1. **Press Ctrl+C** in the terminal running evaluation
2. **Or kill the process:**
   ```bash
   pkill -f evaluate.py
   ```

**Note:** Partial results are not saved - you'll need to restart evaluation.

---

## 📝 Current Evaluation Status

Based on your current run:

- **Command:** `make evaluate` (default: 100 samples)
- **Expected duration:** ~4-9 minutes
- **Progress updates:** Every 10 samples
- **Output:** `outputs/evaluation_report.json`

**To check current progress:**
```bash
# Check logs for progress
tail -20 outputs/logs/crisis_agent_*.log | grep -E "Evaluated|Error evaluating"

# Or check if process is running
ps aux | grep evaluate.py
```

---

## 🎯 Recommended Approach

1. **First run:** Use default (100 samples) to get baseline metrics
2. **If time allows:** Run extended evaluation (200 samples) for more confidence
3. **For quick tests:** Use 10-50 samples during development

---

**Last Updated:** 2026-01-25
