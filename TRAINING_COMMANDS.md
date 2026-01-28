# Training Commands - Quick Reference

## Start Training

```bash
# Activate the conda environment
conda activate crisis_agent

# Navigate to project directory
cd /home/jovyan/work/projects/crisis_agent_finetune

# Start training
python scripts/train.py
```

### With Custom Model Name

```bash
python scripts/train.py --model-name "crisis_agent_v1.0"
```

---

## Monitor Training Progress

### 1. Watch Logs in Real-Time

```bash
# Follow the latest log file
tail -f outputs/logs/crisis_agent_*.log

# Or with line numbers
tail -f -n 50 outputs/logs/crisis_agent_*.log
```

### 2. Check Latest Log Entries

```bash
# Last 50 lines
tail -50 outputs/logs/crisis_agent_*.log

# Last 100 lines with timestamps
tail -100 outputs/logs/crisis_agent_*.log | grep -E "INFO|ERROR|WARNING"
```

### 3. Monitor GPU Usage

```bash
# Watch GPU every 1 second
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

### 4. Check Training Progress (Steps/Loss)

```bash
# Search for training steps and loss
grep -E "step|loss|epoch" outputs/logs/crisis_agent_*.log | tail -20
```

### 5. Check if Training is Running

```bash
# Check for Python training process
ps aux | grep "train.py"

# Or check GPU processes
nvidia-smi | grep python
```

---

## Check Checkpoints

```bash
# List all checkpoints
ls -lh outputs/checkpoints/

# Check latest checkpoint
ls -lht outputs/checkpoints/ | head -5

# Check checkpoint size
du -sh outputs/checkpoints/*
```

---

## Stop Training

```bash
# Find the process ID
ps aux | grep "train.py"

# Kill the process (replace PID with actual process ID)
kill <PID>

# Or use Ctrl+C in the terminal where training is running
```

---

## Resume Training

Training will automatically resume from the last checkpoint if interrupted. Just run:

```bash
python scripts/train.py
```

The trainer will detect existing checkpoints and resume from there.

---

## Quick Status Check

```bash
# One-liner to check everything
echo "=== Training Status ===" && \
ps aux | grep -E "train.py|python.*train" | grep -v grep && \
echo -e "\n=== Latest Log ===" && \
tail -5 outputs/logs/crisis_agent_*.log 2>/dev/null | tail -5 && \
echo -e "\n=== GPU Usage ===" && \
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

---

## Background Training (Optional)

If you want to run training in the background:

```bash
# Run in background and save output
nohup python scripts/train.py > training_output.log 2>&1 &

# Check background job
jobs

# Bring to foreground
fg

# Or detach and continue in background
# (Use screen or tmux for better session management)
```

---

## Expected Output

When training is running, you should see:

```
[PHASE 7] Starting training...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 2,000 | Num Epochs = 3 | Total steps = 750
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 41,943,040 of 7,283,675,136 (0.58% trained)

  5%|▌         | 40/750 [02:17<38:39,  3.27s/it]
```

---

## Evaluation

After training, evaluate the model:

```bash
# Standard evaluation
make evaluate
# or
python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# AI-based evaluation (Claude, OpenAI, or Gemini)
make evaluate-ai           # Claude
make evaluate-ai-openai    # OpenAI
make evaluate-ai-gemini    # Gemini
# or
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
```

Requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` for AI evaluation. See [docs/ai-evaluation.md](docs/ai-evaluation.md).

---

## Troubleshooting Commands

```bash
# Check if environment is activated
conda info --envs
echo $CONDA_DEFAULT_ENV

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check dataset access
python -c "from datasets import load_dataset; ds = load_dataset('ianktoo/crisis-response-training-v2'); print(f'Samples: {len(ds[\"train\"])}')"

# Verify configuration
python scripts/verify_setup.py
```
