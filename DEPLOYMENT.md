# Deployment Checklist

Quick checklist to deploy and start training on your server.

## ✅ Pre-Deployment Checklist

### 1. Local Verification (Optional but Recommended)

```bash
# Verify project structure
ls -la

# Check configs exist
ls configs/

# Verify dataset config
cat configs/dataset_config.yaml | grep hf_dataset_name
# Should show: hf_dataset_name: "ianktoo/crisis-response-training"
```

### 2. Verify Dataset Structure

Before deploying, check your dataset structure:

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("ianktoo/crisis-response-training")

# Check structure
print("Splits:", dataset.keys())
print("\nFirst sample:")
print(dataset["train"][0])
print("\nColumn names:")
print(dataset["train"].features)
```

**Important**: Verify the column names match your config:
- Default: `instruction` and `response`
- Update `configs/dataset_config.yaml` if your dataset uses different column names

## 🚀 Server Deployment Steps

### Step 1: Copy Repository to Server

```bash
# On your local machine (if using git)
git clone https://github.com/ianktoo/crisis_agent_finetune.git
cd crisis_agent_finetune

# Or copy files directly to server
# scp -r crisis_agent_finetune user@server:/path/to/destination/
```

### Step 2: Set Up Environment

```bash
# Navigate to project directory
cd crisis_agent_finetune

# Create .env file from template
cp .env.example .env

# Edit .env file (if dataset is private)
nano .env
# Add: HF_TOKEN=your_huggingface_token_here
```

**Note**: If your dataset `ianktoo/crisis-response-training` is public, you may not need the token.

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or use Makefile
make setup
```

**Expected time**: 5-15 minutes depending on connection speed.

### Step 4: Verify Dataset Access

```bash
# Quick test to verify dataset can be loaded
python -c "from datasets import load_dataset; ds = load_dataset('ianktoo/crisis-response-training'); print(f'Dataset loaded: {len(ds[\"train\"])} training samples')"
```

### Step 5: Verify Column Names

```bash
# Check dataset structure
python -c "
from datasets import load_dataset
ds = load_dataset('ianktoo/crisis-response-training')
print('Columns:', list(ds['train'][0].keys()))
print('Sample:', ds['train'][0])
"
```

**Action Required**: If columns are different from `instruction` and `response`, update `configs/dataset_config.yaml`:

```yaml
dataset:
  instruction_column: "your_actual_column_name"
  response_column: "your_actual_column_name"
```

### Step 6: Quick Configuration Check

```bash
# Verify all configs are valid YAML
python -c "
import yaml
for config in ['configs/dataset_config.yaml', 'configs/model_config.yaml', 'configs/training_config.yaml']:
    with open(config) as f:
        yaml.safe_load(f)
    print(f'✓ {config} is valid')
"
```

### Step 7: Test Dataset Loading

```bash
# Test the full dataset loading pipeline
python -c "
import sys
sys.path.insert(0, '.')
from src.data.load_dataset import load_dataset_from_config
dataset = load_dataset_from_config()
print(f'✓ Dataset loaded successfully')
print(f'  Train samples: {len(dataset[\"train\"])}')
print(f'  Validation samples: {len(dataset.get(\"validation\", []))}')
"
```

### Step 8: Verify GPU Access

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU memory
nvidia-smi
```

## 🎯 Ready to Train?

Once all checks pass, you're ready to start training:

```bash
# Start training
python scripts/train.py

# Or with custom model name
python scripts/train.py --model-name "crisis_agent_v1.0"
```

## ⚠️ Common Issues Before Training

### Issue 1: Dataset Column Mismatch

**Symptom**: Error about missing columns (`KeyError: 'instruction'`)

**Solution**: 
1. Check your dataset columns: `python -c "from datasets import load_dataset; print(load_dataset('ianktoo/crisis-response-training')['train'][0].keys())"`
2. Update `configs/dataset_config.yaml` with correct column names

### Issue 2: Dataset Not Found

**Symptom**: `Dataset not found: ianktoo/crisis-response-training`

**Solution**:
1. Verify dataset name is correct on Hugging Face
2. If private, ensure `HF_TOKEN` is set in `.env`
3. Test access: `python -c "from datasets import load_dataset; load_dataset('ianktoo/crisis-response-training')"`

### Issue 3: CUDA Not Available

**Symptom**: Warning about CUDA not available

**Solution**:
1. Verify GPU: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. May need to reinstall PyTorch with CUDA support

### Issue 4: Out of Memory During Dataset Loading

**Symptom**: Memory error when loading dataset

**Solution**:
1. Limit samples in config: `max_samples: 1000`
2. Use smaller subset for testing first

## 📋 Final Pre-Training Checklist

- [ ] Repository copied to server
- [ ] `.env` file created (if needed for private dataset)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset accessible (`python -c "from datasets import load_dataset; load_dataset('ianktoo/crisis-response-training')"`)
- [ ] Column names verified and config updated if needed
- [ ] GPU accessible (`nvidia-smi` shows GPU)
- [ ] CUDA working (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Config files valid (no YAML errors)
- [ ] Sufficient disk space (check with `df -h`)

## 🚀 Start Training

Once all checks pass:

```bash
# Basic training
python scripts/train.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi

# Monitor logs
tail -f outputs/logs/crisis_agent_*.log
```

## 📊 Expected Output

You should see:

```
================================================================================
Starting crisis-agent fine-tuning pipeline
================================================================================

[PHASE 1] Loading dataset...
Loading dataset: ianktoo/crisis-response-training
Successfully loaded dataset: ianktoo/crisis-response-training
train split: X samples
validation split: Y samples

[PHASE 2] Formatting dataset...
...

[PHASE 3] Loading model...
Loading model: unsloth/Mistral-7B-Instruct-v0.2
Model loaded successfully

[PHASE 4] Applying LoRA adapters...
...

[PHASE 7] Starting training...
Training will save checkpoints to: outputs/checkpoints
...
```

## 🆘 If Something Goes Wrong

1. **Check logs**: `outputs/logs/crisis_agent_*.log`
2. **Verify dataset**: Run the dataset loading test above
3. **Check GPU**: `nvidia-smi`
4. **Review configs**: Ensure all YAML files are valid
5. **Test imports**: `python -c "import sys; sys.path.insert(0, '.'); from src.data.load_dataset import load_dataset_from_config"`

## ✅ Success Indicators

- Dataset loads without errors
- Model loads successfully
- Training starts and shows progress
- Checkpoints are being saved
- GPU utilization is high (85-95%)

---

**You're ready to train!** 🚀

If all checks pass, run `python scripts/train.py` and monitor the logs.
