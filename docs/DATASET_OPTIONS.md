# Dataset Options for JSON Training

## Current Situation

Your current dataset (`ianktoo/crisis-response-training`) contains **structured text responses**, not JSON:
- Format: "FACTS:", "UNCERTAINTIES:", "ANALYSIS:", "GUIDANCE:"
- Size: 2000 samples
- Model was trained correctly on this format, so it generates text (not JSON)

## Options to Get JSON Output

### Option 1: Use Converted Dataset (Recommended - Quick Start)

✅ **Already Done!** I've converted your dataset to JSON format.

**Location:** `data/crisis_response_json.jsonl`

**To use it:**

1. **Option A: Load from local JSONL file** (requires code update)
   - Update `src/data/load_dataset.py` to support loading from JSONL files
   - Or use the script below

2. **Option B: Upload to Hugging Face** (easiest)
   ```bash
   # Install huggingface_hub if needed
   pip install huggingface_hub
   
   # Upload (you'll need HF token)
   python -c "
   from datasets import load_dataset
   from huggingface_hub import login
   login()  # Enter your token
   
   ds = load_dataset('json', data_files='data/crisis_response_json.jsonl')
   ds.push_to_hub('your-username/crisis-response-json')
   ```
   
   Then update `configs/dataset_config.yaml`:
   ```yaml
   hf_dataset_name: "your-username/crisis-response-json"
   instruction_column: "instruction"
   response_column: "response"
   validate_json: true  # Enable JSON validation
   ```

### Option 2: Find/Create a Better Dataset

Look for datasets with JSON responses:
- Search Hugging Face: https://huggingface.co/datasets?search=crisis+json
- Create your own with JSON examples
- Use a dataset that already has structured JSON outputs

### Option 3: Fine-tune with JSON Examples

Add JSON examples to your training:
1. Keep existing text examples (2000 samples)
2. Add new JSON examples (100-500 samples)
3. Model will learn both formats, but JSON examples will guide it

## Quick Fix: Update Dataset Loader

To use the converted JSONL file immediately, you can update the dataset loader:

```python
# In src/data/load_dataset.py, add support for JSONL:
if hf_dataset_name.endswith('.jsonl'):
    dataset = load_dataset('json', data_files=str(Path(hf_dataset_name)))
```

## Next Steps

1. **Re-run conversion** (to fix the confidence parsing issue):
   ```bash
   python scripts/convert_dataset_to_json.py
   ```

2. **Choose your approach:**
   - Upload to HF (easiest)
   - Update loader for local files
   - Find a better dataset

3. **Re-train the model** with JSON dataset:
   ```bash
   python scripts/train.py
   ```

4. **Re-evaluate** to see JSON output:
   ```bash
   python scripts/evaluate.py --checkpoint outputs/checkpoints/final
   ```

## Current Dataset Stats

- **Original**: 2000 text samples
- **Converted**: 2000 JSON samples (saved to `data/crisis_response_json.jsonl`)
- **Format**: JSON with `facts`, `uncertainties`, `analysis`, `guidance`, `confidence`

## Recommendation

**Use Option 1 (Converted Dataset)** - it's ready to go and maintains all your original data, just in JSON format. Upload it to Hugging Face for easiest integration.
