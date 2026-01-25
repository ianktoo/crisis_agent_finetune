# Deployment Notes - AI Emergency Kit

## 🎯 Target Repository

**Hugging Face Model Repository:** [`ianktoo/crisis-agent-v1`](https://huggingface.co/ianktoo/crisis-agent-v1)

**Status:** Repository created, currently empty - ready for first model upload

---

## 📤 Upload Commands

### Option 1: Upload LoRA Checkpoint (Recommended - Smaller, Faster)

```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/checkpoints/final \
  --repo-name ianktoo/crisis-agent-v1
```

**Benefits:**
- Smaller file size (~100MB)
- Faster upload
- Users need base model + LoRA adapters

### Option 2: Upload Merged Model (Standalone)

```bash
# First merge LoRA weights
make merge

# Then upload merged model
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name ianktoo/crisis-agent-v1 \
  --merged
```

**Benefits:**
- Standalone model (~14GB)
- No base model needed
- Easier for end users

---

## 🏷️ Automatic Branding

When you upload, the model card will automatically include:

- **Title:** "AI Emergency Kit - crisis-agent-v1"
- **Description:** "AI Emergency Kit - Your intelligent crisis response assistant..."
- **Tags:** `ai-emergency-kit`, `emergency-assistant`, `crisis-response`, `emergency`, `fine-tuned`, `lora`

The branding is handled automatically by the upload script - no manual editing needed!

---

## ✅ Pre-Upload Checklist

Before uploading:

- [ ] Evaluation completed and results reviewed
- [ ] Model tested with inference script
- [ ] Checkpoint verified (`outputs/checkpoints/final` exists)
- [ ] HF_TOKEN set in `.env` file
- [ ] Repository `ianktoo/crisis-agent-v1` exists on Hugging Face
- [ ] You have write access to the repository

---

## 🚀 Upload Process

1. **Verify Setup:**
   ```bash
   # Check token is set
   grep HF_TOKEN .env
   
   # Verify checkpoint exists
   ls -la outputs/checkpoints/final/
   ```

2. **Run Upload:**
   ```bash
   python scripts/upload_to_hf.py \
     --checkpoint outputs/checkpoints/final \
     --repo-name ianktoo/crisis-agent-v1
   ```

3. **Monitor Progress:**
   - Upload will show progress
   - Model card will be auto-generated
   - Tags will be automatically added

4. **Verify Upload:**
   - Visit: https://huggingface.co/ianktoo/crisis-agent-v1
   - Check model card shows "AI Emergency Kit" branding
   - Verify files are present
   - Test loading the model

---

## 📝 Post-Upload Tasks

After successful upload:

- [ ] Verify model card displays correctly
- [ ] Check "AI Emergency Kit" branding is present
- [ ] Review auto-generated description
- [ ] Add any additional usage examples (optional)
- [ ] Test model loading:
  ```python
  from unsloth import FastLanguageModel
  
  model, tokenizer = FastLanguageModel.from_pretrained(
      "ianktoo/crisis-agent-v1",
      max_seq_length=2048,
      dtype=None,
      load_in_4bit=True,
  )
  ```
- [ ] Share repository link

---

## 🔗 Repository Links

- **Model Repository:** https://huggingface.co/ianktoo/crisis-agent-v1
- **Pipeline Repository:** https://github.com/ianktoo/crisis_agent_finetune
- **Dataset Repository:** https://huggingface.co/datasets/ianktoo/crisis-response-training

---

## 📊 Model Information

**Base Model:** unsloth/Mistral-7B-Instruct-v0.2  
**Fine-tuning Method:** LoRA (Low-Rank Adaptation)  
**Training Dataset:** ianktoo/crisis-response-training  
**Product Name:** AI Emergency Kit  
**Repository Name:** crisis-agent-v1 (for consistency with pipeline)

---

**Last Updated:** 2026-01-25  
**Repository Status:** Ready for upload
