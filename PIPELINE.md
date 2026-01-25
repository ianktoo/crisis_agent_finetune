# AI Emergency Kit - Fine-Tuning Pipeline

Complete pipeline checklist for training, evaluation, and deployment of the **AI Emergency Kit** model.

---

## 📊 Training Phase

### Pre-Training Setup
- [ ] Environment variables configured (`.env` file)
- [ ] Dataset access configured (HF_TOKEN set)
- [ ] Dependencies installed (`make setup` or `pip install -r requirements.txt`)
- [ ] Configuration files reviewed (`configs/` directory)
- [ ] GPU/CUDA verified (`python scripts/verify_setup.py`)

### Training Execution
- [ ] Training started (`make train` or `python scripts/train.py`)
- [ ] Training completed successfully (3 epochs)
- [ ] Checkpoints saved in `outputs/checkpoints/`
  - [ ] `checkpoint-500/` (mid-training checkpoint)
  - [ ] `checkpoint-750/` (final step checkpoint)
  - [ ] `final/` (final checkpoint)
- [ ] Training logs reviewed (`outputs/logs/`)
- [ ] Loss metrics verified (decreasing trend observed)

**Training Summary:**
- Total Steps: 750
- Epochs: 3
- Final Loss: ~0.16
- Training Time: ~41 minutes
- Average Loss: 0.3755

---

## 🔍 Evaluation Phase

### Model Evaluation
- [x] Evaluation script fixed (no_speak() compatibility)
- [x] Evaluation run (`make evaluate`) - **IN PROGRESS**
  - **Default:** 100 samples
  - **Estimated time:** ~4-9 minutes
  - **Progress:** Logged every 10 samples
  - **See:** [docs/evaluation-timing.md](docs/evaluation-timing.md) for timing details
- [ ] Evaluation report generated (`outputs/evaluation_report.json`)
- [ ] Metrics reviewed:
  - [ ] Valid JSON percentage
  - [ ] Valid structure percentage
  - [ ] Error analysis completed
- [ ] Evaluation results documented

**Expected Commands:**
```bash
# Basic evaluation
make evaluate

# Extended evaluation (more samples)
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 200
```

### Test Suite
- [ ] Unit tests run (`make test-unit`)
- [ ] Integration tests run (`make test-integration`)
- [ ] Full test suite run (`make test`)
- [ ] Test coverage checked (`make test-cov`)
- [ ] All tests passing

---

## 🧪 Testing Phase

### Manual Inference Testing
- [ ] Inference script fixed (no_speak() compatibility)
- [ ] Interactive mode tested (`make infer`)
- [ ] Single prompt tested:
  ```bash
  python scripts/infer.py \
    --checkpoint outputs/checkpoints/final \
    --prompt "A building is on fire with people trapped inside" \
    --validate-json
  ```
- [ ] Multiple crisis scenarios tested
- [ ] JSON output format validated
- [ ] Response quality assessed
- [ ] Edge cases tested

### Baseline Comparison (Optional)
- [ ] Base model tested (unsloth/Mistral-7B-Instruct-v0.2)
- [ ] Fine-tuned model tested
- [ ] Responses compared
- [ ] Improvement verified

---

## 🔧 Preparation Phase

### LoRA Weight Merging
- [ ] Backup checkpoints created (optional but recommended)
- [ ] LoRA weights merged (`make merge`)
- [ ] Merged model saved to `outputs/final_model/`
- [ ] Merged model size verified (~14GB)
- [ ] Merged model tested for inference

**Command:**
```bash
make merge
# or
python scripts/merge_lora.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/final_model
```

**Note:** Merged models are larger (~14GB) but standalone (no LoRA files needed).

---

## 🚀 Deployment Phase

### Hugging Face Preparation
- [ ] Hugging Face account created
- [ ] Access token generated (https://huggingface.co/settings/tokens)
- [ ] Token added to `.env` file: `HF_TOKEN=your_token_here`
- [ ] Token permissions verified (write access)
- [ ] Repository name chosen (format: `username/repo-name`)

### Model Upload

#### Option A: Upload LoRA Checkpoint (Recommended)
- [ ] LoRA checkpoint selected (`outputs/checkpoints/final`)
- [ ] Repository name confirmed: `ianktoo/crisis-agent-v1` ✅
- [ ] Upload command prepared:
  ```bash
  python scripts/upload_to_hf.py \
    --checkpoint outputs/checkpoints/final \
    --repo-name ianktoo/crisis-agent-v1
  ```
  **Note:** Model will be automatically branded as "AI Emergency Kit" in the model card.
- [ ] Upload executed successfully
- [ ] Model verified on Hugging Face Hub: https://huggingface.co/ianktoo/crisis-agent-v1

#### Option B: Upload Merged Model (Standalone)
- [ ] Merged model selected (`outputs/final_model`)
- [ ] Repository name confirmed: `ianktoo/crisis-agent-v1` ✅
- [ ] Upload command prepared:
  ```bash
  python scripts/upload_to_hf.py \
    --checkpoint outputs/final_model \
    --repo-name ianktoo/crisis-agent-v1 \
    --merged
  ```
  **Note:** Model will be automatically branded as "AI Emergency Kit" in the model card.
- [ ] Upload executed successfully
- [ ] Model verified on Hugging Face Hub: https://huggingface.co/ianktoo/crisis-agent-v1

### Post-Upload Tasks
- [ ] Model accessible on Hugging Face Hub: https://huggingface.co/ianktoo/crisis-agent-v1
- [ ] Model card created/updated (README.md on HF) - **Will automatically use "AI Emergency Kit" branding**
- [ ] Model description added - **Auto-generated with AI Emergency Kit branding**
- [ ] Training details documented
- [ ] Usage examples provided
- [ ] Evaluation metrics included
- [ ] Model loading tested from HF:
  ```python
  from unsloth import FastLanguageModel
  
  model, tokenizer = FastLanguageModel.from_pretrained(
      "ianktoo/crisis-agent-v1",
      max_seq_length=2048,
      dtype=None,
      load_in_4bit=True,
  )
  ```

---

## 📝 Documentation Phase

### Code Documentation
- [ ] Code comments reviewed
- [ ] Function docstrings complete
- [ ] README.md updated
- [ ] Configuration files documented

### Project Documentation
- [ ] CHANGELOG.md updated
- [ ] POST_TRAINING.md reviewed
- [ ] QUICK_GUIDE.md reviewed
- [ ] DEPLOYMENT.md reviewed
- [ ] Pipeline document (this file) completed

---

## ✅ Final Checklist

### Quality Assurance
- [ ] All tests passing
- [ ] Evaluation metrics acceptable
- [ ] Manual testing completed
- [ ] No critical errors in logs
- [ ] Model performs as expected

### Deployment Verification
- [ ] Model uploaded to Hugging Face
- [ ] Model accessible and loadable
- [ ] Model card complete
- [ ] Usage examples work
- [ ] Repository properly configured

### Project Completion
- [ ] All checkpoints backed up (optional)
- [ ] Training logs archived
- [ ] Evaluation reports saved
- [ ] Documentation complete
- [ ] Project ready for use/sharing

---

## 🎯 Quick Command Reference

| Phase | Command | Output |
|-------|---------|--------|
| **Setup** | `make setup` | Dependencies installed |
| **Verify** | `python scripts/verify_setup.py` | Setup verification |
| **Train** | `make train` | `outputs/checkpoints/` |
| **Evaluate** | `make evaluate` | `outputs/evaluation_report.json` |
| **Test** | `make test` | Test results |
| **Infer** | `make infer` | Interactive inference |
| **Merge** | `make merge` | `outputs/final_model/` |
| **Upload** | `python scripts/upload_to_hf.py ...` | Hugging Face repo |

---

## 📊 Progress Tracking

**Current Status:** Evaluation In Progress ⏳

**Completed Steps:**
- ✅ Training Phase (3 epochs, 750 steps)
- ✅ Checkpoints saved (`outputs/checkpoints/final`)
- ✅ Evaluation script fixed (no_speak() compatibility)
- ⏳ **Currently: Running model evaluation**

**Next Steps:**
- [ ] Complete evaluation and review results
- [ ] Run test suite
- [ ] Test inference manually
- [ ] Merge LoRA weights
- [ ] Upload to Hugging Face: `ianktoo/crisis-agent-v1` (ready, currently empty)

---

## 🔄 Pipeline Status Summary

```
Training Phase:        ✅ COMPLETE
  ├─ Setup:            ✅ COMPLETE
  ├─ Execution:        ✅ COMPLETE (3 epochs, 750 steps)
  └─ Checkpoints:      ✅ COMPLETE (final checkpoint saved)

Evaluation Phase:      ⏳ IN PROGRESS
  ├─ Script Fix:       ✅ COMPLETE (no_speak() compatibility)
  ├─ Evaluation Run:   ⏳ IN PROGRESS (running now)
  ├─ Report Review:    ⏸️  PENDING
  └─ Metrics Analysis: ⏸️  PENDING

Testing Phase:         ⏸️  PENDING
Preparation Phase:      ⏸️  PENDING
Deployment Phase:       ⏸️  PENDING
Documentation Phase:    ⏸️  PENDING
```

---

## 📚 Additional Resources

- [POST_TRAINING.md](POST_TRAINING.md) - Detailed post-training guide
- [QUICK_GUIDE.md](QUICK_GUIDE.md) - Quick reference guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [README.md](README.md) - Project overview

---

**Last Updated:** 2026-01-25  
**Pipeline Version:** 1.0.0  
**Current Activity:** Running model evaluation (`make evaluate`)
