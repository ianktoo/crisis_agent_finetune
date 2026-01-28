# Quick Guide: After Training

**Simple 4-step guide** for anyone who just finished training their model.

---

## ✅ Step 1: Test Your Model

```bash
make evaluate
```

**What it does:** Checks if your model works correctly and shows performance scores.

**Takes:** 2–5 minutes

**You'll see:** A report showing valid structured text (FACTS, UNCERTAINTIES, etc.), valid JSON, and similar metrics.

**Optional – AI evaluation (Claude, OpenAI, or Gemini):**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
```
Requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` in `.env`. See [docs/ai-evaluation.md](docs/ai-evaluation.md).

---

## ✅ Step 2: Try It Out

```bash
make infer
```

**What it does:** Lets you chat with your model interactively.

**How to use:**
1. Type a crisis scenario (e.g., "A building is on fire")
2. Press Enter
3. See the model's response
4. Type `quit` when done

**Example:**
```
Enter prompt: A building is on fire with people trapped inside
[Model responds with JSON action plan]
```

---

## ✅ Step 3: Make It Ready to Share

```bash
make merge
```

**What it does:** Combines all the training files into one complete model.

**Takes:** 5-10 minutes

**Why:** Makes it easier to share and use the model later.

---

## ✅ Step 4: Share on Hugging Face

### First Time Setup (One Time Only)

1. **Create Hugging Face account:** https://huggingface.co/join
2. **Get your token:** https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose "Write" permission
   - Copy the token (starts with `hf_...`)
3. **Add token to `.env` file:**
   ```bash
   # Open .env file and add:
   HF_TOKEN=hf_your_token_here
   ```

### Upload Your Model

```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name YOUR_USERNAME/crisis-agent-v1 \
  --private
```

**Replace:**
- `YOUR_USERNAME` with your Hugging Face username
- `crisis-agent-v1` (maintains consistency) or any name you want

**Example:**
```bash
python scripts/upload_to_hf.py \
  --checkpoint outputs/final_model \
  --repo-name ianktoo/crisis-agent-v1 \
  --private
```

**Takes:** 10-30 minutes (depends on internet speed)

**When done:** Visit `https://huggingface.co/YOUR_USERNAME/crisis-agent-v1` to see your model! The model card will show it as **AI Emergency Kit**.

---

## 🎯 That's It!

Your model is now:
- ✅ Tested and working
- ✅ Ready to use
- ✅ Shared on Hugging Face

---

## ❓ Common Questions

**Q: What if evaluation shows low scores?**  
A: Check the report in `outputs/evaluation_report.json` for details. Try AI evaluation (`--ai`) for quality feedback. You might need to retrain with different settings.

**Q: Can I skip merging?**  
A: Yes, but merged models are easier to use. You can upload the LoRA checkpoint directly if you prefer.

**Q: What if upload fails?**  
A: Check that your `HF_TOKEN` is set correctly. Run `echo $HF_TOKEN` to verify.

**Q: How do I use the model after uploading?**  
A: Others can load it with:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("YOUR_USERNAME/crisis-agent-v1")
```

---

## 📚 Need More Help?

- **Detailed guide:** See [POST_TRAINING.md](POST_TRAINING.md)
- **AI evaluation:** See [docs/ai-evaluation.md](docs/ai-evaluation.md) (Claude, OpenAI, Gemini)
- **Troubleshooting:** Check the "Troubleshooting" section in POST_TRAINING.md
- **Training issues:** See [DEPLOYMENT.md](DEPLOYMENT.md)

---

**Happy deploying! 🚀**
