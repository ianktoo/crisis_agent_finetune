# Evaluation Guide

Comprehensive guide to evaluating your trained model.

---

## Evaluation Methods

The pipeline supports two evaluation approaches:

| Method | Description | Cost |
|--------|-------------|------|
| **Standard** | Structure and JSON validation | Free |
| **AI-Based** | Quality assessment via LLM | API costs |

---

## Standard Evaluation

### Quick Start

```bash
make evaluate
```

### What It Checks

1. **Valid JSON** - Response is valid JSON
2. **Valid Structured Text** - Contains FACTS, UNCERTAINTIES, ANALYSIS, GUIDANCE
3. **Valid Structure** - Proper crisis response format
4. **Error Analysis** - Logs invalid responses

### Command Options

```bash
# Basic evaluation
python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# More samples
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --max-samples 200

# Custom output file
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/custom_report.json
```

### Output Report

**File:** `outputs/evaluation_report.json`

```json
{
  "total_samples": 100,
  "valid_json": 85,
  "valid_json_percentage": 85.0,
  "valid_structured_text": 95,
  "valid_structured_text_percentage": 95.0,
  "valid_structure": 90,
  "valid_structure_percentage": 90.0,
  "errors": [
    {
      "sample_idx": 12,
      "error": "Missing GUIDANCE section"
    }
  ]
}
```

---

## AI-Based Evaluation

### Overview

AI evaluation uses Claude, OpenAI, or Gemini to assess response quality beyond structure validation.

### Setup

1. **Install dependencies** (included in requirements.txt):
   ```bash
   pip install langchain langchain-anthropic langchain-openai langchain-google-genai
   ```

2. **Set API keys** in `.env`:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...    # For Claude
   OPENAI_API_KEY=sk-...           # For OpenAI
   GEMINI_API_KEY=...              # For Gemini
   ```

### Commands

```bash
# Claude (default)
make evaluate-ai

# OpenAI
make evaluate-ai-openai

# Gemini
make evaluate-ai-gemini

# With sample limit (cost control)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/final \
  --ai \
  --ai-max-samples 50
```

### Evaluation Criteria

AI evaluates responses on 5 dimensions (0-100 each):

| Criterion | Description |
|-----------|-------------|
| **Completeness** | Covers all key aspects (FACTS, UNCERTAINTIES, etc.) |
| **Accuracy** | Facts correct, uncertainties acknowledged |
| **Actionability** | Guidance is clear and actionable |
| **Structure** | Well-organized and readable |
| **Appropriateness** | Suitable for crisis severity/type |

### AI Evaluation Output

```json
{
  "ai_evaluation": {
    "enabled": true,
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "average_score": 85.3,
    "evaluated_samples": 50,
    "criterion_averages": {
      "completeness": 88.5,
      "accuracy": 87.2,
      "actionability": 82.1,
      "structure": 86.7,
      "appropriateness": 87.4
    },
    "scores": [
      {
        "sample_idx": 0,
        "overall_score": 85,
        "criterion_scores": {...},
        "feedback": "Strong response with clear structure..."
      }
    ]
  }
}
```

### Console Output

```
================================================================================
EVALUATION SUMMARY
================================================================================
Total samples evaluated: 100
Valid JSON: 85 (85.0%)
Valid structured text: 95 (95.0%)
Total valid responses: 95 (95.0%)

--------------------------------------------------------------------------------
AI EVALUATION SUMMARY (Claude)
--------------------------------------------------------------------------------
Average quality score: 85.3/100
Evaluated samples: 50/100

Criterion averages:
  Completeness: 88.5/100
  Accuracy: 87.2/100
  Actionability: 82.1/100
  Structure: 86.7/100
  Appropriateness: 87.4/100
================================================================================
```

---

## AI Providers Comparison

| Provider | Model | Speed | Cost | Quality |
|----------|-------|-------|------|---------|
| Claude | claude-3-5-sonnet | Fast | Medium | Excellent |
| OpenAI | gpt-4o-mini | Fast | Low | Good |
| Gemini | gemini-1.5-flash | Very Fast | Low/Free | Good |

### Recommended Usage

- **Development**: Gemini (free tier available)
- **Production evaluation**: Claude or OpenAI
- **Cost-conscious**: `--ai-max-samples 50`

---

## Manual Testing

### Interactive Mode

```bash
make infer
```

Then type crisis scenarios:
```
Enter prompt: A building is on fire with people trapped
[Model generates response]

Enter prompt: quit
```

### Single Prompt Testing

```bash
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "A building is on fire with people trapped" \
  --validate-json
```

### Compare Base vs Fine-tuned

```bash
# Test base model
python scripts/infer.py \
  --checkpoint unsloth/Mistral-7B-Instruct-v0.2 \
  --prompt "Your test prompt"

# Test fine-tuned
python scripts/infer.py \
  --checkpoint outputs/checkpoints/final \
  --prompt "Your test prompt"
```

---

## Evaluation Best Practices

### 1. Start with Standard Evaluation

```bash
make evaluate
```

Check for:
- Valid JSON percentage > 80%
- Valid structure percentage > 85%

### 2. Sample Manual Testing

Test 5-10 diverse scenarios:
- Fire emergencies
- Medical situations
- Natural disasters
- Security incidents

### 3. AI Evaluation (if needed)

```bash
# Start small
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-max-samples 20

# Full evaluation if results look good
make evaluate-ai
```

### 4. Document Results

Save evaluation reports:
```bash
cp outputs/evaluation_report.json outputs/eval_$(date +%Y%m%d).json
```

---

## Interpreting Results

### Good Results

| Metric | Good | Excellent |
|--------|------|-----------|
| Valid JSON | > 80% | > 95% |
| Valid Structure | > 85% | > 95% |
| AI Score | > 75 | > 85 |

### Warning Signs

- Valid JSON < 70%: Check training data format
- Structure issues: Review prompt template
- Low AI scores: May need more training data

### When to Retrain

Consider retraining if:
- Valid responses < 80%
- AI score < 70
- Consistent errors in specific areas

---

## Cost Management

### Estimate AI Evaluation Costs

| Samples | Approx Cost (Claude) | Approx Cost (GPT-4o-mini) |
|---------|---------------------|---------------------------|
| 50 | ~$0.50 | ~$0.10 |
| 100 | ~$1.00 | ~$0.20 |
| 500 | ~$5.00 | ~$1.00 |

### Cost-Saving Tips

1. Use `--ai-max-samples` to limit samples
2. Use Gemini for development (free tier)
3. Only run AI eval on final models
4. Cache results for comparison

---

## Troubleshooting

### "API key not found"

```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set in .env
echo "ANTHROPIC_API_KEY=your_key" >> .env
```

### "Rate limit exceeded"

Reduce sample count or add delays:
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-max-samples 20
```

### "LangChain not installed"

```bash
pip install langchain langchain-anthropic langchain-openai langchain-google-genai
```

See [Troubleshooting](Troubleshooting.md) for more solutions.

---

[← Training](Training.md) | [Home](Home.md) | [Deployment →](Deployment.md)
