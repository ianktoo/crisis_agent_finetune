# AI-Based Evaluation with Claude

The evaluation pipeline supports optional AI-based quality assessment using Claude API. This provides detailed feedback on response quality beyond structure validation.

## Setup

### 1. Install LangChain and Provider Integrations

Install LangChain and the provider(s) you want to use:

```bash
# For Claude (Anthropic)
pip install langchain langchain-anthropic

# For OpenAI (GPT models)
pip install langchain langchain-openai

# For Gemini (Google)
pip install langchain langchain-google-genai

# Or install all at once
pip install langchain langchain-anthropic langchain-openai langchain-google-genai
```

Or add to `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Note**: This uses LangChain for a standardized interface, making it easy to switch between different LLM providers (Claude, OpenAI, Gemini).

### 2. Set API Key(s)

Set the API key for your chosen provider as an environment variable:

**For Claude (Anthropic):**
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**For Gemini (Google):**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or add to `.env` file:
```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

> **Note**: You can set all three keys if you want to switch between providers easily.

## Usage

### Basic AI Evaluation (Claude - Default)

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai
```

### With Different Providers

**Using OpenAI:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai
```

**Using Gemini:**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
```

**Using Claude (explicit):**
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider anthropic
```

### With Custom Model

```bash
# Claude
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider anthropic --ai-model claude-3-opus-20240229

# OpenAI
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai --ai-model gpt-4o

# Gemini
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini --ai-model gemini-1.5-pro
```

### Limit AI Evaluation Samples (Cost Control)

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-max-samples 50
```

### Combined with Standard Evaluation

The `--ai` flag adds AI evaluation **in addition to** standard evaluation. Both are included in the report:

```bash
# Standard evaluation + AI evaluation
python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai

# Standard evaluation only (default)
python scripts/evaluate.py --checkpoint outputs/checkpoints/final
```

## Evaluation Criteria

Claude evaluates responses on:

1. **Completeness** (0-100): Coverage of all key aspects (FACTS, UNCERTAINTIES, ANALYSIS, GUIDANCE)
2. **Accuracy** (0-100): Correct identification of facts and proper acknowledgment of uncertainties
3. **Actionability** (0-100): Clarity, specificity, and actionability of guidance
4. **Structure** (0-100): Organization and readability
5. **Crisis Appropriateness** (0-100): Appropriateness for severity and type of crisis

## Output

The evaluation report includes an `ai_evaluation` section:

```json
{
  "ai_evaluation": {
    "enabled": true,
    "average_score": 85.3,
    "evaluated_samples": 100,
    "total_samples": 100,
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
        "criterion_scores": {
          "completeness": 90,
          "accuracy": 85,
          "actionability": 80,
          "structure": 85,
          "appropriateness": 90
        },
        "feedback": "Strong response with clear structure..."
      }
    ]
  }
}
```

## Console Output

When using `--ai`, the console shows:

```
================================================================================
EVALUATION SUMMARY
================================================================================
Total samples evaluated: 100
Valid JSON: 0 (0.0%)
Valid structured text: 95 (95.0%)
Total valid responses: 95 (95.0%)
Valid structure (JSON): 0 (0.0%)
Invalid JSON: 0
Invalid structured text: 5

--------------------------------------------------------------------------------
AI EVALUATION SUMMARY (Claude)
--------------------------------------------------------------------------------
Average quality score: 85.3/100
Evaluated samples: 100/100

Criterion averages:
  Completeness: 88.5/100
  Accuracy: 87.2/100
  Actionability: 82.1/100
  Structure: 86.7/100
  Appropriateness: 87.4/100
================================================================================
```

## Cost Considerations

- Claude API charges per token (input + output)
- Each evaluation uses ~500-1000 tokens
- Use `--ai-max-samples` to limit evaluation for cost control
- Example: `--ai-max-samples 50` evaluates only first 50 samples

## Error Handling

If AI evaluation fails (e.g., API key missing, network error), the script:
- Logs the error
- Continues with standard evaluation
- Includes error in report's `ai_evaluation` section

## Makefile Integration

You can add to `Makefile`:

```makefile
evaluate-ai:
	@echo "Evaluating with AI (Claude)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai

evaluate-ai-openai:
	@echo "Evaluating with AI (OpenAI)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai

evaluate-ai-gemini:
	@echo "Evaluating with AI (Gemini)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini
```

Then run:
```bash
make evaluate-ai           # Uses Claude (default)
make evaluate-ai-openai    # Uses OpenAI
make evaluate-ai-gemini    # Uses Gemini
```

## Troubleshooting

### "API_KEY not found"

Set the environment variable for your chosen provider:
```bash
# For Claude
export ANTHROPIC_API_KEY="your_key_here"

# For OpenAI
export OPENAI_API_KEY="your_key_here"

# For Gemini
export GEMINI_API_KEY="your_key_here"
```

### "LangChain not installed"

Install it with the provider integration you need:
```bash
# For Claude
pip install langchain langchain-anthropic

# For OpenAI
pip install langchain langchain-openai

# For Gemini
pip install langchain langchain-google-genai

# Or all at once
pip install langchain langchain-anthropic langchain-openai langchain-google-genai
```

### API Rate Limits

If you hit rate limits:
- Reduce `--ai-max-samples`
- Add delays between API calls (modify `ai_evaluation.py`)
- Use a different Claude model

### High Costs

- Use `--ai-max-samples` to limit samples
- Evaluate on a subset of your validation set
- Consider using cheaper models:
  - Claude: `claude-3-haiku` (faster, cheaper)
  - OpenAI: `gpt-4o-mini` (default, cost-effective)
  - Gemini: `gemini-1.5-flash` (default, fast and free tier available)
