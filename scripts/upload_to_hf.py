"""
Upload script for crisis-agent fine-tuning.
Uploads trained models to Hugging Face Hub.
Usage: python scripts/upload_to_hf.py [--checkpoint CHECKPOINT_PATH] [--repo-name REPO_NAME]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import handle_errors

logger = setup_logging()


@handle_errors()
def upload_to_huggingface(
    checkpoint_path: Path,
    repo_name: str,
    private: bool = False,
    merged: bool = False,
    commit_message: Optional[str] = None
) -> str:
    """
    Upload model to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to model checkpoint or merged model
        repo_name: Hugging Face repository name (format: username/repo-name)
        private: Whether to make repository private
        merged: Whether this is a merged model (vs LoRA checkpoint)
        commit_message: Custom commit message
        
    Returns:
        Repository URL
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    
    # Get Hugging Face token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.error(
            "HF_TOKEN not found. Set it in .env file or environment:\n"
            "  export HF_TOKEN='your_token_here'\n"
            "  Or add to .env: HF_TOKEN=your_token_here"
        )
        sys.exit(1)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info("=" * 80)
    logger.info("Uploading model to Hugging Face Hub")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Repository: {repo_name}")
    logger.info(f"Private: {private}")
    logger.info(f"Merged model: {merged}")
    
    # Initialize Hugging Face API
    api = HfApi(token=hf_token)
    
    # Create repository if it doesn't exist
    try:
        logger.info(f"Creating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        logger.info("Repository created/verified")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Repository {repo_name} already exists, continuing...")
        else:
            raise
    
    # Prepare commit message
    if not commit_message:
        commit_message = f"Upload {'merged ' if merged else ''}crisis-agent model"
        if merged:
            commit_message += " (merged LoRA weights)"
        else:
            commit_message += " (LoRA checkpoint)"
    
    # Upload model files
    logger.info(f"Uploading files from {checkpoint_path}...")
    logger.info("This may take several minutes for large models...")
    
    try:
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            token=hf_token,
            commit_message=commit_message,
            ignore_patterns=[".git*", "__pycache__", "*.pyc", ".DS_Store"]
        )
        logger.info("Upload completed successfully!")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise
    
    # Create or update README if it doesn't exist
    readme_path = checkpoint_path / "README.md"
    if not readme_path.exists():
        logger.info("Creating model card (README.md)...")
        model_card = _generate_model_card(repo_name, merged)
        readme_path.write_text(model_card, encoding='utf-8')
        
        # Upload README
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            token=hf_token,
            commit_message="Add model card",
            ignore_patterns=[".git*", "__pycache__", "*.pyc", ".DS_Store"]
        )
    
    repo_url = f"https://huggingface.co/{repo_name}"
    logger.info("\n" + "=" * 80)
    logger.info("Upload completed successfully!")
    logger.info(f"Model available at: {repo_url}")
    logger.info("=" * 80)
    
    return repo_url


def _generate_model_card(repo_name: str, merged: bool) -> str:
    """Generate a basic model card."""
    model_type = "merged model" if merged else "LoRA checkpoint"
    
    card = f"""---
license: mit
base_model: unsloth/Mistral-7B-Instruct-v0.2
tags:
  - crisis-response
  - emergency
  - fine-tuned
  - lora
  - ai-emergency-kit
  - emergency-assistant
datasets:
  - ianktoo/crisis-response-training
---

# AI Emergency Kit - {repo_name.split('/')[-1]}

**AI Emergency Kit** - Your intelligent crisis response assistant. Fine-tuned Mistral-7B model that provides structured, actionable guidance during emergency situations.

## Model Details

- **Base Model**: unsloth/Mistral-7B-Instruct-v0.2
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Model Type**: {model_type}
- **Training Dataset**: ianktoo/crisis-response-training

## Usage

### Using Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "{repo_name}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Format prompt
prompt = "<s>[INST] Your crisis scenario here [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Use model for inference
```

## About AI Emergency Kit

**AI Emergency Kit** is designed to be your reliable AI companion during crisis situations. It provides structured, JSON-formatted responses with actionable guidance, resource recommendations, and step-by-step instructions to help navigate emergency scenarios.

## Training Details

This model was fine-tuned using the crisis-agent fine-tuning pipeline to create the **AI Emergency Kit**.

## Limitations

- Model is trained on synthetic crisis scenarios
- Responses should be validated by human experts
- Not intended for real-time emergency response without human oversight

## Citation

If you use this model, please cite:

```bibtex
@software{{crisis_agent_model,
  title = {{Crisis Agent Model}},
  author = {{Your Name}},
  year = {{2026}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```
"""
    return card


def main():
    """Main upload function."""
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned crisis-agent model to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or merged model directory"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repository name (format: username/repo-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Indicate this is a merged model (not LoRA checkpoint)"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for upload"
    )
    
    args = parser.parse_args()
    
    try:
        repo_url = upload_to_huggingface(
            checkpoint_path=Path(args.checkpoint),
            repo_name=args.repo_name,
            private=args.private,
            merged=args.merged,
            commit_message=args.commit_message
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Model uploaded to: {repo_url}")
        print("\nNext steps:")
        print(f"1. Visit {repo_url} to view your model")
        print("2. Edit the README.md to add more details")
        print("3. Share the model with others!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nUpload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUpload failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
