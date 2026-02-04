"""
Upload GGUF file to Hugging Face Hub.

Usage:
    python scripts/upload_gguf_to_hf.py \
        --gguf-file outputs/gguf/model.gguf \
        --repo-name username/model-name-gguf \
        --private
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path (resolve so it works from any cwd)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def _load_env():
    """Load HF_TOKEN from .env (project root and cwd)."""
    try:
        from dotenv import load_dotenv
        use_dotenv = True
    except ImportError:
        use_dotenv = False

    for base in (project_root, Path.cwd()):
        env_file = base / ".env"
        if not env_file.exists():
            continue
        if use_dotenv:
            load_dotenv(env_file, override=False)
        else:
            # Fallback if python-dotenv not installed
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key.upper().startswith("EXPORT "):
                        key = key[7:].strip()
                    value = value.strip().split("#")[0].strip().strip("'\"")
                    if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") and value:
                        os.environ.setdefault(key, value)
            except Exception:
                pass
    # Ensure project .env wins (reload so project root overrides cwd)
    project_env = project_root / ".env"
    if project_env.exists():
        if use_dotenv:
            load_dotenv(project_env, override=True)
        else:
            try:
                for line in project_env.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key.upper().startswith("EXPORT "):
                        key = key[7:].strip()
                    value = value.strip().split("#")[0].strip().strip("'\"")
                    if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") and value:
                        os.environ[key] = value
                        break
            except Exception:
                pass


_load_env()

from src.utils.logging import setup_logging

logger = setup_logging()


def upload_gguf_to_hf(
    gguf_file: Path,
    repo_name: str,
    private: bool = False,
    upload_tokenizer: bool = True,
    tokenizer_dir: Path = None
) -> str:
    """
    Upload GGUF file to Hugging Face Hub.
    
    Args:
        gguf_file: Path to the GGUF file
        repo_name: Hugging Face repository name (format: username/repo-name)
        private: Whether to make repository private
        upload_tokenizer: Whether to also upload tokenizer files
        tokenizer_dir: Directory containing tokenizer files (defaults to gguf_file.parent)
        
    Returns:
        Repository URL
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    
    # Get Hugging Face token (from .env or environment)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        env_locations = [project_root / ".env", Path.cwd() / ".env"]
        logger.error(
            "HF_TOKEN not found. Add your token to a .env file:\n"
            "  In project root: %s\n"
            "  Or in cwd:       %s\n"
            "  Use exactly one line: HF_TOKEN=hf_xxxxxxxx (no spaces around =, not commented)\n"
            "  Run from project root: cd %s\n"
            "  Get a token: https://huggingface.co/settings/tokens",
            env_locations[0], env_locations[1], project_root
        )
        sys.exit(1)
    
    gguf_file = Path(gguf_file)
    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_file}")
    
    logger.info("=" * 80)
    logger.info("Uploading GGUF file to Hugging Face Hub")
    logger.info("=" * 80)
    logger.info(f"GGUF file: {gguf_file}")
    logger.info(f"File size: {gguf_file.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Repository: {repo_name}")
    logger.info(f"Private: {private}")
    
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
    
    # Upload GGUF file
    logger.info(f"Uploading GGUF file: {gguf_file.name}")
    logger.info("This may take several minutes for large files...")
    
    try:
        upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_name,
            token=hf_token,
            commit_message=f"Upload GGUF model: {gguf_file.name}"
        )
        logger.info("GGUF file uploaded successfully!")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise
    
    # Upload tokenizer files if requested
    if upload_tokenizer:
        tokenizer_dir = tokenizer_dir or gguf_file.parent
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
            "chat_template.jinja"
        ]
        
        logger.info("Uploading tokenizer files...")
        for tokenizer_file in tokenizer_files:
            tokenizer_path = tokenizer_dir / tokenizer_file
            if tokenizer_path.exists():
                try:
                    upload_file(
                        path_or_fileobj=str(tokenizer_path),
                        path_in_repo=tokenizer_file,
                        repo_id=repo_name,
                        token=hf_token,
                        commit_message=f"Add tokenizer file: {tokenizer_file}"
                    )
                    logger.info(f"  ✓ {tokenizer_file}")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to upload {tokenizer_file}: {e}")
    
    # Create README
    logger.info("Creating model card (README.md)...")
    readme_content = f"""---
license: mit
base_model: unsloth/Mistral-7B-Instruct-v0.2
tags:
  - crisis-response
  - emergency
  - fine-tuned
  - gguf
  - ai-emergency-kit
  - emergency-assistant
---

# AI Emergency Kit - GGUF Model

**AI Emergency Kit** - Your intelligent crisis response assistant. Fine-tuned Mistral-7B model in GGUF format for efficient local deployment.

## Model Details

- **Base Model**: unsloth/Mistral-7B-Instruct-v0.2
- **Format**: GGUF (Float16)
- **File**: {gguf_file.name}
- **File Size**: {gguf_file.stat().st_size / (1024**3):.2f} GB

## Usage

### Using llama.cpp

```bash
# Download the model
git lfs install
git clone https://huggingface.co/{repo_name}

# Run inference
./llama-cli -m {gguf_file.name} -p "Your prompt here"
```

### Using LM Studio

1. Download the GGUF file from this repository
2. Import into LM Studio
3. Load and chat!

### Using Ollama

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM {gguf_file.name}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create model
ollama create crisis-agent -f Modelfile

# Run
ollama run crisis-agent
```

## About AI Emergency Kit

**AI Emergency Kit** is designed to be your reliable AI companion during crisis situations. It provides structured, JSON-formatted responses with actionable guidance, resource recommendations, and step-by-step instructions to help navigate emergency scenarios.

## Limitations

- Model is trained on synthetic crisis scenarios
- Responses should be validated by human experts
- Not intended for real-time emergency response without human oversight
"""
    
    try:
        upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_name,
            token=hf_token,
            commit_message="Add model card"
        )
        logger.info("README.md uploaded")
    except Exception as e:
        logger.warning(f"Failed to upload README: {e}")
    
    repo_url = f"https://huggingface.co/{repo_name}"
    logger.info("\n" + "=" * 80)
    logger.info("Upload completed successfully!")
    logger.info(f"Model available at: {repo_url}")
    logger.info("=" * 80)
    
    return repo_url


def main():
    """Main upload function."""
    parser = argparse.ArgumentParser(
        description="Upload GGUF file to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload GGUF file with tokenizer
  python scripts/upload_gguf_to_hf.py \\
    --gguf-file outputs/gguf/model.gguf \\
    --repo-name username/crisis-agent-gguf

  # Upload GGUF file only (no tokenizer)
  python scripts/upload_gguf_to_hf.py \\
    --gguf-file outputs/gguf/model.gguf \\
    --repo-name username/crisis-agent-gguf \\
    --no-tokenizer

  # Upload as private repository
  python scripts/upload_gguf_to_hf.py \\
    --gguf-file outputs/gguf/model.gguf \\
    --repo-name username/crisis-agent-gguf \\
    --private
        """
    )
    
    parser.add_argument(
        "--gguf-file",
        type=str,
        required=True,
        help="Path to GGUF file to upload"
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
        "--no-tokenizer",
        action="store_true",
        help="Don't upload tokenizer files"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Directory containing tokenizer files (defaults to GGUF file directory)"
    )
    
    args = parser.parse_args()
    
    try:
        repo_url = upload_gguf_to_hf(
            gguf_file=Path(args.gguf_file),
            repo_name=args.repo_name,
            private=args.private,
            upload_tokenizer=not args.no_tokenizer,
            tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"GGUF model uploaded to: {repo_url}")
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
