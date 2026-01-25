"""
Model loading utilities for crisis-agent fine-tuning.
Handles loading Mistral-8B with 4-bit quantization and Unsloth optimizations.
"""

import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from src.utils.logging import get_logger
from src.utils.error_handling import ModelError, handle_errors, validate_path, check_cuda_available

logger = get_logger(__name__)


@handle_errors(error_type=ModelError)
def load_model_from_config(
    config_path: Path = Path("configs/model_config.yaml")
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from configuration file.
    
    Args:
        config_path: Path to model configuration YAML file
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ModelError: If model loading fails
    """
    # Load configuration
    config = _load_config(config_path)
    model_config = config["model"]
    
    # Check CUDA availability
    cuda_available = check_cuda_available()
    if not cuda_available:
        logger.warning("CUDA not available. Training will be very slow on CPU.")
    
    model_name = model_config.get("model_name", "unsloth/Mistral-7B-Instruct-v0.2")
    logger.info(f"Loading model: {model_name}")
    
    # Prepare model arguments
    model_args = {
        "max_seq_length": model_config.get("max_seq_length", 2048),
        "dtype": None,  # Auto-detect
        "load_in_4bit": model_config.get("load_in_4bit", True),
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }
    
    # Load model with Unsloth (Unsloth handles quantization internally)
    try:
        import torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(
            model_config.get("bnb_4bit_compute_dtype", "float16"),
            torch.float16
        )
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_args["max_seq_length"],
            dtype=compute_dtype,
            load_in_4bit=model_args["load_in_4bit"],
            trust_remote_code=model_args["trust_remote_code"],
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        raise ModelError(f"Failed to load model: {str(e)}") from e
    
    # Log model info
    _log_model_info(model, tokenizer)
    
    return model, tokenizer


@handle_errors(error_type=ModelError)
def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = validate_path(config_path, must_exist=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ModelError(f"Empty or invalid configuration file: {config_path}")
    
    return config


def _log_model_info(model, tokenizer):
    """Log model and tokenizer information."""
    try:
        # Get model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Log memory usage if CUDA available
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        except Exception:
            pass
            
    except Exception as e:
        logger.warning(f"Could not log model info: {str(e)}")


def load_tokenizer_only(
    model_name: str = "unsloth/Mistral-7B-Instruct-v0.2"
) -> AutoTokenizer:
    """
    Load only the tokenizer (useful for inference or evaluation).
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tokenizer instance
    """
    logger.info(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        raise ModelError(f"Failed to load tokenizer: {str(e)}") from e
