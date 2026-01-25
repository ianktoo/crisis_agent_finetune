"""
LoRA adapter application for crisis-agent fine-tuning.
Applies LoRA adapters to the base model with optimal configuration for 16GB VRAM.
"""

import yaml
from pathlib import Path
from typing import Tuple, Any, Dict
from unsloth import FastLanguageModel
from src.utils.logging import get_logger
from src.utils.error_handling import ModelError, handle_errors, validate_path

logger = get_logger(__name__)


@handle_errors(error_type=ModelError)
def apply_lora_from_config(
    model: Any,
    tokenizer: Any,
    config_path: Path = Path("configs/model_config.yaml")
) -> Tuple[Any, Any]:
    """
    Apply LoRA adapters to model based on configuration.
    
    Args:
        model: Base model instance
        tokenizer: Tokenizer instance
        config_path: Path to model configuration YAML file
        
    Returns:
        Tuple of (model_with_lora, tokenizer)
        
    Raises:
        ModelError: If LoRA application fails
    """
    # Load configuration
    config = _load_config(config_path)
    lora_config = config["model"]["lora"]
    
    logger.info("Applying LoRA adapters to model...")
    logger.info(f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, "
                f"dropout={lora_config['lora_dropout']}")
    
    try:
        # Apply LoRA using Unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=True,  # Save memory
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        logger.info("LoRA adapters applied successfully")
        
        # Log LoRA statistics
        _log_lora_info(model)
        
    except Exception as e:
        raise ModelError(f"Failed to apply LoRA adapters: {str(e)}") from e
    
    return model, tokenizer


@handle_errors(error_type=ModelError)
def _load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    config_path = validate_path(config_path, must_exist=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ModelError(f"Empty or invalid configuration file: {config_path}")
    
    return config


def _log_lora_info(model: Any):
    """Log LoRA adapter information."""
    try:
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}% of total)")
        
        # Try to get LoRA-specific info
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
            
    except Exception as e:
        logger.warning(f"Could not log LoRA info: {str(e)}")


def prepare_model_for_training(
    model: Any,
    tokenizer: Any,
    max_seq_length: int = 2048
) -> Tuple[Any, Any]:
    """
    Prepare model for training with optimal settings.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (prepared_model, tokenizer)
    """
    logger.info(f"Preparing model for training (max_seq_length={max_seq_length})...")
    
    try:
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Enable training mode
        model.config.use_cache = False  # Disable cache for training
        
        logger.info("Model prepared for training")
        
    except Exception as e:
        logger.warning(f"Some optimizations may not be applied: {str(e)}")
    
    return model, tokenizer
