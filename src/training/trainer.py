"""
Training utilities for crisis-agent fine-tuning.
Handles training loop, checkpointing, and error recovery.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported
from datasets import DatasetDict
from src.utils.logging import get_logger
from src.utils.error_handling import TrainingError, handle_errors, validate_path, handle_cuda_oom

logger = get_logger(__name__)


@handle_errors(error_type=TrainingError)
@handle_cuda_oom
def create_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config_path: Path = Path("configs/training_config.yaml"),
    final_model_name: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Trainer:
    """
    Create a Trainer instance for fine-tuning.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config_path: Path to training configuration YAML file
        final_model_name: Custom name for final checkpoint (overrides config)
        output_dir: Custom output directory for checkpoints (overrides config)
        
    Returns:
        Configured Trainer instance
        
    Raises:
        TrainingError: If trainer creation fails
    """
    # Load configuration
    config = _load_config(config_path)
    training_config = config["training"]
    
    logger.info("Creating trainer with configuration...")
    
    # Prepare output directories (command-line argument overrides config)
    if output_dir is None:
        output_dir = Path(training_config["output_dir"])
    else:
        output_dir = Path(output_dir)
    logging_dir = Path(training_config["logging_dir"])
    
    validate_path(output_dir, must_exist=False, create_if_missing=True)
    validate_path(logging_dir, must_exist=False, create_if_missing=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        
        # Training
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_config.get("learning_rate", 2.0e-4)),
        warmup_steps=training_config.get("warmup_steps", 100),
        max_steps=training_config.get("max_steps", -1),
        
        # Optimization
        optim=training_config.get("optim", "adamw_torch"),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        
        # Checkpointing
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        save_strategy=training_config.get("save_strategy", "steps"),
        
        # Evaluation
        eval_steps=training_config.get("eval_steps", 500) if eval_dataset else None,
        eval_strategy=training_config.get("eval_strategy", "steps") if eval_dataset else "no",
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        report_to=training_config.get("report_to", []),
        
        # Precision
        fp16=training_config.get("fp16", True) and not is_bfloat16_supported(),
        bf16=training_config.get("bf16", False) and is_bfloat16_supported(),
        
        # Other
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        seed=training_config.get("seed", 42),
        
        # Unsloth optimizations
        fsdp="",
        gradient_checkpointing=True,
    )
    
    # Use DataCollatorForLanguageModeling for causal LM
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    logger.info("Trainer created successfully")
    logger.info(f"Training will save checkpoints to: {output_dir}")
    logger.info(f"Logs will be saved to: {logging_dir}")
    
    # Store final model name in trainer for later use
    if final_model_name is None:
        final_model_name = training_config.get("final_model_name", "final")
    trainer.final_model_name = final_model_name
    
    return trainer


@handle_errors(error_type=TrainingError)
@handle_cuda_oom
def train_model(
    trainer: Trainer,
    resume_from_checkpoint: Optional[Path] = None
) -> Path:
    """
    Train the model with error handling and checkpoint recovery.
    
    Args:
        trainer: Trainer instance
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
        
    Returns:
        Path to final checkpoint
        
    Raises:
        TrainingError: If training fails
    """
    logger.info("Starting training...")
    
    try:
        # Train the model
        checkpoint_path = None
        if resume_from_checkpoint:
            checkpoint_path = str(resume_from_checkpoint)
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        trainer.train(resume_from_checkpoint=checkpoint_path)
        
        # Save final model
        final_model_name = getattr(trainer, 'final_model_name', 'final')
        final_checkpoint = Path(trainer.args.output_dir) / final_model_name
        trainer.save_model(str(final_checkpoint))
        logger.info(f"Training completed. Final model saved to: {final_checkpoint}")
        
        return final_checkpoint
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory during training!")
            logger.error("Consider:")
            logger.error("  - Reducing batch size")
            logger.error("  - Reducing max_seq_length")
            logger.error("  - Increasing gradient_accumulation_steps")
            logger.error("  - Using gradient checkpointing (already enabled)")
            raise TrainingError("CUDA out of memory") from e
        raise
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        # Try to save current checkpoint
        try:
            trainer.save_model(str(Path(trainer.args.output_dir) / "interrupted"))
            logger.info("Saved interrupted checkpoint")
        except Exception as save_error:
            logger.error(f"Failed to save interrupted checkpoint: {str(save_error)}")
        raise
    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}") from e


@handle_errors(error_type=TrainingError)
def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = validate_path(config_path, must_exist=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise TrainingError(f"Empty or invalid configuration file: {config_path}")
    
    return config
