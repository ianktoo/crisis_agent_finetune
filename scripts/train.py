"""
Main training script for crisis-agent fine-tuning.
Usage: python scripts/train.py [--model-name MODEL_NAME]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import handle_errors, check_cuda_available
from src.data.load_dataset import load_dataset_from_config
from src.data.format_records import format_dataset, tokenize_dataset
from src.model.load_model import load_model_from_config
from src.model.apply_lora import apply_lora_from_config, prepare_model_for_training
from src.training.trainer import create_trainer, train_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train crisis-agent fine-tuning model")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for the final model checkpoint (default: from config or 'final')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: from config)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip existing checkpoints and start training from scratch"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting crisis-agent fine-tuning pipeline")
    logger.info("=" * 80)
    
    if args.model_name:
        logger.info(f"Using custom model name: {args.model_name}")
    
    try:
        # Check CUDA
        check_cuda_available()
        
        # Load dataset
        logger.info("\n[PHASE 1] Loading dataset...")
        dataset = load_dataset_from_config()
        
        # Format dataset
        logger.info("\n[PHASE 2] Formatting dataset...")
        formatted_dataset = format_dataset(dataset)
        
        # Load model
        logger.info("\n[PHASE 3] Loading model...")
        model, tokenizer = load_model_from_config()
        
        # Apply LoRA
        logger.info("\n[PHASE 4] Applying LoRA adapters...")
        model, tokenizer = apply_lora_from_config(model, tokenizer)
        
        # Prepare for training
        model, tokenizer = prepare_model_for_training(model, tokenizer)
        
        # Tokenize dataset
        logger.info("\n[PHASE 5] Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(
            formatted_dataset,
            tokenizer,
            max_length=2048
        )
        
        # Create trainer
        logger.info("\n[PHASE 6] Creating trainer...")
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            final_model_name=args.model_name,
            output_dir=args.output_dir,
        )
        
        # Train model
        logger.info("\n[PHASE 7] Starting training...")
        
        # Check for existing checkpoints to resume from
        checkpoint_to_resume = None
        
        if args.no_resume:
            logger.info("--no-resume flag set: Starting training from scratch (skipping existing checkpoints)")
        else:
            output_dir = Path(trainer.args.output_dir)
            
            # Look for checkpoints in output directory
            if output_dir.exists():
                # Find all checkpoint directories (checkpoint-*)
                checkpoint_dirs = sorted(
                    [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                    key=lambda x: int(x.name.split("-")[1]) if len(x.name.split("-")) > 1 and x.name.split("-")[1].isdigit() else 0,
                    reverse=True
                )
                
                if checkpoint_dirs:
                    checkpoint_to_resume = checkpoint_dirs[0]
                    logger.info(f"Found existing checkpoint: {checkpoint_to_resume.name}")
                    logger.info(f"Resuming training from: {checkpoint_to_resume}")
                else:
                    # Check for "interrupted" checkpoint
                    interrupted_checkpoint = output_dir / "interrupted"
                    if interrupted_checkpoint.exists():
                        checkpoint_to_resume = interrupted_checkpoint
                        logger.info(f"Found interrupted checkpoint, resuming from: {checkpoint_to_resume}")
        
        final_checkpoint = train_model(trainer, resume_from_checkpoint=checkpoint_to_resume)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
