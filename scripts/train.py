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
        final_checkpoint = train_model(trainer)
        
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
