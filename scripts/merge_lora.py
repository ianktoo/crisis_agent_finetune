"""
LoRA merge script for crisis-agent fine-tuning.
Merges LoRA weights into the base model for deployment.
Usage: python scripts/merge_lora.py [--checkpoint CHECKPOINT_PATH] [--output OUTPUT_PATH]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import handle_errors, check_cuda_available
from unsloth import FastLanguageModel


def main():
    """Main LoRA merge function."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/final_model",
        help="Path to save merged model"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting LoRA merge process")
    logger.info("=" * 80)
    
    try:
        # Check CUDA
        check_cuda_available()
        
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading LoRA checkpoint: {checkpoint_path}")
        logger.info(f"Output path: {output_path}")
        
        # Load model with LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_path),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=False,  # Load full precision for merge
        )
        
        logger.info("Model loaded successfully")
        
        # Merge LoRA weights
        logger.info("Merging LoRA weights into base model...")
        model = FastLanguageModel.merge_and_unload(model)
        
        logger.info("LoRA weights merged successfully")
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Merged model parameters: {total_params:,}")
        
        # Check model size
        try:
            import torch
            if torch.cuda.is_available():
                # Estimate model size
                model_size_gb = sum(p.numel() * 2 for p in model.parameters()) / 1e9  # FP16
                logger.info(f"Estimated model size: {model_size_gb:.2f} GB (FP16)")
        except Exception:
            pass
        
        logger.info("\n" + "=" * 80)
        logger.info("LoRA merge completed successfully!")
        logger.info(f"Merged model saved to: {output_path}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nMerge interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nMerge failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
