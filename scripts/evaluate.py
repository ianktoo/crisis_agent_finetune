"""
Main evaluation script for crisis-agent fine-tuning.
Usage: python scripts/evaluate.py [--checkpoint CHECKPOINT_PATH]
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
from src.data.format_records import format_dataset
from src.model.load_model import load_model_from_config, load_tokenizer_only
from src.training.evaluation import evaluate_model, generate_evaluation_report, evaluate_safety_alignment
from unsloth import FastLanguageModel


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned crisis-agent model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: use base model)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_report.json",
        help="Path to save evaluation report"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting crisis-agent model evaluation")
    logger.info("=" * 80)
    
    try:
        # Check CUDA
        check_cuda_available()
        
        # Load dataset
        logger.info("\n[PHASE 1] Loading evaluation dataset...")
        dataset = load_dataset_from_config()
        
        # Format dataset
        logger.info("\n[PHASE 2] Formatting dataset...")
        formatted_dataset = format_dataset(dataset)
        
        # Get evaluation split
        eval_dataset = formatted_dataset.get("validation") or formatted_dataset.get("test")
        if eval_dataset is None:
            logger.warning("No validation/test split found, using train split")
            eval_dataset = formatted_dataset["train"]
        
        # Load model
        logger.info("\n[PHASE 3] Loading model...")
        if args.checkpoint:
            logger.info(f"Loading from checkpoint: {args.checkpoint}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            model, tokenizer = load_model_from_config()
            logger.warning("No checkpoint specified, using base model (not fine-tuned)")
        
        # Evaluate model
        logger.info("\n[PHASE 4] Evaluating model...")
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            max_samples=args.max_samples,
        )
        
        # Generate report
        logger.info("\n[PHASE 5] Generating evaluation report...")
        report_path = generate_evaluation_report(metrics, Path(args.output))
        
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Report saved to: {report_path}")
        logger.info("=" * 80)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total samples evaluated: {metrics['total_samples']}")
        print(f"Valid JSON: {metrics['valid_json']} ({metrics['valid_json_percent']:.1f}%)")
        print(f"Valid structure: {metrics['valid_structure']} ({metrics['valid_structure_percent']:.1f}%)")
        print(f"Invalid JSON: {metrics['invalid_json']}")
        if metrics['errors']:
            print(f"Errors: {len(metrics['errors'])}")
        if metrics['warnings']:
            print(f"Warnings: {len(metrics['warnings'])}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nEvaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
