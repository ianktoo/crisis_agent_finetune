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
from src.training.ai_evaluation import evaluate_with_ai, add_ai_evaluation_to_metrics
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
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples to process in parallel (default: 4)"
    )
    parser.add_argument(
        "--fast-generation",
        action="store_true",
        default=True,
        help="Use faster generation settings (greedy decoding) - enabled by default"
    )
    parser.add_argument(
        "--no-fast-generation",
        action="store_false",
        dest="fast_generation",
        help="Disable fast generation (use sampling with temperature)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_report.json",
        help="Path to save evaluation report"
    )
    parser.add_argument(
        "--ai",
        "--ai-eval",
        action="store_true",
        dest="use_ai_eval",
        help="Enable AI-based evaluation (requires API key: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--ai-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "gemini"],
        help="AI provider for evaluation: anthropic (default), openai, or gemini"
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="Model name (optional, uses provider defaults: claude-3-5-sonnet-20241022, gpt-4o-mini, gemini-1.5-flash)"
    )
    parser.add_argument(
        "--ai-max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate with AI (for cost control, default: all samples)"
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
            batch_size=args.batch_size,
            use_fast_generation=args.fast_generation,
            collect_for_ai_eval=args.use_ai_eval,  # Collect prompts/responses if AI eval enabled
        )
        
        # AI-based evaluation (optional)
        if args.use_ai_eval:
            provider_name = args.ai_provider.upper()
            logger.info(f"\n[PHASE 5] Running AI-based evaluation with {provider_name}...")
            try:
                # Get collected prompts and responses from metrics
                prompts = metrics.get("_ai_eval_prompts", [])
                responses = metrics.get("_ai_eval_responses", [])
                
                if prompts and responses:
                    ai_metrics = evaluate_with_ai(
                        prompts=prompts,
                        generated_responses=responses,
                        provider=args.ai_provider,
                        model=args.ai_model,
                        max_samples=args.ai_max_samples,
                    )
                    # Merge AI metrics into base metrics
                    metrics = add_ai_evaluation_to_metrics(metrics, ai_metrics)
                    # Clean up temporary fields
                    metrics.pop("_ai_eval_prompts", None)
                    metrics.pop("_ai_eval_responses", None)
                else:
                    logger.warning("No prompts/responses collected for AI evaluation")
            except Exception as e:
                logger.error(f"AI evaluation failed: {str(e)}")
                logger.warning("Continuing with standard evaluation only")
                metrics["ai_evaluation"] = {
                    "enabled": True,
                    "error": str(e)
                }
        
        # Generate report
        logger.info("\n[PHASE 6] Generating evaluation report...")
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
        print(f"Valid structured text: {metrics['valid_structured_text']} ({metrics['valid_structured_text_percent']:.1f}%)")
        print(f"Total valid responses: {metrics['valid_json'] + metrics['valid_structured_text']} ({metrics['total_valid_percent']:.1f}%)")
        print(f"Valid structure (JSON): {metrics['valid_structure']} ({metrics['valid_structure_percent']:.1f}%)")
        print(f"Invalid JSON: {metrics['invalid_json']}")
        print(f"Invalid structured text: {metrics['invalid_structured_text']}")
        if metrics['errors']:
            print(f"Errors: {len(metrics['errors'])}")
        if metrics['warnings']:
            print(f"Warnings: {len(metrics['warnings'])}")
        
        # AI evaluation summary
        if metrics.get('ai_evaluation', {}).get('enabled'):
            ai_eval = metrics['ai_evaluation']
            provider_name = args.ai_provider.upper() if args.use_ai_eval else "AI"
            if 'error' not in ai_eval:
                print("\n" + "-" * 80)
                print(f"AI EVALUATION SUMMARY ({provider_name})")
                print("-" * 80)
                print(f"Average quality score: {ai_eval.get('average_score', 0):.1f}/100")
                print(f"Evaluated samples: {ai_eval.get('evaluated_samples', 0)}/{ai_eval.get('total_samples', 0)}")
                if 'criterion_averages' in ai_eval:
                    print("\nCriterion averages:")
                    for criterion, score in ai_eval['criterion_averages'].items():
                        print(f"  {criterion.capitalize()}: {score:.1f}/100")
            else:
                print(f"\nAI Evaluation Error: {ai_eval['error']}")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nEvaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
