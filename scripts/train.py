"""
Main training script for crisis-agent fine-tuning.
Usage: python scripts/train.py [--model-name MODEL_NAME]
"""

import sys
import argparse
import yaml
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


def _load_export_gguf_config(project_root: Path) -> dict:
    """Load export_gguf section from training config; returns empty dict if missing."""
    config_path = project_root / "configs" / "training_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("export_gguf") or {}


def _run_gguf_export(trainer, project_root: Path, export_cfg: dict, logger):
    """Export in-memory model to GGUF (and optionally push to Hub). Frees resources on failure."""
    import importlib.util
    export_gguf_path = project_root / "scripts" / "export_gguf.py"
    spec = importlib.util.spec_from_file_location("export_gguf", export_gguf_path)
    export_gguf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_gguf)
    export_to_gguf_in_memory = export_gguf.export_to_gguf_in_memory
    push_to_hub_gguf_in_memory = export_gguf.push_to_hub_gguf_in_memory
    QUANTIZATION_METHODS = export_gguf.QUANTIZATION_METHODS
    output_dir = Path(export_cfg.get("output_dir", "outputs/gguf"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    quantization = export_cfg.get("quantization", "q4_k_m")
    if quantization not in QUANTIZATION_METHODS:
        logger.warning(f"Unknown quantization {quantization}, using q4_k_m")
        quantization = "q4_k_m"
    logger.info("\n[PHASE 8] Exporting to GGUF (in-memory, no reload)...")
    try:
        gguf_path = export_to_gguf_in_memory(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            output_dir=output_dir,
            quantization=quantization,
            logger=logger,
        )
        logger.info(f"GGUF saved: {gguf_path}")
        push_repo = (export_cfg.get("push_to_hub") or "").strip()
        if push_repo:
            logger.info(f"Pushing GGUF to Hugging Face: {push_repo}")
            push_to_hub_gguf_in_memory(
                model=trainer.model,
                tokenizer=trainer.tokenizer,
                repo_name=push_repo,
                quantization=quantization,
                private=bool(export_cfg.get("push_to_hub_private", False)),
                logger=logger,
            )
        logger.info("LM Studio: lms import " + str(gguf_path))
    except Exception as e:
        logger.warning(f"GGUF export failed (you can run scripts/export_gguf.py later): {e}")


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
    parser.add_argument(
        "--no-export-gguf",
        action="store_true",
        help="Skip automatic GGUF export after training (default: export if enabled in config)"
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
        
        # Optional: export to GGUF in-memory (no reload; see configs/training_config.yaml)
        export_cfg = _load_export_gguf_config(project_root)
        do_export = export_cfg.get("enabled", False) and not args.no_export_gguf
        if do_export:
            _run_gguf_export(
                trainer=trainer,
                project_root=project_root,
                export_cfg=export_cfg,
                logger=logger,
            )
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
