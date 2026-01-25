"""
Inference script for crisis-agent fine-tuning.
Run inference with the fine-tuned model.
Usage: python scripts/infer.py [--checkpoint CHECKPOINT_PATH] [--prompt PROMPT]
"""

import sys
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import handle_errors, check_cuda_available
from src.utils.json_validator import validate_json_structure, extract_json_from_text
from unsloth import FastLanguageModel


def _get_generation_context(model: Any):
    """
    Get context manager for model generation.
    Uses no_speak() if available, otherwise returns nullcontext.
    """
    if hasattr(model, 'no_speak'):
        return model.no_speak()
    else:
        return nullcontext()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned crisis-agent model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to use for inference (if not provided, will use interactive mode)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--validate-json",
        action="store_true",
        help="Validate JSON in response"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting crisis-agent inference")
    logger.info("=" * 80)
    
    try:
        # Check CUDA
        check_cuda_available()
        
        checkpoint_path = Path(args.checkpoint)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        logger.info("Model loaded successfully")
        
        # Interactive or single prompt mode
        if args.prompt:
            # Single prompt mode
            prompt = args.prompt
            response = _generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            print("\n" + "=" * 80)
            print("PROMPT:")
            print("=" * 80)
            print(prompt)
            print("\n" + "=" * 80)
            print("RESPONSE:")
            print("=" * 80)
            print(response)
            
            if args.validate_json:
                is_valid, parsed, error = validate_json_structure(response, strict=False)
                if is_valid:
                    print("\n" + "=" * 80)
                    print("VALID JSON DETECTED:")
                    print("=" * 80)
                    print(json.dumps(parsed, indent=2))
                else:
                    print(f"\nJSON validation: {error}")
            
            print("=" * 80)
        else:
            # Interactive mode
            print("\n" + "=" * 80)
            print("INTERACTIVE MODE")
            print("Type 'quit' or 'exit' to stop")
            print("=" * 80)
            
            while True:
                try:
                    prompt = input("\nEnter prompt: ").strip()
                    
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not prompt:
                        continue
                    
                    print("\nGenerating response...")
                    response = _generate_response(
                        model, tokenizer, prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                    
                    print("\n" + "-" * 80)
                    print("RESPONSE:")
                    print("-" * 80)
                    print(response)
                    
                    if args.validate_json:
                        is_valid, parsed, error = validate_json_structure(response, strict=False)
                        if is_valid:
                            print("\n" + "-" * 80)
                            print("VALID JSON:")
                            print("-" * 80)
                            print(json.dumps(parsed, indent=2))
                        else:
                            print(f"\nJSON validation: {error}")
                    
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error during inference: {str(e)}")
                    print(f"Error: {str(e)}")
        
        logger.info("Inference completed")
        
    except KeyboardInterrupt:
        logger.warning("\nInference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nInference failed: {str(e)}", exc_info=True)
        sys.exit(1)


def _generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """
    Generate response from model.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    # Format prompt
    full_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with _get_generation_context(model):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (after [/INST])
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[1].strip()
    else:
        response = generated_text[len(full_prompt):].strip()
    
    return response


if __name__ == "__main__":
    main()
