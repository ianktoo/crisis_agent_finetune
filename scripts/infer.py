"""
Inference script for crisis-agent fine-tuning.
Run inference with the fine-tuned model.
Usage: python scripts/infer.py [--checkpoint CHECKPOINT_PATH] [--prompt PROMPT]
"""

import sys
import argparse
import json
import re
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
        default=0.3,
        help="Sampling temperature (lower = more deterministic, default: 0.3)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (higher = less repetition, default: 1.1)"
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
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
            
            print("\n" + "=" * 80)
            print("PROMPT:")
            print("=" * 80)
            print(prompt)
            print("\n" + "=" * 80)
            print("RESPONSE:")
            print("=" * 80)
            if response and len(response.strip()) > 0:
                print(response)
            else:
                print("(Empty response - model may need different generation parameters)")
                logger.warning("Generated response is empty. Try adjusting temperature or max_new_tokens.")
            
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
    temperature: float = 0.3,
    repetition_penalty: float = 1.1
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
    
    # Generate with improved parameters
    # Note: Lower temperature and repetition_penalty might cause empty responses
    # If response is empty, try increasing temperature to 0.5-0.7
    with _get_generation_context(model):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.1),  # Ensure minimum temperature
            do_sample=True if temperature > 0.1 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,  # Reduce repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (after [/INST])
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[1].strip()
    else:
        response = generated_text[len(full_prompt):].strip()
    
    # Remove trailing </s> tokens
    response = response.replace("</s>", "").strip()
    
    # Clean up response - stop at natural boundaries (but be less aggressive)
    stop_sequences = ["[INST]", "\n\n\n\n"]  # Only stop at very clear boundaries
    for stop_seq in stop_sequences:
        if stop_seq in response:
            response = response.split(stop_seq)[0].strip()
    
    # Clean up excessive repetition patterns (like "of of of", "of the", etc.)
    # Remove standalone "of" words and excessive "of" patterns
    # Pattern: " of " that appears multiple times in a row or excessively
    response = re.sub(r'\s+of\s+of\s+', ' ', response)  # "of of" -> single space
    response = re.sub(r'\s+of\s+of\s+of\s+', ' ', response)  # "of of of" -> single space
    response = re.sub(r'\s+of\s+the\s+of\s+', ' ', response)  # "of the of" -> single space
    
    # Process lines and stop at first serious corruption
    lines = response.split('\n')
    cleaned_lines = []
    corruption_detected = False
    
    for line in lines:
        if corruption_detected:
            break
            
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Detect corruption indicators
        of_count = line.count(' of ')
        total_words = len(line.split())
        of_ratio = (of_count / total_words) if total_words > 0 else 0
        
        # Random characters/numbers pattern (like "5-5 of 5 of of1 of4")
        random_char_pattern = re.search(r'[0-9]+\s*[-]+\s*[0-9]+', line)
        excessive_numbers = len(re.findall(r'\b\d+\b', line)) > 5
        
        # Non-ASCII garbage (like "ď", "û", "ï")
        non_ascii_garbage = len(re.findall(r'[^\x00-\x7F]', line)) > 3
        
        # Excessive special characters
        special_chars = len(re.findall(r'[^\w\s\.\,\!\?\:\-]', line))
        special_char_ratio = (special_chars / len(line)) if len(line) > 0 else 0
        
        # Stop if we detect serious corruption
        if (of_ratio > 0.3 or 
            (of_count > 2 and total_words < 10) or
            random_char_pattern or
            (excessive_numbers and of_count > 2) or
            non_ascii_garbage or
            special_char_ratio > 0.2):
            corruption_detected = True
            break
        
        # Clean up the line: remove excessive "of" patterns
        cleaned_line = re.sub(r'\s+of\s+(of\s+)+', ' ', line)  # Multiple "of"s
        cleaned_line = re.sub(r'\s+of\s+the\s+(of\s+)+', ' ', cleaned_line)  # "of the of"
        cleaned_line = re.sub(r'\s+of\s+$', '', cleaned_line)  # Trailing "of"
        cleaned_line = re.sub(r'^\s+of\s+', '', cleaned_line)  # Leading "of"
        
        # Remove isolated random characters/numbers at end of line
        cleaned_line = re.sub(r'\s+[0-9\-\s]+$', '', cleaned_line)
        
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
    
    # Join and clean up extra whitespace
    cleaned_response = '\n'.join(cleaned_lines)
    cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)  # Multiple newlines -> double
    cleaned_response = re.sub(r'[ \t]+', ' ', cleaned_response)  # Multiple spaces -> single
    
    # If we have content, return it; otherwise return original response
    if cleaned_response.strip():
        return cleaned_response.strip()
    else:
        # Fallback: return original response if cleaning removed everything
        return response.strip()


if __name__ == "__main__":
    main()
