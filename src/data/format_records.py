"""
Dataset formatting utilities for crisis-agent fine-tuning.
Converts dataset records into the unified training format for Unsloth.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import Dataset, DatasetDict
from src.utils.logging import get_logger
from src.utils.error_handling import DatasetError, handle_errors, validate_path
from src.utils.json_validator import validate_json_structure

logger = get_logger(__name__)


@handle_errors(error_type=DatasetError)
def format_dataset(
    dataset: DatasetDict,
    config_path: Path = Path("configs/dataset_config.yaml")
) -> DatasetDict:
    """
    Format dataset records into unified training format.
    
    Args:
        dataset: DatasetDict to format
        config_path: Path to dataset configuration YAML file
        
    Returns:
        Formatted DatasetDict ready for training
        
    Raises:
        DatasetError: If formatting fails
    """
    # Load configuration
    config = _load_config(config_path)
    dataset_config = config["dataset"]
    
    instruction_col = dataset_config.get("instruction_column", "instruction")
    response_col = dataset_config.get("response_column", "response")
    prompt_template = dataset_config.get("prompt_template", "{instruction}\n{response}")
    validate_json = dataset_config.get("validate_json", True)
    strict_json = dataset_config.get("strict_json", False)
    
    logger.info(f"Formatting dataset with template: {prompt_template[:50]}...")
    
    # Format each split
    formatted_splits = {}
    total_skipped = 0
    
    for split_name, split_data in dataset.items():
        logger.info(f"Formatting {split_name} split ({len(split_data)} samples)...")
        
        formatted_records = []
        skipped = 0
        
        for idx, record in enumerate(split_data):
            try:
                formatted_record = _format_single_record(
                    record,
                    instruction_col,
                    response_col,
                    prompt_template,
                    validate_json,
                    strict_json
                )
                
                if formatted_record is not None:
                    formatted_records.append(formatted_record)
                else:
                    skipped += 1
                    if strict_json:
                        logger.warning(f"Skipping record {idx} in {split_name} due to validation failure")
                    
            except Exception as e:
                skipped += 1
                logger.error(f"Error formatting record {idx} in {split_name}: {str(e)}")
                if strict_json:
                    raise DatasetError(f"Failed to format record {idx}: {str(e)}") from e
        
        formatted_splits[split_name] = Dataset.from_list(formatted_records)
        total_skipped += skipped
        
        logger.info(
            f"{split_name} split: {len(formatted_records)} formatted, {skipped} skipped"
        )
    
    if total_skipped > 0:
        logger.warning(f"Total skipped records: {total_skipped}")
    
    return DatasetDict(formatted_splits)


def _format_single_record(
    record: Dict[str, Any],
    instruction_col: str,
    response_col: str,
    prompt_template: str,
    validate_json: bool,
    strict_json: bool
) -> Optional[Dict[str, str]]:
    """
    Format a single dataset record.
    
    Args:
        record: Original dataset record
        instruction_col: Name of instruction column
        response_col: Name of response column
        prompt_template: Template string for formatting
        validate_json: Whether to validate JSON in response
        strict_json: Whether to reject invalid JSON
        
    Returns:
        Formatted record with 'text' field, or None if skipped
    """
    # Extract instruction and response
    instruction = record.get(instruction_col, "")
    response = record.get(response_col, "")
    
    if not instruction or not response:
        logger.warning(f"Missing instruction or response in record: {record.keys()}")
        return None
    
    # Validate JSON if requested
    if validate_json:
        try:
            # Try to parse response as JSON
            if isinstance(response, str):
                json.loads(response)
            elif isinstance(response, dict):
                json.dumps(response)  # Validate it's JSON-serializable
        except (json.JSONDecodeError, TypeError) as e:
            if strict_json:
                raise DatasetError(f"Invalid JSON in response: {str(e)}") from e
            logger.warning(f"Response is not valid JSON: {str(e)}")
            # Continue anyway if not strict
    
    # Format using template
    try:
        formatted_text = prompt_template.format(
            instruction=instruction,
            response=response
        )
    except KeyError as e:
        logger.error(f"Template formatting error: {str(e)}")
        # Fallback to simple concatenation
        formatted_text = f"{instruction}\n{response}"
    
    return {
        "text": formatted_text,
        "instruction": instruction,
        "response": response
    }


@handle_errors(error_type=DatasetError)
def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = validate_path(config_path, must_exist=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise DatasetError(f"Empty or invalid configuration file: {config_path}")
    
    return config


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int = 2048,
    text_column: str = "text"
) -> DatasetDict:
    """
    Tokenize formatted dataset for training.
    
    Args:
        dataset: Formatted DatasetDict
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        text_column: Name of text column to tokenize
        
    Returns:
        Tokenized DatasetDict
    """
    logger.info(f"Tokenizing dataset with max_length={max_length}")
    
    def tokenize_function(examples):
        # When batched=True, examples[text_column] is a list of strings
        texts = examples[text_column] if isinstance(examples[text_column], list) else [examples[text_column]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=False
        )
        # Don't add labels here - DataCollatorForLanguageModeling will handle it
        return tokenized
    
    tokenized_dataset = {}
    for split_name, split_data in dataset.items():
        logger.info(f"Tokenizing {split_name} split...")
        # Remove all columns except text_column before tokenizing
        # After tokenization, the tokenizer will add input_ids, attention_mask, etc.
        columns_to_remove = [col for col in split_data.column_names if col != text_column]
        tokenized_dataset[split_name] = split_data.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove + [text_column],  # Remove text_column after tokenization
            desc=f"Tokenizing {split_name}"
        )
    
    return DatasetDict(tokenized_dataset)
