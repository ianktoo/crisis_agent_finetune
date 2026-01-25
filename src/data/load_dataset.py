"""
Dataset loading utilities for crisis-agent fine-tuning.
Handles loading from Hugging Face datasets with caching and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from src.utils.logging import get_logger
from src.utils.error_handling import DatasetError, handle_errors, validate_path

logger = get_logger(__name__)


@handle_errors(error_type=DatasetError)
def load_dataset_from_config(config_path: Path = Path("configs/dataset_config.yaml")) -> DatasetDict:
    """
    Load dataset from Hugging Face using configuration file.
    
    Args:
        config_path: Path to dataset configuration YAML file
        
    Returns:
        DatasetDict with train and validation splits
        
    Raises:
        DatasetError: If dataset loading fails
    """
    # Load configuration
    config = _load_config(config_path)
    dataset_config = config["dataset"]
    
    hf_dataset_name = dataset_config["hf_dataset_name"]
    logger.info(f"Loading dataset: {hf_dataset_name}")
    
    # Check if loading from local JSONL file
    if hf_dataset_name.endswith('.jsonl') or hf_dataset_name.endswith('.json'):
        jsonl_path = Path(hf_dataset_name)
        if not jsonl_path.is_absolute():
            jsonl_path = Path(__file__).parent.parent.parent / jsonl_path
        
        if not jsonl_path.exists():
            raise DatasetError(f"JSONL file not found: {jsonl_path}")
        
        logger.info(f"Loading from local JSONL file: {jsonl_path}")
        # Load JSONL - it returns a DatasetDict with 'train' key
        dataset = load_dataset('json', data_files=str(jsonl_path))
        logger.info(f"Successfully loaded dataset from {jsonl_path}")
        
        # If it's a single dataset, convert to DatasetDict
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
    else:
        # Load from Hugging Face
        # Validate cache directory
        cache_dir = Path(dataset_config.get("cache_dir", "data/local_cache"))
        validate_path(cache_dir, must_exist=False, create_if_missing=True)
        
        # Get Hugging Face token (from config or environment variable)
        hf_token = dataset_config.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # Prepare load_dataset arguments
        load_kwargs = {
            "path": hf_dataset_name,
            "cache_dir": str(cache_dir) if dataset_config.get("use_cache", True) else None,
        }
        
        # Add token if available (for private datasets)
        if hf_token:
            load_kwargs["token"] = hf_token
            logger.info("Using Hugging Face token for authentication")
        
        # Add dataset config name if specified
        if dataset_config.get("dataset_config_name"):
            load_kwargs["name"] = dataset_config["dataset_config_name"]
            logger.info(f"Using dataset config: {dataset_config['dataset_config_name']}")
        
        # Add revision if specified
        if dataset_config.get("revision"):
            load_kwargs["revision"] = dataset_config["revision"]
            logger.info(f"Using dataset revision: {dataset_config['revision']}")
        
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset(**load_kwargs)
            logger.info(f"Successfully loaded dataset: {hf_dataset_name}")
        except FileNotFoundError as e:
            raise DatasetError(
                f"Dataset not found: {hf_dataset_name}. "
                f"Check that the dataset name is correct and you have access to it. "
                f"If it's a private dataset, set HF_TOKEN environment variable."
            ) from e
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise DatasetError(
                    f"Authentication failed for dataset: {hf_dataset_name}. "
                    f"For private datasets, set HF_TOKEN environment variable: "
                    f"export HF_TOKEN='your_token_here'"
                ) from e
            raise DatasetError(f"Failed to load dataset: {error_msg}") from e
    
    # Handle different dataset formats
    if isinstance(dataset, Dataset):
        # Single dataset - split it
        logger.warning("Single dataset provided, splitting into train/validation")
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["test"]
        })
    elif isinstance(dataset, DatasetDict):
        # Already a DatasetDict
        pass
    else:
        raise DatasetError(f"Unexpected dataset type: {type(dataset)}")
    
    # Limit samples if specified
    max_samples = dataset_config.get("max_samples", -1)
    if max_samples > 0:
        logger.info(f"Limiting to {max_samples} samples per split")
        for split_name in dataset.keys():
            if len(dataset[split_name]) > max_samples:
                dataset[split_name] = dataset[split_name].select(range(max_samples))
    
    # Shuffle if specified
    if dataset_config.get("shuffle", True):
        shuffle_seed = dataset_config.get("shuffle_seed", 42)
        logger.info(f"Shuffling dataset with seed {shuffle_seed}")
        for split_name in dataset.keys():
            dataset[split_name] = dataset[split_name].shuffle(seed=shuffle_seed)
    
    # Log dataset statistics
    for split_name, split_data in dataset.items():
        logger.info(f"{split_name} split: {len(split_data)} samples")
        if len(split_data) > 0:
            logger.info(f"Sample columns: {split_data[0].keys()}")
    
    return dataset


@handle_errors(error_type=DatasetError)
def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = validate_path(config_path, must_exist=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise DatasetError(f"Empty or invalid configuration file: {config_path}")
    
    return config


def get_dataset_info(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Get information about the loaded dataset.
    
    Args:
        dataset: DatasetDict to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "splits": list(dataset.keys()),
        "num_samples": {split: len(ds) for split, ds in dataset.items()},
        "features": {split: list(ds.features.keys()) for split, ds in dataset.items()}
    }
    
    return info
