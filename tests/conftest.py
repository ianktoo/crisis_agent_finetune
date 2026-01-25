"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config_dir(temp_dir):
    """Create a temporary config directory with sample configs."""
    config_dir = temp_dir / "configs"
    config_dir.mkdir()
    
    # Sample dataset config
    dataset_config = {
        "dataset": {
            "hf_dataset_name": "test/dataset",
            "train_split": "train",
            "eval_split": "validation",
            "instruction_column": "instruction",
            "response_column": "response",
            "max_samples": 10,
            "shuffle": False,
            "cache_dir": str(temp_dir / "cache"),
            "use_cache": False,
            "prompt_template": "<s>[INST] {instruction} [/INST] {response}</s>",
            "validate_json": False,
            "strict_json": False
        }
    }
    
    with open(config_dir / "dataset_config.yaml", "w") as f:
        yaml.dump(dataset_config, f)
    
    # Sample model config
    model_config = {
        "model": {
            "model_name": "test/model",
            "load_in_4bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "use_flash_attention_2": False,
            "max_seq_length": 512,
            "trust_remote_code": False
        }
    }
    
    with open(config_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    
    # Sample training config
    training_config = {
        "training": {
            "output_dir": str(temp_dir / "outputs" / "checkpoints"),
            "logging_dir": str(temp_dir / "outputs" / "logs"),
            "num_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "warmup_steps": 10,
            "max_steps": -1,
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "save_steps": 100,
            "save_total_limit": 2,
            "save_strategy": "steps",
            "eval_steps": 100,
            "evaluation_strategy": "steps",
            "logging_steps": 10,
            "report_to": [],
            "fp16": False,
            "bf16": False,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "seed": 42
        }
    }
    
    with open(config_dir / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f)
    
    return config_dir


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    from datasets import Dataset, DatasetDict
    
    data = {
        "instruction": ["Test instruction 1", "Test instruction 2"],
        "response": ['{"action": "test"}', '{"action": "test2"}']
    }
    
    train_dataset = Dataset.from_dict(data)
    val_dataset = Dataset.from_dict(data)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = Mock()
    model.config = Mock()
    model.config.use_cache = True
    model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=1000))])
    model.generate = Mock(return_value=Mock())
    model.no_speak = Mock(return_value=Mock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=None)))
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.__len__ = Mock(return_value=1000)
    tokenizer.encode = Mock(return_value=[1, 2, 3])
    tokenizer.decode = Mock(return_value="decoded text")
    tokenizer.return_value = {"input_ids": Mock()}
    return tokenizer


@pytest.fixture
def sample_json_response():
    """Sample JSON response for testing."""
    return {
        "action": "evacuate",
        "priority": "high",
        "reasoning": "Fire detected",
        "resources": ["fire_department"]
    }


@pytest.fixture
def sample_text_with_json():
    """Sample text containing JSON."""
    return 'Here is the response: {"action": "test", "priority": "high"}'
