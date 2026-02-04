"""
Tests for data formatting utilities.
"""

import pytest
from datasets import Dataset, DatasetDict
from src.data.format_records import format_dataset, tokenize_dataset
from pathlib import Path


class TestFormatDataset:
    """Tests for format_dataset function."""
    
    def test_format_dataset(self, mock_dataset, sample_config_dir, temp_dir):
        """Test dataset formatting."""
        config_path = sample_config_dir / "dataset_config.yaml"
        
        formatted = format_dataset(mock_dataset, config_path=config_path)
        
        assert isinstance(formatted, DatasetDict)
        assert "train" in formatted
        assert "validation" in formatted
        assert len(formatted["train"]) > 0
    
    def test_format_with_text_column(self, mock_dataset, sample_config_dir):
        """Test that formatted dataset has text column."""
        config_path = sample_config_dir / "dataset_config.yaml"
        
        formatted = format_dataset(mock_dataset, config_path=config_path)
        
        # Check that text column exists
        assert "text" in formatted["train"][0]
        assert "[INST]" in formatted["train"][0]["text"]
    
    def test_format_preserves_original_columns(self, mock_dataset, sample_config_dir):
        """Test that original columns are preserved."""
        config_path = sample_config_dir / "dataset_config.yaml"
        
        formatted = format_dataset(mock_dataset, config_path=config_path)
        
        # Check that instruction and response are preserved
        assert "instruction" in formatted["train"][0]
        assert "response" in formatted["train"][0]


class TestTokenizeDataset:
    """Tests for tokenize_dataset function."""
    
    def test_tokenize_dataset(self, mock_dataset, mock_tokenizer):
        """Test dataset tokenization."""
        # First format the dataset
        from src.data.format_records import format_dataset
        from pathlib import Path
        import tempfile
        import yaml
        
        # Create a minimal config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "dataset": {
                    "instruction_column": "instruction",
                    "response_column": "response",
                    "prompt_template": "{instruction}\n{response}"
                }
            }
            yaml.dump(config, f)
            config_path = Path(f.name)
        
        try:
            formatted = format_dataset(mock_dataset, config_path=config_path)
            
            # Mock tokenizer: must accept same kwargs as real tokenizer (truncation, max_length, etc.)
            def tokenize_func(texts, **kwargs):
                n = len(texts) if isinstance(texts, list) else 1
                return {
                    "input_ids": [[1, 2, 3]] * n,
                    "attention_mask": [[1, 1, 1]] * n,
                }
            
            mock_tokenizer.side_effect = tokenize_func
            
            tokenized = tokenize_dataset(
                formatted,
                mock_tokenizer,
                max_length=512,
                text_column="text"
            )
            
            assert isinstance(tokenized, DatasetDict)
            assert "train" in tokenized
        finally:
            config_path.unlink()
