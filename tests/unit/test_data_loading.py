"""
Tests for data loading utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from datasets import Dataset, DatasetDict
from src.data.load_dataset import load_dataset_from_config, get_dataset_info
from src.utils.error_handling import DatasetError


class TestLoadDatasetFromConfig:
    """Tests for load_dataset_from_config function."""
    
    @patch('src.data.load_dataset.load_dataset')
    def test_load_dataset_success(self, mock_load_dataset, sample_config_dir):
        """Test successful dataset loading."""
        # Mock dataset
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({
                "instruction": ["test1", "test2"],
                "response": ["resp1", "resp2"]
            }),
            "validation": Dataset.from_dict({
                "instruction": ["test1"],
                "response": ["resp1"]
            })
        })
        mock_load_dataset.return_value = mock_dataset
        
        config_path = sample_config_dir / "dataset_config.yaml"
        dataset = load_dataset_from_config(config_path=config_path)
        
        assert isinstance(dataset, DatasetDict)
        assert "train" in dataset
        assert "validation" in dataset
        mock_load_dataset.assert_called_once()
    
    @patch('src.data.load_dataset.load_dataset')
    def test_load_single_dataset_auto_split(self, mock_load_dataset, sample_config_dir):
        """Test auto-splitting of single dataset."""
        # Mock single dataset (not DatasetDict)
        mock_dataset = Dataset.from_dict({
            "instruction": ["test1", "test2", "test3"],
            "response": ["resp1", "resp2", "resp3"]
        })
        mock_load_dataset.return_value = mock_dataset
        
        config_path = sample_config_dir / "dataset_config.yaml"
        dataset = load_dataset_from_config(config_path=config_path)
        
        assert isinstance(dataset, DatasetDict)
        assert "train" in dataset
        assert "validation" in dataset
    
    @patch('src.data.load_dataset.load_dataset')
    def test_load_dataset_with_max_samples(self, mock_load_dataset, sample_config_dir):
        """Test dataset loading with max_samples limit."""
        # Create config with max_samples
        import yaml
        config_path = sample_config_dir / "dataset_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["dataset"]["max_samples"] = 1
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({
                "instruction": ["test1", "test2", "test3"],
                "response": ["resp1", "resp2", "resp3"]
            }),
            "validation": Dataset.from_dict({
                "instruction": ["test1", "test2"],
                "response": ["resp1", "resp2"]
            })
        })
        mock_load_dataset.return_value = mock_dataset
        
        dataset = load_dataset_from_config(config_path=config_path)
        
        assert len(dataset["train"]) == 1
        assert len(dataset["validation"]) == 1
    
    @patch('src.data.load_dataset.load_dataset')
    def test_load_dataset_error(self, mock_load_dataset, sample_config_dir):
        """Test error handling in dataset loading."""
        mock_load_dataset.side_effect = FileNotFoundError("Dataset not found")
        
        config_path = sample_config_dir / "dataset_config.yaml"
        
        with pytest.raises(DatasetError):
            load_dataset_from_config(config_path=config_path)


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""
    
    def test_get_dataset_info(self, mock_dataset):
        """Test getting dataset information."""
        info = get_dataset_info(mock_dataset)
        
        assert "splits" in info
        assert "num_samples" in info
        assert "features" in info
        assert "train" in info["splits"]
        assert "validation" in info["splits"]
        assert info["num_samples"]["train"] == 2
