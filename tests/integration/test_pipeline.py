"""
Integration tests for the full pipeline.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, DatasetDict


class TestDatasetLoading:
    """Integration tests for dataset loading."""
    
    @patch('src.data.load_dataset.load_dataset')
    def test_load_dataset_from_config(self, mock_load_dataset, sample_config_dir):
        """Test loading dataset from config."""
        from src.data.load_dataset import load_dataset_from_config
        
        # Mock the dataset
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"instruction": ["test"], "response": ["test"]}),
            "validation": Dataset.from_dict({"instruction": ["test"], "response": ["test"]})
        })
        mock_load_dataset.return_value = mock_dataset
        
        config_path = sample_config_dir / "dataset_config.yaml"
        dataset = load_dataset_from_config(config_path=config_path)
        
        assert isinstance(dataset, DatasetDict)
        assert "train" in dataset
        assert "validation" in dataset
        mock_load_dataset.assert_called_once()


class TestModelLoading:
    """Integration tests for model loading (mocked)."""
    
    @patch('src.model.load_model.FastLanguageModel')
    def test_load_model_from_config(self, mock_fast_model, sample_config_dir):
        """Test loading model from config."""
        from src.model.load_model import load_model_from_config
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
        
        config_path = sample_config_dir / "model_config.yaml"
        model, tokenizer = load_model_from_config(config_path=config_path)
        
        assert model is not None
        assert tokenizer is not None
        mock_fast_model.from_pretrained.assert_called_once()


class TestTrainingPipeline:
    """Integration tests for training pipeline."""
    
    @patch('src.training.trainer.Trainer')
    def test_create_trainer(self, mock_trainer_class, mock_model, mock_tokenizer, mock_dataset, sample_config_dir):
        """Test trainer creation."""
        from src.training.trainer import create_trainer
        
        # Mock trainer instance
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        config_path = sample_config_dir / "training_config.yaml"
        train_data = mock_dataset["train"]
        eval_data = mock_dataset["validation"]
        
        trainer = create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data,
            config_path=config_path
        )
        
        assert trainer is not None
        mock_trainer_class.assert_called_once()
