"""
Tests for error handling utilities.
"""

import pytest
from src.utils.error_handling import (
    CrisisAgentError,
    DatasetError,
    ModelError,
    TrainingError,
    EvaluationError,
    CUDAError,
    handle_errors,
    check_cuda_available,
    validate_path
)
from pathlib import Path


class TestCustomExceptions:
    """Tests for custom exception classes."""
    
    def test_crisis_agent_error(self):
        """Test base exception."""
        with pytest.raises(CrisisAgentError):
            raise CrisisAgentError("Test error")
    
    def test_dataset_error(self):
        """Test dataset exception."""
        with pytest.raises(DatasetError):
            raise DatasetError("Dataset error")
    
    def test_model_error(self):
        """Test model exception."""
        with pytest.raises(ModelError):
            raise ModelError("Model error")
    
    def test_training_error(self):
        """Test training exception."""
        with pytest.raises(TrainingError):
            raise TrainingError("Training error")
    
    def test_evaluation_error(self):
        """Test evaluation exception."""
        with pytest.raises(EvaluationError):
            raise EvaluationError("Evaluation error")
    
    def test_cuda_error(self):
        """Test CUDA exception."""
        with pytest.raises(CUDAError):
            raise CUDAError("CUDA error")


class TestHandleErrors:
    """Tests for handle_errors decorator."""
    
    def test_successful_execution(self):
        """Test decorator with successful function."""
        @handle_errors(error_type=DatasetError)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_error_handling(self):
        """Test decorator catches and re-raises errors."""
        @handle_errors(error_type=DatasetError)
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(DatasetError):
            test_func()
    
    def test_error_without_reraising(self):
        """Test decorator returns default value on error."""
        @handle_errors(error_type=DatasetError, reraise=False, default_return=None)
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        assert result is None


class TestValidatePath:
    """Tests for validate_path function."""
    
    def test_existing_path(self, temp_dir):
        """Test validation of existing path."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        result = validate_path(test_file, must_exist=True)
        assert result == test_file
    
    def test_nonexistent_path_must_exist(self, temp_dir):
        """Test validation fails for non-existent path when must_exist=True."""
        test_file = temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            validate_path(test_file, must_exist=True)
    
    def test_create_if_missing(self, temp_dir):
        """Test path creation when missing."""
        test_dir = temp_dir / "new_dir"
        
        result = validate_path(test_dir, must_exist=False, create_if_missing=True)
        assert result.exists()
        assert result.is_dir()
    
    def test_pathlib_path(self, temp_dir):
        """Test that function works with Path objects."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        result = validate_path(Path(test_file), must_exist=True)
        assert isinstance(result, Path)


class TestCheckCudaAvailable:
    """Tests for check_cuda_available function."""
    
    def test_cuda_check(self):
        """Test CUDA availability check."""
        # This will return True or False depending on system
        result = check_cuda_available()
        assert isinstance(result, bool)
