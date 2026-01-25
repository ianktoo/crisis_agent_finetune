"""
Centralized error handling utilities for the crisis-agent fine-tuning pipeline.
Provides decorators and context managers for graceful error handling.
"""

import functools
import traceback
from typing import Callable, Any, Optional
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CrisisAgentError(Exception):
    """Base exception for crisis-agent pipeline errors."""
    pass


class DatasetError(CrisisAgentError):
    """Raised when dataset loading or formatting fails."""
    pass


class ModelError(CrisisAgentError):
    """Raised when model loading or configuration fails."""
    pass


class TrainingError(CrisisAgentError):
    """Raised when training fails."""
    pass


class EvaluationError(CrisisAgentError):
    """Raised when evaluation fails."""
    pass


class CUDAError(CrisisAgentError):
    """Raised when CUDA-related errors occur."""
    pass


def handle_errors(
    error_type: type[Exception] = CrisisAgentError,
    log_error: bool = True,
    reraise: bool = True,
    default_return: Any = None
):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_type: Type of exception to catch
        log_error: Whether to log the error
        reraise: Whether to re-raise the exception
        default_return: Value to return if error occurs and reraise is False
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Unexpected error in {func.__name__}: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                if reraise:
                    raise error_type(f"Unexpected error: {str(e)}") from e
                return default_return
        return wrapper
    return decorator


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and log status.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available. Training will be slow on CPU.")
        return cuda_available
    except Exception as e:
        logger.error(f"Error checking CUDA: {str(e)}")
        return False


def handle_cuda_oom(func: Callable) -> Callable:
    """
    Decorator specifically for handling CUDA out-of-memory errors.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(
                    f"CUDA OOM error in {func.__name__}: {str(e)}\n"
                    "Consider reducing batch size, sequence length, or using gradient checkpointing."
                )
                raise CUDAError(f"CUDA out of memory: {str(e)}") from e
            raise
    return wrapper


def validate_path(path: Path, must_exist: bool = True, create_if_missing: bool = False) -> Path:
    """
    Validate a file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        create_if_missing: Whether to create the path if it doesn't exist (for directories)
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist and must_exist is True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if create_if_missing and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return path
