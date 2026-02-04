"""
Tests for logging utilities.
"""

import pytest
import logging
from pathlib import Path
from src.utils.logging import setup_logging, get_logger


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_logger_creation(self, temp_dir):
        """Test logger is created successfully."""
        log_dir = temp_dir / "logs"
        logger = setup_logging(log_dir=log_dir, log_level="INFO")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "crisis_agent"
    
    def test_log_dir_creation(self, temp_dir):
        """Test log directory is created if missing."""
        log_dir = temp_dir / "new_logs"
        setup_logging(log_dir=log_dir)
        
        assert log_dir.exists()
        assert log_dir.is_dir()
    
    def test_log_level(self, temp_dir):
        """Test different log levels."""
        log_dir = temp_dir / "logs"
        
        logger = setup_logging(log_dir=log_dir, log_level="DEBUG")
        assert logger.level == logging.DEBUG
        
        logger = setup_logging(log_dir=log_dir, log_level="ERROR")
        assert logger.level == logging.ERROR
    
    def test_console_logging(self, temp_dir):
        """Test console logging can be disabled."""
        log_dir = temp_dir / "logs"
        logger = setup_logging(log_dir=log_dir, log_to_console=False)
        
        # Check that no console (stdout/stderr) handler is present.
        # RotatingFileHandler is a StreamHandler subclass, so exclude FileHandler.
        console_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 0


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_default_logger_name(self):
        """Test default logger name."""
        logger = get_logger()
        
        assert logger.name == "crisis_agent"
