"""
Tests for JSON validation utilities.
"""

import pytest
from src.utils.json_validator import (
    validate_json_structure,
    validate_crisis_response,
    extract_json_from_text
)


class TestValidateJsonStructure:
    """Tests for validate_json_structure function."""
    
    def test_valid_json(self):
        """Test validation of valid JSON."""
        text = '{"action": "test", "priority": "high"}'
        is_valid, parsed, error = validate_json_structure(text)
        
        assert is_valid is True
        assert parsed == {"action": "test", "priority": "high"}
        assert error is None
    
    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        text = '{"action": "test", "priority":}'
        is_valid, parsed, error = validate_json_structure(text, strict=False)
        
        assert is_valid is False
        assert parsed is None
        assert error is not None
    
    def test_json_with_required_keys(self):
        """Test validation with required keys."""
        text = '{"action": "test", "priority": "high"}'
        is_valid, parsed, error = validate_json_structure(
            text,
            required_keys=["action", "priority"]
        )
        
        assert is_valid is True
        assert parsed is not None
    
    def test_json_missing_required_keys(self):
        """Test validation with missing required keys."""
        text = '{"action": "test"}'
        is_valid, parsed, error = validate_json_structure(
            text,
            required_keys=["action", "priority"]
        )
        
        assert is_valid is False
        assert "priority" in error
    
    def test_json_extraction_from_text(self):
        """Test extracting JSON from text with extra content."""
        text = 'Some text before {"action": "test"} and after'
        is_valid, parsed, error = validate_json_structure(text, strict=False)
        
        assert is_valid is True
        assert parsed == {"action": "test"}
    
    def test_non_dict_json(self):
        """Test that non-dict JSON is rejected."""
        text = '["array", "not", "dict"]'
        is_valid, parsed, error = validate_json_structure(text)
        
        assert is_valid is False
        assert "not a dictionary" in error


class TestValidateCrisisResponse:
    """Tests for validate_crisis_response function."""
    
    def test_valid_crisis_response(self):
        """Test validation of valid crisis response."""
        response = {
            "action": "evacuate",
            "priority": "high",
            "reasoning": "Fire detected",
            "resources": ["fire_department"]
        }
        is_valid, warnings = validate_crisis_response(response)
        
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_missing_common_fields(self):
        """Test validation with missing common fields."""
        response = {"action": "test"}
        is_valid, warnings = validate_crisis_response(response)
        
        # Should still be valid but with warnings
        assert len(warnings) > 0
        assert any("priority" in w for w in warnings)
    
    def test_expected_structure_validation(self):
        """Test validation with expected structure (common fields + expected types)."""
        response = {
            "action": "test",
            "priority": "high",
            "reasoning": "test reasoning",
            "resources": [],
        }
        expected_structure = {
            "action": str,
            "priority": str,
        }
        is_valid, warnings = validate_crisis_response(response, expected_structure)
        
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_wrong_type_in_structure(self):
        """Test validation with wrong type."""
        response = {
            "action": 123,  # Should be string
            "priority": "high"
        }
        expected_structure = {
            "action": str,
            "priority": str
        }
        is_valid, warnings = validate_crisis_response(response, expected_structure)
        
        assert len(warnings) > 0
        assert any("wrong type" in w.lower() for w in warnings)
    
    def test_non_dict_response(self):
        """Test validation of non-dict response."""
        response = "not a dict"
        is_valid, warnings = validate_crisis_response(response)
        
        assert is_valid is False
        assert len(warnings) > 0


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""
    
    def test_extract_valid_json(self):
        """Test extracting valid JSON from text."""
        text = 'Response: {"action": "test", "priority": "high"}'
        result = extract_json_from_text(text)
        
        assert result == {"action": "test", "priority": "high"}
    
    def test_extract_invalid_json(self):
        """Test extracting invalid JSON returns None."""
        text = 'Response: {invalid json}'
        result = extract_json_from_text(text)
        
        assert result is None
    
    def test_no_json_in_text(self):
        """Test text with no JSON returns None."""
        text = 'Just plain text, no JSON here'
        result = extract_json_from_text(text)
        
        assert result is None
